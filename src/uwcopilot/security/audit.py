"""Audit events: an append-only, hash-chained JSON Lines log of every view, tool call, recommendation and decision.

    log = AuditLog(Path("data/processed/security/audit_events.jsonl"))
    log.append("view", principal, "view_file", Resource("file", "APP-C-001"), outcome="allowed", reason="...")
    record_access(log, principal, "view_documents", Resource("packet", "PKT-C-024"), decision)
    audit_decision_record(log, stored_record)     # tool calls, recommendation, refused attempts, decision, override
    log.verify()                                   # [] when intact; scripts/verify_audit.py prints the same

One event per line:
    {"schema_version": 1, "seq": 0, "event_id": "AE-00000000", "occurred_at": "...", "event_type": "view",
     "actor": {"principal_id": "uw.1", "role": "underwriter", "role_source": "claimed"},
     "action": "view_file", "resource": {"kind": "file", "id": "APP-C-001"}, "outcome": "allowed",
     "reason": "...", "details": {...}, "data_label": "SYNTHETIC", "prev_hash": "000...", "record_hash": "..."}

Integrity. record_hash is the SHA-256 of the event's canonical JSON without record_hash; prev_hash is the record_hash
of the event before it (64 zeros for the first); seq counts from 0. After every append the log's "head anchor"
(<name>.head.json: the event count and the last record_hash) is rewritten. verify() detects:
- an EDITED event (its record_hash no longer matches its content, or the line is no longer valid JSON);
- a DELETED event in the middle or at the start (seq gap, prev_hash mismatch);
- REORDERED events (seq and prev_hash mismatch);
- a TRUNCATED tail (fewer events, or another last hash, than the head anchor says; or a cut-off last line).
Limit: someone who can rewrite BOTH the log and its head file can rebuild a consistent chain. Copy the head
(count + hash, printed by scripts/verify_audit.py) somewhere they cannot write, and pass it back with
--expect-count / --expect-head. A signed or write-once anchor (Cloud KMS, object lock) is Phase 7, not built.

Privacy. Every reason and every details value goes through observability.redaction (a PiiRedactor: names you give
it, FEINs, SSNs, e-mails, phones, account numbers, and whole values under keys such as rationale, notes, excerpt).
The actor id and the resource id are kept: an audit trail must say who did what to which file. Use opaque ids.

Roles are CLAIMED in this demo: actor.role_source says so on every event ("claimed" unless the caller knows better).
"""

import fcntl
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from uwcopilot.agent.applications import canonical_json
from uwcopilot.agent.vocab import ACCESS_ROLES, SYNTHETIC_LABEL
from uwcopilot.observability.redaction import PiiRedactor
from uwcopilot.security.config import PRINCIPAL_ID_PATTERN
from uwcopilot.security.fileio import write_private_file
from uwcopilot.security.rbac import AGENT_PRINCIPAL, UNKNOWN_ROLE, Decision, Principal, Resource

GENESIS_HASH = "0" * 64
AUDIT_SCHEMA_VERSION = 1
MAX_SHORT_TEXT = 200

EVENT_TYPES: tuple[str, ...] = (
    "view",  # a file, a packet's documents or the audit log was shown
    "access_denied",  # authorize() said no
    "run_started",  # an agent run was started for a file
    "tool_call",  # the agent called a tool
    "recommendation",  # the reference policy's (and memo's) recommendation reached the human checkpoint
    "approval",  # a human decision accepted by the checkpoint ...
    "decline",
    "referral",
    "information_request",
    "override",  # ... that differs from the recommendation (logged in addition to the decision)
    "decision_refused",  # the checkpoint refused a submitted decision
    "permission_change",  # admin and validator actions (authorized by rbac; no code path performs them yet)
    "model_promotion",
    "migration_certificate",
)
OUTCOMES: tuple[str, ...] = ("allowed", "denied", "ok", "error", "refused")
# The event logged by record_access() for an action that authorize() allowed.
ACCESS_EVENT_TYPES: dict[str, str] = {
    "view_file": "view",
    "view_documents": "view",
    "view_audit_log": "view",
    "run_agent": "run_started",
    "manage_permissions": "permission_change",
    "promote_model": "model_promotion",
    "sign_migration_certificate": "migration_certificate",
}
# The event for each reviewer action the checkpoint accepted.
DECISION_EVENT_TYPES: dict[str, str] = {
    "approve": "approval",
    "decline": "decline",
    "refer": "referral",
    "request_more_information": "information_request",
}
REQUIRED_FIELDS: tuple[str, ...] = (
    "schema_version",
    "seq",
    "event_id",
    "occurred_at",
    "event_type",
    "actor",
    "action",
    "outcome",
    "prev_hash",
    "record_hash",
)


class AuditLogError(RuntimeError):
    """The audit log cannot be appended to safely (for example its last line is cut off)."""


class AuditHead(BaseModel):
    """The head anchor: how many events the log should hold and the record_hash of the last one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    head_hash: str
    updated_at: str | None = None


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def event_hash(event: Mapping[str, Any]) -> str:
    """SHA-256 of the event's canonical JSON, without its own record_hash field."""
    body = {key: value for key, value in event.items() if key != "record_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def head_path_for(log_path: Path) -> Path:
    """Where the head anchor of a log lives: audit_events.jsonl -> audit_events.head.json."""
    return log_path.with_name(log_path.stem + ".head.json")


def read_head(path: Path) -> AuditHead | None:
    """The head anchor stored at path, or None when there is none."""
    if not path.is_file():
        return None
    return AuditHead.model_validate(json.loads(path.read_text(encoding="utf-8")))


def actor_of(principal: Principal) -> dict[str, str]:
    """The actor fields of an event."""
    return {"principal_id": principal.principal_id, "role": principal.role, "role_source": principal.role_source}


def resource_of(resource: Resource | None) -> dict[str, Any] | None:
    """The resource fields of an event (line and amount only when known)."""
    if resource is None:
        return None
    fields: dict[str, Any] = {"kind": resource.kind, "id": resource.resource_id}
    if resource.line is not None:
        fields["line"] = resource.line
    if resource.amount is not None:
        fields["amount"] = resource.amount
    return fields


class AuditLog:
    """One JSONL file of audit events plus its head anchor. Events are only ever appended."""

    def __init__(
        self, path: Path, clock: Callable[[], datetime] = utc_now, redactor: PiiRedactor | None = None
    ) -> None:
        """path is created (with its folder) on the first append; redactor defaults to the pattern-only redactor."""
        self.path = path
        self.head_path = head_path_for(path)
        self._clock = clock
        self._redactor = redactor if redactor is not None else PiiRedactor()

    def events(self) -> list[dict[str, Any]]:
        """Every event in file order (an empty list if the file does not exist); AuditLogError on a damaged line."""
        if not self.path.exists():
            return []
        events = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AuditLogError(f"line {number} of the audit log is not valid JSON") from error
        return events

    def head(self) -> AuditHead | None:
        """The stored head anchor, or None."""
        return read_head(self.head_path)

    def append(
        self,
        event_type: str,
        actor: Principal,
        action: str,
        resource: Resource | None = None,
        outcome: str = "ok",
        reason: str | None = None,
        details: Mapping[str, Any] | None = None,
        redactor: PiiRedactor | None = None,
        data_label: str = SYNTHETIC_LABEL,
    ) -> dict[str, Any]:
        """Chain and append one event (reason and details are PII-redacted first); return the stored event."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {event_type!r}; known types are {list(EVENT_TYPES)}")
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; known outcomes are {list(OUTCOMES)}")
        scrub = redactor if redactor is not None else self._redactor
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # One writer at a time (POSIX file lock): read the tail, chain, append, update the head, then unlock.
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                existing = handle.read()
                seq, previous = next_position(existing)
                event: dict[str, Any] = {
                    "schema_version": AUDIT_SCHEMA_VERSION,
                    "seq": seq,
                    "event_id": f"AE-{seq:08d}",
                    "occurred_at": self._clock().isoformat(),
                    "event_type": event_type,
                    "actor": actor_of(actor),
                    "action": action,
                    "resource": resource_of(resource),
                    "outcome": outcome,
                    "reason": scrub.redact_text(reason) if reason else None,
                    "details": scrub.redact(dict(details or {})),
                    "data_label": data_label,
                    "prev_hash": previous,
                }
                event["record_hash"] = event_hash(event)
                handle.write((canonical_json(event) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
                head = AuditHead(count=seq + 1, head_hash=event["record_hash"], updated_at=event["occurred_at"])
                write_private_file(self.head_path, head.model_dump_json(indent=2).encode("utf-8"))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return event

    def verify(self, expected_head: AuditHead | None = None) -> list[str]:
        """Problems found in the chain and against the head anchor; an empty list means the log is intact.

        expected_head (an anchor kept elsewhere) wins over the stored head file.
        """
        return verify_audit_file(self.path, expected_head if expected_head is not None else self.head())


def next_position(existing: bytes) -> tuple[int, str]:
    """(seq, prev_hash) for the next event, from the current file bytes; AuditLogError if the tail is damaged."""
    if not existing.strip():
        return 0, GENESIS_HASH
    if not existing.endswith(b"\n"):
        raise AuditLogError("the audit log's last line is cut off; run scripts/verify_audit.py before writing")
    last_line = existing.rstrip(b"\n").rsplit(b"\n", 1)[-1]
    try:
        last = json.loads(last_line)
        return int(last["seq"]) + 1, str(last["record_hash"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise AuditLogError("the audit log's last event is damaged; run scripts/verify_audit.py") from error


def verify_lines(lines: list[str]) -> list[str]:
    """Chain problems in a list of JSON lines (seq, prev_hash and record_hash of every event)."""
    problems = []
    previous = GENESIS_HASH
    for index, line in enumerate(lines):
        number = index + 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not valid JSON (edited or cut off)")
            previous = ""
            continue
        missing = [field for field in REQUIRED_FIELDS if not isinstance(event, dict) or field not in event]
        if missing:
            problems.append(f"line {number}: missing fields {missing} (edited)")
            previous = ""
            continue
        if event["seq"] != index:
            problems.append(f"line {number}: seq is {event['seq']}, expected {index} (an event was deleted or moved)")
        if event["prev_hash"] != previous:
            problems.append(f"line {number}: prev_hash does not match the event before it (deleted or reordered)")
        if event["record_hash"] != event_hash(event):
            problems.append(f"line {number}: record_hash does not match its content (edited)")
        previous = event["record_hash"]
    return problems


def verify_audit_file(path: Path, head: AuditHead | None) -> list[str]:
    """Chain problems plus head-anchor problems for one audit log file."""
    if not path.is_file():
        return [f"no audit log at {path}"]
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    problems = []
    if text and not text.endswith("\n"):
        problems.append(f"line {len(lines)}: the last line is cut off (truncated while or after writing)")
    problems.extend(verify_lines(lines))
    if head is None:
        if lines:
            problems.append("no head anchor: a truncated tail cannot be ruled out")
        return problems
    if head.count != len(lines):
        problems.append(f"the log holds {len(lines)} events but the head anchor says {head.count} (truncated?)")
    last_hash = last_record_hash(lines)
    if head.count and last_hash != head.head_hash:
        problems.append("the last event's record_hash does not match the head anchor (truncated or rewritten)")
    return problems


def last_record_hash(lines: list[str]) -> str | None:
    """The record_hash of the last line, or None when it cannot be read."""
    if not lines:
        return None
    try:
        value = json.loads(lines[-1]).get("record_hash")
    except (json.JSONDecodeError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def record_access(
    log: AuditLog,
    principal: Principal,
    action: str,
    resource: Resource | None,
    decision: Decision,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Log one authorization outcome: access_denied when refused, else the action's own event (view, ...).

    File decisions are not logged here: the human checkpoint may still refuse them, so they are logged from the
    decision record (audit_decision_record).
    """
    if action not in ACCESS_EVENT_TYPES:
        raise ValueError(f"{action!r} is not logged by record_access; file decisions come from the decision record")
    if not decision.allowed:
        return log.append("access_denied", principal, action, resource, "denied", decision.reason, details)
    return log.append(ACCESS_EVENT_TYPES[action], principal, action, resource, "allowed", decision.reason, details)


def claimed_principal(raw: Mapping[str, Any]) -> Principal:
    """The actor a submitted decision claims to be (role_source "claimed"); 'unknown' when it is not recognisable."""
    reviewer_id = raw.get("reviewer_id")
    role = raw.get("role")
    plain_id = isinstance(reviewer_id, str) and re.fullmatch(PRINCIPAL_ID_PATTERN, reviewer_id) is not None
    principal_id = reviewer_id if plain_id else "unidentified"
    known_role = role if isinstance(role, str) and role in ACCESS_ROLES else UNKNOWN_ROLE
    return Principal(principal_id, known_role, "claimed")


def already_audited(log: AuditLog, record_id: str) -> bool:
    """True when events for this decision record are already in the log (so a replay adds nothing)."""
    return any((event.get("details") or {}).get("decision_record_id") == record_id for event in log.events())


def audit_decision_record(
    log: AuditLog, record: Mapping[str, Any], redactor: PiiRedactor | None = None
) -> list[dict[str, Any]]:
    """Log the events a stored agent decision record holds; idempotent by record_id. Returns the new events.

    In order: one tool_call per tool call (actor: the agent), the recommendation (the agent), each refused decision
    attempt (the claimed reviewer), the accepted decision (the claimed reviewer), and an override when the decision
    differs from the recommendation. The record itself is not copied: events hold ids, statuses and redacted text.
    """
    record_id = str(record["record_id"])
    if already_audited(log, record_id):
        return []
    resource = Resource("file", str(record.get("application_id") or "unknown"))
    base = {"decision_record_id": record_id}
    written = []
    for call in record.get("tool_calls") or []:
        details = {**base, **{key: call.get(key) for key in ("result_id", "tool", "purpose", "status")}}
        outcome = "ok" if call.get("status") == "ok" else "error"
        written.append(
            log.append("tool_call", AGENT_PRINCIPAL, str(call.get("tool")), resource, outcome, None, details, redactor)
        )
    memo = record.get("memo") or {}
    recommendation = {
        **base,
        "reference_recommendation": record.get("reference_recommendation"),
        "memo_recommendation": memo.get("recommendation") if isinstance(memo, dict) else None,
        "memo_status": record.get("memo_status"),
        "fake_llm": record.get("fake_llm"),
    }
    written.append(log.append("recommendation", AGENT_PRINCIPAL, "recommend", resource, "ok", None, recommendation))
    for attempt in record.get("rejected_reviewer_attempts") or []:
        submitted = attempt.get("submitted") if isinstance(attempt.get("submitted"), dict) else {}
        details = {**base, "submitted_action": str(submitted.get("action"))[:MAX_SHORT_TEXT]}
        reason = "; ".join(str(error) for error in attempt.get("errors") or [])[: MAX_SHORT_TEXT * 5]
        actor = claimed_principal(submitted)
        written.append(
            log.append("decision_refused", actor, "decide_file", resource, "refused", reason, details, redactor)
        )
    reviewer = record.get("reviewer") or {}
    action = str(reviewer.get("action"))
    actor = claimed_principal(reviewer)
    decision_details = {
        **base,
        "final_status": record.get("final_status"),
        "reference_recommendation": record.get("reference_recommendation"),
        "is_override": bool(reviewer.get("is_override")),
        "rationale": reviewer.get("rationale"),
    }
    event_type = DECISION_EVENT_TYPES.get(action)
    if event_type is not None:
        written.append(log.append(event_type, actor, action, resource, "ok", None, decision_details, redactor))
    if reviewer.get("is_override"):
        override = {
            **base,
            "action": action,
            "override_reason": reviewer.get("override_reason"),
            "reference_recommendation": record.get("reference_recommendation"),
        }
        written.append(log.append("override", actor, action, resource, "ok", None, override, redactor))
    return written
