"""Tests for audit events (security/audit.py) and scripts/verify_audit.py.

The chain must detect an edited, a deleted, a reordered and a truncated log: one test each, plus the honest limit
(rewriting the log AND its head file is only caught by an anchor kept elsewhere).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from agent.agent_helpers import committed_by_scenario, make_deps, memory_agent

import verify_audit
from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.agent.runner import resume_run, start_run
from uwcopilot.observability.redaction import redactor_for_application
from uwcopilot.security.audit import (
    GENESIS_HASH,
    AuditHead,
    AuditLog,
    AuditLogError,
    audit_decision_record,
    event_hash,
    record_access,
)
from uwcopilot.security.rbac import Decision, Principal, Resource

FIXED = datetime(2026, 10, 14, 9, 0, tzinfo=UTC)
UNDERWRITER = Principal("uw.one", "underwriter")
FILE = Resource("file", "APP-C-901")


def fixed_clock() -> datetime:
    """A fixed time, so events are reproducible."""
    return FIXED


def filled_log(tmp_path: Path, events: int = 4) -> AuditLog:
    """A log with a few view events."""
    log = AuditLog(tmp_path / "audit_events.jsonl", clock=fixed_clock)
    for index in range(events):
        log.append("view", UNDERWRITER, "view_file", Resource("file", f"APP-C-90{index}"), "allowed", "shown")
    return log


def lines_of(log: AuditLog) -> list[str]:
    """The raw lines of a log."""
    return log.path.read_text(encoding="utf-8").splitlines()


def write_lines(log: AuditLog, lines: list[str]) -> None:
    """Overwrite a log with these lines."""
    log.path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------------------------


def test_events_are_chained_numbered_and_anchored(tmp_path: Path) -> None:
    log = filled_log(tmp_path, 3)
    events = log.events()
    assert [event["seq"] for event in events] == [0, 1, 2]
    assert [event["event_id"] for event in events] == ["AE-00000000", "AE-00000001", "AE-00000002"]
    assert events[0]["prev_hash"] == GENESIS_HASH
    assert events[1]["prev_hash"] == events[0]["record_hash"] == event_hash(events[0])
    assert events[0]["actor"] == {"principal_id": "uw.one", "role": "underwriter", "role_source": "claimed"}
    assert events[0]["data_label"] == "SYNTHETIC" and events[0]["occurred_at"] == FIXED.isoformat()
    assert log.head() == AuditHead(count=3, head_hash=events[-1]["record_hash"], updated_at=FIXED.isoformat())
    assert log.verify() == []


def test_reason_and_details_are_pii_redacted(tmp_path: Path) -> None:
    application = committed_by_scenario("clean_term_dental")
    redactor = redactor_for_application(application.model_dump(mode="json"))
    log = AuditLog(tmp_path / "audit.jsonl", clock=fixed_clock, redactor=redactor)
    name = application.applicant.legal_name
    event = log.append(
        "view",
        UNDERWRITER,
        "view_file",
        FILE,
        "allowed",
        f"shown to reviewer; applicant {name}, FEIN 12-3456789, contact jane@example.com",
        {"rationale": "Owner said SSN 123-45-6789", "note": "acct 990012345678", "count": 3},
    )
    text = log.path.read_text(encoding="utf-8")
    for secret in (name, "12-3456789", "jane@example.com", "123-45-6789", "990012345678"):
        assert secret not in text
    assert "[NAME]" in event["reason"] and "[FEIN]" in event["reason"] and "[EMAIL]" in event["reason"]
    assert event["details"] == {"rationale": "[REDACTED]", "note": "[ACCOUNT]", "count": 3}


def test_unknown_event_types_and_outcomes_are_refused(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError, match="unknown event type"):
        log.append("auto_approve", UNDERWRITER, "x")
    with pytest.raises(ValueError, match="unknown outcome"):
        log.append("view", UNDERWRITER, "view_file", outcome="maybe")


def test_append_refuses_a_log_with_a_cut_off_last_line(tmp_path: Path) -> None:
    log = filled_log(tmp_path, 2)
    log.path.write_text(log.path.read_text(encoding="utf-8")[:-10], encoding="utf-8")
    with pytest.raises(AuditLogError, match="cut off"):
        log.append("view", UNDERWRITER, "view_file", FILE, "allowed")


def test_record_access_logs_views_and_denials(tmp_path: Path) -> None:
    log = AuditLog(tmp_path / "audit.jsonl", clock=fixed_clock)
    allowed = record_access(log, UNDERWRITER, "view_file", FILE, Decision(True, "underwriter may view_file"))
    denied = record_access(
        log, Principal("adm.1", "admin"), "view_file", FILE, Decision(False, "admin may not view_file")
    )
    assert (allowed["event_type"], allowed["outcome"]) == ("view", "allowed")
    assert (denied["event_type"], denied["outcome"]) == ("access_denied", "denied")
    with pytest.raises(ValueError, match="decision record"):
        record_access(log, UNDERWRITER, "approve_file", FILE, Decision(True, "ok"))


# ---------------------------------------------------------------------------------------------------------------
# Tampering: edit, delete, reorder, truncate
# ---------------------------------------------------------------------------------------------------------------


def test_an_edited_event_is_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    lines = lines_of(log)
    event = json.loads(lines[1])
    event["actor"]["role"] = "senior_underwriter"
    lines[1] = json.dumps(event)
    write_lines(log, lines)
    problems = log.verify()
    assert any(problem.startswith("line 2: record_hash does not match") for problem in problems)


def test_a_deleted_event_is_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    lines = lines_of(log)
    write_lines(log, lines[:1] + lines[2:])
    problems = log.verify()
    assert any("line 2: seq is 2, expected 1" in problem for problem in problems)
    assert any("line 2: prev_hash does not match" in problem for problem in problems)


def test_deleting_the_first_event_is_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    write_lines(log, lines_of(log)[1:])
    assert any("line 1: prev_hash does not match" in problem for problem in log.verify())


def test_reordered_events_are_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    lines = lines_of(log)
    lines[1], lines[2] = lines[2], lines[1]
    write_lines(log, lines)
    problems = log.verify()
    assert any("line 2: seq is 2, expected 1" in problem for problem in problems)
    assert any("prev_hash does not match" in problem for problem in problems)


def test_a_truncated_log_is_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    write_lines(log, lines_of(log)[:-1])  # the chain of the remaining events is still intact ...
    problems = log.verify()
    assert problems == [
        "the log holds 3 events but the head anchor says 4 (truncated?)",
        "the last event's record_hash does not match the head anchor (truncated or rewritten)",
    ]  # ... so only the head anchor catches it


def test_a_cut_off_last_line_is_detected(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    log.path.write_text(log.path.read_text(encoding="utf-8")[:-25], encoding="utf-8")
    problems = log.verify()
    assert any("the last line is cut off" in problem for problem in problems)
    assert any("line 4: not valid JSON" in problem for problem in problems)


def test_a_rechained_forgery_is_caught_by_the_head_anchor(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    kept = [json.loads(line) for line in lines_of(log)][:2] + [json.loads(lines_of(log)[3])]
    previous = GENESIS_HASH
    for seq, event in enumerate(kept):  # delete event 2 and rebuild every hash
        event["seq"], event["event_id"], event["prev_hash"] = seq, f"AE-{seq:08d}", previous
        event["record_hash"] = event_hash(event)
        previous = event["record_hash"]
    write_lines(log, [json.dumps(event) for event in kept])
    assert log.verify() != []


def test_rewriting_the_head_too_needs_an_anchor_kept_elsewhere(tmp_path: Path) -> None:
    """The honest limit: a writer who controls both files can hide a truncation unless the head is kept elsewhere."""
    log = filled_log(tmp_path)
    external_anchor = log.head()
    lines = lines_of(log)[:-1]
    write_lines(log, lines)
    last = json.loads(lines[-1])
    log.head_path.write_text(AuditHead(count=3, head_hash=last["record_hash"]).model_dump_json(), encoding="utf-8")
    assert log.verify() == []  # the stored head was rewritten as well
    assert log.verify(expected_head=external_anchor) != []  # the copy kept elsewhere still catches it


def test_a_missing_head_is_reported(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    log.head_path.unlink()
    assert log.verify() == ["no head anchor: a truncated tail cannot be ruled out"]


# ---------------------------------------------------------------------------------------------------------------
# Events from an agent decision record
# ---------------------------------------------------------------------------------------------------------------


def test_audit_events_from_a_real_agent_run(tmp_path: Path) -> None:
    application = committed_by_scenario("clean_term_dental")
    agent = memory_agent(make_deps(tmp_path))
    start_run(agent, application.model_dump(mode="json"), "t-audit")
    resume_run(agent, "t-audit", {"action": "approve", "reviewer_id": "adm.1", "role": "admin", "rationale": "x" * 12})
    resume_run(
        agent,
        "t-audit",
        {
            "action": "decline",
            "reviewer_id": "senior.one",
            "role": "senior_underwriter",
            "rationale": "Declining after reading the file; owner jane@example.com.",
            "override_reason": "documented_compensating_factor",
        },
    )
    [stored] = DecisionLog(tmp_path / "decision_records.jsonl").records()
    log = AuditLog(tmp_path / "audit.jsonl", clock=fixed_clock)
    written = audit_decision_record(log, stored, redactor_for_application(application.model_dump(mode="json")))
    types = [event["event_type"] for event in written]
    assert types == ["tool_call"] * len(stored["tool_calls"]) + [
        "recommendation",
        "decision_refused",
        "decline",
        "override",
    ]
    by_type = {event["event_type"]: event for event in written}
    assert by_type["recommendation"]["actor"]["role"] == "system"
    assert by_type["recommendation"]["details"]["reference_recommendation"] == "approve"
    assert by_type["decision_refused"]["actor"] == {
        "principal_id": "adm.1",
        "role": "admin",
        "role_source": "claimed",
    }
    assert "role refused" in by_type["decision_refused"]["reason"]
    assert by_type["decline"]["details"]["rationale"] == "[REDACTED]"
    assert by_type["override"]["details"]["override_reason"] == "documented_compensating_factor"
    assert "jane@example.com" not in log.path.read_text(encoding="utf-8")
    assert audit_decision_record(log, stored) == []  # idempotent: a replay adds nothing
    assert log.verify() == []


def test_unrecognised_reviewer_claims_are_logged_as_unknown(tmp_path: Path) -> None:
    record = {
        "record_id": "DR-x",
        "application_id": "APP-C-901",
        "tool_calls": [],
        "reference_recommendation": "refer",
        "rejected_reviewer_attempts": [
            {"submitted": {"action": "approve", "reviewer_id": "bad id!", "role": "root"}, "errors": ["role: ..."]}
        ],
        "reviewer": {"action": "refer", "reviewer_id": "uw.one", "role": "underwriter", "is_override": False},
        "final_status": "referred_by_human",
    }
    log = AuditLog(tmp_path / "audit.jsonl", clock=fixed_clock)
    written = audit_decision_record(log, record)
    assert [event["event_type"] for event in written] == ["recommendation", "decision_refused", "referral"]
    assert written[1]["actor"] == {"principal_id": "unidentified", "role": "unknown", "role_source": "claimed"}


# ---------------------------------------------------------------------------------------------------------------
# scripts/verify_audit.py
# ---------------------------------------------------------------------------------------------------------------


def test_verify_script_passes_an_intact_log(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    log = filled_log(tmp_path)
    assert verify_audit.main(["--log", str(log.path)]) == 0
    output = capsys.readouterr().out
    assert "intact (4 events)" in output and f"head_hash={log.head().head_hash}" in output


def test_verify_script_fails_a_tampered_log(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    log = filled_log(tmp_path)
    write_lines(log, lines_of(log)[:-1])
    assert verify_audit.main(["--log", str(log.path)]) == 1
    assert "head anchor says 4" in capsys.readouterr().out


def test_verify_script_uses_an_external_anchor(tmp_path: Path) -> None:
    log = filled_log(tmp_path)
    head = log.head()
    good = ["--log", str(log.path), "--expect-count", str(head.count), "--expect-head", head.head_hash]
    assert verify_audit.main(good) == 0
    wrong = ["--log", str(log.path), "--expect-count", "5", "--expect-head", head.head_hash]
    assert verify_audit.main(wrong) == 1
    with pytest.raises(SystemExit):
        verify_audit.main(["--log", str(log.path), "--expect-count", "4"])


def test_verify_script_checks_decision_logs_and_missing_files(tmp_path: Path) -> None:
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append({"record_id": "DR-1", "data_label": "SYNTHETIC"})
    assert verify_audit.main(["--skip-audit-log", "--decision-log", str(decisions.path)]) == 0
    decisions.path.write_text(decisions.path.read_text(encoding="utf-8").replace("DR-1", "DR-2"), encoding="utf-8")
    assert verify_audit.main(["--skip-audit-log", "--decision-log", str(decisions.path)]) == 1
    assert verify_audit.main(["--log", str(tmp_path / "missing.jsonl")]) == 1
