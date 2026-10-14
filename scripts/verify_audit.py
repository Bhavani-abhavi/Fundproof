"""Check the audit log's hash chain and head anchor, and optionally the chain of decision logs (Phase 5 B).

No key is needed: the audit log is plaintext (PII-redacted), and an encrypted decision log keeps its chain fields
in clear (security/encrypted_log.py).

What is detected in the audit log: an edited event, a deleted event, reordered events and a truncated tail (by the
head anchor). Someone who can rewrite both the log and its head file can rebuild a consistent chain, so keep a copy
of the printed head (count and hash) elsewhere and pass it back with --expect-count and --expect-head.

Decision logs (agent.decision_record.DecisionLog and security.encrypted_log.EncryptedDecisionLog) have no head
anchor: their chain detects edits, deletions and reordering, not a truncated tail.

Examples:
    uv run python scripts/verify_audit.py
    uv run python scripts/verify_audit.py --log data/processed/security/audit_events.jsonl \\
        --expect-count 42 --expect-head 3f2a...
    uv run python scripts/verify_audit.py --decision-log data/processed/agent/decision_records.jsonl

Exit code 0 when everything checked is intact, 1 otherwise.
"""

import argparse
import sys
from pathlib import Path

from uwcopilot.agent.decision_record import DecisionLog
from uwcopilot.security.audit import AuditHead, AuditLog
from uwcopilot.security.settings import load_security_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log", type=Path, help="audit log (default UWC_AUDIT_LOG or data/processed/security/...)")
    parser.add_argument("--expect-count", type=int, help="the event count from a head anchor kept elsewhere")
    parser.add_argument("--expect-head", help="the last record_hash from a head anchor kept elsewhere")
    parser.add_argument(
        "--decision-log", type=Path, action="append", default=[], help="also check this decision log's chain"
    )
    parser.add_argument("--skip-audit-log", action="store_true", help="check only the --decision-log files")
    args = parser.parse_args(argv)
    if (args.expect_count is None) != (args.expect_head is None):
        parser.error("--expect-count and --expect-head go together")
    return args


def check_audit_log(args: argparse.Namespace) -> list[str]:
    """Print and return the audit log's problems."""
    path = args.log or load_security_settings().audit_log
    log = AuditLog(path)
    expected = None
    if args.expect_count is not None:
        expected = AuditHead(count=args.expect_count, head_hash=args.expect_head)
    problems = log.verify(expected)
    if problems:
        print(f"Audit log {path}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return problems
    head = expected or log.head()
    count = head.count if head else 0
    print(f"Audit log {path}: intact ({count} events)")
    if head is not None:
        print(f"  head anchor: count={head.count} head_hash={head.head_hash}")
        print("  Keep a copy of this anchor somewhere the log's writer cannot change.")
    return problems


def check_decision_log(path: Path) -> list[str]:
    """Print and return one decision log's chain problems (no key needed)."""
    if not path.is_file():
        problem = f"no decision log at {path}"
        print(f"Decision log {path}: {problem}")
        return [problem]
    log = DecisionLog(path)
    problems = log.verify_chain()
    if problems:
        print(f"Decision log {path}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print(f"Decision log {path}: chain intact ({len(log.records())} records; a truncated tail is not detectable)")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify; 0 when intact, 1 when any problem was found."""
    args = parse_args(argv)
    problems: list[str] = []
    if not args.skip_audit_log:
        problems.extend(check_audit_log(args))
    for path in args.decision_log:
        problems.extend(check_decision_log(path))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
