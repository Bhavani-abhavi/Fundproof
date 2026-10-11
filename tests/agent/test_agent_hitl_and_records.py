"""Tests for the human-decision rules at the checkpoint and the append-only decision log."""

import json
from pathlib import Path

import pytest
from agent_helpers import credit_app, insurance_app

from uwcopilot.agent.applications import screened_parties
from uwcopilot.agent.decision_record import GENESIS_HASH, DecisionLog
from uwcopilot.agent.hitl import allowed_actions, is_override, validate_decision
from uwcopilot.agent.policy import InjectionFlag, PolicyInputs, ScreeningOutcome, evaluate_policy


def policy_for(application, injection: bool = False) -> object:
    """Policy result with clear screening (and an optional injection flag)."""
    screening = [ScreeningOutcome(party_id=party.party_id, status="clear") for party in screened_parties(application)]
    flags = [InjectionFlag(field="notes", patterns=["approval_claim"])] if injection else []
    return evaluate_policy(application, PolicyInputs(screening=screening, injection_flags=flags))


def decision(action: str, role: str = "underwriter", **extra: object) -> dict:
    """A decision dict."""
    return {"action": action, "reviewer_id": "rev.1", "role": role, "rationale": "Reviewed the full file.", **extra}


def test_valid_approval_within_authority() -> None:
    application = credit_app()
    accepted, errors = validate_decision(decision("approve"), policy_for(application), application)
    assert errors == [] and accepted is not None and accepted.action == "approve"


def test_schema_errors_are_refused() -> None:
    application = credit_app()
    policy = policy_for(application)
    for raw in [
        {"action": "approve"},
        decision("approve", rationale="ok"),
        decision("auto_approve"),
        {**decision("approve"), "bypass": True},
        "approve",
    ]:
        accepted, errors = validate_decision(raw, policy, application)
        assert accepted is None and errors


def test_only_a_senior_underwriter_may_decline() -> None:
    application = credit_app(industry="payday_lending")
    policy = policy_for(application)
    assert policy.recommendation == "decline_for_review"
    _, errors = validate_decision(decision("decline"), policy, application)
    assert any("senior_underwriter" in error for error in errors)
    _, errors = validate_decision(decision("refer"), policy, application)
    assert any("decline-for-review" in error for error in errors)
    accepted, errors = validate_decision(decision("decline", role="senior_underwriter"), policy, application)
    assert accepted is not None and errors == []


def test_nothing_is_approved_or_declined_while_a_blocking_referral_is_open() -> None:
    application = credit_app()
    policy = policy_for(application, injection=True)
    assert policy.blocking_open
    for action in ("approve", "decline"):
        _, errors = validate_decision(
            decision(action, role="senior_underwriter", override_reason="referral_resolved_in_writing"),
            policy,
            application,
        )
        assert any("referral" in error for error in errors)
    assert allowed_actions(policy) == ["refer", "request_more_information"]
    accepted, _ = validate_decision(decision("refer"), policy, application)
    assert accepted is not None


def test_approval_authority_and_overrides() -> None:
    large = credit_app(amount=300_000, term_months=84, verified_revenue=1_500_000)
    policy = policy_for(large)
    assert policy.recommendation == "approve"
    _, errors = validate_decision(decision("approve"), policy, large)
    assert any("authority" in error for error in errors)
    accepted, _ = validate_decision(decision("approve", role="senior_underwriter"), policy, large)
    assert accepted is not None
    from uwcopilot.agent.synthetic_insurance import LocationSpec

    big_tiv = insurance_app(locations=(LocationSpec(construction="non_combustible", square_feet=18_000, stories=1),))
    tiv_policy = policy_for(big_tiv)
    assert tiv_policy.recommendation == "approve"
    _, errors = validate_decision(decision("approve"), tiv_policy, big_tiv)
    assert any("authority" in error for error in errors)
    refer_case = credit_app(industry="freight_broker")
    refer_policy = policy_for(refer_case)
    _, errors = validate_decision(decision("approve"), refer_policy, refer_case)
    assert any("senior_underwriter" in error for error in errors)
    _, errors = validate_decision(decision("approve", role="senior_underwriter"), refer_policy, refer_case)
    assert any("override_reason" in error for error in errors)
    accepted, _ = validate_decision(
        decision("approve", role="senior_underwriter", override_reason="documented_compensating_factor"),
        refer_policy,
        refer_case,
    )
    assert accepted is not None
    assert is_override("approve", "refer") and not is_override("refer", "approve")
    _, errors = validate_decision(decision("approve", override_reason="because"), policy, large)
    assert errors


def test_incomplete_files_cannot_be_approved() -> None:
    application = credit_app(missing_documents=("debt_schedule",))
    policy = policy_for(application)
    _, errors = validate_decision(
        decision("approve", role="senior_underwriter", override_reason="policy_exception_approved"), policy, application
    )
    assert any("missing" in error for error in errors)


def test_decision_log_is_chained_idempotent_and_detects_edits(tmp_path: Path) -> None:
    log = DecisionLog(tmp_path / "log" / "records.jsonl")
    first = log.append({"record_id": "DR-1", "value": 1})
    second = log.append({"record_id": "DR-2", "value": 2})
    assert first["previous_record_sha256"] == GENESIS_HASH
    assert second["previous_record_sha256"] == first["record_sha256"]
    assert log.append({"record_id": "DR-1", "value": 99}) == first  # idempotent: not written twice
    assert len(log.records()) == 2
    assert log.verify_chain() == []
    lines = log.path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["value"] = 5
    log.path.write_text(json.dumps(tampered) + "\n" + lines[1] + "\n", encoding="utf-8")
    assert log.verify_chain()
    log.path.write_text(lines[1] + "\n", encoding="utf-8")  # first record deleted
    assert any("previous_record_sha256" in problem for problem in log.verify_chain())
    with pytest.raises(ValueError):
        log.append({"value": 3})
