"""Tests for the reference policy: each implemented rule, precedence, and agreement with the designed labels."""

import json

import pytest
from agent_helpers import APPLICATIONS_DIR, credit_app, insurance_app

from uwcopilot.agent.applications import Claim, parse_application, screened_parties
from uwcopilot.agent.claim_checker import evidence_numbers, is_supported, numbers_in
from uwcopilot.agent.evaluation import ground_truth_screening
from uwcopilot.agent.evidence import build_evidence
from uwcopilot.agent.injection import screen_application
from uwcopilot.agent.policy import (
    Finding,
    InjectionFlag,
    PolicyInputs,
    RiskSignal,
    ScreeningOutcome,
    decide_recommendation,
    evaluate_policy,
)


def clear_screening(application) -> list[ScreeningOutcome]:
    """Every party screened clear."""
    return [
        ScreeningOutcome(party_id=party.party_id, status="clear", result_id=f"TR-{index}")
        for index, party in enumerate(screened_parties(application), start=1)
    ]


def run(application, **inputs) -> object:
    """The policy with clear screening unless other inputs are given."""
    inputs.setdefault("screening", clear_screening(application))
    return evaluate_policy(application, PolicyInputs(**inputs))


def finding(outcome: str, blocking: bool = False) -> Finding:
    """A minimal finding."""
    return Finding(
        rule="x",
        outcome=outcome,
        section_ids=["NBC-7.3"],
        refer_to=["Senior Underwriter"],
        blocking=blocking,
        detail="x",
    )


def test_precedence_blocking_then_decline_then_information_then_refer() -> None:
    assert decide_recommendation([]) == "approve"
    assert decide_recommendation([finding("refer")]) == "refer"
    assert decide_recommendation([finding("refer"), finding("request_more_information")]) == "request_more_information"
    assert decide_recommendation([finding("request_more_information"), finding("decline_for_review")]) == (
        "decline_for_review"
    )
    assert decide_recommendation([finding("decline_for_review"), finding("refer", blocking=True)]) == "refer"


def test_clean_credit_application_is_recommended_for_approval_with_cited_checks() -> None:
    result = run(credit_app())
    assert result.recommendation == "approve"
    assert result.findings == []
    assert result.required_reviewer_role == "underwriter"
    assert {check.rule for check in result.checks_passed} >= {"dscr", "verified_revenue_minimum", "documents_complete"}
    assert all(check.section_ids for check in result.checks_passed)
    assert any(item.rule == "risk_model_pd" for item in result.not_evaluated)


@pytest.mark.parametrize(
    ("changes", "recommendation", "rule"),
    [
        ({"dscr_target": 1.18, "amount": 80_000, "verified_revenue": 400_000}, "refer", "dscr_below_minimum"),
        (
            {"dscr_target": 1.02, "amount": 70_000, "verified_revenue": 380_000, "leverage_target": 1.5},
            "decline_for_review",
            "dscr_below_decline_floor",
        ),
        ({"industry": "payday_lending"}, "decline_for_review", "prohibited_industry"),
        ({"industry": "freight_broker"}, "refer", "restricted_industry"),
        ({"industry": "restaurant", "months_in_business": 30}, "refer", "restricted_industry"),
        ({"customer_pct": 35.0}, "refer", "customer_concentration"),
        ({"months_in_business": 12}, "decline_for_review", "time_in_business_short"),
        ({"credit_scores": (700, 600)}, "decline_for_review", "credit_score_below_minimum"),
        ({"revenue_gap_pct": 12.0}, "refer", "revenue_gap_explainable"),
        ({"verified_revenue": 240_000, "amount": 30_000}, "refer", "revenue_near_miss"),
        ({"verified_revenue": 200_000, "amount": 30_000}, "decline_for_review", "revenue_below_minimum"),
        ({"negative_months": 3}, "refer", "negative_balance_months"),
        ({"new_loans": 3}, "refer", "stacking"),
        ({"leverage_target": 4.6}, "refer", "leverage_above_maximum"),
        ({"structure": "sole_proprietorship"}, "decline_for_review", "ineligible_business_structure"),
        ({"missing_documents": ("debt_schedule",)}, "request_more_information", "missing_documents"),
        ({"ownership_total_pct": 80.0}, "refer", "ownership_discrepancy"),
    ],
)
def test_credit_rules(changes: dict, recommendation: str, rule: str) -> None:
    result = run(credit_app(**changes))
    assert rule in result.rule_codes()
    assert result.recommendation == recommendation


def test_rate_stress_applies_to_variable_rate_lines() -> None:
    application = credit_app(
        product="line_of_credit",
        amount=150_000,
        term_months=12,
        rate_pct=10.5,
        rate_type="variable",
        verified_revenue=1_100_000,
    )
    result = run(application)
    assert {"stress_test_revenue", "stress_test_rate"} <= {check.rule for check in result.checks_passed}
    shortfall = application.debt.ttm_pnl_revenue - 1_100_000
    weak_debt = application.debt.model_copy(update={"ttm_ebitda": 50_000.0 + max(0.0, shortfall)})
    stressed = run(application.model_copy(update={"debt": weak_debt}))
    assert {"stress_test_rate", "stress_test_revenue", "dscr_below_decline_floor"} <= set(stressed.rule_codes())


def test_low_average_balance_is_a_conduct_referral() -> None:
    application = credit_app()
    statements = application.bank_statements.model_copy(update={"average_daily_balance_last_3_months": 10.0})
    result = run(application.model_copy(update={"bank_statements": statements}))
    assert "low_average_balance" in result.rule_codes()


def test_missing_principal_score_requests_information() -> None:
    application = credit_app()
    owners = [application.owners[0].model_copy(update={"credit_score": None}), *application.owners[1:]]
    result = run(application.model_copy(update={"owners": owners}))
    assert "principal_credit_score_missing" in result.rule_codes()
    assert result.recommendation == "request_more_information"


def test_tax_lien_and_bankruptcy_cannot_be_approved() -> None:
    result = run(credit_app(bankruptcy=True).model_copy(update={"tax_lien_unresolved": True}))
    assert {"bankruptcy_within_7_years", "unresolved_tax_lien"} <= set(result.rule_codes())
    assert result.recommendation == "decline_for_review"
    assert result.required_reviewer_role == "senior_underwriter"


def test_sanctions_potential_match_is_a_blocking_compliance_referral() -> None:
    application = credit_app(industry="payday_lending")
    screening = clear_screening(application)
    screening[1] = screening[1].model_copy(update={"status": "potential_match", "top_score": 0.97})
    result = run(application, screening=screening)
    assert result.recommendation == "refer"  # not decline: the file cannot be closed while the match is open
    assert result.blocking_open
    assert result.review_queue == "Compliance"
    match = next(item for item in result.findings if item.rule == "sanctions_potential_match")
    assert match.trigger_ids == ["R-NBC-10"]


def test_missing_or_failed_screening_blocks_the_file() -> None:
    application = credit_app()
    result = run(application, screening=clear_screening(application)[:1])
    assert "screening_not_completed" in result.rule_codes()
    assert result.blocking_open


def test_injection_flag_is_a_blocking_fraud_or_siu_referral() -> None:
    flags = [InjectionFlag(field="notes", patterns=["approval_claim"])]
    credit = run(credit_app(), injection_flags=flags)
    insurance = run(insurance_app(), injection_flags=flags)
    assert credit.recommendation == insurance.recommendation == "refer"
    assert credit.review_queue == "Fraud Operations"
    assert insurance.review_queue == "Special Investigations Unit"


def test_model_pd_above_twelve_percent_is_referred() -> None:
    risk = RiskSignal(result_id="TR-9", product="credit", prediction=0.15, risk_band="E", model_version="m1")
    result = run(credit_app(), risk=risk)
    assert "model_pd_above_threshold" in result.rule_codes()
    low = run(credit_app(), risk=risk.model_copy(update={"prediction": 0.05}))
    assert "model_pd_above_threshold" not in low.rule_codes()


def test_clean_insurance_submission_is_recommended_for_approval() -> None:
    result = run(insurance_app())
    assert result.recommendation == "approve"
    assert result.findings == []


@pytest.mark.parametrize(
    ("changes", "recommendation", "rule"),
    [
        ({"class_code": "salvage_yard"}, "decline_for_review", "prohibited_class"),
        ({"class_code": "body_shop"}, "refer", "restricted_class"),
        ({"class_code": "restaurant_fryer"}, "refer", "restricted_class"),
        ({"years_in_business": 1, "prior_experience": 0}, "refer", "eligibility_experience"),
        ({"tiv_difference_pct": 30.0}, "refer", "tiv_difference_above_15"),
        ({"tiv_difference_pct": 8.0}, "request_more_information", "tiv_difference_correction"),
        ({"valuation_age_days": 120}, "request_more_information", "loss_runs_out_of_date"),
        ({"carrier_produced": False}, "request_more_information", "loss_runs_not_carrier_produced"),
        ({"stated_claim_count_delta": 2}, "refer", "loss_run_reconciliation"),
        ({"gl_limits": (2_000_000, 4_000_000)}, "refer", "gl_limits_above_standard"),
        ({"gl_limits": (3_000_000, 6_000_000)}, "refer", "above_capacity"),
    ],
)
def test_insurance_rules(changes: dict, recommendation: str, rule: str) -> None:
    result = run(insurance_app(**changes))
    assert rule in result.rule_codes()
    assert result.recommendation == recommendation


def test_unbalanced_claim_line_is_a_blocking_siu_referral() -> None:
    application = insurance_app()
    runs = application.loss_runs
    first = runs.years[0]
    bad = Claim(date_of_loss=first.policy_year_start, line="property", paid=1_000, reserve=0, incurred=5_000)
    years = [first.model_copy(update={"claims": [*first.claims, bad]}), *runs.years[1:]]
    changed = application.model_copy(update={"loss_runs": runs.model_copy(update={"years": years})})
    result = run(changed)
    assert "loss_run_integrity" in result.rule_codes()
    assert result.review_queue == "Special Investigations Unit"


def test_flood_zone_location_needs_an_elevation_certificate() -> None:
    application = insurance_app()
    locations = [application.locations[0].model_copy(update={"flood_zone": "AE"})]
    result = run(application.model_copy(update={"locations": locations}))
    missing = next(item for item in result.findings if item.rule == "missing_documents")
    assert "elevation_certificate" in missing.detail


def test_underinsured_building_requests_a_valuation() -> None:
    application = insurance_app()
    location = application.locations[0]
    cheap = location.model_copy(update={"building_value": location.square_feet * 100.0})
    result = run(application.model_copy(update={"locations": [cheap]}))
    assert "possible_underinsurance" in result.rule_codes()


def committed_applications() -> list:
    """Every committed evaluation application."""
    return [
        parse_application(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(APPLICATIONS_DIR.glob("APP-*.json"))
    ]


def test_policy_matches_designed_labels_where_the_detector_is_not_involved() -> None:
    """Regression: with designed screening, every case the injection detector gets right matches its label."""
    checked = 0
    for application in committed_applications():
        hits = screen_application(application)
        if bool(hits) != application.ground_truth.contains_injection:
            continue  # detector errors are measured by the evaluation, not hidden here
        inputs = PolicyInputs(
            screening=ground_truth_screening(application),
            injection_flags=[InjectionFlag(field=hit.field, patterns=hit.patterns) for hit in hits],
        )
        result = evaluate_policy(application, inputs)
        assert result.recommendation == application.ground_truth.expected_recommendation, application.application_id
        assert result.rule_codes() == sorted(application.ground_truth.expected_rules), application.application_id
        checked += 1
    assert checked >= 60


def test_every_detail_number_is_supported_by_its_own_citations() -> None:
    """Each finding and passed check states only numbers found in the items it cites (so memos built on it pass)."""
    for application in committed_applications():
        result = evaluate_policy(application, PolicyInputs(screening=ground_truth_screening(application)))
        evidence = build_evidence(application, result, [], [])
        for entry in [*result.findings, *result.checks_passed]:
            cited = [evidence[item] for item in [*entry.section_ids, *entry.evidence_ids] if item in evidence]
            values = evidence_numbers(cited)
            unsupported = [raw for raw in numbers_in(entry.detail) if not is_supported(raw, values)]
            assert not unsupported, (application.application_id, entry.rule, entry.detail, unsupported)
