"""Tests for the SYNTHETIC packets' applications and problem records (JSON-only packets, no drawing)."""

import pytest

from uwcopilot.agent.applications import parse_application, screened_parties
from uwcopilot.agent.policy import PolicyInputs, ScreeningOutcome, evaluate_policy
from uwcopilot.docintel.deposits import (
    average_daily_balance,
    average_monthly_loan_payments,
    average_monthly_mca_remittance,
    classify_deposit,
    classify_withdrawal,
    monthly_deposits,
    negative_balance_months,
    new_loans_last_90_days,
    nsf_events,
)
from uwcopilot.docintel.levers import apply_substitutions
from uwcopilot.docintel.schemas import LoanPacket, parse_extraction
from uwcopilot.docintel.synthetic import build_packet
from uwcopilot.docintel.synthetic.common import packet_seed
from uwcopilot.docintel.synthetic.problems import evidence_request_ids
from uwcopilot.docintel.truth import PacketTruth

COUNT = 40


def json_only(line: str, count: int = COUNT) -> list[PacketTruth]:
    """JSON-only truths PKT-x-001.. for a line, seed 42."""
    letter = "C" if line == "credit" else "I"
    truths = []
    for number in range(1, count + 1):
        packet_id = f"PKT-{letter}-{number:03d}"
        pdf, truth = build_packet(packet_id, line, packet_seed(42, packet_id), "dev", draw=False)
        assert pdf is None
        truths.append(truth)
    return truths


@pytest.fixture(scope="module")
def credit_truths() -> list[PacketTruth]:
    """40 JSON-only loan packets."""
    return json_only("credit")


@pytest.fixture(scope="module")
def insurance_truths() -> list[PacketTruth]:
    """40 JSON-only insurance packets."""
    return json_only("insurance")


def clean_screening(application: dict) -> PolicyInputs:
    """Policy inputs with every party screened clear."""
    parsed = parse_application(application)
    return PolicyInputs(
        screening=[ScreeningOutcome(party_id=party.party_id, status="clear") for party in screened_parties(parsed)]
    )


@pytest.mark.parametrize("line", ["credit", "insurance"])
def test_applications_validate_and_the_policy_runs(line: str, credit_truths: list, insurance_truths: list) -> None:
    truths = credit_truths if line == "credit" else insurance_truths
    for truth in truths:
        documented = parse_application(truth.application_as_documented)
        latent = parse_application(truth.application_latent)
        assert documented.application_id == latent.application_id == truth.packet_id.replace("PKT", "APP")
        assert documented.ground_truth is None and documented.data_label == "SYNTHETIC"
        result = evaluate_policy(documented, clean_screening(truth.application_as_documented))
        assert result.recommendation in {"approve", "refer", "decline_for_review", "request_more_information"}
        missing = set(truth.missing_required_documents)
        assert missing.isdisjoint(truth.application_as_documented["documents_provided"])


@pytest.mark.parametrize("line", ["credit", "insurance"])
def test_problem_records_are_consistent(line: str, credit_truths: list, insurance_truths: list) -> None:
    truths = credit_truths if line == "credit" else insurance_truths
    requests = set(evidence_request_ids())
    for truth in truths:
        assert truth.scenario == ("clean" if not truth.problems else "+".join(p.problem_type for p in truth.problems))
        doc_ids = {document.doc_id for document in truth.documents}
        for problem in truth.problems:
            assert problem.resolving_request in requests
            assert set(problem.values_by_doc) <= doc_ids and set(problem.doc_ids) <= doc_ids
            for option in problem.resolution_options:
                parse_application(apply_substitutions(truth.application_as_documented, option.substitutions))
            parse_application(apply_substitutions(truth.application_as_documented, problem.latent_substitutions))


@pytest.mark.parametrize("line", ["credit", "insurance"])
def test_injections_point_at_the_printed_text(line: str, credit_truths: list, insurance_truths: list) -> None:
    truths = credit_truths if line == "credit" else insurance_truths
    count = 0
    for truth in truths:
        pages = {page.page_number: " ".join(page.text_lines) for page in truth.pages}
        assert len(truth.injections) == sum(
            1 for problem in truth.problems if problem.problem_type == "prompt_injection"
        )
        for injection in truth.injections:
            count += 1
            assert " ".join(injection.text.split()[:5]) in pages[injection.page_number]
    assert count > 0


def test_json_only_truth_matches_the_drawn_truth() -> None:
    for packet_id, line in (("PKT-C-003", "credit"), ("PKT-I-003", "insurance")):
        _, drawn = build_packet(packet_id, line, packet_seed(42, packet_id), "test", draw=True)
        _, layout = build_packet(packet_id, line, packet_seed(42, packet_id), "test", draw=False)
        assert layout.pdf_file == "" and layout.pdf_sha256 == ""
        assert layout.model_dump(exclude={"pdf_file", "pdf_sha256"}) == drawn.model_dump(
            exclude={"pdf_file", "pdf_sha256"}
        )


def test_rows_classify_as_their_documented_category(credit_truths: list) -> None:
    for truth in credit_truths:
        packet = parse_extraction(truth.oracle_extraction)
        assert isinstance(packet, LoanPacket)
        for statement in packet.statements:
            for row in statement.transactions:
                if row.deposit.found:
                    assert classify_deposit(row.description.value) in {
                        "revenue", "cash_revenue", "transfer", "refund", "loan_proceeds", "owner_contribution",
                        "one_time", "unidentified",
                    }  # fmt: skip
                else:
                    assert classify_withdrawal(row.description.value) in {
                        "loan_payment",
                        "mca_remittance",
                        "nsf_fee",
                        "other",
                    }


def test_as_documented_account_figures_match_deposits_rules(credit_truths: list) -> None:
    """The generator's own computation agrees with docintel/deposits.py run on the oracle extraction."""
    for truth in credit_truths:
        packet = parse_extraction(truth.oracle_extraction)
        statements = packet.statements
        application = truth.application_as_documented
        summary = application["bank_statements"]
        assert [monthly_deposits(statement) for statement in statements] == summary["months"], truth.packet_id
        assert nsf_events(statements) == summary["nsf_events_last_6_months"]
        assert negative_balance_months(statements) == summary["negative_balance_months_last_12"]
        assert average_daily_balance(statements) == summary["average_daily_balance_last_3_months"], truth.packet_id
        debt = application["debt"]
        assert average_monthly_loan_payments(statements) == debt["existing_monthly_payment_observed"], truth.packet_id
        assert average_monthly_mca_remittance(statements) == debt["mca_monthly_remittance"], truth.packet_id
        as_of = packet.application.application_date.value
        assert new_loans_last_90_days(statements, as_of) == application["new_loans_last_90_days"], truth.packet_id


def test_latent_application_applies_every_latent_substitution(credit_truths: list, insurance_truths: list) -> None:
    for truth in [*credit_truths, *insurance_truths]:
        if not truth.problems:
            assert truth.application_latent == truth.application_as_documented
