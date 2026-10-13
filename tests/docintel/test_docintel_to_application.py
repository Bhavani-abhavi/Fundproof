"""Tests for to_application (docintel/to_application*.py): extraction -> agent application JSON with provenance.

The contract test maps the generator's perfect (oracle) extraction of 20 JSON-only packets per line and compares
the result with the generator's application_as_documented, field by field with a $0.01 tolerance. The packets use
ids and a seed outside the evaluation set (PKT-*-201 to 220, seed 2026), so nothing here touches the test split.
"""

from functools import cache
from typing import Any

import pytest

from uwcopilot.agent.applications import parse_application
from uwcopilot.docintel.checks import run_checks
from uwcopilot.docintel.schemas import Cited, InsuranceSubmission, LoanPacket, parse_extraction
from uwcopilot.docintel.synthetic import build_packet
from uwcopilot.docintel.synthetic.common import packet_seed
from uwcopilot.docintel.to_application import to_application
from uwcopilot.docintel.truth import PacketTruth

CONTRACT_SEED = 2026
CONTRACT_IDS = range(201, 221)
TOLERANCE = 0.01


@cache
def json_only_truth(packet_id: str, line: str) -> PacketTruth:
    """The truth of one JSON-only packet (no PDF is drawn)."""
    _, truth = build_packet(packet_id, line, packet_seed(CONTRACT_SEED, packet_id), "dev", draw=False)
    return truth


def differences(first: Any, second: Any, path: str = "") -> list[str]:
    """Paths where two JSON values differ (numbers within TOLERANCE are equal; booleans are compared exactly)."""
    if isinstance(first, dict) and isinstance(second, dict):
        found = []
        for key in sorted(set(first) | set(second)):
            if key not in first or key not in second:
                found.append(f"{path}.{key} (missing on one side)")
            else:
                found.extend(differences(first[key], second[key], f"{path}.{key}"))
        return found
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            return [f"{path} (length {len(first)} != {len(second)})"]
        found = []
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            found.extend(differences(left, right, f"{path}[{index}]"))
        return found
    if isinstance(first, bool) or isinstance(second, bool):
        return [] if first is second else [f"{path} ({first!r} != {second!r})"]
    if isinstance(first, int | float) and isinstance(second, int | float):
        return [] if abs(first - second) <= TOLERANCE else [f"{path} ({first} != {second})"]
    return [] if first == second else [f"{path} ({first!r} != {second!r})"]


@pytest.mark.parametrize("line", ["credit", "insurance"])
def test_oracle_extraction_maps_to_the_as_documented_application(line: str) -> None:
    letter = "C" if line == "credit" else "I"
    for number in CONTRACT_IDS:
        truth = json_only_truth(f"PKT-{letter}-{number:03d}", line)
        application, provenance, unknowns = to_application(parse_extraction(truth.oracle_extraction))
        assert differences(application, truth.application_as_documented) == [], truth.packet_id
        assert unknowns == [], truth.packet_id
        parse_application(application)
        assert provenance["application_date"][0]["doc_id"] == "D01"


def test_credit_documents_can_come_from_the_required_documents_check() -> None:
    truth = json_only_truth("PKT-C-201", "credit")
    extraction = parse_extraction(truth.oracle_extraction)
    with_checks, _, _ = to_application(extraction, run_checks(extraction))
    without_checks, _, _ = to_application(extraction)
    assert with_checks["documents_provided"] == without_checks["documents_provided"]
    assert with_checks["documents_provided"] == truth.application_as_documented["documents_provided"]


def test_provenance_cites_the_page_and_line_of_each_value() -> None:
    truth = json_only_truth("PKT-C-202", "credit")
    extraction = parse_extraction(truth.oracle_extraction)
    _, provenance, _ = to_application(extraction)
    cited = extraction.application.requested_amount
    assert provenance["requested_amount"] == [{"doc_id": cited.doc_id, "page": cited.page, "line": cited.line}]
    assert len(provenance["bank_statements.months[0]"]) == 1
    statement_docs = {source["doc_id"] for source in provenance["bank_statements.nsf_events_last_6_months"]}
    assert statement_docs == {statement.doc_id for statement in extraction.statements}


def test_missing_values_are_none_and_listed_as_unknown() -> None:
    truth = json_only_truth("PKT-C-203", "credit")
    extraction = parse_extraction(truth.oracle_extraction)
    assert isinstance(extraction, LoanPacket)
    extraction.application.fein = Cited()
    extraction.profit_and_loss = None
    application, provenance, unknowns = to_application(extraction)
    assert application["applicant"]["fein"] is None
    assert application["debt"]["ttm_ebitda"] is None
    assert {"applicant.fein", "debt.ttm_ebitda", "debt.ttm_pnl_revenue"} <= set(unknowns)
    assert "applicant.fein" not in provenance
    empty, _, empty_unknowns = to_application(LoanPacket(packet_id="PKT-C-999"))
    assert empty["application_id"] == "APP-C-999"
    assert empty["bank_statements"]["average_daily_balance_last_3_months"] is None
    assert {"application_form", "owners", "bank_statements.account_number"} <= set(empty_unknowns)


def test_loss_runs_count_as_provided_only_when_every_prior_carrier_has_one() -> None:
    truth = json_only_truth("PKT-I-201", "insurance")
    extraction = parse_extraction(truth.oracle_extraction)
    assert isinstance(extraction, InsuranceSubmission)
    application, _, _ = to_application(extraction)
    carriers = {row.carrier.value for row in extraction.application.prior_carriers}
    runs = {run.carrier.value for run in extraction.loss_runs}
    assert ("loss_runs" in application["documents_provided"]) == (carriers <= runs)
    extraction.loss_runs = extraction.loss_runs[1:]
    reduced, _, _ = to_application(extraction)
    assert "loss_runs" not in reduced["documents_provided"]
    assert reduced["documents_provided"][0] == "signed_application"


def test_insurance_application_id_and_defaults() -> None:
    truth = json_only_truth("PKT-I-202", "insurance")
    application, _, _ = to_application(parse_extraction(truth.oracle_extraction))
    assert application["application_id"] == "APP-I-202"
    assert application["applicant"]["prior_experience_years"] == 0
    assert all(location["flood_zone"] == "X" for location in application["locations"])
    assert all(owner["credit_score"] is None for owner in application["owners"])
