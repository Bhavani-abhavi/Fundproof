"""Tests for uwcopilot.tools.schemas."""

import pytest
from pydantic import ValidationError

from uwcopilot.tools.schemas import (
    GuidelineHit,
    GuidelineQuery,
    GuidelineSearchResult,
    ListSnapshot,
    RiskScoreRequest,
    RiskScoreResponse,
    ScreeningCandidate,
    ScreeningRequest,
    ScreeningResult,
)

SNAPSHOT = ListSnapshot(source="OFAC", file_name="sdn.csv", sha256="a" * 64, snapshot_date="2026-10-10")


def candidate(score: float = 0.91) -> ScreeningCandidate:
    """A synthetic watchlist candidate."""
    return ScreeningCandidate(
        list_entry_id="ofac-0001",
        list_name="SDN",
        source="OFAC",
        matched_name="EXAMPLE TRADING LLC",
        score=score,
        explanation=["token 'example' matched", "country agreed"],
    )


def hit(rank: int, section_id: str = "NBC-3.2") -> GuidelineHit:
    """A guideline hit with a given rank."""
    return GuidelineHit(
        section_id=section_id,
        title="Revenue verification",
        source_doc="northwind_business_credit",
        text="Verified revenue is the sum of eligible deposits.",
        score=0.8,
        rank=rank,
    )


def test_guideline_query_defaults_and_unknown_fields() -> None:
    query = GuidelineQuery(query="  DSCR minimum  ")
    assert query.query == "DSCR minimum"
    assert query.top_k == 5
    with pytest.raises(ValidationError):
        GuidelineQuery(query="x", approve_loan=True)
    with pytest.raises(ValidationError):
        GuidelineQuery(query="")
    with pytest.raises(ValidationError):
        GuidelineQuery(query="x", top_k=100)
    with pytest.raises(ValidationError):
        GuidelineQuery(query="x", source_docs=["../etc/passwd"])


def test_guideline_hit_citation_is_derived_and_checked() -> None:
    assert hit(1).citation == "[NBC-3.2]"
    assert hit(1, section_id="HM-4.1").citation == "[HM-4.1]"
    with pytest.raises(ValidationError):
        GuidelineHit(
            section_id="NBC-3.2", title="t", source_doc="northwind_business_credit", text="x", score=1, rank=1,
            citation="[NBC-9.9]",
        )  # fmt: skip
    with pytest.raises(ValidationError):
        GuidelineHit(section_id="NBC 3.2", title="t", source_doc="northwind_business_credit", text="x", score=1, rank=1)


def test_search_result_ranks_must_be_consecutive() -> None:
    query = GuidelineQuery(query="revenue")
    assert len(GuidelineSearchResult(query=query, hits=[hit(1), hit(2)]).hits) == 2
    with pytest.raises(ValidationError):
        GuidelineSearchResult(query=query, hits=[hit(2), hit(1)])


def test_screening_request_validation() -> None:
    request = ScreeningRequest(name="Example Trading LLC", entity_type="organization", country="US")
    assert request.entity_type == "organization"
    with pytest.raises(ValidationError):
        ScreeningRequest(name="x", entity_type="vessel")


def test_potential_match_always_requires_a_human() -> None:
    request = ScreeningRequest(name="Example Trading", entity_type="organization")
    result = ScreeningResult(
        request=request, candidates=[candidate()], status="potential_match", list_snapshots=[SNAPSHOT]
    )
    assert result.requires_human_review is True
    assert result.model_dump()["requires_human_review"] is True


def test_status_must_agree_with_candidates() -> None:
    request = ScreeningRequest(name="Example Trading", entity_type="organization")
    with pytest.raises(ValidationError):
        ScreeningResult(request=request, candidates=[candidate()], status="clear", list_snapshots=[SNAPSHOT])
    with pytest.raises(ValidationError):
        ScreeningResult(request=request, candidates=[], status="potential_match", list_snapshots=[SNAPSHOT])
    with pytest.raises(ValidationError):
        ScreeningResult(request=request, candidates=[], status="cleared_by_ai", list_snapshots=[SNAPSHOT])


def test_clear_result_records_list_snapshots_and_round_trips() -> None:
    request = ScreeningRequest(name="Plain Bakery Inc", entity_type="organization")
    result = ScreeningResult(request=request, candidates=[], status="clear", list_snapshots=[SNAPSHOT])
    assert result.requires_human_review is False
    assert ScreeningResult.model_validate(result.model_dump()) == result
    with pytest.raises(ValidationError):
        ScreeningResult(request=request, candidates=[], status="clear", list_snapshots=[])
    with pytest.raises(ValidationError):
        ListSnapshot(source="OFAC", file_name="sdn.csv", sha256="not-a-checksum")


def test_candidate_needs_an_explanation_and_a_bounded_score() -> None:
    with pytest.raises(ValidationError):
        candidate(score=1.5)
    with pytest.raises(ValidationError):
        ScreeningCandidate(
            list_entry_id="x", list_name="SDN", source="OFAC", matched_name="X", score=0.9, explanation=[]
        )


def test_risk_request_and_response() -> None:
    request = RiskScoreRequest(
        product="credit", features={"dti": 18.5, "annual_inc": None}, overrides={"dti": 25.0}, scenario="dti_up"
    )
    assert request.features["annual_inc"] is None
    with pytest.raises(ValidationError):
        RiskScoreRequest(product="mortgage", features={})
    with pytest.raises(ValidationError):
        RiskScoreRequest(product="credit", features={"bad name; drop": 1})
    response = RiskScoreResponse(
        product="credit",
        prediction=0.07,
        risk_band="C",
        reason_codes=[{"feature": "dti", "contribution": 0.02, "description": "Higher debt-to-income"}],
        model_version="credit-lgbm-1",
        scenario="dti_up",
        unknown_features=["annual_inc"],
    )
    assert response.reason_codes[0].feature == "dti"
