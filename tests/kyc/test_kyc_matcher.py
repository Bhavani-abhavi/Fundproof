"""Tests for uwcopilot.kyc.matcher: the full pipeline on a six-entry test watchlist."""

import pytest
from kyc_fixtures import tiny_matcher

from uwcopilot.kyc.matcher import HUMAN_REVIEW_NOTE
from uwcopilot.tools.schemas import ScreeningRequest, ScreeningResult


def screen(name: str, entity_type: str, country: str | None = None, **kwargs) -> ScreeningResult:
    """Screen one name with a fresh tiny matcher."""
    return tiny_matcher(**kwargs).screen(ScreeningRequest(name=name, entity_type=entity_type, country=country))


def test_noisy_true_match_is_a_potential_match_with_explanation() -> None:
    result = screen("Dr. Mohamed Aly Hasan", "person", "IQ")
    assert result.status == "potential_match"
    assert result.requires_human_review is True
    top = result.candidates[0]
    assert top.list_entry_id == "OFAC-1001"
    assert top.source == "OFAC"
    assert top.list_name == "OFAC SDN (SDGT)"
    text = "\n".join(top.explanation)
    assert "stripped before matching: dr" in text
    assert "matched tokens:" in text
    assert "phonetic agreement:" in text
    assert "entity type agrees: both person" in text
    assert "country agrees: iq" in text
    assert "alert threshold" in text
    assert top.explanation[-1] == HUMAN_REVIEW_NOTE


def test_matched_alias_is_reported_with_the_lists_own_spelling() -> None:
    result = screen("Muhammad Ali Hasan", "person")
    top = result.candidates[0]
    assert top.matched_name == "Muhammad Ali HASAN"
    assert any("matched alias 'Muhammad Ali HASAN'" in line for line in top.explanation)


def test_reordered_organization_with_new_suffix_matches() -> None:
    result = screen("Test Aerocaribbean Airlines GmbH", "organization", "CU")
    assert result.status == "potential_match"
    assert result.candidates[0].list_entry_id == "OFAC-1002"


def test_unrelated_name_is_clear_and_still_carries_snapshots() -> None:
    result = screen("Penelope Quarterfield", "person", "US")
    assert result.status == "clear"
    assert result.candidates == []
    assert result.requires_human_review is False
    assert [snapshot.file_name for snapshot in result.list_snapshots] == ["sdn.csv", "targets.simple.csv"]


def test_type_disagreement_lowers_the_score() -> None:
    matcher = tiny_matcher()
    as_person = matcher.match("Ocean Tester", "person").candidates[0]
    assert as_person.attributes.adjustment < 0
    as_org = matcher.match("Ocean Tester", "organization").candidates[0]
    assert as_org.rules_score == pytest.approx(as_person.rules_score)  # vessel vs either: both disagree


def test_threshold_controls_the_status() -> None:
    assert screen("Johana Kowalczyk", "person", threshold=0.5).status == "potential_match"
    assert screen("Johana Kowalczyk", "person", threshold=0.999).status == "clear"


def test_reranker_adds_embedding_similarity_and_changes_the_final_score() -> None:
    matcher = tiny_matcher(reranker_weight=0.5, with_embedder=True)
    outcome = matcher.match("Ivan Petrovich Sidorenko", "person", "RU")
    best = outcome.candidates[0]
    assert best.embedding_similarity is not None
    assert best.final_score(0.0) == best.rules_score
    assert best.final_score(0.5) == pytest.approx(0.5 * best.rules_score + 0.5 * max(0.0, best.embedding_similarity))
    result = matcher.screen(ScreeningRequest(name="Ivan Petrovich Sidorenko", entity_type="person"))
    assert any("embedding cosine" in line for line in result.candidates[0].explanation)


def test_outcome_reports_stage_sizes() -> None:
    outcome = tiny_matcher().match("Mohammed Ali Hassan", "person")
    assert outcome.blocking.size_before_cap >= len(outcome.blocking.name_ids) >= len(outcome.prerank_name_ids)
    assert outcome.best_score(0.0) == outcome.candidates[0].rules_score
    assert tiny_matcher().match("Q", "person").best_score(0.0) == 0.0
