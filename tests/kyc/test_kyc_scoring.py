"""Tests for uwcopilot.kyc.scoring: name similarity, attribute adjustments and explanation text."""

import pytest

from uwcopilot.kyc.scoring import (
    AttributeCheck,
    ScoringWeights,
    attribute_check,
    monge_elkan,
    name_scores,
    rules_score,
    token_explanations,
    token_similarity,
)

WEIGHTS = ScoringWeights()


def test_initials_match_tokens_with_the_same_first_letter() -> None:
    assert token_similarity("j", "john", 0.9) == 0.9
    assert token_similarity("k", "john", 0.9) == 0.0
    assert token_similarity("john", "john", 0.9) == 1.0


def test_aligned_jaro_winkler_ignores_token_order() -> None:
    forward = name_scores(("ali", "hassan"), ("hassan", "ali"), WEIGHTS)
    assert forward.jaro_winkler_aligned == pytest.approx(1.0)
    assert forward.name_score == pytest.approx(1.0)


def test_a_subset_name_is_not_a_perfect_match() -> None:
    # token_set_ratio alone gives 100 for a subset; the two-direction alignment does not.
    scores = name_scores(("smith",), ("john", "william", "smith"), WEIGHTS)
    assert scores.token_set == pytest.approx(1.0)
    assert scores.jaro_winkler_aligned < 0.9
    assert scores.name_score < 0.95
    assert monge_elkan(("smith",), ("john", "smith"), 0.9) == pytest.approx(1.0)


def test_typos_score_higher_than_different_names() -> None:
    close = name_scores(("mohammed", "hassan"), ("mohammad", "hasan"), WEIGHTS).name_score
    far = name_scores(("mohammed", "hassan"), ("peter", "jones"), WEIGHTS).name_score
    assert close > 0.9 > far


def test_attribute_check_agree_disagree_and_unknown() -> None:
    agree = attribute_check("person", "ir", "person", ("ir",), WEIGHTS)
    assert agree.adjustment == pytest.approx(0.05)
    disagree = attribute_check("organization", "us", "person", ("ir",), WEIGHTS)
    assert disagree.adjustment == pytest.approx(-0.07)
    assert any("entity type differs" in note for note in disagree.notes)
    assert any("country differs" in note for note in disagree.notes)
    unknown = attribute_check("person", None, "other", (), WEIGHTS)
    assert unknown.adjustment == 0.0
    assert all("not compared" in note for note in unknown.notes)


def test_rules_score_stays_in_range_and_keeps_order() -> None:
    perfect = name_scores(("a1", "b2"), ("a1", "b2"), WEIGHTS)
    best = rules_score(perfect, AttributeCheck(adjustment=0.05), WEIGHTS)
    neutral = rules_score(perfect, AttributeCheck(adjustment=0.0), WEIGHTS)
    worst = rules_score(name_scores(("x",), ("y",), WEIGHTS), AttributeCheck(adjustment=-0.07), WEIGHTS)
    assert best == pytest.approx(1.0)
    assert worst == pytest.approx(0.0)
    assert worst < neutral < best


def test_explanations_name_matched_tokens_and_phonetic_agreement() -> None:
    lines = token_explanations(("mohammed", "aly", "hassan", "q"), ("muhammad", "ali", "hassan", "tikriti"), WEIGHTS)
    text = "\n".join(lines)
    assert "'hassan' = 'hassan' (exact)" in text
    assert "'mohammed' ~ 'muhammad' (same Double Metaphone code MHMT" in text
    assert "'aly' ~ 'ali'" in text
    assert "'q' has no counterpart" in text
    assert "phonetic agreement: 3 of 3 customer tokens share a Double Metaphone code" in text
    assert "list-name tokens not matched by the customer name: tikriti" in text
