"""Tests for uwcopilot.risk.fairness, using small groups whose rates can be checked by hand."""

import math

import pytest

from uwcopilot.risk.fairness import adverse_impact_ratio, equal_opportunity_gap, selection_rates

# Group A: 4 of 5 selected (0.8). Group B: 2 of 5 selected (0.4).
SELECTED = [1, 1, 1, 1, 0, 1, 1, 0, 0, 0]
GROUPS = ["A"] * 5 + ["B"] * 5


def test_selection_rates_by_hand() -> None:
    assert selection_rates(SELECTED, GROUPS) == pytest.approx({"A": 0.8, "B": 0.4})


def test_adverse_impact_ratio_defaults_to_highest_rate_group() -> None:
    result = adverse_impact_ratio(SELECTED, GROUPS)
    assert result["reference_group"] == "A"
    assert result["threshold"] == 0.8
    assert result["groups"]["A"]["impact_ratio"] == pytest.approx(1.0)
    assert result["groups"]["B"]["impact_ratio"] == pytest.approx(0.5)  # 0.4 / 0.8
    assert result["groups"]["B"]["below_threshold"] is True
    assert result["groups"]["A"]["below_threshold"] is False
    assert result["groups"]["B"]["n"] == 5


def test_adverse_impact_ratio_with_explicit_reference() -> None:
    result = adverse_impact_ratio(SELECTED, GROUPS, reference_group="B")
    assert result["groups"]["A"]["impact_ratio"] == pytest.approx(2.0)


def test_adverse_impact_ratio_passes_at_four_fifths() -> None:
    # A: 5/5 = 1.0, B: 4/5 = 0.8 -> ratio exactly 0.8 is not below the threshold.
    result = adverse_impact_ratio([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], GROUPS)
    assert result["groups"]["B"]["impact_ratio"] == pytest.approx(0.8)
    assert result["groups"]["B"]["below_threshold"] is False


def test_adverse_impact_ratio_unknown_reference() -> None:
    with pytest.raises(ValueError, match="not found"):
        adverse_impact_ratio(SELECTED, GROUPS, reference_group="C")


def test_groups_with_missing_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        selection_rates([1, 0], ["A", None])


def test_equal_opportunity_gap_by_hand() -> None:
    # Truly favorable people: A rows 0-3 (3 of 4 selected -> TPR 0.75), B rows 5-8 (1 of 4 selected -> 0.25).
    y_true = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    selected = [1, 1, 1, 0, 1, 1, 0, 0, 0, 1]
    result = equal_opportunity_gap(y_true, selected, GROUPS)
    assert result["reference_group"] == "A"
    assert result["groups"]["A"]["tpr"] == pytest.approx(0.75)
    assert result["groups"]["B"]["tpr"] == pytest.approx(0.25)
    assert result["groups"]["B"]["gap_vs_reference"] == pytest.approx(-0.5)
    assert result["groups"]["A"]["n_favorable_outcome"] == 4
    assert result["max_gap"] == pytest.approx(0.5)


def test_equal_opportunity_gap_group_without_favorable_outcomes() -> None:
    y_true = [1, 1, 0, 0]
    selected = [1, 0, 1, 0]
    result = equal_opportunity_gap(y_true, selected, ["A", "A", "B", "B"])
    assert math.isnan(result["groups"]["B"]["tpr"])
    assert result["groups"]["B"]["n_favorable_outcome"] == 0
    assert result["max_gap"] == pytest.approx(0.0)
