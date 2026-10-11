"""Tests for uwcopilot.kyc.evaluation: threshold choice, curves, metrics and bootstrap intervals."""

import math

import numpy as np
import pytest

from uwcopilot.kyc.evaluation import (
    bootstrap_mean,
    bootstrap_screening_metrics,
    choose_threshold,
    paired_recall_difference,
    recall_alert_curve,
    recall_at_budget,
    screening_metrics,
)

# 10 customers: 2 true matches (scores 0.9 and 0.6), 1 hard negative (0.8), 7 clean.
SCORES = np.array([0.9, 0.6, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
LABELS = np.array(["true_match", "true_match", "hard_negative"] + ["clean"] * 7)
IS_MATCH = LABELS == "true_match"


def test_threshold_is_the_lowest_score_within_the_budget() -> None:
    # budget 300 per 1,000 = at most 3 of 10 customers alerted -> threshold 0.7 (0.9, 0.8, 0.7 alerted)
    choice = choose_threshold(SCORES, IS_MATCH, 300)
    assert choice.threshold == pytest.approx(0.7)
    assert choice.alerts_per_1000 == pytest.approx(300)
    assert choice.recall == pytest.approx(0.5)
    # a larger budget reaches the second true match
    assert choose_threshold(SCORES, IS_MATCH, 600).recall == pytest.approx(1.0)


def test_ties_that_break_the_budget_mean_no_alerts() -> None:
    tied = np.array([1.0, 1.0, 1.0, 0.0])
    choice = choose_threshold(tied, np.array([True, False, False, False]), 250)
    assert choice.threshold > 1.0
    assert choice.alerts_per_1000 == 0.0 and choice.recall == 0.0


def test_curve_and_recall_at_budget() -> None:
    curve = recall_alert_curve(SCORES, IS_MATCH)
    assert curve[0] == {"threshold": 0.9, "alerts_per_1000": 100.0, "recall": 0.5}
    assert curve[-1]["alerts_per_1000"] == 1000.0 and curve[-1]["recall"] == 1.0
    assert recall_at_budget(curve, 300) == 0.5
    assert recall_at_budget(curve, 0) == 0.0


def test_screening_metrics() -> None:
    metrics = screening_metrics(SCORES >= 0.6, LABELS)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["alerts_per_1000"] == pytest.approx(400.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["hard_negative_false_positive_rate"] == pytest.approx(1.0)
    assert metrics["clean_false_positive_rate"] == pytest.approx(1 / 7)
    assert math.isnan(screening_metrics(SCORES > 2, LABELS)["precision"])  # no alerts, no precision


def test_bootstrap_is_reproducible_and_brackets_the_point_estimate() -> None:
    first = bootstrap_screening_metrics(SCORES >= 0.6, LABELS, n_resamples=200, seed=42)
    second = bootstrap_screening_metrics(SCORES >= 0.6, LABELS, n_resamples=200, seed=42)
    assert first == second
    alerts = first["alerts_per_1000"]
    assert alerts["ci95_low"] <= alerts["value"] <= alerts["ci95_high"]
    mean = bootstrap_mean(np.array([1.0, 0.0, 1.0, 1.0]), n_resamples=200)
    assert mean["value"] == pytest.approx(0.75)


def test_paired_difference_is_zero_for_identical_methods() -> None:
    alerted = SCORES >= 0.6
    result = paired_recall_difference(alerted, alerted, LABELS, n_resamples=100)
    assert result["difference"] == 0.0
    assert result["ci95_low"] == 0.0 and result["ci95_high"] == 0.0
    assert result["share_of_resamples_challenger_better"] == 0.0


def test_paired_difference_sign() -> None:
    weaker = SCORES >= 0.85  # catches one true match
    stronger = SCORES >= 0.55  # catches both
    result = paired_recall_difference(weaker, stronger, LABELS, n_resamples=200)
    assert result["difference"] == pytest.approx(0.5)
    assert result["ci95_low"] >= 0.0
