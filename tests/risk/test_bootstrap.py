"""Tests for uwcopilot.risk.bootstrap."""

import time

import numpy as np
import pytest

from uwcopilot.risk.bootstrap import bootstrap_ci, paired_bootstrap_diff
from uwcopilot.risk.metrics import brier_score, mean_poisson_deviance, roc_auc

EXPECTED_KEYS = {"estimate", "ci_low", "ci_high", "n_resamples", "n_valid_resamples", "seed", "alpha", "method"}


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def test_bootstrap_ci_keys_and_estimate_on_full_data() -> None:
    values = np.arange(10, dtype=float)
    result = bootstrap_ci(_mean, values, n_resamples=200)
    assert EXPECTED_KEYS <= set(result)
    assert result["estimate"] == pytest.approx(4.5)
    assert result["n_resamples"] == 200
    assert result["n_valid_resamples"] == 200
    assert result["seed"] == 42
    assert result["method"] == "percentile"
    assert result["ci_low"] <= result["estimate"] <= result["ci_high"]


def test_bootstrap_ci_is_reproducible_with_fixed_seed() -> None:
    values = np.random.default_rng(0).normal(size=300)
    first = bootstrap_ci(_mean, values, n_resamples=300, seed=7)
    second = bootstrap_ci(_mean, values, n_resamples=300, seed=7)
    other_seed = bootstrap_ci(_mean, values, n_resamples=300, seed=8)
    assert first == second
    assert first["ci_low"] != other_seed["ci_low"]


def test_bootstrap_ci_matches_manual_percentiles() -> None:
    values = np.random.default_rng(1).normal(size=50)
    result = bootstrap_ci(_mean, values, n_resamples=100, seed=3)
    rng = np.random.default_rng(3)
    manual = [values[rng.integers(0, 50, size=50)].mean() for _ in range(100)]
    assert result["ci_low"] == pytest.approx(np.percentile(manual, 2.5))
    assert result["ci_high"] == pytest.approx(np.percentile(manual, 97.5))


def test_bootstrap_ci_width_close_to_normal_theory() -> None:
    # The standard error of a mean of 2,000 standard normals is 1/sqrt(2000); a 95% CI is about +-1.96 SE.
    values = np.random.default_rng(2).normal(size=2000)
    result = bootstrap_ci(_mean, values)
    expected_width = 2 * 1.96 / np.sqrt(2000)
    assert result["ci_high"] - result["ci_low"] == pytest.approx(expected_width, rel=0.15)


def test_bootstrap_resamples_rows_jointly() -> None:
    # y and p are identical, so every resample keeps them identical and the Brier score is always 0.
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    result = bootstrap_ci(brier_score, y, y.astype(float), n_resamples=100)
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_bootstrap_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        bootstrap_ci(brier_score, [0, 1, 1], [0.1, 0.2])


def test_bootstrap_ignores_nan_resamples_and_counts_them() -> None:
    def nan_when_first_row_missing(values: np.ndarray) -> float:
        return float("nan") if 0 not in values else float(values.mean())

    result = bootstrap_ci(nan_when_first_row_missing, np.arange(5), n_resamples=200)
    assert 0 < result["n_valid_resamples"] < 200


def test_paired_bootstrap_diff_of_identical_models_is_zero() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    scores = rng.random(300)
    result = paired_bootstrap_diff(roc_auc, y, scores, scores, n_resamples=100)
    assert result["estimate"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_paired_bootstrap_diff_detects_better_model() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 1000)
    good = y + rng.normal(0, 0.5, 1000)
    weak = y + rng.normal(0, 2.0, 1000)
    result = paired_bootstrap_diff(roc_auc, y, good, weak, n_resamples=300)
    assert result["estimate"] == pytest.approx(roc_auc(y, good) - roc_auc(y, weak))
    assert result["estimate_a"] == pytest.approx(roc_auc(y, good))
    assert result["ci_low"] > 0
    assert result["share_a_greater"] == 1.0


def test_paired_bootstrap_uses_same_rows_for_both_models() -> None:
    # Model B = model A + 1 everywhere, so with the same rows, mean(A) - mean(B) is exactly -1 in every resample.
    def mean_prediction(y: np.ndarray, pred: np.ndarray) -> float:
        return float(pred.mean())

    pred_a = np.random.default_rng(5).normal(size=100)
    result = paired_bootstrap_diff(mean_prediction, np.zeros(100), pred_a, pred_a + 1.0, n_resamples=100)
    assert result["ci_low"] == pytest.approx(-1.0)
    assert result["ci_high"] == pytest.approx(-1.0)


def test_paired_bootstrap_passes_extra_arrays() -> None:
    rng = np.random.default_rng(6)
    exposure = rng.uniform(0.1, 1.0, 200)
    y = rng.poisson(0.2, 200) / exposure
    pred_a = np.full(200, 0.2)
    pred_b = np.full(200, 0.3)

    def weighted_deviance(y_true: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
        return mean_poisson_deviance(y_true, pred, sample_weight=weights)

    result = paired_bootstrap_diff(weighted_deviance, y, pred_a, pred_b, exposure, n_resamples=50)
    expected = weighted_deviance(y, pred_a, exposure) - weighted_deviance(y, pred_b, exposure)
    assert result["estimate"] == pytest.approx(expected)


def test_bootstrap_is_fast_enough_for_a_large_test_set() -> None:
    # Requirement: 1,000 resamples over 150k rows of a cheap metric in under a minute.
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 150_000)
    probs = rng.random(150_000)
    start = time.perf_counter()
    bootstrap_ci(brier_score, y, probs, n_resamples=1000)
    assert time.perf_counter() - start < 60
