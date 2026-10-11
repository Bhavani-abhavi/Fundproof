"""Tests for uwcopilot.risk.metrics, using small hand-checkable examples and sklearn/scipy as references."""

import math

import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.metrics import mean_gamma_deviance as sk_mean_gamma_deviance
from sklearn.metrics import mean_poisson_deviance as sk_mean_poisson_deviance

from uwcopilot.risk.metrics import (
    brier_score,
    calibration_table,
    expected_calibration_error,
    gini_from_auc,
    ks_statistic,
    lorenz_curve,
    lorenz_gini,
    mean_gamma_deviance,
    mean_poisson_deviance,
    roc_auc,
)

# Classic example: positives scored 0.35 and 0.8, negatives 0.1 and 0.4.
# Of the 4 (positive, negative) pairs, 3 have the positive scored higher, so AUC = 3/4.
Y_SMALL = [0, 0, 1, 1]
SCORES_SMALL = [0.1, 0.4, 0.35, 0.8]


def test_roc_auc_hand_example() -> None:
    assert roc_auc(Y_SMALL, SCORES_SMALL) == pytest.approx(0.75)


def test_roc_auc_counts_ties_as_half() -> None:
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    # Pairs: (0.5 vs 0.5) tie = 0.5, (0.5 vs 0.2) win = 1 -> 1.5 / 2.
    assert roc_auc([0, 0, 1], [0.5, 0.2, 0.5]) == pytest.approx(0.75)


def test_roc_auc_matches_sklearn_with_many_ties() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = rng.integers(0, 20, 500) / 20  # only 20 distinct values -> many ties
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_roc_auc_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both classes"):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_ks_hand_example() -> None:
    # Thresholds from the top: 0.8 -> TPR 0.5, FPR 0; 0.4 -> 0.5, 0.5; 0.35 -> 1, 0.5; 0.1 -> 1, 1. Max gap 0.5.
    assert ks_statistic(Y_SMALL, SCORES_SMALL) == pytest.approx(0.5)


def test_ks_perfect_separation_is_one() -> None:
    assert ks_statistic([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_ks_matches_scipy_two_sample_with_ties() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    scores = rng.integers(0, 10, 400) + y * 2  # positives shifted up, many ties
    expected = ks_2samp(scores[y == 1], scores[y == 0]).statistic
    assert ks_statistic(y, scores) == pytest.approx(expected)


def test_gini_from_auc() -> None:
    assert gini_from_auc(0.75) == pytest.approx(0.5)
    assert gini_from_auc(0.5) == pytest.approx(0.0)


def test_brier_hand_example_and_weights() -> None:
    # ((0.2 - 0)^2 + (0.6 - 1)^2) / 2 = (0.04 + 0.16) / 2 = 0.1
    assert brier_score([0, 1], [0.2, 0.6]) == pytest.approx(0.1)
    # Weighted: (3 * 0.04 + 1 * 0.16) / 4 = 0.07
    assert brier_score([0, 1], [0.2, 0.6], sample_weight=[3, 1]) == pytest.approx(0.07)
    assert brier_score([0, 1], [0.2, 0.6], sample_weight=[3, 1]) == pytest.approx(
        brier_score_loss([0, 1], [0.2, 0.6], sample_weight=[3, 1])
    )


def test_ece_hand_example() -> None:
    # Two equal-width bins. Low bin: predicted mean 0.15, observed 0 -> gap 0.15.
    # High bin: predicted mean 0.8, observed 1 -> gap 0.2. Each bin holds half the rows: 0.5*0.15 + 0.5*0.2.
    y = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.7, 0.9]
    assert expected_calibration_error(y, probs, n_bins=2) == pytest.approx(0.175)
    assert expected_calibration_error(y, probs, n_bins=2, strategy="quantile") == pytest.approx(0.175)


def test_ece_is_zero_when_calibrated_and_handles_probability_one() -> None:
    assert expected_calibration_error([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
    # A probability of exactly 1.0 goes in the last bin, where the observed rate is 1.
    assert expected_calibration_error([1, 0], [1.0, 0.0], n_bins=10) == pytest.approx(0.0)


def test_ece_rejects_bad_strategy_and_range() -> None:
    with pytest.raises(ValueError, match="strategy"):
        expected_calibration_error([0, 1], [0.1, 0.9], strategy="other")
    with pytest.raises(ValueError, match="between 0 and 1"):
        expected_calibration_error([0, 1], [0.1, 1.5])


def test_calibration_table_equal_count_bins() -> None:
    predicted = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    observed = [0, 0, 0, 1, 0, 1, 1, 1, 1, 1]
    table = calibration_table(observed, predicted, n_bins=5)
    assert list(table["bin"]) == [1, 2, 3, 4, 5]
    assert list(table["n_rows"]) == [2, 2, 2, 2, 2]
    assert table["mean_predicted"].tolist() == pytest.approx([0.15, 0.35, 0.55, 0.75, 0.95])
    assert table["mean_observed"].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])


def test_calibration_table_uses_weights() -> None:
    # Total weight 4. Cumulative weight before each row: 0, 1, 2 -> shares 0, 0.25, 0.5 -> bins 1, 1, 2.
    table = calibration_table([0.0, 2.0, 1.0], [0.1, 0.2, 0.3], n_bins=2, sample_weight=[1, 1, 2])
    assert list(table["n_rows"]) == [2, 1]
    assert table["total_weight"].tolist() == pytest.approx([2.0, 2.0])
    assert table["mean_predicted"].tolist() == pytest.approx([0.15, 0.3])
    assert table["mean_observed"].tolist() == pytest.approx([1.0, 1.0])


def test_poisson_deviance_hand_example() -> None:
    # Unit deviances with mu = 1: y=0 -> 2; y=1 -> 0; y=2 -> 2 * (2 ln 2 - 1).
    expected = (2.0 + 0.0 + 2.0 * (2.0 * math.log(2.0) - 1.0)) / 3.0
    assert mean_poisson_deviance([0, 1, 2], [1, 1, 1]) == pytest.approx(expected)


def test_poisson_deviance_matches_sklearn_with_weights() -> None:
    rng = np.random.default_rng(2)
    y = rng.poisson(0.3, 200).astype(float)
    mu = rng.uniform(0.05, 1.0, 200)
    weights = rng.uniform(0.1, 1.0, 200)
    assert mean_poisson_deviance(y, mu, sample_weight=weights) == pytest.approx(
        sk_mean_poisson_deviance(y, mu, sample_weight=weights)
    )


def test_poisson_deviance_rejects_zero_prediction() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        mean_poisson_deviance([0, 1], [0.0, 1.0])


def test_gamma_deviance_hand_example_and_sklearn() -> None:
    # Unit deviances with mu = 1: y=1 -> 0; y=2 -> 2 * (ln(1/2) + 2 - 1) = 2 * (1 - ln 2).
    expected = (0.0 + 2.0 * (1.0 - math.log(2.0))) / 2.0
    assert mean_gamma_deviance([1, 2], [1, 1]) == pytest.approx(expected)
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 500.0, 100)
    mu = rng.uniform(200.0, 2000.0, 100)
    weights = rng.integers(1, 4, 100)
    assert mean_gamma_deviance(y, mu, sample_weight=weights) == pytest.approx(
        sk_mean_gamma_deviance(y, mu, sample_weight=weights)
    )


def test_gamma_deviance_rejects_zero_target() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        mean_gamma_deviance([0.0, 1.0], [1.0, 1.0])


def test_lorenz_curve_hand_example() -> None:
    # Sorted by predicted risk, claims are [0, 0, 1, 1]. Cumulative shares start at 0.
    exposure_share, claim_share = lorenz_curve([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4])
    assert exposure_share.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert claim_share.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.5, 1.0])


def test_lorenz_gini_hand_examples() -> None:
    # Area under the curve above = 0.0625 + 0.1875 = 0.25, so Gini = 1 - 2 * 0.25 = 0.5.
    assert lorenz_gini([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.5)
    # Reversed ranking puts the claims first: Gini is negative.
    assert lorenz_gini([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == pytest.approx(-0.5)


def test_lorenz_gini_with_exposure() -> None:
    # Frequencies [0, 2] with exposures [3, 1] -> claims [0, 2]. Exposure shares [0, 0.75, 1], claims [0, 0, 1].
    # Area = 0.5 * 0.25 * (0 + 1) = 0.125, Gini = 0.75.
    assert lorenz_gini([0.0, 2.0], [0.1, 0.5], exposure=[3.0, 1.0]) == pytest.approx(0.75)


def test_lorenz_curve_merges_tied_predictions() -> None:
    # Two rows share the prediction 0.2: they form one block, so the curve has one point per distinct prediction
    # and runs straight across the block (claims [0], then [1 + 0] together, then [1]).
    exposure_share, claim_share = lorenz_curve([0, 1, 0, 1], [0.1, 0.2, 0.2, 0.3])
    assert exposure_share.tolist() == pytest.approx([0.0, 0.25, 0.75, 1.0])
    assert claim_share.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])


def test_lorenz_gini_does_not_depend_on_the_order_of_ties() -> None:
    rng = np.random.default_rng(8)
    y = rng.poisson(0.4, 2_000).astype(float)
    exposure = rng.uniform(0.1, 1.0, 2_000)
    prediction = np.round(rng.random(2_000), 1)  # only 11 distinct values, so almost every row is tied
    shuffled = rng.permutation(2_000)
    original = lorenz_gini(y, prediction, exposure)
    assert lorenz_gini(y[shuffled], prediction[shuffled], exposure[shuffled]) == pytest.approx(original, abs=1e-12)
    assert lorenz_gini(y[::-1], prediction[::-1], exposure[::-1]) == pytest.approx(original, abs=1e-12)


def test_lorenz_gini_is_zero_when_every_prediction_is_tied() -> None:
    assert lorenz_gini([0, 3, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_lorenz_gini_near_zero_for_random_ranking() -> None:
    rng = np.random.default_rng(4)
    y = rng.poisson(0.5, 20_000).astype(float)
    random_prediction = rng.random(20_000)
    assert abs(lorenz_gini(y, random_prediction)) < 0.03


def test_metrics_reject_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        brier_score([0, 1], [0.2, float("nan")])
