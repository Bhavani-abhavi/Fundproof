"""Model evaluation metrics for credit (binary) and insurance (count and amount) models.

Conventions used in every function:
- Inputs are array-like (lists, numpy arrays, or pandas Series). They are converted with np.asarray.
- For binary metrics, y_true holds 0/1 labels and the score is "higher means more likely to be 1"
  (for credit: higher score = higher probability of default).
- sample_weight, where supported, is a non-negative weight per row (for insurance: policy exposure in years).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.special import xlogy

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert values to a 1-D float array and reject NaN, so a metric never silently ignores rows."""
    array = np.asarray(values, dtype=float).ravel()
    if np.isnan(array).any():
        raise ValueError(f"{name} contains NaN values")
    return array


def _weights_or_ones(sample_weight: ArrayLike | None, n_rows: int) -> np.ndarray:
    """Return sample_weight as a float array, or an array of ones when no weights are given."""
    if sample_weight is None:
        return np.ones(n_rows)
    weights = _as_float_array(sample_weight, "sample_weight")
    if weights.shape[0] != n_rows:
        raise ValueError(f"sample_weight has {weights.shape[0]} rows, expected {n_rows}")
    if (weights < 0).any():
        raise ValueError("sample_weight must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("sample_weight must have a positive sum")
    return weights


def _binary_labels(y_true: ArrayLike) -> np.ndarray:
    """Return y_true as a 0/1 integer array, checking that both classes are present."""
    labels = _as_float_array(y_true, "y_true")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if labels.min() == labels.max():
        raise ValueError("y_true must contain both classes (0 and 1)")
    return labels.astype(int)


# ---------------------------------------------------------------------------
# Binary classification (credit default)
# ---------------------------------------------------------------------------


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values from 1 (smallest) to n; tied values all get the average of their ranks.

    Gives the same result as scipy.stats.rankdata(values) but runs several times faster, which matters
    when a metric is recomputed 1,000 times in a bootstrap.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    # A "block" is a run of equal values in sorted order. Block k covers sorted positions start[k] to end[k] - 1,
    # which are ranks start[k] + 1 to end[k]; their average is (start[k] + 1 + end[k]) / 2.
    starts_new_block = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    block_of_position = np.cumsum(starts_new_block) - 1
    block_start = np.flatnonzero(starts_new_block)
    block_end = np.r_[block_start[1:], values.shape[0]]
    average_rank_of_block = (block_start + 1 + block_end) / 2
    ranks = np.empty(values.shape[0])
    ranks[order] = average_rank_of_block[block_of_position]
    return ranks


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the ROC curve: the chance that a random positive scores above a random negative.

    Computed with the rank-sum (Mann-Whitney U) formula. Tied scores get their average rank, which counts a
    tie between a positive and a negative as half a win, the same as sklearn.metrics.roc_auc_score.
    """
    labels = _binary_labels(y_true)
    scores = _as_float_array(y_score, "y_score")
    ranks = _average_ranks(scores)
    n_pos = labels.sum()
    n_neg = labels.shape[0] - n_pos
    rank_sum_pos = ranks[labels == 1].sum()
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u_statistic / (n_pos * n_neg))


def ks_statistic(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Kolmogorov-Smirnov statistic: the largest gap between the score CDFs of positives and negatives.

    Equal to max over thresholds of |TPR - FPR|. Ranges from 0 (no separation) to 1 (perfect separation).
    """
    labels = _binary_labels(y_true)
    scores = _as_float_array(y_score, "y_score")
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    # Cumulative positives / negatives at or above each threshold. Only evaluate at the last row of each block
    # of tied scores, because a threshold cannot split tied scores (so their order inside the block is irrelevant).
    cum_pos = np.cumsum(sorted_labels)
    cum_neg = np.cumsum(1 - sorted_labels)
    is_threshold_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tpr = cum_pos[is_threshold_end] / cum_pos[-1]
    fpr = cum_neg[is_threshold_end] / cum_neg[-1]
    return float(np.max(np.abs(tpr - fpr)))


def gini_from_auc(auc: float) -> float:
    """Gini coefficient (also called accuracy ratio) for a binary model: 2 * AUC - 1."""
    return 2.0 * auc - 1.0


def brier_score(y_true: ArrayLike, y_prob: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Mean squared difference between predicted probability and the 0/1 outcome. Lower is better."""
    labels = _as_float_array(y_true, "y_true")
    probs = _as_float_array(y_prob, "y_prob")
    weights = _weights_or_ones(sample_weight, labels.shape[0])
    return float(np.average((probs - labels) ** 2, weights=weights))


def _bin_index_equal_width(y_prob: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign probabilities in [0, 1] to n_bins equal-width bins. A probability of exactly 1.0 goes to the last bin."""
    bins = np.floor(y_prob * n_bins).astype(int)
    return np.clip(bins, 0, n_bins - 1)


def _bin_index_equal_weight(values: np.ndarray, weights: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign rows to n_bins groups of (roughly) equal total weight, ordered by value.

    Rows are sorted by value (stable sort, so ties keep their input order). A row goes to the bin that
    contains the cumulative weight *before* it. With equal weights this gives equal-count bins (deciles
    for n_bins=10). Tied values can be split across two neighbouring bins.
    """
    order = np.argsort(values, kind="mergesort")
    sorted_weights = weights[order]
    weight_before = np.cumsum(sorted_weights) - sorted_weights
    share_before = weight_before / sorted_weights.sum()
    sorted_bins = np.minimum(np.floor(share_before * n_bins).astype(int), n_bins - 1)
    bins = np.empty_like(sorted_bins)
    bins[order] = sorted_bins
    return bins


def expected_calibration_error(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Expected calibration error: the bin-size-weighted average of |mean predicted - observed rate|.

    strategy="uniform" (default) uses n_bins equal-width bins on [0, 1]. Empty bins are skipped.
    strategy="quantile" uses n_bins equal-count bins ordered by predicted probability.
    """
    labels = _as_float_array(y_true, "y_true")
    probs = _as_float_array(y_prob, "y_prob")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("y_prob must be between 0 and 1")
    if strategy == "uniform":
        bins = _bin_index_equal_width(probs, n_bins)
    elif strategy == "quantile":
        bins = _bin_index_equal_weight(probs, np.ones_like(probs), n_bins)
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    n_rows = labels.shape[0]
    error = 0.0
    for bin_id in range(n_bins):
        in_bin = bins == bin_id
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        gap = abs(probs[in_bin].mean() - labels[in_bin].mean())
        error += (n_in_bin / n_rows) * gap
    return float(error)


def calibration_table(
    y_true: ArrayLike, y_pred: ArrayLike, n_bins: int = 10, sample_weight: ArrayLike | None = None
) -> pd.DataFrame:
    """Mean predicted vs observed value by predicted-value bin (deciles for n_bins=10).

    Bins hold roughly equal total weight (equal counts when sample_weight is None), ordered from the lowest
    prediction (bin 1) to the highest (bin n_bins). Works for probabilities (credit) and for frequencies
    (insurance: pass y_true = claims / exposure and sample_weight = exposure).

    Returns one row per non-empty bin with columns:
    bin, n_rows, total_weight, mean_predicted, mean_observed, min_predicted, max_predicted.
    Means are weighted by sample_weight.
    """
    observed = _as_float_array(y_true, "y_true")
    predicted = _as_float_array(y_pred, "y_pred")
    weights = _weights_or_ones(sample_weight, observed.shape[0])
    bins = _bin_index_equal_weight(predicted, weights, n_bins)

    rows = []
    for bin_id in range(n_bins):
        in_bin = bins == bin_id
        if not in_bin.any():
            continue
        bin_weights = weights[in_bin]
        rows.append(
            {
                "bin": bin_id + 1,
                "n_rows": int(in_bin.sum()),
                "total_weight": float(bin_weights.sum()),
                "mean_predicted": float(np.average(predicted[in_bin], weights=bin_weights)),
                "mean_observed": float(np.average(observed[in_bin], weights=bin_weights)),
                "min_predicted": float(predicted[in_bin].min()),
                "max_predicted": float(predicted[in_bin].max()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Insurance: claim frequency (Poisson) and claim severity (Gamma)
# ---------------------------------------------------------------------------


def mean_poisson_deviance(y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean Poisson deviance: 2 * (y * log(y / mu) - (y - mu)). Lower is better.

    y_true must be >= 0 and y_pred must be > 0. For claim frequency, pass y_true = claims / exposure,
    y_pred = predicted claims per year, and sample_weight = exposure (as in the scikit-learn freMTPL2 example).
    The y * log(y / mu) term is taken as 0 when y = 0.
    """
    observed = _as_float_array(y_true, "y_true")
    predicted = _as_float_array(y_pred, "y_pred")
    if (observed < 0).any():
        raise ValueError("y_true must be non-negative for Poisson deviance")
    if (predicted <= 0).any():
        raise ValueError("y_pred must be strictly positive for Poisson deviance")
    weights = _weights_or_ones(sample_weight, observed.shape[0])
    unit_deviance = 2.0 * (xlogy(observed, observed) - xlogy(observed, predicted) - observed + predicted)
    return float(np.average(unit_deviance, weights=weights))


def mean_gamma_deviance(y_true: ArrayLike, y_pred: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Weighted mean Gamma deviance: 2 * (log(mu / y) + y / mu - 1). Lower is better.

    Both y_true and y_pred must be > 0. For claim severity, pass y_true = average claim amount per policy,
    and sample_weight = number of claims behind that average.
    """
    observed = _as_float_array(y_true, "y_true")
    predicted = _as_float_array(y_pred, "y_pred")
    if (observed <= 0).any() or (predicted <= 0).any():
        raise ValueError("y_true and y_pred must be strictly positive for Gamma deviance")
    weights = _weights_or_ones(sample_weight, observed.shape[0])
    unit_deviance = 2.0 * (np.log(predicted / observed) + observed / predicted - 1.0)
    return float(np.average(unit_deviance, weights=weights))


def lorenz_curve(
    y_true: ArrayLike, y_pred: ArrayLike, exposure: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Lorenz curve for risk ranking, as in the scikit-learn "Poisson regression and non-normal loss" example.

    Policies are sorted from lowest to highest predicted risk. Rows with exactly the same prediction form one
    block (their exposure and claims are summed), so the curve runs in a straight line across a block of ties
    and the result does not depend on the input order of tied rows. Returns (cumulative share of exposure,
    cumulative share of observed claims), both starting at 0 and ending at 1, with one point per distinct
    prediction. y_true is the per-unit target (for example claims / exposure), so y_true * exposure is the
    claim count of each row. With exposure=None every row has exposure 1.
    """
    observed = _as_float_array(y_true, "y_true")
    predicted = _as_float_array(y_pred, "y_pred")
    weights = _weights_or_ones(exposure, observed.shape[0])
    if observed.shape[0] == 0 or predicted.shape[0] != observed.shape[0]:
        raise ValueError("Lorenz curve needs y_true and y_pred of the same, non-zero length")
    ranking = np.argsort(predicted, kind="mergesort")
    sorted_predictions = predicted[ranking]
    # Index of the first row of each block of equal predictions (the array is sorted, so ties are adjacent).
    block_starts = np.flatnonzero(np.r_[True, sorted_predictions[1:] != sorted_predictions[:-1]])
    ranked_claims = np.add.reduceat(observed[ranking] * weights[ranking], block_starts)
    ranked_exposure = np.add.reduceat(weights[ranking], block_starts)
    if ranked_claims.sum() <= 0:
        raise ValueError("Lorenz curve needs a positive total of y_true * exposure")
    cumulative_claims = np.r_[0.0, np.cumsum(ranked_claims)] / ranked_claims.sum()
    cumulative_exposure = np.r_[0.0, np.cumsum(ranked_exposure)] / ranked_exposure.sum()
    return cumulative_exposure, cumulative_claims


def lorenz_gini(y_true: ArrayLike, y_pred: ArrayLike, exposure: ArrayLike | None = None) -> float:
    """Gini index from the Lorenz curve: 1 - 2 * (area under the curve).

    0 means the model ranks no better than random; higher means better risk ranking (claims concentrate in
    the policies the model scores as riskiest). The area is computed with the trapezoidal rule.
    """
    cumulative_exposure, cumulative_claims = lorenz_curve(y_true, y_pred, exposure)
    area = np.trapezoid(cumulative_claims, cumulative_exposure)
    return float(1.0 - 2.0 * area)
