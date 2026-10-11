"""Population Stability Index (PSI): how much a distribution has shifted between two samples.

PSI = sum over bins of (actual_share - expected_share) * ln(actual_share / expected_share).

"expected" is the reference sample (for example the training vintage) and "actual" is the newer sample
(for example a later vintage or this month's applications). Common rule of thumb, used in the model cards:
PSI < 0.10 stable, 0.10 to 0.25 moderate shift, > 0.25 significant shift (flag for review).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_PSI_THRESHOLD = 0.25
MODERATE_PSI_THRESHOLD = 0.10
# A bin with no rows would make ln(share) undefined. Empty bins get this share instead (a common convention).
EMPTY_BIN_SHARE = 1e-4


def _shares_with_floor(counts: np.ndarray) -> np.ndarray:
    """Turn bin counts into shares of the total, replacing empty bins with EMPTY_BIN_SHARE."""
    shares = counts / counts.sum()
    return np.where(shares == 0, EMPTY_BIN_SHARE, shares)


def _psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    """PSI from two arrays of bin counts that use the same bins."""
    expected_shares = _shares_with_floor(expected_counts)
    actual_shares = _shares_with_floor(actual_counts)
    return float(np.sum((actual_shares - expected_shares) * np.log(actual_shares / expected_shares)))


def quantile_bin_edges(expected: np.ndarray, n_bins: int) -> np.ndarray:
    """Interior bin edges at the 1/n_bins, 2/n_bins, ... quantiles of the expected sample.

    Duplicate edges (from heavily tied values) are removed, so the result may give fewer than n_bins bins.
    The outer edges are open: values below the lowest edge go to the first bin, values above the highest
    edge go to the last bin, so new extreme values in "actual" are still counted.
    """
    probabilities = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(expected, probabilities)
    return np.unique(edges)


def population_stability_index(
    expected: Sequence[float] | np.ndarray | pd.Series,
    actual: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
) -> float:
    """PSI of a numeric variable, with quantile bins taken from the expected sample.

    A value equal to an edge goes to the bin above it. Missing values (NaN) are counted in their own extra
    "missing" bin, so a change in the missing rate also shows up in the PSI.
    """
    expected_values = np.asarray(expected, dtype=float)
    actual_values = np.asarray(actual, dtype=float)
    expected_missing = np.isnan(expected_values)
    actual_missing = np.isnan(actual_values)
    if expected_missing.all() or actual_missing.all():
        raise ValueError("expected and actual must each contain at least one non-missing value")

    edges = quantile_bin_edges(expected_values[~expected_missing], n_bins)
    n_value_bins = edges.shape[0] + 1
    expected_bins = np.searchsorted(edges, expected_values[~expected_missing], side="right")
    actual_bins = np.searchsorted(edges, actual_values[~actual_missing], side="right")
    expected_counts = np.bincount(expected_bins, minlength=n_value_bins).astype(float)
    actual_counts = np.bincount(actual_bins, minlength=n_value_bins).astype(float)

    if expected_missing.any() or actual_missing.any():
        expected_counts = np.r_[expected_counts, expected_missing.sum()]
        actual_counts = np.r_[actual_counts, actual_missing.sum()]
    return _psi_from_counts(expected_counts, actual_counts)


def categorical_population_stability_index(
    expected: Sequence[Any] | np.ndarray | pd.Series,
    actual: Sequence[Any] | np.ndarray | pd.Series,
) -> float:
    """PSI of a categorical variable: one bin per category seen in either sample.

    Missing values are treated as their own category ("<missing>").
    """
    expected_series = pd.Series(expected, dtype="object").fillna("<missing>").astype(str)
    actual_series = pd.Series(actual, dtype="object").fillna("<missing>").astype(str)
    categories = sorted(set(expected_series) | set(actual_series))
    expected_counts = expected_series.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    actual_counts = actual_series.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    return _psi_from_counts(expected_counts, actual_counts)


def psi_flag(value: float, threshold: float = DEFAULT_PSI_THRESHOLD) -> bool:
    """True when the PSI is above the threshold (default 0.25, a significant shift that needs review)."""
    return bool(value > threshold)


def psi_band(value: float) -> str:
    """Label a PSI value: 'stable' (< 0.10), 'moderate shift' (0.10 to 0.25) or 'significant shift' (> 0.25)."""
    if value < MODERATE_PSI_THRESHOLD:
        return "stable"
    if value <= DEFAULT_PSI_THRESHOLD:
        return "moderate shift"
    return "significant shift"
