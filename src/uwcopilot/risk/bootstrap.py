"""Bootstrap confidence intervals for any metric (ground rule 1: 1,000 resamples, fixed seed).

How it works: draw n row indices with replacement, compute the metric on those rows, and repeat
n_resamples times. The confidence interval is the alpha/2 and 1 - alpha/2 percentiles of the resampled
metric values (the "percentile" method). The point estimate is the metric on the full, un-resampled data.

All arrays are resampled jointly: row i of every array stays together, so labels, predictions and weights
never get mixed up between rows.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05


def _to_equal_length_arrays(arrays: tuple[Any, ...]) -> list[np.ndarray]:
    """Convert every input to a numpy array and check they all have the same number of rows."""
    if not arrays:
        raise ValueError("pass at least one array to resample")
    converted = [np.asarray(array) for array in arrays]
    n_rows = converted[0].shape[0]
    for array in converted:
        if array.shape[0] != n_rows:
            raise ValueError("all arrays must have the same number of rows")
    if n_rows == 0:
        raise ValueError("arrays are empty")
    return converted


def _percentile_interval(values: np.ndarray, alpha: float) -> tuple[float, float, int]:
    """Return (low, high, n_valid) percentiles of the finite values. NaN results are left out and counted."""
    finite_values = values[np.isfinite(values)]
    if finite_values.shape[0] == 0:
        return float("nan"), float("nan"), 0
    low = np.percentile(finite_values, 100 * alpha / 2)
    high = np.percentile(finite_values, 100 * (1 - alpha / 2))
    return float(low), float(high), int(finite_values.shape[0])


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: Any,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval for metric_fn(*arrays).

    Example: bootstrap_ci(roc_auc, y_test, p_test) or, with weights,
    bootstrap_ci(lambda y, p, w: mean_poisson_deviance(y, p, sample_weight=w), y, p, exposure).

    Returns a dict with keys: estimate, ci_low, ci_high, n_resamples, n_valid_resamples, seed, alpha, method.
    n_valid_resamples counts resamples whose metric was a finite number (NaN results are left out of the CI).
    """
    data = _to_equal_length_arrays(arrays)
    n_rows = data[0].shape[0]
    rng = np.random.default_rng(seed)

    estimate = float(metric_fn(*data))
    resampled_values = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        resampled_values[i] = metric_fn(*[array[rows] for array in data])

    ci_low, ci_high, n_valid = _percentile_interval(resampled_values, alpha)
    return {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples": n_resamples,
        "n_valid_resamples": n_valid,
        "seed": seed,
        "alpha": alpha,
        "method": "percentile",
    }


def paired_bootstrap_diff(
    metric_fn: Callable[..., float],
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    *extra_arrays: Any,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired bootstrap of metric(model A) - metric(model B) on the same rows.

    Each resample uses one set of row indices for both models, so the interval reflects the difference
    between the models rather than the noise of the test set. metric_fn is called as
    metric_fn(y_true, pred, *extra_arrays), so extra arrays such as exposure weights are resampled too.

    Returns the same keys as bootstrap_ci (estimate is metric(A) - metric(B) on the full data), plus
    estimate_a, estimate_b and share_a_greater (share of valid resamples where the difference is > 0).
    If the CI excludes 0, the difference is unlikely to be resampling noise at level alpha.
    """
    data = _to_equal_length_arrays((y_true, pred_a, pred_b, *extra_arrays))
    labels, scores_a, scores_b = data[0], data[1], data[2]
    extras = data[3:]
    n_rows = labels.shape[0]
    rng = np.random.default_rng(seed)

    estimate_a = float(metric_fn(labels, scores_a, *extras))
    estimate_b = float(metric_fn(labels, scores_b, *extras))
    differences = np.empty(n_resamples)
    for i in range(n_resamples):
        rows = rng.integers(0, n_rows, size=n_rows)
        resampled_extras = [array[rows] for array in extras]
        value_a = metric_fn(labels[rows], scores_a[rows], *resampled_extras)
        value_b = metric_fn(labels[rows], scores_b[rows], *resampled_extras)
        differences[i] = value_a - value_b

    ci_low, ci_high, n_valid = _percentile_interval(differences, alpha)
    finite_differences = differences[np.isfinite(differences)]
    share_a_greater = float((finite_differences > 0).mean()) if n_valid > 0 else float("nan")
    return {
        "estimate": estimate_a - estimate_b,
        "estimate_a": estimate_a,
        "estimate_b": estimate_b,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "share_a_greater": share_a_greater,
        "n_resamples": n_resamples,
        "n_valid_resamples": n_valid,
        "seed": seed,
        "alpha": alpha,
        "method": "percentile",
    }
