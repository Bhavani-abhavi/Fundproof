"""Threshold choice, recall-vs-alerts curves and bootstrap confidence intervals for screening.

Every customer gets one score: the highest final score over its candidate list entries (0.0 when blocking found
nothing). A customer is alerted when score >= threshold.

Threshold objective (documented, fixed before looking at the test split): maximize recall of true matches
subject to an alert budget, i.e. at most `max_alerts_per_1000` alerts per 1,000 screened customers. Lowering the
threshold can only add alerts, so the best threshold is the lowest one whose alert rate is still within the
budget. Thresholds are chosen on the validation split and only applied to the test split.

Bootstrap: customers of the test split are resampled with replacement (1,000 resamples, seed 42) and each
metric is recomputed; the 2.5th and 97.5th percentiles form the 95% interval. For the rules-only vs
rules+reranker comparison, both methods are scored on the same resamples (a paired bootstrap).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

TRUE_MATCH = "true_match"
HARD_NEGATIVE = "hard_negative"
CLEAN = "clean"
N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen threshold and what it achieved on the split it was chosen on."""

    threshold: float
    recall: float
    alerts_per_1000: float
    max_alerts_per_1000: float
    objective: str = "maximize recall of true matches subject to alerts per 1,000 customers <= budget"

    def as_dict(self) -> dict:
        """Plain dictionary for JSON output."""
        return asdict(self)


def alerts_per_1000(alerted: np.ndarray) -> float:
    """Alerts per 1,000 screened customers."""
    return 1000.0 * float(np.mean(alerted)) if len(alerted) else float("nan")


def recall_of(alerted: np.ndarray, is_match: np.ndarray) -> float:
    """Share of true matches that were alerted (NaN when there are no true matches)."""
    return float(np.mean(alerted[is_match])) if is_match.any() else float("nan")


def choose_threshold(scores: np.ndarray, is_match: np.ndarray, max_alerts_per_1000: float) -> ThresholdChoice:
    """Lowest observed score whose alert rate (score >= threshold) stays within the budget.

    If even the single highest score breaks the budget (many customers tied at the top), the threshold is set
    just above the highest score, which means no alerts.
    """
    chosen = math.nextafter(float(np.max(scores)), math.inf) if len(scores) else math.inf
    for threshold in np.unique(scores)[::-1]:  # highest first
        if alerts_per_1000(scores >= threshold) <= max_alerts_per_1000:
            chosen = float(threshold)
        else:
            break  # lower thresholds only add alerts
    alerted = scores >= chosen
    return ThresholdChoice(
        threshold=chosen,
        recall=recall_of(alerted, is_match),
        alerts_per_1000=alerts_per_1000(alerted),
        max_alerts_per_1000=max_alerts_per_1000,
    )


def recall_alert_curve(scores: np.ndarray, is_match: np.ndarray) -> list[dict]:
    """Every distinct threshold with its alerts per 1,000 and recall (highest threshold first)."""
    points = []
    for threshold in np.unique(scores)[::-1]:
        alerted = scores >= threshold
        points.append(
            {
                "threshold": float(threshold),
                "alerts_per_1000": alerts_per_1000(alerted),
                "recall": recall_of(alerted, is_match),
            }
        )
    return points


def recall_at_budget(curve: list[dict], max_alerts_per_1000: float) -> float:
    """Best recall on a curve among points within the alert budget (0.0 if none)."""
    within = [point["recall"] for point in curve if point["alerts_per_1000"] <= max_alerts_per_1000]
    return max(within, default=0.0)


def screening_metrics(alerted: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Recall, alerts per 1,000, precision, hard-negative and clean false-positive rates."""
    is_match = labels == TRUE_MATCH
    is_hard_negative = labels == HARD_NEGATIVE
    is_clean = labels == CLEAN
    alert_count = int(alerted.sum())
    return {
        "recall": recall_of(alerted, is_match),
        "alerts_per_1000": alerts_per_1000(alerted),
        "precision": float(np.mean(is_match[alerted])) if alert_count else float("nan"),
        "hard_negative_false_positive_rate": float(np.mean(alerted[is_hard_negative]))
        if is_hard_negative.any()
        else float("nan"),
        "clean_false_positive_rate": float(np.mean(alerted[is_clean])) if is_clean.any() else float("nan"),
    }


def percentile_interval(values: list[float]) -> tuple[float, float]:
    """2.5th and 97.5th percentiles, ignoring NaN (a resample with no alerts has no precision)."""
    array = np.array(values, dtype=float)
    array = array[~np.isnan(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def bootstrap_indices(n: int, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Resampled row positions, shape (n_resamples, n). The same seed gives the same resamples."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_screening_metrics(
    alerted: np.ndarray, labels: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> dict[str, dict[str, float]]:
    """Point estimate and 95% bootstrap interval for every metric in screening_metrics()."""
    point = screening_metrics(alerted, labels)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for rows in bootstrap_indices(len(alerted), n_resamples, seed):
        for name, value in screening_metrics(alerted[rows], labels[rows]).items():
            samples[name].append(value)
    result = {}
    for name, value in point.items():
        low, high = percentile_interval(samples[name])
        result[name] = {"value": value, "ci95_low": low, "ci95_high": high}
    return result


def bootstrap_mean(values: np.ndarray, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict[str, float]:
    """Mean of a 0/1 or numeric array with its 95% bootstrap interval."""
    means = [float(np.mean(values[rows])) for rows in bootstrap_indices(len(values), n_resamples, seed)]
    low, high = percentile_interval(means)
    return {"value": float(np.mean(values)), "ci95_low": low, "ci95_high": high}


def paired_recall_difference(
    alerted_baseline: np.ndarray,
    alerted_challenger: np.ndarray,
    labels: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Recall(challenger) - recall(baseline) on the same customer resamples, with its 95% interval.

    share_of_resamples_challenger_better is the fraction of resamples where the challenger's recall is higher;
    it is a descriptive number, not a p-value.
    """
    is_match = labels == TRUE_MATCH
    point = recall_of(alerted_challenger, is_match) - recall_of(alerted_baseline, is_match)
    differences = []
    for rows in bootstrap_indices(len(labels), n_resamples, seed):
        resampled_match = is_match[rows]
        differences.append(
            recall_of(alerted_challenger[rows], resampled_match) - recall_of(alerted_baseline[rows], resampled_match)
        )
    low, high = percentile_interval(differences)
    valid = np.array([d for d in differences if not math.isnan(d)])
    return {
        "difference": point,
        "ci95_low": low,
        "ci95_high": high,
        "share_of_resamples_challenger_better": float(np.mean(valid > 0)) if len(valid) else float("nan"),
        "share_of_resamples_equal": float(np.mean(valid == 0)) if len(valid) else float("nan"),
    }
