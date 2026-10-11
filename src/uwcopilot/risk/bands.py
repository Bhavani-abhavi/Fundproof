"""Risk bands for the scoring API: cut-offs are quantiles of the model's own scores on its training data.

A risk band turns a prediction (a probability of default, claims per year, or an average claim amount)
into a coarse label a reviewer can read at a glance. The cut-offs are never typed by hand: the training
scripts compute them from the chosen model's predictions on the training split and store them in the model
metadata (models/<product>/metadata.json) and in results/risk_metrics.json.

The quantile levels below are a design choice, documented here and in the model cards:

    low        below the 50th percentile of training scores
    medium     50th to 80th percentile
    high       80th to 95th percentile
    very high  95th percentile and above

So "high" means "riskier than 80% of the training portfolio by this model's own score". Bands describe
relative position, not an approval or decline rule: a decision always needs a person (ground rule 8).
"""

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

BAND_QUANTILES: tuple[float, ...] = (0.50, 0.80, 0.95)
BAND_LABELS: tuple[str, ...] = ("low", "medium", "high", "very high")
BAND_METHOD = "quantiles of the model's predictions on its training split"


def band_cutoffs(
    scores: Sequence[float] | np.ndarray,
    score_name: str,
    population: str,
    quantiles: Sequence[float] = BAND_QUANTILES,
    labels: Sequence[str] = BAND_LABELS,
) -> dict[str, Any]:
    """Compute band cut-offs as quantiles of training scores, plus the description stored with them.

    score_name says what the score is (for example "probability of default"); population says which rows
    the quantiles were taken over. Returns a JSON-ready dict that assign_band understands.
    """
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        raise ValueError("cannot compute risk bands from an empty set of scores")
    if np.isnan(values).any():
        raise ValueError("scores used for risk bands contain NaN")
    if len(labels) != len(quantiles) + 1:
        raise ValueError("there must be exactly one more label than quantile")
    if list(quantiles) != sorted(quantiles) or not all(0 < q < 1 for q in quantiles):
        raise ValueError("quantiles must be increasing and strictly between 0 and 1")
    cutoffs = [float(np.quantile(values, q)) for q in quantiles]
    return {
        "method": BAND_METHOD,
        "score": score_name,
        "population": population,
        "n_scores": int(values.size),
        "quantiles": [float(q) for q in quantiles],
        "cutoffs": cutoffs,
        "labels": list(labels),
        "rule": "a score equal to a cut-off goes to the higher band",
    }


def assign_band(score: float, bands: Mapping[str, Any]) -> str:
    """Label for one score, using cut-offs made by band_cutoffs (a score equal to a cut-off goes up a band)."""
    cutoffs = list(bands["cutoffs"])
    labels = list(bands["labels"])
    if len(labels) != len(cutoffs) + 1:
        raise ValueError("risk bands need exactly one more label than cut-off")
    if np.isnan(score):
        raise ValueError("cannot assign a risk band to NaN")
    return labels[bisect_right(cutoffs, float(score))]


def describe_bands(bands: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per band with its lower and upper cut-off (None at the open ends), for documents and /health."""
    cutoffs = list(bands["cutoffs"])
    quantiles = list(bands["quantiles"])
    rows = []
    for index, label in enumerate(bands["labels"]):
        rows.append(
            {
                "band": label,
                "from_quantile": quantiles[index - 1] if index > 0 else None,
                "to_quantile": quantiles[index] if index < len(quantiles) else None,
                "lower_cutoff": cutoffs[index - 1] if index > 0 else None,
                "upper_cutoff": cutoffs[index] if index < len(cutoffs) else None,
            }
        )
    return rows
