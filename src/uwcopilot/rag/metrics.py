"""Retrieval metrics and bootstrap confidence intervals (pure functions, no I/O).

Per question, the evaluation records for each gold item the rank (1-based) of the first retrieved chunk that
matches it, or None if it was not retrieved in the top `depth` results. All metrics are computed from that list:

- recall@k: share of the question's gold items found in the top k.
- MRR@k (reciprocal rank): 1 / rank of the first relevant result, 0 if none in the top k.
- nDCG@k with binary relevance: a result position counts as relevant when it is the first match of at least one
  gold item. DCG = sum of 1/log2(rank + 1) over those positions; the ideal DCG puts min(#gold, k) relevant results
  at ranks 1, 2, .... One position that happens to match two gold items counts once, so nDCG stays <= 1.

Confidence intervals resample questions with replacement (percentile bootstrap). The paired bootstrap uses the
same resampled questions for both systems, so the difference is not inflated by question-to-question variation.
"""

import math
from collections.abc import Sequence

import numpy as np

GoldRanks = Sequence[int | None]


def recall_at_k(gold_ranks: GoldRanks, k: int) -> float:
    """Share of gold items whose first match is at rank <= k."""
    if not gold_ranks:
        raise ValueError("recall needs at least one gold item")
    found = sum(1 for rank in gold_ranks if rank is not None and rank <= k)
    return found / len(gold_ranks)


def reciprocal_rank_at_k(gold_ranks: GoldRanks, k: int) -> float:
    """1 / (rank of the first relevant result) if it is within k, else 0."""
    ranks = [rank for rank in gold_ranks if rank is not None and rank <= k]
    return 1.0 / min(ranks) if ranks else 0.0


def ndcg_at_k(gold_ranks: GoldRanks, k: int) -> float:
    """Binary-relevance nDCG@k from the first-match rank of each gold item."""
    if not gold_ranks:
        raise ValueError("nDCG needs at least one gold item")
    relevant_positions = sorted({rank for rank in gold_ranks if rank is not None and rank <= k})
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_positions)
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold_ranks), k) + 1))
    return dcg / ideal


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0-100) with linear interpolation (numpy's default)."""
    if not values:
        raise ValueError("percentile of an empty list")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def bootstrap_mean_ci(
    values: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Mean of values with a percentile bootstrap confidence interval (resampling the values with replacement)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("bootstrap of an empty list")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(n_resamples, data.size))
    means = data[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(data.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def paired_bootstrap_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Mean of (a - b) per question with a paired percentile bootstrap CI.

    share_resamples_at_or_below_zero is the fraction of resamples in which a did not beat b; it is a simple
    description of the bootstrap distribution, not a formal p-value.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("paired bootstrap needs two non-empty lists of the same length")
    differences = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_resamples, differences.size))
    resampled = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(resampled, alpha)),
        "ci_high": float(np.quantile(resampled, 1.0 - alpha)),
        "share_resamples_at_or_below_zero": float(np.mean(resampled <= 0.0)),
    }
