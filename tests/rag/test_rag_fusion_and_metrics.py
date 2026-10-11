"""Tests for the RRF reference implementation and the retrieval metrics / bootstrap helpers."""

import math

import pytest

from uwcopilot.rag.fusion import reciprocal_rank_fusion
from uwcopilot.rag.metrics import (
    bootstrap_mean_ci,
    ndcg_at_k,
    paired_bootstrap_difference,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_rrf_matches_hand_computation() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["b"] == pytest.approx(1 / 61 + 1 / 60)
    assert fused["a"] == pytest.approx(1 / 60)
    assert fused["c"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 61)
    assert [item for item, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]], k=60)] == ["b", "a", "d", "c"]


def test_rrf_ties_keep_first_seen_order_and_k_is_validated() -> None:
    assert [item for item, _ in reciprocal_rank_fusion([["x"], ["y"]], k=2)] == ["x", "y"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["x"]], k=0)


def test_recall_mrr_ndcg_single_gold() -> None:
    assert recall_at_k([3], 1) == 0.0
    assert recall_at_k([3], 3) == 1.0
    assert reciprocal_rank_at_k([3], 10) == pytest.approx(1 / 3)
    assert reciprocal_rank_at_k([None], 10) == 0.0
    assert ndcg_at_k([1], 10) == 1.0
    assert ndcg_at_k([3], 10) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k([11], 10) == 0.0


def test_multi_gold_partial_recall_and_ndcg() -> None:
    ranks = [1, None]
    assert recall_at_k(ranks, 5) == 0.5
    assert reciprocal_rank_at_k(ranks, 10) == 1.0
    assert ndcg_at_k(ranks, 10) == pytest.approx(1.0 / (1.0 + 1.0 / math.log2(3)))
    assert ndcg_at_k([2, 1], 10) == pytest.approx(1.0)


def test_ndcg_never_exceeds_one_when_one_position_matches_two_gold_items() -> None:
    assert ndcg_at_k([1, 1], 10) <= 1.0


def test_metrics_need_gold() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], 5)
    with pytest.raises(ValueError):
        ndcg_at_k([], 5)


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = bootstrap_mean_ci(values, n_resamples=1000, seed=42)
    assert first == bootstrap_mean_ci(values, n_resamples=1000, seed=42)
    assert first["mean"] == pytest.approx(0.7)
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert 0.0 <= first["ci_low"] and first["ci_high"] <= 1.0
    constant = bootstrap_mean_ci([0.5] * 5)
    assert constant["ci_low"] == constant["ci_high"] == 0.5


def test_paired_bootstrap_difference_sign_and_shape() -> None:
    better = [1.0, 1.0, 1.0, 0.0, 1.0]
    worse = [0.0, 1.0, 0.0, 0.0, 0.0]
    result = paired_bootstrap_difference(better, worse, seed=42)
    assert result["mean_difference"] == pytest.approx(0.6)
    assert result["ci_low"] >= 0.0
    assert result["share_resamples_at_or_below_zero"] < 0.05
    with pytest.raises(ValueError):
        paired_bootstrap_difference([1.0], [1.0, 0.0])


def test_percentile_linear_interpolation() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([10.0], 95) == 10.0
    with pytest.raises(ValueError):
        percentile([], 50)
