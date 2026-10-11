"""Run the retrieval evaluation: every question, every mode, with timing; then summarize with bootstrap CIs.

scripts/eval_rag.py does the I/O (loading models and the index, writing results/rag_metrics.json); this module
holds the logic so it can be unit-tested with fake embedders on an in-memory Qdrant.

Latency is wall-clock time of Retriever.search() for one query: embedding the query, the Qdrant query(ies) and,
for hybrid_rerank, the cross-encoder. One untimed warm-up query per mode is run first so model loading and
first-call setup are not counted.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from uwcopilot.rag.eval_set import EvalQuestion, gold_first_ranks
from uwcopilot.rag.metrics import (
    bootstrap_mean_ci,
    ndcg_at_k,
    paired_bootstrap_difference,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
)
from uwcopilot.rag.retriever import RetrievalMode, Retriever

EVAL_DEPTH = 10
RECALL_KS = (1, 3, 5)
WARMUP_QUERY = "warm-up query: what documents are required for an application?"


@dataclass(frozen=True)
class QuestionResult:
    """One question in one mode: what was retrieved, where the gold items ranked, and how long it took."""

    question_id: str
    category: str
    mode: str
    retrieved_ids: list[str]
    top_score: float | None
    gold_ranks: list[int | None]
    latency_ms: float

    def metrics(self) -> dict[str, float]:
        """recall@1/3/5, MRR@10 and nDCG@10 for an answerable question."""
        values = {f"recall@{k}": recall_at_k(self.gold_ranks, k) for k in RECALL_KS}
        values[f"mrr@{EVAL_DEPTH}"] = reciprocal_rank_at_k(self.gold_ranks, EVAL_DEPTH)
        values[f"ndcg@{EVAL_DEPTH}"] = ndcg_at_k(self.gold_ranks, EVAL_DEPTH)
        return values


def run_mode(retriever: Retriever, questions: Sequence[EvalQuestion], mode: RetrievalMode) -> list[QuestionResult]:
    """Search every question (answerable or not) in one mode and record ranks and latency."""
    retriever.search(WARMUP_QUERY, mode=mode, top_k=EVAL_DEPTH)
    results = []
    for question in questions:
        start = time.perf_counter()
        retrieved = retriever.search(question.question, mode=mode, top_k=EVAL_DEPTH)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        payloads = [item.payload for item in retrieved]
        results.append(
            QuestionResult(
                question_id=question.id,
                category=question.category,
                mode=mode,
                retrieved_ids=[item.chunk_id for item in retrieved],
                top_score=retrieved[0].score if retrieved else None,
                gold_ranks=gold_first_ranks(question, payloads) if question.answerable else [],
                latency_ms=elapsed_ms,
            )
        )
    return results


def summarize_mode(
    results: Sequence[QuestionResult],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Metrics with bootstrap 95% CIs over answerable questions, per-category means, and latency percentiles."""
    answerable = [result for result in results if result.gold_ranks]
    per_question = [result.metrics() for result in answerable]
    metric_names = list(per_question[0]) if per_question else []
    overall = {
        name: rounded(bootstrap_mean_ci([values[name] for values in per_question], n_resamples, seed))
        for name in metric_names
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({result.category for result in answerable}):
        rows = [result.metrics() for result in answerable if result.category == category]
        by_category[category] = {"n": len(rows)}
        by_category[category].update(
            {name: round(sum(row[name] for row in rows) / len(rows), 4) for name in metric_names}
        )
    latencies = [result.latency_ms for result in results]
    unanswerable_scores = [r.top_score for r in results if not r.gold_ranks and r.top_score is not None]
    answerable_scores = [r.top_score for r in answerable if r.top_score is not None]
    return {
        "n_answerable": len(answerable),
        "metrics": overall,
        "by_category": by_category,
        "latency_ms": {
            "n_queries": len(latencies),
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
        },
        "top1_score_median": {
            "answerable": round(percentile(answerable_scores, 50), 4) if answerable_scores else None,
            "unanswerable": round(percentile(unanswerable_scores, 50), 4) if unanswerable_scores else None,
        },
    }


def paired_comparison(
    results_a: Sequence[QuestionResult],
    results_b: Sequence[QuestionResult],
    metric: str = "recall@5",
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of (mode a - mode b) for one metric over the answerable questions."""
    by_id_b = {result.question_id: result for result in results_b if result.gold_ranks}
    pairs = [(result, by_id_b[result.question_id]) for result in results_a if result.question_id in by_id_b]
    values_a = [a.metrics()[metric] for a, _ in pairs]
    values_b = [b.metrics()[metric] for _, b in pairs]
    summary = paired_bootstrap_difference(values_a, values_b, n_resamples, seed)
    return {
        "metric": metric,
        "mode_a": results_a[0].mode if results_a else None,
        "mode_b": results_b[0].mode if results_b else None,
        "n_questions": len(pairs),
        "mean_difference": round(summary["mean_difference"], 4),
        "ci95": [round(summary["ci_low"], 4), round(summary["ci_high"], 4)],
        "share_resamples_at_or_below_zero": round(summary["share_resamples_at_or_below_zero"], 4),
        "questions_a_better": sum(1 for a, b in zip(values_a, values_b, strict=True) if a > b),
        "questions_b_better": sum(1 for a, b in zip(values_a, values_b, strict=True) if b > a),
    }


def rounded(summary: dict[str, float]) -> dict[str, Any]:
    """{"mean": m, "ci95": [low, high]} rounded to 4 decimals for the results file."""
    return {"mean": round(summary["mean"], 4), "ci95": [round(summary["ci_low"], 4), round(summary["ci_high"], 4)]}
