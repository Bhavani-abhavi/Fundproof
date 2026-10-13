"""Metrics and bootstrap confidence intervals from the per-packet evaluation records of evaluation.py.

Every metric is a ratio of counts SUMMED over packets (micro-average): for example field F1 = 2 TP / (2 TP + FP +
FN) over every value of every packet. Confidence intervals are percentile bootstrap 95% intervals with packet-level
resampling (1,000 resamples, seed 42, the repo standard): each resample draws packets with replacement, sums their
counts and recomputes the metric, so values of one packet always move together. A resample in which a metric is
undefined (zero denominator) is skipped; n_resamples_used says how many were left.

Paired comparisons (clean vs scanned pages, text layer vs forced OCR, rules vs hybrid) put both conditions of the
same packet into one row, so every resample uses the same packets for both and the interval of the difference is
not widened by packet-to-packet variation.

Metric dicts use the keys value, ci_low, ci_high (rounded to 4 decimals), plus the counts behind them.
"""

from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from uwcopilot.agent.evaluation import latency_summary
from uwcopilot.docintel.contradictions import CONTRADICTION_TYPES
from uwcopilot.docintel.evaluation import CONDITIONS, FIELD_TYPES, TABLE_ROWS
from uwcopilot.docintel.pipeline import STAGES

N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
CountRow = dict[str, float]
Statistic = Callable[[CountRow], float | None]


def totals(rows: list[CountRow]) -> Counter:
    """The counts of all rows added up."""
    summed: Counter = Counter()
    for row in rows:
        summed.update(row)
    return summed


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator, or None when the denominator is zero."""
    return None if denominator == 0 else numerator / denominator


def f1_of(counts: CountRow, prefix: str) -> float | None:
    """F1 = 2 TP / (2 TP + FP + FN) for the counters prefix.tp / .fp / .fn (None when all are zero)."""
    tp = counts.get(f"{prefix}.tp", 0)
    return safe_ratio(2 * tp, 2 * tp + counts.get(f"{prefix}.fp", 0) + counts.get(f"{prefix}.fn", 0))


def precision_of(counts: CountRow, prefix: str) -> float | None:
    """TP / (TP + FP)."""
    tp = counts.get(f"{prefix}.tp", 0)
    return safe_ratio(tp, tp + counts.get(f"{prefix}.fp", 0))


def recall_of(counts: CountRow, prefix: str) -> float | None:
    """TP / (TP + FN)."""
    tp = counts.get(f"{prefix}.tp", 0)
    return safe_ratio(tp, tp + counts.get(f"{prefix}.fn", 0))


def bootstrap(rows: list[CountRow], statistic: Statistic, names: list[str]) -> dict[str, Any]:
    """The statistic on the summed rows, with a packet-level percentile bootstrap 95% interval.

    names are the counters the statistic reads; only they are resampled (the others do not change the result).
    """
    value = statistic(totals(rows))
    result: dict[str, Any] = {"value": None if value is None else round(value, 4), "ci_low": None, "ci_high": None}
    result["n_packets"] = len(rows)
    if value is None or not rows:
        result["n_resamples_used"] = 0
        return result
    matrix = np.array([[row.get(name, 0.0) for name in names] for row in rows], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(rows), size=(N_RESAMPLES, len(rows)))
    estimates = []
    for sums in matrix[indices].sum(axis=1):
        estimate = statistic(dict(zip(names, sums, strict=True)))
        if estimate is not None:
            estimates.append(estimate)
    result["ci_low"] = round(float(np.quantile(estimates, 0.025)), 4)
    result["ci_high"] = round(float(np.quantile(estimates, 0.975)), 4)
    result["n_resamples_used"] = len(estimates)
    return result


def counts_of(records: list[dict[str, Any]], prefix: str = "") -> list[CountRow]:
    """The counter dicts of the records, with every name prefixed (for paired comparisons)."""
    return [{prefix + name: float(value) for name, value in record["counts"].items()} for record in records]


def ratio_metric(records: list[dict[str, Any]], numerator: str, denominator: str) -> dict[str, Any]:
    """counts[numerator] / counts[denominator] with its bootstrap interval and the two summed counts."""
    rows = counts_of(records)
    summed = totals(rows)
    result = bootstrap(
        rows, lambda counts: safe_ratio(counts.get(numerator, 0), counts.get(denominator, 0)), [numerator, denominator]
    )
    result["numerator"] = int(summed.get(numerator, 0))
    result["denominator"] = int(summed.get(denominator, 0))
    return result


def prf_metric(records: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    """Precision, recall and F1 of prefix.tp / .fp / .fn, each with an interval, and the summed counts."""
    rows = counts_of(records)
    summed = totals(rows)
    names = [f"{prefix}.tp", f"{prefix}.fp", f"{prefix}.fn"]
    return {
        "precision": bootstrap(rows, lambda counts: precision_of(counts, prefix), names),
        "recall": bootstrap(rows, lambda counts: recall_of(counts, prefix), names),
        "f1": bootstrap(rows, lambda counts: f1_of(counts, prefix), names),
        "tp": int(summed.get(f"{prefix}.tp", 0)),
        "fp": int(summed.get(f"{prefix}.fp", 0)),
        "fn": int(summed.get(f"{prefix}.fn", 0)),
    }


def paired_f1_difference(
    first: list[dict[str, Any]], second: list[dict[str, Any]], first_prefix: str, second_prefix: str
) -> dict[str, Any]:
    """F1(first) - F1(second) on the same packets (records aligned by packet_id), with a paired interval."""
    by_id = {record["packet_id"]: record for record in second}
    paired = [record for record in first if record["packet_id"] in by_id]
    rows = []
    for record in paired:
        row = {f"a.{name}": float(value) for name, value in record["counts"].items()}
        row.update({f"b.{name}": float(value) for name, value in by_id[record["packet_id"]]["counts"].items()})
        rows.append(row)

    def difference(counts: CountRow) -> float | None:
        a_value = f1_of(counts, f"a.{first_prefix}")
        b_value = f1_of(counts, f"b.{second_prefix}")
        return None if a_value is None or b_value is None else a_value - b_value

    names = [
        f"{side}.{prefix}.{outcome}"
        for side, prefix in (("a", first_prefix), ("b", second_prefix))
        for outcome in ("tp", "fp", "fn")
    ]
    return bootstrap(rows, difference, names)


def field_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Field P/R/F1 overall, per field type, and per page condition (clean = text layer, scanned)."""
    result: dict[str, Any] = {"overall": prf_metric(records, "fields.all.all")}
    result["by_type"] = {kind: prf_metric(records, f"fields.all.{kind}") for kind in FIELD_TYPES}
    result["by_condition"] = {}
    for condition in CONDITIONS[1:]:
        result["by_condition"][condition] = {
            "overall": prf_metric(records, f"fields.{condition}.all"),
            "by_type": {kind: prf_metric(records, f"fields.{condition}.{kind}") for kind in FIELD_TYPES},
        }
    return result


def row_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Row-level P/R for transactions, claims and locations, and the share of found rows with every field right."""
    result = {}
    for name in TABLE_ROWS:
        metric = prf_metric(records, f"rows.{name}")
        metric["exact_share_of_found"] = ratio_metric(records, f"rows.{name}.exact", f"rows.{name}.tp")
        result[name] = metric
    return result


def contradiction_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Contradiction recall and precision overall and per type, and precision by packet condition."""
    result: dict[str, Any] = {
        "recall": ratio_metric(records, "problems.all.detected", "problems.all.total"),
        "precision": ratio_metric(records, "contradictions.all.tp", "contradictions.all.total"),
        "by_type": {},
    }
    for kind in CONTRADICTION_TYPES:
        result["by_type"][kind] = {
            "recall": ratio_metric(records, f"problems.{kind}.detected", f"problems.{kind}.total"),
            "precision": ratio_metric(records, f"contradictions.{kind}.tp", f"contradictions.{kind}.total"),
        }
    clean = [record for record in records if record["n_scanned_pages"] == 0]
    scanned = [record for record in records if record["n_scanned_pages"] > 0]
    result["precision_by_packet_condition"] = {
        "packets_without_scanned_pages": ratio_metric(clean, "contradictions.all.tp", "contradictions.all.total"),
        "packets_with_scanned_pages": ratio_metric(scanned, "contradictions.all.tp", "contradictions.all.total"),
    }
    return result


def explanation_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Benign-vs-unsupported accuracy on detected problems, the share explained "unknown", and per class."""
    result = {
        "accuracy": ratio_metric(records, "explanations.all.correct", "explanations.all.total"),
        "unknown_share": ratio_metric(records, "explanations.all.unknown", "explanations.all.total"),
        "by_class": {},
    }
    for group in ("benign", "unsupported"):
        result["by_class"][group] = {
            "accuracy": ratio_metric(records, f"explanations.{group}.correct", f"explanations.{group}.total"),
            "unknown_share": ratio_metric(records, f"explanations.{group}.unknown", f"explanations.{group}.total"),
        }
    return result


def injection_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Injection recall (overall and by technique) and per-packet false-positive rates."""
    techniques: dict[str, Counter] = {}
    for record in records:
        for outcome in record["injections"]:
            counter = techniques.setdefault(outcome["technique"], Counter())
            counter["total"] += 1
            counter["detected"] += int(outcome["detected"])
    return {
        "recall": ratio_metric(records, "injections.detected", "injections.total"),
        "by_technique": {name: dict(counter) for name, counter in sorted(techniques.items())},
        "per_packet_false_positive_rate": ratio_metric(
            records, "injection_packets.with_false_hit", "injection_packets.total"
        ),
        "clean_packet_false_positive_rate": ratio_metric(
            records, "injection_packets.clean_with_hit", "injection_packets.clean"
        ),
        "false_hits": int(totals(counts_of(records)).get("injection_hits.false", 0)),
    }


def latency_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """p50 / p95 milliseconds per stage and in total (agent.evaluation.latency_summary), plus OCR page counts."""
    result: dict[str, Any] = {"per_stage_ms": {}}
    for stage in (*STAGES, "total"):
        result["per_stage_ms"][stage] = latency_summary(
            [float(record["timings_ms"].get(stage, 0.0)) for record in records]
        )
    ocr_records = [record for record in records if record["ocr_pages"] > 0]
    per_page = [record["timings_ms"].get("ocr", 0.0) / record["ocr_pages"] for record in ocr_records]
    result["ocr_pages_total"] = sum(record["ocr_pages"] for record in records)
    result["packets_with_ocr"] = len(ocr_records)
    result["ocr_ms_per_page"] = latency_summary(per_page)
    return result


def cost_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM calls, tokens and dollars per packet; a cost that is None for any packet makes the total None."""
    costs = [record["usage"]["cost_usd"] for record in records]
    known = None not in costs
    total = sum(costs) if known and costs else (0.0 if not costs else None)
    return {
        "llm_calls": sum(record["usage"]["llm_calls"] for record in records),
        "input_tokens": sum(record["usage"]["input_tokens"] for record in records),
        "output_tokens": sum(record["usage"]["output_tokens"] for record in records),
        "usd_total": None if total is None else round(total, 6),
        "usd_per_packet_mean": None if total is None or not records else round(total / len(records), 6),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every metric of one pipeline run (one extractor and OCR mode) over its packets."""
    lines = Counter(record["line"] for record in records)
    return {
        "n_packets": len(records),
        "packets_by_line": dict(sorted(lines.items())),
        "fields": field_metrics(records),
        "fields_by_line": {
            line: prf_metric([record for record in records if record["line"] == line], "fields.all.all")
            for line in sorted(lines)
        },
        "rows": row_metrics(records),
        "citations": {
            "accuracy": ratio_metric(records, "citations.all.correct", "citations.all.total"),
            "cited_page_is_truth_page": ratio_metric(records, "citations.all.same_page", "citations.all.total"),
            "by_type": {
                kind: ratio_metric(records, f"citations.{kind}.correct", f"citations.{kind}.total")
                for kind in FIELD_TYPES
            },
        },
        "page_classification": {
            condition: ratio_metric(records, f"pages.{condition}.correct", f"pages.{condition}.total")
            for condition in CONDITIONS
        },
        "document_segmentation": ratio_metric(records, "documents.correct", "documents.total"),
        "checklist": {
            "code_accuracy": ratio_metric(records, "checklist.correct", "checklist.total"),
            "packet_exact_match": ratio_metric(records, "checklist.packets_exact", "checklist.packets"),
            "documents_provided_accuracy": ratio_metric(records, "provided.correct", "provided.total"),
        },
        "contradictions": contradiction_metrics(records),
        "benign_vs_unsupported": explanation_metrics(records),
        "injections": injection_metrics(records),
        "latency": latency_metrics(records),
        "cost": cost_metrics(records),
    }
