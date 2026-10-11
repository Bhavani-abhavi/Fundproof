"""Evaluate sanctions screening on the SYNTHETIC KYC customers (Phase 3).

Needs data/raw (OFAC + OpenSanctions, `make data`) and data/processed/kyc/customers.csv (`make kyc-data`).
Downloads sentence-transformers/all-MiniLM-L6-v2 (ONNX, via fastembed) into .cache/models on first run.

What it does, in order:
1. Loads the full watchlist (OFAC SDN + every OpenSanctions entry) and builds the blocking index.
2. Screens every customer (train, validation, test) once, keeping for each the rules-only score and the
   rules+reranker score for every reranker weight in RERANKER_WEIGHT_GRID, plus blocking statistics.
3. Chooses the reranker weight on the TRAIN split (best recall at the primary alert budget).
4. Chooses thresholds on the VALIDATION split: for each alert budget and each method, the lowest threshold
   whose alert rate stays within the budget (this maximizes recall subject to the budget).
5. Applies those thresholds to the TEST split and reports recall, alerts per 1,000, precision, hard-negative
   false-positive rate and blocking recall with 1,000-resample bootstrap intervals (seed 42), plus a paired
   bootstrap of the recall difference between the two methods.
6. Screens the exact-name homonyms (data/processed/kyc/homonyms.csv: made-up customers who share a listed name;
   never used to choose anything) and reports their alert rate at every operating point as a separate stratum.
7. Writes results/kyc_metrics.json, results/figures/kyc_recall_vs_alerts.png, and per-customer scores to
   data/processed/kyc/screening_scores.csv (gitignored).

Usage:
    uv run python scripts/eval_kyc.py
"""

import argparse
import dataclasses
import json
import os
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uwcopilot.common.files import sha256_of_file
from uwcopilot.common.paths import RESULTS_DIR
from uwcopilot.kyc.evaluation import (
    BOOTSTRAP_SEED,
    N_RESAMPLES,
    TRUE_MATCH,
    bootstrap_mean,
    bootstrap_screening_metrics,
    choose_threshold,
    paired_recall_difference,
    recall_alert_curve,
)
from uwcopilot.kyc.matcher import MatcherConfig, SanctionsMatcher
from uwcopilot.kyc.reranker import DEFAULT_EMBEDDING_MODEL, FastEmbedEmbedder
from uwcopilot.kyc.synthetic import (
    CUSTOMERS_FILE,
    GENERATION_REPORT_FILE,
    HOMONYMS_FILE,
    KYC_DATA_DIR,
    SYNTHETIC_LABEL,
)
from uwcopilot.kyc.watchlist import OPENSANCTIONS_ATTRIBUTION, load_watchlist
from uwcopilot.tools.schemas import ScreeningRequest

METRICS_FILE = RESULTS_DIR / "kyc_metrics.json"
FIGURE_FILE = RESULTS_DIR / "figures" / "kyc_recall_vs_alerts.png"
SCORES_FILE = KYC_DATA_DIR / "screening_scores.csv"
HOMONYM_SCORES_FILE = KYC_DATA_DIR / "homonym_scores.csv"
HOMONYM_SPLIT = "homonym_stratum"
# Alert budgets (alerts per 1,000 screened customers), fixed before the test split was scored. The primary
# budget is used to pick the reranker weight and the production operating point.
ALERT_BUDGETS_PER_1000: tuple[int, ...] = (50, 100, 200)
PRIMARY_BUDGET_PER_1000 = 100
RERANKER_WEIGHT_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
UNCAPPED = 10**9
EMBEDDING_THREADS = 2  # the machine is shared; keep ONNX Runtime to two threads
RULES = "rules_only"
RERANK = "rules_plus_reranker"

# Chart colors: reference palette slots 1 and 2 (validated with the dataviz palette validator, light mode).
COLOR_RULES = "#2a78d6"
COLOR_RERANK = "#eb6834"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options (input and output locations)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--customers-file", type=Path, default=CUSTOMERS_FILE)
    parser.add_argument("--metrics-file", type=Path, default=METRICS_FILE)
    parser.add_argument("--figure-file", type=Path, default=FIGURE_FILE)
    parser.add_argument("--scores-file", type=Path, default=SCORES_FILE)
    parser.add_argument("--homonyms-file", type=Path, default=HOMONYMS_FILE)
    parser.add_argument("--homonym-scores-file", type=Path, default=HOMONYM_SCORES_FILE)
    return parser.parse_args(argv)


def weight_column(weight: float) -> str:
    """Column name holding the rules+reranker score for one reranker weight."""
    return f"score_rerank_w{weight:.1f}"


def source_checks(matcher: SanctionsMatcher, outcome, source_row: int) -> dict:
    """Did the true list entry survive each stage (blocking before/after the cap, pre-rank, top-k)?"""
    source_names = matcher.index.names_of_entry(source_row)
    in_blocking = bool(np.isin(source_names, outcome.blocking.name_ids).any())
    if outcome.blocking.capped:
        uncapped = matcher.index.candidates(outcome.query.tokens, max_candidates=UNCAPPED)
        in_uncapped = bool(np.isin(source_names, uncapped.name_ids).any())
    else:
        in_uncapped = in_blocking
    return {
        "source_in_blocking_before_cap": in_uncapped,
        "source_in_blocking": in_blocking,
        "source_in_prerank": bool(np.isin(source_names, outcome.prerank_name_ids).any()),
        "source_in_top_k": any(candidate.entry_row == source_row for candidate in outcome.candidates),
    }


def screen_customers(matcher: SanctionsMatcher, customers: pd.DataFrame) -> pd.DataFrame:
    """Screen every customer once and keep the scores and stage statistics needed for evaluation."""
    row_of_entry = {entry_id: row for row, entry_id in enumerate(matcher.entry_ids)}
    entries = matcher.watchlist.entries
    records = []
    start = time.perf_counter()
    for position, customer in enumerate(customers.itertuples(index=False), start=1):
        outcome = matcher.match(customer.name, customer.entity_type, customer.country or None)
        top = outcome.candidates[0] if outcome.candidates else None
        record = {
            "customer_id": customer.customer_id,
            "split": customer.split,
            "label": customer.label,
            "generation": customer.generation,
            "score_rules": outcome.best_score(0.0),
            "blocking_size_before_cap": outcome.blocking.size_before_cap,
            "blocking_size": int(len(outcome.blocking.name_ids)),
            "blocking_capped": outcome.blocking.capped,
            "prerank_size": int(len(outcome.prerank_name_ids)),
            "top_rules_entry_id": matcher.entry_ids[top.entry_row] if top else "",
            "top_rules_entry_source": entries["source"].iat[top.entry_row] if top else "",
            "top_rules_entry_list": entries["list_name"].iat[top.entry_row] if top else "",
            "seconds_rules": outcome.seconds_rules,
            "seconds_rerank": outcome.seconds_rerank,
        }
        for weight in RERANKER_WEIGHT_GRID:
            record[weight_column(weight)] = outcome.best_score(weight)
        if customer.label == TRUE_MATCH:
            record.update(source_checks(matcher, outcome, row_of_entry[customer.matched_list_entry_id]))
        records.append(record)
        if position % 500 == 0:
            elapsed = time.perf_counter() - start
            print(f"  screened {position:,}/{len(customers):,} customers ({elapsed:.0f}s)", flush=True)
    return pd.DataFrame(records)


def choose_reranker_weight(train: pd.DataFrame) -> tuple[float, list[dict]]:
    """Pick the weight with the best TRAIN recall at the primary budget (ties go to the smaller weight)."""
    is_match = (train["label"] == TRUE_MATCH).to_numpy()
    table = []
    for weight in RERANKER_WEIGHT_GRID:
        choice = choose_threshold(train[weight_column(weight)].to_numpy(), is_match, PRIMARY_BUDGET_PER_1000)
        table.append({"reranker_weight": weight, **choice.as_dict()})
    best = max(table, key=lambda row: (row["recall"], -row["reranker_weight"]))
    return best["reranker_weight"], table


def blocking_recall_on(rows: pd.DataFrame) -> dict:
    """Blocking and stage recall for the true matches in `rows`, with bootstrap intervals."""
    matches = rows[rows["label"] == TRUE_MATCH]
    result = {"true_matches": int(len(matches))}
    for column in ("source_in_blocking_before_cap", "source_in_blocking", "source_in_prerank", "source_in_top_k"):
        result[column] = bootstrap_mean(matches[column].astype(float).to_numpy())
    return result


def candidate_set_sizes(rows: pd.DataFrame) -> dict:
    """Distribution of candidate-set sizes per customer after blocking (before and after the cap)."""

    def describe(values: pd.Series) -> dict:
        """Mean, median, 95th percentile and max."""
        return {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "max": int(values.max()),
        }

    return {
        "customers": int(len(rows)),
        "before_cap": describe(rows["blocking_size_before_cap"]),
        "after_cap": describe(rows["blocking_size"]),
        "share_of_customers_capped": float(rows["blocking_capped"].mean()),
        "share_with_no_candidates": float((rows["blocking_size"] == 0).mean()),
    }


def operating_points(validation: pd.DataFrame, test: pd.DataFrame, columns: dict[str, str]) -> dict:
    """For each budget and method: threshold chosen on validation, test metrics with bootstrap intervals."""
    labels_test = test["label"].to_numpy()
    validation_match = (validation["label"] == TRUE_MATCH).to_numpy()
    points: dict = {}
    for budget in ALERT_BUDGETS_PER_1000:
        point: dict = {"alert_budget_per_1000": budget}
        alerted: dict[str, np.ndarray] = {}
        for method, column in columns.items():
            choice = choose_threshold(validation[column].to_numpy(), validation_match, budget)
            alerted[method] = test[column].to_numpy() >= choice.threshold
            point[method] = {
                "threshold_chosen_on_validation": choice.threshold,
                "validation_recall": choice.recall,
                "validation_alerts_per_1000": choice.alerts_per_1000,
                "test": bootstrap_screening_metrics(alerted[method], labels_test, N_RESAMPLES, BOOTSTRAP_SEED),
                "test_alert_count": int(alerted[method].sum()),
            }
        point["paired_recall_difference_reranker_minus_rules"] = paired_recall_difference(
            alerted[RULES], alerted[RERANK], labels_test, N_RESAMPLES, BOOTSTRAP_SEED
        )
        points[str(budget)] = point
    return points


def recall_by_noise(test: pd.DataFrame, column: str, threshold: float) -> dict:
    """Test recall for each noise operation (a true match counts under every operation applied to it)."""
    matches = test[test["label"] == TRUE_MATCH]
    table: dict[str, dict] = {}
    for operation in ("transliteration", "typo", "initials", "token_reorder", "honorific", "suffix_change", "none"):
        subset = matches[matches["generation"].str.split("+").map(lambda ops, op=operation: op in ops)]
        if len(subset):
            table[operation] = {
                "true_matches": int(len(subset)),
                "alerted": int((subset[column] >= threshold).sum()),
            }
    return table


def false_alert_sources(test: pd.DataFrame, column: str, threshold: float) -> dict:
    """Where the false alerts on clean test customers come from (list of the highest rules-scored entry)."""
    clean_alerts = test[(test["label"] == "clean") & (test[column] >= threshold)]
    by_source = clean_alerts["top_rules_entry_source"].value_counts().to_dict()
    first_dataset = clean_alerts["top_rules_entry_list"].str.replace(r"^OpenSanctions \(", "", regex=True)
    first_dataset = first_dataset.str.split(r";|,|\)").str[0].str.strip()
    return {
        "clean_customers_alerted": int(len(clean_alerts)),
        "by_source": {key: int(value) for key, value in by_source.items()},
        "top_first_listed_dataset": {key: int(value) for key, value in first_dataset.value_counts().head(8).items()},
        "note": "source and first dataset title of the highest rules-scored list entry for each alerted clean customer",
    }


def recall_change_points(curve: list[dict]) -> list[dict]:
    """The points where recall rises; together they describe the whole step curve compactly."""
    points = []
    previous_recall = -1.0
    for point in curve:
        if point["recall"] > previous_recall:
            points.append(point)
            previous_recall = point["recall"]
    return points


def runtime(scores: pd.DataFrame, load_seconds: float, index_seconds: float) -> dict:
    """Wall-clock seconds per 1,000 customers for each stage, plus one-off load and index-build times."""
    count = len(scores)
    rules_total = float(scores["seconds_rules"].sum())
    rerank_total = float(scores["seconds_rerank"].sum())
    return {
        "customers_screened": count,
        "seconds_per_1000_customers_rules_stage": 1000 * rules_total / count,
        "seconds_per_1000_customers_rerank_stage": 1000 * rerank_total / count,
        "seconds_per_1000_customers_total": 1000 * (rules_total + rerank_total) / count,
        "watchlist_load_seconds": load_seconds,
        "index_build_seconds": index_seconds,
        "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cpu_count": os.cpu_count(),
        "embedding_threads": EMBEDDING_THREADS,
        "note": (
            "single process, wall-clock on a shared 4-CPU machine with another job running; rules stage = "
            "normalize + block + pre-rank + rules score; rerank stage = embedding the top candidates (cached per "
            "distinct text, so repeated list names are embedded once)"
        ),
    }


def plot_curves(
    test: pd.DataFrame, columns: dict[str, str], markers: dict[str, dict], reranker_weight: float, path: Path
) -> None:
    """Recall vs alerts per 1,000 on the test split for both methods: full range and a zoom on the low end."""
    plt.switch_backend("Agg")  # render to a file; no display needed
    is_match = (test["label"] == TRUE_MATCH).to_numpy()
    styles = {
        RULES: (COLOR_RULES, "Rules only"),
        RERANK: (COLOR_RERANK, f"Rules + embedding reranker (weight {reranker_weight:.1f})"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE, sharey=True)
    for axis, x_max, title in zip(axes, (1000, 300), ("Full range", "Zoom: 0-300 alerts per 1,000"), strict=True):
        axis.set_facecolor(SURFACE)
        for method, column in columns.items():
            color, label = styles[method]
            curve = recall_alert_curve(test[column].to_numpy(), is_match)
            x = [0.0] + [point["alerts_per_1000"] for point in curve]
            y = [0.0] + [point["recall"] for point in curve]
            axis.step(x, y, where="post", color=color, linewidth=2, label=label)
            marker = markers[method]
            axis.plot(marker["x"], marker["y"], "o", markersize=8, color=color, markeredgecolor=SURFACE, zorder=3)
        axis.axvline(PRIMARY_BUDGET_PER_1000, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        axis.set_xlim(0, x_max)
        axis.set_ylim(0, 1.02)
        axis.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
        axis.set_xlabel("Alerts per 1,000 screened customers", color=TEXT_SECONDARY)
        axis.grid(color=GRID, linewidth=0.8)
        axis.tick_params(colors=TEXT_SECONDARY)
        for spine in axis.spines.values():
            spine.set_color(GRID)
    axes[0].set_ylabel("Recall of true matches", color=TEXT_SECONDARY)
    axes[1].annotate(
        f"budget {PRIMARY_BUDGET_PER_1000}/1,000\n(dots: thresholds chosen\non validation)",
        xy=(PRIMARY_BUDGET_PER_1000, 0.05),
        xytext=(PRIMARY_BUDGET_PER_1000 + 8, 0.05),
        color=TEXT_SECONDARY,
        fontsize=8,
    )
    axes[0].legend(loc="lower right", frameon=False, labelcolor=TEXT_PRIMARY)
    figure.suptitle(
        "Sanctions screening on SYNTHETIC test customers: recall vs alert volume",
        color=TEXT_PRIMARY,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def homonym_stratum(
    homonym_scores: pd.DataFrame, points: dict, columns: dict[str, str], selected: dict, generation_report: dict
) -> dict:
    """Alert rate on the exact-name homonyms at every operating point (thresholds come from validation).

    These customers were never in train, validation or test, so nothing was tuned on them. A name-only screener
    should alert on them (a human then tells the namesake apart from the listed party), so this is the alert volume
    that the clean group's numbers leave out, not an error rate.
    """
    clean = generation_report["clean"]
    by_budget: dict = {}
    for budget, point in points.items():
        by_budget[budget] = {}
        for method, column in columns.items():
            threshold = point[method]["threshold_chosen_on_validation"]
            alerted = (homonym_scores[column] >= threshold).astype(float).to_numpy()
            by_budget[budget][method] = {
                "threshold_chosen_on_validation": threshold,
                "alerted": int(alerted.sum()),
                "alert_rate": bootstrap_mean(alerted),
            }
    selected_column = columns[selected["method"]]
    selected_alerted = (homonym_scores[selected_column] >= selected["threshold"]).astype(float).to_numpy()
    by_type = {}
    for entity_type, group in homonym_scores.groupby("entity_type"):
        group_alerted = (group[selected_column] >= selected["threshold"]).astype(float).to_numpy()
        by_type[str(entity_type)] = {"customers": int(len(group)), "alerted": int(group_alerted.sum())}
    return {
        "description": (
            "SYNTHETIC customers whose made-up name, after screening normalization, exactly equals a list name. "
            "The clean group excludes them; they are scored here as a separate stratum that was never used to "
            "choose a weight or threshold. Country is left blank for all of them."
        ),
        "customers": int(len(homonym_scores)),
        "clean_names_generated": clean["generated"],
        "clean_names_dropped_as_exact_list_names": clean["dropped_list_collisions"],
        "share_of_generated_clean_names_that_were_exact_list_names": clean["dropped_list_collisions"]
        / clean["generated"],
        "alert_rate_at_selected_point": bootstrap_mean(selected_alerted),
        "alerted_at_selected_point_by_entity_type": by_type,
        "alert_rate_by_budget": by_budget,
        "note": (
            "the share of exact list names depends on how Faker draws names and on the lists' size; it describes "
            "this generator, not a real customer base. The clean false-positive rate and alerts per 1,000 reported "
            "on the test split are therefore a lower bound for a population that includes namesakes."
        ),
    }


def example_alert(matcher: SanctionsMatcher, customers: pd.DataFrame, test: pd.DataFrame, config: MatcherConfig):
    """Screen an OFAC-derived organization test true match with the selected configuration (explanation format).

    Only organization names are used, so the committed results file names no listed individual (the agent track
    makes the same choice). Takes the most heavily altered such name that raises an alert; returns None if none
    does.
    """
    original_config = matcher.config
    matcher.config = config
    try:
        test_ids = set(test["customer_id"])
        pool = customers[
            customers["customer_id"].isin(test_ids)
            & (customers["label"] == TRUE_MATCH)
            & (customers["entity_type"] == "organization")
            & customers["matched_list_entry_id"].str.startswith("OFAC-")
        ].copy()
        # Prefer the most heavily altered names: their explanations show more than an exact match would.
        pool["noise_count"] = pool["generation"].str.split("+").map(len)
        pool = pool.sort_values(["noise_count", "customer_id"], ascending=[False, True])
        for customer in pool.itertuples(index=False):
            request = ScreeningRequest(
                name=customer.name, entity_type=customer.entity_type, country=customer.country or None
            )
            result = matcher.screen(request)
            if result.status == "potential_match":
                if any(not candidate.list_entry_id.startswith("OFAC-") for candidate in result.candidates):
                    continue  # keep the example to OFAC organization entries only
                return {
                    "customer_id": customer.customer_id,
                    "note": "SYNTHETIC customer derived from a public OFAC SDN organization entry",
                    "result": result.model_dump(),
                }
        return None
    finally:
        matcher.config = original_config


def main(argv: list[str] | None = None) -> int:
    """Run the whole evaluation and write the results file and the figure."""
    args = parse_args(argv)
    if not args.customers_file.exists():
        print(f"Missing {args.customers_file}. Run `make kyc-data` first.", file=sys.stderr)
        return 1
    customers = pd.read_csv(args.customers_file, keep_default_na=False, dtype=str)
    generation_report = json.loads(GENERATION_REPORT_FILE.read_text(encoding="utf-8"))

    start = time.perf_counter()
    watchlist = load_watchlist()
    load_seconds = time.perf_counter() - start
    snapshot_hashes = {snapshot.file_name: snapshot.sha256 for snapshot in watchlist.snapshots}
    generated_hashes = {item["file_name"]: item["sha256"] for item in generation_report["list_snapshots"]}
    if snapshot_hashes != generated_hashes:
        print("The list files changed since the customers were generated. Run `make kyc-data` again.", file=sys.stderr)
        return 1
    print(f"Loaded {len(watchlist.entries):,} list entries in {load_seconds:.0f}s")

    embedder = FastEmbedEmbedder(threads=EMBEDDING_THREADS)
    matcher = SanctionsMatcher(watchlist, MatcherConfig(), embedder)
    print(f"Built the blocking index ({matcher.index.name_count:,} names) in {matcher.index_build_seconds:.0f}s")

    scores = screen_customers(matcher, customers)
    args.scores_file.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.scores_file, index=False)
    homonym_scores: pd.DataFrame | None = None
    if args.homonyms_file.exists():
        homonyms = pd.read_csv(args.homonyms_file, keep_default_na=False, dtype=str)
        homonyms["split"] = HOMONYM_SPLIT
        print(f"Screening {len(homonyms):,} exact-name homonyms (separate stratum)")
        homonym_scores = screen_customers(matcher, homonyms)
        homonym_scores["entity_type"] = homonyms["entity_type"].to_numpy()
        homonym_scores.to_csv(args.homonym_scores_file, index=False)
    train, validation, test = (scores[scores["split"] == split] for split in ("train", "validation", "test"))

    weight, weight_table = choose_reranker_weight(train)
    columns = {RULES: "score_rules", RERANK: weight_column(weight)}
    points = operating_points(validation, test, columns)
    primary = points[str(PRIMARY_BUDGET_PER_1000)]
    rerank_wins = primary[RERANK]["validation_recall"] > primary[RULES]["validation_recall"]
    selected_method = RERANK if rerank_wins else RULES
    selected = {
        "method": selected_method,
        "reranker_weight": weight if rerank_wins else 0.0,
        "threshold": primary[selected_method]["threshold_chosen_on_validation"],
        "alert_budget_per_1000": PRIMARY_BUDGET_PER_1000,
        "rule": (
            "rules+reranker is selected only if its validation recall at the primary budget is strictly higher "
            "than rules-only; otherwise the simpler rules-only matcher is used"
        ),
        "embedding_model": DEFAULT_EMBEDDING_MODEL if rerank_wins else None,
    }
    selected_column = columns[selected_method]
    selected_config = dataclasses.replace(
        matcher.config, threshold=selected["threshold"], reranker_weight=selected["reranker_weight"]
    )

    markers = {
        method: {
            "x": points[str(PRIMARY_BUDGET_PER_1000)][method]["test"]["alerts_per_1000"]["value"],
            "y": points[str(PRIMARY_BUDGET_PER_1000)][method]["test"]["recall"]["value"],
        }
        for method in columns
    }
    plot_curves(test, columns, markers, weight, args.figure_file)

    is_match_test = (test["label"] == TRUE_MATCH).to_numpy()
    curves = {
        method: recall_change_points(recall_alert_curve(test[column].to_numpy(), is_match_test))
        for method, column in columns.items()
    }
    config = matcher.config
    metrics = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthetic_data": SYNTHETIC_LABEL,
        "description": (
            "Sanctions screening of SYNTHETIC customers against the OFAC SDN list and the full OpenSanctions "
            "default collection. Thresholds chosen on validation, reranker weight on train, metrics on test."
        ),
        "data": {
            "customers_file_sha256": sha256_of_file(args.customers_file),
            "counts_by_label_and_split": generation_report["counts_by_label_and_split"],
            "generation": {key: generation_report[key] for key in ("seed", "clean", "true_match", "hard_negative")},
            "list_snapshots": [snapshot.model_dump() for snapshot in watchlist.snapshots],
            "watchlist_scope": "full: every OFAC SDN entry and every OpenSanctions entry in targets.simple.csv",
            "watchlist_load_report": watchlist.load_report,
            "indexed_names": matcher.index.name_count,
            "vocabulary_size": matcher.index.vocabulary_size,
            "opensanctions_attribution": OPENSANCTIONS_ATTRIBUTION,
        },
        "matcher_config": {
            "blocking": dataclasses.asdict(config.blocking),
            "scoring_weights": dataclasses.asdict(config.weights),
            "prerank_top_names": config.prerank_top_names,
            "rerank_top_k": config.rerank_top_k,
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "embedding_runtime": "fastembed (ONNX Runtime), replaces a PyTorch sentence-transformers model",
        },
        "evaluation_protocol": {
            "alert_budgets_per_1000": list(ALERT_BUDGETS_PER_1000),
            "primary_budget_per_1000": PRIMARY_BUDGET_PER_1000,
            "threshold_objective": (
                "maximize recall of true matches subject to alerts per 1,000 customers <= budget; chosen on the "
                "validation split; alert when score >= threshold"
            ),
            "reranker_weight_grid": list(RERANKER_WEIGHT_GRID),
            "reranker_weight_selection": "best train recall at the primary budget (ties -> smaller weight)",
            "bootstrap": {"resamples": N_RESAMPLES, "seed": BOOTSTRAP_SEED, "interval": "95% percentile"},
            "customer_score": "highest final score over the reranked candidates (0.0 when blocking finds nothing)",
            "protocol_note": (
                "matcher settings, alert budgets and selection rules were fixed before the test split was first "
                "scored; the pipeline is deterministic, so re-running reproduces the same metrics (timings vary)"
            ),
        },
        "reranker_weight_selection_on_train": weight_table,
        "selected_reranker_weight": weight,
        "operating_points": points,
        "selected_operating_point": selected,
        "blocking": {
            "test": blocking_recall_on(test),
            "all_splits": blocking_recall_on(scores),
            "candidate_set_size_test": candidate_set_sizes(test),
            "candidate_set_size_all_splits": candidate_set_sizes(scores),
        },
        "test_recall_by_noise_type_at_selected_point": recall_by_noise(test, selected_column, selected["threshold"]),
        "test_false_alert_sources_at_selected_point": false_alert_sources(test, selected_column, selected["threshold"]),
        "exact_name_homonym_stratum": homonym_stratum(homonym_scores, points, columns, selected, generation_report)
        if homonym_scores is not None
        else "not measured (no homonyms file; run `make kyc-data`)",
        "runtime": runtime(scores, load_seconds, matcher.index_build_seconds),
        "test_curves_recall_change_points": curves,
        "figure": str(args.figure_file.relative_to(RESULTS_DIR.parent)),
        "example_alert": example_alert(matcher, customers, test, selected_config),
        "adverse_media_memo": {
            "status": "not measured (no LLM key)",
            "note": "the memo prompt builder and citation checks are unit-tested with a fake LLM only",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Reranker weight chosen on train: {weight}")
    for budget, point in points.items():
        for method in columns:
            test_metrics = point[method]["test"]
            print(
                f"  budget {budget}/1,000 {method}: threshold {point[method]['threshold_chosen_on_validation']:.4f}, "
                f"test recall {test_metrics['recall']['value']:.3f}, "
                f"alerts/1,000 {test_metrics['alerts_per_1000']['value']:.1f}"
            )
    print(f"Selected operating point: {selected['method']} at threshold {selected['threshold']:.4f}")
    print(f"Wrote {args.metrics_file} and {args.figure_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
