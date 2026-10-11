"""Fairness demonstration on UCI German Credit: fit the models, measure group fairness, write the results.

Writes the "fairness_demo" section of results/risk_metrics.json (other sections are left untouched) and
logs the run to the local MLflow store (mlruns.db). Prints the summary tables exactly as written to JSON.

Usage:
    uv run python scripts/fairness_german.py
    uv run python scripts/fairness_german.py --no-mlflow --n-resamples 1000

Needs data/raw/german_credit/german.data (run `make data` first). See uwcopilot.risk.fairness_demo for
the method and its limitations.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from uwcopilot.common.paths import RESULTS_DIR
from uwcopilot.risk.data.german_credit import GERMAN_DATA_PATH, load_german_credit
from uwcopilot.risk.fairness_demo import COMPARISONS, DemoConfig, run_fairness_demo
from uwcopilot.risk.results import CHECKSUM_LOG, read_results, update_results

RESULTS_PATH = RESULTS_DIR / "risk_metrics.json"
SECTION = "fairness_demo"
EXPERIMENT = "fairness_demo_german_credit"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=Path, default=GERMAN_DATA_PATH, help="path to german.data")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH, help="results JSON to update")
    parser.add_argument("--checksum-log", type=Path, default=CHECKSUM_LOG, help="data/raw/checksums.jsonl")
    parser.add_argument("--n-resamples", type=int, default=1000, help="bootstrap resamples (ground rule: 1000)")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the run to MLflow")
    return parser.parse_args(argv)


def mlflow_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Headline numbers to log to MLflow: AUC per model, and AIR and equal opportunity per comparison."""
    metrics: dict[str, float] = {}
    for model_name, model in payload["models"].items():
        metrics[f"{model_name}.test_auc"] = model["test_auc"]["estimate"]
        metrics[f"{model_name}.test_approval_rate"] = model["test_approval_rate"]["estimate"]
        for key, result in model["fairness"].items():
            metrics[f"{model_name}.{key}.adverse_impact_ratio"] = result["adverse_impact_ratio"]["estimate"]
            metrics[f"{model_name}.{key}.equal_opportunity_difference"] = result["equal_opportunity_difference"][
                "estimate"
            ]
    for key, result in payload["proxy_check"].items():
        auc = result["test_auc"]
        if "estimate" in auc:
            metrics[f"proxy_auc.{key}"] = auc["estimate"]
    return metrics


def log_to_mlflow(payload: dict[str, Any], config: DemoConfig) -> dict[str, str]:
    """Log parameters, headline metrics and the full payload to the local MLflow store. Returns run info."""
    # Imported here so --no-mlflow never touches MLflow. tracking turns MLflow telemetry off before importing
    # mlflow, so mlflow is used through tracking.mlflow rather than imported again in this file.
    from uwcopilot.risk import tracking

    tags = {"dataset": "UCI German Credit", "purpose": "fairness demonstration", "phase": "1C"}
    with tracking.start_run(EXPERIMENT, run_name="fairness_german", tags=tags) as run:
        tracking.log_params({**asdict(config), "lightgbm_n_estimators": _lightgbm_rounds(payload)})
        tracking.log_metrics(mlflow_metrics(payload))
        tracking.mlflow.log_dict(payload, "fairness_demo.json")
        return {"experiment": EXPERIMENT, "run_id": run.info.run_id, "store": "mlruns.db (local SQLite)"}


def _lightgbm_rounds(payload: dict[str, Any]) -> int:
    """Number of LightGBM trees chosen by cross-validation, read from the payload."""
    return int(payload["models"]["lightgbm"]["settings"]["n_estimators_chosen_by_cv"])


def format_value(value: Any) -> str:
    """Numbers to 3 decimals, everything else as text (for the printed table only; JSON keeps full precision)."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print a list of row dicts as an aligned text table."""
    print(f"\n{title}")
    if not rows:
        print("  (no rows)")
        return
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    cells = [[format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [max(len(column), *(len(line[i]) for line in cells)) for i, column in enumerate(columns)]
    print("  " + "  ".join(column.ljust(width) for column, width in zip(columns, widths, strict=True)))
    for line in cells:
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)))


def main(argv: list[str] | None = None) -> int:
    """Run the demo, write results, and print the tables read back from the results file."""
    args = parse_args(argv)
    config = DemoConfig(n_resamples=args.n_resamples)
    frame = load_german_credit(args.data_path)
    payload = run_fairness_demo(frame, config)
    payload["mlflow"] = (
        {"status": "not logged", "reason": "--no-mlflow was passed"}
        if args.no_mlflow
        else log_to_mlflow(payload, config)
    )
    update_results(args.results_path, SECTION, payload, datasets=["german_credit"], checksum_log=args.checksum_log)

    written = read_results(args.results_path)[SECTION]["results"]
    print(f"Wrote section '{SECTION}' to {args.results_path}")
    print(f"note: {written['note']}")
    print(
        f"test rows: {written['split']['n_test']}; bootstrap: {written['bootstrap']['n_resamples']} resamples, "
        f"seed {written['bootstrap']['seed']}"
    )
    print_table("Models (test set; AUC of P(bad); 95% bootstrap CI)", written["summary_table"]["models"])
    print_table(
        "Fairness (test set; AIR = unprivileged / privileged approval rate)", written["summary_table"]["fairness"]
    )
    print_table(
        "Proxy check (test AUC of predicting the unprivileged group from model features)",
        written["summary_table"]["proxy_check"],
    )
    print(f"\ncomparisons: {', '.join(f'{c.key} ({c.role})' for c in COMPARISONS)}")
    print(f"data quality warning: {written['data_quality_warning']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
