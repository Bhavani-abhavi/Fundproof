"""Train and evaluate the Lending Club credit-default models, then write the "credit" results section.

Writes the "credit" section of results/risk_metrics.json (other sections are left untouched), saves the
fitted models to models/credit/ (scorecard, LightGBM booster, metadata.json with feature ranges for
scenario overrides), and logs the run to the local MLflow store (mlruns.db), registering the chosen model.

If the Lending Club zip is missing (it needs Kaggle credentials), nothing is trained: the section is written
as "not measured" with the reason, and the script exits with code 0. No credit metric is ever estimated.

Usage:
    uv run python scripts/train_credit.py
    uv run python scripts/train_credit.py --no-mlflow --n-resamples 1000

See uwcopilot.risk.credit for the method and docs/leakage.md for every column decision.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from uwcopilot.common.paths import REPO_ROOT, RESULTS_DIR
from uwcopilot.risk.credit import (
    LIGHTGBM,
    LOGISTIC,
    CreditConfig,
    CreditModels,
    run_credit_pipeline,
    save_credit_models,
)
from uwcopilot.risk.data.lending_club import DEFAULT_CHUNK_ROWS, DEFAULT_ZIP_PATH, load_lending_club, zip_status
from uwcopilot.risk.results import CHECKSUM_LOG, git_commit, not_measured, read_results, update_results

RESULTS_PATH = RESULTS_DIR / "risk_metrics.json"
MODELS_DIR = REPO_ROOT / "models" / "credit"
SECTION = "credit"
DATASET = "lending_club"
EXPERIMENT = "credit_default_lending_club"
REGISTERED_MODEL_NAME = "uwcopilot_credit_pd"
MISSING_DATA_REASON = "Lending Club requires Kaggle credentials; run make data after adding a Kaggle token"
HEADLINE_METRICS = ("auc", "ks", "gini", "brier", "ece")
# MLflow saves sklearn models with skops, which refuses to load types it does not know. These are the only
# extra types in the scorecard pipeline, and all three are this repo's own code or numpy.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "uwcopilot.risk.credit.QuantileClipper",
    "uwcopilot.risk.credit._log1p_non_negative",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH, help="Kaggle lending-club.zip")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH, help="results JSON to update")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="where to save the fitted models")
    parser.add_argument("--checksum-log", type=Path, default=CHECKSUM_LOG, help="data/raw/checksums.jsonl")
    parser.add_argument("--n-resamples", type=int, default=1000, help="bootstrap resamples (ground rule: 1000)")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="rows per chunk when reading")
    parser.add_argument("--num-threads", type=int, default=4, help="LightGBM threads (the machine has 4 CPUs)")
    parser.add_argument("--no-mlflow", action="store_true", help="do not log the run to MLflow")
    parser.add_argument("--mlflow-db", type=Path, default=None, help="MLflow SQLite file (default: mlruns.db)")
    parser.add_argument("--mlflow-artifacts", type=Path, default=None, help="MLflow artifact folder (default: mlruns/)")
    return parser.parse_args(argv)


def relative_to_repo(path: Path) -> str:
    """Path relative to the repo root when possible (results files never contain machine-specific paths)."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return Path(path).name


def write_not_measured(args: argparse.Namespace, detail: str) -> dict[str, Any]:
    """Record that the credit model was not measured, and why. Returns the section as written."""
    payload = not_measured(MISSING_DATA_REASON)
    payload["detail"] = detail.replace(str(args.zip_path), relative_to_repo(args.zip_path))  # no machine paths
    payload["expected_input"] = relative_to_repo(args.zip_path)
    payload["pipeline"] = (
        "Implemented and tested on SYNTHETIC Lending Club-shaped data only (tests/risk/test_credit.py). "
        "No Lending Club metric has been computed."
    )
    update_results(args.results_path, SECTION, payload, datasets=[DATASET], checksum_log=args.checksum_log)
    return read_results(args.results_path)[SECTION]["results"]


def headline_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Point estimates and CI bounds to log to MLflow."""
    metrics: dict[str, float] = {}
    for model_name in (LOGISTIC, LIGHTGBM):
        test = payload["models"][model_name]["test"]
        for metric in HEADLINE_METRICS:
            for key in ("estimate", "ci_low", "ci_high"):
                metrics[f"{model_name}.test_{metric}.{key}"] = test[metric][key]
    difference = payload["comparison"]["test_auc_difference"]
    for key in ("estimate", "ci_low", "ci_high"):
        metrics[f"test_auc_difference_lightgbm_minus_logistic.{key}"] = difference[key]
    metrics["benchmark_lc_sub_grade.test_auc"] = payload["benchmark_lending_club_sub_grade"]["test_auc"]["estimate"]
    return metrics


def log_to_mlflow(
    args: argparse.Namespace, payload: dict[str, Any], models: CreditModels
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Log params, metrics and both models, register the chosen one, then save and attach the model bundle.

    Returns (run info for the results file, paths of the saved model files).
    """
    # Imported here so --no-mlflow never touches MLflow (tracking turns telemetry off before importing mlflow).
    from uwcopilot.risk import tracking

    store_options: dict[str, Path] = {}
    if args.mlflow_db is not None:
        store_options["db_path"] = args.mlflow_db
    if args.mlflow_artifacts is not None:
        store_options["artifact_dir"] = args.mlflow_artifacts
    tags = {"dataset": "Lending Club (Kaggle wordsforthewise/lending-club)", "phase": "1A", "split": "out-of-time"}
    with tracking.start_run(EXPERIMENT, run_name="train_credit", tags=tags, **store_options) as run:
        tracking.log_params(payload["config"])
        tracking.log_metrics(headline_metrics(payload))
        logistic_info = tracking.mlflow.sklearn.log_model(
            models.logistic.pipeline, name=LOGISTIC, skops_trusted_types=SKOPS_TRUSTED_TYPES
        )
        lightgbm_info = tracking.mlflow.lightgbm.log_model(models.lightgbm.booster, name=LIGHTGBM)
        chosen_uri = logistic_info.model_uri if models.chosen == LOGISTIC else lightgbm_info.model_uri
        chosen_test = payload["models"][models.chosen]["test"]
        version_tags = {
            "chosen_model": models.chosen,
            "selection_rule": payload["model_selection"]["rule"],
            "selection_reason": payload["model_selection"]["reason"],
            "test_auc": f"{chosen_test['auc']['estimate']:.4f}",
            "git_commit": git_commit(),
            "note": "LightGBM needs the category codes and Platt parameters in credit_bundle/metadata.json",
        }
        version = tracking.register_model(chosen_uri, REGISTERED_MODEL_NAME, tags=version_tags)
        models.metadata["model_version"] = f"{REGISTERED_MODEL_NAME} v{version}"
        paths = save_credit_models(models, args.models_dir)
        for path in paths.values():
            tracking.log_artifact(path, artifact_path="credit_bundle")
        tracking.mlflow.log_dict(payload, "credit_results.json")
        info = {
            "experiment": EXPERIMENT,
            "run_id": run.info.run_id,
            "registered_model": REGISTERED_MODEL_NAME,
            "registered_version": version,
            "registered_flavor": models.chosen,
            "store": "local SQLite (mlruns.db) + mlruns/ artifacts",
        }
    return info, paths


def format_metric(result: dict[str, Any]) -> str:
    """'0.712 [0.705, 0.719]' (3 decimals, for the printed table only; JSON keeps full precision)."""
    return f"{result['estimate']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]"


def print_summary(written: dict[str, Any]) -> None:
    """Print the headline table exactly as read back from the results file."""
    split = written["split"]
    counts = ", ".join(f"{name} {split[name]['n_rows']}" for name in ("train", "validation", "test"))
    print(f"rows: {counts}")
    print(f"test bad rate: {split['test']['bad_rate']:.4f}")
    print(f"\n{'model':<20}" + "".join(f"{metric:<26}" for metric in HEADLINE_METRICS))
    for model_name in (LOGISTIC, LIGHTGBM):
        test = written["models"][model_name]["test"]
        print(f"{model_name:<20}" + "".join(f"{format_metric(test[metric]):<26}" for metric in HEADLINE_METRICS))
    print(
        f"\nAUC difference (LightGBM - logistic, test): {format_metric(written['comparison']['test_auc_difference'])}"
    )
    benchmark = written["benchmark_lending_club_sub_grade"]["test_auc"]
    print(f"Benchmark: Lending Club sub_grade test AUC {format_metric(benchmark)}")
    print(f"Chosen model: {written['model_selection']['chosen']} ({written['model_selection']['reason']})")
    flags = written["stability"]["flags_above_threshold"]
    print(f"PSI flags above {written['stability']['threshold']}: {len(flags)}")
    for flag in flags:
        print(f"  {flag['variable']} {flag['issue_year']}: {flag['psi']:.3f}")


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline (or record 'not measured'), write results and models, print the summary."""
    args = parse_args(argv)
    ready, detail = zip_status(args.zip_path)
    if not ready:
        written = write_not_measured(args, detail)
        print(f"Lending Club data not available: {detail}")
        print(f"Wrote section '{SECTION}' to {args.results_path} as '{written['status']}': {written['reason']}")
        # A missing download is expected without Kaggle credentials; a broken one is an error to fix.
        return 0 if not Path(args.zip_path).exists() else 1

    started = time.perf_counter()
    print(f"Reading {args.zip_path} ...")
    data = load_lending_club(args.zip_path, chunk_rows=args.chunk_rows)
    print(f"  {len(data.loans)} individual loans prepared in {time.perf_counter() - started:.0f} s")
    config = CreditConfig(n_resamples=args.n_resamples)
    config.lgbm_params["num_threads"] = args.num_threads
    payload, models = run_credit_pipeline(data.loans, config, quality=data.quality)
    print(f"  pipeline finished at {time.perf_counter() - started:.0f} s")

    if args.no_mlflow:
        payload["mlflow"] = {"status": "not logged", "reason": "--no-mlflow was passed"}
        models.metadata["model_version"] = "unregistered (--no-mlflow)"
        paths = save_credit_models(models, args.models_dir)
    else:
        payload["mlflow"], paths = log_to_mlflow(args, payload, models)
    payload["saved_models"] = {name: relative_to_repo(path) for name, path in paths.items()}

    update_results(args.results_path, SECTION, payload, datasets=[DATASET], checksum_log=args.checksum_log)
    written = read_results(args.results_path)[SECTION]["results"]
    print(f"Wrote section '{SECTION}' to {args.results_path} and models to {args.models_dir}")
    print_summary(written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
