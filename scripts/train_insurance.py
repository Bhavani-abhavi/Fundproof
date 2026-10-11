"""Train the freMTPL2 insurance models (Phase 1, part B) and write every metric to results/risk_metrics.json.

Usage:
    uv run python scripts/train_insurance.py                    # full run (about 5-10 minutes on 4 CPUs)
    uv run python scripts/train_insurance.py --n-resamples 200  # quicker run with fewer bootstrap resamples

What it does, in order:
 1. Load and clean freMTPL2 (uwcopilot.risk.data.fremtpl2) and split it 60/20/20, seeded and stratified on
    "has a claim". freMTPL2 has no date field, so there is no out-of-time test.
 2. Frequency: Poisson GLM (offset log exposure) and LightGBM (objective=poisson, init_score = log exposure),
    early stopping on the validation split.
 3. Severity: large losses capped at the 99.5th percentile of TRAINING claim amounts; Gamma GLM and LightGBM
    (objective=gamma) on the capped average claim amount, weighted by the number of claims.
 4. Test metrics with bootstrap 95% CIs (1,000 resamples, seed 42), paired bootstrap LightGBM minus GLM,
    calibration by predicted decile, expected loss = frequency x severity (GLM pair, LightGBM pair and the
    pair the API serves), SHAP reason codes, and sensitivity checks (claimant-only block, recorded claim
    counts, a split grouped on the rating features).
 5. Plots to results/figures/, models to models/insurance/, runs to MLflow (mlruns.db + mlruns/), metrics to
    results/risk_metrics.json (sections insurance_frequency and insurance_severity), and the data notes
    (docs/insurance_data_notes.md), whose numbers come from the same dictionaries as the JSON.
"""

import argparse
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from uwcopilot.common.paths import DOCS_DIR, REPO_ROOT, RESULTS_DIR
from uwcopilot.risk import insurance
from uwcopilot.risk.data import fremtpl2
from uwcopilot.risk.metrics import lorenz_curve
from uwcopilot.risk.results import CHECKSUM_LOG, git_commit, not_measured, read_results, update_results
from uwcopilot.risk.scenario import FeatureRange, build_feature_ranges, monotonic_check
from uwcopilot.risk.tracking import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DB_PATH,
    log_artifact,
    log_metrics,
    log_params,
    register_model,
    start_run,
)

SEED = 42
N_RESAMPLES = 1000
DATASET_NAME = "fremtpl2"
FREQUENCY_SECTION = "insurance_frequency"
SEVERITY_SECTION = "insurance_severity"
FREQUENCY_REGISTERED_NAME = "insurance_frequency_lgbm"
SEVERITY_REGISTERED_NAME = "insurance_severity_lgbm"

# Chart colours (validated categorical slots 1 and 2 of the project's reference palette) and ink colours.
COLOR_GLM = "#2a78d6"
COLOR_LGBM = "#eb6834"
COLOR_INK = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_REFERENCE = "#8a8984"
COLOR_GRID = "#e4e3df"
COLOR_SURFACE = "#fcfcfb"

NO_TIME_FIELD = (
    "freMTPL2 has no date or policy-year field, so an out-of-time test is impossible. Train, validation and "
    "test are random draws from the same period; the test metrics do not show how the models would hold up "
    "on a later year."
)
DUPLICATE_RISK_UNMEASURED = (
    "The data has no customer or vehicle identifier beyond IDpol. Rows that describe the same insured risk in "
    "different periods cannot be kept in the same split, so if such rows exist the test metrics may be "
    "optimistic. Many rows share identical rating features (counted in data_quality)."
)
LIMITATIONS = [
    NO_TIME_FIELD,
    "Population stability (PSI) across time is not measured, for the same reason.",
    DUPLICATE_RISK_UNMEASURED,
    "The frequency target counts only claims with a recorded amount (see docs/insurance_data_notes.md), so "
    "frequency numbers are not directly comparable with published freMTPL2 results that use the raw ClaimNb.",
    "LightGBM hyperparameters were fixed in advance and not tuned; only the number of trees is chosen, by "
    "early stopping on the validation split.",
    "The validation split that early-stops LightGBM is also the one that chooses between GLM and LightGBM, "
    "which slightly favours LightGBM in that choice. The paired comparison on the untouched test split is the "
    "unbiased check of the choice; a cleaner design would early-stop on a separate fold of the training data.",
    "Claim amounts are capped at the 99.5th percentile of training claims, so expected loss is on a capped "
    "basis with no large-loss loading.",
    "Results are measured on public historical data from one French motor portfolio. They say nothing about "
    "performance on any other portfolio or in production.",
]


def _ci(values: Mapping[str, Any]) -> str:
    """'estimate [ci_low, ci_high]' with 4 decimals."""
    return f"{values['estimate']:.4f} [{values['ci_low']:.4f}, {values['ci_high']:.4f}]"


def _sign(values: Mapping[str, Any]) -> int:
    """+1 if a bootstrap interval lies above 0, -1 if below, 0 if it contains 0."""
    if values["ci_low"] > 0:
        return 1
    if values["ci_high"] < 0:
        return -1
    return 0


def duplicate_risk_limitation(grouped: Mapping[str, Any], headline_difference: Mapping[str, Any]) -> str:
    """The identical-risks limitation, stated with what grouped_split_sensitivity measured (built by code)."""
    if "lgbm_minus_glm" not in grouped:
        return DUPLICATE_RISK_UNMEASURED
    shares = grouped["share_of_test_policies_whose_rating_features_appear_in_train"]
    grouped_difference = grouped["lgbm_minus_glm"]["mean_poisson_deviance"]
    same = _sign(grouped_difference) == _sign(headline_difference) != 0
    conclusion = (
        "so the conclusion of the model comparison does not depend on identical risks sitting in both train and test"
        if same
        else "so the model comparison is sensitive to how identical risks are split, and the headline difference "
        "should be read with that in mind"
    )
    return (
        "The data has no customer or vehicle identifier beyond IDpol, so the same insured risk can sit in train and "
        f"test: {100 * shares['random_split']:.1f}% of test policies share every rating feature with a training "
        "policy. grouped_split_sensitivity refits both frequency models on a split grouped on the nine rating "
        f"features ({100 * shares['grouped_split']:.1f}% shared): LightGBM minus GLM mean Poisson deviance is "
        f"{_ci(grouped_difference)} there against {_ci(headline_difference)} on the random split, {conclusion}."
    )


def limitations(grouped: Mapping[str, Any], headline_difference: Mapping[str, Any]) -> list[str]:
    """LIMITATIONS with the identical-risks item replaced by its measured version."""
    measured = duplicate_risk_limitation(grouped, headline_difference)
    return [measured if item == DUPLICATE_RISK_UNMEASURED else item for item in LIMITATIONS]


# Scenario sanity checks (build plan, Phase 1 D): a documented subset of features with an expected direction.
# The share of policies that move the other way is reported, never asserted to be zero.
MONOTONIC_CHECKS: list[dict[str, Any]] = [
    {
        "feature": "BonusMalus",
        "step": 10.0,
        "expected": "increase",
        "why": "A higher bonus-malus level records a worse claims history, so predicted claim frequency "
        "should not fall when it rises.",
    },
]
NO_SEVERITY_MONOTONIC_EXPECTATION = (
    "no rating feature has a documented monotonic expectation for the average claim amount, so no severity "
    "check was defined"
)


@dataclass
class TrainConfig:
    """Paths and settings for one training run. The defaults are the real run; tests pass temporary paths."""

    data_dir: Path = fremtpl2.DEFAULT_DATA_DIR
    results_path: Path = RESULTS_DIR / "risk_metrics.json"
    figures_dir: Path = RESULTS_DIR / "figures"
    models_dir: Path = REPO_ROOT / "models" / "insurance"
    notes_path: Path = DOCS_DIR / "insurance_data_notes.md"
    checksum_log: Path = CHECKSUM_LOG
    mlflow_db: Path = DEFAULT_DB_PATH
    mlflow_artifacts: Path = DEFAULT_ARTIFACT_DIR
    experiment: str = "insurance_fremtpl2"
    n_resamples: int = N_RESAMPLES
    seed: int = SEED
    shap_sample_size: int = 2000
    min_level_count: int = insurance.DEFAULT_MIN_LEVEL_COUNT
    frequency_params: dict[str, Any] = field(default_factory=lambda: dict(insurance.FREQUENCY_LGBM_PARAMS))
    severity_params: dict[str, Any] = field(default_factory=lambda: dict(insurance.SEVERITY_LGBM_PARAMS))
    max_rounds: int = insurance.MAX_BOOST_ROUNDS
    early_stopping_rounds: int = insurance.EARLY_STOPPING_ROUNDS


class StageTimer:
    """Records how long each stage of the run takes, in seconds."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.last = self.start
        self.stages: dict[str, float] = {}

    def lap(self, stage: str) -> None:
        """Close the current stage and print its duration."""
        now = time.perf_counter()
        self.stages[stage] = round(now - self.last, 2)
        self.last = now
        print(f"  [{self.stages[stage]:7.1f}s] {stage}", flush=True)

    def total(self) -> float:
        """Seconds since the timer started."""
        return round(time.perf_counter() - self.start, 2)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def estimates_only(values: Any) -> Any:
    """Keep only estimate / ci_low / ci_high from nested bootstrap results (for MLflow metrics)."""
    if isinstance(values, Mapping):
        if "estimate" in values and "ci_low" in values:
            return {key: values[key] for key in ("estimate", "ci_low", "ci_high")}
        return {key: estimates_only(item) for key, item in values.items() if isinstance(item, Mapping | float | int)}
    return values


def scenario_sanity_checks(
    frequency_models: Mapping[str, Any], test: pd.DataFrame, ranges: Mapping[str, FeatureRange]
) -> dict[str, Any]:
    """Run MONOTONIC_CHECKS on the test policies for every frequency model (see scenario.monotonic_check)."""
    checks = []
    for spec in MONOTONIC_CHECKS:
        entry: dict[str, Any] = {key: spec[key] for key in ("feature", "step", "expected", "why")}
        for name, model in frequency_models.items():
            entry[name] = monotonic_check(
                model.predict_rate, test, spec["feature"], spec["step"], ranges[spec["feature"]], spec["expected"]
            )
        checks.append(entry)
    return {
        "method": "shift one feature by step on every test policy that stays inside the training range "
        "(others are skipped, never clipped); a violation is a prediction that moves against the expected "
        "direction",
        "population": "test split",
        "checks": checks,
    }


def claimant_block_sensitivity(
    test: pd.DataFrame,
    in_block: np.ndarray,
    predictions: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Frequency test metrics recomputed without the claimant-only IDpol block (a sensitivity check)."""
    keep = ~in_block
    if keep.sum() == 0 or test.loc[keep, "ClaimNb"].sum() == 0:
        return not_measured("no test policies with claims outside the claimant-only block")
    claims = test.loc[keep, "ClaimNb"].to_numpy(dtype=float)
    exposure = test.loc[keep, "Exposure"].to_numpy(dtype=float)
    result: dict[str, Any] = {"n_test_policies_kept": int(keep.sum()), "n_test_policies_removed": int(in_block.sum())}
    for name, predicted in predictions.items():
        metrics = insurance.frequency_metrics(claims, exposure, predicted[keep], n_resamples=n_resamples, seed=seed)
        metrics.pop("calibration_by_decile")
        result[name] = metrics
    result["lgbm_minus_glm"] = {
        "mean_poisson_deviance": insurance.paired_comparison(
            insurance.poisson_deviance_metric,
            claims / exposure,
            predictions["lgbm"][keep],
            predictions["glm"][keep],
            exposure,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "lorenz_gini": insurance.paired_comparison(
            insurance.gini_metric,
            claims / exposure,
            predictions["lgbm"][keep],
            predictions["glm"][keep],
            exposure,
            n_resamples=n_resamples,
            seed=seed,
        ),
    }
    return result


def recorded_count_sensitivity(
    splits: dict[str, pd.DataFrame],
    in_block: np.ndarray,
    primary_predictions: dict[str, np.ndarray],
    config: "TrainConfig",
) -> dict[str, Any]:
    """Refit both frequency models with the alternative treatment: keep claims that have no recorded amount.

    Point estimates only (no bootstrap): this shows the effect of the data-treatment decision, it is not a
    headline metric. The key number is observed / predicted claims on test policies outside the claimant-only
    block, for models trained each way.
    """
    recorded = {name: df.assign(ClaimNb=df["ClaimNbCapped"]) for name, df in splits.items()}
    train, validation, test = recorded["train"], recorded["validation"], recorded["test"]
    models = {
        "glm": insurance.fit_frequency_glm(train, min_level_count=config.min_level_count),
        "lgbm": insurance.fit_frequency_lgbm(
            train,
            validation,
            params=config.frequency_params,
            max_rounds=config.max_rounds,
            early_stopping_rounds=config.early_stopping_rounds,
        ),
    }
    claims = test["ClaimNb"].to_numpy(dtype=float)
    exposure = test["Exposure"].to_numpy(dtype=float)
    outside = ~in_block
    primary_claims = splits["test"]["ClaimNb"].to_numpy(dtype=float)
    result: dict[str, Any] = {
        "treatment": "claims without a recorded amount kept as claims (target = ClaimNbCapped)",
        "estimates": "point estimates on the test split, no bootstrap (sensitivity check, not a headline metric)",
        "n_test_policies_outside_block": int(outside.sum()),
    }
    for name, model in models.items():
        predicted = model.predict_rate(test)
        result[name] = {
            "test_mean_poisson_deviance_on_recorded_counts": insurance.poisson_deviance_metric(
                claims / exposure, predicted, exposure
            ),
            "test_lorenz_gini_on_recorded_counts": insurance.gini_metric(claims / exposure, predicted, exposure),
            "observed_to_predicted_claims_all_test": insurance.ratio_of_sums(claims, predicted * exposure),
            "observed_to_predicted_claims_outside_block": insurance.ratio_of_sums(
                claims[outside], predicted[outside] * exposure[outside]
            ),
            "primary_model_observed_to_predicted_claims_outside_block": insurance.ratio_of_sums(
                primary_claims[outside], primary_predictions[name][outside] * exposure[outside]
            ),
        }
    return result


def grouped_split_sensitivity(
    policies: pd.DataFrame, random_splits: Mapping[str, pd.DataFrame], config: "TrainConfig"
) -> dict[str, Any]:
    """Refit both frequency models on a split grouped on the rating features, and compare them on its test set.

    The headline split is random, and freMTPL2 has no customer identifier, so a policy in the test split can
    have exactly the same rating features as one in training. Here no rating-feature vector is in more than
    one split. Point estimates for each model, and a paired bootstrap of LightGBM minus GLM (same settings as
    the headline metrics) for the comparison that matters.
    """
    grouped = fremtpl2.split_policies_grouped(policies, seed=config.seed)
    train, validation, test = grouped["train"], grouped["validation"], grouped["test"]
    models = {
        "glm": insurance.fit_frequency_glm(train, min_level_count=config.min_level_count),
        "lgbm": insurance.fit_frequency_lgbm(
            train,
            validation,
            params=config.frequency_params,
            max_rounds=config.max_rounds,
            early_stopping_rounds=config.early_stopping_rounds,
        ),
    }
    claims = test["ClaimNb"].to_numpy(dtype=float)
    exposure = test["Exposure"].to_numpy(dtype=float)
    frequency = claims / exposure
    predictions = {name: model.predict_rate(test) for name, model in models.items()}
    result: dict[str, Any] = {
        "method": (
            "sklearn GroupShuffleSplit on the nine rating features (identical vectors stay in one split), "
            f"seed {config.seed}, 60/20/20 of groups, not stratified; both frequency models refitted with the "
            "headline settings"
        ),
        "estimates": "point estimates per model; paired bootstrap 95% CI for LightGBM minus GLM",
        "sizes": fremtpl2.split_summary(grouped),
        "share_of_test_policies_whose_rating_features_appear_in_train": {
            "random_split": fremtpl2.share_with_features_seen_in(random_splits["test"], random_splits["train"]),
            "grouped_split": fremtpl2.share_with_features_seen_in(test, train),
        },
        "lgbm_best_iteration": models["lgbm"].training_info["best_iteration"],
    }
    for name, predicted in predictions.items():
        result[name] = {
            "mean_poisson_deviance": insurance.poisson_deviance_metric(frequency, predicted, exposure),
            "lorenz_gini": insurance.gini_metric(frequency, predicted, exposure),
            "observed_to_predicted_claims": insurance.ratio_of_sums(claims, predicted * exposure),
        }
    result["lgbm_minus_glm"] = {
        metric: insurance.paired_comparison(
            metric_fn,
            frequency,
            predictions["lgbm"],
            predictions["glm"],
            exposure,
            n_resamples=config.n_resamples,
            seed=config.seed,
        )
        for metric, metric_fn in (
            ("mean_poisson_deviance", insurance.poisson_deviance_metric),
            ("lorenz_gini", insurance.gini_metric),
        )
    }
    return result


def expected_loss_section(
    test: pd.DataFrame,
    observed_loss: np.ndarray,
    frequency_predictions: Mapping[str, np.ndarray],
    severity_models: Mapping[str, Any],
    preferred: Mapping[str, str],
    total_observed_uncapped: float,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Expected loss = frequency x severity on the test split for the GLM pair, the LightGBM pair and the pair
    the scoring API serves (each model preferred on validation), all with bootstrap CIs."""
    exposure = test["Exposure"].to_numpy(dtype=float)
    severity_predictions = {name: model.predict_rate(test) for name, model in severity_models.items()}
    section: dict[str, Any] = {
        "definition": (
            "predicted loss per policy = predicted frequency (claims/year) x exposure (years) x predicted "
            "severity (capped amount per claim); totals over all test policies"
        ),
        "basis": (
            "capped claim amounts; claims without a recorded amount are not counted on either side "
            "(see data_quality.modelling_target). On the few policies whose number of amounts differs from the "
            "capped claim count, observed loss includes every recorded amount (see "
            "data_quality.linkage.n_policies_where_severity_rows_differ_from_modelling_claimnb)"
        ),
        "n_test_policies": int(len(test)),
        "n_test_policies_whose_claims_have_no_amounts": int(
            ((test["ClaimNbCapped"] > 0) & (test["n_severity_rows"] == 0)).sum()
        ),
        "total_observed_loss_uncapped": total_observed_uncapped,
    }
    pairs = {
        "glm": ("glm", "glm"),
        "lgbm": ("lgbm", "lgbm"),
        "served_pair": (preferred["frequency"], preferred["severity"]),
    }
    for key, (frequency_name, severity_name) in pairs.items():
        section[key] = {
            "frequency_model": frequency_name,
            "severity_model": severity_name,
            **insurance.expected_loss_summary(
                observed_loss,
                exposure,
                frequency_predictions[frequency_name],
                severity_predictions[severity_name],
                n_resamples=n_resamples,
                seed=seed,
            ),
        }
    section["served_pair"]["description"] = (
        "the frequency and severity models the scoring API serves (each preferred on the validation split)"
    )
    return section


# ---------------------------------------------------------------------------
# Plots (matplotlib object API, no global state)
# ---------------------------------------------------------------------------


def _style_axes(ax: Any, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and axes, text in ink colours."""
    ax.set_facecolor(COLOR_SURFACE)
    ax.set_title(title, color=COLOR_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=COLOR_INK_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=COLOR_INK_SECONDARY, fontsize=9)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.tick_params(colors=COLOR_INK_SECONDARY, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)


def _thin(x: np.ndarray, y: np.ndarray, max_points: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """Keep at most max_points evenly spaced points of a long curve (the plot looks the same, the file is small)."""
    if x.size <= max_points:
        return x, y
    index = np.unique(np.linspace(0, x.size - 1, max_points).astype(int))
    return x[index], y[index]


def plot_lorenz(
    path: Path,
    observed_rate: np.ndarray,
    weight: np.ndarray,
    predictions: dict[str, np.ndarray],
    ginis: dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """Lorenz curves (policies sorted from lowest to highest predicted risk) for GLM and LightGBM."""
    figure = Figure(figsize=(6.4, 5.0), facecolor=COLOR_SURFACE)
    ax = figure.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", color=COLOR_REFERENCE, linewidth=1.2, label="Random ranking")
    oracle_x, oracle_y = lorenz_curve(observed_rate, observed_rate, weight)
    oracle_x, oracle_y = _thin(oracle_x, oracle_y)
    ax.plot(oracle_x, oracle_y, linestyle=":", color=COLOR_REFERENCE, linewidth=1.5, label="Oracle (sorted by outcome)")
    for name, label, color in (("glm", "GLM", COLOR_GLM), ("lgbm", "LightGBM", COLOR_LGBM)):
        x, y = lorenz_curve(observed_rate, predictions[name], weight)
        x, y = _thin(x, y)
        ax.plot(x, y, color=color, linewidth=2, label=f"{label} (Gini {ginis[name]:.3f})")
    _style_axes(ax, title, xlabel, ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_INK, loc="upper left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


def plot_calibration(path: Path, tables: dict[str, list[dict[str, Any]]], title: str, ylabel: str) -> Path:
    """Observed (with 95% bootstrap CI) vs predicted mean per predicted decile, one panel per model."""
    figure = Figure(figsize=(9.6, 4.2), facecolor=COLOR_SURFACE)
    axes = figure.subplots(1, 2, sharey=True)
    for ax, (name, label, color) in zip(
        axes, (("glm", "GLM", COLOR_GLM), ("lgbm", "LightGBM", COLOR_LGBM)), strict=True
    ):
        rows = tables[name]
        deciles = np.array([row["decile"] for row in rows])
        predicted = np.array([row["mean_predicted_rate"] for row in rows])
        observed = np.array([row["mean_observed_rate"] for row in rows])
        low = np.array([row["mean_observed_rate_ci_low"] for row in rows])
        high = np.array([row["mean_observed_rate_ci_high"] for row in rows])
        ax.plot(deciles, predicted, color=color, linewidth=2, marker="o", markersize=5, label="Predicted")
        ax.errorbar(
            deciles,
            observed,
            yerr=[np.maximum(observed - low, 0.0), np.maximum(high - observed, 0.0)],
            fmt="s",
            color=COLOR_INK,
            markersize=4,
            elinewidth=1,
            capsize=2,
            label="Observed (95% CI)",
        )
        _style_axes(ax, f"{label}", "Predicted decile (1 = lowest)", ylabel)
        ax.set_xticks(deciles)
        ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_INK, loc="upper left")
    figure.suptitle(title, color=COLOR_INK, fontsize=11, x=0.01, ha="left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return path


# ---------------------------------------------------------------------------
# Data notes (docs/insurance_data_notes.md), rendered from the computed numbers
# ---------------------------------------------------------------------------


def _n(value: float | int) -> str:
    """Integer with thousands separators."""
    return f"{int(round(value)):,}"


def _pct(value: float) -> str:
    """Share as a percentage with two decimals."""
    return f"{100 * value:.2f}%"


def _table_row(cells: tuple[str, ...]) -> str:
    """One Markdown table row."""
    return "| " + " | ".join(cells) + " |"


def render_data_notes(
    quality: dict[str, Any],
    split: dict[str, Any],
    cap: dict[str, Any],
    recorded_sensitivity: dict[str, Any],
    block_sensitivity: dict[str, Any],
    generated_at: str,
) -> str:
    """Markdown data notes. Every number is taken from the dictionaries written to results/risk_metrics.json."""
    frequency = quality["frequency_file"]
    severity = quality["severity_file"]
    linkage = quality["linkage"]
    above_cap = frequency["policies_above_claim_cap"]
    above_cap_text = ", ".join(f"{row['ClaimNb']} claims in {row['Exposure']:g} yr" for row in above_cap)
    mismatches = linkage["policies_where_severity_rows_differ_from_claimnb"]
    mismatch_text = (
        "; ".join(
            f"IDpol {row['IDpol']}: ClaimNb {row['ClaimNbRecorded']}, {row['n_severity_rows']} amount row(s)"
            for row in mismatches
        )
        or "none"
    )
    modelling_mismatches = linkage.get("policies_where_severity_rows_differ_from_modelling_claimnb")
    n_modelling_mismatches = linkage.get("n_policies_where_severity_rows_differ_from_modelling_claimnb")
    modelling_mismatch_text = (
        "; ".join(
            f"IDpol {row['IDpol']}: modelling ClaimNb {row['ClaimNbModelling']}, {row['n_severity_rows']} amount(s)"
            for row in modelling_mismatches
        )
        or "none"
        if modelling_mismatches is not None
        else "not measured"
    )
    band_rows = "\n".join(
        f"| {row['idpol_band']} | {_n(row['n_policies'])} | {_n(row['n_policies_with_claims'])} | "
        f"{_n(row['n_claim_policies_with_amounts'])} | {_n(row['n_claim_policies_without_amounts'])} |"
        for row in linkage["claim_policies_by_idpol_band"]
    )
    split_rows = "\n".join(
        f"| {name} | {_n(row['n_policies'])} | {row['total_exposure_years']:,.0f} | {_n(row['n_claims'])} | "
        f"{_pct(row['share_policies_with_claims'])} | {_n(row['n_policies_with_claim_amounts'])} |"
        for name, row in split["sizes"].items()
    )
    share_from_block_on = linkage["share_of_policies_with_claims_from_that_idpol_on"]
    share_from_block_on_text = _pct(share_from_block_on) if share_from_block_on is not None else "not applicable"
    cleaning_rows = "\n".join(
        _table_row(cells)
        for cells in (
            (
                "IDpol type",
                f"`IDpol` is `{frequency['idpol_dtype_in_file']}` in the frequency file, integer in the severity file",
                "Convert to int64 after checking every value is a whole number",
                "The two files must join on the same key",
            ),
            (
                "Quoted text",
                f"{_n(frequency['text_values_with_quotes_or_spaces_stripped']['VehGas'])} `VehGas` values carry "
                "literal single quotes",
                "Strip quotes and spaces from all text columns; `VehGas` values are now "
                + ", ".join(frequency["vehgas_values_after_cleaning"]),
                "`'Regular'` and `Regular` must be the same category",
            ),
            (
                "Exposure above 1 year",
                f"{_n(frequency['n_policies_exposure_above_cap'])} policies "
                f"(largest {frequency['max_exposure_recorded']:g} years)",
                f"Cap at {frequency['exposure_cap_years']:g} year",
                'One record covers at most one policy year; the scikit-learn example "Tweedie regression on '
                'insurance claims" applies the same cap. The recorded value is kept in `ExposureRecorded`',
            ),
            (
                f"Claim counts above {frequency['claim_count_cap']}",
                f"{_n(frequency['n_policies_claimnb_above_cap'])} policies (largest "
                f"{frequency['max_claimnb_recorded']}), "
                f"{_n(frequency['n_claims_removed_by_cap'])} claims above the cap",
                f"Cap `ClaimNb` at {frequency['claim_count_cap']}",
                f"Several are implausible for motor liability ({above_cap_text}), so they look like data errors; "
                "the same scikit-learn example caps at 4. The recorded value is kept in `ClaimNbRecorded`",
            ),
            (
                "Non-positive exposure",
                f"{_n(frequency['n_rows_dropped_non_positive_exposure'])} rows",
                "Drop",
                "log(exposure) is the model offset",
            ),
            (
                "Non-positive claim amounts",
                f"{_n(severity['n_rows_dropped_non_positive_amount'])} rows",
                "Drop",
                "The Gamma severity model needs amounts above zero",
            ),
            (
                "Claim amounts for unknown policies",
                f"{_n(linkage['n_severity_rows_with_unknown_idpol'])} amount rows for "
                f"{_n(linkage['n_unknown_idpol_values_in_severity_file'])} IDpol values not in the frequency file",
                "Drop",
                "They have no exposure and no rating features",
            ),
            (
                "Amount rows differ from ClaimNb",
                f"{_n(linkage['n_policies_where_severity_rows_differ_from_claimnb'])} policies against the file's "
                f"ClaimNb ({mismatch_text}). Against the capped modelling ClaimNb: "
                + (
                    f"{_n(n_modelling_mismatches)} policies ({modelling_mismatch_text}); "
                    f"{_n(linkage['n_amount_rows_above_modelling_claimnb'])} amount rows are above the modelling "
                    f"count and {_n(linkage['n_modelling_claims_without_an_amount_row'])} modelled claim(s) have no "
                    "amount row"
                    if n_modelling_mismatches is not None
                    else "not measured"
                ),
                "Keep; severity uses the number of amount rows as the claim count",
                "The average amount must be divided by the number of amounts actually recorded",
            ),
        )
    )
    if linkage["n_policies_below_that_idpol"] > 0:
        block_text = (
            f"Where they sit: every policy with IDpol below {_n(linkage['lowest_idpol_with_zero_claims'])} has at "
            f"least one claim ({_n(linkage['n_policies_below_that_idpol'])} policies, against "
            f"{share_from_block_on_text} of policies with a claim from that IDpol on), and "
            f"{_n(linkage['n_policies_below_that_idpol_without_amounts'])} of them have no claim amounts. This "
            f"claimant-only block holds {_n(linkage['n_claims_below_that_idpol'])} claims, "
            f"{_pct(linkage['share_of_all_claims_below_that_idpol'])} of all claims. A random sample of policies "
            f"in which about {share_from_block_on_text} have a claim would essentially never produce such a run, so "
            "our reading is that the block is an artifact of how the public files were assembled, not a set of "
            "claims that cost nothing. The table shows the same pattern by IDpol band."
        )
    else:
        block_text = "No claimant-only block of low IDpol values was found in this data. Counts by IDpol band:"
    target = quality["modelling_target"]
    treatment_text = (
        "Treatment (a judgement call, stated so it can be challenged): as in the scikit-learn example "
        '"Tweedie regression on insurance claims", a claim with no recorded amount is not counted as a claim '
        f"for modelling. `ClaimNb` is set to 0 on {_n(target['n_policies_set_to_zero_claims'])} policies "
        f"({_n(target['n_claims_not_counted'])} claims); the capped count stays in `ClaimNbCapped` and the file "
        f"value in `ClaimNbRecorded`. The modelling target then has {_n(target['n_claims_for_modelling'])} "
        f"claims, {target['claims_per_policy_year_for_modelling']:.4f} claims per policy-year (against "
        f"{target['claims_per_policy_year_as_recorded_after_cap']:.4f} as recorded). Frequency and severity now "
        "describe the same claims"
        + (
            f", except on {_n(n_modelling_mismatches)} policies whose number of claim amounts differs from the "
            "capped claim count (the cleaning table lists them)"
            if n_modelling_mismatches
            else ""
        )
        + ", so expected loss (frequency x severity) can be compared directly with observed losses."
    )
    block_share = linkage.get("share_of_policies_below_that_idpol_with_amounts")
    if block_share is not None and linkage["n_policies_below_that_idpol"] > 0:
        treatment_text += (
            f" After this step, {_pct(block_share)} of the policies in the claimant-only block have a claim, "
            f"close to the {share_from_block_on_text} of policies with a recorded claim in the rest of the file, "
            "which supports the reading "
            "that the claims without amounts are an artifact."
        )
    if isinstance(recorded_sensitivity.get("glm"), dict):
        glm_alt = recorded_sensitivity["glm"]
        lgbm_alt = recorded_sensitivity["lgbm"]
        treatment_text += (
            "\n\nWhy not keep them as claims? `recorded_claim_count_sensitivity` in `results/risk_metrics.json` "
            "refits both frequency models on the recorded counts. On the "
            f"{_n(recorded_sensitivity['n_test_policies_outside_block'])} test policies outside the claimant-only "
            "block, observed / predicted claims is then "
            f"{glm_alt['observed_to_predicted_claims_outside_block']:.3f} (GLM) and "
            f"{lgbm_alt['observed_to_predicted_claims_outside_block']:.3f} (LightGBM): those models predict too "
            "many claims for ordinary policies, because they learn from the block. With the chosen treatment the "
            f"same ratio is {glm_alt['primary_model_observed_to_predicted_claims_outside_block']:.3f} (GLM) and "
            f"{lgbm_alt['primary_model_observed_to_predicted_claims_outside_block']:.3f} (LightGBM). These are "
            "point estimates from a sensitivity check, not headline metrics."
        )
    if isinstance(block_sensitivity.get("lgbm_minus_glm"), dict):
        deviance_gap = block_sensitivity["lgbm_minus_glm"]["mean_poisson_deviance"]
        treatment_text += (
            "\n\nThe headline frequency metrics are also recomputed without the claimant-only block "
            "(`claimant_only_block_sensitivity`): the LightGBM minus GLM difference in mean Poisson deviance is "
            f"{deviance_gap['estimate']:.4f} (95% CI {deviance_gap['ci_low']:.4f} to {deviance_gap['ci_high']:.4f})."
        )
    train_cap = cap["train"]
    test_cap = cap["test"]

    return f"""# freMTPL2 data notes

> Generated by `scripts/train_insurance.py` from `results/risk_metrics.json` (sections `insurance_frequency`
> and `insurance_severity`, generated {generated_at}). Every number below is read from that file. Do not edit by
> hand: rerun `make train-insurance` (or `uv run python scripts/train_insurance.py --notes-only` to re-render
> these notes without retraining).

Source: freMTPL2freq (OpenML 41214) and freMTPL2sev (OpenML 41215), French motor third-party liability
policies, downloaded by `scripts/download_data.py`. File checksums are in the results metadata.

## Cleaning steps

| Step | What the code found | What the code does | Why |
|---|---|---|---|
{cleaning_rows}

After cleaning: {_n(frequency["n_policies"])} policies, {frequency["total_exposure_years"]:,.0f} policy-years of
exposure, {_n(frequency["n_claims_after_cap"])} claims on {_n(frequency["n_policies_with_claims"])} policies.

## Policies with claims but no claim amounts

{_n(linkage["n_policies_with_claims_but_no_amounts"])} policies with `ClaimNb > 0` have no row in the severity
file. They carry {_n(linkage["n_claims_on_policies_without_amounts"])} claims (after the cap), which is
{_pct(linkage["share_of_claims_without_amounts"])} of all claims and
{_pct(linkage["share_of_claim_policies_without_amounts"])} of policies with claims.

{block_text}

| IDpol band | Policies | Policies with claims | ... with amounts | ... without amounts |
|---|---|---|---|---|
{band_rows}

{treatment_text}

## Split

Seeded ({split["seed"]}) random split, stratified on `ClaimNb > 0`, 60% / 20% / 20%.
{NO_TIME_FIELD}

| Split | Policies | Exposure (years) | Claims | Share with a claim | Policies with amounts |
|---|---|---|---|---|---|
{split_rows}

## Large-loss cap (severity)

Each claim amount is capped at the {100 * cap["quantile"]:g}th percentile of the training split's claim
amounts: {cap["cap_value"]:,.2f}. In the training split this caps {_n(train_cap["n_claims_capped"])} of
{_n(train_cap["n_claims"])} claims ({_pct(train_cap["share_of_claims_capped"])}) and removes
{_pct(train_cap["share_of_amount_removed_by_cap"])} of the total claim amount. In the test split it caps
{_n(test_cap["n_claims_capped"])} of {_n(test_cap["n_claims"])} claims
({_pct(test_cap["share_of_claims_capped"])}), removing {_pct(test_cap["share_of_amount_removed_by_cap"])} of the
amount. Expected loss is therefore on a capped basis; pricing the removed large losses would need a separate
loading, which is not built here.
"""


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------


def log_lightgbm_model(
    model: insurance.TreeModel, artifact_name: str, registered_name: str, tags: dict[str, str]
) -> str:
    """Log a booster to the active MLflow run, register it, and return the registered version."""
    # Imported here, after uwcopilot.risk.tracking has switched MLflow telemetry off before MLflow's first import.
    import mlflow.lightgbm

    info = mlflow.lightgbm.log_model(
        model.booster, name=artifact_name, pip_requirements=[f"lightgbm=={lgb.__version__}"]
    )
    return register_model(info.model_uri, registered_name, tags=tags)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run(config: TrainConfig) -> dict[str, dict[str, Any]]:
    """Train, evaluate and write everything. Returns the two results payloads that were written."""
    timer = StageTimer()
    n_resamples, seed = config.n_resamples, config.seed
    bootstrap_settings = {"n_resamples": n_resamples, "seed": seed, "alpha": 0.05, "method": "percentile"}

    print("Loading freMTPL2 ...", flush=True)
    data = fremtpl2.load_fremtpl2(config.data_dir)
    duplicated = data.policies.duplicated(fremtpl2.RATING_FEATURES, keep=False)
    data.quality["share_of_policies_sharing_rating_features_with_another_policy"] = float(duplicated.mean())
    splits = fremtpl2.split_policies(data.policies, seed=seed)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    split_info = {
        "method": "random, stratified on ClaimNb > 0 (sklearn train_test_split, twice)",
        "seed": seed,
        "fractions": fremtpl2.SPLIT_FRACTIONS,
        "sizes": fremtpl2.split_summary(splits),
        "out_of_time": not_measured(NO_TIME_FIELD),
    }
    timer.lap("load, clean and split")

    # ---- Frequency ---------------------------------------------------------------------------------------
    print("Fitting frequency models ...", flush=True)
    frequency_glm = insurance.fit_frequency_glm(train, min_level_count=config.min_level_count)
    timer.lap("frequency GLM")
    frequency_lgbm = insurance.fit_frequency_lgbm(
        train,
        validation,
        params=config.frequency_params,
        max_rounds=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    timer.lap("frequency LightGBM")

    frequency_models = {"glm": frequency_glm, "lgbm": frequency_lgbm}
    validation_frequency = {
        name: insurance.poisson_deviance_metric(
            validation["ClaimNb"] / validation["Exposure"], model.predict_rate(validation), validation["Exposure"]
        )
        for name, model in frequency_models.items()
    }
    preferred_frequency = min(validation_frequency, key=validation_frequency.get)

    test_claims = test["ClaimNb"].to_numpy(dtype=float)
    test_exposure = test["Exposure"].to_numpy(dtype=float)
    frequency_predictions = {name: model.predict_rate(test) for name, model in frequency_models.items()}
    print("Bootstrapping frequency metrics ...", flush=True)
    frequency_test = {
        name: insurance.frequency_metrics(test_claims, test_exposure, predicted, n_resamples=n_resamples, seed=seed)
        for name, predicted in frequency_predictions.items()
    }
    frequency_test["lgbm_minus_glm"] = {
        metric: insurance.paired_comparison(
            metric_fn,
            test_claims / test_exposure,
            frequency_predictions["lgbm"],
            frequency_predictions["glm"],
            test_exposure,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric, metric_fn in (
            ("mean_poisson_deviance", insurance.poisson_deviance_metric),
            ("lorenz_gini", insurance.gini_metric),
        )
    }
    in_block = fremtpl2.claimant_only_block_mask(
        test, data.quality["linkage"]["lowest_idpol_with_zero_claims"]
    ).to_numpy()
    block_sensitivity = claimant_block_sensitivity(test, in_block, frequency_predictions, n_resamples, seed)
    timer.lap("frequency metrics (bootstrap)")
    recorded_sensitivity = recorded_count_sensitivity(splits, in_block, frequency_predictions, config)
    timer.lap("sensitivity: frequency models refit on recorded claim counts")
    grouped_sensitivity = grouped_split_sensitivity(data.policies, splits, config)
    timer.lap("sensitivity: frequency models refit on a split grouped on rating features")

    explanation = insurance.explain_frequency_model(
        frequency_lgbm, test, sample_size=config.shap_sample_size, seed=seed
    )
    timer.lap("SHAP reason codes")

    # ---- Severity ----------------------------------------------------------------------------------------
    print("Fitting severity models ...", flush=True)
    claims_by_split = {name: data.claims.loc[data.claims["IDpol"].isin(df["IDpol"])] for name, df in splits.items()}
    cap_value = fremtpl2.severity_cap(claims_by_split["train"]["ClaimAmount"])
    cap_info = {
        "quantile": fremtpl2.SEVERITY_CAP_QUANTILE,
        "cap_value": cap_value,
        "computed_on": "individual claim amounts of the training split",
        **{name: fremtpl2.cap_summary(claims["ClaimAmount"], cap_value) for name, claims in claims_by_split.items()},
    }
    severity_tables = {name: fremtpl2.severity_table(df, data.claims, cap_value) for name, df in splits.items()}
    severity_glm = insurance.fit_severity_glm(severity_tables["train"], min_level_count=config.min_level_count)
    severity_lgbm = insurance.fit_severity_lgbm(
        severity_tables["train"],
        severity_tables["validation"],
        params=config.severity_params,
        max_rounds=config.max_rounds,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    severity_models = {"glm": severity_glm, "lgbm": severity_lgbm}
    timer.lap("severity models")

    sev_valid, sev_test = severity_tables["validation"], severity_tables["test"]
    validation_severity = {
        name: insurance.gamma_deviance_metric(
            sev_valid["average_amount_capped"], model.predict_rate(sev_valid), sev_valid["n_claims"]
        )
        for name, model in severity_models.items()
    }
    preferred_severity = min(validation_severity, key=validation_severity.get)
    test_average = sev_test["average_amount_capped"].to_numpy(dtype=float)
    test_n_claims = sev_test["n_claims"].to_numpy(dtype=float)
    severity_predictions = {name: model.predict_rate(sev_test) for name, model in severity_models.items()}
    print("Bootstrapping severity metrics ...", flush=True)
    severity_test = {
        name: insurance.severity_metrics(test_average, test_n_claims, predicted, n_resamples=n_resamples, seed=seed)
        for name, predicted in severity_predictions.items()
    }
    severity_test["lgbm_minus_glm"] = {
        metric: insurance.paired_comparison(
            metric_fn,
            test_average,
            severity_predictions["lgbm"],
            severity_predictions["glm"],
            test_n_claims,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric, metric_fn in (
            ("mean_gamma_deviance", insurance.gamma_deviance_metric),
            ("lorenz_gini", insurance.gini_metric),
        )
    }
    timer.lap("severity metrics (bootstrap)")

    # ---- Risk bands for the scoring API, and scenario sanity checks --------------------------------------
    feature_ranges = build_feature_ranges(
        train[fremtpl2.RATING_FEATURES], categorical_columns=fremtpl2.CATEGORICAL_FEATURES
    )
    bundle = insurance.InsuranceModels(
        frequency_glm=frequency_glm,
        frequency_lgbm=frequency_lgbm,
        severity_glm=severity_glm,
        severity_lgbm=severity_lgbm,
        feature_ranges=feature_ranges,
        metadata={},
    )
    preferred = {"frequency": preferred_frequency, "severity": preferred_severity}
    risk_bands = insurance.risk_bands_for(bundle, train, preferred)
    frequency_scenario_checks = scenario_sanity_checks(frequency_models, test, feature_ranges)
    timer.lap("risk bands and scenario checks")

    # ---- Expected loss = frequency x severity ------------------------------------------------------------
    expected_loss = expected_loss_section(
        test,
        fremtpl2.observed_capped_loss(test, data.claims, cap_value),
        frequency_predictions,
        severity_models,
        preferred,
        float(claims_by_split["test"]["ClaimAmount"].sum()),
        n_resamples,
        seed,
    )
    timer.lap("expected loss")

    # ---- Plots -------------------------------------------------------------------------------------------
    figures = {
        "frequency_lorenz": plot_lorenz(
            config.figures_dir / "insurance_frequency_lorenz.png",
            test_claims / test_exposure,
            test_exposure,
            frequency_predictions,
            {name: frequency_test[name]["lorenz_gini"]["estimate"] for name in ("glm", "lgbm")},
            "Claim frequency: Lorenz curves on the test split (freMTPL2)",
            "Cumulative share of exposure (policies sorted by predicted frequency)",
            "Cumulative share of claims",
        ),
        "frequency_calibration": plot_calibration(
            config.figures_dir / "insurance_frequency_calibration.png",
            {name: frequency_test[name]["calibration_by_decile"] for name in ("glm", "lgbm")},
            "Claim frequency by predicted decile, test split (equal exposure per decile)",
            "Claims per policy-year",
        ),
        "severity_lorenz": plot_lorenz(
            config.figures_dir / "insurance_severity_lorenz.png",
            test_average,
            test_n_claims,
            severity_predictions,
            {name: severity_test[name]["lorenz_gini"]["estimate"] for name in ("glm", "lgbm")},
            "Claim severity: Lorenz curves on the test split (freMTPL2)",
            "Cumulative share of claims (policies sorted by predicted severity)",
            "Cumulative share of capped claim amount",
        ),
        "severity_calibration": plot_calibration(
            config.figures_dir / "insurance_severity_calibration.png",
            {name: severity_test[name]["calibration_by_decile"] for name in ("glm", "lgbm")},
            "Average claim amount by predicted decile, test split (equal claim count per decile)",
            "Average capped claim amount",
        ),
    }
    timer.lap("plots")

    # ---- Payloads ----------------------------------------------------------------------------------------
    run_limitations = limitations(grouped_sensitivity, frequency_test["lgbm_minus_glm"]["mean_poisson_deviance"])
    frequency_payload: dict[str, Any] = {
        "status": "measured",
        "dataset": "freMTPL2freq (OpenML 41214), public French motor third-party liability data",
        "target": (
            "ClaimNb per policy (capped at 4; claims without a recorded amount not counted), with log(Exposure) "
            "offset; rates are claims per policy-year"
        ),
        "data_quality": data.quality,
        "split": split_info,
        "models": {
            "glm": {
                "type": "Poisson GLM, log link, offset log(Exposure) (statsmodels IRLS)",
                "n_parameters": len(frequency_glm.coefficients),
                "converged": frequency_glm.converged,
                "design": frequency_glm.design.to_dict(),
            },
            "lgbm": {
                "type": "LightGBM objective=poisson, init_score = log(Exposure) + log(train claim frequency)",
                "base_log_frequency": frequency_lgbm.base_log_value,
                **frequency_lgbm.training_info,
            },
        },
        "validation_metrics": {name: {"mean_poisson_deviance": value} for name, value in validation_frequency.items()},
        "preferred_model": {"name": preferred_frequency, "rule": "lower mean Poisson deviance on the validation split"},
        "test_metrics": frequency_test,
        "claimant_only_block_sensitivity": block_sensitivity,
        "recorded_claim_count_sensitivity": recorded_sensitivity,
        "grouped_split_sensitivity": grouped_sensitivity,
        "explainability": explanation,
        "risk_bands": risk_bands["insurance_frequency"],
        "scenario_sanity_checks": frequency_scenario_checks,
        "stability": not_measured(f"PSI needs a time field. {NO_TIME_FIELD}"),
        "out_of_time_test": not_measured(NO_TIME_FIELD),
        "bootstrap": bootstrap_settings,
        "limitations": run_limitations,
    }
    severity_payload: dict[str, Any] = {
        "status": "measured",
        "dataset": "freMTPL2sev (OpenML 41215) joined to freMTPL2freq on IDpol",
        "target": "average capped claim amount per policy with claim amounts; weight = number of claim amounts",
        "large_loss_cap": cap_info,
        "split": {**split_info, "severity_rows": {name: int(len(df)) for name, df in severity_tables.items()}},
        "models": {
            "glm": {
                "type": "Gamma GLM, log link, var_weights = number of claims (statsmodels IRLS)",
                "n_parameters": len(severity_glm.coefficients),
                "converged": severity_glm.converged,
                "design": severity_glm.design.to_dict(),
            },
            "lgbm": {
                "type": "LightGBM objective=gamma, weight = number of claims, init_score = log(train mean severity)",
                "base_log_severity": severity_lgbm.base_log_value,
                **severity_lgbm.training_info,
            },
        },
        "validation_metrics": {name: {"mean_gamma_deviance": value} for name, value in validation_severity.items()},
        "preferred_model": {"name": preferred_severity, "rule": "lower mean Gamma deviance on the validation split"},
        "test_metrics": severity_test,
        "expected_loss": expected_loss,
        "explainability": not_measured(
            "SHAP reason codes in this results file cover the frequency model only; the scoring API gives "
            "per-request reason codes for the severity model"
        ),
        "risk_bands": risk_bands["insurance_severity"],
        "scenario_sanity_checks": not_measured(NO_SEVERITY_MONOTONIC_EXPECTATION),
        "stability": not_measured(f"PSI needs a time field. {NO_TIME_FIELD}"),
        "out_of_time_test": not_measured(NO_TIME_FIELD),
        "bootstrap": bootstrap_settings,
        "limitations": run_limitations,
    }

    # ---- MLflow, saved models ----------------------------------------------------------------------------
    print("Logging to MLflow and saving models ...", flush=True)
    commit = git_commit()
    tags = {"dataset": DATASET_NAME, "git_commit": commit, "phase": "1"}
    with start_run(
        config.experiment, "frequency", tags, db_path=config.mlflow_db, artifact_dir=config.mlflow_artifacts
    ) as frequency_run:
        log_params(
            {
                "seed": seed,
                "n_resamples": n_resamples,
                "split_fractions": fremtpl2.SPLIT_FRACTIONS,
                "claim_count_cap": fremtpl2.CLAIM_COUNT_CAP,
                "exposure_cap": fremtpl2.EXPOSURE_CAP,
                "glm_min_level_count": config.min_level_count,
                "lgbm": config.frequency_params,
                "lgbm_best_iteration": frequency_lgbm.training_info["best_iteration"],
            }
        )
        log_metrics(estimates_only({"test": frequency_test, "validation": frequency_payload["validation_metrics"]}))
        for key in ("frequency_lorenz", "frequency_calibration"):
            log_artifact(figures[key], "figures")
        frequency_version = log_lightgbm_model(
            frequency_lgbm, "frequency_lgbm", FREQUENCY_REGISTERED_NAME, {**tags, "selected_on": "validation deviance"}
        )
    with start_run(
        config.experiment, "severity", tags, db_path=config.mlflow_db, artifact_dir=config.mlflow_artifacts
    ) as severity_run:
        log_params(
            {
                "seed": seed,
                "n_resamples": n_resamples,
                "severity_cap_quantile": fremtpl2.SEVERITY_CAP_QUANTILE,
                "severity_cap_value": cap_value,
                "glm_min_level_count": config.min_level_count,
                "lgbm": config.severity_params,
                "lgbm_best_iteration": severity_lgbm.training_info["best_iteration"],
            }
        )
        log_metrics(
            estimates_only(
                {
                    "test": severity_test,
                    "validation": severity_payload["validation_metrics"],
                    "expected_loss": {name: expected_loss[name] for name in ("glm", "lgbm", "served_pair")},
                }
            )
        )
        for key in ("severity_lorenz", "severity_calibration"):
            log_artifact(figures[key], "figures")
        severity_version = log_lightgbm_model(
            severity_lgbm, "severity_lgbm", SEVERITY_REGISTERED_NAME, {**tags, "selected_on": "validation deviance"}
        )

        model_version = {
            "frequency_lgbm": {"registered_model": FREQUENCY_REGISTERED_NAME, "version": frequency_version},
            "severity_lgbm": {"registered_model": SEVERITY_REGISTERED_NAME, "version": severity_version},
            "label": f"freq-lgbm v{frequency_version} / sev-lgbm v{severity_version} @ {commit[:12]}",
        }
        bundle.metadata = {
            "model_version": model_version,
            "git_commit": commit,
            "dataset": DATASET_NAME,
            "trained_on": "train split (60%), seed 42; early stopping on validation split",
            "prediction_units": {
                "frequency": "claims per policy-year",
                "severity": "capped claim amount per claim",
                "expected_loss_per_year": "frequency x severity",
            },
            "severity_cap_value": cap_value,
            "preferred_model": preferred,
            "risk_bands": risk_bands,
            "mlflow_runs": {"frequency": frequency_run.info.run_id, "severity": severity_run.info.run_id},
        }
        saved = insurance.save_models(bundle, config.models_dir)
        log_artifact(saved["metadata"], "models")
    timer.lap("MLflow and model files")

    mlflow_info = {
        "experiment": config.experiment,
        "tracking_uri": "sqlite:///mlruns.db (repo root)",
        "run_ids": {"frequency": frequency_run.info.run_id, "severity": severity_run.info.run_id},
        "registered_models": model_version,
    }
    artifacts = {
        "figures": {key: str(_relative(path)) for key, path in figures.items()},
        "models": {key: str(_relative(path)) for key, path in saved.items()},
        "data_notes": str(_relative(config.notes_path)),
    }
    for payload in (frequency_payload, severity_payload):
        payload["mlflow"] = mlflow_info
        payload["artifacts"] = artifacts
        payload["model_version"] = model_version
        payload["runtime_seconds"] = {"total": timer.total(), "stages": dict(timer.stages)}

    for section, payload in ((FREQUENCY_SECTION, frequency_payload), (SEVERITY_SECTION, severity_payload)):
        update_results(config.results_path, section, payload, datasets=[DATASET_NAME], checksum_log=config.checksum_log)
    write_data_notes(config.results_path, config.notes_path)
    timer.lap("results JSON and data notes")
    print(format_summary(frequency_payload, severity_payload), flush=True)
    return {FREQUENCY_SECTION: frequency_payload, SEVERITY_SECTION: severity_payload}


def _relative(path: Path) -> Path:
    """Path relative to the repo root when possible (results files should not contain home directories)."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT)
    except ValueError:
        return Path(path)


def format_summary(frequency: Mapping[str, Any], severity: Mapping[str, Any]) -> str:
    """Plain-text table of the headline test metrics, exactly as written to the JSON."""
    frequency_test = frequency["test_metrics"]
    severity_test = severity["test_metrics"]
    expected_loss = severity["expected_loss"]
    lines = [
        "",
        "freMTPL2 test metrics (estimate [95% bootstrap CI])",
        f"{'metric':<52} {'GLM':<32} {'LightGBM':<32} {'LightGBM - GLM (paired)':<32}",
    ]
    rows = [
        ("Frequency: mean Poisson deviance (exposure-weighted)", frequency_test, "mean_poisson_deviance"),
        ("Frequency: Lorenz Gini", frequency_test, "lorenz_gini"),
        ("Severity: mean Gamma deviance (claim-weighted)", severity_test, "mean_gamma_deviance"),
        ("Severity: Lorenz Gini", severity_test, "lorenz_gini"),
    ]
    for label, block, key in rows:
        lines.append(
            f"{label:<52} {_ci(block['glm'][key]):<32} {_ci(block['lgbm'][key]):<32} "
            f"{_ci(block['lgbm_minus_glm'][key]):<32}"
        )
    lines.append(
        f"{'Frequency: observed / predicted claims':<52} "
        f"{_ci(frequency_test['glm']['observed_to_predicted_claims']):<32} "
        f"{_ci(frequency_test['lgbm']['observed_to_predicted_claims']):<32}"
    )
    lines.append(
        f"{'Expected loss: observed / predicted (capped)':<52} "
        f"{_ci(expected_loss['glm']['observed_to_predicted']):<32} "
        f"{_ci(expected_loss['lgbm']['observed_to_predicted']):<32}"
    )
    lines.append(
        f"{'Expected loss: pure-premium Lorenz Gini':<52} {_ci(expected_loss['glm']['pure_premium_lorenz_gini']):<32} "
        f"{_ci(expected_loss['lgbm']['pure_premium_lorenz_gini']):<32}"
    )
    served = expected_loss["served_pair"]
    lines.append(
        f"{'Expected loss, served pair':<52} {served['frequency_model']} frequency x {served['severity_model']} "
        f"severity: observed / predicted {_ci(served['observed_to_predicted'])}, Gini "
        f"{_ci(served['pure_premium_lorenz_gini'])}"
    )
    lines.append(
        f"Total test loss: observed capped {expected_loss['glm']['total_observed_loss_capped']:,.0f}; "
        f"predicted GLM {expected_loss['glm']['total_predicted_loss']:,.0f}, "
        f"LightGBM {expected_loss['lgbm']['total_predicted_loss']:,.0f}"
    )
    lines.append(f"Runtime: {frequency['runtime_seconds']['total']:.1f} s")
    return "\n".join(lines)


def write_data_notes(results_path: Path, notes_path: Path) -> Path:
    """Render docs/insurance_data_notes.md from the insurance sections of the results JSON.

    Reading the numbers back from the JSON (rather than from variables in memory) guarantees that the notes
    show exactly what results/risk_metrics.json holds.
    """
    results = read_results(results_path)
    if FREQUENCY_SECTION not in results or SEVERITY_SECTION not in results:
        raise ValueError(f"{results_path} has no insurance sections yet; run the training first")
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text(data_notes_text(results), encoding="utf-8")
    return notes_path


def data_notes_text(results: Mapping[str, Any]) -> str:
    """The data notes for a full results JSON (also used by the test that the committed notes are current)."""
    frequency = results[FREQUENCY_SECTION]["results"]
    severity = results[SEVERITY_SECTION]["results"]
    return render_data_notes(
        frequency["data_quality"],
        frequency["split"],
        severity["large_loss_cap"],
        frequency["recorded_claim_count_sensitivity"],
        frequency["claimant_only_block_sensitivity"],
        generated_at=results[FREQUENCY_SECTION]["metadata"]["generated_at_utc"],
    )


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    """Command-line options (all optional; defaults are the real run)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=fremtpl2.DEFAULT_DATA_DIR)
    parser.add_argument("--results-path", type=Path, default=RESULTS_DIR / "risk_metrics.json")
    parser.add_argument("--n-resamples", type=int, default=N_RESAMPLES)
    parser.add_argument("--shap-sample-size", type=int, default=2000)
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Only re-render docs/insurance_data_notes.md from the existing results JSON (no training).",
    )
    args = parser.parse_args(argv)
    config = TrainConfig(
        data_dir=args.data_dir,
        results_path=args.results_path,
        n_resamples=args.n_resamples,
        shap_sample_size=args.shap_sample_size,
    )
    return config, args.notes_only


def main(argv: list[str] | None = None) -> int:
    """Entry point for `make train-insurance`."""
    config, notes_only = parse_args(argv)
    if notes_only:
        print(f"Wrote {write_data_notes(config.results_path, config.notes_path)}")
    else:
        run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
