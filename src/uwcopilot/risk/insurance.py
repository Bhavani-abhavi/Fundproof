"""Insurance pricing models on freMTPL2: claim frequency, claim severity and expected loss.

Frequency (claims per policy-year; only claims with a recorded amount count, see data/fremtpl2.py)
    - Poisson GLM baseline (statsmodels) with log(Exposure) as an offset. Equivalent formulation: a Poisson
      regression of ClaimNb / Exposure with sample_weight = Exposure gives exactly the same coefficients,
      because both maximise sum_i [ClaimNb_i * log(mu_i) - Exposure_i * mu_i] (tested in test_insurance.py).
    - LightGBM with objective="poisson". The offset goes in as init_score = log(Exposure) + base, where base
      is the log of the training claim frequency, so every tree corrects a policy's starting point instead of
      re-learning the portfolio average. LightGBM does not store init_score in the model, so prediction adds
      base back explicitly: frequency = exp(base + tree output) claims per year.

Severity (average claim amount, for policies with claim amounts)
    - Gamma GLM with a log link baseline vs LightGBM objective="gamma". Each claim is capped at the 99.5th
      percentile of training claim amounts before averaging. Weight = number of claims behind the average.

Expected loss per policy-year = predicted frequency x predicted severity (capped-amount basis).

GLM features are banded so every coefficient reads as a relativity (for example "drivers aged 18-20 have
X times the claim frequency of drivers aged 41-50"). LightGBM sees the raw rating features.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from uwcopilot.risk.bands import band_cutoffs
from uwcopilot.risk.bootstrap import bootstrap_ci, paired_bootstrap_diff
from uwcopilot.risk.explain import TreeShapExplainer, compute_tree_shap, global_importance, reason_codes
from uwcopilot.risk.metrics import lorenz_gini, mean_gamma_deviance, mean_poisson_deviance
from uwcopilot.risk.results import to_jsonable
from uwcopilot.risk.scenario import FeatureRange

GlmFamily = Literal["poisson", "gamma"]
ModelName = Literal["glm", "lgbm"]

# ---------------------------------------------------------------------------
# Feature engineering for the GLMs: bands for ages and power, logs for density
# ---------------------------------------------------------------------------

BONUS_MALUS_CAP = 150
DRIVER_AGE_EDGES = [-np.inf, 21, 26, 31, 41, 51, 71, np.inf]
DRIVER_AGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]
VEHICLE_AGE_EDGES = [-np.inf, 1, 3, 6, 11, 16, np.inf]
VEHICLE_AGE_LABELS = ["0", "1-2", "3-5", "6-10", "11-15", "16+"]
VEHICLE_POWER_EDGES = [-np.inf, 5, 6, 7, 8, 9, np.inf]
VEHICLE_POWER_LABELS = ["4", "5", "6", "7", "8", "9+"]

GLM_CATEGORICAL_COLUMNS = ["DrivAgeBand", "VehAgeBand", "VehPowerBand", "Area", "VehBrand", "VehGas", "Region"]
GLM_NUMERIC_COLUMNS = ["BonusMalusCapped", "LogDensity"]
DEFAULT_MIN_LEVEL_COUNT = 50

# ---------------------------------------------------------------------------
# LightGBM features and default parameters (fixed in advance; no hyperparameter search)
# ---------------------------------------------------------------------------

TREE_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density", "Area", "VehGas", "VehBrand", "Region"]
TREE_CODED_COLUMNS = ["Area", "VehGas", "VehBrand", "Region"]
TREE_CATEGORICAL_FEATURES = ["VehBrand", "Region"]

FREQUENCY_LGBM_PARAMS: dict[str, Any] = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}
SEVERITY_LGBM_PARAMS: dict[str, Any] = {
    **FREQUENCY_LGBM_PARAMS,
    "objective": "gamma",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 200,
}
MAX_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 100

# ---------------------------------------------------------------------------
# Plain-language reasons for SHAP reason codes (frequency model)
# ---------------------------------------------------------------------------

FREQUENCY_REASONS_INCREASE = {
    "BonusMalus": "Bonus-malus level points to a less favourable claims history",
    "DrivAge": "Driver age is in a range with more frequent claims",
    "VehAge": "Vehicle age is in a range with more frequent claims",
    "VehPower": "Vehicle power is in a range with more frequent claims",
    "Density": "Lives in a more densely populated area, where claims are more frequent",
    "Area": "Area class has more frequent claims than average",
    "Region": "Region has more frequent claims than average",
    "VehBrand": "Vehicle brand group has more frequent claims than average",
    "VehGas": "Fuel type is associated with more frequent claims",
}
FREQUENCY_REASONS_DECREASE = {
    "BonusMalus": "Bonus-malus level points to a favourable claims history",
    "DrivAge": "Driver age is in a range with less frequent claims",
    "VehAge": "Vehicle age is in a range with less frequent claims",
    "VehPower": "Vehicle power is in a range with less frequent claims",
    "Density": "Lives in a less densely populated area, where claims are less frequent",
    "Area": "Area class has less frequent claims than average",
    "Region": "Region has less frequent claims than average",
    "VehBrand": "Vehicle brand group has less frequent claims than average",
    "VehGas": "Fuel type is associated with less frequent claims",
}
SEVERITY_REASONS_INCREASE = {
    "BonusMalus": "Bonus-malus level is associated with larger claims",
    "DrivAge": "Driver age is in a range with larger average claims",
    "VehAge": "Vehicle age is in a range with larger average claims",
    "VehPower": "Vehicle power is in a range with larger average claims",
    "Density": "Population density is associated with larger average claims",
    "Area": "Area class has larger average claims than average",
    "Region": "Region has larger average claims than average",
    "VehBrand": "Vehicle brand group has larger average claims than average",
    "VehGas": "Fuel type is associated with larger average claims",
}
SEVERITY_REASONS_DECREASE = {
    "BonusMalus": "Bonus-malus level is associated with smaller claims",
    "DrivAge": "Driver age is in a range with smaller average claims",
    "VehAge": "Vehicle age is in a range with smaller average claims",
    "VehPower": "Vehicle power is in a range with smaller average claims",
    "Density": "Population density is associated with smaller average claims",
    "Area": "Area class has smaller average claims than average",
    "Region": "Region has smaller average claims than average",
    "VehBrand": "Vehicle brand group has smaller average claims than average",
    "VehGas": "Fuel type is associated with smaller average claims",
}

# Which rating feature each GLM design column belongs to (for per-feature reason codes from a GLM).
GLM_TERM_FEATURES = {
    "DrivAgeBand": "DrivAge",
    "VehAgeBand": "VehAge",
    "VehPowerBand": "VehPower",
    "BonusMalusCapped": "BonusMalus",
    "LogDensity": "Density",
    "Area": "Area",
    "VehBrand": "VehBrand",
    "VehGas": "VehGas",
    "Region": "Region",
}


def _bands(values: pd.Series, edges: list[float], labels: list[str], name: str) -> pd.Series:
    """Put numeric values into left-closed bands, e.g. [21, 26) -> "21-25". Missing values are an error."""
    if values.isna().any():
        raise ValueError(f"{name} has missing values")
    banded = pd.cut(values.astype(float), bins=edges, labels=labels, right=False)
    return banded.astype(str)


def glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the nine rating features into the banded / transformed columns the GLMs use."""
    features = pd.DataFrame(index=df.index)
    features["DrivAgeBand"] = _bands(df["DrivAge"], DRIVER_AGE_EDGES, DRIVER_AGE_LABELS, "DrivAge")
    features["VehAgeBand"] = _bands(df["VehAge"], VEHICLE_AGE_EDGES, VEHICLE_AGE_LABELS, "VehAge")
    features["VehPowerBand"] = _bands(df["VehPower"], VEHICLE_POWER_EDGES, VEHICLE_POWER_LABELS, "VehPower")
    for column in ["Area", "VehBrand", "VehGas", "Region"]:
        features[column] = df[column].astype(str)
    features["BonusMalusCapped"] = df["BonusMalus"].astype(float).clip(upper=BONUS_MALUS_CAP)
    features["LogDensity"] = np.log(df["Density"].astype(float).clip(lower=1.0))
    return features


@dataclass
class GlmDesign:
    """How GLM features become a numeric design matrix (one-hot columns with a reference level).

    levels maps each categorical column to its dummy levels; reference maps it to the reference level
    (the most common level in the fitting data, which gets no column). Levels seen fewer than
    min_level_count times when fitting, and levels never seen, are treated as the reference level.
    """

    levels: dict[str, list[str]]
    reference: dict[str, str]
    folded_levels: dict[str, list[str]]
    min_level_count: int

    @classmethod
    def fit(cls, features: pd.DataFrame, min_level_count: int = DEFAULT_MIN_LEVEL_COUNT) -> "GlmDesign":
        """Learn reference and dummy levels from training features (output of glm_features)."""
        levels, reference, folded = {}, {}, {}
        for column in GLM_CATEGORICAL_COLUMNS:
            counts = features[column].value_counts()
            ordered = sorted(counts.index, key=lambda level: (-counts[level], str(level)))
            reference[column] = str(ordered[0])
            levels[column] = sorted(str(level) for level in ordered[1:] if counts[level] >= min_level_count)
            folded[column] = sorted(str(level) for level in ordered[1:] if counts[level] < min_level_count)
        return cls(levels=levels, reference=reference, folded_levels=folded, min_level_count=min_level_count)

    @property
    def column_names(self) -> list[str]:
        """Design matrix columns: intercept, numeric terms, then one dummy per kept level."""
        dummies = [f"{column}={level}" for column in GLM_CATEGORICAL_COLUMNS for level in self.levels[column]]
        return ["const", *GLM_NUMERIC_COLUMNS, *dummies]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Build the float design matrix for rows of glm_features output."""
        columns: dict[str, np.ndarray] = {"const": np.ones(len(features))}
        for column in GLM_NUMERIC_COLUMNS:
            columns[column] = features[column].to_numpy(dtype=float)
        for column in GLM_CATEGORICAL_COLUMNS:
            values = features[column].astype(str).to_numpy()
            for level in self.levels[column]:
                columns[f"{column}={level}"] = (values == level).astype(float)
        return pd.DataFrame(columns, index=features.index)[self.column_names]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for the model metadata JSON."""
        return {
            "numeric_terms": GLM_NUMERIC_COLUMNS,
            "reference_levels": self.reference,
            "dummy_levels": self.levels,
            "folded_into_reference": self.folded_levels,
            "min_level_count": self.min_level_count,
            "bands": {
                "DrivAge": DRIVER_AGE_LABELS,
                "VehAge": VEHICLE_AGE_LABELS,
                "VehPower": VEHICLE_POWER_LABELS,
                "BonusMalus": f"linear, capped at {BONUS_MALUS_CAP}",
                "Density": "natural log",
            },
        }


@dataclass
class GlmModel:
    """A fitted log-link GLM: design + coefficients. predict_rate returns exp(X beta).

    For frequency, the rate is claims per policy-year (the offset is not part of the rate).
    For severity, the rate is the expected average claim amount.
    """

    family: GlmFamily
    design: GlmDesign
    coefficients: dict[str, float]
    standard_errors: dict[str, float]
    n_observations: int
    converged: bool
    design_means: dict[str, float] = field(default_factory=dict)

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted rate for rows that hold the nine rating features."""
        design_matrix = self.design.transform(glm_features(df))
        beta = np.array([self.coefficients[name] for name in self.design.column_names])
        return np.exp(design_matrix.to_numpy() @ beta)

    def contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Per-feature contributions on the log scale, relative to the average training row.

        Returns (values, base_value): values has one column per rating feature, in TREE_FEATURES order, and
        base_value + values.sum(axis=1) equals log(predicted rate) for every row. A design column contributes
        coefficient * (value - training mean of that column), and the columns of one feature (for example
        all DrivAgeBand dummies) are added together. Same idea as the credit scorecard's contributions.
        """
        if not self.design_means:
            raise ValueError("this GLM has no training design means; retrain it with the current code")
        names = self.design.column_names
        design_matrix = self.design.transform(glm_features(df)).to_numpy()
        beta = np.array([self.coefficients[name] for name in names])
        means = np.array([self.design_means[name] for name in names])
        per_column = (design_matrix - means) * beta
        values = np.zeros((len(df), len(TREE_FEATURES)))
        for column_index, name in enumerate(names):
            if name == "const":
                continue
            feature = GLM_TERM_FEATURES[name.split("=", 1)[0]]
            values[:, TREE_FEATURES.index(feature)] += per_column[:, column_index]
        return values, float(np.dot(means, beta))

    def coefficient_table(self) -> list[dict[str, float | str]]:
        """Coefficient, standard error and relativity exp(coefficient) per term."""
        return [
            {
                "term": name,
                "coefficient": self.coefficients[name],
                "std_error": self.standard_errors[name],
                "relativity": float(np.exp(self.coefficients[name])),
            }
            for name in self.design.column_names
        ]


def _fit_glm(
    family: GlmFamily,
    df: pd.DataFrame,
    target: np.ndarray,
    min_level_count: int,
    offset: np.ndarray | None = None,
    var_weights: np.ndarray | None = None,
) -> GlmModel:
    """Fit a log-link Poisson or Gamma GLM with statsmodels (IRLS) and keep only what prediction needs."""
    features = glm_features(df)
    design = GlmDesign.fit(features, min_level_count=min_level_count)
    design_matrix = design.transform(features)
    if family == "poisson":
        glm_family = sm.families.Poisson()
    else:
        glm_family = sm.families.Gamma(link=sm.families.links.Log())
    model = sm.GLM(target, design_matrix, family=glm_family, offset=offset, var_weights=var_weights)
    result = model.fit(maxiter=100)
    return GlmModel(
        family=family,
        design=design,
        coefficients={name: float(value) for name, value in result.params.items()},
        standard_errors={name: float(value) for name, value in result.bse.items()},
        n_observations=int(len(df)),
        converged=bool(getattr(result, "converged", True)),
        design_means={name: float(value) for name, value in design_matrix.mean(axis=0).items()},
    )


def fit_frequency_glm(train: pd.DataFrame, min_level_count: int = DEFAULT_MIN_LEVEL_COUNT) -> GlmModel:
    """Poisson GLM of ClaimNb with offset log(Exposure)."""
    return _fit_glm(
        "poisson",
        train,
        target=train["ClaimNb"].to_numpy(dtype=float),
        min_level_count=min_level_count,
        offset=np.log(train["Exposure"].to_numpy(dtype=float)),
    )


def fit_severity_glm(train: pd.DataFrame, min_level_count: int = DEFAULT_MIN_LEVEL_COUNT) -> GlmModel:
    """Gamma GLM (log link) of the capped average claim amount, weighted by the number of claims.

    var_weights = n_claims because the variance of an average of n claims is the claim variance / n.
    """
    return _fit_glm(
        "gamma",
        train,
        target=train["average_amount_capped"].to_numpy(dtype=float),
        min_level_count=min_level_count,
        var_weights=train["n_claims"].to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# LightGBM models
# ---------------------------------------------------------------------------


@dataclass
class TreeFeatureEncoder:
    """Turns the rating features into the all-numeric frame LightGBM and SHAP use.

    Text columns become integer codes in sorted order (Area A-F keeps its density order; VehGas becomes
    0/1). VehBrand and Region are declared as LightGBM categorical features. Categories never seen in
    training become NaN, which LightGBM treats as missing.
    """

    codes: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "TreeFeatureEncoder":
        """Learn one integer code per category from the training rows."""
        codes = {}
        for column in TREE_CODED_COLUMNS:
            levels = sorted(df[column].astype(str).unique())
            codes[column] = {level: index for index, level in enumerate(levels)}
        return cls(codes=codes)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Numeric frame with columns TREE_FEATURES, in that order."""
        encoded = pd.DataFrame(index=df.index)
        for column in TREE_FEATURES:
            if column in self.codes:
                encoded[column] = df[column].astype(str).map(self.codes[column]).astype(float)
            else:
                encoded[column] = df[column].astype(float)
        return encoded


@dataclass
class TreeModel:
    """A LightGBM booster plus what is needed to use it: the encoder and the base log value.

    predict_rate returns exp(base_log_value + tree output): claims per year for frequency, expected average
    claim amount for severity.
    """

    booster: lgb.Booster
    encoder: TreeFeatureEncoder
    base_log_value: float
    objective: str
    training_info: dict[str, Any] = field(default_factory=dict)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Tree output on the log scale, without base_log_value (this is what SHAP values add up to)."""
        return self.booster.predict(self.encoder.transform(df), raw_score=True)

    def predict_rate(self, df: pd.DataFrame) -> np.ndarray:
        """Predicted rate on the original scale."""
        return np.exp(self.base_log_value + self.predict_raw(df))

    def contributions(self, df: pd.DataFrame, explainer: TreeShapExplainer | None = None) -> tuple[np.ndarray, float]:
        """SHAP values per feature (TREE_FEATURES order) and a base value on the log scale.

        base_value + values.sum(axis=1) equals log(predicted rate): the base includes base_log_value, which
        LightGBM never sees because it enters through init_score. Pass a TreeShapExplainer built on
        self.booster to avoid rebuilding one per call.
        """
        shap_result = (explainer or TreeShapExplainer(self.booster)).explain(self.encoder.transform(df))
        return shap_result.values, self.base_log_value + shap_result.base_value


def _train_booster(
    params: Mapping[str, Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    train_init: np.ndarray,
    valid_x: pd.DataFrame,
    valid_y: np.ndarray,
    valid_init: np.ndarray,
    train_weight: np.ndarray | None = None,
    valid_weight: np.ndarray | None = None,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Train with early stopping on the validation set and keep only the trees up to the best iteration."""
    train_set = lgb.Dataset(
        train_x,
        label=train_y,
        weight=train_weight,
        init_score=train_init,
        categorical_feature=TREE_CATEGORICAL_FEATURES,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        valid_x, label=valid_y, weight=valid_weight, init_score=valid_init, reference=train_set, free_raw_data=False
    )
    evaluation_history: dict[str, Any] = {}
    booster = lgb.train(
        dict(params),
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.record_evaluation(evaluation_history),
        ],
    )
    best_iteration = int(booster.best_iteration) if booster.best_iteration > 0 else booster.current_iteration()
    metric_name = next(iter(evaluation_history["validation"]))
    history = evaluation_history["validation"][metric_name]
    trimmed = lgb.Booster(model_str=booster.model_to_string(num_iteration=best_iteration))
    info = {
        "best_iteration": best_iteration,
        "rounds_trained": len(history),
        "max_rounds": max_rounds,
        "early_stopping_rounds": early_stopping_rounds,
        "early_stopping_metric": metric_name,
        "best_validation_metric": float(history[best_iteration - 1]),
        "params": dict(params),
    }
    return trimmed, info


def fit_frequency_lgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> TreeModel:
    """LightGBM Poisson model of ClaimNb with init_score = log(Exposure) + log(training frequency)."""
    encoder = TreeFeatureEncoder.fit(train)
    base = float(np.log(train["ClaimNb"].sum() / train["Exposure"].sum()))
    booster, info = _train_booster(
        params or FREQUENCY_LGBM_PARAMS,
        encoder.transform(train),
        train["ClaimNb"].to_numpy(dtype=float),
        base + np.log(train["Exposure"].to_numpy(dtype=float)),
        encoder.transform(validation),
        validation["ClaimNb"].to_numpy(dtype=float),
        base + np.log(validation["Exposure"].to_numpy(dtype=float)),
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    return TreeModel(booster=booster, encoder=encoder, base_log_value=base, objective="poisson", training_info=info)


def fit_severity_lgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
    max_rounds: int = MAX_BOOST_ROUNDS,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
) -> TreeModel:
    """LightGBM Gamma model of the capped average claim amount, weighted by the number of claims."""
    encoder = TreeFeatureEncoder.fit(train)
    weights = train["n_claims"].to_numpy(dtype=float)
    base = float(np.log(np.average(train["average_amount_capped"], weights=weights)))
    booster, info = _train_booster(
        params or SEVERITY_LGBM_PARAMS,
        encoder.transform(train),
        train["average_amount_capped"].to_numpy(dtype=float),
        np.full(len(train), base),
        encoder.transform(validation),
        validation["average_amount_capped"].to_numpy(dtype=float),
        np.full(len(validation), base),
        train_weight=weights,
        valid_weight=validation["n_claims"].to_numpy(dtype=float),
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    return TreeModel(booster=booster, encoder=encoder, base_log_value=base, objective="gamma", training_info=info)


# ---------------------------------------------------------------------------
# Metrics with bootstrap confidence intervals
# ---------------------------------------------------------------------------


def poisson_deviance_metric(frequency: np.ndarray, predicted: np.ndarray, exposure: np.ndarray) -> float:
    """Exposure-weighted mean Poisson deviance of observed vs predicted claims per year."""
    return mean_poisson_deviance(frequency, predicted, sample_weight=exposure)


def gamma_deviance_metric(average_amount: np.ndarray, predicted: np.ndarray, n_claims: np.ndarray) -> float:
    """Claim-count-weighted mean Gamma deviance of observed vs predicted average claim amount."""
    return mean_gamma_deviance(average_amount, predicted, sample_weight=n_claims)


def gini_metric(observed_rate: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    """Lorenz Gini, or NaN when a (resampled) set has no claims at all, so the bootstrap skips that resample.

    bootstrap_ci leaves NaN results out of the interval and reports how many resamples were valid.
    """
    if float(np.sum(np.asarray(observed_rate, dtype=float) * np.asarray(weight, dtype=float))) <= 0:
        return float("nan")
    return lorenz_gini(observed_rate, predicted, weight)


def ratio_of_sums(observed: np.ndarray, predicted: np.ndarray) -> float:
    """sum(observed) / sum(predicted): 1.0 means the model's total matches the observed total."""
    return float(np.sum(observed) / np.sum(predicted))


def equal_weight_bins(predicted: np.ndarray, weights: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Bin index 0..n_bins-1 so each bin holds about the same total weight, ordered by prediction.

    Same rule as uwcopilot.risk.metrics.calibration_table (stable sort; a row goes to the bin that holds the
    cumulative weight before it), so the deciles here match that table.
    """
    order = np.argsort(predicted, kind="mergesort")
    sorted_weights = np.asarray(weights, dtype=float)[order]
    share_before = (np.cumsum(sorted_weights) - sorted_weights) / sorted_weights.sum()
    sorted_bins = np.minimum(np.floor(share_before * n_bins).astype(int), n_bins - 1)
    bins = np.empty_like(sorted_bins)
    bins[order] = sorted_bins
    return bins


def calibration_by_decile(
    observed_total: np.ndarray,
    weight: np.ndarray,
    predicted_rate: np.ndarray,
    n_bins: int = 10,
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Observed vs predicted totals per predicted-rate decile, with a bootstrap CI for the observed rate.

    For frequency: observed_total = claims, weight = exposure, predicted_rate = claims per year.
    For severity: observed_total = capped claim amount, weight = number of claims, predicted_rate = average
    amount. Deciles hold equal total weight. The CI resamples policies within the decile.
    """
    observed_total = np.asarray(observed_total, dtype=float)
    weight = np.asarray(weight, dtype=float)
    predicted_rate = np.asarray(predicted_rate, dtype=float)
    bins = equal_weight_bins(predicted_rate, weight, n_bins)
    rows = []
    for bin_id in range(n_bins):
        in_bin = bins == bin_id
        if not in_bin.any():
            continue
        observed = observed_total[in_bin]
        bin_weight = weight[in_bin]
        predicted = predicted_rate[in_bin] * bin_weight
        interval = bootstrap_ci(ratio_of_sums, observed, bin_weight, n_resamples=n_resamples, seed=seed)
        rows.append(
            {
                "decile": bin_id + 1,
                "n_policies": int(in_bin.sum()),
                "total_weight": float(bin_weight.sum()),
                "predicted_total": float(predicted.sum()),
                "observed_total": float(observed.sum()),
                "mean_predicted_rate": float(predicted.sum() / bin_weight.sum()),
                "mean_observed_rate": interval["estimate"],
                "mean_observed_rate_ci_low": interval["ci_low"],
                "mean_observed_rate_ci_high": interval["ci_high"],
                "observed_to_predicted": float(observed.sum() / predicted.sum()),
            }
        )
    return rows


def frequency_metrics(
    claims: np.ndarray, exposure: np.ndarray, predicted_frequency: np.ndarray, n_resamples: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Test metrics for one frequency model: Poisson deviance, Lorenz Gini, totals and deciles, all with CIs."""
    claims = np.asarray(claims, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    predicted_frequency = np.asarray(predicted_frequency, dtype=float)
    observed_frequency = claims / exposure
    predicted_claims = predicted_frequency * exposure
    return {
        "mean_poisson_deviance": bootstrap_ci(
            poisson_deviance_metric,
            observed_frequency,
            predicted_frequency,
            exposure,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "lorenz_gini": bootstrap_ci(
            gini_metric, observed_frequency, predicted_frequency, exposure, n_resamples=n_resamples, seed=seed
        ),
        "total_observed_claims": float(claims.sum()),
        "total_predicted_claims": float(predicted_claims.sum()),
        "observed_to_predicted_claims": bootstrap_ci(
            ratio_of_sums, claims, predicted_claims, n_resamples=n_resamples, seed=seed
        ),
        "calibration_by_decile": calibration_by_decile(
            claims, exposure, predicted_frequency, n_resamples=n_resamples, seed=seed
        ),
    }


def severity_metrics(
    average_amount: np.ndarray,
    n_claims: np.ndarray,
    predicted_severity: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Test metrics for one severity model: Gamma deviance, Lorenz Gini, totals and deciles, all with CIs."""
    average_amount = np.asarray(average_amount, dtype=float)
    n_claims = np.asarray(n_claims, dtype=float)
    predicted_severity = np.asarray(predicted_severity, dtype=float)
    observed_amount = average_amount * n_claims
    predicted_amount = predicted_severity * n_claims
    return {
        "mean_gamma_deviance": bootstrap_ci(
            gamma_deviance_metric, average_amount, predicted_severity, n_claims, n_resamples=n_resamples, seed=seed
        ),
        "lorenz_gini": bootstrap_ci(
            gini_metric, average_amount, predicted_severity, n_claims, n_resamples=n_resamples, seed=seed
        ),
        "total_observed_amount_capped": float(observed_amount.sum()),
        "total_predicted_amount": float(predicted_amount.sum()),
        "observed_to_predicted_amount": bootstrap_ci(
            ratio_of_sums, observed_amount, predicted_amount, n_resamples=n_resamples, seed=seed
        ),
        "calibration_by_decile": calibration_by_decile(
            observed_amount, n_claims, predicted_severity, n_resamples=n_resamples, seed=seed
        ),
    }


def paired_comparison(
    metric_fn: Callable[..., float],
    observed: np.ndarray,
    predicted_lgbm: np.ndarray,
    predicted_glm: np.ndarray,
    weight: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of metric(LightGBM) - metric(GLM) on the same resampled test rows."""
    return paired_bootstrap_diff(
        metric_fn, observed, predicted_lgbm, predicted_glm, weight, n_resamples=n_resamples, seed=seed
    )


def expected_loss_summary(
    observed_loss: np.ndarray,
    exposure: np.ndarray,
    predicted_frequency: np.ndarray,
    predicted_severity: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Total predicted vs observed loss on a set of policies (capped-amount basis), with bootstrap CIs.

    predicted loss per policy = frequency (claims/year) x exposure (years) x severity (amount per claim).
    observed loss per policy = sum of its capped claim amounts (0 when it has none). Because the frequency
    target only counts claims with a recorded amount, both sides describe the same claims, apart from the few
    policies whose number of amounts differs from the capped claim count (counted in the data-quality report
    as n_policies_where_severity_rows_differ_from_modelling_claimnb).
    """
    observed_loss = np.asarray(observed_loss, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    predicted_loss = (
        np.asarray(predicted_frequency, dtype=float) * exposure * np.asarray(predicted_severity, dtype=float)
    )
    return {
        "total_predicted_loss": float(predicted_loss.sum()),
        "total_observed_loss_capped": float(observed_loss.sum()),
        "observed_to_predicted": bootstrap_ci(
            ratio_of_sums, observed_loss, predicted_loss, n_resamples=n_resamples, seed=seed
        ),
        "pure_premium_lorenz_gini": bootstrap_ci(
            gini_metric,
            observed_loss / exposure,
            predicted_loss / exposure,
            exposure,
            n_resamples=n_resamples,
            seed=seed,
        ),
    }


# ---------------------------------------------------------------------------
# SHAP explanations for the LightGBM frequency model
# ---------------------------------------------------------------------------

EXAMPLE_PERCENTILES = (50, 90, 99)


def explain_frequency_model(
    model: TreeModel,
    policies: pd.DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
    top_k: int = 4,
    example_percentiles: tuple[int, ...] = EXAMPLE_PERCENTILES,
) -> dict[str, Any]:
    """Global SHAP importance and reason codes for example policies (LightGBM frequency model).

    SHAP values are on the log-frequency scale: a contribution of +0.10 multiplies the predicted frequency
    by exp(0.10), about 1.105. Example policies are the sampled rows whose predicted frequency is closest to
    the given percentiles of the sample (50th = typical, 90th = elevated, 99th = high risk).
    """
    encoded = model.encoder.transform(policies)
    shap_result = compute_tree_shap(model.booster, encoded, sample_size=sample_size, seed=seed)
    sample_rows = policies.loc[shap_result.data.index]
    raw_prediction = model.booster.predict(shap_result.data, raw_score=True)
    additivity_error = np.abs(shap_result.base_value + shap_result.values.sum(axis=1) - raw_prediction)
    predicted_frequency = np.exp(model.base_log_value + raw_prediction)

    examples = []
    for percentile in example_percentiles:
        target = np.percentile(predicted_frequency, percentile)
        position = int(np.argmin(np.abs(predicted_frequency - target)))
        row = sample_rows.iloc[position]
        values = [row[feature] for feature in shap_result.feature_names]
        examples.append(
            {
                "label": f"policy nearest the {percentile}th percentile of predicted frequency in the sample",
                "IDpol": int(row["IDpol"]),
                "predicted_frequency_per_year": float(predicted_frequency[position]),
                "features": {feature: _plain(row[feature]) for feature in shap_result.feature_names},
                "reasons_increase_risk": reason_codes(
                    shap_result.values[position],
                    shap_result.feature_names,
                    values,
                    FREQUENCY_REASONS_INCREASE,
                    top_k=top_k,
                    direction="increase_risk",
                ),
                "reasons_decrease_risk": reason_codes(
                    shap_result.values[position],
                    shap_result.feature_names,
                    values,
                    FREQUENCY_REASONS_DECREASE,
                    top_k=top_k,
                    direction="decrease_risk",
                ),
            }
        )
    return {
        "method": "shap.TreeExplainer on the LightGBM frequency booster; values in log claims-per-year units",
        "sample_size": int(len(shap_result.data)),
        "seed": seed,
        "shap_base_value": shap_result.base_value,
        "base_log_frequency_added_outside_trees": model.base_log_value,
        "max_additivity_error": float(additivity_error.max()),
        "global_importance": global_importance(shap_result.values, shap_result.feature_names),
        "examples": examples,
    }


def _plain(value: Any) -> Any:
    """numpy scalar -> plain Python value, for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Saving and loading the fitted models
# ---------------------------------------------------------------------------

FREQUENCY_GLM_FILE = "frequency_glm.joblib"
SEVERITY_GLM_FILE = "severity_glm.joblib"
FREQUENCY_LGBM_FILE = "frequency_lgbm.txt"
SEVERITY_LGBM_FILE = "severity_lgbm.txt"
METADATA_FILE = "metadata.json"


@dataclass
class InsuranceModels:
    """All four fitted models plus feature ranges (for scenario overrides) and metadata."""

    frequency_glm: GlmModel
    frequency_lgbm: TreeModel
    severity_glm: GlmModel
    severity_lgbm: TreeModel
    feature_ranges: dict[str, FeatureRange]
    metadata: dict[str, Any]

    def predict(self, df: pd.DataFrame, model: ModelName = "lgbm") -> pd.DataFrame:
        """Predicted frequency (claims/year), severity (capped amount/claim) and expected loss per year."""
        if model == "glm":
            frequency = self.frequency_glm.predict_rate(df)
            severity = self.severity_glm.predict_rate(df)
        elif model == "lgbm":
            frequency = self.frequency_lgbm.predict_rate(df)
            severity = self.severity_lgbm.predict_rate(df)
        else:
            raise ValueError("model must be 'glm' or 'lgbm'")
        return pd.DataFrame(
            {
                "predicted_frequency": frequency,
                "predicted_severity": severity,
                "expected_loss_per_year": frequency * severity,
            },
            index=df.index,
        )

    def model_for(self, target: Literal["frequency", "severity"], name: ModelName) -> GlmModel | TreeModel:
        """The fitted model for one target ("frequency" or "severity") and model name ("glm" or "lgbm")."""
        models: dict[tuple[str, str], GlmModel | TreeModel] = {
            ("frequency", "glm"): self.frequency_glm,
            ("frequency", "lgbm"): self.frequency_lgbm,
            ("severity", "glm"): self.severity_glm,
            ("severity", "lgbm"): self.severity_lgbm,
        }
        if (target, name) not in models:
            raise ValueError(f"no {name!r} model for target {target!r}")
        return models[(target, name)]


BAND_SCORES = {
    "frequency": "predicted claim frequency (claims per policy-year)",
    "severity": "predicted average claim amount (capped basis)",
}


def risk_bands_for(
    models: InsuranceModels, train: pd.DataFrame, preferred: Mapping[str, ModelName]
) -> dict[str, dict[str, Any]]:
    """Risk-band cut-offs for the preferred frequency and severity models, from their training predictions.

    Both are taken over every policy in the training split (the scoring API rates policies, so the bands say
    where a policy sits in the training portfolio). Keys are the API product names.
    """
    bands = {}
    for target in ("frequency", "severity"):
        name = preferred[target]
        scores = models.model_for(target, name).predict_rate(train)
        entry = band_cutoffs(scores, BAND_SCORES[target], "every policy in the training split")
        bands[f"insurance_{target}"] = {"model": name, **entry}
    return bands


def _tree_metadata(model: TreeModel) -> dict[str, Any]:
    """Everything about a TreeModel except the booster itself (which is saved as LightGBM text)."""
    return {
        "objective": model.objective,
        "base_log_value": model.base_log_value,
        "category_codes": model.encoder.codes,
        "features": TREE_FEATURES,
        "categorical_features": TREE_CATEGORICAL_FEATURES,
        "training_info": model.training_info,
    }


def save_models(models: InsuranceModels, models_dir: Path) -> dict[str, Path]:
    """Write GLMs (joblib), boosters (LightGBM text) and metadata.json (feature ranges, encoders, versions)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "frequency_glm": models_dir / FREQUENCY_GLM_FILE,
        "severity_glm": models_dir / SEVERITY_GLM_FILE,
        "frequency_lgbm": models_dir / FREQUENCY_LGBM_FILE,
        "severity_lgbm": models_dir / SEVERITY_LGBM_FILE,
        "metadata": models_dir / METADATA_FILE,
    }
    joblib.dump(models.frequency_glm, paths["frequency_glm"])
    joblib.dump(models.severity_glm, paths["severity_glm"])
    models.frequency_lgbm.booster.save_model(str(paths["frequency_lgbm"]))
    models.severity_lgbm.booster.save_model(str(paths["severity_lgbm"]))
    metadata = {
        **models.metadata,
        "feature_ranges": {name: feature.to_dict() for name, feature in models.feature_ranges.items()},
        "frequency_lgbm": _tree_metadata(models.frequency_lgbm),
        "severity_lgbm": _tree_metadata(models.severity_lgbm),
        "frequency_glm": {
            "design": models.frequency_glm.design.to_dict(),
            "coefficients": models.frequency_glm.coefficient_table(),
        },
        "severity_glm": {
            "design": models.severity_glm.design.to_dict(),
            "coefficients": models.severity_glm.coefficient_table(),
        },
        "files": {name: path.name for name, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(to_jsonable(metadata), allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return paths


def _load_tree_model(booster_path: Path, info: Mapping[str, Any]) -> TreeModel:
    """Rebuild a TreeModel from its LightGBM text file and its metadata entry."""
    return TreeModel(
        booster=lgb.Booster(model_file=str(booster_path)),
        encoder=TreeFeatureEncoder(codes={column: dict(codes) for column, codes in info["category_codes"].items()}),
        base_log_value=float(info["base_log_value"]),
        objective=str(info["objective"]),
        training_info=dict(info.get("training_info", {})),
    )


def load_models(models_dir: Path) -> InsuranceModels:
    """Load what save_models wrote."""
    models_dir = Path(models_dir)
    metadata = json.loads((models_dir / METADATA_FILE).read_text(encoding="utf-8"))
    feature_ranges = {name: FeatureRange.from_dict(entry) for name, entry in metadata["feature_ranges"].items()}
    return InsuranceModels(
        frequency_glm=joblib.load(models_dir / FREQUENCY_GLM_FILE),
        frequency_lgbm=_load_tree_model(models_dir / FREQUENCY_LGBM_FILE, metadata["frequency_lgbm"]),
        severity_glm=joblib.load(models_dir / SEVERITY_GLM_FILE),
        severity_lgbm=_load_tree_model(models_dir / SEVERITY_LGBM_FILE, metadata["severity_lgbm"]),
        feature_ranges=feature_ranges,
        metadata=metadata,
    )
