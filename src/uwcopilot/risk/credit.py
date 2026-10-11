"""Credit-default model on Lending Club: logistic scorecard vs LightGBM, validated out of time.

What the model predicts
    is_bad = 1 if a finished individual loan was charged off or defaulted, 0 if it was fully paid
    (see uwcopilot.risk.data.lending_club). The probability is a lifetime (whole-term) PD for loans that
    finished, not a 12-month PD as used for IFRS 9 stage 1 or Basel; the two must not be mixed up.

Out-of-time split (by issue year)
    train: issued 2015 or earlier; validation: 2016 (early stopping, recalibration, model choice);
    test: 2017-2018 (touched once, for the reported metrics). A random split would let the model learn from
    loans issued after the ones it is tested on, which never happens in real lending.

Models
    1. Logistic-regression scorecard (champion candidate, easy to explain): money amounts are log-transformed,
       every numeric feature is clipped to its 1st-99th training percentile, missing values are filled with
       the training median plus a missing-indicator column, then standardised. Categorical features are
       one-hot encoded (rare levels grouped). One coefficient per encoded column.
    2. LightGBM (challenger): raw features, categorical features as LightGBM categories. Class imbalance is
       handled with scale_pos_weight = (number of good loans) / (number of bad loans) in the training data.
       That weighting inflates the predicted probabilities, so the raw score is recalibrated on the 2016
       validation set with Platt scaling (a one-feature logistic regression on the raw log-odds). Early
       stopping uses validation AUC, which the weighting does not distort.

Model choice (fixed before looking at the test set)
    LightGBM is chosen only if its validation AUC beats the scorecard by at least MIN_AUC_GAIN_FOR_LIGHTGBM
    AND the lower end of the 95% paired-bootstrap interval of that difference is above 0. Otherwise the
    scorecard is chosen, because it is simpler to explain, validate and monitor.

Every reported number is computed here and written to results/risk_metrics.json by scripts/train_credit.py.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from uwcopilot.risk.bands import assign_band, band_cutoffs
from uwcopilot.risk.bootstrap import bootstrap_ci, paired_bootstrap_diff
from uwcopilot.risk.data.lending_club import (
    CATEGORICAL_FEATURES,
    FEATURES,
    MISSING_CATEGORY,
    NUMERIC_FEATURES,
    finished_loans,
)
from uwcopilot.risk.explain import compute_tree_shap, global_importance, reason_codes
from uwcopilot.risk.leakage import (
    BANNED_POST_ORIGINATION_COLUMNS,
    LENDER_OUTPUT_COLUMNS,
    TARGET_COLUMN,
    assert_no_leakage,
    base_feature_name,
)
from uwcopilot.risk.metrics import (
    brier_score,
    calibration_table,
    expected_calibration_error,
    gini_from_auc,
    ks_statistic,
    roc_auc,
)
from uwcopilot.risk.results import to_jsonable
from uwcopilot.risk.scenario import (
    FeatureRange,
    OverrideError,
    apply_overrides,
    build_feature_ranges,
    monotonic_check,
    validate_input_features,
)
from uwcopilot.risk.stability import (
    DEFAULT_PSI_THRESHOLD,
    EMPTY_BIN_SHARE,
    psi_band,
    psi_flag,
    quantile_bin_edges,
)

ModelName = Literal["logistic_scorecard", "lightgbm"]
LOGISTIC: ModelName = "logistic_scorecard"
LIGHTGBM: ModelName = "lightgbm"

CREDIT_FEATURES = FEATURES  # the allow-list; every name here passes assert_no_leakage
TRAIN_LAST_YEAR = 2015
VALIDATION_YEAR = 2016
TEST_YEARS = (2017, 2018)

MIN_AUC_GAIN_FOR_LIGHTGBM = 0.005
SELECTION_RULE = (
    "Chosen on the 2016 validation set, before the test set is scored: LightGBM only if its AUC exceeds the "
    f"logistic scorecard's by at least {MIN_AUC_GAIN_FOR_LIGHTGBM} and the lower end of the 95% paired-bootstrap "
    "interval of the difference is above 0; otherwise the logistic scorecard (simpler to explain and monitor)."
)

# Money amounts get log1p before clipping in the scorecard, so a coefficient reads "per standard deviation of
# log amount" and a few very large balances cannot dominate.
LOG_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "revol_bal",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_rev_hi_lim",
    "avg_cur_bal",
    "bc_open_to_buy",
    "delinq_amnt",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
]
LINEAR_FEATURES = [feature for feature in NUMERIC_FEATURES if feature not in LOG_FEATURES]

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbose": -1,
}

# Risk bands for the scoring interface come from quantiles of the chosen model's PDs on the training split
# (uwcopilot.risk.bands); they are not Lending Club grades and not a decision rule.
BAND_SCORE = "probability of default (chosen model)"
BAND_POPULATION = f"finished loans in the training split (issue year <= {TRAIN_LAST_YEAR})"

# Scenario sanity checks (build plan, Phase 1 D): features with a documented expected direction. The share of
# test loans that move the other way is reported, never asserted to be zero.
MONOTONIC_CHECKS: list[dict[str, Any]] = [
    {
        "feature": "dti",
        "step": 5.0,
        "expected": "increase",
        "why": "A higher debt-to-income ratio leaves less room to repay, so PD should not fall when it rises.",
    },
    {
        "feature": "fico_range_low",
        "step": 20.0,
        "expected": "decrease",
        "why": "A higher credit score reflects a better repayment record, so PD should not rise when it rises.",
    },
]

ECE_BINS = 10
EXAMPLE_PERCENTILES = (50, 90, 99)
PREDICT_BATCH_ROWS = 250_000
TOP_FEATURES_FOR_PSI = 10

# Plain-language reasons, phrased like adverse-action reasons (the condition that raised the estimated risk).
# Wording follows the style of the sample reasons in Regulation B (12 CFR part 1002), Appendix C, Form C-1.
REASON_MAP: dict[str, str] = {
    "loan_amnt": "Amount of credit requested",
    "term_months": "Longer repayment term requested",
    "annual_inc": "Income insufficient for amount of credit requested",
    "loan_to_income": "Amount of credit requested is high relative to income",
    "dti": "Excessive obligations in relation to income",
    "emp_length_years": "Length of employment",
    "fico_range_low": "Credit score",
    "credit_history_months": "Limited length of credit history",
    "delinq_2yrs": "Delinquent past or present credit obligations with others",
    "inq_last_6mths": "Number of recent inquiries on credit bureau report",
    "mths_since_last_delinq": "Time since most recent delinquency",
    "mths_since_last_record": "Time since most recent public record",
    "open_acc": "Number of open credit accounts",
    "pub_rec": "Derogatory public records",
    "revol_bal": "Balances on revolving accounts",
    "revol_util": "High utilization of revolving credit",
    "total_acc": "Number of credit accounts",
    "collections_12_mths_ex_med": "Collection action in the past 12 months",
    "mths_since_last_major_derog": "Time since most recent serious delinquency",
    "acc_now_delinq": "Accounts currently delinquent",
    "tot_coll_amt": "Amounts owed on collection accounts",
    "tot_cur_bal": "Total balances on credit accounts",
    "total_rev_hi_lim": "Total revolving credit limit",
    "acc_open_past_24mths": "Number of accounts opened in the past 24 months",
    "avg_cur_bal": "Average balance on credit accounts",
    "bc_open_to_buy": "Available credit on bank cards",
    "bc_util": "High utilization of bank card credit",
    "chargeoff_within_12_mths": "Charge-off on credit obligations in the past 12 months",
    "delinq_amnt": "Past-due amounts on delinquent accounts",
    "mo_sin_old_il_acct": "Length of installment credit history",
    "mo_sin_old_rev_tl_op": "Length of revolving credit history",
    "mo_sin_rcnt_rev_tl_op": "Time since a revolving account was opened",
    "mo_sin_rcnt_tl": "Time since a credit account was opened",
    "mort_acc": "Number of mortgage accounts",
    "mths_since_recent_bc": "Time since a bank card was opened",
    "mths_since_recent_bc_dlq": "Time since most recent bank card delinquency",
    "mths_since_recent_inq": "Time since most recent credit inquiry",
    "mths_since_recent_revol_delinq": "Time since most recent revolving delinquency",
    "num_accts_ever_120_pd": "Accounts ever 120 or more days past due",
    "num_actv_bc_tl": "Number of active bank card accounts",
    "num_actv_rev_tl": "Number of active revolving accounts",
    "num_bc_sats": "Number of bank card accounts in good standing",
    "num_bc_tl": "Number of bank card accounts",
    "num_il_tl": "Number of installment accounts",
    "num_op_rev_tl": "Number of open revolving accounts",
    "num_rev_accts": "Number of revolving accounts",
    "num_rev_tl_bal_gt_0": "Number of revolving accounts with a balance",
    "num_sats": "Number of accounts in good standing",
    "num_tl_120dpd_2m": "Accounts currently 120 or more days past due",
    "num_tl_30dpd": "Accounts currently 30 days past due",
    "num_tl_90g_dpd_24m": "Serious delinquency in the past 24 months",
    "num_tl_op_past_12m": "Number of accounts opened in the past 12 months",
    "pct_tl_nvr_dlq": "Share of accounts with a delinquency history",
    "percent_bc_gt_75": "Bank cards near their credit limit",
    "pub_rec_bankruptcies": "Bankruptcy",
    "tax_liens": "Tax liens",
    "tot_hi_cred_lim": "Total credit limit",
    "total_bal_ex_mort": "Total non-mortgage balances",
    "total_bc_limit": "Total bank card credit limit",
    "total_il_high_credit_limit": "Total installment credit limit",
    "home_ownership": "Housing status",
    "verification_status": "Income verification status",
    "purpose": "Purpose of the loan",
}

REASON_CODE_REFERENCES = [
    {
        "source": "Regulation B (Equal Credit Opportunity Act), 12 CFR 1002.9(b)(2) and Appendix C",
        "topic": "Adverse action notices must state the specific principal reasons; Appendix C has sample reasons.",
    },
    {
        "source": "CFPB Circular 2022-03, May 26, 2022",
        "topic": "Specific reasons are required even when the credit decision uses a complex algorithm.",
    },
    {
        "source": "CFPB Circular 2023-03, September 19, 2023",
        "topic": "Sample-form reasons cannot be used if they do not reflect the actual principal reasons.",
    },
]
REASON_CODE_CAVEAT = (
    "Demonstration only: reason codes describe how this model uses each feature (SHAP or scorecard "
    "contributions), not causal effects, and correlated features can share credit. Whether and how these "
    "references apply depends on the institution and its products; this is not a compliance program."
)

# loan_to_income is derived from two other features; scenario overrides keep it consistent (see score()).
DERIVED_FEATURE_SOURCES = {"loan_to_income": ("loan_amnt", "annual_inc")}


@dataclass
class CreditConfig:
    """Settings for one run of the credit pipeline. Defaults are the reported configuration."""

    n_resamples: int = 1000
    seed: int = 42
    alpha: float = 0.05
    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    clip_quantiles: tuple[float, float] = (0.01, 0.99)
    min_category_count: int = 50
    lgbm_params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    max_boost_rounds: int = 3000
    early_stopping_rounds: int = 100
    class_imbalance: Literal["scale_pos_weight", "none"] = "scale_pos_weight"
    min_auc_gain_for_lightgbm: float = MIN_AUC_GAIN_FOR_LIGHTGBM
    shap_sample_size: int = 2000
    psi_bins: int = 10
    psi_threshold: float = DEFAULT_PSI_THRESHOLD


# ---------------------------------------------------------------------------
# Out-of-time split
# ---------------------------------------------------------------------------


def out_of_time_split(loans: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split finished loans by issue year: train <= 2015, validation 2016, test 2017-2018.

    Loans issued after 2018 (none in the 2007-2018Q4 file) belong to no split.
    """
    year = loans["issue_year"]
    return {
        "train": loans.loc[year <= TRAIN_LAST_YEAR].reset_index(drop=True),
        "validation": loans.loc[year == VALIDATION_YEAR].reset_index(drop=True),
        "test": loans.loc[year.isin(TEST_YEARS)].reset_index(drop=True),
    }


def split_summary(splits: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Rows, bad loans, bad rate and issue years in each split."""
    summary: dict[str, Any] = {
        "rule": f"train: issue year <= {TRAIN_LAST_YEAR}; validation: {VALIDATION_YEAR}; "
        f"test: {TEST_YEARS[0]}-{TEST_YEARS[-1]} (finished loans only)",
    }
    for name, frame in splits.items():
        n_bad = int(frame[TARGET_COLUMN].sum())
        summary[name] = {
            "n_rows": int(len(frame)),
            "n_bad": n_bad,
            "bad_rate": n_bad / len(frame) if len(frame) else None,
            "issue_years": sorted(int(year) for year in frame["issue_year"].unique()),
        }
    return summary


def _check_split_usable(splits: Mapping[str, pd.DataFrame]) -> None:
    """Every split needs good and bad loans, or AUC and calibration cannot be computed."""
    for name, frame in splits.items():
        labels = set(frame[TARGET_COLUMN].unique().tolist())
        if labels != {0, 1}:
            raise ValueError(f"the {name} split needs both good and bad loans, found labels {sorted(labels)}")


# ---------------------------------------------------------------------------
# Logistic-regression scorecard
# ---------------------------------------------------------------------------


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to [lower, upper] quantiles learned on the training data. Missing values stay missing."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: Any, y: Any = None) -> "QuantileClipper":
        """Learn the clipping bounds from the non-missing values of each column."""
        values = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Clip to the learned bounds."""
        check_is_fitted(self, "lower_bounds_")
        values = validate_data(self, X, ensure_all_finite="allow-nan", dtype=float, reset=False)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def _log1p_non_negative(values: Any) -> Any:
    """log(1 + x) for amounts, treating negative amounts (data errors) as 0."""
    return np.log1p(np.clip(values, 0, None))


def _numeric_steps(config: CreditConfig, log_transform: bool) -> Pipeline:
    """Optional log, clip to training quantiles, median-impute with a missing indicator, standardise."""
    steps: list[tuple[str, Any]] = []
    if log_transform:
        steps.append(("log1p", FunctionTransformer(_log1p_non_negative, feature_names_out="one-to-one")))
    steps += [
        ("clip", QuantileClipper(*config.clip_quantiles)),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps)


def build_scorecard_pipeline(config: CreditConfig, features: list[str]) -> Pipeline:
    """Preprocessing + logistic regression for the given (already leakage-checked) feature list."""
    log_columns = [feature for feature in features if feature in LOG_FEATURES]
    linear_columns = [feature for feature in features if feature in LINEAR_FEATURES]
    categorical_columns = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    unknown = [f for f in features if f not in log_columns + linear_columns + categorical_columns]
    if unknown:
        raise ValueError(f"no preprocessing defined for features: {unknown}")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=config.min_category_count,
                    sparse_output=False,
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("log", _numeric_steps(config, log_transform=True), log_columns),
            ("num", _numeric_steps(config, log_transform=False), linear_columns),
            ("cat", categorical, categorical_columns),
        ]
    )
    model = LogisticRegression(C=config.logistic_c, max_iter=config.logistic_max_iter)
    return Pipeline([("prep", preprocess), ("model", model)])


def original_feature(encoded_name: str) -> str:
    """Map an encoded scorecard column ("cat__purpose_car", "num__missingindicator_dti") to its feature."""
    base = base_feature_name(encoded_name)
    if base in NUMERIC_FEATURES:
        return base
    matches = [feature for feature in CATEGORICAL_FEATURES if base.startswith(feature + "_")]
    if not matches:
        raise ValueError(f"cannot map encoded column '{encoded_name}' to a feature")
    return max(matches, key=len)


@dataclass
class LogisticScorecard:
    """Fitted scorecard pipeline plus what is needed to explain it.

    transformed_means holds the training mean of every encoded column. A row's contribution from one encoded
    column is coefficient * (value - training mean), so contributions are relative to an average training
    applicant and base_value + sum(contributions) equals the row's log-odds.
    """

    pipeline: Pipeline
    features: list[str]
    transformed_means: np.ndarray
    training_info: dict[str, Any] = field(default_factory=dict)

    def encoded_names(self) -> list[str]:
        """Names of the encoded columns that go into the logistic regression."""
        return [str(name) for name in self.pipeline.named_steps["prep"].get_feature_names_out()]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of default for each row (computed in batches to limit memory)."""
        return _in_batches(lambda part: self.pipeline.predict_proba(part[self.features])[:, 1], df)

    def coefficient_table(self) -> list[dict[str, Any]]:
        """Intercept and one coefficient per encoded column (standardised scale), largest |coefficient| first."""
        model = self.pipeline.named_steps["model"]
        rows = [
            {"term": name, "feature": original_feature(name), "coefficient": float(coefficient)}
            for name, coefficient in zip(self.encoded_names(), model.coef_[0], strict=True)
        ]
        rows.sort(key=lambda row: -abs(row["coefficient"]))
        return [{"term": "intercept", "feature": None, "coefficient": float(model.intercept_[0])}, *rows]

    def contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Per-feature log-odds contributions (n_rows, n_features) and the base value (average applicant)."""
        model = self.pipeline.named_steps["model"]
        encoded = self.pipeline.named_steps["prep"].transform(df[self.features])
        coefficients = model.coef_[0]
        per_column = (encoded - self.transformed_means) * coefficients
        owners = [original_feature(name) for name in self.encoded_names()]
        values = np.zeros((encoded.shape[0], len(self.features)))
        for column_index, owner in enumerate(owners):
            values[:, self.features.index(owner)] += per_column[:, column_index]
        base_value = float(model.intercept_[0] + np.dot(self.transformed_means, coefficients))
        return values, base_value


def fit_logistic_scorecard(
    train: pd.DataFrame, config: CreditConfig, features: list[str] | None = None
) -> LogisticScorecard:
    """Fit the scorecard on the training split. Raises LeakageError if a banned column is in features."""
    features = list(features or CREDIT_FEATURES)
    assert_no_leakage(features)
    pipeline = build_scorecard_pipeline(config, features)
    pipeline.fit(train[features], train[TARGET_COLUMN].to_numpy())
    encoded_names = [str(name) for name in pipeline.named_steps["prep"].get_feature_names_out()]
    assert_no_leakage(encoded_names)  # also check what the regression actually sees
    transformed_means = pipeline.named_steps["prep"].transform(train[features]).mean(axis=0)
    model = pipeline.named_steps["model"]
    info = {
        "n_encoded_columns": len(encoded_names),
        "n_iterations": int(model.n_iter_[0]),
        "max_iter": config.logistic_max_iter,
        "converged": bool(model.n_iter_[0] < config.logistic_max_iter),
        "C": config.logistic_c,
        "clip_quantiles": list(config.clip_quantiles),
        "min_category_count": config.min_category_count,
        "log_transformed_features": [f for f in features if f in LOG_FEATURES],
    }
    return LogisticScorecard(
        pipeline=pipeline, features=features, transformed_means=transformed_means, training_info=info
    )


# ---------------------------------------------------------------------------
# LightGBM with class weighting, early stopping and recalibration
# ---------------------------------------------------------------------------


@dataclass
class CategoryEncoder:
    """Integer codes for categorical features (sorted training levels); unseen levels become NaN (missing)."""

    codes: dict[str, dict[str, int]]

    @classmethod
    def fit(cls, df: pd.DataFrame, categorical: list[str]) -> "CategoryEncoder":
        """Learn one code per training level of each categorical feature."""
        codes = {}
        for column in categorical:
            levels = sorted(df[column].astype("str").unique())
            codes[column] = {level: index for index, level in enumerate(levels)}
        return cls(codes=codes)

    def transform(self, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """All-numeric frame with the given feature columns, in that order (index preserved)."""
        encoded = pd.DataFrame(index=df.index)
        for column in features:
            if column in self.codes:
                encoded[column] = df[column].astype("str").map(self.codes[column]).astype(float)
            else:
                encoded[column] = df[column].astype(float)
        return encoded


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Logistic function 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-values))


@dataclass
class LightGbmModel:
    """LightGBM booster with its category encoder and Platt recalibration (fitted on the validation year)."""

    booster: lgb.Booster
    encoder: CategoryEncoder
    features: list[str]
    calibration_slope: float
    calibration_intercept: float
    training_info: dict[str, Any] = field(default_factory=dict)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Model input frame (what SHAP explains)."""
        return self.encoder.transform(df, self.features)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Raw score (log-odds before recalibration); SHAP values add up to this."""
        return _in_batches(lambda part: self.booster.predict(self.encode(part), raw_score=True), df)

    def predict_uncalibrated_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Probability straight from the class-weighted booster (too high on average; shown for comparison)."""
        return _sigmoid(self.predict_raw(df))

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        """Recalibrated probability of default: sigmoid(slope * raw + intercept)."""
        return _sigmoid(self.calibration_slope * self.predict_raw(df) + self.calibration_intercept)


def fit_platt(raw_scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Fit P(bad) = sigmoid(slope * raw + intercept) by (effectively unpenalised) logistic regression.

    The slope must be positive, otherwise recalibration would reverse the risk ranking.
    """
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(np.asarray(raw_scores).reshape(-1, 1), np.asarray(labels))
    slope, intercept = float(model.coef_[0][0]), float(model.intercept_[0])
    if slope <= 0:
        raise ValueError(f"Platt recalibration found a non-positive slope ({slope}); the raw score does not rank risk")
    return slope, intercept


def fit_lightgbm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: CreditConfig,
    features: list[str] | None = None,
) -> LightGbmModel:
    """Train LightGBM with class weighting and early stopping on validation AUC, then recalibrate on validation.

    Raises LeakageError if a banned column is in features.
    """
    features = list(features or CREDIT_FEATURES)
    assert_no_leakage(features)
    categorical = [feature for feature in features if feature in CATEGORICAL_FEATURES]
    encoder = CategoryEncoder.fit(train, categorical)
    train_y = train[TARGET_COLUMN].to_numpy()
    valid_y = validation[TARGET_COLUMN].to_numpy()

    params = dict(config.lgbm_params)
    n_bad = int(train_y.sum())
    n_good = int(len(train_y) - n_bad)
    if config.class_imbalance == "scale_pos_weight":
        params["scale_pos_weight"] = n_good / n_bad

    train_x = encoder.transform(train, features)
    valid_x = encoder.transform(validation, features)
    assert_no_leakage(train_x.columns)
    train_set = lgb.Dataset(train_x, label=train_y, categorical_feature=categorical, free_raw_data=False)
    valid_set = lgb.Dataset(valid_x, label=valid_y, reference=train_set, free_raw_data=False)
    history: dict[str, Any] = {}
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[valid_set],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False), lgb.record_evaluation(history)],
    )
    best_iteration = int(booster.best_iteration) if booster.best_iteration > 0 else booster.current_iteration()
    trimmed = lgb.Booster(model_str=booster.model_to_string(num_iteration=best_iteration))
    assert_no_leakage(trimmed.feature_name())

    raw_validation = trimmed.predict(valid_x, raw_score=True)
    slope, intercept = fit_platt(raw_validation, valid_y)
    validation_auc_history = history["validation"]["auc"]
    info = {
        "class_imbalance": config.class_imbalance,
        "scale_pos_weight": params.get("scale_pos_weight"),
        "n_good_train": n_good,
        "n_bad_train": n_bad,
        "best_iteration": best_iteration,
        "rounds_trained": len(validation_auc_history),
        "max_rounds": config.max_boost_rounds,
        "early_stopping_rounds": config.early_stopping_rounds,
        "early_stopping_metric": "validation AUC (2016)",
        "best_validation_auc_lightgbm_internal": float(validation_auc_history[best_iteration - 1]),
        "recalibration": {
            "method": "Platt scaling on the 2016 validation set (same set used for early stopping)",
            "slope": slope,
            "intercept": intercept,
        },
        "params": params,
        "categorical_features": categorical,
    }
    return LightGbmModel(
        booster=trimmed,
        encoder=encoder,
        features=features,
        calibration_slope=slope,
        calibration_intercept=intercept,
        training_info=info,
    )


def _in_batches(predict: Callable[[pd.DataFrame], np.ndarray], df: pd.DataFrame) -> np.ndarray:
    """Apply predict to PREDICT_BATCH_ROWS rows at a time and join the results (keeps memory flat)."""
    parts = [predict(df.iloc[start : start + PREDICT_BATCH_ROWS]) for start in range(0, len(df), PREDICT_BATCH_ROWS)]
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Metrics with bootstrap confidence intervals
# ---------------------------------------------------------------------------


def gini_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    return gini_from_auc(roc_auc(y_true, y_score))


def ece_metric(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Expected calibration error with ECE_BINS equal-width bins."""
    return expected_calibration_error(y_true, y_prob, n_bins=ECE_BINS, strategy="uniform")


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, config: CreditConfig) -> dict[str, Any]:
    """AUC, KS, Gini, Brier and ECE with bootstrap CIs, plus the calibration table by predicted decile."""

    def ci(metric: Callable[..., float]) -> dict[str, Any]:
        return bootstrap_ci(
            metric, y_true, y_prob, n_resamples=config.n_resamples, seed=config.seed, alpha=config.alpha
        )

    return {
        "n_rows": int(len(y_true)),
        "n_bad": int(np.sum(y_true)),
        "observed_bad_rate": float(np.mean(y_true)),
        "mean_predicted_pd": float(np.mean(y_prob)),
        "auc": ci(roc_auc),
        "ks": ci(ks_statistic),
        "gini": ci(gini_metric),
        "brier": ci(brier_score),
        "ece": ci(ece_metric),
        "ece_bins": ECE_BINS,
        "calibration_by_predicted_decile": calibration_table(y_true, y_prob, n_bins=10).to_dict(orient="records"),
    }


def paired_auc_difference(
    y_true: np.ndarray, p_lightgbm: np.ndarray, p_logistic: np.ndarray, config: CreditConfig
) -> dict[str, Any]:
    """Paired bootstrap of AUC(LightGBM) - AUC(logistic scorecard) on the same rows."""
    result = paired_bootstrap_diff(
        roc_auc, y_true, p_lightgbm, p_logistic, n_resamples=config.n_resamples, seed=config.seed, alpha=config.alpha
    )
    result["definition"] = "AUC(LightGBM) - AUC(logistic scorecard), paired bootstrap on the same rows"
    return result


def choose_model(
    validation_difference: Mapping[str, Any], min_gain: float = MIN_AUC_GAIN_FOR_LIGHTGBM
) -> dict[str, Any]:
    """Apply SELECTION_RULE to the validation paired bootstrap result (never to test results)."""
    gain = float(validation_difference["estimate"])
    ci_low = float(validation_difference["ci_low"])
    lightgbm_wins = gain >= min_gain and ci_low > 0
    chosen = LIGHTGBM if lightgbm_wins else LOGISTIC
    reason = (
        f"validation AUC gain {gain:.4f} (95% CI low {ci_low:.4f}) "
        + ("meets" if lightgbm_wins else "does not meet")
        + f" the rule (gain >= {min_gain} and CI low > 0)"
    )
    return {"rule": SELECTION_RULE, "chosen": chosen, "reason": reason, "min_auc_gain": min_gain}


def sub_grade_score(sub_grade: pd.Series) -> pd.Series:
    """Lending Club sub-grade as an ordinal risk score: A1 = 1, A5 = 5, B1 = 6, ..., G5 = 35 (NaN if invalid)."""
    text = sub_grade.astype("str").str.strip().str.upper()
    letter = text.str[0]
    number = pd.to_numeric(text.str[1:], errors="coerce")
    letter_index = letter.map({grade: index for index, grade in enumerate("ABCDEFG")})
    score = letter_index * 5 + number
    valid = number.between(1, 5) & letter_index.notna()
    return score.where(valid).astype(float)


def lending_club_grade_benchmark(test: pd.DataFrame, config: CreditConfig) -> dict[str, Any]:
    """AUC of Lending Club's own sub-grade on the test loans: a benchmark, not one of our models.

    Lending Club set the sub-grade at origination with its own model and data we do not have, so this is a
    reference point for "how good is a production lender's grade on the same loans", not a fair contest.
    """
    score = sub_grade_score(test["sub_grade"])
    valid = score.notna().to_numpy()
    labels = test[TARGET_COLUMN].to_numpy()[valid]
    result = bootstrap_ci(
        roc_auc, labels, score.to_numpy()[valid], n_resamples=config.n_resamples, seed=config.seed, alpha=config.alpha
    )
    return {
        "score": "Lending Club sub_grade as an ordinal (A1 = 1 ... G5 = 35); higher = riskier",
        "n_rows_scored": int(valid.sum()),
        "n_rows_without_valid_sub_grade": int((~valid).sum()),
        "test_auc": result,
        "note": "Benchmark only. sub_grade is a Lending Club underwriting output and is never a model feature.",
    }


# ---------------------------------------------------------------------------
# Explainability: global importance and adverse-action-style reason codes
# ---------------------------------------------------------------------------


def risk_band(probability: float, bands: Mapping[str, Any]) -> str:
    """PD band from cut-offs made by uwcopilot.risk.bands.band_cutoffs on the training PDs."""
    return assign_band(probability, bands)


def training_risk_bands(train: pd.DataFrame, chosen_pd: Callable[[pd.DataFrame], np.ndarray]) -> dict[str, Any]:
    """Band cut-offs from the chosen model's PDs on the training split (see uwcopilot.risk.bands)."""
    return band_cutoffs(chosen_pd(train), BAND_SCORE, BAND_POPULATION)


def scenario_sanity_checks(
    test: pd.DataFrame,
    predictors: Mapping[str, Callable[[pd.DataFrame], np.ndarray]],
    ranges: Mapping[str, FeatureRange],
) -> dict[str, Any]:
    """Run MONOTONIC_CHECKS on the test loans for each model (see uwcopilot.risk.scenario.monotonic_check)."""
    checks = []
    for spec in MONOTONIC_CHECKS:
        if spec["feature"] not in ranges:
            continue
        entry: dict[str, Any] = {key: spec[key] for key in ("feature", "step", "expected", "why")}
        for name, predict in predictors.items():
            entry[name] = monotonic_check(
                predict, test, spec["feature"], spec["step"], ranges[spec["feature"]], spec["expected"]
            )
        checks.append(entry)
    return {
        "method": "shift one feature by step on every test loan that stays inside the training range (others "
        "are skipped, never clipped); a violation is a PD that moves against the expected direction",
        "population": "test split (finished loans issued 2017-2018)",
        "checks": checks,
    }


def _explanations(
    model_name: ModelName, logistic: LogisticScorecard, lightgbm: LightGbmModel, rows: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """(contributions, base value) in log-odds for rows: SHAP for LightGBM, scorecard contributions otherwise."""
    if model_name == LIGHTGBM:
        shap_result = compute_tree_shap(lightgbm.booster, lightgbm.encode(rows), sample_size=None)
        return shap_result.values, shap_result.base_value
    return logistic.contributions(rows)


def explainability_report(
    sample: pd.DataFrame,
    logistic: LogisticScorecard,
    lightgbm: LightGbmModel,
    chosen: ModelName,
    seed: int,
    bands: Mapping[str, Any],
) -> dict[str, Any]:
    """Global importance for both models, plus reason codes from the chosen model on a test sample.

    seed is the seed used to draw the sample (recorded in the output); bands are the training risk bands.
    """
    features = list(lightgbm.features)
    if list(logistic.features) != features:
        raise ValueError("both models must use the same features, in the same order")
    lightgbm_values, _ = _explanations(LIGHTGBM, logistic, lightgbm, sample)
    logistic_values, _ = _explanations(LOGISTIC, logistic, lightgbm, sample)
    chosen_values = lightgbm_values if chosen == LIGHTGBM else logistic_values
    chosen_pd = lightgbm.predict_pd(sample) if chosen == LIGHTGBM else logistic.predict_pd(sample)

    def codes_for(position: int) -> list[dict[str, Any]]:
        return reason_codes(chosen_values[position], features, sample[features].iloc[position].tolist(), REASON_MAP)

    examples = []
    for percentile in EXAMPLE_PERCENTILES:
        target = np.percentile(chosen_pd, percentile)
        position = int(np.argmin(np.abs(chosen_pd - target)))
        examples.append(
            {
                "pd_percentile_in_sample": percentile,
                "predicted_pd": float(chosen_pd[position]),
                "risk_band": risk_band(float(chosen_pd[position]), bands),
                "issue_year": int(sample["issue_year"].iloc[position]),
                "reason_codes": codes_for(position),
            }
        )

    riskiest = np.argsort(-chosen_pd)[: max(1, len(chosen_pd) // 10)]
    reason_counts: dict[str, int] = {}
    for position in riskiest:
        for code in codes_for(int(position)):
            reason_counts[code["reason"]] = reason_counts.get(code["reason"], 0) + 1
    frequency = [
        {"reason": reason, "n_applicants": count, "share_of_riskiest_decile": count / len(riskiest)}
        for reason, count in sorted(reason_counts.items(), key=lambda item: -item[1])
    ]
    return {
        "sample": {"n_rows": int(len(sample)), "drawn_from": "test split (2017-2018)", "seed": seed},
        "global_importance": {
            LIGHTGBM: {"method": "mean |SHAP| (log-odds)", "ranking": global_importance(lightgbm_values, features)},
            LOGISTIC: {
                "method": "mean |coefficient x (value - training mean)| (log-odds)",
                "ranking": global_importance(logistic_values, features),
            },
        },
        "reason_codes": {
            "model": chosen,
            "top_k": 4,
            "direction": "increase_risk",
            "examples": examples,
            "most_frequent_reasons_in_riskiest_decile": frequency,
            "features_without_plain_language_reason": [f for f in features if f not in REASON_MAP],
            "references": REASON_CODE_REFERENCES,
            "caveat": REASON_CODE_CAVEAT,
        },
    }


# ---------------------------------------------------------------------------
# Stability: PSI of scores and top features across issue-year vintages
# ---------------------------------------------------------------------------


MISSING_BIN = -1


def psi_bin_codes(values: pd.Series, reference: pd.Series, n_bins: int) -> np.ndarray:
    """Bin number of each value for PSI; missing values get MISSING_BIN (their own bin).

    Bins follow the same rule as stability.population_stability_index: interior edges at the quantiles of the
    reference's non-missing values, open outer edges, and a value equal to an edge goes to the bin above.
    """
    reference_numbers = reference.to_numpy(dtype=float)
    edges = quantile_bin_edges(reference_numbers[~np.isnan(reference_numbers)], n_bins)
    numbers = values.to_numpy(dtype=float)
    codes = np.full(numbers.shape[0], MISSING_BIN, dtype=np.int64)
    present = ~np.isnan(numbers)
    codes[present] = np.searchsorted(edges, numbers[present], side="right")
    return codes


def psi_from_counts(reference_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    """PSI from bin counts with the same formula and empty-bin floor as uwcopilot.risk.stability.

    sum over bins of (actual share - reference share) * ln(actual share / reference share), where an empty
    bin's share is replaced by EMPTY_BIN_SHARE. Tests check that this equals the stability module's functions.
    """
    reference_shares = reference_counts / reference_counts.sum()
    actual_shares = actual_counts / actual_counts.sum()
    reference_shares = np.where(reference_shares == 0, EMPTY_BIN_SHARE, reference_shares)
    actual_shares = np.where(actual_shares == 0, EMPTY_BIN_SHARE, actual_shares)
    return float(np.sum((actual_shares - reference_shares) * np.log(actual_shares / reference_shares)))


def psi_by_vintage(
    values: pd.Series, issue_year: pd.Series, categorical: bool, config: CreditConfig
) -> list[dict[str, Any]]:
    """PSI of each issue year against all loans issued in the training period (issue year <= 2015).

    Numeric variables use quantile bins of the reference period; categorical ones use one bin per level.
    Missing values always count as their own bin, so a field that is empty in a whole vintage (for example a
    bureau field not collected for the oldest loans) shows up as a shift instead of raising an error.
    Each value is binned and counted once (one table of bin x issue year), which keeps this fast on
    millions of loans.
    """
    reference_rows = (issue_year <= TRAIN_LAST_YEAR).to_numpy()
    if categorical:
        bins = values.astype("str").fillna("<missing>").to_numpy()
    else:
        bins = psi_bin_codes(values, values[reference_rows], config.psi_bins)
    counts = pd.crosstab(bins, issue_year.to_numpy())
    reference_counts = counts.loc[:, [year for year in counts.columns if year <= TRAIN_LAST_YEAR]].sum(axis=1)
    missing = values.isna().to_numpy()
    rows = []
    for year in counts.columns:
        value = psi_from_counts(reference_counts.to_numpy(dtype=float), counts[year].to_numpy(dtype=float))
        in_year = (issue_year == year).to_numpy()
        rows.append(
            {
                "issue_year": int(year),
                "n_rows": int(in_year.sum()),
                "missing_share": float(missing[in_year].mean()),
                "psi": value,
                "band": psi_band(value),
                "flag": psi_flag(value, config.psi_threshold),
            }
        )
    return rows


def stability_report(
    loans: pd.DataFrame,
    scores: Mapping[str, np.ndarray],
    top_features: list[str],
    config: CreditConfig,
) -> dict[str, Any]:
    """PSI of each model's PD and of the top features, per issue year, against the training period.

    Uses every individual loan (finished or not): PSI compares populations and needs no outcome, so it is
    not distorted by the finished-loan filter that shapes the 2017-2018 test set.
    """
    issue_year = loans["issue_year"].reset_index(drop=True)
    score_psi = {
        name: psi_by_vintage(pd.Series(values), issue_year, categorical=False, config=config)
        for name, values in scores.items()
    }
    feature_psi = {
        feature: psi_by_vintage(
            loans[feature].reset_index(drop=True), issue_year, feature in CATEGORICAL_FEATURES, config
        )
        for feature in top_features
    }
    flags = [
        {"variable": f"score:{name}", "issue_year": row["issue_year"], "psi": row["psi"]}
        for name, rows in score_psi.items()
        for row in rows
        if row["flag"]
    ] + [
        {"variable": feature, "issue_year": row["issue_year"], "psi": row["psi"]}
        for feature, rows in feature_psi.items()
        for row in rows
        if row["flag"]
    ]
    return {
        "reference": f"all individual loans issued in {TRAIN_LAST_YEAR} or earlier (finished or not)",
        "population": "all individual loans per issue year, finished or not (PSI needs no outcome)",
        "threshold": config.psi_threshold,
        "bands": "stable < 0.10 <= moderate shift <= 0.25 < significant shift",
        "n_bins": config.psi_bins,
        "score_psi_by_vintage": score_psi,
        "top_features": top_features,
        "top_features_source": "chosen model's global importance",
        "feature_psi_by_vintage": feature_psi,
        "flags_above_threshold": flags,
    }


# ---------------------------------------------------------------------------
# The fitted models, scoring with scenario overrides, saving and loading
# ---------------------------------------------------------------------------

LOGISTIC_FILE = "logistic_scorecard.joblib"
LIGHTGBM_FILE = "lightgbm.txt"
METADATA_FILE = "metadata.json"


@dataclass
class CreditModels:
    """Both fitted models, which one was chosen, and the feature ranges allowed in scenario overrides."""

    logistic: LogisticScorecard
    lightgbm: LightGbmModel
    chosen: ModelName
    feature_ranges: dict[str, FeatureRange]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_pd(self, df: pd.DataFrame, model: ModelName | None = None) -> np.ndarray:
        """Probability of default from the chosen model (or the named one)."""
        name = model or self.chosen
        if name == LIGHTGBM:
            return self.lightgbm.predict_pd(df)
        if name == LOGISTIC:
            return self.logistic.predict_pd(df)
        raise ValueError(f"model must be '{LOGISTIC}' or '{LIGHTGBM}'")

    def reason_codes(self, df: pd.DataFrame, model: ModelName | None = None, top_k: int = 4) -> list[list[dict]]:
        """Top risk-increasing reasons for every row."""
        name = model or self.chosen
        values, _ = _explanations(name, self.logistic, self.lightgbm, df)
        features = self.lightgbm.features if name == LIGHTGBM else self.logistic.features
        return [
            reason_codes(values[i], features, df[features].iloc[i].tolist(), REASON_MAP, top_k=top_k)
            for i in range(len(df))
        ]

    def _row_frame(self, row: Mapping[str, Any]) -> pd.DataFrame:
        """One-row frame with every feature (missing numeric values as NaN, missing categories as 'missing')."""
        frame = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in CREDIT_FEATURES}])
        for feature in NUMERIC_FEATURES:
            frame[feature] = frame[feature].astype(float)
        for feature in CATEGORICAL_FEATURES:
            frame[feature] = frame[feature].where(frame[feature].notna(), MISSING_CATEGORY).astype("str")
        return frame

    @property
    def risk_bands(self) -> dict[str, Any]:
        """Band cut-offs from the training PDs (stored in the metadata by run_credit_pipeline)."""
        if "risk_bands" not in self.metadata:
            raise ValueError("model metadata has no risk_bands; retrain with the current code")
        return self.metadata["risk_bands"]

    def _score_row(self, row: Mapping[str, Any], model: ModelName) -> dict[str, Any]:
        """PD, risk band and top-4 reasons for one row."""
        frame = self._row_frame(row)
        probability = float(self.predict_pd(frame, model)[0])
        return {
            "pd": probability,
            "risk_band": risk_band(probability, self.risk_bands),
            "reason_codes": self.reason_codes(frame, model)[0],
        }

    def score(
        self,
        features: Mapping[str, Any],
        overrides: Mapping[str, Any] | None = None,
        model: ModelName | None = None,
    ) -> dict[str, Any]:
        """Score one applicant, and again after scenario overrides (validated, never clipped).

        features must hold every feature in CREDIT_FEATURES. A value may be null (None or NaN) only for a
        feature that was missing in some training loans (for example mths_since_last_delinq for an applicant
        with no delinquency); every other value must lie inside its training range or be a category seen in
        training (scenario.validate_input_features), the same rule the insurance models follow.

        Raises LeakageError if a banned column is passed, InputFeaturesError (a ValueError) for unknown,
        missing or out-of-range features, and OverrideError for an invalid override. If loan_amnt or
        annual_inc is overridden, loan_to_income is recomputed and must also stay inside its training range.
        """
        assert_no_leakage(list(features) + list(overrides or {}))
        self.check_features(features)
        name = model or self.chosen
        row = {feature: features.get(feature, np.nan) for feature in CREDIT_FEATURES}
        scenario = None
        if overrides:
            scenario_row = apply_overrides(row, overrides, self.feature_ranges)
            scenario_row = self._recompute_derived(scenario_row, overrides)
            scenario = self._score_row(scenario_row, name)
        return {
            "model": name,
            "model_version": self.metadata.get("model_version"),
            "baseline": self._score_row(row, name),
            "overrides": dict(overrides or {}),
            "scenario": scenario,
        }

    def check_features(self, features: Mapping[str, Any]) -> None:
        """Raise InputFeaturesError unless features is one complete applicant inside the training data."""
        validate_input_features(features, {name: self.feature_ranges[name] for name in CREDIT_FEATURES})

    def _recompute_derived(self, row: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Keep derived features consistent with overridden inputs, and range-check the new value."""
        for derived, sources in DERIVED_FEATURE_SOURCES.items():
            touched = [source for source in sources if source in overrides]
            if not touched:
                continue
            if derived in overrides:
                raise OverrideError(f"override {derived} or its inputs {list(sources)}, not both")
            amount, income = (float(row[source]) for source in sources)
            value = amount / income if income > 0 else float("nan")
            apply_overrides(row, {derived: value}, self.feature_ranges)  # raises if out of the training range
            row[derived] = value
        return row


def save_credit_models(models: CreditModels, models_dir: Path) -> dict[str, Path]:
    """Write the scorecard (joblib), the booster (LightGBM text) and metadata.json (ranges, encoders, choice)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        LOGISTIC: models_dir / LOGISTIC_FILE,
        LIGHTGBM: models_dir / LIGHTGBM_FILE,
        "metadata": models_dir / METADATA_FILE,
    }
    joblib.dump(models.logistic, paths[LOGISTIC])
    models.lightgbm.booster.save_model(str(paths[LIGHTGBM]))
    metadata = {
        **models.metadata,
        "chosen_model": models.chosen,
        "features": CREDIT_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "banned_post_origination_columns": sorted(BANNED_POST_ORIGINATION_COLUMNS),
        "lender_output_columns": sorted(LENDER_OUTPUT_COLUMNS),
        "feature_ranges": {name: feature.to_dict() for name, feature in models.feature_ranges.items()},
        "derived_feature_sources": DERIVED_FEATURE_SOURCES,
        LIGHTGBM: {
            "category_codes": models.lightgbm.encoder.codes,
            "features": models.lightgbm.features,
            "calibration_slope": models.lightgbm.calibration_slope,
            "calibration_intercept": models.lightgbm.calibration_intercept,
            "training_info": models.lightgbm.training_info,
        },
        LOGISTIC: {
            "training_info": models.logistic.training_info,
            "coefficients": models.logistic.coefficient_table(),
        },
        "files": {name: path.name for name, path in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(to_jsonable(metadata), allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return paths


def load_credit_models(models_dir: Path) -> CreditModels:
    """Load what save_credit_models wrote."""
    models_dir = Path(models_dir)
    metadata = json.loads((models_dir / METADATA_FILE).read_text(encoding="utf-8"))
    info = metadata[LIGHTGBM]
    lightgbm = LightGbmModel(
        booster=lgb.Booster(model_file=str(models_dir / LIGHTGBM_FILE)),
        encoder=CategoryEncoder(codes={column: dict(codes) for column, codes in info["category_codes"].items()}),
        features=list(info["features"]),
        calibration_slope=float(info["calibration_slope"]),
        calibration_intercept=float(info["calibration_intercept"]),
        training_info=dict(info["training_info"]),
    )
    return CreditModels(
        logistic=joblib.load(models_dir / LOGISTIC_FILE),
        lightgbm=lightgbm,
        chosen=metadata["chosen_model"],
        feature_ranges={name: FeatureRange.from_dict(entry) for name, entry in metadata["feature_ranges"].items()},
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------

LIMITATIONS = [
    "Finished loans only, with right censoring: loans issued in 2017-2018 had at most about two years to finish "
    "before the end of the data, so the finished ones are mostly early payoffs and early charge-offs. Test bad "
    "rates and calibration describe that finished subset, not the full 2017-2018 vintages (see vintages).",
    "Lifetime PD on finished loans, not a 12-month PD (IFRS 9 stage 1 / Basel) and not a through-the-cycle PD.",
    "Lending Club data covers approved and issued loans only (no reject inference), so the model never sees "
    "applicants Lending Club declined.",
    "Platt recalibration, early stopping and the champion/challenger choice all use the 2016 validation set, "
    "which slightly favours LightGBM in that choice (its number of trees was tuned on the same loans); the "
    "2017-2018 test set is used only once, for the reported metrics, and its paired AUC difference is the "
    "unbiased check of the choice.",
    "Column decisions rely on Lending Club's data dictionary; bureau fields cannot be independently verified "
    "as point-in-time at application.",
    "No protected attributes are available in Lending Club data, so fairness cannot be measured here; see the "
    "separate fairness_demo section (UCI German Credit). Geography (zip_code, addr_state) is excluded as a proxy.",
]


def _feature_ranges(train: pd.DataFrame, features: list[str]) -> dict[str, FeatureRange]:
    """Allowed scenario values learned from the training split (term_months is treated as categorical: 36 or 60)."""
    categorical = [f for f in features if f in CATEGORICAL_FEATURES] + (
        ["term_months"] if "term_months" in features else []
    )
    return build_feature_ranges(train[features], categorical_columns=categorical)


def run_credit_pipeline(
    loans: pd.DataFrame,
    config: CreditConfig | None = None,
    dataset_label: str = "Lending Club accepted loans 2007-2018Q4 (Kaggle wordsforthewise/lending-club)",
    quality: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], CreditModels]:
    """Fit both models, choose one on validation, and measure everything on the test split.

    loans: all individual loans from prepare_loans (finished and unfinished). Returns (payload, models);
    payload is the JSON-ready "credit" section of results/risk_metrics.json.
    """
    config = config or CreditConfig()
    assert_no_leakage(CREDIT_FEATURES)
    modelling = finished_loans(loans)
    splits = out_of_time_split(modelling)
    _check_split_usable(splits)
    train, validation, test = splits["train"], splits["validation"], splits["test"]

    logistic = fit_logistic_scorecard(train, config)
    lightgbm = fit_lightgbm(train, validation, config)

    y_validation = validation[TARGET_COLUMN].to_numpy()
    validation_difference = paired_auc_difference(
        y_validation, lightgbm.predict_pd(validation), logistic.predict_pd(validation), config
    )
    selection = choose_model(validation_difference, config.min_auc_gain_for_lightgbm)
    selection["validation_auc_difference"] = validation_difference

    y_test = test[TARGET_COLUMN].to_numpy()
    p_logistic = logistic.predict_pd(test)
    p_lightgbm = lightgbm.predict_pd(test)
    p_lightgbm_uncalibrated = lightgbm.predict_uncalibrated_pd(test)

    chosen_model = lightgbm if selection["chosen"] == LIGHTGBM else logistic
    bands = training_risk_bands(train, chosen_model.predict_pd)
    sample = test.sample(n=min(config.shap_sample_size, len(test)), random_state=config.seed)
    explainability = explainability_report(sample, logistic, lightgbm, selection["chosen"], config.seed, bands)
    chosen_ranking = explainability["global_importance"][selection["chosen"]]["ranking"]
    top_features = [entry["feature"] for entry in chosen_ranking[:TOP_FEATURES_FOR_PSI]]
    scores = {LOGISTIC: logistic.predict_pd(loans), LIGHTGBM: lightgbm.predict_pd(loans)}

    models = CreditModels(
        logistic=logistic,
        lightgbm=lightgbm,
        chosen=selection["chosen"],
        feature_ranges=_feature_ranges(train, CREDIT_FEATURES),
        metadata={
            "dataset": dataset_label,
            "config": asdict(config),
            "selection_rule": SELECTION_RULE,
            "risk_bands": {"model": selection["chosen"], **bands},
        },
    )
    sanity = scenario_sanity_checks(
        test, {LOGISTIC: logistic.predict_pd, LIGHTGBM: lightgbm.predict_pd}, models.feature_ranges
    )
    payload = {
        "status": "measured",
        "dataset": dataset_label,
        "target_definition": {
            "is_bad_1": "Charged Off, Default, or 'Does not meet the credit policy. Status:Charged Off'",
            "is_bad_0": "Fully Paid, or 'Does not meet the credit policy. Status:Fully Paid'",
            "excluded": "Current, In Grace Period, Late (16-30 days), Late (31-120 days)",
            "population": "individual applications only",
            "pd_horizon": "lifetime of the loan (finished loans), not 12 months",
        },
        "data_quality": dict(quality) if quality is not None else None,
        "split": split_summary(splits),
        "features": {
            "used": CREDIT_FEATURES,
            "n_used": len(CREDIT_FEATURES),
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "lender_outputs_excluded": sorted(LENDER_OUTPUT_COLUMNS),
            "banned_post_origination": sorted(BANNED_POST_ORIGINATION_COLUMNS),
            "leakage_check": "passed: assert_no_leakage on raw features, scorecard encoded columns and booster inputs",
            "decisions_document": "docs/leakage.md",
        },
        "models": {
            LOGISTIC: {
                "description": "logistic regression on log/clipped/standardised numerics + one-hot categoricals",
                "training": logistic.training_info,
                "top_coefficients": logistic.coefficient_table()[:21],
                "test": binary_metrics(y_test, p_logistic, config),
            },
            LIGHTGBM: {
                "description": "LightGBM, scale_pos_weight class weighting, early stopping and Platt recalibration "
                "on 2016",
                "training": lightgbm.training_info,
                "test": binary_metrics(y_test, p_lightgbm, config),
                "test_before_recalibration_point_estimates": {
                    "mean_predicted_pd": float(np.mean(p_lightgbm_uncalibrated)),
                    "brier": brier_score(y_test, p_lightgbm_uncalibrated),
                    "ece": ece_metric(y_test, p_lightgbm_uncalibrated),
                },
            },
        },
        "comparison": {"test_auc_difference": paired_auc_difference(y_test, p_lightgbm, p_logistic, config)},
        "model_selection": selection,
        "benchmark_lending_club_sub_grade": lending_club_grade_benchmark(test, config),
        "explainability": explainability,
        "stability": stability_report(loans, scores, top_features, config),
        "scenario_feature_ranges": {name: feature.to_dict() for name, feature in models.feature_ranges.items()},
        "scenario_sanity_checks": sanity,
        "risk_bands": models.metadata["risk_bands"],
        "bootstrap": {
            "n_resamples": config.n_resamples,
            "seed": config.seed,
            "alpha": config.alpha,
            "method": "percentile",
        },
        "config": asdict(config),
        "limitations": LIMITATIONS,
    }
    return payload, models
