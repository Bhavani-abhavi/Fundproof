"""Fairness demonstration on UCI German Credit (Phase 1, part C).

Lending Club, the main credit dataset, has no protected attributes, so group fairness cannot be measured on
it. This module runs the same kind of credit pipeline on UCI German Credit, which has age and a
"personal status and sex" attribute, and measures:

- discrimination of each model: test AUC of the predicted probability of bad credit;
- approval rate per group;
- adverse impact ratio (AIR) = approval rate of the unprivileged group / approval rate of the privileged
  group, flagged when below 0.8 (the "4/5ths rule");
- equal opportunity difference = true positive rate of the unprivileged group minus that of the
  privileged group, where TPR is the share of truly good applicants who were approved;
- proxy strength: how well the model's own features predict group membership (test AUC). Leaving the
  protected attributes out of the features ("fairness through unawareness") does not stop other features
  from carrying the same information; this check measures how much they do.

Every metric has a percentile bootstrap 95% CI (1,000 resamples, seed 42). The test set is small (about 300
rows, and the smallest groups have a few dozen people), so the intervals are wide.

Pipeline (kept close to the credit model's framework):
1. Stratified 70/30 train/test split on the target, seed 42.
2. Two models on the same features: logistic regression (standardized numerics, one-hot categoricals) and
   LightGBM (one-hot categoricals, number of trees picked by 5-fold cross-validation with early stopping
   on the training set only).
3. Decision rule: approve if predicted P(bad) < threshold. The threshold is set on the TRAINING set so that
   the share approved equals the training good rate (about 70%). It uses out-of-fold predictions (5-fold,
   seed 42) rather than in-sample ones, because in-sample scores of a fitted model are over-confident and
   would put the cut-off in the wrong place. The test approval rate is then measured, not forced.
"""

from dataclasses import asdict, dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from uwcopilot.risk.bootstrap import bootstrap_ci, paired_bootstrap_diff
from uwcopilot.risk.data.german_credit import (
    BAD_COLUMN,
    CORRECTED_PERSONAL_STATUS_MEANING,
    EXCLUDED_FROM_FEATURES,
    GOOD_COLUMN,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    SEX_BY_CODE_CORRECTED,
    SEX_BY_CODE_GERMAN_DOC,
)
from uwcopilot.risk.fairness import FOUR_FIFTHS_THRESHOLD
from uwcopilot.risk.metrics import roc_auc
from uwcopilot.risk.results import not_measured

NOTE = "Demonstration on UCI German Credit; Lending Club has no protected attributes."
LOGISTIC = "logistic_regression"
LIGHTGBM = "lightgbm"


@dataclass(frozen=True)
class GroupComparison:
    """One fairness comparison: a group column, its privileged and unprivileged values, and its role.

    role is "primary" for the headline comparisons and "sensitivity" for alternative definitions.
    Rows whose group is neither value (for example "undetermined") are left out of that comparison.
    """

    key: str
    column: str
    privileged: str
    unprivileged: str
    role: str
    description: str


COMPARISONS: tuple[GroupComparison, ...] = (
    GroupComparison(
        key="sex",
        column="sex",
        privileged="male",
        unprivileged="female",
        role="primary",
        description=(
            "Sex as german.doc labels attribute 9 (A91/A93/A94 male, A92/A95 female). The corrected code table "
            "contradicts these labels; see data_quality_warning."
        ),
    ),
    GroupComparison(
        key="age_lt25",
        column="age_group_25",
        privileged=">=25",
        unprivileged="<25",
        role="primary",
        description="Age from attribute 13: under 25 vs 25 and over.",
    ),
    GroupComparison(
        key="age_le30",
        column="age_group_30",
        privileged=">30",
        unprivileged="<=30",
        role="sensitivity",
        description="Age from attribute 13: 30 and under vs over 30 (sensitivity check on the age cut-off).",
    ),
    GroupComparison(
        key="sex_corrected_codes",
        column="sex_corrected",
        privileged="male",
        unprivileged="female",
        role="sensitivity",
        description=(
            "Sex from the corrected code table (A91/A93 male, A94 female); A92 rows mix both sexes and are "
            "left out. Very few women remain, so this is the least precise comparison."
        ),
    ),
)


@dataclass(frozen=True)
class DemoConfig:
    """Settings for one run of the demo. Defaults are the values used for results/risk_metrics.json."""

    test_size: float = 0.3
    seed: int = 42
    n_resamples: int = 1000
    alpha: float = 0.05
    n_folds: int = 5
    logistic_c: float = 1.0
    logistic_max_iter: int = 1000
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 8
    lgbm_min_child_samples: int = 20
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 1.0
    lgbm_max_rounds: int = 1000
    lgbm_early_stopping_rounds: int = 50


# ---------------------------------------------------------------------------
# Data split and models
# ---------------------------------------------------------------------------


def split_train_test(frame: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seeded train/test split, stratified on the bad-credit label so both sets have the same bad rate."""
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame[BAD_COLUMN])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode the categorical features; standardize the numeric ones if scale_numeric is True.

    Codes not seen in training (for example a purpose code with no training rows) become all-zero columns.
    """
    numeric_step: Any = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_step, list(MODEL_NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(MODEL_CATEGORICAL_FEATURES),
            ),
        ]
    )


def build_logistic_model(config: DemoConfig) -> Pipeline:
    """Logistic regression (L2, C from config) on standardized numerics and one-hot categoricals."""
    model = LogisticRegression(C=config.logistic_c, max_iter=config.logistic_max_iter, random_state=config.seed)
    return Pipeline([("preprocess", build_preprocessor(scale_numeric=True)), ("model", model)])


def lightgbm_params(config: DemoConfig) -> dict[str, Any]:
    """LightGBM settings shared by the cross-validation and the final model (small trees for 700 rows).

    Single-threaded and deterministic so the same seed gives the same model. No class reweighting: the bad
    rate is 30%, and unweighted probabilities keep the approval threshold easy to interpret.
    """
    return {
        "objective": "binary",
        "learning_rate": config.lgbm_learning_rate,
        "num_leaves": config.lgbm_num_leaves,
        "min_child_samples": config.lgbm_min_child_samples,
        "colsample_bytree": config.lgbm_colsample_bytree,
        "reg_lambda": config.lgbm_reg_lambda,
        "random_state": config.seed,
        "deterministic": True,
        "n_jobs": 1,
        "verbose": -1,
    }


def choose_lightgbm_rounds(features: pd.DataFrame, bad: np.ndarray, config: DemoConfig) -> int:
    """Number of boosting rounds picked by stratified k-fold cross-validation with early stopping.

    Uses the training set only. The one-hot encoder is unsupervised (it only learns the category lists), so
    fitting it on the whole training set before the folds does not leak the target.
    """
    matrix = build_preprocessor(scale_numeric=False).fit_transform(features)
    labels = np.asarray(bad)
    splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    folds = list(splitter.split(matrix, labels))
    params = {**lightgbm_params(config), "metric": "binary_logloss"}
    history = lgb.cv(
        params,
        lgb.Dataset(matrix, label=labels),
        num_boost_round=config.lgbm_max_rounds,
        folds=folds,
        callbacks=[lgb.early_stopping(config.lgbm_early_stopping_rounds, verbose=False)],
    )
    return len(history["valid binary_logloss-mean"])


def build_lightgbm_model(n_estimators: int, config: DemoConfig) -> Pipeline:
    """LightGBM classifier with n_estimators trees on one-hot categoricals and raw numerics."""
    model = lgb.LGBMClassifier(n_estimators=n_estimators, **lightgbm_params(config))
    return Pipeline([("preprocess", build_preprocessor(scale_numeric=False)), ("model", model)])


def out_of_fold_bad_probability(
    model: Pipeline, features: pd.DataFrame, bad: np.ndarray, config: DemoConfig
) -> np.ndarray:
    """P(bad) for every training row from a model that did not see that row (stratified k-fold, seeded)."""
    folds = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    probabilities = cross_val_predict(clone(model), features, np.asarray(bad), cv=folds, method="predict_proba")
    return probabilities[:, 1]


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def approval_threshold(p_bad: np.ndarray, target_approval_rate: float) -> float:
    """Threshold t such that approving rows with P(bad) < t approves target_approval_rate of the rows.

    With k = round(target_approval_rate * n) rows to approve, t is the midpoint between the k-th and
    (k+1)-th smallest P(bad). If those two values are tied, slightly fewer than k rows are approved.
    """
    if not 0.0 <= target_approval_rate <= 1.0:
        raise ValueError("target_approval_rate must be between 0 and 1")
    ordered = np.sort(np.asarray(p_bad, dtype=float))
    n_rows = ordered.shape[0]
    if n_rows == 0:
        raise ValueError("p_bad is empty")
    n_approved = int(round(target_approval_rate * n_rows))
    if n_approved == 0:
        return float(ordered[0])
    if n_approved == n_rows:
        return float(np.nextafter(ordered[-1], np.inf))
    return float((ordered[n_approved - 1] + ordered[n_approved]) / 2)


def approve(p_bad: np.ndarray, threshold: float) -> np.ndarray:
    """1 = approved (favorable decision) when P(bad) < threshold, else 0."""
    return (np.asarray(p_bad, dtype=float) < threshold).astype(int)


# ---------------------------------------------------------------------------
# Group metrics that return NaN (instead of raising) when a bootstrap resample has an empty group
# ---------------------------------------------------------------------------


def approval_rate_in_group(selected: np.ndarray, groups: np.ndarray, group: str) -> float:
    """Share approved among rows of one group; NaN if the group has no rows."""
    in_group = np.asarray(groups) == group
    if not in_group.any():
        return float("nan")
    return float(np.asarray(selected)[in_group].mean())


def true_positive_rate_in_group(y_good: np.ndarray, selected: np.ndarray, groups: np.ndarray, group: str) -> float:
    """Share approved among the truly good applicants of one group; NaN if there are none."""
    truly_good_in_group = (np.asarray(groups) == group) & (np.asarray(y_good) == 1)
    if not truly_good_in_group.any():
        return float("nan")
    return float(np.asarray(selected)[truly_good_in_group].mean())


def adverse_impact_ratio_value(selected: np.ndarray, groups: np.ndarray, unprivileged: str, privileged: str) -> float:
    """Approval rate of the unprivileged group divided by that of the privileged group (NaN if undefined)."""
    privileged_rate = approval_rate_in_group(selected, groups, privileged)
    unprivileged_rate = approval_rate_in_group(selected, groups, unprivileged)
    if np.isnan(privileged_rate) or np.isnan(unprivileged_rate) or privileged_rate == 0:
        return float("nan")
    return unprivileged_rate / privileged_rate


def equal_opportunity_difference(
    y_good: np.ndarray, selected: np.ndarray, groups: np.ndarray, unprivileged: str, privileged: str
) -> float:
    """TPR(unprivileged) - TPR(privileged) among truly good applicants. Negative = unprivileged approved less."""
    return true_positive_rate_in_group(y_good, selected, groups, unprivileged) - true_positive_rate_in_group(
        y_good, selected, groups, privileged
    )


def auc_or_nan(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc, but NaN when a resample has only one class (so the bootstrap can skip it)."""
    labels = np.asarray(y_true)
    if labels.min() == labels.max():
        return float("nan")
    return roc_auc(labels, y_score)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ci(metric_fn: Any, *arrays: Any, config: DemoConfig) -> dict[str, Any]:
    """bootstrap_ci with the run's resample count, seed and alpha."""
    return bootstrap_ci(metric_fn, *arrays, n_resamples=config.n_resamples, seed=config.seed, alpha=config.alpha)


def evaluate_comparison(
    y_good: np.ndarray, selected: np.ndarray, groups: np.ndarray, comparison: GroupComparison, config: DemoConfig
) -> dict[str, Any]:
    """Approval rates, TPRs, adverse impact ratio and equal opportunity difference for one comparison."""
    groups = np.asarray(groups).astype(str)
    y_good = np.asarray(y_good).astype(int)
    selected = np.asarray(selected).astype(int)
    privileged, unprivileged = comparison.privileged, comparison.unprivileged

    group_details: dict[str, dict[str, Any]] = {}
    for group in (privileged, unprivileged):
        in_group = groups == group
        n_truly_good = int((in_group & (y_good == 1)).sum())
        group_details[group] = {
            "n": int(in_group.sum()),
            "n_truly_good": n_truly_good,
            "observed_good_rate": float(y_good[in_group].mean()) if in_group.any() else float("nan"),
            "approval_rate": _ci(
                lambda s, g, grp=group: approval_rate_in_group(s, g, grp), selected, groups, config=config
            ),
            "true_positive_rate": _ci(
                lambda y, s, g, grp=group: true_positive_rate_in_group(y, s, g, grp),
                y_good,
                selected,
                groups,
                config=config,
            ),
        }

    air = _ci(lambda s, g: adverse_impact_ratio_value(s, g, unprivileged, privileged), selected, groups, config=config)
    air.update(
        {
            "definition": f"approval rate ({unprivileged}) / approval rate ({privileged})",
            "four_fifths_threshold": FOUR_FIFTHS_THRESHOLD,
            "point_estimate_below_threshold": bool(air["estimate"] < FOUR_FIFTHS_THRESHOLD),
            "ci_entirely_below_threshold": bool(air["ci_high"] < FOUR_FIFTHS_THRESHOLD),
            "ci_contains_threshold": bool(air["ci_low"] <= FOUR_FIFTHS_THRESHOLD <= air["ci_high"]),
        }
    )
    eod = _ci(
        lambda y, s, g: equal_opportunity_difference(y, s, g, unprivileged, privileged),
        y_good,
        selected,
        groups,
        config=config,
    )
    eod["definition"] = (
        f"TPR ({unprivileged}) - TPR ({privileged}); TPR = share of truly good applicants approved. "
        "Negative means truly good applicants in the unprivileged group are approved less often."
    )
    eod["ci_excludes_zero"] = bool(eod["ci_high"] < 0 or eod["ci_low"] > 0)

    n_left_out = int((~np.isin(groups, [privileged, unprivileged])).sum())
    return {
        "column": comparison.column,
        "role": comparison.role,
        "description": comparison.description,
        "privileged_group": privileged,
        "unprivileged_group": unprivileged,
        "n_test_rows_left_out": n_left_out,
        "groups": group_details,
        "adverse_impact_ratio": air,
        "equal_opportunity_difference": eod,
    }


def proxy_strength(
    train: pd.DataFrame, test: pd.DataFrame, comparison: GroupComparison, config: DemoConfig
) -> dict[str, Any]:
    """How well the credit model's features predict membership of the unprivileged group (test AUC).

    A logistic regression with the same preprocessing as the credit model is trained to predict
    "row is in the unprivileged group" on the training set. AUC 0.5 means the features carry no information
    about the group; the higher the AUC, the more a model could rebuild the protected attribute from other
    features, so dropping it ("unawareness") does not make the model blind to it.
    """
    keep_values = [comparison.privileged, comparison.unprivileged]
    train_rows = train[train[comparison.column].astype(str).isin(keep_values)]
    test_rows = test[test[comparison.column].astype(str).isin(keep_values)]
    y_train = (train_rows[comparison.column].astype(str) == comparison.unprivileged).astype(int).to_numpy()
    y_test = (test_rows[comparison.column].astype(str) == comparison.unprivileged).astype(int).to_numpy()
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"test_auc": not_measured("train or test set lacks one of the two groups")}

    model = build_logistic_model(config)
    model.fit(train_rows[list(MODEL_FEATURES)], y_train)
    p_test = model.predict_proba(test_rows[list(MODEL_FEATURES)])[:, 1]

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_[0]
    order = np.argsort(-np.abs(coefficients))[:5]
    top_features = [{"feature": str(feature_names[i]), "coefficient": float(coefficients[i])} for i in order]
    return {
        "target": f"{comparison.column} == '{comparison.unprivileged}'",
        "model": "logistic regression on the credit model's features (protected attributes excluded)",
        "n_train": int(y_train.shape[0]),
        "n_test": int(y_test.shape[0]),
        "test_auc": _ci(auc_or_nan, y_test, p_test, config=config),
        "top_features_by_abs_coefficient": top_features,
        "coefficient_scale": "log-odds per standard deviation (numerics) or per category (one-hot)",
    }


def fit_and_decide(
    name: str, model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, config: DemoConfig
) -> dict[str, Any]:
    """Pick the threshold on out-of-fold train predictions, fit on all of train, and decide on test.

    Returns the threshold, train and test approval rates, test P(bad) and the test decisions.
    """
    x_train, x_test = train[list(MODEL_FEATURES)], test[list(MODEL_FEATURES)]
    bad_train = train[BAD_COLUMN].to_numpy()
    target_approval_rate = float(train[GOOD_COLUMN].mean())

    oof_p_bad = out_of_fold_bad_probability(model, x_train, bad_train, config)
    threshold = approval_threshold(oof_p_bad, target_approval_rate)
    fitted = clone(model).fit(x_train, bad_train)
    test_p_bad = fitted.predict_proba(x_test)[:, 1]
    return {
        "name": name,
        "model": fitted,
        "threshold": threshold,
        "target_approval_rate": target_approval_rate,
        "train_oof_approval_rate": float(approve(oof_p_bad, threshold).mean()),
        "train_oof_auc": roc_auc(bad_train, oof_p_bad),
        "test_p_bad": test_p_bad,
        "test_selected": approve(test_p_bad, threshold),
    }


def run_fairness_demo(frame: pd.DataFrame, config: DemoConfig | None = None) -> dict[str, Any]:
    """Run the whole demo on a frame from load_german_credit and return the results payload (plain dict)."""
    config = config or DemoConfig()
    train, test = split_train_test(frame, config.test_size, config.seed)
    bad_test = test[BAD_COLUMN].to_numpy()
    good_test = test[GOOD_COLUMN].to_numpy()

    lightgbm_rounds = choose_lightgbm_rounds(train[list(MODEL_FEATURES)], train[BAD_COLUMN].to_numpy(), config)
    candidates = {
        LOGISTIC: build_logistic_model(config),
        LIGHTGBM: build_lightgbm_model(lightgbm_rounds, config),
    }
    model_settings = {
        LOGISTIC: {
            "description": "logistic regression, L2 penalty; numerics standardized, categoricals one-hot",
            "C": config.logistic_c,
            "max_iter": config.logistic_max_iter,
        },
        LIGHTGBM: {
            "description": "LightGBM binary classifier; categoricals one-hot; trees chosen by CV early stopping",
            **{key: value for key, value in lightgbm_params(config).items() if key != "verbose"},
            "n_estimators_chosen_by_cv": lightgbm_rounds,
            "cv_max_rounds": config.lgbm_max_rounds,
            "cv_early_stopping_rounds": config.lgbm_early_stopping_rounds,
            "class_weighting": "none",
        },
    }

    decisions = {name: fit_and_decide(name, model, train, test, config) for name, model in candidates.items()}
    models: dict[str, Any] = {}
    for name, decision in decisions.items():
        fairness = {
            comparison.key: evaluate_comparison(
                good_test, decision["test_selected"], test[comparison.column].to_numpy(), comparison, config
            )
            for comparison in COMPARISONS
        }
        models[name] = {
            "settings": model_settings[name],
            "test_auc": _ci(auc_or_nan, bad_test, decision["test_p_bad"], config=config),
            "train_out_of_fold_auc": decision["train_oof_auc"],
            "threshold_p_bad": decision["threshold"],
            "target_approval_rate_from_train": decision["target_approval_rate"],
            "train_out_of_fold_approval_rate": decision["train_oof_approval_rate"],
            "test_approval_rate": _ci(np.mean, decision["test_selected"], config=config),
            "fairness": fairness,
        }

    auc_difference = paired_bootstrap_diff(
        auc_or_nan,
        bad_test,
        decisions[LIGHTGBM]["test_p_bad"],
        decisions[LOGISTIC]["test_p_bad"],
        n_resamples=config.n_resamples,
        seed=config.seed,
        alpha=config.alpha,
    )
    proxy = {comparison.key: proxy_strength(train, test, comparison, config) for comparison in COMPARISONS}

    payload: dict[str, Any] = {
        "note": NOTE,
        "dataset": dataset_summary(frame, train, test),
        "data_quality_warning": DATA_QUALITY_WARNING,
        "caveats": caveats(len(test), frame, test),
        "group_definitions": group_definitions(),
        "excluded_from_features": dict(EXCLUDED_FROM_FEATURES),
        "model_features": list(MODEL_FEATURES),
        "split": {
            "method": "random split stratified on the bad-credit label (German Credit has no date field)",
            "test_size": config.test_size,
            "seed": config.seed,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_good_rate": float(train[GOOD_COLUMN].mean()),
            "test_good_rate": float(test[GOOD_COLUMN].mean()),
        },
        "decision_rule": DECISION_RULE,
        "bootstrap": {
            "n_resamples": config.n_resamples,
            "seed": config.seed,
            "alpha": config.alpha,
            "method": "percentile; rows resampled jointly; resamples where a metric is undefined are skipped "
            "and counted in n_valid_resamples",
        },
        "config": asdict(config),
        "models": models,
        "auc_difference_lightgbm_minus_logistic": auc_difference,
        "proxy_check": proxy,
        "regulatory_context": REGULATORY_CONTEXT,
    }
    payload["summary_table"] = summary_tables(payload)
    return payload


# ---------------------------------------------------------------------------
# Descriptions written next to the numbers
# ---------------------------------------------------------------------------

DATA_QUALITY_WARNING = (
    "german.doc labels attribute 9 (personal status and sex) wrongly. The corrected code table published with "
    "the South German Credit data (Groemping 2019, Report 4/2019, Beuth University of Applied Sciences Berlin; "
    "UCI 2020, doi:10.24432/C5QG88) has the same row counts per code and gives: A91 male divorced/separated, "
    "A92 female non-single or male single, A93 male married/widowed, A94 female single. The 'sex' comparison "
    "follows german.doc (the convention in most fairness work) and so groups some single women as male and "
    "some single men as female; treat it as a demonstration of the method, not a measurement of sex "
    "disparity. 'sex_corrected_codes' uses only the unambiguous codes."
)

DECISION_RULE = {
    "rule": "approve if predicted P(bad) < threshold",
    "threshold_choice": (
        "set on the training set so that the share approved equals the training good rate; computed on "
        "out-of-fold P(bad) (stratified 5-fold, seed 42) so in-sample over-confidence does not move the "
        "cut-off; threshold = midpoint between the k-th and (k+1)-th smallest P(bad), k = round(rate * n)"
    ),
    "favorable_decision": "approve (selected = 1)",
    "truly_good": "target == 1 in german.data (good credit risk)",
}

REGULATORY_CONTEXT = {
    "four_fifths_rule": (
        "Uniform Guidelines on Employee Selection Procedures, 29 CFR 1607.4(D) (1978). An employment-selection "
        "rule of thumb, used here only as a screening statistic."
    ),
    "us_fair_lending": (
        "Equal Credit Opportunity Act, 15 U.S.C. 1691 (1974), and Regulation B, 12 CFR 1002.2(z), list sex, "
        "marital status and age among prohibited bases."
    ),
    "applicability": (
        "The German Credit data is from a German bank and is not subject to US law. Which tests apply depends "
        "on the institution and its risk profile. This is a demonstration, not a compliance program or a legal "
        "finding."
    ),
}


def dataset_summary(frame: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """Row counts and bad rates, so the reader can check the split."""
    return {
        "name": "UCI Statlog (German Credit Data)",
        "file": "german.data",
        "source": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
        "license": "CC BY 4.0",
        "n_rows": int(len(frame)),
        "bad_rate": float(frame[BAD_COLUMN].mean()),
        "personal_status_sex_counts": {
            str(code): int(count)
            for code, count in frame["personal_status_sex"].astype(str).value_counts().sort_index().items()
        },
    }


def caveats(n_test: int, frame: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Plain-language limitations, with the sizes taken from the data."""
    smallest = min(
        int((test[comparison.column].astype(str) == value).sum())
        for comparison in COMPARISONS
        for value in (comparison.privileged, comparison.unprivileged)
    )
    return [
        f"Small test set: {n_test} rows, and the smallest test group in any comparison has {smallest} people, "
        "so confidence intervals are wide and a point estimate below 0.8 is weak evidence on its own.",
        "Fairness through unawareness: sex, age and foreign-worker status are not model features, but other "
        "features can act as proxies (see proxy_check). Excluding a protected attribute does not remove proxy "
        "effects.",
        "Attribute 9 mixes sex with marital status, and german.doc mislabels it (see data_quality_warning).",
        f"The data has {len(frame)} loans from one German bank, 1973 to 1975 (UCI South German Credit page); "
        "it says nothing about any current lending population.",
        "Groups can differ in their observed good rates (see observed_good_rate per group), so a gap in approval "
        "rates may partly reflect that; AIR alone cannot separate risk differences from unfair treatment.",
    ]


def group_definitions() -> dict[str, Any]:
    """How each group column is built from the raw attributes."""
    return {
        "sex": {"source": "attribute 9, german.doc labels", "mapping": dict(SEX_BY_CODE_GERMAN_DOC)},
        "sex_corrected": {
            "source": "attribute 9, corrected code table (Groemping 2019; UCI South German Credit 2020)",
            "corrected_meaning": dict(CORRECTED_PERSONAL_STATUS_MEANING),
            "mapping": dict(SEX_BY_CODE_CORRECTED),
        },
        "age_group_25": {"source": "attribute 13 (age in years)", "groups": ["<25", ">=25"]},
        "age_group_30": {"source": "attribute 13 (age in years)", "groups": ["<=30", ">30"]},
        "comparisons": [asdict(comparison) for comparison in COMPARISONS],
    }


def summary_tables(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flat tables copied from the nested results, for printing and for the README/model card."""
    model_rows = []
    fairness_rows = []
    for model_name, model in payload["models"].items():
        model_rows.append(
            {
                "model": model_name,
                "test_auc": model["test_auc"]["estimate"],
                "test_auc_ci_low": model["test_auc"]["ci_low"],
                "test_auc_ci_high": model["test_auc"]["ci_high"],
                "threshold_p_bad": model["threshold_p_bad"],
                "train_oof_approval_rate": model["train_out_of_fold_approval_rate"],
                "test_approval_rate": model["test_approval_rate"]["estimate"],
            }
        )
        for key, result in model["fairness"].items():
            privileged = result["privileged_group"]
            unprivileged = result["unprivileged_group"]
            air = result["adverse_impact_ratio"]
            eod = result["equal_opportunity_difference"]
            fairness_rows.append(
                {
                    "model": model_name,
                    "comparison": key,
                    "role": result["role"],
                    "privileged": privileged,
                    "unprivileged": unprivileged,
                    "n_privileged": result["groups"][privileged]["n"],
                    "n_unprivileged": result["groups"][unprivileged]["n"],
                    "approval_rate_privileged": result["groups"][privileged]["approval_rate"]["estimate"],
                    "approval_rate_unprivileged": result["groups"][unprivileged]["approval_rate"]["estimate"],
                    "adverse_impact_ratio": air["estimate"],
                    "air_ci_low": air["ci_low"],
                    "air_ci_high": air["ci_high"],
                    "air_below_0_8": air["point_estimate_below_threshold"],
                    "air_ci_contains_0_8": air["ci_contains_threshold"],
                    "equal_opportunity_difference": eod["estimate"],
                    "eod_ci_low": eod["ci_low"],
                    "eod_ci_high": eod["ci_high"],
                }
            )
    proxy_rows = []
    for key, result in payload["proxy_check"].items():
        auc = result["test_auc"]
        if "estimate" not in auc:
            proxy_rows.append({"comparison": key, "proxy_auc": auc})
            continue
        proxy_rows.append(
            {
                "comparison": key,
                "proxy_auc": auc["estimate"],
                "proxy_auc_ci_low": auc["ci_low"],
                "proxy_auc_ci_high": auc["ci_high"],
                "n_test": result["n_test"],
            }
        )
    difference = payload["auc_difference_lightgbm_minus_logistic"]
    model_rows.append(
        {
            "model": "lightgbm_minus_logistic_regression",
            "test_auc": difference["estimate"],
            "test_auc_ci_low": difference["ci_low"],
            "test_auc_ci_high": difference["ci_high"],
        }
    )
    return {"models": model_rows, "fairness": fairness_rows, "proxy_check": proxy_rows}
