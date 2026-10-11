"""Tests for uwcopilot.risk.explain: reason codes by hand, SHAP on a tiny synthetic LightGBM model."""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.explain import ShapResult, compute_tree_shap, global_importance, reason_codes

FEATURES = ["dti", "income", "delinq_2yrs", "revol_util", "emp_length"]
VALUES = [35.0, 40000, 2, 0.9, 10]
SHAP_ROW = [0.40, -0.30, 0.25, 0.10, 0.0]
REASON_MAP = {
    "dti": "Debt-to-income ratio is high",
    "income": "Income level",
    "delinq_2yrs": "Recent delinquencies on file",
}


def test_reason_codes_increase_risk_sorted_and_mapped() -> None:
    reasons = reason_codes(SHAP_ROW, FEATURES, VALUES, REASON_MAP, top_k=4)
    assert [reason["feature"] for reason in reasons] == ["dti", "delinq_2yrs", "revol_util"]
    assert reasons[0] == {
        "feature": "dti",
        "value": 35.0,
        "contribution": pytest.approx(0.40),
        "reason": "Debt-to-income ratio is high",
    }
    # revol_util has no mapping: the fallback names the feature so the gap is visible.
    assert reasons[2]["reason"] == "revol_util increased the estimated risk"


def test_reason_codes_top_k_limits_the_list() -> None:
    reasons = reason_codes(SHAP_ROW, FEATURES, VALUES, REASON_MAP, top_k=1)
    assert [reason["feature"] for reason in reasons] == ["dti"]


def test_reason_codes_decrease_risk() -> None:
    reasons = reason_codes(SHAP_ROW, FEATURES, VALUES, REASON_MAP, direction="decrease_risk")
    assert [reason["feature"] for reason in reasons] == ["income"]
    assert reasons[0]["contribution"] == pytest.approx(-0.30)


def test_reason_codes_converts_numpy_values() -> None:
    reasons = reason_codes(np.array([0.5]), ["dti"], np.array([np.float64(20.0)]), REASON_MAP)
    assert type(reasons[0]["value"]) is float


def test_reason_codes_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        reason_codes([0.1, 0.2], ["dti"], [1.0], REASON_MAP)
    with pytest.raises(ValueError, match="direction"):
        reason_codes([0.1], ["dti"], [1.0], REASON_MAP, direction="sideways")  # type: ignore[arg-type]


def _synthetic_credit_data(n_rows: int = 600) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic data where default depends on x_strong a lot, x_weak a little, and noise not at all."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {
            "x_strong": rng.normal(size=n_rows),
            "x_weak": rng.normal(size=n_rows),
            "noise": rng.normal(size=n_rows),
        }
    )
    logit = 2.0 * features["x_strong"] + 0.5 * features["x_weak"]
    target = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return features, target


def test_compute_tree_shap_on_lightgbm_classifier_is_additive() -> None:
    features, target = _synthetic_credit_data()
    model = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(features, target)
    result = compute_tree_shap(model, features, sample_size=100, seed=1)
    assert isinstance(result, ShapResult)
    assert result.values.shape == (100, 3)
    assert result.feature_names == ["x_strong", "x_weak", "noise"]
    # SHAP values plus the base value add up to the raw (log-odds) prediction of each explained row.
    raw_prediction = model.predict(result.data, raw_score=True)
    assert result.values.sum(axis=1) + result.base_value == pytest.approx(raw_prediction, abs=1e-6)


def test_compute_tree_shap_poisson_regressor_and_full_data() -> None:
    features, _ = _synthetic_credit_data(300)
    counts = np.random.default_rng(1).poisson(np.exp(0.5 * features["x_strong"]))
    model = lgb.LGBMRegressor(objective="poisson", n_estimators=20, random_state=0, verbose=-1).fit(features, counts)
    result = compute_tree_shap(model, features, sample_size=None)
    assert result.values.shape == (300, 3)
    raw_prediction = model.predict(features, raw_score=True)
    assert result.values.sum(axis=1) + result.base_value == pytest.approx(raw_prediction, abs=1e-6)


def test_global_importance_ranks_the_strong_feature_first() -> None:
    features, target = _synthetic_credit_data()
    model = lgb.LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(features, target)
    result = compute_tree_shap(model, features, sample_size=200)
    importance = global_importance(result.values, result.feature_names)
    assert importance[0]["feature"] == "x_strong"
    assert importance[0]["mean_abs_shap"] > importance[-1]["mean_abs_shap"]


def test_global_importance_by_hand() -> None:
    values = np.array([[1.0, -2.0], [-3.0, 0.0]])
    assert global_importance(values, ["a", "b"]) == [
        {"feature": "a", "mean_abs_shap": 2.0},
        {"feature": "b", "mean_abs_shap": 1.0},
    ]
