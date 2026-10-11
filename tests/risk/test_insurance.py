"""Tests for uwcopilot.risk.insurance and scripts/train_insurance.py on SYNTHETIC freMTPL2-shaped data."""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor

import train_insurance
from uwcopilot.risk import insurance
from uwcopilot.risk.data.fremtpl2 import (
    CATEGORICAL_FEATURES,
    FREQUENCY_FILE,
    RATING_FEATURES,
    SEVERITY_FILE,
    prepare_fremtpl2,
    severity_cap,
    severity_table,
    split_policies,
    synthetic_fremtpl2_frames,
)
from uwcopilot.risk.metrics import calibration_table, lorenz_gini
from uwcopilot.risk.scenario import OverrideError, apply_overrides, build_feature_ranges

# One thread: on tiny synthetic data, OpenMP threads only add contention (and slow the suite under load).
SMALL_FREQUENCY_PARAMS = {**insurance.FREQUENCY_LGBM_PARAMS, "min_data_in_leaf": 50, "num_threads": 1}
SMALL_SEVERITY_PARAMS = {**insurance.SEVERITY_LGBM_PARAMS, "min_data_in_leaf": 20, "num_threads": 1}


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    """Synthetic policies split 60/20/20 (same code path as the real data)."""
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=6000, seed=1))
    result = split_policies(data.policies, seed=42)
    cap = severity_cap(data.claims.loc[data.claims["IDpol"].isin(result["train"]["IDpol"]), "ClaimAmount"])
    for name in list(result):
        result[f"severity_{name}"] = severity_table(result[name], data.claims, cap)
    return result


@pytest.fixture(scope="module")
def frequency_lgbm(splits: dict[str, pd.DataFrame]) -> insurance.TreeModel:
    """Small LightGBM frequency model on the synthetic data."""
    return insurance.fit_frequency_lgbm(
        splits["train"], splits["validation"], params=SMALL_FREQUENCY_PARAMS, max_rounds=300, early_stopping_rounds=30
    )


# ---------------------------------------------------------------------------
# GLM features and design
# ---------------------------------------------------------------------------


def test_glm_features_bands_and_transforms() -> None:
    df = pd.DataFrame(
        {
            "DrivAge": [18, 21, 50, 51, 71],
            "VehAge": [0, 2, 3, 15, 16],
            "VehPower": [4, 5, 8, 9, 15],
            "BonusMalus": [50, 100, 150, 151, 230],
            "Density": [1, 10, 100, 1000, 27000],
            "Area": ["A", "B", "C", "D", "F"],
            "VehBrand": ["B1"] * 5,
            "VehGas": ["Regular"] * 5,
            "Region": ["R11"] * 5,
        }
    )
    features = insurance.glm_features(df)
    assert features["DrivAgeBand"].tolist() == ["18-20", "21-25", "41-50", "51-70", "71+"]
    assert features["VehAgeBand"].tolist() == ["0", "1-2", "3-5", "11-15", "16+"]
    assert features["VehPowerBand"].tolist() == ["4", "5", "8", "9+", "9+"]
    assert features["BonusMalusCapped"].tolist() == [50, 100, 150, 150, 150]
    assert features["LogDensity"].tolist() == pytest.approx(np.log([1, 10, 100, 1000, 27000]))


def test_glm_features_reject_missing_values() -> None:
    df = pd.DataFrame(
        {
            "DrivAge": [np.nan],
            "VehAge": [1],
            "VehPower": [5],
            "BonusMalus": [50],
            "Density": [10],
            "Area": ["A"],
            "VehBrand": ["B1"],
            "VehGas": ["Regular"],
            "Region": ["R11"],
        }
    )
    with pytest.raises(ValueError, match="DrivAge"):
        insurance.glm_features(df)


def test_glm_design_reference_rare_and_unseen_levels(splits: dict[str, pd.DataFrame]) -> None:
    features = insurance.glm_features(splits["train"])
    features.loc[features.index[:3], "Region"] = "R_RARE"
    design = insurance.GlmDesign.fit(features, min_level_count=10)

    most_common_region = features["Region"].value_counts().idxmax()
    assert design.reference["Region"] == most_common_region
    assert "R_RARE" in design.folded_levels["Region"]
    assert "Region=R_RARE" not in design.column_names
    assert design.column_names[:3] == ["const", "BonusMalusCapped", "LogDensity"]

    unseen = features.iloc[:1].copy()
    unseen["Region"] = "R_NEVER_SEEN"
    matrix = design.transform(unseen)
    region_columns = [column for column in matrix.columns if column.startswith("Region=")]
    assert matrix[region_columns].to_numpy().sum() == 0  # unseen level is treated as the reference


# ---------------------------------------------------------------------------
# Frequency models
# ---------------------------------------------------------------------------


def test_offset_glm_equals_weighted_rate_regression(splits: dict[str, pd.DataFrame]) -> None:
    """Poisson GLM with offset log(e) on counts == Poisson regression on counts/e with weights e."""
    train = splits["train"]
    glm = insurance.fit_frequency_glm(train, min_level_count=20)
    design_matrix = glm.design.transform(insurance.glm_features(train))
    rate_model = PoissonRegressor(alpha=0.0, solver="newton-cholesky", max_iter=1000, tol=1e-10)
    rate_model.fit(
        design_matrix.drop(columns="const"),
        train["ClaimNb"] / train["Exposure"],
        sample_weight=train["Exposure"],
    )
    assert rate_model.intercept_ == pytest.approx(glm.coefficients["const"], abs=1e-4)
    for name, coefficient in zip(design_matrix.columns[1:], rate_model.coef_, strict=True):
        assert coefficient == pytest.approx(glm.coefficients[name], abs=1e-4)


def test_frequency_glm_balances_and_learns_bonus_malus(splits: dict[str, pd.DataFrame]) -> None:
    train = splits["train"]
    glm = insurance.fit_frequency_glm(train, min_level_count=20)
    assert glm.converged
    # A Poisson GLM with an intercept reproduces the total number of training claims exactly.
    predicted_claims = (glm.predict_rate(train) * train["Exposure"]).sum()
    assert predicted_claims == pytest.approx(train["ClaimNb"].sum(), rel=1e-6)
    # The synthetic data's frequency rises with BonusMalus.
    assert glm.coefficients["BonusMalusCapped"] > 0
    table = glm.coefficient_table()
    assert {"term", "coefficient", "std_error", "relativity"} <= set(table[0])
    assert table[0]["relativity"] == pytest.approx(np.exp(table[0]["coefficient"]))


def test_frequency_lgbm_uses_exposure_as_offset() -> None:
    """Rates must come out per policy-year: with the offset, the model learns the rate, not claims per row.

    Only Area varies (F has four times the claim rate of the other areas); exposure is either 0.1 or 1 year.
    Without the offset the model would predict claims per row, about 0.55 times the rate.
    """
    rng = np.random.default_rng(5)
    n_rows = 20000
    area = rng.choice(list("ABCDEF"), n_rows, p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.15])
    policies = pd.DataFrame(
        {
            "IDpol": np.arange(1, n_rows + 1),
            "Exposure": rng.choice([0.1, 1.0], n_rows),
            "VehPower": 6,
            "VehAge": 5,
            "DrivAge": 40,
            "BonusMalus": 50,
            "Density": 100,
            "Area": area,
            "VehGas": "Regular",
            "VehBrand": "B1",
            "Region": "R11",
        }
    )
    true_rate = np.where(area == "F", 0.6, 0.15)
    policies["ClaimNb"] = rng.poisson(true_rate * policies["Exposure"])
    train, validation = policies.iloc[:15000], policies.iloc[15000:]
    params = {**SMALL_FREQUENCY_PARAMS, "learning_rate": 0.1}

    model = insurance.fit_frequency_lgbm(train, validation, params=params, max_rounds=500, early_stopping_rounds=50)

    assert model.base_log_value == pytest.approx(np.log(train["ClaimNb"].sum() / train["Exposure"].sum()))
    predicted = model.predict_rate(train)
    for in_group in ((train["Area"] == "F").to_numpy(), (train["Area"] != "F").to_numpy()):
        empirical_rate = train["ClaimNb"][in_group].sum() / train["Exposure"][in_group].sum()
        assert predicted[in_group].mean() == pytest.approx(empirical_rate, rel=0.1)
    # The rate does not depend on exposure (exposure is not a feature).
    same_row_twice = policies.iloc[[0, 0]].copy()
    same_row_twice["Exposure"] = [0.1, 1.0]
    rates = model.predict_rate(same_row_twice)
    assert rates[0] == pytest.approx(rates[1])


def test_frequency_lgbm_trims_to_best_iteration(frequency_lgbm: insurance.TreeModel) -> None:
    info = frequency_lgbm.training_info
    assert frequency_lgbm.booster.num_trees() == info["best_iteration"]
    assert info["rounds_trained"] >= info["best_iteration"]
    assert info["early_stopping_metric"] == "poisson"


def test_tree_encoder_codes_and_unseen_categories(splits: dict[str, pd.DataFrame]) -> None:
    encoder = insurance.TreeFeatureEncoder.fit(splits["train"])
    assert encoder.codes["Area"] == {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
    row = splits["test"].iloc[:1].copy()
    row["Region"] = "R_NEVER_SEEN"
    encoded = encoder.transform(row)
    assert list(encoded.columns) == insurance.TREE_FEATURES
    assert np.isnan(encoded["Region"].iloc[0])
    assert all(pd.api.types.is_float_dtype(dtype) for dtype in encoded.dtypes)


# ---------------------------------------------------------------------------
# Severity models
# ---------------------------------------------------------------------------


def test_severity_models_fit_and_predict(splits: dict[str, pd.DataFrame]) -> None:
    train, validation = splits["severity_train"], splits["severity_validation"]
    glm = insurance.fit_severity_glm(train, min_level_count=10)
    lgbm = insurance.fit_severity_lgbm(
        train, validation, params=SMALL_SEVERITY_PARAMS, max_rounds=200, early_stopping_rounds=20
    )
    for model in (glm, lgbm):
        predicted = model.predict_rate(splits["severity_test"])
        assert (predicted > 0).all()
        assert np.isfinite(predicted).all()
    # Gamma GLM with log link and an intercept balances the claim-weighted average on the training data.
    weights = train["n_claims"]
    assert np.average(glm.predict_rate(train), weights=weights) == pytest.approx(
        np.average(train["average_amount_capped"], weights=weights), rel=0.05
    )
    assert lgbm.training_info["early_stopping_metric"] == "gamma"


def test_severity_glm_matches_statsmodels_directly(splits: dict[str, pd.DataFrame]) -> None:
    train = splits["severity_train"]
    glm = insurance.fit_severity_glm(train, min_level_count=10)
    design_matrix = glm.design.transform(insurance.glm_features(train))
    direct = sm.GLM(
        train["average_amount_capped"],
        design_matrix,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=train["n_claims"],
    ).fit()
    assert direct.params["const"] == pytest.approx(glm.coefficients["const"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_equal_weight_bins_match_calibration_table() -> None:
    rng = np.random.default_rng(0)
    predicted = rng.gamma(2.0, 0.05, 3000)
    exposure = rng.uniform(0.1, 1.0, 3000)
    claims = rng.poisson(predicted * exposure)
    rows = insurance.calibration_by_decile(claims, exposure, predicted, n_resamples=20)
    reference = calibration_table(claims / exposure, predicted, n_bins=10, sample_weight=exposure)

    assert [row["decile"] for row in rows] == reference["bin"].tolist()
    assert [row["n_policies"] for row in rows] == reference["n_rows"].tolist()
    assert [row["mean_predicted_rate"] for row in rows] == pytest.approx(reference["mean_predicted"].tolist())
    assert [row["mean_observed_rate"] for row in rows] == pytest.approx(reference["mean_observed"].tolist())
    for row in rows:
        assert row["mean_observed_rate_ci_low"] <= row["mean_observed_rate"] <= row["mean_observed_rate_ci_high"]
        assert row["observed_to_predicted"] == pytest.approx(row["observed_total"] / row["predicted_total"])


def test_frequency_metrics_shape_and_values() -> None:
    rng = np.random.default_rng(1)
    exposure = rng.uniform(0.1, 1.0, 2000)
    rate = rng.gamma(2.0, 0.05, 2000)
    claims = rng.poisson(rate * exposure)
    metrics = insurance.frequency_metrics(claims, exposure, rate, n_resamples=50, seed=42)

    deviance = metrics["mean_poisson_deviance"]
    assert deviance["ci_low"] <= deviance["estimate"] <= deviance["ci_high"]
    assert deviance["n_resamples"] == 50 and deviance["seed"] == 42
    assert metrics["lorenz_gini"]["estimate"] > 0  # the true rate ranks risk better than random
    assert metrics["total_observed_claims"] == claims.sum()
    assert metrics["total_predicted_claims"] == pytest.approx((rate * exposure).sum())
    assert metrics["observed_to_predicted_claims"]["estimate"] == pytest.approx(claims.sum() / (rate * exposure).sum())
    assert len(metrics["calibration_by_decile"]) == 10


def test_severity_metrics_and_paired_comparison() -> None:
    rng = np.random.default_rng(2)
    n_claims = rng.integers(1, 3, 1000).astype(float)
    true_mean = rng.uniform(500, 3000, 1000)
    average = rng.gamma(2.0, true_mean / 2.0)
    metrics = insurance.severity_metrics(average, n_claims, true_mean, n_resamples=30)
    assert metrics["mean_gamma_deviance"]["estimate"] > 0
    assert metrics["total_observed_amount_capped"] == pytest.approx((average * n_claims).sum())

    flat = np.full(1000, np.average(average, weights=n_claims))
    paired = insurance.paired_comparison(
        insurance.gamma_deviance_metric, average, true_mean, flat, n_claims, n_resamples=30
    )
    assert paired["estimate"] == pytest.approx(paired["estimate_a"] - paired["estimate_b"])
    assert paired["estimate"] < 0  # the true mean has lower deviance than a flat prediction


def test_expected_loss_summary_by_hand() -> None:
    observed_loss = np.array([0.0, 1000.0, 0.0, 500.0])
    exposure = np.array([1.0, 0.5, 1.0, 1.0])
    frequency = np.array([0.1, 0.2, 0.1, 0.4])
    severity = np.array([1000.0, 2000.0, 1000.0, 1000.0])
    summary = insurance.expected_loss_summary(observed_loss, exposure, frequency, severity, n_resamples=50)
    # predicted = 0.1*1*1000 + 0.2*0.5*2000 + 0.1*1*1000 + 0.4*1*1000 = 100 + 200 + 100 + 400 = 800
    assert summary["total_predicted_loss"] == pytest.approx(800.0)
    assert summary["total_observed_loss_capped"] == pytest.approx(1500.0)
    assert summary["observed_to_predicted"]["estimate"] == pytest.approx(1500.0 / 800.0)
    # Some resamples of 4 policies have no loss at all; their Gini is NaN and they are left out of the CI.
    gini = summary["pure_premium_lorenz_gini"]
    assert 0 < gini["n_valid_resamples"] < 50


# ---------------------------------------------------------------------------
# SHAP reason codes
# ---------------------------------------------------------------------------


def test_reason_maps_cover_every_tree_feature() -> None:
    assert set(insurance.FREQUENCY_REASONS_INCREASE) == set(insurance.TREE_FEATURES)
    assert set(insurance.FREQUENCY_REASONS_DECREASE) == set(insurance.TREE_FEATURES)


def test_explain_frequency_model(frequency_lgbm: insurance.TreeModel, splits: dict[str, pd.DataFrame]) -> None:
    explanation = insurance.explain_frequency_model(frequency_lgbm, splits["test"], sample_size=300, seed=42)

    assert explanation["sample_size"] == 300
    assert explanation["max_additivity_error"] < 1e-6
    assert {row["feature"] for row in explanation["global_importance"]} == set(insurance.TREE_FEATURES)
    importances = [row["mean_abs_shap"] for row in explanation["global_importance"]]
    assert importances == sorted(importances, reverse=True)
    assert len(explanation["examples"]) == 3
    frequencies = [example["predicted_frequency_per_year"] for example in explanation["examples"]]
    assert frequencies == sorted(frequencies)
    for example in explanation["examples"]:
        assert len(example["reasons_increase_risk"]) <= 4
        for reason in example["reasons_increase_risk"]:
            assert reason["contribution"] > 0
            assert reason["reason"] == insurance.FREQUENCY_REASONS_INCREASE[reason["feature"]]
            assert reason["value"] == example["features"][reason["feature"]]  # original value, not a code
        for reason in example["reasons_decrease_risk"]:
            assert reason["contribution"] < 0
    json.dumps(explanation)  # must be JSON-ready


# ---------------------------------------------------------------------------
# Saving, loading and scenario overrides
# ---------------------------------------------------------------------------


def _bundle(splits: dict[str, pd.DataFrame], frequency_lgbm: insurance.TreeModel) -> insurance.InsuranceModels:
    """All four models on the synthetic data."""
    return insurance.InsuranceModels(
        frequency_glm=insurance.fit_frequency_glm(splits["train"], min_level_count=20),
        frequency_lgbm=frequency_lgbm,
        severity_glm=insurance.fit_severity_glm(splits["severity_train"], min_level_count=10),
        severity_lgbm=insurance.fit_severity_lgbm(
            splits["severity_train"],
            splits["severity_validation"],
            params=SMALL_SEVERITY_PARAMS,
            max_rounds=100,
            early_stopping_rounds=10,
        ),
        feature_ranges=build_feature_ranges(splits["train"][RATING_FEATURES], categorical_columns=CATEGORICAL_FEATURES),
        metadata={"model_version": {"label": "test"}},
    )


def test_save_and_load_round_trip(
    tmp_path: Path, splits: dict[str, pd.DataFrame], frequency_lgbm: insurance.TreeModel
) -> None:
    bundle = _bundle(splits, frequency_lgbm)
    paths = insurance.save_models(bundle, tmp_path / "models")
    assert all(path.exists() for path in paths.values())

    loaded = insurance.load_models(tmp_path / "models")
    test = splits["test"]
    for model in ("glm", "lgbm"):
        assert loaded.predict(test, model=model).to_numpy() == pytest.approx(
            bundle.predict(test, model=model).to_numpy()
        )
    assert loaded.feature_ranges == bundle.feature_ranges
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["model_version"] == {"label": "test"}
    assert metadata["feature_ranges"]["BonusMalus"]["kind"] == "numeric"
    assert metadata["frequency_glm"]["coefficients"][0]["term"] == "const"


def test_predict_expected_loss_and_overrides(
    splits: dict[str, pd.DataFrame], frequency_lgbm: insurance.TreeModel
) -> None:
    bundle = _bundle(splits, frequency_lgbm)
    test = splits["test"]
    predictions = bundle.predict(test, model="glm")
    assert predictions["expected_loss_per_year"].to_numpy() == pytest.approx(
        (predictions["predicted_frequency"] * predictions["predicted_severity"]).to_numpy()
    )
    with pytest.raises(ValueError, match="model"):
        bundle.predict(test, model="xgboost")  # type: ignore[arg-type]

    # A scenario override inside the training range is applied; one outside it is rejected, never clipped.
    row = test.iloc[0][RATING_FEATURES].to_dict()
    ranges = bundle.feature_ranges
    worse = apply_overrides(row, {"BonusMalus": ranges["BonusMalus"].max_value}, ranges)
    base_frequency = bundle.frequency_glm.predict_rate(pd.DataFrame([row]))[0]
    stressed_frequency = bundle.frequency_glm.predict_rate(pd.DataFrame([worse]))[0]
    if row["BonusMalus"] < ranges["BonusMalus"].max_value:
        assert stressed_frequency > base_frequency  # BonusMalus coefficient is positive on this data
    with pytest.raises(OverrideError):
        apply_overrides(row, {"BonusMalus": ranges["BonusMalus"].max_value + 1}, ranges)


def test_glm_contributions_add_up_to_the_log_rate(splits: dict[str, pd.DataFrame]) -> None:
    glm = insurance.fit_frequency_glm(splits["train"], min_level_count=20)
    rows = splits["test"].iloc[:50]
    values, base = glm.contributions(rows)
    assert values.shape == (50, len(insurance.TREE_FEATURES))
    np.testing.assert_allclose(base + values.sum(axis=1), np.log(glm.predict_rate(rows)), rtol=1e-10, atol=1e-10)
    # The average training row has (close to) zero total contribution: contributions are relative to it.
    train_values, _ = glm.contributions(splits["train"])
    assert np.abs(train_values.mean(axis=0)).max() < 1e-8


def test_glm_without_design_means_cannot_explain(splits: dict[str, pd.DataFrame]) -> None:
    glm = insurance.fit_frequency_glm(splits["train"], min_level_count=20)
    glm.design_means = {}
    with pytest.raises(ValueError, match="retrain"):
        glm.contributions(splits["test"].iloc[:2])


def test_tree_contributions_add_up_to_the_log_rate(
    splits: dict[str, pd.DataFrame], frequency_lgbm: insurance.TreeModel
) -> None:
    rows = splits["test"].iloc[:50]
    values, base = frequency_lgbm.contributions(rows)
    np.testing.assert_allclose(base + values.sum(axis=1), np.log(frequency_lgbm.predict_rate(rows)), rtol=1e-8)


def test_risk_bands_for_preferred_models(splits: dict[str, pd.DataFrame], frequency_lgbm: insurance.TreeModel) -> None:
    bundle = _bundle(splits, frequency_lgbm)
    bands = insurance.risk_bands_for(bundle, splits["train"], {"frequency": "lgbm", "severity": "glm"})
    assert set(bands) == {"insurance_frequency", "insurance_severity"}
    assert bands["insurance_frequency"]["model"] == "lgbm"
    frequency = bundle.frequency_lgbm.predict_rate(splits["train"])
    expected = [np.quantile(frequency, q) for q in bands["insurance_frequency"]["quantiles"]]
    assert bands["insurance_frequency"]["cutoffs"] == pytest.approx(expected)
    severity = bundle.severity_glm.predict_rate(splits["train"])
    assert bands["insurance_severity"]["cutoffs"][0] == pytest.approx(np.quantile(severity, 0.5))
    with pytest.raises(ValueError, match="no 'xgb' model"):
        bundle.model_for("frequency", "xgb")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End to end: the training script on synthetic CSV files
# ---------------------------------------------------------------------------


def test_training_script_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frequency_raw, severity_raw = synthetic_fremtpl2_frames(n_policies=5000, seed=4)
    frequency_raw.to_csv(data_dir / FREQUENCY_FILE, index=False)
    severity_raw.to_csv(data_dir / SEVERITY_FILE, index=False)
    config = train_insurance.TrainConfig(
        data_dir=data_dir,
        results_path=tmp_path / "results" / "risk_metrics.json",
        figures_dir=tmp_path / "figures",
        models_dir=tmp_path / "models",
        notes_path=tmp_path / "insurance_data_notes.md",
        checksum_log=tmp_path / "checksums.jsonl",
        mlflow_db=tmp_path / "mlruns.db",
        mlflow_artifacts=tmp_path / "mlruns",
        n_resamples=10,
        shap_sample_size=200,
        min_level_count=10,
        frequency_params=SMALL_FREQUENCY_PARAMS,
        severity_params=SMALL_SEVERITY_PARAMS,
        max_rounds=100,
        early_stopping_rounds=10,
    )
    previous_uri = mlflow.get_tracking_uri()
    try:
        train_insurance.run(config)
    finally:
        mlflow.set_tracking_uri(previous_uri)

    results = json.loads(config.results_path.read_text())
    frequency = results["insurance_frequency"]["results"]
    severity = results["insurance_severity"]["results"]
    for model in ("glm", "lgbm"):
        assert isinstance(frequency["test_metrics"][model]["mean_poisson_deviance"]["estimate"], float)
        assert isinstance(severity["test_metrics"][model]["mean_gamma_deviance"]["estimate"], float)
        assert isinstance(severity["expected_loss"][model]["total_predicted_loss"], float)
    assert frequency["test_metrics"]["lgbm_minus_glm"]["mean_poisson_deviance"]["n_resamples"] == 10
    assert frequency["stability"]["status"] == "not measured"
    assert frequency["out_of_time_test"]["status"] == "not measured"
    assert frequency["split"]["sizes"]["test"]["n_policies"] == 1000
    assert frequency["model_version"]["frequency_lgbm"]["version"] == "1"
    assert results["insurance_frequency"]["metadata"]["datasets"]["fremtpl2"]["files"] == []

    assert all(
        (config.figures_dir / name).exists()
        for name in [
            "insurance_frequency_lorenz.png",
            "insurance_frequency_calibration.png",
            "insurance_severity_lorenz.png",
            "insurance_severity_calibration.png",
        ]
    )
    assert (config.models_dir / "metadata.json").exists()
    saved_metadata = json.loads((config.models_dir / "metadata.json").read_text())
    assert saved_metadata["risk_bands"]["insurance_frequency"] == frequency["risk_bands"]
    assert saved_metadata["risk_bands"]["insurance_severity"] == severity["risk_bands"]
    assert frequency["risk_bands"]["model"] == frequency["preferred_model"]["name"]
    check = frequency["scenario_sanity_checks"]["checks"][0]
    assert check["feature"] == "BonusMalus"
    for model in ("glm", "lgbm"):
        assert 0.0 <= check[model]["share_violations"] <= 1.0
    assert severity["scenario_sanity_checks"]["status"] == "not measured"
    notes = config.notes_path.read_text()
    linkage = frequency["data_quality"]["linkage"]
    assert f"{linkage['n_policies_with_claims_but_no_amounts']:,} policies with `ClaimNb > 0`" in notes
    assert "out-of-time" in notes
    assert "is not counted as a claim" in notes
    target = frequency["data_quality"]["modelling_target"]
    assert (
        frequency["split"]["sizes"]["train"]["n_claims"]
        + frequency["split"]["sizes"]["validation"]["n_claims"]
        + frequency["split"]["sizes"]["test"]["n_claims"]
        == target["n_claims_for_modelling"]
    )
    sensitivity = frequency["recorded_claim_count_sensitivity"]
    assert isinstance(sensitivity["lgbm"]["observed_to_predicted_claims_outside_block"], float)
    assert "observed_to_predicted" in severity["expected_loss"]["glm"]
    served = severity["expected_loss"]["served_pair"]
    assert served["frequency_model"] == frequency["preferred_model"]["name"]
    assert served["severity_model"] == severity["preferred_model"]["name"]
    assert served["observed_to_predicted"]["n_resamples"] == 10
    same_pair = served["frequency_model"] if served["frequency_model"] == served["severity_model"] else None
    if same_pair is not None:
        assert served["total_predicted_loss"] == pytest.approx(
            severity["expected_loss"][same_pair]["total_predicted_loss"]
        )
    grouped = frequency["grouped_split_sensitivity"]
    assert grouped["share_of_test_policies_whose_rating_features_appear_in_train"]["grouped_split"] == 0.0
    assert grouped["lgbm_minus_glm"]["mean_poisson_deviance"]["n_resamples"] == 10
    assert (
        sum(size["n_policies"] for size in grouped["sizes"].values())
        == 5000 - frequency["data_quality"]["frequency_file"]["n_rows_dropped_non_positive_exposure"]
    )
    assert any("grouped_split_sensitivity" in item for item in frequency["limitations"])
    assert any("also the one that chooses between GLM and LightGBM" in item for item in frequency["limitations"])
    linkage = frequency["data_quality"]["linkage"]
    assert "n_policies_where_severity_rows_differ_from_modelling_claimnb" in linkage

    # --notes-only re-renders the same notes from the results JSON.
    config.notes_path.unlink()
    train_insurance.write_data_notes(config.results_path, config.notes_path)
    assert config.notes_path.read_text() == notes

    assert train_insurance.data_notes_text(results) == notes

    summary = train_insurance.format_summary(frequency, severity)
    assert "Poisson deviance" in summary
    assert "served pair" in summary


def _interval(estimate: float, low: float, high: float) -> dict[str, float]:
    return {"estimate": estimate, "ci_low": low, "ci_high": high}


def test_duplicate_risk_limitation_states_what_was_measured() -> None:
    headline = _interval(-0.0091, -0.0107, -0.0075)
    grouped = {
        "share_of_test_policies_whose_rating_features_appear_in_train": {"random_split": 0.265, "grouped_split": 0.0},
        "lgbm_minus_glm": {"mean_poisson_deviance": _interval(-0.0108, -0.0124, -0.0093)},
    }
    text = train_insurance.duplicate_risk_limitation(grouped, headline)
    assert "26.5% of test policies" in text and "(0.0% shared)" in text
    assert "-0.0108 [-0.0124, -0.0093]" in text and "does not depend on" in text
    grouped["lgbm_minus_glm"]["mean_poisson_deviance"] = _interval(-0.001, -0.004, 0.002)
    assert "sensitive to how identical risks are split" in train_insurance.duplicate_risk_limitation(grouped, headline)
    assert train_insurance.duplicate_risk_limitation({}, headline) == train_insurance.DUPLICATE_RISK_UNMEASURED
    measured = train_insurance.limitations(grouped, headline)
    assert len(measured) == len(train_insurance.LIMITATIONS)
    assert train_insurance.DUPLICATE_RISK_UNMEASURED not in measured


def test_estimates_only_keeps_ci_fields() -> None:
    nested = {
        "glm": {"auc": {"estimate": 0.7, "ci_low": 0.6, "ci_high": 0.8, "seed": 42}, "rows": [1, 2], "n": 3},
    }
    assert train_insurance.estimates_only(nested) == {
        "glm": {"auc": {"estimate": 0.7, "ci_low": 0.6, "ci_high": 0.8}, "n": 3}
    }


def test_gini_metric_is_nan_without_claims() -> None:
    zeros = np.zeros(5)
    assert np.isnan(insurance.gini_metric(zeros, np.arange(5.0), np.ones(5)))
    observed = np.array([0.0, 0.0, 1.0, 2.0])
    predicted = np.array([0.1, 0.2, 0.3, 0.4])
    assert insurance.gini_metric(observed, predicted, np.ones(4)) == pytest.approx(
        lorenz_gini(observed, predicted, np.ones(4))
    )


def test_parse_args_defaults_and_notes_only() -> None:
    config, notes_only = train_insurance.parse_args([])
    assert config.n_resamples == 1000 and config.seed == 42 and not notes_only
    config, notes_only = train_insurance.parse_args(["--notes-only", "--n-resamples", "200"])
    assert notes_only and config.n_resamples == 200


def test_write_data_notes_needs_insurance_sections(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no insurance sections"):
        train_insurance.write_data_notes(tmp_path / "missing.json", tmp_path / "notes.md")
