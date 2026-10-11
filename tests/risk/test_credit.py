"""Tests for the credit-default pipeline, on SYNTHETIC Lending Club-shaped data (no real data needed).

The synthetic data only checks that the code works (splits, guard, keys, saving, scoring). Its metrics mean
nothing and are never reported.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.credit import (
    CREDIT_FEATURES,
    DEFAULT_LGBM_PARAMS,
    LIGHTGBM,
    LOGISTIC,
    MIN_AUC_GAIN_FOR_LIGHTGBM,
    REASON_MAP,
    TEST_YEARS,
    TRAIN_LAST_YEAR,
    VALIDATION_YEAR,
    CreditConfig,
    choose_model,
    fit_lightgbm,
    fit_logistic_scorecard,
    fit_platt,
    load_credit_models,
    out_of_time_split,
    psi_bin_codes,
    psi_by_vintage,
    psi_from_counts,
    risk_band,
    run_credit_pipeline,
    save_credit_models,
    sub_grade_score,
)
from uwcopilot.risk.data.lending_club import (
    finished_loans,
    prepare_loans,
    synthetic_accepted_loans,
    write_lending_club_zip,
)
from uwcopilot.risk.leakage import TARGET_COLUMN, LeakageError
from uwcopilot.risk.metrics import roc_auc
from uwcopilot.risk.results import is_not_measured, read_results, to_jsonable
from uwcopilot.risk.scenario import OverrideError
from uwcopilot.risk.stability import categorical_population_stability_index, population_stability_index

METRIC_KEYS = {"estimate", "ci_low", "ci_high", "n_resamples", "n_valid_resamples", "seed", "alpha", "method"}


def small_config() -> CreditConfig:
    """Fast settings for a few thousand synthetic rows (one LightGBM thread: CI machines are small)."""
    params = {**DEFAULT_LGBM_PARAMS, "num_threads": 1, "min_data_in_leaf": 20, "num_leaves": 7}
    return CreditConfig(
        n_resamples=30,
        shap_sample_size=200,
        min_category_count=5,
        lgbm_params=params,
        max_boost_rounds=300,
        early_stopping_rounds=30,
    )


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return synthetic_accepted_loans(n_rows=2500, seed=11)


@pytest.fixture(scope="module")
def loans(raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_loans(raw)


@pytest.fixture(scope="module")
def splits(loans: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return out_of_time_split(finished_loans(loans))


@pytest.fixture(scope="module")
def pipeline_run(loans: pd.DataFrame):
    return run_credit_pipeline(loans, small_config(), dataset_label="SYNTHETIC test data")


# ---------------------------------------------------------------------------
# Out-of-time split
# ---------------------------------------------------------------------------


def test_out_of_time_split_boundaries(loans: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    assert splits["train"]["issue_year"].max() <= TRAIN_LAST_YEAR
    assert set(splits["validation"]["issue_year"].unique()) == {VALIDATION_YEAR}
    assert set(splits["test"]["issue_year"].unique()) <= set(TEST_YEARS)
    assert splits["train"]["issue_date"].max() < splits["validation"]["issue_date"].min()
    assert splits["validation"]["issue_date"].max() < splits["test"]["issue_date"].min()
    assert sum(len(frame) for frame in splits.values()) == int(loans["is_finished"].sum())


def test_split_contains_only_finished_loans(splits: dict[str, pd.DataFrame]) -> None:
    for frame in splits.values():
        assert frame["is_finished"].all()
        assert set(frame[TARGET_COLUMN].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# Leakage guard at fit and score time (required by the build plan)
# ---------------------------------------------------------------------------


def test_banned_column_would_leak_the_outcome(raw: pd.DataFrame) -> None:
    """Why the guard exists: in this data, recoveries alone separates bad from good loans perfectly."""
    finished = raw.loc[raw["loan_status"].isin(["Fully Paid", "Charged Off"])]
    is_bad = (finished["loan_status"] == "Charged Off").astype(int)
    assert roc_auc(is_bad, finished["recoveries"].astype(float)) == 1.0


@pytest.mark.parametrize("banned", ["recoveries", "total_pymnt", "last_fico_range_low", "int_rate", "is_bad"])
def test_leakage_guard_fails_when_banned_column_reaches_the_model(
    raw: pd.DataFrame, splits: dict[str, pd.DataFrame], banned: str
) -> None:
    train = splits["train"].copy()
    validation = splits["validation"].copy()
    for frame in (train, validation):
        if banned not in frame.columns:
            frame[banned] = np.arange(len(frame), dtype=float)
    features = [*CREDIT_FEATURES, banned]
    with pytest.raises(LeakageError, match=banned):
        fit_logistic_scorecard(train, small_config(), features=features)
    with pytest.raises(LeakageError, match=banned):
        fit_lightgbm(train, validation, small_config(), features=features)


def test_score_rejects_banned_inputs(pipeline_run, splits: dict[str, pd.DataFrame]) -> None:
    _, models = pipeline_run
    row = splits["test"].iloc[0][CREDIT_FEATURES].to_dict()
    with pytest.raises(LeakageError):
        models.score({**row, "recoveries": 0.0})
    with pytest.raises(LeakageError):
        models.score(row, overrides={"last_fico_range_low": 700.0})
    with pytest.raises(ValueError, match="unknown features"):
        models.score({**row, "not_a_feature": 1.0})


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_end_to_end_payload_has_the_expected_keys(pipeline_run) -> None:
    payload, _ = pipeline_run
    for key in [
        "status",
        "split",
        "features",
        "models",
        "comparison",
        "model_selection",
        "benchmark_lending_club_sub_grade",
        "explainability",
        "stability",
        "scenario_feature_ranges",
        "bootstrap",
        "limitations",
    ]:
        assert key in payload, key
    for model_name in (LOGISTIC, LIGHTGBM):
        test_metrics = payload["models"][model_name]["test"]
        for metric in ("auc", "ks", "gini", "brier", "ece"):
            assert set(test_metrics[metric]) == METRIC_KEYS
            assert test_metrics[metric]["ci_low"] <= test_metrics[metric]["estimate"] <= test_metrics[metric]["ci_high"]
            assert test_metrics[metric]["n_resamples"] == 30
        assert len(test_metrics["calibration_by_predicted_decile"]) == 10
    difference = payload["comparison"]["test_auc_difference"]
    assert {"estimate_a", "estimate_b", "share_a_greater"} <= set(difference)
    assert difference["estimate"] == pytest.approx(difference["estimate_a"] - difference["estimate_b"])
    assert payload["model_selection"]["chosen"] in (LOGISTIC, LIGHTGBM)
    assert set(payload["benchmark_lending_club_sub_grade"]["test_auc"]) == METRIC_KEYS


def test_payload_is_strict_json(pipeline_run) -> None:
    payload, _ = pipeline_run
    json.dumps(to_jsonable(payload), allow_nan=False)


def test_explainability_reason_codes(pipeline_run) -> None:
    payload, _ = pipeline_run
    codes = payload["explainability"]["reason_codes"]
    assert codes["features_without_plain_language_reason"] == []
    assert [example["pd_percentile_in_sample"] for example in codes["examples"]] == [50, 90, 99]
    for example in codes["examples"]:
        assert 0 < len(example["reason_codes"]) <= 4
        contributions = [code["contribution"] for code in example["reason_codes"]]
        assert contributions == sorted(contributions, reverse=True) and min(contributions) > 0
        assert all(code["reason"] == REASON_MAP[code["feature"]] for code in example["reason_codes"])
    ranking = payload["explainability"]["global_importance"][LIGHTGBM]["ranking"]
    assert {entry["feature"] for entry in ranking} == set(CREDIT_FEATURES)


def test_stability_covers_every_vintage(pipeline_run, loans: pd.DataFrame) -> None:
    payload, _ = pipeline_run
    stability = payload["stability"]
    years = sorted(loans["issue_year"].unique().tolist())
    for rows in stability["score_psi_by_vintage"].values():
        assert [row["issue_year"] for row in rows] == years
        assert sum(row["n_rows"] for row in rows) == len(loans)  # unfinished loans included
        assert all(row["flag"] == (row["psi"] > 0.25) for row in rows)
    assert len(stability["top_features"]) == 10
    assert set(stability["feature_psi_by_vintage"]) == set(stability["top_features"])


def _counts(codes: np.ndarray, bins: np.ndarray) -> np.ndarray:
    return np.array([(codes == b).sum() for b in bins], dtype=float)


def test_binned_psi_matches_the_stability_module() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(np.r_[rng.normal(size=900), [np.nan] * 100])
    actual = pd.Series(np.r_[rng.normal(0.4, 1.2, size=450), [np.nan] * 50])
    codes = psi_bin_codes(pd.concat([reference, actual], ignore_index=True), reference, n_bins=10)
    bins = np.unique(codes)
    value = psi_from_counts(_counts(codes[:1000], bins), _counts(codes[1000:], bins))
    assert value == pytest.approx(population_stability_index(reference, actual, n_bins=10))

    levels_reference = pd.Series(["a"] * 50 + ["b"] * 30 + [None] * 20)
    levels_actual = pd.Series(["a"] * 20 + ["c"] * 60 + [None] * 20)
    names = ["<missing>", "a", "b", "c"]
    reference_counts = levels_reference.fillna("<missing>").value_counts().reindex(names, fill_value=0)
    actual_counts = levels_actual.fillna("<missing>").value_counts().reindex(names, fill_value=0)
    value = psi_from_counts(reference_counts.to_numpy(dtype=float), actual_counts.to_numpy(dtype=float))
    assert value == pytest.approx(categorical_population_stability_index(levels_reference, levels_actual))


def test_psi_by_vintage_matches_pairwise_psi(loans: pd.DataFrame) -> None:
    rows = psi_by_vintage(loans["dti"], loans["issue_year"], categorical=False, config=small_config())
    reference = loans.loc[loans["issue_year"] <= TRAIN_LAST_YEAR, "dti"]
    for row in rows:
        actual = loans.loc[loans["issue_year"] == row["issue_year"], "dti"]
        assert row["psi"] == pytest.approx(population_stability_index(reference, actual, n_bins=10))
    rows = psi_by_vintage(loans["purpose"], loans["issue_year"], categorical=True, config=small_config())
    reference = loans.loc[loans["issue_year"] <= TRAIN_LAST_YEAR, "purpose"]
    for row in rows:
        actual = loans.loc[loans["issue_year"] == row["issue_year"], "purpose"]
        assert row["psi"] == pytest.approx(categorical_population_stability_index(reference, actual))


def test_psi_is_defined_when_a_vintage_is_entirely_missing() -> None:
    values = pd.Series([np.nan] * 50 + list(np.linspace(0, 1, 150)) + list(np.linspace(0, 1, 100)))
    issue_year = pd.Series([2012] * 50 + [2014] * 150 + [2017] * 100)
    rows = psi_by_vintage(values, issue_year, categorical=False, config=small_config())
    by_year = {row["issue_year"]: row for row in rows}
    assert by_year[2012]["missing_share"] == 1.0
    assert by_year[2012]["flag"] is True  # all-missing vs a 25% missing reference is a significant shift
    assert by_year[2017]["psi"] < by_year[2012]["psi"]


def test_reason_map_covers_every_feature() -> None:
    assert set(REASON_MAP) == set(CREDIT_FEATURES)


# ---------------------------------------------------------------------------
# Model pieces
# ---------------------------------------------------------------------------


def test_scorecard_contributions_add_up_to_log_odds(splits: dict[str, pd.DataFrame]) -> None:
    scorecard = fit_logistic_scorecard(splits["train"], small_config())
    rows = splits["test"].iloc[:25]
    values, base_value = scorecard.contributions(rows)
    probabilities = scorecard.predict_pd(rows)
    np.testing.assert_allclose(base_value + values.sum(axis=1), np.log(probabilities / (1 - probabilities)))
    assert scorecard.training_info["converged"]


def test_lightgbm_class_weight_and_recalibration(splits: dict[str, pd.DataFrame]) -> None:
    model = fit_lightgbm(splits["train"], splits["validation"], small_config())
    info = model.training_info
    assert info["scale_pos_weight"] == pytest.approx(info["n_good_train"] / info["n_bad_train"])
    assert model.calibration_slope > 0
    validation = splits["validation"]
    recalibrated = model.predict_pd(validation).mean()
    observed = validation[TARGET_COLUMN].mean()
    # Platt scaling on the validation year makes the average PD match the observed rate there.
    assert recalibrated == pytest.approx(observed, abs=0.01)


def test_fit_platt_rejects_reversed_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1] * 10)
    assert fit_platt(np.where(labels == 1, 1.0, -1.0) + np.linspace(0, 0.1, 60), labels)[0] > 0
    with pytest.raises(ValueError, match="non-positive slope"):
        fit_platt(np.where(labels == 1, -1.0, 1.0) + np.linspace(0, 0.1, 60), labels)


def test_choose_model_rule() -> None:
    gain = MIN_AUC_GAIN_FOR_LIGHTGBM
    assert choose_model({"estimate": gain + 0.01, "ci_low": 0.001})["chosen"] == LIGHTGBM
    assert choose_model({"estimate": gain + 0.01, "ci_low": -0.001})["chosen"] == LOGISTIC
    assert choose_model({"estimate": gain / 2, "ci_low": 0.0001})["chosen"] == LOGISTIC


def test_sub_grade_score() -> None:
    scores = sub_grade_score(pd.Series(["A1", "A5", "B1", "G5", "Z9", None]))
    assert scores.iloc[:4].tolist() == [1.0, 5.0, 6.0, 35.0]
    assert scores.iloc[4:].isna().all()


def test_risk_bands_come_from_training_pd_quantiles(pipeline_run, splits: dict[str, pd.DataFrame]) -> None:
    payload, models = pipeline_run
    bands = payload["risk_bands"]
    assert bands == models.risk_bands
    assert bands["model"] == payload["model_selection"]["chosen"]
    training_pd = models.predict_pd(splits["train"])
    assert bands["n_scores"] == len(splits["train"])
    assert bands["cutoffs"] == pytest.approx([np.quantile(training_pd, q) for q in bands["quantiles"]])
    assert risk_band(0.0, bands) == "low"
    assert risk_band(1.0, bands) == "very high"
    assert risk_band(bands["cutoffs"][0], bands) == "medium"  # a score on a cut-off goes up a band


def test_scenario_sanity_checks_report_shares(pipeline_run) -> None:
    payload, _ = pipeline_run
    checks = payload["scenario_sanity_checks"]["checks"]
    assert [check["feature"] for check in checks] == ["dti", "fico_range_low"]
    for check in checks:
        for model_name in (LOGISTIC, LIGHTGBM):
            result = check[model_name]
            assert result["n_rows_checked"] > 0
            shares = result["share_violations"] + result["share_unchanged"] + result["share_expected_direction"]
            assert shares == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Scoring with scenario overrides, saving and loading
# ---------------------------------------------------------------------------


def test_score_without_overrides_has_no_scenario(pipeline_run, splits: dict[str, pd.DataFrame]) -> None:
    _, models = pipeline_run
    row = splits["test"].iloc[0][CREDIT_FEATURES].to_dict()
    for overrides in (None, {}):
        result = models.score(row, overrides)
        assert result["scenario"] is None
        frame = splits["test"].iloc[[0]]
        assert result["baseline"]["pd"] == pytest.approx(float(models.predict_pd(frame)[0]))


def test_score_with_overrides(pipeline_run, splits: dict[str, pd.DataFrame]) -> None:
    _, models = pipeline_run
    row = splits["test"].iloc[0][CREDIT_FEATURES].to_dict()
    same = models.score(row, {"dti": row["dti"]})
    assert same["scenario"]["pd"] == pytest.approx(same["baseline"]["pd"])
    income_range = models.feature_ranges["annual_inc"]
    lower_income = max(income_range.min_value, row["annual_inc"] * 0.9)
    changed = models.score(row, {"annual_inc": lower_income}, model=LOGISTIC)
    assert changed["scenario"]["pd"] != changed["baseline"]["pd"]
    with pytest.raises(OverrideError, match="outside the training range"):
        models.score(row, {"dti": models.feature_ranges["dti"].max_value + 100})
    with pytest.raises(OverrideError, match="not both"):
        models.score(row, {"annual_inc": lower_income, "loan_to_income": 0.1})
    with pytest.raises(OverrideError, match="not an allowed value"):
        models.score(row, {"term_months": 48})


def test_save_and_load_round_trip(tmp_path: Path, pipeline_run, splits: dict[str, pd.DataFrame]) -> None:
    _, models = pipeline_run
    paths = save_credit_models(models, tmp_path / "credit")
    assert all(path.exists() for path in paths.values())
    loaded = load_credit_models(tmp_path / "credit")
    test = splits["test"]
    for name in (LOGISTIC, LIGHTGBM):
        np.testing.assert_allclose(loaded.predict_pd(test, name), models.predict_pd(test, name))
    assert loaded.chosen == models.chosen
    assert loaded.feature_ranges == models.feature_ranges
    metadata = json.loads(paths["metadata"].read_text())
    assert "recoveries" in metadata["banned_post_origination_columns"]


# ---------------------------------------------------------------------------
# The training script
# ---------------------------------------------------------------------------


def test_script_writes_not_measured_without_data(tmp_path: Path) -> None:
    import train_credit

    results_path = tmp_path / "risk_metrics.json"
    exit_code = train_credit.main(
        [
            "--zip-path",
            str(tmp_path / "lending-club.zip"),
            "--results-path",
            str(results_path),
            "--models-dir",
            str(tmp_path / "models"),
            "--checksum-log",
            str(tmp_path / "none.jsonl"),
        ]
    )
    assert exit_code == 0
    section = read_results(results_path)["credit"]["results"]
    assert is_not_measured(section)
    assert section["reason"] == train_credit.MISSING_DATA_REASON
    assert not (tmp_path / "models").exists()


def test_script_end_to_end_on_synthetic_zip(tmp_path: Path) -> None:
    import train_credit
    from uwcopilot.risk import tracking

    previous_uri = tracking.mlflow.get_tracking_uri()
    zip_path = write_lending_club_zip(synthetic_accepted_loans(n_rows=4000, seed=3), tmp_path / "lending-club.zip")
    results_path = tmp_path / "risk_metrics.json"
    arguments = [
        "--zip-path",
        str(zip_path),
        "--results-path",
        str(results_path),
        "--models-dir",
        str(tmp_path / "models"),
        "--checksum-log",
        str(tmp_path / "none.jsonl"),
        "--n-resamples",
        "20",
        "--num-threads",
        "1",
        "--mlflow-db",
        str(tmp_path / "mlruns.db"),
        "--mlflow-artifacts",
        str(tmp_path / "mlruns"),
    ]
    try:
        exit_code = train_credit.main(arguments)
    finally:
        tracking.mlflow.set_tracking_uri(previous_uri)  # do not leave other tests pointing at tmp_path
    assert exit_code == 0
    section = read_results(results_path)["credit"]["results"]
    assert section["status"] == "measured"
    assert section["mlflow"]["registered_version"] == "1"
    assert section["data_quality"]["n_rows_without_status_or_issue_date"] == 2
    loaded = load_credit_models(tmp_path / "models")
    assert loaded.metadata["model_version"] == "uwcopilot_credit_pd v1"
