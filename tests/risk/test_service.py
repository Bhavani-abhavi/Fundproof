"""Tests for the scoring API (uwcopilot.risk.service).

Every model here is a tiny one trained on SYNTHETIC data inside the test and saved to a temporary folder, so
the tests need neither data/raw nor models/. The synthetic models' predictions mean nothing; the tests only
check that the API serves them faithfully and enforces its rules.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from uwcopilot.risk import credit, insurance
from uwcopilot.risk.bands import assign_band
from uwcopilot.risk.data.fremtpl2 import (
    CATEGORICAL_FEATURES,
    RATING_FEATURES,
    prepare_fremtpl2,
    severity_cap,
    severity_table,
    split_policies,
    synthetic_fremtpl2_frames,
)
from uwcopilot.risk.data.lending_club import finished_loans, prepare_loans, synthetic_accepted_loans
from uwcopilot.risk.scenario import build_feature_ranges
from uwcopilot.risk.service import ModelRegistry, create_app, insurance_model_version

FREQUENCY_PARAMS = {**insurance.FREQUENCY_LGBM_PARAMS, "min_data_in_leaf": 50, "num_threads": 1}
SEVERITY_PARAMS = {**insurance.SEVERITY_LGBM_PARAMS, "min_data_in_leaf": 20, "num_threads": 1}


# ---------------------------------------------------------------------------
# Fixtures: tiny synthetic models saved to temporary folders
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def insurance_data() -> dict[str, pd.DataFrame]:
    """SYNTHETIC freMTPL2-shaped policies, split 60/20/20, plus severity tables."""
    data = prepare_fremtpl2(*synthetic_fremtpl2_frames(n_policies=5000, seed=7))
    splits = split_policies(data.policies, seed=42)
    cap = severity_cap(data.claims.loc[data.claims["IDpol"].isin(splits["train"]["IDpol"]), "ClaimAmount"])
    for name in ("train", "validation", "test"):
        splits[f"severity_{name}"] = severity_table(splits[name], data.claims, cap)
    return splits


@pytest.fixture(scope="module")
def insurance_bundle(insurance_data: dict[str, pd.DataFrame]) -> insurance.InsuranceModels:
    """All four insurance models, with the metadata the training script writes (bands included)."""
    bundle = insurance.InsuranceModels(
        frequency_glm=insurance.fit_frequency_glm(insurance_data["train"], min_level_count=20),
        frequency_lgbm=insurance.fit_frequency_lgbm(
            insurance_data["train"],
            insurance_data["validation"],
            params=FREQUENCY_PARAMS,
            max_rounds=200,
            early_stopping_rounds=20,
        ),
        severity_glm=insurance.fit_severity_glm(insurance_data["severity_train"], min_level_count=10),
        severity_lgbm=insurance.fit_severity_lgbm(
            insurance_data["severity_train"],
            insurance_data["severity_validation"],
            params=SEVERITY_PARAMS,
            max_rounds=100,
            early_stopping_rounds=10,
        ),
        feature_ranges=build_feature_ranges(
            insurance_data["train"][RATING_FEATURES], categorical_columns=CATEGORICAL_FEATURES
        ),
        metadata={},
    )
    preferred = {"frequency": "lgbm", "severity": "glm"}  # the same choice as the real run, so both paths run
    bundle.metadata = {
        "model_version": {
            "frequency_lgbm": {"registered_model": "insurance_frequency_lgbm", "version": "7"},
            "severity_lgbm": {"registered_model": "insurance_severity_lgbm", "version": "7"},
            "label": "SYNTHETIC test bundle",
        },
        "git_commit": "0123456789abcdef",
        "prediction_units": {"frequency": "claims per policy-year", "severity": "capped claim amount per claim"},
        "preferred_model": preferred,
        "risk_bands": insurance.risk_bands_for(bundle, insurance_data["train"], preferred),
        "mlflow_runs": {"frequency": "run-frequency", "severity": "run-severity"},
    }
    return bundle


@pytest.fixture(scope="module")
def insurance_dir(tmp_path_factory: pytest.TempPathFactory, insurance_bundle: insurance.InsuranceModels) -> Path:
    path = tmp_path_factory.mktemp("models") / "insurance"
    insurance.save_models(insurance_bundle, path)
    return path


@pytest.fixture(scope="module")
def credit_loans() -> pd.DataFrame:
    return prepare_loans(synthetic_accepted_loans(n_rows=2500, seed=11))


@pytest.fixture(scope="module")
def credit_models(credit_loans: pd.DataFrame) -> credit.CreditModels:
    """Credit models trained on SYNTHETIC Lending Club-shaped loans with small, fast settings."""
    params = {**credit.DEFAULT_LGBM_PARAMS, "num_threads": 1, "min_data_in_leaf": 20, "num_leaves": 7}
    config = credit.CreditConfig(
        n_resamples=20,
        shap_sample_size=100,
        min_category_count=5,
        lgbm_params=params,
        max_boost_rounds=200,
        early_stopping_rounds=20,
    )
    _, models = credit.run_credit_pipeline(credit_loans, config, dataset_label="SYNTHETIC test data")
    models.metadata["model_version"] = "uwcopilot_credit_pd v1 (SYNTHETIC test)"
    return models


@pytest.fixture(scope="module")
def credit_dir(tmp_path_factory: pytest.TempPathFactory, credit_models: credit.CreditModels) -> Path:
    path = tmp_path_factory.mktemp("models") / "credit"
    credit.save_credit_models(credit_models, path)
    return path


@pytest.fixture(scope="module")
def client(credit_dir: Path, insurance_dir: Path) -> Any:
    """API with every product loaded."""
    with TestClient(create_app(credit_dir=credit_dir, insurance_dir=insurance_dir)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def client_without_credit(tmp_path_factory: pytest.TempPathFactory, insurance_dir: Path) -> Any:
    """API as it runs today: no credit model artifact (no Kaggle token), insurance models present."""
    missing = tmp_path_factory.mktemp("empty") / "credit"
    with TestClient(create_app(credit_dir=missing, insurance_dir=insurance_dir)) as test_client:
        yield test_client


def _policy(insurance_data: dict[str, pd.DataFrame], position: int = 0) -> dict[str, Any]:
    """One synthetic test policy's rating features as JSON-ready Python values."""
    row = insurance_data["test"].iloc[position]
    return {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in RATING_FEATURES}


def _applicant(credit_loans: pd.DataFrame) -> dict[str, Any]:
    """One synthetic finished training loan's features as JSON-ready values (NaN becomes None).

    A training loan is inside every training range by construction, so the API must accept it.
    """
    train = credit.out_of_time_split(finished_loans(credit_loans))["train"]
    row = train.iloc[0][credit.CREDIT_FEATURES].to_dict()
    clean = {}
    for name, value in row.items():
        value = value.item() if hasattr(value, "item") else value
        clean[name] = None if isinstance(value, float) and np.isnan(value) else value
    return clean


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_lists_loaded_models_and_band_definitions(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == ["credit", "insurance_frequency", "insurance_severity"]
    frequency = body["products"]["insurance_frequency"]
    assert frequency["model"] == "lgbm"
    assert frequency["model_version"] == "insurance_frequency_lgbm v7"
    assert frequency["features"] == RATING_FEATURES
    assert len(frequency["risk_band_definition"]["cutoffs"]) == len(frequency["risk_band_definition"]["labels"]) - 1


def test_health_without_credit_model_says_why(client_without_credit: TestClient) -> None:
    body = client_without_credit.get("/health").json()
    assert body["status"] == "degraded"
    assert body["models_loaded"] == ["insurance_frequency", "insurance_severity"]
    credit_status = body["products"]["credit"]
    assert credit_status["loaded"] is False
    assert "Kaggle" in credit_status["detail"] and "make train-credit" in credit_status["detail"]


# ---------------------------------------------------------------------------
# POST /score: insurance
# ---------------------------------------------------------------------------


def test_frequency_score_matches_the_model(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], insurance_bundle: insurance.InsuranceModels
) -> None:
    policy = _policy(insurance_data)
    response = client.post("/score", json={"product": "insurance_frequency", "features": policy})
    assert response.status_code == 200, response.text
    body = response.json()
    expected = float(insurance_bundle.frequency_lgbm.predict_rate(pd.DataFrame([policy]))[0])
    assert body["prediction"] == pytest.approx(expected)
    assert body["prediction_name"] == "expected_claim_frequency"
    assert body["prediction_unit"] == "claims per policy-year"
    assert body["model"] == "lgbm"
    assert body["model_version"] == "insurance_frequency_lgbm v7"
    bands = insurance_bundle.metadata["risk_bands"]["insurance_frequency"]
    assert body["risk_band"] == assign_band(expected, bands)
    assert body["risk_band_definition"]["cutoffs"] == pytest.approx(bands["cutoffs"])
    assert body["scenario"] is None
    reasons = body["reason_codes"]
    assert len(reasons) <= 4
    assert all(reason["contribution"] > 0 for reason in reasons)
    assert [reason["contribution"] for reason in reasons] == sorted((r["contribution"] for r in reasons), reverse=True)
    for reason in reasons:
        assert reason["value"] == policy[reason["feature"]]  # the original value, not an encoded one
        assert reason["reason"] == insurance.FREQUENCY_REASONS_INCREASE[reason["feature"]]


def test_severity_is_served_by_the_preferred_glm(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], insurance_bundle: insurance.InsuranceModels
) -> None:
    policy = _policy(insurance_data, position=3)
    body = client.post("/score", json={"product": "insurance_severity", "features": policy}).json()
    expected = float(insurance_bundle.severity_glm.predict_rate(pd.DataFrame([policy]))[0])
    assert body["prediction"] == pytest.approx(expected)
    assert body["model"] == "glm"
    assert body["model_version"] == "insurance_severity_glm (MLflow run run-severity, commit 0123456789ab)"
    values, base = insurance_bundle.severity_glm.contributions(pd.DataFrame([policy]))
    top = sorted((v for v in values[0] if v > 0), reverse=True)[:4]
    assert [reason["contribution"] for reason in body["reason_codes"]] == pytest.approx(top)
    assert all(
        reason["reason"] == insurance.SEVERITY_REASONS_INCREASE[reason["feature"]] for reason in body["reason_codes"]
    )


def test_override_of_the_same_value_changes_nothing(
    client: TestClient, insurance_data: dict[str, pd.DataFrame]
) -> None:
    policy = _policy(insurance_data)
    body = client.post(
        "/score",
        json={"product": "insurance_frequency", "features": policy, "overrides": {"BonusMalus": policy["BonusMalus"]}},
    ).json()
    scenario = body["scenario"]
    assert scenario["scenario"]["prediction"] == pytest.approx(scenario["baseline"]["prediction"])
    assert scenario["prediction_change"] == pytest.approx(0.0)
    assert scenario["risk_band_changed"] is False


def test_scenario_returns_baseline_and_scenario_side_by_side(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], insurance_bundle: insurance.InsuranceModels
) -> None:
    policy = _policy(insurance_data)
    worst = insurance_bundle.feature_ranges["BonusMalus"].max_value
    overrides = {"BonusMalus": worst, "Area": "F"}
    body = client.post(
        "/score", json={"product": "insurance_frequency", "features": policy, "overrides": overrides}
    ).json()
    scenario = body["scenario"]
    assert scenario["overrides"] == overrides
    assert scenario["baseline"]["prediction"] == pytest.approx(body["prediction"])  # top level = baseline
    expected = float(insurance_bundle.frequency_lgbm.predict_rate(pd.DataFrame([{**policy, **overrides}]))[0])
    assert scenario["scenario"]["prediction"] == pytest.approx(expected)
    change = scenario["scenario"]["prediction"] - scenario["baseline"]["prediction"]
    assert scenario["prediction_change"] == pytest.approx(change)
    assert scenario["relative_change"] == pytest.approx(change / scenario["baseline"]["prediction"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"BonusMalus": 10_000}, "outside the training range"),
        ({"Area": "Z"}, "not an allowed value"),
        ({"NotAFeature": 1}, "unknown feature"),
        ({"DrivAge": "forty"}, "is numeric"),
    ],
)
def test_invalid_overrides_are_rejected_not_clipped(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], overrides: dict[str, Any], message: str
) -> None:
    response = client.post(
        "/score",
        json={"product": "insurance_frequency", "features": _policy(insurance_data), "overrides": overrides},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "invalid_override"
    assert message in detail["message"] and "never clipped" in detail["message"]


def test_invalid_insurance_features_are_rejected(client: TestClient, insurance_data: dict[str, pd.DataFrame]) -> None:
    policy = _policy(insurance_data)
    missing = {name: value for name, value in policy.items() if name != "Region"}
    cases = [
        ({**policy, "ClaimNb": 1}, "unknown features ['ClaimNb']"),  # an outcome column is never an input
        (missing, "missing features ['Region']"),
        ({**policy, "DrivAge": 150}, "outside the training range"),
    ]
    for features, message in cases:
        response = client.post("/score", json={"product": "insurance_severity", "features": features})
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_features"
        assert message in response.json()["detail"]["message"]


def test_fractional_values_of_whole_number_features_are_rejected(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], insurance_bundle: insurance.InsuranceModels
) -> None:
    """VehPower and DrivAge are whole numbers in training, so 6.5 is outside the evidence even if in range."""
    policy = _policy(insurance_data)
    ranges = insurance_bundle.feature_ranges
    assert ranges["VehPower"].integer_only and ranges["DrivAge"].integer_only
    fractional_power = ranges["VehPower"].min_value + 0.5
    as_input = client.post(
        "/score", json={"product": "insurance_frequency", "features": {**policy, "VehPower": fractional_power}}
    )
    assert as_input.status_code == 422
    assert as_input.json()["detail"]["error"] == "invalid_features"
    assert "not a whole number" in as_input.json()["detail"]["message"]
    fractional_age = ranges["DrivAge"].min_value + 0.5
    as_override = client.post(
        "/score",
        json={"product": "insurance_frequency", "features": policy, "overrides": {"DrivAge": fractional_age}},
    )
    assert as_override.status_code == 422
    assert as_override.json()["detail"]["error"] == "invalid_override"
    whole_float = client.post(
        "/score", json={"product": "insurance_frequency", "features": {**policy, "VehPower": float(policy["VehPower"])}}
    )
    assert whole_float.status_code == 200, whole_float.text


def test_unknown_product_and_extra_fields_are_rejected(client: TestClient) -> None:
    assert client.post("/score", json={"product": "mortgage", "features": {}}).status_code == 422
    assert client.post("/score", json={"product": "credit", "features": {}, "debug": True}).status_code == 422


# ---------------------------------------------------------------------------
# POST /score: credit
# ---------------------------------------------------------------------------


def test_credit_returns_503_when_no_model_artifact(
    client_without_credit: TestClient, credit_loans: pd.DataFrame
) -> None:
    response = client_without_credit.post("/score", json={"product": "credit", "features": _applicant(credit_loans)})
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["error"] == "model_unavailable" and detail["product"] == "credit"
    assert "Kaggle" in detail["message"]


@pytest.mark.parametrize("which", ["features", "overrides"])
def test_leakage_guard_runs_even_without_a_credit_model(
    client_without_credit: TestClient, credit_loans: pd.DataFrame, which: str
) -> None:
    applicant = _applicant(credit_loans)
    request = {"product": "credit", "features": applicant}
    if which == "features":
        request["features"] = {**applicant, "total_pymnt": 1234.0}
    else:
        request["overrides"] = {"recoveries": 0.0}
    response = client_without_credit.post("/score", json=request)
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "leakage_guard"


def test_credit_score_matches_the_model(
    client: TestClient, credit_loans: pd.DataFrame, credit_models: credit.CreditModels
) -> None:
    applicant = _applicant(credit_loans)
    response = client.post("/score", json={"product": "credit", "features": applicant})
    assert response.status_code == 200, response.text
    body = response.json()
    expected = credit_models.score({k: (np.nan if v is None else v) for k, v in applicant.items()})["baseline"]
    assert body["prediction"] == pytest.approx(expected["pd"])
    assert body["risk_band"] == expected["risk_band"] == assign_band(expected["pd"], credit_models.risk_bands)
    assert body["model"] == credit_models.chosen
    assert body["model_version"] == "uwcopilot_credit_pd v1 (SYNTHETIC test)"
    assert body["prediction_name"] == "probability_of_default"
    assert 0 < len(body["reason_codes"]) <= 4
    assert all(reason["reason"] == credit.REASON_MAP[reason["feature"]] for reason in body["reason_codes"])


@pytest.mark.parametrize(
    ("request_part", "error"),
    [
        ({"features_extra": {"recoveries": 10.0}}, "leakage_guard"),
        ({"overrides": {"last_fico_range_low": 700}}, "leakage_guard"),
        ({"overrides": {"dti": 10_000.0}}, "invalid_override"),
        ({"features_extra": {"favourite_colour": "blue"}}, "invalid_features"),
    ],
)
def test_credit_rejections(
    client: TestClient, credit_loans: pd.DataFrame, request_part: dict[str, Any], error: str
) -> None:
    applicant = _applicant(credit_loans)
    body = {"product": "credit", "features": {**applicant, **request_part.get("features_extra", {})}}
    if "overrides" in request_part:
        body["overrides"] = request_part["overrides"]
    response = client.post("/score", json=body)
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == error


def test_invalid_credit_features_are_rejected(
    client: TestClient, credit_loans: pd.DataFrame, credit_models: credit.CreditModels
) -> None:
    """The base applicant gets the same checks as an override: nothing is filled in, nothing extrapolated."""
    applicant = _applicant(credit_loans)
    dti_max = credit_models.feature_ranges["dti"].max_value
    never_missing = next(
        name for name in credit.CREDIT_FEATURES if not credit_models.feature_ranges[name].missing_allowed
    )
    without_dti = {name: value for name, value in applicant.items() if name != "dti"}
    cases = [
        ({}, "missing features"),
        (without_dti, "missing features ['dti']"),
        ({**applicant, "dti": 100 * dti_max}, "outside the training range"),
        ({**applicant, "purpose": "time_travel"}, "not an allowed value"),
        ({**applicant, never_missing: None}, f"'{never_missing}' cannot be missing"),
    ]
    for features, message in cases:
        response = client.post("/score", json={"product": "credit", "features": features})
        assert response.status_code == 422, (features, response.text)
        detail = response.json()["detail"]
        assert detail["error"] == "invalid_features"
        assert message in detail["message"]


def test_credit_score_rejects_incomplete_features_directly(
    credit_loans: pd.DataFrame, credit_models: credit.CreditModels
) -> None:
    """CreditModels.score itself refuses an empty applicant (not only the API)."""
    with pytest.raises(ValueError, match="missing features"):
        credit_models.score({})


def test_credit_scenario_and_missing_values(
    client: TestClient, credit_loans: pd.DataFrame, credit_models: credit.CreditModels
) -> None:
    applicant = _applicant(credit_loans)
    assert credit_models.feature_ranges["mths_since_last_delinq"].missing_allowed
    applicant["mths_since_last_delinq"] = None  # missing in training too, so null is accepted and stays valid JSON
    dti_range = credit_models.feature_ranges["dti"]
    response = client.post(
        "/score", json={"product": "credit", "features": applicant, "overrides": {"dti": dti_range.max_value}}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["scenario"]["overrides"] == {"dti": dti_range.max_value}
    assert body["scenario"]["baseline"]["prediction"] == pytest.approx(body["prediction"])
    json.dumps(body, allow_nan=False)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_registry_reports_broken_artifacts(tmp_path: Path, insurance_dir: Path) -> None:
    broken = tmp_path / "credit"
    broken.mkdir()
    (broken / credit.METADATA_FILE).write_text("{not json")
    registry = ModelRegistry.load(broken, tmp_path / "no_insurance")
    assert registry.scorers == {}
    assert "could not be loaded" in registry.unavailable["credit"]
    assert "make train-insurance" in registry.unavailable["insurance_frequency"]


def test_registry_refuses_metadata_without_risk_bands(tmp_path: Path, insurance_dir: Path) -> None:
    copy = tmp_path / "insurance"
    copy.mkdir()
    for path in insurance_dir.iterdir():
        (copy / path.name).write_bytes(path.read_bytes())
    metadata = json.loads((copy / insurance.METADATA_FILE).read_text())
    del metadata["risk_bands"]
    (copy / insurance.METADATA_FILE).write_text(json.dumps(metadata))
    registry = ModelRegistry.load(tmp_path / "no_credit", copy)
    assert "insurance_frequency" not in registry.scorers
    assert "make train-insurance" in registry.unavailable["insurance_frequency"]


def test_lazy_loading_without_startup(credit_dir: Path, insurance_dir: Path) -> None:
    """Without the startup event (no `with`), the first request loads the models."""
    test_client = TestClient(create_app(credit_dir=credit_dir, insurance_dir=insurance_dir))
    assert test_client.get("/health").json()["status"] == "ok"


def test_insurance_model_version_strings() -> None:
    metadata = {
        "model_version": {"frequency_lgbm": {"registered_model": "insurance_frequency_lgbm", "version": "3"}},
        "mlflow_runs": {"severity": "abc"},
        "git_commit": "fedcba9876543210",
    }
    assert insurance_model_version(metadata, "frequency", "lgbm") == "insurance_frequency_lgbm v3"
    assert insurance_model_version(metadata, "severity", "glm") == (
        "insurance_severity_glm (MLflow run abc, commit fedcba987654)"
    )


# ---------------------------------------------------------------------------
# The shared HTTP client against the real app (the contract the agent and the MCP server rely on)
# ---------------------------------------------------------------------------


def test_risk_api_client_round_trip_against_the_real_service(
    client: TestClient, insurance_data: dict[str, pd.DataFrame], insurance_bundle: insurance.InsuranceModels
) -> None:
    """uwcopilot.risk.client talks to the real FastAPI app: baseline and scenario come back as tool responses.

    An earlier client posted the tool model (with a scenario label the service forbids) and expected the tool model
    back; this test fails on that version, so the two contracts cannot drift apart again.
    """
    from uwcopilot.risk.client import RiskApiClient
    from uwcopilot.tools.schemas import RiskScoreRequest

    class ForwardToTestClient(httpx.BaseTransport):
        """An httpx transport that hands each request to the FastAPI TestClient (no network)."""

        def handle_request(self, request: httpx.Request) -> httpx.Response:
            answer = client.request(request.method, str(request.url), content=request.read(), headers=request.headers)
            return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)

    api = RiskApiClient("http://testserver", transport=ForwardToTestClient())
    policy = _policy(insurance_data)
    baseline = api.score(RiskScoreRequest(product="insurance_frequency", features=policy))
    direct = client.post("/score", json={"product": "insurance_frequency", "features": policy}).json()
    assert baseline.prediction == pytest.approx(direct["prediction"])
    assert baseline.risk_band == direct["risk_band"] and baseline.scenario == "baseline"
    assert [reason.description for reason in baseline.reason_codes] == [r["reason"] for r in direct["reason_codes"]]

    worst = insurance_bundle.feature_ranges["BonusMalus"].max_value
    stressed = api.score(
        RiskScoreRequest(
            product="insurance_frequency", features=policy, overrides={"BonusMalus": worst}, scenario="bonus_malus_max"
        )
    )
    expected = float(insurance_bundle.frequency_lgbm.predict_rate(pd.DataFrame([{**policy, "BonusMalus": worst}]))[0])
    assert stressed.prediction == pytest.approx(expected)
    assert stressed.scenario == "bonus_malus_max"
    assert stressed.model_version == direct["model_version"]
