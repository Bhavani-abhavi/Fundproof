"""Tests for uwcopilot.risk.fairness_demo and scripts/fairness_german.py, on SYNTHETIC data only."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import fairness_german
from uwcopilot.risk.data.german_credit import (
    EXCLUDED_FROM_FEATURES,
    MODEL_FEATURES,
    add_derived_columns,
    parse_german_data,
    synthetic_german_data_text,
)
from uwcopilot.risk.fairness import adverse_impact_ratio, equal_opportunity_gap, selection_rates
from uwcopilot.risk.fairness_demo import (
    COMPARISONS,
    LIGHTGBM,
    LOGISTIC,
    NOTE,
    DemoConfig,
    GroupComparison,
    adverse_impact_ratio_value,
    approval_rate_in_group,
    approval_threshold,
    approve,
    auc_or_nan,
    build_lightgbm_model,
    build_logistic_model,
    choose_lightgbm_rounds,
    equal_opportunity_difference,
    evaluate_comparison,
    proxy_strength,
    run_fairness_demo,
    split_train_test,
    true_positive_rate_in_group,
)
from uwcopilot.risk.results import is_not_measured

FAST = DemoConfig(n_resamples=30)


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    """400 SYNTHETIC applicants in the german.data format, with label and group columns."""
    return add_derived_columns(parse_german_data(synthetic_german_data_text(n_rows=400, seed=11)))


@pytest.fixture(scope="module")
def payload(synthetic_frame: pd.DataFrame) -> dict[str, Any]:
    """One full run of the demo on the synthetic frame (few resamples to keep the test fast)."""
    return run_fairness_demo(synthetic_frame, FAST)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------


def test_threshold_by_hand() -> None:
    p_bad = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    threshold = approval_threshold(p_bad, 0.7)  # approve 7 rows: midpoint of 0.7 and 0.8
    assert threshold == pytest.approx(0.75)
    assert approve(p_bad, threshold).tolist() == [1] * 7 + [0] * 3


def test_threshold_order_does_not_matter() -> None:
    rng = np.random.default_rng(0)
    p_bad = rng.random(101)
    assert approval_threshold(p_bad, 0.6) == approval_threshold(rng.permutation(p_bad), 0.6)
    assert approve(p_bad, approval_threshold(p_bad, 0.6)).sum() == round(0.6 * 101)


def test_threshold_edges() -> None:
    p_bad = np.array([0.2, 0.4, 0.6])
    assert approve(p_bad, approval_threshold(p_bad, 0.0)).sum() == 0
    assert approve(p_bad, approval_threshold(p_bad, 1.0)).sum() == 3


def test_threshold_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        approval_threshold(np.array([0.1, 0.2]), 1.5)
    with pytest.raises(ValueError):
        approval_threshold(np.array([]), 0.5)


def test_approve_is_strictly_below_threshold() -> None:
    assert approve(np.array([0.49, 0.5, 0.51]), 0.5).tolist() == [1, 0, 0]


# ---------------------------------------------------------------------------
# Group metrics
# ---------------------------------------------------------------------------

# Group A: 4 of 5 approved; group B: 2 of 5 approved. Truly good: A rows 0-3, B rows 5-7.
SELECTED = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
GROUPS = np.array(["A"] * 5 + ["B"] * 5)
Y_GOOD = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0])


def test_group_metrics_by_hand() -> None:
    assert approval_rate_in_group(SELECTED, GROUPS, "A") == pytest.approx(0.8)
    assert approval_rate_in_group(SELECTED, GROUPS, "B") == pytest.approx(0.4)
    assert adverse_impact_ratio_value(SELECTED, GROUPS, "B", "A") == pytest.approx(0.5)
    assert true_positive_rate_in_group(Y_GOOD, SELECTED, GROUPS, "A") == pytest.approx(1.0)
    assert true_positive_rate_in_group(Y_GOOD, SELECTED, GROUPS, "B") == pytest.approx(2 / 3)
    assert equal_opportunity_difference(Y_GOOD, SELECTED, GROUPS, "B", "A") == pytest.approx(2 / 3 - 1.0)


def test_group_metrics_are_nan_for_missing_groups() -> None:
    assert math.isnan(approval_rate_in_group(SELECTED, GROUPS, "C"))
    assert math.isnan(adverse_impact_ratio_value(SELECTED, GROUPS, "C", "A"))
    assert math.isnan(adverse_impact_ratio_value(np.zeros(10, dtype=int), GROUPS, "B", "A"))  # privileged rate 0
    assert math.isnan(true_positive_rate_in_group(np.zeros(10, dtype=int), SELECTED, GROUPS, "A"))
    assert math.isnan(auc_or_nan(np.ones(5), np.linspace(0, 1, 5)))


def test_group_metrics_match_the_foundation_fairness_module() -> None:
    rng = np.random.default_rng(5)
    selected = rng.integers(0, 2, 500)
    groups = rng.choice(["old", "young"], 500)
    y_good = rng.integers(0, 2, 500)
    rates = selection_rates(selected, groups)
    assert approval_rate_in_group(selected, groups, "young") == pytest.approx(rates["young"])
    air = adverse_impact_ratio(selected, groups, reference_group="old")
    assert adverse_impact_ratio_value(selected, groups, "young", "old") == pytest.approx(
        air["groups"]["young"]["impact_ratio"]
    )
    gap = equal_opportunity_gap(y_good, selected, groups, reference_group="old")
    assert equal_opportunity_difference(y_good, selected, groups, "young", "old") == pytest.approx(
        gap["groups"]["young"]["gap_vs_reference"]
    )


def test_evaluate_comparison_by_hand() -> None:
    comparison = GroupComparison("toy", "toy", privileged="A", unprivileged="B", role="primary", description="toy")
    groups = np.append(GROUPS, "other")  # one row in neither group is left out
    result = evaluate_comparison(np.append(Y_GOOD, 1), np.append(SELECTED, 1), groups, comparison, FAST)
    assert result["n_test_rows_left_out"] == 1
    assert result["groups"]["A"]["n"] == 5
    assert result["groups"]["B"]["n_truly_good"] == 3
    assert result["groups"]["B"]["observed_good_rate"] == pytest.approx(0.6)
    assert result["groups"]["B"]["approval_rate"]["estimate"] == pytest.approx(0.4)
    air = result["adverse_impact_ratio"]
    assert air["estimate"] == pytest.approx(0.5)
    assert air["point_estimate_below_threshold"] is True
    assert air["n_resamples"] == 30
    assert air["ci_low"] <= air["estimate"] <= air["ci_high"]
    assert result["equal_opportunity_difference"]["estimate"] == pytest.approx(-1 / 3)


# ---------------------------------------------------------------------------
# Models and split
# ---------------------------------------------------------------------------


def test_split_is_stratified_and_seeded(synthetic_frame: pd.DataFrame) -> None:
    train, test = split_train_test(synthetic_frame, 0.3, 42)
    again_train, _ = split_train_test(synthetic_frame, 0.3, 42)
    assert len(test) == 120 and len(train) == 280
    assert abs(train["bad"].mean() - test["bad"].mean()) < 0.02
    pd.testing.assert_frame_equal(train, again_train)


@pytest.mark.parametrize("model_name", [LOGISTIC, LIGHTGBM])
def test_protected_attributes_never_reach_the_model(synthetic_frame: pd.DataFrame, model_name: str) -> None:
    features = synthetic_frame[list(MODEL_FEATURES)]
    model = build_logistic_model(FAST) if model_name == LOGISTIC else build_lightgbm_model(20, FAST)
    model.fit(features, synthetic_frame["bad"])
    seen = set(model.named_steps["preprocess"].feature_names_in_)
    assert seen == set(MODEL_FEATURES)
    for protected in [*EXCLUDED_FROM_FEATURES, "sex", "sex_corrected", "age_group_25", "age_group_30"]:
        assert protected not in seen


def test_lightgbm_rounds_are_reproducible(synthetic_frame: pd.DataFrame) -> None:
    features = synthetic_frame[list(MODEL_FEATURES)]
    rounds = choose_lightgbm_rounds(features, synthetic_frame["bad"].to_numpy(), FAST)
    assert 1 <= rounds <= FAST.lgbm_max_rounds
    assert rounds == choose_lightgbm_rounds(features, synthetic_frame["bad"].to_numpy(), FAST)


def test_proxy_strength_not_measured_when_a_group_is_missing(synthetic_frame: pd.DataFrame) -> None:
    train, test = split_train_test(synthetic_frame, 0.3, 42)
    missing = GroupComparison("x", "sex", privileged="male", unprivileged="nobody", role="sensitivity", description="")
    assert is_not_measured(proxy_strength(train, test, missing, FAST)["test_auc"])


# ---------------------------------------------------------------------------
# Full run on synthetic data
# ---------------------------------------------------------------------------


def test_payload_structure(payload: dict[str, Any]) -> None:
    assert payload["note"] == NOTE
    assert payload["note"] == "Demonstration on UCI German Credit; Lending Club has no protected attributes."
    assert set(payload["models"]) == {LOGISTIC, LIGHTGBM}
    assert payload["bootstrap"]["n_resamples"] == 30
    assert payload["split"]["n_test"] == 120
    for model in payload["models"].values():
        assert set(model["fairness"]) == {comparison.key for comparison in COMPARISONS}
        assert 0.0 <= model["test_auc"]["estimate"] <= 1.0
        # The threshold is set on out-of-fold training predictions to match the training good rate.
        assert model["train_out_of_fold_approval_rate"] == pytest.approx(
            model["target_approval_rate_from_train"], abs=1 / payload["split"]["n_train"]
        )
    assert set(payload["proxy_check"]) == {comparison.key for comparison in COMPARISONS}
    assert "personal_status_sex" not in payload["model_features"]
    assert "age_years" not in payload["model_features"]


def test_synthetic_signal_is_learned(payload: dict[str, Any]) -> None:
    # The synthetic target depends on checking account and duration, so both models beat random ranking.
    for model in payload["models"].values():
        assert model["test_auc"]["estimate"] > 0.6


def test_summary_table_copies_the_nested_numbers(payload: dict[str, Any]) -> None:
    rows = payload["summary_table"]["fairness"]
    assert len(rows) == 2 * len(COMPARISONS)
    row = next(r for r in rows if r["model"] == LIGHTGBM and r["comparison"] == "age_lt25")
    nested = payload["models"][LIGHTGBM]["fairness"]["age_lt25"]
    assert row["adverse_impact_ratio"] == nested["adverse_impact_ratio"]["estimate"]
    assert row["equal_opportunity_difference"] == nested["equal_opportunity_difference"]["estimate"]
    model_rows = {r["model"]: r for r in payload["summary_table"]["models"]}
    assert model_rows[LOGISTIC]["test_auc"] == payload["models"][LOGISTIC]["test_auc"]["estimate"]
    difference = model_rows["lightgbm_minus_logistic_regression"]["test_auc"]
    assert difference == pytest.approx(model_rows[LIGHTGBM]["test_auc"] - model_rows[LOGISTIC]["test_auc"])


def test_run_is_deterministic(synthetic_frame: pd.DataFrame, payload: dict[str, Any]) -> None:
    again = run_fairness_demo(synthetic_frame, FAST)
    assert again["summary_table"] == payload["summary_table"]


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------


def test_script_writes_section_and_keeps_others(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_path = tmp_path / "german.data"
    data_path.write_text(synthetic_german_data_text(n_rows=300, seed=2), encoding="utf-8")
    checksum_log = tmp_path / "checksums.jsonl"
    checksum_log.write_text(
        json.dumps({"dataset": "german_credit", "file": "german.data", "sha256": "abc", "size_bytes": 1}) + "\n",
        encoding="utf-8",
    )
    results_path = tmp_path / "risk_metrics.json"
    results_path.write_text(json.dumps({"credit": {"results": {"auc": 1}, "metadata": {}}}), encoding="utf-8")

    exit_code = fairness_german.main(
        [
            "--data-path",
            str(data_path),
            "--results-path",
            str(results_path),
            "--checksum-log",
            str(checksum_log),
            "--n-resamples",
            "20",
            "--no-mlflow",
        ]
    )

    assert exit_code == 0
    contents = json.loads(results_path.read_text(encoding="utf-8"))
    assert contents["credit"] == {"results": {"auc": 1}, "metadata": {}}
    section = contents["fairness_demo"]
    assert section["results"]["note"] == NOTE
    assert section["results"]["mlflow"]["status"] == "not logged"
    assert section["metadata"]["datasets"]["german_credit"]["files"][0]["sha256"] == "abc"
    printed = capsys.readouterr().out
    assert "adverse_impact_ratio" in printed
    assert NOTE in printed


def test_mlflow_metrics_are_finite_numbers(payload: dict[str, Any]) -> None:
    metrics = fairness_german.mlflow_metrics(payload)
    assert f"{LOGISTIC}.test_auc" in metrics
    assert f"{LIGHTGBM}.sex.adverse_impact_ratio" in metrics
    assert all(isinstance(value, float) for value in metrics.values())


def test_format_value() -> None:
    assert fairness_german.format_value(0.12345) == "0.123"
    assert fairness_german.format_value(True) == "True"
    assert fairness_german.format_value(None) == "None"
    assert fairness_german.format_value(3) == "3"
