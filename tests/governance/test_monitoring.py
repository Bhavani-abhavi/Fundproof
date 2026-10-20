"""Monitoring checks and breaches on SYNTHETIC freMTPL2-shaped data (offline, no model training)."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from monitoring_test_helpers import synthetic_policies, toy_frequency
from scipy.stats import poisson

from uwcopilot.governance.model_status import ModelEntry, ModelStatusStore
from uwcopilot.governance.monitoring import (
    SCORE_NAME,
    AlertRecord,
    CheckResult,
    MonitoredPool,
    VariableBins,
    alert_records,
    build_monitored_pool,
    calibration_ratio_check,
    evaluate_window,
    poisson_count_interval,
    psi_from_counts,
    read_alerts,
    record_alerts,
)
from uwcopilot.governance.monitoring_config import Thresholds, load_monitoring_config
from uwcopilot.risk.data.fremtpl2 import RATING_FEATURES
from uwcopilot.risk.stability import categorical_population_stability_index, population_stability_index

AT = datetime(2026, 10, 20, tzinfo=UTC)
MODEL_ID = "freq-glm-test"


@pytest.fixture(scope="module")
def thresholds() -> Thresholds:
    """The committed thresholds."""
    return load_monitoring_config().thresholds


@pytest.fixture(scope="module")
def pool_and_frame(thresholds: Thresholds) -> tuple[MonitoredPool, pd.DataFrame]:
    """A pool of 3,000 SYNTHETIC policies binned against 3,000 others."""
    reference = synthetic_policies(3000, seed=1)
    pool_frame = synthetic_policies(3000, seed=2)
    pool = build_monitored_pool(
        reference, toy_frequency(reference), pool_frame, toy_frequency(pool_frame), RATING_FEATURES, thresholds
    )
    return pool, pool_frame


# ----------------------------------------------------------------------------------------------------------------
# PSI: the same numbers as risk/stability.py
# ----------------------------------------------------------------------------------------------------------------


def test_numeric_psi_equals_risk_stability_with_ties_and_missing_values() -> None:
    rng = np.random.default_rng(0)
    expected = np.round(rng.gamma(2.0, 10.0, 2000))
    actual = np.round(rng.gamma(2.5, 10.0, 700))
    actual[:15] = np.nan
    bins = VariableBins.from_reference("x", expected, n_bins=10)
    expected_codes, n_expected = bins.codes(expected)
    actual_codes, n_actual = bins.codes(actual)
    n_codes = max(n_expected, n_actual)
    value = psi_from_counts(
        np.bincount(expected_codes, minlength=n_codes), np.bincount(actual_codes, minlength=n_codes)
    )
    assert value == pytest.approx(population_stability_index(expected, actual, n_bins=10), abs=1e-12)


def test_categorical_psi_equals_risk_stability_with_unseen_categories_and_missing_values() -> None:
    expected = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
    actual = ["A"] * 10 + ["B"] * 30 + ["D"] * 5 + ["E"] * 3 + [None] * 2
    bins = VariableBins.from_reference("area", expected, n_bins=10)
    assert bins.kind == "categorical" and bins.categories == ("A", "B", "C")
    actual_codes, n_codes = bins.codes(actual)
    expected_codes, _ = bins.codes(expected)
    value = psi_from_counts(
        np.bincount(expected_codes, minlength=n_codes), np.bincount(actual_codes, minlength=n_codes)
    )
    assert value == pytest.approx(categorical_population_stability_index(expected, actual), abs=1e-12)


def test_window_psi_matches_risk_stability_on_the_pool(pool_and_frame: tuple[MonitoredPool, pd.DataFrame]) -> None:
    pool, pool_frame = pool_and_frame
    reference = synthetic_policies(3000, seed=1)
    rows = np.arange(0, 3000, 3)
    result = evaluate_window(pool, rows, pool.recorded_claims[rows], load_monitoring_config().thresholds)
    assert result.find("input_drift", "DrivAge").value == pytest.approx(
        population_stability_index(reference["DrivAge"], pool_frame["DrivAge"].iloc[rows], n_bins=10), abs=1e-12
    )
    assert result.find("input_drift", "Region").value == pytest.approx(
        categorical_population_stability_index(reference["Region"], pool_frame["Region"].iloc[rows]), abs=1e-12
    )
    assert result.find("score_drift", SCORE_NAME).value == pytest.approx(
        population_stability_index(toy_frequency(reference), toy_frequency(pool_frame.iloc[rows]), n_bins=10),
        abs=1e-12,
    )


# ----------------------------------------------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 10, 195])
def test_the_garwood_interval_has_the_textbook_tail_probabilities(count: int) -> None:
    low, high = poisson_count_interval(count, 0.95)
    assert poisson.cdf(count, high) == pytest.approx(0.025, abs=1e-9)
    if count == 0:
        assert low == 0.0
    else:
        assert poisson.sf(count - 1, low) == pytest.approx(0.025, abs=1e-9)
    assert low <= count <= high


def test_ten_claims_give_the_published_interval() -> None:
    low, high = poisson_count_interval(10, 0.95)
    assert (round(low, 3), round(high, 3)) == (4.795, 18.390)


def test_calibration_ratio_levels(thresholds: Thresholds) -> None:
    settings = thresholds.calibration_ratio
    assert calibration_ratio_check(200, 200.0, settings).level == "ok"
    assert calibration_ratio_check(225, 200.0, settings).level == "warn"  # 1.125 outside, interval not
    alert = calibration_ratio_check(300, 200.0, settings)
    assert alert.level == "alert" and alert.interval is not None and alert.interval[0] > settings.band_high
    assert calibration_ratio_check(120, 200.0, settings).level == "alert"
    too_small = calibration_ratio_check(5, 10.0, settings)
    assert too_small.level == "insufficient_data" and too_small.value is None


# ----------------------------------------------------------------------------------------------------------------
# A whole window
# ----------------------------------------------------------------------------------------------------------------


def test_a_random_window_is_quiet_and_a_shifted_one_alerts(pool_and_frame: tuple[MonitoredPool, pd.DataFrame]) -> None:
    pool, pool_frame = pool_and_frame
    thresholds = load_monitoring_config().thresholds
    rows = np.random.default_rng(3).choice(pool.size, size=1500, replace=False)
    quiet = evaluate_window(pool, rows, pool.recorded_claims[rows], thresholds, window=4)
    assert quiet.window == 4 and quiet.n_policies == 1500
    assert all(check.level != "alert" for check in quiet.checks if check.check in ("input_drift", "score_drift"))

    young = np.flatnonzero(pool_frame["DrivAge"].to_numpy() < 30)
    shifted = evaluate_window(pool, young, pool.recorded_claims[young], thresholds)
    drivage = shifted.find("input_drift", "DrivAge")
    assert drivage.level == "alert" and drivage in shifted.alerts and shifted.level == "alert"


def test_more_claims_than_predicted_breaks_calibration_but_not_psi(
    pool_and_frame: tuple[MonitoredPool, pd.DataFrame],
) -> None:
    pool, _ = pool_and_frame
    thresholds = load_monitoring_config().thresholds
    rows = np.arange(pool.size)
    tripled = evaluate_window(pool, rows, pool.recorded_claims[rows] * 3, thresholds)
    assert tripled.find("calibration_ratio", "observed/predicted claims").level == "alert"
    assert tripled.find("input_drift", "Area").level == "ok"


def test_a_window_needs_rows_and_matching_outcomes(pool_and_frame: tuple[MonitoredPool, pd.DataFrame]) -> None:
    pool, _ = pool_and_frame
    thresholds = load_monitoring_config().thresholds
    with pytest.raises(ValueError):
        evaluate_window(pool, np.array([1, 2]), np.array([0]), thresholds)


# ----------------------------------------------------------------------------------------------------------------
# Breaches: alert log first, then the model status
# ----------------------------------------------------------------------------------------------------------------

BREACH = CheckResult("input_drift", "DrivAge", 0.37, "alert", "warn >= 0.1, alert > 0.25")


def registry_with_champion(tmp_path: Path) -> ModelStatusStore:
    """A registry holding the monitored model as champion."""
    store = ModelStatusStore(tmp_path / "model_status.json")
    store.register(ModelEntry(model_id=MODEL_ID, family="fremtpl2_frequency", version="1", status="champion"), at=AT)
    return store


def test_a_breach_is_logged_and_marks_the_model_under_review(tmp_path: Path) -> None:
    store = registry_with_champion(tmp_path)
    log = tmp_path / "alerts.jsonl"
    records = alert_records([BREACH], "RUN1", 1, "fremtpl2_frequency_model", MODEL_ID, window=6, at=AT)
    assert record_alerts(records, log, store) == 1
    assert read_alerts(log) == records
    entry = store.entry(MODEL_ID)
    assert entry.status == "under_review"
    assert entry.history[-1].actor == "monitor" and entry.history[-1].alert_id == "MON-RUN1-0001"
    assert "DrivAge" in entry.history[-1].reason


def test_alerts_are_appended_never_rewritten(tmp_path: Path) -> None:
    store = registry_with_champion(tmp_path)
    log = tmp_path / "alerts.jsonl"
    first = alert_records([BREACH], "RUN1", 1, "fremtpl2_frequency_model", MODEL_ID, at=AT)
    second = alert_records([BREACH, BREACH], "RUN2", 1, "fremtpl2_frequency_model", MODEL_ID, at=AT)
    record_alerts(first, log, store)
    record_alerts(second, log, store)
    assert [record.alert_id for record in read_alerts(log)] == ["MON-RUN1-0001", "MON-RUN2-0001", "MON-RUN2-0002"]
    assert len(store.entry(MODEL_ID).history) == 2  # registered, then under review once


def test_a_record_without_a_model_changes_no_status(tmp_path: Path) -> None:
    store = registry_with_champion(tmp_path)
    before = store.path.read_bytes()
    breach = CheckResult("indicator", "citation_support_rate", 0.95, "alert", "alert < 0.99")
    records = alert_records([breach], "RUN1", 1, "document_extraction", None, at=AT)
    assert records[0].action == "record_only"
    assert record_alerts(records, tmp_path / "alerts.jsonl", None) == 0
    assert store.path.read_bytes() == before


def test_only_alert_level_checks_are_breaches_and_marking_needs_a_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        alert_records([CheckResult("input_drift", "Area", 0.12, "warn", "")], "RUN1", 1, "c", MODEL_ID)
    records = alert_records([BREACH], "RUN1", 1, "fremtpl2_frequency_model", MODEL_ID, at=AT)
    with pytest.raises(ValueError):
        record_alerts(records, tmp_path / "alerts.jsonl", None)
    assert not (tmp_path / "alerts.jsonl").exists()
    with pytest.raises(ValueError):
        AlertRecord(**{**records[0].model_dump(), "model_id": None})
