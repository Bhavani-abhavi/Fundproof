"""config/monitoring.yaml: the committed file is valid, rules read values as documented, bad files are refused."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from uwcopilot.governance.monitoring_config import (
    DEFAULT_CONFIG_PATH,
    INDICATOR_NAMES,
    LevelRule,
    MonitoringConfig,
    load_monitoring_config,
)


def committed_document() -> dict[str, Any]:
    """The committed config as a plain dict, to change one value in a test."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def test_the_committed_config_loads_and_keeps_the_risk_stability_psi_bands() -> None:
    config = load_monitoring_config()
    assert config.data_label == "SYNTHETIC"
    for psi in (config.thresholds.input_drift_psi, config.thresholds.score_drift_psi):
        assert (psi.rule.warn, psi.rule.alert, psi.n_bins) == (0.10, 0.25, 10)
    assert set(config.thresholds.indicators) == set(INDICATOR_NAMES)
    assert any(not scenario.drifted for scenario in config.stream.scenarios.values())
    assert config.bootstrap.resamples == 1000 and config.bootstrap.seed == 42


def test_every_threshold_carries_a_rationale() -> None:
    thresholds = load_monitoring_config().thresholds
    rules = [thresholds.input_drift_psi.rule, thresholds.score_drift_psi.rule, thresholds.calibration_ece.rule]
    rules += list(thresholds.indicators.values())
    assert all(rule.rationale.strip() for rule in rules)
    assert thresholds.calibration_ratio.rationale.strip()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0999, "ok"), (0.10, "warn"), (0.25, "warn"), (0.2501, "alert")],
)
def test_a_higher_is_worse_rule_matches_the_psi_bands_of_risk_stability(value: float, expected: str) -> None:
    rule = LevelRule(direction="higher_is_worse", warn=0.10, alert=0.25, rationale="test")
    assert rule.level(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(1.0, "ok"), (0.9999, "alert")])
def test_a_lower_is_worse_rule_without_warn_alerts_below_the_level(value: float, expected: str) -> None:
    rule = LevelRule(direction="lower_is_worse", alert=1.0, rationale="every required escalation")
    assert rule.level(value) == expected
    assert rule.describe() == "alert < 1"


def test_a_rule_with_warn_beyond_alert_is_refused() -> None:
    with pytest.raises(ValidationError):
        LevelRule(direction="higher_is_worse", warn=0.3, alert=0.2, rationale="x")
    with pytest.raises(ValidationError):
        LevelRule(direction="lower_is_worse", rationale="x")


def write_config(tmp_path: Path, document: dict[str, Any]) -> Path:
    """Write a changed config to a temporary file."""
    path = tmp_path / "monitoring.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_an_unknown_key_is_refused_so_a_typo_cannot_switch_a_check_off(tmp_path: Path) -> None:
    document = committed_document()
    document["thresholds"]["input_drift_psi"]["rule"]["alrt"] = 0.5
    with pytest.raises(ValidationError):
        load_monitoring_config(write_config(tmp_path, document))


def test_a_missing_indicator_rule_is_refused(tmp_path: Path) -> None:
    document = committed_document()
    del document["thresholds"]["indicators"]["override_rate"]
    with pytest.raises(ValidationError):
        load_monitoring_config(write_config(tmp_path, document))


def test_the_stream_needs_a_no_drift_scenario_and_drift_inside_the_stream(tmp_path: Path) -> None:
    document = committed_document()
    document["stream"]["drift_start_window"] = document["stream"]["n_windows"]
    with pytest.raises(ValidationError):
        MonitoringConfig.model_validate(document)
    document = committed_document()
    document["stream"]["scenarios"] = {
        name: scenario
        for name, scenario in document["stream"]["scenarios"].items()
        if "covariate_shift" in scenario or "claim_rate_multiplier" in scenario
    }
    with pytest.raises(ValidationError):
        MonitoringConfig.model_validate(document)


def test_a_calibration_band_must_contain_one_and_a_subgroup_needs_one_condition() -> None:
    document = committed_document()
    document["thresholds"]["calibration_ratio"]["band_low"] = 1.05
    with pytest.raises(ValidationError):
        MonitoringConfig.model_validate(document)
    document = committed_document()
    document["stream"]["scenarios"]["covariate_shift_region"]["covariate_shift"]["subgroup"]["below"] = 3
    with pytest.raises(ValidationError):
        MonitoringConfig.model_validate(document)
