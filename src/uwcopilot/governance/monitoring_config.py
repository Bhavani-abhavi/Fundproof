"""Settings for continuous monitoring (Phase 6 B), read from config/monitoring.yaml.

Every threshold lives in that one file; this module only checks that the file is complete and consistent (unknown
keys are refused, so a typo cannot silently disable a check) and turns it into typed objects.

    config = load_monitoring_config()
    config.thresholds.input_drift_psi.rule.level(0.31)    # -> "alert"

How a LevelRule reads a value (the YAML comment says the same):
    higher_is_worse: "warn" when value >= warn, "alert" when value > alert
    lower_is_worse:  "warn" when value <= warn, "alert" when value < alert
so a PSI of exactly 0.10 is a "moderate shift" (warn) and 0.25 is still moderate, as in risk/stability.py.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from uwcopilot.common.paths import CONFIG_DIR

DEFAULT_CONFIG_PATH = CONFIG_DIR / "monitoring.yaml"
Level = Literal["ok", "warn", "alert", "insufficient_data"]
Direction = Literal["higher_is_worse", "lower_is_worse"]
# The indicators read from results files (monitoring_indicators.py). Each needs a rule in the config.
INDICATOR_NAMES: tuple[str, ...] = (
    "citation_support_rate",
    "ocr_page_share",
    "low_confidence_field_share",
    "required_escalations_escalated",
    "unsupported_claim_rate",
    "override_rate",
)


class StrictModel(BaseModel):
    """A settings object: frozen, and unknown keys are an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class LevelRule(StrictModel):
    """Turns one value into ok / warn / alert (module docstring). warn or alert may be left out."""

    direction: Direction
    warn: float | None = None
    alert: float | None = None
    rationale: str = Field(min_length=1, description="Why this value: every threshold is a design choice")

    @model_validator(mode="after")
    def _warn_comes_before_alert(self) -> "LevelRule":
        """At least one level is set, and the warn level is reached before the alert level."""
        if self.warn is None and self.alert is None:
            raise ValueError("a rule needs a warn level, an alert level or both")
        if self.warn is not None and self.alert is not None:
            if self.direction == "higher_is_worse" and self.warn > self.alert:
                raise ValueError("higher_is_worse: warn must not be above alert")
            if self.direction == "lower_is_worse" and self.warn < self.alert:
                raise ValueError("lower_is_worse: warn must not be below alert")
        return self

    def level(self, value: float) -> Level:
        """The level of one value under this rule."""
        if self.direction == "higher_is_worse":
            if self.alert is not None and value > self.alert:
                return "alert"
            if self.warn is not None and value >= self.warn:
                return "warn"
            return "ok"
        if self.alert is not None and value < self.alert:
            return "alert"
        if self.warn is not None and value <= self.warn:
            return "warn"
        return "ok"

    def describe(self) -> str:
        """The rule in words, for alert records and the report, e.g. 'warn >= 0.1, alert > 0.25'."""
        warn_sign, alert_sign = (">=", ">") if self.direction == "higher_is_worse" else ("<=", "<")
        parts = []
        if self.warn is not None:
            parts.append(f"warn {warn_sign} {self.warn:g}")
        if self.alert is not None:
            parts.append(f"alert {alert_sign} {self.alert:g}")
        return ", ".join(parts)


class PsiSettings(StrictModel):
    """PSI with n_bins quantile bins taken from the reference sample, and the rule for its value."""

    n_bins: int = Field(ge=2, le=50)
    rule: LevelRule

    @model_validator(mode="after")
    def _higher_is_worse(self) -> "PsiSettings":
        """A larger PSI is always a larger shift."""
        if self.rule.direction != "higher_is_worse":
            raise ValueError("a PSI rule must be higher_is_worse")
        return self


class CalibrationRatioSettings(StrictModel):
    """Observed / predicted claims: warn outside [band_low, band_high], alert when the whole interval is outside."""

    band_low: float = Field(gt=0)
    band_high: float = Field(gt=0)
    confidence: float = Field(gt=0, lt=1)
    min_expected_claims: float = Field(ge=0)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _band_contains_one(self) -> "CalibrationRatioSettings":
        """A perfectly calibrated model (ratio 1) must be inside the band."""
        if not self.band_low < 1.0 < self.band_high:
            raise ValueError("the calibration band must contain 1.0")
        return self

    def describe(self) -> str:
        """The rule in words."""
        return (
            f"warn: ratio outside [{self.band_low:g}, {self.band_high:g}]; alert: whole "
            f"{self.confidence:.0%} interval outside it; at least {self.min_expected_claims:g} predicted claims"
        )


class EceSettings(StrictModel):
    """Expected calibration error of P(claim > 0) and the rule for its value."""

    n_bins: int = Field(ge=2, le=50)
    strategy: Literal["uniform", "quantile"]
    rule: LevelRule

    @model_validator(mode="after")
    def _higher_is_worse(self) -> "EceSettings":
        """A larger ECE is always worse."""
        if self.rule.direction != "higher_is_worse":
            raise ValueError("an ECE rule must be higher_is_worse")
        return self


class Thresholds(StrictModel):
    """Every monitoring threshold."""

    input_drift_psi: PsiSettings
    score_drift_psi: PsiSettings
    calibration_ratio: CalibrationRatioSettings
    calibration_ece: EceSettings
    indicators: dict[str, LevelRule]

    @model_validator(mode="after")
    def _every_indicator_has_a_rule(self) -> "Thresholds":
        """Exactly the indicators the code reads, no more and no fewer."""
        missing = sorted(set(INDICATOR_NAMES) - set(self.indicators))
        unknown = sorted(set(self.indicators) - set(INDICATOR_NAMES))
        if missing or unknown:
            raise ValueError(f"indicator rules: missing {missing}, unknown {unknown}")
        return self


class Subgroup(StrictModel):
    """Rows of one subgroup: a numeric column below a value, or a text column equal to a value."""

    column: str = Field(min_length=1)
    below: float | None = None
    equals: str | None = None

    @model_validator(mode="after")
    def _exactly_one_condition(self) -> "Subgroup":
        """Either `below` or `equals`, not both."""
        if (self.below is None) == (self.equals is None):
            raise ValueError("a subgroup needs exactly one of `below` and `equals`")
        return self

    def describe(self) -> str:
        """The condition in words, e.g. 'DrivAge < 26'."""
        if self.below is not None:
            return f"{self.column} < {self.below:g}"
        return f"{self.column} == {self.equals}"


class CovariateShift(StrictModel):
    """Resample each drifted window so that target_share of its rows come from the subgroup."""

    subgroup: Subgroup
    target_share: float = Field(gt=0, lt=1)


class Scenario(StrictModel):
    """One SYNTHETIC stream scenario. Without covariate_shift and with multiplier 1 it has no drift."""

    description: str = Field(min_length=1)
    covariate_shift: CovariateShift | None = None
    claim_rate_multiplier: float = Field(default=1.0, gt=0)

    @property
    def drifted(self) -> bool:
        """True when the scenario injects any drift."""
        return self.covariate_shift is not None or self.claim_rate_multiplier != 1.0


class Demonstration(StrictModel):
    """Which validation stream is reported window by window, alerted on and written to the registry."""

    scenario: str
    stream: int = Field(ge=0)


class StreamSettings(StrictModel):
    """The SIMULATED stream and its scenarios."""

    window_size: int = Field(ge=10)
    n_windows: int = Field(ge=2)
    drift_start_window: int = Field(ge=0)
    n_streams: int = Field(ge=1)
    seed: int
    scenarios: dict[str, Scenario]
    demonstration: Demonstration

    @model_validator(mode="after")
    def _consistent(self) -> "StreamSettings":
        """Drift starts inside the stream, a no-drift scenario exists, the demonstration names a real stream."""
        if not 0 < self.drift_start_window < self.n_windows:
            raise ValueError("drift_start_window must leave at least one clean and one drifted window")
        if not any(not scenario.drifted for scenario in self.scenarios.values()):
            raise ValueError("at least one scenario must have no drift (it measures the false alert rate)")
        if self.demonstration.scenario not in self.scenarios:
            raise ValueError(f"demonstration scenario {self.demonstration.scenario!r} is not defined")
        if self.demonstration.stream >= self.n_streams:
            raise ValueError("the demonstration stream must be one of the validation streams")
        return self


class MonitoredModel(StrictModel):
    """The model the stream is scored with; its id is the one marked under review on a breach."""

    model_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)


class BootstrapSettings(StrictModel):
    """Percentile bootstrap for the validation rates (unit = stream)."""

    resamples: int = Field(ge=100)
    seed: int
    confidence: float = Field(gt=0, lt=1)


class MonitoringConfig(StrictModel):
    """The whole of config/monitoring.yaml."""

    data_label: Literal["SYNTHETIC"]
    monitored_model: MonitoredModel
    thresholds: Thresholds
    stream: StreamSettings
    bootstrap: BootstrapSettings


def load_monitoring_config(path: Path = DEFAULT_CONFIG_PATH) -> MonitoringConfig:
    """Read and validate config/monitoring.yaml (pydantic ValidationError when it is incomplete or inconsistent)."""
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return MonitoringConfig.model_validate(document)
