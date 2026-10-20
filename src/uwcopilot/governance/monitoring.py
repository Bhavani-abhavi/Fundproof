"""Monitoring checks for one window of new business, and what a breach does (Phase 6 B).

Checks of the monitored claim-frequency model (thresholds only in config/monitoring.yaml):
- input drift: PSI of each rating feature, window vs the reference sample (the model's training split);
- score drift: PSI of the predicted claim frequency (claims per policy-year);
- calibration ratio: observed / predicted claims, with an exact (Garwood) Poisson interval for the observed count
  divided by the predicted count (the prediction is treated as fixed, as in an observed/expected ratio);
- calibration ECE: expected calibration error of P(at least one claim) = 1 - exp(-predicted claims) against
  "the policy had a claim" (risk/metrics.py expected_calibration_error).

PSI uses the bins and the formula of risk/stability.py: quantile bins of the reference sample (duplicate edges
removed, outer bins open), a missing-value bin, one bin per category (a category not seen in the reference gets its
own bin), empty bins floored at EMPTY_BIN_SHARE. The only difference is speed: the bin of every candidate row is
computed once (VariableBins.codes), so the PSI of a window is one bincount. A test checks that the values equal
population_stability_index and categorical_population_stability_index.

A breach is a check at level "alert". alert_records() turns the breaches of a window into AlertRecord objects;
record_alerts() appends them to the alert log (JSONL, data/governance/) and, for a record that names a model, calls
ModelStatusStore.mark_under_review. Monitoring can never do more than that: promoting a challenger or closing a
review needs a model validator (governance/model_status.py).
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import chi2

from uwcopilot.governance.model_status import ModelStatusStore
from uwcopilot.governance.monitoring_config import (
    CalibrationRatioSettings,
    EceSettings,
    Level,
    PsiSettings,
    Thresholds,
)
from uwcopilot.risk.metrics import expected_calibration_error
from uwcopilot.risk.stability import _psi_from_counts, quantile_bin_edges

SCORE_NAME = "predicted_frequency"
MISSING_CATEGORY = "<missing>"  # the same label risk/stability.py gives missing categorical values
LEVEL_RANK: dict[str, int] = {"ok": 0, "insufficient_data": 0, "warn": 1, "alert": 2}
CHECK_NAMES: tuple[str, ...] = ("input_drift", "score_drift", "calibration_ratio", "calibration_ece")
MODEL_ACTION = "mark_under_review"
RECORD_ONLY_ACTION = "record_only"


# --------------------------------------------------------------------------------------------------------------
# PSI with bins computed once
# --------------------------------------------------------------------------------------------------------------


def category_text(values: Sequence[object] | np.ndarray | pd.Series) -> pd.Series:
    """Categorical values as text, missing values as '<missing>' (as risk/stability.py does)."""
    return pd.Series(values, dtype="object").fillna(MISSING_CATEGORY).astype(str).reset_index(drop=True)


@dataclass(frozen=True)
class VariableBins:
    """The PSI bins of one variable, learned from the reference sample.

    numeric: interior edges at the reference quantiles (risk/stability.py quantile_bin_edges); categorical: the
    reference categories in sorted order.
    """

    name: str
    kind: Literal["numeric", "categorical"]
    edges: tuple[float, ...] = ()
    categories: tuple[str, ...] = ()

    @classmethod
    def from_reference(
        cls, name: str, values: Sequence[object] | np.ndarray | pd.Series, n_bins: int
    ) -> "VariableBins":
        """Numeric bins for a numeric column, one bin per category otherwise."""
        series = pd.Series(values)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numbers = series.to_numpy(dtype=float)
            present = numbers[~np.isnan(numbers)]
            if present.size == 0:
                raise ValueError(f"{name}: the reference sample has no non-missing value")
            edges = quantile_bin_edges(present, n_bins)
            return cls(name=name, kind="numeric", edges=tuple(float(edge) for edge in edges))
        return cls(name=name, kind="categorical", categories=tuple(sorted(set(category_text(series)))))

    def codes(self, values: Sequence[object] | np.ndarray | pd.Series) -> tuple[np.ndarray, int]:
        """(bin index of every value, number of bins).

        numeric: bins 0..len(edges) by value (a value equal to an edge goes to the bin above), then one bin for
        missing values. categorical: the reference categories, then one bin per unseen category (sorted).
        """
        if self.kind == "numeric":
            numbers = np.asarray(values, dtype=float)
            edges = np.asarray(self.edges, dtype=float)
            n_value_bins = edges.shape[0] + 1
            codes = np.searchsorted(edges, numbers, side="right")
            codes[np.isnan(numbers)] = n_value_bins
            return codes.astype(np.int64), n_value_bins + 1
        text = category_text(values)
        unseen = sorted(set(text) - set(self.categories))
        index = {category: position for position, category in enumerate([*self.categories, *unseen])}
        return text.map(index).to_numpy(dtype=np.int64), len(index)


def psi_from_counts(expected_counts: np.ndarray, actual_counts: np.ndarray) -> float:
    """PSI from bin counts over the same bins, with risk/stability.py's own formula (empty bins floored)."""
    return _psi_from_counts(np.asarray(expected_counts, dtype=float), np.asarray(actual_counts, dtype=float))


# --------------------------------------------------------------------------------------------------------------
# The candidate rows of a stream, precomputed
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitoredPool:
    """Rows that windows are drawn from, with everything a check needs computed once.

    codes / n_codes / reference_counts are keyed by variable: every input feature, then SCORE_NAME.
    expected_claims = predicted frequency x exposure; claim_probability = 1 - exp(-expected_claims).
    """

    input_features: tuple[str, ...]
    codes: dict[str, np.ndarray]
    n_codes: dict[str, int]
    reference_counts: dict[str, np.ndarray]
    expected_claims: np.ndarray
    claim_probability: np.ndarray
    recorded_claims: np.ndarray
    exposure: np.ndarray

    @property
    def size(self) -> int:
        """Number of candidate rows."""
        return int(self.expected_claims.shape[0])


def _coded(bins: VariableBins, reference_values: object, pool_values: object) -> tuple[np.ndarray, int, np.ndarray]:
    """(pool codes, number of bins, reference counts per bin) for one variable."""
    pool_codes, n_pool_codes = bins.codes(pool_values)
    reference_codes, n_reference_codes = bins.codes(reference_values)
    n_codes = max(n_pool_codes, n_reference_codes)
    return pool_codes, n_codes, np.bincount(reference_codes, minlength=n_codes).astype(float)


def build_monitored_pool(
    reference: pd.DataFrame,
    reference_frequency: np.ndarray,
    pool: pd.DataFrame,
    pool_frequency: np.ndarray,
    features: Sequence[str],
    thresholds: Thresholds,
) -> MonitoredPool:
    """Bin the pool against the reference sample and attach the model's predictions and the recorded claims.

    reference / pool hold the rating features, Exposure and ClaimNb; *_frequency are the model's predicted
    claims per policy-year for the same rows.
    """
    codes: dict[str, np.ndarray] = {}
    n_codes: dict[str, int] = {}
    reference_counts: dict[str, np.ndarray] = {}
    for feature in features:
        bins = VariableBins.from_reference(feature, reference[feature], thresholds.input_drift_psi.n_bins)
        codes[feature], n_codes[feature], reference_counts[feature] = _coded(bins, reference[feature], pool[feature])
    score_bins = VariableBins.from_reference(SCORE_NAME, reference_frequency, thresholds.score_drift_psi.n_bins)
    codes[SCORE_NAME], n_codes[SCORE_NAME], reference_counts[SCORE_NAME] = _coded(
        score_bins, reference_frequency, pool_frequency
    )
    exposure = pool["Exposure"].to_numpy(dtype=float)
    expected_claims = np.asarray(pool_frequency, dtype=float) * exposure
    return MonitoredPool(
        input_features=tuple(features),
        codes=codes,
        n_codes=n_codes,
        reference_counts=reference_counts,
        expected_claims=expected_claims,
        claim_probability=1.0 - np.exp(-expected_claims),
        recorded_claims=pool["ClaimNb"].to_numpy(dtype=np.int64),
        exposure=exposure,
    )


# --------------------------------------------------------------------------------------------------------------
# Checks of one window
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """One check of one window: the value, its level and the rule it was read with."""

    check: str
    subject: str
    value: float | None
    level: Level
    rule: str
    interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class WindowResult:
    """Every check of one window."""

    window: int
    n_policies: int
    exposure_years: float
    observed_claims: int
    predicted_claims: float
    checks: tuple[CheckResult, ...]

    @property
    def alerts(self) -> list[CheckResult]:
        """The checks at level alert (the breaches)."""
        return [check for check in self.checks if check.level == "alert"]

    @property
    def level(self) -> Level:
        """The worst level of any check (insufficient_data counts as ok)."""
        worst = max(self.checks, key=lambda check: LEVEL_RANK[check.level])
        return worst.level if LEVEL_RANK[worst.level] > 0 else "ok"

    def find(self, check: str, subject: str) -> CheckResult:
        """One check by name and subject; KeyError when it is not there."""
        for item in self.checks:
            if item.check == check and item.subject == subject:
                return item
        raise KeyError(f"no check {check!r} on {subject!r}")


def psi_check(pool: MonitoredPool, rows: np.ndarray, variable: str, settings: PsiSettings, check: str) -> CheckResult:
    """PSI of one variable on the window rows vs the reference sample."""
    actual = np.bincount(pool.codes[variable][rows], minlength=pool.n_codes[variable])
    value = psi_from_counts(pool.reference_counts[variable], actual)
    return CheckResult(check, variable, value, settings.rule.level(value), settings.rule.describe())


def poisson_count_interval(count: int, confidence: float) -> tuple[float, float]:
    """Exact (Garwood) interval for the mean of a Poisson count: chi-square quantiles / 2."""
    if count < 0:
        raise ValueError("a count cannot be negative")
    alpha = 1.0 - confidence
    low = 0.0 if count == 0 else float(chi2.ppf(alpha / 2, 2 * count) / 2)
    high = float(chi2.ppf(1 - alpha / 2, 2 * count + 2) / 2)
    return low, high


def calibration_ratio_check(observed: int, predicted: float, settings: CalibrationRatioSettings) -> CheckResult:
    """Observed / predicted claims: warn outside the band, alert when the whole interval is outside it."""
    rule = settings.describe()
    if predicted < settings.min_expected_claims or predicted <= 0:
        return CheckResult("calibration_ratio", "observed/predicted claims", None, "insufficient_data", rule)
    count_low, count_high = poisson_count_interval(observed, settings.confidence)
    ratio = observed / predicted
    interval = (count_low / predicted, count_high / predicted)
    if interval[0] > settings.band_high or interval[1] < settings.band_low:
        level: Level = "alert"
    elif not settings.band_low <= ratio <= settings.band_high:
        level = "warn"
    else:
        level = "ok"
    return CheckResult("calibration_ratio", "observed/predicted claims", ratio, level, rule, interval)


def ece_check(had_claim: np.ndarray, claim_probability: np.ndarray, settings: EceSettings) -> CheckResult:
    """Expected calibration error of P(claim > 0) on the window."""
    value = expected_calibration_error(
        had_claim.astype(float), claim_probability, n_bins=settings.n_bins, strategy=settings.strategy
    )
    return CheckResult("calibration_ece", "P(claim > 0)", value, settings.rule.level(value), settings.rule.describe())


def evaluate_window(
    pool: MonitoredPool, rows: np.ndarray, observed_claims: np.ndarray, thresholds: Thresholds, window: int = 0
) -> WindowResult:
    """Run every check on one window: pool rows `rows` whose outcomes are `observed_claims` (same order)."""
    rows = np.asarray(rows, dtype=np.int64)
    observed_claims = np.asarray(observed_claims, dtype=np.int64)
    if rows.shape != observed_claims.shape or rows.size == 0:
        raise ValueError("rows and observed_claims must be non-empty and of the same length")
    checks = [
        psi_check(pool, rows, feature, thresholds.input_drift_psi, "input_drift") for feature in pool.input_features
    ]
    checks.append(psi_check(pool, rows, SCORE_NAME, thresholds.score_drift_psi, "score_drift"))
    predicted = float(pool.expected_claims[rows].sum())
    observed = int(observed_claims.sum())
    checks.append(calibration_ratio_check(observed, predicted, thresholds.calibration_ratio))
    checks.append(ece_check(observed_claims > 0, pool.claim_probability[rows], thresholds.calibration_ece))
    return WindowResult(
        window=window,
        n_policies=int(rows.size),
        exposure_years=float(pool.exposure[rows].sum()),
        observed_claims=observed,
        predicted_claims=predicted,
        checks=tuple(checks),
    )


# --------------------------------------------------------------------------------------------------------------
# Breaches: alert records, the alert log and the model status
# --------------------------------------------------------------------------------------------------------------


class AlertRecord(BaseModel):
    """One breach, as written to the alert log. Holds ids, values and rules, never data rows."""

    model_config = ConfigDict(frozen=True)

    alert_id: str = Field(min_length=1)
    created_at_utc: datetime
    run_id: str = Field(min_length=1)
    data_label: str = "SYNTHETIC"
    component: str = Field(min_length=1, description="What is monitored, e.g. fremtpl2_frequency_model")
    model_id: str | None = Field(default=None, description="The registered model to mark under review, if any")
    check: str
    subject: str
    window: int | None = None
    scenario: str | None = None
    value: float
    interval: tuple[float, float] | None = None
    rule: str
    action: Literal["mark_under_review", "record_only"]

    @model_validator(mode="after")
    def _a_model_to_mark(self) -> "AlertRecord":
        """Only a record that names a model can ask for it to be marked under review."""
        if self.action == MODEL_ACTION and not self.model_id:
            raise ValueError("action mark_under_review needs a model_id")
        return self

    def reason(self) -> str:
        """One line for the model status history."""
        where = f" in window {self.window}" if self.window is not None else ""
        return (
            f"monitoring alert {self.alert_id}: {self.check} on {self.subject} = {self.value:.4g}{where} ({self.rule})"
        )


def utc_now() -> datetime:
    """The current time in UTC, without microseconds."""
    return datetime.now(UTC).replace(microsecond=0)


def new_run_id(at: datetime | None = None) -> str:
    """A run id from the UTC time, e.g. 20261020T120000Z."""
    return (at or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def alert_records(
    breaches: Sequence[CheckResult],
    run_id: str,
    first_number: int,
    component: str,
    model_id: str | None,
    window: int | None = None,
    scenario: str | None = None,
    at: datetime | None = None,
) -> list[AlertRecord]:
    """One AlertRecord per breach, numbered from first_number within the run (MON-<run_id>-0001, ...).

    A record that names a model asks for the model to be marked under review; one without a model (for example the
    document extraction indicators, which are not registered models) is only recorded.
    """
    created = at or utc_now()
    records = []
    for offset, breach in enumerate(breaches):
        if breach.level != "alert" or breach.value is None:
            raise ValueError("only checks at level alert with a value are breaches")
        records.append(
            AlertRecord(
                alert_id=f"MON-{run_id}-{first_number + offset:04d}",
                created_at_utc=created,
                run_id=run_id,
                component=component,
                model_id=model_id,
                check=breach.check,
                subject=breach.subject,
                window=window,
                scenario=scenario,
                value=breach.value,
                interval=breach.interval,
                rule=breach.rule,
                action=MODEL_ACTION if model_id else RECORD_ONLY_ACTION,
            )
        )
    return records


def append_alerts(path: Path, records: Sequence[AlertRecord]) -> None:
    """Append records to the JSONL alert log (one JSON object per line). The log is never rewritten."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")


def read_alerts(path: Path) -> list[AlertRecord]:
    """Every record in the alert log (empty when the log does not exist)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AlertRecord.model_validate_json(line) for line in lines if line.strip()]


def record_alerts(records: Sequence[AlertRecord], alert_log: Path, store: ModelStatusStore | None) -> int:
    """Write the records to the alert log first, then mark each named model under review.

    Returns the number of records whose model was marked (a model already under review stays as it is). With
    store=None nothing is marked, so records asking for it are refused: the caller must say where the registry is.
    """
    to_mark = [record for record in records if record.action == MODEL_ACTION]
    if to_mark and store is None:
        raise ValueError("records ask to mark a model under review but no model status store was given")
    append_alerts(alert_log, records)
    if store is None:
        return 0
    for record in to_mark:
        store.mark_under_review(str(record.model_id), reason=record.reason(), alert_id=record.alert_id)
    return len(to_mark)
