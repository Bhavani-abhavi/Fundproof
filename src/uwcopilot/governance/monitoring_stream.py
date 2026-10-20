"""SIMULATED streams that validate the monitor, and the validation summary (Phase 6 B).

freMTPL2 has no date field, so there is no real stream of new business and no out-of-time split. A stream here is
n_windows windows drawn at random from the held-out split, and drift is INJECTED from window drift_start_window on by
a SYNTHETIC scenario (config/monitoring.yaml):
- covariate shift: each drifted window is resampled so that target_share of its rows come from a subgroup (for
  example drivers under 26). The recorded outcomes of those rows are kept;
- claim-rate drift: the outcomes of each drifted window are SIMULATED. With multiplier m > 1, extra claims drawn from
  Poisson((m - 1) x predicted claims) are added to the recorded claims; with m < 1 each recorded claim is kept with
  probability m. The inputs do not change, so only the calibration checks can see this drift.

Random numbers: stream i of every scenario uses numpy's default_rng([seed, i]). Clean windows consume the same random
numbers in every scenario, so windows 0 .. drift_start_window - 1 are identical across scenarios (common random
numbers) and differences between scenarios come from the drift alone.

Validation, unit = stream:
- false alert rate: share of no-drift streams with at least one alert (any check, any window);
- detection rate: share of drifted streams with an alert in a drifted window;
- detection delay: first drifted window with an alert minus drift_start_window (0 = caught in the first drifted
  window), over the detected streams;
- pre-drift false alert rate: share of drifted streams with an alert before the drift started;
- warn rate (no-drift streams): share with at least one check at level warn or worse, overall and by check.
Every rate and the mean delay get a percentile bootstrap CI over streams (risk/bootstrap.py). All streams draw from
one held-out pool, so they are not independent samples of the world: the intervals describe stream-to-stream
variation for this pool and this model.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from uwcopilot.governance.monitoring import CHECK_NAMES, LEVEL_RANK, MonitoredPool, WindowResult, evaluate_window
from uwcopilot.governance.monitoring_config import (
    BootstrapSettings,
    Scenario,
    StreamSettings,
    Subgroup,
    Thresholds,
)
from uwcopilot.risk.bootstrap import bootstrap_ci


@dataclass(frozen=True)
class Window:
    """One window of a stream: pool rows and their (possibly SIMULATED) claim counts, in the same order."""

    index: int
    drifted: bool
    rows: np.ndarray
    observed_claims: np.ndarray


@dataclass(frozen=True)
class SubgroupRows:
    """Pool rows inside and outside a subgroup."""

    inside: np.ndarray
    outside: np.ndarray


def subgroup_rows(pool_frame: pd.DataFrame, subgroup: Subgroup) -> SubgroupRows:
    """Split the pool rows by the subgroup condition (numeric `below` or text `equals`)."""
    values = pool_frame[subgroup.column]
    if subgroup.below is not None:
        mask = values.to_numpy(dtype=float) < subgroup.below
    else:
        mask = values.astype(str).to_numpy() == subgroup.equals
    inside, outside = np.flatnonzero(mask), np.flatnonzero(~mask)
    if inside.size == 0 or outside.size == 0:
        raise ValueError(f"subgroup {subgroup.describe()} must have rows inside and outside it")
    return SubgroupRows(inside=inside, outside=outside)


def draw_rows(
    rng: np.random.Generator, pool_size: int, window_size: int, shift: tuple[SubgroupRows, float] | None = None
) -> np.ndarray:
    """Rows of one window, without replacement inside the window.

    With shift = (subgroup rows, target share), round(target share x window_size) rows come from the subgroup and
    the rest from outside it.
    """
    if shift is None:
        if window_size > pool_size:
            raise ValueError("the window is larger than the pool")
        return rng.choice(pool_size, size=window_size, replace=False)
    groups, share = shift
    n_inside = int(round(share * window_size))
    n_outside = window_size - n_inside
    if n_inside > groups.inside.size or n_outside > groups.outside.size:
        raise ValueError("the subgroup has too few rows for this window size and target share")
    inside = rng.choice(groups.inside, size=n_inside, replace=False)
    outside = rng.choice(groups.outside, size=n_outside, replace=False)
    return np.concatenate([inside, outside])


def simulate_outcomes(
    rng: np.random.Generator, recorded: np.ndarray, expected: np.ndarray, multiplier: float
) -> np.ndarray:
    """Claim counts of a window: recorded (m = 1), plus SIMULATED extra claims (m > 1), or thinned (m < 1)."""
    if multiplier <= 0:
        raise ValueError("the claim-rate multiplier must be positive")
    if multiplier == 1.0:
        return recorded.copy()
    if multiplier > 1.0:
        return recorded + rng.poisson((multiplier - 1.0) * expected)
    return rng.binomial(recorded, multiplier)


def simulate_stream(
    pool: MonitoredPool,
    scenario: Scenario,
    settings: StreamSettings,
    stream_index: int,
    shift_rows: SubgroupRows | None = None,
) -> list[Window]:
    """The windows of one SIMULATED stream (module docstring). shift_rows is needed for a covariate shift."""
    if scenario.covariate_shift is not None and shift_rows is None:
        raise ValueError("a covariate-shift scenario needs the subgroup rows")
    rng = np.random.default_rng([settings.seed, stream_index])
    windows = []
    for index in range(settings.n_windows):
        drifted = scenario.drifted and index >= settings.drift_start_window
        shift = None
        if drifted and scenario.covariate_shift is not None and shift_rows is not None:
            shift = (shift_rows, scenario.covariate_shift.target_share)
        rows = draw_rows(rng, pool.size, settings.window_size, shift)
        multiplier = scenario.claim_rate_multiplier if drifted else 1.0
        observed = simulate_outcomes(rng, pool.recorded_claims[rows], pool.expected_claims[rows], multiplier)
        windows.append(Window(index=index, drifted=drifted, rows=rows, observed_claims=observed))
    return windows


def evaluate_stream(pool: MonitoredPool, windows: Sequence[Window], thresholds: Thresholds) -> list[WindowResult]:
    """Run every check on every window of a stream."""
    return [evaluate_window(pool, window.rows, window.observed_claims, thresholds, window.index) for window in windows]


# --------------------------------------------------------------------------------------------------------------
# Validation summary
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamOutcome:
    """What the monitor said on one stream: which windows alerted or warned, and which checks did so where.

    warn_windows and warning_checks count level warn or worse; alerting_checks counts level alert only.
    """

    stream: int
    alert_windows: tuple[int, ...]
    warn_windows: tuple[int, ...]
    alerting_checks: tuple[tuple[int, str], ...]
    warning_checks: tuple[tuple[int, str], ...] = ()


def stream_outcome(stream: int, results: Sequence[WindowResult]) -> StreamOutcome:
    """Summarise the window results of one stream."""
    warn_rank = LEVEL_RANK["warn"]
    alert_windows = tuple(result.window for result in results if result.alerts)
    warn_windows = tuple(result.window for result in results if LEVEL_RANK[result.level] >= warn_rank)
    alerting = sorted({(result.window, check.check) for result in results for check in result.alerts})
    warning = sorted(
        {
            (result.window, check.check)
            for result in results
            for check in result.checks
            if LEVEL_RANK[check.level] >= warn_rank
        }
    )
    return StreamOutcome(stream, alert_windows, warn_windows, tuple(alerting), tuple(warning))


def run_scenario(
    pool: MonitoredPool,
    scenario: Scenario,
    settings: StreamSettings,
    thresholds: Thresholds,
    shift_rows: SubgroupRows | None = None,
) -> list[StreamOutcome]:
    """Simulate and check settings.n_streams streams of one scenario."""
    outcomes = []
    for stream_index in range(settings.n_streams):
        windows = simulate_stream(pool, scenario, settings, stream_index, shift_rows)
        outcomes.append(stream_outcome(stream_index, evaluate_stream(pool, windows, thresholds)))
    return outcomes


def rate_with_ci(indicators: Sequence[bool], bootstrap: BootstrapSettings) -> dict[str, Any]:
    """Share of True with a percentile bootstrap CI (unit = stream), plus numerator and denominator."""
    values = np.asarray(indicators, dtype=float)
    if values.size == 0:
        raise ValueError("a rate needs at least one stream")
    interval = bootstrap_ci(
        np.mean, values, n_resamples=bootstrap.resamples, seed=bootstrap.seed, alpha=1 - bootstrap.confidence
    )
    return {
        "estimate": interval["estimate"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "numerator": int(values.sum()),
        "denominator": int(values.size),
        "n_resamples": interval["n_resamples"],
        "seed": interval["seed"],
        "confidence": bootstrap.confidence,
        "method": "percentile bootstrap",
        "unit": "stream",
    }


def first_alert_delay(alert_windows: Sequence[int], drift_start: int) -> int | None:
    """Windows from the drift start to the first alert at or after it; None when nothing alerted after it."""
    after = [window for window in alert_windows if window >= drift_start]
    return min(after) - drift_start if after else None


def delay_summary(delays: Sequence[int], bootstrap: BootstrapSettings) -> dict[str, Any]:
    """Mean delay with a bootstrap CI over detected streams, the median, the maximum and the distribution."""
    if not delays:
        return {"status": "not available", "reason": "no stream was detected, so there is no delay to report"}
    values = np.asarray(delays, dtype=float)
    interval = bootstrap_ci(
        np.mean, values, n_resamples=bootstrap.resamples, seed=bootstrap.seed, alpha=1 - bootstrap.confidence
    )
    distribution = Counter(int(delay) for delay in delays)
    return {
        "unit": "windows after the drift started (0 = the first drifted window)",
        "mean": {
            "estimate": interval["estimate"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "n_resamples": interval["n_resamples"],
            "seed": interval["seed"],
            "confidence": bootstrap.confidence,
            "method": "percentile bootstrap",
            "unit": "detected stream",
        },
        "median": float(np.median(values)),
        "max": int(values.max()),
        "n_detected_streams": int(values.size),
        "streams_by_delay": {str(delay): distribution[delay] for delay in sorted(distribution)},
    }


def summarize_scenario(
    outcomes: Sequence[StreamOutcome],
    scenario: Scenario,
    settings: StreamSettings,
    bootstrap: BootstrapSettings,
) -> dict[str, Any]:
    """Validation metrics of one scenario (module docstring)."""
    start = settings.drift_start_window
    total_windows = len(outcomes) * settings.n_windows
    alert_window_count = sum(len(outcome.alert_windows) for outcome in outcomes)
    summary: dict[str, Any] = {
        "description": scenario.description,
        "data_label": "SIMULATED",
        "drifted": scenario.drifted,
        "n_streams": len(outcomes),
        "windows_with_an_alert": {
            "numerator": alert_window_count,
            "denominator": total_windows,
            "estimate": alert_window_count / total_windows,
        },
    }
    if not scenario.drifted:
        summary["false_alert_rate"] = rate_with_ci([bool(outcome.alert_windows) for outcome in outcomes], bootstrap)
        summary["warn_rate"] = rate_with_ci([bool(outcome.warn_windows) for outcome in outcomes], bootstrap)
        summary["warn_rate_by_check"] = {
            check: rate_with_ci([any(name == check for _, name in o.warning_checks) for o in outcomes], bootstrap)
            for check in CHECK_NAMES
        }
        summary["false_alert_rate_by_check"] = {
            check: rate_with_ci([any(name == check for _, name in o.alerting_checks) for o in outcomes], bootstrap)
            for check in CHECK_NAMES
        }
        return summary
    delays = [first_alert_delay(outcome.alert_windows, start) for outcome in outcomes]
    summary["detection_rate"] = rate_with_ci([delay is not None for delay in delays], bootstrap)
    summary["detection_delay_windows"] = delay_summary([delay for delay in delays if delay is not None], bootstrap)
    summary["pre_drift_false_alert_rate"] = rate_with_ci(
        [any(window < start for window in outcome.alert_windows) for outcome in outcomes], bootstrap
    )
    summary["detection_rate_by_check"] = {
        check: rate_with_ci(
            [any(window >= start and name == check for window, name in o.alerting_checks) for o in outcomes],
            bootstrap,
        )
        for check in CHECK_NAMES
    }
    return summary
