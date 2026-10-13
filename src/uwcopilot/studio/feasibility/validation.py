"""Validation on SYNTHETIC worlds with a known true future (Studio S3; rule E).

For each world (worlds.py) the simulator runs n paths and produces its P10-P90 bands and probabilities. A hidden
"true" future of the same world is drawn separately, under one of two conditions:

- well_specified: the truth is one more draw from the simulator's own model (same distributions, same structure).
  The P10-P90 band should then hold the truth about 80% of the time; this checks the code, not the world.
- misspecified: the truth has effects the simulator does NOT model (generator parameters below, chosen by the
  project author): fat-tailed monthly noise (Student t with MISSPEC_T_DF degrees of freedom, scaled to unit
  variance), a site-level demand error the site's range does not admit (a mean-one lognormal factor with log-sd
  MISSPEC_LEVEL_SD), and with probability DOWNTURN_PROBABILITY a local downturn that cuts demand by a uniform
  DOWNTURN_DEPTH share for DOWNTURN_MONTHS months. Results here are reported as they come out.

The same worlds and the same forecasts are scored against both truths; only the hidden truth changes.

Scores per world:
- band coverage: the share of months whose true revenue (and true net income) lies within the forecast P10-P90;
- Brier score of P(year-2 DSCR >= target) and of P(cash < 0 within 24 months) against the true outcome;
- the same Brier scores for two baselines: a single forecast (every input at its most likely value, no shocks, no
  noise: the plan a spreadsheet shows, read as yes = 1 / no = 0) and the base rate (the share of worlds with the
  outcome, computed in-sample on the same worlds).
Validation on worlds whose generator the same author wrote shows the simulator is internally consistent and how it
degrades when its assumptions are wrong; it says nothing about real-world accuracy.
"""

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from uwcopilot.risk.bootstrap import bootstrap_ci, paired_bootstrap_diff
from uwcopilot.studio.data.fabric_base import FactorValue
from uwcopilot.studio.feasibility.assumptions import FeasibilityAssumptions
from uwcopilot.studio.feasibility.metrics import bands, cash_negative_within, dscr_by_year
from uwcopilot.studio.feasibility.model import (
    Draws,
    PathResults,
    draw,
    input_values,
    run_paths,
    simulate_paths,
    single_point_run,
)
from uwcopilot.studio.feasibility.setup import SimulationSetup, build_setup
from uwcopilot.studio.feasibility.worlds import World, generate_world
from uwcopilot.studio.profiles.models import Profile

Condition = Literal["well_specified", "misspecified"]
CONDITIONS: tuple[Condition, ...] = ("well_specified", "misspecified")
# Misspecification generator parameters (SYNTHETIC; design choices of the project author, not data).
MISSPEC_T_DF = 3
MISSPEC_LEVEL_SD = 0.25
DOWNTURN_PROBABILITY = 0.3
DOWNTURN_MONTHS = 6
DOWNTURN_DEPTH = (0.2, 0.5)
VALIDATION_YEAR = 2
RELIABILITY_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Stream ids keep the world, truth and forecast random numbers independent of each other.
WORLD_STREAM, TRUTH_STREAM, FORECAST_STREAM = 0, 1, 2


@dataclass(frozen=True)
class WorldScore:
    """The forecast and the truth of one world under one condition."""

    world_id: int
    anchored: bool
    coverage_revenue: float
    coverage_net_income: float
    p_dscr: float
    outcome_dscr: bool
    single_point_dscr: bool
    p_cash_negative: float
    outcome_cash_negative: bool
    single_point_cash_negative: bool


def stream_seed(base_seed: int, profile_index: int, world_id: int, stream: int, condition_index: int = 0) -> int:
    """A reproducible, independent seed for one (profile, world, stream, condition)."""
    sequence = np.random.SeedSequence([base_seed, profile_index, world_id, stream, condition_index])
    return int(sequence.generate_state(1)[0])


def true_draws(setup: SimulationSetup, rng: np.random.Generator, condition: Condition) -> Draws:
    """The random numbers of the hidden truth (one path)."""
    truth = draw(setup, 1, rng)
    if condition == "well_specified":
        return truth
    scale = np.sqrt(MISSPEC_T_DF / (MISSPEC_T_DF - 2.0))
    noise = rng.standard_t(MISSPEC_T_DF, size=(1, setup.months)) / scale
    level = np.exp(rng.normal(-0.5 * MISSPEC_LEVEL_SD**2, MISSPEC_LEVEL_SD))
    extra = np.full((1, setup.months), level)
    downturn = rng.uniform() < DOWNTURN_PROBABILITY
    start = int(rng.integers(0, setup.months - DOWNTURN_MONTHS + 1))
    depth = float(rng.uniform(*DOWNTURN_DEPTH))
    if downturn:
        extra[0, start : start + DOWNTURN_MONTHS] *= 1.0 - depth
    return replace(truth, noise_z=noise, extra_demand_multiplier=extra)


def true_path(setup: SimulationSetup, seed: int, condition: Condition) -> PathResults:
    """Draw and run the hidden true future of a world."""
    draws = true_draws(setup, np.random.default_rng(seed), condition)
    return run_paths(setup, input_values(setup, draws), draws)


def band_coverage(forecast: np.ndarray, truth: np.ndarray) -> float:
    """Share of months where the true value lies within the forecast's P10-P90 (inclusive)."""
    p10, _, p90 = bands(forecast)
    inside = (truth[0] >= p10) & (truth[0] <= p90)
    return float(np.mean(inside))


def score_world(
    world: World, setup: SimulationSetup, forecast: PathResults, truth: PathResults, target: float
) -> WorldScore:
    """Compare one forecast with one true future."""
    baseline = single_point_run(setup)
    return WorldScore(
        world_id=world.world_id,
        anchored=world.anchored,
        coverage_revenue=band_coverage(forecast.revenue, truth.revenue),
        coverage_net_income=band_coverage(forecast.net_income, truth.net_income),
        p_dscr=float(np.mean(np.nan_to_num(dscr_by_year(forecast, VALIDATION_YEAR), nan=-np.inf) >= target)),
        outcome_dscr=bool(dscr_by_year(truth, VALIDATION_YEAR)[0] >= target),
        single_point_dscr=bool(dscr_by_year(baseline, VALIDATION_YEAR)[0] >= target),
        p_cash_negative=float(np.mean(cash_negative_within(forecast))),
        outcome_cash_negative=bool(cash_negative_within(truth)[0]),
        single_point_cash_negative=bool(cash_negative_within(baseline)[0]),
    )


def validate_profile(
    profile: Profile,
    profile_index: int,
    metro_inputs: dict[str, FactorValue],
    assumptions: FeasibilityAssumptions,
    n_worlds: int,
    n_paths: int,
    months: int,
    base_seed: int,
    target: float,
) -> dict[Condition, list[WorldScore]]:
    """Score n_worlds SYNTHETIC worlds of one business type under both conditions (same worlds, same forecasts)."""
    scores: dict[Condition, list[WorldScore]] = {condition: [] for condition in CONDITIONS}
    for world_id in range(n_worlds):
        world_rng = np.random.default_rng(stream_seed(base_seed, profile_index, world_id, WORLD_STREAM))
        world = generate_world(profile, metro_inputs, assumptions, world_id, world_rng, months)
        setup = build_setup(
            profile,
            "synthetic_metro",
            "SYNTHETIC",
            metro_inputs,
            world.site,
            world.plan,
            world.loan,
            assumptions,
            months,
        )
        _, _, forecast = simulate_paths(
            setup, n_paths, stream_seed(base_seed, profile_index, world_id, FORECAST_STREAM)
        )
        for condition_index, condition in enumerate(CONDITIONS):
            seed = stream_seed(base_seed, profile_index, world_id, TRUTH_STREAM, condition_index)
            truth = true_path(setup, seed, condition)
            scores[condition].append(score_world(world, setup, forecast, truth, target))
    return scores


def brier(forecast: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared difference between forecast probabilities and 0/1 outcomes (lower is better)."""
    return float(np.mean((np.asarray(forecast, dtype=float) - np.asarray(outcome, dtype=float)) ** 2))


def base_rate_brier(outcome: np.ndarray) -> float:
    """Brier score of a constant forecast equal to the outcome rate of the same worlds (in-sample)."""
    rate = float(np.mean(outcome))
    return brier(np.full(outcome.shape, rate), outcome)


def rounded(stats: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    """Round the float fields of a bootstrap result for the results file."""
    return {key: round(value, digits) if isinstance(value, float) else value for key, value in stats.items()}


def probability_block(
    forecast: np.ndarray, outcome: np.ndarray, single: np.ndarray, n_resamples: int, seed: int
) -> dict[str, Any]:
    """Brier of the simulator and both baselines, each with a bootstrap CI, plus the paired difference."""

    def brier_of(outcome_rows: np.ndarray, forecast_rows: np.ndarray) -> float:
        return brier(forecast_rows, outcome_rows)

    return {
        "outcome_rate": round(float(np.mean(outcome)), 4),
        "mean_forecast": round(float(np.mean(forecast)), 4),
        "brier_simulator": rounded(bootstrap_ci(brier_of, outcome, forecast, n_resamples=n_resamples, seed=seed)),
        "brier_single_forecast": rounded(bootstrap_ci(brier_of, outcome, single, n_resamples=n_resamples, seed=seed)),
        "brier_base_rate_in_sample": rounded(
            bootstrap_ci(base_rate_brier, outcome, n_resamples=n_resamples, seed=seed)
        ),
        "brier_simulator_minus_single_forecast": rounded(
            paired_bootstrap_diff(brier_of, outcome, forecast, single, n_resamples=n_resamples, seed=seed)
        ),
    }


def mean_ci(values: np.ndarray, n_resamples: int, seed: int) -> dict[str, Any]:
    """Mean over worlds with a bootstrap CI (worlds are resampled, so months of one world stay together)."""
    return rounded(bootstrap_ci(np.mean, values, n_resamples=n_resamples, seed=seed))


def column(scores: list[WorldScore], name: str) -> np.ndarray:
    """One field of every WorldScore as an array (booleans become 0/1 floats)."""
    return np.array([getattr(item, name) for item in scores], dtype=float)


def summarize(scores: list[WorldScore], n_resamples: int, seed: int) -> dict[str, Any]:
    """Coverage and Brier scores with bootstrap CIs for one group of worlds."""
    return {
        "n_worlds": len(scores),
        "n_worlds_not_anchored": int(len(scores) - np.sum(column(scores, "anchored"))),
        "band_coverage_revenue": mean_ci(column(scores, "coverage_revenue"), n_resamples, seed),
        "band_coverage_net_income": mean_ci(column(scores, "coverage_net_income"), n_resamples, seed),
        "dscr_year2": probability_block(
            column(scores, "p_dscr"),
            column(scores, "outcome_dscr"),
            column(scores, "single_point_dscr"),
            n_resamples,
            seed,
        ),
        "cash_negative_24m": probability_block(
            column(scores, "p_cash_negative"),
            column(scores, "outcome_cash_negative"),
            column(scores, "single_point_cash_negative"),
            n_resamples,
            seed,
        ),
    }


def reliability_table(scores: list[WorldScore]) -> list[dict[str, Any]]:
    """For bins of the forecast P(year-2 DSCR >= target): how many worlds, mean forecast, observed share."""
    forecast = column(scores, "p_dscr")
    outcome = column(scores, "outcome_dscr")
    rows = []
    edges = RELIABILITY_BINS
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        last = index == len(edges) - 2
        chosen = (forecast >= low) & ((forecast <= high) if last else (forecast < high))
        count = int(np.sum(chosen))
        rows.append(
            {
                "bin": f"[{low:.1f}, {high:.1f}{']' if last else ')'}",
                "n_worlds": count,
                "mean_forecast": round(float(np.mean(forecast[chosen])), 4) if count else None,
                "observed_share": round(float(np.mean(outcome[chosen])), 4) if count else None,
            }
        )
    return rows
