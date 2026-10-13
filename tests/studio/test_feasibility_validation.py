"""SYNTHETIC worlds, hidden truths and the validation scores."""

import numpy as np
import pytest
from feasibility_helpers import setup_for, synthetic_inputs

from uwcopilot.studio.feasibility import load_assumptions
from uwcopilot.studio.feasibility.validation import (
    CONDITIONS,
    DOWNTURN_MONTHS,
    band_coverage,
    base_rate_brier,
    brier,
    reliability_table,
    summarize,
    true_draws,
    validate_profile,
)
from uwcopilot.studio.feasibility.worlds import anchor_demand, generate_world, single_point_dscr
from uwcopilot.studio.profiles import load_profile


def world_for(profile_id: str, seed: int):
    profile = load_profile(profile_id)
    inputs = synthetic_inputs(profile)
    return generate_world(profile, inputs, load_assumptions(), 0, np.random.default_rng(seed))


def test_world_is_reproducible_and_synthetic() -> None:
    first, second = world_for("laundromat", 5), world_for("laundromat", 5)
    assert first == second
    assert first.site.status == "SYNTHETIC" and "SYNTHETIC validation world" in first.site.source
    assert first.plan.owner_equity + first.loan.amount >= first.plan.capex


def test_anchored_world_hits_its_target_dscr() -> None:
    world = world_for("car_wash", 3)
    assert world.anchored
    setup = setup_for("car_wash", site=world.site, plan=world.plan, loan=world.loan)
    assert single_point_dscr(setup, world.site.mode) == pytest.approx(world.target_single_point_dscr, rel=1e-6)


def test_unreachable_target_is_marked_not_anchored() -> None:
    setup = setup_for()
    demand, reached = anchor_demand(setup, 1000.0, 600.0)
    assert (demand, reached) == (600.0, False)


def test_true_draws_differ_only_in_the_misspecified_condition() -> None:
    setup = setup_for()
    well = true_draws(setup, np.random.default_rng(1), "well_specified")
    wrong = true_draws(setup, np.random.default_rng(1), "misspecified")
    assert well.extra_demand_multiplier is None and well.n_paths == 1
    assert wrong.extra_demand_multiplier is not None and wrong.extra_demand_multiplier.shape == (1, 36)
    assert wrong.noise_z.shape == (1, 36)
    # The site-level error is one factor; a downturn, when drawn, lowers DOWNTURN_MONTHS months further.
    distinct = np.unique(np.round(wrong.extra_demand_multiplier, 12))
    assert len(distinct) in (1, 2)
    if len(distinct) == 2:
        assert np.sum(wrong.extra_demand_multiplier == distinct.min()) == DOWNTURN_MONTHS


def test_band_coverage_and_brier_hand_worked() -> None:
    forecast = np.tile(np.arange(1, 101, dtype=float).reshape(100, 1), (1, 4))
    truth = np.array([[50.0, 5.0, 95.0, 10.9]])  # P10 = 10.9, P90 = 90.1
    assert band_coverage(forecast, truth) == 0.5
    assert brier(np.array([0.8, 0.2]), np.array([1, 0])) == pytest.approx(0.04)
    assert base_rate_brier(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(0.1875)


def test_validate_profile_small_run_and_summary() -> None:
    profile = load_profile("retail_storefront")
    scores = validate_profile(profile, 4, synthetic_inputs(profile), load_assumptions(), 6, 200, 36, 42, 1.25)
    assert set(scores) == set(CONDITIONS)
    for items in scores.values():
        assert len(items) == 6
        assert all(0 <= item.p_dscr <= 1 and 0 <= item.coverage_revenue <= 1 for item in items)
    # Same worlds and the same forecasts under both conditions: only the truth changes.
    assert [item.p_dscr for item in scores["well_specified"]] == [item.p_dscr for item in scores["misspecified"]]
    summary = summarize(scores["well_specified"], 50, 42)
    assert summary["n_worlds"] == 6
    assert {"brier_simulator", "brier_single_forecast", "brier_base_rate_in_sample"} <= set(summary["dscr_year2"])
    assert summary["band_coverage_revenue"]["n_resamples"] == 50
    table = reliability_table(scores["well_specified"])
    assert sum(row["n_worlds"] for row in table) == 6


def test_well_specified_bands_cover_about_eighty_percent() -> None:
    # A property of the code: when the truth comes from the simulator's own model, about 80% of true months
    # fall inside P10-P90. Seeded, so the result is the same on every run.
    profile = load_profile("laundromat")
    scores = validate_profile(profile, 1, synthetic_inputs(profile), load_assumptions(), 40, 300, 36, 7, 1.25)
    coverage = np.mean([item.coverage_revenue for item in scores["well_specified"]])
    assert 0.7 <= coverage <= 0.9
