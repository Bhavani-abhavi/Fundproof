"""Write results/studio_feasibility_metrics.json: the S3 feasibility simulator checked on SYNTHETIC worlds.

    uv run python scripts/eval_studio_feasibility.py          # or: make studio-feasibility

For every business-type profile, N SYNTHETIC worlds (a plan, a site demand range and a loan; worlds.py) are
simulated with 1,000 paths each. Each world also gets a hidden true future under two conditions (validation.py):
well_specified (the truth comes from the simulator's own model) and misspecified (the truth adds fat-tailed noise,
a site-level demand error and a local downturn that the simulator does not model). The script reports, per business
type and condition, with bootstrap 95% CIs (1,000 resamples over worlds, seed 42):
- coverage of the P10-P90 band of monthly revenue and net income by the true path (target about 80% when well
  specified);
- the Brier score of P(year-2 DSCR >= target) and of P(cash < 0 within 24 months), next to two baselines: a single
  forecast with every input at its most likely value, and the in-sample base rate;
- the time to simulate 1,000 paths (median of repeats; the machine is shared, so the load average is recorded).

This is validation on worlds whose generator the same author wrote: it shows the simulator is internally consistent
and how it degrades when its assumptions are wrong. It is not evidence of real-world accuracy. No real metro has
outcome data, so real-metro accuracy is "not measured". Every number in the file is computed here.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uwcopilot.common.paths import CONFIG_DIR, RESULTS_DIR
from uwcopilot.governance.provenance import RunProvenance, display_path
from uwcopilot.studio import DISCLAIMER, SIMULATION_LABEL, SYNTHETIC_LABEL
from uwcopilot.studio.data.cache import write_atomic
from uwcopilot.studio.data.fabric import DataFabric, get_fabric
from uwcopilot.studio.feasibility import validation
from uwcopilot.studio.feasibility.assumptions import ASSUMPTIONS_FILE, default_dscr_target, load_assumptions
from uwcopilot.studio.feasibility.model import simulate_paths
from uwcopilot.studio.feasibility.result import SIMULATOR_VERSION
from uwcopilot.studio.feasibility.setup import build_setup
from uwcopilot.studio.feasibility.simulate import run_simulation
from uwcopilot.studio.feasibility.worlds import METRO_ID, generate_world
from uwcopilot.studio.profiles.loader import PROFILES_DIR, load_all_profiles

OUTPUT: Path = RESULTS_DIR / "studio_feasibility_metrics.json"
GENERATED_BY = "scripts/eval_studio_feasibility.py"
DEFAULT_WORLDS = 400
DEFAULT_PATHS = 1000
MONTHS = 36
SEED = 42
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
TIMING_REPEATS = 15
TIMING_PATHS = 1000
SCOPE_NOTE = (
    "Validation on SYNTHETIC worlds whose generator the same author wrote (src/uwcopilot/studio/feasibility/"
    "worlds.py and validation.py). It checks internal consistency and the effect of misspecification; it is not "
    "evidence of real-world accuracy."
)
REAL_METROS = "not measured: no real outcome data exists for any metro, so no real-world accuracy can be computed"


def evaluate(n_worlds: int, n_paths: int, seed: int, fabric: DataFabric) -> dict[str, Any]:
    """Score every business type under both conditions; summarize per type, pooled, and as a reliability table."""
    assumptions = load_assumptions()
    target = default_dscr_target().value
    profiles = load_all_profiles()
    per_type: dict[str, dict[str, Any]] = {condition: {} for condition in validation.CONDITIONS}
    pooled: dict[str, list[validation.WorldScore]] = {condition: [] for condition in validation.CONDITIONS}
    for index, (profile_id, profile) in enumerate(profiles.items()):
        started = time.perf_counter()
        scores = validation.validate_profile(
            profile, index, fabric.metro_inputs(profile), assumptions, n_worlds, n_paths, MONTHS, seed, target
        )
        print(f"  {profile_id:<18} {n_worlds} worlds scored in {time.perf_counter() - started:.1f} s", flush=True)
        for condition, items in scores.items():
            per_type[condition][profile_id] = validation.summarize(items, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
            pooled[condition].extend(items)
    return {
        condition: {
            "per_business_type": per_type[condition],
            "pooled": validation.summarize(pooled[condition], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED),
            "reliability_dscr_year2_pooled": validation.reliability_table(pooled[condition]),
        }
        for condition in validation.CONDITIONS
    }


def timed(function: Any, repeats: int) -> dict[str, float | int]:
    """Median, min and max wall time in milliseconds over repeats (after one warm-up call)."""
    function()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "median_ms": round(statistics.median(durations), 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
        "repeats": repeats,
    }


def timing(fabric: DataFabric, repeats: int) -> dict[str, Any]:
    """Time 1,000 paths (path model only) and a full simulate() call (with tornado and stress) per business type."""
    assumptions = load_assumptions()
    rows: dict[str, Any] = {}
    for index, (profile_id, profile) in enumerate(load_all_profiles().items()):
        inputs = fabric.metro_inputs(profile)
        rng = np.random.default_rng(validation.stream_seed(SEED, index, 0, validation.WORLD_STREAM))
        world = generate_world(profile, inputs, assumptions, 0, rng, MONTHS)
        setup = build_setup(profile, METRO_ID, "SYNTHETIC", inputs, world.site, world.plan, world.loan, assumptions)

        def paths_only(setup: Any = setup) -> None:
            simulate_paths(setup, TIMING_PATHS, SEED)

        def full_call(profile: Any = profile, world: Any = world) -> None:
            run_simulation(profile, fabric, world.site, world.plan, world.loan, n=TIMING_PATHS, seed=SEED)

        rows[profile_id] = {
            "paths_only_1000": timed(paths_only, repeats),
            "simulate_with_tornado_and_stress_1000": timed(full_call, repeats),
            "tornado_inputs": sum(item.varies for item in setup.distributions.values()),
        }
    return {
        "note": "wall time on a shared 4-CPU machine used by several agents at once; see provenance.environment "
        "for the load average at the start and the end of the run",
        "per_business_type": rows,
    }


def headline(evaluated: dict[str, Any], timings: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per condition: pooled band coverage and Brier scores (estimate and CI)."""
    rows = []
    for condition, block in evaluated.items():
        pooled = block["pooled"]
        dscr = pooled["dscr_year2"]
        rows.append(
            {
                "condition": condition,
                "n_worlds": pooled["n_worlds"],
                "band_coverage_revenue": pooled["band_coverage_revenue"],
                "brier_dscr_year2_simulator": dscr["brier_simulator"],
                "brier_dscr_year2_single_forecast": dscr["brier_single_forecast"],
                "brier_dscr_year2_base_rate_in_sample": dscr["brier_base_rate_in_sample"],
            }
        )
    medians = [row["paths_only_1000"]["median_ms"] for row in timings["per_business_type"].values()]
    rows.append({"timing_paths_only_1000_median_ms_across_types": round(statistics.median(medians), 2)})
    return rows


def build_report(n_worlds: int, n_paths: int, seed: int, repeats: int) -> dict[str, Any]:
    """The full results document (without the provenance block)."""
    fabric = get_fabric(METRO_ID)
    target = default_dscr_target()
    evaluated = evaluate(n_worlds, n_paths, seed, fabric)
    timings = timing(fabric, repeats)
    return {
        "title": "Studio S3 feasibility simulator: validation on SYNTHETIC worlds",
        "data_label": SYNTHETIC_LABEL,
        "simulation_label": SIMULATION_LABEL,
        "disclaimer": DISCLAIMER,
        "scope": SCOPE_NOTE,
        "simulator_version": SIMULATOR_VERSION,
        "settings": {
            "metro": METRO_ID,
            "worlds_per_business_type": n_worlds,
            "paths_per_world": n_paths,
            "months": MONTHS,
            "seed": seed,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "world", "alpha": 0.05},
            "dscr_target": target.model_dump(),
            "dscr_year": validation.VALIDATION_YEAR,
        },
        "method": {
            "worlds": "worlds.py: each world draws a plan and a loan, then anchors the site's most likely demand so "
            "the one-forecast year-2 DSCR equals a drawn target; worlds that cannot reach it at capacity are kept and "
            "counted as n_worlds_not_anchored",
            "well_specified": "the true path is one more draw from the simulator's own model",
            "misspecified": (
                f"the true path adds Student-t({validation.MISSPEC_T_DF}) monthly noise scaled to unit variance, a "
                f"mean-one lognormal site demand error (log-sd {validation.MISSPEC_LEVEL_SD}), and with probability "
                f"{validation.DOWNTURN_PROBABILITY} a {validation.DOWNTURN_MONTHS}-month downturn cutting demand by "
                f"a uniform {validation.DOWNTURN_DEPTH[0]}-{validation.DOWNTURN_DEPTH[1]} share"
            ),
            "band_coverage": "share of the 36 months whose true value lies within the forecast P10-P90 "
            "(inclusive), averaged over worlds",
            "brier": "mean of (forecast probability - true 0/1 outcome)^2 over worlds; lower is better",
            "single_forecast_baseline": "every input at its most likely value, no shocks, no noise; read as a "
            "yes (1) or no (0) forecast",
            "base_rate_baseline": "a constant forecast equal to the share of the same worlds with the outcome "
            "(in-sample, so it is optimistic)",
        },
        "headline": headline(evaluated, timings),
        "conditions": evaluated,
        "timing": timings,
        "real_metros": REAL_METROS,
    }


def with_ci(stats: dict[str, Any]) -> str:
    """'estimate [ci_low, ci_high]' for printing."""
    return f"{stats['estimate']} [{stats['ci_low']}, {stats['ci_high']}]"


def headline_line(row: dict[str, Any]) -> str:
    """One printed line of the headline."""
    if "condition" not in row:
        return json.dumps(row)
    return (
        f"{row['condition']:<15} worlds={row['n_worlds']} "
        f"band_coverage_revenue={with_ci(row['band_coverage_revenue'])} "
        f"brier_dscr_y2 simulator={with_ci(row['brier_dscr_year2_simulator'])} "
        f"single_forecast={with_ci(row['brier_dscr_year2_single_forecast'])} "
        f"base_rate={with_ci(row['brier_dscr_year2_base_rate_in_sample'])}"
    )


def synthetic_metro_snapshot() -> dict[str, Any]:
    """Which generated city the metro inputs came from (seed and the generator record's checksum)."""
    fabric = get_fabric(METRO_ID)
    profile = next(iter(load_all_profiles().values()))
    record = fabric.metro_inputs(profile)["prime_rate"].source
    settings = fabric.metro.synthetic
    return {
        "metro": METRO_ID,
        "data_label": SYNTHETIC_LABEL,
        "seed": settings.seed if settings is not None else None,
        "generator_record_sha256": record.sha256 if record is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the validation and timing, add provenance, write the results file and print the headline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worlds", type=int, default=DEFAULT_WORLDS, help="SYNTHETIC worlds per business type")
    parser.add_argument("--paths", type=int, default=DEFAULT_PATHS, help="simulated paths per world")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timing-repeats", type=int, default=TIMING_REPEATS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run = RunProvenance.start(GENERATED_BY)
    report = build_report(args.worlds, args.paths, args.seed, args.timing_repeats)
    inputs = [
        ASSUMPTIONS_FILE,
        CONFIG_DIR / "studio" / "metros" / f"{METRO_ID}.yaml",
        *sorted(PROFILES_DIR.glob("*.yaml")),
    ]
    report["provenance"] = run.finish(
        corpus_checksum="not applicable (no retrieval corpus is used)",
        eval_sets=inputs,
        data_snapshots={"studio_synthetic_metro": synthetic_metro_snapshot()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    for row in report["headline"]:
        print(headline_line(row))
    print(f"Wrote {display_path(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
