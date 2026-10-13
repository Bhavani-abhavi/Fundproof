"""Write results/studio_path_metrics.json: the path-to-approval search (Studio S5) on seeded SYNTHETIC plans.

    uv run python scripts/eval_studio_paths.py          # or: make studio-paths

For every metro (the SYNTHETIC metro and each real metro in its current mode, usually "limited") and every business
type, PLANS_PER_CELL SYNTHETIC plans are generated (path_to_approval/synthetic_plans.py; seeded). Each plan is searched:
1. full mode (1,000 paths), the three default targets: the headline (path-found rate, levers changed, runtime);
2. quick mode (500 paths, coarser grid, re-checked with 1,000 paths): what the website calls, plus a repeat of the same
   request on the warm outcome cache (a founder reloading the page);
3. full mode with the stress target switched off (only the two targets of the original Prompt S5): how often a path
   found without the NSP-4.2 stress target fails that stress, i.e. how often it would have misled the founder;
4. checks: every returned change-set is re-simulated with S3's own simulate() (same seed, 1,000 paths) and must give
   the same three probabilities; its applied plan is compared field by field with the founder's plan, independently
   of guardrails.py, to count any fact that moved (expected 0).
On a subsample of the SYNTHETIC metro's plans the heuristic search is compared with a brute-force reference that
tries every single value and every pair of values on the same grid (exhaustive.py), to count paths the heuristic missed.

Rates and means come with bootstrap 95% CIs (1,000 resamples over plans, seed 42). Runtimes are wall time in a warm
process (fabric, profile and mode loaded once, as in the web app) on a shared machine; the load average is in the
provenance block. Every probability is "simulated, based on stated assumptions"; no plan is real and no number here is
a claim about real-world approval (rule E, rule 9). Every number in the file is computed here.
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from uwcopilot.agent.policy_credit import REVENUE_STRESS_SHARE
from uwcopilot.common.paths import CONFIG_DIR, REPO_ROOT, RESULTS_DIR
from uwcopilot.governance.provenance import RunProvenance
from uwcopilot.risk.bootstrap import bootstrap_ci
from uwcopilot.studio import DISCLAIMER, SIMULATION_LABEL, SYNTHETIC_LABEL
from uwcopilot.studio.data.cache import write_atomic
from uwcopilot.studio.feasibility.assumptions import ASSUMPTIONS_FILE
from uwcopilot.studio.feasibility.result import SIMULATOR_VERSION
from uwcopilot.studio.feasibility.simulate import simulate
from uwcopilot.studio.metros import list_metros
from uwcopilot.studio.path_to_approval import synthetic_plans
from uwcopilot.studio.path_to_approval.evaluate import EvaluationCache
from uwcopilot.studio.path_to_approval.exhaustive import exhaustive_search
from uwcopilot.studio.path_to_approval.levers import LEVERS_FILE
from uwcopilot.studio.path_to_approval.models import PATH_ENGINE_VERSION, ChangeSet, PathResult, PlanCase
from uwcopilot.studio.path_to_approval.search import (
    FULL_SETTINGS,
    QUICK_SETTINGS,
    default_assumptions,
    fabric_for,
    find_paths,
    mode_for,
    prepare_search,
    profile_for,
)
from uwcopilot.studio.path_to_approval.targets import custom_target, default_target, describe
from uwcopilot.studio.profiles.loader import PROFILES_DIR, load_all_profiles

OUTPUT: Path = RESULTS_DIR / "studio_path_metrics.json"
GENERATED_BY = "scripts/eval_studio_paths.py"
SYNTHETIC_METRO = "synthetic_metro"
DEFAULT_PLANS_PER_CELL = 15
DEFAULT_EXHAUSTIVE_PER_TYPE = 15
SEED = 42
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
PROBABILITY_FIELDS: tuple[str, ...] = (
    "p_dscr_at_least_target_year2",
    "p_stressed_dscr_at_least_min_year2",
    "p_cash_negative_within_months",
)
PROGRAM_FILE: Path = REPO_ROOT / "corpus" / "programs" / "northwind_startup_program.md"
SCOPE_NOTE = (
    "Seeded SYNTHETIC plans (src/uwcopilot/studio/path_to_approval/synthetic_plans.py) searched on the SYNTHETIC metro "
    "and on each real metro in its current mode. The path-found rate depends on how the plans are generated and on "
    "the editable targets and lever bounds; it describes the search on these plans, not any real founder's chances."
)


def metro_ids(requested: Sequence[str] | None) -> list[str]:
    """The SYNTHETIC metro first, then the real metros in config order (or the requested ones)."""
    known = list_metros()
    if requested:
        unknown = sorted(set(requested) - set(known))
        if unknown:
            raise SystemExit(f"unknown metro(s) {unknown}; known: {known}")
        return list(requested)
    return [SYNTHETIC_METRO] + sorted(item for item in known if item != SYNTHETIC_METRO)


def top_levers(result: PathResult) -> list[str]:
    """The levers of the rank-1 change-set (empty when there is none)."""
    return [change.lever for change in result.change_sets[0].changes] if result.change_sets else []


def change_signature(change_set: ChangeSet) -> list[list[Any]]:
    """[[lever, value], ...] of a change-set, for comparing two results."""
    return [[change.lever, change.to_value] for change in change_set.changes]


def facts_moved(before: PlanCase, change_set: ChangeSet) -> list[str]:
    """Fields outside the plan choices that differ in the applied plan (independent of guardrails.py)."""
    after = change_set.applied_plan
    levers = {change.lever for change in change_set.changes}
    moved = []
    if after.facts != before.facts:
        moved.append("facts")
    if after.profile_id != before.profile_id or after.metro_id != before.metro_id:
        moved.append("profile or metro")
    if after.site != before.site and "site" not in levers:
        moved.append("site demand")
    for name in ("assumption_overrides", "hourly_wage", "capacity", "owner_draw_per_month", "opening_date"):
        if getattr(after.plan, name) != getattr(before.plan, name):
            moved.append(f"plan.{name}")
    for name in ("annual_rate_pct", "rate_spread_pct", "rate_type", "product"):
        if getattr(after.loan, name) != getattr(before.loan, name):
            moved.append(f"loan.{name}")
    return moved


def s3_gaps(change_set: ChangeSet) -> list[float]:
    """|reported - S3 simulate()| for the three probabilities of a change-set's applied plan (same seed and paths)."""
    plan = change_set.applied_plan
    result = simulate(
        plan.profile_id,
        fabric_for(plan.metro_id),
        plan.site,
        plan.plan,
        plan.loan,
        n=change_set.after.n_paths,
        seed=change_set.after.seed,
        with_tornado=False,
    )
    revenue_stress = next(item for item in result.stress if item.name == "revenue_stress")
    simulated = (
        result.p_dscr_at_least_target["year_2"],
        revenue_stress.p_dscr_year2_at_least_min,
        result.p_cash_negative_within_24_months,
    )
    reported = [getattr(change_set.after, name) for name in PROBABILITY_FIELDS]
    return [abs(a - b) for a, b in zip(reported, simulated, strict=True)]


def run_plan(plan: synthetic_plans.SyntheticPlan, metro_id: str, plan_index: int) -> dict[str, Any]:
    """Search one plan in every configuration and return one row of facts about it."""
    case, candidates = plan.case, list(plan.candidates)
    full = find_paths(case, candidates, mode="full", cache=EvaluationCache())
    quick_cache = EvaluationCache()
    quick = find_paths(case, candidates, mode="quick", cache=quick_cache)
    repeat = find_paths(case, candidates, mode="quick", cache=quick_cache)
    two_targets = find_paths(case, candidates, custom_target(p_stressed_min=0.0), mode="full", cache=EvaluationCache())
    returned = list(full.change_sets) + ([full.closest] if full.closest is not None else [])
    gaps = [s3_gaps(item) for item in returned]
    moved = [field for item in returned for field in facts_moved(case, item)]
    two_top = two_targets.change_sets[0] if two_targets.change_sets else None
    stress_min = full.target.p_stressed_min
    return {
        "metro_id": metro_id,
        "profile_id": case.profile_id,
        "plan_index": plan_index,
        "band": plan.band,
        "target_stressed_dscr_one_forecast": round(plan.target_stressed_dscr, 4),
        "anchored_by": plan.anchored_by,
        "staffing_plan": plan.staffing_plan,
        "before": {name: getattr(full.before, name) for name in PROBABILITY_FIELDS},
        "before_targets_missed": full.before.targets_missed,
        "status_full": full.status,
        "status_quick": quick.status,
        "status_two_targets": two_targets.status,
        "n_change_sets": len(full.change_sets),
        "levers_per_change_set": [item.n_levers for item in full.change_sets],
        "top_levers": top_levers(full),
        "all_levers": [[change.lever for change in item.changes] for item in full.change_sets],
        "top_signature_full": change_signature(full.change_sets[0]) if full.change_sets else None,
        "top_signature_quick": change_signature(quick.change_sets[0]) if quick.change_sets else None,
        "runtime_ms_full": full.search.runtime_ms,
        "runtime_ms_quick": quick.search.runtime_ms,
        "runtime_ms_quick_repeat": repeat.search.runtime_ms,
        "evaluations_full": full.search.evaluations,
        "evaluations_quick": quick.search.evaluations,
        "budget_exhausted_full": full.search.budget_exhausted,
        "budget_exhausted_quick": quick.search.budget_exhausted,
        "blocked_by_program_full": full.search.blocked_by_program,
        "two_targets_top_meets_stress": None
        if two_top is None
        else two_top.after.p_stressed_dscr_at_least_min_year2 >= stress_min,
        "s3_check_change_sets": len(gaps),
        "s3_check_max_gap": max((max(item) for item in gaps), default=None),
        "facts_moved": moved,
        "closest_shortfall": full.closest.after.shortfall if full.closest is not None else None,
    }


def exhaustive_row(plan: synthetic_plans.SyntheticPlan) -> dict[str, Any]:
    """Heuristic search vs the brute-force reference on the quick grid with 500 paths and no re-check."""
    settings = QUICK_SETTINGS.model_copy(update={"confirm_paths": None})
    case, candidates = plan.case, list(plan.candidates)
    cache = EvaluationCache()
    heuristic = find_paths(case, candidates, settings=settings, cache=cache)
    started = time.perf_counter()
    prepared = prepare_search(case, candidates, settings=settings, cache=cache)
    reference = exhaustive_search(prepared)
    best = reference.best
    top = heuristic.change_sets[0] if heuristic.change_sets else None
    return {
        "profile_id": case.profile_id,
        "band": plan.band,
        "status": heuristic.status,
        "heuristic_found": top is not None,
        "reference_found": best is not None,
        "already_meets_target": heuristic.status == "already_meets_target",
        "heuristic_levers": None if top is None else top.n_levers,
        "reference_levers": None if best is None else best.n_levers,
        "heuristic_effort": None if top is None else top.effort,
        "reference_effort": None if best is None else round(best.effort, 6),
        "reference_feasible_change_sets": len(reference.feasible),
        "reference_tried": reference.tried,
        "heuristic_evaluations": heuristic.search.evaluations,
        "heuristic_budget_exhausted": heuristic.search.budget_exhausted,
        "reference_seconds": round(time.perf_counter() - started, 2),
    }


def ci(metric: Callable[[np.ndarray], float], values: Sequence[float]) -> dict[str, Any] | str:
    """A bootstrap CI of metric over the values (rounded), or "not measured" when there are none."""
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return "not measured (no plans in this group)"
    result = bootstrap_ci(metric, data, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()} | {
        "n": int(data.size)
    }


def p50(values: np.ndarray) -> float:
    """Median."""
    return float(np.percentile(values, 50))


def p95(values: np.ndarray) -> float:
    """95th percentile (NumPy's default linear interpolation)."""
    return float(np.percentile(values, 95))


def path_found_block(rows: list[dict[str, Any]], status_key: str = "status_full") -> dict[str, Any]:
    """Status counts and the path-found rate among plans that do not meet the targets as they are."""
    statuses = Counter(row[status_key] for row in rows)
    needing = [row for row in rows if row[status_key] != "already_meets_target"]
    return {
        "n_plans": len(rows),
        "status_counts": {
            name: statuses.get(name, 0) for name in ("already_meets_target", "paths_found", "no_path_within_bounds")
        },
        "path_found_rate_among_plans_needing_a_change": ci(
            np.mean, [row[status_key] == "paths_found" for row in needing]
        ),
        "already_meets_target_rate": ci(np.mean, [row[status_key] == "already_meets_target" for row in rows]),
    }


def levers_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How many levers the returned change-sets change, and which levers they use."""
    found = [row for row in rows if row["status_full"] == "paths_found"]
    all_sets = [count for row in found for count in row["levers_per_change_set"]]
    usage_top = Counter(lever for row in found for lever in row["top_levers"])
    usage_all = Counter(lever for row in found for levers in row["all_levers"] for lever in levers)
    return {
        "n_plans_with_a_path": len(found),
        "average_levers_changed_top_change_set": ci(np.mean, [len(row["top_levers"]) for row in found]),
        "average_levers_changed_all_returned_change_sets": ci(np.mean, all_sets),
        "share_single_lever_top_change_set": ci(np.mean, [len(row["top_levers"]) == 1 for row in found]),
        "change_sets_returned_per_plan": dict(sorted(Counter(row["n_change_sets"] for row in found).items())),
        "lever_use_in_top_change_sets": dict(usage_top.most_common()),
        "lever_use_in_all_returned_change_sets": dict(usage_all.most_common()),
    }


def runtime_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Runtime p50 and p95 (ms) for full, quick and the repeated quick request, with bootstrap CIs."""
    block: dict[str, Any] = {}
    for key, label in (
        ("runtime_ms_full", "full"),
        ("runtime_ms_quick", "quick"),
        ("runtime_ms_quick_repeat", "quick_repeat_warm_cache"),
    ):
        values = [row[key] for row in rows]
        block[label] = {"p50_ms": ci(p50, values), "p95_ms": ci(p95, values), "max_ms": round(max(values), 2)}
    block["simulations_per_search"] = {
        "full_median": statistics.median(row["evaluations_full"] for row in rows),
        "quick_median": statistics.median(row["evaluations_quick"] for row in rows),
        "full_budget_exhausted": sum(row["budget_exhausted_full"] for row in rows),
        "quick_budget_exhausted": sum(row["budget_exhausted_quick"] for row in rows),
    }
    return block


def quick_vs_full(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often quick mode (what the website calls) gives the same answer as full mode."""
    both = [row for row in rows if row["top_signature_full"] is not None and row["top_signature_quick"] is not None]
    full_found = [row for row in rows if row["status_full"] == "paths_found"]
    quick_found = [row for row in rows if row["status_quick"] == "paths_found"]
    return {
        "status_agreement": ci(np.mean, [row["status_full"] == row["status_quick"] for row in rows]),
        "quick_finds_a_path_when_full_does": ci(np.mean, [row["status_quick"] == "paths_found" for row in full_found]),
        "full_finds_a_path_when_quick_does": ci(np.mean, [row["status_full"] == "paths_found" for row in quick_found]),
        "same_levers_in_top_change_set_when_both_find_one": ci(
            np.mean,
            [
                [item[0] for item in row["top_signature_full"]] == [item[0] for item in row["top_signature_quick"]]
                for row in both
            ],
        ),
        "identical_top_change_set_when_both_find_one": ci(
            np.mean, [row["top_signature_full"] == row["top_signature_quick"] for row in both]
        ),
    }


def stress_target_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """With only the two plain targets: how often a path exists, and how often it fails the lender's stress."""
    two_found = [row for row in rows if row["status_two_targets"] == "paths_found"]
    needing = [row for row in rows if row["status_two_targets"] != "already_meets_target"]
    return {
        "question": "If the search used only P(DSCR >= 1.25) and P(cash < 0) (the original Prompt S5 targets), how "
        "often would the change-set it shows fail the NSP-4.2 lender stress target?",
        "path_found_rate_two_targets": ci(np.mean, [row["status_two_targets"] == "paths_found" for row in needing]),
        "two_target_top_change_set_fails_stress_target": ci(
            np.mean, [not row["two_targets_top_meets_stress"] for row in two_found]
        ),
        "plans_already_meeting_two_targets_but_not_the_stress": sum(
            row["status_two_targets"] == "already_meets_target" and row["status_full"] != "already_meets_target"
            for row in rows
        ),
    }


def checks_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The independent checks: S3 re-simulation and facts that moved (both expected to be zero)."""
    gaps = [row["s3_check_max_gap"] for row in rows if row["s3_check_max_gap"] is not None]
    return {
        "s3_resimulation": {
            "change_sets_checked": sum(row["s3_check_change_sets"] for row in rows),
            "max_abs_gap_any_probability": max(gaps) if gaps else "not measured (no change-set returned)",
            "method": "each returned change-set's applied plan is run through feasibility.simulate() with the same "
            "seed and number of paths; the three probabilities must equal the search's (common random numbers)",
        },
        "facts_moved": {
            "count": sum(len(row["facts_moved"]) for row in rows),
            "fields": dict(Counter(field for row in rows for field in row["facts_moved"])),
            "method": "facts, profile, metro, the site's demand (unless the site lever moved), assumption overrides, "
            "wage, capacity, owner draw, opening date and the loan's rate fields of every applied plan are compared "
            "with the founder's plan, without using guardrails.py",
        },
        "blocked_by_program_checks_full_mode": sum(row["blocked_by_program_full"] for row in rows),
    }


def exhaustive_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Heuristic vs brute force on the subsample."""
    needing = [row for row in rows if not row["already_meets_target"]]
    both = [row for row in needing if row["heuristic_found"] and row["reference_found"]]
    return {
        "method": "quick-mode grid, 500 paths, no re-check; the brute force tries every single value and every pair "
        "of values of the available levers with the same guardrails, program checks and random draws "
        "(path_to_approval/exhaustive.py); SYNTHETIC metro only",
        "n_plans": len(rows),
        "n_plans_needing_a_change": len(needing),
        "reference_found_a_path": sum(row["reference_found"] for row in needing),
        "heuristic_found_a_path": sum(row["heuristic_found"] for row in needing),
        "heuristic_missed_a_path": sum(row["reference_found"] and not row["heuristic_found"] for row in needing),
        "heuristic_found_a_path_the_reference_did_not": sum(
            row["heuristic_found"] and not row["reference_found"] for row in needing
        ),
        "same_number_of_levers_when_both_found": sum(
            row["heuristic_levers"] == row["reference_levers"] for row in both
        ),
        "heuristic_effort_over_reference_effort_when_both_found": [
            round(row["heuristic_effort"] / row["reference_effort"], 4) if row["reference_effort"] else None
            for row in both
        ],
        "reference_seconds_median": statistics.median(row["reference_seconds"] for row in rows) if rows else None,
        "rows": rows,
    }


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """path_found_block per value of one row field."""
    values = sorted({row[key] for row in rows})
    return {value: path_found_block([row for row in rows if row[key] == value]) for value in values}


def metro_context(metro_id: str) -> dict[str, Any]:
    """The metro's mode, data label, snapshot and the status of the simulator's metro inputs (per business type)."""
    fabric = fabric_for(metro_id)
    mode, reason = mode_for(metro_id)
    snapshot = getattr(fabric, "snapshot", None)
    inputs = {}
    for profile_id in load_all_profiles():
        values = fabric.metro_inputs(profile_for(profile_id, metro_id))
        inputs[profile_id] = {name: value.status for name, value in values.items()}
    return {
        "mode": mode,
        "mode_reason": reason,
        "data_label": fabric.data_label,
        "snapshot_date": snapshot.snapshot_date.isoformat() if snapshot is not None else None,
        "snapshot_fingerprint": snapshot.fingerprint() if snapshot is not None else None,
        "metro_input_status_by_business_type": inputs,
    }


def margin_check(profile_ids: list[str]) -> dict[str, Any]:
    """Per business type: the profile's cost ratios added up, and whether the NSP-4.2 stress can pass at the modes.

    The revenue stress removes REVENUE_STRESS_SHARE of revenue from NOI dollar for dollar. If the cost ratios alone
    (before labor and rent) leave less than that share per revenue dollar, stressed NOI is negative at ANY demand when
    every input is at its most likely value. Base profiles (no metro overrides); config values, added up here.
    """
    profiles = load_all_profiles()
    block: dict[str, Any] = {
        "stress_share": REVENUE_STRESS_SHARE,
        "method": "sum of the profile's cost ratios (low, mode, high); margin before labor and rent = 1 - sum at the "
        "modes; the stress can pass at the modes only when that margin exceeds the stress share (necessary, not "
        "sufficient: labor, rent and debt service still have to be covered)",
    }
    for profile_id in profile_ids:
        ratios = profiles[profile_id].cost_ratios.values()
        low = sum(item.low for item in ratios)
        mode = sum(item.point() for item in ratios)
        high = sum(item.high for item in ratios)
        block[profile_id] = {
            "cost_ratio_sum_low": round(low, 4),
            "cost_ratio_sum_mode": round(mode, 4),
            "cost_ratio_sum_high": round(high, 4),
            "margin_before_labor_and_rent_at_modes": round(1.0 - mode, 4),
            "stress_can_pass_at_modes_before_labor_and_rent": 1.0 - mode > REVENUE_STRESS_SHARE,
        }
    return block


def anchoring_by_type(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """How each business type's plans were anchored (demand, rent_quote, none)."""
    types = sorted({row["profile_id"] for row in rows})
    return {
        profile_id: {
            lever: sum(row["profile_id"] == profile_id and row["anchored_by"] == lever for row in rows)
            for lever in ("demand", "rent_quote", "none")
        }
        for profile_id in types
    }


def evaluate(
    metros: list[str], profile_ids: list[str], plans_per_cell: int, exhaustive_per_type: int
) -> dict[str, Any]:
    """Generate and search every plan; summarize."""
    assumptions = default_assumptions()
    all_ids = list(load_all_profiles())
    rows: list[dict[str, Any]] = []
    exhaustive_rows: list[dict[str, Any]] = []
    contexts = {}
    for metro_index, metro_id in enumerate(metros):
        fabric = fabric_for(metro_id)
        contexts[metro_id] = metro_context(metro_id)
        started = time.perf_counter()
        for profile_id in profile_ids:
            profile_index = all_ids.index(profile_id)
            profile = profile_for(profile_id, metro_id)
            inputs = fabric.metro_inputs(profile)
            for plan_index in range(plans_per_cell):
                seed = synthetic_plans.plan_seed(SEED, metro_index, profile_index, plan_index)
                plan = synthetic_plans.generate_plan(profile, metro_id, fabric.data_label, inputs, assumptions, seed)
                if not rows:
                    find_paths(plan.case, list(plan.candidates), cache=EvaluationCache())  # warm-up, not timed
                rows.append(run_plan(plan, metro_id, plan_index))
                if metro_id == SYNTHETIC_METRO and plan_index < exhaustive_per_type:
                    exhaustive_rows.append(exhaustive_row(plan))
        print(f"  {metro_id:<22} {len(profile_ids) * plans_per_cell} plans in {time.perf_counter() - started:.1f} s")
    return {"rows": rows, "exhaustive_rows": exhaustive_rows, "contexts": contexts}


def headline(rows: list[dict[str, Any]], synthetic_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The numbers quoted in the walkthrough."""
    overall = path_found_block(rows)
    levers = levers_block(rows)
    runtime = runtime_block(rows)
    return {
        "n_plans": overall["n_plans"],
        "status_counts_full": overall["status_counts"],
        "path_found_rate_full": overall["path_found_rate_among_plans_needing_a_change"],
        "path_found_rate_full_synthetic_metro": path_found_block(synthetic_rows)[
            "path_found_rate_among_plans_needing_a_change"
        ],
        "average_levers_changed_top_change_set": levers["average_levers_changed_top_change_set"],
        "runtime_full_p50_ms": runtime["full"]["p50_ms"],
        "runtime_full_p95_ms": runtime["full"]["p95_ms"],
        "runtime_quick_p50_ms": runtime["quick"]["p50_ms"],
        "runtime_quick_p95_ms": runtime["quick"]["p95_ms"],
    }


def build_report(
    metros: list[str], profile_ids: list[str], plans_per_cell: int, exhaustive_per_type: int
) -> dict[str, Any]:
    """The whole results document (without the provenance block)."""
    evaluated = evaluate(metros, profile_ids, plans_per_cell, exhaustive_per_type)
    rows = evaluated["rows"]
    synthetic_rows = [row for row in rows if row["metro_id"] == SYNTHETIC_METRO]
    target = default_target()
    return {
        "title": "Studio S5 path to approval: the search on seeded SYNTHETIC plans",
        "data_label": SYNTHETIC_LABEL,
        "simulation_label": SIMULATION_LABEL,
        "disclaimer": DISCLAIMER,
        "scope": SCOPE_NOTE,
        "engine_version": PATH_ENGINE_VERSION,
        "simulator_version": SIMULATOR_VERSION,
        "settings": {
            "metros": metros,
            "business_types": profile_ids,
            "plans_per_metro_and_business_type": plans_per_cell,
            "seed": SEED,
            "targets": describe(target),
            "target_sources": target.sources,
            "full_mode": FULL_SETTINGS.model_dump(),
            "quick_mode": QUICK_SETTINGS.model_dump(),
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED, "unit": "plan", "alpha": 0.05},
            "plan_generator": {
                "stressed_dscr_target_one_forecast": list(synthetic_plans.STRESSED_DSCR_TARGET),
                "bands": [list(band) for band in synthetic_plans.TARGET_BANDS],
                "staffing_plan_share": synthetic_plans.STAFFING_PLAN_SHARE,
                "candidate_sites_per_plan": synthetic_plans.N_CANDIDATES,
                "candidate_demand_factor": list(synthetic_plans.CANDIDATE_DEMAND_FACTOR),
                "candidate_rent_quote_share": synthetic_plans.CANDIDATE_RENT_QUOTE_SHARE,
                "status": "design choices of the project author (generator parameters, not data)",
            },
        },
        "definitions": {
            "path_found_rate": "share of plans that do not meet the three targets as they are for which the search "
            "returns at least one change-set of one or two levers inside the default bounds that meets all three",
            "levers_changed": "number of levers in a returned change-set (1 or 2)",
            "runtime": "wall time of one find_paths() call in a warm process, in ms (search.runtime_ms)",
            "band": "band of the plan's drawn one-forecast stressed year-2 DSCR (synthetic_plans.TARGET_BANDS)",
            "anchored_by": "how the plan was anchored to its drawn target: demand, rent_quote or none (not anchored)",
        },
        "headline": headline(rows, synthetic_rows),
        "overall_full": path_found_block(rows),
        "overall_quick": path_found_block(rows, "status_quick"),
        "by_metro": grouped(rows, "metro_id"),
        "by_business_type": grouped(rows, "profile_id"),
        "by_band": grouped(rows, "band"),
        "by_anchoring": grouped(rows, "anchored_by"),
        "anchoring_by_business_type": anchoring_by_type(rows),
        "structural_margin_check": margin_check(profile_ids),
        "targets_missed_before": dict(Counter("+".join(row["before_targets_missed"]) or "none" for row in rows)),
        "levers": levers_block(rows),
        "runtime": runtime_block(rows),
        "runtime_by_metro": {
            metro: {
                "full_p50_ms": round(p50(np.array([r["runtime_ms_full"] for r in rows if r["metro_id"] == metro])), 2),
                "quick_p50_ms": round(
                    p50(np.array([r["runtime_ms_quick"] for r in rows if r["metro_id"] == metro])), 2
                ),
            }
            for metro in metros
        },
        "quick_vs_full": quick_vs_full(rows),
        "stress_target_effect": stress_target_block(rows),
        "checks": checks_block(rows),
        "heuristic_vs_brute_force": exhaustive_block(evaluated["exhaustive_rows"]),
        "metros": evaluated["contexts"],
        "real_world_accuracy": "not measured: no real founder outcomes exist; every plan is SYNTHETIC",
        "rows": rows,
    }


def config_inputs() -> list[Path]:
    """The config files that shape the numbers (hashed into the provenance block)."""
    files = [LEVERS_FILE, ASSUMPTIONS_FILE, PROGRAM_FILE]
    files += sorted(PROFILES_DIR.glob("*.yaml"))
    files += sorted((CONFIG_DIR / "studio" / "metros").glob("*.yaml"))
    return files


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation, add provenance, write the JSON and print the headline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plans", type=int, default=DEFAULT_PLANS_PER_CELL, help="plans per metro x business type")
    parser.add_argument(
        "--exhaustive", type=int, default=DEFAULT_EXHAUSTIVE_PER_TYPE, help="brute-force plans per business type"
    )
    parser.add_argument("--metros", nargs="*", default=None, help="metro ids (default: all)")
    parser.add_argument("--business-types", nargs="*", default=None, help="profile ids (default: all)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    run = RunProvenance.start(GENERATED_BY)
    metros = metro_ids(args.metros)
    known_types = list(load_all_profiles())
    profile_ids = args.business_types or known_types
    unknown = sorted(set(profile_ids) - set(known_types))
    if unknown:
        raise SystemExit(f"unknown business type(s) {unknown}; known: {known_types}")
    report = build_report(metros, profile_ids, args.plans, args.exhaustive)
    snapshots = {metro: block["snapshot_fingerprint"] for metro, block in report["metros"].items()}
    report["provenance"] = run.finish(
        corpus_checksum="not applicable (no retrieval corpus is used)",
        eval_sets=config_inputs(),
        data_snapshots={"studio_snapshots": snapshots, "plan_seed": SEED},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output, (json.dumps(report, indent=2) + "\n").encode("utf-8"))
    print(json.dumps(report["headline"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
