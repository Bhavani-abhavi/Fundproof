"""find_paths: the smallest changes to a founder's plan that reach the simulated targets (Studio S5).

    from uwcopilot.studio.path_to_approval import PlanCase, find_paths
    result = find_paths(case, candidates, target)            # target None = the three default targets
    result.status            # already_meets_target | paths_found | no_path_within_bounds
    result.change_sets[0]    # changes, before/after probabilities, plain-language explanation

How the search works:
1. Simulate the plan as it is (S3 simulator, seeded). If it meets every target, stop.
2. Single-lever moves: for each available lever, try its grid values from the smallest change (least effort) to
   the largest, and keep the first that meets every target. That is the smallest change of that lever.
3. Pairs, only when fewer than max_results single moves were found: for two levers that do not work alone but
   each move the plan closer on its own, first try both at their best single values (a screen). If that meets the
   targets, walk the pair's grid from small to large changes (a staircase over the two sorted value lists) to find
   the least-effort combination. A site candidate is a category, so its partner lever is scanned per site.
   Any change-set that would break a startup-program check the plan meets now (equity injection, loan limits,
   term range; program.py) is refused like a value outside the bounds.
4. Rank change-sets by the number of levers changed, then by effort; return the first max_results.
5. Nothing found: "no path within your bounds" plus the closest result (the smallest total shortfall seen).
Every variant is simulated with the plan's own random draws (common random numbers, evaluate.py) and cached.

Quick mode (for the website): fewer simulated paths and a coarser grid for the search, then every change-set it
returns is re-checked with the full number of paths, so the probabilities shown are the full simulation's.

The pair search is a heuristic: it assumes that moving a lever further in its helpful direction does not hurt.
That holds exactly for owner equity, loan term, build-out cost and the cash reserve under common random numbers
(test_path_monotonicity.py), not for the loan amount or price; it is stated in the result's notes, and
exhaustive.py measures what it costs (scripts/eval_studio_paths.py).
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

from uwcopilot.studio.data.fabric import DataFabric, get_fabric
from uwcopilot.studio.feasibility.assumptions import FeasibilityAssumptions, load_assumptions
from uwcopilot.studio.feasibility.result import SIMULATOR_VERSION
from uwcopilot.studio.feasibility.simulate import resolve_profile
from uwcopilot.studio.path_to_approval import explain
from uwcopilot.studio.path_to_approval.errors import LeverError
from uwcopilot.studio.path_to_approval.evaluate import SHARED_CACHE, EvaluationCache, PathEvaluator, setup_for_case
from uwcopilot.studio.path_to_approval.guardrails import GUARDRAIL_RULES, apply_changes, field_values
from uwcopilot.studio.path_to_approval.levers import (
    LEVER_NAMES,
    LEVERS_FILE,
    LeverBounds,
    LeverConfig,
    effort,
    load_levers,
)
from uwcopilot.studio.path_to_approval.models import (
    AppliedChange,
    Candidate,
    Change,
    ChangeSet,
    LeverSummary,
    Mode,
    Outcome,
    PathResult,
    PlanCase,
    SearchSettings,
    SearchStats,
    Target,
)
from uwcopilot.studio.path_to_approval.program import failed_checks, newly_broken
from uwcopilot.studio.path_to_approval.space import LEVER_WRITES, LeverSpace, build_lever_space
from uwcopilot.studio.path_to_approval.targets import default_target
from uwcopilot.studio.profiles.models import Profile

FULL_SETTINGS = SearchSettings()
QUICK_SETTINGS = SearchSettings(
    mode="quick", n_paths=500, confirm_paths=1000, steps_per_side=4, max_results=3, max_evaluations=400
)
PAIR_HEURISTIC_NOTE = (
    "Pairs are searched only for levers that do not reach the targets alone but move the plan closer on their own, "
    "and the pair grid assumes that moving a lever further in its helpful direction does not hurt; a pair can be "
    "missed when that does not hold."
)


def settings_for(mode: Mode) -> SearchSettings:
    """The preset settings of a mode."""
    return QUICK_SETTINGS if mode == "quick" else FULL_SETTINGS


@lru_cache(maxsize=8)
def fabric_for(metro_id: str) -> DataFabric:
    """The metro's data fabric, loaded once per process (a snapshot is read from disk once)."""
    return get_fabric(metro_id)


@lru_cache(maxsize=8)
def mode_for(metro_id: str) -> tuple[str, str]:
    """The metro's mode and its reason, computed once per process (a real metro's mode reads every profile)."""
    return fabric_for(metro_id).mode()


@lru_cache(maxsize=64)
def profile_for(profile_id: str, metro_id: str) -> Profile:
    """The business-type profile with the metro's overrides, loaded once per process (a frozen model)."""
    return resolve_profile(profile_id, metro_id)


@lru_cache(maxsize=4)
def levers_from(path: Path) -> dict[str, LeverConfig]:
    """The levers file, read once per process and path."""
    return load_levers(path)


@lru_cache(maxsize=1)
def default_assumptions() -> FeasibilityAssumptions:
    """The simulator assumptions file, read once per process."""
    return load_assumptions()


def clear_caches() -> None:
    """Forget every per-process cache (fabrics, modes, profiles, outcomes): call after a config or snapshot change.

    A long-running web process keeps these between requests; the outcome cache is keyed by the plan, the profile,
    the metro inputs and the assumptions, so it cannot serve a stale outcome, but the fabric and profile caches can
    hold an old snapshot or YAML until cleared.
    """
    fabric_for.cache_clear()
    mode_for.cache_clear()
    profile_for.cache_clear()
    levers_from.cache_clear()
    default_assumptions.cache_clear()
    SHARED_CACHE.clear()


@dataclass(frozen=True)
class Tried:
    """One evaluated change-set."""

    changes: tuple[Change, ...]
    outcome: Outcome
    effort: float

    @property
    def n_levers(self) -> int:
        """Number of levers changed."""
        return len(self.changes)

    def rank_key(self) -> tuple[int, float, list[int]]:
        """Fewer levers first, then less effort, then the levers' allow-list order (for ties)."""
        return (self.n_levers, round(self.effort, 12), [LEVER_NAMES.index(change.lever) for change in self.changes])

    def closeness_key(self) -> tuple[float, int, float]:
        """Smaller total shortfall first, then fewer levers, then less effort."""
        return (self.outcome.shortfall, self.n_levers, round(self.effort, 12))


class Search:
    """One search over the levers of one plan (see the module docstring)."""

    def __init__(self, space: LeverSpace, evaluator: PathEvaluator, settings: SearchSettings) -> None:
        self.space = space
        self.evaluator = evaluator
        self.settings = settings
        self.base = evaluator.evaluate(())
        self.closest: Tried | None = None
        self.single_tries: dict[str, list[Tried]] = {}
        self.found: list[Tried] = []
        self.singles_tried = 0
        self.pairs_screened = 0
        self.pairs_searched = 0
        self.budget_exhausted = False
        self.base_checks = space.plan_amounts(space.case).checks
        self.blocked_by_program = 0

    def effort_of(self, changes: Sequence[Change]) -> float:
        """Total effort of a change-set."""
        project_cost = self.space.amounts.project_cost
        return sum(
            effort(self.space.configs[change.lever], self.space.current[change.lever], change.value, project_cost)
            for change in changes
        )

    def breaks_program(self, changes: tuple[Change, ...]) -> bool:
        """True when the change-set would break a startup-program check the plan meets now (program.py).

        Raises LeverError (through the guardrails) when the change-set is not allowed at all.
        """
        after = apply_changes(self.space, changes)
        if newly_broken(self.base_checks, self.space.plan_amounts(after).checks):
            self.blocked_by_program += 1
            return True
        return False

    def try_changes(self, changes: tuple[Change, ...]) -> Tried | None:
        """Evaluate a change-set; None when it is not allowed (bounds, program checks) or the budget is spent."""
        if self.evaluator.evaluations >= self.settings.max_evaluations:
            self.budget_exhausted = True
            return None
        try:
            if self.breaks_program(changes):
                return None
            outcome = self.evaluator.evaluate(changes)
        except LeverError:
            return None
        tried = Tried(changes, outcome, self.effort_of(changes))
        if self.closest is None or tried.closeness_key() < self.closest.closeness_key():
            self.closest = tried
        return tried

    def ordered_values(self, lever: str) -> list[float | str]:
        """A lever's grid from the smallest change to the largest."""
        values = self.space.grids[lever]
        return sorted(values, key=lambda value: (self.effort_of([Change(lever=lever, value=value)]), str(value)))

    def scan_single(self, lever: str) -> None:
        """Try one lever's values from small to large changes; keep the first that meets every target."""
        tries: list[Tried] = []
        for value in self.ordered_values(lever):
            tried = self.try_changes((Change(lever=lever, value=value),))
            if tried is None:
                if self.budget_exhausted:
                    break
                continue
            self.singles_tried += 1
            tries.append(tried)
            if tried.outcome.meets_target:
                self.found.append(tried)
                break
        self.single_tries[lever] = tries

    def best_single(self, lever: str) -> Tried | None:
        """The single move of a lever that came closest, when it moved the plan closer than it is now."""
        tries = self.single_tries.get(lever, [])
        if not tries:
            return None
        best = min(tries, key=Tried.closeness_key)
        return best if best.outcome.shortfall < self.base.shortfall else None

    def helpful_values(self, lever: str, best: Tried) -> list[float | str]:
        """The values to pair: from the smallest change up to the best single value, on the best value's side.

        For the site lever (a category), every candidate that moved the plan closer, closest first.
        """
        if lever == "site":
            helped = [tried for tried in self.single_tries[lever] if tried.outcome.shortfall < self.base.shortfall]
            return [tried.changes[0].value for tried in sorted(helped, key=Tried.closeness_key)]
        current = float(self.space.current[lever])
        best_value = float(best.changes[0].value)
        side = 1.0 if best_value > current else -1.0
        return [
            value
            for value in self.ordered_values(lever)
            if (float(value) - current) * side > 0 and abs(float(value) - current) <= abs(best_value - current)
        ]

    def staircase(self, first: str, firsts: list, second: str, seconds: list) -> Tried | None:
        """Least-effort feasible pair on two sorted value lists, assuming more of either lever never hurts.

        Start with the largest first value and the smallest second value; when a pair meets the targets, record it
        and try a smaller first value; when it does not, try a larger second value. At most len(firsts) +
        len(seconds) evaluations.
        """
        best: Tried | None = None
        i, j = len(firsts) - 1, 0
        while i >= 0 and j < len(seconds):
            tried = self.try_changes((Change(lever=first, value=firsts[i]), Change(lever=second, value=seconds[j])))
            if tried is not None and tried.outcome.meets_target:
                if best is None or tried.rank_key() < best.rank_key():
                    best = tried
                i -= 1
            else:
                if self.budget_exhausted:
                    break
                j += 1
        return best

    def per_site(self, site_values: list, other: str, others: list) -> Tried | None:
        """For each candidate site, the smallest change of the other lever that meets the targets."""
        best: Tried | None = None
        for site in site_values:
            for value in others:
                tried = self.try_changes((Change(lever="site", value=site), Change(lever=other, value=value)))
                if tried is not None and tried.outcome.meets_target:
                    if best is None or tried.rank_key() < best.rank_key():
                        best = tried
                    break
                if self.budget_exhausted:
                    return best
        return best

    def search_pair(self, first: str, second: str, best_first: Tried, best_second: Tried) -> None:
        """Screen one pair at its best single values, then find its least-effort combination."""
        self.pairs_screened += 1
        screen = self.try_changes(
            (
                Change(lever=first, value=best_first.changes[0].value),
                Change(lever=second, value=best_second.changes[0].value),
            )
        )
        if screen is None or not screen.outcome.meets_target:
            return
        self.pairs_searched += 1
        firsts = self.helpful_values(first, best_first)
        seconds = self.helpful_values(second, best_second)
        if first == "site":
            found = self.per_site(firsts, second, seconds)
        elif second == "site":
            found = self.per_site(seconds, first, firsts)
        else:
            found = self.staircase(first, firsts, second, seconds)
        candidates = [screen] if found is None else [screen, found]
        self.found.append(min(candidates, key=Tried.rank_key))

    def run(self) -> None:
        """Singles, then pairs when fewer than max_results singles were found (find_paths checks the plan first)."""
        for lever in self.space.available:
            self.scan_single(lever)
            if self.budget_exhausted:
                return
        if not self.settings.search_pairs or len(self.found) >= self.settings.max_results:
            return
        solved = {tried.changes[0].lever for tried in self.found}
        bests = {lever: self.best_single(lever) for lever in self.space.available if lever not in solved}
        helpful = [lever for lever, best in bests.items() if best is not None]
        pairs = sorted(
            combinations(helpful, 2),
            key=lambda pair: bests[pair[0]].outcome.shortfall + bests[pair[1]].outcome.shortfall,
        )
        for first, second in pairs:
            self.search_pair(first, second, bests[first], bests[second])
            if self.budget_exhausted:
                return

    def best_found(self) -> list[Tried]:
        """The returned change-sets: smallest first, at most max_results."""
        return sorted(self.found, key=Tried.rank_key)[: self.settings.max_results]


def applied_changes(space: LeverSpace, tried: Tried, after: PlanCase) -> list[AppliedChange]:
    """What each lever of a change-set did, for the result."""
    before_fields, after_fields = field_values(space.case), field_values(after)
    project_cost = space.amounts.project_cost
    items = []
    for change in tried.changes:
        config = space.configs[change.lever]
        changed = [path for path in LEVER_WRITES[change.lever] if before_fields[path] != after_fields[path]]
        items.append(
            AppliedChange(
                lever=change.lever,
                label=config.label,
                unit=space.demand_unit if change.lever == "site" else config.unit,
                from_value=space.current[change.lever],
                to_value=change.value,
                effort=round(effort(config, space.current[change.lever], change.value, project_cost), 6),
                fields_changed=changed,
            )
        )
    return items


def change_set(space: LeverSpace, tried: Tried, before: Outcome, target: Target, rank: int) -> ChangeSet:
    """A ChangeSet with its before/after plan and its explanation."""
    after_case = apply_changes(space, tried.changes)
    changes = applied_changes(space, tried, after_case)
    plan_after = space.plan_amounts(after_case)
    return ChangeSet(
        rank=rank,
        changes=changes,
        n_levers=len(changes),
        effort=round(tried.effort, 6),
        before=before,
        after=tried.outcome,
        plan_before=space.plan_amounts(space.case),
        plan_after=plan_after,
        applied_plan=after_case,
        meets_target=tried.outcome.meets_target,
        explanation=explain.explain_change_set(changes, space, after_case, before, tried.outcome, plan_after, target),
        unchanged=explain.unchanged_text(space.case),
    )


def lever_summaries(space: LeverSpace) -> list[LeverSummary]:
    """The levers the search moved, with their current value, bounds and grid size."""
    items = []
    for name in space.available:
        config = space.configs[name]
        low, high = space.bounds.get(name, (None, None))
        items.append(
            LeverSummary(
                name=name,
                label=config.label,
                unit=config.unit,
                current=space.current[name],
                low=None if low is None else round(low, 2),
                high=None if high is None else round(high, 2),
                grid_size=len(space.grids[name]),
                effort_weight=config.effort_weight,
                note=config.note,
            )
        )
    return items


def plan_notes(case: PlanCase, space: LeverSpace, metro_mode: tuple[str, str], setup_notes: Sequence[str]) -> list:
    """Context the founder should see next to the result."""
    mode, reason = metro_mode
    notes = [f"Metro {case.metro_id} runs in {mode} mode: {reason}."]
    failed = failed_checks(space.plan_amounts(case).checks)
    if failed:
        notes.append(
            "Your plan does not meet these rules of the SYNTHETIC startup program: "
            + "; ".join(failed)
            + ". The search does not try to repair them (set a lever's bounds to explore that); a change-set is "
            "never allowed to break a rule your plan meets now."
        )
    if case.facts.other_debt:
        notes.append(
            "Other debt is included as a fact: its monthly payments count in debt service, net income and cash, and "
            "its proceeds in the cash at opening (NSP-1.2). The S3 simulator on its own does not include them."
        )
    if "price" in space.available and case.plan.price is None:
        notes.append(
            "Your plan has no price yet, so it samples the profile's ASSUMPTION price range; a price change-set sets "
            "one price."
        )
    notes.extend(setup_notes)
    notes.append(PAIR_HEURISTIC_NOTE)
    return notes


def confirm(
    chosen: list[Tried], closest: Tried | None, confirmer: PathEvaluator
) -> tuple[list[Tried], Tried | None, int]:
    """Re-check the chosen change-sets (and, when none survives, the closest result) with the full number of paths.

    The coarse search can miss a target by Monte Carlo noise in either direction: a change-set it found may miss in
    the re-check (dropped), and its closest result may meet every target in the re-check (then it is returned).
    Returns (change-sets that meet every target in the re-check, the closest result re-checked or None, number
    dropped).
    """
    kept = []
    for tried in chosen:
        outcome = confirmer.evaluate(tried.changes)
        if outcome.meets_target:
            kept.append(Tried(tried.changes, outcome, tried.effort))
    dropped = len(chosen) - len(kept)
    rechecked = None
    if closest is not None and not kept:
        rechecked = Tried(closest.changes, confirmer.evaluate(closest.changes), closest.effort)
        if rechecked.outcome.meets_target:
            kept, rechecked = [rechecked], None
    return kept, rechecked, dropped


@dataclass(frozen=True)
class PreparedSearch:
    """Everything one search needs: the lever space, the evaluators and the context of the result.

    confirmer is the full-paths evaluator of quick mode (None in full mode). exhaustive.py reuses this, so the
    brute-force reference sees exactly the grids, bounds, guardrails and random draws of the real search.
    """

    plan: PlanCase
    space: LeverSpace
    evaluator: PathEvaluator
    confirmer: PathEvaluator | None
    settings: SearchSettings
    target: Target
    data_label: str
    metro_mode: tuple[str, str]


def prepare_search(
    plan: PlanCase,
    candidates: Sequence[Candidate] = (),
    target: Target | None = None,
    *,
    bounds: Mapping[str, LeverBounds] | None = None,
    locked: Iterable[str] = (),
    settings: SearchSettings = FULL_SETTINGS,
    cache: EvaluationCache | None = None,
    fabric: DataFabric | None = None,
    assumptions: FeasibilityAssumptions | None = None,
    levers_file: Path = LEVERS_FILE,
) -> PreparedSearch:
    """Load the metro, profile and levers; resolve the lever space; build the evaluators (see find_paths)."""
    chosen_target = target or default_target()
    chosen_cache = cache if cache is not None else SHARED_CACHE
    metro = fabric or fabric_for(plan.metro_id)
    if metro.metro.id != plan.metro_id:
        raise ValueError(f"the data fabric is for {metro.metro.id}, but the plan is in {plan.metro_id}")
    metro_mode = metro.mode() if fabric is not None else mode_for(plan.metro_id)
    profile = profile_for(plan.profile_id, plan.metro_id)
    metro_inputs = metro.metro_inputs(profile)
    chosen_assumptions = assumptions or default_assumptions()

    def evaluator_for(space: LeverSpace, n_paths: int) -> PathEvaluator:
        return PathEvaluator(
            space,
            profile,
            metro_inputs,
            metro.data_label,
            chosen_assumptions,
            chosen_target,
            n_paths,
            settings.months,
            settings.seed,
            chosen_cache,
            candidates,
        )

    default_capex = setup_for_case(
        plan, profile, metro_inputs, metro.data_label, chosen_assumptions, settings.months
    ).planned_capex
    space = build_lever_space(
        plan, profile, default_capex, levers_from(levers_file), candidates, bounds, locked, settings.steps_per_side
    )
    confirmer = None
    if settings.confirm_paths and settings.confirm_paths != settings.n_paths:
        confirmer = evaluator_for(space, settings.confirm_paths)
    return PreparedSearch(
        plan=plan,
        space=space,
        evaluator=evaluator_for(space, settings.n_paths),
        confirmer=confirmer,
        settings=settings,
        target=chosen_target,
        data_label=metro.data_label,
        metro_mode=metro_mode,
    )


def find_paths(
    plan: PlanCase,
    candidates: Sequence[Candidate] = (),
    target: Target | None = None,
    *,
    bounds: Mapping[str, LeverBounds] | None = None,
    locked: Iterable[str] = (),
    mode: Mode = "full",
    settings: SearchSettings | None = None,
    cache: EvaluationCache | None = None,
    fabric: DataFabric | None = None,
    assumptions: FeasibilityAssumptions | None = None,
    levers_file: Path = LEVERS_FILE,
) -> PathResult:
    """The smallest plan changes that reach the targets (see the module docstring).

    plan: the founder's plan and facts. candidates: other sites in the same metro. target: None = default_target().
    bounds: {lever: LeverBounds} set by the founder; locked: levers the founder does not want to move.
    mode: "full" or "quick" (settings overrides it). cache: shared outcome cache (default: one per process).
    fabric: the metro's data fabric (default: loaded once per process, like the metro mode and the profile).
    """
    started = time.perf_counter()
    prepared = prepare_search(
        plan,
        candidates,
        target,
        bounds=bounds,
        locked=locked,
        settings=settings or settings_for(mode),
        cache=cache,
        fabric=fabric,
        assumptions=assumptions,
        levers_file=levers_file,
    )
    space, evaluator, confirmer = prepared.space, prepared.evaluator, prepared.confirmer
    chosen_settings, chosen_target = prepared.settings, prepared.target
    search = Search(space, evaluator, chosen_settings)
    before = (confirmer or evaluator).evaluate(())
    if not before.meets_target:
        search.run()
    chosen, closest = search.best_found(), search.closest
    notes = plan_notes(plan, space, prepared.metro_mode, evaluator.base_setup.notes)
    if confirmer is not None:
        chosen, closest, dropped = confirm(chosen, closest, confirmer)
        notes.append(
            f"Quick mode: searched with {chosen_settings.n_paths} simulated paths and a coarser grid; every number "
            f"shown was re-checked with {chosen_settings.confirm_paths} paths ({dropped} change-set(s) dropped "
            "because they missed a target in the re-check)."
        )
    if before.meets_target:
        status, chosen, closest = "already_meets_target", [], None
    elif chosen:
        status, closest = "paths_found", None
    else:
        status = "no_path_within_bounds"
        if closest is not None and closest.outcome.shortfall >= before.shortfall:
            closest = None
        if closest is None:
            notes.append("No change within your bounds moved the plan closer to the targets.")
    evaluators = [evaluator] if confirmer is None else [evaluator, confirmer]
    return PathResult(
        simulator_version=SIMULATOR_VERSION,
        profile_id=plan.profile_id,
        metro_id=plan.metro_id,
        metro_mode=prepared.metro_mode[0],
        data_label=prepared.data_label,
        status=status,
        message=explain.status_message(status, len(chosen)),
        target=chosen_target,
        before=before,
        plan_before=space.plan_amounts(plan),
        change_sets=[
            change_set(space, tried, before, chosen_target, rank) for rank, tried in enumerate(chosen, start=1)
        ],
        closest=None if closest is None else change_set(space, closest, before, chosen_target, 0),
        levers_considered=lever_summaries(space),
        levers_not_available=space.not_available,
        guardrails=list(GUARDRAIL_RULES),
        settings=chosen_settings,
        search=SearchStats(
            evaluations=sum(item.evaluations for item in evaluators),
            cache_hits=sum(item.cache_hits for item in evaluators),
            singles_tried=search.singles_tried,
            pairs_screened=search.pairs_screened,
            pairs_searched=search.pairs_searched,
            blocked_by_program=search.blocked_by_program,
            budget_exhausted=search.budget_exhausted,
            search_complete=not search.budget_exhausted,
            runtime_ms=round((time.perf_counter() - started) * 1000.0, 2),
        ),
        notes=notes,
    )
