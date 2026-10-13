"""find_paths (Studio S5): single-lever then pair search, ranking, explanations, quick mode, infeasible plans."""

import pytest
from path_helpers import FAST, car_wash_case, north_lot, prepared, search

from uwcopilot.studio import DISCLAIMER, SIMULATION_LABEL
from uwcopilot.studio.data.fabric_snapshot import SnapshotFabric
from uwcopilot.studio.metros import load_metro
from uwcopilot.studio.path_to_approval import (
    QUICK_SETTINGS,
    EvaluationCache,
    LeverBounds,
    SearchSettings,
    custom_target,
    find_paths,
)
from uwcopilot.studio.path_to_approval.exhaustive import exhaustive_search
from uwcopilot.studio.path_to_approval.explain import NOT_AN_APPROVAL
from uwcopilot.studio.path_to_approval.search import clear_caches

VOLATILE = {"search": {"runtime_ms", "cache_hits", "evaluations"}}


def test_finds_the_smallest_change_sets_first() -> None:
    result = search(car_wash_case(), [north_lot()])
    assert result.status == "paths_found"
    assert not result.before.meets_target
    assert 1 <= len(result.change_sets) <= 3
    keys = [(item.n_levers, item.effort) for item in result.change_sets]
    assert keys == sorted(keys)
    assert [item.rank for item in result.change_sets] == list(range(1, len(result.change_sets) + 1))
    for item in result.change_sets:
        assert item.meets_target and item.after.meets_target
        assert item.before == result.before
        assert item.n_levers == len(item.changes) <= 2
        assert item.after.n_paths == FAST.n_paths


def test_singles_come_before_pairs_and_pairs_only_fill_up() -> None:
    result = search(car_wash_case(), [north_lot()])
    sizes = [item.n_levers for item in result.change_sets]
    assert sizes == sorted(sizes)
    if result.search.pairs_screened:
        assert sizes.count(1) < FAST.max_results


def test_explanation_is_plain_labeled_and_never_promises_approval() -> None:
    result = search(car_wash_case(), [north_lot()])
    for item in result.change_sets:
        assert SIMULATION_LABEL in item.explanation
        assert NOT_AN_APPROVAL in item.explanation
        assert "All three targets are met." in item.explanation
        assert "Unchanged: your credit score, your experience" in item.unchanged
        lowered = item.explanation.lower()
        assert "will be approved" not in lowered and "guarantee" not in lowered
    assert result.label == SIMULATION_LABEL
    assert result.disclaimer == DISCLAIMER
    assert NOT_AN_APPROVAL in result.message


def test_applied_plan_is_the_plan_with_the_change_applied() -> None:
    result = search(car_wash_case(), [north_lot()])
    top = result.change_sets[0]
    rerun = search(top.applied_plan, [north_lot()])
    assert rerun.before == top.after
    assert rerun.status == "already_meets_target"
    assert rerun.change_sets == [] and rerun.closest is None
    assert "already meets the three targets" in rerun.message


def test_same_seed_same_answer() -> None:
    first = search(car_wash_case(), [north_lot()])
    second = search(car_wash_case(), [north_lot()])
    assert first.model_dump(exclude=VOLATILE) == second.model_dump(exclude=VOLATILE)
    shared = EvaluationCache()
    warm_first = search(car_wash_case(), [north_lot()], cache=shared)
    warm_second = search(car_wash_case(), [north_lot()], cache=shared)
    assert warm_second.search.evaluations == 0 and warm_second.search.cache_hits > 0
    assert warm_first.change_sets == warm_second.change_sets


def test_another_seed_can_change_the_numbers_but_not_the_labels() -> None:
    other = search(car_wash_case(), [north_lot()], settings=FAST.model_copy(update={"seed": 7}))
    assert other.before.seed == 7
    assert other.label == SIMULATION_LABEL


def test_infeasible_target_gives_no_path_and_the_closest_result() -> None:
    impossible = custom_target(dscr_min=10.0, p_dscr_min=0.99)
    result = search(car_wash_case(), [north_lot()], target=impossible)
    assert result.status == "no_path_within_bounds"
    assert result.change_sets == []
    assert result.message.startswith("No path within your bounds")
    assert result.closest is not None and result.closest.rank == 0
    assert not result.closest.meets_target
    assert result.closest.after.shortfall < result.before.shortfall
    assert "Still missed: the year-2 DSCR target" in result.closest.explanation


def test_every_lever_locked_gives_no_path_and_no_closest() -> None:
    every = ["owner_equity", "loan_amount", "build_out_cost", "working_capital_reserve", "price", "loan_term_months"]
    result = search(car_wash_case(), locked=every)
    assert result.status == "no_path_within_bounds"
    assert result.levers_considered == []
    assert result.closest is None
    assert any("No change within your bounds" in note for note in result.notes)


def test_founder_bounds_are_respected() -> None:
    bounds = {
        "owner_equity": LeverBounds(high=620_000.0),
        "loan_amount": LeverBounds(low=1_090_000.0, high=1_120_000.0),
    }
    result = search(car_wash_case(), bounds=bounds)
    for item in result.change_sets + ([result.closest] if result.closest else []):
        assert item.plan_after.owner_equity <= 620_000.0
        assert 1_090_000.0 <= item.plan_after.loan_amount <= 1_120_000.0 or any(
            change.lever in ("owner_equity", "build_out_cost") for change in item.changes
        )


def test_a_change_set_never_breaks_a_program_rule_the_plan_meets() -> None:
    """A plan just above the NSP-3.1 equity minimum (21.0%): borrowing more than about $85,000 extra would push the
    owner's share below 20%, so those loan values are refused even under a cash-only target that wants them."""
    case = car_wash_case(
        plan_changes={"owner_equity": 357_000.0, "capex": 1_500_000.0}, loan_changes={"amount": 1_343_000.0}
    )
    cash_only = custom_target(p_dscr_min=0.0, p_stressed_min=0.0, p_cash_negative_max=0.0)
    others = ["owner_equity", "build_out_cost", "working_capital_reserve", "price", "loan_term_months"]
    result = search(case, target=cash_only, locked=others)
    assert result.plan_before.checks.meets_equity_minimum
    assert result.search.blocked_by_program > 0
    for item in result.change_sets + ([result.closest] if result.closest else []):
        assert item.plan_after.checks.meets_equity_minimum
        assert item.plan_after.checks.equity_injection_pct >= 20.0


def test_quick_mode_rechecks_with_the_full_number_of_paths() -> None:
    result = find_paths(car_wash_case(), [north_lot()], mode="quick", cache=EvaluationCache())
    assert result.settings == QUICK_SETTINGS
    assert result.before.n_paths == QUICK_SETTINGS.confirm_paths
    for item in result.change_sets:
        assert item.after.n_paths == QUICK_SETTINGS.confirm_paths
        assert item.after.meets_target
    assert any(note.startswith("Quick mode") for note in result.notes)


def test_budget_limits_the_search_and_says_so() -> None:
    tiny = SearchSettings(n_paths=200, steps_per_side=4, max_evaluations=3)
    result = search(car_wash_case(mode=300.0), [north_lot(300.0)], settings=tiny)
    assert result.search.budget_exhausted and not result.search.search_complete
    assert result.search.evaluations <= 4


@pytest.mark.parametrize("mode", [300.0, 420.0])
def test_heuristic_matches_brute_force_on_small_plans(mode: float) -> None:
    settings = SearchSettings(n_paths=200, steps_per_side=3, max_evaluations=5000)
    case, candidates = car_wash_case(mode=mode), [north_lot(mode)]
    cache = EvaluationCache()
    heuristic = search(case, candidates, settings=settings, cache=cache)
    reference = exhaustive_search(prepared(case, candidates, settings=settings, cache=cache))
    best = reference.best
    if best is None:
        assert heuristic.status == "no_path_within_bounds"
        return
    assert heuristic.status == "paths_found"
    top = heuristic.change_sets[0]
    assert top.n_levers == best.n_levers
    assert top.effort == pytest.approx(best.effort, abs=1e-6)


def test_real_metro_in_limited_mode_runs_and_says_so() -> None:
    """A real metro without a snapshot on disk (as in CI): limited mode, profile ranges, the founder's quoted rate."""
    fabric = SnapshotFabric(metro=load_metro("houston_tx"), snapshot=None)
    case = car_wash_case().model_copy(update={"metro_id": "houston_tx"})
    result = search(case, fabric=fabric)
    assert result.metro_mode == "limited"
    assert result.data_label == "PUBLIC"
    assert any("limited mode" in note for note in result.notes)
    assert result.status in ("paths_found", "no_path_within_bounds", "already_meets_target")


def test_a_fabric_for_another_metro_is_refused() -> None:
    fabric = SnapshotFabric(metro=load_metro("houston_tx"), snapshot=None)
    with pytest.raises(ValueError, match="data fabric is for houston_tx"):
        search(car_wash_case(), fabric=fabric)


def test_clear_caches_empties_the_shared_cache() -> None:
    find_paths(car_wash_case(), settings=FAST)
    clear_caches()
    from uwcopilot.studio.path_to_approval import SHARED_CACHE

    assert len(SHARED_CACHE) == 0
