"""evaluate.py: common random numbers, agreement with S3's simulate(), other debt, and the outcome cache."""

import pytest
from path_helpers import FAST, METRO, car_wash_case, north_lot, prepared

from uwcopilot.studio.feasibility import load_assumptions, simulate
from uwcopilot.studio.path_to_approval import Change, EvaluationCache, FounderFacts, OtherDebt
from uwcopilot.studio.path_to_approval.evaluate import plan_hash
from uwcopilot.studio.path_to_approval.guardrails import apply_changes
from uwcopilot.studio.path_to_approval.targets import default_target
from uwcopilot.studio.profiles import load_profile


def s3_probabilities(case, n: int, seed: int) -> tuple[float, float, float]:
    result = simulate(
        case.profile_id, case.metro_id, case.site, case.plan, case.loan, n=n, seed=seed, with_tornado=False
    )
    stress = next(item for item in result.stress if item.name == "revenue_stress")
    return (
        result.p_dscr_at_least_target["year_2"],
        stress.p_dscr_year2_at_least_min,
        result.p_cash_negative_within_24_months,
    )


@pytest.mark.parametrize(
    "changes",
    [
        [],
        [Change(lever="loan_term_months", value=120.0)],
        [Change(lever="owner_equity", value=700_000.0), Change(lever="price", value=15.0)],
        [Change(lever="site", value="north lot")],
    ],
)
def test_outcomes_equal_s3_simulate_on_the_changed_plan(changes: list[Change]) -> None:
    """Common random numbers: a variant gives exactly what S3 gives for the changed plan with the same seed."""
    ready = prepared(car_wash_case(), [north_lot()])
    outcome = ready.evaluator.evaluate(changes)
    after = apply_changes(ready.space, changes) if changes else ready.plan
    expected = s3_probabilities(after, FAST.n_paths, FAST.seed)
    got = (
        outcome.p_dscr_at_least_target_year2,
        outcome.p_stressed_dscr_at_least_min_year2,
        outcome.p_cash_negative_within_months,
    )
    assert got == expected
    assert outcome.label == "simulated, based on stated assumptions"


def test_base_case_ratios_use_the_programs_arithmetic() -> None:
    ready = prepared(car_wash_case())
    outcome = ready.evaluator.evaluate([])
    result = simulate(
        "car_wash", METRO, ready.plan.site, ready.plan.plan, ready.plan.loan, n=FAST.n_paths, seed=FAST.seed
    )
    stress = next(item for item in result.stress if item.name == "revenue_stress")
    assert outcome.projected_dscr_base_case == result.projected_dscr_base_case
    assert outcome.stressed_dscr_base_case == stress.base_case_dscr
    assert outcome.annual_debt_service == result.loan.annual_debt_service


def test_other_debt_lowers_coverage_and_counts_in_debt_service() -> None:
    debt = OtherDebt(name="seller note", amount=40_000.0, monthly_payment=900.0)
    plain = prepared(car_wash_case()).evaluator.evaluate([])
    with_debt = prepared(car_wash_case(facts=FounderFacts(other_debt=[debt]))).evaluator.evaluate([])
    assert with_debt.annual_debt_service == round(plain.annual_debt_service + 12 * 900.0, 2)
    assert with_debt.p_dscr_at_least_target_year2 <= plain.p_dscr_at_least_target_year2
    assert with_debt.p_stressed_dscr_at_least_min_year2 <= plain.p_stressed_dscr_at_least_min_year2


def test_targets_missed_and_shortfall() -> None:
    outcome = prepared(car_wash_case()).evaluator.evaluate([])
    target = default_target()
    assert not outcome.meets_target
    assert "cash_negative" in outcome.targets_missed
    gap = outcome.p_cash_negative_within_months - target.p_cash_negative_max
    assert outcome.shortfall >= gap > 0


def test_cache_returns_the_same_outcome_and_counts_hits() -> None:
    cache = EvaluationCache()
    ready = prepared(car_wash_case(), cache=cache)
    first = ready.evaluator.evaluate([Change(lever="loan_term_months", value=96.0)])
    second = ready.evaluator.evaluate([Change(lever="loan_term_months", value=96.0)])
    assert first is second
    assert ready.evaluator.evaluations == 1 and ready.evaluator.cache_hits == 1
    again = prepared(car_wash_case(), cache=cache)
    assert again.evaluator.evaluate([Change(lever="loan_term_months", value=96.0)]) is first


def test_cache_guardrails_run_even_on_a_cache_hit() -> None:
    cache = EvaluationCache()
    ready = prepared(car_wash_case(), cache=cache)
    ready.evaluator.evaluate([Change(lever="price", value=15.0)])
    locked = prepared(car_wash_case(), cache=cache, locked=["price"])
    with pytest.raises(ValueError, match="locked"):
        locked.evaluator.evaluate([Change(lever="price", value=15.0)])


def test_cache_is_bounded_least_recently_used() -> None:
    cache = EvaluationCache(max_entries=2)
    ready = prepared(car_wash_case(), cache=cache)
    for term in (60.0, 72.0, 96.0):
        ready.evaluator.evaluate([Change(lever="loan_term_months", value=term)])
    assert len(cache) == 2
    cache.clear()
    assert len(cache) == 0


def test_plan_hash_changes_with_everything_that_changes_an_outcome() -> None:
    case = car_wash_case()
    profile = load_profile("car_wash", metro_id=METRO)
    assumptions = load_assumptions()
    target = default_target()
    inputs = {}
    base = plan_hash(case, [], target, inputs, profile, assumptions, 300, 36, 42)
    assert base == plan_hash(case, [], target, inputs, profile, assumptions, 300, 36, 42)
    edited = assumptions.model_copy(update={"price_elasticity": assumptions.capex_overrun_share})
    variants = [
        plan_hash(car_wash_case(mode=430.0), [], target, inputs, profile, assumptions, 300, 36, 42),
        plan_hash(case, [north_lot()], target, inputs, profile, assumptions, 300, 36, 42),
        plan_hash(case, [], target.model_copy(update={"dscr_min": 1.3}), inputs, profile, assumptions, 300, 36, 42),
        plan_hash(case, [], target, inputs, profile, edited, 300, 36, 42),
        plan_hash(case, [], target, inputs, profile.model_copy(update={"id": "other"}), assumptions, 300, 36, 42),
        plan_hash(case, [], target, inputs, profile, assumptions, 301, 36, 42),
        plan_hash(case, [], target, inputs, profile, assumptions, 300, 36, 43),
    ]
    assert base not in variants and len(set(variants)) == len(variants)
