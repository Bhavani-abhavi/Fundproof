"""The lender stress preset reuses the lender's own levers and formulas."""

from feasibility_helpers import basic_loan, setup_for

from uwcopilot.agent import policy_credit
from uwcopilot.agent.credit_calculations import coverage_ratio, proposed_annual_debt_service
from uwcopilot.studio.feasibility.result import YearBand
from uwcopilot.studio.feasibility.stress import (
    LENDER_STRESS,
    base_case_stressed_dscr,
    lender_annual_debt_service,
    projected_dscr,
)

YEAR = YearBand(
    year=2,
    revenue_p10=800_000.0,
    revenue_p50=1_000_000.0,
    revenue_p90=1_200_000.0,
    noi_p10=150_000.0,
    noi_p50=300_000.0,
    noi_p90=450_000.0,
    net_income_p10=0.0,
    net_income_p50=100_000.0,
    net_income_p90=250_000.0,
    debt_service_p50=200_000.0,
    dscr_p10=0.8,
    dscr_p50=1.5,
    dscr_p90=2.2,
)


def test_levers_and_thresholds_are_the_policy_constants() -> None:
    revenue, rate = LENDER_STRESS
    assert revenue.levers.revenue_drop_share == policy_credit.REVENUE_STRESS_SHARE
    assert revenue.min_dscr == policy_credit.REVENUE_STRESS_MIN_DSCR and not revenue.variable_rate_only
    assert rate.levers.rate_add_pct == policy_credit.RATE_STRESS_ADD_PCT
    assert rate.min_dscr == policy_credit.RATE_STRESS_MIN_DSCR and rate.variable_rate_only


def test_base_case_formulas_match_the_startup_policy() -> None:
    setup = setup_for(loan=basic_loan(annual_rate_pct=9.0))
    service = round(proposed_annual_debt_service("term_loan", 1_300_000, 9.0, 120), 2)
    assert lender_annual_debt_service(setup) == service
    assert projected_dscr(setup, YEAR) == coverage_ratio(300_000.0, service)
    revenue, rate = LENDER_STRESS
    stressed_noi = round(300_000.0 - policy_credit.REVENUE_STRESS_SHARE * 1_000_000.0, 2)
    assert base_case_stressed_dscr(revenue, setup, YEAR) == coverage_ratio(stressed_noi, service)
    stressed_service = round(proposed_annual_debt_service("term_loan", 1_300_000, 11.0, 120), 2)
    assert base_case_stressed_dscr(rate, setup, YEAR) == coverage_ratio(300_000.0, stressed_service)
    assert stressed_service > service


def test_description_is_built_from_the_constants() -> None:
    revenue, rate = LENDER_STRESS
    assert f"{policy_credit.REVENUE_STRESS_SHARE:.0%}" in revenue.description
    assert f"{policy_credit.RATE_STRESS_ADD_PCT:.2f}" in rate.description
