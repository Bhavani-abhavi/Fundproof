"""Tests for the deterministic credit and insurance calculations."""

from datetime import date

import pytest

from uwcopilot.agent.applications import (
    BankStatementSummary,
    Claim,
    CollateralItem,
    DebtServiceInputs,
    LossRuns,
    LossRunYear,
    MonthlyDeposits,
)
from uwcopilot.agent.credit_calculations import (
    amortizing_monthly_payment,
    collateral_coverage,
    existing_annual_debt_service,
    month_eligibility,
    months_between,
    net_operating_income,
    proposed_annual_debt_service,
    revenue_gap_pct,
    verify_revenue,
)
from uwcopilot.agent.insurance_calculations import (
    capped_incurred,
    claim_out_of_balance,
    claims_trend,
    incurred_difference_pct,
    loss_ratios,
    max_claims_in_rolling_window,
    tiv_difference_pct,
)


def month(name: str, total: float, **parts: float) -> MonthlyDeposits:
    """One month of deposits."""
    return MonthlyDeposits(month=name, total_deposits=total, **parts)


def statements(months: list[MonthlyDeposits]) -> BankStatementSummary:
    """A statement summary with the given months."""
    return BankStatementSummary(
        account_number="9900000000",
        months=months,
        nsf_events_last_6_months=0,
        negative_balance_months_last_12=0,
        average_daily_balance_last_3_months=1000.0,
    )


def test_amortizing_payment_matches_the_standard_formula() -> None:
    assert amortizing_monthly_payment(100_000, 12.0, 12) == pytest.approx(8884.88, abs=0.01)
    assert amortizing_monthly_payment(12_000, 0.0, 12) == pytest.approx(1000.0)


def test_line_of_credit_service_is_interest_plus_ten_percent() -> None:
    assert proposed_annual_debt_service("line_of_credit", 100_000, 10.0, 12) == pytest.approx(20_000.0)
    assert proposed_annual_debt_service("term_loan", 100_000, 12.0, 12) == pytest.approx(12 * 8884.88, abs=0.2)


def test_months_between_counts_whole_months() -> None:
    assert months_between(date(2024, 1, 15), date(2026, 1, 15)) == 24
    assert months_between(date(2024, 1, 15), date(2026, 1, 14)) == 23


def test_month_eligibility_removes_non_revenue_and_caps_cash() -> None:
    row = month("2026-01", 120_000, transfers_between_accounts=10_000, loan_proceeds=10_000, cash_deposits=40_000)
    result = month_eligibility(row, cash_intensive_approved=False)
    # revenue 100,000 = 60,000 non-cash + 40,000 cash; cash allowed = 25% of 60,000 = 15,000.
    assert result.eligible == pytest.approx(75_000)
    assert result.excluded_cash == pytest.approx(25_000)
    assert month_eligibility(row, cash_intensive_approved=True).eligible == pytest.approx(100_000)


def test_verify_revenue_needs_twelve_consecutive_months() -> None:
    full = [month(f"2025-{m:02d}", 10_000) for m in range(1, 13)]
    assert verify_revenue(statements(full)).verified_annual_revenue == pytest.approx(120_000)
    assert verify_revenue(statements(full[:11])).verified_annual_revenue is None
    gap = full[:5] + full[6:] + [month("2026-01", 10_000)]
    result = verify_revenue(statements(gap))
    assert not result.consecutive
    assert result.verified_annual_revenue is None


def test_revenue_gap_and_noi_adjustment() -> None:
    assert revenue_gap_pct(1_000_000, 850_000) == 15.0
    assert revenue_gap_pct(1_000_000, 1_100_000) == -10.0
    assert revenue_gap_pct(0, 10) is None
    debt = DebtServiceInputs(
        ttm_ebitda=200_000,
        ttm_pnl_revenue=1_000_000,
        existing_monthly_payment_schedule=1_000,
        existing_monthly_payment_observed=1_200,
        mca_monthly_remittance=300,
        refinanced_monthly_payment=500,
        existing_total_debt_balance=0,
    )
    assert net_operating_income(debt, 900_000) == pytest.approx(100_000)
    assert net_operating_income(debt, 1_200_000) == pytest.approx(200_000)
    assert existing_annual_debt_service(debt, payoff_letter_in_file=False) == pytest.approx(12 * 1_500)
    assert existing_annual_debt_service(debt, payoff_letter_in_file=True) == pytest.approx(12 * 1_000)


def test_collateral_coverage_uses_advance_rates() -> None:
    items = [CollateralItem(collateral_type="new_equipment", value=100_000)]
    assert collateral_coverage(items, 80_000) == 1.00
    assert collateral_coverage([CollateralItem(collateral_type="inventory", value=100_000)], 80_000) == 0.62


def claim(day: date, line: str = "property", incurred: float = 1_000.0, paid: float | None = None) -> Claim:
    """One claim; paid defaults to incurred (no reserve)."""
    return Claim(date_of_loss=day, line=line, paid=incurred if paid is None else paid, reserve=0, incurred=incurred)


def test_loss_ratios_cap_large_claims_and_need_every_premium() -> None:
    runs = LossRuns(
        carrier_produced=True,
        valued_as_of=date(2026, 10, 1),
        years=[
            LossRunYear(
                policy_year_start=date(2023, 11, 1),
                property_earned_premium=500_000,
                claims=[claim(date(2024, 1, 1), incurred=400_000)],
            ),
            LossRunYear(policy_year_start=date(2024, 11, 1), property_earned_premium=500_000),
        ],
    )
    summary = loss_ratios(runs, "property")
    assert capped_incurred(runs.years[0].claims[0]) == 250_000
    assert summary.annual_pct == [50.0, 0.0]
    assert summary.five_year_pct == 25.0
    missing = runs.model_copy(update={"years": [*runs.years, LossRunYear(policy_year_start=date(2025, 11, 1))]})
    assert loss_ratios(missing, "property").five_year_pct is None


def test_rolling_window_excludes_a_claim_exactly_365_days_later() -> None:
    claims = [claim(date(2024, 1, 1)), claim(date(2024, 12, 31)), claim(date(2025, 6, 1))]
    assert max_claims_in_rolling_window(claims, "property") == 2
    close = [claim(date(2024, 1, 1)), claim(date(2024, 3, 1)), claim(date(2024, 12, 30))]
    assert max_claims_in_rolling_window(close, "property") == 3


def test_trend_reconciliation_and_tiv_difference() -> None:
    years = [LossRunYear(policy_year_start=date(2020 + index, 11, 1)) for index in range(5)]
    years[4] = years[4].model_copy(update={"claims": [claim(date(2025, 1, 1)), claim(date(2025, 2, 1))]})
    runs = LossRuns(carrier_produced=True, valued_as_of=date(2026, 10, 1), years=years)
    assert claims_trend(runs) == (2, 0)
    assert claim_out_of_balance(claim(date(2025, 1, 1), incurred=1_000, paid=990))
    assert not claim_out_of_balance(claim(date(2025, 1, 1), incurred=1_000, paid=999.5))
    assert incurred_difference_pct(110, 100) == 10.0
    assert incurred_difference_pct(10, 0) is None
    assert tiv_difference_pct(1_200_000, 1_000_000) == 20.0
