"""Deterministic credit calculations from the SYNTHETIC Northwind guidelines (NBC-3.2 to NBC-5.1).

Every function is a plain formula with no thresholds: the reference policy (policy_credit.py) compares the
results with the guideline limits. Rounding follows the guideline conventions: ratios are rounded to two decimal
places and percentages to one decimal place before they are compared with a threshold.
"""

from dataclasses import dataclass
from datetime import date

from uwcopilot.agent.applications import BankStatementSummary, CollateralItem, DebtServiceInputs, MonthlyDeposits

MONTHS_REQUIRED = 12
# NBC-3.3: eligible cash is capped at 25% of the month's eligible non-cash deposits.
CASH_CAP_SHARE_OF_NON_CASH = 0.25
# NBC-4.1: a line of credit's proposed debt service is a year of interest plus 10% of the commitment.
LINE_OF_CREDIT_PRINCIPAL_SHARE = 0.10
# NBC-5.1: maximum advance rate by collateral type.
ADVANCE_RATES: dict[str, float] = {
    "owner_occupied_cre": 0.75,
    "new_equipment": 0.80,
    "used_equipment": 0.60,
    "accounts_receivable": 0.70,
    "inventory": 0.50,
}


def round_ratio(value: float) -> float:
    """Ratios (DSCR, leverage, coverage) are compared at two decimal places."""
    return round(value, 2)


def round_pct(value: float) -> float:
    """Percentages are compared at one decimal place."""
    return round(value, 1)


def months_between(start: date, end: date) -> int:
    """Whole calendar months from start to end (a partial last month does not count)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(months, 0)


def amortizing_monthly_payment(principal: float, annual_rate_pct: float, term_months: int) -> float:
    """Level monthly payment that repays principal over term_months at the note rate."""
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    if monthly_rate == 0:
        return principal / term_months
    return principal * monthly_rate / (1.0 - (1.0 + monthly_rate) ** (-term_months))


def proposed_annual_debt_service(product: str, amount: float, annual_rate_pct: float, term_months: int) -> float:
    """NBC-4.1 item 2: 12 amortizing payments for term and equipment loans; interest plus 10% for a line."""
    if product == "line_of_credit":
        return amount * annual_rate_pct / 100.0 + LINE_OF_CREDIT_PRINCIPAL_SHARE * amount
    return 12.0 * amortizing_monthly_payment(amount, annual_rate_pct, term_months)


@dataclass(frozen=True)
class MonthEligibility:
    """Eligible deposits for one month and the cash excluded by the cap."""

    month: str
    eligible: float
    excluded_cash: float
    non_revenue: float


def month_eligibility(month: MonthlyDeposits, cash_intensive_approved: bool) -> MonthEligibility:
    """NBC-3.3: remove non-revenue deposits, then cap cash at 25% of the eligible non-cash deposits."""
    non_revenue = (
        month.transfers_between_accounts
        + month.refunds_and_reversals
        + month.loan_proceeds
        + month.owner_contributions
        + month.one_time_receipts
        + month.unidentified_deposits
    )
    revenue_deposits = max(0.0, month.total_deposits - non_revenue)
    cash = min(month.cash_deposits, revenue_deposits)
    non_cash = revenue_deposits - cash
    allowed_cash = cash if cash_intensive_approved else min(cash, CASH_CAP_SHARE_OF_NON_CASH * non_cash)
    return MonthEligibility(
        month=month.month,
        eligible=non_cash + allowed_cash,
        excluded_cash=cash - allowed_cash,
        non_revenue=non_revenue,
    )


def month_index(month: str) -> int:
    """'2026-03' -> a running month number, so consecutive months differ by one."""
    year, month_number = month.split("-")
    return int(year) * 12 + int(month_number) - 1


@dataclass(frozen=True)
class RevenueVerification:
    """NBC-3.2 result. verified_annual_revenue is None when 12 consecutive months are not available."""

    months_provided: int
    consecutive: bool
    verified_annual_revenue: float | None
    excluded_cash: float
    non_revenue_total: float


def verify_revenue(statements: BankStatementSummary) -> RevenueVerification:
    """Sum eligible deposits over the 12 most recent consecutive months (None when fewer are available)."""
    months = sorted(statements.months, key=lambda item: month_index(item.month))
    recent = months[-MONTHS_REQUIRED:]
    indices = [month_index(item.month) for item in recent]
    consecutive = all(later - earlier == 1 for earlier, later in zip(indices, indices[1:], strict=False))
    results = [month_eligibility(item, statements.cash_intensive_approved) for item in recent]
    complete = len(recent) == MONTHS_REQUIRED and consecutive
    return RevenueVerification(
        months_provided=len(months),
        consecutive=consecutive,
        verified_annual_revenue=round(sum(item.eligible for item in results), 2) if complete else None,
        excluded_cash=round(sum(item.excluded_cash for item in results), 2),
        non_revenue_total=round(sum(item.non_revenue for item in results), 2),
    )


def revenue_gap_pct(stated_revenue: float, verified_revenue: float) -> float | None:
    """NBC-3.4: (stated - verified) / stated, in percent. None when stated revenue is zero."""
    if stated_revenue <= 0:
        return None
    return round_pct((stated_revenue - verified_revenue) / stated_revenue * 100.0)


def net_operating_income(debt: DebtServiceInputs, verified_revenue: float) -> float:
    """NBC-4.1: TTM EBITDA, reduced dollar for dollar when verified revenue is below the P&L revenue."""
    shortfall = max(0.0, debt.ttm_pnl_revenue - verified_revenue)
    return round(debt.ttm_ebitda - shortfall, 2)


def existing_annual_debt_service(debt: DebtServiceInputs, payoff_letter_in_file: bool) -> float:
    """NBC-4.1 item 1 and NBC-4.4: the higher of scheduled and observed payments, plus MCA remittances.

    Debt being refinanced is left out only when its payoff letter is in the file.
    """
    monthly = max(debt.existing_monthly_payment_schedule, debt.existing_monthly_payment_observed)
    monthly += debt.mca_monthly_remittance
    if payoff_letter_in_file:
        monthly = max(0.0, monthly - debt.refinanced_monthly_payment)
    return round(12.0 * monthly, 2)


def coverage_ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator rounded to two decimals; None when the denominator is not positive."""
    if denominator <= 0:
        return None
    return round_ratio(numerator / denominator)


def collateral_coverage(collateral: list[CollateralItem], loan_amount: float) -> float:
    """NBC-5.1: sum of value x maximum advance rate, divided by the loan amount."""
    lendable = sum(item.value * ADVANCE_RATES[item.collateral_type] for item in collateral)
    return round_ratio(lendable / loan_amount)
