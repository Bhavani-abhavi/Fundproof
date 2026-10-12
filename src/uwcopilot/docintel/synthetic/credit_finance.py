"""Financial statements of a SYNTHETIC loan case, worked backwards from a target decision DSCR.

The decision DSCR (NBC-4.1) is NOI / annual debt service, where NOI is P&L EBITDA reduced by any amount the P&L
revenue exceeds verified revenue. The generator draws a target DSCR from a mixture (40% within +-0.05 of the
1.10 / 1.25 / 1.35 thresholds, 40% in 1.5-3.0, 20% in 0.8-1.1), computes the annual debt service from the
as-documented debts and the requested loan, and sets EBITDA so the target is met. NOI is kept at or below 70% of
verified revenue; when the target would need more, the requested amount is reduced (as in agent/synthetic_credit).
"""

import math
import random
from datetime import date, timedelta

from uwcopilot.agent.credit_calculations import ADVANCE_RATES, proposed_annual_debt_service
from uwcopilot.agent.policy_credit import PRODUCT_LIMITS
from uwcopilot.docintel.synthetic.common import APPRAISERS, cents, round_to, weighted_choice
from uwcopilot.docintel.synthetic.credit_case import (
    BalanceSheetData,
    CollateralData,
    Debt,
    ProfitAndLossData,
    TaxYear,
)

MAX_NOI_SHARE_OF_REVENUE = 0.70
THRESHOLDS = (1.10, 1.25, 1.35)
COLLATERAL_ABOVE = 250_000
PNL_NOTES: tuple[str, ...] = (
    "Prepared by management from the company's accounting records.",
    "Amounts are unaudited and stated in US dollars.",
)


def draw_dscr_target(rng: random.Random) -> float:
    """The target decision DSCR from the documented mixture."""
    group = weighted_choice(rng, {"near_threshold": 0.40, "comfortable": 0.40, "weak": 0.20})
    if group == "near_threshold":
        return round(rng.choice(THRESHOLDS) + rng.uniform(-0.05, 0.05), 3)
    if group == "comfortable":
        return round(rng.uniform(1.5, 3.0), 3)
    return round(rng.uniform(0.8, 1.1), 3)


def existing_annual_service(schedule_payment: float, observed_payment: float, mca_payment: float) -> float:
    """NBC-4.1 item 1: 12 x (higher of scheduled and observed payment + MCA remittances); no payoff letters."""
    return 12.0 * (max(schedule_payment, observed_payment) + mca_payment)


def solve_amount_and_noi(
    product: str, amount: float, rate: float, term: int, existing: float, verified: float, dscr: float
) -> tuple[float, float]:
    """(requested amount, NOI): NOI = dscr x total service, capped at 70% of verified revenue.

    When the cap binds the amount is reduced to the largest whole $1,000 that keeps the target; never below the
    NBC-1.1 product minimum (then NOI is the cap and the DSCR ends up below the target).
    """
    per_dollar = proposed_annual_debt_service(product, 1.0, rate, term)
    cap = MAX_NOI_SHARE_OF_REVENUE * verified
    noi = dscr * (existing + per_dollar * amount)
    if noi > cap:
        affordable = (cap / dscr - existing) / per_dollar
        minimum = PRODUCT_LIMITS[product][0]
        amount = max(float(minimum), math.floor(affordable / 1_000) * 1_000.0)
        noi = min(dscr * (existing + per_dollar * amount), cap)
    return amount, cents(noi)


def build_pnl(
    rng: random.Random, period_end: date, revenue: float, ebitda: float, loans: list[Debt], fixed_assets: float
) -> ProfitAndLossData:
    """A trailing-twelve-month P&L with the given revenue and EBITDA (COGS and operating expenses fill the rest)."""
    upper = min(0.55, 1.0 - ebitda / revenue - 0.08)
    cogs_share = rng.uniform(0.15, upper) if upper > 0.15 else max(0.0, upper)
    cogs = cents(revenue * cogs_share)
    operating = cents(revenue - cogs - ebitda)
    depreciation = cents(fixed_assets * rng.uniform(0.08, 0.15))
    interest = cents(sum(debt.balance for debt in loans) * rng.uniform(0.06, 0.10))
    return ProfitAndLossData(
        period_end=period_end,
        revenue=cents(revenue),
        cost_of_goods_sold=cogs,
        operating_expenses=operating,
        ebitda=cents(ebitda),
        depreciation_amortization=depreciation,
        interest_expense=interest,
        net_income=cents(ebitda - depreciation - interest),
        notes=list(PNL_NOTES),
    )


def build_balance_sheet(
    rng: random.Random, as_of: date, operating_cash: float, revenue: float, loans: list[Debt], fixed_assets: float
) -> BalanceSheetData:
    """A balance sheet at the P&L date: cash includes a savings balance; long-term debt is the scheduled loans."""
    cash = cents(max(0.0, operating_cash) + revenue / 12 * rng.uniform(0.05, 0.5))
    receivables = cents(revenue * rng.uniform(0.02, 0.12))
    inventory = cents(revenue * rng.uniform(0.0, 0.10))
    total_assets = cents(cash + receivables + inventory + fixed_assets)
    payables = cents(revenue * rng.uniform(0.02, 0.08))
    long_term = cents(sum(debt.balance for debt in loans))
    liabilities = cents(payables + long_term)
    return BalanceSheetData(
        as_of=as_of,
        cash=cash,
        accounts_receivable=receivables,
        inventory=inventory,
        fixed_assets=cents(fixed_assets),
        total_assets=total_assets,
        accounts_payable=payables,
        long_term_debt=long_term,
        total_liabilities=liabilities,
        equity=cents(total_assets - liabilities),
    )


def build_tax_years(rng: random.Random, latest_receipts: float, application_date: date) -> list[TaxYear]:
    """The two most recent tax years (calendar years), oldest first; the prior year grew by -5% to +12%."""
    latest_year = application_date.year - 1
    prior = latest_receipts / (1.0 + rng.uniform(-0.05, 0.12))
    return [TaxYear(latest_year - 1, float(round(prior))), TaxYear(latest_year, float(round(latest_receipts)))]


def collateral_type(rng: random.Random, product: str) -> str:
    """The collateral that goes with a product."""
    if product == "term_loan":
        return "owner_occupied_cre"
    if product == "equipment_loan":
        return "new_equipment" if rng.random() < 0.8 else "used_equipment"
    return "accounts_receivable" if rng.random() < 0.7 else "inventory"


COLLATERAL_DESCRIPTIONS = {
    "owner_occupied_cre": "Owner-occupied commercial building at the business address",
    "new_equipment": "New production equipment to be purchased with the loan",
    "used_equipment": "Used production equipment owned by the business",
    "accounts_receivable": "Eligible accounts receivable under 90 days",
    "inventory": "Finished goods inventory at the business premises",
}


def build_collateral(rng: random.Random, product: str, amount: float, application_date: date) -> list[CollateralData]:
    """Collateral sized to a coverage of 1.05-1.80 (15% of cases 0.80-0.99) when the request is above $250,000."""
    if amount <= COLLATERAL_ABOVE:
        return []
    kind = collateral_type(rng, product)
    coverage = rng.uniform(0.80, 0.99) if rng.random() < 0.15 else rng.uniform(1.05, 1.80)
    value = round_to(coverage * amount / ADVANCE_RATES[kind], 1_000)
    return [
        CollateralData(
            collateral_type=kind,
            description=COLLATERAL_DESCRIPTIONS[kind],
            value=value,
            valuation_date=application_date - timedelta(days=rng.randint(10, 120)),
            appraiser=rng.choice(APPRAISERS),
        )
    ]
