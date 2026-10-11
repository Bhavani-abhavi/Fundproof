"""Deterministic insurance calculations from the SYNTHETIC Harborline guidelines (HM-3.3 to HM-5.2).

Plain formulas only; thresholds live in policy_insurance.py. Loss ratios and other percentages are rounded to one
decimal place before comparison, as the guideline conventions say.
"""

from dataclasses import dataclass
from datetime import date, timedelta

from uwcopilot.agent.applications import Claim, InsuranceApplication, Location, LossRuns

# HM-1.2 / HM-4.1: a single claim of $250,000 or more incurred is a large loss, and claims are capped there.
LARGE_LOSS_USD = 250_000.0
# HM-3.3: paid plus reserve must equal incurred within $1.00.
RECONCILIATION_TOLERANCE_USD = 1.0
ROLLING_WINDOW_DAYS = 365


def round_pct(value: float) -> float:
    """Percentages are compared at one decimal place."""
    return round(value, 1)


def location_tiv(location: Location) -> float:
    """HM-1.2: building replacement cost + business personal property + 12 months of business income."""
    return location.building_value + location.contents_value + location.business_income_value


def schedule_tiv(application: InsuranceApplication) -> float:
    """Total insured value summed over every location on the property schedule."""
    return round(sum(location_tiv(location) for location in application.locations), 2)


def tiv_difference_pct(stated_tiv: float, schedule_total: float) -> float:
    """HM-5.2: |application TIV - schedule TIV| / schedule TIV, in percent."""
    return round_pct(abs(stated_tiv - schedule_total) / schedule_total * 100.0)


def building_cost_per_sqft(location: Location) -> float:
    """Building replacement cost per square foot, for the underinsurance check in HM-5.2."""
    return round(location.building_value / location.square_feet, 2)


def capped_incurred(claim: Claim) -> float:
    """HM-4.1 item 2: each claim counts at most $250,000 in a loss ratio."""
    return min(claim.incurred, LARGE_LOSS_USD)


def claim_out_of_balance(claim: Claim) -> bool:
    """HM-3.3 / HM-11.1: paid + reserve differs from incurred by more than $1.00."""
    return abs(claim.paid + claim.reserve - claim.incurred) > RECONCILIATION_TOLERANCE_USD


def earned_premium(year_premiums: tuple[float | None, float | None], line: str) -> float | None:
    """Pick the property or general liability premium from (property, gl)."""
    return year_premiums[0] if line == "property" else year_premiums[1]


@dataclass(frozen=True)
class LossRatioSummary:
    """Loss ratios for one line. five_year_pct is None when a covered year has no earned premium."""

    line: str
    annual_pct: list[float | None]
    five_year_pct: float | None
    missing_premium_years: list[str]


def loss_ratios(loss_runs: LossRuns, line: str) -> LossRatioSummary:
    """HM-4.1: annual and 5-year loss ratios for one line over the covered years of the experience period."""
    annual: list[float | None] = []
    missing: list[str] = []
    total_losses = 0.0
    total_premium = 0.0
    for year in loss_runs.years:
        premium = earned_premium((year.property_earned_premium, year.gl_earned_premium), line)
        losses = sum(capped_incurred(claim) for claim in year.claims if claim.line == line)
        if premium is None or premium <= 0:
            missing.append(year.policy_year_start.isoformat())
            annual.append(None)
            continue
        annual.append(round_pct(losses / premium * 100.0))
        total_losses += losses
        total_premium += premium
    five_year = None
    if not missing and total_premium > 0:
        five_year = round_pct(total_losses / total_premium * 100.0)
    return LossRatioSummary(line=line, annual_pct=annual, five_year_pct=five_year, missing_premium_years=missing)


def all_claims(loss_runs: LossRuns) -> list[Claim]:
    """Every claim in the experience period plus the current term."""
    claims = [claim for year in loss_runs.years for claim in year.claims]
    return claims + list(loss_runs.current_term_claims)


def max_claims_in_rolling_window(claims: list[Claim], line: str, window_days: int = ROLLING_WINDOW_DAYS) -> int:
    """HM-4.3: the most claims of one line with a date of loss inside any rolling 12-month window."""
    dates = sorted(claim.date_of_loss for claim in claims if claim.line == line)
    best = 0
    for index, start in enumerate(dates):
        end = start + timedelta(days=window_days)
        count = sum(1 for later in dates[index:] if later < end)
        best = max(best, count)
    return best


def claims_trend(loss_runs: LossRuns) -> tuple[int, int]:
    """HM-4.3: (claims in the 2 most recent policy years, claims in the 3 policy years before them)."""
    counts = [len(year.claims) for year in loss_runs.years]
    recent = sum(counts[-2:])
    earlier = sum(counts[-5:-2])
    return recent, earlier


def loss_run_age_days(loss_runs: LossRuns, effective_date: date) -> int:
    """Days between the loss run valuation date and the proposed effective date (HM-3.2)."""
    return (effective_date - loss_runs.valued_as_of).days


def incurred_difference_pct(stated_total: float, loss_run_total: float) -> float | None:
    """HM-3.3: |stated - loss run| / loss run total incurred, in percent. None when the loss runs show no loss."""
    if loss_run_total <= 0:
        return None
    return round_pct(abs(stated_total - loss_run_total) / loss_run_total * 100.0)
