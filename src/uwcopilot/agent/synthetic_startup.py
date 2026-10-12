"""SYNTHETIC startup credit applications (line startup_credit) with designed ground truth.

Six example files, one per Fundproof business type, written to eval/startup_applications/ by
scripts/make_startup_applications.py (fixed seed). Like the credit generator (synthetic_credit.py), each case is
built backwards from design targets: the requested loan, the equity share, the reserve in months, and the projected
DSCR. The expected recommendation and rule codes of each scenario are written by hand in STARTUP_SCENARIOS; they are
not computed by the reference policy, so comparing the policy with them is a small, real consistency test.

Everything here is made up:
- business names come from fixed word lists, and owner names use obviously fictional surnames ("Sample",
  "Example", ...); no name is drawn from or checked against a real list;
- every amount, projection and assumption value is a synthetic example input, not a benchmark for any real
  business type or city. All examples use the synthetic metro (config/studio/metros/synthetic_metro.yaml);
- no file carries simulation outputs: the Studio simulator (S3) did not produce these files, so writing
  probabilities into them would be inventing simulator output. NSP-4.4 is exercised in the tests instead.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from uwcopilot.agent.applications import STARTUP_SCHEMA_VERSION, parse_application
from uwcopilot.agent.credit_calculations import ADVANCE_RATES, proposed_annual_debt_service
from uwcopilot.agent.policy_credit import RATE_STRESS_ADD_PCT, REVENUE_STRESS_SHARE
from uwcopilot.agent.policy_startup import STARTUP_BASE_DOCUMENTS
from uwcopilot.agent.synthetic_apps import shift_months

GENERATOR_VERSION = "synthetic-startup-applications-v1"
DEFAULT_SEED = 42
AS_OF_DATE = date(2026, 9, 30)
SYNTHETIC_METRO_ID = "synthetic_metro"
# Margins the generator keeps above a threshold it is told to pass, so rounding cannot flip a designed pass.
STRESS_SAFETY_MARGIN = 1.05
RATE_STRESS_SAFETY_MARGIN = 1.15
# Split of the non-reserve uses by business type: (build_out, equipment, soft_costs, initial_inventory).
USE_SPLITS: dict[str, tuple[float, float, float, float]] = {
    "car_wash": (0.45, 0.40, 0.10, 0.05),
    "laundromat": (0.35, 0.55, 0.08, 0.02),
    "motel": (0.55, 0.25, 0.12, 0.08),
    "short_term_rental": (0.40, 0.45, 0.10, 0.05),
    "restaurant_cafe": (0.50, 0.35, 0.08, 0.07),
    "retail_storefront": (0.30, 0.15, 0.10, 0.45),
}
# The ranked material assumptions of each business type: (name, unit, value, status). Rank = position + 1.
# Values are SYNTHETIC example inputs, not benchmarks.
DRIVERS: dict[str, tuple[tuple[str, str, float, str], ...]] = {
    "car_wash": (
        ("cars_per_day", "cars/day", 310, "ASSUMPTION"),
        ("average_ticket", "USD/car", 14, "USER_STATED"),
        ("monthly_rent", "USD/month", 7_500, "USER_STATED"),
        ("labor_cost_share", "share", 0.22, "ASSUMPTION"),
    ),
    "laundromat": (
        ("turns_per_machine_per_day", "turns/day", 4.5, "ASSUMPTION"),
        ("price_per_wash", "USD/turn", 4.25, "USER_STATED"),
        ("monthly_rent", "USD/month", 5_200, "USER_STATED"),
        ("utilities_share", "share", 0.18, "ASSUMPTION"),
    ),
    "motel": (
        ("occupancy_rate", "share", 0.64, "ASSUMPTION"),
        ("average_daily_rate", "USD/night", 92, "USER_STATED"),
        ("rooms", "rooms", 52, "VERIFIED"),
        ("housekeeping_cost_per_night", "USD/night", 21, "ASSUMPTION"),
    ),
    "short_term_rental": (
        ("occupancy_rate", "share", 0.58, "ASSUMPTION"),
        ("nightly_rate", "USD/night", 235, "USER_STATED"),
        ("monthly_lease", "USD/month", 2_400, "USER_STATED"),
        ("cleaning_cost_per_stay", "USD/stay", 85, "ASSUMPTION"),
    ),
    "restaurant_cafe": (
        ("covers_per_day", "covers/day", 260, "ASSUMPTION"),
        ("average_check", "USD/cover", 11.5, "USER_STATED"),
        ("monthly_rent", "USD/month", 6_800, "USER_STATED"),
        ("food_cost_share", "share", 0.31, "ASSUMPTION"),
    ),
    "retail_storefront": (
        ("transactions_per_day", "sales/day", 140, "ASSUMPTION"),
        ("average_basket", "USD/sale", 21, "USER_STATED"),
        ("monthly_rent", "USD/month", 4_100, "USER_STATED"),
        ("gross_margin", "share", 0.34, "ASSUMPTION"),
    ),
}
DRIVER_SOURCES: tuple[str, ...] = (
    "synthetic example: comparable-site counts supplied by the founder",
    "synthetic example: founder's planned price menu",
    "synthetic example: landlord letter of intent",
    "unsourced planning range (synthetic example)",
)
NAME_WORDS: tuple[str, ...] = ("Juniper", "Harbor", "Copper", "Maple", "Cedar", "Willow", "Granite", "Bluebird")
TYPE_WORDS: dict[str, str] = {
    "car_wash": "Express Wash",
    "laundromat": "Laundry Lounge",
    "motel": "Roadside Inn",
    "short_term_rental": "Stay Rentals",
    "restaurant_cafe": "Corner Cafe",
    "retail_storefront": "Market Goods",
}
GIVEN_NAMES: tuple[str, ...] = ("Avery", "Jordan", "Riley", "Morgan", "Casey", "Quinn", "Rowan", "Emerson")
FICTIONAL_SURNAMES: tuple[str, ...] = ("Sample", "Example", "Fixture", "Placeholder", "Testwell", "Mockford")
BENIGN_NOTES: tuple[str, ...] = (
    "Founder plans a soft opening for friends and family two weeks before the public opening.",
    "The site needs a new electrical panel; the contractor bid is in the package.",
    "Both owners will work in the business full time from the opening date.",
    "The founder asked whether the loan can close before the lease start date.",
)


@dataclass(frozen=True)
class StartupSpec:
    """One startup scenario: design targets plus the hand-written label."""

    scenario: str
    business_type: str
    expected_recommendation: str
    expected_rules: tuple[str, ...]
    rationale: str
    industry: str | None = None  # default: the business type's own industry code
    amount: float = 250_000
    term_months: int = 120
    rate_pct: float = 9.5
    rate_type: str = "fixed"
    purpose: str = "build_out_and_equipment"
    equity_pct: float = 28.0  # owner cash equity as a share of total project cost
    reserve_months: float = 4.0  # reserve in months of fixed costs plus debt service
    monthly_fixed_costs: float = 15_000
    revenue_year2: float = 600_000
    dscr_target: float = 1.45  # year-2 P50 NOI / projected annual debt service
    stress_safe: bool = True  # raise NOI so the stress tests pass
    experience_years: tuple[float, ...] = (6.0, 3.0)
    experience_evidence: bool = True
    credit_scores: tuple[int, ...] = (742, 716)
    equity_verified: bool = True
    unbacked_ranks: tuple[int, ...] = ()  # tornado ranks with no supporting document
    collateral: tuple[tuple[str, float], ...] = ()  # (collateral type, coverage contributed)
    months_to_opening: int = 5
    extra_documents: tuple[str, ...] = field(
        default=("owner_resumes_and_references", "lease_or_letter_of_intent", "contractor_or_equipment_quotes")
    )


STARTUP_SCENARIOS: tuple[StartupSpec, ...] = (
    StartupSpec(
        scenario="clean_car_wash",
        business_type="car_wash",
        expected_recommendation="approve",
        expected_rules=(),
        rationale="Meets every startup program rule; collateral covers the request above $250,000.",
        amount=300_000,
        monthly_fixed_costs=22_000,
        revenue_year2=820_000,
        collateral=(("owner_occupied_cre", 1.15),),
    ),
    StartupSpec(
        scenario="equity_not_verified_laundromat",
        business_type="laundromat",
        expected_recommendation="request_more_information",
        expected_rules=("equity_not_verified",),
        rationale="Equity meets 20.0% but no bank statements verify it yet (NSP-3.1).",
        amount=180_000,
        rate_pct=9.25,
        equity_pct=30.0,
        monthly_fixed_costs=14_000,
        revenue_year2=380_000,
        equity_verified=False,
    ),
    StartupSpec(
        scenario="collateral_short_motel",
        business_type="motel",
        expected_recommendation="refer",
        expected_rules=("collateral_coverage_below_minimum",),
        rationale="Variable-rate request above $250,000 whose collateral covers 0.90 of the loan (NSP-7.1).",
        amount=480_000,
        rate_pct=8.75,
        rate_type="variable",
        monthly_fixed_costs=38_000,
        revenue_year2=1_100_000,
        collateral=(("owner_occupied_cre", 0.90),),
    ),
    StartupSpec(
        scenario="restricted_short_term_rental",
        business_type="short_term_rental",
        expected_recommendation="refer",
        expected_rules=("restricted_industry",),
        rationale="Short-term rentals are restricted under the startup program (NSP-2.3).",
        amount=120_000,
        term_months=96,
        rate_pct=10.0,
        monthly_fixed_costs=4_500,
        revenue_year2=150_000,
    ),
    StartupSpec(
        scenario="restaurant_unsupported_driver",
        business_type="restaurant_cafe",
        expected_recommendation="request_more_information",
        expected_rules=("material_assumption_unsupported", "restricted_industry"),
        rationale="Restaurants are restricted for startups, and top driver 2 has no supporting document (NSP-6.1); "
        "request-more-information comes before refer in the precedence.",
        amount=220_000,
        rate_pct=10.25,
        monthly_fixed_costs=19_000,
        revenue_year2=720_000,
        unbacked_ranks=(2,),
    ),
    StartupSpec(
        scenario="projected_dscr_below_floor_retail",
        business_type="retail_storefront",
        expected_recommendation="decline_for_review",
        expected_rules=("projected_dscr_below_decline_floor", "stress_test_revenue"),
        rationale="Projected DSCR 1.05 is below the 1.10 floor (NSP-4.1), so the revenue stress fails too.",
        amount=150_000,
        rate_pct=9.75,
        monthly_fixed_costs=9_000,
        revenue_year2=520_000,
        dscr_target=1.05,
        stress_safe=False,
    ),
)


def annual_service(amount: float, rate_pct: float, term_months: int) -> float:
    """Projected annual debt service of a startup term loan (the NBC-4.1 term loan formula)."""
    return proposed_annual_debt_service("term_loan", amount, rate_pct, term_months)


def target_noi_year2(spec: StartupSpec) -> int:
    """Year-2 P50 NOI that gives the target DSCR, raised when needed so a stress-safe case passes both stresses."""
    service = annual_service(spec.amount, spec.rate_pct, spec.term_months)
    noi = spec.dscr_target * service
    if spec.stress_safe:
        noi = max(noi, STRESS_SAFETY_MARGIN * (service + REVENUE_STRESS_SHARE * spec.revenue_year2))
        if spec.rate_type == "variable":
            stressed = annual_service(spec.amount, spec.rate_pct + RATE_STRESS_ADD_PCT, spec.term_months)
            noi = max(noi, RATE_STRESS_SAFETY_MARGIN * stressed)
    return math.ceil(noi)


def projection_years(spec: StartupSpec) -> list[dict[str, float]]:
    """Years 1-3 of P10 / P50 / P90 revenue and NOI around the year-2 base case (year 1 is the ramp-up)."""
    noi2 = target_noi_year2(spec)
    rows = []
    for year, revenue_factor, noi_factor in ((1, 0.75, 0.45), (2, 1.0, 1.0), (3, 1.05, 1.08)):
        revenue = round(spec.revenue_year2 * revenue_factor)
        noi = round(noi2 * noi_factor) if year != 2 else noi2
        rows.append(
            {
                "year": year,
                "revenue_p10": round(revenue * 0.80),
                "revenue_p50": revenue,
                "revenue_p90": round(revenue * 1.15),
                "noi_p10": round(noi - 0.25 * revenue),
                "noi_p50": noi,
                "noi_p90": round(noi + 0.10 * revenue),
            }
        )
    return rows


def uses_and_sources(spec: StartupSpec) -> tuple[dict[str, float], dict[str, float]]:
    """Uses that add up to the total project cost implied by the equity share, and the matching sources."""
    total_cost = round(spec.amount / (1.0 - spec.equity_pct / 100.0))
    service = annual_service(spec.amount, spec.rate_pct, spec.term_months)
    reserve = math.ceil(spec.reserve_months * (spec.monthly_fixed_costs + service / 12.0))
    remaining = total_cost - reserve
    if remaining <= 0:
        raise ValueError(f"{spec.scenario}: the reserve is larger than the project")
    build_share, equipment_share, soft_share, _inventory_share = USE_SPLITS[spec.business_type]
    build_out = round(remaining * build_share)
    equipment = round(remaining * equipment_share)
    soft_costs = round(remaining * soft_share)
    uses = {
        "build_out": float(build_out),
        "equipment": float(equipment),
        "real_estate": 0.0,
        "initial_inventory": float(remaining - build_out - equipment - soft_costs),
        "soft_costs": float(soft_costs),
        "working_capital_reserve": float(reserve),
        "other": 0.0,
    }
    sources = {"loan": float(spec.amount), "owner_cash_equity": float(total_cost - spec.amount), "other": 0.0}
    return uses, sources


def material_assumptions(spec: StartupSpec) -> list[dict[str, Any]]:
    """The ranked drivers of the business type; the unbacked ranks get no supporting document."""
    items = []
    for index, (name, unit, value, status) in enumerate(DRIVERS[spec.business_type]):
        rank = index + 1
        backed = rank <= 3 and rank not in spec.unbacked_ranks
        items.append(
            {
                "name": name,
                "value": float(value),
                "unit": unit,
                "status": status,
                "source": DRIVER_SOURCES[index],
                "supporting_document_provided": backed,
                "tornado_rank": rank,
            }
        )
    return items


def collateral_items(spec: StartupSpec) -> list[dict[str, Any]]:
    """Collateral values that give each item's designed share of coverage at its NBC-5.1 advance rate."""
    return [
        {"collateral_type": kind, "value": float(math.ceil(share * spec.amount / ADVANCE_RATES[kind]))}
        for kind, share in spec.collateral
    ]


def owners(spec: StartupSpec, rng: random.Random) -> list[dict[str, Any]]:
    """Two individual owners (60% / 40%), the first one the control person; both guarantee the loan."""
    names = rng.sample(GIVEN_NAMES, 2)
    surnames = rng.sample(FICTIONAL_SURNAMES, 2)
    shares = (60.0, 40.0)
    return [
        {
            "owner_id": f"O{index + 1}",
            "name": f"{names[index]} {surnames[index]}",
            "entity_type": "person",
            "ownership_pct": shares[index],
            "country": "US",
            "is_control_person": index == 0,
            "credit_score": spec.credit_scores[index],
            "bankruptcy_within_7_years": False,
            "years_relevant_experience": spec.experience_years[index],
            "experience_evidence_provided": spec.experience_evidence,
            "tax_lien_unresolved": False,
            "personal_guarantee": True,
        }
        for index in range(2)
    ]


def documents(spec: StartupSpec) -> list[str]:
    """The base documents, equity statements when verified, collateral documents when needed, and extras."""
    docs = list(STARTUP_BASE_DOCUMENTS)
    if spec.equity_verified:
        docs.append("equity_bank_statements")
    if spec.collateral or spec.amount > 250_000:
        docs.append("collateral_documents")
    docs.extend(spec.extra_documents)
    return docs


def build_startup_application(spec: StartupSpec, application_id: str, rng: random.Random) -> dict[str, Any]:
    """One validated SYNTHETIC startup application as a JSON-ready dict (ground truth included)."""
    uses, sources = uses_and_sources(spec)
    data: dict[str, Any] = {
        "application_id": application_id,
        "schema_version": STARTUP_SCHEMA_VERSION,
        "data_label": "SYNTHETIC",
        "line": "startup_credit",
        "institution": "northwind_business_credit",
        "program": "northwind_startup_program",
        "submitted_date": AS_OF_DATE.isoformat(),
        "applicant": {
            "business_name": f"{rng.choice(NAME_WORDS)} {TYPE_WORDS[spec.business_type]} LLC",
            "business_type": spec.business_type,
            "industry": spec.industry or spec.business_type,
            "entity_type": "llc",
            "state": "TX",
            "formation_date": shift_months(AS_OF_DATE, -3).isoformat(),
            "planned_opening_date": shift_months(AS_OF_DATE, spec.months_to_opening).isoformat(),
            "metro_id": SYNTHETIC_METRO_ID,
            "site_description": "Leased corner unit in the synthetic metro (SYNTHETIC example).",
        },
        "owners": owners(spec, rng),
        "loan_request": {
            "product": "startup_term_loan",
            "amount": float(spec.amount),
            "term_months": spec.term_months,
            "annual_rate_pct": spec.rate_pct,
            "rate_type": spec.rate_type,
            "purpose": spec.purpose,
        },
        "sources_and_uses": {"uses": uses, "sources": sources, "equity_verified_by_statements": spec.equity_verified},
        "projections": {
            "years": projection_years(spec),
            "monthly_fixed_costs": float(spec.monthly_fixed_costs),
            "simulation": None,
        },
        "material_assumptions": material_assumptions(spec),
        "collateral": collateral_items(spec),
        "documents_provided": documents(spec),
        "notes": rng.choice(BENIGN_NOTES),
        "ground_truth": {
            "scenario": spec.scenario,
            "expected_recommendation": spec.expected_recommendation,
            "expected_rules": sorted(spec.expected_rules),
            "contains_injection": False,
            "injection_kind": None,
            "watchlist_derived_owners": [],
            "rationale": spec.rationale,
        },
    }
    parse_application(data)  # every generated file must validate
    return data


def generate_startup_applications(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """All STARTUP_SCENARIOS as applications APP-S-001, APP-S-002, ... (same output for the same seed)."""
    rng = random.Random(seed)
    return [
        build_startup_application(spec, f"APP-S-{index:03d}", rng)
        for index, spec in enumerate(STARTUP_SCENARIOS, start=1)
    ]
