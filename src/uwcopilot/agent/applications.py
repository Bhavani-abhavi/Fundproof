"""The structured application the agent reviews: a small-business loan (credit) or a commercial insurance submission.

Every application in this project is SYNTHETIC (scripts/make_applications.py). These structured JSON files stand
in for document extraction, which is Phase 2: in production the same fields would be extracted, with page
citations, from bank statements, tax returns, loss runs and property schedules.

Two kinds of field:
- structured fields (amounts, dates, codes, booleans) are validated here and are the only input to the reference
  policy and to tool planning;
- free text (the "notes" field, and names) is UNTRUSTED data. It is screened for instruction-like text, it never
  chooses a tool call, and it reaches the LLM only quoted as data.

ground_truth holds the label the synthetic generator designed for each case (expected recommendation and the rules
it is built to trigger). Nothing in the agent reads it; only the evaluation does.
"""

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from uwcopilot.agent.vocab import Recommendation

# --------------------------------------------------------------------------------------------------------------
# Code lists. The policy decides what each code means (prohibited, restricted, allowed).
# --------------------------------------------------------------------------------------------------------------

CREDIT_INDUSTRIES: dict[str, str] = {
    "professional_services": "professional services (accounting and consulting)",
    "dental_practice": "dental practice",
    "medical_clinic": "outpatient medical clinic",
    "software_development": "software development",
    "landscaping": "landscaping services",
    "hardware_retail": "hardware retail store",
    "auto_repair": "auto repair shop",
    "machine_shop": "machine shop manufacturing",
    "bakery": "bakery",
    "restaurant": "full-service restaurant",
    "commercial_printing": "commercial printing",
    "veterinary_clinic": "veterinary clinic",
    "long_haul_trucking": "long-haul trucking",
    "freight_broker": "freight broker",
    "construction_contractor": "construction contractor",
    "used_vehicle_dealer": "used vehicle dealer",
    "money_services_business": "money services business",
    "cannabis_retail": "cannabis retail",
    "adult_entertainment": "adult entertainment",
    "casino_gambling": "casino and gambling",
    "firearms_manufacturing": "firearms manufacturing",
    "payday_lending": "payday lending",
    "virtual_currency_exchange": "virtual currency exchange",
    "multi_level_marketing": "multi-level marketing",
    "speculative_real_estate": "speculative real estate development",
    "political_lobbying": "political lobbying",
    "shell_company": "shell company with no operating revenue",
}

INSURANCE_CLASSES: dict[str, str] = {
    "office": "professional office",
    "retail_store": "retail store",
    "light_assembly": "light assembly manufacturing",
    "dental_office": "dental office",
    "restaurant_no_fryer": "restaurant without deep-fat frying",
    "restaurant_fryer": "restaurant with deep-fat frying",
    "warehouse": "warehouse and distribution",
    "habitational": "apartment building",
    "general_contractor": "general contractor",
    "woodworking": "woodworking shop",
    "furniture_manufacturing": "furniture manufacturing",
    "plastics_manufacturing": "plastics manufacturing",
    "auto_repair": "auto repair shop",
    "body_shop": "auto body shop",
    "fuel_station": "fuel station",
    "fireworks_storage": "fireworks storage",
    "ammunition_manufacturing": "ammunition manufacturing",
    "cannabis_dispensary": "cannabis dispensary",
    "crypto_mining": "virtual currency mining facility",
    "demolition_contractor": "demolition and blasting contractor",
    "large_nightclub": "nightclub with licensed occupancy above 300",
    "waste_recycling": "waste processing and recycling",
    "salvage_yard": "salvage yard",
}

CREDIT_DOCUMENTS: tuple[str, ...] = (
    "signed_application",
    "bank_statements_12_months",
    "business_tax_returns_2_years",
    "ytd_pnl_and_balance_sheet",
    "debt_schedule",
    "principal_photo_ids",
    "formation_documents",
    "beneficial_ownership_certification",
    "cpa_reviewed_financials",
    "collateral_documents",
    "payoff_letters",
)
INSURANCE_DOCUMENTS: tuple[str, ...] = (
    "signed_application",
    "property_schedule",
    "loss_runs",
    "prior_carrier_history",
    "revenue_and_payroll",
    "beneficial_ownership_statement",
    "inspection_report",
    "elevation_certificate",
)

CreditProduct = Literal["term_loan", "line_of_credit", "equipment_loan"]
BusinessStructure = Literal["corporation", "llc", "partnership", "sole_proprietorship"]
CollateralType = Literal["owner_occupied_cre", "new_equipment", "used_equipment", "accounts_receivable", "inventory"]
ConstructionClass = Literal["frame", "joisted_masonry", "non_combustible", "masonry_non_combustible", "fire_resistive"]
Coverage = Literal["property", "general_liability", "package"]
ClaimLine = Literal["property", "general_liability"]
WildfireHazard = Literal["low", "moderate", "high", "very_high"]
PartyEntityType = Literal["person", "organization"]
CreditIndustry = Literal[tuple(CREDIT_INDUSTRIES)]  # type: ignore[valid-type]
InsuranceClass = Literal[tuple(INSURANCE_CLASSES)]  # type: ignore[valid-type]
CreditDocument = Literal[CREDIT_DOCUMENTS]  # type: ignore[valid-type]
InsuranceDocument = Literal[INSURANCE_DOCUMENTS]  # type: ignore[valid-type]
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
APPLICATION_ID_PATTERN = r"^APP-[CI]-\d{3}$"
MAX_NOTES_CHARS = 4000


class Strict(BaseModel):
    """Base for every application model: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------------------------------------
# Shared parts
# --------------------------------------------------------------------------------------------------------------


class Owner(Strict):
    """One owner of the applicant. Organizations can own shares too (layered ownership)."""

    owner_id: str = Field(pattern=r"^O\d{1,2}$")
    name: str = Field(min_length=1, max_length=200)
    entity_type: PartyEntityType
    ownership_pct: float = Field(ge=0, le=100)
    country: str | None = Field(default=None, max_length=64)
    is_control_person: bool = False
    credit_score: int | None = Field(default=None, ge=300, le=850)
    bankruptcy_within_7_years: bool = False


class WatchlistDerivedOwner(Strict):
    """Ground truth: an owner whose name was made from a real OFAC SDN organization entry with small noise."""

    owner_id: str
    derived_from_list_entry_id: str
    derived_from_name: str
    noise_operations: list[str]


class GroundTruth(Strict):
    """What the generator designed this case to be. Read only by the evaluation, never by the agent."""

    scenario: str = Field(min_length=1)
    expected_recommendation: Recommendation
    expected_rules: list[str]
    contains_injection: bool = False
    injection_kind: str | None = None
    watchlist_derived_owners: list[WatchlistDerivedOwner] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


# --------------------------------------------------------------------------------------------------------------
# Credit (Northwind Business Credit, corpus/guidelines/northwind_business_credit.md)
# --------------------------------------------------------------------------------------------------------------


class CreditApplicant(Strict):
    """The business asking for the loan."""

    legal_name: str = Field(min_length=1, max_length=200)
    fein: str = Field(pattern=r"^\d{2}-\d{7}$")
    business_structure: BusinessStructure
    state: str = Field(pattern=r"^[A-Z]{2}$")
    formation_date: date
    industry: CreditIndustry


class MonthlyDeposits(Strict):
    """One month of deposits to the operating accounts, split the way NBC-3.3 needs.

    total_deposits includes everything below. cash_deposits is the cash part of the revenue deposits.
    unidentified_deposits are deposits whose nature the statement does not show (treated as non-revenue).
    """

    month: str = Field(pattern=MONTH_PATTERN)
    total_deposits: float = Field(ge=0)
    transfers_between_accounts: float = Field(default=0, ge=0)
    refunds_and_reversals: float = Field(default=0, ge=0)
    loan_proceeds: float = Field(default=0, ge=0)
    owner_contributions: float = Field(default=0, ge=0)
    one_time_receipts: float = Field(default=0, ge=0)
    unidentified_deposits: float = Field(default=0, ge=0)
    cash_deposits: float = Field(default=0, ge=0)


class BankStatementSummary(Strict):
    """Summary of the bank statements (NBC-3.2 to NBC-3.5). months are oldest first."""

    account_number: str = Field(pattern=r"^\d{10,12}$")
    months: list[MonthlyDeposits]
    nsf_events_last_6_months: int = Field(ge=0)
    negative_balance_months_last_12: int = Field(ge=0, le=12)
    average_daily_balance_last_3_months: float
    cash_intensive_approved: bool = False


class DebtServiceInputs(Strict):
    """Inputs for NOI and annual debt service (NBC-4.1) and leverage (NBC-4.4). Monthly amounts in USD."""

    ttm_ebitda: float
    ttm_pnl_revenue: float = Field(ge=0)
    existing_monthly_payment_schedule: float = Field(ge=0)
    existing_monthly_payment_observed: float = Field(ge=0)
    mca_monthly_remittance: float = Field(default=0, ge=0)
    refinanced_monthly_payment: float = Field(default=0, ge=0)
    existing_total_debt_balance: float = Field(ge=0)


class CollateralItem(Strict):
    """One pledged collateral item and its value on the basis NBC-5.1 requires."""

    collateral_type: CollateralType
    value: float = Field(gt=0)


class CreditApplication(Strict):
    """A SYNTHETIC small-business loan application."""

    application_id: str = Field(pattern=APPLICATION_ID_PATTERN)
    data_label: Literal["SYNTHETIC"] = "SYNTHETIC"
    line: Literal["credit"] = "credit"
    institution: Literal["northwind_business_credit"] = "northwind_business_credit"
    application_date: date
    applicant: CreditApplicant
    product: CreditProduct
    requested_amount: float = Field(gt=0)
    term_months: int = Field(ge=1, le=360)
    note_rate_pct: float = Field(gt=0, lt=40)
    rate_type: Literal["fixed", "variable"]
    purpose: Literal["working_capital", "equipment_purchase", "expansion", "refinance"]
    stated_annual_revenue: float = Field(ge=0)
    bank_statements: BankStatementSummary
    debt: DebtServiceInputs
    new_loans_last_90_days: int = Field(ge=0)
    tax_lien_unresolved: bool = False
    largest_customer_revenue_pct: float = Field(ge=0, le=100)
    collateral: list[CollateralItem] = Field(default_factory=list)
    owners: list[Owner] = Field(min_length=1)
    documents_provided: list[CreditDocument]
    notes: str = Field(default="", max_length=MAX_NOTES_CHARS)
    ground_truth: GroundTruth | None = None

    @model_validator(mode="after")
    def owner_ids_unique(self) -> "CreditApplication":
        """Owner ids identify parties in tool calls and memos, so they must be unique."""
        ids = [owner.owner_id for owner in self.owners]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate owner ids: {ids}")
        return self


# --------------------------------------------------------------------------------------------------------------
# Insurance (Harborline Mutual, corpus/guidelines/harborline_mutual_commercial.md)
# --------------------------------------------------------------------------------------------------------------


class InsuranceApplicant(Strict):
    """The business asking for coverage."""

    legal_name: str = Field(min_length=1, max_length=200)
    fein: str = Field(pattern=r"^\d{2}-\d{7}$")
    years_in_business: int = Field(ge=0)
    prior_experience_years: int = Field(default=0, ge=0)
    class_code: InsuranceClass
    fryer_suppression_inspected_within_6_months: bool = False


class Location(Strict):
    """One insured location on the property schedule (HM-5.1). Values are replacement cost in USD."""

    location_id: str = Field(pattern=r"^L\d{1,2}$")
    construction_class: ConstructionClass
    year_built: int = Field(ge=1800, le=2100)
    roof_year: int = Field(ge=1800, le=2100)
    systems_updated_year: int | None = Field(default=None, ge=1800, le=2100)
    stories: int = Field(ge=1, le=100)
    square_feet: int = Field(gt=0)
    sprinklered: bool
    protection_class: int = Field(ge=1, le=10)
    knob_and_tube_or_fuse_box: bool = False
    building_value: float = Field(ge=0)
    contents_value: float = Field(ge=0)
    business_income_value: float = Field(ge=0)
    wind_tier: Literal[0, 1, 2] = 0
    flood_zone: str = Field(default="X", pattern=r"^[A-Z]{1,3}$")
    wildfire_hazard: WildfireHazard = "low"
    vacant_days: int = Field(default=0, ge=0)


class Claim(Strict):
    """One claim line from a carrier loss run. incurred should equal paid + reserve (HM-3.3)."""

    date_of_loss: date
    line: ClaimLine
    paid: float = Field(ge=0)
    reserve: float = Field(ge=0)
    incurred: float = Field(ge=0)


class LossRunYear(Strict):
    """One policy year of the experience period. A None premium means the line was not written that year."""

    policy_year_start: date
    property_earned_premium: float | None = Field(default=None, ge=0)
    gl_earned_premium: float | None = Field(default=None, ge=0)
    claims: list[Claim] = Field(default_factory=list)


class LossRuns(Strict):
    """Carrier loss runs (HM-3.2). years are the covered policy years, oldest first."""

    carrier_produced: bool
    valued_as_of: date
    years: list[LossRunYear]
    uncovered_years: int = Field(default=0, ge=0)
    uncovered_years_explained: bool = True
    current_term_claims: list[Claim] = Field(default_factory=list)


class InsuranceApplication(Strict):
    """A SYNTHETIC commercial property and general liability submission."""

    application_id: str = Field(pattern=APPLICATION_ID_PATTERN)
    data_label: Literal["SYNTHETIC"] = "SYNTHETIC"
    line: Literal["insurance"] = "insurance"
    institution: Literal["harborline_mutual_commercial"] = "harborline_mutual_commercial"
    application_date: date
    proposed_effective_date: date
    applicant: InsuranceApplicant
    coverage: Coverage
    gl_limit_per_occurrence: float | None = Field(default=None, gt=0)
    gl_limit_aggregate: float | None = Field(default=None, gt=0)
    stated_tiv: float = Field(gt=0)
    locations: list[Location] = Field(min_length=1)
    loss_runs: LossRuns
    stated_claim_count: int = Field(ge=0)
    stated_total_incurred: float = Field(ge=0)
    annual_revenue: float = Field(ge=0)
    annual_payroll: float = Field(ge=0)
    subcontracted_work_pct: float = Field(default=0, ge=0, le=100)
    has_subcontractor_coi_tracking: bool = False
    foreign_products: bool = False
    owners: list[Owner] = Field(min_length=1)
    documents_provided: list[InsuranceDocument]
    notes: str = Field(default="", max_length=MAX_NOTES_CHARS)
    ground_truth: GroundTruth | None = None

    @model_validator(mode="after")
    def ids_unique(self) -> "InsuranceApplication":
        """Owner and location ids must be unique."""
        owner_ids = [owner.owner_id for owner in self.owners]
        location_ids = [location.location_id for location in self.locations]
        if len(owner_ids) != len(set(owner_ids)) or len(location_ids) != len(set(location_ids)):
            raise ValueError("owner ids and location ids must be unique")
        return self


Application = Annotated[CreditApplication | InsuranceApplication, Field(discriminator="line")]
APPLICATION_ADAPTER: TypeAdapter[CreditApplication | InsuranceApplication] = TypeAdapter(Application)


# --------------------------------------------------------------------------------------------------------------
# Parties to screen, loading and hashing
# --------------------------------------------------------------------------------------------------------------


class Party(Strict):
    """Someone who must be screened (NBC-6.3, HM-8.2). party_id is used instead of the name in memos."""

    party_id: str
    name: str
    entity_type: PartyEntityType
    country: str | None
    role: str


def parse_application(data: dict) -> CreditApplication | InsuranceApplication:
    """Validate a dict as a credit or insurance application (chosen by its "line" field)."""
    return APPLICATION_ADAPTER.validate_python(data)


def load_application(path: Path) -> CreditApplication | InsuranceApplication:
    """Read and validate one application JSON file."""
    return parse_application(json.loads(path.read_text(encoding="utf-8")))


def load_applications(directory: Path) -> list[CreditApplication | InsuranceApplication]:
    """Every APP-*.json file in a directory, sorted by file name."""
    paths = sorted(directory.glob("APP-*.json"))
    if not paths:
        raise FileNotFoundError(f"No APP-*.json files in {directory}. Run `make applications` first.")
    return [load_application(path) for path in paths]


def application_inputs(application: CreditApplication | InsuranceApplication) -> dict:
    """The application as plain JSON data, without the ground-truth label (which the agent never sees)."""
    return application.model_dump(mode="json", exclude={"ground_truth"})


def canonical_json(data: object) -> str:
    """Stable JSON text: sorted keys, no extra spaces. Used for hashes and loop-guard signatures."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def inputs_sha256(application: CreditApplication | InsuranceApplication) -> str:
    """SHA-256 of the application inputs (ground truth excluded), recorded in every decision record."""
    return hashlib.sha256(canonical_json(application_inputs(application)).encode("utf-8")).hexdigest()


def screened_parties(application: CreditApplication | InsuranceApplication) -> list[Party]:
    """The applicant and every listed owner: everyone the guidelines require to be screened."""
    parties = [
        Party(
            party_id="APPLICANT",
            name=application.applicant.legal_name,
            entity_type="organization",
            country="US",
            role="applicant",
        )
    ]
    for owner in application.owners:
        parties.append(
            Party(
                party_id=owner.owner_id,
                name=owner.name,
                entity_type=owner.entity_type,
                country=owner.country,
                role=f"owner {owner.ownership_pct:g}%",
            )
        )
    return parties
