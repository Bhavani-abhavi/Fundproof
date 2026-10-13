"""Request and response models of the Founder Studio API (/api/studio/*). The router docstring shows the JSON.

Where a Studio module already has a model for a part of a response (a FeasibilityResult, a SourceRecord, a lever
config, a path-search Outcome), that model is reused as it is, so the API cannot drift from the engine. Models built
here only rename fields to the names of the API contract (web_v01_spec: factors {name, raw, status, score, weight,
reason, source}; demand {low, p50, high, unit}) or add what the web app needs (site index, project version).

Labels every response carries (Studio addendum rules C and E): `disclaimer` (the Studio disclaimer) and, for
every simulated probability, `label` = "simulated, based on stated assumptions".
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uwcopilot.studio.data.fabric_base import FactorValue
from uwcopilot.studio.data.provenance import SourceRecord
from uwcopilot.studio.feasibility.assumptions import DscrTarget
from uwcopilot.studio.feasibility.result import FeasibilityResult
from uwcopilot.studio.path_to_approval.levers import DefaultBounds
from uwcopilot.studio.path_to_approval.models import (
    AppliedChange,
    LeverSummary,
    Outcome,
    PlanAmounts,
    SearchSettings,
    SearchStats,
    Target,
)
from uwcopilot.studio.profiles.models import Quantity
from uwcopilot.studio.siteiq.models import ContextItem, DemandInput, TradeAreaInfo

STUDIO_ID_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
PROJECT_ID_PATTERN = r"^[0-9a-f]{16}$"
MAX_SITES = 3
IDEA_MAX_CHARS = 2000
MAX_MONEY = 1_000_000_000.0
LOAN_TERM_MIN_MONTHS = 24
LOAN_TERM_MAX_MONTHS = 360
ANNUAL_RATE_MAX_PCT = 30.0
MetroModeName = Literal["full", "limited", "synthetic"]
DataLabelName = Literal["PUBLIC", "SYNTHETIC"]
RateType = Literal["fixed", "variable"]
EntryStatus = Literal["USER_STATED", "ASSUMPTION"]


class RequestModel(BaseModel):
    """Base of every request body: unknown keys are refused, text is stripped."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------------------------


class MetroOption(BaseModel):
    """One metro the founder can pick, with its honest data mode."""

    id: str
    name: str
    mode: MetroModeName = Field(description="full | limited | synthetic, computed from the data present here")
    mode_reason: str
    data_label: DataLabelName
    coverage_note: str = Field(description="One plain-language line for the city picker")
    verified: bool = Field(description="County list read from the Census delineation file")
    n_tracts: int = Field(description="Tracts available for the map and for area suggestions (0 = none)")
    site_input: Literal["map", "address_or_map"] = Field(
        description="map: pick sites on the metro's map (SYNTHETIC metro); address_or_map: type an address "
        "(geocoded by the Census Geocoder) or drop a pin"
    )
    supports_area_suggestions: bool
    prime_rate: FactorValue | None = Field(
        description="The prime rate the simulator adds the spread to when annual_rate_pct is empty (UNKNOWN = "
        "enter the lender's quoted rate)"
    )


class BusinessTypeOption(BaseModel):
    """One business type (a Studio profile)."""

    id: str
    name: str
    demand_unit: str
    build_out_low: float = Field(description="Sum of the profile's capex lows, USD")
    build_out_high: float = Field(description="Sum of the profile's capex highs, USD")
    build_out_status: Literal["SOURCED", "ASSUMPTION"] = Field(description="ASSUMPTION when any capex item is one")
    build_out_note: str


class LeverOption(BaseModel):
    """A plan-choice lever of the path search with its default bounds (DESIGN_CHOICE values)."""

    name: str
    label: str
    unit: str
    kind: str
    default_bounds: DefaultBounds | None
    options: list[int] | None
    round_to: float | None
    effort_weight: float
    note: str
    status: Literal["DESIGN_CHOICE"] = "DESIGN_CHOICE"


class RateDefault(BaseModel):
    """What happens when the founder leaves annual_rate_pct empty."""

    status: Literal["ASSUMPTION"] = "ASSUMPTION"
    rule: str
    spread_over_prime_pct: Quantity


class Limits(BaseModel):
    """Input limits the server enforces (so the form can show them)."""

    max_sites: int = MAX_SITES
    idea_text_max_chars: int = IDEA_MAX_CHARS
    loan_term_months_min: int = LOAN_TERM_MIN_MONTHS
    loan_term_months_max: int = LOAN_TERM_MAX_MONTHS
    annual_rate_pct_max: float = ANNUAL_RATE_MAX_PCT


class OptionsResponse(BaseModel):
    """GET /api/studio/options."""

    metros: list[MetroOption]
    business_types: list[BusinessTypeOption]
    levers: list[LeverOption]
    default_target: Target
    dscr_target: DscrTarget
    annual_rate_default: RateDefault
    limits: Limits = Field(default_factory=Limits)
    disclaimer: str
    simulation_label: str
    synthetic_label: str


# ---------------------------------------------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------------------------------------------


class SiteIn(RequestModel):
    """One candidate site: a map pin (lat, lon) or, for a real metro only, a one-line US address."""

    label: str = Field(min_length=1, max_length=80)
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    address: str | None = Field(default=None, min_length=5, max_length=200)

    @model_validator(mode="after")
    def pin_or_address(self) -> "SiteIn":
        """Exactly one of: both lat and lon, or an address."""
        has_pin = self.lat is not None and self.lon is not None
        if (self.lat is None) != (self.lon is None):
            raise ValueError("give both lat and lon for a map pin")
        if has_pin == (self.address is not None):
            raise ValueError("give either a map pin (lat and lon) or an address, not both and not neither")
        return self


class ProjectIn(RequestModel):
    """POST /api/studio/projects. Money in USD. Empty optional fields fall back to labeled ASSUMPTIONs."""

    name: str | None = Field(default=None, max_length=120)
    metro_id: str = Field(pattern=STUDIO_ID_PATTERN)
    business_type_id: str = Field(pattern=STUDIO_ID_PATTERN)
    idea_text: str | None = Field(default=None, max_length=IDEA_MAX_CHARS)
    budget_total: float | None = Field(
        default=None,
        gt=0,
        le=MAX_MONEY,
        description="Build-out and equipment budget (capex). Empty = the profile's capex range (ASSUMPTION). "
        "owner_equity + loan_amount - budget_total is the cash reserve at opening and may not be negative",
    )
    owner_equity: float = Field(ge=0, le=MAX_MONEY)
    loan_amount: float = Field(gt=0, le=MAX_MONEY)
    loan_term_months: int = Field(ge=LOAN_TERM_MIN_MONTHS, le=LOAN_TERM_MAX_MONTHS)
    annual_rate_pct: float | None = Field(
        default=None,
        ge=0,
        le=ANNUAL_RATE_MAX_PCT,
        description="The lender's quoted rate. Empty = the metro's prime rate + the spread ASSUMPTION",
    )
    rate_type: RateType = Field(default="variable", description="The simulator's default (prime-based pricing)")
    opening_date: date | None = Field(
        default=None, description="Empty = the first day of the month six months after the plan was created"
    )
    sites: list[SiteIn] = Field(min_length=1, max_length=MAX_SITES)


class ProjectUpdate(RequestModel):
    """PUT /api/studio/projects/{id}: any subset of ProjectIn's fields. Omitted = unchanged; null clears an
    optional field (budget_total, annual_rate_pct, idea_text, name, opening_date). `version`, when given, must
    equal the stored version (otherwise 409), so two tabs cannot overwrite each other silently."""

    version: int | None = Field(default=None, ge=1)
    name: str | None = Field(default=None, max_length=120)
    metro_id: str | None = Field(default=None, pattern=STUDIO_ID_PATTERN)
    business_type_id: str | None = Field(default=None, pattern=STUDIO_ID_PATTERN)
    idea_text: str | None = Field(default=None, max_length=IDEA_MAX_CHARS)
    budget_total: float | None = Field(default=None, gt=0, le=MAX_MONEY)
    owner_equity: float | None = Field(default=None, ge=0, le=MAX_MONEY)
    loan_amount: float | None = Field(default=None, gt=0, le=MAX_MONEY)
    loan_term_months: int | None = Field(default=None, ge=LOAN_TERM_MIN_MONTHS, le=LOAN_TERM_MAX_MONTHS)
    annual_rate_pct: float | None = Field(default=None, ge=0, le=ANNUAL_RATE_MAX_PCT)
    rate_type: RateType | None = None
    opening_date: date | None = None
    sites: list[SiteIn] | None = Field(default=None, min_length=1, max_length=MAX_SITES)


class SiteOut(BaseModel):
    """A stored candidate site."""

    index: int
    label: str
    lat: float
    lon: float
    location_source: Literal["map_pin", "census_geocoder"]
    address: str | None
    matched_address: str | None
    tract_geoid: str | None
    location_check: str = Field(description="How the site was checked against the metro, in plain words")
    geocode_source: SourceRecord | None = None


class IdeaOut(BaseModel):
    """The founder's idea text and the injection screen's verdict."""

    text: str | None
    flagged: bool
    flags: list[str]
    note: str


class MoneyOut(BaseModel):
    """The plan's money, each figure with whether the founder typed it."""

    budget_total: float | None
    budget_status: EntryStatus
    owner_equity: float
    loan_amount: float
    loan_term_months: int
    annual_rate_pct: float | None
    annual_rate_status: EntryStatus
    rate_type: RateType
    cash_reserve_at_opening: float | None = Field(description="owner_equity + loan_amount - budget_total")
    note: str


class ProjectOut(BaseModel):
    """One plan (GET/POST/PUT /api/studio/projects...)."""

    id: str
    version: int
    name: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    retention_days: int | None
    metro_id: str
    metro_name: str
    metro_mode: MetroModeName
    data_label: DataLabelName
    business_type_id: str
    business_type_name: str
    demand_unit: str
    idea: IdeaOut
    money: MoneyOut
    opening_date: date
    opening_date_status: EntryStatus
    sites: list[SiteOut]
    disclaimer: str


class ProjectSummary(BaseModel):
    """One row of the plan list."""

    id: str
    version: int
    name: str | None
    metro_id: str
    metro_name: str
    business_type_id: str
    business_type_name: str
    n_sites: int
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    idea_flagged: bool


class ProjectList(BaseModel):
    """GET /api/studio/projects (newest first)."""

    projects: list[ProjectSummary]
    total: int


class ProjectDeleted(BaseModel):
    """DELETE /api/studio/projects/{id}."""

    project_id: str
    deleted: bool
    pending: bool = Field(description="True: the remote copy could not be deleted yet; it is retried")
    message: str


# ---------------------------------------------------------------------------------------------------------------
# Site scores
# ---------------------------------------------------------------------------------------------------------------


class FactorOut(BaseModel):
    """One location factor of one site."""

    name: str
    description: str
    raw: float | None
    unit: str
    status: str = Field(description="DATA, SYNTHETIC, ASSUMPTION, SOURCED, UNKNOWN or FOUNDER_INPUT")
    score: float | None = Field(description="0-100 within the metro; null when not scored")
    percentile: float | None
    weight: float
    direction: str
    used_in_overall: bool
    area: str
    reason: str
    normalization: str
    reference_size: int
    source: str | None = Field(description="One line naming the first source (publisher, dataset, date)")
    sources: list[SourceRecord]


class DemandOut(BaseModel):
    """The site's demand estimate in the profile's unit (a MODEL_ESTIMATE; null values mean UNKNOWN)."""

    status: Literal["MODEL_ESTIMATE"] = "MODEL_ESTIMATE"
    unit: str
    low: float | None
    p10: float | None
    p50: float | None
    mode: float | None
    p90: float | None
    high: float | None
    basis: str
    complete: bool
    missing: list[str]
    method: str
    formula_description: str
    area: str
    capacity_note: str | None
    inputs: list[DemandInput]


class SiteScoreOut(BaseModel):
    """SiteIQ's result for one site (error is set instead when the site could not be scored)."""

    index: int
    label: str
    lat: float
    lon: float
    tract_geoid: str | None = None
    district: str | None = None
    overall_score: float | None = None
    coverage_score: float | None = None
    scored_weight_share: float | None = None
    top_reasons: list[str] = Field(default_factory=list)
    watch_out: str | None = None
    factors: list[FactorOut] = Field(default_factory=list)
    demand: DemandOut | None = None
    metro_inputs: dict[str, FactorValue] = Field(default_factory=dict)
    context: list[ContextItem] = Field(default_factory=list)
    trade_area: TradeAreaInfo | None = None
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


class SiteScoresResponse(BaseModel):
    """POST /api/studio/projects/{id}/sites/score."""

    project_id: str
    project_version: int
    metro_id: str
    metro_name: str
    data_label: DataLabelName
    mode: MetroModeName
    mode_reason: str
    scoring_method: str
    sites: list[SiteScoreOut]
    disclaimer: str


# ---------------------------------------------------------------------------------------------------------------
# Feasibility and path to approval
# ---------------------------------------------------------------------------------------------------------------


class SiteChoice(RequestModel):
    """Which of the plan's sites to use (0-based)."""

    site_index: int = Field(default=0, ge=0, lt=MAX_SITES)


class FeasibilityRequest(SiteChoice):
    """POST /api/studio/projects/{id}/feasibility."""


class Headline(BaseModel):
    """The four headline probabilities (simulated, based on stated assumptions) for the gauges."""

    label: str
    dscr_target: float
    p_dscr_year1_at_least_target: float | None
    p_dscr_year2_at_least_target: float | None
    stress_min_dscr: float | None
    stress_revenue_drop_share: float | None
    p_stressed_dscr_year2_at_least_min: float | None
    p_cash_negative_within_24_months: float
    break_even_p50_month: int | None


class FeasibilityResponse(BaseModel):
    """The simulation of one site of a plan (n = 1,000 paths, seed fixed per plan version)."""

    project_id: str
    project_version: int
    site_index: int
    site_label: str
    seed: int
    label: str
    disclaimer: str
    headline: Headline
    result: FeasibilityResult


class TargetsIn(RequestModel):
    """Edits to the default targets of the path search (omitted = default)."""

    dscr_min: float | None = Field(default=None, gt=0, le=5)
    p_dscr_min: float | None = Field(default=None, ge=0, le=1)
    stress_revenue_share: float | None = Field(default=None, ge=0, lt=1)
    stressed_dscr_min: float | None = Field(default=None, gt=0, le=5)
    p_stressed_min: float | None = Field(default=None, ge=0, le=1)
    p_cash_negative_max: float | None = Field(default=None, ge=0, le=1)
    cash_months: int | None = Field(default=None, ge=1, le=36)


class PathRequest(SiteChoice):
    """POST /api/studio/projects/{id}/path-to-approval."""

    targets: TargetsIn | None = None


class ProjectMoneyPatch(BaseModel):
    """Plan fields after a change-set, ready to PUT to /api/studio/projects/{id}."""

    owner_equity: float
    loan_amount: float
    loan_term_months: int
    budget_total: float | None


class ApplyPatch(BaseModel):
    """What "Apply this change" sends: PUT project_fields, then re-run at site_index."""

    supported: bool
    reason: str | None = None
    site_index: int
    project_fields: ProjectMoneyPatch


class ChangeSetOut(BaseModel):
    """A change-set of the path search with before/after probabilities and how to apply it."""

    rank: int
    changes: list[AppliedChange]
    n_levers: int
    effort: float
    before: Outcome
    after: Outcome
    plan_before: PlanAmounts
    plan_after: PlanAmounts
    meets_target: bool
    explanation: str
    unchanged: str
    apply: ApplyPatch


class PathResponse(BaseModel):
    """POST /api/studio/projects/{id}/path-to-approval (QUICK mode)."""

    project_id: str
    project_version: int
    site_index: int
    site_label: str
    label: str
    disclaimer: str
    status: str
    message: str
    guardrail_note: str
    guardrails: list[str]
    target: Target
    before: Outcome
    plan_before: PlanAmounts
    change_sets: list[ChangeSetOut]
    closest: ChangeSetOut | None
    levers_considered: list[LeverSummary]
    levers_not_available: dict[str, str]
    skipped_candidates: list[str]
    notes: list[str]
    settings: SearchSettings
    search: SearchStats
    metro_mode: str
    data_label: str
    engine_version: str
    simulator_version: str


class AreasRequest(RequestModel):
    """POST /api/studio/areas/suggest."""

    metro_id: str = Field(pattern=STUDIO_ID_PATTERN)
    business_type_id: str = Field(pattern=STUDIO_ID_PATTERN)
