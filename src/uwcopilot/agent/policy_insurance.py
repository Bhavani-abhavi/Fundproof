"""Harborline Mutual rules (corpus/guidelines/harborline_mutual_commercial.md, SYNTHETIC) as plain code.

Each check names the guideline sections it implements; the formulas are in insurance_calculations.py. Rules that
need data this phase does not have (state registry lookups, adverse media, document metadata, catastrophe models,
the quoted premium) are listed as not evaluated.

Interpretation choices (also in docs/agent_design.md):
- The loss ratio is computed over the covered policy years of the experience period. Years before the business
  existed, or without coverage, are "uncovered years" and must be explained in writing (HM-3.2).
- A loss-run claim where paid plus reserve differs from incurred by more than $1.00 is treated as a document
  integrity indicator (HM-11.1, SIU), as well as a reconciliation difference (HM-3.3).
- A sanctions potential match is a blocking Compliance referral (HM-8.2). HM-2.2 also lists an *unresolved* match
  as a prohibited class; that can only be decided after Compliance has dispositioned it.
"""

from uwcopilot.agent.applications import InsuranceApplication
from uwcopilot.agent.insurance_calculations import (
    LARGE_LOSS_USD,
    all_claims,
    building_cost_per_sqft,
    claim_out_of_balance,
    claims_trend,
    incurred_difference_pct,
    location_tiv,
    loss_ratios,
    loss_run_age_days,
    max_claims_in_rolling_window,
    schedule_tiv,
    tiv_difference_pct,
)
from uwcopilot.agent.policy import PolicyInputs, RuleContext, money, pct
from uwcopilot.agent.vocab import (
    CATASTROPHE_MANAGEMENT,
    CHIEF_UNDERWRITING_OFFICER,
    SENIOR_UNDERWRITER,
    SPECIAL_INVESTIGATIONS,
    UNDERWRITING_MANAGER,
)

# HM-1.1 capacity.
MAX_ACCOUNT_TIV = 25_000_000
MAX_LOCATION_TIV = 10_000_000
STANDARD_GL_LIMITS = (1_000_000, 2_000_000)
MAX_GL_LIMITS = (2_000_000, 4_000_000)
# HM-2.2 and HM-2.3.
PROHIBITED_CLASSES = {
    "fireworks_storage",
    "ammunition_manufacturing",
    "cannabis_dispensary",
    "crypto_mining",
    "demolition_contractor",
    "large_nightclub",
    "waste_recycling",
    "salvage_yard",
}
RESTRICTED_CLASSES = {
    "woodworking",
    "furniture_manufacturing",
    "plastics_manufacturing",
    "auto_repair",
    "body_shop",
    "fuel_station",
}
MAX_VACANT_DAYS = 60
MAX_HABITATIONAL_STORIES = 4
MAX_WAREHOUSE_SQFT = 100_000
MAX_FRAME_BUILDING_VALUE = 5_000_000
# HM-2.1 eligibility.
MIN_YEARS_EXPERIENCE = 3
OLD_BUILDING_BEFORE_YEAR = 1950
SYSTEMS_UPDATE_MAX_AGE = 25
MAX_ROOF_AGE = 20
POOR_PROTECTION_CLASSES = {9, 10}
# HM-3.1 documents.
INSPECTION_REPORT_ABOVE_TIV = 5_000_000
# HM-3.2 loss runs.
MAX_LOSS_RUN_AGE_DAYS = 90
MIN_LOSS_HISTORY_YEARS = 3
# HM-3.3 reconciliation.
MAX_INCURRED_DIFFERENCE_PCT = 10.0
# HM-4.2 loss ratios.
SENIOR_LOSS_RATIO_PCT = 55.0
MANAGER_LOSS_RATIO_PCT = 70.0
MAX_ANNUAL_LOSS_RATIO_PCT = 100.0
# HM-4.3 frequency.
MAX_PROPERTY_CLAIMS_12_MONTHS = 2
MAX_GL_CLAIMS = 3
# HM-5.2 TIV and underinsurance.
TIV_CORRECTION_PCT = 5.0
TIV_REFER_PCT = 15.0
MIN_COST_PER_SQFT: dict[str, float] = {
    "frame": 150,
    "joisted_masonry": 175,
    "non_combustible": 160,
    "masonry_non_combustible": 190,
    "fire_resistive": 225,
}
# HM-5.3 and HM-6.
SPRINKLER_REQUIRED_ABOVE_TIV = 2_500_000
TIER1_REFER_ABOVE_TIV = 5_000_000
HIGH_WILDFIRE = {"high", "very_high"}
# HM-7.1.
CONTRACTOR_CLASSES = {"general_contractor", "demolition_contractor"}
MAX_SUBCONTRACTED_PCT = 50.0

NOT_IMPLEMENTED: tuple[tuple[str, list[str], str], ...] = (
    ("named_insured_verification", ["HM-8.1"], "needs a state business registry lookup (not available offline)"),
    ("adverse_media", ["HM-8.3"], "no adverse-media input in the structured submission"),
    ("other_document_integrity", ["HM-11.1"], "needs document metadata and versions (document extraction, Phase 2)"),
    ("premium_deviation", ["HM-10.1"], "the submission has no quoted premium to compare with the model indication"),
    ("tier1_county_aggregate", ["HM-6.1"], "needs Harborline's portfolio aggregate by county"),
    ("earthquake_and_pml", ["HM-6.2"], "needs earthquake zones and a catastrophe model"),
    ("large_loss_descriptions", ["HM-3.3"], "needs the written cause-of-loss descriptions (documents)"),
    ("underwriting_authority", ["HM-9.2"], "checked at the human approval checkpoint, not by the policy"),
)


def coverage_lines(app: InsuranceApplication) -> list[str]:
    """Lines whose loss ratio is evaluated: the ones being quoted."""
    if app.coverage == "package":
        return ["property", "general_liability"]
    return [app.coverage]


def has_gl(app: InsuranceApplication) -> bool:
    """True when general liability is part of the request."""
    return app.coverage in ("general_liability", "package")


def check_capacity(app: InsuranceApplication, account_tiv: float, tiv_id: str, context: RuleContext) -> None:
    """HM-1.1: account TIV up to $25M, $10M per location, GL up to $2M / $4M; above that goes to the CUO."""
    problems = []
    if account_tiv > MAX_ACCOUNT_TIV:
        problems.append(f"account TIV {money(account_tiv)} is above {money(MAX_ACCOUNT_TIV)}")
    for location in app.locations:
        value = location_tiv(location)
        if value > MAX_LOCATION_TIV:
            problems.append(f"location {location.location_id} TIV {money(value)} is above {money(MAX_LOCATION_TIV)}")
    occurrence, aggregate = app.gl_limit_per_occurrence or 0, app.gl_limit_aggregate or 0
    if has_gl(app) and (occurrence > MAX_GL_LIMITS[0] or aggregate > MAX_GL_LIMITS[1]):
        problems.append(
            f"GL limits {money(occurrence)} / {money(aggregate)} are above {money(MAX_GL_LIMITS[0])} / "
            f"{money(MAX_GL_LIMITS[1])}"
        )
    if problems:
        context.find(
            "above_capacity",
            "refer",
            ["HM-1.1", "HM-9.3"],
            [CHIEF_UNDERWRITING_OFFICER],
            "Request is above capacity: " + "; ".join(problems) + ".",
            [tiv_id, "APP-locations", "APP-gl_limits"],
            trigger_ids=["R-HM-17"],
        )
    else:
        context.passed(
            "capacity", ["HM-1.1"], f"Account TIV {money(account_tiv)} and the limits are within capacity.", [tiv_id]
        )


def check_class(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-2.2 prohibited classes (decline-for-review) and HM-2.3 restricted classes (refer)."""
    code = app.applicant.class_code
    vacant = [location.location_id for location in app.locations if location.vacant_days > MAX_VACANT_DAYS]
    if code in PROHIBITED_CLASSES or vacant:
        reason = (
            f"Class code {code} is a prohibited class"
            if code in PROHIBITED_CLASSES
            else (f"Location(s) {', '.join(vacant)} vacant for more than 60 consecutive days")
        )
        context.find(
            "prohibited_class",
            "decline_for_review",
            ["HM-2.2", "HM-9.1"],
            [SENIOR_UNDERWRITER],
            f"{reason}; it is not quoted and no exception may be granted.",
            ["APP-class_code", "APP-locations"],
        )
        return
    reasons = []
    if code in RESTRICTED_CLASSES:
        reasons.append(f"class code {code} is restricted")
    if code == "restaurant_fryer" and not app.applicant.fryer_suppression_inspected_within_6_months:
        reasons.append("deep-fat frying without a suppression system inspected within 6 months")
    for location in app.locations:
        if code == "habitational" and location.stories > MAX_HABITATIONAL_STORIES:
            reasons.append(f"location {location.location_id} is a habitational building of {location.stories} stories")
        if code == "warehouse" and location.square_feet > MAX_WAREHOUSE_SQFT:
            reasons.append(f"location {location.location_id} is a warehouse of {location.square_feet:,} square feet")
        if location.construction_class == "frame" and location.building_value > MAX_FRAME_BUILDING_VALUE:
            reasons.append(
                f"location {location.location_id} is frame construction with a building value above "
                f"{money(MAX_FRAME_BUILDING_VALUE)}"
            )
    if reasons:
        context.find(
            "restricted_class",
            "refer",
            ["HM-2.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            "Restricted class: " + "; ".join(reasons) + ".",
            ["APP-class_code", "APP-locations"],
            trigger_ids=["R-HM-03"],
        )
    else:
        context.passed(
            "class_eligible",
            ["HM-2.2", "HM-2.3"],
            f"Class code {code} is neither prohibited nor restricted.",
            ["APP-class_code"],
        )


def check_eligibility(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-2.1 items 1 to 5 (items 3 and 4 are also HM-5.3 triggers)."""
    effective_year = app.proposed_effective_date.year
    applicant = app.applicant
    if applicant.years_in_business < MIN_YEARS_EXPERIENCE and applicant.prior_experience_years < MIN_YEARS_EXPERIENCE:
        context.find(
            "eligibility_experience",
            "refer",
            ["HM-2.1", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"{applicant.years_in_business} years in business and {applicant.prior_experience_years} years of prior "
            "experience (at least 3 years needed).",
            ["APP-years_in_business", "APP-prior_experience_years"],
            trigger_ids=["R-HM-01"],
        )
    old_buildings = [
        location.location_id
        for location in app.locations
        if location.year_built < OLD_BUILDING_BEFORE_YEAR
        and (
            location.systems_updated_year is None
            or effective_year - location.systems_updated_year > SYSTEMS_UPDATE_MAX_AGE
        )
    ]
    if old_buildings:
        context.find(
            "eligibility_building_age",
            "refer",
            ["HM-2.1", "HM-9.3"],
            [UNDERWRITING_MANAGER],
            f"Location(s) {', '.join(old_buildings)} built before 1950 without systems updated in the last 25 years.",
            ["APP-locations"],
            trigger_ids=["R-HM-02"],
        )
    wiring = [location.location_id for location in app.locations if location.knob_and_tube_or_fuse_box]
    if wiring:
        context.find(
            "knob_and_tube_or_fuse_box",
            "refer",
            ["HM-2.1", "HM-9.3"],
            [UNDERWRITING_MANAGER],
            f"Location(s) {', '.join(wiring)} have knob-and-tube wiring or a fuse-box service.",
            ["APP-locations"],
            trigger_ids=["R-HM-02"],
        )
    old_roofs = [
        f"{location.location_id} ({effective_year - location.roof_year} years)"
        for location in app.locations
        if effective_year - location.roof_year > MAX_ROOF_AGE
    ]
    if old_roofs:
        context.find(
            "roof_older_than_20_years",
            "refer",
            ["HM-2.1", "HM-5.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Roof older than 20 years at location(s) {', '.join(old_roofs)}.",
            ["APP-locations"],
            trigger_ids=["R-HM-01", "R-HM-09"],
        )
    poor_protection = [
        f"{location.location_id} (class {location.protection_class})"
        for location in app.locations
        if location.protection_class in POOR_PROTECTION_CLASSES
    ]
    if poor_protection:
        context.find(
            "protection_class_9_or_10",
            "refer",
            ["HM-2.1", "HM-5.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Public protection class 9 or 10 at location(s) {', '.join(poor_protection)}.",
            ["APP-locations"],
            trigger_ids=["R-HM-01", "R-HM-09"],
        )
    if not (old_buildings or wiring or old_roofs or poor_protection):
        context.passed(
            "building_eligibility",
            ["HM-2.1"],
            "Every location meets the building age, wiring, roof (20 years or less) and protection class rules.",
            ["APP-locations"],
        )


def required_documents(app: InsuranceApplication) -> list[str]:
    """HM-3.1 plus the documents HM-5.3 and HM-6.2 add for large or flood-zone locations."""
    required = [
        "signed_application",
        "property_schedule",
        "loss_runs",
        "prior_carrier_history",
        "beneficial_ownership_statement",
    ]
    if has_gl(app):
        required.append("revenue_and_payroll")
    if any(location_tiv(location) > INSPECTION_REPORT_ABOVE_TIV for location in app.locations):
        required.append("inspection_report")
    if any(location.flood_zone.startswith(("A", "V")) for location in app.locations):
        required.append("elevation_certificate")
    return required


def check_documents(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-3.1: a missing required document means request more information (not a decline)."""
    missing = [name for name in required_documents(app) if name not in app.documents_provided]
    if missing:
        context.find(
            "missing_documents",
            "request_more_information",
            ["HM-3.1"],
            [SENIOR_UNDERWRITER],
            f"Required documents are missing: {', '.join(missing)}.",
            ["APP-documents_provided"],
        )
    else:
        context.passed(
            "documents_complete",
            ["HM-3.1"],
            "Every required document is in the submission.",
            ["APP-documents_provided"],
        )


def check_loss_run_quality(app: InsuranceApplication, context: RuleContext) -> bool:
    """HM-3.2. Returns True when the loss runs can be used for loss ratios and frequency checks."""
    runs = app.loss_runs
    usable = True
    if not runs.carrier_produced:
        context.find(
            "loss_runs_not_carrier_produced",
            "request_more_information",
            ["HM-3.2"],
            [SENIOR_UNDERWRITER],
            "Loss runs were not produced by the carrier; carrier loss runs are required.",
            ["APP-loss_runs"],
        )
        usable = False
    age = loss_run_age_days(runs, app.proposed_effective_date)
    age_id = context.calc(
        "loss_run_age_days",
        "Days from loss run valuation to proposed effective date",
        age,
        "days",
        ["HM-3.2"],
        valued_as_of=runs.valued_as_of.isoformat(),
        proposed_effective_date=app.proposed_effective_date.isoformat(),
    )
    if age > MAX_LOSS_RUN_AGE_DAYS:
        context.find(
            "loss_runs_out_of_date",
            "request_more_information",
            ["HM-3.2"],
            [SENIOR_UNDERWRITER],
            f"Loss runs are valued {age} days before the proposed effective date (more than 90).",
            [age_id],
        )
        usable = False
    if runs.uncovered_years and not runs.uncovered_years_explained:
        context.find(
            "uncovered_years_unexplained",
            "request_more_information",
            ["HM-3.2"],
            [SENIOR_UNDERWRITER],
            f"{runs.uncovered_years} year(s) without coverage are not explained in writing.",
            ["APP-loss_runs"],
        )
    covered = len(runs.years)
    if covered < MIN_LOSS_HISTORY_YEARS:
        context.find(
            "loss_history_under_3_years",
            "refer",
            ["HM-3.2", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Only {covered} year(s) of loss history (fewer than 3).",
            ["APP-loss_runs"],
            trigger_ids=["R-HM-04"],
        )
    if usable and runs.carrier_produced and age <= MAX_LOSS_RUN_AGE_DAYS:
        context.passed(
            "loss_run_quality",
            ["HM-3.2"],
            f"Carrier loss runs valued {age} days before the effective date (90 or fewer).",
            [age_id],
        )
    return usable


def check_reconciliation(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-3.3 and HM-11.1: claim lines must balance; application totals must match the loss runs."""
    claims = all_claims(app.loss_runs)
    unbalanced = [claim for claim in claims if claim_out_of_balance(claim)]
    if unbalanced:
        unbalanced_id = context.calc(
            "unbalanced_claim_lines",
            "Loss run claim lines where paid + reserve differs from incurred by more than $1.00",
            float(len(unbalanced)),
            "count",
            ["HM-3.3", "HM-11.1"],
        )
        context.find(
            "loss_run_integrity",
            "refer",
            ["HM-3.3", "HM-11.1", "HM-9.3"],
            [SPECIAL_INVESTIGATIONS],
            f"{len(unbalanced)} loss run claim line(s) where paid plus reserve differs from incurred by more than "
            "$1.00.",
            [unbalanced_id, "APP-loss_runs"],
            trigger_ids=["R-HM-16"],
            blocking=True,
        )
    loss_run_total = round(sum(claim.incurred for claim in claims), 2)
    difference = incurred_difference_pct(app.stated_total_incurred, loss_run_total)
    difference_id = context.calc(
        "incurred_difference_pct",
        "Difference between stated and loss-run total incurred (percent)",
        difference,
        "pct",
        ["HM-3.3"],
        stated_total_incurred=app.stated_total_incurred,
        loss_run_total_incurred=loss_run_total,
        stated_claim_count=app.stated_claim_count,
        loss_run_claim_count=len(claims),
    )
    count_differs = app.stated_claim_count != len(claims)
    total_differs = (difference is None and app.stated_total_incurred > 0) or (
        difference is not None and difference > MAX_INCURRED_DIFFERENCE_PCT
    )
    if count_differs or total_differs:
        context.find(
            "loss_run_reconciliation",
            "refer",
            ["HM-3.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Application shows {app.stated_claim_count} claims and {money(app.stated_total_incurred)} incurred; the "
            f"loss runs show {len(claims)} claims and {money(loss_run_total)}.",
            [difference_id],
            trigger_ids=["R-HM-05"],
        )
    else:
        context.passed(
            "loss_run_reconciliation",
            ["HM-3.3"],
            f"Application claim count ({len(claims)}) and incurred total match the loss runs.",
            [difference_id],
        )


def check_loss_ratios(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-4.1 / HM-4.2 for every line being quoted."""
    for line in coverage_lines(app):
        summary = loss_ratios(app.loss_runs, line)
        short = "property" if line == "property" else "general_liability"
        if summary.missing_premium_years:
            context.find(
                "earned_premium_missing",
                "request_more_information",
                ["HM-4.1"],
                [SENIOR_UNDERWRITER],
                f"Earned premium for {line.replace('_', ' ')} is missing for policy year(s) starting "
                f"{', '.join(summary.missing_premium_years)}.",
                ["APP-loss_runs"],
            )
            continue
        if summary.five_year_pct is None:
            context.skip(f"loss_ratio_{short}", ["HM-4.1"], "no covered policy years with earned premium")
            continue
        five_id = context.calc(
            f"loss_ratio_5yr_{short}",
            f"Experience-period loss ratio, {line.replace('_', ' ')} (percent)",
            summary.five_year_pct,
            "pct",
            ["HM-4.1"],
            covered_years=len(summary.annual_pct),
        )
        worst = max(value for value in summary.annual_pct if value is not None)
        worst_id = context.calc(
            f"max_annual_loss_ratio_{short}",
            f"Highest annual loss ratio, {line.replace('_', ' ')} (percent)",
            worst,
            "pct",
            ["HM-4.2"],
        )
        line_text = line.replace("_", " ")
        if summary.five_year_pct > MANAGER_LOSS_RATIO_PCT:
            context.find(
                "loss_ratio_above_70",
                "refer",
                ["HM-4.2", "HM-9.3"],
                [UNDERWRITING_MANAGER],
                f"The {line_text} loss ratio is {pct(summary.five_year_pct)} (more than 70.0%).",
                [five_id],
                trigger_ids=["R-HM-06"],
            )
        elif summary.five_year_pct > SENIOR_LOSS_RATIO_PCT:
            context.find(
                "loss_ratio_above_55",
                "refer",
                ["HM-4.2", "HM-9.3"],
                [SENIOR_UNDERWRITER],
                f"The {line_text} loss ratio is {pct(summary.five_year_pct)} (more than 55.0% and up to 70.0%).",
                [five_id],
                trigger_ids=["R-HM-06"],
            )
        else:
            context.passed(
                "loss_ratio",
                ["HM-4.2"],
                f"The {line_text} loss ratio is {pct(summary.five_year_pct)} (55.0% or less).",
                [five_id],
            )
        if worst > MAX_ANNUAL_LOSS_RATIO_PCT:
            context.find(
                "annual_loss_ratio_above_100",
                "refer",
                ["HM-4.2", "HM-9.3"],
                [SENIOR_UNDERWRITER],
                f"The highest annual {line_text} loss ratio is {pct(worst)} (above 100.0%).",
                [worst_id],
                trigger_ids=["R-HM-06"],
            )


def check_frequency(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-4.3: property frequency, GL frequency, large losses and a rising trend."""
    runs = app.loss_runs
    claims = all_claims(runs)
    experience_claims = [claim for year in runs.years for claim in year.claims]
    property_peak = max_claims_in_rolling_window(claims, "property")
    gl_count = sum(1 for claim in experience_claims if claim.line == "general_liability")
    large = [claim for claim in claims if claim.incurred >= LARGE_LOSS_USD]
    recent, earlier = claims_trend(runs)
    frequency_id = context.calc(
        "claim_frequency",
        "Claim frequency measures",
        float(property_peak),
        "count",
        ["HM-4.3"],
        max_property_claims_in_12_months=property_peak,
        gl_claims_in_experience_period=gl_count,
        large_losses=len(large),
        claims_last_2_years=recent,
        claims_prior_3_years=earlier,
    )
    problems = []
    if property_peak > MAX_PROPERTY_CLAIMS_12_MONTHS:
        problems.append(("property_claim_frequency", f"{property_peak} property claims within 12 months (3 or more)"))
    if gl_count > MAX_GL_CLAIMS:
        problems.append(
            ("gl_claim_frequency", f"{gl_count} general liability claims in the experience period (4 or more)")
        )
    if large:
        largest = max(claim.incurred for claim in large)
        problems.append(("large_loss", f"a large loss of {money(largest)} incurred ($250,000 or more)"))
    if recent > earlier:
        problems.append(
            ("rising_claim_trend", f"{recent} claims in the 2 most recent years versus {earlier} in the 3 years before")
        )
    for rule, text in problems:
        evidence = [frequency_id] if rule != "large_loss" else [frequency_id, "APP-loss_runs"]
        context.find(
            rule,
            "refer",
            ["HM-4.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            text[0].upper() + text[1:] + ".",
            evidence,
            trigger_ids=["R-HM-07"],
        )
    if not problems:
        context.passed(
            "claim_frequency",
            ["HM-4.3"],
            f"{property_peak} property claims in any 12 months, {gl_count} GL claims, no large loss, and no "
            "rising trend.",
            [frequency_id],
        )


def check_values(app: InsuranceApplication, account_tiv: float, tiv_id: str, context: RuleContext) -> None:
    """HM-5.2 TIV reconciliation and underinsurance; HM-5.3 sprinklers."""
    difference = tiv_difference_pct(app.stated_tiv, account_tiv)
    difference_id = context.calc(
        "tiv_difference_pct",
        "Difference between application TIV and schedule TIV (percent)",
        difference,
        "pct",
        ["HM-5.2"],
        application_tiv=app.stated_tiv,
        schedule_tiv=account_tiv,
    )
    if difference > TIV_REFER_PCT:
        context.find(
            "tiv_difference_above_15",
            "refer",
            ["HM-5.2", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Application TIV differs from the schedule TIV by {pct(difference)} (more than 15.0%).",
            [difference_id],
            trigger_ids=["R-HM-08"],
        )
    elif difference > TIV_CORRECTION_PCT:
        context.find(
            "tiv_difference_correction",
            "request_more_information",
            ["HM-5.2"],
            [SENIOR_UNDERWRITER],
            f"Application TIV differs from the schedule TIV by {pct(difference)} (more than 5.0% and up to 15.0%): "
            "request a corrected application or schedule.",
            [difference_id],
        )
    else:
        context.passed(
            "tiv_reconciliation",
            ["HM-5.2"],
            f"Application TIV is within {pct(difference)} of the schedule TIV (5.0% or less).",
            [difference_id, tiv_id],
        )
    low_values = []
    for location in app.locations:
        cost = building_cost_per_sqft(location)
        minimum = MIN_COST_PER_SQFT[location.construction_class]
        if location.building_value > 0 and cost < minimum:
            low_values.append(f"{location.location_id} at {money(cost)} per square foot (minimum {money(minimum)})")
    if low_values:
        context.find(
            "possible_underinsurance",
            "request_more_information",
            ["HM-5.2"],
            [SENIOR_UNDERWRITER],
            "Possible underinsurance; an independent valuation is needed: " + "; ".join(low_values) + ".",
            ["APP-locations"],
        )
    unsprinklered = [
        f"{location.location_id} ({money(location_tiv(location))})"
        for location in app.locations
        if not location.sprinklered and location_tiv(location) > SPRINKLER_REQUIRED_ABOVE_TIV
    ]
    if unsprinklered:
        context.find(
            "unsprinklered_large_location",
            "refer",
            ["HM-5.3", "HM-9.3"],
            [SENIOR_UNDERWRITER],
            f"Unsprinklered location(s) above $2,500,000 TIV: {', '.join(unsprinklered)}.",
            ["APP-locations"],
            trigger_ids=["R-HM-09"],
        )


def check_catastrophe(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-6.1 Tier 1 wind above $5M per location, and HM-6.2 high wildfire hazard (Catastrophe Management)."""
    problems = []
    for location in app.locations:
        value = location_tiv(location)
        if location.wind_tier == 1 and value > TIER1_REFER_ABOVE_TIV:
            problems.append(
                (
                    "wind_tier1_large_location",
                    ["HM-6.1", "HM-9.3"],
                    f"Location {location.location_id} is in wind Tier 1 with TIV {money(value)} (above "
                    f"{money(TIER1_REFER_ABOVE_TIV)}).",
                )
            )
        if location.wildfire_hazard in HIGH_WILDFIRE:
            problems.append(
                (
                    "wildfire_hazard",
                    ["HM-6.2", "HM-9.3"],
                    f"Location {location.location_id} has a {location.wildfire_hazard.replace('_', ' ')} "
                    "wildfire hazard score.",
                )
            )
    for rule, sections, detail in problems:
        context.find(
            rule, "refer", sections, [CATASTROPHE_MANAGEMENT], detail, ["APP-locations"], trigger_ids=["R-HM-10"]
        )
    if not problems:
        context.passed(
            "catastrophe_exposure",
            ["HM-6.1", "HM-6.2"],
            "No wind Tier 1 location above $5,000,000 TIV and no high wildfire hazard.",
            ["APP-locations"],
        )


def check_general_liability(app: InsuranceApplication, context: RuleContext) -> None:
    """HM-7.1: limits above the standard, contractors without subcontractor COI tracking, foreign products."""
    if not has_gl(app):
        return
    problems: list[tuple[str, str, str]] = []
    occurrence, aggregate = app.gl_limit_per_occurrence or 0, app.gl_limit_aggregate or 0
    if (occurrence > STANDARD_GL_LIMITS[0] or aggregate > STANDARD_GL_LIMITS[1]) and (
        occurrence <= MAX_GL_LIMITS[0] and aggregate <= MAX_GL_LIMITS[1]
    ):
        problems.append(
            (
                "gl_limits_above_standard",
                SENIOR_UNDERWRITER,
                f"GL limits {money(occurrence)} / {money(aggregate)} are above the standard "
                f"{money(STANDARD_GL_LIMITS[0])} / {money(STANDARD_GL_LIMITS[1])}.",
            )
        )
    if (
        app.applicant.class_code in CONTRACTOR_CLASSES
        and app.subcontracted_work_pct > MAX_SUBCONTRACTED_PCT
        and not app.has_subcontractor_coi_tracking
    ):
        problems.append(
            (
                "subcontractor_coi_missing",
                SENIOR_UNDERWRITER,
                f"Contractor subcontracts {pct(app.subcontracted_work_pct)} of its work (more than 50%) without "
                "certificate-of-insurance tracking.",
            )
        )
    if app.foreign_products:
        problems.append(
            (
                "foreign_products",
                UNDERWRITING_MANAGER,
                "Products-completed operations exposure from products sold outside the United States.",
            )
        )
    for rule, target, detail in problems:
        context.find(
            rule,
            "refer",
            ["HM-7.1", "HM-9.3"],
            [target],
            detail,
            ["APP-gl_limits", "APP-subcontracting"],
            trigger_ids=["R-HM-11"],
        )
    if not problems:
        context.passed(
            "general_liability",
            ["HM-7.1"],
            f"GL limits {money(occurrence)} / {money(aggregate)} are within the standard limits.",
            ["APP-gl_limits"],
        )


def evaluate_insurance_rules(app: InsuranceApplication, inputs: PolicyInputs, context: RuleContext) -> None:
    """Run every Harborline rule that the structured submission supports."""
    del inputs  # sanctions and injection are checked in policy.py for both lines
    account_tiv = schedule_tiv(app)
    tiv_id = context.calc(
        "schedule_tiv",
        "Schedule TIV (all locations)",
        account_tiv,
        "usd",
        ["HM-1.2", "HM-5.2"],
        locations=len(app.locations),
    )
    check_capacity(app, account_tiv, tiv_id, context)
    check_class(app, context)
    check_eligibility(app, context)
    check_documents(app, context)
    loss_runs_in_file = "loss_runs" in app.documents_provided and bool(app.loss_runs.years)
    if loss_runs_in_file:
        usable = check_loss_run_quality(app, context)
        check_reconciliation(app, context)
        if usable:
            check_loss_ratios(app, context)
            check_frequency(app, context)
        else:
            context.skip(
                "loss_ratios_and_frequency",
                ["HM-4.1", "HM-4.2", "HM-4.3"],
                "loss runs are not usable until current carrier loss runs are provided",
            )
    else:
        context.skip("loss_ratios_and_frequency", ["HM-3.2", "HM-4.1", "HM-4.3"], "no loss runs in the submission")
    check_values(app, account_tiv, tiv_id, context)
    check_catastrophe(app, context)
    check_general_liability(app, context)
    for rule, sections, reason in NOT_IMPLEMENTED:
        context.skip(rule, sections, f"not implemented: {reason}")
