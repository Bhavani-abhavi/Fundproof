"""Northwind Business Credit rules (corpus/guidelines/northwind_business_credit.md, SYNTHETIC) as plain code.

Each check names the guideline sections it implements. Thresholds are copied from the guideline text; the
calculations themselves are in credit_calculations.py. Rules that need documents this phase does not have
(identity documents, formation documents, adverse media, PDF metadata) are listed as not evaluated.
"""

from uwcopilot.agent.applications import CreditApplication
from uwcopilot.agent.credit_calculations import (
    collateral_coverage,
    coverage_ratio,
    existing_annual_debt_service,
    months_between,
    net_operating_income,
    proposed_annual_debt_service,
    revenue_gap_pct,
    round_ratio,
    verify_revenue,
)
from uwcopilot.agent.policy import PolicyInputs, RuleContext, money, pct, ratio
from uwcopilot.agent.vocab import (
    CHIEF_CREDIT_OFFICER,
    CREDIT_COMMITTEE,
    KYC_OPERATIONS,
    SENIOR_UNDERWRITER,
)

# NBC-1.1: product limits (minimum amount, maximum amount, minimum term, maximum term in months).
PRODUCT_LIMITS: dict[str, tuple[float, float, int, int]] = {
    "term_loan": (25_000, 2_000_000, 12, 84),
    "line_of_credit": (25_000, 500_000, 12, 12),
    "equipment_loan": (25_000, 1_500_000, 24, 84),
}
PRODUCT_LABELS: dict[str, str] = {
    "term_loan": "term loan",
    "line_of_credit": "line of credit",
    "equipment_loan": "equipment loan",
}
# NBC-2.1 eligibility minimums and the near-miss floors that can go to the exceptions process.
MIN_TIME_IN_BUSINESS_MONTHS = 24
NEAR_MISS_TIME_IN_BUSINESS_MONTHS = 18
MIN_VERIFIED_REVENUE = 250_000
NEAR_MISS_VERIFIED_REVENUE = 225_000
MIN_CREDIT_SCORE = 660
NEAR_MISS_CREDIT_SCORE = 640
PRINCIPAL_OWNERSHIP_PCT = 20.0
ELIGIBLE_STRUCTURES = {"corporation", "llc", "partnership"}
# NBC-2.2 and NBC-2.3.
PROHIBITED_INDUSTRIES = {
    "cannabis_retail",
    "adult_entertainment",
    "casino_gambling",
    "firearms_manufacturing",
    "payday_lending",
    "virtual_currency_exchange",
    "multi_level_marketing",
    "speculative_real_estate",
    "political_lobbying",
    "shell_company",
}
RESTRICTED_INDUSTRIES = {
    "long_haul_trucking",
    "freight_broker",
    "construction_contractor",
    "used_vehicle_dealer",
    "money_services_business",
}
RESTAURANT_MIN_MONTHS = 36
MAX_SINGLE_CUSTOMER_PCT = 30.0
# NBC-3.1 required documents, and the conditional ones.
BASE_DOCUMENTS = (
    "signed_application",
    "bank_statements_12_months",
    "business_tax_returns_2_years",
    "ytd_pnl_and_balance_sheet",
    "debt_schedule",
    "principal_photo_ids",
    "formation_documents",
    "beneficial_ownership_certification",
)
CPA_FINANCIALS_ABOVE = 500_000
# NBC-3.4 revenue gap bands (percent).
BENIGN_GAP_PCT = 10.0
EXPLAINABLE_GAP_PCT = 25.0
# NBC-3.5 account conduct.
MAX_NSF_EVENTS = 6
MAX_NEGATIVE_MONTHS = 2
MIN_BALANCE_SHARE_OF_DEBT_SERVICE = 0.05
# NBC-4.2 DSCR minimums and the decline floor.
DSCR_MINIMUM = {"term_loan": 1.25, "equipment_loan": 1.25, "line_of_credit": 1.35}
DSCR_DECLINE_FLOOR = 1.10
# NBC-4.4 leverage and stacking.
MAX_LEVERAGE = 4.00
STACKING_REFER = 2
STACKING_CCO = 3
# NBC-4.5 stress tests.
STRESS_AMOUNT_ABOVE = 100_000
REVENUE_STRESS_SHARE = 0.20
REVENUE_STRESS_MIN_DSCR = 1.00
RATE_STRESS_ADD_PCT = 2.00
RATE_STRESS_MIN_DSCR = 1.10
# NBC-5.1 collateral.
COLLATERAL_REQUIRED_ABOVE = 250_000
MIN_COLLATERAL_COVERAGE = 1.00
COMMITTEE_COLLATERAL_COVERAGE = 0.80
OWNERSHIP_TOTAL_TOLERANCE_PCT = 0.5

NOT_IMPLEMENTED: tuple[tuple[str, list[str], str], ...] = (
    ("identity_verification", ["NBC-6.2"], "needs ID documents and credit reports (document extraction, Phase 2)"),
    ("adverse_media_and_pep", ["NBC-6.4"], "no adverse-media or PEP input in the structured application"),
    ("document_integrity", ["NBC-9.1"], "needs statement PDFs and metadata (document extraction, Phase 2)"),
    (
        "ownership_matches_formation_documents",
        ["NBC-6.1"],
        "needs formation documents; only the application's own ownership total is checked",
    ),
    ("compensating_factors", ["NBC-4.3"], "an exception decision for the human approver, not a policy calculation"),
    ("approval_authority", ["NBC-7.2"], "checked at the human approval checkpoint, not by the policy"),
)


def principals(application: CreditApplication) -> list:
    """NBC-1.2: individuals owning 20% or more, or the control person."""
    return [
        owner
        for owner in application.owners
        if owner.entity_type == "person" and (owner.ownership_pct >= PRINCIPAL_OWNERSHIP_PCT or owner.is_control_person)
    ]


def check_product_limits(app: CreditApplication, context: RuleContext) -> None:
    """NBC-1.1: amount and term within the product's limits, otherwise refer to the Chief Credit Officer."""
    minimum, maximum, min_term, max_term = PRODUCT_LIMITS[app.product]
    label = PRODUCT_LABELS[app.product]
    evidence = ["APP-requested_amount", "APP-term_months", "APP-product"]
    problems = []
    if app.requested_amount < minimum:
        problems.append(f"requested amount {money(app.requested_amount)} is below the {label} minimum {money(minimum)}")
    if app.requested_amount > maximum:
        problems.append(f"requested amount {money(app.requested_amount)} is above the {label} maximum {money(maximum)}")
    if not min_term <= app.term_months <= max_term:
        problems.append(f"term of {app.term_months} months is outside {min_term} to {max_term} months")
    if problems:
        context.find(
            "outside_product_limits",
            "refer",
            ["NBC-1.1", "NBC-7.3"],
            [CHIEF_CREDIT_OFFICER],
            "Request is outside product limits: " + "; ".join(problems) + ".",
            evidence,
            trigger_ids=["R-NBC-14"],
        )
    else:
        context.passed(
            "product_limits",
            ["NBC-1.1"],
            f"Requested {label} of {money(app.requested_amount)} over {app.term_months} months is within the "
            f"product limits ({money(minimum)} to {money(maximum)}).",
            evidence,
        )


def check_industry(app: CreditApplication, time_in_business: int, context: RuleContext) -> None:
    """NBC-2.2 prohibited industries (decline-for-review) and NBC-2.3 restricted industries (refer)."""
    industry = app.applicant.industry
    if industry in PROHIBITED_INDUSTRIES:
        context.find(
            "prohibited_industry",
            "decline_for_review",
            ["NBC-2.2", "NBC-7.1"],
            [SENIOR_UNDERWRITER],
            f"Industry code {industry} is a prohibited industry; no exception may be granted.",
            ["APP-industry"],
        )
        return
    if industry in RESTRICTED_INDUSTRIES:
        context.find(
            "restricted_industry",
            "refer",
            ["NBC-2.3", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Industry code {industry} is a restricted industry and needs Senior Underwriter review.",
            ["APP-industry"],
            trigger_ids=["R-NBC-01"],
        )
    elif industry == "restaurant" and time_in_business < RESTAURANT_MIN_MONTHS:
        context.find(
            "restricted_industry",
            "refer",
            ["NBC-2.3", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Restaurant in operation for {time_in_business} months (less than 36 months) is restricted.",
            ["APP-industry", "CALC-time_in_business_months"],
            trigger_ids=["R-NBC-01"],
        )
    else:
        context.passed(
            "industry_eligible",
            ["NBC-2.2", "NBC-2.3"],
            f"Industry code {industry} is neither prohibited nor restricted.",
            ["APP-industry"],
        )
    if app.largest_customer_revenue_pct > MAX_SINGLE_CUSTOMER_PCT:
        context.find(
            "customer_concentration",
            "refer",
            ["NBC-2.3", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Largest customer is {pct(app.largest_customer_revenue_pct)} of revenue (more than 30%).",
            ["APP-largest_customer_revenue_pct"],
            trigger_ids=["R-NBC-01"],
        )


def check_time_in_business(app: CreditApplication, context: RuleContext) -> int:
    """NBC-2.1 item 1: at least 24 months; 18 to 23 months is a near miss for the exceptions process."""
    months = months_between(app.applicant.formation_date, app.application_date)
    calc_id = context.calc(
        "time_in_business_months",
        "Time in business (months)",
        months,
        "months",
        ["NBC-2.1"],
        formation_date=app.applicant.formation_date.isoformat(),
        application_date=app.application_date.isoformat(),
    )
    if app.applicant.business_structure not in ELIGIBLE_STRUCTURES:
        context.find(
            "ineligible_business_structure",
            "decline_for_review",
            ["NBC-2.1"],
            [SENIOR_UNDERWRITER],
            f"Business structure {app.applicant.business_structure} is not a corporation, LLC or partnership.",
            ["APP-business_structure"],
        )
    if months >= MIN_TIME_IN_BUSINESS_MONTHS:
        context.passed(
            "time_in_business", ["NBC-2.1"], f"Time in business is {months} months (at least 24).", [calc_id]
        )
    elif months >= NEAR_MISS_TIME_IN_BUSINESS_MONTHS:
        context.find(
            "time_in_business_near_miss",
            "refer",
            ["NBC-2.1", "NBC-7.4"],
            [SENIOR_UNDERWRITER],
            f"Time in business is {months} months: below 24 but at least 18, so only an exception can approve it.",
            [calc_id],
        )
    else:
        context.find(
            "time_in_business_short",
            "decline_for_review",
            ["NBC-2.1"],
            [SENIOR_UNDERWRITER],
            f"Time in business is {months} months, below the 18-month exception floor.",
            [calc_id],
        )
    return months


def check_principals(app: CreditApplication, context: RuleContext) -> None:
    """NBC-2.1 items 3 and 4: principal credit scores and bankruptcies."""
    people = principals(app)
    missing = [owner.owner_id for owner in people if owner.credit_score is None]
    if missing:
        context.find(
            "principal_credit_score_missing",
            "request_more_information",
            ["NBC-2.1", "NBC-3.1"],
            [SENIOR_UNDERWRITER],
            f"No credit score for principal(s) {', '.join(missing)}.",
            ["APP-owners"],
        )
    scores = [owner.credit_score for owner in people if owner.credit_score is not None]
    if scores:
        lowest = min(scores)
        calc_id = context.calc(
            "min_principal_credit_score", "Lowest principal credit score", lowest, "score", ["NBC-2.1"]
        )
        if lowest >= MIN_CREDIT_SCORE:
            context.passed(
                "principal_credit_scores",
                ["NBC-2.1"],
                f"Lowest principal credit score is {lowest} (at least 660).",
                [calc_id],
            )
        elif lowest >= NEAR_MISS_CREDIT_SCORE:
            context.find(
                "credit_score_near_miss",
                "refer",
                ["NBC-2.1", "NBC-7.4"],
                [SENIOR_UNDERWRITER],
                f"Lowest principal credit score is {lowest}: below 660 but at least 640 (exception only).",
                [calc_id],
            )
        else:
            context.find(
                "credit_score_below_minimum",
                "decline_for_review",
                ["NBC-2.1"],
                [SENIOR_UNDERWRITER],
                f"Lowest principal credit score is {lowest}, below the 640 exception floor.",
                [calc_id],
            )
    bankrupt = [owner.owner_id for owner in people if owner.bankruptcy_within_7_years]
    if bankrupt:
        context.find(
            "bankruptcy_within_7_years",
            "decline_for_review",
            ["NBC-2.1", "NBC-7.4"],
            [SENIOR_UNDERWRITER],
            f"Principal(s) {', '.join(bankrupt)} had a bankruptcy within the last 7 years; no exception is possible.",
            ["APP-owners"],
        )
    if app.tax_lien_unresolved:
        context.find(
            "unresolved_tax_lien",
            "decline_for_review",
            ["NBC-2.1", "NBC-7.4"],
            [SENIOR_UNDERWRITER],
            "The applicant has a tax lien that is not on an approved repayment plan.",
            ["APP-tax_lien_unresolved"],
        )


def required_documents(app: CreditApplication) -> list[str]:
    """NBC-3.1: the base documents plus the conditional ones for this request."""
    required = list(BASE_DOCUMENTS)
    if app.requested_amount > CPA_FINANCIALS_ABOVE:
        required.append("cpa_reviewed_financials")
    if app.collateral or app.requested_amount > COLLATERAL_REQUIRED_ABOVE:
        required.append("collateral_documents")
    if app.purpose == "refinance":
        required.append("payoff_letters")
    return required


def check_documents(app: CreditApplication, context: RuleContext) -> None:
    """NBC-3.1: a missing required document puts the file in request-more-information status."""
    missing = [name for name in required_documents(app) if name not in app.documents_provided]
    if missing:
        context.find(
            "missing_documents",
            "request_more_information",
            ["NBC-3.1"],
            [SENIOR_UNDERWRITER],
            f"Required documents are missing: {', '.join(missing)}.",
            ["APP-documents_provided"],
        )
    else:
        context.passed(
            "documents_complete", ["NBC-3.1"], "Every required document is in the file.", ["APP-documents_provided"]
        )


def check_account_conduct(app: CreditApplication, proposed_service: float, context: RuleContext) -> None:
    """NBC-3.5: NSF events, negative months, and average balance against proposed debt service."""
    statements = app.bank_statements
    if statements.nsf_events_last_6_months > MAX_NSF_EVENTS:
        context.find(
            "nsf_events",
            "refer",
            ["NBC-3.5", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"{statements.nsf_events_last_6_months} NSF or overdraft events in the last 6 months (more than 6).",
            ["APP-nsf_events_last_6_months"],
            trigger_ids=["R-NBC-03"],
        )
    if statements.negative_balance_months_last_12 > MAX_NEGATIVE_MONTHS:
        context.find(
            "negative_balance_months",
            "refer",
            ["NBC-3.5", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"{statements.negative_balance_months_last_12} of the last 12 months ended with a negative balance "
            "(3 or more).",
            ["APP-negative_balance_months_last_12"],
            trigger_ids=["R-NBC-03"],
        )
    floor = round(MIN_BALANCE_SHARE_OF_DEBT_SERVICE * proposed_service, 2)
    calc_id = context.calc(
        "average_balance_floor",
        "5% of proposed annual debt service",
        floor,
        "usd",
        ["NBC-3.5"],
        proposed_annual_debt_service=round(proposed_service, 2),
    )
    if statements.average_daily_balance_last_3_months < floor:
        context.find(
            "low_average_balance",
            "refer",
            ["NBC-3.5", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Average daily balance {money(statements.average_daily_balance_last_3_months)} is below "
            f"{money(floor)} (5% of proposed annual debt service).",
            ["APP-average_daily_balance_last_3_months", calc_id],
            trigger_ids=["R-NBC-03"],
        )
    if (
        statements.nsf_events_last_6_months <= MAX_NSF_EVENTS
        and statements.negative_balance_months_last_12 <= MAX_NEGATIVE_MONTHS
        and statements.average_daily_balance_last_3_months >= floor
    ):
        context.passed(
            "account_conduct",
            ["NBC-3.5"],
            f"{statements.nsf_events_last_6_months} NSF events, {statements.negative_balance_months_last_12} "
            f"negative months, and an average daily balance of {money(statements.average_daily_balance_last_3_months)}"
            " meet the account conduct rules.",
            [
                "APP-nsf_events_last_6_months",
                "APP-negative_balance_months_last_12",
                "APP-average_daily_balance_last_3_months",
            ],
        )


def check_revenue(app: CreditApplication, context: RuleContext) -> float | None:
    """NBC-3.2 to NBC-3.4 and NBC-2.1 item 2. Returns verified annual revenue, or None if not verifiable."""
    verification = verify_revenue(app.bank_statements)
    if verification.verified_annual_revenue is None:
        context.find(
            "bank_statements_incomplete",
            "request_more_information",
            ["NBC-3.2", "NBC-3.5"],
            [SENIOR_UNDERWRITER],
            f"Bank statements cover {verification.months_provided} months (12 consecutive months are required), so "
            "revenue cannot be verified.",
            ["APP-statement_months"],
            trigger_ids=["R-NBC-03"],
        )
        for rule, sections in (
            ("verified_revenue_minimum", ["NBC-2.1"]),
            ("revenue_gap", ["NBC-3.4"]),
            ("dscr", ["NBC-4.1", "NBC-4.2"]),
            ("leverage", ["NBC-4.4"]),
            ("stress_tests", ["NBC-4.5"]),
        ):
            context.skip(rule, sections, "verified revenue is not available until 12 consecutive months are provided")
        return None

    verified = verification.verified_annual_revenue
    revenue_id = context.calc(
        "verified_annual_revenue",
        "Verified annual revenue (eligible deposits, 12 months)",
        verified,
        "usd",
        ["NBC-3.2", "NBC-3.3"],
        non_revenue_deposits_excluded=verification.non_revenue_total,
        cash_excluded_by_cap=verification.excluded_cash,
    )
    if verified >= MIN_VERIFIED_REVENUE:
        context.passed(
            "verified_revenue_minimum",
            ["NBC-2.1", "NBC-3.2"],
            f"Verified annual revenue is {money(verified)} (at least {money(MIN_VERIFIED_REVENUE)}).",
            [revenue_id],
        )
    elif verified >= NEAR_MISS_VERIFIED_REVENUE:
        context.find(
            "revenue_near_miss",
            "refer",
            ["NBC-2.1", "NBC-7.4"],
            [SENIOR_UNDERWRITER],
            f"Verified annual revenue is {money(verified)}: below {money(MIN_VERIFIED_REVENUE)} but at least "
            f"{money(NEAR_MISS_VERIFIED_REVENUE)} (exception only).",
            [revenue_id],
        )
    else:
        context.find(
            "revenue_below_minimum",
            "decline_for_review",
            ["NBC-2.1"],
            [SENIOR_UNDERWRITER],
            f"Verified annual revenue is {money(verified)}, below the {money(NEAR_MISS_VERIFIED_REVENUE)} exception "
            "floor.",
            [revenue_id],
        )

    gap = revenue_gap_pct(app.stated_annual_revenue, verified)
    gap_id = context.calc(
        "revenue_gap_pct",
        "Revenue gap (stated minus verified, percent of stated)",
        gap,
        "pct",
        ["NBC-3.4"],
        stated_annual_revenue=app.stated_annual_revenue,
        verified_annual_revenue=verified,
    )
    if gap is None or gap <= BENIGN_GAP_PCT:
        gap_text = "not computable (stated revenue is zero)" if gap is None else pct(gap)
        context.passed("revenue_gap", ["NBC-3.4"], f"Revenue gap is {gap_text} (10.0% or less is benign).", [gap_id])
    elif gap <= EXPLAINABLE_GAP_PCT:
        context.find(
            "revenue_gap_explainable",
            "refer",
            ["NBC-3.4", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Revenue gap is {pct(gap)} (more than 10.0% and up to 25.0%): a written explanation is needed.",
            [gap_id],
            trigger_ids=["R-NBC-02"],
        )
    else:
        context.find(
            "revenue_gap_unsupported",
            "refer",
            ["NBC-3.4", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Revenue gap is {pct(gap)} (more than 25.0%): unsupported until explained with documents.",
            [gap_id],
            trigger_ids=["R-NBC-02"],
        )
    return verified


def check_dscr_and_stress(
    app: CreditApplication, verified: float, existing_service: float, proposed_service: float, context: RuleContext
) -> float:
    """NBC-4.1 / NBC-4.2 decision DSCR, and NBC-4.5 revenue and rate stress. Returns NOI."""
    noi = net_operating_income(app.debt, verified)
    noi_id = context.calc(
        "noi",
        "Net operating income (TTM EBITDA less revenue shortfall)",
        noi,
        "usd",
        ["NBC-4.1"],
        ttm_ebitda=app.debt.ttm_ebitda,
        ttm_pnl_revenue=app.debt.ttm_pnl_revenue,
        verified_annual_revenue=verified,
    )
    annual_service = round(existing_service + proposed_service, 2)
    service_id = context.calc(
        "annual_debt_service",
        "Annual debt service (existing plus proposed)",
        annual_service,
        "usd",
        ["NBC-4.1"],
        existing_annual_debt_service=existing_service,
        proposed_annual_debt_service=round(proposed_service, 2),
    )
    decision_dscr = coverage_ratio(noi, annual_service)
    dscr_id = context.calc(
        "dscr",
        "Decision DSCR (NOI / annual debt service)",
        decision_dscr,
        "ratio",
        ["NBC-4.1", "NBC-4.2"],
        noi=noi,
        annual_debt_service=annual_service,
    )
    stated_noi = round(app.debt.ttm_ebitda - max(0.0, app.debt.ttm_pnl_revenue - app.stated_annual_revenue), 2)
    context.calc(
        "dscr_stated_revenue",
        "DSCR with stated revenue (information only)",
        coverage_ratio(stated_noi, annual_service),
        "ratio",
        ["NBC-4.1"],
        noi_with_stated_revenue=stated_noi,
        annual_debt_service=annual_service,
    )
    minimum = DSCR_MINIMUM[app.product]
    label = PRODUCT_LABELS[app.product]
    evidence = [dscr_id, noi_id, service_id]
    if decision_dscr is None:
        context.skip("dscr", ["NBC-4.2"], "annual debt service is zero")
    elif decision_dscr >= minimum:
        context.passed(
            "dscr",
            ["NBC-4.2"],
            f"Decision DSCR is {ratio(decision_dscr)}, at or above the {label} minimum {ratio(minimum)}.",
            evidence,
        )
    elif decision_dscr >= DSCR_DECLINE_FLOOR:
        context.find(
            "dscr_below_minimum",
            "refer",
            ["NBC-4.2", "NBC-4.3", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Decision DSCR is {ratio(decision_dscr)}, below the {label} minimum {ratio(minimum)} but at least "
            f"{ratio(DSCR_DECLINE_FLOOR)}; approval needs two compensating factors and an exception.",
            evidence,
            trigger_ids=["R-NBC-04"],
        )
    else:
        context.find(
            "dscr_below_decline_floor",
            "decline_for_review",
            ["NBC-4.2", "NBC-7.3"],
            [SENIOR_UNDERWRITER, CREDIT_COMMITTEE],
            f"Decision DSCR is {ratio(decision_dscr)}, below {ratio(DSCR_DECLINE_FLOOR)}: does not meet policy "
            "unless the Credit Committee grants an exception.",
            evidence,
            trigger_ids=["R-NBC-04"],
        )
    check_stress(app, verified, noi, existing_service, annual_service, context)
    return noi


def stress_applies(app: CreditApplication) -> bool:
    """NBC-4.5: term and equipment loans above $100,000, and every line of credit."""
    return app.product == "line_of_credit" or app.requested_amount > STRESS_AMOUNT_ABOVE


def check_stress(
    app: CreditApplication,
    verified: float,
    noi: float,
    existing_service: float,
    annual_service: float,
    context: RuleContext,
) -> None:
    """NBC-4.5 revenue stress (all stressed products) and rate stress (variable rate only)."""
    if not stress_applies(app):
        context.passed(
            "stress_tests",
            ["NBC-4.5"],
            f"No stress test is required for a {PRODUCT_LABELS[app.product]} of {money(app.requested_amount)} "
            f"(not above {money(STRESS_AMOUNT_ABOVE)}).",
            ["APP-requested_amount", "APP-product"],
        )
        return
    stressed_noi = round(noi - REVENUE_STRESS_SHARE * verified, 2)
    stressed = coverage_ratio(stressed_noi, annual_service)
    stress_id = context.calc(
        "stressed_dscr_revenue",
        "DSCR after a 20% revenue stress",
        stressed,
        "ratio",
        ["NBC-4.5"],
        stressed_noi=stressed_noi,
        annual_debt_service=annual_service,
    )
    if stressed is not None and stressed < REVENUE_STRESS_MIN_DSCR:
        context.find(
            "stress_test_revenue",
            "refer",
            ["NBC-4.5", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Revenue-stressed DSCR is {ratio(stressed)}, below {ratio(REVENUE_STRESS_MIN_DSCR)}.",
            [stress_id],
            trigger_ids=["R-NBC-06"],
        )
    elif stressed is not None:
        context.passed(
            "stress_test_revenue",
            ["NBC-4.5"],
            f"Revenue-stressed DSCR is {ratio(stressed)} (at least {ratio(REVENUE_STRESS_MIN_DSCR)}).",
            [stress_id],
        )
    if app.rate_type != "variable":
        return
    stressed_rate = app.note_rate_pct + RATE_STRESS_ADD_PCT
    rate_service = round(
        existing_service
        + proposed_annual_debt_service(app.product, app.requested_amount, stressed_rate, app.term_months),
        2,
    )
    rate_dscr = coverage_ratio(noi, rate_service)
    rate_id = context.calc(
        "stressed_dscr_rate",
        "DSCR with the note rate plus 2.00 percentage points",
        rate_dscr,
        "ratio",
        ["NBC-4.5"],
        stressed_rate_pct=round(stressed_rate, 2),
        annual_debt_service_stressed=rate_service,
        noi=noi,
    )
    if rate_dscr is not None and rate_dscr < RATE_STRESS_MIN_DSCR:
        context.find(
            "stress_test_rate",
            "refer",
            ["NBC-4.5", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"Rate-stressed DSCR is {ratio(rate_dscr)}, below {ratio(RATE_STRESS_MIN_DSCR)}.",
            [rate_id],
            trigger_ids=["R-NBC-06"],
        )
    elif rate_dscr is not None:
        context.passed(
            "stress_test_rate",
            ["NBC-4.5"],
            f"Rate-stressed DSCR is {ratio(rate_dscr)} (at least {ratio(RATE_STRESS_MIN_DSCR)}).",
            [rate_id],
        )


def check_leverage_and_stacking(app: CreditApplication, noi: float, context: RuleContext) -> None:
    """NBC-4.4: total debt after closing / NOI at most 4.00; stacking of new loans in the last 90 days."""
    total_debt = round(app.debt.existing_total_debt_balance + app.requested_amount, 2)
    leverage = round_ratio(total_debt / noi) if noi > 0 else None
    leverage_id = context.calc(
        "leverage",
        "Total debt after closing / NOI",
        leverage,
        "ratio",
        ["NBC-4.4"],
        total_debt_after_closing=total_debt,
        noi=noi,
    )
    if leverage is None or leverage > MAX_LEVERAGE:
        detail = (
            f"Leverage is {ratio(leverage)} (above 4.00)."
            if leverage is not None
            else "NOI is not positive, so leverage is above any limit."
        )
        context.find(
            "leverage_above_maximum",
            "refer",
            ["NBC-4.4", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            detail,
            [leverage_id],
            trigger_ids=["R-NBC-05"],
        )
    else:
        context.passed("leverage", ["NBC-4.4"], f"Leverage is {ratio(leverage)} (4.00 or less).", [leverage_id])
    count = app.new_loans_last_90_days
    if count >= STACKING_CCO:
        context.find(
            "stacking",
            "refer",
            ["NBC-4.4", "NBC-7.3"],
            [CHIEF_CREDIT_OFFICER],
            f"{count} new loans or advances in the last 90 days (3 or more): not eligible for standard processing.",
            ["APP-new_loans_last_90_days"],
            trigger_ids=["R-NBC-05"],
        )
    elif count >= STACKING_REFER:
        context.find(
            "stacking",
            "refer",
            ["NBC-4.4", "NBC-7.3"],
            [SENIOR_UNDERWRITER],
            f"{count} new loans or advances in the last 90 days (2 or more).",
            ["APP-new_loans_last_90_days"],
            trigger_ids=["R-NBC-05"],
        )


def check_collateral(app: CreditApplication, context: RuleContext) -> None:
    """NBC-5.1: collateral coverage of at least 1.00 for secured requests (required above $250,000)."""
    required = app.requested_amount > COLLATERAL_REQUIRED_ABOVE
    if not required and not app.collateral:
        context.passed(
            "collateral",
            ["NBC-5.1"],
            f"A request of {money(app.requested_amount)} may be unsecured "
            f"(not above {money(COLLATERAL_REQUIRED_ABOVE)}).",
            ["APP-requested_amount"],
        )
        return
    coverage = collateral_coverage(app.collateral, app.requested_amount)
    coverage_id = context.calc(
        "collateral_coverage",
        "Collateral coverage (value x advance rate / loan amount)",
        coverage,
        "ratio",
        ["NBC-5.1"],
        loan_amount=app.requested_amount,
    )
    if coverage >= MIN_COLLATERAL_COVERAGE:
        context.passed(
            "collateral", ["NBC-5.1"], f"Collateral coverage is {ratio(coverage)} (at least 1.00).", [coverage_id]
        )
        return
    refer_to = (
        [SENIOR_UNDERWRITER] if coverage >= COMMITTEE_COLLATERAL_COVERAGE else [SENIOR_UNDERWRITER, CREDIT_COMMITTEE]
    )
    context.find(
        "collateral_coverage_below_minimum",
        "refer",
        ["NBC-5.1", "NBC-7.3"],
        refer_to,
        f"Collateral coverage is {ratio(coverage)} (below 1.00).",
        [coverage_id, "APP-collateral"],
        trigger_ids=["R-NBC-07"],
    )


def check_ownership(app: CreditApplication, context: RuleContext) -> None:
    """NBC-6.1: ownership must add up to 100% and name one control person (else Senior UW and KYC Operations)."""
    total = round(sum(owner.ownership_pct for owner in app.owners), 1)
    total_id = context.calc("ownership_total_pct", "Disclosed ownership total (percent)", total, "pct", ["NBC-6.1"])
    controls = [owner for owner in app.owners if owner.is_control_person]
    problems = []
    if abs(total - 100.0) > OWNERSHIP_TOTAL_TOLERANCE_PCT:
        problems.append(f"disclosed ownership adds up to {pct(total)}, not 100%")
    if len(controls) != 1:
        problems.append("the application does not name exactly one control person")
    if problems:
        context.find(
            "ownership_discrepancy",
            "refer",
            ["NBC-6.1", "NBC-7.3"],
            [SENIOR_UNDERWRITER, KYC_OPERATIONS],
            "Ownership discrepancy: " + "; ".join(problems) + ".",
            [total_id, "APP-owners"],
            trigger_ids=["R-NBC-08"],
            blocking=True,
        )
    else:
        context.passed(
            "ownership", ["NBC-6.1"], f"Ownership adds up to {pct(total)} with one control person.", [total_id]
        )


def evaluate_credit_rules(app: CreditApplication, inputs: PolicyInputs, context: RuleContext) -> None:
    """Run every Northwind rule that the structured application supports."""
    del inputs  # sanctions, injection and the risk model are checked in policy.py for both lines
    check_product_limits(app, context)
    time_in_business = check_time_in_business(app, context)
    check_industry(app, time_in_business, context)
    check_principals(app, context)
    check_documents(app, context)
    proposed_service = proposed_annual_debt_service(
        app.product, app.requested_amount, app.note_rate_pct, app.term_months
    )
    context.calc(
        "proposed_annual_debt_service",
        "Proposed annual debt service",
        round(proposed_service, 2),
        "usd",
        ["NBC-4.1"],
        requested_amount=app.requested_amount,
        note_rate_pct=app.note_rate_pct,
        term_months=app.term_months,
    )
    check_account_conduct(app, proposed_service, context)
    verified = check_revenue(app, context)
    if verified is not None:
        payoff_in_file = "payoff_letters" in app.documents_provided
        existing_service = existing_annual_debt_service(app.debt, payoff_in_file)
        noi = check_dscr_and_stress(app, verified, existing_service, proposed_service, context)
        check_leverage_and_stacking(app, noi, context)
    else:
        check_stacking_only(app, context)
    check_collateral(app, context)
    check_ownership(app, context)
    for rule, sections, reason in NOT_IMPLEMENTED:
        context.skip(rule, sections, f"not implemented: {reason}")


def check_stacking_only(app: CreditApplication, context: RuleContext) -> None:
    """Stacking does not need revenue, so it is still checked when the statements are incomplete."""
    count = app.new_loans_last_90_days
    if count >= STACKING_REFER:
        target = CHIEF_CREDIT_OFFICER if count >= STACKING_CCO else SENIOR_UNDERWRITER
        context.find(
            "stacking",
            "refer",
            ["NBC-4.4", "NBC-7.3"],
            [target],
            f"{count} new loans or advances in the last 90 days (2 or more).",
            ["APP-new_loans_last_90_days"],
            trigger_ids=["R-NBC-05"],
        )
