"""Harborline (insurance) document checks over an InsuranceSubmission: documents, identity and TIV.

Thresholds come from agent/policy_insurance.py and formulas from agent/insurance_calculations.py. The loss-run
checks are in checks_loss_runs.py.

- required_documents (HM-3.1): the checklist codes the packet holds (documents_provided below) against the
  required list, built like policy_insurance.required_documents: signed_application, property_schedule,
  loss_runs, prior_carrier_history, beneficial_ownership_statement, plus revenue_and_payroll when general
  liability is requested and inspection_report when a location's TIV is above $5,000,000. The schema has no flood
  zone, so whether an elevation certificate is needed cannot be decided here (values: not_determinable).
  Mapping: signed_application <- the application; property_schedule <- the schedule; loss_runs <- at least one
  carrier loss run (whether they cover every year is loss_run_coverage); prior_carrier_history <- rows in the
  application's prior-carriers table; revenue_and_payroll <- annual revenue and payroll on the application;
  beneficial_ownership_statement <- the ownership statement; inspection_report <- a report for every location
  above $5,000,000, dated no more than 12 months before the proposed effective date when its date is known.
- name_consistency (HM-3.1, 8.1): the named insured on the application, every loss run, the property schedule and
  the ownership statement. fein_consistency: only the application carries a FEIN in the extraction schema, so it
  is normally not evaluated (fewer than two documents).
- owner_consistency (HM-3.1 item 6, 8.1): application owners vs the ownership statement, which lists only owners
  of 25% or more, so an application owner below 25% may be absent from it.
- tiv_consistency (HM-5.2, HM-11.1): the application TIV against the schedule TIV (the sum of each location's
  TIV; a location without a printed TIV uses building + contents + business income). Difference bands (percent of
  the schedule TIV): 5.0% or less passes (use the schedule TIV); more than 5.0% fails (request a corrected
  application or schedule); more than 15.0% fails (refer). Inside the schedule, each location's printed TIV must
  equal building + contents + business income and the printed total must equal the sum of the locations, both
  within $1.00; a difference there is an integrity indicator and makes the check high severity.
"""

from datetime import timedelta

from uwcopilot.agent.applications import Location
from uwcopilot.agent.insurance_calculations import RECONCILIATION_TOLERANCE_USD, location_tiv, tiv_difference_pct
from uwcopilot.agent.policy_insurance import INSPECTION_REPORT_ABOVE_TIV, TIV_CORRECTION_PCT, TIV_REFER_PCT
from uwcopilot.docintel.check_models import (
    INSURANCE_REFS,
    CheckIssue,
    CheckResult,
    Evidence,
    cite,
    document_evidence,
    found_only,
    make_result,
)
from uwcopilot.docintel.checks_identity import owner_consistency, value_consistency
from uwcopilot.docintel.normalize import canonical_fein, canonical_name
from uwcopilot.docintel.schemas import InsuranceSubmission, ScheduleLocation, casefold_key

OWNERSHIP_STATEMENT_MIN_PCT = 25.0  # HM-3.1 item 6: the statement lists every individual owning 25% or more
INSPECTION_MAX_AGE_DAYS = 365  # HM-3.1: an inspection report dated within 12 months
NOT_DETERMINABLE = ("elevation_certificate",)


def requests_gl(submission: InsuranceSubmission) -> bool:
    """True when general liability is requested (package or a liability coverage), or the coverage is unknown."""
    application = submission.application
    if application is None or not application.coverage.found:
        return True
    words = canonical_name(application.coverage.value).split()
    return "package" in words or "liability" in words or "gl" in words


def location_id_of(location: ScheduleLocation) -> str | None:
    """The location id as printed, stripped (None when missing)."""
    return None if not location.location_id.found else str(location.location_id.value).strip()


def parts_total(location: ScheduleLocation) -> float | None:
    """Building + contents + business income with insurance_calculations.location_tiv (None if a part is missing).

    location_tiv reads only those three values, so the other Location fields are not needed.
    """
    parts = (location.building_value, location.contents_value, location.business_income_value)
    if any(not part.found for part in parts):
        return None
    values = Location.model_construct(
        building_value=float(parts[0].value),
        contents_value=float(parts[1].value),
        business_income_value=float(parts[2].value),
    )
    return round(location_tiv(values), 2)


def location_value(location: ScheduleLocation) -> float | None:
    """A location's TIV: the printed TIV, else building + contents + business income."""
    if location.tiv.found:
        return float(location.tiv.value)
    return parts_total(location)


def large_location_ids(submission: InsuranceSubmission) -> list[str]:
    """Location ids (or positions "#n") of schedule locations with TIV above $5,000,000 (HM-3.1)."""
    schedule = submission.property_schedule
    if schedule is None:
        return []
    result = []
    for index, location in enumerate(schedule.locations, start=1):
        value = location_value(location)
        if value is not None and value > INSPECTION_REPORT_ABOVE_TIV:
            result.append(location_id_of(location) or f"#{index}")
    return result


def current_inspections(submission: InsuranceSubmission) -> list[str]:
    """Location ids with an inspection report that is not known to be older than 12 months before the effective date."""
    application = submission.application
    effective = None if application is None else application.proposed_effective_date.value
    result = []
    for report in submission.inspection_reports:
        if not report.location_id.found:
            continue
        inspected = report.inspection_date.value
        if effective is not None and inspected is not None:
            if inspected < effective - timedelta(days=INSPECTION_MAX_AGE_DAYS):
                continue
        result.append(str(report.location_id.value).strip())
    return result


def documents_provided(submission: InsuranceSubmission) -> dict[str, list[Evidence]]:
    """Checklist code -> evidence of the documents that satisfy it (only the codes the packet satisfies)."""
    provided: dict[str, list[Evidence]] = {}
    application = submission.application
    if application is not None:
        provided["signed_application"] = [document_evidence(application, "document", "insurance_application")]
        rows = []
        for row in application.prior_carriers:
            rows.append(
                cite(row.carrier, "application.prior_carriers.carrier", casefold_key(row.carrier), application.doc_id)
            )
        if found_only(rows):
            provided["prior_carrier_history"] = found_only(rows)
        revenue = cite(application.annual_revenue, "application.annual_revenue", None, application.doc_id)
        payroll = cite(application.annual_payroll, "application.annual_payroll", None, application.doc_id)
        if revenue is not None and payroll is not None:
            provided["revenue_and_payroll"] = [revenue, payroll]
    if submission.property_schedule is not None:
        provided["property_schedule"] = [
            document_evidence(submission.property_schedule, "document", "property_schedule")
        ]
    if submission.loss_runs:
        provided["loss_runs"] = [
            document_evidence(item, "document", "loss_run", casefold_key(item.carrier)) for item in submission.loss_runs
        ]
    if submission.ownership_statement is not None:
        statement = submission.ownership_statement
        provided["beneficial_ownership_statement"] = [document_evidence(statement, "document", "ownership_statement")]
    large = large_location_ids(submission)
    inspected = set(current_inspections(submission))
    if large and all(location in inspected for location in large):
        provided["inspection_report"] = [
            document_evidence(report, "document", "inspection_report") for report in submission.inspection_reports
        ]
    return provided


def required_documents_list(submission: InsuranceSubmission) -> list[str]:
    """HM-3.1 required documents for this submission (see the module docstring)."""
    required = [
        "signed_application",
        "property_schedule",
        "loss_runs",
        "prior_carrier_history",
        "beneficial_ownership_statement",
    ]
    if requests_gl(submission):
        required.append("revenue_and_payroll")
    if large_location_ids(submission):
        required.append("inspection_report")
    return required


def check_required_documents(submission: InsuranceSubmission) -> CheckResult:
    """HM-3.1 required documents."""
    refs = INSURANCE_REFS["required_documents"]
    provided = documents_provided(submission)
    required = required_documents_list(submission)
    missing = [code for code in required if code not in provided]
    values = {
        "required": required,
        "provided": sorted(provided),
        "missing": missing,
        "not_determinable": list(NOT_DETERMINABLE),
        "locations_needing_inspection": large_location_ids(submission),
    }
    evidence = [item for items in provided.values() for item in items]
    if missing:
        message = f"Required documents are missing: {', '.join(missing)}."
        return make_result("required_documents", refs, "fail", message, values, evidence)
    message = f"All {len(required)} required documents are in the packet."
    return make_result("required_documents", refs, "pass", message, values, evidence)


def name_sources(submission: InsuranceSubmission) -> list[Evidence]:
    """Every printed named-insured name (see the module docstring), as evidence."""
    items: list[Evidence | None] = []
    application = submission.application
    if application is not None:
        items.append(cite(application.named_insured, "application.named_insured", None, application.doc_id))
    for loss_run in submission.loss_runs:
        key = casefold_key(loss_run.carrier)
        items.append(cite(loss_run.insured_name, "loss_run.insured_name", key, loss_run.doc_id))
    schedule = submission.property_schedule
    if schedule is not None:
        items.append(cite(schedule.insured_name, "property_schedule.insured_name", None, schedule.doc_id))
    statement = submission.ownership_statement
    if statement is not None:
        items.append(cite(statement.entity_name, "ownership_statement.entity_name", None, statement.doc_id))
    return found_only(items)


def check_name(submission: InsuranceSubmission) -> CheckResult:
    """The named insured agrees across documents (names are not repeated in the message)."""
    refs = INSURANCE_REFS["name_consistency"]
    sources = name_sources(submission)
    return value_consistency("name_consistency", refs, "named insured", sources, canonical_name, False)


def check_fein(submission: InsuranceSubmission) -> CheckResult:
    """The FEIN agrees across documents (only the application carries one in the schema; see the module docstring)."""
    refs = INSURANCE_REFS["fein_consistency"]
    application = submission.application
    sources = []
    if application is not None:
        sources = found_only([cite(application.fein, "application.fein", None, application.doc_id)])
    return value_consistency("fein_consistency", refs, "FEIN", sources, canonical_fein, True)


def check_owners(submission: InsuranceSubmission) -> CheckResult:
    """HM-3.1 item 6 / HM-8.1: application owners vs the beneficial ownership statement."""
    application = submission.application
    statement = submission.ownership_statement
    return owner_consistency(
        INSURANCE_REFS["owner_consistency"],
        [] if application is None else application.owners,
        None if application is None else application.doc_id,
        None if statement is None else statement.owners,
        None if statement is None else statement.doc_id,
        "ownership_statement.owners",
        OWNERSHIP_STATEMENT_MIN_PCT,
    )


def location_evidence(location: ScheduleLocation, doc_id: str) -> list[Evidence]:
    """Evidence for a location's TIV and its three parts (key = location id)."""
    key = location_id_of(location)
    names = ("tiv", "building_value", "contents_value", "business_income_value")
    return found_only(
        [cite(getattr(location, name), f"property_schedule.locations.{name}", key, doc_id) for name in names]
    )


def location_arithmetic_issue(location: ScheduleLocation, doc_id: str) -> CheckIssue | None:
    """A location whose printed TIV is not building + contents + business income (within $1.00)."""
    expected = parts_total(location)
    if expected is None or not location.tiv.found:
        return None
    printed = float(location.tiv.value)
    if abs(printed - expected) <= RECONCILIATION_TOLERANCE_USD:
        return None
    key = location_id_of(location)
    evidence = location_evidence(location, doc_id)
    return CheckIssue(
        issue="location_tiv_arithmetic",
        entity=f"location:{key or 'unknown'}",
        fields=["property_schedule.locations.tiv", "property_schedule.locations.building_value"],
        side_a=evidence[:1],
        side_b=evidence[1:],
        data={"location_id": key, "printed_tiv": printed, "parts_total": expected},
        message=f"Location {key} shows TIV ${printed:,.2f} but its building, contents and business income add up "
        f"to ${expected:,.2f}.",
    )


def schedule_total_issue(submission: InsuranceSubmission, schedule_total: float) -> CheckIssue | None:
    """The printed schedule total vs the sum of the location TIVs (within $1.00)."""
    schedule = submission.property_schedule
    if schedule is None or not schedule.total_tiv.found or not schedule.locations:
        return None
    printed = float(schedule.total_tiv.value)
    if abs(printed - schedule_total) <= RECONCILIATION_TOLERANCE_USD:
        return None
    side_b = []
    for location in schedule.locations:
        side_b.extend(location_evidence(location, schedule.doc_id)[:1])
    return CheckIssue(
        issue="schedule_total_arithmetic",
        entity="business",
        fields=["property_schedule.total_tiv", "property_schedule.locations.tiv"],
        side_a=found_only([cite(schedule.total_tiv, "property_schedule.total_tiv", None, schedule.doc_id)]),
        side_b=side_b,
        data={"printed_total": printed, "sum_of_locations": schedule_total},
        message=f"The schedule's printed total TIV ${printed:,.2f} differs from the sum of its locations "
        f"(${schedule_total:,.2f}).",
    )


def schedule_tiv(submission: InsuranceSubmission) -> tuple[float | None, list[Evidence]]:
    """The schedule TIV (sum of locations, or the printed total without locations) and its evidence."""
    schedule = submission.property_schedule
    if schedule is None:
        return None, []
    if not schedule.locations:
        total = cite(schedule.total_tiv, "property_schedule.total_tiv", None, schedule.doc_id)
        return (None, []) if total is None else (float(schedule.total_tiv.value), [total])
    values = [location_value(location) for location in schedule.locations]
    if any(value is None for value in values):
        return None, []
    evidence = []
    for location in schedule.locations:
        evidence.extend(location_evidence(location, schedule.doc_id)[:1])
    return round(sum(value for value in values if value is not None), 2), evidence


def tiv_band(difference_pct: float) -> str:
    """HM-5.2 treatment band for a TIV difference."""
    if difference_pct <= TIV_CORRECTION_PCT:
        return "use_schedule_tiv"
    if difference_pct <= TIV_REFER_PCT:
        return "request_correction"
    return "refer"


def check_tiv(submission: InsuranceSubmission) -> CheckResult:
    """HM-5.2 application TIV vs the property schedule, and the schedule's own arithmetic (module docstring)."""
    refs = INSURANCE_REFS["tiv_consistency"]
    application = submission.application
    schedule = submission.property_schedule
    total, schedule_evidence = schedule_tiv(submission)
    if schedule is None or total is None or total <= 0:
        message = "The property schedule TIV could not be determined."
        return make_result("tiv_consistency", refs, "not_evaluated", message, {}, schedule_evidence)
    issues = []
    for location in schedule.locations:
        issue = location_arithmetic_issue(location, schedule.doc_id)
        if issue is not None:
            issues.append(issue)
    total_issue = schedule_total_issue(submission, total)
    if total_issue is not None:
        issues.append(total_issue)
    values: dict = {"schedule_tiv": total, "tolerance_usd": RECONCILIATION_TOLERANCE_USD}
    stated = (
        None
        if application is None
        else cite(application.stated_tiv, "application.stated_tiv", None, application.doc_id)
    )
    if stated is not None:
        stated_value = float(application.stated_tiv.value)
        difference = tiv_difference_pct(stated_value, total)
        values.update({"stated_tiv": stated_value, "difference_pct": difference, "band": tiv_band(difference)})
        if difference > TIV_CORRECTION_PCT:
            issues.append(
                CheckIssue(
                    issue="tiv_differs",
                    entity="business",
                    fields=["application.stated_tiv", "property_schedule.locations.tiv"],
                    side_a=[stated],
                    side_b=schedule_evidence,
                    data=dict(values),
                    message=f"The application TIV ${stated_value:,.2f} differs from the schedule TIV ${total:,.2f} "
                    f"by {difference:.1f}% (HM-5.2 band: {values['band']}).",
                )
            )
    evidence = ([stated] if stated is not None else []) + schedule_evidence
    values["issue_codes"] = [issue.issue for issue in issues]
    if issues:
        integrity = any(issue.issue.endswith("_arithmetic") for issue in issues)
        message = " ".join(issue.message for issue in issues)
        severity = "high" if integrity else "medium"
        return make_result("tiv_consistency", refs, "fail", message, values, evidence, issues, severity)
    if stated is None:
        message = "The application TIV was not found; the schedule's own totals add up."
        return make_result("tiv_consistency", refs, "not_evaluated", message, values, evidence)
    message = f"The application TIV is within {values['difference_pct']:.1f}% of the schedule TIV ${total:,.2f}."
    return make_result("tiv_consistency", refs, "pass", message, values, evidence)
