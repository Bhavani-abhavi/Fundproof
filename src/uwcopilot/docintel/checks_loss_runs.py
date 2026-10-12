"""Harborline loss-run checks over an InsuranceSubmission: claim lines, totals and loss history (HM-3.3, 11.1).

Formulas come from agent/insurance_calculations.py and thresholds from agent/policy_insurance.py. Coverage and
valuation age are in checks_loss_run_coverage.py.

- loss_run_line_balance (HM-3.3, HM-11.1): on every claim line, paid + reserve must equal incurred within $1.00
  (insurance_calculations.claim_out_of_balance). A line that does not is a document integrity indicator.
- loss_run_totals (HM-11.1): each loss run's printed summary must equal its claim lines: the claim count exactly,
  and total paid, reserve and incurred within $1.00. A loss run whose summary is not found is skipped.
- claims_history_consistency (HM-3.3): the claim count and total incurred stated on the application against the
  loss runs. A loss run contributes its claim lines, or its printed summary when no claim line was found. Any
  difference in claim count, or a total incurred difference above 10.0% of the loss-run total
  (insurance_calculations.incurred_difference_pct), fails. The values also count the current-term claims (date
  of loss on or after the current-term start) for the explanation rules. As in the agent's SYNTHETIC
  submissions, the current term starts one year before the proposed effective date.
"""

from datetime import date

from uwcopilot.agent.applications import Claim
from uwcopilot.agent.insurance_calculations import (
    RECONCILIATION_TOLERANCE_USD,
    claim_out_of_balance,
    incurred_difference_pct,
)
from uwcopilot.agent.policy_insurance import MAX_INCURRED_DIFFERENCE_PCT
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
from uwcopilot.docintel.schemas import InsuranceSubmission, LossRun, LossRunClaim, casefold_key

PLACEHOLDER_LOSS_DATE = date(2000, 1, 1)  # claim_out_of_balance reads only paid, reserve and incurred


def years_before(day: date, years: int) -> date:
    """The same calendar day `years` years earlier (29 February becomes 28 February)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def carrier_label(loss_run: LossRun) -> str:
    """A loss run label for messages: its document id (carrier names stay in the evidence)."""
    return loss_run.doc_id


def claim_key(claim: LossRunClaim) -> str | None:
    """A claim's row key: its claim number as printed, stripped."""
    return None if not claim.claim_number.found else str(claim.claim_number.value).strip()


def claim_evidence(claim: LossRunClaim, doc_id: str, names: tuple[str, ...]) -> list[Evidence]:
    """Evidence for the named claim fields (key = claim number)."""
    key = claim_key(claim)
    return found_only([cite(getattr(claim, name), f"loss_run.claims.{name}", key, doc_id) for name in names])


def line_out_of_balance(claim: LossRunClaim) -> bool:
    """HM-3.3 through insurance_calculations.claim_out_of_balance (False when a value is missing)."""
    if not (claim.paid.found and claim.reserve.found and claim.incurred.found):
        return False
    model = Claim.model_construct(
        date_of_loss=PLACEHOLDER_LOSS_DATE,
        line="property",
        paid=float(claim.paid.value),
        reserve=float(claim.reserve.value),
        incurred=float(claim.incurred.value),
    )
    return claim_out_of_balance(model)


def check_line_balance(submission: InsuranceSubmission) -> CheckResult:
    """HM-3.3 / HM-11.1: paid + reserve = incurred on every claim line."""
    refs = INSURANCE_REFS["loss_run_line_balance"]
    issues = []
    evidence = []
    lines_checked = 0
    for loss_run in submission.loss_runs:
        for claim in loss_run.claims:
            if not (claim.paid.found and claim.reserve.found and claim.incurred.found):
                continue
            lines_checked += 1
            evidence.extend(claim_evidence(claim, loss_run.doc_id, ("incurred",)))
            if not line_out_of_balance(claim):
                continue
            paid, reserve, incurred = (float(claim.paid.value), float(claim.reserve.value), float(claim.incurred.value))
            difference = round(paid + reserve - incurred, 2)
            issues.append(
                CheckIssue(
                    issue="paid_reserve_incurred",
                    entity=f"carrier:{casefold_key(loss_run.carrier) or loss_run.doc_id}",
                    fields=["loss_run.claims.paid", "loss_run.claims.reserve", "loss_run.claims.incurred"],
                    side_a=claim_evidence(claim, loss_run.doc_id, ("paid", "reserve")),
                    side_b=claim_evidence(claim, loss_run.doc_id, ("incurred",)),
                    data={
                        "claim_number": claim_key(claim),
                        "paid": paid,
                        "reserve": reserve,
                        "incurred": incurred,
                        "difference": difference,
                    },
                    message=f"On {carrier_label(loss_run)} claim {claim_key(claim)} shows paid ${paid:,.2f} + reserve "
                    f"${reserve:,.2f} but incurred ${incurred:,.2f}.",
                )
            )
    values = {"claim_lines_checked": lines_checked, "tolerance_usd": RECONCILIATION_TOLERANCE_USD}
    if issues:
        message = "Claim lines do not balance (integrity indicator): " + " ".join(issue.message for issue in issues)
        return make_result("loss_run_line_balance", refs, "fail", message, values, evidence, issues)
    if lines_checked == 0:
        message = "No claim line with paid, reserve and incurred was found."
        evidence = [document_evidence(item, "document", "loss_run") for item in submission.loss_runs]
        return make_result("loss_run_line_balance", refs, "not_evaluated", message, values, evidence)
    message = f"All {lines_checked} claim lines balance within $1.00."
    return make_result("loss_run_line_balance", refs, "pass", message, values, evidence)


def summary_differences(loss_run: LossRun) -> dict[str, tuple[float, float]]:
    """Summary item -> (printed, sum of the claim lines) for every summary value that disagrees with the lines."""
    claims = loss_run.claims
    summary = loss_run.summary
    differences: dict[str, tuple[float, float]] = {}
    if summary.claim_count.found and int(summary.claim_count.value) != len(claims):
        differences["claim_count"] = (float(summary.claim_count.value), float(len(claims)))
    for total_name, line_name in (("total_paid", "paid"), ("total_reserve", "reserve"), ("total_incurred", "incurred")):
        printed = getattr(summary, total_name)
        if not printed.found:
            continue
        line_sum = round(sum(float(getattr(claim, line_name).value or 0.0) for claim in claims), 2)
        if abs(float(printed.value) - line_sum) > RECONCILIATION_TOLERANCE_USD:
            differences[total_name] = (float(printed.value), line_sum)
    return differences


def summary_evidence(loss_run: LossRun) -> list[Evidence]:
    """Evidence for the printed summary block (key = carrier)."""
    key = casefold_key(loss_run.carrier)
    names = ("claim_count", "total_paid", "total_reserve", "total_incurred")
    summary = loss_run.summary
    return found_only(
        [cite(getattr(summary, name), f"loss_run.summary.{name}", key, loss_run.doc_id) for name in names]
    )


def check_totals(submission: InsuranceSubmission) -> CheckResult:
    """HM-11.1: each loss run's summary equals its claim lines (see the module docstring)."""
    refs = INSURANCE_REFS["loss_run_totals"]
    issues = []
    evidence = []
    checked = 0
    for loss_run in submission.loss_runs:
        printed = summary_evidence(loss_run)
        if not printed:
            continue
        checked += 1
        evidence.extend(printed)
        differences = summary_differences(loss_run)
        if not differences:
            continue
        rows = []
        for claim in loss_run.claims:
            rows.extend(claim_evidence(claim, loss_run.doc_id, ("incurred",)))
        parts = [f"{name} {pair[0]:,.2f} vs lines {pair[1]:,.2f}" for name, pair in differences.items()]
        issues.append(
            CheckIssue(
                issue="summary_vs_lines",
                entity=f"carrier:{casefold_key(loss_run.carrier) or loss_run.doc_id}",
                fields=[f"loss_run.summary.{name}" for name in differences] + ["loss_run.claims.incurred"],
                side_a=printed,
                side_b=rows,
                data={
                    "differences": {name: list(pair) for name, pair in differences.items()},
                    "claim_lines": len(loss_run.claims),
                },
                message=f"The summary on {carrier_label(loss_run)} does not equal its claim lines: "
                + "; ".join(parts)
                + ".",
            )
        )
    values = {"loss_runs_checked": checked}
    if issues:
        message = "Loss-run totals differ from the claim lines (integrity indicator): " + " ".join(
            issue.message for issue in issues
        )
        return make_result("loss_run_totals", refs, "fail", message, values, evidence, issues)
    if checked == 0:
        return make_result("loss_run_totals", refs, "not_evaluated", "No loss-run summary was found.", values, [])
    message = f"The summary of each of the {checked} loss run(s) equals its claim lines."
    return make_result("loss_run_totals", refs, "pass", message, values, evidence)


def current_term_start(submission: InsuranceSubmission) -> date | None:
    """Start of the current policy term: one year before the proposed effective date (None if unknown)."""
    application = submission.application
    if application is None or not application.proposed_effective_date.found:
        return None
    return years_before(application.proposed_effective_date.value, 1)


def loss_run_history(submission: InsuranceSubmission) -> tuple[int, float, int | None, list[Evidence]]:
    """(claim count, total incurred, current-term claim count, evidence) over all loss runs (module docstring)."""
    start = current_term_start(submission)
    count = 0
    incurred = 0.0
    current: int | None = None if start is None else 0
    evidence: list[Evidence] = []
    for loss_run in submission.loss_runs:
        if loss_run.claims:
            for claim in loss_run.claims:
                count += 1
                incurred += float(claim.incurred.value or 0.0)
                evidence.extend(claim_evidence(claim, loss_run.doc_id, ("incurred",)))
                loss_date = claim.date_of_loss.value
                if current is not None and loss_date is not None and start is not None and loss_date >= start:
                    current += 1
            continue
        summary = loss_run.summary
        count += int(summary.claim_count.value or 0)
        incurred += float(summary.total_incurred.value or 0.0)
        evidence.extend(summary_evidence(loss_run))
    return count, round(incurred, 2), current, evidence


def check_claims_history(submission: InsuranceSubmission) -> CheckResult:
    """HM-3.3: application claim count and total incurred vs the loss runs."""
    refs = INSURANCE_REFS["claims_history_consistency"]
    application = submission.application
    if application is None or not submission.loss_runs:
        message = "The application and at least one loss run are needed."
        return make_result("claims_history_consistency", refs, "not_evaluated", message, {}, [])
    stated = found_only(
        [
            cite(application.stated_claim_count, "application.stated_claim_count", None, application.doc_id),
            cite(application.stated_total_incurred, "application.stated_total_incurred", None, application.doc_id),
        ]
    )
    count, incurred, current, run_evidence = loss_run_history(submission)
    if not application.stated_claim_count.found or not application.stated_total_incurred.found:
        message = "The application's claim count or total incurred was not found."
        return make_result("claims_history_consistency", refs, "not_evaluated", message, {}, stated + run_evidence)
    stated_count = int(application.stated_claim_count.value)
    stated_incurred = float(application.stated_total_incurred.value)
    difference_pct = incurred_difference_pct(stated_incurred, incurred)
    values = {
        "stated_claim_count": stated_count,
        "loss_run_claim_count": count,
        "stated_total_incurred": stated_incurred,
        "loss_run_total_incurred": incurred,
        "incurred_difference_pct": difference_pct,
        "current_term_claim_count": current,
    }
    count_differs = stated_count != count
    if difference_pct is None:
        incurred_differs = stated_incurred > RECONCILIATION_TOLERANCE_USD
    else:
        incurred_differs = difference_pct > MAX_INCURRED_DIFFERENCE_PCT
    if not count_differs and not incurred_differs:
        message = f"The application's {stated_count} claim(s) and total incurred match the loss runs."
        return make_result("claims_history_consistency", refs, "pass", message, values, stated + run_evidence)
    shown_pct = "n/a" if difference_pct is None else f"{difference_pct:.1f}%"
    issue = CheckIssue(
        issue="history_differs",
        entity="business",
        fields=["application.stated_claim_count", "application.stated_total_incurred", "loss_run.claims.incurred"],
        side_a=stated,
        side_b=run_evidence,
        data=dict(values),
        message=f"The application states {stated_count} claim(s) and ${stated_incurred:,.2f} incurred; the loss runs "
        f"show {count} claim(s) and ${incurred:,.2f} (incurred difference {shown_pct}).",
    )
    return make_result("claims_history_consistency", refs, "fail", issue.message, values, [], [issue])
