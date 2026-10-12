"""Northwind (credit) document checks over a LoanPacket extraction. Thresholds come from agent/policy_credit.py.

- required_documents (NBC-3.1): which checklist items the packet holds (see documents_provided) against the base
  list plus the conditional items (CPA-reviewed financials above $500,000, collateral documents for secured
  requests or above $250,000, payoff letters for a refinance). No packet document type exists for CPA-reviewed
  financials or payoff letters, so they are always missing when required (values: no_packet_document_type).
- statement_months_complete (NBC-3.1, 3.2, 3.5): the 12 most recent consecutive months must be present and the
  most recent statement must end no more than 60 days before the application date. The window ends at the most
  recent statement month, or at the month before the application month when that window holds at least as many
  of the statements (so a missing latest month is named as missing, as NBC-3.1 asks for the MOST RECENT months).
- balance_continuity (NBC-9.1): opening + deposits - withdrawals = closing within $1.00 on every statement; the
  printed totals equal the sum of the rows within $1.00; each month opens at the previous month's closing balance
  within $1.00; one account number on every statement.
- account_conduct (NBC-3.5, informational): NSF events, negative months and average daily balance computed from
  the transactions (docintel/deposits.py) against the NBC-3.5 triggers. The reference policy raises the referral
  from the application fields; this check shows the computed values with their citations. It never creates a
  contradiction.
- tax_return_vs_pnl (NBC-9.1): the most recent tax return's gross receipts against the P&L revenue "for the same
  year". The summary's tax year is treated as a calendar year, so the check runs only when the P&L period ends on
  31 December of that year; otherwise it is not evaluated.
revenue_vs_deposits and debt_payment_consistency are in checks_revenue.py.
"""

from datetime import date

from uwcopilot.agent.credit_calculations import month_index, proposed_annual_debt_service, round_pct
from uwcopilot.agent.policy_credit import (
    BASE_DOCUMENTS,
    COLLATERAL_REQUIRED_ABOVE,
    CPA_FINANCIALS_ABOVE,
    MAX_NEGATIVE_MONTHS,
    MAX_NSF_EVENTS,
    MIN_BALANCE_SHARE_OF_DEBT_SERVICE,
    PRINCIPAL_OWNERSHIP_PCT,
)
from uwcopilot.docintel.check_models import CheckIssue, CheckResult, Evidence, cite, document_evidence, make_result
from uwcopilot.docintel.checks_identity import group_by_key, majority_key
from uwcopilot.docintel.deposits import (
    average_daily_balance,
    classify_rows,
    month_of,
    negative_balance_months,
    nsf_events,
    period_bounds,
    sorted_by_month,
    statements_in_window,
)
from uwcopilot.docintel.normalize import canonical_fein, canonical_name
from uwcopilot.docintel.schemas import BankStatement, LoanPacket, transaction_key

REFS: dict[str, list[str]] = {
    "required_documents": ["NBC-3.1"],
    "statement_months_complete": ["NBC-3.1", "NBC-3.2", "NBC-3.5"],
    "balance_continuity": ["NBC-9.1"],
    "revenue_vs_deposits": ["NBC-3.2", "NBC-3.3", "NBC-3.4"],
    "name_consistency": ["NBC-3.1", "NBC-6.2", "NBC-9.1"],
    "fein_consistency": ["NBC-3.1", "NBC-6.2"],
    "owner_consistency": ["NBC-6.1"],
    "debt_payment_consistency": ["NBC-3.1", "NBC-4.1"],
    "account_conduct": ["NBC-3.5"],
    "tax_return_vs_pnl": ["NBC-9.1"],
    "embedded_instructions": ["NBC-9.2"],
}
MONTHS_REQUIRED = 12  # NBC-3.1 / NBC-3.2
MAX_STATEMENT_AGE_DAYS = 60  # NBC-3.1: the most recent statement ends no more than 60 days before the application
BALANCE_TOLERANCE_USD = 1.00  # NBC-9.1: opening + credits - debits vs closing, more than $1.00 is an indicator
TAX_RETURN_DIFFERENCE_PCT = 10.0  # NBC-9.1: tax return receipts vs P&L revenue for the same year
NO_PACKET_DOCUMENT_TYPE = ("cpa_reviewed_financials", "payoff_letters")


def month_label(index: int) -> str:
    """Inverse of credit_calculations.month_index: a running month number back to "YYYY-MM"."""
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def statement_label(statement: BankStatement) -> str:
    """A statement label for messages, for example D03 (2026-03)."""
    return f"{statement.doc_id} ({month_of(statement) or 'month unknown'})"


def product_code(text: str | None) -> str | None:
    """The agent's product code from the application's product text (None when it is not recognized)."""
    key = canonical_name(text)
    if "line" in key and "credit" in key:
        return "line_of_credit"
    if "equipment" in key:
        return "equipment_loan"
    if "term" in key:
        return "term_loan"
    return None


def principals(packet: LoanPacket) -> list[tuple[int, str]]:
    """(owner position, canonical name) of each NBC-1.2 principal: a person with 20% or more, or the control person."""
    if packet.application is None:
        return []
    result = []
    for index, owner in enumerate(packet.application.owners, start=1):
        if owner.entity_type.value == "organization" or not owner.name.found:
            continue
        pct = owner.ownership_pct.value
        if (pct is not None and pct >= PRINCIPAL_OWNERSHIP_PCT) or owner.is_control_person.value is True:
            result.append((index, canonical_name(owner.name.value)))
    return result


def consecutive_months(packet: LoanPacket) -> bool:
    """True when the 12 most recent statement months present are consecutive."""
    indices = sorted({month_index(month_of(statement) or "") for statement in sorted_by_month(packet.statements)})
    recent = indices[-MONTHS_REQUIRED:]
    return len(recent) == MONTHS_REQUIRED and recent[-1] - recent[0] == MONTHS_REQUIRED - 1


def documents_provided(packet: LoanPacket) -> dict[str, list[Evidence]]:
    """Checklist code -> evidence of the documents that satisfy it (only the codes the packet satisfies)."""
    provided: dict[str, list[Evidence]] = {}
    if packet.application is not None:
        provided["signed_application"] = [document_evidence(packet.application, "document", "loan_application")]
    if consecutive_months(packet):
        provided["bank_statements_12_months"] = [
            document_evidence(statement, "document", "bank_statement", month_of(statement))
            for statement in sorted_by_month(packet.statements)[-MONTHS_REQUIRED:]
        ]
    years = {item.tax_year.value for item in packet.tax_returns if item.tax_year.value is not None}
    if len(years) >= 2:
        provided["business_tax_returns_2_years"] = [
            document_evidence(item, "document", "tax_return_summary") for item in packet.tax_returns
        ]
    if packet.profit_and_loss is not None and packet.balance_sheet is not None:
        provided["ytd_pnl_and_balance_sheet"] = [
            document_evidence(packet.profit_and_loss, "document", "profit_and_loss"),
            document_evidence(packet.balance_sheet, "document", "balance_sheet"),
        ]
    if packet.debt_schedule is not None:
        provided["debt_schedule"] = [document_evidence(packet.debt_schedule, "document", "debt_schedule")]
    id_names = {canonical_name(item.name.value) for item in packet.photo_ids if item.name.found}
    needed = {name for _, name in principals(packet)}
    if (needed and needed <= id_names) or (not needed and packet.photo_ids):
        provided["principal_photo_ids"] = [document_evidence(item, "document", "photo_id") for item in packet.photo_ids]
    if packet.formation is not None:
        provided["formation_documents"] = [document_evidence(packet.formation, "document", "formation_document")]
        if packet.formation.members:
            provided["beneficial_ownership_certification"] = list(provided["formation_documents"])
    if packet.collateral:
        provided["collateral_documents"] = [
            document_evidence(item, "document", "collateral_valuation") for item in packet.collateral
        ]
    return provided


def required_documents_list(packet: LoanPacket) -> list[str]:
    """NBC-3.1 base documents plus the conditional ones (mirrors policy_credit.required_documents)."""
    required = list(BASE_DOCUMENTS)
    application = packet.application
    amount = None if application is None else application.requested_amount.value
    if amount is not None and amount > CPA_FINANCIALS_ABOVE:
        required.append("cpa_reviewed_financials")
    if packet.collateral or (amount is not None and amount > COLLATERAL_REQUIRED_ABOVE):
        required.append("collateral_documents")
    if application is not None and "refinanc" in canonical_name(application.purpose.value):
        required.append("payoff_letters")
    return required


def check_required_documents(packet: LoanPacket) -> CheckResult:
    """NBC-3.1 required documents."""
    provided = documents_provided(packet)
    required = required_documents_list(packet)
    missing = [code for code in required if code not in provided]
    values = {
        "required": required,
        "provided": sorted(provided),
        "missing": missing,
        "no_packet_document_type": [code for code in NO_PACKET_DOCUMENT_TYPE if code in required],
    }
    evidence = [item for items in provided.values() for item in items]
    if missing:
        message = f"Required documents are missing: {', '.join(missing)}."
        return make_result("required_documents", REFS["required_documents"], "fail", message, values, evidence)
    message = f"All {len(required)} required documents are in the packet."
    return make_result("required_documents", REFS["required_documents"], "pass", message, values, evidence)


def window_end(present: list[int], application_date: date | None) -> int:
    """The last month of the required 12-month window (see the module docstring)."""
    latest = present[-1]
    if application_date is None:
        return latest
    before_application = month_index(application_date.strftime("%Y-%m")) - 1
    if before_application <= latest:
        return latest

    def held(end: int) -> int:
        return sum(1 for index in present if end - MONTHS_REQUIRED < index <= end)

    return before_application if held(before_application) >= held(latest) else latest


def check_statement_months(packet: LoanPacket) -> CheckResult:
    """NBC-3.1 / 3.2 / 3.5: 12 consecutive recent months, the most recent within 60 days of the application."""
    refs = REFS["statement_months_complete"]
    application_date = None if packet.application is None else packet.application.application_date.value
    evidence = []
    if packet.application is not None:
        item = cite(packet.application.application_date, "application.application_date", None, packet.application.doc_id)
        evidence.extend([item] if item else [])
    ordered = sorted_by_month(packet.statements)
    for statement in ordered:
        item = cite(statement.period_start, "statement.period_start", month_of(statement), statement.doc_id)
        evidence.append(item or document_evidence(statement, "document", "bank_statement", month_of(statement)))
    if not ordered:
        message = "No bank statements with a known month were found; 12 months are required."
        return make_result("statement_months_complete", refs, "fail", message, {"months_present": []}, evidence)
    present = sorted({month_index(month_of(statement) or "") for statement in ordered})
    end = window_end(present, application_date)
    window = list(range(end - MONTHS_REQUIRED + 1, end + 1))
    missing = [month_label(index) for index in window if index not in present]
    values: dict = {
        "months_present": [month_label(index) for index in present],
        "window": [month_label(window[0]), month_label(window[-1])],
        "missing_months": missing,
        "max_days_before_application": MAX_STATEMENT_AGE_DAYS,
    }
    problems = []
    if missing:
        problems.append(f"missing statement month(s) {', '.join(missing)}")
    bounds = period_bounds(ordered[-1])
    if bounds is not None and application_date is not None:
        age = (application_date - bounds[1]).days
        values["most_recent_period_end"] = bounds[1].isoformat()
        values["days_before_application"] = age
        if age > MAX_STATEMENT_AGE_DAYS:
            problems.append(f"the most recent statement ends {age} days before the application (more than 60)")
    if problems:
        message = "Statements are incomplete: " + "; ".join(problems) + "."
        return make_result("statement_months_complete", refs, "fail", message, values, evidence)
    message = f"12 consecutive statement months from {values['window'][0]} to {values['window'][1]} are present."
    return make_result("statement_months_complete", refs, "pass", message, values, evidence)


def arithmetic_issue(statement: BankStatement) -> CheckIssue | None:
    """Opening + deposits - withdrawals vs closing on one statement (None when it adds up or a value is missing)."""
    parts = (statement.opening_balance, statement.total_deposits, statement.total_withdrawals)
    if any(not part.found for part in parts) or not statement.closing_balance.found:
        return None
    expected = round(parts[0].value + parts[1].value - parts[2].value, 2)
    difference = round(statement.closing_balance.value - expected, 2)
    if abs(difference) <= BALANCE_TOLERANCE_USD:
        return None
    month = month_of(statement)
    names = ("statement.opening_balance", "statement.total_deposits", "statement.total_withdrawals")
    side_a = [cite(part, name, month, statement.doc_id) for part, name in zip(parts, names, strict=True)]
    side_b = [cite(statement.closing_balance, "statement.closing_balance", month, statement.doc_id)]
    return CheckIssue(
        issue="arithmetic",
        entity="business",
        fields=[*names, "statement.closing_balance"],
        side_a=[item for item in side_a if item],
        side_b=[item for item in side_b if item],
        data={"month": month, "expected_closing": expected, "printed_closing": statement.closing_balance.value,
              "difference": difference},
        message=f"On {statement_label(statement)} opening + deposits - withdrawals = ${expected:,.2f} but the "
        f"closing balance is ${statement.closing_balance.value:,.2f}.",
    )


def row_total_issue(statement: BankStatement, kind: str) -> CheckIssue | None:
    """Printed total deposits (or withdrawals) vs the sum of the rows (None when they agree or cannot be compared)."""
    total = statement.total_deposits if kind == "deposit" else statement.total_withdrawals
    rows = [row for row in classify_rows(statement) if row.kind == kind]
    if not total.found or not rows:
        return None
    row_sum = round(sum(row.amount for row in rows), 2)
    difference = round(total.value - row_sum, 2)
    if abs(difference) <= BALANCE_TOLERANCE_USD:
        return None
    month = month_of(statement)
    total_field = "statement.total_deposits" if kind == "deposit" else "statement.total_withdrawals"
    row_field = f"statement.transactions.{kind}"
    side_b = []
    for row in rows:
        amount = row.transaction.deposit if kind == "deposit" else row.transaction.withdrawal
        item = cite(amount, row_field, transaction_key(row.transaction), statement.doc_id)
        side_b.extend([item] if item else [])
    return CheckIssue(
        issue=f"{kind}_total",
        entity="business",
        fields=[total_field, row_field],
        side_a=[cite(total, total_field, month, statement.doc_id)],
        side_b=side_b,
        data={"month": month, "printed_total": total.value, "row_sum": row_sum, "difference": difference},
        message=f"On {statement_label(statement)} the printed total {kind}s are ${total.value:,.2f} but the rows add "
        f"up to ${row_sum:,.2f}.",
    )


def continuity_issue(previous: BankStatement, current: BankStatement) -> CheckIssue | None:
    """A month that does not open at the previous month's closing balance (consecutive months only)."""
    if month_index(month_of(current) or "") - month_index(month_of(previous) or "") != 1:
        return None
    if not previous.closing_balance.found or not current.opening_balance.found:
        return None
    difference = round(current.opening_balance.value - previous.closing_balance.value, 2)
    if abs(difference) <= BALANCE_TOLERANCE_USD:
        return None
    return CheckIssue(
        issue="discontinuity",
        entity="business",
        fields=["statement.closing_balance", "statement.opening_balance"],
        side_a=[cite(previous.closing_balance, "statement.closing_balance", month_of(previous), previous.doc_id)],
        side_b=[cite(current.opening_balance, "statement.opening_balance", month_of(current), current.doc_id)],
        data={"month": month_of(current), "previous_closing": previous.closing_balance.value,
              "opening": current.opening_balance.value, "difference": difference},
        message=f"{statement_label(current)} opens at ${current.opening_balance.value:,.2f} but "
        f"{statement_label(previous)} closed at ${previous.closing_balance.value:,.2f}.",
    )


def account_number_issues(statements: list[BankStatement]) -> list[CheckIssue]:
    """Statements that show a different account number from the majority (NBC-9.1)."""
    sources = []
    for statement in statements:
        item = cite(statement.account_number, "statement.account_number", month_of(statement), statement.doc_id)
        sources.extend([item] if item else [])
    groups = group_by_key(sources, canonical_fein)
    if len(groups) < 2:
        return []
    main = majority_key(groups)
    issues = []
    for key, items in groups.items():
        if key != main:
            issues.append(
                CheckIssue(
                    issue="account_number_change",
                    entity="business",
                    fields=["statement.account_number"],
                    side_a=groups[main],
                    side_b=items,
                    data={"majority_account": main, "odd_account": key},
                    message=f"{len(items)} statement(s) show a different account number from the other statements.",
                )
            )
    return issues


def check_balance_continuity(packet: LoanPacket) -> CheckResult:
    """NBC-9.1 statement arithmetic, row totals, month-to-month continuity and account number."""
    refs = REFS["balance_continuity"]
    ordered = sorted_by_month(packet.statements)
    if not ordered:
        return make_result("balance_continuity", refs, "not_evaluated", "No bank statements were found.", {}, [])
    issues: list[CheckIssue] = []
    for statement in ordered:
        for issue in (arithmetic_issue(statement), row_total_issue(statement, "deposit"),
                      row_total_issue(statement, "withdrawal")):
            issues.extend([issue] if issue else [])
    for previous, current in zip(ordered, ordered[1:], strict=False):
        issue = continuity_issue(previous, current)
        issues.extend([issue] if issue else [])
    issues.extend(account_number_issues(ordered))
    evidence = []
    for statement in ordered:
        for name in ("opening_balance", "total_deposits", "total_withdrawals", "closing_balance"):
            item = cite(getattr(statement, name), f"statement.{name}", month_of(statement), statement.doc_id)
            evidence.extend([item] if item else [])
    values = {"statements": len(ordered), "issue_codes": [issue.issue for issue in issues],
              "tolerance_usd": BALANCE_TOLERANCE_USD}
    if issues:
        message = "Statement integrity indicators: " + " ".join(issue.message for issue in issues)
        return make_result("balance_continuity", refs, "fail", message, values, evidence, issues)
    if not evidence:
        message = "No statement balances were found to check."
        return make_result("balance_continuity", refs, "not_evaluated", message, values, evidence)
    message = f"Balances add up and continue across all {len(ordered)} statements."
    return make_result("balance_continuity", refs, "pass", message, values, evidence)


def conduct_evidence(statements: list[BankStatement]) -> list[Evidence]:
    """NSF fee rows (last 6 months) and the closing balances of the last 12 months."""
    evidence = []
    for statement in statements_in_window(statements, 6):
        for row in classify_rows(statement):
            if row.category == "nsf_fee":
                item = cite(row.transaction.withdrawal, "statement.transactions.withdrawal",
                            transaction_key(row.transaction), statement.doc_id)
                evidence.extend([item] if item else [])
    for statement in statements_in_window(statements, 12):
        item = cite(statement.closing_balance, "statement.closing_balance", month_of(statement), statement.doc_id)
        evidence.extend([item] if item else [])
    return evidence


def proposed_service(packet: LoanPacket) -> tuple[float | None, list[Evidence]]:
    """NBC-4.1 proposed annual debt service from the application (None when an input is missing)."""
    application = packet.application
    if application is None:
        return None, []
    product = product_code(application.product.value)
    inputs = (application.requested_amount, application.note_rate_pct, application.term_months)
    evidence = []
    for cited, name in zip(inputs, ("requested_amount", "note_rate_pct", "term_months"), strict=True):
        item = cite(cited, f"application.{name}", None, application.doc_id)
        evidence.extend([item] if item else [])
    if product is None or any(not cited.found for cited in inputs):
        return None, evidence
    amount, rate, term = (cited.value for cited in inputs)
    return proposed_annual_debt_service(product, float(amount), float(rate), int(term)), evidence


def check_account_conduct(packet: LoanPacket) -> CheckResult:
    """NBC-3.5 conduct figures from the transactions (informational; see the module docstring)."""
    refs = REFS["account_conduct"]
    if not packet.statements:
        return make_result("account_conduct", refs, "not_evaluated", "No bank statements were found.", {}, [])
    nsf = nsf_events(packet.statements)
    negative = negative_balance_months(packet.statements)
    adb = average_daily_balance(packet.statements)
    service, service_evidence = proposed_service(packet)
    floor = None if service is None else round(MIN_BALANCE_SHARE_OF_DEBT_SERVICE * service, 2)
    values = {"nsf_events_last_6_months": nsf, "negative_balance_months_last_12": negative,
              "average_daily_balance_last_3_months": adb, "average_balance_floor": floor}
    triggers = []
    if nsf > MAX_NSF_EVENTS:
        triggers.append(f"{nsf} NSF or overdraft events in the last 6 months (more than {MAX_NSF_EVENTS})")
    if negative > MAX_NEGATIVE_MONTHS:
        triggers.append(f"{negative} of the last 12 months closed negative (3 or more)")
    if adb is not None and floor is not None and adb < floor:
        triggers.append(f"average daily balance ${adb:,.2f} is below 5% of proposed annual debt service "
                        f"(${floor:,.2f})")
    evidence = conduct_evidence(packet.statements) + service_evidence
    summary = f"{nsf} NSF events, {negative} negative months"
    if adb is not None:
        summary += f", average daily balance ${adb:,.2f}"
    if triggers:
        message = f"Account conduct ({summary}) meets NBC-3.5 referral triggers: " + "; ".join(triggers) + "."
        return make_result("account_conduct", refs, "fail", message, values, evidence)
    message = f"Account conduct ({summary}) meets no NBC-3.5 trigger."
    return make_result("account_conduct", refs, "pass", message, values, evidence)


def check_tax_return_vs_pnl(packet: LoanPacket) -> CheckResult:
    """NBC-9.1: most recent tax return gross receipts vs P&L revenue for the same year (within 10.0%)."""
    refs = REFS["tax_return_vs_pnl"]
    pnl = packet.profit_and_loss
    returns = [item for item in packet.tax_returns if item.tax_year.found and item.gross_receipts.found]
    if pnl is None or not returns or not pnl.revenue.found or not pnl.period_end.found or pnl.revenue.value <= 0:
        message = "A P&L revenue figure and a tax return with gross receipts are both needed."
        return make_result("tax_return_vs_pnl", refs, "not_evaluated", message, {}, [])
    latest = max(returns, key=lambda item: item.tax_year.value)
    year, period_end = latest.tax_year.value, pnl.period_end.value
    evidence = [
        cite(latest.gross_receipts, "tax_return.gross_receipts", str(year), latest.doc_id),
        cite(pnl.revenue, "profit_and_loss.revenue", None, pnl.doc_id),
        cite(pnl.period_end, "profit_and_loss.period_end", None, pnl.doc_id),
    ]
    evidence = [item for item in evidence if item]
    values: dict = {"tax_year": year, "pnl_period_end": period_end.isoformat()}
    if not (period_end.year == year and period_end.month == 12 and period_end.day == 31):
        message = f"The P&L period (ended {period_end.isoformat()}) is not tax year {year}; not compared."
        return make_result("tax_return_vs_pnl", refs, "not_evaluated", message, values, evidence)
    receipts, revenue = float(latest.gross_receipts.value), float(pnl.revenue.value)
    difference = round_pct(abs(receipts - revenue) / revenue * 100.0)
    values.update({"gross_receipts": receipts, "pnl_revenue": revenue, "difference_pct": difference})
    if difference <= TAX_RETURN_DIFFERENCE_PCT:
        message = f"Tax year {year} gross receipts are within {difference:.1f}% of the P&L revenue."
        return make_result("tax_return_vs_pnl", refs, "pass", message, values, evidence)
    issue = CheckIssue(
        issue="tax_return_differs",
        entity="business",
        fields=["tax_return.gross_receipts", "profit_and_loss.revenue"],
        side_a=evidence[:1],
        side_b=evidence[1:2],
        data=dict(values),
        message=f"Tax year {year} gross receipts ${receipts:,.2f} differ from the P&L revenue ${revenue:,.2f} by "
        f"{difference:.1f}% (more than 10.0%).",
    )
    return make_result("tax_return_vs_pnl", refs, "fail", issue.message, values, evidence, [issue])
