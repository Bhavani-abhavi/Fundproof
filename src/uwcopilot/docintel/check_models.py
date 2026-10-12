"""Result models shared by the document checks (checks.py) and the contradiction builder (contradictions.py).

A check never uses an LLM. It reads values from the extraction (docintel/schemas.py), computes with the agent's
formulas, and returns a CheckResult whose message is built from the values. Every value a check used is cited as
Evidence: the document, page and line it was printed on, the schema field path and row key (the same field and
key rules as schemas.flatten_fields), the value and the source text.

A failing check also lists CheckIssues: each issue is one concrete disagreement with the evidence on each side and
the numbers the explanation rules need. contradictions.py turns issues into Contradictions.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from uwcopilot.docintel.schemas import Cited, plain

CheckStatus = Literal["pass", "fail", "not_evaluated"]
Severity = Literal["high", "medium", "low"]
EvidenceValue = str | float | int | bool | None


class Evidence(BaseModel):
    """One cited value a check or contradiction relies on."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None
    page: int | None
    line: int | None
    field: str
    key: str | None = None
    value: EvidenceValue
    source_text: str | None = None


class CheckIssue(BaseModel):
    """One disagreement found by a check: what it is, which entity it concerns, and the evidence on each side."""

    model_config = ConfigDict(extra="forbid")

    issue: str
    entity: str
    fields: list[str]
    side_a: list[Evidence]
    side_b: list[Evidence]
    # Evidence that is on neither side but supports an explanation (for example the deposit rows behind a gap).
    supporting: list[Evidence] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    message: str


class CheckResult(BaseModel):
    """The outcome of one deterministic document check."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    title: str
    status: CheckStatus
    severity: Severity
    guideline_refs: list[str]
    message: str
    values: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)
    issues: list[CheckIssue] = Field(default_factory=list)


def cite(cited: Cited, field: str, key: str | None = None, doc_id: str | None = None) -> Evidence | None:
    """Evidence for one extracted value (None when the extractor did not find it).

    doc_id is the containing document's id, used when the value itself carries none.
    """
    if not cited.found:
        return None
    return Evidence(
        doc_id=cited.doc_id or doc_id,
        page=cited.page,
        line=cited.line,
        field=field,
        key=key,
        value=plain(cited.value),
        source_text=cited.source_text,
    )


def cite_each(items: list[tuple[Cited, str, str | None, str | None]]) -> list[Evidence]:
    """Evidence for every (cited, field, key, doc_id) item whose value was found."""
    evidence = []
    for cited, field, key, doc_id in items:
        item = cite(cited, field, key, doc_id)
        if item is not None:
            evidence.append(item)
    return evidence


def document_evidence(model: BaseModel, field: str, value: EvidenceValue, key: str | None = None) -> Evidence:
    """Evidence that a document is present: its doc id and the page and line of its first found value."""
    doc_id = getattr(model, "doc_id", None)
    for name in type(model).model_fields:
        item = getattr(model, name)
        if isinstance(item, Cited) and item.found and item.page is not None:
            return Evidence(
                doc_id=item.doc_id or doc_id,
                page=item.page,
                line=item.line,
                field=field,
                key=key,
                value=value,
                source_text=item.source_text,
            )
    return Evidence(doc_id=doc_id, page=None, line=None, field=field, key=key, value=value)


def unique_evidence(items: list[Evidence]) -> list[Evidence]:
    """The evidence list without exact duplicates, in first-seen order."""
    seen: set[str] = set()
    result = []
    for item in items:
        marker = item.model_dump_json()
        if marker not in seen:
            seen.add(marker)
            result.append(item)
    return result


def make_result(
    check_id: str,
    guideline_refs: list[str],
    status: CheckStatus,
    message: str,
    values: dict[str, Any],
    evidence: list[Evidence],
    issues: list[CheckIssue] | None = None,
    severity: Severity | None = None,
) -> CheckResult:
    """Build a CheckResult with the check's registered title and default severity.

    The result's evidence is the given evidence plus every issue's evidence, without duplicates.
    """
    issue_list = issues or []
    all_evidence = list(evidence)
    for issue in issue_list:
        all_evidence.extend(issue.side_a)
        all_evidence.extend(issue.side_b)
        all_evidence.extend(issue.supporting)
    return CheckResult(
        check_id=check_id,
        title=CHECK_TITLES[check_id],
        status=status,
        severity=severity or CHECK_SEVERITY[check_id],
        guideline_refs=guideline_refs,
        message=message,
        values=values,
        evidence=unique_evidence(all_evidence),
        issues=issue_list,
    )


# Stable check ids (the "check:<id>" levers in config/resolution_templates.yaml use them) and their titles.
CHECK_TITLES: dict[str, str] = {
    "required_documents": "Required documents are in the packet",
    "statement_months_complete": "Twelve consecutive, recent statement months",
    "balance_continuity": "Statement balances continue and add up",
    "revenue_vs_deposits": "Stated revenue versus verified bank deposits",
    "name_consistency": "Business name is the same on every document",
    "fein_consistency": "FEIN is the same on every document",
    "owner_consistency": "Owners and ownership percentages match",
    "debt_payment_consistency": "Debt schedule payments match the statements",
    "account_conduct": "Account conduct computed from the transactions",
    "tax_return_vs_pnl": "Tax return receipts versus P&L revenue",
    "loss_run_line_balance": "Loss run claim lines balance",
    "loss_run_totals": "Loss run totals equal the claim lines",
    "claims_history_consistency": "Application loss history matches the loss runs",
    "tiv_consistency": "Application TIV matches the property schedule",
    "loss_run_coverage": "Loss runs cover the experience period and every prior carrier",
    "loss_run_valuation_age": "Loss runs are valued within 90 days of the effective date",
    "embedded_instructions": "No instruction-like text in the documents",
}
# Default severity when the check fails: high = integrity or blocking referral, medium = referral or request for
# information, low = informational or already resolved by a guideline rule.
CHECK_SEVERITY: dict[str, Severity] = {
    "required_documents": "medium",
    "statement_months_complete": "medium",
    "balance_continuity": "high",
    "revenue_vs_deposits": "medium",
    "name_consistency": "medium",
    "fein_consistency": "medium",
    "owner_consistency": "high",
    "debt_payment_consistency": "low",
    "account_conduct": "low",
    "tax_return_vs_pnl": "high",
    "loss_run_line_balance": "high",
    "loss_run_totals": "high",
    "claims_history_consistency": "medium",
    "tiv_consistency": "medium",
    "loss_run_coverage": "medium",
    "loss_run_valuation_age": "medium",
    "embedded_instructions": "high",
}
