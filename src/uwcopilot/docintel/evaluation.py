"""Evaluation of the document pipeline against the SYNTHETIC ground truth: per-packet matching and counting.

evaluate_packet(truth, result, documents_provided) compares one PacketResult (docintel/pipeline.py) with the
packet's ground truth and returns a JSON-simple record of COUNTS. evaluation_summary.py adds the counts of many
packets, computes the metrics and their bootstrap confidence intervals. Every function here is deterministic.

The truth is read by attribute only (a truth.PacketTruth works). This module does not import truth.py: the spec
lets only scripts and tests import it, so the pipeline and the evaluation code stay apart from the generator.

What is counted (see each function):
- fields: (field, key, value) rows of schemas.flatten_fields matched with the truth's FieldTruth rows on (field,
  key) with normalized values (normalize_value). A row with the right (field, key) but a wrong value counts as one
  false positive AND one false negative (the usual rule for extraction). Counted per field type (FIELD_TYPES) and
  per page condition: "clean" (text layer) or "scanned" pages of the TRUTH row (a false positive takes the
  condition of the page it cites; one without a page is counted only under "all").
- rows: table rows (transactions, claims, locations) matched by their row key; "exact" rows have every field right;
- citations: for every correctly extracted value, whether the CITED page's truth text lines contain the value in
  one of its printed forms (printed_forms), and whether the cited page is the truth page;
- pages: predicted page type vs the truth document type of every page; documents: truth documents whose exact
  pages and type the pipeline found;
- checklist: for every checklist code of the line, "missing" (required_documents check) vs the truth's
  missing_required_documents, and "provided" (to_application's documents_provided) vs the as-documented application;
- contradictions: detected contradictions matched with injected problems (match_contradictions);
- injections: planted instructions found by the injection scan on their page, and hits on other pages.
"""

import re
from collections import Counter
from datetime import date
from typing import Any, NamedTuple

from uwcopilot.agent.applications import CREDIT_DOCUMENTS, INSURANCE_DOCUMENTS
from uwcopilot.docintel.evaluation_problems import contradiction_counts, injection_counts, match_contradictions
from uwcopilot.docintel.pipeline import PacketResult
from uwcopilot.docintel.schemas import flatten_fields

# fmt: off
FIELD_TYPES: dict[str, tuple[str, ...]] = {
    "money": (
        "accounts_receivable", "annual_payroll", "annual_revenue", "balance", "building_value",
        "business_income_value", "cash", "closing_balance", "contents_value", "cost_of_goods_sold", "deposit",
        "depreciation_amortization", "ebitda", "equity", "gl_earned_premium", "gl_limit_aggregate",
        "gl_limit_per_occurrence", "gl_premium", "gross_receipts", "incurred", "interest_expense", "inventory",
        "long_term_debt", "monthly_payment", "net_income", "opening_balance", "operating_expenses", "paid",
        "property_earned_premium", "property_premium", "requested_amount", "reserve", "revenue",
        "stated_annual_revenue", "stated_tiv", "stated_total_incurred", "tiv", "total_assets", "total_balance",
        "total_deposits", "total_incurred", "total_liabilities", "total_monthly_payment", "total_paid",
        "total_reserve", "total_tiv", "total_withdrawals", "value", "withdrawal",
    ),
    "percent": ("largest_customer_revenue_pct", "note_rate_pct", "ownership_pct", "subcontracted_work_pct"),
    "date": (
        "application_date", "as_of", "date_of_birth", "date_of_loss", "expiry_date", "formation_date",
        "inspection_date", "maturity", "period_end", "period_start", "policy_year_start", "posted_date",
        "proposed_effective_date", "sent_date", "signed_date", "valuation_date", "valued_as_of",
    ),
    "identifier": ("account_number", "claim_number", "fein", "id_number", "location_id", "policy_number"),
    "name": (
        "account_holder", "bank_name", "business_name", "carrier", "entity_name", "insured_name", "legal_name",
        "lender", "name", "named_insured", "recipient", "sender",
    ),
    "code": (
        "business_class", "business_structure", "collateral_type", "construction_class", "country", "coverage",
        "debt_type", "entity_type", "industry", "line", "occupancy", "product", "purpose", "rate_type", "state",
        "status", "title",
    ),
    "integer": (
        "claim_count", "credit_score", "protection_class", "roof_year", "square_feet", "stated_claim_count",
        "stories", "tax_year", "term_months", "year_built", "years_in_business",
    ),
    "boolean": (
        "bankruptcy_within_7_years", "carrier_produced", "foreign_products", "has_subcontractor_coi_tracking",
        "is_control_person", "sprinklered", "tax_lien_unresolved", "to_be_refinanced",
    ),
    "free_text": ("address", "body", "business_address", "description", "remarks", "subject"),
}
# fmt: on
TYPE_OF_LEAF: dict[str, str] = {leaf: kind for kind, leaves in FIELD_TYPES.items() for leaf in leaves}
CONDITIONS: tuple[str, ...] = ("all", "clean", "scanned")
TABLE_ROWS: dict[str, str] = {
    "transactions": "statement.transactions",
    "claims": "loss_run.claims",
    "locations": "property_schedule.locations",
}
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
# Booleans printed as a phrase rather than Yes/No (the same phrases the generator's own test accepts).
BOOLEAN_PHRASES: dict[str, str] = {"is_control_person": "control person", "carrier_produced": "carrier-produced"}


class Row(NamedTuple):
    """One extracted or true value: field path, row key, value, document and page (None when not cited)."""

    field: str
    key: str | None
    value: Any
    doc_id: str | None
    page: int | None


def field_type(field: str) -> str:
    """The type of a field path from its last part ("statement.transactions.deposit" -> "money")."""
    return TYPE_OF_LEAF.get(field.rsplit(".", 1)[-1], "free_text")


def normalize_value(kind: str, value: Any) -> Any:
    """A value as the evaluation compares it: amounts to cents, text casefolded with single spaces.

    Identifiers also lose every space ("BWC 20-1" == "BWC20-1"); dates are compared as ISO strings.
    """
    if value is None:
        return None
    if kind == "boolean":
        return bool(value)
    if kind in ("money", "percent"):
        return round(float(value), 2)
    if kind == "integer":
        return int(value)
    text = " ".join(str(value).split()).casefold()
    if kind == "identifier":
        return text.replace(" ", "")
    return text


def truth_rows(truth: Any) -> list[Row]:
    """The truth's FieldTruth rows as Rows."""
    return [Row(item.field, item.key, item.value, item.doc_id, item.page_number) for item in truth.fields]


def predicted_rows(extraction: Any) -> list[Row]:
    """The extraction flattened (schemas.flatten_fields) into Rows."""
    return [Row(item.field, item.key, item.value, item.doc_id, item.page) for item in flatten_fields(extraction)]


class FieldMatch(NamedTuple):
    """The result of matching predicted rows with truth rows."""

    pairs: list[tuple[Row, Row]]  # (truth row, predicted row) with equal normalized values
    missed: list[Row]  # truth rows without a correct prediction (false negatives)
    extra: list[Row]  # predicted rows without a matching truth row (false positives)


def match_fields(truth: list[Row], predicted: list[Row]) -> FieldMatch:
    """Match rows on (field, key) and normalized value; duplicates are matched one to one, in order."""
    expected: dict[tuple[str, str | None], list[Row]] = {}
    for row in truth:
        expected.setdefault((row.field, row.key), []).append(row)
    pairs: list[tuple[Row, Row]] = []
    extra: list[Row] = []
    for row in predicted:
        candidates = expected.get((row.field, row.key), [])
        kind = field_type(row.field)
        wanted = normalize_value(kind, row.value)
        hit = next((item for item in candidates if normalize_value(kind, item.value) == wanted), None)
        if hit is None:
            extra.append(row)
        else:
            candidates.remove(hit)
            pairs.append((hit, row))
    missed = [row for rows in expected.values() for row in rows]
    return FieldMatch(pairs, missed, extra)


def page_conditions(truth: Any) -> dict[int, str]:
    """{page number: "scanned" or "clean"} from the truth pages."""
    return {page.page_number: ("scanned" if page.scanned else "clean") for page in truth.pages}


def field_counts(match: FieldMatch, conditions: dict[int, str]) -> Counter:
    """tp / fp / fn per condition (all, clean, scanned) and field type (plus "all"), as flat counter names."""
    counts: Counter = Counter()

    def add(row: Row, outcome: str, page: int | None) -> None:
        groups = ["all"]
        if page in conditions:
            groups.append(conditions[page])
        for group in groups:
            for kind in ("all", field_type(row.field)):
                counts[f"fields.{group}.{kind}.{outcome}"] += 1

    for truth_row, _ in match.pairs:
        add(truth_row, "tp", truth_row.page)
    for row in match.missed:
        add(row, "fn", row.page)
    for row in match.extra:
        add(row, "fp", row.page)
    return counts


def row_counts(truth: list[Row], predicted: list[Row], match: FieldMatch) -> Counter:
    """Table rows (TABLE_ROWS) matched by row key: tp / fp / fn rows and "exact" rows with every field right."""
    counts: Counter = Counter()
    wrong_keys = {(row.field.rsplit(".", 1)[0], row.key) for row in [*match.missed, *match.extra]}
    for name, prefix in TABLE_ROWS.items():
        true_rows = rows_per_key(truth, prefix)
        found_rows = rows_per_key(predicted, prefix)
        for key in set(true_rows) | set(found_rows):
            matched = min(true_rows.get(key, 0), found_rows.get(key, 0))
            counts[f"rows.{name}.tp"] += matched
            counts[f"rows.{name}.fn"] += true_rows.get(key, 0) - matched
            counts[f"rows.{name}.fp"] += found_rows.get(key, 0) - matched
            if matched and (prefix, key) not in wrong_keys:
                counts[f"rows.{name}.exact"] += matched
    return counts


def rows_per_key(rows: list[Row], prefix: str) -> dict[str | None, int]:
    """How many table rows each row key has: the largest number of values of one field under that key."""
    per_field: Counter = Counter((row.key, row.field) for row in rows if row.field.rsplit(".", 1)[0] == prefix)
    result: dict[str | None, int] = {}
    for (key, _field), count in per_field.items():
        result[key] = max(result.get(key, 0), count)
    return result


def printed_forms(field: str, value: Any) -> list[str]:
    """Ways a true value can be printed on a page (casefolded): amounts, percents, three date styles, labels."""
    kind = field_type(field)
    leaf = field.rsplit(".", 1)[-1]
    if kind == "boolean":
        forms = ["yes" if value else "no"]
        return forms + ([BOOLEAN_PHRASES[leaf]] if leaf in BOOLEAN_PHRASES else [])
    if kind == "integer":
        return [str(value), f"{int(value):,}"]
    if kind == "money":
        return [f"{abs(float(value)):,.2f}"]
    if kind == "percent":
        return [f"{float(value):.1f}%", f"{float(value):.2f}%"]
    text = str(value)
    if kind == "date" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        day = date.fromisoformat(text)
        styles = [
            text,
            f"{day.month:02d}/{day.day:02d}/{day.year:04d}",
            f"{day.day:02d}-{MONTHS[day.month - 1]}-{day.year}",
        ]
        return [style.casefold() for style in styles]
    return list(dict.fromkeys([text.casefold(), text.replace("_", " ").casefold()]))


def value_on_page(field: str, value: Any, lines: list[str]) -> bool:
    """True when a printed form of the value is on one of the lines (or across lines) as a whole token.

    "Whole token" means the characters around the match are not letters or digits, so "US" is not found inside
    "BUSINESS" and "no" is not found inside "Account no".
    """
    texts = [line.casefold() for line in lines]
    texts.append(" ".join(texts))
    for form in printed_forms(field, value):
        pattern = re.compile(r"(?<![0-9a-z])" + re.escape(form) + r"(?![0-9a-z])")
        if any(pattern.search(text) for text in texts):
            return True
    return False


def citation_counts(match: FieldMatch, page_lines: dict[int, list[str]]) -> Counter:
    """For every correct value: is it printed on the cited page (truth text lines), and is that the truth page?"""
    counts: Counter = Counter()
    for truth_row, found in match.pairs:
        for kind in ("all", field_type(truth_row.field)):
            counts[f"citations.{kind}.total"] += 1
            lines = page_lines.get(found.page) if found.page is not None else None
            if lines is not None and value_on_page(truth_row.field, truth_row.value, lines):
                counts[f"citations.{kind}.correct"] += 1
            if found.page == truth_row.page:
                counts[f"citations.{kind}.same_page"] += 1
    return counts


def page_counts(truth: Any, result: PacketResult) -> Counter:
    """Page type accuracy (all / clean / scanned pages) and exactly recovered documents (pages and type)."""
    counts: Counter = Counter()
    predicted = {item.page_number: item.doc_type for item in result.page_types}
    for page in truth.pages:
        right = predicted.get(page.page_number) == page.doc_type
        for group in ("all", "scanned" if page.scanned else "clean"):
            counts[f"pages.{group}.total"] += 1
            counts[f"pages.{group}.correct"] += int(right)
    found = {(document.doc_type, tuple(document.pages)) for document in result.extraction.documents}
    for document in truth.documents:
        if document.pages:
            counts["documents.total"] += 1
            counts["documents.correct"] += int((document.doc_type, tuple(document.pages)) in found)
    return counts


def checklist_counts(truth: Any, result: PacketResult, documents_provided: list[str] | None) -> Counter:
    """Per checklist code: predicted missing vs truth missing, and predicted provided vs as-documented provided."""
    counts: Counter = Counter()
    codes = CREDIT_DOCUMENTS if truth.line == "credit" else INSURANCE_DOCUMENTS
    check = next((item for item in result.checks if item.check_id == "required_documents"), None)
    predicted_missing = set(check.values.get("missing", [])) if check is not None else set()
    true_missing = set(truth.missing_required_documents)
    for code in codes:
        counts["checklist.total"] += 1
        counts["checklist.correct"] += int((code in predicted_missing) == (code in true_missing))
    counts["checklist.packets"] += 1
    counts["checklist.packets_exact"] += int(predicted_missing == true_missing)
    if documents_provided is not None:
        true_provided = set(truth.application_as_documented.get("documents_provided", []))
        for code in codes:
            counts["provided.total"] += 1
            counts["provided.correct"] += int((code in documents_provided) == (code in true_provided))
    return counts


def usage_summary(result: PacketResult) -> dict[str, Any]:
    """LLM calls, tokens and cost of the packet (the rules extractor makes no call: zero calls, cost 0.0)."""
    usage = result.llm_usage
    return {
        "llm_calls": int(usage.get("calls", 0)),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "cost_usd": usage.get("cost_usd", 0.0) if usage else 0.0,
    }


def evaluate_packet(truth: Any, result: PacketResult, documents_provided: list[str] | None = None) -> dict[str, Any]:
    """Compare one pipeline result with its truth; returns the packet's JSON-simple evaluation record.

    documents_provided is to_application's documents_provided for this result (None skips that comparison).
    The record holds the counters (see the module docstring), the stage timings, the OCR page count, LLM usage
    and one outcome per injected problem, detected contradiction and planted instruction.
    """
    expected = truth_rows(truth)
    found = predicted_rows(result.extraction)
    match = match_fields(expected, found)
    page_lines = {page.page_number: list(page.text_lines) for page in truth.pages}
    page_docs = {page.page_number: page.doc_id for page in truth.pages}
    counts: Counter = Counter()
    counts.update(field_counts(match, page_conditions(truth)))
    counts.update(row_counts(expected, found, match))
    counts.update(citation_counts(match, page_lines))
    counts.update(page_counts(truth, result))
    counts.update(checklist_counts(truth, result, documents_provided))
    problems, contradictions = match_contradictions(
        list(truth.problems), list(result.contradictions.contradictions), page_docs
    )
    counts.update(contradiction_counts(problems, contradictions))
    injection_counter, injections = injection_counts(truth, result)
    counts.update(injection_counter)
    return {
        "packet_id": truth.packet_id,
        "line": truth.line,
        "scenario": truth.scenario,
        "n_pages": len(truth.pages),
        "n_scanned_pages": sum(1 for page in truth.pages if page.scanned),
        "extractor": result.extractor,
        "ocr_mode": result.ocr_mode,
        "counts": dict(sorted(counts.items())),
        "timings_ms": dict(result.timings_ms),
        "ocr_pages": len(result.ocr_pages),
        "usage": usage_summary(result),
        "problems": problems,
        "contradictions": contradictions,
        "injections": injections,
    }
