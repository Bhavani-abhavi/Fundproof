"""Rules extraction of the SYNTHETIC insurance submission documents (loss runs are in rules_loss_runs.py).

- insurance application: label/value lines (APPLICATION_FIELDS); "Not requested" GL limits stay missing; the prior
  carriers table (rows between its header and "Section 6"): the policy number (letters, "-", digits) splits the
  carrier name on its left from the policy year start on its right, then the two premium cells; the owners table
  (rules_tables.parse_owner_row) and the Remarks lines between "Section 7 - Remarks" and "Section 8";
- property schedule: each location row starts with a location id ("L1"). It is read from the right: four amounts
  (building, contents, business income, TIV), protection class, sprinklered Yes/No, square feet, stories, roof
  year, year built; from the left: the location id, then the construction class (the longest construction label
  of agent.applications.ConstructionClass that the words start with), and the occupancy is what is left. The next
  "Address: ..." line is that location's address. Total TIV and "Date signed" are label/value lines;
- cover email: From / To / Subject / Date in the first lines; the body is every line after the Date line;
- ownership statement: Entity name and the owners table;
- inspection report: Location and Inspection date.
"""

import re
from typing import get_args

from uwcopilot.agent.applications import ConstructionClass
from uwcopilot.docintel.classify import Segment, normalize_for_match
from uwcopilot.docintel.layout import Word
from uwcopilot.docintel.parsing import (
    code_from_label,
    fix_ocr_digits,
    normalize_fein,
    parse_amount,
    parse_bool,
    parse_date,
    parse_identifier,
    parse_int,
    parse_percent,
    parse_text,
    strip_edges,
)
from uwcopilot.docintel.rules_common import (
    LineRef,
    amount_words,
    cite_lines,
    cite_words,
    document_lines,
    find_line,
    find_value,
    header_index,
    lines_between,
    lines_until,
    parse_value_words,
    title_box,
    value_after_label,
    words_text,
)
from uwcopilot.docintel.rules_tables import owner_rows, premium_cells
from uwcopilot.docintel.schemas import (
    CoverEmail,
    InspectionReport,
    InsuranceApplicationForm,
    OwnershipStatement,
    PriorCarrier,
    PropertySchedule,
    ScheduleLocation,
)

APPLICATION_FIELDS: tuple[tuple[str, tuple[str, ...], object], ...] = (
    ("named_insured", ("Named insured",), parse_text),
    ("fein", ("FEIN",), normalize_fein),
    ("years_in_business", ("Years in business",), parse_int),
    ("business_class", ("Business class",), code_from_label),
    ("coverage", ("Coverage",), code_from_label),
    ("gl_limit_per_occurrence", ("GL limit per occurrence",), parse_amount),
    ("gl_limit_aggregate", ("GL limit aggregate",), parse_amount),
    ("application_date", ("Application date",), parse_date),
    ("proposed_effective_date", ("Proposed effective date",), parse_date),
    ("stated_tiv", ("Total insured value (all locations)", "Total insured value"), parse_amount),
    ("annual_revenue", ("Annual revenue",), parse_amount),
    ("annual_payroll", ("Annual payroll",), parse_amount),
    ("subcontracted_work_pct", ("Subcontracted work",), parse_percent),
    ("has_subcontractor_coi_tracking", ("Subcontractor certificates tracked",), parse_bool),
    ("foreign_products", ("Foreign products sold",), parse_bool),
    ("stated_claim_count", ("Number of claims",), parse_int),
    ("stated_total_incurred", ("Total incurred losses",), parse_amount),
)
OWNER_TITLES = ("Name", "Type", "Title", "Ownership", "Control person", "Country")
CARRIER_TITLES = ("Carrier", "Policy number", "Policy year start", "Property premium", "GL premium")
POLICY_NUMBER = re.compile(r"^[A-Z]{2,5}-\d{5,9}$")
LOCATION_ID = re.compile(r"^L\d{1,2}$")
CONSTRUCTION_LABELS = sorted(
    (code.replace("_", " ") for code in get_args(ConstructionClass)), key=lambda label: len(label), reverse=True
)
SCHEDULE_TITLES = ("Loc", "Construction", "Occupancy", "Built", "Roof", "Stories", "Sprk", "Building", "TIV")
SCHEDULE_AMOUNTS = ("building_value", "contents_value", "business_income_value", "tiv")
SCHEDULE_NUMBERS = ("protection_class", "sprinklered", "square_feet", "stories", "roof_year", "year_built")
EMAIL_HEADER_LINES = 6


def policy_number_index(words: list[Word]) -> int | None:
    """Index of the policy-number word ("ABC-1234567") in a prior-carrier row, or None."""
    for index, word in enumerate(words):
        text = parse_identifier(word.text) or ""
        prefix, _, digits = text.partition("-")
        if POLICY_NUMBER.match(f"{prefix}-{fix_ocr_digits(digits)}"):
            return index
    return None


def parse_prior_carrier(ref: LineRef, columns: dict[str, float], doc_id: str) -> PriorCarrier | None:
    """One prior-carrier row (see the module docstring), or None."""
    words = ref.words
    policy_at = policy_number_index(words)
    if policy_at is None or policy_at == 0 or policy_at + 1 >= len(words):
        return None
    start = parse_date(words[policy_at + 1].text)
    if start is None:
        return None
    carrier = words[:policy_at]
    row = PriorCarrier(
        carrier=cite_words(ref, carrier, parse_text(words_text(carrier)), doc_id),
        policy_number=cite_words(ref, [words[policy_at]], parse_identifier(words[policy_at].text), doc_id),
        policy_year_start=cite_words(ref, [words[policy_at + 1]], start, doc_id),
    )
    cells = premium_cells(ref, words[policy_at + 2 :], ("property", "gl"), columns, doc_id)
    if "property" in cells:
        row.property_premium = cells["property"]
    if "gl" in cells:
        row.gl_premium = cells["gl"]
    return row


def prior_carriers(refs: list[LineRef], doc_id: str) -> list[PriorCarrier]:
    """The prior carriers table of the application."""
    start = header_index(refs, CARRIER_TITLES, 3)
    if start is None:
        return []
    columns = {}
    for name, title in (("property", "Property premium"), ("gl", "GL premium")):
        box = title_box(refs[start], title)
        if box is not None:
            columns[name] = box[1]
    rows = []
    for ref in lines_until(refs[start + 1 :], ("Section 6", "Owners")):
        row = parse_prior_carrier(ref, columns, doc_id)
        if row is not None:
            rows.append(row)
    return rows


def extract_application(segment: Segment) -> InsuranceApplicationForm:
    """The two-page commercial lines application."""
    refs = document_lines(segment)
    doc_id = segment.doc_id
    form = InsuranceApplicationForm(doc_id=doc_id)
    for name, labels, parser in APPLICATION_FIELDS:
        setattr(form, name, find_value(refs, labels, parser, doc_id))
    form.prior_carriers = prior_carriers(refs, doc_id)
    start = header_index(refs, OWNER_TITLES, 4)
    if start is not None:
        form.owners = owner_rows(lines_until(refs[start + 1 :], ("Section 7", "Remarks")), doc_id)
    remarks = lines_between(refs, "Section 7 - Remarks", ("Section 8", "Signature"))
    if remarks:
        form.remarks = cite_lines(remarks, " ".join(ref.text for ref in remarks), doc_id)
    return form


def location_id_word(word: Word) -> str | None:
    """ "L1" from a location-id word (an OCR "Ll" or "LI" is fixed), or None."""
    text = strip_edges(word.text)
    if not text[:1] == "L":
        return None
    fixed = "L" + fix_ocr_digits(text[1:])
    return fixed if LOCATION_ID.match(fixed) else None


def construction_split(words: list[Word]) -> tuple[list[Word], list[Word]]:
    """(construction words, occupancy words): the longest construction label the words start with."""
    text = normalize_for_match(words_text(words))
    for label in CONSTRUCTION_LABELS:
        size = len(label.split())
        if text.startswith(label) and len(words) > size:
            return words[:size], words[size:]
    return words[:1], words[1:]


def parse_location(ref: LineRef, doc_id: str) -> ScheduleLocation | None:
    """One location row of the schedule (see the module docstring), or None."""
    words = ref.words
    location_id = location_id_word(words[0]) if words else None
    amounts = amount_words(words)
    if location_id is None or len(amounts) < len(SCHEDULE_AMOUNTS):
        return None
    amounts = amounts[-len(SCHEDULE_AMOUNTS) :]
    location = ScheduleLocation(location_id=cite_words(ref, [words[0]], location_id, doc_id))
    for name, word in zip(SCHEDULE_AMOUNTS, amounts, strict=True):
        setattr(location, name, cite_words(ref, [word], parse_amount(word.text), doc_id))
    middle = [word for word in words[1:] if word.x1 < amounts[0].x0]
    numbers = list(reversed(middle[-len(SCHEDULE_NUMBERS) :]))
    if len(middle) > len(SCHEDULE_NUMBERS):
        for name, word in zip(SCHEDULE_NUMBERS, numbers, strict=True):
            parser = parse_bool if name == "sprinklered" else parse_int
            value = parser(word.text)
            if value is not None:
                setattr(location, name, cite_words(ref, [word], value, doc_id))
        text_words = middle[: -len(SCHEDULE_NUMBERS)]
        construction, occupancy = construction_split(text_words)
        if construction:
            code = code_from_label(words_text(construction))
            location.construction_class = cite_words(ref, construction, code, doc_id)
        if occupancy:
            location.occupancy = cite_words(ref, occupancy, parse_text(words_text(occupancy)), doc_id)
    return location


def extract_property_schedule(segment: Segment) -> PropertySchedule:
    """The property schedule / statement of values."""
    refs = document_lines(segment)
    doc_id = segment.doc_id
    schedule = PropertySchedule(
        doc_id=doc_id,
        insured_name=find_value(refs, ("Insured",), parse_text, doc_id),
        total_tiv=find_value(refs, ("Total TIV (all locations)", "Total TIV"), parse_amount, doc_id),
    )
    for index, ref in enumerate(refs):
        location = parse_location(ref, doc_id)
        if location is None:
            continue
        if index + 1 < len(refs):
            address_ref = refs[index + 1]
            address = value_after_label(address_ref.line, "Address", address_ref.fuzzy)
            if address:
                location.address = cite_words(address_ref, address, parse_text(words_text(address)), doc_id)
        schedule.locations.append(location)
    signed = find_line(refs, "Date signed")
    if signed is not None:
        ref = refs[signed]
        for word in reversed(ref.words):
            if parse_date(word.text) is not None:
                schedule.signed_date = cite_words(ref, [word], parse_date(word.text), doc_id)
                break
    return schedule


def extract_cover_email(segment: Segment) -> CoverEmail:
    """The broker's cover email; the body is untrusted free text (every line after the Date line)."""
    refs = document_lines(segment)
    doc_id = segment.doc_id
    header = refs[:EMAIL_HEADER_LINES]
    email = CoverEmail(
        doc_id=doc_id,
        sender=find_value(header, ("From",), parse_text, doc_id),
        recipient=find_value(header, ("To",), parse_text, doc_id),
        subject=find_value(header, ("Subject",), parse_text, doc_id),
    )
    for index, ref in enumerate(header):
        value, used = parse_value_words(value_after_label(ref.line, "Date", ref.fuzzy) or [], parse_date)
        if value is None:
            continue
        email.sent_date = cite_words(ref, used, value, doc_id)
        body = refs[index + 1 :]
        if body:
            email.body = cite_lines(body, " ".join(line.text for line in body), doc_id)
        break
    return email


def extract_ownership_statement(segment: Segment) -> OwnershipStatement:
    """The beneficial ownership statement."""
    refs = document_lines(segment)
    statement = OwnershipStatement(
        doc_id=segment.doc_id, entity_name=find_value(refs, ("Entity name",), parse_text, segment.doc_id)
    )
    start = header_index(refs, OWNER_TITLES, 4)
    if start is not None:
        statement.owners = owner_rows(lines_until(refs[start + 1 :], ("I certify",)), segment.doc_id)
    return statement


def extract_inspection_report(segment: Segment) -> InspectionReport:
    """One loss control inspection report."""
    refs = document_lines(segment)
    doc_id = segment.doc_id
    report = InspectionReport(doc_id=doc_id, inspection_date=find_value(refs, ("Inspection date",), parse_date, doc_id))
    for ref in refs:
        words = value_after_label(ref.line, "Location", ref.fuzzy)
        if words and location_id_word(words[0]) is not None:
            report.location_id = cite_words(ref, [words[0]], location_id_word(words[0]), doc_id)
            break
    return report
