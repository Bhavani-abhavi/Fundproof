"""Instruction-like text inside packet documents (NBC-9.2, HM-11.2): a tripwire over page text and free-text fields.

Documents are data. This module never follows anything it reads; it only records where text that addresses a
reviewer, a system or an AI assistant appears, so the file can be referred to Fraud Operations / the SIU.

Detection reuses uwcopilot.agent.injection.detect (the agent's fixed pattern set: rag.prompt.INSTRUCTION_PATTERNS
plus agent.injection.AGENT_PATTERNS). No pattern was added or tuned on the synthetic packets. Before detection
every text is normalized with docintel.normalize.normalize_text (invisible characters removed, NFKC, homoglyph
fold), and any message hidden in Unicode tag characters is decoded and appended, so it is checked too.

Input interfaces (so this module does not depend on the page-text models in docintel/layout.py):
- scan_lines(lines): any iterable of (doc_id, page, line_number, text) tuples or ScanLine named tuples.
  Lines are grouped by (doc_id, page) in the order given. Each line is checked alone, and each pair of adjacent
  lines is checked joined with a space, to catch an instruction split across two lines. A pattern found only in a
  joined pair is reported on the pair's first line.
- scan_pages(pages, doc_ids): objects with `page_number: int` and `lines`, where each line has `text: str` and
  `line_number: int` (1-based), for example the PageText model of docintel/layout.py. doc_ids maps a page
  number to its document id (a page not in the map uses its own `doc_id` attribute if it has one, else None).
- scan_extraction_fields(extraction): the free-text values of an extraction (remarks, transaction descriptions,
  the cover email subject and body, business and owner names), each with its own citation.
Every hit is a DocumentInjectionHit with the document, page, line, pattern names and a short excerpt of the
normalized text (the excerpt is untrusted data kept for the audit record).
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict

from uwcopilot.agent.injection import MAX_EXCERPT_CHARS, detect
from uwcopilot.docintel.normalize import hidden_tag_text, normalize_text
from uwcopilot.docintel.schemas import Cited, InsuranceSubmission, LoanPacket, PacketExtraction


class DocumentInjectionHit(BaseModel):
    """Instruction-like text found on one line of a document (or in one extracted free-text field)."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None
    page: int | None
    line: int | None
    patterns: list[str]
    excerpt: str
    field: str | None = None


class ScanLine(NamedTuple):
    """One line of page text: the document, page number, 1-based line number and the raw text."""

    doc_id: str | None
    page: int
    line: int
    text: str


class LineLike(Protocol):
    """A line of page text as docintel/layout.py provides it."""

    text: str
    line_number: int


class PageLike(Protocol):
    """A page of text as docintel/layout.py provides it."""

    page_number: int
    lines: Sequence[LineLike]


def detection_text(raw: str) -> str:
    """The text the detector sees: normalized visible text plus any decoded tag-character message."""
    visible = normalize_text(raw)
    hidden = normalize_text(hidden_tag_text(raw))
    if hidden:
        return f"{visible} {hidden}".strip()
    return visible


def excerpt_of(text: str) -> str:
    """A short excerpt for the audit record."""
    return text[:MAX_EXCERPT_CHARS]


def group_by_page(lines: Iterable[ScanLine | tuple[str | None, int, int, str]]) -> list[list[ScanLine]]:
    """Lines grouped by (doc_id, page), keeping the order in which pages and lines were given."""
    groups: dict[tuple[str | None, int], list[ScanLine]] = {}
    for item in lines:
        line = ScanLine(*item)
        groups.setdefault((line.doc_id, line.page), []).append(line)
    return list(groups.values())


def add_hit(hits: dict[tuple, DocumentInjectionHit], line: ScanLine, patterns: list[str], text: str) -> None:
    """Record patterns for a line, merging with an earlier hit on the same line."""
    marker = (line.doc_id, line.page, line.line)
    if marker in hits:
        known = hits[marker]
        merged = known.patterns + [name for name in patterns if name not in known.patterns]
        hits[marker] = known.model_copy(update={"patterns": merged})
        return
    hits[marker] = DocumentInjectionHit(
        doc_id=line.doc_id, page=line.page, line=line.line, patterns=list(patterns), excerpt=excerpt_of(text)
    )


def scan_page_lines(page_lines: list[ScanLine], hits: dict[tuple, DocumentInjectionHit]) -> None:
    """Check each line alone, then each adjacent pair joined (see the module docstring)."""
    texts = [detection_text(line.text) for line in page_lines]
    found = [detect(text) for text in texts]
    for line, text, patterns in zip(page_lines, texts, found, strict=True):
        if patterns:
            add_hit(hits, line, patterns, text)
    for index in range(len(page_lines) - 1):
        joined = detection_text(page_lines[index].text + " " + page_lines[index + 1].text)
        already = set(found[index]) | set(found[index + 1])
        new_patterns = [name for name in detect(joined) if name not in already]
        if new_patterns:
            add_hit(hits, page_lines[index], new_patterns, joined)


def scan_lines(lines: Iterable[ScanLine | tuple[str | None, int, int, str]]) -> list[DocumentInjectionHit]:
    """Scan (doc_id, page, line_number, text) lines; one hit per line with instruction-like text."""
    hits: dict[tuple, DocumentInjectionHit] = {}
    for page_lines in group_by_page(lines):
        scan_page_lines(page_lines, hits)
    return list(hits.values())


def scan_pages(pages: Iterable[PageLike], doc_ids: Mapping[int, str] | None = None) -> list[DocumentInjectionHit]:
    """Scan page-text objects (see PageLike); doc_ids maps page numbers to document ids.

    A page that is not in doc_ids uses its own doc_id attribute when it has one, otherwise None.
    """
    mapping = doc_ids or {}
    lines = []
    for page in pages:
        doc_id = mapping.get(page.page_number, getattr(page, "doc_id", None))
        for line in page.lines:
            lines.append(ScanLine(doc_id, page.page_number, line.line_number, line.text))
    return scan_lines(lines)


def free_text_fields(extraction: PacketExtraction) -> list[tuple[str, Cited, str | None]]:
    """(field path, cited value, containing doc id) for every free-text value an applicant or broker controls."""
    fields: list[tuple[str, Cited, str | None]] = []
    if isinstance(extraction, LoanPacket):
        application = extraction.application
        if application is not None:
            fields.append(("application.remarks", application.remarks, application.doc_id))
            fields.append(("application.legal_name", application.legal_name, application.doc_id))
            for owner in application.owners:
                fields.append(("application.owners.name", owner.name, application.doc_id))
        for statement in extraction.statements:
            for transaction in statement.transactions:
                fields.append(("statement.transactions.description", transaction.description, statement.doc_id))
        return fields
    if isinstance(extraction, InsuranceSubmission):
        application = extraction.application
        if application is not None:
            fields.append(("application.remarks", application.remarks, application.doc_id))
            fields.append(("application.named_insured", application.named_insured, application.doc_id))
            for owner in application.owners:
                fields.append(("application.owners.name", owner.name, application.doc_id))
        email = extraction.cover_email
        if email is not None:
            fields.append(("cover_email.subject", email.subject, email.doc_id))
            fields.append(("cover_email.body", email.body, email.doc_id))
    return fields


def scan_extraction_fields(extraction: PacketExtraction) -> list[DocumentInjectionHit]:
    """Scan the extraction's free-text values; each hit carries the value's own citation and field path."""
    hits = []
    for field, cited, doc_id in free_text_fields(extraction):
        if not cited.found:
            continue
        text = detection_text(str(cited.value))
        patterns = detect(text)
        if patterns:
            hits.append(
                DocumentInjectionHit(
                    doc_id=cited.doc_id or doc_id,
                    page=cited.page,
                    line=cited.line,
                    patterns=patterns,
                    excerpt=excerpt_of(text),
                    field=field,
                )
            )
    return hits


def merge_hits(
    page_hits: list[DocumentInjectionHit], field_hits: list[DocumentInjectionHit]
) -> list[DocumentInjectionHit]:
    """Page-text hits plus the field hits on a (doc_id, page, line) the page scan did not already report."""
    seen = {(hit.doc_id, hit.page, hit.line) for hit in page_hits}
    merged = list(page_hits)
    for hit in field_hits:
        if (hit.doc_id, hit.page, hit.line) not in seen:
            merged.append(hit)
            seen.add((hit.doc_id, hit.page, hit.line))
    return merged
