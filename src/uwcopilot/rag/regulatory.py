"""Load public regulatory documents (downloaded into data/raw/regulatory/) as windowed chunks.

These files are not committed (ground rule 2); scripts/download_data.py --datasets regulatory_docs fetches them and
logs their checksums. Each document is listed in REGULATORY_SOURCES with its issuer, date and URL, so every chunk
can be cited with its primary source and date (ground rule 10).

Two loaders:
- PDF (pypdf): one text per page. Page-number lines are dropped, footnotes are separated from the body and placed
  after the sentence that references them, and each sentence keeps the page(s) it came from.
- HTML (BeautifulSoup): the NY DFS circular letter numbers its paragraphs with <ol start="N"> lists. Each sentence
  keeps its paragraph number, which is how the letter cross-references itself ("paragraphs 12 and 15").

The text is then cut into token windows with overlap (chunking.py). Chunk ids look like "SR26-2-p5-1" (first
window that starts on PDF page 5) or "NYDFS-CL7-para15-1" (first window that starts in paragraph 15).

Downloaded documents are untrusted data. Nothing in them is executed or followed; they are only parsed for text.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup, NavigableString, Tag
from bs4.element import Comment
from pypdf import PdfReader

from uwcopilot.common.paths import RAW_DATA_DIR
from uwcopilot.rag.chunking import (
    ChunkingParams,
    Note,
    TextUnit,
    TokenCounter,
    count_words,
    insert_notes,
    join_units,
    normalize_space,
    pack_windows,
    split_sentences,
)
from uwcopilot.rag.chunks import Chunk

REGULATORY_DIR: Path = RAW_DATA_DIR / "regulatory"


@dataclass(frozen=True)
class RegulatorySource:
    """One public regulatory document: where it is saved, how it is identified and cited."""

    file_name: str
    doc_id: str
    short_id: str
    short_title: str
    title: str
    issuer: str
    doc_date: str
    url: str
    file_format: Literal["pdf", "html"]


# The same file names and URLs as REGULATORY_DOCS in scripts/download_data.py (a test checks they agree).
REGULATORY_SOURCES: tuple[RegulatorySource, ...] = (
    RegulatorySource(
        file_name="frb_sr_26_2.pdf",
        doc_id="frb_sr_26_2",
        short_id="SR26-2",
        short_title="FRB SR 26-2",
        title="Federal Reserve SR 26-2: Revised Guidance on Model Risk Management (with attachment)",
        issuer="Board of Governors of the Federal Reserve System (attachment issued jointly with the OCC and FDIC)",
        doc_date="2026-04-17",
        url="https://www.federalreserve.gov/supervisionreg/srletters/SR2602.pdf",
        file_format="pdf",
    ),
    RegulatorySource(
        file_name="nydfs_circular_letter_2024_07.html",
        doc_id="nydfs_cl_2024_07",
        short_id="NYDFS-CL7",
        short_title="NY DFS Insurance Circular Letter No. 7 (2024)",
        title=(
            "NY DFS Insurance Circular Letter No. 7 (2024): Use of Artificial Intelligence Systems and External "
            "Consumer Data and Information Sources in Insurance Underwriting and Pricing"
        ),
        issuer="New York State Department of Financial Services",
        doc_date="2024-07-11",
        url="https://www.dfs.ny.gov/industry-guidance/circular-letters/cl2024-07",
        file_format="html",
    ),
)

# Downloaded files that are deliberately not indexed, with the reason.
SKIPPED_REGULATORY_FILES: dict[str, str] = {
    "frb_sr_26_2.htm": (
        "web version of the SR 26-2 cover letter only; the same text is pages 1-2 of frb_sr_26_2.pdf, so indexing "
        "both would put duplicate chunks in every result list"
    ),
}

# ---------------------------------------------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------------------------------------------

PAGE_NUMBER_LINE = re.compile(r"^\d{1,3}$")
FOOTNOTE_START = re.compile(r"^(?P<number>\d{1,2}) (?P<text>\S.*)$")
# Footnote markers printed right after punctuation that follows a word, for example "website.1" or "use.3 ".
# A digit after a digit and a dot ("0.5") is a decimal, not a marker, so the character before the punctuation
# must be a letter, a closing bracket or a closing quote.
FOOTNOTE_MARKER = re.compile(r"(?<=[A-Za-z)”\"])(?P<punct>[.,;:”\"])(?P<number>\d{1,2})(?=\s|$)")
ROMAN_HEADING = re.compile(r"^[IVX]+\.\s+[A-Z][A-Z ,'’&\-–]+$")
SUBHEADING_WORD = re.compile(r"^(?:[A-Z][a-z]+|and|of|the|for|to|in|on|or)$")
SENTENCE_END = (".", ":", ";", "?", "!", ")", "”", '"')
MARKER_TEMPLATE = "⟦{key}⟧"
MARKER_IN_TEXT = re.compile(r"⟦(?P<key>[^⟧]+)⟧")


def read_pdf_pages(path: Path) -> list[str]:
    """Extract the text of every page with pypdf (index 0 = PDF page 1)."""
    reader = PdfReader(str(path))
    return [page.extract_text() or "" for page in reader.pages]


def split_body_and_footnotes(page_text: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Split one page into body lines and footnotes [(number, text)].

    The first non-empty line is dropped when it is only a number (a printed page number). Footnotes start after
    a blank line with "<n> <text>"; each later footnote must be numbered one higher than the one before, so a
    continuation line that happens to start with a number ("12 CFR ...") is not mistaken for a new footnote.
    """
    lines = [line.strip() for line in page_text.splitlines()]
    first_content = next((i for i, line in enumerate(lines) if line), None)
    if first_content is not None and PAGE_NUMBER_LINE.match(lines[first_content]):
        lines[first_content] = ""

    footnote_start: int | None = None
    for index in range(1, len(lines)):
        has_body_before = any(lines[:index])
        if has_body_before and lines[index - 1] == "" and FOOTNOTE_START.match(lines[index]):
            footnote_start = index
            break
    if footnote_start is None:
        return [line for line in lines if line], []

    body = [line for line in lines[:footnote_start] if line]
    footnotes: list[tuple[str, str]] = []
    for line in lines[footnote_start:]:
        if not line:
            continue
        match = FOOTNOTE_START.match(line)
        expected_next = str(int(footnotes[-1][0]) + 1) if footnotes else None
        if match and (not footnotes or match["number"] == expected_next):
            footnotes.append((match["number"], match["text"]))
        elif footnotes:
            footnotes[-1] = (footnotes[-1][0], join_pdf_lines([footnotes[-1][1], line]))
    return body, footnotes


def join_pdf_lines(lines: Sequence[str]) -> str:
    """Join wrapped lines with spaces; a line ending in a hyphen joins the next one directly ("rule-" + "based")."""
    text = ""
    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and len(text) > 1 and text[-2].isalpha():
            text += line
        else:
            text += " " + line
    return text


def is_pdf_heading(line: str, previous_line: str | None, previous_was_heading: bool) -> bool:
    """True for "IV. MODEL DEVELOPMENT AND MODEL USE" style headings and short Title Case subheadings.

    A subheading ("Model Inventory", "Components of Model Validation") must have at most 5 words, contain no
    punctuation, and follow a line that ended a sentence or another heading; that keeps address blocks and
    signature lines on the cover letter from being taken as headings.
    """
    if ROMAN_HEADING.match(line):
        return True
    words = line.split()
    if not 1 <= len(words) <= 5 or not all(SUBHEADING_WORD.match(word) for word in words):
        return False
    if not words[0][0].isupper():
        return False
    if previous_was_heading:
        return True
    if previous_line is None:
        return False
    previous_without_marker = FOOTNOTE_MARKER.sub(lambda match: match["punct"], previous_line)
    return previous_without_marker.endswith(SENTENCE_END)


def pdf_pages_to_units(page_texts: Sequence[str]) -> list[TextUnit]:
    """Turn PDF page texts into sentence units with headings, pages and footnotes placed after their reference.

    Body lines of consecutive pages are joined, so a sentence that runs across a page break stays whole and
    records both pages. Footnote markers are replaced by placeholders, which are removed again once each footnote
    has been attached to its sentence. Text before the first heading gets the heading "Front matter".
    """
    notes: dict[str, Note] = {}
    sections: list[tuple[tuple[str, ...], list[tuple[str, int]]]] = [(("Front matter",), [])]
    level_one: str | None = None
    previous_line: str | None = None
    previous_was_heading = False

    for page_number, page_text in enumerate(page_texts, start=1):
        body_lines, footnotes = split_body_and_footnotes(page_text)
        for number, text in footnotes:
            key = f"p{page_number}:{number}"
            notes[key] = Note(label=number, text=normalize_space(text), pages=(page_number,))
        for line in body_lines:
            if is_pdf_heading(line, previous_line, previous_was_heading):
                if ROMAN_HEADING.match(line):
                    level_one = line
                    heading_path: tuple[str, ...] = (line,)
                else:
                    heading_path = (level_one, line) if level_one else (line,)
                sections.append((heading_path, []))
                previous_was_heading = True
            else:
                sections[-1][1].append((mark_footnotes(line, page_number), page_number))
                previous_was_heading = False
            previous_line = line

    units: list[TextUnit] = []
    for block, (heading_path, lines) in enumerate(sections):
        units.extend(lines_to_sentence_units(lines, heading_path, block))
    return [strip_markers(unit) for unit in insert_notes(resolve_note_keys(units, notes), notes)]


def mark_footnotes(line: str, page_number: int) -> str:
    """Replace "use.3" with "use⟦p5:3⟧." so the sentence still ends with its punctuation and the marker is kept."""

    def placeholder(match: re.Match[str]) -> str:
        key = f"p{page_number}:{match['number']}"
        return MARKER_TEMPLATE.format(key=key) + match["punct"]

    return FOOTNOTE_MARKER.sub(placeholder, line)


def lines_to_sentence_units(
    lines: Sequence[tuple[str, int]], heading_path: tuple[str, ...], block: int
) -> list[TextUnit]:
    """Join (line, page) pairs, split into sentences, and record the pages each sentence spans."""
    if not lines:
        return []
    text = ""
    spans: list[tuple[int, int, int]] = []  # (start, end, page) of every line in the joined text
    for raw_line, page in lines:
        line = normalize_space(raw_line)
        if text and not (text.endswith("-") and len(text) > 1 and text[-2].isalpha()):
            text += " "
        start = len(text)
        text += line
        spans.append((start, len(text), page))

    units: list[TextUnit] = []
    position = 0
    for sentence in split_sentences(text):
        # Sentences are exact substrings of text (splitting only drops the space between them).
        found = text.find(sentence, position)
        start = found if found >= 0 else position
        end = start + len(sentence)
        position = end
        pages = tuple(sorted({page for span_start, span_end, page in spans if span_start < end and span_end > start}))
        note_refs = tuple(match["key"] for match in MARKER_IN_TEXT.finditer(sentence))
        units.append(TextUnit(text=sentence, heading_path=heading_path, pages=pages, block=block, note_refs=note_refs))
    return units


def resolve_note_keys(units: Sequence[TextUnit], notes: dict[str, Note]) -> list[TextUnit]:
    """Point a marker at the footnote printed on its page, or on the next page when the note was carried over."""
    resolved: list[TextUnit] = []
    for unit in units:
        keys: list[str] = []
        for key in unit.note_refs:
            page_text, _, number = key.partition(":")
            page = int(page_text[1:])
            candidates = [f"p{page}:{number}", f"p{page + 1}:{number}"]
            keys.append(next((candidate for candidate in candidates if candidate in notes), key))
        resolved.append(replace(unit, note_refs=tuple(keys)))
    return resolved


def strip_markers(unit: TextUnit) -> TextUnit:
    """Remove footnote placeholders from a unit's text."""
    return replace(unit, text=normalize_space(MARKER_IN_TEXT.sub("", unit.text)))


# ---------------------------------------------------------------------------------------------------------------
# HTML with numbered paragraphs
# ---------------------------------------------------------------------------------------------------------------

HTML_HEADING = re.compile(r"^(?:[IVX]+|[A-H])\.\s+\S")
BLOCK_TAGS = ("p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "td")
# Endnote containers are <div id="edn3">; the markers in the text are <a href="#_edn3">.
ENDNOTE_ID = re.compile(r"^edn(?P<number>\d+)$")
ENDNOTE_HREF = re.compile(r"^#_edn(?P<number>\d+)$")


def numbered_paragraph(element: Tag) -> int | None:
    """The paragraph number of the nearest <li> whose parent is an <ol type="1"> (None outside numbered lists).

    The number is the list's start attribute plus the item's position. The page's markup nests some lists
    inside the previous list item; looking for the nearest numbered item (not the outermost) still gives the
    right number.
    """
    for ancestor in [element, *element.parents]:
        if not isinstance(ancestor, Tag) or ancestor.name != "li":
            continue
        parent = ancestor.parent
        if isinstance(parent, Tag) and parent.name == "ol" and parent.get("type", "1") == "1":
            start = int(parent.get("start", 1))
            siblings = parent.find_all("li", recursive=False)
            # Compare by identity: bs4 tags compare equal when their markup is equal.
            position = next(index for index, sibling in enumerate(siblings) if sibling is ancestor)
            return start + position
    return None


def is_html_heading(block: Tag) -> bool:
    """A <p> whose only content is bold text such as "II. Fairness Principles" or "B. Unfair ... Discrimination"."""
    if block.name != "p":
        return False
    text = normalize_space(block.get_text(" "))
    strong_text = normalize_space(" ".join(strong.get_text(" ") for strong in block.find_all(["strong", "b"])))
    return bool(HTML_HEADING.match(text)) and strong_text == text and len(text.split()) <= 15


def find_letter_body(soup: BeautifulSoup) -> Tag:
    """The element that holds the letter: the parent of its first numbered list (falls back to <main>/<body>)."""
    first_list = soup.find("ol", attrs={"type": "1"})
    if isinstance(first_list, Tag) and isinstance(first_list.parent, Tag):
        return first_list.parent
    fallback = soup.find("main") or soup.body
    if not isinstance(fallback, Tag):
        raise ValueError("HTML document has no body to read.")
    return fallback


def html_to_units(html: str | bytes) -> list[TextUnit]:
    """Turn a numbered-paragraph letter into sentence units with heading paths and paragraph numbers.

    Text is read in document order, one block element (paragraph, list item) at a time, so text that the
    markup nests inside an earlier list item is still assigned to its own block and paragraph. Endnote markers
    (<a href="#_ednN">) are removed from the text and their endnotes are placed after the referencing sentence.
    Text before the first heading gets the heading "Preamble".
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    notes = extract_endnotes(soup)
    body = find_letter_body(soup)

    blocks: dict[int, HtmlBlock] = {}  # keyed by id() of the block tag
    order: list[HtmlBlock] = []
    current_heading: list[tuple[str, ...]] = [("Preamble",)]

    def block_for(tag: Tag) -> "HtmlBlock":
        """Return the HtmlBlock for a block tag, creating it (or updating the heading path) on first sight."""
        if id(tag) not in blocks:
            heading = is_html_heading(tag)
            if heading:
                title = normalize_space(tag.get_text(" "))
                previous = current_heading[0]
                is_level_one = bool(re.match(r"^[IVX]+\.", title)) or previous == ("Preamble",)
                current_heading[0] = (title,) if is_level_one else (previous[0], title)
            block = HtmlBlock(
                is_heading=heading,
                heading_path=current_heading[0],
                paragraph=numbered_paragraph(tag),
                order=len(order),
            )
            blocks[id(tag)] = block
            order.append(block)
        return blocks[id(tag)]

    for node in body.descendants:
        if isinstance(node, Tag) and node.name == "a" and ENDNOTE_HREF.match(str(node.get("href", ""))):
            owner = nearest_block(node)
            if owner is not None:
                block_for(owner).note_refs.append("edn" + ENDNOTE_HREF.match(str(node.get("href")))["number"])
            continue
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        if node.find_parent("a", href=ENDNOTE_HREF):
            continue  # the marker digit itself
        owner = nearest_block(node)
        if owner is not None:
            block_for(owner).parts.append(str(node))

    units: list[TextUnit] = []
    for block in order:
        if block.is_heading:
            continue
        sentences = split_sentences("".join(block.parts))
        for index, sentence in enumerate(sentences):
            is_last = index == len(sentences) - 1
            units.append(
                TextUnit(
                    text=sentence,
                    heading_path=block.heading_path,
                    paragraph=block.paragraph,
                    block=block.order,
                    note_refs=tuple(block.note_refs) if is_last else (),
                )
            )
    return insert_notes(units, notes)


@dataclass
class HtmlBlock:
    """Text collected for one block element while reading the HTML in order."""

    is_heading: bool
    heading_path: tuple[str, ...]
    paragraph: int | None
    order: int
    parts: list[str] = field(default_factory=list)
    note_refs: list[str] = field(default_factory=list)


def extract_endnotes(soup: BeautifulSoup) -> dict[str, Note]:
    """Remove every <div id="ednN"> endnote from the page and return them as Notes keyed "ednN"."""
    notes: dict[str, Note] = {}
    containers = [tag for tag in soup.find_all(True) if ENDNOTE_ID.match(str(tag.get("id", "")))]
    for container in containers:
        number = ENDNOTE_ID.match(str(container.get("id")))["number"]  # type: ignore[index]
        for back_link in container.find_all("a", href=re.compile(r"^#_ednref")):
            back_link.decompose()
        notes[f"edn{number}"] = Note(label=number, text=normalize_space(container.get_text(" ")))
        container.decompose()
    return notes


def nearest_block(node: Tag | NavigableString) -> Tag | None:
    """The closest enclosing block element (p, li, heading or table cell) of a node."""
    for ancestor in node.parents:
        if isinstance(ancestor, Tag) and ancestor.name in BLOCK_TAGS:
            return ancestor
    return None


# ---------------------------------------------------------------------------------------------------------------
# Units -> chunks
# ---------------------------------------------------------------------------------------------------------------


def unit_label(unit: TextUnit, file_format: str) -> str:
    """The locator used in a chunk id: "p5" (PDF page), "para15" (numbered paragraph) or "pre" (preamble)."""
    if file_format == "pdf":
        return f"p{unit.pages[0]}" if unit.pages else "p0"
    return f"para{unit.paragraph}" if unit.paragraph is not None else "pre"


def locator_text(pages: list[int], paragraphs: list[int]) -> str:
    """Human-readable location: "p. 4-5", "para. 15-16" or "preamble"."""
    if pages:
        first, last = pages[0], pages[-1]
        return f"p. {first}" if first == last else f"p. {first}-{last}"
    if paragraphs:
        first, last = paragraphs[0], paragraphs[-1]
        return f"para. {first}" if first == last else f"para. {first}-{last}"
    return "preamble"


def units_to_chunks(
    units: Sequence[TextUnit],
    source: RegulatorySource,
    params: ChunkingParams,
    count_tokens: TokenCounter,
) -> list[Chunk]:
    """Pack units into windows and turn each window into a Chunk with a stable, readable id."""
    chunks: list[Chunk] = []
    label_counts: dict[str, int] = {}
    for window in pack_windows(units, params, count_tokens):
        label = unit_label(window[0], source.file_format)
        label_counts[label] = label_counts.get(label, 0) + 1
        pages = sorted({page for unit in window for page in unit.pages})
        paragraphs = sorted({unit.paragraph for unit in window if unit.paragraph is not None})
        heading_path = list(window[0].heading_path)
        chunks.append(
            Chunk(
                chunk_id=f"{source.short_id}-{label}-{label_counts[label]}",
                doc_id=source.doc_id,
                doc_title=source.title,
                doc_kind="regulatory",
                title=f"{source.short_title}, {locator_text(pages, paragraphs)}: {' > '.join(heading_path)}",
                heading_path=heading_path,
                text=join_units(window),
                pages=pages,
                paragraphs=paragraphs,
                source_file=f"data/raw/regulatory/{source.file_name}",
                synthetic=False,
                issuer=source.issuer,
                doc_date=source.doc_date,
                source_url=source.url,
            )
        )
    return chunks


def load_regulatory_document(
    path: Path,
    source: RegulatorySource,
    params: ChunkingParams | None = None,
    count_tokens: TokenCounter = count_words,
) -> list[Chunk]:
    """Read one regulatory file (PDF or HTML) and return its window chunks (default ChunkingParams())."""
    params = params or ChunkingParams()
    if source.file_format == "pdf":
        units = pdf_pages_to_units(read_pdf_pages(path))
    else:
        units = html_to_units(path.read_bytes())
    if not units:
        raise ValueError(f"No text extracted from {path}.")
    return units_to_chunks(units, source, params, count_tokens)
