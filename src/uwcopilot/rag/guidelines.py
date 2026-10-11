"""Load the SYNTHETIC underwriting guideline markdown files: one chunk per numbered section.

Each guideline file has an H1 title, a "Document id" line, and sections headed "## <ID> <title>", for example
"## NBC-3.2 Revenue verification". A section is the natural unit to retrieve and cite: it is short (a few hundred
tokens), it covers one rule, and its id is what a credit memo cites. So sections are not split into windows.

The text before the first section (the SYNTHETIC banner, version and the "at least / more than" conventions) is
not a chunk: it has no section id to cite, and the banner is repeated in every hit's metadata (synthetic=True).

One exception to "one section, one chunk": a section longer than the embedding model can read (512 tokens for
BAAI/bge-small-en-v1.5; longer input is silently truncated) is split at line boundaries into parts that keep the
same section id (split_long_section). A part that starts inside a markdown table repeats the table's header rows.
"""

import re
from pathlib import Path

from uwcopilot.common.paths import CORPUS_DIR
from uwcopilot.rag.chunking import TokenCounter
from uwcopilot.rag.chunks import Chunk

GUIDELINES_DIR: Path = CORPUS_DIR / "guidelines"
DOC_TITLE_LINE = re.compile(r"^# (?P<title>.+?)\s*$")
DOC_ID_LINE = re.compile(r"^- Document id: `(?P<doc_id>[a-z0-9_]+)`\s*$")
SECTION_HEADING = re.compile(r"^## (?P<section_id>[A-Z]+-\d+\.\d+) (?P<title>.+?)\s*$")


class GuidelineFormatError(ValueError):
    """Raised when a guideline file does not have the expected title, document id or section headings."""


def parse_guideline_markdown(text: str, source_file: str) -> list[Chunk]:
    """Turn one guideline markdown document into one Chunk per "## <ID> <title>" section, in file order."""
    doc_title: str | None = None
    doc_id: str | None = None
    sections: list[tuple[str, str, list[str]]] = []  # (section_id, title, body lines)
    for line in text.splitlines():
        heading = SECTION_HEADING.match(line)
        if heading:
            sections.append((heading["section_id"], heading["title"], []))
            continue
        if sections:
            sections[-1][2].append(line)
            continue
        title_match = DOC_TITLE_LINE.match(line)
        if title_match and doc_title is None:
            doc_title = title_match["title"]
        id_match = DOC_ID_LINE.match(line)
        if id_match:
            doc_id = id_match["doc_id"]

    if doc_title is None or doc_id is None:
        raise GuidelineFormatError(f"{source_file}: missing '# <title>' or '- Document id: `<id>`' line.")
    if not sections:
        raise GuidelineFormatError(f"{source_file}: no '## <ID> <title>' sections found.")

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for section_id, title, body_lines in sections:
        if section_id in seen:
            raise GuidelineFormatError(f"{source_file}: duplicate section id {section_id}.")
        seen.add(section_id)
        body = "\n".join(body_lines).strip()
        if not body:
            raise GuidelineFormatError(f"{source_file}: section {section_id} is empty.")
        chunks.append(
            Chunk(
                chunk_id=section_id,
                doc_id=doc_id,
                doc_title=doc_title,
                doc_kind="guideline",
                title=f"{section_id} {title}",
                heading_path=[doc_title, f"{section_id} {title}"],
                text=body,
                source_file=source_file,
                synthetic=True,
            )
        )
    return chunks


def load_guideline_chunks(guidelines_dir: Path = GUIDELINES_DIR) -> list[Chunk]:
    """Load every *.md file in guidelines_dir (sorted by name) and return all their section chunks."""
    paths = sorted(guidelines_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No guideline markdown files in {guidelines_dir}.")
    chunks: list[Chunk] = []
    for path in paths:
        source_file = f"corpus/guidelines/{path.name}"
        chunks.extend(parse_guideline_markdown(path.read_text(encoding="utf-8"), source_file))
    return chunks


def table_header_before(lines: list[str], index: int) -> list[str]:
    """The header and separator rows of the markdown table whose data row is line `index` ([] otherwise)."""
    if not lines[index].startswith("|"):
        return []
    start = index
    while start > 0 and lines[start - 1].startswith("|"):
        start -= 1
    if index < start + 2:
        return []  # the line is the header or separator row itself
    header = lines[start : start + 2]
    if set(header[1].replace("|", "").strip()) <= {"-", ":", " "}:
        return header
    return []


def split_long_section(chunk: Chunk, max_tokens: int, count_tokens: TokenCounter) -> list[Chunk]:
    """Split a section whose embedding text is longer than max_tokens into parts, at line boundaries.

    Every part keeps the section's chunk_id (the citation stays "[HM-9.3]") and records part / part_count.
    Returns [chunk] unchanged when it already fits.
    """
    if count_tokens(chunk.embedding_text()) <= max_tokens:
        return [chunk]
    lines = chunk.text.splitlines()
    parts: list[list[str]] = []
    current: list[str] = []
    for index, line in enumerate(lines):
        candidate = [*current, line]
        probe = chunk.model_copy(update={"text": "\n".join(candidate), "part": 2, "part_count": 2})
        if current and count_tokens(probe.embedding_text()) > max_tokens:
            parts.append(current)
            current = [*table_header_before(lines, index), line]
        else:
            current = candidate
    if current:
        parts.append(current)
    return [
        chunk.model_copy(update={"text": "\n".join(part).strip(), "part": number, "part_count": len(parts)})
        for number, part in enumerate(parts, start=1)
    ]
