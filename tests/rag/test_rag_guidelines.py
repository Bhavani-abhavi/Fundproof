"""Tests for the guideline markdown loader: one chunk per section, ids kept, long sections split into parts."""

import re

import pytest

from uwcopilot.common.paths import CORPUS_DIR
from uwcopilot.rag.chunking import count_words
from uwcopilot.rag.guidelines import (
    GuidelineFormatError,
    load_guideline_chunks,
    parse_guideline_markdown,
    split_long_section,
)

FIXTURE = """> SYNTHETIC banner.

# Test Lender Guidelines (SYNTHETIC)

- Document id: `test_lender`

Conventions paragraph that is not a section.

## TL-1.1 Purpose

These rules cover small loans.

## TL-2.1 Referral triggers

| Trigger | Condition | Refer to |
|---|---|---|
| R-1 | Coverage below 1.25 | Senior Underwriter |
| R-2 | Sanctions potential match | Compliance |
| R-3 | Document tampering | Fraud Operations |
"""


def test_parse_keeps_section_ids_titles_and_heading_path() -> None:
    chunks = parse_guideline_markdown(FIXTURE, "corpus/guidelines/test.md")
    assert [chunk.chunk_id for chunk in chunks] == ["TL-1.1", "TL-2.1"]
    first = chunks[0]
    assert first.doc_id == "test_lender"
    assert first.title == "TL-1.1 Purpose"
    assert first.heading_path == ["Test Lender Guidelines (SYNTHETIC)", "TL-1.1 Purpose"]
    assert first.text == "These rules cover small loans."
    assert first.synthetic and first.doc_kind == "guideline"
    assert "Conventions paragraph" not in " ".join(chunk.text for chunk in chunks)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("## TL-1.1 A\n\nbody\n", "Document id"),
        ("# T\n- Document id: `t`\n", "no '## <ID> <title>' sections"),
        ("# T\n- Document id: `t`\n## TL-1.1 A\nx\n## TL-1.1 B\ny\n", "duplicate section id"),
        ("# T\n- Document id: `t`\n## TL-1.1 A\n\n## TL-1.2 B\ny\n", "is empty"),
    ],
)
def test_parse_rejects_malformed_files(text: str, message: str) -> None:
    with pytest.raises(GuidelineFormatError, match=message):
        parse_guideline_markdown(text, "bad.md")


def test_real_corpus_has_one_chunk_per_section_with_matching_ids() -> None:
    chunks = load_guideline_chunks()
    for path in sorted((CORPUS_DIR / "guidelines").glob("*.md")):
        heading_ids = re.findall(r"^## ([A-Z]+-\d+\.\d+) ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.source_file.endswith(path.name)]
        assert chunk_ids == heading_ids
        assert len(chunk_ids) == 30
    assert all(chunk.title.startswith(chunk.chunk_id + " ") for chunk in chunks)
    assert all("\n## " not in chunk.text for chunk in chunks)


def test_split_long_section_keeps_id_and_repeats_table_header() -> None:
    section = parse_guideline_markdown(FIXTURE, "t.md")[1]
    parts = split_long_section(section, max_tokens=30, count_tokens=count_words)
    assert len(parts) > 1
    assert {part.chunk_id for part in parts} == {"TL-2.1"}
    assert [part.part for part in parts] == list(range(1, len(parts) + 1))
    assert all(part.part_count == len(parts) for part in parts)
    for part in parts[1:]:
        assert part.text.startswith("| Trigger | Condition | Refer to |\n|---|---|---|")
    rows = [line for part in parts for line in part.text.splitlines() if line.startswith("| R-")]
    assert rows == [line for line in section.text.splitlines() if line.startswith("| R-")]
    assert len({part.point_key for part in parts}) == len(parts)


def test_split_long_section_returns_short_section_unchanged() -> None:
    section = parse_guideline_markdown(FIXTURE, "t.md")[0]
    assert split_long_section(section, max_tokens=500, count_tokens=count_words) == [section]
