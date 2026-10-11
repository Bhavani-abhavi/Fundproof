"""Tests for the synthetic guideline corpus: banner, stable section ids, and section count."""

import re
from pathlib import Path

import pytest

from uwcopilot.common.paths import CORPUS_DIR
from uwcopilot.tools.schemas import GUIDELINE_DOC_IDS, SECTION_ID_PATTERN

GUIDELINE_PREFIXES = {"northwind_business_credit": "NBC", "harborline_mutual_commercial": "HM"}
HEADING = re.compile(r"^## (\S+) (.+)$", re.MULTILINE)


def guideline_path(doc_id: str) -> Path:
    """Path of a guideline document by its id."""
    return CORPUS_DIR / "guidelines" / f"{doc_id}.md"


def test_every_guideline_document_exists() -> None:
    assert set(GUIDELINE_PREFIXES) == set(GUIDELINE_DOC_IDS)
    for doc_id in GUIDELINE_DOC_IDS:
        assert guideline_path(doc_id).exists()


@pytest.mark.parametrize("doc_id", sorted(GUIDELINE_PREFIXES))
def test_document_starts_with_a_synthetic_banner(doc_id: str) -> None:
    text = guideline_path(doc_id).read_text(encoding="utf-8")
    first_line = text.splitlines()[0]
    assert "SYNTHETIC" in first_line
    title = next(line for line in text.splitlines() if line.startswith("# "))
    assert "SYNTHETIC" in title


@pytest.mark.parametrize("doc_id", sorted(GUIDELINE_PREFIXES))
def test_sections_have_unique_stable_ids(doc_id: str) -> None:
    text = guideline_path(doc_id).read_text(encoding="utf-8")
    headings = HEADING.findall(text)
    ids = [section_id for section_id, _title in headings]
    assert 20 <= len(ids) <= 30
    assert len(ids) == len(set(ids))
    prefix = GUIDELINE_PREFIXES[doc_id]
    for section_id in ids:
        assert re.fullmatch(rf"{prefix}-\d+\.\d+", section_id), section_id
        assert re.fullmatch(SECTION_ID_PATTERN, section_id)


@pytest.mark.parametrize("doc_id", sorted(GUIDELINE_PREFIXES))
def test_cross_references_point_at_real_sections(doc_id: str) -> None:
    text = guideline_path(doc_id).read_text(encoding="utf-8")
    ids = {section_id for section_id, _title in HEADING.findall(text)}
    prefix = GUIDELINE_PREFIXES[doc_id]
    referenced = set(re.findall(rf"\b{prefix}-\d+\.\d+\b", text))
    assert referenced <= ids, sorted(referenced - ids)


def test_corpus_readme_states_synthetic_provenance() -> None:
    readme = (CORPUS_DIR / "README.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" in readme
    assert "Provenance" in readme
