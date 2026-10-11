"""Tests for the regulatory loaders: PDF pages and numbered-paragraph HTML to sentence units and window chunks."""

from pathlib import Path

from rag_helpers import make_text_pdf

import download_data
from uwcopilot.rag.chunking import ChunkingParams, count_words
from uwcopilot.rag.corpus import load_corpus
from uwcopilot.rag.regulatory import (
    REGULATORY_SOURCES,
    SKIPPED_REGULATORY_FILES,
    RegulatorySource,
    html_to_units,
    pdf_pages_to_units,
    read_pdf_pages,
    split_body_and_footnotes,
    units_to_chunks,
)

PAGE_ONE = (
    " \n2  \nrisk management practices that apply. \n"
    "This guidance does not set forth enforceable standards; accordingly, non-compliance will not result in "
    "criticism.1  \nII. PURPOSE AND SCOPE \n"
    "This guidance is expected to be most relevant to banking organizations with over $30 \n \n"
    "1 See 12 CFR Part 4, Subpart F, Appendix A (OCC); 12 CFR Part 262, \nAppendix A (Board). \n"
)
PAGE_TWO = (
    " \n3  \nbillion in total assets.2 The term “model” excludes simple arithmetic calculations and rule-\n"
    "based processes.3 \nModel Inventory \nAn inventory is common practice. \n \n"
    "2 For a description of supervised organizations. \n3 Generative AI and agentic AI models are novel. \n"
)

TEST_PDF_SOURCE = RegulatorySource(
    file_name="test.pdf",
    doc_id="test_letter",
    short_id="TL26-1",
    short_title="Test Letter 26-1",
    title="Test Letter 26-1 (fixture)",
    issuer="Test Agency",
    doc_date="2026-01-01",
    url="https://example.org/test.pdf",
    file_format="pdf",
)

NUMBERED_HTML = """<html><body><main><nav><ol><li>Home</li></ol></nav><div class="page">
<p><strong>Circular Letter No. 9</strong></p>
<ul><li>Comment theme. Commenters asked things<a href="#_edn1" id="_ednref1"><sup>1</sup></a>.</li></ul>
<p><strong>I. Purpose</strong></p>
<ol type="1"><li>First paragraph text.</li><li>Second paragraph.</li></ol>
<p><strong>II. Fairness</strong></p>
<ol type="1" start="3"><li><p>Third paragraph intro:</p><ol type="i"><li>Step one.</li><li>Step two.</li></ol>
<p><strong>A. Testing</strong></p><ol type="1" start="4"><li>Fourth paragraph, nested by broken markup.</li></ol>
</li></ol>
</div>
<div id="edn1"><p><a href="#_ednref1" id="_edn1"><sup>1</sup></a> Endnote one text.</p></div>
</main></body></html>"""


def test_split_body_and_footnotes_drops_page_number_and_joins_continuations() -> None:
    body, footnotes = split_body_and_footnotes(PAGE_ONE)
    assert body[0] == "risk management practices that apply."
    assert "2" not in body
    assert footnotes == [("1", "See 12 CFR Part 4, Subpart F, Appendix A (OCC); 12 CFR Part 262, Appendix A (Board).")]


def test_pdf_units_keep_pages_headings_and_place_footnotes_after_their_sentence() -> None:
    units = pdf_pages_to_units([PAGE_ONE, PAGE_TWO])
    texts = [unit.text for unit in units]
    assert texts == [
        "risk management practices that apply.",
        "This guidance does not set forth enforceable standards; accordingly, non-compliance will not result in "
        "criticism.",
        "[Note 1] See 12 CFR Part 4, Subpart F, Appendix A (OCC); 12 CFR Part 262, Appendix A (Board).",
        "This guidance is expected to be most relevant to banking organizations with over $30 billion in total assets.",
        "[Note 2] For a description of supervised organizations.",
        "The term “model” excludes simple arithmetic calculations and rule-based processes.",
        "[Note 3] Generative AI and agentic AI models are novel.",
        "An inventory is common practice.",
    ]
    by_text = {unit.text: unit for unit in units}
    assert by_text[texts[0]].heading_path == ("Front matter",)
    assert by_text[texts[3]].heading_path == ("II. PURPOSE AND SCOPE",)
    assert by_text[texts[3]].pages == (1, 2)  # the sentence runs across the page break
    assert by_text[texts[6]].pages == (2,)
    assert by_text[texts[7]].heading_path == ("II. PURPOSE AND SCOPE", "Model Inventory")
    assert not any("⟦" in text for text in texts)


def test_read_pdf_pages_reads_a_generated_pdf(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pdf"
    path.write_bytes(make_text_pdf([["SR 99-1", "First page line."], ["Second page line."]]))
    pages = read_pdf_pages(path)
    assert len(pages) == 2
    assert "First page line." in pages[0] and "SR 99-1" in pages[0]
    assert "Second page line." in pages[1]


def test_pdf_chunks_have_page_ids_titles_and_metadata() -> None:
    units = pdf_pages_to_units([PAGE_ONE, PAGE_TWO])
    chunks = units_to_chunks(units, TEST_PDF_SOURCE, ChunkingParams(max_tokens=40, overlap_tokens=10), count_words)
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids[0] == "TL26-1-p1-1"
    assert len(ids) == len(set(ids))
    purpose = next(chunk for chunk in chunks if chunk.heading_path == ["II. PURPOSE AND SCOPE"])
    assert purpose.pages == [1, 2]
    assert purpose.title.startswith("Test Letter 26-1, p. 1-2: II. PURPOSE AND SCOPE")
    assert purpose.issuer == "Test Agency" and purpose.doc_date == "2026-01-01"
    assert not purpose.synthetic and purpose.doc_kind == "regulatory"
    assert "[Note 2]" in purpose.text


def test_html_units_use_list_numbering_even_when_markup_is_nested() -> None:
    units = html_to_units(NUMBERED_HTML)
    rows = [(unit.text, unit.heading_path, unit.paragraph) for unit in units]
    assert rows == [
        ("Circular Letter No. 9", ("Preamble",), None),
        ("Comment theme.", ("Preamble",), None),
        ("Commenters asked things.", ("Preamble",), None),
        ("[Note 1] Endnote one text.", ("Preamble",), None),
        ("First paragraph text.", ("I. Purpose",), 1),
        ("Second paragraph.", ("I. Purpose",), 2),
        ("Third paragraph intro:", ("II. Fairness",), 3),
        ("Step one.", ("II. Fairness",), 3),
        ("Step two.", ("II. Fairness",), 3),
        ("Fourth paragraph, nested by broken markup.", ("II. Fairness", "A. Testing"), 4),
    ]
    assert "Home" not in " ".join(unit.text for unit in units)


def test_html_chunks_have_paragraph_ids() -> None:
    source = RegulatorySource(
        file_name="t.html",
        doc_id="test_html",
        short_id="TH-1",
        short_title="Test HTML",
        title="Test HTML letter",
        issuer="Test",
        doc_date="2024-01-01",
        url="https://example.org/t",
        file_format="html",
    )
    chunks = units_to_chunks(html_to_units(NUMBERED_HTML), source, ChunkingParams(), count_words)
    assert [chunk.chunk_id for chunk in chunks] == ["TH-1-pre-1", "TH-1-para1-1", "TH-1-para3-1", "TH-1-para4-1"]
    assert chunks[2].paragraphs == [3]
    assert chunks[2].text == "Third paragraph intro:\nStep one.\nStep two."
    assert chunks[1].title == "Test HTML, para. 1-2: I. Purpose"


def test_sources_match_the_download_script() -> None:
    for source in REGULATORY_SOURCES:
        url, _marker = download_data.REGULATORY_DOCS[source.file_name]
        assert source.url == url
    for file_name in SKIPPED_REGULATORY_FILES:
        assert file_name in download_data.REGULATORY_DOCS


def test_load_corpus_reports_missing_regulatory_files(tmp_path: Path) -> None:
    corpus = load_corpus(regulatory_dir=tmp_path)
    assert sorted(corpus.missing_regulatory) == sorted(source.file_name for source in REGULATORY_SOURCES)
    assert set(corpus.chunk_counts()) == {"northwind_business_credit", "harborline_mutual_commercial"}
    assert len(corpus.checksum()) == 64
    assert load_corpus(regulatory_dir=tmp_path, include_regulatory=False).missing_regulatory == []
