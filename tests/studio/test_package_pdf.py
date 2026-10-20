"""render_pdf (Studio S6): a real PDF whose every page carries the product name, the disclaimer and "Page x/y" (and
the SYNTHETIC tag), whose own text traces to the ledger, that is deterministic and that refuses an unaudited
package. Figures are drawn offline with matplotlib (no tiles)."""

import pytest
from package_helpers import package, pdf

from uwcopilot.studio.package import PackageNotAudited, audit_pdf, pdf_page_count, pdf_text, render_pdf
from uwcopilot.studio.package.audit import audit_package, page_texts, pages_missing_labels
from uwcopilot.studio.package.constants import FULL_DISCLAIMER, PRODUCT_NAME, SYNTHETIC_BANNER
from uwcopilot.studio.package.models import Package, Paragraph
from uwcopilot.studio.package.pdf_figures import draw_cash_band, draw_map, draw_tornado

METRO, TYPE = "synthetic_metro", "car_wash"
PNG_MAGIC = b"\x89PNG"


def test_the_pdf_starts_with_the_magic_bytes_and_has_pages() -> None:
    document = pdf(METRO, TYPE)
    assert document.startswith(b"%PDF")
    assert pdf_page_count(document) >= 20


def test_every_page_carries_product_disclaimer_page_number_and_synthetic_tag() -> None:
    document = pdf(METRO, TYPE)
    assert pages_missing_labels(document, package(METRO, TYPE)) == {
        "product": [],
        "disclaimer": [],
        "page_number": [],
        "synthetic": [],
    }
    first = " ".join(page_texts(document)[0].split())
    assert " ".join(SYNTHETIC_BANNER.split()) in first and FULL_DISCLAIMER in first and PRODUCT_NAME in first


def test_the_pdf_text_traces_to_the_ledger() -> None:
    assert audit_pdf(pdf(METRO, TYPE), package(METRO, TYPE).ledger) == []


def test_the_pdf_shows_every_section_and_the_evidence_appendix() -> None:
    text = " ".join(pdf_text(pdf(METRO, TYPE)).split())
    for section in package(METRO, TYPE).sections:
        assert f"{section.number}. {section.title}" in text
    assert "Claims of the founder ledger" in text and "Checks (assertions) of the founder ledger" in text


def test_rendering_is_deterministic() -> None:
    assert render_pdf(package(METRO, TYPE)) == pdf(METRO, TYPE)


def with_paragraph(built: Package, draft: str) -> Package:
    """The package with one more paragraph in its first section; its stored audit is left as it was (clean)."""
    first = built.sections[0]
    edited = first.model_copy(update={"blocks": [*first.blocks, Paragraph(draft=draft)]})
    return built.model_copy(update={"sections": [edited, *built.sections[1:]]})


@pytest.mark.parametrize(
    "draft",
    [
        "Revenue could change 37% next year.",
        "The owner adds $777,777 of equity.",
        "Rank 93 of the inputs.",
    ],
)
def test_a_package_with_untraced_numbers_is_not_printed_even_if_its_stored_audit_is_clean(draft: str) -> None:
    built = package(METRO, TYPE)
    assert built.audit.untraced == []
    with pytest.raises(PackageNotAudited):
        render_pdf(with_paragraph(built, draft))


def test_a_label_number_without_a_unit_is_still_layout() -> None:
    built = package(METRO, TYPE)
    assert audit_package(with_paragraph(built, "See the Year 2 column and Change-set 0.")).untraced == []


def test_figures_are_png_images_drawn_from_the_package() -> None:
    built = package(METRO, TYPE)
    assert draw_map(built.map).startswith(PNG_MAGIC)
    assert draw_cash_band(built.ledger, [f"2027-{month:02d}" for month in range(1, 13)]).startswith(PNG_MAGIC)
    assert draw_tornado(built.ledger, ["TORNADO.r1.swing", "TORNADO.r2.swing"]).startswith(PNG_MAGIC)
