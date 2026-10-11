"""Tests for the generic windowing code (sentences, notes, token windows with overlap)."""

import pytest

from uwcopilot.rag.chunking import (
    ChunkingParams,
    Note,
    TextUnit,
    count_words,
    insert_notes,
    join_units,
    pack_windows,
    split_long_unit,
    split_sentences,
)


def units_of(texts: list[str], heading: tuple[str, ...] = ("H",), block: int = 0) -> list[TextUnit]:
    """TextUnits with the same heading and block."""
    return [TextUnit(text=text, heading_path=heading, block=block) for text in texts]


def test_split_sentences_splits_at_sentence_ends_and_keeps_closing_quotes() -> None:
    text = "The term is “model risk.” It matters. Is it (e.g., spreadsheets) included? Yes."
    assert split_sentences(text) == [
        "The term is “model risk.”",
        "It matters.",
        "Is it (e.g., spreadsheets) included?",
        "Yes.",
    ]


def test_split_sentences_does_not_split_before_lowercase_or_digits() -> None:
    assert split_sentences("See 12 U.S.C. 1813 for details. Next one.") == [
        "See 12 U.S.C. 1813 for details.",
        "Next one.",
    ]
    assert split_sentences("   ") == []


def test_chunking_params_validation() -> None:
    with pytest.raises(ValueError):
        ChunkingParams(max_tokens=50, overlap_tokens=50)
    with pytest.raises(ValueError):
        ChunkingParams(max_section_tokens=0)


def test_pack_windows_respects_max_tokens_and_overlaps() -> None:
    sentences = [f"Sentence number {index} has exactly seven words." for index in range(10)]
    windows = pack_windows(units_of(sentences), ChunkingParams(max_tokens=21, overlap_tokens=7), count_words)
    for window in windows:
        assert sum(count_words(unit.text) for unit in window) <= 21
    # Each new window starts with the last sentence of the previous one (7 words of overlap).
    for previous, current in zip(windows, windows[1:], strict=False):
        assert current[0] == previous[-1]
    covered = {unit.text for window in windows for unit in window}
    assert covered == set(sentences)


def test_pack_windows_never_crosses_a_heading() -> None:
    units = units_of(["One two three."], heading=("A",)) + units_of(["Four five six."], heading=("B",))
    windows = pack_windows(units, ChunkingParams(max_tokens=100, overlap_tokens=10), count_words)
    assert [[unit.heading_path for unit in window] for window in windows] == [[("A",)], [("B",)]]


def test_split_long_unit_splits_by_sentence_then_by_words() -> None:
    unit = TextUnit(text="Alpha beta gamma delta. Epsilon zeta eta theta iota kappa.", heading_path=("H",))
    pieces = split_long_unit(unit, max_tokens=4, count_tokens=count_words)
    assert [piece.text for piece in pieces] == ["Alpha beta gamma delta.", "Epsilon zeta eta theta", "iota kappa."]
    assert all(piece.heading_path == ("H",) for piece in pieces)


def test_insert_notes_places_note_after_referencing_unit_and_appends_unreferenced() -> None:
    units = [
        TextUnit(text="First.", heading_path=("H",), pages=(4,), note_refs=("p4:1",)),
        TextUnit(text="Second.", heading_path=("H",), pages=(4,)),
    ]
    notes = {"p4:1": Note(label="1", text="A footnote.", pages=(4,)), "p9:2": Note(label="2", text="Orphan.")}
    result = insert_notes(units, notes)
    assert [unit.text for unit in result] == ["First.", "[Note 1] A footnote.", "Second.", "[Note 2] Orphan."]
    assert result[1].heading_path == ("H",) and result[1].pages == (4,)
    assert result[3].heading_path == ("Notes",)


def test_join_units_uses_newline_between_blocks() -> None:
    units = units_of(["A.", "B."], block=0) + units_of(["C."], block=1)
    assert join_units(units) == "A. B.\nC."
