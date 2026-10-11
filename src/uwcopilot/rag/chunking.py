"""Generic text-window chunking used for the regulatory documents.

Regulatory PDFs and web pages are long and do not have the project's section ids, so they are cut into windows:

1. The loaders (regulatory.py) turn a document into TextUnits: sentences, each carrying the headings above it
   and the page or numbered paragraph it came from.
2. Footnotes and endnotes become their own units and are placed right after the sentence that references them,
   so a note travels with the text it explains (insert_notes).
3. Units under the same headings are packed greedily into windows of at most max_tokens tokens. Each new window
   starts with the last few sentences of the previous one (at most overlap_tokens), so a statement cut at a window
   boundary still appears whole in one window (pack_windows).

Windows never cross a heading boundary. Token counts come from a counting function passed in by the caller: the
index build uses the embedding model's own tokenizer, and unit tests use a word count.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

TokenCounter = Callable[[str], int]

# A sentence ends at ., ! or ? (optionally followed by a closing quote or bracket) and the next one starts with
# an uppercase letter, an opening quote or an opening bracket. Only the whitespace between them is removed.
SENTENCE_BOUNDARY = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"”’)\]]))\s+(?=[A-Z“\"(\[])")


def count_words(text: str) -> int:
    """A simple token counter: whitespace-separated words. Used in tests and as a default."""
    return len(text.split())


@dataclass(frozen=True)
class ChunkingParams:
    """Sizes in tokens of the counting function.

    - max_tokens / overlap_tokens: regulatory windows (the header line added for embedding is not counted, which
      is why the window limit sits well below the embedding model's 512-token input).
    - max_section_tokens: a guideline section whose embedding text is longer than this is split into parts.
    """

    max_tokens: int = 400
    overlap_tokens: int = 60
    max_section_tokens: int = 500

    def __post_init__(self) -> None:
        """Overlap must be smaller than the window, or windows would never advance."""
        if self.max_tokens <= 0 or self.overlap_tokens < 0 or self.overlap_tokens >= self.max_tokens:
            raise ValueError("need max_tokens > overlap_tokens >= 0")
        if self.max_section_tokens <= 0:
            raise ValueError("max_section_tokens must be positive")


@dataclass(frozen=True)
class TextUnit:
    """A sentence (or a note) with where it came from.

    - heading_path: headings above the text, outermost first. Windows never mix two heading paths.
    - pages: 1-based PDF pages the text spans (a sentence can run across a page break).
    - paragraph: the document's own paragraph number, when it numbers paragraphs.
    - block: index of the source paragraph or list item; units from different blocks are joined with a newline.
    - note_refs: keys of footnotes referenced in this unit (see insert_notes).
    """

    text: str
    heading_path: tuple[str, ...]
    pages: tuple[int, ...] = ()
    paragraph: int | None = None
    block: int = 0
    note_refs: tuple[str, ...] = field(default=())


def normalize_space(text: str) -> str:
    """Collapse all whitespace (including non-breaking spaces) to single spaces and trim the ends."""
    return " ".join(text.replace("\xa0", " ").split())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences at ., ! or ? followed by whitespace and an uppercase start.

    It is a plain rule, not a language model: an abbreviation followed by a capitalized word ("D. Guynn") is split
    too. That is harmless here because sentences are only packing units; windows join them back together.
    """
    cleaned = normalize_space(text)
    if not cleaned:
        return []
    return [part for part in SENTENCE_BOUNDARY.split(cleaned) if part.strip()]


def split_long_unit(unit: TextUnit, max_tokens: int, count_tokens: TokenCounter) -> list[TextUnit]:
    """Split a unit that is longer than max_tokens: first into sentences, then (if still too long) into words."""
    if count_tokens(unit.text) <= max_tokens:
        return [unit]
    pieces: list[TextUnit] = []
    sentences = split_sentences(unit.text)
    if len(sentences) > 1:
        for sentence in sentences:
            pieces.extend(split_long_unit(replace(unit, text=sentence), max_tokens, count_tokens))
        return pieces
    words = unit.text.split()
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and count_tokens(candidate) > max_tokens:
            pieces.append(replace(unit, text=" ".join(current)))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(replace(unit, text=" ".join(current)))
    return pieces


@dataclass(frozen=True)
class Note:
    """A footnote or endnote: its label as printed (for example "3") and its text."""

    label: str
    text: str
    pages: tuple[int, ...] = ()


def insert_notes(units: Sequence[TextUnit], notes: dict[str, Note]) -> list[TextUnit]:
    """Place each note right after the first unit that references it; append unreferenced notes at the end.

    A placed note takes the heading path, paragraph and block of the referencing unit, so it lands in the same
    window as the sentence it explains. Its text is prefixed with "[Note <label>]".
    """
    placed: set[str] = set()
    result: list[TextUnit] = []
    for unit in units:
        result.append(unit)
        for key in unit.note_refs:
            note = notes.get(key)
            if note is None or key in placed:
                continue
            placed.add(key)
            result.append(
                TextUnit(
                    text=f"[Note {note.label}] {note.text}",
                    heading_path=unit.heading_path,
                    pages=note.pages or unit.pages,
                    paragraph=unit.paragraph,
                    block=unit.block,
                )
            )
    for key, note in notes.items():
        if key not in placed:
            result.append(TextUnit(text=f"[Note {note.label}] {note.text}", heading_path=("Notes",), pages=note.pages))
    return result


def group_by_heading(units: Sequence[TextUnit]) -> list[list[TextUnit]]:
    """Split units into runs of consecutive units that share the same heading path."""
    groups: list[list[TextUnit]] = []
    for unit in units:
        if groups and groups[-1][0].heading_path == unit.heading_path:
            groups[-1].append(unit)
        else:
            groups.append([unit])
    return groups


def pack_windows(
    units: Sequence[TextUnit],
    params: ChunkingParams,
    count_tokens: TokenCounter = count_words,
) -> list[list[TextUnit]]:
    """Pack units into windows of at most params.max_tokens tokens, with overlap, never crossing a heading.

    The token count of a window is the sum of its units' counts. With a tokenizer that adds special tokens to
    every call this overestimates the joined text slightly, which keeps windows safely under the limit.
    """
    windows: list[list[TextUnit]] = []
    for group in group_by_heading(units):
        pieces: list[TextUnit] = []
        for unit in group:
            pieces.extend(split_long_unit(unit, params.max_tokens, count_tokens))
        sizes = [count_tokens(piece.text) for piece in pieces]
        current: list[int] = []  # indices into pieces
        for index, size in enumerate(sizes):
            current_size = sum(sizes[i] for i in current)
            if current and current_size + size > params.max_tokens:
                windows.append([pieces[i] for i in current])
                current = overlap_tail(current, sizes, params.overlap_tokens)
                while current and sum(sizes[i] for i in current) + size > params.max_tokens:
                    current = current[1:]
            current.append(index)
        if current:
            windows.append([pieces[i] for i in current])
    return windows


def overlap_tail(indices: list[int], sizes: list[int], overlap_tokens: int) -> list[int]:
    """The trailing units of a window whose sizes add up to at most overlap_tokens (they start the next window)."""
    tail: list[int] = []
    total = 0
    for index in reversed(indices):
        if total + sizes[index] > overlap_tokens:
            break
        tail.insert(0, index)
        total += sizes[index]
    return tail


def join_units(units: Sequence[TextUnit]) -> str:
    """Join a window's units: a space within the same block, a newline between blocks."""
    parts: list[str] = []
    previous_block: int | None = None
    for unit in units:
        if parts:
            parts.append(" " if unit.block == previous_block else "\n")
        parts.append(unit.text)
        previous_block = unit.block
    return "".join(parts)
