"""Table-row parsers of the rules extractor that more than one document type uses.

Rows are parsed from their word types rather than from fixed column positions, so the parsers survive small OCR
shifts and both bank layouts:
- owner rows (loan application, insurance application, ownership statement): "Name | Type | Title | Ownership % |
  Control person | Country". The Type word (Person / Organization) splits the name from the title; the first
  word with a "%" after it is the ownership share; the next word is the control-person Yes/No; the rest is the
  country.
- two-slot premium rows (prior carriers on the insurance application, earned premium by policy year on a loss
  run): the words after the date are the two premium cells; a cell is an amount or "n/a" (no premium, value
  None). When the header line was found, each amount goes to the column whose right edge is nearest; otherwise
  the cells are taken in order when there are exactly two.
"""

from rapidfuzz import fuzz

from uwcopilot.docintel.layout import Word
from uwcopilot.docintel.parsing import code_from_label, parse_amount, parse_bool, parse_percent, parse_text
from uwcopilot.docintel.rules_common import LineRef, cite_words, nearest_column, words_text
from uwcopilot.docintel.schemas import Cited, ExtractedOwner

ENTITY_TYPES = ("person", "organization")
TYPE_MIN_SCORE = 80.0


def entity_type_of(word: Word) -> str | None:
    """ "person" / "organization" when the word spells one (approximately), else None."""
    text = word.text.strip(".,:;").casefold()
    for entity_type in ENTITY_TYPES:
        if text == entity_type or fuzz.ratio(text, entity_type) >= TYPE_MIN_SCORE:
            return entity_type
    return None


def percent_index(words: list[Word], start: int) -> int | None:
    """Index of the first word at or after start that is a percentage printed with "%"."""
    for index in range(start, len(words)):
        if "%" in words[index].text and parse_percent(words[index].text) is not None:
            return index
    return None


def parse_owner_row(ref: LineRef, doc_id: str) -> ExtractedOwner | None:
    """An owners-table row (see the module docstring), or None when the line is not one."""
    words = ref.words
    type_index = next((index for index, word in enumerate(words) if entity_type_of(word)), None)
    if type_index is None or type_index == 0:
        return None
    pct_at = percent_index(words, type_index + 1)
    if pct_at is None:
        return None
    owner = ExtractedOwner(
        name=cite_words(ref, words[:type_index], parse_text(words_text(words[:type_index])), doc_id),
        entity_type=cite_words(ref, [words[type_index]], entity_type_of(words[type_index]), doc_id),
        ownership_pct=cite_words(ref, [words[pct_at]], parse_percent(words[pct_at].text), doc_id),
    )
    title_words = words[type_index + 1 : pct_at]
    if title_words:
        owner.title = cite_words(ref, title_words, parse_text(words_text(title_words)), doc_id)
    rest = words[pct_at + 1 :]
    if rest and parse_bool(rest[0].text) is not None:
        owner.is_control_person = cite_words(ref, [rest[0]], parse_bool(rest[0].text), doc_id)
        rest = rest[1:]
    if rest:
        owner.country = cite_words(ref, rest, parse_text(words_text(rest)), doc_id)
    return owner


def owner_rows(refs: list[LineRef], doc_id: str) -> list[ExtractedOwner]:
    """Every owners-table row among the lines, in order."""
    owners = []
    for ref in refs:
        owner = parse_owner_row(ref, doc_id)
        if owner is not None:
            owners.append(owner)
    return owners


def premium_cells(
    ref: LineRef, cells: list[Word], names: tuple[str, str], columns: dict[str, float], doc_id: str
) -> dict[str, Cited]:
    """Column name -> cited premium for the cells after the date (see the module docstring).

    names are the two column names in printed order; columns holds the right edge of each column title found on
    the header line.
    """
    if all(name in columns for name in names):
        pairs = [(nearest_column(word.x1, columns), word) for word in cells]
    elif len(cells) == len(names):
        pairs = list(zip(names, cells, strict=True))
    else:
        pairs = []
    result: dict[str, Cited] = {}
    for name, word in pairs:
        value = parse_amount(word.text)
        if name is not None and value is not None and name not in result:
            result[name] = cite_words(ref, [word], value, doc_id)
    return result


def code_cited(ref: LineRef, words: list[Word], doc_id: str) -> Cited:
    """A printed label cited as its code ("Term loan" -> "term_loan")."""
    return cite_words(ref, words, code_from_label(words_text(words)), doc_id)
