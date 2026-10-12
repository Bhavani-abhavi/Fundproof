"""Text normalization for untrusted document text (S2), run before injection detection and name comparison.

Attackers can hide instructions from a keyword detector with characters a reader never sees or that only look
like ASCII letters. normalize_text() undoes the common tricks, in this order:

1. strip invisible characters: zero-width characters U+200B-U+200F, the word joiner U+2060, the byte order mark
   U+FEFF and the Unicode "tag" characters U+E0000-U+E007F (used for "ASCII smuggling"), plus two more that
   render as nothing inside a word: the soft hyphen U+00AD and the invisible operators U+2061-U+2064;
2. NFKC normalization (fullwidth letters become ASCII, ligatures such as "fi" are split, the no-break space
   becomes a plain space);
3. a small documented homoglyph fold (HOMOGLYPHS below): Cyrillic and Greek letters that look like Latin letters
   are replaced by those Latin letters, and typographic quotes and dashes by their ASCII forms;
4. runs of whitespace become one space.

Tag characters can also carry a hidden ASCII message (U+E0041 is a hidden "A"). hidden_tag_text() decodes that
message so the detector can read it too; normalize_text() itself only removes the characters.

The module also holds the comparison keys used by the consistency checks: canonical_name() for business and
person names and canonical_fein() for FEINs. Nothing here interprets the text; it only makes comparisons fair.
"""

import re
import unicodedata

# 1. Invisible characters removed before anything else (zero-width, word joiner, BOM, soft hyphen, invisible operators).
INVISIBLE_CODEPOINTS: tuple[int, ...] = (*range(0x200B, 0x2010), 0x2060, 0xFEFF, 0x00AD, *range(0x2061, 0x2065))
TAG_FIRST = 0xE0000
TAG_LAST = 0xE007F
# Tag characters U+E0020-U+E007E mirror printable ASCII 0x20-0x7E.
TAG_PRINTABLE_FIRST = 0xE0020
TAG_PRINTABLE_LAST = 0xE007E
TAG_TO_ASCII_OFFSET = 0xE0000

# 3. Homoglyph fold: lookalike letters from Cyrillic and Greek, plus typographic punctuation. Deliberately small;
# every entry is a character that renders like the Latin letter in common fonts.
HOMOGLYPHS: dict[str, str] = {
    # Cyrillic lowercase
    "а": "a",  # а
    "е": "e",  # е
    "о": "o",  # о
    "р": "p",  # р
    "с": "c",  # с
    "у": "y",  # у
    "х": "x",  # х
    "ѕ": "s",  # ѕ
    "і": "i",  # і
    "ј": "j",  # ј
    "һ": "h",  # һ
    "ӏ": "l",  # ӏ
    "ԁ": "d",  # ԁ
    # Cyrillic uppercase
    "А": "A",  # А
    "В": "B",  # В
    "Е": "E",  # Е
    "К": "K",  # К
    "М": "M",  # М
    "Н": "H",  # Н
    "О": "O",  # О
    "Р": "P",  # Р
    "С": "C",  # С
    "Т": "T",  # Т
    "Х": "X",  # Х
    "Ѕ": "S",  # Ѕ
    "І": "I",  # І
    "Ј": "J",  # Ј
    # Greek
    "ο": "o",  # ο
    "α": "a",  # α
    "ι": "i",  # ι
    "κ": "k",  # κ
    "ν": "v",  # ν
    "ρ": "p",  # ρ
    "Α": "A",  # Α
    "Β": "B",  # Β
    "Ε": "E",  # Ε
    "Ζ": "Z",  # Ζ
    "Η": "H",  # Η
    "Ι": "I",  # Ι
    "Κ": "K",  # Κ
    "Μ": "M",  # Μ
    "Ν": "N",  # Ν
    "Ο": "O",  # Ο
    "Ρ": "P",  # Ρ
    "Τ": "T",  # Τ
    "Υ": "Y",  # Υ
    "Χ": "X",  # Χ
    # Typographic punctuation
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "‐": "-",  # hyphen (NFKC turns the non-breaking hyphen U+2011 into this)
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "−": "-",  # minus sign
}

WHITESPACE_RUN = re.compile(r"\s+")
# Characters dropped from names before comparison (punctuation that varies between documents).
NAME_PUNCTUATION = re.compile(r"[.,'\"()&/\\-]")


def is_invisible(character: str) -> bool:
    """True for a character in INVISIBLE_CODEPOINTS or a Unicode tag character."""
    codepoint = ord(character)
    return codepoint in INVISIBLE_CODEPOINTS or TAG_FIRST <= codepoint <= TAG_LAST


def strip_invisible(text: str) -> str:
    """Remove zero-width, word-joiner, BOM and tag characters."""
    return "".join(character for character in text if not is_invisible(character))


def hidden_tag_text(text: str) -> str:
    """The ASCII message hidden in Unicode tag characters ("" when there is none)."""
    hidden = []
    for character in text:
        codepoint = ord(character)
        if TAG_PRINTABLE_FIRST <= codepoint <= TAG_PRINTABLE_LAST:
            hidden.append(chr(codepoint - TAG_TO_ASCII_OFFSET))
    return "".join(hidden)


def fold_homoglyphs(text: str) -> str:
    """Replace lookalike letters and typographic punctuation using the HOMOGLYPHS table."""
    return "".join(HOMOGLYPHS.get(character, character) for character in text)


def normalize_text(text: str) -> str:
    """Strip invisible characters, apply NFKC, fold homoglyphs and collapse whitespace (see the module docstring)."""
    visible = strip_invisible(text)
    compatible = unicodedata.normalize("NFKC", visible)
    folded = fold_homoglyphs(compatible)
    return WHITESPACE_RUN.sub(" ", folded).strip()


def canonical_name(name: str | None) -> str:
    """A comparison key for a business or person name: normalized, casefolded, without punctuation.

    "Harbor Bakery, L.L.C." and "HARBOR BAKERY LLC" give the same key ("harbor bakery llc"). A one-letter typo
    still gives a different key, which is what the name checks need to see.
    """
    if not name:
        return ""
    text = normalize_text(name).casefold()
    text = text.replace("l.l.c.", "llc")
    text = NAME_PUNCTUATION.sub(" ", text)
    return WHITESPACE_RUN.sub(" ", text).strip()


def edit_distance(first: str, second: str) -> int:
    """Levenshtein distance: the fewest single-character insertions, deletions or substitutions between two texts."""
    previous = list(range(len(second) + 1))
    for row, first_char in enumerate(first, start=1):
        current = [row]
        for column, second_char in enumerate(second, start=1):
            insert = current[column - 1] + 1
            delete = previous[column] + 1
            substitute = previous[column - 1] + (0 if first_char == second_char else 1)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def digits_only(text: str | None) -> str:
    """The ASCII digits of a normalized text, in order ("" for None). Used for account numbers and FEINs."""
    if not text:
        return ""
    return "".join(character for character in normalize_text(text) if character in "0123456789")


def canonical_fein(fein: str | None) -> str:
    """A comparison key for a FEIN: its digits as NN-NNNNNNN when there are nine, otherwise the digits alone."""
    digits = digits_only(fein)
    if len(digits) == 9:
        return f"{digits[:2]}-{digits[2:]}"
    return digits
