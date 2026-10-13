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
person names (punctuation, "&"/"and" and company-suffix spellings folded) and canonical_fein() for FEINs.
Nothing here interprets the text; it only makes comparisons fair.

The homoglyph table is written with \\u escapes and the Unicode character name in a comment, because a Cyrillic
small a (U+0430) and a Latin "a" look the same in an editor.
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
    "\u0430": "a",  # cyrillic small letter a
    "\u0435": "e",  # cyrillic small letter ie
    "\u043e": "o",  # cyrillic small letter o
    "\u0440": "p",  # cyrillic small letter er
    "\u0441": "c",  # cyrillic small letter es
    "\u0443": "y",  # cyrillic small letter u
    "\u0445": "x",  # cyrillic small letter ha
    "\u0455": "s",  # cyrillic small letter dze
    "\u0456": "i",  # cyrillic small letter byelorussian-ukrainian i
    "\u0458": "j",  # cyrillic small letter je
    "\u04bb": "h",  # cyrillic small letter shha
    "\u04cf": "l",  # cyrillic small letter palochka
    "\u0501": "d",  # cyrillic small letter komi de
    # Cyrillic uppercase
    "\u0410": "A",  # cyrillic capital letter a
    "\u0412": "B",  # cyrillic capital letter ve
    "\u0415": "E",  # cyrillic capital letter ie
    "\u041a": "K",  # cyrillic capital letter ka
    "\u041c": "M",  # cyrillic capital letter em
    "\u041d": "H",  # cyrillic capital letter en
    "\u041e": "O",  # cyrillic capital letter o
    "\u0420": "P",  # cyrillic capital letter er
    "\u0421": "C",  # cyrillic capital letter es
    "\u0422": "T",  # cyrillic capital letter te
    "\u0425": "X",  # cyrillic capital letter ha
    "\u0405": "S",  # cyrillic capital letter dze
    "\u0406": "I",  # cyrillic capital letter byelorussian-ukrainian i
    "\u0408": "J",  # cyrillic capital letter je
    # Greek
    "\u03bf": "o",  # greek small letter omicron
    "\u03b1": "a",  # greek small letter alpha
    "\u03b9": "i",  # greek small letter iota
    "\u03ba": "k",  # greek small letter kappa
    "\u03bd": "v",  # greek small letter nu
    "\u03c1": "p",  # greek small letter rho
    "\u0391": "A",  # greek capital letter alpha
    "\u0392": "B",  # greek capital letter beta
    "\u0395": "E",  # greek capital letter epsilon
    "\u0396": "Z",  # greek capital letter zeta
    "\u0397": "H",  # greek capital letter eta
    "\u0399": "I",  # greek capital letter iota
    "\u039a": "K",  # greek capital letter kappa
    "\u039c": "M",  # greek capital letter mu
    "\u039d": "N",  # greek capital letter nu
    "\u039f": "O",  # greek capital letter omicron
    "\u03a1": "P",  # greek capital letter rho
    "\u03a4": "T",  # greek capital letter tau
    "\u03a5": "Y",  # greek capital letter upsilon
    "\u03a7": "X",  # greek capital letter chi
    # Typographic punctuation
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark
    "\u201c": '"',  # left double quotation mark
    "\u201d": '"',  # right double quotation mark
    "\u2010": "-",  # hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
}

WHITESPACE_RUN = re.compile(r"\s+")
# Punctuation that varies between documents. Periods are deleted ("L.L.C" -> "llc", "Inc." -> "inc"); the other
# characters become spaces.
NAME_PERIOD = "."
NAME_PUNCTUATION = re.compile(r"[,'\"()/\\-]")
# Spellings of the same word or company suffix that documents use interchangeably, folded to one form. Applied to
# whole words after punctuation is removed; the phrase entry is applied first.
NAME_PHRASES: dict[str, str] = {"limited liability company": "llc"}
NAME_WORDS: dict[str, str] = {
    "&": "and",
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
}


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

    Steps: normalize_text, casefold, delete periods, turn the other NAME_PUNCTUATION into spaces, fold the
    NAME_PHRASES and NAME_WORDS spellings, collapse spaces. "Harbor Bakery, L.L.C.", "HARBOR BAKERY LLC" and
    "Harbor Bakery Limited Liability Company" give the same key ("harbor bakery llc"); "Smith & Sons Co." and
    "Smith and Sons Company" give "smith and sons co". A one-letter typo still gives a different key, which is
    what the name checks need to see.
    """
    if not name:
        return ""
    text = normalize_text(name).casefold().replace(NAME_PERIOD, "")
    text = NAME_PUNCTUATION.sub(" ", text).replace("&", " & ")
    text = " ".join(text.split())
    for phrase, replacement in NAME_PHRASES.items():
        text = f" {text} ".replace(f" {phrase} ", f" {replacement} ").strip()
    words = [NAME_WORDS.get(word, word) for word in text.split()]
    return " ".join(words)


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
