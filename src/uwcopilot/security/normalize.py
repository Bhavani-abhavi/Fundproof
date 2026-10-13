"""The canonical normalization API for untrusted text (S2): use this module, not docintel/normalize.py.

Untrusted text (an applicant's notes, OCR output, retrieved snippets, a model's answer before it is scored) is
normalized before the injection tripwire reads it and before it is rendered into a prompt. normalize_text():

1. strips invisible characters: zero-width characters U+200B-U+200F, the word joiner U+2060, the byte order mark
   U+FEFF, the Unicode tag characters U+E0000-U+E007F ("ASCII smuggling"), the soft hyphen U+00AD and the invisible
   operators U+2061-U+2064;
2. applies NFKC (fullwidth letters become ASCII, ligatures are split, the no-break space becomes a space);
3. folds a small documented table of homoglyphs (Cyrillic and Greek letters that look Latin, typographic quotes and
   dashes);
4. collapses runs of whitespace to one space.

detection_text() is what a tripwire should read: the normalized visible text plus the ASCII message hidden in tag
characters, decoded and normalized, so a smuggled instruction is checked too (the prompt itself never shows it).

Where the implementation lives: for now the functions are imported from uwcopilot.docintel.normalize, which the
Phase 2 build wrote and still owns. When Phase 2 is finished, the implementation will move into this module and
docintel/normalize.py will become a thin re-export, so every import of this module keeps working unchanged. The
comparison keys for names and FEINs (canonical_name, canonical_fein) are document-consistency helpers, not injection
controls, and stay in docintel.
"""

from uwcopilot.docintel.normalize import (
    HOMOGLYPHS,
    INVISIBLE_CODEPOINTS,
    TAG_FIRST,
    TAG_LAST,
    fold_homoglyphs,
    hidden_tag_text,
    is_invisible,
    normalize_text,
    strip_invisible,
)

__all__ = [
    "HOMOGLYPHS",
    "INVISIBLE_CODEPOINTS",
    "TAG_FIRST",
    "TAG_LAST",
    "detection_text",
    "fold_homoglyphs",
    "hidden_tag_text",
    "is_invisible",
    "normalize_text",
    "strip_invisible",
]


def detection_text(raw: str) -> str:
    """The text a tripwire should read: normalized visible text, then any decoded tag-character message.

    "Thanks" followed by tag characters spelling "ignore previous instructions" gives
    "Thanks ignore previous instructions". Text without tag characters gives normalize_text(raw).
    """
    visible = normalize_text(raw)
    hidden = normalize_text(hidden_tag_text(raw))
    if hidden:
        return f"{visible} {hidden}".strip()
    return visible
