"""Deterministic numeric-provenance checker for free-form text (docs/innovations.md, SIG step 2).

Where the model must restate numbers (free-form RAG answers, KYC adverse-media memos), this checker asks one question
per number: does the same quantity appear in a source that the sentence cites and that was in the model's context?

check_numbers(text, sources) splits the text into sentences. For each numeric mention in a sentence:
- UNTRACEABLE_NUMBER: no cited source (among `sources`) contains the same quantity (numbers.same_quantity). A sentence
  that cites nothing makes all its numbers untraceable. A number the model computed itself (a derived number) is
  untraceable by design: the checker cannot verify arithmetic it did not see.
- COMPARATOR_MISMATCH: the number is traceable, but the comparator next to it in the answer (for example "at least")
  belongs to a different class than every comparator the source pairs with that value (for example "more than").
- comparator_unverifiable (not an error, counted): the source pairs the value with SEVERAL comparator classes
  (NBC-3.4 has both "10.0% or less" and "more than 10.0%"), so the comparator cannot be checked.

Comparator classes: GE (at least, or more, at or above, minimum, no less than), GT (more than, exceeds, above, greater
than, over), LE (at most, or less, up to, maximum, no more than, not more than), LT (below, less than, under, fewer
than, lower than). The comparator of a number is the closest lexicon phrase within 5 tokens on either side.

Known blind spot (measured by the mutation test as `wrong_row`): when a section contains the same value in another
row of a table (the revolver's 1.35 attached to a term loan), the number looks traceable. The checker sees values,
not row bindings.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from uwcopilot.provenance.numbers import NumberMention, extract_numbers, same_quantity

ComparatorClass = Literal["GE", "GT", "LE", "LT"]
MentionStatus = Literal["traceable", "untraceable", "comparator_mismatch", "comparator_unverifiable"]

# Longest phrases first so "no more than" is not read as "more than".
COMPARATOR_LEXICON: tuple[tuple[str, ComparatorClass], ...] = (
    ("at or above", "GE"),
    ("at or below", "LE"),
    ("no less than", "GE"),
    ("not less than", "GE"),
    ("no more than", "LE"),
    ("not more than", "LE"),
    ("no fewer than", "GE"),
    ("greater than", "GT"),
    ("higher than", "GT"),
    ("larger than", "GT"),
    ("more than", "GT"),
    ("less than", "LT"),
    ("lower than", "LT"),
    ("fewer than", "LT"),
    ("smaller than", "LT"),
    ("at least", "GE"),
    ("at most", "LE"),
    ("or more", "GE"),
    ("or greater", "GE"),
    ("or higher", "GE"),
    ("or less", "LE"),
    ("or fewer", "LE"),
    ("or lower", "LE"),
    ("up to", "LE"),
    ("minimum", "GE"),
    ("min", "GE"),
    ("maximum", "LE"),
    ("max", "LE"),
    ("exceeds", "GT"),
    ("exceeding", "GT"),
    ("exceed", "GT"),
    ("above", "GT"),
    ("below", "LT"),
    ("under", "LT"),
)
# Words of a phrase may be separated by any whitespace, including a line break inside a markdown paragraph.
COMPARATOR_PATTERN = re.compile(
    r"\b(?:" + "|".join(r"\s+".join(map(re.escape, phrase.split())) for phrase, _ in COMPARATOR_LEXICON) + r")\b",
    re.IGNORECASE,
)
CLASS_OF: dict[str, ComparatorClass] = {phrase: cls for phrase, cls in COMPARATOR_LEXICON}
WINDOW_TOKENS = 5
CITATION_IDS = re.compile(r"\[([^\[\]]+)\]")
# Split after a full stop before a capital letter; never before "[" so a trailing citation stays with its sentence.
# A soft line wrap inside a markdown paragraph is not a break; a blank line, a bullet or a table row is.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z{\"'(])|\n\s*\n|\n(?=\s*(?:[-*|]|\d+[.)])\s?)")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE = re.compile(r"^\s*\|\s*:?-{2,}")


@dataclass(frozen=True)
class MentionCheck:
    """The verdict on one number in the checked text."""

    sentence_index: int
    mention: NumberMention
    status: MentionStatus
    comparator: ComparatorClass | None
    matched_source: str | None
    cited: tuple[str, ...]
    detail: str


@dataclass
class NumericCheck:
    """The verdict on a whole text."""

    mentions: list[MentionCheck] = field(default_factory=list)

    @property
    def untraceable(self) -> list[MentionCheck]:
        """Numbers no cited source contains."""
        return [item for item in self.mentions if item.status == "untraceable"]

    @property
    def comparator_mismatches(self) -> list[MentionCheck]:
        """Numbers whose comparator disagrees with every comparator the source uses for that value."""
        return [item for item in self.mentions if item.status == "comparator_mismatch"]

    @property
    def comparator_unverifiable(self) -> list[MentionCheck]:
        """Numbers whose source uses the value with several comparator classes."""
        return [item for item in self.mentions if item.status == "comparator_unverifiable"]

    @property
    def flagged(self) -> bool:
        """True when any number is untraceable or has a mismatched comparator."""
        return bool(self.untraceable or self.comparator_mismatches)


# "1.25. [NBC-4.2] Next" -> "1.25 [NBC-4.2]. Next": a citation written after the full stop belongs to that sentence.
TRAILING_CITATION = re.compile(r"([.!?])\s*((?:\[[^\[\]]+\]\s*)+)(?=\s|$)")


def split_sentences(text: str) -> list[tuple[int, str]]:
    """Sentences with their start offsets (in the text after trailing citations are moved before the full stop).

    A citation written right after a sentence's full stop stays with that sentence, so the next sentence cannot
    borrow it.
    """
    text = TRAILING_CITATION.sub(lambda match: f" {match.group(2).strip()}{match.group(1)} ", text)
    sentences = []
    position = 0
    for part in SENTENCE_SPLIT.split(text):
        start = text.find(part, position)
        if part.strip():
            sentences.append((start, part))
        position = start + len(part)
    return sentences


def cited_ids(sentence: str) -> tuple[str, ...]:
    """Ids cited in a sentence: '[NBC-4.2]', '[NBC-4.2, NBC-4.5]', '[S1][S3]' -> ('NBC-4.2', ...)."""
    ids: list[str] = []
    for block in CITATION_IDS.findall(sentence):
        for item in re.split(r"[,;]\s*", block):
            item = item.strip()
            if item and not item.startswith("^"):
                ids.append(item)
    return tuple(dict.fromkeys(ids))


def token_distance(text: str, start: int, end: int) -> int:
    """Number of whitespace-separated tokens strictly between two character positions."""
    between = text[min(start, end) : max(start, end)]
    return len(between.split())


def comparator_near(text: str, mention: NumberMention) -> ComparatorClass | None:
    """The class of the closest comparator phrase within WINDOW_TOKENS tokens of the number, or None.

    The window is measured from the first digit or number word, so "3 or more months" finds "or more".
    """
    best: tuple[int, ComparatorClass] | None = None
    anchor_start = mention.start
    anchor_end = mention.start + len(mention.text.split()[0]) if mention.text.split() else mention.end
    for match in COMPARATOR_PATTERN.finditer(text):
        if match.end() <= anchor_start:
            distance = token_distance(text, match.end(), anchor_start)
        elif match.start() >= anchor_end:
            distance = token_distance(text, anchor_end, match.start())
        else:
            distance = 0
        if distance <= WINDOW_TOKENS and (best is None or distance < best[0]):
            best = (distance, CLASS_OF[" ".join(match.group(0).lower().split())])
    return None if best is None else best[1]


def split_cells(line: str) -> list[str]:
    """'| a | b |' -> ['a', 'b']."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def expand_tables(source_text: str) -> str:
    """Append one sentence per table cell, binding the cell to its column header and row label.

    A markdown table states "| Term loan | $25,000 |" under the header "Minimum amount"; the comparator lives in the
    header, far from the cell. Appending "The minimum amount for Term loan is $25,000." lets comparator detection see
    that pairing. The original text is kept unchanged in front.
    """
    lines = source_text.splitlines()
    extra: list[str] = []
    index = 0
    while index < len(lines):
        if not TABLE_ROW.match(lines[index]):
            index += 1
            continue
        block = []
        while index < len(lines) and TABLE_ROW.match(lines[index]):
            block.append(lines[index])
            index += 1
        header = split_cells(block[0])
        for row_line in block[1:]:
            cells = split_cells(row_line)
            if TABLE_RULE.match(row_line) or len(cells) != len(header):
                continue
            for column, cell in zip(header[1:], cells[1:], strict=True):
                extra.append(f"The {column.lower()} for {cells[0]} is {cell}.")
    return source_text if not extra else source_text + "\n\n" + "\n".join(extra)


def source_comparators(source_text: str, mention: NumberMention) -> set[ComparatorClass]:
    """Comparator classes the source pairs with the same quantity (empty when it uses none).

    Tables are expanded first, and each comparator is looked for only inside the sentence (or line) of the value.
    """
    classes: set[ComparatorClass] = set()
    for _, sentence in split_sentences(expand_tables(source_text)):
        for source_mention in extract_numbers(sentence):
            if same_quantity(source_mention, mention):
                found = comparator_near(sentence, source_mention)
                if found is not None:
                    classes.add(found)
    return classes


def check_mention(
    sentence: str, mention: NumberMention, cited: tuple[str, ...], sources: dict[str, str]
) -> tuple[MentionStatus, ComparatorClass | None, str | None, str]:
    """Status, answer comparator, matching source id and a short detail for one number.

    Comparator rule (docs/innovations.md SIG step 2): if the source never pairs the value with the answer's
    comparator class, that is a COMPARATOR_MISMATCH. This includes a source that states the value with no comparator
    at all (an answer adding "at least" asserts something the source does not say); the detail says which case it
    was. When the source pairs the value with several classes, the comparator cannot be checked: the mention passes
    and is counted as comparator_unverifiable.
    """
    comparator = comparator_near(sentence, mention)
    for source_id in cited:
        source_text = sources.get(source_id)
        if source_text is None:
            continue
        if not any(same_quantity(other, mention) for other in extract_numbers(source_text)):
            continue
        if comparator is None:
            return "traceable", None, source_id, "value found in cited source"
        classes = source_comparators(source_text, mention)
        if len(classes) > 1:
            return "comparator_unverifiable", comparator, source_id, f"source uses {sorted(classes)}"
        if comparator in classes:
            return "traceable", comparator, source_id, "value and comparator found in cited source"
        detail = "source states the value without a comparator" if not classes else f"source uses {sorted(classes)}"
        return "comparator_mismatch", comparator, source_id, detail
    detail = "no cited source in context" if not any(item in sources for item in cited) else "value not in sources"
    return "untraceable", comparator, None, detail


def check_numbers(text: str, sources: dict[str, str]) -> NumericCheck:
    """Check every number in text against the cited sources that were in context (see the module docstring).

    sources maps an id (a section id such as "NBC-4.2", or a snippet id such as "S1") to the text the model saw.
    A cited id missing from sources (not in context, or fabricated) cannot make a number traceable.
    """
    result = NumericCheck()
    for index, (_, sentence) in enumerate(split_sentences(text)):
        cited = cited_ids(sentence)
        for mention in extract_numbers(sentence):
            status, comparator, matched, detail = check_mention(sentence, mention, cited, sources)
            result.mentions.append(MentionCheck(index, mention, status, comparator, matched, cited, detail))
    return result
