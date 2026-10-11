"""The retrieval evaluation set (eval/rag_questions.jsonl): question schema, gold matching and checks.

Each line is one question with its gold references. A gold reference is one of:
- {"section": "NBC-3.2"}: a guideline section (matches the chunk with that id, any part of it);
- {"doc": "frb_sr_26_2", "page": 5, "quote": "..."}: a passage in a PDF;
- {"doc": "nydfs_cl_2024_07", "paragraph": 15, "quote": "..."}: a passage in a numbered-paragraph letter
  (leave "paragraph" out for text in the letter's unnumbered preamble; the quote alone locates it);
- {"any_of": [ref, ref, ...]}: satisfied by any one of the listed references (the same fact stated in two
  places).

A question needs every gold reference (all-of). Unanswerable questions have "gold": [].

Regulatory gold is a short verbatim quote plus the page or paragraph, not a chunk id. Chunk ids depend on the
window size; the quote does not, so the labels stay valid if the chunking changes. A chunk matches a passage
when it comes from that document, covers that page/paragraph, and contains the quote (after normalizing case,
whitespace, quotes and dashes). validate_gold() checks that every gold reference matches at least one indexed
chunk, so a typo in a quote is caught before any number is computed.
"""

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from uwcopilot.common.paths import REPO_ROOT

QUESTIONS_FILE: Path = REPO_ROOT / "eval" / "rag_questions.jsonl"
Category = Literal["single_section", "multi_section", "regulatory", "unanswerable"]
STOPWORDS = frozenset(
    """a an and are as at be by can do does for from has have how i if in is it its of on or our so that the
    their them then there these they this to under was we what when where which who whom why will with would
    you your should must may any all not no""".split()
)
WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


class GoldRef(BaseModel):
    """One gold reference (see the module docstring for the four shapes)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    section: str | None = None
    doc: str | None = None
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=1)
    quote: str | None = Field(default=None, min_length=8)
    any_of: list["GoldRef"] | None = None

    @model_validator(mode="after")
    def exactly_one_shape(self) -> "GoldRef":
        """Exactly one of: section, (doc + quote), any_of."""
        shapes = [self.section is not None, self.doc is not None, self.any_of is not None]
        if sum(shapes) != 1:
            raise ValueError("a gold reference needs exactly one of: section, doc (+ quote), any_of")
        if self.doc is not None and self.quote is None:
            raise ValueError("a passage reference needs a quote")
        if self.doc is None and (self.quote or self.page or self.paragraph):
            raise ValueError("quote, page and paragraph only go with doc")
        if self.any_of is not None and len(self.any_of) < 2:
            raise ValueError("any_of needs at least two alternatives")
        return self

    def describe(self) -> str:
        """Short text form, for reports."""
        if self.section:
            return self.section
        if self.any_of:
            return " | ".join(ref.describe() for ref in self.any_of)
        place = f"p{self.page}" if self.page else (f"para{self.paragraph}" if self.paragraph else "preamble")
        return f"{self.doc}@{place}"


class EvalQuestion(BaseModel):
    """One evaluation question, phrased as an underwriter would ask it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Z]{1,3}\d{2,3}$")
    question: str = Field(min_length=10)
    category: Category
    gold: list[GoldRef]
    note: str = ""

    @model_validator(mode="after")
    def gold_fits_category(self) -> "EvalQuestion":
        """Unanswerable questions have no gold; every other question has some; multi_section has at least two."""
        if self.category == "unanswerable" and self.gold:
            raise ValueError(f"{self.id}: unanswerable questions must have gold = []")
        if self.category != "unanswerable" and not self.gold:
            raise ValueError(f"{self.id}: answerable questions need at least one gold reference")
        if self.category == "multi_section" and len(self.gold) < 2:
            raise ValueError(f"{self.id}: multi_section questions need at least two gold references")
        return self

    @property
    def answerable(self) -> bool:
        """True when the question has gold references."""
        return bool(self.gold)


def load_questions(path: Path = QUESTIONS_FILE) -> list[EvalQuestion]:
    """Read the JSONL file; ids must be unique."""
    questions = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(EvalQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path.name} line {line_number}: {error}") from error
    ids = [question.id for question in questions]
    duplicates = sorted({question_id for question_id in ids if ids.count(question_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate question ids: {duplicates}")
    return questions


def normalize_for_match(text: str) -> str:
    """Lowercase, unify curly quotes and dashes, and collapse whitespace (for quote matching)."""
    table = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\xa0": " "})
    return " ".join(text.translate(table).lower().split())


def chunk_matches(payload: Mapping[str, Any], ref: GoldRef) -> bool:
    """True when a retrieved chunk (its payload) satisfies the gold reference."""
    if ref.any_of is not None:
        return any(chunk_matches(payload, alternative) for alternative in ref.any_of)
    if ref.section is not None:
        return payload.get("chunk_id") == ref.section
    if payload.get("doc_id") != ref.doc:
        return False
    if ref.page is not None and ref.page not in payload.get("pages", []):
        return False
    paragraphs = payload.get("paragraphs", [])
    if ref.paragraph is not None and ref.paragraph not in paragraphs:
        return False
    return normalize_for_match(ref.quote or "") in normalize_for_match(str(payload.get("text", "")))


def gold_first_ranks(question: EvalQuestion, retrieved: Sequence[Mapping[str, Any]]) -> list[int | None]:
    """For each gold reference, the 1-based rank of the first retrieved chunk that matches it (None if none)."""
    ranks: list[int | None] = []
    for ref in question.gold:
        rank = next((index for index, payload in enumerate(retrieved, start=1) if chunk_matches(payload, ref)), None)
        ranks.append(rank)
    return ranks


def leaf_refs(ref: GoldRef) -> list[GoldRef]:
    """The plain references inside a reference (itself, or every alternative of an any_of)."""
    return [leaf for alternative in ref.any_of for leaf in leaf_refs(alternative)] if ref.any_of else [ref]


def validate_gold(questions: Iterable[EvalQuestion], indexed: Sequence[Mapping[str, Any]]) -> list[str]:
    """Problems with the gold labels against the indexed chunks ([] when every reference resolves).

    Every plain reference, including each any_of alternative, must match at least one indexed chunk.
    """
    problems = []
    for question in questions:
        for ref in question.gold:
            for leaf in leaf_refs(ref):
                if not any(chunk_matches(payload, leaf) for payload in indexed):
                    problems.append(f"{question.id}: gold {leaf.describe()} matches no indexed chunk")
    return problems


def content_words(text: str) -> set[str]:
    """Lowercase word tokens of text without common stopwords."""
    return {word for word in WORD.findall(text.lower()) if word not in STOPWORDS}


def lexical_overlap_stats(
    questions: Sequence[EvalQuestion], indexed: Sequence[Mapping[str, Any]]
) -> dict[str, float | int | str]:
    """How many of a question's content words appear verbatim in its gold text, compared with other chunks.

    For each answerable question: gold_overlap = share of its content words found in the text (plus title) of
    the chunks matching its gold references; best_other_overlap = the highest such share over all non-gold
    chunks. If questions copied the guideline wording, gold_overlap would be close to 1 and far above
    best_other_overlap. No stemming is applied ("verify" and "verified" are different words).
    """
    gold_overlaps: list[float] = []
    other_overlaps: list[float] = []
    for question in questions:
        if not question.answerable:
            continue
        words = content_words(question.question)
        if not words:
            continue
        gold_payloads = [payload for payload in indexed if any(chunk_matches(payload, ref) for ref in question.gold)]
        gold_text = " ".join(f"{payload.get('title', '')} {payload.get('text', '')}" for payload in gold_payloads)
        gold_overlaps.append(len(words & content_words(gold_text)) / len(words))
        other_best = 0.0
        for payload in indexed:
            if payload in gold_payloads:
                continue
            other_words = content_words(f"{payload.get('title', '')} {payload.get('text', '')}")
            other_best = max(other_best, len(words & other_words) / len(words))
        other_overlaps.append(other_best)
    count = len(gold_overlaps)
    ties_or_worse = sum(1 for gold, other in zip(gold_overlaps, other_overlaps, strict=True) if other >= gold)
    return {
        "n_questions": count,
        "mean_share_of_question_words_in_gold_text": round(sum(gold_overlaps) / count, 4) if count else 0.0,
        "mean_best_share_in_any_non_gold_chunk": round(sum(other_overlaps) / count, 4) if count else 0.0,
        "share_of_questions_where_a_non_gold_chunk_overlaps_at_least_as_much": (
            round(ties_or_worse / count, 4) if count else 0.0
        ),
        "method": "content words (lowercase, stopwords removed, no stemming) shared with chunk title + text",
    }
