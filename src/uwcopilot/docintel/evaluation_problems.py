"""Evaluation of contradictions and embedded-instruction hits against the injected problems of the ground truth.

Contradictions (match_contradictions):
- Only problem types that a contradiction can have are scored (contradictions.CONTRADICTION_TYPES: every problem
  type except missing_document and missing_pages, which the checklist evaluation covers).
- A detected contradiction MATCHES an injected problem when the type is the same and they overlap: they share a
  truth document (the documents of the pages the contradiction's evidence cites) or a field (equal paths, or one
  path inside the other, such as "profit_and_loss" and "profit_and_loss.revenue").
- Recall: an injected problem counts as detected when at least one contradiction matches it. Precision: a
  contradiction counts as correct when it matches at least one injected problem. Several contradictions about one
  problem therefore all count as correct; that is reported, not hidden (contradictions.all.total vs problems).
- Benign vs unsupported: for each detected problem whose true classification is benign or unsupported, the
  explanation of its FIRST matching contradiction (report order) is compared with the classification. "unknown"
  counts as wrong and is counted separately.

Embedded instructions (injection_counts): a planted instruction is detected when the injection scan reports a hit
on the same page. A hit on a page without a planted instruction is a false hit.

The truth is read by attribute (a truth.PacketTruth works); this module does not import truth.py.
"""

from collections import Counter
from typing import Any

from uwcopilot.docintel.contradictions import CONTRADICTION_TYPES, Contradiction
from uwcopilot.docintel.pipeline import PacketResult

EXPLAINED_CLASSES: tuple[str, ...] = ("benign", "unsupported")


def contradiction_docs(contradiction: Contradiction, page_docs: dict[int, str]) -> set[str]:
    """The truth documents of the pages the contradiction's evidence cites (its own doc ids when nothing is cited)."""
    docs = {page_docs[item.page] for item in contradiction.sides if item.page in page_docs}
    return docs or set(contradiction.doc_ids)


def contradiction_fields(contradiction: Contradiction) -> set[str]:
    """The field paths a contradiction names, plus the fields of its evidence."""
    return set(contradiction.fields) | {item.field for item in contradiction.sides}


def fields_overlap(first: set[str], second: set[str]) -> bool:
    """True when a path is in both sets, or a path of one set lies inside a path of the other."""
    for one in first:
        for other in second:
            if one == other or one.startswith(other + ".") or other.startswith(one + "."):
                return True
    return False


def matches(problem: Any, contradiction: Contradiction, page_docs: dict[int, str]) -> bool:
    """Same type, and a shared truth document or field (see the module docstring)."""
    if contradiction.type != problem.problem_type:
        return False
    if contradiction_docs(contradiction, page_docs) & set(problem.doc_ids):
        return True
    return fields_overlap(contradiction_fields(contradiction), set(problem.fields))


def match_contradictions(
    problems: list[Any], contradictions: list[Contradiction], page_docs: dict[int, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One outcome per scorable injected problem and one per detected contradiction (JSON-simple dicts)."""
    scorable = [problem for problem in problems if problem.problem_type in CONTRADICTION_TYPES]
    problem_outcomes = []
    for problem in scorable:
        found = [item for item in contradictions if matches(problem, item, page_docs)]
        first = found[0] if found else None
        problem_outcomes.append(
            {
                "problem_id": problem.problem_id,
                "type": problem.problem_type,
                "classification": problem.classification,
                "detected": first is not None,
                "contradiction_id": None if first is None else first.contradiction_id,
                "explanation": None if first is None else first.explanation,
                "explanation_code": None if first is None else first.explanation_code,
            }
        )
    contradiction_outcomes = []
    for contradiction in contradictions:
        matched = [problem.problem_id for problem in scorable if matches(problem, contradiction, page_docs)]
        contradiction_outcomes.append(
            {
                "contradiction_id": contradiction.contradiction_id,
                "type": contradiction.type,
                "explanation": contradiction.explanation,
                "matched_problem_ids": matched,
            }
        )
    return problem_outcomes, contradiction_outcomes


def contradiction_counts(problem_outcomes: list[dict], contradiction_outcomes: list[dict]) -> Counter:
    """Counters for recall (problems.*), precision (contradictions.*) and explanations (explanations.*)."""
    counts: Counter = Counter()
    for outcome in problem_outcomes:
        for kind in ("all", outcome["type"]):
            counts[f"problems.{kind}.total"] += 1
            counts[f"problems.{kind}.detected"] += int(outcome["detected"])
        classification = outcome["classification"]
        if outcome["detected"] and classification in EXPLAINED_CLASSES:
            for group in ("all", classification):
                counts[f"explanations.{group}.total"] += 1
                counts[f"explanations.{group}.correct"] += int(outcome["explanation"] == classification)
                counts[f"explanations.{group}.unknown"] += int(outcome["explanation"] == "unknown")
    for outcome in contradiction_outcomes:
        for kind in ("all", outcome["type"]):
            counts[f"contradictions.{kind}.total"] += 1
            counts[f"contradictions.{kind}.tp"] += int(bool(outcome["matched_problem_ids"]))
    return counts


def hit_pages(result: PacketResult) -> list[tuple[int | None, Any]]:
    """(page, hit) for every injection hit; a hit without a page takes its document's first page."""
    document_pages = {document.doc_id: document.pages for document in result.extraction.documents}
    placed = []
    for hit in result.injection_hits:
        page = hit.page
        if page is None and document_pages.get(hit.doc_id):
            page = document_pages[hit.doc_id][0]
        placed.append((page, hit))
    return placed


def injection_counts(truth: Any, result: PacketResult) -> tuple[Counter, list[dict[str, Any]]]:
    """Counters for injection recall and false hits, and one outcome per planted instruction."""
    counts: Counter = Counter()
    placed = hit_pages(result)
    hit_page_numbers = {page for page, _ in placed if page is not None}
    planted_pages = {injection.page_number for injection in truth.injections}
    outcomes = []
    for injection in truth.injections:
        detected = injection.page_number in hit_page_numbers
        counts["injections.total"] += 1
        counts["injections.detected"] += int(detected)
        outcomes.append({"page": injection.page_number, "technique": injection.technique, "detected": detected})
    false_hits = [hit for page, hit in placed if page not in planted_pages]
    counts["injection_hits.total"] += len(placed)
    counts["injection_hits.false"] += len(false_hits)
    counts["injection_packets.total"] += 1
    counts["injection_packets.with_false_hit"] += int(bool(false_hits))
    if not truth.injections:
        counts["injection_packets.clean"] += 1
        counts["injection_packets.clean_with_hit"] += int(bool(placed))
    return counts, outcomes
