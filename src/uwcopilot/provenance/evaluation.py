"""Scoring for the numeric-provenance evaluation (scripts/eval_provenance.py). Pure functions, no LLM.

Three measurements:
1. Mutation test of the free-form checker (numeric_check.check_numbers): for each mutation type, the share of
   mutated sentences it flags (recall); for the controls, the share it flags wrongly (false-flag rate).
2. The same mutations scored by the INCUMBENT checker that the agent already uses (agent.claim_checker): a number is
   "supported" when some number in the cited section rounds to it at the written precision. It has no units and no
   comparators, so it is the baseline the new checker must beat on unit and comparator errors.
3. The memo grammar validator (memo_grammar.validate) on a frozen hand-written set: recall per expected violation
   code, and the false-reject rate on clean sentences.

Every rate carries a Wilson 95% interval (common.stats.rate_summary).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from uwcopilot.agent.claim_checker import is_supported, numbers_in
from uwcopilot.common.stats import rate_summary
from uwcopilot.provenance.memo_grammar import validate
from uwcopilot.provenance.mutations import CONTROL_TYPES, MUTATION_TYPES, Mutation
from uwcopilot.provenance.numeric_check import check_numbers

Flagger = Callable[[Mutation, dict[str, str]], bool]


def sig_checker_flags(mutation: Mutation, sections: dict[str, str]) -> bool:
    """True when the SIG checker flags the mutated sentence (sources = its cited section only)."""
    return check_numbers(mutation.mutated, {mutation.section_id: sections[mutation.section_id]}).flagged


def sig_checker_unverifiable(mutation: Mutation, sections: dict[str, str]) -> bool:
    """True when any number in the sentence is comparator_unverifiable (the documented known limit)."""
    result = check_numbers(mutation.mutated, {mutation.section_id: sections[mutation.section_id]})
    return bool(result.comparator_unverifiable)


def incumbent_flags(mutation: Mutation, sections: dict[str, str]) -> bool:
    """True when the incumbent claim checker finds a number in the sentence that the cited section lacks."""
    source_numbers = [abs(float(raw.replace(",", ""))) for raw in numbers_in(sections[mutation.section_id])]
    return any(not is_supported(raw, source_numbers) for raw in numbers_in(mutation.mutated))


@dataclass(frozen=True)
class TypeScore:
    """Flagged count for one mutation type or control."""

    mutation_type: str
    n: int
    flagged: int


def score_by_type(mutations: Iterable[Mutation], sections: dict[str, str], flagger: Flagger) -> dict[str, TypeScore]:
    """Flagged counts per mutation type and control type."""
    counts: dict[str, list[int]] = {}
    for mutation in mutations:
        pair = counts.setdefault(mutation.mutation_type, [0, 0])
        pair[0] += 1
        pair[1] += int(flagger(mutation, sections))
    return {kind: TypeScore(kind, n, flagged) for kind, (n, flagged) in counts.items()}


def mutation_report(scores: dict[str, TypeScore]) -> dict:
    """recall_by_type for the mutation types and false_flag_rate for the controls, with Wilson intervals."""
    recall = {}
    for kind in MUTATION_TYPES:
        score = scores.get(kind, TypeScore(kind, 0, 0))
        summary = rate_summary(score.flagged, score.n)
        recall[kind] = {
            "n": score.n,
            "detected": score.flagged,
            "rate": summary["rate"],
            "wilson_ci95": summary["wilson_ci95"],
        }
    false_flags = {}
    for kind in CONTROL_TYPES:
        score = scores.get(kind, TypeScore(kind, 0, 0))
        summary = rate_summary(score.flagged, score.n)
        false_flags[kind] = {
            "n": score.n,
            "flagged": score.flagged,
            "rate": summary["rate"],
            "wilson_ci95": summary["wilson_ci95"],
        }
    return {"recall_by_type": recall, "false_flag_rate": false_flags}


@dataclass(frozen=True)
class AdversarialRow:
    """One row of the frozen validator test set."""

    kind: str  # "reject" or "clean"
    text: str
    expected_code: str | None
    required_refs: list[str]


def validator_report(rows: list[AdversarialRow], context: dict) -> dict:
    """Recall per expected code on reject rows and the false-reject rate on clean rows."""
    claim_ids = set(context["claim_ids"])
    assertion_ids = set(context["assertion_ids"])
    allowed = set(context["allowed_sections"])
    per_code: dict[str, list[int]] = {}
    clean_total = clean_rejected = 0
    missed: list[dict] = []
    false_rejects: list[dict] = []
    for row in rows:
        violations = validate(row.text, claim_ids, assertion_ids, row.required_refs, allowed)
        codes = {violation.code for violation in violations}
        if row.kind == "reject":
            pair = per_code.setdefault(row.expected_code or "", [0, 0])
            pair[0] += 1
            hit = row.expected_code in codes
            pair[1] += int(hit)
            if not hit:
                missed.append({"text": row.text, "expected_code": row.expected_code, "got": sorted(codes)})
        else:
            clean_total += 1
            if violations:
                clean_rejected += 1
                false_rejects.append({"text": row.text, "got": sorted(codes)})
    recall = {code: rate_summary(hit, n) for code, (n, hit) in sorted(per_code.items())}
    total_n = sum(n for n, _ in per_code.values())
    total_hit = sum(hit for _, hit in per_code.values())
    return {
        "n_should_reject": total_n,
        "recall_overall": rate_summary(total_hit, total_n),
        "recall": recall,
        "n_clean": clean_total,
        "false_reject_rate": rate_summary(clean_rejected, clean_total),
        "missed": missed,
        "false_rejects": false_rejects,
    }
