"""Rules-based name scoring and the explanation an analyst sees for every alert.

The rules score of a customer name against one list name is

    name_score  = W_JW * jaro_winkler_aligned + W_SET * token_set
    rules_score = (name_score + attribute_adjustment + 0.07) / 1.12      (rescaled to 0..1, see rules_score())

- jaro_winkler_aligned: Jaro-Winkler similarity applied token by token (the Monge-Elkan method). Each token of
  one name is paired with its most similar token in the other name, the best similarities are averaged, and
  the average is taken in both directions (customer -> list and list -> customer) and then averaged again.
  Pairing tokens makes the score insensitive to token order ("Hassan Ali" vs "Ali Hassan"), and the
  two-direction average stops a one-word name from matching every longer name that contains that word.
  A one-letter initial matches a token with the same first letter at INITIAL_SIMILARITY (0.9), not 1.0.
- token_set: rapidfuzz token_set_ratio / 100. It compares the shared tokens and the leftovers as sets, so
  extra middle names cost little.
- attribute_adjustment: small, and only when both sides are known: entity type agreement +0.02, disagreement
  -0.05 (a person is not a vessel); country agreement +0.03, disagreement -0.02 (people move and hold more than
  one nationality, so a country mismatch is weak evidence). Names dominate; attributes only nudge.

Weights W_JW = 0.6 and W_SET = 0.4 are fixed design choices, not tuned on the test data: Jaro-Winkler is the
main typo/transliteration signal; token_set adds robustness to extra or missing tokens.
"""

from dataclasses import dataclass, field

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from uwcopilot.kyc.blocking import phonetic_codes
from uwcopilot.kyc.normalize import is_initial

# A token pair at or above this Jaro-Winkler similarity is described as "similar spelling" in explanations.
SIMILAR_SPELLING_MIN = 0.85


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the rules score (see module docstring for the reasoning)."""

    jaro_winkler: float = 0.6
    token_set: float = 0.4
    initial_similarity: float = 0.9
    type_agree_bonus: float = 0.02
    type_disagree_penalty: float = 0.05
    country_agree_bonus: float = 0.03
    country_disagree_penalty: float = 0.02


@dataclass(frozen=True)
class NameScores:
    """The two name-similarity components and their weighted combination."""

    jaro_winkler_aligned: float
    token_set: float
    name_score: float


@dataclass(frozen=True)
class AttributeCheck:
    """How entity type and country compare, the resulting score change, and a sentence for each."""

    adjustment: float
    notes: tuple[str, ...] = field(default_factory=tuple)


def token_similarity(left: str, right: str, initial_similarity: float) -> float:
    """Jaro-Winkler similarity of two tokens; an initial matches any token that starts with the same letter."""
    if is_initial(left) or is_initial(right):
        return initial_similarity if left[0] == right[0] else 0.0
    return JaroWinkler.normalized_similarity(left, right)


def best_alignment(source: tuple[str, ...], target: tuple[str, ...], initial_similarity: float) -> list[tuple]:
    """For each source token: (source token, most similar target token, similarity)."""
    pairs = []
    for token in source:
        best_token, best_score = "", 0.0
        for other in target:
            score = token_similarity(token, other, initial_similarity)
            if score > best_score:
                best_token, best_score = other, score
        pairs.append((token, best_token, best_score))
    return pairs


def monge_elkan(source: tuple[str, ...], target: tuple[str, ...], initial_similarity: float) -> float:
    """Average, over source tokens, of the best token similarity found in target (one direction)."""
    if not source or not target:
        return 0.0
    pairs = best_alignment(source, target, initial_similarity)
    return sum(score for _, _, score in pairs) / len(pairs)


def name_scores(query: tuple[str, ...], candidate: tuple[str, ...], weights: ScoringWeights) -> NameScores:
    """Compute the token-aligned Jaro-Winkler, the token-set ratio, and their weighted sum."""
    forward = monge_elkan(query, candidate, weights.initial_similarity)
    backward = monge_elkan(candidate, query, weights.initial_similarity)
    aligned = (forward + backward) / 2
    token_set = fuzz.token_set_ratio(" ".join(query), " ".join(candidate)) / 100.0
    combined = weights.jaro_winkler * aligned + weights.token_set * token_set
    return NameScores(jaro_winkler_aligned=aligned, token_set=token_set, name_score=combined)


def attribute_check(
    query_type: str | None,
    query_country: str | None,
    entry_type: str | None,
    entry_countries: tuple[str, ...],
    weights: ScoringWeights,
) -> AttributeCheck:
    """Compare entity type and country when both sides know them. Unknown values change nothing."""
    adjustment = 0.0
    notes: list[str] = []
    if query_type and entry_type and entry_type != "other":
        if query_type == entry_type:
            adjustment += weights.type_agree_bonus
            notes.append(f"entity type agrees: both {entry_type} (+{weights.type_agree_bonus:.2f})")
        else:
            adjustment -= weights.type_disagree_penalty
            notes.append(
                f"entity type differs: customer {query_type}, list {entry_type} (-{weights.type_disagree_penalty:.2f})"
            )
    else:
        notes.append("entity type not compared: unknown on the list entry or the customer")
    if query_country and entry_countries:
        if query_country in entry_countries:
            adjustment += weights.country_agree_bonus
            notes.append(f"country agrees: {query_country} (+{weights.country_agree_bonus:.2f})")
        else:
            listed = ", ".join(entry_countries)
            adjustment -= weights.country_disagree_penalty
            notes.append(
                f"country differs: customer {query_country}, list {listed} (-{weights.country_disagree_penalty:.2f})"
            )
    else:
        notes.append("country not compared: unknown on the list entry or the customer")
    return AttributeCheck(adjustment=adjustment, notes=tuple(notes))


def rules_score(name: NameScores, attributes: AttributeCheck, weights: ScoringWeights) -> float:
    """Name score plus attribute adjustment, rescaled so the result always lies in [0, 1].

    The raw sum ranges from -0.07 (name score 0, both attributes disagree) to 1.05 (identical names, both
    attributes agree). Mapping that range linearly onto [0, 1] keeps the order of every score, unlike clipping
    at 1.0, which would tie many different near-identical names at exactly 1.0.
    """
    lowest = -(weights.type_disagree_penalty + weights.country_disagree_penalty)
    highest = weights.type_agree_bonus + weights.country_agree_bonus
    raw = name.name_score + attributes.adjustment
    return (raw - lowest) / (1.0 + highest - lowest)


def pair_kind(query_token: str, list_token: str, similarity: float) -> str | None:
    """How a customer token relates to its best list token: exact, initial, phonetic, similar, or None."""
    if not list_token or similarity <= 0.0:
        return None
    if query_token == list_token:
        return "exact"
    if is_initial(query_token) or is_initial(list_token):
        return "initial"
    if set(phonetic_codes(query_token)) & set(phonetic_codes(list_token)):
        return "phonetic"
    if similarity >= SIMILAR_SPELLING_MIN:
        return "similar"
    return None


def describe_token_pair(query_token: str, list_token: str, similarity: float) -> str:
    """One phrase saying how a customer token relates to its best list token."""
    kind = pair_kind(query_token, list_token, similarity)
    if kind == "exact":
        return f"'{query_token}' = '{list_token}' (exact)"
    if kind == "initial":
        return f"'{query_token}' ~ '{list_token}' (initial)"
    if kind == "phonetic":
        code = sorted(set(phonetic_codes(query_token)) & set(phonetic_codes(list_token)))[0]
        return f"'{query_token}' ~ '{list_token}' (same Double Metaphone code {code}, Jaro-Winkler {similarity:.2f})"
    if kind == "similar":
        return f"'{query_token}' ~ '{list_token}' (similar spelling, Jaro-Winkler {similarity:.2f})"
    if not list_token or similarity <= 0.0:
        return f"'{query_token}' has no counterpart"
    return f"'{query_token}' has no close counterpart (best '{list_token}', Jaro-Winkler {similarity:.2f})"


def token_explanations(query: tuple[str, ...], candidate: tuple[str, ...], weights: ScoringWeights) -> list[str]:
    """Sentences on matched tokens, phonetic agreement, and list-name tokens nothing matched."""
    pairs = best_alignment(query, candidate, weights.initial_similarity)
    kinds = [pair_kind(q, c, s) for q, c, s in pairs]
    lines = ["matched tokens: " + "; ".join(describe_token_pair(q, c, s) for q, c, s in pairs)]
    phonetic_agreements = sum(
        1 for (q, c, _), kind in zip(pairs, kinds, strict=True) if kind in ("exact", "phonetic") and phonetic_codes(q)
    )
    key_tokens = sum(1 for q in query if not is_initial(q))
    lines.append(
        f"phonetic agreement: {phonetic_agreements} of {key_tokens} customer tokens share a Double Metaphone code"
    )
    used = {c for (_, c, _), kind in zip(pairs, kinds, strict=True) if kind is not None}
    leftover = [token for token in candidate if token not in used]
    if leftover:
        lines.append("list-name tokens not matched by the customer name: " + ", ".join(leftover))
    return lines
