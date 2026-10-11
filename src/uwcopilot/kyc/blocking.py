"""Blocking: find the few thousand watchlist names worth scoring, out of millions.

Comparing every customer with every one of ~2.6 million list names is too slow, so screening first builds an
index and only scores names that share a "blocking key" with the customer. The index works at token level:

- every distinct token in the list (the vocabulary) gets three kinds of keys:
  1. the token itself (exact match),
  2. its Double Metaphone codes (primary and alternate), so "mohammed", "muhammad" and "mohamad" share "MHMT",
  3. its character 3-grams with "#" padding ("#ha", "has", "ass", "ssa", "san", "an#"), so a typo still
     shares most 3-grams with the correct spelling;
- each vocabulary token has a posting list: the list names that contain it;
- the first token of every list name has its own posting list (the classic "first token" block).

For a customer name, each token (two or more characters; initials such as "J" are not used as keys) is matched
to similar vocabulary tokens: the exact token, tokens with the same Double Metaphone code, and tokens whose
3-gram Dice coefficient is at least `trigram_dice_threshold`. Phonetic and 3-gram matches must also have
a similar length. Tokens of only two characters ("al", "de") match exactly or not at all, because a two-letter
key would pull in a large share of the list.

A list name becomes a candidate when at least half of the customer's key tokens (rounded up) have a similar
token in it ("votes"), or when its first token equals the customer's first token. If more than
`max_candidates` names qualify, the ones with the most votes are kept (then first-token agreement, then the
closest token count). The candidate-set size before and after this cap is returned with every query, so the
evaluation can report it, together with blocking recall (how often the true list entry survives blocking).
"""

import math
from array import array
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from metaphone import doublemetaphone

from uwcopilot.kyc.normalize import is_initial

PAD = "#"


@dataclass(frozen=True)
class BlockingConfig:
    """Settings for candidate generation. Defaults are the values used in results/kyc_metrics.json."""

    trigram_dice_threshold: float = 0.5
    # Tokens shorter than this match only exactly (no phonetic or 3-gram expansion).
    min_token_length_for_fuzzy_keys: int = 3
    # A phonetic or 3-gram match may differ in length by at most max(this, 25% of the query token's length).
    max_length_difference: int = 2
    max_candidates: int = 3000


@dataclass
class BlockingResult:
    """Candidates for one query: list-name ids, their vote counts, and the candidate-set sizes."""

    name_ids: np.ndarray
    votes: np.ndarray
    key_tokens: tuple[str, ...]
    required_votes: int
    size_before_cap: int
    capped: bool


def padded_trigrams(token: str) -> set[str]:
    """Character 3-grams of "#token#", e.g. "ali" -> {"#al", "ali", "li#"}."""
    padded = f"{PAD}{token}{PAD}"
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def phonetic_codes(token: str) -> tuple[str, ...]:
    """Double Metaphone codes (primary, and alternate when different) of a token; empty if none."""
    primary, alternate = doublemetaphone(token)
    codes = [code for code in (primary, alternate) if code]
    return tuple(dict.fromkeys(codes))


def key_tokens_of(tokens: Iterable[str]) -> tuple[str, ...]:
    """Distinct tokens that act as blocking keys: everything except one-character initials."""
    return tuple(dict.fromkeys(token for token in tokens if not is_initial(token)))


def required_votes_for(key_token_count: int) -> int:
    """At least half of the key tokens, rounded up, and at least one."""
    return max(1, math.ceil(key_token_count / 2))


def compressed_postings(keys: np.ndarray, values: np.ndarray, key_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Group `values` by `keys` into CSR form: postings[offsets[k]:offsets[k + 1]] are the values of key k."""
    order = np.argsort(keys, kind="stable")
    postings = values[order].astype(np.int32)
    counts = np.bincount(keys, minlength=key_count)
    offsets = np.zeros(key_count + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return offsets, postings


class BlockingIndex:
    """Token-level inverted index over every normalized list name.

    Name ids are positions 0..n_names-1 in the order the names were added. name_entry[i] is the watchlist row
    the name belongs to; name_texts[i] is the normalized name; name tokens are stored in CSR form.
    """

    def __init__(self, normalized_names: Iterable[tuple[int, tuple[str, ...]]], config: BlockingConfig | None = None):
        """Build the index from (watchlist row, normalized tokens) pairs. Names with no tokens are skipped."""
        self.config = config or BlockingConfig()
        self.vocabulary: dict[str, int] = {}
        self.tokens_by_id: list[str] = []
        name_entry = array("i")
        name_lengths = array("i")
        flat_token_ids = array("i")
        self.name_texts: list[str] = []
        for entry_row, tokens in normalized_names:
            if not tokens:
                continue
            name_entry.append(entry_row)
            name_lengths.append(len(tokens))
            self.name_texts.append(" ".join(tokens))
            for token in tokens:
                token_id = self.vocabulary.get(token)
                if token_id is None:
                    token_id = len(self.tokens_by_id)
                    self.vocabulary[token] = token_id
                    self.tokens_by_id.append(token)
                flat_token_ids.append(token_id)

        self.name_entry = np.frombuffer(name_entry, dtype=np.int32).copy()
        self.name_token_counts = np.frombuffer(name_lengths, dtype=np.int32).copy()
        self.name_token_offsets = np.zeros(len(self.name_entry) + 1, dtype=np.int64)
        np.cumsum(self.name_token_counts, out=self.name_token_offsets[1:])
        self.name_token_ids = np.frombuffer(flat_token_ids, dtype=np.int32).copy()
        vocabulary_size = len(self.tokens_by_id)

        # token -> names containing it, and first token -> names starting with it
        name_of_token = np.repeat(np.arange(self.name_count, dtype=np.int32), self.name_token_counts)
        self.token_offsets, self.token_postings = compressed_postings(
            self.name_token_ids, name_of_token, vocabulary_size
        )
        first_token_ids = self.name_token_ids[self.name_token_offsets[:-1]]
        self.first_token_offsets, self.first_token_postings = compressed_postings(
            first_token_ids, np.arange(self.name_count, dtype=np.int32), vocabulary_size
        )
        self.token_lengths = np.array([len(token) for token in self.tokens_by_id], dtype=np.int32)
        self.phonetic_index = self.build_phonetic_index()
        self.build_trigram_index()

    @property
    def name_count(self) -> int:
        """Number of indexed list names (primary names plus aliases, after de-duplication)."""
        return int(len(self.name_entry))

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct normalized tokens in the list."""
        return len(self.tokens_by_id)

    def build_phonetic_index(self) -> dict[str, np.ndarray]:
        """Double Metaphone code -> ids of vocabulary tokens with that code."""
        min_length = self.config.min_token_length_for_fuzzy_keys
        by_code: dict[str, list[int]] = {}
        for token_id, token in enumerate(self.tokens_by_id):
            if len(token) < min_length:
                continue
            for code in phonetic_codes(token):
                by_code.setdefault(code, []).append(token_id)
        return {code: np.array(ids, dtype=np.int32) for code, ids in by_code.items()}

    def build_trigram_index(self) -> None:
        """3-gram -> ids of vocabulary tokens containing it (CSR), plus each token's 3-gram count."""
        trigram_ids: dict[str, int] = {}
        pair_trigrams = array("i")
        pair_tokens = array("i")
        trigram_counts = np.zeros(self.vocabulary_size, dtype=np.int32)
        for token_id, token in enumerate(self.tokens_by_id):
            trigrams = padded_trigrams(token)
            trigram_counts[token_id] = len(trigrams)
            for trigram in trigrams:
                trigram_id = trigram_ids.setdefault(trigram, len(trigram_ids))
                pair_trigrams.append(trigram_id)
                pair_tokens.append(token_id)
        self.trigram_ids = trigram_ids
        self.token_trigram_counts = trigram_counts
        self.trigram_offsets, self.trigram_postings = compressed_postings(
            np.frombuffer(pair_trigrams, dtype=np.int32),
            np.frombuffer(pair_tokens, dtype=np.int32),
            len(trigram_ids),
        )

    def tokens_of_name(self, name_id: int) -> tuple[str, ...]:
        """The normalized tokens of one indexed name."""
        start, end = self.name_token_offsets[name_id], self.name_token_offsets[name_id + 1]
        return tuple(self.tokens_by_id[token_id] for token_id in self.name_token_ids[start:end])

    def names_of_entry(self, entry_row: int) -> np.ndarray:
        """Ids of every indexed name (primary and aliases) that belongs to one watchlist row."""
        return np.flatnonzero(self.name_entry == entry_row)

    def length_ok(self, query_token: str, token_ids: np.ndarray) -> np.ndarray:
        """Boolean mask: which vocabulary tokens are close enough in length to the query token."""
        allowed = max(self.config.max_length_difference, round(0.25 * len(query_token)))
        return np.abs(self.token_lengths[token_ids] - len(query_token)) <= allowed

    def similar_token_ids(self, query_token: str) -> np.ndarray:
        """Vocabulary tokens similar to one query token: exact, same Double Metaphone code, or 3-gram Dice."""
        found: list[np.ndarray] = []
        exact_id = self.vocabulary.get(query_token)
        if exact_id is not None:
            found.append(np.array([exact_id], dtype=np.int32))
        if len(query_token) >= self.config.min_token_length_for_fuzzy_keys:
            for code in phonetic_codes(query_token):
                ids = self.phonetic_index.get(code)
                if ids is not None:
                    found.append(ids[self.length_ok(query_token, ids)])
            found.append(self.trigram_matches(query_token))
        if not found:
            return np.empty(0, dtype=np.int32)
        return np.unique(np.concatenate(found))

    def trigram_matches(self, query_token: str) -> np.ndarray:
        """Vocabulary tokens whose padded 3-gram Dice coefficient with the query token passes the threshold."""
        query_trigrams = padded_trigrams(query_token)
        slices = []
        for trigram in query_trigrams:
            trigram_id = self.trigram_ids.get(trigram)
            if trigram_id is not None:
                slices.append(
                    self.trigram_postings[self.trigram_offsets[trigram_id] : self.trigram_offsets[trigram_id + 1]]
                )
        if not slices:
            return np.empty(0, dtype=np.int32)
        token_ids, shared = np.unique(np.concatenate(slices), return_counts=True)
        dice = 2.0 * shared / (len(query_trigrams) + self.token_trigram_counts[token_ids])
        keep = (dice >= self.config.trigram_dice_threshold) & self.length_ok(query_token, token_ids)
        return token_ids[keep]

    def names_containing(self, token_ids: np.ndarray) -> np.ndarray:
        """Concatenated posting lists (names containing any of these tokens); may contain repeats."""
        if len(token_ids) == 0:
            return np.empty(0, dtype=np.int32)
        slices = [self.token_postings[self.token_offsets[t] : self.token_offsets[t + 1]] for t in token_ids]
        return np.concatenate(slices)

    def names_starting_with(self, token: str) -> np.ndarray:
        """Names whose first normalized token is exactly `token`."""
        token_id = self.vocabulary.get(token)
        if token_id is None:
            return np.empty(0, dtype=np.int32)
        return self.first_token_postings[self.first_token_offsets[token_id] : self.first_token_offsets[token_id + 1]]

    def candidates(self, query_tokens: tuple[str, ...], max_candidates: int | None = None) -> BlockingResult:
        """Run blocking for one normalized query name.

        max_candidates overrides the configured cap; the evaluation uses a very large value to measure blocking
        recall before the cap.
        """
        cap = self.config.max_candidates if max_candidates is None else max_candidates
        key_tokens = key_tokens_of(query_tokens)
        required = required_votes_for(len(key_tokens))
        empty = np.empty(0, dtype=np.int32)
        if not key_tokens:
            return BlockingResult(empty, empty, key_tokens, required, 0, False)

        votes = np.zeros(self.name_count, dtype=np.int16)
        seen = np.zeros(self.name_count, dtype=bool)
        for token in key_tokens:
            names = self.names_containing(self.similar_token_ids(token))
            seen[:] = False
            seen[names] = True  # a token votes at most once per name
            votes += seen
        first_token_block = np.zeros(self.name_count, dtype=bool)
        if len(query_tokens[0]) >= 2:
            first_token_block[self.names_starting_with(query_tokens[0])] = True

        name_ids = np.flatnonzero((votes >= required) | first_token_block).astype(np.int32)
        size_before_cap = int(len(name_ids))
        capped = size_before_cap > cap
        if capped:
            token_gap = np.abs(self.name_token_counts[name_ids] - len(query_tokens))
            # np.lexsort sorts by the LAST key first: most votes, then first-token agreement, then token gap.
            order = np.lexsort((token_gap, ~first_token_block[name_ids], -votes[name_ids].astype(np.int32)))
            name_ids = name_ids[order[:cap]]
        return BlockingResult(
            name_ids=name_ids,
            votes=votes[name_ids].astype(np.int32),
            key_tokens=key_tokens,
            required_votes=required,
            size_before_cap=size_before_cap,
            capped=capped,
        )
