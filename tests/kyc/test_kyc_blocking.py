"""Tests for uwcopilot.kyc.blocking on a hand-made list of names."""

import numpy as np

from uwcopilot.kyc.blocking import (
    BlockingConfig,
    BlockingIndex,
    key_tokens_of,
    padded_trigrams,
    phonetic_codes,
    required_votes_for,
)

NAMES: list[tuple[int, tuple[str, ...]]] = [
    (0, ("mohammed", "ali", "hassan")),
    (0, ("muhammad", "ali", "hasan")),  # alias of the same entry
    (1, ("ivan", "petrovich", "sidorenko")),
    (2, ("aerocaribbean", "airlines")),
    (3, ("john", "smith")),
    (4, ("john", "doe")),
    (5, ("jane", "smith")),
    (6, ("smith", "trading", "house")),
]


def build(max_candidates: int = 100) -> BlockingIndex:
    """Index over NAMES with a configurable cap."""
    return BlockingIndex(NAMES, BlockingConfig(max_candidates=max_candidates))


def entries_found(index: BlockingIndex, tokens: tuple[str, ...]) -> set[int]:
    """Watchlist rows reached by blocking for a query."""
    result = index.candidates(tokens)
    return {int(row) for row in index.name_entry[result.name_ids]}


def test_key_helpers() -> None:
    assert padded_trigrams("ali") == {"#al", "ali", "li#"}
    assert phonetic_codes("mohammed") == phonetic_codes("muhammad") == ("MHMT",)
    assert key_tokens_of(("m", "ali", "ali", "hassan")) == ("ali", "hassan")  # initials are not keys
    assert required_votes_for(1) == 1 and required_votes_for(3) == 2 and required_votes_for(0) == 1


def test_index_structure() -> None:
    index = build()
    assert index.name_count == len(NAMES)
    assert index.tokens_of_name(2) == ("ivan", "petrovich", "sidorenko")
    assert list(index.names_of_entry(0)) == [0, 1]
    assert index.vocabulary_size == len({token for _, tokens in NAMES for token in tokens})


def test_transliteration_typo_and_reordering_still_block() -> None:
    index = build()
    assert 0 in entries_found(index, ("mukhammad", "aly", "hassan"))  # transliteration
    assert 1 in entries_found(index, ("sidorenko", "ivan"))  # reordered
    assert 2 in entries_found(index, ("aerocaribean",))  # typo: shares most 3-grams
    assert 1 in entries_found(index, ("ivan", "petrovitch", "sydorenko"))


def test_an_unrelated_name_finds_nothing() -> None:
    index = build()
    result = index.candidates(("zebulon", "quarterfield"))
    assert len(result.name_ids) == 0 and result.size_before_cap == 0


def test_only_initials_means_no_keys() -> None:
    result = build().candidates(("j", "s"))
    assert result.key_tokens == () and len(result.name_ids) == 0


def test_half_of_the_tokens_must_vote() -> None:
    index = build()
    # three key tokens -> two votes needed; "smith trading house" shares only "smith"
    result = index.candidates(("john", "smith", "zzzz"))
    found = {int(index.name_entry[name_id]) for name_id in result.name_ids}
    assert result.required_votes == 2
    assert 3 in found  # john smith: two votes
    assert 6 not in found  # one vote only, and its first token is not "john"


def test_first_token_block_adds_names_starting_with_the_same_token() -> None:
    index = build()
    result = index.candidates(("john", "zzzz", "qqqq"))
    found = {int(index.name_entry[name_id]) for name_id in result.name_ids}
    assert found == {3, 4}  # only one vote each, but both start with "john"


def test_cap_keeps_the_names_with_most_votes_and_reports_size_before_cap() -> None:
    index = build(max_candidates=1)
    result = index.candidates(("john", "smith"))
    assert result.capped
    assert result.size_before_cap > 1
    assert len(result.name_ids) == 1
    assert index.tokens_of_name(int(result.name_ids[0])) == ("john", "smith")
    uncapped = index.candidates(("john", "smith"), max_candidates=10**9)
    assert not uncapped.capped and len(uncapped.name_ids) == result.size_before_cap
    assert np.all(uncapped.votes >= 1)
