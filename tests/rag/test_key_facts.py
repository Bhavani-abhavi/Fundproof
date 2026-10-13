"""The frozen key-fact set (eval/rag_key_facts.jsonl) is checkable: every fact is stated in its gold section.

The key facts were written before any model answered these questions. This test proves that each fact appears in
the gold section's text (numbers through the SIG number normalizer, roles and terms as normalized substrings), so
key-fact recall never rewards or punishes an answer for a fact the guidelines do not contain.
"""

from collections import Counter

import pytest

from uwcopilot.mcp_server.guideline_store import GuidelineStore
from uwcopilot.provenance.numbers import extract_numbers
from uwcopilot.rag.eval_set import leaf_refs, load_questions
from uwcopilot.rag.frozen_hits import KEY_FACTS_FILE
from uwcopilot.rag.generation_eval import KeyFact, fact_in_text, load_key_facts

KEY_FACTS = load_key_facts(KEY_FACTS_FILE)
QUESTIONS = {question.id: question for question in load_questions()}
STORE = GuidelineStore()


def test_key_fact_set_size_and_kinds() -> None:
    assert 40 <= len(KEY_FACTS) <= 60
    kinds = Counter(fact.kind for row in KEY_FACTS.values() for fact in row.key_facts)
    assert set(kinds) <= {"number", "role", "term"}
    assert kinds["number"] > 0 and kinds["role"] > 0 and kinds["term"] > 0


def test_every_keyed_question_is_an_answerable_guideline_question_with_that_gold_section() -> None:
    for question_id, row in KEY_FACTS.items():
        question = QUESTIONS[question_id]
        assert question.category in ("single_section", "multi_section"), question_id
        gold_sections = {leaf.section for ref in question.gold for leaf in leaf_refs(ref) if leaf.section}
        assert row.gold_section in gold_sections, question_id


@pytest.mark.parametrize("question_id", sorted(KEY_FACTS))
def test_every_key_fact_appears_in_its_gold_section(question_id: str) -> None:
    row = KEY_FACTS[question_id]
    section_text = STORE.get_section(row.gold_section).text
    for fact in row.key_facts:
        assert fact_in_text(fact, section_text), f"{question_id}: {fact.text!r} not in {row.gold_section}"


def test_number_facts_contain_a_quantity_and_term_facts_are_not_bare_numbers() -> None:
    for row in KEY_FACTS.values():
        for fact in row.key_facts:
            if fact.kind == "number":
                assert extract_numbers(fact.text), fact.text
            else:
                assert any(character.isalpha() for character in fact.text) or fact.text.isdigit(), fact.text


def test_number_matching_uses_the_sig_normalizer() -> None:
    fact = KeyFact(text="$500,000", kind="number")
    assert fact_in_text(fact, "The cap is $500k [NBC-1.1].")
    assert fact_in_text(fact, "Up to 500,000 dollars [NBC-1.1].")
    assert not fact_in_text(fact, "Up to 500,000 [NBC-1.1].")  # no currency: a different quantity
    assert not fact_in_text(fact, "Up to $50,000 [NBC-1.1].")
    months = KeyFact(text="24 months", kind="number")
    assert fact_in_text(months, "at least two years")


def test_term_matching_is_casefolded_and_normalized() -> None:
    fact = KeyFact(text="large-loss excess", kind="term")
    assert fact_in_text(fact, "It is reported as LARGE–LOSS   excess.")
    assert not fact_in_text(fact, "It is reported as a large loss.")


def test_a_number_fact_without_a_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="no quantity"):
        fact_in_text(KeyFact(text="Senior Underwriter", kind="number"), "Senior Underwriter")
