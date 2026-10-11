"""Tests for the evaluation set file, gold matching, and the evaluation runner (with fakes)."""

from collections import Counter

import pytest
from pydantic import ValidationError
from rag_helpers import build_fake_retriever

from uwcopilot.rag.eval_set import (
    EvalQuestion,
    GoldRef,
    chunk_matches,
    gold_first_ranks,
    leaf_refs,
    lexical_overlap_stats,
    load_questions,
    normalize_for_match,
    validate_gold,
)
from uwcopilot.rag.evaluation import paired_comparison, run_mode, summarize_mode
from uwcopilot.rag.guidelines import load_guideline_chunks
from uwcopilot.rag.regulatory import REGULATORY_SOURCES


def test_committed_question_file_is_well_formed() -> None:
    questions = load_questions()
    assert 80 <= len(questions) <= 120
    counts = Counter(question.category for question in questions)
    assert 8 <= counts["unanswerable"] <= 12
    assert counts["multi_section"] >= 10 and counts["regulatory"] >= 10
    section_ids = {chunk.chunk_id for chunk in load_guideline_chunks()}
    regulatory_docs = {source.doc_id for source in REGULATORY_SOURCES}
    for question in questions:
        for ref in question.gold:
            for leaf in leaf_refs(ref):
                if leaf.section:
                    assert leaf.section in section_ids, question.id
                else:
                    assert leaf.doc in regulatory_docs and leaf.quote, question.id


def test_guideline_gold_resolves_against_real_sections() -> None:
    payloads = [chunk.payload() for chunk in load_guideline_chunks()]
    guideline_only = [
        question
        for question in load_questions()
        if all(leaf.section for ref in question.gold for leaf in leaf_refs(ref))
    ]
    assert validate_gold(guideline_only, payloads) == []


@pytest.mark.parametrize(
    "data",
    [
        {"section": "NBC-1.1", "doc": "x", "quote": "abcdefgh"},
        {"doc": "frb_sr_26_2", "page": 5},
        {"section": "NBC-1.1", "page": 5},
        {"any_of": [{"section": "NBC-1.1"}]},
    ],
)
def test_gold_ref_shapes_are_validated(data: dict) -> None:
    with pytest.raises(ValidationError):
        GoldRef.model_validate(data)


def test_question_category_rules() -> None:
    with pytest.raises(ValidationError):
        EvalQuestion(
            id="UA99", question="Something unanswerable?", category="unanswerable", gold=[GoldRef(section="X-1")]
        )
    with pytest.raises(ValidationError):
        EvalQuestion(
            id="MS99", question="Two sections please?", category="multi_section", gold=[GoldRef(section="X-1")]
        )


def test_passage_matching_normalizes_quotes_and_checks_page() -> None:
    payload = {"doc_id": "frb_sr_26_2", "pages": [4, 5], "paragraphs": [], "text": "a banking organization’s  risk"}
    ref = GoldRef(doc="frb_sr_26_2", page=5, quote="Banking organization's risk")
    assert chunk_matches(payload, ref)
    assert not chunk_matches(payload, GoldRef(doc="frb_sr_26_2", page=6, quote="Banking organization's risk"))
    assert not chunk_matches(payload, GoldRef(doc="other_doc", quote="Banking organization's risk"))
    assert normalize_for_match("A – “B”") == 'a - "b"'


def test_gold_first_ranks_with_any_of() -> None:
    question = EvalQuestion(
        id="MS98",
        question="Two things at once?",
        category="multi_section",
        gold=[GoldRef(section="B-1"), GoldRef(any_of=[GoldRef(section="C-1"), GoldRef(section="D-1")])],
    )
    retrieved = [{"chunk_id": "A-1"}, {"chunk_id": "D-1"}, {"chunk_id": "B-1"}]
    assert gold_first_ranks(question, retrieved) == [3, 2]


def test_lexical_overlap_stats_counts_shared_content_words() -> None:
    question = EvalQuestion(
        id="NB99", question="revenue verification deposits", category="single_section", gold=[GoldRef(section="A-1")]
    )
    payloads = [
        {"chunk_id": "A-1", "title": "", "text": "Revenue is verified from deposits."},
        {"chunk_id": "B-1", "title": "", "text": "Revenue verification."},
    ]
    stats = lexical_overlap_stats([question], payloads)
    assert stats["mean_share_of_question_words_in_gold_text"] == 0.6667
    assert stats["mean_best_share_in_any_non_gold_chunk"] == 0.6667


def test_evaluation_runner_with_fake_retriever() -> None:
    retriever = build_fake_retriever()
    questions = [
        EvalQuestion(
            id="NB01",
            question="coverage ratio for a line of credit",
            category="single_section",
            gold=[GoldRef(section="NBC-4.2")],
        ),
        EvalQuestion(
            id="NB02",
            question="coastal windstorm deductible",
            category="single_section",
            gold=[GoldRef(section="HM-6.1")],
        ),
        EvalQuestion(id="UA01", question="crop insurance deductibles", category="unanswerable", gold=[]),
    ]
    dense = run_mode(retriever, questions, "dense")
    rerank = run_mode(retriever, questions, "hybrid_rerank")
    assert [result.gold_ranks for result in dense] == [[1], [1], []]
    summary = summarize_mode(dense, n_resamples=50, seed=42)
    assert summary["n_answerable"] == 2
    assert summary["metrics"]["recall@1"] == {"mean": 1.0, "ci95": [1.0, 1.0]}
    assert set(summary["metrics"]) == {"recall@1", "recall@3", "recall@5", "mrr@10", "ndcg@10"}
    assert summary["latency_ms"]["n_queries"] == 3 and summary["latency_ms"]["p50"] >= 0
    comparison = paired_comparison(rerank, dense, "recall@5", n_resamples=50, seed=42)
    assert comparison["n_questions"] == 2 and comparison["mean_difference"] == 0.0
