"""Index and retrieval tests on an in-memory Qdrant with deterministic fake embedders (no downloads)."""

from collections.abc import Iterator

import pytest
from qdrant_client import QdrantClient
from rag_helpers import SAMPLE_CHUNKS, TEST_COLLECTION, build_fake_retriever, make_chunk

from uwcopilot.rag import search as search_module
from uwcopilot.rag.fusion import reciprocal_rank_fusion
from uwcopilot.rag.index import (
    IndexNotBuiltError,
    index_matches_corpus,
    make_qdrant_client,
    point_id,
    read_all_payloads,
)
from uwcopilot.rag.retriever import RETRIEVAL_MODES, RetrievalParams, Retriever, to_guideline_hits
from uwcopilot.rag.search import search_guidelines, set_default_retriever
from uwcopilot.rag.settings import RagSettings
from uwcopilot.tools.registry import call_tool
from uwcopilot.tools.schemas import GuidelineQuery, GuidelineSearchResult

QUERY = "minimum debt service coverage ratio for a line of credit"


@pytest.fixture(autouse=True)
def reset_default_retriever() -> Iterator[None]:
    """Never leak an installed retriever into other tests."""
    yield
    set_default_retriever(None)


@pytest.mark.parametrize("mode", RETRIEVAL_MODES)
def test_every_mode_ranks_the_matching_section_first(mode: str) -> None:
    retriever = build_fake_retriever()
    results = retriever.search(QUERY, mode=mode, top_k=3)
    assert results[0].chunk_id == "NBC-4.2"
    assert [result.rank for result in results] == list(range(1, len(results) + 1))


def test_source_docs_filter_limits_results() -> None:
    retriever = build_fake_retriever()
    for mode in RETRIEVAL_MODES:
        results = retriever.search(
            "referred to a senior underwriter", mode=mode, source_docs=["harborline_mutual_commercial"]
        )
        assert results and {result.payload["doc_id"] for result in results} == {"harborline_mutual_commercial"}


def test_hybrid_order_equals_reference_rrf_of_dense_and_bm25() -> None:
    params = RetrievalParams(prefetch_limit=4, rrf_k=60)
    retriever = build_fake_retriever(params=params)
    query = "coastal windstorm deductible and loss ratio referral"
    dense_ids = [point.payload["chunk_id"] for point in retriever._dense(query, params.prefetch_limit, None)]
    bm25_ids = [point.payload["chunk_id"] for point in retriever._bm25(query, params.prefetch_limit, None)]
    expected = reciprocal_rank_fusion([dense_ids, bm25_ids], k=params.rrf_k)
    hybrid = retriever.search(query, mode="hybrid", top_k=len(expected))
    assert [result.chunk_id for result in hybrid] == [item for item, _score in expected]
    for result, (_item, score) in zip(hybrid, expected, strict=True):
        assert result.score == pytest.approx(score)


def test_reranker_reorders_hybrid_candidates_by_its_score() -> None:
    retriever = build_fake_retriever()
    results = retriever.search("loss ratio senior underwriter", mode="hybrid_rerank", top_k=5)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].chunk_id == "HM-4.2"


def test_hybrid_rerank_without_reranker_is_an_error() -> None:
    retriever = build_fake_retriever(with_reranker=False)
    with pytest.raises(ValueError, match="needs a reranker"):
        retriever.search(QUERY, mode="hybrid_rerank")
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        retriever.search(QUERY, mode="keyword")  # type: ignore[arg-type]


def test_parts_of_one_section_are_returned_once() -> None:
    parts = [
        make_chunk("HM-9.3", "Referral triggers part one: loss ratio and windstorm.", part=1, part_count=2),
        make_chunk("HM-9.3", "Referral triggers part two: loss ratio and sanctions.", part=2, part_count=2),
        make_chunk("NBC-4.2", "Coverage minimums."),
    ]
    retriever = build_fake_retriever(parts)
    for mode in RETRIEVAL_MODES:
        ids = [result.chunk_id for result in retriever.search("referral triggers loss ratio", mode=mode, top_k=3)]
        assert ids.count("HM-9.3") == 1


def test_index_round_trip_and_staleness_check() -> None:
    retriever = build_fake_retriever()
    payloads = read_all_payloads(retriever.client, TEST_COLLECTION)
    assert len(payloads) == len(SAMPLE_CHUNKS)
    assert index_matches_corpus(payloads, SAMPLE_CHUNKS) == []
    edited = [*SAMPLE_CHUNKS[:-1], SAMPLE_CHUNKS[-1].model_copy(update={"text": "Loss ratio above 60 percent."})]
    assert index_matches_corpus(payloads, edited) == ["text changed: HM-4.2"]
    assert index_matches_corpus(payloads, SAMPLE_CHUNKS[:-1]) == ["not in current corpus: HM-4.2"]


def test_point_ids_are_deterministic_and_distinct_per_part() -> None:
    first = make_chunk("HM-9.3", "x", part=1, part_count=2)
    second = make_chunk("HM-9.3", "y", part=2, part_count=2)
    assert point_id(first) == point_id(make_chunk("HM-9.3", "other text", part=1, part_count=2))
    assert point_id(first) != point_id(second)


def test_missing_collection_raises_index_not_built() -> None:
    client = make_qdrant_client(RagSettings(qdrant_path=":memory:"))
    with pytest.raises(IndexNotBuiltError, match="make index"):
        read_all_payloads(client, "does_not_exist")
    retriever = build_fake_retriever()
    with pytest.raises(IndexNotBuiltError):
        Retriever(QdrantClient(location=":memory:"), "nope", retriever.dense, retriever.sparse)


def test_hits_follow_the_tool_schema() -> None:
    retriever = build_fake_retriever()
    hits = to_guideline_hits(retriever.search(QUERY, top_k=3))
    assert hits[0].citation == "[NBC-4.2]"
    assert hits[0].source_doc == "northwind_business_credit"
    result = search_guidelines(GuidelineQuery(query=QUERY, top_k=3), retriever=retriever)
    assert isinstance(result, GuidelineSearchResult)
    assert [hit.rank for hit in result.hits] == [1, 2, 3]


def test_registry_calls_the_rag_handler_with_an_installed_retriever() -> None:
    set_default_retriever(build_fake_retriever())
    result = call_tool("search_guidelines", {"query": QUERY, "top_k": 2})
    assert isinstance(result, GuidelineSearchResult)
    assert result.hits[0].section_id == "NBC-4.2"
    assert search_module.get_default_retriever() is search_module._default_retriever
