"""Integration test: real fastembed models + downloaded regulatory documents, local embedded Qdrant in tmp_path.

Excluded from `make test` (CI has no models and no data/raw). Run with `make test-integration`.
"""

from pathlib import Path

import pytest

from uwcopilot.rag.corpus import load_corpus
from uwcopilot.rag.eval_set import load_questions, validate_gold
from uwcopilot.rag.regulatory import REGULATORY_DIR, REGULATORY_SOURCES
from uwcopilot.rag.settings import DENSE_MODEL_MAX_TOKENS, RagSettings

pytestmark = pytest.mark.integration


def test_real_index_builds_and_search_tool_works(tmp_path: Path) -> None:
    missing = [source.file_name for source in REGULATORY_SOURCES if not (REGULATORY_DIR / source.file_name).exists()]
    if missing:
        pytest.skip(f"regulatory documents not downloaded: {missing} (run `make data-regulatory`)")

    from uwcopilot.rag.embeddings import FastEmbedBM25, FastEmbedDense
    from uwcopilot.rag.index import create_collection, index_chunks, make_qdrant_client, read_all_payloads
    from uwcopilot.rag.rerank import FastEmbedReranker
    from uwcopilot.rag.retriever import Retriever
    from uwcopilot.rag.search import search_guidelines
    from uwcopilot.tools.schemas import GuidelineQuery

    settings = RagSettings(qdrant_path=str(tmp_path / "qdrant"), collection="integration")
    dense = FastEmbedDense(cache_dir=settings.model_cache_dir, threads=2)
    sparse = FastEmbedBM25(cache_dir=settings.model_cache_dir, threads=2)
    corpus = load_corpus(count_tokens=dense.count_tokens)
    assert corpus.missing_regulatory == []
    assert max(dense.count_tokens(chunk.embedding_text()) for chunk in corpus.chunks) <= DENSE_MODEL_MAX_TOKENS

    client = make_qdrant_client(settings)
    create_collection(client, settings.collection, dense.dim)
    assert index_chunks(client, settings.collection, corpus.chunks, dense, sparse) == len(corpus.chunks)
    payloads = read_all_payloads(client, settings.collection)
    assert validate_gold(load_questions(), payloads) == []

    retriever = Retriever(
        client, settings.collection, dense, sparse, FastEmbedReranker(cache_dir=settings.model_cache_dir)
    )
    result = search_guidelines(GuidelineQuery(query="model risk management guidance", top_k=3), retriever=retriever)
    assert len(result.hits) == 3
    assert all(hit.citation == f"[{hit.section_id}]" for hit in result.hits)
    client.close()
