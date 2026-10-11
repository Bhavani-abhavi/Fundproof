"""Rerankers: score (query, passage) pairs to reorder the first-stage candidates.

Why rerank? The first stage embeds the query and every passage separately (a "bi-encoder"), so it never looks at
the query and a passage together. A cross-encoder reads both at once and can judge relevance much more closely,
but it is too slow to run on every chunk. So the hybrid search proposes ~20 candidates and the cross-encoder
reorders only those.

FastEmbedReranker uses fastembed's TextCrossEncoder (default Xenova/ms-marco-MiniLM-L-6-v2, an ONNX export of
a MiniLM cross-encoder trained on MS MARCO passage ranking). Scores are relevance logits: higher is more relevant;
they are not probabilities.

TermOverlapReranker is a deterministic FAKE for offline tests.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from uwcopilot.rag.embeddings import words
from uwcopilot.rag.settings import RERANK_MODEL_ID


class Reranker(Protocol):
    """Scores passages for a query; higher means more relevant."""

    model_id: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """One score per passage, in the same order as passages."""
        ...


class FastEmbedReranker:
    """Cross-encoder reranking with fastembed's TextCrossEncoder."""

    def __init__(self, model_id: str = RERANK_MODEL_ID, cache_dir: Path | None = None, threads: int | None = None):
        """Load the cross-encoder (downloads it into cache_dir on first use)."""
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_id = model_id
        self._model = TextCrossEncoder(model_id, cache_dir=str(cache_dir) if cache_dir else None, threads=threads)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logits for each passage."""
        if not passages:
            return []
        return [float(value) for value in self._model.rerank(query, list(passages))]


class TermOverlapReranker:
    """FAKE reranker: the share of the query's distinct words that appear in the passage."""

    model_id = "fake-term-overlap"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Fraction of query words found in each passage."""
        query_words = set(words(query))
        if not query_words:
            return [0.0 for _ in passages]
        return [len(query_words & set(words(passage))) / len(query_words) for passage in passages]
