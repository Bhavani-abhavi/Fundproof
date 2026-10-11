"""Search the Qdrant index in four modes and return ranked chunks.

- "dense": nearest neighbours of the query's BGE embedding (meaning-based).
- "bm25": BM25 over the sparse vectors (exact words, ids and numbers).
- "hybrid": Qdrant query API with two prefetches (dense and BM25, each `prefetch_limit` deep) fused by
  reciprocal rank fusion with constant `rrf_k` (see fusion.py for the formula).
- "hybrid_rerank": the top `rerank_candidates` hybrid results reordered by a cross-encoder (see rerank.py).

All parameters are fixed before evaluation (RetrievalParams defaults) and are not tuned on the evaluation set.

A long guideline section can be stored as several parts with the same chunk_id; results are de-duplicated by
chunk_id (the best-ranked part is kept), so a section is never listed twice.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from qdrant_client import QdrantClient, models

from uwcopilot.rag.chunks import chunk_from_payload
from uwcopilot.rag.embeddings import DenseEmbedder, SparseEmbedder
from uwcopilot.rag.index import DENSE_VECTOR, SPARSE_VECTOR, require_collection
from uwcopilot.rag.rerank import Reranker
from uwcopilot.tools.schemas import GuidelineHit

RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]
RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class RetrievalParams:
    """Retrieval settings, chosen before evaluation.

    - prefetch_limit: candidates taken from each of dense and BM25 before fusion.
    - rrf_k: RRF constant. 60 is the value used by Cormack et al. (2009); Qdrant's own default is 2.
    - rerank_candidates: hybrid results the cross-encoder reorders.
    - dedup_margin: extra results fetched so that removing duplicate parts still leaves top_k results.
    """

    prefetch_limit: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    dedup_margin: int = 5


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result: the stored payload, the mode's score, and its 1-based rank."""

    payload: dict[str, Any] = field(repr=False)
    score: float
    rank: int

    @property
    def chunk_id(self) -> str:
        """The citation id of the result."""
        return str(self.payload["chunk_id"])


class Retriever:
    """Runs searches against one Qdrant collection with the given embedders and (optional) reranker."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        dense: DenseEmbedder,
        sparse: SparseEmbedder,
        reranker: Reranker | None = None,
        params: RetrievalParams | None = None,
    ) -> None:
        """Store the components; check that the collection exists."""
        require_collection(client, collection)
        self.client = client
        self.collection = collection
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.params = params or RetrievalParams()

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid_rerank",
        top_k: int = 5,
        source_docs: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top_k chunks for query (best first), optionally limited to some document ids."""
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; use one of {RETRIEVAL_MODES}.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_filter = doc_filter(source_docs)
        fetch = top_k + self.params.dedup_margin
        if mode == "dense":
            points = self._dense(query, fetch, query_filter)
        elif mode == "bm25":
            points = self._bm25(query, fetch, query_filter)
        elif mode == "hybrid":
            points = self._hybrid(query, fetch, query_filter)
        else:
            candidates = self._hybrid(query, max(self.params.rerank_candidates, fetch), query_filter)
            points = self._rerank(query, candidates)
        return rank_unique(points, top_k)

    def _dense(self, query: str, limit: int, query_filter: models.Filter | None) -> list[models.ScoredPoint]:
        """Nearest neighbours by cosine similarity of the dense vectors."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=self.dense.embed_query(query),
            using=DENSE_VECTOR,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def _bm25(self, query: str, limit: int, query_filter: models.Filter | None) -> list[models.ScoredPoint]:
        """BM25 scores from the sparse vectors (an empty result when the query has no indexable words)."""
        vector = self.sparse.embed_query(query)
        if not vector.indices:
            return []
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=vector.indices, values=vector.values),
            using=SPARSE_VECTOR,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def _hybrid(self, query: str, limit: int, query_filter: models.Filter | None) -> list[models.ScoredPoint]:
        """Dense and BM25 prefetches fused by RRF inside Qdrant."""
        prefetches = [
            models.Prefetch(
                query=self.dense.embed_query(query),
                using=DENSE_VECTOR,
                limit=self.params.prefetch_limit,
                filter=query_filter,
            )
        ]
        sparse_vector = self.sparse.embed_query(query)
        if sparse_vector.indices:
            prefetches.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                    using=SPARSE_VECTOR,
                    limit=self.params.prefetch_limit,
                    filter=query_filter,
                )
            )
        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetches,
            query=models.RrfQuery(rrf=models.Rrf(k=self.params.rrf_k)),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return list(response.points)

    def _rerank(self, query: str, candidates: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
        """Reorder candidates by cross-encoder score (ties keep the hybrid order)."""
        if self.reranker is None:
            raise ValueError("mode 'hybrid_rerank' needs a reranker")
        texts = [chunk_from_payload(point.payload or {}).embedding_text() for point in candidates]
        scores = self.reranker.score(query, texts)
        order = sorted(range(len(candidates)), key=lambda index: -scores[index])
        return [candidates[index].model_copy(update={"score": scores[index]}) for index in order]


def doc_filter(source_docs: Sequence[str] | None) -> models.Filter | None:
    """A payload filter on doc_id, or None to search everything."""
    if not source_docs:
        return None
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=list(source_docs)))])


def rank_unique(points: Sequence[models.ScoredPoint], top_k: int) -> list[RetrievedChunk]:
    """Keep the first occurrence of each chunk_id, number them 1..n and cut at top_k."""
    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for point in points:
        payload = point.payload or {}
        chunk_id = str(payload.get("chunk_id"))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        results.append(RetrievedChunk(payload=payload, score=float(point.score), rank=len(results) + 1))
        if len(results) == top_k:
            break
    return results


def to_guideline_hits(results: Sequence[RetrievedChunk]) -> list[GuidelineHit]:
    """Convert search results to the tool schema's GuidelineHit (citation "[<chunk_id>]" is filled in)."""
    hits = []
    for result in results:
        payload = result.payload
        title = payload["title"]
        if payload.get("part_count", 1) > 1:
            title = f"{title} (part {payload['part']} of {payload['part_count']})"
        hits.append(
            GuidelineHit(
                section_id=payload["chunk_id"],
                title=title,
                source_doc=payload["doc_id"],
                text=payload["text"],
                score=result.score,
                rank=result.rank,
            )
        )
    return hits
