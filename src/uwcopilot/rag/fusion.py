"""Reciprocal rank fusion (RRF), written out in plain Python.

Hybrid search runs a dense search and a BM25 search and must merge two ranked lists whose scores are on
different scales (cosine similarity vs BM25). RRF ignores the scores and uses only the ranks:

    fused_score(item) = sum over lists containing the item of  1 / (k + position)

where position is the item's 0-based position in that list (Qdrant's convention; the paper by Cormack, Clarke
and Buettcher (2009) writes it as 1/(k + rank) with 1-based rank, which is the same as using k - 1 here). A larger
k flattens the difference between the top positions. Items that appear high in both lists win.

The retriever does not call this function: Qdrant computes the fusion itself (query API with prefetch + RrfQuery).
This module is the reference used by the tests to check that the Qdrant query is configured the way we intend,
and it documents the formula in one place.
"""

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists of ids; return (id, fused score), best first.

    Ties keep the order in which ids were first seen (first list first), like Qdrant's local implementation.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
