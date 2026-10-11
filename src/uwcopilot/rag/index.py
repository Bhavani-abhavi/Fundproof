"""Write chunks into a Qdrant collection and read them back.

Collection layout (one point per chunk, or per part of a long guideline section):
- named dense vector "dense": cosine distance, size = the dense model's dimension (384 for bge-small-en-v1.5);
- named sparse vector "bm25": BM25 term weights, with modifier=IDF so Qdrant computes the inverse document
  frequency from the collection itself (it stays correct when documents are added or removed);
- payload: every Chunk field plus text_sha256 (see chunks.Chunk.payload).

Point ids are UUIDs derived from the chunk's point_key (uuid5), so rebuilding the index gives every chunk the same
id again.

make_qdrant_client() returns a client for local embedded mode (a folder, or ":memory:") or for a server
(UWC_QDRANT_URL); the rest of the code is the same for both.
"""

import uuid
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.rag.chunks import Chunk
from uwcopilot.rag.embeddings import DenseEmbedder, SparseEmbedder
from uwcopilot.rag.settings import IN_MEMORY, RagSettings

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
# A fixed namespace so the same chunk always gets the same point id (the string is a label, not a web address).
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "urn:uwcopilot:rag-chunk")


class IndexNotBuiltError(RuntimeError):
    """Raised when the collection does not exist yet (run `make index`)."""


def make_qdrant_client(settings: RagSettings) -> QdrantClient:
    """Client for the configured Qdrant: server URL, in-memory, or a local folder."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if settings.qdrant_path == IN_MEMORY:
        return QdrantClient(location=IN_MEMORY)
    return QdrantClient(path=settings.qdrant_path)


def describe_qdrant(client: QdrantClient, settings: RagSettings) -> dict[str, Any]:
    """Which Qdrant was used: mode, client library version, and the server version in server mode.

    In local embedded and in-memory mode there is no server: qdrant-client itself stores and searches the
    vectors, so server_version is None.
    """
    server_version = None
    if settings.mode == "server":
        server_version = client.info().version
    return {
        "mode": settings.mode,
        "qdrant_client_version": version("qdrant-client"),
        "server_version": server_version,
        "url": settings.qdrant_url,
        "path": None if settings.mode == "server" else display_path(settings.qdrant_path),
        "collection": settings.collection,
    }


def display_path(path: str) -> str:
    """A storage path relative to the repo root when it is inside the repo (so results files hold no local paths)."""
    if path == IN_MEMORY:
        return path
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path


def point_id(chunk: Chunk) -> str:
    """Deterministic UUID for a chunk (or chunk part)."""
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk.point_key))


def create_collection(client: QdrantClient, collection: str, dense_dim: int, recreate: bool = True) -> None:
    """Create the collection with the named dense and sparse vectors (dropping an existing one if recreate)."""
    if client.collection_exists(collection):
        if not recreate:
            return
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def create_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Keyword indexes for filtered search (server mode only; local mode filters by scanning)."""
    for field_name in ("doc_id", "doc_kind", "chunk_id"):
        client.create_payload_index(collection, field_name=field_name, field_schema=models.PayloadSchemaType.KEYWORD)


def index_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    batch_size: int = 16,
) -> int:
    """Embed chunks (dense + sparse) and upsert them in batches. Returns the number of points written."""
    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = list(chunks[start : start + batch_size])
        texts = [chunk.embedding_text() for chunk in batch]
        dense_vectors = dense.embed_documents(texts)
        sparse_vectors = sparse.embed_documents(texts)
        points = [
            models.PointStruct(
                id=point_id(chunk),
                vector={
                    DENSE_VECTOR: dense_vector,
                    SPARSE_VECTOR: models.SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                },
                payload=chunk.payload(),
            )
            for chunk, dense_vector, sparse_vector in zip(batch, dense_vectors, sparse_vectors, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        written += len(points)
    return written


def require_collection(client: QdrantClient, collection: str) -> None:
    """Raise IndexNotBuiltError if the collection does not exist."""
    if not client.collection_exists(collection):
        raise IndexNotBuiltError(f"Qdrant collection {collection!r} does not exist. Build it with `make index`.")


def read_all_payloads(client: QdrantClient, collection: str, page_size: int = 256) -> list[dict[str, Any]]:
    """Every point's payload in the collection (used to check the index against the source files)."""
    require_collection(client, collection)
    payloads: list[dict[str, Any]] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection, limit=page_size, offset=offset, with_payload=True, with_vectors=False
        )
        payloads.extend(record.payload or {} for record in records)
        if offset is None:
            return payloads


def index_matches_corpus(payloads: Sequence[dict[str, Any]], chunks: Sequence[Chunk]) -> list[str]:
    """Differences between the indexed payloads and freshly loaded chunks ([] when they match exactly).

    Compares (point key -> text checksum) in both directions, so an added, removed or edited chunk is reported.
    """
    indexed: dict[str, str] = {}
    for payload in payloads:
        key = payload["chunk_id"] if payload.get("part_count", 1) == 1 else f"{payload['chunk_id']}#{payload['part']}"
        indexed[key] = payload.get("text_sha256", "")
    expected = {chunk.point_key: chunk.text_sha256 for chunk in chunks}
    problems = [f"missing from index: {key}" for key in sorted(expected.keys() - indexed.keys())]
    problems += [f"not in current corpus: {key}" for key in sorted(indexed.keys() - expected.keys())]
    problems += [
        f"text changed: {key}" for key in sorted(expected.keys() & indexed.keys()) if expected[key] != indexed[key]
    ]
    return problems
