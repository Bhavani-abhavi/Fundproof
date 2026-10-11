"""Shared helpers for the RAG tests: small chunks, a fake-embedder retriever on in-memory Qdrant, a tiny PDF.

Everything here runs offline: the Hashing* embedders and TermOverlapReranker are deterministic fakes, and the
Qdrant client runs in-process (":memory:").
"""

from collections.abc import Sequence

from qdrant_client import QdrantClient

from uwcopilot.rag.chunks import Chunk
from uwcopilot.rag.embeddings import HashingDenseEmbedder, HashingSparseEmbedder
from uwcopilot.rag.index import create_collection, index_chunks
from uwcopilot.rag.rerank import TermOverlapReranker
from uwcopilot.rag.retriever import RetrievalParams, Retriever

TEST_COLLECTION = "test_guidelines"


def make_chunk(chunk_id: str, text: str, doc_id: str = "northwind_business_credit", **fields: object) -> Chunk:
    """A guideline-style chunk with sensible defaults."""
    values: dict[str, object] = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_title": "Test Guidelines (SYNTHETIC)",
        "doc_kind": "guideline",
        "title": f"{chunk_id} Test section",
        "heading_path": ["Test Guidelines (SYNTHETIC)", f"{chunk_id} Test section"],
        "text": text,
        "source_file": "corpus/guidelines/test.md",
        "synthetic": True,
    }
    values.update(fields)
    return Chunk.model_validate(values)


SAMPLE_CHUNKS: list[Chunk] = [
    make_chunk("NBC-4.2", "Minimum debt service coverage ratio: term loan 1.25, line of credit 1.35."),
    make_chunk("NBC-6.3", "Sanctions screening against the OFAC list; a potential match goes to Compliance."),
    make_chunk("NBC-3.3", "Transfers between accounts, refunds and loan proceeds are excluded from revenue."),
    make_chunk(
        "HM-6.1",
        "Windstorm tiers: coastal county locations need a named storm deductible of at least 2 percent.",
        doc_id="harborline_mutual_commercial",
    ),
    make_chunk(
        "HM-4.2",
        "Loss ratio above 55 percent is referred to a Senior Underwriter.",
        doc_id="harborline_mutual_commercial",
    ),
]


def build_fake_retriever(
    chunks: Sequence[Chunk] = SAMPLE_CHUNKS,
    with_reranker: bool = True,
    params: RetrievalParams | None = None,
) -> Retriever:
    """Index chunks with the fake embedders into a fresh in-memory Qdrant and return a Retriever over it."""
    client = QdrantClient(location=":memory:")
    dense = HashingDenseEmbedder(dim=128)
    sparse = HashingSparseEmbedder()
    create_collection(client, TEST_COLLECTION, dense.dim)
    index_chunks(client, TEST_COLLECTION, list(chunks), dense, sparse)
    reranker = TermOverlapReranker() if with_reranker else None
    return Retriever(client, TEST_COLLECTION, dense, sparse, reranker, params)


def make_text_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per string, one page per list (for pypdf tests)."""
    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids: list[int] = []
    next_id = 4
    for lines in pages:
        commands = ["BT", "/F1 11 Tf", "14 TL", "72 720 Td"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"({escaped}) Tj T*")
        commands.append("ET")
        stream = "\n".join(commands).encode("latin-1")
        content_id, page_id = next_id, next_id + 1
        next_id += 2
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    output = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(output)
        output += f"{object_id} 0 obj\n".encode() + objects[object_id] + b"\nendobj\n"
    xref_offset = len(output)
    size = max(objects) + 1
    output += f"xref\n0 {size}\n".encode() + b"0000000000 65535 f \n"
    for object_id in range(1, size):
        output += f"{offsets[object_id]:010d} 00000 n \n".encode()
    output += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(output)
