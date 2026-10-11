"""Build the Qdrant index of guideline sections and regulatory documents.

Steps:
1. Load the fastembed models (downloaded into .cache/models/ on first run): BAAI/bge-small-en-v1.5 (dense) and
   Qdrant/bm25 (sparse).
2. Load the corpus: corpus/guidelines/*.md (SYNTHETIC) and the regulatory documents in data/raw/regulatory/
   (download them first with `make data-regulatory`). Windows are sized with the dense model's own tokenizer.
3. Check that no chunk is longer than the dense model's 512-token input (longer text would be truncated).
4. Recreate the collection (named dense vector + sparse BM25 vector with IDF) and upsert every chunk.
5. Write results/rag_index_manifest.json: source files and checksums, chunk counts, token statistics, models,
   chunking parameters and the Qdrant mode/version.

Usage:
    python scripts/build_index.py                     # local embedded Qdrant in qdrant_storage/
    UWC_QDRANT_URL=http://localhost:6333 python scripts/build_index.py   # a Qdrant server
    python scripts/build_index.py --guidelines-only   # skip regulatory documents
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from uwcopilot.common.paths import RESULTS_DIR
from uwcopilot.rag.chunking import ChunkingParams
from uwcopilot.rag.corpus import Corpus, load_corpus
from uwcopilot.rag.embeddings import FastEmbedBM25, FastEmbedDense
from uwcopilot.rag.index import (
    create_collection,
    create_payload_indexes,
    describe_qdrant,
    index_chunks,
    make_qdrant_client,
)
from uwcopilot.rag.metrics import percentile
from uwcopilot.rag.settings import DENSE_MODEL_MAX_TOKENS, load_rag_settings

MANIFEST_FILE = RESULTS_DIR / "rag_index_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=2, help="ONNX threads for embedding (default 2)")
    parser.add_argument("--batch-size", type=int, default=16, help="chunks embedded and upserted per batch")
    parser.add_argument("--guidelines-only", action="store_true", help="index only the guideline markdown files")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE, help="where to write the build manifest")
    return parser.parse_args(argv)


def token_stats(corpus: Corpus, dense: FastEmbedDense) -> dict[str, dict[str, float]]:
    """Min / median / max tokens of the embedded text per document, counted with the dense model's tokenizer."""
    by_doc: dict[str, list[int]] = {}
    for chunk in corpus.chunks:
        by_doc.setdefault(chunk.doc_id, []).append(dense.count_tokens(chunk.embedding_text()))
    return {
        doc_id: {"min": min(counts), "median": percentile(counts, 50), "max": max(counts), "chunks": len(counts)}
        for doc_id, counts in sorted(by_doc.items())
    }


def build_manifest(
    corpus: Corpus,
    dense: FastEmbedDense,
    sparse: FastEmbedBM25,
    qdrant: dict,
    params: ChunkingParams,
    written: int,
    seconds: float,
) -> dict:
    """Everything needed to say exactly what was indexed and how."""
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generated_by": "scripts/build_index.py",
        "corpus_checksum_sha256": corpus.checksum(),
        "source_files": [asdict(source) for source in corpus.files],
        "missing_regulatory_files": corpus.missing_regulatory,
        "skipped_regulatory_files": corpus.skipped_regulatory,
        "chunks_by_doc": corpus.chunk_counts(),
        "points_written": written,
        "split_sections": sorted({chunk.chunk_id for chunk in corpus.chunks if chunk.part_count > 1}),
        "embedded_text_tokens_by_doc": token_stats(corpus, dense),
        "chunking": asdict(params),
        "models": {
            "dense": {"id": dense.model_id, "dim": dense.dim, "query_prefix": dense.query_prefix},
            "sparse": {"id": sparse.model_id, "qdrant_modifier": "idf"},
            "fastembed_version": version("fastembed"),
        },
        "qdrant": qdrant,
        "build_seconds": round(seconds, 2),
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the index and write the manifest. Returns a process exit code."""
    args = parse_args(argv)
    settings = load_rag_settings()
    started = time.perf_counter()
    print(f"Loading models into {settings.model_cache_dir} ...")
    dense = FastEmbedDense(cache_dir=settings.model_cache_dir, threads=args.threads)
    sparse = FastEmbedBM25(cache_dir=settings.model_cache_dir, threads=args.threads)

    params = ChunkingParams()
    corpus = load_corpus(count_tokens=dense.count_tokens, params=params, include_regulatory=not args.guidelines_only)
    if corpus.missing_regulatory:
        print(f"WARNING: regulatory files not downloaded (run `make data-regulatory`): {corpus.missing_regulatory}")
    too_long = [
        (chunk.point_key, dense.count_tokens(chunk.embedding_text()))
        for chunk in corpus.chunks
        if dense.count_tokens(chunk.embedding_text()) > DENSE_MODEL_MAX_TOKENS
    ]
    if too_long:
        print(f"ERROR: chunks longer than {DENSE_MODEL_MAX_TOKENS} tokens would be truncated: {too_long}")
        return 1
    print(f"Chunks: {corpus.chunk_counts()} (corpus checksum {corpus.checksum()[:12]}...)")

    client = make_qdrant_client(settings)
    try:
        create_collection(client, settings.collection, dense.dim, recreate=True)
        if settings.mode == "server":
            create_payload_indexes(client, settings.collection)
        written = index_chunks(client, settings.collection, corpus.chunks, dense, sparse, batch_size=args.batch_size)
        qdrant = describe_qdrant(client, settings)
    finally:
        client.close()
    seconds = time.perf_counter() - started

    manifest = build_manifest(corpus, dense, sparse, qdrant, params, written, seconds)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Indexed {written} points into {settings.collection!r} ({qdrant['mode']}) in {seconds:.1f} s.")
    print(f"Manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
