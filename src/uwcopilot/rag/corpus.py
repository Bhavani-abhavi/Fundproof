"""Load the whole retrieval corpus (guidelines + regulatory documents) and fingerprint the source files.

The corpus checksum is a SHA-256 over the names and SHA-256 checksums of every indexed source file. It is written
into the index build manifest and into results/rag_metrics.json, so a reported number can always be tied to the
exact documents it was measured on.

Regulatory files that have not been downloaded are reported in Corpus.missing_regulatory instead of failing, so
the guideline index can still be built; the evaluation refuses to run when gold passages are missing.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from uwcopilot.common.files import sha256_of_file
from uwcopilot.rag.chunking import ChunkingParams, TokenCounter, count_words
from uwcopilot.rag.chunks import Chunk
from uwcopilot.rag.guidelines import GUIDELINES_DIR, load_guideline_chunks, split_long_section
from uwcopilot.rag.regulatory import (
    REGULATORY_DIR,
    REGULATORY_SOURCES,
    SKIPPED_REGULATORY_FILES,
    load_regulatory_document,
)


@dataclass(frozen=True)
class SourceFile:
    """One indexed source file: its path as recorded (relative to the repo), document id and checksum."""

    path: str
    doc_id: str
    sha256: str


@dataclass
class Corpus:
    """All chunks plus the files they came from, and which regulatory files were missing or skipped."""

    chunks: list[Chunk]
    files: list[SourceFile]
    missing_regulatory: list[str] = field(default_factory=list)
    skipped_regulatory: dict[str, str] = field(default_factory=dict)

    def checksum(self) -> str:
        """SHA-256 over "path sha256" lines of every source file, sorted by path."""
        lines = sorted(f"{source.path} {source.sha256}" for source in self.files)
        return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()

    def chunk_counts(self) -> dict[str, int]:
        """Number of chunks per document id."""
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
        return counts


def load_corpus(
    guidelines_dir: Path = GUIDELINES_DIR,
    regulatory_dir: Path = REGULATORY_DIR,
    count_tokens: TokenCounter = count_words,
    params: ChunkingParams | None = None,
    include_regulatory: bool = True,
) -> Corpus:
    """Load guideline sections and every downloaded regulatory document listed in REGULATORY_SOURCES.

    count_tokens sizes the regulatory windows and decides which guideline sections are too long and must be
    split; pass the embedding model's (untruncated) token counter for a real index.
    """
    params = params or ChunkingParams()
    sections = load_guideline_chunks(guidelines_dir)
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(split_long_section(section, params.max_section_tokens, count_tokens))
    files: list[SourceFile] = []
    for path in sorted(guidelines_dir.glob("*.md")):
        doc_id = next(chunk.doc_id for chunk in sections if chunk.source_file.endswith(path.name))
        files.append(SourceFile(path=f"corpus/guidelines/{path.name}", doc_id=doc_id, sha256=sha256_of_file(path)))

    missing: list[str] = []
    if include_regulatory:
        for source in REGULATORY_SOURCES:
            path = regulatory_dir / source.file_name
            if not path.exists():
                missing.append(source.file_name)
                continue
            chunks.extend(load_regulatory_document(path, source, params, count_tokens))
            files.append(
                SourceFile(
                    path=f"data/raw/regulatory/{source.file_name}", doc_id=source.doc_id, sha256=sha256_of_file(path)
                )
            )

    keys = [chunk.point_key for chunk in chunks]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate chunk ids in corpus: {duplicates}")
    skipped = {name: reason for name, reason in SKIPPED_REGULATORY_FILES.items() if (regulatory_dir / name).exists()}
    return Corpus(chunks=chunks, files=files, missing_regulatory=missing, skipped_regulatory=skipped)
