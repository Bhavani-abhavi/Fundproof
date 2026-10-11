"""The Chunk model: one retrievable piece of a guideline or regulatory document, with its citation metadata.

A chunk's chunk_id is what a memo cites, for example "[NBC-3.2]" for a guideline section or "[SR26-2-p5-1]" for
the first chunk that starts on page 5 of the Federal Reserve letter SR 26-2. Everything else in the chunk is
metadata that is stored as the Qdrant payload and shown to people next to the citation.
"""

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from uwcopilot.tools.schemas import SECTION_ID_PATTERN, SOURCE_DOC_PATTERN

DocKind = Literal["guideline", "regulatory"]


class Chunk(BaseModel):
    """One retrievable unit of text.

    - chunk_id: the citation id (a guideline section id such as "NBC-3.2", or a regulatory window id).
    - doc_id: lowercase id of the source document, for example "northwind_business_credit".
    - title: a human-readable label shown with search hits (section title, or document + page/paragraph + heading).
    - heading_path: the headings above this text, outermost first.
    - pages: 1-based PDF page numbers the text came from (empty for markdown and HTML).
    - paragraphs: numbered paragraphs covered, for documents that number their paragraphs (empty otherwise).
    - synthetic: True for the project's own SYNTHETIC guidelines, False for public regulatory documents.
    - part / part_count: a guideline section longer than the embedding model's input is stored as several parts
      that share one chunk_id (and so one citation); search results keep only the best part of each section.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(pattern=SECTION_ID_PATTERN)
    doc_id: str = Field(pattern=SOURCE_DOC_PATTERN)
    doc_title: str = Field(min_length=1)
    doc_kind: DocKind
    title: str = Field(min_length=1)
    heading_path: list[str]
    text: str = Field(min_length=1)
    pages: list[int] = Field(default_factory=list)
    paragraphs: list[int] = Field(default_factory=list)
    source_file: str = Field(min_length=1)
    synthetic: bool
    issuer: str | None = None
    doc_date: str | None = None
    source_url: str | None = None
    part: int = Field(default=1, ge=1)
    part_count: int = Field(default=1, ge=1)

    @property
    def point_key(self) -> str:
        """Unique key of this stored piece: the chunk id, plus "#<part>" when a section has several parts."""
        return self.chunk_id if self.part_count == 1 else f"{self.chunk_id}#{self.part}"

    @property
    def text_sha256(self) -> str:
        """SHA-256 of the chunk text, used to check that an index still matches the source files."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def embedding_text(self) -> str:
        """The text that is embedded and reranked: a one-line header (document and title) plus the chunk text.

        The header lets a query such as "Harborline loss runs" match on the document name and section title,
        not only on words in the body.
        """
        part_note = f" (part {self.part} of {self.part_count})" if self.part_count > 1 else ""
        return f"{self.doc_title} | {self.title}{part_note}\n{self.text}"

    def payload(self) -> dict[str, Any]:
        """Everything stored with the vectors in Qdrant: the chunk fields plus the text checksum."""
        data = self.model_dump()
        data["text_sha256"] = self.text_sha256
        return data


def chunk_from_payload(payload: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from a Qdrant payload (the stored checksum is dropped; it is recomputed from the text)."""
    fields = {key: value for key, value in payload.items() if key != "text_sha256"}
    return Chunk.model_validate(fields)
