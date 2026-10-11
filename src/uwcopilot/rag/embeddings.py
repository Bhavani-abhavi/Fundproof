"""Text embedders: fastembed models for the real index, and deterministic fakes for offline tests.

Two kinds of vectors are stored for every chunk:
- dense: BAAI/bge-small-en-v1.5 (384 numbers, cosine similarity) captures meaning ("revenue" ~ "sales");
- sparse: Qdrant/bm25 (token ids with term-frequency weights; Qdrant applies the IDF part) captures exact words,
  ids and numbers ("R-NBC-12", "1.25").

fastembed runs ONNX models on the CPU (no torch). Models are downloaded from Hugging Face on first use into
.cache/models/ (gitignored).

The Hashing* classes are NOT semantic models. They hash words into vectors so that unit tests and CI can exercise
the Qdrant code paths without downloading anything. Never report retrieval quality measured with them.
"""

import copy
import re
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from uwcopilot.rag.settings import BGE_QUERY_INSTRUCTION, DENSE_MODEL_ID, SPARSE_MODEL_ID

WORD = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")


@dataclass(frozen=True)
class SparseVector:
    """A sparse vector: parallel lists of token ids and weights."""

    indices: list[int]
    values: list[float]


class DenseEmbedder(Protocol):
    """Turns text into fixed-size dense vectors."""

    model_id: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages to be stored in the index."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (some models want a query instruction prefix)."""
        ...

    def count_tokens(self, text: str) -> int:
        """Number of model tokens in text, without truncation."""
        ...


class SparseEmbedder(Protocol):
    """Turns text into sparse (token id -> weight) vectors."""

    model_id: str

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed passages to be stored in the index."""
        ...

    def embed_query(self, text: str) -> SparseVector:
        """Embed a search query."""
        ...


# ---------------------------------------------------------------------------------------------------------------
# fastembed models
# ---------------------------------------------------------------------------------------------------------------


class FastEmbedDense:
    """Dense embeddings with fastembed's TextEmbedding (default BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_id: str = DENSE_MODEL_ID,
        cache_dir: Path | None = None,
        threads: int | None = None,
        query_prefix: str | None = None,
    ) -> None:
        """Load the model (downloads it into cache_dir on first use).

        query_prefix defaults to the BGE query instruction for BGE models and to "" otherwise.
        """
        from fastembed import TextEmbedding

        self.model_id = model_id
        self._model = TextEmbedding(model_id, cache_dir=str(cache_dir) if cache_dir else None, threads=threads)
        if query_prefix is None:
            query_prefix = BGE_QUERY_INSTRUCTION if model_id.startswith("BAAI/bge") else ""
        self.query_prefix = query_prefix
        self.dim = len(self.embed_documents(["dimension probe"])[0])
        self._tokenizer = self._untruncated_tokenizer()

    def _untruncated_tokenizer(self) -> Any:
        """A copy of the model's tokenizer with truncation and padding switched off.

        fastembed's own token_count() truncates at the model's 512-token limit, which would hide exactly the
        overlong texts we need to find.
        """
        tokenizer = getattr(getattr(self._model, "model", None), "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(f"Cannot reach the tokenizer of {self.model_id} through fastembed.")
        untruncated = copy.deepcopy(tokenizer)
        untruncated.no_truncation()
        untruncated.no_padding()
        return untruncated

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages (no prefix)."""
        return [vector.tolist() for vector in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the model's query instruction prefix."""
        return next(iter(self._model.embed([self.query_prefix + text]))).tolist()

    def count_tokens(self, text: str) -> int:
        """Tokens the model would see for text, including the [CLS] and [SEP] tokens, without truncation."""
        return len(self._tokenizer.encode(text).ids)


class FastEmbedBM25:
    """Sparse BM25 vectors with fastembed's SparseTextEmbedding("Qdrant/bm25").

    Documents get BM25 term-frequency weights; queries get weight 1 per distinct token. The IDF part of BM25 is
    computed by Qdrant from the collection (the sparse vector is configured with modifier=IDF).
    """

    def __init__(self, model_id: str = SPARSE_MODEL_ID, cache_dir: Path | None = None, threads: int | None = None):
        """Load the BM25 tokenizer/stemmer files (downloads them into cache_dir on first use)."""
        from fastembed import SparseTextEmbedding

        self.model_id = model_id
        self._model = SparseTextEmbedding(model_id, cache_dir=str(cache_dir) if cache_dir else None, threads=threads)

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed passages with BM25 term weights."""
        return [
            SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())
            for vector in self._model.embed(list(texts))
        ]

    def embed_query(self, text: str) -> SparseVector:
        """Embed a query (each distinct stemmed token with weight 1)."""
        vector = next(iter(self._model.query_embed(text)))
        return SparseVector(indices=vector.indices.tolist(), values=vector.values.astype(float).tolist())


# ---------------------------------------------------------------------------------------------------------------
# Deterministic fakes (offline tests only; not semantic)
# ---------------------------------------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Lowercase word tokens (letters/digits, keeping ids such as "nbc-3.2" together)."""
    return WORD.findall(text.lower())


def stable_hash(token: str) -> int:
    """A hash that is the same in every Python process (unlike hash())."""
    return zlib.crc32(token.encode("utf-8"))


class HashingDenseEmbedder:
    """FAKE dense embedder: hashed word counts in `dim` buckets, L2-normalized (similar wording, similar vector)."""

    def __init__(self, dim: int = 256) -> None:
        """dim is the vector size."""
        self.model_id = f"fake-hashing-dense-{dim}"
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        """Bag-of-words vector of text."""
        vector = [0.0] * self.dim
        for word in words(text):
            vector[stable_hash(word) % self.dim] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else [1.0 / self.dim**0.5] * self.dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages."""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query (same as a passage)."""
        return self._vector(text)

    def count_tokens(self, text: str) -> int:
        """Word count."""
        return len(text.split())


class HashingSparseEmbedder:
    """FAKE sparse embedder: hashed word ids with term counts (documents) or 1.0 (queries)."""

    model_id = "fake-hashing-sparse"

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed passages with raw term counts."""
        vectors = []
        for text in texts:
            counts = Counter(stable_hash(word) for word in words(text))
            indices = sorted(counts)
            vectors.append(SparseVector(indices=indices, values=[float(counts[index]) for index in indices]))
        return vectors

    def embed_query(self, text: str) -> SparseVector:
        """Embed a query: weight 1 per distinct word."""
        indices = sorted({stable_hash(word) for word in words(text)})
        return SparseVector(indices=indices, values=[1.0] * len(indices))
