"""Embedding reranker for the top rules-based candidates.

After the rules score has picked the top few list entries for a customer, each customer/list name pair also gets
a cosine similarity between sentence embeddings of the two normalized names. The final score blends them:

    final = (1 - w) * rules_score + w * max(0, cosine)

w (the reranker weight) is chosen on the training split and w = 0 is the rules-only baseline. Only the top
candidates are embedded, so the model never has to look at millions of list names.

The production embedder is sentence-transformers/all-MiniLM-L6-v2 run through fastembed (ONNX Runtime, no
torch). It replaces the PyTorch sentence-transformers package with the same model weights exported to ONNX.
Model files are downloaded from Hugging Face on first use into .cache/models (gitignored).

HashingEmbedder is a deterministic stand-in for tests: character 3-gram counts hashed into a fixed-size vector.
It needs no download and no network, and it is never used for reported results.
"""

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from uwcopilot.common.paths import MODEL_CACHE_DIR

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class TextEmbedder(Protocol):
    """Anything that turns texts into L2-normalized vectors (one row per text)."""

    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with unit-length rows."""
        ...


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to unit length (rows of zeros stay zero)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class FastEmbedEmbedder:
    """all-MiniLM-L6-v2 (or another fastembed model) with an in-memory cache of already embedded texts."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = MODEL_CACHE_DIR,
        threads: int = 2,
        batch_size: int = 256,
    ) -> None:
        """Load the ONNX model (downloads it on first use). threads limits CPU use on shared machines."""
        from fastembed import TextEmbedding  # imported here so tests and light imports never load onnxruntime

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir), threads=threads)
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts, computing only the ones not seen before."""
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            vectors = np.array(list(self.model.embed(missing, batch_size=self.batch_size)), dtype=np.float32)
            for text, vector in zip(missing, normalize_rows(vectors), strict=True):
                self.cache[text] = vector
        return np.stack([self.cache[text] for text in texts]) if texts else np.zeros((0, 1), dtype=np.float32)


class HashingEmbedder:
    """Deterministic offline embedder for tests: hashed character 3-gram counts, L2-normalized."""

    model_name = "test-hashing-trigram-embedder"

    def __init__(self, dim: int = 256) -> None:
        """dim is the vector length; collisions are harmless for tests."""
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        """One hashed 3-gram count vector per text."""
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"#{text}#"
            for i in range(len(padded) - 2):
                digest = hashlib.sha256(padded[i : i + 3].encode("utf-8")).digest()
                vectors[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        return normalize_rows(vectors)


def cosine_similarities(embedder: TextEmbedder, query: str, candidates: list[str]) -> np.ndarray:
    """Cosine similarity between one query text and each candidate text (vectors are unit length)."""
    if not candidates:
        return np.zeros(0, dtype=np.float32)
    vectors = embedder.embed([query, *candidates])
    return vectors[1:] @ vectors[0]


def blend(rules: float, cosine: float | None, weight: float) -> float:
    """final = (1 - weight) * rules + weight * max(0, cosine). With weight 0 or no cosine, final = rules."""
    if weight <= 0.0 or cosine is None:
        return rules
    return (1.0 - weight) * rules + weight * max(0.0, float(cosine))
