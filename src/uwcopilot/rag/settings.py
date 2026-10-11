"""Settings for the vector index: where Qdrant is and which models are used.

The same code runs against two kinds of Qdrant:
- local embedded mode (default): QdrantClient(path="qdrant_storage/"), no server needed; the folder is gitignored.
  Only one process can open the folder at a time.
- a Qdrant server: set UWC_QDRANT_URL (for example http://localhost:6333); UWC_QDRANT_API_KEY if it needs one.

Like the LLM settings, values are read only from these names, and a value in the repo .env file wins over the
process environment: UWC_QDRANT_URL, UWC_QDRANT_API_KEY, UWC_QDRANT_PATH, UWC_QDRANT_COLLECTION.

Model ids are fixed in code (not settings) because the evaluation results are only valid for these exact models.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from uwcopilot.common.paths import ENV_FILE, MODEL_CACHE_DIR, REPO_ROOT
from uwcopilot.common.settings import env_reader, is_http_url

DENSE_MODEL_ID = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_ID = "Qdrant/bm25"
RERANK_MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"
# BGE v1.5 model card: for short-query-to-passage retrieval, prefix queries (not passages) with this instruction.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Maximum input length of BAAI/bge-small-en-v1.5 (longer text is truncated by the model).
DENSE_MODEL_MAX_TOKENS = 512

DEFAULT_COLLECTION = "uwc_guidelines"
DEFAULT_QDRANT_PATH = REPO_ROOT / "qdrant_storage"
IN_MEMORY = ":memory:"


class RagSettingsError(ValueError):
    """Raised when a RAG setting has an invalid value."""


@dataclass(frozen=True)
class RagSettings:
    """Where the index lives. qdrant_url set means server mode; otherwise qdrant_path (or ":memory:") is used."""

    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_path: str = str(DEFAULT_QDRANT_PATH)
    collection: str = DEFAULT_COLLECTION
    model_cache_dir: Path = MODEL_CACHE_DIR

    @property
    def mode(self) -> str:
        """Which kind of Qdrant: server, in-memory or local-embedded."""
        if self.qdrant_url:
            return "server"
        if self.qdrant_path == IN_MEMORY:
            return "in-memory"
        return "local-embedded"


def validate_qdrant_url(url: str) -> str:
    """Require an http(s) URL; trailing slashes are removed."""
    cleaned = url.strip().rstrip("/")
    if not is_http_url(cleaned):
        raise RagSettingsError(f"UWC_QDRANT_URL must start with http:// or https:// (got {url!r}).")
    return cleaned


def load_rag_settings(env_file: Path | None = ENV_FILE, environ: Mapping[str, str] | None = None) -> RagSettings:
    """Build RagSettings from the .env file and the process environment (the .env file wins).

    Pass env_file=None to ignore the .env file (tests do this).
    """
    read = env_reader(env_file, environ)
    url = read("UWC_QDRANT_URL")
    return RagSettings(
        qdrant_url=validate_qdrant_url(url) if url else None,
        qdrant_api_key=read("UWC_QDRANT_API_KEY"),
        qdrant_path=read("UWC_QDRANT_PATH") or str(DEFAULT_QDRANT_PATH),
        collection=read("UWC_QDRANT_COLLECTION") or DEFAULT_COLLECTION,
    )
