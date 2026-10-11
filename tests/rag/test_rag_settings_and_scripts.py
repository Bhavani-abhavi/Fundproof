"""Tests for RAG settings (explicit names only, .env wins) and the scripts' command-line defaults."""

from pathlib import Path

import pytest

import build_index
import eval_rag
from uwcopilot.rag.settings import DEFAULT_COLLECTION, RagSettingsError, load_rag_settings


def test_defaults_use_local_embedded_storage() -> None:
    settings = load_rag_settings(env_file=None, environ={})
    assert settings.mode == "local-embedded"
    assert settings.collection == DEFAULT_COLLECTION
    assert settings.qdrant_path.endswith("qdrant_storage")


def test_server_url_and_api_key_are_read_from_project_names_only() -> None:
    environ = {"UWC_QDRANT_URL": "http://localhost:6333/", "UWC_QDRANT_API_KEY": "secret-key", "QDRANT_URL": "x"}
    settings = load_rag_settings(env_file=None, environ=environ)
    assert settings.mode == "server"
    assert settings.qdrant_url == "http://localhost:6333"
    assert "secret-key" not in repr(settings)
    assert load_rag_settings(env_file=None, environ={"QDRANT_URL": "http://elsewhere"}).mode == "local-embedded"


def test_env_file_wins_over_process_environment(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("UWC_QDRANT_COLLECTION=from_file\nUWC_QDRANT_PATH=:memory:\n", encoding="utf-8")
    settings = load_rag_settings(env_file=env_file, environ={"UWC_QDRANT_COLLECTION": "from_env"})
    assert settings.collection == "from_file"
    assert settings.mode == "in-memory"


def test_invalid_url_is_rejected() -> None:
    with pytest.raises(RagSettingsError):
        load_rag_settings(env_file=None, environ={"UWC_QDRANT_URL": "localhost:6333"})


def test_script_defaults() -> None:
    build_args = build_index.parse_args([])
    assert build_args.threads == 2 and not build_args.guidelines_only
    eval_args = eval_rag.parse_args([])
    assert eval_args.modes == ["dense", "bm25", "hybrid", "hybrid_rerank"]
    assert eval_args.resamples == 1000 and eval_args.seed == 42
    assert eval_args.output.name == "rag_metrics.json"
