"""Tests for the Hugging Face dataset store with a fake Hub API (no network)."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from uwcopilot.web.app import create_app
from uwcopilot.web.crypto import Encryptor, generate_key
from uwcopilot.web.hf_store import HFDatasetStore
from uwcopilot.web.store import LocalFileStore, StoreError, owner_digest

OWNER = "hf_oauth:6123abc"
MARKER = "SYNTHETIC-lease-quote-Marker-4410"


class FakeHubApi:
    """An in-memory dataset repo with the HfApi methods HFDatasetStore calls; records every call."""

    def __init__(self, private: bool = True) -> None:
        self.private = private
        self.files: dict[str, bytes] = {".gitattributes": b"*.enc binary\n"}
        self.calls: list[str] = []
        self.fail_deletes = False

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        assert repo_type == "dataset" and private is True and exist_ok is True
        self.calls.append("create_repo")

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any:
        return SimpleNamespace(private=self.private)

    def snapshot_download(self, repo_id: str, *, repo_type: str, local_dir: str) -> str:
        self.calls.append("snapshot_download")
        for path, data in self.files.items():
            target = Path(local_dir) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return local_dir

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        return sorted(self.files)

    def create_commit(self, repo_id: str, operations: list[Any], *, commit_message: str, repo_type: str) -> None:
        self.calls.append(f"commit:{len(operations)}")
        for operation in operations:
            self.files[operation.path_in_repo] = operation.path_or_fileobj

    def delete_files(self, repo_id: str, delete_patterns: list[str], *, repo_type: str, commit_message: str) -> None:
        if self.fail_deletes:
            raise OSError("Hub unavailable")
        self.calls.append(f"delete:{len(delete_patterns)}")
        for pattern in delete_patterns:
            for path in list(self.files):
                if path == pattern or (pattern.endswith("/") and path.startswith(pattern)):
                    del self.files[path]


@pytest.fixture
def key() -> bytes:
    return generate_key().encode()


def make_store(root: Path, api: FakeHubApi, key: bytes) -> HFDatasetStore:
    """A restored dataset store whose working copy is root."""
    store = HFDatasetStore(LocalFileStore(root, Encryptor([key])), api, "astra6/fundproof-data", flush_every_s=3600)
    store.restore()
    return store


def remote_object_paths(api: FakeHubApi) -> list[str]:
    """Remote files other than .gitattributes."""
    return sorted(path for path in api.files if path != ".gitattributes")


def test_restore_creates_a_private_repo_and_refuses_a_public_one(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    make_store(tmp_path / "a", api, key)
    assert api.calls[:2] == ["create_repo", "snapshot_download"]
    with pytest.raises(StoreError, match="public"):
        make_store(tmp_path / "b", FakeHubApi(private=False), key)


def test_writes_are_batched_until_flush_and_uploaded_encrypted(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.put_json(OWNER, "projects", "p1", {"note": MARKER})
    store.put_bytes(OWNER, "uploads", "quote.pdf", MARKER.encode())
    assert remote_object_paths(api) == []  # nothing is sent before the flush
    assert store.flush() == 4  # two objects, payload + metadata each, in ONE commit
    assert [call for call in api.calls if call.startswith("commit")] == ["commit:4"]
    digest = owner_digest(OWNER)
    assert f"{digest}/projects/p1.enc" in api.files
    assert all(MARKER.encode() not in data for data in api.files.values())
    assert store.flush() == 0  # nothing new


def test_a_new_process_restores_the_data(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    first = make_store(tmp_path / "first", api, key)
    first.put_json(OWNER, "projects", "p1", {"note": MARKER})
    first.close()  # close() flushes the last batch
    second = make_store(tmp_path / "second", api, key)
    assert second.get_json(OWNER, "projects", "p1") == {"note": MARKER}


def test_deletes_reach_the_hub_at_once(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.put_json(OWNER, "projects", "p1", {"a": 1})
    store.put_json(OWNER, "projects", "p2", {"a": 2})
    store.flush()
    assert store.delete(OWNER, "projects", "p1") is True
    assert not any("/p1." in path for path in api.files)  # gone remotely without waiting for a flush
    store.put_json(OWNER, "projects", "draft", {"a": 3})
    calls_before = len(api.calls)
    store.delete(OWNER, "projects", "draft")  # never uploaded: no Hub call needed
    assert len(api.calls) == calls_before
    assert store.flush() == 0  # and the deleted draft is not uploaded later


def test_delete_owner_removes_the_remote_folder(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.put_json(OWNER, "projects", "p1", {"a": 1})
    store.put_json("hf_oauth:other", "projects", "p1", {"b": 1})
    store.flush()
    assert store.delete_owner(OWNER) == 1
    assert not any(path.startswith(owner_digest(OWNER)) for path in api.files)
    assert any(path.startswith(owner_digest("hf_oauth:other")) for path in api.files)


def test_purge_expired_deletes_remotely(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.put_json(OWNER, "projects", "p1", {"a": 1}, retention_days=1)
    store.flush()
    real_clock = store.local.clock
    store.local.clock = lambda: real_clock() + timedelta(days=2)
    assert [meta.key for meta in store.purge_expired()] == ["p1"]
    assert remote_object_paths(api) == []


def test_failed_remote_delete_is_retried_at_the_next_flush(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.put_json(OWNER, "projects", "p1", {"a": 1})
    store.flush()
    api.fail_deletes = True
    store.delete(OWNER, "projects", "p1")  # logged, not raised: the local copy is already gone
    assert remote_object_paths(api) != []
    api.fail_deletes = False
    store.flush()
    assert remote_object_paths(api) == []


def test_background_thread_starts_and_close_flushes(tmp_path: Path, key: bytes) -> None:
    api = FakeHubApi()
    store = make_store(tmp_path / "work", api, key)
    store.start()
    store.put_json(OWNER, "projects", "p1", {"a": 1})
    store.close()
    assert len(remote_object_paths(api)) == 2


def test_app_opens_and_closes_the_dataset_store(make_settings, key: bytes) -> None:
    api = FakeHubApi()
    settings = make_settings(
        UWC_STORE="hf_dataset",
        UWC_HF_DATASET_REPO="astra6/fundproof-data",
        HF_TOKEN="hf_test_token",
        UWC_FERNET_KEY=key.decode(),
    )
    with TestClient(create_app(settings, hub_api=api)) as test_client:
        store = test_client.app.state.store
        assert isinstance(store, HFDatasetStore)
        store.put_json(OWNER, "projects", "p1", {"a": 1})
        assert remote_object_paths(api) == []
    # leaving the client runs the shutdown, which flushes the last batch
    assert len(remote_object_paths(api)) == 2
