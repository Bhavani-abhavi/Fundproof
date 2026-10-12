"""A Store mirrored to a PRIVATE Hugging Face dataset repo (used in the Space when UWC_STORE=hf_dataset).

HFDatasetStore keeps a LocalFileStore (store.py) as a working copy of a PRIVATE Hugging Face dataset repo:
- at startup restore() creates the repo if missing (private), refuses a public repo, and downloads a snapshot;
- writes go to the local copy and are uploaded in one batched commit every flush_every_s seconds (default 300) by
  a background thread, plus once more at shutdown. Data written less than one interval before a crash is lost;
- deletes and purges are sent to the Hub at once (HfApi.delete_files). If that call fails, it is retried at the
  next flush; until then the dataset still holds the encrypted file (eventual consistency).
- the Hub limits how often one repo can be committed to, which is why writes are batched; deletes are not, so a
  burst of deletions makes a burst of commits. One lock serializes uploads and deletes, so the Hub never receives
  an upload of a file after its deletion.
"""

# Postponed annotations: HFDatasetStore has a method named `list`, which would shadow the built-in list type inside
# the class body if annotations were evaluated.
from __future__ import annotations

import logging
import threading
from typing import Any, Protocol

from huggingface_hub import CommitOperationAdd

from uwcopilot.web.store import LocalFileStore, ObjectMeta, StoreError, owner_digest

logger = logging.getLogger(__name__)


class HubApi(Protocol):
    """The part of huggingface_hub.HfApi that HFDatasetStore uses (tests pass a fake)."""

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> Any:
        """Create the repo (no error if it exists)."""
        ...

    def repo_info(self, repo_id: str, *, repo_type: str) -> Any:
        """Repo facts; must have a `private` attribute."""
        ...

    def snapshot_download(self, repo_id: str, *, repo_type: str, local_dir: str) -> Any:
        """Download every file into local_dir."""
        ...

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        """Every file path in the repo."""
        ...

    def create_commit(
        self, repo_id: str, operations: list[CommitOperationAdd], *, commit_message: str, repo_type: str
    ) -> Any:
        """Upload files in one commit."""
        ...

    def delete_files(self, repo_id: str, delete_patterns: list[str], *, repo_type: str, commit_message: str) -> Any:
        """Delete files or folders in one commit."""
        ...


class HFDatasetStore:
    """A LocalFileStore mirrored to a private Hugging Face dataset repo (see the module docstring)."""

    def __init__(self, local: LocalFileStore, api: HubApi, repo_id: str, flush_every_s: int = 300) -> None:
        """local: the working copy (its root receives the snapshot); api: HfApi or a fake."""
        self.local = local
        self.api = api
        self.repo_id = repo_id
        self.flush_every_s = flush_every_s
        self._lock = threading.Lock()
        self._dirty: set[str] = set()
        self._pending_deletes: set[str] = set()
        self._remote_paths: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def restore(self) -> None:
        """Create the private repo if needed, refuse a public one, and download its files into the working copy."""
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = self.api.repo_info(self.repo_id, repo_type="dataset")
        if not getattr(info, "private", False):
            raise StoreError(f"dataset {self.repo_id} is public; user data may only go to a private dataset")
        self.local.root.mkdir(parents=True, exist_ok=True)
        self.api.snapshot_download(self.repo_id, repo_type="dataset", local_dir=str(self.local.root))
        with self._lock:
            self._remote_paths = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))

    def start(self) -> None:
        """Start the background thread that flushes every flush_every_s seconds."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._flush_loop, name="hf-dataset-store-flush", daemon=True)
        self._thread.start()

    def _flush_loop(self) -> None:
        """Flush until close() is called; a failed flush is logged and retried next time."""
        while not self._stop.wait(self.flush_every_s):
            try:
                self.flush()
            except Exception:
                logger.exception("dataset store flush failed; it will be retried")

    def close(self) -> None:
        """Stop the thread and flush one last time."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None
        self.flush()

    # ---- syncing ----

    def flush(self) -> int:
        """Send pending deletes, then upload every changed file in one commit. Returns the number uploaded."""
        with self._lock:
            if self._pending_deletes:
                self._delete_remote(sorted(self._pending_deletes))
            operations = [
                CommitOperationAdd(path_in_repo=relative, path_or_fileobj=(self.local.root / relative).read_bytes())
                for relative in sorted(self._dirty)
                if (self.local.root / relative).is_file()
            ]
            if operations:
                self.api.create_commit(
                    self.repo_id,
                    operations,
                    commit_message=f"store: upload {len(operations)} encrypted file(s)",
                    repo_type="dataset",
                )
                self._remote_paths.update(operation.path_in_repo for operation in operations)
            self._dirty.clear()
            return len(operations)

    def _delete_remote(self, paths: list[str]) -> None:
        """Delete paths (files or 'folder/') on the Hub now; on failure remember them for the next flush.

        Call with the lock held.
        """
        if not paths:
            return
        try:
            self.api.delete_files(
                self.repo_id,
                delete_patterns=paths,
                repo_type="dataset",
                commit_message=f"store: delete {len(paths)} path(s)",
            )
        except Exception:
            logger.exception("deleting %d path(s) from the dataset failed; retrying at the next flush", len(paths))
            self._pending_deletes.update(paths)
            return
        self._pending_deletes.difference_update(paths)
        for path in paths:
            if path.endswith("/"):
                self._remote_paths = {remote for remote in self._remote_paths if not remote.startswith(path)}
            else:
                self._remote_paths.discard(path)

    def _mark_dirty(self, owner: str, namespace: str, key: str) -> None:
        """Remember an object's files for the next upload (call with the lock held)."""
        paths = self.local.relative_paths(owner, namespace, key)
        self._dirty.update(paths)
        self._pending_deletes.difference_update(paths)

    # ---- Store interface ----

    def put_json(
        self, owner: str, namespace: str, key: str, value: Any, *, retention_days: int | None = None
    ) -> ObjectMeta:
        """Write locally; uploaded at the next flush."""
        with self._lock:
            meta = self.local.put_json(owner, namespace, key, value, retention_days=retention_days)
            self._mark_dirty(owner, namespace, key)
            return meta

    def get_json(self, owner: str, namespace: str, key: str) -> Any:
        """Read from the working copy."""
        return self.local.get_json(owner, namespace, key)

    def put_bytes(
        self, owner: str, namespace: str, key: str, data: bytes, *, retention_days: int | None = None
    ) -> ObjectMeta:
        """Write locally; uploaded at the next flush."""
        with self._lock:
            meta = self.local.put_bytes(owner, namespace, key, data, retention_days=retention_days)
            self._mark_dirty(owner, namespace, key)
            return meta

    def get_bytes(self, owner: str, namespace: str, key: str) -> bytes:
        """Read from the working copy."""
        return self.local.get_bytes(owner, namespace, key)

    def list(self, owner: str, namespace: str) -> list[ObjectMeta]:
        """List from the working copy."""
        return self.local.list(owner, namespace)

    def delete(self, owner: str, namespace: str, key: str) -> bool:
        """Delete locally and on the Hub right away."""
        with self._lock:
            found = self.local.delete(owner, namespace, key)
            paths = self.local.relative_paths(owner, namespace, key)
            self._dirty.difference_update(paths)
            self._delete_remote([path for path in paths if path in self._remote_paths])
            return found

    def delete_owner(self, owner: str) -> int:
        """Delete the owner's folder locally and on the Hub right away."""
        with self._lock:
            count = self.local.delete_owner(owner)
            prefix = owner_digest(owner) + "/"
            self._dirty = {path for path in self._dirty if not path.startswith(prefix)}
            if any(path.startswith(prefix) for path in self._remote_paths):
                self._delete_remote([prefix])
            return count

    def purge_expired(self) -> list[ObjectMeta]:
        """Purge locally, then delete the purged files on the Hub in one commit."""
        with self._lock:
            purged = self.local.purge_expired()
            paths = [path for meta in purged for path in meta.paths]
            self._dirty.difference_update(paths)
            self._delete_remote([path for path in paths if path in self._remote_paths])
            return purged
