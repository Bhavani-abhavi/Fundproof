"""A Store mirrored to a PRIVATE Hugging Face dataset repo (used in the Space when UWC_STORE=hf_dataset).

HFDatasetStore keeps a LocalFileStore (store.py) as a working copy of a PRIVATE Hugging Face dataset repo:
- at startup restore() creates the repo if missing (private), refuses a public repo, and downloads a snapshot;
- writes go to the local copy and are uploaded in one batched commit every flush_every_s seconds (default 300) by
  a background thread, plus once more at shutdown. Data written less than one interval before a crash is lost;
- the Hub limits how often one repo can be committed to, which is why writes are batched. One lock serializes
  uploads and deletes, so the Hub never receives an upload of a file after its deletion.

Deleting (delete, delete_owner, purge_expired) has three steps, because a dataset repo is a git repo and a plain
delete only adds a commit: every earlier revision would still hold the deleted files.
1. The files are deleted on the Hub at once (HfApi.delete_files, one commit).
2. At the next flush the commit history is squashed into one commit (HfApi.super_squash_history), so no earlier
   revision holds a deleted file. huggingface_hub 1.33.0 documents the squash as non-revertible: "Once squashed,
   the commit history cannot be retrieved." Squashing also drops the old versions of overwritten objects.
3. Then LFS objects that no current file uses are permanently deleted (HfApi.permanently_delete_lfs_files), because
   a squash does not remove LFS objects (large files, for example a big encrypted upload, are stored as LFS).
Steps 2-3 run at most once per flush interval, so a burst of deletions makes one squash. They also run once after
every startup, so a squash that a restart interrupted still happens.

If step 1 fails (the Hub is unreachable), the paths are kept as "pending deletes": in memory and in a small file
next to the working copy (.pending_deletes.json, never uploaded). They are retried at every flush, and restore()
re-applies them after the snapshot download, so an object deleted on this disk does not come back when the
process restarts. delete() and delete_owner() then raise DeletionPending, so the caller does not tell the user the
data is gone. Limit: in a Space the container disk does not survive a restart; if the Hub stays unreachable until
the container restarts, the pending delete is lost and the user has to delete again (they were told it was pending).
"""

# Postponed annotations: HFDatasetStore has a method named `list`, which would shadow the built-in list type inside
# the class body if annotations were evaluated.
from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Iterable
from typing import Any, Protocol

from huggingface_hub import CommitOperationAdd

from uwcopilot.web.store import (
    DeletionPending,
    LocalFileStore,
    ObjectMeta,
    StoreError,
    owner_digest,
    write_atomically,
)

logger = logging.getLogger(__name__)

PENDING_DELETES_FILE = ".pending_deletes.json"


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

    def super_squash_history(self, repo_id: str, *, repo_type: str, commit_message: str) -> Any:
        """Replace the whole commit history of the main branch with one commit of the current files."""
        ...

    def list_repo_tree(self, repo_id: str, *, recursive: bool, repo_type: str) -> Iterable[Any]:
        """The current files; an LFS file has `lfs.sha256`, other entries have `lfs` None or no `lfs`."""
        ...

    def list_lfs_files(self, repo_id: str, *, repo_type: str) -> Iterable[Any]:
        """Every LFS object stored for the repo; each has `file_oid` (its SHA-256)."""
        ...

    def permanently_delete_lfs_files(self, repo_id: str, lfs_files: Iterable[Any], *, repo_type: str) -> Any:
        """Delete LFS objects from storage for good."""
        ...


def is_safe_relative_path(path: str) -> bool:
    """True for a store path such as 'abc/projects/p1.enc' or 'abc/' (no absolute path, no '..')."""
    return bool(path) and not path.startswith("/") and "\\" not in path and ".." not in path.split("/")


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
        # True when deleted data may still sit in the Hub history (see "Deleting" in the module docstring).
        self._history_needs_erasing = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def restore(self) -> None:
        """Create the private repo if needed, refuse a public one, download its files, re-apply pending deletes."""
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = self.api.repo_info(self.repo_id, repo_type="dataset")
        if not getattr(info, "private", False):
            raise StoreError(f"dataset {self.repo_id} is public; user data may only go to a private dataset")
        self.local.root.mkdir(parents=True, exist_ok=True)
        self.api.snapshot_download(self.repo_id, repo_type="dataset", local_dir=str(self.local.root))
        with self._lock:
            self._remote_paths = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
            self._pending_deletes = self._read_pending_deletes()
            for path in self._pending_deletes:
                self._remove_local(path)  # the snapshot brought back files whose deletion never reached the Hub
            if self._pending_deletes:
                self._delete_remote(sorted(self._pending_deletes))
            # A squash that a restart interrupted must still happen; squashing an already squashed history is harmless.
            self._history_needs_erasing = True

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
        if self._pending_deletes:
            logger.error("%d deletion(s) did not reach the dataset before shutdown", len(self._pending_deletes))

    # ---- syncing ----

    def flush(self) -> int:
        """Send pending deletes, upload every changed file in one commit, then erase deleted data from the history.

        Returns the number of files uploaded.
        """
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
            if self._history_needs_erasing:
                self._erase_history()
            return len(operations)

    def _erase_history(self) -> None:
        """Squash the commit history, then permanently delete LFS objects no current file uses.

        Call with the lock held. On failure the work is retried at the next flush.
        """
        try:
            self.api.super_squash_history(
                self.repo_id, repo_type="dataset", commit_message="store: squash history (erases deleted data)"
            )
            tree = self.api.list_repo_tree(self.repo_id, recursive=True, repo_type="dataset")
            in_use: set[str] = set()
            for entry in tree:
                lfs = getattr(entry, "lfs", None)
                if lfs is not None:
                    in_use.add(lfs.sha256)
            stored = self.api.list_lfs_files(self.repo_id, repo_type="dataset")
            unused = [item for item in stored if item.file_oid not in in_use]
            if unused:
                self.api.permanently_delete_lfs_files(self.repo_id, unused, repo_type="dataset")
        except Exception:
            logger.exception("erasing deleted data from the dataset history failed; retrying at the next flush")
            return
        self._history_needs_erasing = False

    def _delete_remote(self, paths: list[str]) -> bool:
        """Delete paths (files or 'folder/') on the Hub now. True on success; on failure keep them pending.

        Call with the lock held.
        """
        if not paths:
            return True
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
            self._write_pending_deletes()
            return False
        self._pending_deletes.difference_update(paths)
        self._write_pending_deletes()
        self._history_needs_erasing = True
        for path in paths:
            if path.endswith("/"):
                self._remote_paths = {remote for remote in self._remote_paths if not remote.startswith(path)}
            else:
                self._remote_paths.discard(path)
        return True

    def _read_pending_deletes(self) -> set[str]:
        """The pending deletes saved next to the working copy (empty when there is no file or it is unreadable)."""
        path = self.local.root / PENDING_DELETES_FILE
        if not path.is_file():
            return set()
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.error("%s is unreadable; pending deletes in it are lost", path)
            return set()
        if not isinstance(saved, list):
            return set()
        return {item for item in saved if isinstance(item, str) and is_safe_relative_path(item)}

    def _write_pending_deletes(self) -> None:
        """Save the pending deletes next to the working copy (remove the file when there are none)."""
        path = self.local.root / PENDING_DELETES_FILE
        if not self._pending_deletes:
            path.unlink(missing_ok=True)
            return
        write_atomically(path, json.dumps(sorted(self._pending_deletes)).encode("utf-8"))

    def _remove_local(self, path: str) -> None:
        """Remove a file, or a folder for a path ending in '/', from the working copy (missing is fine)."""
        if not is_safe_relative_path(path):
            return
        target = self.local.root / path
        if path.endswith("/"):
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)

    def _mark_dirty(self, owner: str, namespace: str, key: str) -> None:
        """Remember an object's files for the next upload (call with the lock held)."""
        paths = self.local.relative_paths(owner, namespace, key)
        self._dirty.update(paths)
        if self._pending_deletes.intersection(paths):
            self._pending_deletes.difference_update(paths)
            self._write_pending_deletes()

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
        """Delete locally and on the Hub right away; DeletionPending if the Hub could not be updated."""
        with self._lock:
            found = self.local.delete(owner, namespace, key)
            paths = self.local.relative_paths(owner, namespace, key)
            self._dirty.difference_update(paths)
            remote = [path for path in paths if path in self._remote_paths]
            if not self._delete_remote(remote):
                raise DeletionPending(f"{namespace}/{key}: deleted here, the dataset copy is deleted at a retry")
            return found

    def delete_owner(self, owner: str) -> int:
        """Delete the owner's folder locally and on the Hub right away; DeletionPending as in delete()."""
        with self._lock:
            count = self.local.delete_owner(owner)
            prefix = owner_digest(owner) + "/"
            self._dirty = {path for path in self._dirty if not path.startswith(prefix)}
            if any(path.startswith(prefix) for path in self._remote_paths) and not self._delete_remote([prefix]):
                raise DeletionPending("the owner's data is deleted here; the dataset copy is deleted at a retry")
            return count

    def purge_expired(self) -> list[ObjectMeta]:
        """Purge locally, then delete the purged files on the Hub in one commit (a failure is retried at a flush)."""
        with self._lock:
            purged = self.local.purge_expired()
            paths = [path for meta in purged for path in meta.paths]
            self._dirty.difference_update(paths)
            self._delete_remote([path for path in paths if path in self._remote_paths])
            return purged
