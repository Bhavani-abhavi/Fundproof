"""Small file helpers: checksums, human-readable sizes, and a checksum log."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

CHUNK_SIZE_BYTES = 1024 * 1024


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def human_size(num_bytes: int) -> str:
    """Format a byte count as a short string such as '5.5 MB'."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size = size / 1024
    return f"{size:.1f} GB"


def append_checksum_log(log_path: Path, dataset: str, file_path: Path, source_url: str, action: str) -> dict:
    """Hash a data file and append one JSON line describing it to the log.

    action says what happened to the file in this run, for example "downloaded" or "verified_existing".
    Returns the record that was written.
    """
    record = {
        "dataset": dataset,
        "file": file_path.name,
        "source": source_url,
        "action": action,
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_of_file(file_path),
        "hashed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    return record
