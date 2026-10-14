"""Writing small private files safely (shared by the packet store and the audit log)."""

import os
from pathlib import Path


def write_private_file(path: Path, data: bytes) -> None:
    """Write data to path (mode 0600) via a temporary file and a rename, so a crash never leaves half a file."""
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
