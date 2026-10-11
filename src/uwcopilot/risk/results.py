"""Write metrics to results/*.json with provenance, so every reported number can be traced (ground rule 1).

Each results file is a JSON object with one entry per section, for example:

    {
      "credit": {
        "results": {... whatever the script passed ...},
        "metadata": {"git_commit": ..., "generated_at_utc": ..., "python_version": ...,
                     "package_versions": {...}, "datasets": {...}}
      },
      "insurance_frequency": {...}
    }

update_results replaces one section and leaves the others untouched, so several scripts can write to the
same file. A run that was not done is recorded with not_measured(reason), never with a made-up number.
"""

import fcntl
import json
import math
import os
import platform
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uwcopilot.common.paths import RAW_DATA_DIR, REPO_ROOT

NOT_MEASURED = "not measured"
CHECKSUM_LOG = RAW_DATA_DIR / "checksums.jsonl"
KEY_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "statsmodels",
    "shap",
    "mlflow",
)


def not_measured(reason: str) -> dict[str, str]:
    """Placeholder for a metric that was not computed, with the reason. Example: missing dataset."""
    if not reason.strip():
        raise ValueError("a not-measured entry must say why")
    return {"status": NOT_MEASURED, "reason": reason}


def is_not_measured(value: Any) -> bool:
    """True if value is a placeholder made by not_measured()."""
    return isinstance(value, dict) and value.get("status") == NOT_MEASURED


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas / Path values to plain JSON types.

    NaN and infinity become None (JSON has no NaN). DataFrames become a list of row dicts.
    """
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series | np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Current git commit hash, or 'unknown' if git or the repository is not available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def git_worktree_dirty(repo_root: Path = REPO_ROOT) -> bool | str:
    """True if there are uncommitted changes (so the commit hash alone does not pin the code), else False.

    Changes under results/ are ignored, because writing results is exactly what the scripts do.
    Returns 'unknown' if git is not available.
    """
    command = ["git", "status", "--porcelain", "--", ".", ":(exclude)results"]
    try:
        completed = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return bool(completed.stdout.strip())


def package_versions(packages: Iterable[str] = KEY_PACKAGES) -> dict[str, str]:
    """Installed version of each package, or 'not installed'."""
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def dataset_checksums(datasets: Iterable[str], checksum_log: Path = CHECKSUM_LOG) -> dict[str, dict[str, Any]]:
    """Latest checksum record per file for each named dataset, read from data/raw/checksums.jsonl.

    The log is append-only, so the last line for a file is its latest record. A dataset with no records gets
    an empty file list and a note saying so.
    """
    records: list[dict[str, Any]] = []
    if checksum_log.exists():
        for line in checksum_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    result: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        latest_by_file: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.get("dataset") == dataset:
                latest_by_file[record["file"]] = record
        files = [
            {
                "file": record["file"],
                "sha256": record.get("sha256"),
                "size_bytes": record.get("size_bytes"),
                "source": record.get("source"),
                "hashed_at_utc": record.get("hashed_at_utc"),
            }
            for record in latest_by_file.values()
        ]
        entry: dict[str, Any] = {"files": files}
        if not files:
            entry["note"] = f"no checksum record found in {checksum_log.name}"
        result[dataset] = entry
    return result


def build_metadata(
    datasets: Iterable[str] = (), checksum_log: Path = CHECKSUM_LOG, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Provenance stamp for one results section: code version, time, environment and input data checksums."""
    return {
        "git_commit": git_commit(repo_root),
        "git_worktree_dirty": git_worktree_dirty(repo_root),
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "package_versions": package_versions(),
        "datasets": dataset_checksums(datasets, checksum_log),
    }


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on path + '.lock' so two scripts never update the same results file at once."""
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def read_results(path: Path) -> dict[str, Any]:
    """Read a results JSON file, or return {} if it does not exist yet."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_results(
    path: Path,
    section: str,
    payload: dict[str, Any],
    datasets: Iterable[str] = (),
    checksum_log: Path = CHECKSUM_LOG,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Replace one section of a results JSON file (creating the file if needed) and stamp its metadata.

    payload is converted to plain JSON types (NaN becomes null). datasets names the datasets whose checksum
    records (from checksum_log) are copied into the metadata, for example ["fremtpl2"].
    The file is written to a temporary file first and then renamed, so a crash never leaves half a file.
    Returns the full contents of the file after the update.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    section_entry = {
        "results": to_jsonable(payload),
        "metadata": to_jsonable(build_metadata(datasets, checksum_log, repo_root)),
    }
    with _exclusive_lock(path):
        contents = read_results(path)
        contents[section] = section_entry
        temporary_path = path.with_name(path.name + ".tmp")
        temporary_path.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary_path, path)
    return contents
