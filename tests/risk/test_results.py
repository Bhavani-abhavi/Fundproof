"""Tests for uwcopilot.risk.results: sections merge, metadata is stamped, NaN never reaches the JSON."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uwcopilot.risk.results import (
    NOT_MEASURED,
    dataset_checksums,
    git_commit,
    is_not_measured,
    not_measured,
    package_versions,
    read_results,
    to_jsonable,
    update_results,
)


@pytest.fixture
def checksum_log(tmp_path: Path) -> Path:
    """A synthetic checksum log: two records for the same file (the second is newer) and one other file."""
    records = [
        {"dataset": "fremtpl2", "file": "freq.csv", "sha256": "old", "size_bytes": 1, "source": "s", "action": "a"},
        {"dataset": "german_credit", "file": "german.data", "sha256": "g1", "size_bytes": 2, "source": "s"},
        {"dataset": "fremtpl2", "file": "freq.csv", "sha256": "new", "size_bytes": 1, "source": "s", "action": "b"},
        {"dataset": "fremtpl2", "file": "sev.csv", "sha256": "sev1", "size_bytes": 3, "source": "s"},
    ]
    path = tmp_path / "checksums.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def test_not_measured_helper() -> None:
    entry = not_measured("Lending Club data not downloaded (no Kaggle key)")
    assert entry == {"status": NOT_MEASURED, "reason": "Lending Club data not downloaded (no Kaggle key)"}
    assert NOT_MEASURED == "not measured"
    assert is_not_measured(entry)
    assert not is_not_measured({"estimate": 0.7})
    with pytest.raises(ValueError):
        not_measured("  ")


def test_to_jsonable_converts_numpy_pandas_and_nan() -> None:
    payload = {
        "int": np.int64(3),
        "float": np.float32(0.5),
        "nan": float("nan"),
        "array": np.array([1.0, np.inf]),
        "table": pd.DataFrame({"bin": [1, 2], "rate": [0.1, 0.2]}),
        "path": Path("results/x.json"),
        "nested": {"flag": np.bool_(True)},
    }
    converted = to_jsonable(payload)
    assert converted == {
        "int": 3,
        "float": 0.5,
        "nan": None,
        "array": [1.0, None],
        "table": [{"bin": 1, "rate": 0.1}, {"bin": 2, "rate": 0.2}],
        "path": "results/x.json",
        "nested": {"flag": True},
    }
    json.dumps(converted, allow_nan=False)


def test_dataset_checksums_keeps_latest_record_per_file(checksum_log: Path) -> None:
    result = dataset_checksums(["fremtpl2", "lending_club"], checksum_log)
    files = {entry["file"]: entry["sha256"] for entry in result["fremtpl2"]["files"]}
    assert files == {"freq.csv": "new", "sev.csv": "sev1"}
    assert result["lending_club"]["files"] == []
    assert "no checksum record" in result["lending_club"]["note"]


def test_update_results_creates_file_and_stamps_metadata(tmp_path: Path, checksum_log: Path) -> None:
    path = tmp_path / "results" / "risk_metrics.json"
    update_results(path, "insurance_frequency", {"poisson_deviance": np.float64(0.31)}, ["fremtpl2"], checksum_log)
    contents = json.loads(path.read_text(encoding="utf-8"))
    section = contents["insurance_frequency"]
    assert section["results"] == {"poisson_deviance": 0.31}
    metadata = section["metadata"]
    assert set(metadata) >= {"git_commit", "generated_at_utc", "python_version", "package_versions", "datasets"}
    assert metadata["python_version"].startswith("3.")
    assert "numpy" in metadata["package_versions"]
    assert {entry["file"] for entry in metadata["datasets"]["fremtpl2"]["files"]} == {"freq.csv", "sev.csv"}


def test_update_results_merges_sections(tmp_path: Path, checksum_log: Path) -> None:
    path = tmp_path / "risk_metrics.json"
    update_results(path, "insurance_frequency", {"gini": 0.3}, checksum_log=checksum_log)
    update_results(path, "credit", not_measured("no data"), checksum_log=checksum_log)
    update_results(path, "insurance_frequency", {"gini": 0.35}, checksum_log=checksum_log)
    contents = read_results(path)
    assert set(contents) == {"insurance_frequency", "credit"}
    assert contents["insurance_frequency"]["results"] == {"gini": 0.35}
    assert is_not_measured(contents["credit"]["results"])
    assert not (tmp_path / "risk_metrics.json.tmp").exists()


def test_read_results_missing_file(tmp_path: Path) -> None:
    assert read_results(tmp_path / "missing.json") == {}


def test_git_commit_outside_a_repository_is_unknown(tmp_path: Path) -> None:
    assert git_commit(tmp_path) == "unknown"


def test_package_versions_reports_missing_packages() -> None:
    versions = package_versions(["numpy", "surely-not-an-installed-package"])
    assert versions["numpy"] == np.__version__
    assert versions["surely-not-an-installed-package"] == "not installed"
