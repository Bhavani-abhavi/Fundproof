"""Tests for uwcopilot.common.files."""

import hashlib
import json
from pathlib import Path

from uwcopilot.common.files import append_checksum_log, human_size, sha256_of_file


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_bytes(b"underwriting copilot")
    assert sha256_of_file(sample) == hashlib.sha256(b"underwriting copilot").hexdigest()


def test_human_size_units() -> None:
    assert human_size(512) == "512 B"
    assert human_size(2048) == "2.0 KB"
    assert human_size(5 * 1024 * 1024) == "5.0 MB"
    assert human_size(3 * 1024**3) == "3.0 GB"


def test_append_checksum_log_writes_one_json_line(tmp_path: Path) -> None:
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")
    log_path = tmp_path / "logs" / "checksums.jsonl"

    record = append_checksum_log(log_path, "demo", sample, "https://example.org/sample.csv")
    append_checksum_log(log_path, "demo", sample, "https://example.org/sample.csv")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["sha256"] == record["sha256"]
    assert record["size_bytes"] == sample.stat().st_size
    assert record["dataset"] == "demo"
