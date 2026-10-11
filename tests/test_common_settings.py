"""Tests for uwcopilot.common.settings, and that .env.example documents every setting the code reads."""

import re
from pathlib import Path

import pytest

from uwcopilot.common.paths import REPO_ROOT
from uwcopilot.common.settings import env_reader, is_http_url

SETTING_NAME = re.compile(r"[\"'](UWC_[A-Z0-9_]+)[\"']")


def test_env_file_wins_and_blank_values_count_as_unset(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A=from-file\nB=   \n", encoding="utf-8")
    read = env_reader(env_file, {"A": "from-env", "B": "env-b", "C": " c "})
    assert read("A") == "from-file"
    assert read("B") == "env-b"
    assert read("C") == "c"
    assert read("D") is None


def test_missing_env_file_and_none_are_skipped(tmp_path: Path) -> None:
    assert env_reader(tmp_path / "absent.env", {"A": "x"})("A") == "x"
    assert env_reader(None, {"A": "x"})("A") == "x"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:8000", True),
        ("https://risk.example.test/v1", True),
        ("http://[::1]:8000", True),
        ("localhost:8000", False),
        ("ftp://host", False),
        ("http:///path", False),
        ("", False),
    ],
)
def test_is_http_url(url: str, expected: bool) -> None:
    assert is_http_url(url) is expected


def test_env_example_documents_every_project_setting() -> None:
    documented = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    read_by_code: set[str] = set()
    for folder in ("src", "scripts"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            read_by_code.update(SETTING_NAME.findall(path.read_text(encoding="utf-8")))
    assert read_by_code, "the scan found no settings at all"
    missing = sorted(name for name in read_by_code if name not in documented)
    assert not missing, f".env.example does not mention: {missing}"
