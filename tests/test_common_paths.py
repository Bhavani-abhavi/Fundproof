"""Tests for uwcopilot.common.paths."""

from pathlib import Path

import pytest

from uwcopilot.common import paths


def test_repo_root_is_the_project_folder() -> None:
    assert (paths.REPO_ROOT / "pyproject.toml").exists()
    assert (paths.REPO_ROOT / "src" / "uwcopilot").is_dir()


def test_data_folders_hang_off_the_repo_root() -> None:
    assert paths.RAW_DATA_DIR == paths.REPO_ROOT / "data" / "raw"
    assert paths.RESULTS_DIR == paths.REPO_ROOT / "results"


def test_env_override_changes_the_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("UWCOPILOT_ROOT", str(tmp_path))
    assert paths.find_repo_root() == tmp_path.resolve()
