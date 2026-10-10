"""Tests for scripts/download_data.py that need no network access."""

from pathlib import Path

import pytest

import download_data


def test_every_dataset_has_a_downloader() -> None:
    expected = {"lending_club", "fremtpl2", "german_credit", "ofac_sdn", "opensanctions"}
    assert set(download_data.DOWNLOADERS) == expected


def test_parse_args_defaults_to_all_datasets() -> None:
    args = download_data.parse_args([])
    assert set(args.datasets) == set(download_data.DOWNLOADERS)
    assert args.force is False


def test_lending_club_skips_cleanly_without_kaggle_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    with pytest.raises(download_data.DatasetSkipped):
        download_data.download_lending_club(force=False)


def test_download_url_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "already_here.csv"
    existing.write_text("x\n", encoding="utf-8")
    # The URL is never fetched because the file exists and force is False.
    result = download_data.download_url("https://invalid.example/never-fetched", existing, force=False)
    assert result == existing
    assert existing.read_text(encoding="utf-8") == "x\n"
