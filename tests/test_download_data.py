"""Tests for scripts/download_data.py that need no network access."""

import zipfile
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


def test_missing_kaggle_credentials_raise_dataset_skipped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in download_data.KAGGLE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(download_data.DatasetSkipped):
        download_data.authenticated_kaggle_api()


def test_empty_kaggle_env_vars_are_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "")
    monkeypatch.setenv("KAGGLE_KEY", "  ")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "real-looking-token")
    download_data.drop_empty_kaggle_env_vars()
    assert "KAGGLE_USERNAME" not in download_data.os.environ
    assert "KAGGLE_KEY" not in download_data.os.environ
    assert download_data.os.environ["KAGGLE_API_TOKEN"] == "real-looking-token"


def test_partial_or_corrupt_lending_club_zip_is_not_complete(tmp_path: Path) -> None:
    zip_path = tmp_path / "lending-club.zip"
    assert not download_data.lending_club_zip_is_complete(zip_path)

    zip_path.write_bytes(b"not a zip file")
    assert not download_data.lending_club_zip_is_complete(zip_path)

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("accepted.csv", "id\n1\n")
    assert download_data.lending_club_zip_is_complete(zip_path)

    (tmp_path / "lending-club.zip.kaggle-partial").write_text("resume", encoding="utf-8")
    assert not download_data.lending_club_zip_is_complete(zip_path)


def test_download_url_skips_existing_file(tmp_path: Path) -> None:
    existing = tmp_path / "already_here.csv"
    existing.write_text("x\n", encoding="utf-8")
    # The URL is never fetched because the file exists and force is False.
    action = download_data.download_url("https://invalid.example/never-fetched", existing, force=False)
    assert action == "verified_existing"
    assert existing.read_text(encoding="utf-8") == "x\n"
