"""Download every public dataset used by the project into data/raw/.

Nothing downloaded here is committed to git (data/ is gitignored).
Each downloaded file is hashed (SHA-256) and appended to data/raw/checksums.jsonl,
and its size is printed.

Usage:
    python scripts/download_data.py                      # all datasets
    python scripts/download_data.py --datasets ofac_sdn german_credit
    python scripts/download_data.py --force              # re-download existing files

Datasets and licenses:
    lending_club   Kaggle "wordsforthewise/lending-club" (needs a Kaggle API token or
                   KAGGLE_USERNAME / KAGGLE_KEY; see the Kaggle dataset page for its terms)
    fremtpl2       OpenML 41214 (freMTPL2freq) and 41215 (freMTPL2sev); OpenML lists CC0
    german_credit  UCI Statlog (German Credit Data), CC BY 4.0
    ofac_sdn       US Treasury OFAC SDN list (sdn.csv + alt.csv aliases), US government public data
    opensanctions  OpenSanctions "default" dataset, simplified CSV, CC BY-NC 4.0
                   (attribution: "Data from OpenSanctions.org")
"""

import argparse
import contextlib
import io
import os
import sys
import zipfile
from pathlib import Path

import requests
from dotenv import load_dotenv

from uwcopilot.common.files import append_checksum_log, human_size
from uwcopilot.common.paths import RAW_DATA_DIR, REPO_ROOT

CHECKSUM_LOG = RAW_DATA_DIR / "checksums.jsonl"
KAGGLE_ENV_VARS = ["KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR"]
KAGGLE_HELP = (
    "Kaggle credentials not found or not valid. Either generate an API token at "
    "https://www.kaggle.com/settings/api and set KAGGLE_API_TOKEN, or create a legacy API key "
    "(kaggle.json) and set KAGGLE_USERNAME and KAGGLE_KEY. Put them in .env (see .env.example)."
)
HTTP_TIMEOUT_SECONDS = 120
STREAM_CHUNK_BYTES = 1024 * 1024

LENDING_CLUB_KAGGLE_SLUG = "wordsforthewise/lending-club"
FREMTPL2_OPENML_IDS = {"freMTPL2freq": 41214, "freMTPL2sev": 41215}
GERMAN_CREDIT_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german"
GERMAN_CREDIT_FILES = ["german.data", "german.data-numeric", "german.doc"]
OFAC_URLS = {
    "sdn.csv": "https://www.treasury.gov/ofac/downloads/sdn.csv",
    "alt.csv": "https://www.treasury.gov/ofac/downloads/alt.csv",
}
OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/default/targets.simple.csv"


class DatasetSkipped(Exception):
    """Raised when a dataset cannot be downloaded for a known, user-fixable reason."""


def report_file(dataset: str, path: Path, source: str, action: str) -> None:
    """Log the checksum of a file and print its size.

    action is "downloaded" for a fresh download or "verified_existing" for a file that was already present.
    """
    record = append_checksum_log(CHECKSUM_LOG, dataset, path, source, action)
    size = human_size(record["size_bytes"])
    print(f"  {path.relative_to(REPO_ROOT)}  {size}  sha256={record['sha256'][:16]}...  ({action})")


def download_url(url: str, destination: Path, force: bool) -> str:
    """Stream a URL to a file, writing to a .part file first so an interrupted download never looks complete.

    Skips the download if the file exists and force is False.
    Returns "downloaded" or "verified_existing".
    """
    if destination.exists() and not force:
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                handle.write(chunk)
    partial.replace(destination)
    return "downloaded"


def drop_empty_kaggle_env_vars() -> None:
    """Remove KAGGLE_* variables that are set but empty.

    The kaggle package lets any KAGGLE_* environment variable override kaggle.json, even an empty one,
    so an untouched line from .env.example would otherwise hide valid file-based credentials.
    """
    for name in KAGGLE_ENV_VARS:
        if name in os.environ and not os.environ[name].strip():
            del os.environ[name]


def authenticated_kaggle_api():
    """Return an authenticated KaggleApi, or raise DatasetSkipped with setup instructions.

    The kaggle package itself decides which credential sources are valid (API token, kaggle.json in
    ~/.kaggle or ~/.config/kaggle, KAGGLE_CONFIG_DIR, or environment variables).
    """
    # Imported here because importing kaggle runs an authentication attempt as a side effect
    # (it prints a login banner and creates a config folder when no key is set).
    # The banner is captured so the script prints one clear message instead.
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        try:
            api.authenticate()
        except (Exception, SystemExit) as error:  # noqa: BLE001 - kaggle exits on missing credentials
            raise DatasetSkipped(KAGGLE_HELP) from error
    return api


def lending_club_zip_is_complete(zip_path: Path) -> bool:
    """True if the zip exists, has no kaggle resume marker next to it, and is a readable zip file."""
    resume_marker = zip_path.with_name(zip_path.name + ".kaggle-partial")
    return zip_path.exists() and not resume_marker.exists() and zipfile.is_zipfile(zip_path)


def download_lending_club(force: bool) -> None:
    """Download the Lending Club loans dump (2007-2018; accepted and rejected files) from Kaggle."""
    target_dir = RAW_DATA_DIR / "lending_club"
    zip_path = target_dir / f"{LENDING_CLUB_KAGGLE_SLUG.split('/')[1]}.zip"
    source = f"https://www.kaggle.com/datasets/{LENDING_CLUB_KAGGLE_SLUG}"
    if lending_club_zip_is_complete(zip_path) and not force:
        report_file("lending_club", zip_path, source, "verified_existing")
        return
    api = authenticated_kaggle_api()
    target_dir.mkdir(parents=True, exist_ok=True)
    # kaggle resumes a partial download on its own and re-downloads when force is True.
    api.dataset_download_files(LENDING_CLUB_KAGGLE_SLUG, path=str(target_dir), force=force, unzip=False, quiet=False)
    if not lending_club_zip_is_complete(zip_path):
        raise RuntimeError(f"Kaggle download did not produce a complete zip at {zip_path}")
    report_file("lending_club", zip_path, source, "downloaded")


def download_fremtpl2(force: bool) -> None:
    """Download freMTPL2freq and freMTPL2sev from OpenML and save them as CSV."""
    from sklearn.datasets import fetch_openml

    target_dir = RAW_DATA_DIR / "fremtpl2"
    target_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = target_dir / "openml_cache"
    for name, data_id in FREMTPL2_OPENML_IDS.items():
        csv_path = target_dir / f"{name}.csv"
        source = f"https://www.openml.org/d/{data_id}"
        action = "verified_existing"
        if force or not csv_path.exists():
            bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(cache_dir))
            bunch.frame.to_csv(csv_path, index=False)
            action = "downloaded"
        # The CSV is written by pandas, so its hash can change with a library upgrade.
        # The cached OpenML file is the upstream original, so log it as well.
        for upstream_file in sorted(cache_dir.rglob(f"{name}.arff.gz")):
            report_file("fremtpl2", upstream_file, source, action)
        report_file("fremtpl2", csv_path, source, action)


def download_german_credit(force: bool) -> None:
    """Download the UCI Statlog German Credit files."""
    target_dir = RAW_DATA_DIR / "german_credit"
    for filename in GERMAN_CREDIT_FILES:
        url = f"{GERMAN_CREDIT_BASE_URL}/{filename}"
        path = target_dir / filename
        action = download_url(url, path, force)
        report_file("german_credit", path, url, action)


def download_ofac(force: bool) -> None:
    """Download the OFAC SDN list and its alias file."""
    target_dir = RAW_DATA_DIR / "ofac"
    for filename, url in OFAC_URLS.items():
        path = target_dir / filename
        action = download_url(url, path, force)
        report_file("ofac_sdn", path, url, action)


def download_opensanctions(force: bool) -> None:
    """Download the OpenSanctions default dataset (simplified CSV, CC BY-NC 4.0)."""
    target_dir = RAW_DATA_DIR / "opensanctions"
    path = target_dir / "targets.simple.csv"
    action = download_url(OPENSANCTIONS_URL, path, force)
    report_file("opensanctions", path, OPENSANCTIONS_URL, action)


DOWNLOADERS = {
    "lending_club": download_lending_club,
    "fremtpl2": download_fremtpl2,
    "german_credit": download_german_credit,
    "ofac_sdn": download_ofac,
    "opensanctions": download_opensanctions,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DOWNLOADERS),
        default=list(DOWNLOADERS),
        help="Which datasets to download (default: all).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Download the requested datasets. Returns a non-zero exit code if any dataset failed or was skipped."""
    load_dotenv(REPO_ROOT / ".env")
    drop_empty_kaggle_env_vars()
    args = parse_args(argv)
    problems: dict[str, str] = {}
    for name in args.datasets:
        print(f"[{name}]")
        try:
            DOWNLOADERS[name](args.force)
        except DatasetSkipped as reason:
            problems[name] = f"skipped: {reason}"
            print(f"  SKIPPED: {reason}")
        except Exception as error:  # noqa: BLE001 - report every failure and keep going
            problems[name] = f"failed: {error}"
            print(f"  FAILED: {error}")
    print()
    print(f"Checksum log: {CHECKSUM_LOG.relative_to(REPO_ROOT)}")
    if problems:
        print("Datasets not downloaded:")
        for name, message in problems.items():
            print(f"  - {name}: {message}")
        return 1
    print("All requested datasets downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
