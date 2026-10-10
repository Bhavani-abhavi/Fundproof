"""Download every public dataset used by the project into data/raw/.

Nothing downloaded here is committed to git (data/ is gitignored).
Each downloaded file is hashed (SHA-256) and appended to data/raw/checksums.jsonl,
and its size is printed.

Usage:
    python scripts/download_data.py                      # all datasets
    python scripts/download_data.py --datasets ofac_sdn german_credit
    python scripts/download_data.py --force              # re-download existing files

Datasets and licenses:
    lending_club   Kaggle "wordsforthewise/lending-club" (needs KAGGLE_USERNAME / KAGGLE_KEY;
                   see the Kaggle dataset page for its terms)
    fremtpl2       OpenML 41214 (freMTPL2freq) and 41215 (freMTPL2sev); OpenML lists CC0
    german_credit  UCI Statlog (German Credit Data), CC BY 4.0
    ofac_sdn       US Treasury OFAC SDN list (sdn.csv + alt.csv aliases), US government public data
    opensanctions  OpenSanctions "default" dataset, simplified CSV, CC BY-NC 4.0
                   (attribution: "Data from OpenSanctions.org")
"""

import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from uwcopilot.common.files import append_checksum_log, human_size
from uwcopilot.common.paths import RAW_DATA_DIR, REPO_ROOT

CHECKSUM_LOG = RAW_DATA_DIR / "checksums.jsonl"
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


def report_file(dataset: str, path: Path, source: str) -> None:
    """Log the checksum of a file and print its size."""
    record = append_checksum_log(CHECKSUM_LOG, dataset, path, source)
    size = human_size(record["size_bytes"])
    print(f"  {path.relative_to(REPO_ROOT)}  {size}  sha256={record['sha256'][:16]}...")


def download_url(url: str, destination: Path, force: bool) -> Path:
    """Stream a URL to a file. Skips the download if the file exists and force is False."""
    if destination.exists() and not force:
        print(f"  already present, skipping download: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=HTTP_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                handle.write(chunk)
    partial.replace(destination)
    return destination


def kaggle_credentials_available() -> bool:
    """True if Kaggle credentials are set in the environment or in ~/.kaggle/kaggle.json."""
    has_env = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env or has_file


def download_lending_club(force: bool) -> None:
    """Download the Lending Club accepted/rejected loans dump (2007-2018) from Kaggle."""
    if not kaggle_credentials_available():
        raise DatasetSkipped(
            "Kaggle credentials not found. Create an API token at https://www.kaggle.com/settings, "
            "then put KAGGLE_USERNAME and KAGGLE_KEY in .env (see .env.example)."
        )
    # Imported here because the kaggle package tries to authenticate on import.
    from kaggle.api.kaggle_api_extended import KaggleApi

    target_dir = RAW_DATA_DIR / "lending_club"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_zips = list(target_dir.glob("*.zip"))
    if existing_zips and not force:
        print("  already present, skipping download")
    else:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(LENDING_CLUB_KAGGLE_SLUG, path=str(target_dir), unzip=False, quiet=False)
    source = f"https://www.kaggle.com/datasets/{LENDING_CLUB_KAGGLE_SLUG}"
    for zip_path in sorted(target_dir.glob("*.zip")):
        report_file("lending_club", zip_path, source)


def download_fremtpl2(force: bool) -> None:
    """Download freMTPL2freq and freMTPL2sev from OpenML and save them as CSV."""
    from sklearn.datasets import fetch_openml

    target_dir = RAW_DATA_DIR / "fremtpl2"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name, data_id in FREMTPL2_OPENML_IDS.items():
        csv_path = target_dir / f"{name}.csv"
        if csv_path.exists() and not force:
            print(f"  already present, skipping download: {csv_path.name}")
        else:
            bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(target_dir / "openml_cache"))
            bunch.frame.to_csv(csv_path, index=False)
        report_file("fremtpl2", csv_path, f"https://www.openml.org/d/{data_id}")


def download_german_credit(force: bool) -> None:
    """Download the UCI Statlog German Credit files."""
    target_dir = RAW_DATA_DIR / "german_credit"
    for filename in GERMAN_CREDIT_FILES:
        url = f"{GERMAN_CREDIT_BASE_URL}/{filename}"
        path = download_url(url, target_dir / filename, force)
        report_file("german_credit", path, url)


def download_ofac(force: bool) -> None:
    """Download the OFAC SDN list and its alias file."""
    target_dir = RAW_DATA_DIR / "ofac"
    for filename, url in OFAC_URLS.items():
        path = download_url(url, target_dir / filename, force)
        report_file("ofac_sdn", path, url)


def download_opensanctions(force: bool) -> None:
    """Download the OpenSanctions default dataset (simplified CSV, CC BY-NC 4.0)."""
    target_dir = RAW_DATA_DIR / "opensanctions"
    path = download_url(OPENSANCTIONS_URL, target_dir / "targets.simple.csv", force)
    report_file("opensanctions", path, OPENSANCTIONS_URL)


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
