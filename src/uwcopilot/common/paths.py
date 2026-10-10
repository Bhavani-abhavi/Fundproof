"""Standard folder locations used across the project."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RESULTS_DIR: Path = REPO_ROOT / "results"
DOCS_DIR: Path = REPO_ROOT / "docs"
