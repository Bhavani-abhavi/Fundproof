"""Standard folder locations used across the project.

By default the repo root is found from this file's location, which works for the normal editable install
(`uv sync`). Set UWCOPILOT_ROOT to point somewhere else, for example inside a container image where the
package is installed without the repo around it.
"""

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Return UWCOPILOT_ROOT if it is set, otherwise the folder three levels above this package."""
    override = os.getenv("UWCOPILOT_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = find_repo_root()
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RESULTS_DIR: Path = REPO_ROOT / "results"
DOCS_DIR: Path = REPO_ROOT / "docs"
