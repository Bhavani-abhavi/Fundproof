"""Load metro configs from config/studio/metros/*.yaml.

    metro = load_metro("houston_tx")
    ids = list_metros()

Adding a metro is adding one YAML file here and running the coverage report (docs/studio/walkthrough_s1.md).
"""

from pathlib import Path

import yaml

from uwcopilot.common.paths import CONFIG_DIR
from uwcopilot.studio.metros.models import MetroConfig

METROS_DIR: Path = CONFIG_DIR / "studio" / "metros"


class MetroNotFound(LookupError):
    """No YAML file exists for the requested metro id."""


def list_metros(metros_dir: Path = METROS_DIR) -> list[str]:
    """Every metro id with a YAML file, sorted."""
    return sorted(path.stem for path in metros_dir.glob("*.yaml"))


def load_metro(metro_id: str, metros_dir: Path = METROS_DIR) -> MetroConfig:
    """Load and validate one metro config; the id inside the file must match the file name."""
    path = metros_dir / f"{metro_id}.yaml"
    if not path.is_file():
        raise MetroNotFound(f"no metro {metro_id!r}; known: {list_metros(metros_dir)}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must hold a YAML mapping")
    metro = MetroConfig.model_validate(data)
    if metro.id != metro_id:
        raise ValueError(f"{path.name} declares id {metro.id!r}; the id must match the file name")
    return metro


def load_all_metros(metros_dir: Path = METROS_DIR) -> dict[str, MetroConfig]:
    """{metro id: MetroConfig} for every YAML file."""
    return {metro_id: load_metro(metro_id, metros_dir) for metro_id in list_metros(metros_dir)}
