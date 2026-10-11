"""Thin helpers around MLflow experiment tracking, using a local store inside the repo.

Store choice: MLflow 3.17 (the version in uv.lock) refuses the old file store ("file:./mlruns") unless
MLFLOW_ALLOW_FILE_STORE=true is set, because that backend is in maintenance mode. So run metadata goes to
a local SQLite database, mlruns.db at the repo root, and artifacts (model files, plots) go to the mlruns/
folder next to it. Both are gitignored. Nothing is sent to a server.

To browse runs: uv run mlflow ui --backend-store-uri sqlite:///mlruns.db
"""

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from typing import Any

# MLflow sends anonymous usage telemetry unless this is set before it is imported. This project runs offline
# and reproducibly, so telemetry is off unless the user has set the variable themselves.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402  (must come after the telemetry setting above)
from mlflow.tracking import MlflowClient  # noqa: E402

from uwcopilot.common.paths import REPO_ROOT  # noqa: E402

DEFAULT_DB_PATH = REPO_ROOT / "mlruns.db"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "mlruns"


def tracking_uri_for(db_path: Path) -> str:
    """SQLite tracking URI for an absolute database path, for example sqlite:////home/me/repo/mlruns.db."""
    return f"sqlite:///{Path(db_path).resolve()}"


def configure_tracking(db_path: Path = DEFAULT_DB_PATH) -> str:
    """Point MLflow at the local SQLite store and return the tracking URI that was set."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri_for(db_path)
    mlflow.set_tracking_uri(uri)
    return uri


def get_or_create_experiment(name: str, artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> str:
    """Return the id of the named experiment, creating it (with artifacts under artifact_dir) if needed.

    Call configure_tracking first. The experiment also becomes the active one for mlflow.start_run.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        artifact_location = (Path(artifact_dir).resolve() / name).as_uri()
        experiment_id = client.create_experiment(name, artifact_location=artifact_location)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


@contextmanager
def start_run(
    experiment: str,
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> Iterator[mlflow.ActiveRun]:
    """Configure the local store, select the experiment, and open a run. Use as `with start_run(...) as run:`."""
    configure_tracking(db_path)
    experiment_id = get_or_create_experiment(experiment, artifact_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=dict(tags or {})) as run:
        yield run


def flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into one level with dotted keys: {"auc": {"estimate": 0.7}} -> {"auc.estimate": 0.7}."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten(value, full_key))
        else:
            flat[full_key] = value
    return flat


def log_params(params: Mapping[str, Any]) -> dict[str, str]:
    """Log parameters (nested dicts are flattened; values are stored as strings). Returns what was logged."""
    logged = {key: str(value) for key, value in flatten(params).items()}
    mlflow.log_params(logged)
    return logged


def log_metrics(metrics: Mapping[str, Any], step: int | None = None) -> dict[str, float]:
    """Log every finite number in metrics (nested dicts are flattened). Returns what was logged.

    Non-numeric entries (for example a "not measured" status string), booleans, NaN and infinity are
    skipped, because MLflow metrics must be finite numbers.
    """
    logged: dict[str, float] = {}
    for key, value in flatten(metrics).items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        if not math.isfinite(float(value)):
            continue
        logged[key] = float(value)
    mlflow.log_metrics(logged, step=step)
    return logged


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    """Copy a local file (for example a plot or a results JSON) into the active run's artifacts."""
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def register_model(model_uri: str, name: str, tags: Mapping[str, str] | None = None) -> str:
    """Register a logged model under name, add tags to the new version, and return its version number.

    model_uri is the model_uri returned by a log_model call, for example
    mlflow.lightgbm.log_model(model, name="model").model_uri.
    """
    model_version = mlflow.register_model(model_uri, name)
    client = MlflowClient()
    for key, value in (tags or {}).items():
        client.set_model_version_tag(name, model_version.version, key, str(value))
    return str(model_version.version)
