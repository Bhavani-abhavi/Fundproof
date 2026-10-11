"""Tests for uwcopilot.risk.tracking, using a throwaway SQLite store in a temporary folder."""

from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression

from uwcopilot.risk.tracking import (
    configure_tracking,
    flatten,
    get_or_create_experiment,
    log_artifact,
    log_metrics,
    log_params,
    register_model,
    start_run,
    tracking_uri_for,
)


@pytest.fixture
def store(tmp_path: Path):
    """Paths for a temporary tracking database and artifact folder; restores the previous tracking URI after."""
    previous_uri = mlflow.get_tracking_uri()
    yield {"db_path": tmp_path / "mlruns.db", "artifact_dir": tmp_path / "mlruns"}
    mlflow.set_tracking_uri(previous_uri)


def test_tracking_uri_is_absolute_sqlite(tmp_path: Path) -> None:
    uri = tracking_uri_for(tmp_path / "mlruns.db")
    assert uri == f"sqlite:///{(tmp_path / 'mlruns.db').resolve()}"


def test_flatten_nested_dict() -> None:
    assert flatten({"auc": {"estimate": 0.7, "ci_low": 0.6}, "n": 3}) == {
        "auc.estimate": 0.7,
        "auc.ci_low": 0.6,
        "n": 3,
    }


def test_run_logs_params_metrics_and_artifacts(store: dict, tmp_path: Path) -> None:
    report = tmp_path / "report.txt"
    report.write_text("synthetic test artifact", encoding="utf-8")
    with start_run("unit-test", run_name="demo", tags={"phase": "1"}, **store) as run:
        logged_params = log_params({"model": "lightgbm", "lgbm": {"num_leaves": 31}})
        logged_metrics = log_metrics(
            {
                "auc": {"estimate": np.float64(0.71), "ci_low": 0.69},
                "n_rows": np.int64(1000),
                "credit": {"status": "not measured", "reason": "no data"},
                "flag": True,
                "missing": float("nan"),
            }
        )
        log_artifact(report, artifact_path="reports")
        run_id = run.info.run_id

    assert logged_params == {"model": "lightgbm", "lgbm.num_leaves": "31"}
    assert logged_metrics == {"auc.estimate": 0.71, "auc.ci_low": 0.69, "n_rows": 1000.0}
    stored = MlflowClient().get_run(run_id)
    assert stored.data.params["lgbm.num_leaves"] == "31"
    assert stored.data.metrics["auc.estimate"] == pytest.approx(0.71)
    assert stored.data.tags["phase"] == "1"
    artifact_files = [item.path for item in MlflowClient().list_artifacts(run_id, "reports")]
    assert artifact_files == ["reports/report.txt"]
    assert store["db_path"].exists()
    assert (store["artifact_dir"] / "unit-test").is_dir()


def test_get_or_create_experiment_is_idempotent(store: dict) -> None:
    configure_tracking(store["db_path"])
    first = get_or_create_experiment("same-name", store["artifact_dir"])
    second = get_or_create_experiment("same-name", store["artifact_dir"])
    assert first == second


def test_register_model_with_version_tag(store: dict) -> None:
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(features, target)
    with start_run("registry-test", **store):
        # pip_requirements skips MLflow's slow environment inference; this test only checks registration.
        info = mlflow.sklearn.log_model(model, name="model", pip_requirements=["scikit-learn"])
    version = register_model(info.model_uri, "synthetic-demo-model", tags={"status": "candidate"})
    assert version == "1"
    stored = MlflowClient().get_model_version("synthetic-demo-model", version)
    assert stored.tags["status"] == "candidate"
