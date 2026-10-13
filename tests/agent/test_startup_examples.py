"""The committed SYNTHETIC startup examples (eval/startup_applications/) and their generator script."""

import json
from pathlib import Path

import make_startup_applications
from uwcopilot.agent.applications import STARTUP_BUSINESS_TYPES, StartupCreditApplication, load_applications
from uwcopilot.agent.evaluation import evaluate_policy_case
from uwcopilot.common.paths import REPO_ROOT

STARTUP_DIR = REPO_ROOT / "eval" / "startup_applications"


def test_committed_examples_are_reproduced_byte_for_byte(tmp_path: Path) -> None:
    assert make_startup_applications.main(["--out-dir", str(tmp_path)]) == 0
    committed = sorted(path.name for path in STARTUP_DIR.iterdir())
    assert sorted(path.name for path in tmp_path.iterdir()) == committed
    for name in committed:
        assert (tmp_path / name).read_bytes() == (STARTUP_DIR / name).read_bytes(), name


def test_one_synthetic_example_per_business_type() -> None:
    applications = load_applications(STARTUP_DIR)
    assert [application.application_id for application in applications] == [f"APP-S-00{n}" for n in range(1, 7)]
    assert all(isinstance(application, StartupCreditApplication) for application in applications)
    assert all(application.data_label == "SYNTHETIC" for application in applications)
    assert sorted(application.applicant.business_type for application in applications) == sorted(STARTUP_BUSINESS_TYPES)
    assert all(application.projections.simulation is None for application in applications)
    manifest = json.loads((STARTUP_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_label"] == "SYNTHETIC" and manifest["count"] == 6
    assert manifest["expected_recommendations"] == {
        "approve": 1,
        "refer": 2,
        "decline_for_review": 1,
        "request_more_information": 2,
    }


def test_reference_policy_agrees_with_every_designed_label() -> None:
    cases = [evaluate_policy_case(application) for application in load_applications(STARTUP_DIR)]
    assert [(case.scenario, case.recommendation_match, case.rules_match) for case in cases] == [
        (case.scenario, True, True) for case in cases
    ]
