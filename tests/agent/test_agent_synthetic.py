"""Tests for the SYNTHETIC application generator (offline, over a fictional watchlist)."""

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from agent_helpers import FIXTURE_POOL

from uwcopilot.agent.applications import parse_application
from uwcopilot.agent.synthetic_apps import (
    INJECTION_NOTES,
    generate_applications,
    ownership_split,
    shift_months,
    watchlist_pool,
)
from uwcopilot.agent.synthetic_credit import CREDIT_SCENARIOS
from uwcopilot.agent.synthetic_insurance import INSURANCE_SCENARIOS
from uwcopilot.agent.vocab import RECOMMENDATIONS


def test_generates_every_scenario_with_valid_labels() -> None:
    applications = generate_applications(FIXTURE_POOL)
    assert len(applications) == len(CREDIT_SCENARIOS) + len(INSURANCE_SCENARIOS) == 65
    for application in applications:
        parsed = parse_application(application)
        assert parsed.data_label == "SYNTHETIC"
        assert parsed.ground_truth.expected_recommendation in RECOMMENDATIONS
    truths = [application["ground_truth"] for application in applications]
    assert sum(truth["contains_injection"] for truth in truths) == len(INJECTION_NOTES) == 7
    planted = [owner for truth in truths for owner in truth["watchlist_derived_owners"]]
    assert len(planted) == 5
    assert {owner["derived_from_list_entry_id"] for owner in planted} <= {org.list_entry_id for org in FIXTURE_POOL}


def test_generation_is_deterministic_in_process_and_across_hash_seeds(tmp_path: Path) -> None:
    first = json.dumps(generate_applications(FIXTURE_POOL), sort_keys=True)
    assert first == json.dumps(generate_applications(FIXTURE_POOL), sort_keys=True)
    script = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from agent_helpers import FIXTURE_POOL;"
        "from uwcopilot.agent.synthetic_apps import generate_applications;"
        "print(json.dumps(generate_applications(FIXTURE_POOL), sort_keys=True))"
    )
    outputs = []
    for seed in ("1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1] == first


def test_watchlist_pool_keeps_latin_ofac_organizations_only() -> None:
    entries = pd.DataFrame(
        [
            {
                "list_entry_id": "OFAC-2",
                "name": "ZETA TEST TRADING CO",
                "entity_type": "organization",
                "countries": ("ir",),
                "source": "OFAC",
            },
            {
                "list_entry_id": "OFAC-1",
                "name": "DOE, John",
                "entity_type": "person",
                "countries": (),
                "source": "OFAC",
            },
            {
                "list_entry_id": "OS-1",
                "name": "OMEGA TEST LLC",
                "entity_type": "organization",
                "countries": (),
                "source": "OpenSanctions",
            },
            {
                "list_entry_id": "OFAC-3",
                "name": "ШТОРМ",
                "entity_type": "organization",
                "countries": (),
                "source": "OFAC",
            },
        ]
    )
    pool = watchlist_pool(entries)
    assert [org.list_entry_id for org in pool] == ["OFAC-2"]
    assert pool[0].country == "IR"


def test_helpers() -> None:
    assert shift_months(date(2026, 9, 30), -20) == date(2025, 1, 28)
    assert shift_months(date(2026, 1, 15), -1) == date(2025, 12, 15)
    assert ownership_split(70.0, 2) == [42.0, 28.0]
    assert sum(ownership_split(100.0, 3)) == 100.0


def test_reserved_watchlist_names_are_drawn_again() -> None:
    from uwcopilot.kyc.synthetic import collision_key

    plain = generate_applications(FIXTURE_POOL)
    taken = plain[0]["owners"][0]["name"]
    report: dict = {}
    redrawn = generate_applications(
        FIXTURE_POOL, reserved_keys=frozenset({collision_key(taken, "person")}), report=report
    )
    assert report["clean_name_collisions_redrawn"] >= 1
    names = [owner["name"] for application in redrawn for owner in application["owners"]]
    assert taken not in names
