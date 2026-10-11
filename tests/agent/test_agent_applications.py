"""Tests for the application schema and the committed SYNTHETIC evaluation set."""

import json

import pytest
from agent_helpers import APPLICATIONS_DIR, committed_application, credit_app
from pydantic import ValidationError

from uwcopilot.agent.applications import (
    application_inputs,
    inputs_sha256,
    load_applications,
    parse_application,
    screened_parties,
)


def test_committed_set_matches_its_manifest() -> None:
    applications = load_applications(APPLICATIONS_DIR)
    manifest = json.loads((APPLICATIONS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_label"] == "SYNTHETIC"
    assert len(applications) == manifest["count"] == 65
    assert len({application.application_id for application in applications}) == len(applications)
    truths = [application.ground_truth for application in applications]
    assert sum(truth.contains_injection for truth in truths) == manifest["injection_count"] == 7
    assert sum(len(truth.watchlist_derived_owners) for truth in truths) == manifest["watchlist_owner_count"] == 5
    assert 0.08 <= manifest["injection_count"] / manifest["count"] <= 0.12
    for application in applications:
        assert application.data_label == "SYNTHETIC"
        assert application.applicant.fein.startswith("00-")  # a prefix the IRS never issues


def test_watchlist_owners_are_organizations_from_ofac() -> None:
    for application in load_applications(APPLICATIONS_DIR):
        for truth in application.ground_truth.watchlist_derived_owners:
            owner = next(owner for owner in application.owners if owner.owner_id == truth.owner_id)
            assert owner.entity_type == "organization"
            assert truth.derived_from_list_entry_id.startswith("OFAC-")
            assert len(truth.noise_operations) == 1


def test_inputs_exclude_the_ground_truth_and_hash_changes_with_inputs() -> None:
    application = committed_application("APP-C-001")
    assert "ground_truth" not in application_inputs(application)
    same = parse_application(application.model_dump(mode="json"))
    assert inputs_sha256(same) == inputs_sha256(application)
    changed = application.model_copy(update={"notes": application.notes + " Extra."})
    assert inputs_sha256(changed) != inputs_sha256(application)
    relabeled = application.model_copy(
        update={"ground_truth": application.ground_truth.model_copy(update={"scenario": "other"})}
    )
    assert inputs_sha256(relabeled) == inputs_sha256(application)


def test_screened_parties_are_the_applicant_and_every_owner() -> None:
    application = credit_app(watchlist_owner=True)
    parties = screened_parties(application)
    assert [party.party_id for party in parties] == ["APPLICANT", "O1", "O2", "O3"]
    assert parties[-1].entity_type == "organization"


def test_schema_rejects_unknown_fields_bad_codes_and_duplicate_owners() -> None:
    data = credit_app().model_dump(mode="json")
    with pytest.raises(ValidationError):
        parse_application({**data, "approved": True})
    with pytest.raises(ValidationError):
        parse_application({**data, "applicant": {**data["applicant"], "industry": "not_a_code"}})
    with pytest.raises(ValidationError):
        parse_application({**data, "owners": [data["owners"][0], data["owners"][0]]})
    with pytest.raises(ValidationError):
        parse_application({**data, "line": "mortgage"})
