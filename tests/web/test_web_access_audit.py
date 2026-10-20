"""The access log of founders' plans (src/uwcopilot/web/audit_log.py): one event for every access, written before it,
holding ids only, and a refused request (503) when the event cannot be written.

Every plan is SYNTHETIC test input on the SYNTHETIC metro; the compute routes run the real (fast) engines on it.
Nothing touches the network.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from web_helpers import csrf_headers, dev_login

from uwcopilot.security.audit import verify_audit_file
from uwcopilot.studio.siteiq.context import metro_context
from uwcopilot.web.app import create_app
from uwcopilot.web.audit_log import ACTION_EVENT_TYPES, AUDIT_UNAVAILABLE_MESSAGE, PlanAccessLog, owner_actor
from uwcopilot.web.auth import Principal
from uwcopilot.web.services.projects import PROJECTS_NAMESPACE
from uwcopilot.web.store import owner_digest

SYNTHETIC_METRO = "synthetic_metro"
PLAN_NAME = "SYNTHETIC Brightwater laundromat"
IDEA_TEXT = "SYNTHETIC idea: a laundromat with a quiet reading corner."
SITE_LABEL = "SYNTHETIC Harbor lot"


def plan_body(**changes: Any) -> dict[str, Any]:
    """A SYNTHETIC laundromat plan with recognisable text, so a test can show none of it reaches the log."""
    tracts = metro_context(SYNTHETIC_METRO).tracts
    body: dict[str, Any] = {
        "name": PLAN_NAME,
        "metro_id": SYNTHETIC_METRO,
        "business_type_id": "laundromat",
        "idea_text": IDEA_TEXT,
        "budget_total": 400_000.0,
        "owner_equity": 150_000.0,
        "loan_amount": 350_000.0,
        "loan_term_months": 120,
        "annual_rate_pct": 9.0,
        "sites": [{"label": SITE_LABEL, "lat": float(tracts.lats[0]), "lon": float(tracts.lons[0])}],
    }
    body.update(changes)
    return body


@pytest.fixture
def studio(make_settings) -> Iterator[TestClient]:
    """A dev-login app signed in as founder A."""
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1"))) as client:
        dev_login(client, "founder_a", "founder")
        yield client


def owner_of(client: TestClient) -> str:
    """The owner id of the signed-in user (as the store sees it)."""
    me = client.get("/api/me").json()
    return f"{me['auth_mode']}:{me['subject']}"


def access_log(client: TestClient) -> PlanAccessLog:
    """The app's plan access log."""
    return client.app.state.access_log


def actions(client: TestClient) -> list[str]:
    """The action of every event so far."""
    return [event["action"] for event in access_log(client).events()]


def create(client: TestClient) -> dict[str, Any]:
    """POST a plan and return its JSON (asserting 201)."""
    response = client.post("/api/studio/projects", json=plan_body(), headers=csrf_headers(client))
    assert response.status_code == 201, response.text
    return response.json()


def test_the_log_sits_next_to_the_store_and_is_a_setting(make_settings, tmp_path: Path) -> None:
    assert make_settings().web_audit_log == tmp_path / "audit" / "plan_access_events.jsonl"
    custom = tmp_path / "elsewhere" / "access.jsonl"
    assert make_settings(UWC_WEB_AUDIT_LOG=str(custom)).web_audit_log == custom


def test_every_plan_route_writes_one_event(studio: TestClient) -> None:
    plan = create(studio)
    plan_id = plan["id"]
    headers = csrf_headers(studio)
    calls = [
        ("GET", "/api/studio/projects", None, "list_plans"),
        ("GET", f"/api/studio/projects/{plan_id}", None, "read_plan"),
        ("PUT", f"/api/studio/projects/{plan_id}", {"loan_term_months": 96}, "update_plan"),
        ("POST", f"/api/studio/projects/{plan_id}/sites/score", None, "score_sites"),
        ("POST", f"/api/studio/projects/{plan_id}/feasibility", {"site_index": 0}, "run_feasibility"),
        ("POST", f"/api/studio/projects/{plan_id}/path-to-approval", {"site_index": 0}, "run_path_to_approval"),
        ("DELETE", f"/api/studio/projects/{plan_id}", None, "delete_plan"),
    ]
    for method, url, body, _ in calls:
        response = studio.request(method, url, json=body, headers=headers)
        assert response.status_code == 200, f"{method} {url}: {response.text}"
    assert actions(studio) == ["create_plan"] + [action for *_, action in calls]

    digest = owner_digest(owner_of(studio))
    for event in access_log(studio).events():
        assert event["actor"] == {"principal_id": digest, "role": "unknown", "role_source": "dev_login"}
        assert event["event_type"] == ACTION_EVENT_TYPES[event["action"]]
        assert event["outcome"] == "allowed" and event["data_label"] == "USER_DATA"
        assert event["details"]["actor_kind"] == "founder"
        if event["action"] != "list_plans":
            assert event["resource"] == {"kind": "studio_plan", "id": plan_id}
    assert access_log(studio).audit.verify() == []


def test_events_hold_ids_only_never_the_owner_id_or_the_plan(studio: TestClient) -> None:
    plan = create(studio)
    studio.get(f"/api/studio/projects/{plan['id']}")
    text = access_log(studio).path.read_text(encoding="utf-8")
    owner = owner_of(studio)
    for secret in (owner, "founder_a", PLAN_NAME, IDEA_TEXT, SITE_LABEL, "Brightwater", "350000.0", "150000.0"):
        assert secret not in text, secret
    assert owner_digest(owner) in text and plan["id"] in text


def test_another_users_plan_id_writes_no_event(studio: TestClient) -> None:
    plan = create(studio)
    dev_login(studio, "founder_b", "founder")
    before = len(access_log(studio).events())
    assert studio.get(f"/api/studio/projects/{plan['id']}").status_code == 404
    assert studio.delete(f"/api/studio/projects/{plan['id']}", headers=csrf_headers(studio)).status_code == 404
    assert len(access_log(studio).events()) == before


def test_an_hf_session_is_logged_as_session_allowlist() -> None:
    principal = Principal("6123abc", "someone", "Someone", "hf_oauth", ("founder",))
    actor = owner_actor(principal)
    assert actor.principal_id == owner_digest("hf_oauth:6123abc") and actor.role_source == "session_allowlist"


def test_an_unknown_action_is_a_programming_error(tmp_path: Path) -> None:
    log = PlanAccessLog(tmp_path / "access.jsonl")
    principal = Principal("founder_a", "founder_a", "founder_a", "dev", ("founder",))
    with pytest.raises(ValueError, match="unknown plan access action"):
        log.record_owner_access(principal, "export_everything")
    assert not log.path.exists()


# --- the failure policy: no event, no access ---


def break_the_log(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every append fail, as a full disk would."""

    def failing_append(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError("No space left on device (test)")

    monkeypatch.setattr(access_log(client).audit, "append", failing_append)


def test_a_read_is_refused_when_its_event_cannot_be_written(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create(studio)
    break_the_log(studio, monkeypatch)
    for url in (f"/api/studio/projects/{plan['id']}", "/api/studio/projects"):
        response = studio.get(url)
        assert response.status_code == 503 and response.json() == {"detail": AUDIT_UNAVAILABLE_MESSAGE}
        assert PLAN_NAME not in response.text and plan["id"] not in response.text


def test_a_change_is_not_made_when_its_event_cannot_be_written(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create(studio)
    owner = owner_of(studio)
    store = studio.app.state.store
    break_the_log(studio, monkeypatch)
    headers = csrf_headers(studio)
    url = f"/api/studio/projects/{plan['id']}"
    assert studio.put(url, json={"loan_term_months": 96}, headers=headers).status_code == 503
    assert studio.delete(url, headers=headers).status_code == 503
    assert studio.post("/api/studio/projects", json=plan_body(), headers=headers).status_code == 503
    assert [meta.key for meta in store.list(owner, PROJECTS_NAMESPACE)] == [plan["id"]]
    assert store.get_json(owner, PROJECTS_NAMESPACE, plan["id"])["version"] == 1


def test_no_calculation_starts_when_its_event_cannot_be_written(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create(studio)
    owner = owner_of(studio)
    break_the_log(studio, monkeypatch)
    compute = studio.app.state.services.compute
    headers = csrf_headers(studio)
    for route, body in (("sites/score", None), ("feasibility", {"site_index": 0}), ("path-to-approval", {})):
        response = studio.post(f"/api/studio/projects/{plan['id']}/{route}", json=body, headers=headers)
        assert response.status_code == 503, route
    assert compute.pending_jobs() == 0
    assert compute.forget(("studio", owner)) == 0  # nothing was computed for the plan, so nothing is cached


def test_a_damaged_log_refuses_access_and_the_damage_is_reported(studio: TestClient) -> None:
    plan = create(studio)
    path = access_log(studio).path
    with path.open("ab") as handle:
        handle.write(b'{"cut off')  # a last line without its end: AuditLog refuses to append after it
    response = studio.get(f"/api/studio/projects/{plan['id']}")
    assert response.status_code == 503
    problems = verify_audit_file(path, access_log(studio).audit.head())
    assert any("cut off" in problem for problem in problems)
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["action"] == "create_plan"
