"""DELETE /api/me/data (src/uwcopilot/web/routers/account.py): every object of the signed-in user is deleted, their
results are forgotten, their share links stop working, they are signed out, and nobody else's data changes. With the
Hugging Face dataset store (a fake Hub from test_web_hf_store.py, no network) the files are deleted on the Hub too,
or kept as pending deletes when the Hub cannot be reached.

Every plan is SYNTHETIC test input on the SYNTHETIC metro; the DEMO decision record is a SYNTHETIC placeholder
written straight into the store.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_web_hf_store import FakeHubApi, remote_object_paths
from web_helpers import csrf_headers, dev_login

from uwcopilot.studio.siteiq.context import metro_context
from uwcopilot.web.app import create_app
from uwcopilot.web.audit_log import AUDIT_UNAVAILABLE_MESSAGE
from uwcopilot.web.crypto import generate_key
from uwcopilot.web.hf_store import PENDING_DELETES_FILE, HFDatasetStore
from uwcopilot.web.services.lender_demo import DECISIONS_NAMESPACE
from uwcopilot.web.services.projects import PROJECTS_NAMESPACE
from uwcopilot.web.services.share_links import SHARES_NAMESPACE
from uwcopilot.web.store import owner_digest

SYNTHETIC_METRO = "synthetic_metro"
NAMESPACES = (PROJECTS_NAMESPACE, DECISIONS_NAMESPACE, SHARES_NAMESPACE)
DEMO_FILE = "APP-C-001"
OWNER_A = "dev:founder_a"
OWNER_B = "dev:founder_b"


def plan_body() -> dict[str, Any]:
    """A SYNTHETIC laundromat plan on one map pin of the SYNTHETIC city (test inputs, not benchmarks)."""
    tracts = metro_context(SYNTHETIC_METRO).tracts
    return {
        "metro_id": SYNTHETIC_METRO,
        "business_type_id": "laundromat",
        "budget_total": 400_000.0,
        "owner_equity": 150_000.0,
        "loan_amount": 350_000.0,
        "loan_term_months": 120,
        "annual_rate_pct": 9.0,
        "sites": [{"label": "Site 1", "lat": float(tracts.lats[0]), "lon": float(tracts.lons[0])}],
    }


def create_plan(client: TestClient) -> str:
    """POST a plan; return its id."""
    response = client.post("/api/studio/projects", json=plan_body(), headers=csrf_headers(client))
    assert response.status_code == 201, response.text
    return response.json()["id"]


def share(client: TestClient, plan_id: str) -> str:
    """Make a share link; return its token."""
    response = client.post(f"/api/studio/projects/{plan_id}/share", headers=csrf_headers(client))
    assert response.status_code == 200, response.text
    return response.json()["token"]


def store_demo_record(client: TestClient, owner: str) -> None:
    """A SYNTHETIC placeholder DEMO decision record, written straight into the store."""
    placeholder = {"note": "SYNTHETIC placeholder decision record written by tests/web/test_web_delete_my_data"}
    record = {"label": "DEMO", "data_label": "SYNTHETIC", "file_id": DEMO_FILE, "record": placeholder}
    client.app.state.store.put_json(owner, DECISIONS_NAMESPACE, DEMO_FILE, record)


def fill_two_accounts(client: TestClient) -> dict[str, Any]:
    """Founder A: two plans, a computed result, a DEMO record and a share link. Founder B: one plan.
    Ends signed in as founder A. Returns the ids the tests check."""
    dev_login(client, "founder_b", "founder")
    plan_b = create_plan(client)
    dev_login(client, "founder_a", "founder")
    plan_a, other_plan_a = create_plan(client), create_plan(client)
    scored = client.post(f"/api/studio/projects/{plan_a}/sites/score", headers=csrf_headers(client))
    assert scored.status_code == 200, scored.text
    store_demo_record(client, OWNER_A)
    token = share(client, plan_a)
    return {"plan_a": plan_a, "other_plan_a": other_plan_a, "plan_b": plan_b, "token": token}


def delete_my_data(client: TestClient) -> Any:
    """DELETE /api/me/data with the CSRF token."""
    return client.delete("/api/me/data", headers=csrf_headers(client))


@pytest.fixture
def app_client(make_settings) -> Iterator[TestClient]:
    """A dev-login app with the local store, nobody signed in yet."""
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1"))) as client:
        yield client


def test_everything_of_the_user_is_deleted_and_nothing_of_anyone_else(app_client: TestClient, tmp_path: Path) -> None:
    ids = fill_two_accounts(app_client)
    store = app_client.app.state.store
    compute = app_client.app.state.services.compute
    result_key = ("studio", OWNER_A, ids["plan_a"], 1, "sites")
    assert compute.cached(result_key)[0]
    assert app_client.get("/api/me").json()["username"] == "founder_a"

    response = delete_my_data(app_client)

    assert response.status_code == 200, response.text
    assert response.json() == {"deleted_objects": 4, "pending": False, "signed_out": True}
    assert response.headers["cache-control"] == "no-store"
    assert "uwc_session=;" in response.headers["set-cookie"] and "Max-Age=0" in response.headers["set-cookie"]
    # founder A: signed out, nothing stored in any namespace, no folder, no result in memory, the link is dead
    assert app_client.get("/api/me").json()["signed_in"] is False
    assert all(store.list(OWNER_A, namespace) == [] for namespace in NAMESPACES)
    assert not (tmp_path / "store" / owner_digest(OWNER_A)).exists()
    assert compute.cached(result_key) == (False, None)
    assert app_client.get(f"/api/share/{ids['token']}").status_code == 404
    # founder B: untouched
    assert [meta.key for meta in store.list(OWNER_B, PROJECTS_NAMESPACE)] == [ids["plan_b"]]
    dev_login(app_client, "founder_b", "founder")
    assert app_client.get(f"/api/studio/projects/{ids['plan_b']}").status_code == 200
    # founder A signs in again and finds nothing
    dev_login(app_client, "founder_a", "founder")
    assert app_client.get("/api/studio/projects").json() == {"projects": [], "total": 0}


def test_the_deletion_is_in_the_access_log_under_the_owner_digest(app_client: TestClient) -> None:
    fill_two_accounts(app_client)
    assert delete_my_data(app_client).status_code == 200
    log = app_client.app.state.access_log
    [event] = [event for event in log.events() if event["action"] == "delete_all_data"]
    assert event["actor"]["principal_id"] == owner_digest(OWNER_A)
    assert event["resource"] is None and event["event_type"] == "permission_change"
    text = log.path.read_text(encoding="utf-8")
    assert OWNER_A not in text and "founder_a" not in text and OWNER_B not in text


def test_an_account_with_nothing_stored_gets_zero(app_client: TestClient) -> None:
    dev_login(app_client, "founder_c", "founder")
    response = delete_my_data(app_client)
    assert response.json() == {"deleted_objects": 0, "pending": False, "signed_out": True}


def test_signed_out_and_csrf_refusals_delete_nothing(app_client: TestClient) -> None:
    ids = fill_two_accounts(app_client)
    assert app_client.delete("/api/me/data").status_code == 403  # no X-CSRF-Token header
    app_client.post("/api/auth/logout", headers=csrf_headers(app_client))
    response = delete_my_data(app_client)
    assert response.status_code == 401 and response.json() == {"detail": "sign in required"}
    keys = [meta.key for meta in app_client.app.state.store.list(OWNER_A, PROJECTS_NAMESPACE)]
    assert sorted(keys) == sorted([ids["plan_a"], ids["other_plan_a"]])


def test_nothing_is_deleted_when_the_event_cannot_be_written(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = fill_two_accounts(app_client)

    def failing_append(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError("No space left on device (test)")

    monkeypatch.setattr(app_client.app.state.access_log.audit, "append", failing_append)
    response = delete_my_data(app_client)
    assert response.status_code == 503 and response.json() == {"detail": AUDIT_UNAVAILABLE_MESSAGE}
    assert "set-cookie" not in response.headers
    assert len(app_client.app.state.store.list(OWNER_A, PROJECTS_NAMESPACE)) == 2
    monkeypatch.undo()
    assert app_client.get(f"/api/share/{ids['token']}").status_code == 200
    assert app_client.get("/api/me").json()["signed_in"] is True


# --- the Hugging Face dataset store ---


@pytest.fixture
def hub() -> FakeHubApi:
    """An in-memory private dataset repo."""
    return FakeHubApi()


@pytest.fixture
def dataset_client(make_settings, hub: FakeHubApi) -> Iterator[TestClient]:
    """A dev-login app whose store is mirrored to the fake Hub."""
    settings = make_settings(
        UWC_DEV_AUTH="1",
        UWC_STORE="hf_dataset",
        UWC_HF_DATASET_REPO="astra6/fundproof-data",
        HF_TOKEN="hf_test_token",
        UWC_FERNET_KEY=generate_key(),
    )
    with TestClient(create_app(settings, hub_api=hub)) as client:
        yield client


def remote_paths_of(hub: FakeHubApi, owner: str) -> list[str]:
    """The owner's files in the dataset repo."""
    return [path for path in remote_object_paths(hub) if path.startswith(owner_digest(owner) + "/")]


def history_paths_of(hub: FakeHubApi, owner: str) -> list[str]:
    """The owner's files in ANY revision of the dataset repo."""
    prefix = owner_digest(owner) + "/"
    return sorted({path for revision in hub.history for path in revision if path.startswith(prefix)})


def test_with_the_dataset_store_the_files_are_deleted_on_the_hub_and_from_its_history(
    dataset_client: TestClient, hub: FakeHubApi
) -> None:
    fill_two_accounts(dataset_client)
    store: HFDatasetStore = dataset_client.app.state.store
    store.flush()
    assert len(remote_paths_of(hub, OWNER_A)) == 8 and len(remote_paths_of(hub, OWNER_B)) == 2

    response = delete_my_data(dataset_client)

    assert response.json() == {"deleted_objects": 4, "pending": False, "signed_out": True}
    assert remote_paths_of(hub, OWNER_A) == []  # deleted on the Hub at once
    assert history_paths_of(hub, OWNER_A) != []  # ... but earlier revisions still hold them until the squash
    store.flush()
    assert history_paths_of(hub, OWNER_A) == []
    assert len(remote_paths_of(hub, OWNER_B)) == 2


def test_with_the_hub_down_the_deletion_is_pending_and_scheduled(dataset_client: TestClient, hub: FakeHubApi) -> None:
    ids = fill_two_accounts(dataset_client)
    store: HFDatasetStore = dataset_client.app.state.store
    store.flush()
    hub.fail_deletes = True

    response = delete_my_data(dataset_client)

    assert response.status_code == 200
    assert response.json() == {"deleted_objects": 4, "pending": True, "signed_out": True}
    assert dataset_client.get("/api/me").json()["signed_in"] is False
    # deleted here (the link no longer works), scheduled for removal on the Hub, saved to survive a restart
    assert all(store.list(OWNER_A, namespace) == [] for namespace in NAMESPACES)
    assert dataset_client.get(f"/api/share/{ids['token']}").status_code == 404
    owner_files = remote_paths_of(hub, OWNER_A)
    assert len(owner_files) == 8  # still on the Hub for now
    # the pending delete names each of the owner's files (not their folder, so a later write is never caught by it)
    assert store._pending_deletes == set(owner_files)
    saved = (store.local.root / PENDING_DELETES_FILE).read_text(encoding="utf-8")
    assert all(path in saved for path in owner_files)

    hub.fail_deletes = False
    store.flush()  # the retry
    assert remote_paths_of(hub, OWNER_A) == [] and history_paths_of(hub, OWNER_A) == []
    assert store._pending_deletes == set()
    assert len(remote_paths_of(hub, OWNER_B)) == 2
