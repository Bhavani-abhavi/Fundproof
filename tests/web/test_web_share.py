"""Share links (src/uwcopilot/web/services/share_links.py, routers/share.py): read-only, minimal, expiring,
revocable, owner-only, and every bad link gets the same 404.

Every plan is SYNTHETIC test input: SYNTHETIC-metro map pins, or for the address test a made-up address answered by
the fake geocoder of test_web_studio.py. Nothing touches the network.
"""

import base64
import hashlib
import hmac
import logging
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.fernet import Fernet, InvalidToken
from fastapi.testclient import TestClient
from test_web_hf_store import FakeHubApi
from test_web_studio import FakeGeocoder, first_verified_metro
from web_helpers import csrf_headers, dev_login

from uwcopilot.studio.data.geocode import GeocodeMatch
from uwcopilot.studio.metros import load_metro
from uwcopilot.studio.siteiq.context import metro_context
from uwcopilot.web.app import create_app
from uwcopilot.web.crypto import CryptoError, Encryptor, generate_key
from uwcopilot.web.routers.share import SHARE_LINK_NOT_FOUND
from uwcopilot.web.services.errors import Conflict
from uwcopilot.web.services.share_links import (
    MAX_LINKS_PER_PLAN,
    SHARES_NAMESPACE,
    TOKEN_KEY_LABEL,
    ShareLinkInvalid,
    ShareLinks,
    share_key,
)
from uwcopilot.web.store import LocalFileStore, owner_digest

SYNTHETIC_METRO = "synthetic_metro"
IDEA_TEXT = "SYNTHETIC idea: a laundromat next to the night market."
ADDRESS = "100 Test Street (SYNTHETIC)"
NOT_FOUND_BODY = {"detail": SHARE_LINK_NOT_FOUND}
SHARED_FIELDS = {
    "plan_name",
    "business_type",
    "metro_id",
    "metro_name",
    "money",
    "opening_date",
    "sites",
    "expires_at",
    "disclaimer",
    "notice",
}


def plan_body(**changes: Any) -> dict[str, Any]:
    """A SYNTHETIC laundromat plan on map pins of the SYNTHETIC city (test inputs, not benchmarks)."""
    tracts = metro_context(SYNTHETIC_METRO).tracts
    body: dict[str, Any] = {
        "name": "SYNTHETIC shared plan",
        "metro_id": SYNTHETIC_METRO,
        "business_type_id": "laundromat",
        "idea_text": IDEA_TEXT,
        "budget_total": 400_000.0,
        "owner_equity": 150_000.0,
        "loan_amount": 350_000.0,
        "loan_term_months": 120,
        "annual_rate_pct": 9.0,
        "sites": [
            {"label": "North pin", "lat": float(tracts.lats[0]), "lon": float(tracts.lons[0])},
            {"label": "South pin", "lat": float(tracts.lats[1]), "lon": float(tracts.lons[1])},
        ],
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
    """The owner id of the signed-in user."""
    me = client.get("/api/me").json()
    return f"{me['auth_mode']}:{me['subject']}"


def create_plan(client: TestClient, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST a plan and return it."""
    response = client.post("/api/studio/projects", json=body or plan_body(), headers=csrf_headers(client))
    assert response.status_code == 201, response.text
    return response.json()


def share(client: TestClient, plan_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """POST a share link and return its JSON (asserting 200)."""
    response = client.post(f"/api/studio/projects/{plan_id}/share", json=body, headers=csrf_headers(client))
    assert response.status_code == 200, response.text
    return response.json()


def view(client: TestClient, token: str) -> Any:
    """GET the shared plan."""
    return client.get(f"/api/share/{token}")


def assert_invalid(response: Any) -> None:
    """The one answer every bad link gets."""
    assert response.status_code == 404, response.text
    assert response.json() == NOT_FOUND_BODY
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


# --- a valid link ---


def test_a_valid_link_shows_the_plan_read_only_without_sign_in(studio: TestClient) -> None:
    plan = create_plan(studio)
    created = share(studio, plan["id"])
    assert created["url"] == "/share?" + "t=" + created["token"].replace("=", "%3D")
    assert parse_qs(urlsplit(created["url"]).query)["t"] == [created["token"]]
    studio.post("/api/auth/logout", headers=csrf_headers(studio))
    assert studio.get("/api/me").json()["signed_in"] is False

    response = view(studio, created["token"])

    assert response.status_code == 200, response.text
    shared = response.json()
    assert set(shared) == SHARED_FIELDS
    assert shared["plan_name"] == "SYNTHETIC shared plan" and shared["business_type"] == "Laundromat"
    assert shared["metro_id"] == SYNTHETIC_METRO and "SYNTHETIC" in shared["metro_name"]
    assert shared["money"] == plan["money"] and shared["opening_date"] == plan["opening_date"]
    assert shared["sites"] == [
        {"name": site["label"], "lat": round(site["lat"], 2), "lon": round(site["lon"], 2)} for site in plan["sites"]
    ]
    assert "approximate location (rounded to about 1 km)" in shared["notice"]
    assert shared["expires_at"] == created["expires_at"]
    assert shared["disclaimer"].startswith("Educational planning tool") and "read-only" in shared["notice"]
    assert IDEA_TEXT not in response.text and plan["id"] not in response.text and "founder_a" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_the_default_link_lasts_seven_days_and_the_limit_is_thirty(studio: TestClient) -> None:
    plan = create_plan(studio)
    before = datetime.now(UTC)
    expires = datetime.fromisoformat(share(studio, plan["id"])["expires_at"])
    assert timedelta(days=7) - timedelta(seconds=5) <= expires - before <= timedelta(days=7)
    expires = datetime.fromisoformat(share(studio, plan["id"], {"expires_in_days": 2})["expires_at"])
    assert expires - before <= timedelta(days=2)
    headers = csrf_headers(studio)
    for days in (0, 31, "a week"):
        body = {"expires_in_days": days}
        assert studio.post(f"/api/studio/projects/{plan['id']}/share", json=body, headers=headers).status_code == 422


def test_a_link_never_outlives_the_plans_retention(make_settings) -> None:
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1", UWC_RETENTION_DAYS="3"))) as client:
        dev_login(client, "founder_a", "founder")
        plan = create_plan(client)
        created = share(client, plan["id"], {"expires_in_days": 30})
        assert datetime.fromisoformat(created["expires_at"]) <= datetime.fromisoformat(plan["expires_at"])


class PreciseGeocoder(FakeGeocoder):
    """The fake geocoder, answering with a precise (SYNTHETIC) point, as the Census Geocoder interpolates one."""

    def geocode(self, address: str) -> GeocodeMatch:
        return replace(super().geocode(address), lat=29.761234, lon=-95.369876)


def test_no_street_address_is_shared(studio: TestClient) -> None:
    """SL-2: the shared position used to be the exact geocoded point of the address; it is now rounded (~1 km)."""
    metro_id = first_verified_metro()
    if metro_id is None:
        pytest.skip("no real metro has a verified county list in this checkout")
    studio.app.state.services.geocoding.geocoder = PreciseGeocoder(load_metro(metro_id).county_geoids()[0])
    plan = create_plan(studio, plan_body(metro_id=metro_id, sites=[{"label": "Main St", "address": ADDRESS}]))
    assert plan["sites"][0]["address"] == ADDRESS
    assert (plan["sites"][0]["lat"], plan["sites"][0]["lon"]) == (29.761234, -95.369876)  # the owner sees it all
    response = view(studio, share(studio, plan["id"])["token"])
    assert response.status_code == 200
    assert response.json()["sites"] == [{"name": "Main St", "lat": 29.76, "lon": -95.37}]
    for hidden in (ADDRESS, "TEST ADDRESS", "census_geocoder", "fake geocoder", "29.7612", "95.3698"):
        assert hidden not in response.text, hidden


# --- every bad link: the same 404 ---


def test_an_expired_link_is_refused(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"], {"expires_in_days": 1})["token"]
    links: ShareLinks = studio.app.state.share_links
    links.clock = lambda: datetime.now(UTC) + timedelta(days=1, minutes=1)
    assert_invalid(view(studio, token))
    links.clock = lambda: datetime.now(UTC)
    assert view(studio, token).status_code == 200


def test_the_record_expiry_is_enforced_too(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    owner = owner_of(studio)
    store = studio.app.state.store
    [meta] = store.list(owner, SHARES_NAMESPACE)
    record = store.get_json(owner, SHARES_NAMESPACE, meta.key)
    record["expires_at"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    store.put_json(owner, SHARES_NAMESPACE, meta.key, record)  # the token itself is still unexpired
    assert_invalid(view(studio, token))


def test_tampered_and_malformed_tokens_are_refused(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    middle = len(token) // 2
    flipped = token[:middle] + ("A" if token[middle] != "A" else "B") + token[middle + 1 :]
    other_key = Fernet(Fernet.generate_key()).encrypt(b'{"v":1}').decode()
    unused_bits = token.rstrip("=")[:-1] + chr(ord(token.rstrip("=")[-1]) ^ 1) + "=" * token.count("=")
    bad_tokens = (flipped, token[:-4], token + "AAAA", token + "=", unused_bits, "not-a-token", "x" * 2000, "ü" * 10)
    for bad in (*bad_tokens, other_key):
        assert_invalid(view(studio, bad))
    assert view(studio, token).status_code == 200


def test_a_token_made_with_another_session_secret_is_refused(studio: TestClient) -> None:
    plan = create_plan(studio)
    owner = owner_of(studio)
    forged = ShareLinks("another-session-secret-0123456789-abcdef").create(
        studio.app.state.store, owner, plan["id"], 7, None
    )
    assert_invalid(view(studio, forged.token))  # its record exists, but the token does not decrypt here


def test_a_revoked_link_is_refused(studio: TestClient) -> None:
    plan = create_plan(studio)
    first, second = share(studio, plan["id"])["token"], share(studio, plan["id"])["token"]
    other_plan = create_plan(studio)
    kept = share(studio, other_plan["id"])["token"]
    response = studio.delete(f"/api/studio/projects/{plan['id']}/share", headers=csrf_headers(studio))
    assert response.status_code == 200 and response.json() == {"revoked": 2, "pending": False}
    assert_invalid(view(studio, first))
    assert_invalid(view(studio, second))
    assert view(studio, kept).status_code == 200  # another plan's link is untouched
    again = studio.delete(f"/api/studio/projects/{plan['id']}/share", headers=csrf_headers(studio))
    assert again.json() == {"revoked": 0, "pending": False}


def test_a_revoke_whose_backup_deletion_is_pending_says_so(make_settings) -> None:
    """SL-1: the dataset store could not delete the record's backup copy yet. The link stops working here at once,
    but the answer says pending (it used to say nothing), because a restart before the retry would bring it back."""
    hub = FakeHubApi()
    settings = make_settings(
        UWC_DEV_AUTH="1",
        UWC_STORE="hf_dataset",
        UWC_HF_DATASET_REPO="astra6/fundproof-data",
        HF_TOKEN="hf_test_token",
        UWC_FERNET_KEY=generate_key(),
    )
    with TestClient(create_app(settings, hub_api=hub)) as client:
        dev_login(client, "founder_a", "founder")
        plan = create_plan(client)
        token = share(client, plan["id"])["token"]
        store = client.app.state.store
        store.flush()
        record_paths = [path for path in hub.files if f"/{SHARES_NAMESPACE}/" in path]
        assert len(record_paths) == 2
        hub.fail_deletes = True

        response = client.delete(f"/api/studio/projects/{plan['id']}/share", headers=csrf_headers(client))

        assert response.status_code == 200 and response.json() == {"revoked": 1, "pending": True}
        assert_invalid(view(client, token))  # it stops working here at once
        assert all(path in hub.files for path in record_paths)  # ... while the backup copy still exists
        hub.fail_deletes = False
        store.flush()  # the retry
        assert not any(path in hub.files for path in record_paths)


def test_expired_links_are_neither_counted_nor_reported_as_revoked(tmp_path: Path) -> None:
    """SL-4: a record's retention is whole days rounded up, so it can outlive its link by up to a day. Such links
    used to count against the cap (409) and in the revoke count."""
    now = datetime(2026, 10, 1, tzinfo=UTC)
    clock = {"now": now}
    store = LocalFileStore(tmp_path / "store", Encryptor([generate_key().encode()]), clock=lambda: clock["now"])
    links = ShareLinks("SYNTHETIC-test-session-secret-0123456789", clock=lambda: clock["now"])
    owner = "dev:founder_a"
    plan_expires_at = now + timedelta(days=4, hours=12)  # every 7-day link is cut to the plan's retention end
    for _ in range(MAX_LINKS_PER_PLAN):
        links.create(store, owner, "SYNTHETIC-plan", 7, plan_expires_at)
    with pytest.raises(Conflict):
        links.create(store, owner, "SYNTHETIC-plan", 7, plan_expires_at)

    clock["now"] = plan_expires_at + timedelta(hours=3)  # every link has expired ...
    assert len(store.list(owner, SHARES_NAMESPACE)) == MAX_LINKS_PER_PLAN  # ... but the records are kept to day 5

    assert links.links_for(store, owner, "SYNTHETIC-plan") == []
    assert links.revoke(store, owner, "SYNTHETIC-plan") == (0, False)
    links.create(store, owner, "SYNTHETIC-plan", 7, None)  # no 409: none of the old links works


def test_deleting_the_plan_kills_its_links(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    assert studio.delete(f"/api/studio/projects/{plan['id']}", headers=csrf_headers(studio)).status_code == 200
    assert_invalid(view(studio, token))
    assert studio.app.state.store.list(owner_of(studio), SHARES_NAMESPACE) == []  # its record went with it


def test_a_link_to_a_plan_that_is_gone_is_refused_even_with_its_record(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    studio.app.state.store.delete(owner_of(studio), "studio_projects", plan["id"])  # the plan only, not the record
    assert_invalid(view(studio, token))


def test_every_failure_has_the_same_body(studio: TestClient) -> None:
    plan = create_plan(studio)
    revoked = share(studio, plan["id"])["token"]
    studio.delete(f"/api/studio/projects/{plan['id']}/share", headers=csrf_headers(studio))
    expired = share(studio, plan["id"], {"expires_in_days": 1})["token"]
    deleted_plan = create_plan(studio)
    deleted = share(studio, deleted_plan["id"])["token"]
    studio.delete(f"/api/studio/projects/{deleted_plan['id']}", headers=csrf_headers(studio))
    studio.app.state.share_links.clock = lambda: datetime.now(UTC) + timedelta(days=2)
    answers = [view(studio, token) for token in (revoked, expired, deleted, "garbage", expired[:-1])]
    assert {(answer.status_code, answer.content) for answer in answers} == {(404, answers[0].content)}


# --- only the owner ---


def test_only_the_owner_can_share_or_revoke(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    dev_login(studio, "founder_b", "founder")
    headers = csrf_headers(studio)
    url = f"/api/studio/projects/{plan['id']}/share"
    assert studio.post(url, json={"expires_in_days": 30}, headers=headers).status_code == 404
    assert studio.delete(url, headers=headers).status_code == 404
    assert view(studio, token).status_code == 200  # founder B could not revoke founder A's link
    assert studio.app.state.store.list(owner_of(studio), SHARES_NAMESPACE) == []


def test_a_plan_keeps_a_bounded_number_of_links(studio: TestClient) -> None:
    plan = create_plan(studio)
    for _ in range(MAX_LINKS_PER_PLAN):
        share(studio, plan["id"])
    response = studio.post(f"/api/studio/projects/{plan['id']}/share", json=None, headers=csrf_headers(studio))
    assert response.status_code == 409 and "revoke" in response.json()["detail"]


# --- the token ---


def test_the_token_key_is_derived_from_the_session_secret_with_a_label(make_settings) -> None:
    secret = make_settings().session_secret.get_secret_value()
    expected = base64.urlsafe_b64encode(hmac.new(secret.encode(), TOKEN_KEY_LABEL, hashlib.sha256).digest())
    assert share_key(secret) == expected
    assert TOKEN_KEY_LABEL == b"fundproof-share-link-v1"
    assert share_key(secret) != share_key(secret + "x")


def test_the_token_hides_the_owner_id_and_is_not_readable_with_the_store_key(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    owner = owner_of(studio)
    decoded = base64.urlsafe_b64decode(token)
    for plain in (owner, "founder_a", plan["id"]):
        assert plain not in token and plain.encode() not in decoded
    store_encryptor = studio.app.state.store.encryptor  # the (dev) key of the local store
    with pytest.raises(CryptoError):
        store_encryptor.decrypt(token.encode())
    with pytest.raises(InvalidToken):
        Fernet(Fernet.generate_key()).decrypt(token.encode())


def test_resolve_names_the_failed_check_only_on_the_server(studio: TestClient) -> None:
    links: ShareLinks = studio.app.state.share_links
    store = studio.app.state.store
    other_key = Fernet(Fernet.generate_key()).encrypt(b"{}").decode()
    with pytest.raises(ShareLinkInvalid, match="does not decrypt"):
        links.resolve(store, other_key)
    with pytest.raises(ShareLinkInvalid, match="not spelled"):
        links.resolve(store, other_key + "AAAA")


# --- the access log and the request log ---


def test_link_events_are_logged_without_the_owner_id(studio: TestClient) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    owner = owner_of(studio)
    [meta] = studio.app.state.store.list(owner, SHARES_NAMESPACE)
    link_id = meta.key
    assert view(studio, token).status_code == 200
    studio.delete(f"/api/studio/projects/{plan['id']}/share", headers=csrf_headers(studio))
    assert_invalid(view(studio, token))  # an invalid link reaches no plan: no event

    events = studio.app.state.access_log.events()
    assert [event["action"] for event in events] == [
        "create_plan",
        "create_share_link",
        "view_shared_plan",
        "revoke_share_links",
    ]
    created, viewed, revoked = events[1:]
    assert created["details"]["share_link_id"] == link_id and created["details"]["expires_in_days"] == 7
    assert created["actor"]["principal_id"] == owner_digest(owner) == revoked["actor"]["principal_id"]
    assert viewed["actor"] == {"principal_id": f"share_link:{link_id}", "role": "unknown", "role_source": "claimed"}
    assert viewed["resource"] == {"kind": "studio_plan", "id": plan["id"]}
    assert viewed["details"] == {"actor_kind": "share_link"}
    text = studio.app.state.access_log.path.read_text(encoding="utf-8")
    assert owner not in text and "founder_a" not in text and token not in text and "testclient" not in text


def test_a_view_is_refused_when_its_event_cannot_be_written(
    studio: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]

    def failing_append(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OSError("No space left on device (test)")

    monkeypatch.setattr(studio.app.state.access_log.audit, "append", failing_append)
    response = view(studio, token)
    assert response.status_code == 503 and "SYNTHETIC shared plan" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"  # every answer of the route: 200, 404 and 503


def test_the_request_log_never_holds_a_token(studio: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    plan = create_plan(studio)
    token = share(studio, plan["id"])["token"]
    with caplog.at_level(logging.INFO, logger="uwcopilot.web.access"):
        view(studio, token)
        studio.get("/share", params={"t": token})
    lines = [record.getMessage() for record in caplog.records if record.name == "uwcopilot.web.access"]
    assert any(line.startswith("GET /api/share/* 200") for line in lines)
    assert any(line.startswith("GET /share?t=* ") for line in lines)
    assert all(token not in line and token[:40] not in line for line in lines)


def test_the_share_route_is_rate_limited(make_settings) -> None:
    settings = make_settings(UWC_RATE_LIMIT_IP_BURST="2", UWC_RATE_LIMIT_IP_PER_MIN="1")
    with TestClient(create_app(settings)) as client:
        answers = [client.get("/api/share/not-a-token").status_code for _ in range(3)]
    assert answers == [404, 404, 429]


def test_openapi_lists_the_share_routes(client: TestClient) -> None:
    paths = client.get("/api/openapi.json").json()["paths"]
    assert set(paths["/api/studio/projects/{project_id}/share"]) == {"post", "delete"}
    assert set(paths["/api/share/{token}"]) == {"get"}
