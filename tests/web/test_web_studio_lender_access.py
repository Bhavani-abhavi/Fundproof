"""Sign-in and CSRF rules of EVERY Founder Studio and Lender Desk route: one table, checked against the routes the
app really serves.

The rules (docstrings of src/uwcopilot/web/routers/studio.py and lender.py, src/uwcopilot/web/security.py):
1. Signed out -> 401 "sign in required". The two public Studio routes (the options and the tract map layer) hold
   no personal data and answer 200 to anyone.
2. A state-changing request (POST, PUT, DELETE) whose X-CSRF-Token header does not match the uwc_csrf cookie -> 403
   from the CSRF middleware, before any route code runs, so nothing in the store changes. Signed in or not.
3. The control: a signed-in user who sends the token reaches the route itself (200, 201 or 404, never 401 or 403).
   GET routes need no token.

ROUTES must list exactly the routes the app serves under /api/studio and /api/lender. A new route without a row
fails test_the_table_lists_every_route_the_app_serves, and every row is then checked by the tests below.

Every plan is SYNTHETIC test input on the SYNTHETIC metro; the lender files are the repo's SYNTHETIC demo
applications and the decision record is a SYNTHETIC placeholder written straight into the store. Where a route
would do heavy work (the simulation, the path search, the agent run), the control uses an id that does not exist,
so the route answers 404 after the access checks. No network.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from web_helpers import csrf_headers, dev_login

from uwcopilot.studio.siteiq.context import metro_context
from uwcopilot.web.app import create_app
from uwcopilot.web.cookies import CSRF_COOKIE
from uwcopilot.web.routers import lender as lender_router
from uwcopilot.web.routers import studio as studio_router
from uwcopilot.web.security import UNSAFE_METHODS
from uwcopilot.web.services.lender_demo import DECISIONS_NAMESPACE

SYNTHETIC_METRO = "synthetic_metro"
CHECKED_PREFIXES = ("/api/studio", "/api/lender")
SIGN_IN_REQUIRED = "sign in required"
CSRF_REFUSAL = "CSRF token missing or invalid"
DEMO_FILE = "APP-C-001"  # a SYNTHETIC demo application that exists
GUIDELINE_SECTION = "NBC-4.2"  # a section of the SYNTHETIC guideline
# Ids for the control requests: none of these objects exist, so the routes stop at 404 after the access checks.
MISSING_IDS = {
    "project_id": "0123456789abcdef",
    "file_id": "APP-C-99999",
    "section_id": "NBC-99.9",
    "metro_id": SYNTHETIC_METRO,
}


def synthetic_sites(count: int = 2) -> list[dict[str, Any]]:
    """Map pins on tract centroids of the SYNTHETIC city."""
    tracts = metro_context(SYNTHETIC_METRO).tracts
    return [
        {"label": f"Site {index + 1}", "lat": float(tracts.lats[index]), "lon": float(tracts.lons[index])}
        for index in range(count)
    ]


def plan_body() -> dict[str, Any]:
    """A SYNTHETIC laundromat plan (test inputs, not benchmarks)."""
    return {
        "metro_id": SYNTHETIC_METRO,
        "business_type_id": "laundromat",
        "budget_total": 400_000.0,
        "owner_equity": 150_000.0,
        "loan_amount": 350_000.0,
        "loan_term_months": 120,
        "annual_rate_pct": 9.0,
        "sites": synthetic_sites(),
    }


@dataclass(frozen=True)
class RouteCase:
    """One route and how these tests call it.

    path is the route's template exactly as FastAPI lists it; body is a valid JSON body when the route takes one;
    allowed_status is what a signed-in user who sends the CSRF token gets for the ids in MISSING_IDS.
    """

    method: str
    path: str
    public: bool = False
    body: dict[str, Any] | None = None
    allowed_status: int = 404

    @property
    def changes_state(self) -> bool:
        """True for POST, PUT and DELETE: the CSRF middleware guards them."""
        return self.method in UNSAFE_METHODS

    def url(self, ids: dict[str, str]) -> str:
        """The path with its {placeholders} filled in."""
        return self.path.format(**ids)

    def label(self) -> str:
        """The test id, e.g. 'PUT /api/studio/projects/{project_id}'."""
        return f"{self.method} {self.path}"


DECISION_BODY = {"action": "refer", "rationale": "Reviewed the SYNTHETIC demo file in an access test."}

ROUTES: tuple[RouteCase, ...] = (
    # Founder Studio
    RouteCase("GET", "/api/studio/options", public=True, allowed_status=200),
    RouteCase("POST", "/api/studio/projects", body=plan_body(), allowed_status=201),
    RouteCase("GET", "/api/studio/projects", allowed_status=200),
    RouteCase("GET", "/api/studio/projects/{project_id}"),
    RouteCase("PUT", "/api/studio/projects/{project_id}", body={"loan_term_months": 96}),
    RouteCase("DELETE", "/api/studio/projects/{project_id}"),
    RouteCase("POST", "/api/studio/projects/{project_id}/sites/score"),
    RouteCase("POST", "/api/studio/projects/{project_id}/feasibility", body={"site_index": 0}),
    RouteCase("POST", "/api/studio/projects/{project_id}/path-to-approval", body={"site_index": 0}),
    # An unknown metro, so the control does not rank tracts: the route itself answers 422 "unknown metro".
    RouteCase(
        "POST",
        "/api/studio/areas/suggest",
        body={"metro_id": "atlantis", "business_type_id": "laundromat"},
        allowed_status=422,
    ),
    RouteCase("GET", "/api/studio/metros/{metro_id}/tracts.geojson", public=True, allowed_status=200),
    # Lender Desk (DEMO)
    RouteCase("GET", "/api/lender/queue", allowed_status=200),
    RouteCase("GET", "/api/lender/files/{file_id}"),
    RouteCase("GET", "/api/lender/sections/{section_id}"),
    RouteCase("POST", "/api/lender/files/{file_id}/decision", body=DECISION_BODY),
    RouteCase("GET", "/api/lender/files/{file_id}/record"),
    RouteCase("DELETE", "/api/lender/files/{file_id}/record"),
)
STATE_CHANGING = tuple(case for case in ROUTES if case.changes_state)


def served_routes(client: TestClient) -> set[tuple[str, str]]:
    """(method, path) of every route under /api/studio and /api/lender: each one the app's OpenAPI document lists,
    plus every route of the two feature routers (so a route left out of the document is still found)."""
    found: set[tuple[str, str]] = set()
    for path, operations in client.get("/api/openapi.json").json()["paths"].items():
        if path.startswith(CHECKED_PREFIXES):
            found.update((method.upper(), path) for method in operations)
    for router in (studio_router.router, lender_router.router):
        for route in router.routes:
            if isinstance(route, APIRoute):
                found.update((method, route.path) for method in route.methods)
    return found


def send(client: TestClient, case: RouteCase, ids: dict[str, str], headers: dict[str, str] | None = None) -> Any:
    """Call the route of `case` with its body (if any) and the given headers."""
    return client.request(case.method, case.url(ids), json=case.body, headers=headers or {})


def store_files(store_dir: Path) -> dict[str, bytes]:
    """Every file of the local store, by relative path (to show that a refused request changed nothing)."""
    if not store_dir.is_dir():
        return {}
    return {str(path.relative_to(store_dir)): path.read_bytes() for path in store_dir.rglob("*") if path.is_file()}


@pytest.fixture
def app_client(make_settings) -> Iterator[TestClient]:
    """A dev-login app, nobody signed in yet."""
    with TestClient(create_app(make_settings(UWC_DEV_AUTH="1"))) as client:
        yield client


@pytest.fixture
def existing(app_client: TestClient) -> dict[str, str]:
    """Signs in founder A and gives them a plan and a DEMO decision record (both SYNTHETIC). Returns the ids of
    objects that exist, so a refused request is refused for a real object, not for a missing one."""
    me = dev_login(app_client, "founder_a", "founder")
    response = app_client.post("/api/studio/projects", json=plan_body(), headers=csrf_headers(app_client))
    assert response.status_code == 201, response.text
    owner = f"{me['auth_mode']}:{me['subject']}"
    placeholder = {"note": "SYNTHETIC placeholder decision record written by tests/web/test_web_studio_lender_access"}
    app_client.app.state.store.put_json(
        owner,
        DECISIONS_NAMESPACE,
        DEMO_FILE,
        {"label": "DEMO", "data_label": "SYNTHETIC", "file_id": DEMO_FILE, "record": placeholder},
    )
    return {
        "project_id": response.json()["id"],
        "file_id": DEMO_FILE,
        "section_id": GUIDELINE_SECTION,
        "metro_id": SYNTHETIC_METRO,
    }


# --- the table matches the app ---


def test_the_table_lists_every_route_the_app_serves(client: TestClient) -> None:
    table = {(case.method, case.path) for case in ROUTES}
    served = served_routes(client)
    assert served - table == set(), "add a RouteCase for each new route (and its access tests pass)"
    assert table - served == set(), "remove the RouteCase of a route the app no longer serves"
    assert len(table) == len(ROUTES), "a route is listed twice"


def test_only_the_options_and_the_tract_layer_are_public() -> None:
    public = {case.path for case in ROUTES if case.public}
    assert public == {"/api/studio/options", "/api/studio/metros/{metro_id}/tracts.geojson"}
    assert all(case.method == "GET" for case in ROUTES if case.public)


# --- rule 1: signed out ---


@pytest.mark.parametrize("case", ROUTES, ids=RouteCase.label)
def test_signed_out_requests_get_401_unless_the_route_is_public(
    app_client: TestClient, existing: dict[str, str], case: RouteCase
) -> None:
    signed_out = app_client.post("/api/auth/logout", headers=csrf_headers(app_client))
    assert signed_out.status_code == 200 and app_client.get("/api/me").json()["signed_in"] is False
    # Send a valid CSRF token, so a 401 comes from the sign-in check and not from the CSRF middleware.
    headers = csrf_headers(app_client) if case.changes_state else {}
    response = send(app_client, case, existing, headers)
    if case.public:
        assert response.status_code == 200, response.text
    else:
        assert response.status_code == 401, response.text
        assert response.json() == {"detail": SIGN_IN_REQUIRED}


# --- rule 2: CSRF ---


@pytest.mark.parametrize("variant", ["no_header", "wrong_header", "no_cookie", "signed_out_no_header"])
@pytest.mark.parametrize("case", STATE_CHANGING, ids=RouteCase.label)
def test_state_changing_requests_without_the_csrf_token_get_403_and_change_nothing(
    app_client: TestClient, existing: dict[str, str], tmp_path: Path, case: RouteCase, variant: str
) -> None:
    token = csrf_headers(app_client)["X-CSRF-Token"]  # the uwc_csrf cookie is set
    headers: dict[str, str] = {}
    if variant == "wrong_header":
        headers = {"X-CSRF-Token": "not-" + token}
    elif variant == "no_cookie":
        headers = {"X-CSRF-Token": token}
        app_client.cookies.delete(CSRF_COOKIE)
    elif variant == "signed_out_no_header":
        app_client.post("/api/auth/logout", headers={"X-CSRF-Token": token})
        assert app_client.get("/api/me").json()["signed_in"] is False
    before = store_files(tmp_path / "store")
    assert before, "the fixture stored a plan and a record"

    response = send(app_client, case, existing, headers)

    assert response.status_code == 403, response.text
    assert response.json()["detail"].startswith(CSRF_REFUSAL)
    assert store_files(tmp_path / "store") == before
    assert app_client.app.state.services.compute.pending_jobs() == 0


def test_the_refused_requests_left_the_plan_and_the_record_in_place(
    app_client: TestClient, existing: dict[str, str]
) -> None:
    for case in STATE_CHANGING:
        assert send(app_client, case, existing).status_code == 403, case.label()
    plan = app_client.get(f"/api/studio/projects/{existing['project_id']}")
    assert plan.status_code == 200 and plan.json()["version"] == 1
    assert app_client.get("/api/studio/projects").json()["total"] == 1
    me = app_client.get("/api/me").json()
    records = app_client.app.state.store.list(f"{me['auth_mode']}:{me['subject']}", DECISIONS_NAMESPACE)
    assert [meta.key for meta in records] == [DEMO_FILE]


# --- rule 3: the control ---


@pytest.mark.parametrize("case", ROUTES, ids=RouteCase.label)
def test_a_signed_in_user_with_the_token_reaches_the_route(app_client: TestClient, case: RouteCase) -> None:
    dev_login(app_client, "founder_b", "founder")
    # GET routes are sent WITHOUT the token: reading needs none.
    headers = csrf_headers(app_client) if case.changes_state else {}
    response = send(app_client, case, MISSING_IDS, headers)
    assert response.status_code == case.allowed_status, response.text
