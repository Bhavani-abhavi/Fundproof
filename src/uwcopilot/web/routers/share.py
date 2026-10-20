"""Share links of Founder Studio plans: read-only, expiring, revocable. Token design: services/share_links.py.

POST /api/studio/projects/{project_id}/share  (owner only; X-CSRF-Token)  body {"expires_in_days": 1..30} (default 7;
    the body may be left out) -> 200 {"url": "/share?t=<token>", "token": "<token>", "expires_at": "<ISO 8601 UTC>"}
    The link expires after expires_in_days, or when the plan's retention ends if that is sooner (expires_at says
    which). 404 for an unknown plan id and for another user's plan (as every Studio route); 409 when the plan
    already has 10 working links; 422 for a number of days outside 1..30.
DELETE /api/studio/projects/{project_id}/share  (owner only; X-CSRF-Token) -> 200 {"revoked": 2, "pending": false}
    Every link of the plan stops working at once. "revoked" counts the links that were still working (expired links
    are not counted). "pending": true means the backup dataset copy of a share record (UWC_STORE=hf_dataset) could
    not be deleted yet: the link stops working here at once and the deletion is retried, but if the app restarts
    before the backup deletion goes through, the record comes back from the dataset, so the page must say so and ask
    the owner to revoke again after a restart. Deleting the plan, or all your data, revokes its links too.
GET /api/share/{token}  (NO sign-in; rate limited like every /api route) -> 200 SharedPlanOut
    {"plan_name", "business_type" (display name), "metro_id", "metro_name", "money": {MoneyOut, as in a Project},
     "opening_date", "sites": [{"name", "lat", "lon"}], "expires_at", "disclaimer", "notice"}
    Read-only and minimal: no idea text, no street addresses, nothing about the owner's account.
    A malformed, tampered, expired or revoked token, or one for a deleted plan -> 404 with ONE body for every case:
    {"detail": "This share link is not valid. It may have expired or been revoked."}
    Headers on every answer of this route (200, 404, 503): Cache-Control: no-store and Referrer-Policy: no-referrer.
    Answers made before the route runs (a 429 from the rate limiter) or by the app's error handling (an unexpected
    500) do not carry them; none of those holds plan data or the token.

The page for a link is /share?t=<token>: a query parameter, because the Next.js app is a static export and cannot
pre-render a dynamic path segment. The token is a credential: the access log writes /api/share/* instead of the
path (access_log.py) and the page's query values are never logged.

Access log (web/audit_log.py): creating and revoking links are logged as the owner; every view of a valid link is
logged with the actor "share_link:<link id>" (never an IP address). The event is written before the plan is shown,
and when it cannot be written the request is refused with 503.
"""

from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status

from uwcopilot.web.audit_log import AuditUnavailable, PlanAccessLog, get_access_log
from uwcopilot.web.auth import Principal, require_role
from uwcopilot.web.schemas_privacy import ShareCreated, SharedPlanOut, ShareRequest, SharesRevoked
from uwcopilot.web.schemas_studio import PROJECT_ID_PATTERN
from uwcopilot.web.services import projects
from uwcopilot.web.services.errors import NotFound
from uwcopilot.web.services.http import get_share_links, get_store, no_store
from uwcopilot.web.services.share_links import DEFAULT_EXPIRES_IN_DAYS, ShareLinkInvalid, ShareLinks, new_link_id
from uwcopilot.web.store import Store

router = APIRouter(prefix="/api", tags=["share"])

SignedIn = Annotated[Principal, Depends(require_role())]
StoreDep = Annotated[Store, Depends(get_store)]
AccessLog = Annotated[PlanAccessLog, Depends(get_access_log)]
ShareLinksDep = Annotated[ShareLinks, Depends(get_share_links)]
ProjectId = Annotated[str, Path(pattern=PROJECT_ID_PATTERN)]
SHARE_PAGE = "/share"
SHARE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}
SHARE_LINK_NOT_FOUND = "This share link is not valid. It may have expired or been revoked."


def share_page_url(token: str) -> str:
    """The frontend page that shows a shared plan: /share?t=<token> (the token percent-encoded)."""
    return f"{SHARE_PAGE}?{urlencode({'t': token})}"


@router.post("/studio/projects/{project_id}/share", response_model=ShareCreated, dependencies=[Depends(no_store)])
def create_share_link(
    project_id: ProjectId,
    principal: SignedIn,
    store: StoreDep,
    links: ShareLinksDep,
    access_log: AccessLog,
    body: ShareRequest | None = None,
) -> ShareCreated:
    """Make a read-only link to one of your plans."""
    owner = principal.owner_id
    projects.load(store, owner, project_id)  # 404 for another user's plan: nobody can share a plan they do not own
    days = body.expires_in_days if body is not None else DEFAULT_EXPIRES_IN_DAYS
    link_id = new_link_id()
    details = {"share_link_id": link_id, "expires_in_days": days}
    access_log.record_owner_access(principal, "create_share_link", project_id, details)
    meta = projects.meta_for(store, owner, project_id)
    link = links.create(store, owner, project_id, days, meta.expires_at if meta else None, link_id=link_id)
    return ShareCreated(url=share_page_url(link.token), token=link.token, expires_at=link.expires_at)


@router.delete("/studio/projects/{project_id}/share", response_model=SharesRevoked, dependencies=[Depends(no_store)])
def revoke_share_links(
    project_id: ProjectId, principal: SignedIn, store: StoreDep, links: ShareLinksDep, access_log: AccessLog
) -> SharesRevoked:
    """Stop every link to one of your plans (pending: the backup copy of a record is not deleted yet)."""
    projects.load(store, principal.owner_id, project_id)
    access_log.record_owner_access(principal, "revoke_share_links", project_id)
    revoked, pending = links.revoke(store, principal.owner_id, project_id)
    return SharesRevoked(revoked=revoked, pending=pending)


@router.get(
    "/share/{token}",
    response_model=SharedPlanOut,
    responses={status.HTTP_404_NOT_FOUND: {"description": SHARE_LINK_NOT_FOUND}},
)
def view_shared_plan(
    token: str, response: Response, store: StoreDep, links: ShareLinksDep, access_log: AccessLog
) -> SharedPlanOut:
    """The plan behind a share link, read-only (no sign-in). Every invalid link gets the same 404."""
    try:
        link = links.resolve(store, token)
        record = projects.load(store, link.owner, link.project_id)
    except (ShareLinkInvalid, NotFound):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=SHARE_LINK_NOT_FOUND, headers=SHARE_HEADERS
        ) from None
    try:
        access_log.record_share_view(link.link_id, link.project_id)
    except AuditUnavailable as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error.message, headers=SHARE_HEADERS
        ) from None
    response.headers.update(SHARE_HEADERS)
    return projects.shared_plan_out(record, link.expires_at)
