"""Your data: delete everything you stored, in one request.

DELETE /api/me/data  (signed in; X-CSRF-Token) -> 200 {"deleted_objects": 3, "pending": false, "signed_out": true}
    Deletes every object stored under your account with store.delete_owner: your plans, your DEMO decision records
    and your share-link records (so every link you made stops working). Drops your results kept in memory (site
    scores, simulations, path searches), writes an access event (web/audit_log.py), then ends the session: the
    session cookie is cleared, so the next request is signed out.
    "pending": true means the copy in the private backup dataset (UWC_STORE=hf_dataset) could not be deleted yet:
    it is deleted here and the dataset deletion is retried automatically, but until then the data is not fully gone,
    and the page must say so. 401 when signed out; 403 without the CSRF token; 503 (and nothing deleted) when the
    access event cannot be written.
    A plan create that was still waiting for its address lookups when the deletion ran saves nothing and answers 409
    (services/deletions.py, with its limits: a deletion in the last milliseconds before a save is not caught).

What stays after "Delete all my data" (and after retention deletion): the plan access log (UWC_WEB_AUDIT_LOG,
web/audit_log.py) keeps its events about you. They hold your owner digest (the first 32 hex characters of SHA-256 of
your account id, so anyone who knows the account id can link them back to it), plan ids, action names and times,
never plan content, names, addresses or IPs. In the hosted Space this log is on the container disk, which does not
survive a restart, unless the operator points UWC_WEB_AUDIT_LOG at persistent storage; there is no automatic deletion
of the access log yet. The CSRF cookie stays (it is not tied to an account).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from uwcopilot.web.audit_log import PlanAccessLog, get_access_log
from uwcopilot.web.auth import Principal, require_role
from uwcopilot.web.cookies import SESSION_COOKIE, clear_cookie
from uwcopilot.web.schemas_privacy import DataDeleted
from uwcopilot.web.services import WebServices, get_services
from uwcopilot.web.services.http import get_store
from uwcopilot.web.store import DeletionPending, Store

router = APIRouter(prefix="/api", tags=["account"])

SignedIn = Annotated[Principal, Depends(require_role())]
Services = Annotated[WebServices, Depends(get_services)]
StoreDep = Annotated[Store, Depends(get_store)]
AccessLog = Annotated[PlanAccessLog, Depends(get_access_log)]


@router.delete("/me/data", response_model=DataDeleted)
def delete_my_data(
    request: Request, principal: SignedIn, services: Services, store: StoreDep, access_log: AccessLog
) -> JSONResponse:
    """Delete every object you stored, forget your results, and sign you out."""
    owner = principal.owner_id
    access_log.record_owner_access(principal, "delete_all_data")
    services.deletions.record(owner)  # a create still waiting for address lookups now saves nothing (409)
    services.compute.forget(("studio", owner))
    try:
        deleted_objects, pending = store.delete_owner(owner), False
    except DeletionPending as error:
        deleted_objects, pending = error.deleted_objects, True
    body = DataDeleted(deleted_objects=deleted_objects, pending=pending, signed_out=True)
    response = JSONResponse(body.model_dump(), headers={"Cache-Control": "no-store"})
    clear_cookie(response, SESSION_COOKIE, request.app.state.cookie_policy)
    return response
