"""The access log of founders' plans: one audit event for every access to a plan, written BEFORE the access.

    access_log = PlanAccessLog(settings.web_audit_log)        # create_app() keeps one on app.state.access_log
    access_log.record_owner_access(principal, "read_plan", project_id)
    access_log.record_share_view(link_id, project_id)

It reuses the repo's hash-chained AuditLog (uwcopilot.security.audit): the same event shape, chain, head anchor, file
lock and redactor, so scripts/verify_audit.py checks it too (pass --log with this file). It is its own FILE, apart
from the agent's audit log (UWC_AUDIT_LOG): UWC_WEB_AUDIT_LOG, by default data/web/audit/plan_access_events.jsonl,
next to the store folder and never inside it, so the dataset store never uploads it. The startup stops when the log
cannot be written (check_writable).

Where it lives in the hosted Space. The log is on the container disk, which does not survive a restart, unless the
operator points UWC_WEB_AUDIT_LOG at persistent storage. On the container disk, every restart starts the log (and its
head anchor) again empty, while the plans come back from the dataset, so the log does not cover the plans' whole
life. With the dataset store and UWC_WEB_AUDIT_LOG unset, the startup logs an error saying so
(settings.read_web_audit_log).

What stays after "Delete all my data" and after retention deletion: the events in this log. They hold the owner
digest, plan ids, action names and times, never plan content, names, addresses or IPs. There is no automatic deletion
of the access log yet.

What is logged. One event per request, for each action below; the exact action is always in `action`:

| action | route | event_type |
|---|---|---|
| list_plans | GET /api/studio/projects | view |
| read_plan | GET /api/studio/projects/{id} | view |
| create_plan | POST /api/studio/projects | view |
| update_plan | PUT /api/studio/projects/{id} | view |
| score_sites, run_feasibility, run_path_to_approval | the three compute routes of a plan | view |
| view_shared_plan | GET /api/share/{token} | view |
| create_share_link, revoke_share_links | POST / DELETE /api/studio/projects/{id}/share | permission_change |
| delete_plan | DELETE /api/studio/projects/{id} | permission_change |
| delete_all_data | DELETE /api/me/data | permission_change |

security/audit.py has no event type for changing or deleting data (its list was written for lender files, and the
agent's log shares it), so event_type is the closest existing type: "view" when the response shows the plan or
results computed from it (create and update answer with the whole plan), "permission_change" when the request
changes who can reach the plan (a link grants or loses read access; a deletion ends every access, links included).
Adding "data_change" and "data_deletion" types to security/audit.py is an open follow-up.

Who. The actor id is the owner digest (store.owner_digest: the first 32 hex characters of SHA-256 of the owner id),
never the account id or the username in clear. The digest is not anonymous: anyone who knows an account id can
compute it and find that account's events. A share-link view has the actor "share_link:<link id>", never an IP
address.
security.rbac has no founder role, so actor.role is "unknown" ("no access role") and details.actor_kind says
"founder" or "share_link". actor.role_source is "dev_login" for a dev session, "session_allowlist" for a Hugging
Face session (docs/security.md section 9), and "claimed" for a link holder (the token proves only that they hold it).

What is not in an event: plan text, names, addresses, money, the owner id itself. Callers pass ids only (the project
id as the resource id, a share link id, a number of days); the AuditLog redactor is a second line of defence, not
the plan. data_label is "USER_DATA": a plan is the founder's own input and may be real, so it is never labelled
SYNTHETIC by assumption.

Failure policy (a decision): every event is written BEFORE the access happens. If it cannot be written (full disk,
damaged log, any error), the access does NOT happen: the request is refused with 503 and a message that holds no plan
data, because an unaudited access must not happen. This holds for deletion too (the founder is asked to try again).
outcome is "allowed": the access was authorized (the plan exists and is the caller's, or the link is valid) and was
then attempted. A request that fails after its event (for example a 422 for an unusable value in an update) changed
nothing, although its event exists. A request refused BEFORE the event (401, 403, 404 for an unknown or another
user's plan id, an invalid share link) did not reach any plan and writes no event.

Recovery. A failed write leaves no half-written line: AuditLog cuts the file back to its size before the append, so
once the cause is gone (for example the disk has room again) the next request works. A torn last line left by a
crash is cut off by the next append when the head anchor shows it was an append that never completed (a warning is
logged). Any other damage keeps every plan route at 503 until an operator runs scripts/verify_audit.py --log
<this file> and decides what to do; nothing repairs it automatically.

Cost and size. An append reads only the end of the file (security/audit.py, "Appending"), so a long log does not slow
requests down. The file is never rotated and nothing deletes old events yet; it grows by one event (a few hundred
bytes) per plan access, bounded only by the rate limits.
"""

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import Request

from uwcopilot.security import rbac
from uwcopilot.security.audit import AuditLog
from uwcopilot.web.auth import Principal
from uwcopilot.web.services.errors import Unavailable
from uwcopilot.web.store import owner_digest

logger = logging.getLogger(__name__)

PLAN_RESOURCE_KIND = "studio_plan"
SHARE_LINK_ACTOR_PREFIX = "share_link:"
DATA_LABEL = "USER_DATA"
OUTCOME = "allowed"
ACTION_EVENT_TYPES: dict[str, str] = {
    "list_plans": "view",
    "read_plan": "view",
    "create_plan": "view",
    "update_plan": "view",
    "score_sites": "view",
    "run_feasibility": "view",
    "run_path_to_approval": "view",
    "view_shared_plan": "view",
    "create_share_link": "permission_change",
    "revoke_share_links": "permission_change",
    "delete_plan": "permission_change",
    "delete_all_data": "permission_change",
}
AUDIT_UNAVAILABLE_MESSAGE = (
    "The access log could not be written, so this request was not carried out. Please try again in a moment."
)


class AuditUnavailable(Unavailable):
    """The access event could not be written, so the access is refused (503)."""


class AuditLogNotWritable(RuntimeError):
    """The access log cannot be written at all (checked at startup, so the app does not start half working)."""


def owner_actor(principal: Principal) -> rbac.Principal:
    """The audit actor of a signed-in founder: their owner digest, never the account id or the username."""
    role_source = "dev_login" if principal.auth_mode == "dev" else "session_allowlist"
    return rbac.Principal(owner_digest(principal.owner_id), rbac.UNKNOWN_ROLE, role_source)


def share_link_actor(link_id: str) -> rbac.Principal:
    """The audit actor of whoever opens a share link: the link id, never an IP address."""
    return rbac.Principal(f"{SHARE_LINK_ACTOR_PREFIX}{link_id}", rbac.UNKNOWN_ROLE, "claimed")


def plan_resource(project_id: str) -> rbac.Resource:
    """The audit resource of one plan (its random id; nothing about its content)."""
    return rbac.Resource(PLAN_RESOURCE_KIND, project_id)


class PlanAccessLog:
    """Writes the access events of founders' plans into one hash-chained AuditLog file."""

    def __init__(self, path: Path) -> None:
        """path is created (with its folder) on the first event."""
        self.audit = AuditLog(path)

    @property
    def path(self) -> Path:
        """The JSON Lines file of the events."""
        return self.audit.path

    def check_writable(self) -> None:
        """Fail fast at startup: create the log's folder, open the log for appending (it is created empty when
        missing) and check that the folder takes new files (the head anchor is written there).

        Without this check, a log the app's user cannot write (for example in a folder it cannot create) is found
        only at the first plan access, and then every plan route answers 503 while /api/health says healthy.
        AuditLogNotWritable names the file and the setting to change.
        """
        folder = self.path.parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab"):
                pass
            if not os.access(folder, os.W_OK | os.X_OK):
                raise PermissionError(f"the folder {folder} does not take new files")
        except OSError as error:
            raise AuditLogNotWritable(
                f"the plan access log {self.path} cannot be written ({error}); set UWC_WEB_AUDIT_LOG to a writable "
                "file outside the store folder"
            ) from error

    def record(
        self,
        actor: rbac.Principal,
        action: str,
        resource: rbac.Resource | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write one event and return it; AuditUnavailable (503) when it cannot be written.

        A ValueError for an action this module does not know is a programming error and is raised as it is.
        """
        event_type = ACTION_EVENT_TYPES.get(action)
        if event_type is None:
            raise ValueError(f"unknown plan access action {action!r}; known actions are {list(ACTION_EVENT_TYPES)}")
        try:
            return self.audit.append(
                event_type, actor, action, resource, outcome=OUTCOME, details=details, data_label=DATA_LABEL
            )
        except Exception as error:
            # The log line names the action only; the error itself goes to the server log, never to the client.
            logger.error("the plan access event %r could not be written; the request is refused", action, exc_info=True)
            raise AuditUnavailable(AUDIT_UNAVAILABLE_MESSAGE) from error

    def record_owner_access(
        self,
        principal: Principal,
        action: str,
        project_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """record() for a founder acting on their own plan (project_id None: on the plan list or on all their data)."""
        resource = plan_resource(project_id) if project_id is not None else None
        return self.record(owner_actor(principal), action, resource, {"actor_kind": "founder", **(details or {})})

    def record_share_view(self, link_id: str, project_id: str) -> dict[str, Any]:
        """record() for someone opening a valid share link."""
        details = {"actor_kind": "share_link"}
        return self.record(share_link_actor(link_id), "view_shared_plan", plan_resource(project_id), details)

    def events(self) -> list[dict[str, Any]]:
        """Every event written so far, in order."""
        return self.audit.events()


def get_access_log(request: Request) -> PlanAccessLog:
    """FastAPI dependency: the app's plan access log."""
    return request.app.state.access_log
