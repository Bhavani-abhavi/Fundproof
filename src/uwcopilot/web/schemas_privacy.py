"""Request and response models of the privacy routes: share links (routers/share.py) and "delete all my data"
(routers/account.py). The router docstrings show the JSON.

SharedPlanOut is the whole of what a share link shows. It is built field by field from the stored plan
(services/projects.shared_plan_out), so a new field of the plan is never shared by accident. Left out on purpose
(data minimization): the founder's idea text, street addresses and the geocoder's matched address and source, the
plan's id, version and seed, and everything about the owner's account.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from uwcopilot.web.schemas_studio import MoneyOut, RequestModel
from uwcopilot.web.services.share_links import DEFAULT_EXPIRES_IN_DAYS, MAX_EXPIRES_IN_DAYS


class ShareRequest(RequestModel):
    """POST /api/studio/projects/{id}/share (the body may be left out: 7 days)."""

    expires_in_days: int = Field(default=DEFAULT_EXPIRES_IN_DAYS, ge=1, le=MAX_EXPIRES_IN_DAYS)


class ShareCreated(BaseModel):
    """A new share link. Show the url once; the server cannot show the token again later."""

    url: str = Field(description="The page that shows the shared plan: /share?t=<token>")
    token: str
    expires_at: datetime


class SharesRevoked(BaseModel):
    """DELETE /api/studio/projects/{id}/share."""

    revoked: int = Field(description="How many working links of the plan stopped working (expired links not counted)")
    pending: bool = Field(
        description=(
            "True: the backup dataset copy of a share record could not be deleted yet. The link stops working here at "
            "once; if the app restarts before the backup deletion goes through, revoke again"
        )
    )


class SharedSite(BaseModel):
    """A candidate site as a share link shows it: its name and map position, never a street address."""

    name: str
    lat: float
    lon: float


class SharedPlanOut(BaseModel):
    """GET /api/share/{token}: a read-only view of one plan."""

    plan_name: str | None
    business_type: str = Field(description="The business type's display name, e.g. 'Laundromat'")
    metro_id: str
    metro_name: str
    money: MoneyOut
    opening_date: date
    sites: list[SharedSite]
    expires_at: datetime
    disclaimer: str
    notice: str


class DataDeleted(BaseModel):
    """DELETE /api/me/data."""

    deleted_objects: int
    pending: bool = Field(description="True: the backup dataset copy could not be deleted yet; it is retried")
    signed_out: Literal[True]
