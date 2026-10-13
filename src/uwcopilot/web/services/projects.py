"""Founder Studio plans ("projects"): validation, geocoding, storage per user, versions and deletion.

A project is stored as one encrypted JSON object in the web Store (uwcopilot.web.store): namespace
"studio_projects", key = the project id, owner = the signed-in user's stable owner id (Principal.owner_id). The
Store encrypts it at rest (Fernet) and applies the retention period; a user can only ever read their own objects,
and another user's project id simply is "not found".

Every change (PUT) increases `version`. The simulation seed is derived from (project id, version), so the same
version always gives the same numbers, and the result caches (compute.py) are keyed by the version.

Sites:
- SYNTHETIC metro: map pins only; a pin must lie within 3 miles of a tract centroid of the generated city.
- real metro: an address is geocoded with the Census Geocoder (geocoding.py) and must fall in one of the metro's
  counties (when the county list is verified); a pin is checked against the snapshot's tracts when there are any,
  otherwise it is accepted and the site says "not checked".
Free text is checked by text_checks.py (no SSNs or account numbers; the idea text is screened for injection).
"""

import hashlib
import secrets
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from uwcopilot.studio import DISCLAIMER, SYNTHETIC_LABEL
from uwcopilot.studio.data.errors import NotAvailable, StudioDataError
from uwcopilot.studio.data.provenance import SourceRecord
from uwcopilot.studio.siteiq.context import OWN_TRACT_MAX_MILES
from uwcopilot.web.schemas_studio import (
    IdeaOut,
    MoneyOut,
    ProjectIn,
    ProjectOut,
    ProjectSummary,
    ProjectUpdate,
    RateType,
    SiteIn,
    SiteOut,
)
from uwcopilot.web.services.errors import Conflict, InputError, NotFound, Unavailable
from uwcopilot.web.services.geocoding import Geocoder
from uwcopilot.web.services.studio_options import MetroEntry, metro_entry, profile_for
from uwcopilot.web.services.text_checks import injection_flags, refuse_identifiers
from uwcopilot.web.store import ObjectMeta, ObjectNotFound, Store

PROJECTS_NAMESPACE = "studio_projects"
SCHEMA_VERSION = "studio-project-v1"
# Design choice (labelled ASSUMPTION in the project): an opening date six months after the plan was created.
DEFAULT_OPENING_MONTHS_AHEAD = 6
FLAGGED_NOTE = (
    "Flagged by the injection screen: the text looks like an instruction to a system. It is kept as your own text "
    "but never given to any language model."
)
CLEAN_NOTE = "Your idea, stored as you typed it. v0.1 gives no idea text to any language model."
MONEY_NOTE = "USD as entered. owner_equity + loan_amount - budget_total is the cash reserve at opening."


class StoredSite(BaseModel):
    """A site as stored: where it is and how that was established."""

    model_config = ConfigDict(extra="forbid")

    label: str
    lat: float
    lon: float
    location_source: Literal["map_pin", "census_geocoder"]
    address: str | None = None
    matched_address: str | None = None
    tract_geoid: str | None = None
    location_check: str
    geocode_source: SourceRecord | None = None


class ProjectRecord(BaseModel):
    """The stored project (the user's own data; encrypted at rest by the Store)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["studio-project-v1"] = SCHEMA_VERSION
    id: str
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    name: str | None = None
    metro_id: str
    business_type_id: str
    idea_text: str | None = None
    idea_flags: list[str] = Field(default_factory=list)
    budget_total: float | None = None
    owner_equity: float
    loan_amount: float
    loan_term_months: int
    annual_rate_pct: float | None = None
    rate_type: RateType = "variable"
    opening_date: date
    opening_date_given: bool
    sites: list[StoredSite]

    @property
    def seed(self) -> int:
        """The simulation seed of this version: the first 8 hex digits of SHA-256(id:version)."""
        return int(hashlib.sha256(f"{self.id}:{self.version}".encode()).hexdigest()[:8], 16)


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def default_opening_date(created: datetime) -> date:
    """The first day of the month DEFAULT_OPENING_MONTHS_AHEAD months after created."""
    index = created.year * 12 + created.month - 1 + DEFAULT_OPENING_MONTHS_AHEAD
    return date(index // 12, index % 12 + 1, 1)


def new_project_id() -> str:
    """16 random hex characters."""
    return secrets.token_hex(8)


# ---------------------------------------------------------------------------------------------------------------
# Site placement
# ---------------------------------------------------------------------------------------------------------------


def check_pin(entry: MetroEntry, site: SiteIn) -> StoredSite:
    """A map pin, checked against the metro's tracts when it has any."""
    lat, lon = float(site.lat), float(site.lon)  # type: ignore[arg-type]  # SiteIn guarantees both are set
    context = entry.context
    if len(context.tracts) == 0:
        return StoredSite(
            label=site.label,
            lat=lat,
            lon=lon,
            location_source="map_pin",
            location_check="not checked: no tract data for this metro here (limited mode)",
        )
    index = context.own_tract(lat, lon)
    if index is None:
        raise InputError(
            f"site '{site.label}' is not within {OWN_TRACT_MAX_MILES:g} miles of any tract of {entry.config.name}; "
            "pick a point on the metro's map"
        )
    return StoredSite(
        label=site.label,
        lat=lat,
        lon=lon,
        location_source="map_pin",
        tract_geoid=context.tracts.ids[index],
        location_check=f"within {OWN_TRACT_MAX_MILES:g} miles of a tract centroid of {entry.config.name}",
    )


def check_address(entry: MetroEntry, site: SiteIn, geocoder: Geocoder) -> StoredSite:
    """An address, geocoded (real metros only) and checked against the metro's counties."""
    if entry.config.data_label == SYNTHETIC_LABEL:
        raise InputError("the SYNTHETIC metro has no addresses; pick sites on its map (lat and lon)")
    address = str(site.address)
    try:
        match = geocoder.geocode(address)
    except NotAvailable as error:
        raise InputError(f"site '{site.label}': {error}") from error
    except StudioDataError as error:  # unreachable host, an error answer, an unreadable body
        raise Unavailable(
            "the Census Geocoder could not be reached; try again later or drop a pin on the map instead"
        ) from error
    county = match.state_fips + match.county_fips
    if entry.config.verified:
        if county not in entry.config.county_geoids():
            raise InputError(
                f"site '{site.label}' is in county {county}, which is not one of the counties of "
                f"{entry.config.name} (Census delineation file)"
            )
        check = f"geocoded by the Census Geocoder; county {county} is in {entry.config.name}"
    else:
        check = f"geocoded by the Census Geocoder (county {county}); the metro's county list is not verified"
    return StoredSite(
        label=site.label,
        lat=match.lat,
        lon=match.lon,
        location_source="census_geocoder",
        address=address,
        matched_address=match.matched_address,
        tract_geoid=match.tract_geoid,
        location_check=check,
        geocode_source=match.source,
    )


def place_sites(
    entry: MetroEntry, sites: list[SiteIn], geocoder: Geocoder, known: list[StoredSite] | None = None
) -> list[StoredSite]:
    """Check every site; an address already geocoded in the stored plan (known) is reused, not geocoded again."""
    labels = [site.label.casefold() for site in sites]
    if len(set(labels)) != len(labels):
        raise InputError("every site needs its own label")
    reusable = {item.address: item for item in known or [] if item.address}
    placed = []
    for site in sites:
        refuse_identifiers(site.label, "a site label")
        if site.address is not None and site.address in reusable:
            reused = reusable[site.address].model_copy(update={"label": site.label})
            placed.extend(recheck_stored_sites(entry, [reused]))
        elif site.address is not None:
            placed.append(check_address(entry, site, geocoder))
        else:
            placed.append(check_pin(entry, site))
    return placed


def recheck_stored_sites(entry: MetroEntry, sites: list[StoredSite]) -> list[StoredSite]:
    """Stored sites under a (possibly new) metro: pins are checked again, addresses must be in the metro."""
    checked = []
    for site in sites:
        if site.location_source == "map_pin":
            checked.append(check_pin(entry, SiteIn(label=site.label, lat=site.lat, lon=site.lon)))
            continue
        if entry.config.data_label == SYNTHETIC_LABEL:
            raise InputError("the SYNTHETIC metro has no addresses; pick sites on its map (lat and lon)")
        county = site.geocode_source.geography[:5] if site.geocode_source and site.geocode_source.geography else ""
        if entry.config.verified and county not in entry.config.county_geoids():
            raise InputError(f"site '{site.label}' is not in {entry.config.name}; give its sites again")
        checked.append(site)
    return checked


# ---------------------------------------------------------------------------------------------------------------
# Building and checking a record
# ---------------------------------------------------------------------------------------------------------------


def check_money(record: ProjectRecord) -> None:
    """The plan's sources must cover its build-out budget (the simulator refuses a negative starting cash)."""
    if record.budget_total is not None and record.owner_equity + record.loan_amount < record.budget_total:
        raise InputError(
            "owner_equity plus loan_amount must cover budget_total (the cash reserve at opening cannot be negative)"
        )


def check_text(name: str | None, idea_text: str | None) -> list[str]:
    """Refuse forbidden identifiers in the name and idea; return the idea's injection flags."""
    refuse_identifiers(name, "the plan name")
    refuse_identifiers(idea_text, "the idea text")
    return injection_flags(idea_text)


def build_record(body: ProjectIn, geocoder: Geocoder, now: datetime | None = None) -> ProjectRecord:
    """A new, checked ProjectRecord (version 1) from a POST body. Geocodes addresses (network) for real metros."""
    created = now or utc_now()
    entry = metro_entry(body.metro_id)
    profile_for(body.business_type_id, body.metro_id)
    flags = check_text(body.name, body.idea_text)
    record = ProjectRecord(
        id=new_project_id(),
        version=1,
        created_at=created,
        updated_at=created,
        name=body.name or None,
        metro_id=body.metro_id,
        business_type_id=body.business_type_id,
        idea_text=body.idea_text or None,
        idea_flags=flags,
        budget_total=body.budget_total,
        owner_equity=body.owner_equity,
        loan_amount=body.loan_amount,
        loan_term_months=body.loan_term_months,
        annual_rate_pct=body.annual_rate_pct,
        rate_type=body.rate_type,
        opening_date=body.opening_date or default_opening_date(created),
        opening_date_given=body.opening_date is not None,
        sites=place_sites(entry, body.sites, geocoder),
    )
    check_money(record)
    return record


def updated_record(
    current: ProjectRecord, body: ProjectUpdate, geocoder: Geocoder, now: datetime | None = None
) -> ProjectRecord:
    """The next version of a record with the PUT body's fields applied and checked again."""
    if body.version is not None and body.version != current.version:
        raise Conflict(
            f"the plan changed meanwhile (stored version {current.version}, yours {body.version}); reload it"
        )
    given = body.model_fields_set - {"version"}
    for required in ("metro_id", "business_type_id", "owner_equity", "loan_amount", "loan_term_months", "rate_type"):
        if required in given and getattr(body, required) is None:
            raise InputError(f"{required} cannot be empty")
    values: dict[str, Any] = current.model_dump()
    for field in given - {"sites"}:
        values[field] = getattr(body, field)
    if "opening_date" in given:
        values["opening_date_given"] = body.opening_date is not None
        if body.opening_date is None:
            values["opening_date"] = default_opening_date(current.created_at)
    entry = metro_entry(values["metro_id"])
    profile_for(values["business_type_id"], values["metro_id"])
    values["idea_flags"] = check_text(values["name"], values["idea_text"])
    if "sites" in given and body.sites is not None:
        sites = place_sites(entry, body.sites, geocoder, known=current.sites)
    else:
        sites = recheck_stored_sites(entry, current.sites)
    values.update(sites=sites, version=current.version + 1, updated_at=now or utc_now())
    record = ProjectRecord.model_validate(values)
    check_money(record)
    return record


# ---------------------------------------------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------------------------------------------


def save(store: Store, owner: str, record: ProjectRecord) -> ObjectMeta:
    """Write (or overwrite) the record; a write starts a new retention period."""
    return store.put_json(owner, PROJECTS_NAMESPACE, record.id, record.model_dump(mode="json"))


def load(store: Store, owner: str, project_id: str) -> ProjectRecord:
    """The user's record, or NotFound (also for another user's id and for an expired plan)."""
    try:
        data = store.get_json(owner, PROJECTS_NAMESPACE, project_id)
    except ObjectNotFound as error:
        raise NotFound("no such plan") from error
    return ProjectRecord.model_validate(data)


def meta_for(store: Store, owner: str, project_id: str) -> ObjectMeta | None:
    """The stored object's metadata (retention), or None."""
    for meta in store.list(owner, PROJECTS_NAMESPACE):
        if meta.key == project_id:
            return meta
    return None


def list_records(store: Store, owner: str) -> list[tuple[ProjectRecord, ObjectMeta]]:
    """Every unexpired plan of the user, newest change first."""
    items = []
    for meta in store.list(owner, PROJECTS_NAMESPACE):
        try:
            items.append((load(store, owner, meta.key), meta))
        except NotFound:
            continue
    return sorted(items, key=lambda item: item[0].updated_at, reverse=True)


# ---------------------------------------------------------------------------------------------------------------
# Response shapes
# ---------------------------------------------------------------------------------------------------------------


def site_out(index: int, site: StoredSite) -> SiteOut:
    """A stored site as the API shows it."""
    return SiteOut(index=index, **site.model_dump())


def money_out(record: ProjectRecord) -> MoneyOut:
    """The plan's money with entry statuses."""
    reserve = None
    if record.budget_total is not None:
        reserve = round(record.owner_equity + record.loan_amount - record.budget_total, 2)
    return MoneyOut(
        budget_total=record.budget_total,
        budget_status="USER_STATED" if record.budget_total is not None else "ASSUMPTION",
        owner_equity=record.owner_equity,
        loan_amount=record.loan_amount,
        loan_term_months=record.loan_term_months,
        annual_rate_pct=record.annual_rate_pct,
        annual_rate_status="USER_STATED" if record.annual_rate_pct is not None else "ASSUMPTION",
        rate_type=record.rate_type,
        cash_reserve_at_opening=reserve,
        note=MONEY_NOTE,
    )


def project_out(record: ProjectRecord, meta: ObjectMeta | None) -> ProjectOut:
    """The full project response."""
    entry = metro_entry(record.metro_id)
    profile = profile_for(record.business_type_id, record.metro_id)
    flagged = bool(record.idea_flags)
    return ProjectOut(
        id=record.id,
        version=record.version,
        name=record.name,
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=meta.expires_at if meta else None,
        retention_days=meta.retention_days if meta else None,
        metro_id=record.metro_id,
        metro_name=entry.config.name,
        metro_mode=entry.option.mode,
        data_label=entry.config.data_label,
        business_type_id=record.business_type_id,
        business_type_name=profile.display_name,
        demand_unit=profile.demand_unit,
        idea=IdeaOut(
            text=record.idea_text,
            flagged=flagged,
            flags=record.idea_flags,
            note=FLAGGED_NOTE if flagged else CLEAN_NOTE,
        ),
        money=money_out(record),
        opening_date=record.opening_date,
        opening_date_status="USER_STATED" if record.opening_date_given else "ASSUMPTION",
        sites=[site_out(index, site) for index, site in enumerate(record.sites)],
        disclaimer=DISCLAIMER,
    )


def project_summary(record: ProjectRecord, meta: ObjectMeta | None) -> ProjectSummary:
    """One row of the plan list."""
    entry = metro_entry(record.metro_id)
    profile = profile_for(record.business_type_id, record.metro_id)
    return ProjectSummary(
        id=record.id,
        version=record.version,
        name=record.name,
        metro_id=record.metro_id,
        metro_name=entry.config.name,
        business_type_id=record.business_type_id,
        business_type_name=profile.display_name,
        n_sites=len(record.sites),
        created_at=record.created_at,
        updated_at=record.updated_at,
        expires_at=meta.expires_at if meta else None,
        idea_flagged=bool(record.idea_flags),
    )
