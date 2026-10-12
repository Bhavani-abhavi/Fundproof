"""The data fabric: ONE interface that SiteIQ (S2) and the feasibility simulator (S3) read metro data through.

    from uwcopilot.studio.data.fabric import get_fabric
    fabric = get_fabric("synthetic_metro")          # or a real metro id
    fabric.mode()                                    # ("synthetic" | "full" | "limited", plain-language reason)
    fabric.tract_ids()                               # every tract in the metro
    fabric.tract("SYN000042")                        # TractRecord: centroid, land area, district
    fabric.tract_features("SYN000042")               # {factor: FactorValue} for tract-level factors
    fabric.tract_table()                             # {geoid: {factor: value or None}} for metro-wide percentiles
    fabric.tract_at(lat, lon)                        # tract id of a site (or None outside the metro)
    fabric.tracts_within(lat, lon, radius_miles)     # [(geoid, distance)] for trade-area sums
    fabric.sites_near(lat, lon, radius_miles, tags)  # PoiSearch: POIs matching OSM tag filters, nearest first
    fabric.nearest_road(lat, lon)                    # RoadResult: nearest segment with AADT
    fabric.str_comparables(lat, lon, radius_miles)   # ListingSearch: short-term rental comparables
    fabric.metro_inputs(profile)                     # {commercial_rent, local_wages, prime_rate: FactorValue}
    fabric.coverage(profile)                         # [FactorCoverage]: per factor, data or not, share of tracts

The rules every implementation follows (Studio addendum rules D, E and G):
- a value is never filled silently: no data -> status UNKNOWN with a reason; an author's assumption -> ASSUMPTION;
- every value carries a status (DATA, SYNTHETIC, ASSUMPTION or UNKNOWN), a plain-language reason and, when it came
  from a dataset, a SourceRecord (provenance.py);
- the hidden true demand of the SYNTHETIC city is never reachable through the fabric (validation code reads it from
  synthetic_city directly).

Two implementations:
- SyntheticFabric: the generated SYNTHETIC city (synthetic_city.py). Mode "synthetic". Used by the app, the tests and
  the S2/S3 validations until real snapshots exist.
- SnapshotFabric: a real metro read from its newest snapshot under data/studio/snapshots/<metro id>/ (written by
  `make studio-data METRO=<id>`, see snapshot.py). Without a snapshot every factor is UNKNOWN and the metro runs in
  limited mode. Site-level lookups (roads, POIs, licenses) are made live through a PoliteClient only when one is
  passed in (`live=`); otherwise they are UNKNOWN.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from uwcopilot.studio.data.errors import NotAvailable, StudioDataError
from uwcopilot.studio.data.geo import haversine_miles_array
from uwcopilot.studio.data.http import PoliteClient
from uwcopilot.studio.data.osm import Poi, pois_near, tag_matches
from uwcopilot.studio.data.paths import SNAPSHOTS_DIR
from uwcopilot.studio.data.provenance import SourceRecord
from uwcopilot.studio.data.str_comps import StrListing
from uwcopilot.studio.data.synthetic_city import SyntheticCity, generate_city
from uwcopilot.studio.data.traffic_hpms import RoadSegment, nearest_road, nearest_segment
from uwcopilot.studio.metros.loader import load_metro
from uwcopilot.studio.metros.models import MetroConfig, MetroMode, metro_mode
from uwcopilot.studio.profiles.factors import DEMAND_FORMULAS, LOCATION_FACTORS, METRO_LEVEL_INPUTS
from uwcopilot.studio.profiles.loader import load_all_profiles
from uwcopilot.studio.profiles.models import Profile

FactorStatus = Literal["DATA", "SYNTHETIC", "ASSUMPTION", "UNKNOWN"]
TRACT_FACTORS: tuple[str, ...] = tuple(name for name, item in LOCATION_FACTORS.items() if item.scope == "tract")
# A site is "on" a road when the nearest segment with traffic counts is within this distance (a design choice).
ROAD_SEARCH_MILES = 0.5
SYNTHETIC_PUBLISHER = "Underwriting Copilot project (generated, not a real place)"


class FactorValue(BaseModel):
    """One factor's value with its status, reason and provenance. value None means UNKNOWN."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor: str
    value: float | None = None
    low: float | None = None
    high: float | None = None
    unit: str
    status: FactorStatus
    reason: str = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list, description="Assumptions used to derive the value")
    source: SourceRecord | None = None

    @property
    def known(self) -> bool:
        """True when a value is present (DATA, SYNTHETIC or ASSUMPTION)."""
        return self.value is not None


class TractRecord(BaseModel):
    """Where a tract is. data_label says whether it is a real tract (PUBLIC) or generated (SYNTHETIC)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    geoid: str
    lat: float
    lon: float
    land_area_sq_miles: float | None
    data_label: Literal["PUBLIC", "SYNTHETIC"]
    district: str | None = None


@dataclass(frozen=True)
class PoiSearch:
    """POIs found near a site. status UNKNOWN means the layer is missing, which is different from 'none found'."""

    pois: list[Poi]
    status: FactorStatus
    reason: str
    source: SourceRecord | None = None


@dataclass(frozen=True)
class RoadResult:
    """The nearest road segment with traffic counts, or UNKNOWN with the reason."""

    segment: RoadSegment | None
    distance_miles: float | None
    status: FactorStatus
    reason: str
    source: SourceRecord | None = None


@dataclass(frozen=True)
class ListingSearch:
    """Short-term rental comparables near a site, or UNKNOWN with the reason."""

    listings: list[StrListing]
    status: FactorStatus
    reason: str
    source: SourceRecord | None = None


class FactorCoverage(BaseModel):
    """Whether one factor has data in this metro, from where, how fresh, and for what share of tracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    factor: str
    scope: str
    weight: float = Field(ge=0, description="The profile's weight for this factor (0 for demand-only inputs)")
    status: FactorStatus
    share_of_tracts: float = Field(ge=0, le=1, description="Share of tracts the factor has data for (0 if none)")
    source: str | None
    freshness: str | None
    reason: str


def required_factors(profile: Profile) -> dict[str, float]:
    """{factor: weight} for the profile's weighted factors plus the data factors its demand formula reads."""
    factors = {name: item.weight for name, item in profile.design.location_weights.items()}
    for name in DEMAND_FORMULAS[profile.demand.formula].data_factors:
        factors.setdefault(name, 0.0)
    return factors


def coverage_share(rows: list[FactorCoverage]) -> float:
    """Weighted share of the profile's factors backed by data (DATA or SYNTHETIC); ASSUMPTION and UNKNOWN count 0.

    Demand-only inputs (weight 0) count with the average weight, so a missing demand input still lowers the score.
    """
    if not rows:
        return 0.0
    positive = [row.weight for row in rows if row.weight > 0]
    fallback = sum(positive) / len(positive) if positive else 1.0
    weights = [row.weight if row.weight > 0 else fallback for row in rows]
    covered = [row.share_of_tracts if row.status in ("DATA", "SYNTHETIC") else 0.0 for row in rows]
    return float(sum(w * c for w, c in zip(weights, covered, strict=True)) / sum(weights))


class DataFabric(ABC):
    """The interface. See the module docstring for the method list and the rules."""

    metro: MetroConfig

    @property
    def data_label(self) -> Literal["PUBLIC", "SYNTHETIC"]:
        """PUBLIC for a real metro, SYNTHETIC for the generated one."""
        return self.metro.data_label

    @abstractmethod
    def mode(self, profiles: dict[str, Profile] | None = None) -> tuple[MetroMode, str]:
        """(full | limited | synthetic, reason)."""

    @abstractmethod
    def tract_ids(self) -> list[str]:
        """Every tract id of the metro (empty when no tract layer exists)."""

    @abstractmethod
    def tract(self, geoid: str) -> TractRecord:
        """A tract's position; KeyError for an unknown id."""

    @abstractmethod
    def tract_features(self, geoid: str) -> dict[str, FactorValue]:
        """Tract-level factors (TRACT_FACTORS) for one tract, each with status, reason and provenance."""

    @abstractmethod
    def tract_table(self) -> dict[str, dict[str, float | None]]:
        """{geoid: {tract factor: value or None}} for every tract (for percentile ranks within the metro)."""

    @abstractmethod
    def sites_near(self, lat: float, lon: float, radius_miles: float, tags: list[str]) -> PoiSearch:
        """POIs matching any OSM tag filter within radius_miles, nearest first."""

    @abstractmethod
    def nearest_road(self, lat: float, lon: float) -> RoadResult:
        """The nearest road segment with AADT within ROAD_SEARCH_MILES."""

    @abstractmethod
    def str_comparables(self, lat: float, lon: float, radius_miles: float) -> ListingSearch:
        """Short-term rental comparables within radius_miles."""

    @abstractmethod
    def metro_inputs(self, profile: Profile) -> dict[str, FactorValue]:
        """Metro-level inputs for the simulator: commercial_rent, local_wages and prime_rate."""

    @abstractmethod
    def coverage(self, profile: Profile) -> list[FactorCoverage]:
        """One FactorCoverage per required factor of the profile."""

    def tract_centroids(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(ids, latitudes, longitudes) of every tract, for vectorized distance searches."""
        ids = self.tract_ids()
        records = [self.tract(geoid) for geoid in ids]
        return ids, np.array([item.lat for item in records]), np.array([item.lon for item in records])

    def tracts_within(self, lat: float, lon: float, radius_miles: float) -> list[tuple[str, float]]:
        """[(geoid, distance in miles)] of tracts whose centroid is within radius_miles, nearest first."""
        ids, lats, lons = self.tract_centroids()
        if not ids:
            return []
        distances = haversine_miles_array(lat, lon, lats, lons)
        order = np.argsort(distances)
        return [(ids[i], float(distances[i])) for i in order if distances[i] <= radius_miles]

    def tract_at(self, lat: float, lon: float, max_miles: float = 3.0) -> str | None:
        """The tract whose centroid is nearest the site (None when none is within max_miles).

        Nearest centroid is an approximation of "the tract containing the point"; for a real address the Census
        Geocoder gives the exact tract (geocode.tract_for_point).
        """
        nearest = self.tracts_within(lat, lon, max_miles)
        return nearest[0][0] if nearest else None


@dataclass
class SyntheticFabric(DataFabric):
    """The fabric over the generated SYNTHETIC city."""

    metro: MetroConfig
    city: SyntheticCity
    tract_index: dict[str, int] = field(init=False)
    record: SourceRecord = field(init=False)
    poi_lats: np.ndarray = field(init=False)
    poi_lons: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.tract_index = {tract.geoid: index for index, tract in enumerate(self.city.tracts)}
        self.record = synthetic_record(self.metro, self.city)
        self.poi_lats = np.array([poi.lat for poi in self.city.pois])
        self.poi_lons = np.array([poi.lon for poi in self.city.pois])

    def mode(self, profiles: dict[str, Profile] | None = None) -> tuple[MetroMode, str]:
        return metro_mode(self.metro, None)

    def tract_ids(self) -> list[str]:
        return self.city.tract_ids()

    def tract(self, geoid: str) -> TractRecord:
        item = self.city.tracts[self.tract_index[geoid]]
        return TractRecord(
            geoid=item.geoid,
            lat=item.lat,
            lon=item.lon,
            land_area_sq_miles=item.land_area_sq_miles,
            data_label="SYNTHETIC",
            district=item.district,
        )

    def tract_table(self) -> dict[str, dict[str, float | None]]:
        return {tract.geoid: self.raw_values(tract.geoid) for tract in self.city.tracts}

    def raw_values(self, geoid: str) -> dict[str, float | None]:
        """The tract's factor values (None where suppressed)."""
        item = self.city.tracts[self.tract_index[geoid]]
        return {
            "resident_population": item.resident_population,
            "daytime_workers": item.daytime_workers,
            "households": item.households,
            "renter_households": item.renter_households,
            "median_household_income": item.median_household_income,
        }

    def tract_features(self, geoid: str) -> dict[str, FactorValue]:
        features: dict[str, FactorValue] = {}
        for factor, value in self.raw_values(geoid).items():
            unit = LOCATION_FACTORS[factor].unit
            if value is None:
                features[factor] = FactorValue(
                    factor=factor,
                    unit=unit,
                    status="UNKNOWN",
                    reason="suppressed in the SYNTHETIC published table (as real survey tables suppress small cells)",
                )
            else:
                features[factor] = FactorValue(
                    factor=factor,
                    value=value,
                    unit=unit,
                    status="SYNTHETIC",
                    reason=f"SYNTHETIC tract {geoid} of the generated city",
                    source=self.record,
                )
        return features

    def sites_near(self, lat: float, lon: float, radius_miles: float, tags: list[str]) -> PoiSearch:
        distances = haversine_miles_array(lat, lon, self.poi_lats, self.poi_lons)
        found = []
        for index in np.flatnonzero(distances <= radius_miles):
            poi = self.city.pois[index]
            if any(tag_matches(poi.tags, tag) for tag in tags):
                found.append(replace(poi, distance_miles=float(distances[index])))
        found.sort(key=lambda poi: poi.distance_miles or 0.0)
        reason = f"{len(found)} SYNTHETIC places within {radius_miles:g} miles"
        return PoiSearch(pois=found, status="SYNTHETIC", reason=reason, source=self.record)

    def nearest_road(self, lat: float, lon: float) -> RoadResult:
        try:
            segment, distance = nearest_segment(self.city.roads, lat, lon)
        except NotAvailable as exc:
            return RoadResult(None, None, "UNKNOWN", exc.reason)
        if distance > ROAD_SEARCH_MILES:
            return RoadResult(None, None, "UNKNOWN", f"no SYNTHETIC road within {ROAD_SEARCH_MILES} miles")
        reason = f"SYNTHETIC road {segment.route_name}, {distance:.2f} miles away"
        return RoadResult(segment, distance, "SYNTHETIC", reason, self.record)

    def str_comparables(self, lat: float, lon: float, radius_miles: float) -> ListingSearch:
        lats = np.array([item.lat for item in self.city.str_listings])
        lons = np.array([item.lon for item in self.city.str_listings])
        distances = haversine_miles_array(lat, lon, lats, lons)
        found = [
            replace(item, distance_miles=float(distances[index]))
            for index, item in enumerate(self.city.str_listings)
            if distances[index] <= radius_miles
        ]
        found.sort(key=lambda item: item.distance_miles or 0.0)
        reason = f"{len(found)} SYNTHETIC listings within {radius_miles:g} miles"
        return ListingSearch(listings=found, status="SYNTHETIC", reason=reason, source=self.record)

    def metro_inputs(self, profile: Profile) -> dict[str, FactorValue]:
        rent = profile.rent.monthly_rent
        wage = profile.labor.hourly_wage
        rent_low, rent_high = rent.bounds()
        wage_low, wage_high = wage.bounds()
        city = self.city
        note = "SYNTHETIC metro index applied to the profile's ASSUMPTION range"
        return {
            "commercial_rent": FactorValue(
                factor="commercial_rent",
                value=rent.point() * city.rent_index,
                low=rent_low * city.rent_index,
                high=rent_high * city.rent_index,
                unit=rent.unit,
                status="SYNTHETIC",
                reason=f"{note} (rent index {city.rent_index}); a founder's quote replaces it",
                assumptions=[f"profile rent.monthly_rent is an ASSUMPTION: {rent.note}"],
                source=self.record,
            ),
            "local_wages": FactorValue(
                factor="local_wages",
                value=wage.point() * city.wage_index,
                low=wage_low * city.wage_index,
                high=wage_high * city.wage_index,
                unit=wage.unit,
                status="SYNTHETIC",
                reason=f"{note} (wage index {city.wage_index})",
                assumptions=[f"profile labor.hourly_wage is an ASSUMPTION: {wage.note}"],
                source=self.record,
            ),
            "prime_rate": FactorValue(
                factor="prime_rate",
                value=city.prime_rate_pct,
                unit="percent",
                status="SYNTHETIC",
                reason="SYNTHETIC prime rate of the generated city; NOT a published rate",
                source=self.record,
            ),
        }

    def coverage(self, profile: Profile) -> list[FactorCoverage]:
        rows: list[FactorCoverage] = []
        table = self.tract_table()
        freshness = f"generated from seed {self.city.seed}"
        for factor, weight in required_factors(profile).items():
            scope = LOCATION_FACTORS[factor].scope
            if scope == "tract":
                share = sum(values[factor] is not None for values in table.values()) / max(len(table), 1)
                reason = f"{share:.0%} of SYNTHETIC tracts have a value"
            elif scope == "road_layer":
                share = self.road_share()
                reason = f"{share:.0%} of SYNTHETIC tract centroids have a road with AADT within {ROAD_SEARCH_MILES} mi"
            else:
                share = 1.0
                reason = "SYNTHETIC layer generated for the whole metro"
            rows.append(
                FactorCoverage(
                    factor=factor,
                    scope=scope,
                    weight=weight,
                    status="SYNTHETIC",
                    share_of_tracts=round(share, 4),
                    source=f"synthetic_city (seed {self.city.seed})",
                    freshness=freshness,
                    reason=reason,
                )
            )
        return rows

    def road_share(self) -> float:
        """Share of tract centroids with a road segment within ROAD_SEARCH_MILES."""
        hits = 0
        for tract in self.city.tracts:
            hits += self.nearest_road(tract.lat, tract.lon).segment is not None
        return hits / max(len(self.city.tracts), 1)


def synthetic_record(metro: MetroConfig, city: SyntheticCity) -> SourceRecord:
    """The SourceRecord every SYNTHETIC value carries; sha256 fingerprints the generated tract table."""
    table = [[t.geoid, t.lat, t.lon, t.resident_population, t.households, t.median_household_income] for t in city.tracts]
    fingerprint = hashlib.sha256(json.dumps(table, sort_keys=True).encode("utf-8")).hexdigest()
    settings = metro.synthetic
    params = {"seed": str(city.seed)}
    if settings is not None:
        params.update({"n_tracts": str(settings.n_tracts), "extent_miles": str(settings.extent_miles)})
    return SourceRecord(
        source_id="synthetic_city",
        publisher=SYNTHETIC_PUBLISHER,
        dataset="SYNTHETIC city generator (src/uwcopilot/studio/data/synthetic_city.py)",
        url=f"synthetic://{metro.id}",
        params=params,
        retrieved_at=generated_at(),
        sha256=fingerprint,
        license_note="SYNTHETIC data generated by this repository; no real place, person or business",
        data_label="SYNTHETIC",
        origin="synthetic",
        vintage=f"seed {city.seed}",
    )


def generated_at() -> datetime:
    """The time the city was generated (it is regenerated from the seed on every run)."""
    return datetime.now(UTC).replace(microsecond=0)


@dataclass
class Snapshot:
    """A real metro's snapshot as written by snapshot.py: manifest plus the tract table."""

    folder: Path
    manifest: dict[str, Any]
    tracts: dict[str, dict[str, Any]]

    @property
    def snapshot_date(self) -> date:
        """The snapshot's date (its folder name)."""
        return date.fromisoformat(self.manifest["snapshot_date"])

    def sources(self, step: str) -> list[SourceRecord]:
        """The SourceRecords stored for one step of the snapshot."""
        for item in self.manifest.get("steps", []):
            if item.get("name") == step:
                return [SourceRecord.model_validate(record) for record in item.get("sources", [])]
        return []

    def step_status(self, step: str) -> tuple[str, str]:
        """(status, detail) of one step, or ('missing', ...) when the step did not run."""
        for item in self.manifest.get("steps", []):
            if item.get("name") == step:
                return str(item.get("status")), str(item.get("detail", ""))
        return "missing", f"step {step!r} is not in this snapshot"


def latest_snapshot(metro_id: str, snapshots_dir: Path = SNAPSHOTS_DIR, pinned: date | None = None) -> Snapshot | None:
    """The pinned snapshot, or the newest one on disk for the metro; None when there is none."""
    folder = snapshots_dir / metro_id
    if not folder.is_dir():
        return None
    candidates = sorted(path for path in folder.iterdir() if (path / "manifest.json").is_file())
    if pinned is not None:
        candidates = [path for path in candidates if path.name == pinned.isoformat()]
    if not candidates:
        return None
    chosen = candidates[-1]
    manifest = json.loads((chosen / "manifest.json").read_text(encoding="utf-8"))
    tracts_file = chosen / "tracts.json"
    tracts = json.loads(tracts_file.read_text(encoding="utf-8"))["tracts"] if tracts_file.is_file() else {}
    return Snapshot(folder=chosen, manifest=manifest, tracts=tracts)


@dataclass
class SnapshotFabric(DataFabric):
    """The fabric over a real metro's snapshot. Missing layers are UNKNOWN with the reason, never filled."""

    metro: MetroConfig
    snapshot: Snapshot | None
    live: PoliteClient | None = None

    def no_snapshot_reason(self) -> str:
        """Why a factor is UNKNOWN when there is no snapshot."""
        return f"no data snapshot for {self.metro.id} (run `make studio-data METRO={self.metro.id}`); limited mode"

    def mode(self, profiles: dict[str, Profile] | None = None) -> tuple[MetroMode, str]:
        snapshot_date = self.snapshot.snapshot_date if self.snapshot else None
        if snapshot_date is None:
            return metro_mode(self.metro, None, snapshot_date=None)
        shares = [coverage_share(self.coverage(profile)) for profile in (profiles or load_all_profiles()).values()]
        return metro_mode(self.metro, min(shares) if shares else None, snapshot_date=snapshot_date)

    def tract_ids(self) -> list[str]:
        return sorted(self.snapshot.tracts) if self.snapshot else []

    def tract(self, geoid: str) -> TractRecord:
        if self.snapshot is None:
            raise KeyError(geoid)
        item = self.snapshot.tracts[geoid]
        return TractRecord(
            geoid=geoid,
            lat=float(item["lat"]),
            lon=float(item["lon"]),
            land_area_sq_miles=item.get("land_area_sq_miles"),
            data_label="PUBLIC",
        )

    def tract_table(self) -> dict[str, dict[str, float | None]]:
        if self.snapshot is None:
            return {}
        return {geoid: {factor: item.get(factor) for factor in TRACT_FACTORS} for geoid, item in self.snapshot.tracts.items()}

    def tract_features(self, geoid: str) -> dict[str, FactorValue]:
        if self.snapshot is None:
            return {f: FactorValue(factor=f, unit=LOCATION_FACTORS[f].unit, status="UNKNOWN", reason=self.no_snapshot_reason()) for f in TRACT_FACTORS}
        item = self.snapshot.tracts[geoid]
        acs = {record.geography: record for record in self.snapshot.sources("acs_tracts")}
        county_record = acs.get(f"county {geoid[:5]}")
        features: dict[str, FactorValue] = {}
        for factor in TRACT_FACTORS:
            unit = LOCATION_FACTORS[factor].unit
            value = item.get(factor)
            if value is None:
                status, detail = self.snapshot.step_status("acs_tracts")
                reason = (
                    "daytime population source not built yet (ACS tract tables count residents only)"
                    if factor == "daytime_workers"
                    else f"no ACS estimate in the snapshot (acs_tracts step: {status}; {detail})"
                )
                features[factor] = FactorValue(factor=factor, unit=unit, status="UNKNOWN", reason=reason)
            else:
                features[factor] = FactorValue(
                    factor=factor, value=float(value), unit=unit, status="DATA",
                    reason=f"ACS 5-year estimate for tract {geoid}", source=county_record,
                )
        return features

    def sites_near(self, lat: float, lon: float, radius_miles: float, tags: list[str]) -> PoiSearch:
        if self.live is None:
            return PoiSearch([], "UNKNOWN", "OpenStreetMap lookups are live per site and live lookups are off")
        try:
            pois, record = pois_near(self.live, lat, lon, radius_miles, tags)
        except StudioDataError as exc:
            return PoiSearch([], "UNKNOWN", f"OpenStreetMap lookup failed: {exc}")
        return PoiSearch(pois, "DATA", f"{len(pois)} OpenStreetMap places within {radius_miles:g} miles", record)

    def nearest_road(self, lat: float, lon: float) -> RoadResult:
        if self.live is None:
            return RoadResult(None, None, "UNKNOWN", "HPMS road lookups are live per site and live lookups are off")
        best: RoadResult | None = None
        problems: list[str] = []
        for state in self.metro.states:
            try:
                found = nearest_road(self.live, state, lat, lon, ROAD_SEARCH_MILES)
            except StudioDataError as exc:
                problems.append(f"{state}: {exc}")
                continue
            if best is None or found.distance_miles < (best.distance_miles or float("inf")):
                reason = f"HPMS segment {found.segment.route_id}, {found.distance_miles:.2f} miles away"
                best = RoadResult(found.segment, found.distance_miles, "DATA", reason, found.source)
        if best is None:
            return RoadResult(None, None, "UNKNOWN", "; ".join(problems) or "no HPMS segment near the site")
        return best

    def str_comparables(self, lat: float, lon: float, radius_miles: float) -> ListingSearch:
        adapters = [item.adapter for item in self.metro.enabled_enhancements("str_comps")]
        reason = f"short-term rental comparables are UNKNOWN: Inside Airbnb coverage not verified for {adapters or 'this metro'}"
        return ListingSearch([], "UNKNOWN", reason)

    def metro_inputs(self, profile: Profile) -> dict[str, FactorValue]:
        rent = profile.rent.monthly_rent
        rent_low, rent_high = rent.bounds()
        inputs = {
            "commercial_rent": FactorValue(
                factor="commercial_rent", value=rent.point(), low=rent_low, high=rent_high, unit=rent.unit,
                status="ASSUMPTION", reason="no free commercial-rent source; the founder's quote replaces this range",
                assumptions=[f"profile rent.monthly_rent: {rent.note}"],
            ),
        }
        stored = (self.snapshot.manifest.get("metro_inputs", {}) if self.snapshot else {}) or {}
        wage = (stored.get("local_wages") or {}).get(profile.id)
        inputs["local_wages"] = (
            FactorValue.model_validate(wage) if wage else FactorValue(
                factor="local_wages", unit="USD/hour", status="UNKNOWN",
                reason=self.no_snapshot_reason() if self.snapshot is None else "no QCEW wage in the snapshot for this profile",
            )
        )
        prime = stored.get("prime_rate")
        inputs["prime_rate"] = (
            FactorValue.model_validate(prime) if prime else FactorValue(
                factor="prime_rate", unit="percent", status="UNKNOWN",
                reason=self.no_snapshot_reason() if self.snapshot is None else "no FRED observation in the snapshot",
            )
        )
        return inputs

    def coverage(self, profile: Profile) -> list[FactorCoverage]:
        rows: list[FactorCoverage] = []
        table = self.tract_table()
        snapshot_date = self.snapshot.snapshot_date.isoformat() if self.snapshot else None
        inputs = self.metro_inputs(profile) if any(f in METRO_LEVEL_INPUTS for f in required_factors(profile)) else {}
        for factor, weight in required_factors(profile).items():
            scope = LOCATION_FACTORS[factor].scope
            status, share, source, reason = self.factor_status(factor, scope, table, inputs)
            rows.append(
                FactorCoverage(
                    factor=factor, scope=scope, weight=weight, status=status, share_of_tracts=round(share, 4),
                    source=source, freshness=snapshot_date, reason=reason,
                )
            )
        return rows

    def factor_status(
        self, factor: str, scope: str, table: dict[str, dict[str, float | None]], inputs: dict[str, FactorValue]
    ) -> tuple[FactorStatus, float, str | None, str]:
        """(status, share of tracts, source, reason) for one factor of this metro."""
        if self.snapshot is None:
            return "UNKNOWN", 0.0, None, self.no_snapshot_reason()
        if scope == "tract":
            share = sum(values.get(factor) is not None for values in table.values()) / max(len(table), 1)
            if factor == "daytime_workers":
                return "UNKNOWN", 0.0, None, "daytime population source not built yet"
            status: FactorStatus = "DATA" if share > 0 else "UNKNOWN"
            return status, share if table else 0.0, "census_acs5", f"{share:.0%} of tracts have an ACS estimate"
        if scope == "road_layer":
            roads = self.snapshot.manifest.get("layers", {}).get("roads") or {}
            total = roads.get("counties_total") or 0
            share = (roads.get("counties_with_aadt") or 0) / total if total else 0.0
            status = "DATA" if share > 0 else "UNKNOWN"
            reason = f"HPMS segments with AADT in {roads.get('counties_with_aadt', 0)} of {total} counties (live per site)"
            return status, share, "fhwa_hpms", reason
        if scope == "metro":
            value = inputs.get(factor)
            if value is None or value.status == "UNKNOWN":
                return "UNKNOWN", 0.0, None, value.reason if value else "not in the snapshot"
            source_id = value.source.source_id if value.source else None
            return value.status, 1.0 if value.status == "DATA" else 0.0, source_id, value.reason
        if factor == "str_comparables":
            return "UNKNOWN", 0.0, None, "Inside Airbnb coverage not verified for this metro"
        poi_status, detail = self.snapshot.step_status("osm_pois")
        if poi_status == "ok":
            return "DATA", 1.0, "osm_overpass", "OpenStreetMap reachable; queried live per site"
        return "UNKNOWN", 0.0, None, f"OpenStreetMap layer unavailable ({poi_status}: {detail})"


@lru_cache(maxsize=8)
def cached_city(metro_id: str, seed: int, n_tracts: int, lat: float, lon: float, extent: float) -> SyntheticCity:
    """Generate (once per process) the SYNTHETIC city for these settings."""
    return generate_city(metro_id, seed, n_tracts, lat, lon, extent, load_all_profiles())


def synthetic_city_for(metro: MetroConfig, seed: int | None = None) -> SyntheticCity:
    """The SYNTHETIC city of a SYNTHETIC metro config (optionally with another seed, for validation runs)."""
    settings = metro.synthetic
    if settings is None:
        raise ValueError(f"{metro.id} is not a SYNTHETIC metro")
    chosen = settings.seed if seed is None else seed
    return cached_city(metro.id, chosen, settings.n_tracts, settings.origin_lat, settings.origin_lon, settings.extent_miles)


def get_fabric(metro_id: str, live: PoliteClient | None = None, seed: int | None = None) -> DataFabric:
    """The fabric for a metro id: SyntheticFabric for the SYNTHETIC metro, SnapshotFabric for a real one."""
    metro = load_metro(metro_id)
    if metro.data_label == "SYNTHETIC":
        return SyntheticFabric(metro=metro, city=synthetic_city_for(metro, seed))
    return SnapshotFabric(metro=metro, snapshot=latest_snapshot(metro_id, pinned=metro.snapshot_date), live=live)
