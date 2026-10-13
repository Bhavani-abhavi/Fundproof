"""SnapshotFabric: the data fabric over a real metro's snapshot (written by snapshot.py)."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from uwcopilot.studio.data.errors import StudioDataError
from uwcopilot.studio.data.fabric_base import (
    ROAD_SEARCH_MILES,
    TRACT_FACTORS,
    DataFabric,
    FactorCoverage,
    FactorStatus,
    FactorValue,
    ListingSearch,
    PoiSearch,
    RoadResult,
    TractRecord,
    coverage_share,
    required_factors,
    share_basis_for,
)
from uwcopilot.studio.data.http import PoliteClient
from uwcopilot.studio.data.osm import pois_near
from uwcopilot.studio.data.paths import SNAPSHOTS_DIR
from uwcopilot.studio.data.provenance import SourceRecord
from uwcopilot.studio.data.traffic_hpms import nearest_road
from uwcopilot.studio.metros.models import MetroConfig, MetroMode, metro_mode
from uwcopilot.studio.profiles.factors import LOCATION_FACTORS, METRO_LEVEL_INPUTS
from uwcopilot.studio.profiles.loader import load_all_profiles
from uwcopilot.studio.profiles.models import Profile


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
        return {
            geoid: {factor: item.get(factor) for factor in TRACT_FACTORS}
            for geoid, item in self.snapshot.tracts.items()
        }

    def tract_features(self, geoid: str) -> dict[str, FactorValue]:
        if self.snapshot is None:
            return {
                f: FactorValue(
                    factor=f, unit=LOCATION_FACTORS[f].unit, status="UNKNOWN", reason=self.no_snapshot_reason()
                )
                for f in TRACT_FACTORS
            }
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
                    factor=factor,
                    value=float(value),
                    unit=unit,
                    status="DATA",
                    reason=f"ACS 5-year estimate for tract {geoid}",
                    source=county_record,
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
        where = adapters or "this metro"
        reason = f"short-term rental comparables are UNKNOWN: Inside Airbnb coverage not verified for {where}"
        return ListingSearch([], "UNKNOWN", reason)

    def metro_inputs(self, profile: Profile) -> dict[str, FactorValue]:
        rent = profile.rent.monthly_rent
        rent_low, rent_high = rent.bounds()
        inputs = {
            "commercial_rent": FactorValue(
                factor="commercial_rent",
                value=rent.point(),
                low=rent_low,
                high=rent_high,
                unit=rent.unit,
                status="ASSUMPTION",
                reason="no free commercial-rent source; the founder's quote replaces this range",
                assumptions=[f"profile rent.monthly_rent: {rent.note}"],
            ),
        }
        stored = (self.snapshot.manifest.get("metro_inputs", {}) if self.snapshot else {}) or {}
        wage = (stored.get("local_wages") or {}).get(profile.id)
        inputs["local_wages"] = (
            FactorValue.model_validate(wage)
            if wage
            else FactorValue(
                factor="local_wages",
                unit="USD/hour",
                status="UNKNOWN",
                reason=self.no_snapshot_reason()
                if self.snapshot is None
                else "no QCEW wage in the snapshot for this profile",
            )
        )
        prime = stored.get("prime_rate")
        inputs["prime_rate"] = (
            FactorValue.model_validate(prime)
            if prime
            else FactorValue(
                factor="prime_rate",
                unit="percent",
                status="UNKNOWN",
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
                    factor=factor,
                    scope=scope,
                    weight=weight,
                    status=status,
                    share_of_tracts=round(share, 4),
                    share_basis=share_basis_for(scope, real_metro=True),
                    source=source,
                    freshness=snapshot_date,
                    reason=reason,
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
            reason = (
                f"HPMS segments with AADT in {roads.get('counties_with_aadt', 0)} of {total} counties (live per site)"
            )
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
