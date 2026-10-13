"""SnapshotFabric: the data fabric over a real metro's snapshot (written by snapshot.py)."""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from uwcopilot.studio.data.cache import sha256_bytes
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
    """A real metro's snapshot as written by snapshot.py: manifest plus the tract table.

    manifest_sha256 / tracts_sha256 are the SHA-256 of the two files as they were read. A same-day rerun of
    `make studio-data` replaces the folder, so results files record these (fingerprint()) to pin the exact data.
    """

    folder: Path
    manifest: dict[str, Any]
    tracts: dict[str, dict[str, Any]]
    manifest_sha256: str = ""
    tracts_sha256: str | None = None

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

    def fingerprint(self) -> dict[str, str | None]:
        """{snapshot_date, manifest_sha256, tracts_sha256}: what a results file records to pin this snapshot."""
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "manifest_sha256": self.manifest_sha256,
            "tracts_sha256": self.tracts_sha256,
        }


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
    manifest_bytes = (chosen / "manifest.json").read_bytes()
    tracts_file = chosen / "tracts.json"
    tracts_bytes = tracts_file.read_bytes() if tracts_file.is_file() else None
    return Snapshot(
        folder=chosen,
        manifest=json.loads(manifest_bytes.decode("utf-8")),
        tracts=json.loads(tracts_bytes.decode("utf-8"))["tracts"] if tracts_bytes is not None else {},
        manifest_sha256=sha256_bytes(manifest_bytes),
        tracts_sha256=sha256_bytes(tracts_bytes) if tracts_bytes is not None else None,
    )


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
            status, share, source, reason, live = self.factor_status(factor, scope, table, inputs)
            rows.append(
                FactorCoverage(
                    factor=factor,
                    scope=scope,
                    weight=weight,
                    status=status,
                    share_of_tracts=round(share, 4),
                    share_basis=share_basis_for(scope, real_metro=True),
                    live_per_site=live,
                    source=source,
                    freshness=snapshot_date,
                    reason=reason,
                )
            )
        return rows

    def factor_status(
        self, factor: str, scope: str, table: dict[str, dict[str, float | None]], inputs: dict[str, FactorValue]
    ) -> tuple[FactorStatus, float, str | None, str, bool]:
        """(status, share of tracts, source, reason, available live per site) for one factor of this metro.

        Road and POI layers are not stored per tract in a snapshot (they are looked up live for one site), so they
        are UNKNOWN for ranking tracts and count 0 here even when the live lookup works (Studio rule G).
        """
        if self.snapshot is None:
            return "UNKNOWN", 0.0, None, self.no_snapshot_reason(), False
        if scope == "tract":
            share = sum(values.get(factor) is not None for values in table.values()) / max(len(table), 1)
            if factor == "daytime_workers":
                return "UNKNOWN", 0.0, None, "daytime population source not built yet", False
            status: FactorStatus = "DATA" if share > 0 else "UNKNOWN"
            reason = f"{share:.0%} of tracts have an ACS estimate"
            return status, share if table else 0.0, "census_acs5", reason, False
        if scope == "road_layer":
            return self.road_layer_status()
        if scope == "metro":
            value = inputs.get(factor)
            if value is None or value.status == "UNKNOWN":
                return "UNKNOWN", 0.0, None, value.reason if value else "not in the snapshot", False
            source_id = value.source.source_id if value.source else None
            return value.status, 1.0 if value.status == "DATA" else 0.0, source_id, value.reason, False
        if factor == "str_comparables":
            return "UNKNOWN", 0.0, None, "Inside Airbnb coverage not verified for this metro", False
        poi_status, detail = self.snapshot.step_status("osm_pois")
        if poi_status == "ok":
            reason = (
                "no per-tract place counts in the snapshot, so tracts cannot be ranked on it (UNKNOWN); "
                "available for one site through a live OpenStreetMap lookup"
            )
            return "UNKNOWN", 0.0, "osm_overpass", reason, True
        return "UNKNOWN", 0.0, None, f"OpenStreetMap layer unavailable ({poi_status}: {detail})", False

    def road_layer_status(self) -> tuple[FactorStatus, float, str | None, str, bool]:
        """Coverage of traffic counts: never per tract in a snapshot; live per site where HPMS has segments."""
        roads = (self.snapshot.manifest.get("layers", {}) if self.snapshot else {}).get("roads") or {}
        with_aadt, total = roads.get("counties_with_aadt") or 0, roads.get("counties_total") or 0
        if with_aadt == 0:
            status, detail = self.snapshot.step_status("hpms_roads") if self.snapshot else ("missing", "")
            reason = f"no HPMS road layer in the snapshot (hpms_roads step: {status}; {detail})"
            return "UNKNOWN", 0.0, None, reason, False
        reason = (
            "no per-tract traffic values in the snapshot, so tracts cannot be ranked on traffic (UNKNOWN); "
            f"available for one site through a live HPMS lookup (segments with AADT in {with_aadt} of {total} counties)"
        )
        return "UNKNOWN", 0.0, "fhwa_hpms", reason, True
