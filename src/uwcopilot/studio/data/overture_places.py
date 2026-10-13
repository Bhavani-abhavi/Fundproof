"""Overture Maps places for one metro: competitors, anchors and tourism places as a small, cached point extract.

    extract = extract_metro_places(client, "houston_tx", bbox)
    extract.payload["rows"]   # [[lon, lat, confidence, [tag index], category index, dataset index, license index]]
    extract.record            # SourceRecord: release, query, SHA-256 of the extract, request date, license note

What happens:
1. The release's places files are listed from the public S3 bucket over HTTPS (the polite client caches the
   listing; each file is identified by its key, ETag and size).
2. DuckDB reads the files over HTTPS with predicate pushdown on the bbox columns (duckdb_http.py), twice:
   - path counts: how many places of each category path are in the bounding box, with and without enough
     confidence and by operating status (for the accounting: mapped, unmapped, filtered);
   - places: the coordinates, category, confidence and first source's dataset and license of every place whose
     hierarchy contains a mapped category, above the confidence threshold and not permanently closed.
3. Each place gets its profile tags from the reviewed mapping (overture_config.CategoryMap.tags_for). Places whose
   most specific mapped category is "not counted" are dropped. The number kept must equal the accounting's number.
4. The extract is stored in the response cache (data/studio/cache) with its SHA-256 and the request date, under a key
   made of the release, the files' ETags, the bounding box, the settings and the mapping file's SHA-256, so the same
   extract is reused until one of them changes (or --refresh).

Only category, coordinates, confidence, the dataset name and its license are read; names, ids, addresses, phones
and websites are never read. The reader is injectable (PlacesReader) so tests run on local files without a network.
"""

import json
import re
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from uwcopilot.studio.data import duckdb_http
from uwcopilot.studio.data.cache import CachedResponse, request_key, sha256_bytes
from uwcopilot.studio.data.errors import HostUnreachable, UpstreamError
from uwcopilot.studio.data.geo import miles_to_degrees
from uwcopilot.studio.data.http import FetchResult, PoliteClient
from uwcopilot.studio.data.overture_config import (
    CATEGORY_MAP_FILE,
    CategoryMap,
    OvertureSettings,
    file_sha256,
    load_category_map,
    load_settings,
)
from uwcopilot.studio.data.provenance import SourceRecord

SOURCE_ID = "overture_places"
PUBLISHER = "Overture Maps Foundation"
FORMAT = "studio-overture-places-v1"
COLUMNS: tuple[str, ...] = ("lon", "lat", "confidence", "tags", "category", "dataset", "license")
S3_NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
UNMAPPED_EXAMPLES = 15


@dataclass(frozen=True)
class BBox:
    """A longitude/latitude box in degrees."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def around(cls, lats: Sequence[float], lons: Sequence[float], margin_miles: float) -> "BBox":
        """The box around the points, widened by margin_miles on every side (at the box's northern edge)."""
        north, south = max(lats), min(lats)
        dlat, _ = miles_to_degrees(north, margin_miles, 0.0)
        _, dlon = miles_to_degrees(max(abs(north), abs(south)), 0.0, margin_miles)
        return cls(
            xmin=round(min(lons) - dlon, 6),
            xmax=round(max(lons) + dlon, 6),
            ymin=round(south - dlat, 6),
            ymax=round(north + dlat, 6),
        )

    def as_list(self) -> list[float]:
        """[xmin, xmax, ymin, ymax]."""
        return [self.xmin, self.xmax, self.ymin, self.ymax]

    def text(self) -> str:
        """'xmin,ymin,xmax,ymax' for SourceRecord params and the cache key."""
        return f"{self.xmin:.6f},{self.ymin:.6f},{self.xmax:.6f},{self.ymax:.6f}"


@dataclass(frozen=True)
class ReleaseFile:
    """One Parquet file of the release, as the bucket listing describes it."""

    key: str
    etag: str
    size: int

    def url(self, base: str) -> str:
        """The file's HTTPS URL."""
        return f"{base}/{self.key}"


@dataclass(frozen=True)
class PathCount:
    """How many places in the box share one category path, confidence flag and operating status."""

    hierarchy: tuple[str, ...] | None
    confident: bool
    status: str | None
    count: int


@dataclass(frozen=True)
class PlaceRow:
    """One place as read (no name, no id, no address)."""

    hierarchy: tuple[str, ...]
    lon: float
    lat: float
    confidence: float
    dataset: str | None
    license: str | None


class PlacesReader(Protocol):
    """Reads places from the release files; DuckDbPlacesReader in production, a local-file reader in tests."""

    def path_counts(self, files: Sequence[str], bbox: BBox, min_confidence: float) -> list[PathCount]:
        """Counts per (category path, confidence >= min_confidence, operating status) inside the box."""

    def places(
        self,
        files: Sequence[str],
        bbox: BBox,
        categories: Sequence[str],
        min_confidence: float,
        excluded_status: Sequence[str],
    ) -> list[PlaceRow]:
        """Places inside the box whose hierarchy contains one of the categories, filtered by confidence and status."""


BOX_FILTER = "bbox.xmin BETWEEN ? AND ? AND bbox.ymin BETWEEN ? AND ?"
PATH_COUNTS_SQL = (
    "SELECT taxonomy.hierarchy AS hierarchy, coalesce(confidence >= ?, false) AS confident, "
    "operating_status AS status, count(*) AS n "
    f"FROM read_parquet(?) WHERE {BOX_FILTER} GROUP BY ALL"
)
PLACES_SQL = (
    "SELECT taxonomy.hierarchy AS hierarchy, bbox.xmin AS lon, bbox.ymin AS lat, confidence, "
    "sources[1].dataset AS dataset, sources[1].license AS license "
    f"FROM read_parquet(?) WHERE {BOX_FILTER} "
    "AND confidence >= ? AND (operating_status IS NULL OR NOT list_contains(?, operating_status)) "
    "AND list_has_any(taxonomy.hierarchy, ?)"
)


@dataclass
class DuckDbPlacesReader:
    """Reads the release's Parquet files with DuckDB (over HTTPS for URLs, directly for local paths)."""

    def connection(self, files: Sequence[str]) -> Any:
        """A DuckDB connection, with httpfs and the proxy set up when any file is an https:// URL."""
        return duckdb_http.connect(https=any(name.startswith("https://") for name in files))

    def path_counts(self, files: Sequence[str], bbox: BBox, min_confidence: float) -> list[PathCount]:
        connection = self.connection(files)
        try:
            rows = connection.execute(PATH_COUNTS_SQL, [min_confidence, list(files), *box_params(bbox)]).fetchall()
        finally:
            connection.close()
        return [PathCount(as_path(row[0]), bool(row[1]), row[2], int(row[3])) for row in rows]

    def places(
        self,
        files: Sequence[str],
        bbox: BBox,
        categories: Sequence[str],
        min_confidence: float,
        excluded_status: Sequence[str],
    ) -> list[PlaceRow]:
        params = [list(files), *box_params(bbox), min_confidence, list(excluded_status), list(categories)]
        connection = self.connection(files)
        try:
            rows = connection.execute(PLACES_SQL, params).fetchall()
        finally:
            connection.close()
        return [
            PlaceRow(as_path(row[0]) or (), float(row[1]), float(row[2]), float(row[3]), row[4], row[5])
            for row in rows
        ]


def box_params(bbox: BBox) -> list[float]:
    """The four values of BOX_FILTER, in order."""
    return [bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax]


def as_path(value: Any) -> tuple[str, ...] | None:
    """A hierarchy list as a tuple (None when the place has no taxonomy)."""
    if not value:
        return None
    return tuple(str(item) for item in value)


def check_listing(fetched: FetchResult) -> None:
    """Accept an S3 listing only when it is a ListBucketResult XML document without a DOCTYPE (an http.Accept)."""
    text = fetched.body[:2000].decode("utf-8", errors="replace")
    if "<!DOCTYPE" in text or "<!ENTITY" in text:
        raise UpstreamError("overturemaps-us-west-2", "listing has a DOCTYPE or ENTITY declaration; refused")
    if "<ListBucketResult" not in text:
        raise UpstreamError("overturemaps-us-west-2", "answer is not an S3 bucket listing")


def parse_listing(body: bytes) -> tuple[list[ReleaseFile], str | None]:
    """(Parquet files, continuation token or None) from one ListObjectsV2 page."""
    root = ElementTree.fromstring(body)
    files = []
    for item in root.findall(f"{S3_NAMESPACE}Contents"):
        key = item.findtext(f"{S3_NAMESPACE}Key") or ""
        if key.endswith(".parquet"):
            etag = (item.findtext(f"{S3_NAMESPACE}ETag") or "").strip('"')
            files.append(ReleaseFile(key=key, etag=etag, size=int(item.findtext(f"{S3_NAMESPACE}Size") or 0)))
    truncated = (root.findtext(f"{S3_NAMESPACE}IsTruncated") or "false").lower() == "true"
    token = root.findtext(f"{S3_NAMESPACE}NextContinuationToken") if truncated else None
    return files, token


def list_release_files(client: PoliteClient, settings: OvertureSettings) -> tuple[list[ReleaseFile], list[FetchResult]]:
    """Every places Parquet file of the pinned release (following continuation tokens), and the listing answers."""
    release = settings.release
    files: list[ReleaseFile] = []
    answers: list[FetchResult] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": release.places_prefix}
        if token:
            params["continuation-token"] = token
        fetched = client.get(f"{release.s3_https_base}/", params, accept=check_listing)
        answers.append(fetched)
        page, token = parse_listing(fetched.body)
        files += page
        if not token:
            break
    if not files:
        raise UpstreamError("overturemaps-us-west-2", f"no places files under {release.places_prefix}")
    return sorted(files, key=lambda item: item.key), answers


@dataclass
class Accounting:
    """Where every place in the box went: kept, filtered or unmapped (the counts go into the snapshot step)."""

    places_in_bbox: int = 0
    without_taxonomy: int = 0
    unmapped: int = 0
    not_counted_by_row: int = 0
    mapped: int = 0
    mapped_below_confidence: int = 0
    mapped_excluded_status: int = 0
    kept: int = 0
    unmapped_categories: Counter = field(default_factory=Counter)

    def as_json(self) -> dict[str, Any]:
        """The counts, plus the number of distinct unmapped categories and the most frequent ones."""
        return {
            "places_in_bbox": self.places_in_bbox,
            "without_taxonomy": self.without_taxonomy,
            "unmapped": self.unmapped,
            "unmapped_distinct_categories": len(self.unmapped_categories),
            "unmapped_top_categories": dict(self.unmapped_categories.most_common(UNMAPPED_EXAMPLES)),
            "not_counted_by_row": self.not_counted_by_row,
            "mapped": self.mapped,
            "mapped_below_confidence": self.mapped_below_confidence,
            "mapped_excluded_status": self.mapped_excluded_status,
            "kept": self.kept,
        }


def account(counts: Sequence[PathCount], cmap: CategoryMap, excluded_status: Sequence[str]) -> Accounting:
    """Sort every counted place into kept / filtered / not counted / unmapped / no taxonomy."""
    result = Accounting()
    for item in counts:
        result.places_in_bbox += item.count
        if not item.hierarchy:
            result.without_taxonomy += item.count
            continue
        tags = cmap.tags_for(item.hierarchy)
        if tags is None:
            result.unmapped += item.count
            result.unmapped_categories[item.hierarchy[-1]] += item.count
        elif not tags:
            result.not_counted_by_row += item.count
        else:
            result.mapped += item.count
            if not item.confident:
                result.mapped_below_confidence += item.count
            elif item.status in excluded_status:
                result.mapped_excluded_status += item.count
            else:
                result.kept += item.count
    return result


@dataclass(frozen=True)
class TaggedPlace:
    """A kept place with its profile tags."""

    lon: float
    lat: float
    confidence: float
    tags: tuple[str, ...]
    category: str
    dataset: str
    license: str


def tag_places(rows: Sequence[PlaceRow], cmap: CategoryMap) -> list[TaggedPlace]:
    """The rows that the mapping counts, with their tags (rows whose deciding row has no tags are dropped)."""
    kept = []
    for row in rows:
        tags = cmap.tags_for(row.hierarchy)
        if tags:
            kept.append(
                TaggedPlace(
                    lon=round(row.lon, 6),
                    lat=round(row.lat, 6),
                    confidence=round(row.confidence, 3),
                    tags=tags,
                    category=row.hierarchy[-1],
                    dataset=row.dataset or "unknown",
                    license=row.license or "unknown",
                )
            )
    kept.sort(key=lambda item: (item.lat, item.lon, item.category, item.confidence))
    return kept


def index_of(table: list[str], value: str) -> int:
    """Position of value in a string table, appending it when new."""
    if value not in table:
        table.append(value)
    return table.index(value)


def build_payload(
    places: Sequence[TaggedPlace],
    accounting: Accounting,
    settings: OvertureSettings,
    cmap: CategoryMap,
    bbox: BBox,
    map_sha256: str,
) -> dict[str, Any]:
    """The extract as stored: string tables plus one compact row per place."""
    tags: list[str] = sorted(cmap.mapped_tags())
    categories: list[str] = []
    datasets: list[str] = []
    licenses: list[str] = []
    rows = [
        [
            place.lon,
            place.lat,
            place.confidence,
            sorted(tags.index(tag) for tag in place.tags),
            index_of(categories, place.category),
            index_of(datasets, place.dataset),
            index_of(licenses, place.license),
        ]
        for place in places
    ]
    return {
        "format": FORMAT,
        "data_label": "PUBLIC",
        "release": settings.release.id,
        "schema_version": settings.release.schema_version,
        "bbox": bbox.as_list(),
        "min_confidence": settings.extract.min_confidence.value,
        "exclude_operating_status": list(settings.extract.exclude_operating_status.values),
        "category_map_sha256": map_sha256,
        "mapped_tags": tags,
        "unmapped_profile_tags": {item.tag: item.reason for item in cmap.unmapped_profile_tags},
        "accounting": accounting.as_json(),
        "kept_by_license": dict(Counter(place.license for place in places)),
        "kept_by_dataset": dict(Counter(place.dataset for place in places)),
        "tags": tags,
        "categories": categories,
        "datasets": datasets,
        "licenses": licenses,
        "columns": list(COLUMNS),
        "rows": rows,
    }


@dataclass(frozen=True)
class PlacesExtract:
    """One metro's extract: the payload (as stored), its bytes' SHA-256, provenance and the files read."""

    payload: dict[str, Any]
    body: bytes
    record: SourceRecord
    files: list[ReleaseFile]
    listing: list[FetchResult]


def extract_key(settings: OvertureSettings, files: Sequence[ReleaseFile], bbox: BBox, map_sha256: str) -> dict:
    """The public facts the extract depends on (they form its cache key and its SourceRecord params)."""
    etags = sha256_bytes("\n".join(f"{item.key} {item.etag} {item.size}" for item in files).encode("utf-8"))
    return {
        "release": settings.release.id,
        "files": str(len(files)),
        "files_etag_sha256": etags,
        "bbox": bbox.text(),
        "min_confidence": f"{settings.extract.min_confidence.value:g}",
        "exclude_operating_status": ",".join(settings.extract.exclude_operating_status.values),
        "category_map_sha256": map_sha256,
    }


def extract_metro_places(
    client: PoliteClient,
    metro_id: str,
    bbox: BBox,
    reader: PlacesReader | None = None,
    settings: OvertureSettings | None = None,
    cmap: CategoryMap | None = None,
    map_sha256: str | None = None,
) -> PlacesExtract:
    """The metro's places extract: from the cache when the same extract was made before, else built with DuckDB."""
    settings = settings or load_settings()
    cmap = cmap or load_category_map()
    map_sha256 = map_sha256 or file_sha256(CATEGORY_MAP_FILE)
    files, listing = list_release_files(client, settings)
    params = extract_key(settings, files, bbox, map_sha256)
    url = settings.release.places_url
    key = request_key("DUCKDB", url, params)
    cached = None if client.cache is None or client.refresh else client.cache.get(key)
    if cached is None:
        if client.offline:
            raise HostUnreachable("overturemaps-us-west-2", "offline mode is on and the extract is not in the cache")
        body = build_extract_body(files, settings, cmap, bbox, map_sha256, reader or DuckDbPlacesReader())
        cached = CachedResponse(
            body=body,
            sha256=sha256_bytes(body),
            retrieved_at=datetime.now(UTC).replace(microsecond=0),
            url=url,
            params=params,
            status_code=200,
            content_type="application/json",
        )
        if client.cache is not None:
            client.cache.put(key, cached)
        origin = "live"
    else:
        origin = "cache"
    payload = json.loads(cached.body.decode("utf-8"))
    record = SourceRecord(
        source_id=SOURCE_ID,
        publisher=PUBLISHER,
        dataset=(
            f"Overture Maps places theme, release {settings.release.id} (schema {settings.release.schema_version}); "
            "extract of mapped categories in the metro's bounding box"
        ),
        url=url,
        params=params,
        retrieved_at=cached.retrieved_at,
        sha256=cached.sha256,
        license_note=settings.license_note(),
        data_label="PUBLIC",
        origin=origin,
        vintage=f"release {settings.release.id} of {settings.release.release_date}",
        geography=f"{metro_id} bbox {bbox.text()}",
    )
    return PlacesExtract(payload=payload, body=cached.body, record=record, files=files, listing=listing)


def build_extract_body(
    files: Sequence[ReleaseFile],
    settings: OvertureSettings,
    cmap: CategoryMap,
    bbox: BBox,
    map_sha256: str,
    reader: PlacesReader,
) -> bytes:
    """Query the files, apply the mapping, check the accounting and serialize the extract."""
    urls = [item.url(settings.release.s3_https_base) for item in files]
    minimum = settings.extract.min_confidence.value
    excluded = settings.extract.exclude_operating_status.values
    accounting = account(reader.path_counts(urls, bbox, minimum), cmap, excluded)
    rows = reader.places(urls, bbox, cmap.search_categories(), minimum, excluded)
    places = tag_places(rows, cmap)
    if len(places) != accounting.kept:
        raise UpstreamError(
            "overturemaps-us-west-2",
            f"inconsistent answers: {len(places)} places read but the path counts say {accounting.kept}",
        )
    payload = build_payload(places, accounting, settings, cmap, bbox, map_sha256)
    return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def summary(payload: dict[str, Any]) -> str:
    """One line for the snapshot step's detail: what was kept, filtered and ignored, and the licenses."""
    counts = payload["accounting"]
    licenses = ", ".join(f"{name} {n}" for name, n in sorted(payload["kept_by_license"].items()))
    return (
        f"release {payload['release']}: {counts['places_in_bbox']} places in the bbox; kept {counts['kept']} "
        f"(mapped {counts['mapped']}, below confidence {counts['mapped_below_confidence']}, closed "
        f"{counts['mapped_excluded_status']}); not counted by a reviewed row {counts['not_counted_by_row']}; "
        f"unmapped {counts['unmapped']} in {counts['unmapped_distinct_categories']} categories (ignored); "
        f"no taxonomy {counts['without_taxonomy']}; licenses of kept places: {licenses}"
    )


RELEASE_PATTERN = re.compile(r"^release/(\d{4}-\d{2}-\d{2}\.\d+)/")
