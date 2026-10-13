"""Census County Business Patterns (CBP): establishment counts by county and NAICS code. Needs CENSUS_API_KEY.

Studio uses these counts as competition context: same-industry establishments per 10,000 residents in the site's
county compared with the metro median (S2). They have no locations, so distance-based competition comes from
OpenStreetMap (osm.py). CBP suppresses or omits small cells; a county without a row is UNKNOWN, never zero.

The variable names (ESTAB, NAICS2017) were checked against the keyless metadata endpoint
https://api.census.gov/data/<year>/cbp/variables.json on 2026-10-12 for 2022 and 2023.
"""

from dataclasses import dataclass
from typing import Any

from uwcopilot.studio.data.errors import UpstreamError
from uwcopilot.studio.data.http import PoliteClient
from uwcopilot.studio.data.keys import require_key
from uwcopilot.studio.data.provenance import SourceInfo, SourceRecord

HOST = "api.census.gov"
DEFAULT_YEAR = 2023
SOURCE = SourceInfo(
    source_id="census_cbp",
    publisher="U.S. Census Bureau",
    dataset="County Business Patterns (establishments by county and NAICS)",
    license_note="Terms to verify at https://www.census.gov/data/developers.html (not yet recorded)",
)


@dataclass(frozen=True)
class EstablishmentCounts:
    """{county GEOID: establishments} for one NAICS code in one state, with provenance."""

    naics: str
    year: int
    counts: dict[str, int]
    source: SourceRecord


def county_request(year: int, state_fips: str, naics: str) -> tuple[str, dict[str, str]]:
    """(url, public params) for every county of one state and one NAICS code."""
    params = {"get": "ESTAB,NAICS2017", "for": "county:*", "in": f"state:{state_fips}", "NAICS2017": naics}
    return f"https://{HOST}/data/{year}/cbp", params


def parse_county_rows(rows: list[list[Any]]) -> dict[str, int]:
    """{county GEOID: establishments} from the API's array of arrays (first row is the header)."""
    if not rows or not isinstance(rows[0], list):
        raise UpstreamError(HOST, "expected a JSON array of arrays with a header row")
    header = rows[0]
    for column in ("ESTAB", "state", "county"):
        if column not in header:
            raise UpstreamError(HOST, f"column {column!r} missing from the CBP response")
    index = {name: header.index(name) for name in header}
    counts: dict[str, int] = {}
    for row in rows[1:]:
        raw = row[index["ESTAB"]]
        if raw in (None, ""):
            continue
        counts[f"{row[index['state']]}{row[index['county']]}"] = int(raw)
    return counts


def fetch_county_counts(
    client: PoliteClient, state_fips: str, naics: str, year: int = DEFAULT_YEAR
) -> EstablishmentCounts:
    """Establishments for one NAICS code in every county of a state. Raises MissingApiKey without CENSUS_API_KEY."""
    key = require_key("CENSUS_API_KEY")
    url, params = county_request(year, state_fips, naics)
    fetched = client.get(url, {**params, "key": key}, secret_params=["key"])
    if "html" in fetched.content_type.lower():
        raise UpstreamError(HOST, "answered with an HTML page instead of data (is CENSUS_API_KEY valid?)")
    counts = parse_county_rows(fetched.json())
    record = SOURCE.record(fetched, vintage=f"CBP {year}", geography=f"state {state_fips}")
    return EstablishmentCounts(naics=naics, year=year, counts=counts, source=record)
