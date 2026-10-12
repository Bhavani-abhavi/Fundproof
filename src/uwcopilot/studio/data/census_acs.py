"""Census ACS 5-year tract tables: population, households, renter households and median household income.

Data requests need a free key (CENSUS_API_KEY): without one, api.census.gov redirects to a "missing key" page. The
variable metadata endpoint (variables/<id>.json) answers without a key, so verify_variables() can check that every
variable below still means what this module says before a snapshot uses it.

Daytime workers are NOT in these tables: the ACS tract tables count residents. A daytime-population source (for
example the Census LEHD origin-destination files) is not built yet, so `daytime_workers` stays UNKNOWN for real metros.

Census marks unavailable estimates with large negative codes. Every variable used here is a count or a dollar amount
that cannot be negative, so any negative value is read as "no estimate" (None), never as a number.
"""

from dataclasses import dataclass
from typing import Any

from uwcopilot.studio.data.errors import NotAvailable, UpstreamError
from uwcopilot.studio.data.http import PoliteClient
from uwcopilot.studio.data.keys import require_key
from uwcopilot.studio.data.provenance import SourceInfo, SourceRecord

HOST = "api.census.gov"
DEFAULT_YEAR = 2024
# Location factor -> (ACS variable, the label the variable metadata endpoint must return).
ACS_VARIABLES: dict[str, tuple[str, str]] = {
    "resident_population": ("B01003_001E", "Estimate!!Total"),
    "households": ("B11001_001E", "Estimate!!Total:"),
    "renter_households": ("B25003_003E", "Estimate!!Total:!!Renter occupied"),
    "median_household_income": ("B19013_001E", "Estimate!!Median household income in the past 12 months"),
}
SOURCE = SourceInfo(
    source_id="census_acs5",
    publisher="U.S. Census Bureau",
    dataset="American Community Survey 5-year estimates, detailed tables (tract level)",
    license_note="Terms to verify at https://www.census.gov/data/developers.html (not yet recorded)",
)


@dataclass(frozen=True)
class AcsTractTable:
    """{tract GEOID: {factor: value or None}} for one county, with its provenance."""

    year: int
    values: dict[str, dict[str, float | None]]
    source: SourceRecord


def dataset_url(year: int) -> str:
    """The ACS 5-year detailed-tables endpoint for a year."""
    return f"https://{HOST}/data/{year}/acs/acs5"


def tract_request(year: int, state_fips: str, county_fips: str) -> tuple[str, dict[str, str]]:
    """(url, public params) for every tract of one county. The key is added separately and never stored."""
    variables = ",".join(variable for variable, _ in ACS_VARIABLES.values())
    params = {"get": f"NAME,{variables}", "for": "tract:*", "in": f"state:{state_fips} county:{county_fips}"}
    return dataset_url(year), params


def to_estimate(raw: Any) -> float | None:
    """A Census estimate cell as a number; None for empty cells and negative 'not available' codes."""
    if raw is None or raw == "":
        return None
    value = float(raw)
    return None if value < 0 else value


def parse_tract_rows(rows: list[list[Any]]) -> dict[str, dict[str, float | None]]:
    """Parse the API's JSON array of arrays (first row is the header) into {GEOID: {factor: value}}."""
    if not rows or not isinstance(rows[0], list):
        raise UpstreamError(HOST, "expected a JSON array of arrays with a header row")
    header = rows[0]
    for column in ("state", "county", "tract"):
        if column not in header:
            raise UpstreamError(HOST, f"column {column!r} missing from the ACS response")
    index = {name: header.index(name) for name in header}
    table: dict[str, dict[str, float | None]] = {}
    for row in rows[1:]:
        geoid = f"{row[index['state']]}{row[index['county']]}{row[index['tract']]}"
        table[geoid] = {factor: to_estimate(row[index[variable]]) for factor, (variable, _) in ACS_VARIABLES.items()}
    return table


def fetch_tracts(client: PoliteClient, state_fips: str, county_fips: str, year: int = DEFAULT_YEAR) -> AcsTractTable:
    """Every tract of one county. Needs CENSUS_API_KEY (raises MissingApiKey without it)."""
    key = require_key("CENSUS_API_KEY")
    url, params = tract_request(year, state_fips, county_fips)
    fetched = client.get(url, {**params, "key": key}, secret_params=["key"])
    if "html" in fetched.content_type.lower():
        raise UpstreamError(HOST, "answered with an HTML page instead of data (is CENSUS_API_KEY valid?)")
    values = parse_tract_rows(fetched.json())
    if not values:
        raise NotAvailable(f"the ACS returned no tracts for county {state_fips}{county_fips}")
    record = SOURCE.record(fetched, vintage=f"ACS 5-year {year}", geography=f"county {state_fips}{county_fips}")
    return AcsTractTable(year=year, values=values, source=record)


def variable_request(year: int, variable: str) -> str:
    """The keyless metadata URL of one variable."""
    return f"{dataset_url(year)}/variables/{variable}.json"


def check_variable_label(metadata: dict[str, Any], variable: str, expected_label: str) -> str:
    """Return the variable's label if it starts with the expected label; raise UpstreamError otherwise."""
    label = str(metadata.get("label", ""))
    if metadata.get("name") != variable or not label.startswith(expected_label):
        raise UpstreamError(HOST, f"variable {variable} now reads {label!r}; expected {expected_label!r}")
    return label


def verify_variables(client: PoliteClient, year: int = DEFAULT_YEAR) -> dict[str, str]:
    """{variable: label} for every variable used, checked against the keyless metadata endpoint."""
    labels: dict[str, str] = {}
    for variable, expected in ACS_VARIABLES.values():
        fetched = client.get(variable_request(year, variable))
        labels[variable] = check_variable_label(fetched.json(), variable, expected)
    return labels
