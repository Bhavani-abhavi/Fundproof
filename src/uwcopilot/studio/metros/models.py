"""Pydantic models for a metro config (config/studio/metros/<id>.yaml) and the metro "mode".

A metro is either a real US metro (data_label PUBLIC) or the generated SYNTHETIC metro used for demos, tests and
validation. A real metro's county list stays empty and `verified` stays false until someone opens the Census
delineation file and records it as `delineation_source`; the model refuses counties without that source, so FIPS
codes cannot be typed in from memory.

The mode tells the app how much to trust a metro's data:

- synthetic: the SYNTHETIC metro; everything is generated and labeled SYNTHETIC.
- limited: a real metro without a data snapshot, with an unverified county list, or with data coverage below the
  metro's threshold. Factors without data are UNKNOWN; the app says "limited mode".
- full: a real metro with a verified county list, a snapshot and coverage at or above the threshold.
"""

import re
from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from uwcopilot.studio.profiles.models import Source, StrictModel

MetroMode = Literal["full", "limited", "synthetic"]
DataLabel = Literal["PUBLIC", "SYNTHETIC"]
METRO_ID = re.compile(r"^[a-z][a-z0-9_]*$")
STATE_POSTAL = re.compile(r"^[A-Z]{2}$")
# Design choice: a real metro needs at least this share of its required factors covered to run in full mode.
DEFAULT_COVERAGE_THRESHOLD = 0.8


class County(StrictModel):
    """One county of a metro, as listed in the Census delineation file."""

    state_fips: str = Field(pattern=r"^\d{2}$")
    county_fips: str = Field(pattern=r"^\d{3}$")
    name: str = Field(min_length=1)
    central_or_outlying: Literal["central", "outlying"] | None = None

    @property
    def geoid(self) -> str:
        """The 5-digit county GEOID (state FIPS + county FIPS)."""
        return self.state_fips + self.county_fips


class BoundingBox(StrictModel):
    """A latitude/longitude box (WGS 84)."""

    min_lat: float = Field(ge=-90, le=90)
    min_lon: float = Field(ge=-180, le=180)
    max_lat: float = Field(ge=-90, le=90)
    max_lon: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def check_order(self) -> "BoundingBox":
        """min must be below max."""
        if self.min_lat >= self.max_lat or self.min_lon >= self.max_lon:
            raise ValueError("bounding box min values must be below max values")
        return self


class TradeArea(StrictModel):
    """How the metro scales the profile's trade-area rings (a design choice, not a fact about the metro)."""

    ring_multiplier: float = Field(gt=0, le=5)
    status: Literal["DESIGN_CHOICE"]
    note: str = Field(min_length=1)


class Enhancement(StrictModel):
    """A metro-specific data source that improves on the national layer (state DOT counts, city data, comparables)."""

    kind: Literal["traffic_state", "city_open_data", "str_comps"]
    adapter: str = Field(min_length=1, description="Adapter id inside the connector module, e.g. 'TX' or 'chicago'")
    enabled: bool = True
    note: str | None = None


class SyntheticSettings(StrictModel):
    """Generator settings for the SYNTHETIC metro (studio.data.synthetic_city)."""

    seed: int
    n_tracts: int = Field(ge=20, le=5000)
    origin_lat: float = Field(ge=-80, le=80)
    origin_lon: float = Field(ge=-180, le=180)
    extent_miles: float = Field(gt=1, le=200)
    note: str = Field(min_length=1)


class MetroConfig(StrictModel):
    """One metro. See config/studio/metros/*.yaml and docs/studio/metros.md."""

    id: str
    name: str = Field(min_length=1)
    data_label: DataLabel
    verified: bool
    states: list[str]
    cbsa_code: str | None = Field(default=None, pattern=r"^\d{5}$")
    counties: list[County] = Field(default_factory=list)
    delineation_source: Source | None = None
    crs: Literal["EPSG:4326"] = "EPSG:4326"
    bbox: BoundingBox | None = None
    trade_area: TradeArea
    enhancements: list[Enhancement] = Field(default_factory=list)
    snapshot_date: date | None = None
    coverage_threshold: float = Field(default=DEFAULT_COVERAGE_THRESHOLD, gt=0, le=1)
    synthetic: SyntheticSettings | None = None
    note: str | None = None

    @field_validator("id")
    @classmethod
    def id_is_snake_case(cls, value: str) -> str:
        """Metro ids are lowercase snake_case (they are also file names)."""
        if not METRO_ID.match(value):
            raise ValueError(f"metro id must be lowercase snake_case (got {value!r})")
        return value

    @model_validator(mode="after")
    def check_metro(self) -> "MetroConfig":
        """Synthetic metros have generator settings and no real geography; real metros never list unsourced counties."""
        if self.data_label == "SYNTHETIC":
            if self.synthetic is None:
                raise ValueError("a SYNTHETIC metro needs synthetic generator settings")
            if self.states or self.counties or self.cbsa_code or self.verified:
                raise ValueError("a SYNTHETIC metro has no states, counties, CBSA code or verified flag")
            return self
        if self.synthetic is not None:
            raise ValueError("only a SYNTHETIC metro may have synthetic generator settings")
        if not self.states or any(not STATE_POSTAL.match(state) for state in self.states):
            raise ValueError("a real metro needs its states as two-letter postal codes")
        if self.counties and self.delineation_source is None:
            raise ValueError("counties need delineation_source (the Census delineation file that lists them)")
        if self.verified and (not self.counties or self.delineation_source is None or self.cbsa_code is None):
            raise ValueError("verified: true needs counties, a CBSA code and delineation_source")
        if self.counties and not self.verified:
            raise ValueError("counties may only be filled together with verified: true")
        return self

    def enabled_enhancements(self, kind: str | None = None) -> list[Enhancement]:
        """The enabled enhancements, optionally only of one kind."""
        return [item for item in self.enhancements if item.enabled and (kind is None or item.kind == kind)]


def metro_mode(metro: MetroConfig, coverage_share: float | None) -> tuple[MetroMode, str]:
    """(mode, plain-language reason) for a metro, given the share of required factors that have data.

    coverage_share is None when it could not be computed (for example no snapshot).
    """
    if metro.data_label == "SYNTHETIC":
        return "synthetic", "SYNTHETIC metro: generated data for demonstration and validation only"
    if metro.snapshot_date is None:
        return "limited", "no data snapshot"
    if not metro.verified:
        return "limited", "county list not yet verified against the Census delineation file"
    if coverage_share is None:
        return "limited", "data coverage could not be computed"
    if coverage_share < metro.coverage_threshold:
        return "limited", f"data coverage {coverage_share:.2f} is below the threshold {metro.coverage_threshold:.2f}"
    return "full", f"data coverage {coverage_share:.2f} meets the threshold {metro.coverage_threshold:.2f}"
