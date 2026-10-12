# Founder Studio: data sources (Phase S0)

Written on 2026-10-12.

> **Status: nothing on this page has been opened or verified.** This build environment's network policy blocks every
> Studio data host (api.census.gov, geocoding.geo.census.gov, www.census.gov, api.bls.gov, bls.gov,
> fred.stlouisfed.org, api.stlouisfed.org, overpass-api.de, fhwa.dot.gov, huduser.gov, insideairbnb.com,
> data.cityofchicago.org, data.cityofnewyork.us, openrouteservice, sba.gov) and Hugging Face. The lead engineer
> verified this on 2026-10-12; a re-check the same day of www.census.gov, api.census.gov, api.bls.gov,
> fred.stlouisfed.org, overpass-api.de and huggingface.co got HTTP 403 from the proxy. Only PyPI, npm and Docker Hub
> are reachable.
>
> So the license, terms, rate limits, key requirement and update frequency of every source below are written as
> **"to verify"**, with the page to check. The URLs are starting points written from the plan author's knowledge;
> none was opened here. S1 records, for each source, the exact page, the date it was read and the relevant wording.

## 1. How every source enters Studio (addendum rule F and the S1 design)

- One connector module per source in `src/uwcopilot/studio/data/`. Typed outputs.
- Every response is cached on disk with its SHA-256 and the request date. Downloads follow the repo's existing rule:
  write the file, then log `{dataset, file, source, sha256, hashed_at_utc}` with `common.files.append_checksum_log`.
- Polite rate limiting and retries, set per source once its limits are verified.
- **CI never calls the network.** Tests use fixtures. Real responses cannot be recorded in this environment, so until
  they can, fixtures are small hand-written files labeled **"SYNTHETIC fixture, not a recorded response"**. They are
  replaced by real recordings only if the source's terms allow committing an excerpt (otherwise they stay synthetic;
  CLAUDE.md rule 2 forbids committing raw datasets).
- Every connector returns values **plus a provenance record** that the evidence ledger stores. Proposed fields:
  `source_id`, `publisher`, `dataset_or_table`, `url`, `query`, `request_date_utc`, `snapshot_or_vintage`,
  `response_sha256`, `terms_note` (as recorded in S1), `metro_id`, `geography_id` (for example a tract GEOID).
- A value with no data is **UNKNOWN**, never filled silently (addendum rule G). Each site result shows a data-coverage
  score. A metro below the coverage threshold runs in "limited mode".
- When S1 adds a source, it is added to the README "Datasets and licenses" table with its verified terms.

## 2. National layer (works in every metro)

### 2a. What each source is for

| Source | Studio use | Connector (S1) | Used by | Fallback if unavailable |
|---|---|---|---|---|
| US Census Geocoder | Address to latitude/longitude and census tract GEOID | `geocode.py` | S2, S7 | The founder places the pin on the map; the tract is then looked up from tract geometry. If neither works, the site is excluded, not guessed. |
| Census ACS 5-year API | Tract and block-group population, households, renters, median income, age bands, commuting | `census_acs.py` | S2 demand, S2 suggest areas | None for demographics. Demand factors become UNKNOWN; the site or metro runs in limited mode. |
| Census County / ZIP Business Patterns | Establishment counts by NAICS (competition context, saturation vs metro median) | `business_patterns.py` | S2 competition | OpenStreetMap competitor counts only, labeled as the only source. |
| BLS QCEW | Local wages and employment by county and industry | `wages.py` | S3 labor cost | Profile ASSUMPTION wage range, user-editable. |
| BLS OEWS | Wages by occupation and metro | `wages.py` | S3 labor cost | Same as QCEW. |
| FHWA HPMS | AADT (traffic counts) on road segments; nearest segment to a site | `traffic_hpms.py` | S2 drive-by demand (car wash, retail, motel) | A state DOT count where available; else AADT is UNKNOWN and drive-by demand is not estimated from traffic. |
| OpenStreetMap via Overpass | Competitors and anchors by profile tags, within a radius or polygon | `osm.py` | S2 competition and anchors | Business Patterns counts (no locations, so distance factors are UNKNOWN). |
| FRED | Interest rates (for the S3 default rate: a published rate plus a stated, editable spread) | `rates.py` | S3 | The founder or lender types the rate; it becomes USER_STATED. |
| HUD Fair Market Rents | Residential rent reference only (short-term rental context) | `hud_fmr.py` | S2, S3 (short-term rental only) | Omitted. Never used as commercial rent. |
| OpenRouteService (optional) | Drive-time areas (isochrones) instead of distance rings, only if a key is set | in `siteiq` (S2) | S2 | Distance rings; the result records which method was used. |

### 2b. Terms to verify (none verified here)

| Source | Where to verify (not opened) | Free key needed? | License / terms | Rate limits | Update frequency |
|---|---|---|---|---|---|
| US Census Geocoder | https://geocoding.geo.census.gov/geocoder/ | to verify at this URL | to verify at this URL | to verify at this URL | to verify at this URL |
| Census ACS 5-year API | https://www.census.gov/data/developers.html and https://api.census.gov/data.html | to verify at these URLs | to verify at these URLs | to verify at these URLs | to verify at these URLs |
| Census County / ZIP Business Patterns | https://www.census.gov/programs-surveys/cbp.html | to verify at this URL | to verify at this URL | to verify at this URL | to verify at this URL |
| BLS QCEW | https://www.bls.gov/cew/ and https://www.bls.gov/developers/ | to verify at these URLs | to verify at these URLs | to verify at these URLs | to verify at these URLs |
| BLS OEWS | https://www.bls.gov/oes/ and https://www.bls.gov/developers/ | to verify at these URLs | to verify at these URLs | to verify at these URLs | to verify at these URLs |
| FHWA HPMS | https://www.fhwa.dot.gov/policyinformation/hpms.cfm | to verify at this URL | to verify at this URL | to verify at this URL | to verify at this URL |
| OpenStreetMap via Overpass | https://www.openstreetmap.org/copyright and https://wiki.openstreetmap.org/wiki/Overpass_API | to verify at these URLs | to verify at these URLs (attribution wording too) | to verify at these URLs | to verify at these URLs |
| FRED | https://fred.stlouisfed.org/docs/api/fred/ | to verify at this URL | to verify at this URL (terms for each series as well) | to verify at this URL | to verify at this URL (per series) |
| HUD Fair Market Rents | https://www.huduser.gov/portal/datasets/fmr.html | to verify at this URL | to verify at this URL | to verify at this URL | to verify at this URL |
| OpenRouteService | https://openrouteservice.org/ | to verify at this URL | to verify at this URL | to verify at this URL | not applicable (a routing service) |

## 3. Metro enhancements (one row per metro)

Each enhancement improves on the national layer for one metro. The Studio plan says: add a city or county dataset
**only where a real, documented dataset exists**. Every cell here is UNVERIFIED.

| Metro (UNVERIFIED, see [metros.md](metros.md)) | State DOT traffic counts | City / county open data | Inside Airbnb coverage |
|---|---|---|---|
| New York-Newark-Jersey City | Needs NY, NJ and PA (per the plan). Dataset per state: to find in S1 on each state DOT site; license, format and update frequency to verify there. | NYC Open Data (https://data.cityofnewyork.us/, blocked here). The plan names it; which dataset (for example business licenses or permits) is chosen in S1, only if documented. | To check at https://insideairbnb.com/get-the-data/ (blocked here). If not covered, short-term rental comparables are UNKNOWN. |
| Los Angeles-Long Beach-Anaheim | Needs CA. To find in S1. | "LA open data" is a plan example; the portal and dataset are to find in S1, only if documented. | To check (same URL). |
| Chicago-Naperville-Elgin | Needs IL and IN. To find in S1. | City of Chicago business licenses on https://data.cityofchicago.org/ (plan example, blocked here). Dataset id, terms and update frequency to verify. | To check (same URL). |
| Dallas-Fort Worth-Arlington | Needs TX. To find in S1 (one adapter can serve Dallas and Houston if the dataset covers the state). | "Dallas open data" is a plan example; to find in S1, only if documented. | To check (same URL). |
| Houston-Pasadena-The Woodlands | Needs TX (same adapter as Dallas, if confirmed). | "Houston open data" is a plan example; to find in S1, only if documented. | To check (same URL). |

Fallback for every enhancement: the national layer. A missing enhancement lowers the coverage score; it never
blocks the metro.

## 4. Commercial rent

There is no free, reliable public source for commercial rent (the Studio plan says so, and nothing here changes it).
The plan, restated:

1. **The founder's quotes come first.** A typed rent is a USER_STATED claim. An uploaded quote or lease letter with a
   page locator is VERIFIED (document provenance).
2. **Otherwise a labeled metro ASSUMPTION range.** It is wide, user-editable, shown as ASSUMPTION everywhere, and
   carries its source or the word "unsourced". No range is written here; S1 sets it with its provenance.
3. The simulator (S3) samples rent from that range when no quote exists, and the tornado chart shows how much the
   result depends on it.
4. HUD Fair Market Rents are residential. They are never used as commercial rent.

## 5. Not in the plan, worth deciding

- **Weather.** Car-wash and dessert-shop seasonality likely depends on weather, but the plan lists no weather source.
  Until one is added and verified, seasonality curves stay ASSUMPTION. (Listed in [open_questions.md](open_questions.md).)
- **Map tiles for the website.** S7 needs a free vector-tile provider whose terms allow this use. To choose and
  verify in S7.

## 6. What has not been measured

Data volume per metro, download time, cache size and coverage scores are **not measured**. S1 measures them with
`scripts/studio_metro_coverage.py` and writes `results/studio_metro_coverage.json`.
