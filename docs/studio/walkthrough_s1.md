# Walkthrough: Studio S1, profiles, metros and the data fabric

Written on 2026-10-12 so that both Rahul Vajja and Bhavani Adula can explain this phase in their own words.

**What S1 is.** S1 builds the data foundation of Founder Studio: one config file per business type (what drives its
demand, its costs, its risks), one config file per metro (which counties it covers), the connectors that fetch public
data, a SYNTHETIC city for demos and validation, and one interface (the "data fabric") that SiteIQ (S2) and the
feasibility simulator (S3) read from. It also writes a coverage report that says, per metro and business type, which
factors are backed by data and which are not.

**What was done and what was not.**
- Done: six business-type profiles and their validation; five real metro configs whose CBSA codes, titles and county
  FIPS codes were read from the Census delineation file by a script; a check of the "five largest metros" rule against
  the Census population estimates file; 12 connectors (9 national, 3 metro enhancements) with a shared polite client, cache and provenance record; the
  SYNTHETIC city with hidden true demand; the data fabric; `make studio-data` snapshots for the five real metros using
  the keyless sources; the coverage report; tests with recorded and labeled hand-written fixtures.
- Not done: ACS tract data and County Business Patterns (they need `CENSUS_API_KEY`, which this session could not
  see), OpenStreetMap places (overpass-api.de reset every connection from this environment), BLS API v2 and HUD (keys
  not visible), state DOT traffic adapters (no documented dataset recorded yet), Inside Airbnb coverage (the site
  answered 403). The owners added a FRED key, but environment variables load only when a session starts, so this
  session used FRED's keyless CSV download instead. All five real metros therefore run in **limited mode**.
- No profile number is SOURCED. Every rate, price, cost ratio and curve is the project author's labeled ASSUMPTION with
  a wide range and a note naming the source that should replace it.

## 1. Files

### Config

| File | What it does | Why it is built this way |
|---|---|---|
| `config/studio/business_types/*.yaml` (6 files) | One profile per v1 business type: NAICS codes, demand unit and formula, price, capacity, hours, cost ratios, labor, rent, capex lines, seasonality (12 values), ramp-up, location-factor weights, trade-area rings, competitor and anchor OpenStreetMap tags, key risks, metro overrides. | Adding a business type is a config file, not code. Every number is `{value or low/mode/high, unit, status, source, note}` so its provenance travels with it (addendum rule D). |
| `config/studio/metros/<5 real metros>.yaml` | Official CBSA title, CBSA code, states, every county (state FIPS, county FIPS, name, central/outlying), the delineation record (URL, date, file SHA-256, retrieval time), trade-area ring multiplier, enabled enhancements, coverage threshold. | The name, code, states and counties are written by `scripts/studio_fetch_delineation.py`, never by hand. Ring multipliers and enhancements are labeled design choices. |
| `config/studio/metros/synthetic_metro.yaml` | Generator settings of the SYNTHETIC city: seed, 300 tracts, origin in open ocean, 25-mile extent. | A place that cannot be mistaken for a real neighborhood; `data_label: SYNTHETIC`. |

### Profiles and metros (`src/uwcopilot/studio/profiles`, `src/uwcopilot/studio/metros`)

| File | What it does |
|---|---|
| `profiles/models.py` | Pydantic models. `Quantity` enforces the provenance rule: SOURCED needs a source; ASSUMPTION must be a range (low < high) with a note. Cross-field checks: weights sum to 1 and name known factors, seasonality has 12 values averaging 1.0, cost ratios are shares of revenue, demand inputs match the formula, OpenStreetMap tags match a safe pattern. |
| `profiles/factors.py` | The shared vocabulary: the 13 location factors (and whether each lives per tract, on a POI layer, on a road layer or per metro) and the six demand formulas. A typo in YAML fails a test instead of dropping a factor. |
| `profiles/loader.py` | `load_profile(id, metro_id=None)`, `list_profiles()`, `load_all_profiles()`. A metro override replaces one Quantity or Curve by dotted path, and the result is validated again. |
| `metros/models.py` | `MetroConfig`, `County`, `DelineationRecord`. Refuses counties without a delineation record, refuses `verified: true` without counties and a CBSA code, checks that counties span exactly the listed states and that no enabled state adapter names another state. `metro_mode()` returns full, limited or synthetic with a plain reason. |
| `metros/loader.py` | `load_metro(id)`, `list_metros()`. |
| `metros/delineation.py` | Reads the Census delineation workbook (.xlsx) with the standard library (zip + XML), refusing any DOCTYPE and oversized parts because a download is untrusted input. Finds a metro by CBSA code or title prefix (an ambiguous prefix fails). |
| `metros/ranking.py` | Ranks metropolitan statistical areas by the latest `POPESTIMATE<year>` column of the Census PEP CBSA totals file (CBSA total rows only). |

### Data (`src/uwcopilot/studio/data`)

| File | What it does |
|---|---|
| `fabric.py` | The entry point: `get_fabric(metro_id)` and the method list every S2/S3 caller uses. |
| `fabric_base.py` | The `DataFabric` interface and its value types: `FactorValue` (value or None, status DATA/SYNTHETIC/ASSUMPTION/UNKNOWN, reason, assumptions, SourceRecord), `TractRecord`, `PoiSearch`, `RoadResult`, `ListingSearch`, `FactorCoverage`, and `coverage_share()`. |
| `fabric_synthetic.py` | `SyntheticFabric`: the interface over the generated city. Every value is SYNTHETIC with a SourceRecord that fingerprints the generated tract table. The hidden truth is not reachable through it. |
| `fabric_snapshot.py` | `SnapshotFabric`: the interface over a real metro's newest snapshot. Missing layers are UNKNOWN with the reason. Road and POI lookups go live per site only when a client is passed in. |
| `synthetic_city.py` | The seeded SYNTHETIC city (section 3). |
| `snapshot.py` | `make studio-data METRO=<id>|all`: one step per source; a failed step never stops the others; the command exits 1 and names every unreachable host and missing key variable. |
| `http.py` | `PoliteClient`: per-host minimum interval, retries with backoff on network errors and 429/5xx (Retry-After honored), explicit timeouts, `trust_env=True` for the proxy, an offline mode. Secret parameters, JSON fields and headers are kept out of the cache key, the stored request, logs and error messages. |
| `cache.py` | On-disk cache under `data/studio/cache/` keyed by the SHA-256 of the public request; stores the body's SHA-256 and the request date, and refuses a body that no longer matches its checksum. |
| `provenance.py` | `SourceRecord` (publisher, dataset, URL, public params, retrieved_at, SHA-256, license note, data label, origin, vintage, geography) and `SourceInfo.record(...)`. Refuses secret parameter names. |
| `keys.py`, `errors.py` | The five key variables and what each is for; `MissingApiKey` names the variable, never a value. `NotAvailable`, `HostUnreachable`, `UpstreamError`, `CacheCorrupted`. |
| `geo.py`, `paths.py` | Haversine distances in miles and point-to-polyline distance; the data folders. |
| `geocode.py` | Census Geocoder: address or pin to coordinates and tract GEOID (keyless). |
| `census_acs.py` | ACS 5-year tract tables (key). Variable labels are checked against the keyless metadata endpoint; negative "not available" codes become None. |
| `census_geography.py` | Census Gazetteer tract file: the tract list and internal points of a metro's counties, and each state's postal code (keyless). |
| `business_patterns.py` | County Business Patterns establishments by NAICS (key). A missing county is UNKNOWN, not zero. |
| `wages.py` | BLS QCEW open-data wages by metro and industry (keyless) and BLS API v2 series (key). |
| `traffic_hpms.py` | FHWA HPMS AADT from the geo.dot.gov FeatureServer layers (keyless): nearest segment to a site, segment counts per county. |
| `osm.py` | Overpass query builder and parser for competitor and anchor POIs (keyless; unreachable from here). |
| `rates.py` | FRED prime rate: the API when `FRED_API_KEY` is set, otherwise the keyless CSV. |
| `hud_fmr.py` | HUD Fair Market Rents (token; residential reference only, never commercial rent). |
| `traffic_state.py` | One adapter per state; all raise NotAvailable until a documented dataset is recorded. |
| `city_open_data.py` | City of Chicago Business Licenses and NYC DCWP Issued Licenses (Socrata). Only categories, coordinates and expiration dates are requested, never names. |
| `str_comps.py` | Inside Airbnb comparables: NotAvailable until coverage is verified; a parser for a listings file downloaded by hand. |

### Scripts, tests, Make targets

| File | What it does |
|---|---|
| `scripts/studio_fetch_delineation.py` (`make studio-metros`) | Downloads the delineation file and the PEP CBSA totals file, logs their SHA-256 in `data/studio/checksums.jsonl`, fills the five real metros from the file, disables state adapters for states the file does not list, and writes `results/studio_metro_ranking.json`. |
| `scripts/studio_metro_coverage.py` (`make studio-coverage`) | Writes `results/studio_metro_coverage.json` with a provenance block. Offline. |
| `tests/studio/fixtures/record_fixtures.py` (`make studio-fixtures`) | Re-records the fixtures from keyless live hosts. Never run in CI. |
| `tests/studio/*.py` (`make studio-test`) | Profiles, metros, the workbook reader and ranking, the HTTP client and cache, provenance and keys, every connector, fixture labels and checksums, the SYNTHETIC city, the fabric, the snapshot builder and the coverage script. No test touches the network. |

## 2. Design choices

1. **Every number says where it came from.** A profile number is SOURCED (with publisher, title, date, URL, page) or
   an ASSUMPTION range with a note. A fabric value is DATA, SYNTHETIC, ASSUMPTION or UNKNOWN with a reason. A test
   fails if any profile field is SOURCED today, so nobody flips a label without opening a source.
2. **FIPS codes come from the file, not memory.** The model refuses counties without a delineation record. The test
   `test_county_fips_match_the_recorded_delineation_rows` compares the committed configs with rows recorded from the
   Census file. YAML digit strings are written quoted, so a county code such as `039` stays a string.
3. **The ranking rule is checked, not assumed.** `results/studio_metro_ranking.json` ranks 387 metropolitan
   statistical areas by `POPESTIMATE2025`; the five configured metros are the top five
   (`configured_equals_top_five: true`). One finding: in the July 2023 delineation, CBSA 35620 is titled
   "New York-Newark-Jersey City, NY-NJ" and lists no Pennsylvania county, so the script set the metro's states to
   NY and NJ and disabled the PA traffic adapter. The config id `new_york_ny_nj_pa` was kept so nothing else breaks.
4. **A failed source never hides.** `make studio-data` writes what succeeded and exits 1 with one line per failure,
   naming the host or the key variable. The fabric turns every missing layer into UNKNOWN with that reason.
5. **No broad wage stand-ins.** QCEW does not disclose small industries in a metro. The connector tries the requested
   6-digit NAICS code and its 5-digit parent only; a 3-digit group (for example "repair and maintenance" for a car
   wash) would misstate wages, so the wage stays UNKNOWN and the profile's ASSUMPTION range is used. QCEW reports
   weekly wages without hours, so the hourly range divides by an hours-per-week ASSUMPTION that is listed next to the
   value.
6. **Secrets cannot leak by construction.** Keys are read only by name, kept out of cache keys, stored requests,
   SourceRecords, logs, error messages and fixtures; `SourceRecord` refuses secret parameter names.
7. **Coverage is a weighted share, and it says what it measured.** `coverage_share` weights each required factor by
   the profile's weight (demand-only inputs get the average weight) and counts ASSUMPTION and UNKNOWN as zero.
   `share_basis` says whether a share is over tracts, over counties (real road layer: HPMS is queried per site) or
   over the whole metro (a layer exists or not). A real metro runs in full mode only when every business type meets
   its threshold (0.8, a design choice).
8. **The fabric is split in small files.** One interface (`fabric_base.py`), two implementations, one entry point.
   S2 and S3 code against `DataFabric`, so a real snapshot can replace the SYNTHETIC city without changing them.

## 3. The SYNTHETIC city and why its truth is not SiteIQ's formula

`generate_city(...)` (seed 20261012 in the config) places 300 tract centroids around a downtown and three
sub-centers, with population, households, renters, median income (a few values suppressed, like real survey tables)
and daytime workers; radial highways, a ring highway, an arterial grid and a local street through every centroid,
cut into ~1-mile segments with AADT; every competitor and anchor tag the profiles use, plus tourism tags, with
competitors clustered where their own demand is; short-term rental listings; and a SYNTHETIC wage index, rent index
and prime rate. Every constant is a generator parameter, not data.

It also computes a hidden **true demand** per tract and business type, in the profile's demand unit. That formula
deliberately differs from anything SiteIQ will do: smooth gravity decay instead of distance rings, a Huff-style
saturating competition share instead of a competitor count, interactions (office workers x office anchors, drive-by x
walk-in traffic), saturation curves on income and market size, a spatially correlated latent "neighborhood appeal"
field that no feature exposes, multiplicative noise, and the true income where the published table is suppressed
(`TRUE_DEMAND_NOTES` lists each formula). A test checks that a linear fit of log truth on the published tract
features does not explain it fully (R-squared below 0.9 for the restaurant and laundromat types). Validation in S2 reads the truth from `synthetic_city` directly; the fabric never
exposes it.

## 4. Results (copied from files the scripts wrote)

From `results/studio_metro_ranking.json` (PEP file `cbsa-est2025-alldata.csv`, column `POPESTIMATE2025`):

| Rank | CBSA | Name in the file | Population estimate |
|---|---|---|---|
| 1 | 35620 | New York-Newark-Jersey City, NY-NJ | 20112448 |
| 2 | 31080 | Los Angeles-Long Beach-Anaheim, CA | 12844441 |
| 3 | 16980 | Chicago-Naperville-Elgin, IL-IN | 9434123 |
| 4 | 19100 | Dallas-Fort Worth-Arlington, TX | 8477157 |
| 5 | 26420 | Houston-Pasadena-The Woodlands, TX | 7904627 |

From `results/studio_metro_coverage.json` (snapshots of 2026-10-12, built with keyless sources only):

| Metro | Counties | Tracts | Mode | Lowest coverage share (any business type) |
|---|---|---|---|---|
| new_york_ny_nj_pa | 22 | 4928 | limited | 0.0 |
| los_angeles_ca | 2 | 3112 | limited | 0.0 |
| chicago_il_in | 13 | 2292 | limited | 0.0 |
| dallas_fort_worth_tx | 11 | 1704 | limited | 0.0 |
| houston_tx | 10 | 1615 | limited | 0.0 |
| synthetic_metro | - | 300 | synthetic | 0.996 |

In every real metro the per-type shares are 0.35 (car wash), 0.25 (motel), 0.2 (retail storefront) and 0.0
(restaurant/cafe, laundromat, short-term rental): only the traffic factor has data (HPMS, in every county), because
ACS needs the Census key and OpenStreetMap was unreachable. QCEW disclosed a wage for 1 (Chicago), 2 (Houston),
3 (Dallas-Fort Worth, New York) and 6 (Los Angeles) of the six business types; the others stay UNKNOWN. The prime
rate is DATA (FRED DPRIME) in every snapshot.

## 5. Limitations

- Real metros cannot leave limited mode until `CENSUS_API_KEY` is visible to a session and OpenStreetMap is
  reachable; then `make studio-data METRO=all && make studio-coverage` refreshes everything.
- Daytime workers have no real source yet (ACS tract tables count residents); restaurant and retail coverage stays
  below the threshold until a source such as the Census LEHD origin-destination files is added and cited.
- License and terms of every source are still "to verify" in each SourceRecord's `license_note`.
- The HUD entity-id format and the Inside Airbnb column names were written from the author's knowledge, are marked so,
  and are tested only with hand-written fixtures.
- NAICS codes in the profiles are `verified: false`; the Census NAICS site returned 403 here.
- Commercial rent has no free source; it stays an ASSUMPTION range until the founder enters a quote.

## 6. Likely interview questions

1. **How do you add a new city?** Add `config/studio/metros/<id>.yaml` with `cbsa_title_prefix`, states, trade-area
   settings and enhancements; run `make studio-metros` (fills the CBSA code and counties from the Census file),
   `make studio-data METRO=<id>` and `make studio-coverage`. No code changes, unless the city needs a new state
   traffic adapter or city dataset.
2. **Why these five cities?** The rule is "the five largest US metropolitan statistical areas by population in the
   latest Census estimates". The script ranks the Census file and confirms the five configured metros are the top
   five; the sixth on the same list is next.
3. **How do you stop someone from typing a made-up benchmark?** Every number must be SOURCED with a full citation or an
   ASSUMPTION range with a note; the model rejects anything else, and a test fails if any profile field is SOURCED
   before a source has been opened and cited.
4. **What happens when a data source is down or a key is missing?** The step fails, the run continues, the snapshot
   keeps what succeeded, the command exits 1 naming the host or variable, and the fabric reports UNKNOWN with the
   reason. The metro then runs in limited mode and the app says so.
5. **Why validate on a synthetic city?** Real outcome data (which sites did well) does not exist for this project. A
   synthetic world with a known true demand lets S2 measure whether its ranking recovers the truth. Because the same
   author wrote both, the truth formula is deliberately different from SiteIQ's (gravity, saturation, interactions,
   hidden appeal, noise), and the result is reported as validation on a synthetic world, not real-world accuracy.
6. **How do you keep API keys out of logs and fixtures?** Keys are read by name only when needed; the client removes
   them from the cache key, stored request, logs and errors; `SourceRecord` refuses secret parameter names; a test
   scans every fixture for key-like parameters.
7. **What does "coverage 0.35" mean for a car wash in Houston?** The car-wash profile weights traffic 0.35; only
   traffic has data there (HPMS segments with AADT in every county). Households, income, competition and anchors are
   UNKNOWN and rent is an ASSUMPTION, so they count zero. 0.35 is below the 0.8 threshold, so the metro is limited.
8. **Why does New York have no Pennsylvania county?** The July 2023 Census delineation lists CBSA 35620 with New York
   and New Jersey counties only; the config follows the file. We kept the id `new_york_ny_nj_pa` to avoid breaking
   references, and the PA adapter is disabled with a note.
9. **Why not use the 3-digit NAICS wage when the car-wash code is suppressed?** A broad group mixes in other
   industries and would misstate the wage. UNKNOWN plus the labeled assumption range is more honest than a precise
   wrong number.
10. **What is cached, and how do you know a cached file was not changed?** Every response body is stored with its
    SHA-256 and the original request date; reading it recomputes the checksum and refuses a mismatch. Snapshots and
    results record those checksums, so a number can be traced to the exact response.
