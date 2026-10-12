# Founder Studio: build prompts for Underwriting Copilot (v2, multi-city)

Repo: https://github.com/vajja1405/Underwriting-Copilot-
Team: Rahul Vajja + Bhavani Adula (joint project; each person claims only what they built)
Prepared: 2026-10-12

## What we are adding (one paragraph)

Underwriting Copilot today is the **lender's** side: a document packet goes in, a cited, stress-tested
recommendation comes out, and a human decides. **Founder Studio** adds the **borrower's** side. A person with
a business idea (an ice cream shop, a car wash, a motel, an Airbnb, a laundromat, a store) picks a city,
describes the idea, names candidate addresses, and enters their budget. Studio finds the best location,
simulates 1,000 possible futures for each site, shows the chance the business can cover its loan payments,
tells them exactly what to change to become approvable, and produces a lender-ready package where every
number links to its source. The package drops straight into the existing lender copilot as a pre-verified
file. After funding, Studio compares real results to the projection and warns both sides early.

**One-line thesis:** *From idea to bankable: pick the right site, prove the numbers, show the path to yes,
and hand the lender a file it can verify.*

## The innovations (what nobody else combines)

| # | Innovation | Why it is new |
|---|---|---|
| I1 | **Provable numbers** (evidence-linked package) | Every figure in the founder's plan is a ledger claim with its source (public dataset row, user document, model + version, or labeled assumption). AI plan generators today invent growth rates with no basis. Reuses the repo's planned Phase 4 ledger and the SIG "provenance-locked generation" idea. |
| I2 | **Site-level Monte Carlo feasibility** | 1,000 simulated futures per candidate address: demand, seasonality, ramp-up, a competitor opening, rent and rate shocks, local wage levels. Outputs the probability of meeting the lender's debt-service coverage, the chance of running out of cash in 24 months, and P10/P50/P90 revenue. Location tools forecast revenue for chains; plan tools ignore location; neither gives a probability of repaying. |
| I3 | **Path to approval** (counterfactual search) | Instead of yes/no, the smallest set of changes that makes the loan safe: a better site among the candidates, a smaller loan, more owner equity, a longer term, lower build-out cost, different hours or capacity. |
| I4 | **"Suggest a better area"** | Scores every census tract in the chosen metro for the business type and proposes the top areas with plain-language reasons. |
| I5 | **Two-sided handoff** | The founder's package enters the lender copilot (agent + human approval) already structured and evidence-checked; the lender sees the same ledger, stress sliders and path-to-approval history. |
| I6 | **After-funding monitoring with an outcome flywheel** | Founders report monthly actuals; drift against the projection bands raises early warnings to both sides. Stored "predicted vs actual" outcomes become the dataset that improves the simulator over time (the long-term moat). |
| I7 | **Voice intake** (later) | A founder who cannot write a business plan talks to an agent, in their own language, which fills the intake form and reads it back for confirmation. |
| I8 | **Multi-city data fabric with honest coverage** | One national data layer works in any US metro; each metro adds local enhancements. A coverage score per site shows exactly which factors are backed by data and which are assumptions, so a thin-data city is labeled, not guessed. Adding a city is a config file plus a coverage report, not new code. |

## Cities in v1: the five largest US metros

**Selection rule (the answer to "why these cities?"):** the five largest US metropolitan areas by
population in the latest Census Bureau estimates. Largest metros mean the most new small businesses and
loan applications, and the richest public data. The rule is objective and easy to repeat: the next cities
are simply the next ones on the same list.

Expected list (S0 must confirm it against the latest Census estimates and cite the table; if the ranking
differs, use the actual top five):

| Metro | What it adds to testing |
|---|---|
| **New York-Newark-Jersey City (NY-NJ-PA)** | Densest market; walk-in and transit traffic; a **three-state metro**, so the data layer must handle metros that cross state lines; strong tourism for motels and short-term rentals. |
| **Los Angeles-Long Beach-Anaheim (CA)** | Very large, car-dependent and spread out; California-specific rules and costs; tourism. |
| **Chicago-Naperville-Elgin (IL-IN)** | Dense core plus suburbs; a rich city open-data portal (for example business licenses); also crosses a state line. |
| **Dallas-Fort Worth-Arlington (TX)** | Sprawling, drive-by heavy (car washes, drive-thru food); scale of tracts and competitors. |
| **Houston-Pasadena-The Woodlands (TX)** | Large, car-dependent, no city zoning code; a contrasting Texas market to Dallas. |

Together they span dense urban, sprawling car-dependent, multi-state and tourism-led markets.
**Any claim about a metro in docs (rank, size, growth, tourism, zoning) must be backed by a cited source in
S0; do not state it from memory.** More metros are added later with the onboarding checklist in S1.

## Business types in v1 (each is a config profile, so adding more is config, not code)

1. Restaurant / cafe / dessert shop (ice cream, coffee, bakery)
2. Car wash (express tunnel and self-serve)
3. Motel / small hotel
4. Short-term rental (Airbnb-style home or unit)
5. Laundromat
6. Small retail storefront (convenience or specialty)

## How to use this file

1. Paste the **Ground rules addendum** and **Prompt S0** first. S0 changes no product code; it aligns the
   plan with what already exists.
2. Then one prompt per session or PR, in order: S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8 → S11 → S12.
   S9 (monitoring) and S10 (voice) come after the first deployment.
3. After each prompt: review the PR, run `make ci` locally, and have **Rahul merge** (he owns the repo).
   Do not run two prompts at once if they touch the same package.
4. Never paste secrets into a prompt. API keys go in `.env` (local) and in Hugging Face Space secrets
   (deployment, Prompt S11).
5. Coordinate with Rahul's open phases: Phase 2 (document intelligence), Phase 4 (evidence engine),
   Phase 6 (governance), Phase 7 (UI + deploy). S4 builds the **shared** minimal evidence ledger to the
   Phase 4 spec; S7 builds the **shared** website (founder portal + lender portal). Nothing is duplicated.
6. Data work is the biggest task. S1 builds the national layer first (works for all five metros), then
   per-metro enhancements one metro at a time, each in its own commit with its coverage report.

---

## Ground rules addendum (paste before S0; S0 writes it into CLAUDE.md after both owners agree)

```
STUDIO TRACK ADDENDUM (applies to src/uwcopilot/studio and the web app; all existing CLAUDE.md rules still apply)

A. Two owners. This repo is a joint project of Rahul Vajja and Bhavani Adula. Walkthrough docs for Studio
   phases are written so BOTH can explain the work. Each commit message says which phase it belongs to.
B. Data. Repo, tests, fixtures and evaluations use public or SYNTHETIC data only, labeled synthetic.
   The deployed app may accept real founder inputs ONLY under docs/studio/privacy.md (Prompt S8):
   data minimization, no SSNs, no tax returns, no bank account numbers in v1, encryption, deletion on request
   and after a fixed retention period, PII redaction in logs and traces.
C. No lending decisions and no advice claims. Studio prepares and explains; it never approves, declines,
   promises approval, or recommends a specific lender. Every results page shows:
   "Educational planning tool. Not financial, legal or lending advice. Lenders make their own decisions."
D. Benchmarks are sourced or labeled. Every industry or metro number (ticket size, occupancy, capture rate,
   cost ratios, wages, rent, build-out cost) carries a source (publisher, document, date, URL, page/table)
   OR is marked ASSUMPTION with a range and is user-editable. Never invent a benchmark. Never present an
   assumption as data. Never state a fact about a city from memory.
E. Probabilities are simulation estimates. Label them "simulated, based on stated assumptions". With no
   real outcome data there is no claim of real-world accuracy. Validation is on synthetic worlds with known
   ground truth, reported with confidence intervals.
F. Public APIs: respect rate limits and terms, cache responses with checksums, and record the request date
   in provenance. Tests use recorded fixtures; CI never calls the network.
G. Coverage is shown, not hidden. Every site result carries a data-coverage score; factors without data
   are UNKNOWN or ASSUMPTION, never silently filled. A metro without enough data runs in "limited mode".
H. Payments run in Stripe TEST mode only until the owners confirm the business entity and legal review.
I. Accessibility and quality are measured, not claimed: Lighthouse and axe results are written to
   results/*.json by a script.
```

---

## Prompt S0: Align the Studio plan with the existing repo (no product code)

```
Phase S0 (Founder Studio kickoff). Read CLAUDE.md, README.md, docs/build_prompts.md, docs/innovations.md,
docs/agent_design.md, docs/mcp.md and the code under src/uwcopilot (risk, agent, rag, tools, llm,
observability). Do not write product code in this phase.

Deliverables:
1. docs/studio/architecture.md: how Founder Studio connects to the existing system. Mermaid diagram:
   Founder intake -> studio.profiles + studio.metros + studio.data -> studio.siteiq -> studio.feasibility ->
   evidence ledger (shared, Phase 4 spec) -> studio.path_to_approval -> studio.package -> lender copilot
   (agent + human approval) -> monitoring (Phase 6). Mark each box "exists", "shared with Rahul's phase N",
   or "new in Studio". Cover innovations I1-I8 and where each lives.
2. A reuse table: for each existing module we will call (risk/scenario.py, agent/credit_calculations.py,
   agent/policy_credit.py, agent/hitl.py, agent/decision_record.py, rag, llm, observability, tools/registry),
   say exactly which function we reuse and why. If something we need does not exist, say so.
3. docs/studio/business_types.md: the six v1 business types (restaurant/cafe/dessert, car wash,
   motel/small hotel, short-term rental, laundromat, small retail) with revenue drivers, cost structure,
   typical capex items, seasonality, ramp-up, location factors that matter, main risks, and planned data
   sources. Mark every number without a primary source as "ASSUMPTION - to source in S1".
4. docs/studio/metros.md: the five v1 metros = the five largest US metropolitan statistical areas by
   population in the latest Census Bureau estimates (expected: New York-Newark-Jersey City; Los
   Angeles-Long Beach-Anaheim; Chicago-Naperville-Elgin; Dallas-Fort Worth-Arlington; Houston-Pasadena-
   The Woodlands). Open the Census table, confirm the ranking, cite it (table name, vintage, URL), and use
   the actual top five if it differs. For each: the counties included (by FIPS), the states involved, and
   the market type it tests. Back every factual statement with a cited primary source; otherwise remove it.
   Include a short "Why these cities?" answer: the selection rule is "five largest US metros by population",
   nothing personal.
5. docs/studio/data_sources.md, split into
   (a) NATIONAL layer (works in every metro): US Census Geocoder; Census ACS 5-year API; Census
       County/ZIP Business Patterns; BLS QCEW and OEWS (local wages by metro and industry); FHWA HPMS
       (national AADT traffic on road segments); OpenStreetMap via Overpass; FRED (rates); HUD Fair Market
       Rents (residential reference only).
   (b) METRO enhancements, one row per metro: the state DOT traffic-count dataset(s) for every state in the
       metro (New York needs NY, NJ and PA; Chicago needs IL and IN; Dallas and Houston need TX; Los
       Angeles needs CA), city/county open-data portals (for example NYC Open Data, City of Chicago
       business licenses, LA open data, Dallas and Houston open data, only where a real documented
       dataset exists), and Inside Airbnb coverage (check each city; if not covered, short-term rental
       comparables are UNKNOWN for that metro).
   For every source: license/terms, rate limits, free key needed or not, update frequency, and the fallback.
   Commercial rent has no free reliable source: plan for user-entered quotes plus a labeled metro
   assumption range.
6. Append the STUDIO TRACK ADDENDUM (pasted above) to CLAUDE.md under a new heading, and add a
   "Founder Studio" section to docs/build_prompts.md listing phases S1-S12.
7. A risks-and-open-questions list for Rahul and Bhavani to answer before S1 (data volume per metro,
   storage, keys needed, which metro to finish first).

Commit in small commits and open a PR titled "Studio S0: plan and alignment".
```

## Prompt S1: Profiles, metro configs and the data fabric

```
Phase S1, in src/uwcopilot/studio/profiles, src/uwcopilot/studio/metros and src/uwcopilot/studio/data.

A) Business-type profiles (YAML in config/studio/business_types/*.yaml, validated by Pydantic):
   - id, display name, NAICS codes, unit of demand (covers/day, cars/day, room-nights, booked nights,
     machine turns, transactions), price/ticket, capacity limits, operating hours, cost ratios
     (COGS, labor, utilities, supplies, marketing, insurance, maintenance), rent handling, capex items,
     seasonality curve (12 values), ramp-up curve (months to maturity), demand-driver formula inputs
     (car wash: AADT x capture rate; restaurant: resident + daytime population x visit rate;
     motel: rooms x occupancy x ADR; short-term rental: nights x ADR from comparables; laundromat:
     renter households x usage; retail: foot/drive traffic x conversion x basket), location factor
     weights, competitor POI tags, anchor POI tags, key risks.
   - EVERY numeric field is {value or low/mode/high, unit, status: SOURCED|ASSUMPTION, source:{publisher,
     title, date, url, page_or_table}}. SOURCED needs a real primary source you opened; otherwise ASSUMPTION
     with a wide range. Unit tests fail if a SOURCED field lacks a source.
   - Metro overrides: a profile may carry per-metro values (for example wages from BLS for that metro,
     rent assumption ranges, seasonality for tourism metros), each with its own provenance.

B) Metro configs (YAML in config/studio/metros/*.yaml), one per metro confirmed in S0 (expected:
   new_york_ny_nj_pa, los_angeles_ca, chicago_il_in, dallas_fort_worth_tx, houston_tx). Each lists county
   FIPS, states, CRS/bounding box, enabled enhancements, and the data snapshot date. Adding a metro =
   adding one YAML file + running the coverage report.

C) Data connectors (one module each; cached with SHA-256 checksums and request date; polite rate limiting;
   retries; typed outputs; recorded fixtures for tests; CI never touches the network). Build the NATIONAL
   layer first, then enhancements:
   NATIONAL: geocode.py (Census Geocoder: address -> lat/lon, tract GEOID); census_acs.py (tract/block
   group: population, households, renters, median income, age bands, commuting); business_patterns.py
   (CBP/ZBP establishments by NAICS); wages.py (BLS QCEW/OEWS by metro and industry); traffic_hpms.py
   (FHWA HPMS AADT segments; nearest segment to a site); osm.py (Overpass: competitors and anchors by
   profile tags within a radius or polygon); rates.py (FRED); hud_fmr.py (residential reference only).
   ENHANCEMENTS: traffic_state.py with one adapter per state DOT dataset (NY, NJ, PA, CA, IL, IN, TX)
   that improves on HPMS where available; city_open_data.py (start with City of Chicago business
   licenses and NYC Open Data; add others only if the source is real and documented); str_comps.py
   (Inside Airbnb for covered cities; otherwise UNKNOWN).
   Every connector returns values plus a provenance record the evidence ledger (S4) can store.

D) Coverage report: scripts/studio_metro_coverage.py writes results/studio_metro_coverage.json: for each
   metro x business type x factor, whether data exists, its source, freshness, and the share of tracts
   covered. A metro below a set coverage threshold runs in "limited mode" in the app.

E) make studio-data METRO=<id> (and METRO=all) downloads snapshots. Do the national layer for all five
   metros first, then enhancements metro by metro, one commit each. docs/studio/walkthrough_s1.md explains
   every file with 8-10 interview Q&As, including "How do you add a new city?". Tests for every module.
   PR: "Studio S1: profiles, metros, data fabric".
```

## Prompt S2: SiteIQ location scoring and "suggest a better area"

```
Phase S2, in src/uwcopilot/studio/siteiq. Must work for every configured metro.

A) score_site(profile, metro, lat, lon, inputs) -> SiteScore:
   - Trade area: distance rings (profile-specific, e.g. 0.5/1/3/5 miles; tighter in dense New York and
     Chicago, wider in Dallas-Fort Worth, Houston and Los Angeles, set in metro config); optional drive-time isochrones via OpenRouteService
     if a key is set (fallback to rings, recorded).
   - Factors: demand (population, households, renters, income, daytime/commuter flows, AADT for drive-by
     businesses, tourism anchors for motel/short-term rental), competition (count and distance of
     same-type POIs, saturation per 10k residents vs the metro median), complementary anchors (schools,
     offices, malls, highway exits, transit stations, attractions), access/visibility proxies, cost (rent
     quote or the metro ASSUMPTION range, local wages).
   - Each factor: raw value with provenance, normalized score (method stated; normalized within the metro
     so cities are comparable), weight from the profile, and a plain-language reason
     ("3 express washes within 2 miles; nearest 0.6 miles").
   - Overall score 0-100 plus the data-coverage score and notes for ASSUMPTION or UNKNOWN inputs.
   - Demand estimate for the simulator: low/mode/high of the profile's demand unit, every input traceable.

B) suggest_areas(profile, metro, budget) -> top 5 census tracts with scores, reasons and a GeoJSON map
   layer. Exclude tracts where required data is missing rather than guessing.

C) Validation: a SYNTHETIC city generator with known "true" demand; report rank correlation between
   SiteIQ scores and true demand with a bootstrap CI (1,000 resamples, fixed seed) in
   results/studio_siteiq_metrics.json. Also a per-metro sanity table on real tracts (no accuracy claim).

D) Tests, docs/studio/walkthrough_s2.md, make studio-siteiq. PR: "Studio S2: SiteIQ".
```

## Prompt S3: Site-level Monte Carlo feasibility

```
Phase S3, in src/uwcopilot/studio/feasibility. Deterministic, seeded, plain Python + NumPy.

A) simulate(profile, metro, site_score, plan, loan, n=1000, months=36, seed) -> FeasibilityResult:
   - Plan inputs: build-out/capex, opening date, hours/capacity, pricing, staffing, owner equity, working
     capital, loan amount/rate/term (rate default from FRED prime + a stated spread, editable).
   - Each path samples: demand from SiteIQ's distribution, seasonality (metro override where sourced),
     ramp-up, price elasticity band, cost ratios, local wages (BLS for the metro), rent escalation, a
     competitor-entry shock (probability from local saturation), an interest-rate shock, a one-off
     repair/vacancy shock. All distributions are listed in the result.
   - Monthly P&L and cash balance per path; debt service from the loan schedule (reuse
     agent/credit_calculations.py where it fits; tests against hand-worked examples).
   - Outputs: P(DSCR >= threshold) in years 1 and 2, P(cash < 0 within 24 months), break-even month
     distribution, P10/P50/P90 revenue and net income per month, and a tornado chart of the assumptions
     that drive the outcome most.
   - The DSCR threshold is a policy parameter cited to its primary source (SBA SOP 50 10, version and
     date) or labeled "lender policy, editable". Do not hard-code an unverified number.

B) Connect to the existing stress interface (risk/scenario.py): a "lender stress" preset reuses the same
   revenue-drop / rate-rise levers so the founder sees what the lender will test.

C) Validation on synthetic worlds with known truth: coverage of the P10-P90 band (target ~80%) and Brier
   score of P(DSCR >= threshold), with bootstrap CIs, per business type, in
   results/studio_feasibility_metrics.json. Report time per 1,000 paths.

D) Tests, docs/studio/walkthrough_s3.md, make studio-feasibility. PR: "Studio S3: feasibility simulator".
```

## Prompt S4: Shared evidence ledger + provenance-locked founder narrative

```
Phase S4. Build (or extend, if Rahul has started it) the shared evidence ledger in src/uwcopilot/evidence
to the Phase 4 spec in docs/build_prompts.md and the SIG section of docs/innovations.md. Do not create a
second ledger.

A) Ledger claims for a founder package. Statuses: VERIFIED (public dataset row or user document with
   locator), USER_STATED, MODEL_ESTIMATE (with distribution and model/version), ASSUMPTION (profile or
   metro range with source or "unsourced"), CONFLICTING (two sources disagree; link both), UNKNOWN
   (needed but absent). Every claim: id, value, unit, status, provenance (incl. metro and snapshot date),
   input hash.

B) Provenance-locked narrative: the LLM writes the plan's prose using {{claim:ID}} and {{assert:ID}} slots
   only; it cannot type digits, currency, percents or comparatives. A deterministic renderer fills values
   from the ledger. Reuse the repo's LLM layer (provider-neutral, cost cap, cache) and add a Hugging Face
   Inference provider (OpenAI-compatible endpoint) behind the same interface, configured by .env.
   A checker rejects any number not coming from the ledger; mutation tests prove it catches injected
   wrong numbers, by error type.

C) Metrics in results/studio_evidence_metrics.json: share of package numbers with provenance (target 100%),
   checker recall per mutation type, false-flag rate, LLM fallback rate. With no LLM key, run the fake
   client and write "not measured" for LLM-dependent metrics.

D) Tests, docs/studio/walkthrough_s4.md. PR: "Studio S4: evidence ledger + locked narrative".
```

## Prompt S5: Path to approval (counterfactual engine)

```
Phase S5, in src/uwcopilot/studio/path_to_approval.

A) find_paths(plan, candidates, target) -> ranked change-sets that reach the target (default:
   P(DSCR >= threshold in year 2) >= 0.80 and P(cash < 0 within 24 months) <= 0.10; both editable).
   Levers (each with user-set bounds and an effort weight): another candidate site (same metro),
   loan amount, owner equity, loan term, build-out cost, opening hours/capacity, price, working-capital
   reserve. Search single-lever moves first, then pairs, using the S3 simulator with common random numbers.
   Return the 3 smallest change-sets with before/after probabilities and a plain-language explanation.
B) Guardrails: never suggest misstating information, hiding debt, or changing a fact; only plan choices.
   Infeasible targets return "no path within your bounds" plus the closest result.
C) Metrics on synthetic plans across all metros: path-found rate, average levers changed, runtime, in
   results/studio_path_metrics.json.
D) Tests, docs/studio/walkthrough_s5.md. PR: "Studio S5: path to approval".
```

## Prompt S6: Lender-ready package and handoff into the copilot

```
Phase S6, in src/uwcopilot/studio/package.

A) Package (PDF from an HTML template + JSON): cover; executive summary; owner and business description;
   market and location analysis (map, competitor table, demographics, traffic, coverage score); site
   comparison; 36-month projections (monthly year 1, annual years 2-3) with P10/P50/P90; sources and uses
   of funds; loan schedule and DSCR table; stress tests (the levers the lender uses); risks and
   mitigations; path-to-approval history; evidence appendix listing every claim with status and source.
   Align section names with SBA business plan guidance (cite the sba.gov page and date). Synthetic
   examples carry the synthetic banner.
B) Handoff: convert the package into the application schema used by eval/applications (add fields if
   needed, versioned) so the existing LangGraph agent runs on it and stops at the human checkpoint. The
   lender sees the founder's ledger with assumptions marked; the agent may request evidence for UNKNOWN or
   ASSUMPTION claims.
C) Golden tests: 30 SYNTHETIC founder scenarios (each of the 6 business types in each of the 5 metros),
   rendered end to end; the agent run on each reaches the human checkpoint with no unsupported numbers.
D) docs/studio/walkthrough_s6.md. PR: "Studio S6: lender-ready package + handoff".
```

## Prompt S7: The website (founder portal + lender portal)

```
Phase S7. Production web app. Stack: Next.js (App Router, TypeScript, Tailwind, shadcn/ui, MapLibre GL
with a free vector-tile provider that allows this use, Recharts) built as a static export, served by
FastAPI, which also serves /api. One container, port 7860 (for Hugging Face Spaces later).
Auth: Supabase Auth (email magic link) with a free Postgres project for app data; DB layer behind an
interface so it can be swapped. Roles: founder, lender_reviewer, lender_senior, admin.

Founder portal:
 1. Landing page: what it does, the five supported cities, honest limits, sample report (synthetic).
 2. Wizard: city (five metros; others show "coming soon") -> business type -> describe the idea ->
    budget, equity, loan need -> 1-3 candidate addresses OR "suggest areas" -> optional quotes/lease letter
    upload (no SSN, tax returns or bank account numbers in v1) -> review.
 3. Results dashboard: map with site scores, reasons and coverage badges; probability gauges (cover loan
    payments, cash runway); revenue fan chart; tornado chart; path-to-approval cards with "apply this
    change"; evidence drawer (click any number to see its source and status); download PDF; "share with
    a lender" link (expiring, revocable).
 4. Pay-as-you-go: free tier = one site score; credits unlock full simulations and the package.
    Stripe in TEST mode only.

Lender portal: queue of shared packages (filter by metro and business type); ledger by status; stress
sliders (risk scenario API); agent memo with citations; approve / refer / request information / decline
(decline requires a reason and the senior role); decision record export. The copilot never decides.

Quality bar (measured by scripts into results/studio_web_metrics.json): Lighthouse performance,
accessibility and best practices >= 90 on the landing page and dashboard; axe-core zero serious
violations; mobile layouts at 375px; Playwright end-to-end tests for the founder path in at least two
metros, the lender review path, and an error path. Every page shows the disclaimer.

docs/studio/walkthrough_s7.md with screenshots of the local build. PR: "Studio S7: web app".
```

## Prompt S8: Privacy, security and trust for real users

```
Phase S8. Make the app safe for real founders.
 - docs/studio/privacy.md and in-app Privacy Policy + Terms of Use (plain language: what is collected,
   why, retention, deletion, no selling of data, no lending decisions).
 - Data minimization in the intake schema; block uploads that look like tax returns or contain SSN or
   bank-account patterns (reuse PII patterns from observability/redaction.py).
 - Encryption at rest for stored documents; signed, expiring share links; per-user access checks with
   tests; audit log of every access to a founder's package.
 - Automatic deletion after the retention period; "delete my data" button with a test proving deletion.
 - Rate limiting, request size limits, abuse protection on public endpoints; prompt-injection screening on
   uploaded text (reuse agent/injection.py); dependency and secret scanning in CI.
 - Threat model doc (STRIDE-style table). PR: "Studio S8: privacy and security".
```

## Prompt S9: After-funding monitoring and the outcome flywheel (after first deployment)

```
Phase S9, in src/uwcopilot/studio/monitoring (shared with Phase 6 governance).
 - Founder uploads monthly actuals (CSV template or manual form): revenue, units, costs, cash.
 - Compare against the projection bands; drift alerts when actuals fall below P10 for N months or a CUSUM
   statistic crosses a threshold; both sides see the same alert with the likely driver (which assumption
   is off).
 - Store consented, de-identified "predicted vs actual" pairs by metro and business type; a recalibration
   script for profile and metro ranges, run only on synthetic data until real consented data exists.
 - Metrics on synthetic drift scenarios: detection delay and false-alert rate, with CIs.
PR: "Studio S9: monitoring".
```

## Prompt S10: Voice intake agent (later)

```
Phase S10. Voice option in the founder wizard: browser microphone -> speech-to-text (Whisper via the
Hugging Face provider) -> an interview agent that asks the intake questions one at a time, fills the same
intake schema, and reads every field back for confirmation before saving. English plus at least one more
language. The agent can only write intake fields; it cannot run tools or change results. Measure
field-capture accuracy on 20 scripted SYNTHETIC conversations and report it. PR: "Studio S10: voice intake".
```

## Prompt S11: Deploy to Hugging Face Spaces (run when the Space details are ready)

```
Phase S11. Deploy to a Hugging Face Docker Space owned by the team account.
 - Dockerfile (multi-stage: build the Next.js static export, then a slim Python 3.11 image with FastAPI),
   non-root user, port 7860, healthcheck. Space README with the YAML header (sdk: docker, app_port: 7860).
 - All secrets (LLM key, Supabase URL/keys, Stripe test keys, FRED/Census/BLS keys) as Space secrets;
   never in git. Document each in .env.example.
 - Data for five metros: decide what is prebuilt into the image vs downloaded on first start vs stored
   externally; measure image size and cold start; keep the Space within its storage limits and document
   the choice. Qdrant: the mode the repo already supports that fits a Space (embedded/local or a managed
   free tier).
 - GitHub Actions: on a tagged release, run make ci and the Playwright suite, then push to the Space.
 - After deploy: smoke tests against the live URL for each metro, results and a screenshot in
   docs/studio/deployment.md, cold-start and p95 latency recorded. Do not claim a service not actually used.
PR: "Studio S11: Hugging Face deployment".
```

## Prompt S12: Evaluation, README and resume lines

```
Phase S12. Final polish for the Studio track.
 - scripts/eval_studio.py runs the 30 synthetic founder scenarios end to end and writes
   results/studio_business_metrics.json: time to a complete package, evidence coverage, data-coverage
   score by metro, path-found rate, agent handoff success, cost per package. Define the baseline in
   writing first (a plan made without SiteIQ, simulation and the ledger). If a small usability test with
   real people was done, report n, protocol and raw times; otherwise "not measured".
 - README: Founder Studio in the architecture, supported cities and how to add one, results table filled
   by script, live URL, demo GIF.
 - docs/studio/walkthrough_s12.md with 10 interview Q&As each for Rahul and Bhavani, including
   "Why these five cities?" and "How would you add a sixth?".
 - Resume lines: separate, accurate bullets for each person based on git history of who built what;
   numbers only from results/*.json.
PR: "Studio S12: evaluation and docs".
```

---

## Decisions already made

- Ownership: joint (Rahul + Bhavani); Rahul reviews and merges.
- Business types: six in v1, configurable.
- Cities: the five largest US metros by population in v1 (expected New York, Los Angeles, Chicago,
  Dallas-Fort Worth, Houston; confirmed in S0); more by config, next ones from the same ranking.
- Website: Next.js + FastAPI, production quality, measured.
- Hosting: Hugging Face Pro Space (team account) after the GitHub build is complete.

## Before charging real money

- Form the business entity and agree ownership in writing.
- Bhavani: confirm OPT rules for self-employment with the school DSO before earning from it.
- Check state loan-broker rules before taking any lender referral fees.
- Keep Stripe in test mode until all of the above is done.
