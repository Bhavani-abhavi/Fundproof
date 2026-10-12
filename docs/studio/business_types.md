# Founder Studio: the six v1 business types (Phase S0)

Written on 2026-10-12. This file is a planning checklist for S1, where each type becomes a YAML profile in
`config/studio/business_types/`. Adding a seventh type later is a new YAML file, not new code.

## Rules for this file

- **No numbers from memory.** Every numeric parameter below is marked **ASSUMPTION - to source in S1**. No value or
  range is given here, on purpose. In S1, each one either gets a primary source (publisher, document, date, URL,
  page or table) and becomes SOURCED, or it stays ASSUMPTION with a wide, user-editable range (addendum rule D).
- **A flat curve is also an assumption.** If S1 cannot source a seasonality or ramp-up curve, a flat curve must be
  labeled ASSUMPTION too. It is not a neutral default.
- **Qualitative lists are a checklist, not data.** The revenue drivers come from the Studio plan (Prompt S1). The cost,
  capex, location and risk lists are a starting checklist written from general knowledge. S1 checks them against
  sources and may change them.
- **No facts about a city here.** Anything that depends on a city (rules, tourism, rents, wages) is stated in S1 with
  a cited source, or not at all.
- **NAICS codes** are looked up in S1 from the Census NAICS reference (https://www.census.gov/naics/, not opened here
  because the host is blocked), then used for County and ZIP Business Patterns and BLS wages.

## Fields every profile has (from the Studio plan, Prompt S1)

| Field | What it holds |
|---|---|
| id, display name, NAICS codes | Identity and the codes used to query public data |
| unit of demand | covers/day, cars/day, room-nights, booked nights, machine turns or transactions |
| price / ticket | Average spend per unit of demand |
| capacity limits, operating hours | Caps on how much demand can be served |
| cost ratios | COGS, labor, utilities, supplies, marketing, insurance, maintenance |
| rent handling | A founder's quote, or the metro ASSUMPTION range |
| capex items | Build-out and equipment lines |
| seasonality curve | 12 monthly values (metro overrides where sourced) |
| ramp-up curve | Months to maturity and the shape of the ramp |
| demand-driver inputs | The inputs of the type's demand formula (below) |
| location factor weights | How SiteIQ (S2) weighs each factor |
| competitor and anchor POI tags | OpenStreetMap tags for same-type businesses and for traffic generators |
| key risks | Shown to the founder and to the lender |

Every numeric field has the form `{value or low/mode/high, unit, status: SOURCED | ASSUMPTION, source}`.

---

## 1. Restaurant / cafe / dessert shop (ice cream, coffee, bakery)

- **Unit of demand:** covers or transactions per day.
- **Revenue driver (plan):** (resident population + daytime population in the trade area) x visit rate x average
  ticket, capped by capacity (seats or service rate) and opening hours.
- **Cost structure (checklist):** food and paper cost (COGS), labor, rent, utilities, supplies, marketing, insurance,
  maintenance, payment processing.
- **Capex items (checklist):** build-out (plumbing, electrical, ventilation), kitchen equipment, freezers and dipping
  cabinets (ice cream), espresso equipment (coffee), ovens (bakery), furniture and fixtures, point of sale, signage,
  permits and fees, opening inventory, working capital.
- **Seasonality:** expected to vary by month, likely more for dessert shops. Curve: ASSUMPTION - to source in S1.
- **Ramp-up:** months to maturity and ramp shape: ASSUMPTION - to source in S1.
- **Location factors (S2):** residents and households, income, daytime (worker) population, foot-traffic proxies
  (transit stations, offices, schools), same-type competitors (count and distance), anchors (malls, attractions),
  visibility, parking in car-dependent areas.
- **Main risks:** thin margins, labor cost and availability, food-cost swings, competitor saturation, slower ramp-up
  than planned, lease terms, health permits.
- **Planned data sources:** Census ACS (residents, households, income, commuting), County / ZIP Business Patterns
  (establishments by NAICS), OpenStreetMap (competitors, anchors), BLS QCEW / OEWS (local wages), city open data
  where a documented dataset exists (for example business licenses), founder quotes (rent, build-out).
- **No free source identified yet:** visit rate, average ticket, cost ratios. These start as ASSUMPTION.

## 2. Car wash (express tunnel and self-serve)

- **Unit of demand:** cars per day.
- **Revenue driver (plan):** AADT (annual average daily traffic) on the adjacent road x capture rate x average ticket.
  If membership plans are modeled, the membership share and price are separate inputs.
- **Cost structure (checklist):** labor (more for tunnel than self-serve), water and sewer, electricity, chemicals and
  supplies, equipment maintenance, rent or land costs, insurance, marketing, payment processing.
- **Capex items (checklist):** land and site work, building, tunnel conveyor and wash equipment (express) or bays and
  wands (self-serve), water reclaim, vacuums, pay stations, signage, permits.
- **Seasonality:** expected to depend on weather. Curve: ASSUMPTION - to source in S1. The plan's data sources have
  no weather source (see [open_questions.md](open_questions.md)).
- **Ramp-up:** ASSUMPTION - to source in S1.
- **Location factors (S2):** AADT on the frontage road, ease of entry and exit, visibility, nearby households and
  vehicles, same-type competitors within driving distance, anchors (grocery stores, retail centers, highway exits).
- **Main risks:** a competitor opening nearby, weather, water and utility costs, equipment downtime, zoning and
  permits.
- **Planned data sources:** FHWA HPMS AADT (national), state DOT traffic counts where they improve on HPMS,
  OpenStreetMap (car-wash and anchor tags), Census ACS, County / ZIP Business Patterns, BLS wages.
- **No free source identified yet:** capture rate, ticket, membership share, cost ratios. These start as ASSUMPTION.

## 3. Motel / small hotel

- **Unit of demand:** room-nights.
- **Revenue driver (plan):** rooms x occupancy x ADR (average daily rate).
- **Cost structure (checklist):** housekeeping and front-desk labor, utilities, linens and supplies, maintenance,
  property taxes, insurance, booking and franchise fees if any, marketing, payment processing.
- **Capex items (checklist):** acquisition or construction, room renovation, furniture and fixtures, property
  management system, signage, safety systems.
- **Seasonality:** expected to follow travel demand; metro overrides where sourced. Curve: ASSUMPTION - to source in
  S1.
- **Ramp-up:** ASSUMPTION - to source in S1.
- **Location factors (S2):** highway exits and AADT, tourism anchors (attractions, venues), airports, hospitals and
  business parks, same-type competitors, short-term rental supply nearby.
- **Main risks:** occupancy and rate swings, competition (including short-term rentals), high fixed costs,
  renovation needs, labor, local lodging rules and taxes (checked per city in S1 with a cited source).
- **Planned data sources:** OpenStreetMap (lodging and anchor tags), FHWA HPMS AADT, Inside Airbnb where a city is
  covered (competing supply), BLS wages, Census ACS.
- **No free source identified yet:** market occupancy and ADR (industry data is usually paid and is not planned).
  These start as ASSUMPTION.

## 4. Short-term rental (Airbnb-style home or unit)

- **Unit of demand:** booked nights.
- **Revenue driver (plan):** nights available x occupancy x ADR, with occupancy and ADR from comparables (Inside
  Airbnb) where the city is covered.
- **Cost structure (checklist):** cleaning, platform fees, utilities, supplies, maintenance, insurance, property
  management, mortgage or rent, local taxes.
- **Capex items (checklist):** furnishing, renovation, appliances, smart locks and safety devices.
- **Seasonality:** expected to follow travel demand. Curve: ASSUMPTION - to source in S1 (from comparables where
  covered).
- **Ramp-up:** ASSUMPTION - to source in S1 (reviews and listing history).
- **Location factors (S2):** tourism anchors, transit access, walkability, density of comparable listings.
- **Main risks:** **local short-term rental rules**, which can restrict or prohibit this business. They differ by
  city and change over time, so S1 must check each city with a cited primary source and never state them from
  memory. Also: platform dependence, seasonality, comparables that are UNKNOWN where Inside Airbnb has no coverage.
- **Planned data sources:** Inside Airbnb (covered cities only; otherwise comparables are UNKNOWN), Census ACS, OSM
  anchors, HUD Fair Market Rents (residential reference only).
- **Lender fit:** the SYNTHETIC Northwind policy has no rules for this type (see [reuse.md](reuse.md) section 2).

## 5. Laundromat

- **Unit of demand:** machine turns (washer and dryer cycles).
- **Revenue driver (plan):** renter households in the trade area x usage per household x spend per visit. Drop-off
  service, if offered, is a separate input.
- **Cost structure (checklist):** water, sewer, gas and electricity, rent, equipment repair, attendant labor if
  staffed, insurance, supplies, payment processing.
- **Capex items (checklist):** washers, dryers, water heaters, plumbing and venting, payment system, build-out,
  seating and folding tables, signage.
- **Seasonality:** ASSUMPTION - to source in S1.
- **Ramp-up:** ASSUMPTION - to source in S1.
- **Location factors (S2):** renter households and density, income, same-type competitors, parking, visibility,
  transit access.
- **Main risks:** utility costs, equipment repair and replacement, competition, a lease shorter than the equipment's
  life.
- **Planned data sources:** Census ACS (renter households, income), OpenStreetMap (laundry tags), County / ZIP
  Business Patterns, BLS wages.
- **No free source identified yet:** usage per household, spend per visit, utility cost per turn. These start as
  ASSUMPTION.

## 6. Small retail storefront (convenience or specialty)

- **Unit of demand:** transactions per day.
- **Revenue driver (plan):** foot or drive-by traffic x conversion rate x average basket.
- **Cost structure (checklist):** cost of goods, labor, rent, utilities, shrink (inventory loss), insurance,
  marketing, payment processing.
- **Capex items (checklist):** build-out, shelving, refrigeration (convenience), point of sale, security systems,
  signage, opening inventory.
- **Seasonality:** ASSUMPTION - to source in S1 (specialty retail may differ a lot from convenience).
- **Ramp-up:** ASSUMPTION - to source in S1.
- **Location factors (S2):** foot-traffic proxies, AADT for drive-by sites, residents and income, same-type
  competitors (including chains), anchors.
- **Main risks:** thin margins, competition from chains, shrink, inventory management.
- **Planned data sources:** Census ACS, County / ZIP Business Patterns, OpenStreetMap (shop tags), FHWA HPMS and state
  DOT counts, BLS wages, city business-license data where documented.
- **No free source identified yet:** conversion rate, basket size, margins. These start as ASSUMPTION.

---

## Numeric parameters (all six types)

Every row is **ASSUMPTION - to source in S1** until S1 records a primary source. No value is filled in here.

| Parameter | Unit | Restaurant / cafe / dessert | Car wash | Motel / small hotel | Short-term rental | Laundromat | Small retail |
|---|---|---|---|---|---|---|---|
| Demand driver rate (visit rate, capture rate, occupancy, usage, conversion) | share or per-household rate | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 (comparables where covered) | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Price / ticket / ADR / basket | USD per unit | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 (comparables where covered) | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Capacity | units per day | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | founder input (rooms) | founder input (nights available) | founder input (machines) | ASSUMPTION - to source in S1 |
| COGS or supplies ratio | share of revenue | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Labor | hours x local wage | wage from BLS for the metro (S1); hours ASSUMPTION | same | same | ASSUMPTION - to source in S1 | same | same |
| Utilities | share of revenue or USD per month | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Marketing, insurance, maintenance | share of revenue or USD per month | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Rent | USD per month | founder quote, else metro ASSUMPTION range | same | same | same | same | same |
| Capex total | USD | founder quotes, else ASSUMPTION - to source in S1 | same | same | same | same | same |
| Seasonality | 12 monthly factors | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Months to maturity | months | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 | ASSUMPTION - to source in S1 |
| Location factor weights | weights summing to one | design choice in S2, documented | same | same | same | same | same |

"Founder input" values become USER_STATED ledger claims (VERIFIED only with a document such as a quote). Location
factor weights are a design choice, not a fact about the world; S2 documents how they were set.
