# Founder Studio: risks and open questions before S1 (Phase S0)

Written on 2026-10-12 for Rahul Vajja and Bhavani Adula. Each item needs an answer from the owners. Where this file
makes a proposal, it says "proposal"; nothing here is decided.

## A. Access and accounts

| # | Question | Why it matters | Needed before |
|---|---|---|---|
| A1 | **Where does S1 download data?** Every Studio data host and Hugging Face is blocked in this build environment (checked 2026-10-12). Only PyPI, npm and Docker Hub work. Options: run `make studio-data` on an owner's machine, or ask for these hosts to be allowed here. | Without it, S1 can build and test connectors on SYNTHETIC fixtures only, and no metro, ranking or license can be confirmed. | S1 |
| A2 | **Who registers the free API keys** (Census, BLS, FRED, OpenRouteService if used), under which account? Whether each one needs a key is itself still to verify. Keys go in `.env` (never in git) and later in Space secrets. | Connectors cannot run for real without them. | S1 (S2 for OpenRouteService) |
| A3 | **Who owns the accounts** for Supabase (auth + Postgres), Stripe (test mode only) and the Hugging Face team account (Space, and Inference if used)? Who pays? | S7 and S11 depend on them. The Stripe and Supabase hosts are also blocked here. | S7, S11 |
| A4 | **Can recorded API responses be committed as test fixtures?** CLAUDE.md forbids committing raw datasets. Proposal: keep fixtures SYNTHETIC and labeled, unless a source's terms clearly allow a small excerpt. | Decides whether CI tests use real or synthetic shapes. | S1 |

## B. Data, storage and order

| # | Question | Why it matters | Needed before |
|---|---|---|---|
| B1 | **Data volume and storage per metro** are not measured. Traffic segments, OSM extracts and tract tables could be large. | Decides what is prebuilt into the image, downloaded on first start, or stored outside (S11), and how big the Space gets. | Measure in S1; decide in S11 |
| B2 | **Which metro to finish first?** Proposal for discussion: one single-state metro end to end first (fewer adapters), then a multi-state one to test cross-state joins. The ranking itself is still UNVERIFIED ([metros.md](metros.md)). | The plan builds the national layer for all five metros first, then enhancements one metro at a time. | S1 |
| B3 | **Which Census vintage, and when to re-rank?** Proposal: freeze the vintage used in S1 for v1; re-check at each release; never drop a built metro. | Keeps "why these cities?" stable and repeatable. | S1 |
| B4 | **Weather data** for seasonality (car wash, dessert shop) is not in the plan's source list. Add a source, or keep seasonality as ASSUMPTION? | Otherwise the seasonality curves for these types stay unsourced. | S1 |

## C. Fit with the lender side

| # | Question | Why it matters | Needed before |
|---|---|---|---|
| C1 | **How does the lender side treat a new business?** The SYNTHETIC Northwind reference policy needs a full run of bank statements and has minimum time-in-business rules, so every pre-revenue founder file ends in "request more information" or "refer". Options: (a) add a SYNTHETIC "new business" section to the guideline with rules on projections, labeled as projection-based; (b) keep the policy and show the founder that the lender would refer; (c) limit v1 to expansions by existing operators. | The two-sided handoff (I5) is only meaningful if the lender side can evaluate the file. | S6 (design in S3) |
| C2 | **New application schema.** `CreditApplication` forbids unknown fields and has no projection fields. Four of the six business types have no SYNTHETIC industry code. A versioned schema change is needed. | Changing the schema can break the 65 committed agent applications and their checksums if done carelessly. | S6 |
| C3 | **Ledger statuses.** Studio needs USER_STATED and ASSUMPTION in the shared Phase 4 ledger. Agree with whoever builds Phase 4 now, so there is one status list. | Avoids a second ledger or a later migration. | Phase 4 / S4 |
| C4 | **Website roles.** Proposal: `lender_reviewer` = `underwriter`, `lender_senior` = `senior_underwriter`; `founder` and `admin` can never decide. | `agent/hitl.py` knows only the two reviewer roles today. | S7 |
| C5 | **Who are the lenders in the deployed app?** Addendum rule C forbids recommending a specific lender. Proposal: the lender portal is a demo with the SYNTHETIC Northwind lender until the business and legal items in section E are done. | Avoids implying a real lending relationship. | S7, S11 |
| C6 | **DSCR threshold in S3.** The plan wants it cited to SBA SOP 50 10 or labeled "lender policy, editable". sba.gov is blocked here. Proposal: use the SYNTHETIC Northwind constants labeled "lender policy (SYNTHETIC), editable" until the SBA source can be opened and cited. | No unverified number is hard-coded. | S3 |
| C7 | **Deployment target for Phase 7 C.** The original Phase 7 named Azure or AWS; the Studio plan decided on a Hugging Face Space. Confirm the Space deployment (S11) is the one Phase 7 deployment. | Avoids two deployments and two security write-ups. | S11 |

## D. Licenses that matter once money is involved

| # | Question | Why it matters | Needed before |
|---|---|---|---|
| D1 | **OpenSanctions is used under CC BY-NC 4.0** (non-commercial, per the README). A paid app is commercial use. Options: no OpenSanctions in the paid product, OFAC only, or a commercial license. | The lender side screens names with it. | Before any paid use |
| D2 | **qwen2.5 3B license.** `docs/innovations.md` says the model's upstream Hugging Face card listed the Qwen Research License (checked by the plan author on 2026-10-10). The lead engineer's environment notes say the Docker Hub artifact used here carries an Apache-2.0 license file. Resolve before a public or paid deployment, or use a model with a clear license. | A model license can restrict commercial use. | S11 |
| D3 | **Data source terms** (section 2b of [data_sources.md](data_sources.md)) are all unverified, including attribution wording and whether commercial use is allowed. | A paid product must respect each source's terms. | S1 (record), S7 (show attribution) |
| D4 | **Which LLM runs in the deployed app?** Local Ollama inside the Space, a Hugging Face Inference endpoint (needs a key and has a cost), or the deterministic template only. The slot design works with no LLM. | Cost cap ($10 per run, CLAUDE.md rule 4), license (D2), cold start. | S4, S11 |

## E. Owner decisions (not engineering)

These come from the Studio plan's "Before charging real money" list. No code depends on them before S7, and Stripe
stays in test mode until all are done.

- Form the business entity and agree ownership in writing.
- Bhavani: confirm OPT rules for self-employment with the school DSO before earning from it.
- Check state loan-broker rules before taking any lender referral fees.

## F. Process

| # | Question | Needed before |
|---|---|---|
| F1 | **CLAUDE.md sign-off.** The Studio track addendum was appended to CLAUDE.md as "pending Rahul's sign-off". Both owners must agree before it is final. | S1 |
| F2 | **Naming.** `docs/innovations.md` already uses "S1" to "S6" for innovation items. Proposal: say "innovation S1" and "Studio S1" in commits and docs. | S1 |
| F3 | **Node toolchain in CI.** S7 adds a Next.js app, so CI needs Node and a committed npm lockfile. | S7 |
| F4 | **Walkthrough owners.** Addendum rule A asks that both owners can explain each Studio phase. Agree who reviews each walkthrough. | S1 |
