# Walkthrough: Studio S0, aligning Founder Studio with the existing repo

Written on 2026-10-12 so that both Rahul Vajja and Bhavani Adula can explain this phase in their own words.

**What S0 is.** S0 writes no product code. It reads the existing system and the Studio plan, and writes down how
Founder Studio fits in: what already exists, what is shared, what is new, what we can and cannot claim yet, and
what the owners must decide before S1.

**What was done and what was not.**
- Done: read CLAUDE.md, README.md, docs/build_prompts.md, docs/innovations.md, docs/agent_design.md, docs/mcp.md and
  the code under `src/uwcopilot` (risk, agent, rag, tools, llm, observability, provenance and the in-progress
  docintel, governance and evidence work). Every function named in [reuse.md](reuse.md) was checked against the code.
- Not done: confirming the Census metro ranking, the county lists, or any data source's license, terms, rate limits
  or update frequency. Every Studio data host is blocked in this build environment (verified by the lead on
  2026-10-12; a re-check the same day got HTTP 403 from the proxy). So those items are marked UNVERIFIED or
  "to verify", with the page to check.
- No number in these docs was measured, and none was typed from memory.

## 1. Files

| File | What it does | Why it is built this way |
|---|---|---|
| `docs/studio/architecture.md` | The one-product picture: a Mermaid diagram from founder intake to monitoring, each box labeled exists / in progress / shared / new, the I1-I8 innovations and where each lives, the data-to-ledger contract, safety boundaries, a plain list of what does not exist yet, and the merged phase map (Phases 0-8 and Studio S0-S12). | One page that answers "how does Studio connect to what we have?". Status words are in the box text, so the diagram does not rely on color. |
| `docs/studio/reuse.md` | For each existing module Studio will call: the exact functions, which Studio phase uses them, why, and the caveats. Then a list of what we need that does not exist or does not fit. | Prevents duplicate code (two DSCR formulas, two ledgers, two approval rules). Also records real mismatches, such as the risk client/service contract and the missing revenue/rate levers in `risk/scenario.py`. |
| `docs/studio/business_types.md` | The six v1 business types: demand unit, revenue driver formula, cost structure, capex checklist, seasonality, ramp-up, location factors, risks and planned data sources. Every numeric parameter is "ASSUMPTION - to source in S1". | Addendum rule D: never invent a benchmark. Leaving the values empty is more honest than a typed "typical" figure. |
| `docs/studio/metros.md` | The selection rule, the expected five metros marked UNVERIFIED, what each is meant to test (as the plan's rationale), the Census tables to open, a citation table and a county FIPS table to fill, the "Why these cities?" answer and how to add a sixth. | The plan forbids stating any fact about a metro from memory. The structure is ready; the facts wait for a cited source. |
| `docs/studio/data_sources.md` | The national data layer and the per-metro enhancements: what each source is for, its connector, its fallback, and its terms to verify (with the page to check). Also the commercial-rent plan and the caching / fixture / provenance rules. | Terms and limits change and could not be opened here. A wrong license claim is worse than "to verify". |
| `docs/studio/open_questions.md` | Risks and decisions for the owners: network access, keys and accounts, storage, which metro first, the startup-policy gap, roles, licenses that matter once money is involved, owner-only business items, and process items. | Some blockers are decisions, not code. Writing them down early avoids building on a wrong assumption. |
| `docs/studio/walkthrough_s0.md` | This file. | CLAUDE.md rule 5 and addendum rule A. |
| `CLAUDE.md` (appended) | The Studio track addendum, copied verbatim from the plan, under "Studio track addendum (pending Rahul's sign-off)". | The plan says S0 writes it in after both owners agree; it is marked pending until they do. |
| `docs/build_prompts.md` (appended) | A "Founder Studio (merged into this project)" section: the merge decisions and one line per Studio phase, with what each is shared with. | So anyone reading the main plan sees that Studio is part of it, not a side project. |

## 2. Design choices

1. **One product, not two.** Studio uses the lender side's ledger, loan math, approval rules, decision log, LLM
   layer and provenance renderer. The only new code is what the borrower side truly needs (profiles, metros, data,
   SiteIQ, simulator, path to approval, package).
2. **Status labels everywhere.** Every box and module is "exists", "in progress", "shared with Phase N" or "new in
   Studio". This makes it hard to claim something is built when it is not.
3. **Two new ledger statuses, one ledger.** Studio needs USER_STATED (what the founder typed) and ASSUMPTION (a
   labeled range). These extend the Phase 4 status list instead of creating a second ledger.
4. **A fictional lender policy, labeled.** The S3 DSCR threshold must be cited or labeled "lender policy, editable".
   The SBA source cannot be opened here, so the proposal is to use the SYNTHETIC Northwind constants, labeled as a
   fictional lender's editable policy.
5. **Empty cells instead of guesses.** Business-type numbers, metro facts, county codes and data-source terms are
   left as "to source", "to fill" or "to verify". A table with honest blanks is still useful: it is the S1 to-do
   list.
6. **Proposals are marked as proposals.** Role mapping, which metro first, the vintage freeze rule and the fixture
   policy are suggestions for the owners, not decisions.

## 3. Findings worth knowing

- **The reference policy is for operating businesses.** A pre-revenue founder file would always end in "request more
  information" or "refer" under the SYNTHETIC Northwind policy. This is the biggest design question for the
  two-sided handoff (open question C1).
- **The Phase 1 risk models do not apply to founders.** They are a consumer credit model (not trained here) and a
  French motor insurance model. Studio does not call `/score` for founder files.
- **`risk/scenario.py` has no revenue or rate levers.** The plan's S3 text assumed it did. The levers are in
  `agent/policy_credit.py` and the planned Phase 4 scenarios.
- **The risk client cannot get a successful score today** because of a schema mismatch with the service (already
  documented; a fix is planned in the Phase 5 additions).
- **Licenses matter once money is involved.** OpenSanctions is used under a non-commercial license, and the qwen2.5
  3B license notes disagree between sources (open questions D1 and D2).
- **Name clash.** "S1"-"S6" already name innovation items in `docs/innovations.md`.

## 4. Limitations

- Nothing about the metros or the data sources is verified. The docs say exactly what to open when the hosts are
  reachable.
- The business-type lists of costs, capex, location factors and risks are a starting checklist from general
  knowledge, not sourced data.
- The architecture is a plan. Each Studio phase may change details, and its walkthrough must say so.
- No result, metric or business value exists for Studio yet. Anything that is not run stays "not measured".

## 5. Likely interview questions

1. **Why merge the borrower and lender sides into one project?**
   Because they need the same things: numbers with sources, the same loan math, the same stress tests, and a human
   who decides. If the founder's tool and the lender's tool computed DSCR differently, they would disagree for no
   good reason. One repo means one ledger, one set of formulas and one approval path. The founder's package arrives
   at the lender already structured, and the lender sees the same evidence the founder saw.

2. **Why these five cities, and how do you add a sixth?**
   The rule is "the five largest US metropolitan areas by population in the latest Census Bureau estimates". It is
   objective and anyone can check it, and the next city is simply the sixth on the same list. Adding it is a config
   file: the counties from the Census delineation file, the states, the trade-area settings and the enabled data
   enhancements. Then we download its data and run the coverage report. Studio code changes only if the new metro
   adds a state whose traffic data needs a new adapter. In this environment the ranking itself could not be
   confirmed, so the docs still mark the list UNVERIFIED.

3. **Why is so much marked UNVERIFIED?**
   Because we could not open the sources. The build environment blocks the Census, BLS, FRED, OpenStreetMap,
   FHWA, HUD and city data hosts. The project's rules say never to state a fact about a city from memory and never
   to invent a benchmark. So instead of guessing, each doc says what is expected, why, and exactly which page will
   confirm it. That is the same honesty rule the repo applies to metrics: "not measured" instead of a made-up number.

4. **How does Studio reuse the evidence ledger?**
   The Phase 4 ledger holds claims: a value, a unit, a status and where it came from. Studio puts its numbers into
   the same ledger. A Census value is VERIFIED with the dataset row as provenance, a founder's typed rent is
   USER_STATED, a simulator output is MODEL_ESTIMATE with its version and seed, a profile range is ASSUMPTION, and a
   missing input is UNKNOWN. Studio only adds two statuses to the list. The existing renderer then fills the
   founder's narrative from that ledger, and the lender sees the same claims.

5. **Why can the LLM not type a number in the founder's plan?**
   Wrong numbers are the most expensive error in a financial document. With the SIG design, the model writes prose
   with slots such as `{{claim:dscr_year2}}`, and a deterministic renderer fills the value from the ledger. A
   validator rejects digits, number words, symbols and unslotted comparisons. So every number in the rendered plan
   is traceable by construction, and the plan still works with no LLM, using a template.

6. **Why doesn't Studio use the Phase 1 risk models to score a founder?**
   They do not fit. The credit model is trained on consumer Lending Club loans (and is not trained in this
   environment), and the insurance models are for French motor policies. A new car wash or laundromat has none of
   their input fields. Using them would be making up a score, so any model-based probability of default for a
   founder file is recorded as UNKNOWN with that reason.

7. **What stops Studio from making or promising a lending decision?**
   Three things. The addendum forbids it: Studio never approves, declines, promises approval or recommends a lender,
   and every results page shows a disclaimer. The code has no approve or decline tool, and the narrative validator
   rejects decision wording. And on the lender side, a decision can only be recorded through the existing human
   approval step, which checks the reviewer's role, authority limit and override reason.

8. **What happens in a city with thin data?**
   Every site result carries a data-coverage score that says which factors are backed by data and which are
   assumptions or unknown. Missing factors are never filled silently. If a metro falls below a coverage threshold,
   the app runs it in "limited mode" and says so. The plan calls this "coverage is shown, not hidden".

9. **Why not fill the business-type numbers with typical industry figures?**
   Because we would be stating them from memory, without a source we opened. The rule is that every benchmark has
   a source with a date and a page, or it is an ASSUMPTION with a range that the user can edit. S1 will source what
   it can. Whatever stays an assumption is shown as one, and the simulator's tornado chart shows how much the result
   depends on it.

10. **What is the biggest risk in the plan right now?**
    Two things. First, data access: without network access to the public data hosts, S1 can only build connectors
    against synthetic fixtures. Second, the lender policy: the existing reference policy is written for businesses
    with a revenue history, so a startup file would always be referred. The owners need to decide how the lender
    side treats a new business before the handoff (S6) is meaningful.
