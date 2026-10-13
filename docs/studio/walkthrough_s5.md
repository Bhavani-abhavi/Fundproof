# Walkthrough: Studio S5, path to approval

Written on 2026-10-13 so that both Rahul Vajja and Bhavani Adula can explain this phase in their own words. (The
Studio addendum in CLAUDE.md is still a proposal until both owners sign it; this phase follows it anyway.)

**What S5 is.** A founder's plan often misses the simulated targets. S5 answers: "what is the smallest change to MY
plan that would reach them?" It tries changes the founder can actually make (borrow less or more, put in more of
their own cash, keep a bigger cash reserve, ask for a longer term, spend less on the build-out, change the price,
open more or fewer days, pick another of their own candidate sites) one lever at a time, then two at a time. It
simulates each variant with the S3 simulator on the same random draws, and returns up to three smallest change-sets
with before and after probabilities and a plain-language explanation. Every probability is labeled **"simulated,
based on stated assumptions"**, and every answer says that reaching the targets does not mean a lender will approve
the loan (Studio rule C).

**The three default targets (all editable).**
1. P(year-2 DSCR >= 1.25) >= 0.80;
2. P(year-2 DSCR >= 1.00 under the lender's revenue stress) >= 0.80: revenue 20% lower and NOI lower by the same
   dollars, applied to every simulated path (NSP-4.2 of the SYNTHETIC startup program);
3. P(cash < 0 within 24 months) <= 0.10.

**What was done and what was not.**
- Done: `find_paths()` (single-lever moves, then pairs; common random numbers; a cache keyed by plan hash and lever
  values; the three smallest change-sets; before/after probabilities; explanations); guardrails that only let plan
  choices move; a startup-program check that refuses change-sets that would newly break the equity minimum or the
  product limits; "no path within your bounds" plus the closest result; a quick mode for the website;
  `config/studio/levers.yaml`; the evaluation script and `results/studio_path_metrics.json`; 64 tests; Make targets.
- Checked, not assumed: every change-set the evaluation returned was re-simulated with S3's own `simulate()` and gave
  the same three probabilities; every applied plan was compared field by field with the founder's plan (no fact
  moved); the heuristic search was compared with a brute-force search on 90 plans.
- Not done: the website does not call it yet (S7 will, through quick mode); the S6 package does not carry the path
  history yet. Change-sets of three or more levers are not searched (the build plan asks for singles then pairs).
  A smaller build-out that also cuts capacity is not modeled, so the build-out lever only means "the same business
  for less". No real-world accuracy can be measured: there are no real founder outcomes, so the results file says
  "not measured".

## 1. Files

### Package `src/uwcopilot/studio/path_to_approval`

| File | What it does | Why it is built this way |
|---|---|---|
| `__init__.py` | Exports `find_paths`, `prepare_search`, the models, the default targets, the settings presets, `clear_caches`. | One import line for the web app and the S6 package. |
| `models.py` | Frozen pydantic models. Inputs: `PlanCase` (profile, metro, the site's demand, the S3 `Plan` and `Loan`, and `FounderFacts`), `FounderFacts` (credit score, experience, other debt, verified fields), `OtherDebt`, `Candidate`, `Target`, `Change`, `SearchSettings`. Outputs: `Outcome` (the three probabilities, the lender's base-case ratios, targets missed, shortfall), `ProgramChecks`, `PlanAmounts`, `AppliedChange`, `ChangeSet` (with `applied_plan`, the whole changed plan), `LeverSummary`, `SearchStats`, `PathResult`. | `extra="forbid"` and `frozen=True`, as everywhere in Studio. Facts live in their own model so the guardrails can compare them as one block. `Outcome.label` only accepts "simulated, based on stated assumptions". |
| `targets.py` | `default_target()` (the three targets, each with its source), `custom_target(**edits)` (marks edited fields "edited by the user"), `describe()`. | The 1.25, the 20% and the 1.00 are imported from `agent/policy_credit.py` (through `feasibility.assumptions`), never retyped. The probability levels (0.80, 0.80, 0.10) are labeled as design choices of the build plan, not lender rules. |
| `levers.py` | Loads `config/studio/levers.yaml` and checks it against the code's allow-list (`LEVER_NAMES`, each with a fixed kind and balancing lever). Resolves bounds, builds each lever's search grid, computes effort. | A config edit can never turn a fact into a lever: the loader refuses any name not on the allow-list, and what a lever writes is code, not config. |
| `space.py` | `build_lever_space()`: the plan's money (sources = uses), each lever's current value, bounds and grid, and why a lever is not available (locked, verified, not applicable, needs an input the plan lacks). `LEVER_WRITES`: the plan fields each lever may write. | Decided once per search, so the guardrails and the search read the same numbers. A lever that is not available always says why (shown to the founder). |
| `guardrails.py` | `apply_changes()`: checks each lever, adds up the dollar moves, rounds them to the cent, checks every amount and bound, builds the changed plan, then compares every field of the old and new plan and refuses any change outside the levers' own fields. `NOT_A_LEVER`: names people might try (credit score, experience, other debt, note rate, demand, assumptions...) with the reason each is not a plan choice. `GUARDRAIL_RULES`: the rules shown on every result. | Defense in depth: an allow-list on the way in AND a field-by-field diff on the way out. A test monkeypatches the plan builder to slip a credit-score change through and checks the diff catches it. |
| `program.py` | `program_checks()`: the SYNTHETIC startup program's equity injection (NSP-3.1, >= 20.0%), loan limits and term range (NSP-8.1), with the thresholds and the half-up rounding imported from `agent/policy_startup.py`; `newly_broken()`. | A change-set that reaches the simulated targets but breaks a plain program rule the plan met (for example borrowing so much that the owner's share falls below 20%) is not a path. Rules the plan already fails are shown, not repaired. |
| `evaluate.py` | `PathEvaluator`: draws the plan's random numbers once and simulates every variant on them (common random numbers); `outcome_of()` computes the three probabilities, the NSP-4.1 and NSP-4.2 base-case ratios and the shortfall; other debt is added to debt service, net income and cash. `EvaluationCache`: a bounded least-recently-used store behind a lock; `plan_hash()`. | One function turns a simulated plan into the three numbers, so the search, the quick-mode re-check and the tests cannot disagree. |
| `search.py` | `find_paths()` and `prepare_search()`: load the metro, profile and levers (cached once per process), build the lever space and evaluators, run `Search` (singles, then pairs with a screen and a staircase), rank, re-check in quick mode, explain. `FULL_SETTINGS`, `QUICK_SETTINGS`, `clear_caches()`. | `prepare_search()` is shared with the brute-force reference, so the reference sees exactly the same grids, bounds, guardrails and draws. |
| `explain.py` | Plain-language sentences for each change, the plan after the change, each probability before -> after against its target, the lender's base-case ratios, what stayed unchanged, the status message. | Every number in the text comes from the search result; the text adds none of its own. It never says a loan will be approved. |
| `sites.py` | `candidate_from_site_score()`: a SiteIQ `SiteScore` (S2) as a candidate site, read through the simulator's own adapter (`feasibility.inputs.as_site_demand`). | A change inside SiteIQ cannot break the search. |
| `exhaustive.py` | `exhaustive_search()`: tries every single value and every pair of values on the same grid, with the same guardrails and draws. For tests and the evaluation only. | The only honest way to say how often the heuristic misses a path. |
| `synthetic_plans.py` | Seeded SYNTHETIC plans for the evaluation and the tests (section 7). | Separate from the search so the plan generator can be read and criticised on its own. |
| `errors.py` | `GuardrailViolation` (not a plan choice), `LeverError` and `OutOfBounds` (a plan choice with a value the plan cannot take). | The search skips `LeverError`s quietly (a value outside the bounds is simply not a candidate) but never swallows a `GuardrailViolation`. |

### Config, script, tests, Make targets

| File | What it does |
|---|---|
| `config/studio/levers.yaml` | The eight levers with label, unit, default bounds, rounding step, effort weight and a note. `status: DESIGN_CHOICE`: every number is the author's choice of how far to look by default and how hard a change is assumed to be; none is data. |
| `scripts/eval_studio_paths.py` | The evaluation (section 7) -> `results/studio_path_metrics.json` with a provenance block (`uwcopilot.governance.provenance.RunProvenance`). Options: `--plans`, `--exhaustive`, `--metros`, `--business-types`, `--output`. |
| `tests/studio/path_helpers.py` | A hand-built SYNTHETIC car-wash plan that misses the targets but has single-lever fixes; fast settings. |
| `tests/studio/test_path_levers.py` | Allow-list (a config that adds `credit_score` or changes a lever's kind is refused), bounds, grids, rounding, effort. |
| `tests/studio/test_path_guardrails.py` | Every attempt to move a fact fails: each `NOT_A_LEVER` name, credit score, experience, a verified figure, other debt, a tampered plan builder (credit score, site demand), a candidate in another metro; sources = uses for every dollar lever; a cent-rounding regression. |
| `tests/studio/test_path_evaluate.py` | Outcomes equal S3's `simulate()` on the changed plan exactly; base-case ratios use the program's arithmetic; other debt; cache hits, LRU, guardrails on cache hits; the plan hash changes with everything that changes an outcome. |
| `tests/studio/test_path_monotonicity.py` | More equity, a longer term, a lower build-out and a bigger reserve never make a probability worse; borrowing less is not monotone (it trades cash for coverage). |
| `tests/studio/test_path_search.py` | Ranking, labels and disclaimer, applied plan, determinism with seeds (and with a warm cache), infeasible target, all levers locked, founder bounds, the program check, quick mode, budget, heuristic = brute force on small plans, a real metro in limited mode. |
| `tests/studio/test_path_targets.py`, `test_path_synthetic_plans.py`, `test_path_script.py` | Default targets and sources; the plan generator's determinism and anchoring; the script on tiny settings. |
| `Makefile` | `make studio-paths` (the evaluation) and `make studio-paths-test` (these tests only). |

## 2. How one search works

1. **Load once, per process.** The metro's data fabric, its mode, the business-type profile (with the metro's
   overrides), the levers file and the simulator assumptions are read once and kept (`fabric_for`, `mode_for`,
   `profile_for`, `levers_from`, `default_assumptions`). A real metro's mode alone reads every profile YAML, which
   took about 0.36 s per call before it was cached.
2. **Build the lever space.** The plan's money: equity + loan + other debt = build-out + cash reserve at opening.
   For each lever: its current value, its bounds (default from the YAML, or the founder's own), its grid (8 values
   on each side by default, rounded to $500, $0.25, whole days or the listed terms), or the reason it is not
   available.
3. **Simulate the plan as it is** (1,000 paths, 36 months, seed 42). If it meets all three targets, stop:
   `already_meets_target`.
4. **Single-lever moves.** For each available lever, try its grid from the smallest change (least effort) to the
   largest and keep the first value that meets all three targets: the smallest change of that lever.
5. **Pairs, only when fewer than three single moves were found.** Take the levers that did not reach the targets alone
   but moved the plan closer on their own. For each pair, first try both at their best single values (a screen); if
   that meets the targets, walk a "staircase" over the two sorted value lists to find the least-effort combination
   (start with the biggest change of the first lever and the smallest of the second; on success make the first
   smaller, on failure make the second bigger). A candidate site is a category, so its partner lever is scanned once
   per site.
6. **Refuse what is not allowed.** Every change-set goes through the guardrails (section 4) and the program check
   before it is simulated. A refused change-set is not a candidate; it is not an error.
7. **Rank and explain.** Fewer levers first, then less total effort. Return the first three, each with before and
   after probabilities, the plan after the change, the lender's base-case ratios and an explanation. If none meets the
   targets: `no_path_within_bounds`, "No path within your bounds", and the closest result (the smallest total
   shortfall seen), when one moved the plan closer.

**Common random numbers.** Every variant of a plan is simulated with the SAME random draws (each uncertain input keeps
its name, so it keeps its uniforms). The difference between two variants is then the effect of the change, not the
luck of the draw: with 1,000 paths a probability is only known to about +/- 0.03, so two independent runs could
easily rank a worse change above a better one. A test checks that the after-probabilities equal what S3's own
`simulate()` gives for the changed plan with the same seed, exactly.

**Quick mode (for the website).** 500 paths, 4 grid values per side and at most 400 simulations for the search, then
every change-set it returns is re-checked with 1,000 paths, so every number the founder sees is a full simulation's.
A change-set that misses a target in the re-check is dropped (the note says how many).

## 3. The targets, and why the lender stress target exists

The original build plan (Prompt S5) had two targets: year-2 DSCR and the cash shortfall. The lead engineer added the
third after the startup program was written (DELTA 2026-10-13), because the program's binding rule is the lender's
revenue stress (NSP-4.2), not the 1.25 DSCR minimum (NSP-4.1).

The stress takes 20% of year-2 revenue away and takes the SAME dollars out of NOI (every cost is treated as fixed).
The stressed DSCR is at least 1.00 only when NOI >= debt service + 0.20 x revenue. Divide by debt service: a plan
passes only when DSCR >= 1 + 0.20 x revenue / debt service. If a business keeps m cents of NOI per revenue dollar,
then revenue = NOI / m and the condition becomes DSCR >= m / (m - 0.20). Examples:

| NOI per revenue dollar (m) | DSCR needed to pass the stress |
|---|---|
| 0.50 | 1.67 |
| 0.30 | 3.00 |
| 0.25 | 5.00 |
| 0.20 or less | never passes, at any DSCR |

So a plan can clear 1.25 comfortably and still fail the lender's stress test badly. A path that only aimed at 1.25
would show the founder a "fix" that the lender's own test still flags. The evaluation measures how often that happens
(section 8, "the stress target's effect"). The 0.80 probability level for it is a design choice, like the other two.

## 4. Guardrails: only plan choices move

| May move (plan choices) | May never move (facts and assumptions) |
|---|---|
| owner equity, loan amount, build-out cost, cash reserve at opening, price, days open, loan term, which of the founder's own candidate sites | credit score, experience, other debt (never hidden or reduced), the site's demand estimate, every assumption range, the note rate and rate type, the wage, capacity, the owner's draw, the opening date, and every figure the founder marked as verified |

How it is enforced:
1. **Allow-list in code.** `levers.LEVER_NAMES`; the YAML loader refuses anything else. `Change(lever="credit_score")`
   raises `GuardrailViolation` with the reason ("the owner's credit score is a fact; it is never changed or
   misstated").
2. **Verified figures.** `FounderFacts.verified_fields` (for example `loan.amount` backed by a term sheet): every lever
   that writes such a field is not available, and forcing one raises `GuardrailViolation`.
3. **Field-by-field diff.** After building the changed plan, every field is compared with the founder's plan; any
   difference outside the levers' own fields (`LEVER_WRITES`) is a `GuardrailViolation`.
4. **Inputs the founder has not typed are not levers.** The build-out lever needs a typed build-out cost (otherwise the
   current figure is the profile's ASSUMPTION, and lowering an assumption to pass is exactly what rule D forbids). Days
   open needs a typed staffing plan (staff hours scale with open days; without the founder's number the cost of more
   days would be guessed).
5. **Bounds whichever lever moves an amount.** The reserve lever spends owner equity, so equity's bounds hold for it
   too; borrowing less lowers the reserve, so the reserve may not go below zero. Amounts are rounded to the cent
   BEFORE these checks (a regression test covers the cent that once slipped through).
6. **Program check.** A change-set may not break an NSP-3.1 or NSP-8.1 rule the plan meets now.
7. **Other debt is a fact.** Its payments count in debt service, net income and cash in every simulated month; its
   proceeds count in the cash at opening. The explanation says how many other debts were kept unchanged.

## 5. Levers, bounds and effort (`config/studio/levers.yaml`, all DESIGN_CHOICE)

| Lever | What changes | Absorbed by (sources = uses) | Default bounds | Effort weight |
|---|---|---|---|---|
| owner equity | more of the founder's own cash | the loan (borrow that much less) | current .. current + 20% of project cost | 2.0 |
| loan amount | borrow more or less | the cash reserve at opening | current - 25% .. current + 10% of project cost (never below the program's $50,000 minimum unless already below) | 1.0 |
| build-out cost | the same business built for less | the loan | 15% saving .. current | 1.5 |
| cash reserve | keep more cash at opening | owner equity (founder's own cash) | 0 .. current + 20% of project cost | 2.0 |
| price | price per unit of demand | (demand responds through the ASSUMPTION elasticity) | -15% .. +15% | 1.5 |
| days open | days per week; paid hours scale in proportion | (needs a typed staffing plan) | 5 .. 7 | 1.0 |
| loan term | ask for a longer or shorter term | | 60 .. 120 months (the program's NSP-8.1 range) | 0.5 |
| site | another of the founder's candidate sites, same metro | (its own demand estimate and rent quote) | the listed candidates | 0.3 per switch |

Effort of one change = weight x size: dollars moved / total project cost for the dollar levers, the relative change for
price, days and term, and 1 for a site switch. The weights encode one judgement: asking a lender for a longer term is
easier than finding more of your own cash. They only order change-sets that have the same number of levers.

## 6. Design choices

- **Singles, then pairs, ranked by the number of levers first.** A founder can act on one change more easily than on
  two, and one change is easier to explain and to check. Effort only breaks ties within the same number of levers.
- **The staircase assumes "further in the helpful direction never hurts".** That holds exactly for owner equity, loan
  term, build-out cost and the reserve under common random numbers (`test_path_monotonicity.py`), and does not hold
  for the loan amount (borrowing less helps coverage but lowers cash) or for price (elasticity). The brute-force
  comparison in section 8 measures what this costs.
- **Bounds the founder can tighten.** The defaults are wide on purpose (they are a first look); the founder's own
  answer to "how much more cash could you really put in?" replaces them, and a founder bound must include today's
  value.
- **Program rules as constraints, probabilities as targets.** The equity minimum and the product limits are plain
  rules a lender checks first; breaking one to improve a probability would be a worse plan, not a path.
- **Cache key = everything that changes an outcome.** The plan, its facts, the candidates, the target, the metro
  inputs, the profile (with the metro's overrides), the simulator assumptions, the number of paths, months, seed and
  both engine versions. A stale entry can never be read for another plan or after a config edit. The cache is bounded
  (20,000 outcomes, least recently used dropped) and locked, so web requests in several threads can share it.
  `clear_caches()` resets the per-process caches after a snapshot or config change.
- **Quick mode re-checks with the full number of paths.** Fewer paths and a coarser grid make the search faster;
  re-checking what it returns keeps every number shown at full precision.
- **`ChangeSet.applied_plan`.** The whole changed plan travels with each change-set, so the website's "apply this
  change" button sends it back as the founder's new plan, and S6 can put it in the package's path history.

## 7. How the evaluation works (`scripts/eval_studio_paths.py`)

**Plans.** For every metro (the SYNTHETIC metro first, then the five real metros in their current mode) and each of
the six business types, 15 SYNTHETIC plans (540 in total), seeded (`synthetic_plans.py`):
1. the plan and loan come from the S3 validation generator unchanged (`feasibility.worlds.draw_plan`);
2. half of the founders also typed a staffing plan (so the days lever can be used);
3. the plan is anchored so that its ONE-FORECAST lender-stressed year-2 DSCR (every input at its most likely value,
   no shocks, no noise) equals a target drawn uniformly from 0.6 to 2.6, first through the site's demand, then (as in
   S3) through a rent quote; a plan that neither lever can anchor is kept and counted as "none";
4. each plan gets two other candidate sites in the same metro.
The 0.6-2.6 range is a design choice. It reaches far past 1.00 because the simulated year-2 median is usually well
below the one-forecast case (ramp-up, shocks, right-skewed cost ranges, operating leverage).

**Each plan is searched four ways:** full mode (the headline), quick mode, quick mode again on the warm cache (a founder
reloading the page), and full mode with the stress target switched off (only the two original targets).

**Checks on every returned change-set:** re-simulated with S3's `simulate()` (same seed and paths); applied plan
compared field by field with the founder's plan, without using `guardrails.py`.

**Brute force:** on the 90 SYNTHETIC-metro plans, the heuristic is compared with `exhaustive_search()` on the quick
grid with 500 paths and no re-check (so both see identical outcomes).

**Statistics:** rates and means with bootstrap 95% CIs (1,000 resamples over plans, seed 42). Runtimes are wall time
of one `find_paths()` call in a warm process on a shared 4-CPU machine used by other agents at the same time; the
load average at the start and end of the run is in the provenance block.

## 8. Results (copied from `results/studio_path_metrics.json`)

RESULTS_PLACEHOLDER

## 10. Likely interview questions

QA_PLACEHOLDER
