# Walkthrough: Studio S4, the founder ledger and the provenance-locked narrative

Written on 2026-10-14 so that both Rahul Vajja and Bhavani Adula can explain this phase in their own words. (The
Studio addendum in CLAUDE.md is still a proposal until both owners sign it; this phase follows it anyway.)

**What S4 is.** A founder's plan is full of numbers: what they typed (loan, equity, a rent quote), what Studio
assumed for them (cost shares, wages, the spread over prime), what the public data says about the site, what the
simulation estimates (chances, medians, break-even), and what the SYNTHETIC startup program requires. S4 puts every
one of those numbers into the project's ONE evidence ledger (`uwcopilot.evidence`, built in Phase 4) as a claim that
says how much to trust it, and writes the plan's summary so that it cannot contain a number the ledger does not hold:
the writer (a fixed template, or an LLM) may only place slots like `{{claim:SIM.p_dscr_year2}}`, and a deterministic
renderer fills in the values. Every probability is labelled "simulated, based on stated assumptions", every text ends
with the Studio disclaimer, and nothing in S4 approves, declines or recommends a lender.

**What was done and what was not.**
- Done: `build_founder_ledger` (also exported as the build plan's `ledger_from_plan`); claims for founder inputs,
  documents, assumptions, public data, SiteIQ, the simulation, the path search, program thresholds and Studio
  targets; program arithmetic with a "never more certain than its inputs" status rule; checks against the SYNTHETIC
  startup program (NSP sections); the deterministic template narrative; the LLM path (prompt registry, validation,
  one retry with the validator's findings, template fallback); the `hf_inference` provider in the LLM layer
  (disabled by default, tested only against a mocked endpoint); a bound checker and its mutation test on founder
  narratives; the 36 golden scenarios; `scripts/eval_studio_evidence.py`; tests; Make targets.
- Not done here (S6 owns them): the PDF package, `to_application`, the agent handoff and the startup items of
  `evidence/open_items.py`. S4's "share of package numbers with provenance" is measured on the rendered narrative and
  on an evidence appendix of every claim; the PDF itself is measured by S6.
- Never run: the `hf_inference` provider. It can cost money and the owners decided on no extra spending. No document
  in this repository may say it ran.

## 1. Files

### Package `src/uwcopilot/studio/narrative`

| File | What it does |
|---|---|
| `__init__.py` | The public API: `build_founder_ledger`, `ledger_from_plan`, `render_narrative`, `NarrativeResult`, `FounderDocumentValue`, `template_draft`, `check_draft`, `all_prompt_hashes`. |
| `ids.py` | Every claim and assertion id in one place (prefixes PLAN, FOUNDER, DOC, ASSUME, RATE, SITE, SIM, TORNADO, STRESS, PATH, PROGRAM, TARGET, CALC; assertions start with `A-`). Ids fit the slot grammar. |
| `units.py` | One rule for units and display formats: shares and probabilities become percents (`pct1`), dollars `usd0`/`usd2`, months `months`, counts `int`, small values `ratio2`. |
| `derive.py` | The derived-number status rule (section 3) and small constructors: `derived_claim`, `policy_claim`, `model_claim`, `assumption_range_claim`, `public_data_claim`. |
| `models.py` | `FounderDocumentValue` (a value read from an uploaded document, with page and line), `AttemptRecord` and `NarrativeResult`. |
| `founder_claims.py` | What the founder typed: USER_STATED, or VERIFIED / CONFLICTING when a document backs or contradicts it; owner facts (UNKNOWN when not entered); other debt. |
| `assumption_claims.py` | Every simulator input the founder did not type (profile and metro ranges, public data), the build-out assumption, the note rate, and the site's demand range. |
| `site_claims.py` | SiteIQ: overall score (MODEL_ESTIMATE), coverage (VERIFIED calculation), each factor with its locator or its UNKNOWN reason, and the metro inputs. |
| `simulation_claims.py` | The S3 outputs as MODEL_ESTIMATE claims with the run identity (simulator version, seed, paths, months, plan SHA-256). |
| `path_claims.py` | The S5 search: status, probabilities before and after each change-set, each lever's new value. |
| `program_thresholds.py` | The SYNTHETIC program's thresholds (imported from `agent/policy_startup.py` and `policy_credit.py`, never retyped) and the Studio targets, as VERIFIED policy claims with their comparator. |
| `program_calcs.py` | The program arithmetic (project cost, equity injection, debt service, reserve, projected and stressed DSCR) with the program's own rounding. |
| `program_checks.py` | The checks (assertions) with their NSP sections, trigger ids and the phrases the renderer prints. |
| `founder_ledger.py` | `build_founder_ledger`: assembles the claims with `Ledger.builder()`, checks that plan, simulation and SiteIQ describe the same plan (`PlanMismatch`). |
| `prose.py` | The plain words around the slots (business type, metro, inputs, factors), each checked with the grammar validator before use. |
| `template.py` | The deterministic slot-only draft (the default writer and the fallback); `required_refs` and `allowed_sections` for the validator. |
| `prompts.py` | The prompt registry (`founder_narrative_v1`, `founder_narrative_retry_v1`, and the template's own id and hash) and the request the LLM gets. |
| `generate.py` | `render_narrative`: template by default; for a model, validate, retry once with the findings, render, audit, fall back to the template. |
| `checker.py` | The bound check (each number must equal a claim its own sentence cites) and the mutation test on founder sentences. |
| `golden.py` | The 36 golden SYNTHETIC scenarios (6 business types x the SYNTHETIC metro and the 5 real metros in their current mode). |
| `evaluation.py` | Per-scenario measurements and pooled rates with Wilson and scenario-bootstrap intervals. |

### LLM layer

- `src/uwcopilot/llm/hf_inference_client.py`: the Hugging Face Inference Providers client (OpenAI-compatible
  chat-completions router). Off unless `UWC_HF_INFERENCE_ENABLED=1`, `HF_TOKEN` and a model are set, and the model
  has a verified price in `config/llm_pricing.yaml` (otherwise the budget layer refuses the call before any request).
- `src/uwcopilot/llm/factory.py`: `get_llm_client("hf_inference", ...)` builds it as `Cached(Budgeted(client))`, like
  every paid provider. It is not in `settings.PROVIDERS`, so `UWC_LLM_PROVIDER` cannot select it by accident.

### Script, results, tests, Make targets

- `scripts/eval_studio_evidence.py` writes `results/studio_evidence_metrics.json` (template provider) or
  `results/studio_evidence_metrics_<provider>.json` (an LLM provider), each with a `uwcopilot.governance.provenance`
  block (git SHA, prompt hashes, config file hashes, metro snapshot fingerprints, model digest for a local model).
- Tests: `tests/studio/test_narrative_ledger.py` (statuses, provenance, documents, conflicts, unknowns, arithmetic
  against the lender side), `test_narrative_render.py` (template validity for every business type, the LLM path with
  the fake client: retry, fallback, unreachable model, hf_inference refused; the prompt holds no values and no
  founder text; the registry hashes), `test_narrative_checker.py` (bound check vs round-trip audit, the mutation
  test), `test_narrative_golden.py` (36 scenarios, reproducibility, a real metro in its current mode, the script),
  `tests/llm/test_hf_inference_client.py` (disabled by default, request shape, errors, cost cap; every HTTP call goes
  to `httpx.MockTransport`).
- Make: `make studio-evidence`, `make studio-evidence-ollama`, `make studio-narrative-test`.

## 2. Statuses: what each founder number becomes

| Where the number comes from | Status | Provenance |
|---|---|---|
| Typed by the founder (loan, equity, term, a quote) | USER_STATED | kind `user_input`, the plan field |
| Typed AND printed the same by an uploaded document | VERIFIED | kind `document`, citation doc id / page / line |
| Typed, but a document prints a different value | CONFLICTING (both) | each links the other (`conflict_ids`) |
| Marked "verified" by the founder without a document in the ledger | USER_STATED | a note says why it stays USER_STATED |
| Profile or simulator range (cost shares, ramp-up, shocks) | ASSUMPTION | source "unsourced" (the profile's note says what should replace it), range low-high |
| Profile value from a cited publication | ASSUMPTION | the publication as source, range |
| Public data row (Census, BLS, HPMS, Overture via SiteIQ) | VERIFIED | kind `public_data`, locator: publisher, dataset, URL, retrieval date, response SHA-256, geography |
| SYNTHETIC city value (the demo metro) | VERIFIED | kind `public_data`, label and source say SYNTHETIC |
| Public data converted with an assumption (weekly wage / hours) | ASSUMPTION | the data source, range |
| The site demand handed to the simulator | MODEL_ESTIMATE (SiteIQ), USER_STATED (founder's range) or ASSUMPTION (a SYNTHETIC or assumed range) | |
| Simulation output (chances, medians, break-even, tornado, stress) | MODEL_ESTIMATE | model `studio-feasibility` and version, seed, paths, months, plan SHA-256 |
| SiteIQ score | MODEL_ESTIMATE | model `siteiq`, uncertainty "not measured per site" |
| Path search output | MODEL_ESTIMATE | model `studio-path-to-approval` and version |
| Program threshold, Studio target | VERIFIED | kind `policy`, the NSP section or the target's source, its comparator |
| Needed but absent (credit score not entered, a factor without data) | UNKNOWN | the reason; nothing is filled in |

## 3. Design choices

1. **One ledger, not a second one.** The Phase 4 ledger already had the six statuses (USER_STATED and ASSUMPTION
   were added for Studio), the validators that tie a status to its provenance kind, and the renderer protocol.
   `Ledger.builder()` was made for callers like Studio. S4 only adds constructors and ids.
2. **A derived number is never more certain than its least certain input.** Order: UNKNOWN < CONFLICTING <
   MODEL_ESTIMATE < ASSUMPTION < USER_STATED < VERIFIED. The equity injection computed from the founder's figures is
   USER_STATED; the required reserve, which uses assumed wages and rent, is an ASSUMPTION with a range computed from
   its inputs' ranges; the projected DSCR, which uses the simulated median NOI, is a MODEL_ESTIMATE. This is stricter
   than the lender ledger, which calls its arithmetic VERIFIED: a founder must never read a number computed from their
   own guesses as if it had been checked. A test fails if any CALC claim of a golden plan is VERIFIED.
3. **The program's own arithmetic.** Ratios and percentages use `policy_startup.program_ratio` / `program_pct` (exact
   decimals, rounded half up) and the NBC-4.1 payment, so the founder sees the number the lender copilot will compute.
4. **Slots only, values never in the prompt.** The model sees slot ids and what each one means, never a value, so it
   has nothing to copy. The renderer types every number. The grammar (provenance/memo_grammar.py) rejects digits,
   number words, currency and percent symbols, unknown or malformed slots, comparative words outside a sentence with
   an assert slot, omitted findings or open claims, decision wording and unknown citations.
5. **One retry, then the template.** A rejected draft goes back once with the validator's findings (code and the text
   that broke it). If the second draft fails too, or the model cannot be reached, or the rendered text fails the
   audit, the deterministic template is used and the result says so (`fallback_used`, `fallback_reason`). The template
   is the default because it always works, costs nothing and gives the same text for the same ledger.
6. **Founder free text never reaches a prompt or a code field.** Site labels and the names of other debt stay in
   claim values and details; the words around the slots come from fixed tables, and anything read from config (a
   metro name, a lever label) is run through the grammar validator before it is used.
7. **Assertion phrases are written for the checker.** The phrases the renderer prints ("meets the program's minimum",
   "falls short of the stress minimum") hold no digits, no decision words, and no comparator except "minimum" right
   before the threshold it describes. Otherwise the bound checker would read, say, "below" as the comparator of the
   founder's own number.
8. **Two checks, and why.** The round-trip audit (Phase 4 / SIG) asks "does this number equal some claim of the
   ledger?". It cannot see a right number in the wrong place. The bound check asks "does this number equal a claim
   that THIS sentence cites through its footnote marker, with the threshold's comparator?". Rendered text always has
   the markers, so the binding is free.
9. **Golden scenarios reuse the S5 generator.** The plan, loan and demand range come from
   `path_to_approval/synthetic_plans.py` (anchored so plans span short / near / past the lender's stress), plus
   SYNTHETIC founder facts. SiteIQ scores a seeded tract centroid when the metro has tracts here. The simulation uses
   the generator's demand range (recorded as an ASSUMPTION sourced to the generator), not SiteIQ's estimate. A metro
   without a prime rate in this environment gets a SYNTHETIC quoted rate, noted on the scenario.
10. **hf_inference: three locks.** Disabled flag, token and model, verified price (cost cap). A refusal in
    `render_narrative` is a template fallback with the reason, never an error for the founder.

## 4. How a narrative is made

1. `build_founder_ledger(case, feasibility, site_score, path_result)` builds the claims (sections 2-3) and the checks.
2. `template_draft(ledger)` writes the slot-only draft from the ledger: the plan and eligibility checks; the site; the
   simulated chances, the median second year and the break-even; the program's coverage checks; the top drivers;
   the path search; open items (every UNKNOWN or CONFLICTING claim); a closing reminder that the lender decides.
3. With a model, `prompts.build_request` sends the rules, the slots with their meanings and statuses, the checks with
   their sections, the slots it must use and the template draft as an example of the style.
4. `check_draft` validates; `render` fills the slots and adds one footnote per claim (label, provenance, inputs
   hash); `round_trip_audit` must find every number of the body in the ledger.
5. The text ends with "Every probability above is simulated, based on stated assumptions." and the disclaimer.

## 5. How the evaluation works (`scripts/eval_studio_evidence.py`)

For each golden scenario (seeded), the script builds the ledger, renders the template narrative, counts the numbers
and the ones each check cannot trace, renders every claim of the ledger on its own line (the evidence appendix) and
counts again, and runs the mutation test: every sentence of the rendered narrative that holds a number becomes a
reference sentence whose "section" is the evidence text of the claims its markers point to, and
`provenance.mutations.build_mutations` breaks one number (value_perturb, magnitude_x10, unit_swap, comparator_flip,
cross_section with a donor from another sentence of the same narrative) or keeps it (unmutated,
equivalent_rewrite). wrong_row does not apply: a founder narrative has no tables. With an LLM provider it also asks
the model for each narrative and records first-try validity, validity after the retry, the fallback and its reason,
the validator's findings and latency. Rates are pooled over scenarios with Wilson 95% intervals and bootstrap 95%
intervals that resample scenarios (1,000 resamples, seed 42).

## 6. Results

RESULTS_PLACEHOLDER

## 7. Limits and open issues

- **The template's numbers trace by construction.** The renderer types every number from the ledger, so a 100% traced
  share for the template is a check that the construction holds on every scenario, not evidence about how often a
  model would write a wrong number. The LLM run (section 6) is the measurement of a model.
- **The mutation test uses sentences we wrote.** The founder sentences come from the template, the same author wrote
  the mutations' rules, and the bound check's recall on cross_section follows from its design (the donor value is
  drawn so that it is absent from the cited claims). It shows the implementation has no gap (sentence splitting,
  marker mapping, comparators); it is not a held-out benchmark.
- **Display rounding.** The renderer has one percent format with one decimal (`pct1`), so a note rate of 8.25% prints
  as 8.2%. The claim keeps the exact value and the template does not print the rate; S6 should print rates from the
  claim value with two decimals or add a display format to the shared renderer.
- **"(stated by the applicant)" on computed figures.** A CALC claim computed only from the founder's figures is
  USER_STATED, so the renderer prints that suffix (for example after the equity injection); its footnote says
  "calculated from the founder's figures". The suffix text belongs to the shared renderer and was not changed.
- **Text claims hold no checked numbers.** The round-trip audit ignores claims displayed as text, so a text value with
  a digit (a founder's site label "Lot 7") would show as untraced; the template therefore never prints a site label.
- **Prompt registry.** The founder prompts are registered in `studio/narrative/prompts.py` with the same id + SHA-256
  convention as `rag/prompts.py`, not inside `rag.prompts.PROMPTS` (whose builders take a question and guideline
  hits). Phase 6's release gate should read `narrative.prompts.all_prompt_hashes()` too.
- **Golden scenarios.** The demand comes from the S5 generator, not from SiteIQ; real metros run in this machine's
  mode (all five were "limited" here, coverage below the 0.80 threshold, with snapshots present); the generated loan
  amounts are often above the program's $500,000 maximum (the profiles' project sizes), so many narratives report
  that finding. None of the golden plans carries a document, so CONFLICTING appears only in the unit tests.
- **hf_inference never ran.** It has no price in `config/llm_pricing.yaml`, so even with the flag and a token the
  budget layer refuses the call. Its variables are documented here and in the module docstring, not in the shared
  `.env.example`.
- **Ledger size.** About 380-430 claims per plan, most of them monthly and yearly simulation bands. A ledger-wide
  check gets weaker as the ledger grows (more values to match by accident), which is one reason for the bound check.

## 8. Likely interview questions

**1. Why put every number in a ledger instead of printing the simulation's results?**
Because a founder (and later a lender) has to know how much to trust each number. A typed loan amount, an assumed
cost share, a Census row and a simulated chance look the same on a page. In the ledger each one carries its status
and where it came from: the field the founder typed, the profile range and "unsourced", the dataset URL with the
retrieval date and the response's SHA-256, or the simulator version, seed, number of paths and the plan's hash. The
narrative and the package can only show what the ledger holds.

**2. How does a founder's number become VERIFIED?**
Only with a locator. When the founder uploads a document and its value (with page and line) matches the typed figure
(dollars within $1.00), the claim becomes VERIFIED with that citation. If the document prints a different value, both
become CONFLICTING and link each other, and the narrative lists them as open items. Ticking "verified" without a
document leaves the figure USER_STATED, with a note saying why.

**3. Why is the projected DSCR a "model estimate" and the reserve requirement an "assumption"?**
A computed number is never more certain than its least certain input. The projected DSCR divides the simulated median
NOI, a model estimate, so it is a MODEL_ESTIMATE. The required reserve uses assumed wages and rent, so it is an
ASSUMPTION, with a range computed from its inputs' ranges. The equity injection uses only the founder's figures, so
it stays USER_STATED. A test fails if any computed figure of a golden plan is VERIFIED.

**4. How can a model write a narrative full of numbers if it may not type a digit?**
It writes slots, `{{claim:SIM.p_dscr_year2}}`, and checks, `{{assert:A-NSP-4.1}}`. The prompt lists each slot with
what it means, never its value. The validator rejects digits, number words, currency and percent signs, unknown
slots, comparative words outside a sentence with a check, missing findings and decision wording. The renderer then
fills every slot from the ledger and adds a footnote, and the round-trip audit re-reads every number.

**5. What happens when the model breaks the rules?**
It gets one retry with the validator's findings ("V1_DIGIT: 250000"). If the second draft fails too, the model cannot
be reached, or the rendered text fails the audit, the deterministic template is used and the result says so
(`fallback_used`, `fallback_reason`). The local 1.5B model's fallback rate is in section 6.

**6. The round-trip audit already exists. Why add a bound check?**
The audit asks whether a number equals some claim of the ledger. With about 400 claims, a wrong number often equals
some other claim: on the founder sentences it missed every value moved between sentences and every flipped
comparator, and some perturbed values (section 6). The bound check uses the footnote markers the renderer already
writes: a number must equal a claim that its own sentence cites, with the threshold's comparator ("minimum 1.25").

**7. Is a recall of 1.0 a real result?**
It shows the implementation holds on 686 sentences from 36 narratives, for the error types we can generate. It is not
a claim about real errors: the mutations break the template's own sentences, the same author wrote both, and for
cross_section the result follows from the design. We report the round-trip audit on the same mutations so the
difference is visible, and wrong_row is reported as not applicable because a founder narrative has no tables.

**8. Why is hf_inference disabled, and how would the owners turn it on?**
It can cost money and the owners decided on no extra spending. It needs `UWC_HF_INFERENCE_ENABLED=1`, `HF_TOKEN`, a
model id, and the model's verified price in `config/llm_pricing.yaml`; without the price the cost-cap layer refuses
every call before a request is sent. It goes through the same Cached(Budgeted(...)) wrapper as the other paid
providers, and `UWC_LLM_PROVIDER` cannot select it. It is tested only against a mocked endpoint and never ran.

**9. Could a founder's text inject instructions into the model?**
Founder free text never reaches the prompt: the prompt holds slot ids, fixed meanings from code tables, statuses,
section ids and the template draft, all built from code and config, and config phrases are checked with the grammar
validator. A test checks that the site labels and the loan value are not in the prompt. The system prompt also says
the data block is data, not instructions, and the validator would reject decision wording anyway.

**10. Why is the template the default?**
It always works, costs nothing, needs no network and gives the same text for the same ledger, which is what the Space
needs. A model can make the prose read better, but on CPU it is slow and a small model often breaks the grammar; the
fallback keeps the founder's page correct either way, and the results show how often it was needed.
