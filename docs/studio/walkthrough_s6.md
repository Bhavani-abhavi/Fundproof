# Walkthrough: Studio S6, the lender-ready package and the handoff into the lender copilot

Studio S6 turns one founder plan into a **lender-ready package** (a JSON document, a PDF and a startup credit
application) and **hands it to the lender copilot**: the existing LangGraph agent runs on the application and stops
at the human checkpoint. Nothing is approved or declined without a person.

Everything here is **SYNTHETIC**: the founders, owners, businesses, documents, the SYNTHETIC city and the Northwind
startup program. Every page of the PDF says: "Educational planning tool. Not financial, legal or lending advice.
Lenders make their own decisions." Probabilities are "simulated, based on stated assumptions".

Written so both owners (Rahul Vajja and Bhavani Adula) can explain the work.

```
FounderProject (S2-S5 results + owners, entity, documents)
   │ build_package
   ▼
Package  ── one founder ledger (S4) ── sections of blocks that point at claim ids
   │ render_pdf (re-audits the text, prints, re-audits the PDF's own text)
   ├──────────► PDF (reportlab)
   │ to_application (re-audits, maps ledger -> StartupCreditApplication)
   ▼
handoff: the existing agent graph -> human checkpoint
         + evidence engine (open items, ranked requests) + founder ledger with material claims marked
```

## 1. Files

### Package `src/uwcopilot/studio/package`

| File | What it does | Why it is built this way |
|---|---|---|
| `__init__.py` | The public API: `FounderProject`, `build_package`, `render_pdf`, `to_application`, `handoff`. | The web layer and the scripts import from one place. |
| `project.py` | `FounderProject`: the S2-S5 results of one plan plus what a lender application needs that the simulator does not (business name, legal entity, owners, documents in the file, other debt, collateral). | One typed input. Owners must agree with the plan facts (checked), so the package cannot contradict the simulation. |
| `golden.py` | The 36 golden SYNTHETIC projects: S4's golden scenarios plus seeded owners, entities and documents (`golden_documents`: none, matching statements, or contradicting statements). | Seeded and deterministic, so tests and the evaluation see the same founders every run. |
| `build.py` | `build_package`: builds the founder ledger (S4's `build_founder_ledger`) with the package's own claims, renders the locked narrative, builds the sections, the map layer and the audit. `inputs_sha256` hashes the project, the ledger and the narrative draft; the package id is its prefix. | Every number lives in ONE ledger. The id changes whenever anything printed could change (review fix 4). |
| `claims_finance.py`, `claims_sites.py` | Claims the package adds (ids `PKG.*`): names and ownership shares (text claims, printed as data), the amortization schedule, simulation settings, other candidate sites (simulated with the same seed and paths), competition details, map rings. | The package never prints a number that is not a claim, so these figures become claims first. |
| `assumptions.py` | The plan's material assumptions as the program lists them (NSP-4.3), with tornado ranks for the top drivers (NSP-6.1). | One row per simulator input, read from the ledger with the claim's own status. |
| `blocks.py` | Small constructors (`claim(...)` cell, `slot(...)` draft) and `BuildContext`. | Section builders only arrange claim ids and words; they never format a number. |
| `sections_overview.py` | 1 Executive summary (key facts, narrative, every program check), 2 Business, owners and plan. | Matches the SBA "Executive summary", "Company description", "Organization and management", "Service or product line". |
| `sections_market.py` | 3 Market and location analysis (map, SiteIQ factors, competition, demand range, metro inputs), 4 Site comparison. | A metro without data here runs in limited mode and the section says so (Studio rule G). |
| `sections_finance.py` | 5 Funding request, 6 Financial projections (year 1 by month, years 1-3, break-even, top drivers), 7 Loan schedule and DSCR, 8 Stress tests. | P10 / P50 / P90 throughout. Break-even P10 is labeled the early case and P90 the late case (review fix 2). |
| `sections_review.py` | 9 Risks and mitigations, 10 Path-to-approval history. | A mitigation is always a plan choice or a document to provide, never a changed fact. |
| `sections_appendix.py` | 11 Evidence appendix: every claim (value, range, status, source), every check, the model runs, the status legend. | Every value printed elsewhere links to its row `[k]`. |
| `sba.py` | The package sections and the SBA section names each covers (`SECTION_SPECS`); `sba_citation()`. | The SBA names are read from the fetched page (`config/studio/sba_business_plan_sections.json`), never typed. See [sources.md](sources.md). |
| `cover.py` | The cover and contents words, shared by the PDF and the audit. | The audit reads the cover too. |
| `text.py` | Every printed value, status, range, source and draft, exactly as the PDF prints it. | The PDF and the audit read the same words, so the audit checks what the reader sees. |
| `models.py` | The `Package` schema (`founder-package-v1`): blocks (`Paragraph`, `Table`, `KeyFacts`, `Bullets`, `Figure`) that point at claim ids. | A package is plain JSON data; a block never holds a number. |
| `audit.py` | `audit_package` and `audit_pdf`: the S4 round-trip audit on the package text and on the text pypdf extracts from the PDF. Founder text claims are blanked only where printed; layout labels ("Page 3/14", P10, "Year 2", "Month 7", "Loan year 9", "rank 3", "Change-set 0") are bounded patterns. | An untraced number is a defect. Labels can never hide a figure: no decimal, percent sign or unit may follow a label number, and each has a maximum (review fix 1). |
| `pdf.py`, `pdf_styles.py`, `pdf_figures.py` | `render_pdf` with reportlab (DejaVu Sans from matplotlib's font folder), figures drawn with matplotlib (map, cash band, tornado). Every page: product name, disclaimer, "Page x/y", the SYNTHETIC tag. | No browser, no network, no system font. `render_pdf` re-runs the audit before printing and audits the finished PDF (review fix 1). |
| `mapdata.py` | The map layer: a GeoJSON FeatureCollection of tract centroids, rings and the site. | No tile provider, so the PDF needs no network. |
| `runs.py` | The model runs behind MODEL_ESTIMATE claims ("run A", "run B" ...), listed once. | Hundreds of claims share one run identity. |
| `application.py` | `to_application`: the package -> `StartupCreditApplication` (`startup-application-v1`). Re-runs the audit; refuses missing fields with `ApplicationNotReady`; a CONFLICTING equity goes in at its lowest figure. | Every field comes from the ledger or the applicant facts; nothing is typed. Evidence flags count only with their document in the file. |
| `handoff.py` | `handoff`: runs the existing agent to the human checkpoint, the evidence engine on the same file, and builds the lender review request: the founder ledger with material claims marked, open items, ranked requests and `uncovered_material`. | The agent is reused unchanged; handoff never resumes it, so only a person finishes a run. |
| `evaluation.py` | The measurements behind `results/studio_handoff_metrics.json`. | One function per case type; rates with bootstrap intervals. |

### Scripts, results, tests, Make targets

| Item | What it is |
|---|---|
| `scripts/render_studio_package.py` / `make studio-package` | Builds, audits and prints one golden package (`PACKAGE_METRO=`, `PACKAGE_TYPE=`) into `data/studio_packages/` (gitignored). Exits 1 if any number in the PDF is untraced. |
| `scripts/eval_studio_handoff.py` / `make studio-handoff` | The 35 startup applications and the 36 golden packages end to end -> `results/studio_handoff_metrics.json`. `make studio-handoff-ground-truth` uses designed screening (no sanctions lists). |
| `scripts/studio_fetch_sba_plan_guide.py` | Fetches the SBA page once and writes the section names with the retrieval record. |
| `tests/studio/test_package_*.py` / `make studio-package-test` | Build, audit, PDF, application, handoff and script tests (offline). `make studio-package-golden` adds the slow 36-scenario run. |

## 2. How a package is made

1. **One ledger.** `build_package` builds a first founder ledger to read the claims the package needs, then the
   final ledger with the package's own `PKG.*` claims. Every printed number is a claim in that ledger.
2. **Sections are blocks of claim ids.** A table cell is `claim("CALC.projected_dscr")`; a paragraph is a slot
   draft (`{{claim:ID}}`) filled by the S4 provenance renderer. No section builder formats a number.
3. **Audit before and after printing.** `render_pdf` re-runs `audit_package` on the package as it is now (not the
   stored audit), prints with reportlab, then extracts the PDF's own text with pypdf and audits that too. Either
   audit finding an untraced number raises `PackageNotAudited`, so no unaudited PDF is ever returned.
4. **The application.** `to_application` maps the ledger and applicant facts onto `StartupCreditApplication`, the
   schema the agent and `eval/startup_applications` use. Codes stay codes; founder text goes only into name fields.
5. **The handoff.** `handoff` runs the agent graph (injection screen, sanctions screening, the startup program
   policy, the template memo, the claim checker). The graph always interrupts at `human_review`.

## 3. Design choices

- **reportlab, not an HTML template.** The S6 prompt said "PDF from an HTML template". An HTML-to-PDF route needs a
  browser engine or a system library (WeasyPrint needs Pango/Cairo; Chromium needs a headless browser) in CI and in
  the Space image. reportlab is pure Python, and its output is byte-for-byte deterministic, which the tests check
  (`test_rendering_is_deterministic`). The cost: the layout is code, not a template a designer can edit.
- **36 golden scenarios, not 30.** The prompt asked for 6 business types x 5 metros. The golden set is 6 types x
  (the SYNTHETIC metro + 5 real metros) = 36, because the SYNTHETIC metro is the only one with full data in a
  clean checkout (section 7).
- **The audit is re-run, never trusted.** A package is JSON and can be loaded or edited after it was built. The
  stored `package.audit` is a record, not a permission.
- **Founder words are data where they are printed, and only there.** A business called "Route 9 Wash" prints a 9
  that traces to no number. The audit blanks that text claim's value where it appears, matched word by word so a
  line break inside it still matches. The same 9 anywhere else must trace like any other number.
- **Conflicting equity is carried at its lowest figure.** The agent cannot tell "conflicting" from "missing", so
  the application carries the lower of the founder's figure and the bank statement's, and the program's equity check
  can only pass on a figure no document disputes.
- **Material follows calculations.** A claim is material when a check reads it, when it is a top-three driver
  input, or when a material claim is calculated from it. The DSCR is checked, so the note rate under the annual debt
  service is material.
- **What no request covers is shown.** The evidence catalog has no document for some material assumptions (the
  note rate before a term sheet, for example). `uncovered_material` lists them so the reviewer judges them, instead
  of the package implying they are covered.

## 4. Review fixes (S6 review, 2026-10)

A review of the S6 branch found one blocker and thirteen should-fix items. What changed:

| # | Finding | Fix |
|---|---|---|
| 1 | `walkthrough_s6.md` missing | This file. |
| 2 | `render_pdf` trusted the stored audit | `render_pdf` and `to_application` re-run the audit; `render_pdf` also audits its PDF. Tests feed a package whose stored audit is clean but whose text holds "change 37%", "$777,777" or "Rank 93": each is refused. |
| 3 | Layout masks could hide a real figure ("change 37%") | Label patterns are bounded and never followed by a decimal, "%" or a unit word. |
| 4 | Numbers in founder text were whitelisted everywhere | Scoped to where the text is printed (tests: the digits of a name elsewhere are flagged). |
| 5 | Numbers in square brackets are never audited | Bracketed text is provenance written by code from claim fields. The one constant typed into such a string (the 20% revenue stress) is now written from the policy constant, with a test. The rest stays unaudited by design (section 7). |
| 7 | CONFLICTING equity reached the application as the founder typed it | Lowest figure (test). |
| 8 | Not reproducible: the SYNTHETIC city stamped today's date; the id hashed only the project | Fixed stamp, labeled "generated from its seed on every run (no retrieval date)"; the id hashes the ledger and the narrative draft (test: different city data, different id). |
| 10 | "Material" too shallow; UNKNOWN never tested | Transitive material set; `uncovered_material`; a test with an UNKNOWN material claim. |
| 11 | Break-even P10 / P90 labels read backwards | Early case / late case, with a table note. |
| 13 | Make targets missing | `studio-package`, `studio-package-test`, `studio-package-golden`, `studio-handoff`, `studio-handoff-ground-truth`. |
| 14 | `docs/studio/sources.md` missing | Written: the SBA record, the SYNTHETIC program, metro data, documents, sanctions lists. |
| 6, 9, 12 | Real metros never exercised; S4 gaps; reportlab undocumented | Documented in sections 3 and 7; not fixed in S6. |

## 5. How the evaluation works (`scripts/eval_studio_handoff.py`)

Two SYNTHETIC sets:

1. **35 startup applications** (`eval/startup_applications`, designed labels): the agent graph runs on each to the
   human checkpoint. Measured: checkpoint reached, decisions recorded without a human (must be 0), agreement of the
   reference recommendation and the exact rule set with the designed labels, memo checks, unsupported numbers.
2. **36 golden founder packages**: package built, PDF rendered and audited (magic bytes, pages, labels on every
   page, untraced numbers in the package text and the PDF text), then `handoff()`. These have no designed label,
   so label agreement is "not applicable".

Rates carry percentile bootstrap 95% intervals over cases (1,000 resamples, seed 42) and a Wilson 95% interval.
The memo writer is the deterministic template (FAKE LLM - test mode): no language model is called and nothing is
spent.

## 6. Results

All numbers below are copied from `results/studio_handoff_metrics.json`, written by `make studio-handoff` at commit
`cf60df5` with a clean tree (`git_dirty: false`), full OFAC SDN + OpenSanctions screening, 1,000 simulated paths
per plan, golden seed 4404. Runtime 349.1 s. Intervals: percentile bootstrap over cases (1,000 resamples, seed 42);
when every case passes, the bootstrap interval is [1.0, 1.0], so the Wilson 95% interval is given too.

### 6.1 The 35 SYNTHETIC startup applications (designed labels)

| Measure | Value |
|---|---|
| Reached the human checkpoint | 35/35 (Wilson 95% [0.9011, 1.0]) |
| Decisions recorded without a human | 0 of 35 (Wilson 95% [0.0, 0.0989]) |
| Reference recommendation agrees with the designed label | 35/35 (Wilson 95% [0.9011, 1.0]) |
| Exact rule set agrees with the designed label | 35/35 (Wilson 95% [0.9011, 1.0]) |
| Memo check passed / unsupported numbers in memos | 35/35 / 0 |
| Reference recommendations | refer 16, request_more_information 10, decline_for_review 7, approve 2 |
| Median latency per application | 120.85 ms |

The labels are hand-written, one scenario per program rule, by the same project that wrote the rules
(`agent/synthetic_startup.py`). Full agreement shows that the graph applies the coded SYNTHETIC program as
designed; it says nothing about agreement with a real lender.

### 6.2 The 36 golden founder packages

| Measure | Value |
|---|---|
| Metro modes | SYNTHETIC 6, limited 30 |
| PDF starts with `%PDF` / every page labeled | 36/36 / 36/36 (Wilson 95% [0.9036, 1.0]) |
| Pages per PDF | min 27, median 30, max 34 |
| Numbers read back by the audit (package text, all 36) | 30,681 |
| Untraced numbers: package text / PDF text | 0 / 0 (audit empty in 36/36 for both) |
| Claims per package | min 405, median 438.5, max 469 |
| Reached the human checkpoint | 36/36 (Wilson 95% [0.9036, 1.0]) |
| Decisions recorded without a human | 0 of 36 (Wilson 95% [0.0, 0.0964]) |
| Memo check passed / unsupported numbers in memos | 36/36 / 0 |
| Evidence engine's recommendation equals the agent's | 36/36 (Wilson 95% [0.9036, 1.0]) |
| Reference recommendations | request_more_information 19, decline_for_review 17 |
| Evidence requests per package | min 4, median 5, max 7 |
| Top-ranked request | equity_bank_statements 24, missing_document 5, build_out_quotes 4, driver_comparable_data 3 |
| Seconds per scenario (build, PDF, handoff) | min 5.896, median 6.81, max 8.432 |

Across all 71 cases: 71/71 reached the human checkpoint (Wilson 95% [0.9487, 1.0]) and 0 decisions were recorded
without a human (Wilson 95% [0.0, 0.0513]).

How to read this:
- **"0 untraced" is partly by construction.** Every printed number is a claim by design, so the audit is a guard
  against regressions (a section builder or a narrative printing a figure that is not a claim), not a discovery.
  The review fixes made it stricter: labels are bounded and founder text is scoped, and the totals stayed at 0.
- **No golden package got "approve" or "refer".** The golden plans were made to exercise the ledger, not to be
  good loans: most have no bank statements or statements that contradict the founder's equity (bank statements are the
  top request in 24 of 36), or fall short of a program rule.
- **The 30 real-metro packages ran in limited mode** (no data snapshot), so their market sections show UNKNOWN or
  ASSUMPTION factors.
- The evaluation does not count `uncovered_material` per package; that field was added in the review and is
  covered by tests only.

## 7. Limits and open issues

- **Real metros ran in limited mode only.** No real-metro data snapshot was present in these runs, so all 30
  real-metro golden packages were built with UNKNOWN or ASSUMPTION factors and no map. The market section has not
  been exercised on real data. `make studio-data METRO=<id>` builds a snapshot; that run has not been done here.
- **The S4 gaps remain.** Few claims carry a metro or a snapshot date, model-estimate claims carry no structured
  range, and the run identity records no data-snapshot fingerprint. The package id now covers the data through the
  ledger hash, but the gaps themselves are S4 work.
- **Bracketed provenance text is not audited.** Citations, claim ids, hashes, seeds and formula text in square
  brackets are skipped by the number extractor. That text is written by code from claim fields and constants, not
  by a model, so the risk is a typed constant drifting; the one found was fixed. A per-claim provenance audit is
  not built.
- **Synthetic everything.** The program is the SYNTHETIC Northwind program, the documents are generated, and no
  lender has seen a package. Nothing here says a real lender would accept the format.
- **Review nits not done:** the sources table rounds to $1; the monthly payment row could carry its formula; some
  status words could be plainer; top drivers could show units; the map's GeoJSON carries no provenance block;
  re-running a handoff on the same package reuses its thread id; `founder_claims_of` parses rank ids with `rsplit`.

## 8. Likely interview questions

1. **How do you know every number in the PDF is real?**
   Every printed number is a ledger claim. After printing, pypdf extracts the PDF's text and the S4 round-trip audit
   looks each number up in the ledger in the claim's own display format. Any untraced number stops the render.

2. **Why re-run the audit when the package already stores one?**
   The package is JSON. Someone can load it, edit a paragraph and print it. A stored "clean" flag would wave that
   through; re-running checks the package as it is now. A test does exactly that edit and is refused.

3. **A business is called "Route 9 Wash". How does the audit handle the 9?**
   The name is a text claim. The audit blanks it where it is printed, word by word so a line break still matches.
   The 9 is data there and only there; a 9 elsewhere must trace like any number.

4. **What stops the label masks from hiding a figure?**
   Each label is a bounded pattern (Year 1-5, Month 1-60, Loan year 1-30 ...) that may not be followed by a
   decimal, a percent sign or a unit word. "change 37%" or "Year 47" stay figures. Tests check both directions.

5. **What happens when the founder's equity disagrees with the bank statement?**
   Both figures are printed as CONFLICTING. The application carries the lower one until a person resolves it, the
   verified flag stays false, and the evidence engine asks for the statements.

6. **Can the agent approve a founder's loan?**
   No. The graph always interrupts at the human checkpoint and `handoff` never resumes it. The evaluation counts
   decisions recorded without a human; the requirement is zero.

7. **What does "material" mean, and why did it change?**
   A claim a decision rests on: one a program check reads, a top-three driver input, or anything a material claim
   is calculated from. Before the fix, the note rate under the checked DSCR was not material, so a lender could miss
   that the DSCR rests on an assumed rate.

8. **Why reportlab instead of HTML to PDF?**
   It is pure Python with deterministic output: no browser or system library in CI or the Space image, and a test
   can compare two renders byte for byte. The trade-off is that the layout lives in code.

9. **How is the package reproducible?**
   Fixed seeds, a fixed stamp on the SYNTHETIC city (it is generated, not retrieved, and says so), and an id that
   hashes the project, the ledger and the narrative draft. The same inputs give the same JSON and the same PDF bytes.

10. **What would you do before showing this to a real lender?**
    Build real-metro snapshots and re-run the evaluation on them, add a per-claim provenance audit, have a lender
    review the format, and put the privacy rules of `docs/studio/privacy.md` in front of any real founder data.
