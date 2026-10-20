# Sources cited by the founder package (Studio S6)

This page lists every outside source the founder package (`src/uwcopilot/studio/package`) cites, and the
SYNTHETIC documents it uses in place of real ones. Every package page says: "Educational planning tool. Not
financial, legal or lending advice. Lenders make their own decisions."

## 1. Section names: SBA "Write your business plan"

| Field | Value |
|---|---|
| Publisher | U.S. Small Business Administration |
| Page | "Plan your business", section "Write your business plan" |
| URL requested | https://www.sba.gov/business-guide/plan-your-business/write-your-business-plan |
| URL that answered (after redirect) | https://www.sba.gov/counseling/plan-your-business/ |
| Retrieved | 2026-10-14 (UTC), HTTP 200 |
| Page SHA-256 | `ef776b3118903e2d5a27f61d761b70926676697413d7f2d946275cf6d2d79657` |
| What is used | The eight section names under "Common traditional business plan sections": Executive summary, Company description, Market analysis, Organization and management, Service or product line, Marketing and sales, Funding request, Financial projections |
| Where it is stored | `config/studio/sba_business_plan_sections.json`, written by `scripts/studio_fetch_sba_plan_guide.py`. The page body is kept under `data/studio/reference/` (gitignored) and its SHA-256 is in `data/studio/checksums.jsonl` |

How it is used: `package/sba.py` maps each package section to the SBA section names it covers. A package section
the SBA list does not name (for example the evidence appendix) says so, and the cover lists any SBA section the
package does not cover. The citation (with the retrieval date and the page hash) is printed on the cover and in the evidence appendix.

What it is not: the SBA page is guidance on how to write a business plan. Following its section names says
nothing about whether a lender, or the SBA, would approve a loan. The package never says it would.

If the page cannot be fetched, the script writes nothing and the package cites nothing (it does not fall back to
section names typed from memory).

## 2. The lending program: SYNTHETIC

The program rules (minimum DSCR, equity injection, credit score, experience, loan size and term, revenue stress,
reserve months) are those of the **SYNTHETIC Northwind startup term loan program**
(`corpus/programs/northwind_startup_program.md`, version 2026.1), coded in `src/uwcopilot/agent/policy_startup.py`.
Northwind is not a real lender. The rules are a plausible demonstration, not any institution's policy; their
applicability to a real loan depends on the institution and its risk profile.

## 3. Metro data

The market section reads the metro's data fabric (`src/uwcopilot/studio/data`). Each figure carries its own
source line (publisher, dataset, vintage, URL, retrieval date, response SHA-256), printed in the evidence appendix.
The public sources and their licenses are listed in [data_sources.md](data_sources.md).

- **SYNTHETIC metro.** Every value comes from the city generator (`studio/data/synthetic_city.py`) and is labeled
  SYNTHETIC. The city is regenerated from its seed on every run, so its source line says "generated from its seed
  on every run (no retrieval date)" instead of printing a date.
- **Real metros without a data snapshot run in limited mode** (Studio rule G): factors with no data are UNKNOWN or
  ASSUMPTION, the package says so, and no map is drawn. In this repository's test and evaluation runs no real-metro
  snapshot was present, so every real-metro package was built in limited mode (see walkthrough_s6.md, section 7).

## 4. Founder documents

The founder documents in the tests and the evaluation (bank statements, quotes, resumes) are SYNTHETIC values
generated in `src/uwcopilot/studio/package/golden.py` (`golden_documents`). A founder figure a document confirms is VERIFIED; one a
document contradicts is CONFLICTING, and both figures are printed.

## 5. Sanctions lists (handoff only)

`make studio-handoff` screens the applicant and owners against the OFAC SDN list (US Treasury, public data) and the
OpenSanctions "default" dataset (CC BY-NC 4.0; data from OpenSanctions.org, used non-commercially with
attribution), downloaded by `make data`. Every name screened is SYNTHETIC. `make studio-handoff-ground-truth` uses
designed screening outcomes instead and reads no list.
