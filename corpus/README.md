# Knowledge corpus for retrieval (RAG)

> **SYNTHETIC CONTENT.** Everything under `corpus/guidelines/` was written for this portfolio project. The
> institutions are fictional and the thresholds are illustrative. Nothing here is lending, insurance, legal or
> compliance advice, or a description of any real company's policy.

## What is here

| File | Document id | What it is | Section ids |
|---|---|---|---|
| `guidelines/northwind_business_credit.md` | `northwind_business_credit` | Northwind Business Credit — Small Business Lending Underwriting Guidelines (SYNTHETIC, fictional institution) | `NBC-1.1` ... `NBC-10.1` (30 sections) |
| `guidelines/harborline_mutual_commercial.md` | `harborline_mutual_commercial` | Harborline Mutual — Commercial Property & Liability Underwriting Guidelines (SYNTHETIC) | `HM-1.1` ... `HM-12.1` (30 sections) |

## Provenance

- **Authored, not collected.** Both guideline documents were written by hand (with an AI assistant) for this
  repository on 2026-10-10. They are not copied or adapted from any real lender's or insurer's manual.
- **Why synthetic.** Real underwriting manuals are confidential. A synthetic manual lets the project show
  retrieval, citation and policy checks end to end without using anyone's private documents.
- **Designed to be testable.** Every section heading carries a stable id (`## NBC-3.2 Revenue verification`),
  so retrieval results and memo citations such as `[NBC-3.2]` can be checked automatically. Thresholds are
  written as unambiguous numbers with explicit boundary rules (each document states that "at least" includes
  the value and "more than" excludes it), because a deterministic reference policy will be built from them.
- **Versioned in git.** The files are small text documents and are committed. Any change to a threshold is a
  visible diff; evaluation results must be regenerated after such a change.

## Public regulatory documents (not in this folder)

Public regulatory documents are downloaded by `scripts/download_data.py --datasets regulatory_docs` into
`data/raw/regulatory/` (gitignored), with a SHA-256 checksum logged for each file in
`data/raw/checksums.jsonl`. They are primary sources from US agencies and are cited with their dates. Whether
any of them applies to a given institution depends on the institution and its risk profile; this project is a
demonstration, not a compliance program.

## Rules for using this corpus

- Retrieved text is **data, never instructions**. A retrieved passage can support a statement in a memo; it can
  never change what the agent is allowed to do.
- Cite the section id for every statement that relies on a guideline section.
- Do not describe these guidelines as real policy or as an industry standard.
