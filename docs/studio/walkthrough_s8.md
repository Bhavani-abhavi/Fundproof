# Walkthrough: Studio S8, privacy, security and trust

Studio S8 prepares the web app (Founder Studio + Lender Desk) for real founders. It adds an access log for founders'
plans, expiring and revocable share links, a "Delete all my data" button, the screen a future upload route must call,
dependency and secret scanning in CI, in-app Privacy and Terms pages, and a STRIDE threat model.

This is a demonstration prototype. Nothing here is a compliance claim: whether any privacy or lending law applies
depends on who runs the app and how. The app still asks visitors to use made-up plans until the owners approve
[privacy.md](privacy.md). Written so both owners (Rahul Vajja and Bhavani Adula) can explain the work.

## 1. What S8 asked for, and what exists now

| S8 asked for | Where it is | State |
|---|---|---|
| `docs/studio/privacy.md` and in-app Privacy Policy + Terms of Use | [privacy.md](privacy.md), `frontend/app/privacy/page.tsx`, `frontend/app/terms/page.tsx` | Built. Both pages carry a "Draft - pending the owners' approval" banner; owner commitments such as "data is never sold" are left to the owners |
| Data minimization in the intake schema | v0.1 schemas (`web/schemas_studio.py`): no e-mail, no SSN, no documents; typed text refuses SSN and account-number shapes (`web/services/text_checks.py`) | Already in v0.1; S8 adds minimization of what a share link shows |
| Block uploads that look like tax returns or hold SSN / bank-account patterns | `web/services/upload_checks.py` | The screen is built and tested; **there is no upload route** in v0.1, so nothing calls it yet |
| Encryption at rest | `web/crypto.py`, `web/store.py` (Fernet) | Already in v0.1 |
| Signed, expiring share links | `web/services/share_links.py`, `web/routers/share.py`, `frontend/components/studio/project/share-links.tsx`, `frontend/app/share/page.tsx` | Built |
| Per-user access checks with tests | `tests/web/test_web_studio.py`, `tests/web/test_web_share.py` | v0.1 had them; S8 adds them for the share routes |
| Audit log of every access to a founder's package | `web/audit_log.py` (reusing `security/audit.py`) | Built for **plans**: the web app does not serve the S6 package yet, so the plan is what is accessed |
| Automatic deletion after the retention period | `web/store.py` `purge_expired` (30 days by default) | Already in v0.1 |
| "Delete my data" button with a test proving deletion | `web/routers/account.py`, `frontend/components/session/delete-my-data.tsx`, `tests/web/test_web_delete_my_data.py` | Built |
| Rate limits, request size limits, abuse protection | `web/security.py` | Already in v0.1; the anonymous share route uses the same per-IP limit |
| Prompt-injection screening on uploaded text | `upload_checks.py` flags it (data, never instructions) | Built in the screen; no upload route |
| Dependency and secret scanning in CI | `.github/workflows/ci.yml` jobs `audit-deps` and `scan-secrets`, `Makefile`, `.gitleaks.toml`, `tools/pip-audit/` | Built; both jobs passed on GitHub Actions |
| Threat model (STRIDE) | [threat_model.md](threat_model.md) | Written |

## 2. Files

### Backend (`src/uwcopilot/web`)

| File | What it does | Why it is built this way |
|---|---|---|
| `audit_log.py` | `PlanAccessLog`: one event per access to a plan (list, read, create, update, delete, each compute route, share create / revoke / view, delete-all), written BEFORE the access. If the event cannot be written, the request is refused with 503. | An access that is not logged must not happen. It reuses the repo's hash-chained `AuditLog` (no second chain), so `scripts/verify_audit.py --log <file>` checks it. Actors are owner digests or `share_link:<link id>`, never an account id, username or IP. |
| `../security/audit.py` (changed) | `append` now reads only the end of the log (backwards in 4 KiB steps) and takes a failed write back, so a full disk cannot leave a torn line; a torn tail from a crash is cut only when the head file vouches for the last complete event. | Review findings SL-3 / S8-AUD-2 (cost grew with the log, so anonymous views could slow every request) and S8-AUD-1 (a torn line blocked every later request). |
| `services/share_links.py` | `ShareLinks`: create, resolve, revoke. The token is a Fernet token whose key is derived from the session secret with HMAC-SHA256 and a fixed label; it carries owner, plan, link id and expiry, encrypted. A record in the owner's own store namespace `shares` must also exist. | Encrypted so the owner id never shows in a link; derived key so the token key opens no stored object; the record makes revocation, plan deletion and "delete all" kill the link. One canonical spelling per token. |
| `routers/share.py` | `POST` / `DELETE /api/studio/projects/{id}/share` (owner only) and `GET /api/share/{token}` (anonymous). One 404 body for every bad link; `Cache-Control: no-store` and `Referrer-Policy: no-referrer` on 200, 404 and 503. | No oracle: a guesser cannot tell expired from revoked from forged. |
| `schemas_privacy.py` | `SharedPlanOut` and the response models of the new routes. | The shared view is built field by field: plan name, business type, city, money, opening date, site names and site locations rounded to 2 decimals (about 1 km). Never the idea text, street addresses or account data. |
| `routers/account.py` | `DELETE /api/me/data`: deletes every stored object of the account (plans, DEMO records, share records), drops its in-memory results, writes an access event, signs out. `pending: true` when the backup dataset copy is not deleted yet. | One request, one test that proves it. The access log keeps its events (an audit trail outlives the data it describes); the privacy page says so. |
| `services/deletions.py` | Remembers, in memory, when each account deleted its data, so a slow plan creation that started earlier saves nothing. | Review finding S8-DEL-3 (a create waiting on the geocoder could save a plan after the deletion answered "all deleted"). |
| `services/upload_checks.py` | `screen_upload(filename, content_type, data, text)`: refuses wrong or mismatched file types (magic bytes), oversize or empty files, unread or scanned PDFs without text, SSN shapes, labeled account numbers, ABA routing numbers (prefix + check digit), and tax-return markers; flags prompt-injection text without refusing. The message never repeats the matched text. | Studio rule B: no SSNs, no tax returns, no bank account numbers in v1. Its "Known limits" section lists what it does not catch. |
| `hf_store.py` (changed) | Files of a commit that may have landed are treated as remote, and owner deletions name files instead of folders. | Review findings S8-DEL-1 (delete-all said "done" while the plan stayed on the Hub) and S8-DEL-2 (a pending folder delete removed a later plan). |
| `settings.py`, `app.py`, `access_log.py`, `services/projects.py`, `routers/studio.py` (changed) | `UWC_WEB_AUDIT_LOG` (default next to the store folder, never inside it, never at the filesystem root; checked at startup), the request log masks `/api/share/*`, the plan routes write access events, and deleting a plan revokes its links first. | |

### Frontend (`frontend/`)

| File | What it does |
|---|---|
| `app/privacy/page.tsx`, `app/terms/page.tsx`, `components/legal/*` | The plain-language Privacy Policy and Terms of Use, with the draft banner. The Terms show the exact disclaimer (`DISCLAIMER` in `frontend/lib/config.ts`): "Educational planning tool. Not financial, legal or lending advice. Lenders make their own decisions." |
| `components/session/delete-my-data.tsx`, `app/about/page.tsx` | "Delete all my data" with a confirmation step, the result (or "pending" on the storage service), then signed out. |
| `components/studio/project/share-links.tsx`, `lib/share-api.ts` | Create a link (1, 3, 7, 14 or 30 days), copy it, revoke all links; the pending-revoke message never says a link "no longer works" when its backup deletion is pending. |
| `app/share/page.tsx`, `components/share/shared-plan-view.tsx` | The read-only page for `/share?t=<token>` (a query parameter, because the app is a static export). Every value is rendered as text; the page sets `referrer=no-referrer`. |
| `mock-api/server.mjs` | Mock answers for the four new routes. |

### CI and docs

| Item | What it is |
|---|---|
| `make audit-deps` / CI job `audit-deps` | pip-audit, installed from a hashed lock (`tools/pip-audit/requirements.txt`) into a throwaway virtualenv, checks the locked dependencies (`uv export --locked`). Fails on any known vulnerability; accepted ones would be listed in `PIP_AUDIT_IGNORE` with a reason (none today). |
| `make scan-secrets` / CI job `scan-secrets` | gitleaks 8.30.1 (release tarball, SHA-256 checked) over every commit, merge commits included. `.gitleaks.toml` allow-lists 24 false positives of the first scan, each by rule, path and the exact shape of the value (digests of committed eval files and similar), so a real key in the same file is still reported. |
| [threat_model.md](threat_model.md) | STRIDE table: threat, control in the code, test that shows it, residual risk; open items for the owners. |
| [privacy.md](privacy.md) | Updated for everything S8 built, including the access log and what stays after deletion. |

## 3. How the pieces work

**A share link.** The founder clicks "Create share link" on a plan. The server makes a random 20-letter link id,
stores `{link_id, project_id, created_at, expires_at}` in the founder's own `shares` namespace (encrypted, kept until
the link expires), and returns a Fernet token holding `{owner, project_id, link_id, expires_at}`. When someone opens
`/share?t=<token>`, the server checks, in order: the token is in its one canonical spelling, it decrypts with the
derived key, it has not expired, its record still exists and matches, and the plan still exists. Any failure gives
the same 404. A valid view writes an access event with the actor `share_link:<link id>` and returns the minimal view.

**Delete all my data.** One request deletes the account's whole store folder (every namespace), forgets its
in-memory results, notes the deletion time (so a slow create that started earlier saves nothing), writes an access
event and clears the session cookie. With the dataset store the deletion is also sent to the Hub and its history is
squashed; when the Hub cannot be reached, the answer says `pending` and the page says the deletion is still running.

**The access log.** Every route that reads or changes a plan calls the log first. The event names the action, the
plan id and the actor; never the plan's content, names, addresses, money or an IP. In a Space the default log is on
the container disk, which does not survive a restart: the app logs an error at startup when that is certain (default
store folder) and a warning when the operator chose the store folder.

## 4. Design choices

- **Fail closed on the access log.** An unlogged access is worse than a refused one, so a log that cannot be written
  turns the request into a 503 (with no plan data). The cost: a broken disk stops plan access until it is fixed.
- **A derived key for share tokens.** Using the store key would let a token open stored data or the reverse; a
  separate label-derived key keeps the two apart. Rotating the session secret ends every link (and every session).
- **Rounded site locations in shared views.** A typed street address is geocoded to an exact point; sharing that
  point would give the address back. Two decimals (about 1 km) keep the map useful for a lender and hide the street.
- **The upload screen without an upload route.** The rules exist and are tested now, so the first upload route can
  call them; adding the route is a product decision (and needs its own body-size limit and storage).
- **Pinned and hashed scanners.** The secret scanner is a checksum-verified release; pip-audit is installed from a
  hashed lock, so a compromised dependency of the scanner itself cannot change the result unnoticed.

## 5. How it was built and reviewed

1. Four agents built the backend, the upload screen, CI scanning and the frontend in separate git worktrees against
   one written API contract; the branches were merged and the full test suite run.
2. An adversarial review ran four lenses (share links; delete-all and the access log; uploads; CI and every
   user-facing claim). Each lens's findings went to a separate skeptic who tried to reproduce or refute them: 32
   findings, 29 confirmed, 3 refuted, none critical. The medium ones: a revoke that could silently come back after a
   restart with the dataset store; the access log's cost growing with its size (an anonymous link holder could slow
   every request); a torn log line after a full disk blocking every plan request; delete-all reporting success while a
   plan stayed on the Hub; a quadratic regex (a 128 KB upload took 107 s); a scanned PDF passing unscreened; and the
   privacy texts not mentioning the access log.
3. Four fixers fixed them in separate worktrees, and four verifiers re-ran each original reproduction against the
   fix: all 29 resolved. The verifiers also found what the fixes broke: three upload-pattern widenings that refused
   ordinary text (payroll-tax rows, "account." at the end of a sentence, negative amounts), revoke leaving expired
   link records behind, a startup error that could be a false alarm, and one stale sentence in the plan wizard. Those
   were fixed afterwards, with the verifiers' examples as regression tests.

## 6. Checks

Run on 2026-10-20 at commit `7bf01f8` (the code of this walkthrough; only the docs changed after it):

| Check | Result |
|---|---|
| `uv run ruff check .` and `uv run ruff format --check .` | clean |
| Full offline test suite (`uv run pytest -o addopts='' -m 'not integration'`) | 4,485 passed, 36 skipped, 10 deselected (integration) |
| New S8 test files | `tests/web/test_web_access_audit.py`, `test_web_share.py`, `test_web_delete_my_data.py`, `test_upload_checks.py`, `tests/security/test_scan_config.py`, and new cases in `tests/security/test_audit.py`, `tests/web/test_web_hf_store.py`, `test_web_settings.py` |
| `make scan-secrets` (local) | 145 commits scanned, "no leaks found", exit 0 |
| `make audit-deps` (local) | 205 packages resolved, "No known vulnerabilities found", exit 0 |
| Frontend `npm run lint`, `npm run typecheck` | exit 0 (the static build passed in the fix round's frontend worktree) |
| GitHub Actions on the first merged S8 push (`d5dfedb`) | `lint-and-test`, `audit-deps` and `scan-secrets` all succeeded |

These are checks, not metrics: S8 adds no measured rate, so there is nothing to put in `results/*.json` and no
confidence interval to report. A vulnerability scan that finds nothing today says only that no advisory was known
for these versions on that date.

## 7. Limits and open items

- **No upload route.** The screen is ready; the route, its storage and its body-size limit are not built. The
  screen's own limits (OCR errors, unlabeled numbers, non-US forms, nine-digit ids that pass the routing check about
  37 times in 1,000) are listed in its module docstring.
- **The web app does not serve the S6 package.** Share links and the access log cover plans, not packages.
- **The access log's home.** In the Space it is lost on restart unless `UWC_WEB_AUDIT_LOG` points at persistent
  storage; its head anchor sits next to it; it has no retention, rotation or automatic deletion yet; and it keeps an
  owner digest that anyone who knows the account id can link back.
- **Owner decisions** (also in [threat_model.md](threat_model.md), section 5): the commitments the policy marks as
  pending, the access log's retention, the demo reviewer role on SYNTHETIC lender files (docs/security.md section 9).
- **Not scanned:** the frontend's npm dependencies; no scheduled CI run between pushes.
- **Not tested in a browser:** the new pages passed lint, typecheck and the static build; no browser was driven.

## 8. Likely interview questions

1. **Why write the audit event before the access instead of after?**
   If the event is written after, a crash or a full disk between the access and the write leaves an access with no
   record. Writing first, and refusing when the write fails, means every access that happened has an event. An event
   can exist for a request that then failed validation; the log says "attempted", not "succeeded".

2. **How does a share link stay secret and revocable at the same time?**
   The token is encrypted (the owner id inside cannot be read) and authenticated (it cannot be changed). Revocation
   works through a server-side record: the token only works while its record exists, so deleting the record kills the
   link even though the token still decrypts.

3. **Why does every bad link get the same 404?**
   Different answers for "expired", "revoked" and "forged" would tell a guesser which tokens were once real. One body
   and the same headers give nothing away; the review also measured the timing difference and found none usable.

4. **What does "Delete all my data" not delete, and why?**
   The access log's events about the account (owner digest, plan ids, actions, times; never content). An audit trail
   that disappears with the data would let anyone erase the record of what they did. The privacy page says so.

5. **What was the regex problem, and how was it fixed?**
   Two adjacent `\s*` around an optional character let the engine try every split of a long run of spaces, so time grew
   with the square of the run. Making the runs possessive (`\s*+`, never given back) keeps the same matches in linear
   time; the tests bound the work on adversarial inputs by counting, not by wall-clock time.

6. **Why round shared coordinates?**
   A street address the founder typed is geocoded to an exact point. Sharing the point would give the address back to
   anyone with the link, while the page promised addresses are never shared. About 1 km keeps the neighbourhood
   visible and hides the street.

7. **How do you know the secret scan does not hide a real secret?**
   Each allow-list entry names one rule, the files and the exact shape of the flagged value (for example a 64-hex
   digest under a `key_facts` key in a results file). A different value, another rule, or another file is still
   reported. The review narrowed two entries that were broader than that.

8. **Why is the upload screen built before any upload route?**
   So the rules can be designed, tested and reviewed on their own, and the first upload route has a tested gate to
   call. It also fails closed: a PDF whose text was never read is refused, because it cannot be shown to be clean.

9. **What would you do before accepting real founder data?**
   Have the owners approve privacy.md and the commitments it leaves open, give the access log persistent storage and
   an off-site head anchor, decide its retention, run the pages in a real browser (including the Space iframe), and
   get an external security review. None of that is done.

10. **How was this reviewed, and what did the review catch that the builders missed?**
    An adversarial pass with a separate skeptic per lens: 29 confirmed findings, then fixes, then verifiers that
    re-ran the original reproductions. The verifiers also caught what the fixes broke (upload patterns that refused
    payroll rows and ordinary sentences), which is why each fix was checked by someone other than its author.
