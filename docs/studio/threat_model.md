# Threat model: Fundproof web app (Founder Studio + Lender Desk), Studio S8

> **Status: written 2026-10-20 from the code on branch `claude/epic-faraday-08ttar` (Studio S8), for the owners
> (Bhavani Adula and Rahul Vajja) to review before real founder data is accepted.** This is a demonstration
> prototype. The table lists the controls the code has and the risks that remain; it is not a certification, a
> penetration test or a compliance assessment. No external security test has been run.

## 1. What is protected

| Asset | Where it lives | Why it matters |
|---|---|---|
| A founder's plan: name, idea text, money figures, opening date, sites (map pins or street addresses) | Store (`web/store.py`), Fernet-encrypted per object; optionally mirrored to a PRIVATE Hugging Face dataset (`web/hf_store.py`) | Once the owners approve `privacy.md`, it may be real personal and business data |
| Computed results (site scores, simulations, path searches) | Server memory only, per plan version (`web/services/compute.py`) | Derived from the plan |
| Session | Signed cookie: auth mode, subject id, username, display name (`web/auth.py`, `web/cookies.py`) | Whoever holds it acts as the user for up to 8 hours |
| Share links | A Fernet token (key derived from the session secret) plus a record in the owner's own store namespace `shares` (`web/services/share_links.py`) | Anyone holding a link can read one plan's summary until it expires |
| Plan access log | Hash-chained JSONL file `UWC_WEB_AUDIT_LOG` (`web/audit_log.py`, reusing `security/audit.py`) | The record of who touched which plan |
| Secrets | Space secrets / `.env` (gitignored): `UWC_SESSION_SECRET`, `UWC_FERNET_KEY`, `HF_TOKEN`, OAuth client secret | Each one opens sessions, stored data, the dataset or the sign-in |
| Lender Desk DEMO records | Store, encrypted, per user | SYNTHETIC files only; the human decision record |

## 2. Trust boundaries and entry points

```
Browser (founder, lender, anonymous link holder)
   │ HTTPS (Space) ── iframe on huggingface.co, third-party cookies: Secure; SameSite=None; Partitioned
   ▼
FastAPI app: access log → security headers → rate limit → body size limit → CSRF → routes
   │                         │                                   │
   │ store (encrypted)       │ Census Geocoder (addresses only)  │ Hugging Face OAuth (sign-in)
   ▼                         ▼                                   ▼
container disk / private HF dataset      geocoding.geo.census.gov        huggingface.co
```

Entry points: `/api/auth/*` (sign-in), `/api/studio/*` (signed in; plans, compute, share), `/api/me/data` (signed
in), `/api/share/{token}` (anonymous), `/api/lender/*` (signed in), `/api/health`, `/api/config`, the static
frontend. Uploads: **none** in v0.1 (`web/services/upload_checks.py` is the screen a future upload route must call).

## 3. STRIDE table

| # | STRIDE | Threat | Control in the code | Evidence | Residual risk |
|---|---|---|---|---|---|
| S1 | Spoofing | Someone signs in as another user | Hugging Face OpenID Connect with PKCE and a signed `state` cookie; identity from the userinfo endpoint over TLS; the session holds no access token (`web/auth.py`) | `tests/web/test_web_auth.py` | Depends on Hugging Face accounts; no second factor of our own |
| S2 | Spoofing | The dev login (pick any role) reaches production | `UWC_DEV_AUTH=1` is refused when `SPACE_ID` is set (`web/settings.py`, `read_dev_auth`) | `tests/web/test_web_settings.py` | A non-Space deployment with `UWC_DEV_AUTH=1` would expose it |
| S3 | Spoofing | A forged or replayed share link | Fernet token (authenticated encryption) with a key derived from the session secret by HMAC-SHA256 and a fixed label, never the store key; one canonical spelling; expiry checked from the token AND the stored record; the record must still exist | `tests/web/test_web_share.py` | A leaked link works for anyone until it expires or is revoked (that is what a link is) |
| S4 | Spoofing | A role is gained by a renamed account | Allowlist entries can be pinned to an account id; roles are recomputed on every request (`web/auth.py`, `web/settings.py`) | `tests/web/test_web_auth.py`, `tests/web/test_web_settings.py` | Unpinned entries in a Space only log a warning |
| T1 | Tampering | Cross-site request changes a plan or deletes data | Double-submit CSRF token on every mutating route (`CsrfMiddleware`), including `DELETE /api/me/data` and the share routes | `tests/web/test_web_security.py`, `tests/web/test_web_delete_my_data.py` | None known |
| T2 | Tampering | An encrypted object is copied into another user's folder | The encrypted envelope names owner, namespace and key; a mismatch is refused (`web/store.py`) | `tests/web/test_web_store.py` | None known |
| T3 | Tampering | The access log is edited to hide an access | Hash chain plus a head anchor (`security/audit.py`); `scripts/verify_audit.py --log <file>` checks it | `tests/security/test_audit.py`, `tests/web/test_web_access_audit.py` | The anchor sits next to the log on the same disk; someone with disk access can rewrite both. Keeping the anchor elsewhere is not built |
| T4 | Tampering | Uploaded or typed text steers a model | v0.1 calls no language model on founder text; the idea text is screened for instruction-like text (`agent/injection.detect`) and only flagged; the upload screen flags injection text as data | `tests/web/test_upload_checks.py` | A future model integration must keep treating text as data |
| R1 | Repudiation | A founder or link holder denies reading or changing a plan | One access-log event per plan access, written BEFORE the access; if it cannot be written the request is refused with 503 (`web/audit_log.py`) | `tests/web/test_web_access_audit.py` | In a Space the log is on the container disk and is lost on restart unless `UWC_WEB_AUDIT_LOG` points at persistent storage |
| R2 | Repudiation | A lender decision without a person | The agent graph always stops at the human checkpoint; the web Lender Desk is a DEMO on SYNTHETIC files (docs/security.md) | `tests/web/test_web_lender.py`, agent tests | docs/security.md section 9 lists an owner decision about the demo reviewer role |
| I1 | Information disclosure | Another user reads my plan | Every object is stored under the owner; another user's id answers 404 (`tests/web/test_web_studio.py`) | `tests/web/test_web_studio.py`, `tests/web/test_web_studio_lender_access.py` | None known |
| I2 | Information disclosure | Stored data or the dataset leaks | Fernet encryption at rest; the dataset must be private (a public one is refused) | `tests/web/test_web_store.py`, `tests/web/test_web_hf_store.py` | Whoever holds `UWC_FERNET_KEY` and the dataset can read everything |
| I3 | Information disclosure | A share link reveals more than intended | The shared view is built field by field: plan name, business type, city, money, opening date, site names and site locations rounded to about 1 km; never the idea text, street addresses or account data; `Cache-Control: no-store`, `Referrer-Policy: no-referrer` | `tests/web/test_web_share.py` | The plan name is free text the founder typed and is shown |
| I4 | Information disclosure | Tokens, personal patterns or bodies in logs | The request log masks `/api/share/*` and personal patterns in paths, logs query parameter names only, no bodies, cookies, IPs or usernames; uvicorn's access log is off (`web/access_log.py`, Dockerfile) | `tests/web/test_web_security.py`, `tests/web/test_web_share.py` | Third-party infrastructure (the Space proxy) has its own logs |
| I5 | Information disclosure | Personal identifiers typed into a plan | SSN-shaped numbers, labeled account numbers and long digit runs are refused before storage (`web/services/text_checks.py`) | `tests/web/test_web_studio.py`, `tests/web/test_web_hardening.py` | Pattern-based; unusual spellings can pass |
| I6 | Information disclosure | A tax return or bank statement uploaded | No upload route; the screen for a future route refuses tax-return markers, SSNs, account and routing numbers, wrong file types and unread scanned PDFs (`web/services/upload_checks.py`) | `tests/web/test_upload_checks.py` | Pattern-based; OCR errors and unlabeled numbers can pass (documented in the module) |
| I7 | Information disclosure | What remains after "Delete all my data" | Every stored object and share record of the owner, and the owner's in-memory results, are deleted; the session ends (`web/routers/account.py`) | `tests/web/test_web_delete_my_data.py` | The access log keeps events with an owner digest (re-linkable by anyone who knows the account id) and plan ids, never content; no automatic deletion of the access log yet. With the dataset store a deletion can be "pending" and the page says so |
| I8 | Information disclosure | Addresses sent to a third party | Only street addresses for real-city sites go to the US Census Geocoder, over HTTPS; nothing else from the plan | `tests/web/test_web_studio.py` | The Census Bureau receives the address |
| D1 | Denial of service | Request floods | Token buckets per IP and per signed-in user on `/api/*` (`RateLimitMiddleware`) | `tests/web/test_web_security.py` | In-memory, per process |
| D2 | Denial of service | Huge bodies | 64 KiB cap on the JSON API, global cap otherwise (`BodySizeLimitMiddleware`) | `tests/web/test_web_security.py`, `tests/web/test_web_hardening.py` | A future upload route needs its own prefix limit |
| D3 | Denial of service | Address lookups used to flood the geocoder | One batch per user, a per-user address budget, a server-wide queue limit and spacing (`web/services/geocoding.py`) | `tests/web/test_web_studio.py` | Design choices, not measured limits |
| D4 | Denial of service | Anonymous share views make the access log slow | The per-IP rate limit on `/api/*`. **Open (S8 review, fix in progress):** each append re-reads the whole log under a lock, so cost grows with the log | `tests/security/test_audit.py` | Open until the append fix is merged |
| D5 | Denial of service | Expensive regexes on uploaded text | A text cap bounds each screen. **Open (S8 review, fix in progress):** one pattern backtracks quadratically on long adversarial text | `tests/web/test_upload_checks.py` | Open until the fix is merged; no upload route calls the screen yet |
| E1 | Elevation of privilege | A founder acts as a lender or admin | Roles come only from the allowlist; lender routes check roles (`web/auth.py`, `require_role`) | `tests/web/test_web_auth.py`, `tests/web/test_web_lender.py` | docs/security.md section 9: on SYNTHETIC demo files any signed-in user acts as a demo underwriter; the owners decide whether to keep that sandbox |
| E2 | Elevation of privilege | Script injection in the browser | Content-Security-Policy with per-page hashes for inline scripts, no third-party scripts; React renders data as text (no `dangerouslySetInnerHTML` in the frontend) | `tests/web/test_web_security.py`, `tests/web/test_web_static.py`; S8 review | None known |
| E3 | Elevation of privilege | A vulnerable dependency or a committed secret | CI runs `make audit-deps` (pip-audit on the locked dependencies) and `make scan-secrets` (gitleaks over the whole history, pinned release, SHA-256 checked) on every push | CI jobs `audit-deps`, `scan-secrets` | The frontend's npm dependencies are not audited in CI; no scheduled run between pushes |

## 4. Not covered here

- The Hugging Face platform itself (Space isolation, OAuth provider, dataset storage).
- Physical and operator access to the container and its secrets.
- A real upload route, a real lender integration, payments (Studio rule H: Stripe test mode only, not built).
- Legal review. Whether any privacy or lending law applies depends on who runs the app and how; this page does not
  claim compliance with any.

## 5. Open items for the owners

1. Give the access log a persistent home in the Space and keep its head anchor elsewhere (docs/security.md 9.5).
2. Decide the access log's own retention, and whether delete-all should also remove the owner's events.
3. Decide the owner commitments the privacy page marks as pending (for example "data is never sold").
4. Decide the demo reviewer role on SYNTHETIC lender files (docs/security.md section 9, step 2).
5. Add `npm audit` (or similar) for the frontend and a scheduled CI run for newly published advisories.
