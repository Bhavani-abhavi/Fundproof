# Founder Studio and Lender Desk: privacy notice for the v0.1 web app

> **Status: draft, written 2026-10-13 from the v0.1 code and updated 2026-10-20 for what Studio S8 built (share
> links, "delete all my data", the plan access log, the privacy and terms pages), for the owners (Bhavani Adula and
> Rahul Vajja) to review before the first public deploy.** Studio addendum rule B says the deployed app may take real
> founder input only under this document. Everything below describes what the code in this repository does today,
> with the file that does it; what is not built yet is listed at the end. This is a demonstration prototype, not a
> compliance program: whether any privacy law applies depends on who runs it and how, and this page does not claim
> compliance with any.

The app shows this notice in plain words on its privacy policy page (`frontend/app/privacy/page.tsx`, next to the
Terms of Use, `frontend/app/terms/page.tsx`) and a short version on its About page (section "Your information").
Update the privacy page and this document together. Until the owners approve this document, the privacy and terms
pages carry a "Draft - pending the owners' approval" banner (`frontend/components/legal/draft-banner.tsx`) and the
Founder Studio pages show a "Prototype: please use a made-up plan" notice
(`frontend/components/studio/prototype-data-notice.tsx`) asking visitors to use invented figures and public addresses
only; remove both when this document is approved.

## 1. What the app asks for, and what it refuses

| Input | Where | Kept? |
|---|---|---|
| Hugging Face sign-in | every signed-in page | The account id, username and display name in a signed session cookie (8 hours by default, `UWC_SESSION_MAX_AGE_S`). Only the `openid profile` scopes are requested (`web/settings.py`, `OAUTH_SCOPES`); no e-mail address. |
| A plan: city, business type, idea text (up to 2,000 characters), money figures, opening date, 1 to 3 sites (a map pin, or for a real city a street address), and through the API an optional plan name (up to 120 characters; the wizard does not ask for one) | Founder Studio wizard | Yes, as one encrypted object per plan (section 3). |
| A share link: how many days it works (1 to 30, 7 by default) | "Share this plan" on a plan's page | A record of the link (section 2). |
| A DEMO decision on a SYNTHETIC file: action and rationale | Lender Desk | Yes, as one encrypted record per user and file. |

Every access to a plan is also written to the plan access log (section 5).

**Refused, never stored** (`web/services/text_checks.py`, 422 with a plain message that does not repeat the text):
an SSN-shaped number (`123-45-6789`, also with spaces), a labeled account number (`acct 12345678`) or any run of 8
to 17 digits, in the plan name, the idea, a site name, a site address and a decision rationale. A site address is
checked **before** it is sent anywhere. The metro's own 11-digit census tract ids are allowed in site names (the
wizard writes them; a public tract id is not personal data).

There is **no upload route** in v0.1, so no tax returns, bank statements or other documents can be sent. The
screening such a route must call before it keeps a file is built (`web/services/upload_checks.py`: it refuses wrong
or mismatched file types, files that are too large, empty or unread, SSN shapes, labeled account numbers, bank
routing numbers and IRS tax-form markers, and its answer never repeats the file's text), but no route calls it yet
(section 6).

Requests are small by design: a JSON body over 64 KiB is refused (413) before it is read further
(`web/security.py`, `API_MAX_BODY_BYTES`).

## 2. Who else sees what you type

- **The US Census Geocoder** (a public service of the US Census Bureau at geocoding.geo.census.gov). When you type a
  street address for a real city, the server sends that address to it over HTTPS to find the location and census
  tract (`web/services/geocoding.py`). Nothing else from your plan is sent. The answer is not cached anywhere on the
  server (not on disk and not in a shared memory cache); the match is stored inside your encrypted plan, and a later
  change to the plan reuses it instead of asking again. Its terms of use are listed as "to verify" in
  [data_sources.md](data_sources.md).
- **People you send a share link to** (`web/routers/share.py`, `web/services/share_links.py`). Only if you create
  one, on the plan's page. Anyone who has the link can read the plan's shared view, without signing in, until the
  link expires (1 to 30 days, and never after the plan's own deletion date as it was when the link was made) or is
  revoked. What a link shows is built field by field (`web/schemas_privacy.py`, `SharedPlanOut`;
  `web/services/projects.py`, `shared_plan_out`), and it is exactly:
  - the plan's name, if it has one: free text the founder typed, screened only by the checks of section 1 (a plan
    made in the wizard has none, and the shared page then shows "business type in city");
  - the business type and the city;
  - the money figures, each with who wrote it (and, for a figure written by an applied plan change, when it was
    applied and the evidence still needed);
  - the opening date;
  - each site's name and approximate location: latitude and longitude rounded to 2 decimal places (about 1 km), so
    a link never gives the exact point of a typed street address;
  - when the link stops working, the disclaimer and a notice, which among other things says "approximate location
    (rounded to about 1 km)".

  Never the idea text, street addresses, the geocoder's matched address, the plan's id, version or seed, or anything
  about your account. The link's token is encrypted (Fernet, with a key derived from `UWC_SESSION_SECRET` and never
  the store key), so the account id inside it does not show to whoever holds it. Answers carry `Cache-Control:
  no-store` and `Referrer-Policy: no-referrer`, and every link that does not work gets the same 404. A plan keeps at
  most 10 working links. A link stops working when it expires, when you use "Revoke all share links" (section 4),
  when the plan is deleted or reaches its deletion date, after "Delete all my data", and when `UWC_SESSION_SECRET`
  changes. Every view of a link is written to the plan access log (section 5) with the link's id, never an IP
  address.
- **No language model** reads your idea text, your numbers or your rationale in v0.1. The idea text is screened for
  instruction-like text (`agent/injection.detect` after normalization) and, when flagged, marked as such; it is
  shown back only to you. The Lender Desk demo writes its memo with a fixed template (no model).
- **Nobody else signed in** can read your plans or records: every object is stored under your account id, and
  another user's id answers "not found" (tests: `tests/web/test_web_studio.py`, `tests/web/test_web_lender.py`).
- **Hugging Face** hosts the app (the Space) and, if the owners turn it on, a private dataset that mirrors the
  encrypted store (section 3). Apart from the geocoder and the share links you create, the app has no code that
  passes your data to anyone else, and no analytics or tracking script (the Content-Security-Policy allows scripts
  and connections only to the app's own origin). Whether the owners commit to "never sold" in a policy is their
  decision (section 6).

## 3. How it is stored

- **Encrypted at rest** with Fernet (`web/crypto.py`, `web/store.py`): plans, decision records and share-link
  records are encrypted before they are written, both on the container disk and in the optional private Hugging
  Face dataset (`UWC_STORE=hf_dataset`, `web/hf_store.py`, which refuses a public dataset).
- **Per account**: objects live under a digest of the account id (`store.owner_digest`). A share-link record (the
  link's id, the plan's id, when it was made and when it expires) is stored in the owner's own "shares" namespace;
  nothing lists the links of all users.
- **Results** (site scores, simulations, path searches, lender file runs) are kept in the server's memory only, per
  plan version, and dropped when the plan changes or is deleted, and when the server restarts
  (`web/services/compute.py`).
- **Address lookups** run in their own small worker pool with per-user limits (one batch at a time, a small budget
  of addresses per minute), so they cannot be used to flood the geocoder from this server.
- **The plan access log is not in the store**: it is its own file of plain (not encrypted) JSON Lines, never
  uploaded to the dataset (section 5).

## 4. Deletion and retention

- **One plan**: "Delete plan" on a plan's page deletes the plan, revokes its share links first and drops its results
  from memory.
- **Share links**: "Revoke all share links" on a plan's page (`DELETE /api/studio/projects/{id}/share`, answer
  `{"revoked": <int>, "pending": <bool>}`) deletes the records of every link of that plan, so the links stop working
  on this server at once. `revoked` counts the working links (expired ones are not counted). `pending` is true when
  the backup dataset copy of a link's record could not be deleted yet: the link stops working here at once, but if
  the app restarts before the backup deletion goes through, the record can come back from the dataset, so the
  founder has to revoke again. The plan page says so, and does not say that the link "no longer works".
- **A DEMO decision record** can be deleted from its file (`DELETE /api/lender/files/{id}/record`); the Lender Desk
  page has no button for it yet.
- **Everything at once**: "Delete all my data" on the About page (`DELETE /api/me/data`, `web/routers/account.py`)
  deletes every object stored under your account with `store.delete_owner` (plans, DEMO decision records and
  share-link records, so every link you made stops working), drops your results kept in memory, and ends the session
  (the session cookie is cleared). It first writes an event to the plan access log; when that event cannot be
  written, nothing is deleted (503) and the page asks you to try again. `"pending": true` means the copy in the
  dataset could not be deleted yet: it is deleted here, the dataset deletion is retried automatically, and the page
  says the deletion is pending instead of claiming it is done.
- **What stays after "Delete all my data", and after a plan's deletion or retention deletion**: the plan access log
  keeps its events, including the event of the deletion itself (section 5). They hold an owner digest, plan ids,
  action names and times, never plan content, names, addresses or IP addresses. The CSRF cookie also stays (it is not
  tied to an account).
- **With the dataset store**, the deleted files are removed from the dataset at once and its history is squashed at
  the next flush so earlier revisions do not keep them (`web/hf_store.py`); if the Hub cannot be reached the app says
  the deletion is pending instead of claiming it is done.
- **Automatically**: every stored object expires after its retention period and is deleted at startup and then every
  hour (`UWC_PURGE_EVERY_S`). Plans and decision records keep the default period (`UWC_RETENTION_DAYS`, 30 days by
  default, a design choice); a plan's page shows its own deletion date. A share-link record is kept until its link
  expires, rounded up to whole days. The plan access log has no retention period: there is no automatic deletion of
  the access log yet (section 6).
- Without the dataset store, everything also disappears when the Space restarts or rebuilds.

## 5. Logs

- One line per request with the method, the path with personal patterns masked (a share-link token in a path is
  logged as `/api/share/*`), the **names** of query parameters (never their values), the status and the time
  (`web/access_log.py`). No bodies, cookies, headers, IP addresses or usernames are logged; uvicorn's own access log
  is off (`--no-access-log` in the Dockerfile).
- A failed address lookup is logged with the geocoder's URL without its query and the error, not the address.
- A failing Lender Desk tool is logged in full on the server only; the page and the DEMO record show a plain
  sentence (`web/services/lender_demo.py`, `SafeTool`).
- **The plan access log** (`web/audit_log.py`): one event written **before** every access to a founder's plan. The
  actions are listing your plans, reading, creating and changing a plan, the three computations (site scores,
  feasibility, path to approval), creating and revoking share links, deleting a plan, "Delete all my data", and every
  view of a valid share link. When the event cannot be written, the request is refused (503) and the access does not
  happen.
  - What an event holds: its sequence number and id, the time, the event type, the action name, the actor (an id,
    the role "unknown" and the kind of sign-in), the plan
    as the resource (kind `studio_plan` and the plan's random id; none for the plan list and for "Delete all my
    data"), the outcome, details (who acted, "founder" or "share_link"; for a new link, its id and number of days),
    the data label `USER_DATA`, and the hashes that chain it to the event before. The actor of a founder is the
    **owner digest**: the first 32 hex characters of SHA-256 of the account id (`store.owner_digest`, the same digest
    that names the account's store folder). It is pseudonymous, not anonymous: anyone who knows the account id can
    compute it and link the events back to the account. The actor of a link view is `share_link:<link id>`.
  - What an event never holds: plan content, names, addresses, money, the account id itself, the username or an IP
    address.
  - Where: the file `UWC_WEB_AUDIT_LOG` (by default `data/web/audit/plan_access_events.jsonl`, next to the store
    folder and never inside it, so the dataset store never uploads it), plain JSON Lines (not encrypted),
    hash-chained with the repo's `AuditLog` (`security/audit.py`); `scripts/verify_audit.py --log <file>` checks the
    chain.
  - How long: the log keeps its events after "Delete all my data", after a plan is deleted and after retention
    deletion, and there is no automatic deletion of the access log yet. In the hosted Space this log is on the
    container disk, which does not survive a restart, unless the operator points `UWC_WEB_AUDIT_LOG` at persistent
    storage.

## 6. Not built yet

- **An upload route.** The screening it must call is built (`web/services/upload_checks.py`, section 1); no route
  accepts a file in v0.1.
- **The owners' own commitments**, for example that data is never sold, and their approval of this document and of
  the privacy and terms pages (both pages carry a draft banner until then).
- **A persistent home for the plan access log.** In the hosted Space it is on the container disk and is lost at a
  restart unless the operator points `UWC_WEB_AUDIT_LOG` at persistent storage; it also has no retention period or
  automatic deletion yet (section 5).
