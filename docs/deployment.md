# Deployment: one container, a Hugging Face Docker Space

> **Status: not deployed yet.** The owners created the Space `astra6/FundProof` and added the `HF_TOKEN` repository
> secret on 2026-10-13; the first deploy runs once CI is green.
> There is no public URL. This page gets the URL only after a real deploy has run.

Fundproof (working name; rename in `src/uwcopilot/web/settings.py` and `frontend/lib/config.ts`) is deployed as a
**prototype** on a Hugging Face **Docker Space**. Nothing in this repository pays for anything: no payment code, no
paid hardware or storage is requested (the deploy script calls `create_repo` without `space_hardware` or
`space_storage`, so the Space gets whatever the account's default is; check it in the Space settings).

Every number on this page is copied from `results/deployment_metrics.json`, which `scripts/smoke_deploy.py`
writes; the file says which machine and which target each number comes from.

## 1. What runs where

```
 Browser
   |  https://huggingface.co/spaces/<owner>/<space>  (the app inside an iframe)
   |  or https://<owner>-<space>.hf.space            (the app on its own, "open in a new tab")
   v
 Hugging Face Space front end (TLS ends here; how many proxies append X-Forwarded-For is NOT verified yet)
   |  plain HTTP, port 7860 (README metadata app_port: 7860)
   v
 Docker container built from ./Dockerfile, running as uid 1000
   uvicorn uwcopilot.web.app:app   (ONE process: rate limits, the compute cache and a generated session secret
                                    are per process)
     |-- /api/*            FastAPI: /api/health, /api/config, /api/me, /api/auth/*,
     |      |                       /api/studio/* (Founder Studio), /api/lender/* (Lender Desk DEMO)
     |      |-- sign-in -> huggingface.co OpenID Connect (OAUTH_* variables injected because hf_oauth: true)
     |      |-- request bodies over 64 KiB -> 413; validation answers never echo the input
     |      |-- heavy work (simulation, path search, agent run) -> worker threads, 60 s per request, then 503;
     |      |                  at most 3 jobs per user running or queued
     |      |-- lender demo -> the LangGraph agent with the template memo (no language model), the guideline
     |      |                  index in /app/qdrant_storage, OFAC lists in /app/data/raw/ofac (UWC_SCREENING_LISTS=ofac)
     |      |-- address sites -> Census Geocoder (keyless, real metros only; needs the Space's internet access;
     |      |                  its own 2-thread pool, one lookup batch per user, 10 s timeout, no cache)
     |      '-- Store   -> /app/data on the container disk (lost on restart), or, with UWC_STORE=hf_dataset,
     |                     an encrypted mirror in a PRIVATE Hugging Face dataset (Hub commits never block a
     |                     request: they hold only the Hub lock; store calls run in worker threads)
     '-- everything else  the Next.js static export in /app/frontend/out (served by uwcopilot.web.static)
```

What v0.1 can do in the image (section 4b has the measurements):

- **Founder Studio** works end to end on the SYNTHETIC demo metro: wizard, site scores, the 1,000-path simulation,
  the path to approval and "Apply this change". The five real metros run in **limited mode** in the image, because
  the Studio data snapshots (`data/studio/`) are not in it: real-metro sites can be added by address, their factors
  show as UNKNOWN, and the simulation, the path search and area suggestions answer 409 "limited mode".
- **Lender Desk DEMO** shows the SYNTHETIC structured applications of `eval/applications` and
  `eval/startup_applications` (credit, insurance and startup credit). Each file
  runs the agent graph to the human checkpoint with the deterministic template memo, real guideline search and OFAC
  screening; a person records a DEMO decision through `agent.hitl.validate_decision`.

The deploy pipeline (`.github/workflows/deploy-space.yml`):

```
 "Run workflow" in GitHub Actions, or a pushed tag v*
   '-- job verify: ruff, pytest (offline suite), frontend lint + typecheck + contrast + build,
   |              docker build of ./Dockerfile, run it, scripts/smoke_deploy.py --target local-container
   '-- job deploy (only if verify passed):
          secret HF_TOKEN set?  no -> a notice "Deploy skipped", nothing is pushed (Space id: secret or variable
          HF_SPACE_ID, else the default astra6/FundProof)
          yes -> scripts/deploy_space.py bundle   copy the allow-listed files, stamp the commit into the Dockerfile
              -> scripts/deploy_space.py push     create_repo(space, docker, exist_ok, public), set a fresh
                                                  UWC_SESSION_SECRET Space secret, upload_folder
              -> (the Space builds the same Dockerfile itself)
              -> scripts/deploy_space.py wait     until https://<host>/api/health names this commit
              -> scripts/smoke_deploy.py --target hf-space   results kept as the workflow artifact "space-smoke"
```

## 2. Files

| File | What it does |
|---|---|
| `Dockerfile` | Four stages: `node:22-slim` runs `npm ci` + `npm run build` (frontend/out); `python:3.11-slim` + uv runs `uv sync --locked --no-dev` into /app/.venv; `assets` downloads the OFAC SDN + alternate-name lists (required, two retries) and the public regulatory documents (optional), builds the Qdrant guideline index with `scripts/build_index.py` and runs one real guideline search, which also downloads the reranker (the build fails if it finds nothing); the runtime stage copies only the virtualenv, `src/`, `config/`, `corpus/`, `frontend/out`, the SYNTHETIC demo applications, `results/kyc_metrics.json` (the screening threshold) and the assets (lists, `checksums.jsonl`, fastembed models, index, index manifest), compiles `src/` to bytecode, sets `UWC_SCREENING_LISTS=ofac`, and runs as `user` (uid 1000; only `data/` and `qdrant_storage/` are writable). Base images are pinned by digest. `HEALTHCHECK` calls `/api/health`. `CMD` is uvicorn on 0.0.0.0:7860 with `--no-access-log` (the app logs its own redacted line) and `--no-proxy-headers` (client IPs come only from the app's `UWC_TRUSTED_PROXY_HOPS` rule). |
| `.dockerignore` | An allow-list: everything is excluded, then only `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `config/`, `corpus/`, `frontend/`, `eval/applications/`, `eval/startup_applications/`, `results/kyc_metrics.json`, `scripts/download_data.py` and `scripts/build_index.py` come back, minus `node_modules`, `.next`, `out`, the mock API and screenshot helpers, caches and any `.env`. So `.env`, `data/`, the other results, `.venv` and the git history never reach the build. |
| `deploy/space/README.md` | The Space card. Its YAML header is the Space configuration: `sdk: docker`, `app_port: 7860`, `hf_oauth: true` (Sign in with Hugging Face), title, colors, `short_description`. No `license` key (see Owner decisions). The body has the disclaimer and the SYNTHETIC note. It becomes `README.md` in the Space. |
| `scripts/deploy_space.py` | `bundle`: copies the allow-list into an empty folder, puts the card in as `README.md`, writes the commit into `ARG UWC_GIT_SHA=`. `push`: creates the Space if missing and replaces its files (`delete_patterns="*"`; the Hub keeps `.gitattributes`). `wait`: waits until the Space's `/api/health` names the pushed commit, failing early on a build or runtime error. Reads `HF_TOKEN` and `HF_SPACE_ID` from the environment and never prints them. |
| `scripts/smoke_deploy.py` | Checks a running deployment and times it (see section 4). Writes `results/deployment_metrics.json`, one entry per target, each with a provenance block. |
| `.github/workflows/deploy-space.yml` | The pipeline above. Manual or on tags `v*`; one deploy at a time. Every action is pinned to a full commit SHA (with its release in a comment, resolved 2026-10-13), uv is pinned to the Dockerfile's version (0.11.32), and checkouts do not keep the GitHub token (`persist-credentials: false`): the deploy job runs code with `HF_TOKEN` in its environment, so a moved tag must not be able to change that code. `ci.yml` uses the same pins. |
| `scripts/web_metrics.py` | Drives a LOCAL server or container that runs WITH the dev login in headless Chromium (Playwright): the signed-out pages, the founder flow on the SYNTHETIC metro (wizard, results, edited targets, "Apply this change", delete), one Lender Desk DEMO decision and the error paths; axe-core scans of every page state in light and dark; Lighthouse (mobile and desktop presets) on the landing page, the results page and the lender file. Writes `results/web_metrics.json` (one entry per target, with a provenance block) and screenshots to `docs/web/screenshots/e2e/`. |
| `tools/web-metrics/` | `package.json` + `package-lock.json` pinning the Node tools the script uses: Lighthouse 13.5.0 (released 2026-09-18) and axe-core 4.13.0 (2026-08-05). Not part of the app or the image. |
| `scripts/optimize_pngs.py` | Re-encodes PNGs losslessly (oxipng when installed, else Pillow) and keeps a file only when it is smaller and decodes to the same pixels. |
| `Makefile` | `docker-build`, `docker-run`, `docker-smoke`, `space-bundle`, `web-metrics-tools`, `web-metrics`. |
| `tests/web/test_deploy_space.py`, `tests/web/test_smoke_deploy.py`, `tests/web/test_web_metrics_script.py`, `tests/web/test_optimize_pngs.py` | Offline tests: a fake Hub, `httpx.MockTransport`, a fake clock, a fake `docker`, hand-written axe and Lighthouse results, generated PNGs. |

## 3. Run it locally

```bash
make docker-build     # docker build -t fundproof:dev .
make docker-run       # http://127.0.0.1:7860, dev login OFF, data lost when the container stops
make docker-smoke     # start the image, smoke-test it, stop it -> results/deployment_metrics.json (runs.local-container)
make space-bundle     # write what the workflow would upload to build/space, to inspect it
make web-metrics-tools  # once: npm ci in tools/web-metrics (Lighthouse + axe-core)
make web-metrics      # start the image WITH the dev login on 127.0.0.1:7862, run scripts/web_metrics.py, stop it
                      # -> results/web_metrics.json (runs.local-container) + docs/web/screenshots/e2e/
uv run --with pyoxipng==9.1.1 python scripts/optimize_pngs.py docs/web/screenshots   # lossless, smaller PNGs
```

The image needs network access while it builds (OFAC lists, regulatory documents, fastembed models from
huggingface.co). At runtime it needs the internet only for Hugging Face sign-in, the Census Geocoder (real-metro
addresses) and, with `UWC_STORE=hf_dataset`, the dataset repo: the containers measured in sections 4 and 4b could not
resolve any internet host (checked from inside the image: `huggingface.co` and `geocoding.geo.census.gov` gave "Name or
service not known"), and the Studio and Lender Desk flows still ran.

Behind a TLS-inspecting proxy (the machine these numbers were measured on has one), pass the proxy's CA bundle as
an optional BuildKit secret: `make docker-build EXTRA_CA_CERTS=/path/to/ca-bundle.crt`. It is mounted only while
`npm ci`, `pip install uv`, `uv sync` and the asset stage download and is not stored in the image. Without it
(GitHub Actions, Hugging Face) the base images' normal certificate stores are used. On the machine these numbers were
measured on, that proxy listens on the host's loopback, which build steps cannot reach on Docker's default network,
so the image was built with:

```bash
docker build --network=host --build-arg HTTPS_PROXY="$HTTPS_PROXY" --build-arg HTTP_PROXY="$HTTP_PROXY" \
    --build-arg NO_PROXY="$NO_PROXY" --secret id=extra_ca_certs,src=/root/.ccr/ca-bundle.crt -t fundproof:dev .
```

(Docker's predefined proxy build arguments are not stored in the image or its history.)

## 4. Measured: local container (NOT the Space)

From `results/deployment_metrics.json`, entry `runs.local-container`, measured 2026-10-13T19:04:57Z by
`scripts/smoke_deploy.py --url http://127.0.0.1:7860 --target local-container --docker-image fundproof:dev
--container fundproof-smoke` (run by `make docker-smoke`). **This is the v0.1 image running on the build machine,
reached over 127.0.0.1, with the dev login OFF: no internet, no Space proxy, not Space hardware.** The code was commit
`6e16420` with uncommitted v0.1 work in the tree (`git_dirty: true`), on a 4-CPU machine shared with other jobs (load
average 5.31 at the start of the run).

| What | Value | How it was measured |
|---|---|---|
| Image size on disk | 2.57GB | `docker image ls`, DISK USAGE column (2571584975 bytes from `docker image inspect`) |
| Image content size (compressed) | 659MB | `docker image ls`, CONTENT SIZE column |
| Container start to first healthy `/api/health` | 2.709 s | Docker's `State.StartedAt` to the first 200, polling every 0.1 s |
| First request to `/api/config` | 135.67 ms | the first call loads the Studio metro and business-type YAML files |
| All paths, 100 requests after the first | p50 2.3 ms, p95 3.28 ms | 20 sequential GETs each to `/api/health`, `/`, `/studio`, `/api/config`, `/api/me` |
| Checks | 9 of 9 passed | see the list below |

The checks: `/api/health` is ok; `/` and `/studio` are HTML with a Content-Security-Policy and `nosniff`;
`/api/config` has the product name, `payments_enabled: false` and `dev_auth: false`; `/api/me` is signed out; an
unknown `/api` path is a JSON 404; no request was rate limited or failed; the container runs the measured image.

These are smoke-test timings from one client sending one request at a time, not a load test, and they say nothing
about the Space: its hardware, network and cold start (including waking from sleep) are different and are
**not measured** until a real deploy runs (the deploy job then measures them and keeps them as the
`space-smoke` artifact). The previous shell-only image measured 2.16GB on disk and 1.104 s to healthy (this file's
earlier version, commit `017667d`); v0.1 adds the guideline index, the fastembed models, the OFAC lists and the
feature routers. Section 4b has the browser measurements of the same image.

## 4b. Measured: browser flows, accessibility and Lighthouse (local container WITH the dev login)

From `results/web_metrics.json`, entry `runs.local-container`, measured 2026-10-13T19:09:56Z by `make web-metrics`
(`scripts/web_metrics.py` against the same image, id `sha256:ff604e03465b...`, started with `UWC_DEV_AUTH=1` on
127.0.0.1:7862). **The dev login is on only because a script cannot sign in with Hugging Face; the Space refuses it,
so none of this was measured on the Space.** Same machine and commit as section 4 (`6e16420` + uncommitted work), load
average 5.43 at the start. Headless Chromium 141.0.7390.37 (Playwright 1.56.0), axe-core 4.13.0, Lighthouse 13.5.0.

End-to-end flows, all passed, with no unexpected console error or failed API answer (the expected ones are the 404s
of the error paths):

| Flow | Steps | What it shows |
|---|---|---|
| signed_out | 3 of 3 | public pages load; a signed-in page asks to sign in with a `?next=` link; an unknown page is the 404 page |
| founder_synthetic_plan | 9 of 9 | dev login form, the six wizard steps on the SYNTHETIC metro (one suggested area, one tract from the list), results page with the disclaimer, the "simulated, based on stated assumptions" label and the SYNTHETIC label; with the default targets the path search answered `no_path_within_bounds`; with edited targets `paths_found`, and "Apply this change" saved version 2 and re-ran |
| lender_demo_decision | 3 of 3 | queue, file `APP-C-001` (template memo, memo check passed, guideline search `ok`, OFAC `sdn.csv` and `alt.csv` with snapshot date 2026-10-13, screening `clear`), a DEMO `approve` by the mapped `underwriter` role, recorded as `approved_by_human` |
| error_paths | 2 of 2 | an unknown plan id gets the API's 404 and the page says "Not found"; the wizard refuses to continue without a city |
| founder_delete_plan | 1 of 1 | "Delete plan" -> DELETE 200 -> the plan's page then says "Not found" |

Timings inside those flows (one run, milliseconds from the click; the machine was busy with other jobs):

| What | Value |
|---|---|
| "Create plan" click to the plan answer / site scores / simulation / path search answers | 129.3 / 326.2 / 793.3 / 1253.3 ms |
| "Create plan" click to the path-to-approval result on screen | 1391.1 ms |
| "Apply this change": PUT answer / re-run path result | 140.9 / 412.5 ms |
| First lender file after the container started (agent run, retriever and matcher load) | 14608.7 ms to the API answer |
| Recording the DEMO decision | 338.4 ms |

axe-core: 28 scans (14 page states x light and dark, at 1280 px): **0 critical and 0 serious violations, and no
violation of any impact**. 16 scans have a "serious" item axe could not decide (`incomplete`), which needs a person to
review. Zero violations is what automated checks can find, not a statement of WCAG conformance.

Lighthouse, median of 3 runs per page and preset (scores as Lighthouse shows them, 0-100):

| Page | Preset | Performance (3 runs) | Accessibility | Best practices | LCP (median) | TBT (median) | CLS (median) |
|---|---|---|---|---|---|---|---|
| Landing `/` | mobile | 82 (82, 78, 82) | 100 | 100 | 4142 ms | 162 ms | 0.001 |
| Landing `/` | desktop | 98 (98, 99, 98) | 100 | 100 | 915 ms | 0 ms | 0.003 |
| Studio results | mobile | 41 (41, 43, 41) | 100 | 100 | 7080 ms | 2051 ms | 0.112 |
| Studio results | desktop | 84 (77, 84, 84) | 100 | 100 | 1563 ms | 186 ms | 0.139 |
| Lender file | mobile | 77 (74, 77, 94) | 100 | 100 | 3590 ms | 294 ms | 0.013 |
| Lender file | desktop | 91 (91, 91, 91) | 100 | 100 | 1369 ms | 59 ms | 0.139 |

"mobile" is Lighthouse's default configuration (simulated throttling of a mid-range phone); "desktop" is
`--preset=desktop`. The HTTP cache was cleared before each run; the session cookie was kept. The results page is
the slow one on mobile (long main-thread work while the charts and the map render), and the results page and the
lender file shift layout on desktop (CLS above 0.1); both are open frontend issues. LCP, TBT and CLS values are
rounded here for reading; the results file has the exact values. Screenshots of the run:
`docs/web/screenshots/e2e/`.

**Why the image is large.** The runtime installs the project's full dependency set from `uv.lock` (`uv sync
--locked --no-dev`), as the build spec says: MLflow, LightGBM, SHAP, SciPy, pandas, pyarrow, onnxruntime and others.
The v0.1 routers import the agent (LangGraph), the Studio engine and the retrieval stack (fastembed, onnxruntime,
qdrant-client), but still not everything (MLflow, SHAP and the document-intelligence stack are unused by the web
app); the image also carries the fastembed models and the guideline index. Making the image smaller needs a `web`
dependency group or extra in `pyproject.toml`; that is an owners' decision (below) because it changes the shared
dependency file.

## 5. Deploy to the Space

One-time setup (an owner does this; nothing here can do it for you):

1. **Create the Space** (done by the owners: `astra6/FundProof`, Docker SDK). On huggingface.co: New Space, SDK
   "Docker", blank template, under the account that will own it. The app refuses to start in a Space without
   `UWC_SESSION_SECRET`; `scripts/deploy_space.py push` sets that Space secret to a fresh random value on every
   deploy (so each deploy signs everyone out; pass `--keep-session-secret` if the owners manage it themselves).
   Note: `create_repo(private=False)` does not change an existing Space, so a private Space stays private.
2. **Create a Hugging Face access token** with write access that can update that Space (a fine-grained token
   limited to it is the safest choice).
3. **Add the GitHub repository secret** `HF_TOKEN` (GitHub: Settings -> Secrets and variables -> Actions); done
   by the owners on 2026-10-13. The Space id defaults to `astra6/FundProof`; set a repository secret or variable
   `HF_SPACE_ID` (`owner/name`) only to deploy somewhere else.
4. **Run the workflow**: Actions -> "Deploy to Hugging Face Space" -> Run workflow (pick a branch or tag), or push a
   tag: `git tag v0.2.0 && git push origin v0.2.0`.

Every run after that replaces the Space's files with the bundle (edits made directly in the Space repo are lost)
and the Space rebuilds the Dockerfile. To roll back, run the workflow from an older tag.

## 6. Secrets and settings

Space secrets are set by an owner in the Space settings; the only one the workflow sets is `UWC_SESSION_SECRET`
(a fresh random value per deploy, never printed). The full list with comments is at
the end of `.env.example`; the meaning of each is in `src/uwcopilot/web/settings.py`.

| Name | Where | Required? | Purpose |
|---|---|---|---|
| `HF_TOKEN` | GitHub repository secret | to deploy | write token used by the deploy job |
| `HF_SPACE_ID` | GitHub repository secret or variable | no (default `astra6/FundProof`) | `owner/name` of the Space |
| `UWC_SESSION_SECRET` | Space secret | **yes** (startup refuses without it in a Space); set by the deploy job | signs session cookies; 32+ characters |
| `UWC_ROLE_ALLOWLIST` | Space secret or variable | for lender/admin roles | `hfuser@<id>:role,...` (pin each entry to the account id, `subject` in `/api/me`; see `docs/web/auth.md`, Roles); everyone else is a founder |
| `UWC_RETENTION_DAYS`, `UWC_PURGE_EVERY_S` | Space variables | no | how long stored objects live (default 30 days) and how often expired ones are deleted (default 3600 s, plus once at startup) |
| `UWC_STORE=hf_dataset`, `UWC_HF_DATASET_REPO` | Space variables | for data that survives restarts | private dataset repo, created private if missing |
| `HF_TOKEN` | Space secret | with `UWC_STORE=hf_dataset` | write access to that dataset (a different token from the deploy one is fine) |
| `UWC_FERNET_KEY` | Space secret | with `UWC_STORE=hf_dataset` | encrypts stored data; comma-separated for key rotation |
| `UWC_DEV_AUTH` | never in a Space | no | dev login; startup refuses it when `SPACE_ID` is set |
| `SPACE_ID`, `SPACE_HOST`, `OAUTH_*`, `OPENID_PROVIDER_URL` | set by Hugging Face | - | never set them by hand |

Without `UWC_STORE=hf_dataset`, everything users store lives on the container disk and is lost whenever the
Space restarts or rebuilds (the Hugging Face Docker Spaces docs say data written on disk is lost on restart;
read 2026-10-12 at huggingface.co/docs/hub/spaces-sdks-docker).

**Deletion and retention** (`src/uwcopilot/web/hf_store.py`, `src/uwcopilot/web/app.py`):

- A dataset repo is a git repo, so a plain delete would leave the file in earlier revisions. After a deletion the
  next flush (at most `UWC_HF_FLUSH_EVERY_S`, default 300 s, later; also at shutdown) squashes the dataset's
  history into one commit (`super_squash_history`) and permanently deletes LFS objects no current file uses
  (`permanently_delete_lfs_files`). The history is also squashed once after every startup.
- If the Hub cannot be reached, the deletion is kept as "pending" (in memory and in
  `<store dir>/.pending_deletes.json`), retried at every flush and re-applied at startup, and the caller gets
  `DeletionPending` instead of a success, so the app must not tell the user the data is gone. If the container
  restarts before the Hub is reachable again, the pending deletion is lost with the disk; the user was told it was
  pending and has to delete again.
- Expired objects (older than `UWC_RETENTION_DAYS`) are hidden at once and deleted once at startup and then every
  `UWC_PURGE_EVERY_S` seconds, in both stores.
- All of this is tested with a fake Hub that keeps every revision and LFS object (`tests/web/test_web_hf_store.py`);
  it has not run against the real Hub yet (section 8).

## 7. The iframe and cookies

A Space shows the app inside an iframe on huggingface.co. For the browser the app's cookies are then third-party
cookies, so in a Space the session cookie is `Secure; SameSite=None; Partitioned`, the CSP allows
`frame-ancestors 'self' https://huggingface.co`, and the sign-in page offers "open in a new tab" (the Space's own
`*.hf.space` address), because some browsers still block sign-in cookies inside the iframe. The iframe and the new
tab keep separate cookies. Details, with the Hugging Face docs quote: `docs/web/auth.md`, "Why `SameSite=None;
Partitioned` in a Space".

## 8. Check on the first deploy

None of these can be checked without a real Space:

- Sign in with Hugging Face inside the iframe and in a new tab (Chrome, Firefox, Safari).
- The OAuth callback URL (`https://<SPACE_HOST>/api/auth/callback`) is accepted.
- How many proxies append to `X-Forwarded-For` in front of the app. `UWC_TRUSTED_PROXY_HOPS` defaults to 1 in a
  Space; if that is wrong, rate limits may count every user as one client. (Several `X-Forwarded-For` header
  lines are joined in order before counting, so a client cannot hide the proxy's entry behind its own line.)
- Cold start and latency on the Space (the deploy job's `space-smoke` artifact), including waking from sleep.
- Whether the Space honors the Dockerfile `HEALTHCHECK` (not documented on the Docker Spaces page read
  2026-10-12; `/api/health` works either way).
- With `UWC_STORE=hf_dataset` and a SYNTHETIC test object: data survives a restart; after deleting it and waiting
  one flush, the dataset has a single commit (`list_repo_commits`), the object's files are gone, and
  `list_lfs_files` lists no object that the current files do not use.
- v0.1: the Space's build log shows the OFAC lists downloaded and a `guideline search check: N hits` line with N above 0 (the build fails
  if `www.treasury.gov` refuses the Space's builder three times; whether it does is not known yet). The regulatory
  documents are optional: if they are missing, `results/rag_index_manifest.json` in the image lists them.
- v0.1: a Lender Desk file shows "guideline search: ok" and the OFAC snapshot date; the first file opened after a
  start is slow (the retriever and the screening matcher load), and its time on Space hardware is not measured.
- v0.1: a real-metro plan with a street address geocodes through the Census Geocoder from the Space (not tested
  from the local container, which had no internet access) and then shows the limited-mode messages.
- The browser flows of `scripts/web_metrics.py` need the dev login, so they cannot run against the Space; check the
  same pages there by hand after signing in with Hugging Face.

## 9. Owner decisions (open)

- **Privacy notice.** Studio addendum rule B lets the deployed app take real founder input only under
  `docs/studio/privacy.md`. That file is a draft written from the v0.1 code on 2026-10-13 (what is collected, the
  Census Geocoder, encryption, deletion and retention, logs, and what S8 still has to build); the owners review and
  approve it before the first public deploy.
- **License.** The repository has no LICENSE file, so the Space card has no `license` key. Choosing one is the
  owners' decision.
- **Space visibility.** The workflow creates the Space public (`private=False`), as planned for a usable prototype.
  The files of a public Space are public too: the bundle (`src/`, `config/`, `corpus/`, the frontend source,
  `Dockerfile`, `pyproject.toml`, `uv.lock`) becomes readable on huggingface.co even if the GitHub repository is
  private.
- **Persistent storage.** Container disk only (lost on restart) or the private dataset store (needs
  `UWC_FERNET_KEY` and a token), and the retention period (`UWC_RETENTION_DAYS`, default 30 days, a design choice).
- **Image size.** Keep the full dependency set, or add a `web` dependency group so the image installs only what the
  web app imports.
- **Real metros in the Space.** The image has no Studio data snapshots, so every real metro runs in limited mode.
  Building them into the image needs the Census, BLS, FRED and HUD keys as Space build secrets (not set up).
- **Who may deploy.** Any branch can be deployed with "Run workflow"; restrict it (for example to tags only) if
  wanted.
- **`/api/docs`.** Swagger UI loads one pinned script and one pinned stylesheet (`swagger-ui-dist@5.33.0`) from
  cdn.jsdelivr.net, each with a Subresource Integrity hash, and that page's CSP allows exactly those two URLs, so
  a changed file is refused by the browser. Keep the page public, or serve it only in development.
- **Final product name** (two constants, see the top of this page).

## 10. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Deploy job ends with "Deploy skipped" | `HF_TOKEN` repository secret missing |
| Space shows "Runtime error", log says `UWC_SESSION_SECRET is required in a Space` | the push ran with `--keep-session-secret` and the secret is missing: add it, then restart the Space |
| Space log says `UWC_DEV_AUTH=1 is refused inside a Hugging Face Space` | remove `UWC_DEV_AUTH` from the Space settings |
| Space log says `UWC_STORE=hf_dataset needs ...` | add `UWC_HF_DATASET_REPO`, `HF_TOKEN` and `UWC_FERNET_KEY` |
| `wait` step times out | the Space build takes longer than 30 minutes or is queued; see the Space's build log |
| Sign-in fails inside the Space page | use "open in a new tab" (section 7) |
