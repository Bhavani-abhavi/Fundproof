# Deployment: one container, a Hugging Face Docker Space

> **Status: not deployed yet: waiting for the `HF_TOKEN` and `HF_SPACE_ID` repository secrets.**
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
   uvicorn uwcopilot.web.app:app   (ONE process: rate limits and a generated session secret are per process)
     |-- /api/*            FastAPI: /api/health, /api/config, /api/me, /api/auth/* (feature routers plug in later)
     |      |-- sign-in -> huggingface.co OpenID Connect (OAUTH_* variables injected because hf_oauth: true)
     |      '-- Store   -> /app/data on the container disk (lost on restart), or, with UWC_STORE=hf_dataset,
     |                     an encrypted mirror in a PRIVATE Hugging Face dataset
     '-- everything else  the Next.js static export in /app/frontend/out (served by uwcopilot.web.static)
```

The deploy pipeline (`.github/workflows/deploy-space.yml`):

```
 "Run workflow" in GitHub Actions, or a pushed tag v*
   '-- job verify: ruff, pytest (offline suite), frontend lint + typecheck + contrast + build,
   |              docker build of ./Dockerfile, run it, scripts/smoke_deploy.py --target local-container
   '-- job deploy (only if verify passed):
          secrets HF_TOKEN and HF_SPACE_ID both set?  no -> a notice "Deploy skipped", nothing is pushed
          yes -> scripts/deploy_space.py bundle   copy the allow-listed files, stamp the commit into the Dockerfile
              -> scripts/deploy_space.py push     create_repo(space, docker, exist_ok, public) + upload_folder
              -> (the Space builds the same Dockerfile itself)
              -> scripts/deploy_space.py wait     until https://<host>/api/health names this commit
              -> scripts/smoke_deploy.py --target hf-space   results kept as the workflow artifact "space-smoke"
```

## 2. Files

| File | What it does |
|---|---|
| `Dockerfile` | Three stages: `node:22-slim` runs `npm ci` + `npm run build` (frontend/out); `python:3.11-slim` + uv runs `uv sync --locked --no-dev` into /app/.venv; the runtime stage copies only the virtualenv, `src/`, `config/`, `corpus/` and `frontend/out`, compiles `src/` to bytecode, and runs as `user` (uid 1000). Base images are pinned by digest. `HEALTHCHECK` calls `/api/health`. `CMD` is uvicorn on 0.0.0.0:7860 with `--no-access-log` (the app logs its own redacted line) and `--no-proxy-headers` (client IPs come only from the app's `UWC_TRUSTED_PROXY_HOPS` rule). |
| `.dockerignore` | An allow-list: everything is excluded, then only `pyproject.toml`, `uv.lock`, `README.md`, `src/`, `config/`, `corpus/` and `frontend/` come back, minus `node_modules`, `.next`, `out`, caches and any `.env`. So `.env`, `data/`, `results/`, `.venv` and the git history never reach the build. |
| `deploy/space/README.md` | The Space card. Its YAML header is the Space configuration: `sdk: docker`, `app_port: 7860`, `hf_oauth: true` (Sign in with Hugging Face), title, colors, `short_description`. No `license` key (see Owner decisions). The body has the disclaimer and the SYNTHETIC note. It becomes `README.md` in the Space. |
| `scripts/deploy_space.py` | `bundle`: copies the allow-list into an empty folder, puts the card in as `README.md`, writes the commit into `ARG UWC_GIT_SHA=`. `push`: creates the Space if missing and replaces its files (`delete_patterns="*"`; the Hub keeps `.gitattributes`). `wait`: waits until the Space's `/api/health` names the pushed commit, failing early on a build or runtime error. Reads `HF_TOKEN` and `HF_SPACE_ID` from the environment and never prints them. |
| `scripts/smoke_deploy.py` | Checks a running deployment and times it (see section 4). Writes `results/deployment_metrics.json`, one entry per target, each with a provenance block. |
| `.github/workflows/deploy-space.yml` | The pipeline above. Manual or on tags `v*`; one deploy at a time. |
| `Makefile` | `docker-build`, `docker-run`, `docker-smoke`, `space-bundle`. |
| `tests/web/test_deploy_space.py`, `tests/web/test_smoke_deploy.py` | Offline tests: a fake Hub, `httpx.MockTransport`, a fake clock and a fake `docker`. |

## 3. Run it locally

```bash
make docker-build     # docker build -t fundproof:dev .
make docker-run       # http://127.0.0.1:7860, dev login OFF, data lost when the container stops
make docker-smoke     # start the image, smoke-test it, stop it -> results/deployment_metrics.json (runs.local-container)
make space-bundle     # write what the workflow would upload to build/space, to inspect it
```

Behind a TLS-inspecting proxy (the machine these numbers were measured on has one), pass the proxy's CA bundle as
an optional BuildKit secret: `make docker-build EXTRA_CA_CERTS=/path/to/ca-bundle.crt`. It is mounted only while
`npm ci`, `pip install uv` and `uv sync` download packages and is not stored in the image. Without it (GitHub
Actions, Hugging Face) the base images' normal certificate stores are used.

## 4. Measured: local container (NOT the Space)

From `results/deployment_metrics.json`, entry `runs.local-container`, measured 2026-10-12T20:05:58Z by
`make docker-smoke`. **This is the image running on the build machine, reached over 127.0.0.1: no internet, no Space
proxy, not Space hardware.** The code was commit `b76803d` with uncommitted work in the tree (`git_dirty: true`),
on a 4-CPU machine shared with other jobs (load average 6.15 at the start of the run).

| What | Value | How it was measured |
|---|---|---|
| Image size on disk | 2.16GB | `docker image ls`, DISK USAGE column (2161253092 bytes from `docker image inspect`) |
| Image content size (compressed) | 487MB | `docker image ls`, CONTENT SIZE column |
| Container start to first healthy `/api/health` | 1.11 s | Docker's `State.StartedAt` to the first 200, polling every 0.1 s |
| First request to `/api/config` | 132.31 ms | the first call loads the Studio metro and business-type YAML files |
| All paths, 100 requests after the first | p50 1.67 ms, p95 2.42 ms | 20 sequential GETs each to `/api/health`, `/`, `/studio`, `/api/config`, `/api/me` |
| Checks | 9 of 9 passed | see the list below |

The checks: `/api/health` is ok; `/` and `/studio` are HTML with a Content-Security-Policy and `nosniff`;
`/api/config` has the product name, `payments_enabled: false` and `dev_auth: false`; `/api/me` is signed out; an
unknown `/api` path is a JSON 404; no request was rate limited or failed; the container runs the measured image.

These are smoke-test timings from one client sending one request at a time, not a load test, and they say nothing
about the Space: its hardware, network and cold start (including waking from sleep) are different and are
**not measured** until a real deploy runs (the deploy job then measures them and keeps them as the
`space-smoke` artifact).

**Why the image is large.** The runtime installs the project's full dependency set from `uv.lock` (`uv sync
--locked --no-dev`), as the build spec says: MLflow, LightGBM, SHAP, SciPy, pandas, pyarrow, onnxruntime and others.
The web shell itself imports only a small part of them today, but the feature routers planned next (agent,
document intelligence, Studio) will import more. Making the image smaller needs a `web` dependency group or extra in
`pyproject.toml`; that is an owners' decision (below) because it changes the shared dependency file.

## 5. Deploy to the Space

One-time setup (an owner does this; nothing here can do it for you):

1. **Create the Space first, with its secrets** (recommended order). On huggingface.co: New Space, SDK "Docker",
   blank template, under the account that will own it. In the Space's Settings -> Variables and secrets add at least
   `UWC_SESSION_SECRET` (section 6). If you skip this step, the first workflow run creates the Space, but the app
   refuses to start without `UWC_SESSION_SECRET` (a runtime error, by design), the `wait` step fails, and you add
   the secret and restart the Space by hand. Note: `create_repo(private=False)` does not change an existing Space,
   so a Space you created as private stays private.
2. **Create a Hugging Face access token** with write access that can update that Space (a fine-grained token
   limited to it is the safest choice).
3. **Add two GitHub repository secrets** (GitHub: Settings -> Secrets and variables -> Actions):
   `HF_TOKEN` (the token) and `HF_SPACE_ID` (`owner/name`, for example `astra6/fundproof`).
4. **Run the workflow**: Actions -> "Deploy to Hugging Face Space" -> Run workflow (pick a branch or tag), or push a
   tag: `git tag v0.2.0 && git push origin v0.2.0`.

Every run after that replaces the Space's files with the bundle (edits made directly in the Space repo are lost)
and the Space rebuilds the Dockerfile. To roll back, run the workflow from an older tag.

## 6. Secrets and settings

Space secrets are set by an owner in the Space settings, never by the workflow. The full list with comments is at
the end of `.env.example`; the meaning of each is in `src/uwcopilot/web/settings.py`.

| Name | Where | Required? | Purpose |
|---|---|---|---|
| `HF_TOKEN` | GitHub repository secret | to deploy | write token used by the deploy job |
| `HF_SPACE_ID` | GitHub repository secret | to deploy | `owner/name` of the Space |
| `UWC_SESSION_SECRET` | Space secret | **yes** (startup refuses without it in a Space) | signs session cookies; 32+ characters |
| `UWC_ROLE_ALLOWLIST` | Space secret or variable | for lender/admin roles | `hfuser:role,...`; everyone else is a founder |
| `UWC_STORE=hf_dataset`, `UWC_HF_DATASET_REPO` | Space variables | for data that survives restarts | private dataset repo, created private if missing |
| `HF_TOKEN` | Space secret | with `UWC_STORE=hf_dataset` | write access to that dataset (a different token from the deploy one is fine) |
| `UWC_FERNET_KEY` | Space secret | with `UWC_STORE=hf_dataset` | encrypts stored data; comma-separated for key rotation |
| `UWC_DEV_AUTH` | never in a Space | no | dev login; startup refuses it when `SPACE_ID` is set |
| `SPACE_ID`, `SPACE_HOST`, `OAUTH_*`, `OPENID_PROVIDER_URL` | set by Hugging Face | - | never set them by hand |

Without `UWC_STORE=hf_dataset`, everything users store lives on the container disk and is lost whenever the
Space restarts or rebuilds (the Hugging Face Docker Spaces docs say data written on disk is lost on restart;
read 2026-10-12 at huggingface.co/docs/hub/spaces-sdks-docker).

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
  Space; if that is wrong, rate limits may count every user as one client.
- Cold start and latency on the Space (the deploy job's `space-smoke` artifact), including waking from sleep.
- Whether the Space honors the Dockerfile `HEALTHCHECK` (not documented on the Docker Spaces page read
  2026-10-12; `/api/health` works either way).
- With `UWC_STORE=hf_dataset`: data survives a restart, and deletion removes it from the dataset.

## 9. Owner decisions (open)

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
- **Who may deploy.** Any branch can be deployed with "Run workflow"; restrict it (for example to tags only) if
  wanted.
- **`/api/docs`.** Swagger UI loads its script from cdn.jsdelivr.net (only that page's CSP allows it); keep it
  public or serve it only in development.
- **Final product name** (two constants, see the top of this page).

## 10. Troubleshooting

| Symptom | Likely cause |
|---|---|
| Deploy job ends with "Deploy skipped" | `HF_TOKEN` or `HF_SPACE_ID` repository secret missing |
| Space shows "Runtime error", log says `UWC_SESSION_SECRET is required in a Space` | add the Space secret, then restart the Space |
| Space log says `UWC_DEV_AUTH=1 is refused inside a Hugging Face Space` | remove `UWC_DEV_AUTH` from the Space settings |
| Space log says `UWC_STORE=hf_dataset needs ...` | add `UWC_HF_DATASET_REPO`, `HF_TOKEN` and `UWC_FERNET_KEY` |
| `wait` step times out | the Space build takes longer than 30 minutes or is queued; see the Space's build log |
| Sign-in fails inside the Space page | use "open in a new tab" (section 7) |
