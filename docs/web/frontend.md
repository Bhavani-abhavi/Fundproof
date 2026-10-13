# Fundproof web frontend (Studio S7 shell)

Written 2026-10-12 for Bhavani Adula and Rahul Vajja. This page explains every file in `frontend/`, the design
choices, how to run and check it, and what is not built yet. It covers the **shell** only: layout, sign-in, the
landing page, empty Founder Studio and Lender Desk pages, and the About page. Feature pages (founder wizard,
results dashboard, lender file view, decisions) come in a later wave and plug into this shell.

> Educational planning tool. Not financial, legal or lending advice. Lenders make their own decisions.
> Every example number in the frontend is SYNTHETIC and labeled so on the page.

## 1. What it is, in one paragraph

A Next.js app (App Router, TypeScript) built as a **static export**: `npm run build` writes plain HTML, CSS, JS and
font files to `frontend/out/`. There is no Node server in production. The FastAPI app (`uwcopilot.web.app`, built
by the backend shell) serves `frontend/out/` for every non-API path and `/api/*` for data, from one origin, inside
one container on port 7860. The browser talks only to that origin. Sign-in is "Sign in with Hugging Face" (handled
by the server); locally a clearly labeled DEV LOGIN replaces it.

## 2. How to run it

Requires Node 22 and npm 10 (checked here: Node v22.22.0, npm 10.9.4).

| Command | What it does |
|---|---|
| `make web-install` | `npm ci`: installs the exact versions in `frontend/package-lock.json`. |
| `make web-build` | `next build`: static export to `frontend/out/`. |
| `make web-serve` | (backend target) FastAPI on http://127.0.0.1:7860 serving the API and `frontend/out/`. |
| `make web-dev` | FastAPI with `UWC_DEV_AUTH=1` on :8000 plus `next dev` on http://localhost:3000. `next dev` proxies `/api/*` to :8000, so cookies and CSRF behave as in production. Both listen on 127.0.0.1 only (`next dev --hostname 127.0.0.1`): the DEV LOGIN lets anyone pick any role, so it must not be reachable from the local network. |
| `make web-lint` | ESLint (`--max-warnings=0`), TypeScript (`next typegen && tsc --noEmit`), and the contrast check. |
| `make web-test` | `pytest tests/web` (backend) and `make web-lint`. |
| `make web-screenshots` | Playwright screenshots of `frontend/out/` (see section 8). Add `LIVE_URL=http://127.0.0.1:8000` to also drive a running dev server through DEV LOGIN. |

Inside `frontend/` the same steps are `npm run build`, `npm run lint`, `npm run typecheck`,
`npm run check:contrast`, `npm run dev`. `build`, `dev` and `typecheck` run Next.js through
`scripts/run-next.mjs`, which switches telemetry off (`NEXT_TELEMETRY_DISABLED=1`) and does not pass
secret-looking environment variables (names containing KEY, TOKEN, SECRET, ...; see the script) to Next.js:
Turbopack keeps a snapshot of its environment in `.next/cache`, and the data-API keys have no business there.

**Static preview without the API**: `cd frontend/out && python -m http.server 4173`, then open
http://127.0.0.1:4173/. Every `/api/*` request gets the file server's HTML 404, and the pages show a friendly
"server is not reachable" state instead of breaking (section 6).

## 3. The static export contract (shared with the backend)

- `next.config.ts` sets `output: "export"` and `trailingSlash: true`. Each route is written as
  `<route>/index.html` (`out/studio/index.html`, `out/lender/index.html`, ...) and links are rendered as
  `/studio/`. This works with any plain static file server, including `python -m http.server`.
- The backend's `src/uwcopilot/web/static.py` serves the exact file first, then `<path>/index.html`, then
  `<path>.html`, else `out/404.html` with status 404. It caches `/_next/static/*` for a year (file names carry
  content hashes) and sends `no-cache` for HTML.
- Next also writes small route payload files next to each page (`index.txt`, `__next.*.txt`). The client router
  fetches them during navigation, so the server must serve them as plain files (it does: they have extensions).
- **No dynamic route segments.** A static export cannot render `/lender/file/[id]` for ids it does not know at build
  time. Record pages must use query strings: `/lender/file/?id=...`.
- **CSP.** Each exported HTML file contains 2 inline bootstrap scripts (counted with a regex over `out/**/*.html`
  on 2026-10-12). The backend hashes them per file and sends `script-src 'self' 'sha256-...'`, so no
  `'unsafe-inline'` is needed for scripts. Styles use `style-src 'self' 'unsafe-inline'` (allowed by the spec).
  The live run in section 8 showed no CSP violations in the browser console.
- In development only, `next.config.ts` drops the export and adds a rewrite `/api/:path* ->
  $UWC_API_DEV_ORIGIN/api/:path*` (default http://127.0.0.1:8000), with trailing slashes off so `/api/me` is not
  redirected before the proxy sees it. `agentRules: false` stops `next dev` from writing `AGENTS.md` and
  `CLAUDE.md` into `frontend/` (agent instructions in this repo belong to the owners).

## 4. Every file

### Configuration
| File | Purpose |
|---|---|
| `package.json` | Exact versions (no `^`/`~`), scripts. `.npmrc` sets `save-exact=true` so new installs stay exact. |
| `package-lock.json` | Commit it. `npm ci` (Docker, CI) installs exactly this. All 589 entries resolve to registry.npmjs.org. |
| `next.config.ts` | Static export vs dev proxy (section 3). |
| `tsconfig.json` | Strict TypeScript, `noUncheckedIndexedAccess`, path alias `@/*`. |
| `eslint.config.mjs` | ESLint flat config: `eslint-config-next` core-web-vitals (React, hooks, some jsx-a11y rules) + TypeScript. |
| `postcss.config.mjs` | Runs Tailwind CSS v4. |
| `.gitignore` | `node_modules/`, `.next/`, `out/`, `next-env.d.ts`, `*.tsbuildinfo`. |

### `app/` (routes)
| File | Route | Purpose |
|---|---|---|
| `layout.tsx` | all | Fonts, `globals.css`, `<SessionProvider>`, skip link, header, `<main id="main">`, footer with the disclaimer. Page titles use the template `%s · Fundproof`. `robots: noindex` while it is a prototype. |
| `globals.css` | all | Design tokens (section 5) for light and dark, mapped to Tailwind color names. |
| `page.tsx` | `/` | Landing: hero, SYNTHETIC sample loan file, the two sides, how it works (4 steps), the evidence chip, honest limits. Static text only. |
| `signin/page.tsx` | `/signin/` | Renders `SignInPanel`. |
| `studio/page.tsx` | `/studio/` | Renders `StudioHome`. |
| `lender/page.tsx` | `/lender/` | Renders `LenderHome`. |
| `about/page.tsx` | `/about/` | What Fundproof is, disclaimer, data sources and licenses (mirrors the README table), planned Studio sources, privacy note, team, server status. |
| `not-found.tsx` | 404 | Exported as `out/404.html`. Uses the UNKNOWN chip. |
| `icon.svg` | favicon | The brand mark with fixed colors. |

Page files are server components so they can export `metadata`; the interactive part of each page is a client
component in `components/`.

### `lib/` (logic, no UI)
| File | Purpose |
|---|---|
| `config.ts` | `PRODUCT_NAME` (rename the product here and in `src/uwcopilot/web/settings.py`), `TAGLINE`, `DISCLAIMER`, `REPO_URL`, `OWNERS`, `HF_LOGIN_PATH`. The server's `/api/config` values win at runtime. |
| `api.ts` | Typed fetch wrapper. Same-origin `/api`, `credentials: "include"`, `X-CSRF-Token` on every non-GET (token from `GET /api/auth/csrf`, cached, refreshed once on a 403). Throws `ApiError` with `kind: "offline" \| "http"`, `status` and `retryAfterS` (the Retry-After of a 429). Types for health, config, me, csrf. |
| `auth.tsx` | `SessionProvider` + `useSession()`: loads `/api/config` and `/api/me` once; `status` is `loading`, `ready`, `offline` or `error` (section 6); exposes `error`, `refresh`, `signOut`, `devLogin`. An error keeps the last known user instead of showing "signed out". It never decides access: the server checks roles on every request. |
| `sign-in.ts` | `signInPageHref(next)` and `hfLoginHref(next)` (sign-in returns to the page it started from; `next` is checked like the server's `safe_next_path`), and the wording for each `/signin?error=` code. |
| `location.ts` | `useQueryParam()`, `useCurrentPath()`: the URL's query read after hydration (keeps the static HTML; `useSearchParams` would need a Suspense boundary in a static export). |
| `roles.ts` | Role names, labels, descriptions; `primaryRole()` for the header badge; `hasLenderRole()` for hints. |
| `evidence.ts` | The six evidence statuses and their plain-language meanings (one source of truth for the words). |
| `sample-file.ts` | The SYNTHETIC landing-page loan file. Values are typed for illustration; the 12-month deposit total is computed from the monthly figure so the "conflicting" example is consistent. |
| `options.ts` | Turns `/api/config` `metros` and `business_types` into labels without assuming a fixed shape. |
| `use-is-framed.ts` | `useIsFramed()`: true inside an iframe (the Hugging Face Space page). |
| `utils.ts` | `cn()`: join class names, later Tailwind classes win. |

### `components/`
| Folder / file | Purpose |
|---|---|
| `ui/*` | Vendored shadcn/ui-style components on Radix primitives (`radix-ui` package): Button, Card, Badge, Input, Label, Select, Tabs, Dialog, Sheet, Tooltip, Slider, Progress, Separator, Skeleton, Alert. Restyled with our tokens. Tabs, Slider, Progress and Tooltip are not used yet; they are ready for the feature pages. |
| `evidence/evidence-chip.tsx` | **EvidenceChip**: label + icon + color per status, plus screen-reader text "Evidence status:". |
| `evidence/figure.tsx` | **Figure**: a value with tabular numerals, its chip, an optional note and an optional "Source" button (`onShowSource(button)`, which opens the evidence drawer and gets the button so focus can return to it). Layouts `tile` and `row`. |
| `evidence/evidence-legend.tsx` | All six statuses with their meanings. |
| `landing/sample-file-card.tsx` | The SYNTHETIC sample file; "Source" opens a preview of the evidence drawer (a Sheet). The drawer gives focus back to the Source button that opened it when it closes (`onCloseAutoFocus`), and keeps its content while closing: copy this pattern for the real drawer. |
| `layout/site-header.tsx`, `mobile-nav.tsx`, `nav-links.ts` | Sticky header: brand, "Founder Studio", "Lender Desk", "About", sign-in state. Below 768 px a menu button opens a left Sheet. `aria-current="page"` marks the active link. |
| `layout/site-footer.tsx` | Disclaimer on every page, prototype / SYNTHETIC / no payments line, links. |
| `layout/synthetic-banner.tsx` | **SyntheticBanner** and **SyntheticTag**: hatched "specimen tape" marking SYNTHETIC data. |
| `layout/skip-link.tsx`, `brand-mark.tsx`, `page-intro.tsx` | Skip-to-content link, the logo, the page heading block. |
| `session/session-controls.tsx` | Header sign-in state: skeleton, "Offline preview", "Server busy" / "Server error", Sign in (remembering the page), or name + role badge + DEV tag + Sign out. Three layouts (desktop, menu, compact phone bar; the compact bar also shows DEV for a dev session). |
| `session/sign-in-panel.tsx` | The page's h1; why the last sign-in failed (`/signin?error=<code>`, in an `role="alert"` box; "Open in a new tab" for `state_mismatch`); the Hugging Face button (a link to `/api/auth/login?next=...`, opened with `target="_blank"` inside an iframe as the Hugging Face docs ask), disabled with an explanation when `/api/config` says `hf_oauth: false`; "open in a new tab" notice inside an iframe; DEV LOGIN form when `dev_auth` is true (it returns to `next` too). |
| `session/session-error-notice.tsx` | The server answered with an error: "Too many requests right now" with the Retry-After wait, or the HTTP status; says the sign-in is kept; "Try again". |
| `session/dev-login-form.tsx` | DEV LOGIN: username + role, POST `/api/auth/dev-login`. Dashed amber box labeled "Dev login: local development only". |
| `session/api-offline-notice.tsx` | The friendly offline state with a "Try again" button. |
| `session/sign-in-prompt.tsx`, `role-badge.tsx` | Sign-in call to action (an h2 card; its link carries `?next=` this page); role badge with icon and words. |
| `workspace/studio-home.tsx` | Founder Studio landing: states, empty plan list (the "Start a new plan" button is disabled and says why), "What will appear here", cities and business types from the server. |
| `workspace/lender-home.tsx` | Lender Desk landing: the "never decides" notice, states, empty queue, role hint, SYNTHETIC demo note, "What will appear here". |
| `workspace/never-decides-notice.tsx` | "The copilot never decides. A person does." |
| `workspace/planned-sections.tsx`, `workspace-skeleton.tsx` | "Not built yet" cards; loading placeholder. |
| `about/system-status.tsx` | One line from `GET /api/health` (status, version, commit when known). |

### `scripts/`
| File | Purpose |
|---|---|
| `run-next.mjs` | Runs the Next.js CLI without secret-looking environment variables (section 2). Used by `npm run build`, `dev` and `typecheck`. |
| `check-contrast.mjs` | Reads the tokens from `globals.css` and checks 21 text/background pairs per theme against WCAG AA (4.5:1). Exit code 1 on failure. |
| `screenshots.py` | Serves `out/` with `http.server`, screenshots every page at 1280 and 375 px in light and dark, fails on sideways overflow; with `--live-url` also signs in through DEV LOGIN on a running server and fails on any browser console error. |

## 5. Visual language

The product's idea is that every number carries its evidence status, so the design is a paper ledger:

- **Paper and ink**: warm off-white background (`--paper`), raised cards (`--paper-raised`), sunken bands
  (`--paper-sunken`), near-black ink text. Dark mode inverts to dark ink paper with light text. Dark mode follows
  the operating system (`prefers-color-scheme`); there is no toggle.
- **One accent**: ink blue (`--accent`) for links, primary buttons, eyebrows and the focus ring.
- **Six evidence statuses**, the same everywhere: VERIFIED (green, check badge icon), USER_STATED (amber, person
  icon), ASSUMPTION (amber, sliders icon, dashed border because it is editable), MODEL_ESTIMATE (amber, sigma icon),
  CONFLICTING (red, warning triangle), UNKNOWN (slate, question mark, dashed border). The amber family is told
  apart by label and icon, never by color alone. Chips are uppercase, letter-spaced "stamps".
- **SYNTHETIC tape**: a diagonal hatch with a dashed border (no extra hue).
- **Type**: Inter (UI) and Source Serif 4 (headings), both self-hosted from `@fontsource-variable` packages.
  No Google Fonts request at build time or run time. The CSS lists every script subset, but browsers download
  only the subsets a page uses (the dev server log showed only the latin files being fetched). Figures use tabular,
  lining numerals (`.figure-nums`).
- **Contrast**: `npm run check:contrast` (run 2026-10-12) reports all 21 pairs passing in both themes; the lowest
  ratio is 5.48:1 in light (`--ink-muted` on `--paper-sunken`) and 7.14:1 in dark (`--ink-muted` on
  `--paper-raised`). Disabled buttons (55% opacity) are exempt in WCAG and are not checked.

## 6. States every page handles

| State | When | What the user sees |
|---|---|---|
| loading | First request in flight; also the pre-rendered HTML | Skeletons where session-dependent content goes; static content is already there. |
| offline | `/api/config` unreachable, or it answers with non-JSON (a static server's HTML 404, a proxy error page while a Space wakes) | "The Fundproof server is not reachable right now" with Try again; header shows "Offline preview"; the Hugging Face button is disabled with a reason. |
| error | the API answered with a JSON error, for example 429 "rate limit exceeded" or a 500 | "Too many requests right now ... Try again in N seconds" (or the HTTP status) with Try again; the last known user stays signed in on screen; header shows "Server busy" / "Server error" when nobody was signed in. Checked with a server started with `UWC_RATE_LIMIT_USER_BURST=5`. |
| signed out | `/api/me` says `signed_in: false` (or 401) | Sign-in prompts on Studio and Lender Desk. |
| signed in | `/api/me` says `signed_in: true` | Name, role badge (DEV tag for dev sessions), Sign out, empty workspaces. |

Inside the Hugging Face iframe, some browsers block the session cookie even with `SameSite=None; Partitioned`. The
sign-in page then shows "Sign-in not sticking? Open in a new tab", linking to `<space_url>/signin/` from
`/api/config` (the direct `*.hf.space` address).

## 7. Accessibility, as built (not yet measured)

Skip link to `<main id="main">`; one visible focus ring (`:focus-visible`, 2 px accent outline); landmarks
(`header`, `nav aria-label="Main"`, `main`, `footer`, labeled `section`s); `aria-current="page"` on the active
link; every form control has a `<Label>`; errors use `role="alert"`; chips include screen-reader text; icons are
`aria-hidden`; Radix handles keyboard and focus for the menu, drawer and select; `prefers-reduced-motion` turns
animations off; the layout was checked at 375 px with no sideways scrolling (section 8).

**Lighthouse and axe scores: not measured yet.** The spec puts them in `results/web_metrics.json`, written by a
script in a later step. Nothing here claims a score.

## 8. Screenshots and what the runs checked

`docs/web/screenshots/` (30 PNG files) is produced by `frontend/scripts/screenshots.py`:

- `{landing,signin,studio,lender,about}-{1280,375}-{light,dark}.png`: the static export served by
  `python -m http.server`, so the API is unreachable and pages show the offline state.
- `menu-375-{light,dark}.png` (mobile menu open) and `drawer-1280-{light,dark}.png` (evidence drawer preview open).
- `signin-devserver-*`, `studio-devserver-founder-*`, `lender-devserver-senior-*` (1280 and 375, light): a real
  local server (`UWC_DEV_AUTH=1 uv run uvicorn uwcopilot.web.app:app --port 8765`), signed in through the DEV LOGIN
  form as SYNTHETIC demo users "demo-founder" (founder) and "demo-senior" (senior lender). The Studio city and
  business-type lists in these images come from that server's `/api/config`.

The last run on 2026-10-12 printed "no horizontal overflow and no browser console errors on the live server":
every page at 375 px fits the viewport, and dev login, sign-out, CSRF and the server's CSP produced no browser
console error. Separately, a `curl` run through `next dev` (port 3123, proxying to the dev server) completed
csrf -> dev-login -> `/api/me` returning `signed_in: true`.

## 9. Dependencies and versions

Pinned exactly; every version was released at least two weeks before 2026-10-12 (checked with `npm view <pkg> time`):

| Package | Version | Why |
|---|---|---|
| next | 16.3.6 | App Router, static export. (16.4.0 was 6 days old.) |
| react, react-dom, react-is | 19.3.0 | UI; `react-is` is a Recharts peer. |
| typescript | 5.9.3 | 7.0 is the native rewrite; Next's type checking and typescript-eslint use the 5.x JS API. |
| tailwindcss, @tailwindcss/postcss | 4.3.3 | Styling with CSS-variable tokens. postcss 8.5.28. |
| radix-ui | 1.6.7 | Accessible primitives (the unified package). |
| lucide-react | 1.48.0 | Icons. |
| class-variance-authority, clsx, tailwind-merge | 0.7.1, 2.1.1, 3.7.0 | Component variants and class merging (shadcn pattern). |
| @fontsource-variable/inter, /source-serif-4 | 5.3.0 | Self-hosted fonts. |
| recharts | 3.10.1 | Charts for the results dashboard. **Installed, not imported yet.** |
| maplibre-gl | 6.11.2 | Maps for site scoring. **Installed, not imported yet.** |
| eslint, eslint-config-next | 9.39.5, 16.3.6 | Linting. See the note below. |
| @types/node, @types/react, @types/react-dom | 22.20.4, 19.3.0, 19.3.0 | Types (Node 22 to match the runtime). |

Notes:
- **ESLint 9**: npm prints "eslint@9.39.5: This version is no longer supported". ESLint 10 was tried; three plugins
  inside eslint-config-next 16.3.6 (import, jsx-a11y, react) declare peer support only up to ESLint 9, and npm
  reported invalid peers. Move to ESLint 10 when eslint-config-next supports it.
- `npm ls` lists six optional WebAssembly fallback packages (for example `@img/sharp-wasm32`) as "extraneous"
  after `npm ci`; they are optional platform fallbacks recorded in the lockfile and are harmless.
- **Keep Recharts and MapLibre out of the landing bundle**: import them only inside feature components loaded with
  `next/dynamic` and `ssr: false` (MapLibre needs `window`). A grep of `out/_next/static/chunks/` on 2026-10-12
  found neither library in the current bundle.

## 10. Adding a feature page (for the next wave)

1. Add the endpoint types and a function to `lib/api.ts` (use `request`, so CSRF and offline handling come free).
2. Create `app/<route>/page.tsx` (server component with `metadata`) that renders a client component from
   `components/<area>/`.
3. Use `useSession()` for state; show `ApiOfflineNotice`, `SignInPrompt` and a skeleton like the existing pages.
4. Show every number with `<Figure>` and its real status from the API, and pass `onShowSource` to open the
   evidence drawer (build it on `Sheet`, and return focus to the Source button on close as
   `landing/sample-file-card.tsx` does). Wrap SYNTHETIC data in `SyntheticBanner`. Label simulated values
   "simulated, based on stated assumptions".
5. Record ids go in the query string (`/lender/file/?id=...`) because of the static export.
6. Run `make web-lint`, `make web-build` and `make web-screenshots`, and look at the 375 px images.

## 11. Honest limits and open items

- The web app does not yet do any planning or review: the workspaces are empty and the copy says so ("Not built
  yet" tags, disabled "Start a new plan").
- Hugging Face sign-in has not been tried end to end: it needs the Space (OAuth variables are injected there).
  Only DEV LOGIN was exercised.
- Lighthouse, axe and end-to-end tests of feature flows are not measured yet.
- Fonts include all script subsets in `out/` (about 2.0 MB for the whole export by `du -sh out`, most of it font
  files and JS chunks); only the used subsets are downloaded by browsers.
- CI (`.github/workflows/ci.yml`) does not run the frontend checks yet; the deploy workflow's verify job is
  planned to build the frontend.

## 12. Questions you may be asked (and short answers)

1. **Why a static export instead of a Next.js server?** One container, one origin, no Node runtime in production.
   FastAPI already serves the API; serving files from it keeps cookies, CSRF and CSP in one place.
2. **How does the frontend know who is signed in?** It calls `GET /api/me` once (`SessionProvider`). The cookie is
   HttpOnly and set by the server; the frontend never stores tokens. Every API call is checked by the server again.
3. **How is CSRF handled?** Double-submit: `GET /api/auth/csrf` sets the `uwc_csrf` cookie and returns the token;
   the client sends it as `X-CSRF-Token` on every non-GET request. The server compares them.
4. **What happens when the API is down?** `api.ts` classifies network errors and non-JSON answers as "offline";
   pages show a friendly notice with Try again instead of crashing. Static content still renders.
5. **Why can't you use `/lender/file/[id]`?** A static export must know every page at build time. Records use
   query strings.
6. **How do you avoid relying on color for evidence status?** Each chip has its own label and icon; assumptions
   and unknowns also have dashed borders. Screen readers hear "Evidence status: Verified".
7. **How did you check contrast?** A script reads the CSS tokens and computes WCAG ratios for every text and
   background pair the components use, in both themes; it fails below 4.5:1.
8. **Why the "open in a new tab" link on sign-in?** Spaces run inside an iframe on huggingface.co, where browsers
   may block third-party cookies; opening the direct `*.hf.space` address avoids that.
9. **Why are Recharts and MapLibre installed but unused?** They are the chosen libraries for the next wave; they will
   be loaded with `next/dynamic` so the landing page does not pay for them.
10. **Where is the product name?** `frontend/lib/config.ts` `PRODUCT_NAME` and the server's settings; at runtime the
    server's `/api/config` value is used.
