"""The web app: one FastAPI process that serves the JSON API under /api and the Next.js static export.

Modules:
- settings.py: every setting read from the environment, checked once at startup (refuses unsafe combinations);
- app.py: create_app(settings) and the module-level `app` that uvicorn runs (uwcopilot.web.app:app);
- auth.py: Hugging Face sign-in (OpenID Connect + PKCE), dev login, signed session cookies, roles, require_role;
- security.py: security headers (CSP), body size limit, rate limiter, CSRF check;
- access_log.py: one redacted log line per request;
- static.py: serves frontend/out with correct MIME types, cache headers and the route-to-file fallback;
- crypto.py: Fernet encryption for everything a user stores;
- store.py: the Store interface with a local-folder store and a private Hugging Face dataset store;
- routers/: the /api endpoints (system.py: health and config; auth.py: me, csrf, login, logout, dev login).
"""
