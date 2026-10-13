/**
 * Typed client for the Fundproof FastAPI backend (`uwcopilot.web.app`).
 *
 * - Same origin: every call goes to `/api/...` on the host that served the page (no CORS).
 * - `credentials: "include"` so the session cookie travels with each request.
 * - CSRF double-submit: before any state-changing request (POST/PUT/PATCH/DELETE) the client fetches a token
 *   from GET /api/auth/csrf (the server also sets the `uwc_csrf` cookie) and sends it in the `X-CSRF-Token` header.
 * - Errors are thrown as `ApiError`. `kind === "offline"` means the API could not be reached at all, which is
 *   normal in a static preview (`python -m http.server` answers /api/* with an HTML 404 page). Pages use it to show
 *   a friendly offline state instead of crashing.
 */

import { PRODUCT_NAME } from "@/lib/config";

/** Roles the server can grant. Every signed-in user is a founder; lender roles come from the owners' allowlist. */
export type Role = "founder" | "lender_reviewer" | "lender_senior" | "admin";

/** How the current session was established. */
export type AuthMode = "hf_oauth" | "dev" | "none";

/** GET /api/health */
export interface HealthResponse {
  status: string;
  version: string;
  git_sha?: string | null;
}

/**
 * GET /api/config. `metros` and `business_types` are empty (with `metros_note` saying why) until the Studio
 * loaders are available. Optional fields are read defensively because the server shell is still evolving.
 */
export interface ConfigResponse {
  product_name: string;
  disclaimer: string;
  dev_auth: boolean;
  sides: string[];
  data_labels?: unknown;
  metros: unknown[];
  business_types: unknown[];
  metros_note?: string | null;
  roles?: string[];
  /** True when Hugging Face sign-in is configured (inside a Space with `hf_oauth: true`). */
  hf_oauth?: boolean;
  is_space?: boolean;
  /** The Space's direct URL (outside the huggingface.co iframe), when running in a Space. */
  space_url?: string | null;
  payments_enabled?: boolean;
}

/** GET /api/me */
export interface MeResponse {
  signed_in: boolean;
  username: string | null;
  display_name: string | null;
  /** The user's own stable account id (Hugging Face "sub"); owners pin role allowlist entries to it. */
  subject?: string | null;
  roles: string[];
  auth_mode: AuthMode;
}

/** GET /api/auth/csrf */
export interface CsrfResponse {
  token: string;
}

/** The session shape used whenever nobody is signed in (or the API is unreachable). */
export const SIGNED_OUT: MeResponse = {
  signed_in: false,
  username: null,
  display_name: null,
  roles: [],
  auth_mode: "none",
};

export type ApiErrorKind = "offline" | "http";

/**
 * An API failure. `status` is 0 when the request never reached a server. `retryAfterS` is the server's
 * Retry-After header in seconds (sent with 429 "rate limit exceeded"), or null.
 */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number;
  readonly detail: unknown;
  readonly retryAfterS: number | null;

  constructor(kind: ApiErrorKind, status: number, message: string, detail: unknown = null, retryAfterS: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.detail = detail;
    this.retryAfterS = retryAfterS;
  }
}

/** Retry-After as seconds: either a number of seconds or an HTTP date. Null when absent or unreadable. */
function parseRetryAfter(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  if (/^\d+$/.test(trimmed)) {
    return Number(trimmed);
  }
  const date = Date.parse(trimmed);
  if (Number.isNaN(date)) {
    return null;
  }
  return Math.max(0, Math.ceil((date - Date.now()) / 1000));
}

/** True when a request was cancelled with an AbortController (for example because a page unmounted). */
export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

/** True when the error means "the API is not there" (network failure, static preview, server starting). */
export function isOfflineError(error: unknown): boolean {
  return error instanceof ApiError && error.kind === "offline";
}

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

interface RequestOptions {
  method?: HttpMethod;
  /** Sent as a JSON body. */
  json?: unknown;
  signal?: AbortSignal;
  /** "json": the response must be JSON. "any": JSON is parsed if present, an empty body is fine. */
  expect?: "json" | "any";
}

const OFFLINE_MESSAGE = `The ${PRODUCT_NAME} server could not be reached.`;

let cachedCsrfToken: string | null = null;

/** Forget the CSRF token (call after sign-in or sign-out, when the server may rotate it). */
export function clearCsrfToken(): void {
  cachedCsrfToken = null;
}

async function getCsrfToken(forceRefresh = false): Promise<string> {
  if (cachedCsrfToken !== null && !forceRefresh) {
    return cachedCsrfToken;
  }
  const response = await request<CsrfResponse>("/api/auth/csrf");
  cachedCsrfToken = response.token;
  return cachedCsrfToken;
}

/** Pull a readable message out of a FastAPI error body ({"detail": "..."}), if there is one. */
function messageFromBody(body: unknown): string | null {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
  }
  return null;
}

async function request<T>(path: string, options: RequestOptions = {}, isRetry = false): Promise<T> {
  const method = options.method ?? "GET";
  const expect = options.expect ?? "json";
  const headers: Record<string, string> = { Accept: "application/json" };
  let body: string | undefined;

  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.json);
  }
  if (method !== "GET") {
    headers["X-CSRF-Token"] = await getCsrfToken(isRetry);
  }

  let response: Response;
  try {
    response = await fetch(path, {
      method,
      headers,
      body,
      credentials: "include",
      cache: "no-store",
      signal: options.signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    throw new ApiError("offline", 0, OFFLINE_MESSAGE);
  }

  const contentType = response.headers.get("content-type") ?? "";
  const isJson = contentType.includes("application/json");

  if (!isJson) {
    // An HTML or empty answer to a JSON endpoint means we are not talking to the API
    // (static preview, a proxy error page while the Space starts, or a SPA fallback page).
    if (expect === "json" || !response.ok) {
      throw new ApiError("offline", response.status, OFFLINE_MESSAGE);
    }
    return undefined as T;
  }

  const payload: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    // A rotated or missing CSRF token gives 403: fetch a fresh token and try once more.
    if (response.status === 403 && method !== "GET" && !isRetry) {
      return request<T>(path, options, true);
    }
    const message = messageFromBody(payload) ?? `Request failed with status ${response.status}.`;
    const retryAfterS = parseRetryAfter(response.headers.get("retry-after"));
    throw new ApiError("http", response.status, message, payload, retryAfterS);
  }

  return payload as T;
}

/** The endpoints the app shell uses. Feature endpoints (studio, lender) are added here as they are built. */
export const api = {
  health: (signal?: AbortSignal) => request<HealthResponse>("/api/health", { signal }),
  config: (signal?: AbortSignal) => request<ConfigResponse>("/api/config", { signal }),
  me: (signal?: AbortSignal) => request<MeResponse>("/api/me", { signal }),
  /** Local development only: the server refuses this inside a Hugging Face Space. */
  devLogin: (username: string, role: Role) =>
    request<unknown>("/api/auth/dev-login", { method: "POST", json: { username, role }, expect: "any" }),
  logout: () => request<unknown>("/api/auth/logout", { method: "POST", expect: "any" }),
};
