/**
 * Share links for one plan (Studio S8).
 *
 * - The plan's owner creates a link (POST /api/studio/projects/{id}/share {expires_in_days}) and can revoke every link
 *   of that plan at once (DELETE /api/studio/projects/{id}/share). Both go through `request`, so they carry the
 *   session cookie and the CSRF header like every other change.
 * - Anyone who has the link reads the plan without signing in (GET /api/share/{token}). The server decides what the
 *   answer holds: the plan's figures, city, business type, site names and map positions, the disclaimer and a notice.
 *   Never the founder's idea text, street addresses or anything about the owner's account (data minimization).
 * - An invalid, tampered, expired or revoked link, or one to a deleted plan, answers 404 with one identical body,
 *   so a visitor cannot tell the cases apart. The page shows one message for all of them.
 * - The page for a link is /share?t=<token>: a query parameter, because the static export cannot pre-render dynamic
 *   path segments.
 */

import { request } from "@/lib/api";
import type { ProjectMoney } from "@/lib/studio-types";

/** The expiry the server accepts, in days (inclusive), and its default. */
export const SHARE_DAYS_MIN = 1;
export const SHARE_DAYS_MAX = 30;
export const SHARE_DAYS_DEFAULT = 7;

/** The expiry choices offered on the plan page (all inside the server's 1 to 30 days). */
export const SHARE_DAY_CHOICES = [1, 3, SHARE_DAYS_DEFAULT, 14, SHARE_DAYS_MAX] as const;

/**
 * The share answer carries no data label, so the generated demo city is recognized by its id (it mirrors
 * SYNTHETIC_METRO_ID in src/uwcopilot/agent/synthetic_startup.py) to put the SYNTHETIC banner on the page.
 */
export const SYNTHETIC_METRO_ID = "synthetic_metro";

/** POST /api/studio/projects/{id}/share. `url` is a path on this site ("/share?t=..."); `expires_at` is ISO 8601 UTC. */
export interface ShareLinkCreated {
  url: string;
  token: string;
  expires_at: string;
}

/** DELETE /api/studio/projects/{id}/share: how many links of the plan were revoked. */
export interface ShareLinksRevoked {
  revoked: number;
}

/** A site as a share link shows it: its name and map position, never its street address. */
export interface SharedSite {
  name: string;
  lat: number;
  lon: number;
}

/** GET /api/share/{token}: the read-only plan (SharedPlanOut on the server). */
export interface SharedPlan {
  plan_name: string | null;
  business_type: string;
  metro_id: string;
  metro_name: string;
  /** The plan's money figures with who wrote each one (the same shape as the plan page's). */
  money: ProjectMoney;
  opening_date: string;
  sites: SharedSite[];
  expires_at: string;
  disclaimer: string;
  /** The server's plain-language note about what a shared plan is. */
  notice: string;
}

function sharePath(projectId: string): string {
  return `/api/studio/projects/${encodeURIComponent(projectId)}/share`;
}

/** The full address of a share link, to show and copy ("https://<this site>/share?t=..."). Browser only. */
export function absoluteShareUrl(path: string): string {
  return new URL(path, window.location.origin).toString();
}

export const shareApi = {
  create: (projectId: string, expiresInDays: number) =>
    request<ShareLinkCreated>(sharePath(projectId), { method: "POST", json: { expires_in_days: expiresInDays } }),

  revokeAll: (projectId: string) => request<ShareLinksRevoked>(sharePath(projectId), { method: "DELETE" }),

  read: (token: string, signal?: AbortSignal) => request<SharedPlan>(`/api/share/${encodeURIComponent(token)}`, { signal }),
};
