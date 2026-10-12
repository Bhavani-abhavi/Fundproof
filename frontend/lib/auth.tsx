"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import {
  api,
  ApiError,
  clearCsrfToken,
  isAbortError,
  isOfflineError,
  SIGNED_OUT,
  type ConfigResponse,
  type MeResponse,
  type Role,
} from "@/lib/api";
import { DISCLAIMER, PRODUCT_NAME } from "@/lib/config";

/**
 * Session context: who is signed in (GET /api/me) and the server's public settings (GET /api/config).
 *
 * status:
 *   "loading" - first request still in flight (also what the static HTML is pre-rendered with)
 *   "ready"   - the API answered; `me` and `config` are filled in
 *   "offline" - the API could not be reached (static preview, server starting, network down)
 *
 * Every page reads it with `useSession()`. Nothing here decides access: the server checks roles on every request.
 */

export type SessionStatus = "loading" | "ready" | "offline";

export interface SessionValue {
  status: SessionStatus;
  me: MeResponse;
  config: ConfigResponse | null;
  /** Product name and disclaimer: the server's values when known, else the built-in fallbacks. */
  productName: string;
  disclaimer: string;
  /** The last non-offline error (for example a 500 from /api/me), or null. */
  error: ApiError | null;
  refresh: () => Promise<void>;
  signOut: () => Promise<void>;
  devLogin: (username: string, role: Role) => Promise<void>;
}

const SessionContext = createContext<SessionValue | null>(null);

interface LoadResult {
  status: SessionStatus;
  me: MeResponse;
  config: ConfigResponse | null;
  error: ApiError | null;
}

/** Ask the server for its config and the current user. Never throws (except on abort). */
async function loadSession(signal?: AbortSignal): Promise<LoadResult> {
  let config: ConfigResponse;
  try {
    config = await api.config(signal);
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    // Unreachable, or answering with errors: either way the live features are off.
    const apiError = error instanceof ApiError && !isOfflineError(error) ? error : null;
    return { status: "offline", me: SIGNED_OUT, config: null, error: apiError };
  }

  try {
    const me = await api.me(signal);
    return { status: "ready", me, config, error: null };
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    // 401 simply means "not signed in"; anything else is kept so a page can mention it.
    const apiError = error instanceof ApiError ? error : null;
    const keepError = apiError !== null && apiError.status !== 401 ? apiError : null;
    return { status: "ready", me: SIGNED_OUT, config, error: keepError };
  }
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LoadResult>({
    status: "loading",
    me: SIGNED_OUT,
    config: null,
    error: null,
  });

  const refresh = useCallback(async () => {
    const result = await loadSession();
    setState(result);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    loadSession(controller.signal)
      .then((result) => setState(result))
      .catch(() => {
        // Aborted because the provider unmounted: nothing to update.
      });
    return () => controller.abort();
  }, []);

  const signOut = useCallback(async () => {
    await api.logout();
    clearCsrfToken();
    await refresh();
  }, [refresh]);

  const devLogin = useCallback(
    async (username: string, role: Role) => {
      await api.devLogin(username, role);
      clearCsrfToken();
      await refresh();
    },
    [refresh],
  );

  const value = useMemo<SessionValue>(
    () => ({
      ...state,
      productName: state.config?.product_name || PRODUCT_NAME,
      disclaimer: state.config?.disclaimer || DISCLAIMER,
      refresh,
      signOut,
      devLogin,
    }),
    [state, refresh, signOut, devLogin],
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

/** Read the session. Must be used inside <SessionProvider> (the root layout provides it). */
export function useSession(): SessionValue {
  const value = useContext(SessionContext);
  if (value === null) {
    throw new Error("useSession() must be used inside <SessionProvider>.");
  }
  return value;
}
