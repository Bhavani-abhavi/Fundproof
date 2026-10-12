"use client";

import { LogIn, LogOut, WifiOff } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { RoleBadge } from "@/components/session/role-badge";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/lib/auth";
import { primaryRole } from "@/lib/roles";
import { cn } from "@/lib/utils";

/**
 * Sign-in state for the header and the mobile menu: a skeleton while loading, an "Offline preview" badge when the
 * API is unreachable, a Sign in button when signed out, and name + role badge + Sign out when signed in.
 *
 * layout: "inline" (desktop header), "stacked" (mobile menu), "compact" (phone header bar: status only; the full
 * controls are in the menu).
 */
export function SessionControls({
  layout = "inline",
  onNavigate,
}: {
  layout?: "inline" | "stacked" | "compact";
  onNavigate?: () => void;
}) {
  const { status, me, signOut } = useSession();
  const [signingOut, setSigningOut] = useState(false);
  const [signOutFailed, setSignOutFailed] = useState(false);
  const stacked = layout === "stacked";

  async function handleSignOut() {
    setSigningOut(true);
    setSignOutFailed(false);
    try {
      await signOut();
    } catch {
      setSignOutFailed(true);
    } finally {
      setSigningOut(false);
    }
  }

  const compact = layout === "compact";

  if (status === "loading") {
    const skeletonWidth = { inline: "w-24", stacked: "w-full", compact: "w-16" }[layout];
    return <Skeleton className={cn("h-8", skeletonWidth)} aria-label="Checking sign-in" />;
  }

  if (status === "offline") {
    return (
      <Badge variant="muted" className="h-8 gap-1.5 rounded-md px-2.5">
        <WifiOff aria-hidden="true" />
        {compact ? "Offline" : "Offline preview"}
      </Badge>
    );
  }

  if (!me.signed_in) {
    return (
      <Button asChild size="sm" variant={stacked ? "default" : "outline"} className={cn(stacked && "w-full")}>
        <Link href="/signin" onClick={onNavigate}>
          <LogIn aria-hidden="true" />
          Sign in
        </Link>
      </Button>
    );
  }

  const role = primaryRole(me.roles);
  const name = me.display_name || me.username || "Signed in";

  if (compact) {
    return role ? <RoleBadge role={role} /> : null;
  }

  return (
    <div className={cn("flex items-center gap-2", stacked && "flex-col items-stretch gap-3")}>
      <div className={cn("flex min-w-0 items-center gap-2", stacked && "justify-between")}>
        <span className="max-w-[10rem] truncate text-sm font-medium" title={me.username ?? undefined}>
          {name}
        </span>
        {role ? <RoleBadge role={role} /> : null}
        {me.auth_mode === "dev" ? (
          <Badge variant="outline" className="border-dashed font-semibold tracking-wide">
            DEV
          </Badge>
        ) : null}
      </div>
      <Button
        type="button"
        size="sm"
        variant={stacked ? "outline" : "ghost"}
        onClick={handleSignOut}
        disabled={signingOut}
      >
        <LogOut aria-hidden="true" />
        {signingOut ? "Signing out..." : "Sign out"}
      </Button>
      {signOutFailed ? (
        <p role="alert" className="text-xs text-conflicting">
          Sign-out failed. Try again.
        </p>
      ) : null}
    </div>
  );
}
