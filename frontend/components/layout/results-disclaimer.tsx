"use client";

import { Info } from "lucide-react";

import { useSession } from "@/lib/auth";
import { cn } from "@/lib/utils";

/**
 * The disclaimer at the top of every results page (Studio addendum rule C), in addition to the footer that shows it
 * on every page. The server's wording (GET /api/config) wins over the built-in copy; `text` wins over both (a shared
 * plan shows the disclaimer that came with it).
 */
export function ResultsDisclaimer({ className, text }: { className?: string; text?: string }) {
  const session = useSession();
  const disclaimer = text || session.disclaimer;
  return (
    <p
      role="note"
      className={cn(
        "flex items-start gap-2 rounded-lg border border-accent/30 bg-accent-soft px-3 py-2 text-sm font-medium leading-relaxed text-ink",
        className,
      )}
    >
      <Info className="mt-0.5 size-4 shrink-0 text-accent" aria-hidden="true" />
      <span>{disclaimer}</span>
    </p>
  );
}
