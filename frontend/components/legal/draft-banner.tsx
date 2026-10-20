import { FilePenLine } from "lucide-react";

import { cn } from "@/lib/utils";

/**
 * The banner at the top of the Privacy and Terms pages until the owners (Bhavani Adula and Rahul Vajja) approve them.
 * The pages describe what the v0.1 code does (docs/studio/privacy.md); commitments that only the owners can make
 * (for example that data is never sold) are not made here, and the pages say so where it matters.
 * Remove the banner, and the "Prototype: please use a made-up plan" notice, when the owners approve both pages.
 */
export function DraftBanner({ className }: { className?: string }) {
  return (
    <aside
      aria-labelledby="draft-banner-title"
      className={cn("flex items-start gap-3 rounded-xl border-2 border-dashed border-stated-line bg-stated-soft p-4 sm:p-5", className)}
    >
      <FilePenLine className="mt-0.5 size-5 shrink-0 text-stated" aria-hidden="true" />
      <div className="flex flex-col gap-1">
        <h2 id="draft-banner-title" className="text-base font-semibold text-foreground">
          Draft - pending the owners&apos; approval
        </h2>
        <p className="text-sm leading-relaxed text-foreground/85">
          This page describes what the prototype&apos;s code does today. The project&apos;s owners have not approved it yet,
          so it makes no promise beyond that. Until they do, please try the app with made-up plans only.
        </p>
      </div>
    </aside>
  );
}
