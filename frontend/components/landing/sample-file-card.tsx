"use client";

import { FileText } from "lucide-react";
import { useState } from "react";

import { EvidenceChip } from "@/components/evidence/evidence-chip";
import { Figure } from "@/components/evidence/figure";
import { SyntheticBanner, SyntheticTag } from "@/components/layout/synthetic-banner";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { EVIDENCE_STATUS_INFO } from "@/lib/evidence";
import { SAMPLE_FIGURES, SAMPLE_FILE_TITLE, SAMPLE_LOAN_REQUEST, type SampleFigure } from "@/lib/sample-file";

/**
 * The landing page's SYNTHETIC sample loan file: six figures, one per evidence status, as ledger rows.
 * "Source" opens a preview of the evidence drawer for that figure (the real drawer comes with the file view).
 */
export function SampleFileCard() {
  const [selected, setSelected] = useState<SampleFigure | null>(null);

  return (
    <section
      aria-labelledby="sample-file-title"
      className="relative rounded-xl border border-border bg-card shadow-[0_1px_0_0_var(--rule),0_18px_40px_-24px_rgb(20_24_32/0.35)]"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3 sm:px-5">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.1em] text-muted-foreground">
          <FileText className="size-4" aria-hidden="true" />
          Sample loan file
        </div>
        <SyntheticTag />
      </div>

      <div className="px-4 pb-1 pt-4 sm:px-5">
        <h2 id="sample-file-title" className="font-serif text-xl font-semibold leading-snug text-foreground">
          {SAMPLE_FILE_TITLE}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">{SAMPLE_LOAN_REQUEST}</p>
      </div>

      <div className="divide-y divide-border px-4 sm:px-5">
        {SAMPLE_FIGURES.map((figure) => (
          <Figure
            key={figure.id}
            layout="row"
            label={figure.label}
            value={figure.value}
            status={figure.status}
            note={figure.note}
            onShowSource={() => setSelected(figure)}
          />
        ))}
      </div>

      <div className="px-4 pb-4 pt-1 sm:px-5 sm:pb-5">
        <SyntheticBanner>Illustrative values typed for this page. Not a real business and not model output.</SyntheticBanner>
      </div>

      <Sheet open={selected !== null} onOpenChange={(open) => (open ? null : setSelected(null))}>
        <SheetContent side="right">
          {selected ? (
            <>
              <SheetHeader>
                <p className="text-xs font-semibold uppercase tracking-[0.12em] text-accent">Evidence drawer (preview)</p>
                <SheetTitle>{selected.label}</SheetTitle>
                <SheetDescription className="figure-nums text-2xl font-semibold text-foreground">{selected.value}</SheetDescription>
              </SheetHeader>
              <div className="flex flex-col gap-2">
                <EvidenceChip status={selected.status} size="md" className="self-start" />
                <p className="text-sm text-muted-foreground">{EVIDENCE_STATUS_INFO[selected.status].description}</p>
              </div>
              <div className="flex flex-col gap-1.5 rounded-lg border border-border bg-paper p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.1em] text-muted-foreground">Source</p>
                <p className="text-sm leading-relaxed text-foreground">{selected.source}</p>
              </div>
              <p className="text-sm leading-relaxed text-muted-foreground">
                In the file view, this drawer is planned to show the document page or the public-data request behind the
                figure, with its date.
              </p>
              <SyntheticBanner />
            </>
          ) : null}
        </SheetContent>
      </Sheet>
    </section>
  );
}
