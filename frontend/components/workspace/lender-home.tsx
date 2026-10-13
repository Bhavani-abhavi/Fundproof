"use client";

import { Activity, BookOpenCheck, FileSearch, GitCompareArrows, Inbox, ListOrdered, Rows3 } from "lucide-react";

import { PageIntro } from "@/components/layout/page-intro";
import { SyntheticBanner } from "@/components/layout/synthetic-banner";
import { ApiOfflineNotice } from "@/components/session/api-offline-notice";
import { RoleBadge } from "@/components/session/role-badge";
import { SessionErrorNotice } from "@/components/session/session-error-notice";
import { SignInPrompt } from "@/components/session/sign-in-prompt";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { NeverDecidesNotice } from "@/components/workspace/never-decides-notice";
import { PlannedSections, type PlannedSection } from "@/components/workspace/planned-sections";
import { WorkspaceSkeleton } from "@/components/workspace/workspace-skeleton";
import { useSession } from "@/lib/auth";
import { hasLenderRole, primaryRole, ROLE_DESCRIPTIONS } from "@/lib/roles";

const LENDER_SECTIONS: PlannedSection[] = [
  {
    icon: Inbox,
    title: "Review queue",
    body: "Packages founders share with you, plus SYNTHETIC demo files marked DEMO, filterable by city and business type.",
  },
  {
    icon: FileSearch,
    title: "Documents with citations",
    body: "Each extracted figure links to the page it came from, with the passage highlighted.",
  },
  {
    icon: Rows3,
    title: "Ledger by evidence status",
    body: "Every figure grouped as verified, stated, assumed, estimated, conflicting or unknown.",
  },
  {
    icon: GitCompareArrows,
    title: "Contradictions",
    body: "Where documents disagree, side by side, separated into explainable and unsupported differences.",
  },
  {
    icon: Activity,
    title: "Stress tests",
    body: "Sliders for revenue, rate and claims, with the baseline and the stressed result next to each other.",
  },
  {
    icon: ListOrdered,
    title: "Next best evidence",
    body: "The document worth requesting next, ranked by whether it could change the outcome.",
  },
  {
    icon: BookOpenCheck,
    title: "Memo and decision record",
    body: "A draft memo with citations, your decision and its reason, exported as a record.",
  },
];

/** Lender Desk landing: the "never decides" notice, sign-in state, an empty queue, and what will appear here. */
export function LenderHome() {
  const { status, me } = useSession();
  const role = primaryRole(me.roles);
  const isLender = hasLenderRole(me.roles);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <PageIntro
        eyebrow="Lender Desk"
        title="Review the evidence. Make the call yourself."
        description={<p>Read a small-business loan file with the source and status of every figure beside it.</p>}
      />

      <NeverDecidesNotice />

      {status === "loading" ? <WorkspaceSkeleton /> : null}

      {status === "offline" ? <ApiOfflineNotice /> : null}

      {status === "error" ? <SessionErrorNotice /> : null}

      {status === "ready" && !me.signed_in ? (
        <SignInPrompt
          area="Lender Desk"
          reason="Any signed-in user can open the SYNTHETIC demo files once they are added. Decisions on real shared files need a lender role granted by the project owners."
        />
      ) : null}

      {(status === "ready" || status === "error") && me.signed_in ? (
        <Card>
          <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex flex-col gap-1.5">
              <CardTitle as="h2">Review queue</CardTitle>
              <CardDescription>Signed in as {me.display_name || me.username}. Your queue is empty.</CardDescription>
            </div>
            {role ? <RoleBadge role={role} className="self-start" /> : null}
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2 rounded-lg border border-dashed border-rule-strong p-5">
              <p className="text-sm text-foreground">
                Files will appear here when a founder shares a package with you, and when the SYNTHETIC demo files are
                added.
              </p>
              <p className="text-sm text-muted-foreground">
                {isLender && role
                  ? `Your role: ${ROLE_DESCRIPTIONS[role]}`
                  : "You are signed in as a founder. You will be able to open the SYNTHETIC demo files; decisions on real shared files need a lender role that the project owners grant."}
              </p>
            </div>
            <SyntheticBanner>Demo files in this desk are computer-generated loan packets, marked DEMO. They describe no real applicant.</SyntheticBanner>
          </CardContent>
        </Card>
      ) : null}

      <PlannedSections title="What will appear here" sections={LENDER_SECTIONS} />
    </div>
  );
}
