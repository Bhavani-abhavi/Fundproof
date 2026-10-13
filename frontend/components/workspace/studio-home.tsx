"use client";

import { Building2, FileCheck2, LineChart, ListChecks, MapPinned, Plus, Route } from "lucide-react";

import { PageIntro } from "@/components/layout/page-intro";
import { ApiOfflineNotice } from "@/components/session/api-offline-notice";
import { RoleBadge } from "@/components/session/role-badge";
import { SignInPrompt } from "@/components/session/sign-in-prompt";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { PlannedSections, type PlannedSection } from "@/components/workspace/planned-sections";
import { WorkspaceSkeleton } from "@/components/workspace/workspace-skeleton";
import { useSession } from "@/lib/auth";
import { toOptionLabels, type OptionLabel } from "@/lib/options";
import { primaryRole } from "@/lib/roles";

const STUDIO_SECTIONS: PlannedSection[] = [
  {
    icon: MapPinned,
    title: "City and business type",
    body: "Choose a supported city and a business type. Cities with thin public data run in limited mode.",
  },
  {
    icon: Building2,
    title: "Candidate sites",
    body: "Up to three addresses, each scored with its reasons and a data-coverage score.",
  },
  {
    icon: LineChart,
    title: "Revenue and cash ranges",
    body: "Low, middle and high cases, simulated, based on stated assumptions you can edit.",
  },
  {
    icon: Route,
    title: "Path to approval",
    body: "The changes that would make the plan easier for a lender to check, with what each one needs.",
  },
  {
    icon: FileCheck2,
    title: "Lender-ready package",
    body: "A PDF and data file with every figure's evidence status, shared only when you choose to.",
  },
  {
    icon: ListChecks,
    title: "Evidence checklist",
    body: "Which documents would turn stated or assumed figures into verified ones.",
  },
];

function OptionList({ title, items, emptyText }: { title: string; items: OptionLabel[]; emptyText: string }) {
  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      {items.length === 0 ? (
        <p className="text-sm text-muted-foreground">{emptyText}</p>
      ) : (
        <ul className="flex min-w-0 flex-wrap gap-2">
          {items.map((item) => (
            <li key={item.key}>
              <Badge variant="outline" className="inline-block max-w-full whitespace-normal rounded-md py-1 text-left leading-snug">
                {item.label}
                {item.detail ? <span className="whitespace-nowrap text-muted-foreground"> · {item.detail}</span> : null}
              </Badge>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Founder Studio landing: sign-in state, an empty project list, what will appear here, and the server's city list. */
export function StudioHome() {
  const { status, me, config } = useSession();
  const role = primaryRole(me.roles);
  const metros = toOptionLabels(config?.metros);
  const businessTypes = toOptionLabels(config?.business_types);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <PageIntro
        eyebrow="Founder Studio"
        title="Plan a location. Prepare a file a lender can check."
        description={
          <p>
            Founder Studio turns a business idea into a plan where every number says where it came from. It prepares and
            explains; it does not promise approval or pick a lender for you.
          </p>
        }
      />

      {status === "loading" ? <WorkspaceSkeleton /> : null}

      {status === "offline" ? <ApiOfflineNotice /> : null}

      {status === "ready" && !me.signed_in ? (
        <SignInPrompt
          area="Founder Studio"
          reason="Your plans are private to your account. Sign in with Hugging Face to start one."
        />
      ) : null}

      {status === "ready" && me.signed_in ? (
        <Card>
          <CardHeader className="gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="flex flex-col gap-1.5">
              <CardTitle>Your plans</CardTitle>
              <CardDescription>
                Signed in as {me.display_name || me.username}. You have no plans yet.
              </CardDescription>
            </div>
            {role ? <RoleBadge role={role} className="self-start" /> : null}
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col items-start gap-3 rounded-lg border border-dashed border-rule-strong p-5 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-muted-foreground">
                The plan wizard is not built yet. When it is, your plans will be listed here.
              </p>
              <Button type="button" disabled aria-describedby="new-plan-note">
                <Plus aria-hidden="true" />
                Start a new plan
              </Button>
            </div>
            <p id="new-plan-note" className="sr-only">
              Disabled: the plan wizard is not built yet.
            </p>
          </CardContent>
        </Card>
      ) : null}

      <PlannedSections title="What will appear here" sections={STUDIO_SECTIONS} />

      {status === "ready" ? (
        <Card>
          <CardHeader>
            <CardTitle>Cities and business types</CardTitle>
            <CardDescription>As reported by the server right now.</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-6 sm:grid-cols-2">
            <OptionList
              title="Cities"
              items={metros}
              emptyText={config?.metros_note || "Not available yet. Cities appear once their public data is loaded."}
            />
            <OptionList
              title="Business types"
              items={businessTypes}
              emptyText="Not available yet. Business types appear once their profiles are loaded."
            />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
