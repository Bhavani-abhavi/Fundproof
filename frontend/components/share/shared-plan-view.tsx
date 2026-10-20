"use client";

import { ArrowRight, Clock, Eye, LinkIcon } from "lucide-react";
import Link from "next/link";

import { Figure } from "@/components/evidence/figure";
import { PageIntro } from "@/components/layout/page-intro";
import { ResultsDisclaimer } from "@/components/layout/results-disclaimer";
import { SyntheticBanner } from "@/components/layout/synthetic-banner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ProblemNotice } from "@/components/workspace/problem-notice";
import { WorkspaceSkeleton } from "@/components/workspace/workspace-skeleton";
import { formatDate, formatDateTime, formatNumber, formatPercent, formatUsd, isNumber, NOT_AVAILABLE } from "@/lib/format";
import { useIsHydrated, useQueryParam } from "@/lib/location";
import { shareApi, SYNTHETIC_METRO_ID, type SharedPlan } from "@/lib/share-api";
import type { ProjectMoney } from "@/lib/studio-types";
import { useApiResource } from "@/lib/use-api";

/**
 * /share?t=<token>: a plan its owner shared, read-only, for anyone who has the link (no sign-in).
 *
 * The token is read from the query string after hydration and sent to GET /api/share/{token}; the page does not
 * store it. The server answers every link that does not work (invalid, tampered, expired, revoked, or to a deleted
 * plan) with the same 404, so the page shows one message for all of them, and the same message when the address has
 * no token at all. Other failures (server not reachable, too many requests) use the usual ProblemNotice.
 *
 * The page shows only what the answer holds: the plan's name if it has one (else "<business type> in <city>"), the
 * city and business type, the plan's figures with who wrote each one, the opening date, the sites' names and
 * approximate locations (the server rounds latitude and longitude to 2 decimal places, about 1 km, so they are shown
 * with 2 decimals), when the link stops working, the disclaimer and the notice.
 */

/** The plan's money as tiles, each with its status chip. Labels speak of "the owner": the reader did not write them. */
function SharedFigures({ money }: { money: ProjectMoney }) {
  const budgetLeftEmpty = money.budget_total === null;
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-6">
      <Figure
        label="Build-out and equipment budget"
        value={budgetLeftEmpty ? formatUsd(money.build_out_assumed) : formatUsd(money.budget_total)}
        status={money.budget_status}
        note={budgetLeftEmpty ? "Left empty by the owner: the simulation assumes this build-out." : undefined}
      />
      <Figure label="The owner's own cash (equity)" value={formatUsd(money.owner_equity)} status={money.owner_equity_status} />
      <Figure label="Loan amount" value={formatUsd(money.loan_amount)} status={money.loan_amount_status} />
      <Figure
        label="Loan term"
        value={isNumber(money.loan_term_months) ? `${money.loan_term_months} months` : NOT_AVAILABLE}
        status={money.loan_term_status}
      />
      <Figure
        label={`Interest rate (${money.rate_type})`}
        value={money.annual_rate_pct === null ? "Prime + spread" : formatPercent(money.annual_rate_pct)}
        status={money.annual_rate_status}
      />
      <Figure
        label="Cash reserve at opening"
        value={formatUsd(money.cash_reserve_at_opening)}
        status={money.cash_reserve_status}
        note="Cash plus the loan, minus the build-out (calculated by the server)."
      />
    </div>
  );
}

function SharedPlanBody({ plan }: { plan: SharedPlan }) {
  const title = plan.plan_name || `${plan.business_type} in ${plan.metro_name}`;
  return (
    <div className="flex flex-col gap-10">
      <PageIntro
        eyebrow="Shared plan · read-only"
        title={title}
        description={
          <p>
            {plan.business_type} in {plan.metro_name}. The plan&apos;s owner shared it with a link. Nothing on this page can
            be changed.
          </p>
        }
      >
        <p className="flex items-center gap-1.5 text-sm text-muted-foreground">
          <Clock className="size-4 shrink-0" aria-hidden="true" />
          This link works until {formatDateTime(plan.expires_at)}.
        </p>
      </PageIntro>

      <div className="flex flex-col gap-3">
        <ResultsDisclaimer text={plan.disclaimer} />
        <Alert>
          <Eye aria-hidden="true" />
          <AlertTitle>About this shared plan</AlertTitle>
          <AlertDescription className="text-foreground/85">
            <p>{plan.notice}</p>
          </AlertDescription>
        </Alert>
        {plan.metro_id === SYNTHETIC_METRO_ID ? (
          <SyntheticBanner>This plan uses the generated demo city. Its places describe no real place.</SyntheticBanner>
        ) : null}
      </div>

      <section aria-labelledby="shared-money-title" className="flex flex-col gap-4">
        <h2 id="shared-money-title" className="font-serif text-2xl font-semibold">
          The plan&apos;s numbers
        </h2>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
          Each figure says who wrote it: typed by the plan&apos;s owner (user stated), a stated default (assumption), a change
          suggested by the path search, or calculated by the server.
        </p>
        <SharedFigures money={plan.money} />
        <p className="text-sm">
          <span className="text-muted-foreground">Planned opening date: </span>
          <span className="font-medium">{formatDate(plan.opening_date)}</span>
        </p>
      </section>

      <section aria-labelledby="shared-sites-title" className="flex flex-col gap-4">
        <h2 id="shared-sites-title" className="font-serif text-2xl font-semibold">
          Candidate sites
        </h2>
        <p className="text-sm text-muted-foreground">
          Names and approximate locations (rounded to about 1 km) only; street addresses and exact points are never shared.
        </p>
        <ol className="grid gap-2 text-sm sm:grid-cols-3">
          {plan.sites.map((site, index) => (
            <li key={`${index}-${site.name}`} className="flex items-start gap-2">
              <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-viz-1 text-[0.6875rem] font-semibold text-accent-ink" aria-hidden="true">
                {index + 1}
              </span>
              <span className="flex min-w-0 flex-col">
                <span className="break-words font-medium">{site.name}</span>
                <span className="figure-nums text-xs text-muted-foreground">
                  Approximate location {formatNumber(site.lat, 2)}, {formatNumber(site.lon, 2)}
                </span>
              </span>
            </li>
          ))}
        </ol>
      </section>

      <p className="text-sm text-muted-foreground">
        A shared plan holds the plan&apos;s own figures only. Site scores, the simulation and the path to approval are shown to
        the plan&apos;s owner when signed in.
      </p>
    </div>
  );
}

/** One message for every link that does not work, so a visitor cannot tell an expired link from a made-up one. */
function InvalidLink() {
  return (
    <Alert variant="warning" role="alert" className="gap-y-2">
      <LinkIcon aria-hidden="true" />
      <AlertTitle>This link is invalid or has expired</AlertTitle>
      <AlertDescription className="text-foreground/85">
        <p>Ask the person who shared the plan for a new link.</p>
        <div className="pt-1">
          <Button asChild variant="outline" size="sm">
            <Link href="/about/">
              What is this site?
              <ArrowRight aria-hidden="true" />
            </Link>
          </Button>
        </div>
      </AlertDescription>
    </Alert>
  );
}

export function SharedPlanView() {
  const hydrated = useIsHydrated();
  const token = useQueryParam("t");
  const shared = useApiResource(token ? `share:${token}` : null, (signal) => shareApi.read(token ?? "", signal), "Opening the shared plan");
  const plan = shared.data;
  const linkDoesNotWork = (hydrated && !token) || shared.problem?.kind === "not_found";

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 sm:px-6 sm:py-14">
      {/* Until the plan loads (or when it cannot), the page still needs its one h1. */}
      {plan ? null : <PageIntro eyebrow="Shared plan · read-only" title="A shared plan" />}
      {!hydrated || shared.status === "loading" ? <WorkspaceSkeleton /> : null}
      {linkDoesNotWork ? <InvalidLink /> : null}
      {shared.problem && !linkDoesNotWork ? (
        <ProblemNotice problem={shared.problem} onRetry={shared.reload} retrying={shared.refreshing} />
      ) : null}
      {plan && !linkDoesNotWork ? <SharedPlanBody plan={plan} /> : null}
    </div>
  );
}
