"use client";

import { Check, Copy, Link2, Unlink } from "lucide-react";
import { useId, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ProblemNotice } from "@/components/workspace/problem-notice";
import { describeProblem, type Problem } from "@/lib/api-errors";
import { formatDateTime } from "@/lib/format";
import { absoluteShareUrl, SHARE_DAY_CHOICES, SHARE_DAYS_DEFAULT, shareApi, type ShareLinkCreated } from "@/lib/share-api";

/**
 * "Share this plan" on a plan's page (Studio S8): a read-only link for someone without an account.
 *
 * - "Create share link" POSTs {expires_in_days} and shows the full link, when it stops working, and a Copy button.
 *   There is no list of earlier links (the API has none), so a link is shown only right after it is created.
 * - "Revoke all share links" DELETEs every link of this plan at once and says how many the server revoked.
 * What a link shows is decided by the server (lib/share-api.ts); the text here says it in plain words.
 */

/** What the status line says after "Revoke all share links". */
function revokedText(count: number): string {
  if (count === 0) {
    return "There was no share link to revoke.";
  }
  if (count === 1) {
    return "Revoked 1 share link. It no longer works.";
  }
  return `Revoked ${count} share links. None of them works any more.`;
}

export function ShareLinks({ projectId }: { projectId: string }) {
  const [days, setDays] = useState(String(SHARE_DAYS_DEFAULT));
  const [link, setLink] = useState<ShareLinkCreated | null>(null);
  const [busy, setBusy] = useState<"create" | "revoke" | null>(null);
  const [copied, setCopied] = useState<"copied" | "failed" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const linkInput = useRef<HTMLInputElement | null>(null);
  const daysId = useId();
  const linkId = useId();

  async function create() {
    setBusy("create");
    setProblem(null);
    setMessage(null);
    try {
      const created = await shareApi.create(projectId, Number(days));
      setLink(created);
      setCopied(null);
      setMessage(`Share link created. It works until ${formatDateTime(created.expires_at)}.`);
    } catch (error) {
      setProblem(describeProblem(error, "Creating the share link"));
    } finally {
      setBusy(null);
    }
  }

  async function revokeAll() {
    setBusy("revoke");
    setProblem(null);
    setMessage(null);
    try {
      const answer = await shareApi.revokeAll(projectId);
      setLink(null);
      setMessage(revokedText(answer.revoked));
    } catch (error) {
      setProblem(describeProblem(error, "Revoking the share links"));
    } finally {
      setBusy(null);
    }
  }

  async function copy() {
    if (!link) {
      return;
    }
    try {
      await navigator.clipboard.writeText(absoluteShareUrl(link.url));
      setCopied("copied");
    } catch {
      // No clipboard access (an older browser, or a page not served over HTTPS): select the link for a manual copy.
      linkInput.current?.select();
      setCopied("failed");
    }
  }

  return (
    <section aria-labelledby="share-title" className="flex flex-col gap-4 rounded-xl border border-border bg-card p-4 sm:p-5">
      <div className="flex flex-col gap-1">
        <h3 id="share-title" className="flex items-center gap-2 font-semibold">
          <Link2 className="size-4 text-accent" aria-hidden="true" />
          Share this plan
        </h3>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Anyone with the link can see this plan&apos;s figures, its city and business type, and its site names and map
          positions, without signing in, until the link expires. Not your idea text, your street addresses or anything
          about your account. A new link does not stop earlier ones; &ldquo;Revoke all share links&rdquo; stops them all, and
          so does deleting the plan.
        </p>
      </div>

      <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={daysId}>The link works for</Label>
          <select
            id={daysId}
            value={days}
            onChange={(event) => setDays(event.target.value)}
            className="h-10 w-full min-w-0 rounded-md border border-input bg-card px-3 text-base text-foreground focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ring sm:w-48 sm:text-sm"
          >
            {SHARE_DAY_CHOICES.map((choice) => (
              <option key={choice} value={choice}>
                {choice} {choice === 1 ? "day" : "days"}
                {choice === SHARE_DAYS_DEFAULT ? " (default)" : ""}
              </option>
            ))}
          </select>
        </div>
        <Button type="button" onClick={create} disabled={busy !== null} className="self-start sm:self-auto">
          <Link2 aria-hidden="true" />
          {busy === "create" ? "Creating..." : "Create share link"}
        </Button>
      </div>

      {link ? (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={linkId}>Share link</Label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Input
              ref={linkInput}
              id={linkId}
              readOnly
              value={absoluteShareUrl(link.url)}
              onFocus={(event) => event.currentTarget.select()}
              className="font-mono sm:text-xs"
            />
            <Button type="button" variant="outline" onClick={copy} className="self-start">
              {copied === "copied" ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
              {copied === "copied" ? "Copied" : "Copy link"}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground">
            Works until {formatDateTime(link.expires_at)}. This page shows the link only now; keep it if you need it again.
          </p>
          {copied === "failed" ? (
            <p role="status" className="text-xs text-foreground">
              Copying did not work in this browser. The link is selected: copy it with Ctrl+C (or Cmd+C on a Mac).
            </p>
          ) : null}
        </div>
      ) : null}

      {message ? (
        <p role="status" className="rounded-lg border border-accent/30 bg-accent-soft p-3 text-sm text-ink">
          {message}
        </p>
      ) : null}
      {problem ? <ProblemNotice problem={problem} /> : null}

      <Button type="button" variant="outline" onClick={revokeAll} disabled={busy !== null} className="self-start text-conflicting">
        <Unlink aria-hidden="true" />
        {busy === "revoke" ? "Revoking..." : "Revoke all share links"}
      </Button>
    </section>
  );
}
