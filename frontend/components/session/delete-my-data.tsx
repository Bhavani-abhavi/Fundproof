"use client";

import { Clock, LogIn, Trash2, UserX } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { ProblemNotice } from "@/components/workspace/problem-notice";
import type { DeleteMyDataResult } from "@/lib/api";
import { describeProblem, type Problem } from "@/lib/api-errors";
import { useSession } from "@/lib/auth";
import { formatNumber } from "@/lib/format";
import { useCurrentPath } from "@/lib/location";
import { signInPageHref } from "@/lib/sign-in";
import { cn } from "@/lib/utils";

/**
 * "Delete all my data" (Studio S8), on the About page under "Your information" (anchor #delete-data, which the privacy
 * policy and the Terms of Use link to).
 *
 * Signed in: the button opens a confirmation dialog that says what will be deleted and that it cannot be undone.
 * "Delete everything" calls DELETE /api/me/data through useSession().deleteMyData, which then reads the session again,
 * so the whole page (header included) shows the user as signed out. The result stays on screen after the sign-out:
 * how many stored objects were deleted, or that the deletion is pending on the storage service (the copy in the
 * Hugging Face dataset could not be deleted yet), in the server's words when it sent some.
 * Not signed in: a sign-in link, because the server deletes the data of the signed-in account only.
 */

const PENDING_TEXT =
  "The deletion is pending on the storage service: your data is deleted on the app's server, but the copy in the backup " +
  "dataset could not be deleted yet. The server retries it; until then your data is not fully gone.";

/** "Deleted 4 stored objects ..." from the server's count (null when the server did not report one). */
function countText(deletedObjects: number | null): string {
  if (deletedObjects === null) {
    return "The server did not report how many stored objects were deleted.";
  }
  if (deletedObjects === 0) {
    return "Nothing was stored under your account.";
  }
  const plural = deletedObjects === 1 ? "object" : "objects";
  return `Deleted ${formatNumber(deletedObjects)} stored ${plural} (plans, DEMO decision records and share links).`;
}

function DeletionResult({ result, signedIn }: { result: DeleteMyDataResult; signedIn: boolean }) {
  const ref = useRef<HTMLDivElement | null>(null);

  // The dialog and its button are gone; move keyboard and screen-reader focus to what happened.
  useEffect(() => {
    ref.current?.focus();
  }, []);

  const Icon = result.pending ? Clock : UserX;
  return (
    <div
      ref={ref}
      tabIndex={-1}
      role="status"
      className={cn(
        "flex items-start gap-3 rounded-lg border p-4 text-sm outline-none focus-visible:outline-2 focus-visible:outline-ring",
        result.pending ? "border-dashed border-unknown-line bg-unknown-soft text-unknown" : "border-accent/30 bg-accent-soft text-ink",
      )}
    >
      <Icon className="mt-0.5 size-5 shrink-0" aria-hidden="true" />
      <div className="flex flex-col gap-1.5 leading-relaxed">
        <p className="font-semibold">{result.pending ? "Deletion pending on the storage service" : "Your data is deleted"}</p>
        <p>{countText(result.deleted_objects)}</p>
        {result.pending ? <p>{result.message ?? PENDING_TEXT}</p> : null}
        <p>{signedIn ? "The server still shows you as signed in: use Sign out at the top of the page." : "You are signed out."}</p>
      </div>
    </div>
  );
}

export function DeleteMyData() {
  const { status, me, deleteMyData } = useSession();
  const currentPath = useCurrentPath();
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [result, setResult] = useState<DeleteMyDataResult | null>(null);
  const name = me.display_name || me.username;

  function onOpenChange(next: boolean) {
    setOpen(next);
    if (next) {
      setProblem(null);
    }
  }

  async function deleteEverything() {
    setDeleting(true);
    setProblem(null);
    try {
      const answer = await deleteMyData();
      setOpen(false);
      setResult(answer);
    } catch (error) {
      setProblem(describeProblem(error, "Deleting your data"));
    } finally {
      setDeleting(false);
    }
  }

  let body: ReactNode;
  if (result) {
    body = <DeletionResult result={result} signedIn={me.signed_in} />;
  } else if (status === "loading") {
    body = <Skeleton className="h-10 w-48" aria-label="Checking sign-in" />;
  } else if (!me.signed_in) {
    body =
      status === "ready" ? (
        <div className="flex flex-col items-start gap-2">
          <p className="text-sm text-muted-foreground">Sign in first: the server deletes the data of the signed-in account.</p>
          <Button asChild variant="outline" size="sm">
            <Link href={signInPageHref(currentPath)}>
              <LogIn aria-hidden="true" />
              Sign in
            </Link>
          </Button>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">This needs the server, which is not reachable right now.</p>
      );
  } else {
    body = (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogTrigger asChild>
          <Button type="button" variant="outline" className="self-start text-conflicting">
            <Trash2 aria-hidden="true" />
            Delete all my data
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete all your data?</DialogTitle>
            <DialogDescription>
              This deletes everything stored under your account{name ? ` (${name})` : ""}. It cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <ul className="list-disc space-y-1 pl-5 text-sm text-foreground">
            <li>Every plan, with its sites and figures, and the results kept for it.</li>
            <li>Every DEMO decision record from the Lender Desk.</li>
            <li>Every share link you made: they stop working at once.</li>
          </ul>
          <p className="text-sm text-muted-foreground">
            You will be signed out. To keep a plan&apos;s results, download them from the plan&apos;s page first.
          </p>
          {problem ? <ProblemNotice problem={problem} onRetry={deleteEverything} retrying={deleting} /> : null}
          <DialogFooter>
            <DialogClose asChild>
              <Button type="button" variant="outline">
                Keep my data
              </Button>
            </DialogClose>
            <Button type="button" variant="destructive" onClick={deleteEverything} disabled={deleting}>
              <Trash2 aria-hidden="true" />
              {deleting ? "Deleting..." : "Delete everything"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <div id="delete-data" className="flex scroll-mt-24 flex-col gap-3 rounded-xl border border-border bg-card p-5 sm:p-6">
      <div className="flex flex-col gap-1">
        <h3 className="text-lg font-semibold text-foreground">Delete all my data</h3>
        <p className="text-sm leading-relaxed text-muted-foreground">
          Deletes every plan, DEMO decision record and share link stored under your account, drops your results from the
          server&apos;s memory and signs you out. It cannot be undone.
        </p>
      </div>
      {body}
    </div>
  );
}
