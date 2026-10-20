import type { Metadata } from "next";
import Link from "next/link";

import { DraftBanner } from "@/components/legal/draft-banner";
import { PolicyList, PolicySection } from "@/components/legal/policy-section";
import { PageIntro } from "@/components/layout/page-intro";
import { DISCLAIMER, PRODUCT_NAME, REPO_URL } from "@/lib/config";

export const metadata: Metadata = {
  title: "Privacy policy",
  description: `What ${PRODUCT_NAME} collects, why, who else sees it, how it is stored, how long it is kept and how to delete it.`,
};

/**
 * /privacy/: the plain-language privacy policy.
 *
 * Source of truth: docs/studio/privacy.md (what the v0.1 code does, with the file that does it, including the Studio
 * S8 share links, "delete all my data" and the plan access log). Nothing here goes beyond it. Where privacy.md says a
 * commitment is the owners' decision ("never sold"), the page states only the fact the code supports and the
 * DraftBanner says the page is not approved yet. Update this page and privacy.md together.
 */

/** The facts that matter most, repeated in detail in the sections below. */
const SHORT_VERSION = [
  <>
    <strong>Collected:</strong> your Hugging Face account id, username and display name (no e-mail address), the plans and
    DEMO decisions you save, a record of each share link you create, and an access-log event for each access to one of
    your plans.
  </>,
  <>
    <strong>Stored under your account only,</strong> with plans and decisions encrypted, and deleted automatically after
    the retention period (30 days by default), or sooner when you delete it.
  </>,
  <>
    <strong>Shared</strong> in two ways only: a street address for a real city is sent to the US Census Bureau Geocoder,
    and a share link you create shows whoever has the link that plan&apos;s name (if it has one), business type, city,
    money figures, opening date, and site names and approximate site locations. Hugging Face hosts the app.
  </>,
  <>
    <strong>No language model</strong> reads what you type in version 0.1. There are no analytics or tracking scripts and
    no uploads.
  </>,
  <>
    <strong>Delete everything stored under your account</strong> at any time with &ldquo;Delete all my data&rdquo; on
    the{" "}
    <Link href="/about/#delete-data" className="font-medium text-accent underline underline-offset-4">
      About page
    </Link>
    .
  </>,
  <>
    <strong>The access log stays.</strong> Its events about your plans (a digest of your account id, plan ids, actions and
    times; never plan content) are not removed by &ldquo;Delete all my data&rdquo; or by the retention period, and
    nothing deletes them automatically yet (see{" "}
    <a href="#logs" className="font-medium text-accent underline underline-offset-4">
      Logs
    </a>
    ).
  </>,
];

export default function PrivacyPage() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <PageIntro
        eyebrow="Privacy"
        title="Privacy policy"
        description={
          <p>
            What {PRODUCT_NAME} collects when you use it, why, who else sees it, how it is stored, how long it is kept and
            how to delete it. Written from what the code of version 0.1 does.
          </p>
        }
      />

      <DraftBanner />

      <PolicySection id="short" title="The short version">
        <PolicyList items={SHORT_VERSION} />
      </PolicySection>

      <PolicySection id="collected" title="What is collected, and why">
        <PolicyList
          items={[
            <>
              <strong>Your sign-in.</strong> {PRODUCT_NAME} uses &ldquo;Sign in with Hugging Face&rdquo; and asks only for the{" "}
              <code className="text-sm">openid profile</code> scopes, so it never receives your e-mail address. It keeps
              your account id, username and display name in a signed session cookie, which lasts 8 hours by default. Why:
              to keep your plans apart from everyone else&apos;s.
            </>,
            <>
              <strong>Your plans.</strong> For each plan: the city, the business type, your idea text (up to 2,000
              characters), the money figures, the opening date, 1 to 3 sites (a map pin, or for a real city a street
              address), and a name if you give the plan one. Why: to score the sites and simulate the plan. Each plan is
              stored as one encrypted object.
            </>,
            <>
              <strong>Your DEMO decisions.</strong> On the Lender Desk, the action and rationale you record on a SYNTHETIC
              demo file, stored as one encrypted record per user and file. Why: to show the decision record.
            </>,
            <>
              <strong>Your share links.</strong> When you create a share link for a plan, the server stores a record of
              that link under your account (its id, the plan&apos;s id and two dates), so it can check, expire and revoke
              it.
            </>,
            <>
              <strong>Accesses to your plans.</strong> Before the server lists, opens, creates, changes, scores, simulates,
              shares or deletes a plan of yours, revokes its share links or deletes all your data, and before it shows a
              plan to someone who opens one of your share links, it writes an event to the plan access log (see{" "}
              <a href="#logs" className="font-medium text-accent underline underline-offset-4">
                Logs
              </a>
              ). Why: a record of who reached each plan, and when. If the event cannot be written, the request is refused.
            </>,
          ]}
        />
      </PolicySection>

      <PolicySection id="refused" title="What the app refuses">
        <p>
          A Social Security number pattern (like 123-45-6789, also with spaces), a labeled account number (like
          &ldquo;acct 12345678&rdquo;) or any run of 8 to 17 digits is refused, and never stored, in a plan name, an idea, a
          site name, a site address and a decision rationale. The refusal is a plain message that does not repeat what you
          typed. A site address is checked before it is sent anywhere. The city&apos;s own 11-digit census tract numbers are
          allowed in site names (they are public, not personal).
        </p>
        <p>
          There are <strong>no uploads</strong> in version 0.1, so no tax returns, bank statements or other documents can
          be sent. A request body larger than 64 KiB is refused before it is read further.
        </p>
      </PolicySection>

      <PolicySection id="who-sees" title="Who else sees what you type">
        <PolicyList
          items={[
            <>
              <strong>The US Census Geocoder</strong> (a public service of the US Census Bureau). When you type a street
              address for a real city, the server sends that address, and nothing else from your plan, over HTTPS to find
              its location and census tract. The answer is not cached on the server; the match is stored inside your
              encrypted plan, and later changes to the plan reuse it instead of asking again.
            </>,
            <>
              <strong>No language model</strong> reads your idea text, your numbers or your rationale in version 0.1. The
              app&apos;s own code checks the idea text for instruction-like wording and marks it when it finds some; it is
              shown back only to you. The Lender Desk demo writes its memo from a fixed template.
            </>,
            <>
              <strong>Nobody else who signs in</strong> can read your plans or records: every object is stored under your
              account id, and another account gets &ldquo;not found&rdquo;.
            </>,
            <>
              <strong>People you send a share link to.</strong> Only if you create one. Anyone who has the link can see,
              without signing in, that plan&apos;s name if it has one (free text you typed), its business type and city,
              its money figures, its opening date, and its site names and approximate site locations (rounded to about
              1 km), until the link expires (you choose 1 to 30 days) or you revoke it. They do not see your idea text,
              street addresses or anything about your account. Deleting the plan also ends its links.
            </>,
            <>
              <strong>Hugging Face</strong> hosts the app and, if the owners turn it on, a private dataset that holds a copy
              of the encrypted store (see &ldquo;How it is stored&rdquo;).
            </>,
          ]}
        />
        <p>
          The app has no code that sells your data or shares it beyond what this page describes, and no analytics or
          tracking script: its Content Security Policy allows scripts and connections only to the app&apos;s own site. A formal promise that your data will never be sold is the owners&apos; decision and is not made on this draft
          page.
        </p>
      </PolicySection>

      <PolicySection id="storage" title="How it is stored">
        <PolicyList
          items={[
            <>
              <strong>Encrypted at rest.</strong> Plans, decision records and share-link records are encrypted (Fernet)
              before they are written, both on the server&apos;s disk and in the optional private Hugging Face dataset. The
              app refuses to use a public dataset.
            </>,
            <>
              <strong>Per account.</strong> Everything you save is stored under a digest of your account id.
            </>,
            <>
              <strong>Results in memory only.</strong> Site scores, simulations, path searches and Lender Desk file runs are
              kept in the server&apos;s memory, per plan version. They are dropped when the plan changes or is deleted, and
              when the server restarts.
            </>,
            <>
              <strong>Address lookups</strong> run in their own small worker pool with per-user limits (one batch at a time
              and a small number of addresses per minute), so they cannot be used to flood the geocoder.
            </>,
          ]}
        />
      </PolicySection>

      <PolicySection id="retention" title="How long it is kept">
        <PolicyList
          items={[
            <>
              Every object stored under your account is deleted automatically when its retention period ends: plans and
              DEMO decision records after <strong>30 days</strong> by default (a design choice, set on the server), a
              share-link record within about a day after its link expires. Expired objects are deleted when the server
              starts and then every hour. Each plan&apos;s page shows its own deletion date.
            </>,
            <>
              The sign-in cookie lasts <strong>8 hours</strong> by default; then you sign in again.
            </>,
            <>
              Without the optional dataset copy, everything also disappears when the app&apos;s server restarts or is
              rebuilt.
            </>,
            <>
              <strong>The plan access log has no retention period yet:</strong> nothing deletes its events automatically.
              In the hosted app it is on the server&apos;s own disk, which does not survive a restart, unless the operator
              moves it to lasting storage (see{" "}
              <a href="#logs" className="font-medium text-accent underline underline-offset-4">
                Logs
              </a>
              ).
            </>,
          ]}
        />
      </PolicySection>

      <PolicySection id="delete" title="How to delete your data">
        <PolicyList
          items={[
            <>
              <strong>Everything at once:</strong> &ldquo;Delete all my data&rdquo; on the{" "}
              <Link href="/about/#delete-data" className="font-medium text-accent underline underline-offset-4">
                About page
              </Link>{" "}
              deletes every plan, DEMO decision record and share-link record stored under your account, drops your results
              from the server&apos;s memory, and signs you out. Your share links stop working. It cannot be undone. The server
              first writes an access-log event for the deletion. <strong>What stays:</strong> that event and every earlier
              access-log event about your plans (see{" "}
              <a href="#logs" className="font-medium text-accent underline underline-offset-4">
                Logs
              </a>
              ).
            </>,
            <>
              <strong>One plan:</strong> &ldquo;Delete plan&rdquo; on the plan&apos;s page deletes the plan, ends its share
              links and drops its results from memory.
            </>,
            <>
              <strong>Share links only:</strong> &ldquo;Revoke all share links&rdquo; on the plan&apos;s page stops every link to
              that plan. With the optional dataset copy, if the copy of a link&apos;s record there cannot be deleted at once,
              the page says so: the app&apos;s server stops accepting the link at once, but if the app restarts before that
              deletion goes through, revoke again.
            </>,
            <>
              <strong>One DEMO decision record:</strong> the server can delete a single record, but the Lender Desk page has
              no button for it yet; &ldquo;Delete all my data&rdquo; deletes all of them.
            </>,
          ]}
        />
        <p>
          With the optional dataset copy, deleted files are removed from the dataset at once, and its history is squashed at
          the next sync so earlier versions do not keep them. If Hugging Face cannot be reached, the app says the deletion
          is <strong>pending</strong> instead of claiming it is done.
        </p>
      </PolicySection>

      <PolicySection id="logs" title="Logs">
        <PolicyList
          items={[
            <>
              One line per request: the method, the path with personal patterns and share-link tokens masked, the names of
              query parameters (never their values), the status and the time. No request bodies, cookies, headers, IP
              addresses or usernames are logged.
            </>,
            <>
              A failed address lookup is logged with the geocoder&apos;s web address (without the query) and the error, never
              the address you typed.
            </>,
            <>
              When a Lender Desk demo tool fails, the full error is logged on the server only; the page and the DEMO record
              show a plain sentence.
            </>,
            <>
              <strong>The plan access log.</strong> One event for each access listed under &ldquo;What is collected&rdquo;.
              An event holds the time, the action (for example &ldquo;read plan&rdquo; or &ldquo;view shared plan&rdquo;),
              the plan&apos;s id, a share link&apos;s id and number of days where one is involved, and who acted: for you,
              an owner digest (the first 32 hex characters of the SHA-256 of your account id); for someone opening a share
              link, the link&apos;s id. It never holds plan content, names, addresses or IP addresses. The digest is not
              anonymous: anyone who knows your account id can compute it and link the events to your account. The log is
              a plain (not encrypted) file, kept apart from your stored data and never copied to the dataset.{" "}
              <strong>
                It is not deleted by &ldquo;Delete all my data&rdquo;, by deleting a plan or by the retention period, and
                nothing deletes it automatically yet.
              </strong>{" "}
              In the hosted app it is on the server&apos;s own disk, which does not survive a restart, unless the operator
              points the server&apos;s <code className="text-sm">UWC_WEB_AUDIT_LOG</code> setting at lasting storage.
            </>,
          ]}
        />
      </PolicySection>

      <PolicySection id="no-decisions" title="No lending decisions">
        <p className="rounded-lg border border-accent/30 bg-accent-soft px-3 py-2 font-medium text-foreground">{DISCLAIMER}</p>
        <p>
          {PRODUCT_NAME} prepares and explains. It never approves or declines a loan, never promises approval and never
          recommends a lender. Its probabilities are simulated, based on stated assumptions. The Lender Desk is a DEMO on
          SYNTHETIC files, where a person records every decision.
        </p>
      </PolicySection>

      <PolicySection id="about-this-page" title="About this page">
        <p>
          {PRODUCT_NAME} is a demonstration prototype, not a compliance program. Whether any privacy law applies depends on
          who runs it and how, and this page does not claim compliance with any. The code that does everything described
          here is public on{" "}
          <a href={REPO_URL} rel="noopener noreferrer" className="font-medium text-accent underline underline-offset-4">
            GitHub
          </a>
          . See also the{" "}
          <Link href="/terms/" className="font-medium text-accent underline underline-offset-4">
            Terms of Use
          </Link>
          .
        </p>
      </PolicySection>
    </div>
  );
}
