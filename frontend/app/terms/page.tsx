import type { Metadata } from "next";
import Link from "next/link";

import { DraftBanner } from "@/components/legal/draft-banner";
import { PolicyList, PolicySection } from "@/components/legal/policy-section";
import { PageIntro } from "@/components/layout/page-intro";
import { DISCLAIMER, PRODUCT_NAME } from "@/lib/config";

export const metadata: Metadata = {
  title: "Terms of Use",
  description: `The rules for using ${PRODUCT_NAME}, a prototype educational planning tool, in plain words.`,
};

/**
 * /terms/: the plain-language Terms of Use.
 *
 * It restates the project's standing rules (Studio addendum rule C: no lending decisions, no advice, the disclaimer
 * word for word) and the prototype notice, and lists what not to enter, matching what the server refuses
 * (docs/studio/privacy.md, section 1). It adds no legal promise of its own: the owners have not approved it yet
 * (DraftBanner).
 */
export default function TermsPage() {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-10 px-4 py-10 sm:px-6 sm:py-14">
      <PageIntro
        eyebrow="Terms"
        title="Terms of Use"
        description={<p>The rules for using {PRODUCT_NAME}, in plain words.</p>}
      />

      <DraftBanner />

      <PolicySection id="what-it-is" title="What it is">
        <p className="rounded-lg border-2 border-accent bg-accent-soft p-4 font-medium text-foreground">{DISCLAIMER}</p>
        <p>
          Founder Studio helps you put a business plan together: it scores candidate sites, simulates revenue and cash from
          stated assumptions, and shows plan changes that would reach common planning targets. The Lender Desk is a DEMO
          that shows how a reviewer could check a loan file; it uses SYNTHETIC (computer-generated) files only.
        </p>
      </PolicySection>

      <PolicySection id="not-advice" title="Not advice, and no lending decisions">
        <PolicyList
          items={[
            <>
              Nothing in {PRODUCT_NAME} is financial, legal or lending advice. Talk to a qualified professional before you
              rely on a figure for a real decision.
            </>,
            <>
              Probabilities and ranges are <strong>simulated, based on stated assumptions</strong>. They make no claim of
              real-world accuracy.
            </>,
            <>
              {PRODUCT_NAME} never approves or declines a loan and never promises approval. Reaching the planning targets
              does not mean a lender will approve your loan. <strong>Lenders make their own decisions.</strong>
            </>,
            <>{PRODUCT_NAME} does not approve, endorse or recommend any lender.</>,
          ]}
        />
      </PolicySection>

      <PolicySection id="prototype" title="A prototype">
        <p>
          {PRODUCT_NAME} is a demonstration prototype, built in the open. It can change, be unavailable, or lose saved plans
          (for example when its server restarts). Do not use it as the only copy of anything you need: download a
          plan&apos;s results from its page if you want to keep them.
        </p>
      </PolicySection>

      <PolicySection id="made-up-plans" title="Please use made-up plans">
        <p>
          Until the owners approve the{" "}
          <Link href="/privacy/" className="font-medium text-accent underline underline-offset-4">
            privacy policy
          </Link>
          , try Founder Studio with invented figures and public addresses only, and do not describe real people.
        </p>
      </PolicySection>

      <PolicySection id="acceptable-use" title="Acceptable use">
        <p>Do not enter:</p>
        <PolicyList
          items={[
            <>personal identifiers, such as Social Security numbers or other identity numbers, or details about real people;</>,
            <>tax returns or anything copied from them;</>,
            <>bank account, card or other account numbers.</>,
          ]}
        />
        <p>
          The app refuses Social Security number patterns, labeled account numbers and long runs of digits, but that check is
          a safety net, not permission. There are no uploads in this version.
        </p>
        <p>Do not try to break the service. That includes:</p>
        <PolicyList
          items={[
            <>trying to get around sign-in, or to read or change another person&apos;s plans or records;</>,
            <>trying to get around the request limits, or sending requests to overload the server;</>,
            <>trying to make the app run instructions hidden in the text you type;</>,
            <>using the app to break any law or to harm anyone.</>,
          ]}
        />
      </PolicySection>

      <PolicySection id="share-links" title="Share links">
        <p>
          A share link is created only when you ask for one. Anyone who has it can see that plan&apos;s figures, its city and
          business type, and its site names and map positions, without signing in, until it expires or you revoke it. Send
          it only to people you choose. Your idea text, street addresses and account are never shown through it.
        </p>
      </PolicySection>

      <PolicySection id="your-data" title="Your data">
        <p>
          The{" "}
          <Link href="/privacy/" className="font-medium text-accent underline underline-offset-4">
            privacy policy
          </Link>{" "}
          says what is stored and for how long. You can delete everything stored under your account at any time with
          &ldquo;Delete all my data&rdquo; on the{" "}
          <Link href="/about/#delete-data" className="font-medium text-accent underline underline-offset-4">
            About page
          </Link>
          .
        </p>
      </PolicySection>

      <PolicySection id="sources" title="Data sources">
        <p>
          The public datasets {PRODUCT_NAME} uses keep their own terms and licenses, listed on the{" "}
          <Link href="/about/" className="font-medium text-accent underline underline-offset-4">
            About page
          </Link>
          . For example, the OpenSanctions data is licensed CC BY-NC 4.0, for non-commercial use with attribution.
        </p>
      </PolicySection>

      <PolicySection id="changes" title="Changes to these terms">
        <p>
          These terms are a draft. The owners may change them before they approve them; the banner at the top of this page
          goes away when they do.
        </p>
      </PolicySection>
    </div>
  );
}
