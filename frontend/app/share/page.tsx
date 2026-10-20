import type { Metadata } from "next";

import { SharedPlanView } from "@/components/share/shared-plan-view";

export const metadata: Metadata = {
  title: "Shared plan",
  description: "A plan its owner shared with a link: read-only figures and site names.",
  // The link's token is in this page's address: never send the address to another site in a Referer header.
  referrer: "no-referrer",
};

/** /share?t=<token>: a query string, because a static export cannot pre-render unknown tokens. No sign-in needed. */
export default function SharePage() {
  return <SharedPlanView />;
}
