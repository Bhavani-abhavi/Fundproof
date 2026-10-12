import { LogIn } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/** Shown on signed-in areas when nobody is signed in. */
export function SignInPrompt({ area, reason }: { area: string; reason: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Sign in to open {area}</CardTitle>
        <CardDescription>{reason}</CardDescription>
      </CardHeader>
      <CardContent>
        <Button asChild>
          <Link href="/signin">
            <LogIn aria-hidden="true" />
            Sign in
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}
