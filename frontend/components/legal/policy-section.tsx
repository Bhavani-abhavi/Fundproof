import type { ReactNode } from "react";

/**
 * One titled section of the Privacy or Terms page: a serif h2 and plain paragraphs or lists under it.
 * `id` is also the anchor (for example /privacy/#delete links straight to "How to delete your data").
 */
export function PolicySection({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="flex scroll-mt-24 flex-col gap-3">
      <h2 id={`${id}-title`} className="font-serif text-2xl font-semibold text-foreground">
        {title}
      </h2>
      <div className="flex flex-col gap-3 text-base leading-relaxed text-muted-foreground [&_strong]:font-semibold [&_strong]:text-foreground">
        {children}
      </div>
    </section>
  );
}

/** A bulleted list in a policy section (same bullets as the About page). */
export function PolicyList({ items }: { items: ReactNode[] }) {
  return (
    <ul className="flex flex-col gap-2">
      {items.map((item, index) => (
        <li key={index} className="flex gap-2.5">
          <span className="mt-2.5 size-1.5 shrink-0 rounded-full bg-accent" aria-hidden="true" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}
