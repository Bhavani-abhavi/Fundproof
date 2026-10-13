import { cn } from "@/lib/utils";

/**
 * Shows the small Markdown subset the guideline sections and decision records use (headings, "- " lists, pipe
 * tables, paragraphs) as plain React elements. No HTML from the API is ever rendered: every piece of text goes in
 * as a React text node, so a section or record cannot inject markup. Anything else is shown as plain text.
 */

type Block =
  | { kind: "heading"; text: string }
  | { kind: "list"; items: string[] }
  | { kind: "table"; header: string[]; rows: string[][] }
  | { kind: "paragraph"; text: string };

function cells(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function isSeparator(line: string): boolean {
  return /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line.trim());
}

/** Split the text into blocks, line by line. */
export function parseBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index] ?? "";
    const trimmed = line.trim();
    if (trimmed === "") {
      index += 1;
    } else if (/^#{1,6}\s/.test(trimmed)) {
      blocks.push({ kind: "heading", text: trimmed.replace(/^#{1,6}\s+/, "") });
      index += 1;
    } else if (trimmed.startsWith("|")) {
      const tableLines: string[] = [];
      while (index < lines.length && (lines[index] ?? "").trim().startsWith("|")) {
        tableLines.push(lines[index] ?? "");
        index += 1;
      }
      const rows = tableLines.filter((row) => !isSeparator(row)).map(cells);
      blocks.push({ kind: "table", header: rows[0] ?? [], rows: rows.slice(1) });
    } else if (/^[-*]\s/.test(trimmed)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s/.test((lines[index] ?? "").trim())) {
        items.push((lines[index] ?? "").trim().slice(2));
        index += 1;
      }
      blocks.push({ kind: "list", items });
    } else {
      const paragraph: string[] = [];
      while (index < lines.length) {
        const next = (lines[index] ?? "").trim();
        if (next === "" || next.startsWith("|") || /^#{1,6}\s/.test(next) || /^[-*]\s/.test(next)) {
          break;
        }
        paragraph.push(next);
        index += 1;
      }
      blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
    }
  }
  return blocks;
}

export function PlainMarkdown({ text, className }: { text: string; className?: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className={cn("flex flex-col gap-3 text-sm leading-relaxed [overflow-wrap:anywhere]", className)}>
      {blocks.map((block, index) => {
        switch (block.kind) {
          case "heading":
            return (
              <p key={index} className="font-semibold">
                {block.text}
              </p>
            );
          case "list":
            return (
              <ul key={index} className="list-disc space-y-1 pl-5">
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
                ))}
              </ul>
            );
          case "table":
            return (
              <div key={index} className="overflow-x-auto rounded-md border border-border">
                <table className="figure-nums w-full text-left text-xs [overflow-wrap:normal]">
                  <thead className="bg-paper-sunken">
                    <tr>
                      {block.header.map((cell, cellIndex) => (
                        <th key={cellIndex} scope="col" className="px-2.5 py-1.5 font-semibold">
                          {cell}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr key={rowIndex} className="border-t border-border">
                        {row.map((cell, cellIndex) => (
                          <td key={cellIndex} className="px-2.5 py-1.5 align-top">
                            {cell}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "paragraph":
            return <p key={index}>{block.text}</p>;
        }
      })}
    </div>
  );
}
