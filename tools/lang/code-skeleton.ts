/**
 * The "skeleton" of a TypeScript source: its text with every comment removed — the language's
 * comments, and the `//` / `/* ... *\/` comments inside template literals, where the WGSL lives — and
 * whitespace collapsed. Two versions of a file with the same skeleton differ only in comments and
 * whitespace; that is the check a comment-translation commit has to pass (ADR 0130 decision 6: the
 * translation of comments changes no code).
 *
 * Usage (from the repository root):
 *   node tools/lang/code-skeleton.ts --compare <git-ref> [--head <git-ref>] [file...]
 *       compares the skeleton of every changed `*.ts` file (or the given files) between `<git-ref>`
 *       and the working tree (or `--head`); exit 1 when a skeleton differs
 *   node tools/lang/code-skeleton.ts <file>
 *       prints the skeleton of one file
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseSync } from "oxc-parser";

type Span = readonly [start: number, end: number];

type AstNode = { readonly type?: unknown; readonly start?: unknown; readonly end?: unknown };

const isNode = (value: unknown): value is AstNode =>
  typeof value === "object" && value !== null && typeof (value as AstNode).type === "string";

/** Depth-first walk over every object with a `type` field (oxc's ESTree-shaped program). */
const walk = (value: unknown, visit: (node: AstNode) => void): void => {
  if (Array.isArray(value)) {
    for (const item of value) walk(item, visit);
    return;
  }
  if (!isNode(value)) return;
  visit(value);
  for (const [key, child] of Object.entries(value)) {
    if (key === "type" || key === "start" || key === "end" || key === "range") continue;
    if (typeof child === "object" && child !== null) walk(child, visit);
  }
};

/**
 * Spans of comment-like text inside a template quasi. `//` starts a comment unless it follows `:` (a
 * URL) or `/`; it ends at the line end. `/*` ends at the next `*\/` or at the end of the quasi.
 */
export const commentSpansInTemplate = (text: string, offset: number): Span[] => {
  const spans: Span[] = [];
  let i = 0;
  while (i < text.length) {
    if (text.startsWith("/*", i)) {
      const close = text.indexOf("*/", i + 2);
      const end = close < 0 ? text.length : close + 2;
      spans.push([offset + i, offset + end]);
      i = end;
      continue;
    }
    if (text.startsWith("//", i) && (i === 0 || (text[i - 1] !== ":" && text[i - 1] !== "/"))) {
      const newline = text.indexOf("\n", i);
      const end = newline < 0 ? text.length : newline;
      spans.push([offset + i, offset + end]);
      i = end;
      continue;
    }
    i += 1;
  }
  return spans;
};

type TemplateElement = AstNode & { readonly value?: { readonly raw?: unknown } };
type TemplateLiteral = AstNode & { readonly quasis?: readonly TemplateElement[] };

/**
 * The raw text of a template element and its offset in `source`. oxc's span of a `TemplateElement`
 * includes its delimiters (the opening backtick or `}` and the closing `${` or backtick), so the raw
 * text starts one character after `start`; the position is verified against the source.
 */
const rawTextOf = (
  source: string,
  quasi: TemplateElement,
): readonly [text: string, offset: number] | undefined => {
  const raw = quasi.value?.raw;
  if (typeof raw !== "string" || typeof quasi.start !== "number") return undefined;
  const offset = quasi.start + 1;
  if (source.slice(offset, offset + raw.length) !== raw) {
    throw new Error(`code-skeleton: template element text does not match the source at ${offset}`);
  }
  return [raw, offset];
};

/** The skeleton of `source` (a TypeScript module named `filename`). Throws when the source does not parse. */
export const skeletonOf = (filename: string, source: string): string => {
  const result = parseSync(filename, source);
  if (result.errors.length > 0) {
    throw new Error(
      `code-skeleton: cannot parse ${filename}: ${result.errors.map((error) => error.message).join("; ")}`,
    );
  }
  const cuts: Span[] = result.comments.map((comment) => [comment.start, comment.end]);
  walk(result.program, (node) => {
    if (node.type !== "TemplateLiteral") return;
    for (const quasi of (node as TemplateLiteral).quasis ?? []) {
      const raw = rawTextOf(source, quasi);
      if (raw === undefined) continue;
      cuts.push(...commentSpansInTemplate(raw[0], raw[1]));
    }
  });
  cuts.sort((a, b) => a[0] - b[0]);
  let text = "";
  let cursor = 0;
  for (const [start, end] of cuts) {
    if (start < cursor) continue; // nested inside a cut already applied
    text += source.slice(cursor, start) + " ";
    cursor = end;
  }
  text += source.slice(cursor);
  return text.replace(/\s+/g, " ").trim();
};

const gitShow = (root: string, ref: string, file: string): string | undefined => {
  try {
    return execFileSync("git", ["-C", root, "show", `${ref}:${file}`], {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    return undefined;
  }
};

const changedTypeScriptFiles = (root: string, base: string, head: string | undefined): string[] => {
  const args = ["-C", root, "diff", "--name-only", "--diff-filter=AMR", base];
  if (head !== undefined) args.push(head);
  args.push("--", "*.ts");
  const output = execFileSync("git", args, { encoding: "utf8", maxBuffer: 64 * 1024 * 1024 });
  return output.split("\n").filter((line) => line.length > 0);
};

/** The first index at which two strings differ, or -1 when they are equal. */
const firstDifference = (a: string, b: string): number => {
  const length = Math.min(a.length, b.length);
  for (let i = 0; i < length; i += 1) if (a[i] !== b[i]) return i;
  return a.length === b.length ? -1 : length;
};

const main = (argv: readonly string[]): void => {
  const root = process.cwd();
  let base: string | undefined;
  let head: string | undefined;
  const files: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--compare") {
      base = argv[i + 1];
      i += 1;
    } else if (arg === "--head") {
      head = argv[i + 1];
      i += 1;
    } else files.push(arg);
  }
  if (base === undefined) {
    if (files.length !== 1)
      throw new Error("usage: code-skeleton.ts --compare <ref> [--head <ref>] [file...] | <file>");
    console.log(skeletonOf(files[0], readFileSync(join(root, files[0]), "utf8")));
    return;
  }
  const targets = files.length > 0 ? files : changedTypeScriptFiles(root, base, head);
  let differing = 0;
  for (const file of targets) {
    const before = gitShow(root, base, file);
    const after =
      head === undefined ? readFileSync(join(root, file), "utf8") : gitShow(root, head, file);
    if (before === undefined || after === undefined) {
      console.log(`${file}: added or removed (not comparable)`);
      differing += 1;
      continue;
    }
    const a = skeletonOf(file, before);
    const b = skeletonOf(file, after);
    const at = firstDifference(a, b);
    if (at < 0) continue;
    differing += 1;
    console.log(`${file}: code differs near: …${a.slice(Math.max(0, at - 40), at + 60)}…`);
  }
  console.log(
    `compared ${targets.length} files against ${base}; ${differing} differ beyond comments`,
  );
  if (differing > 0) process.exitCode = 1;
};

if (import.meta.main) {
  main(process.argv.slice(2));
}
