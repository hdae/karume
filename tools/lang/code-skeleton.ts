/**
 * The "skeleton" of a TypeScript source: its text with every comment removed — the language's
 * comments, and the `//` / `/* ... *\/` comments inside template literals, where the WGSL lives — and
 * whitespace collapsed. Two versions of a file with the same skeleton differ only in comments and
 * whitespace; that is the check a comment-translation commit has to pass (ADR 0130 decision 6: the
 * translation of comments changes no code).
 *
 * The strings mode (`--strings`) also lifts every string literal and template literal out of the
 * skeleton into a list of cells. Two versions with the same strings skeleton differ only in
 * comments, whitespace and the text of their literals, and the cells align one to one, so the tool
 * can list exactly which literals changed. A changed literal whose old text held no Japanese is
 * reported as a code change: that is the check a message-translation commit has to pass (step 2-9e
 * of ADR 0130 — only Japanese strings change). A template literal's cell is its text without the
 * WGSL comments, so a WGSL comment edit is not a literal change while a WGSL token edit is.
 *
 * Usage (from the repository root):
 *   node tools/lang/code-skeleton.ts --compare <git-ref> [--head <git-ref>] [file...]
 *       compares the skeleton of every changed `*.ts` file (or the given files) between `<git-ref>`
 *       and the working tree (or `--head`); exit 1 when a skeleton differs
 *   node tools/lang/code-skeleton.ts --strings --compare <git-ref> [--head <git-ref>] [file...]
 *       the same with the strings skeleton; lists every changed literal; exit 1 when a skeleton
 *       differs or a literal without Japanese changed
 *   node tools/lang/code-skeleton.ts <file>
 *       prints the skeleton of one file
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parseSync } from "oxc-parser";
import { JAPANESE } from "./scan-japanese.ts";

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

/** How a template quasi ends: in no comment, inside a `//` line, or inside a `/* … *\/` block. */
export type CommentCarry = "none" | "line" | "block";

/**
 * Spans of comment-like text inside a template quasi. `//` starts a comment unless it follows `:` (a
 * URL) or `/`; it ends at the line end. `/*` ends at the next `*\/` or at the end of the quasi. A
 * comment that an interpolation (`${…}`) interrupts continues in the next quasi: `carry` says how
 * the previous quasi ended, and the returned carry says how this one ends (the interpolated
 * expression itself stays code).
 */
export const commentSpansInTemplate = (
  text: string,
  offset: number,
  carry: CommentCarry = "none",
): { readonly spans: Span[]; readonly carry: CommentCarry } => {
  const spans: Span[] = [];
  let open = carry;
  let i = 0;
  while (i < text.length) {
    if (open === "block" || text.startsWith("/*", i)) {
      const close = text.indexOf("*/", open === "block" ? i : i + 2);
      const end = close < 0 ? text.length : close + 2;
      spans.push([offset + i, offset + end]);
      i = end;
      open = close < 0 ? "block" : "none";
      continue;
    }
    if (
      open === "line" ||
      (text.startsWith("//", i) && (i === 0 || (text[i - 1] !== ":" && text[i - 1] !== "/")))
    ) {
      const newline = text.indexOf("\n", i);
      const end = newline < 0 ? text.length : newline;
      spans.push([offset + i, offset + end]);
      i = end;
      open = newline < 0 ? "line" : "none";
      continue;
    }
    i += 1;
  }
  return { spans, carry: open };
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

type StringLiteralNode = AstNode & { readonly value?: unknown };

/** A literal lifted out of the strings skeleton: where it starts and the text the cell compares. */
type LiteralCell = { readonly start: number; readonly text: string };

/** The skeleton text with, in strings mode, the literals it was cut from (in source order). */
export type Skeleton = { readonly text: string; readonly literals: readonly string[] };

/** The placeholder a lifted literal leaves in the strings skeleton (never produced by whitespace collapsing). */
const LITERAL_MARK = "\u{1}";

/**
 * The skeleton of `source` (a TypeScript module named `filename`). In `"comments"` mode only
 * comments are cut; in `"strings"` mode string literals and template literals are cut too and
 * returned as cells. Throws when the source does not parse.
 */
export const analyze = (
  filename: string,
  source: string,
  mode: "comments" | "strings" = "comments",
): Skeleton => {
  const result = parseSync(filename, source);
  if (result.errors.length > 0) {
    throw new Error(
      `code-skeleton: cannot parse ${filename}: ${result.errors.map((error) => error.message).join("; ")}`,
    );
  }
  const cuts: Span[] = result.comments.map((comment) => [comment.start, comment.end]);
  const literals: Span[] = [];
  const cells: LiteralCell[] = [];
  walk(result.program, (node) => {
    if (node.type === "Literal" && mode === "strings") {
      const { value, start, end } = node as StringLiteralNode;
      if (typeof value !== "string" || typeof start !== "number" || typeof end !== "number") return;
      literals.push([start, end]);
      cells.push({ start, text: value });
      return;
    }
    if (node.type !== "TemplateLiteral") return;
    let carry: CommentCarry = "none";
    const parts: string[] = [];
    for (const quasi of (node as TemplateLiteral).quasis ?? []) {
      const raw = rawTextOf(source, quasi);
      if (raw === undefined) continue;
      const found = commentSpansInTemplate(raw[0], raw[1], carry);
      cuts.push(...found.spans);
      carry = found.carry;
      if (mode === "strings") {
        literals.push([raw[1], raw[1] + raw[0].length]);
        parts.push(withoutSpans(raw[0], raw[1], found.spans));
      }
    }
    if (mode === "strings" && typeof node.start === "number") {
      // Whitespace collapsed like the skeleton: a re-wrapped WGSL comment leaves a different run of
      // blanks behind, and that is no token change
      cells.push({ start: node.start, text: parts.join("${}").replace(/\s+/g, " ").trim() });
    }
  });
  const marks = new Set(literals.map(([start]) => start));
  cuts.push(...literals);
  // The longer span first at equal starts: a lifted template literal covers the WGSL comments at
  // its own start, and the outer cut wins
  cuts.sort((a, b) => a[0] - b[0] || b[1] - a[1]);
  let text = "";
  let cursor = 0;
  for (const [start, end] of cuts) {
    if (start < cursor) continue; // nested inside a cut already applied
    text += source.slice(cursor, start) + (marks.has(start) ? LITERAL_MARK : " ");
    cursor = end;
  }
  text += source.slice(cursor);
  cells.sort((a, b) => a.start - b.start);
  return { text: text.replace(/\s+/g, " ").trim(), literals: cells.map((cell) => cell.text) };
};

/** `text` (which starts at `offset` in the source) with the given source spans removed. */
const withoutSpans = (text: string, offset: number, spans: readonly Span[]): string => {
  let out = "";
  let cursor = 0;
  for (const [start, end] of spans) {
    out += text.slice(cursor, start - offset);
    cursor = end - offset;
  }
  return out + text.slice(cursor);
};

/** The comments-mode skeleton of `source` (a TypeScript module named `filename`). Throws when the source does not parse. */
export const skeletonOf = (filename: string, source: string): string =>
  analyze(filename, source, "comments").text;

/** One literal whose text differs between two versions with the same strings skeleton. */
export type LiteralChange = {
  readonly index: number;
  readonly before: string;
  readonly after: string;
  /** Whether the old text held Japanese — a change without it is a code change, not a translation. */
  readonly japaneseBefore: boolean;
};

/**
 * The literals that differ between two strings skeletons of the same text (the cells align one to
 * one when the texts are equal). Throws when the texts differ — the caller reports that as a code
 * change.
 */
export const literalChanges = (before: Skeleton, after: Skeleton): LiteralChange[] => {
  if (before.text !== after.text || before.literals.length !== after.literals.length) {
    throw new Error("code-skeleton: the skeletons differ; the literals do not align");
  }
  const changes: LiteralChange[] = [];
  before.literals.forEach((text, index) => {
    const next = after.literals[index];
    if (next === text) return;
    changes.push({ index, before: text, after: next, japaneseBefore: JAPANESE.test(text) });
  });
  return changes;
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

/** A literal's text on one line, shortened for the report. */
const shown = (text: string): string => {
  const flat = JSON.stringify(text);
  return flat.length > 120 ? `${flat.slice(0, 117)}…` : flat;
};

const main = (argv: readonly string[]): void => {
  const root = process.cwd();
  let base: string | undefined;
  let head: string | undefined;
  let mode: "comments" | "strings" = "comments";
  const files: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--compare") {
      base = argv[i + 1];
      i += 1;
    } else if (arg === "--head") {
      head = argv[i + 1];
      i += 1;
    } else if (arg === "--strings") mode = "strings";
    else files.push(arg);
  }
  if (base === undefined) {
    if (files.length !== 1) {
      throw new Error(
        "usage: code-skeleton.ts [--strings] --compare <ref> [--head <ref>] [file...] | <file>",
      );
    }
    console.log(analyze(files[0], readFileSync(join(root, files[0]), "utf8"), mode).text);
    return;
  }
  const targets = files.length > 0 ? files : changedTypeScriptFiles(root, base, head);
  let differing = 0;
  let changed = 0;
  let withoutJapanese = 0;
  for (const file of targets) {
    const before = gitShow(root, base, file);
    const after =
      head === undefined ? readFileSync(join(root, file), "utf8") : gitShow(root, head, file);
    if (before === undefined || after === undefined) {
      console.log(`${file}: added or removed (not comparable)`);
      differing += 1;
      continue;
    }
    const a = analyze(file, before, mode);
    const b = analyze(file, after, mode);
    const at = firstDifference(a.text, b.text);
    if (at >= 0) {
      differing += 1;
      console.log(`${file}: code differs near: …${a.text.slice(Math.max(0, at - 40), at + 60)}…`);
      continue;
    }
    if (mode !== "strings") continue;
    for (const change of literalChanges(a, b)) {
      changed += 1;
      if (!change.japaneseBefore) withoutJapanese += 1;
      const note = change.japaneseBefore ? "" : " (no Japanese before — a code change)";
      console.log(
        `${file}: literal ${change.index}${note}: ${shown(change.before)} -> ${shown(change.after)}`,
      );
    }
  }
  console.log(
    mode === "strings"
      ? `compared ${targets.length} files against ${base} (strings mode); ${changed} literals changed, ` +
          `${withoutJapanese} without Japanese before; ${differing} differ beyond comments and strings`
      : `compared ${targets.length} files against ${base}; ${differing} differ beyond comments`,
  );
  if (differing > 0 || withoutJapanese > 0) process.exitCode = 1;
};

if (import.meta.main) {
  main(process.argv.slice(2));
}
