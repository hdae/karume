/**
 * Checks that relative Markdown links and the `DECIDED:` / `SPEC:` pointers of source files resolve: the
 * target file exists and, when the link carries a `#fragment`, the target has a heading with that slug
 * (GitHub's slug rule). A renamed file or a reworded heading turns up here before it turns up as a dead
 * link.
 *
 * Usage (from the repository root):
 *   node tools/lang/check-doc-links.ts [--root <dir>]      exit 1 when anything is broken
 *
 * Scope: every tracked `*.md` file, and every `DECIDED:` line and `SPEC: docs/spec/…` pointer in tracked
 * `*.ts` and `*.py` files. Links under the local asset roots (`models/`, `outputs/`, `inputs/` —
 * untracked by design) are skipped and counted. `tests/repository_links_test.ts` runs the same check on
 * the repository in the core lane.
 */

import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, normalize, relative, resolve } from "node:path";

export type BrokenLink = {
  readonly file: string;
  readonly line: number;
  readonly target: string;
  readonly reason: "missing file" | "missing heading" | "unknown ADR";
};

export type LinkReport = {
  readonly checked: number;
  readonly skipped: number;
  readonly broken: readonly BrokenLink[];
};

const SCHEME = /^[a-z][a-z0-9+.-]*:/i;
const LOCAL_ASSET_ROOTS = ["models", "outputs", "inputs"];

/** GitHub's heading slug: lower case, letters / numbers / spaces / hyphens kept, spaces to hyphens. */
export const slugOf = (heading: string): string =>
  heading
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s/g, "-");

/** Every heading slug of a Markdown text, with GitHub's `-1`, `-2` suffixes for duplicates. */
export const headingSlugs = (markdown: string): Set<string> => {
  const seen = new Map<string, number>();
  const slugs = new Set<string>();
  let inFence = false;
  for (const line of markdown.split("\n")) {
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const match = /^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/.exec(line);
    if (match === null) continue;
    const base = slugOf(match[1]);
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    slugs.add(count === 0 ? base : `${base}-${count}`);
  }
  return slugs;
};

type Occurrence = { readonly line: number; readonly target: string };

const LINK = /!?\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+"[^"]*")?\s*\)/g;
const REFERENCE_DEFINITION = /^\s{0,3}\[[^\]]+\]:\s*<?(\S+)>?/;

/** Link targets of a Markdown text (inline links and reference definitions), skipping fenced code. */
export const linkTargets = (markdown: string): Occurrence[] => {
  const targets: Occurrence[] = [];
  let inFence = false;
  const lines = markdown.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*(```|~~~)/.test(line)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const definition = REFERENCE_DEFINITION.exec(line);
    if (definition !== null) targets.push({ line: index + 1, target: definition[1] });
    for (const match of line.matchAll(LINK)) {
      // A target without `.`, `/` or `#` is prose that merely looks like a link (`E[a](k=3)`).
      if (!/[./#]/.test(match[1])) continue;
      targets.push({ line: index + 1, target: match[1] });
    }
  }
  return targets;
};

const DECIDED_PATH = /docs\/decisions\/([0-9]{4}-[\w.-]+?\.md)(#[^\s)）、。」]+)?/g;
const DECIDED_ADR = /\bADR\s*0*([0-9]{1,4})\b/g;

/** `DECIDED:` pointers of a source text: decision-record paths and `ADR nnnn` numbers. */
export const decidedPointers = (source: string): Occurrence[] => {
  const targets: Occurrence[] = [];
  const lines = source.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const at = line.indexOf("DECIDED:");
    if (at < 0) continue;
    const rest = line.slice(at);
    for (const match of rest.matchAll(DECIDED_PATH)) {
      targets.push({ line: index + 1, target: `docs/decisions/${match[1]}${match[2] ?? ""}` });
    }
    const withoutPaths = rest.replace(DECIDED_PATH, "");
    for (const match of withoutPaths.matchAll(DECIDED_ADR)) {
      targets.push({ line: index + 1, target: `adr:${match[1].padStart(4, "0")}` });
    }
  }
  return targets;
};

// A pointer to the rule in force is the marker followed by a path under `docs/spec/`. Requiring the path
// in the pattern keeps an identifier such as `const SPEC: Type` from reading as a pointer.
const SPEC_POINTER = /\bSPEC:\s+(docs\/spec\/[\w./-]+?\.md)(#[\p{L}\p{N}-]+)?/gu;

/** `SPEC:` pointers of a source text: paths under `docs/spec/`, with their heading fragment if any. */
export const specPointers = (source: string): Occurrence[] => {
  const targets: Occurrence[] = [];
  const lines = source.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    for (const match of lines[index].matchAll(SPEC_POINTER)) {
      targets.push({ line: index + 1, target: `${match[1]}${match[2] ?? ""}` });
    }
  }
  return targets;
};

const headingCache = new Map<string, Set<string>>();

const slugsOfFile = (path: string): Set<string> => {
  const cached = headingCache.get(path);
  if (cached !== undefined) return cached;
  const slugs = headingSlugs(readFileSync(path, "utf8"));
  headingCache.set(path, slugs);
  return slugs;
};

/** Decision records by number (`docs/decisions/<nnnn>-*.md`), read from the directory, not from git. */
const adrFiles = (root: string): Map<string, string> => {
  const byNumber = new Map<string, string>();
  const directory = join(root, "docs", "decisions");
  if (!existsSync(directory)) return byNumber;
  for (const file of readdirSync(directory)) {
    const match = /^([0-9]{4})-.*\.md$/.exec(file);
    if (match !== null) byNumber.set(match[1], `docs/decisions/${file}`);
  }
  return byNumber;
};

const underLocalAssets = (root: string, absolute: string): boolean => {
  const rel = relative(root, absolute).split(/[\\/]/)[0];
  return LOCAL_ASSET_ROOTS.includes(rel);
};

/** Tracked files under `root` (from `git ls-files`), optionally restricted to the given prefixes. */
export const trackedFiles = (root: string, prefixes: readonly string[] = []): string[] => {
  const output = execFileSync("git", ["-C", root, "ls-files", "-z", "--", ...prefixes], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  return output.split("\0").filter((file) => file.length > 0);
};

/**
 * Checks the links of the given Markdown files and the `DECIDED:` / `SPEC:` pointers of the given source
 * files (both repository-relative). Exported with explicit file lists so that a test can run it on a
 * temporary directory without git.
 */
export const checkLinks = (
  root: string,
  markdownFiles: readonly string[],
  sourceFiles: readonly string[] = [],
): LinkReport => {
  headingCache.clear();
  const broken: BrokenLink[] = [];
  let checked = 0;
  let skipped = 0;
  const adrs = adrFiles(root);

  // Markdown links are relative to their file; `DECIDED:` and `SPEC:` pointers are repository-relative paths.
  const checkTarget = (file: string, line: number, target: string, base: string): void => {
    if (target.startsWith("adr:")) {
      checked += 1;
      if (!adrs.has(target.slice(4))) broken.push({ file, line, target, reason: "unknown ADR" });
      return;
    }
    if (SCHEME.test(target)) return;
    const [pathPart, fragment] = target.split("#", 2);
    const decoded = decodeURIComponent(pathPart);
    const absolute = decoded.length === 0 ? join(root, file) : resolve(base, decoded);
    if (underLocalAssets(root, absolute)) {
      skipped += 1;
      return;
    }
    checked += 1;
    if (!existsSync(absolute)) {
      broken.push({ file, line, target, reason: "missing file" });
      return;
    }
    if (fragment === undefined || fragment.length === 0) return;
    if (statSync(absolute).isDirectory() || !absolute.endsWith(".md")) return;
    const slug = decodeURIComponent(fragment).toLowerCase();
    if (!slugsOfFile(absolute).has(slug)) {
      broken.push({ file, line, target, reason: "missing heading" });
    }
  };

  for (const file of markdownFiles) {
    const text = readFileSync(join(root, file), "utf8");
    const base = dirname(join(root, file));
    for (const { line, target } of linkTargets(text)) checkTarget(file, line, target, base);
  }
  for (const file of sourceFiles) {
    const text = readFileSync(join(root, file), "utf8");
    for (const { line, target } of decidedPointers(text)) checkTarget(file, line, target, root);
    for (const { line, target } of specPointers(text)) checkTarget(file, line, target, root);
  }
  return { checked, skipped, broken };
};

export type RepositoryLinkReport = LinkReport & {
  readonly markdownFiles: number;
  readonly sourceFiles: number;
};

/** The check over every tracked Markdown file and every tracked `*.ts` / `*.py` file under `root`. */
export const checkRepository = (root: string): RepositoryLinkReport => {
  const tracked = trackedFiles(root);
  const markdown = tracked.filter((file) => file.endsWith(".md"));
  const sources = tracked.filter((file) => file.endsWith(".ts") || file.endsWith(".py"));
  return {
    ...checkLinks(root, markdown, sources),
    markdownFiles: markdown.length,
    sourceFiles: sources.length,
  };
};

/** One line per broken link: where it is, why it is broken, what it points at. */
export const describeBroken = (broken: readonly BrokenLink[]): string[] =>
  broken.map((link) => `${link.file}:${link.line}: ${link.reason}: ${link.target}`);

const main = (argv: readonly string[]): void => {
  let root = process.cwd();
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--root") {
      root = resolve(argv[i + 1] ?? root);
      i += 1;
    }
  }
  const report = checkRepository(normalize(root));
  for (const line of describeBroken(report.broken)) console.log(line);
  console.log(
    `checked ${report.checked} links in ${report.markdownFiles} Markdown files and DECIDED: / SPEC: pointers in ${report.sourceFiles} source files; skipped ${report.skipped} under local asset roots; broken ${report.broken.length}`,
  );
  if (report.broken.length > 0) process.exitCode = 1;
};

if (import.meta.main) {
  main(process.argv.slice(2));
}
