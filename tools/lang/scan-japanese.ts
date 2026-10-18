/**
 * Counts the lines that carry Japanese text, per area of the repository — the progress measure of
 * ADR 0130 phase 2 and the basis of the language gate. Files come from `git ls-files`; a file with a
 * NUL byte in its first 8 KiB is treated as binary and skipped.
 *
 * Usage (from the repository root):
 *   node tools/lang/scan-japanese.ts [--list] [--root <dir>] [path-prefix...]
 *
 * `--list` prints every file with its count; path prefixes restrict the scan.
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

/** A character that marks a line as Japanese text (the three scripts and the common punctuation). */
export const JAPANESE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}、。「」・〈〉]/u;

export type ScanEntry = {
  readonly file: string;
  /** Lines of the file that contain at least one Japanese character. */
  readonly lines: number;
};

export type AreaTotal = {
  readonly area: string;
  readonly files: number;
  readonly lines: number;
};

/** The number of lines in `text` that carry Japanese text. */
export const countJapaneseLines = (text: string): number => {
  let count = 0;
  for (const line of text.split("\n")) {
    if (JAPANESE.test(line)) count += 1;
  }
  return count;
};

/**
 * The reporting area of a repository-relative path: the package or tool it belongs to, split into
 * `src` / `tests` / `tests/fixtures` for the packages, with `docs/` split into the decision record,
 * the research snapshots, the Japanese glossary and the rest.
 */
export const areaOf = (file: string): string => {
  const parts = file.split("/");
  if (parts[0] === "docs") {
    if (parts[1] === "decisions") return "docs/decisions";
    if (parts[1] === "research") return "docs/research";
    if (parts[1] === "GLOSSARY.ja.md") return "docs/GLOSSARY.ja.md";
    return "docs (other)";
  }
  if (parts[0] === "packages" && parts.length > 2) {
    const base = `packages/${parts[1]}`;
    if (parts[2] === "src") return `${base}/src`;
    if (parts[2] === "tests")
      return parts[3] === "fixtures" ? `${base}/tests/fixtures` : `${base}/tests`;
    return base;
  }
  if (parts[0] === "tools" && parts.length > 2) {
    if (parts[1] === "exporter")
      return parts[2] === "tests" ? "tools/exporter/tests" : "tools/exporter";
    return `tools/${parts[1]}`;
  }
  if (parts.length === 1) return "root";
  return parts[0];
};

const BINARY_PROBE_BYTES = 8192;

const isBinary = (bytes: Buffer): boolean => {
  const probe = bytes.subarray(0, BINARY_PROBE_BYTES);
  return probe.includes(0);
};

/** Tracked files under `root` (from `git ls-files`), optionally restricted to the given prefixes. */
export const trackedFiles = (root: string, prefixes: readonly string[] = []): string[] => {
  const output = execFileSync("git", ["-C", root, "ls-files", "-z", "--", ...prefixes], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  return output.split("\0").filter((file) => file.length > 0);
};

/** Scans the given repository-relative files; binary and unreadable-as-text files are skipped. */
export const scanFiles = (root: string, files: readonly string[]): ScanEntry[] => {
  const entries: ScanEntry[] = [];
  for (const file of files) {
    let bytes: Buffer;
    try {
      bytes = readFileSync(join(root, file));
    } catch {
      continue; // a tracked path that is not a regular file on disk (submodule, missing symlink target)
    }
    if (isBinary(bytes)) continue;
    const lines = countJapaneseLines(bytes.toString("utf8"));
    if (lines > 0) entries.push({ file, lines });
  }
  return entries;
};

/** Totals per area, sorted by area name. */
export const totalsByArea = (entries: readonly ScanEntry[]): AreaTotal[] => {
  const totals = new Map<string, { files: number; lines: number }>();
  for (const entry of entries) {
    const area = areaOf(entry.file);
    const total = totals.get(area) ?? { files: 0, lines: 0 };
    total.files += 1;
    total.lines += entry.lines;
    totals.set(area, total);
  }
  return [...totals.entries()]
    .map(([area, total]) => ({ area, files: total.files, lines: total.lines }))
    .sort((a, b) => a.area.localeCompare(b.area));
};

const main = (argv: readonly string[]): void => {
  let root = process.cwd();
  let list = false;
  const prefixes: string[] = [];
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--list") list = true;
    else if (arg === "--root") {
      root = argv[i + 1] ?? root;
      i += 1;
    } else prefixes.push(arg);
  }
  const entries = scanFiles(root, trackedFiles(root, prefixes));
  if (list) {
    for (const entry of [...entries].sort((a, b) => a.file.localeCompare(b.file))) {
      console.log(`${String(entry.lines).padStart(6)}  ${entry.file}`);
    }
  }
  const totals = totalsByArea(entries);
  const width = Math.max(4, ...totals.map((total) => total.area.length));
  console.log(`${"area".padEnd(width)}  files   lines`);
  for (const total of totals) {
    console.log(
      `${total.area.padEnd(width)}  ${String(total.files).padStart(5)}  ${String(total.lines).padStart(6)}`,
    );
  }
  const files = entries.length;
  const lines = entries.reduce((sum, entry) => sum + entry.lines, 0);
  console.log(
    `${"total".padEnd(width)}  ${String(files).padStart(5)}  ${String(lines).padStart(6)}`,
  );
};

if (import.meta.main) {
  main(process.argv.slice(2));
}
