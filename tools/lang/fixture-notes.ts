/**
 * Fixture notes: compares a JSON fixture against a git ref and proves that only its notes changed.
 * A note is a string under one of the note keys (`_doc`, `why`, `note`); everything else — keys and
 * their order, numbers, booleans, data strings — must be identical (ADR 0130, step 2-9g).
 *
 * Usage (from the repository root):
 *   node tools/lang/fixture-notes.ts --compare <git-ref> <file...>
 *       per file: the number of notes that changed and the notes that still hold Japanese;
 *       exit 1 when anything but a note differs or a note still holds Japanese
 */

import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { JAPANESE } from "./scan-japanese.ts";

export const NOTE_KEYS = new Set(["_doc", "why", "note"]);

export type NotesReport = {
  readonly changed: number;
  readonly japaneseLeft: readonly string[];
  readonly structuralDifference?: string;
};

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };

/** Walks both values in parallel; returns the first structural difference as a path, or undefined. */
const compare = (
  before: Json,
  after: Json,
  path: string,
  underNote: boolean,
  out: { changed: number; japaneseLeft: string[] },
): string | undefined => {
  if (typeof before === "string" && typeof after === "string") {
    if (underNote) {
      if (before !== after) out.changed += 1;
      if (JAPANESE.test(after)) out.japaneseLeft.push(`${path}: ${after.slice(0, 60)}`);
      return undefined;
    }
    return before === after ? undefined : path;
  }
  if (Array.isArray(before) && Array.isArray(after)) {
    if (before.length !== after.length)
      return `${path} (length ${before.length} → ${after.length})`;
    for (let i = 0; i < before.length; i += 1) {
      const diff = compare(before[i], after[i], `${path}[${i}]`, underNote, out);
      if (diff !== undefined) return diff;
    }
    return undefined;
  }
  if (
    before !== null &&
    after !== null &&
    typeof before === "object" &&
    typeof after === "object"
  ) {
    if (Array.isArray(before) || Array.isArray(after)) return path;
    const keysBefore = Object.keys(before);
    const keysAfter = Object.keys(after);
    if (keysBefore.join("\u0000") !== keysAfter.join("\u0000")) return `${path} (keys)`;
    for (const key of keysBefore) {
      const diff = compare(
        before[key],
        after[key],
        `${path}.${key}`,
        underNote || NOTE_KEYS.has(key),
        out,
      );
      if (diff !== undefined) return diff;
    }
    return undefined;
  }
  return before === after ? undefined : path;
};

export const compareNotes = (beforeText: string, afterText: string): NotesReport => {
  const out = { changed: 0, japaneseLeft: [] as string[] };
  const difference = compare(JSON.parse(beforeText), JSON.parse(afterText), "$", false, out);
  return difference === undefined
    ? { changed: out.changed, japaneseLeft: out.japaneseLeft }
    : { changed: out.changed, japaneseLeft: out.japaneseLeft, structuralDifference: difference };
};

const main = (argv: readonly string[]): void => {
  const at = argv.indexOf("--compare");
  const ref = at >= 0 ? argv[at + 1] : undefined;
  const files = argv.filter((a, i) => a !== "--compare" && i !== at + 1);
  if (ref === undefined || files.length === 0) {
    throw new Error("usage: fixture-notes.ts --compare <ref> <file...>");
  }
  let red = 0;
  for (const file of files) {
    const before = execFileSync("git", ["show", `${ref}:${file}`], { encoding: "utf8" });
    const report = compareNotes(before, readFileSync(file, "utf8"));
    if (report.structuralDifference !== undefined) {
      red += 1;
      console.log(`${file}: differs beyond notes at ${report.structuralDifference}`);
      continue;
    }
    if (report.japaneseLeft.length > 0) red += 1;
    console.log(
      `${file}: ${report.changed} notes changed, ${report.japaneseLeft.length} with Japanese left`,
    );
    for (const left of report.japaneseLeft) console.log(`  ${left}`);
  }
  if (red > 0) process.exitCode = 1;
};

if (import.meta.main) {
  main(process.argv.slice(2));
}
