/**
 * Reproduces the `Precompiled` normaliser of T5 (SentencePiece) by looking up baked tables only.
 *
 * The canonical implementation is Rust's `tokenizers::normalizers::Precompiled`. Its actual
 * behaviour differs from SentencePiece normalisation and has the shape **whole-replacement per
 * grapheme cluster + shortest prefix wins**:
 *
 *     for each cluster g:
 *       UTF-8 length(g) < 6 and some prefix of g is in the rule table
 *         → emit the value of the **shortest** matching prefix and drop the rest of g
 *       otherwise look up one code point at a time (a character without a rule stays as is)
 *
 * Measured:
 *   `A`+U+0301+U+0301 → `Á` (the third character disappears) / `A`+U+0302+U+0301 → `Â`
 *   (the 2-cp prefix wins over the 3-cp rule `Ấ`). A naive "longest match" or NFKC implementation
 *   is certain to fail on these two examples.
 *
 * **Do not decode the DARTS trie in TS / do not implement UAX#29 in TS**. The exporter applies the
 * canonical implementation to every code point and folds the result into
 *   - only the rules that can fire (no rule on a proper prefix, under 6 UTF-8 bytes)
 *   - the three closed-interval tables needed for cluster boundaries (extend / breakAfter /
 *     prepend)
 * which is what this module looks up. The equivalence of the folding is checked exhaustively plus
 * by random sampling at every emit.
 */

import { assertCodePoint, setUnique } from "../../text/asset-gates.ts";
import { toCodePoints } from "../../text/code-points.ts";
import { type CodeRanges, inCodeRanges, parseCodeRanges } from "../../text/code-ranges.ts";
import { utf8Length } from "./code-ranges.ts";

/**
 * Upper UTF-8 length of a cluster that tries whole replacement (the canonical spm_precompiled's
 * `len() < 6`).
 */
const CLUSTER_BYTE_LIMIT = 6;

const CR = 0x0d;
const LF = 0x0a;

/** Combined key of a 2-code-point key (code points are at most 0x10FFFF, so no collision). */
const pairKey = (first: number, second: number): number => first * 0x110000 + second;

/** Baked normalisation tables. */
export type SpmTables = {
  /** Mapping of 1-code-point keys. */
  readonly single: ReadonlyMap<number, string>;
  /** Mapping of 2-code-point keys (only those without a rule on a proper prefix). */
  readonly multi: ReadonlyMap<number, string>;
  /** Characters that join the cluster of the preceding character. */
  readonly extend: CodeRanges;
  /** Characters after which a cluster always ends. */
  readonly breakAfter: CodeRanges;
  /** Characters that pull the following character into their cluster. */
  readonly prepend: CodeRanges;
};

/** Reproduces `Precompiled` from the baked tables alone. */
export const normalizeSpm = (tables: SpmTables, text: string): string => {
  const cps = toCodePoints(text);
  const out: string[] = [];
  let i = 0;
  while (i < cps.length) {
    const end = clusterEnd(tables, cps, i);
    let bytes = 0;
    for (let k = i; k < end; k++) bytes += utf8Length(cps[k]);
    let replacement: string | undefined;
    // MUST: dropping the 6-byte boundary silently breaks emoji / Hangul (clusters that should not
    // enter the whole-replacement path enter it).
    if (bytes < CLUSTER_BYTE_LIMIT) {
      // The shortest matching prefix wins — look up 1 cp first.
      replacement = tables.single.get(cps[i]);
      if (replacement === undefined && end - i >= 2) {
        replacement = tables.multi.get(pairKey(cps[i], cps[i + 1]));
      }
    }
    if (replacement !== undefined) {
      out.push(replacement);
    } else {
      for (let k = i; k < end; k++) {
        out.push(tables.single.get(cps[k]) ?? String.fromCodePoint(cps[k]));
      }
    }
    i = end;
  }
  return out.join("");
};

/**
 * End (exclusive) of the cluster starting at i. prepend* + base + extend* / CRLF / a lone control.
 */
const clusterEnd = (tables: SpmTables, cps: readonly number[], i: number): number => {
  let j = i;
  while (j < cps.length && inCodeRanges(tables.prepend, cps[j])) j++;
  // A control character right after prepend ends the cluster there (not pulled in).
  if (j < cps.length && !(j > i && inCodeRanges(tables.breakAfter, cps[j]))) {
    if (cps[j] === CR && cps[j + 1] === LF) {
      j += 2; // GB3: CR × LF is one cluster
    } else if (inCodeRanges(tables.breakAfter, cps[j])) {
      j += 1;
    } else {
      j += 1;
      while (j < cps.length && inCodeRanges(tables.extend, cps[j])) j++;
    }
  }
  return j;
};

/**
 * JSON form of the asset / fixture (the counterpart of Python's `SpmTables.to_json`).
 *
 * MUST: check the code-point range of the keys and duplicates too. The uniqueness of
 * {@link pairKey} assumes "keys are at most 0x10FFFF"; an out-of-range value collapses into the
 * same combined key as another pair. Duplicate keys also pass with `Map`'s last-wins, and neither
 * throws: **only the normalisation result** changes silently (the tokenize id sequence becomes a
 * different one).
 */
export const parseSpmTables = (raw: unknown, label: string): SpmTables => {
  if (typeof raw !== "object" || raw === null) throw new Error(`${label}: not an object`);
  const obj = raw as Record<string, unknown>;
  const single = new Map<number, string>();
  for (const entry of asArray(obj["single"], `${label}.single`)) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "string"
    ) {
      throw new Error(`${label}.single: not [cp, string]`);
    }
    assertCodePoint(entry[0], `${label}.single`);
    setUnique(single, entry[0], entry[1], `${label}.single`);
  }
  const multi = new Map<number, string>();
  for (const entry of asArray(obj["multi"], `${label}.multi`)) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 3 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "number" ||
      typeof entry[2] !== "string"
    ) {
      throw new Error(`${label}.multi: not [cp, cp, string]`);
    }
    assertCodePoint(entry[0], `${label}.multi`);
    assertCodePoint(entry[1], `${label}.multi`);
    setUnique(multi, pairKey(entry[0], entry[1]), entry[2], `${label}.multi`);
  }
  return {
    single,
    multi,
    extend: parseCodeRanges(obj["extend"], `${label}.extend`),
    breakAfter: parseCodeRanges(obj["breakAfter"], `${label}.breakAfter`),
    prepend: parseCodeRanges(obj["prepend"], `${label}.prepend`),
  };
};

const asArray = (value: unknown, label: string): unknown[] => {
  if (!Array.isArray(value)) throw new Error(`${label}: not an array`);
  return value;
};
