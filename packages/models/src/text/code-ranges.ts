/**
 * Tables of closed code-point ranges (ascending, non-overlapping), binary search over them, and
 * reading them from asset JSON.
 *
 * The text layer does not reimplement Unicode classification in TS. The source of truth for
 * classification is the exporter-side Python / Rust (`tokenizers`), and this only looks up the
 * closed-range tables that **fold the actual evaluation of every code point** there. With the check
 * implemented in this one place, classification cannot drift as long as the tables are right.
 *
 * Reading ({@link parseCodeRanges}) also lives here across families — the two families anima and
 * Wan read tables of the same shape from their assets (in a family directory, another family using
 * it would mean a cross-family import). sbv2's `parseRanges` has different check messages and is
 * kept separately.
 */

import { assertCodePoint } from "./asset-gates.ts";

/** Ascending list of code-point ranges including both ends. */
export type CodeRanges = readonly (readonly [number, number])[];

/** Whether cp is contained in the range table. */
export const inCodeRanges = (ranges: CodeRanges, cp: number): boolean => {
  let low = 0;
  let high = ranges.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const [start, end] = ranges[mid];
    if (cp < start) high = mid - 1;
    else if (cp > end) low = mid + 1;
    else return true;
  }
  return false;
};

/**
 * Structural check of an external boundary (asset JSON). A broken table is never silently used as
 * an empty table.
 *
 * MUST: check **integers, code-point range, ascending order and no overlap**. `inCodeRanges` is a
 * binary search, so if these premises break it raises no exception and becomes "a quietly different
 * character classification" (`[65.5, 90]` moves only a boundary to a fractional position, and a table out of order
 * drops one range from the `\p{L}` check — both change where pre-tokens split and give a different
 * id sequence). The value-range discipline is the same as sbv2's isomorphic `parseRanges`.
 */
export const parseCodeRanges = (raw: unknown, label: string): CodeRanges => {
  if (!Array.isArray(raw)) throw new Error(`${label}: 区間表が配列でない`);
  const ranges: (readonly [number, number])[] = [];
  for (const [index, entry] of raw.entries()) {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "number"
    ) {
      throw new Error(`${label}[${index}]: 区間が [start, end] の数値対でない`);
    }
    assertCodePoint(entry[0], `${label}[${index}]`);
    assertCodePoint(entry[1], `${label}[${index}]`);
    if (entry[0] > entry[1]) {
      throw new Error(`${label}[${index}]: 区間 [${entry[0]}, ${entry[1]}] の始端が終端より大きい`);
    }
    const previous = ranges[index - 1];
    if (previous !== undefined && previous[1] >= entry[0]) {
      throw new Error(
        `${label}[${index}]: 区間 [${entry[0]}, ${entry[1]}] が前の区間 ` +
          `[${previous[0]}, ${previous[1]}] と重なる / 昇順でない`,
      );
    }
    ranges.push([entry[0], entry[1]]);
  }
  return ranges;
};
