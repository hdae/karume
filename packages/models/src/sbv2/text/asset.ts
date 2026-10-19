/**
 * Checks the tokenizer JSON of the run-time assets (the form `karume dist` ships) and turns it
 * into a {@link DebertaTokenizer}, and no further.
 *
 * It looks only at the structure of the asset and touches neither the GPU nor run-time state — its
 * dependencies are limited to `./tokenizer.ts` and the shared value-range checks of asset JSON
 * (`../../text/asset-gates.ts`).
 *
 * MUST: do not import `sbv2/pipeline.ts`. `text/` stays a leaf of pipeline in one direction (a
 * single dependency in the opposite direction makes it impossible to exercise the asset checks
 * alone without a GPU).
 */

import { isRecord } from "../../config/readers.ts";
import { MAX_CODE_POINT } from "../../text/asset-gates.ts";
import { type CleanRanges, DebertaTokenizer } from "./tokenizer.ts";

/**
 * Checks and reads the range table of `cleanRanges`.
 *
 * MUST: check integers, the code-point range, `start <= end`, and **ascending and
 * non-overlapping**. `inCodeRanges` (`text/code-ranges.ts`) is a binary search, so when this
 * premise breaks it raises no exception and **silently misses** — `inputIds` and `baseWord2ph` are
 * built from a `bertText` whose removal / spacing rules alone changed, broken in the same way, so
 * they also pass the length check in `text/model-input.ts`, and audio synthesised from different
 * BERT embeddings comes out as is. Do not rescue by sorting (fail at construction as an invalid
 * asset).
 */
const parseRanges = (raw: unknown, where: string): (readonly [number, number])[] => {
  if (!Array.isArray(raw)) throw new Error(`${where}: the range table is not an array`);
  let previousEnd = -1;
  return raw.map((entry, index) => {
    if (
      !Array.isArray(entry) ||
      entry.length !== 2 ||
      typeof entry[0] !== "number" ||
      typeof entry[1] !== "number"
    ) {
      throw new Error(`${where}[${index}]: the range is not a [start, end] pair of numbers`);
    }
    const [start, end]: [number, number] = [entry[0], entry[1]];
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > MAX_CODE_POINT) {
      throw new Error(
        `${where}[${index}]: range [${start}, ${end}] is not a pair of integers in 0..${MAX_CODE_POINT}`,
      );
    }
    if (start > end) {
      throw new Error(`${where}[${index}]: range [${start}, ${end}] has start greater than end`);
    }
    if (start <= previousEnd) {
      throw new Error(
        `${where}[${index}]: range [${start}, ${end}] starts at or below the previous end ${previousEnd}` +
          " (not ascending and non-overlapping — the binary search would silently miss)",
      );
    }
    previousEnd = end;
    return [start, end] as const;
  });
};

/** Upper bound of i32 (token ids are finally written to an `Int32Array`). */
const MAX_TOKEN_ID = 2147483647;

/**
 * Checks and reads the tokenizer JSON of the run-time assets (the form `karume dist` ships).
 *
 * MUST: check the structure before use. A broken vocabulary does not fail as "cannot read"; it
 * stays silent in the form **every token is `[UNK]`**, and only the BERT features quietly become
 * meaningless (the doc of `text/tokenizer.ts`).
 */
export const parseTokenizerAsset = (raw: unknown, where: string): DebertaTokenizer => {
  if (!isRecord(raw)) throw new Error(`${where}: not an object`);
  const special = raw["special"];
  if (!isRecord(special)) throw new Error(`${where}.special: not an object`);
  const [clsId, sepId, unkId] = [special["clsId"], special["sepId"], special["unkId"]];
  if (typeof clsId !== "number" || typeof sepId !== "number" || typeof unkId !== "number") {
    throw new Error(`${where}.special: clsId / sepId / unkId are not numbers`);
  }
  // MUST: an integer within the i32 range. ids are written to an `Int32Array`, so a non-integer is
  // **silently truncated** and an out-of-range value wraps — both are silent wrong values that
  // "point at a different token" (fitting within the vocabulary's row count is checked by
  // `fromVocabText`, the side that holds the vocabulary).
  for (const [name, id] of [
    ["clsId", clsId],
    ["sepId", sepId],
    ["unkId", unkId],
  ] as const) {
    if (!Number.isInteger(id) || id < 0 || id > MAX_TOKEN_ID) {
      throw new Error(`${where}.special.${name}: not an integer in 0..${MAX_TOKEN_ID} (${id})`);
    }
  }
  const vocabText = raw["vocabText"];
  if (typeof vocabText !== "string" || vocabText.length === 0) {
    throw new Error(`${where}.vocabText: empty`);
  }
  const clean = raw["cleanRanges"];
  if (!isRecord(clean)) throw new Error(`${where}.cleanRanges: not an object`);
  const ranges: CleanRanges = {
    removed: parseRanges(clean["removed"], `${where}.cleanRanges.removed`),
    spaced: parseRanges(clean["spaced"], `${where}.cleanRanges.spaced`),
  };
  return DebertaTokenizer.fromVocabText(vocabText, ranges, { clsId, sepId, unkId });
};
