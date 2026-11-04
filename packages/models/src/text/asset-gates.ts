/**
 * Value-range checks for asset JSON (an external boundary) — gathered into one place across
 * families.
 *
 * Distribution assets are baked by the exporter, but `fromPretrained(ref)` can point at any repo,
 * so asset JSON is treated as **untrusted external input**. What this pins down is only the shape
 * in which "a broken table raises no exception and quietly turns into another token sequence /
 * another embedding row" (silently wrong values):
 *
 *   - a table where line number = id with duplicate lines makes **earlier ids unreachable** through
 *     `Map`'s later-wins
 *   - duplicate keys of a keyed map are later-wins too, and only normalization / character
 *     classification changes quietly
 *   - a value that is not a code point entering a range table or a composite key silently breaks
 *     binary search or key uniqueness
 *   - length / frequency fields that are 0 or negative still pass truncation and matching looking
 *     like "another normal"
 *   - a non-integer / out-of-i32 id is truncated or wraps when written to an `Int32Array`, pointing
 *     at another row
 *
 * The granularity of these checks varied per family (strictest: sbv2's `parseRanges` /
 * `symbols.ts`; loosest: anima's text assets), so they were gathered here **aligned to the strict
 * side**.
 */

export const asRecord = (value: unknown, label: string): Record<string, unknown> => {
  if (typeof value !== "object" || value === null) throw new Error(`${label}: not an object`);
  return value as Record<string, unknown>;
};

export const asString = (value: unknown, label: string): string => {
  if (typeof value !== "string") throw new Error(`${label}: not a string`);
  return value;
};

export const asNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number") throw new Error(`${label}: not a number`);
  return value;
};

/**
 * Upper bound of Unicode code points (both range tables and composite keys are keyed by code
 * point).
 */
export const MAX_CODE_POINT = 0x10ffff;

/**
 * A number accepted as a code point.
 *
 * MUST: do not accept on `typeof number` alone. Range tables are looked up by binary search and
 * maps by composite keys such as `cp * 0x110000 + cp` — both turn non-integer or out-of-range
 * values into "no hit / a hit on another key" **without an exception**, so only character
 * classification and normalization change quietly.
 */
export const assertCodePoint = (cp: number, label: string): void => {
  if (!Number.isInteger(cp) || cp < 0 || cp > MAX_CODE_POINT) {
    throw new Error(`${label}: code point is not an integer in 0..${MAX_CODE_POINT} (${cp})`);
  }
};

/** Upper bound of i32 (ids are eventually written to an `Int32Array`). */
const MAX_ID = 2147483647;

/**
 * A number accepted as a token id.
 *
 * MUST: do not accept on `typeof number` alone. The encoded ids are written to an `Int32Array`, so
 * a non-integer is **silently truncated** and an out-of-i32 value wraps — both are silently wrong
 * values that "point at another token" and do not surface before the graph's embedding gather.
 */
export const asId = (value: unknown, label: string): number => {
  const id = asNumber(value, label);
  if (!Number.isInteger(id) || id < 0 || id > MAX_ID) {
    throw new Error(`${label}: token id is not an integer in 0..${MAX_ID} (${id})`);
  }
  return id;
};

/**
 * An id pointing at a vocabulary row ({@link asId} plus being below the vocabulary's row count).
 */
export const asVocabId = (value: unknown, label: string, vocabSize: number): number => {
  const id = asId(value, label);
  if (id >= vocabSize) {
    throw new Error(`${label}: token id ${id} is not below the vocabulary row count ${vocabSize}`);
  }
  return id;
};

/**
 * A number accepted as a length, count or frequency (a positive safe integer).
 *
 * MUST: check the upper bound too. Above 2^53, additions and comparisons lose precision and
 * silently become other values, and used as a `slice` bound, "truncation that never takes effect"
 * passes looking like a normal value.
 */
export const asPositiveInteger = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${label}: not a positive safe integer (${String(value)})`);
  }
  return value;
};

/**
 * A number accepted as a real value, such as a score.
 *
 * MUST: check finiteness too. `NaN` / `±Infinity` pollute the min / max folds, and Viterbi's
 * unknown-node weight and search width collapse into "every path is equal" (no exception).
 */
export const asFiniteNumber = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label}: not a finite number (${String(value)})`);
  }
  return value;
};

/**
 * Checks that a table where line number = id has no duplicate lines.
 *
 * MUST: matching the line count is not enough — duplicates do not change the line count, so they
 * pass even against the count of `vocabCount` / `scores`. Past that point `Map`'s later-wins makes
 * the earlier id unreachable, and the embedding gather reads **another, legal row** (a distribution
 * corruption that raises no exception).
 */
export const assertUniqueLines = (lines: readonly string[], label: string): void => {
  const firstAt = new Map<string, number>();
  for (const [index, line] of lines.entries()) {
    const seen = firstAt.get(line);
    if (seen !== undefined) {
      throw new Error(
        `${label}: line ${index} has the same token as line ${seen}: ${JSON.stringify(line)}` +
          " (line number = id, so a duplicate is a corrupt distribution)",
      );
    }
    firstAt.set(line, index);
  }
};

/**
 * Puts one entry into a map. Fails if the same key is already there (forbids `Map.set`'s
 * later-wins).
 *
 * MUST: do not rescue duplicate keys by "normalizing". BPE merge order, added vocabularies and
 * normalization tables all assume unique keys; overwritten by later-wins, only the segmentation
 * rule changes quietly.
 */
export const setUnique = <K, V>(map: Map<K, V>, key: K, value: V, label: string): void => {
  if (map.has(key)) {
    throw new Error(
      `${label}: duplicate key ${JSON.stringify(key)} (the later entry would silently overwrite the earlier one)`,
    );
  }
  map.set(key, value);
};

/** Added vocabulary (a sequence of `[string, id]` pairs) into a map. */
export const parseAddedTokens = (raw: unknown, label: string): Map<string, number> => {
  if (!Array.isArray(raw)) throw new Error(`${label}: not an array`);
  const out = new Map<string, number>();
  for (const entry of raw) {
    if (!Array.isArray(entry) || entry.length !== 2) {
      throw new Error(`${label}: not [string, id]`);
    }
    // NOTE: not bounded by the vocabulary row count — added tokens can be numbered **outside** the
    // vocabulary table (anima's Qwen2 actually does this: the real asset has 151643..151668 against
    // a 151643-row vocabulary). All that is checked here is that it is sound as an i32.
    setUnique(out, asString(entry[0], label), asId(entry[1], label), label);
  }
  return out;
};

/**
 * Added vocabulary whose every token is a row of the vocabulary table (Irodori and the T5 family —
 * anima's T5 and Wan's umT5): {@link parseAddedTokens} plus the match against that table.
 *
 * MUST: an id is below the vocabulary row count. The shared {@link parseAddedTokens} checks only
 * the i32 range (anima's Qwen2 legitimately numbers tokens outside its vocabulary table — the shared
 * side is not narrowed), but these families' added tokens are all rows of the vocabulary table, and
 * encode pushes the mapped values into the graph input as they are. An out-of-range id that gets
 * through does not surface until the OOB of the embedding gather (non-finite — ADR 0061).
 * MUST: if the vocabulary has a row with the same spelling, the id equals that row number. On a
 * mismatch, only the spellings split out as added tokens point at a different row than the
 * vocabulary table, a silent wrong value (it passes with matching shape and range).
 */
export const parseAddedTokensInVocab = (
  raw: unknown,
  label: string,
  vocab: ReadonlyMap<string, { readonly id: number }>,
): Map<string, number> => {
  const added = parseAddedTokens(raw, label);
  for (const [token, id] of added) {
    if (id >= vocab.size) {
      throw new Error(
        `${label}: ${JSON.stringify(token)} has id ${id}, which is >= the vocabulary row count ${vocab.size}`,
      );
    }
    const row = vocab.get(token)?.id;
    if (row !== undefined && row !== id) {
      throw new Error(
        `${label}: ${JSON.stringify(token)} has id ${id}, which disagrees with vocabulary row ${row}`,
      );
    }
  }
  return added;
};
