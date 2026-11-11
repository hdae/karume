/**
 * The u32 identity comparison of two word arrays, with NaN handled by a declared policy — the one
 * comparison loop behind the kernel identity checks (`compareWords` in packages/kernel-checks) and
 * the fused-vs-unfused checks (`checkFusionTwin` there).
 *
 * Contract class E (ADR 0147 decision 2) is u32 equality. A NaN is the one word a backend may write
 * in more than one form: the payload and sign of a NaN an arithmetic operation produces are
 * backend-defined, so two kernels that compute the same values can still write different NaN words.
 * The policy says which reading a comparison makes: `"payload"` for words that are only moved (a copy
 * keeps every bit, NaN payloads included), `"class"` for words that are computed (a NaN must stay a
 * NaN, and every other word must be equal).
 */

/**
 * How a NaN word compares. `"payload"`: every word must be equal. `"class"`: where the expected word
 * is a NaN, the actual word must be a NaN of any payload and sign; every other word must be equal, so
 * an actual NaN against a finite or infinite expected word fails.
 */
export type NanPolicy = "class" | "payload";

/** One failing word: its index and both values as u32 words. */
export type WordMismatch = {
  readonly index: number;
  readonly expected: number;
  readonly actual: number;
};

/** The result of {@link wordIdentity}. */
export type WordIdentity = {
  /** The number of compared words: the length of `expected`. */
  readonly words: number;
  /**
   * The number of failing words. A length difference counts each missing word (on either side) as
   * one failure, so a longer `actual` can fail more words than `words`.
   */
  readonly failures: number;
  /**
   * The first failing word, present exactly when `failures > 0`. When only the lengths differ it is
   * the first index past the shorter array, with 0 for the side that has no word there.
   */
  readonly first?: WordMismatch;
};

/** Whether a u32 word is an f32 NaN (exponent all ones and a non-zero mantissa, either sign). */
const isNanWord = (word: number): boolean => (word & 0x7fffffff) > 0x7f800000;

const wordsMatch = (actual: number, expected: number, nan: NanPolicy): boolean =>
  actual === expected || (nan === "class" && isNanWord(expected) && isNanWord(actual));

/** Compares `actual` with `expected` word by word under the NaN policy `nan`. */
export const wordIdentity = (
  actual: Uint32Array<ArrayBuffer>,
  expected: Uint32Array<ArrayBuffer>,
  nan: NanPolicy,
): WordIdentity => {
  const common = Math.min(actual.length, expected.length);
  let failures = Math.abs(actual.length - expected.length);
  let first: WordMismatch | undefined;
  for (let i = 0; i < common; i += 1) {
    if (wordsMatch(actual[i], expected[i], nan)) continue;
    failures += 1;
    first ??= { index: i, expected: expected[i], actual: actual[i] };
  }
  if (failures > 0 && first === undefined) {
    first = { index: common, expected: expected[common] ?? 0, actual: actual[common] ?? 0 };
  }
  return first === undefined
    ? { words: expected.length, failures }
    : { words: expected.length, failures, first };
};
