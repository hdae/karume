/**
 * Host implementation that generates, at the actual length, the relative-position index tables used
 * by DeBERTa-v2's disentangled attention.
 *
 * ## Why the host builds the tables
 *
 * These 2 tables are constants determined by T alone, so the exporter's constant folding **baked in
 * two** `[1,512,512]` i32 tables at Tmax = 512 (measured 2,097,152 B = 0.68% of the distribution).
 * Their content is Toeplitz and the two are transposes of each other, so in substance they equal an
 * i32 vector of 1,023 elements. Following the precedent of flow / voice, which promoted their
 * tables to graph inputs for the same reason (ADR 0013), they were moved to external supply (ADR
 * 0045 wave 3).
 *
 * ## MUST: byte-identical to the Python side
 *
 * The canonical formula is `build_rel_pos_tables` in `tools/export-recipes/deberta/patch.py`. Even
 * if the formulas diverge, **the shapes still match**, so the model does not fail and silently
 * gathers different position embeddings (a silent wrong value).
 * `packages/models/tests/sbv2_rel_pos_parity_test.ts` pins byte equality with the Python-generated
 * tables stored in the golden io, on real data.
 *
 * The original formula is transformers' `make_log_bucket_position`, and torch computes it in
 * **float32**. Agreement with the host's float64 has been confirmed by measurement (0 mismatches at
 * every point of T = 11 / 35 / 128 / 300 / 512 — ADR 0045 wave 3). This is the empirical claim that
 * no float32/float64 difference appears at the boundaries of log and ceil, so re-measure when
 * `bucketSize` / `maxPosition` change.
 */

import type { Tensor } from "@karume/runtime";

/**
 * Return value of {@link buildRelPosTables} (a form that can be passed as is to the graph inputs
 * `c2p_pos` / `p2c_pos`).
 */
type RelPosTables = {
  /** content→position gather index. `c2p[i][j] = clamp(bucket(i − j) + span, 0, 2·span − 1)`. */
  readonly c2pPos: Tensor;
  /**
   * position→content gather index. **The transpose of `c2p`** (exact, because bucketing is an odd
   * function).
   */
  readonly p2cPos: Tensor;
};

/**
 * Checks whether `(bucketSize, maxPosition)` is a pair for which the log-bucket formula holds.
 *
 * MUST: `mid = ⌊bucketSize / 2⌋ ≥ 2` and `maxPosition − 1 > mid`. Breaking the former makes the
 * far-range coefficient `mid − 1` zero; breaking the latter makes the log denominator
 * `log((maxPosition − 1) / mid)` zero or negative, and `(2, 2)` produces `Infinity × 0 = NaN`. NaN
 * slips through the comparisons of clamp and becomes 0 in an Int32Array, giving a wrong gather
 * index of the right shape (a silent wrong value without an exception). Upstream transformers uses
 * the same formula without a check, so this is an entry check added for the asset.
 *
 * Both the asset parser (`bertRelPos` in `symbols.ts`) and {@link buildRelPosTables} go through
 * this one function.
 */
export const assertLogBucketDomain = (
  bucketSize: number,
  maxPosition: number,
  where: string,
): void => {
  const mid = Math.floor(bucketSize / 2);
  if (mid < 2 || maxPosition - 1 <= mid) {
    throw new RangeError(
      `${where}: the log-bucket formula does not hold with bucketSize ${bucketSize} / maxPosition ${maxPosition}` +
        ` (bucketSize ≥ 4 and maxPosition − 1 > ⌊bucketSize / 2⌋ are required)`,
    );
  }
};

/**
 * Buckets `rel` with the same formula as `make_log_bucket_position` (transformers).
 *
 * The near range (`|rel| < mid`) stays linear and the far range is compressed logarithmically. The
 * `abs_pos <= mid` branch overlaps with the assignment of `mid - 1`, so the near range always
 * returns `rel` itself.
 */
const logBucketPosition = (rel: number, bucketSize: number, maxPosition: number): number => {
  const mid = Math.floor(bucketSize / 2);
  const absPos = rel < mid && rel > -mid ? mid - 1 : Math.abs(rel);
  if (absPos <= mid) return rel;
  const sign = Math.sign(rel);
  const logPos =
    Math.ceil((Math.log(absPos / mid) / Math.log((maxPosition - 1) / mid)) * (mid - 1)) + mid;
  return logPos * sign;
};

/**
 * Builds the 2 index tables for actual length `length`.
 *
 * `bucketSize` / `maxPosition` come from the distribution's `symbols.json` (`bertRelPos`) — so that
 * values from DeBERTa's config are not copied onto the host (the same discipline as ADR 0039
 * decision 3).
 *
 * The tables depend only on `i − j` (Toeplitz), so the `2·length − 1` per-diagonal values are built
 * first and then expanded. Even at T = 512 the actual bucket computation runs only 1,023 times.
 */
export const buildRelPosTables = (
  length: number,
  bucketSize: number,
  maxPosition: number,
): RelPosTables => {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`Relative position table length ${length} is not an integer >= 1`);
  }
  if (!Number.isInteger(bucketSize) || bucketSize < 2) {
    throw new RangeError(`bucketSize ${bucketSize} is not an integer >= 2`);
  }
  if (!Number.isInteger(maxPosition) || maxPosition < 2) {
    throw new RangeError(`maxPosition ${maxPosition} is not an integer >= 2`);
  }
  assertLogBucketDomain(bucketSize, maxPosition, "relative position table");
  const span = bucketSize;
  const upper = 2 * span - 1;
  // Values on diagonal d = i − j ∈ [−(length−1), length−1] (the tables depend only on d).
  const diagonal = new Int32Array(2 * length - 1);
  for (let d = -(length - 1); d <= length - 1; d += 1) {
    const shifted = logBucketPosition(d, bucketSize, maxPosition) + span;
    // MUST: check finiteness before writing to i32 (NaN slips through the clamp below and turns
    // into 0). With the current formula assertLogBucketDomain above excludes it first, so this is
    // not reached — a defence for when the formula changes.
    if (!Number.isFinite(shifted)) {
      throw new Error(`The bucket of relative position ${d} is not finite (${shifted})`);
    }
    diagonal[d + length - 1] = shifted < 0 ? 0 : shifted > upper ? upper : shifted;
  }
  const c2p = new Int32Array(length * length);
  const p2c = new Int32Array(length * length);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const value = diagonal[i - j + length - 1];
      c2p[i * length + j] = value;
      p2c[j * length + i] = value;
    }
  }
  return {
    c2pPos: { dtype: "i32", shape: [length, length], data: c2p },
    p2cPos: { dtype: "i32", shape: [length, length], data: p2c },
  };
};
