/**
 * Builds umT5-XXL's relative-position bucket table on the host (ADR 0119 decision 3).
 *
 * T5-style attention maps the position difference (key position − query position) to 32 buckets
 * and adds the value looked up from a per-layer table `[32, heads]` to the scores. The table
 * `[L, L]` depends only on the valid length L, but L is one symbol of the graph's dynamic shape
 * (decision 4), so baking it as a constant would need `[:L, :L]` sliced from the max-length table.
 * Following the promotion to inputs of DeBERTa (ADR 0045) and Anima's RoPE (ADR 0034), the host
 * builds it and passes it as a graph input.
 *
 * ## MUST: byte-identical to the Python side
 *
 * The source of truth is `relative_position_buckets` in the recipe
 * `tools/export-recipes/wan/umt5_patch.py` (which calls upstream transformers 5.14.1's
 * `UMT5Attention._relative_position_bucket` as is). If the formula splits, the shape still matches,
 * so the model does not fail but silently adds the bias of another distance (a silent wrong value).
 * The golden tables are also made by Python, so if host and golden share the same error the E2E
 * slips through as well. `packages/models/tests/umt5_relative_position_test.ts` pins the fixture
 * (`fixtures/wan-umt5/`) and every table for valid lengths 2 to 512 byte for byte (the same duty as
 * sbv2's `sbv2_rel_pos_parity_test.ts`).
 *
 * Upstream uses torch's f32 `log`; this uses f64 `Math.log`. That truncation does not split at the
 * boundaries (distances 8, 12, 16, 23, 32, 46, 64, 91) is a measured claim (research
 * 2026-10-03-umt5-encoder-recon §4.3 — all distances ±511 match), so measure again when changing the
 * configuration ({@link UMT5_RELATIVE_POSITION}) or the upper limit.
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { Tensor } from "@karume/runtime";

/** An i32 tensor (the type of the bucket table — it can be passed to a Session input as is). */
export type I32Tensor = Extract<Tensor, { readonly dtype: "i32" }>;

/**
 * The bucket configuration (umT5-XXL's `text_encoder/config.json` — only the bidirectional
 * (encoder) branch).
 *
 * There is no field to declare the configuration: only this one pair was compared with Python, and
 * making it declarable would let a distribution pick an unverified combination (the same judgement
 * as `WAN_UNIPC_CONFIG` in `src/wan/scheduler.ts`). The recipe's pytest
 * (`test_umt5_host_fixture.py`) and the fixture test pin the values to the pinned config.
 */
export const UMT5_RELATIVE_POSITION = { numBuckets: 32, maxDistance: 128 } as const;

/**
 * The lower limit of the valid length (the lower limit of the graph's symbolic dimension — 2, which
 * avoids `torch.export`'s 0 / 1 specialization; ADR 0119 decision 4; the same reason as anima's
 * `PROMPT_MIN_TOKENS`). The lower limit of the tokenizer's accepted set is this value too (rejects
 * length 1, only `</s>` — `./tokenizer.ts`).
 */
export const UMT5_MIN_TOKENS = 2;

/**
 * The upper limit of the valid length (the upper limit of the graph's symbolic dimension, upstream's
 * `max_sequence_length` — decision 4). The match with Python's tables was also verified only up to
 * here (distance ±511).
 */
export const UMT5_MAX_LENGTH = 512;

/** The bucket of `distance` (key position − query position) (the bidirectional branch of upstream's formula). */
const bucketOf = (distance: number): number => {
  const { numBuckets, maxDistance } = UMT5_RELATIVE_POSITION;
  // Bidirectional: each direction gets half (positive distance = key after the query — the upper
  // half, 16 to 31).
  const half = numBuckets / 2;
  const base = distance > 0 ? half : 0;
  const magnitude = Math.abs(distance);
  // Half of the half (0 to 7) is the distance itself; the rest (8 to 15) is a logarithmic scale up to
  // max_distance.
  const maxExact = half / 2;
  if (magnitude < maxExact) return base + magnitude;
  // Upstream's `log_ratio.to(torch.long)` (truncation toward 0 — the log ratio is non-negative, so
  // the same as floor).
  const large =
    maxExact +
    Math.trunc(
      (Math.log(magnitude / maxExact) / Math.log(maxDistance / maxExact)) * (half - maxExact),
    );
  return base + Math.min(large, half - 1);
};

/**
 * The bucket table `[L, L]` for valid length `length` (i32, row-major).
 * `table[i][j] = bucket(j − i)` (the same orientation as upstream's `compute_bias` — rows are
 * queries, columns keys).
 *
 * The table depends only on `j − i` (Toeplitz), so the `2L − 1` per-diagonal values are built first
 * and then expanded (even at L = 512 the bucket is computed 1,023 times).
 */
export const buildUmt5RelativePositionBuckets = (length: number): I32Tensor => {
  if (!Number.isInteger(length) || length < UMT5_MIN_TOKENS || length > UMT5_MAX_LENGTH) {
    throw new RangeError(
      `umT5 valid length ${length} is not an integer in ${UMT5_MIN_TOKENS}..${UMT5_MAX_LENGTH}`,
    );
  }
  // The values for diagonals d = j − i ∈ [−(L−1), L−1] (index d + L − 1).
  const diagonal = new Int32Array(2 * length - 1);
  for (let distance = 1 - length; distance < length; distance += 1) {
    diagonal[distance + length - 1] = bucketOf(distance);
  }
  const data = new Int32Array(length * length);
  for (let query = 0; query < length; query += 1) {
    // Row i is a straight copy of a contiguous range of the diagonal values (d = −i to L−1−i).
    const start = length - 1 - query;
    data.set(diagonal.subarray(start, start + length), query * length);
  }
  return { dtype: "i32", shape: [length, length], data };
};
