/**
 * Host implementation that generates the `(T, T)` tables of windowed relative-position attention at
 * the actual length (for SBV2's flow / voice).
 *
 * ## Why this lives here and not in runtime
 *
 * These tables are **SBV2-specific** (or rather specific to "models whose windowed
 * relative-position attention was turned into a gather") and are not part of the `@karume/runtime`
 * vocabulary. runtime is "a general runtime that executes IR" and must not know how a particular
 * model's inputs are built (ADR 0008's thin public surface). **There are two users: the pipeline
 * (`pipeline.ts` calls it for each T) and verification (`packages/models/tests/`)**, and the
 * canonical copy sits in the models package, which holds the model-side knowledge (both import it
 * directly — no thin surface that copies the implementation is made).
 *
 * ## Why the host builds the tables
 *
 * front (P ≤ 512) has the same formula inside the graph, and the exporter's constant folding bakes
 * it at Pmax. For flow / voice, T ≤ 4096 makes the tables O(T²), and baking them would take 134MB
 * for the constants alone, so they were **promoted to graph inputs** (ADR 0013). As a result,
 * producing the table values became the host's responsibility.
 *
 * ## MUST: byte-identical to the Python side
 *
 * The canonical formula is `build_relattn_tables` in `tools/export-recipes/sbv2/patch.py`. Even if
 * the formulas diverge, **the shapes still match**, so the model does not fail and silently reads
 * different embeddings (a silent wrong value). `packages/models/tests/sbv2_relattn_parity_test.ts`
 * pins byte equality with the Python-generated tables stored in the golden io, on real data.
 */

import type { Tensor } from "@karume/runtime";

/**
 * Window width of relative-position attention. SBV2's enc_p / flow use 4 (the embedding has
 * `2w+1 = 9` rows).
 *
 * MUST (silent-wrong-value class): when this value differs from the model's `window_size`, `idx_k`
 * stays `clamp(rel+4, 0, 8)` and **gathers an embedding of a different width**. The element counts
 * add up, so there is no shape error and only the output is silently wrong. On the Python side
 * `sbv2.export._assert_window_size` (`tools/export-recipes/sbv2/export.py`) fails at ckpt load,
 * and on the host side the parity test fails by comparing it with **the width of the value-side
 * table baked into the model container** (`2w+1`) — the check sits on both sides because with only
 * one, a path remains where "the host and the golden share the same error and slip through
 * verification".
 */
export const RELATTN_WINDOW_SIZE = 4;

/**
 * Return value of `buildRelattnTables` (a form that can be passed as is to the graph inputs
 * `idx_k` / `valid`).
 */
type RelattnTables = {
  /** `idx_k[i][j] = clamp(w + j − i, 0, 2w)` — key-side embedding index. */
  readonly idxK: Tensor;
  /** `valid[i][j] = |j − i| ≤ w ? 1 : 0` — 0/1 mask that drops what is outside the window. */
  readonly valid: Tensor;
};

/**
 * Builds the relative-position tables for actual length `length`.
 *
 * `idx_k` is clamped, so even outside the window it points at **a different, in-range embedding**
 * (a form that does not depend on the out-of-range convention of gather — ADR 0013). What removes
 * the contribution outside the window is the multiplication by `valid`, so the two are always used
 * as a pair.
 */
export const buildRelattnTables = (
  length: number,
  windowSize: number = RELATTN_WINDOW_SIZE,
): RelattnTables => {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`相対位置表の長さ ${length} が 1 以上の整数でない`);
  }
  if (!Number.isInteger(windowSize) || windowSize < 0) {
    throw new RangeError(`窓幅 ${windowSize} が 0 以上の整数でない`);
  }
  const idxK = new Int32Array(length * length);
  const valid = new Float32Array(length * length);
  const upper = 2 * windowSize;
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      const rel = j - i;
      const shifted = rel + windowSize;
      idxK[i * length + j] = shifted < 0 ? 0 : shifted > upper ? upper : shifted;
      valid[i * length + j] = rel >= -windowSize && rel <= windowSize ? 1 : 0;
    }
  }
  const shape = [length, length] as const;
  return {
    idxK: { dtype: "i32", shape: [...shape], data: idxK },
    valid: { dtype: "f32", shape: [...shape], data: valid },
  };
};
