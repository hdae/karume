/**
 * The host-built position inputs of the Qwen3.5 text tower (ADR 0137 decision 2): the positions
 * `[T]` (0 … T−1) the graph builds its causal mask from, and the RoPE rows
 * `rope_cos` / `rope_sin` `[1, T, rotaryDim]`.
 *
 * ## Why the host builds them
 *
 * The exported graph does not trace upstream's rotary module (it writes `freqs[..., idx] = …` in
 * place and calls `cos`, neither of which is IR vocabulary), and folding the rows into a constant
 * would need a `[Tmax, rotaryDim]` table sliced on the symbolic axis. The recipe therefore takes the
 * rows as graph inputs (`tools/export-recipes/qwen3_5/patch.py`, module doc "RoPE: host-supplied
 * rows" — the gemma4 precedent, ADR 0091), and the causal mask is a run-time `ge` over `positions`
 * (no T² constant).
 *
 * ## MUST: the same formula as the recipe's `rope_rows`
 *
 * `invFreq[i] = theta^(−2i/rotaryDim)` (i < rotaryDim/2), `angle = position · invFreq[i]`, row =
 * `cat(angle, angle)`, every step in f64 and the cosine / sine rounded to f32 once. For text-only
 * input Qwen3.5's interleaved mRoPE has three equal position rows, so this 1-D form is upstream's
 * rotation exactly (the recipe's module doc). The container's CPU reference was run on the recipe's
 * rows, so a different formula here would not fail any shape check — it would rotate every
 * attention layer's q / k by another angle (a silent wrong value). `qwen3_5_session_io_test.ts`
 * pins every row of positions 0 … 4095 against the recipe's output (fixture
 * `fixtures/qwen3_5-encoder/rope-rows.json`).
 *
 * Upstream's own rotary module computes in f32, so neither side is bit-identical to it; the bound
 * on that difference is ADR 0091 decision 2's position-proportional one (`rope_bound` in the
 * recipe).
 *
 * ## The constants are pinned, not read from the container
 *
 * The container records neither theta nor the rotary width (it has no assets and its graph holds
 * no RoPE constant), so both are pinned here from the checkpoint the series is exported from
 * ({@link QWEN3_5_ROPE_THETA}). The rotary width is checked against the container (the last axis
 * of `rope_cos` / `rope_sin` — `./graph-contract.ts`); theta cannot be, and a checkpoint with
 * another theta needs this module changed and re-measured (ADR 0137, open point 1).
 *
 * MUST: modules are side-effect free (CLAUDE.md).
 */

import type { Tensor } from "@karume/runtime";

/** An f32 tensor (the type of the host-built inputs — passed to a Session input as is). */
export type F32Tensor = Extract<Tensor, { readonly dtype: "f32" }>;

/**
 * The RoPE base of Qwen3.5-9B's text tower (`text_config.rope_parameters.rope_theta` of
 * `OpenMOSS-Team/MOSS-Image-SFT@f972bd4b…/text_encoder/config.json`, the checkpoint the
 * `qwen3.5-9b-text-moss-image-sft-*` series are written from — `qwen3_5.checkpoint.SOURCES` in the
 * recipe; read 2026-10-31).
 */
export const QWEN3_5_ROPE_THETA = 10_000_000;

/**
 * The rotated width of one head: `head_dim · partial_rotary_factor` = 256 · 0.25 (the same
 * checkpoint; the recipe's `rotary_dim`). The remaining 192 dimensions of each head pass through.
 */
export const QWEN3_5_ROTARY_DIM = 64;

/** The positions `0 … length−1` as f32 `[length]` (the graph's `positions` input). */
export const buildQwen3_5Positions = (length: number): F32Tensor => ({
  dtype: "f32",
  shape: [length],
  data: Float32Array.from({ length }, (_, position) => position),
});

/**
 * The RoPE rows for positions `0 … length−1`: `rope_cos` / `rope_sin` as f32
 * `[1, length, rotaryDim]` (the formula is the module doc's MUST).
 */
export const buildQwen3_5RopeRows = (
  length: number,
): { readonly cos: F32Tensor; readonly sin: F32Tensor } => {
  const width = QWEN3_5_ROTARY_DIM;
  const half = width / 2;
  const inverse = Float64Array.from(
    { length: half },
    (_, index) => 1 / QWEN3_5_ROPE_THETA ** ((2 * index) / width),
  );
  const cos = new Float32Array(length * width);
  const sin = new Float32Array(length * width);
  for (let position = 0; position < length; position += 1) {
    const row = position * width;
    for (let index = 0; index < half; index += 1) {
      const angle = position * inverse[index];
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      cos[row + index] = c;
      cos[row + half + index] = c;
      sin[row + index] = s;
      sin[row + half + index] = s;
    }
  }
  const shape = [1, length, width];
  return { cos: { dtype: "f32", shape, data: cos }, sin: { dtype: "f32", shape, data: sin } };
};
