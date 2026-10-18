/**
 * The host side of the sinusoidal stage of Wan2.1's DiT timestep embedding (graph input
 * `timesteps_proj [1,freq_dim]` — ADR 0118 decision 3). The MLP (`freq_dim → dim → dim → 6·dim`)
 * stays inside the graph.
 *
 * The source of truth is upstream `WanTimeTextImageEmbedding.timesteps_proj` =
 * `Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)` (diffusers'
 * `get_timestep_embedding`). **cos in the first half, sin in the second half**. The input is the
 * integer timestep the upstream pipeline passes (flow matching UniPC's `σ·1000` truncated to int64
 * — decision 5).
 */

const f32 = Math.fround;

/** The default `max_period` of upstream `get_timestep_embedding`. */
const MAX_PERIOD = 10000;

// Third-party code notice. `timestepsProj` below is adapted from `get_timestep_embedding` in
// huggingface/diffusers (`src/diffusers/models/embeddings.py`, `diffusers==0.39.0`), specialized to
// `flip_sin_to_cos=True`, `downscale_freq_shift=0`, `scale=1` and an even width, with each f32
// operation written out in the upstream order. License: Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0). Upstream copyright notice, copied verbatim from the
// header of that file:
//
//   Copyright 2026 The HuggingFace Team. All rights reserved.

/**
 * `timesteps_proj` `[width]` for one step (upstream's f32 operation order copied one operation at a
 * time).
 *
 * Operation order: `exponent_i = f32(f32(−ln(max_period)) · i) / half` (f32 division) →
 * `f32(exp(exponent_i))` → `angle_i = f32(f32(timestep) · freq_i)` → first half `cos(angle_i)`,
 * second half `sin(angle_i)`. `-math.log(max_period)` drops to f32 in the product with an f32 tensor
 * (torch's scalar promotion — the same handling as Anima's `timestepsProj`).
 *
 * ## MUST: this function is **not** bit-identical to the reference (do not tighten to a match)
 *
 * torch CPU's f32 `exp` / `sin` / `cos` are SLEEF's 1.0-ULP implementations, which can split in the
 * last bit from JS's `Math.*` (computed in f64 and rounded to f32). The comparison uses an atol
 * derived from measurements (`packages/models/tests/wan_dit_host_test.ts` — the measured values are
 * in its doc).
 *
 * `width` comes from the static dimension of the graph input `timesteps_proj` (the caller does not
 * write 256).
 */
export const timestepsProj = (timestep: number, width: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(width) || width <= 0 || width % 2 !== 0) {
    throw new RangeError(`timesteps_proj の幅 ${width} が正の偶数でない`);
  }
  if (!Number.isInteger(timestep) || timestep < 0) {
    throw new RangeError(`timestep ${timestep} が 0 以上の整数でない（上流は int64 の timestep）`);
  }
  const half = width / 2;
  const time = f32(timestep);
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const out = new Float32Array(width);
  for (let index = 0; index < half; index += 1) {
    // downscale_freq_shift = 0, so the denominator is half itself.
    const exponent = f32(f32(logMaxPeriod * index) / half);
    const angle = f32(time * f32(Math.exp(exponent)));
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};
