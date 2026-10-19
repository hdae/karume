/**
 * Host generation of `t_embed` (a copy of upstream's `get_timestep_embedding`).
 *
 * It cannot go into the graph because `cos` is not in the IR's op vocabulary (the layer-1 decision
 * that added only `sin` — ADR 0043 / 0047 decision 4).
 *
 * MUST: f32 rounding is applied one operation at a time with `Math.fround`. JS numbers are f64, so
 * computing in bulk and rounding afterwards changes the last digits relative to torch's sequential
 * f32 computation (the same section of `anima/sampler.ts`).
 *
 * ## NOTE: it is **not** bit-identical to the reference (do not tighten expecting a match)
 *
 * torch CPU's f32 `exp` / `sin` / `cos` are SLEEF's 1.0-ULP implementations, and their last bits
 * split from JS's `Math.*` (computed in f64 and rounded to f32). The comparison is bounded by an
 * atol derived from measurements against the golden table (the docstring of the constant on the
 * test side).
 */

const f32 = Math.fround;

/** Upstream's `max_period` (the 10000 in `1000 · exp(−ln(10000)·k/half)`). */
const MAX_PERIOD = 10000;
/** Overall scale of the frequencies (upstream's `1000.0`). */
const FREQ_SCALE = 1000;

/**
 * Frequency table `[dim/2]`. It does not depend on t, so it is built once and reused (avoids
 * recomputing it for 40 steps).
 *
 * The order of the formula is also copied from upstream: round `−log(10000)` to f32, multiply by
 * `k`, divide by `half`, take `exp`, then multiply by `1000`. Computing it all in f64 changes the
 * last bits.
 */
export const timestepFrequencies = (dim: number): Float32Array<ArrayBuffer> => {
  if (!Number.isInteger(dim) || dim <= 0 || dim % 2 !== 0) {
    throw new RangeError(`t_embed width ${dim} is not a positive even number`);
  }
  const half = dim / 2;
  const logMaxPeriod = f32(-Math.log(MAX_PERIOD));
  const freqs = new Float32Array(half);
  for (let index = 0; index < half; index += 1) {
    const exponent = f32(f32(logMaxPeriod * index) / half);
    freqs[index] = f32(FREQ_SCALE * f32(Math.exp(exponent)));
  }
  return freqs;
};

/**
 * `t_embed` `[2·freqs.length]` for one step. **First half cos, second half sin** (upstream's
 * `torch.cat`).
 *
 * MUST: do not swap the halves. Both have the range `[-1,1]`, so a mix-up changes neither the
 * shape nor the statistics; only the output becomes something else.
 */
export const timestepEmbedding = (
  t: number,
  freqs: Float32Array<ArrayBuffer>,
): Float32Array<ArrayBuffer> => {
  const half = freqs.length;
  const out = new Float32Array(half * 2);
  for (let index = 0; index < half; index += 1) {
    const angle = f32(t * freqs[index]);
    out[index] = f32(Math.cos(angle));
    out[half + index] = f32(Math.sin(angle));
  }
  return out;
};
