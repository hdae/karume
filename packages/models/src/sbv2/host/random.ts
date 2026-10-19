/**
 * Deterministic standard normal random numbers (used for sdp's `z_noise` and the noise of z_p).
 *
 * Random numbers are not baked into the graph (ADR 0013 — run-time knobs and random numbers live on
 * the host). The pipeline holds a seeded generator so that **the same seed gives the same WAV**
 * can be the premise of reproducibility (with `Math.random` the comparison with the torch
 * reference would be "a different input every time", and feeding it the dumped random sequence
 * would mean little).
 *
 * The implementation is splitmix64 (state transition) + Box–Muller (normalisation). splitmix64 uses
 * 64-bit constants, so it runs on `BigInt` — the amount generated is about 2·P + C·T (tens of
 * thousands to hundreds of thousands of elements), negligible next to the rest of the host work.
 */

import { assertAcceptableSeed } from "../../request-gates.ts";

const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;
const MIX_1 = 0xbf58476d1ce4e5b9n;
const MIX_2 = 0x94d049bb133111ebn;
const MASK_64 = (1n << 64n) - 1n;
/** The denominator for making a uniform random number in [0,1) (53 mantissa bits). */
const DENOMINATOR = 1n << 53n;

/** One step of splitmix64. */
const nextUint64 = (state: bigint): { readonly state: bigint; readonly value: bigint } => {
  const next = (state + GOLDEN_GAMMA) & MASK_64;
  let z = next;
  z = ((z ^ (z >> 30n)) * MIX_1) & MASK_64;
  z = ((z ^ (z >> 27n)) * MIX_2) & MASK_64;
  return { state: next, value: (z ^ (z >> 31n)) & MASK_64 };
};

/**
 * Generator of a deterministic standard normal sequence (the same `seed` gives the same sequence).
 */
export class Randn {
  #state: bigint;

  constructor(seed: number) {
    // The owner of the acceptance set is the one family-wide `request-gates.ts` (the same range for
    // anima / sbv2 / irodori / wan / the generation sampler). Copying the condition here would one
    // day split by one side loosening.
    assertAcceptableSeed(seed);
    this.#state = BigInt(seed) & MASK_64;
  }

  /** A uniform random number in [0, 1). */
  #uniform(): number {
    const { state, value } = nextUint64(this.#state);
    this.#state = state;
    return Number(value >> 11n) / Number(DENOMINATOR);
  }

  /**
   * Draws `count` elements of a standard normal sequence, multiplies by `scale` and returns f32.
   *
   * MUST: even for an odd length, consume the Box–Muller pair whole (do not carry the leftover sin
   * side over to the next call). Carrying it over makes "the same seed gives a different sequence
   * depending on call order", and the correspondence with the dumped random sequence becomes
   * unreadable.
   */
  normals(count: number, scale = 1): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`Element count ${count} is not a non-negative integer`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 gives log(0) = -Inf. Move it to the midpoint 0.5/2^53 of the 53-bit lattice (radius
      // ≈8.65σ = on par with the lattice's natural maximum ≈8.57σ. The old Number.MIN_VALUE is the
      // smallest **subnormal** and produced an outlier of ≈38.6σ). It occurs with probability
      // exactly 2^-53 per draw, so this replacement does not in practice move the output bits of
      // existing seeds (the amounts generated for the frozen sha values never hit it).
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2) * scale;
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2) * scale;
    }
    return out;
  }
}
