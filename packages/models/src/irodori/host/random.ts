/**
 * Deterministic standard normal random numbers (used for the Euler initial noise `x_t`).
 *
 * Random numbers are not baked into the graph (ADR 0013 — runtime knobs and random numbers are on
 * the host side). The pipeline holds a seeded generator because reproducibility rests on **the same
 * seed giving the same latent**.
 *
 * The implementation is splitmix64 (state transition) + Box–Muller (to a normal).
 *
 * NOTE: a **deliberate duplicate** of `sbv2/host/random.ts`. Families do not import from each other
 * (the same duplicate exists between sbv2 and anima, the precedent) — a random sequence is an input
 * that decides that family's output, so sharing it would let one side's needs (consumption order,
 * amount generated, distribution) quietly move the other side's waveform / latent.
 *
 * MUST: it is not bit-identical to torch's `randn`. A comparison that needs the same sequence as
 * upstream (golden-like) is done by **injecting the noise from outside**
 * (`GenerateRequest.initialNoise`).
 */

import { assertAcceptableSeed } from "../../request-gates.ts";

const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;
const MIX_1 = 0xbf58476d1ce4e5b9n;
const MIX_2 = 0x94d049bb133111ebn;
const MASK_64 = (1n << 64n) - 1n;
/** Denominator for uniform random numbers in [0,1) (53 mantissa bits). */
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
 *
 * NOTE: the owner of the seed acceptance set (non-negative safe integers) is the single
 * {@link assertAcceptableSeed} in `request-gates.ts`, and this family keeps no copy (ADR 0107
 * decision 6). The generator is created after the DiT step has started, so the request check of
 * `pipeline.ts` calls the same function first, **at the entry of generation** — with the check only
 * here, it would fail after a GB-class load and running 5 graphs.
 */
export class Randn {
  #state: bigint;

  constructor(seed: number) {
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
   * Draws `count` elements of the standard normal sequence, multiplies by `scale` and returns f32.
   *
   * MUST: consume whole Box–Muller pairs even for odd lengths (do not carry the leftover sin side
   * into the next call). Carrying it would make "the same seed gives a different sequence depending
   * on the call order".
   */
  normals(count: number, scale = 1): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`Element count ${count} is not a non-negative integer`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 gives log(0) = -Inf. Move it to the midpoint 0.5/2^53 of the 53-bit lattice (radius
      // ≈8.65σ = about the lattice's natural maximum ≈8.57σ; the old Number.MIN_VALUE is the
      // smallest **subnormal** and produced an outlier of ≈38.6σ). It occurs with probability
      // exactly 2^-53 per draw, so this substitution practically moves no output bits of existing
      // seeds (the frozen sha generation volume never hits it).
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2) * scale;
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2) * scale;
    }
    return out;
  }
}
