/**
 * Deterministic standard normal random numbers (Wan's initial noise — ADR 0118 decision 5).
 *
 * Random numbers are not baked into the graph (ADR 0013 — runtime knobs and random numbers live on
 * the host). There is a seeded generator so that reproducibility can rest on **the same seed giving
 * the same video**.
 *
 * The implementation is splitmix64 + Box–Muller. It is a **deliberate duplicate of the same core** as
 * the `Randn` of `anima/random.ts` / `sbv2/host/random.ts` / `irodori/host/random.ts`, and none of
 * them import each other (the precedent is the doc of `generation/random.ts` — a random sequence is
 * an input that determines its consumer's output, so sharing it would let one consumer's needs
 * (consumption order, amount drawn) silently move the other's output).
 *
 * ## MUST: this is not torch's `randn`
 *
 * The same seed does not give torch's sequence. Comparison with the reference does not use latents
 * made by this generator; it injects the fixture's `latents_init` through
 * `GenerateRequest.latents`.
 */

// The owner of the accepted seed set (non-negative safe integers) is the one cross-family
// `request-gates.ts`.
import { assertAcceptableSeed } from "../request-gates.ts";

const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;
const MIX_1 = 0xbf58476d1ce4e5b9n;
const MIX_2 = 0x94d049bb133111ebn;
const MASK_64 = (1n << 64n) - 1n;
/** The denominator for drawing a uniform number in [0,1) (a 53-bit mantissa). */
const DENOMINATOR = 1n << 53n;

/** One step of splitmix64. */
const nextUint64 = (state: bigint): { readonly state: bigint; readonly value: bigint } => {
  const next = (state + GOLDEN_GAMMA) & MASK_64;
  let z = next;
  z = ((z ^ (z >> 30n)) * MIX_1) & MASK_64;
  z = ((z ^ (z >> 27n)) * MIX_2) & MASK_64;
  return { state: next, value: (z ^ (z >> 31n)) & MASK_64 };
};

/** Generator of a deterministic standard normal sequence (the same `seed` gives the same sequence). */
export class WanRandn {
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
   * Draws `count` elements of the standard normal sequence and returns them as f32.
   *
   * MUST: consume whole Box–Muller pairs even for an odd length (do not carry the leftover sin side
   * over to the next call).
   */
  normals(count: number): Float32Array<ArrayBuffer> {
    if (!Number.isInteger(count) || count < 0) {
      throw new RangeError(`Element count ${count} is not a non-negative integer`);
    }
    const out = new Float32Array(count);
    for (let i = 0; i < count; i += 2) {
      // u1 = 0 yields log(0) = -Inf. Move it to the midpoint 0.5/2^53 of the 53-bit grid (the same
      // handling as anima).
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2);
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2);
    }
    return out;
  }
}
