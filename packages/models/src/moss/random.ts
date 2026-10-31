/**
 * Deterministic standard normal random numbers — MOSS-Image's initial noise from a seed (ADR 0138
 * decision 9, stage 4).
 *
 * Random numbers are not baked into the graph (ADR 0013 — runtime knobs and random numbers live on
 * the host). The generator is seeded so that reproducibility rests on **the same seed giving the
 * same image**.
 *
 * The implementation is splitmix64 + Box–Muller. It is a **deliberate duplicate of the same core**
 * as `WanRandn` (`../wan/random.ts`) and the `Randn` of anima / sbv2 / irodori, and none of them
 * import each other (the doc of `../generation/random.ts` — a random sequence is an input that
 * decides its consumer's output, so sharing it would let one consumer's needs — consumption order,
 * amount drawn — silently move the other's output).
 *
 * ## MUST: this is not torch's `randn` (a deliberate departure from upstream)
 *
 * Upstream draws its initial noise with `randn_tensor(shape, generator=…, dtype=float32)` (torch's
 * Philox / mt19937 streams — `MOSSImagePipeline.__call__`). The same seed does not give torch's
 * sequence here: the port takes the seeded noise of its other diffusion families — splitmix64 +
 * Box–Muller as in Wan, deliberately not torch's `randn` (ADR 0138 decision 9; the Wan precedent
 * ADR 0118 decision 5). Comparisons with upstream never use this generator: they inject the
 * reference's own initial latents through the request's `latents` (`moss.reference` records
 * them as `latents.init` — `tools/export-recipes/moss/pipeline_ref.py`).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
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
export class MossRandn {
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
      // handling as anima and Wan).
      const u1 = this.#uniform() || 0.5 / 2 ** 53;
      const u2 = this.#uniform();
      const radius = Math.sqrt(-2 * Math.log(u1));
      out[i] = radius * Math.cos(2 * Math.PI * u2);
      if (i + 1 < count) out[i + 1] = radius * Math.sin(2 * Math.PI * u2);
    }
    return out;
  }
}
