/**
 * Deterministic [0,1) uniform random numbers (used for the sampling draw).
 *
 * Random numbers are not baked into the graph (ADR 0013 — runtime knobs and random numbers stay on
 * the host). The generator is seeded because reproducibility rests on **the same seed giving the
 * same token sequence**.
 *
 * The implementation is splitmix64 (the state transition). It is a **deliberate copy of the same
 * core** as `Randn` in `anima/random.ts` / `sbv2/host/random.ts` / `irodori/host/random.ts`, and
 * none of them imports another (the same copy already exists among those three, which is the
 * precedent) — a random stream is an input that decides its consumer's output, so sharing it would
 * let one side's needs (consumption order, amount drawn) silently move the other side's image /
 * waveform / token sequence. The only difference is the normalization step: this one skips
 * Box–Muller and emits the uniform value as is (a multinomial draw needs just one [0,1) value).
 *
 * MUST: not bit-identical to torch's sampling (ADR 0083 decision 7). Comparisons that need the same
 * token sequence as HF's `generate` are taken at **temperature 0 (= the greedy degenerate form)** —
 * parity is not attainable on a path where the draw runs.
 */

import { assertAcceptableSeed } from "../request-gates.ts";

const GOLDEN_GAMMA = 0x9e3779b97f4a7c15n;
const MIX_1 = 0xbf58476d1ce4e5b9n;
const MIX_2 = 0x94d049bb133111ebn;
const MASK_64 = (1n << 64n) - 1n;
/** Denominator for making a [0,1) uniform value (53 mantissa bits). */
const DENOMINATOR = 1n << 53n;

/** One splitmix64 step. */
const nextUint64 = (state: bigint): { readonly state: bigint; readonly value: bigint } => {
  const next = (state + GOLDEN_GAMMA) & MASK_64;
  let z = next;
  z = ((z ^ (z >> 30n)) * MIX_1) & MASK_64;
  z = ((z ^ (z >> 27n)) * MIX_2) & MASK_64;
  return { state: next, value: (z ^ (z >> 31n)) & MASK_64 };
};

/** Generator of a deterministic [0,1) uniform sequence (the same `seed` gives the same one). */
export class Randu {
  #state: bigint;

  constructor(seed: number) {
    // The acceptance set has one owner, `request-gates.ts` (holding a copy of the same range lets
    // one side loosen alone — ADR 0107 decision 6). This place holds no condition; it only calls
    // that check.
    assertAcceptableSeed(seed);
    this.#state = BigInt(seed) & MASK_64;
  }

  /** The next [0, 1) uniform value. */
  uniform(): number {
    const { state, value } = nextUint64(this.#state);
    this.#state = state;
    return Number(value >> 11n) / Number(DENOMINATOR);
  }
}
