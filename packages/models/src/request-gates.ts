/**
 * The home of the generation-request checks whose **accepted set is the same across families**.
 *
 * Per-family checks (resolution, style name, chunk length) live in the family — only checks that can
 * be stated as "exactly the same values pass in anima, sbv2 and irodori alike" come here. When
 * several families keep copies of an accepted set, they always diverge (one loosens, one switches to
 * a falsy test), so the job of this module is to **give it a single owner**.
 *
 * MUST: zero module side effects (no import-time execution, no global mutable state — CLAUDE.md).
 */

import { ModelInputError } from "./errors.ts";

/**
 * The check of the accepted set of `seed` (**a non-negative safe integer**). Shared by anima / sbv2 /
 * irodori / wan and gemma's generation sampler (`generation/random.ts`).
 *
 * All five turn the received `seed` into a `BigInt` as splitmix64 state, so the reasons for the
 * conditions split three ways:
 *
 * - **Non-negative** is the API's value-range policy. A negative value also converts to `BigInt`
 *   (`BigInt(-1) === -1n`) and becomes a valid state once folded to 64 bits, but it is not accepted
 *   as a `seed` request.
 * - **The upper bound `Number.MAX_SAFE_INTEGER`** is there because adjacent integers cannot be told
 *   apart from 2^53 on (2^53 itself is exact, but 2^53 + 1 collapses to 2^53). Opening it would make
 *   "the value passed" and "the value used" disagree silently.
 * - **Fractions** make the `BigInt` conversion itself throw a `RangeError` (`BigInt(1.5)`), so
 *   without rejecting them at the entry they surface inside the random generator as an exception
 *   with a different face.
 *
 * NOTE: it is called at **the generation entry** (the start of each `generate`), not only in the
 * random generator's constructor. The generator is created after the weights are on the GPU, so if
 * the check lived only there, a typo would fail with "seed is not an integer" **after waiting for a
 * GB-scale load**. Rejecting at the entry returns a typo in a millisecond.
 */
export const assertAcceptableSeed = (seed: number): void => {
  if (!Number.isInteger(seed) || seed < 0 || seed > Number.MAX_SAFE_INTEGER) {
    throw new ModelInputError(`seed ${seed} が非負の安全整数でない`);
  }
};
