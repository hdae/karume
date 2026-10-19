/**
 * A tool that lays out cleanup steps so that "the rest always runs even if one fails midway"
 * (**pipeline-independent shared processing** — used the same way by the `dispose()` of families
 * that hold resident Sessions).
 *
 * MUST: not exported from the barrel (for the same reason as the co-located `options.ts` /
 * `gpu-features.ts` — an internal mechanism, not a surface the user touches).
 *
 * ## Why sequential `await` does not work
 *
 * It is tempting to write `dispose()` as `await session.dispose(); if (ownsGpu) gpu.destroy();`, but
 * `Session.dispose()` can reject (`RunArena` propagates a flush failure instead of swallowing it).
 * Written sequentially, a failure in the earlier step means the later one is never reached, and the invariant
 * that each family's `openXState` upholds in its catch — "**a GPU acquired internally is always
 * returned**" — breaks on the dispose path only. On top of that, a rejected promise stays in
 * `#disposal`, so a second `dispose()` is not a retry but just returns the same failure (`#state` is
 * private, so the caller has no other handle).
 *
 * `try { … } finally { gpu.destroy(); }` is worse too — an exception inside `finally` **overwrites
 * the original failure**, and what broke first is lost.
 */

/**
 * Runs the steps in declaration order, collects the failures, then rethrows (the caller decides the
 * disposal order).
 *
 * Each step catches on its own, so even if one throws midway the remaining steps always run. With
 * one failure it throws **that exception as is** (so the caller's
 * `assertRejects(…, Error, "device lost")` stays straightforward); with two or more it carries them
 * in an `AggregateError` (without losing which step failed).
 */
export const disposeSteps = async (
  steps: readonly (() => Promise<void> | void)[],
): Promise<void> => {
  const failures: unknown[] = [];
  for (const step of steps) {
    try {
      await step();
    } catch (error) {
      failures.push(error);
    }
  }
  if (failures.length === 1) throw failures[0];
  if (failures.length > 1) {
    throw new AggregateError(failures, `dispose: ${failures.length} clean-up steps failed`);
  }
};
