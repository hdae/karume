/**
 * A small concurrency primitive that paces the dispatch of the whole-file surface
 * {@link ../src/fetch.ts fetchAssets}.
 *
 * It has a single caller, but it carries an invariant (no lost wakeup) whose violation shows only
 * as "it hangs now and then". It is split out into a form that can be observed on its own so that
 * tests freeze the behaviour itself.
 *
 * MUST: zero module side effects — state always stays closed inside each create* call (top-level
 * mutable state would make two unrelated downloads share the same budget).
 */

/** Lending and returning of slots, as returned by {@link createByteAdmission}. */
export type ByteAdmission = {
  /** Waits until `bytes` worth of slots are free and resolves once they are acquired. */
  readonly admit: (bytes: number) => Promise<void>;
  /** Returns the slots when a download settles (success or failure alike). */
  readonly release: (bytes: number) => void;
};

/**
 * Admission control that caps the total in-flight bytes. Requests pass **in arrival order** and are
 * never overtaken (if a later small file could jump past a huge one, the dispatch order would
 * depend on the run-time size distribution and reproducibility would be lost).
 *
 * MUST: **only one dispatch loop at a time** calls `admit`. There is a single waiter slot, so
 * waiting from several callers would lose a wakeup from `release` (this constraint is also what
 * makes head-of-line blocking hold trivially).
 *
 * When nothing is in flight, a single request larger than the budget always passes — waiting while
 * nobody runs would hang forever, with no one left to return the slots.
 *
 * Why there is no lost wakeup: (1) there is no await between the check and the waiter registration,
 * so a `release` cannot slip into that gap. (2) If another `release` arrives before the woken
 * waiter re-checks, then either in-flight is 0 at that point, the condition is false and the waiter
 * proceeds, or it is not 0, in which case an unsettled download remains, so a future `release` will
 * come.
 */
export const createByteAdmission = (maxBytes: number): ByteAdmission => {
  let inFlight = 0;
  let waiting: (() => void) | undefined;
  return {
    admit: async (bytes: number): Promise<void> => {
      while (inFlight > 0 && inFlight + bytes > maxBytes) {
        await new Promise<void>((resume) => {
          waiting = resume;
        });
      }
      inFlight += bytes;
    },
    release: (bytes: number): void => {
      inFlight -= bytes;
      const resume = waiting;
      waiting = undefined;
      resume?.();
    },
  };
};
