/**
 * A tool that observes an abort of construction at stage boundaries (**pipeline-independent shared
 * processing**).
 *
 * MUST: not exported from the barrel. The abort knob is each pipeline's `options.signal`; this is an
 * internal mechanism that keeps the placement of that check uniform.
 */

/**
 * Checks for abort at a stage boundary **in a form that can observe it** (yields to the event loop
 * once, then `throwIfAborted`).
 *
 * MUST: the yield goes to a **macrotask**. `abort()` arrives through **task** delivery — clicks,
 * timers, worker messages — so `await Promise.resolve()` (a microtask) stays inside the current task
 * and cannot observe an abort task that has not run yet. A check without the yield is dead text that
 * only catches "already aborted at the time of the call".
 *
 * NOTE: the yield is fixed to **`setTimeout(0)`** (not `MessageChannel`). In a hidden browser tab,
 * timer throttling becomes the generation speed as is (one timer per token), but switching to
 * `MessageChannel` **loses the ordering with aborts delivered by timers
 * (`setTimeout(() => abort())` / `AbortSignal.timeout`)** — a message task can be delivered before a
 * timer armed earlier, so "an abort that arrives after execution starts takes effect at the first
 * stage boundary" (the check in `anima_pipeline_test` / `irodori_pipeline_test`) breaks depending on
 * load (measured in the full verify of 2026-09-05). Reliable observation of aborts takes priority
 * over speed. The hidden-tab throttling is recorded in the limitations section "Browser: in a hidden
 * tab, generation is pinned to the timer-throttling tick".
 *
 * Without a `signal` it does not yield (calls that do not subscribe do not pay a task round trip).
 */
export const settleAbort = async (signal: AbortSignal | undefined): Promise<void> => {
  if (signal === undefined) return;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  signal.throwIfAborted();
};

/**
 * The abort checks of **a loop of runs** ({@link createAbortLap}): one `settle()` per stage boundary,
 * `clear()` when the loop is left.
 */
export type AbortLap = {
  /**
   * The stage-boundary check: resolves once the event loop has delivered a timer since the previous
   * `settle()`, then throws `signal.reason` if the signal is aborted.
   */
  settle(): Promise<void>;
  /** Drops the timer the last `settle()` armed (a loop that ends has no next boundary). */
  clear(): void;
};

/**
 * {@link settleAbort} for a loop whose every lap awaits a run: the yield to the event loop is taken
 * **while the run is awaited** instead of after it.
 *
 * A lap of such a loop — a decode token — already hands the thread back to the event loop for as
 * long as the GPU works on the run, so a second, serial yield per lap only adds idle time: under
 * Node a `setTimeout(0)` is a 1 ms timer, and a decode token of a few milliseconds pays it in full
 * (measured: perf-ledger H-31). Each `settle()` therefore arms a `setTimeout(0)` **for the next
 * boundary** and does not wait for it. The next `settle()`:
 *
 * - finds that timer **delivered**: the event loop has run its timers since the previous boundary —
 *   what the yield is for — so it checks the signal at once;
 * - finds it **not delivered** (the lap awaited no task, or returned inside the timer's delay): it
 *   drops it and yields exactly as {@link settleAbort} does, with a timer armed now.
 *
 * MUST: the proof that the event loop turned is a **delivered timer**, never the fact that a run
 * was awaited. A run that settles through microtasks alone (a fake Session, a cached result) crosses
 * no task boundary, and a loop that took its word for it would never let an abort task run. The
 * timer is a `setTimeout(0)` for the reason {@link settleAbort} gives: it is ordered after every
 * timer armed before it, so an abort a timer delivers is not overtaken.
 *
 * What changes against a serial yield per lap: an abort whose task is still **queued** when a run
 * returns — it became ready after the event loop's last turn inside that run — is seen one lap
 * later, after the next run, instead of before it. An abort whose task **ran** at any point before
 * the boundary, a synchronous `abort()` from the consumer included, is seen at that boundary as
 * before; so is every abort of a loop whose laps await no task.
 *
 * The first `settle()` has no earlier boundary and yields as {@link settleAbort} does. Without a
 * `signal` nothing is armed and nothing yields.
 *
 * SPEC: docs/spec/generation.md#abort-checks
 */
export const createAbortLap = (signal: AbortSignal | undefined): AbortLap => {
  /** The timer armed at the last boundary: whether it was delivered, and its handle. */
  let armed: { delivered: boolean; readonly handle: ReturnType<typeof setTimeout> } | undefined;
  const arm = (): void => {
    const next = {
      delivered: false,
      handle: setTimeout(() => {
        next.delivered = true;
      }, 0),
    };
    armed = next;
  };
  const clear = (): void => {
    if (armed === undefined) return;
    clearTimeout(armed.handle);
    armed = undefined;
  };
  return {
    settle: async () => {
      if (signal === undefined) return;
      const delivered = armed?.delivered === true;
      clear();
      if (!delivered) await new Promise<void>((resolve) => setTimeout(resolve, 0));
      signal.throwIfAborted();
      arm();
    },
    clear,
  };
};
