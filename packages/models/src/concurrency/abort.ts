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
