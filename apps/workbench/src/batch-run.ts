/**
 * A batch run, as the browser shell and the Node command both do it: the selected steps in file
 * order, each one ordinary run through {@link runEntry} on the host (the environment and the
 * checkout read when the step starts) whose record carries the `batch` field. The stop and
 * continue rules live here:
 *
 * - a step that ends as `error` or with verdict `fail` does not stop the batch;
 * - an aborted signal (Cancel, SIGINT) cancels the running step — its record is `cancelled` and
 *   saved as any other — and the remaining steps do not run;
 * - a step whose run rejects (the environment or the checkout cannot be read, or the host cannot
 *   save) stops the batch: that failure is the host's, not the step's.
 *
 * SPEC: docs/spec/workbench.md#batches
 */

import type { ResolvedBatch, ResolvedStep } from "./batch.ts";
import { type RunHost, runEntry, type SingleRun } from "./execute.ts";
import type { Outcome } from "./record.ts";

/** How a step of a batch run settled. */
export type StepState<Saved> =
  /** It ran and left a record (whatever its outcome). */
  | { readonly kind: "ran"; readonly run: SingleRun<Saved> }
  /** Its run rejected before or after the entry ran, which stopped the batch. */
  | { readonly kind: "host-failed"; readonly error: unknown }
  /** It was selected, but the batch was cancelled or stopped before it. */
  | { readonly kind: "not-run" }
  | { readonly kind: "not-selected" };

export type SettledStep<Saved> = {
  readonly step: ResolvedStep;
  readonly state: StepState<Saved>;
};

/** What a batch run reports while it runs: one `settled` per step of the file, in file order. */
export type BatchEvent<Saved> =
  | { readonly kind: "started"; readonly step: ResolvedStep }
  | { readonly kind: "progress"; readonly step: ResolvedStep; readonly text: string }
  | ({ readonly kind: "settled" } & SettledStep<Saved>);

export type BatchRun<Saved> = {
  /** When the batch run started (ISO 8601), as every record of the run says it. */
  readonly startedAt: string;
  /** Every step of the file, in file order. */
  readonly steps: readonly SettledStep<Saved>[];
};

export const runBatch = async <Saved>(options: {
  readonly batch: ResolvedBatch;
  /** The 1-based numbers of the steps to run; the others are not selected. */
  readonly selected: readonly number[];
  readonly signal: AbortSignal;
  readonly host: RunHost<Saved>;
  readonly onEvent: (event: BatchEvent<Saved>) => void;
}): Promise<BatchRun<Saved>> => {
  const { batch, signal, host, onEvent } = options;
  const startedAt = new Date().toISOString();
  const steps: SettledStep<Saved>[] = [];
  let stopped = false;
  for (const step of batch.steps) {
    let state: StepState<Saved>;
    if (!options.selected.includes(step.number)) state = { kind: "not-selected" };
    else if (stopped || signal.aborted) state = { kind: "not-run" };
    else {
      onEvent({ kind: "started", step });
      try {
        const run = await runEntry(host, {
          entry: step.entry,
          parameters: step.parameters,
          signal,
          progress: (text) => onEvent({ kind: "progress", step, text }),
          batch: { id: batch.id, startedAt, step: step.number, steps: batch.steps.length },
        });
        state = { kind: "ran", run };
      } catch (error) {
        stopped = true;
        state = { kind: "host-failed", error };
      }
    }
    steps.push({ step, state });
    onEvent({ kind: "settled", step, state });
  }
  return { startedAt, steps };
};

const plural = (count: number, noun: string): string => `${count} ${noun}${count === 1 ? "" : "s"}`;

/**
 * The one line that sums a batch run up, for the Node command's last line and the shell's status
 * line: `batch <id>: 9 steps — 7 done (1 with verdict fail), 1 error, 1 did not run`.
 */
export const batchSummary = (id: string, run: BatchRun<unknown>): string => {
  const count = (test: (state: StepState<unknown>) => boolean): number =>
    run.steps.filter(({ state }) => test(state)).length;
  const outcome = (wanted: Outcome) => (state: StepState<unknown>) =>
    state.kind === "ran" && state.run.record.outcome === wanted;
  const failed = count(
    (state) =>
      state.kind === "ran" &&
      state.run.record.outcome === "done" &&
      state.run.record.verdict === "fail",
  );
  const parts: [number, string][] = [
    [count(outcome("done")), failed === 0 ? "done" : `done (${failed} with verdict fail)`],
    [count(outcome("error")), "error"],
    [count(outcome("cancelled")), "cancelled"],
    [count((state) => state.kind === "host-failed"), "failed around the run"],
    [count((state) => state.kind === "not-run"), "did not run"],
    [count((state) => state.kind === "not-selected"), "not selected"],
  ];
  const counted = parts
    .filter(([number]) => number > 0)
    .map(([number, what]) => `${number} ${what}`)
    .join(", ");
  return `batch ${id}: ${plural(run.steps.length, "step")} — ${counted}`;
};
