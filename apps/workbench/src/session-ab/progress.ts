/**
 * The session A/B entries' status lines: where the runner is (round, visit, arm, timed or not) and
 * how many runs have ended, the count probes' steps, and the bytes a component fetch has moved.
 *
 * Every line is reported outside the timed spans: the arms report their steps from open, from the
 * report the runner calls after the clock stopped, and from close, never from a run. A browser
 * paints a status change at its next rendering opportunity, which could fall inside the timed run
 * that follows a report; so the arm stepper hands the event loop one turn after each report, before
 * the arm hands control back to the runner.
 */

import type { AssetProgress } from "@karume/hub";

import type { ArmStep, ProbeStep, RunPosition } from "./run.ts";

const MIB = 2 ** 20;

const positionText = (position: RunPosition | undefined, arm: string): string =>
  position === undefined
    ? arm
    : `${position.round === 0 ? "warm-up round" : `round ${position.round}`} · visit ${position.visit} · ${arm}${position.timed ? " (timed)" : " (untimed)"}`;

/**
 * The line of one arm step: where the runner is, and how many runs have ended. `opens` names what an
 * arm opens (the Gemma target's pipeline, the Session targets' Session).
 */
export const armStepLine = (
  step: ArmStep,
  positions: readonly RunPosition[],
  ended: number,
  opens: string,
): string => {
  const count = `${ended} / ${positions.length} runs`;
  switch (step.step) {
    case "open":
      return `opening the ${opens} · ${positionText(positions[ended], step.arm)} · ${count}`;
    case "ran":
      return `ran · ${positionText(positions[ended - 1], step.arm)} · ${count}`;
    case "close":
      return `closing the ${opens} of ${step.arm} · ${count}`;
  }
};

/** The line of one probe step, and how many probes have ended. */
export const probeStepLine = (step: ProbeStep, ended: number, total: number): string => {
  const count = `${ended} / ${total} probes`;
  switch (step.step) {
    case "open":
      return `probing the counts · building the pipeline of ${step.arm} · ${count}`;
    case "run":
      return `probing the counts · one untimed generation of ${step.arm} · ${count}`;
    case "close":
      return `probing the counts · closing the pipeline of ${step.arm} · ${count}`;
  }
};

/** The line of a component's weight fetch, in whole MiB. */
export const fetchLine = (component: string, progress: AssetProgress): string =>
  `fetching the ${component} component · ${Math.round(progress.loaded / MIB)} / ${Math.round(progress.total / MIB)} MiB`;

/** One turn of the event loop (a macrotask): a browser may render before it ends. */
export const nextTask = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

/**
 * The step reporter of an A/B's arms: on every step it reports {@link armStepLine}, then waits one
 * turn of the event loop (`yieldTurn`) before the arm goes on. A cancel takes effect at the next
 * visit's open (a run takes no signal on the Session targets, and an open is never timed).
 */
export const armStepper = (input: {
  readonly positions: readonly RunPosition[];
  readonly opens: string;
  readonly signal: AbortSignal;
  readonly progress: (text: string) => void;
  readonly yieldTurn?: () => Promise<void>;
}): ((step: ArmStep) => Promise<void>) => {
  const { positions, opens, signal, progress, yieldTurn = nextTask } = input;
  let ended = 0;
  return async (step) => {
    if (step.step === "open") signal.throwIfAborted();
    if (step.step === "ran") ended += 1;
    progress(armStepLine(step, positions, ended, opens));
    await yieldTurn();
  };
};
