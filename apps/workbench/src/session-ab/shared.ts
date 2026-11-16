/**
 * What both session A/B entries share (the Gemma 4 decode of `gemma4-session-ab`, the Session
 * targets of `session-ab`): the two parameters they declare alike, the two arms of a request, the
 * preset rule, the environment the runner's record names, the order the runner visits arms in, and
 * the arms' step reports.
 *
 * {@link runPositions} is the runner's visiting order as a list, so an entry can say where a run is
 * (`runSessionAb` reports no progress of its own); `vp test` drives the real runner with fake arms
 * against it.
 */

import type { Manifest } from "@karume/hub";
import type { SessionAbAdapter, SessionAbArm, SessionAbEnvironment } from "@karume/runtime/tune";

import { type IntegerParameter, type NumberParameter, ParameterError } from "../entry.ts";
import { type FoundPreset, modelPreset } from "../model-preset.ts";

/** The timed rounds: at least the runner's minimum of 3, at most a generous 1000. */
export const ROUNDS_PARAMETER: IntegerParameter = {
  kind: "integer",
  name: "rounds",
  label: "Timed rounds (one untimed warm-up round comes first)",
  min: 3,
  max: 1000,
  default: 4,
};

/** The minimum gain: a fraction of the base's time, 0 to 1 (a gain cannot exceed the whole time). */
export const MIN_GAIN_PARAMETER: NumberParameter = {
  kind: "number",
  name: "minGain",
  label: "Minimum gain (a fraction of the base's time: 0.01 = 1%)",
  min: 0,
  max: 1,
  default: 0.01,
};

/** What every A/B runs and is judged with: the timed rounds and the minimum gain. */
export type RoundsAndGain = {
  readonly rounds: number;
  readonly minGain: number;
};

/** One arm of a request: its name and the rules it passes. */
export type RequestArm = { readonly name: string; readonly fusionRules: readonly string[] };

/** The arms of every request: `base` with no rule first, then `rule` with `ruleName`. */
export const twoArms = (ruleName: string): readonly RequestArm[] => [
  { name: "base", fusionRules: [] },
  { name: "rule", fusionRules: [ruleName] },
];

/**
 * The preset a run loads: `chosen` when it is a preset of `model`, the model's `defaultQuant` when
 * `chosen` is empty. Anything else is a {@link ParameterError} of `preset` that lists the presets.
 */
export const presetOf = (manifest: Manifest, model: string, chosen: string): FoundPreset => {
  const preset = modelPreset(manifest, model, chosen === "" ? undefined : chosen);
  if (preset.ok) return preset;
  throw new ParameterError(
    "preset",
    `'${chosen}' is not a preset of model '${model}'; its presets are ${preset.presets.join(", ")}`,
  );
};

/**
 * What the runner's record says about where an A/B ran on `adapter`, now. WebGPU does not tell the
 * backend and the runtime's record requires one: it is the host's (`RunContext.backend`), and
 * `unknown` when a browser run declared none.
 */
export const runnerEnvironment = (
  adapter: SessionAbAdapter,
  backend: string | undefined,
): SessionAbEnvironment => ({
  adapter,
  userAgent: navigator.userAgent,
  backend: backend ?? "unknown",
  date: new Date().toISOString(),
});

/** Where one run sits in the runner's order. */
export type RunPosition = {
  /** 0 = the untimed warm-up round, 1 to `rounds` the timed rounds. */
  readonly round: number;
  /** 1 = the forward pass over the arms, 2 = the reverse pass. */
  readonly visit: 1 | 2;
  readonly arm: number;
  /** 1 or, with `openPerVisit`, 2 (the fresh pipeline's untimed first run comes first). */
  readonly run: 1 | 2;
  /** Whether the runner times this run. */
  readonly timed: boolean;
};

/**
 * Every run in the runner's order: per round the arms forward then in reverse (A B B A), one run
 * per visit, or two with `openPerVisit` (an untimed one on the fresh pipeline, then the timed one).
 */
export const runPositions = (
  arms: number,
  rounds: number,
  openPerVisit: boolean,
): readonly RunPosition[] => {
  const order = Array.from({ length: arms }, (_, arm) => arm);
  const runs: (1 | 2)[] = openPerVisit ? [1, 2] : [1];
  const positions: RunPosition[] = [];
  for (let round = 0; round <= rounds; round += 1) {
    for (const [visit, arm] of [
      ...order.map((arm) => [1, arm] as const),
      ...order.toReversed().map((arm) => [2, arm] as const),
    ]) {
      for (const run of runs) {
        positions.push({ round, visit, arm, run, timed: round > 0 && run === runs.length });
      }
    }
  }
  return positions;
};

/**
 * What one arm did, as a caller shows it: `open` before the arm builds its pipeline or Session, `ran`
 * after a run returned (from the report, outside the timed span), `close` before it disposes it.
 */
export type ArmStep = { readonly arm: string; readonly step: "open" | "ran" | "close" };

/**
 * Where an arm's steps go. The arm awaits it before it goes on, so a reporter MAY hand the event
 * loop a turn (a browser paints then) before the runner's next timed run.
 */
export type ArmStepReporter = (step: ArmStep) => void | Promise<void>;

/**
 * `arm` with its steps reported, as the Gemma target's arms report theirs: `onStep` is called (and
 * awaited) with `open` before the arm opens, `ran` after its report has read a run's counts and
 * `close` before it closes — never inside `run`, so a status update never lands in a timed span.
 * The arm's runs and counts pass through unchanged.
 */
export const steppedArm = (arm: SessionAbArm, onStep: ArmStepReporter): SessionAbArm => ({
  name: arm.name,
  open: async () => {
    await onStep({ arm: arm.name, step: "open" });
    await arm.open();
  },
  run: arm.run,
  report: async () => {
    const counts = (await arm.report?.()) ?? {};
    await onStep({ arm: arm.name, step: "ran" });
    return counts;
  },
  close: async () => {
    await onStep({ arm: arm.name, step: "close" });
    await arm.close();
  },
});

/**
 * The prefix of a rule's dispatch keys: an external rule `<namespace>/<rule>` keys every dispatch
 * under `ext:<namespace>:` (ADR 0148 decision 5).
 */
export const ruleKeyPrefix = (ruleName: string): string => {
  const slash = ruleName.indexOf("/");
  if (slash <= 0) throw new Error(`The rule name "${ruleName}" is not <namespace>/<rule>`);
  return `ext:${ruleName.slice(0, slash)}:`;
};
