/**
 * The record of the `anima-step-timing` entry, its pure parts: the timeline of one generation's
 * `onEvent` callbacks folded into step and stage times, the stalled steps of a preset, the record
 * `karume-anima-step-timing/1` and its summary lines.
 *
 * SPEC: docs/spec/workbench.md#the-anima-step-timing-record
 */

import type { SessionSpec } from "@karume/hub";
import type { MountName } from "@karume/local-distributions/mounts";

import type { DistributionFacts } from "../entry.ts";

export const ANIMA_STEP_TIMING_FORMAT = "karume-anima-step-timing/1";

/** The Anima mounts the entry reads. */
export const TIMING_MOUNTS = [
  "anima-2.9b",
  "anima",
  "anima-extra",
] as const satisfies readonly MountName[];
export type TimingMount = (typeof TIMING_MOUNTS)[number];

/** The presets every Anima distribution declares, in the order a run visits them. */
export const ANIMA_PRESETS = [
  "bf16",
  "bf16+dit8",
  "bf16+dit8-a8",
  "bf16+dit8-a8-attn8",
  "bf16+dit8-a8-attn8-s16",
  "bf16-c16",
] as const;
export type AnimaPreset = (typeof ANIMA_PRESETS)[number];

/**
 * The model of the `anima` mount when none is named: its manifest's default is a turbo model (8
 * steps without classifier-free guidance), and the stall was seen on the 20-step base with guidance.
 */
export const ANIMA_MOUNT_MODEL = "anima-v1.0";

/** A step slower than this many times its preset's median step is a stalled step. */
export const STALL_FACTOR = 3;

/** How a step's and a stage's time are taken (the record's `clock`). */
export const CLOCK =
  "wall clock: performance.now() read in each onEvent callback of generate; a step runs from the previous denoise-step callback (step 1: from the transformer stage's start, so it includes building the DiT Session) to its own";

/** The model a run times: the one named, else {@link ANIMA_MOUNT_MODEL} on `anima`, else the manifest's default. */
export const timingModel = (mount: TimingMount, named: string, manifestDefault: string): string => {
  if (named !== "") return named;
  return mount === "anima" ? ANIMA_MOUNT_MODEL : manifestDefault;
};

/** The generate events the timeline reads (the `onEvent` events without their payloads). */
export type TimingEvent =
  | { readonly kind: "stage"; readonly component: string; readonly at: "start" | "end" }
  | { readonly kind: "denoise-step"; readonly step: number; readonly steps: number }
  | { readonly kind: "residency"; readonly action: string; readonly reason: string };

/** One callback: its event and `performance.now()` when it ran. */
export type TimedEvent = { readonly event: TimingEvent; readonly ms: number };

export type StageTime = { readonly component: string; readonly seconds: number };

export type ResidencyRecord = {
  readonly action: string;
  readonly reason: string;
  /** From the start of `generate`. */
  readonly atSeconds: number;
};

/** One generation's times. */
export type GenerationTiming = {
  readonly seed: number;
  /** `generate` from its call until it resolved. */
  readonly totalSeconds: number;
  /** Step 1 first ({@link CLOCK}). */
  readonly stepSeconds: readonly number[];
  /** Each stage from its start to its end callback, in the order they ended. */
  readonly stages: readonly StageTime[];
  /** Every residency event (none in the default per-stage residency). */
  readonly residency: readonly ResidencyRecord[];
};

/**
 * Folds the callbacks of one generation into its times. A step event out of order and a stage that
 * ends without having started are errors: the timeline would not mean what the record says.
 */
export const generationTiming = (
  seed: number,
  startedMs: number,
  endedMs: number,
  events: readonly TimedEvent[],
): GenerationTiming => {
  const stepSeconds: number[] = [];
  const stages: StageTime[] = [];
  const residency: ResidencyRecord[] = [];
  const open = new Map<string, number>();
  let previousMark: number | undefined;
  for (const { event, ms } of events) {
    switch (event.kind) {
      case "stage": {
        if (event.at === "start") {
          open.set(event.component, ms);
          if (event.component === "transformer") previousMark = ms;
          break;
        }
        const started = open.get(event.component);
        if (started === undefined) {
          throw new Error(`stage ${event.component} ended without having started`);
        }
        open.delete(event.component);
        stages.push({ component: event.component, seconds: (ms - started) / 1000 });
        break;
      }
      case "denoise-step": {
        if (event.step !== stepSeconds.length + 1 || previousMark === undefined) {
          throw new Error(
            `denoise step ${event.step} arrived after ${stepSeconds.length} steps${previousMark === undefined ? " and before the transformer stage started" : ""}`,
          );
        }
        stepSeconds.push((ms - previousMark) / 1000);
        previousMark = ms;
        break;
      }
      case "residency":
        residency.push({
          action: event.action,
          reason: event.reason,
          atSeconds: (ms - startedMs) / 1000,
        });
        break;
    }
  }
  return { seed, totalSeconds: (endedMs - startedMs) / 1000, stepSeconds, stages, residency };
};

export const median = (values: readonly number[]): number => {
  if (values.length === 0) throw new Error("the median of no values");
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

export type StalledStep = {
  /** 1-based, as the summary names it. */
  readonly generation: number;
  /** 1-based. */
  readonly step: number;
  readonly seconds: number;
  /** `seconds` ÷ the preset's median step. */
  readonly ratio: number;
};

export type StepStatistics = {
  /** The median over every step but step 1 of every generation. */
  readonly medianStepSeconds: number;
  /** How many steps the median and the stall rule read. */
  readonly stepsCounted: number;
  /** The steps slower than {@link STALL_FACTOR} × the median, in generation and step order. */
  readonly stalled: readonly StalledStep[];
};

/**
 * The median step and the stalled steps of a preset's generations. Step 1 is left out of both: its
 * time includes building the DiT Session (the default per-stage residency builds it in every
 * generate), which is no step of the denoise loop.
 */
export const stepStatistics = (
  generations: readonly Pick<GenerationTiming, "stepSeconds">[],
): StepStatistics => {
  const counted = generations.flatMap((generation) => generation.stepSeconds.slice(1));
  const medianStepSeconds = median(counted);
  const stalled = generations.flatMap((generation, index) =>
    generation.stepSeconds.flatMap((seconds, step) =>
      step > 0 && seconds > STALL_FACTOR * medianStepSeconds
        ? [{ generation: index + 1, step: step + 1, seconds, ratio: seconds / medianStepSeconds }]
        : [],
    ),
  );
  return { medianStepSeconds, stepsCounted: counted.length, stalled };
};

/** One preset's run: its pipeline's build and its generations. */
export type PresetTiming = StepStatistics & {
  readonly preset: string;
  /** The preset's Session options as the manifest declares them (`attentionCompute` and the like). */
  readonly session: SessionSpec;
  /** `fromPretrained` until it resolved. */
  readonly buildSeconds: number;
  readonly generations: readonly GenerationTiming[];
};

export type AnimaStepTimingRequest = {
  readonly mount: TimingMount;
  readonly model: string;
  readonly presets: readonly string[];
  readonly generations: number;
  readonly resolution: { readonly width: number; readonly height: number };
  /** The model's `pipelineConfig.defaults` the generations ran with (`null` when it declares none). */
  readonly steps: number | null;
  readonly guidanceScale: number | null;
  readonly prompt: string;
  /** Generation g (1-based) of every preset uses `seeds[g - 1]`. */
  readonly seeds: readonly number[];
};

export type AnimaStepTimingData = {
  readonly format: typeof ANIMA_STEP_TIMING_FORMAT;
  readonly request: AnimaStepTimingRequest;
  readonly distribution: DistributionFacts;
  readonly clock: typeof CLOCK;
  readonly stallRule: { readonly factor: number; readonly firstStepLeftOut: true };
  readonly presets: readonly PresetTiming[];
};

/** A preset's record from its build and generations (its statistics derived from them). */
export const presetTiming = (fields: Omit<PresetTiming, keyof StepStatistics>): PresetTiming => ({
  ...fields,
  ...stepStatistics(fields.generations),
});

export const animaStepTimingData = (
  fields: Pick<AnimaStepTimingData, "request" | "distribution" | "presets">,
): AnimaStepTimingData => ({
  format: ANIMA_STEP_TIMING_FORMAT,
  ...fields,
  clock: CLOCK,
  stallRule: { factor: STALL_FACTOR, firstStepLeftOut: true },
});

const s1 = (value: number): string => value.toFixed(1);
const s2 = (value: number): string => value.toFixed(2);

const stalledText = (stalled: readonly StalledStep[]): string =>
  stalled.length === 0
    ? "0 stalled steps"
    : `${stalled.length} stalled step${stalled.length === 1 ? "" : "s"} (${stalled
        .map(
          (step) =>
            `generation ${step.generation} step ${step.step} ${s1(step.seconds)} s ×${s1(step.ratio)}`,
        )
        .join("; ")})`;

/** The lines a person reads first: the request, per preset its stalls, and every step list. */
export const animaStepTimingSummary = (data: AnimaStepTimingData): string[] => {
  const { request } = data;
  const residency = data.presets.flatMap((preset) =>
    preset.generations.flatMap((generation, index) =>
      generation.residency.map(
        (event) =>
          `${preset.preset} generation ${index + 1}: ${event.action} / ${event.reason} at ${s1(event.atSeconds)} s`,
      ),
    ),
  );
  return [
    `${request.mount} / ${request.model} at ${request.resolution.width}×${request.resolution.height}, ${request.steps ?? "default"} steps, guidance ${request.guidanceScale ?? "default"}: ${request.presets.length} preset${request.presets.length === 1 ? "" : "s"} × ${request.generations} generation${request.generations === 1 ? "" : "s"}; steps timed by the wall clock between onEvent callbacks, a stall is a step over ×${data.stallRule.factor} the preset's median (step 1 left out)`,
    ...data.presets.map(
      (preset) =>
        `${preset.preset}: median step ${s2(preset.medianStepSeconds)} s over ${preset.stepsCounted} steps, ${stalledText(preset.stalled)}; generations ${preset.generations.map((generation) => `${s1(generation.totalSeconds)} s`).join(", ")}`,
    ),
    ...data.presets.flatMap((preset) =>
      preset.generations.map(
        (generation, index) =>
          `${preset.preset} #${index + 1} steps ${generation.stepSeconds.map(s1).join(" ")}`,
      ),
    ),
    residency.length === 0 ? "residency events: none" : `residency events: ${residency.join("; ")}`,
  ];
};
