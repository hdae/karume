/**
 * The time of every denoise step of Anima generations, per preset: one pipeline per preset
 * (`anima.Pipeline.fromPretrained` on a device of its own, acquired for that preset and destroyed
 * after it), `generations` images of one fixed request in the default per-stage residency, every
 * step, stage and residency change timed by the wall clock in the `onEvent` callbacks. Per preset
 * the median step and the steps over ×3 of it (the stalled steps) — the periodic stall the int8
 * attention presets showed on one device. No image is kept. It only measures: no verdict.
 *
 * SPEC: docs/spec/workbench.md#the-entries
 */

import { loadManifest, resolveSelection } from "@karume/hub";
import { type GenerateEvent, Pipeline } from "@karume/models/anima";
import { acquireGpu } from "@karume/runtime";

import {
  ANIMA_PRESETS,
  animaStepTimingData,
  animaStepTimingSummary,
  type GenerationTiming,
  generationTiming,
  type PresetTiming,
  presetTiming,
  type TimedEvent,
  type TimingEvent,
  TIMING_MOUNTS,
  timingModel,
} from "../../anima-step-timing/record.ts";
import {
  choicesValue,
  choiceValue,
  type Entry,
  integerValue,
  ParameterError,
  textValue,
} from "../../entry.ts";
import { SOURCE_KINDS, sourceParameter } from "../../local-parameters.ts";

/** A neutral request; its words do not change the shapes the DiT runs at. */
const PROMPT =
  "a lighthouse on a rocky coast at sunset, waves, clouds, detailed, masterpiece, best quality";
const FIRST_SEED = 42;

/** The event without its payload (a `denoise-step` carries a closure over the step's latent). */
const timingEvent = (event: GenerateEvent): TimingEvent | undefined => {
  switch (event.kind) {
    case "stage":
      return { kind: "stage", component: event.component, at: event.at };
    case "denoise-step":
      return { kind: "denoise-step", step: event.step, steps: event.steps };
    case "residency":
      return { kind: "residency", action: event.action, reason: event.reason };
    case "vae-tile":
      return undefined;
  }
};

/** A number of the model's `pipelineConfig.defaults`, or `null` when it declares none. */
const defaultNumber = (config: Readonly<Record<string, unknown>>, key: string): number | null => {
  const defaults = config["defaults"];
  if (typeof defaults !== "object" || defaults === null) return null;
  const value: unknown = Object.entries(defaults).find(([name]) => name === key)?.[1];
  return typeof value === "number" ? value : null;
};

const entry: Entry = {
  id: "anima-step-timing",
  title: "Anima step timing",
  description:
    "The wall-clock time of every denoise step of Anima generations per preset (one pipeline and one device per preset, the default per-stage residency): each stage's time, the median step and the steps slower than ×3 of it. No verdict, no image kept.",
  parameters: [
    {
      kind: "choice",
      name: "mount",
      label: "Mount",
      options: TIMING_MOUNTS,
      default: "anima-2.9b",
    },
    {
      kind: "text",
      name: "model",
      label: "Model (empty: anima-v1.0 on the anima mount, the manifest's default elsewhere)",
      default: "",
    },
    {
      kind: "choices",
      name: "presets",
      label: "Presets, each its own pipeline and device, in this order",
      options: ANIMA_PRESETS,
      default: ["bf16+dit8-a8", "bf16+dit8-a8-attn8-s16"],
    },
    {
      kind: "integer",
      name: "generations",
      label: "Generations per preset (seeds 42, 43, …)",
      min: 1,
      max: 50,
      default: 3,
    },
    { kind: "integer", name: "width", label: "Width", min: 64, max: 4096, default: 1024 },
    { kind: "integer", name: "height", label: "Height", min: 64, max: 4096, default: 1024 },
    sourceParameter("How a browser reads the distribution"),
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution }) => {
    const mount = choiceValue(parameters, "mount", TIMING_MOUNTS);
    const presets = choicesValue(parameters, "presets", ANIMA_PRESETS);
    if (presets.length === 0) throw new ParameterError("presets", "name at least one preset");
    const generations = integerValue(parameters, "generations");
    const resolution = {
      width: integerValue(parameters, "width"),
      height: integerValue(parameters, "height"),
    };
    const opened = await openDistribution(mount, choiceValue(parameters, "source", SOURCE_KINDS));
    const { manifest } = await loadManifest(opened.ref, { signal });
    const model = timingModel(mount, textValue(parameters, "model"), manifest.defaultModel);
    // Every preset resolved before the GPU is touched: an unknown model or preset fails at once.
    for (const preset of presets) resolveSelection(manifest, { model, quant: preset });
    const modelEntry = manifest.models[model];
    const seeds = Array.from({ length: generations }, (_, index) => FIRST_SEED + index);

    const timings: PresetTiming[] = [];
    for (const preset of presets) {
      signal.throwIfAborted();
      const quant = modelEntry.quants[preset];
      const gpu = await acquireGpu({
        ...gpuOptions,
        ...(quant.gpuFeatures?.shaderF16 === true ? { shaderF16: true } : {}),
      });
      try {
        progress(`${preset}: building the pipeline`);
        const building = performance.now();
        const pipeline = await Pipeline.fromPretrained(opened.ref, {
          gpu,
          model,
          quant: preset,
          signal,
        });
        const buildSeconds = (performance.now() - building) / 1000;
        try {
          const runs: GenerationTiming[] = [];
          for (const [index, seed] of seeds.entries()) {
            signal.throwIfAborted();
            const label = `${preset}: generation ${index + 1} of ${generations}`;
            progress(label);
            const events: TimedEvent[] = [];
            const started = performance.now();
            await pipeline.generate({
              prompt: PROMPT,
              seed,
              resolution,
              onEvent: (event) => {
                const ms = performance.now();
                // A throw from the callback is how a generate stops at a step (GenerateRequest.onEvent).
                signal.throwIfAborted();
                const timed = timingEvent(event);
                if (timed === undefined) return;
                events.push({ event: timed, ms });
                if (timed.kind === "denoise-step") {
                  progress(`${label}: step ${timed.step} of ${timed.steps}`);
                }
              },
            });
            runs.push(generationTiming(seed, started, performance.now(), events));
          }
          timings.push(
            presetTiming({ preset, session: quant.session, buildSeconds, generations: runs }),
          );
        } finally {
          await pipeline.dispose();
        }
      } finally {
        gpu.destroy();
      }
    }

    const data = animaStepTimingData({
      request: {
        mount,
        model,
        presets,
        generations,
        resolution,
        steps: defaultNumber(modelEntry.pipelineConfig, "steps"),
        guidanceScale: defaultNumber(modelEntry.pipelineConfig, "guidanceScale"),
        prompt: PROMPT,
        seeds,
      },
      distribution: opened.facts,
      presets: timings,
    });
    return { summary: animaStepTimingSummary(data), data };
  },
};

export default entry;
