/**
 * The Gemma 4 decode bench: a decode A/B of arms — a model of regular Gemma 4, its q4_0 QAT line
 * (`gemma.Pipeline`) or the mobile QAT distribution (`gemma4QatMobile.Pipeline`) × a preset × the
 * GPU tap on, split per dispatch or off × linear at the level `reorder` × the integer GEMV × the PLE
 * on the host or the GPU × the decode run-ahead depth — visited in A B B A rounds over a fixed prompt set, `chats` fresh chats per
 * prompt and visit (chat 0 is the cold one and is left out of the summary). It reports wall-clock
 * decode figures and, through the GPU tap (`src/gemma4-bench/tap.ts`), the GPU time and the
 * dispatches, passes and submits per token.
 *
 * The device is acquired through the tap, so karume's timing mode stays off and the product's own
 * pass structure is what is timed. One pipeline is resident at a time: each visit builds its arm's
 * pipeline and releases it before the next. Every arm is resolved before the device is acquired, so
 * a refused arm touches no GPU.
 */

import { chatPrompt } from "@karume/models/gemma";

import { booleanValue, choiceValue, type Entry, textValue } from "../../entry.ts";
import { buildGemmaPipeline, type GemmaPipeline } from "../../gemma4/pipeline.ts";
import { BENCH_PROMPT_IDS, SAMPLER_KINDS, samplerOf } from "../../gemma4-bench/plan.ts";
import { benchProgressLine } from "../../gemma4-bench/progress.ts";
import { benchSettingsOf, openBenchArms } from "../../gemma4-bench/request.ts";
import { decodeBenchResult } from "../../gemma4-bench/result.ts";
import { type BenchPipeline, type RunArm, settleDecodeBench } from "../../gemma4-bench/run.ts";
import { createGpuTap } from "../../gemma4-bench/tap.ts";
import {
  acquireWithSubgroups,
  SOURCE_KINDS,
  sourceParameter,
  SUBGROUPS_PARAMETER,
} from "../../local-parameters.ts";
import { sha256Hex } from "../../sha256.ts";

/** What the bench loop needs of a Gemma 4 pipeline (its own tokenizer renders the prompt). */
const benchPipelineOf = (pipeline: GemmaPipeline): BenchPipeline => ({
  encodePrompt: (messages) => chatPrompt(pipeline.tokenizer, messages),
  chat: (messages, options) => pipeline.chat(messages, options),
  dispose: () => pipeline.dispose(),
});

const entry: Entry = {
  id: "gemma4-bench",
  title: "Gemma 4 decode bench",
  description:
    "Decode A/B of Gemma 4 arms (model × preset × GPU tap on, split per dispatch or off × linear at reorder × integer GEMV × PLE on the GPU × decode run-ahead depth) in A B B A rounds over a fixed prompt set, one pipeline resident at a time: decode tokens/s and ms/token, and through the GPU tap the GPU time and dispatches, passes and submits per token.",
  // The minima are the bench's own (a warm chat besides the cold one, a decode token after the
  // prefill's, one round); the maxima only bound a typo.
  parameters: [
    {
      kind: "text",
      name: "arms",
      label:
        "Arms, comma-separated, arm 1 the one the others are compared with; an arm is <mount>/<model>[@<preset>][:<switch>]… with a mount of gemma4, gemma4-qat, gemma4-qat-mobile, no preset for the manifest's default, and the switches tap-off (the tap is on otherwise), split (one timed pass per dispatch), linear-reorder (linear at the level reorder on a preset that does not declare it), integer (integer GEMV), ple-gpu (PLE on the GPU), ahead2 / ahead4 / ahead8 (decode run-ahead depth; needs ple-gpu except on 12b) — e.g. gemma4/e2b,gemma4-qat/e2b@i4-fast:integer (empty: one arm per mount whose mirror is present, its default model and preset, regular Gemma 4 first)",
      default: "",
    },
    {
      kind: "integer",
      name: "chats",
      label: "Chats per prompt per visit (chat 0 is the cold one and is left out of the summary)",
      min: 2,
      max: 1000,
      default: 3,
    },
    {
      kind: "integer",
      name: "newTokens",
      label: "New tokens per chat (maxNewTokens; the first comes from the prefill)",
      min: 2,
      max: 32_768,
      default: 128,
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Rounds (even rounds visit the arms in order, odd rounds in reverse)",
      min: 1,
      max: 1000,
      default: 4,
    },
    {
      kind: "choice",
      name: "sampler",
      label: "Sampler: greedy (the same tokens in every round) or the distribution's with seed 0",
      options: SAMPLER_KINDS,
      default: "greedy",
    },
    {
      kind: "choices",
      name: "prompts",
      label:
        "Prompts, run in this order in every visit (few-ids: fewer than 32 distinct token ids, many-ids: at least 32)",
      options: BENCH_PROMPT_IDS,
      default: BENCH_PROMPT_IDS,
    },
    SUBGROUPS_PARAMETER,
    sourceParameter("How a browser reads the distributions"),
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution }) => {
    const settings = benchSettingsOf(parameters);
    const openedArms = await openBenchArms(
      textValue(parameters, "arms"),
      choiceValue(parameters, "source", SOURCE_KINDS),
      openDistribution,
      signal,
    );
    const arms: readonly RunArm[] = openedArms.map((arm) => ({
      ...arm,
      sampler: samplerOf(settings.sampler, arm.declaredSampler),
    }));
    const host: GPU | undefined = gpuOptions.gpu ?? navigator.gpu;
    if (host === undefined) throw new Error("This host has no WebGPU (navigator.gpu)");
    // The tap wraps the host's GPU and the device is acquired through it: it requests
    // timestamp-query itself and hides it from the runtime, so karume's timing mode stays off.
    const tap = createGpuTap(host);
    const gpu = await acquireWithSubgroups({ gpu: tap.gpu }, booleanValue(parameters, "subgroups"));
    try {
      const probe = tap.probe(gpu.device);
      if (probe === undefined) throw new Error("The device was not acquired through the GPU tap");
      if (gpu.gpuTimingEnabled) {
        throw new Error(
          "The device reports gpuTimingEnabled: the tap failed to hide timestamp-query, and karume would encode one dispatch per pass. The bench does not run on it.",
        );
      }
      const outcome = await settleDecodeBench({
        arms,
        settings,
        signal,
        dependencies: {
          build: async (arm, buildSignal) =>
            benchPipelineOf(
              await buildGemmaPipeline({
                ref: arm.distribution.ref,
                gpu,
                load: { mount: arm.mount, model: arm.model, quant: arm.quant },
                signal: buildSignal,
                ...(arm.session === undefined ? {} : { sessionOptions: arm.session }),
                ...(arm.pipeline === undefined ? {} : { pipelineOptions: arm.pipeline }),
              }),
            ),
          probe,
          now: () => performance.now(),
          sha256: sha256Hex,
        },
        onProgress: (step) => progress(benchProgressLine(step, arms, settings.chats)),
      });
      return decodeBenchResult({ timestampQuery: probe.timestamps, settings, arms, outcome });
    } finally {
      gpu.destroy();
    }
  },
};

export default entry;
