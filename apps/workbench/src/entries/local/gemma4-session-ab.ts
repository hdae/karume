/**
 * The session A/B on a Gemma 4 decode: the wall-clock A/B of `@karume/runtime/tune`
 * (`runSessionAb`, judged by `judgeSessionAb`) of two arms that load the same model and preset —
 * `base` with no rule, `rule` with the candidate rule `candidate/gelu_mul` of
 * `@karume/candidate-rules` through the gemma pipeline's experimental `fusionRules`. Every timed run
 * is one greedy generation from a fixed prompt whose token ids the runner compares with the base's
 * first run, so identity is token identity. The arms are built without the diagnostics callback
 * (every run is the production path); the rule's hits and the dispatches come from a count probe of
 * each arm before the A/B.
 *
 * The verdict is token identity; the judgement (adopt, reject, inconclusive) is a measurement and
 * goes to the summary and the data.
 */

import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { chatPrompt } from "@karume/models/gemma";
import { acquireGpu, type SessionOptions } from "@karume/runtime";
import { judgeSessionAb, runSessionAb } from "@karume/runtime/tune";

import { booleanValue, choiceValue, type Entry, textValue } from "../../entry.ts";
import { buildGemmaPipeline, type GemmaPipeline } from "../../gemma4/pipeline.ts";
import { EXPERT_SPECULATION_DEPTH } from "../../gemma4/settings.ts";
import {
  MODELS_VERSION,
  type SessionAbArmProbe,
  sessionAbGemmaData,
} from "../../session-ab/data.ts";
import { gemmaRequest, openGemmaModel, parseModelText } from "../../session-ab/gemma-target.ts";
import { armStepper, probeStepLine } from "../../session-ab/progress.ts";
import { SESSION_AB_DEFAULTS, SESSION_AB_LIMITS, SOURCE_KINDS } from "../../session-ab/request.ts";
import {
  type ArmSpec,
  pipelineArm,
  probeCounts,
  runnerEnvironment,
  runPositions,
} from "../../session-ab/run.ts";
import { sessionAbGemmaResult } from "../../session-ab/summary.ts";

const entry: Entry = {
  id: "gemma4-session-ab",
  title: "Gemma 4 session A/B",
  description:
    "Wall-clock A/B (runSessionAb, judgeSessionAb) of a Gemma 4 greedy decode with and without the candidate rule candidate/gelu_mul, the counts from a probe of each arm first. The verdict is token identity; the judgement is in the summary.",
  readsLocalDistribution: true,
  parameters: [
    {
      kind: "text",
      name: "model",
      label:
        "Model as <mount>/<model>, a mount of gemma4, gemma4-qat, gemma4-qat-mobile (empty: the default model of the first mount whose mirror is present)",
      default: "",
    },
    {
      kind: "text",
      name: "preset",
      label: "Preset (empty: the manifest's default preset of the model)",
      default: "",
    },
    {
      kind: "integer",
      name: "tokens",
      label: "Tokens per run (the first comes from the prefill)",
      min: SESSION_AB_LIMITS.minTokens,
      max: SESSION_AB_LIMITS.maxTokens,
      default: SESSION_AB_DEFAULTS.tokens,
    },
    {
      kind: "integer",
      name: "rounds",
      label: "Timed rounds (one untimed warm-up round comes first)",
      min: SESSION_AB_LIMITS.minRounds,
      max: SESSION_AB_LIMITS.maxRounds,
      default: SESSION_AB_DEFAULTS.rounds,
    },
    {
      kind: "number",
      name: "minGain",
      label: "Minimum gain (a fraction of the base's time: 0.01 = 1%)",
      min: SESSION_AB_LIMITS.minGain,
      max: SESSION_AB_LIMITS.maxGain,
      default: SESSION_AB_DEFAULTS.minGain,
    },
    {
      kind: "boolean",
      name: "openPerVisit",
      label:
        "Open and close each arm's pipeline per visit (off: both stay loaded; two pipelines of a large model do not fit together)",
      default: SESSION_AB_DEFAULTS.openPerVisit,
    },
    {
      kind: "boolean",
      name: "probeCounts",
      label: "Probe the counts first (one extra load and untimed run per arm)",
      default: SESSION_AB_DEFAULTS.probeCounts,
    },
    {
      kind: "boolean",
      name: "f16Kv",
      label: "f16 KV (full layers)",
      default: false,
    },
    {
      kind: "boolean",
      name: "subgroups",
      label:
        "Acquire the device with the subgroup features when it has them (off: a device without them)",
      default: true,
    },
    {
      kind: "text",
      name: "expertCacheSlots",
      label:
        'Expert cache slots, gemma4-qat/26b only: one count from 16 to 128, a comma list (one per routed layer) or "fitted" (empty: every expert resident)',
      default: "",
    },
    {
      kind: "text",
      name: "hostTier",
      label:
        'Host tier in GiB, or "all" (empty: every miss reads the distribution; needs the slots)',
      default: "",
    },
    {
      kind: "integer",
      name: "depth",
      label: "Self-speculation depth (0: one run per decode token; needs the slots)",
      min: 0,
      max: EXPERT_SPECULATION_DEPTH.max,
      default: 0,
    },
    {
      kind: "choice",
      name: "source",
      label: "How a browser reads the distribution",
      options: SOURCE_KINDS,
      default: "cached",
    },
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution, backend }) => {
    const { key, opened, manifest } = await openGemmaModel(
      parseModelText(textValue(parameters, "model")),
      choiceValue(parameters, "source", SOURCE_KINDS),
      openDistribution,
      signal,
    );
    const request = gemmaRequest(parameters, key, manifest, GELU_MUL_RULE_NAME);
    // With the subgroup features the device can run the 32-lane kernel forms, as the pages of the
    // Playground acquire it; without them the arms run on the device a plain `acquireGpu()` gives.
    // Which one fits can differ: the 26B model with 64 expert slots builds a second pipeline on a
    // 12 GB GPU only without them (docs/known-issues.md).
    const gpu = await acquireGpu({
      ...gpuOptions,
      ...(booleanValue(parameters, "subgroups") ? { subgroups: "available" as const } : {}),
    });
    try {
      const specOf = (
        name: string,
        fusionRules: SessionOptions["fusionRules"],
      ): ArmSpec<GemmaPipeline> => ({
        name,
        ...(fusionRules === undefined ? {} : { fusionRules }),
        request,
        prompt: (pipeline) =>
          chatPrompt(pipeline.tokenizer, [{ role: "user", content: request.prompt }]),
        build: (options) =>
          buildGemmaPipeline({
            ref: opened.ref,
            gpu,
            load: request,
            signal,
            ...(options.fusionRules === undefined
              ? {}
              : { sessionOptions: { fusionRules: options.fusionRules } }),
            ...(options.onRunDiagnostics === undefined
              ? {}
              : { onRunDiagnostics: options.onRunDiagnostics }),
          }),
        signal,
      });
      const specs = [specOf("base", undefined), specOf("rule", [GELU_MUL_RULE])];
      // The count probes come first, one arm at a time, all untimed; the arms are never built with
      // the diagnostics callback.
      const probes: SessionAbArmProbe[] = [];
      if (request.probeCounts) {
        for (const spec of specs) {
          const counts = await probeCounts({
            ...spec,
            onStep: (step) => progress(probeStepLine(step, probes.length, specs.length)),
          });
          probes.push({ name: spec.name, counts });
        }
      }
      const onStep = armStepper({
        positions: runPositions(request.arms.length, request.rounds, request.openPerVisit),
        opens: "pipeline",
        signal,
        progress,
      });
      const arms = specs.map((spec) => pipelineArm({ ...spec, onStep }));
      const record = await runSessionAb({
        arms,
        rounds: request.rounds,
        openPerVisit: request.openPerVisit,
        environment: runnerEnvironment(gpu.adapterInfo, backend),
      });
      const judgements = judgeSessionAb(record, { minGain: request.minGain });
      return sessionAbGemmaResult(
        sessionAbGemmaData({
          request,
          distribution: opened.facts,
          modelsVersion: MODELS_VERSION,
          ruleName: GELU_MUL_RULE_NAME,
          arms: arms.map((arm) => ({ name: arm.name, facts: arm.facts() })),
          ...(request.probeCounts ? { probes } : {}),
          record,
          judgements,
        }),
      );
    } finally {
      gpu.destroy();
    }
  },
};

export default entry;
