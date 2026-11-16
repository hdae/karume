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
import { EXPERT_SPECULATION_DEPTH } from "@karume/gemma4-app-support/settings";
import { chatPrompt } from "@karume/models/gemma";
import type { SessionOptions } from "@karume/runtime";
import { judgeSessionAb, runSessionAb } from "@karume/runtime/tune";

import { booleanValue, choiceValue, type Entry, textValue } from "../../entry.ts";
import { buildGemmaPipeline, type GemmaPipeline } from "../../gemma4/pipeline.ts";
import {
  acquireWithSubgroups,
  SOURCE_KINDS,
  sourceParameter,
  SUBGROUPS_PARAMETER,
} from "../../local-parameters.ts";
import { type SessionAbArmProbe, sessionAbGemmaData } from "../../session-ab/data.ts";
import { type ArmSpec, pipelineArm, probeCounts } from "../../session-ab/gemma-arm.ts";
import { gemmaRequest, openGemmaModel, parseModelText } from "../../session-ab/gemma-target.ts";
import { armStepper, probeStepLine } from "../../session-ab/progress.ts";
import {
  MIN_GAIN_PARAMETER,
  ROUNDS_PARAMETER,
  runnerEnvironment,
  runPositions,
} from "../../session-ab/shared.ts";
import { sessionAbGemmaResult } from "../../session-ab/summary.ts";

const entry: Entry = {
  id: "gemma4-session-ab",
  title: "Gemma 4 session A/B",
  description:
    "Wall-clock A/B (runSessionAb, judgeSessionAb) of a Gemma 4 greedy decode with and without the candidate rule candidate/gelu_mul, the counts from a probe of each arm first. The verdict is token identity; the judgement is in the summary.",
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
    // At least 2: the first token comes from the prefill, so a run of one token has no decode run
    // for the probe to count. At most what SESSION_AB_CAPACITY holds next to the prompt.
    {
      kind: "integer",
      name: "tokens",
      label: "Tokens per run (the first comes from the prefill)",
      min: 2,
      max: 512,
      default: 96,
    },
    ROUNDS_PARAMETER,
    MIN_GAIN_PARAMETER,
    {
      kind: "boolean",
      name: "openPerVisit",
      label:
        "Open and close each arm's pipeline per visit (off: both stay loaded; two pipelines of a large model do not fit together)",
      default: true,
    },
    {
      kind: "boolean",
      name: "probeCounts",
      label: "Probe the counts first (one extra load and untimed run per arm)",
      default: true,
    },
    {
      kind: "boolean",
      name: "f16Kv",
      label: "f16 KV (full layers)",
      default: false,
    },
    SUBGROUPS_PARAMETER,
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
    sourceParameter("How a browser reads the distribution"),
  ],
  run: async ({ parameters, gpuOptions, signal, progress, openDistribution, backend }) => {
    const { key, opened, manifest } = await openGemmaModel(
      parseModelText(textValue(parameters, "model")),
      choiceValue(parameters, "source", SOURCE_KINDS),
      openDistribution,
      signal,
    );
    const request = gemmaRequest(parameters, key, manifest, GELU_MUL_RULE_NAME);
    const gpu = await acquireWithSubgroups(gpuOptions, booleanValue(parameters, "subgroups"));
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
            ...(fusionRules === undefined ? {} : { sessionOptions: { fusionRules } }),
            ...(options.onRunDiagnostics === undefined
              ? {}
              : { onRunDiagnostics: options.onRunDiagnostics }),
          }),
        signal,
      });
      // The request's arms name the rule: an arm that names one passes candidate/gelu_mul.
      const specs = request.arms.map(({ name, fusionRules }) =>
        specOf(name, fusionRules.length === 0 ? undefined : [GELU_MUL_RULE]),
      );
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
