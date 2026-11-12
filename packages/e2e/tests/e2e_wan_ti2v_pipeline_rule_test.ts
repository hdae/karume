/**
 * The candidate rule `candidate/vae_rms_silu_g8` through the Wan2.2 TI2V pipeline's VAE stage (opt-in:
 * `KARUME_PIPELINE_RULE_AB=<dir>` — the shared runner `helpers/pipeline-rule-ab.ts`).
 *
 * The request is the family's smallest e2e request (the seed path of e2e_wan_ti2v_pipeline_test.ts:
 * `boxing-cats` on the precomputed text path, seed 42, 1280×704, 17 frames, 2 steps, guidance 5,
 * shift 5) at the mirror's default preset (`i8-a8-attn8-online-conv8`). One whole generation takes
 * about 157 s on the RTX 3080 Ti (the VAE stage about 143 s of it — the conv8 seed case's record of
 * 2026-11-11), so the A/B's seven generations per arm would take about 18 minutes per arm. The A/B
 * therefore runs **the VAE stage alone**, through the pipeline's own decode entry
 * (`decodeWanVaeStage` of `models/src/wan/tile-decode.ts`, the body `generate` calls): its input is
 * the request's own final latents `[48, 5, 44, 80]`, taken once from a real 2-step generation that
 * stops at the end of its DiT stage, and each arm's state is built by `loadWanFromPretrained` (the
 * body of `Ti2vPipeline.fromPretrained`), `base` without and `rule` with
 * `fusionRules: [vaeRmsSiluRule(8)]`, so the option travels the pipeline's own path to the decoder
 * Sessions. Every decoded video `[3, 17, 704, 1280]` f32 is compared word for word with the base's
 * first.
 *
 * It asserts that identity for every decode of both arms, and that the rule took its sites in the
 * rule arm's decoder chunk graphs and none in the base arm (summed over the decode's tiles from
 * `onRunDiagnostics` — `lastRunFusions`); never a verdict on times. The record keeps each decode's
 * stage time (the `vae_decoder` stage events) beside the timed span.
 *
 * The rule MUST NOT be passed on Metal (its rounding barrier is measured not to hold there), so the
 * case SKIPs on that backend. MUST: never run it while other work uses the GPU; run it under
 * `flock /tmp/karume-gpu.lock`. Without the variable, without the mirror or without an adapter, the
 * case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertStrictEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { vaeRmsSiluRule, vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { type PipelineOptions, Ti2vPipeline } from "../../models/wan.ts";
import {
  loadWanFromPretrained,
  planWanGeneration,
  WAN22_TI2V_FAMILY,
  type WanState,
} from "../../models/src/wan/family.ts";
import { decodeWanVaeStage } from "../../models/src/wan/tile-decode.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import {
  readWanTi2vDistributionManifestText,
  WAN_TI2V_ASSEMBLE_COMMAND,
  WAN_TI2V_DIST_ROOT,
} from "../../models/tests/helpers/wan-ti2v-pipeline.ts";
import {
  inferredBackend,
  type PipelineRuleArm,
  pipelineRuleAbSkipReason,
  runPipelineRuleAb,
  stageClock,
} from "./helpers/pipeline-rule-ab.ts";

const MIRROR_PRESENT = (await readWanTi2vDistributionManifestText()) !== undefined;

const RULE_NAME = vaeRmsSiluRuleName(8);

const skipReason = pipelineRuleAbSkipReason([
  [
    MIRROR_PRESENT,
    `${WAN_TI2V_DIST_ROOT.pathname} has no manifest (build it with: ${WAN_TI2V_ASSEMBLE_COMMAND})`,
  ],
  [GPU_AVAILABLE, "no GPU adapter"],
  [
    inferredBackend() !== "metal",
    `${RULE_NAME} must not run on Metal (its rounding barrier is measured not to hold there)`,
  ],
]);
if (skipReason !== undefined) {
  console.warn(`[karume] the Wan2.2 pipeline rule A/B SKIPs: ${skipReason}`);
}

/** The seed path's request of e2e_wan_ti2v_pipeline_test.ts (the prompt is the asset row's name). */
const REQUEST = {
  promptName: "boxing-cats",
  seed: 42,
  width: 1280,
  height: 704,
  frames: 17,
  steps: 2,
  guidance: 5,
  shift: 5,
} as const;

const ROUNDS = 3;

const OWNER = WAN22_TI2V_FAMILY.owner;

/** The construction options of every pipeline here (the precomputed text path: no umT5). */
const BASE_OPTIONS = { textEncoder: "precomputed" } as const satisfies PipelineOptions;

/** The request with the asset row's text as the prompt (the row comes from `prompts`). */
const requestOf = (prompts: readonly { readonly name: string; readonly prompt: string }[]) => {
  const { promptName, ...knobs } = REQUEST;
  const entry = prompts.find(({ name }) => name === promptName);
  if (entry === undefined) throw new Error(`the embedding asset has no '${promptName}'`);
  return { ...knobs, prompt: entry.prompt };
};

/** Where the latent capture stops the generation (after the DiT stage, before the VAE opens). */
const STOP = new Error("stop after the DiT stage");

/** The request's final latents: a real generation, stopped once its DiT stage has ended. */
const finalLatents = async (): Promise<Float32Array<ArrayBuffer>> => {
  await using pipeline = await Ti2vPipeline.fromPretrained(
    nodeDirectory(WAN_TI2V_DIST_ROOT),
    BASE_OPTIONS,
  );
  const controller = new AbortController();
  let latents: Float32Array<ArrayBuffer> | undefined;
  const error = await pipeline
    .generate({
      ...requestOf(pipeline.prompts),
      signal: controller.signal,
      onEvent: (event) => {
        if (event.kind === "denoise-step" && event.step === event.steps) {
          latents = event.copyLatents().data;
        }
        if (event.kind === "stage" && event.component === "transformer" && event.at === "end") {
          controller.abort(STOP);
        }
      },
    })
    .then(
      () => undefined,
      (reason: unknown) => reason,
    );
  assertStrictEquals(error, STOP, "the generation did not stop at the end of its DiT stage");
  assert(latents !== undefined, "no final latents were captured");
  return latents;
};

/** One arm: the pipeline's state built by its `open`, its GPU destroyed by its `close`. */
const arm = (
  name: string,
  options: PipelineOptions,
  latents: Float32Array<ArrayBuffer>,
): PipelineRuleArm => {
  let state: WanState | undefined;
  let plan: ReturnType<typeof planWanGeneration> | undefined;
  /** The rule's hits per component over the decode in flight. */
  let hits: Record<string, number> = {};
  const clock = stageClock();
  return {
    name,
    open: async () => {
      state = await loadWanFromPretrained(WAN22_TI2V_FAMILY, nodeDirectory(WAN_TI2V_DIST_ROOT), {
        ...BASE_OPTIONS,
        ...options,
        onRunDiagnostics: (component, diagnostics) => {
          hits[component] = (hits[component] ?? 0) + (diagnostics.lastRunFusions?.[RULE_NAME] ?? 0);
        },
      });
      plan = planWanGeneration(
        requestOf(state.textEmbeds.entries),
        state.textEmbeds,
        state.layout,
        state.config,
        WAN22_TI2V_FAMILY.generation,
        OWNER,
        {
          ...(state.vaeTile32 === undefined ? {} : { layout: state.vaeTile32.layout }),
          quant: state.quantName,
        },
      );
    },
    generate: async () => {
      if (state === undefined || plan === undefined) throw new Error(`${name}: not open`);
      hits = {};
      const video = await decodeWanVaeStage(
        state,
        plan,
        latents,
        WAN22_TI2V_FAMILY.generation,
        (event) => {
          clock.observe(event);
          return Promise.resolve();
        },
        undefined,
        OWNER,
      );
      return { video };
    },
    report: () => Promise.resolve({ fusions: { ...hits } }),
    stages: clock.take,
    close: () => {
      if (state?.ownsGpu === true) state.gpu.destroy();
      state = undefined;
      return Promise.resolve();
    },
  };
};

describe.skipIf(skipReason !== undefined)(
  `${RULE_NAME} through wan.Ti2vPipeline's VAE stage (opt-in)`,
  () => {
    it("every decoded video of both arms is word-identical to the base's first, and the rule took the decoder's sites", async () => {
      const latents = await finalLatents();
      const { record } = await runPipelineRuleAb({
        family: "wan-ti2v",
        rule: RULE_NAME,
        request: {
          ...REQUEST,
          textEncoder: "precomputed",
          quant: "default (manifest)",
          timed: "the VAE stage alone (decodeWanVaeStage) on the request's own final latents",
        },
        arms: [
          arm("base", {}, latents),
          arm("rule", { fusionRules: [vaeRmsSiluRule(8)] }, latents),
        ],
        rounds: ROUNDS,
      });
      const [base, rule] = record.arms;
      assertEquals([base.identity.worst.failures, rule.identity.worst.failures], [0, 0]);
      assert(
        Object.values(base.fusions ?? {}).every((count) => count === 0),
        `the base arm took the rule: ${JSON.stringify(base.fusions)}`,
      );
      assert(
        (rule.fusions?.["vae_decoder_next"] ?? 0) > 0,
        `the rule arm's decoder took no site: ${JSON.stringify(rule.fusions)}`,
      );
    });
  },
);
