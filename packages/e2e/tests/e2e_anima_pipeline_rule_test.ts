/**
 * The candidate rule `candidate/vae_rms_silu_g8` through the whole anima pipeline (opt-in:
 * `KARUME_PIPELINE_RULE_AB=<dir>` — the shared runner `helpers/pipeline-rule-ab.ts`): two
 * `anima.Pipeline`s from the distribution mirror at its default model and preset
 * (`anima-turbo-v1.1`, `f16+dit8-a8-attn8-s16`), `base` and `rule` (the same plus
 * `fusionRules: [vaeRmsSiluRule(8)]`, which reaches the DiT and the VAE decoder — not the text
 * stages), generate the family's smallest e2e request (512² × the turbo model's 8 steps, guidance 1,
 * seed 42 — the size of e2e_anima_test.ts's smallest case) and every image (the pipeline's RGBA8
 * output, read as u32 words) is compared word for word with the base's first.
 *
 * It asserts that identity for every generation of both arms, and that the rule took its sites in
 * the rule arm's VAE decoder and none anywhere in the base arm (the hits are summed over the
 * generation's runs from `onRunDiagnostics` — `lastRunFusions`; one 512² image is one VAE tile);
 * never a verdict on times. The record keeps each generation's stage times (the pipeline's `stage`
 * events) beside the whole generation's.
 *
 * NOTE: the output compared is the pipeline's 8-bit image, so a difference below one level of the
 * quantisation would not show here; the rule's f32 identity on the VAE graph itself is
 * e2e_vae_rms_silu_rule_test.ts and e2e_vae_session_ab_test.ts.
 *
 * The rule MUST NOT be passed on Metal (its rounding barrier is measured not to hold there), so the
 * case SKIPs on that backend. MUST: never run it while other work uses the GPU; run it under
 * `flock /tmp/karume-gpu.lock`. Without the variable, without the mirror or without an adapter, the
 * case SKIPs with the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { vaeRmsSiluRule, vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { Pipeline, type PipelineOptions } from "../../models/anima.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import {
  inferredBackend,
  type PipelineRuleArm,
  pipelineRuleAbSkipReason,
  runPipelineRuleAb,
  stageClock,
} from "./helpers/pipeline-rule-ab.ts";

/** The local mirror (docs/assets-layout.md). */
const ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
const MIRROR_PRESENT = (await readTextIfPresent(new URL("karume.json", ASSETS_DIR))) !== undefined;

const RULE_NAME = vaeRmsSiluRuleName(8);

const skipReason = pipelineRuleAbSkipReason([
  [MIRROR_PRESENT, `${ASSETS_DIR.pathname} has no manifest (bake it with the exporter's dist.py)`],
  [GPU_AVAILABLE, "no GPU adapter"],
  [
    inferredBackend() !== "metal",
    `${RULE_NAME} must not run on Metal (its rounding barrier is measured not to hold there)`,
  ],
]);
if (skipReason !== undefined) {
  console.warn(`[karume] the anima pipeline rule A/B SKIPs: ${skipReason}`);
}

/** The request (the turbo model's defaults fill the steps and guidance). */
const REQUEST = {
  prompt: "a cat sitting on a windowsill, soft light, masterpiece, best quality",
  resolution: { width: 512, height: 512 },
  seed: 42,
} as const;

const ROUNDS = 3;

/** One arm: a pipeline built by its `open`, disposed by its `close`, with its hits and stages. */
const arm = (name: string, options: PipelineOptions): PipelineRuleArm => {
  let pipeline: Pipeline | undefined;
  /** The rule's hits per component over the generation in flight. */
  let hits: Record<string, number> = {};
  const clock = stageClock();
  return {
    name,
    open: async () => {
      pipeline = await Pipeline.fromPretrained(nodeDirectory(ASSETS_DIR), {
        ...options,
        onRunDiagnostics: (component, diagnostics) => {
          hits[component] = (hits[component] ?? 0) + (diagnostics.lastRunFusions?.[RULE_NAME] ?? 0);
        },
      });
    },
    generate: async () => {
      if (pipeline === undefined) throw new Error(`${name}: not open`);
      hits = {};
      const image = await pipeline.generate({ ...REQUEST, onEvent: clock.observe });
      return { image: new Uint32Array(image.data.buffer.slice(0)) };
    },
    report: () => Promise.resolve({ fusions: { ...hits } }),
    stages: clock.take,
    close: async () => {
      await pipeline?.dispose();
      pipeline = undefined;
    },
  };
};

describe.skipIf(skipReason !== undefined)(`${RULE_NAME} through anima.Pipeline (opt-in)`, () => {
  it("every image of both arms is word-identical to the base's first, and the rule took the VAE's sites", async () => {
    const { record } = await runPipelineRuleAb({
      family: "anima",
      rule: RULE_NAME,
      request: { ...REQUEST, model: "default (manifest)", quant: "default (manifest)" },
      arms: [arm("base", {}), arm("rule", { fusionRules: [vaeRmsSiluRule(8)] })],
      rounds: ROUNDS,
    });
    const [base, rule] = record.arms;
    assertEquals([base.identity.worst.failures, rule.identity.worst.failures], [0, 0]);
    assert(
      Object.values(base.fusions ?? {}).every((count) => count === 0),
      `the base arm took the rule: ${JSON.stringify(base.fusions)}`,
    );
    assert(
      (rule.fusions?.["vae_decoder"] ?? 0) > 0,
      `the rule arm's VAE took no site: ${JSON.stringify(rule.fusions)}`,
    );
  });
});
