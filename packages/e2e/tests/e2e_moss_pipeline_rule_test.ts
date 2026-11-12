/**
 * The candidate rule `candidate/ew_chain` through the whole MOSS-Image pipeline (opt-in:
 * `KARUME_PIPELINE_RULE_AB=<dir>` — the shared runner `helpers/pipeline-rule-ab.ts`): two
 * `moss.Pipeline`s from the distribution mirror at its default preset (`f32+dit8-a8-attn8-online`),
 * `base` and `rule` (the same plus `fusionRules: [EW_CHAIN_RULE]`, which reaches `dit_text`, `dit`,
 * `dit_skip22` and the VAE decoder — not the text encoder), generate the family's smallest e2e
 * request (256² × 2 steps, the manifest's guidance 3 and empty negative prompt, seed 42 — the size
 * and steps of e2e_moss_pipeline_test.ts) and every image `[3, 256, 256]` f32 is compared word for
 * word with the base's first.
 *
 * It asserts that identity for every generation of both arms; never a verdict on times. The pipeline
 * reports no stage events and no Session diagnostics, so the record holds the whole generation's
 * time only and no rule hit count: the hits on the DiT are the session A/B's
 * (e2e_moss_session_ab_test.ts — 292 per `dit` forward).
 *
 * MUST: never run it while other work uses the GPU; run it under `flock /tmp/karume-gpu.lock`.
 * Without the variable, without the mirror or without an adapter, the case SKIPs with the reason
 * (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { nodeDirectory } from "@karume/hub/node";
import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/candidate-rules";
import { Pipeline, type PipelineOptions } from "../../models/moss.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";
import {
  type PipelineRuleArm,
  pipelineRuleAbSkipReason,
  runPipelineRuleAb,
} from "./helpers/pipeline-rule-ab.ts";

/** The local mirror (`dist.py --pipeline moss-image` builds it — docs/assets-layout.md). */
const DIST_ROOT = new URL("../../../models/karume-moss-image/", import.meta.url);
const ASSEMBLE = "cd tools/export-recipes && uv run python dist.py --pipeline moss-image";
const MIRROR_PRESENT = (await readTextIfPresent(new URL("karume.json", DIST_ROOT))) !== undefined;

const skipReason = pipelineRuleAbSkipReason([
  [MIRROR_PRESENT, `${DIST_ROOT.pathname} has no manifest (build it with: ${ASSEMBLE})`],
  [GPU_AVAILABLE, "no GPU adapter"],
]);
if (skipReason !== undefined) {
  console.warn(`[karume] the MOSS-Image pipeline rule A/B SKIPs: ${skipReason}`);
}

/** The request (the manifest's defaults fill guidance, shift and the negative prompt). */
const REQUEST = {
  prompt: "A red apple on a wooden table by a window, soft morning light.",
  width: 256,
  height: 256,
  steps: 2,
  seed: 42,
} as const;

const ROUNDS = 3;

/** One arm: a pipeline built by its `open`, disposed by its `close`. */
const arm = (name: string, options: PipelineOptions): PipelineRuleArm => {
  let pipeline: Pipeline | undefined;
  return {
    name,
    open: async () => {
      pipeline = await Pipeline.fromPretrained(nodeDirectory(DIST_ROOT.pathname), options);
    },
    generate: async () => {
      if (pipeline === undefined) throw new Error(`${name}: not open`);
      return { image: (await pipeline.generate(REQUEST)).data };
    },
    close: async () => {
      await pipeline?.dispose();
      pipeline = undefined;
    },
  };
};

describe.skipIf(skipReason !== undefined)(
  "candidate/ew_chain through moss.Pipeline (opt-in)",
  () => {
    it("every image of both arms is word-identical to the base's first", async () => {
      const { record } = await runPipelineRuleAb({
        family: "moss",
        rule: EW_CHAIN_RULE_NAME,
        request: { ...REQUEST, quant: "default (manifest)", rounds: ROUNDS },
        arms: [arm("base", {}), arm("rule", { fusionRules: [EW_CHAIN_RULE] })],
        rounds: ROUNDS,
      });
      const [base, rule] = record.arms;
      assertEquals([base.identity.worst.failures, rule.identity.worst.failures], [0, 0]);
    });
  },
);
