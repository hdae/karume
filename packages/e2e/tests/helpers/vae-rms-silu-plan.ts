/**
 * The rule `models/vae_rms_silu` (`@karume/models/wan` and `/anima`, the VAE's channel RMS norm and
 * the SiLU after it in one dispatch) planned on a distribution graph, for the family fusion-count
 * tests that pin its hits (Wan2.1, Wan2.2, anima): the plan with a rule list — by default the VAE
 * decoder's (`vaeDecoderFusionRules`, the list the Wan and anima pipelines give their VAE decoder
 * Sessions) — and the plan with the same options and the rule disabled by name (the test-only
 * `DISABLED_FUSION_RULES`, the plain side), both through {@link planDistributionFusions} (the
 * Session's path), and the rule's steps per tail as its dispatch key names it (`silu`: the 10-node
 * chain that ends with the SiLU, `none`: the 8-node norm alone). Pure; no side effects.
 */

import type { IrGraph } from "../../../runtime/src/format/ir.ts";
import { DISABLED_FUSION_RULES } from "../../../runtime/src/runtime/session-types.ts";
import type { SessionOptions } from "@karume/runtime";
import { VAE_RMS_SILU_RULE_NAME } from "@karume/models/wan";
import { vaeDecoderFusionRules } from "../../../models/src/vae/fusion-rules.ts";
import { type DistributionFusionPlan, planDistributionFusions } from "./fusion-counts.ts";

export type VaeRmsSiluPlans = {
  readonly plain: DistributionFusionPlan;
  readonly fused: DistributionFusionPlan;
  /** The rule's steps per tail of its key. */
  readonly tails: Readonly<Record<string, number>>;
};

/**
 * The two plans of `graph` at `inputShapes`: with `fusionRules` (default: the VAE decoder's list
 * without caller rules) and with the same and `models/vae_rms_silu` disabled.
 */
export const vaeRmsSiluPlans = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  fusionRules: NonNullable<SessionOptions["fusionRules"]> = vaeDecoderFusionRules(undefined),
): VaeRmsSiluPlans => {
  const fused = planDistributionFusions(graph, inputShapes, { fusionRules });
  const plain = planDistributionFusions(graph, inputShapes, {
    fusionRules,
    [DISABLED_FUSION_RULES]: [VAE_RMS_SILU_RULE_NAME],
  });
  const tails: Record<string, number> = {};
  for (const step of fused.plan.steps) {
    if (step.kind !== "fused" || step.rule !== VAE_RMS_SILU_RULE_NAME) continue;
    // `ext:models:vae_rms_silu:v1:g8:<tail>:f32:wg256:xor`
    const tail = step.dispatches[0].key.split(":")[5];
    tails[tail] = (tails[tail] ?? 0) + 1;
  }
  return { plain, fused, tails };
};
