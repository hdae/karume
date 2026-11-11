/**
 * The candidate rule `candidate/vae_rms_silu_g8` / `_g4` (`@karume/candidate-rules`, the VAE's
 * channel RMS norm and the SiLU after it in one dispatch) planned on a distribution graph, for the
 * family fusion-count tests that pin its hits (Wan2.1, Wan2.2, anima): the plan without any rule
 * and the plan with the rule, both through {@link planDistributionFusions} (the Session's path), and
 * the rule's steps per tail as its dispatch key names it (`silu`: the 10-node chain that ends with
 * the SiLU, `none`: the 8-node norm alone). Pure; no side effects.
 */

import type { IrGraph } from "../../../runtime/src/format/ir.ts";
import { type VaeRmsSiluGroups, vaeRmsSiluRule, vaeRmsSiluRuleName } from "@karume/candidate-rules";
import { type DistributionFusionPlan, planDistributionFusions } from "./fusion-counts.ts";

export type VaeRmsSiluPlans = {
  readonly plain: DistributionFusionPlan;
  readonly fused: DistributionFusionPlan;
  /** The rule's steps per tail of its key. */
  readonly tails: Readonly<Record<string, number>>;
};

/** The two plans of `graph` at `inputShapes`, the second with the rule of `groups` given. */
export const vaeRmsSiluPlans = (
  graph: IrGraph,
  inputShapes: Readonly<Record<string, readonly number[]>>,
  groups: VaeRmsSiluGroups,
): VaeRmsSiluPlans => {
  const name = vaeRmsSiluRuleName(groups);
  const plain = planDistributionFusions(graph, inputShapes);
  const fused = planDistributionFusions(graph, inputShapes, {
    fusionRules: [vaeRmsSiluRule(groups)],
  });
  const tails: Record<string, number> = {};
  for (const step of fused.plan.steps) {
    if (step.kind !== "fused" || step.rule !== name) continue;
    // `ext:candidate:vae_rms_silu:v1:g<G>:<tail>:f32:wg256:xor`
    const tail = step.dispatches[0].key.split(":")[5];
    tails[tail] = (tails[tail] ?? 0) + 1;
  }
  return { plain, fused, tails };
};
