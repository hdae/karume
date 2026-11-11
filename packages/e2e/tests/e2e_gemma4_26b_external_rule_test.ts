/**
 * A fusion rule written outside the runtime on a real graph (no GPU — `planFusions` on part 0
 * only): the test-support rule `test/gelu_mul` (`packages/models/tests/helpers/gelu-mul-rule.ts`,
 * written against `@karume/runtime/kernel` alone) planned on Gemma 4 26B A4B's decode graph through
 * the path a Session takes (`planDistributionFusions` with `fusionRules`, so the Session's own list
 * composition and checks). It takes each layer's two `gelu_tanh → mul` chains — the dense MLP's,
 * with the up projection's `linear` between them (gate linear → gelu_tanh → up linear → mul), and
 * the routed experts', adjacent (gelu_tanh of one half of the gate/up `expert_linear` → mul with the
 * other half): 30 + 30 = 60 per plan, the count the 2026-11-10 prototype reached through a load hook
 * (perf-ledger K-105). Every other counter stays the plan's without the rule, and the rule disabled
 * by name gives that plan back. Token identity on the real weights is not checked here (this
 * machine's GPU cannot hold 26B, and the 26B decode tests run through the gemma pipeline, which
 * passes no rules); the rule's values are checked against the unfused plan on the GPU in
 * `e2e_external_fusion_rule_test.ts`.
 *
 * MUST: the asset is the distribution mirror `models/karume-gemma4-qat/` with its `26b` model
 * (untracked local asset); without it the case SKIPs explicitly (ADR 0005).
 */

import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { IrGraph } from "../../runtime/src/format/ir.ts";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import type { ContainerManifest } from "../../runtime/tests/helpers/container-graph.ts";
import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "../../models/tests/helpers/gelu-mul-rule.ts";
import {
  GEMMA4_26B_IN_MIRROR,
  GEMMA4_QAT_MIRROR,
} from "../../models/tests/helpers/gemma-mirror.ts";
import {
  type DistributionFusionPlan,
  decodeFusionPlan,
  readDistributionGraph,
} from "./helpers/fusion-counts.ts";

if (!GEMMA4_26B_IN_MIRROR) {
  console.warn(
    `[karume] ${GEMMA4_QAT_MIRROR.pathname} has no 26b model, so the external gelu -> mul rule on` +
      " Gemma 4 26B's graph SKIPs (get the q4_0 QAT distribution mirror with its 26b model)",
  );
}

/** The q4_0 QAT distribution is self-contained: a cross-repo reference fails loudly. */
const NO_MIRRORS: ReadonlyMap<string, URL> = new Map();

const read26bGraph = async (): Promise<IrGraph> => {
  const manifest: ContainerManifest = JSON.parse(
    await readTextFile(new URL("karume.json", GEMMA4_QAT_MIRROR)),
  );
  return await readDistributionGraph(GEMMA4_QAT_MIRROR, manifest, "26b", "model", "i4", NO_MIRRORS);
};

/** Decode (M = 1) and a prefill chunk (M = 32). */
const ROWS = [1, 32] as const;

/** The 30 layers' chains of each window form. */
const LAYERS = 30;

/**
 * The window each of the rule's steps folded, read from the graph's node order: the `gelu_tanh`
 * right before the `mul` that writes the step's output (`adjacent`), or two nodes before it with a
 * `linear` between (`over a linear`); with the op that wrote the step's gate.
 */
const windows = (graph: IrGraph, fused: DistributionFusionPlan): readonly string[] => {
  const producerAt = new Map(
    graph.nodes.flatMap((node, at) => node.outs.map((name): [string, number] => [name, at])),
  );
  return fused.plan.steps.flatMap((step) => {
    if (step.kind !== "fused" || step.rule !== GELU_MUL_RULE_NAME) return [];
    const mul = producerAt.get(step.outputName) ?? -1;
    const before = graph.nodes[mul - 1]?.op;
    const form =
      before === "gelu_tanh"
        ? "adjacent"
        : before === "linear" && graph.nodes[mul - 2]?.op === "gelu_tanh"
          ? "over a linear"
          : "unknown";
    const gate = graph.nodes[producerAt.get(step.binds[0]) ?? -1]?.op ?? "an input";
    return [`${form}, gate from ${gate}`];
  });
};

const tally = (labels: readonly string[]): Readonly<Record<string, number>> => {
  const counts: Record<string, number> = {};
  for (const label of labels) counts[label] = (counts[label] ?? 0) + 1;
  return counts;
};

describe.skipIf(!GEMMA4_26B_IN_MIRROR)(
  "the external gelu -> mul rule on Gemma 4 26B A4B (no GPU)",
  () => {
    it("takes the 30 dense and the 30 routed-expert chains, 60 per plan, and leaves every other counter as the plan without it has it", async () => {
      const graph = await read26bGraph();
      for (const rows of ROWS) {
        const fused = decodeFusionPlan(graph, rows, { fusionRules: [GELU_MUL_RULE] });
        const plain = decodeFusionPlan(graph, rows);
        assertEquals(fused.plan.counts[GELU_MUL_RULE_NAME], 2 * LAYERS, `M = ${rows}: hits`);
        assertEquals(
          tally(windows(graph, fused)),
          { "over a linear, gate from linear": LAYERS, "adjacent, gate from slice": LAYERS },
          `M = ${rows}: windows`,
        );
        assertEquals(
          fused.plan.counts,
          { ...plain.plan.counts, [GELU_MUL_RULE_NAME]: 2 * LAYERS },
          `M = ${rows}: the other counters`,
        );
        const disabled = decodeFusionPlan(graph, rows, {
          fusionRules: [GELU_MUL_RULE],
          [DISABLED_FUSION_RULES]: [GELU_MUL_RULE_NAME],
        });
        assertEquals(
          disabled.plan.counts,
          { ...plain.plan.counts, [GELU_MUL_RULE_NAME]: 0 },
          `M = ${rows}: disabled by name`,
        );
        assertEquals(
          disabled.plan.steps.length,
          plain.plan.steps.length,
          `M = ${rows}: disabled steps`,
        );
      }
    });
  },
);
