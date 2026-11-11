/**
 * The fusion context the fusion-count checks plan under (`./helpers/fusion-counts.ts`) on synthetic
 * IR (no GPU, no real assets — the core lane): the knobs and weight facts come from the shared
 * functions Session construction uses (`sessionFusionKnobs`, `plannedResidentWeights`,
 * `sharedOrPooledWeights` in packages/runtime/src/runtime/session-fusion-context.ts), not from a
 * copy of the mapping.
 */

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { sessionFusionRules } from "../../runtime/src/runtime/fusion.ts";
import { countUses, ExecutionError } from "../../runtime/src/runtime/plan.ts";
import {
  plannedResidentWeights,
  sessionFusionContext,
  sessionFusionKnobs,
  sharedOrPooledWeights,
} from "../../runtime/src/runtime/session-fusion-context.ts";
import { DISABLED_FUSION_RULES } from "../../runtime/src/runtime/session-types.ts";
import { planWeightResidency } from "../../runtime/src/runtime/weight-residency.ts";
import { mergeGraph } from "../../runtime/tests/helpers/merged-graph.ts";
import type { DeclarationJson } from "../../runtime/tests/helpers/model-fixture.ts";
import { type FusionCountOptions, planDistributionFusions } from "./helpers/fusion-counts.ts";

const SHAPE = [4, 8];

/** `y = neg(x)` — no rule takes it, so only the context is under test. */
const GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["neg"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: SHAPE }],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: SHAPE } },
  nodes: [{ op: "neg", ins: ["x"], outs: ["y"], attrs: {} }],
};

describe("planDistributionFusions", () => {
  it("plans under the context the shared Session mapping gives for the same options", () => {
    const graph = mergeGraph(GRAPH);
    const residency = planWeightResidency(graph);
    for (const options of [
      {},
      { linearCompute: "a8" },
      {
        linearGemvReduce: "parallel",
        packedStaticQuantize: true,
        integerGemv: true,
        fuseRmsNormAdd: true,
        rmsNormReduce: "subgroup32",
        [DISABLED_FUSION_RULES]: ["silu"],
      },
      { [DISABLED_FUSION_RULES]: "all" },
    ] satisfies readonly FusionCountOptions[]) {
      const rules = sessionFusionRules(undefined);
      assertEquals(
        planDistributionFusions(graph, { x: SHAPE }, options).context,
        sessionFusionContext({
          rules,
          knobs: sessionFusionKnobs(options, rules),
          useCounts: countUses(graph),
          outputNames: new Set(graph.outputs),
          limits: {
            maxStorageBufferBindingSize: 128 * 1024 * 1024,
            maxComputeWorkgroupsPerDimension: 65535,
          },
          residentWeights: plannedResidentWeights(residency),
          sharedOrPooledWeights: sharedOrPooledWeights(residency, undefined),
        }),
        JSON.stringify(options),
      );
    }
  });

  it("refuses what Session construction refuses: a combination, and a disabled name outside the rule list", () => {
    const graph = mergeGraph(GRAPH);
    assertThrows(
      () => planDistributionFusions(graph, { x: SHAPE }, { integerGemv: true }),
      Error,
      "fusion-count options: integerGemv requires packedStaticQuantize: true",
    );
    assertThrows(
      () => planDistributionFusions(graph, { x: SHAPE }, { [DISABLED_FUSION_RULES]: ["sliu"] }),
      ExecutionError,
      "the disabled name(s) 'sliu' are not names of the rule list",
    );
  });
});
