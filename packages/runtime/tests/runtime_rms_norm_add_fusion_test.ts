import { assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import { planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import { rmsNormSubgroupKey } from "../src/kernels/rms-norm-subgroup.ts";
import { rmsNormAddKey } from "../src/kernels/rms-norm.ts";
import { rmsNormAddGraph, type RmsNormAddGraphOptions } from "./helpers/rms-norm-add-graph.ts";
import { GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

const fuse = (options: RmsNormAddGraphOptions = {}, enabled?: boolean, subgroup = false) => {
  const graph = mergedGraph(
    memoryModel(rmsNormAddGraph(options), []).graphs[GRAPH_NAME],
    GRAPH_NAME,
  );
  return planFusions(planGraph(graph, {}).nodes, {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    ...(enabled === undefined ? {} : { fuseRmsNormAdd: enabled }),
    rmsNormReduce: subgroup ? "subgroup32" : "workgroup",
  });
};

describe("opt-in RMS → add fusion", () => {
  it("keeps the reference path when omitted or false", () => {
    for (const enabled of [undefined, false]) {
      const plan = fuse({}, enabled);
      assertEquals(plan.counts.rmsNormAdd, 0);
      assertEquals(
        plan.steps.map((step) => step.kind),
        ["node", "node", "node"],
      );
    }
  });
  it("accepts both input orders and the measured widths, and keeps the total reference count of external inputs", () => {
    for (const dim of [256, 1536, 2560, 2816, 3840]) {
      for (const normFirst of [false, true]) {
        const plan = fuse({ dim, normFirst }, true),
          step = plan.steps[0];
        assertEquals(plan.counts.rmsNormAdd, 1);
        if (step.kind !== "fused") throw Error("not fused");
        assertEquals(step.binds, ["x", "w", "r"]);
        assertEquals(step.ins, ["x", "w", "r"]);
        assertEquals(step.outputName, "y");
        assertEquals(
          step.dispatches[0].key,
          rmsNormAddKey(normFirst ? "norm-residual" : "residual-norm", dim),
        );
        assertEquals(step.dispatches[0].params[3], 0, "the rounding XOR uniform is 0");
      }
    }
  });
  it("subgroup32 keeps the fusion's eligibility conditions and picks separate keys for the two addition orders", () => {
    for (const normFirst of [false, true]) {
      const plan = fuse({ normFirst }, true, true),
        step = plan.steps[0];
      if (step.kind !== "fused") throw Error("not fused");
      assertEquals(
        step.dispatches[0].key,
        rmsNormSubgroupKey(normFirst ? "norm-residual" : "residual-norm"),
      );
      assertEquals(fuse({ normOutput: true, normFirst }, true, true).counts.rmsNormAdd, 0);
      assertEquals(fuse({ normFirst }, false, true).counts.rmsNormAdd, 0);
    }
  });
  it("does not fuse a public or shared norm output, non-adjacent nodes, a broadcast, a duplicate binding or an unvalidated width", () => {
    for (const options of [
      { normOutput: true },
      { extraNormConsumer: true },
      { interpose: true },
      { broadcast: true },
      { sharedResidual: true },
      { dim: 128 },
      { dim: 512 },
      { dim: 1537 },
    ])
      assertEquals(fuse(options, true).counts.rmsNormAdd, 0, JSON.stringify(options));
  });
});
