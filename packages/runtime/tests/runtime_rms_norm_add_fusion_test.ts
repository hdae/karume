import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { parseIrGraph } from "../src/format/ir.ts";
import { planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import { rmsNormSubgroupKey } from "../src/kernels/rms-norm-subgroup.ts";
import { rmsNormAddKey } from "../src/kernels/rms-norm.ts";
import { rmsNormAddGraph, type RmsNormAddGraphOptions } from "./helpers/rms-norm-add-graph.ts";

const fuse = (options: RmsNormAddGraphOptions = {}, enabled?: boolean, subgroup = false) => {
  const graph = parseIrGraph(JSON.stringify(rmsNormAddGraph(options)));
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

describe("RMS→addの任意融合", () => {
  it("省略・falseでは参照経路を維持する", () => {
    for (const enabled of [undefined, false]) {
      const plan = fuse({}, enabled);
      assertEquals(plan.counts.rmsNormAdd, 0);
      assertEquals(plan.steps.map((step) => step.kind), ["node", "node", "node"]);
    }
  });
  it("両入力順と実測した幅を受理し、外部入力の延べ参照を保持する", () => {
    for (const dim of [256, 1536, 2560]) {
      for (const normFirst of [false, true]) {
        const plan = fuse({ dim, normFirst }, true), step = plan.steps[0];
        assertEquals(plan.counts.rmsNormAdd, 1);
        if (step.kind !== "fused") throw Error("融合されていない");
        assertEquals(step.binds, ["x", "w", "r"]);
        assertEquals(step.ins, ["x", "w", "r"]);
        assertEquals(step.outputName, "y");
        assertEquals(
          step.dispatches[0].key,
          rmsNormAddKey(normFirst ? "norm-residual" : "residual-norm"),
        );
        assertEquals(step.dispatches[0].params[3], 0, "丸め用XORのuniformは0");
      }
    }
  });
  it("subgroup32は融合の適格条件を保ち、両加算順を別キーへ選ぶ", () => {
    for (const normFirst of [false, true]) {
      const plan = fuse({ normFirst }, true, true), step = plan.steps[0];
      if (step.kind !== "fused") throw Error("融合されていない");
      assertEquals(
        step.dispatches[0].key,
        rmsNormSubgroupKey(normFirst ? "norm-residual" : "residual-norm"),
      );
      assertEquals(fuse({ normOutput: true, normFirst }, true, true).counts.rmsNormAdd, 0);
      assertEquals(fuse({ normFirst }, false, true).counts.rmsNormAdd, 0);
    }
  });
  it("公開・共有・非隣接・broadcast・重複binding・未検収幅は融合しない", () => {
    for (
      const options of [
        { normOutput: true },
        { extraNormConsumer: true },
        { interpose: true },
        { broadcast: true },
        { sharedResidual: true },
        { dim: 128 },
        { dim: 512 },
        { dim: 1537 },
      ]
    ) assertEquals(fuse(options, true).counts.rmsNormAdd, 0, JSON.stringify(options));
  });
});
