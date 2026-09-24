import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { mergedGraph } from "../src/format/container/bind.ts";
import { type FusionWeightLayout, planFusions } from "../src/runtime/fusion.ts";
import { countUses, planGraph } from "../src/runtime/plan.ts";
import { linearGemvStaticQuantizeKey } from "../src/kernels/linear-gemv.ts";
import {
  linearStaticQuantizeModel,
  type LinearStaticQuantizeOptions,
} from "./helpers/linear-static-quantize-graph.ts";
import { GRAPH_NAME } from "./helpers/model-fixture.ts";

const fuse = (
  o: LinearStaticQuantizeOptions = {},
  context: Partial<Parameters<typeof planFusions>[1]> = {},
) => {
  // 合流結果（格納つき IrGraph）は供給元に依らない — メモリ内容器 1 本で足りる。
  const graph = mergedGraph(linearStaticQuantizeModel(o).graphs[GRAPH_NAME], GRAPH_NAME);
  const weight: FusionWeightLayout = o.storage === "i4"
    ? { storage: "i4", groupSize: o.group ?? 512 }
    : { storage: o.storage ?? "i8" };
  return planFusions(planGraph(graph, {}).nodes, {
    useCounts: countUses(graph),
    outputNames: new Set(graph.outputs),
    limits: {
      maxStorageBufferBindingSize: 128 * 1024 * 1024,
      maxComputeWorkgroupsPerDimension: 65535,
    },
    fuseLinearStaticQuantize: true,
    linearGemvReduce: "parallel",
    linearCompute: "f32",
    weightLayouts: new Map([["w", weight]]),
    ...context,
  });
};
describe("linear→static_quantizeの任意融合", () => {
  it("明示指定と実測形だけを受理し、出力・外部入力・scaleの借用元を保持する", () => {
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const m of [1, 4, 8]) {
        const n = storage === "i2" ? 12288 : 256,
          plan = fuse({ storage, m, n }),
          step = plan.steps[0];
        assertEquals(plan.counts.linearStaticQuantize, 1);
        if (step.kind !== "fused") throw Error("融合されていない");
        assertEquals(step.ins, ["x", "w", "b"]);
        assertEquals(step.binds, ["x", "w", "b"]);
        assertEquals(step.outputName, "y");
        assertEquals(step.temps, []);
        assertEquals(
          step.dispatches[0].key,
          linearGemvStaticQuantizeKey(
            storage,
            storage === "i4" ? 512 : undefined,
            storage === "i2" ? 2 : 32,
          ),
        );
        assertEquals(step.dispatches[0].operands?.at(-1), { kind: "weightScale", index: 1 });
        assertEquals(step.dispatches[0].params[3], 0);
      }
    }
  });
  it("無指定・別計算方式・展開済み重み・未検収形は既存経路へ残す", () => {
    for (
      const context of [
        { fuseLinearStaticQuantize: undefined },
        { fuseLinearStaticQuantize: false },
        { linearGemvReduce: "sequential" },
        { linearGemvReduce: "parallel-subgroup32" },
        { linearCompute: "a8" },
        { linearCompute: "f16" },
        { weightLayouts: new Map() },
        { weightLayouts: new Map([["w", { storage: "f16" }]]) },
      ] satisfies Partial<Parameters<typeof planFusions>[1]>[]
    ) {
      assertEquals(fuse({}, context).counts.linearStaticQuantize, 0);
    }
    for (
      const o of [
        { m: 9 },
        { n: 260 },
        { k: 1504 },
        { publicLinear: true },
        { sharedLinear: true },
        { interpose: true },
      ]
    ) assertEquals(fuse(o).counts.linearStaticQuantize, 0);
  });
  it("scale 0は既存の恒等演算を維持し、tile数の上限を黙って縮めない", () => {
    const step = fuse({ scale: 0 }).steps[0];
    if (step.kind !== "fused") throw Error("融合されていない");
    assertEquals(step.dispatches[0].params[262], 1);
    assertThrows(() =>
      fuse({}, {
        limits: {
          maxStorageBufferBindingSize: 128 * 1024 * 1024,
          maxComputeWorkgroupsPerDimension: 2,
        },
      })
    );
  });
});
