import { assertEquals, assertRejects } from "@std/assert";
import { checkGemvSubgroup, GEMV_SUBGROUP_CASES } from "@karume/kernel-checks";
import { beforeAll, describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";
import { checkGemvSubgroupCensus } from "./helpers/gemv-subgroup-census.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

describe.skipIf(!GPU_AVAILABLE)("並列GEMV subgroup32の明示指定", () => {
  it("不正な値・必要なdevice機能の不足を重み転送前に拒否する", async () => {
    const gpu = await acquireGpu();
    const opened = await openModelBytes(rmsNormAddGraph(), []);
    try {
      for (const value of [false, "auto", "parallel-subgroup32"]) {
        const options: SessionOptions = {};
        Reflect.set(options, "linearGemvReduce", value);
        await assertRejects(
          () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
          Error,
          "linearGemvReduce",
        );
      }
    } finally {
      gpu.destroy();
    }
  });
  describe("対応実装ではCPU参照・parallel一致・行数・適用箇所を検査する", () => {
    let supported = false;
    beforeAll(async () => {
      const adapter = await navigator.gpu.requestAdapter();
      const features = new Set<string>(adapter?.features ?? []);
      const host: GPU & { wgslLanguageFeatures?: Iterable<string> } = navigator.gpu;
      supported =
        features.has("subgroups") &&
        features.has("subgroup-size-control") &&
        new Set(host.wgslLanguageFeatures).has("subgroup_id");
    });
    it("subgroups実走（Deno未対応なら明示SKIP、Chromeは別途同じ検査を実行）", async (context) => {
      // Adapter features are only known at run time, so the skip is decided here.
      context.skip(!supported, "subgroups / subgroup-size-control / subgroup_id unsupported");
      const gpu = await acquireGpu({ subgroups: true });
      try {
        const part = await checkGemvSubgroup(gpu);
        // i2 / i8: 3 shapes, i4: 6 shapes; each × 5 lane counts × M = 1 / 4 / 8.
        assertEquals(part.cases.length, GEMV_SUBGROUP_CASES["gemv-subgroup32"]);
        assertCheckPart(part);
        await checkGemvSubgroupCensus(gpu);
      } finally {
        gpu.destroy();
      }
    });
  });
});
