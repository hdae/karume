import { assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";
import { checkGemvSubgroup } from "./helpers/gemv-subgroup-check.ts";
import { checkGemvSubgroupCensus } from "./helpers/gemv-subgroup-census.ts";

describe({
  name: "並列GEMV subgroup32の明示指定",
  ignore: !GPU_AVAILABLE,
  fn: () => {
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
    it("対応実装ではCPU参照・parallel一致・行数・適用箇所を検査する", async (t) => {
      const adapter = await navigator.gpu.requestAdapter();
      const features = new Set<string>(adapter?.features ?? []);
      const host: GPU & { wgslLanguageFeatures?: Iterable<string> } = navigator.gpu;
      const supported = features.has("timestamp-query") && features.has("subgroups") &&
        features.has("subgroup-size-control") &&
        new Set(host.wgslLanguageFeatures).has("subgroup_id");
      await t.step({
        name: "subgroups実走（Deno未対応なら明示SKIP、Chromeは別途同じ検査を実行）",
        ignore: !supported,
        fn: async () => {
          const gpu = await acquireGpu({ subgroups: true, gpuTiming: true });
          try {
            await checkGemvSubgroup(gpu);
            await checkGemvSubgroupCensus(gpu);
          } finally {
            gpu.destroy();
          }
        },
      });
    });
  },
});
