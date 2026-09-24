import { assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";
import { checkRmsSubgroup } from "./helpers/rms-subgroup-check.ts";

describe({
  name: "RMS subgroup32の明示指定",
  ignore: !GPU_AVAILABLE,
  fn: () => {
    it("不正な値・必要なdevice機能の不足を重み転送前に拒否する", async () => {
      const gpu = await acquireGpu();
      const opened = await openModelBytes(rmsNormAddGraph(), []);
      try {
        for (const value of [null, false, "auto", "subgroup32"]) {
          const options: SessionOptions = {};
          Reflect.set(options, "rmsNormReduce", value);
          await assertRejects(
            () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
            Error,
            "rmsNormReduce",
          );
        }
        // 診断で利用者の変換を呼ばないこと: 変換が走れば `ExecutionError` ではなく "boom" が抜ける。
        for (
          const value of [
            {
              toString(): string {
                throw new Error("boom");
              },
            },
            {
              [Symbol.toPrimitive](): string {
                throw new Error("boom");
              },
            },
          ]
        ) {
          const options: SessionOptions = {};
          Reflect.set(options, "rmsNormReduce", value);
          const error = await assertRejects(
            () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
            ExecutionError,
            "rmsNormReduce",
          );
          // 値そのものではなく型名だけを出す。
          assertEquals(error.message.includes("object"), true, error.message);
        }
      } finally {
        gpu.destroy();
      }
    });
    it("対応実装ではCPU参照・融合・行数・grid-strideを検査する", async (t) => {
      const adapter = await navigator.gpu.requestAdapter();
      const features = new Set<string>(adapter?.features ?? []);
      const host: GPU & { wgslLanguageFeatures?: Iterable<string> } = navigator.gpu;
      const supported = features.has("subgroups") && features.has("subgroup-size-control") &&
        new Set(host.wgslLanguageFeatures).has("subgroup_id");
      await t.step({
        name: "subgroups実走（Deno未対応なら明示SKIP、Chromeは別途同じ検査を実行）",
        ignore: !supported,
        fn: async () => {
          const gpu = await acquireGpu({ subgroups: true });
          try {
            await checkRmsSubgroup(gpu);
          } finally {
            gpu.destroy();
          }
        },
      });
    });
  },
});
