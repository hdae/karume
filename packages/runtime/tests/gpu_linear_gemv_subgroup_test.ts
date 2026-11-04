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

describe.skipIf(!GPU_AVAILABLE)("explicitly selecting the parallel GEMV subgroup32", () => {
  it("rejects invalid values and missing device features before the weights are uploaded", async () => {
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
  describe("on a supporting implementation, checks the CPU reference, parity with parallel, the row counts and where it applies", () => {
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
    it("runs with subgroups (explicit SKIP where Deno lacks support; Chrome runs the same checks separately)", async (context) => {
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
