import { assertEquals, assertRejects } from "@std/assert";
import { checkRmsSubgroup, RMS_SUBGROUP_CASES } from "@karume/kernel-checks";
import { beforeAll, describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer, type SessionOptions } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { rmsNormAddGraph } from "./helpers/rms-norm-add-graph.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";

describe.skipIf(!GPU_AVAILABLE)("explicit RMS subgroup32", () => {
  it("rejects an invalid value or a missing device feature before uploading weights", async () => {
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
      // The diagnostic must not call the caller's conversion: if it ran, "boom" would escape
      // instead of `ExecutionError`.
      for (const value of [
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
      ]) {
        const options: SessionOptions = {};
        Reflect.set(options, "rmsNormReduce", value);
        const error = await assertRejects(
          () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
          ExecutionError,
          "rmsNormReduce",
        );
        // Report only the type name, not the value itself.
        assertEquals(error.message.includes("object"), true, error.message);
      }
    } finally {
      gpu.destroy();
    }
  });
  describe("on a supporting implementation, checks the CPU reference, fusion, row counts and grid-stride", () => {
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
    // The adapter's features are known only at run time, so the skip happens inside the case.
    it("runs with subgroups (explicit SKIP when Deno lacks support; Chrome runs the same checks separately)", async (context) => {
      if (!supported) context.skip();
      const gpu = await acquireGpu({ subgroups: true });
      try {
        const parts = await checkRmsSubgroup(gpu);
        // 12 widths × 4 row counts; 3 widths × 2 addition orders × with and without cancellation;
        // one grid-stride run.
        assertEquals(
          parts.map(({ id, cases }) => [id, cases.length]),
          Object.entries(RMS_SUBGROUP_CASES),
        );
        for (const part of parts) assertCheckPart(part);
      } finally {
        gpu.destroy();
      }
    });
  });
});
