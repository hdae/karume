import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { linearStaticQuantizeModel } from "./helpers/linear-static-quantize-graph.ts";
import { GRAPH_NAME } from "./helpers/model-fixture.ts";

const input = (m: number, k = 1536) => ({
  x: {
    dtype: "f32" as const,
    shape: [m, k],
    data: Float32Array.from({ length: m * k }, (_, i) => Math.sin(i * 0.037) * 0.75),
  },
});
const words = (data: ArrayBufferView<ArrayBuffer>) =>
  new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);

describe.skipIf(!GPU_AVAILABLE)("Session fusion of linear→SRQ and its lifetimes (real GPU)", () => {
  it("INT2/4/8 at M1/4/8 match the unfused path on row 0, and M9 stays unfused", async () => {
    const gpu = await acquireGpu();
    try {
      for (const storage of ["i2", "i4", "i8"] as const) {
        const n = storage === "i2" ? 12288 : 256;
        let first: Uint32Array<ArrayBuffer> | undefined;
        for (const m of [1, 4, 8, 9]) {
          const model = linearStaticQuantizeModel({ storage, n, m });
          const baseline = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
            linearGemvReduce: "parallel",
          });
          try {
            const fused = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
              linearGemvReduce: "parallel",
              fuseLinearStaticQuantize: true,
            });
            try {
              const x = input(m),
                expected = await baseline.run(x);
              // The second run uses the derived plan. The same scale reference is kept, and x lives until the later neg.
              for (let repeat = 0; repeat < 2; repeat++) {
                const actual = await fused.run(x);
                assertEquals(words(actual.y.data), words(expected.y.data));
                assertEquals(words(actual.copy.data), words(expected.copy.data));
                assertEquals(
                  fused.diagnostics().lastRunFusions?.linearStaticQuantize,
                  m <= 8 ? 1 : 0,
                );
                if (m <= 8) {
                  first ??= words(actual.y.data).slice(0, n);
                  assertEquals(words(actual.y.data).slice(0, n), first);
                }
              }
            } finally {
              await fused.dispose();
            }
          } finally {
            await baseline.dispose();
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  });
  it("a borrowed scale protects the lender's lifetime and stays reusable from the lender after the borrower is destroyed", async () => {
    const gpu = await acquireGpu();
    try {
      for (const storage of ["i2", "i4", "i8"] as const) {
        const n = storage === "i2" ? 12288 : 256;
        const owner = await createSessionFromContainer(
          gpu,
          linearStaticQuantizeModel({ storage, n }),
          GRAPH_NAME,
          { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
        );
        try {
          const expected = await owner.run(input(4));
          const borrower = await createSessionFromContainer(
            gpu,
            linearStaticQuantizeModel({ storage, n, sharedWeight: true }),
            GRAPH_NAME,
            {
              linearGemvReduce: "parallel",
              fuseLinearStaticQuantize: true,
              sharedWeights: { w: owner.exportWeight("w") },
            },
          );
          try {
            await assertRejects(() => owner.dispose(), ExecutionError);
            for (let repeat = 0; repeat < 2; repeat++) {
              const actual = await borrower.run(input(4));
              assertEquals(words(actual.y.data), words(expected.y.data));
              assertEquals(borrower.diagnostics().lastRunFusions?.linearStaticQuantize, 1);
            }
          } finally {
            await borrower.dispose();
          }
          assertEquals(words((await owner.run(input(4))).y.data), words(expected.y.data));
        } finally {
          await owner.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  });
  it("rejects invalid types and unsupported compute modes at construction", async () => {
    const gpu = await acquireGpu();
    try {
      const model = linearStaticQuantizeModel();
      for (const value of [null, 0, 1, "true", [], {}]) {
        const options: SessionOptions = { linearGemvReduce: "parallel" };
        Object.defineProperty(options, "fuseLinearStaticQuantize", { value });
        await assertRejects(
          () => createSessionFromContainer(gpu, model, GRAPH_NAME, options),
          ExecutionError,
          "fuseLinearStaticQuantize",
        );
      }
      for (const options of [
        {},
        { linearGemvReduce: "sequential" },
        { linearGemvReduce: "parallel-subgroup32" },
        { linearGemvReduce: "parallel", linearCompute: "f16" },
        { linearGemvReduce: "parallel", linearCompute: "a8" },
      ] satisfies SessionOptions[]) {
        await assertRejects(
          () =>
            createSessionFromContainer(gpu, model, GRAPH_NAME, {
              ...options,
              fuseLinearStaticQuantize: true,
            }),
          ExecutionError,
          "fuseLinearStaticQuantize",
        );
      }
      const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, {
        fuseLinearStaticQuantize: false,
      });
      try {
        const result = await session.run(input(4));
        assert(result.y.data.every(Number.isFinite));
        assertEquals(session.diagnostics().lastRunFusions?.linearStaticQuantize, 0);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  });
});
