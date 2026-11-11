// The twin cases of the linear→SRQ Session fusion live in @karume/kernel-checks
// (checkLinearStaticQuantizeTwins), which the playground's /kernel-check page runs on the browser's
// backend too; the row identity across M, the borrowed scale's lifetimes and the construction
// rejections stay here.

import { assert, assertEquals, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  checkLinearStaticQuantizeTwins,
  LINEAR_STATIC_QUANTIZE_ROWS,
  LINEAR_STATIC_QUANTIZE_STORAGES,
  LINEAR_STATIC_QUANTIZE_TWIN_CASES,
  linearStaticQuantizeInput,
  linearStaticQuantizeWidth,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { linearStaticQuantizeModel } from "./helpers/linear-static-quantize-graph.ts";
import { GRAPH_NAME } from "./helpers/model-fixture.ts";

const input = linearStaticQuantizeInput;
const words = (data: ArrayBufferView<ArrayBuffer>) =>
  new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);

describe.skipIf(!GPU_AVAILABLE)("Session fusion of linear→SRQ and its lifetimes (real GPU)", () => {
  it("INT2/4/8 at M1/4/8 match the unfused path, and M9 stays unfused", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkLinearStaticQuantizeTwins(gpu);
      assertEquals(
        [part.id, part.cases.length],
        Object.entries(LINEAR_STATIC_QUANTIZE_TWIN_CASES)[0],
      );
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });
  it("INT2/4/8 at M1/4/8 return the same row 0, on both runs", async () => {
    const gpu = await acquireGpu();
    try {
      for (const storage of LINEAR_STATIC_QUANTIZE_STORAGES) {
        const n = linearStaticQuantizeWidth(storage);
        let first: Uint32Array<ArrayBuffer> | undefined;
        for (const m of LINEAR_STATIC_QUANTIZE_ROWS.filter((rows) => rows <= 8)) {
          const fused = await createSessionFromContainer(
            gpu,
            linearStaticQuantizeModel({ storage, n, m }),
            GRAPH_NAME,
            { linearGemvReduce: "parallel", fuseLinearStaticQuantize: true },
          );
          try {
            const x = input(m);
            // The second run uses the derived plan.
            for (let repeat = 0; repeat < 2; repeat++) {
              const row = words((await fused.run(x)).y.data).slice(0, n);
              first ??= row;
              assertEquals(row, first);
            }
          } finally {
            await fused.dispose();
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
      for (const storage of LINEAR_STATIC_QUANTIZE_STORAGES) {
        const n = linearStaticQuantizeWidth(storage);
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
