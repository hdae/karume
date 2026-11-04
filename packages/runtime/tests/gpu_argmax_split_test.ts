/**
 * Boundary of argmax's two-phase split (the "two-phase split" of src/kernels/argmax.ts) — input
 * with an empty outer extent.
 *
 * `argmax([0, D])`, whose product of leading dimensions is 0, is valid by contract and returns the
 * empty output `[0, 1]`. On a long row (D ≥ `ARGMAX_SPLIT_MIN_DIM`) entering the split path makes
 * the temporary for the partial results 0 B, which trips the recipe's lifetime contract (a
 * temporary is at least 1 B) and fails with an internal error unrelated to the input. As for the
 * same shape of topk(k=1) (gpu_topk_split_test.ts), this pins that an empty outer extent stays on
 * the single-dispatch form.
 */
import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, prepareContainer } from "../mod.ts";
import { GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

test.skipIf(!GPU_AVAILABLE)(
  "argmax: with an empty outer extent, even a long row creates no split temporary and returns the empty output",
  async function () {
    const gpu = await acquireGpu();
    try {
      for (const dim of [16383, 16384, 262144]) {
        const prepared = prepareContainer(
          await openModelBytes(
            singleOpDeclaration("argmax", [[0, dim]], [[0, 1]], { outDtypes: ["i32"] }),
            [],
          ),
          GRAPH_NAME,
        );
        const expected = prepared.estimate();
        const session = await prepared.createContainerSession(gpu);
        try {
          for (let frame = 0; frame < 2; frame++) {
            const out = await session.run({
              x0: { dtype: "f32", shape: [0, dim], data: new Float32Array(0) },
            });
            assertEquals(out.y.data.length, 0, `dim=${dim}`);
            const actual = session.diagnostics().lastRun?.peakTransientBytes;
            assert(actual !== undefined);
            assert(expected.scenarios[0].workspaceBytes >= actual);
          }
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
