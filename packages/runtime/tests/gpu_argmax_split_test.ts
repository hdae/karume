/**
 * argmax の 2 相分割（src/kernels/argmax.ts の「2 相分割」）の境界 — 外側が空の入力。
 *
 * 先行次元の積が 0 の `argmax([0, D])` は契約上 valid で、空の出力 `[0, 1]` を返す。長い行
 * （D ≥ `ARGMAX_SPLIT_MIN_DIM`）で分割経路へ入ると部分結果の一時が 0 B になり、レシピの寿命契約
 * （一時は 1 B 以上）に掛かって入力と無関係な内部エラーで落ちる。topk(k=1) の同形
 * （gpu_topk_split_test.ts）と同じく、外側が空なら 1 dispatch 形に残ることを固定する。
 */
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, prepareContainer } from "../mod.ts";
import { GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

Deno.test({
  name: "argmax: 外側が空なら長い行でも分割用一時を作らず空の出力を返す",
  ignore: !GPU_AVAILABLE,
  async fn() {
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
});
