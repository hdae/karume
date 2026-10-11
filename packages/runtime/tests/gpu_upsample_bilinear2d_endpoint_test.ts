/**
 * upsample_bilinear2d の 1 軸が 2^24 を超える形で、端点の添字が隣の平面へはみ出さないこと。
 *
 * scale = `fl((in − 1) / (out − 1))` と源座標 `fl(scale · 出力添字)` の f32 丸めで、末尾の源座標が
 * `in − 1` を超えうる（in_w = 16777220・out_w = 2 で 16777220 = in_w）。整数化した添字に上限が
 * 無いと、平面 0 の端点が平面 1 の先頭を読む（例外の出ない誤値）。torch の
 * `guard_index_and_lambda` と同じく添字を `in − 1` に留め、端点は入力の末尾になる。
 *
 * 入力は 2 平面 × 16,777,220 要素（約 134 MB）。この上限は 1 軸がおよそ 2^23 を超える形でだけ
 * 発動する（in_w 8,650,802・out_w 34 でも末尾の源座標が in_w に届く）。
 */
import { assertEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { acquireGpu, prepareContainer } from "../mod.ts";
import { referenceUpsampleBilinear2d } from "../src/reference/ops.ts";
import { GRAPH_NAME, openModelBytes, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const WIDTH_IN = 16_777_220;
const INPUT_SHAPE = [1, 2, 1, WIDTH_IN];
const OUTPUT_SHAPE = [1, 2, 1, 2];

/** 平面 0 = [1, 1, …, 7]・平面 1 = [100, 2, …, 9]（端と先頭だけ値を変えて、どこを読んだか分かる形）。 */
const input = (): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(2 * WIDTH_IN);
  data.fill(1, 0, WIDTH_IN);
  data[WIDTH_IN - 1] = 7;
  data.fill(2, WIDTH_IN);
  data[WIDTH_IN] = 100;
  data[2 * WIDTH_IN - 1] = 9;
  return data;
};

// 各平面の [先頭, 末尾]（縮小 out_w = 2 は両端の 2 点だけを読む）。
const EXPECTED = [1, 7, 100, 9];

describe("upsample_bilinear2d: 1 軸が 2^24 を超える縮小の端点", () => {
  it("CPU 参照は各平面の末尾を読む（隣の平面の先頭を読まない）", () => {
    const out = referenceUpsampleBilinear2d(
      { dtype: "f32", shape: INPUT_SHAPE, data: input() },
      { output_size: [1, 2] },
    );
    assertEquals([...out.data], EXPECTED);
  });

  it({
    name: "GPU も各平面の末尾を読む（実 GPU）",
    ignore: !GPU_AVAILABLE,
    async fn() {
      const gpu = await acquireGpu();
      try {
        const prepared = prepareContainer(
          await openModelBytes(
            singleOpDeclaration("upsample_bilinear2d", [INPUT_SHAPE], [OUTPUT_SHAPE], {
              attrs: { output_size: [1, 2] },
            }),
            [],
          ),
          GRAPH_NAME,
        );
        const session = await prepared.createContainerSession(gpu);
        try {
          const out = await session.run({
            x0: { dtype: "f32", shape: INPUT_SHAPE, data: input() },
          });
          assertEquals([...out.y.data], EXPECTED);
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    },
  });
});
