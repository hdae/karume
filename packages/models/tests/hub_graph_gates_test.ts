// 焼かれたグラフの宣言と `pipelineConfig` の宣言の突合（`src/hub/graph-gates.ts`）。
// 4 family（birefnet / depth-anything / irodori / siglip2）が同じ 1 本を通る。
//
// この門は**失敗経路しか無い**: 通ってしまった食い違いはホスト側を最後まで素通りし、落ちるのは
// Session の shape 検査 = 「宣言とグラフのどちらが正しいのか」が読み手に伝わらない場所になる。
// 見るのは 3 点:
//
// ① 一致は通し、1 軸でもずれたら落とす（比較は `===` — 記号次元の文字列は数と一致しない）。
// ② 入力名そのものが無い場合は、軸を読む前に名指しで落とす。
// ③ 文言には family・どの宣言を見ていたか（`where`）・入力名・軸・実値・期待値が全部入る。

import { assert, assertThrows } from "@std/assert";
import { stubModel } from "./helpers/stub-model.ts";
import { assertGraphInputDim } from "../src/hub/graph-gates.ts";

/** 224² の vision グラフ（`patch` で 1 点だけ壊す）。 */
const visionGraph = (shape: readonly (number | string)[] = [1, 3, 224, 224]) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape }],
    outputs: ["pooler_output"],
    values: { pooler_output: [1, 768] },
  });

Deno.test("assertGraphInputDim: 宣言どおりの軸は通る", () => {
  assertGraphInputDim("siglip2", visionGraph(), "pixel_values", 2, 224, "imageHeight");
  assertGraphInputDim("siglip2", visionGraph(), "pixel_values", 1, 3, "RGB の 3 チャネル");
});

Deno.test("assertGraphInputDim: 軸の食い違いは family・where・実値・期待値を名指しで落とす", () => {
  const error = assertThrows(
    () =>
      assertGraphInputDim(
        "siglip2",
        visionGraph([1, 3, 384, 384]),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "siglip2: imageHeight — グラフ入力 'pixel_values' の軸 2 が 384",
  );
  assert(error.message.includes("pipelineConfig は 224"), error.message);
});

Deno.test("assertGraphInputDim: family ラベルは呼び手のものがそのまま出る", () => {
  assertThrows(
    () =>
      assertGraphInputDim("depth-anything", visionGraph(), "pixel_values", 3, 518, "imageWidth"),
    Error,
    "depth-anything: imageWidth — グラフ入力 'pixel_values' の軸 3 が 224、pipelineConfig は 518",
  );
});

Deno.test("assertGraphInputDim: 入力名そのものが無ければ軸を読む前に落ちる", () => {
  assertThrows(
    () => assertGraphInputDim("irodori", visionGraph(), "pixel", 2, 224, "imageHeight"),
    Error,
    "irodori: グラフ入力 'pixel' が無い（imageHeight）",
  );
});

Deno.test("assertGraphInputDim: 記号次元の軸は静的値と一致しないので落ちる", () => {
  // 記号次元（データ依存の軸）を静的宣言として要求した形。`===` なので文字列は数と一致せず、
  // 文言には記号名がそのまま出る（「静的に焼かれていない」と読める）。
  assertThrows(
    () =>
      assertGraphInputDim(
        "birefnet",
        visionGraph([1, 3, "H", 224]),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "birefnet: imageHeight — グラフ入力 'pixel_values' の軸 2 が H、pipelineConfig は 224",
  );
});

Deno.test("assertGraphInputDim: 宣言の次元数より外の軸は undefined として落ちる", () => {
  // 軸番号の取り違え（4 次元の宣言に軸 4 を要求）。黙って通すと「見ていない軸」が増える。
  assertThrows(
    () => assertGraphInputDim("birefnet", visionGraph(), "pixel_values", 4, 224, "imageHeight"),
    Error,
    "birefnet: imageHeight — グラフ入力 'pixel_values' の軸 4 が undefined、pipelineConfig は 224",
  );
});
