// Pillow の 8bit の分離型 resample の共通の核（`src/image/pillow-resample.ts`）の**事前条件**。
// 値の一致は呼び手ごとの fixture が縛る（BiRefNet の `birefnet_matte_test.ts`・Wan の LANCZOS の
// fixture）ので、ここは入力・出力寸法・窓の食い違いを黙って途中まで読まずに `RangeError` で落とす
// ことだけを見る（GPU 不要・資産不要）。

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  PILLOW_BICUBIC,
  PILLOW_LANCZOS,
  resampleU8Window,
  resizeU8PlaneBicubic,
} from "../src/image/pillow-resample.ts";

describe("resizeU8PlaneBicubic の事前条件", () => {
  it("平面の長さが寸法と合わなければ RangeError（短くても長くても）", () => {
    assertThrows(
      () => resizeU8PlaneBicubic(new Uint8Array(5), 3, 2, 4, 4),
      RangeError,
      "BICUBIC の入力 3×2×1 と長さ 5 が合わない",
    );
    assertThrows(
      () => resizeU8PlaneBicubic(new Uint8Array(7), 3, 2, 4, 4),
      RangeError,
      "と長さ 7 が合わない",
    );
  });

  it("入力の寸法が正の整数でなければ RangeError（長さが積と合っていても）", () => {
    // 0 × n は長さ 0 と積が合うので、長さの検査だけでは通ってしまう。
    assertThrows(() => resizeU8PlaneBicubic(new Uint8Array(0), 0, 3, 2, 2), RangeError, "入力");
    assertThrows(() => resizeU8PlaneBicubic(new Uint8Array(3), 1.5, 2, 2, 2), RangeError, "入力");
  });

  it("出力の寸法が正の整数でなければ RangeError", () => {
    for (const [width, height] of [[0, 2], [2, -1], [2.5, 2]] as const) {
      assertThrows(
        () => resizeU8PlaneBicubic(new Uint8Array(6), 3, 2, width, height),
        RangeError,
        `BICUBIC の出力寸法 ${width}×${height} が正の整数でない`,
      );
    }
  });

  it("事前条件を満たす入力は通る（上の検査が恒真に落とすだけでないことの対照）", () => {
    const out = resizeU8PlaneBicubic(new Uint8Array(6).fill(200), 3, 2, 5, 4);
    assertEquals(out.length, 20);
    assertEquals([...new Set(out)], [200]);
  });
});

describe("resampleU8Window の事前条件", () => {
  const image = { data: new Uint8Array(4 * 3 * 3), width: 4, height: 3, channels: 3 };

  it("窓が出力の内側に無ければ RangeError（はみ出し・負の座標・空の窓）", () => {
    for (
      const window of [
        { left: 6, top: 0, width: 2, height: 2 },
        { left: 0, top: 4, width: 2, height: 2 },
        { left: -1, top: 0, width: 2, height: 2 },
        { left: 0, top: 0, width: 0, height: 2 },
      ]
    ) {
      assertThrows(
        () => resampleU8Window(image, 7, 5, window, PILLOW_LANCZOS),
        RangeError,
        "LANCZOS の窓",
      );
    }
  });

  it("チャネル数が画素列の長さと合わなければ RangeError", () => {
    assertThrows(
      () =>
        resampleU8Window(
          { ...image, channels: 1 },
          7,
          5,
          { left: 0, top: 0, width: 7, height: 5 },
          PILLOW_BICUBIC,
        ),
      RangeError,
      "BICUBIC の入力 4×3×1 と長さ 36 が合わない",
    );
  });
});
