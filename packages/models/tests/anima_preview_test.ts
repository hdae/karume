// 途中 latent → RGB プレビュー近似の挙動テスト。GPU も資産も要らない純関数。

import { assertEquals, assertThrows } from "@std/assert";
import { approximatePreview } from "../src/anima/preview.ts";

/**
 * 16ch・2×2 の合成 latent（planar `[1,16,2,2]`）。値は `{-1,-0.5,0,0.5,1}` を
 * チャネル × 画素でずらして配る — 全チャネル・全画素が違う寄与を持つので、係数表のどの行が
 * 書き換わっても出力が動く（下の golden の前提）。
 */
const syntheticLatents = (): Float32Array<ArrayBuffer> =>
  Float32Array.from(
    { length: 16 * 4 },
    (_, at) => ((Math.floor(at / 4) + (at % 4)) % 5 - 2) * 0.5,
  );

Deno.test("approximatePreview: 入口ゲート（軸数 / B>1 / チャネル数 / 要素数）", () => {
  assertThrows(
    () => approximatePreview(new Float32Array(16), [16, 2, 2]),
    Error,
    "4 軸でない",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(2 * 16 * 4), [2, 16, 2, 2]),
    Error,
    "batch=1 前提",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(4 * 4), [1, 4, 2, 2]),
    Error,
    "チャネル数",
  );
  assertThrows(
    () => approximatePreview(new Float32Array(16 * 4 + 1), [1, 16, 2, 2]),
    Error,
    "要素数",
  );
});

Deno.test("approximatePreview: 寸法は latent の H/W・アルファ 255", () => {
  // 幅と高さを取り違えても要素数は合ってしまうので、非正方（2×3）で位置を縛る。
  const got = approximatePreview(new Float32Array(16 * 6), [1, 16, 3, 2]);
  assertEquals([got.width, got.height], [2, 3]);
  assertEquals(got.rgba.length, 2 * 3 * 4);
  for (let index = 0; index < 6; index += 1) {
    assertEquals(got.rgba[index * 4 + 3], 255, `画素 ${index} のアルファ`);
  }
});

Deno.test("approximatePreview: 合成 latent の RGBA golden（係数表の書き換わりを止める）", () => {
  // 係数表は非公開の内部定数なので、外から縛れるのは出力だけ。この golden が係数の門。
  //
  // フォールト注入（2026-08-24 実測・注入は即座に戻した）:
  // ①`preview.ts` の係数表 1 行目 `-0.1299` を `-0.1199` に書き換えてこの 1 本を回すと
  //   `FAILED | 2 passed | 1 failed` で落ちた（差分は 4 バイト目 `78` → `77` の 1 箇所）。
  // ②係数表 16×3 + bias 3 の**51 個すべて**を 1 個ずつ +0.01 して掃いた結果は
  //   `delta=0.01 で golden が動かない定数: 無し` — どの定数が書き換わっても落ちる。
  const got = approximatePreview(syntheticLatents(), [1, 16, 2, 2]);
  assertEquals([got.width, got.height], [2, 2]);
  assertEquals([...got.rgba], [
    133,
    173,
    84,
    255,
    78,
    127,
    32,
    255,
    102,
    120,
    114,
    255,
    48,
    4,
    25,
    255,
  ]);
});
