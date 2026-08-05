// 解像度の綴りと受理集合。
//
// **受理集合の破れは実行時の深い場所でしか落ちない**（rope 素表の行数超過 / タイル配置の
// 不成立 / Dim("S") の上限）ので、入口で条件を名指しできていることをここで固定する。
// 資産も GPU も要らない純関数。

import { assert, assertEquals, assertThrows } from "@std/assert";
import {
  assertAcceptableResolution,
  formatResolution,
  MAX_DIT_TOKENS,
  MAX_RESOLUTION_SIDE,
  MIN_RESOLUTION_SIDE,
  parseResolution,
  RESOLUTION_GRANULARITY,
} from "../src/anima/resolution.ts";

Deno.test("parseResolution: WxH と正方の略記を受ける（往復は恒等）", () => {
  assertEquals(parseResolution("1344x768"), { width: 1344, height: 768 });
  assertEquals(parseResolution("768x1344"), { width: 768, height: 1344 });
  assertEquals(parseResolution("512"), { width: 512, height: 512 });
  assertEquals(parseResolution("1024X1024"), { width: 1024, height: 1024 });
  for (const raw of ["512", "1024", "1344x768", "896x1152"]) {
    assertEquals(formatResolution(parseResolution(raw)), raw, `往復: ${raw}`);
  }
});

Deno.test("parseResolution: 綴りでない入力を落とす（既定へ縮退させない）", () => {
  for (const raw of ["", "1344*768", "1344x", "x768", "1344x768x1", "-512", "51 2", "1e3"]) {
    assertThrows(() => parseResolution(raw), Error, "が WxH でも", `綴り: ${JSON.stringify(raw)}`);
  }
});

Deno.test("受理集合: 4 条件をそれぞれ名指しで落とす", () => {
  // ①刻み（16 の倍数）— 8 の倍数でも latent が patch 2 で割り切れない。
  assertThrows(() => parseResolution("1352x768"), Error, `${RESOLUTION_GRANULARITY} の倍数でない`);
  // ②下限（VAE タイル decoder の latent 64 が入らない）。
  assertThrows(() => parseResolution(`1024x${MIN_RESOLUTION_SIDE - 16}`), Error, "下限");
  // ③天井（rope 素表の行数）。
  assertThrows(() => parseResolution(`${MAX_RESOLUTION_SIDE + 16}x512`), Error, "上限");
  // ④S の上限。**各辺は受理集合の内側でも面積で超える**（2048×2048 = 16,384 がちょうど上限）。
  assertEquals(parseResolution("2048x2048"), { width: 2048, height: 2048 });
});

Deno.test("assertAcceptableResolution: 構造体で渡す経路も同じ 4 条件で見る", () => {
  // manifest の `pipelineConfig.defaults.resolution` と `generate` の request はここを通る
  // （綴りを経由しないので、綴り側だけに検査があると素通しになる）。
  assertAcceptableResolution({ width: 1024, height: 1024 });
  assertThrows(() => assertAcceptableResolution({ width: 1024, height: 1000 }), Error, "倍数");
  assertThrows(() => assertAcceptableResolution({ width: 256, height: 1024 }), Error, "下限");
  assertThrows(() => assertAcceptableResolution({ width: 4096, height: 512 }), Error, "上限");
  assertThrows(() => assertAcceptableResolution({ width: 1024.5, height: 1024 }), Error, "整数");
});

Deno.test("受理集合: 4 パターン（16:9 / 3:4 の縦横）が全て S=4,032 で受理される", () => {
  // **縦横の対を両方**受けることと、S が Dim("S") の内側であることを、literal を置かずに
  // 数で押さえる。
  const patterns = ["1344x768", "768x1344", "1152x896", "896x1152"];
  for (const raw of patterns) {
    const size = parseResolution(raw);
    const tokens = (size.width / RESOLUTION_GRANULARITY) * (size.height / RESOLUTION_GRANULARITY);
    assertEquals(tokens, 4032, `${raw} のトークン長 S`);
    assert(tokens <= MAX_DIT_TOKENS, `${raw}: S が Dim("S") の上限を超えている`);
  }
});
