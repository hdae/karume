// Asset-free structural checks of the host-side `t_embed` (no golden, no GPU). The torch golden
// comparison of the `t_embed` table and of the t schedule reads outputs/series, so it lives in
// packages/e2e/tests/e2e_irodori_t_embed_test.ts.

import { test } from "@karume/test-support/test";
import { assertEquals, assertThrows } from "@std/assert";
import { timestepEmbedding, timestepFrequencies } from "../src/irodori/host/t-embed.ts";

// ---- 構造の門（golden 不要）------------------------------------------------
//
// NOTE: the golden comparison referred to below is in packages/e2e/tests/e2e_irodori_t_embed_test.ts.
//
// 上の golden 突合は 2 本とも `ignore: !AVAILABLE` なので、資産の無い機では `t_embed` の門が
// 0 本になる。anima の同型 `timestepsProj` は資産不要の構造テストを持つ（`anima_sampler_test.ts`）
// ので、こちら側にも同じ密度で置く。

test("timestepEmbedding: t=0 では前半が全て 1・後半が全て 0（前半 cos / 後半 sin）", () => {
  // 値域が同じ [-1,1] なので、前後半を取り違えても shape も統計も変わらない（doc の MUST）。
  // 反転していれば `[0,0,0,0,1,1,1,1]` になってここで落ちる。
  assertEquals([...timestepEmbedding(0, timestepFrequencies(8))], [1, 1, 1, 1, 0, 0, 0, 0]);
});

test("timestepFrequencies: k=0 は厳密に 1000・以降は上流の式順どおり", () => {
  // `FREQ_SCALE`（1000）の掛け忘れは k=0 が 1 になるのでここで落ちる（golden 突合での実測は
  // 最大絶対差 2.00e+0）。k=1 は `1000·exp(f32(-ln(10000)·1)/2)` を 1 演算ずつ f32 で踏んだ値。
  const expected = Math.fround(
    1000 * Math.fround(Math.exp(Math.fround(Math.fround(-Math.log(10000) * 1) / 2))),
  );
  assertEquals([...timestepFrequencies(4)], [1000, expected]);
});

test("timestepFrequencies: 幅の受理集合（正の偶数）", () => {
  // 前半 cos / 後半 sin に割るので奇数幅は組めない（`config.ts` の `timestepEmbedDim` と
  // 同じ条件がこちら側にもある）。
  for (const dim of [0, -2, 3, 2.5]) {
    assertThrows(() => timestepFrequencies(dim), RangeError, "正の偶数でない");
  }
});
