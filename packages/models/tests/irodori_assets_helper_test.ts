// `tests/helpers/irodori-assets.ts` の比較子 `worstDifference` の挙動テスト。
//
// この比較子は実 GPU E2E（`e2e_irodori_latent_test.ts` / `e2e_irodori_w8a8_test.ts`）の
// tolerance 門が見る**唯一の数**で、資産も GPU も要る門の中に埋まっている。そのため比較子
// 自身の退行は「門が緑のまま何も検出しない」形でしか出ない — ここが資産なしで踏める唯一の席。

import { assertEquals, assertThrows } from "@std/assert";
import { worstDifference } from "./helpers/irodori-assets.ts";

Deno.test("worstDifference: 最大絶対差とその位置を返す", () => {
  const actual = Float32Array.from([1, 2, 3, 4]);
  const golden = Float32Array.from([1, 2.5, 3, 4.25]);
  const { maxAbs, at } = worstDifference(actual, golden);
  assertEquals(at, 1, "最悪の位置");
  assertEquals(Math.round(maxAbs * 100) / 100, 0.5);
});

Deno.test("worstDifference: 要素数が違えば落とす", () => {
  assertThrows(
    () => worstDifference(Float32Array.from([1]), Float32Array.from([1, 2])),
    Error,
    "要素数が違う",
  );
});

Deno.test("worstDifference: 全要素 NaN を「差 0」として通さない", () => {
  // 素朴な `difference > maxAbs` は NaN で常に偽なので、maxAbs は初期値 0 のまま返り
  // 上流の `maxAbs > zAtol` が偽 = **全 NaN の latent が tolerance 門を通り抜ける**。
  const nan = Float32Array.from([NaN, NaN, NaN]);
  assertThrows(
    () => worstDifference(nan, Float32Array.from([0, 0, 0])),
    Error,
    "非有限の標本が 3 個ある",
  );
});

Deno.test("worstDifference: 有限標本に紛れた 1 個の非有限も落とす（位置を添える）", () => {
  // 「有限 1 標本 + 残り NaN」で maxAbs が小さく出る形（判別帯の下限では捕まらない）。
  const actual = Float32Array.from([0.1, Infinity, 0.2]);
  const error = assertThrows(
    () => worstDifference(actual, Float32Array.from([0.1, 0.1, 0.2])),
    Error,
    "非有限の標本が 1 個ある",
  );
  assertEquals(error.message.includes("最初は 1"), true, error.message);
});
