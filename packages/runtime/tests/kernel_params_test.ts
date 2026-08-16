// uniform の u32 域検査（src/codegen/params.ts の `assertU32Params`）と、それを通した
// カーネル params の入口（GPU 不要）。
//
// 検証の眼目は 2 点:
//
// 1. **域の端**（0 と 0xffff_ffff）は通し、その外側（超過 / 負 / 非整数 / NaN）は必ず落ちる。
//    `Uint32Array` への代入は 2^32 の剰余を黙って取るので、超過を通すと dispatch は回るのに
//    要素数だけ別物という沈黙誤値になる。
// 2. **文言にパラメタ名と値が載る**。溢れた欄が分からない例外は、uniform を数語持つ
//    カーネルでは診断の起点にならない。
//
// カーネル側は gather を代表に取る（要素数を素の safe-integer 判定で受けていた側 —
// 一本化前は u32 超過が黙って narrowing されていた）。

import { assertEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { assertU32Params } from "../src/codegen/params.ts";
import { GATHER_OOB_BITS, gatherParams } from "../src/kernels/gather.ts";

const U32_MAX = 0xffff_ffff;

Deno.test("assertU32Params は u32 の域（0 〜 0xffff_ffff）をそのまま通す", () => {
  assertU32Params("where", { zero: 0, max: U32_MAX, mid: 1 });
});

Deno.test("assertU32Params は域の外（超過 / 負 / 非整数 / NaN / 非有限）を fail loudly", () => {
  const cases: readonly (readonly [string, number])[] = [
    ["超過（0xffff_ffff + 1）", U32_MAX + 1],
    ["負", -1],
    ["非整数", 1.5],
    ["NaN", NaN],
    ["+Infinity", Infinity],
  ];
  for (const [label, value] of cases) {
    assertThrows(
      () => assertU32Params("gather params", { count: value }),
      CodegenError,
      "count",
      label,
    );
  }
});

Deno.test("assertU32Params の文言はパラメタ名と値を含む（どの欄が溢れたかが分かる）", () => {
  const error = assertThrows(
    () => assertU32Params("rope params", { n: 4, sequence: U32_MAX + 1 }),
    CodegenError,
  );
  assertEquals(error.message, "rope params: sequence は u32 の非負整数（4294967296）");
});

Deno.test("gatherParams は u32 を超える要素数を narrowing せずに落とす", () => {
  // 域の内側は従来どおり（検査の一本化で有効入力の挙動は 1 バイトも変わらない）
  assertEquals([...gatherParams(12, 3, 9)], [12, 3, 9, GATHER_OOB_BITS]);
  // 2^32 は Uint32Array へ入れると 0 に化ける（= 全要素が範囲外扱いで出力が NaN 一色）
  assertThrows(() => gatherParams(U32_MAX + 1, 3, 9), CodegenError, "count");
});
