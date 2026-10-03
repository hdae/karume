// umT5 のグラフの入出力を組む純関数（src/wan/umt5/session-io.ts — ADR 0119 決定 3・4）の挙動。
// GPU も実資産も要らない。グラフ入力の名前は recipe（`wan/umt5_patch.py` の `INPUT_NAMES`）が
// fixture `wan-umt5/relative-position.json` に書いた綴りと突き合わせる。

import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import type { Tensor } from "@karume/runtime";
import {
  buildUmt5RelativePositionBuckets,
  type I32Tensor,
} from "../src/wan/umt5/relative-position.ts";
import {
  padUmt5Context,
  umt5SessionInputs,
  WAN_UMT5_INPUT_IDS,
  WAN_UMT5_RELATIVE_POSITION_BUCKETS,
} from "../src/wan/umt5/session-io.ts";

const meta = JSON.parse(
  await Deno.readTextFile(new URL("./fixtures/wan-umt5/relative-position.json", import.meta.url)),
) as { readonly input_names: readonly string[] };

const idsOf = (...values: number[]): Int32Array<ArrayBuffer> => Int32Array.from(values);

Deno.test("umT5 のグラフ入力の名前が recipe の宣言（export の入力の並び）と一致", () => {
  assertEquals([WAN_UMT5_INPUT_IDS, WAN_UMT5_RELATIVE_POSITION_BUCKETS], meta.input_names);
});

Deno.test("umt5SessionInputs: id 列 [L] を [1, L] に・バケット表はそのまま渡す（写さない）", () => {
  const ids = idsOf(289, 748, 1);
  const buckets = buildUmt5RelativePositionBuckets(ids.length);

  const inputs = umt5SessionInputs(ids, buckets);

  assertEquals(Object.keys(inputs).sort(), [...meta.input_names].sort());
  assertEquals(inputs[WAN_UMT5_INPUT_IDS].dtype, "i32");
  assertEquals(inputs[WAN_UMT5_INPUT_IDS].shape, [1, 3]);
  assertStrictEquals(inputs[WAN_UMT5_INPUT_IDS].data, ids);
  assertStrictEquals(inputs[WAN_UMT5_RELATIVE_POSITION_BUCKETS], buckets);
});

Deno.test("umt5SessionInputs: 表の長さが id 列と違えば落とす（L は 2 本の入力で 1 つ）", () => {
  const ids = idsOf(289, 748, 1);
  for (const length of [2, 4]) {
    assertThrows(
      () => umt5SessionInputs(ids, buildUmt5RelativePositionBuckets(length)),
      Error,
      "でない",
      `表の長さ ${length}`,
    );
  }
  const flat: I32Tensor = { dtype: "i32", shape: [9], data: new Int32Array(9) };
  assertThrows(() => umt5SessionInputs(ids, flat), Error, "でない");
  const short: I32Tensor = { dtype: "i32", shape: [3, 3], data: new Int32Array(8) };
  assertThrows(() => umt5SessionInputs(ids, short), Error, "要素数");
});

Deno.test("umt5SessionInputs: id 列の長さが 2〜512 の外なら落とす", () => {
  const buckets = buildUmt5RelativePositionBuckets(2);
  assertThrows(() => umt5SessionInputs(idsOf(1), buckets), Error, "の外");
  assertThrows(() => umt5SessionInputs(new Int32Array(513), buckets), Error, "の外");
});

/** `[1, tokens, width]` の出力（値 = 1 始まりの通し番号 — 位置の取り違えが値で見える）。 */
const outputOf = (tokens: number, width: number): Tensor => ({
  dtype: "f32",
  shape: [1, tokens, width],
  data: Float32Array.from({ length: tokens * width }, (_, index) => index + 1),
});

Deno.test("padUmt5Context: 有効長の行はそのまま・後ろの行はゼロ", () => {
  const tokens = 3;
  const width = 4;
  const rows = 6;

  const padded = padUmt5Context(outputOf(tokens, width), tokens, rows, width);

  assertEquals(padded.length, rows * width);
  assertEquals(
    [...padded.subarray(0, tokens * width)],
    Array.from({ length: tokens * width }, (_, index) => index + 1),
  );
  assertEquals([...padded.subarray(tokens * width)], new Array((rows - tokens) * width).fill(0));
});

Deno.test("padUmt5Context: 有効長が文脈の行数と同じなら詰めずに全行", () => {
  const padded = padUmt5Context(outputOf(5, 2), 5, 5, 2);

  assertEquals([...padded], Array.from({ length: 10 }, (_, index) => index + 1));
});

Deno.test("padUmt5Context: 形・dtype・行数の食い違いを落とす", () => {
  const output = outputOf(3, 4);
  // 入力に渡した長さと出力の行数が違う（positive / negative の取り違えなど）。
  assertThrows(() => padUmt5Context(output, 2, 6, 4), Error, "でない");
  // DiT の文脈の幅と違う。
  assertThrows(() => padUmt5Context(output, 3, 6, 8), Error, "でない");
  // 文脈の行数を超える。
  assertThrows(() => padUmt5Context(output, 3, 2, 4), Error, "超える");
  const integer: Tensor = { dtype: "i32", shape: [1, 3, 4], data: new Int32Array(12) };
  assertThrows(() => padUmt5Context(integer, 3, 6, 4), Error, "f32");
  const rank2: Tensor = { dtype: "f32", shape: [3, 4], data: new Float32Array(12) };
  assertThrows(() => padUmt5Context(rank2, 3, 6, 4), Error, "でない");
  const short: Tensor = { dtype: "f32", shape: [1, 3, 4], data: new Float32Array(11) };
  assertThrows(() => padUmt5Context(short, 3, 6, 4), Error, "要素数");
});
