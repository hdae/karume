// GenerationContext の GPU 非依存な純関数 4 本（`assertChunkLength` / `assertChunkBuckets` /
// `resolveBindings` / `resolveSlotShape`）の受理集合。4 本は**見積り（estimate.ts）と実構築が
// 共有する唯一の受理集合**なので、門の正本は device を要らない側に置く — 実 GPU 経由の同じ門
// （gpu_generation_context_test.ts）はアダプタ無しの環境では 1 本も走らない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import {
  assertChunkBuckets,
  assertChunkLength,
  resolveBindings,
  resolveSlotShape,
} from "../src/runtime/generation-context.ts";
import { ExecutionError, type SymbolBindings } from "../src/runtime/plan.ts";
import type { GraphJson } from "./helpers/format.ts";

/**
 * 記号 2 本のグラフ。`__proto__` を宣言しているのは器の性質を撃つため — シンボルの文法
 * `[A-Za-z_][A-Za-z0-9_]*` はこの綴りにマッチする。
 */
const symbolGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["add"] },
  symbols: ["T", "__proto__"],
  inputs: [
    { name: "x", dtype: "f32", shape: ["T", 4] },
    { name: "p", dtype: "f32", shape: ["__proto__", 4] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["T", 4] } },
  nodes: [{ op: "add", ins: ["x", "p"], outs: ["y"], attrs: {} }],
});

const parse = (graph: GraphJson): IrGraph => parseIrGraph(JSON.stringify(graph));

Deno.test("assertChunkLength は 1..0xffffffff の整数だけを受理する", () => {
  // 0 / 負 / 小数 / u32 の 1 つ上（搬送先が u32 なので上限はここ）。
  for (const chunkLength of [0, -1, 1.5, 2 ** 32]) {
    const error = assertThrows(() => assertChunkLength(chunkLength), ExecutionError);
    assert(error.message.includes(`chunkLength ${chunkLength}`), error.message);
  }
  assertChunkLength(1);
  assertChunkLength(0xffffffff);
});

Deno.test("assertChunkBuckets は 2..chunkLength-1 の狭義昇順だけを受理する", () => {
  // 省略と空は「追加の実行形なし」— 従来どおり prefill 形 1 本 + decode 形。
  assertChunkBuckets(undefined, 8);
  assertChunkBuckets([], 8);
  assertChunkBuckets([2], 8);
  assertChunkBuckets([2, 3, 5, 7], 8);

  // 値域外: 1（= decode 形そのもの）/ 0 / 負 / 非整数 / chunkLength / chunkLength 超。
  for (const buckets of [[1], [0], [-2], [2.5], [8], [9]]) {
    const error = assertThrows(() => assertChunkBuckets(buckets, 8), ExecutionError);
    assert(error.message.includes("chunkBuckets[0]"), error.message);
    assert(error.message.includes("2..7 の整数でない"), error.message);
  }

  // 順序: 重複も降順も「queryLength 以上の最小」を線形走査で決められなくする。
  for (const buckets of [[2, 2], [4, 3]]) {
    const error = assertThrows(() => assertChunkBuckets(buckets, 8), ExecutionError);
    assert(error.message.includes("chunkBuckets[1]"), error.message);
    assert(error.message.includes("狭義昇順でない"), error.message);
  }

  // chunkLength = 1（decode 形しか無い context）ではどんなバケットも値域に入らない。
  assertChunkBuckets(undefined, 1);
  assertThrows(() => assertChunkBuckets([2], 1), ExecutionError);
});

Deno.test("resolveBindings は未知の記号と非負整数でない値を拒否する", () => {
  const graph = parse(symbolGraph());

  const unknown = assertThrows(() => resolveBindings(graph, { D: 8 }), ExecutionError);
  assert(unknown.message.includes("束縛 'D'"), unknown.message);

  for (const value of [-1, 1.5]) {
    const error = assertThrows(() => resolveBindings(graph, { T: value }), ExecutionError);
    assert(error.message.includes("非負整数でない"), error.message);
  }

  // 0 は「容量 0」として下流（resolveSlotShape）が落とす — ここは非負まで。
  assertEquals(resolveBindings(graph, { T: 0 })["T"], 0);
});

Deno.test("resolveBindings の器は null プロトタイプで '__proto__' が own property として残る", () => {
  const graph = parse(symbolGraph());
  // 計算キーで書く（`{ __proto__: 4 }` は [[Prototype]] 設定の構文で own property を作らない）。
  const resolved = resolveBindings(graph, { ["__proto__"]: 4, T: 2 });

  assertEquals(Object.getPrototypeOf(resolved), null);
  assert(Object.hasOwn(resolved, "__proto__"));
  assertEquals(resolved["__proto__"], 4);
  assertEquals(resolved["T"], 2);
});

Deno.test("resolveSlotShape は未束縛シンボルと正でない容量を拒否する", () => {
  const bound: SymbolBindings = { C: 8 };
  assertEquals(resolveSlotShape("k", [1, 2, "C", 4], bound), [1, 2, 8, 4]);
  // 数値次元だけの shape は束縛が空でもそのまま返る。
  assertEquals(resolveSlotShape("k", [1, 2, 8, 4], {}), [1, 2, 8, 4]);

  const unbound = assertThrows(() => resolveSlotShape("k", [1, "C", 4], {}), ExecutionError);
  assert(unbound.message.includes("'C' が束縛されていない"), unbound.message);

  const empty = assertThrows(
    () => resolveSlotShape("k", [1, "C", 4], { C: 0 }),
    ExecutionError,
  );
  assert(empty.message.includes("容量 0"), empty.message);
});
