// GenerationContext の GPU 非依存な純関数 3 本（`assertChunkLength` / `resolveBindings` /
// `resolveSlotShape`）の受理集合。3 本は**見積り（estimate.ts）と実構築が共有する唯一の
// 受理集合**なので、門の正本は device を要らない側に置く — 実 GPU 経由の同じ門
// （gpu_generation_context_test.ts）はアダプタ無しの環境では 1 本も走らない。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import {
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
