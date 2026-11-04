// The accepted sets of GenerationContext's 4 GPU-independent pure functions (`assertChunkLength` /
// `assertChunkBuckets` / `resolveBindings` / `resolveSlotShape`). The 4 are **the only accepted
// sets shared by the estimate (estimate.ts) and the real build**, so the source of truth for the
// check sits on the side that needs no device — the same check through a real GPU
// (gpu_generation_context_test.ts) runs not at all in an environment without an adapter.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { mergedGraph } from "../src/format/container/bind.ts";
import type { IrGraph } from "../src/format/ir.ts";
import {
  assertChunkBuckets,
  assertChunkLength,
  resolveBindings,
  resolveSlotShape,
} from "../src/runtime/generation-context.ts";
import { ExecutionError, type SymbolBindings } from "../src/runtime/plan.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

/**
 * A graph with 2 symbols. It declares `__proto__` to exercise the container object's properties —
 * the symbol grammar `[A-Za-z_][A-Za-z0-9_]*` matches this spelling.
 */
const symbolGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
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

/** Declaration → merged graph (0 initializers, so no supply is needed). */
const parse = (graph: DeclarationJson): IrGraph =>
  mergedGraph(memoryModel(graph, []).graphs[GRAPH_NAME], GRAPH_NAME);

test("assertChunkLength accepts only integers in 1..0xffffffff", () => {
  // 0 / negative / fractional / one above u32 (the value is carried as a u32, so the limit is
  // here).
  for (const chunkLength of [0, -1, 1.5, 2 ** 32]) {
    const error = assertThrows(() => assertChunkLength(chunkLength), ExecutionError);
    assert(error.message.includes(`chunkLength ${chunkLength}`), error.message);
  }
  assertChunkLength(1);
  assertChunkLength(0xffffffff);
});

test("assertChunkBuckets accepts only strictly ascending values in 2..chunkLength-1", () => {
  // Omitted and empty mean "no additional execution shapes" — one prefill shape + the decode shape,
  // as before.
  assertChunkBuckets(undefined, 8);
  assertChunkBuckets([], 8);
  assertChunkBuckets([2], 8);
  assertChunkBuckets([2, 3, 5, 7], 8);

  // Out of range: 1 (= the decode shape itself) / 0 / negative / non-integer / chunkLength / above
  // chunkLength.
  for (const buckets of [[1], [0], [-2], [2.5], [8], [9]]) {
    const error = assertThrows(() => assertChunkBuckets(buckets, 8), ExecutionError);
    assert(error.message.includes("chunkBuckets[0]"), error.message);
    assert(error.message.includes("is not an integer in 2..7"), error.message);
  }

  // Order: duplicates and descending order both prevent deciding "the smallest at or above
  // queryLength" by a linear scan.
  for (const buckets of [
    [2, 2],
    [4, 3],
  ]) {
    const error = assertThrows(() => assertChunkBuckets(buckets, 8), ExecutionError);
    assert(error.message.includes("chunkBuckets[1]"), error.message);
    assert(error.message.includes("not strictly ascending"), error.message);
  }

  // With chunkLength = 1 (a context with only the decode shape), no bucket is in range.
  assertChunkBuckets(undefined, 1);
  assertThrows(() => assertChunkBuckets([2], 1), ExecutionError);
});

test("resolveBindings rejects unknown symbols and values that are not non-negative integers", () => {
  const graph = parse(symbolGraph());

  const unknown = assertThrows(() => resolveBindings(graph, { D: 8 }), ExecutionError);
  assert(unknown.message.includes("Binding 'D'"), unknown.message);

  for (const value of [-1, 1.5]) {
    const error = assertThrows(() => resolveBindings(graph, { T: value }), ExecutionError);
    assert(error.message.includes("is not a non-negative integer"), error.message);
  }

  // 0 is rejected downstream (resolveSlotShape) as "capacity 0" — this only goes as far as
  // non-negative.
  assertEquals(resolveBindings(graph, { T: 0 })["T"], 0);
});

test("resolveBindings' container has a null prototype, so '__proto__' remains an own property", () => {
  const graph = parse(symbolGraph());
  // Written as a computed key (`{ __proto__: 4 }` is the syntax for setting [[Prototype]] and
  // creates no own property).
  const resolved = resolveBindings(graph, { ["__proto__"]: 4, T: 2 });

  assertEquals(Object.getPrototypeOf(resolved), null);
  assert(Object.hasOwn(resolved, "__proto__"));
  assertEquals(resolved["__proto__"], 4);
  assertEquals(resolved["T"], 2);
});

test("resolveSlotShape rejects unbound symbols and capacities that are not positive", () => {
  const bound: SymbolBindings = { C: 8 };
  assertEquals(resolveSlotShape("k", [1, 2, "C", 4], bound), [1, 2, 8, 4]);
  // A shape with only numeric dimensions is returned as is even with empty bindings.
  assertEquals(resolveSlotShape("k", [1, 2, 8, 4], {}), [1, 2, 8, 4]);

  const unbound = assertThrows(() => resolveSlotShape("k", [1, "C", 4], {}), ExecutionError);
  assert(unbound.message.includes("unbound symbol 'C'"), unbound.message);

  const empty = assertThrows(() => resolveSlotShape("k", [1, "C", 4], { C: 0 }), ExecutionError);
  assert(empty.message.includes("capacity 0"), empty.message);
});
