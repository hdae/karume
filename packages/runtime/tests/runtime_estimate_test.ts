// 見積りの中間バッファ規則（`src/runtime/estimate.ts` の `transientSlotBytes`）の門。
// GPU を 1 つも触らない純関数テスト。
//
// It pins one point: the intermediate total is the **sum of the plan's regions** (ADR 0093:
// live-interval packing — each slot, largest first, goes to the conflict-free region it grows least,
// a new region when every candidate would grow by the slot's full size, best total of two orders —
// with 256-byte offset alignment, and two slots read and written by the same dispatch in different
// regions), **not** the maximum of simultaneously live bytes. The rule's source of truth is the one
// `planTransients`, but when the allocation program the estimate builds (one dispatch per node)
// drifts from the execution phase the symptom is silent (no exception, no warning, only the number
// moves), so the formula itself is pinned with hand-derived constants.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import type { PreparedModel } from "../src/runtime/executor.ts";
import { openGraph, weight } from "./helpers/estimate-graphs.ts";
import { f32Bytes } from "./helpers/model-fixture.ts";

const workspaceBytes = (model: PreparedModel): number => {
  const report = model.estimate();
  assertEquals(
    report.scenarios.map((scenario) => scenario.name),
    ["run"],
  );
  return report.scenarios[0].workspaceBytes;
};

/**
 * `x[1,2] → matmul(w1[2,2]) → h1[1,2] → matmul(w2[2,3]) → h2[1,3] → matmul(w3[3,1]) → h3[1,1]`
 * の 3 段。中間は 8 / 12 / 4 バイトで、**同時に生きるのは高々 2 本**（8+12 = 20）。
 */
const chainModel = (): PreparedModel =>
  openGraph(
    {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["matmul"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
      outputs: ["h3"],
      initializers: { w1: {}, w2: {}, w3: {} },
      values: {
        w1: { dtype: "f32", shape: [2, 2] },
        w2: { dtype: "f32", shape: [2, 3] },
        w3: { dtype: "f32", shape: [3, 1] },
        h1: { dtype: "f32", shape: [1, 2] },
        h2: { dtype: "f32", shape: [1, 3] },
        h3: { dtype: "f32", shape: [1, 1] },
      },
      nodes: [
        { op: "matmul", ins: ["x", "w1"], outs: ["h1"], attrs: {} },
        { op: "matmul", ins: ["h1", "w2"], outs: ["h2"], attrs: {} },
        { op: "matmul", ins: ["h2", "w3"], outs: ["h3"], attrs: {} },
      ],
    },
    [
      weight("w1", f32Bytes(new Array(4).fill(1))),
      weight("w2", f32Bytes(new Array(6).fill(1))),
      weight("w3", f32Bytes(new Array(3).fill(1))),
    ],
  );

test("the intermediate total is the sum of the regions, and a freed range is handed out regardless of size", () => {
  // 8 (h1) + 12 (h2) + 4 (h3); intervals h1 [0,2), h2 [1,4), h3 [3,end). h2 (largest) opens
  // region 0. h1 overlaps h2, and behind h2 it would start at the aligned offset 256 and grow
  // region 0 by 256 + 8 - 12 = 252, more than its own 8 B, so it opens region 1 (growth decides;
  // the read / write rule would also keep it out). h3 reads h2 and takes h1's dead 8-byte range in
  // region 1 (growth 0; behind h2 it would grow region 0 by 248) = 12 + 8 = 20, the live peak (the
  // old exact-size pool summed all three to 24).
  assertEquals(workspaceBytes(chainModel()), 20);
});

/**
 * A `relu` chain with `h3` as the graph output (the shape does not change, so every intermediate is
 * in the same size class).
 */
const reluChainModel = (): PreparedModel =>
  openGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["relu"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
    outputs: ["h3"],
    initializers: {},
    values: {
      h1: { dtype: "f32", shape: [1, 2] },
      h2: { dtype: "f32", shape: [1, 2] },
      h3: { dtype: "f32", shape: [1, 2] },
    },
    nodes: [
      { op: "relu", ins: ["x"], outs: ["h1"], attrs: {} },
      { op: "relu", ins: ["h1"], outs: ["h2"], attrs: {} },
      { op: "relu", ins: ["h2"], outs: ["h3"], attrs: {} },
    ],
  });

test("生存を終えた区間は配り直され、総量が増えない", () => {
  // Three 8 B intermediates; intervals h1 [0,2), h2 [1,4), h3 [3,end). h2 overlaps h1, and behind
  // it would grow region 0 by 256 + 8 - 8 = 256, not less than its own 8 B, so it opens region 1.
  // h3 reads h2 and takes h1's dead range in region 0 (growth 0) = 2 regions × 8 = 16.
  assertEquals(workspaceBytes(reluChainModel()), 16);
});

/**
 * The relu chain behind a dead 260-byte slot, with `h1` and `h3` as graph outputs:
 * `x[1,65] → relu → b[1,65] → slice(0..2) → a[1,2] → relu → h1 → relu → h2 → relu → h3`.
 * `b` (260 B) ends its life before `h1` is allocated and only the slice reads it, so later slots can
 * be placed over its bytes. That region is larger than one 256-byte alignment step, which is what
 * lets a slot sit at the next aligned offset without growing the region by its full size.
 */
const pinnedNeighbourModel = (): PreparedModel =>
  openGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["relu", "slice"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 65] }],
    outputs: ["h1", "h3"],
    initializers: {},
    values: {
      b: { dtype: "f32", shape: [1, 65] },
      a: { dtype: "f32", shape: [1, 2] },
      h1: { dtype: "f32", shape: [1, 2] },
      h2: { dtype: "f32", shape: [1, 2] },
      h3: { dtype: "f32", shape: [1, 2] },
    },
    nodes: [
      { op: "relu", ins: ["x"], outs: ["b"], attrs: {} },
      { op: "slice", ins: ["b"], outs: ["a"], attrs: { dim: 1, start: 0, end: 2 } },
      { op: "relu", ins: ["a"], outs: ["h1"], attrs: {} },
      { op: "relu", ins: ["h1"], outs: ["h2"], attrs: {} },
      { op: "relu", ins: ["h2"], outs: ["h3"], attrs: {} },
    ],
  });

test("グラフ出力（pinned）は生存を終えず、後続は整列した次の区間へ回る", () => {
  // Slots by allocation clock, [start, end): b 260 B [0, 2), a 8 B [1, 4), h1 8 B [3, end) pinned,
  // h2 8 B [5, 7), h3 8 B [6, end) pinned. Read / write pairs (never in one region): b–a, a–h1,
  // h1–h2, h2–h3.
  // Order 1 (size, then start: b, a, h1, h2, h3): b opens region 0 (260). a conflicts with b and
  // opens region 1 (8). h1 goes to region 0 at offset 0 (b is dead: growth 0; region 1 holds a).
  // h2 conflicts with h1 and goes to region 1 over the dead a (offset 0, growth 0). h3 conflicts with
  // h2, so only region 0 is a candidate, where the pinned h1 still overlaps it at [0, 8): h3 takes
  // the next aligned offset 256 and grows the region 260 → 264 by 4 B, less than its own 8 B, so it
  // stays there = 264 + 8 = 272.
  // Order 2 (size, then lifetime: b, h1 5, a 3, h2 2, h3 2) puts every slot in the same region at
  // the same offset = 272; the tie keeps order 1.
  // Had h1 ended its life, h3 would take offset 0 and the total would be 260 + 8 = 268 (offset
  // alignment is the device's minStorageBufferOffsetAlignment, spec default 256 — ADR 0093
  // decision 1).
  assertEquals(workspaceBytes(pinnedNeighbourModel()), 272);
});

test("渡された maxBufferSize を中間の領域の上限に使う（実行相と同じ granted 値 — ADR 0093 決定 4）", () => {
  // The same graph as above. Without maxBufferSize the region limit is the larger of the binding
  // limit and the core default (256 MiB), so h3 sits at offset 256 of region 0 = 272. With
  // maxBufferSize 260, offset 256 + 8 = 264 exceeds the region limit; region 1 holds h2, which h3
  // reads, so h3 opens a third region = 260 + 8 + 8 = 276 (the same in both orders).
  const model = pinnedNeighbourModel();
  const scenario = (options: Parameters<PreparedModel["estimate"]>[0]) => {
    const report = model.estimate(options);
    assertEquals(
      report.scenarios.map(({ name }) => name),
      ["run"],
    );
    return report.scenarios[0];
  };
  assertEquals(scenario({ maxStorageBufferBindingSize: 260 }).workspaceBytes, 272);
  assertEquals(
    scenario({ maxStorageBufferBindingSize: 260, maxBufferSize: 260 }).workspaceBytes,
    276,
  );
});
