// メモリ必要量 estimator（ADR 0070 決定 5）の門。
//
// estimator は「必要側」の数字しか出さない純関数なので、固定するのは**カテゴリごとの算式が
// 実装（executor / GenerationContext / RunArena）と同じ導出元から出ていること**に尽きる。
// 期待値は全て**手計算した定数**で置く — 実装と同じ式で組み直すと恒真化して、算式がずれた
// ときに 1 件も落ちない。
//
// 実 GPU 突合は 1 本だけ置く（アダプタ無しは明示 SKIP）。厳密一致を主張できるのは診断が
// 実測している 2 カテゴリ（圧縮常駐・展開）と state 容量で、中間ピークは**近似**なので
// 突合しない（融合が中間を消し、行ブロック分割が一時を足す — どちらも estimator の
// unaccounted 欄が認めている差）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ContainerError, type KarumeModel, openModel } from "../src/format/container.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { numel } from "../src/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { type ExecStep, planFusions } from "../src/runtime/fusion.ts";
import { countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import { derivePlanSlots, type StepOutput, type StepRecipe } from "../src/runtime/recipe.ts";
import { f32Bytes, type GraphJson, type TensorSpec } from "./helpers/format.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./helpers/f16.ts";
import { fill, graphModelBuffer } from "./helpers/graph.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";

const openGraph = (graph: GraphJson, tensors: readonly TensorSpec[] = []): KarumeModel =>
  openModel(graphModelBuffer(graph, tensors));

/** f16 のバイト列（値そのものは見ないので 0 で埋める — 見るのはバイト数だけ）。 */
const f16Zeros = (count: number): Uint8Array<ArrayBuffer> =>
  f16BytesFromBits(new Array(count).fill(f32ToF16Bits(0)));

const i32Bytes = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  new Uint8Array(Int32Array.from(values).buffer);

// ---------------------------------------------------------------------------
// 重みのカテゴリ分け
// ---------------------------------------------------------------------------

/**
 * 圧縮しない格納だけのグラフ（f32 の matmul 重み・f32 の embedding 表・i32 の添字）。
 * `T` を持つので io と中間ピークの束縛依存もここで見る。
 */
const plainGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "embedding"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["h", "g"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    emb: { tensor: "m.emb", storage: { dtype: "f32" } },
    idx: { tensor: "m.idx", storage: { dtype: "i32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    emb: { dtype: "f32", shape: [5, 3] },
    idx: { dtype: "i32", shape: [2] },
    h: { dtype: "f32", shape: ["T", 3] },
    g: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "embedding", ins: ["emb", "idx"], outs: ["g"], attrs: { padding_idx: -1 } },
  ],
});

const plainModel = (): KarumeModel =>
  openGraph(plainGraph(), [
    { name: "m.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(1)) },
    { name: "m.emb", dtype: "F32", shape: [5, 3], data: f32Bytes(new Array(15).fill(1)) },
    { name: "m.idx", dtype: "I32", shape: [2], data: i32Bytes([0, 1]) },
  ]);

Deno.test("圧縮しない格納（f32 / i32）は実バイトのまま非圧縮の欄に数える", () => {
  const estimate = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  // w 4×3×4=48 + emb 5×3×4=60 + idx 2×4=8（いずれも 4 バイト整列済み）
  assertEquals(estimate.uncompressedWeightBytes, 116);
  assertEquals(estimate.compressedWeightBytes, 0);
  assertEquals(estimate.expandedWeightBytes, 0);
  assertEquals(estimate.stateBytes, 0);
});

/**
 * 格納 `bf16` の initializer を 1 本だけ持つ形。IR の語彙としては valid だが
 * `RUNTIME_SUPPORT.storage` に無いので、`createSession` は必ず capability 不足で落ちる。
 */
const bf16Model = (): KarumeModel => {
  const graph = plainGraph();
  return openGraph(
    {
      ...graph,
      initializers: { ...graph.initializers, w: { tensor: "m.w", storage: { dtype: "bf16" } } },
    },
    [
      { name: "m.w", dtype: "BF16", shape: [4, 3], data: new Uint8Array(24) },
      { name: "m.emb", dtype: "F32", shape: [5, 3], data: f32Bytes(new Array(15).fill(1)) },
      { name: "m.idx", dtype: "I32", shape: [2], data: i32Bytes([0, 1]) },
    ],
  );
};

Deno.test("実構築が拒否する格納 dtype（bf16）に見積りを返さない", () => {
  assertThrows(
    () => estimateSessionMemory(bf16Model(), { bindings: { T: 7 } }),
    ContainerError,
    "capability 不足",
  );
});

Deno.test("io は入力バッファ + 出力 readback staging（0 要素は 4 バイト床）", () => {
  const model = plainModel();
  // x[7,4]=112 + h[7,3]=84 + g[2,3]=24
  assertEquals(estimateSessionMemory(model, { bindings: { T: 7 } }).ioBytes, 220);
  // T=0 では x も h も 0 要素 → 床の 4 バイトずつ。g は 24 のまま
  assertEquals(estimateSessionMemory(model, { bindings: { T: 0 } }).ioBytes, 32);
});

Deno.test("totalBytes は全カテゴリの合計", () => {
  const estimate = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  assertEquals(
    estimate.totalBytes,
    estimate.compressedWeightBytes + estimate.uncompressedWeightBytes +
      estimate.expandedWeightBytes + estimate.stateBytes +
      estimate.ioBytes + estimate.transientBytes,
  );
  // 0 + 116 + 0 + 0 + 220 + 108
  assertEquals(estimate.totalBytes, 444);
});

/** linear の重み（適格）と mul の被演算子（適格外）に同じ格納 dtype を置くグラフ。 */
const twoPathGraph = (
  storage: Record<string, unknown>,
  extra: Record<string, unknown>,
): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 3] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { ...storage } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
    g: { tensor: "m.g", storage: { ...extra } },
  },
  values: {
    w: { dtype: "f32", shape: [3, 3] },
    b: { dtype: "f32", shape: [3] },
    g: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: [2, 3] },
    y: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "g"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("f16 は適格なら 4 バイト切り上げで常駐・適格外なら f32 展開", () => {
  const model = openGraph(
    twoPathGraph({ dtype: "f16" }, { dtype: "f16" }),
    [
      // 整列降順（F32 → F16）で詰める（I4 / F32 の整列単位を跨がせない）
      { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
      { name: "m.w", dtype: "F16", shape: [3, 3], data: f16Zeros(9) },
      { name: "m.g", dtype: "F16", shape: [3], data: f16Zeros(3) },
    ],
  );
  const estimate = estimateSessionMemory(model);
  // w は numel 9 → 18 バイトを 4 バイトへ切り上げて 20
  assertEquals(estimate.compressedWeightBytes, 20);
  // b は f32 のまま 12（非圧縮の欄）
  assertEquals(estimate.uncompressedWeightBytes, 12);
  // g は mul の被演算子なので適格外 → 3×4
  assertEquals(estimate.expandedWeightBytes, 12);
});

Deno.test("i8 適格は numel の 4 バイト切り上げ + per-channel scale（奇数 numel）", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 5] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [3, 5] },
      b: { dtype: "f32", shape: [3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const model = openGraph(graph, [
    { name: "m.b", dtype: "F32", shape: [3], data: f32Bytes([0, 0, 0]) },
    { name: "m.s", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
    { name: "m.w", dtype: "I8", shape: [3, 5], data: new Uint8Array(15) },
  ]);
  const estimate = estimateSessionMemory(model);
  // w numel 15 → 16（切り上げ）+ scale 3×4=12
  assertEquals(estimate.compressedWeightBytes, 28);
  assertEquals(estimate.uncompressedWeightBytes, 12);
  assertEquals(estimate.expandedWeightBytes, 0);
  // x[2,5]=40 + y[2,3]=24
  assertEquals(estimate.ioBytes, 64);
});

Deno.test("i4 適格は numel÷2 + group scale（展開経路を持つ重みスロット限定）", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: 16 } },
      b: { tensor: "m.b", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [2, 16] },
      b: { dtype: "f32", shape: [2] },
      y: { dtype: "f32", shape: [2, 2] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  const model = openGraph(graph, [
    { name: "m.b", dtype: "F32", shape: [2], data: f32Bytes([0, 0]) },
    { name: "m.s", dtype: "F32", shape: [2, 1], data: f32Bytes([1, 1]) },
    { name: "m.w", dtype: "I4", shape: [2, 16], data: new Uint8Array(16) },
  ]);
  const estimate = estimateSessionMemory(model);
  // w numel 32 → 16 バイト + group scale 2×4=8
  assertEquals(estimate.compressedWeightBytes, 24);
  // b は f32 のまま 8
  assertEquals(estimate.uncompressedWeightBytes, 8);
  assertEquals(estimate.expandedWeightBytes, 0);
});

Deno.test("i4 の embedding も packed のまま常駐する（ADR 0069 決定 5 の embedding 追補）", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: 16 } },
    },
    values: {
      w: { dtype: "f32", shape: [4, 16] },
      y: { dtype: "f32", shape: [2, 16] },
    },
    nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  const model = openGraph(graph, [
    { name: "m.s", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "m.w", dtype: "I4", shape: [4, 16], data: new Uint8Array(32) },
  ]);
  const estimate = estimateSessionMemory(model);
  // 語彙表 numel 64 → 32 バイト + group scale 4×4=16（展開経路は embedding カーネルにもある）
  assertEquals(estimate.compressedWeightBytes, 48);
  assertEquals(estimate.uncompressedWeightBytes, 0);
  assertEquals(estimate.expandedWeightBytes, 0);
});

/** `conv1d(x, w, b)` 1 本のグラフ（w は i4 + rank 2 の group scale — 波 J-5b）。 */
const i4Conv1dGraph = (groups: number): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 32, 6] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: 16 } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    // 行長 = Cin/groups · K（groups == 1 なら 32·2 = 64 = g16 が 4 つ）
    w: { dtype: "f32", shape: [4, 32 / groups, 2] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 5] },
  },
  nodes: [{
    op: "conv1d",
    ins: ["x", "w", "b"],
    outs: ["y"],
    attrs: { stride: 1, padding: 0, dilation: 1, groups },
  }],
});

Deno.test("i4 の conv1d(groups==1) も packed のまま常駐する（ADR 0069 決定 5 の conv1d 追補）", () => {
  const model = openGraph(i4Conv1dGraph(1), [
    { name: "m.b", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    // scale は rank 非依存の rank 2 形 `[Cout, (Cin·K)/g]` = [4,4]（重みは rank 3）
    { name: "m.s", dtype: "F32", shape: [4, 4], data: f32Bytes(new Array(16).fill(1)) },
    { name: "m.w", dtype: "I4", shape: [4, 32, 2], data: new Uint8Array(128) },
  ]);
  const estimate = estimateSessionMemory(model);
  // w numel 256 → 128 バイト + group scale 16×4 = 64
  assertEquals(estimate.compressedWeightBytes, 192);
  // b は f32 のまま 16
  assertEquals(estimate.uncompressedWeightBytes, 16);
  assertEquals(estimate.expandedWeightBytes, 0);
});

Deno.test("i4 の conv1d(groups>1) は適格外で f32 展開へ回る", () => {
  // 直接カーネルに展開経路は無い（groups > 1 は igemm へ流れない — 波 J-5b）。
  const model = openGraph(i4Conv1dGraph(2), [
    { name: "m.b", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    { name: "m.s", dtype: "F32", shape: [4, 2], data: f32Bytes(new Array(8).fill(1)) },
    { name: "m.w", dtype: "I4", shape: [4, 16, 2], data: new Uint8Array(64) },
  ]);
  const estimate = estimateSessionMemory(model);
  assertEquals(estimate.compressedWeightBytes, 0);
  assertEquals(estimate.uncompressedWeightBytes, 16);
  // w numel 128 → f32 展開で 512 バイト
  assertEquals(estimate.expandedWeightBytes, 128 * 4);
});

Deno.test("グラフ出力になった i4 は適格外で f32 展開へ回る", () => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y", "w"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i4", scale: "m.s", group_size: 16 } },
    },
    values: {
      w: { dtype: "f32", shape: [4, 16] },
      y: { dtype: "f32", shape: [2, 16] },
    },
    nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  const model = openGraph(graph, [
    { name: "m.s", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "m.w", dtype: "I4", shape: [4, 16], data: new Uint8Array(32) },
  ]);
  const estimate = estimateSessionMemory(model);
  assertEquals(estimate.compressedWeightBytes, 0);
  assertEquals(estimate.uncompressedWeightBytes, 0);
  assertEquals(estimate.expandedWeightBytes, 64 * 4);
});

// ---------------------------------------------------------------------------
// state スロット
// ---------------------------------------------------------------------------

/** 記号容量 `C` の k と数値容量の v を持つグラフ（append は 1 スロット 1 本 MUST）。 */
const stateGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "state_append"] },
  symbols: ["T", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f32" } },
    chunk: { tensor: "m.chunk", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    chunk: { dtype: "f32", shape: [1, 2, 4, 4] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  states: {
    k: { dtype: "f32", shape: [1, 2, "C", 4] },
    v: { dtype: "f32", shape: [1, 2, 6, 4] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "state_append", ins: ["chunk"], outs: [], attrs: {}, states: { slot: "v" } },
  ],
});

const stateModel = (): KarumeModel =>
  openGraph(stateGraph(), [
    { name: "m.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0)) },
    {
      name: "m.chunk",
      dtype: "F32",
      shape: [1, 2, 4, 4],
      data: f32Bytes(new Array(32).fill(0)),
    },
  ]);

Deno.test("state は記号容量を解決したスロット合計 + 論理長 uniform", () => {
  const estimate = estimateSessionMemory(stateModel(), {
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  // k 1×2×8×4=64 要素 + v 1×2×6×4=48 要素 → (64+48)×4 = 448、論理長 uniform 8
  assertEquals(estimate.stateBytes, 456);
});

Deno.test("state の記号容量は generation.bindings でしか動かない", () => {
  const model = stateModel();
  const at = (capacity: number): number =>
    estimateSessionMemory(model, {
      bindings: { T: 2 },
      generation: { chunkLength: 4, bindings: { C: capacity } },
    }).stateBytes;
  // C だけが動く（v の 48 要素 + uniform 8 は固定）
  assertEquals(at(16), (1 * 2 * 16 * 4 + 48) * 4 + 8);
  assertEquals(at(1), (1 * 2 * 1 * 4 + 48) * 4 + 8);
});

Deno.test("generation を渡さなければ state は数えない", () => {
  assertEquals(estimateSessionMemory(plainModel(), { bindings: { T: 7 } }).stateBytes, 0);
});

// ---------------------------------------------------------------------------
// 中間（transient）ピーク
// ---------------------------------------------------------------------------

/** 単項の直列チェーン（`x → t1 → t2 → y`）。`outputs` を差し替えると pin の効きが見える。 */
const chainGraph = (outputs: readonly string[]): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: [...outputs],
  initializers: {},
  values: {
    t1: { dtype: "f32", shape: [64] },
    t2: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["t1"], attrs: {} },
    { op: "neg", ins: ["t1"], outs: ["t2"], attrs: {} },
    { op: "neg", ins: ["t2"], outs: ["y"], attrs: {} },
  ],
});

/** 3 本に分岐してから畳む形（同時生存が 4 本になる）。 */
const fanOutGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: ["y"],
  initializers: {},
  values: {
    t1: { dtype: "f32", shape: [64] },
    t2: { dtype: "f32", shape: [64] },
    t3: { dtype: "f32", shape: [64] },
    s: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["t1"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t2"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t3"], attrs: {} },
    { op: "add", ins: ["t1", "t2"], outs: ["s"], attrs: {} },
    { op: "add", ins: ["s", "t3"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("直列チェーンのピークは同時生存 2 本ぶん（消費が尽きた中間は解放される）", () => {
  // 64 要素 = 256 バイト。t2 を確保した時点が 2 本 = 512
  assertEquals(estimateSessionMemory(openGraph(chainGraph(["y"]))).transientBytes, 512);
});

Deno.test("幅広 fan-out のピークは同時生存 4 本ぶん（解放が効いても畳む直前で重なる）", () => {
  // s を確保した時点で t1 / t2 / t3 / s の 4 本 = 1024
  assertEquals(estimateSessionMemory(openGraph(fanOutGraph())).transientBytes, 1024);
});

Deno.test("グラフ出力は消費が尽きても pinned のまま（解放されない）", () => {
  // t1 を graph.outputs に載せると、t2 の消費後も t1 が居座って 3 本ぶん重なる
  assertEquals(estimateSessionMemory(openGraph(chainGraph(["t1", "y"]))).transientBytes, 768);
});

Deno.test("initializer とグラフ入力は中間ピークに数えない（重み・io 側の勘定）", () => {
  // matmul の h（T=7 → 84）と embedding の g（24）だけが中間。x / w / emb / idx は数えない
  assertEquals(estimateSessionMemory(plainModel(), { bindings: { T: 7 } }).transientBytes, 108);
});

Deno.test("slot の再利用はサイズクラスの厳密一致だけ（断片化ぶんを過小に数えない）", () => {
  // 8 → 12 → 4 バイトの 3 段。生存ピークは 20 だが、実行側の exact-size 再利用では
  // 解放済みの 8 が 12 にも 4 にも使えず slot が 3 本累積する（8 + 12 + 4 = 24）。
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [1, 2] }],
    outputs: ["y"],
    initializers: {
      w1: { tensor: "m.w1", storage: { dtype: "f32" } },
      w2: { tensor: "m.w2", storage: { dtype: "f32" } },
      w3: { tensor: "m.w3", storage: { dtype: "f32" } },
    },
    values: {
      w1: { dtype: "f32", shape: [2, 2] },
      w2: { dtype: "f32", shape: [2, 3] },
      w3: { dtype: "f32", shape: [3, 1] },
      t1: { dtype: "f32", shape: [1, 2] },
      t2: { dtype: "f32", shape: [1, 3] },
      y: { dtype: "f32", shape: [1, 1] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w1"], outs: ["t1"], attrs: {} },
      { op: "matmul", ins: ["t1", "w2"], outs: ["t2"], attrs: {} },
      { op: "matmul", ins: ["t2", "w3"], outs: ["y"], attrs: {} },
    ],
  };
  const model = openGraph(graph, [
    { name: "m.w1", dtype: "F32", shape: [2, 2], data: f32Bytes([1, 0, 0, 1]) },
    { name: "m.w2", dtype: "F32", shape: [2, 3], data: f32Bytes(new Array(6).fill(1)) },
    { name: "m.w3", dtype: "F32", shape: [3, 1], data: f32Bytes([1, 1, 1]) },
  ]);
  assertEquals(estimateSessionMemory(model).transientBytes, 24);
});

// ---------------------------------------------------------------------------
// 中間ピークの別名（reshape / 恒等 expand — ADR 0011）
// ---------------------------------------------------------------------------

/**
 * 別名の鎖（`reshape` → `reshape`）と、鎖の**根**が生きたままの消費。
 *
 * `h` は 2 度消費される（鎖の入口と末尾の `add`）ので、実行相では鎖のあいだ根の実体が
 * プールへ返らない。別名を独立した値として確保すると、この重なりぶんだけ slot が余計に生える。
 */
const aliasChainGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "add", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    r1: { dtype: "f32", shape: [8, 8] },
    r2: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["r1"], attrs: {} },
    { op: "reshape", ins: ["r1"], outs: ["r2"], attrs: {} },
    { op: "add", ins: ["h", "r2"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("別名の鎖は根の実体 1 本だけを数える（reshape → reshape）", () => {
  // 確保が出るのは `h`（256）と `y`（256）の 2 本きり = 512。鎖の 2 本を確保すると、`h` が
  // 生きている間は再利用できず 768 になる。
  assertEquals(estimateSessionMemory(openGraph(aliasChainGraph())).transientBytes, 512);
});

/** `neg` → `expand` の 2 本。出力 shape で恒等 / 非恒等を切り替える。 */
const aliasExpandGraph = (outShape: readonly number[]): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "expand"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [1, 4] },
    y: { dtype: "f32", shape: [...outShape] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "expand", ins: ["h"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("恒等 expand は確保を出さず、複製軸のある expand は実体化ぶんを数える", () => {
  // 恒等（[1,4] → [1,4]）は別名 = `h` の 16 バイト 1 本だけ。
  assertEquals(estimateSessionMemory(openGraph(aliasExpandGraph([1, 4]))).transientBytes, 16);
  // 複製軸あり（[1,4] → [3,4]）は strided 実体化コピーへ戻る = 16 + 48。
  assertEquals(estimateSessionMemory(openGraph(aliasExpandGraph([3, 4]))).transientBytes, 64);
});

/**
 * グラフ出力が**別名名義**の形。`p` は `h` の実体そのものなので、pin が効くのは根の側。
 * `h` の消費が尽きても根はプールへ戻らず、後続の `t` / `y` は新しい slot を掴む。
 */
const pinnedAliasGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "reshape"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [64] }],
  outputs: ["p", "y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    p: { dtype: "f32", shape: [8, 8] },
    t: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "reshape", ins: ["h"], outs: ["p"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["t"], attrs: {} },
    { op: "neg", ins: ["t"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("グラフ出力が別名名義でも pin が効くのは根の実体（プールへ返らない）", () => {
  // `h`（pin された根）+ `t` + `y` の 3 本 = 768。pin が根へ届かないと `h` がプールへ戻って
  // `t` に配り直され、512 に縮む（= readback 可能な実体を他の値と共有する誤り）。
  assertEquals(estimateSessionMemory(openGraph(pinnedAliasGraph())).transientBytes, 768);
});

// ---------------------------------------------------------------------------
// 実行計画（derivePlanSlots）との相互検証
// ---------------------------------------------------------------------------

/**
 * 判定に使う device の能力（WebGPU core 既定 — 128MiB / 65535）。行ブロック分割の枚数だけが
 * これを読む。
 */
const TEST_LIMITS = {
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
} as const;

/**
 * 素のノード列 → slot 導出用のレシピ列。GPU に触らずに recipe-builder の `#buildStep` の
 * **簿記面だけ**（確保仕様・`uses` / `pinned`・消費の延べ列）を写す — dispatch と一時は
 * slot 導出の別軸なので空で足りる。
 *
 * NOTE: 別名元は `#buildStep` と同じく入力の先頭。initializer を指す形だけは `#buildStep` が
 * `{ kind: "resident" }` を作るが、`derivePlanSlots` は env に載らない値をどちらも
 * 「slot 無し」と扱うので、slot 導出の観点では `{ kind: "value" }` と同値。
 */
const slotRecipes = (steps: readonly ExecStep[], graph: IrGraph): readonly StepRecipe[] => {
  const uses = countUses(graph);
  const outputNames = new Set(graph.outputs);
  return steps.map((step) => {
    if (step.kind !== "node") throw new Error("融合ステップの写しはここでは作らない");
    const outputs: readonly StepOutput[] = step.plan.outputs.map(({ name, shape }) => {
      const bookkeeping = { name, uses: uses.get(name) ?? 0, pinned: outputNames.has(name) };
      return step.aliasesInput
        ? {
          ...bookkeeping,
          kind: "alias" as const,
          source: { kind: "value" as const, name: step.plan.node.ins[0] },
        }
        : { ...bookkeeping, kind: "alloc" as const, byteLength: numel(shape) * 4 };
    });
    return {
      outputs,
      temps: [],
      dispatches: [],
      releases: step.plan.node.ins,
      writesState: false,
    };
  });
};

/**
 * 別名 3 種（鎖 / 恒等 expand / pin された別名）と非恒等 expand を 1 本に混ぜた形。**5 ルールの
 * どの先頭 op にも掛からない**ように組んである（`reshape` は upsample2x の先頭 op だが、続く
 * 並びが合わないので窓を掴まない — テスト本体が機械的に確認する）。
 */
const aliasMixGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "add", "reshape", "expand"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [64] },
    { name: "z", dtype: "f32", shape: [1, 4] },
    { name: "q", dtype: "f32", shape: [5] },
  ],
  outputs: ["v", "yv", "kv", "m", "p1", "p2"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [64] },
    r1: { dtype: "f32", shape: [8, 8] },
    r2: { dtype: "f32", shape: [64] },
    e1: { dtype: "f32", shape: [64] },
    g: { dtype: "f32", shape: [1, 4] },
    w: { dtype: "f32", shape: [3, 4] },
    v: { dtype: "f32", shape: [3, 4] },
    s: { dtype: "f32", shape: [64] },
    y: { dtype: "f32", shape: [64] },
    yv: { dtype: "f32", shape: [8, 8] },
    k: { dtype: "f32", shape: [5] },
    kv: { dtype: "f32", shape: [1, 5] },
    m: { dtype: "f32", shape: [5] },
    p1: { dtype: "f32", shape: [64] },
    p2: { dtype: "f32", shape: [64] },
  },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    // 別名の鎖（根は `h`）。
    { op: "reshape", ins: ["h"], outs: ["r1"], attrs: {} },
    { op: "reshape", ins: ["r1"], outs: ["r2"], attrs: {} },
    // 恒等 expand（同じ根を指す 3 本目の別名）。
    { op: "expand", ins: ["h"], outs: ["e1"], attrs: {} },
    { op: "neg", ins: ["z"], outs: ["g"], attrs: {} },
    // 複製軸あり = 実体化コピー（別名にならない）。
    { op: "expand", ins: ["g"], outs: ["w"], attrs: {} },
    { op: "neg", ins: ["w"], outs: ["v"], attrs: {} },
    { op: "add", ins: ["h", "r2"], outs: ["s"], attrs: {} },
    { op: "add", ins: ["s", "e1"], outs: ["y"], attrs: {} },
    // グラフ出力が別名名義（pin は根の `y` の実体へ効く）。
    { op: "reshape", ins: ["y"], outs: ["yv"], attrs: {} },
    // 同じ形をこのグラフで唯一のサイズクラス（20 バイト）で作り直す。pin が根へ届かないと
    // `k` の実体がプールへ戻って `m` に配り直され、slot が 1 本減る。
    { op: "neg", ins: ["q"], outs: ["k"], attrs: {} },
    { op: "reshape", ins: ["k"], outs: ["kv"], attrs: {} },
    { op: "neg", ins: ["q"], outs: ["m"], attrs: {} },
    // 末尾の 256 バイト 2 本は、鎖が返した slot を**ちょうど汲み尽くす**位置に置いてある。
    // 別名の消費が根へ合算されないと 1 本足りず、ここで新しい slot が生える。
    { op: "neg", ins: ["x"], outs: ["p1"], attrs: {} },
    { op: "neg", ins: ["x"], outs: ["p2"], attrs: {} },
  ],
});

/**
 * estimator の中間総量と、実行計画の slot 表（`derivePlanSlots`）の総バイト数を突き合わせる。
 *
 * 融合が 1 本も掛からない形でだけ主張できる**厳密一致** — 融合が掴む形では実行側が中間を
 * 消し、行ブロック分割が一時を足すので、estimator の `unaccounted` が認めている差になる。
 * 別名の扱いが 2 実装で割れると、ここが例外なしで割れる（それが起票 R6V-2 の姿）。
 */
Deno.test("融合が掛からない形では estimator の中間総量と slot 表の総バイトが一致する", () => {
  const model = openGraph(aliasMixGraph());
  const plan = planGraph(model.graph, {});
  const fusion = planFusions(plan.nodes, {
    useCounts: countUses(model.graph),
    outputNames: new Set(model.graph.outputs),
    limits: TEST_LIMITS,
  });
  // 前提の確認: 融合が 1 本でも掛かると「融合前の列を歩く」estimator の勘定と比べられない。
  assertEquals(fusion.steps.every((step) => step.kind === "node"), true, "融合が掛かっている");
  assertEquals(fusion.counts.identityExpand, 1);

  const slots = derivePlanSlots(slotRecipes(fusion.steps, model.graph));
  const slotTotal = slots.bytes.reduce((sum, size) => sum + size, 0);
  // 手計算: h 256 + g 16 + w 48 + v 48 + s 256 + y 256 + k 20 + m 20 = 920
  // （別名 r1 / r2 / e1 / yv / kv は 1 本も生やさない）。
  assertEquals(slots.bytes, [256, 16, 48, 48, 256, 256, 20, 20]);
  assertEquals(slotTotal, 920);
  assertEquals(estimateSessionMemory(model).transientBytes, slotTotal);
});

// ---------------------------------------------------------------------------
// fail loudly
// ---------------------------------------------------------------------------

Deno.test("未束縛の記号次元は fail loudly（黙って 0 で埋めない）", () => {
  assertThrows(
    () => estimateSessionMemory(plainModel()),
    ExecutionError,
    "シンボル 'T' が束縛されていない",
  );
});

Deno.test("states 専用記号を options.bindings で受けない（束縛点は generation の側）", () => {
  assertThrows(
    () => estimateSessionMemory(stateModel(), { bindings: { T: 2, C: 8 } }),
    ExecutionError,
    "states 専用記号",
  );
});

Deno.test("states 形グラフを generation 無しで見積らない（context が要る旨で落ちる）", () => {
  assertThrows(
    () => estimateSessionMemory(stateModel(), { bindings: { T: 2 } }),
    Error,
    "GenerationContext",
  );
});

Deno.test("states 宣言の無いグラフに generation を渡すと fail loudly", () => {
  assertThrows(
    () =>
      estimateSessionMemory(plainModel(), {
        bindings: { T: 7 },
        generation: { chunkLength: 4 },
      }),
    ExecutionError,
    "states 宣言を持たない",
  );
});

Deno.test("実構築が拒否する chunkLength に見積りを返さない", () => {
  assertThrows(
    () =>
      estimateSessionMemory(stateModel(), {
        bindings: { T: 2 },
        generation: { chunkLength: 0, bindings: { C: 8 } },
      }),
    ExecutionError,
    "chunkLength 0 が",
  );
});

Deno.test("states と入力の両方に現れる記号の食い違いは fail loudly（run と同じ門）", () => {
  // v の容量を T にして共有記号を作る（k は C のまま — states 専用記号の分担は不変）
  const graph = stateGraph();
  graph.states = {
    ...graph.states,
    v: { dtype: "f32", shape: [1, 2, "T", 4] },
  };
  const model = openGraph(graph, [
    { name: "m.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0)) },
    {
      name: "m.chunk",
      dtype: "F32",
      shape: [1, 2, 4, 4],
      data: f32Bytes(new Array(32).fill(0)),
    },
  ]);
  assertThrows(
    () =>
      estimateSessionMemory(model, {
        bindings: { T: 2 },
        generation: { chunkLength: 4, bindings: { C: 8, T: 3 } },
      }),
    ExecutionError,
    "記号 'T' が",
  );
});

Deno.test("グラフに無い記号の束縛は fail loudly", () => {
  assertThrows(
    () => estimateSessionMemory(plainModel(), { bindings: { T: 7, Z: 3 } }),
    ExecutionError,
    "'Z'",
  );
});

// ---------------------------------------------------------------------------
// unaccounted
// ---------------------------------------------------------------------------

Deno.test("unaccounted は勘定に入っていないものを明示する（見積りは絶対保証ではない）", () => {
  const { unaccounted } = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  const joined = unaccounted.join("\n");
  assert(joined.includes("融合"), joined);
  assert(joined.includes("params"), joined);
  assert(joined.includes("writeBuffer"), joined);
});

// ---------------------------------------------------------------------------
// 実 GPU 突合
// ---------------------------------------------------------------------------

/**
 * 3 つの欄が全て非 0 になるモデル — i8 の embedding 表（適格 = 圧縮のまま常駐）・f16 の
 * mul 被演算子（適格外 = f32 展開）・f32 の add 被演算子（非圧縮のまま常駐）。
 *
 * 圧縮 / 展開の 2 欄は診断 `storage` と厳密一致を主張でき、f32 は診断に現れない
 * （`uncompressedWeightBytes` の欄を分けている理由そのもの）ので手計算定数と突合する。
 */
const gpuWeightModel = (): KarumeModel => {
  const table = fill([5, 3], (i) => (i % 7) - 3);
  const quantized = quantizeI8(table.data, [5, 3], 0);
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["embedding", "mul", "add"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y"],
    initializers: {
      w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
      g: { tensor: "m.g", storage: { dtype: "f16" } },
      c: { tensor: "m.c", storage: { dtype: "f32" } },
    },
    values: {
      w: { dtype: "f32", shape: [5, 3] },
      g: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      e: { dtype: "f32", shape: [2, 3] },
      h: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "embedding", ins: ["w", "ids"], outs: ["e"], attrs: { padding_idx: -1 } },
      { op: "mul", ins: ["e", "g"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "c"], outs: ["y"], attrs: {} },
    ],
  };
  return openGraph(graph, [
    { name: "m.s", dtype: "F32", shape: [5, 1], data: f32Bytes([...quantized.scale]) },
    { name: "m.c", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
    { name: "m.g", dtype: "F16", shape: [3], data: f16Zeros(3) },
    { name: "m.w", dtype: "I8", shape: [5, 3], data: quantized.bytes },
  ]);
};

Deno.test({
  name: "estimator の重み・state が実測診断と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  fn: async () => {
    const gpu = await acquireGpu();
    try {
      const weightModel = gpuWeightModel();
      const estimate = estimateSessionMemory(weightModel);
      const session = await createSession(gpu, weightModel);
      try {
        await session.run({ ids: fill([2], (i) => i, "i32") });
        const storage = session.diagnostics().storage;
        assertEquals(estimate.compressedWeightBytes, storage.residentCompressedBytes);
        assertEquals(estimate.expandedWeightBytes, storage.hostExpandedBytes);
        // f32 の c は診断に現れない（ADR 0006 の storage 診断は低精度格納だけ）— 手計算 3×4
        assertEquals(estimate.uncompressedWeightBytes, 12);
      } finally {
        await session.dispose();
      }

      const model = stateModel();
      const generation = { chunkLength: 4, bindings: { C: 8 } };
      const stateEstimate = estimateSessionMemory(model, { bindings: { T: 2 }, generation });
      const stateSession = await createSession(gpu, model);
      try {
        const context = await stateSession.createGenerationContext(generation);
        assertEquals(
          stateEstimate.stateBytes,
          stateSession.diagnostics().stateBacking.residentBytes,
        );
        await context.dispose();
      } finally {
        await stateSession.dispose();
      }
      // transientBytes は突合しない — 融合が中間を消し、行ブロック分割が一時を足すので
      // 実測（planBacking.residentBytes / lastRun.peakTransientBytes）とは原理的にずれる
      // （estimator の unaccounted 欄が認めている差そのもの）。
    } finally {
      gpu.destroy();
    }
  },
});
