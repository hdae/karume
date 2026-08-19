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
import { acquireGpu } from "../src/gpu/device.ts";
import { createSession } from "../src/runtime/executor.ts";
import { estimateSessionMemory } from "../src/runtime/estimate.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
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

Deno.test("i4 適格は numel÷2 + group scale（linear の重みスロット限定）", () => {
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

Deno.test("i4 が linear 以外の重みスロット（embedding）で消費されると f32 展開へ回る", () => {
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
  // packed のままでは常駐できない（展開経路は linear のタイル読みにしか無い — ADR 0069 決定 5）
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
