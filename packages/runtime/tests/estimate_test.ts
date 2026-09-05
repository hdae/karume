// メモリ必要量 estimator（ADR 0070 決定 5）の門。
//
// estimator は「必要側」の数字しか出さない純関数なので、固定するのは**カテゴリごとの算式が
// 実装（executor / GenerationContext / RunArena）と同じ導出元から出ていること**に尽きる。
// 期待値は全て**手計算した定数**で置く — 実装と同じ式で組み直すと恒真化して、算式がずれた
// ときに 1 件も落ちない。
//
// 報告は常駐（`resident`）+ run の形ごと（`scenarios`）の 2 段。generation を渡さない見積りは
// `"run"` の 1 本で、渡すと ADR 0066 決定 4 の実行 2 形（`"prefill"` / `"decode"`）が別々に
// 並ぶ。`peakAccountedBytes` は常駐 + シナリオ側の**最大**（和ではない — slot backing が
// 同時に 1 本だから）。
//
// 実 GPU 突合は 1 本だけ置く（アダプタ無しは明示 SKIP）。厳密一致を主張できるのは診断が
// 実測している 2 カテゴリ（圧縮常駐・展開）と state 容量で、中間ピークは**近似**なので
// 突合しない（融合が中間を消し、行ブロック分割が一時を足す — どちらも estimator の
// unaccounted 欄が認めている差）。

import { assert, assertEquals, assertThrows } from "@std/assert";
import { ContainerError, type KarumeModel, openModel } from "../src/format/container.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { numel, OpContractError } from "../src/ops.ts";
import { createSession } from "../src/runtime/executor.ts";
import {
  type AdmissionReport,
  type AdmissionScenario,
  estimateSessionMemory,
} from "../src/runtime/estimate.ts";
import { type ExecStep, planFusions } from "../src/runtime/fusion.ts";
import { countUses, ExecutionError, planGraph } from "../src/runtime/plan.ts";
import { derivePlanSlots, type StepOutput, type StepRecipe } from "../src/runtime/recipe.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";
import { planWeightBuffers, planWeightResidency } from "../src/runtime/weight-residency.ts";
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

/**
 * シナリオが `"run"` の 1 本だけであることを確かめて、その 1 本を返す。
 *
 * generation を渡さない見積りの io / workspace を引く唯一の口 — 「1 本だけ」の主張を
 * 引くたびに一緒に押さえる（2 本に増えたら黙って先頭だけ見る形にしない）。
 */
const runScenario = (report: AdmissionReport): AdmissionScenario => {
  assertEquals(report.scenarios.map((scenario) => scenario.name), ["run"]);
  return report.scenarios[0];
};

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
  const { resident } = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  // w 4×3×4=48 + emb 5×3×4=60 + idx 2×4=8（いずれも 4 バイト整列済み）
  assertEquals(resident.weights.uncompressedBytes, 116);
  assertEquals(resident.weights.compressedBytes, 0);
  assertEquals(resident.weights.expandedBytes, 0);
  // 3 欄の和が weights.totalBytes
  assertEquals(resident.weights.totalBytes, 116);
  assertEquals(resident.stateBytes, 0);
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
  assertEquals(runScenario(estimateSessionMemory(model, { bindings: { T: 7 } })).ioBytes, 220);
  // T=0 では x も h も 0 要素 → 床の 4 バイトずつ。g は 24 のまま
  assertEquals(runScenario(estimateSessionMemory(model, { bindings: { T: 0 } })).ioBytes, 32);
});

Deno.test("peakAccountedBytes は常駐の総和 + シナリオ側の最大", () => {
  const report = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  const scenario = runScenario(report);
  assertEquals(
    report.peakAccountedBytes,
    report.resident.weights.totalBytes + report.resident.stateBytes +
      scenario.ioBytes + scenario.workspaceBytes,
  );
  // 116（非圧縮常駐）+ 0（state）+ 220（io）+ 108（workspace）
  assertEquals(report.peakAccountedBytes, 444);
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
  const { weights } = estimateSessionMemory(model).resident;
  // w は numel 9 → 18 バイトを 4 バイトへ切り上げて 20
  assertEquals(weights.compressedBytes, 20);
  // b は f32 のまま 12（非圧縮の欄）
  assertEquals(weights.uncompressedBytes, 12);
  // g は mul の被演算子なので適格外 → 3×4
  assertEquals(weights.expandedBytes, 12);
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
  const report = estimateSessionMemory(model);
  // w numel 15 → 16（切り上げ）+ scale 3×4=12
  assertEquals(report.resident.weights.compressedBytes, 28);
  assertEquals(report.resident.weights.uncompressedBytes, 12);
  assertEquals(report.resident.weights.expandedBytes, 0);
  // x[2,5]=40 + y[2,3]=24
  assertEquals(runScenario(report).ioBytes, 64);
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
  const { weights } = estimateSessionMemory(model).resident;
  // w numel 32 → 16 バイト + group scale 2×4=8
  assertEquals(weights.compressedBytes, 24);
  // b は f32 のまま 8
  assertEquals(weights.uncompressedBytes, 8);
  assertEquals(weights.expandedBytes, 0);
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
  const { weights } = estimateSessionMemory(model).resident;
  // 語彙表 numel 64 → 32 バイト + group scale 4×4=16（展開経路は embedding カーネルにもある）
  assertEquals(weights.compressedBytes, 48);
  assertEquals(weights.uncompressedBytes, 0);
  assertEquals(weights.expandedBytes, 0);
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
  const { weights } = estimateSessionMemory(model).resident;
  // w numel 256 → 128 バイト + group scale 16×4 = 64
  assertEquals(weights.compressedBytes, 192);
  // b は f32 のまま 16
  assertEquals(weights.uncompressedBytes, 16);
  assertEquals(weights.expandedBytes, 0);
});

Deno.test("i4 の conv1d(groups>1) は適格外で f32 展開へ回る", () => {
  // 直接カーネルに展開経路は無い（groups > 1 は igemm へ流れない — 波 J-5b）。
  const model = openGraph(i4Conv1dGraph(2), [
    { name: "m.b", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    { name: "m.s", dtype: "F32", shape: [4, 2], data: f32Bytes(new Array(8).fill(1)) },
    { name: "m.w", dtype: "I4", shape: [4, 16, 2], data: new Uint8Array(64) },
  ]);
  const { weights } = estimateSessionMemory(model).resident;
  assertEquals(weights.compressedBytes, 0);
  assertEquals(weights.uncompressedBytes, 16);
  // w numel 128 → f32 展開で 512 バイト
  assertEquals(weights.expandedBytes, 128 * 4);
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
  const { weights } = estimateSessionMemory(model).resident;
  assertEquals(weights.compressedBytes, 0);
  assertEquals(weights.uncompressedBytes, 0);
  assertEquals(weights.expandedBytes, 64 * 4);
});

/**
 * 5 席（`raw` / `f16` / `i8` / `i4` / `expanded`）を**1 グラフに同居**させた形。
 *
 * linear 3 段で圧縮 3 席を重みスロットに置き（bias 3 本が `raw`）、最後の `mul` の被演算子に
 * f16 を置いて `expanded` を作る。上の各テストは席ごとに別グラフで欄を見ているので、
 * 「`planWeightBuffers` の全要素が 3 欄のどれかにちょうど 1 度数えられる」= 席を増やして
 * `weightEstimate` の分岐を直し忘れたら赤くなる、という網羅の主張は誰も見ていない。
 */
const allSeatsModel = (): KarumeModel => {
  const graph: GraphJson = {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["linear", "mul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
    outputs: ["y"],
    initializers: {
      w4: { tensor: "m.w4", storage: { dtype: "i4", scale: "m.s4", group_size: 16 } },
      b4: { tensor: "m.b4", storage: { dtype: "f32" } },
      w8: { tensor: "m.w8", storage: { dtype: "i8", scale: "m.s8" } },
      b8: { tensor: "m.b8", storage: { dtype: "f32" } },
      w16: { tensor: "m.w16", storage: { dtype: "f16" } },
      b16: { tensor: "m.b16", storage: { dtype: "f32" } },
      g: { tensor: "m.g", storage: { dtype: "f16" } },
    },
    values: {
      w4: { dtype: "f32", shape: [4, 16] },
      b4: { dtype: "f32", shape: [4] },
      w8: { dtype: "f32", shape: [4, 4] },
      b8: { dtype: "f32", shape: [4] },
      w16: { dtype: "f32", shape: [4, 4] },
      b16: { dtype: "f32", shape: [4] },
      g: { dtype: "f32", shape: [4] },
      h: { dtype: "f32", shape: [2, 4] },
      i: { dtype: "f32", shape: [2, 4] },
      j: { dtype: "f32", shape: [2, 4] },
      y: { dtype: "f32", shape: [2, 4] },
    },
    nodes: [
      { op: "linear", ins: ["x", "w4", "b4"], outs: ["h"], attrs: {} },
      { op: "linear", ins: ["h", "w8", "b8"], outs: ["i"], attrs: {} },
      { op: "linear", ins: ["i", "w16", "b16"], outs: ["j"], attrs: {} },
      // 重みスロットではない消費なので、この f16 だけが f32 展開席へ回る。
      { op: "mul", ins: ["j", "g"], outs: ["y"], attrs: {} },
    ],
  };
  // 整列降順（F32 → F16 → I8 → I4）で詰める。
  return openGraph(graph, [
    { name: "m.b4", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    { name: "m.b8", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    { name: "m.b16", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
    { name: "m.s4", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "m.s8", dtype: "F32", shape: [4, 1], data: f32Bytes([1, 1, 1, 1]) },
    { name: "m.w16", dtype: "F16", shape: [4, 4], data: f16Zeros(16) },
    { name: "m.g", dtype: "F16", shape: [4], data: f16Zeros(4) },
    { name: "m.w8", dtype: "I8", shape: [4, 4], data: new Uint8Array(16) },
    { name: "m.w4", dtype: "I4", shape: [4, 16], data: new Uint8Array(32) },
  ]);
};

Deno.test("5 席同居のグラフで、常駐バッファは 3 欄のどれかにちょうど 1 度数えられる", () => {
  const model = allSeatsModel();
  const { weights } = estimateSessionMemory(model).resident;

  // 手計算: i4 payload 32 + i4 scale 16 + i8 payload 16 + i8 scale 16 + f16 payload 32
  assertEquals(weights.compressedBytes, 112);
  // bias 3 本（f32 [4]）
  assertEquals(weights.uncompressedBytes, 48);
  // mul の被演算子 f16 [4] を f32 展開した 16（整列前の宣言バイト数で数える欄）
  assertEquals(weights.expandedBytes, 16);
  assertEquals(weights.totalBytes, 176);

  // 網羅の門: 席の分岐（weightEstimate の switch）を通らなかったバッファが 1 本でもあれば、
  // 総和がプランナ側の総和と合わなくなる。席を増やして分岐を直し忘れると赤くなる。
  const buffers = planWeightBuffers(planWeightResidency(model.graph));
  assertEquals(new Set(buffers.map((buffer) => buffer.seat)).size, 5, "5 席が同居していない");
  assertEquals(
    weights.totalBytes,
    buffers.reduce(
      (sum, buffer) =>
        sum +
        (buffer.seat === "expanded" && buffer.kind === "payload"
          ? buffer.declaredBytes
          : buffer.byteLength),
      0,
    ),
  );
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
  const report = estimateSessionMemory(stateModel(), {
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  // k 1×2×8×4=64 要素 + v 1×2×6×4=48 要素 → (64+48)×4 = 448、論理長 uniform 8
  assertEquals(report.resident.stateBytes, 456);
});

Deno.test("state の記号容量は generation.bindings でしか動かない", () => {
  const model = stateModel();
  const at = (capacity: number): number =>
    estimateSessionMemory(model, {
      bindings: { T: 2 },
      generation: { chunkLength: 4, bindings: { C: capacity } },
    }).resident.stateBytes;
  // C だけが動く（v の 48 要素 + uniform 8 は固定）
  assertEquals(at(16), (1 * 2 * 16 * 4 + 48) * 4 + 8);
  assertEquals(at(1), (1 * 2 * 1 * 4 + 48) * 4 + 8);
});

Deno.test("generation を渡さなければ state は数えない", () => {
  const report = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  assertEquals(report.resident.stateBytes, 0);
});

// ---------------------------------------------------------------------------
// シナリオ（ADR 0066 決定 4 の実行 2 形）
// ---------------------------------------------------------------------------

/**
 * 物理 chunk 行が**記号**で載っているグラフ（`state_append` の入力 `h` が `[1,2,M,4]`）。
 *
 * `stateGraph` の `chunk` は数値次元なので、prefill / decode で shape が動くのはこちら。
 * `M` は入力にも現れる（states 専用記号ではない）ので、束縛点が chunkLength 側であることも
 * この形でだけ観測できる。
 */
const chunkSymbolGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["neg", "state_append"] },
  symbols: ["M", "C"],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 2, "M", 4] }],
  outputs: ["y"],
  initializers: {},
  values: {
    h: { dtype: "f32", shape: [1, 2, "M", 4] },
    y: { dtype: "f32", shape: [1, 2, "M", 4] },
  },
  states: { k: { dtype: "f32", shape: [1, 2, "C", 4] } },
  nodes: [
    { op: "neg", ins: ["x"], outs: ["h"], attrs: {} },
    { op: "state_append", ins: ["h"], outs: [], attrs: {}, states: { slot: "k" } },
    { op: "neg", ins: ["h"], outs: ["y"], attrs: {} },
  ],
});

Deno.test("generation ありは prefill / decode の 2 本を自動導出する（chunk 記号だけが動く）", () => {
  const report = estimateSessionMemory(openGraph(chunkSymbolGraph()), {
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  // prefill は M=4: x / y とも 1×2×4×4=32 要素 → io 128+128、中間は h と y の 128 ずつ
  // （h は append と 2 本目の neg に消費されるので、y の確保時点でプールへ返っていない）。
  // decode は M=1: 全て 1/4 の 8 要素ぶん。
  assertEquals(report.scenarios, [
    { name: "prefill", ioBytes: 256, workspaceBytes: 256 },
    { name: "decode", ioBytes: 64, workspaceBytes: 64 },
  ]);
  // k 1×2×8×4=64 要素 ×4 = 256 + 論理長 uniform 8
  assertEquals(report.resident.stateBytes, 264);
  assertEquals(report.resident.weights.totalBytes, 0);
  // 常駐 264 + max(512, 128) = 776。和で足すと 904 になる（backing は同時に 1 本）。
  assertEquals(report.peakAccountedBytes, 776);
});

Deno.test("chunk 行が数値次元のグラフでは prefill と decode が同じ数字になる", () => {
  // `stateGraph` の append 入力は initializer の `[1,2,4,4]`。記号で動かない形の decode step は
  // 物理 chunk 行を prefill と同じまま queryLength=1 で回る（assertGenerationRun は
  // rows === chunkLength を decode でも許す）ので、2 本が一致するのが正しい。
  const report = estimateSessionMemory(stateModel(), {
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  const [prefill, decode] = report.scenarios;
  assertEquals(prefill.name, "prefill");
  assertEquals(decode.name, "decode");
  assertEquals(prefill.ioBytes, decode.ioBytes);
  assertEquals(prefill.workspaceBytes, decode.workspaceBytes);
});

Deno.test("generation なしの見積りはシナリオ 1 本（io / 中間の導出は従来と同じ）", () => {
  const report = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  // x[7,4]=112 + h[7,3]=84 + g[2,3]=24 = 220 / 中間は h 84 + g 24 = 108（改名前と同値）
  assertEquals(report.scenarios, [{ name: "run", ioBytes: 220, workspaceBytes: 108 }]);
});

Deno.test("chunk 行の記号を options.bindings で受けない（束縛点は chunkLength と decode の 1）", () => {
  assertThrows(
    () =>
      estimateSessionMemory(openGraph(chunkSymbolGraph()), {
        bindings: { M: 4 },
        generation: { chunkLength: 4, bindings: { C: 8 } },
      }),
    ExecutionError,
    "物理 chunk 行 M の記号",
  );
});

Deno.test("chunk 記号を持つグラフを generation 無しで見積らない", () => {
  assertThrows(
    () => estimateSessionMemory(openGraph(chunkSymbolGraph())),
    ExecutionError,
    "options.generation.chunkLength で",
  );
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
  assertEquals(
    runScenario(estimateSessionMemory(openGraph(chainGraph(["y"])))).workspaceBytes,
    512,
  );
});

Deno.test("幅広 fan-out のピークは同時生存 4 本ぶん（解放が効いても畳む直前で重なる）", () => {
  // s を確保した時点で t1 / t2 / t3 / s の 4 本 = 1024
  assertEquals(runScenario(estimateSessionMemory(openGraph(fanOutGraph()))).workspaceBytes, 1024);
});

Deno.test("グラフ出力は消費が尽きても pinned のまま（解放されない）", () => {
  // t1 を graph.outputs に載せると、t2 の消費後も t1 が居座って 3 本ぶん重なる
  const report = estimateSessionMemory(openGraph(chainGraph(["t1", "y"])));
  assertEquals(runScenario(report).workspaceBytes, 768);
});

Deno.test("initializer とグラフ入力は中間ピークに数えない（重み・io 側の勘定）", () => {
  // matmul の h（T=7 → 84）と embedding の g（24）だけが中間。x / w / emb / idx は数えない
  const report = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  assertEquals(runScenario(report).workspaceBytes, 108);
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
  assertEquals(runScenario(estimateSessionMemory(model)).workspaceBytes, 24);
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
  const report = estimateSessionMemory(openGraph(aliasChainGraph()));
  assertEquals(runScenario(report).workspaceBytes, 512);
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
  const workspaceOf = (outShape: readonly number[]): number =>
    runScenario(estimateSessionMemory(openGraph(aliasExpandGraph(outShape)))).workspaceBytes;
  // 恒等（[1,4] → [1,4]）は別名 = `h` の 16 バイト 1 本だけ。
  assertEquals(workspaceOf([1, 4]), 16);
  // 複製軸あり（[1,4] → [3,4]）は strided 実体化コピーへ戻る = 16 + 48。
  assertEquals(workspaceOf([3, 4]), 64);
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
  const report = estimateSessionMemory(openGraph(pinnedAliasGraph()));
  assertEquals(runScenario(report).workspaceBytes, 768);
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
  assertEquals(runScenario(estimateSessionMemory(model)).workspaceBytes, slotTotal);
});

// ---------------------------------------------------------------------------
// states 形 attention のノード内一時（スコア S / 行統計）
// ---------------------------------------------------------------------------

/**
 * states 形 attention 1 本 + `state_append` 2 本（gpu_state_execution_test の実行形と同じ姿の
 * 最小版）。`B=1` / `Hkv=2`（GQA）/ `D=8` で、`M` が物理 chunk 行・`C` がスロット容量。
 *
 * `window` を渡すと sliding 変種（読み書き同式 MUST — attention と append の両方に載せる）。
 * `heads`（既定 4）は **S の 1 行バイト数 `H·colCap·4` だけを動かす**軸 — state スロットは
 * `Hkv·C·D·4` で H に依らないので、H を上げると「スロットは束縛上限に収まるが S の 1 行は
 * 収まらない」形（= 行ブロックが複数枚に割れる形）を絞った device 上で作れる。
 */
const stateAttentionGraph = (window?: number, heads = 4): GraphJson => {
  const windowAttrs: Record<string, number> = window === undefined ? {} : { window };
  const append = (name: string, slot: string) => ({
    op: "state_append",
    ins: [name],
    outs: [] as string[],
    attrs: { ...windowAttrs },
    states: { slot },
  });
  return {
    format: "karume-ir",
    version: 1,
    requires: { ops: ["attention", "state_append"] },
    symbols: ["M", "C"],
    inputs: [
      { name: "q", dtype: "f32", shape: [1, heads, "M", 8] },
      { name: "k", dtype: "f32", shape: [1, 2, "M", 8] },
      { name: "v", dtype: "f32", shape: [1, 2, "M", 8] },
    ],
    outputs: ["o"],
    initializers: {},
    values: { o: { dtype: "f32", shape: [1, heads, "M", 8] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 2, "C", 8] },
      vslot: { dtype: "f32", shape: [1, 2, "C", 8] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: 0.5, ...windowAttrs },
        states: { k: "kslot", v: "vslot" },
      },
      append("k", "kslot"),
      append("v", "vslot"),
    ],
  };
};

/** ストレージ束縛の上限が効かない大きさ（行ブロックが常に 1 枚になる）。 */
const WIDE_LIMIT = 1 << 20;

const stateAttentionReport = (options: {
  readonly capacity: number;
  readonly chunkLength: number;
  readonly window?: number;
  readonly limit?: number;
}): AdmissionReport =>
  estimateSessionMemory(openGraph(stateAttentionGraph(options.window)), {
    generation: { chunkLength: options.chunkLength, bindings: { C: options.capacity } },
    maxStorageBufferBindingSize: options.limit ?? WIDE_LIMIT,
  });

/** prefill / decode の 2 本を名前つきで引く（並びの前提もここで一緒に押さえる）。 */
const bothScenarios = (
  report: AdmissionReport,
): { readonly prefill: AdmissionScenario; readonly decode: AdmissionScenario } => {
  assertEquals(report.scenarios.map((scenario) => scenario.name), ["prefill", "decode"]);
  return { prefill: report.scenarios[0], decode: report.scenarios[1] };
};

Deno.test("states 形 attention の S / 行統計が中間に乗る（full 変種・行ブロック 1 枚）", () => {
  const { prefill, decode } = bothScenarios(
    stateAttentionReport({ capacity: 16, chunkLength: 4 }),
  );
  // B·H = 4・列容量 = C = 16。prefill（M=4）は 1 行 4×16×4=256B で上限に余裕があり 1 枚。
  //   出力 o [1,4,4,8]=128 要素 → 512 / S = 4×4×16×4 = 1024 / 行統計 = 4×4×2×4 = 128
  assertEquals(prefill.workspaceBytes, 512 + 1024 + 128);
  // io は q 512 + k 256 + v 256 + o 512
  assertEquals(prefill.ioBytes, 1536);
  // decode（M=1）: o 128 / S = 4×1×16×4 = 256 / 行統計 = 4×1×2×4 = 32
  assertEquals(decode.workspaceBytes, 128 + 256 + 32);
  assertEquals(decode.ioBytes, 384);
});

Deno.test("S は capacity に比例して増える（full 変種は列容量 = スロット容量）", () => {
  const at = (capacity: number): number =>
    bothScenarios(stateAttentionReport({ capacity, chunkLength: 4 })).prefill.workspaceBytes;
  // 出力 512 + 行統計 128 は動かず、S だけが C に比例する（1024 → 2048 → 4096）。
  assertEquals(at(16), 512 + 1024 + 128);
  assertEquals(at(32), 512 + 2048 + 128);
  assertEquals(at(64), 512 + 4096 + 128);
});

Deno.test("S と行統計は chunkLength に比例して増える（prefill 側だけ・decode は M=1 固定）", () => {
  const at = (chunkLength: number): AdmissionScenario =>
    bothScenarios(stateAttentionReport({ capacity: 32, chunkLength })).prefill;
  // M=4: o 512 + S 4×4×32×4=2048 + 行統計 4×4×2×4=128
  assertEquals(at(4).workspaceBytes, 512 + 2048 + 128);
  // M=8: 3 項とも 2 倍（列容量は C のままなので S も行数ぶんだけ伸びる）
  assertEquals(at(8).workspaceBytes, 1024 + 4096 + 256);
  // decode は chunkLength に依らず M=1
  const decode = (chunkLength: number): number =>
    bothScenarios(stateAttentionReport({ capacity: 32, chunkLength })).decode.workspaceBytes;
  assertEquals(decode(4), 128 + 512 + 32);
  assertEquals(decode(8), 128 + 512 + 32);
});

Deno.test("sliding 変種の列容量は W−1+M（capacity を上げても S は動かない）", () => {
  const at = (capacity: number): AdmissionScenario =>
    bothScenarios(stateAttentionReport({ capacity, chunkLength: 4, window: 6 })).prefill;
  // 列容量 = 6−1+4 = 9 → S = 4×4×9×4 = 576（C に依存しない）
  assertEquals(at(16).workspaceBytes, 512 + 576 + 128);
  assertEquals(at(32).workspaceBytes, 512 + 576 + 128);
  // state スロットは C に比例したまま（S だけが窓で頭打ちになる）
  assertEquals(
    stateAttentionReport({ capacity: 32, chunkLength: 4, window: 6 }).resident.stateBytes,
    (1 * 2 * 32 * 8) * 4 * 2 + 8,
  );
});

Deno.test("行ブロックの枚数は maxStorageBufferBindingSize で変わる（1 枚ぶんだけが同時生存）", () => {
  const at = (limit: number): number =>
    bothScenarios(stateAttentionReport({ capacity: 16, chunkLength: 4, limit })).prefill
      .workspaceBytes;
  // 1 行 = 4×16×4 = 256B。上限に余裕があれば 1 枚（4 行）: S 1024 + 行統計 128
  assertEquals(at(WIDE_LIMIT), 512 + 1024 + 128);
  // 512B → 1 枚 2 行の 2 枚。2 枚目は 1 枚目が返した slot を掴むので総バイトは 1 枚ぶん。
  assertEquals(at(512), 512 + 512 + 64);
  // 256B → 1 枚 1 行の 4 枚。
  assertEquals(at(256), 512 + 256 + 32);
});

Deno.test("端数で 1 行狭いブロックが混ざるとサイズクラス 2 種類ぶんが乗る", () => {
  // M=3・上限 512B → 1 枚 2 行で 2 枚（等分は 2 行 + 1 行）。exact-size 再利用なので
  // 2 枚目の 1 行ぶんは 1 枚目の slot を掴めず、新しい slot が生える。
  const { prefill } = bothScenarios(
    stateAttentionReport({ capacity: 16, chunkLength: 3, limit: 512 }),
  );
  const wide = 4 * 2 * 16 * 4 + 4 * 2 * 2 * 4; // 512 + 64
  const narrow = 4 * 1 * 16 * 4 + 4 * 1 * 2 * 4; // 256 + 32
  // 出力 o [1,4,3,8]=96 要素 → 384
  assertEquals(prefill.workspaceBytes, 384 + wide + narrow);
});

Deno.test("上限が動かすのは中間だけ（io・state・重みの欄は 1 バイトも動かない）", () => {
  const wide = stateAttentionReport({ capacity: 16, chunkLength: 4 });
  const narrow = stateAttentionReport({ capacity: 16, chunkLength: 4, limit: 256 });
  assertEquals(wide.resident, narrow.resident);
  assertEquals(
    wide.scenarios.map((scenario) => scenario.ioBytes),
    narrow.scenarios.map((scenario) => scenario.ioBytes),
  );
  assert(wide.scenarios[0].workspaceBytes > narrow.scenarios[0].workspaceBytes);
});

Deno.test("states 形 attention を持たないグラフは上限を渡しても数字が変わらない", () => {
  const without = estimateSessionMemory(plainModel(), { bindings: { T: 7 } });
  const with_ = estimateSessionMemory(plainModel(), {
    bindings: { T: 7 },
    maxStorageBufferBindingSize: 256,
  });
  assertEquals(without, with_);
  // state_append だけを持つグラフ（一時を出さないノード）も同じ。
  const generation = { chunkLength: 4, bindings: { C: 8 } };
  assertEquals(
    estimateSessionMemory(stateModel(), { bindings: { T: 2 }, generation }),
    estimateSessionMemory(stateModel(), {
      bindings: { T: 2 },
      generation,
      maxStorageBufferBindingSize: 256,
    }),
  );
});

Deno.test("上限の値域はグラフの形に依らず入口で見る（読まない形でも黙って受理しない）", () => {
  // 読み手（stateAttentionTemps）の内側だけに置くと、states 形 attention を持たないこの
  // グラフでは 1 度も実行されず、不正値が「読まないから無害」として黙って通る。
  for (const limit of [-1, 0, 1.5, Number.NaN]) {
    assertThrows(
      () =>
        estimateSessionMemory(plainModel(), {
          bindings: { T: 7 },
          maxStorageBufferBindingSize: limit,
        }),
      ExecutionError,
      "正の安全整数",
    );
  }
  // 対照: 正当値は従来どおり通る（上の 4 本が「何を渡しても落ちる」ではないことの証明）。
  assertEquals(
    estimateSessionMemory(plainModel(), {
      bindings: { T: 7 },
      maxStorageBufferBindingSize: 256,
    }).peakAccountedBytes,
    estimateSessionMemory(plainModel(), { bindings: { T: 7 } }).peakAccountedBytes,
  );
});

Deno.test("states 形 attention の見積りに上限を渡さないのは fail loudly", () => {
  assertThrows(
    () =>
      estimateSessionMemory(openGraph(stateAttentionGraph()), {
        generation: { chunkLength: 4, bindings: { C: 16 } },
      }),
    ExecutionError,
    "options.maxStorageBufferBindingSize が要る",
  );
});

/**
 * S / 行統計の算式が recipe-builder と同じ導出元（`planStateAttention`）から出ていることの
 * 唯一の実測門。
 *
 * このグラフは融合が 1 本も掛からない（states を触るノードは窓を掴まない — ADR 0067 決定 5b）
 * ので、`workspaceBytes` は slot 表の総バイト = `planBacking.residentBytes` と**厳密一致**する。
 * 算式そのものは両者が共有する（= ここでは割れない）が、共有関数へ**渡す材料**（`B·H`・窓・
 * 容量の出どころ）と、返ったバイト数を実行相が確保する位置・サイズクラス再利用の規則が
 * estimator の写しとずれれば、ここが例外なしで割れる。
 *
 * 呼ぶのは full / sliding の 2 変種（下の 2 本）— `colCap` は変種で式が分かれる唯一の欄なので、
 * 片方だけでは分岐のもう一方が無門のままになる。
 *
 * `limitCap` を渡すと device の `maxStorageBufferBindingSize` を絞って**行ブロックを複数枚に
 * 割る**（estimator は `ROW_BLOCK_SPLIT` の受け口を持たないので、枚数を寄せる手は形と上限しか
 * 無い）。複数枚でだけ効く 2 つの規則 — ブロック跨ぎのプール再利用と、端数で 1 行狭い
 * ブロックが混ざったときのサイズクラス 2 種 — は、絞らない呼び方では 1 度も踏まれない。
 * MUST: 絞ったときは `expectedBlocks` を渡して枚数を先に固定する（1 枚に落ちた形で緑になると
 * 「複数枚での一致」を見たことにならない）。
 */
const assertPlanBackingMatchesEstimate = async (
  variant: {
    readonly capacity: number;
    readonly window?: number;
    readonly heads?: number;
    readonly chunkLength?: number;
    readonly limitCap?: number;
    readonly expectedBlocks?: number;
  },
): Promise<void> => {
  const heads = variant.heads ?? 4;
  const chunkLength = variant.chunkLength ?? 4;
  const gpu = await acquireGpu(
    variant.limitCap === undefined
      ? {}
      : { [LIMIT_CAPS]: { maxStorageBufferBindingSize: variant.limitCap } },
  );
  try {
    const limit = gpu.limits.maxStorageBufferBindingSize;
    if (variant.limitCap !== undefined) {
      assertEquals(limit, variant.limitCap, "requiredLimits が絞られていない（門が空振りする）");
    }
    if (variant.expectedBlocks !== undefined) {
      // 実行と見積りが共有する純関数そのもので枚数を固定する（前提の可視化）。
      const blocks = planStateAttention({
        batchHeads: heads,
        chunkRows: chunkLength,
        capacity: variant.capacity,
        window: variant.window,
      }, limit).blocks;
      assertEquals(blocks.length, variant.expectedBlocks, "行ブロックの枚数");
    }
    const model = openGraph(stateAttentionGraph(variant.window, heads));
    const generation = { chunkLength, bindings: { C: variant.capacity } };
    const { prefill } = bothScenarios(
      estimateSessionMemory(model, { generation, maxStorageBufferBindingSize: limit }),
    );
    const session = await createSession(gpu, model);
    try {
      const context = await session.createGenerationContext(generation);
      try {
        // slot backing は同じ signature の 2 run 目で組まれる（1 run 目はアリーナ経路）。
        for (let step = 0; step < 2; step += 1) {
          await session.run(
            {
              q: fill([1, heads, chunkLength, 8], (i) => ((i % 5) - 2) / 4),
              k: fill([1, 2, chunkLength, 8], (i) => ((i % 3) - 1) / 4),
              v: fill([1, 2, chunkLength, 8], (i) => ((i % 7) - 3) / 4),
            },
            {},
            { context, queryLength: chunkLength },
          );
        }
      } finally {
        await context.dispose();
      }
      assertEquals(session.diagnostics().planBacking.residentBytes, prefill.workspaceBytes);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

Deno.test({
  name: "states 形 attention の中間総量が実行計画の slot 表と厳密一致する（full 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = C = 16（S は容量に比例する側）。
  fn: () => assertPlanBackingMatchesEstimate({ capacity: 16 }),
});

Deno.test({
  name: "states 形 attention の中間総量が実行計画の slot 表と厳密一致する（sliding 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = W−1+M = 8−1+4 = 11（容量 C とは別の式 — full と同じ数にならない形を選ぶ）。
  fn: () => assertPlanBackingMatchesEstimate({ capacity: 8, window: 8 }),
});

/**
 * 行ブロック**複数枚**での 2 実装一致。H=64 / Hkv=2 / D=8 / C=16 では S の 1 行が
 * 64·16·4 = 4096B なのに対し state スロットは 2·16·8·4 = 1024B なので、上限 8192B に絞ると
 * 「スロットは束縛できるが S 4 行は束縛できない」形になり、行ブロックが必ず割れる。
 */
Deno.test({
  name: "states 形 attention の中間総量が複数枚でも slot 表と厳密一致する（full 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 1 枚 2 行の 2 枚（8192 ÷ 4096 = 2 行／枚・M=4）。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 4,
      limitCap: 8192,
      expectedBlocks: 2,
    }),
});

Deno.test({
  name: "states 形 attention の中間総量が端数ブロックでも slot 表と厳密一致する（実 GPU）",
  ignore: !GPU_AVAILABLE,
  // M=5 を 2 行／枚で割ると 3 枚（2 行 + 2 行 + 1 行）— サイズクラスが 2 種同時に生きる唯一の形。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 5,
      limitCap: 8192,
      expectedBlocks: 3,
    }),
});

Deno.test({
  name: "states 形 attention の中間総量が複数枚でも slot 表と厳密一致する（sliding 変種・実 GPU）",
  ignore: !GPU_AVAILABLE,
  // 列容量 = W−1+M = 8−1+4 = 11 → 1 行 64·11·4 = 2816B。上限 5632B で 2 行／枚の 2 枚。
  fn: () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      window: 8,
      chunkLength: 4,
      limitCap: 5632,
      expectedBlocks: 2,
    }),
});

Deno.test("1 行でも上限に入らない形は fail loudly（行ブロックでは割り切れない）", () => {
  assertThrows(
    // 1 行 = 4×16×4 = 256B > 128B
    () => stateAttentionReport({ capacity: 16, chunkLength: 4, limit: 128 }),
    ExecutionError,
    "既にストレージ束縛の上限を超える",
  );
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

Deno.test("契約違反のグラフは常駐計画より先に契約検査で落ちる（nodes[i] (op) を名乗る）", () => {
  // conv1d の attrs から `groups` を落とした形（ADR 0015 の「既定値補完をしない」）。
  // 常駐計画（planWeightResidency → i4 適格判定）が先に走ると `nodes (conv1d)` という
  // 添字を持たない診断になり、実構築（PreparedModel）と別の文言で落ちる。
  const graph = i4Conv1dGraph(1);
  const model = openGraph(
    {
      ...graph,
      nodes: [{ ...graph.nodes[0], attrs: { stride: 1, padding: 0, dilation: 1 } }],
    },
    [
      { name: "m.b", dtype: "F32", shape: [4], data: f32Bytes([0, 0, 0, 0]) },
      { name: "m.s", dtype: "F32", shape: [4, 4], data: f32Bytes(new Array(16).fill(1)) },
      { name: "m.w", dtype: "I4", shape: [4, 32, 2], data: new Uint8Array(128) },
    ],
  );
  const error = assertThrows(() => estimateSessionMemory(model), OpContractError, "nodes[");
  assert(error.message.includes("conv1d"), error.message);
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
// 実構築と同じ門（グラフ全体を見ないと決まらない契約）
// ---------------------------------------------------------------------------

// 全量面は `PreparedModel` を経由しないので、契約検査を estimator 側で通していないと
// 「createSession が必ず落ちるモデル」に完全な AdmissionReport を返す（admission の目的と逆）。

/** `kv` スロットに読者だけが居て `state_append` を 1 本も持たないグラフ（決定 5b 違反）。 */
const readerOnlyStateGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["attention"] },
  symbols: ["M", "C"],
  inputs: [
    { name: "q", dtype: "f32", shape: [1, 8, "M", 4] },
    { name: "k", dtype: "f32", shape: [1, 2, "M", 4] },
    { name: "v", dtype: "f32", shape: [1, 2, "M", 4] },
  ],
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 8, "M", 4] } },
  states: {
    "kv.k": { dtype: "f32", shape: [1, 2, "C", 4] },
    "kv.v": { dtype: "f32", shape: [1, 2, "C", 4] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kv.k", v: "kv.v" },
    },
  ],
});

Deno.test("読者だけの state スロットを持つグラフに見積りを返さない（実構築と同じ契約検査）", () => {
  const error = assertThrows(
    () =>
      estimateSessionMemory(openGraph(readerOnlyStateGraph()), {
        generation: { chunkLength: 4, bindings: { C: 8 } },
      }),
    ExecutionError,
    "state_append が 1 本も無い",
  );
  assert(error.message.includes("kv.k"), error.message);
});

/** 連結軸に異なるシンボルが混ざった cat（ADR 0046 の残る拒否）。 */
const mixedSymbolCatGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["cat"] },
  symbols: ["T", "U"],
  inputs: [
    { name: "a", dtype: "f32", shape: ["T", 2] },
    { name: "b", dtype: "f32", shape: ["U", 2] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: ["T", 2] } },
  nodes: [{ op: "cat", ins: ["a", "b"], outs: ["y"], attrs: { dim: 0 } }],
});

Deno.test("cat の連結軸に異なるシンボルが混ざるグラフに見積りを返さない", () => {
  assertThrows(
    () => estimateSessionMemory(openGraph(mixedSymbolCatGraph()), { bindings: { T: 2, U: 3 } }),
    ExecutionError,
    "異なるシンボル",
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

// states 形 attention の S / 行統計は**勘定に入った**（融合の成立に依存せず必ず出るので、
// 融合の項の文言では覆えない大きさだった）。残る非勘定は「states 形でない attention」と
// linear の i8a8 量子化中間で、unaccounted はそちらを名乗る。
Deno.test("unaccounted は states 形でない attention の一時を名乗り、S / 行統計は勘定済みと書く", () => {
  const { unaccounted } = estimateSessionMemory(stateModel(), {
    bindings: { T: 2 },
    generation: { chunkLength: 4, bindings: { C: 8 } },
  });
  const joined = unaccounted.join("\n");
  assert(joined.includes("states 形でない attention のノード内一時"), joined);
  assert(joined.includes("i8a8"), joined);
  assert(joined.includes("行統計は勘定に入っている"), joined);
});

// ---------------------------------------------------------------------------
// 実 GPU 突合
// ---------------------------------------------------------------------------

/**
 * 3 つの欄が全て非 0 になるモデル — i8 の embedding 表（適格 = 圧縮のまま常駐）・f16 の
 * mul 被演算子（適格外 = f32 展開）・f32 の add 被演算子（非圧縮のまま常駐）。
 *
 * 圧縮 / 展開の 2 欄は診断 `storage` と厳密一致を主張でき、f32 は診断に現れない
 * （`weights.uncompressedBytes` の欄を分けている理由そのもの）ので手計算定数と突合する。
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
        const { weights } = estimate.resident;
        assertEquals(weights.compressedBytes, storage.residentCompressedBytes);
        assertEquals(weights.expandedBytes, storage.hostExpandedBytes);
        // f32 の c は診断に現れない（ADR 0006 の storage 診断は低精度格納だけ）— 手計算 3×4
        assertEquals(weights.uncompressedBytes, 12);
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
          stateEstimate.resident.stateBytes,
          stateSession.diagnostics().stateBacking.residentBytes,
        );
        await context.dispose();
      } finally {
        await stateSession.dispose();
      }
      // workspaceBytes は突合しない — 融合が中間を消し、行ブロック分割が一時を足すので
      // 実測（planBacking.residentBytes / lastRun.peakTransientBytes）とは原理的にずれる
      // （estimator の unaccounted 欄が認めている差そのもの）。
    } finally {
      gpu.destroy();
    }
  },
});
