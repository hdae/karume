// 重み・state の確保寸法を device の絶対上限と突き合わせる門（確保の**前**に落とす）。
//
// 動機は「確保失敗の検出は shard 単位 errorScope に全面依存」（ADR 0070 決定 4）の弱点 —
// errorScope の網は実装の報告品質に依存し（out-of-memory scope が黙る device が実在する —
// docs/known-issues.md の Metal 節）、捕まえても診断は shard 粒度で、数 GiB 転送した後にしか
// 出ない。無効バッファへの writeBuffer は警告も出さない no-op なので、網が抜けた先は
// 「重みが空のまま走り出す」になる。
//
// ここが固定するのは 4 点:
// ① 寸法の取り方 — 適格席は payload・適格外席は **f32 展開後**・companion scale も 1 本数える
// ② 上限は 2 本とも見る（maxStorageBufferBindingSize / maxBufferSize）
// ③ 超過は**全件を 1 回で**列挙し、上限ぴったりは通る（境界）
// ④ 実際に確保より前で落ちる — フェイク device の createBuffer が 1 回も呼ばれない
//
// 実 GPU は使わない。上限は環境ごとに違い、実物で超過を作るには数 GiB の資産が要るので、門
// そのものは「limits を絞ったフェイク device」でしか決定論的に固定できない。

import { assertEquals, assertRejects, assertStringIncludes, assertThrows } from "@std/assert";
import { openModel } from "../src/format/container.ts";
import { type IrGraph, parseIrGraph } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { createSession } from "../src/runtime/executor.ts";
import {
  assertChunkLength,
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { assertWeightsWithinLimits, planWeightResidency } from "../src/runtime/weight-residency.ts";
import { baseGraph, f32Bytes, type GraphJson, withStateReaders } from "./helpers/format.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./helpers/f16.ts";
import { graphModelBuffer } from "./helpers/graph.ts";

// ---------------------------------------------------------------------------
// 材料（グラフ・フェイク device）
// ---------------------------------------------------------------------------

/**
 * linear 1 本（重み `w` は f16 で適格 = payload がそのまま GPU に載る）。
 * w の payload は 8×4×2 = **64 バイト**・bias `b` は f32 の生バイト常駐で **32 バイト**。
 */
const f16LinearGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "f16" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [8, 4] },
    b: { dtype: "f32", shape: [8] },
    y: { dtype: "f32", shape: [2, 8] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/**
 * 4 種類の確保が同居するグラフ（席ごとに寸法の出どころが違うことを 1 本で見る）:
 * - `w` = i8 常駐 → payload **16 バイト**（8×2）+ companion scale **32 バイト**（8 チャネル×4）
 * - `b` = 生バイト常駐 → **32 バイト**
 * - `g` = 重みスロット以外（mul）の消費で適格外 → payload は 16 バイトだが**確保は展開後の
 *   32 バイト**
 *
 * scale が payload より大きいのは意図的（`[8,2]` は行が短くチャネルが多い）— scale を数えない
 * 実装が「payload だけ見て通す」形をここで踏む。
 */
const mixedSeatGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "m.w", storage: { dtype: "i8", scale: "m.s" } },
    b: { tensor: "m.b", storage: { dtype: "f32" } },
    g: { tensor: "m.g", storage: { dtype: "f16" } },
  },
  values: {
    w: { dtype: "f32", shape: [8, 2] },
    b: { dtype: "f32", shape: [8] },
    g: { dtype: "f32", shape: [8] },
    h: { dtype: "f32", shape: [2, 8] },
    y: { dtype: "f32", shape: [2, 8] },
  },
  nodes: [
    { op: "linear", ins: ["x", "w", "b"], outs: ["h"], attrs: {} },
    { op: "mul", ins: ["h", "g"], outs: ["y"], attrs: {} },
  ],
});

/** グラフ JSON → 常駐計画（配布形のバイト列は 1 バイトも要らない — プランナは純関数）。 */
const residencyOf = (graph: GraphJson): ReturnType<typeof planWeightResidency> =>
  planWeightResidency(parseIrGraph(JSON.stringify(graph)));

/**
 * 上限 2 本だけを指定した limits（他のキーは門が見ないので WebGPU core 既定で埋める）。
 * 引数の順は「束縛上限・バッファ上限」で、**あえて逆転させられる**（2 本を独立に見ている
 * ことの検出条件そのもの）。
 */
const limits = (
  maxStorageBufferBindingSize: number,
  maxBufferSize: number,
): RequiredLimits => ({
  maxBufferSize,
  maxStorageBufferBindingSize,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
});

/** 確保の到達を数える器（門が確保より前で落ちることの観測点）。 */
type AllocSpy = { createBuffer: number };

/**
 * createBuffer が呼ばれたら**必ず分かる**フェイク device。DOM 型全体は再現しないので cast で
 * 渡す（tests/helpers/fake-gpu.ts と同じテスト専用の境界）。errorScope と queue は後始末経路が
 * 触るだけなので素通しの no-op でよい。
 */
const spyDevice = (spy: AllocSpy): GPUDevice =>
  ({
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    pushErrorScope: (): void => {},
    popErrorScope: (): Promise<GPUError | null> => Promise.resolve(null),
    createBuffer: (): GPUBuffer => {
      spy.createBuffer += 1;
      throw new Error("注入: createBuffer が呼ばれた");
    },
    queue: {
      writeBuffer: (): void => {},
      onSubmittedWorkDone: (): Promise<undefined> => Promise.resolve(undefined),
    },
  }) as unknown as GPUDevice;

const spyGpu = (spy: AllocSpy, gpuLimits: RequiredLimits): GpuContext =>
  new GpuContext(spyDevice(spy), readAdapterInfo({}), gpuLimits, new Set());

// ---------------------------------------------------------------------------
// chunkLength の値域門（GPU 非依存の純関数 — src/runtime/generation-context.ts）
// ---------------------------------------------------------------------------

/**
 * 論理長の搬送先が u32 なので上限は 0xffffffff（ADR 0066 決定 4 / 追記 4）。ここに書くのは
 * 実装の private 定数の**写し**ではなく、文言に出る上限をそのまま観測する形にしてある
 * （実装が上限を変えれば下の assertStringIncludes が落ちる）。
 */
const MAX_LOGICAL_LENGTH = 0xffffffff;

Deno.test("assertChunkLength は 1..u32 上限の整数だけを通す", () => {
  // 境界の内側（下端・上端とも通る）
  assertChunkLength(1);
  assertChunkLength(MAX_LOGICAL_LENGTH);
  // 0 / 負 / 非整数 / 上限超え / NaN は全て同じ門で落ちる。estimator も同じ門を通るので、
  // ここが緩むと「実構築が拒否する指定に見積りだけが正常値を返す」形になる。
  for (const bad of [0, -1, 1.5, MAX_LOGICAL_LENGTH + 1, Number.NaN]) {
    const error = assertThrows(() => assertChunkLength(bad), ExecutionError, "chunkLength");
    assertStringIncludes(error.message, `chunkLength ${bad}`);
    assertStringIncludes(error.message, `1..${MAX_LOGICAL_LENGTH}`);
  }
});

// ---------------------------------------------------------------------------
// 寸法の取り方（純関数の門）
// ---------------------------------------------------------------------------

Deno.test("上限を超える payload は席ごとの確保寸法で名指しされる", () => {
  const residency = residencyOf(f16LinearGraph());
  // w は 64 バイト（束縛上限 32 を超える）・b は 32 バイトちょうどで通る。
  const error = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(32, 64)),
    ExecutionError,
  );
  assertStringIncludes(error.message, "重みバッファ 1 本");
  assertStringIncludes(
    error.message,
    "initializer 'w' の payload（席 f16・確保 64 バイト）: " +
      "maxStorageBufferBindingSize 32 バイトを 32 バイト超える",
  );
});

Deno.test("上限は maxStorageBufferBindingSize と maxBufferSize の 2 本とも見る", () => {
  const residency = residencyOf(f16LinearGraph());
  // 束縛上限だけが緩い device（仕様が期待する大小関係が崩れた形）でも素通ししない。
  const bufferOnly = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(1024, 32)),
    ExecutionError,
  );
  assertStringIncludes(bufferOnly.message, "maxBufferSize 32 バイトを 32 バイト超える");
  assertEquals(
    bufferOnly.message.includes("maxStorageBufferBindingSize"),
    false,
    "超えていない上限は文言に出さない（どれを直せばよいかが埋もれる）",
  );

  // 両方を超えるときは 1 行に 2 本とも並ぶ（どちらを上げても足りないことが読める）。
  const both = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(16, 48)),
    ExecutionError,
  );
  assertStringIncludes(both.message, "maxStorageBufferBindingSize 16 バイトを 48 バイト超える");
  assertStringIncludes(both.message, "maxBufferSize 48 バイトを 16 バイト超える");
});

Deno.test("適格外席は f32 展開後の寸法で見る・companion scale も 1 本として見る", () => {
  const residency = residencyOf(mixedSeatGraph());
  const error = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(16, 16)),
    ExecutionError,
  );
  // 適格外の `g` は payload 16 バイト（= 上限ちょうど）だが、GPU に載るのは展開後の 32 バイト。
  assertStringIncludes(
    error.message,
    "initializer 'g' の payload（f32 展開後）（席 expanded・確保 32 バイト）",
  );
  // scale は payload と別に確保されるので、payload が通っても scale で落ちる。
  assertStringIncludes(error.message, "initializer 'w' の scale（席 i8・確保 32 バイト）");
  assertEquals(
    error.message.includes("initializer 'w' の payload"),
    false,
    "16 バイトの payload は上限ちょうどで通る（scale だけが超過）",
  );
});

Deno.test("超過は全件を 1 回で列挙する（1 本ずつ落とさない）", () => {
  const error = assertThrows(
    () => assertWeightsWithinLimits(residencyOf(mixedSeatGraph()), limits(16, 16)),
    ExecutionError,
  );
  // w.scale / b / g の 3 本（w.payload だけが 16 バイトで通る）。
  assertStringIncludes(error.message, "重みバッファ 3 本");
  assertEquals(
    error.message.split("\n").filter((line) => line.startsWith("  - initializer")).length,
    3,
    "超過した確保の数だけ行が出る",
  );
});

Deno.test("上限ぴったりは通る（境界の内側 / 外側）", () => {
  const mixed = residencyOf(mixedSeatGraph());
  // 最大の確保は 32 バイト（w.scale / b / g の展開後）。
  assertWeightsWithinLimits(mixed, limits(32, 32));
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(31, 32)), ExecutionError);
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(32, 31)), ExecutionError);

  const f16 = residencyOf(f16LinearGraph());
  assertWeightsWithinLimits(f16, limits(64, 64));
  assertThrows(() => assertWeightsWithinLimits(f16, limits(63, 64)), ExecutionError);
});

// ---------------------------------------------------------------------------
// Session 構築の入口で実際に通ること（確保より前に落ちる）
// ---------------------------------------------------------------------------

/** f16LinearGraph に対応する実配布形（宣言と現物のバイト数は container の門が見る）。 */
const f16LinearModelBuffer = (): ArrayBuffer =>
  graphModelBuffer(f16LinearGraph(), [
    // 整列降順（F32 → F16）で詰める（整列単位を跨がせない）。
    { name: "m.b", dtype: "F32", shape: [8], data: f32Bytes(new Array(8).fill(0)) },
    {
      name: "m.w",
      dtype: "F16",
      shape: [8, 4],
      data: f16BytesFromBits(new Array(32).fill(f32ToF16Bits(0))),
    },
  ]);

Deno.test("createSession は上限超過の重みを createBuffer より前に落とす", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(32, 64));
  const error = await assertRejects(
    () => createSession(gpu, openModel(f16LinearModelBuffer())),
    ExecutionError,
  );
  assertStringIncludes(error.message, "device の上限を超える（確保の前に検出）");
  assertEquals(
    spy.createBuffer,
    0,
    "確保に 1 本も到達しない（errorScope が沈黙する device でも検出できる条件そのもの）",
  );
});

Deno.test("上限に収まる重みは門を素通りして確保へ進む（門は全拒否ではない）", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(1024, 1024));
  // 同じモデル・同じ経路で、違うのは limits だけ。確保まで進んだことは注入した失敗で分かる。
  const error = await assertRejects(
    () => createSession(gpu, openModel(f16LinearModelBuffer())),
    Error,
  );
  assertStringIncludes(error.message, "注入: createBuffer が呼ばれた");
  assertEquals(spy.createBuffer, 1);
});

// ---------------------------------------------------------------------------
// state スロット側（既存の束縛上限ゲートの補完 — ADR 0066 追記 5）
// ---------------------------------------------------------------------------

/** state スロット 1 本（`kv`）だけを持つグラフ（参照側 = state_append は helpers が足す）。 */
const stateGraph = (shape: readonly number[]): IrGraph => {
  const graph = baseGraph();
  graph.states = { kv: { dtype: "f32", shape: [...shape] } };
  return parseIrGraph(JSON.stringify(withStateReaders(graph)));
};

const stateHost = (gpu: GpuContext, graph: IrGraph): GenerationContextHost => ({
  gpu,
  graph,
  flush: () => Promise.resolve(),
  serialize: <T>(body: () => Promise<T>): Promise<T> => body(),
  forget: () => {},
});

Deno.test("state スロットの maxBufferSize 超過も createBuffer より前に落ちる", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  // 束縛上限には収まるがバッファ上限を超える形（2 本を独立に見ていなければ素通りする）。
  const gpu = spyGpu(spy, limits(1024, 16));
  const graph = stateGraph([8]);
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, graph), { chunkLength: 1 }),
    ExecutionError,
  );
  assertStringIncludes(
    error.message,
    "state 'kv': 容量 [8] の 32 バイトが maxBufferSize 16 バイトを超える",
  );
  assertEquals(spy.createBuffer, 0, "確保に 1 本も到達しない");
});

Deno.test("state スロットの maxStorageBufferBindingSize 超過も createBuffer より前に落ちる", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  // バッファ上限には収まるが束縛上限を超える形（上の maxBufferSize 側と対 — 2 本を独立に
  // 見ていなければ、どちらか片方だけが検出器になる）。
  const gpu = spyGpu(spy, limits(16, 1024));
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, stateGraph([8])), { chunkLength: 1 }),
    ExecutionError,
  );
  assertStringIncludes(
    error.message,
    "state 'kv': 容量 [8] の 32 バイトが maxStorageBufferBindingSize 16 バイトを超える",
  );
  assertEquals(spy.createBuffer, 0, "確保に 1 本も到達しない");
});

Deno.test("state スロットが両上限に収まれば確保へ進む（門は全拒否ではない）", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(1024, 1024));
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, stateGraph([8])), { chunkLength: 1 }),
    Error,
  );
  assertStringIncludes(error.message, "注入: createBuffer が呼ばれた");
  assertEquals(spy.createBuffer, 1);
});
