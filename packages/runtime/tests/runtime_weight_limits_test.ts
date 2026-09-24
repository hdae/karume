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
import type { BoundContainer } from "../src/format/container/bind.ts";
import type { IrGraph } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { estimateGraphMemory } from "../src/runtime/estimate.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import {
  assertChunkLength,
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { assertWeightsWithinLimits, planWeightResidency } from "../src/runtime/weight-residency.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./helpers/f16.ts";
import { autoTensors, mergeGraph, type StorageMap } from "./helpers/merged-graph.ts";
import {
  baseDeclaration,
  type DeclarationJson,
  GRAPH_NAME,
  memoryModel,
  withStateReaders,
} from "./helpers/model-fixture.ts";

// ---------------------------------------------------------------------------
// 材料（グラフ・フェイク device）
// ---------------------------------------------------------------------------

/**
 * linear 1 本（重み `w` は f16 で適格 = payload がそのまま GPU に載る）。
 * w の payload は 8×4×2 = **64 バイト**・bias `b` は f32 の生バイト常駐で **32 バイト**。
 */
const f16LinearGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
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
const mixedSeatGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", "mul"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 2] }],
  outputs: ["y"],
  initializers: {
    w: {},
    b: {},
    g: {},
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

/** このファイルの 2 グラフの格納（IR v2 の宣言は持たない — 束縛表側の欄）。 */
const STORAGE: StorageMap = { w: "f16", g: "f16" };
const MIXED_STORAGE: StorageMap = { w: "int8-sym", g: "f16" };

/** 宣言 → 常駐計画（重みの中身は 1 バイトも要らない — プランナは純関数）。 */
const residencyOf = (
  graph: DeclarationJson,
  storage: StorageMap,
): ReturnType<typeof planWeightResidency> => planWeightResidency(mergeGraph(graph, storage));

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
  const residency = residencyOf(f16LinearGraph(), STORAGE);
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
  const residency = residencyOf(f16LinearGraph(), STORAGE);
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
  const residency = residencyOf(mixedSeatGraph(), MIXED_STORAGE);
  const error = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(16, 16)),
    ExecutionError,
  );
  // 適格外の `m.g` は payload 16 バイト（= 上限ちょうど）だが、GPU に載るのは展開後の 32 バイト。
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
    () => assertWeightsWithinLimits(residencyOf(mixedSeatGraph(), MIXED_STORAGE), limits(16, 16)),
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
  const mixed = residencyOf(mixedSeatGraph(), MIXED_STORAGE);
  // 最大の確保は 32 バイト（w.scale / b / g の展開後）。
  assertWeightsWithinLimits(mixed, limits(32, 32));
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(31, 32)), ExecutionError);
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(32, 31)), ExecutionError);

  const f16 = residencyOf(f16LinearGraph(), STORAGE);
  assertWeightsWithinLimits(f16, limits(64, 64));
  assertThrows(() => assertWeightsWithinLimits(f16, limits(63, 64)), ExecutionError);
});

// ---------------------------------------------------------------------------
// Session 構築の入口で実際に通ること（確保より前に落ちる）
// ---------------------------------------------------------------------------

/** f16LinearGraph に対応する供給（宣言と現物のバイト数は合流層の門が見る）。 */
const f16LinearModel = (): BoundContainer => {
  const graph = f16LinearGraph();
  // `w` だけは実バイト列を渡す（f16 の payload 長が宣言から決まることを合流層が見る）。
  const supplied = autoTensors(graph, STORAGE).map((tensor) =>
    tensor.initializer === "w"
      ? {
        ...tensor,
        bytes: f16BytesFromBits(new Array(32).fill(f32ToF16Bits(0))),
      }
      : tensor
  );
  return memoryModel(graph, supplied);
};

Deno.test("createSession は上限超過の重みを createBuffer より前に落とす", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(32, 64));
  const error = await assertRejects(
    () => createSessionFromContainer(gpu, f16LinearModel(), GRAPH_NAME),
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
    () => createSessionFromContainer(gpu, f16LinearModel(), GRAPH_NAME),
    Error,
  );
  assertStringIncludes(error.message, "注入: createBuffer が呼ばれた");
  assertEquals(spy.createBuffer, 1);
});

// ---------------------------------------------------------------------------
// state スロット側（既存の束縛上限ゲートの補完 — ADR 0066 追記 5）
// ---------------------------------------------------------------------------

/** state スロット 1 本（`kv`）だけを持つグラフ（参照側の `state_append` は withStateReaders が足す）。 */
const stateGraph = (shape: readonly number[]): IrGraph => {
  const graph = baseDeclaration();
  graph.states = { kv: { dtype: "f32", shape: [...shape] } };
  return mergeGraph(withStateReaders(graph));
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

// ---------------------------------------------------------------------------
// 見積り側（estimateGraphMemory）— 実構築と**同じ境界**であること（ADR 0066 追記 5 / Q4-3）
//
// 見積りだけが受理する寸法があると、CLI のように `--capacity` の値検査を見積りに兼ねさせて
// いる呼び手で「起動時には数字が出て、最初のターンで落ちる」形になる。
// ---------------------------------------------------------------------------

/**
 * 見積りと実構築の**両方**を通せる最小グラフ。上の `stateGraph` は `state_append` の入力が
 * rank-4 契約を満たさない（実構築は graph を計画しないので通るが、見積りは planGraph を通る）
 * ので、対で見るテストはこちらを使う。スロット `kv` は [1,2,1,4] = 8 要素 × 4 = **32 バイト**
 * （`stateGraph([8])` と同じ大きさなので境界の数字も同じ）。
 */
const pairedStateGraph = (): IrGraph => {
  const graph = baseDeclaration();
  graph.inputs.push({ name: "chunk", dtype: "f32", shape: [1, 2, 1, 4] });
  graph.states = { kv: { dtype: "f32", shape: [1, 2, 1, 4] } };
  graph.requires.ops.push("state_append");
  graph.nodes.push({
    op: "state_append",
    ins: ["chunk"],
    outs: [],
    attrs: {},
    states: { slot: "kv" },
  });
  return mergeGraph(graph);
};

/** 同じグラフ・同じ上限で見積りを引く（state スロットは `kv` 1 本 = 8 要素 × 4 = 32 バイト）。 */
const stateEstimateAt = (
  graph: IrGraph,
  maxStorageBufferBindingSize: number,
  maxBufferSize: number,
): number =>
  estimateGraphMemory(graph, planWeightResidency(graph), {
    bindings: { T: 1 },
    generation: { chunkLength: 1 },
    maxStorageBufferBindingSize,
    maxBufferSize,
  }).resident.stateBytes;

/** 実構築を同じ上限で走らせ、拒否されたなら文言を返す（確保へ進んだなら undefined）。 */
const createRejectionAt = async (
  graph: IrGraph,
  maxStorageBufferBindingSize: number,
  maxBufferSize: number,
): Promise<string | undefined> => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(maxStorageBufferBindingSize, maxBufferSize));
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, graph), { chunkLength: 1 }),
    Error,
  );
  if (spy.createBuffer > 0) return undefined;
  return error.message;
};

Deno.test("束縛上限の境界は見積りと実構築で同じ（1 バイト下回ると両方が同じ文言で落ちる）", async () => {
  const graph = pairedStateGraph();
  // 上限ぴったり（32 バイト）は両方とも通る — 実構築は確保へ進み、見積りは数字を返す。
  assertEquals(stateEstimateAt(graph, 32, 1024), 32 + 8, "スロット 32 + 論理長 uniform 8");
  assertEquals(await createRejectionAt(graph, 32, 1024), undefined, "境界ちょうどは確保へ進む");
  // 1 バイト下回ると、見積りも実構築も落ちる。文言は 1 本の門を共有しているので同一。
  const rejection = await createRejectionAt(graph, 31, 1024);
  const estimateError = assertThrows(
    () => stateEstimateAt(graph, 31, 1024),
    ExecutionError,
  );
  assertEquals(estimateError.message, rejection);
  assertStringIncludes(
    estimateError.message,
    "state 'kv': 容量 [1,2,1,4] の 32 バイトが maxStorageBufferBindingSize 31 バイトを超える",
  );
});

Deno.test("バッファ上限の境界も見積りと実構築で同じ（束縛上限に収まる形で片側だけを割る）", async () => {
  const graph = pairedStateGraph();
  assertEquals(stateEstimateAt(graph, 1024, 32), 32 + 8);
  assertEquals(await createRejectionAt(graph, 1024, 32), undefined);
  const rejection = await createRejectionAt(graph, 1024, 31);
  const estimateError = assertThrows(() => stateEstimateAt(graph, 1024, 31), ExecutionError);
  assertEquals(estimateError.message, rejection);
  assertStringIncludes(
    estimateError.message,
    "state 'kv': 容量 [1,2,1,4] の 32 バイトが maxBufferSize 31 バイトを超える",
  );
});

Deno.test("上限を渡さない見積りは既定値を捏造しない（device を持たない呼び手は従来どおり）", () => {
  const graph = pairedStateGraph();
  // 片側だけを渡す形も含めて、渡した欄だけが検査に効く（未指定の側は見ない）。
  assertEquals(
    estimateGraphMemory(graph, planWeightResidency(graph), {
      bindings: { T: 1 },
      generation: { chunkLength: 1 },
    }).resident.stateBytes,
    32 + 8,
  );
  assertThrows(
    () =>
      estimateGraphMemory(graph, planWeightResidency(graph), {
        bindings: { T: 1 },
        generation: { chunkLength: 1 },
        maxBufferSize: 31,
      }),
    ExecutionError,
    "maxBufferSize 31 バイトを超える",
  );
});
