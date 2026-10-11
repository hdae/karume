// run の途中の送り込みの待ち（ADR 0123 — `executeBakedPlanPaced` が返す Promise を run が `await` する）が
// 失敗する経路の門。フェイク device の Session で撃つ（GPU は要らない）。
//
// 縛るのは 2 点（待ちを足したことで run に入った非同期の分岐の失敗経路 — 成功経路は
// `gpu_submit_backpressure_test.ts` の実 GPU の門と `runtime_recipe_test.ts` が見る）:
//
// ① 印の待ちの間に device が失われたら、run はハングせず `GpuDeviceLostError` で決着し、消失の購読を基線へ
//    戻す（ADR 0054 / `raceDeviceLost` — 消失後の `onSubmittedWorkDone` が解決しない実装がありうる）。
// ② state を書く generation run が待ちの後で落ちたら、context を poison する（ADR 0066 追記 3 の失敗の
//    原子性 — 待つ前に未 submit を出し切るので、待ちを挟んだ失敗は「submit した後の失敗」になる。過少 poison は
//    壊れた KV で生成が続く沈黙破壊）。
//
// NOTE: フェイク device は run が触る GPU の面（パイプライン・bind group・コマンド列・バッファ・queue）だけを
// 持つ。値は計算しない — 見るのは失敗の決着と後始末で、出力の値ではない。

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertStrictEquals,
} from "@std/assert";
import {
  GpuContext,
  GpuDeviceLostError,
  readAdapterInfo,
  type RequiredLimits,
} from "../src/gpu/device.ts";
import {
  IN_FLIGHT_MARKER_INTERVAL,
  IN_FLIGHT_MAX_MARKERS,
  type SubmitPolicy,
} from "../src/gpu/submit.ts";
import { createSessionFromContainer, type Session, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { type DeclarationJson, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";

/** 上限は門に掛からない値でよい（見たいのは失敗経路で、確保の寸法ではない）。 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 1 << 28,
  maxStorageBufferBindingSize: 1 << 27,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 8,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

/** 1 dispatch = 1 submit（印の位置を dispatch の数から決められる形 — 実 GPU の門と同じ政策）。 */
const ONE_PER_SUBMIT: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 1,
  minChunkSize: 1,
  maxChunkSize: 1,
};

/** 未完了の印が上限 + 1 本に届く（run がステップの間で最も古い印を待つ）だけの add の段数。 */
const CHAIN = IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1) + 8;

/** y = x + x + … + x（add を {@link CHAIN} 段つなぐ）。 */
const chainNodes = () =>
  Array.from({ length: CHAIN }, (_, i) => ({
    op: "add",
    ins: [i === 0 ? "x" : `v${i - 1}`, "x"],
    outs: [i === CHAIN - 1 ? "y" : `v${i}`],
    attrs: {},
  }));

const chainValues = () =>
  Object.fromEntries(
    Array.from({ length: CHAIN }, (_, i) => [
      i === CHAIN - 1 ? "y" : `v${i}`,
      { dtype: "f32", shape: [4, 3] },
    ]),
  );

/** state を書かないグラフ（{@link CHAIN} 段の add）。 */
const CHAIN_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [4, 3] }],
  outputs: ["y"],
  initializers: {},
  values: chainValues(),
  nodes: chainNodes(),
};

/** 先頭で state を書き（`state_append`）、その後に {@link CHAIN} 段の add を積むグラフ。 */
const STATE_THEN_CHAIN_GRAPH: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add", "state_append"] },
  symbols: ["M"],
  inputs: [
    { name: "k", dtype: "f32", shape: [1, 1, "M", 4] },
    { name: "x", dtype: "f32", shape: [4, 3] },
  ],
  outputs: ["y"],
  initializers: {},
  values: chainValues(),
  states: { kslot: { dtype: "f32", shape: [1, 1, 8, 4] } },
  nodes: [
    { op: "state_append", ins: ["k"], outs: [], attrs: {}, states: { slot: "kslot" } },
    ...chainNodes(),
  ],
};

/** 引き金が渡す消失情報（`reason` は GpuContext の分岐に使われない）。 */
const LOSS: GPUDeviceLostInfo = { reason: "destroyed", message: "テストによる消失" };

const x = (): Tensor => ({ dtype: "f32", shape: [4, 3], data: new Float32Array(12) });

/** 2 行ぶんの `k`（`M = 2`）。 */
const k = (): Tensor => ({ dtype: "f32", shape: [1, 1, 2, 4], data: new Float32Array(8) });

/**
 * run が触る GPU の面だけを持つフェイク device と、`onSubmittedWorkDone`（印とフェンス）の差し替え口・消失の引き金。
 *
 * NOTE: GpuContext は実物（区間ロック・消失購読・`raceDeviceLost` は本番の実装 — `helpers/fake-gpu.ts` と同じ方針）。
 */
const pacedGpu = () => {
  let workDone: () => Promise<void> = () => Promise.resolve();
  const lost = Promise.withResolvers<GPUDeviceLostInfo>();
  const pass = {
    setPipeline: (): void => undefined,
    setBindGroup: (): void => undefined,
    dispatchWorkgroups: (): void => undefined,
    end: (): void => undefined,
  };
  const device = {
    lost: lost.promise,
    features: new Set<string>(),
    limits: {
      ...LIMITS,
      minStorageBufferOffsetAlignment: 256,
      minUniformBufferOffsetAlignment: 256,
    },
    destroy: (): void => undefined,
    pushErrorScope: (): void => undefined,
    popErrorScope: (): Promise<GPUError | null> => Promise.resolve(null),
    createShaderModule: () => ({}),
    createComputePipelineAsync: () => Promise.resolve({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder: () => ({
      beginComputePass: () => pass,
      copyBufferToBuffer: (): void => undefined,
      finish: () => ({}),
    }),
    createBuffer: (descriptor: GPUBufferDescriptor) => ({
      size: descriptor.size,
      usage: descriptor.usage,
      destroy: (): void => undefined,
      mapAsync: (): Promise<void> => Promise.resolve(),
      getMappedRange: () => new ArrayBuffer(descriptor.size),
      unmap: (): void => undefined,
    }),
    queue: {
      submit: (): void => undefined,
      writeBuffer: (): void => undefined,
      onSubmittedWorkDone: (): Promise<void> => workDone(),
    },
  };
  return {
    // DOM 型全体は再現しないため cast で渡す（テスト専用の境界）。
    gpu: new GpuContext(device as unknown as GPUDevice, readAdapterInfo({}), LIMITS, new Set()),
    /** 以後の `onSubmittedWorkDone` の戻りを差し替える。 */
    setWorkDone: (next: () => Promise<void>): void => {
      workDone = next;
    },
    lose: (): void => lost.resolve(LOSS),
  };
};

const session = async (gpu: GpuContext, graph: DeclarationJson): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME, {
    submitPolicy: ONE_PER_SUBMIT,
  });

Deno.test("印の待ちの間に device が失われたら、run はハングせず GpuDeviceLostError で決着し、消失の購読を基線へ戻す", async () => {
  const { gpu, setWorkDone, lose } = pacedGpu();
  const chain = await session(gpu, CHAIN_GRAPH);
  const baseline = gpu.pendingLostListeners;
  let markers = 0;
  // 印は解決しない（消失後の onSubmittedWorkDone が解決しない実装の形）。上限 + 1 本目の印を置いた後
  // （= run が最も古い印を待つ位置）で device を失う。
  setWorkDone(() => {
    markers += 1;
    if (markers === IN_FLIGHT_MAX_MARKERS + 1) setTimeout(lose, 0);
    return new Promise<void>(() => {});
  });

  const error = await assertRejects(() => chain.run({ x: x() }));
  assertInstanceOf(error, GpuDeviceLostError);
  assertEquals(markers, IN_FLIGHT_MAX_MARKERS + 1, "前提: run が印の待ちまで進んだ");
  assertEquals(gpu.pendingLostListeners, baseline, "消失の購読が積み残っている");
});

Deno.test("state を書く generation run が印の待ちの後で落ちたら、context を poison し、失敗をそのまま投げる", async () => {
  const { gpu, setWorkDone } = pacedGpu();
  const model = await session(gpu, STATE_THEN_CHAIN_GRAPH);
  const context = await model.createGenerationContext({ chunkLength: 2 });
  const failure = new Error("模擬: 印の待ちの失敗");
  // run の中で置く印だけを落とす（Session と context の構築は上の既定の解決のまま通す）。
  setWorkDone(() => Promise.reject(failure));

  const error = await assertRejects(() =>
    model.run({ k: k(), x: x() }, {}, { context, queryLength: 2 })
  );
  assertStrictEquals(error, failure);

  setWorkDone(() => Promise.resolve());
  // 待つ前に state を書く dispatch を submit しているので、同じ context では続けられない。
  await assertRejects(
    () => model.run({ k: k(), x: x() }, {}, { context, queryLength: 1 }),
    ExecutionError,
    "汚染された GenerationContext は使えない",
  );
  // 対照: poison は context に閉じる — 新しい context では同じ Session がそのまま回る。
  const fresh = await model.createGenerationContext({ chunkLength: 2 });
  const outputs = await model.run({ k: k(), x: x() }, {}, { context: fresh, queryLength: 2 });
  assert(Object.hasOwn(outputs, "y"));
});
