// Session の構築・実行の受け口のうち、GPU を取らずに回る門（実 GPU 側は
// gpu_runtime_executor_test.ts）と、容器の読み手が束縛の取れないシンボルを拒否する門。

import { assertEquals, assertInstanceOf, assertRejects, assertStringIncludes } from "@std/assert";
import { ContainerFormatError, writeHeader } from "../src/format/container/header.ts";
import { HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";
import { type IrGraph, parseIrDeclarationValue } from "../src/format/ir.ts";
import {
  GpuContext,
  GpuDeviceLostError,
  readAdapterInfo,
  type RequiredLimits,
} from "../src/gpu/device.ts";
import { type GenerationRun, Session, type SessionOptions } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { buildSessionState } from "../src/runtime/session-build.ts";
import { SharedWeight, type WeightResidency } from "../src/runtime/weight-residency.ts";
import { chainGraph, chainTensors } from "./helpers/chain-graph.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import { GRAPH_NAME } from "./helpers/model-fixture.ts";
import { writeModelContainer } from "./helpers/container-write.ts";

// deno-lint-ignore no-explicit-any
const anyOf = (value: unknown): any => value;

Deno.test({
  name: "束縛が取れないシンボルは容器の読み手が受理しない（入力 shape に素の形で現れない）",
  fn: async () => {
    // 書き手は宣言をパースしてから書くので、壊れた宣言は容器に入らない。読み手の側の門
    // （`descriptor.ts` の宣言パース）を撃つには、**正しい krm を書いてからグラフ記述の
    // バイト列を壊す**しかない。
    const written = await writeModelContainer({
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(chainGraph()) },
      consts: [],
      weights: chainTensors(),
      assets: [],
      provenance: { license: "test" },
    }, {});
    const descriptorDoc = JSON.parse(new TextDecoder().decode(written.graphDescriptorBytes));
    const declaration = anyOf(descriptorDoc).graphs[GRAPH_NAME];
    // `S` は入力 shape に素の形で現れないので、どの実行時値からも束縛が取れない。
    declaration.symbols = ["S", "T"];
    declaration.values.h = { dtype: "f32", shape: ["S", 3] };
    const graphBytes = new TextEncoder().encode(JSON.stringify(descriptorDoc)) as Uint8Array<
      ArrayBuffer
    >;
    // 単一形は block の offset が part 0 の長さから決まるので、長さの変わる書き換えは
    // part 列で渡す（part 0 = ヘッダ + 2 文書）。
    const parts = [...written.parts];
    const model = written.modelDescriptorBytes;
    const part0 = new Uint8Array(
      new ArrayBuffer(HEADER_BYTES + graphBytes.byteLength + model.byteLength),
    );
    part0.set(
      writeHeader({
        kind: "model",
        version: 1,
        graphDescriptorLength: graphBytes.byteLength,
        modelDescriptorLength: model.byteLength,
      }),
      0,
    );
    part0.set(graphBytes, HEADER_BYTES);
    part0.set(model, HEADER_BYTES + graphBytes.byteLength);
    parts[0] = part0;
    await assertRejects(
      () => openContainer({ kind: "parts", parts }),
      ContainerFormatError,
      "束縛が取れない",
    );
  },
});

// ---------------------------------------------------------------------------
// 入力境界（型の外から来る JS の呼び手）— フェイク device の Session で撃つ。受け口の検査は
// dispatch を 1 本も出さない同期区間の話なので、GPU 実体は要らない。
// ---------------------------------------------------------------------------

/** 上限は門に掛からない値でよい（見たいのは受け口で、確保の寸法ではない）。 */
const LIMITS: RequiredLimits = {
  maxBufferSize: 1 << 20,
  maxStorageBufferBindingSize: 1 << 20,
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

/** 構築相（`buildSessionState`）が `minStorageBufferOffsetAlignment` を 1 回読むぶんだけ足す。 */
const fakeGpu = (): GpuContext =>
  new GpuContext(
    Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } }),
    readAdapterInfo({}),
    LIMITS,
    new Set(),
  );

/** ノード 0 本のグラフ（受け口の検査は積む中身を見ない）。 */
const emptyGraph = (): IrGraph =>
  mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: [],
    inputs: [],
    outputs: [],
    initializers: {},
    values: {},
    nodes: [],
  });

const buildSession = async (
  gpu: GpuContext,
  graph: IrGraph,
  options: SessionOptions,
): Promise<Session> => {
  async function* noShards(): AsyncGenerator<never> {}
  return await Session.build(gpu, graph, new Map(), noShards(), options);
};

/** 数値へ変換できるが、変換されたら数える値（診断が利用者の変換を呼んだかの観測点）。 */
const convertible = (): { readonly value: object; readonly conversions: () => number } => {
  let conversions = 0;
  return {
    value: {
      [Symbol.toPrimitive](): number {
        conversions += 1;
        return 4096;
      },
    },
    conversions: () => conversions,
  };
};

Deno.test("構築オプションの数値ノブの診断は利用者の変換を呼ばず、型外の値を ExecutionError で拒否する", async () => {
  for (const knob of ["planBackingBudgetBytes", "linearGemvRowsThreadTarget"] as const) {
    for (
      const make of [() => convertible(), () => ({
        value: Symbol(knob),
        conversions: () => 0,
      })]
    ) {
      const { value, conversions } = make();
      const options: SessionOptions = {};
      Object.defineProperty(options, knob, { value, enumerable: true });
      const refused = await assertRejects(
        () => buildSession(fakeGpu(), emptyGraph(), options),
        ExecutionError,
        knob,
      );
      // 非数値は型名だけを出す（値そのものを綴ると利用者の変換が走る）。
      assertStringIncludes(refused.message, `${knob} ${typeof value} `);
      assertEquals(conversions(), 0);
    }
  }
});

Deno.test("run の generation.commit の診断は利用者の変換を呼ばず、symbol でも ExecutionError で拒否する", async () => {
  const session = await buildSession(fakeGpu(), emptyGraph(), {});
  try {
    const object = convertible();
    for (const commit of [object.value, Symbol("deferred")]) {
      const generation = { context: {}, queryLength: 1, commit } as unknown as GenerationRun;
      const refused = await assertRejects(
        () => session.run({}, {}, generation),
        ExecutionError,
        "generation.commit",
      );
      assertStringIncludes(refused.message, `generation.commit ${typeof commit} `);
    }
    assertEquals(object.conversions(), 0);
  } finally {
    await session.dispose();
  }
});

Deno.test("run の generation の読みと context の内部面の取り出しは、同期 throw ではなく戻り Promise の reject になる", async () => {
  const session = await buildSession(fakeGpu(), emptyGraph(), {});
  try {
    const failure = new Error("getter が throw した");
    const throwingGetter = {
      get context(): never {
        throw failure;
      },
      queryLength: 1,
    } as unknown as GenerationRun;
    const missingContext = { context: null, queryLength: 1 } as unknown as GenerationRun;
    for (
      const [generation, expected] of [[throwingGetter, failure], [
        missingContext,
        undefined,
      ]] as const
    ) {
      // 同期 throw なら `.then` に届かずここで例外が抜ける（受け口の契約違反としてテストが落ちる）。
      const outcome = await session.run({}, {}, generation).then(
        () => "resolved",
        (cause: unknown) => cause,
      );
      if (expected === undefined) assertInstanceOf(outcome, TypeError);
      else assertEquals(outcome, expected);
    }
  } finally {
    await session.dispose();
  }
});

Deno.test("sharedWeights に SharedWeight でない値を渡すと、内部面を読む前に ExecutionError で拒否する", async () => {
  const graph = mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: [],
    inputs: [],
    outputs: ["w"],
    initializers: { w: { shared: true } },
    values: { w: { dtype: "f32", shape: [2] } },
    nodes: [],
  });
  for (const value of [{}, null, 1]) {
    await assertRejects(
      () =>
        buildSession(fakeGpu(), graph, {
          sharedWeights: { w: value } as unknown as SessionOptions["sharedWeights"],
        }),
      ExecutionError,
      "SharedWeight でない値",
    );
  }
});

// ---------------------------------------------------------------------------
// 構築と device の生存 — 構築は重みの完了待ちのほかに device を見ていなかったので、重みを 1 本も
// 上げないグラフ（initializer 無し・全て shared）は消失済み device の上でも成功していた。
// ---------------------------------------------------------------------------

/** 消失を後から起こせる {@link fakeGpu}。 */
const losableFakeGpu = (): { readonly gpu: GpuContext; readonly lose: () => void } => {
  const lost = Promise.withResolvers<GPUDeviceLostInfo>();
  return {
    gpu: new GpuContext(
      Object.assign(fakeDevice({ lost: lost.promise }), {
        limits: { minStorageBufferOffsetAlignment: 256 },
      }),
      readAdapterInfo({}),
      LIMITS,
      new Set(),
    ),
    lose: (): void => lost.resolve({ reason: "destroyed", message: "テストによる消失" }),
  };
};

/** 重みが全て借り物（shared 1 本）のグラフと、その常駐分類。 */
const allSharedGraph = (): IrGraph =>
  mergeGraph({
    format: "karume-ir",
    version: 2,
    requires: { ops: [] },
    symbols: [],
    inputs: [],
    outputs: ["w"],
    initializers: { w: { shared: true } },
    values: { w: { dtype: "f32", shape: [2] } },
    nodes: [],
  });
const ALL_SHARED_RESIDENCY: ReadonlyMap<string, WeightResidency> = new Map([
  ["w", { seat: "shared", eligible: false, i4Eligible: false, i2Eligible: false }],
]);

/** 借用の積み下ろしを数える貸し手の重み（実体は構築の照合を通る最小の形）。 */
const countedSharedWeight = (
  gpu: GpuContext,
): { readonly weight: SharedWeight; readonly borrowed: () => number } => {
  let borrowed = 0;
  const weight = new SharedWeight({
    initializer: "w",
    gpu,
    buffer: {} as unknown as GPUBuffer,
    resident: undefined,
    seat: "raw",
    rowAxis: undefined,
    codec: "f32",
    shape: [2],
    dtype: "f32",
    retain: (): void => {
      borrowed += 1;
    },
    release: (): void => {
      borrowed -= 1;
    },
  });
  return { weight, borrowed: () => borrowed };
};

Deno.test("消失済み・破棄要求済みの device では、重みを上げないグラフでも Session の構築が GpuDeviceLostError で reject する", async () => {
  const destroyed = fakeGpu();
  destroyed.destroy();
  await assertRejects(
    () => buildSession(destroyed, emptyGraph(), {}),
    GpuDeviceLostError,
    "Session の構築",
  );

  const { gpu, lose } = losableFakeGpu();
  lose();
  await gpu.device.lost;
  await assertRejects(
    () => buildSession(gpu, emptyGraph(), {}),
    GpuDeviceLostError,
    "Session の構築",
  );
  const { weight, borrowed } = countedSharedWeight(gpu);
  async function* noShards(): AsyncGenerator<never> {}
  await assertRejects(
    () =>
      Session.build(gpu, allSharedGraph(), ALL_SHARED_RESIDENCY, noShards(), {
        sharedWeights: { w: weight },
      }),
    GpuDeviceLostError,
    "Session の構築",
  );
  assertEquals(borrowed(), 0, "落ちた構築が借用を積んだまま");
});

Deno.test("構築の途中（重みの供給待ち）で device が失われたら、構築は成功で返らず借用を返す", async () => {
  const { gpu, lose } = losableFakeGpu();
  const { weight, borrowed } = countedSharedWeight(gpu);
  const gate = Promise.withResolvers<void>();
  // 供給は 1 本も出さずに終わるが、終わりを gate まで待たせる。
  const stalledShards: AsyncIterable<never> = {
    [Symbol.asyncIterator]: () => ({
      next: async (): Promise<IteratorResult<never>> => {
        await gate.promise;
        return { done: true, value: undefined };
      },
    }),
  };
  const building = Session.build(gpu, allSharedGraph(), ALL_SHARED_RESIDENCY, stalledShards, {
    sharedWeights: { w: weight },
  });
  // 借用は供給を待つ前に積む（構築中に貸し手が dispose される窓を塞ぐ）。
  while (borrowed() === 0) await Promise.resolve();
  lose();
  await gpu.device.lost;
  gate.resolve();
  await assertRejects(() => building, GpuDeviceLostError, "Session の構築");
  assertEquals(borrowed(), 0, "落ちた構築が借用を返していない");
});

Deno.test("構築は入口時点の options で決まり、構築中の書き換えは検査を経ない state を作らない", async () => {
  // 組合せの検査（ADR 0105 決定 1）は入口で 1 回だけ走る。構築は await を跨ぐので、待つ前に
  // 呼び手が options を書き換えると、検査では拒否される組合せ（sequential + packedStaticQuantize）の
  // state ができていた。submitPolicy は中身の書き換えも同じ形で効いていた。
  const policy = {
    timeBudgetMs: 100,
    initialChunkSize: 4,
    minChunkSize: 1,
    maxChunkSize: 1024,
  };
  const options: { -readonly [K in keyof SessionOptions]: SessionOptions[K] } = {
    linearGemvReduce: "sequential",
    submitPolicy: policy,
  };
  async function* noShards(): AsyncGenerator<never> {}
  const building = buildSessionState(fakeGpu(), emptyGraph(), new Map(), noShards(), options);
  options.packedStaticQuantize = true;
  options.fuseRmsNormAdd = true;
  policy.initialChunkSize = 64;
  const state = await building;
  assertEquals(
    state.packedStaticQuantize,
    false,
    "検査の後に書き換えた packedStaticQuantize が届いた",
  );
  assertEquals(state.fuseRmsNormAdd, false, "検査の後に書き換えた fuseRmsNormAdd が届いた");
  assertEquals(
    state.scheduler.stats.currentChunkSize,
    4,
    "検査の後に書き換えた submitPolicy の中身が届いた",
  );
});

Deno.test("構築の入口の写しは非列挙・prototype 由来の欄も読み、写す前と同じ値を拒否する", async () => {
  // 写しを spread で取ると自前の列挙可能な欄しか写らず、非列挙・継承の欄が黙って既定に化けて通る。
  const nonEnumerable: SessionOptions = {};
  Object.defineProperty(nonEnumerable, "stateAttentionReduce", { value: "bogus" });
  const inherited: SessionOptions = Object.create({ stateAttentionReduce: "bogus" });
  for (const options of [nonEnumerable, inherited]) {
    await assertRejects(
      () => buildSession(fakeGpu(), emptyGraph(), options),
      ExecutionError,
      "stateAttentionReduce",
    );
  }
});
