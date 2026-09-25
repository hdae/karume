// Session の構築・実行の受け口のうち、GPU を取らずに回る門（実 GPU 側は
// gpu_runtime_executor_test.ts）と、容器の読み手が束縛の取れないシンボルを拒否する門。

import { assertEquals, assertInstanceOf, assertRejects, assertStringIncludes } from "@std/assert";
import { ContainerFormatError, writeHeader } from "../src/format/container/header.ts";
import { HEADER_BYTES } from "../src/format/container/limits.ts";
import { openContainer } from "../src/format/container/open.ts";
import { type IrGraph, parseIrDeclarationValue } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { type GenerationRun, Session, type SessionOptions } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
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
