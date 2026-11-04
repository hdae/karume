// The checks of Session's build / run entry points that run without a GPU (the real-GPU side is
// gpu_runtime_executor_test.ts), and the check that the container reader rejects unbindable
// symbols.

import { platformObject } from "@karume/test-support/webgpu";
import { test } from "vite-plus/test";
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

test("the container reader rejects a symbol that cannot be bound (one that does not appear bare in an input shape)", async () => {
  // The writer parses declarations before writing, so a broken declaration never enters a
  // container. The only way to exercise the reader-side check (declaration parsing in
  // `descriptor.ts`) is to **write a correct krm and then corrupt the bytes of the graph
  // descriptor**.
  const written = await writeModelContainer(
    {
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(chainGraph()) },
      consts: [],
      weights: chainTensors(),
      assets: [],
      provenance: { license: "test" },
    },
    {},
  );
  const descriptorDoc = JSON.parse(new TextDecoder().decode(written.graphDescriptorBytes));
  const declaration = anyOf(descriptorDoc).graphs[GRAPH_NAME];
  // `S` does not appear bare in any input shape, so no runtime value can bind it.
  declaration.symbols = ["S", "T"];
  declaration.values.h = { dtype: "f32", shape: ["S", 3] };
  const graphBytes = new TextEncoder().encode(
    JSON.stringify(descriptorDoc),
  ) as Uint8Array<ArrayBuffer>;
  // In the single form a block's offset is determined by the length of part 0, so a rewrite that
  // changes the length is passed as a part sequence (part 0 = header + 2 documents).
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
    "cannot be bound",
  );
});

// ---------------------------------------------------------------------------
// Input boundary (JS callers coming from outside the types) — exercised on a Session over a fake
// device. The entry checks happen in a synchronous part that issues no dispatch, so no real GPU is
// needed.
// ---------------------------------------------------------------------------

/**
 * Any limits that do not trip a check will do (what matters is the entry point, not allocation
 * sizes).
 */
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

/**
 * Adds only what the build phase (`buildSessionState`) reads once:
 * `minStorageBufferOffsetAlignment`.
 */
const fakeGpu = (): GpuContext =>
  new GpuContext(
    Object.assign(fakeDevice(), { limits: { minStorageBufferOffsetAlignment: 256 } }),
    readAdapterInfo({}),
    LIMITS,
    new Set(),
  );

/** A graph with 0 nodes (the entry checks do not look at what is enqueued). */
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

/**
 * A value convertible to a number that counts its conversions (the observation point for whether a
 * diagnostic called the caller's conversion).
 */
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

test("diagnostics for numeric build-option knobs do not call the caller's conversions, and out-of-type values are rejected with ExecutionError", async () => {
  for (const knob of [
    "planBackingBudgetBytes",
    "linearGemvRowsThreadTarget",
    "attentionRowBlockMaxBytes",
  ] as const) {
    for (const make of [
      () => convertible(),
      () => ({
        value: Symbol(knob),
        conversions: () => 0,
      }),
    ]) {
      const { value, conversions } = make();
      const options: SessionOptions = {};
      Object.defineProperty(options, knob, { value, enumerable: true });
      const refused = await assertRejects(
        () => buildSession(fakeGpu(), emptyGraph(), options),
        ExecutionError,
        knob,
      );
      // A non-number reports only its type name (spelling out the value itself would run the
      // caller's conversion).
      assertStringIncludes(refused.message, `${knob} ${typeof value} `);
      assertEquals(conversions(), 0);
    }
  }
});

test("the diagnostic for run's generation.commit does not call the caller's conversions and rejects even a symbol with ExecutionError", async () => {
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

test("reading run's generation and extracting the context's internal surface reject the returned Promise instead of throwing synchronously", async () => {
  const session = await buildSession(fakeGpu(), emptyGraph(), {});
  try {
    const failure = new Error("the getter threw");
    const throwingGetter = {
      get context(): never {
        throw failure;
      },
      queryLength: 1,
    } as unknown as GenerationRun;
    const missingContext = { context: null, queryLength: 1 } as unknown as GenerationRun;
    for (const [generation, expected] of [
      [throwingGetter, failure],
      [missingContext, undefined],
    ] as const) {
      // A synchronous throw would escape here without reaching `.then` (the test fails as an
      // entry-point contract violation).
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

test("passing a value that is not a SharedWeight in sharedWeights is rejected with ExecutionError before the internal surface is read", async () => {
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
      "not a SharedWeight",
    );
  }
});

// ---------------------------------------------------------------------------
// Build and device liveness — the build looked at nothing but the weights' completion, so a graph
// that uploads no weight (no initializers, all shared) succeeded even on a lost device.
// ---------------------------------------------------------------------------

/** A {@link fakeGpu} whose loss can be triggered later. */
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
    lose: (): void =>
      lost.resolve(
        platformObject<GPUDeviceLostInfo>({ reason: "destroyed", message: "lost by the test" }),
      ),
  };
};

/** A graph whose weights are all borrowed (one shared), and its residency classification. */
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

/**
 * A lender's weight that counts borrow acquisitions and releases (the instance is the minimal shape
 * that passes the build's matching).
 */
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

test("on a lost or destroy-requested device, Session construction rejects with GpuDeviceLostError even for a graph that uploads no weights", async () => {
  const destroyed = fakeGpu();
  destroyed.destroy();
  await assertRejects(
    () => buildSession(destroyed, emptyGraph(), {}),
    GpuDeviceLostError,
    "Session construction",
  );

  const { gpu, lose } = losableFakeGpu();
  lose();
  await gpu.device.lost;
  await assertRejects(
    () => buildSession(gpu, emptyGraph(), {}),
    GpuDeviceLostError,
    "Session construction",
  );
  const { weight, borrowed } = countedSharedWeight(gpu);
  async function* noShards(): AsyncGenerator<never> {}
  await assertRejects(
    () =>
      Session.build(gpu, allSharedGraph(), ALL_SHARED_RESIDENCY, noShards(), {
        sharedWeights: { w: weight },
      }),
    GpuDeviceLostError,
    "Session construction",
  );
  assertEquals(borrowed(), 0, "the failed build still holds its borrows");
});

test("if the device is lost mid-build (waiting for weight supply), the build does not return success and returns its borrows", async () => {
  const { gpu, lose } = losableFakeGpu();
  const { weight, borrowed } = countedSharedWeight(gpu);
  const gate = Promise.withResolvers<void>();
  // The supply ends without issuing anything, but its end is held back until the gate.
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
  // Borrows are taken before waiting for the supply (closing the window where the lender is
  // disposed mid-build).
  while (borrowed() === 0) await Promise.resolve();
  lose();
  await gpu.device.lost;
  gate.resolve();
  await assertRejects(() => building, GpuDeviceLostError, "Session construction");
  assertEquals(borrowed(), 0, "the failed build did not return its borrows");
});

test("if the device is lost after the build, a run / enqueue with 0 outputs rejects with GpuDeviceLostError instead of turning into success", async () => {
  // A graph with 0 outputs has no read-back, so the body's failure detection (errorScope,
  // completion wait) alone is not guaranteed to catch the loss. The device is checked in the
  // intake's synchronous part.
  const destroyed = fakeGpu();
  const onDestroyed = await buildSession(destroyed, emptyGraph(), {});
  destroyed.destroy();
  await assertRejects(() => onDestroyed.run({}), GpuDeviceLostError, "run acceptance");
  await onDestroyed.dispose();

  // The enqueue goes into a scope opened before the loss (after the loss, opening a scope itself
  // fails). A scope holds the errorScope scope lock, so it is not left open on the same device as
  // the run.
  const enqueued = fakeGpu();
  const onEnqueued = await buildSession(enqueued, emptyGraph(), {});
  const opened = await enqueued.beginBatch();
  enqueued.destroy();
  await assertRejects(
    () => onEnqueued.enqueue({}, { batch: opened }),
    GpuDeviceLostError,
    "enqueue acceptance",
  );
  await assertRejects(
    () => onEnqueued.enqueueRead({}, { batch: opened }).admitted,
    GpuDeviceLostError,
    "enqueue acceptance",
  );
  // The scope's own settlement also fails on the loss (what this checks is intake; cleaning up the
  // scope is another check).
  await assertRejects(() => opened.finish(), GpuDeviceLostError);
  await onEnqueued.dispose();

  const { gpu, lose } = losableFakeGpu();
  const onLost = await buildSession(gpu, emptyGraph(), {});
  lose();
  await gpu.device.lost;
  await assertRejects(() => onLost.run({}), GpuDeviceLostError, "run acceptance");
  await onLost.dispose();
});

test("the build is determined by the options at entry, and rewriting them mid-build does not create unchecked state", async () => {
  // The combination check (ADR 0105 decision 1) runs only once, at entry. The build spans awaits,
  // so if the caller rewrote options before the wait, state for a combination the check rejects
  // (sequential + packedStaticQuantize) was created. Rewriting submitPolicy's contents had the same
  // effect.
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
    "packedStaticQuantize rewritten after the check got through",
  );
  assertEquals(state.fuseRmsNormAdd, false, "fuseRmsNormAdd rewritten after the check got through");
  assertEquals(
    state.scheduler.stats.currentChunkSize,
    4,
    "submitPolicy contents rewritten after the check got through",
  );
});

test("attentionRowBlockMaxBytes reaches the Session state from the entry copy, and rewrites after the check do not", async () => {
  // The budget is a per-Session invariant that sets the number of row blocks (ADR 0124). Derived
  // plan keys do not include the budget, so if the value moved after the build, the same key would
  // point to a plan with a different count.
  const options: { -readonly [K in keyof SessionOptions]: SessionOptions[K] } = {
    attentionRowBlockMaxBytes: 64 * 1024 * 1024,
  };
  async function* noShards(): AsyncGenerator<never> {}
  const building = buildSessionState(fakeGpu(), emptyGraph(), new Map(), noShards(), options);
  options.attentionRowBlockMaxBytes = 1;
  assertEquals((await building).attentionRowBlockMaxBytes, 64 * 1024 * 1024);
  // Omitted means no budget (the binding limit as is).
  const plain = await buildSessionState(fakeGpu(), emptyGraph(), new Map(), noShards(), {});
  assertEquals(plain.attentionRowBlockMaxBytes, undefined);
});

test("the build entry's copy also reads non-enumerable and prototype fields, and rejects the same values as before copying", async () => {
  // Taking the copy with spread copies only own enumerable fields, so non-enumerable and inherited
  // fields silently turn into defaults and pass.
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
