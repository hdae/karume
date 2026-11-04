// The checks of the failure paths of the mid-run submission wait (ADR 0123 — the run `await`s the
// Promise `executeBakedPlanPaced` returns). Exercised on a Session over a fake device (no GPU
// needed).
//
// Two points are pinned (the failure paths of the asynchronous branch the wait added to run — the
// success paths are covered by the real-GPU check in `gpu_submit_backpressure_test.ts` and by
// `runtime_recipe_test.ts`):
//
// ① if the device is lost while waiting on a marker, run settles with `GpuDeviceLostError`
//    instead of hanging, and returns the loss subscriptions to the baseline (ADR 0054 /
//    `raceDeviceLost` — an implementation may never resolve `onSubmittedWorkDone` after a loss).
// ② if a generation run that writes state fails after the wait, it poisons the context (the
//    failure atomicity of ADR 0066 addendum 3 — everything unsubmitted is flushed before waiting,
//    so a failure across the wait is "a failure after submit". Under-poisoning is silent
//    corruption where generation continues on broken KV).
//
// NOTE: the fake device has only the GPU surfaces run touches (pipelines, bind groups, command
// sequences, buffers, queue). It computes no values — what is checked is the failure's settlement
// and cleanup, not the output values.

import { platformObject } from "@karume/test-support/webgpu";
import { test } from "vite-plus/test";
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

/**
 * Any limits that do not trip a check will do (what matters is the failure path, not allocation
 * sizes).
 */
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

/**
 * 1 dispatch = 1 submit (so the marker positions follow from the dispatch count — the same policy
 * as the real-GPU check).
 */
const ONE_PER_SUBMIT: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 1,
  minChunkSize: 1,
  maxChunkSize: 1,
};

/**
 * The number of add steps that just reaches limit + 1 outstanding markers (so run waits on the
 * oldest marker between steps).
 */
const CHAIN = IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 1) + 8;

/** y = x + x + … + x ({@link CHAIN} adds chained). */
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

/** A graph that writes no state ({@link CHAIN} adds). */
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

/** A graph that writes state first (`state_append`) and then stacks {@link CHAIN} adds. */
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

/** The loss info the trigger passes (`reason` is not used by GpuContext's branching). */
const LOSS: GPUDeviceLostInfo = platformObject<GPUDeviceLostInfo>({
  reason: "destroyed",
  message: "lost by the test",
});

const x = (): Tensor => ({ dtype: "f32", shape: [4, 3], data: new Float32Array(12) });

/** `k` for 2 rows (`M = 2`). */
const k = (): Tensor => ({ dtype: "f32", shape: [1, 1, 2, 4], data: new Float32Array(8) });

/**
 * A fake device with only the GPU surfaces run touches, plus a hook to replace
 * `onSubmittedWorkDone` (markers and fences) and a loss trigger.
 *
 * NOTE: GpuContext is the real one (the scope lock, loss subscription and `raceDeviceLost` are the
 * production implementation — the same approach as `helpers/fake-gpu.ts`).
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
    // The full DOM types are not reproduced, so it is passed with a cast (a test-only boundary).
    gpu: new GpuContext(device as unknown as GPUDevice, readAdapterInfo({}), LIMITS, new Set()),
    /** Replaces the return of subsequent `onSubmittedWorkDone` calls. */
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

test("if the device is lost while waiting on a marker, run settles with GpuDeviceLostError instead of hanging and returns the loss subscriptions to the baseline", async () => {
  const { gpu, setWorkDone, lose } = pacedGpu();
  const chain = await session(gpu, CHAIN_GRAPH);
  const baseline = gpu.pendingLostListeners;
  let markers = 0;
  // Markers never resolve (the shape of an implementation whose onSubmittedWorkDone never resolves
  // after a loss). The device is lost after the limit + 1th marker is placed (= the position where
  // run waits on the oldest marker).
  setWorkDone(() => {
    markers += 1;
    if (markers === IN_FLIGHT_MAX_MARKERS + 1) setTimeout(lose, 0);
    return new Promise<void>(() => {});
  });

  const error = await assertRejects(() => chain.run({ x: x() }));
  assertInstanceOf(error, GpuDeviceLostError);
  assertEquals(
    markers,
    IN_FLIGHT_MAX_MARKERS + 1,
    "precondition: run got as far as waiting on a marker",
  );
  assertEquals(gpu.pendingLostListeners, baseline, "loss subscriptions were left behind");
});

test("if a generation run that writes state fails after waiting on a marker, it poisons the context and throws the failure as is", async () => {
  const { gpu, setWorkDone } = pacedGpu();
  const model = await session(gpu, STATE_THEN_CHAIN_GRAPH);
  const context = await model.createGenerationContext({ chunkLength: 2 });
  const failure = new Error("simulated: marker wait failure");
  // Only the markers placed inside run fail (building the Session and context goes through with the
  // default resolution above).
  setWorkDone(() => Promise.reject(failure));

  const error = await assertRejects(() =>
    model.run({ k: k(), x: x() }, {}, { context, queryLength: 2 }),
  );
  assertStrictEquals(error, failure);

  setWorkDone(() => Promise.resolve());
  // The dispatch that writes state was submitted before the wait, so the same context cannot
  // continue.
  await assertRejects(
    () => model.run({ k: k(), x: x() }, {}, { context, queryLength: 1 }),
    ExecutionError,
    "a poisoned GenerationContext cannot be used",
  );
  // Control: the poison stays within the context — the same Session runs fine on a new context.
  const fresh = await model.createGenerationContext({ chunkLength: 2 });
  const outputs = await model.run({ k: k(), x: x() }, {}, { context: fresh, queryLength: 2 });
  assert(Object.hasOwn(outputs, "y"));
});
