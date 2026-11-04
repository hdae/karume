// The check that compares the allocation sizes of weights and state with the device's absolute
// limits (failing **before** allocation).
//
// The motivation is the weakness of "detecting allocation failures relies entirely on per-block
// errorScopes" (ADR 0070 decision 4) — the errorScope net depends on the implementation's reporting
// quality (devices whose out-of-memory scope stays silent do exist — the Metal section of
// docs/known-issues.md), and even when it catches something the diagnostic is per block and comes
// only after several GiB have been transferred. A writeBuffer into an invalid buffer is a no-op
// without even a warning, so whatever slips through the net becomes "running with empty weights".
//
// Four things are pinned here:
// ① how sizes are taken — the payload for eligible classes, **the size after f32 expansion** for
//    ineligible ones, and a companion scale counts as one buffer too
// ② both limits are checked (maxStorageBufferBindingSize / maxBufferSize)
// ③ excesses are listed **all at once**, and exactly the limit passes (the boundary)
// ④ it really fails before allocation — the fake device's createBuffer is never called
//
// No real GPU is used. The limits differ per environment, and producing an excess on real hardware
// needs assets of several GiB, so the check itself can only be pinned deterministically with "a
// fake device with tightened limits".

import { test } from "vite-plus/test";
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
// Fixtures (graphs, a fake device)
// ---------------------------------------------------------------------------

/**
 * A single linear (the weight `w` is f16 and eligible = the payload goes onto the GPU as is).
 * w's payload is 8×4×2 = **64 bytes**; bias `b` is f32 with raw-byte residency, **32 bytes**.
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
 * A graph where 4 kinds of allocation coexist (one graph shows that the size comes from a different
 * place per residency class):
 * - `w` = i8-resident → payload **16 bytes** (8×2) + companion scale **32 bytes** (8 channels×4)
 * - `b` = raw-byte residency → **32 bytes**
 * - `g` = ineligible through a consumer other than a weight slot (mul) → the payload is 16 bytes,
 *   but **the allocation is the 32 bytes after expansion**
 *
 * The scale is larger than the payload on purpose (`[8,2]` has short rows and many channels) — an
 * implementation that does not count the scale trips here on "looking only at the payload and
 * passing".
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

/**
 * The storage of this file's 2 graphs (IR v2 declarations do not carry it — a binding-table field).
 */
const STORAGE: StorageMap = { w: "f16", g: "f16" };
const MIXED_STORAGE: StorageMap = { w: "int8-sym", g: "f16" };

/**
 * Declarations → residency plan (not a single byte of weight data is needed — the planner is a pure
 * function).
 */
const residencyOf = (
  graph: DeclarationJson,
  storage: StorageMap,
): ReturnType<typeof planWeightResidency> => planWeightResidency(mergeGraph(graph, storage));

/**
 * Limits with only the 2 limits specified (the check does not look at the other keys, so they are
 * filled with the WebGPU core defaults). The argument order is "binding limit, buffer limit", and
 * the two **can deliberately be inverted** (that is exactly the condition that detects whether both
 * are checked independently).
 */
const limits = (maxStorageBufferBindingSize: number, maxBufferSize: number): RequiredLimits => ({
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

/**
 * A counter of allocations reached (the observation point for the check failing before allocation).
 */
type AllocSpy = { createBuffer: number };

/**
 * A fake device that **always notices** a createBuffer call. The whole DOM type is not reproduced,
 * so it is passed with a cast (the same test-only boundary as tests/helpers/fake-gpu.ts).
 * errorScope and queue are only touched by the cleanup path, so pass-through no-ops suffice.
 */
const spyDevice = (spy: AllocSpy): GPUDevice =>
  ({
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    pushErrorScope: (): void => {},
    popErrorScope: (): Promise<GPUError | null> => Promise.resolve(null),
    createBuffer: (): GPUBuffer => {
      spy.createBuffer += 1;
      throw new Error("injected: createBuffer was called");
    },
    queue: {
      writeBuffer: (): void => {},
      onSubmittedWorkDone: (): Promise<undefined> => Promise.resolve(undefined),
    },
  }) as unknown as GPUDevice;

const spyGpu = (spy: AllocSpy, gpuLimits: RequiredLimits): GpuContext =>
  new GpuContext(spyDevice(spy), readAdapterInfo({}), gpuLimits, new Set());

// ---------------------------------------------------------------------------
// The range check of chunkLength (a GPU-independent pure function —
// src/runtime/generation-context.ts)
// ---------------------------------------------------------------------------

/**
 * The logical length is carried in a u32, so the limit is 0xffffffff (ADR 0066 decision 4 /
 * addendum 4). What is written here is not a **copy** of the implementation's private constant; it
 * observes the limit as it appears in the message (if the implementation changes the limit, the
 * assertStringIncludes below fails).
 */
const MAX_LOGICAL_LENGTH = 0xffffffff;

test("assertChunkLength passes only integers from 1 to the u32 limit", () => {
  // Inside the boundaries (both the lower and the upper end pass)
  assertChunkLength(1);
  assertChunkLength(MAX_LOGICAL_LENGTH);
  // 0 / negative / non-integer / over the limit / NaN all fail in the same check. The estimator
  // goes through the same check, so loosening it here would let "only the estimate return a normal
  // value for a setting the real construction rejects".
  for (const bad of [0, -1, 1.5, MAX_LOGICAL_LENGTH + 1, Number.NaN]) {
    const error = assertThrows(() => assertChunkLength(bad), ExecutionError, "chunkLength");
    assertStringIncludes(error.message, `chunkLength ${bad}`);
    assertStringIncludes(error.message, `1..${MAX_LOGICAL_LENGTH}`);
  }
});

// ---------------------------------------------------------------------------
// How sizes are taken (the pure-function check)
// ---------------------------------------------------------------------------

test("a payload over the limit is named with the allocation size of its residency class", () => {
  const residency = residencyOf(f16LinearGraph(), STORAGE);
  // w is 64 bytes (over the binding limit of 32); b is exactly 32 bytes and passes.
  const error = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(32, 64)),
    ExecutionError,
  );
  assertStringIncludes(error.message, "1 weight buffers");
  assertStringIncludes(
    error.message,
    "initializer 'w' payload (residency class f16, allocated 64 bytes): " +
      "exceeds maxStorageBufferBindingSize 32 bytes by 32 bytes",
  );
});

test("both limits, maxStorageBufferBindingSize and maxBufferSize, are checked", () => {
  const residency = residencyOf(f16LinearGraph(), STORAGE);
  // Not passed through even on a device where only the binding limit is loose (the size relation
  // the spec expects is broken).
  const bufferOnly = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(1024, 32)),
    ExecutionError,
  );
  assertStringIncludes(bufferOnly.message, "exceeds maxBufferSize 32 bytes by 32 bytes");
  assertEquals(
    bufferOnly.message.includes("maxStorageBufferBindingSize"),
    false,
    "a limit that is not exceeded does not appear in the message (it would bury which one to fix)",
  );

  // When both are exceeded, both appear on one line (so it reads that raising either one is not
  // enough).
  const both = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(16, 48)),
    ExecutionError,
  );
  assertStringIncludes(both.message, "exceeds maxStorageBufferBindingSize 16 bytes by 48 bytes");
  assertStringIncludes(both.message, "exceeds maxBufferSize 48 bytes by 16 bytes");
});

test("ineligible classes are checked at the size after f32 expansion, and a companion scale counts as one buffer", () => {
  const residency = residencyOf(mixedSeatGraph(), MIXED_STORAGE);
  const error = assertThrows(
    () => assertWeightsWithinLimits(residency, limits(16, 16)),
    ExecutionError,
  );
  // The ineligible `m.g` has a 16-byte payload (= exactly the limit), but what goes onto the GPU is
  // the 32 bytes after expansion.
  assertStringIncludes(
    error.message,
    "initializer 'g' payload (decoded to f32) (residency class expanded, allocated 32 bytes)",
  );
  // The scale is allocated separately from the payload, so it fails on the scale even when the
  // payload passes.
  assertStringIncludes(
    error.message,
    "initializer 'w' scale (residency class i8, allocated 32 bytes)",
  );
  assertEquals(
    error.message.includes("initializer 'w' payload"),
    false,
    "the 16-byte payload passes at exactly the limit (only the scale exceeds it)",
  );
});

test("excesses are listed all at once (not failing one at a time)", () => {
  const error = assertThrows(
    () => assertWeightsWithinLimits(residencyOf(mixedSeatGraph(), MIXED_STORAGE), limits(16, 16)),
    ExecutionError,
  );
  // The 3 buffers w.scale / b / g (only w.payload passes, at 16 bytes).
  assertStringIncludes(error.message, "3 weight buffers");
  assertEquals(
    error.message.split("\n").filter((line) => line.startsWith("  - initializer")).length,
    3,
    "one line per allocation that exceeds",
  );
});

test("exactly the limit passes (inside / outside the boundary)", () => {
  const mixed = residencyOf(mixedSeatGraph(), MIXED_STORAGE);
  // The largest allocation is 32 bytes (w.scale / b / g after expansion).
  assertWeightsWithinLimits(mixed, limits(32, 32));
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(31, 32)), ExecutionError);
  assertThrows(() => assertWeightsWithinLimits(mixed, limits(32, 31)), ExecutionError);

  const f16 = residencyOf(f16LinearGraph(), STORAGE);
  assertWeightsWithinLimits(f16, limits(64, 64));
  assertThrows(() => assertWeightsWithinLimits(f16, limits(63, 64)), ExecutionError);
});

// ---------------------------------------------------------------------------
// That it really runs at the Session construction entry (failing before allocation)
// ---------------------------------------------------------------------------

/**
 * The supplies for f16LinearGraph (the merge layer's check compares the declared and the real byte
 * counts).
 */
const f16LinearModel = (): BoundContainer => {
  const graph = f16LinearGraph();
  // Only `w` is passed real bytes (the merge layer checks that the f16 payload length follows from
  // the declaration).
  const supplied = autoTensors(graph, STORAGE).map((tensor) =>
    tensor.initializer === "w"
      ? {
          ...tensor,
          bytes: f16BytesFromBits(new Array(32).fill(f32ToF16Bits(0))),
        }
      : tensor,
  );
  return memoryModel(graph, supplied);
};

test("createSessionFromContainer fails weights over the limit before createBuffer", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(32, 64));
  const error = await assertRejects(
    () => createSessionFromContainer(gpu, f16LinearModel(), GRAPH_NAME),
    ExecutionError,
  );
  assertStringIncludes(error.message, "exceed the device limits (detected before allocation)");
  assertEquals(
    spy.createBuffer,
    0,
    "no allocation is reached (exactly the condition that detects it even on a device whose errorScope stays silent)",
  );
});

test("weights within the limits pass the check and proceed to allocation (the check does not reject everything)", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(1024, 1024));
  // The same model on the same path; only the limits differ. The injected failure shows that
  // allocation was reached.
  const error = await assertRejects(
    () => createSessionFromContainer(gpu, f16LinearModel(), GRAPH_NAME),
    Error,
  );
  assertStringIncludes(error.message, "injected: createBuffer was called");
  assertEquals(spy.createBuffer, 1);
});

// ---------------------------------------------------------------------------
// The state-slot side (complementing the existing binding-limit check — ADR 0066 addendum 5)
// ---------------------------------------------------------------------------

/**
 * A graph with a single state slot (`kv`) only (withStateReaders adds the reading `state_append`).
 */
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
  adopt: (allocate) => allocate(),
});

test("a state slot over maxBufferSize also fails before createBuffer", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  // Within the binding limit but over the buffer limit (passes through unless both are checked
  // independently).
  const gpu = spyGpu(spy, limits(1024, 16));
  const graph = stateGraph([8]);
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, graph), { chunkLength: 1 }),
    ExecutionError,
  );
  assertStringIncludes(
    error.message,
    "state 'kv': capacity [8] takes 32 bytes, which exceeds maxBufferSize 16 bytes",
  );
  assertEquals(spy.createBuffer, 0, "no allocation is reached");
});

test("a state slot over maxStorageBufferBindingSize also fails before createBuffer", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  // Within the buffer limit but over the binding limit (the counterpart of the maxBufferSize case
  // above — unless both are checked independently, only one of them works as a detector).
  const gpu = spyGpu(spy, limits(16, 1024));
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, stateGraph([8])), { chunkLength: 1 }),
    ExecutionError,
  );
  assertStringIncludes(
    error.message,
    "state 'kv': capacity [8] takes 32 bytes, which exceeds maxStorageBufferBindingSize 16 bytes",
  );
  assertEquals(spy.createBuffer, 0, "no allocation is reached");
});

test("a state slot within both limits proceeds to allocation (the check does not reject everything)", async () => {
  const spy: AllocSpy = { createBuffer: 0 };
  const gpu = spyGpu(spy, limits(1024, 1024));
  const error = await assertRejects(
    () => GenerationContext.create(stateHost(gpu, stateGraph([8])), { chunkLength: 1 }),
    Error,
  );
  assertStringIncludes(error.message, "injected: createBuffer was called");
  assertEquals(spy.createBuffer, 1);
});

// ---------------------------------------------------------------------------
// The estimate side (estimateGraphMemory) — that it has **the same boundaries** as the real
// construction (ADR 0066 addendum 5 / Q4-3)
//
// If some size were accepted only by the estimate, a caller that lets the estimate double as the
// value check of `--capacity` (like the CLI) would "print numbers at startup and fail on the first
// turn".
// ---------------------------------------------------------------------------

/**
 * The smallest graph that can go through **both** the estimate and the real construction. The
 * `stateGraph` above has a `state_append` input that does not meet the rank-4 contract (the real
 * construction passes because it does not plan the graph, but the estimate goes through
 * planGraph), so the paired tests use this one. The slot `kv` is [1,2,1,4] = 8 elements × 4 = **32
 * bytes** (the same size as `stateGraph([8])`, so the boundary numbers are the same too).
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

/**
 * Takes the estimate for the same graph with the same limits (the state slot is the single `kv` = 8
 * elements × 4 = 32 bytes).
 */
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

/**
 * Runs the real construction with the same limits and returns the message if it was rejected
 * (undefined if it proceeded to allocation).
 */
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

test("the binding-limit boundary is the same for the estimate and the real construction (1 byte below, both fail with the same message)", async () => {
  const graph = pairedStateGraph();
  // Exactly the limit (32 bytes) passes both — the real construction proceeds to allocation and
  // the estimate returns a number.
  assertEquals(stateEstimateAt(graph, 32, 1024), 32 + 8, "slot 32 + logical-length uniform 8");
  assertEquals(
    await createRejectionAt(graph, 32, 1024),
    undefined,
    "exactly the boundary proceeds to allocation",
  );
  // 1 byte below, both the estimate and the real construction fail. They share one check, so the
  // message is identical.
  const rejection = await createRejectionAt(graph, 31, 1024);
  const estimateError = assertThrows(() => stateEstimateAt(graph, 31, 1024), ExecutionError);
  assertEquals(estimateError.message, rejection);
  assertStringIncludes(
    estimateError.message,
    "state 'kv': capacity [1,2,1,4] takes 32 bytes, which exceeds maxStorageBufferBindingSize 31 bytes",
  );
});

test("the buffer-limit boundary is also the same for the estimate and the real construction (only one side is undercut, within the binding limit)", async () => {
  const graph = pairedStateGraph();
  assertEquals(stateEstimateAt(graph, 1024, 32), 32 + 8);
  assertEquals(await createRejectionAt(graph, 1024, 32), undefined);
  const rejection = await createRejectionAt(graph, 1024, 31);
  const estimateError = assertThrows(() => stateEstimateAt(graph, 1024, 31), ExecutionError);
  assertEquals(estimateError.message, rejection);
  assertStringIncludes(
    estimateError.message,
    "state 'kv': capacity [1,2,1,4] takes 32 bytes, which exceeds maxBufferSize 31 bytes",
  );
});

test("an estimate given no limits does not invent default values (callers without a device behave as before)", () => {
  const graph = pairedStateGraph();
  // Including when only one side is passed, only the passed fields affect the check (the
  // unspecified side is not looked at).
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
    "exceeds maxBufferSize 31 bytes",
  );
});
