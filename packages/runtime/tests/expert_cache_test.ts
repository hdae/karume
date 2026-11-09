// The exact expert cache without a GPU (ADR 0146): the option's checks, the planner's pool sizes and
// refusals, where every expert lies in a container (pieces of uneven expert counts, whole and split
// scales), the victim choice of a load, and the Session's surface over a fake device — the refusals
// of the plain run surfaces, the generation-run rule, a load's writes and counters, the digest check
// of an unverified source, and the unusable state after a failed GPU section. The GPU paths are in
// gpu_expert_linear_pool_test.ts (the kernel) and gpu_expert_cache_test.ts (bit identity).
//
// Expected sizes and slots are written out by hand (deriving them from the code under test would
// make the checks tautological).

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { type BlockSource, openContainer, partsSource } from "../src/format/container/open.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { GpuValidationError } from "../src/gpu/error-scope.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import {
  createSessionFromContainer,
  type ExpertRouting,
  type GenerationRun,
  prepareContainer,
  type Session,
  type SessionOptions,
} from "../src/runtime/executor.ts";
import {
  type ExpertGraphPlan,
  planExpertGraph,
  planExpertLoad,
  planExpertStore,
} from "../src/runtime/expert-plan.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { estimateGraphMemory } from "../src/runtime/estimate.ts";
import { sessionOptionsViolation } from "../src/runtime/session-options-check.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import { writeModelContainer } from "./helpers/container-write.ts";
import {
  EXPERT_GROUP,
  expertGraph,
  type ExpertGraphShape,
  type ExpertStorage,
  expertTensors,
} from "./helpers/expert-graph.ts";
import { mergeGraph, type StorageMap, type StorageSpec } from "./helpers/merged-graph.ts";
import { type DeclarationJson, GRAPH_NAME, memoryTensorOf } from "./helpers/model-fixture.ts";

// deno-lint-ignore no-explicit-any
const anyOf = (value: unknown): any => value;

/**
 * 16 experts, top-4, width 64, F 32, two layers: one gate_up expert is 2 · 32 · 64 / 2 = 2048 code
 * bytes and 64 · 64 / 32 = 128 steps; one down expert 64 · 32 / 2 = 1024 code bytes and 64 steps.
 */
const SHAPE: ExpertGraphShape = { experts: 16, picks: 4, width: 64, ffn: 32, layers: 2 };

const planOf = (
  declaration: DeclarationJson,
  storage: ExpertStorage,
  slots: number,
): ExpertGraphPlan => {
  const graph = mergeGraph(declaration, storageOf(declaration, storage));
  return planExpertGraph(graph, planWeightResidency(graph), { slots });
};

/** The storage map of the expert weights (`mergeGraph` fills zero bytes of the declared length). */
const storageOf = (declaration: DeclarationJson, storage: ExpertStorage): StorageMap => {
  const spec: StorageSpec =
    storage === "f32"
      ? "f32"
      : { codec: storage === "i4" ? "int4-sym-g" : "int4-sym-g-f16", groupSize: EXPERT_GROUP };
  const map: Record<string, StorageSpec> = {};
  for (const name of Object.keys(declaration.initializers)) {
    if (name.startsWith("gate_up") || name.startsWith("down")) map[name] = spec;
  }
  return map;
};

describe("the expertCache option's own checks (no graph)", () => {
  it("accepts an absent option and a positive integer of slots", () => {
    assertEquals(sessionOptionsViolation({}), undefined);
    assertEquals(sessionOptionsViolation({ expertCache: { slots: 64 } }), undefined);
  });

  it("accepts a positive integer of host bytes, and its absence", () => {
    assertEquals(
      sessionOptionsViolation({ expertCache: { slots: 64, hostBytes: 14 * 2 ** 30 } }),
      undefined,
    );
    assertEquals(
      sessionOptionsViolation({ expertCache: { slots: 64, hostBytes: undefined } }),
      undefined,
    );
  });

  for (const [label, value] of [
    ["0 slots", { slots: 0 }],
    ["negative slots", { slots: -1 }],
    ["fractional slots", { slots: 1.5 }],
    ["NaN slots", { slots: Number.NaN }],
    ["string slots", { slots: "8" }],
    ["null", null],
    ["a bare number", 8],
    ["0 host bytes", { slots: 8, hostBytes: 0 }],
    ["negative host bytes", { slots: 8, hostBytes: -4096 }],
    ["fractional host bytes", { slots: 8, hostBytes: 4096.5 }],
    ["NaN host bytes", { slots: 8, hostBytes: Number.NaN }],
    ["string host bytes", { slots: 8, hostBytes: "4096" }],
    ["null host bytes", { slots: 8, hostBytes: null }],
  ] as const) {
    it(`refuses ${label} with a message naming the option`, () => {
      const violation = sessionOptionsViolation({ expertCache: anyOf(value) });
      assert(violation !== undefined, `${label} was accepted`);
      assertStringIncludes(violation, "options.expertCache");
    });
  }
});

describe("the expert cache's plan of a graph", () => {
  it("sizes every expert weight as a pool of K experts and appends every routed index to the outputs", () => {
    const plan = planOf(expertGraph(SHAPE), "i4f16", 8);
    assertEquals(
      plan.layers.map((layer) => layer.index),
      ["index0", "index1"],
    );
    assertEquals(plan.declaredOutputs, ["out"]);
    assertEquals(plan.graph.outputs, ["out", "index0", "index1"]);
    assertEquals([plan.slots, plan.experts, plan.picks, plan.tableBytes], [8, 16, 4, 64]);
    const [gateUp, down] = plan.layers[0].weights;
    // f16 steps: 128 · 2 bytes for gate_up, 64 · 2 for down.
    assertEquals([gateUp.name, gateUp.codesBytes, gateUp.stepsBytes], ["gate_up0", 2048, 256]);
    assertEquals([down.name, down.codesBytes, down.stepsBytes], ["down0", 1024, 128]);
    assertEquals(plan.residency.get("gate_up0"), {
      seat: "i4f16",
      payloadBytes: 8 * 2048,
      scaleBytes: 8 * 256,
      groupSize: EXPERT_GROUP,
    });
    assertEquals(plan.residency.get("down1"), {
      seat: "i4f16",
      payloadBytes: 8 * 1024,
      scaleBytes: 8 * 128,
      groupSize: EXPERT_GROUP,
    });
    // The routers keep their residency.
    assertEquals(plan.residency.get("router_w0"), { seat: "raw", payloadBytes: 16 * 64 * 4 });
  });

  it("lays one expert's entry out as each weight's codes then steps, and sizes the ring by it", () => {
    const plan = planOf(expertGraph(SHAPE), "i4f16", 8);
    const [gateUp, down] = plan.layers[0].weights;
    assertEquals(
      [gateUp.entryCodesAt, gateUp.entryStepsAt, down.entryCodesAt, down.entryStepsAt],
      [0, 2048, 2304, 3328],
    );
    assertEquals(
      plan.layers.map((layer) => layer.entryBytes),
      [3456, 3456],
    );
    // 2 layers · 8 slots = 16 experts at most in one load, under the 32 of the ring.
    assertEquals(plan.ring, { slots: 16, slotBytes: 3456 });
    // f32 has no steps: down's codes follow gate_up's.
    const [f32GateUp, f32Down] = planOf(expertGraph(SHAPE), "f32", 4).layers[0].weights;
    assertEquals([f32GateUp.entryCodesAt, f32Down.entryCodesAt], [0, 64 * 64 * 4]);
    // gate_up [64, 64] and down [64, 32] f32 values.
    assertEquals(
      planOf(expertGraph(SHAPE), "f32", 4).layers[0].entryBytes,
      64 * 64 * 4 + 64 * 32 * 4,
    );
    assertEquals(plan.hostBytes, undefined);
  });

  it("keeps a host budget that holds an expert, and refuses one that holds none", () => {
    const declaration = expertGraph(SHAPE);
    const graph = mergeGraph(declaration, storageOf(declaration, "i4f16"));
    const residency = planWeightResidency(graph);
    assertEquals(planExpertGraph(graph, residency, { slots: 8, hostBytes: 3456 }).hostBytes, 3456);
    assertThrows(
      () => planExpertGraph(graph, residency, { slots: 8, hostBytes: 3455 }),
      ExecutionError,
      "hostBytes 3455 holds no expert (one expert of a routed layer is 3456 bytes)",
    );
  });

  it("sizes f32 and i4 (f32 steps) pools from their own storage", () => {
    const f32 = planOf(expertGraph(SHAPE), "f32", 4).layers[0].weights[0];
    assertEquals([f32.storage, f32.codesBytes, f32.stepsBytes], ["f32", 64 * 64 * 4, 0]);
    const i4 = planOf(expertGraph(SHAPE), "i4", 4).layers[0].weights[0];
    assertEquals([i4.storage, i4.codesBytes, i4.stepsBytes], ["i4", 2048, 128 * 4]);
  });

  it("reads an index that is already a declared output back once", () => {
    const declaration = expertGraph(SHAPE);
    declaration.outputs = ["out", "index1"];
    assertEquals(planOf(declaration, "f32", 4).graph.outputs, ["out", "index1", "index0"]);
  });

  it("accepts K from the topk k to E, and refuses K outside it", () => {
    assertEquals(planOf(expertGraph(SHAPE), "f32", 4).slots, 4);
    assertEquals(planOf(expertGraph(SHAPE), "f32", 16).slots, 16);
    for (const slots of [3, 17]) {
      assertThrows(
        () => planOf(expertGraph(SHAPE), "f32", slots),
        ExecutionError,
        "must be from the topk k 4 to the experts E 16",
      );
    }
  });

  it("refuses a graph without expert_linear", () => {
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["topk"] },
      symbols: ["T"],
      inputs: [{ name: "x", dtype: "f32", shape: ["T", 8] }],
      outputs: ["index"],
      initializers: {},
      values: {
        picked: { dtype: "f32", shape: ["T", 2] },
        index: { dtype: "i32", shape: ["T", 2] },
      },
      nodes: [{ op: "topk", ins: ["x"], outs: ["picked", "index"], attrs: { k: 2 } }],
    };
    const graph = mergeGraph(declaration, {});
    assertThrows(
      () => planExpertGraph(graph, planWeightResidency(graph), { slots: 2 }),
      ExecutionError,
      "needs a graph with expert_linear",
    );
  });

  it("refuses an expert weight that another op reads or that is a graph output (a pool holds slots, not experts)", () => {
    const read = expertGraph(SHAPE);
    read.values["twice"] = { dtype: "f32", shape: [16, 64, 32] };
    read.nodes.push({ op: "mul", ins: ["down0", "down0"], outs: ["twice"], attrs: {} });
    assertThrows(() => planOf(read, "f32", 4), ExecutionError, "is also input 0 of nodes[");
    const output = expertGraph(SHAPE);
    output.outputs = ["out", "gate_up1"];
    assertThrows(() => planOf(output, "f32", 4), ExecutionError, "is a graph output");
  });
});

describe("the estimate under the expert cache", () => {
  it("counts the pools and the tables in place of the expert weights, and the routing readback", () => {
    const declaration = expertGraph(SHAPE);
    const graph = mergeGraph(declaration, storageOf(declaration, "i4f16"));
    const residency = planWeightResidency(graph);
    const plain = estimateGraphMemory(graph, residency, { bindings: { T: 2 } });
    const cached = estimateGraphMemory(graph, residency, {
      bindings: { T: 2 },
      expertCache: { slots: 8 },
    });
    // One layer's experts: gate_up 16 · (2048 + 256), down 16 · (1024 + 128) = 55,296 bytes.
    assertEquals(plain.resident.weights.compressedBytes, 2 * 55296);
    assertEquals(cached.resident.weights.compressedBytes, 2 * 27648);
    // The routers (16 · 64 · 4 + 16 · 4 per layer) and, with the cache, two 64-byte tables.
    assertEquals(plain.resident.weights.uncompressedBytes, 2 * 4160);
    assertEquals(cached.resident.weights.uncompressedBytes, 2 * 4160 + 2 * 64);
    // x and out are [2, 64] f32; the cache also reads both [2, 4] i32 indices back.
    assertEquals(plain.scenarios[0].ioBytes, 1024);
    assertEquals(cached.scenarios[0].ioBytes, 1024 + 2 * 32);
  });

  it("counts neither the staging ring nor the host tier, and names both as unaccounted", () => {
    const declaration = expertGraph(SHAPE);
    const graph = mergeGraph(declaration, storageOf(declaration, "i4f16"));
    const residency = planWeightResidency(graph);
    const plain = estimateGraphMemory(graph, residency, { bindings: { T: 2 } });
    const cached = estimateGraphMemory(graph, residency, {
      bindings: { T: 2 },
      expertCache: { slots: 8 },
    });
    const held = estimateGraphMemory(graph, residency, {
      bindings: { T: 2 },
      expertCache: { slots: 8, hostBytes: 2 * 16 * 3456 },
    });
    assertEquals(held.resident, cached.resident);
    assertEquals(held.peakAccountedBytes, cached.peakAccountedBytes);
    assertEquals(cached.unaccounted.slice(0, plain.unaccounted.length), plain.unaccounted);
    const extra = (report: typeof cached) => report.unaccounted.slice(plain.unaccounted.length);
    assertEquals(extra(cached).length, 1);
    assertStringIncludes(extra(cached)[0], "staging ring");
    assertEquals(extra(held).length, 2);
    assertStringIncludes(extra(held)[1], "host tier");
  });

  it("refuses a value construction refuses, with the same message", () => {
    const declaration = expertGraph(SHAPE);
    const graph = mergeGraph(declaration, storageOf(declaration, "f32"));
    assertThrows(
      () =>
        estimateGraphMemory(graph, planWeightResidency(graph), {
          bindings: { T: 1 },
          expertCache: { slots: 3 },
        }),
      ExecutionError,
      "must be from the topk k 4 to the experts E 16",
    );
  });
});

/** The stored bytes of one expert weight, as the writer was given them. */
const storedOf = (tensors: ReturnType<typeof expertTensors>, name: string) => {
  const tensor = tensors.find((candidate) => candidate.initializer === name);
  if (tensor === undefined) throw new Error(`no tensor ${name}`);
  return { codes: tensor.bytes, steps: tensor.encoding.scale?.bytes };
};

describe("where every expert lies in a container (derived once from the descriptor)", () => {
  // 6144-byte blocks: gate_up's codes split 3 experts a piece (the last piece holds 1), down's 6 (the
  // last holds 4), both scales whole. 2048-byte blocks: gate_up's codes 1 expert a piece and its
  // 4096-byte scale split along them (rule ③), down's codes 2 a piece and its scale whole.
  for (const [blockBytes, pieces] of [
    [6144, { gate_up0: 6, down0: 3 }],
    [2048, { gate_up0: 16, down0: 8 }],
  ] as const) {
    it(`reads each expert's codes and steps exactly from the pieces of ${blockBytes}-byte blocks`, async () => {
      const tensors = expertTensors(SHAPE, "i4f16");
      const written = await writeModelContainer(
        {
          graphs: { [GRAPH_NAME]: parseIrDeclarationValue(expertGraph(SHAPE)) },
          consts: [],
          weights: tensors,
          assets: [],
          provenance: { license: "test" },
        },
        { blockBytes },
      );
      const opened = await openContainer({ kind: "parts", parts: written.parts });
      const prepared = prepareContainer(opened, GRAPH_NAME);
      const plan = planExpertGraph(prepared.graph, planWeightResidency(prepared.graph), {
        slots: 4,
      });
      const supplies = opened.graphs[GRAPH_NAME].supplies;
      const store = planExpertStore(plan, { supplies, ranges: opened.ranges });
      const weights = plan.layers[0].weights;
      for (const [w, weight] of weights.entries()) {
        const ranges = store.ranges[0][w];
        const stored = storedOf(tensors, weight.name);
        const expected = pieces[weight.name as keyof typeof pieces];
        assertEquals(new Set(ranges.codes.map((range) => range.block)).size, expected);
        assertEquals(supplies.get(weight.name)?.blocks.length, expected);
        for (let expert = 0; expert < SHAPE.experts; expert += 1) {
          const codes = ranges.codes[expert];
          assertEquals(codes.length, weight.codesBytes);
          assertEquals(
            await opened.ranges.read(codes.block, codes.offset, codes.length),
            stored.codes.subarray(expert * weight.codesBytes, (expert + 1) * weight.codesBytes),
            `${weight.name} expert ${expert} codes`,
          );
          const steps = ranges.steps?.[expert];
          assert(steps !== undefined && stored.steps !== undefined);
          assertEquals(
            await opened.ranges.read(steps.block, steps.offset, steps.length),
            stored.steps.subarray(expert * weight.stepsBytes, (expert + 1) * weight.stepsBytes),
            `${weight.name} expert ${expert} steps`,
          );
        }
      }
      // The split scale: one steps block per expert of gate_up at 2048-byte blocks.
      const stepBlocks = new Set(store.ranges[0][0].steps?.map((range) => range.block));
      assertEquals(stepBlocks.size, blockBytes === 2048 ? 16 : 1);
    });
  }

  it("refuses a container that cannot read ranges, and the in-memory pieces whose range read re-reads the piece", () => {
    const declaration = expertGraph(SHAPE);
    const tensors = expertTensors(SHAPE, "f32");
    const graphs = { [GRAPH_NAME]: parseIrDeclarationValue(declaration) };
    const whole = openMemoryContainer({
      graphs,
      tensors: {
        [GRAPH_NAME]: Object.fromEntries(tensors.map((t) => [t.initializer, memoryTensorOf(t)])),
      },
    });
    const plan = planOf(declaration, "f32", 4);
    const supplies = whole.graphs[GRAPH_NAME].supplies;
    assertThrows(
      () => planExpertStore(plan, { supplies, ranges: undefined }),
      ExecutionError,
      "cannot read byte ranges",
    );
    // Whole bytes are seek: accepted.
    assertEquals(planExpertStore(plan, { supplies, ranges: whole.ranges }).ranges.length, 2);
    const rowBytes = 2 * 32 * 64 * 4;
    const split = openMemoryContainer({
      graphs,
      tensors: {
        [GRAPH_NAME]: Object.fromEntries(
          tensors.map((t) => [
            t.initializer,
            t.initializer === "gate_up0"
              ? {
                  encoding: { codec: "f32" },
                  pieces: [
                    { rows: [0, 5], read: () => Promise.resolve(t.bytes.slice(0, 5 * rowBytes)) },
                    { rows: [5, 16], read: () => Promise.resolve(t.bytes.slice(5 * rowBytes)) },
                  ],
                }
              : memoryTensorOf(t),
          ]),
        ),
      },
    });
    assertThrows(
      () =>
        planExpertStore(plan, {
          supplies: split.graphs[GRAPH_NAME].supplies,
          ranges: split.ranges,
        }),
      ExecutionError,
      "re-reads the whole block",
    );
  });
});

describe("the slot choice of one layer's load", () => {
  /** A table of 8 experts with experts 2 → slot 0, 5 → slot 2 resident (slot 1 and 3 free). */
  const table = (): Int32Array<ArrayBuffer> => Int32Array.from([-1, -1, 0, -1, -1, 2, -1, -1]);

  it("fills the free slots first, in ascending order", () => {
    assertEquals(planExpertLoad(4, table(), [2, 5], [7, 1]), {
      load: [
        { expert: 7, slot: 1 },
        { expert: 1, slot: 3 },
      ],
      evict: [],
    });
  });

  it("loads nothing for the experts already resident", () => {
    assertEquals(planExpertLoad(4, table(), [5, 2], [2, 5]), { load: [], evict: [] });
  });

  it("then takes the least recently used expert the routing does not need, pinning the ones it needs", () => {
    // Full: 2 → 0, 5 → 2, 6 → 1, 0 → 3; recency 2 (least recent), 5, 6, 0.
    const full = Int32Array.from([3, -1, 0, -1, -1, 2, 1, -1]);
    // 2 is needed (pinned), so 5 then 6 give their slots up.
    assertEquals(planExpertLoad(4, full, [2, 5, 6, 0], [2, 7, 4]), {
      load: [
        { expert: 7, slot: 2 },
        { expert: 4, slot: 1 },
      ],
      evict: [
        { expert: 5, slot: 2 },
        { expert: 6, slot: 1 },
      ],
    });
  });

  it("refuses a layer that needs more distinct experts than it has slots", () => {
    assertThrows(
      () => planExpertLoad(2, table(), [2, 5], [1, 3, 4]),
      ExecutionError,
      "more than its 2 slots",
    );
  });
});

// ---------------------------------------------------------------------------
// A Session over a fake device: construction issues no dispatch, a load only writeBuffers and pops
// its scopes, so the whole surface but the run itself is exercised without a GPU.
// ---------------------------------------------------------------------------

const LIMITS: RequiredLimits = {
  maxBufferSize: 1 << 24,
  maxStorageBufferBindingSize: 1 << 24,
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
 * Bytes that landed in a buffer: by `writeBuffer` (the tables) or by a submitted
 * `copyBufferToBuffer` (an expert's range, out of a staging-ring slot).
 */
type Write = {
  readonly buffer: number;
  readonly offset: number;
  readonly bytes: Uint8Array;
  readonly via: "writeBuffer" | "copy";
};

/** One recorded `copyBufferToBuffer` of a fake command encoder. */
type FakeCopy = {
  readonly source: GPUBuffer;
  readonly sourceOffset: number;
  readonly target: GPUBuffer;
  readonly targetOffset: number;
  readonly size: number;
};

/** `MAP_WRITE | COPY_SRC` — the staging ring's buffers. */
const UPLOAD_USAGE = BUFFER_USAGE.MAP_WRITE | BUFFER_USAGE.COPY_SRC;

/**
 * A fake device that records every buffer it creates (by number, in creation order, with its usage
 * and whether it was mapped at creation) and the bytes that land in a buffer (`writes`: a copy of
 * the bytes at `writeBuffer`, which copies them too, and at a submit for each recorded
 * `copyBufferToBuffer` — the source's contents then). A buffer keeps its bytes, hands its mapped
 * range out only while mapped, and a submit that uses a mapped buffer or one with a pending map
 * throws (WebGPU's validation). `submits` counts the copies of every submit. `failNextPop` makes the
 * next pop pair report a validation failure. DOM types are not reproduced, so it is passed with a
 * cast (the same test-only boundary as tests/helpers/fake-gpu.ts).
 */
const recordingGpu = () => {
  const sizes: number[] = [];
  const usages: number[] = [];
  const mappedAtCreation: boolean[] = [];
  const writes: Write[] = [];
  const submits: number[] = [];
  const ids = new WeakMap<object, number>();
  const memory = new WeakMap<
    object,
    { readonly bytes: Uint8Array<ArrayBuffer>; mapped: boolean; pending: boolean }
  >();
  const faults = { failNextPop: false };
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    limits: { minStorageBufferOffsetAlignment: 256 },
    destroy: (): void => undefined,
    pushErrorScope: (): void => undefined,
    popErrorScope: (): Promise<GPUError | null> => {
      if (!faults.failNextPop) return Promise.resolve(null);
      faults.failNextPop = false;
      return Promise.resolve({ message: "injected validation failure" } as GPUError);
    },
    createBuffer: (descriptor: GPUBufferDescriptor): GPUBuffer => {
      const state = {
        bytes: new Uint8Array(descriptor.size),
        mapped: descriptor.mappedAtCreation === true,
        pending: false,
      };
      const buffer = {
        size: descriptor.size,
        destroy: (): void => undefined,
        getMappedRange: (): ArrayBuffer => {
          if (!state.mapped) throw new Error("fake: getMappedRange of an unmapped buffer");
          return state.bytes.buffer;
        },
        unmap: (): void => {
          state.mapped = false;
        },
        mapAsync: (): Promise<void> => {
          if (state.mapped || state.pending) {
            return Promise.reject(new Error("fake: mapAsync of a mapped buffer"));
          }
          state.pending = true;
          return Promise.resolve().then(() => {
            state.pending = false;
            state.mapped = true;
          });
        },
      };
      ids.set(buffer, sizes.length);
      memory.set(buffer, state);
      sizes.push(descriptor.size);
      usages.push(descriptor.usage);
      mappedAtCreation.push(descriptor.mappedAtCreation === true);
      return buffer as unknown as GPUBuffer;
    },
    createCommandEncoder: () => {
      const copies: FakeCopy[] = [];
      return {
        copyBufferToBuffer: (
          source: GPUBuffer,
          sourceOffset: number,
          target: GPUBuffer,
          targetOffset: number,
          size: number,
        ): void => {
          copies.push({ source, sourceOffset, target, targetOffset, size });
        },
        finish: () => ({ copies }),
      };
    },
    queue: {
      writeBuffer: (buffer: GPUBuffer, offset: number, data: ArrayBufferView): void => {
        const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
        writes.push({ buffer: ids.get(buffer) ?? -1, offset, bytes, via: "writeBuffer" });
      },
      submit: (commands: readonly { readonly copies?: readonly FakeCopy[] }[]): void => {
        let count = 0;
        for (const command of commands) {
          for (const copy of command.copies ?? []) {
            const source = memory.get(copy.source);
            if (source === undefined || source.mapped || source.pending) {
              throw new Error("fake: a submit used a mapped buffer (WebGPU validation)");
            }
            writes.push({
              buffer: ids.get(copy.target) ?? -1,
              offset: copy.targetOffset,
              bytes: source.bytes.slice(copy.sourceOffset, copy.sourceOffset + copy.size),
              via: "copy",
            });
            count += 1;
          }
        }
        submits.push(count);
      },
      onSubmittedWorkDone: (): Promise<void> => Promise.resolve(),
    },
  } as unknown as GPUDevice;
  return {
    gpu: new GpuContext(device, readAdapterInfo({}), LIMITS, new Set()),
    sizes,
    usages,
    mappedAtCreation,
    writes,
    submits,
    faults,
  };
};

/**
 * A part source whose reads can be corrupted after construction (the bytes of a local file changing
 * under the Session). Unverified, so construction keeps a digest of every expert range. `reads`
 * counts every read; with `corruptFrom` set, the reads from that count on are corrupted too.
 */
const tamperableSource = (parts: readonly Uint8Array<ArrayBuffer>[]) => {
  const inner = partsSource(parts);
  const state = { corrupt: false, reads: 0, corruptFrom: undefined as number | undefined };
  const source: BlockSource = {
    partCount: inner.partCount,
    verified: false,
    partLength: (index) => inner.partLength(index),
    read: async (part, offset, length) => {
      const index = state.reads;
      state.reads += 1;
      const bytes = await inner.read(part, offset, length);
      if (!state.corrupt && (state.corruptFrom === undefined || index < state.corruptFrom)) {
        return bytes;
      }
      const flipped = bytes.slice();
      flipped[0] ^= 0xff;
      return flipped;
    },
  };
  return { source, state };
};

const buildCached = async (
  slots: number,
  storage: ExpertStorage = "f32",
  options: SessionOptions = {},
  extra: { readonly hostBytes?: number; readonly shape?: ExpertGraphShape } = {},
) => {
  const shape = extra.shape ?? SHAPE;
  const recording = recordingGpu();
  const tensors = expertTensors(shape, storage);
  const written = await writeModelContainer(
    {
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(expertGraph(shape)) },
      consts: [],
      weights: tensors,
      assets: [],
      provenance: { license: "test" },
    },
    {},
  );
  const tamper = tamperableSource(written.parts);
  const opened = await openContainer({ kind: "source", source: tamper.source });
  const session = await createSessionFromContainer(recording.gpu, opened, GRAPH_NAME, {
    ...options,
    expertCache: extra.hostBytes === undefined ? { slots } : { slots, hostBytes: extra.hostBytes },
  });
  return { ...recording, session, tensors, tamper };
};

const cacheOf = (session: Session) => {
  const cache = session.expertCache;
  if (cache === undefined) throw new Error("no expert cache");
  return cache;
};

/** A routing of one row: `ids[layer]` are its picks (top-4 of 16). */
const routingOf = (ids: readonly (readonly number[])[]): ExpertRouting => ({
  layers: ids.length,
  rows: 1,
  picks: 4,
  ids: Int32Array.from(ids.flat()),
});

describe("a Session built with expertCache (fake device)", () => {
  it("refuses Session.run, enqueue and enqueueRead, pointing at expertCache.run", async () => {
    const { session } = await buildCached(4);
    // The refusal comes before the batch is touched, so a stand-in batch is enough.
    const batch = anyOf({});
    try {
      await assertRejects(() => session.run({}), ExecutionError, "session.expertCache.run");
      await assertRejects(() => session.enqueue({}, { batch }), ExecutionError, "expertCache");
      await assertRejects(
        () => session.enqueueRead({}, { batch }).admitted,
        ExecutionError,
        "expertCache",
      );
    } finally {
      await session.dispose();
    }
  });

  it("refuses a generation run that is not deferred, and a chained one", async () => {
    const { session } = await buildCached(4);
    const cache = cacheOf(session);
    // The rule is checked before the context is touched, so a stand-in context is enough.
    const run = (generation: Partial<GenerationRun>) =>
      cache.run({}, {}, anyOf({ context: {}, queryLength: 1, ...generation }));
    try {
      await assertRejects(() => run({}), ExecutionError, "needs commit 'deferred'");
      await assertRejects(() => run({ commit: "immediate" }), ExecutionError, "commit 'deferred'");
      await assertRejects(
        () => run({ commit: "deferred", chain: true }),
        ExecutionError,
        "cannot be chained",
      );
    } finally {
      await session.dispose();
    }
  });

  it("has no expert cache without the option", async () => {
    const recording = recordingGpu();
    const opened = await openContainer({
      kind: "parts",
      parts: (
        await writeModelContainer(
          {
            graphs: { [GRAPH_NAME]: parseIrDeclarationValue(expertGraph(SHAPE)) },
            consts: [],
            weights: expertTensors(SHAPE, "f32"),
            assets: [],
            provenance: { license: "test" },
          },
          {},
        )
      ).parts,
    });
    const session = await createSessionFromContainer(recording.gpu, opened, GRAPH_NAME);
    try {
      assertEquals(session.expertCache, undefined);
    } finally {
      await session.dispose();
    }
  });

  it("allocates pools of K experts and −1 tables, and uploads no expert at construction", async () => {
    const { session, sizes, usages, mappedAtCreation, writes } = await buildCached(4, "i4f16");
    try {
      // Per layer: the router 16 · 64 · 4 = 4096 and its bias 64, a 64-byte table, gate_up 4 · 2048
      // codes + 4 · 256 steps, down 4 · 1024 + 4 · 128 — nothing of [16, …] (32768 bytes of codes).
      assertEquals(
        sizes.filter((_, id) => usages[id] !== UPLOAD_USAGE).sort((a, b) => a - b),
        [64, 64, 64, 64, 512, 512, 1024, 1024, 4096, 4096, 4096, 4096, 8192, 8192],
      );
      // The staging ring: one slot per expert a load can need (2 layers · 4 slots), each one
      // expert's entry (2048 + 256 + 1024 + 128 bytes), mapped at creation for the first load.
      const ring = sizes.flatMap((size, id) =>
        usages[id] === UPLOAD_USAGE ? [[size, mappedAtCreation[id]]] : [],
      );
      assertEquals(ring, new Array(8).fill([3456, true]));
      // The tables come first, every entry −1; then only the routers are uploaded.
      for (const table of writes.slice(0, 2)) {
        assertEquals([...new Int32Array(table.bytes.buffer)], new Array(16).fill(-1));
      }
      // (in the container's block order)
      assertEquals(
        writes
          .slice(2)
          .map((write) => write.bytes.byteLength)
          .sort((a, b) => a - b),
        [64, 64, 4096, 4096],
      );
      assertEquals(
        session.diagnostics().storage.residentCompressedBytes,
        2 * (8192 + 1024 + 4096 + 512),
      );
    } finally {
      await session.dispose();
    }
  });

  it("writes each missing expert into its slot, then the table, and counts the load", async () => {
    const { session, writes, sizes, tensors } = await buildCached(4, "i4f16");
    const cache = cacheOf(session);
    try {
      writes.length = 0;
      const first = await cache.load(
        routingOf([
          [3, 5, 3, 9],
          [7, 1, 2, 0],
        ]),
      );
      // Layer 0 needs 3, 5, 9 (free slots 0, 1, 2), layer 1 needs 7, 1, 2, 0 (slots 0 to 3).
      assertEquals([first.loaded, first.evicted], [7, 0]);
      assertEquals(first.bytes, 7 * (2048 + 256 + 1024 + 128));
      // Without a host tier every expert is read from the container, and what is read is uploaded.
      assertEquals([first.fromHost, first.fromContainer, first.uploadedBytes], [0, 7, first.bytes]);
      // Layer 0's writes, expert by expert in pick order: gate_up codes and steps, down codes and
      // steps, each the stored bytes of that expert at its slot.
      const gateUp = storedOf(tensors, "gate_up0");
      const down = storedOf(tensors, "down0");
      const slice = (bytes: Uint8Array<ArrayBuffer> | undefined, expert: number, size: number) =>
        bytes?.slice(expert * size, (expert + 1) * size);
      for (const [order, expert, slot] of [
        [0, 3, 0],
        [1, 5, 1],
        [2, 9, 2],
      ]) {
        const [codes, steps, downCodes, downSteps] = writes.slice(4 * order, 4 * order + 4);
        assertEquals([codes.offset, codes.bytes], [slot * 2048, slice(gateUp.codes, expert, 2048)]);
        assertEquals([steps.offset, steps.bytes], [slot * 256, slice(gateUp.steps, expert, 256)]);
        assertEquals(
          [downCodes.offset, downCodes.bytes],
          [slot * 1024, slice(down.codes, expert, 1024)],
        );
        assertEquals(
          [downSteps.offset, downSteps.bytes],
          [slot * 128, slice(down.steps, expert, 128)],
        );
        assertEquals(
          [codes.buffer, steps.buffer, downCodes.buffer, downSteps.buffer].map((id) => sizes[id]),
          [8192, 1024, 4096, 512],
        );
      }
      // 7 experts × 4 ranges, copied out of the staging ring in one submit, then the tables by
      // writeBuffer: layer 0 then layer 1.
      assertEquals(writes.length, 7 * 4 + 2);
      assertEquals(
        writes.map((write) => write.via),
        [...new Array(7 * 4).fill("copy"), "writeBuffer", "writeBuffer"],
      );
      const tables = writes.slice(-2).map((write) => [...new Int32Array(write.bytes.buffer)]);
      const expected0 = new Array(16).fill(-1);
      [expected0[3], expected0[5], expected0[9]] = [0, 1, 2];
      const expected1 = new Array(16).fill(-1);
      [expected1[7], expected1[1], expected1[2], expected1[0]] = [0, 1, 2, 3];
      assertEquals(tables, [expected0, expected1]);

      // Nothing missing: no write, no count.
      writes.length = 0;
      assertEquals(
        await cache.load(
          routingOf([
            [9, 3, 5, 5],
            [0, 1, 2, 7],
          ]),
        ),
        {
          loaded: 0,
          evicted: 0,
          fromHost: 0,
          fromContainer: 0,
          bytes: 0,
          uploadedBytes: 0,
          ms: 0,
          readMs: 0,
          uploadMs: 0,
        },
      );
      assertEquals(writes.length, 0);

      // Recency: touch 3 then 5 in layer 0 (9 stays the least recent); a load of 4, 6 takes the
      // free slot 3 and evicts 9.
      cache.touch(
        routingOf([
          [3, 5, 3, 5],
          [0, 1, 2, 7],
        ]),
        1,
      );
      const second = await cache.load(
        routingOf([
          [4, 6, 4, 6],
          [0, 1, 2, 7],
        ]),
      );
      assertEquals([second.loaded, second.evicted], [2, 1]);
      const table0 = [...new Int32Array(writes.at(-1)?.bytes.buffer ?? new ArrayBuffer(0))];
      assertEquals([table0[3], table0[5], table0[9], table0[4], table0[6]], [0, 1, -1, 3, 2]);
      const stats = cache.stats();
      assertEquals(
        [stats.loads, stats.loadedExperts, stats.evictedExperts, stats.bytesRead],
        [2, 9, 1, first.bytes + 2 * (2048 + 256 + 1024 + 128)],
      );
      assertEquals([stats.runs, stats.missedRuns], [0, 0]);
      assertEquals(
        [
          stats.hostExperts,
          stats.containerExperts,
          stats.bytesUploaded,
          stats.hostHeldExperts,
          stats.hostHeldBytes,
        ],
        [0, 9, stats.bytesRead, 0, 0],
      );
    } finally {
      await session.dispose();
    }
  });

  it("refuses a routing of the wrong shape, an id out of range, more experts than slots, and touching a missing expert", async () => {
    const { session } = await buildCached(4);
    const cache = cacheOf(session);
    try {
      await assertRejects(() => cache.load(routingOf([[0, 1, 2, 3]])), ExecutionError, "routing");
      await assertRejects(
        () =>
          cache.load(
            routingOf([
              [0, 1, 2, 16],
              [0, 1, 2, 3],
            ]),
          ),
        ExecutionError,
        "not an expert in [0, 16)",
      );
      const wide: ExpertRouting = {
        layers: 2,
        rows: 2,
        picks: 4,
        ids: Int32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3]),
      };
      await assertRejects(() => cache.load(wide), ExecutionError, "more than its 4 slots");
      assertThrows(
        () =>
          cache.touch(
            routingOf([
              [0, 1, 2, 3],
              [0, 1, 2, 3],
            ]),
            1,
          ),
        ExecutionError,
        "is not resident",
      );
    } finally {
      await session.dispose();
    }
  });

  it("refuses a range whose bytes changed after construction on an unverified source, and stays usable", async () => {
    const { session, tamper } = await buildCached(4, "i4");
    const cache = cacheOf(session);
    const routing = routingOf([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
    ]);
    try {
      tamper.state.corrupt = true;
      const error = await assertRejects(() => cache.load(routing), ContainerFormatError);
      assertStringIncludes(error.message, "differ from the bytes verified at construction");
      // A failed read issued no GPU write, so the cache is consistent and the next load goes on.
      tamper.state.corrupt = false;
      assertEquals((await cache.load(routing)).loaded, 8);
    } finally {
      await session.dispose();
    }
  });

  it("becomes unusable when a load's GPU section fails (the tables can no longer be trusted)", async () => {
    const { session, faults } = await buildCached(4);
    const cache = cacheOf(session);
    const routing = routingOf([
      [0, 1, 2, 3],
      [4, 5, 6, 7],
    ]);
    try {
      faults.failNextPop = true;
      const failure = await assertRejects(() => cache.load(routing));
      assertInstanceOf(failure, GpuValidationError);
      await assertRejects(() => cache.load(routing), ExecutionError, "the cache is unusable");
      await assertRejects(() => cache.run({ x: anyOf({}) }), ExecutionError, "unusable");
      assertThrows(() => cache.touch(routing, 0), ExecutionError, "unusable");
    } finally {
      await session.dispose();
    }
  });
});

// ---------------------------------------------------------------------------
// The host tier and the staging ring through a load (fake device): where an expert's bytes come
// from, what lands in its pool slot, and the rounds of a load larger than the ring.
// ---------------------------------------------------------------------------

/** Bytes of one i4f16 expert of {@link SHAPE}: gate_up 2048 codes + 256 steps, down 1024 + 128. */
const ENTRY = 2048 + 256 + 1024 + 128;

/** The bytes that landed by copy (out of a ring slot), in order: `[offset, bytes]`. */
const copiesOf = (writes: readonly Write[]) =>
  writes.filter((write) => write.via === "copy").map((write) => [write.offset, write.bytes]);

/**
 * What a load of `moves` (`[layer, expert, slot]`, in move order) lands: each expert's gate_up codes
 * and steps, then down's codes and steps — the stored bytes of that expert, at its slot.
 */
const expectedCopies = (
  tensors: ReturnType<typeof expertTensors>,
  moves: readonly (readonly [number, number, number])[],
) =>
  moves.flatMap(([layer, expert, slot]) => {
    const gateUp = storedOf(tensors, `gate_up${layer}`);
    const down = storedOf(tensors, `down${layer}`);
    const slice = (bytes: Uint8Array<ArrayBuffer> | undefined, size: number): Uint8Array => {
      if (bytes === undefined) throw new Error(`no stored steps of layer ${layer}`);
      return bytes.slice(expert * size, (expert + 1) * size);
    };
    return [
      [slot * 2048, slice(gateUp.codes, 2048)],
      [slot * 256, slice(gateUp.steps, 256)],
      [slot * 1024, slice(down.codes, 1024)],
      [slot * 128, slice(down.steps, 128)],
    ];
  });

/** Three layers of 16 experts: with 16 slots a load can need 48 experts, more than the ring's 32. */
const SHAPE3: ExpertGraphShape = { ...SHAPE, layers: 3 };

/** Four rows whose picks are experts 0 to 15 in every one of the three layers. */
const WIDE: ExpertRouting = {
  layers: 3,
  rows: 4,
  picks: 4,
  ids: Int32Array.from({ length: 3 * 16 }, (_, at) => at % 16),
};

const IDENTITY16 = Array.from({ length: 16 }, (_, expert) => expert);

describe("the host tier and the staging ring through a load (fake device)", () => {
  it("holds every expert at construction under a budget for all, and loads without a container read", async () => {
    const all = 2 * 16 * ENTRY;
    const { session, writes, tensors, tamper } = await buildCached(
      4,
      "i4f16",
      {},
      {
        hostBytes: all,
      },
    );
    const cache = cacheOf(session);
    try {
      assertEquals([cache.stats().hostHeldExperts, cache.stats().hostHeldBytes], [32, all]);
      // Any container read would now fail its digest check (and it is counted).
      tamper.state.corrupt = true;
      tamper.state.reads = 0;
      writes.length = 0;
      const load = await cache.load(
        routingOf([
          [3, 5, 3, 9],
          [7, 1, 2, 0],
        ]),
      );
      assertEquals(
        [load.loaded, load.fromHost, load.fromContainer, load.bytes, load.uploadedBytes],
        [7, 7, 0, 0, 7 * ENTRY],
      );
      assertEquals(tamper.state.reads, 0);
      // The RAM entries carry exactly the stored bytes of each expert to its slot.
      assertEquals(
        copiesOf(writes),
        expectedCopies(tensors, [
          [0, 3, 0],
          [0, 5, 1],
          [0, 9, 2],
          [1, 7, 0],
          [1, 1, 1],
          [1, 2, 2],
          [1, 0, 3],
        ]),
      );
      const stats = cache.stats();
      assertEquals(
        [
          stats.hostExperts,
          stats.containerExperts,
          stats.bytesRead,
          stats.bytesUploaded,
          stats.hostHeldExperts,
        ],
        [7, 0, 0, 7 * ENTRY, 32],
      );
    } finally {
      await session.dispose();
    }
    // The dispose released the tier's RAM.
    assertEquals([cache.stats().hostHeldExperts, cache.stats().hostHeldBytes], [0, 0]);
  });

  it("holds the first experts read under a small budget, takes them from RAM, and reads and keeps the rest", async () => {
    const budget = 6 * ENTRY;
    const { session, writes, tensors, tamper } = await buildCached(
      4,
      "i4f16",
      {},
      {
        hostBytes: budget,
      },
    );
    const cache = cacheOf(session);
    try {
      assertEquals([cache.stats().hostHeldExperts, cache.stats().hostHeldBytes], [6, budget]);
      tamper.state.reads = 0;
      writes.length = 0;
      // Layer 0's experts 0 and 1 are among the first read (layer 0's gate_up, in expert order);
      // 6 and 7 of layer 0 and 0 to 3 of layer 1 are not.
      const load = await cache.load(
        routingOf([
          [0, 1, 6, 7],
          [0, 1, 2, 3],
        ]),
      );
      assertEquals(
        [load.loaded, load.fromHost, load.fromContainer, load.bytes, load.uploadedBytes],
        [8, 2, 6, 6 * ENTRY, 8 * ENTRY],
      );
      // Four ranges per expert read; none for the two taken from RAM.
      assertEquals(tamper.state.reads, 6 * 4);
      assertEquals(
        copiesOf(writes),
        expectedCopies(tensors, [
          [0, 0, 0],
          [0, 1, 1],
          [0, 6, 2],
          [0, 7, 3],
          [1, 0, 0],
          [1, 1, 1],
          [1, 2, 2],
          [1, 3, 3],
        ]),
      );
      // The read experts were kept, evicting the least recent: the tier stays at its budget.
      assertEquals([cache.stats().hostHeldExperts, cache.stats().hostHeldBytes], [6, budget]);
      // A miss of the tier is read from the container and still checked against the construction
      // digest: a changed source fails it before any GPU write, and the cache stays usable.
      tamper.state.corrupt = true;
      const error = await assertRejects(
        () =>
          cache.load(
            routingOf([
              [0, 1, 6, 7],
              [4, 1, 2, 3],
            ]),
          ),
        ContainerFormatError,
      );
      assertStringIncludes(error.message, "differ from the bytes verified at construction");
      tamper.state.corrupt = false;
      assertEquals(
        (
          await cache.load(
            routingOf([
              [0, 1, 6, 7],
              [4, 1, 2, 3],
            ]),
          )
        ).fromContainer,
        1,
      );
    } finally {
      await session.dispose();
    }
  });

  it("runs a load larger than the ring in rounds, each submitted with its tables after it", async () => {
    const { session, writes, submits, tensors } = await buildCached(
      16,
      "i4f16",
      {},
      {
        shape: SHAPE3,
      },
    );
    const cache = cacheOf(session);
    try {
      writes.length = 0;
      submits.length = 0;
      const load = await cache.load(WIDE);
      assertEquals([load.loaded, load.evicted, load.fromContainer], [48, 0, 48]);
      // 32 experts (layers 0 and 1), then 16 (layer 2): four copies each.
      assertEquals(submits, [32 * 4, 16 * 4]);
      // Each round's tables come right after its copies (by writeBuffer, after the submit).
      assertEquals(
        writes.map((write) => write.via),
        [
          ...new Array(32 * 4).fill("copy"),
          "writeBuffer",
          "writeBuffer",
          ...new Array(16 * 4).fill("copy"),
          "writeBuffer",
        ],
      );
      assertEquals(
        writes
          .filter((write) => write.via === "writeBuffer")
          .map((write) => [...new Int32Array(write.bytes.buffer)]),
        [IDENTITY16, IDENTITY16, IDENTITY16],
      );
      assertEquals(
        copiesOf(writes),
        expectedCopies(
          tensors,
          [0, 1, 2].flatMap((layer) =>
            IDENTITY16.map((expert): [number, number, number] => [layer, expert, expert]),
          ),
        ),
      );
    } finally {
      await session.dispose();
    }
  });

  it("keeps the rounds before a failed read resident, changes nothing after it, and stays usable", async () => {
    const { session, writes, submits, tamper } = await buildCached(
      16,
      "i4f16",
      {},
      {
        shape: SHAPE3,
      },
    );
    const cache = cacheOf(session);
    try {
      writes.length = 0;
      submits.length = 0;
      tamper.state.reads = 0;
      // The first round reads 32 experts · 4 ranges; every read after them is corrupted.
      tamper.state.corruptFrom = 32 * 4;
      await assertRejects(() => cache.load(WIDE), ContainerFormatError, "differ from the bytes");
      assertEquals(submits, [32 * 4]);
      assertEquals(
        writes
          .filter((write) => write.via === "writeBuffer")
          .map((write) => [...new Int32Array(write.bytes.buffer)]),
        [IDENTITY16, IDENTITY16],
      );
      const after = cache.stats();
      assertEquals(
        [after.loads, after.loadedExperts, after.containerExperts, after.bytesRead],
        [1, 32, 32, 32 * ENTRY],
      );
      // Layers 0 and 1 are resident (touching them passes); layer 2 is not.
      const resident: ExpertRouting = { ...WIDE, rows: 4 };
      assertThrows(() => cache.touch(resident, 4), ExecutionError, "of layer 2 is not resident");
      // The same routing loads the rest, layer 2's 16, in one round.
      tamper.state.corruptFrom = undefined;
      writes.length = 0;
      submits.length = 0;
      const rest = await cache.load(WIDE);
      assertEquals([rest.loaded, rest.evicted], [16, 0]);
      assertEquals(submits, [16 * 4]);
      cache.touch(WIDE, 4);
    } finally {
      await session.dispose();
    }
  });
});
