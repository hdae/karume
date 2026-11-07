// Checks for GenerationContext (the owner of the mutable state of one generation — ADR 0066).
//
// At wave C there is **no dispatch that reads or writes state yet** (the states field of nodes,
// state_append and the attention integration are wave D). So what this pins is contracts, not
// values: ① ownership and lifetime (allocate → dispose, double dispose, order independence from the
// Session) ② the entry gates (graphs without states, chunkLength, symbolic capacity, the binding
// limit on capacity) ③ an allocation failure is fail loudly through an errorScope ④ the bounds of
// advancing and rewinding the logical length ⑤ rejection on poisoning and device loss
// ⑥ the logical-length uniform **really is on the GPU** ⑦ the context has no effect at all on the
// plan key ⑧ the lifetime contract of the 3rd argument of `Session.run` (snapshot at issue /
// rejection of concurrent issues to the same context).
// ⑨ `fork` (branching the conversation state — ADR 0066 addendum of 2026-10-11): what is copied,
// independence of the branches, the contract of the synchronous section, the rejection matrix and
// the diagnostics (details at the top of the section).
//
// Without ⑥, a no-op writeBuffer (an invalid buffer or an alignment violation does not even warn)
// cannot be detected.
// Without ⑦, an implementation that mixes the context into the key in wave D would pass as
// "the values stay correct while decode re-derives every step" (with no exception and no warning).

import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertNotEquals,
  assertRejects,
  assertStrictEquals,
  assertThrows,
} from "@std/assert";
import type { IrGraph } from "../src/format/ir.ts";
import {
  acquireGpu,
  type GpuContext,
  GpuDeviceLostError,
  LIMIT_CAPS,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import {
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { OpContractError } from "../src/ops.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  baseDeclaration,
  baseTensors,
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openGraphModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** WebGPU core's default storage binding limit (the capacity-gate check reproduces it). */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

/**
 * Bytes of the logical-length uniform (4 words: pastLength, queryLength and the two of the bidirectional block — ADR 0144 decision 3).
 */
const LENGTHS_BYTES = 16;

/** A graph with 2 numeric-capacity state slots (k / v = [1,2,C,4]). y = x·w. */
const STATE_CAPACITY = 16;
const SLOT_BYTES = 1 * 2 * STATE_CAPACITY * 4 * 4;

/**
 * The input of `state_append` (this step's chunk — the declared shape MUST be the rank-4
 * `[B,Hkv,M,D]`).
 * A form whose B / Hkv / D match the slot `[1,2,C,4]` (② of ADR 0067 decision 4).
 */
const CHUNK_SHAPE: readonly number[] = [1, 2, 4, 4];

/**
 * A declared slot is **always referenced by a node** (reference completeness — ADR 0067 decisions
 * 4 / 5), so each slot gets one minimal `state_append`. At wave C a "declaration-only graph" was
 * enough, but now that the referencing field exists, that is rejected as an orphan declaration.
 */
const stateGraph = (
  states: DeclarationJson["states"],
  extra: Partial<DeclarationJson> = {},
  /** The attrs of the `state_append` node (pass `{ window }` only to make it sliding). */
  appendAttrs: Record<string, number> = {},
): DeclarationJson => {
  const slots = Object.keys(states ?? {});
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: slots.length > 0 ? ["matmul", "state_append"] : ["matmul"] },
    symbols: ["T"],
    inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
    outputs: ["y"],
    initializers: { w: {}, chunk: {} },
    values: {
      w: { dtype: "f32", shape: [4, 3] },
      y: { dtype: "f32", shape: ["T", 3] },
      chunk: { dtype: "f32", shape: [...CHUNK_SHAPE] },
    },
    states,
    nodes: [
      { op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} },
      ...slots.map((slot) => ({
        op: "state_append",
        ins: ["chunk"],
        outs: [],
        attrs: { ...appendAttrs },
        states: { slot },
      })),
    ],
    ...extra,
  };
};

const NUMERIC_STATES: DeclarationJson["states"] = {
  k: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
  v: { dtype: "f32", shape: [1, 2, STATE_CAPACITY, 4] },
};

/**
 * A graph with symbolic capacity. The capacity symbol `C` **appears in no input** — it is bound on
 * the side of `createGenerationContext(spec.bindings)` (binding point 2 of ADR 0066 addendum 7) and
 * is never used to resolve value shapes either. Wave C placed an input `cap` to satisfy
 * bindability, but in wave D-1 the check was relaxed to "input shapes ∪ states shapes", so it can
 * be written as **a plain states-only symbol**.
 */
const symbolicGraph = (): DeclarationJson =>
  stateGraph({ k: { dtype: "f32", shape: [1, 2, "C", 4] } }, { symbols: ["T", "C"] });

/**
 * The data of the 2 declared initializers (in v2 the initializer name is the key of the data as
 * is).
 */
const stateTensors = (): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: f32Bytes([0.5, -1.5, 2, 0.25, -0.75, 1, 0.125, -0.25, 1.5, 2, -1, 0.75]),
    encoding: { codec: "f32" },
  },
  {
    graph: GRAPH_NAME,
    initializer: "chunk",
    bytes: f32Bytes(new Array(1 * 2 * 4 * 4).fill(0)),
    encoding: { codec: "f32" },
  },
];

const openStateModel = (graph: DeclarationJson): Promise<OpenedContainer> =>
  openModelBytes(graph, stateTensors());

const stateSession = async (
  gpu: GpuContext,
  graph: DeclarationJson = stateGraph(NUMERIC_STATES),
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openStateModel(graph), GRAPH_NAME);

/**
 * Drives the internal surface (the advance, poisoning and transport that the execution integration
 * of wave D calls) from the test.
 */
const internals = (context: GenerationContext) => context[RUNTIME_INTERNAL];

/**
 * Upper bound of the logical length (carried in a Uint32Array / WGSL u32 — ADR 0066 addendum 4).
 */
const U32_MAX = 0xffffffff;

/**
 * Faults to inject (a synchronous throw part way through allocation / a reject of pop / a window
 * where pop never settles).
 */
type Fault = {
  readonly failCreateAt?: number;
  readonly popError?: Error;
  readonly hangPop?: boolean;
};

/**
 * Wraps the real device in a proxy, injects faults into the allocation path, and **counts the
 * destroy calls of the returned buffers in allocation order** (a leak = 0 calls and a double
 * destroy = 2 calls are visible at the same observation point).
 *
 * A substitute for opening an injection surface in production code. The OOM check (64GiB below) can
 * only detect leaks by "using up the headroom so nothing more can be allocated", is skipped
 * entirely on machines below 1GiB, and cannot target the N-th allocation, a reject of pop or a loss
 * separately.
 */
const injectFaults = (
  gpu: GpuContext,
  fault: Fault,
): { readonly gpu: GpuContext; readonly destroyCounts: number[] } => {
  const destroyCounts: number[] = [];
  let creations = 0;
  const device = new Proxy(gpu.device, {
    get(target, prop): unknown {
      if (prop === "createBuffer") {
        return (descriptor: GPUBufferDescriptor): GPUBuffer => {
          creations += 1;
          if (creations === fault.failCreateAt) {
            throw new Error(`injected: createBuffer #${creations} throws synchronously`);
          }
          const buffer = target.createBuffer(descriptor);
          const index = destroyCounts.push(0) - 1;
          return new Proxy(buffer, {
            get(inner, innerProp): unknown {
              if (innerProp === "destroy") {
                return (): void => {
                  destroyCounts[index] += 1;
                  inner.destroy();
                };
              }
              const value: unknown = Reflect.get(inner, innerProp);
              return typeof value === "function" ? value.bind(inner) : value;
            },
          });
        };
      }
      if (prop === "popErrorScope" && (fault.popError !== undefined || fault.hangPop === true)) {
        return (): Promise<GPUError | null> => {
          if (fault.hangPop === true) {
            // MUST: the never-settling window is made **without issuing a real pop**. This case
            // destroys the device right after, so the LIFO balance does not matter (the whole stack
            // goes away), whereas leaving a real op that never settles after the loss would make
            // the test's own op the thing being waited on.
            return new Promise<GPUError | null>(() => {});
          }
          // MUST: the side that injects a reject **always issues the real pop**. Swallowing it
          // would leave the 2 pushed scopes on the device's LIFO, and later test failures would be
          // absorbed by the wrong scope.
          void target.popErrorScope().catch(() => null);
          return Promise.reject(fault.popError);
        };
      }
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const proxied = new Proxy(gpu, {
    get(target, prop): unknown {
      if (prop === "device") return device;
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  return { gpu: proxied, destroyCounts };
};

/**
 * A surface that builds a context directly on the injected gpu (without a Session — only a few
 * fields are borrowed).
 *
 * `adopt` is a pass-through without registration bookkeeping (it calls `allocate` synchronously, as
 * the MUST of `Session.#adopt` requires).
 * Only the fork fault injection replaces `adopt` ({@link delayedAdoptHost}).
 */
const injectedHost = (gpu: GpuContext, graph: IrGraph): GenerationContextHost => ({
  gpu,
  graph,
  fullStateStorage: "f32",
  flush: () => Promise.resolve(),
  serialize: <T>(body: () => Promise<T>): Promise<T> => body(),
  forget: () => {},
  adopt: (allocate) => allocate(),
});

test.skipIf(!GPU_AVAILABLE)(
  "createGenerationContext allocates physically per slot and returns it on dispose (with diagnostics, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      assertEquals(
        session.diagnostics().stateBacking,
        { residentBytes: 0, contextCount: 0, rebindCount: 0 },
        "a Session that made no context pays no state bytes at all",
      );

      const context = await session.createGenerationContext({ chunkLength: 4 });
      assertEquals(context.chunkLength, 4);
      assertEquals(context.pastLength, 0, "the logical length right after creation is 0");
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: 1,
          rebindCount: 0,
        },
        "the 2 slots + the logical-length uniform are resident",
      );
      // The declared shape is in the data as the capacity with its bindings resolved (the size is
      // checked on the GPUBuffer side — a 4-byte floor or rounding up would silently change
      // arrayLength()).
      assertEquals([...internals(context).slots.keys()].sort(), ["k", "v"]);
      for (const [name, slot] of internals(context).slots) {
        assertEquals(slot.shape, [1, 2, STATE_CAPACITY, 4], name);
        assertEquals(slot.byteLength, SLOT_BYTES, name);
        assertEquals(slot.buffer.size, SLOT_BYTES, name);
      }
      assertEquals(internals(context).lengths.size, LENGTHS_BYTES);

      // A second context has its own data (it does not share the first one's).
      const second = await session.createGenerationContext({ chunkLength: 1 });
      assertEquals(session.diagnostics().stateBacking.contextCount, 2);
      assertEquals(
        session.diagnostics().stateBacking.residentBytes,
        (SLOT_BYTES * 2 + LENGTHS_BYTES) * 2,
      );
      assert(
        internals(second).slots.get("k")?.buffer !== internals(context).slots.get("k")?.buffer,
        "without a separate buffer per context, the second would overwrite the first one's KV",
      );

      // A double dispose is harmless (it returns **the same Promise** — the same discipline as
      // Session.dispose. An implementation that returns a different Promise could regress into
      // "only the second call resolves before the flush settles").
      const disposal = context.dispose();
      const again = context.dispose();
      assertStrictEquals(again, disposal, "later calls return the first completion itself");
      assertEquals(await Promise.all([disposal, again]), [undefined, undefined]);
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: 2,
          rebindCount: 0,
        },
        "resident bytes are derived from the live set (the cumulative count does not go down)",
      );

      // Operations on a disposed context are fail loudly (reads included — the data is gone).
      assertThrows(() => context.pastLength, ExecutionError, "disposed");
      assertThrows(() => context.rewind(0), ExecutionError, "disposed");
      // The internal surface closes at **stage 2** of dispose (after the destruction itself has
      // run) — this has been awaited, so stage 2 is done.
      assertThrows(() => internals(context).advance(0, 1), ExecutionError, "disposed");
      assertThrows(() => internals(context).pastLength(), ExecutionError, "disposed");

      // A context can outlive the Session (no dependency on the order of dispose).
      await session.dispose();
      await second.dispose();
      assertEquals(session.diagnostics().stateBacking.residentBytes, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a Session disposed across the allocation await is not left holding a context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // createGenerationContext does not go on the serialization chain (putting it there would
      // drag it into the same kind of cycle as an unsettled run inside a batch section), so the
      // allocation await spans the dispose check made at call time. Without a re-check after the
      // allocation, the context would be registered in the live set of a disposed Session, and the
      // diagnostics would keep claiming residency for data that remains (the buffers remain until
      // the user calls context.dispose()).
      const pending = session.createGenerationContext({ chunkLength: 4 });
      const disposal = session.dispose();
      await disposal;
      const error = await assertRejects(() => pending, ExecutionError);
      assert(error.message.includes("disposed Session"), error.message);
      assertEquals(
        session.diagnostics().stateBacking,
        { residentBytes: 0, contextCount: 0, rebindCount: 0 },
        "what was allocated is destroyed, and it appears neither in the live set nor in the cumulative count",
      );
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a GenerationContext cannot be made for a graph without a states declaration (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await createSessionFromContainer(
      gpu,
      await openModelBytes(baseDeclaration(), baseTensors()),
      GRAPH_NAME,
    );
    try {
      const error = await assertRejects(
        () => session.createGenerationContext({ chunkLength: 1 }),
        ExecutionError,
      );
      assert(error.message.includes("has no states declarations"), error.message);
      assertEquals(session.diagnostics().stateBacking.contextCount, 0);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "spec checks: chunkLength is an integer ≥ 1, and symbolic capacity is determined by bindings alone (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const symbolic = await stateSession(gpu, symbolicGraph());
    try {
      for (const chunkLength of [0, -1, 1.5, Number.NaN]) {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength }),
          ExecutionError,
        );
        assert(error.message.includes("chunkLength"), error.message);
      }

      // An unbound symbolic capacity is fail loudly (it is not filled with 0 or 1).
      const unbound = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2 }),
        ExecutionError,
      );
      assert(unbound.message.includes("unbound symbol 'C'"), unbound.message);

      // A binding missing from symbols is fail loudly too (a misspelling is not silently ignored).
      const unknown = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2, bindings: { D: 8 } }),
        ExecutionError,
      );
      assert(unknown.message.includes("Binding 'D'"), unknown.message);

      // A binding with capacity 0 is "a slot that cannot have data", so it is rejected.
      const empty = await assertRejects(
        () => symbolic.createGenerationContext({ chunkLength: 2, bindings: { C: 0 } }),
        ExecutionError,
      );
      assert(empty.message.includes("capacity 0"), empty.message);

      // A resolved symbolic capacity becomes the physical capacity as is (the input `cap` was never
      // passed).
      const context = await symbolic.createGenerationContext({
        chunkLength: 2,
        bindings: { C: 8 },
      });
      try {
        const slot = internals(context).slots.get("k");
        assertEquals(slot?.shape, [1, 2, 8, 4]);
        assertEquals(slot?.byteLength, 1 * 2 * 8 * 4 * 4);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      await symbolic.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a capacity whose single slot exceeds maxStorageBufferBindingSize is fail loudly (narrowed device, real GPU)",
  async () => {
    // A 1GiB slot of the same kind as the full layers of Gemma 4 E2B (131K capacity × f32). On a
    // machine with the core default of 128MiB it cannot be bound = the form ADR 0066 addendum 5
    // requires to be rejected.
    const capacity = 131072;
    const heads = 8;
    const headDim = 256;
    const slotBytes = heads * capacity * headDim * 4;
    assert(
      slotBytes > CORE_STORAGE_BINDING_LIMIT,
      `the synthetic slot's ${slotBytes}B does not exceed the core default limit (the check would miss)`,
    );
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits were not narrowed (if they were not narrowed, the check looks at nothing)",
      );
      const session = await stateSession(
        gpu,
        stateGraph({ big: { dtype: "f32", shape: [1, heads, capacity, headDim] } }),
      );
      try {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength: 1 }),
          ExecutionError,
        );
        assert(error.message.includes(`${slotBytes} bytes`), error.message);
        assert(error.message.includes(`${CORE_STORAGE_BINDING_LIMIT} bytes`), error.message);
        assertEquals(
          session.diagnostics().stateBacking.contextCount,
          0,
          "a rejected context is not counted (the diagnostics do not claim what has no data)",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Premise of the OOM check: an adapter that can bind one slot (1GiB), on which a request
 * **totalling 64GiB** exceeds the device's headroom. The former cannot be narrowed (`LIMIT_CAPS`
 * only lowers limits), so on an adapter whose binding limit is below 1GiB the capacity gate fails
 * first and the OOM path cannot be observed = SKIP explicitly.
 * On a machine that does not satisfy the latter (more than 64GiB of VRAM) this check **turns red**
 * (it does not pass silently).
 */
const OOM_SLOT_BYTES = 1024 * 1024 * 1024;
const OOM_SLOT_COUNT = 64;
const detectBindingLimit = async (): Promise<number> => {
  const adapter = await navigator.gpu?.requestAdapter();
  return adapter === null || adapter === undefined ? 0 : adapter.limits.maxStorageBufferBindingSize;
};
const BINDING_LIMIT: number = GPU_AVAILABLE ? await detectBindingLimit() : 0;
if (GPU_AVAILABLE && BINDING_LIMIT < OOM_SLOT_BYTES) {
  console.warn(
    "[karume] skipping the GenerationContext OOM check on an adapter whose " +
      "maxStorageBufferBindingSize is below 1GiB (the capacity gate fails first, so the allocation-failure path cannot be reached)",
  );
}

test.skipIf(!GPU_AVAILABLE || BINDING_LIMIT < OOM_SLOT_BYTES)(
  "a failed state allocation is fail loudly through an out-of-memory errorScope (real GPU)",
  async () => {
    const elements = OOM_SLOT_BYTES / 4;
    const states: DeclarationJson["states"] = {};
    for (let i = 0; i < OOM_SLOT_COUNT; i += 1) {
      states[`kv${i}`] = { dtype: "f32", shape: [1, 1, elements / 1024, 1024] };
    }
    const gpu = await acquireGpu();
    const session = await stateSession(gpu, stateGraph(states));
    // Detector for the clean-up on the failure path: OOM is the state of "headroom used up", so if
    // what was allocated had not been returned, this 1GiB could not be allocated (leaked data is
    // unreachable from dispose too, and the amount is at its maximum in this Session).
    const survivor = await stateSession(
      gpu,
      stateGraph({ kv: { dtype: "f32", shape: [1, 1, elements / 1024, 1024] } }),
    );
    try {
      const error = await assertRejects(
        () => session.createGenerationContext({ chunkLength: 1 }),
        GpuOutOfMemoryError,
      );
      assert(error.message.startsWith("GenerationContext state allocation: "), error.message);
      assertEquals(session.diagnostics().stateBacking.contextCount, 0);
      // Dawn releases a destroyed allocation only once the submission serial that follows the
      // destroy() has completed (wgpu freed it at destroy()); with nothing submitted since, the
      // memory would stay reserved. Submit an empty command buffer and wait for it so the survivor's
      // 1 GiB measures what the failure path actually returned.
      gpu.device.queue.submit([gpu.device.createCommandEncoder().finish()]);
      await gpu.device.queue.onSubmittedWorkDone();
      const context = await survivor.createGenerationContext({ chunkLength: 1 });
      assertEquals(context.pastLength, 0);
      await context.dispose();
    } finally {
      await session.dispose();
      await survivor.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a failure part way through allocation returns each allocated buffer exactly once (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    const create = (host: GenerationContextHost): Promise<GenerationContext> =>
      GenerationContext.create(host, { chunkLength: 1 });
    try {
      // The success path destroys none (the owner is the returned context — only dispose returns
      // them).
      const ok = injectFaults(gpu, {});
      const context = await create(injectedHost(ok.gpu, graph));
      assertEquals(
        ok.destroyCounts,
        [0, 0, 0],
        "allocated the 2 slots + the logical-length uniform",
      );
      await context.dispose();
      assertEquals(ok.destroyCounts, [1, 1, 1], "dispose returns each exactly once");

      // ① A synchronous throw part way through: failing at the second returns the first exactly
      //   once.
      const midway = injectFaults(gpu, { failCreateAt: 2 });
      await assertRejects(
        () => create(injectedHost(midway.gpu, graph)),
        Error,
        "injected: createBuffer #2",
      );
      assertEquals(midway.destroyCounts, [1], "the 1 allocated buffer does not leak");

      // ② A reject of pop: everything allocated is returned exactly once (with the pop outside the
      // try/finally, all 3 would become unreachable here and could not be returned from dispose
      // either).
      const rejected = injectFaults(gpu, {
        popError: new Error("injected: popErrorScope rejects"),
      });
      await assertRejects(
        () => create(injectedHost(rejected.gpu, graph)),
        Error,
        "injected: popErrorScope rejects",
      );
      assertEquals(rejected.destroyCounts, [1, 1, 1]);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a device loss during allocation becomes a GpuDeviceLostError and returns everything allocated (fault injection, real GPU)",
  async () => {
    // This case breaks the device, so it takes a dedicated GpuContext.
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    // Makes a window where pop never settles. Without passing the wait through raceDeviceLost, this
    // case would **hang** rather than "fail" (an implementation may never resolve popErrorScope
    // after a loss).
    const injected = injectFaults(gpu, { hangPop: true });
    const pending = GenerationContext.create(injectedHost(injected.gpu, graph), { chunkLength: 1 });
    assertEquals(
      injected.destroyCounts,
      [0, 0, 0],
      "allocation ends within the synchronous section of create",
    );

    gpu.destroy();
    const error = await assertRejects(() => pending, GpuDeviceLostError);
    assert(error.message.includes("GenerationContext state allocation"), error.message);
    assertEquals(
      injected.destroyCounts,
      [1, 1, 1],
      "even on a loss, every allocated buffer is returned",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the spec's chunkBuckets is published as a frozen copy and builds the allowed set (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // The caller's array is **copied** (rewriting it later does not split the allowed set from
      // the published surface).
      const buckets = [2, 4];
      const context = await session.createGenerationContext({
        chunkLength: 8,
        chunkBuckets: buckets,
      });
      try {
        assertEquals(context.chunkBuckets, [2, 4]);
        assert(Object.isFrozen(context.chunkBuckets), "the published surface is a frozen copy");
        buckets.push(6);
        assertEquals(context.chunkBuckets, [2, 4], "a later rewrite does not reach the context");

        // Allowed set = {1} ∪ chunkBuckets ∪ {chunkLength} (the only form the pre-run check reads).
        assertEquals([...internals(context).allowedRows], [1, 2, 4, 8]);
        // chunkLength stays the maximum (the upper bound of queryLength does not move either).
        assertEquals(context.chunkLength, 8);
        assertThrows(() => internals(context).writeLengths(0, 9), ExecutionError, "chunkLength");
      } finally {
        await context.dispose();
      }

      // A context that omits it has, as before, one prefill shape + the decode shape.
      const plain = await session.createGenerationContext({ chunkLength: 8 });
      try {
        assertEquals(plain.chunkBuckets, []);
        assertEquals([...internals(plain).allowedRows], [1, 8]);
      } finally {
        await plain.dispose();
      }

      // Range and order violations fail before allocation (the source of truth for the accepted set
      // is assertChunkBuckets).
      for (const chunkBuckets of [[1], [8], [4, 3]]) {
        const error = await assertRejects(
          () => session.createGenerationContext({ chunkLength: 8, chunkBuckets }),
          ExecutionError,
        );
        assert(error.message.includes("chunkBuckets"), error.message);
      }
      // The cumulative count stays at the 2 above (a rejected spec allocated nothing).
      assertEquals(session.diagnostics().stateBacking.contextCount, 2);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a push to chunkBuckets across the allocation await does not reach the allowed set (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // There is a state-allocation await between the check (assertChunkBuckets) and the
      // constructor call, so a form that re-reads from the spec would put a rewrite made in **this
      // window** into the allowed set unchecked. Pushing inside the window pins that the
      // implementation carries its copy along (a different window from the test above, which looks
      // only at a synchronous push — here create is suspended).
      const buckets = [2, 4];
      const pending = session.createGenerationContext({ chunkLength: 8, chunkBuckets: buckets });
      buckets.push(6);
      const context = await pending;
      try {
        assertEquals(
          context.chunkBuckets,
          [2, 4],
          "a push across the allocation await does not reach it",
        );
        assertEquals([...internals(context).allowedRows], [1, 2, 4, 8]);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "rewriting chunkLength across the allocation await does not reach the context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // Closes the same window for the length as for the array. A form that re-reads from `spec`
      // would pass the check with 8 and still give the context an upper bound of 1 (the allowed set
      // shrinks to {1} too) — with no exception and no warning, every later prefill would then fail
      // at the queryLength check.
      const spec = { chunkLength: 8 };
      const pending = session.createGenerationContext(spec);
      spec.chunkLength = 1;
      const context = await pending;
      try {
        assertEquals(
          context.chunkLength,
          8,
          "the value that passed the check is the context's upper bound",
        );
        assertEquals([...internals(context).allowedRows], [1, 8]);
        // The upper bound has not shrunk to the rewritten 1 (a run of 8 rows passes).
        internals(context).writeLengths(0, 8);
      } finally {
        await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the logical length advances on a successful run, and rewind accepts only integers in 0..pastLength (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // It advances for both a prefill chunk (queryLength ≤ chunkLength) and decode (1).
      // The first argument is the pastLength the run captured at its start (checked against the
      // context's current value).
      internals(context).advance(0, 4);
      internals(context).advance(4, 1);
      assertEquals(context.pastLength, 5);

      for (const queryLength of [0, -1, 1.5, 5]) {
        assertThrows(
          () => internals(context).advance(5, queryLength),
          ExecutionError,
          "queryLength",
        );
      }
      assertEquals(context.pastLength, 5, "a rejected advance does not move the logical length");

      // An advance whose captured P differs from the current value is fail loudly (with a lease it
      // cannot happen = a broken internal invariant).
      for (const captured of [4, 6]) {
        const split = assertThrows(() => internals(context).advance(captured, 1), ExecutionError);
        assert(split.message.includes("differs from"), split.message);
      }
      assertThrows(() => internals(context).writeLengths(4, 1), ExecutionError, "differs from");
      assertEquals(context.pastLength, 5, "a rejected advance does not move the logical length");

      context.rewind(2);
      assertEquals(context.pastLength, 2);
      context.rewind(2);
      assertEquals(context.pastLength, 2, "a rewind to the same position is a no-op");
      context.rewind(0);
      assertEquals(context.pastLength, 0);

      internals(context).advance(0, 3);
      for (const position of [-1, 4, 1.5, Number.NaN]) {
        assertThrows(() => context.rewind(position), ExecutionError, "rewind");
      }
      assertEquals(context.pastLength, 3, "a rejected rewind does not move the logical length");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the logical length is fail loudly at the u32 limit (blocks silent truncation into a Uint32Array, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    try {
      // Exactly the limit is accepted (chunkLength is a plan-time constant; only the slot capacity
      // decides the physical allocation).
      const context = await session.createGenerationContext({ chunkLength: U32_MAX });
      try {
        assertEquals(context.chunkLength, U32_MAX);
        // A queryLength above u32 fails at the chunkLength check (it is not truncated to 0 and
        // written).
        assertThrows(
          () => internals(context).writeLengths(0, U32_MAX + 1),
          ExecutionError,
          "chunkLength",
        );
        // An overflow after the addition can only be seen here (the sum overflows even when both
        // terms are within u32).
        internals(context).advance(0, U32_MAX);
        assertEquals(context.pastLength, U32_MAX);
        const overflow = assertThrows(() => internals(context).advance(U32_MAX, 1), ExecutionError);
        assert(overflow.message.includes("u32 limit"), overflow.message);
        assertEquals(
          context.pastLength,
          U32_MAX,
          "a rejected advance does not move the logical length",
        );
        // The rewind position is bounded by pastLength too, so it cannot exceed u32.
        assertThrows(() => context.rewind(U32_MAX + 1), ExecutionError, "rewind");
      } finally {
        await context.dispose();
      }

      // A chunkLength of the limit + 1 is rejected (accepting it would open the queryLength check
      // beyond u32).
      const rejected = await assertRejects(
        () => session.createGenerationContext({ chunkLength: U32_MAX + 1 }),
        ExecutionError,
      );
      assert(rejected.message.includes("chunkLength"), rejected.message);
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "after poisoning, every operation except dispose is rejected (reads included, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    try {
      internals(context).advance(0, 2);
      internals(context).poison("a run containing a state-changing dispatch failed");
      // A second poisoning does not overwrite the root cause.
      internals(context).poison("a later, different failure");

      // "Every later operation is fail loudly" of ADR 0066 addendum 3 — reading `pastLength` is
      // behind the barrier too (returning the logical length of an unrecoverable context as a valid
      // value makes the host always read it as "generation can resume from here").
      for (const operation of [
        () => context.pastLength,
        () => context.pendingCommit,
        () => context.rewind(0),
        () => context.commit(0),
        () => internals(context).pastLength(),
        () => internals(context).acquireRun(),
        () => internals(context).advance(2, 1),
        () => internals(context).writeLengths(2, 1),
      ]) {
        const error = assertThrows(operation, ExecutionError);
        assert(error.message.includes("poisoned"), error.message);
        assert(
          error.message.includes("a run containing a state-changing dispatch failed"),
          `the root cause is gone: ${error.message}`,
        );
      }
    } finally {
      // A poisoned context can still be disposed (the physical buffers can be returned).
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "after a device loss every operation of a GenerationContext is a GpuDeviceLostError (real GPU)",
  async () => {
    // This case breaks the device, so it takes a dedicated GpuContext.
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    internals(context).advance(0, 1);

    gpu.destroy();
    await gpu.device.lost;
    assert(gpu.lost !== undefined, "the loss was not recorded (the check would miss)");

    assertThrows(() => context.pastLength, GpuDeviceLostError, "device was lost");
    assertThrows(() => context.rewind(0), GpuDeviceLostError);
    assertThrows(() => internals(context).acquireRun(), GpuDeviceLostError);
    assertThrows(() => internals(context).advance(1, 1), GpuDeviceLostError);
    assertThrows(() => internals(context).writeLengths(1, 1), GpuDeviceLostError);

    // dispose "returns the buffers, then propagates the flush failure" (the same discipline as
    // RunArena).
    const failure = await assertRejects(() => context.dispose(), GpuDeviceLostError);
    const again = await assertRejects(() => context.dispose(), GpuDeviceLostError);
    assertStrictEquals(again, failure, "later calls return the same completion (the same failure)");
    await session.dispose().catch(() => undefined);
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "every operation is rejected right after device.destroy() (before lost reacts, real GPU)",
  async () => {
    // This case breaks the device, so it takes a dedicated GpuContext.
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 2 });
    internals(context).advance(0, 1);

    gpu.destroy();
    // MUST: lost must **not be recorded yet** here. If it were, the existing lost check alone would
    // pass, which is no evidence that "the synchronous destroy flag is checked" (the check would be
    // vacuous).
    assertStrictEquals(gpu.lost, undefined, "the lost reaction ran right after destroy");
    assert(gpu.destroyRequested, "the destroy request was not set synchronously");

    assertThrows(() => context.pastLength, GpuDeviceLostError, "device was lost");
    assertThrows(() => context.rewind(0), GpuDeviceLostError);
    assertThrows(() => internals(context).advance(1, 1), GpuDeviceLostError);
    // writeLengths is the most dangerous hole (a writeBuffer to a destroyed buffer is a no-op that
    // does not even warn, and the host's logical length and the value the GPU sees silently split).
    assertThrows(() => internals(context).writeLengths(1, 1), GpuDeviceLostError);
    // The creation entry uses the same check (no context with an empty KV may be made).
    await assertRejects(
      () => session.createGenerationContext({ chunkLength: 1 }),
      GpuDeviceLostError,
    );

    await session.dispose().catch(() => undefined);
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the logical-length uniform is rewritten in full every time, and the written values are on the GPU (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    /**
     * Reads the logical-length uniform back through staging (the only observation point that
     * detects a no-op writeBuffer).
     */
    const readLengths = async (): Promise<readonly number[]> => {
      const staging = gpu.device.createBuffer({
        size: LENGTHS_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = gpu.device.createCommandEncoder();
        encoder.copyBufferToBuffer(internals(context).lengths, 0, staging, 0, LENGTHS_BYTES);
        gpu.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        // The first two words are the lengths; the block's words have their own test.
        const words = Array.from(new Uint32Array(staging.getMappedRange().slice(0, 8)));
        staging.unmap();
        return words;
      } finally {
        staging.destroy();
      }
    };
    try {
      internals(context).writeLengths(0, 4);
      assertEquals(await readLengths(), [0, 4], "the first prefill is past 0 / query 4");

      internals(context).advance(0, 4);
      internals(context).writeLengths(4, 3);
      assertEquals(await readLengths(), [4, 3], "past and query are independent scalars");

      // Moving to the decode shape (queryLength=1) changes only the query word = evidence that the
      // whole range is rewritten.
      internals(context).advance(4, 3);
      internals(context).writeLengths(7, 1);
      assertEquals(await readLengths(), [7, 1]);

      // A rewind takes effect directly from the next writeLengths (the context owns the logical
      // length).
      context.rewind(2);
      internals(context).writeLengths(2, 1);
      assertEquals(await readLengths(), [2, 1]);

      for (const queryLength of [0, 5, 1.5]) {
        assertThrows(() => internals(context).writeLengths(2, queryLength), ExecutionError);
      }
      assertEquals(await readLengths(), [2, 1], "a rejected write does not touch the uniform");
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// In wave C this checked that "even with 2 contexts, the second run hits the same plan key".
// Wave D-1 added reference completeness, so **a graph that declares states always has state-
// referencing nodes**, and a 1-shot `run` no longer works (the GenerationContext owns the slot
// data, and the execution path was wired in wave D-3). What this pins was rearranged into 2 points:
// ① creating a context runs the plan derivation not even once (if the context were in the key, the
//   derivation would run)
// ② a 1-shot run of a graph with state-referencing nodes is fail loudly (it does not silently run
//   without state)
// NOTE: "hits the same key (hit: true)" = the run side of acceptance condition ③ of ADR 0066 was
// wired in wave D-3, and tests/gpu_state_execution_test.ts holds it in the form "another context of
// the same capacity: the second one hits".
test.skipIf(!GPU_AVAILABLE)(
  "creating a context runs no plan derivation, and a 1-shot run of a state-referencing graph is rejected (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const input: Tensor = {
      dtype: "f32",
      shape: [2, 4],
      data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
    };
    try {
      const baseline = session.diagnostics();
      assertEquals(baseline.lastRunPrepared, undefined, "no plan record before a run");

      const contexts: GenerationContext[] = [
        await session.createGenerationContext({ chunkLength: 4 }),
        await session.createGenerationContext({ chunkLength: 1 }),
      ];
      try {
        const after = session.diagnostics();
        assertEquals(
          after.lastRunPrepared,
          undefined,
          "creating a context runs no plan derivation",
        );
        assertEquals(after.planBacking.buildCount, 0, "no slot backing is baked either");
        assertEquals(after.stateBacking.contextCount, 2, "only the diagnostics count the contexts");

        // It fails in the shape computation layer (OpContractError) — a run without the slots'
        // resolved shapes cannot get further than that.
        const error = await assertRejects(() => session.run({ x: input }), OpContractError);
        assert(error.message.includes("GenerationContext"), error.message);
        assertEquals(
          session.diagnostics().lastRunPrepared,
          undefined,
          "a rejected run does not pollute the plan cache",
        );
      } finally {
        for (const context of contexts) await context.dispose();
      }
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

// The lifetime contract of the 3rd argument of `Session.run` (R6-1 / CG5-2). The run body goes
// through `#serialize` (one microtask or later) and reads `context` / `queryLength` at several
// points across awaits, so this pins 2 points:
//   ① **snapshot at issue** (the same discipline as `inputs` / `bindings`) — rewriting the argument
//     object after issuing does not change this run, which uses the 2 fields as of issue (the lease
//     is set from the same copy).
//   ② **at most 1 unsettled run per context** (the second is fail loudly in the synchronous section
//     of issue).
// Breaking either gives a position shift / a split logical length with no exception or warning, so
// they are pinned by contract, not by values.

/**
 * A mutable 3rd argument for the snapshot-at-issue test (the 2 fields of `GenerationRun` are
 * readonly, so it holds a copy).
 */
type MutableGenerationRun = { context: GenerationContext; queryLength: number };

const RUN_INPUT: Tensor = {
  dtype: "f32",
  shape: [2, 4],
  data: Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
};

test.skipIf(!GPU_AVAILABLE)(
  "a run's queryLength is fixed at issue (a rewrite after issue is not read, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      const generation: MutableGenerationRun = { context, queryLength: 4 };
      const running = session.run({ x: RUN_INPUT }, {}, generation);
      // Rewrite it before the body runs (in the same tick). An implementation that carries the
      // reference would dispatch for 4 rows while only the uniform and the advance become 1 (= the
      // GPU's scan range and the host's bookkeeping split).
      generation.queryLength = 1;
      await running;
      assertEquals(
        context.pastLength,
        4,
        "it runs with the queryLength at issue and advances by exactly that",
      );
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a run's context is fixed at issue, and the lease is set on the same copy (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const first = await session.createGenerationContext({ chunkLength: 4 });
    const second = await session.createGenerationContext({ chunkLength: 4 });
    try {
      const generation: MutableGenerationRun = { context: first, queryLength: 1 };
      const running = session.run({ x: RUN_INPUT }, {}, generation);
      generation.context = second;
      // The lease is set on the copy at issue (= first). After the rewrite, this one still fails
      // with "a run in progress".
      const blocked = assertThrows(() => first.rewind(0), ExecutionError);
      assert(blocked.message.includes("generation run is in progress"), blocked.message);
      // The replacement takes no part in the run, so its rewind passes through (evidence that the
      // lease did not leak).
      second.rewind(0);
      await running;
      assertEquals(first.pastLength, 1, "the context at issue is the one that wrote the KV");
      assertEquals(second.pastLength, 0, "the replacement does not advance a single row");
    } finally {
      await first.dispose();
      await second.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "concurrent un-awaited issues to the same context reject the second (another context passes, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    const other = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // MUST: settle both together (the rejection is raised in the synchronous section of issue, so
      // an await in between would fail the test before the rejection is observed as unhandled).
      const [first, second] = await Promise.allSettled([
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
      ]);
      assertEquals(first.status, "fulfilled", "the first passes as before");
      assertEquals(second.status, "rejected");
      assert(second.status === "rejected");
      assert(
        second.reason instanceof ExecutionError,
        `wrong rejection type: ${String(second.reason)}`,
      );
      assert(second.reason.message.includes("issued concurrently"), second.reason.message);
      assertEquals(context.pastLength, 1, "only the first one's rows advanced it");

      // Concurrent issues to another context work as before (only the same context is blocked).
      const parallel = await Promise.allSettled([
        session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        session.run({ x: RUN_INPUT }, {}, { context: other, queryLength: 1 }),
      ]);
      assertEquals(
        parallel.map((result) => result.status),
        ["fulfilled", "fulfilled"],
      );
      assertEquals(context.pastLength, 2);
      assertEquals(other.pastLength, 1);
    } finally {
      await context.dispose();
      await other.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// deferred commit (the verify form of speculative decoding — the logical length does not advance on
// a successful run but is confirmed by the accepted row count)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "a deferred run holds the logical length back, and commit(rows) advances it by the accepted row count (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    /**
     * Reads the logical-length uniform back (the observation point that "the past the GPU saw"
     * follows commit).
     */
    const readPast = async (): Promise<number> => {
      const staging = gpu.device.createBuffer({
        size: LENGTHS_BYTES,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      try {
        const encoder = gpu.device.createCommandEncoder();
        encoder.copyBufferToBuffer(internals(context).lengths, 0, staging, 0, LENGTHS_BYTES);
        gpu.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        return words[0];
      } finally {
        staging.destroy();
      }
    };
    try {
      // The default (no commit field) advances on a successful run as before.
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 });
      assertEquals(context.pastLength, 1);
      assertEquals(context.pendingCommit, undefined, "an immediate run creates no pending commit");

      // deferred writes 3 rows and holds the logical length.
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 3, commit: "deferred" });
      assertEquals(context.pastLength, 1, "deferred does not advance on a successful run");
      assertEquals(context.pendingCommit, { pastLength: 1, queryLength: 3 });

      // While a commit is pending, neither the next run nor a rewind passes (the logical length
      // still has only one path that moves it).
      const rejected = await assertRejects(
        () => session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 }),
        ExecutionError,
      );
      assert(rejected.message.includes("awaiting commit"), rejected.message);
      const blocked = assertThrows(() => context.rewind(0), ExecutionError);
      assert(blocked.message.includes("awaiting commit"), blocked.message);

      // Range: a non-negative integer, up to the rows written. A rejection moves neither the
      // pending commit nor the logical length.
      for (const rows of [-1, 1.5, 4]) {
        assertThrows(() => context.commit(rows), ExecutionError, "commit");
      }
      assertEquals(context.pendingCommit, { pastLength: 1, queryLength: 3 });
      assertEquals(context.pastLength, 1);

      // Accepting 2 rows = the third is discarded (it stays written in the physical ring but is not
      // in the logical length).
      context.commit(2);
      assertEquals(context.pastLength, 3);
      assertEquals(context.pendingCommit, undefined, "commit clears the pending commit");

      // A commit without a pending one is fail loudly (a silent no-op would apply the accepted row
      // count twice).
      const orphan = assertThrows(() => context.commit(1), ExecutionError);
      assert(orphan.message.includes("no generation run is awaiting commit"), orphan.message);

      // The past the next run carries is the value after commit (the host's bookkeeping and the
      // value the GPU sees agree).
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 1 });
      assertEquals(await readPast(), 3, "the past the GPU saw does not reflect the commit");
      assertEquals(context.pastLength, 4);

      // commit(0) means "accept no rows" — the logical length does not move and only the pending
      // commit is cleared.
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      context.commit(0);
      assertEquals(context.pastLength, 4, "commit(0) does not move the logical length");
      assertEquals(context.pendingCommit, undefined);

      // A misspelling coming from outside the type is rejected rather than defaulted.
      const spelling = await assertRejects(
        () =>
          session.run(
            { x: RUN_INPUT },
            {},
            { context, queryLength: 1, commit: "defered" as "deferred" },
          ),
        ExecutionError,
      );
      assert(spelling.message.includes("'immediate' or 'deferred'"), spelling.message);
      assertEquals(
        context.pastLength,
        4,
        "a rejected run moves neither the logical length nor the pending commit",
      );
      assertEquals(context.pendingCommit, undefined);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "pendingCommit is returned frozen, and the upper bound of the accepted row count cannot be rewritten from outside (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      const pending = context.pendingCommit;
      assert(pending !== undefined, "the deferred run did not create a pending commit");
      assert(Object.isFrozen(pending), "the object carrying commit's upper bound MUST be frozen");
      // TS readonly does not affect assignment compatibility, so the public surface can be
      // rewritten in this direction **without a type check** (`commit` trusts this `queryLength` as
      // "the rows written"). Freezing is the only line of detection, and in strict mode the
      // assignment itself is a TypeError.
      const writable: { pastLength: number; queryLength: number } = pending;
      assertThrows(() => {
        writable.queryLength = 4;
      }, TypeError);
      assertEquals(
        context.pendingCommit,
        { pastLength: 0, queryLength: 2 },
        "the upper bound did not move",
      );

      // Confirming rows that were not written is still rejected (if the upper bound had opened to
      // 4, this would pass).
      const over = assertThrows(() => context.commit(3), ExecutionError);
      assert(over.message.includes("queryLength 2"), over.message);
      context.commit(2);
      assertEquals(context.pastLength, 2);
    } finally {
      await context.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a failed deferred run creates no pending commit / dispose clears it along with the context (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    try {
      // A run that fails at the queryLength check (it queues no dispatch = the physical ring is
      // intact).
      await assertRejects(
        () => session.run({ x: RUN_INPUT }, {}, { context, queryLength: 5, commit: "deferred" }),
        ExecutionError,
      );
      assertEquals(context.pendingCommit, undefined, "a failed run created a pending commit");
      assertEquals(context.pastLength, 0);

      // Disposing with a pending commit left clears it (there is nothing to confirm it against, so
      // it is dropped).
      await session.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" });
      assertEquals(context.pendingCommit, { pastLength: 0, queryLength: 2 });
      await context.dispose();
      assertThrows(() => context.commit(1), ExecutionError, "disposed");
      assertThrows(() => context.pendingCommit, ExecutionError, "disposed");
    } finally {
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * A graph with one sliding slot (window 8). The capacity is the symbol `C`, given from the context
 * side.
 */
const slidingGraph = (): DeclarationJson =>
  stateGraph(
    { k: { dtype: "f32", shape: [1, 2, "C", 4] } },
    { symbols: ["T", "C"] },
    { window: 8 },
  );

test.skipIf(!GPU_AVAILABLE)(
  "slidingSlack is the minimum of capacity − window over the sliding slots (undefined without sliding)",
  async () => {
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const sliding = await stateSession(gpu, slidingGraph());
    try {
      // A context with only full slots has no notion of slack (do not confuse undefined with 0).
      const full = await session.createGenerationContext({ chunkLength: 1 });
      try {
        assertEquals(full.slidingSlack, undefined);
      } finally {
        await full.dispose();
      }

      // Capacity 16 / window 8 → slack 8. Exactly the capacity (8) gives 0.
      for (const [capacity, slack] of [
        [16, 8],
        [8, 0],
      ] as const) {
        const context = await sliding.createGenerationContext({
          chunkLength: 1,
          bindings: { C: capacity },
        });
        try {
          assertEquals(context.slidingSlack, slack, `C=${capacity}`);
        } finally {
          await context.dispose();
        }
      }

      // window > capacity is fail loudly at allocation (rejects ③ of ADR 0067 decision 4 before any
      // run).
      const tooNarrow = await assertRejects(
        () => sliding.createGenerationContext({ chunkLength: 1, bindings: { C: 4 } }),
        ExecutionError,
      );
      assert(tooNarrow.message.includes("window 8"), tooNarrow.message);
    } finally {
      await sliding.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a deferred run's queryLength is limited to the sliding slack (an issue beyond it is rejected in the synchronous section, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const sliding = await stateSession(gpu, slidingGraph());
    // Capacity 9 / window 8 → slack 1. deferred allows up to the "slack" = 1 row (the lower end of
    // the window of a borrower = a readonly reader is `P+m−W`, one column lower than the lender, so
    // the condition is `Q ≤ slack`).
    const context = await sliding.createGenerationContext({ chunkLength: 4, bindings: { C: 9 } });
    try {
      assertEquals(context.slidingSlack, 1);
      // A 2-row deferred fails in the synchronous section of issue, leaving neither a lease nor a
      // pending commit (even with 0 accepted rows, the discarded rows would fall outside the
      // borrower's live window, which exceeds the slack condition).
      const rejected = await assertRejects(
        () => sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 2, commit: "deferred" }),
        ExecutionError,
      );
      assert(rejected.message.includes("slack 1"), rejected.message);
      assertEquals(context.pendingCommit, undefined);
      assertEquals(context.pastLength, 0);
      // immediate confirms every row, so the upper bound stays chunkLength.
      await sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 3 });
      assertEquals(context.pastLength, 3);
      // Exactly the slack passes.
      await sliding.run({ x: RUN_INPUT }, {}, { context, queryLength: 1, commit: "deferred" });
      assertEquals(context.pendingCommit, { pastLength: 3, queryLength: 1 });
      context.commit(0);
      assertEquals(context.pastLength, 3);
    } finally {
      await context.dispose();
      await sliding.dispose();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// fork (branching the conversation state — ADR 0066 addendum of 2026-10-11)
// ---------------------------------------------------------------------------
//
// It pins 4 things:
//   ① what is copied: every byte of every slot (read back as u32), pastLength and the immutable
//     values. Baked bundles are not copied
//   ② independence of the branches: for the same inputs the outputs of the original and the copy
//     match as u32, and advancing only the copy does not move the original's bytes
//   ③ the contract of the synchronous section: a run / rewind / dispose / second fork issued to the
//     original in the same tick without awaiting `fork()` all pass, and the copy has the state as
//     of the call
//   ④ the rejection matrix ((g) borrowers are in gpu_borrowed_context_test.ts) and the diagnostics
// That the ②③ checks are not vacuous is shown with 2 kinds of fault injection (a copy dispatch that
// binds the original's slots, and a host that inserts an await before the copy).

/**
 * Slot capacity and depth of the fork checks (full attention, 1 head — the smallest form whose
 * values depend on the slot contents).
 */
const FORK_CAPACITY = 16;
const FORK_DEPTH = 4;
const FORK_SLOT_BYTES = FORK_CAPACITY * FORK_DEPTH * 4;

/**
 * A graph that reads the written KV (full attention + 2 `state_append`). The output `o` depends on
 * the contents of the past KV, so "the same inputs give the same output" observes the identity of
 * the state.
 */
const forkGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["attention", "state_append"] },
  symbols: ["M"],
  inputs: ["q", "k", "v"].map((name) => ({
    name,
    dtype: "f32",
    shape: [1, 1, "M", FORK_DEPTH],
  })),
  outputs: ["o"],
  initializers: {},
  values: { o: { dtype: "f32", shape: [1, 1, "M", FORK_DEPTH] } },
  states: {
    kslot: { dtype: "f32", shape: [1, 1, FORK_CAPACITY, FORK_DEPTH] },
    vslot: { dtype: "f32", shape: [1, 1, FORK_CAPACITY, FORK_DEPTH] },
  },
  nodes: [
    {
      op: "attention",
      ins: ["q", "k", "v"],
      outs: ["o"],
      attrs: { scale: 0.5 },
      states: { k: "kslot", v: "vslot" },
    },
    ...[
      ["k", "kslot"],
      ["v", "vslot"],
    ].map(([name, slot]) => ({
      op: "state_append",
      ins: [name],
      outs: [],
      attrs: {},
      states: { slot },
    })),
  ],
});

const forkSession = async (gpu: GpuContext): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(forkGraph()), GRAPH_NAME);

/**
 * Deterministic input whose contents change with `salt` (`rows` rows — no pad rows = valid rows =
 * physical rows).
 */
const forkInputs = (rows: number, salt: number): Record<string, Tensor> =>
  Object.fromEntries(
    ["q", "k", "v"].map((name, n) => [
      name,
      {
        dtype: "f32",
        shape: [1, 1, rows, FORK_DEPTH],
        data: Float32Array.from(
          { length: rows * FORK_DEPTH },
          (_, i) => (((i * 7 + n * 3 + salt * 5) % 17) - 8) * 0.13,
        ),
      } satisfies Tensor,
    ]),
  );

/**
 * 1 step (runs all `rows` rows as valid rows).
 *
 * MUST: do not make it async — the run is issued **in the synchronous section that calls**
 * `session.run` (the checks that line things up in the same tick rely on this property).
 */
const forkStep = (
  session: Session,
  context: GenerationContext,
  rows: number,
  salt: number,
): Promise<Record<string, Tensor>> =>
  session.run(forkInputs(rows, salt), {}, { context, queryLength: rows });

/** Output bits (compared as u32 — checks identity, not closeness of values). */
const outputWords = (outputs: Record<string, Tensor>): number[] => {
  const data = outputs["o"].data;
  return Array.from(new Uint32Array(data.buffer, data.byteOffset, data.length));
};

/** Reads the whole GPU buffer back as a u32 sequence. */
const readWords = async (
  gpu: GpuContext,
  buffer: GPUBuffer,
  byteLength: number,
): Promise<number[]> => {
  const staging = gpu.device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const words = Array.from(new Uint32Array(staging.getMappedRange().slice(0)));
    staging.unmap();
    return words;
  } finally {
    staging.destroy();
  }
};

/** Reads back every slot of the context (slot name → u32 sequence of all its bytes). */
const readSlotWords = async (
  gpu: GpuContext,
  context: GenerationContext,
): Promise<Record<string, number[]>> => {
  const words: Record<string, number[]> = {};
  for (const [name, slot] of internals(context).slots) {
    words[name] = await readWords(gpu, slot.buffer, slot.byteLength);
  }
  return words;
};

test.skipIf(!GPU_AVAILABLE)(
  "fork copies every byte of every slot and the logical length, and the branches advance independently (u32, with diagnostics, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4, chunkBuckets: [2] });
    let forked: GenerationContext | undefined;
    try {
      // 1 prefill + 2 decodes. The second decode hits the derived plan and creates a backing, so
      // the original context has baked once for the decode-shape backing (the starting point of the
      // re-bake count).
      await forkStep(session, source, 4, 0);
      await forkStep(session, source, 1, 1);
      await forkStep(session, source, 1, 2);
      assertEquals(source.pastLength, 6);
      const before = session.diagnostics().stateBacking;
      const words = await readSlotWords(gpu, source);
      // The expected values are not trivial (if the written rows stayed 0, a missed copy would
      // match too).
      for (const [name, slot] of Object.entries(words)) {
        assert(
          slot.some((word) => word !== 0),
          `${name}: the original slot is all 0 (the check would miss)`,
        );
      }

      forked = await source.fork();
      // ① What is copied: the logical length and the immutable values, every byte of every slot.
      //   The data is in separate buffers.
      assertEquals(forked.pastLength, 6);
      assertEquals(forked.chunkLength, 4);
      assertEquals(forked.chunkBuckets, [2]);
      assertEquals([...internals(forked).allowedRows], [1, 2, 4]);
      assertEquals(forked.slidingSlack, undefined);
      assertEquals(forked.pendingCommit, undefined);
      for (const [name, slot] of internals(forked).slots) {
        assert(
          slot.buffer !== internals(source).slots.get(name)?.buffer,
          `${name}: the copy shares the original's buffer`,
        );
      }
      assertEquals(await readSlotWords(gpu, forked), words, "every byte of every slot matches");
      // Diagnostics: residency grows by one original's worth and the cumulative count by 1. The
      // copy itself bakes nothing.
      assertEquals(
        session.diagnostics().stateBacking,
        {
          residentBytes: before.residentBytes + FORK_SLOT_BYTES * 2 + LENGTHS_BYTES,
          contextCount: before.contextCount + 1,
          rebindCount: before.rebindCount,
        },
        "the copy keeps the 2 slots + the logical-length uniform resident, and no re-bake happens until a run",
      );

      // ② Advancing both with the same decode input gives u32-identical outputs. The only re-bake
      // is the copy's first run (once per backing used); the original's continuing runs add none.
      const fromSource = await forkStep(session, source, 1, 3);
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        before.rebindCount,
        "the original's continuing run re-baked (it threw away the original's bundles)",
      );
      const fromFork = await forkStep(session, forked, 1, 3);
      assertEquals(
        session.diagnostics().stateBacking.rebindCount,
        before.rebindCount + 1,
        "the copy's first run bakes exactly once per backing used",
      );
      assertEquals(
        outputWords(fromFork),
        outputWords(fromSource),
        "the outputs for the same input match as u32",
      );
      const sourceWords = await readSlotWords(gpu, source);
      assertEquals(
        await readSlotWords(gpu, forked),
        sourceWords,
        "the 2 advanced with the same input have the same bytes",
      );

      // Advancing only the copy does not move the original's bytes (positive control: the copy's
      // run wrote to its own slots).
      await forkStep(session, forked, 1, 9);
      assertEquals(session.diagnostics().stateBacking.rebindCount, before.rebindCount + 1);
      assertEquals(
        await readSlotWords(gpu, source),
        sourceWords,
        "the copy's run wrote to the original's slots",
      );
      assertNotEquals(
        await readSlotWords(gpu, forked),
        sourceWords,
        "the copy's run did not write",
      );
      assertEquals([source.pastLength, forked.pastLength], [7, 8]);

      // The original may be disposed first (no parent–child dependency).
      await source.dispose();
      await forkStep(session, forked, 1, 10);
      assertEquals(forked.pastLength, 9);
      assertEquals(
        session.diagnostics().stateBacking.residentBytes,
        FORK_SLOT_BYTES * 2 + LENGTHS_BYTES,
        "after returning the original only the copy's share remains",
      );
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "even with a run issued to the original in the same tick without awaiting fork(), the copy has the state as of the call (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    let forked: GenerationContext | undefined;
    try {
      await forkStep(session, source, 4, 0);
      const words = await readSlotWords(gpu, source);
      // Issue fork → run in that order in the same synchronous section. fork sets no reservation on
      // the original, so the run is not rejected, and the copy has the contents as of the fork call
      // (before the run's writes).
      const forking = source.fork();
      const running = forkStep(session, source, 1, 5);
      [forked] = await Promise.all([forking, running]);
      assertEquals(
        await readSlotWords(gpu, forked),
        words,
        "the copy picked up the writes of the later run",
      );
      assertEquals(forked.pastLength, 4);
      assertEquals(source.pastLength, 5);
      // Positive control: the run in the same tick really writes to the original's slots (if it did
      // not, the match above would check nothing about the order).
      assertNotEquals(
        await readSlotWords(gpu, source),
        words,
        "the run in the same tick did not write",
      );
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a rewind / second fork / dispose right after fork() (before the await) pass (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    const forks: GenerationContext[] = [];
    try {
      await forkStep(session, source, 4, 0);
      const words = await readSlotWords(gpu, source);
      // All issued in the same synchronous section (fork takes no reservation, so it does not block
      // the original's later operations).
      const first = source.fork();
      source.rewind(2);
      const second = source.fork();
      const disposal = source.dispose();
      // A fork after a dispose request is rejected (rejection (a)).
      const late = assertRejects(() => source.fork(), ExecutionError, "disposed");
      forks.push(...(await Promise.all([first, second])));
      await disposal;
      await late;

      const [atFour, atTwo] = forks;
      assertEquals(
        [atFour.pastLength, atTwo.pastLength],
        [4, 2],
        "the logical length is the value as of the call",
      );
      // rewind moves only the logical length, so the bytes of both are the original's contents
      // after the prefill.
      assertEquals(await readSlotWords(gpu, atFour), words);
      assertEquals(await readSlotWords(gpu, atTwo), words);
      // Even after the original is returned, both copies advance independently.
      await forkStep(session, atFour, 1, 6);
      await forkStep(session, atTwo, 1, 6);
      assertEquals([atFour.pastLength, atTwo.pastLength], [5, 3]);
      assertEquals(session.diagnostics().stateBacking.contextCount, 3);
    } finally {
      for (const forked of forks) await forked.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "fork rejections: poisoned, run in progress, reserved by a batch, awaiting commit, disposed Session (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const sink = await gpu.createResident(FORK_DEPTH * 4);
    const contexts: GenerationContext[] = [];
    const fresh = async (): Promise<GenerationContext> => {
      const context = await session.createGenerationContext({ chunkLength: 4 });
      contexts.push(context);
      return context;
    };
    try {
      // (c) After poisoning (no copying a context whose underlying physical state is lost).
      const poisoned = await fresh();
      internals(poisoned).poison("fault injected by the test");
      await assertRejects(() => poisoned.fork(), ExecutionError, "poisoned");

      // (d) An unsettled run (the reservation is set in the synchronous section of issue, so an
      // un-awaited sequence exercises it).
      const busy = await fresh();
      const running = forkStep(session, busy, 1, 0);
      await assertRejects(() => busy.fork(), ExecutionError, "generation run is in progress");
      await running;
      contexts.push(await busy.fork());

      // (e) Reserved for use by a batch (even after the enqueue Promise returns, the reservation
      // remains until the section settles).
      const batched = await fresh();
      const batch = await gpu.beginBatch();
      try {
        await session.enqueue(forkInputs(1, 0), {
          batch,
          generation: { context: batched, queryLength: 1 },
          copyOutputs: { o: sink },
        });
        await assertRejects(
          () => batched.fork(),
          ExecutionError,
          "waiting for the final settlement of a batch",
        );
      } finally {
        await batch.finish();
      }
      contexts.push(await batched.fork());

      // (f) Awaiting a deferred commit (how much of the rows written to the physical ring is
      // confirmed is not decided).
      const deferred = await fresh();
      await session.run(
        forkInputs(1, 0),
        {},
        {
          context: deferred,
          queryLength: 1,
          commit: "deferred",
        },
      );
      await assertRejects(() => deferred.fork(), ExecutionError, "awaiting commit");
      deferred.commit(1);
      contexts.push(await deferred.fork());

      // A rejected fork allocated nothing (the cumulative count is only the 4 made + the 3 forks
      // that passed).
      assertEquals(session.diagnostics().stateBacking.contextCount, 7);

      // (h) When the Session's dispose spans the allocation await: what was allocated is returned
      // and is put neither in the live set nor in the cumulative count (the original remains).
      const target = await fresh();
      const before = session.diagnostics().stateBacking;
      const pending = target.fork();
      await session.dispose();
      const crossed = await assertRejects(() => pending, ExecutionError);
      // The message carries the fork context (to tell it apart from the create message).
      assert(
        crossed.message.includes("fork: cannot fork a GenerationContext on a disposed Session"),
        crossed.message,
      );
      assertEquals(
        session.diagnostics().stateBacking,
        before,
        "the allocated copy stayed in the live set",
      );
      // (h) A Session already disposed at the time of the call.
      await assertRejects(
        () => target.fork(),
        ExecutionError,
        "fork: cannot fork a GenerationContext on a disposed Session",
      );
      assertEquals(target.pastLength, 0, "a rejected fork does not pollute the original");
    } finally {
      for (const context of contexts) await context.dispose();
      await session.dispose();
      sink.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "fork after a device loss is a GpuDeviceLostError (real GPU)",
  async () => {
    // This case breaks the device, so it takes a dedicated GpuContext.
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const context = await session.createGenerationContext({ chunkLength: 4 });
    gpu.destroy();
    // Rejected right after destroy (before lost reacts) — an allocation on a lost device only
    // returns invalid buffers, so letting it through would make a copy with an empty KV.
    await assertRejects(() => context.fork(), GpuDeviceLostError, "device was lost");
    await session.dispose().catch(() => undefined);
  },
);

/**
 * A device for fork fault injection (a synchronous throw at the N-th allocation and a reject of pop
 * are set up **later**).
 *
 * Unlike {@link injectFaults}, the returned buffers are **plain GPUBuffers** (only `destroy` is
 * wrapped, as an own property). fork passes the allocated buffers to `copyBufferToBuffer`, so this
 * avoids handing a Proxy to the WebGPU implementation. The traps are set with `arm` after the
 * original context is made (so the original's allocation is not caught up in them).
 */
const armableFaults = (
  gpu: GpuContext,
): {
  readonly gpu: GpuContext;
  readonly destroyCounts: number[];
  arm(fault: Omit<Fault, "hangPop">): void;
} => {
  const destroyCounts: number[] = [];
  let creations = 0;
  let fault: Omit<Fault, "hangPop"> = {};
  const device = new Proxy(gpu.device, {
    get(target, prop): unknown {
      if (prop === "createBuffer") {
        return (descriptor: GPUBufferDescriptor): GPUBuffer => {
          creations += 1;
          if (creations === fault.failCreateAt) {
            throw new Error(`injected: createBuffer #${creations} throws synchronously`);
          }
          const buffer = target.createBuffer(descriptor);
          const index = destroyCounts.push(0) - 1;
          const destroy = buffer.destroy.bind(buffer);
          Object.defineProperty(buffer, "destroy", {
            value: (): void => {
              destroyCounts[index] += 1;
              destroy();
            },
          });
          return buffer;
        };
      }
      if (prop === "popErrorScope" && fault.popError !== undefined) {
        const error = fault.popError;
        return (): Promise<GPUError | null> => {
          // MUST: always issue the real pop (the same reason as {@link injectFaults} — do not break
          // the LIFO).
          void target.popErrorScope().catch(() => null);
          return Promise.reject(error);
        };
      }
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  const proxied = new Proxy(gpu, {
    get(target, prop): unknown {
      if (prop === "device") return device;
      const value: unknown = Reflect.get(target, prop);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
  return {
    gpu: proxied,
    destroyCounts,
    arm: (next): void => {
      fault = next;
      creations = 0;
    },
  };
};

test.skipIf(!GPU_AVAILABLE)(
  "a failure part way through a fork's allocation returns each allocated buffer exactly once and leaves the original intact (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    const injected = armableFaults(gpu);
    try {
      const source = await GenerationContext.create(injectedHost(injected.gpu, graph), {
        chunkLength: 1,
      });
      internals(source).advance(0, 1);
      assertEquals(
        injected.destroyCounts,
        [0, 0, 0],
        "the original is 2 slots + the logical-length uniform",
      );

      // ① A synchronous throw part way through: failing at the copy's second returns its first
      //   exactly once.
      injected.arm({ failCreateAt: 2 });
      await assertRejects(() => source.fork(), Error, "injected: createBuffer #2");
      assertEquals(injected.destroyCounts, [0, 0, 0, 1], "the 1 allocated buffer does not leak");

      // ② A reject of pop: everything the copy allocated (2 slots + the logical-length uniform) is
      //   returned exactly once.
      injected.arm({ popError: new Error("injected: popErrorScope rejects") });
      await assertRejects(() => source.fork(), Error, "injected: popErrorScope rejects");
      assertEquals(injected.destroyCounts, [0, 0, 0, 1, 1, 1, 1]);

      // The original is intact: its logical length can be read, and with the traps removed fork
      // passes too.
      injected.arm({});
      assertEquals(source.pastLength, 1);
      const forked = await source.fork();
      assertEquals(forked.pastLength, 1);
      await forked.dispose();
      await source.dispose();
      assertEquals(injected.destroyCounts, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the branch-independence check is not vacuous — when the copy's dispatch binds the original's slots, the original moves (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const session = await forkSession(gpu);
    const source = await session.createGenerationContext({ chunkLength: 4 });
    let forked: GenerationContext | undefined;
    try {
      // Advance until the original holds bundles baked for the decode-shape backing.
      await forkStep(session, source, 4, 0);
      await forkStep(session, source, 1, 1);
      await forkStep(session, source, 1, 2);
      forked = await source.fork();
      const sourceWords = await readSlotWords(gpu, source);
      const forkedWords = await readSlotWords(gpu, forked);

      // Injection: copy the bundles the original baked to the copy (reproduces an implementation
      // that copies `#baked`; an implementation that shares buffers gives the same observation in
      // that "the copy's dispatch binds the original's slots"). Backing generations increase
      // monotonically from 1 (`planBacking.buildCount` is the maximum).
      let injected = 0;
      for (let build = 1; build <= session.diagnostics().planBacking.buildCount; build += 1) {
        const groups = internals(source).bakedGroups(build);
        if (groups === undefined) continue;
        internals(forked).setBakedGroups(build, groups);
        injected += 1;
      }
      assert(injected > 0, "no bundles to copy (the fault injection would miss)");
      const rebinds = session.diagnostics().stateBacking.rebindCount;
      await forkStep(session, forked, 1, 9);

      // All 3 observations of the independence check turn red: the original's bytes move, the
      // copy's bytes do not, and the copy's first run does not re-bake.
      assertNotEquals(
        await readSlotWords(gpu, source),
        sourceWords,
        "the original's bytes do not move even with the injection (the independence check would miss)",
      );
      assertEquals(await readSlotWords(gpu, forked), forkedWords);
      assertEquals(session.diagnostics().stateBacking.rebindCount, rebinds);
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

/**
 * A host that inserts an await into `adopt` (reproduces a future change that breaks the contract of
 * the synchronous section — fault injection).
 * `allocate` is called after the wait, so the copy takes the state after the writes queued during
 * the wait.
 */
const delayedAdoptHost = (
  gpu: GpuContext,
  graph: IrGraph,
  delay: () => Promise<void>,
): GenerationContextHost => ({
  ...injectedHost(gpu, graph),
  adopt: async (allocate) => {
    await delay();
    return await allocate();
  },
});

test.skipIf(!GPU_AVAILABLE)(
  "the synchronous-section check is not vacuous — with a host that awaits before the copy, the later writes are copied (fault injection, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const graph = prepareContainer(
      await openStateModel(stateGraph(NUMERIC_STATES)),
      GRAPH_NAME,
    ).graph;
    const earlier = new Uint32Array(SLOT_BYTES / 4).fill(0x3f800000);
    const later = new Uint32Array(SLOT_BYTES / 4).fill(0x40000000);
    /**
     * Fills the original's `k` with `earlier`, overwrites it with `later` in **the same synchronous
     * section** that calls `fork()`, then reads the copy's `k` (the queue order is the issue order
     * of writeBuffer and submit).
     */
    const forkThenOverwrite = async (host: GenerationContextHost): Promise<number[]> => {
      const source = await GenerationContext.create(host, { chunkLength: 1 });
      try {
        const slot = internals(source).slots.get("k");
        assert(slot !== undefined, "slot k cannot be looked up");
        gpu.device.queue.writeBuffer(slot.buffer, 0, earlier);
        const pending = source.fork();
        gpu.device.queue.writeBuffer(slot.buffer, 0, later);
        const forked = await pending;
        try {
          const copied = internals(forked).slots.get("k");
          assert(copied !== undefined, "the copy's slot k cannot be looked up");
          return await readWords(gpu, copied.buffer, copied.byteLength);
        } finally {
          await forked.dispose();
        }
      } finally {
        await source.dispose();
      }
    };
    try {
      assertEquals(
        await forkThenOverwrite(injectedHost(gpu, graph)),
        [...earlier],
        "a host that keeps the contract: the copy has the contents as of the fork() call",
      );
      // With a wait inserted, later writes in the same synchronous section get into the copy (= the
      // check above turns red). Even one microtask is caught here (the writes are inside the
      // synchronous section). Writes through a run have their body queued after a microtask, so
      // there is no guarantee that those catch a microtask shift (speculation).
      for (const [label, delay] of [
        ["microtask", () => Promise.resolve()],
        ["macrotask", () => new Promise<void>((resolve) => setTimeout(resolve, 0))],
      ] as const) {
        assertEquals(
          await forkThenOverwrite(delayedAdoptHost(gpu, graph, delay)),
          [...later],
          `${label}: the contents as of the call remain even with the await (the check would miss)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "a real Session's fork also copies in the synchronous section — later writes in the same section do not get into the copy (real GPU)",
  async () => {
    // The check above builds its own host, so it does not catch a regression that adds an await on
    // the side of `Session.#adopt` (decision → calling `allocate` synchronously). This pins the
    // same form through a real Session's registration path. The regression is reproduced by
    // inserting one `await Promise.resolve()` before `allocate()` in `#adopt` (`allocate` then runs
    // after the later writeBuffer, so the copy reads `later`).
    const gpu = await acquireGpu();
    const session = await stateSession(gpu);
    const earlier = new Uint32Array(SLOT_BYTES / 4).fill(0x3f800000);
    const later = new Uint32Array(SLOT_BYTES / 4).fill(0x40000000);
    const source = await session.createGenerationContext({ chunkLength: 1 });
    let forked: GenerationContext | undefined;
    try {
      const slot = internals(source).slots.get("k");
      assert(slot !== undefined, "slot k cannot be looked up");
      gpu.device.queue.writeBuffer(slot.buffer, 0, earlier);
      const pending = source.fork();
      gpu.device.queue.writeBuffer(slot.buffer, 0, later);
      forked = await pending;
      const copied = internals(forked).slots.get("k");
      assert(copied !== undefined, "the copy's slot k cannot be looked up");
      assertEquals(
        await readWords(gpu, copied.buffer, copied.byteLength),
        [...earlier],
        "the copy picked up writes made after the fork() call (an await got into the registration path)",
      );
    } finally {
      await forked?.dispose();
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "records how long a fork takes (2 slots × 64 MiB, recording only, real GPU)",
  async () => {
    // The copy is its own submit outside the scheduler, so there is no other observation point for
    // the duration of a huge copy. No value check is set (it moves with the device and the load) —
    // it remains as material for reading whether a weak GPU gets close to the TDR range.
    // MUST: wait for the queue to finish at the end of the interval. The resolution of the
    // popErrorScope that fork waits for does not wait for the copy to finish executing (observed
    // 0.1 ms for 128 MiB), so that alone cannot measure the copy's duration.
    const rows = (64 * 1024 * 1024) / (2 * 4 * 4);
    const gpu = await acquireGpu();
    const session = await stateSession(
      gpu,
      stateGraph({
        k: { dtype: "f32", shape: [1, 2, rows, 4] },
        v: { dtype: "f32", shape: [1, 2, rows, 4] },
      }),
    );
    const source = await session.createGenerationContext({ chunkLength: 1 });
    try {
      // Move the earlier allocation and initialization submits out of the interval. NOTE: the
      // original's slots were never written, so the implementation's lazy zero initialization may
      // be charged to the first use (= this copy) (speculation) — the value may come out larger
      // than the copy alone.
      await gpu.device.queue.onSubmittedWorkDone();
      const started = performance.now();
      const forked = await source.fork();
      await gpu.device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - started;
      try {
        assertEquals(internals(forked).bytes, internals(source).bytes);
        console.log(
          `[gpu] GenerationContext.fork: ${(internals(source).bytes / (1024 * 1024)).toFixed(
            0,
          )} MiB in ${elapsed.toFixed(1)} ms (allocation + until the copy finishes executing)`,
        );
      } finally {
        await forked.dispose();
      }
    } finally {
      await source.dispose();
      await session.dispose();
      gpu.destroy();
    }
  },
);
