// The weight supply uploads one block at a time and reads at most 2 blocks ahead of it, across part
// boundaries (lazy `WeightBatch.items` — ADR 0108 decision 9, addendum 14).
//
// Session construction (`createSessionFromContainer` → `containerBatches` → `buildSessionState`)
// does not read a whole part before uploading it: each block is written as soon as it is taken,
// while the next 2 blocks — also those of the next parts — are read meanwhile, and the fence (an
// empty submit + waiting for completion) comes once per part. The bound is what keeps the weight
// bytes alive on the JS side at one item + 2 blocks instead of one part (how many bytes GC actually
// frees cannot be observed in a unit test, so the order is pinned here — measured with
// tools/ram-peak).
//
// Also pinned:
// - `SessionBuildStats.supplyWaitMs` includes the wait for block reads and the wait until the end
//   of the sequence is known (the supply's cost does not vanish from the stats)
// - when a read fails mid-part, the arena is destroyed with the uploaded weights, and no read is
//   issued beyond the read-ahead bound (the transaction boundary — the "first half of a part
//   uploaded" state that only lazy items create)
// - a read failing ahead of its turn is reported when its block is reached, after the earlier
//   blocks' failures; a failed or abandoned build settles the reads in flight before it returns
//
// No real GPU is used. The order and the counts can be pinned deterministically with a fake device
// that records allocations, writes and fences. That writeBuffer "copies at the time of the call"
// can only be pinned on real hardware, so tests/gpu_write_buffer_copy_test.ts holds that.

import {
  assert,
  assertEquals,
  assertInstanceOf,
  assertRejects,
  assertStrictEquals,
} from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { GpuOutOfMemoryError } from "../src/gpu/error-scope.ts";
import { containerBatches, createSessionFromContainer, Session } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { ReadyInitializer } from "../src/runtime/session-build.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import { f32Bytes, GRAPH_NAME } from "./helpers/model-fixture.ts";

// ---------------------------------------------------------------------------
// Fixtures (a model with 3 parts, a recording fake device)
// ---------------------------------------------------------------------------

/**
 * The smallest graph that splits into 3 parts: y = x·w + b + c.
 *
 * The in-memory container puts whole supplies (`b` / `c`) into the same part and each piece into a
 * part of its own (the part split of `openMemoryContainer`), so the parts are `{b, c}` → `w` piece
 * 1 → `w` piece 2. The first part has 2 blocks, so the "interleaving within a part" is visible too.
 */
const declaration = () =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, 4] }],
    outputs: ["y"],
    initializers: { w: {}, b: {}, c: {} },
    values: {
      w: { dtype: "f32", shape: [4, 3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      h: { dtype: "f32", shape: [2, 3] },
      t: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "b"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "c"], outs: ["y"], attrs: {} },
    ],
  });

/** The byte length of one row of `w` (f32 × 3 columns). */
const W_ROW_BYTES = 3 * 4;

const container = (): BoundContainer => {
  const w = f32Bytes(Array.from({ length: 12 }, (_, i) => i * 0.25));
  return openMemoryContainer({
    graphs: { [GRAPH_NAME]: declaration() },
    tensors: {
      [GRAPH_NAME]: {
        w: {
          encoding: { codec: "f32" },
          // Each read of a piece returns a different buffer (a copy) — so the write record can tell
          // blocks apart by buffer.
          pieces: [
            { rows: [0, 2], read: () => Promise.resolve(w.slice(0, 2 * W_ROW_BYTES)) },
            {
              rows: [2, 4],
              read: () => Promise.resolve(w.slice(2 * W_ROW_BYTES, 4 * W_ROW_BYTES)),
            },
          ],
        },
        b: { encoding: { codec: "f32" }, bytes: f32Bytes([1, 2, 3]) },
        c: { encoding: { codec: "f32" }, bytes: f32Bytes([4, 5, 6]) },
      },
    },
  });
};

/**
 * The limits only need values that pass the checks (what matters is the order, not the allocation
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

/** A step inserted before each source read (the injection point for delays and failures). */
type BeforeRead = (id: string) => Promise<void>;

/**
 * A recording fake device, and a wrapper around the source.
 *
 * The record is one chronological sequence `events`: `read <block>` (a source read starts) /
 * `write <block>` (`queue.writeBuffer` — the passed buffer is looked up to find which block's read
 * returned it) / `push` / `pop` (push / pop of one errorScope) / `submit` (an empty submit) /
 * `fence` (waiting for `onSubmittedWorkDone` to complete).
 *
 * The whole DOM type is not reproduced, so it is passed with a cast (the same test-only boundary as
 * tests/helpers/fake-gpu.ts). GpuContext itself is the real one (the device-loss subscription and
 * the errorScope paths go through the production implementation).
 */
const recorder = (
  // What the n-th `popErrorScope` (0-based, over the whole build) resolves with — the injection
  // point of a scope failure (default: every pop resolves with null).
  popResult: (pop: number) => Promise<GPUError | null> = () => Promise.resolve(null),
) => {
  const events: string[] = [];
  let pops = 0;
  const blockOf = new Map<ArrayBufferLike, string>();
  const counts = { created: 0, destroyed: 0, openScopes: 0 };
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    limits: { minStorageBufferOffsetAlignment: 256 },
    pushErrorScope: (): void => {
      events.push("push");
      counts.openScopes += 1;
    },
    popErrorScope: (): Promise<GPUError | null> => {
      events.push("pop");
      counts.openScopes -= 1;
      const pop = pops;
      pops += 1;
      return popResult(pop);
    },
    createBuffer: (descriptor: GPUBufferDescriptor): GPUBuffer => {
      counts.created += 1;
      return {
        size: descriptor.size,
        destroy: (): void => {
          counts.destroyed += 1;
        },
      } as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer: (_buffer: GPUBuffer, _offset: number, data: ArrayBufferView): void => {
        events.push(`write ${blockOf.get(data.buffer) ?? "(a buffer no read returned)"}`);
      },
      submit: (): void => {
        events.push("submit");
      },
      onSubmittedWorkDone: (): Promise<void> => {
        events.push("fence");
        return Promise.resolve();
      },
    },
  } as unknown as GPUDevice;
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const trace = (inner: BoundContainer, before?: BeforeRead): BoundContainer => ({
    graphs: inner.graphs,
    readBlock: async (id) => {
      events.push(`read ${id}`);
      await before?.(id);
      const bytes = await inner.readBlock(id);
      blockOf.set(bytes.buffer, id);
      return bytes;
    },
  });
  return { events, counts, gpu, trace };
};

// Node's timers run on libuv's millisecond loop time and can fire up to about 1 ms before
// `performance.now()` says the delay has elapsed; the timing assertions below compare against that
// clock, so sleep until the deadline is reached on it.
const sleep = async (ms: number): Promise<void> => {
  const until = performance.now() + ms;
  for (let now = performance.now(); now < until; now = performance.now()) {
    await new Promise((resolve) => setTimeout(resolve, Math.ceil(until - now)));
  }
};

// ---------------------------------------------------------------------------

describe("the weight supply uploads one block at a time", () => {
  it("writes each block as it is taken with the reads 2 blocks ahead, an errorScope pair per block and one fence per part", async () => {
    const { events, gpu, trace } = recorder();
    const session = await createSessionFromContainer(gpu, trace(container()), GRAPH_NAME);
    // The flush on dispose also waits for completion once, so copy the sequence as it stands when
    // construction finishes.
    const built = [...events];
    const { partCount } = session.diagnostics().buildStats;
    await session.dispose();

    // One errorScope pair per block (item) (push the 2 scopes out-of-memory + validation → write →
    // pop the 2). Coarsening it to per part would drop the attribution of an allocation failure
    // from the block to the part. None is set around the empty submit (it involves no allocation
    // and no validation — the NOTE on the fence in session-build.ts).
    assertEquals(built, [
      // The first part (whole supplies b / c): uploaded one at a time without reading the part
      // through. Taking b reads b and the next 2 blocks, c and piece 1 of the next part.
      "read model/b",
      "read model/c",
      "read model/w#piece1",
      "push",
      "push",
      "write model/b",
      "pop",
      "pop",
      // Taking c reads 2 ahead of it: piece 2, two parts on.
      "read model/w#piece2",
      "push",
      "push",
      "write model/c",
      "pop",
      "pop",
      "submit",
      "fence",
      // The next parts' blocks were read before the fence (the read-ahead crosses part boundaries).
      "push",
      "push",
      "write model/w#piece1",
      "pop",
      "pop",
      "submit",
      "fence",
      "push",
      "push",
      "write model/w#piece2",
      "pop",
      "pop",
      "submit",
      "fence",
    ]);
    // The exact match of the event sequence above pins the number of fences. What this counts is
    // the number of batches consumed.
    assertEquals(partCount, 3, "the number of batches consumed = the number of parts");
  });

  it("supplyWaitMs includes the wait for block reads (a slow source)", async () => {
    const DELAY_MS = 20;
    const { gpu, trace } = recorder();
    let readMs = 0;
    let reads = 0;
    const slow = trace(container(), async () => {
      const start = performance.now();
      await sleep(DELAY_MS);
      readMs += performance.now() - start;
      reads += 1;
    });
    const session = await createSessionFromContainer(gpu, slow, GRAPH_NAME);
    const { supplyWaitMs } = session.diagnostics().buildStats;
    await session.dispose();

    assertEquals(reads, 4, "reads each of the 4 blocks (b / c / the 2 pieces of w) once");
    assert(
      readMs >= DELAY_MS,
      `the read delay did not take effect (${readMs} ms — the detector would miss)`,
    );
    // The lower bound on supplyWaitMs is one read's delay: the read-ahead keeps up to 3 reads in
    // flight at once, so their delays overlap with one another and with the uploads, and the wait
    // can be shorter than their sum. The first read cannot be hidden behind anything (nothing has
    // been uploaded yet), and its whole interval lies inside the first iteration wait, measured on
    // the same clock.
    const SUM_ORDER_EPSILON_MS = 1e-6;
    assert(
      supplyWaitMs >= DELAY_MS - SUM_ORDER_EPSILON_MS,
      `supplyWaitMs ${supplyWaitMs} ms does not include the first read's wait of ${DELAY_MS} ms`,
    );
  });

  it("supplyWaitMs also includes the wait until the end of the sequence is known (a sequence that is late after its last item)", async () => {
    // On the container path (`containerBatches`) the end of the sequence involves no read, so a
    // delay in the source reads never reaches it. So a sequence that is late before announcing its
    // end is passed directly to the construction entry (`Session.build`) — with 0 items, the only
    // wait measured is the wait for the end of the sequence.
    const DELAY_MS = 20;
    const { gpu } = recorder();
    let tailMs = 0;
    const lateEnd: AsyncIterable<ReadyInitializer> = {
      [Symbol.asyncIterator]: () => ({
        next: async (): Promise<IteratorResult<ReadyInitializer>> => {
          const start = performance.now();
          await sleep(DELAY_MS);
          tailMs += performance.now() - start;
          return { done: true, value: undefined };
        },
      }),
    };
    const emptyGraph = mergeGraph({
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
    async function* oneBatch() {
      yield { origin: "part 1", items: lateEnd };
    }
    const session = await Session.build(gpu, emptyGraph, new Map(), oneBatch(), {});
    const { supplyWaitMs, partCount } = session.diagnostics().buildStats;
    await session.dispose();

    assertEquals(partCount, 1);
    assert(
      tailMs >= DELAY_MS,
      `the end-of-sequence delay did not take effect (${tailMs} ms — the detector would miss)`,
    );
    const SUM_ORDER_EPSILON_MS = 1e-6;
    assert(
      supplyWaitMs >= tailMs - SUM_ORDER_EPSILON_MS,
      `supplyWaitMs ${supplyWaitMs} ms does not include the end-of-sequence wait of ${tailMs} ms`,
    );
  });

  it("when a read fails mid-part, destroys the arena with the uploaded weights and reads nothing beyond the read-ahead bound", async () => {
    const { events, counts, gpu, trace } = recorder();
    const injected = new Error("injected: block fetch failed");
    // Fail at the second block of the first part (after b is uploaded, before the fence).
    const failing = trace(container(), (id) =>
      id === "model/c" ? Promise.reject(injected) : Promise.resolve(),
    );
    const error = await assertRejects(() => createSessionFromContainer(gpu, failing, GRAPH_NAME));

    assertStrictEquals(error, injected, "a read failure passes through as is, without re-wrapping");
    assertEquals(
      events.filter((event) => event.startsWith("read ") || event.startsWith("write ")),
      // Taking c (the failing block) read 2 ahead of it, up to piece 2 — the last block here. The
      // bound itself is pinned on a longer supply below.
      [
        "read model/b",
        "read model/c",
        "read model/w#piece1",
        "write model/b",
        "read model/w#piece2",
      ],
      "the reads around the failure differ (the supply was not closed at the bound)",
    );
    assertEquals(events.includes("submit"), false, "the fence of the failed part was set");
    assert(
      counts.created > 0,
      "no weights were uploaded (the state with the first half of a part uploaded was not produced)",
    );
    assertEquals(
      counts.destroyed,
      counts.created,
      "the buffers of the uploaded weights were not destroyed",
    );
    assertEquals(counts.openScopes, 0, "errorScopes were left open");
  });
});

// ---------------------------------------------------------------------------
// Read-ahead: up to 2 blocks beyond the item being uploaded, across part boundaries (ADR 0108
// addendum 14 — perf-ledger L-26)
// ---------------------------------------------------------------------------

/** Pieces of `w` in the wide model — one part each, so every piece read crosses a part boundary. */
const WIDE_PIECES = 6;
/** Rows of `w` per piece. */
const WIDE_PIECE_ROWS = 2;

/**
 * The same graph as {@link declaration} with a taller `w` (y = x·w + b + c, w: [12, 3]), split
 * into {@link WIDE_PIECES} pieces: parts `{b, c}` → `w` piece 1 → … → `w` piece 6 — 7 parts and 8
 * blocks, enough to see the read-ahead stop at its bound.
 */
const wideContainer = (): BoundContainer => {
  const rows = WIDE_PIECES * WIDE_PIECE_ROWS;
  const w = f32Bytes(Array.from({ length: rows * 3 }, (_, i) => i * 0.125));
  const graph = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, rows] }],
    outputs: ["y"],
    initializers: { w: {}, b: {}, c: {} },
    values: {
      w: { dtype: "f32", shape: [rows, 3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      h: { dtype: "f32", shape: [2, 3] },
      t: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "b"], outs: ["t"], attrs: {} },
      { op: "add", ins: ["t", "c"], outs: ["y"], attrs: {} },
    ],
  });
  return openMemoryContainer({
    graphs: { [GRAPH_NAME]: graph },
    tensors: {
      [GRAPH_NAME]: {
        w: {
          encoding: { codec: "f32" },
          // A fresh copy per read, so the recorder can tell the blocks apart by their buffer.
          pieces: Array.from({ length: WIDE_PIECES }, (_, piece) => {
            const start = piece * WIDE_PIECE_ROWS;
            const stop = start + WIDE_PIECE_ROWS;
            return {
              rows: [start, stop] as const,
              read: () => Promise.resolve(w.slice(start * W_ROW_BYTES, stop * W_ROW_BYTES)),
            };
          }),
        },
        b: { encoding: { codec: "f32" }, bytes: f32Bytes([1, 2, 3]) },
        c: { encoding: { codec: "f32" }, bytes: f32Bytes([4, 5, 6]) },
      },
    },
  });
};

/** The blocks of the wide model in supply order. */
const WIDE_BLOCKS = [
  "model/b",
  "model/c",
  ...Array.from({ length: WIDE_PIECES }, (_, piece) => `model/w#piece${piece + 1}`),
];

/** Rows of each int8 weight of the quantized model, and its pieces (2 rows = 32 B each). */
const Q_ROWS = 6;
const Q_ROW_LENGTH = 16;
const Q_PIECES = 3;

/**
 * Two int8 linears whose weights ride a scale on piece 1: y = linear(x, u, bu) + linear(x, v, bv),
 * u and v int8-sym `[6, 16]` in 3 pieces each. Parts: `{bu, bv}` → u piece 1 (+ its scale) → u
 * piece 2 → u piece 3 → v piece 1 (+ its scale) → …, so taking piece 1 takes two blocks.
 */
const quantizedContainer = (): BoundContainer => {
  const graph = parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, Q_ROW_LENGTH] }],
    outputs: ["y"],
    initializers: { u: {}, bu: {}, v: {}, bv: {} },
    values: {
      u: { dtype: "f32", shape: [Q_ROWS, Q_ROW_LENGTH] },
      bu: { dtype: "f32", shape: [Q_ROWS] },
      v: { dtype: "f32", shape: [Q_ROWS, Q_ROW_LENGTH] },
      bv: { dtype: "f32", shape: [Q_ROWS] },
      hu: { dtype: "f32", shape: [2, Q_ROWS] },
      hv: { dtype: "f32", shape: [2, Q_ROWS] },
      y: { dtype: "f32", shape: [2, Q_ROWS] },
    },
    nodes: [
      { op: "linear", ins: ["x", "u", "bu"], outs: ["hu"], attrs: {} },
      { op: "linear", ins: ["x", "v", "bv"], outs: ["hv"], attrs: {} },
      { op: "add", ins: ["hu", "hv"], outs: ["y"], attrs: {} },
    ],
  });
  const rowsPerPiece = Q_ROWS / Q_PIECES;
  const quantized = (seed: number) => {
    const payload = Uint8Array.from({ length: Q_ROWS * Q_ROW_LENGTH }, (_, i) => (i + seed) % 7);
    return {
      encoding: {
        codec: "int8-sym" as const,
        rowAxis: 0 as const,
        groupSize: Q_ROW_LENGTH,
        scale: f32Bytes(new Array(Q_ROWS).fill(0.5)),
      },
      pieces: Array.from({ length: Q_PIECES }, (_, piece) => {
        const start = piece * rowsPerPiece;
        const stop = start + rowsPerPiece;
        return {
          rows: [start, stop] as const,
          read: () => Promise.resolve(payload.slice(start * Q_ROW_LENGTH, stop * Q_ROW_LENGTH)),
        };
      }),
    };
  };
  return openMemoryContainer({
    graphs: { [GRAPH_NAME]: graph },
    tensors: {
      [GRAPH_NAME]: {
        u: quantized(0),
        bu: { encoding: { codec: "f32" }, bytes: f32Bytes(new Array(Q_ROWS).fill(0.25)) },
        v: quantized(3),
        bv: { encoding: { codec: "f32" }, bytes: f32Bytes(new Array(Q_ROWS).fill(0.75)) },
      },
    },
  });
};

/** The items of the quantized model in supply order, with the blocks each one takes. */
const QUANTIZED_ITEMS: readonly (readonly string[])[] = [
  ["model/bu"],
  ["model/bv"],
  ...["u", "v"].flatMap((name) =>
    Array.from({ length: Q_PIECES }, (_, piece) =>
      piece === 0
        ? [`model/${name}#piece1`, `model/${name}#scale`]
        : [`model/${name}#piece${piece + 1}`],
    ),
  ),
];

/** Counts the reads of `inner` that have settled (fulfilled or rejected). */
const countSettled = (inner: BoundContainer) => {
  const counts = { started: 0, settled: 0 };
  const source: BoundContainer = {
    graphs: inner.graphs,
    readBlock: (id) => {
      counts.started += 1;
      const read = inner.readBlock(id);
      read.then(
        () => (counts.settled += 1),
        () => (counts.settled += 1),
      );
      return read;
    },
  };
  return { counts, source };
};

const onlyReadsAndWrites = (events: readonly string[]): string[] =>
  events.filter((event) => event.startsWith("read ") || event.startsWith("write "));

describe("Session build reads up to 2 blocks ahead of the item it uploads, across part boundaries", () => {
  it("starts reading the next parts' blocks before the current part's fence", async () => {
    const { events, gpu, trace } = recorder();
    const session = await createSessionFromContainer(gpu, trace(wideContainer()), GRAPH_NAME);
    const built = [...events];
    await session.dispose();

    const firstSubmit = built.indexOf("submit");
    assert(firstSubmit >= 0, "no part was fenced");
    // Part `{b, c}`: taking b reads b, c and piece 1; taking c reads piece 2 — both in later parts.
    assertEquals(
      built.slice(0, firstSubmit).filter((event) => event.startsWith("read ")),
      ["read model/b", "read model/c", "read model/w#piece1", "read model/w#piece2"],
    );
    // Every block is still read once and written once, in supply order.
    assertEquals(
      built.filter((event) => event.startsWith("read ")),
      WIDE_BLOCKS.map((id) => `read ${id}`),
    );
    assertEquals(
      built.filter((event) => event.startsWith("write ")),
      WIDE_BLOCKS.map((id) => `write ${id}`),
    );
  });

  it("never has more than 2 blocks read or in flight beyond the block being uploaded", async () => {
    const { events, gpu, trace } = recorder();
    let inFlight = 0;
    let maxInFlight = 0;
    const slow = trace(wideContainer(), async () => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await sleep(2);
      inFlight -= 1;
    });
    const session = await createSessionFromContainer(gpu, slow, GRAPH_NAME);
    const built = [...events];
    await session.dispose();

    let reads = 0;
    let writes = 0;
    let maxAhead = 0;
    for (const event of built) {
      if (event.startsWith("read ")) reads += 1;
      if (!event.startsWith("write ")) continue;
      // At the upload of the block in supply position `writes`, the reads beyond it.
      writes += 1;
      maxAhead = Math.max(maxAhead, reads - writes);
    }
    assertEquals(maxAhead, 2, `reads ran more or less than 2 blocks ahead: ${built.join(", ")}`);
    assertEquals(maxInFlight, 3, "at most the block being taken + 2 ahead may be in flight");
  });

  it("counts the riding scale as a block: 4 blocks alive on piece 1, still at most 3 reads in flight", async () => {
    const { events, gpu, trace } = recorder();
    let inFlight = 0;
    let maxInFlight = 0;
    const slow = trace(quantizedContainer(), async () => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await sleep(2);
      inFlight -= 1;
    });
    const session = await createSessionFromContainer(gpu, slow, GRAPH_NAME);
    const built = [...events];
    await session.dispose();

    assertEquals(
      built.filter((event) => event.startsWith("read ")),
      QUANTIZED_ITEMS.flat().map((id) => `read ${id}`),
    );
    // At the start of each item's upload (its first scope push), the blocks read and not yet let go:
    // the item's own blocks + those read ahead (the earlier items were let go in their iteration).
    let reads = 0;
    let pushes = 0;
    let released = 0;
    const alive: number[] = [];
    for (const event of built) {
      if (event.startsWith("read ")) reads += 1;
      if (event !== "push") continue;
      pushes += 1;
      if (pushes % 2 === 0) continue;
      const item = (pushes - 1) / 2;
      alive.push(reads - released);
      released += QUANTIZED_ITEMS[item].length;
    }
    // bu, bv: 1 + 2; piece 1 with its scale: 2 + 2; other pieces: 1 + 2 (the last ones: fewer left).
    assertEquals(alive, [3, 3, 4, 3, 3, 4, 2, 1], `blocks alive per item: ${built.join(", ")}`);
    assertEquals(maxInFlight, 3, "at most the block being taken + 2 ahead may be in flight");
  });

  it("reports a failing read in the next part only when its block is reached, after the current part is fenced", async () => {
    const { events, counts, gpu, trace } = recorder();
    const injected = new Error("injected read failure on w piece 1");
    const failing = trace(wideContainer(), (id) =>
      id === "model/w#piece1" ? Promise.reject(injected) : Promise.resolve(),
    );
    const error = await assertRejects(() => createSessionFromContainer(gpu, failing, GRAPH_NAME));

    assertStrictEquals(error, injected, "the read failure is wrapped or replaced");
    const fence = events.indexOf("fence");
    assert(fence >= 0, "the part before the failing block was not fenced");
    assertEquals(
      events.slice(fence).filter((event) => event.startsWith("write ")),
      [],
      "a block after the failing one was uploaded",
    );
    assertEquals(
      events.filter((event) => event.startsWith("write ")),
      ["write model/b", "write model/c"],
    );
    // Taking piece 1 read up to piece 3; nothing further.
    assertEquals(
      events.filter((event) => event.startsWith("read ")),
      WIDE_BLOCKS.slice(0, 5).map((id) => `read ${id}`),
    );
    assertEquals(counts.destroyed, counts.created, "uploaded weights are not destroyed");
  });

  it("reports an earlier block's failing read even when a later block's read failed first", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      const { events, gpu, trace } = recorder();
      const earlier = new Error("injected late read failure on c");
      const later = new Error("injected early read failure on w piece 1");
      let laterFailedAt = Number.NaN;
      let earlierFailedAt = Number.NaN;
      const failing = trace(wideContainer(), async (id) => {
        if (id === "model/c") {
          await sleep(20);
          earlierFailedAt = performance.now();
          throw earlier;
        }
        if (id === "model/w#piece1") {
          laterFailedAt = performance.now();
          throw later;
        }
      });
      const error = await assertRejects(() => createSessionFromContainer(gpu, failing, GRAPH_NAME));

      assertStrictEquals(error, earlier, "the later block's failure was reported first");
      assert(laterFailedAt < earlierFailedAt, "the later read did not fail first");
      assertEquals(
        events.filter((event) => event.startsWith("write ")),
        ["write model/b"],
      );
      // Unhandled rejections are reported after the microtask queue drains; give it a few turns.
      await new Promise((resolve) => setTimeout(resolve, 50));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assertEquals(unhandled, [], "a read failure ahead of its turn went unhandled");
  });

  it("reports a part's scope failure over a slower read failure in a later part, once that read has settled", async () => {
    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);
    try {
      // Pops 0 / 1 are b's pair; pop 3 is c's out-of-memory scope (validation first, then OOM).
      const injectedOom = (pop: number): Promise<GPUError | null> =>
        pop === 3
          ? sleep(5).then(() => ({ message: "injected OOM on c" }) as GPUError)
          : Promise.resolve(null);
      const { events, counts: device, gpu, trace } = recorder(injectedOom);
      // Piece 2 (two parts on) is read ahead when c is taken, and fails long after c's scope.
      const slowFailure = new Error("injected slow read failure on w piece 2");
      const traced = trace(wideContainer(), async (id) => {
        if (id !== "model/w#piece2") return;
        await sleep(30);
        throw slowFailure;
      });
      const { counts, source } = countSettled(traced);
      const error = await assertRejects(() => createSessionFromContainer(gpu, source, GRAPH_NAME));
      const settledAtRejection = counts.settled;

      assertInstanceOf(error, GpuOutOfMemoryError);
      // The memory container stacks whole supplies from part 2 (`WHOLE_PART` in memory.ts).
      assertEquals(error.message, "part 2 weight upload for initializer 'c': injected OOM on c");
      assert(events.includes("read model/w#piece2"), "the later part's read was not in flight");
      assertEquals(events.includes("submit"), false, "the failed part's fence was issued");
      const stillRunning = "the build rejected with reads still running";
      assertEquals(settledAtRejection, counts.started, stillRunning);
      assertEquals(device.destroyed, device.created, "uploaded weights are not destroyed");
      assertEquals(device.openScopes, 0, "error scopes are left on the stack");
      // Unhandled rejections are reported after the microtask queue drains; give it a few turns.
      await new Promise((resolve) => setTimeout(resolve, 50));
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
    assertEquals(unhandled, [], "the later part's read failure went unhandled");
  });

  it("stops reading when the build fails, and rejects only after the reads in flight have settled", async () => {
    const { events, counts: device, gpu, trace } = recorder();
    const injected = new Error("injected read failure on c");
    const traced = trace(wideContainer(), async (id) => {
      if (id === "model/c") throw injected;
      // The reads ahead of the failure are still running when it is reported.
      if (id !== "model/b") await sleep(30);
    });
    const { counts, source } = countSettled(traced);
    const error = await assertRejects(() => createSessionFromContainer(gpu, source, GRAPH_NAME));
    const settledAtRejection = counts.settled;

    assertStrictEquals(error, injected);
    // Taking c read up to piece 2; no read was issued after the failure.
    assertEquals(
      onlyReadsAndWrites(events).filter((event) => event.startsWith("read ")),
      WIDE_BLOCKS.slice(0, 4).map((id) => `read ${id}`),
    );
    assertEquals(counts.started, 4);
    assertEquals(settledAtRejection, counts.started, "the build rejected with reads still running");
    assertEquals(device.destroyed, device.created, "uploaded weights are not destroyed");
    assertEquals(device.openScopes, 0, "error scopes are left on the stack");
  });

  it("stops reading when the consumer stops early, and returns only after the reads in flight have settled", async () => {
    const inner = wideContainer();
    const { counts, source } = countSettled({
      graphs: inner.graphs,
      readBlock: async (id) => {
        await sleep(10);
        return await inner.readBlock(id);
      },
    });
    const batches = containerBatches(source, GRAPH_NAME);
    for await (const batch of batches) {
      for await (const _item of batch.items) break;
      break;
    }

    // Taking b read b, c and piece 1; breaking out issues nothing more and waits for c and piece 1.
    assertEquals(counts.started, 3);
    assertEquals(counts.settled, counts.started, "the iterator returned with reads still running");
  });
});

describe("the read-ahead after the supply is closed", () => {
  it("fails a pull loudly instead of issuing a read", async () => {
    const { counts, source } = countSettled(wideContainer());
    const batches = containerBatches(source, GRAPH_NAME);
    const first = await batches.next();
    assert(first.done !== true, "the supply has no part");
    await batches.return(undefined);

    await assertRejects(
      () => first.value.items[Symbol.asyncIterator]().next(),
      ExecutionError,
      "taken after the supply was closed",
    );
    assertEquals(counts.started, 0, "a read was issued after the supply was closed");
  });
});
