// Session build issues each initializer's error-scope pair inside that item's synchronous interval
// but awaits the outcomes at the part fence, in item order (ADR 0108 addendum 13 — perf-ledger
// H-38).
//
// Pinned here: the deferral itself (every item's pops are issued before the first item's scopes
// settle), the early stop (once a failure has settled — a scope failure or a rejected pop — the rest
// of the part is not uploaded), the wait for the scopes at the part's end counted in
// `uploadFenceMs`, and the attribution contract the per-item await used to give for free:
// - the error reported is the first failing item in item order, with the same type and message the
//   per-item await reported — even when a later item's scope settles first (also when that early
//   settle is what stops the part), and even when a later item fails on the host side or its block
//   read fails;
// - a failure that settles after the last item still fails the build, before the part's fence;
// - several failing items leave no unhandled rejection behind;
// - a part whose items all succeed writes the same bytes into the same buffers.
//
// No real GPU: the order in which scopes settle cannot be chosen on hardware, so a fake device
// injects per-item scope failures and settle delays (the GpuContext itself is the real one, as in
// runtime_weight_supply_test.ts).

import { assert, assertEquals, assertInstanceOf, assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { GpuOutOfMemoryError, GpuValidationError } from "../src/gpu/error-scope.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { f32Bytes, GRAPH_NAME } from "./helpers/model-fixture.ts";

// ---------------------------------------------------------------------------
// Material: one part of four whole initializers, and a fake device with per-item faults
// ---------------------------------------------------------------------------

/** y = x + a + b + c + d — four whole initializers, which the memory container puts in one part. */
const declaration = () =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [3] }],
    outputs: ["y"],
    initializers: { a: {}, b: {}, c: {}, d: {} },
    values: {
      a: { dtype: "f32", shape: [3] },
      b: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      d: { dtype: "f32", shape: [3] },
      t1: { dtype: "f32", shape: [3] },
      t2: { dtype: "f32", shape: [3] },
      t3: { dtype: "f32", shape: [3] },
      y: { dtype: "f32", shape: [3] },
    },
    nodes: [
      { op: "add", ins: ["x", "a"], outs: ["t1"], attrs: {} },
      { op: "add", ins: ["t1", "b"], outs: ["t2"], attrs: {} },
      { op: "add", ins: ["t2", "c"], outs: ["t3"], attrs: {} },
      { op: "add", ins: ["t3", "d"], outs: ["y"], attrs: {} },
    ],
  });

const ITEMS = ["a", "b", "c", "d"] as const;
type Item = (typeof ITEMS)[number];

const BYTES: Readonly<Record<Item, Uint8Array<ArrayBuffer>>> = {
  a: f32Bytes([1, 2, 3]),
  b: f32Bytes([4, 5, 6]),
  c: f32Bytes([7, 8, 9]),
  d: f32Bytes([10, 11, 12]),
};

const container = (): BoundContainer =>
  openMemoryContainer({
    graphs: { [GRAPH_NAME]: declaration() },
    tensors: {
      [GRAPH_NAME]: Object.fromEntries(
        ITEMS.map((item) => [item, { encoding: { codec: "f32" }, bytes: BYTES[item] }]),
      ),
    },
  });

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

/** What one item's error-scope pair reports. */
type ScopeFault =
  | { readonly kind: "out-of-memory"; readonly message: string }
  | { readonly kind: "validation"; readonly message: string }
  /** The pop itself rejects (an implementation rejecting popErrorScope after a loss, for example). */
  | { readonly kind: "reject"; readonly error: Error };

type Faults = {
  /** Per-item scope failures. */
  readonly scopes?: Partial<Record<Item, ScopeFault>>;
  /** Per-item delay before the item's two pops settle (default 0 — settled in a macrotask). */
  readonly settleMs?: Partial<Record<Item, number>>;
  /** Items whose `createBuffer` throws synchronously (a host-side failure inside the interval). */
  readonly throwOnAllocate?: readonly Item[];
  /** Items whose block read rejects (a supply failure). */
  readonly rejectRead?: Partial<Record<Item, Error>>;
  /** Per-item delay before the item's block read resolves (default 0 — no timer). */
  readonly readDelayMs?: Partial<Record<Item, number>>;
};

const itemOf = (blockId: string): Item => {
  const name = blockId.replace(`${GRAPH_NAME}/`, "");
  if (!(ITEMS as readonly string[]).includes(name)) throw new Error(`unexpected block ${blockId}`);
  return name as Item;
};

/**
 * A fake device that attributes every call to the item being uploaded, counted from the scope pushes
 * (each item pushes its pair before its upload, in item order).
 *
 * Events: `read <item>`, `pop <item>` (one per issued pop), `settled <item>` (the item's pops have
 * settled), `submit`, `fence`. DOM types are not reproduced, so it is passed with a cast (the same
 * test-only boundary as tests/helpers/fake-gpu.ts).
 */
const faultyDevice = (faults: Faults = {}) => {
  const events: string[] = [];
  const writes: { buffer: number; offset: number; bytes: number[] }[] = [];
  const counts = { created: 0, destroyed: 0, openScopes: 0 };
  let current: Item | undefined;
  let pushes = 0;
  // The two pops of one item are issued back to back: validation first, then out-of-memory.
  let popsOfCurrent = 0;
  const bufferIds = new WeakMap<object, number>();
  const settleAfter = (item: Item): Promise<void> =>
    new Promise((resolve) => setTimeout(resolve, faults.settleMs?.[item] ?? 0));
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(),
    limits: { minStorageBufferOffsetAlignment: 256 },
    pushErrorScope: (): void => {
      counts.openScopes += 1;
      // The first push of a pair opens the next item's interval.
      if (pushes % 2 === 0) current = ITEMS[pushes / 2];
      pushes += 1;
      popsOfCurrent = 0;
    },
    popErrorScope: (): Promise<GPUError | null> => {
      counts.openScopes -= 1;
      const item = current;
      if (item === undefined) throw new Error("pop before any scope was pushed");
      events.push(`pop ${item}`);
      popsOfCurrent += 1;
      const scope = popsOfCurrent === 1 ? "validation" : "out-of-memory";
      const fault = faults.scopes?.[item];
      return settleAfter(item).then(() => {
        if (scope === "out-of-memory") events.push(`settled ${item}`);
        if (fault === undefined) return null;
        if (fault.kind === "reject") {
          if (scope === "validation") throw fault.error;
          return null;
        }
        return fault.kind === scope ? ({ message: fault.message } as GPUError) : null;
      });
    },
    createBuffer: (descriptor: GPUBufferDescriptor): GPUBuffer => {
      if (current !== undefined && faults.throwOnAllocate?.includes(current) === true) {
        throw new TypeError(`injected createBuffer failure on ${current}`);
      }
      counts.created += 1;
      const buffer = {
        size: descriptor.size,
        destroy: (): void => {
          counts.destroyed += 1;
        },
      };
      bufferIds.set(buffer, counts.created);
      return buffer as unknown as GPUBuffer;
    },
    queue: {
      writeBuffer: (buffer: GPUBuffer, offset: number, data: ArrayBufferView): void => {
        const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        writes.push({ buffer: bufferIds.get(buffer) ?? -1, offset, bytes: [...view] });
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
  const source = (inner: BoundContainer): BoundContainer => ({
    graphs: inner.graphs,
    readBlock: async (id) => {
      const item = itemOf(id);
      events.push(`read ${item}`);
      const delay = faults.readDelayMs?.[item];
      if (delay !== undefined) await new Promise((resolve) => setTimeout(resolve, delay));
      const rejected = faults.rejectRead?.[item];
      if (rejected !== undefined) throw rejected;
      return await inner.readBlock(id);
    },
  });
  return {
    events,
    writes,
    counts,
    build: () => createSessionFromContainer(gpu, source(container()), GRAPH_NAME),
  };
};

/**
 * The message the per-item await reported for a scope failure of `item` (`part 2`: the memory
 * container stacks whole supplies from part 2 — `WHOLE_PART` in format/container/memory.ts).
 */
const scopeMessage = (item: Item, message: string): string =>
  `part 2 weight upload for initializer '${item}': ${message}`;

// ---------------------------------------------------------------------------

describe("Session build awaits each initializer's error scopes at the part fence", () => {
  describe("when one item's scope catches a failure", () => {
    it("fails the build with that item's error type and message", async () => {
      const { build, counts } = faultyDevice({
        scopes: { c: { kind: "out-of-memory", message: "injected OOM on c" } },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("c", "injected OOM on c"));
      assertEquals(counts.destroyed, counts.created, "uploaded weights are not destroyed");
      assertEquals(counts.openScopes, 0, "error scopes are left on the stack");
    });
  });

  describe("when several items fail", () => {
    it("reports the first failing item in item order, even when a later item settles first", async () => {
      const { build, events } = faultyDevice({
        scopes: {
          b: { kind: "validation", message: "injected validation on b" },
          d: { kind: "out-of-memory", message: "injected OOM on d" },
        },
        // b settles last, d first: the first to settle MUST NOT win.
        settleMs: { a: 5, b: 30 },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuValidationError);
      assertEquals(error.message, scopeMessage("b", "injected validation on b"));
      assert(
        events.indexOf("settled d") < events.indexOf("settled b"),
        `the later item did not settle first, so the order is not exercised: ${events.join(", ")}`,
      );
    });

    it("reports an earlier item's scope failure over a later item's host-side failure", async () => {
      const { build, counts } = faultyDevice({
        scopes: { b: { kind: "out-of-memory", message: "injected OOM on b" } },
        settleMs: { b: 20 },
        throwOnAllocate: ["d"],
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("b", "injected OOM on b"));
      assertEquals(counts.openScopes, 0, "the failing item's scopes are left on the stack");
    });

    it("reports an earlier item's scope failure over a later block's read failure", async () => {
      const { build } = faultyDevice({
        scopes: { b: { kind: "out-of-memory", message: "injected OOM on b" } },
        settleMs: { b: 20 },
        rejectRead: { d: new Error("injected read failure on d") },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("b", "injected OOM on b"));
    });

    it("passes a rejected pop through as is when it is the first failure in item order", async () => {
      const rejected = new Error("injected popErrorScope rejection on b");
      const { build } = faultyDevice({
        scopes: {
          b: { kind: "reject", error: rejected },
          c: { kind: "out-of-memory", message: "injected OOM on c" },
        },
        settleMs: { b: 20 },
      });
      const error = await assertRejects(build);

      assertEquals(error, rejected, "the rejection is replaced or wrapped");
    });

    it("leaves no unhandled rejection behind when several pops reject", async () => {
      const unhandled: unknown[] = [];
      const onUnhandled = (reason: unknown): void => {
        unhandled.push(reason);
      };
      process.on("unhandledRejection", onUnhandled);
      try {
        const first = new Error("injected rejection on a");
        const { build } = faultyDevice({
          scopes: {
            a: { kind: "reject", error: first },
            b: { kind: "reject", error: new Error("injected rejection on b") },
            c: { kind: "reject", error: new Error("injected rejection on c") },
            d: { kind: "out-of-memory", message: "injected OOM on d" },
          },
          // The later rejections settle before the first one is thrown.
          settleMs: { a: 30 },
        });
        const error = await assertRejects(build);
        assertEquals(error, first);
        // Unhandled rejections are reported after the microtask queue drains; give it a few turns.
        await new Promise((resolve) => setTimeout(resolve, 50));
      } finally {
        process.off("unhandledRejection", onUnhandled);
      }

      assertEquals(unhandled, [], "a later item's rejection went unhandled");
    });
  });

  describe("when a failure has settled before the next item is pulled", () => {
    it("stops uploading the part and reports that failure", async () => {
      const { build, counts, events, writes } = faultyDevice({
        scopes: { b: { kind: "out-of-memory", message: "injected OOM on b" } },
        // b's scopes settle in the first macrotask; d's block arrives 20 ms later, so the failure
        // is known when d is pulled (c is pulled within the same microtask run as b).
        readDelayMs: { d: 20 },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("b", "injected OOM on b"));
      assert(events.includes("settled b"), `b's scopes never settled: ${events.join(", ")}`);
      assertEquals(
        writes.map(({ bytes }) => bytes),
        [[...BYTES.a], [...BYTES.b], [...BYTES.c]],
        "an item was written after the failure was known",
      );
      assertEquals(counts.created, 3, "a buffer was allocated after the failure was known");
      assertEquals(events.includes("pop d"), false, "d was uploaded after the failure was known");
      assertEquals(counts.destroyed, counts.created, "uploaded weights are not destroyed");
      assertEquals(counts.openScopes, 0, "error scopes are left on the stack");
    });

    it("still reports the first failing item in item order when a later item's failure stopped the part", async () => {
      const { build, events } = faultyDevice({
        scopes: {
          a: { kind: "out-of-memory", message: "injected OOM on a" },
          b: { kind: "validation", message: "injected validation on b" },
        },
        // b's failure settles first and stops the part when d is pulled (20 ms in); a's settles
        // only after the loop has left (40 ms in), and it is the one the per-item await reported.
        settleMs: { a: 40 },
        readDelayMs: { d: 20 },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("a", "injected OOM on a"));
      assertEquals(
        events.includes("pop d"),
        false,
        `the part did not stop early: ${events.join(", ")}`,
      );
      assert(
        events.indexOf("settled b") < events.indexOf("settled a"),
        `the later item did not settle first, so the order is not exercised: ${events.join(", ")}`,
      );
    });

    it("stops the part on a rejected pop too, and passes the rejection through", async () => {
      const rejected = new Error("injected popErrorScope rejection on b");
      const { build, counts, events } = faultyDevice({
        scopes: { b: { kind: "reject", error: rejected } },
        readDelayMs: { d: 20 },
      });
      const error = await assertRejects(build);

      assertEquals(error, rejected, "the rejection is replaced or wrapped");
      assert(events.includes("settled b"), `b's scopes never settled: ${events.join(", ")}`);
      assertEquals(counts.created, 3, "a buffer was allocated after the rejection was known");
      assertEquals(events.includes("pop d"), false, "d was uploaded after the rejection was known");
      assertEquals(counts.openScopes, 0, "error scopes are left on the stack");
    });
  });

  describe("when a failure settles after the last item was uploaded", () => {
    it("still fails the build, and the part's fence comes after every item has settled", async () => {
      const { build, events } = faultyDevice({
        scopes: { a: { kind: "out-of-memory", message: "injected OOM on a" } },
        settleMs: { a: 30 },
      });
      const error = await assertRejects(build);

      assertInstanceOf(error, GpuOutOfMemoryError);
      assertEquals(error.message, scopeMessage("a", "injected OOM on a"));
      assertEquals(events.includes("submit"), false, "the failed part's fence was issued");
    });
  });

  describe("when every item succeeds", () => {
    it("writes each block's bytes once into its own buffer, and settles every item before the fence", async () => {
      const { build, events, writes } = faultyDevice({
        // Settle out of order, so a fence that does not wait for them would show.
        settleMs: { a: 20, c: 10 },
      });
      const session = await build();
      const built = [...events];
      await session.dispose();

      assertEquals(
        writes,
        ITEMS.map((item, index) => ({ buffer: index + 1, offset: 0, bytes: [...BYTES[item]] })),
      );
      const submit = built.indexOf("submit");
      assert(submit >= 0, "the part's fence was not issued");
      for (const item of ITEMS) {
        const settled = built.indexOf(`settled ${item}`);
        assert(
          settled >= 0 && settled < submit,
          `item ${item} settled after the fence: ${built.join(", ")}`,
        );
      }
      // The deferral itself: every item's pops are issued before a's scopes settle (a per-item
      // await issues b's pops only after a has settled, 20 ms in).
      assert(
        built.lastIndexOf("pop d") < built.indexOf("settled a"),
        `an item's pops waited for an earlier item's scopes: ${built.join(", ")}`,
      );
      // Blocks are read and each item's two pops are issued in item order.
      assertEquals(
        built.filter((event) => event.startsWith("read ")),
        ITEMS.map((item) => `read ${item}`),
      );
      assertEquals(
        built.filter((event) => event.startsWith("pop ")),
        ITEMS.flatMap((item) => [`pop ${item}`, `pop ${item}`]),
      );
    });

    it("counts the wait for the part's error scopes in uploadFenceMs, not in supplyWaitMs", async () => {
      // The fake fence resolves at once and no read is delayed, so a's 80 ms settle delay is the
      // only wait of the build.
      const { build } = faultyDevice({ settleMs: { a: 80 } });
      const session = await build();
      const { uploadFenceMs, supplyWaitMs } = session.diagnostics().buildStats;
      await session.dispose();

      assert(
        uploadFenceMs >= 60,
        `the scope wait is missing from uploadFenceMs: ${uploadFenceMs} ms`,
      );
      assert(supplyWaitMs < 60, `the scope wait was counted as a supply wait: ${supplyWaitMs} ms`);
    });
  });
});
