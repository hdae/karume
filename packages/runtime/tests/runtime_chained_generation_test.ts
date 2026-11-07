// Chained deferred runs (`GenerationRun.chain`) at the context's lease, without a GPU.
//
// The execution integration (`Session.run` / `Session.enqueue`) drives a GenerationContext through
// its internal surface in a fixed order: take the lease at issue → read the position and write the
// lengths uniform in the body → defer on success (a batch: every member in acceptance order) →
// return the lease. These cases drive that same order against a context allocated on a fake device,
// so the admission rules, the placement and the commit of a chain are checked where no adapter
// exists. The values on a real GPU (each chained run reads its own lengths) are
// `gpu_chained_generation_test.ts`.

import { describe, expect, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import {
  GpuContext,
  readAdapterInfo,
  type RequiredLimits,
  RUNTIME_INTERNAL,
} from "../src/gpu/device.ts";
import {
  type ChainedRun,
  GenerationContext,
  type GenerationContextHost,
} from "../src/runtime/generation-context.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { fakeDevice } from "./helpers/fake-gpu.ts";
import { type DeclarationJson, GRAPH_NAME, memoryModel } from "./helpers/model-fixture.ts";

/** Row capacity of both slots. With the window 4 of the sliding form the slack is 8 − 4 = 4. */
const CAPACITY = 8;
const WINDOW = 4;

/** attention + two state_append over `[1,1,8,4]` slots (sliding when `window` is given). */
const declaration = (window?: number): DeclarationJson => {
  const attrs = window === undefined ? {} : { window };
  return {
    format: "karume-ir",
    version: 2,
    symbols: ["M"],
    requires: { ops: ["attention", "state_append"] },
    initializers: {},
    inputs: ["q", "k", "v"].map((name) => ({ name, dtype: "f32", shape: [1, 1, "M", 4] })),
    outputs: ["o"],
    values: { o: { dtype: "f32", shape: [1, 1, "M", 4] } },
    states: {
      kslot: { dtype: "f32", shape: [1, 1, CAPACITY, 4] },
      vslot: { dtype: "f32", shape: [1, 1, CAPACITY, 4] },
    },
    nodes: [
      {
        op: "attention",
        ins: ["q", "k", "v"],
        outs: ["o"],
        attrs: { scale: 0.5, ...attrs },
        states: { k: "kslot", v: "vslot" },
      },
      ...[
        ["k", "kslot"],
        ["v", "vslot"],
      ].map(([name, slot]) => ({
        op: "state_append",
        ins: [name],
        outs: [],
        attrs,
        states: { slot },
      })),
    ],
  };
};

const LIMITS: RequiredLimits = {
  maxBufferSize: 2 ** 30,
  maxStorageBufferBindingSize: 2 ** 30,
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

type Rig = {
  readonly context: GenerationContext;
  /** Every `(pastLength, queryLength)` pair written to the lengths uniform, in order. */
  readonly lengths: (readonly [number, number])[];
};

/** A context on a fake device (`window` given = sliding slots). */
const rig = async (options: { window?: number; chunkLength: number }): Promise<Rig> => {
  const lengths: (readonly [number, number])[] = [];
  const device = fakeDevice({
    createBuffer: (descriptor) =>
      ({
        size: descriptor.size,
        label: descriptor.label,
        destroy: () => undefined,
      }) as unknown as GPUBuffer,
    writeBuffer: (_buffer, _offset, data) => {
      const words = new Uint32Array(data as Uint32Array<ArrayBuffer>);
      lengths.push([words[0], words[1]]);
    },
  });
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const graph = mergedGraph(
    memoryModel(declaration(options.window), []).graphs[GRAPH_NAME],
    GRAPH_NAME,
  );
  const host: GenerationContextHost = {
    gpu,
    graph,
    fullStateStorage: "f32",
    flush: () => Promise.resolve(),
    serialize: <T>(body: () => Promise<T>): Promise<T> => body(),
    forget: () => undefined,
    adopt: (allocate) => allocate(),
  };
  const context = await GenerationContext.create(host, { chunkLength: options.chunkLength });
  return { context, lengths };
};

/** One immediate run, the way `Session.run` drives the lease (prefill / decode). */
const runImmediate = (context: GenerationContext, queryLength: number): void => {
  const lease = context[RUNTIME_INTERNAL];
  lease.acquireRun();
  const past = lease.pastLength();
  lease.writeLengths(past, queryLength);
  lease.advance(past, queryLength);
  lease.releaseRun();
};

/** One unchained deferred run that succeeds (the verify shape of speculative decoding). */
const runDeferred = (context: GenerationContext, queryLength: number): void => {
  const lease = context[RUNTIME_INTERNAL];
  lease.acquireRun();
  const past = lease.pastLength();
  lease.writeLengths(past, queryLength);
  lease.defer(past, queryLength);
  lease.releaseRun();
};

/** The body of a chained run: read the reserved position and write it to the lengths uniform. */
const encodeChained = (context: GenerationContext, run: ChainedRun): number => {
  const lease = context[RUNTIME_INTERNAL];
  const past = lease.pastLength(run);
  lease.writeLengths(past, run.queryLength, run);
  return past;
};

/** A batch settling: every member completes in acceptance order, then every lease returns. */
const settle = (context: GenerationContext, runs: readonly ChainedRun[], ok = true): void => {
  const lease = context[RUNTIME_INTERNAL];
  if (ok) for (const run of runs) lease.defer(run.pastLength, run.queryLength, run);
  for (const run of runs) lease.releaseRun(run);
};

describe("chained deferred runs (GenerationRun.chain)", () => {
  describe("inside one batch", () => {
    it("places each run right after the previous one and writes that position for it", async () => {
      const { context, lengths } = await rig({ window: WINDOW, chunkLength: 4 });
      runImmediate(context, 3);
      const scope = {};
      const lease = context[RUNTIME_INTERNAL];
      const runs = [1, 1, 1].map((rows) => lease.acquireChainedRun(rows, scope));

      expect(runs.map((run) => run.pastLength)).toEqual([3, 4, 5]);
      expect(runs.map((run) => encodeChained(context, run))).toEqual([3, 4, 5]);
      // The prefill's pair, then one pair per chained run carrying its own position.
      expect(lengths).toEqual([
        [0, 3],
        [3, 1],
        [4, 1],
        [5, 1],
      ]);
    });

    it("joins the pending commit only when the batch settles, and commit folds the whole range", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      runImmediate(context, 3);
      const scope = {};
      const runs = [1, 1, 1].map((rows) =>
        context[RUNTIME_INTERNAL].acquireChainedRun(rows, scope),
      );
      for (const run of runs) encodeChained(context, run);

      expect(context.pendingCommit).toBeUndefined();
      expect(() => context.commit(0)).toThrow(ExecutionError);

      settle(context, runs);
      expect(context.pendingCommit).toEqual({ pastLength: 3, queryLength: 3 });
      expect(context.pastLength).toBe(3);

      context.commit(2);
      expect(context.pastLength).toBe(5);
      expect(context.pendingCommit).toBeUndefined();
      // The next run is placed after the committed rows (the rejected row is overwritten).
      runImmediate(context, 1);
      expect(context.pastLength).toBe(6);
    });

    it("folds nothing when the batch fails, leaving the context where it was", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      runImmediate(context, 2);
      const scope = {};
      const runs = [1, 1].map((rows) => context[RUNTIME_INTERNAL].acquireChainedRun(rows, scope));
      settle(context, runs, false);

      expect(context.pendingCommit).toBeUndefined();
      expect(context.pastLength).toBe(2);
      // A failure that wrote no state leaves a usable context: the next chain starts over at 2.
      expect(context[RUNTIME_INTERNAL].acquireChainedRun(1, {}).pastLength).toBe(2);
    });

    it("hands the position of an acceptance refused after reserving back to the next run", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      const scope = {};
      const lease = context[RUNTIME_INTERNAL];
      const first = lease.acquireChainedRun(1, scope);
      const refused = lease.acquireChainedRun(1, scope);
      lease.releaseRun(refused);
      const next = lease.acquireChainedRun(1, scope);

      expect(next.pastLength).toBe(refused.pastLength);
      settle(context, [first, next]);
      expect(context.pendingCommit).toEqual({ pastLength: 0, queryLength: 2 });
    });

    it("refuses to fold a run that does not continue the uncommitted range", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      const scope = {};
      const lease = context[RUNTIME_INTERNAL];
      const [first, second] = [1, 1].map((rows) => lease.acquireChainedRun(rows, scope));

      expect(() => lease.defer(second.pastLength, 1, second)).toThrow("does not continue");
      lease.defer(first.pastLength, 1, first);
      lease.defer(second.pastLength, 1, second);
      expect(context.pendingCommit).toEqual({ pastLength: 0, queryLength: 2 });
    });
  });

  describe("after a settled deferred run", () => {
    it("continues from the end of the pending range", async () => {
      const { context, lengths } = await rig({ window: WINDOW, chunkLength: 4 });
      runImmediate(context, 3);
      runDeferred(context, 2);
      expect(context.pendingCommit).toEqual({ pastLength: 3, queryLength: 2 });

      // A chained `Session.run` (no scope) next to the pending commit.
      const run = context[RUNTIME_INTERNAL].acquireChainedRun(1, undefined);
      expect(run.pastLength).toBe(5);
      expect(encodeChained(context, run)).toBe(5);
      settle(context, [run]);

      expect(lengths.at(-1)).toEqual([5, 1]);
      expect(context.pendingCommit).toEqual({ pastLength: 3, queryLength: 3 });
      context.commit(3);
      expect(context.pastLength).toBe(6);
    });

    it("still refuses an unchained run while the commit is pending", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      runDeferred(context, 2);
      expect(() => context[RUNTIME_INTERNAL].acquireRun()).toThrow("awaiting commit");
      expect(() => context[RUNTIME_INTERNAL].acquireRun("batch")).toThrow("awaiting commit");
    });
  });

  describe("the bound of the uncommitted range", () => {
    it("is the sliding slack on a context with sliding slots", async () => {
      // chunkLength 6 > slack 4, so the slack is the bound that bites.
      const { context } = await rig({ window: WINDOW, chunkLength: 6 });
      expect(context.slidingSlack).toBe(CAPACITY - WINDOW);
      runDeferred(context, 2);
      const scope = {};
      const lease = context[RUNTIME_INTERNAL];
      const runs = [lease.acquireChainedRun(1, scope), lease.acquireChainedRun(1, scope)];

      expect(() => lease.acquireChainedRun(1, scope)).toThrow("slack 4");
      // The refusal reserved nothing: the batch settles as before and commits all four rows.
      settle(context, runs);
      expect(context.pendingCommit).toEqual({ pastLength: 0, queryLength: 4 });
      context.commit(4);
      expect(context.pastLength).toBe(4);
    });

    it("is chunkLength on a context without sliding slots", async () => {
      const { context } = await rig({ chunkLength: 3 });
      expect(context.slidingSlack).toBeUndefined();
      const scope = {};
      const lease = context[RUNTIME_INTERNAL];
      const runs = [1, 1, 1].map((rows) => lease.acquireChainedRun(rows, scope));

      expect(() => lease.acquireChainedRun(1, scope)).toThrow("chunkLength 3");
      settle(context, runs);
      expect(context.pendingCommit).toEqual({ pastLength: 0, queryLength: 3 });
    });
  });

  describe("concurrency", () => {
    it("refuses a chained run of another batch, and a chained run() while one is in flight", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      const lease = context[RUNTIME_INTERNAL];
      const run = lease.acquireChainedRun(1, {});

      expect(() => lease.acquireChainedRun(1, {})).toThrow("outside its BatchScope");
      expect(() => lease.acquireChainedRun(1, undefined)).toThrow("outside its BatchScope");
      settle(context, [run]);
      // Once the first chain settled, a chained run() continues it.
      expect(lease.acquireChainedRun(1, undefined).pastLength).toBe(1);
    });

    it("keeps every refusal of unchained callers while chained runs are in flight", async () => {
      const { context } = await rig({ chunkLength: 4 });
      const lease = context[RUNTIME_INTERNAL];
      const run = lease.acquireChainedRun(1, {});

      expect(() => lease.acquireRun()).toThrow("in progress");
      expect(() => lease.acquireRun("batch")).toThrow("in progress");
      expect(() => context.commit(0)).toThrow("in progress");
      expect(() => context.rewind(0)).toThrow("in progress");
      await expect(context.fork()).rejects.toThrow("batch");
      settle(context, [run]);
      expect(() => context.rewind(0)).toThrow("awaiting commit");
    });

    it("refuses a chained run once the context is poisoned", async () => {
      const { context } = await rig({ window: WINDOW, chunkLength: 4 });
      const lease = context[RUNTIME_INTERNAL];
      const runs = [lease.acquireChainedRun(1, {})];
      // The failure path of a run that submitted state writes (`#poisonOnStateWrite`).
      lease.poison("injected failure after a state write");
      settle(context, runs, false);

      expect(() => lease.acquireChainedRun(1, undefined)).toThrow("poisoned");
      expect(() => context.pendingCommit).toThrow("poisoned");
    });
  });
});
