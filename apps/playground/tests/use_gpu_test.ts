import type { AcquireGpuOptions, GpuContext } from "@karume/runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vite-plus/test";

import {
  createGpuLifecycle,
  createGpuQueue,
  type GpuLease,
  type GpuOptions,
  type GpuState,
  leaseFor,
  leaseState,
  runOnGpuQueue,
} from "../src/hooks/use-gpu.ts";

type FakeGpu = { readonly id: number; readonly destroy: () => void };
type DeviceLostHandler = NonNullable<AcquireGpuOptions["onDeviceLost"]>;

/** What the fake `acquireGpu` did, in order ("acquire 1", "destroy 1", …). */
const events: string[] = [];
let acquired = 0;
/** While set, an acquisition resolves only when it settles (a slow adapter request). */
let acquireGate: Promise<void> | undefined;
/** The `onDeviceLost` each acquisition was given, by id. */
const lostHandlers = new Map<number, DeviceLostHandler>();
/** While set, every `destroy` throws it (after logging the destroy). */
let destroyFailure: Error | undefined;

vi.mock("@karume/runtime", () => ({
  acquireGpu: (options: AcquireGpuOptions): Promise<GpuContext> => {
    acquired += 1;
    const id = acquired;
    events.push(`acquire ${id}`);
    if (options.onDeviceLost !== undefined) lostHandlers.set(id, options.onDeviceLost);
    const gpu: FakeGpu = {
      id,
      destroy: () => {
        events.push(`destroy ${id}`);
        if (destroyFailure !== undefined) throw destroyFailure;
      },
    };
    // The lifecycle reads nothing of the context but `destroy`.
    return (acquireGate ?? Promise.resolve()).then(() => gpu as unknown as GpuContext);
  },
}));

/** Lets every queued macrotask (and the microtasks behind it) run. */
const flushTasks = async (rounds = 5): Promise<void> => {
  for (let round = 0; round < rounds; round += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
};

const deferred = (): { readonly promise: Promise<void>; readonly resolve: () => void } => {
  let resolve = (): void => undefined;
  const promise = new Promise<void>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
};

/** Opens a lease and waits until it reports its device. */
const openReady = async (
  lifecycle: ReturnType<typeof createGpuLifecycle>,
): Promise<{ readonly release: () => void; readonly states: GpuState[] }> => {
  const states: GpuState[] = [];
  const release = lifecycle.open({}, (state) => states.push(state));
  await flushTasks();
  expect(states.map(({ kind }) => kind)).toEqual(["ready"]);
  return { release, states };
};

const newLifecycle = () => createGpuLifecycle(createGpuQueue());

/** A device-lost info (a DOM type with a brand the fake cannot construct). */
const lostInfo = (reason: GPUDeviceLostReason, message: string): GPUDeviceLostInfo =>
  ({ reason, message }) as unknown as GPUDeviceLostInfo;

beforeEach(() => {
  events.length = 0;
  acquired = 0;
  acquireGate = undefined;
  lostHandlers.clear();
  destroyFailure = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("createGpuLifecycle (useGpu's device order)", () => {
  describe("when work tracked before the release is still running", () => {
    it("destroys the device only after that work settles", async () => {
      const lifecycle = newLifecycle();
      const { release } = await openReady(lifecycle);
      const run = deferred();
      void lifecycle.track(run.promise);

      release();
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      run.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1"]);
    });

    it("also waits for work that rejects, and passes the rejection on", async () => {
      const lifecycle = newLifecycle();
      const { release } = await openReady(lifecycle);
      const { promise: failing, reject } = Promise.withResolvers<void>();
      const tracked = lifecycle.track(failing);
      const outcome = tracked.then(
        () => "resolved",
        (error: unknown) => error,
      );

      release();
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      reject("lost");
      await flushTasks();
      expect(await outcome).toBe("lost");
      expect(events).toEqual(["acquire 1", "destroy 1"]);
    });
  });

  describe("when a later cleanup of the same commit tracks a disposal", () => {
    it("waits for that disposal, tracked after the release but before the next macrotask", async () => {
      const lifecycle = newLifecycle();
      const { release } = await openReady(lifecycle);
      const disposal = deferred();

      // React runs useGpu's cleanup first, then the page's cleanups of the same commit, synchronously.
      release();
      void lifecycle.track(disposal.promise);
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      disposal.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1"]);
    });
  });

  describe("when a run that settles while the destroy waits tracks a disposal", () => {
    it("waits for that disposal too", async () => {
      const lifecycle = newLifecycle();
      const { release } = await openReady(lifecycle);
      const run = deferred();
      const disposal = deferred();
      void lifecycle.track(run.promise.then(() => void lifecycle.track(disposal.promise)));

      release();
      // The destroy's macrotask has fired: it is waiting for the run when the run tracks its disposal.
      await flushTasks();
      run.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      disposal.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1"]);
    });
  });

  describe("when the next lease opens right after a release", () => {
    it("acquires only after the previous device is destroyed", async () => {
      const lifecycle = newLifecycle();
      const { release } = await openReady(lifecycle);
      const disposal = deferred();
      release();
      void lifecycle.track(disposal.promise);
      const next: GpuState[] = [];
      lifecycle.open({}, (state) => next.push(state));
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);
      expect(next).toEqual([]);

      disposal.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1", "acquire 2"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });

  describe("when the released device's destroy throws", () => {
    it("reports the error and still lets every later acquisition through", async () => {
      const report = vi.spyOn(console, "error").mockImplementation(() => undefined);
      const failure = new Error("destroy failed");
      destroyFailure = failure;
      const queue = createGpuQueue();
      const { release } = await openReady(createGpuLifecycle(queue));
      release();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1"]);
      expect(report).toHaveBeenCalledWith("Destroying the GPU device failed", failure);

      destroyFailure = undefined;
      const second = await openReady(createGpuLifecycle(queue));
      second.release();
      await openReady(createGpuLifecycle(queue));
      expect(events).toEqual(["acquire 1", "destroy 1", "acquire 2", "destroy 2", "acquire 3"]);
    });
  });

  describe("when another lifecycle shares the queue (the next page, or the same page mounted again)", () => {
    it("acquires only after the first lifecycle's device is destroyed", async () => {
      const queue = createGpuQueue();
      const leaving = createGpuLifecycle(queue);
      const { release } = await openReady(leaving);
      const run = deferred();
      void leaving.track(run.promise);

      release();
      const next: GpuState[] = [];
      createGpuLifecycle(queue).open({}, (state) => next.push(state));
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);
      expect(next).toEqual([]);

      run.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1", "acquire 2"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });

  describe("when released while still acquiring", () => {
    it("acquires the next device without waiting, and destroys the abandoned one when it arrives", async () => {
      const lifecycle = newLifecycle();
      const gate = deferred();
      acquireGate = gate.promise;
      const first: GpuState[] = [];
      const release = lifecycle.open({}, (state) => first.push(state));
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      // The adapter request of the first never settles for now; the next one is not held up by it.
      acquireGate = undefined;
      release();
      const next: GpuState[] = [];
      lifecycle.open({}, (state) => next.push(state));
      await flushTasks();
      expect(events).toEqual(["acquire 1", "acquire 2"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);

      gate.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "acquire 2", "destroy 1"]);
      expect(first).toEqual([]);
    });

    it("skips an acquisition released before it started (React StrictMode's setup → cleanup → setup)", async () => {
      const lifecycle = newLifecycle();
      const first: GpuState[] = [];
      const release = lifecycle.open({}, (state) => first.push(state));
      release();
      const next: GpuState[] = [];
      lifecycle.open({}, (state) => next.push(state));
      await flushTasks();

      expect(events).toEqual(["acquire 1"]);
      expect(first).toEqual([]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });

  describe("when a device is lost", () => {
    it("reports the loss while the lease is open", async () => {
      const lifecycle = newLifecycle();
      const { states } = await openReady(lifecycle);
      lostHandlers.get(1)?.(lostInfo("unknown", "GPU process crashed"));
      expect(states.map(({ kind }) => kind)).toEqual(["ready", "lost"]);
    });

    it("reports nothing after the release (its own destroy loses the device)", async () => {
      const lifecycle = newLifecycle();
      const { release, states } = await openReady(lifecycle);
      release();
      await flushTasks();
      lostHandlers.get(1)?.(lostInfo("destroyed", "Device was destroyed."));
      expect(states.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });
});

describe("runOnGpuQueue (a task with a device of its own, such as runGeometrySweep)", () => {
  /** A task that logs its start and end and settles when told to. */
  const queuedTask = (name: string) => {
    const done = deferred();
    return {
      done,
      task: async (): Promise<string> => {
        events.push(`${name} start`);
        await done.promise;
        events.push(`${name} end`);
        return name;
      },
    };
  };

  describe("when the previous page's device is still waiting for its work", () => {
    it("starts the task only after that device is destroyed", async () => {
      const queue = createGpuQueue();
      const leaving = createGpuLifecycle(queue);
      const { release } = await openReady(leaving);
      const run = deferred();
      void leaving.track(run.promise);
      release();

      const sweep = queuedTask("sweep");
      const result = runOnGpuQueue(queue, sweep.task);
      await flushTasks();
      expect(events).toEqual(["acquire 1"]);

      run.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "destroy 1", "sweep start"]);
      sweep.done.resolve();
      expect(await result).toBe("sweep");
    });
  });

  describe("while the task runs", () => {
    it("holds the next page's acquisition until the task has settled", async () => {
      const queue = createGpuQueue();
      const sweep = queuedTask("sweep");
      void runOnGpuQueue(queue, sweep.task);
      // The next page mounts in a later commit and opens its device in its effect.
      const next: GpuState[] = [];
      createGpuLifecycle(queue).open({}, (state) => next.push(state));
      await flushTasks();
      expect(events).toEqual(["sweep start"]);
      expect(next).toEqual([]);

      sweep.done.resolve();
      await flushTasks();
      expect(events).toEqual(["sweep start", "sweep end", "acquire 1"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });

    it("holds a second task (the page entered again and run again) until the first has settled", async () => {
      const queue = createGpuQueue();
      const first = queuedTask("first");
      const second = queuedTask("second");
      void runOnGpuQueue(queue, first.task);
      void runOnGpuQueue(queue, second.task);
      await flushTasks();
      expect(events).toEqual(["first start"]);

      first.done.resolve();
      await flushTasks();
      expect(events).toEqual(["first start", "first end", "second start"]);
    });
  });

  describe("when a useGpu device is released while the task runs (the two writers of the queue)", () => {
    it("holds the next acquisition until both the task and that device are gone", async () => {
      const queue = createGpuQueue();
      const { release } = await openReady(createGpuLifecycle(queue));
      const sweep = queuedTask("sweep");
      void runOnGpuQueue(queue, sweep.task);
      release();
      const next: GpuState[] = [];
      createGpuLifecycle(queue).open({}, (state) => next.push(state));
      await flushTasks();
      expect(events).toEqual(["acquire 1", "sweep start", "destroy 1"]);
      expect(next).toEqual([]);

      sweep.done.resolve();
      await flushTasks();
      expect(events).toEqual(["acquire 1", "sweep start", "destroy 1", "sweep end", "acquire 2"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });

  describe("when the task rejects", () => {
    it("passes the rejection on and still lets the next acquisition through", async () => {
      const queue = createGpuQueue();
      const failing = runOnGpuQueue(queue, () => Promise.reject(new Error("no adapter")));
      const outcome = failing.then(
        () => "resolved",
        (error: unknown) => (error instanceof Error ? error.message : "other"),
      );
      const next: GpuState[] = [];
      createGpuLifecycle(queue).open({}, (state) => next.push(state));
      await flushTasks();
      expect(await outcome).toBe("no adapter");
      expect(events).toEqual(["acquire 1"]);
      expect(next.map(({ kind }) => kind)).toEqual(["ready"]);
    });
  });
});

describe("leaseFor / leaseState (the state a useGpu render shows)", () => {
  const PLAIN: GpuOptions = {};
  const TIMED: GpuOptions = { gpuTiming: true };
  const gpu = { id: 1 } as unknown as GpuContext;
  const readyFor = (lease: GpuLease) => ({ lease, state: { kind: "ready", gpu } as const });

  describe("while the options and the attempt stay the same", () => {
    it("keeps the lease, so the held state shows", () => {
      const lease = leaseFor({ options: PLAIN, attempt: 0 }, PLAIN, 0);
      const next = leaseFor(lease, PLAIN, 0);
      expect(next).toBe(lease);
      expect(leaseState(readyFor(lease), next)).toEqual({ kind: "ready", gpu });
    });
  });

  describe("in the render that changes the options or the attempt", () => {
    it("shows 'acquiring' at once, before the new device reports", () => {
      const lease: GpuLease = { options: PLAIN, attempt: 0 };
      const held = readyFor(lease);
      expect(leaseState(held, leaseFor(lease, TIMED, 0))).toEqual({ kind: "acquiring" });
      expect(leaseState(held, leaseFor(lease, PLAIN, 1))).toEqual({ kind: "acquiring" });
    });
  });

  describe("when the options return to earlier values before the new device reports", () => {
    it("still shows 'acquiring', never the released device", () => {
      const first: GpuLease = { options: PLAIN, attempt: 0 };
      const held = readyFor(first);
      const switched = leaseFor(first, TIMED, 0);
      const reverted = leaseFor(switched, PLAIN, 0);
      expect(reverted).not.toBe(first);
      expect(leaseState(held, reverted)).toEqual({ kind: "acquiring" });
    });
  });

  it("shows 'acquiring' before any state was reported", () => {
    expect(leaseState(undefined, { options: PLAIN, attempt: 0 })).toEqual({ kind: "acquiring" });
  });
});
