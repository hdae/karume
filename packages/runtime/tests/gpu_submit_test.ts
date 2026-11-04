import { test } from "vite-plus/test";
import { assert, assertEquals, assertLess, assertRejects, assertThrows } from "@std/assert";
import { type GpuContext, GpuDeviceLostError } from "../src/gpu/device.ts";
import {
  DEFAULT_SUBMIT_POLICY,
  IN_FLIGHT_MARKER_INTERVAL,
  IN_FLIGHT_MAX_MARKERS,
  MEASURED_HISTORY,
  type SubmitPolicy,
  SubmitPolicyError,
  SubmitScheduler,
} from "../src/gpu/submit.ts";
import { BUFFER_USAGE } from "../src/gpu/webgpu-constants.ts";
import { fakeGpuContext } from "./helpers/fake-gpu.ts";

/**
 * A fake with only the surface SubmitScheduler touches. Passed through a cast because the whole
 * DOM type is not reproduced.
 */
type FakeGpu = {
  /** Only the device is fake; GpuContext (loss subscription, lock) is the production code. */
  readonly context: GpuContext;
  /** The dispatch count of each submitted command buffer. */
  readonly submitted: number[];
  /**
   * The number of `queue.onSubmittedWorkDone` calls.
   *
   * MUST: keep the tests that look at this. Calling it on every submit makes some implementations
   * block the host in the synchronous part until the GPU completes, fully serializing CPU encoding
   * and GPU execution (the module doc of submit.ts). **Nothing but the call count catches this
   * regression** — neither the outputs nor the statistics change by a single bit; it only gets
   * slower and stays green.
   */
  readonly calls: { workDone: number };
  /**
   * The destroy count of each timing resource (querySet, buffers), in allocation order. Leaks on
   * paths that fail midway through allocation or encoding, and double destruction on the normal
   * path, are checked here.
   */
  readonly destroys: number[];
};

type FakeOptions = {
  readonly lost?: Promise<GPUDeviceLostInfo>;
  readonly workDone?: () => Promise<void>;
  /**
   * When given, the device **has GPU timing enabled** (it has `timestamp-query`), and each chunk
   * reads back one set of these timestamps (in query order = begin, end per dispatch). A real GPU
   * can produce neither non-monotonic values nor a querySet overflow, so the clamping and capacity
   * checks can be reached only here.
   */
  readonly timestamps?: readonly (readonly bigint[])[];
  /**
   * Throws synchronously at the n-th timing-resource allocation (1-based, in allocation order)
   * (fault injection).
   */
  readonly failCreateAt?: number;
  /**
   * Throws synchronously when opening a timed pass (one with `timestampWrites`) (fault
   * injection).
   */
  readonly failTimedPass?: boolean;
};

const createFakeGpu = (options: FakeOptions = {}): FakeGpu => {
  const submitted: number[] = [];
  const calls = { workDone: 0 };
  const workDone = options.workDone ?? ((): Promise<void> => Promise.resolve());
  const timing = options.timestamps !== undefined;
  const stamps = [...(options.timestamps ?? [])];
  const destroys: number[] = [];
  /**
   * Records one allocation in the ledger and returns its destroy (at the fault-injection index
   * it throws without recording).
   */
  const track = (): (() => void) => {
    const index = destroys.length;
    if (options.failCreateAt === index + 1) {
      throw new TypeError(
        `fault injection: synchronous exception at timing-resource allocation ${index + 1}`,
      );
    }
    destroys.push(0);
    return (): void => {
      destroys[index] += 1;
    };
  };
  const device = {
    lost: options.lost ?? new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set<string>(timing ? ["timestamp-query"] : []),
    queue: {
      submit: (buffers: readonly { readonly count: number }[]): void => {
        for (const buffer of buffers) {
          submitted.push(buffer.count);
        }
      },
      onSubmittedWorkDone: (): Promise<void> => {
        calls.workDone += 1;
        return workDone();
      },
    },
    createQuerySet: () => ({ destroy: track() }),
    // The timing path creates two buffers, one for resolve and one for MAP_READ. Only the latter
    // carries timestamps, handed out one set at a time from this list in readback (chunk) order.
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const destroy = track();
      const mapped =
        (descriptor.usage & BUFFER_USAGE.MAP_READ) === 0
          ? new BigUint64Array(0)
          : BigUint64Array.from(stamps.shift() ?? []);
      return {
        mapAsync: (): Promise<void> => Promise.resolve(),
        getMappedRange: (): ArrayBuffer => mapped.buffer,
        unmap: (): void => {},
        destroy,
      };
    },
    createCommandEncoder: () => {
      let count = 0;
      return {
        beginComputePass: (descriptor?: GPUComputePassDescriptor) => {
          if (options.failTimedPass === true && descriptor?.timestampWrites !== undefined) {
            throw new TypeError("fault injection: synchronous exception in a timed pass");
          }
          return {
            setPipeline: (): void => {},
            setBindGroup: (): void => {},
            dispatchWorkgroups: (): void => {
              count += 1;
            },
            end: (): void => {},
          };
        },
        copyBufferToBuffer: (): void => {},
        resolveQuerySet: (): void => {},
        finish: () => ({ count }),
      };
    },
  };
  return { context: fakeGpuContext(device as unknown as GPUDevice), submitted, calls, destroys };
};

/**
 * Objects for the tests that look only at the copy surface (the fake encoder never touches their
 * contents).
 */
const fakeSource = {} as unknown as GPUBuffer;
const fakeTarget = {} as unknown as GPUBuffer;

/**
 * A fake clock that produces "the GPU ran for `elapsedMs` during the window".
 *
 * The measurement window is "the window's first submit → completion of the flush's
 * `onSubmittedWorkDone`", so advancing the clock at the moment of completion makes the window's
 * measurement exactly `elapsedMs` (the injected time source lets the rule be pinned).
 */
const gpuTimeClock = (
  elapsedMs: () => number,
): {
  readonly now: () => number;
  /** Produces time passing on the host side (= spent on encoding). */
  readonly advance: (ms: number) => void;
  readonly workDone: () => Promise<void>;
} => {
  let clock = 0;
  return {
    now: (): number => clock,
    advance: (ms: number): void => {
      clock += ms;
    },
    workDone: (): Promise<void> => {
      clock += elapsedMs();
      return Promise.resolve();
    },
  };
};

const fakePipeline = {} as unknown as GPUComputePipeline;
const fakeBindGroup = {} as unknown as GPUBindGroup;

/**
 * A fixed chunk policy with adaptation stopped (to observe the split boundaries
 * deterministically).
 */
const fixedPolicy = (size: number): SubmitPolicy => ({
  timeBudgetMs: 100,
  initialChunkSize: size,
  minChunkSize: size,
  maxChunkSize: size,
});

/**
 * 1 dispatch = one workgroup of work (so the unit of the time budget matches the dispatch
 * count).
 */
const dispatchMany = (scheduler: SubmitScheduler, count: number): void => {
  for (let i = 0; i < count; i += 1) {
    scheduler.dispatch(fakePipeline, fakeBindGroup, [1, 1, 1], "test:fake");
  }
};

/** The adaptive policy (4 dispatches until there is a measurement / a 100 ms budget). */
const adaptivePolicy: SubmitPolicy = {
  timeBudgetMs: 100,
  initialChunkSize: 4,
  minChunkSize: 1,
  maxChunkSize: 1024,
};

test("SubmitScheduler splits the dispatch sequence by chunk size and submits each chunk", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 10);
  assertEquals(gpu.submitted, [4, 4], "each chunk is submitted as soon as it fills");

  await scheduler.flush();
  assertEquals(gpu.submitted, [4, 4, 2], "flush submits the remainder");
  assertEquals(scheduler.stats.submitCount, 3);
  assertEquals(scheduler.stats.dispatchCount, 10);
});

test("SubmitScheduler's flush submits nothing when nothing is pending", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  await scheduler.flush();
  assertEquals(gpu.submitted, []);
  assertEquals(scheduler.stats.submitCount, 0);
  // It still waits for completion when empty (flush-before-destroy is an invariant that holds even
  // with "nothing to submit").
  assertEquals(gpu.calls.workDone, 1, "an empty flush still waits for completion");
  // No window was ever opened. Counting an empty flush as "a window measured at 0" would fill the
  // diagnostic history with empty entries and hide the adaptation's trend.
  assertEquals(scheduler.stats.measuredMs, [], "a flush that submitted nothing makes no window");
  assertEquals(scheduler.stats.measuredCount, 0);
});

// The only structural detector of option 1 (the MUST on {@link FakeGpu.calls}). Going back to an
// implementation that calls onSubmittedWorkDone on every submit changes neither the outputs nor
// the statistics by a single bit — only this test, which counts the calls, goes red.
test("onSubmittedWorkDone is called only once, by flush (not on every submit)", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4), () => 0);

  dispatchMany(scheduler, 20);
  assertEquals(gpu.submitted, Array(5).fill(4), "5 chunks submitted");
  assertEquals(gpu.calls.workDone, 0, "never called at submit time");

  await scheduler.flush();
  assertEquals(scheduler.stats.submitCount, 5);
  assertEquals(gpu.calls.workDone, 1, "5 submits, and still only the one wait in flush");

  // In the 2nd run (up to the next flush) too, only the flush adds to it.
  dispatchMany(scheduler, 12);
  assertEquals(gpu.calls.workDone, 1);
  await scheduler.flush();
  assertEquals(scheduler.stats.submitCount, 8);
  assertEquals(gpu.calls.workDone, 2, "once per flush");
});

test("SubmitScheduler rejects a policy equivalent to chunk size 0 at construction", () => {
  const gpu = createFakeGpu();

  for (const key of ["initialChunkSize", "minChunkSize", "maxChunkSize"] as const) {
    assertThrows(
      () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, [key]: 0 }),
      SubmitPolicyError,
      key,
    );
  }
  assertThrows(
    () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, initialChunkSize: 1.5 }),
    SubmitPolicyError,
  );
  assertThrows(
    () => new SubmitScheduler(gpu.context, { ...DEFAULT_SUBMIT_POLICY, timeBudgetMs: 0 }),
    SubmitPolicyError,
  );
  assertThrows(
    () =>
      new SubmitScheduler(gpu.context, {
        ...DEFAULT_SUBMIT_POLICY,
        minChunkSize: 8,
        maxChunkSize: 4,
      }),
    SubmitPolicyError,
  );
});

test("SubmitScheduler copies the policy at construction, so rewriting the original object after the check does not lift the limits", async () => {
  // If the checked values and the effective values were different, both the rejection of chunk size
  // 0 (the hang check) and the limit on submits (the TDR check) would be lifted by a one-line
  // rewrite after construction.
  const gpu = createFakeGpu();
  const policy: { -readonly [K in keyof SubmitPolicy]: number } = { ...fixedPolicy(16) };
  const scheduler = new SubmitScheduler(gpu.context, policy);
  policy.initialChunkSize = Infinity;
  policy.minChunkSize = Infinity;
  policy.maxChunkSize = Infinity;

  dispatchMany(scheduler, 32);
  assertEquals(gpu.submitted, [16, 16], "the chunks were cut by the rewritten limits");
  await scheduler.flush();
});

test("SubmitScheduler's copy of the policy also reads inherited and non-enumerable fields (accepting what was accepted before the copy)", () => {
  // A spread copy takes only own enumerable fields, and a policy that passed before the copy would
  // turn into a rejection for 0.
  const gpu = createFakeGpu();
  const inherited: SubmitPolicy = Object.create(fixedPolicy(4));
  const hidden: SubmitPolicy = { ...fixedPolicy(4) };
  for (const key of Object.keys(hidden)) {
    Object.defineProperty(hidden, key, { enumerable: false });
  }
  for (const policy of [inherited, hidden]) {
    const scheduler = new SubmitScheduler(gpu.context, policy);
    assertEquals(scheduler.stats.currentChunkSize, 4);
  }
});

test("DEFAULT_SUBMIT_POLICY is frozen, so a rewrite does not spread to every Session using the default", () => {
  assertThrows(() => {
    (DEFAULT_SUBMIT_POLICY as { initialChunkSize: number }).initialChunkSize = Infinity;
  }, TypeError);
  assertEquals(DEFAULT_SUBMIT_POLICY.initialChunkSize, 16);
});

// The querySet capacity with timing enabled (the chunk's dispatch count × 2) is enforced at
// construction. With only the encode-time check, it would first fail in **a later run**, after a
// measurement backs the estimate and the limit switches to maxChunkSize (something that worked
// starts failing midway).
test("a device with timing enabled rejects a maxChunkSize beyond the querySet capacity at construction", () => {
  const timed = createFakeGpu({ timestamps: [] });

  assertThrows(
    () => new SubmitScheduler(timed.context, { ...DEFAULT_SUBMIT_POLICY, maxChunkSize: 2048 }),
    SubmitPolicyError,
    "maxChunkSize",
  );
  // The default policy (1024 → 2,048 queries = exactly the capacity) passes (no effect on
  // existing callers).
  const ok = new SubmitScheduler(timed.context, DEFAULT_SUBMIT_POLICY);
  assertEquals(ok.stats.currentChunkSize, DEFAULT_SUBMIT_POLICY.initialChunkSize);

  // A device with timing disabled creates no querySet at all, so the same policy passes.
  const plain = createFakeGpu();
  const wide = new SubmitScheduler(plain.context, {
    ...DEFAULT_SUBMIT_POLICY,
    maxChunkSize: 2048,
  });
  assertEquals(wide.timing, undefined, "a device with timing disabled has no breakdown");
});

// A copy is zero work (it does not count toward the time budget) but counts as one entry toward
// the chunk's dispatch limit (the NOTE on copyBuffer in submit.ts). Dropping that keeps the outputs
// correct, so the regression "chunks grow larger than intended" can only be caught by the
// **number** of submits.
test("copyBuffer stays zero work and counts as one entry toward the chunk's entry limit", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [], "the first entry does not fill the chunk");

  scheduler.copyBuffer(fakeSource, 0, fakeTarget, 0, 16);
  assertEquals(
    gpu.submitted,
    [1],
    "the copy reaches the limit as the second entry (holding one dispatch)",
  );
  assertEquals(scheduler.stats.dispatchCount, 1, "the copy is not counted as a dispatch");

  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [1], "one more entry does not reach the limit");
  await scheduler.flush();
  assertEquals(gpu.submitted, [1, 1], "flush submits the remainder");
});

test("discard does not count a copy in discardedDispatches", () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 1);
  scheduler.copyBuffer(fakeSource, 0, fakeTarget, 0, 16);
  scheduler.discard();

  assertEquals(gpu.submitted, [], "what was discarded is not submitted");
  assertEquals(
    scheduler.stats.discardedDispatches,
    1,
    'a copy is not "a dispatch that did not run"',
  );
});

// A non-monotonic timestamp (begin > end) cannot be produced on a real GPU, so this branch is
// never reached by the measured-0 assertion (gpu_timing_test.ts). Only this fake reaches the
// clamped side.
test("a non-monotonic timestamp is clamped to 0 and the clamped count is kept", async () => {
  // A chunk of 2 dispatches = 4 queries. The 1st has begin > end (negative), the 2nd is 60 ns.
  const gpu = createFakeGpu({ timestamps: [[100n, 50n, 200n, 260n]] });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  dispatchMany(scheduler, 2);
  await scheduler.flush();

  const timing = scheduler.timing;
  assertEquals(timing?.totalNs, 60, "a negative delta is added as 0 (only the 60 ns one remains)");
  assertEquals(timing?.clampedNegativeSamples, 1, "the clamped count is not silently dropped");
  assertEquals(timing?.dispatchCount, 2, "the count is independent of the clamping");
  assertEquals(timing?.entries.length, 1, "two dispatches of the same key fold into one entry");
});

test("a synchronous exception midway through allocating timing resources still destroys the resources already allocated", async () => {
  // Allocation order: querySet → resolve buffer → readback buffer. Failing at the 3rd leaves the
  // first two unreachable even from discard, since they are not yet on the reclaim list (left to
  // the GC).
  const gpu = createFakeGpu({ timestamps: [], failCreateAt: 3 });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 1);
  await assertRejects(() => scheduler.flush(), TypeError, "fault injection");
  scheduler.discard();
  assertEquals(
    gpu.destroys,
    [1, 1],
    "the allocated querySet and resolve buffer were not destroyed",
  );
});

test("a synchronous exception midway through a timed encode still lets discard destroy all 3 resources", async () => {
  const gpu = createFakeGpu({ timestamps: [], failTimedPass: true });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 1);
  await assertRejects(() => scheduler.flush(), TypeError, "fault injection");
  scheduler.discard();
  assertEquals(
    gpu.destroys,
    [1, 1, 1],
    "the timing resources of the chunk that failed mid-encode remain",
  );
});

test("timing resources are destroyed exactly once on the normal path, and a later discard does not destroy them twice", async () => {
  const gpu = createFakeGpu({ timestamps: [[0n, 1_000n]] });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 1);
  await scheduler.flush();
  assertEquals(gpu.destroys, [1, 1, 1], "destroyed after the readback");
  scheduler.discard();
  assertEquals(
    gpu.destroys,
    [1, 1, 1],
    "discard destroyed already-reclaimed resources a second time",
  );
});

test("SubmitScheduler's flush throws on device loss instead of waiting forever", async () => {
  const lostInfo = { reason: "unknown", message: "test" } as unknown as GPUDeviceLostInfo;
  const gpu = createFakeGpu({
    lost: Promise.resolve(lostInfo),
    // Reproduces a lost device, where the completion notification never comes.
    workDone: () => new Promise<void>(() => {}),
  });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 2);
  await assertRejects(() => scheduler.flush(), GpuDeviceLostError);
});

// The check of the window definition itself: the start is "the time of the **first** submit in the
// window" (not the latest submit), and the work is "the workgroup total of **every chunk** queued
// in the window" (not just the last chunk's).
test("the measurement window runs from the first submit to flush completion, and the estimate divides by the window's total workgroups", async () => {
  const clock = gpuTimeClock(() => 11);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2), clock.now);

  // Submit chunk1 at t=0, spend 1 ms on the host, then submit chunk2 at t=1.
  dispatchMany(scheduler, 2);
  clock.advance(1);
  dispatchMany(scheduler, 2);
  assertEquals(gpu.submitted, [2, 2], "both submitted before the flush");
  assertEquals(scheduler.stats.measuredMs, [], "the window does not close until the flush");
  assertEquals(scheduler.stats.msPerWorkgroup, undefined);

  // The flush completes at t=1+11=12. Starting at "the latest submit (t=1)" would give 11 ms.
  await scheduler.flush();
  assertEquals(scheduler.stats.measuredMs, [12], "the start is the first submit (t=0)");
  assertEquals(scheduler.stats.measuredCount, 1);
  // 12 ms / 4 workgroups. Dividing by the last chunk's share (2 workgroups) would give 6.
  assertEquals(
    scheduler.stats.msPerWorkgroup,
    3,
    "the estimate is the window's measurement ÷ the window's total workgroups",
  );
  assertEquals(scheduler.stats.discardedDispatches, 0, "nothing is discarded on the success path");
});

test("measuredMs is capped at the latest MEASURED_HISTORY entries, and the cumulative count is in measuredCount", async () => {
  let elapsed = 0;
  const clock = gpuTimeClock(() => elapsed);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1), clock.now);
  const total = MEASURED_HISTORY + 5;

  // The i-th window measures i+1 ms (so it is visible that the retained ones are "the latest").
  for (let i = 0; i < total; i += 1) {
    elapsed = i + 1;
    dispatchMany(scheduler, 1);
    await scheduler.flush();
  }

  const stats = scheduler.stats;
  assertEquals(stats.measuredCount, total, "the cumulative count keeps counting past the history");
  assertEquals(
    stats.measuredMs.length,
    MEASURED_HISTORY,
    "the history is bounded (not proportional to the number of flushes)",
  );
  assertLess(stats.measuredMs.length, total, "no unbounded growth");
  assertEquals(stats.measuredMs[0], total - MEASURED_HISTORY + 1, "the oldest are dropped first");
  assertEquals(
    stats.measuredMs[MEASURED_HISTORY - 1],
    total,
    "the newest measurement stays at the end",
  );
});

test("discard drops pending encodes without submitting them and reports the count in the diagnostics", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(4));

  dispatchMany(scheduler, 6);
  assertEquals(gpu.submitted, [4], "only the one full chunk is submitted");

  scheduler.discard();
  assertEquals(scheduler.stats.discardedDispatches, 2);

  // The discarded leftovers are not submitted by a later flush either (the reason the failure
  // path's cleanup does not replay them).
  await scheduler.flush();
  assertEquals(gpu.submitted, [4], "flush does not resubmit the leftovers");
  assertEquals(scheduler.stats.submitCount, 1);
  assertEquals(scheduler.stats.dispatchCount, 6, "the queued count itself is not rewritten");
});

// Invariant 1 (the module doc of submit.ts): a measurement of 0 means "no information", not
// "fast". Using it as grounds for growth would inflate the chunks unconditionally in environments
// with a coarse timer, until the dispatch count reaches the whole graph and everything goes into
// one submit (= TDR territory).
test("while measurements stay 0, the chunk does not grow beyond the initial limit", async () => {
  const gpu = createFakeGpu();
  // The time never advances = every window measures 0 (reproducing a value below the resolution).
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, () => 0);

  for (let window = 0; window < 10; window += 1) {
    dispatchMany(scheduler, 4);
    await scheduler.flush();
  }
  assertEquals(
    gpu.submitted,
    Array(10).fill(4),
    "without a backing measurement it stays at the initial limit",
  );
  assertEquals(scheduler.stats.measuredCount, 10);
  assertEquals(scheduler.stats.measuredMs, Array(10).fill(0), "every window measured 0");
  assertEquals(
    scheduler.stats.msPerWorkgroup,
    undefined,
    "a window measured at 0 does not update the estimate",
  );

  // Ten windows of "no information" do not move the limit (going back to doubling turns this red).
  dispatchMany(scheduler, 12);
  assertEquals(gpu.submitted, Array(13).fill(4), "a measurement of 0 is no grounds for growth");
  assertEquals(scheduler.stats.currentChunkSize, 4);
});

// The flip side of invariant 1 (the division-by-zero side). Updating on a zero-work window makes
// msPerWorkgroup Infinity, after which every dispatch is judged over budget and the chunk sticks
// to minChunkSize (the opposite direction from runaway growth, but the same failure of "acting on
// a measurement that carries no information").
test("a zero-work window does not update the estimate", async () => {
  const clock = gpuTimeClock(() => 5);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2), clock.now);

  // Queue only dispatches with 0 workgroups (a no-op on the GPU, but a submit still happens).
  scheduler.dispatch(fakePipeline, fakeBindGroup, [0, 1, 1], "test:fake");
  scheduler.dispatch(fakePipeline, fakeBindGroup, [0, 1, 1], "test:fake");
  await scheduler.flush();

  assertEquals(gpu.submitted, [2], "the submit itself happens");
  assertEquals(
    scheduler.stats.measuredMs,
    [5],
    "the window closes and the measurement is recorded",
  );
  assertEquals(
    scheduler.stats.msPerWorkgroup,
    undefined,
    "a zero-work window carries no information",
  );
  assertEquals(scheduler.stats.currentChunkSize, 2, "stays at the unbacked limit");
});

// Invariant 2: one chunk's estimated time does not exceed the budget. The cut is at
// `timeBudgetMs * CHUNK_TIME_SAFETY / msPerWorkgroup` = 100 * 0.5 / 2 = 25 workgroups.
test("once a measurement backs the estimate, chunks are cut by the workgroup budget", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "8ms / 4 workgroup");
  assertEquals(
    scheduler.stats.currentChunkSize,
    1024,
    "once backed, the limit opens up to the hard limit",
  );

  dispatchMany(scheduler, 60);
  assertEquals(
    gpu.submitted,
    [4, 25, 25],
    "cut at exactly the 25-workgroup budget (the remainder of 10 is held)",
  );
});

// Invariant 2 continued: the budget check happens **before queuing**. Shrinking after queuing
// would leave the over-budget chunk already submitted, and "going over the budget once every
// time" would become the steady state (invariant 3).
test("a dispatch that alone exceeds the budget closes the chunk before it is queued and goes out on its own", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "the budget is 25 workgroups");

  dispatchMany(scheduler, 3);
  assertEquals(gpu.submitted, [4], "3 dispatches (3 workgroups) are still within the budget");
  // One dispatch of 30 workgroups = over the budget on its own. The 3 before it are closed off
  // before it is queued.
  scheduler.dispatch(fakePipeline, fakeBindGroup, [30, 1, 1], "test:fake");
  assertEquals(
    gpu.submitted,
    [4, 3],
    "the preceding ones are closed off without mixing in the heavy dispatch",
  );
  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [4, 3, 1], "a dispatch that cannot be split is a chunk by itself");
});

// The flip side of invariant 3: a slow measurement is fed forward to the next chunk (it shrinks).
test("after a slow chunk, the next chunk shrinks", async () => {
  // 200 ms for 4 workgroups = twice the 100 ms budget.
  const clock = gpuTimeClock(() => 200);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 50, "200ms / 4 workgroup");

  // Budget 50 ms ÷ 50 ms = 1 workgroup → shrinks to one dispatch at a time.
  dispatchMany(scheduler, 3);
  assertEquals(gpu.submitted, [4, 1, 1], "cut as soon as the budget is expected to be exceeded");
});

// chunkBudget (estimated side): the counter for the precondition of filing the cost proxy, "was an
// over-budget chunk observed". Only chunks that could not be cut (a dispatch that alone exceeds
// the budget) raise it.
test("the chunk diagnostics count the maximum estimated time and the over-budget chunks", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);

  dispatchMany(scheduler, 4);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 2, "8ms / 4 workgroup");
  assertEquals(scheduler.stats.chunkBudget.budgetMs, 50, "100 ms × safety factor 0.5");
  // The 1st chunk went out before there was a backing measurement = it has no estimated time (it
  // is not counted as 0 ms).
  assertEquals(
    scheduler.stats.chunkBudget.maxEstimatedMs,
    undefined,
    "chunks from before the backing measurement are not on the estimated side",
  );
  assertEquals(scheduler.stats.chunkBudget.overBudgetChunks, 0);

  // Close off at 3 workgroups (estimated 6 ms), then issue a lone 30-workgroup dispatch
  // (estimated 60 ms > 50 ms budget).
  dispatchMany(scheduler, 3);
  scheduler.dispatch(fakePipeline, fakeBindGroup, [30, 1, 1], "test:fake");
  assertEquals(
    scheduler.stats.chunkBudget.maxEstimatedMs,
    6,
    "the 3 workgroups that were closed off",
  );
  assertEquals(
    scheduler.stats.chunkBudget.overBudgetChunks,
    0,
    "chunks within the budget are not counted",
  );

  dispatchMany(scheduler, 1);
  assertEquals(gpu.submitted, [4, 3, 1], "a dispatch that cannot be split is a chunk by itself");
  assertEquals(scheduler.stats.chunkBudget.maxEstimatedMs, 60, "30 workgroup × 2ms");
  assertEquals(
    scheduler.stats.chunkBudget.overBudgetChunks,
    1,
    "the one that could not be cut is counted",
  );

  // The control has not changed by a single bit (the diagnostics do not affect the split
  // boundaries).
  await scheduler.flush();
  assertEquals(gpu.submitted, [4, 3, 1, 1], "flush only submits the remainder");
  assertEquals(
    scheduler.stats.chunkBudget.overBudgetChunks,
    1,
    "the remainder chunk is within the budget",
  );
});

// chunkBudget (measured side): when the "workgroup count = work" proxy is off, the estimated-side
// counter keeps saying "within budget" (it only compares values estimated with the same proxy, so
// in principle it cannot see it). The window mean is computed without any attribution, so it
// becomes the signal that picks up the discrepancy (it includes host time, so it is not a lower
// bound on the GPU's chunk time — see the "window where the host stalled" test below).
test("the window-mean chunk time catches, as a signal, the proxy discrepancy the estimate misses", async () => {
  let elapsed = 1;
  const clock = gpuTimeClock(() => elapsed);
  const gpu = createFakeGpu({ workDone: clock.workDone });
  // 1 dispatch = 1 chunk (makes the window's chunk count deterministic).
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1), clock.now);

  // Back the estimate with a light kernel: 1 ms per workgroup.
  dispatchMany(scheduler, 1);
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 1, "1ms / 1 workgroup");
  assertEquals(
    scheduler.stats.chunkBudget.maxWindowMeanMs,
    1,
    "for a one-chunk window the mean = the measurement",
  );

  // The same workgroup count, yet one window takes 300 ms (= the proxy is off by orders of
  // magnitude).
  elapsed = 300;
  dispatchMany(scheduler, 2);
  await scheduler.flush();

  const budget = scheduler.stats.chunkBudget;
  assertEquals(budget.maxWindowMeanMs, 150, "300 ms / 2 chunks");
  // 150 ms > the 50 ms budget. While the estimated side says "within budget", the window mean rises
  // past the budget.
  assertLess(budget.budgetMs, budget.maxWindowMeanMs ?? 0, "the window mean rises past the budget");
  assertEquals(
    budget.maxEstimatedMs,
    1,
    "the estimated side stays at 1 ms (it cannot see the proxy discrepancy)",
  );
  assertEquals(budget.overBudgetChunks, 0, "the estimated-side counter does not rise here at all");
});

// The window mean is wall-clock time "from the first submit → after the flush", so it includes host
// stalls, encoding and waits. With a stalled host it exceeds the budget even when the GPU is fast,
// so the window mean is not a lower bound on the GPU's chunk time — GPU-side overruns are judged by
// the per-submit GPU time (submitGpuTime).
test("in a window where the host stalled, the window mean exceeds the budget even though the GPU's submit is within it", async () => {
  const clock = gpuTimeClock(() => 1);
  // The width of one GPU chunk = 1,000,000 ns = 1 ms.
  const gpu = createFakeGpu({ workDone: clock.workDone, timestamps: [[0n, 1_000_000n]] });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1), clock.now);

  dispatchMany(scheduler, 1);
  // The host stalls for 200 ms after the window opens (GC, another task, a background tab, ...).
  clock.advance(200);
  await scheduler.flush();

  const budget = scheduler.stats.chunkBudget;
  assertEquals(budget.maxWindowMeanMs, 201, "200 ms host stall + 1 ms GPU");
  assertLess(budget.budgetMs, budget.maxWindowMeanMs ?? 0, "the window mean exceeds the budget");
  assertEquals(budget.submitGpuTime?.maxNs, 1_000_000, "the GPU submit is 1 ms");
  assertLess(
    (budget.submitGpuTime?.maxNs ?? Infinity) / 1_000_000,
    budget.budgetMs,
    "no over-budget chunk on the GPU side (the window mean alone cannot establish an overrun)",
  );
});

// submitGpuTime (the GPU-side clock): counts, by the width of each submit, the "one heavy submit"
// that the window mean (a window average including host time) cannot see
// (the addendum to ADR 0118 decision 6 — against the OS's per-submit timeout, only the per-submit
// maximum says anything). The width is the passes' timestamp minimum → maximum, gaps between
// passes included (not the sum of the dispatch differences).
test("the GPU time per submit is counted by the timestamp width, and chunks from before the backing measurement are counted separately", async () => {
  const clock = gpuTimeClock(() => 8);
  const gpu = createFakeGpu({
    workDone: clock.workDone,
    timestamps: [
      // Pre-backing chunk 1 (4 dispatches): width 992. It differs from the sum of the dispatch
      // differences (982) by the 10 of gaps between passes.
      [10n, 20n, 30n, 1000n, 1000n, 1001n, 1001n, 1002n],
      // Pre-backing chunk 2: width 400.
      [5000n, 5100n, 5100n, 5200n, 5200n, 5300n, 5300n, 5400n],
      // Post-backing chunk (one heavy dispatch): width 3000.
      [0n, 3000n],
    ],
  });
  const scheduler = new SubmitScheduler(gpu.context, adaptivePolicy, clock.now);
  assertEquals(
    scheduler.stats.chunkBudget.submitGpuTime,
    undefined,
    "not filled before the readback",
  );

  dispatchMany(scheduler, 8);
  assertEquals(gpu.submitted, [4, 4], "without a backing measurement, cut at initialChunkSize");
  await scheduler.flush();
  assertEquals(scheduler.stats.msPerWorkgroup, 1, "8 ms / 8 workgroups backs the estimate");
  assertEquals(scheduler.stats.chunkBudget.submitGpuTime, {
    submits: 2,
    maxNs: 992,
    unbackedSubmits: 2,
    maxUnbackedNs: 992,
    maxDispatchNs: 970,
    maxDispatchKey: "test:fake",
  });

  scheduler.dispatch(fakePipeline, fakeBindGroup, [1, 1, 1], "test:heavy");
  await scheduler.flush();
  assertEquals(
    scheduler.stats.chunkBudget.submitGpuTime,
    {
      submits: 3,
      maxNs: 3000,
      unbackedSubmits: 2,
      maxUnbackedNs: 992,
      maxDispatchNs: 3000,
      maxDispatchKey: "test:heavy",
    },
    "a post-backing chunk does not move the pre-backing maximum",
  );
  // The readback rides on the flush's wait. Waiting for completion on every submit would serialize
  // the CPU and the GPU (the MUST in the module doc of submit.ts) — the only waits are the 2
  // flushes.
  assertEquals(gpu.calls.workDone, 2, "no new completion wait was added");
});

test("the width of a submit's GPU time does not go negative when non-monotonic samples are mixed in", async () => {
  // The 1st dispatch has begin > end (negative). The width is min 50 → max 260; the largest single
  // dispatch is the 2nd, at 60.
  const gpu = createFakeGpu({ timestamps: [[100n, 50n, 200n, 260n]] });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  dispatchMany(scheduler, 2);
  await scheduler.flush();

  const observed = scheduler.stats.chunkBudget.submitGpuTime;
  assertEquals(observed?.maxNs, 210);
  assertEquals(observed?.maxDispatchNs, 60, "a negative difference is compared as 0");
});

test("while every single-dispatch difference is 0, no key is named as the largest dispatch", async () => {
  // In chunk 1 both dispatches have difference 0 (the width is only the gap between passes). The
  // first positive difference comes in chunk 2.
  const gpu = createFakeGpu({
    timestamps: [
      [100n, 100n, 200n, 200n],
      [0n, 5n],
    ],
  });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  dispatchMany(scheduler, 2);
  await scheduler.flush();

  const zeroOnly = scheduler.stats.chunkBudget.submitGpuTime;
  assertEquals(zeroOnly?.maxDispatchNs, 0);
  assertEquals(
    zeroOnly !== undefined && "maxDispatchKey" in zeroOnly,
    false,
    "not filled with the key of a dispatch that is not the largest",
  );

  scheduler.dispatch(fakePipeline, fakeBindGroup, [1, 1, 1], "test:heavy");
  await scheduler.flush();
  const observed = scheduler.stats.chunkBudget.submitGpuTime;
  assertEquals(observed?.maxDispatchNs, 5);
  assertEquals(observed?.maxDispatchKey, "test:heavy", "filled by the first positive difference");
});

test("a device with timing disabled has no per-submit GPU time", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  dispatchMany(scheduler, 4);
  await scheduler.flush();

  assertEquals(
    "submitGpuTime" in scheduler.stats.chunkBudget,
    false,
    "an unmeasured value is not filled with 0",
  );
});

// ---------------------------------------------------------------------------
// In-flight submit bound (ADR 0123)
// ---------------------------------------------------------------------------

/**
 * Runs one round of microtasks and timers (advances Promise settlement to the observation
 * point).
 */
const settle = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/**
 * An `onSubmittedWorkDone` that stores the resolvers in marker order (GPU completion is advanced
 * by hand).
 */
const manualWorkDone = (): {
  readonly workDone: () => Promise<void>;
  readonly complete: (index: number) => void;
} => {
  const resolvers: (() => void)[] = [];
  return {
    workDone: () => new Promise<void>((resolve) => resolvers.push(resolve)),
    complete: (index) => resolvers[index](),
  };
};

/**
 * One step of a run (the shape of `executeBakedPlanPaced`): queue `count` dispatches, then ask for
 * pacing. A completion marker is placed only when asked.
 */
const step = (scheduler: SubmitScheduler, count: number): Promise<void> | undefined => {
  dispatchMany(scheduler, count);
  return scheduler.backpressure();
};

test("a completion marker is placed, once, when a run asks for pacing after 64 submits since the previous marker, and the count restarts at a fence", async () => {
  const gpu = createFakeGpu();
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1));

  // The path that does not ask for pacing (enqueue) places no markers however many submits it
  // makes.
  dispatchMany(scheduler, IN_FLIGHT_MARKER_INTERVAL * (IN_FLIGHT_MAX_MARKERS + 2));
  assertEquals(gpu.calls.workDone, 0, "submits alone place none");
  await scheduler.flush();
  assertEquals(gpu.calls.workDone, 1, "one fence from flush");

  // The steps are not awaited: each assertion reads the marker count right after the step returns.
  void step(scheduler, IN_FLIGHT_MARKER_INTERVAL - 1);
  assertEquals(gpu.calls.workDone, 1, "none before reaching 64");
  void step(scheduler, 1);
  assertEquals(gpu.calls.workDone, 2, "one at the step that reaches 64");
  void step(scheduler, IN_FLIGHT_MARKER_INTERVAL + 30);
  assertEquals(gpu.calls.workDone, 3, "still one when a single step goes past 64");
  void step(scheduler, 30);
  assertEquals(gpu.calls.workDone, 3, "none after 30 since the previous marker");

  // After a fence both the window's chunk count and the marker position restart from 0 (the
  // previous window's remainder is not carried over).
  await scheduler.flush();
  assertEquals(gpu.calls.workDone, 4);
  void step(scheduler, IN_FLIGHT_MARKER_INTERVAL - 1);
  assertEquals(gpu.calls.workDone, 4, "the remainder from before the fence is not carried over");
  void step(scheduler, 1);
  assertEquals(gpu.calls.workDone, 5, "the marker position also returns to 0 at the fence");

  // A single-fence run (a fence that does not go through flush) restarts the count the same way.
  scheduler.submitPending();
  scheduler.closeMeasurementWindowAfterFence();
  void step(scheduler, IN_FLIGHT_MARKER_INTERVAL - 1);
  assertEquals(gpu.calls.workDone, 5);
  void step(scheduler, 1);
  assertEquals(gpu.calls.workDone, 6);

  // Discarding a window that never reached a fence restarts the count too.
  scheduler.discardMeasurementWindow();
  void step(scheduler, IN_FLIGHT_MARKER_INTERVAL - 1);
  assertEquals(gpu.calls.workDone, 6);
  void step(scheduler, 1);
  assertEquals(gpu.calls.workDone, 7);
});

test("up to the bound of pending markers no wait is made; beyond it the oldest marker is awaited and its completion releases the wait", async () => {
  const done = manualWorkDone();
  const gpu = createFakeGpu({ workDone: done.workDone });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1));
  const baseline = gpu.context.pendingLostListeners;

  for (let marker = 0; marker < IN_FLIGHT_MAX_MARKERS; marker += 1) {
    assertEquals(step(scheduler, IN_FLIGHT_MARKER_INTERVAL), undefined, "no wait up to the bound");
  }
  const wait = step(scheduler, IN_FLIGHT_MARKER_INTERVAL);
  assert(wait !== undefined, "beyond the bound it returns a wait");
  let released = false;
  void wait.then(() => {
    released = true;
  });

  done.complete(1);
  await settle();
  assertEquals(
    released,
    false,
    "it waits for the oldest marker (the 2nd resolving first does not release it)",
  );
  done.complete(0);
  await settle();
  assertEquals(released, true, "the oldest marker's completion releases it");
  assertEquals(
    scheduler.backpressure(),
    undefined,
    "the resolved marker is dropped and it is back within the bound",
  );
  assertEquals(gpu.context.pendingLostListeners, baseline, "no loss subscriptions are left behind");
});

test("only when it returns a wait does it first submit everything pending", () => {
  const gpu = createFakeGpu({ workDone: () => new Promise<void>(() => {}) });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(2));

  // 2 dispatches = 1 submit. Each step leaves one dispatch pending.
  assertEquals(step(scheduler, 3), undefined);
  assertEquals(gpu.submitted, [2]);
  for (let marker = 0; marker < IN_FLIGHT_MAX_MARKERS; marker += 1) {
    const before = gpu.submitted.length;
    assertEquals(step(scheduler, IN_FLIGHT_MARKER_INTERVAL * 2), undefined);
    assertEquals(
      gpu.submitted.length,
      before + IN_FLIGHT_MARKER_INTERVAL,
      "within the bound the chunk is not cut",
    );
  }
  const before = gpu.submitted.length;
  assert(step(scheduler, IN_FLIGHT_MARKER_INTERVAL * 2) !== undefined);
  assertEquals(
    gpu.submitted.length,
    before + IN_FLIGHT_MARKER_INTERVAL + 1,
    "the remainder was submitted before waiting",
  );
  assertEquals(gpu.submitted.at(-1), 1);
});

test("losing the device while waiting for a marker fails with GpuDeviceLostError (no hang)", async () => {
  const lost = Promise.withResolvers<GPUDeviceLostInfo>();
  const gpu = createFakeGpu({
    lost: lost.promise,
    // Reproduces a lost device, where the completion notification never comes.
    workDone: () => new Promise<void>(() => {}),
  });
  const scheduler = new SubmitScheduler(gpu.context, fixedPolicy(1));

  let wait: Promise<void> | undefined;
  for (let marker = 0; marker <= IN_FLIGHT_MAX_MARKERS; marker += 1) {
    wait = step(scheduler, IN_FLIGHT_MARKER_INTERVAL);
  }
  assert(wait !== undefined);
  lost.resolve({ reason: "unknown", message: "test" } as unknown as GPUDeviceLostInfo);
  await assertRejects(() => wait, GpuDeviceLostError, "in-flight submit bound wait");
});
