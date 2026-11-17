// The sweeps' timing protocol on a fake device (no GPU needed): the warm start and the calibration
// check of src/tune/harness.ts (`warmStart`, `calibrate`), the geometry sweep's case starting warm,
// and the GEMV sweep's row count (src/tune/gemv-sweep.ts) — its warm start on the baseline, its
// interleaved rounds and the spread of the baseline's rounds.
//
// The fake device's passes take the time the test gives them: a timed pass's timestamp difference
// (`gpuNs`, by the pass's pipeline key, dispatch count and index), and the real wall clock a pass
// takes to complete (`wallMs` — the probe reads only that). GpuContext is the real one; only the
// device is replaced (as in tune_harness_callbacks_test.ts).
//
// Faults injected, one per rule: a dispatch that speeds up over its first passes (the warm start must
// wait until its passes agree), passes that agree from the start (it must still spend its floor),
// passes that never agree (it must stop at its ceiling), passes too short to add up the floor at the
// reps bound (it must stop at its cut-off of passes), a probe that reads the dispatch slow and
// passes that then come out short (the passes must be resized), a probe taken while the dispatch was
// slow (the calibration must be done again after more load), rounds of the baseline that differ by
// 1 % (the spread must say so) and a cancel in the middle of the rounds (no row may come of them).
// The warm start of each sweep, and the rotation of the rounds, are read from the log of passes.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { GpuContext, readAdapterInfo, type RequiredLimits } from "../src/gpu/device.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import { SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import {
  type GemvCandidate,
  type GemvGates,
  gemvBaseline,
  gemvCandidates,
  type GemvMeasurementPlan,
  gemvProfileChoice,
  gemvProgram,
} from "../src/tune/gemv-candidates.ts";
import type { GemvCase } from "../src/tune/gemv-cases.ts";
import { gemvLaunch, sweepGemvCase } from "../src/tune/gemv-sweep.ts";
import {
  calibrate,
  casePlan,
  createResources,
  createSweepContext,
  destroySweepContext,
  prepareLaunch,
  type ReadyLaunch,
  runSweep,
  SWEEP_MAX_REPS,
  type SweepContext,
  WARM_START_MAX_PASSES,
  warmStart,
} from "../src/tune/harness.ts";
import { WARM_START_MAX_NS, WARM_START_MIN_NS } from "../src/tune/measurement.ts";

/** One pass the fake device ran. */
type FakePass = {
  /** The key of the pipeline the pass set. */
  readonly key: string;
  readonly dispatches: number;
  readonly timed: boolean;
  /** Its index among the passes of its key. */
  readonly index: number;
  /** Its index among the timed passes of its key (counted when it is timed). */
  readonly timedIndex: number;
};

/** A pass as the log keeps it: with its timestamp difference when it was timed. */
type LoggedPass = FakePass & { readonly gpuNs?: number };

type FakeTiming = {
  /** A timed pass's timestamp difference (ns). */
  readonly gpuNs: (pass: FakePass) => number;
  /** The real time a pass takes to complete (ms; absent: at once). */
  readonly wallMs?: (pass: FakePass) => number;
};

/** A compute pass while it is encoded. */
type OpenPass = { key: string; dispatches: number; readonly timed: boolean };

/** What the fake encoder's `finish` gives `submit`. */
type FakeCommands = { readonly pass?: OpenPass };

const LIMITS: RequiredLimits = {
  maxBufferSize: 2 ** 30,
  maxStorageBufferBindingSize: 2 ** 30,
  maxUniformBufferBindingSize: 65536,
  maxStorageBuffersPerShaderStage: 10,
  maxUniformBuffersPerShaderStage: 12,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  maxComputeWorkgroupSizeX: 1024,
  maxComputeWorkgroupSizeY: 1024,
  maxComputeWorkgroupSizeZ: 64,
  maxComputeWorkgroupsPerDimension: 65535,
};

/**
 * A fake device with `timestamp-query` and only the surface the sweeps touch. Every pass is logged in
 * submission order; a timestamp readback returns the difference of the last timed pass; an output
 * readback returns zeros.
 */
const fakeTimingDevice = (
  timing: FakeTiming,
): { readonly device: GPUDevice; readonly passes: readonly LoggedPass[] } => {
  const passes: LoggedPass[] = [];
  const counts = new Map<string, { readonly all: number; readonly timed: number }>();
  let lastTimed: LoggedPass | undefined;
  let lastSubmitted: LoggedPass | undefined;
  const createBuffer = (descriptor: GPUBufferDescriptor) => ({
    size: descriptor.size,
    destroy: (): void => undefined,
    unmap: (): void => undefined,
    mapAsync: (): Promise<void> => Promise.resolve(),
    getMappedRange: (_offset = 0, length = descriptor.size): ArrayBuffer =>
      descriptor.label === "geometry-sweep-timestamps"
        ? new BigUint64Array([0n, BigInt(Math.round(lastTimed?.gpuNs ?? 0))]).buffer
        : new ArrayBuffer(length),
  });
  const createCommandEncoder = () => {
    let pass: OpenPass | undefined;
    return {
      clearBuffer: (): void => undefined,
      copyBufferToBuffer: (): void => undefined,
      resolveQuerySet: (): void => undefined,
      beginComputePass: (descriptor: GPUComputePassDescriptor) => {
        const current: OpenPass = {
          key: "",
          dispatches: 0,
          timed: descriptor.timestampWrites !== undefined,
        };
        pass = current;
        return {
          setPipeline: (pipeline: { readonly label: string }): void => {
            current.key = pipeline.label;
          },
          setBindGroup: (): void => undefined,
          dispatchWorkgroups: (): void => {
            current.dispatches += 1;
          },
          end: (): void => undefined,
        };
      },
      finish: (): FakeCommands => (pass === undefined ? {} : { pass }),
    };
  };
  const submit = (commands: readonly FakeCommands[]): void => {
    lastSubmitted = undefined;
    for (const { pass } of commands) {
      if (pass === undefined) continue;
      const count = counts.get(pass.key) ?? { all: 0, timed: 0 };
      const fake: FakePass = { ...pass, index: count.all, timedIndex: count.timed };
      counts.set(pass.key, { all: count.all + 1, timed: count.timed + (pass.timed ? 1 : 0) });
      const logged: LoggedPass = pass.timed ? { ...fake, gpuNs: timing.gpuNs(fake) } : fake;
      passes.push(logged);
      if (pass.timed) lastTimed = logged;
      lastSubmitted = logged;
    }
  };
  const device = {
    lost: new Promise<GPUDeviceLostInfo>(() => {}),
    features: new Set(["timestamp-query"]),
    destroy: (): void => undefined,
    pushErrorScope: (): void => undefined,
    popErrorScope: (): Promise<GPUError | null> => Promise.resolve(null),
    createBuffer,
    createQuerySet: () => ({ destroy: (): void => undefined }),
    createShaderModule: () => ({
      getCompilationInfo: () => Promise.resolve({ messages: [] }),
    }),
    createComputePipelineAsync: (descriptor: GPUComputePipelineDescriptor) =>
      Promise.resolve({ label: descriptor.label ?? "", getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder,
    queue: {
      submit,
      writeBuffer: (): void => undefined,
      onSubmittedWorkDone: (): Promise<void> => {
        const ms = lastSubmitted === undefined ? 0 : (timing.wallMs?.(lastSubmitted) ?? 0);
        return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();
      },
    },
  };
  // Test-only boundary: the whole DOM GPUDevice type is not reproduced (only the surface the sweeps
  // touch)
  return { device: device as unknown as GPUDevice, passes };
};

/** Runs `body` with a sweep context (unit `ns`) on a fake device of `timing`. */
const withContext = async <T>(
  timing: FakeTiming,
  body: (context: SweepContext, passes: readonly LoggedPass[]) => Promise<T>,
): Promise<T> => {
  const { device, passes } = fakeTimingDevice(timing);
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const context = await createSweepContext(gpu, "ns");
  try {
    return await body(context, passes);
  } finally {
    destroySweepContext(context);
  }
};

/** The geometry case with the smallest resources (filling them is fast on the fake device). */
const SMALLEST: SweepCase = SWEEP_CASES.map((sweepCase) => ({
  sweepCase,
  bytes: casePlan(sweepCase, 65535, false).resources.reduce((sum, spec) => sum + spec.bytes, 0),
})).sort((left, right) => left.bytes - right.bytes)[0].sweepCase;

/** Prepares the default geometry's launch of {@link SMALLEST} and runs `body` on it. */
const withDefaultLaunch = async <T>(
  context: SweepContext,
  body: (ready: ReadyLaunch) => Promise<T>,
): Promise<T> => {
  const plan = casePlan(SMALLEST, 65535, false);
  const resources = await createResources(context, plan.resources);
  try {
    const ready = await prepareLaunch(context, plan.launch(plan.defaultCandidate), resources);
    try {
      return await body(ready);
    } finally {
      ready.params.destroy();
    }
  } finally {
    for (const buffer of resources.values()) buffer.destroy();
  }
};

/** The timed passes of the log. */
const timedOf = (passes: readonly LoggedPass[]): LoggedPass[] =>
  passes.filter((pass) => pass.timed);

/** The total of timestamp differences of the timed passes. */
const gpuTotal = (passes: readonly LoggedPass[]): number =>
  timedOf(passes).reduce((sum, pass) => sum + (pass.gpuNs ?? 0), 0);

describe("the warm start", () => {
  it("waits until its last passes agree on a dispatch that speeds up (fault: 2 % faster each pass for 20 passes)", async () => {
    // Per dispatch 10 µs falling by 0.2 µs a pass to 6 µs from the 21st timed pass on; a probe that
    // reads no wall time stacks the bound, so a pass is 16384 dispatches.
    const perDispatch = (index: number): number => (index < 20 ? 10_000 - 200 * index : 6_000);
    const result = await withContext(
      { gpuNs: (pass) => pass.dispatches * perDispatch(pass.timedIndex) },
      (context) => withDefaultLaunch(context, (ready) => warmStart(context, ready)),
    );
    // Passes 21–23 are the first three that agree (6 µs each); 2.95 s of load by then.
    assertEquals(result, { stable: true, passes: 23, loadNs: SWEEP_MAX_REPS * 180_000 });
  });

  it("spends its floor of load even when its passes agree from the start (fault: a steady device)", async () => {
    const passNs = SWEEP_MAX_REPS * 1_000;
    const result = await withContext({ gpuNs: (pass) => pass.dispatches * 1_000 }, (context) =>
      withDefaultLaunch(context, (ready) => warmStart(context, ready)),
    );
    assertEquals(result.stable, true);
    assert(result.loadNs >= WARM_START_MIN_NS, `${result.loadNs} ns`);
    assert(result.loadNs - passNs < WARM_START_MIN_NS, `${result.loadNs} ns`);
    assertEquals(result.passes, Math.ceil(WARM_START_MIN_NS / passNs));
  });

  it("stops at its ceiling when its passes never agree (fault: passes alternating 10 % apart)", async () => {
    // Passes of 82 and 90 ms: the ceiling of load comes before the cut-off of passes.
    const perDispatch = (index: number): number => (index % 2 === 0 ? 5_000 : 5_500);
    const result = await withContext(
      { gpuNs: (pass) => pass.dispatches * perDispatch(pass.timedIndex) },
      (context) => withDefaultLaunch(context, (ready) => warmStart(context, ready)),
    );
    assertEquals(result.stable, false);
    assert(result.loadNs >= WARM_START_MAX_NS, `${result.loadNs} ns`);
    assert(result.loadNs - SWEEP_MAX_REPS * 5_500 < WARM_START_MAX_NS, `${result.loadNs} ns`);
    assert(result.passes < WARM_START_MAX_PASSES, `${result.passes} passes`);
  });

  it("stops after its cut-off of passes when they stay too short to add up its floor (fault: 0.1 µs per dispatch at the reps bound)", async () => {
    const result = await withContext({ gpuNs: (pass) => pass.dispatches * 100 }, (context) =>
      withDefaultLaunch(context, (ready) => warmStart(context, ready)),
    );
    assertEquals(result, {
      stable: false,
      passes: WARM_START_MAX_PASSES,
      loadNs: WARM_START_MAX_PASSES * SWEEP_MAX_REPS * 100,
    });
    assertEquals(WARM_START_MAX_PASSES, 125);
  });

  it("resizes its passes to the target when they come out short (fault: a probe that reads 25 ms per dispatch, 1 ms on the device)", async () => {
    // The probe's two passes take 25 ms per dispatch of wall time: it settles on 4 dispatches, whose
    // pass is 4 ms on the device; the warm start then stacks the 80 that fill the 80 ms target.
    const result = await withContext(
      {
        gpuNs: (pass) => pass.dispatches * 1e6,
        wallMs: (pass) => (pass.timed ? 0 : pass.dispatches * 25),
      },
      (context, passes) =>
        withDefaultLaunch(context, async (ready) => ({
          warm: await warmStart(context, ready),
          dispatches: timedOf(passes).map((pass) => pass.dispatches),
        })),
    );
    assertEquals(result.warm, { stable: true, passes: 14, loadNs: 4e6 + 13 * 80e6 });
    assertEquals(result.dispatches, [4, ...new Array<number>(13).fill(80)]);
  });
});

describe("the calibration", () => {
  it("is done again after more load when its pass comes out far below the target (fault: a probe taken while the dispatch was 25 times slower)", async () => {
    // The first probe reads 25 ms per dispatch of wall time (4 reps); the device runs it in 1 µs, so
    // the check pass is 4 µs. After the warm start the probe reads no wall time: the bound.
    let probes = 0;
    const result = await withContext(
      {
        gpuNs: (pass) => pass.dispatches * 1_000,
        wallMs: (pass) => {
          if (pass.timed) return 0;
          probes += 1;
          return probes <= 2 ? pass.dispatches * 25 : 0;
        },
      },
      (context, passes) =>
        withDefaultLaunch(context, async (ready) => ({
          reps: await calibrate(context, ready),
          load: gpuTotal(passes),
        })),
    );
    assertEquals(result.reps, SWEEP_MAX_REPS);
    assert(result.load >= WARM_START_MIN_NS, `${result.load} ns of load before the second probe`);
  });

  it("is accepted at once when its pass fills the target (control: the device runs the dispatch as the probe read it)", async () => {
    const result = await withContext(
      {
        gpuNs: (pass) => pass.dispatches * 25e6,
        wallMs: (pass) => (pass.timed ? 0 : pass.dispatches * 25),
      },
      (context, passes) =>
        withDefaultLaunch(context, async (ready) => ({
          reps: await calibrate(context, ready),
          timed: timedOf(passes).length,
        })),
    );
    // One pass checked the 4 reps (100 ms ≥ a quarter of the target); no warm start followed.
    assertEquals(result, { reps: 4, timed: 1 });
  });
});

describe("the geometry sweep", () => {
  it("starts a case warm on its default geometry: the warm start's floor of load comes before any candidate runs", async () => {
    const plan = casePlan(SMALLEST, 65535, false);
    const defaultKey = plan.launch(plan.defaultCandidate).key;
    const passes = await withContext(
      { gpuNs: (pass) => pass.dispatches * 1_000 },
      async (context, log) => {
        await runSweep(context, [SMALLEST], {
          rounds: 1,
          candidateSet: "quick",
          timestampUnit: "ns",
        });
        return log;
      },
    );
    const firstOther = passes.findIndex(
      (pass) => pass.key !== defaultKey && plan.prelude.every(({ key }) => key !== pass.key),
    );
    assert(firstOther > 0, "no candidate geometry ran");
    const before = passes.slice(0, firstOther).filter((pass) => pass.key === defaultKey);
    // The warm start's floor, then the default's own warm-up and round
    assert(gpuTotal(before) >= WARM_START_MIN_NS, `${gpuTotal(before)} ns`);
  });
});

/** A small GEMV shape: 100 columns, k = 1024 (as the GPU test's i8 case). */
const GEMV_CASE: GemvCase = {
  id: "test-i8-n100-k1024",
  storage: "i8",
  n: 100,
  k: 1024,
  rows: [],
  source: "test",
};

const GATES: GemvGates = { subgroups: false, maxComputeWorkgroupsPerDimension: 65535 };

/** The M = 1 plan of the baseline and the identical candidates `names` (no oracle needed). */
const planOf = (names: readonly string[]): GemvMeasurementPlan => {
  const candidates = names.map((name): GemvCandidate => {
    const found = gemvCandidates(GEMV_CASE, 1, "full", GATES).find((one) => one.name === name);
    if (found === undefined) throw new Error(`no candidate ${name}`);
    return found;
  });
  const baseline = gemvBaseline(GEMV_CASE, 1);
  return {
    m: 1,
    baseline,
    candidates,
    profile: { exact: baseline.name, reorder: gemvProfileChoice(GEMV_CASE, 1, "reorder").name },
  };
};

/** The pipeline key of a kernel of the plan. */
const keyOf = (candidate: GemvCandidate): string =>
  gemvLaunch(gemvProgram(GEMV_CASE, 1, candidate.choice), "out", 65535).key;

/** Sweeps one plan of {@link GEMV_CASE} on a fake device of `timing`. */
const sweepPlan = (
  timing: FakeTiming,
  plan: GemvMeasurementPlan,
  rounds: number,
  signal?: AbortSignal,
) =>
  withContext(timing, async (context, passes) => ({
    result: await sweepGemvCase(
      context,
      GEMV_CASE,
      [plan],
      { rounds, tolerance: opTolerance("linear") },
      signal === undefined ? {} : { signal },
    ),
    passes,
  }));

describe("the GEMV sweep's row count", () => {
  const steady: FakeTiming = { gpuNs: (pass) => pass.dispatches * 5_000 };

  it("starts warm on the baseline: it spends the warm start's floor of load before any other kernel runs", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const { passes } = await sweepPlan(steady, plan, 2);
    const baselineKey = keyOf(plan.baseline);
    const firstOther = passes.findIndex((pass) => pass.key !== baselineKey);
    assert(firstOther > 0, "no other kernel ran");
    assert(gpuTotal(passes.slice(0, firstOther)) >= WARM_START_MIN_NS);
  });

  it("times every kernel once per round, the baseline included, in an order that rotates by an even step (4 kernels, 2 rounds)", async () => {
    const plan = planOf(["c16u2", "c64u8", "c16u8"]);
    const { result, passes } = await sweepPlan(steady, plan, 2);
    const keys = [plan.baseline, ...plan.candidates].map(keyOf);
    const roundsPasses = timedOf(passes).slice(-2 * keys.length);
    // Round 1 starts at kernel 0, round 2 at kernel ⌊1 · 4 / 2⌋ = 2.
    assertEquals(
      roundsPasses.map((pass) => pass.key),
      [keys[0], keys[1], keys[2], keys[3], keys[2], keys[3], keys[0], keys[1]],
    );
    for (const row of result.rows) {
      assertEquals(row.error, undefined, row.candidate);
      assertEquals(row.rounds?.length, 2, row.candidate);
      assertEquals(row.identicalToBaseline ?? true, true, row.candidate);
    }
  });

  it("starts each round one kernel further when there are more rounds than kernels (3 kernels, 5 rounds)", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const { passes } = await sweepPlan(steady, plan, 5);
    const [b, c1, c2] = [plan.baseline, ...plan.candidates].map(keyOf);
    assertEquals(
      timedOf(passes)
        .slice(-15)
        .map((pass) => pass.key),
      [b, c1, c2, c1, c2, b, c2, b, c1, b, c1, c2, c1, c2, b],
    );
  });

  it("records the spread of the baseline's rounds (fault: its passes alternate 1 % apart)", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const baselineKey = keyOf(plan.baseline);
    const { result } = await sweepPlan(
      {
        gpuNs: (pass) =>
          pass.dispatches * (pass.key === baselineKey && pass.timedIndex % 2 === 1 ? 5_050 : 5_000),
      },
      plan,
      3,
    );
    const [summary] = result.summaries;
    assert(
      summary.baselineSpread !== undefined && Math.abs(summary.baselineSpread - 1.01) < 1e-9,
      `spread ${summary.baselineSpread}`,
    );
    assertEquals(
      result.rows.map(({ candidate, isBaseline }) => [candidate, isBaseline]),
      [
        [plan.baseline.name, true],
        ["c16u2", false],
        ["c64u8", false],
      ],
    );
  });

  it("leaves no row and no summary for the row count a cancel stopped in its rounds", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const controller = new AbortController();
    const candidateKey = keyOf(plan.candidates[0]);
    const { result } = await sweepPlan(
      {
        gpuNs: (pass) => {
          if (pass.key === candidateKey && pass.timed) controller.abort();
          return pass.dispatches * 5_000;
        },
      },
      plan,
      3,
      controller.signal,
    );
    assertEquals(result, { rows: [], summaries: [], aborted: true });
  });
});
