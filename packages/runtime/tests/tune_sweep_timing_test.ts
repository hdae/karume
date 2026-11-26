// The sweeps' timing protocol on a fake device (no GPU needed): the warm start and the calibration
// check of src/tune/harness.ts (`warmStart`, `calibrate`), the geometry sweep's case starting warm,
// and the GEMV sweep's row count (src/tune/gemv-sweep.ts) — its warm start on the baseline, its
// interleaved rounds and the spread of the baseline's rounds.
//
// The quick look of both sweeps: the kernels within the margin of their class's bar go on, the rest
// are screened rows (fault: a candidate at ×1.3 the fastest, a geometry at ×1.5 the default); a
// margin of 0 times everything (fault: a candidate at ×3); the references go on whatever their time
// (fault: the baseline and the default's choice at reorder at ×3); a group the derivation reads only
// for a regression weighs the reorder class against the default's choice (fault: a candidate at
// ×2.25 the fastest but faster than that choice); a kernel that goes on calibrates from its quick
// look's probe (fault: a calibration that probes again would add untimed passes); the quick looks
// follow the group's shuffled order.
//
// The GEMV sweep's skipping at M = 4 and 8: a reorder candidate its M = 1 group leaves unadoptable is
// not measured there (faults: a candidate ×1.05 the default's choice at M = 1, one screened at M = 1);
// a run narrowed to M = 4 skips nothing. With the explore set's rule, M = 4 prepares only the three
// fastest rows of M = 1 and the row-blocked form of their lanes (fault: the full set's rule).
//
// The GEMV sweep's progress steps: each phase of a row count counts from 0 to its total (fault: a
// kernel screened, so the calibration counts one kernel fewer than the quick look).
//
// The fake device's passes take the time the test gives them: a timed pass's timestamp difference
// (`gpuNs`, by the pass's pipeline key, dispatch count and index), and the wall clock a pass takes to
// complete (`wallMs` — the probe reads only that), on the fake device's own clock, which the sweep
// context is given: no real timer runs, so a run under load reads the same times. GpuContext is the
// real one; only the device is replaced (as in tune_harness_callbacks_test.ts).
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
import type { Tolerance } from "../src/reference/allclose.ts";
import { opTolerance } from "../src/reference/op-tolerance.ts";
import { SWEEP_CASES, type SweepCase } from "../src/tune/cases.ts";
import {
  type GemvCandidate,
  type GemvGates,
  gemvBaseline,
  gemvCandidates,
  gemvExploreCarriedFromM1,
  type GemvMeasurementPlan,
  gemvProfileChoice,
  gemvProgram,
} from "../src/tune/gemv-candidates.ts";
import { GEMV_CASES, type GemvCase } from "../src/tune/gemv-cases.ts";
import { gemvKernelId } from "../src/tune/gemv-fingerprint.ts";
import type { GemvSweepRow } from "../src/tune/gemv-report.ts";
import { gemvLaunch, type GemvSweepStep, sweepGemvCase } from "../src/tune/gemv-sweep.ts";
import {
  calibrate,
  candidatesFor,
  casePlan,
  createResources,
  createSweepContext,
  destroySweepContext,
  prepareLaunch,
  quickLookOrder,
  type ReadyLaunch,
  runSweep,
  sweepCase,
  SWEEP_MAX_REPS,
  type SweepContext,
  WARM_START_MAX_PASSES,
  warmStart,
} from "../src/tune/harness.ts";
import {
  QUICK_LOOK_MARGIN,
  WARM_START_MAX_NS,
  WARM_START_MIN_NS,
} from "../src/tune/measurement.ts";

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
  /** The time a pass takes to complete on the fake device's clock (ms; absent: none). */
  readonly wallMs?: (pass: FakePass) => number;
  /** The pipeline keys whose pipeline cannot be made (absent: none). */
  readonly failingPipelines?: ReadonlySet<string>;
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
): {
  readonly device: GPUDevice;
  readonly passes: readonly LoggedPass[];
  /** The fake device's clock (ms): it moves only by the `wallMs` of the passes that completed. */
  readonly now: () => number;
} => {
  const passes: LoggedPass[] = [];
  let clockMs = 0;
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
      timing.failingPipelines?.has(descriptor.label ?? "") === true
        ? Promise.reject(new Error(`no pipeline for ${descriptor.label}`))
        : Promise.resolve({ label: descriptor.label ?? "", getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createCommandEncoder,
    queue: {
      submit,
      writeBuffer: (): void => undefined,
      onSubmittedWorkDone: (): Promise<void> => {
        clockMs += lastSubmitted === undefined ? 0 : (timing.wallMs?.(lastSubmitted) ?? 0);
        return Promise.resolve();
      },
    },
  };
  // Test-only boundary: the whole DOM GPUDevice type is not reproduced (only the surface the sweeps
  // touch)
  return { device: device as unknown as GPUDevice, passes, now: () => clockMs };
};

/** Runs `body` with a sweep context (unit `ns`) on a fake device of `timing`. */
const withContext = async <T>(
  timing: FakeTiming,
  body: (context: SweepContext, passes: readonly LoggedPass[]) => Promise<T>,
): Promise<T> => {
  const { device, passes, now } = fakeTimingDevice(timing);
  const gpu = new GpuContext(device, readAdapterInfo({}), LIMITS, new Set());
  const context = await createSweepContext(gpu, "ns", now);
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
          quickLookMargin: QUICK_LOOK_MARGIN,
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
  margin = QUICK_LOOK_MARGIN,
) =>
  withContext(timing, async (context, passes) => ({
    result: await sweepGemvCase(
      context,
      GEMV_CASE,
      [plan],
      { rounds, tolerance: opTolerance("linear"), quickLookMargin: margin },
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

/** E2B's per_layer_model_projection: its default's choice at reorder is parallel-l4 (a parallel row). */
const PLE_CASE: GemvCase = (() => {
  const found = GEMV_CASES.find(({ id }) => id === "i4g32-n8960-k1536");
  if (found === undefined) throw new Error("no case i4g32-n8960-k1536");
  return found;
})();

/** The plan of `gemvCase` at `m` rows with the candidates `names` of the full set. */
const namedPlan = (
  gemvCase: GemvCase,
  m: number,
  names: readonly string[],
): GemvMeasurementPlan => {
  const all = gemvCandidates(gemvCase, m, "full", GATES);
  const baseline = gemvBaseline(gemvCase, m);
  return {
    m,
    baseline,
    candidates: names.map((name) => {
      const found = all.find((one) => one.name === name);
      if (found === undefined) throw new Error(`${gemvCase.id} M=${m}: no candidate ${name}`);
      return found;
    }),
    profile: { exact: baseline.name, reorder: gemvProfileChoice(gemvCase, m, "reorder").name },
  };
};

/** The pipeline key of a kernel of a case at a row count. */
const keyAt = (gemvCase: GemvCase, m: number, candidate: GemvCandidate): string =>
  gemvLaunch(gemvProgram(gemvCase, m, candidate.choice), "out", 65535).key;

/** Each kernel's time per dispatch by its key (5 µs where `perKernel` names none). */
const byKey =
  (times: ReadonlyMap<string, number>): FakeTiming["gpuNs"] =>
  (pass) =>
    pass.dispatches * (times.get(pass.key) ?? 5_000);

/** Sweeps one plan of any case on a fake device. */
const sweepCasePlan = (
  gemvCase: GemvCase,
  timing: FakeTiming,
  plan: GemvMeasurementPlan,
  margin: number,
) =>
  withContext(timing, async (context, passes) => ({
    result: await sweepGemvCase(context, gemvCase, [plan], {
      rounds: 2,
      tolerance: opTolerance("linear"),
      quickLookMargin: margin,
    }),
    passes,
  }));

const rowOf = (rows: readonly GemvSweepRow[], name: string): GemvSweepRow => {
  const found = rows.find((row) => row.candidate === name);
  if (found === undefined) throw new Error(`no row ${name}`);
  return found;
};

describe("the GEMV sweep's quick look", () => {
  it("lets the kernels within the margin of their class's fastest go on and screens the rest: no rounds, no check, its quick time (fault: c64u8 at ×1.3 the fastest)", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const times = new Map([[keyOf(plan.candidates[1]), 6_500]]);
    const { result } = await sweepPlan({ gpuNs: byKey(times) }, plan, 2);
    const screened = rowOf(result.rows, "c64u8");
    assertEquals(screened.screened, true);
    assertEquals(screened.quickPerDispatch, 6_500);
    assertEquals(
      [screened.rounds, screened.perDispatch, screened.identicalToBaseline, screened.error],
      [undefined, undefined, undefined, undefined],
    );
    assertEquals(screened.kernel?.length, 16);
    const timed = rowOf(result.rows, "c16u2");
    assertEquals(
      [timed.screened, timed.rounds?.length, timed.quickPerDispatch],
      [undefined, 2, 5_000],
    );
    assertEquals(result.summaries[0]?.screened, 1);
  });

  it("times every kernel at a margin of 0 (fault: c64u8 at ×3 the fastest)", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const times = new Map([[keyOf(plan.candidates[1]), 15_000]]);
    const { result } = await sweepPlan({ gpuNs: byKey(times) }, plan, 2, undefined, 0);
    for (const row of result.rows) {
      assertEquals(
        [row.screened, row.rounds?.length, row.quickPerDispatch],
        [undefined, 2, undefined],
      );
    }
    assertEquals(result.summaries[0]?.screened, 0);
  });

  it("lets the references go on whatever their time (fault: the baseline and the default's choice at reorder 3 times the fastest of their classes)", async () => {
    const plan = namedPlan(PLE_CASE, 1, ["c64u8", "parallel-l4", "parallel-l8", "staged-l32"]);
    assertEquals(plan.profile.reorder, "parallel-l4");
    const slow = [plan.baseline, plan.candidates[1]].map((candidate) =>
      keyAt(PLE_CASE, 1, candidate),
    );
    const times = new Map(slow.map((key) => [key, 15_000]));
    const { result } = await sweepCasePlan(
      PLE_CASE,
      { gpuNs: byKey(times) },
      plan,
      QUICK_LOOK_MARGIN,
    );
    for (const name of [plan.baseline.name, "parallel-l4"]) {
      const row = rowOf(result.rows, name);
      assertEquals([row.screened, row.rounds?.length], [undefined, 2], name);
    }
    for (const name of ["c64u8", "parallel-l8", "staged-l32"]) {
      assertEquals(rowOf(result.rows, name).screened, undefined, name);
    }
  });

  it("weighs the reorder class against the default's choice where the derivation only checks a regression (M = 4: parallel-l8 at ×2.25 the fastest but faster than the default's choice goes on; staged-l2 at ×1.3 parallel-l4 does not)", async () => {
    // The default's choice at M = 4: the row-blocked kernel of the case's row
    const reference = "staged-l4r4";
    const plan = namedPlan(PLE_CASE, 4, [reference, "parallel-l8", "staged-l2", "staged-l32"]);
    assertEquals(plan.profile.reorder, reference);
    const key = (name: string): string => {
      const candidate = plan.candidates.find((one) => one.name === name);
      if (candidate === undefined) throw new Error(name);
      return keyAt(PLE_CASE, 4, candidate);
    };
    const times = new Map([
      [key(reference), 10_000],
      [key("parallel-l8"), 9_000],
      [key("staged-l2"), 13_000],
      [key("staged-l32"), 4_000],
    ]);
    const { result } = await sweepCasePlan(
      PLE_CASE,
      { gpuNs: byKey(times) },
      plan,
      QUICK_LOOK_MARGIN,
    );
    assertEquals(rowOf(result.rows, "parallel-l8").screened, undefined);
    assertEquals(rowOf(result.rows, "staged-l2").screened, true);
    assertEquals(rowOf(result.rows, "staged-l32").screened, undefined);
  });

  it("calibrates a kernel that goes on from its quick look's probe: no second probe (fault: a calibration that probes again)", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const { passes } = await sweepPlan({ gpuNs: byKey(new Map()) }, plan, 2);
    const key = keyOf(plan.candidates[0]);
    // The quick look's probe is timed; after it the kernel's one untimed pass is its run from a
    // zeroed output (a probe of its own would add 15 untimed passes of 1 to 16384 dispatches)
    assertEquals(
      passes.filter((pass) => pass.key === key && !pass.timed).map((pass) => pass.dispatches),
      [1],
    );
  });

  it("takes the quick looks in the group's shuffled order, not the plan's", async () => {
    const plan = planOf(["c16u2", "c64u8", "c16u8", "c64u2", "c32u8"]);
    const { passes } = await sweepPlan({ gpuNs: byKey(new Map()) }, plan, 2);
    const kernels = [plan.baseline, ...plan.candidates];
    const looked = timedOf(passes)
      .filter((pass) => pass.dispatches === 1)
      .map((pass) => pass.key);
    const expected = quickLookOrder(kernels, `${GEMV_CASE.id} M=1`).map(keyOf);
    assertEquals(looked, expected);
    assert(
      expected.some((key, index) => key !== kernels.map(keyOf)[index]),
      "the shuffle kept the plan's order",
    );
  });
});

describe("the GEMV sweep's skipping at M = 4 and 8", () => {
  /** A band so wide that the fake device's zero outputs hold it: every check passes, times decide. */
  const WIDE = { atol: 1e30, rtol: 0 };
  /**
   * The default's choice at reorder at M = 1 and three candidates of the class. At M = 4 the row's
   * choice is its row-blocked kernel, which these plans do not name; the kernel of M = 1 is measured
   * there whatever its own row says.
   */
  const REORDER = ["parallel-l4", "parallel-l8", "staged-l32", "staged-l2"];
  const identical = gemvCandidates(PLE_CASE, 4, "full", GATES).find(
    (candidate) => candidate.class === "identical",
  );
  if (identical === undefined) throw new Error(`${PLE_CASE.id} M=4: no identical candidate`);
  const planAt = (m: number): GemvMeasurementPlan =>
    namedPlan(PLE_CASE, m, m === 4 ? [identical.name, ...REORDER] : REORDER);
  const keyOfName = (m: number, name: string): string => {
    const candidate = planAt(m).candidates.find((one) => one.name === name);
    if (candidate === undefined) throw new Error(`M=${m}: no candidate ${name}`);
    return keyAt(PLE_CASE, m, candidate);
  };
  // A parallel kernel's key does not hold M, so these are its times at both row counts: at M = 1
  // parallel-l8 is ×1.25 the default's choice, staged-l32 ×1.05, and staged-l2 is screened (×2.5 the
  // fastest of its class)
  const times = new Map([
    [keyOfName(1, "parallel-l4"), 10_000],
    [keyOfName(1, "parallel-l8"), 8_000],
    [keyOfName(1, "staged-l32"), 9_500],
    [keyOfName(1, "staged-l2"), 20_000],
  ]);
  const sweep = (
    plans: readonly GemvMeasurementPlan[],
    failingPipelines: ReadonlySet<string> = new Set(),
  ) => {
    const progress: string[] = [];
    return withContext({ gpuNs: byKey(times), failingPipelines }, async (context, passes) => ({
      result: await sweepGemvCase(
        context,
        PLE_CASE,
        plans,
        { rounds: 2, tolerance: WIDE, quickLookMargin: QUICK_LOOK_MARGIN },
        { onProgress: (message) => progress.push(message) },
      ),
      passes,
      progress,
    }));
  };
  const rowAt = (rows: readonly GemvSweepRow[], m: number, name: string): GemvSweepRow => {
    const found = rows.find((row) => row.m === m && row.candidate === name);
    if (found === undefined) throw new Error(`no row ${name} at M=${m}`);
    return found;
  };

  it("measures at M = 4 only the reorder candidates its M = 1 group leaves adoptable: the others get a skipped row with their kernel and launch, and no pass", async () => {
    const { result, passes, progress } = await sweep([planAt(1), planAt(4)]);
    assertEquals(
      ["parallel-l8", "staged-l32"].map((name) => rowAt(result.rows, 1, name).withinBand),
      [true, true],
    );
    assertEquals(rowAt(result.rows, 1, "staged-l2").screened, true);
    for (const name of ["staged-l32", "staged-l2"]) {
      const row = rowAt(result.rows, 4, name);
      const candidate = planAt(4).candidates.find((one) => one.name === name);
      assert(candidate !== undefined);
      assertEquals(row.skipped, true, name);
      assertEquals(row.kernel, gemvKernelId(PLE_CASE, 4, candidate.choice), name);
      assertEquals(row.keys, [keyOfName(4, name)], name);
      assert(row.workgroups !== undefined, name);
      assertEquals(
        [
          row.isBaseline,
          row.screened,
          row.quickPerDispatch,
          row.reps,
          row.rounds,
          row.perDispatch,
          row.withinBand,
          row.speedupVsBaseline,
          row.error,
        ],
        [
          false,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
        ],
        name,
      );
    }
    for (const name of [planAt(4).baseline.name, identical.name, "parallel-l4", "parallel-l8"]) {
      const row = rowAt(result.rows, 4, name);
      assertEquals([row.skipped, row.rounds?.length], [undefined, 2], name);
    }
    assertEquals(
      result.summaries.map(({ m, screened, skipped }) => [m, screened, skipped]),
      [
        [1, 1, 0],
        [4, 0, 2],
      ],
    );
    // No pipeline is prepared for them at M = 4, and from its warm start on none of their passes runs
    assert(
      progress.includes(
        `${PLE_CASE.id} M=4: preparing 4 kernels (2 skipped: not adoptable from M = 1)`,
      ),
      progress.join("\n"),
    );
    const fourStarts = passes.findIndex(
      (pass) => pass.key === keyAt(PLE_CASE, 4, planAt(4).baseline),
    );
    assert(fourStarts > 0, "M = 4 never started");
    const atFour = passes.slice(fourStarts).map((pass) => pass.key);
    for (const name of ["staged-l32", "staged-l2"]) {
      assert(!atFour.includes(keyOfName(4, name)), name);
    }
    assert(atFour.includes(keyOfName(4, "parallel-l8")), "parallel-l8 did not run at M = 4");
  });

  it("skips nothing in a run narrowed to M = 4: without its M = 1 group the quick look alone decides", async () => {
    const { result } = await sweep([planAt(4)]);
    assert(
      result.rows.every((row) => row.skipped === undefined),
      "a row was skipped",
    );
    assertEquals(
      result.summaries.map(({ m, skipped }) => [m, skipped]),
      [[4, 0]],
    );
    assertEquals(rowAt(result.rows, 4, "staged-l32").rounds?.length, 2);
    // staged-l2 at ×2 the default's choice: screened by this group's quick look, not skipped
    assertEquals(rowAt(result.rows, 4, "staged-l2").screened, true);
  });

  it("skips nothing at M = 4 when its M = 1 baseline was not timed: that group reads nothing (fault: the baseline's pipeline cannot be made at M = 1)", async () => {
    const baselineKey = keyAt(PLE_CASE, 1, planAt(1).baseline);
    const { result } = await sweep([planAt(1), planAt(4)], new Set([baselineKey]));
    const baseline = result.rows.find((row) => row.m === 1 && row.isBaseline);
    assert(baseline?.error !== undefined, "the M = 1 baseline did not fail");
    assertEquals(
      result.summaries.map(({ m, skipped }) => [m, skipped]),
      [
        [1, 0],
        [4, 0],
      ],
    );
    // staged-l32 (×1.05 at M = 1) is timed at M = 4: M = 1 ruled nothing out
    assertEquals(rowAt(result.rows, 4, "staged-l32").rounds?.length, 2);
  });
});

describe("the GEMV sweep's progress steps", () => {
  /** The steps of one sweep of `plans`, and its result. */
  const stepsOf = (
    gemvCase: GemvCase,
    plans: readonly GemvMeasurementPlan[],
    timing: FakeTiming,
    settings: {
      readonly margin: number;
      readonly tolerance?: Tolerance;
      readonly explore?: boolean;
    },
  ) =>
    withContext(timing, async (context) => {
      const steps: GemvSweepStep[] = [];
      const result = await sweepGemvCase(
        context,
        gemvCase,
        plans,
        {
          rounds: 2,
          tolerance: settings.tolerance ?? opTolerance("linear"),
          quickLookMargin: settings.margin,
          ...(settings.explore === true ? { carriedFromM1: gemvExploreCarriedFromM1 } : {}),
        },
        { onStep: (step) => steps.push(step) },
      );
      return { steps, result };
    });

  /** Per row count, each phase's total and the counts it reported, in the order they came. */
  const phasesOf = (steps: readonly GemvSweepStep[], m: number) =>
    steps
      .filter((step) => step.m === m)
      .reduce<{ phase: string; total: number; done: number[] }[]>((phases, step) => {
        const last = phases.at(-1);
        if (last !== undefined && last.phase === step.phase && last.total === step.total) {
          last.done.push(step.done);
        } else {
          phases.push({ phase: step.phase, total: step.total, done: [step.done] });
        }
        return phases;
      }, []);

  const upTo = (total: number): number[] => Array.from({ length: total + 1 }, (_, index) => index);

  it("counts each phase of a row count from 0 to its total: the kernels prepared and looked at, the warm start, the kernels that went on, every pass of the rounds (fault: c64u8 screened at ×1.3)", async () => {
    const plan = planOf(["c16u2", "c64u8", "c16u8"]);
    const times = new Map([[keyOf(plan.candidates[1]), 6_500]]);
    const { steps, result } = await stepsOf(
      GEMV_CASE,
      [plan],
      { gpuNs: byKey(times) },
      {
        margin: QUICK_LOOK_MARGIN,
      },
    );
    assertEquals(rowOf(result.rows, "c64u8").screened, true);
    assertEquals(phasesOf(steps, 1), [
      { phase: "prepare", total: 4, done: upTo(4) },
      { phase: "warm-start", total: 1, done: upTo(1) },
      { phase: "quick-look", total: 4, done: upTo(4) },
      { phase: "calibrate", total: 3, done: upTo(3) },
      { phase: "rounds", total: 6, done: upTo(6) },
    ]);
    assert(steps.every((step) => step.caseId === GEMV_CASE.id));
  });

  it("reports no quick look at a margin of 0, and calibrates and times every kernel", async () => {
    const plan = planOf(["c16u2", "c64u8"]);
    const { steps } = await stepsOf(GEMV_CASE, [plan], { gpuNs: byKey(new Map()) }, { margin: 0 });
    assertEquals(
      phasesOf(steps, 1).map(({ phase, total }) => [phase, total]),
      [
        ["prepare", 3],
        ["warm-start", 1],
        ["calibrate", 3],
        ["rounds", 6],
      ],
    );
  });

  it("prepares at M = 4 only the kernels its M = 1 group carries: with the explore set's rule, the three fastest rows and the row-blocked form of their lanes (fault: the full set's rule)", async () => {
    // At M = 1 every row is adoptable over the default's parallel-l4 (×1.25 to ×2); parallel-l16
    // is the fourth fastest. A parallel kernel's key does not hold M: the same times at M = 4.
    const named = ["parallel-l8", "staged-l8", "staged-l16", "parallel-l16"];
    const one = namedPlan(PLE_CASE, 1, ["parallel-l4", ...named]);
    const four = namedPlan(PLE_CASE, 4, [...named, "staged-l8r4"]);
    const keyIn = (plan: GemvMeasurementPlan, name: string): string => {
      const candidate = plan.candidates.find((candidate) => candidate.name === name);
      if (candidate === undefined) throw new Error(name);
      return keyAt(PLE_CASE, plan.m, candidate);
    };
    const times = new Map([
      [keyIn(one, "parallel-l4"), 10_000],
      [keyIn(one, "parallel-l8"), 5_000],
      [keyIn(one, "staged-l8"), 6_000],
      [keyIn(one, "staged-l16"), 7_000],
      [keyIn(one, "parallel-l16"), 8_000],
    ]);
    const settings = { margin: 0, tolerance: { atol: 1e30, rtol: 0 } };
    const explore = await stepsOf(
      PLE_CASE,
      [one, four],
      { gpuNs: byKey(times) },
      {
        ...settings,
        explore: true,
      },
    );
    const rowAt4 = (rows: readonly GemvSweepRow[], name: string) =>
      rows.find((row) => row.m === 4 && row.candidate === name);
    assertEquals(rowAt4(explore.result.rows, "parallel-l16")?.skipped, true);
    for (const name of ["parallel-l8", "staged-l8", "staged-l16", "staged-l8r4"]) {
      assertEquals(rowAt4(explore.result.rows, name)?.rounds?.length, 2, name);
    }
    assertEquals(
      explore.result.summaries.map(({ m, skipped }) => [m, skipped]),
      [
        [1, 0],
        [4, 1],
      ],
    );
    // The baseline and the four carried kernels
    assertEquals(phasesOf(explore.steps, 4)[0], { phase: "prepare", total: 5, done: upTo(5) });
    // The control: the full set's rule carries the fourth row too
    const full = await stepsOf(PLE_CASE, [one, four], { gpuNs: byKey(times) }, settings);
    assertEquals(rowAt4(full.result.rows, "parallel-l16")?.skipped, undefined);
    assertEquals(phasesOf(full.steps, 4)[0]?.total, 6);
  });
});

describe("the geometry sweep's quick look", () => {
  const plan = casePlan(SMALLEST, 65535, false);
  const [near, far] = candidatesFor(SMALLEST, "quick").filter(
    (candidate) => candidate.name !== plan.defaultCandidate.name,
  );
  const times = new Map([
    [plan.launch(plan.defaultCandidate).key, 5_000],
    [plan.launch(near).key, 5_500],
    [plan.launch(far).key, 7_500],
  ]);
  const sweep = (margin: number) =>
    withContext({ gpuNs: byKey(times) }, (context) =>
      sweepCase(context, SMALLEST, [near, far], { rounds: 2, quickLookMargin: margin }),
    );

  it("times the default and a candidate within the margin of the default's quick time, and screens one beyond it (fault: a geometry at ×1.5 the default)", async () => {
    const { rows, summary } = await sweep(QUICK_LOOK_MARGIN);
    const [base, kept, screened] = rows;
    assertEquals([base.isDefault, base.rounds?.length], [true, 2]);
    assertEquals([kept.geometry, kept.screened, kept.rounds?.length], [near.name, undefined, 2]);
    assertEquals(kept.quickPerDispatch, 5_500);
    assertEquals(
      [screened.geometry, screened.screened, screened.quickPerDispatch],
      [far.name, true, 7_500],
    );
    assertEquals(
      [
        screened.rounds,
        screened.outputSha256,
        screened.identicalToDefault,
        screened.speedupVsDefault,
      ],
      [undefined, undefined, undefined, undefined],
    );
    assertEquals(summary.screened, 1);
  });

  it("times every geometry at a margin of 0", async () => {
    const { rows, summary } = await sweep(0);
    for (const row of rows) {
      assertEquals(
        [row.screened, row.rounds?.length, row.quickPerDispatch],
        [undefined, 2, undefined],
      );
    }
    assertEquals(summary.screened, 0);
  });
});
