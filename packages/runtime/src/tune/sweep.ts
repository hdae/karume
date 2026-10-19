/**
 * The sweep entry point (`runGeometrySweep` of the public surface `@karume/runtime/tune` — ADR 0117
 * decision 2).
 *
 * It acquires its own dedicated device, sweeps, and discards it when done. It takes no caller
 * `GpuContext` so that "the sweep and inference share a device" cannot be expressed in the types
 * (decision 9's MUST kept by structure). The Deno CLI (`tools/geometry-sweep/main.ts`, removed in
 * 0.15) and the GPU lab's sweep tab (`tools/gpu-lab`, removed in 0.15) were shells calling this
 * too; re-acquiring the device, measuring and assembling the record live only here (kept
 * separately, two records would silently drift apart).
 *
 * MUST: never called from the acquire / Session path — the runtime does not measure geometries at
 * run time (ADR 0022 addendum's ban on autotuning, ADR 0117 decision 9). Measuring happens only
 * when the user calls this explicitly.
 */
import { acquireGpu } from "../gpu/device.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../kernels/geometry-profile.ts";
import { SWEEP_CASES, SWEEP_OPS, type SweepCase, type SweepOp } from "./cases.ts";
import {
  CANDIDATE_SETS,
  type CandidateSet,
  DEFAULT_CANDIDATE_SET,
  isCandidateSet,
} from "./geometries.ts";
import {
  geometryProfileKernelsId,
  sweepCandidateKernelsId,
  sweepCaseSetId,
} from "./fingerprint.ts";
import { createSweepContext, destroySweepContext, runSweep, SWEEP_MAX_REPS } from "./harness.ts";
import { ROUNDS, TARGET_PASS_MS, WARMUP_MIN_RUNS, WARMUP_NS } from "./measurement.ts";
import {
  type CaseSummary,
  type Report,
  REPORT_FORMAT,
  type ReportAdapter,
  roundsLookQuantized,
  type SweepRow,
  type TimingUnit,
  WALL_TIMING_NOTE,
} from "./report.ts";

/**
 * The feature name looked up in the adapter's list (the one `acquireGpu({ gpuTiming: true })`
 * requests).
 */
const TIMESTAMP_QUERY: GPUFeatureName = "timestamp-query";

/**
 * Sweep progress (passed to {@link GeometrySweepOptions.onProgress} — per case / geometry).
 *
 * - `started` — the device is acquired and the measurement context exists (unit, dp4a and case
 *   count are fixed).
 * - `status` — what is being measured now ("<case>: <geometry> (i/n)" and the like — one line for
 *   display).
 * - `row` — one (case, geometry) row is final. A case's first row is the default geometry
 *   (`isDefault`); the following rows carry the ratio to the default (`speedupVsDefault`) and the
 *   output match (`identicalToDefault`).
 * - `case` — the case is fully measured (it carries `defaultRepeat.driftRatio`, the ratio of the
 *   default re-measured at the end).
 * - `deviceLost` — the device was lost (the remaining rows become failure rows).
 */
export type GeometrySweepProgress =
  | {
      readonly kind: "started";
      readonly adapter: ReportAdapter;
      readonly unit: TimingUnit;
      readonly dp4a: boolean;
      readonly caseCount: number;
    }
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "row"; readonly row: SweepRow }
  | { readonly kind: "case"; readonly summary: CaseSummary }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

/** Options of {@link runGeometrySweep} (all optional). */
export type GeometrySweepOptions = {
  /** The `GPU` instance to use (default `navigator.gpu` — same as `acquireGpu({ gpu })`). */
  readonly gpu?: GPU;
  /** Passed to `requestAdapter` (`powerPreference` etc. — same as `acquireGpu({ adapter })`). */
  readonly adapter?: GPURequestAdapterOptions;
  /** The candidate set (default `quick+`). */
  readonly candidateSet?: CandidateSet;
  /** The op families to sweep (default: all families). */
  readonly ops?: readonly SweepOp[];
  /**
   * Narrows by case id (default: every case of `ops` — the CLI's `--case`; the CLI was removed in
   * 0.15).
   */
  readonly cases?: readonly string[];
  /**
   * The number of measurement rounds per geometry (default 5 — the representative value is their
   * min).
   */
  readonly rounds?: number;
  /**
   * Measures with GPU timestamps when the adapter lists `timestamp-query` (default true). With
   * `false`, or on an adapter that does not list it, measures by wall clock (unit `wall` — the
   * record can be kept, but the generator does not build a table from it).
   */
  readonly timestamps?: boolean;
  /**
   * Abort (stops once the current geometry is measured — the record holds the rows measured so
   * far).
   */
  readonly signal?: AbortSignal;
  /** Progress notifications ({@link GeometrySweepProgress}). */
  readonly onProgress?: (progress: GeometrySweepProgress) => void;
};

/** The sweep settings with options checked and defaults filled in, and the chosen cases. */
type SweepPlan = {
  readonly candidateSet: CandidateSet;
  readonly ops: readonly SweepOp[];
  readonly caseIds: readonly string[];
  readonly rounds: number;
  readonly cases: readonly SweepCase[];
};

const isSweepOp = (value: string): value is SweepOp =>
  (SWEEP_OPS as readonly string[]).includes(value);

/**
 * Checks the options and chooses the cases (before touching the GPU — a misspelled set, op or case
 * is never run as the default). The cases are those matching both op and id (every case of the op
 * when no id is given), in the order of the shape table.
 */
const planSweep = (options: GeometrySweepOptions): SweepPlan => {
  const candidateSet = options.candidateSet ?? DEFAULT_CANDIDATE_SET;
  if (!isCandidateSet(candidateSet)) {
    throw new Error(`Candidate set ${candidateSet} is none of ${CANDIDATE_SETS.join(" / ")}`);
  }
  const ops = options.ops ?? SWEEP_OPS;
  const unknownOp = ops.find((op) => !isSweepOp(op));
  if (unknownOp !== undefined) {
    throw new Error(`op ${unknownOp} is none of ${SWEEP_OPS.join(" / ")}`);
  }
  const caseIds = options.cases ?? [];
  const unknownCase = caseIds.find((id) => !SWEEP_CASES.some((sweepCase) => sweepCase.id === id));
  if (unknownCase !== undefined) throw new Error(`Case ${unknownCase} is not in the shape table`);
  const rounds = options.rounds ?? ROUNDS;
  if (!Number.isInteger(rounds) || rounds < 1) {
    throw new Error(`rounds must be an integer of at least 1 (${rounds})`);
  }
  const cases = SWEEP_CASES.filter(
    (sweepCase) =>
      ops.includes(sweepCase.op) && (caseIds.length === 0 || caseIds.includes(sweepCase.id)),
  );
  if (cases.length === 0) {
    throw new Error(`No case matches op [${ops.join(", ")}] and cases [${caseIds.join(", ")}]`);
  }
  return { candidateSet, ops, caseIds, rounds, cases };
};

/**
 * The unit of timestamp differences. Deno's WebGPU (wgpu) returns timestamps as raw ticks without
 * converting them to ns (docs/known-issues.md section "Intel Arc B570 (Linux / Vulkan ANV / xe)" —
 * 1 tick = 52.08 ns on B570), hence `deno-raw-tick`; everything else (Chrome / Dawn) is `ns` as the
 * specification says. The implementation is told apart by `navigator.userAgent` (Deno returns
 * `Deno/<version>`).
 */
export const timestampUnitFor = (userAgent: string): "ns" | "deno-raw-tick" =>
  userAgent.startsWith("Deno/") ? "deno-raw-tick" : "ns";

/**
 * Acquires a dedicated device, sweeps the cases, and returns the record
 * (`karume-geometry-sweep/2`). The device is discarded when done (also on abort or failure).
 * Per-row failures (allocation, compilation, device lost) are kept as rows of the record and the
 * sweep continues; option errors and environments where no device can be acquired throw before
 * touching the GPU / at the acquisition step.
 *
 * MUST: never run alongside inference (it pollutes the measurement, and both compete for the GPU) —
 * the runtime cannot see how another device is used, so this is a contract with the caller (stated
 * in the public surface's doc).
 */
export const runGeometrySweep = async (options: GeometrySweepOptions = {}): Promise<Report> => {
  const plan = planSweep(options);
  // The start time is taken before acquiring the device (`date` is when the record is assembled =
  // the end time — the difference is the duration, ADR 0117 decision 8)
  const startedAt = new Date().toISOString();
  const notify = options.onProgress;
  // timestamp-query is requested when the adapter lists it (decision 2). acquireGpu's gpuTiming is
  // either "required" or "not requested" and has no "use if present", so an adapter is acquired
  // once and discarded only to look at the list. NOTE: the specification does not guarantee that
  // calling requestAdapter twice picks the same physical adapter — if the re-picked adapter lacks
  // timestamp-query, acquireGpu fails with GpuFeatureError (it does not silently drop to wall
  // clock). Without navigator.gpu, acquireGpu fails with GpuUnavailableError.
  const webgpu: GPU | undefined = options.gpu ?? navigator.gpu;
  const probe =
    options.timestamps === false || webgpu === undefined
      ? null
      : await webgpu.requestAdapter(options.adapter);
  const timestamps = probe?.features.has(TIMESTAMP_QUERY) === true;
  let deviceLost: Report["deviceLost"] = null;
  const gpu = await acquireGpu({
    ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
    ...(options.adapter === undefined ? {} : { adapter: options.adapter }),
    ...(timestamps ? { gpuTiming: true } : {}),
    onDeviceLost: (info) => {
      deviceLost = { reason: info.reason, message: info.message };
      notify?.({ kind: "deviceLost", reason: info.reason, message: info.message });
    },
  });
  try {
    const timestampUnit = timestampUnitFor(navigator.userAgent);
    const context = await createSweepContext(gpu, timestampUnit);
    const info = gpu.adapterInfo;
    const adapter: ReportAdapter = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
    let rows: readonly SweepRow[];
    let cases: readonly CaseSummary[];
    let aborted: boolean;
    try {
      notify?.({
        kind: "started",
        adapter,
        unit: context.unit,
        dp4a: context.dp4a,
        caseCount: plan.cases.length,
      });
      ({ rows, cases, aborted } = await runSweep(
        context,
        plan.cases,
        {
          rounds: plan.rounds,
          candidateSet: plan.candidateSet,
          timestampUnit,
        },
        {
          signal: options.signal,
          onProgress: (message) => notify?.({ kind: "status", message }),
          onRow: (row) => notify?.({ kind: "row", row }),
          onCase: (summary) => notify?.({ kind: "case", summary }),
        },
      ));
    } finally {
      destroySweepContext(context);
    }
    return {
      format: REPORT_FORMAT,
      date: new Date().toISOString(),
      startedAt,
      aborted,
      // Write the ratio baseline, the kernels of the measured candidates and the case-set version
      // into the record (the generator matches them against the current runtime — ADR 0117 decision
      // 8). All are pure functions that read no GPU, with a negligible cost next to the sweep (the
      // default fingerprint takes about 13 ms; the candidate fingerprint stays under 1 second even
      // for every row of full)
      caseSet: sweepCaseSetId(),
      defaultKernels: geometryProfileKernelsId(DEFAULT_GEOMETRY_PROFILE),
      candidateKernels: sweepCandidateKernelsId(rows, context.dp4a),
      userAgent: navigator.userAgent,
      adapter,
      gpuTiming: {
        feature: gpu.gpuTimingEnabled,
        unit: context.unit,
        quantized: roundsLookQuantized(rows, context.unit),
      },
      dp4a: context.dp4a,
      settings: {
        candidateSet: plan.candidateSet,
        quick: plan.candidateSet === "quick",
        ops: plan.ops,
        ...(plan.caseIds.length === 0 ? {} : { cases: plan.caseIds }),
        rounds: plan.rounds,
        targetPassMs: TARGET_PASS_MS,
        maxReps: SWEEP_MAX_REPS,
        warmupNs: WARMUP_NS,
        warmupMinRuns: WARMUP_MIN_RUNS,
        wallTimingNote: WALL_TIMING_NOTE,
      },
      deviceLost,
      cases,
      rows,
    };
  } finally {
    gpu.destroy();
  }
};
