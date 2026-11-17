/**
 * The GEMV sweep (`runGemvSweep` of `@karume/runtime/tune`): measures the GEMV choices of `linear` —
 * which sequential variant, how many rows per row block, which lane count of the parallel GEMV,
 * staged or not, subgroup or not — on a dedicated device, case by case, and returns the record
 * (`karume-gemv-sweep/1`). It derives no table and changes no choice.
 *
 * Per case and row count it measures the baseline first (what a Session at the level `exact` runs
 * with the default profile), then every candidate of the set (gemv-candidates.ts), then the baseline
 * once more (the drift ratio). Timing is the geometry sweep's protocol through harness.ts's
 * primitives (`createResources`, `prepareLaunch`, `timeLaunch`): reps per pass from the wall-clock
 * slope, the warm-up, the min of the rounds, GPU timestamps or the wall clock. The inputs are the
 * harness's seeded fills, one buffer set per case at the case's largest row count.
 *
 * Correctness, per row:
 * - identical class: every output word equals the baseline's (`compareBufferWords`); a difference
 *   fails the row's adoption, however fast;
 * - reorder class: the largest absolute error against the host's f64 oracle (gemv-oracle.ts) and
 *   its use of the band of `linear` (`opTolerance("linear")`); outside the band fails the row's
 *   adoption. The baseline's own error and band use are recorded at the same row counts.
 *
 * MUST: never called from the acquire / Session path — the runtime measures nothing at run time
 * (ADR 0022's ban on autotuning). Never run it alongside inference on the same GPU.
 */

import { tiledWorkgroups, gridStrideWorkgroups } from "../codegen/dispatch.ts";
import type { GpuContext } from "../gpu/device.ts";
import type { Tolerance } from "../reference/allclose.ts";
import { opTolerance } from "../reference/op-tolerance.ts";
import type { LinearProgram } from "../runtime/linear-program.ts";
import {
  type GemvCandidate,
  type GemvCandidateSet,
  type GemvGates,
  type GemvMeasurementPlan,
  GEMV_CANDIDATE_SETS,
  gemvMeasurementPlans,
  gemvProgram,
  gemvRowCountsIn,
  isGemvCandidateSet,
} from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase } from "./gemv-cases.ts";
import { gemvCaseTableId, gemvKernelsId } from "./gemv-fingerprint.ts";
import { bandAgainstOracle, gemvOracle } from "./gemv-oracle.ts";
import {
  choiceRecord,
  GEMV_SWEEP_FORMAT,
  GEMV_SWEEP_NOT_MEASURED,
  type GemvSweepCaseSummary,
  type GemvSweepGates,
  type GemvSweepReport,
  type GemvSweepRow,
} from "./gemv-report.ts";
import {
  compareBufferWords,
  createResources,
  createSweepContext,
  destroySweepContext,
  type Launch,
  prepareLaunch,
  readBufferWords,
  type ResourceSpec,
  resourceWordStream,
  SWEEP_MAX_REPS,
  type SweepContext,
  timeLaunch,
  type Timing,
} from "./harness.ts";
import { ROUNDS, TARGET_PASS_MS, WARMUP_MIN_RUNS, WARMUP_NS } from "./measurement.ts";
import {
  type ReportAdapter,
  roundsLookQuantized,
  type TimingUnit,
  WALL_TIMING_NOTE,
} from "./report.ts";
import { acquireSweepGpu, timestampUnitFor } from "./sweep.ts";

/**
 * Sweep progress: `started` once the device and the measurement context exist; `status` — what is
 * measured now; `row` — one row is final (a case's row count starts with its baseline); `case` — a
 * case is measured, with its summaries (one per row count); `deviceLost`.
 */
export type GemvSweepProgress =
  | {
      readonly kind: "started";
      readonly adapter: ReportAdapter;
      readonly unit: TimingUnit;
      /** Whether the subgroup form is measured on this device. */
      readonly subgroups: boolean;
      readonly caseCount: number;
    }
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "row"; readonly row: GemvSweepRow }
  | {
      readonly kind: "case";
      readonly caseId: string;
      readonly summaries: readonly GemvSweepCaseSummary[];
    }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

/** Options of {@link runGemvSweep} (all optional). */
export type GemvSweepOptions = {
  /** The `GPU` instance to use (default `navigator.gpu`). */
  readonly gpu?: GPU;
  /**
   * `quick` (the default: M = 1, every reorder candidate, the identical class at a small grid around
   * the baseline) or `full` (every row count, the whole grid).
   */
  readonly candidateSet?: GemvCandidateSet;
  /** Narrows by case id (default: every case of the table). */
  readonly cases?: readonly string[];
  /** Measurement rounds per kernel (default 5 — the min is kept). */
  readonly rounds?: number;
  /** Abort: stops once the current kernel is measured; the record holds what was measured. */
  readonly signal?: AbortSignal;
  readonly onProgress?: (progress: GemvSweepProgress) => void;
};

/** The resources of a case, in fill order (the index of an entry seeds its stream). */
const caseResources = (gemvCase: GemvCase, rows: number): ResourceSpec[] => {
  const { storage, group, n, k } = gemvCase;
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  const scales = storage === "i4" && group !== undefined ? n * (k / group) : n;
  return [
    { name: "x", bytes: rows * k * 4, fill: "f32" },
    // Every bit pattern is a valid i2 / i4 code; i8 takes the fill's [-127, 127].
    { name: "w", bytes: (n * k * bits) / 8, fill: "i8" },
    { name: "bias", bytes: n * 4, fill: "f32" },
    { name: "wscale", bytes: scales * 4, fill: "scale" },
    { name: "out", bytes: rows * n * 4, fill: "none" },
    // The baseline writes here, so the candidates' outputs can be compared with it.
    { name: "reference", bytes: rows * n * 4, fill: "none" },
  ];
};

/** The word stream a resource was filled from (the same seed as `createResources`). */
const streamOf = (specs: readonly ResourceSpec[], name: string): (() => number) => {
  const index = specs.findIndex((spec) => spec.name === name);
  const spec = specs[index];
  if (spec === undefined || spec.fill === "none") throw new Error(`No filled resource ${name}`);
  return resourceWordStream(spec.fill, index);
};

/** The resource each operand of a GEMV program binds (the output is named per launch). */
const OPERAND_RESOURCES = { x: "x", weight: "w", bias: "bias", weightScale: "wscale" } as const;

/**
 * The launch of a program whose one dispatch writes `output`: its key, WGSL, params and grid as the
 * program states them, its bindings mapped to the case's resources by position. The sweep times
 * programs of one dispatch without temporaries (every GEMV program is one).
 */
export const gemvLaunch = (
  program: LinearProgram,
  output: "out" | "reference",
  limit: number,
): Launch => {
  const [dispatch] = program.dispatches;
  if (dispatch === undefined || program.dispatches.length !== 1 || program.temps.length > 0) {
    throw new Error(
      `${program.where}: the GEMV sweep times a program of one dispatch without temporaries`,
    );
  }
  const bindings = dispatch.bindings.map(({ binding, operand }, index) => {
    if (binding !== index + 1) {
      throw new Error(`${program.where}: binding ${binding} is at position ${index + 1}`);
    }
    if (typeof operand !== "string") throw new Error(`${program.where}: binds a temporary`);
    return operand === "out" ? output : OPERAND_RESOURCES[operand];
  });
  const { grid } = dispatch;
  return {
    key: dispatch.key,
    wgsl: dispatch.wgsl(),
    params: dispatch.params,
    bindings,
    writes: [output],
    workgroups:
      grid.kind === "tiled"
        ? [
            tiledWorkgroups(grid.axes[0].items, grid.axes[0].tile, limit, program.where),
            tiledWorkgroups(grid.axes[1].items, grid.axes[1].tile, limit, program.where),
            tiledWorkgroups(grid.axes[2].items, grid.axes[2].tile, limit, program.where),
          ]
        : [gridStrideWorkgroups(grid.items, grid.workgroupSize, limit), 1, 1],
  };
};

const describeCause = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

const ALL_ROUNDS_NEGATIVE =
  "timestamp negative in every round (only samples clamped to 0, so no min can be taken)";

/** The settings one case is measured with. */
export type GemvCaseSettings = {
  readonly rounds: number;
  /** The band the reorder class is held to (`opTolerance("linear")` in the sweep). */
  readonly tolerance: Tolerance;
};

export type GemvCaseHooks = {
  readonly onProgress?: (message: string) => void;
  readonly onRow?: (row: GemvSweepRow) => void;
  readonly signal?: AbortSignal;
};

/** The rows and summaries of one case, and whether an abort cut it short. */
export type GemvCaseResult = {
  readonly rows: readonly GemvSweepRow[];
  readonly summaries: readonly GemvSweepCaseSummary[];
  readonly aborted: boolean;
};

/** A row's identity fields. */
const rowHead = (
  gemvCase: GemvCase,
  m: number,
  candidate: GemvCandidate,
  isBaseline: boolean,
): GemvSweepRow => ({
  caseId: gemvCase.id,
  m,
  candidate: candidate.name,
  class: candidate.class,
  isBaseline,
  choice: choiceRecord(candidate.choice, gemvCase.storage),
});

const errorRow = (head: GemvSweepRow, cause: unknown): GemvSweepRow => ({
  ...head,
  error: describeCause(cause),
});

/** The timing fields of a row. */
const timedRow = (head: GemvSweepRow, launch: Launch, timing: Timing): GemvSweepRow => ({
  ...head,
  keys: [launch.key],
  workgroups: launch.workgroups,
  reps: timing.reps,
  rounds: timing.rounds,
  wallRounds: timing.wallRounds,
  clampedNegativeSamples: timing.clampedNegativeSamples,
  wallNsPerDispatch: timing.wallNsPerDispatch,
  ...(timing.perDispatch === undefined
    ? { error: ALL_ROUNDS_NEGATIVE }
    : { perDispatch: timing.perDispatch }),
});

/** The correctness fields of an identical-class row: its output words against the baseline's. */
const identicalFields = (
  words: Awaited<ReturnType<typeof compareBufferWords>>,
): Pick<GemvSweepRow, "identicalToBaseline" | "differingWords" | "firstDifference"> => ({
  identicalToBaseline: words.differing === 0,
  differingWords: words.differing,
  ...(words.first === undefined
    ? {}
    : {
        firstDifference: {
          index: words.first.index,
          baseline: words.first.a,
          candidate: words.first.b,
        },
      }),
});

/** The baseline's time ÷ the row's, when both are positive. */
const speedFields = (
  baseline: GemvSweepRow,
  row: GemvSweepRow,
): Pick<GemvSweepRow, "speedupVsBaseline"> =>
  baseline.perDispatch !== undefined &&
  baseline.perDispatch > 0 &&
  row.perDispatch !== undefined &&
  row.perDispatch > 0
    ? { speedupVsBaseline: baseline.perDispatch / row.perDispatch }
    : {};

/**
 * Measures one case: its buffers once, at its largest row count, and the host oracle once, at the
 * largest row count with reorder candidates; then per measurement plan the baseline, the
 * candidates and the baseline again. A failure of the buffers makes every row a failure row; a lost
 * device makes the rest failure rows; an abort makes no rows for the rest (the summary of the row
 * count it stopped in has no re-measurement). The user's callbacks run outside the measurement's
 * `try`, so an exception they throw is the caller's, never a failure row.
 */
export const sweepGemvCase = async (
  context: SweepContext,
  gemvCase: GemvCase,
  plans: readonly GemvMeasurementPlan[],
  settings: GemvCaseSettings,
  hooks: GemvCaseHooks = {},
): Promise<GemvCaseResult> => {
  const { gpu } = context;
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  const rows: GemvSweepRow[] = [];
  const summaries: GemvSweepCaseSummary[] = [];
  const emit = (row: GemvSweepRow): void => {
    rows.push(row);
    hooks.onRow?.(row);
  };
  if (plans.length === 0) return { rows, summaries, aborted: false };
  const maxRows = Math.max(...plans.map(({ m }) => m));
  const oracleRows = Math.max(
    0,
    ...plans
      .filter(({ candidates }) => candidates.some((candidate) => candidate.class === "reorder"))
      .map(({ m }) => m),
  );
  const specs = caseResources(gemvCase, maxRows);
  hooks.onProgress?.(`${gemvCase.id}: preparing inputs`);
  let resources: Map<string, GPUBuffer>;
  try {
    resources = await createResources(context, specs);
  } catch (cause) {
    for (const plan of plans) {
      for (const [index, candidate] of [plan.baseline, ...plan.candidates].entries()) {
        emit(errorRow(rowHead(gemvCase, plan.m, candidate, index === 0), cause));
      }
    }
    return { rows, summaries, aborted: false };
  }
  const buffer = (name: string): GPUBuffer => {
    const found = resources.get(name);
    if (found === undefined) throw new Error(`${gemvCase.id}: resource ${name} is missing`);
    return found;
  };
  /** Prepares and times one launch (its params buffer released after). */
  const time = async (launch: Launch): Promise<Timing> => {
    const ready = await prepareLaunch(context, launch, resources);
    try {
      return await timeLaunch(context, ready, buffer(launch.writes[0] ?? ""), settings.rounds);
    } finally {
      ready.params.destroy();
    }
  };
  const launchOf = (m: number, candidate: GemvCandidate, output: "out" | "reference"): Launch =>
    gemvLaunch(gemvProgram(gemvCase, m, candidate.choice), output, limit);
  let oracle: Float64Array<ArrayBuffer> | Error | undefined;
  /** The band reading of the first `m` rows of `output`. */
  const band = async (output: "out" | "reference", m: number) => {
    if (oracle === undefined) throw new Error(`${gemvCase.id}: no oracle for ${m} rows`);
    if (oracle instanceof Error) throw oracle;
    const words = await readBufferWords(context, buffer(output), m * gemvCase.n * 4);
    return bandAgainstOracle(
      new Float32Array(words.buffer),
      oracle.subarray(0, m * gemvCase.n),
      settings.tolerance,
    );
  };
  /** The correctness fields and the ratio to the baseline of a timed row. */
  const checked = async (
    row: GemvSweepRow,
    m: number,
    baseline: GemvSweepRow | undefined,
  ): Promise<GemvSweepRow> => {
    if (row.isBaseline) return m <= oracleRows ? { ...row, ...(await band("reference", m)) } : row;
    if (baseline === undefined || baseline.error !== undefined) return row;
    const speed = speedFields(baseline, row);
    if (row.class === "reorder") return { ...row, ...speed, ...(await band("out", m)) };
    const words = await compareBufferWords(
      context,
      buffer("reference"),
      buffer("out"),
      m * gemvCase.n * 4,
    );
    return { ...row, ...speed, ...identicalFields(words) };
  };
  /** The baseline measured once more (the drift ratio), unless it has no first time or the device is lost. */
  const repeatBaseline = async (
    plan: GemvMeasurementPlan,
    baseline: GemvSweepRow | undefined,
  ): Promise<Pick<GemvSweepCaseSummary, "baselineRepeat" | "baselineRepeatError">> => {
    const initial = baseline?.perDispatch;
    if (initial === undefined || !(initial > 0) || gpu.lost !== undefined) return {};
    hooks.onProgress?.(`${gemvCase.id} M=${plan.m}: re-measuring the baseline`);
    try {
      const timing = await time(launchOf(plan.m, plan.baseline, "reference"));
      if (timing.perDispatch === undefined) throw new Error(ALL_ROUNDS_NEGATIVE);
      if (!(timing.perDispatch > 0)) {
        throw new Error(
          "perDispatch of the baseline re-measurement is 0 (no ratio can be computed)",
        );
      }
      return {
        baselineRepeat: {
          perDispatch: timing.perDispatch,
          driftRatio: timing.perDispatch / initial,
        },
      };
    } catch (cause) {
      return { baselineRepeatError: describeCause(cause) };
    }
  };
  /** Read afresh each time: the signal is aborted from outside while the case awaits the GPU. */
  const abortRequested = (): boolean => hooks.signal?.aborted === true;
  let aborted = false;
  try {
    if (oracleRows > 0) {
      hooks.onProgress?.(`${gemvCase.id}: the f64 oracle of ${oracleRows} row(s) on the host`);
      try {
        oracle = await gemvOracle(gemvCase, oracleRows, {
          x: streamOf(specs, "x"),
          weight: streamOf(specs, "w"),
          bias: streamOf(specs, "bias"),
          scale: streamOf(specs, "wscale"),
        });
      } catch (cause) {
        oracle = cause instanceof Error ? cause : new Error(String(cause));
      }
    }
    for (const plan of plans) {
      if (abortRequested()) {
        aborted = true;
        break;
      }
      const started = performance.now();
      const { m } = plan;
      const ordered = [plan.baseline, ...plan.candidates];
      let baseline: GemvSweepRow | undefined;
      for (const [index, candidate] of ordered.entries()) {
        if (abortRequested()) {
          aborted = true;
          break;
        }
        const head = rowHead(gemvCase, m, candidate, index === 0);
        if (gpu.lost !== undefined) {
          emit(errorRow(head, new Error(`device lost: ${gpu.lost.message}`)));
          continue;
        }
        hooks.onProgress?.(
          `${gemvCase.id} M=${m}: ${candidate.name} (${index + 1}/${ordered.length})`,
        );
        let row: GemvSweepRow;
        try {
          const launch = launchOf(m, candidate, head.isBaseline ? "reference" : "out");
          row = timedRow(head, launch, await time(launch));
          if (row.error === undefined) row = await checked(row, m, baseline);
        } catch (cause) {
          row = errorRow(head, cause);
        }
        if (head.isBaseline) baseline = row;
        emit(row);
      }
      const repeat = aborted ? {} : await repeatBaseline(plan, baseline);
      summaries.push({
        caseId: gemvCase.id,
        m,
        elapsedMs: performance.now() - started,
        profile: plan.profile,
        ...repeat,
      });
      if (aborted) break;
    }
  } finally {
    for (const resource of resources.values()) resource.destroy();
  }
  return { rows, summaries, aborted };
};

/** The device facts that decide the candidates, as the record writes them. */
export const gemvSweepGates = (
  gpu: Pick<GpuContext, "features" | "wgslLanguageFeatures"> & {
    readonly limits: { readonly maxComputeWorkgroupsPerDimension: number };
  },
): GemvSweepGates => {
  const features = {
    subgroups: gpu.features.has("subgroups"),
    subgroupSizeControl: gpu.features.has("subgroup-size-control"),
    subgroupId: gpu.wgslLanguageFeatures.has("subgroup_id"),
  };
  return {
    subgroups: features.subgroups && features.subgroupSizeControl && features.subgroupId,
    features,
    maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
  };
};

/** The options checked and the cases chosen. */
type GemvSweepPlan = {
  readonly candidateSet: GemvCandidateSet;
  readonly caseIds: readonly string[];
  readonly rounds: number;
  readonly cases: readonly GemvCase[];
};

/** Checks the options and chooses the cases (before the GPU is touched). */
const planGemvSweep = (options: GemvSweepOptions): GemvSweepPlan => {
  const candidateSet = options.candidateSet ?? "quick";
  if (!isGemvCandidateSet(candidateSet)) {
    throw new Error(`Candidate set ${candidateSet} is none of ${GEMV_CANDIDATE_SETS.join(" / ")}`);
  }
  const caseIds = options.cases ?? [];
  const unknown = caseIds.find((id) => !GEMV_CASES.some((gemvCase) => gemvCase.id === id));
  if (unknown !== undefined)
    throw new Error(`Case ${unknown} is not in the GEMV sweep's case table`);
  const rounds = options.rounds ?? ROUNDS;
  if (!Number.isInteger(rounds) || rounds < 1) {
    throw new Error(`rounds must be an integer of at least 1 (${rounds})`);
  }
  return {
    candidateSet,
    caseIds,
    rounds,
    cases: GEMV_CASES.filter((gemvCase) => caseIds.length === 0 || caseIds.includes(gemvCase.id)),
  };
};

/** What the record is assembled from. */
export type GemvSweepParts = {
  readonly startedAt: string;
  readonly date: string;
  readonly aborted: boolean;
  readonly userAgent: string;
  readonly adapter: ReportAdapter;
  readonly gates: GemvSweepGates;
  readonly timing: { readonly feature: boolean; readonly unit: TimingUnit };
  readonly candidateSet: GemvCandidateSet;
  /** The case ids the sweep was narrowed to (empty: every case). */
  readonly caseIds: readonly string[];
  readonly rounds: number;
  readonly tolerance: Tolerance;
  readonly deviceLost: GemvSweepReport["deviceLost"];
  /** The planned cases. */
  readonly cases: readonly GemvCase[];
  readonly summaries: readonly GemvSweepCaseSummary[];
  readonly rows: readonly GemvSweepRow[];
};

/** The record of a sweep (pure: the fingerprints are recomputed from the case table and the rows). */
export const gemvSweepReport = (parts: GemvSweepParts): GemvSweepReport => ({
  format: GEMV_SWEEP_FORMAT,
  date: parts.date,
  startedAt: parts.startedAt,
  aborted: parts.aborted,
  caseTable: gemvCaseTableId(),
  kernels: gemvKernelsId(parts.rows),
  userAgent: parts.userAgent,
  adapter: parts.adapter,
  gates: parts.gates,
  gpuTiming: {
    feature: parts.timing.feature,
    unit: parts.timing.unit,
    quantized: roundsLookQuantized(parts.rows, parts.timing.unit),
  },
  settings: {
    candidateSet: parts.candidateSet,
    ...(parts.caseIds.length === 0 ? {} : { cases: parts.caseIds }),
    rounds: parts.rounds,
    targetPassMs: TARGET_PASS_MS,
    maxReps: SWEEP_MAX_REPS,
    warmupNs: WARMUP_NS,
    warmupMinRuns: WARMUP_MIN_RUNS,
    wallTimingNote: WALL_TIMING_NOTE,
    tolerance: { atol: parts.tolerance.atol, rtol: parts.tolerance.rtol },
    notMeasured: GEMV_SWEEP_NOT_MEASURED,
  },
  deviceLost: parts.deviceLost,
  cases: parts.cases.map((gemvCase): GemvCase => ({
    id: gemvCase.id,
    storage: gemvCase.storage,
    ...(gemvCase.group === undefined ? {} : { group: gemvCase.group }),
    n: gemvCase.n,
    k: gemvCase.k,
    rows: gemvRowCountsIn(gemvCase, parts.candidateSet),
    source: gemvCase.source,
  })),
  summaries: parts.summaries,
  rows: parts.rows,
});

/**
 * Acquires a dedicated device (with the subgroup features where the device runs them, and
 * `timestamp-query` where the adapter lists it), sweeps the chosen cases and returns the record
 * (`karume-gemv-sweep/1`). The device is destroyed when done (also on abort or failure). Per-row
 * failures stay rows of the record and the sweep goes on; option errors throw before the GPU is
 * touched. A lost device ends the sweep after the case it was lost in.
 */
export const runGemvSweep = async (options: GemvSweepOptions = {}): Promise<GemvSweepReport> => {
  const plan = planGemvSweep(options);
  const startedAt = new Date().toISOString();
  const notify = options.onProgress;
  let deviceLost: GemvSweepReport["deviceLost"] = null;
  const gpu = await acquireSweepGpu({
    ...(options.gpu === undefined ? {} : { gpu: options.gpu }),
    subgroups: "available",
    onDeviceLost: (lost) => {
      deviceLost = lost;
      notify?.({ kind: "deviceLost", reason: lost.reason, message: lost.message });
    },
  });
  try {
    const context = await createSweepContext(gpu, timestampUnitFor(navigator.userAgent));
    const info = gpu.adapterInfo;
    const adapter: ReportAdapter = {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
    const gates = gemvSweepGates(gpu);
    const candidateGates: GemvGates = {
      subgroups: gates.subgroups,
      maxComputeWorkgroupsPerDimension: gates.maxComputeWorkgroupsPerDimension,
    };
    const tolerance = opTolerance("linear");
    const rows: GemvSweepRow[] = [];
    const summaries: GemvSweepCaseSummary[] = [];
    let aborted = false;
    try {
      notify?.({
        kind: "started",
        adapter,
        unit: context.unit,
        subgroups: gates.subgroups,
        caseCount: plan.cases.length,
      });
      for (const gemvCase of plan.cases) {
        if (gpu.lost !== undefined) break;
        if (options.signal?.aborted === true) {
          aborted = true;
          break;
        }
        const result = await sweepGemvCase(
          context,
          gemvCase,
          gemvMeasurementPlans(gemvCase, plan.candidateSet, candidateGates),
          { rounds: plan.rounds, tolerance },
          {
            ...(options.signal === undefined ? {} : { signal: options.signal }),
            onProgress: (message) => notify?.({ kind: "status", message }),
            onRow: (row) => notify?.({ kind: "row", row }),
          },
        );
        rows.push(...result.rows);
        summaries.push(...result.summaries);
        notify?.({ kind: "case", caseId: gemvCase.id, summaries: result.summaries });
        if (result.aborted) {
          aborted = true;
          break;
        }
      }
    } finally {
      destroySweepContext(context);
    }
    return gemvSweepReport({
      startedAt,
      date: new Date().toISOString(),
      aborted,
      userAgent: navigator.userAgent,
      adapter,
      gates,
      timing: { feature: gpu.gpuTimingEnabled, unit: context.unit },
      candidateSet: plan.candidateSet,
      caseIds: plan.caseIds,
      rounds: plan.rounds,
      tolerance,
      deviceLost,
      cases: plan.cases,
      summaries,
      rows,
    });
  } finally {
    gpu.destroy();
  }
};
