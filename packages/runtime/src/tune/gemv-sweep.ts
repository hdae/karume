/**
 * The GEMV sweep (`runGemvSweep` of `@karume/runtime/tune`): measures the GEMV choices of `linear` —
 * which sequential variant, how many rows per row block, which lane count of the parallel GEMV,
 * staged or not, subgroup or not — on a dedicated device, case by case, and returns the record
 * (`karume-gemv-sweep/4`). It derives no table and changes no choice.
 *
 * Per case and row count it measures the baseline (what a Session at the level `exact` runs with
 * the default profile), the default profile's choice at `reorder` and the candidates of the set
 * (gemv-candidates.ts — all, or those the caller names) together, through harness.ts's primitives
 * (its measurement protocol). A case's row counts are measured in ascending order, so at M = 4 and 8
 * the run has the case's M = 1 group (when it measured it): a reorder candidate whose row there
 * leaves it unadoptable is skipped (gemv-candidates.ts's `gemvCarriedFromM1`) — its program and
 * fingerprint are built, it takes none of the steps below, and its row says `skipped`.
 * 1. the pipelines of every kernel are made first, so no compile leaves the device idle later;
 * 2. a warm start on the baseline (`warmStart`): the device is loaded until its passes are steady;
 * 3. the quick look (harness.ts's protocol rule 4 — skipped at a margin of 0): every kernel's
 *    calibration probe, with timestamps, in a shuffled order (`quickLook`); the references and the
 *    kernels within the margin of their class's bar go on (`quickLookGoesOn` — the bar is the
 *    fastest of the class where the derivation picks a winner, the default's choice at `reorder`
 *    where it only checks for a regression: `gemvGroupRole`), the others become screened rows;
 * 4. every kernel that goes on, the baseline first, is run once from a zeroed output, calibrated
 *    (`calibrate`, from its quick look's reps: done again after more load when its pass comes out
 *    far below the target) and checked while its output is still in the buffer;
 * 5. the rounds: each times every kernel that went on once, the baseline included, in plan order
 *    rotated so that round r starts at kernel ⌊r · kernels / rounds⌋ (at kernel r mod kernels when
 *    there are more rounds than kernels). A kernel's time is the min of its rounds, GPU timestamps
 *    or the wall clock.
 * Interleaving puts every kernel under the device states the row count went through, so a heating
 * device slows them alike instead of whichever was measured last; how far the state moved is the
 * spread of the baseline's rounds (the summary's `baselineSpread`). There is no per-kernel warm-up
 * (the geometry sweep's protocol rule 3): after the warm start, the calibrations and the rounds keep
 * the device loaded. The inputs are the harness's seeded fills, one buffer set per case at the
 * case's largest row count.
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
  type GemvM1Group,
  type GemvMeasurementPlan,
  GEMV_CANDIDATE_SETS,
  gemvCarriedFromM1,
  gemvGroupRole,
  gemvKernelsAt,
  gemvMeasurementPlans,
  gemvProgram,
  gemvRowCountsIn,
  type GemvSelection,
  isGemvCandidateSet,
} from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase } from "./gemv-cases.ts";
import { gemvCaseFingerprint, gemvProgramKernelId } from "./gemv-fingerprint.ts";
import { bandAgainstOracle, gemvOracle } from "./gemv-oracle.ts";
import {
  choiceRecord,
  GEMV_SWEEP_FORMAT,
  GEMV_SWEEP_NOT_MEASURED,
  type GemvSweepCaseEntry,
  type GemvSweepCaseSummary,
  type GemvSweepGates,
  type GemvSweepReport,
  type GemvSweepRow,
} from "./gemv-report.ts";
import {
  calibrate,
  compareBufferWords,
  createResources,
  createSweepContext,
  destroySweepContext,
  type Launch,
  measurePass,
  type PassSample,
  prepareLaunch,
  type QuickLook,
  quickLook,
  quickLookGoesOn,
  quickLookOrder,
  readBufferWords,
  type ReadyLaunch,
  type ResourceSpec,
  resourceWordStream,
  roundsTiming,
  runLaunchOnce,
  SWEEP_MAX_REPS,
  type SweepContext,
  type Timing,
  WARM_START_MAX_PASSES,
  warmStart,
} from "./harness.ts";
import {
  CALIBRATION_ATTEMPTS,
  quickLookMarginOf,
  RECALIBRATE_BELOW,
  ROUNDS,
  TARGET_PASS_MS,
  WARM_START_MAX_NS,
  WARM_START_MIN_NS,
  WARM_START_PASSES,
  WARM_START_TOLERANCE,
} from "./measurement.ts";
import {
  type ReportAdapter,
  roundsLookQuantized,
  type TimingUnit,
  WALL_TIMING_NOTE,
} from "./report.ts";
import { acquireSweepGpu, timestampUnitFor } from "./sweep.ts";

/**
 * Sweep progress: `started` once the device and the measurement context exist; `status` — what is
 * measured now; `row` — one row is final (the rows of a row count come together once its rounds are
 * done, its baseline first); `case` — a case is measured, with its summaries (one per row count);
 * `deviceLost`.
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
   * the baseline), `full` (every row count a derivation rule reads — M = 1, 4 and 8 —, the whole
   * grid) or `row-blocks` (M = 16 and 64: the row-block heights, which no rule reads yet).
   */
  readonly candidateSet?: GemvCandidateSet;
  /** Narrows by case id (default: every case of the table). */
  readonly cases?: readonly string[];
  /** Narrows by row count, of the set's (default: every row count of the set). */
  readonly rows?: readonly number[];
  /**
   * Narrows by candidate name (`c32u4`, `parallel-l4`, …; default: every candidate of the set). The
   * references — the baseline and the default profile's choice at `reorder` — are measured either
   * way. A name that is no candidate of any planned case and row count throws.
   */
  readonly candidates?: readonly string[];
  /** Measurement rounds per kernel (default 5 — the min is kept). */
  readonly rounds?: number;
  /**
   * The quick look's margin (default measurement.ts's `QUICK_LOOK_MARGIN`, ×1.25; harness.ts's
   * protocol rule 4). 0 turns the quick look off (every kernel is timed); any other value below 1
   * throws.
   */
  readonly quickLookMargin?: number;
  /**
   * Abort: stops once the current pass is done. The record holds the row counts measured before; the
   * row count it stopped in leaves no row (its kernels are measured together, so none was finished).
   */
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
  /** The quick look's margin (0: off). */
  readonly quickLookMargin: number;
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

/** The quick look's field of a row. */
const quickFields = (quick: QuickLook | undefined): Pick<GemvSweepRow, "quickPerDispatch"> =>
  quick?.perDispatch === undefined ? {} : { quickPerDispatch: quick.perDispatch };

/** The row of a kernel the quick look did not let go on: its launch and its quick time only. */
const screenedRow = (
  head: GemvSweepRow,
  launch: Launch,
  quick: QuickLook | undefined,
): GemvSweepRow => ({
  ...head,
  keys: [launch.key],
  workgroups: launch.workgroups,
  ...quickFields(quick),
  screened: true,
});

/** The row of a kernel the M = 1 group of the same run left out: its launch only. */
const skippedRow = (head: GemvSweepRow, launch: Launch): GemvSweepRow => ({
  ...head,
  keys: [launch.key],
  workgroups: launch.workgroups,
  skipped: true,
});

/** The timing fields of a row. */
const timedRow = (
  head: GemvSweepRow,
  launch: Launch,
  timing: Timing,
  quick: QuickLook | undefined,
): GemvSweepRow => ({
  ...head,
  keys: [launch.key],
  workgroups: launch.workgroups,
  ...quickFields(quick),
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

/** The correctness fields a kernel's check reads (by class — none for a candidate without a baseline). */
type CheckFields = Pick<
  GemvSweepRow,
  | "identicalToBaseline"
  | "differingWords"
  | "firstDifference"
  | "maxAbsError"
  | "bandUse"
  | "withinBand"
  | "nonFiniteOutputs"
>;

/** Where a kernel of a row count stands. */
type KernelState =
  | { readonly kind: "planned" }
  | {
      readonly kind: "prepared";
      readonly launch: Launch;
      readonly ready: ReadyLaunch;
      readonly quick?: QuickLook;
    }
  | { readonly kind: "screened"; readonly launch: Launch; readonly quick?: QuickLook }
  /** Left out by the M = 1 group of the same run: its launch for the row, nothing measured. */
  | { readonly kind: "skipped"; readonly launch: Launch }
  | {
      readonly kind: "calibrated";
      readonly launch: Launch;
      readonly ready: ReadyLaunch;
      readonly quick?: QuickLook;
      readonly reps: number;
      /** Read right after its calibration, while its output is still in the buffer. */
      readonly checks: CheckFields;
      /** Its rounds so far, in round order. */
      readonly samples: PassSample[];
    }
  | { readonly kind: "failed"; readonly cause: unknown };

/**
 * One kernel of a row count: its row's identity (with its kernel's fingerprint once its program is
 * built), the buffer it writes and its state.
 */
type Kernel = {
  readonly candidate: GemvCandidate;
  head: GemvSweepRow;
  readonly output: "out" | "reference";
  state: KernelState;
};

/**
 * The kernels of round `round` of `rounds`: plan order rotated to start at kernel
 * ⌊round · kernels / rounds⌋ — with more rounds than kernels, at kernel `round mod kernels`. A
 * kernel's place in the round moves by an even step from round to round, so each kernel takes the
 * early and the late places of a round alike, and two consecutive rounds never share an order (of
 * two kernels or more).
 */
const roundOrder = <T>(kernels: readonly T[], round: number, rounds: number): T[] => {
  if (kernels.length === 0) return [];
  const start =
    rounds > kernels.length
      ? round % kernels.length
      : Math.floor((round * kernels.length) / rounds);
  return [...kernels.slice(start), ...kernels.slice(0, start)];
};

/** The baseline's largest positive round ÷ its smallest (rounds clamped to 0 left out). */
const spreadOf = (samples: readonly PassSample[]): Pick<GemvSweepCaseSummary, "baselineSpread"> => {
  const kept = samples
    .filter((sample) => !sample.clamped)
    .map((sample) => sample.gpu ?? sample.wallNs);
  const smallest = Math.min(...kept);
  return kept.length > 0 && smallest > 0 ? { baselineSpread: Math.max(...kept) / smallest } : {};
};

/**
 * Measures one case: its buffers once, at its largest row count, and the host oracle once, at the
 * largest row count with reorder candidates; then per measurement plan its kernels together (the
 * module doc's steps 1–4), their rows coming at the end of the plan, the baseline's first. A failure
 * of the buffers makes every row a failure row; a kernel that fails to be prepared, warmed on,
 * calibrated, checked or timed makes its own failure row and leaves the rounds; a lost device makes
 * every kernel without all its rounds a failure row, in this plan and the rest (a skipped kernel
 * keeps its skipped row: it was never to be measured); an abort makes no
 * row for the plan it stops in or the rest. The user's callbacks run outside the measurement's
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
  /**
   * The correctness fields of a kernel whose output is in its buffer: the baseline's band reading
   * at a row count with reorder rows; a candidate's words against the baseline's (identical class)
   * or its band reading (reorder class) — none when the baseline has no output to compare with.
   */
  const checksOf = async (kernel: Kernel, m: number, baseline: Kernel): Promise<CheckFields> => {
    if (kernel.head.isBaseline) return m <= oracleRows ? await band("reference", m) : {};
    if (baseline.state.kind !== "calibrated") return {};
    if (kernel.head.class === "reorder") return await band("out", m);
    const words = await compareBufferWords(
      context,
      buffer("reference"),
      buffer("out"),
      m * gemvCase.n * 4,
    );
    return identicalFields(words);
  };
  /** Read afresh each time: the signal is aborted from outside while the case awaits the GPU. */
  const abortRequested = (): boolean => hooks.signal?.aborted === true;
  /**
   * The kernels of one plan through the module doc's steps 1–5, or nothing when an abort stopped
   * it. Every params buffer it made is released before it returns.
   */
  const measurePlan = async (
    plan: GemvMeasurementPlan,
    m1Group: GemvM1Group | undefined,
  ): Promise<{ rows: GemvSweepRow[]; summary: GemvSweepCaseSummary } | undefined> => {
    const started = performance.now();
    const { m } = plan;
    const label = `${gemvCase.id} M=${m}`;
    const kernels: Kernel[] = [plan.baseline, ...plan.candidates].map((candidate, index) => ({
      candidate,
      head: rowHead(gemvCase, m, candidate, index === 0),
      output: index === 0 ? "reference" : "out",
      state: { kind: "planned" },
    }));
    const [baseline] = kernels;
    // The kernels M = 1 leaves out: their row keeps the kernel and its launch, which take no GPU work
    for (const kernel of kernels.slice(1)) {
      if (gemvCarriedFromM1(m, kernel.candidate, plan.profile, m1Group)) continue;
      try {
        const program = gemvProgram(gemvCase, m, kernel.candidate.choice);
        kernel.head = { ...kernel.head, kernel: gemvProgramKernelId(program) };
        kernel.state = { kind: "skipped", launch: gemvLaunch(program, kernel.output, limit) };
      } catch (cause) {
        kernel.state = { kind: "failed", cause };
      }
    }
    const skipped = kernels.filter((kernel) => kernel.state.kind === "skipped").length;
    const prepared: ReadyLaunch[] = [];
    try {
      // 1. Every pipeline first (compiling idles the device)
      hooks.onProgress?.(
        `${label}: preparing ${kernels.length - skipped} kernels${skipped === 0 ? "" : ` (${skipped} skipped: not adoptable from M = 1)`}`,
      );
      for (const kernel of kernels) {
        if (abortRequested()) return undefined;
        if (gpu.lost !== undefined) break;
        if (kernel.state.kind !== "planned") continue;
        try {
          const program = gemvProgram(gemvCase, m, kernel.candidate.choice);
          kernel.head = { ...kernel.head, kernel: gemvProgramKernelId(program) };
          const launch = gemvLaunch(program, kernel.output, limit);
          const ready = await prepareLaunch(context, launch, resources);
          prepared.push(ready);
          kernel.state = { kind: "prepared", launch, ready };
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
      }
      // 2. The warm start, on the baseline unless it failed (then on the next kernel that did not)
      for (const kernel of kernels) {
        if (abortRequested()) return undefined;
        const { state } = kernel;
        if (gpu.lost !== undefined) break;
        if (state.kind !== "prepared") continue;
        hooks.onProgress?.(`${label}: warm start on ${kernel.head.candidate}`);
        try {
          await warmStart(context, state.ready);
          break;
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
      }
      // 3. The quick looks, in an order that does not follow the plan's, and the screening
      if (settings.quickLookMargin > 0) {
        const looked = quickLookOrder(kernels, label);
        for (const [position, kernel] of looked.entries()) {
          if (abortRequested()) return undefined;
          const { state } = kernel;
          if (gpu.lost !== undefined) break;
          if (state.kind !== "prepared") continue;
          hooks.onProgress?.(`${label}: quick look ${position + 1}/${looked.length}`);
          try {
            kernel.state = { ...state, quick: await quickLook(context, state.ready) };
          } catch (cause) {
            kernel.state = { kind: "failed", cause };
          }
        }
        const goesOn = quickLookGoesOn(
          kernels.map((kernel) => ({
            name: kernel.head.candidate,
            class: kernel.head.class,
            ...(kernel.state.kind === "prepared" && kernel.state.quick?.perDispatch !== undefined
              ? { perDispatch: kernel.state.quick.perDispatch }
              : {}),
            reference: kernel.head.isBaseline || kernel.head.candidate === plan.profile.reorder,
          })),
          settings.quickLookMargin,
          // A winner is picked among the fastest; a regression is read against the default's choice
          (kernelClass) =>
            kernelClass === "reorder" && gemvGroupRole(m, "reorder") === "no-regression"
              ? { against: "reference", reference: plan.profile.reorder }
              : { against: "fastest" },
        );
        kernels.forEach((kernel, index) => {
          const { state } = kernel;
          if (state.kind !== "prepared" || goesOn[index] === true) return;
          kernel.state = {
            kind: "screened",
            launch: state.launch,
            ...(state.quick === undefined ? {} : { quick: state.quick }),
          };
        });
      }
      // 4. Every kernel's calibration and check
      for (const [index, kernel] of kernels.entries()) {
        if (abortRequested()) return undefined;
        const { state } = kernel;
        if (gpu.lost !== undefined) break;
        if (state.kind !== "prepared") continue;
        hooks.onProgress?.(
          `${label}: calibrating ${kernel.head.candidate} (${index + 1}/${kernels.length})`,
        );
        try {
          await runLaunchOnce(context, state.ready, buffer(kernel.output));
          const reps = await calibrate(context, state.ready, state.quick?.reps);
          const checks = await checksOf(kernel, m, baseline);
          kernel.state = { ...state, kind: "calibrated", reps, checks, samples: [] };
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
      }
      // 5. The rounds, interleaved
      const timed = kernels.filter((kernel) => kernel.state.kind === "calibrated");
      for (
        let round = 0;
        round < settings.rounds && timed.length > 0 && gpu.lost === undefined;
        round += 1
      ) {
        hooks.onProgress?.(`${label}: round ${round + 1}/${settings.rounds}`);
        for (const kernel of roundOrder(timed, round, settings.rounds)) {
          if (abortRequested()) return undefined;
          const { state } = kernel;
          if (gpu.lost !== undefined) break;
          if (state.kind !== "calibrated") continue;
          try {
            state.samples.push(await measurePass(context, state.ready, state.reps));
          } catch (cause) {
            kernel.state = { kind: "failed", cause };
          }
        }
      }
    } finally {
      for (const ready of prepared) ready.params.destroy();
    }
    /** A kernel's row without the comparison with the baseline, and the checks it read. */
    const measuredOf = (kernel: Kernel): { row: GemvSweepRow; checks: CheckFields } => {
      const { state } = kernel;
      if (state.kind === "failed") return { row: errorRow(kernel.head, state.cause), checks: {} };
      if (state.kind === "screened") {
        return { row: screenedRow(kernel.head, state.launch, state.quick), checks: {} };
      }
      if (state.kind === "skipped")
        return { row: skippedRow(kernel.head, state.launch), checks: {} };
      if (state.kind !== "calibrated" || state.samples.length < settings.rounds) {
        // Only a lost device stops a kernel short of its rounds without a failure of its own
        const lost = new Error(`device lost: ${gpu.lost?.message ?? "no loss reported"}`);
        return { row: errorRow(kernel.head, lost), checks: {} };
      }
      return {
        row: timedRow(
          kernel.head,
          state.launch,
          roundsTiming(state.reps, state.samples),
          state.quick,
        ),
        checks: state.checks,
      };
    };
    const first = measuredOf(baseline);
    const baselineRow =
      first.row.error === undefined ? { ...first.row, ...first.checks } : first.row;
    const candidateRows = kernels.slice(1).map((kernel) => {
      const { row, checks } = measuredOf(kernel);
      // Without the baseline's time there is no ratio, and no words or band are compared with it;
      // a screened or skipped row has neither
      if (
        row.error !== undefined ||
        row.screened === true ||
        row.skipped === true ||
        baselineRow.error !== undefined
      ) {
        return row;
      }
      return { ...row, ...speedFields(baselineRow, row), ...checks };
    });
    return {
      rows: [baselineRow, ...candidateRows],
      summary: {
        caseId: gemvCase.id,
        m,
        elapsedMs: performance.now() - started,
        profile: plan.profile,
        ...(baselineRow.error === undefined && baseline.state.kind === "calibrated"
          ? spreadOf(baseline.state.samples)
          : {}),
        screened: candidateRows.filter((row) => row.screened === true).length,
        skipped: candidateRows.filter((row) => row.skipped === true).length,
      },
    };
  };
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
    // The case's M = 1 group of this run, once measured (the plans go up in row count). A group
    // whose baseline has no time reads nothing (the candidates' checks need the baseline, and a lost
    // device leaves every row a failure), so it leaves no candidate out of the later row counts.
    let m1Group: GemvM1Group | undefined;
    for (const plan of plans) {
      const measured = abortRequested() ? undefined : await measurePlan(plan, m1Group);
      if (measured === undefined) {
        aborted = true;
        break;
      }
      for (const row of measured.rows) emit(row);
      summaries.push(measured.summary);
      const baselineTimed = measured.rows.some(
        (row) => row.isBaseline && row.perDispatch !== undefined,
      );
      if (plan.m === 1 && baselineTimed) m1Group = { ...measured.summary, rows: measured.rows };
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
  readonly selection: GemvSelection;
  readonly rounds: number;
  readonly quickLookMargin: number;
  readonly cases: readonly GemvCase[];
};

/** The first value given twice, if any. */
const givenTwice = <T>(values: readonly T[]): T | undefined =>
  values.find((value, index) => values.indexOf(value) !== index);

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
  const quickLookMargin = quickLookMarginOf(options.quickLookMargin);
  const cases = GEMV_CASES.filter(
    (gemvCase) => caseIds.length === 0 || caseIds.includes(gemvCase.id),
  );
  const rows = options.rows === undefined || options.rows.length === 0 ? undefined : options.rows;
  if (rows !== undefined) {
    const twice = givenTwice(rows);
    if (twice !== undefined) throw new Error(`Row count ${twice} is given twice in rows`);
    const measured = [
      ...new Set(cases.flatMap((gemvCase) => gemvRowCountsIn(gemvCase, candidateSet))),
    ];
    const foreign = rows.find((m) => !measured.includes(m));
    if (foreign !== undefined) {
      throw new Error(
        `Row count ${foreign} is not one the ${candidateSet} set measures (${measured.join(", ")})`,
      );
    }
  }
  const candidates =
    options.candidates === undefined || options.candidates.length === 0
      ? undefined
      : options.candidates;
  if (candidates !== undefined) {
    const twice = givenTwice(candidates);
    if (twice !== undefined) throw new Error(`Candidate ${twice} is given twice in candidates`);
    // Device-free: a name the device's gates leave out (the subgroup form, a grid over its limit)
    // is a candidate of the set all the same, and simply not measured on that device
    const names = new Set(
      cases.flatMap((gemvCase) =>
        gemvRowCountsIn(gemvCase, candidateSet)
          .filter((m) => rows === undefined || rows.includes(m))
          .flatMap((m) =>
            gemvKernelsAt(gemvCase, m, candidateSet)
              .slice(1)
              .map(({ name }) => name),
          ),
      ),
    );
    const foreign = candidates.find((name) => !names.has(name));
    if (foreign !== undefined) {
      throw new Error(
        `Candidate ${foreign} is no candidate of the ${candidateSet} set at any planned case and row count`,
      );
    }
  }
  return {
    candidateSet,
    caseIds,
    selection: {
      ...(rows === undefined ? {} : { rows }),
      ...(candidates === undefined ? {} : { candidates }),
    },
    rounds,
    quickLookMargin,
    cases,
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
  /** The row counts and candidates it was narrowed to (absent: all). */
  readonly selection: GemvSelection;
  readonly rounds: number;
  readonly quickLookMargin: number;
  readonly tolerance: Tolerance;
  readonly deviceLost: GemvSweepReport["deviceLost"];
  /** The planned cases. */
  readonly cases: readonly GemvCase[];
  readonly summaries: readonly GemvSweepCaseSummary[];
  readonly rows: readonly GemvSweepRow[];
};

/** The record of a sweep (pure: each case's fingerprint is computed from the case). */
export const gemvSweepReport = (parts: GemvSweepParts): GemvSweepReport => ({
  format: GEMV_SWEEP_FORMAT,
  date: parts.date,
  startedAt: parts.startedAt,
  aborted: parts.aborted,
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
    ...(parts.selection.rows === undefined ? {} : { rows: parts.selection.rows }),
    ...(parts.selection.candidates === undefined ? {} : { candidates: parts.selection.candidates }),
    rounds: parts.rounds,
    quickLookMargin: parts.quickLookMargin,
    targetPassMs: TARGET_PASS_MS,
    maxReps: SWEEP_MAX_REPS,
    warmStart: {
      minNs: WARM_START_MIN_NS,
      maxNs: WARM_START_MAX_NS,
      maxPasses: WARM_START_MAX_PASSES,
      stablePasses: WARM_START_PASSES,
      tolerance: WARM_START_TOLERANCE,
    },
    recalibrateBelow: RECALIBRATE_BELOW,
    calibrationAttempts: CALIBRATION_ATTEMPTS,
    wallTimingNote: WALL_TIMING_NOTE,
    tolerance: { atol: parts.tolerance.atol, rtol: parts.tolerance.rtol },
    notMeasured: GEMV_SWEEP_NOT_MEASURED,
  },
  deviceLost: parts.deviceLost,
  cases: parts.cases.map((gemvCase): GemvSweepCaseEntry => ({
    id: gemvCase.id,
    storage: gemvCase.storage,
    ...(gemvCase.group === undefined ? {} : { group: gemvCase.group }),
    n: gemvCase.n,
    k: gemvCase.k,
    rows: gemvRowCountsIn(gemvCase, parts.candidateSet).filter(
      (m) => parts.selection.rows === undefined || parts.selection.rows.includes(m),
    ),
    source: gemvCase.source,
    fingerprint: gemvCaseFingerprint(gemvCase),
  })),
  summaries: parts.summaries,
  rows: parts.rows,
});

/**
 * Acquires a dedicated device (with the subgroup features where the device runs them, and
 * `timestamp-query` where the adapter lists it), sweeps the chosen cases and returns the record
 * (`karume-gemv-sweep/4`). The device is destroyed when done (also on abort or failure). Per-row
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
          gemvMeasurementPlans(gemvCase, plan.candidateSet, candidateGates, plan.selection),
          { rounds: plan.rounds, tolerance, quickLookMargin: plan.quickLookMargin },
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
      selection: plan.selection,
      rounds: plan.rounds,
      quickLookMargin: plan.quickLookMargin,
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
