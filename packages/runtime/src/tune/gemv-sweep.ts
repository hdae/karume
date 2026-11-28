/**
 * The GEMV sweep (`runGemvSweep` of `@karume/runtime/tune`): measures the GEMV choices of `linear` —
 * which sequential variant, how many rows per row block, which lane count of the parallel GEMV,
 * staged or not, subgroup or not — on a dedicated device, case by case, and returns the record
 * (`karume-gemv-sweep/4`). It derives no table and changes no choice.
 *
 * Per case and row count it measures the baseline (what a Session at the level `exact` runs with
 * the default profile's GEMV choices — where that is the GEMM, a float-stored case past one row, with
 * the tile geometry of the profile the sweep's device resolves to, as a Session on it would; the
 * record names that profile), the default profile's choice at `reorder`, the choices of the device's
 * table at both levels (the profile it resolves to: the table a derivation for this adapter keeps
 * unless a challenger beats its choice in the same group) and the candidates of the set
 * (gemv-candidates.ts — all, or those the caller names) together, through harness.ts's primitives
 * (its measurement protocol). Those four references are measured whatever the set, a narrowing or
 * the case's M = 1 group says, and the quick look does not screen them out. A case's row counts are measured in ascending order, so at M = 4 and 8
 * the run has the case's M = 1 group (when it measured it): a reorder candidate whose row there
 * leaves it unadoptable is skipped (gemv-candidates.ts's `gemvCarriedFromM1`; the `explore` set's
 * narrower `gemvExploreCarriedFromM1`) — its program and fingerprint are built, it takes none of
 * the steps below, and its row says `skipped`. There, in every set, the quick look does not screen
 * out a plain or staged row whose row-blocked form goes on (`gemvQuickLookGoesOn`): the derivation
 * reads a row block against that unblocked kernel.
 * 1. the pipelines of every kernel are made first, so no compile leaves the device idle later;
 * 2. a warm start on the baseline (`warmStart`): the device is loaded until its passes are steady —
 *    with at least measurement.ts's `WARM_START_MIN_NS` of load in the case's first row count, and
 *    with `WARM_START_LOADED_MIN_NS` (none) in a row count that follows one of the same case that
 *    timed a kernel, which left the device under load;
 * 3. the quick look (harness.ts's protocol rule 4 — skipped at a margin of 0): the baseline is run
 *    once into its buffer from a zeroed output; then every kernel's calibration probe, with
 *    timestamps, in a shuffled order (`quickLook` — a candidate's from a zeroed output), each
 *    followed by its check on the output the probe left (the comparison of step 4); the references
 *    and the kernels within the margin of their class's bar go on (`quickLookGoesOn` — the bar is
 *    the fastest of the class among the kernels whose check held where the derivation picks a
 *    winner, the subgroup form, which no table row holds, setting none (`gemvQuickLookAdoptable`),
 *    the default's choice at `reorder` where it only checks for a regression:
 *    `gemvGroupRole`), the others become screened rows (a kernel whose check failed goes on or is
 *    screened by its own time; its row records step 4's check);
 * 4. every kernel that goes on, the baseline first, is run once from a zeroed output, calibrated
 *    (`calibrate`, from its quick look's reps: done again after more load when its pass comes out
 *    far below the target) and checked while its output is still in the buffer;
 * 5. the rounds: each times every kernel that went on once, the baseline included, in plan order
 *    rotated so that round r starts at kernel ⌊r · kernels / rounds⌋ (at kernel r mod kernels when
 *    there are more rounds than kernels). A kernel's time is the min of its rounds, GPU timestamps
 *    or the wall clock. Then every reference that finished its rounds is timed once more, in plan
 *    order (its row's `retimedPerDispatch`): a reference whose time moved within the group shows
 *    there, which the rule of one steady record reads. A pass that reads no time (its timestamp
 *    difference negative, clamped, or 0) is marked so (`retimingUnread`), not left without a value:
 *    then the rule reads the reference's group as no evidence, where a record written before the
 *    re-timing existed has nothing to test.
 * Interleaving puts every kernel under the device states the row count went through, so a heating
 * device slows them alike instead of whichever was measured last; how far the state moved is the
 * spread of the baseline's rounds (the summary's `baselineSpread`). There is no per-kernel warm-up
 * (the geometry sweep's protocol rule 3): after the warm start, the calibrations and the rounds keep
 * the device loaded. The inputs are the harness's seeded fills, one buffer set per case at the
 * case's largest row count.
 *
 * The `explore` set (gemv-explore.ts) measures the sentinels of the whole case table first, each
 * with every candidate; every other case measures at M = 1 the references and the candidates its
 * nearest sentinels of its storage predict (`gemvExploreSelection` over the sentinels the caller measured earlier,
 * `sentinelEvidence`, and those this call measured so far), and at M = 4 and 8 what its own M = 1
 * group carries.
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
import { sessionGeometryProfile } from "../runtime/geometry-selection.ts";
import type { LinearProgram } from "../runtime/linear-program.ts";
import {
  type GemvCandidate,
  type GemvCandidateSet,
  type GemvGates,
  type GemvM1Group,
  type GemvMeasurementPlan,
  GEMV_CANDIDATE_SETS,
  gemvCarriedFromM1,
  gemvExploreCarriedFromM1,
  gemvGroupRole,
  gemvKernelsAt,
  gemvMeasurementPlans,
  gemvPlanReferences,
  gemvProgram,
  gemvQuickLookGoesOn,
  gemvRowCountsIn,
  type GemvSelection,
  isGemvCandidateSet,
} from "./gemv-candidates.ts";
import { isFloatStorage, type LinearGemvChoices } from "../kernels/linear-gemv.ts";
import { GEMV_CASES, type GemvCase } from "./gemv-cases.ts";
import {
  type GemvExploreEvidence,
  gemvExploreOrder,
  gemvExploreSelection,
  gemvSentinels,
} from "./gemv-explore.ts";
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
import { gemvSteadiness } from "./gemv-rule.ts";
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
  EXPLORE_QUICK_LOOK_MARGIN,
  QUICK_LOOK_MARGIN,
  quickLookMarginOf,
  RECALIBRATE_BELOW,
  ROUNDS,
  TARGET_PASS_MS,
  WARM_START_LOADED_MIN_NS,
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

/** The phases of a row count's measurement (the module doc's steps 1–5), in their order. */
export type GemvSweepPhase = "prepare" | "warm-start" | "quick-look" | "calibrate" | "rounds";

/**
 * Where a row count's measurement stands: `done` of the `total` units of a phase — a kernel's
 * pipeline (`prepare`), the warm start (one unit), a kernel's quick look and check, a kernel's
 * calibration and check, a kernel's pass in a round (`rounds`: rounds × the kernels timed, and one
 * pass more for each reference timed — its re-timing after the rounds). A phase
 * reports 0 of its total when it begins and each unit when it ends (a unit that failed counts as
 * done). The quick look is absent at a margin of 0, and a phase with nothing to do reports nothing.
 */
export type GemvSweepStep = {
  readonly caseId: string;
  readonly m: number;
  readonly phase: GemvSweepPhase;
  readonly done: number;
  readonly total: number;
};

/** A row count of a case as the case's `plan` event states it. */
export type GemvSweepPlannedRowCount = {
  readonly m: number;
  /**
   * The kernels planned: the baseline, the default profile's choice at `reorder` and the candidates.
   * At M = 4 and 8 those the case's M = 1 group does not carry are skipped and not prepared — that
   * row count's `prepare` step says how many it prepares.
   */
  readonly kernels: number;
};

/**
 * Sweep progress: `started` once the device and the measurement context exist; `plan` — a case
 * starts, with its row counts and their kernels; `step` — a unit of a row count's measurement
 * ({@link GemvSweepStep}); `status` — what is measured now, as a line; `row` — one row is final (the
 * rows of a row count come together once its rounds are done, its baseline first); `case` — a case
 * is measured, with its summaries (one per row count); `deviceLost`.
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
  | {
      readonly kind: "plan";
      readonly caseId: string;
      /** The case's place among the planned cases of the call, from 0. */
      readonly index: number;
      readonly rowCounts: readonly GemvSweepPlannedRowCount[];
    }
  | ({ readonly kind: "step" } & GemvSweepStep)
  | { readonly kind: "status"; readonly message: string }
  | { readonly kind: "row"; readonly row: GemvSweepRow }
  | {
      readonly kind: "case";
      readonly caseId: string;
      readonly summaries: readonly GemvSweepCaseSummary[];
    }
  | { readonly kind: "deviceLost"; readonly reason: string; readonly message: string };

/**
 * The page load or process a sweep runs in ({@link gemvSweepSession}): a random id, made once per
 * page load or process, and when it began (`performance.now()`, ms). Every sweep of the session writes
 * the id and the time since that start into its record, so two records of one session are told from
 * records of two.
 */
export type GemvSweepSession = { readonly id: string; readonly startedAt: number };

/**
 * A new session: a random id (`crypto.randomUUID`) and the time it began (`performance.now()`). Made
 * once by the app's entry point — when the page loads, when the process starts — and handed to every
 * sweep it runs ({@link GemvSweepOptions.session}).
 */
export const gemvSweepSession = (): GemvSweepSession => ({
  id: crypto.randomUUID(),
  startedAt: performance.now(),
});

/** Options of {@link runGemvSweep} (all optional). */
export type GemvSweepOptions = {
  /** The `GPU` instance to use (default `navigator.gpu`). */
  readonly gpu?: GPU;
  /**
   * The session the sweep runs in: the record names its id and the time since its start
   * (`session`). Without it the record has no `session`.
   */
  readonly session?: GemvSweepSession;
  /**
   * `quick` (the default: M = 1, every reorder candidate, the identical class at a small grid around
   * the baseline), `full` (every row count a derivation rule reads — M = 1, 4 and 8 —, the whole
   * grid), `row-blocks` (M = 16 and 64: the row-block heights, which no rule reads yet) or `explore`
   * (the row counts of `full`: the sentinels of the case table with every candidate, first, and the
   * other cases with the candidates the sentinels predict — gemv-explore.ts).
   */
  readonly candidateSet?: GemvCandidateSet;
  /** Narrows by case id (default: every case of the table). */
  readonly cases?: readonly string[];
  /** Narrows by row count, of the set's (default: every row count of the set). */
  readonly rows?: readonly number[];
  /**
   * Narrows by candidate name (`c32u4`, `parallel-l4`, …; default: every candidate of the set). The
   * references — the baseline and the default profile's choice at `reorder` — are measured either
   * way. A name that is no candidate of any planned case and row count throws, and so does the
   * option with the `explore` set, which chooses its candidates itself.
   */
  readonly candidates?: readonly string[];
  /**
   * The `explore` set only: sentinels measured by earlier calls (`gemvSweepExploreEvidence` of their
   * records — the workbench measures one case per call), each a sentinel of the case table at most
   * once and none this call measures. A case that is no sentinel is predicted from these and from the
   * sentinels this call has measured before it. Given with another set, it throws.
   */
  readonly sentinelEvidence?: readonly GemvExploreEvidence[];
  /** Measurement rounds per kernel (default 5 — the min is kept). */
  readonly rounds?: number;
  /**
   * The quick look's margin (harness.ts's protocol rule 4; default measurement.ts's
   * `QUICK_LOOK_MARGIN`, ×1.25, and `EXPLORE_QUICK_LOOK_MARGIN`, ×1.10, for the `explore` set). 0
   * turns the quick look off (every kernel is timed); any other value below 1 throws.
   */
  readonly quickLookMargin?: number;
  /**
   * Abort: stops once the current pass is done. The record holds the row counts measured before; the
   * row count it stopped in leaves no row (its kernels are measured together, so none was finished).
   */
  readonly signal?: AbortSignal;
  /**
   * Each progress event ({@link GemvSweepProgress}). The step callback runs synchronously between
   * timed passes, so it MUST stay cheap (a page should only store the event and paint on its own
   * schedule).
   */
  readonly onProgress?: (progress: GemvSweepProgress) => void;
};

/** The resources of a case, in fill order (the index of an entry seeds its stream). */
const caseResources = (gemvCase: GemvCase, rows: number): ResourceSpec[] => {
  const { storage, group, n, k } = gemvCase;
  const x: ResourceSpec = { name: "x", bytes: rows * k * 4, fill: "f32" };
  const bias: ResourceSpec = { name: "bias", bytes: n * 4, fill: "f32" };
  if (isFloatStorage(storage)) {
    // A float storage has no scale buffer: its words are the elements (k is a multiple of the
    // elements of a 16 B word, so the byte length is whole words).
    return [
      x,
      { name: "w", bytes: n * k * (storage === "f32" ? 4 : 2), fill: storage },
      bias,
      { name: "out", bytes: rows * n * 4, fill: "none" },
      { name: "reference", bytes: rows * n * 4, fill: "none" },
    ];
  }
  const bits = storage === "i2" ? 2 : storage === "i4" ? 4 : 8;
  const scales = storage === "i4" && group !== undefined ? n * (k / group) : n;
  return [
    x,
    // Every bit pattern is a valid i2 / i4 code; i8 takes the fill's [-127, 127].
    { name: "w", bytes: (n * k * bits) / 8, fill: "i8" },
    bias,
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
 * programs of one dispatch without temporaries (every GEMV program is one, and so is the GEMM that
 * is a float-stored case's baseline past one row). A program names the
 * weight's scale among its bindings whatever the storage; it is bound only where the case has one
 * (`weightScale` — false for a float storage, whose WGSL declares no such binding), as the recipe
 * builder binds it.
 */
export const gemvLaunch = (
  program: LinearProgram,
  output: "out" | "reference",
  limit: number,
  weightScale = true,
): Launch => {
  const [dispatch] = program.dispatches;
  if (dispatch === undefined || program.dispatches.length !== 1 || program.temps.length > 0) {
    throw new Error(
      `${program.where}: the GEMV sweep times a program of one dispatch without temporaries`,
    );
  }
  const bindings = dispatch.bindings
    .filter(({ operand }) => weightScale || operand !== "weightScale")
    .map(({ binding, operand }, index) => {
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

/**
 * Whether the derivation could adopt a kernel of a group, as the quick look's fastest bar reads it
 * (harness.ts's `QuickLookKernel.adoptable`): its check held on the output its quick look left, and
 * it is a form a table row holds — the subgroup form is a Session option the derivation only notes,
 * so a faster time of its own must not screen out a plain or staged row. Pure.
 */
export const gemvQuickLookAdoptable = (candidate: GemvCandidate, checkHeld: boolean): boolean =>
  checkHeld && candidate.choice.id !== "gemv-parallel-subgroup32";

const ALL_ROUNDS_NEGATIVE =
  "timestamp negative in every round (only samples clamped to 0, so no min can be taken)";

/** The settings one case is measured with. */
export type GemvCaseSettings = {
  readonly rounds: number;
  /** The band the reorder class is held to (`opTolerance("linear")` in the sweep). */
  readonly tolerance: Tolerance;
  /** The quick look's margin (0: off). */
  readonly quickLookMargin: number;
  /**
   * Which candidates a row count of the `no-regression` role measures, given the case's M = 1 group
   * (default `gemvCarriedFromM1`; the `explore` set's `gemvExploreCarriedFromM1`).
   */
  readonly carriedFromM1?: typeof gemvCarriedFromM1;
  /**
   * Which kernels go on after the quick look, given those its margin lets go on and the case's M = 1
   * group (default `gemvQuickLookGoesOn`, which keeps the base of a row block that goes on — every
   * set's rule).
   */
  readonly quickLookGoesOn?: typeof gemvQuickLookGoesOn;
};

export type GemvCaseHooks = {
  readonly onProgress?: (message: string) => void;
  readonly onStep?: (step: GemvSweepStep) => void;
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
      /** Whether its check held on the output its quick look left (the screening's bar reads it). */
      readonly checkHeld?: boolean;
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
  // The profile a Session on this device runs with: the GEMM reference takes its geometry
  const reference = sessionGeometryProfile(gpu);
  const weightScale = !isFloatStorage(gemvCase.storage);
  const carriedFromM1 = settings.carriedFromM1 ?? gemvCarriedFromM1;
  const keptPastQuickLook = settings.quickLookGoesOn ?? gemvQuickLookGoesOn;
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
   * or its band reading (reorder class) — none when the baseline has no output to compare with
   * (`baselineWritten` false).
   */
  const checksOf = async (
    kernel: Kernel,
    m: number,
    baselineWritten: boolean,
  ): Promise<CheckFields> => {
    if (kernel.head.isBaseline) return m <= oracleRows ? await band("reference", m) : {};
    if (!baselineWritten) return {};
    if (kernel.head.class === "reorder") return await band("out", m);
    const words = await compareBufferWords(
      context,
      buffer("reference"),
      buffer("out"),
      m * gemvCase.n * 4,
    );
    return identicalFields(words);
  };
  /**
   * Whether a kernel's check held on the output in its buffer (the comparison of
   * {@link checksOf}): the baseline's always (its words are the reference, and its band reading is
   * shown, not judged); a candidate's when its words are the baseline's (identical class) or it is
   * inside the band without a non-finite output (reorder class) — not when the baseline has no
   * output to compare with.
   */
  const checkHeldOf = async (
    kernel: Kernel,
    m: number,
    baselineWritten: boolean,
  ): Promise<boolean> => {
    if (kernel.head.isBaseline) return true;
    const checks = await checksOf(kernel, m, baselineWritten);
    return kernel.head.class === "identical"
      ? checks.identicalToBaseline === true
      : checks.withinBand === true && (checks.nonFiniteOutputs ?? 0) === 0;
  };
  /** Read afresh each time: the signal is aborted from outside while the case awaits the GPU. */
  const abortRequested = (): boolean => hooks.signal?.aborted === true;
  /**
   * The kernels of one plan through the module doc's steps 1–5, or nothing when an abort stopped
   * it; its warm start spends at least `warmStartMinNs` of load. Every params buffer it made is
   * released before it returns.
   */
  const measurePlan = async (
    plan: GemvMeasurementPlan,
    m1Group: GemvM1Group | undefined,
    warmStartMinNs: number,
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
    const references = gemvPlanReferences(plan);
    /**
     * The re-timed pass of each reference that finished its rounds: per dispatch, or why it read no
     * time (a negative timestamp difference, clamped to 0, or a difference of 0).
     */
    const retimed = new Map<Kernel, number | "clamped" | "zero">();
    const step = (phase: GemvSweepPhase, done: number, total: number): void =>
      hooks.onStep?.({ caseId: gemvCase.id, m, phase, done, total });
    /** The kernels in `kind` now. */
    const countIn = (kind: KernelState["kind"]): number =>
      kernels.filter((kernel) => kernel.state.kind === kind).length;
    // The kernels M = 1 leaves out: their row keeps the kernel and its launch, which take no GPU work
    for (const kernel of kernels.slice(1)) {
      if (references.has(kernel.candidate.name)) continue;
      if (carriedFromM1(m, kernel.candidate, plan.profile, m1Group)) continue;
      try {
        const program = gemvProgram(gemvCase, m, kernel.candidate.choice, reference);
        kernel.head = { ...kernel.head, kernel: gemvProgramKernelId(program) };
        kernel.state = {
          kind: "skipped",
          launch: gemvLaunch(program, kernel.output, limit, weightScale),
        };
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
      const toPrepare = countIn("planned");
      if (toPrepare > 0) step("prepare", 0, toPrepare);
      let preparedCount = 0;
      for (const kernel of kernels) {
        if (abortRequested()) return undefined;
        if (gpu.lost !== undefined) break;
        if (kernel.state.kind !== "planned") continue;
        try {
          const program = gemvProgram(gemvCase, m, kernel.candidate.choice, reference);
          kernel.head = { ...kernel.head, kernel: gemvProgramKernelId(program) };
          const launch = gemvLaunch(program, kernel.output, limit, weightScale);
          const ready = await prepareLaunch(context, launch, resources);
          prepared.push(ready);
          kernel.state = { kind: "prepared", launch, ready };
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
        preparedCount += 1;
        step("prepare", preparedCount, toPrepare);
      }
      // 2. The warm start, on the baseline unless it failed (then on the next kernel that did not)
      const warming = countIn("prepared") > 0 && gpu.lost === undefined;
      if (warming) step("warm-start", 0, 1);
      for (const kernel of kernels) {
        if (abortRequested()) return undefined;
        const { state } = kernel;
        if (gpu.lost !== undefined) break;
        if (state.kind !== "prepared") continue;
        hooks.onProgress?.(`${label}: warm start on ${kernel.head.candidate}`);
        try {
          await warmStart(context, state.ready, warmStartMinNs);
          break;
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
      }
      if (warming) step("warm-start", 1, 1);
      // 3. The quick looks, in an order that does not follow the plan's, each with its check, and
      // the screening
      if (settings.quickLookMargin > 0) {
        if (abortRequested()) return undefined;
        // The baseline's words first, so that every candidate's check reads against them
        let baselineWritten = false;
        const first = baseline.state;
        if (first.kind === "prepared" && gpu.lost === undefined) {
          try {
            await runLaunchOnce(context, first.ready, buffer(baseline.output));
            baselineWritten = true;
          } catch (cause) {
            baseline.state = { kind: "failed", cause };
          }
        }
        const looked = quickLookOrder(kernels, label);
        const toLook = countIn("prepared");
        if (toLook > 0 && gpu.lost === undefined) step("quick-look", 0, toLook);
        let lookedCount = 0;
        for (const [position, kernel] of looked.entries()) {
          if (abortRequested()) return undefined;
          const { state } = kernel;
          if (gpu.lost !== undefined) break;
          if (state.kind !== "prepared") continue;
          hooks.onProgress?.(`${label}: quick look ${position + 1}/${looked.length}`);
          try {
            // A candidate's probe starts from a zeroed output, so the buffer then holds its own
            // output; the baseline's rewrites the words its first run left
            const quick = await quickLook(
              context,
              state.ready,
              kernel.head.isBaseline ? undefined : buffer(kernel.output),
            );
            const checkHeld = await checkHeldOf(kernel, m, baselineWritten);
            kernel.state = { ...state, quick, checkHeld };
          } catch (cause) {
            kernel.state = { kind: "failed", cause };
          }
          lookedCount += 1;
          step("quick-look", lookedCount, toLook);
        }
        const byMargin = quickLookGoesOn(
          kernels.map((kernel) => ({
            name: kernel.head.candidate,
            class: kernel.head.class,
            ...(kernel.state.kind === "prepared" && kernel.state.quick?.perDispatch !== undefined
              ? { perDispatch: kernel.state.quick.perDispatch }
              : {}),
            reference: kernel.head.isBaseline || references.has(kernel.head.candidate),
            adoptable: gemvQuickLookAdoptable(
              kernel.candidate,
              kernel.state.kind === "prepared" && kernel.state.checkHeld === true,
            ),
          })),
          settings.quickLookMargin,
          // A winner is picked among the fastest; a regression is read against the default's choice
          (kernelClass) =>
            kernelClass === "reorder" && gemvGroupRole(m, "reorder") === "no-regression"
              ? { against: "reference", reference: plan.profile.reorder }
              : { against: "fastest" },
        );
        // The margin lets a kernel without a quick time go on, a skipped or failed one too: only one
        // still prepared goes on, and only such a one may keep the base of its row block
        const going = kernels.map(
          (kernel, index) => kernel.state.kind === "prepared" && byMargin[index] === true,
        );
        const goesOn = keptPastQuickLook(
          m,
          kernels.map(({ candidate }) => candidate),
          going,
          m1Group,
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
      const toCalibrate = countIn("prepared");
      if (toCalibrate > 0 && gpu.lost === undefined) step("calibrate", 0, toCalibrate);
      let calibratedCount = 0;
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
          const checks = await checksOf(kernel, m, baseline.state.kind === "calibrated");
          kernel.state = { ...state, kind: "calibrated", reps, checks, samples: [] };
        } catch (cause) {
          kernel.state = { kind: "failed", cause };
        }
        calibratedCount += 1;
        step("calibrate", calibratedCount, toCalibrate);
      }
      // 5. The rounds, interleaved, then each reference once more
      const timed = kernels.filter((kernel) => kernel.state.kind === "calibrated");
      const retiming = timed.filter((kernel) => references.has(kernel.candidate.name));
      const passes = settings.rounds * timed.length + retiming.length;
      if (passes > 0 && gpu.lost === undefined) step("rounds", 0, passes);
      let passCount = 0;
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
          // A kernel that failed in an earlier round takes no more passes; its places count as done
          if (state.kind === "calibrated") {
            try {
              state.samples.push(await measurePass(context, state.ready, state.reps));
            } catch (cause) {
              kernel.state = { kind: "failed", cause };
            }
          }
          passCount += 1;
          step("rounds", passCount, passes);
        }
      }
      // The re-timing: a reference whose rounds all finished gets one pass more, after every
      // kernel's rounds, so the record shows whether its time moved within the group
      if (retiming.length > 0 && gpu.lost === undefined) {
        hooks.onProgress?.(`${label}: the references once more`);
      }
      for (const kernel of retiming) {
        if (abortRequested()) return undefined;
        const { state } = kernel;
        if (gpu.lost !== undefined) break;
        if (state.kind === "calibrated" && state.samples.length === settings.rounds) {
          try {
            const sample = await measurePass(context, state.ready, state.reps);
            const value = (sample.gpu ?? sample.wallNs) / state.reps;
            retimed.set(kernel, sample.clamped ? "clamped" : value > 0 ? value : "zero");
          } catch (cause) {
            kernel.state = { kind: "failed", cause };
          }
        }
        passCount += 1;
        step("rounds", passCount, passes);
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
      const again = retimed.get(kernel);
      const row = timedRow(
        kernel.head,
        state.launch,
        roundsTiming(state.reps, state.samples),
        state.quick,
      );
      return {
        row:
          again === undefined || row.error !== undefined
            ? row
            : typeof again === "number"
              ? { ...row, retimedPerDispatch: again }
              : { ...row, retimingUnread: again },
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
        table: plan.table,
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
          // A float storage has no scale, and the oracle reads none for it
          scale: weightScale ? streamOf(specs, "wscale") : () => 0,
        });
      } catch (cause) {
        oracle = cause instanceof Error ? cause : new Error(String(cause));
      }
    }
    // The case's M = 1 group of this run, once measured (the plans go up in row count). A group
    // whose baseline has no time reads nothing (the candidates' checks need the baseline, and a lost
    // device leaves every row a failure), so it leaves no candidate out of the later row counts.
    let m1Group: GemvM1Group | undefined;
    // Whether the group just measured timed a kernel: its calibrations and rounds kept the device
    // under load up to the next group, which follows with no idle time. A group that timed nothing
    // (every kernel failed) may have left the device idle, so the next one warms from the floor.
    let underLoad = false;
    for (const plan of plans) {
      // Typed: inferred, it would depend on `measured` through the loop's flow
      const warmStartMinNs: number = underLoad ? WARM_START_LOADED_MIN_NS : WARM_START_MIN_NS;
      const measured = abortRequested()
        ? undefined
        : await measurePlan(plan, m1Group, warmStartMinNs);
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
      underLoad = measured.rows.some((row) => row.perDispatch !== undefined);
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
export type GemvSweepPlan = {
  readonly candidateSet: GemvCandidateSet;
  /**
   * The case ids the sweep was narrowed to, in measuring order (`settings.cases` of its record;
   * empty: every case).
   */
  readonly caseIds: readonly string[];
  readonly selection: GemvSelection;
  readonly rounds: number;
  readonly quickLookMargin: number;
  /** In measuring order: the table's, the sentinels first for the `explore` set. */
  readonly cases: readonly GemvCase[];
  /** The `explore` set only: the sentinels of the whole case table, by id. */
  readonly sentinels?: readonly string[];
  /** The sentinels measured earlier (`GemvSweepOptions.sentinelEvidence`; empty: none). */
  readonly sentinelEvidence: readonly GemvExploreEvidence[];
};

/** The first value given twice, if any. */
const givenTwice = <T>(values: readonly T[]): T | undefined =>
  values.find((value, index) => values.indexOf(value) !== index);

/**
 * The quick look's margin a GEMV sweep of `candidateSet` runs with: `margin` when given (0 or at
 * least 1, else it throws), otherwise the set's default — `EXPLORE_QUICK_LOOK_MARGIN` (×1.10) for
 * `explore`, `QUICK_LOOK_MARGIN` (×1.25) for every other set.
 */
export const gemvSweepQuickLookMargin = (candidateSet: GemvCandidateSet, margin?: number): number =>
  quickLookMarginOf(
    margin,
    candidateSet === "explore" ? EXPLORE_QUICK_LOOK_MARGIN : QUICK_LOOK_MARGIN,
  );

/** Checks the sentinel evidence of an `explore` sweep that measures `cases` (before the GPU is touched). */
const checkSentinelEvidence = (
  evidence: readonly GemvExploreEvidence[],
  sentinels: readonly string[],
  cases: readonly GemvCase[],
): void => {
  const ids = evidence.map(({ gemvCase }) => gemvCase.id);
  const twice = givenTwice(ids);
  if (twice !== undefined) throw new Error(`Sentinel ${twice} is given twice in sentinelEvidence`);
  for (const { gemvCase } of evidence) {
    const table = GEMV_CASES.find(({ id }) => id === gemvCase.id);
    if (table === undefined || !sentinels.includes(gemvCase.id)) {
      throw new Error(
        `sentinelEvidence holds ${gemvCase.id}, which is no sentinel of the case table (${sentinels.join(", ")})`,
      );
    }
    if (gemvCaseFingerprint(gemvCase) !== gemvCaseFingerprint(table)) {
      throw new Error(`sentinelEvidence holds ${gemvCase.id} with another shape than the table's`);
    }
    if (cases.some(({ id }) => id === gemvCase.id)) {
      throw new Error(`sentinelEvidence holds ${gemvCase.id}, which this sweep measures itself`);
    }
  }
};

/** Checks the options and chooses the cases (before the GPU is touched). */
export const planGemvSweep = (options: GemvSweepOptions): GemvSweepPlan => {
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
  const quickLookMargin = gemvSweepQuickLookMargin(candidateSet, options.quickLookMargin);
  const explore = candidateSet === "explore";
  if (explore && options.candidates !== undefined && options.candidates.length > 0) {
    throw new Error(
      "The explore set chooses its candidates from its sentinels: candidates cannot narrow it",
    );
  }
  const sentinelEvidence = options.sentinelEvidence ?? [];
  if (!explore && sentinelEvidence.length > 0) {
    throw new Error(`sentinelEvidence is read by the explore set only, not by ${candidateSet}`);
  }
  const table = GEMV_CASES.filter(
    (gemvCase) => caseIds.length === 0 || caseIds.includes(gemvCase.id),
  );
  const cases = explore ? gemvExploreOrder(table) : table;
  const sentinels = explore ? gemvSentinels(GEMV_CASES) : undefined;
  if (sentinels !== undefined) checkSentinelEvidence(sentinelEvidence, sentinels, cases);
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
    // In measuring order, as the workbench's merged record of the same cases holds them
    caseIds: caseIds.length === 0 ? [] : cases.map(({ id }) => id),
    selection: {
      ...(rows === undefined ? {} : { rows }),
      ...(candidates === undefined ? {} : { candidates }),
    },
    rounds,
    quickLookMargin,
    cases,
    ...(sentinels === undefined ? {} : { sentinels }),
    sentinelEvidence,
  };
};

/**
 * The measurement plans of a case: the set's (narrowed by the plan's row counts and candidates)
 * with the choices of the device's table `table` among them, and, where `narrowedTo` is given (an
 * `explore` case that is no sentinel), the M = 1 group narrowed to those names and the references.
 */
export const gemvCasePlans = (
  gemvCase: GemvCase,
  plan: Pick<GemvSweepPlan, "candidateSet" | "selection">,
  gates: GemvGates,
  narrowedTo: readonly string[] | undefined,
  table: LinearGemvChoices,
): GemvMeasurementPlan[] =>
  gemvMeasurementPlans(gemvCase, plan.candidateSet, gates, plan.selection, table).map((one) => {
    if (narrowedTo === undefined || one.m !== 1) return one;
    const references = gemvPlanReferences(one);
    return {
      ...one,
      candidates: one.candidates.filter(
        ({ name }) => references.has(name) || narrowedTo.includes(name),
      ),
    };
  });

/**
 * The sentinel evidence a record holds: the cases of an `explore` record that are its sentinels,
 * each with the rows of its M = 1 group (none for another set). A later sweep takes it as
 * `GemvSweepOptions.sentinelEvidence` — the workbench measures one case per call.
 */
export const gemvSweepExploreEvidence = (report: GemvSweepReport): GemvExploreEvidence[] => {
  const sentinels = report.settings.sentinels ?? [];
  return report.cases
    .filter(({ id }) => sentinels.includes(id))
    .map(({ fingerprint: _fingerprint, ...gemvCase }) => ({
      gemvCase,
      rows: report.rows.filter((row) => row.caseId === gemvCase.id && row.m === 1),
    }));
};

/** What the record is assembled from. */
export type GemvSweepParts = {
  readonly startedAt: string;
  readonly date: string;
  readonly aborted: boolean;
  readonly userAgent: string;
  readonly adapter: ReportAdapter;
  /** The record's session ({@link GemvSweepReport.session}); absent without one. */
  readonly session?: GemvSweepReport["session"];
  /** The id of the profile the sweep's device resolved to ({@link GemvSweepReport.geometryProfile}). */
  readonly geometryProfile: string;
  readonly gates: GemvSweepGates;
  readonly timing: { readonly feature: boolean; readonly unit: TimingUnit };
  readonly candidateSet: GemvCandidateSet;
  /** The `explore` set only: the sentinels of the whole case table, by id. */
  readonly sentinels?: readonly string[];
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

/**
 * The record of a sweep (pure: each case's fingerprint is computed from the case, and the record's
 * class from its summaries).
 */
export const gemvSweepReport = (parts: GemvSweepParts): GemvSweepReport => ({
  format: GEMV_SWEEP_FORMAT,
  date: parts.date,
  startedAt: parts.startedAt,
  aborted: parts.aborted,
  userAgent: parts.userAgent,
  adapter: parts.adapter,
  ...(parts.session === undefined ? {} : { session: parts.session }),
  steadiness: gemvSteadiness(parts.summaries),
  geometryProfile: parts.geometryProfile,
  gates: parts.gates,
  gpuTiming: {
    feature: parts.timing.feature,
    unit: parts.timing.unit,
    quantized: roundsLookQuantized(parts.rows, parts.timing.unit),
  },
  settings: {
    candidateSet: parts.candidateSet,
    ...(parts.sentinels === undefined ? {} : { sentinels: parts.sentinels }),
    ...(parts.caseIds.length === 0 ? {} : { cases: parts.caseIds }),
    ...(parts.selection.rows === undefined ? {} : { rows: parts.selection.rows }),
    ...(parts.selection.candidates === undefined ? {} : { candidates: parts.selection.candidates }),
    rounds: parts.rounds,
    quickLookMargin: parts.quickLookMargin,
    targetPassMs: TARGET_PASS_MS,
    maxReps: SWEEP_MAX_REPS,
    warmStart: {
      minNs: WARM_START_MIN_NS,
      loadedMinNs: WARM_START_LOADED_MIN_NS,
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
  const session =
    options.session === undefined
      ? undefined
      : { id: options.session.id, elapsedMs: performance.now() - options.session.startedAt };
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
      // What a case that is no sentinel is predicted from: the sentinels measured earlier, then
      // those of this call as each is measured (the plan puts them first)
      const evidence: GemvExploreEvidence[] = [...plan.sentinelEvidence];
      const explore = plan.candidateSet === "explore";
      const carriedFromM1 = explore ? gemvExploreCarriedFromM1 : gemvCarriedFromM1;
      // The choices of the table this device runs with, timed besides the set's candidates
      const table = sessionGeometryProfile(gpu).linearGemv;
      for (const [index, gemvCase] of plan.cases.entries()) {
        if (gpu.lost !== undefined) break;
        if (options.signal?.aborted === true) {
          aborted = true;
          break;
        }
        const sentinel = plan.sentinels?.includes(gemvCase.id) === true;
        const narrowedTo =
          plan.sentinels === undefined || sentinel
            ? undefined
            : gemvExploreSelection(gemvCase, evidence);
        const plans = gemvCasePlans(gemvCase, plan, candidateGates, narrowedTo, table);
        notify?.({
          kind: "plan",
          caseId: gemvCase.id,
          index,
          rowCounts: plans.map(({ m, candidates }) => ({ m, kernels: 1 + candidates.length })),
        });
        const result = await sweepGemvCase(
          context,
          gemvCase,
          plans,
          {
            rounds: plan.rounds,
            tolerance,
            quickLookMargin: plan.quickLookMargin,
            carriedFromM1,
          },
          {
            ...(options.signal === undefined ? {} : { signal: options.signal }),
            onProgress: (message) => notify?.({ kind: "status", message }),
            onStep: (step) => notify?.({ kind: "step", ...step }),
            onRow: (row) => notify?.({ kind: "row", row }),
          },
        );
        const caseSummaries =
          narrowedTo === undefined
            ? result.summaries
            : result.summaries.map((summary) =>
                summary.m === 1 ? { ...summary, narrowedTo } : summary,
              );
        rows.push(...result.rows);
        summaries.push(...caseSummaries);
        if (sentinel) evidence.push({ gemvCase, rows: result.rows.filter(({ m }) => m === 1) });
        notify?.({ kind: "case", caseId: gemvCase.id, summaries: caseSummaries });
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
      ...(session === undefined ? {} : { session }),
      geometryProfile: sessionGeometryProfile(gpu).id,
      gates,
      timing: { feature: gpu.gpuTimingEnabled, unit: context.unit },
      candidateSet: plan.candidateSet,
      ...(plan.sentinels === undefined ? {} : { sentinels: plan.sentinels }),
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
