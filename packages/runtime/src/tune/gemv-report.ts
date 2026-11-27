/**
 * The GEMV sweep's record (`karume-gemv-sweep/4`) and the pure pieces it is assembled from. Its own
 * format: its rows are kernels of the GEMV family at a case and a row count, not geometries of one
 * kernel, so they share no row shape with the geometry sweep's record.
 *
 * `/2` replaced `/1` when the rounds became interleaved across the kernels of a row count: a `/1`
 * summary held the baseline measured once more at the end of the row count (`baselineRepeat`, a
 * drift ratio), a `/2` summary holds the spread of the baseline's own rounds, and the settings name
 * the warm start and the calibration check instead of the per-kernel warm-up. `/3` replaced `/2`
 * when the evidence became per row: a row carries the fingerprint of its own kernel and a case its
 * own (the record-wide `caseTable` and `kernels` are gone), and the quick look adds a row's quick
 * time and its `screened` mark, a summary's count of screened kernels and the settings' margin,
 * candidates and row counts. `/4` replaced `/3` when a row gained a state: a reorder candidate that
 * the same run's M = 1 group of the case leaves unadoptable is not measured at M = 4 and 8, and its
 * row there carries the `skipped` mark, its summary a count of skipped kernels.
 *
 * A record of the `explore` set is a `/4` record: it adds two optional fields no reader needs — the
 * sentinel ids in the settings and, in the M = 1 summary of a case that is no sentinel, the names
 * its group was narrowed to — and holds rows only of the kernels it measured, which the readers take
 * as they take any record that holds part of the candidates. Its skipped rows follow the set's own
 * rule of what M = 1 carries (gemv-candidates.ts's `gemvExploreCarriedFromM1`) and its screened rows
 * its own margin, so the plan of missing measurements counts only the rows it timed. A reader that predates the set refuses
 * such a record by its candidate set.
 *
 * Four more optional fields came with the rule of one steady record (2026-11-28), each additive, so a
 * record that has them is still a `/4` record and one without them (written before) reads as it did:
 * the record's `session` and `steadiness`, a summary's `table` (the device table's choices, which the
 * sweep times besides the set's candidates) and a reference row's `retimedPerDispatch` (its one pass
 * more after the rounds). An older reader misreads nothing: it ignores the fields, and the rows of
 * the table's choices are rows like any other.
 */

import type { LinearGemvParallelLanes, LinearGemvParallelRows } from "../kernels/linear-gemv.ts";
import type { LinearGemvChoice } from "../runtime/linear-strategies.ts";
import {
  type GemvCandidateClass,
  type GemvCandidateSet,
  type GemvKernelChoice,
  isGemmChoice,
} from "./gemv-candidates.ts";
import type { GemvCase, GemvSweepStorage } from "./gemv-cases.ts";
import type { GemvSteadiness } from "./gemv-rule.ts";
import type { ReportAdapter, TimingUnit } from "./report.ts";

export const GEMV_SWEEP_FORMAT = "karume-gemv-sweep/4";

/**
 * A candidate's choice as the record writes it (a `LinearGemvChoice` without its unset fields): the
 * member of the GEMV family, the lane count of the parallel forms, the column form and the row form.
 */
export type GemvSweepFamilyChoice = {
  readonly id: LinearGemvChoice["id"];
  readonly storage: GemvSweepStorage;
  readonly groupSize?: number;
  readonly lanes?: LinearGemvParallelLanes;
  readonly variant: { readonly cols: number; readonly unroll: number };
  readonly rowsVariant?: { readonly cols: number; readonly unroll: number; readonly rows: number };
  /** The rows a thread of the row-blocked parallel form holds (absent for every other kernel). */
  readonly parallelRows?: LinearGemvParallelRows;
};

/**
 * The GEMM under f32 compute as the record writes it: the baseline of a float-stored weight at
 * M ≥ 2, where no sequential GEMV exists. Its tile geometry is not written — the row's `keys` name it
 * and its `kernel` fingerprint covers it.
 */
export type GemvSweepGemmChoice = {
  readonly id: "gemm-f32";
  readonly storage: GemvSweepStorage;
  /** Whether the kernel writes `vec4` outputs. */
  readonly v4: boolean;
};

/** The choice of a row: a member of the GEMV family, or the GEMM as a reference. */
export type GemvSweepChoice = GemvSweepFamilyChoice | GemvSweepGemmChoice;

/** One measured kernel: a candidate, or the baseline, of a case at a row count. */
export type GemvSweepRow = {
  readonly caseId: string;
  /** The row count M. */
  readonly m: number;
  /**
   * The candidate's name (`c32u4`, `c32u4r8`, `parallel-l4`, `staged-l32`, `subgroup32-l8`; `gemm`
   * for the GEMM as the baseline of a float-stored weight at M ≥ 2).
   */
  readonly candidate: string;
  readonly class: GemvCandidateClass;
  /** Whether this is the baseline: what a Session at the level `exact` runs with the default profile. */
  readonly isBaseline: boolean;
  readonly choice: GemvSweepChoice;
  /**
   * The fingerprint of the kernel the row measured (gemv-fingerprint.ts's `gemvProgramKernelId` of
   * its program): a derivation reads the row as current only when its runtime builds the same kernel
   * for the candidate. Absent when the program could not be built.
   */
  readonly kernel?: string;
  /** The pipeline keys of the program's dispatches (one for every program the sweep measures). */
  readonly keys?: readonly string[];
  readonly workgroups?: readonly [number, number, number];
  /**
   * The quick look's time of one dispatch (the last pass of its probe ÷ its dispatches, unit: the
   * record's `gpuTiming.unit`) — absent when the quick look was off or read no time.
   */
  readonly quickPerDispatch?: number;
  /**
   * The quick look did not let the kernel go on (its quick time was beyond the margin of its class's
   * bar): the row has no rounds and no check, and no derivation reads it as evidence.
   */
  readonly screened?: true;
  /**
   * The kernel was not measured: a reorder candidate at a row count the derivation reads only for a
   * regression (M = 4 and 8) that the set's rule of what the same run's M = 1 group of the case
   * carries leaves out — its row there leaves it unadoptable (gemv-candidates.ts's
   * `gemvCarriedFromM1`), or, in the `explore` set, it is also not among the fastest rows that set
   * carries (`gemvExploreCarriedFromM1`). The row has its kernel, its keys and its workgroups, and no
   * quick time, no rounds and no check; it is never a baseline, and no derivation reads it as
   * evidence.
   */
  readonly skipped?: true;
  /** Dispatches stacked in one pass. */
  readonly reps?: number;
  /**
   * The pass's time per round, in round order (unit: the record's `gpuTiming.unit`). Every round
   * times every kernel of the row count once, so round `i` of two rows was taken in the same round.
   */
  readonly rounds?: readonly number[];
  /** The pass's wall clock per round (submit → completion, ns). */
  readonly wallRounds?: readonly number[];
  /** Rounds whose timestamp difference was negative (kept as 0, left out of the min). */
  readonly clampedNegativeSamples?: number;
  /** `min(non-negative rounds) / reps`: the time of one dispatch. */
  readonly perDispatch?: number;
  /**
   * A reference (the baseline, the default profile's choice at `reorder` and the device table's
   * choices — the summary's `profile` and `table`) timed once more after the rounds: that one pass ÷
   * `reps`, in the record's unit. Absent for any other row, for a reference whose rounds did not all
   * finish or whose extra pass was clamped, and in a record written before the field existed.
   */
  readonly retimedPerDispatch?: number;
  readonly wallNsPerDispatch?: number;
  /** The baseline's `perDispatch` ÷ this row's (above 1: faster than the baseline). */
  readonly speedupVsBaseline?: number;
  /** Identical class: whether every output word equals the baseline's. */
  readonly identicalToBaseline?: boolean;
  /** Identical class: the output words that differ from the baseline's. */
  readonly differingWords?: number;
  /** Identical class: the first differing word (u32 values). */
  readonly firstDifference?: {
    readonly index: number;
    readonly baseline: number;
    readonly candidate: number;
  };
  /**
   * Reorder class, and the baseline at a row count with reorder rows: the largest absolute error
   * against the host's f64 oracle.
   */
  readonly maxAbsError?: number;
  /** The largest error in units of the band (up to 1 inside it). */
  readonly bandUse?: number;
  readonly withinBand?: boolean;
  /** Outputs that are NaN or infinite. */
  readonly nonFiniteOutputs?: number;
  readonly error?: string;
};

/** The summary of one case at one row count. */
export type GemvSweepCaseSummary = {
  readonly caseId: string;
  readonly m: number;
  /**
   * The wall clock of this row count of the case (ms): its pipelines, warm start, calibrations and
   * rounds (the case's buffers and the host oracle are made once per case, before its first row
   * count).
   */
  readonly elapsedMs: number;
  /** The candidates the default profile chooses for the shape at each level (by name). */
  readonly profile: { readonly exact: string; readonly reorder: string };
  /**
   * The candidates the device's table (the profile the sweep's device resolved to, the record's
   * `geometryProfile`) chooses for the shape at each level, by name: timed whatever the set or its
   * narrowing, as the references are. The names of `profile` where the table holds the default's
   * choices, or where its choice is a kernel the sweep does not measure (packed activations). Absent
   * in a record written before the field existed.
   */
  readonly table?: { readonly exact: string; readonly reorder: string };
  /**
   * How steady the device was over the rounds: the baseline's largest round ÷ its smallest (rounds
   * clamped to 0 left out; 1 with a single round). Absent when the baseline's row failed or has no
   * positive round.
   */
  readonly baselineSpread?: number;
  /** How many kernels of the row count the quick look screened out (0 with the quick look off). */
  readonly screened: number;
  /**
   * How many kernels of the row count were skipped because the same run's M = 1 group of the case
   * does not carry them (0 at M = 1, and when the run did not measure M = 1).
   */
  readonly skipped: number;
  /**
   * The `explore` set only, at M = 1 of a case that is no sentinel: the candidate names the group was
   * narrowed to (gemv-explore.ts's `gemvExploreSelection` — the winners of the nearest sentinels of
   * its storage and their grid neighbours; the references are measured besides, and a name that is no candidate of
   * the case on the device selects nothing). Absent when the group measured every candidate.
   */
  readonly narrowedTo?: readonly string[];
};

/** A planned case as the record lists it. */
export type GemvSweepCaseEntry = GemvCase & {
  /**
   * The fingerprint of the case's dispatch-deciding fields (gemv-fingerprint.ts's
   * `gemvCaseFingerprint`): a derivation reads the case's rows as current only when its runtime's
   * case of that id has the same.
   */
  readonly fingerprint: string;
};

/** The device facts that decided which candidates exist. */
export type GemvSweepGates = {
  /** The subgroup form was measured (the three capabilities below together). */
  readonly subgroups: boolean;
  readonly features: {
    readonly subgroups: boolean;
    readonly subgroupSizeControl: boolean;
    /** The WGSL language feature `subgroup_id`. */
    readonly subgroupId: boolean;
  };
  /** A candidate whose grid exceeds this was not measured. */
  readonly maxComputeWorkgroupsPerDimension: number;
};

/** What the record says the sweep does not measure. */
export const GEMV_SWEEP_NOT_MEASURED =
  "The packed-activation form of the parallel GEMV is not measured: it reads the int8 codes of the " +
  "quantizing producer (static_quantize), which the sweep does not run.";

/** The sweep record (`karume-gemv-sweep/4`). */
export type GemvSweepReport = {
  readonly format: typeof GEMV_SWEEP_FORMAT;
  /** When the record was assembled (ISO 8601 — the sweep's end). */
  readonly date: string;
  /** When the sweep started (ISO 8601 — before the device was acquired). */
  readonly startedAt: string;
  /** A cancel stopped the sweep before every planned measurement was made. */
  readonly aborted: boolean;
  readonly userAgent: string;
  readonly adapter: ReportAdapter;
  /**
   * The page load or process the sweep ran in (`GemvSweepOptions.session`): its random id and the
   * time from its start to the sweep's start (ms), so two records of one session are told from two
   * sessions. Absent when the caller gave none, and in a record written before the field existed.
   */
  readonly session?: { readonly id: string; readonly elapsedMs: number };
  /**
   * The record's class by its own summaries (gemv-rule.ts's `gemvSteadiness`): `steady`, `unsteady`
   * or `unclassified`, with the groups that have a baseline spread and how many of those spread
   * beyond ×1.05 — so the page that ran it can say whether one more run is needed. A statement for
   * the reader of one run: the derivation computes the class from the summaries itself. Absent in a
   * record written before the field existed.
   */
  readonly steadiness?: GemvSteadiness;
  /**
   * The id of the geometry profile the sweep's device resolved to (the embedded table of its
   * adapter, or `default`): the profile a Session on that device runs with. The GEMM that is the
   * baseline of a float-stored case past one row was built with its tile geometry; every other
   * kernel of the sweep reads nothing of it. A record written before the field existed has none (its
   * GEMM rows were built with the default profile's geometry); a row's `kernel` fingerprint is what
   * a derivation compares.
   */
  readonly geometryProfile: string;
  readonly gates: GemvSweepGates;
  readonly gpuTiming: {
    readonly feature: boolean;
    readonly unit: TimingUnit;
    /** Every round is a multiple of 100 µs (Chrome's timestamp quantization — unit `ns` only). */
    readonly quantized: boolean;
  };
  readonly settings: {
    readonly candidateSet: GemvCandidateSet;
    /**
     * The `explore` set only: the sentinels of the whole case table, by id (gemv-explore.ts's
     * `gemvSentinels` — the same whatever the sweep was narrowed to): the cases measured with every
     * candidate, before the others.
     */
    readonly sentinels?: readonly string[];
    /** When narrowed by case id: the cases in measuring order (the sentinels first for `explore`). */
    readonly cases?: readonly string[];
    /** When narrowed by row count (of the set's). */
    readonly rows?: readonly number[];
    /**
     * When narrowed by candidate name: the candidates measured besides the references (the baseline
     * and the default profile's choice at `reorder`, measured either way).
     */
    readonly candidates?: readonly string[];
    readonly rounds: number;
    /**
     * The quick look's margin (harness.ts's protocol rule 4 — measurement.ts's `QUICK_LOOK_MARGIN`
     * unless the caller gave another; 0: the quick look was off).
     */
    readonly quickLookMargin: number;
    readonly targetPassMs: number;
    readonly maxReps: number;
    /**
     * The warm start before a row count's calibrations (measurement.ts's `WARM_START_*` and
     * harness.ts's `WARM_START_MAX_PASSES`): the time under load it spends at least and at most
     * (ns), the passes it runs at most, and how many consecutive passes must agree within
     * `tolerance` (largest ÷ smallest at most 1 + it). `minNs` is the floor of a case's first row
     * count, `loadedMinNs` the floor of a row count that follows one of the same case that timed a
     * kernel (`WARM_START_LOADED_MIN_NS`). A record written before the field existed has no
     * `loadedMinNs`; no reader needs it.
     */
    readonly warmStart: {
      readonly minNs: number;
      readonly loadedMinNs: number;
      readonly maxNs: number;
      readonly maxPasses: number;
      readonly stablePasses: number;
      readonly tolerance: number;
    };
    /**
     * A calibration whose pass is below this share of `targetPassMs` is done again after a warm
     * start, up to `calibrationAttempts` calibrations (measurement.ts's `RECALIBRATE_BELOW`).
     */
    readonly recalibrateBelow: number;
    readonly calibrationAttempts: number;
    readonly wallTimingNote: string;
    /** The band of `linear` the reorder class is held to. */
    readonly tolerance: { readonly atol: number; readonly rtol: number };
    readonly notMeasured: string;
  };
  readonly deviceLost: { readonly reason: string; readonly message: string } | null;
  /**
   * The planned cases with their sources and fingerprints, each with the row counts the set (and the
   * narrowing by row count) measures.
   */
  readonly cases: readonly GemvSweepCaseEntry[];
  readonly summaries: readonly GemvSweepCaseSummary[];
  readonly rows: readonly GemvSweepRow[];
};

/** A choice as the record writes it (unset fields left out); {@link choiceOfRecord} reads it back. */
export const choiceRecord = (
  choice: GemvKernelChoice,
  storage: GemvSweepStorage,
): GemvSweepChoice => {
  if (isGemmChoice(choice)) {
    if (choice.id !== "gemm-f32") {
      throw new Error(`The GEMV sweep measures the GEMM under f32 compute only (${choice.id})`);
    }
    return { id: choice.id, storage, v4: choice.v4 };
  }
  return {
    id: choice.id,
    storage,
    ...(choice.groupSize === undefined ? {} : { groupSize: choice.groupSize }),
    ...(choice.lanes === undefined ? {} : { lanes: choice.lanes }),
    variant: { cols: choice.variant.cols, unroll: choice.variant.unroll },
    ...(choice.rowsVariant === undefined
      ? {}
      : {
          rowsVariant: {
            cols: choice.rowsVariant.cols,
            unroll: choice.rowsVariant.unroll,
            rows: choice.rowsVariant.rows,
          },
        }),
    ...(choice.parallelRows === undefined ? {} : { parallelRows: choice.parallelRows }),
  };
};

/** The choice a record's choice stands for. */
export const choiceOfRecord = (choice: GemvSweepChoice): GemvKernelChoice =>
  choice.id === "gemm-f32"
    ? { id: choice.id, compute: "f32", v4: choice.v4, groupSize: undefined }
    : {
        id: choice.id,
        storage: choice.storage,
        groupSize: choice.groupSize,
        lanes: choice.lanes,
        variant: choice.variant,
        rowsVariant: choice.rowsVariant,
        ...(choice.parallelRows === undefined ? {} : { parallelRows: choice.parallelRows }),
      };
