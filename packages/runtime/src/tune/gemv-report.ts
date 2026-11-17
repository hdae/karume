/**
 * The GEMV sweep's record (`karume-gemv-sweep/2`) and the pure pieces it is assembled from. Its own
 * format: its rows are kernels of the GEMV family at a case and a row count, not geometries of one
 * kernel, so they share no row shape with the geometry sweep's record.
 *
 * `/2` replaced `/1` when the rounds became interleaved across the kernels of a row count: a `/1`
 * summary held the baseline measured once more at the end of the row count (`baselineRepeat`, a
 * drift ratio), a `/2` summary holds the spread of the baseline's own rounds, and the settings name
 * the warm start and the calibration check instead of the per-kernel warm-up.
 */

import type { LinearGemvParallelLanes } from "../kernels/linear-gemv.ts";
import type { LinearGemvChoice } from "../runtime/linear-strategies.ts";
import type { GemvCandidateClass, GemvCandidateSet } from "./gemv-candidates.ts";
import type { GemvCase, GemvSweepStorage } from "./gemv-cases.ts";
import type { ReportAdapter, TimingUnit } from "./report.ts";

export const GEMV_SWEEP_FORMAT = "karume-gemv-sweep/2";

/**
 * A candidate's choice as the record writes it (a `LinearGemvChoice` without its unset fields): the
 * member of the GEMV family, the lane count of the parallel forms, the column form and the row form.
 */
export type GemvSweepChoice = {
  readonly id: LinearGemvChoice["id"];
  readonly storage: GemvSweepStorage;
  readonly groupSize?: number;
  readonly lanes?: LinearGemvParallelLanes;
  readonly variant: { readonly cols: number; readonly unroll: number };
  readonly rowsVariant?: { readonly cols: number; readonly unroll: number; readonly rows: number };
};

/** One measured kernel: a candidate, or the baseline, of a case at a row count. */
export type GemvSweepRow = {
  readonly caseId: string;
  /** The row count M. */
  readonly m: number;
  /** The candidate's name (`c32u4`, `c32u4r8`, `parallel-l4`, `staged-l32`, `subgroup32-l8`). */
  readonly candidate: string;
  readonly class: GemvCandidateClass;
  /** Whether this is the baseline: what a Session at the level `exact` runs with the default profile. */
  readonly isBaseline: boolean;
  readonly choice: GemvSweepChoice;
  /** The pipeline keys of the program's dispatches (one for every GEMV program). */
  readonly keys?: readonly string[];
  readonly workgroups?: readonly [number, number, number];
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
   * How steady the device was over the rounds: the baseline's largest round ÷ its smallest (rounds
   * clamped to 0 left out; 1 with a single round). Absent when the baseline's row failed or has no
   * positive round.
   */
  readonly baselineSpread?: number;
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

/** The sweep record (`karume-gemv-sweep/2`). */
export type GemvSweepReport = {
  readonly format: typeof GEMV_SWEEP_FORMAT;
  /** When the record was assembled (ISO 8601 — the sweep's end). */
  readonly date: string;
  /** When the sweep started (ISO 8601 — before the device was acquired). */
  readonly startedAt: string;
  /** A cancel stopped the sweep before every planned measurement was made. */
  readonly aborted: boolean;
  /** The fingerprint of the case table (its dispatch-deciding fields). */
  readonly caseTable: string;
  /** The fingerprint of the kernels the rows without an error measured. */
  readonly kernels: string;
  readonly userAgent: string;
  readonly adapter: ReportAdapter;
  readonly gates: GemvSweepGates;
  readonly gpuTiming: {
    readonly feature: boolean;
    readonly unit: TimingUnit;
    /** Every round is a multiple of 100 µs (Chrome's timestamp quantization — unit `ns` only). */
    readonly quantized: boolean;
  };
  readonly settings: {
    readonly candidateSet: GemvCandidateSet;
    /** When narrowed by case id. */
    readonly cases?: readonly string[];
    readonly rounds: number;
    readonly targetPassMs: number;
    readonly maxReps: number;
    /**
     * The warm start before a row count's calibrations (measurement.ts's `WARM_START_*` and
     * harness.ts's `WARM_START_MAX_PASSES`): the time under load it spends at least and at most
     * (ns), the passes it runs at most, and how many consecutive passes must agree within
     * `tolerance` (largest ÷ smallest at most 1 + it).
     */
    readonly warmStart: {
      readonly minNs: number;
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
  /** The planned cases with their sources, each with the row counts the set measures. */
  readonly cases: readonly GemvCase[];
  readonly summaries: readonly GemvSweepCaseSummary[];
  readonly rows: readonly GemvSweepRow[];
};

/** A choice as the record writes it (unset fields left out); {@link choiceOfRecord} reads it back. */
export const choiceRecord = (
  choice: LinearGemvChoice,
  storage: GemvSweepStorage,
): GemvSweepChoice => ({
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
});

/** The `LinearGemvChoice` a record's choice stands for. */
export const choiceOfRecord = (choice: GemvSweepChoice): LinearGemvChoice => ({
  id: choice.id,
  storage: choice.storage,
  groupSize: choice.groupSize,
  lanes: choice.lanes,
  variant: choice.variant,
  rowsVariant: choice.rowsVariant,
});
