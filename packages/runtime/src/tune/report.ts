/**
 * The shape of the sweep record (`karume-geometry-sweep/3`) and the pure functions that compare
 * rows with the default geometry.
 *
 * Only the sweep entry point `runGeometrySweep` (`sweep.ts`) assembles the record, so the Deno CLI
 * (`tools/geometry-sweep`, removed in 0.15), the browser page (the sweep tab of `tools/gpu-lab`,
 * removed in 0.15) and user applications get JSON of the same shape — assembled separately, the
 * shapes would silently drift apart at the step that lines up the JSON of M2 / Chrome and B570 /
 * Deno.
 */
import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import type { I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import type { SweepOp } from "./cases.ts";
import type { GeometryProfileFamilyIds } from "./families.ts";
import type { CandidateSet } from "./geometries.ts";

/**
 * The record format. /2 = /1 with `nsPerDispatch` renamed to the unit-neutral `perDispatch`, plus
 * the per-case `cases[]` (re-measurement of the default geometry) and the idle-run and wall-clock
 * notes in `settings`. /3 = /2 with the single `caseSet` and `defaultKernels` replaced by one value
 * per op family, `caseSets` and `defaultKernels` (ADR 0117 addendum 2026-11-12); the generator reads
 * /3 only.
 */
export const REPORT_FORMAT = "karume-geometry-sweep/3";

/**
 * When the ratio of the default geometry's re-measurement (re-measurement ÷ first) is outside this
 * range, it is a sign to re-measure that case (heat or clocks likely moved during the sweep — the
 * default value the ratios rest on is unstable).
 */
export const DEFAULT_DRIFT_RANGE = { min: 0.9, max: 1.1 } as const;

/**
 * Whether the ratio of the default geometry's re-measurement is outside
 * {@link DEFAULT_DRIFT_RANGE}.
 */
export const driftOutOfRange = (driftRatio: number): boolean =>
  !(driftRatio >= DEFAULT_DRIFT_RANGE.min && driftRatio <= DEFAULT_DRIFT_RANGE.max);

/**
 * The note on wall-clock values (put in the JSON's `settings.wallTimingNote` — the same text as the
 * Output section of the README of `tools/geometry-sweep`, removed in 0.15). Wall clock includes the
 * submit-to-completion floor (≈11 ms / pass on Deno), so dividing it per dispatch adds a share of
 * the floor.
 */
export const WALL_TIMING_NOTE =
  'Wall-clock values (unit "wall", wallNsPerDispatch, and TFLOPS under deno-raw-tick) include ' +
  "the submit-to-completion floor (about 11 ms per pass on Deno): TFLOPS derived from them reads " +
  "low, and ratios between geometries shrink toward 1.";

/**
 * The time unit. `ns` = timestamp-query (Chrome / Dawn give ns as the specification says);
 * `deno-raw-tick` = timestamp-query taken on Deno (wgpu does not convert raw ticks —
 * docs/known-issues.md section "Intel Arc B570 (Linux / Vulkan ANV / xe)"; 1 tick = 52.08 ns on
 * B570); `wall` = the wall clock from submit to `onSubmittedWorkDone` (ns) on a device without
 * timestamp-query.
 */
export type TimingUnit = "ns" | "deno-raw-tick" | "wall";

/**
 * The result of one (case, geometry) pair. A failed pair stays with only `error` (the sweep does
 * not stop).
 */
export type SweepRow = {
  readonly caseId: string;
  readonly op: SweepOp;
  readonly shape: string;
  readonly censusCount: number;
  /** The geometry's display name (same spelling as the key fragment). */
  readonly geometry: string;
  readonly geometryParams: GemmGeometry | I8a8Geometry;
  /**
   * Whether this is the default profile's geometry (the ratio baseline). The geometry a Session
   * actually uses is the adapter's profile.
   */
  readonly isDefault: boolean;
  readonly key?: string;
  readonly workgroups?: readonly [number, number, number];
  /** The number of dispatches stacked in one pass. */
  readonly reps?: number;
  /** The whole pass's time per measurement round (unit: the report's `gpuTiming.unit`). */
  readonly rounds?: readonly number[];
  /** The pass's wall clock per measurement round (submit → completion, ns). */
  readonly wallRounds?: readonly number[];
  /**
   * The number of rounds whose timestamp difference was negative. They stay in `rounds` clamped to
   * 0 (not silently dropped) but are excluded from the min candidates of `perDispatch`. If every
   * round is negative the row becomes an `error`.
   */
  readonly clampedNegativeSamples?: number;
  /** `min(non-negative rounds) / reps` (unit: `gpuTiming.unit`). */
  readonly perDispatch?: number;
  /**
   * `min(wallRounds) / reps` (ns — includes the submit-to-completion floor,
   * {@link WALL_TIMING_NOTE}).
   */
  readonly wallNsPerDispatch?: number;
  /**
   * Operations / time of one dispatch (TFLOPS; the TOPS equivalent for i8a8). The time is
   * `perDispatch` when the unit is ns, otherwise `wallNsPerDispatch` (raw ticks are not ns — it
   * includes the floor and so reads low, {@link WALL_TIMING_NOTE}).
   */
  readonly tflops?: number;
  /**
   * The default geometry's `perDispatch` / this row's `perDispatch` (above 1 = faster than the
   * default).
   */
  readonly speedupVsDefault?: number;
  /**
   * The output digest (`outputDigest` of {@link "./harness.ts"} — a two-level SHA-256 over 64 MiB
   * chunks).
   */
  readonly outputSha256?: string;
  /**
   * Whether the output digest matches the default geometry's (a match is expected: f32 by the K
   * reduction-order contract, i8a8 by exact integer arithmetic).
   */
  readonly identicalToDefault?: boolean;
  readonly error?: string;
};

/**
 * The per-case record. `defaultRepeat` is the default geometry **measured once more at the end of
 * the case** (`perDispatch` in the same unit as the rows, `driftRatio` = re-measurement ÷ first). A
 * case whose first default measurement failed is not re-measured (no ratio baseline). If the
 * re-measurement itself fails, `defaultRepeatError`.
 */
export type CaseSummary = {
  readonly caseId: string;
  /**
   * The case's wall clock (ms — planning, input preparation, measuring every geometry and the
   * default re-measurement). A field for reading from the record which case takes the time (ADR
   * 0117 decision 8); the generator does not read it. JSON from before this field lacks it.
   */
  readonly elapsedMs: number;
  readonly defaultRepeat?: { readonly perDispatch: number; readonly driftRatio: number };
  readonly defaultRepeatError?: string;
};

/** The swept adapter (the four `GPUAdapterInfo` fields — empty strings kept as values). */
export type ReportAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/**
 * The sweep record (`karume-geometry-sweep/3` — the input of the generator `derive.ts`, the shape
 * written as JSON).
 */
export type Report = {
  readonly format: typeof REPORT_FORMAT;
  /** When the record was assembled (ISO 8601 — the sweep's end time). */
  readonly date: string;
  /**
   * The sweep's start time (ISO 8601 — before acquiring the device). The difference to `date` is
   * the duration (ADR 0117 decision 8). JSON from before this field lacks it.
   */
  readonly startedAt: string;
  /**
   * Whether an abort (`signal`) stopped the sweep before the planned cases, geometries and default
   * re-measurements were done. A field for telling it apart, from the record alone, from a sweep
   * narrowed by op or case (`settings.ops` / `settings.cases`); the generator's verdicts do not use
   * it (a case cut off midway has no default re-measurement, so the re-measurement-ratio check
   * drops it from the ratio material). A sweep stopped by device lost is recorded in `deviceLost`.
   * JSON from before this field lacks it.
   */
  readonly aborted: boolean;
  /**
   * The case-set version of each op family of the planned cases (ADR 0117 decisions 4 / 8, per
   * family since its addendum 2026-11-12), hashed over the candidate set's own case table restricted
   * to the family (`sweepCaseSetIds(<candidate set>)` for those families). For every family the
   * record has a row of, the generator requires the entry and rejects a record whose value differs
   * from the current runtime's.
   */
  readonly caseSets: GeometryProfileFamilyIds;
  /**
   * The kernel fingerprint of the default table used as the ratio baseline
   * (`DEFAULT_GEOMETRY_PROFILE`, ADR 0117 decisions 4 / 8), per op family of the planned cases over
   * the same case table (`sweepDefaultKernelsIds(<candidate set>)` for those families). For every
   * family the record has a row of, the generator requires the entry and rejects a record whose value
   * differs from the current runtime's (no table is built from a record with a different baseline).
   */
  readonly defaultKernels: GeometryProfileFamilyIds;
  /**
   * The kernel fingerprint of the measured candidates (ADR 0117 decision 8 — fingerprint.ts's
   * `sweepCandidateKernelsId`: the kernels that the (case, geometry) of the non-failed rows build
   * with the `dp4a` variant). The generator re-derives it from the record's rows with the current
   * runtime and rejects a record that differs (no table is built from old measurements on a runtime
   * where only the candidates' WGSL changed — `defaultKernels` protects only the ratio baseline).
   * JSON from before this field lacks it.
   */
  readonly candidateKernels: string;
  /**
   * `navigator.userAgent` in a browser; `{ deno: Deno.version.deno }` from the Deno CLI (removed in
   * 0.15).
   */
  readonly userAgent: string | { readonly deno: string };
  readonly adapter: ReportAdapter;
  /**
   * The id of the embedded geometry profile a Session on this adapter selects automatically
   * (`selectGeometryProfile` — `SessionDiagnostics.geometryProfile` of a Session without an injected
   * table; `"default"` when no embedded table matches). What a reader compares the candidates with
   * as "this adapter's table"; the generator does not read it. JSON from before 2026-11-11 lacks
   * it.
   */
  readonly geometryProfile?: string;
  readonly checkout?: string;
  readonly checkoutDirty?: boolean;
  /** The sha256 of the page's bundle (absent from the CLI, which does not bundle). */
  readonly bundleSha256?: string;
  readonly gpuTiming: {
    /** Whether the device enabled `timestamp-query`. */
    readonly feature: boolean;
    readonly unit: TimingUnit;
    /**
     * Every round's value is a multiple of 100 µs (suspected Chrome timestamp quantization — judged
     * only when the unit is ns).
     */
    readonly quantized: boolean;
  };
  /**
   * Whether i8a8's integer dot product used `dot4I8Packed` (from the list of WGSL language features
   * — the numerics are identical).
   */
  readonly dp4a: boolean;
  readonly settings: {
    /**
     * The candidate set (`quick` / `quick+` / `full`). JSON from before this field has only
     * `quick`.
     */
    readonly candidateSet: CandidateSet;
    /** Compatibility field: true only when `candidateSet === "quick"` (false for `quick+`). */
    readonly quick: boolean;
    readonly ops: readonly SweepOp[];
    /**
     * When narrowed with `--case` (CLI flag, removed in 0.15) or the `cases` option (absent = every
     * case of the op).
     */
    readonly cases?: readonly string[];
    readonly rounds: number;
    readonly targetPassMs: number;
    /**
     * This sweep's upper bound of `reps` (harness.ts's `SWEEP_MAX_REPS`). JSON from before the
     * sweep got its own bound used the 1024 shared with opbench (removed in 0.15) — this field
     * tells which bound a sweep was taken with.
     */
    readonly maxReps: number;
    /** The lower bound of idle runs per geometry (cumulative ns — measurement.ts's `WARMUP_NS`). */
    readonly warmupNs: number;
    /** The minimum number of idle runs per geometry (likewise `WARMUP_MIN_RUNS`). */
    readonly warmupMinRuns: number;
    /** {@link WALL_TIMING_NOTE}. */
    readonly wallTimingNote: string;
  };
  readonly deviceLost: { readonly reason: string; readonly message: string } | null;
  readonly cases: readonly CaseSummary[];
  readonly rows: readonly SweepRow[];
};

/** The reference taken from the default geometry's row (`perDispatch` and digest). */
export type DefaultReference = {
  readonly perDispatch?: number;
  readonly outputSha256?: string;
};

/**
 * Adds the comparison with the default geometry (speed ratio and digest match) to a row. If the
 * default row failed there is nothing to compare, so nothing is added (a silent `false` would read
 * as "the geometry changed the values"). The speed ratio is written only when both the numerator
 * (the default) and the denominator (this row) are positive — if either is 0 (a broken timestamp
 * with a difference of 0) the ratio turns into 0 or infinity, and the generator, which accepts only
 * positive ratios, would reject the whole record (without a ratio, the row merely stops being ratio
 * material).
 */
export const compareToDefault = (row: SweepRow, reference: DefaultReference): SweepRow => {
  if (row.error !== undefined) return row;
  return {
    ...row,
    ...(reference.perDispatch === undefined ||
    !(reference.perDispatch > 0) ||
    row.perDispatch === undefined ||
    !(row.perDispatch > 0)
      ? {}
      : { speedupVsDefault: reference.perDispatch / row.perDispatch }),
    ...(reference.outputSha256 === undefined || row.outputSha256 === undefined
      ? {}
      : { identicalToDefault: reference.outputSha256 === row.outputSha256 }),
  };
};

/** The step to which Chrome rounds timestamps (when WebGPU Developer Features is off). */
export const CHROME_TIMESTAMP_QUANTUM_NS = 100_000;

/**
 * The GPU time summary that {@link looksQuantized} reads (the per-stage summary `StageGpuTiming` of
 * `tools/anima-residency` (removed in 0.15) satisfied this shape too — only the fields the check
 * reads are required).
 */
export type QuantizationSample = {
  readonly totalNs: number;
  readonly entries: readonly { readonly ns: number }[];
};

/**
 * If every key's `ns` is a multiple of 100 µs, Chrome's timestamp quantization is very likely in
 * effect.
 *
 * WHY: under quantization each difference of 1 dispatch = 1 pass is 0 or a multiple of 100 µs, and
 * the sum stays a multiple. In a DiT dominated by short dispatches the breakdown becomes
 * unreliable, so this makes it noticeable in the display (the verdict is for display only — the
 * numbers are not touched). Shared by the sweep and the GPU lab's per-stage Anima summary (GPU lab
 * removed in 0.15) (moved here from `tools/anima-residency/timing.ts`, removed in 0.15, by ADR 0117
 * decision 1).
 */
export const looksQuantized = (gpu: QuantizationSample): boolean =>
  gpu.totalNs > 0 && gpu.entries.every((entry) => entry.ns % CHROME_TIMESTAMP_QUANTUM_NS === 0);

/**
 * Suspected Chrome 100 µs quantization ({@link looksQuantized} reused on per-pass values). Judged
 * only when the unit is ns — raw ticks and wall clock have no quantization step.
 */
export const roundsLookQuantized = (rows: readonly SweepRow[], unit: TimingUnit): boolean => {
  if (unit !== "ns") return false;
  const entries = rows.flatMap((row) => (row.rounds ?? []).map((ns) => ({ ns })));
  return looksQuantized({
    totalNs: entries.reduce((sum, entry) => sum + entry.ns, 0),
    entries,
  });
};
