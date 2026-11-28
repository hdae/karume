/**
 * Pure functions that derive one adapter's **geometry profile** (`GeometryProfile` in the runtime's
 * `src/kernels/geometry-profile.ts`) from sweep records (`karume-geometry-sweep/4` — `report.ts`)
 * (perf-ledger K-71). Independent of Deno's API — {@link deriveGeometryProfile} of the public
 * surface `@karume/runtime/tune` (user applications), the CLI (`main.ts profile` of
 * `tools/geometry-sweep`) and the browser page (the profile tab of `tools/gpu-lab`) — both tools
 * removed in 0.15 — build tables by the same rules (kept separately, tables built by the page or an
 * application and the CLI's output would silently drift apart). Rendering the generated TS
 * registered in the repository lives on the tool side (`tools/geometry-sweep/render.ts`, removed in
 * 0.15) (ADR 0117 decision 1).
 *
 * In the runtime, geometry selection is "a static table of shape × adapter", and run-time
 * autotuning stays banned (the MUST NOT of the ADR 0022 addendum). This generator is **the side
 * that turns the results measured by explicit tuning (the sweep) into a table**; the runtime only
 * picks one embedded table by the adapter's (vendor, architecture, description), or uses an
 * injected table — nothing is measured on the acquire / Session path (ADR 0117 decision 9). A table
 * without `match` (`optIn`; the CLI's `--opt-in`, removed in 0.15) is an injection-only table that
 * is never selected automatically. A table a user built from their own sweep can be injected with
 * `acquireGpu({ geometryProfile })` ({@link geometryProfileJson}).
 *
 * ## Deriving the rules
 *
 * For each profile field (= class), among the geometries that satisfy, in **every case** of that
 * class, "the output matches the default (`identicalToDefault`)" and "the ratio to the default
 * (`speedupVsDefault`) ≥ `minSpeedup` (the CLI's `--min-speedup`, removed in 0.15)", the one with
 * the largest geometric mean across the cases is taken. If there is none, the default (the geometry
 * of the sweep's default row) is written.
 *
 * - Winning in every case is required so that a big win in one case does not hide a regression in
 *   another (a class's geometry applies to every shape of that class).
 * - The mean is geometric because it averages ratios (the arithmetic mean of ×2 and ×0.5 is ×1.25,
 *   and wins and losses would not cancel).
 * - The output match is a check because this is the only measured evidence that f32's K reduction
 *   order does not depend on the geometry (ADR 0022) — a mismatching geometry is never a candidate,
 *   however fast.
 * - If several sweeps measured the same (case, geometry), the ratio is the geometric mean of the
 *   observations, and match and failure are judged over all observations (for overlaying quick and
 *   full — a geometry that mismatched or failed even once is not taken).
 * - A row the quick look screened out (`screened`, harness.ts's protocol rule 4) is no observation:
 *   it has no ratio and no output match, so a geometry screened out in a case that no other sweep
 *   timed it in is not taken (its verdict says so). So is a row the `explore` set skipped
 *   (`skipped` — an earlier case of the class ruled the geometry out in that sweep,
 *   geometry-explore.ts): a record of that set is read by these rules alone, as a record that holds
 *   part of the candidates.
 * - A skeleton form of the f32 GEMM and conv2d's counter decode (perf-ledger K-88) are part of a
 *   geometry: a tile with a form is a candidate of its own, judged by the same rule, and an adopted
 *   one is written into the field with it (`gemmRows` the form, `conv2d` the form and the decode).
 *   The sweep measures them only in the cases whose kernel takes them (harness.ts's
 *   `candidatesFor`), so a class with a case that does not would never adopt one (not measured
 *   there); every linear / matmul / bmm and conv2d case of the sweep's table takes them.
 * - If the geometry of the sweep's default row differs from the current runtime's default, nothing
 *   is generated (fail loudly). The ratio to the default is relative to the default row, so if the
 *   baseline differs from the current default, "faster than the default" does not hold on the
 *   current runtime.
 *
 * Material check: per sweep, a case whose default re-measurement ratio
 * (`cases[].defaultRepeat.driftRatio`) is outside `DEFAULT_DRIFT_RANGE` (report.ts — 0.9–1.1), or
 * whose re-measurement failed or is missing, is dropped from that sweep's ratio observations
 * ({@link deriveProfile}). The ratio baseline (the default value) likely moved during the case, so
 * no geometry's ratio to the default can be trusted. Only the ratios are dropped; output mismatches
 * and failures still count for that case (the output match is a correctness check and does not
 * depend on heat). Every dropped case stays in the verdict lines.
 *
 * Input check (ADR 0115 §4 as amended by ADR 0117 decision 3): only sweeps measured with GPU
 * timestamps (`gpuTiming.unit` is `ns` or `deno-raw-tick`) are accepted. Wall clock includes the
 * submit-to-completion floor, whose error moves per pass, so no bound can be derived from the
 * record alone. Timestamp rounding (Chrome's 100 µs without flags) does not reject a whole sweep;
 * instead, per observation (one row in one sweep) the upper bound of the ratio's error E = e(row) +
 * e(default row) (e = step ÷ shortest round) is computed, and observations above
 * {@link ROUNDING_ERROR_LIMIT} are dropped from that sweep's ratio material
 * ({@link roundingBound}). A case whose default row's e exceeds it has every observation dropped.
 * Dropped observations stay in the verdict lines with their E.
 *
 * The record's matching keys (ADR 0117 decision 8, per op family since its addendum 2026-11-12):
 * for every op family the record has a row of, `caseSets[family]` (the family's case-set version)
 * and `defaultKernels[family]` (the default table's kernel fingerprint over the family's cases) must
 * be present and equal the current runtime's; a missing or differing key rejects the record. If the
 * record carries `candidateKernels` (the kernel fingerprint of the measured candidates), a value
 * that differs from the current runtime's is rejected. `aborted`, `startedAt` and
 * `cases[].elapsedMs` are not read.
 *
 * Class boundaries: linear / matmul / bmm by buckets of the row count M (the boundaries are
 * `PROFILE_GEMM_ROWS_BOUNDS` of the sweep's shape table = ≤ 16 / 17–32 / 33–64 / 65–128 / 129–256 /
 * 257–512 / > 512 — seven buckets refining the default table `GEMM_ROWS_BUCKETS`'s 64 / 512 / ∞,
 * ADR 0116), fused attention's ①QK / ③PV, conv2d's m tile of 64 / 32 rows (the default row's
 * tileM), the i8a8 linear by buckets of M of its own (`I8A8_LINEAR_ROWS_BOUNDS` = ≤ 256 / 257–512 /
 * > 512 — ADR 0115 addendum decision 14), and i8a8's ①QK / ③PV / conv3d. A class with no case in
 * the sweep gets the runtime's default as is (the reason is left in the generated file's comment). A
 * bucket's default is the geometry of the default table's rule covering its range
 * ({@link profileRowsSegments}, {@link i8a8LinearSegments} — the default geometry for every
 * i8a8.linear bucket).
 *
 * ## The second input: attention-bench records (ADR 0117 addendum 2026-11-12)
 *
 * The one field that is not a geometry, `attentionOnline` (the online attention's tiled Vᵀ key axis
 * — perf-ledger K-106), is decided from attention-bench records (`runAttentionRouteBench` with
 * `onlineTiled`, {@link deriveAttentionOnline}): adopted only when every shape of every record is
 * identical and every shape of the bench's set gains at least `minSpeedup` per node, kernel and
 * producer together, from each arm's minimum pass. Without records the field is absent.
 *
 * ## Carrying families over from an existing table (`base` — ADR 0117 addendum 2026-11-12)
 *
 * An op family is **covered** when some record has a row of one of its cases. A covered family is
 * derived from the records as above. With `base`, a family the records do not cover takes `base`'s
 * fields of that family and its provenance entry verbatim, after its `kernels` and `caseSet` are
 * checked against the current runtime ({@link carryOver}); the attention-bench block and
 * `attentionOnline` likewise when no attention-bench records are given. Without `base` (or for a
 * family `base` has no entry for), an uncovered family keeps the default and gets no entry. A
 * base's `linearGemv` (the GEMV choices, which only GEMV sweep records decide —
 * `deriveLinearGemvChoices` in gemv-derive.ts) is carried as it is.
 *
 * ## Shared tables (ADR 0156)
 *
 * With `shared`, one table comes from the records of two or more adapters of one vendor and
 * architecture ({@link deriveSharedProfile}): each adapter's records are judged as above on their
 * own, and a geometry is taken only when it passes the rule on every adapter. With `base`, a family
 * some adapter covers only in part keeps the default instead of being refused. Without `shared`,
 * records of two adapters are refused.
 */
import {
  defaultGemmGeometry,
  GEMM_ROWS_BUCKETS,
  type GemmGeometry,
  gemmTileM,
} from "../kernels/gemm-geometry.ts";
import { gemmMTileGeometry } from "../kernels/gemm.ts";
import { CodegenError } from "../codegen/errors.ts";
import {
  type AttentionOnlineReduce,
  attentionOnlineI8a8Key,
  attentionOnlineKeyWords,
} from "../kernels/attention-online-i8a8.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  GEOMETRY_PROFILE_FAMILIES,
  type GeometryProfile,
  type GeometryProfileFamily,
  type GeometryProfileFamilyProvenance,
} from "../kernels/geometry-profile.ts";
import { defaultI8a8Geometry, type I8a8Geometry } from "../kernels/i8a8-geometry.ts";
import {
  attentionValueQuantization,
  planAttentionOnlineWindows,
} from "../runtime/recipe-builders/attention.ts";
import { ATTENTION_BENCH_FORMAT, onlineDispatchRows } from "./attention-bench.ts";
import { ATTENTION_BENCH_SHAPES, type AttentionBenchShape } from "./attention-shapes.ts";
import {
  caseShape,
  I8A8_LINEAR_ROWS_BOUNDS,
  PROFILE_GEMM_ROWS_BOUNDS,
  SWEEP_CASES,
  type SweepCase,
  SWEEP_OPS,
  type SweepOp,
} from "./cases.ts";
import {
  familiesOf,
  familyEntries,
  type GeometryProfileFamilyIds,
  profileSlotFamily,
} from "./families.ts";
import {
  attentionBenchKernelsId,
  attentionBenchShapeSetId,
  familyCaseSetId,
  familyKernelsId,
  INFINITY_JSON,
  KernelsIdError,
  sweepCandidateKernelsId,
} from "./fingerprint.ts";
import {
  CANDIDATE_SETS,
  conv2dCandidate,
  EVALUATION_SETS,
  evaluationSetLedgerId,
  EXPLORE_SET,
  gemmCandidate,
  i8a8Candidate,
  isSweepSet,
  type SweepSet,
} from "./geometries.ts";
import { GRID_SET } from "./gemm-grid.ts";
import { casePlan } from "./harness.ts";
import { CHROME_TIMESTAMP_QUANTUM_NS, driftOutOfRange, REPORT_FORMAT } from "./report.ts";

/**
 * The default of `minSpeedup` (the CLI's `--min-speedup`, removed in 0.15) (a win below this ratio
 * to the default is not told apart from measurement noise).
 */
export const DEFAULT_MIN_SPEEDUP = 1.05;

/**
 * The threshold of the rounding-error bound E of an observation's ratio (ADR 0117 decision 3). One
 * fifth of the margin (5%) of the ×1.05 adoption threshold, one tenth of the allowed range of the
 * re-measurement ratio (0.9–1.1). For an 80 ms pass E is about 0.25%; only short passes where the
 * reps estimate missed exceed it. One rule — neither an argument nor an option (and no switch to
 * disable the exclusion).
 */
export const ROUNDING_ERROR_LIMIT = 0.01;

/**
 * The adapter the table applies to (the generated `match`). `vendor` / `architecture` /
 * `description` use exact string equality as in the runtime's selection, and `description` can be
 * given only together with `architecture` (the same condition as the runtime's check). `optIn` is
 * an **injection-only** table without `match` — the runtime never picks it automatically; the
 * application looks it up by id and passes it to `acquireGpu({ geometryProfile })` (ADR 0115
 * addendum decision 7).
 */
export type ProfileTarget =
  | {
      readonly optIn?: undefined;
      readonly vendor: string;
      readonly architecture?: string;
      readonly description?: string;
    }
  | {
      readonly optIn: true;
      readonly vendor?: undefined;
      readonly architecture?: undefined;
      readonly description?: undefined;
    };

type Geometry = GemmGeometry | I8a8Geometry;

/**
 * The swept adapter (the record's `adapter` — the four `GPUAdapterInfo` fields, empty strings kept
 * as values).
 */
type SweepAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/** The fields of one sweep row that generation reads. */
type SweepObservation = {
  readonly caseId: string;
  readonly op: SweepOp;
  readonly shape: string;
  readonly geometry: string;
  readonly geometryParams: Geometry;
  readonly isDefault: boolean;
  readonly speedupVsDefault?: number;
  readonly identicalToDefault?: boolean;
  readonly error?: string;
  /**
   * The quick look screened the geometry out in this case (report.ts): no ratio, no output match,
   * never ratio material.
   */
  readonly screened?: true;
  /**
   * The `explore` set did not measure the geometry in this case (report.ts): no ratio, no output
   * match, never ratio material — read as a screened row is.
   */
  readonly skipped?: true;
  /**
   * The pass time per measurement round (unit: `gpuTiming.unit`; negative rounds stay clamped to 0
   * — report.ts). Only the rounding-error bound ({@link roundingBound}) reads it.
   */
  readonly rounds?: readonly number[];
};

/**
 * The default re-measurement at the end of a case (the fields of `cases[]` that generation reads —
 * report.ts's `CaseSummary`, by its field names, so a record's case summary is one as it is).
 */
type SweepCaseRepeat = {
  readonly caseId: string;
  /** The re-measurement; only its ratio, re-measurement ÷ first, is read. */
  readonly defaultRepeat?: { readonly driftRatio?: number };
  /** The re-measurement failure. */
  readonly defaultRepeatError?: string;
};

/** One sweep record (checked at the `unknown` boundary). */
export type SweepSource = {
  /**
   * The path shown in the output and in the regeneration command (relative to the repository root).
   */
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly adapter: SweepAdapter;
  /**
   * The candidate set (`settings.candidateSet` — old records without the field give `quick` /
   * `full` from `settings.quick`), or the `explore` set.
   */
  readonly candidateSet: SweepSet;
  /**
   * The browser that ran the sweep (the record's `userAgent` turned into a string by
   * {@link userAgentText} — material for the `userAgent` of the table's family entries). undefined
   * for records without the field.
   */
  readonly userAgent?: string;
  /**
   * The timestamp quantization step q (same unit as `rounds`). Chrome's 100 µs when
   * `gpuTiming.quantized` is true, 0 (no rounding) when false.
   */
  readonly timestampQuantum: number;
  readonly cases: readonly SweepCaseRepeat[];
  readonly rows: readonly SweepObservation[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const requireString = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} is not a string`);
  return value;
};

const optional = <T>(
  record: Record<string, unknown>,
  key: string,
  accept: (value: unknown) => value is T,
  where: string,
): T | undefined => {
  const value = record[key];
  if (value === undefined) return undefined;
  if (!accept(value))
    throw new Error(`${where}: ${key} has the wrong type (${JSON.stringify(value)})`);
  return value;
};

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";

const isString = (value: unknown): value is string => typeof value === "string";

const isTrue = (value: unknown): value is true => value === true;

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0;

/**
 * The shape of `rounds` (report.ts — negative rounds are kept clamped to 0, so a negative value
 * means a broken record).
 */
const isRounds = (value: unknown): value is readonly number[] =>
  Array.isArray(value) &&
  value.every((entry) => typeof entry === "number" && Number.isFinite(entry) && entry >= 0);

const isSweepOp = (value: unknown): value is SweepOp =>
  typeof value === "string" && (SWEEP_OPS as readonly string[]).includes(value);

const isI8a8Op = (op: SweepOp): boolean =>
  op === "i8a8-linear" || op === "i8a8-attention" || op === "i8a8-conv3d";

/**
 * The optional fields a row's geometry may carry, by op: the i8a8 family the vec4 skeleton
 * (perf-ledger K-85); linear / matmul / bmm the skeleton form and conv2d the form and the counter
 * decode (perf-ledger K-88) — where the profile takes them (`assertGeometryProfile`). The fused
 * attention takes none: its kernels keep the original form, so a row with one would be the tile's
 * own kernel under another name.
 */
const optionalGeometryFields = (op: SweepOp): readonly ("skeleton" | "form" | "decode")[] => {
  switch (op) {
    case "linear":
    case "matmul":
    case "bmm":
      return ["form"];
    case "conv2d":
      return ["form", "decode"];
    case "attention":
      return [];
    case "i8a8-linear":
    case "i8a8-attention":
    case "i8a8-conv3d":
      return ["skeleton"];
  }
};

/**
 * The geometry fields (4 values for the f32 family, 5 including tileK for the i8a8 family) with
 * the optional fields of the row's op ({@link optionalGeometryFields}); a wrong combination of op
 * and fields, or a value a field does not have, fails.
 */
const parseGeometryParams = (value: unknown, op: SweepOp, where: string): Geometry => {
  if (!isRecord(value)) throw new Error(`${where}: geometryParams is not an object`);
  const keys = isI8a8Op(op)
    ? (["regM", "regN", "wgX", "wgY", "tileK"] as const)
    : (["regM", "regN", "wgX", "wgY"] as const);
  const allowed: readonly string[] = [...keys, ...optionalGeometryFields(op)];
  const extra = Object.keys(value).filter((key) => !allowed.includes(key));
  if (extra.length > 0) {
    throw new Error(`${where}: geometryParams has unexpected fields ${extra.join(", ")}`);
  }
  const { skeleton, form, decode } = value;
  if (skeleton !== undefined && skeleton !== "vec4") {
    throw new Error(
      `${where}: geometryParams.skeleton must be "vec4" or absent (${JSON.stringify(skeleton)})`,
    );
  }
  if (form !== undefined && form !== "tb-un" && form !== "tas-tb-un") {
    throw new Error(
      `${where}: geometryParams.form must be "tb-un", "tas-tb-un" or absent (${JSON.stringify(form)})`,
    );
  }
  if (decode !== undefined && decode !== "counter") {
    throw new Error(
      `${where}: geometryParams.decode must be "counter" or absent (${JSON.stringify(decode)})`,
    );
  }
  const params: Record<string, number> = {};
  for (const key of keys) {
    const field = value[key];
    if (typeof field !== "number" || !Number.isSafeInteger(field) || field < 1) {
      throw new Error(
        `${where}: geometryParams.${key} must be a positive integer (${JSON.stringify(field)})`,
      );
    }
    params[key] = field;
  }
  return isI8a8Op(op)
    ? ({
        regM: params.regM,
        regN: params.regN,
        wgX: params.wgX,
        wgY: params.wgY,
        tileK: params.tileK,
        // Absent stays absent (a geometry without it keeps its key and WGSL byte for byte)
        ...(skeleton === undefined ? {} : { skeleton }),
      } satisfies I8a8Geometry)
    : ({
        regM: params.regM,
        regN: params.regN,
        wgX: params.wgX,
        wgY: params.wgY,
        // Absent stays absent, and the fields keep one order: the table's literal is written from
        // this object
        ...(form === undefined ? {} : { form }),
        ...(decode === undefined ? {} : { decode }),
      } satisfies GemmGeometry);
};

const parseRow = (value: unknown, where: string): SweepObservation => {
  if (!isRecord(value)) throw new Error(`${where}: row is not an object`);
  const op = value.op;
  if (!isSweepOp(op)) throw new Error(`${where}: op is none of ${SWEEP_OPS.join(" / ")}`);
  const isDefault = value.isDefault;
  if (!isBoolean(isDefault)) throw new Error(`${where}: isDefault is not a boolean`);
  const speedupVsDefault = optional(value, "speedupVsDefault", isPositiveNumber, where);
  const identicalToDefault = optional(value, "identicalToDefault", isBoolean, where);
  const error = optional(value, "error", isString, where);
  const rounds = optional(value, "rounds", isRounds, where);
  const screened = optional(value, "screened", isTrue, where);
  if (screened === true && isDefault) {
    throw new Error(`${where}: the default row is screened (the default geometry always goes on)`);
  }
  const skipped = optional(value, "skipped", isTrue, where);
  if (skipped === true && isDefault) {
    throw new Error(
      `${where}: the default row is skipped (the default geometry is always measured)`,
    );
  }
  if (skipped === true && screened === true) {
    throw new Error(
      `${where}: the row is both skipped and screened (a skipped row had no quick look)`,
    );
  }
  // reps is only checked: the rounding-error bound e = q ÷ min(round) bounds the relative error of
  // perDispatch (= min ÷ reps) because reps is an integer and carries no error (ADR 0117 decision
  // 3). The value itself does not affect the bound
  optional(value, "reps", isPositiveInteger, where);
  return {
    caseId: requireString(value, "caseId", where),
    op,
    shape: requireString(value, "shape", where),
    geometry: requireString(value, "geometry", where),
    geometryParams: parseGeometryParams(value.geometryParams, op, where),
    isDefault,
    ...(speedupVsDefault === undefined ? {} : { speedupVsDefault }),
    ...(identicalToDefault === undefined ? {} : { identicalToDefault }),
    ...(error === undefined ? {} : { error }),
    ...(rounds === undefined ? {} : { rounds }),
    ...(screened === undefined ? {} : { screened }),
    ...(skipped === undefined ? {} : { skipped }),
  };
};

const parseCaseRepeat = (value: unknown, where: string): SweepCaseRepeat => {
  if (!isRecord(value)) throw new Error(`${where}: not an object`);
  const repeat = value.defaultRepeat;
  if (repeat !== undefined && !isRecord(repeat)) {
    throw new Error(`${where}: defaultRepeat is not an object`);
  }
  const driftRatio =
    repeat === undefined
      ? undefined
      : optional(repeat, "driftRatio", isPositiveNumber, `${where} defaultRepeat`);
  const error = optional(value, "defaultRepeatError", isString, where);
  return {
    caseId: requireString(value, "caseId", where),
    ...(driftRatio === undefined ? {} : { defaultRepeat: { driftRatio } }),
    ...(error === undefined ? {} : { defaultRepeatError: error }),
  };
};

/**
 * The record's candidate set (goes into the table's `provenance.candidateSet` — ADR 0117 decision
 * 4). `settings.candidateSet` is the source of truth; old records without the field (before quick+
 * was introduced) read `quick` / `full` from `settings.quick`. A candidate set or the `explore` set,
 * which sweeps `full`'s candidates (geometries.ts's `EXPLORE_SET`).
 */
const parseCandidateSet = (settings: unknown, where: string): SweepSet => {
  if (!isRecord(settings))
    throw new Error(`${where}: settings is missing (the candidate set cannot be read)`);
  const { candidateSet, quick } = settings;
  if (candidateSet !== undefined) {
    if (typeof candidateSet !== "string" || !isSweepSet(candidateSet)) {
      throw new Error(
        `${where}: settings.candidateSet is none of ${CANDIDATE_SETS.join(" / ")} (${JSON.stringify(
          candidateSet,
        )}) nor ${EXPLORE_SET}`,
      );
    }
    return candidateSet;
  }
  if (!isBoolean(quick)) {
    throw new Error(
      `${where}: settings has neither candidateSet nor quick (the candidate set cannot be read)`,
    );
  }
  return quick ? "quick" : "full";
};

/**
 * The string of the record's `userAgent` (the spelling of a family entry's `userAgent`). A
 * browser record keeps the `navigator.userAgent` string as is; a record of the Deno CLI (removed in
 * 0.15) (`{ deno: <version> }`) becomes `Deno/<version>` — the same spelling as Deno's
 * `navigator.userAgent`, so that an application can compare it with its current
 * `navigator.userAgent`. undefined when the field is absent; any other shape fails (not silently
 * skipped).
 */
const userAgentText = (value: unknown, where: string): string | undefined => {
  if (value === undefined || typeof value === "string") return value;
  if (isRecord(value) && typeof value.deno === "string" && Object.keys(value).length === 1) {
    return `Deno/${value.deno}`;
  }
  throw new Error(
    `${where}: userAgent is neither a string nor { deno: string } (${JSON.stringify(value)})`,
  );
};

/**
 * Compares the record's `candidateKernels` with the current runtime's value (ADR 0117 decision 8).
 * A record without the field is not matched on it. The current value is derived only when the field
 * is present.
 */
const assertRecordedCandidateKernels = (
  record: Record<string, unknown>,
  current: () => string,
  where: string,
): void => {
  const recorded = record.candidateKernels;
  if (recorded === undefined) return;
  if (typeof recorded !== "string") {
    throw new Error(`${where}: candidateKernels is not a string (${JSON.stringify(recorded)})`);
  }
  const now = current();
  if (recorded !== now) {
    throw new Error(
      `${where}: candidateKernels ${recorded} differs from the current runtime's ${now} (the record's kernels of the measured candidates differ from the current runtime, so no table is built` +
        " from it — sweep again with the current runtime)",
    );
  }
};

/**
 * Compares a per-family matching key of the record (`caseSets` / `defaultKernels` — ADR 0117
 * decision 8, per op family since its addendum 2026-11-12) with the current runtime's value, for
 * every family the record has a row of. A missing key throws: the record cannot be matched for that
 * family (only a `/3` writer writes the keys, and it writes one per family it planned). Every own
 * key of the map is checked ({@link recordedFamilyIds}), but only the families with rows are
 * compared with the current runtime. The current value is derived only for the families compared.
 */
const assertRecordedFamilyKeys = (
  record: Record<string, unknown>,
  key: "caseSets" | "defaultKernels",
  families: readonly GeometryProfileFamily[],
  current: (family: GeometryProfileFamily) => string,
  meaning: string,
  where: string,
): void => {
  const recorded = record[key];
  const ids = isRecord(recorded) ? recordedFamilyIds(recorded, key, where) : undefined;
  if (families.length === 0) return;
  if (ids === undefined) {
    throw new Error(
      `${where}: ${key} is not an object (${JSON.stringify(recorded)} — the record has rows of ${families.join(", ")} and cannot be matched against the current runtime; sweep again with the current runtime)`,
    );
  }
  for (const family of families) {
    const value = ids[family];
    if (value === undefined) {
      throw new Error(
        `${where}: ${key}.${family} is missing (the record has rows of family ${family} but no ${meaning} for it, so it cannot be matched against the current runtime — sweep this family again with the current runtime)`,
      );
    }
    const now = current(family);
    if (value !== now) {
      throw new Error(
        `${where}: ${key}.${family} ${value} differs from the current runtime's ${now} (the record's ${meaning} of family ${family} differs from the current runtime, so no table is built` +
          " from it — sweep this family again with the current runtime)",
      );
    }
  }
};

/**
 * The record's per-family map `key` (`caseSets` / `defaultKernels`) with every own key checked: an
 * op family name with a string value. Fails loudly, naming the record, the map and the key — also
 * for families without rows, whose values are not compared ({@link assertRecordedFamilyKeys}), so a
 * misspelt family or a broken value is not carried along unseen.
 */
const recordedFamilyIds = (
  map: Record<string, unknown>,
  key: "caseSets" | "defaultKernels",
  where: string,
): GeometryProfileFamilyIds => {
  const ids: { [Family in GeometryProfileFamily]?: string } = {};
  for (const [name, value] of Object.entries(map)) {
    const family = GEOMETRY_PROFILE_FAMILIES.find((entry) => entry === name);
    if (family === undefined) {
      throw new Error(
        `${where}: ${key}.${name} is not an op family (the families are ${GEOMETRY_PROFILE_FAMILIES.join(", ")})`,
      );
    }
    if (typeof value !== "string") {
      throw new Error(`${where}: ${key}.${family} is not a string (${JSON.stringify(value)})`);
    }
    ids[family] = value;
  }
  return ids;
};

/** The record's `dp4a` (the measured i8a8 variant — read when deriving `candidateKernels`). */
const recordedDp4a = (record: Record<string, unknown>, where: string): boolean => {
  const { dp4a } = record;
  if (!isBoolean(dp4a)) {
    throw new Error(
      `${where}: dp4a is not a boolean (${JSON.stringify(dp4a)} — candidateKernels cannot be derived)`,
    );
  }
  return dp4a;
};

/**
 * Reads a sweep record (the `unknown` boundary — checks only the fields generation reads and fails
 * loudly).
 *
 * MUST: each case has exactly one default row. With no ratio baseline, or two, it is undefined
 * which geometry the ratios were taken against.
 */
export const parseSweepReport = (
  parsed: unknown,
  meta: { readonly path: string; readonly sha256: string },
): SweepSource => {
  const where = meta.path;
  if (!isRecord(parsed)) throw new Error(`${where}: the JSON is not an object`);
  if (parsed.format !== REPORT_FORMAT) {
    throw new Error(`${where}: format is not ${REPORT_FORMAT} (${String(parsed.format)})`);
  }
  // MUST: no table is derived from an evaluation set (perf-ledger K-88): its cases are
  // outside the sweep's case set, so its verdicts would decide a field on shapes the table does not
  // stand for
  const settings = parsed.settings;
  const recordedSet = isRecord(settings) ? settings.candidateSet : undefined;
  const evaluation = EVALUATION_SETS.find((set) => set === recordedSet);
  if (evaluation !== undefined) {
    throw new Error(
      `${where}: settings.candidateSet ${evaluation} is an evaluation set (perf-ledger ${evaluationSetLedgerId(evaluation)}): ` +
        "it measures cases outside the sweep's case set, and no table is derived from it",
    );
  }
  // MUST: no field is derived from the grid set either: it times the grid-fill rule's ladder on cases
  // of its own, and only that rule's derivation reads it
  if (recordedSet === GRID_SET) {
    throw new Error(
      `${where}: settings.candidateSet ${GRID_SET} is the grid set: it times the grid-fill rule's ladder on cases outside the sweep's case set, and only the derivation of that rule reads it`,
    );
  }
  const adapter = parsed.adapter;
  if (!isRecord(adapter)) throw new Error(`${where}: adapter is missing`);
  // MUST: only sweeps measured with GPU timestamps are accepted (ADR 0115 §4, ADR 0117 decision 3).
  // Wall clock includes the submit-to-completion floor, so ratios between geometries shrink toward
  // 1, and the floor moves per pass, so the error cannot be bounded by a known constant as rounding
  // can (no per-row bound can be derived from the record alone)
  const timingFix =
    "the Deno CLI measures with timestamps when the adapter lists timestamp-query" +
    " (unit deno-raw-tick), so measure again on an adapter with timestamp-query; on the Chrome page," +
    " check 'Measure with GPU timestamps' and measure again";
  const gpuTiming = parsed.gpuTiming;
  if (!isRecord(gpuTiming)) {
    throw new Error(`${where}: gpuTiming is missing (the time unit is unknown) — ${timingFix}`);
  }
  const unit = gpuTiming.unit;
  if (unit !== "ns" && unit !== "deno-raw-tick") {
    throw new Error(
      `${where}: gpuTiming.unit is ${JSON.stringify(unit)} (must be ns or deno-raw-tick — ` +
        `wall is the wall clock, where ratios shrink toward 1) — ${timingFix}`,
    );
  }
  // Quantization (Chrome's 100 µs rounding) does not reject a whole sweep — the rounding step q is
  // used for the per-observation error bound (deriveProfile, ADR 0117 decision 3)
  const quantized = gpuTiming.quantized;
  if (!isBoolean(quantized)) {
    throw new Error(
      `${where}: gpuTiming.quantized is not a boolean (${JSON.stringify(quantized)})`,
    );
  }
  // The writer (roundsLookQuantized in report.ts) judges quantization only for unit ns. Applying an
  // ns step to raw ticks would give a different bound, so a record with a broken combination fails
  if (quantized && unit !== "ns") {
    throw new Error(
      `${where}: gpuTiming.quantized is true but the unit is ${unit} (quantization is judged only for ns)`,
    );
  }
  const userAgent = userAgentText(parsed.userAgent, where);
  // The default re-measurements (cases[]) are read by the material check — a record without them
  // would mean "no case has a re-measurement", so fail instead of silently dropping every case
  if (!Array.isArray(parsed.cases)) {
    throw new Error(
      `${where}: cases is not an array (the per-case default re-measurements cannot be read)`,
    );
  }
  const cases = parsed.cases.map((entry, index) =>
    parseCaseRepeat(entry, `${where} cases[${index}]`),
  );
  const caseIds = new Set<string>();
  for (const { caseId } of cases) {
    if (caseIds.has(caseId)) throw new Error(`${where}: cases has ${caseId} twice`);
    caseIds.add(caseId);
  }
  if (!Array.isArray(parsed.rows)) throw new Error(`${where}: rows is not an array`);
  const rows = parsed.rows.map((row, index) => parseRow(row, `${where} rows[${index}]`));
  // MUST: only records whose ratio baseline (the default table's kernels) and case set equal the
  // current runtime's, for every op family the record has a row of, are accepted (ADR 0117 decision
  // 8, per family since its addendum 2026-11-12 — the same reason as checking the default row (ADR
  // 0115 decision 4): with a different baseline, "faster than the default" does not hold on the
  // current runtime)
  const families = familiesOf(rows);
  assertRecordedFamilyKeys(parsed, "caseSets", families, familyCaseSetId, "case set", where);
  assertRecordedFamilyKeys(
    parsed,
    "defaultKernels",
    families,
    (family) => familyKernelsId(DEFAULT_GEOMETRY_PROFILE, family),
    "kernels of the default table used as the ratio baseline",
    where,
  );
  // MUST: only records whose measured candidates' kernels equal the current runtime's are accepted
  // (ADR 0117 decision 8) — defaultKernels protects only the ratio baseline, and the speed and
  // output match of an adopted candidate come from the candidate's own kernels. The current value
  // is derived from the record's rows (sweepCandidateKernelsId in fingerprint.ts — the same single
  // function as the writer)
  assertRecordedCandidateKernels(
    parsed,
    () => sweepCandidateKernelsId(rows, recordedDp4a(parsed, where)),
    where,
  );
  const defaults = new Map<string, number>();
  for (const row of rows) {
    defaults.set(row.caseId, (defaults.get(row.caseId) ?? 0) + (row.isDefault ? 1 : 0));
  }
  for (const [caseId, count] of defaults) {
    if (count !== 1) {
      throw new Error(`${where}: ${caseId} has ${count} default rows (must be exactly 1)`);
    }
  }
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: requireString(parsed, "date", where),
    adapter: {
      vendor: requireString(adapter, "vendor", `${where} adapter`),
      architecture: requireString(adapter, "architecture", `${where} adapter`),
      device: requireString(adapter, "device", `${where} adapter`),
      description: requireString(adapter, "description", `${where} adapter`),
    },
    candidateSet: parseCandidateSet(parsed.settings, where),
    ...(userAgent === undefined ? {} : { userAgent }),
    timestampQuantum: geometryTimestampQuantum(quantized),
    cases,
    rows,
  };
};

/**
 * The field name of a gemmRows bucket. A bucket is spelled by its row range (`gemmRows ≤ 16`,
 * `gemmRows 17〜32`, `gemmRows > 512` — {@link profileRowsSegments}). It is not spelled by index
 * (`gemmRows[1]`) because `gemmRows[1]` of the three-bucket era (65–512) and the second of seven
 * (17–32) would have the same spelling for different ranges (ADR 0116 decision 6).
 */
export type GemmRowsSlot = `gemmRows ${string}`;

/**
 * The field name of an `i8a8.linear` bucket, spelled by its row range the way the gemmRows buckets
 * are (`i8a8.linear ≤ 256`, `i8a8.linear 257〜512`, `i8a8.linear > 512` — ADR 0115 addendum
 * decision 14).
 */
export type I8a8LinearSlot = `i8a8.linear ${string}`;

/** A profile field (= class). The generated file's comments name fields with this spelling too. */
export type ProfileSlot =
  | GemmRowsSlot
  | "attention.qk"
  | "attention.pv"
  | "conv2d.rows64"
  | "conv2d.rows32"
  | I8a8LinearSlot
  | "i8a8.attentionQk"
  | "i8a8.attentionPv"
  | "i8a8.conv3d";

type SlotSpec = {
  readonly slot: ProfileSlot;
  /** The range the field applies to (for comments). */
  readonly scope: string;
  /**
   * The current runtime's default and its display name (same spelling as the sweep's `geometry`).
   * The value written for a class with no case in the sweep, and also the ratio baseline compared
   * with the sweep's default row.
   */
  readonly fallback: () => { readonly name: string; readonly geometry: Geometry };
};

/**
 * One row-count bucket of `gemmRows` ({@link profileRowsSegments}) or of `i8a8.linear`
 * ({@link i8a8LinearSegments}).
 */
export type RowsSegment<
  Slot extends ProfileSlot = GemmRowsSlot,
  G extends Geometry = GemmGeometry,
> = {
  readonly slot: Slot;
  /** The range the field applies to (for comments). */
  readonly scope: string;
  readonly maxRows: number;
  /**
   * The geometry of the default table's bucket covering this bucket's range (`GEMM_ROWS_BUCKETS`
   * for gemmRows, the default profile's one `i8a8.linear` rule for the i8a8 linear). The value
   * written for a bucket with no adoption or no case in the sweep, and also the ratio baseline
   * compared with the sweep's default row (the sweep builds it from the default table,
   * `gemmGeometryForRows(m)` / `i8a8LinearGeometry(DEFAULT_GEOMETRY_PROFILE, m)`).
   */
  readonly fallback: G;
};

/**
 * The spelling of the row range of bucket `index` (`≤ 16`, `17〜32`, `> 512`). The boundaries are a
 * checked list (positive integers, strictly ascending, ending in Infinity).
 */
export const rowsRange = (bounds: readonly number[], index: number): string => {
  const maxRows = bounds[index];
  if (index === 0) return `≤ ${maxRows}`;
  const previous = bounds[index - 1];
  return maxRows === Number.POSITIVE_INFINITY ? `> ${previous}` : `${previous + 1}〜${maxRows}`;
};

/**
 * A pure function that checks the gemmRows bucket boundaries (the `maxRows` column) and turns them
 * into a list of buckets (ADR 0116 decision 3). The generator passes `PROFILE_GEMM_ROWS_BOUNDS` of
 * the sweep's shape table (the boundary constant lives in one place — not duplicated here).
 *
 * MUST: the boundaries end in Infinity, the others are positive integers in strictly ascending
 * order, and they form a refinement that contains every `maxRows` of the default table
 * (`GEMM_ROWS_BUCKETS`). Otherwise generation stops (fail loudly). A bucket that is not a
 * refinement would split the geometry of the sweep's default row in two within the bucket (two
 * ratio baselines), and the bucket's default (fallback) would not be unique either. A bucket's
 * default is the geometry of the default bucket covering its range (as a refinement, every row
 * count of the bucket falls in the same default bucket).
 *
 * NOTE: the default is taken from `gemm-geometry.ts` (the generated values are not the source of
 * truth for the default). The generator does depend on the generated files, though — through the
 * candidate set `quick+` (geometries.ts reads `geometry-profiles/index.ts`). If a registered
 * generated file breaks syntactically, the generator itself can no longer be loaded; in that case,
 * remove its registration from index.ts and then regenerate.
 */
export const profileRowsSegments = (bounds: readonly number[]): readonly RowsSegment[] =>
  rowsSegments(
    "gemmRows",
    bounds,
    GEMM_ROWS_BUCKETS,
    (range) => `gemmRows ${range}`,
    (range) => `linear / matmul / bmm row count ${range}`,
  );

/**
 * The buckets of `i8a8.linear` (ADR 0115 addendum decision 14), checked and built by the same rule
 * as {@link profileRowsSegments}; the generator passes `I8A8_LINEAR_ROWS_BOUNDS`. The default is
 * the default profile's one rule (`defaultI8a8Geometry("linear")` up to Infinity), so every bucket's
 * default is that geometry and any bounds ending in Infinity refine it.
 */
export const i8a8LinearSegments = (
  bounds: readonly number[],
): readonly RowsSegment<I8a8LinearSlot, I8a8Geometry>[] =>
  rowsSegments(
    "i8a8.linear",
    bounds,
    DEFAULT_GEOMETRY_PROFILE.i8a8.linear,
    (range) => `i8a8.linear ${range}`,
    (range) => `i8a8 linear row count ${range}`,
  );

/**
 * The body of {@link profileRowsSegments} / {@link i8a8LinearSegments}: the bounds of `field`
 * checked against its default rules (`defaults`), and one bucket per bound with its slot, scope
 * and the default rule covering it.
 */
const rowsSegments = <Slot extends ProfileSlot, G extends Geometry>(
  field: string,
  bounds: readonly number[],
  defaults: readonly { readonly maxRows: number; readonly geometry: G }[],
  slotOf: (range: string) => Slot,
  scopeOf: (range: string) => string,
): readonly RowsSegment<Slot, G>[] => {
  const where = `${field} bucket bounds [${bounds.join(", ")}]`;
  if (bounds.at(-1) !== Number.POSITIVE_INFINITY) {
    throw new Error(`${where}: the last bound is not Infinity (no bucket covers a larger M)`);
  }
  bounds.slice(0, -1).forEach((maxRows, index) => {
    if (!Number.isSafeInteger(maxRows) || maxRows < 1) {
      throw new Error(
        `${where}: bounds other than the last must be positive integers (${maxRows})`,
      );
    }
    if (!(maxRows < bounds[index + 1])) {
      throw new Error(
        `${where}: not strictly ascending (${maxRows} → ${
          bounds[index + 1]
        } — the later bucket would cover no row count)`,
      );
    }
  });
  const missing = defaults
    .map((rule) => rule.maxRows)
    .filter((maxRows) => !bounds.includes(maxRows));
  if (missing.length > 0) {
    throw new Error(
      `${where}: missing the default table's bounds ${missing.join(" / ")} (not a refinement of the default) — ` +
        "a bucket that straddles a default bound splits the geometry of the sweep's default row within the bucket, and the bucket's default is not unique either",
    );
  }
  return bounds.map((maxRows, index) => {
    const range = rowsRange(bounds, index);
    const covering =
      defaults.find((rule) => maxRows <= rule.maxRows) ?? defaults[defaults.length - 1];
    return { slot: slotOf(range), scope: scopeOf(range), maxRows, fallback: covering.geometry };
  });
};

/**
 * The fields between the gemmRows buckets and the i8a8.linear buckets (in the order of the
 * generated fields and of the verdicts in the comments).
 */
const F32_FIXED_SLOTS: readonly SlotSpec[] = [
  {
    slot: "attention.qk",
    scope: "fused attention f32 ①QK",
    fallback: () => gemmCandidate(defaultGemmGeometry()),
  },
  {
    slot: "attention.pv",
    scope: "fused attention f32 ③PV",
    fallback: () => gemmCandidate(defaultGemmGeometry()),
  },
  {
    slot: "conv2d.rows64",
    scope: "conv2d implicit GEMM m tile of 64 rows",
    fallback: () => conv2dCandidate(gemmMTileGeometry(64)),
  },
  {
    slot: "conv2d.rows32",
    scope: "conv2d implicit GEMM m tile of 32 rows",
    fallback: () => conv2dCandidate(gemmMTileGeometry(32)),
  },
];

/** The fields after the i8a8.linear buckets (likewise in order). */
const I8A8_FIXED_SLOTS: readonly SlotSpec[] = [
  {
    slot: "i8a8.attentionQk",
    scope: "i8a8 attention ①QK",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("attention_qk")),
  },
  {
    slot: "i8a8.attentionPv",
    scope: "i8a8 attention ③PV",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("attention_pv")),
  },
  {
    slot: "i8a8.conv3d",
    scope: "i8a8 conv3d (w8a8 implicit GEMM)",
    fallback: () => i8a8Candidate(defaultI8a8Geometry("conv3d")),
  },
];

/**
 * The list of fields (in the order of the generated fields and of the verdicts in the comments —
 * gemmRows buckets → fused attention f32 and conv2d → i8a8.linear buckets → the other i8a8 fields).
 */
const profileSlots = (
  segments: readonly RowsSegment[],
  i8a8Segments: readonly RowsSegment<I8a8LinearSlot, I8a8Geometry>[],
): readonly SlotSpec[] => [
  ...segments.map((segment) => ({
    slot: segment.slot,
    scope: segment.scope,
    fallback: () => gemmCandidate(segment.fallback),
  })),
  ...F32_FIXED_SLOTS,
  ...i8a8Segments.map((segment) => ({
    slot: segment.slot,
    scope: segment.scope,
    fallback: () => i8a8Candidate(segment.fallback),
  })),
  ...I8A8_FIXED_SLOTS,
];

/** The row count of linear / matmul (`M{m} N{n} K{k}` of `caseShape`). */
const LINEAR_SHAPE = /^M(\d+) N\d+ K\d+$/;
/**
 * The row count of one bmm matrix (`B{batch} M{m} N{n} K{k}` of `caseShape` — the batch is on the z
 * axis and does not affect the bucket).
 */
const BMM_SHAPE = /^B\d+ M(\d+) N\d+ K\d+$/;
/** The stage of fused attention (`{qk|pv} BH…` of `caseShape`). */
const ATTENTION_SHAPE = /^(qk|pv) BH\d+ M\d+ N\d+ D\d+$/;

/** What {@link slotOf} reads of a row or a case: its id, op and shape (`caseShape`'s spelling). */
type SlotEntry = Pick<SweepObservation, "caseId" | "op" | "shape">;

const attentionStage = (row: SlotEntry, where: string): "qk" | "pv" => {
  const matched = ATTENTION_SHAPE.exec(row.shape);
  if (matched === null) {
    throw new Error(`${where}: cannot read the shape of ${row.caseId} (${row.shape})`);
  }
  return matched[1] === "qk" ? "qk" : "pv";
};

/**
 * The bucket of `segments` a row's M falls in (the first with `rows <= maxRows` — the same rule
 * by which the production paths look up their rules by M). The bucket list ends in Infinity
 * (checked by {@link rowsSegments}), so some bucket always matches.
 */
const segmentOf = <Slot extends ProfileSlot, G extends Geometry>(
  row: SlotEntry,
  pattern: RegExp,
  segments: readonly RowsSegment<Slot, G>[],
  where: string,
): Slot => {
  const matched = pattern.exec(row.shape);
  if (matched === null) {
    throw new Error(`${where}: cannot read the shape of ${row.caseId} (${row.shape})`);
  }
  const rows = Number(matched[1]);
  return (segments.find((segment) => rows <= segment.maxRows) ?? segments[segments.length - 1])
    .slot;
};

/**
 * A row's field. For linear / matmul / bmm, the gemmRows bucket of the shape's M (the three
 * production paths look up the same gemmRows table by M — src/runtime/linear-program.ts and
 * src/runtime/recipe-builders/linear.ts,
 * gemmRowsGeometry in geometry-profile.ts); for the i8a8 linear, the i8a8.linear bucket of its M
 * (`i8a8LinearGeometry`); for attention, the shape's stage; for conv2d, **the tileM of the case's
 * default geometry** (`conv2dDefaultTileM` — the side chosen by the production m-tile selection
 * `conv2dIgemmMTile`: the derivation reads it from the case's default row, the geometry the sweep
 * measured as the default being the source of truth; {@link sweepCaseSlot} from the case plan that
 * row comes from).
 *
 * MUST: the one mapping of a case to its class. The derivation judges a class over these cases, and
 * the sweep's `explore` set drops a candidate for the rest of the cases of the same class
 * (geometry-explore.ts, through {@link sweepCaseSlot}); two mappings could drop a candidate for a
 * case the derivation reads in another class.
 */
const slotOf = (
  row: SlotEntry,
  segments: readonly RowsSegment[],
  i8a8Segments: readonly RowsSegment<I8a8LinearSlot, I8a8Geometry>[],
  conv2dDefaultTileM: number | undefined,
  where: string,
): ProfileSlot => {
  switch (row.op) {
    case "linear":
    case "matmul":
    case "bmm":
      return segmentOf(row, row.op === "bmm" ? BMM_SHAPE : LINEAR_SHAPE, segments, where);
    case "attention":
      return attentionStage(row, where) === "qk" ? "attention.qk" : "attention.pv";
    case "conv2d": {
      if (conv2dDefaultTileM === 64) return "conv2d.rows64";
      if (conv2dDefaultTileM === 32) return "conv2d.rows32";
      throw new Error(
        `${where}: the default m tile of ${row.caseId} is not 64 / 32 (${conv2dDefaultTileM})`,
      );
    }
    case "i8a8-linear":
      return segmentOf(row, LINEAR_SHAPE, i8a8Segments, where);
    case "i8a8-attention":
      return attentionStage(row, where) === "qk" ? "i8a8.attentionQk" : "i8a8.attentionPv";
    case "i8a8-conv3d":
      return "i8a8.conv3d";
  }
};

/**
 * The profile field (= class) of a sweep case: {@link slotOf} of its shape, with the buckets the
 * derivation reads by ({@link deriveProfile}'s) and, for conv2d, the tileM of the default geometry
 * its case plan measures — the geometry its default row carries. The sweep's `explore` set measures
 * class by class through this (geometry-explore.ts). Pure (the case plan builds no GPU object).
 */
export const sweepCaseSlot = (sweepCase: SweepCase): ProfileSlot =>
  slotOf(
    { caseId: sweepCase.id, op: sweepCase.op, shape: caseShape(sweepCase) },
    profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS),
    i8a8LinearSegments(I8A8_LINEAR_ROWS_BOUNDS),
    sweepCase.op === "conv2d"
      ? gemmTileM(casePlan(sweepCase, Number.POSITIVE_INFINITY, false).defaultCandidate.geometry)
      : undefined,
    sweepCase.id,
  );

/**
 * The identity of a geometry (an i8a8 geometry with the vec4 skeleton, and an f32 geometry with a
 * skeleton form or the counter decode, is another geometry than the same tile without it — another
 * kernel, a candidate of its own).
 */
const geometryKey = (geometry: Geometry): string =>
  "tileK" in geometry
    ? `${geometry.regM},${geometry.regN},${geometry.wgX},${geometry.wgY},${geometry.tileK}${
        geometry.skeleton === undefined ? "" : `,${geometry.skeleton}`
      }`
    : `${geometry.regM},${geometry.regN},${geometry.wgX},${geometry.wgY}${
        geometry.form === undefined ? "" : `,${geometry.form}`
      }${geometry.decode === undefined ? "" : `,${geometry.decode}`}`;

const geometricMean = (values: readonly number[]): number =>
  Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);

/**
 * The display of a ratio (same spelling in the generated file's comments and on standard output).
 */
export const formatRatio = (value: number): string => `×${value.toFixed(3)}`;

/**
 * The display of two ratios a rejection reason compares (`<value> < <bar>`, `<geomean> (below the
 * adopted <geomean>)`): both at the fewest decimals (3 or more) at which they read differently. At 3
 * decimals a value just under the bar (1.0498 against 1.05) would print as the bar itself
 * (`×1.050 < ×1.050`); more decimals only where needed keep every other reason in the
 * {@link formatRatio} spelling, so the generated headers of values not at the bar do not change.
 * Equal ratios (a tie broken by name) keep 3 decimals.
 */
const formatRatioPair = (left: number, right: number): readonly [string, string] => {
  let digits = 3;
  // Two distinct doubles read differently by 17 decimals; 100 is the largest `toFixed` accepts.
  while (left !== right && digits < 100 && left.toFixed(digits) === right.toFixed(digits)) {
    digits += 1;
  }
  return [`×${left.toFixed(digits)}`, `×${right.toFixed(digits)}`];
};

/** The reason of an eligible candidate whose geometric mean is short of the adopted one's. */
const belowAdopted = (geomean: number, adopted: number): string => {
  const [mine, theirs] = formatRatioPair(geomean, adopted);
  return `geometric mean ${mine} (below the adopted ${theirs})`;
};

export type SlotVerdict = {
  readonly slot: ProfileSlot;
  readonly scope: string;
  /** The class's cases (ascending by id). */
  readonly cases: readonly string[];
  readonly outcome:
    | {
        readonly kind: "adopted";
        readonly name: string;
        readonly geometry: Geometry;
        /** The geometric mean across the cases. */
        readonly geomean: number;
        readonly min: number;
        readonly max: number;
      }
    | {
        readonly kind: "default";
        readonly name: string;
        readonly geometry: Geometry;
        readonly reason: string;
      };
  /** The geometries not taken and why (ascending by name). */
  readonly rejected: readonly { readonly name: string; readonly reason: string }[];
  /**
   * The cases and observations of this class dropped from the ratio material, per sweep (in the
   * order the sweeps were passed → ascending case id → per-case exclusions → ascending geometry
   * name). There are per-case exclusions (re-measurement ratio out of range, the default row's
   * rounding error) and per-observation exclusions (rounding-error bound E > 1% — with `geometry`).
   */
  readonly excluded: readonly ExcludedCase[];
};

/**
 * A case dropped from the ratio material within one sweep ({@link exclusionReason}), or one
 * geometry's observation of a case (with `geometry` — a row whose rounding-error bound was
 * exceeded, {@link roundingBound}) — output match and failures are still checked.
 */
export type ExcludedCase = {
  readonly path: string;
  readonly caseId: string;
  /**
   * When only one geometry's observation of the case was dropped, that geometry's name (absent =
   * every observation of the case).
   */
  readonly geometry?: string;
  /**
   * E.g. "the default re-measurement ratio ×1.160 is out of range" ({@link excludedCaseLine} makes
   * it a sentence).
   */
  readonly reason: string;
};

/**
 * One line for a dropped case or observation (the same sentence in the generated file's comments,
 * on standard output and on the page).
 */
export const excludedCaseLine = (excluded: ExcludedCase): string =>
  `Sweep ${excluded.path}: ${excluded.caseId}${
    excluded.geometry === undefined ? "" : ` geometry ${excluded.geometry}`
  } dropped from the ratio material due to ${excluded.reason} (output matches and failures are still checked)`;

/**
 * The reason to drop a case from that sweep's ratio material (undefined if it is kept).
 *
 * If the default re-measurement ratio (the default measured once more at the end of the case ÷ the
 * first) is outside report.ts's `DEFAULT_DRIFT_RANGE`, heat or clocks likely moved during the case
 * and the ratio baseline (the default value) itself is unstable — the case's ratios to the default
 * cannot be trusted for any geometry. A case whose re-measurement failed or is missing cannot be
 * checked for drift, so it is treated the same.
 */
const exclusionReason = (repeat: CaseRepeatFields | undefined): string | undefined => {
  if (repeat?.defaultRepeatError !== undefined) {
    return `failed default re-measurement (${repeat.defaultRepeatError})`;
  }
  const driftRatio = repeat?.defaultRepeat?.driftRatio;
  if (driftRatio === undefined) return "no default re-measurement";
  return driftOutOfRange(driftRatio)
    ? `default re-measurement ratio ${formatRatio(driftRatio)} out of range`
    : undefined;
};

/** The fields of a record's case summary that the case's exclusion reads. */
type CaseRepeatFields = Pick<SweepCaseRepeat, "defaultRepeat" | "defaultRepeatError">;

/** The rounding-error bound (`bound` = relative error), or why no bound can be given. */
type RoundingBound = { readonly bound: number } | { readonly unknown: string };

/**
 * A row's rounding-error bound e(r) = q ÷ min(rounds) (ADR 0117 decision 3). The rounding error of
 * a difference of two timestamps is under one step, so e bounds the relative error of the min round
 * (= `perDispatch` × reps) (to first order).
 *
 * Rounds of 0 are not min candidates — negative rounds stay in the record as 0 (report.ts) and are
 * excluded from the min candidates of `perDispatch`. If a round rounded to 0 (not negative) is the
 * min, `perDispatch` is 0 and excluding it does not change the verdict: as a candidate row it has
 * no ratio to the default; as the default row the ratio baseline is 0, so no row of the case has a
 * ratio to the default and there is no default re-measurement either (the comparison in report.ts
 * and the re-measurement in harness.ts — neither writes the 0 side).
 */
const roundingBound = (row: Pick<SweepObservation, "rounds">, quantum: number): RoundingBound => {
  // No rounding means no error — old records without rounds pass here too
  if (quantum === 0) return { bound: 0 };
  if (row.rounds === undefined) return { unknown: "no rounds" };
  const kept = row.rounds.filter((round) => round > 0);
  if (kept.length === 0) return { unknown: "no positive round" };
  return { bound: quantum / Math.min(...kept) };
};

/** The display of a percentage (the rounding-error bound — the spelling in the verdict lines). */
const formatPercent = (value: number): string => `${(value * 100).toFixed(2)}%`;

const LIMIT_LABEL = `${ROUNDING_ERROR_LIMIT * 100}%`;

/**
 * When the default row's rounding-error bound e(d) exceeds the threshold or cannot be given, the E
 * of every observation of the case exceeds it (E = e(r) + e(d) ≥ e(d)), so this is the reason to
 * drop the whole case (undefined if it is kept).
 */
const defaultRoundingReason = (bound: RoundingBound): string | undefined => {
  if ("unknown" in bound)
    return `unknown rounding-error bound of the default row (${bound.unknown})`;
  return bound.bound > ROUNDING_ERROR_LIMIT
    ? `rounding-error bound e ${formatPercent(bound.bound)} of the default row (> ${LIMIT_LABEL})`
    : undefined;
};

/**
 * The reason to drop an observation when its ratio's rounding-error bound E = e(r) + e(d) exceeds
 * the threshold or cannot be given.
 */
const rowRoundingReason = (bound: RoundingBound, defaultBound: number): string | undefined => {
  if ("unknown" in bound) return `unknown rounding-error bound (${bound.unknown})`;
  const total = bound.bound + defaultBound;
  return total > ROUNDING_ERROR_LIMIT
    ? `rounding-error bound E ${formatPercent(total)} (> ${LIMIT_LABEL})`
    : undefined;
};

/**
 * The rounding step q of a record's timestamps ({@link roundingBound}): Chrome's 100 µs when the
 * record's `gpuTiming.quantized` is true, 0 (no rounding) when it is false.
 */
export const geometryTimestampQuantum = (quantized: boolean): number =>
  quantized ? CHROME_TIMESTAMP_QUANTUM_NS : 0;

/**
 * Why the derivation drops a case of one record from that record's ratio material (undefined: it
 * is kept): the default re-measurement failed, is missing or its ratio is out of range
 * ({@link exclusionReason}), or — when the default row did not fail — the default row's
 * rounding-error bound exceeds 1% or cannot be given ({@link defaultRoundingReason}). `repeat` is the
 * record's summary of the case (`cases[]`), `defaultRow` the case's default row and
 * `timestampQuantum` the record's step ({@link geometryTimestampQuantum}). A dropped case leaves no
 * ratio for any geometry of the record there; its output matches and failures are still read.
 */
export const geometryCaseExclusion = (
  repeat: CaseRepeatFields | undefined,
  defaultRow: Pick<SweepObservation, "error" | "rounds">,
  timestampQuantum: number,
): string | undefined =>
  exclusionReason(repeat) ??
  (defaultRow.error === undefined
    ? defaultRoundingReason(roundingBound(defaultRow, timestampQuantum))
    : undefined);

/**
 * Why the derivation drops one row's observation from its record's ratio material, its case kept
 * (undefined: it is kept, or the row is no ratio material — the default row and a row without a
 * ratio to the default, a failure and the like, are not checked): the ratio's rounding-error bound
 * E = e(r) + e(d) exceeds 1% or cannot be given ({@link rowRoundingReason}). Without a bound of the
 * default row the case itself is dropped ({@link geometryCaseExclusion}).
 */
export const geometryRowExclusion = (
  row: Pick<SweepObservation, "isDefault" | "speedupVsDefault" | "rounds">,
  defaultRow: Pick<SweepObservation, "rounds">,
  timestampQuantum: number,
): string | undefined => {
  if (row.isDefault || row.speedupVsDefault === undefined) return undefined;
  const base = roundingBound(defaultRow, timestampQuantum);
  return "unknown" in base
    ? undefined
    : rowRoundingReason(roundingBound(row, timestampQuantum), base.bound);
};

/** A row assigned to a class (`included` = kept as material). */
type SlotRow = { readonly row: SweepObservation; readonly included: boolean };

type Eligibility =
  | {
      readonly eligible: true;
      readonly geomean: number;
      readonly min: number;
      readonly max: number;
    }
  | { readonly eligible: false; readonly reason: string };

/**
 * The verdict on one candidate. Looks at **every case** of the class — if even one case is
 * unmeasured, failed, mismatched or lacks a ratio, the candidate is dropped (the first such case in
 * ascending case id); if every case is present, the check is applied to the case it loses most.
 *
 * Failures and output mismatches are checked including the rows dropped from the ratio material
 * (`included: false`) — the output match is a correctness check and does not depend on heat or
 * clocks, so a geometry that mismatched or failed in any one sweep is not taken. Ratios come only
 * from the rows kept as material; a case dropped from the ratio material in every sweep
 * (`unmeasured`) counts as having no ratio measured for any geometry.
 */
const judgeCandidate = (
  cases: readonly string[],
  observations: ReadonlyMap<string, readonly SlotRow[]>,
  unmeasured: ReadonlySet<string>,
  minSpeedup: number,
): Eligibility => {
  const perCase: { readonly caseId: string; readonly speedup: number }[] = [];
  for (const caseId of cases) {
    const all = (observations.get(caseId) ?? []).map(({ row }) => row);
    const failed = all.find((row) => row.error !== undefined);
    if (failed !== undefined) {
      return { eligible: false, reason: `failed in ${caseId} (${failed.error})` };
    }
    if (all.some((row) => row.identicalToDefault === false)) {
      return { eligible: false, reason: `output differs from the default in ${caseId}` };
    }
    const seen = (observations.get(caseId) ?? []).flatMap(({ row, included }) =>
      included ? [row] : [],
    );
    if (seen.length === 0) {
      return {
        eligible: false,
        reason: unmeasured.has(caseId)
          ? `not measured in ${caseId} (dropped from the ratio material in every sweep)`
          : all.length > 0 && all.every((row) => row.screened === true)
            ? `screened out by the quick look in ${caseId} (no timing to compare)`
            : all.length > 0 && all.every((row) => row.skipped === true)
              ? `not measured in ${caseId} (the explore set dropped it for the class)`
              : `not measured in ${caseId}`,
      };
    }
    const speedups: number[] = [];
    for (const row of seen) {
      if (row.identicalToDefault !== true || row.speedupVsDefault === undefined) {
        return {
          eligible: false,
          reason: `no comparison with the default in ${caseId} (a failed default row or the like)`,
        };
      }
      speedups.push(row.speedupVsDefault);
    }
    perCase.push({ caseId, speedup: geometricMean(speedups) });
  }
  const worst = perCase.reduce((low, entry) => (entry.speedup < low.speedup ? entry : low));
  if (worst.speedup < minSpeedup) {
    const [value, bar] = formatRatioPair(worst.speedup, minSpeedup);
    return { eligible: false, reason: `${worst.caseId}: ${value} < ${bar}` };
  }
  const speedups = perCase.map((entry) => entry.speedup);
  return {
    eligible: true,
    geomean: geometricMean(speedups),
    min: Math.min(...speedups),
    max: Math.max(...speedups),
  };
};

/** A candidate of a class: its geometry and its rows per case (dropped rows included). */
type SlotCandidate = {
  readonly geometry: Geometry;
  readonly observations: ReadonlyMap<string, readonly SlotRow[]>;
};

/**
 * The rows of one class read for judging (records of one adapter): its cases, the cases dropped from
 * the ratio material in every sweep, the sweep's default row and the candidates by name.
 */
type SlotRows = {
  readonly cases: readonly string[];
  readonly unmeasured: ReadonlySet<string>;
  readonly defaultRow: SweepObservation;
  readonly candidates: ReadonlyMap<string, SlotCandidate>;
};

/** The reason of a class no record has a case of. */
const NO_CASE = "the sweeps have no case of this class";

/**
 * Reads a class's rows (one or more, records of one adapter) for judging, after the checks of its
 * default rows (fails loudly).
 */
const readSlotRows = (spec: SlotSpec, slotRows: readonly SlotRow[]): SlotRows => {
  // The default-row check and the case list include the rows dropped from the ratio material (only
  // the ratio observations are dropped — the baseline check, output match and failures apply to the
  // record itself)
  const rows = slotRows.map(({ row }) => row);
  const cases = [...new Set(rows.map((row) => row.caseId))].sort();
  const measured = new Set(
    slotRows.filter(({ included }) => included).map(({ row }) => row.caseId),
  );
  const unmeasured = new Set(cases.filter((caseId) => !measured.has(caseId)));
  const defaults = rows.filter((row) => row.isDefault);
  const defaultKeys = new Set(defaults.map((row) => geometryKey(row.geometryParams)));
  // MUST: a class has one default. If it splits, the ratio baseline differs per case (sweeps of
  // different versions and the like)
  if (defaultKeys.size !== 1) {
    throw new Error(
      `${spec.slot}: the geometry of the default rows is split (${[
        ...new Set(defaults.map((row) => row.geometry)),
      ].join(" / ")})`,
    );
  }
  const defaultRow = defaults[0];
  const runtimeDefault = spec.fallback();
  // MUST: the ratio baseline (the sweep's default row) is the current runtime's default. Otherwise
  // the ratios are relative to another geometry, and "faster than the default" does not hold on the
  // current runtime
  if (geometryKey(defaultRow.geometryParams) !== geometryKey(runtimeDefault.geometry)) {
    throw new Error(
      `${spec.slot}: the sweep's default row (${defaultRow.geometry}) differs from the current runtime's default (${runtimeDefault.name})` +
        " — a sweep taken with an old runtime, or shifted bounds; the ratio baseline differs, so nothing is generated",
    );
  }
  const byName = new Map<string, { geometry: Geometry; observations: Map<string, SlotRow[]> }>();
  for (const slotRow of slotRows) {
    const { row } = slotRow;
    if (row.isDefault || row.geometry === defaultRow.geometry) continue;
    const entry = byName.get(row.geometry) ?? {
      geometry: row.geometryParams,
      observations: new Map<string, SlotRow[]>(),
    };
    if (geometryKey(entry.geometry) !== geometryKey(row.geometryParams)) {
      throw new Error(
        `${spec.slot}: geometryParams of geometry ${row.geometry} differ between rows`,
      );
    }
    // Dropped rows are kept as observations too — not used for ratios, but failures and output
    // mismatches count in the verdict (judgeCandidate)
    entry.observations.set(row.caseId, [...(entry.observations.get(row.caseId) ?? []), slotRow]);
    byName.set(row.geometry, entry);
  }
  return { cases, unmeasured, defaultRow, candidates: byName };
};

const judgeSlot = (
  spec: SlotSpec,
  slotRows: readonly SlotRow[],
  excluded: readonly ExcludedCase[],
  minSpeedup: number,
): SlotVerdict => {
  if (slotRows.length === 0) {
    const { name, geometry } = spec.fallback();
    return {
      slot: spec.slot,
      scope: spec.scope,
      cases: [],
      outcome: { kind: "default", name, geometry, reason: NO_CASE },
      rejected: [],
      excluded,
    };
  }
  const { cases, unmeasured, defaultRow, candidates } = readSlotRows(spec, slotRows);
  const judged = [...candidates.entries()]
    .map(([name, entry]) => ({
      name,
      geometry: entry.geometry,
      eligibility: judgeCandidate(cases, entry.observations, unmeasured, minSpeedup),
    }))
    .sort((left, right) => (left.name < right.name ? -1 : left.name > right.name ? 1 : 0));
  const eligible = judged
    .flatMap((candidate) =>
      candidate.eligibility.eligible ? [{ ...candidate, ...candidate.eligibility }] : [],
    )
    // Descending geometric mean, ties by ascending name (decided independently of input order)
    .sort((left, right) => right.geomean - left.geomean || (left.name < right.name ? -1 : 1));
  const winner = eligible[0];
  const rejected = judged
    .filter((candidate) => candidate.name !== winner?.name)
    .map((candidate) => ({
      name: candidate.name,
      reason: candidate.eligibility.eligible
        ? belowAdopted(candidate.eligibility.geomean, winner.geomean)
        : candidate.eligibility.reason,
    }));
  return {
    slot: spec.slot,
    scope: spec.scope,
    cases,
    outcome:
      winner === undefined
        ? {
            kind: "default",
            name: defaultRow.geometry,
            geometry: defaultRow.geometryParams,
            reason:
              unmeasured.size === cases.length
                ? "every case of the class was dropped from the ratio material in every sweep"
                : `no geometry matches the output in every case at ${formatRatio(minSpeedup)} or faster`,
          }
        : {
            kind: "adopted",
            name: winner.name,
            geometry: winner.geometry,
            geomean: winner.geomean,
            min: winner.min,
            max: winner.max,
          },
    rejected,
    excluded,
  };
};

/**
 * The display of the table's target (`vendor / architecture / description` — only the given
 * fields).
 */
export const targetLabel = (target: ProfileTarget): string =>
  target.optIn === true
    ? "(injection only — no match)"
    : [target.vendor, target.architecture, target.description]
        .filter((part) => part !== undefined)
        .join(" / ");

/**
 * Produces the per-field verdicts from sweep records (one or more, same adapter) (a pure function).
 *
 * Material check: per sweep, a case whose default re-measurement ratio is out of range or whose
 * re-measurement failed or is missing ({@link exclusionReason}) is dropped from **that sweep's**
 * ratio observations. If another sweep measured the same case, its ratios decide; a case left in no
 * sweep counts as having no ratio measured for any geometry (by the rule of winning every case of
 * the class, the field stays at the default). Output mismatches and failures stay in the verdict
 * even for dropped cases — the output match is a correctness check and does not depend on heat, so
 * a geometry that mismatched or failed in a dropped sweep is no candidate even if it matched in
 * other sweeps. There is no option for whether to drop (one rule).
 *
 * Rounding check (ADR 0117 decision 3): a case whose default row's rounding-error bound e(d)
 * exceeds 1% (or cannot be given) is likewise dropped from that sweep's ratio observations
 * ({@link defaultRoundingReason}); for the other rows, only observations whose E = e(r) + e(d)
 * exceeds 1% are dropped ({@link rowRoundingReason}). Without quantization (q = 0), e = 0 and
 * rounds are not read.
 */
export const deriveProfile = (
  sources: readonly SweepSource[],
  options: ProfileTarget & { readonly minSpeedup: number },
): readonly SlotVerdict[] => {
  if (sources.length === 0) throw new Error("No sweep records");
  // Stop here if the bucket boundaries do not refine the default table (the ratio baseline would
  // split within a bucket — profileRowsSegments)
  const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
  const i8a8Segments = i8a8LinearSegments(I8A8_LINEAR_ROWS_BOUNDS);
  // The same conditions as the runtime's check (assertGeometryProfile) — stop before generating
  // rather than failing at the injection or registration step. description is the machine model's
  // name, a field only for splitting within a vendor / architecture
  if (options.description === "")
    throw new Error("description must not be an empty string (omit it when unspecified)");
  if (options.description !== undefined && options.architecture === undefined) {
    throw new Error("description must be given together with both vendor and architecture");
  }
  const seen = new Map<string, string>();
  for (const source of sources) {
    assertRequestedSource(source, seen, options);
    // MUST: all sweeps agree on all four adapter fields (two empty values match, one empty value
    // does not — ADR 0117 decision 4). The table's provenance holds a single adapter, and matching
    // (geometryProfileMismatch) looks at those four fields. With optIn or without description (the
    // CLI's --opt-in / --description, removed in 0.15) the check above does not look at
    // description, so mixing other models of the same vendor / architecture (M2 and M5, say) is
    // stopped here too. A table from several adapters is a shared table, derived by a rule of its
    // own (deriveSharedProfile — no timing of one adapter is read against another's)
    const first = sources[0].adapter;
    for (const field of ADAPTER_FIELDS) {
      if (source.adapter[field] !== first[field]) {
        throw new Error(
          `${source.path}: adapter ${field} ${JSON.stringify(source.adapter[field])} differs from ${
            sources[0].path
          }'s ${JSON.stringify(first[field])} (only sweeps that agree on all four fields are combined into one table)`,
        );
      }
    }
  }
  const { bySlot, excludedBySlot } = slotRowsOf(sources, segments, i8a8Segments);
  return profileSlots(segments, i8a8Segments).map((spec) =>
    judgeSlot(
      spec,
      bySlot.get(spec.slot) ?? [],
      excludedBySlot.get(spec.slot) ?? [],
      options.minSpeedup,
    ),
  );
};

/** The four adapter fields a record names (`GPUAdapterInfo`'s). */
const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * The checks of one record against the records before it and the requested target (fails loudly):
 * the same record twice, and an adapter the target does not take.
 */
const assertRequestedSource = (
  source: SweepSource,
  seen: Map<string, string>,
  options: ProfileTarget,
): void => {
  const twin = seen.get(source.sha256);
  // Counting the same record twice would give that record's observations double weight
  if (twin !== undefined) {
    throw new Error(`The same sweep is passed twice (${twin} and ${source.path})`);
  }
  seen.set(source.sha256, source.path);
  // MUST: measurements of different adapters are not mixed (a table is a per-adapter answer —
  // B570 and M2 point in opposite directions). When description is given, only sweeps of that
  // model (other models of the same vendor / architecture are not mixed)
  const { vendor, architecture, description } = source.adapter;
  if (
    options.optIn !== true &&
    (vendor !== options.vendor ||
      (options.architecture !== undefined && architecture !== options.architecture) ||
      (options.description !== undefined && description !== options.description))
  ) {
    throw new Error(
      `${source.path}: adapter ${vendor} / ${architecture} / ${description} does not match the requested ${targetLabel(
        options,
      )}`,
    );
  }
};

/**
 * The rows of records of one adapter by class, each marked as ratio material or not, and per class
 * the cases and observations dropped from the ratio material ({@link deriveProfile}'s material and
 * rounding checks).
 */
const slotRowsOf = (
  sources: readonly SweepSource[],
  segments: readonly RowsSegment[],
  i8a8Segments: readonly RowsSegment<I8a8LinearSlot, I8a8Geometry>[],
): {
  readonly bySlot: ReadonlyMap<ProfileSlot, readonly SlotRow[]>;
  readonly excludedBySlot: ReadonlyMap<ProfileSlot, readonly ExcludedCase[]>;
} => {
  const bySlot = new Map<ProfileSlot, SlotRow[]>();
  const excludedBySlot = new Map<ProfileSlot, ExcludedCase[]>();
  for (const source of sources) {
    const conv2dDefaultTileM = new Map(
      source.rows
        .filter((row) => row.op === "conv2d" && row.isDefault)
        .map((row) => [row.caseId, gemmTileM(row.geometryParams)] as const),
    );
    const repeats = new Map(source.cases.map((repeat) => [repeat.caseId, repeat] as const));
    const defaultRows = new Map(
      source.rows.filter((row) => row.isDefault).map((row) => [row.caseId, row] as const),
    );
    const excludedHere: { readonly slot: ProfileSlot; readonly excluded: ExcludedCase }[] = [];
    const excludedCases = new Set<string>();
    for (const row of source.rows) {
      const slot = slotOf(
        row,
        segments,
        i8a8Segments,
        conv2dDefaultTileM.get(row.caseId),
        source.path,
      );
      const defaultRow = defaultRows.get(row.caseId);
      // parseSweepReport has checked one default row per case
      if (defaultRow === undefined) {
        throw new Error(`${source.path}: ${row.caseId} has no default row`);
      }
      // The reason to drop the whole case (re-measurement ratio → the default row's rounding). A
      // case whose default row failed has no ratios, so rounding is not checked (no ratio material
      // — judgeCandidate looks at the failure)
      const caseReason = geometryCaseExclusion(
        repeats.get(row.caseId),
        defaultRow,
        source.timestampQuantum,
      );
      // The reason to drop an observation (E = e(r) + e(d)), named only in a case that is kept
      const rowReason =
        caseReason === undefined
          ? geometryRowExclusion(row, defaultRow, source.timestampQuantum)
          : undefined;
      if (caseReason !== undefined && !excludedCases.has(row.caseId)) {
        excludedCases.add(row.caseId);
        excludedHere.push({
          slot,
          excluded: { path: source.path, caseId: row.caseId, reason: caseReason },
        });
      }
      if (rowReason !== undefined) {
        excludedHere.push({
          slot,
          excluded: {
            path: source.path,
            caseId: row.caseId,
            geometry: row.geometry,
            reason: rowReason,
          },
        });
      }
      // A screened or skipped row is no ratio material (it has no ratio) and no exclusion either:
      // the quick look or the explore set left it out, and judgeCandidate names it
      bySlot.set(slot, [
        ...(bySlot.get(slot) ?? []),
        {
          row,
          included:
            caseReason === undefined &&
            rowReason === undefined &&
            row.screened !== true &&
            row.skipped !== true,
        },
      ]);
    }
    // Ascending case id → per-case exclusions first → ascending geometry name
    const order = (excluded: ExcludedCase): string =>
      `${excluded.caseId}\u0000${excluded.geometry ?? ""}`;
    const sorted = [...excludedHere].sort((left, right) => {
      const [a, b] = [order(left.excluded), order(right.excluded)];
      return a < b ? -1 : a > b ? 1 : 0;
    });
    for (const { slot, excluded } of sorted) {
      excludedBySlot.set(slot, [...(excludedBySlot.get(slot) ?? []), excluded]);
    }
  }
  return { bySlot, excludedBySlot };
};

// ---------------------------------------------------------------------------
// Shared tables: one table from the records of several adapters of one vendor and architecture
// (SPEC: docs/spec/geometry-profiles.md#shared-tables — DECIDED: ADR 0156)
// ---------------------------------------------------------------------------

/** One device of a shared derivation: its adapter and the label its verdicts name it by. */
type SharedDevice = {
  readonly adapter: SweepAdapter;
  /** The adapter's description (a shared derivation tells devices apart by it). */
  readonly label: string;
};

/** One device's ratios of a geometry eligible on it (over that device's cases of the class). */
export type SharedDeviceRatios = {
  readonly label: string;
  readonly geomean: number;
  readonly min: number;
  readonly max: number;
};

/**
 * The verdict on one field of a shared table ({@link deriveSharedProfile}): per device its cases
 * and exclusions, the outcome — the adopted geometry with its geometric mean over the devices and
 * each device's ratios, or the default with the reason — and the geometries not taken.
 */
export type SharedSlotVerdict = {
  readonly slot: ProfileSlot;
  readonly scope: string;
  /** In the order the devices' first records were given. */
  readonly devices: readonly {
    readonly label: string;
    /** The device's cases of the class (ascending by id; empty when it has none). */
    readonly cases: readonly string[];
    readonly excluded: readonly ExcludedCase[];
  }[];
  readonly outcome:
    | {
        readonly kind: "adopted";
        readonly name: string;
        readonly geometry: Geometry;
        /** The geometric mean over the devices of each device's geometric mean. */
        readonly geomean: number;
        readonly perDevice: readonly SharedDeviceRatios[];
      }
    | {
        readonly kind: "default";
        readonly name: string;
        readonly geometry: Geometry;
        readonly reason: string;
      };
  /** The geometries not taken and why, with the ratio per device (ascending by name). */
  readonly rejected: readonly { readonly name: string; readonly reason: string }[];
};

/** The identity of an adapter by its four fields. */
const adapterKey = (adapter: SweepAdapter): string =>
  JSON.stringify(ADAPTER_FIELDS.map((field) => adapter[field]));

/**
 * The records grouped by device, in the order each device's first record was given, each labelled
 * by its description.
 *
 * MUST: a shared derivation tells devices apart by the adapter's description, so every record
 * carries one and the records of one description agree on all four fields. Grouping by the four
 * fields would let one GPU pass as two devices — Chrome reports the same machine with its
 * developer-features flag (a description and a device id) and without it (both empty) — and the
 * rule of at least two devices would then hold on one machine.
 */
const sharedDevices = (
  sources: readonly SweepSource[],
): readonly (SharedDevice & { readonly sources: readonly SweepSource[] })[] => {
  const groups = new Map<
    string,
    { adapter: SweepAdapter; firstPath: string; sources: SweepSource[] }
  >();
  for (const source of sources) {
    const { description } = source.adapter;
    if (description === "") {
      throw new Error(
        `${source.path}: the adapter reports no description, so its records cannot be told apart from another device's (a shared derivation tells devices apart by their descriptions) — sweep where the adapter reports its description (in Chrome, with the WebGPU developer-features flag: chrome://flags/#enable-webgpu-developer-features)`,
      );
    }
    const group = groups.get(description);
    if (group === undefined) {
      groups.set(description, {
        adapter: source.adapter,
        firstPath: source.path,
        sources: [source],
      });
      continue;
    }
    for (const field of ADAPTER_FIELDS) {
      if (source.adapter[field] !== group.adapter[field]) {
        throw new Error(
          `${source.path}: adapter ${field} ${JSON.stringify(source.adapter[field])} differs from ${group.firstPath}'s ${JSON.stringify(group.adapter[field])} under the same description ${JSON.stringify(description)} (a shared derivation tells devices apart by their descriptions, so the records of one description must be of one adapter)`,
        );
      }
    }
    group.sources.push(source);
  }
  return [...groups.values()].map(({ adapter, sources: grouped }) => ({
    adapter,
    label: adapter.description,
    sources: grouped,
  }));
};

/**
 * A family one device of a shared derivation covers only in part: the device has a row of the
 * family, and some case of the family's current case set has no usable row of the default geometry
 * on it — no row at all, or only failed default rows.
 */
type PartialFamily = {
  readonly family: GeometryProfileFamily;
  readonly label: string;
  /** The cases without a usable default row, in the order of the current case set. */
  readonly missing: readonly { readonly caseId: string; readonly failed: boolean }[];
};

/**
 * The families each device covers only in part ({@link PartialFamily}), per device in the order
 * of `devices`, per family in the fixed order. A family a device has no row of is not partial
 * there (its fields keep the default by the rule of a class some device has no case of).
 *
 * A case counts as covered when some record of the device has a row of the default geometry of it
 * that did not fail: without one there is no baseline on that device, so no geometry can be shown
 * faster there (a device that cannot run the default in a case — Metal's command-buffer time limit
 * — never gets one, however often it is swept).
 */
const partialFamilies = (
  devices: readonly { readonly label: string; readonly sources: readonly SweepSource[] }[],
): readonly PartialFamily[] =>
  devices.flatMap(({ label, sources }) => {
    const rows = sources.flatMap((source) => source.rows);
    const covered = new Set(
      rows.filter((row) => row.isDefault && row.error === undefined).map((row) => row.caseId),
    );
    const swept = new Set(rows.map((row) => row.caseId));
    return familiesOf(rows).flatMap((family) => {
      const missing = familyEntries(SWEEP_CASES, family)
        .filter(({ id }) => !covered.has(id))
        .map(({ id }) => ({ caseId: id, failed: swept.has(id) }));
      return missing.length === 0 ? [] : [{ family, label, missing }];
    });
  });

/** `Apple M2 covers family i8a8-conv3d only in part: no usable row of the default geometry in …`. */
const partialReason = (partial: readonly PartialFamily[]): string =>
  `${partial
    .map(
      ({ family, label, missing }) =>
        `${label} covers family ${family} only in part: no usable row of the default geometry in ${missing
          .map(
            ({ caseId, failed }) => `${caseId} (${failed ? "the default's row failed" : "no row"})`,
          )
          .join(", ")}`,
    )
    .join(
      "; ",
    )} (regenerated on a base, a shared table takes a geometry for a family only where every device measured every case of its case set)`;

/** One device's eligibility of a geometry in one class (its own cases, its own default). */
type DeviceEligibility = { readonly label: string; readonly eligibility: Eligibility };

/** `Apple M5: ×1.300 (×1.250–×1.350); Apple M2: linear-m1024: ×0.950 < ×1.050`. */
const perDeviceText = (entries: readonly DeviceEligibility[]): string =>
  entries
    .map(({ label, eligibility }) =>
      eligibility.eligible
        ? `${label}: ${formatRatio(eligibility.geomean)} (${formatRatio(eligibility.min)}–${formatRatio(eligibility.max)})`
        : `${label}: ${eligibility.reason}`,
    )
    .join("; ");

/**
 * The verdict on one field of a shared table. A device that has no case of the class keeps the
 * field at the default (named in the reason). Otherwise each device's rows are read as one
 * adapter's ({@link readSlotRows} — its default-row checks included) and each geometry is judged
 * per device by {@link judgeCandidate}, on that device's cases against that device's default: a
 * geometry is eligible only when it is eligible on every device, and among those the one with the
 * largest geometric mean of the per-device geometric means is adopted (ties by ascending name).
 * No ratio of one device is read against another's. A field of a family some device covers only in
 * part (`partial`: the entries of the field's family) keeps the default, its reason naming each
 * such device and its cases; every device's rows still go through the default-row checks first.
 */
const judgeSharedSlot = (
  spec: SlotSpec,
  devices: readonly {
    readonly label: string;
    readonly slotRows: readonly SlotRow[];
    readonly excluded: readonly ExcludedCase[];
  }[],
  minSpeedup: number,
  partial: readonly PartialFamily[],
): SharedSlotVerdict => {
  const read = devices.map((device) => ({
    label: device.label,
    excluded: device.excluded,
    rows: device.slotRows.length === 0 ? undefined : readSlotRows(spec, device.slotRows),
  }));
  const listed = read.map(({ label, excluded, rows }) => ({
    label,
    cases: rows?.cases ?? [],
    excluded,
  }));
  const present = read.flatMap(({ label, rows }) => (rows === undefined ? [] : [{ label, rows }]));
  const fallback = spec.fallback();
  const keepDefault = (reason: string, rejected: SharedSlotVerdict["rejected"] = []) => ({
    slot: spec.slot,
    scope: spec.scope,
    devices: listed,
    outcome: {
      kind: "default" as const,
      name: present[0]?.rows.defaultRow.geometry ?? fallback.name,
      geometry: present[0]?.rows.defaultRow.geometryParams ?? fallback.geometry,
      reason,
    },
    rejected,
  });
  if (partial.length > 0) return keepDefault(partialReason(partial));
  if (present.length === 0) return keepDefault(NO_CASE);
  const missing = read.filter(({ rows }) => rows === undefined).map(({ label }) => label);
  if (missing.length > 0) {
    return keepDefault(
      `no case of this class in the records of ${missing.join(", ")} (a shared table takes a geometry only for a class every device measured)`,
    );
  }
  const geometries = new Map<string, Geometry>();
  for (const { rows } of present) {
    for (const [name, candidate] of rows.candidates) {
      const known = geometries.get(name);
      if (known !== undefined && geometryKey(known) !== geometryKey(candidate.geometry)) {
        throw new Error(`${spec.slot}: geometryParams of geometry ${name} differ between records`);
      }
      geometries.set(name, candidate.geometry);
    }
  }
  const judged = [...geometries.entries()]
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([name, geometry]) => ({
      name,
      geometry,
      perDevice: present.map(({ label, rows }): DeviceEligibility => ({
        label,
        // A geometry a device has no row of is judged on no observation: not measured there
        eligibility: judgeCandidate(
          rows.cases,
          rows.candidates.get(name)?.observations ?? new Map<string, readonly SlotRow[]>(),
          rows.unmeasured,
          minSpeedup,
        ),
      })),
    }));
  const eligible = judged
    .flatMap((candidate) => {
      const ratios = candidate.perDevice.flatMap(({ label, eligibility }) =>
        eligibility.eligible ? [{ label, ...eligibility }] : [],
      );
      return ratios.length === candidate.perDevice.length
        ? [{ ...candidate, ratios, geomean: geometricMean(ratios.map((entry) => entry.geomean)) }]
        : [];
    })
    // Descending geometric mean, ties by ascending name (decided independently of input order)
    .sort((left, right) => right.geomean - left.geomean || (left.name < right.name ? -1 : 1));
  const winner = eligible[0];
  const rejected = judged
    .filter((candidate) => candidate.name !== winner?.name)
    .map((candidate) => {
      const ratios = perDeviceText(candidate.perDevice);
      const own = eligible.find((entry) => entry.name === candidate.name);
      return {
        name: candidate.name,
        reason:
          own === undefined
            ? ratios
            : `${belowAdopted(own.geomean, winner.geomean)} over the devices — ${ratios}`,
      };
    });
  if (winner === undefined) {
    const dropped = present.find(({ rows }) => rows.unmeasured.size === rows.cases.length);
    return keepDefault(
      dropped === undefined
        ? `no geometry matches the output in every case at ${formatRatio(minSpeedup)} or faster on every device`
        : `every case of the class was dropped from the ratio material in every sweep of ${dropped.label}`,
      rejected,
    );
  }
  return {
    slot: spec.slot,
    scope: spec.scope,
    devices: listed,
    outcome: {
      kind: "adopted",
      name: winner.name,
      geometry: winner.geometry,
      geomean: winner.geomean,
      perDevice: winner.ratios.map(({ label, geomean, min, max }) => ({
        label,
        geomean,
        min,
        max,
      })),
    },
    rejected,
  };
};

/**
 * Produces the per-field verdicts of a **shared table** from the sweep records of two or more
 * adapters of one vendor and architecture (a pure function — SPEC:
 * docs/spec/geometry-profiles.md#shared-tables, DECIDED: ADR 0156).
 *
 * The records are grouped by device, told apart by the adapter's description
 * ({@link sharedDevices}). Inside one group every rule of
 * {@link deriveProfile} applies as it does to a measured table — the records of a group combine,
 * the material and rounding checks drop the same observations, a geometry must match the default's
 * output and reach `minSpeedup` on every case of the class the group holds. Timings are never
 * compared or pooled across groups: every ratio is a geometry against its own device's default on
 * that device. A geometry is eligible only when it is eligible in every group, and the one with
 * the largest geometric mean of the per-device geometric means is adopted (devices weigh equally,
 * not cases). A class some group has no case of keeps the default.
 *
 * With `onBase` (the derivation regenerates a table — `DeriveGeometryProfileOptions.base`), every
 * field of a family some group covers only in part ({@link partialFamilies}) keeps the default,
 * its reason naming the group and the cases: carrying over promises that every covered family is
 * derived from every case of its case set, and where a device cannot give that, the family's
 * fields hold the default openly instead of a choice for shapes that device did not measure.
 * (The measured derivation refuses such records instead — {@link carryOver}.)
 *
 * MUST: vendor and architecture are given and description is not (the table is matched by vendor
 * and architecture, below every table matched by description); every record is of that vendor and
 * architecture; every record carries a description, the records of one description are of one
 * adapter, and the records hold at least two descriptions. Anything else throws.
 */
export const deriveSharedProfile = (
  sources: readonly SweepSource[],
  options: ProfileTarget & { readonly minSpeedup: number; readonly onBase: boolean },
): readonly SharedSlotVerdict[] => {
  if (sources.length === 0) throw new Error("No sweep records");
  const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
  const i8a8Segments = i8a8LinearSegments(I8A8_LINEAR_ROWS_BOUNDS);
  if (options.optIn === true) {
    throw new Error(
      "A shared table is matched by vendor and architecture: it cannot be an injection-only table",
    );
  }
  if (options.architecture === undefined) {
    throw new Error("A shared derivation needs both vendor and architecture");
  }
  if (options.description !== undefined) {
    throw new Error(
      "A shared table is matched without a description (it predicts for the adapters of its vendor and architecture that no table names by description): drop description",
    );
  }
  const target = { vendor: options.vendor, architecture: options.architecture };
  const seen = new Map<string, string>();
  for (const source of sources) assertRequestedSource(source, seen, target);
  const devices = sharedDevices(sources);
  if (devices.length < 2) {
    throw new Error(
      `A shared derivation needs the records of at least two devices (told apart by their descriptions); every record given is of ${devices[0].label} — derive a measured table of one adapter instead`,
    );
  }
  const grouped = devices.map((device) => ({
    label: device.label,
    ...slotRowsOf(device.sources, segments, i8a8Segments),
  }));
  const partial = options.onBase ? partialFamilies(devices) : [];
  return profileSlots(segments, i8a8Segments).map((spec) =>
    judgeSharedSlot(
      spec,
      grouped.map(({ label, bySlot, excludedBySlot }) => ({
        label,
        slotRows: bySlot.get(spec.slot) ?? [],
        excluded: excludedBySlot.get(spec.slot) ?? [],
      })),
      options.minSpeedup,
      partial.filter(({ family }) => family === profileSlotFamily(spec.slot)),
    ),
  );
};

/**
 * The verdict lines of a shared table (the same lines in the generated file's header and on
 * standard output): per field, the adopted geometry with its geometric mean over the devices and
 * each device's ratios, or the default and why; then per device its dropped cases and observations,
 * and the geometries not taken with each device's ratio or reason.
 */
export const sharedVerdictLines = (verdicts: readonly SharedSlotVerdict[]): string[] =>
  verdicts.flatMap((verdict) => {
    const counts = verdict.devices
      .map(({ label, cases }) => `${label} ${cases.length} cases`)
      .join(", ");
    const head = `${verdict.slot} (${verdict.scope}; ${counts})`;
    const { outcome } = verdict;
    const line =
      outcome.kind === "adopted"
        ? `- ${head}: adopted ${outcome.name} ${formatRatio(outcome.geomean)} over the devices (${outcome.perDevice
            .map(
              ({ label, geomean, min, max }) =>
                `${label} ${formatRatio(geomean)} (${formatRatio(min)}–${formatRatio(max)})`,
            )
            .join(", ")})`
        : `- ${head}: keeps the default ${outcome.name} (${outcome.reason})`;
    return [
      line,
      ...verdict.devices.flatMap(({ excluded }) =>
        excluded.map((entry) => `  - ${excludedCaseLine(entry)}`),
      ),
      ...verdict.rejected.map(({ name, reason }) => `  - ${name}: ${reason}`),
    ];
  });

/**
 * The list of verdicts (the same lines in the generated file's header comment and on standard
 * output).
 */
export const verdictLines = (verdicts: readonly SlotVerdict[]): string[] =>
  verdicts.flatMap((verdict) => {
    const head = `${verdict.slot} (${verdict.scope}, ${verdict.cases.length} cases)`;
    const { outcome } = verdict;
    const line =
      outcome.kind === "adopted"
        ? `- ${head}: adopted ${outcome.name} ${formatRatio(outcome.geomean)} (${formatRatio(
            outcome.min,
          )}–${formatRatio(outcome.max)})`
        : `- ${head}: keeps the default ${outcome.name} (${outcome.reason})`;
    return [
      line,
      ...verdict.excluded.map((excluded) => `  - ${excludedCaseLine(excluded)}`),
      ...verdict.rejected.map(({ name, reason }) => `  - ${name}: ${reason}`),
    ];
  });

// ---------------------------------------------------------------------------
// The second input: attention-bench records → `attentionOnline` (ADR 0117 addendum 2026-11-12)
// ---------------------------------------------------------------------------

/**
 * One attention-bench record given to the derivation (`deriveGeometryProfile`'s `attentionBench`):
 * the record (`karume-attention-bench/2` as `runAttentionRouteBench` returns it or `JSON.parse` of
 * its saved text — an `unknown` boundary, {@link parseAttentionBenchReport}) and the identification
 * that goes into the table's `provenance.attentionBench`.
 */
export type AttentionBenchInput = {
  readonly report: unknown;
  /** The record's name (a file path or the like). */
  readonly path: string;
  /** The SHA-256 of the record's bytes (hex — passing two records with the same value fails). */
  readonly sha256: string;
};

/** One paired timing of a record (A = the form N gives, B = tiled), as the derivation reads it. */
type BenchPair = {
  readonly keys: readonly [string, string];
  readonly reps?: number;
  readonly rounds: readonly { readonly a: readonly number[]; readonly b: readonly number[] }[];
  readonly error?: string;
};

/** One shape of a record, as the derivation reads it. */
type BenchShapeObservation = {
  readonly shape: AttentionBenchShape;
  /** The online route's reduction form on the measured device. */
  readonly reduce: AttentionOnlineReduce;
  /** The rows of the measured dispatch. */
  readonly rows: number;
  readonly kernel: BenchPair;
  readonly producer: BenchPair;
  readonly identity: {
    readonly identical: boolean;
    readonly differingWords?: number;
    readonly differingScaleWords?: number;
    readonly error?: string;
  };
};

/** One attention-bench record (checked at the `unknown` boundary). */
export type AttentionBenchSource = {
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly adapter: SweepAdapter;
  /** The integer dot-product variant the record measured (`variants.dp4a.dot`). */
  readonly dp4a: boolean;
  /** The timestamp quantization step (same unit as the passes — 100 µs or 0, {@link benchQuantum}). */
  readonly timestampQuantum: number;
  readonly shapes: readonly BenchShapeObservation[];
};

const isNonNegativeNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value >= 0;

const recordAt = (value: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`${where} is not an object`);
  return value;
};

const positiveIntegerAt = (record: Record<string, unknown>, key: string, where: string): number => {
  const value = record[key];
  if (!isPositiveInteger(value)) {
    throw new Error(`${where}: ${key} must be a positive integer (${JSON.stringify(value)})`);
  }
  return value;
};

/** A pair of the record (`onlineTiled.kernel` or `quantization.tiledVt`). */
const parseBenchPair = (value: unknown, where: string): BenchPair => {
  const pair = recordAt(value, where);
  const { keys } = pair;
  if (!Array.isArray(keys) || keys.length !== 2 || !keys.every(isString)) {
    throw new Error(`${where}: keys is not a pair of strings (${JSON.stringify(keys)})`);
  }
  const error = optional(pair, "error", isString, where);
  const reps = optional(pair, "reps", isPositiveInteger, where);
  const rounds = pair.rounds;
  if (rounds === undefined && error !== undefined) {
    return { keys: [keys[0], keys[1]], rounds: [], error };
  }
  if (!Array.isArray(rounds)) throw new Error(`${where}: rounds is not an array`);
  return {
    keys: [keys[0], keys[1]],
    ...(reps === undefined ? {} : { reps }),
    rounds: rounds.map((round, index) => {
      const at = `${where} rounds[${index}]`;
      const fields = recordAt(round, at);
      const passes = (arm: "a" | "b"): readonly number[] => {
        const values = fields[arm];
        if (!Array.isArray(values) || !values.every(isNonNegativeNumber)) {
          throw new Error(`${at}: ${arm} is not an array of non-negative numbers`);
        }
        return values;
      };
      return { a: passes("a"), b: passes("b") };
    }),
    ...(error === undefined ? {} : { error }),
  };
};

const parseBenchShape = (value: unknown, where: string): AttentionBenchShape => {
  const shape = recordAt(value, `${where} shape`);
  const rowsPerDispatch = optional(shape, "rowsPerDispatch", isPositiveInteger, where);
  const scale = shape.scale;
  if (typeof scale !== "number" || !Number.isFinite(scale)) {
    throw new Error(`${where}: shape.scale is not a finite number (${JSON.stringify(scale)})`);
  }
  return {
    id: requireString(shape, "id", `${where} shape`),
    heads: positiveIntegerAt(shape, "heads", `${where} shape`),
    queries: positiveIntegerAt(shape, "queries", `${where} shape`),
    keys: positiveIntegerAt(shape, "keys", `${where} shape`),
    depth: positiveIntegerAt(shape, "depth", `${where} shape`),
    scale,
    ...(rowsPerDispatch === undefined ? {} : { rowsPerDispatch }),
  };
};

const parseBenchShapeResult = (value: unknown, where: string): BenchShapeObservation => {
  const result = recordAt(value, where);
  const shape = parseBenchShape(result.shape, where);
  const at = `${where} (${shape.id})`;
  const online = recordAt(result.online, `${at} online`);
  if (online.eligible !== true || (online.reduce !== "sg32" && online.reduce !== "wg")) {
    throw new Error(
      `${at}: the online route did not run (${JSON.stringify(online.reason ?? online.error)}) — no observation of the key axis`,
    );
  }
  const tiled = recordAt(result.onlineTiled, `${at} onlineTiled`);
  if (tiled.eligible !== true) {
    throw new Error(
      `${at}: the tiled key axis did not run (${JSON.stringify(tiled.reason)}) — measure again`,
    );
  }
  const identity = recordAt(tiled.identity, `${at} onlineTiled.identity`);
  if (!isBoolean(identity.identical)) {
    throw new Error(`${at}: onlineTiled.identity.identical is not a boolean`);
  }
  const counted = (key: string): number | undefined => {
    const count = identity[key];
    if (count === undefined) return undefined;
    if (typeof count !== "number" || !Number.isSafeInteger(count) || count < 0) {
      throw new Error(
        `${at}: onlineTiled.identity.${key} is not a count (${JSON.stringify(count)})`,
      );
    }
    return count;
  };
  const differingWords = counted("differingWords");
  const differingScaleWords = counted("differingScaleWords");
  const identityError = optional(identity, "error", isString, `${at} onlineTiled.identity`);
  const quantization = recordAt(result.quantization, `${at} quantization`);
  return {
    shape,
    reduce: online.reduce,
    rows: positiveIntegerAt(tiled, "rows", `${at} onlineTiled`),
    kernel: parseBenchPair(tiled.kernel, `${at} onlineTiled.kernel`),
    producer: parseBenchPair(quantization.tiledVt, `${at} quantization.tiledVt`),
    identity: {
      identical: identity.identical,
      ...(differingWords === undefined ? {} : { differingWords }),
      ...(differingScaleWords === undefined ? {} : { differingScaleWords }),
      ...(identityError === undefined ? {} : { error: identityError }),
    },
  };
};

/**
 * The timestamp quantization step of a record: Chrome's 100 µs when the unit is ns and every pass of
 * both pairs at every shape is a multiple of it (the reading of report.ts's `roundsLookQuantized` —
 * the bench record carries no `quantized` flag), else 0.
 */
const benchQuantum = (unit: string, shapes: readonly BenchShapeObservation[]): number => {
  if (unit !== "ns") return 0;
  const passes = shapes.flatMap(({ kernel, producer }) =>
    [kernel, producer].flatMap(({ rounds }) => rounds.flatMap(({ a, b }) => [...a, ...b])),
  );
  const total = passes.reduce((sum, pass) => sum + pass, 0);
  return total > 0 && passes.every((pass) => pass % CHROME_TIMESTAMP_QUANTUM_NS === 0)
    ? CHROME_TIMESTAMP_QUANTUM_NS
    : 0;
};

/**
 * Reads an attention-bench record (the `unknown` boundary — checks only what the derivation reads and
 * fails loudly). A file of the playground's `/attention-bench` page holds the record under `bench`:
 * pass that value.
 */
export const parseAttentionBenchReport = (
  parsed: unknown,
  meta: { readonly path: string; readonly sha256: string },
): AttentionBenchSource => {
  const where = meta.path;
  const record = recordAt(parsed, `${where}: the JSON`);
  if (record.format !== ATTENTION_BENCH_FORMAT) {
    throw new Error(
      `${where}: format is not ${ATTENTION_BENCH_FORMAT} (${JSON.stringify(record.format)}${
        isRecord(record.bench)
          ? " — a playground file holds the record under `bench`; pass that"
          : ""
      })`,
    );
  }
  const timing = recordAt(record.timing, `${where} timing`);
  // MUST: GPU timestamps only (ADR 0117 decision 3): the wall clock adds the submit-to-completion
  // floor, which moves per pass, so a ratio of minimums cannot be bounded from the record
  if (timing.unit !== "ns" && timing.unit !== "deno-raw-tick") {
    throw new Error(
      `${where}: timing.unit is ${JSON.stringify(timing.unit)} (must be ns or deno-raw-tick — a wall-clock record is not read; acquire the device with gpuTiming: true)`,
    );
  }
  const settings = recordAt(record.settings, `${where} settings`);
  if (settings.onlineTiled !== true || settings.quantization !== true) {
    throw new Error(
      `${where}: settings.onlineTiled and settings.quantization must both be true (the kernel pair and the producer pair)`,
    );
  }
  const adapter = recordAt(record.adapter, `${where} adapter`);
  const variants = recordAt(record.variants, `${where} variants`);
  const dot = recordAt(variants.dp4a, `${where} variants.dp4a`).dot;
  if (dot !== "dp4a" && dot !== "emu") {
    throw new Error(`${where}: variants.dp4a.dot is neither dp4a nor emu (${JSON.stringify(dot)})`);
  }
  if (!Array.isArray(record.shapes)) throw new Error(`${where}: shapes is not an array`);
  const shapes = record.shapes.map((entry, index) =>
    parseBenchShapeResult(entry, `${where} shapes[${index}]`),
  );
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: requireString(record, "date", where),
    adapter: {
      vendor: requireString(adapter, "vendor", `${where} adapter`),
      architecture: requireString(adapter, "architecture", `${where} adapter`),
      device: requireString(adapter, "device", `${where} adapter`),
      description: requireString(adapter, "description", `${where} adapter`),
    },
    dp4a: dot === "dp4a",
    timestampQuantum: benchQuantum(timing.unit, shapes),
    shapes,
  };
};

/** One record's observation of one shape: its gain, or why it is no ratio material. */
export type AttentionBenchObservation = {
  readonly path: string;
  /** (N's node + N's producer) ÷ (tiled node + tiled producer), from each arm's minimum pass. */
  readonly gain?: number;
  /** Why the observation is dropped from the ratio material. */
  readonly excluded?: string;
};

/** The verdict on `attentionOnline` ({@link deriveAttentionOnline}). */
export type AttentionOnlineVerdict = {
  /** Per shape of the bench's set, in its order: the combined gain and every observation. */
  readonly shapes: readonly {
    readonly id: string;
    /** The geometric mean of the kept observations' gains (absent: none kept). */
    readonly gain?: number;
    readonly observations: readonly AttentionBenchObservation[];
  }[];
  /** The observations that were not identical (record, shape, what differed). */
  readonly mismatches: readonly string[];
  readonly outcome:
    | {
        readonly kind: "adopted";
        /** The geometric mean of the shapes' gains, and the smallest and largest. */
        readonly geomean: number;
        readonly min: number;
        readonly max: number;
      }
    | { readonly kind: "default"; readonly reason: string };
};

/** The minimum positive pass of an arm over every round (undefined when none). */
const armMinimum = (pair: BenchPair, arm: "a" | "b"): number | undefined => {
  const kept = pair.rounds.flatMap((round) => round[arm]).filter((pass) => pass > 0);
  return kept.length === 0 ? undefined : Math.min(...kept);
};

/** Whether two shapes are the same measurement (every field — the shape set's id hashes them all). */
const sameBenchShape = (left: AttentionBenchShape, right: AttentionBenchShape): boolean =>
  left.id === right.id &&
  left.heads === right.heads &&
  left.queries === right.queries &&
  left.keys === right.keys &&
  left.depth === right.depth &&
  left.scale === right.scale &&
  left.rowsPerDispatch === right.rowsPerDispatch;

/**
 * The record checks of one observation against the current runtime (fail loudly — measure again):
 * the shape is one of the bench's set, the pairs measured the current runtime's kernels and
 * producers for the record's dp4a variant and the online route's reduction form, and the measured
 * rows are the first window production runs.
 */
const assertCurrentObservation = (
  source: AttentionBenchSource,
  observation: BenchShapeObservation,
): void => {
  const { shape } = observation;
  const where = `${source.path} (${shape.id})`;
  const current = ATTENTION_BENCH_SHAPES.find((entry) => entry.id === shape.id);
  if (current === undefined) {
    throw new Error(
      `${where}: the shape is not in the bench's shape set (${ATTENTION_BENCH_SHAPES.map(({ id }) => id).join(", ")})`,
    );
  }
  if (!sameBenchShape(current, shape)) {
    throw new Error(
      `${where}: the shape's dimensions ${JSON.stringify(shape)} differ from the bench's ${JSON.stringify(current)} — measure again with the current runtime`,
    );
  }
  const { heads, queries, keys, depth } = shape;
  const own = attentionOnlineKeyWords(keys);
  const expected = {
    kernel: [
      attentionOnlineI8a8Key(source.dp4a, depth, observation.reduce, "bhsd", own),
      attentionOnlineI8a8Key(source.dp4a, depth, observation.reduce, "bhsd", "tiled"),
    ],
    producer: [
      attentionValueQuantization(heads * depth, keys, own).key,
      attentionValueQuantization(heads * depth, keys, "tiled").key,
    ],
  } as const;
  for (const pair of ["kernel", "producer"] as const) {
    const [a, b] = observation[pair].keys;
    if (a !== expected[pair][0] || b !== expected[pair][1]) {
      throw new Error(
        `${where}: the ${pair} pair measured ${a} / ${b}, not the current runtime's ${expected[pair][0]} / ${expected[pair][1]} — measure again with the current runtime`,
      );
    }
  }
  const windows = planAttentionOnlineWindows(queries, { batch: heads, cols: keys, depth });
  const rows = onlineDispatchRows(windows[0].rows, queries);
  if (observation.rows !== rows) {
    throw new Error(
      `${where}: the measured dispatch covers ${observation.rows} rows, not the ${rows} of the first window production runs`,
    );
  }
};

/** Why an observation is not identical (undefined when it is). */
const identityMismatch = (observation: BenchShapeObservation): string | undefined => {
  const { identity } = observation;
  if (identity.error !== undefined) return `the identity block failed (${identity.error})`;
  if (identity.identical && identity.differingWords === 0 && identity.differingScaleWords === 0) {
    return undefined;
  }
  return `${identity.differingWords ?? "?"} differing O words, ${identity.differingScaleWords ?? "?"} differing scale words`;
};

/**
 * The gain of one observation, or why it is no ratio material: a failed pair, a pass without a
 * positive minimum, or a rounding-error bound above {@link ROUNDING_ERROR_LIMIT} on either pair
 * (E = q ÷ min(A) + q ÷ min(B) — ADR 0117 decision 3).
 */
const observationGain = (
  source: AttentionBenchSource,
  observation: BenchShapeObservation,
): AttentionBenchObservation => {
  const { path } = source;
  const minima: number[] = [];
  for (const pair of ["kernel", "producer"] as const) {
    const timing = observation[pair];
    if (timing.error !== undefined)
      return { path, excluded: `the ${pair} pair failed (${timing.error})` };
    const a = armMinimum(timing, "a");
    const b = armMinimum(timing, "b");
    if (a === undefined || b === undefined || timing.reps === undefined) {
      return { path, excluded: `the ${pair} pair has no positive pass` };
    }
    const bound = source.timestampQuantum / a + source.timestampQuantum / b;
    if (bound > ROUNDING_ERROR_LIMIT) {
      return {
        path,
        excluded: `rounding-error bound E ${formatPercent(bound)} of the ${pair} pair (> ${LIMIT_LABEL})`,
      };
    }
    minima.push(a / timing.reps, b / timing.reps);
  }
  const [kernelA, kernelB, producerA, producerB] = minima;
  // The node runs the kernel over every query row; the measured dispatch is the first window
  const scale = observation.shape.queries / observation.rows;
  return { path, gain: (kernelA * scale + producerA) / (kernelB * scale + producerB) };
};

/**
 * The verdict on `attentionOnline` from attention-bench records (a pure function — the rule of the
 * module doc, ADR 0117 addendum 2026-11-12): adopted (the tiled key axis) only when every shape of
 * every record is identical, every shape of `ATTENTION_BENCH_SHAPES` has ratio material, and every
 * shape's gain (the geometric mean over the records that measured it) is at least `minSpeedup`.
 *
 * MUST: records are refused, not skipped, when they cannot stand for the current runtime on this
 * adapter: another adapter than the sweeps' (`adapter`), a record given twice, a shape outside the
 * bench's set or with other dimensions, other kernels or producers than the current runtime builds,
 * another measured window, or a set the records do not cover together.
 */
export const deriveAttentionOnline = (
  sources: readonly AttentionBenchSource[],
  adapter: SweepAdapter,
  minSpeedup: number,
): AttentionOnlineVerdict => {
  if (sources.length === 0) throw new Error("No attention-bench records");
  const seen = new Map<string, string>();
  for (const source of sources) {
    const twin = seen.get(source.sha256);
    if (twin !== undefined) {
      throw new Error(
        `The same attention-bench record is passed twice (${twin} and ${source.path})`,
      );
    }
    seen.set(source.sha256, source.path);
    // MUST: the field is a per-adapter answer like the geometries (the RTX 3080 Ti adopts, the RTX
    // 5070 Ti does not — perf-ledger K-106)
    for (const field of ["vendor", "architecture", "device", "description"] as const) {
      if (source.adapter[field] !== adapter[field]) {
        throw new Error(
          `${source.path}: adapter ${field} ${JSON.stringify(source.adapter[field])} differs from the sweep records' ${JSON.stringify(adapter[field])}`,
        );
      }
    }
    const ids = new Set<string>();
    for (const observation of source.shapes) {
      if (ids.has(observation.shape.id)) {
        throw new Error(`${source.path}: shape ${observation.shape.id} appears twice`);
      }
      ids.add(observation.shape.id);
      assertCurrentObservation(source, observation);
    }
  }
  const measured = new Set(sources.flatMap(({ shapes }) => shapes.map(({ shape }) => shape.id)));
  const missing = ATTENTION_BENCH_SHAPES.filter(({ id }) => !measured.has(id));
  if (missing.length > 0) {
    throw new Error(
      `The attention-bench records do not cover the bench's shape set: ${missing.map(({ id }) => id).join(", ")} not measured`,
    );
  }
  const mismatches = sources.flatMap((source) =>
    source.shapes.flatMap((observation) => {
      const mismatch = identityMismatch(observation);
      return mismatch === undefined
        ? []
        : [`${source.path} (${observation.shape.id}): ${mismatch}`];
    }),
  );
  const shapes = ATTENTION_BENCH_SHAPES.map(({ id }) => {
    const observations = sources.flatMap((source) =>
      source.shapes
        .filter((observation) => observation.shape.id === id)
        .map((observation) => observationGain(source, observation)),
    );
    const gains = observations.flatMap(({ gain }) => (gain === undefined ? [] : [gain]));
    return {
      id,
      ...(gains.length === 0 ? {} : { gain: geometricMean(gains) }),
      observations,
    };
  });
  return { shapes, mismatches, outcome: attentionOnlineOutcome(shapes, mismatches, minSpeedup) };
};

/** The outcome of {@link deriveAttentionOnline} from its per-shape gains and its mismatches. */
const attentionOnlineOutcome = (
  shapes: AttentionOnlineVerdict["shapes"],
  mismatches: readonly string[],
  minSpeedup: number,
): AttentionOnlineVerdict["outcome"] => {
  if (mismatches.length > 0) {
    return { kind: "default", reason: `not identical in ${mismatches.length} observation(s)` };
  }
  const bare = shapes.find((shape) => shape.gain === undefined);
  if (bare !== undefined) {
    return {
      kind: "default",
      reason: `${bare.id}: every observation was dropped from the ratio material`,
    };
  }
  const gains = shapes.map(({ gain }) => gain ?? 0);
  const worst = shapes.reduce((low, shape) => ((shape.gain ?? 0) < (low.gain ?? 0) ? shape : low));
  if ((worst.gain ?? 0) < minSpeedup) {
    const [value, bar] = formatRatioPair(worst.gain ?? 0, minSpeedup);
    return { kind: "default", reason: `${worst.id}: ${value} < ${bar}` };
  }
  return {
    kind: "adopted",
    geomean: geometricMean(gains),
    min: Math.min(...gains),
    max: Math.max(...gains),
  };
};

/** The verdict lines of `attentionOnline` (after the geometry fields' — {@link verdictLines}). */
export const attentionOnlineVerdictLines = (verdict: AttentionOnlineVerdict): string[] => {
  const records = new Set(
    verdict.shapes.flatMap(({ observations }) => observations.map(({ path }) => path)),
  ).size;
  const head = `attentionOnline (the online attention's Vᵀ key axis, kernel + producer per node, ${verdict.shapes.length} shapes, ${records} records)`;
  const { outcome } = verdict;
  return [
    outcome.kind === "adopted"
      ? `- ${head}: adopted tiled ${formatRatio(outcome.geomean)} (${formatRatio(outcome.min)}–${formatRatio(outcome.max)})`
      : `- ${head}: keeps the form N gives (${outcome.reason})`,
    ...verdict.mismatches.map((mismatch) => `  - not identical: ${mismatch}`),
    ...verdict.shapes.map(
      ({ id, gain, observations }) =>
        `  - ${id}: ${gain === undefined ? "no ratio material" : formatRatio(gain)} (${observations
          .map(({ path, gain: observed, excluded }) =>
            observed === undefined
              ? `${path} dropped: ${excluded ?? "?"}`
              : `${path} ${formatRatio(observed)}`,
          )
          .join("; ")})`,
    ),
  ];
};

/** The bench records and their verdict, as {@link buildGeometryProfile} writes them. */
export type AttentionBenchDerivation = {
  readonly sources: readonly AttentionBenchSource[];
  readonly verdict: AttentionOnlineVerdict;
};

/** Reads the attention-bench inputs and derives their verdict (the second input of the derivation). */
const deriveAttentionBench = (
  inputs: readonly AttentionBenchInput[],
  adapter: SweepAdapter,
  minSpeedup: number,
): AttentionBenchDerivation => {
  const sources = inputs.map(({ report, path, sha256 }) =>
    parseAttentionBenchReport(report, { path, sha256 }),
  );
  return { sources, verdict: deriveAttentionOnline(sources, adapter, minSpeedup) };
};

/** A table built from sweeps ({@link buildGeometryProfile} — always carries `provenance`). */
export type GeneratedProfile = GeometryProfile & {
  readonly provenance: NonNullable<GeometryProfile["provenance"]>;
};

/** The attention-bench block of a table's provenance. */
type AttentionBenchProvenance = NonNullable<GeneratedProfile["provenance"]["attentionBench"]>;

/**
 * What a derivation carries over from an existing table (`DeriveGeometryProfileOptions.base`,
 * checked by {@link carryOver} — ADR 0117 addendum 2026-11-12): the families the records do not
 * cover whose provenance entry in the base is current, with that entry, and the base's
 * attention-bench block when no attention-bench records are given (its `attentionOnline` goes with
 * it).
 */
export type CarryOver = {
  readonly base: GeneratedProfile;
  readonly families: readonly {
    readonly family: GeometryProfileFamily;
    readonly entry: GeometryProfileFamilyProvenance;
  }[];
  readonly attentionBench?: AttentionBenchProvenance;
};

/** The fields of `family` in `table` replaced by `base`'s, verbatim. */
const withBaseFields = (
  table: GeometryProfile,
  base: GeometryProfile,
  family: GeometryProfileFamily,
): GeometryProfile => {
  switch (family) {
    case "gemm":
      return { ...table, gemmRows: base.gemmRows };
    case "attention":
      return { ...table, attention: base.attention };
    case "conv2d":
      return { ...table, conv2d: base.conv2d };
    case "i8a8-linear":
      return { ...table, i8a8: { ...table.i8a8, linear: base.i8a8.linear } };
    case "i8a8-attention":
      return {
        ...table,
        i8a8: {
          ...table.i8a8,
          attentionQk: base.i8a8.attentionQk,
          attentionPv: base.i8a8.attentionPv,
        },
      };
    case "i8a8-conv3d":
      return { ...table, i8a8: { ...table.i8a8, conv3d: base.i8a8.conv3d } };
  }
};

/**
 * The checks of `base` against the records and the current runtime, and what it carries over
 * ({@link CarryOver} — ADR 0117 addendum 2026-11-12). Every refusal throws (fail loudly):
 *
 * - `base` has `provenance`, its adapter equals the records' four fields, and its
 *   `provenance.families` is an object (not a table saved in the earlier shape).
 * - Every family the records cover has every case of the current case set in the records: carrying
 *   over promises that nothing else moves, so a half-swept family must not silently put its unswept
 *   fields back to the default.
 * - Per family the records do not cover: with an entry, its `kernels` and `caseSet` equal the
 *   current runtime's for `base`'s fields (a stale entry: sweep this family again) — the family is
 *   carried; without an entry, `base`'s fields of the family build the default table's kernels (a
 *   table edited by hand otherwise) — the family keeps the default.
 * - Without attention-bench records: `base`'s block is carried after its `shapes` (and `kernels`,
 *   when present) are checked against the current runtime (stale: measure the attention bench
 *   again); a `base` carrying `attentionOnline` without the block or without its `kernels`, and a
 *   block with `kernels` on a `base` without `attentionOnline`, are refused.
 * - A shared derivation (`shared`) takes only a shared table whose adapters are the records'
 *   adapters (as a set). The completeness check above is the measured derivation's: a family one
 *   adapter covers only in part keeps the default in a shared derivation, named in its verdicts
 *   ({@link deriveSharedProfile} with `onBase`), and a covered family one adapter has no row of
 *   (which keeps the default) is refused unless the base holds the default for it. A measured
 *   derivation takes only a measured table.
 */
const carryOver = (
  base: GeometryProfile,
  sources: readonly SweepSource[],
  benchGiven: boolean,
  shared: boolean,
): CarryOver => {
  const where = `Base table '${base.id}'`;
  const { provenance } = base;
  if (provenance === undefined) {
    throw new Error(
      `${where} has no provenance (nothing to carry over — a base is a table derived from sweeps)`,
    );
  }
  // A measured derivation's records are one adapter's whatever their description (deriveProfile)
  const devices = shared ? sharedDevices(sources) : [];
  if (provenance.adapters !== undefined) {
    if (!shared) {
      throw new Error(
        `${where} is a shared table (derived from ${provenance.adapters.length} adapters): it is regenerated by a shared derivation`,
      );
    }
    // A shared table's families all stand on the same devices: a family carried onto records of
    // other devices would cite evidence the table's adapters do not name
    const keys = (adapters: readonly SweepAdapter[]): string =>
      adapters.map(adapterKey).sort().join("\n");
    if (keys(provenance.adapters) !== keys(devices.map((device) => device.adapter))) {
      throw new Error(
        `${where}: its adapters (${provenance.adapters.map((adapter) => ADAPTER_FIELDS.map((field) => adapter[field]).join(" / ")).join("; ")}) are not the sweep records' (${devices.map((device) => ADAPTER_FIELDS.map((field) => device.adapter[field]).join(" / ")).join("; ")}) — a shared table is carried over only onto records of the same adapters`,
      );
    }
  } else {
    if (shared) {
      throw new Error(
        `${where} is a measured table (one adapter): a shared derivation does not take it as its base`,
      );
    }
    const adapter = sources[0].adapter;
    for (const field of ADAPTER_FIELDS) {
      if (provenance.adapter[field] !== adapter[field]) {
        throw new Error(
          `${where}: adapter ${field} ${JSON.stringify(provenance.adapter[field])} differs from the sweep records' ${JSON.stringify(adapter[field])} (a table is carried over only onto records of the same adapter)`,
        );
      }
    }
  }
  // The type says `families` is an object, but a table saved by an earlier runtime (the single
  // top-level keys) and read back with JSON.parse and a cast has none (the same check as matching's)
  const savedFamilies: unknown = provenance.families;
  if (typeof savedFamilies !== "object" || savedFamilies === null) {
    throw new Error(
      `${where}: its provenance has no per-family entries (provenance.families is ${JSON.stringify(savedFamilies)} — a table saved by an earlier runtime; sweep again)`,
    );
  }
  const rows = sources.flatMap((source) => source.rows);
  const covered = familiesOf(rows);
  // A measured table's records cover a family they have a row of completely. A shared table's family
  // that one device covers only in part keeps the default, named in its verdicts
  // (deriveSharedProfile with onBase — partialFamilies); one that a device has no row of keeps the
  // default too, so it is derived only where the base holds the default for it: elsewhere it would
  // silently go back to the default
  const groups = shared
    ? devices.map((device) => ({
        label: device.label,
        rows: device.sources.flatMap((source) => source.rows),
      }))
    : [{ label: "", rows }];
  for (const group of groups) {
    const swept = new Set(group.rows.map((row) => row.caseId));
    const own = familiesOf(group.rows);
    for (const family of covered) {
      if (shared) {
        if (
          !own.includes(family) &&
          familyKernelsId(base, family) !== familyKernelsId(DEFAULT_GEOMETRY_PROFILE, family)
        ) {
          throw new Error(
            `The records of ${group.label} have no row of family ${family}, and the base table '${base.id}' holds other than the default there: a shared field takes a geometry only where every device measured it, ` +
              "so the family would silently go back to the default — sweep it on that device too",
          );
        }
        continue;
      }
      const missing = familyEntries(SWEEP_CASES, family).filter(({ id }) => !swept.has(id));
      if (missing.length > 0) {
        throw new Error(
          `The records cover family ${family} only in part (no row of ${missing.map(({ id }) => id).join(", ")}): with a base, a covered family is derived from every case of its current case set, ` +
            "or its unswept fields would silently go back to the default — sweep the missing cases",
        );
      }
    }
  }
  const families: CarryOver["families"][number][] = [];
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    if (covered.includes(family)) continue;
    const at = `${where}, family ${family}`;
    let kernels: string;
    try {
      kernels = familyKernelsId(base, family);
    } catch (cause) {
      if (cause instanceof KernelsIdError) {
        throw new Error(
          `${at}: its geometry cannot build the kernels of the sweep shapes (${cause.message}) — sweep this family again`,
          { cause },
        );
      }
      throw cause;
    }
    const entry = provenance.families[family];
    if (entry === undefined) {
      const defaults = familyKernelsId(DEFAULT_GEOMETRY_PROFILE, family);
      if (kernels !== defaults) {
        throw new Error(
          `${where} has no provenance for family ${family} but its fields are not the default (fingerprint ${kernels} / the default table's ${defaults} — a table edited by hand, or the runtime's default geometry of the family changed since the table was written; sweep this family)`,
        );
      }
      continue;
    }
    if (entry.kernels !== kernels) {
      throw new Error(
        `${at} is stale: its kernel fingerprint ${entry.kernels} differs from the current runtime's ${kernels} — sweep this family again`,
      );
    }
    const caseSet = familyCaseSetId(family);
    if (entry.caseSet !== caseSet) {
      throw new Error(
        `${at} is stale: its case-set version ${entry.caseSet} differs from the current runtime's ${caseSet} — sweep this family again`,
      );
    }
    families.push({ family, entry });
  }
  const generated = { ...base, provenance };
  if (benchGiven) return { base: generated, families };
  const attentionBench = carriedAttentionBench(generated);
  return attentionBench === undefined
    ? { base: generated, families }
    : { base: generated, families, attentionBench };
};

/**
 * The attention-bench block `base` carries over when no attention-bench records are given (checked
 * against the current runtime — {@link carryOver}); undefined when `base` has none.
 */
const carriedAttentionBench = (base: GeneratedProfile): AttentionBenchProvenance | undefined => {
  const where = `Base table '${base.id}'`;
  const block = base.provenance.attentionBench;
  if (block === undefined) {
    if (base.attentionOnline !== undefined) {
      throw new Error(
        `${where} carries attentionOnline but no provenance.attentionBench (nothing to check the form against) — measure the attention bench again`,
      );
    }
    return undefined;
  }
  if (base.attentionOnline !== undefined && block.kernels === undefined) {
    throw new Error(
      `${where} carries attentionOnline but its provenance.attentionBench has no kernels — measure the attention bench again`,
    );
  }
  if (base.attentionOnline === undefined && block.kernels !== undefined) {
    throw new Error(
      `${where}: its provenance.attentionBench names the online attention's kernels but the table chooses no form (no attentionOnline — the block does not belong to the table's fields) — measure the attention bench again`,
    );
  }
  const shapes = attentionBenchShapeSetId();
  if (block.shapes !== shapes) {
    throw new Error(
      `${where}: the attention bench's shape set ${block.shapes} differs from the current runtime's ${shapes} — measure the attention bench again`,
    );
  }
  if (block.kernels !== undefined) {
    const kernels = attentionBenchKernelsId();
    if (block.kernels !== kernels) {
      throw new Error(
        `${where}: the online attention's kernel fingerprint ${block.kernels} differs from the current runtime's ${kernels} — measure the attention bench again`,
      );
    }
  }
  return block;
};

/** What a table's field takes from a verdict: a measured table's or a shared table's. */
type SlotChoice = {
  readonly slot: ProfileSlot;
  readonly outcome: { readonly geometry: Geometry };
};

const slotGeometry = (verdicts: readonly SlotChoice[], slot: ProfileSlot): Geometry => {
  const verdict = verdicts.find((entry) => entry.slot === slot);
  if (verdict === undefined) throw new Error(`${slot} has no verdict`);
  return verdict.outcome.geometry;
};

const gemmSlot = (verdicts: readonly SlotChoice[], slot: ProfileSlot): GemmGeometry => {
  const geometry = slotGeometry(verdicts, slot);
  if ("tileK" in geometry) throw new Error(`${slot} holds an i8a8 geometry`);
  return geometry;
};

const i8a8Slot = (verdicts: readonly SlotChoice[], slot: ProfileSlot): I8a8Geometry => {
  const geometry = slotGeometry(verdicts, slot);
  if (!("tileK" in geometry)) throw new Error(`${slot} holds a geometry of the f32 skeleton`);
  return geometry;
};

/**
 * Builds the table's values from the verdicts (the source of truth both for the values of the
 * generated TS registered in the repository (`tools/geometry-sweep/render.ts`, removed in 0.15) and
 * for the values injected into `acquireGpu({ geometryProfile })`).
 *
 * MUST: only this function writes `provenance` (ADR 0117 decision 4, per op family since its
 * addendum 2026-11-12). Per family the records cover (some record has a row of it), the entry joins
 * the paths / sha256 / dates / candidate sets of **the records with a row of the family** with
 * `", "` in the order given, its userAgent is their deduplicated array ({@link distinctUserAgents}),
 * and its kernel fingerprint and case-set version are derived with the current runtime over the
 * family's cases (matching, `geometryProfileMismatch`, re-derives them with the same functions). A
 * family carried over (`carry` — {@link carryOver}) takes the base's fields and entry verbatim. Any
 * other family gets no entry, and its fields must build the default table's kernels (they are the
 * default by the verdicts — a failure here means the verdicts were not the derivation's). adapter
 * is the four fields as they are ({@link deriveProfile} has checked that every record agrees); a
 * shared table (`spec.shared` — {@link deriveSharedProfile}) has `adapters` instead, each adapter
 * of the records once in the order its first record was given.
 * With attention-bench records (`attention` — ADR 0117 addendum 2026-11-12), the table carries
 * `attentionOnline: { keyWords: "tiled" }` exactly when their verdict adopted it, and
 * `provenance.attentionBench` (the records' paths / sha256 / dates joined with `", "`, the id of the
 * bench's shape set, and the online attention's kernel fingerprint exactly when the field is
 * adopted) whatever the verdict. Without them, a carried block brings the base's `attentionOnline`
 * (if any) with it. A carried base brings its `linearGemv` as it is (no sweep record decides it).
 */
export const buildGeometryProfile = (
  spec: { readonly id: string; readonly shared?: boolean } & ProfileTarget,
  sources: readonly SweepSource[],
  verdicts: readonly SlotChoice[],
  attention?: AttentionBenchDerivation,
  carry?: CarryOver,
): GeneratedProfile => {
  // MUST: the last rule covers every row count (the same condition as the runtime's check — fail
  // early at generation). profileRowsSegments checks the trailing Infinity and the refinement of
  // the default
  const segments = profileRowsSegments(PROFILE_GEMM_ROWS_BOUNDS);
  const i8a8Segments = i8a8LinearSegments(I8A8_LINEAR_ROWS_BOUNDS);
  if (sources.length === 0) throw new Error("No sweep records (provenance cannot be written)");
  const derived: GeometryProfile = {
    id: spec.id,
    ...(spec.optIn === true
      ? {}
      : {
          match: {
            vendor: spec.vendor,
            ...(spec.architecture === undefined ? {} : { architecture: spec.architecture }),
            ...(spec.description === undefined ? {} : { description: spec.description }),
          },
        }),
    gemmRows: segments.map((segment) => ({
      maxRows: segment.maxRows,
      geometry: gemmSlot(verdicts, segment.slot),
    })),
    attention: {
      qk: gemmSlot(verdicts, "attention.qk"),
      pv: gemmSlot(verdicts, "attention.pv"),
    },
    conv2d: {
      rows64: gemmSlot(verdicts, "conv2d.rows64"),
      rows32: gemmSlot(verdicts, "conv2d.rows32"),
    },
    i8a8: {
      linear: i8a8Segments.map((segment) => ({
        maxRows: segment.maxRows,
        geometry: i8a8Slot(verdicts, segment.slot),
      })),
      attentionQk: i8a8Slot(verdicts, "i8a8.attentionQk"),
      attentionPv: i8a8Slot(verdicts, "i8a8.attentionPv"),
      conv3d: i8a8Slot(verdicts, "i8a8.conv3d"),
    },
  };
  const fields =
    carry === undefined
      ? derived
      : carry.families.reduce(
          (table, { family }) => withBaseFields(table, carry.base, family),
          derived,
        );
  const adopted = attention?.verdict.outcome.kind === "adopted";
  const attentionOnline =
    attention === undefined
      ? carry?.attentionBench === undefined
        ? undefined
        : carry.base.attentionOnline
      : adopted
        ? ({ keyWords: "tiled" } as const)
        : undefined;
  // The GEMV choices come from GEMV sweep records (`deriveLinearGemvChoices`), never from these: a
  // base's field is carried as it is, so a regeneration of the tile geometry does not drop it
  const linearGemv = carry?.base.linearGemv;
  // The grid-fill rule likewise comes from grid sweep records (`deriveGemmGrid`) and is carried as it
  // is: a regeneration of the M buckets leaves it in force
  const gemmGrid = carry?.base.gemmGrid;
  const table: GeometryProfile = {
    ...fields,
    ...(attentionOnline === undefined ? {} : { attentionOnline }),
    ...(linearGemv === undefined ? {} : { linearGemv }),
    ...(gemmGrid === undefined ? {} : { gemmGrid }),
  };
  const attentionBench: AttentionBenchProvenance | undefined =
    attention === undefined
      ? carry?.attentionBench
      : {
          records: attention.sources.map((source) => source.path).join(", "),
          sha256: attention.sources.map((source) => source.sha256).join(", "),
          date: attention.sources.map((source) => source.date).join(", "),
          shapes: attentionBenchShapeSetId(),
          ...(adopted ? { kernels: generatedAttentionKernelsId(table) } : {}),
        };
  const adapterOf = ({ vendor, architecture, device, description }: SweepAdapter) => ({
    vendor,
    architecture,
    device,
    description,
  });
  const adapters =
    spec.shared === true ? sharedDevices(sources).map((device) => adapterOf(device.adapter)) : [];
  if (spec.shared === true && adapters.length < 2) {
    throw new Error(
      `Generated table '${spec.id}': a shared table needs the records of two or more adapters (${adapters.length} given)`,
    );
  }
  const generated: GeneratedProfile = {
    ...table,
    provenance: {
      ...(spec.shared === true ? { adapters } : { adapter: adapterOf(sources[0].adapter) }),
      families: generatedFamilies(table, sources, carry),
      ...(attentionBench === undefined ? {} : { attentionBench }),
    },
  };
  assertGeneratedProfile(generated);
  return generated;
};

/**
 * The generated table's `provenance.families` ({@link buildGeometryProfile}): per family in the
 * fixed order, an entry from the records with a row of it, the carried entry, or none (then the
 * family's fields must build the default table's kernels).
 */
const generatedFamilies = (
  table: GeometryProfile,
  sources: readonly SweepSource[],
  carry: CarryOver | undefined,
): GeneratedProfile["provenance"]["families"] => {
  const families: { [Family in GeometryProfileFamily]?: GeometryProfileFamilyProvenance } = {};
  for (const family of GEOMETRY_PROFILE_FAMILIES) {
    const kernels = generatedKernelsId(table, family);
    const swept = sources.filter((source) => familyEntries(source.rows, family).length > 0);
    if (swept.length > 0) {
      const joined = (pick: (source: SweepSource) => string): string => swept.map(pick).join(", ");
      const userAgent = distinctUserAgents(swept);
      families[family] = {
        sweep: joined((source) => source.path),
        sha256: joined((source) => source.sha256),
        date: joined((source) => source.date),
        candidateSet: joined((source) => source.candidateSet),
        ...(userAgent === undefined ? {} : { userAgent }),
        kernels,
        caseSet: familyCaseSetId(family),
      };
      continue;
    }
    const carried = carry?.families.find((entry) => entry.family === family);
    if (carried !== undefined) {
      // MUST: the carried entry vouches for the fields actually written. carryOver checked the entry
      // against `base`'s fields of the family through the family's cases, but withBaseFields copies
      // them by its own field → family switch, a second encoding beside PROFILE_FIELD_FAMILIES; if
      // the two disagree, the table would carry an entry for fields it does not hold (an
      // implementation error, not bad input)
      if (carried.entry.kernels !== kernels) {
        throw new Error(
          `Generated table '${table.id}': the entry of family ${family} carried over from the base has kernel fingerprint ${carried.entry.kernels}, but the fields written for it build ${kernels} (an implementation error: the carried fields and the family map disagree)`,
        );
      }
      families[family] = carried.entry;
      continue;
    }
    if (kernels !== familyKernelsId(DEFAULT_GEOMETRY_PROFILE, family)) {
      throw new Error(
        `Generated table '${table.id}': family ${family} has no row in the records and nothing carried over, but its fields are not the default (no provenance could stand for them)`,
      );
    }
  }
  return families;
};

/**
 * Runs the generated table through the injection check (`assertGeometryProfile` — the check that
 * `acquireGpu` and reading back a stored table go through). A table that fails it (an empty id and
 * the like — the public generator writes the user's options as they are) is stopped at generation
 * instead of being carried to injection or read-back. The conditions live only in that one check
 * (not duplicated on the generator side).
 */
const assertGeneratedProfile = (profile: GeneratedProfile): void => {
  try {
    assertGeometryProfile(profile);
  } catch (cause) {
    if (cause instanceof CodegenError) {
      throw new Error(
        `Generated table '${profile.id}' fails the injection check — ${cause.message}`,
        {
          cause,
        },
      );
    }
    throw cause;
  }
};

/**
 * A family entry's `userAgent` (ADR 0117 addendum 2026-10-02): the records' `userAgent` as an
 * array deduplicated in order of appearance (one element if they are all the same — directly
 * comparable with the current browser). It is not joined with `", "` like the other fields because
 * a browser's userAgent itself contains `", "` (`(KHTML, like Gecko)`), and joined values could not
 * be split again. If any record lacks `userAgent`, nothing is written (a browser of only some
 * records is not kept as the value of the whole entry).
 */
const distinctUserAgents = (sources: readonly SweepSource[]): string[] | undefined => {
  const agents: string[] = [];
  for (const source of sources) {
    if (source.userAgent === undefined) return undefined;
    if (!agents.includes(source.userAgent)) agents.push(source.userAgent);
  }
  return agents.length === 0 ? undefined : agents;
};

/**
 * The kernel fingerprint of one family of the generated table. When the table's geometry cannot
 * build the kernels of the family's sweep shapes (a codegen check, the dispatch-count limit — the
 * fingerprint builds with the 65535 every device guarantees, so a geometry that passed the sweep on
 * a device with a larger limit can still fail), throws naming the reason (case, field, geometry)
 * instead of building the table — no table that would always mismatch at matching gets stored.
 */
const generatedKernelsId = (table: GeometryProfile, family: GeometryProfileFamily): string => {
  try {
    return familyKernelsId(table, family);
  } catch (cause) {
    if (cause instanceof KernelsIdError) {
      throw new Error(
        `Cannot derive provenance.families.${family}.kernels (the kernel fingerprint) of generated table '${table.id}' — ${cause.message}`,
        { cause },
      );
    }
    throw cause;
  }
};

/** The online attention's kernel fingerprint of a generated table that adopted the field (as {@link generatedKernelsId}). */
const generatedAttentionKernelsId = (table: GeometryProfile): string => {
  try {
    return attentionBenchKernelsId();
  } catch (cause) {
    if (cause instanceof KernelsIdError) {
      throw new Error(
        `Cannot derive provenance.attentionBench.kernels (the online attention's kernel fingerprint) of generated table '${table.id}' — ${cause.message}`,
        { cause },
      );
    }
    throw cause;
  }
};

/**
 * One input of the public generator ({@link deriveGeometryProfile}) — a sweep record and the
 * identification that goes into the table's `provenance`. `report` is an `unknown` boundary (a
 * value from `JSON.parse` of saved JSON or the return value of `runGeometrySweep` — the fields
 * generation reads are checked by {@link parseSweepReport}).
 */
export type GeometrySweepInput = {
  readonly report: unknown;
  /**
   * The record's name (a file path or the like — goes into `provenance.families[family].sweep` of
   * every family it has a row of).
   */
  readonly path: string;
  /**
   * The SHA-256 of the record's bytes (hex — goes into `provenance.families[family].sha256`.
   * Passing two records with the same value fails).
   */
  readonly sha256: string;
};

/**
 * Options of {@link deriveGeometryProfile} (the table's id, the adapter the table applies to, the
 * adoption threshold, the attention-bench records and the table to carry families over from).
 */
export type DeriveGeometryProfileOptions = ProfileTarget & {
  readonly id: string;
  /**
   * The adoption threshold for the ratio to the default (default {@link DEFAULT_MIN_SPEEDUP}, at
   * least 1 — the CLI's `--min-speedup`, removed in 0.15). The `attentionOnline` field uses the
   * same threshold for its per-node gain.
   */
  readonly minSpeedup?: number;
  /**
   * The second input (ADR 0117 addendum 2026-11-12): attention-bench records
   * (`runAttentionRouteBench` with `onlineTiled` and `quantization`, GPU timestamps) of the
   * same adapter as the sweeps, which together cover `ATTENTION_BENCH_SHAPES`. They decide
   * `attentionOnline` ({@link deriveAttentionOnline}); omitted, the table has no such field (unless
   * `base` carries it over).
   */
  readonly attentionBench?: readonly AttentionBenchInput[];
  /**
   * An existing table of the same adapter (ADR 0117 addendum 2026-11-12) whose op families the
   * records do not cover are carried over: its fields of such a family and its provenance entry are
   * copied verbatim once the entry's `kernels` and `caseSet` are checked against the current runtime
   * (a stale entry throws — sweep that family again), and, without `attentionBench`, its
   * `attentionOnline` and attention-bench block likewise (stale: measure the attention bench again).
   * Its `linearGemv` is carried as it is: GEMV sweep records decide it (`deriveLinearGemvChoices`).
   * The records must cover every family they have a row of completely (every case of its current
   * case set). `base` needs `provenance` with the records' adapter. `id` and `match` still come from
   * these options. With `shared`, `base` is a shared table of the records' adapters, and a family
   * some adapter covers only in part (a case of its case set with no row of the default geometry
   * there, or only failed ones) keeps the default in every field, its verdict lines naming the
   * adapter and the cases.
   */
  readonly base?: GeometryProfile;
  /**
   * Derive a **shared table** from the records of two or more adapters of one vendor and
   * architecture ({@link deriveSharedProfile} — SPEC: docs/spec/geometry-profiles.md#shared-tables):
   * it needs `vendor` and `architecture`, takes no `description`, no `optIn` and no
   * `attentionBench`, tells the devices apart by their descriptions (every record carries one), and
   * writes `provenance.adapters`. Omitted or false, the records must agree on
   * all four adapter fields (a measured table).
   */
  readonly shared?: boolean;
};

/** The return value of {@link deriveGeometryProfile} (the table and the verdict lines). */
export type GeometryProfileDerivation = {
  /**
   * The generated table (with `provenance` — saved with {@link geometryProfileJson} and injected
   * into `acquireGpu`).
   */
  readonly profile: GeometryProfile;
  /**
   * Per-field verdicts, rejected geometries and why, and the cases dropped from the ratio material
   * ({@link verdictLines} — the same lines as the header comment of the repository's generated
   * files and the CLI's standard output (the CLI was removed in 0.15)); a field carried over from
   * `base` gets one line naming the base table and its sweeps instead.
   */
  readonly verdicts: readonly string[];
};

/**
 * The verdict lines of a derivation: {@link verdictLines} per field, except one line per field of a
 * family carried over from the base, naming the base table and the sweeps of its entry.
 */
const derivationVerdictLines = <
  Verdict extends { readonly slot: ProfileSlot; readonly scope: string },
>(
  verdicts: readonly Verdict[],
  carry: CarryOver | undefined,
  linesOf: (verdicts: readonly Verdict[]) => string[],
): string[] =>
  verdicts.flatMap((verdict) => {
    const family = profileSlotFamily(verdict.slot);
    const carried = carry?.families.find((entry) => entry.family === family);
    return carry === undefined || carried === undefined
      ? linesOf([verdict])
      : [
          `- ${verdict.slot} (${verdict.scope}): carried over from table '${carry.base.id}' (family ${family}, sweeps ${carried.entry.sweep})`,
        ];
  });

/** The verdict line of `attentionOnline` carried over from the base with its attention-bench block. */
const carriedAttentionLine = (base: GeometryProfile, block: AttentionBenchProvenance): string =>
  `- attentionOnline (the online attention's Vᵀ key axis): carried over from table '${base.id}' (${
    base.attentionOnline === undefined ? "the form N gives" : "tiled"
  }; attention-bench records ${block.records})`;

/**
 * Builds a table from one or more sweep records (the generator of the public surface
 * `@karume/runtime/tune` — ADR 0117 decision 2). The rules are {@link deriveProfile} (ADR 0115
 * decision 4, ADR 0116, ADR 0117 decision 3) and the table's values come from the single
 * {@link buildGeometryProfile}, so they equal the tables the CLI and the GPU lab (both removed in
 * 0.15) built. With `base`, the families the records do not cover are carried over from it
 * ({@link carryOver} — ADR 0117 addendum 2026-11-12). With `shared`, the table is a shared table of
 * several adapters ({@link deriveSharedProfile}); it carries tile geometries only. Input that fails
 * a check throws with the reason (fail loudly).
 */
export const deriveGeometryProfile = (
  reports: readonly GeometrySweepInput[],
  options: DeriveGeometryProfileOptions,
): GeometryProfileDerivation => {
  const minSpeedup = options.minSpeedup ?? DEFAULT_MIN_SPEEDUP;
  // Allowing less than 1 could adopt a geometry slower than the default (the ratio check would lose
  // its meaning — the same condition as the CLI's --min-speedup, removed in 0.15)
  if (!Number.isFinite(minSpeedup) || minSpeedup < 1) {
    throw new Error(`minSpeedup must be a number of at least 1 (${minSpeedup})`);
  }
  const sources = reports.map(({ report, path, sha256 }) =>
    parseSweepReport(report, { path, sha256 }),
  );
  const { attentionBench, base, shared, ...target } = options;
  const spec = { ...target, minSpeedup };
  if (shared === true) {
    // MUST: a shared table carries tile geometries only — the online attention's key axis is read
    // from attention-bench records of one adapter, and no rule combines them across devices
    if (attentionBench !== undefined) {
      throw new Error(
        "A shared table carries tile geometries only: attention-bench records (the online attention's key axis) are not derived for it",
      );
    }
    const sharedVerdicts = deriveSharedProfile(sources, { ...spec, onBase: base !== undefined });
    const sharedCarry = base === undefined ? undefined : carryOver(base, sources, false, true);
    return {
      profile: buildGeometryProfile(
        { ...spec, shared: true },
        sources,
        sharedVerdicts,
        undefined,
        sharedCarry,
      ),
      verdicts: derivationVerdictLines(sharedVerdicts, sharedCarry, sharedVerdictLines),
    };
  }
  const verdicts = deriveProfile(sources, spec);
  const carry =
    base === undefined ? undefined : carryOver(base, sources, attentionBench !== undefined, false);
  const attention =
    attentionBench === undefined
      ? undefined
      : deriveAttentionBench(attentionBench, sources[0].adapter, minSpeedup);
  const carriedBench = carry?.attentionBench;
  return {
    profile: buildGeometryProfile(spec, sources, verdicts, attention, carry),
    verdicts: [
      ...derivationVerdictLines(verdicts, carry, verdictLines),
      ...(attention !== undefined
        ? attentionOnlineVerdictLines(attention.verdict)
        : carry === undefined || carriedBench === undefined
          ? []
          : [carriedAttentionLine(carry.base, carriedBench)]),
    ],
  };
};

/**
 * The marker {@link geometryProfileJson} puts temporarily where Infinity goes (NUL never appears in
 * the table's strings).
 */
const INFINITY_MARK = "\u0000karume-infinity\u0000";

/**
 * The JSON of a table for injection (the result of `JSON.parse` can be passed to
 * `acquireGpu({ geometryProfile })` as is).
 *
 * The trailing `maxRows` of `gemmRows` and `i8a8.linear` (Infinity) has no JSON value, so it is
 * written as `1e999` — `JSON.parse` reads an out-of-range number as Infinity. Plain
 * `JSON.stringify` turns Infinity into null, and a table with null fails the runtime's check (the
 * last rule is Infinity).
 */
export const geometryProfileJson = (profile: GeometryProfile): string => infinityJson(profile);

/**
 * The same shape as `JSON.stringify(value, null, 2)`, writing only `Infinity` as `1e999` (the same
 * spelling as {@link geometryProfileJson} — records that carry an injected table (the GPU lab's
 * Anima JSON; GPU lab removed in 0.15) do not drop the table's values to null either). The spelling
 * is fingerprint.ts's `INFINITY_JSON` (one and the same with the canonical JSON of the matching
 * keys).
 */
export const infinityJson = (value: unknown): string => {
  const plain = JSON.stringify(value);
  if (plain.includes(JSON.stringify(INFINITY_MARK).slice(1, -1))) {
    throw new Error("A JSON string contains the same spelling as the Infinity marker");
  }
  return JSON.stringify(
    value,
    (_key, entry: unknown) => (entry === Number.POSITIVE_INFINITY ? INFINITY_MARK : entry),
    2,
  ).replaceAll(JSON.stringify(INFINITY_MARK), INFINITY_JSON);
};
