/**
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) from GEMV sweep records
 * (`karume-gemv-sweep/4` — gemv-report.ts): {@link deriveLinearGemvChoices} of
 * `@karume/runtime/tune`; and says what the records lack for that derivation to be complete under
 * the current runtime ({@link planGemvMeasurements}). Pure functions of the records and the current
 * runtime (its case table, its kernels and the default profile's choices); they measure nothing.
 *
 * ## The unit of evidence: a group
 *
 * A group is one record's measurement of one case at one row count (its summary and its rows). Any
 * number of records of one adapter are read group by group, so a record that holds half of the
 * cases, or one whose kernels changed in part, leaves its other groups standing:
 *
 * - A row is current when the current runtime has its candidate at that case and row count
 *   (`gemvKernelsAt`) and builds the same kernel for it (the row's `kernel` fingerprint).
 * - A group is current when its case is the current runtime's (the record's case fingerprint, and
 *   the row count in the case table). Which choices the record's summary names as the default's is
 *   not read for it: a record made under older default choices stands wherever it timed the kernels
 *   the current default runs.
 * - A candidate is read against a reference: the kernel the current default profile runs for its
 *   class at the case and row count (identical class: the choice at `exact`; reorder class: the
 *   choice at `reorder`). A group is usable for a class when the reference's row in it is present,
 *   timed, without error and current, and — for the identical class, where the reference is not the
 *   record's own baseline — its words were shown equal to that baseline's (then candidate, record
 *   baseline and reference write the same words). The GEMM that is the baseline of a float-stored
 *   case past one row takes its tile geometry from a geometry profile, so its current kernel is the
 *   one the table the records derive for builds (what a Session on the adapter runs once the table
 *   is its table): a GEMM measured with another geometry — under another profile — is no reference.
 *   A record's baseline that is not the current reference is a candidate like any other, whose
 *   words are its own.
 * - The reference's time at a case and row count is its fastest `perDispatch` over every current
 *   group of the records in which its row is current, timed and without error, and it is a time
 *   only when a second record repeats it within {@link GEMV_REFERENCE_REPEAT}. A candidate's gain in
 *   a group is that time ÷ its own time there. On a device whose reference kernel runs in two
 *   timing modes, a bound on the spread of one record's rounds discards groups whose fastest round
 *   is the device's fastest (182 of 265 such groups on the RTX 3080 Ti) and admits groups that
 *   stayed slow throughout, which overstate a gain (bf16 262144 × 1536 at one row: ×2.78 against a
 *   record's baseline that stayed slow, ×1.67 against the device's fastest); against the repeated
 *   fastest time a slow record can only lower a gain.
 * - A group is evidence for a candidate when it is current, usable for the candidate's class, the
 *   reference's time is repeated, and the candidate's row is current and timed (not screened out by
 *   the quick look, not skipped — because the same record's M = 1 group already left it
 *   unadoptable, or, in an `explore` record, because that set's M = 1 group did not carry it) with
 *   its check held. A row that failed or whose check failed in a current group, evidence or
 *   not, rules the candidate out: correctness does not depend on timing.
 * - A record of the `explore` set is read by these rules alone: it holds rows of the kernels it
 *   measured (the sentinels' whole grid, the predicted candidates elsewhere), and a candidate
 *   without a row has no evidence in it, as in any record that holds part of the candidates.
 *
 * Groups whose case is not current are reported (the verdict lines, the provenance's counts and
 * {@link LinearGemvDerivation.stale}), and so are the current groups without a usable reference;
 * no record is refused for them. A record is refused only for what no group of it could stand for:
 * another format (no older one is read), wall-clock or quantized timing, another adapter, the same
 * record twice.
 *
 * ## The two rules
 *
 * A record's class comes from its summaries (gemv-rule.ts's `gemvSteadiness`): steady with at least
 * 20 groups that have a baseline spread, at most 20 % of them beyond ×1.05. A decision (a sequential
 * key, a parallel row) one of whose cases a steady record measured at M = 1 with a usable reference
 * is decided by the rule of one steady record over the steady records (gemv-one-record.ts — the
 * maintainer, 2026-11-28); any other decision by the rule of two records over every record (below —
 * the maintainer, 2026-11-17). A failure or a failed check in a current group of any record rules a
 * candidate out under either.
 *
 * The rule of one steady record reads, in each steady record's group, a candidate's time against the
 * reference's row of the same group — no second record need repeat it —, both rows' gaps (second-
 * smallest round ÷ smallest) and the reference's re-timing; a row whose pass is shorter than 20 ms
 * is no evidence. Those readings are the decision's `evidence`, margin-free, and the margin is applied
 * by `decideOneRecord` alone, so {@link redecideLinearGemvChoices} decides an output again under
 * another margin by the same code. Where the table the records derive for holds another choice than
 * the default's, that choice is the one to beat; the field is the table's choices with every
 * decision's derived choice in place, so a shape no record measured keeps the table's choice — and so
 * does a decision of the rule of two records that fewer than two records are evidence for, unless a
 * record rules that choice out (then the default's is written back, as the rule of one steady record
 * does).
 *
 * ## The rule of two records (the maintainer, 2026-11-17)
 *
 * The timing of these small dispatches moves between runs, so one record of a device that is not
 * steady is not evidence:
 *
 * - A candidate replaces the default profile's choice for a shape only when, at M = 1, groups of at
 *   least {@link MIN_RECORDS} different records are evidence for it and it is at least
 *   {@link GEMV_MIN_SPEEDUP} faster than that choice in every one, its check holding in every current
 *   group (identical class: the baseline's words; reorder class: inside the band). Among several,
 *   the one whose smallest gain is largest. A candidate's time is its row's `perDispatch` (the
 *   minimum of its rounds ÷ `reps`); the choice's is the reference's repeated fastest time.
 * - The two classes are derived separately: the identical class writes the sequential variant's
 *   per-shape exceptions (`sequential.shapes`, from M = 1) and changes no number of any Session; the
 *   reorder class writes the parallel rows (`parallelShapes`: lanes and the staged flag, from the
 *   plain and staged candidates at M = 1) and changes the numbers of Sessions at `reorder`, inside
 *   the band. A parallel row serves M = 1 to 8, so at each other row count of that range (the
 *   `no-regression` role of gemv-candidates.ts's `gemvGroupRole` — M = 4 and 8 in the case table)
 *   groups of at least {@link MIN_RECORDS} records must be evidence for the kernel the row runs
 *   there, and it may be slower than the default's choice in none of them: a path that was not
 *   measured is not moved by a table.
 * - A parallel row's row block is chosen by one rule, against a reference row of the same lanes:
 *   the default's row where M = 1 keeps it, the candidate without a row block where M = 1 adopts
 *   one. The row blocks of one lane count write the same words, so the choice is speed alone: a row
 *   block (or none) replaces the reference's only when, at the other row counts where the two run
 *   different kernels, it is slower in no group that is evidence (groups of {@link MIN_RECORDS}
 *   records at each) and at least {@link GEMV_MIN_SPEEDUP} faster in every group of one row count
 *   or more; among several, the one whose smallest gain over those row counts is largest. A row
 *   count at which both run the same kernel (M = 4 under a row block of 4 and of 8) neither refuses
 *   nor counts ({@link rowBlockRefusal}).
 *
 * What the field does not let the records decide: a sequential key has no group length, so cases
 * that share it and disagree keep the default; a row whose default takes packed activations stays
 * (the packed kernels are not measured); the subgroup candidates are a Session option, not a row,
 * and only show in the decision's notes; the row-block thread target is one number for every shape,
 * and no rule turns the per-shape row-block heights into it, so it stays; the general sequential
 * variant also sets the row blocks' columns and read-ahead, so it stays.
 *
 * SPEC: docs/spec/geometry-profiles.md#deriving-the-gemv-choices
 */

import { CodegenError } from "../codegen/errors.ts";
import { canonicalJsonText } from "../format/container/json.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
  type GeometryProfile,
  selectGeometryProfile,
} from "../kernels/geometry-profile.ts";
import {
  LINEAR_GEMV_PARALLEL_ROWS,
  type LinearGemvChoices,
  type LinearGemvParallelLanes,
  type LinearGemvParallelRows,
  linearGemvParallelRowsFits,
  linearGemvParallelRowsFor,
  type LinearGemvParallelShape,
  type LinearGemvShapeVariant,
  type LinearGemvVariant,
} from "../kernels/linear-gemv.ts";
import { formatRatio } from "./derive.ts";
import {
  GEMV_CANDIDATE_SETS,
  GEMV_MIN_SPEEDUP,
  GEMV_REFERENCE_REPEAT,
  type GemvCandidateClass,
  type GemvCandidateSet,
  gemvBaseline,
  gemvCandidates,
  type GemvGates,
  gemvGroupRole,
  gemvKernelsAt,
  gemvParallelName,
  gemvProfileChoice,
  gemvProgram,
  gemvRowCountsIn,
  isGemvCandidateSet,
} from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, gemvCaseOf, type GemvSweepStorage } from "./gemv-cases.ts";
import { gemvCaseFingerprint, gemvCaseTableId, gemvKernelId } from "./gemv-fingerprint.ts";
import {
  decideOneRecord,
  gemvTableNotHeld,
  type GemvCandidateEvidence,
  type GemvKeptRow,
  type GemvOneRecordEvidence,
  type GemvOneRecordOutcome,
  type GemvPairReading,
  type GemvRowBlockOption,
  type GemvRowCountEvidence,
} from "./gemv-one-record.ts";
import {
  GEMV_SWEEP_FORMAT,
  type GemvSweepChoice,
  type GemvRetimingUnread,
  type GemvSweepFamilyChoice,
  type GemvSweepRow,
} from "./gemv-report.ts";
import {
  GEMV_MIN_PASS_MS,
  GEMV_STEADY_GROUP_SPREAD,
  GEMV_STEADY_MAX_SHARE,
  GEMV_STEADY_MIN_GROUPS,
  gemvBreakEven,
  gemvMarginOf,
  gemvRecordReading,
  gemvRoundsGap,
  type GemvSteadiness,
  gemvSteadiness,
  gemvTimingRefusal,
  isGemvTimestampUnit,
} from "./gemv-rule.ts";
import { ROUNDS } from "./measurement.ts";

/** The fewest different records whose groups must be evidence: one record is not evidence. */
const MIN_RECORDS = 2;

/**
 * At a row count of the `no-regression` role, a parallel candidate may not be slower than the
 * default's choice (a gain below this is a regression).
 */
const NO_REGRESSION = 1;

/**
 * One GEMV sweep record given to {@link deriveLinearGemvChoices}: the record (`runGemvSweep`'s
 * return value or `JSON.parse` of its saved text — an `unknown` boundary) and its identification.
 */
export type GemvSweepInput = {
  readonly report: unknown;
  /** The record's name (a file path or the like). */
  readonly path: string;
  /** The SHA-256 of the record's bytes (hex — passing two records with the same value fails). */
  readonly sha256: string;
};

/** The four adapter fields of a record. */
type Adapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

/** The fields of one row the derivation reads. */
type RowObservation = {
  readonly caseId: string;
  readonly m: number;
  readonly candidate: string;
  readonly class: GemvCandidateClass;
  readonly isBaseline: boolean;
  readonly choice: GemvSweepChoice;
  readonly kernel?: string;
  readonly perDispatch?: number;
  /** Dispatches per pass (the rule of one steady record reads the pass's length). */
  readonly reps?: number;
  /** The pass's time per round (clamped rounds are 0): the rule of one steady record reads the gap. */
  readonly rounds?: readonly number[];
  /** A reference's pass once more after the rounds ÷ `reps` (a record of 2026-11-28 on). */
  readonly retimedPerDispatch?: number;
  /** A reference whose pass once more read no time (clamped, 0, or the device lost): no evidence is read against it. */
  readonly retimingUnread?: GemvRetimingUnread;
  readonly identicalToBaseline?: boolean;
  readonly withinBand?: boolean;
  readonly screened?: true;
  readonly skipped?: true;
  readonly error?: string;
};

/** One record's measurement of a case at a row count: its summary's fields and its rows by candidate. */
type GroupObservation = {
  readonly caseId: string;
  readonly m: number;
  readonly profile: { readonly exact: string; readonly reorder: string };
  readonly rows: ReadonlyMap<string, RowObservation>;
};

/** One record as the derivation reads it (checked at the `unknown` boundary). */
type GemvSweepSource = {
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly candidateSet: GemvCandidateSet;
  readonly userAgent?: string;
  /** The timestamps' unit: the rule of one steady record reads a pass's length in ns only. */
  readonly unit: "ns" | "deno-raw-tick";
  /** The record's class by its summaries (whatever the record states of itself). */
  readonly steadiness: GemvSteadiness;
  readonly adapter: Adapter;
  /** The id of the profile the sweep's device resolved to (absent in a record written before the field). */
  readonly geometryProfile?: string;
  readonly gates: GemvGates;
  /** The planned cases' fingerprints, by id. */
  readonly cases: ReadonlyMap<string, string>;
  /** By {@link groupKey}, in the record's order. */
  readonly groups: ReadonlyMap<string, GroupObservation>;
};

const groupKey = (caseId: string, m: number): string => `${caseId} M=${m}`;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const objectAt = (value: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`${where} is not an object`);
  return value;
};

const stringAt = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} is not a string`);
  return value;
};

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value > 0;

const isPositiveNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value) && value > 0;

/** An optional field: absent, or of the type `guard` accepts. */
const optionalAt = <T>(
  record: Record<string, unknown>,
  key: string,
  guard: (value: unknown) => value is T,
  where: string,
): T | undefined => {
  const value = record[key];
  if (value === undefined) return undefined;
  if (!guard(value)) throw new Error(`${where}: ${key} is invalid (${JSON.stringify(value)})`);
  return value;
};

const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";
const isString = (value: unknown): value is string => typeof value === "string";
const isTrue = (value: unknown): value is true => value === true;
const isRetimingUnread = (value: unknown): value is GemvRetimingUnread =>
  value === "clamped" || value === "zero" || value === "lost";
/** A row's rounds: finite times of at least 0 (a clamped round is 0). */
const isRounds = (value: unknown): value is number[] =>
  Array.isArray(value) &&
  value.every((round) => typeof round === "number" && Number.isFinite(round) && round >= 0);

const CHOICE_IDS = [
  "gemv-m1",
  "gemv-rows",
  "gemv-parallel",
  "gemv-parallel-packed",
  "gemv-parallel-staged",
  "gemv-parallel-subgroup32",
] as const satisfies readonly GemvSweepFamilyChoice["id"][];

const isChoiceId = (value: unknown): value is GemvSweepFamilyChoice["id"] =>
  CHOICE_IDS.some((id) => id === value);

const STORAGES = [
  "i2",
  "i4",
  "i8",
  "bf16",
  "f16",
  "f32",
] as const satisfies readonly GemvSweepStorage[];

const isStorage = (value: unknown): value is GemvSweepStorage =>
  STORAGES.some((storage) => storage === value);

const LANES = [2, 4, 8, 16, 32] as const satisfies readonly LinearGemvParallelLanes[];

const isLanes = (value: unknown): value is LinearGemvParallelLanes =>
  LANES.some((lanes) => lanes === value);

const isParallelRows = (value: unknown): value is LinearGemvParallelRows =>
  LINEAR_GEMV_PARALLEL_ROWS.some((rows) => rows === value);

const isClass = (value: unknown): value is GemvCandidateClass =>
  value === "identical" || value === "reorder";

/** The column form of a choice (`variant`, and inside `rowsVariant` with `rows`). */
const variantAt = (value: unknown, where: string): LinearGemvVariant => {
  const variant = objectAt(value, where);
  const { cols, unroll } = variant;
  if (!isPositiveInteger(cols) || !isPositiveInteger(unroll)) {
    throw new Error(`${where}: cols and unroll must be positive integers`);
  }
  return { cols, unroll };
};

const parseChoice = (value: unknown, where: string): GemvSweepChoice => {
  const choice = objectAt(value, `${where} choice`);
  const { id, storage } = choice;
  if (!isStorage(storage)) {
    throw new Error(
      `${where}: choice.storage ${JSON.stringify(storage)} is none of ${STORAGES.join(", ")}`,
    );
  }
  // The GEMM as the baseline of a float-stored weight at M ≥ 2
  if (id === "gemm-f32") {
    if (!isBoolean(choice.v4)) throw new Error(`${where}: choice.v4 is not a boolean`);
    return { id, storage, v4: choice.v4 };
  }
  if (!isChoiceId(id)) {
    throw new Error(
      `${where}: choice.id ${JSON.stringify(id)} is neither a GEMV form nor gemm-f32`,
    );
  }
  const groupSize = optionalAt(choice, "groupSize", isPositiveInteger, `${where} choice`);
  const lanes = optionalAt(choice, "lanes", isLanes, `${where} choice`);
  const variant = variantAt(choice.variant, `${where} choice.variant`);
  const rowsAt = choice.rowsVariant;
  let rowsVariant: GemvSweepFamilyChoice["rowsVariant"];
  if (rowsAt !== undefined) {
    const rows = objectAt(rowsAt, `${where} choice.rowsVariant`).rows;
    if (!isPositiveInteger(rows)) {
      throw new Error(`${where}: choice.rowsVariant.rows must be a positive integer`);
    }
    rowsVariant = { ...variantAt(rowsAt, `${where} choice.rowsVariant`), rows };
  }
  const parallelRows = optionalAt(choice, "parallelRows", isParallelRows, `${where} choice`);
  return {
    id,
    storage,
    ...(groupSize === undefined ? {} : { groupSize }),
    ...(lanes === undefined ? {} : { lanes }),
    variant,
    ...(rowsVariant === undefined ? {} : { rowsVariant }),
    ...(parallelRows === undefined ? {} : { parallelRows }),
  };
};

const parseRow = (value: unknown, where: string): RowObservation => {
  const row = objectAt(value, where);
  const { m, isBaseline } = row;
  if (!isPositiveInteger(m)) throw new Error(`${where}: m is not a positive integer`);
  if (!isClass(row.class)) throw new Error(`${where}: class is neither identical nor reorder`);
  if (!isBoolean(isBaseline)) throw new Error(`${where}: isBaseline is not a boolean`);
  const kernel = optionalAt(row, "kernel", isString, where);
  const perDispatch = optionalAt(row, "perDispatch", isPositiveNumber, where);
  const reps = optionalAt(row, "reps", isPositiveInteger, where);
  const rounds = optionalAt(row, "rounds", isRounds, where);
  const retimedPerDispatch = optionalAt(row, "retimedPerDispatch", isPositiveNumber, where);
  const retimingUnread = optionalAt(row, "retimingUnread", isRetimingUnread, where);
  if (retimingUnread !== undefined && retimedPerDispatch !== undefined) {
    throw new Error(`${where}: a row has retimingUnread or retimedPerDispatch, not both`);
  }
  const identicalToBaseline = optionalAt(row, "identicalToBaseline", isBoolean, where);
  const withinBand = optionalAt(row, "withinBand", isBoolean, where);
  const screened = optionalAt(row, "screened", isTrue, where);
  const skipped = optionalAt(row, "skipped", isTrue, where);
  const error = optionalAt(row, "error", isString, where);
  if (screened === true && (isBaseline || perDispatch !== undefined)) {
    throw new Error(`${where}: a screened row is no baseline and has no time`);
  }
  if (skipped === true && (isBaseline || perDispatch !== undefined || screened === true)) {
    throw new Error(`${where}: a skipped row is no baseline, has no time and is not screened`);
  }
  return {
    caseId: stringAt(row, "caseId", where),
    m,
    candidate: stringAt(row, "candidate", where),
    class: row.class,
    isBaseline,
    choice: parseChoice(row.choice, where),
    ...(kernel === undefined ? {} : { kernel }),
    ...(perDispatch === undefined ? {} : { perDispatch }),
    ...(reps === undefined ? {} : { reps }),
    ...(rounds === undefined ? {} : { rounds }),
    ...(retimedPerDispatch === undefined ? {} : { retimedPerDispatch }),
    ...(retimingUnread === undefined ? {} : { retimingUnread }),
    ...(identicalToBaseline === undefined ? {} : { identicalToBaseline }),
    ...(withinBand === undefined ? {} : { withinBand }),
    ...(screened === undefined ? {} : { screened }),
    ...(skipped === undefined ? {} : { skipped }),
    ...(error === undefined ? {} : { error }),
  };
};

/**
 * Reads a GEMV sweep record (the `unknown` boundary — checks what the derivation reads and fails
 * loudly on what no group could stand for): the format, GPU timestamps that are not quantized, the
 * adapter, the gates, the planned cases with their fingerprints and the groups. Whether a group is of
 * the current runtime is not decided here ({@link groupStates}).
 */
const parseGemvSweepRecord = (
  report: unknown,
  meta: { readonly path: string; readonly sha256: string },
): GemvSweepSource => {
  const where = meta.path;
  const record = objectAt(report, `${where}: the JSON`);
  if (record.format !== GEMV_SWEEP_FORMAT) {
    throw new Error(
      `${where}: format ${JSON.stringify(record.format)} is not ${GEMV_SWEEP_FORMAT} — a record of an older format is not read; measure again with the current runtime`,
    );
  }
  const timing = objectAt(record.gpuTiming, `${where} gpuTiming`);
  // The wall clock adds the submit-to-completion floor, which moves per pass; quantized timestamps
  // move a small dispatch's minimum by a whole step
  const { unit } = timing;
  if (!isGemvTimestampUnit(unit)) {
    throw new Error(
      `${where}: gpuTiming.unit is ${JSON.stringify(timing.unit)} (must be ns or deno-raw-tick — a wall-clock record is not read)`,
    );
  }
  if (gemvTimingRefusal(timing) === "quantized") {
    throw new Error(
      `${where}: gpuTiming.quantized is ${JSON.stringify(timing.quantized)} (quantized timestamps are not read)`,
    );
  }
  const candidateSet = objectAt(record.settings, `${where} settings`).candidateSet;
  if (typeof candidateSet !== "string" || !isGemvCandidateSet(candidateSet)) {
    throw new Error(
      `${where}: settings.candidateSet ${JSON.stringify(candidateSet)} is none of ${GEMV_CANDIDATE_SETS.join(", ")}`,
    );
  }
  const adapterAt = objectAt(record.adapter, `${where} adapter`);
  const adapter: Adapter = {
    vendor: stringAt(adapterAt, "vendor", `${where} adapter`),
    architecture: stringAt(adapterAt, "architecture", `${where} adapter`),
    device: stringAt(adapterAt, "device", `${where} adapter`),
    description: stringAt(adapterAt, "description", `${where} adapter`),
  };
  const gatesAt = objectAt(record.gates, `${where} gates`);
  const { subgroups, maxComputeWorkgroupsPerDimension: limit } = gatesAt;
  if (!isBoolean(subgroups) || !isPositiveInteger(limit)) {
    throw new Error(
      `${where}: gates.subgroups is not a boolean or gates.maxComputeWorkgroupsPerDimension not a positive integer`,
    );
  }
  if (!Array.isArray(record.cases)) throw new Error(`${where}: cases is not an array`);
  const cases = new Map<string, string>();
  record.cases.forEach((value, index) => {
    const at = `${where} cases[${index}]`;
    const entry = objectAt(value, at);
    const id = stringAt(entry, "id", at);
    if (cases.has(id)) throw new Error(`${at}: case ${id} twice`);
    cases.set(id, stringAt(entry, "fingerprint", at));
  });
  if (!Array.isArray(record.summaries)) throw new Error(`${where}: summaries is not an array`);
  const groups = new Map<string, GroupObservation & { rows: Map<string, RowObservation> }>();
  const spreads: { baselineSpread?: number }[] = [];
  record.summaries.forEach((value, index) => {
    const at = `${where} summaries[${index}]`;
    const summary = objectAt(value, at);
    const caseId = stringAt(summary, "caseId", at);
    const { m } = summary;
    if (!isPositiveInteger(m)) throw new Error(`${at}: m is not a positive integer`);
    const baselineSpread = optionalAt(summary, "baselineSpread", isPositiveNumber, at);
    spreads.push(baselineSpread === undefined ? {} : { baselineSpread });
    if (!cases.has(caseId)) throw new Error(`${at}: case ${caseId} is not among the planned cases`);
    const profileAt = objectAt(summary.profile, `${at} profile`);
    const key = groupKey(caseId, m);
    if (groups.has(key)) throw new Error(`${at}: ${key} is summarised twice`);
    groups.set(key, {
      caseId,
      m,
      profile: {
        exact: stringAt(profileAt, "exact", `${at} profile`),
        reorder: stringAt(profileAt, "reorder", `${at} profile`),
      },
      rows: new Map(),
    });
  });
  if (!Array.isArray(record.rows)) throw new Error(`${where}: rows is not an array`);
  record.rows.forEach((value, index) => {
    const at = `${where} rows[${index}]`;
    const row = parseRow(value, at);
    const key = groupKey(row.caseId, row.m);
    const group = groups.get(key);
    if (group === undefined) throw new Error(`${at}: ${key} has no summary`);
    if (group.rows.has(row.candidate)) throw new Error(`${at}: ${row.candidate} of ${key} twice`);
    if (row.isBaseline !== (row.candidate === group.profile.exact)) {
      throw new Error(
        `${at}: ${row.candidate} of ${key} is ${row.isBaseline ? "" : "not "}the baseline, while the summary names ${group.profile.exact}`,
      );
    }
    group.rows.set(row.candidate, row);
  });
  const userAgent = typeof record.userAgent === "string" ? record.userAgent : undefined;
  const geometryProfile = optionalAt(record, "geometryProfile", isString, where);
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: stringAt(record, "date", where),
    candidateSet,
    ...(userAgent === undefined ? {} : { userAgent }),
    unit,
    steadiness: gemvSteadiness(spreads),
    adapter,
    ...(geometryProfile === undefined ? {} : { geometryProfile }),
    gates: { subgroups, maxComputeWorkgroupsPerDimension: limit },
    cases,
    groups,
  };
};

/** The current runtime's kernels of a case at a row count, by candidate name (fingerprints). */
type CurrentKernels = (gemvCase: GemvCase, m: number) => ReadonlyMap<string, string>;

/**
 * {@link CurrentKernels} with each (case, row count) built once. `table` is the profile the adapter
 * runs with: the GEMM reference of a float-stored case takes its tile geometry (the GEMV kernels
 * read nothing of it).
 */
const currentKernels = (table: GeometryProfile): CurrentKernels => {
  const cache = new Map<string, ReadonlyMap<string, string>>();
  return (gemvCase, m) => {
    const key = groupKey(gemvCase.id, m);
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    const kernels = new Map(
      gemvKernelsAt(gemvCase, m).map(({ name, choice }) => [
        name,
        gemvKernelId(gemvCase, m, choice, (one, rows, chosen) =>
          gemvProgram(one, rows, chosen, table),
        ),
      ]),
    );
    cache.set(key, kernels);
    return kernels;
  };
};

/** Why a row of a current group is not of the current runtime (undefined: it is). */
const rowStaleness = (
  row: RowObservation,
  kernels: ReadonlyMap<string, string>,
): string | undefined => {
  const current = kernels.get(row.candidate);
  if (current === undefined) return "no longer a candidate";
  if (row.kernel === undefined) return "no kernel fingerprint (its program was not built)";
  return row.kernel === current ? undefined : "its kernel changed";
};

/** The level whose default choice is a class's reference. */
type ReferenceLevel = "exact" | "reorder";

const REFERENCE_LEVELS = ["exact", "reorder"] as const satisfies readonly ReferenceLevel[];

/** The identical class reads against the choice at `exact` (the baseline), the reorder class against the choice at `reorder`. */
const levelOf = (candidateClass: GemvCandidateClass): ReferenceLevel =>
  candidateClass === "identical" ? "exact" : "reorder";

/** The kernel the current default profile runs at `level` for a case and row count (by candidate name). */
const referenceName = (gemvCase: GemvCase, m: number, level: ReferenceLevel): string =>
  gemvProfileChoice(gemvCase, m, level).name;

/**
 * The kernel a class's candidates are read against at a case and row count, by candidate name
 * (the module doc's "reference"): the current default profile's choice at `exact` (identical class)
 * or at `reorder` (reorder class), whatever a record's summary names.
 */
const classReference = (
  gemvCase: GemvCase,
  m: number,
  candidateClass: GemvCandidateClass,
): string => referenceName(gemvCase, m, levelOf(candidateClass));

/** {@link classReference} by case id; undefined for an id the case table does not hold. */
export const gemvClassReference = (
  caseId: string,
  m: number,
  candidateClass: GemvCandidateClass,
): string | undefined => {
  const gemvCase = gemvCaseOf(caseId);
  return gemvCase === undefined ? undefined : classReference(gemvCase, m, candidateClass);
};

/**
 * Why a group's row of its class's reference is no reference, as far as the row itself says
 * ({@link gemvReferenceRefusal}): it failed, it has no time, or — for the identical class, where it
 * is not the record's own baseline — its words were not shown equal to the baseline's.
 */
export type GemvReferenceRefusal = "failed" | "untimed" | "unshown";

/**
 * Why a group's row of the reference ({@link gemvClassReference}) is no reference for
 * `candidateClass`, from the row alone (undefined: it is one). The derivation also refuses a row
 * that is not of the current runtime (its kernel fingerprint against the table it derives for),
 * which this does not read.
 */
export const gemvReferenceRefusal = (
  row: Pick<GemvSweepRow, "error" | "perDispatch" | "isBaseline" | "identicalToBaseline">,
  candidateClass: GemvCandidateClass,
): GemvReferenceRefusal | undefined => {
  if (row.error !== undefined) return "failed";
  if (row.perDispatch === undefined) return "untimed";
  // A candidate of the identical class shows its words equal to the record's baseline. Against
  // another kernel that says something only when that kernel's own words equal the baseline's
  if (candidateClass === "identical" && !row.isBaseline && row.identicalToBaseline !== true) {
    return "unshown";
  }
  return undefined;
};

/**
 * What one group's row says of its candidate against the reference's time there
 * ({@link gemvRowReading}): evidence with its gain (the reference's time ÷ its own); ruled out by a
 * failure or a failed check (its words not shown equal to the baseline's; outside the band — with
 * the gain when there is one); or no evidence (screened out by the quick look, skipped, no
 * reference time, no time of its own).
 */
export type GemvRowReading =
  | { readonly kind: "evidence"; readonly gain: number }
  | { readonly kind: "failed" }
  | { readonly kind: "unshown"; readonly gain?: number }
  | { readonly kind: "outside-band"; readonly gain?: number }
  | { readonly kind: "screened" }
  | { readonly kind: "skipped" }
  | { readonly kind: "no-reference" }
  | { readonly kind: "untimed" };

/**
 * How the derivation reads a current row of a group as a candidate of `candidateClass` against
 * `referenceTime` (the reference's time there; undefined when there is none). Correctness does not
 * depend on the reference: a failure or a failed check rules the candidate out whatever the time.
 * The derivation's reference time is the reference's fastest time over the records, repeated by a
 * second record; a reader of one record has its own group's time only.
 */
export const gemvRowReading = (
  row: Pick<
    GemvSweepRow,
    | "error"
    | "screened"
    | "skipped"
    | "perDispatch"
    | "isBaseline"
    | "identicalToBaseline"
    | "withinBand"
  >,
  candidateClass: GemvCandidateClass,
  referenceTime: number | undefined,
): GemvRowReading => {
  if (row.error !== undefined) return { kind: "failed" };
  if (row.screened === true) return { kind: "screened" };
  if (row.skipped === true) return { kind: "skipped" };
  // A failed check keeps its gain: the reader sees how fast the candidate was
  const gain =
    referenceTime === undefined || row.perDispatch === undefined
      ? undefined
      : referenceTime / row.perDispatch;
  const withGain = gain === undefined ? {} : { gain };
  // The record's baseline writes its own words: it is a candidate where the default moved off it
  if (candidateClass === "identical" && !row.isBaseline && row.identicalToBaseline !== true) {
    return { kind: "unshown", ...withGain };
  }
  if (candidateClass === "reorder" && row.withinBand !== true) {
    return { kind: "outside-band", ...withGain };
  }
  if (referenceTime === undefined) return { kind: "no-reference" };
  if (gain === undefined) return { kind: "untimed" };
  return { kind: "evidence", gain };
};

/** What a current group reads a class's candidates against (the module doc's "reference"). */
type GroupReference =
  /** Usable in the group, and its fastest time over the records is repeated: that time. */
  | { readonly kind: "timed"; readonly time: number }
  /** The group's row of it is no reference (missing, failed, untimed, not current, or words not shown equal). */
  | { readonly kind: "unusable"; readonly reason: string }
  /** Usable in the group, but no second record repeats its fastest time. */
  | { readonly kind: "unrepeated"; readonly reason: string }
  /** The sweep does not measure the default's choice (a row with packed activations). */
  | { readonly kind: "unmeasured"; readonly reason: string };

/** What a group is to the current runtime. */
type GroupState =
  | {
      readonly kind: "current";
      readonly group: GroupObservation;
      readonly gemvCase: GemvCase;
      /** The current kernels of the group's case and row count. */
      readonly kernels: ReadonlyMap<string, string>;
      /** By the level whose default choice is the reference. */
      readonly references: Readonly<Record<ReferenceLevel, GroupReference>>;
    }
  | {
      readonly kind: "stale";
      readonly group: GroupObservation;
      /** Why its case is not the current runtime's. */
      readonly reason: string;
    };

/** A group's case against the current runtime: its case and current kernels, or why it is not current. */
type CaseReading =
  | {
      readonly kind: "current";
      readonly gemvCase: GemvCase;
      readonly kernels: ReadonlyMap<string, string>;
    }
  | { readonly kind: "stale"; readonly reason: string };

/** Whether a group's case is the current runtime's (the module doc's "current"). */
const readCase = (
  source: GemvSweepSource,
  group: GroupObservation,
  kernelsOf: CurrentKernels,
): CaseReading => {
  const { caseId, m } = group;
  const gemvCase = gemvCaseOf(caseId);
  if (gemvCase === undefined) {
    return { kind: "stale", reason: `case ${caseId} is not in the case table` };
  }
  if (source.cases.get(caseId) !== gemvCaseFingerprint(gemvCase)) {
    return { kind: "stale", reason: `case ${caseId} changed (its fingerprint differs)` };
  }
  if (!gemvCase.rows.includes(m)) {
    return { kind: "stale", reason: `case ${caseId} has no row count ${m} in the case table` };
  }
  return { kind: "current", gemvCase, kernels: kernelsOf(gemvCase, m) };
};

/**
 * Why a group's row of the reference `name` is no reference at `level` (undefined: it is one).
 * `where` names the record and the row count in the reason; `tableId` the table the current kernels
 * were built for.
 */
const referenceRefusal = (
  source: GemvSweepSource,
  group: GroupObservation,
  kernels: ReadonlyMap<string, string>,
  reference: { readonly name: string; readonly candidateClass: GemvCandidateClass },
  where: string,
  tableId: string,
): string | undefined => {
  const { name, candidateClass } = reference;
  const row = group.rows.get(name);
  if (row === undefined) return `the reference ${name} is not measured ${where}`;
  const refusal = gemvReferenceRefusal(row, candidateClass);
  if (refusal === "failed") return `the reference ${name} failed ${where} (${row.error})`;
  if (refusal === "untimed") return `the reference ${name} has no time ${where}`;
  const staleness = rowStaleness(row, kernels);
  if (staleness !== undefined) {
    // The GEMM takes its tile geometry from the profile the sweep's device resolved to: a reading
    // against another geometry says nothing about this table's adapter
    if (row.choice.id === "gemm-f32" && row.kernel !== undefined) {
      const measured =
        source.geometryProfile === undefined
          ? "a profile the record does not name"
          : `the profile '${source.geometryProfile}'`;
      return `the GEMM reference ${where} was measured under ${measured}, and it is not the GEMM kernel the table '${tableId}' runs there`;
    }
    return `the reference ${name} is not current ${where} (${staleness})`;
  }
  if (refusal === "unshown") {
    return `the reference ${name} is not the baseline ${where}, and its words were not shown equal to the baseline's`;
  }
  return undefined;
};

/** A kernel's time in one record's group. */
type RecordTime = { readonly label: string; readonly time: number };

/**
 * The time of a reference from its times in the records' current groups: the fastest, when a second
 * record holds a time within {@link GEMV_REFERENCE_REPEAT} of it.
 */
const referenceTime = (times: readonly RecordTime[], name: string, m: number): GroupReference => {
  const fastest = times.reduce((best, entry) => (entry.time < best.time ? entry : best));
  const holders = new Set(
    times
      .filter(({ time }) => time <= fastest.time * GEMV_REFERENCE_REPEAT)
      .map(({ label }) => label),
  );
  if (holders.size >= MIN_RECORDS) return { kind: "timed", time: fastest.time };
  return {
    kind: "unrepeated",
    reason: `the reference ${name}'s fastest time at M=${m} (${fastest.label}) is repeated within ${formatRatio(GEMV_REFERENCE_REPEAT)} by no second record`,
  };
};

/** A record with its groups' states and its label in the verdict lines (its 1-based position). */
type Labelled = {
  readonly source: GemvSweepSource;
  readonly label: string;
  /** By {@link groupKey}. */
  readonly states: ReadonlyMap<string, GroupState>;
};

/**
 * The records with their groups' states. A reference's time is read over every record, so the cases
 * are read first, then the times of the current rows of every current group, then each current
 * group's references.
 */
const labelled = (
  sources: readonly GemvSweepSource[],
  kernelsOf: CurrentKernels,
  tableId: string,
): Labelled[] => {
  const read = sources.map((source, index) => ({
    source,
    label: `[${index + 1}]`,
    groups: [...source.groups].map(([key, group]) => ({
      key,
      group,
      reading: readCase(source, group, kernelsOf),
    })),
  }));
  // By group key and candidate: the time of every current row with a time, per record
  const times = new Map<string, RecordTime[]>();
  for (const { label, groups } of read) {
    for (const { key, group, reading } of groups) {
      if (reading.kind !== "current") continue;
      for (const row of group.rows.values()) {
        if (row.error !== undefined || row.perDispatch === undefined) continue;
        if (rowStaleness(row, reading.kernels) !== undefined) continue;
        const at = `${key} ${row.candidate}`;
        times.set(at, [...(times.get(at) ?? []), { label, time: row.perDispatch }]);
      }
    }
  }
  return read.map(({ source, label, groups }) => ({
    source,
    label,
    states: new Map(
      groups.map(({ key, group, reading }): [string, GroupState] => {
        if (reading.kind === "stale")
          return [key, { kind: "stale", group, reason: reading.reason }];
        const { gemvCase, kernels } = reading;
        const referenceAt = (candidateClass: GemvCandidateClass): GroupReference => {
          const name = classReference(gemvCase, group.m, candidateClass);
          if (!kernels.has(name)) {
            return {
              kind: "unmeasured",
              reason: `the default's choice at ${levelOf(candidateClass)} (${name}) is a kernel the sweep does not measure`,
            };
          }
          const where = `in ${label} at M=${group.m}`;
          const refusal = referenceRefusal(
            source,
            group,
            kernels,
            { name, candidateClass },
            where,
            tableId,
          );
          if (refusal !== undefined) return { kind: "unusable", reason: refusal };
          // The group's own row is current and timed, so the times hold at least it
          return referenceTime(times.get(`${key} ${name}`) ?? [], name, group.m);
        };
        return [
          key,
          {
            kind: "current",
            group,
            gemvCase,
            kernels,
            references: { exact: referenceAt("identical"), reorder: referenceAt("reorder") },
          },
        ];
      }),
    ),
  }));
};

/** A candidate in one group, against the reference of its class there. */
type Reading =
  | { readonly kind: "evidence"; readonly gain: number }
  /** A failure or a failed check in a current group: the candidate is not taken. */
  | { readonly kind: "ruled-out"; readonly reason: string; readonly gain?: number }
  /** The group is no evidence for it (not measured, stale, screened, no usable or repeated reference). */
  | { readonly kind: "none"; readonly reason: string };

const readGroup = (
  { states, label }: Labelled,
  caseId: string,
  m: number,
  candidate: string,
  candidateClass: GemvCandidateClass,
): Reading => {
  const state = states.get(groupKey(caseId, m));
  if (state === undefined) return { kind: "none", reason: `M=${m} is not measured in ${label}` };
  if (state.kind === "stale") {
    return { kind: "none", reason: `M=${m} is not current in ${label} (${state.reason})` };
  }
  const row = state.group.rows.get(candidate);
  if (row === undefined) return { kind: "none", reason: `not measured in ${label} at M=${m}` };
  const staleness = rowStaleness(row, state.kernels);
  if (staleness !== undefined) {
    return { kind: "none", reason: `not current in ${label} at M=${m} (${staleness})` };
  }
  const reference = state.references[levelOf(candidateClass)];
  const reading = gemvRowReading(
    row,
    candidateClass,
    reference.kind === "timed" ? reference.time : undefined,
  );
  const withGain = (gain: number | undefined) => (gain === undefined ? {} : { gain });
  switch (reading.kind) {
    case "evidence":
      return reading;
    case "failed":
      return { kind: "ruled-out", reason: `failed in ${label} at M=${m} (${row.error})` };
    case "screened":
      return { kind: "none", reason: `screened out by the quick look in ${label} at M=${m}` };
    case "skipped":
      return {
        kind: "none",
        reason: `skipped in ${label} at M=${m} (its M=1 group there left it unadoptable)`,
      };
    case "unshown":
      return {
        kind: "ruled-out",
        reason: `its words differ from the baseline's in ${label} at M=${m}`,
        ...withGain(reading.gain),
      };
    case "outside-band":
      return {
        kind: "ruled-out",
        reason: `outside the band in ${label} at M=${m}`,
        ...withGain(reading.gain),
      };
    case "untimed":
      return { kind: "none", reason: `has no time in ${label} at M=${m}` };
    case "no-reference":
      // gemvRowReading says so only when it was given no reference time: the reference is not timed
      return {
        kind: "none",
        reason: reference.kind === "timed" ? `has no time in ${label} at M=${m}` : reference.reason,
      };
  }
};

/**
 * The gain of a candidate in one group that is evidence: the time of the default's choice (the
 * reference's repeated fastest time over the records) ÷ its own time in the group.
 */
export type LinearGemvGain = {
  /** The record (its path, as given). */
  readonly record: string;
  readonly caseId: string;
  readonly gain: number;
};

/** A gain of a parallel row's kernel at one of its other row counts (M = 4 or 8). */
export type LinearGemvRowGain = LinearGemvGain & { readonly m: number };

/** One decision of the derivation: a shape (or the thread target) of one class, adopted or kept. */
export type LinearGemvDecision = {
  readonly class: GemvCandidateClass;
  readonly field: "sequential.shapes" | "parallelShapes" | "rowsThreadTarget";
  /**
   * The key of the field's row: `i4 n2048 k1536` (`sequential.shapes`, no group length) or
   * `i4 g32 n2048 k1536` (`parallelShapes`); empty for `rowsThreadTarget`.
   */
  readonly shape: string;
  /** The GEMV sweep cases the decision read. */
  readonly cases: readonly string[];
  /** The default profile's choice (`c32u4`, `parallel-l4`, `staged-l32`, `packed-l32`, `16384`). */
  readonly defaultChoice: string;
  /**
   * Which rule decided it. `one-record`: a steady record measured a case of it at M = 1 with a usable
   * reference, so the steady records decide by the rule of one steady record, and the choice to beat
   * is the table's (`tableChoice`). `two-record`: none did, so the records decide by the rule of two
   * records (×1.10 in every record against the reference's repeated fastest time), and the choice to
   * beat is the default's. The thread target's is `two-record` (no rule derives it).
   */
  readonly path: "one-record" | "two-record";
  /**
   * `one-record`, and a `two-record` decision that fewer than two records are evidence for (no
   * candidate at M = 1, no other row block of the default's row) and whose table's choice no record
   * rules out (failed, or its check failed, at M = 1): the choice the table the records
   * derive for holds for the shape (the default's where the table holds it; `no row (…)` for a
   * parallel decision without a row there).
   */
  readonly tableChoice?: string;
  /** Beside {@link tableChoice}, `parallelShapes`: the row block of the table's row (absent: none). */
  readonly tableRowBlock?: LinearGemvParallelRows;
  /**
   * The derived choice: under `one-record` the table's when kept, under `two-record` the default's
   * when kept — the table's where fewer than two records are evidence and no record rules it out
   * ({@link tableChoice}).
   */
  readonly derivedChoice: string;
  /**
   * `kept`: the choice to beat stays — under `one-record` the table's choice with its row block,
   * under `two-record` the default's (its row block may still move: then `adopted`), or the table's
   * where fewer than two records are evidence (no evidence moves nothing) and no record rules it out.
   */
  readonly outcome: "adopted" | "kept";
  /** Why the choice to beat stays (`outcome` kept), or under `one-record` why the derived choice won. */
  readonly reason?: string;
  /**
   * The candidate {@link gains} belong to: the adopted one, or when kept the one nearest to the
   * rule. Absent when no candidate was compared.
   */
  readonly candidate?: string;
  /** At M = 1, per group that is evidence (and per case of the key). */
  readonly gains: readonly LinearGemvGain[];
  /** `parallelShapes`: the row block of the default's row (absent: none). */
  readonly defaultRowBlock?: LinearGemvParallelRows;
  /** `parallelShapes`: the row block of the derived row (absent: none). */
  readonly derivedRowBlock?: LinearGemvParallelRows;
  /**
   * `parallelShapes`, when the derived row differs from the default's: the derived row's kernels at
   * its other row counts against the default's choice there, per group that is evidence.
   */
  readonly rowGains?: readonly LinearGemvRowGain[];
  /**
   * `one-record`: what the rule read, per case of the decision (several for a sequential key): every
   * candidate a steady record timed at M = 1 with its readings — gain, both gaps, the reference's
   * re-timing and the break-even margin — against the default's choice and the table's, its check,
   * and the kernels of its row at the other row counts under each row block with theirs. Margin-free:
   * {@link redecideLinearGemvChoices} decides it again under another margin from this alone.
   */
  readonly evidence?: readonly GemvOneRecordEvidence[];
  /** What the records showed that the field does not hold (the subgroup form). */
  readonly notes: readonly string[];
};

/** A group of a record whose case is not the current runtime's. */
export type LinearGemvStaleGroup = {
  readonly record: string;
  readonly caseId: string;
  readonly m: number;
  /** How its case differs (out of the case table, another fingerprint, a row count it no longer has). */
  readonly reason: string;
};

/** How much of a record stands for the current runtime. */
export type LinearGemvRecordGroups = {
  /** Groups whose case is the current runtime's. */
  readonly current: number;
  /** Groups whose case is not (each in {@link LinearGemvDerivation.stale}). */
  readonly stale: number;
  /**
   * Current groups whose row of a class's reference is no reference — missing, failed, untimed or
   * not current, or at `exact` neither the record's baseline nor shown to write its words —, by the
   * level whose default choice it is (`exact`: the identical class's, `reorder`: the reorder
   * class's). Such a group is no evidence for that class; a failure or a failed check in it still
   * rules a candidate out. A choice the sweep does not measure (a row with packed activations) is
   * not counted.
   */
  readonly withoutReference: { readonly exact: number; readonly reorder: number };
  /** Rows of current groups that are not current (their candidate's kernel changed, or it went). */
  readonly staleRows: number;
};

/** The return value of {@link deriveLinearGemvChoices}. */
export type LinearGemvDerivation = {
  /**
   * The field a table carries: the default's choices with the adopted ones in place, or absent when
   * every choice is the default's (a table without the field runs with the default's).
   */
  readonly linearGemv?: LinearGemvChoices;
  /** The default profile's choices, which the records measured against. */
  readonly defaults: LinearGemvChoices;
  /**
   * The choices of the table the records derive for (the default's where it carries none): what the
   * field starts from, each decision's derived choice then written in place; the rule of one steady
   * record keeps them unless beaten, and a shape no decision covers keeps its choice.
   */
  readonly tableChoices: LinearGemvChoices;
  /** Per class and shape, in the case table's order; the thread target last. */
  readonly decisions: readonly LinearGemvDecision[];
  /** The groups no evidence comes from because their case is not the current runtime's. */
  readonly stale: readonly LinearGemvStaleGroup[];
  /** The decisions as lines (a generated table's header, a generator's dry run). */
  readonly verdicts: readonly string[];
  /** The rule's numbers: those of the rule of one steady record, then those of the rule of two records. */
  readonly rule: LinearGemvRule;
  /** The records used: the adapter, the case table's fingerprint, and each record. */
  readonly provenance: {
    readonly adapter: Adapter;
    readonly caseTable: string;
    readonly records: readonly {
      readonly path: string;
      readonly sha256: string;
      readonly date: string;
      readonly candidateSet: GemvCandidateSet;
      readonly userAgent?: string;
      readonly groups: LinearGemvRecordGroups;
      /** Its class by its summaries. */
      readonly steadiness: GemvSteadiness;
      /** Whether the rule of one steady record reads it (steady, timestamps in ns). */
      readonly steady: boolean;
    }[];
  };
};

/** The numbers of the derivation's two rules. */
export type LinearGemvRule = {
  /** The rule of one steady record: the margin a gain at M = 1 must clear, times the two gaps. */
  readonly margin: number;
  /** A record's class: at least this many groups with a baseline spread … */
  readonly steadyMinGroups: number;
  /** … of which at most this share spreads beyond {@link LinearGemvRule.steadyGroupSpread}. */
  readonly steadyMaxShare: number;
  readonly steadyGroupSpread: number;
  /** A kernel whose pass is shorter (ms) makes its group no evidence under the rule of one record. */
  readonly minPassMs: number;
  /** The rounds a sweep takes by default (a gap is the second-smallest of them ÷ the smallest). */
  readonly rounds: number;
  /** The rule of two records: the threshold in every record. */
  readonly minSpeedup: number;
  /** The rule of two records: the factor within which a second record repeats a reference's fastest time. */
  readonly referenceRepeat: number;
};

/** Options of {@link deriveLinearGemvChoices}. */
export type DeriveLinearGemvOptions = {
  /**
   * The table the field is derived for: the records must be of an adapter its match selects and,
   * when it has provenance, of its adapter by vendor, architecture and device (a table is one
   * adapter's answer).
   */
  readonly table: GeometryProfile;
  /**
   * The margin of the rule of one steady record (default gemv-rule.ts's `GEMV_MARGIN`, ×1.065; a
   * finite number of at least 1).
   */
  readonly margin?: number;
};

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * The fields a table's provenance is compared by. Not the description: under Node on Dawn it is the
 * driver's version string ("NVIDIA: 620.70.06 …"), so comparing it would refuse every record made
 * after a driver update. The records of one derivation still agree on it among themselves
 * ({@link ADAPTER_FIELDS}), so records of two driver versions are not read together.
 */
const PROVENANCE_ADAPTER_FIELDS = ["vendor", "architecture", "device"] as const;

/** The records agree on the adapter, which the table's match selects and its provenance names. */
const assertOneAdapter = (
  sources: readonly GemvSweepSource[],
  table: GeometryProfile | undefined,
): void => {
  const [first] = sources;
  for (const source of sources) {
    for (const field of ADAPTER_FIELDS) {
      if (source.adapter[field] !== first.adapter[field]) {
        throw new Error(
          `${source.path}: adapter ${field} ${JSON.stringify(source.adapter[field])} differs from ${first.path}'s ${JSON.stringify(first.adapter[field])} (records of one adapter only)`,
        );
      }
    }
  }
  if (table === undefined) return;
  const { match, provenance } = table;
  if (match !== undefined) {
    for (const field of ["vendor", "architecture", "description"] as const) {
      const wanted = match[field];
      if (wanted !== undefined && first.adapter[field] !== wanted) {
        throw new Error(
          `${first.path}: adapter ${field} ${JSON.stringify(first.adapter[field])} is not table '${table.id}''s match ${JSON.stringify(wanted)}`,
        );
      }
    }
  }
  if (provenance !== undefined) {
    // MUST: no GEMV choices for a shared table — its tile geometry is derived from several devices
    // under a rule of its own, and no such rule exists for the GEMV choices yet
    // (SPEC: docs/spec/geometry-profiles.md#shared-tables)
    if (provenance.adapters !== undefined) {
      throw new Error(
        `Table '${table.id}' is a shared table (derived from ${provenance.adapters.length} adapters): it carries tile geometries only, and its GEMV choices are the default profile's`,
      );
    }
    for (const field of PROVENANCE_ADAPTER_FIELDS) {
      if (provenance.adapter[field] !== first.adapter[field]) {
        throw new Error(
          `${first.path}: adapter ${field} ${JSON.stringify(first.adapter[field])} differs from table '${table.id}''s provenance ${JSON.stringify(provenance.adapter[field])} (a table is one adapter's answer)`,
        );
      }
    }
  }
};

/** Reads the records and checks what no group of them could stand for (fails loudly). */
const readRecords = (
  inputs: readonly GemvSweepInput[],
  table: GeometryProfile | undefined,
): GemvSweepSource[] => {
  if (inputs.length === 0) throw new Error("No GEMV sweep record is given");
  const seen = new Map<string, string>();
  for (const { path, sha256 } of inputs) {
    const twin = seen.get(sha256);
    if (twin !== undefined) {
      throw new Error(`The same GEMV sweep record is passed twice (${twin} and ${path})`);
    }
    seen.set(sha256, path);
  }
  const sources = inputs.map(({ report, path, sha256 }) =>
    parseGemvSweepRecord(report, { path, sha256 }),
  );
  assertOneAdapter(sources, table);
  return sources;
};

/** A candidate's gains at M = 1 over its evidence, and the first way it fails the rule (if any). */
type Evaluation = {
  readonly name: string;
  readonly choice: GemvSweepChoice;
  readonly gains: readonly LinearGemvGain[];
  readonly failure?: string;
  /** The row block its row takes at the other row counts (a parallel row's; absent: none). */
  readonly rowBlock?: LinearGemvParallelRows;
  /** Its kernels' gains at the other row counts, with that row block. */
  readonly rowGains: readonly LinearGemvRowGain[];
};

/** A row's choice as a member of the GEMV family, or undefined for the GEMM. */
const familyOf = (choice: GemvSweepChoice): GemvSweepFamilyChoice | undefined =>
  choice.id === "gemm-f32" ? undefined : choice;

/**
 * The GEMV-family choice of an evaluated candidate. The judgements take the column forms and the
 * parallel forms only, so the GEMM (a reference of a float-stored weight at M ≥ 2) is never one.
 */
const familyChoiceOf = (evaluation: Evaluation): GemvSweepFamilyChoice => {
  if (evaluation.choice.id === "gemm-f32") {
    throw new Error(`${evaluation.name} is the GEMM, which no judgement takes as a candidate`);
  }
  return evaluation.choice;
};

/** The smallest of an evaluation's gains (0 without any — never the largest). */
const smallestGain = (evaluation: Evaluation): number =>
  evaluation.gains.length === 0 ? 0 : Math.min(...evaluation.gains.map(({ gain }) => gain));

/** The evaluation whose smallest gain is largest (the first of a tie); undefined for none. */
const largestSmallest = (evaluations: readonly Evaluation[]): Evaluation | undefined =>
  evaluations.reduce<Evaluation | undefined>(
    (best, entry) =>
      best === undefined || smallestGain(entry) > smallestGain(best) ? entry : best,
    undefined,
  );

/**
 * The case's row counts of the `no-regression` role (gemv-candidates.ts's `gemvGroupRole`). MUST: a
 * parallel row serves all of them, so a candidate without evidence at one of them cannot move the row
 * (a `quick` record measures M = 1 only).
 */
const noRegressionRows = (gemvCase: GemvCase): number[] =>
  gemvCase.rows.filter((m) => gemvGroupRole(m, "reorder") === "no-regression");

/** How a candidate set is judged for one case and class. */
type CaseJudgement = {
  readonly gemvCase: GemvCase;
  readonly class: GemvCandidateClass;
  /** The default's choice at row count `m` (by candidate name). */
  readonly reference: (m: number) => string;
  /** The candidates at M = 1 this judgement compares (by the row). */
  readonly takes: (row: Pick<RowObservation, "class" | "choice">) => boolean;
  /** The row counts besides M = 1 the decision reads (no regression allowed there). */
  readonly others: readonly number[];
  /**
   * Whether a candidate is read at the other row counts with each row block its lanes can carry (a
   * parallel row's candidates), instead of by its own kernel alone.
   */
  readonly rowBlocks: boolean;
};

/**
 * The candidate name of the kernel a parallel row runs at `m` rows: its row-blocked form under the
 * cap `rowBlock`, or — without a row block, and at M = 1 — the row's own (`name`).
 */
const rowKernelAt = (
  name: string,
  lanes: LinearGemvParallelLanes | undefined,
  rowBlock: LinearGemvParallelRows | undefined,
  m: number,
): string => {
  if (rowBlock === undefined || lanes === undefined) return name;
  const rows = linearGemvParallelRowsFor(m, rowBlock);
  return rows === undefined ? name : gemvParallelName("staged", lanes, rows);
};

/**
 * The row blocks a parallel row of `lanes` can carry: none first, then every cap the kernel exists
 * at. A float storage has none of the second kind, so a derived float row never carries a row block.
 */
const rowBlockOptions = (
  storage: GemvCase["storage"],
  lanes: LinearGemvParallelLanes | undefined,
): (LinearGemvParallelRows | undefined)[] => [
  undefined,
  ...(lanes === undefined
    ? []
    : LINEAR_GEMV_PARALLEL_ROWS.filter((rows) => linearGemvParallelRowsFits(storage, lanes, rows))),
];

/** A row's kernels at its other row counts: their gains, and the first way they fail. */
type OthersReading = {
  readonly gains: readonly LinearGemvRowGain[];
  readonly failure?: string;
};

/**
 * Reads the kernels `nameAt` names at the judgement's other row counts against the default's choice
 * there: at each, evidence from {@link MIN_RECORDS} records, no failure or failed check in a current
 * group, and no gain below {@link NO_REGRESSION}.
 */
const readOthers = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  nameAt: (m: number) => string,
): OthersReading => {
  const { gemvCase } = judgement;
  let failure: string | undefined;
  const fail = (text: string): void => {
    failure ??= text;
  };
  const gains: LinearGemvRowGain[] = [];
  for (const m of judgement.others) {
    const other = records.map((record) => ({
      label: record.label,
      path: record.source.path,
      reading: readGroup(record, gemvCase.id, m, nameAt(m), judgement.class),
    }));
    for (const { reading, label, path } of other) {
      if (reading.kind === "ruled-out") fail(reading.reason);
      if (reading.kind !== "evidence") continue;
      gains.push({ record: path, caseId: gemvCase.id, m, gain: reading.gain });
      if (reading.gain >= NO_REGRESSION) continue;
      fail(`slower than the default's choice in ${label} at M=${m} (${formatRatio(reading.gain)})`);
    }
    if (evidenceRecords(other).length < MIN_RECORDS) {
      fail(
        `evidence at M=${m} from fewer than ${MIN_RECORDS} records (a parallel row serves M = 1 to 8: ${noEvidenceText(other)})`,
      );
    }
  }
  return { gains, ...(failure === undefined ? {} : { failure }) };
};

/**
 * One row block against the reference row at one other row count where the two run different
 * kernels: per group that is evidence for both kernels the reference's time ÷ the row block's, the
 * number of records those groups come from, a failure or failed check of the row block's kernel in a
 * current group (`ruledOut`), and why the other groups are no evidence (`missing`).
 */
export type RowBlockAt = {
  readonly m: number;
  readonly gains: readonly { readonly label: string; readonly gain: number }[];
  readonly records: number;
  readonly ruledOut?: string;
  readonly missing?: string;
};

/**
 * Why a row block (or none) does not replace the one its reference row has, read from the row counts
 * where their kernels differ (`undefined`: it does). The row blocks of one lane count write the same
 * words, so the choice among them is speed alone, and one rule decides it wherever a row is derived:
 * it replaces the reference's only when it is never slower than the reference in a group that is
 * evidence — evidence from {@link MIN_RECORDS} records at each such row count, its check held in every
 * current group — and at least {@link GEMV_MIN_SPEEDUP} faster in every group of one row count or
 * more. A row count at which both run the same kernel is no reading: it neither refuses nor counts.
 */
export const rowBlockRefusal = (at: readonly RowBlockAt[]): string | undefined => {
  if (at.length === 0) return "runs the reference's kernel at every other row count";
  for (const { m, gains, records, ruledOut, missing } of at) {
    if (ruledOut !== undefined) return ruledOut;
    if (records < MIN_RECORDS) {
      return `evidence at M=${m} from fewer than ${MIN_RECORDS} records (a parallel row serves M = 1 to 8: ${missing ?? ""})`;
    }
    const slower = gains.find(({ gain }) => gain < NO_REGRESSION);
    if (slower !== undefined) {
      return `slower than the reference's row block in ${slower.label} at M=${m} (${formatRatio(slower.gain)})`;
    }
  }
  if (at.some(({ gains }) => gains.every(({ gain }) => gain >= GEMV_MIN_SPEEDUP))) return undefined;
  const best = Math.max(...at.map(({ gains }) => Math.min(...gains.map(({ gain }) => gain))));
  return `at no row count ${formatRatio(GEMV_MIN_SPEEDUP)} faster than the reference's row block in every group (at most ${formatRatio(best)})`;
};

/**
 * Of the row blocks that replace the reference's ({@link rowBlockRefusal}), the one whose smallest
 * gain over the row counts where it differs from the reference is largest (the first of a tie);
 * `undefined` when none does, and the reference's stays.
 */
export const pickRowBlock = <Option extends { readonly at: readonly RowBlockAt[] }>(
  options: readonly Option[],
): Option | undefined => {
  const score = ({ at }: Option): number =>
    Math.min(...at.flatMap(({ gains }) => gains.map(({ gain }) => gain)));
  return options
    .filter((option) => rowBlockRefusal(option.at) === undefined)
    .reduce<Option | undefined>(
      (best, option) => (best === undefined || score(option) > score(best) ? option : best),
      undefined,
    );
};

/** A row block read against a reference row of the same lanes ({@link rowBlockReadings}). */
type RowBlockOption = {
  readonly rowBlock: LinearGemvParallelRows | undefined;
  readonly at: readonly RowBlockAt[];
};

/**
 * Reads every row block of `options` against the reference row (`name` at `lanes` with the row
 * block `reference`) at the judgement's other row counts, where the two run different kernels. Both
 * kernels are read in the same group against the default's choice there, so a group is evidence only
 * when it is for both.
 */
const rowBlockReadings = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  name: string,
  lanes: LinearGemvParallelLanes | undefined,
  reference: LinearGemvParallelRows | undefined,
  options: readonly (LinearGemvParallelRows | undefined)[],
): RowBlockOption[] =>
  options.map((rowBlock) => ({
    rowBlock,
    at: judgement.others.flatMap((m): RowBlockAt[] => {
      const own = rowKernelAt(name, lanes, rowBlock, m);
      const theirs = rowKernelAt(name, lanes, reference, m);
      if (own === theirs) return [];
      const read = (kernel: string, record: Labelled): Reading =>
        readGroup(record, judgement.gemvCase.id, m, kernel, judgement.class);
      const gains: { label: string; gain: number }[] = [];
      const missing: string[] = [];
      let ruledOut: string | undefined;
      for (const record of records) {
        const mine = read(own, record);
        const base = read(theirs, record);
        if (mine.kind === "ruled-out") ruledOut ??= mine.reason;
        else if (mine.kind === "none") missing.push(mine.reason);
        else if (base.kind !== "evidence") {
          missing.push(`the reference's kernel ${theirs}: ${base.reason}`);
        } else gains.push({ label: record.label, gain: mine.gain / base.gain });
      }
      return [
        {
          m,
          gains,
          records: new Set(gains.map(({ label }) => label)).size,
          ...(ruledOut === undefined ? {} : { ruledOut }),
          ...(missing.length === 0 ? {} : { missing: missing.join("; ") }),
        },
      ];
    }),
  }));

/**
 * Every candidate the judgement takes that a record measured at M = 1 and the current runtime still
 * has there, in the order they were first measured.
 */
const candidatesOf = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  kernelsOf: CurrentKernels,
): { readonly name: string; readonly choice: GemvSweepChoice }[] => {
  const found = new Map<string, GemvSweepChoice>();
  const reference = judgement.reference(1);
  const current = kernelsOf(judgement.gemvCase, 1);
  for (const { source } of records) {
    const group = source.groups.get(groupKey(judgement.gemvCase.id, 1));
    for (const row of group?.rows.values() ?? []) {
      // A record's baseline that is not the current reference is a candidate like any other
      if (row.candidate === reference || !judgement.takes(row)) continue;
      if (!current.has(row.candidate)) continue;
      if (!found.has(row.candidate)) found.set(row.candidate, row.choice);
    }
  }
  return [...found].map(([name, choice]) => ({ name, choice }));
};

/** The distinct records of a list of readings that are evidence. */
const evidenceRecords = (
  readings: readonly { readonly label: string; readonly reading: Reading }[],
): string[] => [
  ...new Set(
    readings.flatMap(({ label, reading }) => (reading.kind === "evidence" ? [label] : [])),
  ),
];

/** The reasons a row count has too little evidence, in record order. */
const noEvidenceText = (
  readings: readonly { readonly label: string; readonly reading: Reading }[],
): string =>
  readings.flatMap(({ reading }) => (reading.kind === "none" ? [reading.reason] : [])).join("; ");

const evaluate = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  candidate: { readonly name: string; readonly choice: GemvSweepChoice },
): Evaluation => {
  const { gemvCase } = judgement;
  let failure: string | undefined;
  const fail = (text: string): void => {
    failure ??= text;
  };
  const readAt = (m: number) =>
    records.map((record) => ({
      label: record.label,
      path: record.source.path,
      reading: readGroup(record, gemvCase.id, m, candidate.name, judgement.class),
    }));
  const first = readAt(1);
  const gains: LinearGemvGain[] = [];
  for (const { reading, path, label } of first) {
    if (reading.kind === "ruled-out") fail(reading.reason);
    if (reading.kind !== "evidence") continue;
    gains.push({ record: path, caseId: gemvCase.id, gain: reading.gain });
    if (reading.gain < GEMV_MIN_SPEEDUP) {
      fail(`${formatRatio(reading.gain)} < ${formatRatio(GEMV_MIN_SPEEDUP)} in ${label}`);
    }
  }
  if (evidenceRecords(first).length < MIN_RECORDS) {
    fail(`evidence at M=1 from fewer than ${MIN_RECORDS} records (${noEvidenceText(first)})`);
  }
  // The row block of the row the candidate would give: against the candidate without one, by the
  // one rule of a row block ({@link rowBlockRefusal}). Then that row — its own kernel, or the
  // row-blocked one — may be slower than the default's choice at no other row count
  const { lanes } = familyOf(candidate.choice) ?? {};
  const rowBlock = judgement.rowBlocks
    ? pickRowBlock(
        rowBlockReadings(
          records,
          judgement,
          candidate.name,
          lanes,
          undefined,
          rowBlockOptions(gemvCase.storage, lanes).filter((option) => option !== undefined),
        ),
      )?.rowBlock
    : undefined;
  const others = readOthers(records, judgement, (m) =>
    rowKernelAt(candidate.name, lanes, rowBlock, m),
  );
  if (others.failure !== undefined) fail(others.failure);
  return {
    ...candidate,
    gains,
    ...(failure === undefined ? {} : { failure }),
    ...(rowBlock === undefined ? {} : { rowBlock }),
    rowGains: others.gains,
  };
};

/**
 * The outcome of one case and class: the adopted candidate, or why the default stays — and whether
 * the records judged the case at all (`judged`: groups of {@link MIN_RECORDS} records or more are
 * evidence for some candidate at M = 1).
 */
type CaseOutcome =
  | { readonly kind: "adopted"; readonly evaluation: Evaluation }
  | {
      readonly kind: "kept";
      readonly reason: string;
      readonly nearest?: Evaluation;
      readonly judged: boolean;
    };

/** Whether groups of {@link MIN_RECORDS} records or more are evidence for an evaluation at M = 1. */
const hasEvidence = ({ gains }: Evaluation): boolean =>
  new Set(gains.map(({ record }) => record)).size >= MIN_RECORDS;

const judgeCase = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  kernelsOf: CurrentKernels,
): CaseOutcome => {
  const evaluations = candidatesOf(records, judgement, kernelsOf).map((candidate) =>
    evaluate(records, judgement, candidate),
  );
  const qualifying = evaluations.filter((evaluation) => evaluation.failure === undefined);
  const best = largestSmallest(qualifying);
  if (best !== undefined) return { kind: "adopted", evaluation: best };
  const nearest = largestSmallest(evaluations);
  if (nearest === undefined) {
    return { kind: "kept", reason: "no candidate was measured", judged: false };
  }
  return {
    kind: "kept",
    reason: `no candidate is ${formatRatio(GEMV_MIN_SPEEDUP)} faster in the groups of ${MIN_RECORDS} or more records that are evidence, with its check held (nearest ${nearest.name}: ${nearest.failure ?? "?"})`,
    nearest,
    judged: evaluations.some(hasEvidence),
  };
};

/**
 * The reason of a decision of the rule of two records that no two records are evidence for: the
 * table's choice stays (no evidence is no ground to write the default's back), with why.
 */
const unjudgedReason = (reason: string): string =>
  `fewer than ${MIN_RECORDS} records are evidence for any candidate at M=1, so the table's choice stays (${reason})`;

/*
 * The rule of one steady record: what it reads of the steady records (margin-free evidence), which
 * gemv-one-record.ts's `decideOneRecord` turns into a decision at the margin.
 */

/** Whether the rule of one steady record reads a record: steady by its summaries, timestamps in ns. */
const readsSteady = (source: GemvSweepSource): boolean =>
  gemvRecordReading(source.steadiness, { unit: source.unit, quantized: false }) === "one-record";

/** ns per ms: a record read by the rule of one steady record times in ns. */
const NS_PER_MS = 1e6;

/**
 * A kernel's row in a steady record's group as the rule of one steady record reads its time, with
 * its re-timing (absent: not re-timed; `unread`: re-timed, and the pass read no time).
 */
type SteadyTime =
  | {
      readonly kind: "time";
      readonly time: number;
      readonly gap: number;
      readonly retime?: number | "unread";
    }
  | { readonly kind: "none" };

/**
 * A kernel's time, gap and re-timing in a group, or none: the row is current, timed and without
 * error, has a gap (two positive rounds or more), and its pass (its smallest round) is no shorter
 * than gemv-rule.ts's `GEMV_MIN_PASS_MS` — a shorter pass makes the group no evidence.
 */
const steadyTime = (state: GroupState | undefined, name: string): SteadyTime => {
  if (state?.kind !== "current") return { kind: "none" };
  const row = state.group.rows.get(name);
  if (row === undefined || rowStaleness(row, state.kernels) !== undefined) return { kind: "none" };
  return steadyRowTime(row);
};

/** A row's time, gap and re-timing as {@link steadyTime} reads them, whatever its group. */
const steadyRowTime = (
  row: Pick<
    RowObservation,
    "error" | "perDispatch" | "reps" | "rounds" | "retimedPerDispatch" | "retimingUnread"
  >,
): SteadyTime => {
  const { perDispatch, reps } = row;
  if (row.error !== undefined || perDispatch === undefined || reps === undefined) {
    return { kind: "none" };
  }
  // The pass the time is read from: the smallest positive round (perDispatch × reps)
  const pass = Math.min(...(row.rounds ?? []).filter((round) => round > 0));
  if (!(pass >= GEMV_MIN_PASS_MS * NS_PER_MS)) return { kind: "none" };
  const gap = gemvRoundsGap(row.rounds);
  if (gap === undefined) return { kind: "none" };
  return {
    kind: "time",
    time: perDispatch,
    gap,
    ...(row.retimingUnread !== undefined
      ? { retime: "unread" as const }
      : row.retimedPerDispatch === undefined
        ? {}
        : { retime: row.retimedPerDispatch / perDispatch }),
  };
};

/** The reading of two kernels' times in one group (`own` against `other`), or none. */
const readingOf = (
  own: SteadyTime,
  other: SteadyTime,
  where: { readonly record: string; readonly caseId: string; readonly m: number },
): GemvPairReading[] => {
  if (own.kind !== "time" || other.kind !== "time") return [];
  const gain = other.time / own.time;
  return [
    {
      ...where,
      gain,
      gap: own.gap,
      referenceGap: other.gap,
      ...(other.retime === undefined ? {} : { retime: other.retime }),
      breakEven: gemvBreakEven(gain, own.gap, other.gap),
    },
  ];
};

/** One steady record's reading of `name` against `reference` in its group of a case at `m`. */
const pairReading = (
  record: Labelled,
  caseId: string,
  m: number,
  name: string,
  reference: string,
): GemvPairReading[] => {
  const state = record.states.get(groupKey(caseId, m));
  return readingOf(steadyTime(state, name), steadyTime(state, reference), {
    record: record.source.path,
    caseId,
    m,
  });
};

/**
 * How the rule of one steady record reads a candidate's row against its class's reference row of
 * the same group, as a reader of one record asks it (the benchmark page): the reading — gain, both
 * gaps, the reference's re-timing and the break-even margin — when the reference's row is one
 * (`gemvReferenceRefusal`), the candidate's row is evidence with its check held (`gemvRowReading`),
 * and both rows have a time a steady record's group would be read by (no error, a pass of at least
 * 20 ms, two positive rounds or more); none otherwise, and none for a record not timed in ns (its
 * passes' length is not known). Whether the reading clears a margin is `gemvReadingIsEvidence` and
 * its break-even margin. The derivation reads rows by the same functions, with whether each row is
 * of the current runtime besides. `record` names the reading's record. Pure.
 */
export const gemvSteadyReading = (
  row: Pick<
    GemvSweepRow,
    | "caseId"
    | "m"
    | "error"
    | "screened"
    | "skipped"
    | "perDispatch"
    | "reps"
    | "rounds"
    | "retimedPerDispatch"
    | "retimingUnread"
    | "isBaseline"
    | "identicalToBaseline"
    | "withinBand"
  >,
  reference: Pick<
    GemvSweepRow,
    | "error"
    | "perDispatch"
    | "reps"
    | "rounds"
    | "retimedPerDispatch"
    | "retimingUnread"
    | "isBaseline"
    | "identicalToBaseline"
  >,
  candidateClass: GemvCandidateClass,
  unit: string,
  record = "",
): GemvPairReading | undefined => {
  if (unit !== "ns" || gemvReferenceRefusal(reference, candidateClass) !== undefined) {
    return undefined;
  }
  if (gemvRowReading(row, candidateClass, reference.perDispatch).kind !== "evidence") {
    return undefined;
  }
  const [reading] = readingOf(steadyRowTime(row), steadyRowTime(reference), {
    record,
    caseId: row.caseId,
    m: row.m,
  });
  return reading;
};

/** Whether a group's reference for a class is usable (its row is a reference, timed or not repeated). */
const referenceUsable = (state: GroupState | undefined, candidateClass: GemvCandidateClass) =>
  state?.kind === "current" &&
  (state.references[levelOf(candidateClass)].kind === "timed" ||
    state.references[levelOf(candidateClass)].kind === "unrepeated");

/** One steady record's reading of `name` against its class's reference (the default's choice) at `m`. */
const classReading = (
  record: Labelled,
  gemvCase: GemvCase,
  m: number,
  name: string,
  candidateClass: GemvCandidateClass,
): GemvPairReading[] =>
  referenceUsable(record.states.get(groupKey(gemvCase.id, m)), candidateClass)
    ? pairReading(record, gemvCase.id, m, name, classReference(gemvCase, m, candidateClass))
    : [];

/** The first failure or failed check of `name` in a current group of any record at `m`. */
const ruledOutIn = (
  records: readonly Labelled[],
  caseId: string,
  m: number,
  name: string,
  candidateClass: GemvCandidateClass,
): string | undefined => {
  for (const record of records) {
    const reading = readGroup(record, caseId, m, name, candidateClass);
    if (reading.kind === "ruled-out") return reading.reason;
  }
  return undefined;
};

/** What the derivation reads every decision with. */
type DerivationContext = {
  /** Every record, in the order given. */
  readonly records: readonly Labelled[];
  /** The records the rule of one steady record reads. */
  readonly steady: readonly Labelled[];
  readonly kernelsOf: CurrentKernels;
  readonly defaults: LinearGemvChoices;
  /** The choices of the table the records derive for (the default's where it carries none). */
  readonly table: LinearGemvChoices;
  readonly margin: number;
};

/**
 * Whether a steady record measured a case at M = 1 with a usable reference for the class: then the
 * case's decision of that class is the rule of one steady record's, in the derivation and in the plan.
 */
const measuredBySteady = (
  context: Pick<DerivationContext, "steady">,
  gemvCase: GemvCase,
  candidateClass: GemvCandidateClass,
): boolean =>
  context.steady.some((record) =>
    referenceUsable(record.states.get(groupKey(gemvCase.id, 1)), candidateClass),
  );

/**
 * The kernels of a row at the judgement's other row counts under each row block of `options`, read
 * against the default's choice there and, where it runs another kernel than the reference row (`name`
 * at `lanes` with the row block `reference`), against that row's kernel.
 */
const rowOptionsOf = (
  context: DerivationContext,
  judgement: CaseJudgement,
  row: {
    readonly name: string;
    readonly lanes: LinearGemvParallelLanes | undefined;
    readonly reference: LinearGemvParallelRows | undefined;
  },
  options: readonly (LinearGemvParallelRows | undefined)[],
): GemvRowBlockOption[] => {
  const { gemvCase } = judgement;
  return options.map((rowBlock) => ({
    ...(rowBlock === undefined ? {} : { rowBlock }),
    at: judgement.others.map((m): GemvRowCountEvidence => {
      const kernel = rowKernelAt(row.name, row.lanes, rowBlock, m);
      const theirs = rowKernelAt(row.name, row.lanes, row.reference, m);
      const ruledOut = ruledOutIn(context.records, gemvCase.id, m, kernel, judgement.class);
      return {
        m,
        kernel,
        readings: context.steady.flatMap((record) =>
          classReading(record, gemvCase, m, kernel, judgement.class),
        ),
        ...(ruledOut === undefined ? {} : { ruledOut }),
        ...(kernel === theirs
          ? {}
          : {
              block: {
                reference: theirs,
                readings: context.steady.flatMap((record) =>
                  pairReading(record, gemvCase.id, m, kernel, theirs),
                ),
              },
            }),
      };
    }),
  }));
};

/** A parallel row of `choices` for a case's shape (the first that matches, as the selection takes). */
const parallelRowOf = (
  choices: LinearGemvChoices,
  gemvCase: GemvCase,
): LinearGemvParallelShape | undefined =>
  choices.parallelShapes.find((row) => sameParallelShape(row, gemvCase));

/** The parallel name of a choice at M = 1 (`parallel-l8`, `staged-l32`), with its lanes; else none. */
const parallelNameOf = (
  name: string,
): { readonly staged: boolean; readonly lanes: LinearGemvParallelLanes } | undefined => {
  const match = /^(parallel|staged)-l(\d+)$/.exec(name);
  const lanes = match === null ? undefined : Number(match[2]);
  if (match === null || !isLanes(lanes)) return undefined;
  return { staged: match[1] === "staged", lanes };
};

/** The class of a kernel of a case at M = 1, by name (the table's choice may be of the other class). */
const classOfKernel = (gemvCase: GemvCase, name: string): GemvCandidateClass | undefined =>
  gemvKernelsAt(gemvCase, 1).find((candidate) => candidate.name === name)?.class;

/**
 * The table's choice for a judgement's case at M = 1 (by kernel name) beside the default's, and why
 * it is not held: where it is another kernel than the default's choice, it failed or its check failed
 * in a current group of a record. Both rules read it here, so a table's choice that a record rules
 * out stays under neither — not even where no record is evidence for any candidate.
 */
const tableAtM1 = (
  context: Pick<DerivationContext, "records" | "table">,
  judgement: CaseJudgement,
): { readonly defaultChoice: string; readonly tableChoice: string; readonly ruledOut?: string } => {
  const { gemvCase } = judgement;
  const defaultChoice = judgement.reference(1);
  const tableChoice = gemvProfileChoice(gemvCase, 1, levelOf(judgement.class), context.table).name;
  const tableClass = classOfKernel(gemvCase, tableChoice) ?? judgement.class;
  const ruledOut =
    tableChoice === defaultChoice
      ? undefined
      : ruledOutIn(context.records, gemvCase.id, 1, tableChoice, tableClass);
  return { defaultChoice, tableChoice, ...(ruledOut === undefined ? {} : { ruledOut }) };
};

/** What the rule of one steady record reads for one case and class (gemv-one-record.ts). */
const oneRecordEvidence = (
  context: DerivationContext,
  judgement: CaseJudgement,
): GemvOneRecordEvidence => {
  const { gemvCase } = judgement;
  const { defaultChoice, tableChoice, ruledOut: tableRuledOut } = tableAtM1(context, judgement);
  const parallel = judgement.class === "reorder";
  const defaultRow = parallel ? parallelRowOf(context.defaults, gemvCase) : undefined;
  const tableRow = parallel ? parallelRowOf(context.table, gemvCase) : undefined;
  const differs = tableChoice !== defaultChoice;
  const tableTimed =
    !differs ||
    context.steady.some(
      (record) =>
        steadyTime(record.states.get(groupKey(gemvCase.id, 1)), tableChoice).kind === "time",
    );
  // Against the table's choice only where it is another kernel and still held
  const againstTable = (name: string): GemvPairReading[] =>
    !differs || tableRuledOut !== undefined
      ? []
      : context.steady.flatMap((record) =>
          referenceUsable(record.states.get(groupKey(gemvCase.id, 1)), judgement.class)
            ? pairReading(record, gemvCase.id, 1, name, tableChoice)
            : [],
        );
  const candidates = candidatesOf(context.records, judgement, context.kernelsOf).flatMap(
    ({ name, choice }): GemvCandidateEvidence[] => {
      const m1 = context.steady.flatMap((record) =>
        classReading(record, gemvCase, 1, name, judgement.class),
      );
      const vsTable = name === tableChoice ? [] : againstTable(name);
      if (m1.length === 0 && vsTable.length === 0) return [];
      const ruledOut = ruledOutIn(context.records, gemvCase.id, 1, name, judgement.class);
      const { lanes } = familyOf(choice) ?? {};
      const rows = judgement.rowBlocks
        ? rowOptionsOf(
            context,
            judgement,
            { name, lanes, reference: undefined },
            rowBlockOptions(gemvCase.storage, lanes),
          )
        : judgement.others.length > 0
          ? rowOptionsOf(context, judgement, { name, lanes, reference: undefined }, [undefined])
          : [];
      return [
        { candidate: name, ...(ruledOut === undefined ? {} : { ruledOut }), m1, vsTable, rows },
      ];
    },
  );
  // A kept row may move its row block: the table's row and, where it differs, the default's
  const keptOf = (choice: string, row: LinearGemvParallelShape | undefined): GemvKeptRow[] => {
    if (!judgement.rowBlocks || row === undefined || row.packedActivations) return [];
    return [
      {
        choice,
        ...(row.rowBlock === undefined ? {} : { rowBlock: row.rowBlock }),
        options: rowOptionsOf(
          context,
          judgement,
          { name: choice, lanes: row.lanes, reference: row.rowBlock },
          rowBlockOptions(gemvCase.storage, row.lanes).filter(
            (rowBlock) => rowBlock !== row.rowBlock,
          ),
        ),
      },
    ];
  };
  return {
    caseId: gemvCase.id,
    class: judgement.class,
    defaultChoice,
    ...(defaultRow?.rowBlock === undefined ? {} : { defaultRowBlock: defaultRow.rowBlock }),
    tableChoice,
    ...(tableRow?.rowBlock === undefined ? {} : { tableRowBlock: tableRow.rowBlock }),
    ...(tableRuledOut === undefined ? {} : { tableRuledOut }),
    tableTimed,
    candidates,
    defaultVsTable: againstTable(defaultChoice),
    keptRows: [
      ...keptOf(tableChoice, tableRow),
      ...(differs ? keptOf(defaultChoice, defaultRow) : []),
    ],
  };
};

/** The M = 1 gains of the candidate an outcome names, as a decision carries them. */
const outcomeGains = (
  evidence: GemvOneRecordEvidence,
  outcome: GemvOneRecordOutcome,
): LinearGemvGain[] =>
  (evidence.candidates.find(({ candidate }) => candidate === outcome.candidate)?.m1 ?? []).map(
    ({ record, caseId, gain }) => ({ record, caseId, gain }),
  );

/** The decision of a sequential key from its cases' evidence (they must agree, or the table's stays). */
const sequentialOfEvidence = (
  head: Omit<
    LinearGemvDecision,
    "derivedChoice" | "outcome" | "reason" | "candidate" | "gains" | "path" | "evidence"
  >,
  evidence: readonly GemvOneRecordEvidence[],
  margin: number,
): LinearGemvDecision => {
  const outcomes = evidence.map((one) => decideOneRecord(one, margin));
  const [first] = evidence;
  if (first === undefined) throw new Error(`The key ${head.shape} has no case`);
  const tableChoice = first.tableChoice;
  const names = new Set(outcomes.map(({ choice }) => choice));
  const agreed = names.size === 1 ? outcomes[0] : undefined;
  const common = { ...head, path: "one-record", tableChoice, evidence } as const;
  if (agreed !== undefined) {
    const shown = agreed.candidate;
    return {
      ...common,
      derivedChoice: agreed.choice,
      outcome: agreed.choice === tableChoice ? "kept" : "adopted",
      reason: agreed.reason,
      ...(shown === undefined ? {} : { candidate: shown }),
      gains: evidence.flatMap((one, index) => {
        const outcome = outcomes[index];
        return outcome === undefined ? [] : outcomeGains(one, outcome);
      }),
    };
  }
  return {
    ...common,
    derivedChoice: tableChoice,
    outcome: "kept",
    reason: `the cases that share the key disagree, so the table's choice stays (${evidence
      .map((one, index) => `${one.caseId} ${outcomes[index]?.choice ?? "?"}`)
      .join("; ")})`,
    gains: [],
  };
};

/** The name a parallel decision gives a choice that is no parallel row. */
const noRowName = (name: string): string =>
  parallelNameOf(name) === undefined && !name.startsWith("no row") ? `no row (${name})` : name;

/** The decision of a parallel row from its case's evidence. */
const parallelOfEvidence = (
  head: Omit<
    LinearGemvDecision,
    | "derivedChoice"
    | "outcome"
    | "reason"
    | "candidate"
    | "gains"
    | "path"
    | "evidence"
    | "derivedRowBlock"
    | "rowGains"
  >,
  evidence: GemvOneRecordEvidence,
  margin: number,
): LinearGemvDecision => {
  const outcome = decideOneRecord(evidence, margin);
  const chosen = evidence.candidates.find(({ candidate }) => candidate === outcome.choice);
  const rowGains = (
    chosen?.rows.find(({ rowBlock }) => rowBlock === outcome.rowBlock)?.at ?? []
  ).flatMap(({ readings }) =>
    readings.map(({ record, caseId, m, gain }) => ({ record, caseId, m, gain })),
  );
  return {
    ...head,
    path: "one-record",
    tableChoice: noRowName(evidence.tableChoice),
    ...(evidence.tableRowBlock === undefined ? {} : { tableRowBlock: evidence.tableRowBlock }),
    derivedChoice: noRowName(outcome.choice),
    ...(outcome.rowBlock === undefined ? {} : { derivedRowBlock: outcome.rowBlock }),
    outcome: outcome.outcome,
    reason: outcome.reason,
    ...(outcome.candidate === undefined ? {} : { candidate: outcome.candidate }),
    gains: outcomeGains(evidence, outcome),
    ...(rowGains.length === 0 ? {} : { rowGains }),
    evidence: [evidence],
  };
};

/** A sequential variant by its name (`c32u4`). */
const variantOfName = (name: string): LinearGemvVariant => {
  const match = /^c(\d+)u(\d+)$/.exec(name);
  if (match === null) throw new Error(`${name} is no sequential variant`);
  return { cols: Number(match[1]), unroll: Number(match[2]) };
};

/**
 * The field from `base` (the table's choices) with every decision's derived choice in place: a
 * sequential key's exception replaced, added, or removed when the derived variant is the general one;
 * a parallel row replaced in place, added when the shape had none, or removed when the derived choice
 * is no row. A decision whose derived choice the base already holds changes nothing, and so does a
 * row with packed activations (never measured). The general variant and the thread target are the
 * default's.
 */
const applyDecisions = (
  base: LinearGemvChoices,
  defaults: LinearGemvChoices,
  decisions: readonly LinearGemvDecision[],
): LinearGemvChoices => {
  let shapes: LinearGemvShapeVariant[] = [...base.sequential.shapes];
  let parallelShapes: LinearGemvParallelShape[] = [...base.parallelShapes];
  const general = defaults.sequential.variant;
  for (const decision of decisions) {
    const [caseId] = decision.cases;
    const gemvCase = caseId === undefined ? undefined : gemvCaseOf(caseId);
    if (gemvCase === undefined) continue;
    if (decision.field === "sequential.shapes") {
      const variant = variantOfName(decision.derivedChoice);
      const index = shapes.findIndex((row) => sameSequentialShape(row, gemvCase));
      const isGeneral = variant.cols === general.cols && variant.unroll === general.unroll;
      const held = index < 0 ? general : shapes[index]?.variant;
      if (held?.cols === variant.cols && held.unroll === variant.unroll) continue;
      const row: LinearGemvShapeVariant = {
        storage: gemvCase.storage,
        n: gemvCase.n,
        k: gemvCase.k,
        variant,
      };
      if (isGeneral) shapes = shapes.filter((_, at) => at !== index);
      else if (index < 0) shapes = [...shapes, row];
      else shapes = shapes.map((entry, at) => (at === index ? row : entry));
      continue;
    }
    if (decision.field !== "parallelShapes") continue;
    const index = parallelShapes.findIndex((entry) => sameParallelShape(entry, gemvCase));
    const held = parallelShapes[index];
    if (held?.packedActivations === true) continue;
    const parallel = parallelNameOf(decision.derivedChoice);
    if (parallel === undefined) {
      if (index >= 0) parallelShapes = parallelShapes.filter((_, at) => at !== index);
      continue;
    }
    if (
      held !== undefined &&
      held.lanes === parallel.lanes &&
      (held.stagedActivations === true) === parallel.staged &&
      held.rowBlock === decision.derivedRowBlock
    ) {
      continue;
    }
    const row: LinearGemvParallelShape = {
      storage: gemvCase.storage,
      n: gemvCase.n,
      k: gemvCase.k,
      ...(gemvCase.group === undefined ? {} : { group: gemvCase.group }),
      lanes: parallel.lanes,
      packedActivations: false,
      ...(parallel.staged ? { stagedActivations: true } : {}),
      ...(decision.derivedRowBlock === undefined ? {} : { rowBlock: decision.derivedRowBlock }),
    };
    // The selection takes the first row that matches, so the first one is replaced
    parallelShapes =
      index < 0
        ? [...parallelShapes, row]
        : parallelShapes.map((entry, at) => (at === index ? row : entry));
  }
  return {
    sequential: { variant: general, shapes },
    rowsThreadTarget: defaults.rowsThreadTarget,
    parallelShapes,
  };
};

const variantName = ({ cols, unroll }: LinearGemvVariant): string => `c${cols}u${unroll}`;

const parallelName = (row: LinearGemvParallelShape): string =>
  `${row.packedActivations ? "packed" : row.stagedActivations === true ? "staged" : "parallel"}-l${row.lanes}`;

const groupPart = (group: number | undefined): string => (group === undefined ? "" : ` g${group}`);

const sequentialKey = ({ storage, n, k }: GemvCase): string => `${storage} n${n} k${k}`;

const parallelKey = ({ storage, group, n, k }: GemvCase): string =>
  `${storage}${groupPart(group)} n${n} k${k}`;

const sameParallelShape = (row: LinearGemvParallelShape, gemvCase: GemvCase): boolean =>
  row.storage === gemvCase.storage &&
  row.n === gemvCase.n &&
  row.k === gemvCase.k &&
  row.group === gemvCase.group;

const sameSequentialShape = (row: LinearGemvShapeVariant, gemvCase: GemvCase): boolean =>
  row.storage === gemvCase.storage && row.n === gemvCase.n && row.k === gemvCase.k;

/**
 * The members of the GEMV family whose M = 1 rows a table holds, per class: the sequential variant
 * (`sequential.shapes`) and the plain and staged parallel GEMV (`parallelShapes`). The subgroup form
 * is a Session option, not a row (only a decision's notes name it).
 */
const TABLE_CHOICES = {
  identical: ["gemv-m1"],
  reorder: ["gemv-parallel", "gemv-parallel-staged"],
} as const satisfies Readonly<Record<GemvCandidateClass, readonly GemvSweepChoice["id"][]>>;

/**
 * The rows of `candidateClass` a judgement takes at M = 1: those of the members `ids`, without a row
 * block (a row-blocked kernel is a row's form at M ≥ 2, no candidate at M = 1).
 */
const takesMembers =
  (candidateClass: GemvCandidateClass, ids: readonly GemvSweepChoice["id"][]) =>
  (row: Pick<RowObservation, "class" | "choice">): boolean =>
    row.class === candidateClass &&
    ids.includes(row.choice.id) &&
    familyOf(row.choice)?.parallelRows === undefined;

/**
 * Whether the derivation takes a row of an M = 1 group (`gemvGroupRole`'s `winner`) as a candidate
 * that a table can hold: a sequential variant of the identical class, or a plain or staged parallel
 * row of the reorder class. Such a candidate replaces the default's choice for its shape when it is
 * at least `GEMV_MIN_SPEEDUP` faster than that choice, with its check held, in the groups of two or
 * more records. Pure.
 */
export const gemvTableCandidate = (row: Pick<GemvSweepRow, "class" | "choice">): boolean =>
  takesMembers(row.class, TABLE_CHOICES[row.class])(row);

/** The identical class of one case at M = 1: the column form's variants against the baseline. */
const identicalJudgement = (gemvCase: GemvCase): CaseJudgement => {
  const baseline = classReference(gemvCase, 1, "identical");
  return {
    gemvCase,
    class: "identical",
    reference: () => baseline,
    takes: takesMembers("identical", TABLE_CHOICES.identical),
    others: [],
    rowBlocks: false,
  };
};

/**
 * The reorder class of one case: for the table the plain and the staged form, read at the other row
 * counts with the row blocks their lanes can carry; for the notes the subgroup form, by its own
 * kernel. A row-blocked kernel is no candidate at M = 1 (it is a row's form at M ≥ 2).
 */
const reorderJudgement = (
  gemvCase: GemvCase,
  ids: readonly GemvSweepChoice["id"][],
  rowBlocks: boolean,
): CaseJudgement => ({
  gemvCase,
  class: "reorder",
  reference: (m) => classReference(gemvCase, m, "reorder"),
  takes: takesMembers("reorder", ids),
  others: noRegressionRows(gemvCase),
  rowBlocks,
});

/**
 * The sequential key's decision from its cases' (they must agree): by the rule of one steady record
 * when a steady record measured a case of the key, otherwise by the rule of two records.
 */
const sequentialDecision = (
  context: DerivationContext,
  cases: readonly GemvCase[],
): LinearGemvDecision => {
  const { records, defaults, kernelsOf } = context;
  const [first] = cases;
  const defaultChoice = variantName(
    defaults.sequential.shapes.find((row) => sameSequentialShape(row, first))?.variant ??
      defaults.sequential.variant,
  );
  const head = {
    class: "identical",
    field: "sequential.shapes",
    shape: sequentialKey(first),
    cases: cases.map(({ id }) => id),
    defaultChoice,
    notes: [],
  } as const;
  if (cases.some((gemvCase) => measuredBySteady(context, gemvCase, "identical"))) {
    return sequentialOfEvidence(
      head,
      cases.map((gemvCase) => oneRecordEvidence(context, identicalJudgement(gemvCase))),
      context.margin,
    );
  }
  const outcomes = cases.map((gemvCase) => ({
    gemvCase,
    outcome: judgeCase(records, identicalJudgement(gemvCase), kernelsOf),
  }));
  const twoRecord = { ...head, path: "two-record" } as const;
  const adopted = outcomes.flatMap(({ outcome }) =>
    outcome.kind === "adopted" ? [outcome.evaluation] : [],
  );
  const names = new Set(adopted.map(({ name }) => name));
  if (adopted.length === outcomes.length && names.size === 1) {
    const [chosen] = adopted;
    return {
      ...twoRecord,
      derivedChoice: variantName(familyChoiceOf(chosen).variant),
      outcome: "adopted",
      candidate: chosen.name,
      gains: adopted.flatMap(({ gains }) => gains),
    };
  }
  const nearest = outcomes.flatMap(({ outcome }) =>
    outcome.kind === "adopted"
      ? [outcome.evaluation]
      : outcome.nearest === undefined
        ? []
        : [outcome.nearest],
  );
  const said = (outcome: CaseOutcome): string =>
    outcome.kind === "kept" ? outcome.reason : `adopts ${outcome.evaluation.name}`;
  const reason =
    outcomes.length === 1
      ? said(outcomes[0].outcome)
      : adopted.length === 0
        ? outcomes.map(({ gemvCase, outcome }) => `${gemvCase.id}: ${said(outcome)}`).join("; ")
        : `the cases that share the key disagree, so the default stays (${outcomes
            .map(({ gemvCase, outcome }) => `${gemvCase.id} ${said(outcome)}`)
            .join("; ")})`;
  const shown = nearest.length === 1 ? nearest[0] : undefined;
  const shownFields = {
    ...(shown === undefined ? {} : { candidate: shown.name }),
    gains: shown === undefined ? [] : shown.gains,
  };
  // MUST: records that are evidence for no candidate of the key do not move the table's choice
  // (SPEC: docs/spec/geometry-profiles.md#deriving-the-gemv-choices)
  if (outcomes.every(({ outcome }) => outcome.kind === "kept" && !outcome.judged)) {
    const tableChoice = gemvProfileChoice(first, 1, "exact", context.table).name;
    // MUST: unless a record rules that choice out, as under the rule of one steady record
    const ruled = cases
      .map((gemvCase) => ({
        gemvCase,
        ruledOut: tableAtM1(context, identicalJudgement(gemvCase)).ruledOut,
      }))
      .find(({ ruledOut }) => ruledOut !== undefined);
    if (ruled?.ruledOut !== undefined) {
      const where = cases.length === 1 ? ruled.ruledOut : `${ruled.gemvCase.id}: ${ruled.ruledOut}`;
      return {
        ...twoRecord,
        derivedChoice: defaultChoice,
        outcome: "kept",
        reason: gemvTableNotHeld(tableChoice, where, reason),
        ...shownFields,
      };
    }
    return {
      ...twoRecord,
      tableChoice,
      derivedChoice: tableChoice,
      outcome: "kept",
      reason: unjudgedReason(reason),
      ...shownFields,
    };
  }
  return {
    ...twoRecord,
    derivedChoice: defaultChoice,
    outcome: "kept",
    reason,
    ...shownFields,
  };
};

/**
 * Whether the subgroup form would meet the rule of one steady record at the case (a Session option,
 * not a row: only the decision's notes name it), against the default's choice.
 */
const subgroupNote = (context: DerivationContext, gemvCase: GemvCase): string[] => {
  const evidence = oneRecordEvidence(
    context,
    reorderJudgement(gemvCase, ["gemv-parallel-subgroup32"], false),
  );
  const outcome = decideOneRecord(
    { ...evidence, tableChoice: evidence.defaultChoice, keptRows: [] },
    context.margin,
  );
  return outcome.outcome === "adopted" && outcome.candidate !== undefined
    ? [`${outcome.candidate} meets the rule too (${outcome.reason}): a Session option, not a row`]
    : [];
};

/**
 * The parallel row's decision of one case: by the rule of one steady record when a steady record
 * measured the case, otherwise by the rule of two records.
 */
const parallelDecision = (context: DerivationContext, gemvCase: GemvCase): LinearGemvDecision => {
  const { records, defaults, kernelsOf } = context;
  const defaultRow = defaults.parallelShapes.find((row) => sameParallelShape(row, gemvCase));
  const defaultChoice =
    defaultRow === undefined
      ? `no row (${gemvBaseline(gemvCase, 1).name})`
      : parallelName(defaultRow);
  const path = measuredBySteady(context, gemvCase, "reorder") ? "one-record" : "two-record";
  const subgroup =
    path === "one-record"
      ? undefined
      : judgeCase(
          records,
          reorderJudgement(gemvCase, ["gemv-parallel-subgroup32"], false),
          kernelsOf,
        );
  const notes =
    path === "one-record"
      ? subgroupNote(context, gemvCase)
      : subgroup?.kind === "adopted"
        ? [
            `${subgroup.evaluation.name} meets the rule too (${gainsText(subgroup.evaluation.gains, labelsOf(records.map(({ source }) => source.path)))}): a Session option, not a row`,
          ]
        : [];
  const head = {
    class: "reorder",
    field: "parallelShapes",
    shape: parallelKey(gemvCase),
    cases: [gemvCase.id],
    defaultChoice,
    notes,
    ...(defaultRow?.rowBlock === undefined ? {} : { defaultRowBlock: defaultRow.rowBlock }),
  } as const;
  /** The fields of a decision whose row keeps the default's row block. */
  const sameRowBlock =
    defaultRow?.rowBlock === undefined ? {} : { derivedRowBlock: defaultRow.rowBlock };
  if (defaultRow?.packedActivations === true) {
    return {
      ...head,
      path,
      derivedChoice: defaultChoice,
      outcome: "kept",
      reason: "the default takes packed activations, whose kernels the sweep does not measure",
      gains: [],
      ...sameRowBlock,
    };
  }
  const judgement = reorderJudgement(gemvCase, TABLE_CHOICES.reorder, true);
  if (path === "one-record") {
    return parallelOfEvidence(head, oneRecordEvidence(context, judgement), context.margin);
  }
  const twoRecord = { ...head, path } as const;
  const outcome = judgeCase(records, judgement, kernelsOf);
  if (outcome.kind === "kept") {
    // M = 1 keeps the default's row: its lanes stay, and another row block may still replace its
    // own, by the one rule of a row block with the default's row as the reference
    const tried =
      defaultRow === undefined
        ? []
        : rowBlockReadings(
            records,
            judgement,
            defaultChoice,
            defaultRow.lanes,
            defaultRow.rowBlock,
            rowBlockOptions(gemvCase.storage, defaultRow.lanes).filter(
              (rowBlock) => rowBlock !== defaultRow.rowBlock,
            ),
          );
    const other = pickRowBlock(tried);
    if (defaultRow !== undefined && other !== undefined) {
      return {
        ...twoRecord,
        derivedChoice: defaultChoice,
        outcome: "adopted",
        gains: [],
        ...(other.rowBlock === undefined ? {} : { derivedRowBlock: other.rowBlock }),
        rowGains: readOthers(records, judgement, (m) =>
          rowKernelAt(defaultChoice, defaultRow.lanes, other.rowBlock, m),
        ).gains,
      };
    }
    // Why no other row block replaced the default's, where one reached the threshold in some group
    // (a row whose row blocks are nowhere near it needs no line)
    const rowBlockNote = tried.some(({ at }) =>
      at.some(({ gains }) => gains.some(({ gain }) => gain >= GEMV_MIN_SPEEDUP)),
    )
      ? [
          `no other row block replaces the default's: ${tried
            .map(
              ({ rowBlock, at }) =>
                `${rowBlock === undefined ? "none" : `${rowBlock} rows`} — ${rowBlockRefusal(at) ?? "?"}`,
            )
            .join("; ")}`,
        ]
      : [];
    const nearestFields = {
      ...(outcome.nearest === undefined ? {} : { candidate: outcome.nearest.name }),
      gains: outcome.nearest?.gains ?? [],
    };
    // MUST: records that are evidence for no candidate at M = 1, and for no other row block of the
    // default's row, do not move the table's row (SPEC: docs/spec/geometry-profiles.md#deriving-the-gemv-choices)
    const rowBlockJudged = tried.some(({ at }) => at.some(({ records }) => records >= MIN_RECORDS));
    const unjudged = !outcome.judged && !rowBlockJudged;
    const tableRow = parallelRowOf(context.table, gemvCase);
    const tableChoice =
      tableRow === undefined
        ? `no row (${gemvBaseline(gemvCase, 1).name})`
        : parallelName(tableRow);
    // MUST: unless a record rules that choice out, as under the rule of one steady record
    const tableRuledOut = unjudged ? tableAtM1(context, judgement).ruledOut : undefined;
    if (unjudged && tableRuledOut === undefined) {
      const rowBlock = tableRow?.rowBlock;
      return {
        ...twoRecord,
        notes: [...notes, ...rowBlockNote],
        tableChoice,
        ...(rowBlock === undefined ? {} : { tableRowBlock: rowBlock }),
        derivedChoice: tableChoice,
        outcome: "kept",
        reason: unjudgedReason(outcome.reason),
        ...nearestFields,
        ...(rowBlock === undefined ? {} : { derivedRowBlock: rowBlock }),
      };
    }
    return {
      ...twoRecord,
      notes: [...notes, ...rowBlockNote],
      derivedChoice: defaultChoice,
      outcome: "kept",
      reason:
        tableRuledOut === undefined
          ? outcome.reason
          : gemvTableNotHeld(tableChoice, tableRuledOut, outcome.reason),
      ...nearestFields,
      ...sameRowBlock,
    };
  }
  const { evaluation } = outcome;
  if (familyChoiceOf(evaluation).lanes === undefined) {
    throw new Error(`${evaluation.name} of ${gemvCase.id} has no lanes`);
  }
  return {
    ...twoRecord,
    derivedChoice: evaluation.name,
    outcome: "adopted",
    candidate: evaluation.name,
    gains: evaluation.gains,
    ...(evaluation.rowBlock === undefined ? {} : { derivedRowBlock: evaluation.rowBlock }),
    rowGains: evaluation.rowGains,
  };
};

/** The thread target: kept (the field holds one number, the records per-shape heights). */
const threadTargetDecision = (
  records: readonly Labelled[],
  defaults: LinearGemvChoices,
): LinearGemvDecision => {
  // The quick and explore sets measure no row-block height
  const rowBlocks = records.filter(
    ({ source }) => source.candidateSet === "full" || source.candidateSet === "row-blocks",
  );
  const target = String(defaults.rowsThreadTarget);
  return {
    class: "identical",
    field: "rowsThreadTarget",
    shape: "",
    cases: [],
    defaultChoice: target,
    path: "two-record",
    derivedChoice: target,
    outcome: "kept",
    reason:
      rowBlocks.length === 0
        ? "only records of the full and row-blocks sets measure row blocks (M ≥ 2), and none is given"
        : "the full and row-blocks sets measure the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it",
    gains: [],
    notes: [],
  };
};

/** A record's label in the verdict lines, by its path (the path itself for one not among them). */
type LabelOf = (path: string) => string;

const labelsOf =
  (paths: readonly string[]): LabelOf =>
  (path) => {
    const index = paths.indexOf(path);
    return index < 0 ? path : `[${index + 1}]`;
  };

const gainsText = (gains: readonly LinearGemvGain[], labelOf: LabelOf): string => {
  const caseCount = new Set(gains.map(({ caseId }) => caseId)).size;
  return gains
    .map(
      ({ record, caseId, gain }) =>
        `${formatRatio(gain)} ${labelOf(record)}${caseCount > 1 ? ` ${caseId}` : ""}`,
    )
    .join(", ");
};

const rowGainsText = (gains: readonly LinearGemvRowGain[], labelOf: LabelOf): string =>
  gains.map(({ record, m, gain }) => `${formatRatio(gain)} ${labelOf(record)} M=${m}`).join(", ");

/** The break-even margins of the candidate a one-record decision names, at M = 1 (`×1.143 [1]`). */
const breakEvenText = (decision: LinearGemvDecision, labelOf: LabelOf): string => {
  const readings = (decision.evidence ?? []).flatMap(
    (one) => one.candidates.find(({ candidate }) => candidate === decision.candidate)?.m1 ?? [],
  );
  return readings.length === 0
    ? ""
    : `; break-even ${readings.map(({ breakEven, record }) => `${formatRatio(breakEven)} ${labelOf(record)}`).join(", ")}`;
};

const decisionLine = (decision: LinearGemvDecision, labelOf: LabelOf): string => {
  const shape = decision.field === "rowsThreadTarget" ? "rowsThreadTarget" : decision.shape;
  const cases =
    decision.cases.length === 0
      ? ""
      : ` (${decision.cases.length === 1 ? "case" : "cases"} ${decision.cases.join(", ")})`;
  const gains =
    decision.candidate === undefined || decision.gains.length === 0
      ? ""
      : decision.outcome === "adopted"
        ? ` (${gainsText(decision.gains, labelOf)}${breakEvenText(decision, labelOf)})`
        : `; ${decision.candidate}: ${gainsText(decision.gains, labelOf)}${breakEvenText(decision, labelOf)}`;
  // A parallel row is named with its row block, the form it takes at M ≥ 2
  const blocked = (choice: string, rowBlock: LinearGemvParallelRows | undefined): string =>
    rowBlock === undefined ? choice : `${choice} with row blocks of ${rowBlock}`;
  const defaultText = blocked(decision.defaultChoice, decision.defaultRowBlock);
  // Under the rule of one steady record, and where no two records are evidence, the table's choice
  const from =
    decision.tableChoice === undefined
      ? defaultText
      : blocked(decision.tableChoice, decision.tableRowBlock);
  const fromText = from === defaultText ? from : `${from} (the default's: ${defaultText})`;
  const rowGains =
    decision.rowGains === undefined || decision.rowGains.length === 0
      ? ""
      : `; at the other row counts: ${rowGainsText(decision.rowGains, labelOf)}`;
  const rule = decision.path === "one-record" ? "one steady record: " : "";
  const body =
    decision.outcome === "adopted"
      ? `${rule}${fromText} -> ${blocked(decision.derivedChoice, decision.derivedRowBlock)}${gains}${rowGains}`
      : `${rule}keeps ${fromText} (${decision.reason ?? "?"})${gains}`;
  return [`  - ${shape}${cases}: ${body}`, ...decision.notes.map((note) => `    - ${note}`)].join(
    "\n",
  );
};

const classLines = (
  decisions: readonly LinearGemvDecision[],
  labelOf: LabelOf,
  head: string,
): string[] => {
  const adopted = decisions.filter(({ outcome }) => outcome === "adopted").length;
  return [
    `- ${head}: ${adopted} adopted, ${decisions.length - adopted} kept`,
    ...decisions.flatMap((decision) => decisionLine(decision, labelOf).split("\n")),
  ];
};

/** How much of each record stands for the current runtime. */
const recordGroups = (record: Labelled): LinearGemvRecordGroups => {
  let current = 0;
  let staleRows = 0;
  const withoutReference = { exact: 0, reorder: 0 };
  for (const state of record.states.values()) {
    if (state.kind !== "current") continue;
    current += 1;
    for (const row of state.group.rows.values()) {
      if (rowStaleness(row, state.kernels) !== undefined) staleRows += 1;
    }
    for (const level of REFERENCE_LEVELS) {
      if (state.references[level].kind === "unusable") withoutReference[level] += 1;
    }
  }
  return { current, stale: record.states.size - current, withoutReference, staleRows };
};

/** The groups of the records whose case is not current, in record order. */
const staleGroups = (records: readonly Labelled[]): LinearGemvStaleGroup[] =>
  records.flatMap(({ source, states }) =>
    [...states.values()].flatMap((state) =>
      state.kind === "stale"
        ? [
            {
              record: source.path,
              caseId: state.group.caseId,
              m: state.group.m,
              reason: state.reason,
            },
          ]
        : [],
    ),
  );

type ProvenanceRecord = LinearGemvDerivation["provenance"]["records"][number];

/** A record's class as a line reads it. */
const steadinessText = ({ steadiness, steady }: ProvenanceRecord): string => {
  const { groups, unsteadyGroups, share } = steadiness;
  const counted = `${unsteadyGroups} of ${groups} groups beyond ${formatRatio(GEMV_STEADY_GROUP_SPREAD)}${share === undefined ? "" : `, ${(share * 100).toFixed(1)} %`}`;
  const read = steady ? "read by the rule of one steady record" : "read by the rule of two records";
  return steadiness.class === "unclassified"
    ? `no class (${groups} groups with a baseline spread, fewer than ${GEMV_STEADY_MIN_GROUPS}): ${read}`
    : `${steadiness.class} (${counted}): ${read}${steadiness.class === "steady" && !steady ? " (its timestamps are not in ns)" : ""}`;
};

/** One record's line. */
const recordLine = (record: ProvenanceRecord, label: string): string => {
  const { current, stale, withoutReference, staleRows } = record.groups;
  return `  - ${label} ${record.path} (${record.date}, set ${record.candidateSet}): ${current + stale} groups — ${current} current, ${stale} not current (their case is not the current one); of the current, ${withoutReference.exact} without a usable reference at exact and ${withoutReference.reorder} at reorder; ${staleRows} rows of current groups not current; ${steadinessText(record)}`;
};

const verdictLinesOf = (
  decisions: readonly LinearGemvDecision[],
  records: readonly ProvenanceRecord[],
  margin: number,
): string[] => {
  const labelOf = labelsOf(records.map(({ path }) => path));
  return [
    `- linearGemv (the GEMV choices of linear), from ${records.length} GEMV sweep records (the rule of one steady record at the margin ${formatRatio(margin)}):`,
    ...records.map((record, index) => recordLine(record, `[${index + 1}]`)),
    ...classLines(
      decisions.filter(({ field }) => field === "sequential.shapes"),
      labelOf,
      "identical class (sequential.shapes, M = 1)",
    ),
    ...classLines(
      decisions.filter(({ field }) => field === "parallelShapes"),
      labelOf,
      "reorder class (parallelShapes, M = 1 to 8)",
    ),
    ...decisions
      .filter(({ field }) => field === "rowsThreadTarget")
      .flatMap((decision) =>
        decisionLine(decision, labelOf)
          .split("\n")
          .map((line) => line.slice(2)),
      ),
  ];
};

/** The rule's numbers at a margin. */
const ruleOf = (margin: number): LinearGemvRule => ({
  margin,
  steadyMinGroups: GEMV_STEADY_MIN_GROUPS,
  steadyMaxShare: GEMV_STEADY_MAX_SHARE,
  steadyGroupSpread: GEMV_STEADY_GROUP_SPREAD,
  minPassMs: GEMV_MIN_PASS_MS,
  rounds: ROUNDS,
  minSpeedup: GEMV_MIN_SPEEDUP,
  referenceRepeat: GEMV_REFERENCE_REPEAT,
});

/**
 * The field the decisions give (the table's choices with each decision's derived choice in place),
 * checked as a profile's; absent when it is the default's choices (a table without the field runs
 * with them).
 */
const fieldOf = (
  table: LinearGemvChoices,
  defaults: LinearGemvChoices,
  decisions: readonly LinearGemvDecision[],
): LinearGemvChoices | undefined => {
  const choices = applyDecisions(table, defaults, decisions);
  try {
    assertGeometryProfile({ ...DEFAULT_GEOMETRY_PROFILE, linearGemv: choices });
  } catch (cause) {
    if (cause instanceof CodegenError) {
      throw new Error(`The derived GEMV choices fail the profile's check — ${cause.message}`, {
        cause,
      });
    }
    throw cause;
  }
  return canonicalJsonText(choices) === canonicalJsonText(defaults) ? undefined : choices;
};

/**
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) from GEMV sweep records of one
 * adapter, by the rules of the module doc. A shape a steady record measured is decided by the rule
 * of one steady record at `options.margin` (×1.065 by default): a candidate whose gain over the
 * default's choice in the same group clears margin × its gap × the reference's gap in every steady
 * record that times it, its check held in every current group, its row not slower than the default's
 * choice at the other row counts in those records — and where the table holds another choice, one
 * that beats that choice by the same bar. Any other shape is decided by the rule of two records:
 * ×1.10 in every group of two records or more against the reference's repeated fastest time. The
 * field is the table's choices with the derived ones in place. Groups whose case is not current and
 * current groups without a usable reference are reported, not refused; a record that no group could
 * stand for throws with the reason (fail loudly).
 */
export const deriveLinearGemvChoices = (
  inputs: readonly GemvSweepInput[],
  options: DeriveLinearGemvOptions,
): LinearGemvDerivation => {
  const margin = gemvMarginOf(options.margin);
  const sources = readRecords(inputs, options.table);
  const kernelsOf = currentKernels(options.table);
  const records = labelled(sources, kernelsOf, options.table.id);
  const defaults = DEFAULT_GEOMETRY_PROFILE.linearGemv;
  const table = options.table.linearGemv ?? defaults;
  const context: DerivationContext = {
    records,
    steady: records.filter(({ source }) => readsSteady(source)),
    kernelsOf,
    defaults,
    table,
    margin,
  };
  const cases = GEMV_CASES.filter((gemvCase) =>
    sources.some((source) => source.cases.has(gemvCase.id)),
  );
  // Sequential keys in the order of their first case
  const keys = new Map<string, GemvCase[]>();
  for (const gemvCase of cases) {
    const key = sequentialKey(gemvCase);
    keys.set(key, [...(keys.get(key) ?? []), gemvCase]);
  }
  const decisions = [
    ...[...keys.values()].map((shared) => sequentialDecision(context, shared)),
    ...cases.map((gemvCase) => parallelDecision(context, gemvCase)),
    threadTargetDecision(records, defaults),
  ];
  const linearGemv = fieldOf(table, defaults, decisions);
  const [first] = sources;
  const provenanceRecords = records.map((record): ProvenanceRecord => ({
    path: record.source.path,
    sha256: record.source.sha256,
    date: record.source.date,
    candidateSet: record.source.candidateSet,
    ...(record.source.userAgent === undefined ? {} : { userAgent: record.source.userAgent }),
    groups: recordGroups(record),
    steadiness: record.source.steadiness,
    steady: readsSteady(record.source),
  }));
  return {
    ...(linearGemv === undefined ? {} : { linearGemv }),
    defaults,
    tableChoices: table,
    decisions,
    stale: staleGroups(records),
    verdicts: verdictLinesOf(decisions, provenanceRecords, margin),
    rule: ruleOf(margin),
    provenance: {
      adapter: first.adapter,
      caseTable: gemvCaseTableId(),
      records: provenanceRecords,
    },
  };
};

/** A decision whose derived choice another margin changes. */
export type LinearGemvChangedDecision = {
  readonly field: LinearGemvDecision["field"];
  readonly shape: string;
  readonly cases: readonly string[];
  /** The choice and row block under the derivation's margin. */
  readonly from: { readonly choice: string; readonly rowBlock?: LinearGemvParallelRows };
  /** Under the other margin. */
  readonly to: { readonly choice: string; readonly rowBlock?: LinearGemvParallelRows };
};

/** The return value of {@link redecideLinearGemvChoices}. */
export type LinearGemvRedecision = {
  /** The derivation as it is under `margin`: its decisions, field, lines and rule. */
  readonly derivation: LinearGemvDerivation;
  /** The decisions whose derived choice or row block differs from the given derivation's. */
  readonly changed: readonly LinearGemvChangedDecision[];
};

/**
 * Decides a derivation's output again under another margin, from the output alone (no record):
 * every decision of the rule of one steady record is decided again from the evidence it carries by
 * the same function the derivation calls (gemv-one-record.ts's `decideOneRecord`), every other
 * decision (the rule of two records, the packed rows, the thread target) stays, the field is built
 * again from the table's choices the derivation started from, and the decisions that change are
 * listed. Pure. The decisions' notes (the subgroup form) stay those of the derivation's margin.
 */
export const redecideLinearGemvChoices = (
  derivation: LinearGemvDerivation,
  margin: number,
): LinearGemvRedecision => {
  const checked = gemvMarginOf(margin);
  const decisions = derivation.decisions.map((decision): LinearGemvDecision => {
    const { evidence } = decision;
    if (decision.path !== "one-record" || evidence === undefined) return decision;
    const {
      derivedChoice: _choice,
      derivedRowBlock: _rowBlock,
      outcome: _outcome,
      reason: _reason,
      candidate: _candidate,
      gains: _gains,
      rowGains: _rowGains,
      path: _path,
      evidence: _evidence,
      tableChoice: _table,
      tableRowBlock: _tableRowBlock,
      ...head
    } = decision;
    if (decision.field === "sequential.shapes") {
      return sequentialOfEvidence(head, evidence, checked);
    }
    const [one] = evidence;
    if (one === undefined) throw new Error(`The decision of ${decision.shape} carries no evidence`);
    return parallelOfEvidence(head, one, checked);
  });
  const changed = derivation.decisions.flatMap((before, index): LinearGemvChangedDecision[] => {
    const after = decisions[index];
    if (
      after === undefined ||
      (after.derivedChoice === before.derivedChoice &&
        after.derivedRowBlock === before.derivedRowBlock)
    ) {
      return [];
    }
    const side = (decision: LinearGemvDecision) => ({
      choice: decision.derivedChoice,
      ...(decision.derivedRowBlock === undefined ? {} : { rowBlock: decision.derivedRowBlock }),
    });
    return [
      {
        field: before.field,
        shape: before.shape,
        cases: before.cases,
        from: side(before),
        to: side(after),
      },
    ];
  });
  const linearGemv = fieldOf(derivation.tableChoices, derivation.defaults, decisions);
  const { linearGemv: _field, ...rest } = derivation;
  return {
    derivation: {
      ...rest,
      ...(linearGemv === undefined ? {} : { linearGemv }),
      decisions,
      verdicts: verdictLinesOf(decisions, derivation.provenance.records, checked),
      rule: ruleOf(checked),
    },
    changed,
  };
};

/** A case and row count the records hold too little of, and the candidates missing there. */
export type GemvMissingMeasurement = {
  readonly caseId: string;
  readonly m: number;
  /** In the sweep's measuring order. */
  readonly candidates: readonly string[];
};

/** The return value of {@link planGemvMeasurements}. */
export type GemvMeasurementPlanning = {
  /** The device facts the records were measured under, which decide the candidates. */
  readonly gates: GemvGates;
  /** The (case, row count) pairs of the `full` set the plan read. */
  readonly groups: number;
  /**
   * What records of any class fill (the decisions of the rule of two records: two records needed),
   * in the case table's order, then by row count — what a batch of narrowed steps measures.
   */
  readonly missing: readonly GemvMissingMeasurement[];
  /**
   * What only a steady record fills, in the same order: the candidates of decisions that the rule of
   * one steady record takes (a steady record timed the case at M = 1) and that no steady record is
   * evidence for. A narrowed step's record of a few groups has no class, so no batch step fills
   * these: a steady record that times the named kernels does (a run of the `full` or the `explore`
   * set on the device).
   */
  readonly missingSteady: readonly GemvMissingMeasurement[];
  /** The groups of the records that are not current. */
  readonly stale: readonly LinearGemvStaleGroup[];
};

/** Options of {@link planGemvMeasurements}. */
export type PlanGemvMeasurementsOptions = {
  /** The table the records will derive for: as {@link DeriveLinearGemvOptions.table}, checked alike. */
  readonly table?: GeometryProfile;
};

/**
 * What the records lack for {@link deriveLinearGemvChoices} to be complete under the current
 * runtime: per case and row count of the `full` set, the candidates the derivation reads there (the
 * `winner` and `no-regression` roles of gemv-candidates.ts's `gemvGroupRole`, on the records' gates)
 * that the records could not be evidence for by the rule the derivation will take for the
 * candidate's decision. Where a steady record timed the case at M = 1 with a usable reference for
 * the class (the identical class: a case of its sequential key), the decision is the rule of one
 * steady record's, which reads the steady records alone: a candidate no steady record could be
 * evidence for (a usable reference in its group) is in `missingSteady`, and records of no class
 * cannot fill it. Elsewhere the rule of two records decides: a candidate fewer than two records
 * could be evidence for (a current group whose reference for the candidate's class is usable and
 * whose time is repeated) is in `missing`. Either way it is a current
 * row of the candidate — timed, screened out by the quick look (a screened row is a current look
 * that the candidate did not come near enough to matter; measuring it again would screen it again),
 * or skipped because the record's M = 1 group of the case already left it unadoptable
 * (gemv-candidates.ts's `gemvCarriedFromM1`; a run that measures M = 1 again would skip it again). A
 * row that failed is missing, and so is a skipped or screened row of an `explore` record: that set
 * skips by a narrower rule of its own (`gemvExploreCarriedFromM1`) and screens at a narrower margin
 * (measurement.ts's `EXPLORE_QUICK_LOOK_MARGIN`), and a run of the `full` set would measure, or
 * time, the candidate — so of an explore record only the rows it timed count. A case and row count
 * no record has a current group of misses every
 * candidate, and so does one whose reference has no repeated time: every measurement of a group
 * takes the references, so measuring the candidates measures the reference again. A candidate whose
 * reference the sweep does not measure (a row with packed activations) is not listed: no
 * measurement makes it evidence. Measures nothing. At least one record; with none there is nothing
 * to plan from — measure everything.
 */
export const planGemvMeasurements = (
  inputs: readonly GemvSweepInput[],
  options: PlanGemvMeasurementsOptions = {},
): GemvMeasurementPlanning => {
  const sources = readRecords(inputs, options.table);
  const [first] = sources;
  const gates = first.gates;
  for (const source of sources) {
    if (
      source.gates.subgroups !== gates.subgroups ||
      source.gates.maxComputeWorkgroupsPerDimension !== gates.maxComputeWorkgroupsPerDimension
    ) {
      throw new Error(
        `${source.path}: its gates (subgroups ${String(source.gates.subgroups)}, grid limit ${source.gates.maxComputeWorkgroupsPerDimension}) differ from ${first.path}'s (records of one device configuration only)`,
      );
    }
  }
  // Without a table to derive for: the embedded table the adapter of the records selects
  const table = options.table ?? selectGeometryProfile(first.adapter);
  const kernelsOf = currentKernels(table);
  const records = labelled(sources, kernelsOf, table.id);
  const steady = { steady: records.filter(({ source }) => readsSteady(source)) };
  /**
   * Whether the derivation decides a case's class by the rule of one steady record: the identical
   * class by its sequential key (a steady record measured one of the key's cases at M = 1), the
   * reorder class by its parallel row (the case itself) — as {@link deriveLinearGemvChoices} does.
   */
  const oneRecordPath = (gemvCase: GemvCase, candidateClass: GemvCandidateClass): boolean =>
    candidateClass === "identical"
      ? GEMV_CASES.some(
          (other) =>
            sequentialKey(other) === sequentialKey(gemvCase) &&
            measuredBySteady(steady, other, "identical"),
        )
      : measuredBySteady(steady, gemvCase, "reorder");
  const nameOf = ({ name }: { readonly name: string }): string => name;
  const missing: GemvMissingMeasurement[] = [];
  const missingSteady: GemvMissingMeasurement[] = [];
  let groups = 0;
  for (const gemvCase of GEMV_CASES) {
    for (const m of gemvRowCountsIn(gemvCase, "full")) {
      groups += 1;
      // The default's choice at reorder is a reference: every measurement of the group takes it
      const reorder = referenceName(gemvCase, m, "reorder");
      const kernels = kernelsOf(gemvCase, m);
      const read = gemvCandidates(gemvCase, m, "full", gates).filter(
        (candidate) =>
          candidate.name !== reorder &&
          gemvGroupRole(m, candidate.class) !== "unread" &&
          kernels.has(classReference(gemvCase, m, candidate.class)),
      );
      /**
       * The records that could be evidence for a candidate here: under the rule of two records a
       * group whose reference's time is repeated, under the rule of one steady record a steady
       * record's group whose reference is usable (its own time).
       */
      const covering = (
        { name, class: candidateClass }: (typeof read)[number],
        steady: boolean,
      ): number =>
        records.filter(({ source, states }) => {
          const state = states.get(groupKey(gemvCase.id, m));
          if (state?.kind !== "current") return false;
          if (
            steady
              ? !readsSteady(source) || !referenceUsable(state, candidateClass)
              : state.references[levelOf(candidateClass)].kind !== "timed"
          ) {
            return false;
          }
          const row = state.group.rows.get(name);
          return (
            row !== undefined &&
            row.error === undefined &&
            // The explore set skips by its own narrower rule and screens at a narrower margin: the
            // full set would measure, or time, the row
            !(
              source.candidateSet === "explore" &&
              (row.skipped === true || row.screened === true)
            ) &&
            rowStaleness(row, state.kernels) === undefined
          );
        }).length;
      // By the rule the derivation will take for the candidate's decision: where a steady record
      // timed the case at M = 1 only steady records count, and only one fills the gap
      const lacking = read.filter((candidate) =>
        oneRecordPath(gemvCase, candidate.class)
          ? covering(candidate, true) === 0
          : covering(candidate, false) < MIN_RECORDS,
      );
      const add = (to: GemvMissingMeasurement[], names: readonly string[]): void => {
        if (names.length > 0) to.push({ caseId: gemvCase.id, m, candidates: names });
      };
      add(
        missing,
        lacking.filter((candidate) => !oneRecordPath(gemvCase, candidate.class)).map(nameOf),
      );
      add(
        missingSteady,
        lacking.filter((candidate) => oneRecordPath(gemvCase, candidate.class)).map(nameOf),
      );
    }
  }
  return { gates, groups, missing, missingSteady, stale: staleGroups(records) };
};
