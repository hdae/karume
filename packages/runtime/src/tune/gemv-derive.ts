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
 *   the quick look, not skipped because the same record's M = 1 group already left it unadoptable)
 *   with its check held. A row that failed or whose check failed in a current group, evidence or
 *   not, rules the candidate out: correctness does not depend on timing.
 *
 * Groups whose case is not current are reported (the verdict lines, the provenance's counts and
 * {@link LinearGemvDerivation.stale}), and so are the current groups without a usable reference;
 * no record is refused for them. A record is refused only for what no group of it could stand for:
 * another format (no older one is read), wall-clock or quantized timing, another adapter, the same
 * record twice.
 *
 * ## The adoption rule (the maintainer, 2026-11-17)
 *
 * The timing of these small dispatches moves between runs, so one record is not evidence:
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
  GEMV_SWEEP_FORMAT,
  type GemvSweepChoice,
  type GemvSweepFamilyChoice,
} from "./gemv-report.ts";

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
  if (timing.unit !== "ns" && timing.unit !== "deno-raw-tick") {
    throw new Error(
      `${where}: gpuTiming.unit is ${JSON.stringify(timing.unit)} (must be ns or deno-raw-tick — a wall-clock record is not read)`,
    );
  }
  if (timing.quantized !== false) {
    throw new Error(
      `${where}: gpuTiming.quantized is ${JSON.stringify(timing.quantized)} (quantized timestamps are not read)`,
    );
  }
  const candidateSet = objectAt(record.settings, `${where} settings`).candidateSet;
  if (typeof candidateSet !== "string" || !isGemvCandidateSet(candidateSet)) {
    throw new Error(
      `${where}: settings.candidateSet ${JSON.stringify(candidateSet)} is none of quick, full, row-blocks`,
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
  record.summaries.forEach((value, index) => {
    const at = `${where} summaries[${index}]`;
    const summary = objectAt(value, at);
    const caseId = stringAt(summary, "caseId", at);
    const { m } = summary;
    if (!isPositiveInteger(m)) throw new Error(`${at}: m is not a positive integer`);
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
  reference: { readonly name: string; readonly level: ReferenceLevel },
  where: string,
  tableId: string,
): string | undefined => {
  const { name, level } = reference;
  const row = group.rows.get(name);
  if (row === undefined) return `the reference ${name} is not measured ${where}`;
  if (row.error !== undefined) return `the reference ${name} failed ${where} (${row.error})`;
  if (row.perDispatch === undefined) return `the reference ${name} has no time ${where}`;
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
  // A candidate of the identical class shows its words equal to the record's baseline. Against
  // another kernel that says something only when that kernel's own words equal the baseline's
  if (level === "exact" && !row.isBaseline && row.identicalToBaseline !== true) {
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
        const referenceAt = (level: ReferenceLevel): GroupReference => {
          const name = referenceName(gemvCase, group.m, level);
          if (!kernels.has(name)) {
            return {
              kind: "unmeasured",
              reason: `the default's choice at ${level} (${name}) is a kernel the sweep does not measure`,
            };
          }
          const where = `in ${label} at M=${group.m}`;
          const refusal = referenceRefusal(source, group, kernels, { name, level }, where, tableId);
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
            references: { exact: referenceAt("exact"), reorder: referenceAt("reorder") },
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
  if (row.error !== undefined) {
    return { kind: "ruled-out", reason: `failed in ${label} at M=${m} (${row.error})` };
  }
  if (row.screened === true) {
    return { kind: "none", reason: `screened out by the quick look in ${label} at M=${m}` };
  }
  if (row.skipped === true) {
    return {
      kind: "none",
      reason: `skipped in ${label} at M=${m} (its M=1 group there left it unadoptable)`,
    };
  }
  const reference = state.references[levelOf(candidateClass)];
  // A failed check keeps its gain: the reader sees how fast the candidate was
  const gain =
    reference.kind !== "timed" || row.perDispatch === undefined
      ? undefined
      : reference.time / row.perDispatch;
  const withGain = gain === undefined ? {} : { gain };
  // Correctness does not depend on the reference: a failed check rules out in any current group.
  // The record's baseline writes its own words: it is a candidate where the default moved off it
  if (candidateClass === "identical" && !row.isBaseline && row.identicalToBaseline !== true) {
    return {
      kind: "ruled-out",
      reason: `its words differ from the baseline's in ${label} at M=${m}`,
      ...withGain,
    };
  }
  if (candidateClass === "reorder" && row.withinBand !== true) {
    return { kind: "ruled-out", reason: `outside the band in ${label} at M=${m}`, ...withGain };
  }
  if (reference.kind !== "timed") return { kind: "none", reason: reference.reason };
  if (gain === undefined) return { kind: "none", reason: `has no time in ${label} at M=${m}` };
  return { kind: "evidence", gain };
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
  /** The derived choice (the default's when kept). */
  readonly derivedChoice: string;
  readonly outcome: "adopted" | "kept";
  /** Why the default's choice stays (`outcome` kept). */
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
  /** Per class and shape, in the case table's order; the thread target last. */
  readonly decisions: readonly LinearGemvDecision[];
  /** The groups no evidence comes from because their case is not the current runtime's. */
  readonly stale: readonly LinearGemvStaleGroup[];
  /** The decisions as lines (a generated table's header, a generator's dry run). */
  readonly verdicts: readonly string[];
  /**
   * The rule's threshold, and the factor within which a second record repeats a reference's fastest
   * time for that time to be read.
   */
  readonly rule: { readonly minSpeedup: number; readonly referenceRepeat: number };
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
    }[];
  };
};

/** Options of {@link deriveLinearGemvChoices}. */
export type DeriveLinearGemvOptions = {
  /**
   * The table the field is derived for: the records must be of an adapter its match selects and,
   * when it has provenance, of its adapter (all four fields — a table is one adapter's answer).
   */
  readonly table: GeometryProfile;
};

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

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
    for (const field of ADAPTER_FIELDS) {
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

/** The outcome of one case and class: the adopted candidate, or why the default stays. */
type CaseOutcome =
  | { readonly kind: "adopted"; readonly evaluation: Evaluation }
  | { readonly kind: "kept"; readonly reason: string; readonly nearest?: Evaluation };

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
  if (nearest === undefined) return { kind: "kept", reason: "no candidate was measured" };
  return {
    kind: "kept",
    reason: `no candidate is ${formatRatio(GEMV_MIN_SPEEDUP)} faster in the groups of ${MIN_RECORDS} or more records that are evidence, with its check held (nearest ${nearest.name}: ${nearest.failure ?? "?"})`,
    nearest,
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

/** The identical class of one case at M = 1: the column form's variants against the baseline. */
const identicalJudgement = (gemvCase: GemvCase): CaseJudgement => {
  const baseline = referenceName(gemvCase, 1, levelOf("identical"));
  return {
    gemvCase,
    class: "identical",
    reference: () => baseline,
    takes: (row) => row.class === "identical" && row.choice.id === "gemv-m1",
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
  reference: (m) => referenceName(gemvCase, m, levelOf("reorder")),
  takes: (row) =>
    row.class === "reorder" &&
    ids.includes(row.choice.id) &&
    familyOf(row.choice)?.parallelRows === undefined,
  others: noRegressionRows(gemvCase),
  rowBlocks,
});

const gainsText = (gains: readonly LinearGemvGain[], records: readonly Labelled[]): string => {
  const caseCount = new Set(gains.map(({ caseId }) => caseId)).size;
  return gains
    .map(({ record, caseId, gain }) => {
      const label = records.find(({ source }) => source.path === record)?.label ?? record;
      return `${formatRatio(gain)} ${label}${caseCount > 1 ? ` ${caseId}` : ""}`;
    })
    .join(", ");
};

const rowGainsText = (gains: readonly LinearGemvRowGain[], records: readonly Labelled[]): string =>
  gains
    .map(({ record, m, gain }) => {
      const label = records.find(({ source }) => source.path === record)?.label ?? record;
      return `${formatRatio(gain)} ${label} M=${m}`;
    })
    .join(", ");

/** The sequential key's decision from its cases' (they must agree). */
const sequentialDecision = (
  records: readonly Labelled[],
  cases: readonly GemvCase[],
  defaults: LinearGemvChoices,
  kernelsOf: CurrentKernels,
): { readonly decision: LinearGemvDecision; readonly variant?: LinearGemvVariant } => {
  const [first] = cases;
  const defaultChoice = variantName(
    defaults.sequential.shapes.find((row) => sameSequentialShape(row, first))?.variant ??
      defaults.sequential.variant,
  );
  const outcomes = cases.map((gemvCase) => ({
    gemvCase,
    outcome: judgeCase(records, identicalJudgement(gemvCase), kernelsOf),
  }));
  const head = {
    class: "identical",
    field: "sequential.shapes",
    shape: sequentialKey(first),
    cases: cases.map(({ id }) => id),
    defaultChoice,
    notes: [],
  } as const;
  const adopted = outcomes.flatMap(({ outcome }) =>
    outcome.kind === "adopted" ? [outcome.evaluation] : [],
  );
  const names = new Set(adopted.map(({ name }) => name));
  if (adopted.length === outcomes.length && names.size === 1) {
    const [chosen] = adopted;
    return {
      decision: {
        ...head,
        derivedChoice: chosen.name,
        outcome: "adopted",
        candidate: chosen.name,
        gains: adopted.flatMap(({ gains }) => gains),
      },
      variant: familyChoiceOf(chosen).variant,
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
  return {
    decision: {
      ...head,
      derivedChoice: defaultChoice,
      outcome: "kept",
      reason,
      ...(shown === undefined ? {} : { candidate: shown.name }),
      gains: shown === undefined ? [] : shown.gains,
    },
  };
};

/** The parallel row's decision of one case, and the row it writes when adopted. */
const parallelDecision = (
  records: readonly Labelled[],
  gemvCase: GemvCase,
  defaults: LinearGemvChoices,
  kernelsOf: CurrentKernels,
): { readonly decision: LinearGemvDecision; readonly row?: LinearGemvParallelShape } => {
  const defaultRow = defaults.parallelShapes.find((row) => sameParallelShape(row, gemvCase));
  const defaultChoice =
    defaultRow === undefined
      ? `no row (${gemvBaseline(gemvCase, 1).name})`
      : parallelName(defaultRow);
  const subgroup = judgeCase(
    records,
    reorderJudgement(gemvCase, ["gemv-parallel-subgroup32"], false),
    kernelsOf,
  );
  const notes =
    subgroup.kind === "adopted"
      ? [
          `${subgroup.evaluation.name} meets the rule too (${gainsText(subgroup.evaluation.gains, records)}): a Session option, not a row`,
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
      decision: {
        ...head,
        derivedChoice: defaultChoice,
        outcome: "kept",
        reason: "the default takes packed activations, whose kernels the sweep does not measure",
        gains: [],
        ...sameRowBlock,
      },
    };
  }
  const judgement = reorderJudgement(gemvCase, ["gemv-parallel", "gemv-parallel-staged"], true);
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
      const { rowBlock: _rowBlock, ...bare } = defaultRow;
      return {
        decision: {
          ...head,
          derivedChoice: defaultChoice,
          outcome: "adopted",
          gains: [],
          ...(other.rowBlock === undefined ? {} : { derivedRowBlock: other.rowBlock }),
          rowGains: readOthers(records, judgement, (m) =>
            rowKernelAt(defaultChoice, defaultRow.lanes, other.rowBlock, m),
          ).gains,
        },
        row: { ...bare, ...(other.rowBlock === undefined ? {} : { rowBlock: other.rowBlock }) },
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
    return {
      decision: {
        ...head,
        notes: [...notes, ...rowBlockNote],
        derivedChoice: defaultChoice,
        outcome: "kept",
        reason: outcome.reason,
        ...(outcome.nearest === undefined ? {} : { candidate: outcome.nearest.name }),
        gains: outcome.nearest?.gains ?? [],
        ...sameRowBlock,
      },
    };
  }
  const { evaluation } = outcome;
  const { lanes, id } = familyChoiceOf(evaluation);
  if (lanes === undefined) throw new Error(`${evaluation.name} of ${gemvCase.id} has no lanes`);
  const row: LinearGemvParallelShape = {
    storage: gemvCase.storage,
    n: gemvCase.n,
    k: gemvCase.k,
    ...(gemvCase.group === undefined ? {} : { group: gemvCase.group }),
    lanes,
    packedActivations: false,
    ...(id === "gemv-parallel-staged" ? { stagedActivations: true } : {}),
    ...(evaluation.rowBlock === undefined ? {} : { rowBlock: evaluation.rowBlock }),
  };
  return {
    decision: {
      ...head,
      derivedChoice: evaluation.name,
      outcome: "adopted",
      candidate: evaluation.name,
      gains: evaluation.gains,
      ...(evaluation.rowBlock === undefined ? {} : { derivedRowBlock: evaluation.rowBlock }),
      rowGains: evaluation.rowGains,
    },
    row,
  };
};

/** The thread target: kept (the field holds one number, the records per-shape heights). */
const threadTargetDecision = (
  records: readonly Labelled[],
  defaults: LinearGemvChoices,
): LinearGemvDecision => {
  const rowBlocks = records.filter(({ source }) => source.candidateSet !== "quick");
  const target = String(defaults.rowsThreadTarget);
  return {
    class: "identical",
    field: "rowsThreadTarget",
    shape: "",
    cases: [],
    defaultChoice: target,
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

const decisionLine = (decision: LinearGemvDecision, records: readonly Labelled[]): string => {
  const shape = decision.field === "rowsThreadTarget" ? "rowsThreadTarget" : decision.shape;
  const cases =
    decision.cases.length === 0
      ? ""
      : ` (${decision.cases.length === 1 ? "case" : "cases"} ${decision.cases.join(", ")})`;
  const gains =
    decision.candidate === undefined || decision.gains.length === 0
      ? ""
      : decision.outcome === "adopted"
        ? ` (${gainsText(decision.gains, records)})`
        : `; ${decision.candidate}: ${gainsText(decision.gains, records)}`;
  // A parallel row is named with its row block, the form it takes at M ≥ 2
  const blocked = (choice: string, rowBlock: LinearGemvParallelRows | undefined): string =>
    rowBlock === undefined ? choice : `${choice} with row blocks of ${rowBlock}`;
  const from = blocked(decision.defaultChoice, decision.defaultRowBlock);
  const rowGains =
    decision.rowGains === undefined || decision.rowGains.length === 0
      ? ""
      : `; at the other row counts: ${rowGainsText(decision.rowGains, records)}`;
  const body =
    decision.outcome === "adopted"
      ? `${from} -> ${blocked(decision.derivedChoice, decision.derivedRowBlock)}${gains}${rowGains}`
      : `keeps ${from} (${decision.reason ?? "?"})${gains}`;
  return [`  - ${shape}${cases}: ${body}`, ...decision.notes.map((note) => `    - ${note}`)].join(
    "\n",
  );
};

const classLines = (
  decisions: readonly LinearGemvDecision[],
  records: readonly Labelled[],
  head: string,
): string[] => {
  const adopted = decisions.filter(({ outcome }) => outcome === "adopted").length;
  return [
    `- ${head}: ${adopted} adopted, ${decisions.length - adopted} kept`,
    ...decisions.flatMap((decision) => decisionLine(decision, records).split("\n")),
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

/** One record's line. */
const recordLine = (record: Labelled): string => {
  const { source, label } = record;
  const { current, stale, withoutReference, staleRows } = recordGroups(record);
  return `  - ${label} ${source.path} (${source.date}, set ${source.candidateSet}): ${current + stale} groups — ${current} current, ${stale} not current (their case is not the current one); of the current, ${withoutReference.exact} without a usable reference at exact and ${withoutReference.reorder} at reorder; ${staleRows} rows of current groups not current`;
};

const verdictLinesOf = (
  decisions: readonly LinearGemvDecision[],
  records: readonly Labelled[],
): string[] => [
  `- linearGemv (the GEMV choices of linear), from ${records.length} GEMV sweep records:`,
  ...records.map(recordLine),
  ...classLines(
    decisions.filter(({ field }) => field === "sequential.shapes"),
    records,
    "identical class (sequential.shapes, M = 1)",
  ),
  ...classLines(
    decisions.filter(({ field }) => field === "parallelShapes"),
    records,
    "reorder class (parallelShapes, M = 1 to 8)",
  ),
  ...decisions
    .filter(({ field }) => field === "rowsThreadTarget")
    .flatMap((decision) =>
      decisionLine(decision, records)
        .split("\n")
        .map((line) => line.slice(2)),
    ),
];

/**
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) from GEMV sweep records of one
 * adapter, by the rule of the module doc: per shape and class, a candidate at least
 * {@link GEMV_MIN_SPEEDUP} faster than the default profile's choice in every group that is evidence
 * — groups of at least two records, read against the reference's repeated fastest time — with its
 * check held in every current group, and at the parallel row's other row counts not slower in any,
 * replaces that choice; otherwise the default's stays. Groups whose case is not current and current
 * groups without a usable reference are reported, not refused; a record that no group
 * could stand for throws with the reason (fail loudly).
 */
export const deriveLinearGemvChoices = (
  inputs: readonly GemvSweepInput[],
  options: DeriveLinearGemvOptions,
): LinearGemvDerivation => {
  const sources = readRecords(inputs, options.table);
  const kernelsOf = currentKernels(options.table);
  const records = labelled(sources, kernelsOf, options.table.id);
  const defaults = DEFAULT_GEOMETRY_PROFILE.linearGemv;
  const cases = GEMV_CASES.filter((gemvCase) =>
    sources.some((source) => source.cases.has(gemvCase.id)),
  );
  // Sequential keys in the order of their first case
  const keys = new Map<string, GemvCase[]>();
  for (const gemvCase of cases) {
    const key = sequentialKey(gemvCase);
    keys.set(key, [...(keys.get(key) ?? []), gemvCase]);
  }
  const sequential = [...keys.values()].map((shared) =>
    sequentialDecision(records, shared, defaults, kernelsOf),
  );
  const parallel = cases.map((gemvCase) =>
    parallelDecision(records, gemvCase, defaults, kernelsOf),
  );
  let shapes: LinearGemvShapeVariant[] = [...defaults.sequential.shapes];
  for (const { decision, variant } of sequential) {
    if (variant === undefined) continue;
    const [gemvCase] = keys.get(decision.shape) ?? [];
    if (gemvCase === undefined) throw new Error(`No case of the key ${decision.shape}`);
    const general = defaults.sequential.variant;
    const index = shapes.findIndex((row) => sameSequentialShape(row, gemvCase));
    const isGeneral = variant.cols === general.cols && variant.unroll === general.unroll;
    const row: LinearGemvShapeVariant = {
      storage: gemvCase.storage,
      n: gemvCase.n,
      k: gemvCase.k,
      variant,
    };
    if (isGeneral) shapes = shapes.filter((_, at) => at !== index);
    else if (index < 0) shapes = [...shapes, row];
    else shapes = shapes.map((entry, at) => (at === index ? row : entry));
  }
  let parallelShapes: LinearGemvParallelShape[] = [...defaults.parallelShapes];
  for (const { row } of parallel) {
    if (row === undefined) continue;
    // The selection takes the first row that matches, so the first one is replaced
    const index = parallelShapes.findIndex(
      (entry) =>
        entry.storage === row.storage &&
        entry.n === row.n &&
        entry.k === row.k &&
        entry.group === row.group,
    );
    parallelShapes =
      index < 0
        ? [...parallelShapes, row]
        : parallelShapes.map((entry, at) => (at === index ? row : entry));
  }
  const decisions = [
    ...sequential.map(({ decision }) => decision),
    ...parallel.map(({ decision }) => decision),
    threadTargetDecision(records, defaults),
  ];
  const changed = decisions.some(({ outcome }) => outcome === "adopted");
  const choices: LinearGemvChoices = {
    sequential: { variant: defaults.sequential.variant, shapes },
    rowsThreadTarget: defaults.rowsThreadTarget,
    parallelShapes,
  };
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
  const stale = staleGroups(records);
  const [first] = sources;
  return {
    ...(changed ? { linearGemv: choices } : {}),
    defaults,
    decisions,
    stale,
    verdicts: verdictLinesOf(decisions, records),
    rule: { minSpeedup: GEMV_MIN_SPEEDUP, referenceRepeat: GEMV_REFERENCE_REPEAT },
    provenance: {
      adapter: first.adapter,
      caseTable: gemvCaseTableId(),
      records: records.map((record) => ({
        path: record.source.path,
        sha256: record.source.sha256,
        date: record.source.date,
        candidateSet: record.source.candidateSet,
        ...(record.source.userAgent === undefined ? {} : { userAgent: record.source.userAgent }),
        groups: recordGroups(record),
      })),
    },
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
  /** In the case table's order, then by row count. */
  readonly missing: readonly GemvMissingMeasurement[];
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
 * that fewer than two records could be evidence for: a current group whose reference for the
 * candidate's class is usable and whose time is repeated (the module doc's rules), with a current
 * row of the candidate — timed, screened out by the quick look (a screened row is a current look
 * that the candidate did not come near enough to matter; measuring it again would screen it again),
 * or skipped because the record's M = 1 group of the case already left it unadoptable
 * (gemv-candidates.ts's `gemvCarriedFromM1`; a run that measures M = 1 again would skip it again). A
 * row that failed is missing. A case and row count no record has a current group of misses every
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
  const missing: GemvMissingMeasurement[] = [];
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
          kernels.has(referenceName(gemvCase, m, levelOf(candidate.class))),
      );
      const covered = ({ name, class: candidateClass }: (typeof read)[number]): number =>
        records.filter(({ states }) => {
          const state = states.get(groupKey(gemvCase.id, m));
          if (state?.kind !== "current") return false;
          if (state.references[levelOf(candidateClass)].kind !== "timed") return false;
          const row = state.group.rows.get(name);
          return (
            row !== undefined &&
            row.error === undefined &&
            rowStaleness(row, state.kernels) === undefined
          );
        }).length;
      const lacking = read
        .filter((candidate) => covered(candidate) < MIN_RECORDS)
        .map(({ name }) => name);
      if (lacking.length > 0) missing.push({ caseId: gemvCase.id, m, candidates: lacking });
    }
  }
  return { gates, groups, missing, stale: staleGroups(records) };
};
