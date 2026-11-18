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
 *   the row count in the case table), the default profile's choices its summary names are the
 *   current ones, and the rows of those choices — the baseline, and the choice at `reorder` where
 *   the sweep measures it — are current and timed. It is steady when its baseline spread is at most
 *   {@link GEMV_STEADY_SPREAD}.
 * - A group is evidence for a candidate when it is current and steady and the candidate's row is
 *   current and timed (not screened out by the quick look, not skipped because the same record's
 *   M = 1 group already left it unadoptable) with its check held. A row that failed or
 *   whose check failed in a current group, steady or not, rules the candidate out: correctness does
 *   not depend on heat.
 *
 * Records and groups that are not current are reported (the verdict lines, the provenance's counts
 * and {@link LinearGemvDerivation.stale}), never refused wholesale. A record is refused only for what
 * no group of it could stand for: another format (no older one is read), wall-clock or quantized
 * timing, another adapter, the same record twice.
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
 *   minimum of its rounds ÷ `reps`).
 * - The two classes are derived separately: the identical class writes the sequential variant's
 *   per-shape exceptions (`sequential.shapes`, from M = 1) and changes no number of any Session; the
 *   reorder class writes the parallel rows (`parallelShapes`: lanes and the staged flag, from the
 *   plain and staged candidates at M = 1) and changes the numbers of Sessions at `reorder`, inside
 *   the band. A parallel row serves M = 1 to 8, so at each other row count of that range (the
 *   `no-regression` role of gemv-candidates.ts's `gemvGroupRole` — M = 4 and 8 in the case table)
 *   groups of at least {@link MIN_RECORDS} records must be evidence for the candidate, and it may be
 *   slower than the default's choice in none of them: a path that was not measured is not moved by a
 *   table.
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
} from "../kernels/geometry-profile.ts";
import type {
  LinearGemvChoices,
  LinearGemvParallelLanes,
  LinearGemvParallelShape,
  LinearGemvShapeVariant,
  LinearGemvVariant,
} from "../kernels/linear-gemv.ts";
import { formatRatio } from "./derive.ts";
import {
  GEMV_MIN_SPEEDUP,
  GEMV_STEADY_SPREAD,
  type GemvCandidateClass,
  type GemvCandidateSet,
  gemvBaseline,
  gemvCandidates,
  type GemvGates,
  gemvGroupRole,
  gemvKernelsAt,
  gemvProfileChoice,
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
  readonly baselineSpread?: number;
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
  return {
    id,
    storage,
    ...(groupSize === undefined ? {} : { groupSize }),
    ...(lanes === undefined ? {} : { lanes }),
    variant,
    ...(rowsVariant === undefined ? {} : { rowsVariant }),
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
    const baselineSpread = optionalAt(summary, "baselineSpread", isPositiveNumber, at);
    groups.set(key, {
      caseId,
      m,
      profile: {
        exact: stringAt(profileAt, "exact", `${at} profile`),
        reorder: stringAt(profileAt, "reorder", `${at} profile`),
      },
      ...(baselineSpread === undefined ? {} : { baselineSpread }),
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
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: stringAt(record, "date", where),
    candidateSet,
    ...(userAgent === undefined ? {} : { userAgent }),
    adapter,
    gates: { subgroups, maxComputeWorkgroupsPerDimension: limit },
    cases,
    groups,
  };
};

/** The current runtime's kernels of a case at a row count, by candidate name (fingerprints). */
type CurrentKernels = (gemvCase: GemvCase, m: number) => ReadonlyMap<string, string>;

/** {@link CurrentKernels} with each (case, row count) built once. */
const currentKernels = (): CurrentKernels => {
  const cache = new Map<string, ReadonlyMap<string, string>>();
  return (gemvCase, m) => {
    const key = groupKey(gemvCase.id, m);
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    const kernels = new Map(
      gemvKernelsAt(gemvCase, m).map(({ name, choice }) => [
        name,
        gemvKernelId(gemvCase, m, choice),
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

/** What a group is to the current runtime. */
type GroupState =
  | {
      readonly kind: "current";
      readonly group: GroupObservation;
      readonly gemvCase: GemvCase;
      /** The current kernels of the group's case and row count. */
      readonly kernels: ReadonlyMap<string, string>;
      /** Why it is unsteady (undefined: steady). */
      readonly unsteady?: string;
    }
  | {
      readonly kind: "stale";
      readonly group: GroupObservation;
      /** What is not current: the case, the default's choices or a reference. */
      readonly cause: "case" | "choices" | "references";
      readonly reason: string;
    };

/** A group's state against the current runtime (the module doc's "current" and "steady"). */
const groupState = (
  source: GemvSweepSource,
  group: GroupObservation,
  kernelsOf: CurrentKernels,
): GroupState => {
  const { caseId, m } = group;
  const stale = (cause: "case" | "choices" | "references", reason: string): GroupState => ({
    kind: "stale",
    group,
    cause,
    reason,
  });
  const gemvCase = gemvCaseOf(caseId);
  if (gemvCase === undefined) return stale("case", `case ${caseId} is not in the case table`);
  if (source.cases.get(caseId) !== gemvCaseFingerprint(gemvCase)) {
    return stale("case", `case ${caseId} changed (its fingerprint differs)`);
  }
  if (!gemvCase.rows.includes(m)) {
    return stale("case", `case ${caseId} has no row count ${m} in the case table`);
  }
  const expected = {
    exact: gemvBaseline(gemvCase, m).name,
    reorder: gemvProfileChoice(gemvCase, m, "reorder").name,
  };
  for (const level of ["exact", "reorder"] as const) {
    if (group.profile[level] !== expected[level]) {
      return stale(
        "choices",
        `the default profile's choice at ${level} is ${group.profile[level]} in the record, ${expected[level]} now`,
      );
    }
  }
  const kernels = kernelsOf(gemvCase, m);
  // The choice at reorder is a reference where the sweep measures it (not a packed row)
  const references = [
    expected.exact,
    ...(expected.reorder !== expected.exact && kernels.has(expected.reorder)
      ? [expected.reorder]
      : []),
  ];
  for (const name of references) {
    const row = group.rows.get(name);
    if (row === undefined) return stale("references", `the reference ${name} has no row`);
    if (row.error !== undefined) {
      return stale("references", `the reference ${name} failed (${row.error})`);
    }
    if (row.perDispatch === undefined) {
      return stale("references", `the reference ${name} has no time`);
    }
    const staleness = rowStaleness(row, kernels);
    if (staleness !== undefined) {
      return stale("references", `the reference ${name} is not current (${staleness})`);
    }
  }
  const spread = group.baselineSpread;
  const unsteady =
    spread === undefined
      ? "no baseline spread"
      : spread > GEMV_STEADY_SPREAD
        ? `baseline spread ${formatRatio(spread)} > ${formatRatio(GEMV_STEADY_SPREAD)}`
        : undefined;
  return {
    kind: "current",
    group,
    gemvCase,
    kernels,
    ...(unsteady === undefined ? {} : { unsteady }),
  };
};

/** A record with its groups' states and its label in the verdict lines (its 1-based position). */
type Labelled = {
  readonly source: GemvSweepSource;
  readonly label: string;
  /** By {@link groupKey}. */
  readonly states: ReadonlyMap<string, GroupState>;
};

const labelled = (sources: readonly GemvSweepSource[], kernelsOf: CurrentKernels): Labelled[] =>
  sources.map((source, index) => ({
    source,
    label: `[${index + 1}]`,
    states: new Map(
      [...source.groups].map(([key, group]) => [key, groupState(source, group, kernelsOf)]),
    ),
  }));

/** A candidate in one group, against the default's choice of its class there. */
type Reading =
  | { readonly kind: "evidence"; readonly gain: number }
  /** A failure or a failed check in a current group: the candidate is not taken. */
  | { readonly kind: "ruled-out"; readonly reason: string; readonly gain?: number }
  /** The group is no evidence for it (not measured, stale, screened, unsteady). */
  | { readonly kind: "none"; readonly reason: string };

const readGroup = (
  { states, label }: Labelled,
  caseId: string,
  m: number,
  candidate: string,
  reference: string,
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
  // The group is current, so the reference has a time
  const time = state.group.rows.get(reference)?.perDispatch;
  // A failed check keeps its gain: the reader sees how fast the candidate was
  const gain =
    time === undefined || row.perDispatch === undefined ? undefined : time / row.perDispatch;
  const withGain = gain === undefined ? {} : { gain };
  if (candidateClass === "identical" && row.identicalToBaseline !== true) {
    return {
      kind: "ruled-out",
      reason: `its words differ from the baseline's in ${label} at M=${m}`,
      ...withGain,
    };
  }
  if (candidateClass === "reorder" && row.withinBand !== true) {
    return { kind: "ruled-out", reason: `outside the band in ${label} at M=${m}`, ...withGain };
  }
  if (state.unsteady !== undefined) {
    return { kind: "none", reason: `unsteady in ${label} at M=${m} (${state.unsteady})` };
  }
  if (gain === undefined) return { kind: "none", reason: `has no time in ${label} at M=${m}` };
  return { kind: "evidence", gain };
};

/** The gain of a candidate in one group that is evidence: the default's choice's time ÷ its own. */
export type LinearGemvGain = {
  /** The record (its path, as given). */
  readonly record: string;
  readonly caseId: string;
  readonly gain: number;
};

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
  /** What the records showed that the field does not hold (the subgroup form). */
  readonly notes: readonly string[];
};

/** A group of a record that is not of the current runtime. */
export type LinearGemvStaleGroup = {
  readonly record: string;
  readonly caseId: string;
  readonly m: number;
  /** What is not current: the case, the default's choices or a reference's row. */
  readonly cause: "case" | "choices" | "references";
  readonly reason: string;
};

/** How much of a record stands for the current runtime. */
export type LinearGemvRecordGroups = {
  /** Current and steady: evidence for its current rows. */
  readonly current: number;
  /** Current but unsteady (no evidence; a failure or a failed check still rules out). */
  readonly unsteady: number;
  /** Not current (each in {@link LinearGemvDerivation.stale}). */
  readonly stale: number;
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
  /** The groups no evidence comes from because they are not of the current runtime. */
  readonly stale: readonly LinearGemvStaleGroup[];
  /** The decisions as lines (a generated table's header, a generator's dry run). */
  readonly verdicts: readonly string[];
  /** The rule's threshold and steadiness bound. */
  readonly rule: { readonly minSpeedup: number; readonly steadySpread: number };
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
};

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
};

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
      if (row.candidate === reference || row.isBaseline || !judgement.takes(row)) continue;
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
      reading: readGroup(
        record,
        gemvCase.id,
        m,
        candidate.name,
        judgement.reference(m),
        judgement.class,
      ),
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
  for (const m of judgement.others) {
    const other = readAt(m);
    for (const { reading, label } of other) {
      if (reading.kind === "ruled-out") fail(reading.reason);
      else if (reading.kind === "evidence" && reading.gain < NO_REGRESSION) {
        fail(
          `slower than the default's choice in ${label} at M=${m} (${formatRatio(reading.gain)})`,
        );
      }
    }
    if (evidenceRecords(other).length < MIN_RECORDS) {
      fail(
        `evidence at M=${m} from fewer than ${MIN_RECORDS} records (a parallel row serves M = 1 to 8: ${noEvidenceText(other)})`,
      );
    }
  }
  return { ...candidate, gains, ...(failure === undefined ? {} : { failure }) };
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
  const baseline = gemvBaseline(gemvCase, 1).name;
  return {
    gemvCase,
    class: "identical",
    reference: () => baseline,
    takes: (row) => row.class === "identical" && row.choice.id === "gemv-m1",
    others: [],
  };
};

/** The reorder class of one case for the table (plain and staged, or subgroup for the notes). */
const reorderJudgement = (
  gemvCase: GemvCase,
  ids: readonly GemvSweepChoice["id"][],
): CaseJudgement => ({
  gemvCase,
  class: "reorder",
  reference: (m) => gemvProfileChoice(gemvCase, m, "reorder").name,
  takes: (row) => row.class === "reorder" && ids.includes(row.choice.id),
  others: noRegressionRows(gemvCase),
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
    reorderJudgement(gemvCase, ["gemv-parallel-subgroup32"]),
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
  } as const;
  if (defaultRow?.packedActivations === true) {
    return {
      decision: {
        ...head,
        derivedChoice: defaultChoice,
        outcome: "kept",
        reason: "the default takes packed activations, whose kernels the sweep does not measure",
        gains: [],
      },
    };
  }
  const outcome = judgeCase(
    records,
    reorderJudgement(gemvCase, ["gemv-parallel", "gemv-parallel-staged"]),
    kernelsOf,
  );
  if (outcome.kind === "kept") {
    return {
      decision: {
        ...head,
        derivedChoice: defaultChoice,
        outcome: "kept",
        reason: outcome.reason,
        ...(outcome.nearest === undefined ? {} : { candidate: outcome.nearest.name }),
        gains: outcome.nearest?.gains ?? [],
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
  };
  return {
    decision: {
      ...head,
      derivedChoice: evaluation.name,
      outcome: "adopted",
      candidate: evaluation.name,
      gains: evaluation.gains,
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
  const body =
    decision.outcome === "adopted"
      ? `${decision.defaultChoice} -> ${decision.derivedChoice}${gains}`
      : `keeps ${decision.defaultChoice} (${decision.reason ?? "?"})${gains}`;
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
  const states = [...record.states.values()];
  let staleRows = 0;
  for (const state of states) {
    if (state.kind !== "current") continue;
    for (const row of state.group.rows.values()) {
      if (rowStaleness(row, state.kernels) !== undefined) staleRows += 1;
    }
  }
  const current = states.filter(
    (state) => state.kind === "current" && state.unsteady === undefined,
  ).length;
  const unsteady = states.filter(
    (state) => state.kind === "current" && state.unsteady !== undefined,
  ).length;
  return { current, unsteady, stale: states.length - current - unsteady, staleRows };
};

/** The groups of the records that are not current, in record order. */
const staleGroups = (records: readonly Labelled[]): LinearGemvStaleGroup[] =>
  records.flatMap(({ source, states }) =>
    [...states.values()].flatMap((state) =>
      state.kind === "stale"
        ? [
            {
              record: source.path,
              caseId: state.group.caseId,
              m: state.group.m,
              cause: state.cause,
              reason: state.reason,
            },
          ]
        : [],
    ),
  );

const CAUSE_TEXT = {
  case: "the case is not the current one",
  choices: "the default profile's choices moved",
  references: "a reference's row is not current",
} as const satisfies Record<LinearGemvStaleGroup["cause"], string>;

/** One record's line, and per cause the stale groups it has. */
const recordLines = (record: Labelled, stale: readonly LinearGemvStaleGroup[]): string[] => {
  const { source, label } = record;
  const groups = recordGroups(record);
  const total = groups.current + groups.unsteady + groups.stale;
  const mine = stale.filter((entry) => entry.record === source.path);
  const causes = (Object.keys(CAUSE_TEXT) as LinearGemvStaleGroup["cause"][]).flatMap((cause) => {
    const count = mine.filter((entry) => entry.cause === cause).length;
    return count === 0 ? [] : [`    - ${count} not current: ${CAUSE_TEXT[cause]}`];
  });
  return [
    `  - ${label} ${source.path} (${source.date}, set ${source.candidateSet}): ${total} groups — ${groups.current} current and steady, ${groups.unsteady} unsteady, ${groups.stale} not current; ${groups.staleRows} rows of current groups not current`,
    ...causes,
  ];
};

const verdictLinesOf = (
  decisions: readonly LinearGemvDecision[],
  records: readonly Labelled[],
  stale: readonly LinearGemvStaleGroup[],
): string[] => [
  `- linearGemv (the GEMV choices of linear), from ${records.length} GEMV sweep records:`,
  ...records.flatMap((record) => recordLines(record, stale)),
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
 * — groups of at least two records, current and steady — with its check held in every current group,
 * and at the parallel row's other row counts not slower in any, replaces that choice; otherwise the
 * default's stays. Groups that are not current are reported, not refused; a record that no group
 * could stand for throws with the reason (fail loudly).
 */
export const deriveLinearGemvChoices = (
  inputs: readonly GemvSweepInput[],
  options: DeriveLinearGemvOptions,
): LinearGemvDerivation => {
  const sources = readRecords(inputs, options.table);
  const kernelsOf = currentKernels();
  const records = labelled(sources, kernelsOf);
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
    verdicts: verdictLinesOf(decisions, records, stale),
    rule: { minSpeedup: GEMV_MIN_SPEEDUP, steadySpread: GEMV_STEADY_SPREAD },
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
 * that fewer than two records hold a current, steady group with a current row of — timed, screened
 * out by the quick look (a screened row is a current look that the candidate did not come near
 * enough to matter; measuring it again would screen it again), or skipped because the record's M = 1
 * group of the case already left it unadoptable (gemv-candidates.ts's `gemvCarriedFromM1`; a run
 * that measures M = 1 again would skip it again). A row that failed is missing. A case
 * and row count no record has a current, steady group of misses every candidate. Measures nothing.
 * At least one record; with none there is nothing to plan from — measure everything.
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
  const kernelsOf = currentKernels();
  const records = labelled(sources, kernelsOf);
  const missing: GemvMissingMeasurement[] = [];
  let groups = 0;
  for (const gemvCase of GEMV_CASES) {
    for (const m of gemvRowCountsIn(gemvCase, "full")) {
      groups += 1;
      // The default's choice at reorder is a reference: every measurement of the group takes it
      const reference = gemvProfileChoice(gemvCase, m, "reorder").name;
      const read = gemvCandidates(gemvCase, m, "full", gates).filter(
        (candidate) =>
          candidate.name !== reference && gemvGroupRole(m, candidate.class) !== "unread",
      );
      const covered = (name: string): number =>
        records.filter(({ states }) => {
          const state = states.get(groupKey(gemvCase.id, m));
          if (state?.kind !== "current" || state.unsteady !== undefined) return false;
          const row = state.group.rows.get(name);
          return (
            row !== undefined &&
            row.error === undefined &&
            rowStaleness(row, state.kernels) === undefined
          );
        }).length;
      const lacking = read
        .filter(({ name }) => covered(name) < MIN_RECORDS)
        .map(({ name }) => name);
      if (lacking.length > 0) missing.push({ caseId: gemvCase.id, m, candidates: lacking });
    }
  }
  return { gates, groups, missing, stale: staleGroups(records) };
};
