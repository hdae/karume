/**
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) from GEMV sweep records
 * (`karume-gemv-sweep/2` — gemv-report.ts): {@link deriveLinearGemvChoices} of
 * `@karume/runtime/tune`. A pure function of the records and the current runtime (its case table,
 * its kernels and the default profile's choices); it measures nothing.
 *
 * ## The adoption rule (the maintainer, 2026-11-17)
 *
 * The timing of these small dispatches moves between runs, so one record is not evidence:
 *
 * - The input is two or more records of one adapter, measured with the current kernels and case
 *   table (the records' fingerprints, recomputed here), with GPU timestamps that are not quantized,
 *   against the current default profile's choices. Any other record is refused, and so are fewer
 *   than two.
 * - A candidate replaces the default profile's choice for a shape only when it is at least
 *   {@link GEMV_MIN_SPEEDUP} faster than that choice in every record, its check holding in every
 *   record (identical class: the baseline's words; reorder class: inside the band). Among several,
 *   the one whose smallest gain over the records is largest. A candidate's time is the record's
 *   `perDispatch` (the minimum of its rounds ÷ `reps`).
 * - A (case, row count) whose baseline spread exceeds {@link GEMV_STEADY_SPREAD} in a record is
 *   unsteady in that record; a shape the decision reads at an unsteady (case, row count) is kept.
 * - The two classes are derived separately: the identical class writes the sequential variant's
 *   per-shape exceptions (`sequential.shapes`, from M = 1) and changes no number of any Session; the
 *   reorder class writes the parallel rows (`parallelShapes`: lanes and the staged flag, from the
 *   plain and staged candidates at M = 1) and changes the numbers of Sessions at `reorder`, inside
 *   the band. A parallel row serves M = 1 to 8, so every record must have measured the case's other
 *   row counts of that range (the `full` set does; a `quick` record keeps the row), the candidate
 *   not slower than the default's choice at any of them: a path that was not measured is not moved
 *   by a table.
 *
 * What the field does not let the records decide: a sequential key has no group length, so cases
 * that share it and disagree keep the default; a row whose default takes packed activations stays
 * (the packed kernels are not measured); the subgroup candidates are a Session option, not a row,
 * and only show in the decision's notes; the row-block thread target is one number for every shape,
 * and no rule turns the per-shape row-block heights of the `full` set into it, so it stays; the
 * general sequential variant also sets the row blocks' columns and read-ahead, so it stays.
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
  type GemvCandidateClass,
  type GemvCandidateSet,
  gemvBaseline,
  gemvProfileChoice,
  isGemvCandidateSet,
} from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, gemvCaseOf, type GemvSweepStorage } from "./gemv-cases.ts";
import { gemvCaseTableId, gemvKernelsId } from "./gemv-fingerprint.ts";
import { GEMV_SWEEP_FORMAT, type GemvSweepChoice } from "./gemv-report.ts";

/**
 * The adoption threshold of the GEMV choices (the maintainer, 2026-11-17): a candidate replaces the
 * default profile's choice for a shape only when it is at least this much faster than that choice in
 * every record. It is not the tile geometry's `minSpeedup`.
 */
export const GEMV_MIN_SPEEDUP = 1.1;

/**
 * The steadiness bound: a (case, row count) whose baseline rounds spread (`baselineSpread`, the
 * baseline's largest round ÷ its smallest) beyond this in a record is unsteady in that record, and a
 * shape with an unsteady record is not changed.
 *
 * Source: the two `karume-gemv-sweep/2` records of the `quick` set on the RTX 3080 Ti under Node
 * (2026-11-17T11-51-15 and T12-29-00), grouped by the worse spread of a case over the two records: a
 * candidate's gain over the table's choice agrees between the records within ±10 % for 95.6 % of the
 * rows below 1.05, 96.1 % in [1.05, 1.10), 97.2 % in [1.10, 1.25) and 90.3 % from 1.25 up (12 of 66
 * cases). Below 1.25 the spread does not predict a less repeatable gain: the device slows as a case
 * goes on (the last round is the slowest in 39 % of the rows), every kernel of a round sees it alike,
 * and the minimum reads the early rounds.
 */
export const GEMV_STEADY_SPREAD = 1.25;

/** The fewest records a derivation takes: one record is not evidence. */
const MIN_RECORDS = 2;

/**
 * At a row count of the parallel range other than M = 1, a parallel candidate may not be slower than
 * the default's choice (a gain below this is a regression).
 */
const NO_REGRESSION = 1;

/** The largest row count the parallel GEMV serves (its rows apply at M = 1..8). */
const PARALLEL_MAX_ROWS = 8;

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

/** The fields of one row the derivation reads (and the fingerprint recomputes). */
type RowObservation = {
  readonly caseId: string;
  readonly m: number;
  readonly candidate: string;
  readonly class: GemvCandidateClass;
  readonly isBaseline: boolean;
  readonly choice: GemvSweepChoice;
  readonly perDispatch?: number;
  readonly identicalToBaseline?: boolean;
  readonly withinBand?: boolean;
  readonly error?: string;
};

/** One measured (case, row count) of a record: its baseline spread and its rows by candidate. */
type Measured = {
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
  readonly kernels: string;
  /** In the record's order. */
  readonly rows: readonly RowObservation[];
  /** By {@link measuredKey}. */
  readonly measured: ReadonlyMap<string, Measured>;
};

const measuredKey = (caseId: string, m: number): string => `${caseId} M=${m}`;

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

const CHOICE_IDS = [
  "gemv-m1",
  "gemv-rows",
  "gemv-parallel",
  "gemv-parallel-packed",
  "gemv-parallel-staged",
  "gemv-parallel-subgroup32",
] as const satisfies readonly GemvSweepChoice["id"][];

const isChoiceId = (value: unknown): value is GemvSweepChoice["id"] =>
  CHOICE_IDS.some((id) => id === value);

const STORAGES = ["i2", "i4", "i8"] as const satisfies readonly GemvSweepStorage[];

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
  if (!isChoiceId(id)) throw new Error(`${where}: choice.id ${JSON.stringify(id)} is no GEMV form`);
  if (!isStorage(storage)) {
    throw new Error(`${where}: choice.storage ${JSON.stringify(storage)} is not i2, i4 or i8`);
  }
  const groupSize = optionalAt(choice, "groupSize", isPositiveInteger, `${where} choice`);
  const lanes = optionalAt(choice, "lanes", isLanes, `${where} choice`);
  const variant = variantAt(choice.variant, `${where} choice.variant`);
  const rowsAt = choice.rowsVariant;
  let rowsVariant: GemvSweepChoice["rowsVariant"];
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
  const perDispatch = optionalAt(row, "perDispatch", isPositiveNumber, where);
  const identicalToBaseline = optionalAt(row, "identicalToBaseline", isBoolean, where);
  const withinBand = optionalAt(row, "withinBand", isBoolean, where);
  const error = optionalAt(row, "error", isString, where);
  return {
    caseId: stringAt(row, "caseId", where),
    m,
    candidate: stringAt(row, "candidate", where),
    class: row.class,
    isBaseline,
    choice: parseChoice(row.choice, where),
    ...(perDispatch === undefined ? {} : { perDispatch }),
    ...(identicalToBaseline === undefined ? {} : { identicalToBaseline }),
    ...(withinBand === undefined ? {} : { withinBand }),
    ...(error === undefined ? {} : { error }),
  };
};

/** The case of a record's id at a row count it plans (fail loudly — measure again). */
const currentCase = (caseId: string, m: number, where: string): GemvCase => {
  const gemvCase = gemvCaseOf(caseId);
  if (gemvCase === undefined) {
    throw new Error(`${where}: case ${caseId} is not in the GEMV sweep's case table`);
  }
  if (!gemvCase.rows.includes(m)) {
    throw new Error(
      `${where}: case ${caseId} has no row count ${m} in the GEMV sweep's case table`,
    );
  }
  return gemvCase;
};

/**
 * Reads a GEMV sweep record (the `unknown` boundary — checks what the derivation reads and that the
 * record stands for the current runtime, and fails loudly): the format, GPU timestamps that are not
 * quantized, the case table's and the kernels' fingerprints, and per (case, row count) the default
 * profile's choices the record measured against.
 */
const parseGemvSweepRecord = (
  report: unknown,
  meta: { readonly path: string; readonly sha256: string },
): GemvSweepSource => {
  const where = meta.path;
  const record = objectAt(report, `${where}: the JSON`);
  if (record.format !== GEMV_SWEEP_FORMAT) {
    throw new Error(
      `${where}: format is not ${GEMV_SWEEP_FORMAT} (${JSON.stringify(record.format)}) — measure again with the current runtime`,
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
  const caseTable = stringAt(record, "caseTable", where);
  const currentTable = gemvCaseTableId();
  if (caseTable !== currentTable) {
    throw new Error(
      `${where}: its case table ${caseTable} is not the current runtime's ${currentTable} — measure again with the current runtime`,
    );
  }
  const candidateSet = objectAt(record.settings, `${where} settings`).candidateSet;
  if (typeof candidateSet !== "string" || !isGemvCandidateSet(candidateSet)) {
    throw new Error(
      `${where}: settings.candidateSet ${JSON.stringify(candidateSet)} is neither quick nor full`,
    );
  }
  const adapterAt = objectAt(record.adapter, `${where} adapter`);
  const adapter: Adapter = {
    vendor: stringAt(adapterAt, "vendor", `${where} adapter`),
    architecture: stringAt(adapterAt, "architecture", `${where} adapter`),
    device: stringAt(adapterAt, "device", `${where} adapter`),
    description: stringAt(adapterAt, "description", `${where} adapter`),
  };
  if (!Array.isArray(record.rows)) throw new Error(`${where}: rows is not an array`);
  const rows = record.rows.map((row, index) => parseRow(row, `${where} rows[${index}]`));
  const kernels = stringAt(record, "kernels", where);
  let currentKernels: string;
  try {
    currentKernels = gemvKernelsId(rows);
  } catch (cause) {
    throw new Error(`${where}: ${cause instanceof Error ? cause.message : String(cause)}`, {
      cause,
    });
  }
  if (kernels !== currentKernels) {
    throw new Error(
      `${where}: its kernel fingerprint ${kernels} is not the current runtime's ${currentKernels} for its rows — measure again with the current runtime`,
    );
  }
  if (!Array.isArray(record.summaries)) throw new Error(`${where}: summaries is not an array`);
  const measured = new Map<
    string,
    { baselineSpread?: number; rows: Map<string, RowObservation> }
  >();
  record.summaries.forEach((value, index) => {
    const at = `${where} summaries[${index}]`;
    const summary = objectAt(value, at);
    const caseId = stringAt(summary, "caseId", at);
    const { m } = summary;
    if (!isPositiveInteger(m)) throw new Error(`${at}: m is not a positive integer`);
    const gemvCase = currentCase(caseId, m, at);
    const profile = objectAt(summary.profile, `${at} profile`);
    // MUST: the gains are against the default profile's choices; a record measured against other
    // choices (the default moved since) labels other kernels as the baseline and the reorder choice
    const expected = {
      exact: gemvBaseline(gemvCase, m).name,
      reorder: gemvProfileChoice(gemvCase, m, "reorder").name,
    };
    for (const level of ["exact", "reorder"] as const) {
      const named = stringAt(profile, level, `${at} profile`);
      if (named !== expected[level]) {
        throw new Error(
          `${at}: the default profile's choice at ${level} for ${caseId} at M=${m} is ${named} in the record, ${expected[level]} in the current runtime — measure again with the current runtime`,
        );
      }
    }
    const key = measuredKey(caseId, m);
    if (measured.has(key)) throw new Error(`${at}: ${key} is summarised twice`);
    const baselineSpread = optionalAt(summary, "baselineSpread", isPositiveNumber, at);
    measured.set(key, {
      ...(baselineSpread === undefined ? {} : { baselineSpread }),
      rows: new Map(),
    });
  });
  rows.forEach((row, index) => {
    const at = `${where} rows[${index}]`;
    const key = measuredKey(row.caseId, row.m);
    const entry = measured.get(key);
    if (entry === undefined) throw new Error(`${at}: ${key} has no summary`);
    if (entry.rows.has(row.candidate)) throw new Error(`${at}: ${row.candidate} of ${key} twice`);
    if (
      row.isBaseline &&
      row.candidate !== gemvBaseline(currentCase(row.caseId, row.m, at), row.m).name
    ) {
      throw new Error(
        `${at}: the baseline of ${key} is ${row.candidate}, not the current runtime's`,
      );
    }
    entry.rows.set(row.candidate, row);
  });
  const userAgent = typeof record.userAgent === "string" ? record.userAgent : undefined;
  return {
    path: meta.path,
    sha256: meta.sha256,
    date: stringAt(record, "date", where),
    candidateSet,
    ...(userAgent === undefined ? {} : { userAgent }),
    adapter,
    kernels,
    rows,
    measured,
  };
};

/** The gain of a candidate in one record: the default's choice's time ÷ its own. */
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
  /** At M = 1, per record (and per case of the key). */
  readonly gains: readonly LinearGemvGain[];
  /** What the records showed that the field does not hold (the subgroup form). */
  readonly notes: readonly string[];
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
      readonly kernels: string;
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
const assertOneAdapter = (sources: readonly GemvSweepSource[], table: GeometryProfile): void => {
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

/** A record's label in the verdict lines: its 1-based position. */
type Labelled = { readonly source: GemvSweepSource; readonly label: string };

/** The time of a candidate against the default's choice in one record at one (case, row count). */
type Reading = { readonly gain?: number; readonly failure?: string };

const readGain = (
  { source, label }: Labelled,
  caseId: string,
  m: number,
  candidate: string,
  reference: string,
  candidateClass: GemvCandidateClass,
): Reading => {
  const at = source.measured.get(measuredKey(caseId, m));
  if (at === undefined) return { failure: `M=${m} not measured in ${label}` };
  const row = at.rows.get(candidate);
  if (row === undefined) return { failure: `not measured in ${label} at M=${m}` };
  if (row.error !== undefined) return { failure: `failed in ${label} at M=${m} (${row.error})` };
  if (row.perDispatch === undefined) return { failure: `has no time in ${label} at M=${m}` };
  const time = at.rows.get(reference)?.perDispatch;
  if (time === undefined) {
    return { failure: `the default's ${reference} has no time in ${label} at M=${m}` };
  }
  // A failed check keeps its gain: the reader sees how fast the candidate was
  const gain = time / row.perDispatch;
  if (candidateClass === "identical" && row.identicalToBaseline !== true) {
    return { gain, failure: `its words differ from the baseline's in ${label} at M=${m}` };
  }
  if (candidateClass === "reorder" && row.withinBand !== true) {
    return { gain, failure: `outside the band in ${label} at M=${m}` };
  }
  return { gain };
};

/** A candidate's gains at M = 1 over the records, and the first way it fails the rule (if any). */
type Evaluation = {
  readonly name: string;
  readonly choice: GemvSweepChoice;
  readonly gains: readonly LinearGemvGain[];
  readonly failure?: string;
};

/** The smallest of an evaluation's gains (0 when a record has none — never the largest). */
const smallestGain = (evaluation: Evaluation, records: number): number =>
  evaluation.gains.length < records ? 0 : Math.min(...evaluation.gains.map(({ gain }) => gain));

/** The evaluation whose smallest gain is largest (the first of a tie); undefined for none. */
const largestSmallest = (
  evaluations: readonly Evaluation[],
  records: number,
): Evaluation | undefined =>
  evaluations.reduce<Evaluation | undefined>(
    (best, entry) =>
      best === undefined || smallestGain(entry, records) > smallestGain(best, records)
        ? entry
        : best,
    undefined,
  );

/**
 * The row counts of the parallel range above 1 in the case table. MUST: a parallel row serves all
 * of them, so a record that did not measure one cannot move the row (a `quick` record measures
 * M = 1 only).
 */
const otherParallelRows = (gemvCase: GemvCase): number[] =>
  gemvCase.rows.filter((m) => m > 1 && m <= PARALLEL_MAX_ROWS);

/** How a candidate set is judged for one case and class. */
type CaseJudgement = {
  readonly gemvCase: GemvCase;
  readonly class: GemvCandidateClass;
  /** The default's choice at row count `m` (by candidate name). */
  readonly reference: (m: number) => string;
  /** The candidates at M = 1 this judgement compares (by the row). */
  readonly takes: (row: RowObservation) => boolean;
  /** The row counts besides M = 1 the decision reads in every record (no regression allowed there). */
  readonly others: readonly number[];
};

/** Every candidate the judgement takes, over the records, in the order they were first measured. */
const candidatesOf = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
): { readonly name: string; readonly choice: GemvSweepChoice }[] => {
  const found = new Map<string, GemvSweepChoice>();
  const reference = judgement.reference(1);
  for (const { source } of records) {
    const at = source.measured.get(measuredKey(judgement.gemvCase.id, 1));
    for (const row of at?.rows.values() ?? []) {
      if (row.candidate === reference || row.isBaseline || !judgement.takes(row)) continue;
      if (!found.has(row.candidate)) found.set(row.candidate, row.choice);
    }
  }
  return [...found].map(([name, choice]) => ({ name, choice }));
};

const evaluate = (
  records: readonly Labelled[],
  judgement: CaseJudgement,
  candidate: { readonly name: string; readonly choice: GemvSweepChoice },
): Evaluation => {
  const { gemvCase } = judgement;
  const gains: LinearGemvGain[] = [];
  let failure: string | undefined;
  const fail = (text: string): void => {
    failure ??= text;
  };
  for (const record of records) {
    const first = readGain(
      record,
      gemvCase.id,
      1,
      candidate.name,
      judgement.reference(1),
      judgement.class,
    );
    if (first.failure !== undefined) fail(first.failure);
    if (first.gain === undefined) continue;
    gains.push({ record: record.source.path, caseId: gemvCase.id, gain: first.gain });
    if (first.gain < GEMV_MIN_SPEEDUP) {
      fail(`${formatRatio(first.gain)} < ${formatRatio(GEMV_MIN_SPEEDUP)} in ${record.label}`);
    }
    for (const m of judgement.others) {
      const other = readGain(
        record,
        gemvCase.id,
        m,
        candidate.name,
        judgement.reference(m),
        judgement.class,
      );
      if (other.failure !== undefined) fail(other.failure);
      else if (other.gain !== undefined && other.gain < NO_REGRESSION) {
        fail(
          `slower than the default's choice in ${record.label} at M=${m} (${formatRatio(other.gain)})`,
        );
      }
    }
  }
  return { ...candidate, gains, ...(failure === undefined ? {} : { failure }) };
};

/** Why a case cannot change in some record (not measured, or unsteady), or undefined. */
const unusable = (records: readonly Labelled[], judgement: CaseJudgement): string | undefined => {
  const { gemvCase } = judgement;
  for (const { source, label } of records) {
    for (const m of [1, ...judgement.others]) {
      const at = source.measured.get(measuredKey(gemvCase.id, m));
      if (at === undefined) {
        return m === 1
          ? `not measured in ${label}`
          : `M=${m} is not measured in ${label} (a parallel row serves M = 1 to ${PARALLEL_MAX_ROWS}: records of the full set measure them)`;
      }
      const spread = at.baselineSpread;
      if (spread === undefined) return `no baseline spread in ${label} at M=${m}`;
      if (spread > GEMV_STEADY_SPREAD) {
        return `unsteady in ${label} at M=${m} (baseline spread ${formatRatio(spread)} > ${formatRatio(GEMV_STEADY_SPREAD)})`;
      }
    }
  }
  return undefined;
};

/** The outcome of one case and class: the adopted candidate, or why the default stays. */
type CaseOutcome =
  | { readonly kind: "adopted"; readonly evaluation: Evaluation }
  | { readonly kind: "kept"; readonly reason: string; readonly nearest?: Evaluation };

const judgeCase = (records: readonly Labelled[], judgement: CaseJudgement): CaseOutcome => {
  const evaluations = candidatesOf(records, judgement).map((candidate) =>
    evaluate(records, judgement, candidate),
  );
  const qualifying = evaluations.filter((evaluation) => evaluation.failure === undefined);
  const best = largestSmallest(qualifying, records.length);
  const nearest = best ?? largestSmallest(evaluations, records.length);
  const withNearest = nearest === undefined ? {} : { nearest };
  const blocked = unusable(records, judgement);
  if (blocked !== undefined) return { kind: "kept", reason: blocked, ...withNearest };
  if (best !== undefined) return { kind: "adopted", evaluation: best };
  if (nearest === undefined) return { kind: "kept", reason: "no candidate was measured" };
  return {
    kind: "kept",
    reason: `no candidate is ${formatRatio(GEMV_MIN_SPEEDUP)} faster in every record with its check held (nearest ${nearest.name}: ${nearest.failure ?? "?"})`,
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
  others: otherParallelRows(gemvCase),
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
): { readonly decision: LinearGemvDecision; readonly variant?: LinearGemvVariant } => {
  const [first] = cases;
  const defaultChoice = variantName(
    defaults.sequential.shapes.find((row) => sameSequentialShape(row, first))?.variant ??
      defaults.sequential.variant,
  );
  const outcomes = cases.map((gemvCase) => ({
    gemvCase,
    outcome: judgeCase(records, identicalJudgement(gemvCase)),
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
      variant: chosen.choice.variant,
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
): { readonly decision: LinearGemvDecision; readonly row?: LinearGemvParallelShape } => {
  const defaultRow = defaults.parallelShapes.find((row) => sameParallelShape(row, gemvCase));
  const defaultChoice =
    defaultRow === undefined
      ? `no row (${gemvBaseline(gemvCase, 1).name})`
      : parallelName(defaultRow);
  const subgroup = judgeCase(records, reorderJudgement(gemvCase, ["gemv-parallel-subgroup32"]));
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
  const { lanes } = evaluation.choice;
  if (lanes === undefined) throw new Error(`${evaluation.name} of ${gemvCase.id} has no lanes`);
  const row: LinearGemvParallelShape = {
    storage: gemvCase.storage,
    n: gemvCase.n,
    k: gemvCase.k,
    ...(gemvCase.group === undefined ? {} : { group: gemvCase.group }),
    lanes,
    packedActivations: false,
    ...(evaluation.choice.id === "gemv-parallel-staged" ? { stagedActivations: true } : {}),
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
  const full = records.filter(({ source }) => source.candidateSet === "full");
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
      full.length === 0
        ? "only records of the full set measure row blocks (M ≥ 2), and none is given"
        : "the full set measures the row-block heights per case and row count, but the field is one thread target for every shape, and no rule turns the heights into it",
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

const verdictLinesOf = (
  decisions: readonly LinearGemvDecision[],
  records: readonly Labelled[],
): string[] => [
  `- linearGemv (the GEMV choices of linear), from ${records.length} GEMV sweep records:`,
  ...records.map(
    ({ source, label }) =>
      `  - ${label} ${source.path} (${source.date}, set ${source.candidateSet})`,
  ),
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
 * Derives the GEMV choices of linear (`GeometryProfile.linearGemv`) from two or more GEMV sweep
 * records of one adapter, by the rule of the module doc: per shape and class, a candidate at least
 * {@link GEMV_MIN_SPEEDUP} faster than the default profile's choice in every record, its check held in
 * every record and no (case, row count) the decision reads unsteady (baseline spread above
 * {@link GEMV_STEADY_SPREAD}), replaces that choice; otherwise the default's stays. Input that fails a
 * check throws with the reason (fail loudly).
 */
export const deriveLinearGemvChoices = (
  inputs: readonly GemvSweepInput[],
  options: DeriveLinearGemvOptions,
): LinearGemvDerivation => {
  if (inputs.length < MIN_RECORDS) {
    throw new Error(
      `The GEMV choices are derived from at least ${MIN_RECORDS} GEMV sweep records of one adapter (${inputs.length} given): the timing of these dispatches moves between runs, so one record is not evidence — measure again`,
    );
  }
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
  assertOneAdapter(sources, options.table);
  const records: Labelled[] = sources.map((source, index) => ({ source, label: `[${index + 1}]` }));
  const defaults = DEFAULT_GEOMETRY_PROFILE.linearGemv;
  const cases = GEMV_CASES.filter((gemvCase) =>
    sources.some((source) => source.rows.some((row) => row.caseId === gemvCase.id)),
  );
  // Sequential keys in the order of their first case
  const keys = new Map<string, GemvCase[]>();
  for (const gemvCase of cases) {
    const key = sequentialKey(gemvCase);
    keys.set(key, [...(keys.get(key) ?? []), gemvCase]);
  }
  const sequential = [...keys.values()].map((shared) =>
    sequentialDecision(records, shared, defaults),
  );
  const parallel = cases.map((gemvCase) => parallelDecision(records, gemvCase, defaults));
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
  const [first] = sources;
  return {
    ...(changed ? { linearGemv: choices } : {}),
    defaults,
    decisions,
    verdicts: verdictLinesOf(decisions, records),
    rule: { minSpeedup: GEMV_MIN_SPEEDUP, steadySpread: GEMV_STEADY_SPREAD },
    provenance: {
      adapter: first.adapter,
      caseTable: gemvCaseTableId(),
      records: sources.map((source) => ({
        path: source.path,
        sha256: source.sha256,
        date: source.date,
        candidateSet: source.candidateSet,
        ...(source.userAgent === undefined ? {} : { userAgent: source.userAgent }),
        kernels: source.kernels,
      })),
    },
  };
};
