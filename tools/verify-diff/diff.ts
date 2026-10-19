/**
 * The pure part of verify-diff — a sequence of loaded `results.json` → a "case × environment key"
 * matrix.
 *
 * Touches neither the FS nor the GPU (loading is main.ts's job). The source of truth for the schema
 * is the writer, `packages/runtime/tests/helpers/results.ts`; this is the **reader**. So it checks
 * only the fields the tool actually reads, but a missing field, a wrong type or a different `schema`
 * throws instead of silently becoming empty (an empty matrix cannot be told apart from "no
 * differences", and would look green while the results diverge).
 *
 * Cross-device identity of sha256 is not guaranteed by the spec (ADR 0106 / docs/limitations.md), so
 * an artifact sha that differs between environments is **only listed as a difference, not a
 * failure**.
 */

/** The outcome of one case (the same vocabulary as `ResultStatus` in results.ts). */
export type ResultStatus = "pass" | "fail" | "written" | "rewritten";

/** The tolerance band used for acceptance. */
export type Tolerance = { readonly atol: number; readonly rtol: number };

/** An observed tolerance check (the cases of the golden family record one per output). */
export type Measurement = {
  /** The output name in the graph. */
  readonly output: string;
  /** In JSON a non-finite value becomes null. */
  readonly maxAbs: number | null;
  readonly maxRel: number | null;
  /** The band used for acceptance. */
  readonly tolerance: Tolerance;
  /** The stage that accepted it (on fail, the last stage measured). */
  readonly stage: "karume" | "spec";
};

/** The band of the self A/B check (the same shape as `ComparisonBand` in results.ts). */
export type ComparisonBand = {
  readonly metric: "relRms";
  readonly floor: number;
  readonly ceiling: number;
};

/**
 * An observation of the self A/B check (`Comparison` in results.ts — the difference between the
 * reference preset and the practical preset at the same observation point).
 */
export type Comparison = {
  readonly output: string;
  readonly reference: string;
  readonly practical: string;
  /** In JSON a non-finite value becomes null. */
  readonly relRms: number | null;
  readonly maxAbs: number | null;
  /** Absent on a run whose band is not derived (or an observation that is only recorded). */
  readonly band?: ComparisonBand;
};

/** One item of `cases` in `results.json`. */
export type ResultEntry = {
  readonly id: string;
  readonly status: ResultStatus;
  readonly expected?: string;
  readonly actual?: string;
  readonly artifact?: string;
  readonly elapsedMs: number;
  readonly note?: string;
  readonly measurements?: readonly Measurement[];
  readonly comparisons?: readonly Comparison[];
};

/** The checkout the results were taken from (null for results taken on a machine without `git`). */
export type Checkout = { readonly sha: string; readonly dirty: boolean };

/**
 * One `results.json`.
 *
 * `environment.key` is required because on a machine without an environment key the writer cannot
 * pick the results directory and throws, so every existing `results.json` has the key. A document without
 * it cannot tell which environment's column its results belong to, so the reader rejects it too.
 */
export type ResultsDocument = {
  readonly schema: 1;
  readonly family: string;
  readonly environment: { readonly key: string };
  readonly checkout: Checkout | null;
  readonly startedAt: string;
  readonly cases: readonly ResultEntry[];
};

/** One loaded `results.json` and the date of its results directory. */
export type LoadedResults = {
  /** Where it came from (printed in warnings and failure messages). */
  readonly path: string;
  /** The date taken from the results directory's directory name (`YYYY-MM-DD`). */
  readonly date: string;
  readonly document: ResultsDocument;
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const requireRecord = (value: unknown, where: string): Record<string, unknown> => {
  if (!isRecord(value)) throw new Error(`${where}: not an object`);
  return value;
};

const requireString = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} is not a string`);
  return value;
};

const optionalString = (
  record: Record<string, unknown>,
  key: string,
  where: string,
): string | undefined => {
  if (!Object.hasOwn(record, key)) return undefined;
  return requireString(record, key, where);
};

const requireNumber = (record: Record<string, unknown>, key: string, where: string): number => {
  const value = record[key];
  if (typeof value !== "number") throw new Error(`${where}: ${key} is not a number`);
  return value;
};

/** In JSON a non-finite value becomes null, so `null` passes as is, meaning "could not be measured". */
const nullableNumber = (
  record: Record<string, unknown>,
  key: string,
  where: string,
): number | null => {
  if (record[key] === null) return null;
  return requireNumber(record, key, where);
};

const STATUSES: readonly ResultStatus[] = ["pass", "fail", "written", "rewritten"];
const STAGES: readonly Measurement["stage"][] = ["karume", "spec"];

const requireStatus = (record: Record<string, unknown>, where: string): ResultStatus => {
  const status = requireString(record, "status", where);
  // Passing an unknown spelling silently would put an "unknown outcome" into the matrix and mix it
  // into the difference check.
  const known = STATUSES.find((candidate) => candidate === status);
  if (known === undefined) throw new Error(`${where}: unknown status '${status}'`);
  return known;
};

const parseMeasurement = (value: unknown, where: string): Measurement => {
  const record = requireRecord(value, where);
  const stage = requireString(record, "stage", where);
  const known = STAGES.find((candidate) => candidate === stage);
  if (known === undefined) throw new Error(`${where}: unknown stage '${stage}'`);
  const tolerance = requireRecord(record.tolerance, `${where}.tolerance`);
  return {
    output: requireString(record, "output", where),
    maxAbs: nullableNumber(record, "maxAbs", where),
    maxRel: nullableNumber(record, "maxRel", where),
    tolerance: {
      atol: requireNumber(tolerance, "atol", `${where}.tolerance`),
      rtol: requireNumber(tolerance, "rtol", `${where}.tolerance`),
    },
    stage: known,
  };
};

const BAND_METRICS: readonly ComparisonBand["metric"][] = ["relRms"];

const parseComparison = (value: unknown, where: string): Comparison => {
  const record = requireRecord(value, where);
  let band: ComparisonBand | undefined;
  if (Object.hasOwn(record, "band")) {
    const raw = requireRecord(record.band, `${where}.band`);
    const metric = requireString(raw, "metric", `${where}.band`);
    // Passing an unknown metric silently would list a band of another quantity as a relRMS band.
    const known = BAND_METRICS.find((candidate) => candidate === metric);
    if (known === undefined) throw new Error(`${where}.band: unknown metric '${metric}'`);
    band = {
      metric: known,
      floor: requireNumber(raw, "floor", `${where}.band`),
      ceiling: requireNumber(raw, "ceiling", `${where}.band`),
    };
  }
  return {
    output: requireString(record, "output", where),
    reference: requireString(record, "reference", where),
    practical: requireString(record, "practical", where),
    relRms: nullableNumber(record, "relRms", where),
    maxAbs: nullableNumber(record, "maxAbs", where),
    ...(band === undefined ? {} : { band }),
  };
};

const parseEntry = (value: unknown, where: string): ResultEntry => {
  const record = requireRecord(value, where);
  const id = requireString(record, "id", where);
  const at = `${where} (${id})`;
  const expected = optionalString(record, "expected", at);
  const actual = optionalString(record, "actual", at);
  const artifact = optionalString(record, "artifact", at);
  const note = optionalString(record, "note", at);
  const measurements = record.measurements;
  if (measurements !== undefined && !Array.isArray(measurements)) {
    throw new Error(`${at}: measurements is not an array`);
  }
  const comparisons = record.comparisons;
  if (comparisons !== undefined && !Array.isArray(comparisons)) {
    throw new Error(`${at}: comparisons is not an array`);
  }
  return {
    id,
    status: requireStatus(record, at),
    ...(expected === undefined ? {} : { expected }),
    ...(actual === undefined ? {} : { actual }),
    ...(artifact === undefined ? {} : { artifact }),
    elapsedMs: requireNumber(record, "elapsedMs", at),
    ...(note === undefined ? {} : { note }),
    ...(measurements === undefined
      ? {}
      : {
          measurements: measurements.map((one, index) =>
            parseMeasurement(one, `${at}.measurements[${index}]`),
          ),
        }),
    ...(comparisons === undefined
      ? {}
      : {
          comparisons: comparisons.map((one, index) =>
            parseComparison(one, `${at}.comparisons[${index}]`),
          ),
        }),
  };
};

const parseCheckout = (value: unknown, where: string): Checkout | null => {
  if (value === null) return null;
  const record = requireRecord(value, `${where}.checkout`);
  const dirty = record.dirty;
  if (typeof dirty !== "boolean") throw new Error(`${where}.checkout: dirty is not a boolean`);
  return { sha: requireString(record, "sha", `${where}.checkout`), dirty };
};

/**
 * Checks the contents of a `results.json` (`unknown`) and types them.
 *
 * MUST: throw when a check fails (the value of this tool is "pointing at where the results
 * diverge", so swallowing an unreadable document as empty would print a false "no differences").
 */
export const parseResultsDocument = (value: unknown, source = "results.json"): ResultsDocument => {
  const record = requireRecord(value, source);
  const schema = record.schema;
  if (schema !== 1) {
    throw new Error(`${source}: schema is not 1 (${JSON.stringify(schema)})`);
  }
  const environment = requireRecord(record.environment, `${source}.environment`);
  const cases = record.cases;
  if (!Array.isArray(cases)) throw new Error(`${source}: cases is not an array`);
  const parsed = cases.map((one, index) => parseEntry(one, `${source}.cases[${index}]`));
  // Rows are looked up by case ID (only the first one is taken), so letting a duplicate through
  // would silently drop the outcomes from the second one on — reject it at the boundary, like an
  // unreadable document.
  const ids = new Set<string>();
  for (const entry of parsed) {
    if (ids.has(entry.id)) throw new Error(`${source}: duplicate case ID '${entry.id}'`);
    ids.add(entry.id);
  }
  return {
    schema: 1,
    family: requireString(record, "family", source),
    environment: { key: requireString(environment, "key", `${source}.environment`) },
    checkout: parseCheckout(record.checkout, source),
    startedAt: requireString(record, "startedAt", source),
    cases: parsed,
  };
};

/** Filters of the matrix. */
export type DiffOptions = {
  /** Only these families (unset = every family). */
  readonly families?: readonly string[];
  /** Only this date (unset = the newest date per environment). */
  readonly date?: string;
};

/** One cell of the matrix (the outcome that environment holds for that case). */
export type Cell = {
  readonly status: ResultStatus;
  readonly actual?: string;
  readonly elapsedMs: number;
  readonly note?: string;
  readonly measurements?: readonly Measurement[];
  readonly comparisons?: readonly Comparison[];
};

/** For one family, the document taken from that environment. */
export type SelectedDocument = {
  readonly environment: string;
  readonly date: string;
  readonly path: string;
  readonly checkout: Checkout | null;
};

/** The kind of a difference (an artifact sha mismatch is not a failure — see the header note). */
export type DifferenceKind = "status" | "actual" | "missing";

export type Difference = {
  readonly caseId: string;
  readonly kind: DifferenceKind;
  /**
   * Environment key → that environment's value (the status or the artifact sha). Environments
   * without a value are not included.
   */
  readonly values: Readonly<Record<string, string>>;
  /** The environments that do not have the case (only when `kind` is `missing`). */
  readonly missing?: readonly string[];
};

export type MatrixRow = {
  readonly id: string;
  /** Environment key → cell. An environment without a result has no field at all. */
  readonly cells: Readonly<Record<string, Cell>>;
};

export type FamilyMatrix = {
  readonly family: string;
  /** The columns (lexicographic order). */
  readonly environments: readonly string[];
  readonly selected: readonly SelectedDocument[];
  /** Environments present under the root that have no results for this family (and `--date`). */
  readonly absent: readonly string[];
  /** The rows (the union of case IDs, lexicographic order). */
  readonly rows: readonly MatrixRow[];
  readonly differences: readonly Difference[];
  readonly warnings: readonly string[];
};

export type DiffReport = {
  readonly families: readonly FamilyMatrix[];
  readonly warnings: readonly string[];
};

const toCell = (entry: ResultEntry): Cell => ({
  status: entry.status,
  ...(entry.actual === undefined ? {} : { actual: entry.actual }),
  elapsedMs: entry.elapsedMs,
  ...(entry.note === undefined ? {} : { note: entry.note }),
  ...(entry.measurements === undefined ? {} : { measurements: entry.measurements }),
  ...(entry.comparisons === undefined ? {} : { comparisons: entry.comparisons }),
});

const shortSha = (sha: string): string => sha.slice(0, 7);

/** Looks at the checkouts of the selected documents (are results of different commits side by side?). */
const checkoutWarnings = (selected: readonly SelectedDocument[]): string[] => {
  const warnings: string[] = [];
  const known = selected.flatMap((one) =>
    one.checkout === null ? [] : [{ environment: one.environment, checkout: one.checkout }],
  );
  if (new Set(known.map((one) => one.checkout.sha)).size > 1) {
    const columns = known
      .map((one) => `${one.environment}=${shortSha(one.checkout.sha)}`)
      .join(" / ");
    warnings.push(
      `checkout sha differs between environments (results of different commits side by side): ${columns}`,
    );
  }
  for (const one of selected) {
    if (one.checkout === null) {
      warnings.push(`${one.environment}: no checkout (results taken on a machine without git)`);
      continue;
    }
    if (one.checkout.dirty) {
      warnings.push(
        `${one.environment}: results taken from a checkout with a dirty working tree` +
          `（${shortSha(one.checkout.sha)}）`,
      );
    }
  }
  return warnings;
};

const differencesOf = (
  environments: readonly string[],
  rows: readonly MatrixRow[],
): Difference[] => {
  const differences: Difference[] = [];
  for (const row of rows) {
    const statuses: Record<string, string> = {};
    const actuals: Record<string, string> = {};
    const missing: string[] = [];
    for (const environment of environments) {
      const cell = Object.hasOwn(row.cells, environment) ? row.cells[environment] : undefined;
      if (cell === undefined) {
        missing.push(environment);
        continue;
      }
      statuses[environment] = cell.status;
      if (cell.actual !== undefined) actuals[environment] = cell.actual;
    }
    // A case only one side has (not listed with a single environment, where nothing can diverge).
    if (environments.length > 1 && missing.length > 0) {
      differences.push({ caseId: row.id, kind: "missing", values: statuses, missing });
    }
    if (new Set(Object.values(statuses)).size > 1) {
      differences.push({ caseId: row.id, kind: "status", values: statuses });
    }
    // Compare only the environments that have a sha (a case where one side takes no sha has not
    // diverged).
    if (new Set(Object.values(actuals)).size > 1) {
      differences.push({ caseId: row.id, kind: "actual", values: actuals });
    }
  }
  return differences;
};

/**
 * Turns the loaded results into one matrix per family.
 *
 * When the same (family, environment key) has several dates, the **newest date** is taken (with
 * `date` given, only that date). Result entries are split down to the date, so this is "the last run
 * of that environment".
 */
export const buildDiff = (
  loaded: readonly LoadedResults[],
  options: DiffOptions = {},
): DiffReport => {
  const warnings: string[] = [];
  const wanted = options.families === undefined ? undefined : new Set(options.families);
  if (wanted !== undefined) {
    const present = new Set(loaded.map((one) => one.document.family));
    for (const family of [...wanted].sort()) {
      // Always report a filter that matched nothing, so a mistyped family name does not turn into
      // "no differences".
      if (!present.has(family)) warnings.push(`no results match --family ${family}`);
    }
  }
  // A mistyped date (a day not in the calendar, a day with no run) also turns into "no
  // differences", so, as with --family, always report a filter that matched nothing.
  if (options.date !== undefined && !loaded.some((one) => one.date === options.date)) {
    warnings.push(`no results directory matches --date ${options.date}`);
  }

  const allEnvironments = [...new Set(loaded.map((one) => one.document.environment.key))].sort();
  const byFamily = new Map<string, Map<string, LoadedResults>>();
  for (const entry of loaded) {
    const { family, environment } = entry.document;
    if (wanted !== undefined && !wanted.has(family)) continue;
    if (options.date !== undefined && entry.date !== options.date) continue;
    const perEnvironment = byFamily.get(family) ?? new Map<string, LoadedResults>();
    byFamily.set(family, perEnvironment);
    const held = perEnvironment.get(environment.key);
    // Two results directories with the same (family, environment key, date) silently drop one of them
    // (two machines with the same GPU model report the same environment key, so a hand copy does
    // this routinely). The reader cannot tell which one stayed, so both paths are listed. The exit
    // code does not change (ADR 0106: "1 only for unreadable documents or a different schema").
    if (held !== undefined && held.date === entry.date && held.path !== entry.path) {
      warnings.push(
        `two results directories with the same family, environment key and date (${family} / ${environment.key} / ${entry.date})` +
          ` — only one is taken: ${held.path} / ${entry.path}`,
      );
    }
    if (held === undefined || held.date < entry.date) perEnvironment.set(environment.key, entry);
  }

  const families = [...byFamily.entries()]
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([family, perEnvironment]): FamilyMatrix => {
      const chosen = [...perEnvironment.entries()].sort(([left], [right]) =>
        left < right ? -1 : left > right ? 1 : 0,
      );
      const environments = chosen.map(([environment]) => environment);
      const selected = chosen.map(([environment, entry]): SelectedDocument => ({
        environment,
        date: entry.date,
        path: entry.path,
        checkout: entry.document.checkout,
      }));
      const caseIds = new Set<string>();
      for (const [, entry] of chosen) for (const one of entry.document.cases) caseIds.add(one.id);
      const rows = [...caseIds].sort().map((id): MatrixRow => {
        const cells: Record<string, Cell> = {};
        for (const [environment, entry] of chosen) {
          const found = entry.document.cases.find((one) => one.id === id);
          if (found !== undefined) cells[environment] = toCell(found);
        }
        return { id, cells };
      });
      return {
        family,
        environments,
        selected,
        absent: allEnvironments.filter((environment) => !perEnvironment.has(environment)),
        rows,
        differences: differencesOf(environments, rows),
        warnings: checkoutWarnings(selected),
      };
    });

  return { families, warnings };
};
