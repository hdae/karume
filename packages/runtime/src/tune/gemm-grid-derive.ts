/**
 * The derivation of the grid-fill rule's two integers (the table field `gemmGrid` — ADR 0157) from
 * records of the geometry sweep's `grid` set (gemm-grid.ts, harness.ts's `sweepGridCase`). A pure
 * function of the records and the table the field is derived for ({@link deriveGemmGrid}); the
 * generator (`tools/geometry-profiles`) writes its result into the table.
 *
 * The rule takes the ideas of the GEMV derivation's rule of one steady record where they carry over
 * (gemv-rule.ts — the same functions and numbers):
 *
 * - **A record's class**: steady when at least `GEMV_STEADY_MIN_GROUPS` of its cases have a default
 *   row with two positive rounds or more and at most `GEMV_STEADY_MAX_SHARE` of those spread (largest
 *   round ÷ smallest) beyond `GEMV_STEADY_GROUP_SPREAD` (`gemvSteadiness` over the cases' default
 *   rows). A steady record timed in ns is evidence on its own; any other record only beside a second
 *   record that times the same pair of tiles (`gemvRecordReading`).
 * - **A reading** of a tile X against today's tile T (the table's choice for the case's shape,
 *   `gemmShapeGeometry`) in one record: the gain `T.perDispatch ÷ X.perDispatch` (each the least of
 *   its aligned interleaved rounds ÷ reps) and both rows' gaps (second-smallest round ÷ smallest,
 *   `gemvRoundsGap`). X is **not slower than today** in that reading when
 *   `gain × margin × gap(X) × gap(T) ≥ 1`: the GEMV rule's bar of a gain, `margin × gaps`, read as a
 *   bound on a loss. The margin is `GEMV_MARGIN` (×1.065) by default.
 *
 * What stays the tile derivation's ×1.05 (`DEFAULT_MIN_SPEEDUP` of derive.ts, the `minSpeedup`
 * option): not a bar per case — the rule is one choice over every shape — but the total. A pair of
 * integers qualifies when, on every case of the set, the tile it picks is identical to the default
 * (in every record that timed it, no failure), is measured against today's tile (evidence as above)
 * and is not slower than today in any reading that is evidence; and when the census-weighted GEMM time
 * of the set under it is at most `1 ÷ minSpeedup` of today's (each case weighted by its
 * `censusCount` × today's time, its time divided by the geometric mean of its gains, a gain of 1
 * where the pair picks today's tile). Among the pairs that qualify, the lowest weighted time, ties to
 * the smaller pair; none qualifying, the table carries no field.
 *
 * The pairs tried are the grid of the recon of 2026-11-28 ({@link GEMM_GRID_LARGEST_MIN_WORKGROUPS}
 * × {@link GEMM_GRID_MIN_WORKGROUPS}); the ladder, the K floor and the padding bound are the code's
 * (gemm-grid.ts) and are written into the field with the two integers.
 *
 * SPEC: docs/spec/geometry-profiles.md#deriving-the-grid-fill-rule
 */

import type { GemmGeometry } from "../kernels/gemm-geometry.ts";
import {
  assertGeometryProfile,
  type GemmGridRule,
  gemmGridStep,
  gemmShapeGeometry,
  type GeometryProfile,
} from "../kernels/geometry-profile.ts";
import { type BmmCase, caseShape, gemmCaseShape, GRID_CASES, type LinearCase } from "./cases.ts";
import { DEFAULT_MIN_SPEEDUP } from "./derive.ts";
import {
  candidateSetCaseSetIds,
  candidateSetDefaultKernelsIds,
  sweepCandidateKernelsId,
} from "./fingerprint.ts";
import { GEMM_GRID_LADDER, GEMM_GRID_MAX_PADDING, GEMM_GRID_MIN_K, GRID_SET } from "./gemm-grid.ts";
import { gemmCandidate } from "./geometries.ts";
import {
  gemvMarginOf,
  gemvRecordReading,
  gemvRoundsGap,
  gemvSteadiness,
  type GemvSteadiness,
} from "./gemv-rule.ts";
import { REPORT_FORMAT } from "./report.ts";

/** The least workgroup counts of the largest tile's grid the derivation tries. */
export const GEMM_GRID_LARGEST_MIN_WORKGROUPS: readonly number[] = [
  128, 192, 256, 288, 320, 384, 480,
];

/** The least workgroup counts of a smaller tile's grid the derivation tries. */
export const GEMM_GRID_MIN_WORKGROUPS: readonly number[] = [32, 48, 64, 96, 128];

/** One grid record given to the derivation. */
export type GemmGridInput = {
  readonly report: unknown;
  /** The record's name (a file path or the like). */
  readonly path: string;
  /** The SHA-256 of the record's bytes (two records with one value are refused). */
  readonly sha256: string;
};

export type DeriveGemmGridOptions = {
  /** The table the field is derived for: its choice for each case's shape is today's tile. */
  readonly table: GeometryProfile;
  /** The least census-weighted gain over today's table (default ×1.05). */
  readonly minSpeedup?: number;
  /** The margin of a reading's bar (default `GEMV_MARGIN`, ×1.065). */
  readonly margin?: number;
};

/** The fields of a grid row the derivation reads. */
type GridRow = {
  readonly caseId: string;
  readonly geometry: string;
  readonly isDefault: boolean;
  readonly rounds: readonly number[];
  readonly perDispatch?: number;
  readonly identicalToDefault?: boolean;
  readonly error?: string;
};

/** One grid record as the derivation reads it. */
type GridSource = {
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly adapter: GridAdapter;
  readonly steadiness: GemvSteadiness;
  readonly reading: "one-record" | "two-record";
  /** Case id → geometry name → row. */
  readonly rows: ReadonlyMap<string, ReadonlyMap<string, GridRow>>;
};

type GridAdapter = {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
};

const ADAPTER_FIELDS = ["vendor", "architecture", "device", "description"] as const;

/**
 * The fields a table's provenance is compared by, as the GEMV derivation compares them: not the
 * description, which under Node on Dawn is the driver's version string, so that a table takes
 * records made after a driver update. The records of one derivation still agree on all four.
 */
const PROVENANCE_ADAPTER_FIELDS = ["vendor", "architecture", "device"] as const;

/** A record of the derivation, as its verdicts name it. */
export type GemmGridRecord = {
  readonly path: string;
  readonly sha256: string;
  readonly date: string;
  readonly steadiness: GemvSteadiness;
  readonly reading: "one-record" | "two-record";
};

/** One record's reading of a case's tile against today's tile. */
export type GemmGridReading = {
  readonly record: string;
  /** Today's least round ÷ the tile's (above 1: the tile is faster). */
  readonly gain: number;
  readonly gap: number;
  readonly todayGap: number;
  /** Whether the reading is evidence (a steady record, or beside a second reading). */
  readonly evidence: boolean;
};

/** What a pair of integers does on one case. */
export type GemmGridCaseOutcome = {
  readonly caseId: string;
  readonly censusCount: number;
  /** The ladder step the pair takes (0: the largest tile). */
  readonly step: number;
  readonly tile: string;
  readonly today: string;
  /** The geometric mean of the evidence readings' gains (1 where the tile is today's). */
  readonly gain?: number;
  /** Today's time per dispatch (the geometric mean over the records that timed it), the weight's. */
  readonly todayTime?: number;
  /** Why the pair is refused on the case (absent when it holds there). */
  readonly refused?: string;
  readonly readings: readonly GemmGridReading[];
};

/** The verdict on one pair of integers. */
export type GemmGridPairVerdict = {
  /** `keepLargest.minWorkgroups`. */
  readonly largestMinWorkgroups: number;
  readonly minWorkgroups: number;
  readonly qualifies: boolean;
  /** The census-weighted GEMM time under the pair ÷ today's (absent when a case is not measured). */
  readonly weightedTime?: number;
  /** The first reason the pair does not qualify. */
  readonly reason?: string;
  readonly cases: readonly GemmGridCaseOutcome[];
};

export type GemmGridDerivation = {
  /** The derived field (absent: no pair qualifies — the table carries no field). */
  readonly gemmGrid?: GemmGridRule;
  /** The chosen pair, or, when none qualifies, the measured pair with the lowest weighted time. */
  readonly shown?: GemmGridPairVerdict;
  readonly pairs: readonly GemmGridPairVerdict[];
  readonly records: readonly GemmGridRecord[];
  readonly rule: {
    readonly margin: number;
    readonly minSpeedup: number;
    readonly ladder: readonly string[];
    readonly minK: number;
    readonly maxPadding: number;
  };
  readonly verdicts: readonly string[];
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const stringAt = (record: Record<string, unknown>, key: string, where: string): string => {
  const value = record[key];
  if (typeof value !== "string") throw new Error(`${where}: ${key} is not a string`);
  return value;
};

const gridCases = GRID_CASES.filter(
  (sweepCase): sweepCase is LinearCase | BmmCase =>
    sweepCase.op === "linear" || sweepCase.op === "bmm",
);

/** A row's tile (its four integers and, where it has one, its skeleton form). */
const geometryOf = (value: unknown, where: string): GemmGeometry => {
  if (!isRecord(value)) throw new Error(`${where}: geometryParams is missing`);
  const count = (key: string): number => {
    const field = value[key];
    if (typeof field !== "number" || !Number.isInteger(field) || field < 1) {
      throw new Error(`${where}: geometryParams.${key} is not a positive integer`);
    }
    return field;
  };
  const { form } = value;
  if (form !== undefined && form !== "tb-un" && form !== "tas-tb-un") {
    throw new Error(`${where}: geometryParams.form is ${JSON.stringify(form)}`);
  }
  return {
    regM: count("regM"),
    regN: count("regN"),
    wgX: count("wgX"),
    wgY: count("wgY"),
    ...(form === undefined ? {} : { form }),
  };
};

/** One row of a grid record (the `unknown` boundary). */
const parseGridRow = (
  value: unknown,
  where: string,
): GridRow & { readonly geometryParams: GemmGeometry } => {
  if (!isRecord(value)) throw new Error(`${where} is not an object`);
  const caseId = stringAt(value, "caseId", where);
  const sweepCase = gridCases.find(({ id }) => id === caseId);
  if (sweepCase === undefined) {
    throw new Error(`${where}: case ${caseId} is not in the grid set's table`);
  }
  if (value.op !== sweepCase.op || value.shape !== caseShape(sweepCase)) {
    throw new Error(`${where}: case ${caseId} has another op or shape than the table's`);
  }
  if (value.screened !== undefined || value.skipped !== undefined) {
    throw new Error(`${where}: a grid row is never screened or skipped`);
  }
  if (typeof value.isDefault !== "boolean") throw new Error(`${where}: isDefault is not a boolean`);
  const geometryParams = geometryOf(value.geometryParams, where);
  const error = value.error;
  if (error !== undefined && typeof error !== "string") {
    throw new Error(`${where}: error is not a string`);
  }
  const rounds = value.rounds ?? [];
  if (
    !Array.isArray(rounds) ||
    !rounds.every((round) => typeof round === "number" && Number.isFinite(round) && round >= 0)
  ) {
    throw new Error(`${where}: rounds is not an array of non-negative numbers`);
  }
  const { perDispatch, identicalToDefault } = value;
  if (perDispatch !== undefined && !(typeof perDispatch === "number" && perDispatch > 0)) {
    throw new Error(`${where}: perDispatch is not a positive number`);
  }
  if (identicalToDefault !== undefined && typeof identicalToDefault !== "boolean") {
    throw new Error(`${where}: identicalToDefault is not a boolean`);
  }
  return {
    caseId,
    geometry: stringAt(value, "geometry", where),
    isDefault: value.isDefault,
    rounds,
    ...(perDispatch === undefined ? {} : { perDispatch }),
    ...(identicalToDefault === undefined ? {} : { identicalToDefault }),
    ...(error === undefined ? {} : { error }),
    geometryParams,
  };
};

/**
 * Reads a grid record (the `unknown` boundary): a `karume-geometry-sweep/4` record of the grid set,
 * timed by GPU timestamps that were not quantised, whose case set, default kernels and measured
 * candidates' kernels are the current runtime's, with one default row per case. Anything else
 * throws, naming the record.
 */
const parseGridRecord = ({ report, path, sha256 }: GemmGridInput): GridSource => {
  if (!isRecord(report)) throw new Error(`${path}: the JSON is not an object`);
  if (report.format !== REPORT_FORMAT) {
    throw new Error(`${path}: format is not ${REPORT_FORMAT} (${String(report.format)})`);
  }
  const settings = isRecord(report.settings) ? report.settings : {};
  if (settings.candidateSet !== GRID_SET) {
    throw new Error(
      `${path}: settings.candidateSet is ${JSON.stringify(settings.candidateSet)}, not ${GRID_SET} (the grid-fill rule is derived from grid records only)`,
    );
  }
  const timing = isRecord(report.gpuTiming) ? report.gpuTiming : {};
  if (timing.unit !== "ns" && timing.unit !== "deno-raw-tick") {
    throw new Error(
      `${path}: gpuTiming.unit is ${JSON.stringify(timing.unit)} (the rule reads GPU timestamps, ns or deno-raw-tick)`,
    );
  }
  if (timing.quantized !== false) {
    throw new Error(
      `${path}: gpuTiming.quantized is ${JSON.stringify(timing.quantized)} (rounded timestamps move a short dispatch's least round by a whole step)`,
    );
  }
  if (!isRecord(report.adapter)) throw new Error(`${path}: adapter is missing`);
  const adapter: GridAdapter = {
    vendor: stringAt(report.adapter, "vendor", `${path} adapter`),
    architecture: stringAt(report.adapter, "architecture", `${path} adapter`),
    device: stringAt(report.adapter, "device", `${path} adapter`),
    description: stringAt(report.adapter, "description", `${path} adapter`),
  };
  const expect = (key: string, got: unknown, want: string | undefined, what: string): void => {
    if (got !== want) {
      throw new Error(
        `${path}: ${key} ${JSON.stringify(got)} differs from the current runtime's ${JSON.stringify(want)} (${what} — measure the grid set again)`,
      );
    }
  };
  expect(
    "caseSets.gemm",
    isRecord(report.caseSets) ? report.caseSets.gemm : undefined,
    candidateSetCaseSetIds(GRID_SET, ["gemm"]).gemm,
    "the grid set's cases",
  );
  expect(
    "defaultKernels.gemm",
    isRecord(report.defaultKernels) ? report.defaultKernels.gemm : undefined,
    candidateSetDefaultKernelsIds(GRID_SET, ["gemm"]).gemm,
    "the kernels of the default rows",
  );
  if (!Array.isArray(report.rows)) throw new Error(`${path}: rows is not an array`);
  const parsed = report.rows.map((row, index) => parseGridRow(row, `${path} rows[${index}]`));
  if (typeof report.dp4a !== "boolean") throw new Error(`${path}: dp4a is not a boolean`);
  expect(
    "candidateKernels",
    report.candidateKernels,
    sweepCandidateKernelsId(
      parsed.map(({ caseId, geometryParams, error }) => ({
        caseId,
        geometryParams,
        ...(error === undefined ? {} : { error }),
      })),
      report.dp4a,
    ),
    "the kernels of the measured tiles",
  );
  const rows = new Map<string, Map<string, GridRow>>();
  for (const { geometryParams: _geometry, ...row } of parsed) {
    const byGeometry = rows.get(row.caseId) ?? new Map<string, GridRow>();
    if (byGeometry.has(row.geometry)) {
      throw new Error(`${path}: ${row.caseId} has ${row.geometry} twice`);
    }
    byGeometry.set(row.geometry, row);
    rows.set(row.caseId, byGeometry);
  }
  for (const [caseId, byGeometry] of rows) {
    const defaults = [...byGeometry.values()].filter((row) => row.isDefault).length;
    if (defaults !== 1) {
      throw new Error(`${path}: ${caseId} has ${defaults} default rows (must be exactly 1)`);
    }
  }
  const spreads = [...rows.values()].flatMap((byGeometry) => {
    const reference = [...byGeometry.values()].find((row) => row.isDefault);
    const positive = (reference?.rounds ?? []).filter((round) => round > 0);
    return positive.length < 2
      ? []
      : [{ baselineSpread: Math.max(...positive) / Math.min(...positive) }];
  });
  const steadiness = gemvSteadiness(spreads);
  const reading = gemvRecordReading(steadiness, timing);
  if (reading === "refused") throw new Error(`${path}: its timing is no evidence`);
  return {
    path,
    sha256,
    date: stringAt(report, "date", path),
    adapter,
    steadiness,
    reading,
    rows,
  };
};

const geometricMean = (values: readonly number[]): number =>
  Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);

const ratio = (value: number): string => `×${value.toFixed(3)}`;

/** The rule a pair of integers gives, over the code's ladder and shape constants. */
const pairRule = (largestMinWorkgroups: number, minWorkgroups: number): GemmGridRule => ({
  ladder: GEMM_GRID_LADDER,
  keepLargest: { minWorkgroups: largestMinWorkgroups, minK: GEMM_GRID_MIN_K },
  minWorkgroups,
  maxPadding: GEMM_GRID_MAX_PADDING,
});

/** A usable row: timed, not failed. */
const timedRow = (row: GridRow | undefined): row is GridRow & { readonly perDispatch: number } =>
  row !== undefined && row.error === undefined && row.perDispatch !== undefined;

/**
 * What a tile does on one case against today's tile, over the records: refused when a record found
 * it not identical or failed, unmeasured when it has no reading that is evidence, slower when a
 * reading that is evidence breaks the bar, else its gain.
 */
const caseOutcome = (
  sweepCase: LinearCase | BmmCase,
  step: number,
  todayGeometry: GemmGeometry,
  sources: readonly GridSource[],
  margin: number,
): GemmGridCaseOutcome => {
  const tile = gemmCandidate(GEMM_GRID_LADDER[step]).name;
  const today = gemmCandidate(todayGeometry).name;
  const head = { caseId: sweepCase.id, censusCount: sweepCase.censusCount, step, tile, today };
  const todayRows = sources.flatMap((source) => {
    const row = source.rows.get(sweepCase.id)?.get(today);
    return timedRow(row) ? [row] : [];
  });
  if (todayRows.length === 0) {
    return {
      ...head,
      refused: `today's tile ${today} has no time in ${sweepCase.id}`,
      readings: [],
    };
  }
  const todayTime = geometricMean(todayRows.map((row) => row.perDispatch));
  if (tile === today) return { ...head, gain: 1, todayTime, readings: [] };
  for (const source of sources) {
    const row = source.rows.get(sweepCase.id)?.get(tile);
    if (row?.error !== undefined) {
      return { ...head, todayTime, refused: `${tile} failed in ${source.path}`, readings: [] };
    }
    if (row !== undefined && !row.isDefault && row.identicalToDefault !== true) {
      return {
        ...head,
        todayTime,
        refused: `${tile} is not identical to the default in ${source.path}`,
        readings: [],
      };
    }
  }
  const raw = sources.flatMap((source) => {
    const byGeometry = source.rows.get(sweepCase.id);
    const ours = byGeometry?.get(tile);
    const theirs = byGeometry?.get(today);
    if (!timedRow(ours) || !timedRow(theirs)) return [];
    const gap = gemvRoundsGap(ours.rounds);
    const todayGap = gemvRoundsGap(theirs.rounds);
    if (gap === undefined || todayGap === undefined) return [];
    return [
      {
        record: source.path,
        gain: theirs.perDispatch / ours.perDispatch,
        gap,
        todayGap,
        oneRecord: source.reading === "one-record",
      },
    ];
  });
  const enough = raw.length >= 2 || raw.some(({ oneRecord }) => oneRecord);
  const readings: GemmGridReading[] = raw.map(({ record, gain, gap, todayGap }) => ({
    record,
    gain,
    gap,
    todayGap,
    evidence: enough,
  }));
  if (!enough) {
    return {
      ...head,
      todayTime,
      refused:
        raw.length === 0
          ? `${tile} is not timed against ${today} in ${sweepCase.id}`
          : `${tile} against ${today} in ${sweepCase.id} is read by one record that is not steady`,
      readings,
    };
  }
  const slower = readings.find(({ gain, gap, todayGap }) => gain * margin * gap * todayGap < 1);
  const gain = geometricMean(readings.map((reading) => reading.gain));
  if (slower !== undefined) {
    return {
      ...head,
      gain,
      todayTime,
      refused: `${sweepCase.id}: ${tile} is slower than today's ${today} in ${slower.record} (${ratio(slower.gain)}, below 1 ÷ (${ratio(margin)} × gaps ${ratio(slower.gap)} × ${ratio(slower.todayGap)}) = ${ratio(1 / (margin * slower.gap * slower.todayGap))})`,
      readings,
    };
  }
  return { ...head, gain, todayTime, readings };
};

/** The verdict on one pair over every case of the grid set. */
const pairVerdict = (
  largestMinWorkgroups: number,
  minWorkgroups: number,
  table: GeometryProfile,
  sources: readonly GridSource[],
  margin: number,
  minSpeedup: number,
): GemmGridPairVerdict => {
  const rule = pairRule(largestMinWorkgroups, minWorkgroups);
  const cases = gridCases.map((sweepCase) => {
    const shape = gemmCaseShape(sweepCase);
    return caseOutcome(
      sweepCase,
      gemmGridStep(rule, shape),
      gemmShapeGeometry(table, shape),
      sources,
      margin,
    );
  });
  const head = { largestMinWorkgroups, minWorkgroups, cases };
  const weighed = cases.every((entry) => entry.gain !== undefined && entry.todayTime !== undefined);
  const weightedTime = weighed
    ? cases.reduce(
        (sum, entry) => sum + (entry.censusCount * (entry.todayTime ?? 0)) / (entry.gain ?? 1),
        0,
      ) / cases.reduce((sum, entry) => sum + entry.censusCount * (entry.todayTime ?? 0), 0)
    : undefined;
  const refusal = cases.find((entry) => entry.refused !== undefined);
  const timeHead = weightedTime === undefined ? {} : { weightedTime };
  if (refusal !== undefined) {
    return { ...head, ...timeHead, qualifies: false, reason: refusal.refused };
  }
  if (weightedTime === undefined) {
    return { ...head, qualifies: false, reason: "a case has no weight" };
  }
  if (weightedTime > 1 / minSpeedup) {
    return {
      ...head,
      weightedTime,
      qualifies: false,
      reason: `the weighted gain ${ratio(1 / weightedTime)} is below ${ratio(minSpeedup)}`,
    };
  }
  return { ...head, weightedTime, qualifies: true };
};

/**
 * The adapter of the records (they agree on its four fields), checked against the table's: its match
 * fits the adapter, and its provenance names the adapter's vendor, architecture and device.
 */
const recordsAdapter = (sources: readonly GridSource[], table: GeometryProfile): GridAdapter => {
  const [first] = sources;
  for (const source of sources) {
    for (const field of ADAPTER_FIELDS) {
      if (source.adapter[field] !== first.adapter[field]) {
        throw new Error(
          `${source.path}: adapter ${field} ${JSON.stringify(source.adapter[field])} differs from ${first.path}'s ${JSON.stringify(first.adapter[field])} (the integers are one device's)`,
        );
      }
    }
  }
  const { match, provenance } = table;
  for (const field of ["vendor", "architecture", "description"] as const) {
    const wanted = match?.[field];
    if (wanted !== undefined && first.adapter[field] !== wanted) {
      throw new Error(
        `${first.path}: adapter ${field} ${JSON.stringify(first.adapter[field])} is not table '${table.id}''s match ${JSON.stringify(wanted)}`,
      );
    }
  }
  if (provenance?.adapters !== undefined) {
    throw new Error(
      `Table '${table.id}' is a shared table: it takes no grid-fill rule (the integers are one device's saturation points)`,
    );
  }
  if (provenance?.adapter !== undefined) {
    for (const field of PROVENANCE_ADAPTER_FIELDS) {
      if (provenance.adapter[field] !== first.adapter[field]) {
        throw new Error(
          `Table '${table.id}': adapter ${field} ${JSON.stringify(provenance.adapter[field])} differs from the grid records' ${JSON.stringify(first.adapter[field])} (a table takes the rule of its own device)`,
        );
      }
    }
  }
  return first.adapter;
};

const caseLine = (entry: GemmGridCaseOutcome): string =>
  `  - ${entry.caseId} (×${entry.censusCount}): ${entry.tile === entry.today ? `today's ${entry.today}` : `${entry.tile} against today's ${entry.today}`}${
    entry.gain === undefined ? "" : ` ${ratio(entry.gain)}`
  }${
    entry.readings.length === 0
      ? ""
      : ` (${entry.readings
          .map(
            (reading) =>
              `${reading.record} ${ratio(reading.gain)}, gaps ${ratio(reading.gap)} / ${ratio(reading.todayGap)}`,
          )
          .join("; ")})`
  }${entry.refused === undefined ? "" : ` — ${entry.refused}`}`;

const pairName = (verdict: GemmGridPairVerdict): string =>
  `(${verdict.largestMinWorkgroups}, ${verdict.minWorkgroups})`;

/**
 * The verdict on the grid-fill rule from grid records, for `table` (a pure function — the rule of
 * the module doc). Refuses, throwing: no record, a record given twice, records of two adapters, a
 * record of the grid set that does not stand for the current runtime (format, set, timing, case
 * set, kernels, rows), and a table of another adapter or a shared table.
 */
export const deriveGemmGrid = (
  inputs: readonly GemmGridInput[],
  options: DeriveGemmGridOptions,
): GemmGridDerivation => {
  if (inputs.length === 0) throw new Error("No grid records");
  assertGeometryProfile(options.table);
  const seen = new Map<string, string>();
  for (const input of inputs) {
    const twin = seen.get(input.sha256);
    if (twin !== undefined) {
      throw new Error(`The same grid record is given twice (${twin} and ${input.path})`);
    }
    seen.set(input.sha256, input.path);
  }
  const sources = inputs.map(parseGridRecord);
  recordsAdapter(sources, options.table);
  const margin = gemvMarginOf(options.margin);
  const minSpeedup = options.minSpeedup ?? DEFAULT_MIN_SPEEDUP;
  if (!(Number.isFinite(minSpeedup) && minSpeedup >= 1)) {
    throw new Error(`minSpeedup must be a finite number of at least 1 (${minSpeedup})`);
  }
  const pairs = GEMM_GRID_LARGEST_MIN_WORKGROUPS.flatMap((largest) =>
    GEMM_GRID_MIN_WORKGROUPS.map((smaller) =>
      pairVerdict(largest, smaller, options.table, sources, margin, minSpeedup),
    ),
  );
  const byTime = (a: GemmGridPairVerdict, b: GemmGridPairVerdict): number =>
    (a.weightedTime ?? Infinity) - (b.weightedTime ?? Infinity) ||
    a.largestMinWorkgroups - b.largestMinWorkgroups ||
    a.minWorkgroups - b.minWorkgroups;
  const chosen = pairs.filter((pair) => pair.qualifies).toSorted(byTime)[0];
  const shown =
    chosen ?? pairs.filter((pair) => pair.weightedTime !== undefined).toSorted(byTime)[0];
  const gemmGrid =
    chosen === undefined ? undefined : pairRule(chosen.largestMinWorkgroups, chosen.minWorkgroups);
  const records = sources.map(({ path, sha256, date, steadiness, reading }) => ({
    path,
    sha256,
    date,
    steadiness,
    reading,
  }));
  const recordText = records
    .map(
      ({ path, steadiness, reading }) =>
        `${path} ${steadiness.class}${steadiness.share === undefined ? "" : ` (the default spread beyond ×1.05 in ${steadiness.unsteadyGroups} of ${steadiness.groups} cases)`}, ${reading === "one-record" ? "evidence alone" : "evidence beside a second record"}`,
    )
    .join("; ");
  const qualifying = pairs.filter((pair) => pair.qualifies).length;
  const head = `gemmGrid (the grid-fill rule of the GEMM; ${gridCases.length} cases; ${qualifying} of ${pairs.length} pairs qualify; ${recordText})`;
  const verdicts = [
    chosen === undefined
      ? `- ${head}: carries no field (no pair qualifies${shown === undefined ? "" : `; the lowest weighted time ${ratio(shown.weightedTime ?? 1)} is ${pairName(shown)}: ${shown.reason ?? ""}`})`
      : `- ${head}: adopted ${pairName(chosen)} — weighted GEMM time ${ratio(chosen.weightedTime ?? 1)} of today's (gain ${ratio(1 / (chosen.weightedTime ?? 1))} ≥ ${ratio(minSpeedup)})`,
    ...pairs.map(
      (pair) =>
        `  - ${pairName(pair)}: ${pair.qualifies ? "qualifies" : "refused"}${pair.weightedTime === undefined ? "" : `, weighted time ${ratio(pair.weightedTime)}`}${pair.reason === undefined ? "" : ` — ${pair.reason}`}`,
    ),
    ...(shown === undefined
      ? []
      : [
          `  - the cases of ${pairName(shown)} (a gain is today's time ÷ the tile's, the geometric mean over the readings):`,
          ...shown.cases.map(caseLine).map((line) => `  ${line}`),
        ]),
  ];
  return {
    ...(gemmGrid === undefined ? {} : { gemmGrid }),
    ...(shown === undefined ? {} : { shown }),
    pairs,
    records,
    rule: {
      margin,
      minSpeedup,
      ladder: GEMM_GRID_LADDER.map((geometry) => gemmCandidate(geometry).name),
      minK: GEMM_GRID_MIN_K,
      maxPadding: GEMM_GRID_MAX_PADDING,
    },
    verdicts,
  };
};
