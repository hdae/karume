/**
 * The K-88 evaluation sweep on this machine (opt-in: `KARUME_K88_SWEEP=<dir>`): `runGeometrySweep`
 * of `@karume/runtime/tune` with the candidate set `"k88"` — the f32 GEMM skeleton forms and the
 * conv2d counter decode (perf-ledger K-88, the optional geometry fields `form` / `decode`) — on that
 * set's cases (`geometrySweepCases("k88")`: the gemma4 E2B prefill, umT5, siglip2 and guard linears,
 * the square matmul, and the Wan2.2 and anima VAE conv2d), one call per case (each acquires and
 * destroys its own device) with idle time between them for cooling (`KARUME_K88_IDLE_SECONDS`,
 * default 60). `KARUME_K88_CASES=<id>,<id>` narrows the cases (a quick check of the path; a unit's sum
 * then covers only the cases run), `KARUME_K88_ROUNDS` sets the rounds per geometry (default 5).
 *
 * It needs no model asset: the cases are shapes, and the sweep fills them with seeded values. Every
 * per-case record is written under the directory the variable names, with a summary: per case this
 * adapter's table geometry (the record's `geometryProfile`; the default profile when it names none)
 * and, of that same geometry, the fastest form, the decode alone and the fastest decode + form, each
 * as table time ÷ variant time (> 1: the variant is faster); and per unit of `censusGroup` the
 * census-weighted sum (each case's `censusCount` × its time per dispatch) of the table geometry and
 * of each variant.
 *
 * It asserts what the sweep is the evidence for: GPU timestamps, no failed row, and every candidate's
 * output equal to the default geometry's as words (`identicalToDefault`); never a verdict on times
 * (they go to the summary and the log). The ratios are within one run of one machine (the sweep's
 * protocol rule 5); the sweep measures the candidates of a case one after another (no interleaving
 * inside a round), so each case's drift (the default geometry re-measured at its end ÷ at its
 * start) is part of the summary.
 *
 * NOTE: this summary and the K-88 summary of the playground's `/geometry-sweep` page
 * (apps/playground/src/models/geometry-sweep/k88.ts) are computed separately from the same record
 * fields; the e2e tests cannot import the playground.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable or without an adapter, the case SKIPs with
 * the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import {
  BUILTIN_GEOMETRY_PROFILES,
  type GemmGeometry,
  type GeometryProfile,
} from "@karume/runtime";
import {
  type GeometrySweepCase,
  type GeometrySweepReport,
  type GeometrySweepRow,
  geometrySweepCases,
  runGeometrySweep,
} from "@karume/runtime/tune";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const DIRECTORY = process.env["KARUME_K88_SWEEP"];
const IDLE_SECONDS = Number(process.env["KARUME_K88_IDLE_SECONDS"] ?? "60");
const ROUNDS = Number(process.env["KARUME_K88_ROUNDS"] ?? "5");
const CASE_FILTER = (process.env["KARUME_K88_CASES"] ?? "")
  .split(",")
  .map((id) => id.trim())
  .filter((id) => id !== "");

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_K88_SWEEP is not set (opt-in: KARUME_K88_SWEEP=<dir> writes the records there)"
    : !Number.isInteger(IDLE_SECONDS) || IDLE_SECONDS < 0
      ? `KARUME_K88_IDLE_SECONDS is not a non-negative integer (${String(IDLE_SECONDS)})`
      : !Number.isInteger(ROUNDS) || ROUNDS < 1
        ? `KARUME_K88_ROUNDS is not a positive integer (${String(ROUNDS)})`
        : !GPU_AVAILABLE
          ? "no GPU adapter"
          : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the K-88 GEMM sweep SKIPs: ${skipReason}`);
}

const sleep = (seconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, seconds * 1000));

type Fields = Pick<GemmGeometry, "form" | "decode">;

/** Whether a row measured exactly this f32 geometry with exactly these fields. */
const isGeometry = (row: GeometrySweepRow, base: GemmGeometry, fields: Fields): boolean => {
  const params = row.geometryParams;
  return (
    !("tileK" in params) &&
    params.regM === base.regM &&
    params.regN === base.regN &&
    params.wgX === base.wgX &&
    params.wgY === base.wgY &&
    params.form === fields.form &&
    params.decode === fields.decode
  );
};

/** A row that ran and whose output matched the default (the default row itself included). */
const usable = (
  row: GeometrySweepRow | undefined,
): row is GeometrySweepRow & { perDispatch: number } =>
  row !== undefined &&
  row.error === undefined &&
  row.perDispatch !== undefined &&
  (row.isDefault || row.identicalToDefault === true);

const ms = (ns: number | undefined): number | undefined =>
  ns === undefined ? undefined : ns / 1e6;

/**
 * The geometry this adapter's table gives a k88 case: the sweep's default row (the default
 * profile's geometry) when the table is the default one (`profile` undefined); else the table's
 * `gemmRows` rule of the case's M (read from the row's shape, `M{m} …` — the spelling derive.ts's
 * slotOf reads too), or its conv2d field of the class the default row's tile edge names (64 / 32
 * rows).
 */
const tableGeometry = (
  profile: GeometryProfile | undefined,
  sweepCase: GeometrySweepCase,
  rows: readonly GeometrySweepRow[],
): GemmGeometry => {
  const defaultRow = rows.find((row) => row.isDefault);
  assert(defaultRow !== undefined, `${sweepCase.id}: no default row`);
  const params = defaultRow.geometryParams;
  assert(!("tileK" in params), `${sweepCase.id}: an i8a8 default row`);
  if (profile === undefined) return params;
  if (sweepCase.op === "conv2d") {
    return params.regM * params.wgY === 32 ? profile.conv2d.rows32 : profile.conv2d.rows64;
  }
  const m = Number(/^M(\d+) /.exec(defaultRow.shape)?.[1]);
  const rule = profile.gemmRows.find((entry) => m <= entry.maxRows);
  assert(rule !== undefined, `${sweepCase.id}: no gemmRows rule for M ${m}`);
  return rule.geometry;
};

const FORMS = ["tb-un", "tas-tb-un"] as const;

describe.skipIf(skipReason !== undefined)("the K-88 GEMM sweep on this machine (opt-in)", () => {
  it("every k88 candidate matches the default geometry's words on every case, timed with GPU timestamps", async () => {
    const directory = DIRECTORY ?? "";
    await mkdir(directory, { recursive: true });
    const cases = geometrySweepCases("k88").filter(
      ({ id }) => CASE_FILTER.length === 0 || CASE_FILTER.includes(id),
    );
    assert(cases.length > 0, `KARUME_K88_CASES names no k88 case (${CASE_FILTER.join(", ")})`);
    const reports: GeometrySweepReport[] = [];
    for (const [index, sweepCase] of cases.entries()) {
      if (index > 0 && IDLE_SECONDS > 0) await sleep(IDLE_SECONDS);
      const report = await runGeometrySweep({
        candidateSet: "k88",
        ops: [sweepCase.op],
        cases: [sweepCase.id],
        rounds: ROUNDS,
        onProgress: (progress) => {
          if (progress.kind === "case") {
            console.log(
              `[k88] ${progress.summary.caseId}: ${(progress.summary.elapsedMs / 1000).toFixed(0)} s, drift ${
                progress.summary.defaultRepeat?.driftRatio.toFixed(3) ??
                progress.summary.defaultRepeatError
              }`,
            );
          }
        },
      });
      await writeTextFile(
        `${directory}/geometry-sweep_k88_${sweepCase.id}.json`,
        `${JSON.stringify(report, null, 2)}\n`,
      );
      reports.push(report);
    }

    const [first] = reports;
    assert(first !== undefined);
    const tableId = first.geometryProfile ?? "default";
    // This adapter's table (the embedded table a Session selects); the default table is the sweep's
    // default row
    const profile = BUILTIN_GEOMETRY_PROFILES.find((entry) => entry.id === tableId);

    const perCase = reports.map((report, index) => {
      const entry = cases[index];
      assert(entry !== undefined);
      const rows = report.rows;
      const table = tableGeometry(profile, entry, rows);
      const timeOf = (fields: Fields): number | undefined => {
        const row = rows.find((candidate) => isGeometry(candidate, table, fields));
        return usable(row) ? row.perDispatch : undefined;
      };
      const fastest = (values: readonly (number | undefined)[]): number | undefined => {
        const known = values.filter((value): value is number => value !== undefined);
        return known.length === 0 ? undefined : Math.min(...known);
      };
      const tableNs = timeOf({});
      const formNs = fastest(FORMS.map((form) => timeOf({ form })));
      const conv = entry.op === "conv2d";
      const decodeNs = conv ? timeOf({ decode: "counter" }) : undefined;
      const bothNs = conv
        ? fastest(FORMS.map((form) => timeOf({ form, decode: "counter" })))
        : undefined;
      const ratio = (variant: number | undefined): number | undefined =>
        tableNs === undefined || variant === undefined ? undefined : tableNs / variant;
      const summary = report.cases[0];
      return {
        caseId: entry.id,
        censusCount: entry.censusCount ?? 0,
        censusGroup: entry.censusGroup,
        rows: rows.length,
        errors: rows
          .filter((row) => row.error !== undefined)
          .map((row) => `${row.geometry}: ${row.error ?? ""}`),
        mismatches: rows
          .filter((row) => row.identicalToDefault === false)
          .map((row) => row.geometry),
        identical: rows.every((row) => row.error === undefined && row.identicalToDefault !== false),
        driftRatio: summary?.defaultRepeat?.driftRatio,
        table: { geometry: table, ms: ms(tableNs) },
        formMs: ms(formNs),
        decodeMs: ms(decodeNs),
        bothMs: ms(bothNs),
        formVsTable: ratio(formNs),
        decodeVsTable: ratio(decodeNs),
        bothVsTable: ratio(bothNs),
      };
    });

    // The census-weighted sums per unit (the cases with a count)
    const units = [
      ...new Set(
        perCase.flatMap((entry) => (entry.censusGroup === undefined ? [] : [entry.censusGroup])),
      ),
    ];
    const sums = units.map((unit) => {
      const members = perCase.filter(
        (entry) => entry.censusGroup === unit && entry.censusCount > 0,
      );
      const sum = (pick: (entry: (typeof perCase)[number]) => number | undefined) =>
        members.reduce<number | undefined>((total, entry) => {
          const value = pick(entry);
          return total === undefined || value === undefined
            ? undefined
            : total + entry.censusCount * value;
        }, 0);
      const tableMs = sum((entry) => entry.table.ms);
      const of = (pick: (entry: (typeof perCase)[number]) => number | undefined) => {
        const value = sum(pick);
        return {
          ms: value,
          vsTable: tableMs === undefined || value === undefined ? undefined : tableMs / value,
        };
      };
      const conv = members.some((entry) => entry.decodeMs !== undefined);
      return {
        unit,
        cases: members.map((entry) => entry.caseId),
        nodes: members.reduce((total, entry) => total + entry.censusCount, 0),
        tableMs,
        bestForm: of((entry) => entry.formMs),
        ...(conv
          ? { decode: of((entry) => entry.decodeMs), bestBoth: of((entry) => entry.bothMs) }
          : {}),
      };
    });

    const summary = {
      format: "karume-e2e-k88-gemm-sweep/1",
      adapter: first.adapter,
      userAgent: first.userAgent,
      timing: first.gpuTiming,
      candidateSet: first.settings.candidateSet,
      caseIds: cases.map(({ id }) => id),
      rounds: ROUNDS,
      idleSeconds: IDLE_SECONDS,
      caseSets: first.caseSets,
      table: { id: tableId },
      ratios: "table time ÷ variant time of the same geometry (> 1: the variant is faster)",
      sums,
      cases: perCase,
    };
    await writeTextFile(`${directory}/k88-summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
    for (const entry of sums) console.log(`[k88] sum ${JSON.stringify(entry)}`);
    for (const entry of perCase) {
      const x = (value: number | undefined): string =>
        value === undefined ? "-" : `×${value.toFixed(3)}`;
      console.log(
        `[k88] ${entry.caseId}: table ${entry.table.ms?.toFixed(4) ?? "?"} ms, form ${x(entry.formVsTable)}, decode ${x(entry.decodeVsTable)}, both ${x(entry.bothVsTable)}, drift ${entry.driftRatio?.toFixed(3) ?? "?"}, identical ${String(entry.identical)}`,
      );
    }

    for (const report of reports) assertEquals(report.gpuTiming.unit, "ns", "GPU timestamps");
    for (const entry of perCase) {
      assertEquals(entry.errors, [], `${entry.caseId}: failed rows`);
      assertEquals(entry.mismatches, [], `${entry.caseId}: candidates whose words differ`);
      assert(entry.table.ms !== undefined, `${entry.caseId}: the table geometry was not measured`);
      assert(entry.formMs !== undefined, `${entry.caseId}: no form candidate measured`);
    }
  });
});
