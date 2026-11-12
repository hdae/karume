/**
 * The K-85 evaluation sweep on this machine (opt-in: `KARUME_K85_SWEEP=<dir>`): `runGeometrySweep`
 * of `@karume/runtime/tune` with the candidate set `"k85"` — the vec4 skeleton of the i8a8 linear
 * (perf-ledger K-85) — on that set's cases (`geometrySweepCases("k85")`: the Wan2.2 TI2V-5B DiT
 * linears at 33 frames and the small-M guard), one call per case (each acquires and destroys its
 * own device) with idle time between them for cooling (`KARUME_K85_IDLE_SECONDS`, default 60).
 * `KARUME_K85_CASES=<id>,<id>` narrows the cases (a quick check of the path; the forward sum then
 * covers only the cases run).
 *
 * It needs no model asset: the cases are shapes, and the sweep fills them with seeded values. Every
 * per-case record is written under the directory the variable names, with a summary of the
 * census-weighted sum over one Wan2.2 33f DiT forward (each case's `censusCount` × its time per
 * dispatch) for this adapter's table geometry (the record's `geometryProfile`), the default, the
 * fastest skeleton candidate of each case, and the one skeleton geometry with the smallest sum (a
 * profile field holds one geometry for every M).
 *
 * It asserts what the sweep is the evidence for: GPU timestamps, no failed row, and every candidate's
 * output equal to the default geometry's as words (`identicalToDefault`); never a verdict on times
 * (they go to the summary and the log). The ratios are within one run of one machine (the sweep's
 * protocol rule 5).
 *
 * NOTE: the forward sum here and the K-85 summary of the playground's `/geometry-sweep` page
 * (apps/playground/src/models/geometry-sweep/k85.ts) are computed separately from the same record
 * fields; the e2e tests cannot import the playground.
 *
 * MUST: never run it while other work uses the GPU (the times would measure the contention); run it
 * under `flock /tmp/karume-gpu.lock`. Without the variable or without an adapter, the case SKIPs with
 * the reason (ADR 0005).
 */

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { mkdir, writeTextFile } from "@karume/test-support/fs";
import { BUILTIN_GEOMETRY_PROFILES, type GemmGeometry, type I8a8Geometry } from "@karume/runtime";
import {
  type GeometrySweepReport,
  type GeometrySweepRow,
  geometrySweepCases,
  runGeometrySweep,
} from "@karume/runtime/tune";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";

const DIRECTORY = process.env["KARUME_K85_SWEEP"];
const IDLE_SECONDS = Number(process.env["KARUME_K85_IDLE_SECONDS"] ?? "60");
const CASE_FILTER = (process.env["KARUME_K85_CASES"] ?? "")
  .split(",")
  .map((id) => id.trim())
  .filter((id) => id !== "");

const skipReason =
  DIRECTORY === undefined || DIRECTORY === ""
    ? "KARUME_K85_SWEEP is not set (opt-in: KARUME_K85_SWEEP=<dir> writes the records there)"
    : !Number.isInteger(IDLE_SECONDS) || IDLE_SECONDS < 0
      ? `KARUME_K85_IDLE_SECONDS is not a non-negative integer (${String(IDLE_SECONDS)})`
      : !GPU_AVAILABLE
        ? "no GPU adapter"
        : undefined;
if (skipReason !== undefined) {
  console.warn(`[karume] the K-85 linear sweep SKIPs: ${skipReason}`);
}

const ROUNDS = 5;

const sleep = (seconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, seconds * 1000));

const isI8a8 = (geometry: GemmGeometry | I8a8Geometry): geometry is I8a8Geometry =>
  "tileK" in geometry;

/** Whether a row measured exactly this i8a8 geometry (the skeleton included). */
const isGeometry = (row: GeometrySweepRow, geometry: I8a8Geometry): boolean => {
  const params = row.geometryParams;
  return (
    "tileK" in params &&
    params.regM === geometry.regM &&
    params.regN === geometry.regN &&
    params.wgX === geometry.wgX &&
    params.wgY === geometry.wgY &&
    params.tileK === geometry.tileK &&
    params.skeleton === geometry.skeleton
  );
};

const isSkeleton = (row: GeometrySweepRow): boolean =>
  "tileK" in row.geometryParams && row.geometryParams.skeleton === "vec4";

/** The fastest measured row whose output matched the default (the default row included). */
const fastest = (rows: readonly GeometrySweepRow[]): GeometrySweepRow | undefined =>
  rows
    .filter(
      (row) =>
        row.error === undefined &&
        row.perDispatch !== undefined &&
        (row.isDefault || row.identicalToDefault === true),
    )
    .reduce<GeometrySweepRow | undefined>(
      (best, row) =>
        best?.perDispatch === undefined || (row.perDispatch ?? Infinity) < best.perDispatch
          ? row
          : best,
      undefined,
    );

const ms = (ns: number | undefined): number | undefined =>
  ns === undefined ? undefined : ns / 1e6;

describe.skipIf(skipReason !== undefined)("the K-85 linear sweep on this machine (opt-in)", () => {
  it("every k85 candidate matches the default geometry's words on every case, timed with GPU timestamps", async () => {
    const directory = DIRECTORY ?? "";
    await mkdir(directory, { recursive: true });
    const cases = geometrySweepCases("k85").filter(
      ({ id }) => CASE_FILTER.length === 0 || CASE_FILTER.includes(id),
    );
    assert(cases.length > 0, `KARUME_K85_CASES names no k85 case (${CASE_FILTER.join(", ")})`);
    const reports: GeometrySweepReport[] = [];
    for (const [index, sweepCase] of cases.entries()) {
      if (index > 0 && IDLE_SECONDS > 0) await sleep(IDLE_SECONDS);
      const report = await runGeometrySweep({
        candidateSet: "k85",
        ops: ["i8a8-linear"],
        cases: [sweepCase.id],
        rounds: ROUNDS,
        onProgress: (progress) => {
          if (progress.kind === "case") {
            console.log(
              `[k85] ${progress.summary.caseId}: ${(progress.summary.elapsedMs / 1000).toFixed(0)} s, drift ${
                progress.summary.defaultRepeat?.driftRatio.toFixed(3) ??
                progress.summary.defaultRepeatError
              }`,
            );
          }
        },
      });
      await writeTextFile(
        `${directory}/geometry-sweep_k85_${sweepCase.id}.json`,
        `${JSON.stringify(report, null, 2)}\n`,
      );
      reports.push(report);
    }

    const [first] = reports;
    assert(first !== undefined);
    const tableId = first.geometryProfile ?? "default";
    // This adapter's table (the embedded table a Session selects); the default table is the sweep's
    // default row
    const defaultParams = first.rows.find((row) => row.isDefault)?.geometryParams;
    const table: I8a8Geometry | undefined =
      BUILTIN_GEOMETRY_PROFILES.find((profile) => profile.id === tableId)?.i8a8.linear ??
      (defaultParams !== undefined && isI8a8(defaultParams) ? defaultParams : undefined);
    assert(table !== undefined, `no i8a8 geometry for the table ${tableId}`);

    const perCase = reports.map((report, index) => {
      const entry = cases[index];
      assert(entry !== undefined);
      const rows = report.rows;
      const tableRow = rows.find((row) => isGeometry(row, table));
      const defaultRow = rows.find((row) => row.isDefault);
      const bestSkeleton = fastest(rows.filter(isSkeleton));
      const best = fastest(rows);
      const summary = report.cases[0];
      return {
        caseId: entry.id,
        censusCount: entry.censusCount ?? 0,
        rows: rows.length,
        errors: rows
          .filter((row) => row.error !== undefined)
          .map((row) => `${row.geometry}: ${row.error ?? ""}`),
        mismatches: rows
          .filter((row) => row.identicalToDefault === false)
          .map((row) => row.geometry),
        driftRatio: summary?.defaultRepeat?.driftRatio,
        defaultMs: ms(defaultRow?.perDispatch),
        table: { geometry: tableRow?.geometry, ms: ms(tableRow?.perDispatch) },
        bestSkeleton: { geometry: bestSkeleton?.geometry, ms: ms(bestSkeleton?.perDispatch) },
        best: { geometry: best?.geometry, ms: ms(best?.perDispatch) },
        bestSkeletonVsTable:
          tableRow?.perDispatch !== undefined && bestSkeleton?.perDispatch !== undefined
            ? tableRow.perDispatch / bestSkeleton.perDispatch
            : undefined,
      };
    });

    // The census-weighted sum over one Wan2.2 33f DiT forward (the cases with a count)
    const forward = perCase.filter((entry) => entry.censusCount > 0);
    const sum = (
      pick: (entry: (typeof perCase)[number]) => number | undefined,
    ): number | undefined =>
      forward.reduce<number | undefined>((total, entry) => {
        const value = pick(entry);
        return total === undefined || value === undefined
          ? undefined
          : total + entry.censusCount * value;
      }, 0);
    // The one skeleton geometry with the smallest sum, among those measured identical on every forward case
    const timesOf = (geometry: string): number | undefined =>
      forward.reduce<number | undefined>((total, entry) => {
        const report = reports[perCase.indexOf(entry)];
        const row = report?.rows.find(
          (candidate) =>
            candidate.geometry === geometry &&
            candidate.error === undefined &&
            candidate.identicalToDefault === true,
        );
        return total === undefined || row?.perDispatch === undefined
          ? undefined
          : total + entry.censusCount * row.perDispatch;
      }, 0);
    const skeletonNames = [...new Set(first.rows.filter(isSkeleton).map((row) => row.geometry))];
    const single = skeletonNames
      .map((geometry) => ({ geometry, ns: timesOf(geometry) }))
      .filter((entry): entry is { geometry: string; ns: number } => entry.ns !== undefined)
      .reduce<{ geometry: string; ns: number } | undefined>(
        (best, entry) => (best === undefined || entry.ns < best.ns ? entry : best),
        undefined,
      );
    const tableForward = sum((entry) => entry.table.ms);
    const summary = {
      format: "karume-e2e-k85-linear-sweep/1",
      adapter: first.adapter,
      userAgent: first.userAgent,
      dp4a: first.dp4a,
      timing: first.gpuTiming,
      candidateSet: first.settings.candidateSet,
      caseIds: cases.map(({ id }) => id),
      rounds: ROUNDS,
      idleSeconds: IDLE_SECONDS,
      caseSets: first.caseSets,
      table: { id: tableId, geometry: table },
      forward: {
        what: "one Wan2.2 TI2V-5B 33f DiT forward (each case's censusCount × its ms per dispatch)",
        nodes: forward.reduce((total, entry) => total + entry.censusCount, 0),
        tableMs: tableForward,
        defaultMs: sum((entry) => entry.defaultMs),
        perCaseBestSkeletonMs: sum((entry) => entry.bestSkeleton.ms),
        perCaseBestMs: sum((entry) => entry.best.ms),
        singleBestSkeleton:
          single === undefined
            ? undefined
            : {
                geometry: single.geometry,
                ms: single.ns / 1e6,
                vsTable: tableForward === undefined ? undefined : tableForward / (single.ns / 1e6),
              },
      },
      cases: perCase,
    };
    await writeTextFile(`${directory}/k85-summary.json`, `${JSON.stringify(summary, null, 2)}\n`);
    console.log(`[k85] summary: ${JSON.stringify(summary.forward)}`);
    for (const entry of perCase) {
      console.log(
        `[k85] ${entry.caseId} ×${entry.censusCount}: table ${entry.table.geometry ?? "?"} ${entry.table.ms?.toFixed(4) ?? "?"} ms, best skeleton ${entry.bestSkeleton.geometry ?? "?"} ${entry.bestSkeleton.ms?.toFixed(4) ?? "?"} ms (×${entry.bestSkeletonVsTable?.toFixed(3) ?? "?"}), drift ${entry.driftRatio?.toFixed(3) ?? "?"}`,
      );
    }

    for (const report of reports) assertEquals(report.gpuTiming.unit, "ns", "GPU timestamps");
    for (const entry of perCase) {
      assertEquals(entry.errors, [], `${entry.caseId}: failed rows`);
      assertEquals(entry.mismatches, [], `${entry.caseId}: candidates whose words differ`);
      assert(entry.table.ms !== undefined, `${entry.caseId}: the table geometry was not measured`);
      assert(
        entry.bestSkeleton.ms !== undefined,
        `${entry.caseId}: no skeleton candidate measured`,
      );
    }
  });
});
