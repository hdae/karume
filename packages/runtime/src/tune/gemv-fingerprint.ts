/**
 * The two fingerprints of a GEMV sweep record, built as fingerprint.ts builds the geometry sweep's:
 * synchronous FNV-1a 64-bit hashes ({@link fnv1a64Hex}) of canonical JSON and of one JSON line per
 * dispatch (pipeline key, params, the WGSL's token text and workgroups), reading neither the GPU nor
 * the clock. A later derivation recomputes both with its runtime and refuses a record that differs:
 * one measured with other kernels, or on another case table.
 *
 * - {@link gemvCaseTableId}: the dispatch-deciding fields of the case table (id, storage, group, n,
 *   k and the row counts — not the source, a description).
 * - {@link gemvKernelsId}: the programs the rows without an error measured, each rebuilt from its
 *   case, row count and choice through the sweep's own program builder (`gemvProgram`), so the
 *   kernels measured and the kernels fingerprinted are never built by two paths.
 *
 * MUST: nothing is computed at import time.
 */

import type { LinearProgram } from "../runtime/linear-program.ts";
import type { LinearGemvChoice } from "../runtime/linear-strategies.ts";
import { gemvProgram, programWorkgroups } from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase, gemvCaseOf } from "./gemv-cases.ts";
import { choiceOfRecord, type GemvSweepRow } from "./gemv-report.ts";
import { canonicalJson, fnv1a64Hex, wgslTokenText } from "./fingerprint.ts";

/** The fingerprint of a case table's dispatch-deciding fields (default: the sweep's table). */
export const gemvCaseTableId = (cases: readonly GemvCase[] = GEMV_CASES): string =>
  fnv1a64Hex(
    canonicalJson(
      cases.map(({ id, storage, group, n, k, rows }) => ({ id, storage, group, n, k, rows })),
    ),
  );

/** The program of a choice for a case at a row count (`gemvProgram` — tests inject another). */
export type GemvProgramBuilder = (
  gemvCase: GemvCase,
  m: number,
  choice: LinearGemvChoice,
) => LinearProgram;

/** The fields of a row the kernel fingerprint reads. */
export type GemvMeasuredRow = Pick<GemvSweepRow, "caseId" | "m" | "choice" | "error">;

/**
 * The fingerprint of the kernels the rows without an error measured, in the rows' order. A row of a
 * case the table does not have throws.
 */
export const gemvKernelsId = (
  rows: readonly GemvMeasuredRow[],
  cases: readonly GemvCase[] = GEMV_CASES,
  programFor: GemvProgramBuilder = gemvProgram,
): string => {
  const lines: string[] = [];
  for (const row of rows) {
    if (row.error !== undefined) continue;
    const gemvCase = gemvCaseOf(row.caseId, cases);
    if (gemvCase === undefined) {
      throw new Error(`Case ${row.caseId} of a measured row is not in the GEMV sweep's case table`);
    }
    const program = programFor(gemvCase, row.m, choiceOfRecord(row.choice));
    const workgroups = programWorkgroups(program);
    program.dispatches.forEach((dispatch, index) => {
      lines.push(
        JSON.stringify([
          dispatch.key,
          [...dispatch.params],
          wgslTokenText(dispatch.wgsl()),
          workgroups[index],
        ]),
      );
    });
  }
  return fnv1a64Hex(lines.join("\n"));
};

/**
 * The kernel fingerprint of a record's rows (`kernels`), derived without a GPU — what an application
 * that merges the records of several `runGemvSweep` calls recomputes over the merged rows.
 */
export const gemvSweepKernelsId = (rows: readonly GemvMeasuredRow[]): string => gemvKernelsId(rows);
