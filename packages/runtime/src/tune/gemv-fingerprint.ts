/**
 * The fingerprints of a GEMV sweep record, built as fingerprint.ts builds the geometry sweep's:
 * synchronous FNV-1a 64-bit hashes ({@link fnv1a64Hex}) of canonical JSON and of one JSON line per
 * dispatch (pipeline key, params, the WGSL's token text and workgroups), reading neither the GPU nor
 * the clock. They are per piece of evidence, so that changing one kernel or adding one case leaves
 * the other rows' evidence standing: a derivation recomputes each with its runtime and reads a row
 * whose fingerprints differ as not current.
 *
 * - {@link gemvCaseFingerprint}: one case's dispatch-deciding fields (id, storage, group, n, k — not
 *   its row counts, which each row and summary carries, nor its source, a description). A record's
 *   `cases[].fingerprint`.
 * - {@link gemvProgramKernelId}: the kernel one row measured — its program's dispatches, built by the
 *   sweep's own program builder (`gemvProgram`), so the kernels measured and the kernels
 *   fingerprinted are never built by two paths. A record's `rows[].kernel`.
 * - {@link gemvCaseTableId}: the whole case table (the derivation's provenance names the table it
 *   read against).
 *
 * MUST: nothing is computed at import time.
 */

import type { LinearProgram } from "../runtime/linear-program.ts";
import { type GemvKernelChoice, gemvProgram, programWorkgroups } from "./gemv-candidates.ts";
import { GEMV_CASES, type GemvCase } from "./gemv-cases.ts";
import { canonicalJson, fnv1a64Hex, wgslTokenText } from "./fingerprint.ts";

/** A case's dispatch-deciding fields. */
const caseFields = ({ id, storage, group, n, k }: GemvCase) => ({ id, storage, group, n, k });

/** The fingerprint of one case's dispatch-deciding fields (a record's `cases[].fingerprint`). */
export const gemvCaseFingerprint = (gemvCase: GemvCase): string =>
  fnv1a64Hex(canonicalJson(caseFields(gemvCase)));

/** The fingerprint of a case table's dispatch-deciding fields and row counts (default: the sweep's table). */
export const gemvCaseTableId = (cases: readonly GemvCase[] = GEMV_CASES): string =>
  fnv1a64Hex(
    canonicalJson(cases.map((gemvCase) => ({ ...caseFields(gemvCase), rows: gemvCase.rows }))),
  );

/** The program of a choice for a case at a row count (`gemvProgram` — tests inject another). */
export type GemvProgramBuilder = (
  gemvCase: GemvCase,
  m: number,
  choice: GemvKernelChoice,
) => LinearProgram;

/**
 * The fingerprint of the kernel a program dispatches (a record's `rows[].kernel`): one JSON line per
 * dispatch — pipeline key, params, the WGSL's token text, workgroups — joined with newlines.
 */
export const gemvProgramKernelId = (program: LinearProgram): string => {
  const workgroups = programWorkgroups(program);
  return fnv1a64Hex(
    program.dispatches
      .map((dispatch, index) =>
        JSON.stringify([
          dispatch.key,
          [...dispatch.params],
          wgslTokenText(dispatch.wgsl()),
          workgroups[index],
        ]),
      )
      .join("\n"),
  );
};

/** The fingerprint of the kernel of a choice for a case at a row count. */
export const gemvKernelId = (
  gemvCase: GemvCase,
  m: number,
  choice: GemvKernelChoice,
  programFor: GemvProgramBuilder = gemvProgram,
): string => gemvProgramKernelId(programFor(gemvCase, m, choice));
