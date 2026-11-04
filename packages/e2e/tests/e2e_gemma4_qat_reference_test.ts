/**
 * The q4_0 QAT line's (`karume-gemma4-qat`) **sha256 reference row per quant seat** (ADR 0106 / ADR 0110 decision 7): the
 * greedy token ids of the fixed prompt equal this machine's reference value, token for token.
 *
 * The run, the artifact and the settlement are `helpers/gemma-reference.ts`'s, shared with the
 * regular line (`e2e_gemma4_reference_test.ts`) and the mobile QAT line
 * (`e2e_gemma4_qat_mobile_reference_test.ts`); this file holds only the
 * places — its own fixture, because the case ids (`<model>/<quant>`) are the regular line's
 * spellings, and its own results folder. The seats are enumerated from the mirror's manifest
 * (`mirrorSeats`): the seat `i4` of every model, a reference row (its `session` is empty), and
 * E2B's / E4B's parallel and fused seats `i4-gemvpar` / `i4-fast`, practical rows (their
 * `session` declares the regular line's flags).
 * The seats run without the drafter — speculation selects the same tokens (the identity is pinned
 * by `e2e_gemma4_qat_mirror_test.ts` ③), so a speculative row would fix the same ids twice.
 *
 * The asset is the local mirror `models/karume-gemma4-qat/` (outside git); without it the
 * cases SKIP explicitly (`mirrorAvailable`).
 */

import { test } from "vite-plus/test";
import { Pipeline } from "../../models/gemma.ts";
import { openReferences, registerReferenceGate } from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import { mirrorDirectory, mirrorRoot } from "../../models/tests/helpers/mirror.ts";
import {
  mirrorSeats,
  seatCaseId,
  settleSeatReference,
} from "../../models/tests/helpers/gemma-reference.ts";

const MIRROR = mirrorRoot("karume-gemma4-qat");
/** The sha256 of the token ids, one row per environment key (`runtime/tests/helpers/reference.ts`). */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/gemma4-qat.json", import.meta.url),
);
/** Artifacts and settlements (`outputs/verify/<environment key>/<date>_gemma4-qat/` — safe to delete). */
const results = openResults("gemma4-qat");

const AVAILABLE = mirrorAvailable(MIRROR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR.pathname} is absent (or not karume/5), so the q4_0 QAT line's sha256` +
      " reference rows SKIP. Assemble it with dist.py --pipeline gemma4-qat in tools/export-recipes",
  );
}
const RUNNABLE = AVAILABLE && GPU_AVAILABLE;

/** Every seat of the mirror (empty without the mirror). */
const SEATS = mirrorSeats(MIRROR);

for (const seat of SEATS) {
  const id = seatCaseId(seat);
  test.skipIf(!RUNNABLE || references.lacksReference(id))(
    `gemma4-qat ${id} (${seat.row} row): the greedy token ids of the fixed prompt equal this machine's reference (real GPU)`,
    () =>
      settleSeatReference({
        references,
        results,
        seat,
        open: (choice) => Pipeline.fromPretrained(mirrorDirectory(MIRROR), choice),
      }),
  );
}

/** Every case id of this gate (the missing ones are announced once at registration). */
const CASE_IDS: readonly string[] = SEATS.map(seatCaseId);
if (RUNNABLE) references.warnMissing(CASE_IDS);
// A missing reference of this environment is not a silent green (ADR 0005's rule).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
