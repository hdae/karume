/**
 * gemma4-qat-mobile's **sha reference rows per quant preset** (ADR 0106 / ADR 0110 decision 7) —
 * the greedy token id sequence of one fixed prompt does not differ by a single token from this
 * machine's reference value.
 *
 * The run, the artifact and the settlement are the one path in `helpers/gemma-reference.ts`
 * (shared with gemma4). This file holds only the places (the fixture / the results folder) and the
 * one function that narrows the manifest's model names to the QAT selection axis.
 *
 * The presets are not held in a fixed table; **every model × every quant preset** is enumerated
 * from the mirror's manifest (`mirrorSeats`). The row class is derived from the preset's `session`
 * too: a preset with an empty `session` (`i4`) gives a reference-path row, one with a non-empty
 * `session` (those that bundle knobs, such as `i4-gemvpar` / `i4-fast`) a practical-preset row.
 * When the distribution gains a preset, it gains a row.
 *
 * The asset is the mirror `models/karume-gemma4-qat-mobile/` (untracked by git). A machine without
 * it, or with an old major left in place, SKIPs explicitly (`mirrorAvailable` in
 * `helpers/gemma-mirror.ts`). ADR 0005's "an all-SKIP run is an explicit FAIL" guard looks only at
 * whether a GPU adapter exists and is independent of this SKIP.
 */

import { test } from "vite-plus/test";
import { nodeDirectory } from "@karume/hub/node";
import { Pipeline } from "../../models/gemma4-qat-mobile.ts";
import { openReferences, registerReferenceGate } from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { mirrorAvailable } from "../../models/tests/helpers/gemma-mirror.ts";
import {
  mirrorSeats,
  type Seat,
  seatCaseId,
  settleSeatReference,
} from "../../models/tests/helpers/gemma-reference.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4-qat-mobile/", import.meta.url);
/**
 * sha256 reference values of the token id sequence (rows per environment key —
 * `runtime/tests/helpers/reference.ts`).
 */
const references = openReferences(
  new URL("../../models/tests/fixtures/references/gemma4-qat-mobile.json", import.meta.url),
);
/**
 * Where the artifacts and settlements go (`outputs/verify/<environment
 * key>/<date>_gemma4-qat-mobile/` — safe to delete).
 */
const results = openResults("gemma4-qat-mobile");

const AVAILABLE = mirrorAvailable(MIRROR_DIR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR_DIR.pathname} is absent (or not karume/5), so the gemma4-qat-mobile quant seats' ` +
      "sha256 reference rows SKIP. Assemble it with dist.py --pipeline gemma4-qat-mobile in tools/export-recipes",
  );
}
const RUNNABLE = AVAILABLE && GPU_AVAILABLE;

/**
 * The models `gemma4QatMobile.Pipeline.fromPretrained` accepts
 * (`gemma4QatMobile.FromPretrainedOptions.model`).
 */
const QAT_MODELS = ["e2b", "e4b"] as const;
type QatModel = (typeof QAT_MODELS)[number];

/**
 * Narrows the manifest's model names to the QAT selection axis. An unknown model throws at
 * registration — dropping it silently would go green while a preset the distribution gained has no
 * row.
 */
const qatSeat = (seat: Seat): Seat<QatModel> => {
  const model = QAT_MODELS.find((known) => known === seat.model);
  if (model === undefined) {
    throw new Error(
      `${MIRROR_DIR.pathname}: model '${seat.model}' is not on gemma4QatMobile.Pipeline's selection axis` +
        ` (accepted: ${QAT_MODELS.join(" / ")})`,
    );
  }
  return { ...seat, model };
};

/**
 * The presets to run (every quant preset of the mirror — the row class is derived from `session`;
 * empty without the mirror).
 */
const SEATS: readonly Seat<QatModel>[] = mirrorSeats(MIRROR_DIR).map(qatSeat);

for (const seat of SEATS) {
  const id = seatCaseId(seat);
  test.skipIf(!RUNNABLE || references.lacksReference(id))(
    `gemma4-qat-mobile ${id}（${seat.row === "reference" ? "reference-path row" : "practical-preset row"}）: ` +
      "the greedy token id sequence of the fixed prompt matches this machine's reference value (real GPU)",
    () =>
      settleSeatReference({
        references,
        results,
        seat,
        open: (choice) => Pipeline.fromPretrained(nodeDirectory(MIRROR_DIR), choice),
      }),
  );
}

/**
 * Every case ID this check holds (those without a reference value are reported once, at
 * registration).
 */
const CASE_IDS: readonly string[] = SEATS.map(seatCaseId);
if (RUNNABLE) references.warnMissing(CASE_IDS);
// A guard so that "this environment has no reference value yet" is not a silent green (the same
// approach as ADR 0005).
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
