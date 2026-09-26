/**
 * gemma4 の **quant 席ごとの sha 参照行**（ADR 0106 / ADR 0110 決定 7）— 固定プロンプト 1 本の
 * greedy の token id 列が、この機の参照値と 1 token も違わないこと。
 *
 * 走行・実物・決着の中身は `helpers/gemma-reference.ts` の 1 本（gemma4-qat と共有）。ここが
 * 持つのは置き場（fixture / 結果の席）だけである。
 *
 * 席は固定表で持たず、ミラーの manifest から**全 quant 席**を列挙する（`mirrorSeats`）。行の
 * クラスも席の `session` から導く: `session` が空の席（`i4`）は参照行、非空の席（`i4-gemvpar` /
 * `i4-fast` など束のノブを持つもの）は実用行。配布形に席が増えれば行も増える。
 *
 * 資産はミラー `models/karume-gemma4/`（git 追跡外）。無い機・旧 major が残っている機は明示
 * SKIP（`helpers/gemma-mirror.ts` の `mirrorAvailable`）。ADR 0005 の「全 SKIP は明示 FAIL」
 * 門番は GPU アダプタの有無だけを見ており、この SKIP とは独立。
 */

import { denoDirectory } from "@karume/hub/deno";
import { Gemma4Pipeline } from "../gemma.ts";
import { openReferences, registerReferenceGate } from "../../runtime/tests/helpers/reference.ts";
import { openResults } from "../../runtime/tests/helpers/results.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { mirrorAvailable } from "./helpers/gemma-mirror.ts";
import { mirrorSeats, seatCaseId, settleSeatReference } from "./helpers/gemma-reference.ts";

const MIRROR_DIR = new URL("../../../models/karume-gemma4/", import.meta.url);
/** token id 列の sha256 参照値（環境キーごとの行 — `runtime/tests/helpers/reference.ts`）。 */
const references = openReferences(new URL("fixtures/references/gemma4.json", import.meta.url));
/** 実物と決着の置き場（`outputs/verify/<環境キー>/<日付>_gemma4/` — 消して安全）。 */
const results = openResults("gemma4");

const AVAILABLE = mirrorAvailable(MIRROR_DIR);
if (!AVAILABLE) {
  console.warn(
    `[karume] ${MIRROR_DIR.pathname} が無い（か karume/5 でない）ため gemma4 の quant 席の ` +
      "sha 参照行を SKIP する。tools/export-recipes の dist.py --pipeline gemma4 で作成する",
  );
}
const RUNNABLE = AVAILABLE && GPU_AVAILABLE;

/** 並べる席（ミラーの全 quant 席 — 行のクラスは `session` から導かれる。ミラーが無ければ空）。 */
const SEATS = mirrorSeats(MIRROR_DIR);

for (const seat of SEATS) {
  const id = seatCaseId(seat);
  Deno.test({
    name: `gemma4 ${id}（${seat.row === "reference" ? "参照行" : "実用行"}）: ` +
      "固定プロンプトの greedy の token id 列がこの機の参照値と一致する（実 GPU）",
    ignore: !RUNNABLE || references.lacksReference(id),
    fn: () =>
      settleSeatReference({
        references,
        results,
        seat,
        open: (choice) => Gemma4Pipeline.fromPretrained(denoDirectory(MIRROR_DIR), choice),
      }),
  });
}

/** この門が持つケース ID 全部（参照値が無いものを登録時に 1 度だけ知らせる）。 */
const CASE_IDS: readonly string[] = SEATS.map(seatCaseId);
if (RUNNABLE) references.warnMissing(CASE_IDS);
// 「この環境の参照値がまだ無い」を無音の緑にしないための門番（ADR 0005 と同じ流儀）。
registerReferenceGate(references, { runnable: RUNNABLE, caseIds: CASE_IDS });
