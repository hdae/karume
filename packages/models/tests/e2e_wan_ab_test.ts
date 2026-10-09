/**
 * Wan2.1 の**自機 A/B 門**（実 GPU・ADR 0110 決定 5 / ADR 0120 決定 4 / 5・段 4 と段 3 のノブ単位の記録）。実用席
 * `f16+dit8-a8-attn8-s16`（`session` に linear a8 / attention a8 / S の f16 格納を束ねた席）を、**同じ機・同じ i8 重み**で
 * `session` が空の参照席（manifest から導く — `referencePartnerOf`。今の配布形では `f16+dit8`）と比べる。比較相手は
 * 同機の参照層で、torch golden は使わない（`f16` 席は重みが違うので床が恒真になる — ADR 0110 決定 5 ②）。
 *
 * 1 ケース（= フレーム数 1 つ）の手順（形は `e2e_anima_ab_test.ts` と同じ・本体は 2.2 と共有の
 * `helpers/wan-ab-gate.ts` の {@link runWanAbCase}）:
 *
 * 1. 参照席の pipeline で **step 1 の潜在**（2 ステップの通しの 1 step 目 — 席が最初に効いた直後・決定 5-4）を採り、
 *    購読側の throw で生成を打ち切る（`captureThenAbort`）。dispose して解放を待つ。
 * 2. 実用席で同じことを **2 回**（独立の 2 pipeline）。
 * 3. 門を 3 つ掛ける:
 *    - **決定性**（決定 6 の MUST）— 実用席の 2 回の潜在がビット一致（`assertBitIdentical`）。
 *    - **census**（決定 5-1）— 実用席の DiT の各パスで linear 307 本ちょうどが i8a8・参照経路の linear 0 本（束の
 *      census 表の行 — `assertRowCensus`）。attention の 2 ノブは表に書かない規約（行ブロックの枚数が device の
 *      束縛上限で変わる）なので 1 本以上（`assertSeatsApplied`）。
 *    - **帯**（決定 5-2 / 5-3）— relRMS が {@link CASES} の上限の内・非有限なし・参照席と 1 bit 以上違う（`judgeAb`）。
 *
 * 33 フレームのケースは続けて故障注入 5 件（どれも「門が赤になる」ことを assert する — 決定 5-5）と、ノブ単位の記録
 * （段 3・{@link knobRungs}）を回す。81 フレームのノブ単位の記録は opt-in（`KARUME_WAN_FULL_PIPELINE=1`）。
 *
 * 最終出力（フレーム）はここでは作らない — sha 行（`e2e_wan_pipeline_test.ts` の参照行 / 実用行）が持ち、帯にしない
 * （決定 5-4）。結果の席は `wan-ab`。
 *
 * VRAM: pipeline は 1 本ずつ順に張る（同時に張らない — 構築では Session を張らず、DiT の Session は step 1 の打ち切りで
 * 畳まれる）。B570 は `destroy()` の解放が遅れる（ADR 0118 段 8 の結果・ADR 0120 リスク 6）ので、観測 1 回ごとに
 * {@link settleReleases} で解放を待ち、確保の大きい 81 フレームを先に回す。
 *
 * MUST: 資産は `models/karume-wan2.1/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU 無し環境は理由を出して
 * **明示 SKIP** する（ADR 0005）。故障注入とノブ単位の席は同じ配布形を symlink で借りた一時 manifest
 * （`withManifestOverride`）で張り、元の配布形は 1 バイトも書き換えない。
 */

import { assertEquals } from "@std/assert";
import type { SessionSpec } from "@karume/hub";
import type { GpuContext } from "@karume/runtime";
import { WanPipeline } from "../wan.ts";
import { assertSeatsApplied, mergeCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  measureAb,
} from "./helpers/ab-gate.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { WAN_ASSEMBLE_COMMAND } from "./helpers/wan-distribution.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { settleReleases } from "./helpers/settle-releases.ts";
import {
  declaredSessionOf,
  effectiveSessionOf,
  errorText,
  observeOverridden,
  runWanAbCase,
  WAN_AB_OBSERVATION,
  type WanAbSubject,
  type WanStep1Observation,
} from "./helpers/wan-ab-gate.ts";

/** 配布形の置き場（`dist.py --pipeline wan` の出力先）。 */
const ASSETS_DIR = new URL("../../../models/karume-wan2.1/", import.meta.url);
/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_wan-ab/` — 消して安全）。 */
const results = openResults("wan-ab");

/** 対象のモデル（帯はモデル × 席 × フレーム数ごとの宣言なので明示する）。 */
const MODEL = "t2v-1.3b";
/** 対象の実用席（ADR 0120 決定 1）。 */
const PRACTICAL = "f16+dit8-a8-attn8-s16";

/** 50 ステップ級の重い記録の opt-in（81 フレームのノブ単位の記録 — `e2e_wan_pipeline_test.ts` と同じ変数）。 */
const FULL_PIPELINE = Deno.env.get("KARUME_WAN_FULL_PIPELINE") === "1";

/**
 * ケース（フレーム数）と帯。並びは確保の大きい 81 フレームが先（B570 の `destroy()` の遅れ — モジュール doc）。
 *
 * 帯は**宣言**で、環境キー別の行にしない・`KARUME_REFERENCE` で書き換えない（ADR 0110 決定 4）。導出規則（決定 5）:
 * 上限 = 観測点の実測 relRMS × 2 程度をケースごとに独立に導く（33 フレームの値を 81 フレームへ外挿しない — attention a8
 * の誤差は行の長さ N で増える・ADR 0120 決定 4）。床 = 1 bit 以上違う（MUST — {@link judgeAb} が帯と無関係に常に見る）
 * で、`floor` を 0 より上に置くのはノブ 1 つを外した実測が全部入りより明確に小さいと示せたときだけ。
 *
 * MUST: 導出前は `undefined` — {@link judgeAb} が実測を出して赤で止まる（ADR 0120 決定 4）。通る値を仮置きして検出力の
 * 無い門を増やさない（`ab-gate.ts` の MUST）。段 4 の B570 の実測 × 2 を導出表と一緒にここへ書く。anima の 1024² の値
 * （3.0e-2）は参考であって Wan の根拠にならない（attention a8 の誤差は行の長さ N で増える — 決定 4）。
 *
 * | ケース | 実測 relRMS | 実測 maxAbs | floor | ceiling          | 採った日・機 |
 * | ------ | ----------- | ----------- | ----- | ---------------- | ------------ |
 * | 81f    | 2.0956e-1   | 5.3956e-1   | 0     | 4.2e-1（× 2 切上） | 2026-10-03・B570 |
 * | 33f    | 1.0676e-1   | 2.8391e-1   | 0     | 2.2e-1（× 2 切上） | 2026-10-03・B570 |
 *
 * ノブ単位（33 フレーム・記録）: linear の a8 だけ 2.9991e-2 → + attention の a8 1.04e-1 → + s16 1.0676e-1。attention の a8 が
 * 誤差の大半で、81 フレーム（N = 32,760）では 33 フレーム（N = 14,040）の約 2 倍（ADR 0120 リスク 1 の実測）。故障注入 ③
 * （attention の a8 を外した席）は 3.0778e-2、⑤（別 seed）は 1.3828e+0。
 */
const CASES: readonly { readonly frames: number; readonly band: AbBand | undefined }[] = [
  { frames: 81, band: { floor: 0, ceiling: 4.2e-1 } },
  { frames: 33, band: { floor: 0, ceiling: 2.2e-1 } },
];

/** 故障注入を回すフレーム数（短い 33 — 門の空振りを示すのに長さは要らない）。 */
const FAULT_FRAMES = 33;

/**
 * ノブ単位の記録（ADR 0120 決定 5・段 3）の段 — 同じ i8 重みにノブを 1 つずつ足す（`a8` → `+ attn8` → `+ s16`）。
 * 値は manifest の実用席の宣言から引く（束の値をテストに書き写さない）。最後の段 = 実用席の宣言そのもの。
 *
 * MUST: 宣言が 3 ノブちょうどでなければ落とす（ノブが増減した束で段の並びを黙って作らない — 並びを決め直す）。
 */
const knobRungs = (
  declared: SessionSpec,
  where: string,
): readonly { readonly label: string; readonly session: SessionSpec }[] => {
  const { linearCompute, attentionCompute, attentionScoreStorage, ...rest } = declared;
  if (
    linearCompute === undefined || attentionCompute === undefined ||
    attentionScoreStorage === undefined || Object.keys(rest).length > 0
  ) {
    throw new Error(
      `${where}: 実用席の宣言 ${
        JSON.stringify(declared)
      } が linear / attention / S の格納の 3 ノブちょうど` +
        "でない（段の並びを決め直す）",
    );
  }
  return [
    { label: `linearCompute=${linearCompute}`, session: { linearCompute } },
    {
      label: `+ attentionCompute=${attentionCompute}`,
      session: { linearCompute, attentionCompute },
    },
    { label: `+ attentionScoreStorage=${attentionScoreStorage}`, session: declared },
  ];
};

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため Wan の自機 A/B 門を SKIP する` +
      `（組み立て: ${WAN_ASSEMBLE_COMMAND}）`,
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

/**
 * 2.1 の門の対象（共有の段 {@link runWanAbCase} へ渡す）。要求の固定値（`request`）は省いて世代の既定（832×480・
 * guidance / shift は manifest の既定）に乗せる — `e2e_wan_pipeline_test.ts` の seed 経路の sha 行と同じ要求。
 *
 * MUST: 呼ぶのは {@link RUNNABLE} のテストの中だけ（配布形が無いのにここへ来たら ignore の判定の破れなので落とす）。
 */
const subjectOf = (): WanAbSubject => {
  if (manifestText === undefined) {
    throw new Error(
      `${ASSETS_DIR.pathname} に karume.json が無いのに A/B 門が走った（ignore の判定の破れ）`,
    );
  }
  return {
    label: "wan",
    family: "wan",
    model: MODEL,
    practical: PRACTICAL,
    assetsRoot: ASSETS_DIR,
    manifestText,
    results,
    fromPretrained: (source, options) => WanPipeline.fromPretrained(source, options),
  };
};

/**
 * ノブ単位の記録（ADR 0120 決定 5・段 3 — **記録のみ・門ではない**）。同じ i8 重みに {@link knobRungs} の順でノブを
 * 1 つずつ足した席の step 1 の潜在の relRMS（参照席に対する値と、1 段前に対する値）を results.json の note と
 * comparisons に残す。最後の段（全部入り）は A/B 門の実用席の 1 回目をそのまま使う。
 *
 * 赤にするのは記録が意味を失う形だけ: 段の席にそのノブが届いていない（`assertSeatsApplied`）と非有限。
 *
 * NOTE: 層ごとの差は採らない — i8 の重みで層別の出口を持つ計測用グラフ（`…-i8-dyn-probe`）がまだ無い（export 台本の
 * `--layers` は f16 系列だけ）。
 */
const recordKnobLadder = async (
  t: Deno.TestContext,
  gpu: GpuContext,
  subject: WanAbSubject,
  frames: number,
  reference: { readonly name: string; readonly observation: WanStep1Observation },
  practical: WanStep1Observation,
): Promise<void> => {
  await t.step(`ノブ単位の記録（${frames} フレーム・門ではない）`, async () => {
    const id = `knobs-${frames}f`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const rungs = knobRungs(declaredSessionOf(subject), id);
      const notes: string[] = [];
      let previous = reference.observation;
      for (const [index, { label, session }] of rungs.entries()) {
        const where = `${id} ${label}`;
        const observed = index === rungs.length - 1
          ? practical
          : await observeOverridden(gpu, subject, session, frames);
        const effective = effectiveSessionOf(subject, session, where);
        const census = mergeCensus(observed.passes, where);
        // 段の席にそのノブが届いていなければ、その段の relRMS は名乗るものを測っていない。
        assertSeatsApplied(census, effective, where);
        const toReference = measureAb(observed.latent, reference.observation.latent);
        const toPrevious = measureAb(observed.latent, previous.latent);
        recorded.comparisons.push(
          comparisonOf(
            {
              output: WAN_AB_OBSERVATION,
              reference: reference.name,
              practical: `${PRACTICAL}（${label} まで）`,
            },
            toReference,
            undefined,
          ),
        );
        notes.push(
          `${label}: 参照席に対し ${describeAb(toReference)}・1 段前に対し ${
            describeAb(toPrevious)
          }・census ${describeBundleCensus(census, effective)}`,
        );
        assertEquals(Number.isFinite(toReference.relRms), true, `${where}: 非有限`);
        previous = observed;
      }
      const note = notes.join(" / ");
      console.log(`[e2e] wan ノブ単位 ${frames} フレーム: ${note}`);
      return { status: "pass", note };
    });
  });
};

for (const { frames, band } of CASES) {
  const id = `${PRACTICAL}-${frames}f`;
  const faults = frames === FAULT_FRAMES;
  const knobs = frames === FAULT_FRAMES || FULL_PIPELINE;
  Deno.test({
    name:
      `e2e(実GPU): Wan ${id} の step 1 の潜在が参照席と決定性・census・帯で釣り合う（自機 A/B）` +
      (faults ? "・故障注入 5 件は赤" : "") +
      (knobs ? "・ノブ単位の relRMS を記録" : ""),
    ignore: !RUNNABLE,
    fn: async (t) => {
      const gpu = await acquireTestGpu();
      try {
        assertAdapterMatchesEnvironment(gpu);
        const subject = subjectOf();
        const { reference, referenceRun, first } = await runWanAbCase(t, gpu, subject, {
          id,
          frames,
          band,
          faults,
        });

        if (knobs) {
          await recordKnobLadder(
            t,
            gpu,
            subject,
            frames,
            { name: reference, observation: referenceRun },
            first,
          );
        }
      } finally {
        await settleReleases(gpu);
        gpu.destroy();
      }
    },
  });
}
