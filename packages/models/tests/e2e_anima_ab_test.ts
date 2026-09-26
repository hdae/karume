/**
 * Anima の**自機 A/B 門**（実 GPU・ADR 0110 決定 5）。既定席 `f16+dit8-a8-attn8-s16`（実用層 —
 * `session` に linear a8 / attention a8 / S の f16 格納を束ねた席）を、**同じ機・同じ重み**で
 * `session` が空の参照席（manifest から導く — {@link referencePartnerOf}。今の配布形では
 * `f16+dit8`）と比べる。比較相手は同機の参照層で、torch golden は使わない。
 *
 * 1 ケース（= 解像度 1 つ）の手順:
 *
 * 1. 参照席の pipeline で **step 1 の latent**（席が最初に効いた直後 — 決定 5-4）を採り、購読側の
 *    throw で生成を打ち切る（`onEvent` の `copyLatents()`・{@link captureThenAbort}）。dispose。
 * 2. 実用席で同じことを **2 回**（独立の 2 pipeline）。
 * 3. 門を 3 つ掛ける（どれも赤なら決着は `fail`）:
 *    - **決定性**（決定 6 の MUST）— 実用席の 2 回の latent がビット一致する
 *      （{@link assertBitIdentical}）。赤にする故障注入: ②（2 回目だけ別 seed）。
 *    - **census**（決定 5-1 の MUST）— 実用席の transformer の run で、束の各ノブの変種が 1 本以上
 *      走っている（`assertSeatsApplied` — 期待は**元の配布形**の宣言）。本数は決着の note に残る。
 *      赤にする故障注入: ③（`attentionCompute` だけ外した席）。
 *    - **帯**（決定 5-2 / 5-3）— relRMS が {@link CASES} の帯に入り、非有限が無く、参照席と
 *      1 bit 以上違う（{@link judgeAb}）。赤にする故障注入: ①（実用席の `session` を空にした
 *      manifest → 床）。
 *
 * 最終出力（PNG）はここでは作らない — 実用行の sha 門（`e2e_anima_test.ts`）が持ち、帯にしない
 * （決定 5-4）。結果の席は `anima-ab`（`e2e_anima_test.ts` の `anima` と別 — 同じ系列名の 2 ファイルは
 * `results.json` を上書きし合う）。
 *
 * MUST: 資産は `models/karume-anima/`（untracked・実 GPU 機のローカル資産）。無い環境と GPU 無し
 * 環境は理由を出して**明示 SKIP** する（ADR 0005）。取得は手元の配布形を取得元ハンドル
 * （`denoDirectory`）で渡す — 故障注入は同じ配布形を symlink で借りた一時 manifest
 * （{@link withManifestOverride}）で、元の配布形は 1 バイトも書き換えない。
 */

import { assertThrows } from "@std/assert";
import { type DistributionSource, type Manifest, parseManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import type { SessionDiagnostics } from "@karume/runtime";
import { AnimaPipeline, type ImageSize } from "../mod.ts";
import { formatResolution } from "../anima.ts";
import {
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
} from "../../runtime/tests/helpers/pipeline-census.ts";
import { openResults, runRecordedCase } from "../../runtime/tests/helpers/results.ts";
import {
  type AbBand,
  assertBitIdentical,
  captureThenAbort,
  comparisonOf,
  describeAb,
  describeBundleCensus,
  judgeAb,
  measureAb,
  overrideQuantSession,
  quantOf,
  referencePartnerOf,
} from "./helpers/ab-gate.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { withManifestOverride } from "./helpers/manifest-override.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";

/** 資産の置き場（リポ直下 `models/karume-anima/`）。 */
const ASSETS_DIR = new URL("../../../models/karume-anima/", import.meta.url);
/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_anima-ab/` — 消して安全）。 */
const results = openResults("anima-ab");

/** 対象のモデル（既定モデル — 帯はモデル × 席 × 解像度ごとの宣言なので明示する）。 */
const MODEL = "anima-turbo-v1.1";
/** 対象の実用席（既定席）。 */
const PRACTICAL = "f16+dit8-a8-attn8-s16";
/** 観測点の名前（`comparisons[].output`）。 */
const OBSERVATION = "step1-latent";

/** 入力（`e2e_anima_test.ts` の sha 門と同じ — step 1 の sigma は steps で決まるので揃える）。 */
const PROMPT = "1girl, solo, long hair, blue eyes, school uniform, cherry blossoms, outdoors, " +
  "smile, upper body, masterpiece, best quality";
const STEPS = 8;
const SEED = 42;

/**
 * ケース（解像度）と帯。`band: undefined` = 未導出 — 門は実測を出して赤で止まる（ADR 0050 決定 4 の
 * 形。通る値を仮置きして検出力の無い門を増やさない）。
 *
 * 帯は**宣言**で、環境キー別の行にしない・`KARUME_REFERENCE` で書き換えない（ADR 0110 決定 4）。
 * 導出規則（決定 5 — 今の開発機は 1 台なので 1 台の実測から導く）:
 *
 * - **上限** = 観測点の実測 relRMS × 2 程度（決定 5-3 — 前例は ADR 0028 の 1.6 倍・irodori の
 *   2.0 倍）。「実測の 5〜10 倍」は使わない（決定 4 の MUST NOT）。量子化席なので理論値との整合
 *   （a8 の丸め誤差 ≈ 層ごとの理論値と中央値 1.8% で一致 — ADR 0025）を表の備考に残す。
 * - **床** = 1 bit 以上違う（決定 5-2 の MUST — {@link judgeAb} が帯と無関係に常に見る）。
 *   `floor` をそれより上に置くのは、ノブ 1 つを外した一時 manifest の実測が**全部入りの実測より
 *   明確に小さい**と示せたときだけ（反証 verify/A2 §4-4 — 記録からは係数 0.35〜0.67 の幅しか
 *   言えない）。示せなければ `floor: 0` とし、ノブ単位の縮退は census に任せる（決定 5-1）。
 *
 * 導出表（2026-09-26・Intel Arc B570 / Deno・参照席 `f16+dit8`）:
 *
 * | ケース | 実測 relRMS | 実測 maxAbs | ③ の relRMS（attn 外し） | floor | ceiling | 採った日・機 |
 * | ------ | ----------- | ----------- | ------------------------ | ----- | ------- | ------------ |
 * | 1024   | 1.4294e-2   | 8.5310e-2   | —（512 でだけ測る）      | 0     | 3.0e-2  | 2026-09-26 B570 |
 * | 512    | 1.1899e-2   | 7.2329e-2   | 1.0758e-2（×0.90）       | 0     | 2.4e-2  | 2026-09-26 B570 |
 *
 * 上限 = 実測 × 2 を有効数字 2 桁へ丸めた値。③（attentionCompute を外す）は全部入りの 0.90 倍で「明確に
 * 小さい」とは言えないので floor は 0（ノブ単位の縮退は census が持つ）。a8 の理論値（ADR 0025: 層ごとの
 * 丸め誤差の中央値 1.8%）と同じ桁で、w8a8 の step 1 判別帯（実測の 1.6 倍・ADR 0025 決定 6）と整合する。
 */
const CASES: readonly { readonly resolution: ImageSize; readonly band: AbBand | undefined }[] = [
  { resolution: { width: 1024, height: 1024 }, band: { floor: 0, ceiling: 3.0e-2 } },
  { resolution: { width: 512, height: 512 }, band: { floor: 0, ceiling: 2.4e-2 } },
];

/** 故障注入を回す解像度（最短の 512 — 門の空振りを示すのに解像度は要らない）。 */
const FAULT_CASE = CASES[1];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため Anima の自機 A/B 門を SKIP する` +
      "（exporter の dist.py で焼く）",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

const readManifest = (): Manifest => parseManifest(manifestText as string);

/** 1 run ぶんの観測（step 1 の latent と、そこまでの transformer の census）。 */
type Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

/**
 * `quant` の pipeline を組み、step 1 の latent を採って打ち切る（dispose まで — GPU も
 * pipeline が自前で取って返す）。census は transformer の run だけ（quant の `session` が効くのは
 * DiT の Session だけ — `anima/pipeline.ts` の `sessionOptions`）。
 */
const observeStep1 = async (
  source: DistributionSource,
  quant: string,
  resolution: ImageSize,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / ${formatResolution(resolution)} / seed ${seed}`;
  const runs: SessionDiagnostics["lastRunPipelines"][] = [];
  await using pipeline = await AnimaPipeline.fromPretrained(source, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      if (component === "transformer") runs.push(diagnostics.lastRunPipelines);
    },
  });
  const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(
    where,
    (capture) =>
      pipeline.generate({
        prompt: PROMPT,
        resolution,
        steps: STEPS,
        seed,
        onEvent: (event) => {
          if (event.kind === "denoise-step" && event.step === 1) capture(event.copyLatents().data);
        },
      }),
  );
  return { latent, census: mergeCensus(runs, `${where} の transformer`) };
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

for (const { resolution, band } of CASES) {
  const id = `${PRACTICAL}-${formatResolution(resolution)}`;
  Deno.test({
    name: `e2e(実GPU): ${id} の step 1 latent が参照席と決定性・census・帯で釣り合う（自機 A/B）`,
    ignore: !RUNNABLE,
    fn: async () => {
      await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const declared = quantOf(manifest, MODEL, PRACTICAL).session;
        const source = denoDirectory(ASSETS_DIR);
        const referenceRun = await observeStep1(source, reference, resolution, SEED);
        const first = await observeStep1(source, PRACTICAL, resolution, SEED);
        const second = await observeStep1(source, PRACTICAL, resolution, SEED);
        const measured = measureAb(first.latent, referenceRun.latent);
        recorded.comparisons.push(
          comparisonOf({ output: OBSERVATION, reference, practical: PRACTICAL }, measured, band),
        );
        const census = describeBundleCensus(first.census, declared);
        console.log(
          `[e2e] anima A/B ${id}: ${reference} → ${PRACTICAL} ${describeAb(measured)} ` +
            `/ census ${census}`,
        );
        assertBitIdentical(first.latent, second.latent, `${id}: 実用席の 2 回`);
        assertSeatsApplied(first.census, declared, `${id}: 実用席の transformer`);
        judgeAb(measured, band, id);
        return { status: "pass", note: `census ${census}` };
      });
    },
  });
}

// --- 故障注入（各門が空振りでないことの実証 — ADR 0110 決定 5-5）------------------
//
// どれも「門が赤になること」を assert する（緑 = 門が故障を見逃した）。一時 manifest は同じ配布形を
// symlink で借り、実用席の `session` だけを差し替える。解像度は {@link FAULT_CASE} の 1 本だけ。

Deno.test({
  name: "e2e(実GPU): 故障注入 ① 実用席の session を空にした manifest は床で赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const { resolution, band } = FAULT_CASE;
    const id = `fault1-empty-session-${formatResolution(resolution)}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const injected = overrideQuantSession(
        JSON.parse(manifestText as string),
        MODEL,
        PRACTICAL,
        {},
      );
      const referenceRun = await observeStep1(
        denoDirectory(ASSETS_DIR),
        reference,
        resolution,
        SEED,
      );
      const faulty = await withManifestOverride(
        ASSETS_DIR,
        injected,
        (source) => observeStep1(source, PRACTICAL, resolution, SEED),
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL}（session 空）` },
          measured,
          band,
        ),
      );
      console.log(`[e2e] anima 故障注入 ① ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, band, id), Error, "床の失敗");
    });
  },
});

Deno.test({
  name: "e2e(実GPU): 故障注入 ② 実用席の 2 回目だけ別 seed にすると決定性の門が赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const { resolution } = FAULT_CASE;
    const source = denoDirectory(ASSETS_DIR);
    const first = await observeStep1(source, PRACTICAL, resolution, SEED);
    const second = await observeStep1(source, PRACTICAL, resolution, SEED + 1);
    assertThrows(
      () => assertBitIdentical(first.latent, second.latent, "故障注入 ②"),
      Error,
      "ビット一致しない",
    );
  },
});

/**
 * 故障注入 ③ — `attentionCompute` だけ外した席。census（元の配布形の宣言が期待）が赤になることを
 * assert する。帯の判定は**記録だけ**（ADR 0030 決定 5 の再演 — 積み重ねた席の後段の失効は数値で
 * 掴みにくい。帯が緑のままなら census が要ることの実証、赤なら床の係数 k を 0 より上に置く材料）。
 */
Deno.test({
  name: "e2e(実GPU): 故障注入 ③ attentionCompute だけ外した席は census で赤になる（帯は記録）",
  ignore: !RUNNABLE,
  fn: async () => {
    const { resolution, band } = FAULT_CASE;
    const id = `fault3-no-attention-a8-${formatResolution(resolution)}`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const manifest = readManifest();
      const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
      const declared = quantOf(manifest, MODEL, PRACTICAL).session;
      const { attentionCompute: _dropped, ...withoutAttention } = declared;
      const injected = overrideQuantSession(
        JSON.parse(manifestText as string),
        MODEL,
        PRACTICAL,
        withoutAttention,
      );
      const referenceRun = await observeStep1(
        denoDirectory(ASSETS_DIR),
        reference,
        resolution,
        SEED,
      );
      const faulty = await withManifestOverride(
        ASSETS_DIR,
        injected,
        (source) => observeStep1(source, PRACTICAL, resolution, SEED),
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL}（attentionCompute 外し）` },
          measured,
          band,
        ),
      );
      let bandVerdict = "緑";
      try {
        judgeAb(measured, band, id);
      } catch (cause) {
        bandVerdict = `赤（${errorText(cause)}）`;
      }
      console.log(`[e2e] anima 故障注入 ③ ${describeAb(measured)} / 帯 ${bandVerdict}`);
      assertThrows(
        () => assertSeatsApplied(faulty.census, declared, id),
        Error,
        "attentionCompute",
      );
      return { status: "pass", note: `帯 ${bandVerdict}` };
    });
  },
});
