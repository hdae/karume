/**
 * Irodori の**自機 A/B 門**（実 GPU・ADR 0110 決定 5）。実用席 `i8-a8`（既定席 — `dit` の linear を
 * 活性まで整数内積で回す `linearCompute: "a8"`）を、**同じ機・同じ i8 重み**で `session` が空の
 * 参照席（manifest から導く — {@link referencePartnerOf}。今の配布形では `i8`）と比べる。
 *
 * 形は `e2e_anima_ab_test.ts` と同じ（手順・門・故障注入の対応はそちらのモジュール doc）。違いは:
 *
 * - 観測点は **step 1 の latent**（`denoise-step` の `copyLatents()` — CFG の内側の forward 数では
 *   なく 1 step 完了後の潜在 `[frames, latentDim]`）。`durationSeconds` で S を固定し、`duration`
 *   グラフを回さない（quant の `session` は `dit` にだけ渡るので、S の決定は席に依らない）。
 * - 束のノブは `linearCompute` の 1 つだけなので、故障注入は ①（`session` を空にした manifest →
 *   床）と ②（2 回目だけ別 seed → 決定性）の 2 本。「1 ノブだけ外す」③ は束が 1 ノブの席では ① と
 *   同じ形になるので置かない。
 * - `e2e_irodori_w8a8_test.ts`（`i8` の torch golden との判別帯 + キー本数の census）は残す —
 *   あちらは torch 側との距離、こちらは同機の参照層との距離で、主張が違う。
 *
 * 帯の導出規則は `e2e_anima_ab_test.ts` の `CASES` の doc と同じ（上限 = 実測 × 2 程度・床 =
 * 1 bit 以上違う MUST + 示せたときだけ係数）。
 *
 * MUST: 資産は `models/karume-irodori-v4.1-small/`（untracked・実 GPU 機のローカル資産 —
 * examples の台本既定の配布形）。無い環境と GPU 無し環境は理由を出して**明示 SKIP** する
 * （ADR 0005）。
 */

import { assertThrows } from "@std/assert";
import { type DistributionSource, type Manifest, parseManifest } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import type { SessionDiagnostics } from "@karume/runtime";
import { IrodoriPipeline } from "../mod.ts";
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

/** 配布形の置き場（`karume dist --pipeline irodori` の v4.1 の出力先）。 */
const ASSETS_DIR = new URL("../../../models/karume-irodori-v4.1-small/", import.meta.url);
/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_irodori-ab/` — 消して安全）。 */
const results = openResults("irodori-ab");

const MODEL = "v4.1-small";
const PRACTICAL = "i8-a8";
const OBSERVATION = "step1-latent";
/** 発話長（秒）。S を固定して `duration` グラフを回さない。 */
const DURATION_SECONDS = 3;
const SEED = 1235;

/** 1 ケースの入力と帯。 */
type AbCase = {
  readonly name: string;
  readonly text: string;
  readonly caption?: string;
  readonly band: AbBand | undefined;
};

/**
 * ケースと帯（`band: undefined` = 未導出 — 門は実測を出して赤で止まる）。テキストと caption は
 * `e2e_irodori_wav_test.ts` と同じ文（参照音声は使わない — speaker 経路は席に依らない）。
 *
 * 導出表（2026-09-26・Intel Arc B570 / Deno・参照席 `i8`・上限 = 実測 × 2 を有効数字 2 桁へ）:
 *
 * | ケース  | 実測 relRMS | 実測 maxAbs | floor | ceiling | 採った日・機 |
 * | ------- | ----------- | ----------- | ----- | ------- | ------------ |
 * | no-ref  | 3.5879e-3   | 1.4885e-2   | 0     | 7.2e-3  | 2026-09-26 B570 |
 * | caption | 5.8457e-3   | 2.2464e-2   | 0     | 1.2e-2  | 2026-09-26 B570 |
 */
const CASES: readonly AbCase[] = [
  {
    name: "no-ref",
    text: "本日はお越しいただき、誠にありがとうございます。",
    band: { floor: 0, ceiling: 7.2e-3 },
  },
  {
    name: "caption",
    text: "今日は近くの店まで歩いて行きました。とても良い天気でしたね。",
    caption:
      "若く元気な女性の声。カフェの店員のように、明るくハキハキとした少し高めのトーンで話している。",
    band: { floor: 0, ceiling: 1.2e-2 },
  },
];

/** 故障注入を回すケース（最小構成の no-ref）。 */
const FAULT_CASE = CASES[0];

const manifestText = await readTextIfPresent(new URL("karume.json", ASSETS_DIR));
if (manifestText === undefined) {
  console.warn(
    `[karume] ${ASSETS_DIR.pathname} に karume.json が無いため Irodori の自機 A/B 門を SKIP する` +
      "（cd tools/exporter && uv run karume dist --pipeline irodori）",
  );
}
const RUNNABLE = GPU_AVAILABLE && manifestText !== undefined;

const readManifest = (): Manifest => parseManifest(manifestText as string);

type Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  readonly census: readonly CensusRow[];
};

/**
 * `quant` の pipeline を組み、step 1 の潜在を採って打ち切る。census は `dit` の run だけ
 * （`onEvent` を渡すと DiT ループはホスト経路で回り、forward ごとに観測席へ届く）。
 */
const observeStep1 = async (
  source: DistributionSource,
  quant: string,
  input: AbCase,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / ${input.name} / seed ${seed}`;
  const runs: SessionDiagnostics["lastRunPipelines"][] = [];
  await using pipeline = await IrodoriPipeline.fromPretrained(source, {
    model: MODEL,
    quant,
    onRunDiagnostics: (component, diagnostics) => {
      if (component === "dit") runs.push(diagnostics.lastRunPipelines);
    },
  });
  const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(
    where,
    (capture) =>
      pipeline.generateLatent({
        text: input.text,
        ...(input.caption === undefined ? {} : { caption: input.caption }),
        seed,
        durationSeconds: DURATION_SECONDS,
        onEvent: (event) => {
          if (event.kind === "denoise-step" && event.step === 1) capture(event.copyLatents().data);
        },
      }),
  );
  return { latent, census: mergeCensus(runs, `${where} の dit`) };
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

for (const input of CASES) {
  const id = `${PRACTICAL}-${input.name}`;
  Deno.test({
    name: `e2e(実GPU): ${id} の step 1 潜在が参照席と決定性・census・帯で釣り合う（自機 A/B）`,
    ignore: !RUNNABLE,
    fn: async () => {
      await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const declared = quantOf(manifest, MODEL, PRACTICAL).session;
        const source = denoDirectory(ASSETS_DIR);
        const referenceRun = await observeStep1(source, reference, input, SEED);
        const first = await observeStep1(source, PRACTICAL, input, SEED);
        const second = await observeStep1(source, PRACTICAL, input, SEED);
        const measured = measureAb(first.latent, referenceRun.latent);
        recorded.comparisons.push(
          comparisonOf(
            { output: OBSERVATION, reference, practical: PRACTICAL },
            measured,
            input.band,
          ),
        );
        const census = describeBundleCensus(first.census, declared);
        console.log(
          `[e2e] irodori A/B ${id}: ${reference} → ${PRACTICAL} ${describeAb(measured)} ` +
            `/ census ${census}`,
        );
        assertBitIdentical(first.latent, second.latent, `${id}: 実用席の 2 回`);
        assertSeatsApplied(first.census, declared, `${id}: 実用席の dit`);
        judgeAb(measured, input.band, id);
        return { status: "pass", note: `census ${census}` };
      });
    },
  });
}

// --- 故障注入（ADR 0110 決定 5-5 — 門が赤になることを assert する）-------------------

Deno.test({
  name: "e2e(実GPU): 故障注入 ① 実用席の session を空にした manifest は床で赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const id = `fault1-empty-session-${FAULT_CASE.name}`;
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
        FAULT_CASE,
        SEED,
      );
      const faulty = await withManifestOverride(
        ASSETS_DIR,
        injected,
        (source) => observeStep1(source, PRACTICAL, FAULT_CASE, SEED),
      );
      const measured = measureAb(faulty.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: OBSERVATION, reference, practical: `${PRACTICAL}（session 空）` },
          measured,
          FAULT_CASE.band,
        ),
      );
      console.log(`[e2e] irodori 故障注入 ① ${describeAb(measured)}`);
      assertThrows(() => judgeAb(measured, FAULT_CASE.band, id), Error, "床の失敗");
    });
  },
});

Deno.test({
  name: "e2e(実GPU): 故障注入 ② 実用席の 2 回目だけ別 seed にすると決定性の門が赤になる",
  ignore: !RUNNABLE,
  fn: async () => {
    const source = denoDirectory(ASSETS_DIR);
    const first = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED);
    const second = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED + 1);
    assertThrows(
      () => assertBitIdentical(first.latent, second.latent, "故障注入 ②"),
      Error,
      "ビット一致しない",
    );
  },
});
