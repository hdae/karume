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
 * - The exact key census of the practical seat lives here too (the "key census" test below): over
 *   one full generation on the resident DiT path, every `dit` / `dit_context` run dispatches
 *   exactly the census-table number of i8a8 GEMMs and `quantize_rows`, zero f32 linear kernels,
 *   and `dit_context` runs once per generation. There is no band against the `i8` torch golden:
 *   activation quantization cannot be a numeric-parity gate (ADR 0025 decision 6 / ADR 0026), and
 *   the silent-f32-fallback floor that band guarded is covered by the census and fault injection ①.
 *
 * 帯の導出規則は `e2e_anima_ab_test.ts` の `CASES` の doc と同じ（上限 = 実測 × 2 程度・床 =
 * 1 bit 以上違う MUST + 示せたときだけ係数）。
 *
 * MUST: 資産は `models/karume-irodori-v4.1-small/`（untracked・実 GPU 機のローカル資産 —
 * examples の台本既定の配布形）。無い環境と GPU 無し環境は理由を出して**明示 SKIP** する
 * （ADR 0005）。
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { type DistributionSource, type Manifest, parseManifest } from "@karume/hub";
import { nodeDirectory } from "@karume/hub/node";
import type { SessionDiagnostics } from "@karume/runtime";
import { IrodoriPipeline, type IrodoriRunComponent } from "../../models/irodori.ts";
import {
  assertSeatsApplied,
  type CensusRow,
  mergeCensus,
  requireCensus,
  SEAT_SIGNATURES,
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
import {
  assertRowCensus,
  censusRowOf,
  effectiveSessionOptions,
} from "../../models/tests/helpers/census-table.ts";
import { GPU_AVAILABLE } from "../../models/tests/helpers/gpu.ts";
import { withManifestOverride } from "../../models/tests/helpers/manifest-override.ts";
import { readTextIfPresent } from "../../models/tests/helpers/read-if-present.ts";

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
  const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(where, (capture) =>
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
  test.skipIf(!RUNNABLE)(
    `e2e(実GPU): ${id} の step 1 潜在が参照席と決定性・census・帯で釣り合う（自機 A/B）`,
    async () => {
      await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const declared = quantOf(manifest, MODEL, PRACTICAL).session;
        const source = nodeDirectory(ASSETS_DIR);
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
  );
}

// --- Key census of the practical seat (exact dispatch counts per run) -----------------

/** census の生成で使う発話長（秒）。`duration` を回さず S を小さく固定する。 */
const CENSUS_SECONDS = 1;

/** census を採る部品（quant の `session` が降りる 2 本 — 観測席の綴り）。 */
type CensusComponent = Extract<IrodoriRunComponent, "dit" | "dit-context">;
const CENSUS_COMPONENTS: readonly CensusComponent[] = ["dit", "dit-context"];
const isCensusComponent = (component: IrodoriRunComponent): component is CensusComponent =>
  (CENSUS_COMPONENTS as readonly IrodoriRunComponent[]).includes(component);
/** census 表の相（`dit` は denoise の step ごと・`dit_context` は生成 1 回の run）。 */
const PHASE_OF: Readonly<Record<CensusComponent, string>> = { dit: "step", "dit-context": "run" };

/**
 * 部品ごとの run 1 回あたりに走る `quantize_rows` の dispatch 数（`dit` は 1 forward・
 * `dit_context` は生成 1 回の 1 run）。
 *
 * DiT の linear は分割前の 1 グラフで 317 本（k ∈ {32, 192, 512, 768, 1280, 3680} — 全て 4 の
 * 倍数で i8-a8 適格・量子化 recon の sizeBreakdown）。ADR 0114 で条件側 K/V 射影 72 本〈12
 * ブロック × text / speaker / caption × K / V〉を `dit_context` へ割り出したので、`dit` 245 本 +
 * `dit_context` 72 本になった — **グラフが変わったための期待値の変更**で、本数を緩めたのでは
 * ない。合計 317 の検査は置かない（部品ごとにちょうどで縛れば足り、合計は片方の取り違えを
 * 打ち消しうる）。i8a8 の linear は「活性を per-token i8 へ落とす `quantize_rows` → 整数内積の
 * GEMM」の対で走るので、期待は本数の関数として書ける。GEMM 側（席 `linearCompute: "a8"` の
 * 変種キー）の本数は束の census 表（`helpers/census-table.ts`）が持つ — 席の本数の正本を
 * 1 か所にするため。
 *
 * To re-derive after a graph change, read the measured counts from the census log line this test
 * prints (the source is `lastRunPipelines`, which does not depend on timing).
 */
const QUANTIZE_ROWS_PER_RUN: Readonly<Record<CensusComponent, number>> = {
  dit: 245,
  "dit-context": 72,
};

/** パイプラインキーの分類（綴りの正本は `src/kernels/linear{,-i8a8}.ts` / `quantize-rows.ts`）。 */
const isLinearKey = (key: string): boolean => key.startsWith("linear:");
const isI8a8 = SEAT_SIGNATURES.linearCompute.a8;
const isQuantizeRows = (key: string): boolean => key.startsWith("quantize_rows:");

/** 1 run ぶんの分類済み dispatch 数。 */
type KeyCensus = {
  readonly i8a8Linear: number;
  readonly quantizeRows: number;
  readonly plainLinear: number;
};

const censusOf = (diagnostics: SessionDiagnostics, component: CensusComponent): KeyCensus => {
  // MUST: census の無い run を「0 本」として数えない — キー検査が黙って空振りする
  //（requireCensus が undefined / 空を落とす）。
  const entries = requireCensus(diagnostics.lastRunPipelines, component);
  let i8a8Linear = 0;
  let quantizeRows = 0;
  let plainLinear = 0;
  for (const entry of entries) {
    if (isLinearKey(entry.key)) {
      if (isI8a8(entry.key)) i8a8Linear += entry.dispatchCount;
      else plainLinear += entry.dispatchCount;
    } else if (isQuantizeRows(entry.key)) {
      quantizeRows += entry.dispatchCount;
    }
  }
  return { i8a8Linear, quantizeRows, plainLinear };
};

test.skipIf(!RUNNABLE)(
  `e2e(実GPU): dit / dit_context の run が i8a8 GEMM と quantize_rows だけで回る（キー census）`,
  async () => {
    const manifest = readManifest();
    // 束の census 表の行（鍵は quant 席の宣言を家族の合成に通した実効設定）。この門は `i8-a8`
    // 席そのものの門なので、行が無いのは表の欠落として落とす。
    const where = `irodori ${MODEL}/${PRACTICAL}`;
    const row = censusRowOf(
      "irodori",
      MODEL,
      effectiveSessionOptions("irodori", quantOf(manifest, MODEL, PRACTICAL).session, {}, where),
    );
    assert(row !== undefined, `${where}: 束の census 表（helpers/census-table.ts）に行が無い`);
    const observed: Record<CensusComponent, KeyCensus[]> = { dit: [], "dit-context": [] };
    const runs: Record<CensusComponent, SessionDiagnostics["lastRunPipelines"][]> = {
      dit: [],
      "dit-context": [],
    };
    // No `onEvent` and no timing device: the DiT loop takes the resident path (the production
    // path), whose diagnostics arrive once per `enqueue` — the A/B cases above observe the host
    // path instead.
    await using pipeline = await IrodoriPipeline.fromPretrained(nodeDirectory(ASSETS_DIR), {
      model: MODEL,
      quant: PRACTICAL,
      onRunDiagnostics: (component, diagnostics) => {
        if (!isCensusComponent(component)) return;
        observed[component].push(censusOf(diagnostics, component));
        runs[component].push(diagnostics.lastRunPipelines);
      },
    });
    // `durationSeconds` で S を固定するのは `duration` を回さず短く保つため（見たいのは
    // 長さではなく「何が走ったか」）。参照も caption も無い最小構成で足りる。
    await pipeline.generateLatent({
      text: FAULT_CASE.text,
      seed: SEED,
      durationSeconds: CENSUS_SECONDS,
    });

    if (observed.dit.length === 0) throw new Error("dit の run が 1 回も観測されなかった");
    // 条件側 K/V は生成 1 回に 1 run（forward ごとに回っていたら ADR 0114 のホイストが外れている）。
    assertEquals(observed["dit-context"].length, 1, "dit_context の run が生成 1 回に 1 回でない");
    for (const component of CENSUS_COMPONENTS) {
      const first = observed[component][0];
      const count = observed[component].length;
      console.log(
        `[e2e] irodori ${MODEL}/${PRACTICAL} census: ${component} run ${count} 回 / ` +
          `i8a8 linear ${first.i8a8Linear} 本 / quantize_rows ${first.quantizeRows} 本 / ` +
          `f32 linear ${first.plainLinear} 本`,
      );
      // 全 forward が同じ計画で走る（uncond はマスク還元 — ADR 0047 決定 1）ので、本数は
      // run ごとに揺れない。揺れたら「一部の run だけ別の経路へ落ちた」の直接の証拠になる。
      for (const [index, census] of observed[component].entries()) {
        assertEquals(census, first, `${component} の run ${index} だけ dispatch の内訳が違う`);
      }
      assertEquals(
        first.plainLinear,
        0,
        `${component}: f32 骨格の linear が走っている（適格判定が外れて一部が黙って f32 経路へ落ちた）`,
      );
    }
    // 席の変種（i8a8 GEMM）と参照経路に残った linear を run ごとにちょうどの本数で見る
    // （相 → 部品は census 表の軸の綴り — `dit` は step・`dit_context` は run）。
    const checked = assertRowCensus(
      row,
      new Map(CENSUS_COMPONENTS.map((component) => [PHASE_OF[component], runs[component]])),
      (phase) => (phase === PHASE_OF.dit ? "dit" : "dit-context"),
      where,
    );
    assert(checked > 0, `${where}: 表の期待と突き合わせた run が 1 本も無い`);
    for (const component of CENSUS_COMPONENTS) {
      assertEquals(
        observed[component][0].quantizeRows,
        QUANTIZE_ROWS_PER_RUN[component],
        `${component} の quantize_rows の dispatch 数が期待と違う（DiT の linear 本数か融合が動いた）`,
      );
    }
  },
);

// --- 故障注入（ADR 0110 決定 5-5 — 門が赤になることを assert する）-------------------

test.skipIf(!RUNNABLE)(
  "e2e(実GPU): 故障注入 ① 実用席の session を空にした manifest は床で赤になる",
  async () => {
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
        nodeDirectory(ASSETS_DIR),
        reference,
        FAULT_CASE,
        SEED,
      );
      const faulty = await withManifestOverride(ASSETS_DIR, injected, (source) =>
        observeStep1(source, PRACTICAL, FAULT_CASE, SEED),
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
);

test.skipIf(!RUNNABLE)(
  "e2e(実GPU): 故障注入 ② 実用席の 2 回目だけ別 seed にすると決定性の門が赤になる",
  async () => {
    const source = nodeDirectory(ASSETS_DIR);
    const first = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED);
    const second = await observeStep1(source, PRACTICAL, FAULT_CASE, SEED + 1);
    assertThrows(
      () => assertBitIdentical(first.latent, second.latent, "故障注入 ②"),
      Error,
      "ビット一致しない",
    );
  },
);
