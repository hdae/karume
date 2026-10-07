/**
 * Wan2.1 の**自機 A/B 門**（実 GPU・ADR 0110 決定 5 / ADR 0120 決定 4 / 5・段 4 と段 3 のノブ単位の記録）。実用席
 * `f16+dit8-a8-attn8-s16`（`session` に linear a8 / attention a8 / S の f16 格納を束ねた席）を、**同じ機・同じ i8 重み**で
 * `session` が空の参照席（manifest から導く — {@link referencePartnerOf}。今の配布形では `f16+dit8`）と比べる。比較相手は
 * 同機の参照層で、torch golden は使わない（`f16` 席は重みが違うので床が恒真になる — ADR 0110 決定 5 ②）。
 *
 * 1 ケース（= フレーム数 1 つ）の手順（形は `e2e_anima_ab_test.ts` と同じ）:
 *
 * 1. 参照席の pipeline で **step 1 の潜在**（2 ステップの通しの 1 step 目 — 席が最初に効いた直後・決定 5-4）を採り、
 *    購読側の throw で生成を打ち切る（{@link captureThenAbort}）。dispose して解放を待つ。
 * 2. 実用席で同じことを **2 回**（独立の 2 pipeline）。
 * 3. 門を 3 つ掛ける:
 *    - **決定性**（決定 6 の MUST）— 実用席の 2 回の潜在がビット一致（{@link assertBitIdentical}）。
 *    - **census**（決定 5-1）— 実用席の DiT の各パスで linear 307 本ちょうどが i8a8・参照経路の linear 0 本（束の
 *      census 表の行 — {@link assertRowCensus}）。attention の 2 ノブは表に書かない規約（行ブロックの枚数が device の
 *      束縛上限で変わる）なので 1 本以上（`assertSeatsApplied`）。
 *    - **帯**（決定 5-2 / 5-3）— relRMS が {@link CASES} の上限の内・非有限なし・参照席と 1 bit 以上違う（{@link judgeAb}）。
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
 * （{@link withManifestOverride}）で張り、元の配布形は 1 バイトも書き換えない。
 */

import { assertEquals, assertThrows } from "@std/assert";
import {
  type DistributionSource,
  type Manifest,
  parseManifest,
  type SessionSpec,
} from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import type { GpuContext, SessionDiagnostics, SessionOptions } from "@karume/runtime";
import { WanPipeline } from "../wan.ts";
import { assertSeatsApplied, mergeCensus } from "../../runtime/tests/helpers/pipeline-census.ts";
import { assertAdapterMatchesEnvironment } from "../../runtime/tests/helpers/environment.ts";
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
  type BundleCensusRow,
  censusRowOf,
  effectiveSessionOptions,
} from "./helpers/census-table.ts";
import { acquireTestGpu, GPU_AVAILABLE } from "./helpers/gpu.ts";
import { withManifestOverride } from "./helpers/manifest-override.ts";
import { WAN_ASSEMBLE_COMMAND } from "./helpers/wan-distribution.ts";
import { readTextIfPresent } from "./helpers/read-if-present.ts";
import { settleReleases } from "./helpers/settle-releases.ts";

/** 配布形の置き場（`dist.py --pipeline wan` の出力先）。 */
const ASSETS_DIR = new URL("../../../models/karume-wan2.1/", import.meta.url);
/** 実測と決着の置き場（`outputs/verify/<環境キー>/<日付>_wan-ab/` — 消して安全）。 */
const results = openResults("wan-ab");

/** 対象のモデル（帯はモデル × 席 × フレーム数ごとの宣言なので明示する）。 */
const MODEL = "t2v-1.3b";
/** 対象の実用席（ADR 0120 決定 1）。 */
const PRACTICAL = "f16+dit8-a8-attn8-s16";
/** 観測点の名前（`comparisons[].output`）。 */
const OBSERVATION = "step1-latent";

/**
 * 入力（`e2e_wan_pipeline_test.ts` の seed 経路の sha 行と同じ要求 — 832×480・guidance / shift は manifest の既定）。
 * step 1 の σ は steps で決まるので 2 に固定する（2 ステップの通しの 1 step 目）。
 */
const PROMPT = "boxing-cats";
const SEED = 42;
const STEPS = 2;
/** step 1 までの DiT のパス数（CFG は 1 step に uncond → cond の 2 パス — guidance 5 > 1）。 */
const PASSES_TO_STEP1 = 2;

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

const readManifest = (): Manifest => parseManifest(manifestText as string);

/** 1 run ぶんの観測（step 1 の潜在と、そこまでの DiT の各パスの census）。 */
type Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  /** DiT の 1 パス = 1 要素（{@link PASSES_TO_STEP1} 本）。 */
  readonly passes: readonly SessionDiagnostics["lastRunPipelines"][];
};

/**
 * `quant` の pipeline を `gpu` の上に組み、step 1 の潜在を採って打ち切る（dispose して解放を待つまで）。census は
 * DiT の run だけ（quant の `session` が効くのは DiT の Session だけ — `wan/pipeline.ts` の VAE の段は `{}`）。
 */
const observeStep1 = async (
  gpu: GpuContext,
  source: DistributionSource,
  quant: string,
  frames: number,
  seed: number,
): Promise<Observation> => {
  const where = `${quant} / ${frames} フレーム / seed ${seed}`;
  const passes: SessionDiagnostics["lastRunPipelines"][] = [];
  try {
    await using pipeline = await WanPipeline.fromPretrained(source, {
      gpu,
      model: MODEL,
      quant,
      // A/B の対象は DiT の席 — 事前計算の埋め込みを明示する（既定の "gpu" は umT5 の段が乗り、プロンプトの
      // 文脈も別の値になる — 段 3 / 4 で採った帯の前提が崩れる）。
      textEncoder: "precomputed",
      onRunDiagnostics: (component, diagnostics) => {
        if (component === "transformer") passes.push(diagnostics.lastRunPipelines);
      },
    });
    const prompt = pipeline.prompts.find(({ name }) => name === PROMPT)?.prompt;
    if (prompt === undefined) throw new Error(`${where}: 埋め込み資産に '${PROMPT}' が無い`);
    const latent = await captureThenAbort<Float32Array<ArrayBuffer>>(
      where,
      (capture) =>
        pipeline.generate({
          prompt,
          seed,
          steps: STEPS,
          frames,
          onEvent: (event) => {
            if (event.kind === "denoise-step" && event.step === 1) {
              capture(event.copyLatents().data);
            }
          },
        }),
    );
    assertEquals(passes.length, PASSES_TO_STEP1, `${where}: step 1 までの DiT のパス数`);
    return { latent, passes };
  } finally {
    // 次の観測の pipeline の確保の前に、この観測の確保の解放を待つ（B570 — {@link settleReleases}）。
    await settleReleases(gpu);
  }
};

/** 配布形の実用席の宣言（manifest の `session` の字面 — 一時 manifest の差し替えの元）。 */
const declaredSession = (manifest: Manifest): SessionSpec =>
  quantOf(manifest, MODEL, PRACTICAL).session;

/** 宣言を家族の合成（受理表 — `WAN_SESSION_POLICY`）に通した実効設定（census 表の鍵・census の期待）。 */
const effectiveOf = (session: SessionSpec, where: string): SessionOptions =>
  effectiveSessionOptions("wan", session, {}, where);

/** 実用席の束の census 表の行（無ければ fail loudly — この門は実用席そのものの門なので、行の欠落は表の欠落）。 */
const practicalRowOf = (effective: SessionOptions, where: string): BundleCensusRow => {
  const row = censusRowOf("wan", MODEL, effective);
  if (row === undefined) {
    throw new Error(`${where}: 束の census 表（helpers/census-table.ts）に行が無い`);
  }
  return row;
};

/**
 * 実用席の census（決定 5-1）: 各パスを表の行とちょうどの本数で突き合わせ（linear）、束の各ノブの変種が 1 本以上
 * 走ったことを見る（attention — 表に書かない規約）。
 */
const assertPracticalCensus = (
  row: BundleCensusRow,
  observation: Observation,
  effective: SessionOptions,
  where: string,
): void => {
  const checked = assertRowCensus(
    row,
    new Map([["pass", observation.passes]]),
    () => "transformer",
    where,
  );
  assertEquals(checked, observation.passes.length, `${where}: 表の期待と突き合わせたパスの数`);
  assertSeatsApplied(mergeCensus(observation.passes, where), effective, where);
};

/**
 * census 表の行の linear の期待本数を `delta` だけずらした写し（故障注入 ④ — 「ちょうど」の検査が実物の census で
 * 1 本の過不足を掴むこと）。
 */
const shiftedLinearRow = (row: BundleCensusRow, delta: number): BundleCensusRow => {
  if (row.family !== "wan") throw new Error(`census 表の行が wan でない（${row.family}）`);
  const pass = row.census.transformer?.pass;
  const linear = pass?.linearCompute;
  if (pass === undefined || linear === undefined) {
    throw new Error("census 表の wan の行に transformer/pass の linearCompute が無い");
  }
  return {
    ...row,
    census: {
      transformer: {
        pass: { ...pass, linearCompute: { ...linear, variant: linear.variant + delta } },
      },
    },
  };
};

const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

/**
 * `session` だけを差し替えた実用席を一時 manifest で張って step 1 を採る（故障注入とノブ単位の席 — 重みと他の席は
 * 触らない）。
 */
const observeOverridden = (
  gpu: GpuContext,
  session: Readonly<Record<string, unknown>>,
  frames: number,
): Promise<Observation> =>
  withManifestOverride(
    ASSETS_DIR,
    overrideQuantSession(JSON.parse(manifestText as string), MODEL, PRACTICAL, session),
    (source) => observeStep1(gpu, source, PRACTICAL, frames, SEED),
  );

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
  frames: number,
  reference: { readonly name: string; readonly observation: Observation },
  practical: Observation,
): Promise<void> => {
  await t.step(`ノブ単位の記録（${frames} フレーム・門ではない）`, async () => {
    const id = `knobs-${frames}f`;
    await runRecordedCase(results, { id, failureNote: errorText }, async (recorded) => {
      const rungs = knobRungs(declaredSession(readManifest()), id);
      const notes: string[] = [];
      let previous = reference.observation;
      for (const [index, { label, session }] of rungs.entries()) {
        const where = `${id} ${label}`;
        const observed = index === rungs.length - 1
          ? practical
          : await observeOverridden(gpu, session, frames);
        const effective = effectiveOf(session, where);
        const census = mergeCensus(observed.passes, where);
        // 段の席にそのノブが届いていなければ、その段の relRMS は名乗るものを測っていない。
        assertSeatsApplied(census, effective, where);
        const toReference = measureAb(observed.latent, reference.observation.latent);
        const toPrevious = measureAb(observed.latent, previous.latent);
        recorded.comparisons.push(
          comparisonOf(
            {
              output: OBSERVATION,
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
        const manifest = readManifest();
        const reference = referencePartnerOf(manifest, MODEL, PRACTICAL);
        const effective = effectiveOf(declaredSession(manifest), id);
        const row = practicalRowOf(effective, id);
        const source = denoDirectory(ASSETS_DIR);
        const referenceRun = await observeStep1(gpu, source, reference, frames, SEED);
        const first = await observeStep1(gpu, source, PRACTICAL, frames, SEED);
        const second = await observeStep1(gpu, source, PRACTICAL, frames, SEED);

        await t.step(`${id}: 決定性・census・帯`, async () => {
          await runRecordedCase(results, { id, failureNote: errorText }, (recorded) => {
            const measured = measureAb(first.latent, referenceRun.latent);
            recorded.comparisons.push(
              comparisonOf(
                { output: OBSERVATION, reference, practical: PRACTICAL },
                measured,
                band,
              ),
            );
            const census = describeBundleCensus(mergeCensus(first.passes, id), effective);
            console.log(
              `[e2e] wan A/B ${id}: ${reference} → ${PRACTICAL} ${
                describeAb(measured)
              } / census ${census}`,
            );
            assertBitIdentical(first.latent, second.latent, `${id}: 実用席の 2 回`);
            assertPracticalCensus(row, first, effective, `${id}: 実用席の transformer`);
            judgeAb(measured, band, id);
            return Promise.resolve({ status: "pass", note: `census ${census}` });
          });
        });

        if (faults) {
          // --- 故障注入（各門が空振りでないことの実証 — ADR 0110 決定 5-5）-----------------------
          // どれも「門が赤になること」を assert する（緑 = 門が故障を見逃した）。①〜③ は ADR 0120 決定 4 の 3 件、
          // ④⑤ は census の「ちょうど」と上限が実物の値で効くことの裏取り。

          await t.step(
            "故障注入 ① 実用席の session を空にした manifest（= 参照席そのもの）は床で赤",
            async () => {
              const faultId = `fault1-empty-session-${frames}f`;
              await runRecordedCase(
                results,
                { id: faultId, failureNote: errorText },
                async (recorded) => {
                  const faulty = await observeOverridden(gpu, {}, frames);
                  const measured = measureAb(faulty.latent, referenceRun.latent);
                  recorded.comparisons.push(
                    comparisonOf(
                      { output: OBSERVATION, reference, practical: `${PRACTICAL}（session 空）` },
                      measured,
                      band,
                    ),
                  );
                  console.log(`[e2e] wan 故障注入 ① ${describeAb(measured)}`);
                  assertThrows(() => judgeAb(measured, band, faultId), Error, "床の失敗");
                },
              );
            },
          );

          await t.step(
            "故障注入 ② / ⑤ 実用席を別 seed で回す — 1 回目との決定性・参照席に対する上限が赤",
            async () => {
              const faultId = `fault2-5-other-seed-${frames}f`;
              await runRecordedCase(
                results,
                { id: faultId, failureNote: errorText },
                async (recorded) => {
                  const shifted = await observeStep1(
                    gpu,
                    denoDirectory(ASSETS_DIR),
                    PRACTICAL,
                    frames,
                    SEED + 1,
                  );
                  // ② 決定性: 同じ席の 2 回目だけ入力が違う形（ビット一致の門が実物の潜在で差を掴む）。
                  assertThrows(
                    () => assertBitIdentical(first.latent, shifted.latent, "故障注入 ②"),
                    Error,
                    "ビット一致しない",
                  );
                  // ⑤ 上限: 観測点が参照と無関係な潜在になった形（崩壊の代理 — 別の初期ノイズの step 1）。
                  const measured = measureAb(shifted.latent, referenceRun.latent);
                  recorded.comparisons.push(
                    comparisonOf(
                      {
                        output: OBSERVATION,
                        reference,
                        practical: `${PRACTICAL}（seed ${SEED + 1}）`,
                      },
                      measured,
                      band,
                    ),
                  );
                  console.log(`[e2e] wan 故障注入 ⑤ ${describeAb(measured)}`);
                  assertThrows(() => judgeAb(measured, band, faultId), Error, "上限の失敗");
                },
              );
            },
          );

          // ③ 帯の判定は記録だけ（ADR 0030 決定 5 の再演 — 積み重ねた席の後段の失効は数値で掴みにくい。帯が緑の
          // ままなら census が要ることの実証、赤なら床の係数を 0 より上に置く材料）。
          await t.step(
            "故障注入 ③ attentionCompute だけ外した席は census で赤（帯は記録）",
            async () => {
              const faultId = `fault3-no-attention-a8-${frames}f`;
              await runRecordedCase(
                results,
                { id: faultId, failureNote: errorText },
                async (recorded) => {
                  const { attentionCompute: _dropped, ...withoutAttention } = declaredSession(
                    readManifest(),
                  );
                  const faulty = await observeOverridden(gpu, withoutAttention, frames);
                  const measured = measureAb(faulty.latent, referenceRun.latent);
                  recorded.comparisons.push(
                    comparisonOf(
                      {
                        output: OBSERVATION,
                        reference,
                        practical: `${PRACTICAL}（attentionCompute 外し）`,
                      },
                      measured,
                      band,
                    ),
                  );
                  let bandVerdict = "緑";
                  try {
                    judgeAb(measured, band, faultId);
                  } catch (cause) {
                    bandVerdict = `赤（${errorText(cause)}）`;
                  }
                  console.log(`[e2e] wan 故障注入 ③ ${describeAb(measured)} / 帯 ${bandVerdict}`);
                  assertThrows(
                    () => assertPracticalCensus(row, faulty, effective, faultId),
                    Error,
                    "attentionCompute",
                  );
                  return { status: "pass", note: `帯 ${bandVerdict}` };
                },
              );
            },
          );

          await t.step(
            "故障注入 ④ census 表の linear の本数を 1 ずらすと実物の census で赤（GPU の追加なし）",
            () => {
              for (const delta of [-1, 1]) {
                assertThrows(
                  () =>
                    assertPracticalCensus(
                      shiftedLinearRow(row, delta),
                      first,
                      effective,
                      `故障注入 ④（${delta > 0 ? "+" : ""}${delta}）`,
                    ),
                  Error,
                  "linearCompute=a8 の変種",
                );
              }
            },
          );
        }

        if (knobs) {
          await recordKnobLadder(
            t,
            gpu,
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
