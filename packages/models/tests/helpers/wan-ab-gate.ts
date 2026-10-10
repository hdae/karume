/**
 * Wan の**自機 A/B 門**（ADR 0110 決定 5 / ADR 0120 決定 4）の 1 ケースぶんの段 — 2.1（`e2e_wan_ab_test.ts`）と 2.2
 * （`e2e_wan_ti2v_ab_test.ts`）が共有する。2 世代の違い（pipeline の class・census 表の系列・モデル・席・配布形の置き場・
 * 要求の固定値・ログの接頭）は {@link WanAbSubject} で受け、手順・判定の順・故障注入の形・結果の席への記録は 1 本にする（世代で
 * 門の主張が割れないように）。「何を赤にするか」の判定そのものは `ab-gate.ts` の純関数。
 *
 * 1 ケース（= フレーム数 1 つ）の手順（{@link runWanAbCase}）:
 *
 * 1. 参照席（manifest から導く — `referencePartnerOf`）の pipeline で **step 1 の潜在**（2 ステップの通しの 1 step 目 — 席が
 *    最初に効いた直後・決定 5-4）を採り、購読側の throw で生成を打ち切る（`captureThenAbort`）。dispose して解放を待つ。
 * 2. 実用席で同じことを **2 回**（独立の 2 pipeline）。
 * 3. 門を 3 つ掛ける: 決定性（実用席の 2 回がビット一致 — 決定 6 の MUST）・census（実用席の DiT の各パスを束の census 表の行と
 *    ちょうどの本数で突き合わせ、attention の 2 ノブは 1 本以上 — 決定 5-1）・帯（`judgeAb` — 決定 5-2 / 5-3）。
 * 4. 求められたケースだけ故障注入 5 件（どれも「門が赤になる」ことを assert する — 決定 5-5）。
 *
 * VRAM: pipeline は 1 本ずつ順に張る（同時に張らない — 構築では Session を張らず、DiT の Session は step 1 の打ち切りで
 * 畳まれる）。観測 1 回ごとに `settleReleases` で解放を待つ（B570 の `destroy()` の解放の遅れ — ADR 0120 リスク 6）。
 *
 * MUST: 故障注入の席は同じ配布形を symlink で借りた一時 manifest（`withManifestOverride`）で張り、元の配布形は 1 バイトも
 * 書き換えない。
 */

import { assert, assertEquals, assertThrows } from "@std/assert";
import { type DistributionSource, parseManifest, type SessionSpec } from "@karume/hub";
import { denoDirectory } from "@karume/hub/deno";
import type { GpuContext, SessionDiagnostics, SessionOptions } from "@karume/runtime";
import type {
  GeneratedVideo,
  WanFromPretrainedOptions,
  WanPrompt,
  WanTi2vGenerateRequest,
} from "../../wan.ts";
import { assertSeatsApplied, mergeCensus } from "../../../runtime/tests/helpers/pipeline-census.ts";
import { type Results, runRecordedCase } from "../../../runtime/tests/helpers/results.ts";
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
} from "./ab-gate.ts";
import {
  assertRowCensus,
  type BundleCensusRow,
  censusRowOf,
  effectiveSessionOptions,
} from "./census-table.ts";
import { withManifestOverride } from "./manifest-override.ts";
import { settleReleases } from "./settle-releases.ts";

/** 観測点の名前（`comparisons[].output`）。 */
export const WAN_AB_OBSERVATION = "step1-latent";

/**
 * 入力（各世代の seed 経路の sha 行と同じプロンプトと seed・寸法 / guidance / shift と I2V の条件画像は {@link WanAbSubject}
 * の `request` — 省いた世代は manifest の既定）。step 1 の σ は steps と shift で決まるので、steps は 2 に固定する（2 ステップの通しの
 * 1 step 目）。
 */
const PROMPT = "boxing-cats";
const SEED = 42;
const STEPS = 2;
/**
 * step 1 までの DiT のパス数（CFG は 1 step に uncond → cond の 2 パス — guidance > 1 が前提。前提が崩れたら
 * {@link observeStep1} のパス数の検査が落とす）。
 */
const PASSES_TO_STEP1 = 2;

/**
 * A/B に要る pipeline の面（`WanPipeline` と `WanTi2vPipeline` が構造的に満たす）。要求は広い方の型（2.2 の I2V の欄を
 * 含む `WanTi2vGenerateRequest`）で受ける — 2.1 の `generate` は狭い型を受けるが、引数は反変なので面を満たす。
 */
export type WanAbPipeline = AsyncDisposable & {
  readonly prompts: readonly WanPrompt[];
  generate(request: WanTi2vGenerateRequest): Promise<GeneratedVideo>;
};

/** 世代ごとの違い（門の手順と判定は {@link runWanAbCase} の 1 本）。 */
export type WanAbSubject = {
  /** ログの接頭（`[e2e] <label> A/B …`）。 */
  readonly label: string;
  /** census 表と受理表の系列（manifest の pipeline 名）。 */
  readonly family: "wan" | "wan-ti2v";
  /** 対象のモデル（帯はモデル × 席 × フレーム数ごとの宣言なので明示する）。 */
  readonly model: string;
  /** 対象の実用席。 */
  readonly practical: string;
  /** 配布形の置き場（`karume.json` と容器）。 */
  readonly assetsRoot: URL;
  /** 配布形の `karume.json` の生のテキスト（席の宣言と一時 manifest の差し替えの元）。 */
  readonly manifestText: string;
  /** 実測と決着の席。 */
  readonly results: Results;
  /** 世代の pipeline の構築口（`WanPipeline.fromPretrained` / `WanTi2vPipeline.fromPretrained`）。 */
  readonly fromPretrained: (
    source: DistributionSource,
    options: WanFromPretrainedOptions,
  ) => Promise<WanAbPipeline>;
  /**
   * 要求の固定値（寸法と、step 1 の潜在を決める guidance / shift — I2V なら条件画像も）。省けば世代の既定（manifest の
   * `pipelineConfig` と既定の寸法）に乗る — 2.1 は sha 行と同じく既定に乗る。明示する世代は全部を明示する（寸法だけ明示して
   * σ と CFG のパス数を既定に残すと、既定が動いたときに帯の前提が黙って動く）。I2V の step 1 の潜在は `copyLatents` の
   * スケジューラの状態（先頭の潜在フレームを条件で置き換える前 — `WanGenerateEvent` の doc）。
   */
  readonly request?: {
    readonly width: number;
    readonly height: number;
    readonly guidance: number;
    readonly shift: number;
    /**
     * I2V の条件画像（Wan2.2 だけ — 省けば T2V）。寸法は上の `width` / `height` で明示する（画像の縦横比での選択に
     * 預けない — 帯の前提の寸法が画像で黙って動かないように）。2.1 の pipeline に渡すと `generate` が
     * `ModelInputError` で拒む（fail loudly — 型は 2 世代で共有の 1 本）。渡すと各観測で encoder の 3 グラフの run が
     * 届いたことを見る（画像が黙って落ちて T2V で回る退行を、帯と census の外で掴む）。
     */
    readonly image?: WanTi2vGenerateRequest["image"];
    /** 条件画像の寸法の合わせ方（`image` を渡したときだけ — 省けば公式の `"crop"`）。 */
    readonly fit?: WanTi2vGenerateRequest["fit"];
  };
};

/** 1 run ぶんの観測（step 1 の潜在と、そこまでの DiT の各パスの census）。 */
export type WanStep1Observation = {
  readonly latent: Float32Array<ArrayBuffer>;
  /** DiT の 1 パス = 1 要素（{@link PASSES_TO_STEP1} 本）。 */
  readonly passes: readonly SessionDiagnostics["lastRunPipelines"][];
};

export const errorText = (cause: unknown): string =>
  cause instanceof Error ? cause.message : String(cause);

/**
 * `quant` の pipeline を `gpu` の上に組み、step 1 の潜在を採って打ち切る（dispose して解放を待つまで）。census は
 * DiT の run だけ（quant の `session` が効くのは DiT の Session だけ — `wan/tile-decode.ts` の VAE の段は `{}`）。
 */
const observeStep1 = async (
  gpu: GpuContext,
  subject: WanAbSubject,
  source: DistributionSource,
  quant: string,
  frames: number,
  seed: number,
): Promise<WanStep1Observation> => {
  const where = `${quant} / ${frames} フレーム / seed ${seed}`;
  const passes: SessionDiagnostics["lastRunPipelines"][] = [];
  /** 診断が届いた run の component（I2V の条件画像が encoder の段まで届いたことの検査）。 */
  const ran = new Set<string>();
  try {
    await using pipeline = await subject.fromPretrained(source, {
      gpu,
      model: subject.model,
      quant,
      // A/B の対象は DiT の席 — 事前計算の埋め込みを明示する（既定の "gpu" は umT5 の段が乗り、プロンプトの
      // 文脈も別の値になる — 帯を採った前提が崩れる）。
      textEncoder: "precomputed",
      onRunDiagnostics: (component, diagnostics) => {
        ran.add(component);
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
          ...subject.request,
          onEvent: (event) => {
            if (event.kind === "denoise-step" && event.step === 1) {
              capture(event.copyLatents().data);
            }
          },
        }),
    );
    assertEquals(passes.length, PASSES_TO_STEP1, `${where}: step 1 までの DiT のパス数`);
    if (subject.request?.image !== undefined) {
      // 条件画像が要求から落ちて T2V の経路で回る退行では、両席とも T2V の潜在になって帯（relRMS）は T2V の値で緑のまま
      // 通り、census も DiT しか見ないので掴めない。encoder の 3 グラフの run が届いたことで、I2V の経路を通ったことを縛る。
      for (const component of ["vae_encoder_pre", "vae_encoder_attn", "vae_encoder_post"]) {
        assert(
          ran.has(component),
          `${where}: 条件画像を渡したのに ${component} の run が届いていない`,
        );
      }
    }
    return { latent, passes };
  } finally {
    // 次の観測の pipeline の確保の前に、この観測の確保の解放を待つ（B570 — settleReleases）。
    await settleReleases(gpu);
  }
};

/** 配布形の実用席の宣言（manifest の `session` の字面 — 一時 manifest の差し替えの元）。 */
export const declaredSessionOf = (subject: WanAbSubject): SessionSpec =>
  quantOf(parseManifest(subject.manifestText), subject.model, subject.practical).session;

/** 宣言を家族の合成（受理表 — `WAN_SESSION_POLICY`）に通した実効設定（census 表の鍵・census の期待）。 */
export const effectiveSessionOf = (
  subject: WanAbSubject,
  session: SessionSpec,
  where: string,
): SessionOptions => effectiveSessionOptions(subject.family, session, {}, where);

/** 実用席の束の census 表の行（無ければ fail loudly — この門は実用席そのものの門なので、行の欠落は表の欠落）。 */
const practicalRowOf = (
  subject: WanAbSubject,
  effective: SessionOptions,
  where: string,
): BundleCensusRow => {
  const row = censusRowOf(subject.family, subject.model, effective);
  if (row === undefined) {
    throw new Error(`${where}: 束の census 表（helpers/census-table.ts）に行が無い`);
  }
  return row;
};

/**
 * 実用席の census（決定 5-1）: 各パスを表の行とちょうどの本数で突き合わせ（linear）、束の各ノブの変種が 1 本以上
 * 走ったことを見る（attention — 表に書かない規約: 行ブロックの枚数が device の束縛上限で変わる）。
 */
const assertPracticalCensus = (
  row: BundleCensusRow,
  observation: WanStep1Observation,
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
  if (row.family !== "wan" && row.family !== "wan-ti2v") {
    throw new Error(`census 表の行が wan の系列でない（${row.family}）`);
  }
  const pass = row.census.transformer?.pass;
  const linear = pass?.linearCompute;
  if (pass === undefined || linear === undefined) {
    throw new Error(`census 表の ${row.family} の行に transformer/pass の linearCompute が無い`);
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

/**
 * `session` だけを差し替えた実用席を一時 manifest で張って step 1 を採る（故障注入とノブ単位の席 — 重みと他の席は
 * 触らない）。
 */
export const observeOverridden = (
  gpu: GpuContext,
  subject: WanAbSubject,
  session: Readonly<Record<string, unknown>>,
  frames: number,
): Promise<WanStep1Observation> =>
  withManifestOverride(
    subject.assetsRoot,
    overrideQuantSession(
      JSON.parse(subject.manifestText),
      subject.model,
      subject.practical,
      session,
    ),
    (source) => observeStep1(gpu, subject, source, subject.practical, frames, SEED),
  );

/** {@link runWanAbCase} が呼び手へ返す観測（2.1 のノブ単位の記録が A/B の観測を使い回す）。 */
export type WanAbRun = {
  /** 参照席の名前（manifest から導いた値）。 */
  readonly reference: string;
  readonly referenceRun: WanStep1Observation;
  /** 実用席の 1 回目。 */
  readonly first: WanStep1Observation;
};

/** 1 ケースの指定。 */
export type WanAbCase = {
  /** 結果の席の `cases[].id`（テストの名前と同じ値を呼び手が組む）。 */
  readonly id: string;
  readonly frames: number;
  /** 宣言の帯（導出前は `undefined` — `judgeAb` が実測を出して赤で止まる）。 */
  readonly band: AbBand | undefined;
  /** 故障注入 5 件を回すか（門の空振りを示すのに長さは要らないので短いケースだけ）。 */
  readonly faults: boolean;
};

/**
 * 1 ケース（フレーム数 1 つ）の自機 A/B 門を `t` の step として回す（モジュール doc の手順）。device の取得・アダプタの
 * 検査・後始末は呼び手が持つ（Deno.test 1 本で device 1 つ）。
 */
export const runWanAbCase = async (
  t: Deno.TestContext,
  gpu: GpuContext,
  subject: WanAbSubject,
  { id, frames, band, faults }: WanAbCase,
): Promise<WanAbRun> => {
  const { label, practical, results } = subject;
  const manifest = parseManifest(subject.manifestText);
  const reference = referencePartnerOf(manifest, subject.model, practical);
  const effective = effectiveSessionOf(
    subject,
    quantOf(manifest, subject.model, practical).session,
    id,
  );
  const row = practicalRowOf(subject, effective, id);
  const source = denoDirectory(subject.assetsRoot);
  const referenceRun = await observeStep1(gpu, subject, source, reference, frames, SEED);
  const first = await observeStep1(gpu, subject, source, practical, frames, SEED);
  const second = await observeStep1(gpu, subject, source, practical, frames, SEED);

  await t.step(`${id}: 決定性・census・帯`, async () => {
    await runRecordedCase(results, { id, failureNote: errorText }, (recorded) => {
      const measured = measureAb(first.latent, referenceRun.latent);
      recorded.comparisons.push(
        comparisonOf(
          { output: WAN_AB_OBSERVATION, reference, practical },
          measured,
          band,
        ),
      );
      const census = describeBundleCensus(mergeCensus(first.passes, id), effective);
      console.log(
        `[e2e] ${label} A/B ${id}: ${reference} → ${practical} ${
          describeAb(measured)
        } / census ${census}`,
      );
      assertBitIdentical(first.latent, second.latent, `${id}: 実用席の 2 回`);
      assertPracticalCensus(row, first, effective, `${id}: 実用席の transformer`);
      judgeAb(measured, band, id);
      return Promise.resolve({ status: "pass", note: `census ${census}` });
    });
  });

  if (!faults) return { reference, referenceRun, first };

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
          const faulty = await observeOverridden(gpu, subject, {}, frames);
          const measured = measureAb(faulty.latent, referenceRun.latent);
          recorded.comparisons.push(
            comparisonOf(
              { output: WAN_AB_OBSERVATION, reference, practical: `${practical}（session 空）` },
              measured,
              band,
            ),
          );
          console.log(`[e2e] ${label} 故障注入 ① ${describeAb(measured)}`);
          assertThrows(() => judgeAb(measured, band, faultId), Error, "床の失敗");
        },
      );
    },
  );

  // NOTE: ⑤ は帯の上限で赤になることを見るので、帯が未導出（`undefined`）の間は `judgeAb` が「帯が未導出」で投げて
  // この step も赤になる（帯の宣言の後に有効 — 2.1 の段 3 / 4 と同じ扱い。② はその間も効く）。
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
            subject,
            denoDirectory(subject.assetsRoot),
            practical,
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
                output: WAN_AB_OBSERVATION,
                reference,
                practical: `${practical}（seed ${SEED + 1}）`,
              },
              measured,
              band,
            ),
          );
          console.log(`[e2e] ${label} 故障注入 ⑤ ${describeAb(measured)}`);
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
          const { attentionCompute: _dropped, ...withoutAttention } = declaredSessionOf(subject);
          const faulty = await observeOverridden(gpu, subject, withoutAttention, frames);
          const measured = measureAb(faulty.latent, referenceRun.latent);
          recorded.comparisons.push(
            comparisonOf(
              {
                output: WAN_AB_OBSERVATION,
                reference,
                practical: `${practical}（attentionCompute 外し）`,
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
          console.log(`[e2e] ${label} 故障注入 ③ ${describeAb(measured)} / 帯 ${bandVerdict}`);
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

  return { reference, referenceRun, first };
};
