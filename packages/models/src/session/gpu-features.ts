/**
 * `Quant` が宣言する **GPU 前提**（`gpuFeatures` = device 生成時にしか要求できない GPU
 * feature — ADR 0038 §3 / `requiredLimits` = 満たすべき device limit の最小値 — ADR 0038 §7）
 * → `acquireGpu` の要求と、突き合わせ相手（取得済み `GpuContext` またはアダプタ実測値）に
 * 対する検査の写像（**パイプライン非依存の共通処理** — feature 面は 7 家族の `fromAssets` が、
 * limits 面は 8 家族の `fromPretrained` が同じ形で使う）。
 *
 * MUST: barrel には出さない（同居する `options.ts` と同じ理由 — 配布形の宣言を runtime の
 * ノブへ翻訳する内部機構で、利用者が触る面ではない）。`session` の写像と同居させてあるのは、
 * どちらも**同じ `Quant` の欄**を runtime のノブへ翻訳する処理だから。
 *
 * MUST: 表のキー集合は `Required<GpuFeaturesSpec>` の**網羅** — `GpuFeaturesSpec` に feature が
 * 増えたらこの宣言が型検査で落ちる。7 家族が `quant.gpuFeatures?.<欄> === true` を 1 行ずつ
 * 独立に読む形だと、欄が増えたとき**どの家族も型検査を通ったまま**その要求を黙って落とす
 * （= 配布形が要求した能力を持たない device で実行が進む沈黙劣化）。
 *
 * MUST: 「要求（`acquireGpu` へ何を渡すか）」と「検査（共有 GPU で有効になっているか）」を
 * 同じ 1 エントリに置く。別の表に分けると、片方だけ追随した形（要求はするが検査しない）が
 * 型検査を通ってしまう。
 *
 * NOTE: `requiredLimits` にこの網羅表の形は流用できない — feature が「全欄を知って要求を組む」
 * 全射なのに対し、limits は**部分写像**（配布形が書いた limit だけを見て、書かれていない
 * limit には何も主張しない — 欄なし = WebGPU 保証既定で動く、が ADR 0089 決定 3 の意味論）。
 * 語彙のずれ自体は `tests/limit_vocabulary_test.ts` が hub / runtime の両側で縛る。
 */

import type { GpuFeaturesSpec, RequiredLimitsSpec } from "@karume/hub";
import {
  type AcquireGpuOptions,
  type GpuContext,
  readAdapterLimits,
  type RequiredLimits,
} from "@karume/runtime";

type GpuFeatureEntry = {
  /** 配布形がこの feature を要求しているか。 */
  readonly wanted: (spec: GpuFeaturesSpec) => boolean;
  /** 要求するときに `acquireGpu` へ足す欄。 */
  readonly acquire: AcquireGpuOptions;
  /** 取得済みの GpuContext で実際に有効になっているか。 */
  readonly enabled: (gpu: GpuContext) => boolean;
  /** 診断に載せる feature の綴り（WebGPU 側の名前）。 */
  readonly label: string;
  /** 診断に載せる取り直し方。 */
  readonly reacquire: string;
};

const FEATURES: { readonly [K in keyof Required<GpuFeaturesSpec>]: GpuFeatureEntry } = {
  shaderF16: {
    wanted: (spec) => spec.shaderF16 === true,
    acquire: { shaderF16: true },
    enabled: (gpu) => gpu.shaderF16Enabled,
    label: "shader-f16",
    reacquire: "acquireGpu({ shaderF16: true })",
  },
};

/** 宣言された feature だけを要求する `acquireGpu` のオプションを組む。 */
export const toAcquireGpuOptions = (spec: GpuFeaturesSpec | undefined): AcquireGpuOptions => {
  if (spec === undefined) return {};
  let options: AcquireGpuOptions = {};
  for (const feature of Object.values(FEATURES)) {
    if (feature.wanted(spec)) options = { ...options, ...feature.acquire };
  }
  return options;
};

/**
 * 宣言された feature が取得済みの GpuContext で有効になっていることを確かめる。
 *
 * MUST: 共有 GPU（`options.gpu`）を渡された経路では feature を要求できない（device 作成時に
 * しか要求できない — ADR 0028）ので、能力が足りないことを**ここで**名指しして落とす。通すと
 * Session 構築まで進んでから落ちるか、黙って別の経路へ縮退する。
 *
 * @param where 診断の主語（`"AnimaPipeline: quant 'f16+dit8-a8-attn8-s16'"`）。
 */
export const assertGpuFeaturesGranted = (
  spec: GpuFeaturesSpec | undefined,
  gpu: GpuContext,
  where: string,
): void => {
  if (spec === undefined) return;
  for (const feature of Object.values(FEATURES)) {
    if (feature.wanted(spec) && !feature.enabled(gpu)) {
      throw new Error(
        `${where} は ${feature.label} を要求するが、渡された GpuContext で` +
          `有効になっていない（${feature.reacquire} で取り直す）`,
      );
    }
  }
};

/**
 * 配布形が宣言した `requiredLimits` を、GPU 側の limits と突き合わせる（ADR 0089 決定 5）。
 *
 * 見るのは **spec に書かれた limit だけ**（部分写像 — モジュール doc の NOTE）。不足は
 * **全件を 1 回で**列挙して落とす（1 本ずつ潰させると、環境が足りないことの全体像が出ない）。
 *
 * @param limits 突き合わせ相手。共有 GPU なら `GpuContext.limits`、自前取得の経路なら
 *   `readAdapterLimits()` の実測値 — どちらも `planRequiredLimits` の同じ式なので同じ物差し。
 * @param where 診断の主語（`"AnimaPipeline: quant 'f16'"`）。
 */
export const assertRequiredLimitsSatisfied = (
  spec: RequiredLimitsSpec | undefined,
  limits: RequiredLimits,
  where: string,
): void => {
  if (spec === undefined) return;
  // 型注釈で index signature 面へ移す（キーの語彙は hub / runtime で一致していることが
  // `tests/limit_vocabulary_test.ts` の門で、ここは名前で引くだけ）。
  const declared: Readonly<Record<string, number | undefined>> = spec;
  const granted: Readonly<Record<string, number | undefined>> = limits;
  const shortfalls: string[] = [];
  for (const name of Object.keys(declared)) {
    const required = declared[name];
    if (required === undefined) continue;
    const actual = granted[name];
    if (actual === undefined) {
      // 語彙のずれ（hub が受理した名前を runtime が持たない）。黙って読み飛ばすと
      // 「宣言された limit が誰にも見られない」= 静かな頭打ちに戻る。
      throw new Error(
        `${where} が要求する limit '${name}' は runtime の limits に無い` +
          `（hub と runtime の語彙のずれ — tests/limit_vocabulary_test.ts）`,
      );
    }
    if (actual < required) shortfalls.push(`${name}: 要求 ${required} / この GPU は ${actual}`);
  }
  if (shortfalls.length > 0) {
    throw new Error(
      `${where} が要求する device limit をこの GPU が満たさない: ${shortfalls.join(", ")}` +
        `（別の quant を選ぶ）`,
    );
  }
};

/**
 * `fromPretrained` の admission 席から呼ぶ **DL 前**の limits 検査（ADR 0089 決定 5）。
 *
 * 共有 GPU（`options.gpu`）を渡されていればその `limits`、自前取得の経路ならアダプタを
 * 1 度だけ読んで（{@link readAdapterLimits} — device は作らず、アダプタも持ち回らない）
 * 突き合わせる。GPU の無い環境ではここが `GpuUnavailableError` になり、**重みを 1 バイトも
 * 落とさない**（それが席の存在理由 — 従来は数 GiB を転送した後に落ちていた）。
 *
 * MUST: 宣言が無い（`spec === undefined`）配布形ではアダプタを取りに行かない。取りに行くと、
 * 何も要求していない配布形が GPU 無し環境で**ロードすらできなく**なる（既定スペックで動く
 * という「欄なし」の意味論を壊す）。
 *
 * NOTE: 共有 GPU の経路は各家族の admission 関数が同じ検査を既に通している（あちらは
 * `fromAssets` 面も守る席）。ここが同じ比較をもう一度するのは、`fromPretrained` の 8 家族が
 * **同じ 1 行**で書けるようにするため — 純粋な数値比較なので、二重に見ても副作用は無い。
 */
export const assertRequiredLimitsBeforeDownload = async (
  spec: RequiredLimitsSpec | undefined,
  sharedGpu: GpuContext | undefined,
  where: string,
): Promise<void> => {
  if (spec === undefined) return;
  const limits = sharedGpu === undefined ? await readAdapterLimits() : sharedGpu.limits;
  assertRequiredLimitsSatisfied(spec, limits, where);
};
