/**
 * manifest の `gpuFeatures`（device 生成時にしか要求できない GPU feature — ADR 0038 §3）→
 * `acquireGpu` の要求と、取得済み `GpuContext` に対する検査の写像
 * （**パイプライン非依存の共通処理** — 7 家族の `fromAssets` が同じ形で使う）。
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
 */

import type { GpuFeaturesSpec } from "@karume/hub";
import type { AcquireGpuOptions, GpuContext } from "@karume/runtime";

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
