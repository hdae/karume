/**
 * BiRefNet 系の α マットの後処理（グラフの出す logit の地図 → 入力画像と同じ寸法の 8bit α）。
 *
 * `pipeline.ts` のモジュール doc の段 4。GPU にもグラフにも触らない純関数なので、パイプラインの
 * 外に置き、CPU だけのテストで直接縛る。
 *
 * ## 上流の実際の手順（系列ごとに違う）
 *
 * - **BiRefNet_HR**（同梱 `handler.py:131-137`・モデルカードの利用例も同じ順）: 半精度のモデルが
 *   出す logit → 半精度の `sigmoid` → `transforms.ToPILImage()`（torchvision 0.28 は
 *   `(x * 255).astype(np.uint8)` = 切り捨て）→ Pillow の `pred_pil.resize(image.size)`（resample
 *   を省くと BICUBIC・8bit の 1 バンド）。
 * - **Lucida**（モデルカード `README.md:86-89`）: f32 の logit → `sigmoid` →
 *   `transforms.functional.resize`（テンソル枝・bilinear・antialias 既定）→
 *   `(alpha.numpy() * 255).astype("uint8")` = 切り捨て。
 *
 * ## 今の経路はどちらの上流とも違う独自の経路（裁定 4 で A/B 中）
 *
 * 今の経路は σ を f64 で計算して f32 に格納 → `resizePlaneF32`（antialias bilinear・f64・パスの
 * 間で丸めない）→ `Math.round(255·α)`。段の順序（sigmoid → resize → 量子化）は Lucida の上流と
 * 同じだが、HR の上流とは量子化と resize の順序・補間法が違い、両系列とも丸め（四捨五入か
 * 切り捨てか）が違う。上流の手順と目視で比べて採否を決める作業（裁定 4 — R6B-001）の最中である。
 */

import { resizePlaneF32 } from "../image/preprocess.ts";

/** 8bit の α の最大値。 */
const ALPHA_MAX = 255;

/** 入力と同じ寸法の α マット（画素あたり 1 バイト・行優先）。 */
export type AlphaMatte = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/**
 * グラフの出す logit の地図 → 入力と同じ寸法の 8bit α（`pipeline.ts` のモジュール doc の段 4）。
 *
 * 段の順序は sigmoid → resize → 量子化（モジュール doc の「今の経路」）。8bit へ落とすのは
 * **この配布形の出力形式の決定**で、量子化は四捨五入（α の 1 LSB は目に見えず、切り捨ては
 * 系統的に前景を薄くする — 上流はどちらの系列も切り捨て）。
 *
 * NOTE: `export` はパイプラインとテストのため（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */
export const matteFromLogits = (
  logits: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): AlphaMatte => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    const raw = logits[index];
    // MUST: 非有限値を黙って返さない。α の量子化（`Uint8Array` 代入）は NaN を 0 =
    // 「完全に透明な画素」という**正常に見えるマット**へ変換してしまう。resize は重み付き和で
    // NaN を伝播するだけなので、発生源に一番近いここで落とす（`anima/image.ts` と同じ検査点）。
    if (!Number.isFinite(raw)) {
      const x = index % sourceWidth;
      const y = Math.floor(index / sourceWidth);
      throw new Error(`birefnet: マットの logit (x=${x}, y=${y}) が非有限値`);
    }
    alpha[index] = 1 / (1 + Math.exp(-raw));
  }
  const scaled = resizePlaneF32(alpha, sourceWidth, sourceHeight, width, height);
  const data = new Uint8Array(scaled.length) as Uint8Array<ArrayBuffer>;
  for (let index = 0; index < scaled.length; index += 1) {
    data[index] = Math.round(scaled[index] * ALPHA_MAX);
  }
  return { data, width, height };
};
