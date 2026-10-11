/**
 * BiRefNet 系の α マットの後処理（グラフの出す logit の地図 → 入力画像と同じ寸法の 8bit α）。
 *
 * `pipeline.ts` のモジュール doc の段 4。GPU にもグラフにも触らない純関数なので、パイプラインの
 * 外に置き、CPU だけのテスト（`tests/birefnet_matte_test.ts` — 上流の関数で作った golden
 * `tests/fixtures/birefnet-matte/` と突き合わせる）で直接縛る。
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
 * ## 手順（{@link MatteProcedure} — ADR 0125）
 *
 * 上流との一致は計算結果の数値の水準（または数値の意図の水準）で取り、上流の誤りや速度のための
 * 選択は写さない（ADR 0125 決定 1）。写すのは系列ごとの補間のカーネルと段の順序で、写さないのは
 * 切り捨て（全手順とも四捨五入 = `Math.round(255·α)`・0.5 は切り上げ）と HR の半精度。
 *
 * - `float-bilinear`（Lucida の上流の意図 = **今の経路そのもの**・パイプラインの既定）: σ を f64 で
 *   計算して f32 に格納 → `resizePlaneF32`（antialias bilinear・f64・パスの間で丸めない）→
 *   `Math.round(255·α)`。
 * - `u8-pil-bicubic`（HR の上流の意図）: σ を f64 で計算 → `Math.round(255σ)` で 8bit の平面 →
 *   Pillow 12.3.0 の 8bit・1 バンドの BICUBIC（`src/image/pillow-resample.ts` — 22 bit の固定小数点・
 *   中間 uint8・寸法が同じ軸はパスを飛ばす）。
 *
 * どちらの系列の配布形にどちらの手順を当てるかは、HR で上流と今の経路を目視で比べる A/B（裁定 4 —
 * R6B-001）の結果で決める。それまでパイプラインは今の経路（`float-bilinear`）だけを使い、手順の
 * 引数はこのモジュールの内部の口に留める（`mod.ts` / サブパス面には出さない — ADR 0008）。
 */

import { resizePlaneF32 } from "../image/preprocess.ts";
import { resizeU8PlaneBicubic } from "../image/pillow-resample.ts";

/** 8bit の α の最大値。 */
const ALPHA_MAX = 255;

/** 入力と同じ寸法の α マット（画素あたり 1 バイト・行優先）。 */
export type AlphaMatte = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
};

/** 後処理の手順の名前（モジュール doc「手順」— 名前の正本は ADR 0125）。 */
export const MATTE_PROCEDURES = ["float-bilinear", "u8-pil-bicubic"] as const;

/** {@link MATTE_PROCEDURES} の 1 つ。 */
export type MatteProcedure = typeof MATTE_PROCEDURES[number];

/** 1 つの手順の本体（logit の地図 → 入力と同じ寸法の 8bit α の画素列）。 */
type ProcedureBody = (
  logits: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
) => Uint8Array<ArrayBuffer>;

/**
 * logit 1 つの σ（f64）。
 *
 * MUST: 非有限値を黙って返さない。α の量子化（`Uint8Array` 代入）は NaN を 0 = 「完全に透明な
 * 画素」という**正常に見えるマット**へ変換してしまう。resize は重み付き和で NaN を伝播するだけ
 * なので、発生源に一番近いここで落とす（`anima/image.ts` と同じ検査点）。
 */
const sigmoidAt = (
  logits: Float32Array<ArrayBuffer>,
  index: number,
  sourceWidth: number,
): number => {
  const raw = logits[index];
  if (!Number.isFinite(raw)) {
    const x = index % sourceWidth;
    const y = Math.floor(index / sourceWidth);
    throw new Error(`birefnet: マットの logit (x=${x}, y=${y}) が非有限値`);
  }
  return 1 / (1 + Math.exp(-raw));
};

/** `float-bilinear`（モジュール doc — 今の経路そのもの）。 */
const floatBilinear: ProcedureBody = (logits, sourceWidth, sourceHeight, width, height) => {
  const alpha = new Float32Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    alpha[index] = sigmoidAt(logits, index, sourceWidth);
  }
  const scaled = resizePlaneF32(alpha, sourceWidth, sourceHeight, width, height);
  const data = new Uint8Array(scaled.length) as Uint8Array<ArrayBuffer>;
  for (let index = 0; index < scaled.length; index += 1) {
    data[index] = Math.round(scaled[index] * ALPHA_MAX);
  }
  return data;
};

/**
 * `u8-pil-bicubic`（モジュール doc — HR の上流の意図）。
 *
 * MUST: σ を f32 に格納してから丸めない（`float-bilinear` と違い、f64 の σ を直接 `Math.round` に
 * 渡す — golden の生成器〈`tools/export-recipes/birefnet/matte_fixture.py`〉と同じ式。f32 へ格納すると
 * `k + 0.5` の近くの標本の丸めの向きが変わりうる）。
 */
const u8PilBicubic: ProcedureBody = (logits, sourceWidth, sourceHeight, width, height) => {
  const quantized = new Uint8Array(logits.length);
  for (let index = 0; index < logits.length; index += 1) {
    quantized[index] = Math.round(sigmoidAt(logits, index, sourceWidth) * ALPHA_MAX);
  }
  return resizeU8PlaneBicubic(quantized, sourceWidth, sourceHeight, width, height);
};

const PROCEDURES: Readonly<Record<MatteProcedure, ProcedureBody>> = {
  "float-bilinear": floatBilinear,
  "u8-pil-bicubic": u8PilBicubic,
};

/**
 * グラフの出す logit の地図 → 入力と同じ寸法の 8bit α（`pipeline.ts` のモジュール doc の段 4）。
 *
 * `procedure` を省くと今の経路（`float-bilinear`）。8bit へ落とすのは**この配布形の出力形式の
 * 決定**で、量子化はどの手順も四捨五入（ADR 0125 — 上流はどちらの系列も切り捨て）。
 *
 * NOTE: `export` はパイプライン・テスト・A/B の道具（`tools/birefnet-ab/`）のため（`mod.ts` /
 * サブパス面には出さない — ADR 0008）。
 */
export const matteFromLogits = (
  logits: Float32Array<ArrayBuffer>,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
  procedure: MatteProcedure = "float-bilinear",
): AlphaMatte => {
  // MUST: `Object.hasOwn` で引く — JS の呼び手が `"toString"` 等を渡すと Object.prototype の
  // メンバを手順として受けてしまう。名前の綴り違いなので素の `Error`（ADR 0107 決定 3）。
  if (!Object.hasOwn(PROCEDURES, procedure)) {
    throw new Error(
      `birefnet: 後処理の手順 ${JSON.stringify(procedure)} は未対応（期待 ${
        MATTE_PROCEDURES.map((name) => `'${name}'`).join(" / ")
      }）`,
    );
  }
  return {
    data: PROCEDURES[procedure](logits, sourceWidth, sourceHeight, width, height),
    width,
    height,
  };
};
