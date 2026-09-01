/**
 * 途中 latent → RGB プレビューの線形近似（VAE を通さない）。
 *
 * `denoise-step` イベントの `copyLatents()` が返す latent を、チャネル方向の 1 次結合だけで
 * 色に起こす。VAE は DiT を解放した**後**にしかロードできない（VRAM の MUST — `pipeline.ts`
 * のモジュール doc）ので、step ごとの経過を絵で見せる手段はこの近似しかない。
 *
 * MUST: これは**厳密な decode ではない**。16 チャネルを 3 チャネルへ潰す線形射影なので、
 * 細部・質感・文字は再現しない（潰れた latent 解像度の 1/8 サムネイルとして見る）。生成結果の
 * 正しさをこの出力で判定しない — 出す絵の正本は `generate` の返り値である。
 */

import { imageToRgba } from "./image.ts";

/**
 * 16ch latent → RGB の線形近似係数（行 = latent のチャネル・列 = R,G,B）と定数項。
 *
 * 出所: https://huggingface.co/spaces/KAD001/Wan2GP/blob/main/shared/RGB_factors.py の
 * `"qwen"` family（ComfyUI 系のプレビューアが同じ表を使う）。
 *
 * MUST: `export` しない。これはプレビューの見え方を決めるだけの内部定数で、公開すると
 * 「消費側が自前で当てる」経路ができ、較正空間の取り違え（下記）が再び起こりうる。
 *
 * ## 較正空間の実測（2026-08-24）
 *
 * この表が**どちらの latent 空間**で較正されているかの一次記録は見つからなかったので、実測で
 * 決めた。リグ = local の turbo 配布形（`models/karume-anima` の turbo / 既定 quant）・512×512・
 * seed 0・manifest 既定 steps（8）で 1 枚生成し、最終 step の latent を捕まえて
 * (A) 正規化 latent 直当て / (B) `denormalizeLatents` を通した逆正規化 latent 直当て の 2 通りを、
 * 生成画像を 8×8 box 平均で latent 解像度へ落とした参照と突き合わせた。
 *
 * - Pearson 相関（clamp 前の線形出力）: A = R 0.9794 / G 0.9742 / B 0.9759 / 輝度 0.9771
 *   （平均 0.9767）、B = R 0.9798 / G 0.9743 / B 0.9730 / 輝度 0.9771（平均 0.9761）。
 *   **相関だけでは決まらない**（差 0.0006）— 逆正規化は latent のチャネルごとのアフィン変換で、
 *   射影後も「係数を重み付けし直した射影」にしかならず、相関はそこにほぼ不変だから。
 * - 決め手は**絶対値の合い方**: 参照との RMSE が A 0.0797 / B 0.3614、値域 `[0,1]` から
 *   飛び出して clamp で潰れる画素の割合が A 1.2% / B 51.7%。B は絵として白飛びして成立しない。
 *   よって**較正空間は正規化 latent**（A）。ComfyUI の慣行とも一致する。
 * - 同じ実測で**出力の値域規約**も決まった: 線形出力は `[0,1]` ではなく **VAE sample と同じ
 *   `[-1,1]`** に載る（`[0,1]` として扱うと RMSE 0.5454 / clamp 50.2% で半分が潰れる）。
 *   だから最終画像とまったく同じ postprocess（{@link imageToRgba}）を通す。
 */
const LATENT_RGB_FACTORS: readonly (readonly [number, number, number])[] = [
  [-0.1299, -0.1692, 0.2932],
  [0.0671, 0.0406, 0.0442],
  [0.3568, 0.2548, 0.1747],
  [0.0372, 0.2344, 0.1420],
  [0.0313, 0.0189, -0.0328],
  [0.0296, -0.0956, -0.0665],
  [-0.3477, -0.4059, -0.2925],
  [0.0166, 0.1902, 0.1975],
  [-0.0412, 0.0267, -0.1364],
  [-0.1293, 0.0740, 0.1636],
  [0.0680, 0.3019, 0.1128],
  [0.0032, 0.0581, 0.0639],
  [-0.1251, 0.0927, 0.1699],
  [0.0060, -0.0633, 0.0005],
  [0.3477, 0.2275, 0.2950],
  [0.1984, 0.0913, 0.1861],
];

/** {@link LATENT_RGB_FACTORS} と対の定数項（`latent_rgb_factors_bias`）。 */
const LATENT_RGB_BIAS: readonly [number, number, number] = [-0.1835, -0.0868, -0.3360];

/**
 * 途中 latent 1 枚を RGBA プレビューへ近似する。返る画像は **latent 解像度**（元画像の 1/8）。
 *
 * MUST: 渡すのは `denoise-step` の `copyLatents()` の返り値**そのまま**（= 正規化 latent）で、
 * `denormalizeLatents` に通してはならない。係数は正規化空間で較正されている（モジュール内の
 * 実測記録）ので、逆正規化した値を渡すと白飛びした別物になる。
 *
 * MUST: **B=1 前提**を入口で検査する。チャネルごとの区間を `height × width` で割り出して
 * いるので、B>1 の `[B,C,H,W]` を渡すと 1 枚目の C 面が 2 枚目まで食い込んだ**沈黙誤値**に
 * なる（長さは合っているので黙って通る）。`denormalizeLatents` と同じ流儀。
 */
export const approximatePreview = (
  latents: Float32Array,
  shape: readonly number[],
): {
  readonly width: number;
  readonly height: number;
  readonly rgba: Uint8ClampedArray<ArrayBuffer>;
} => {
  if (shape.length !== 4) {
    throw new Error(`プレビュー近似は [1,C,H,W] 前提（shape [${shape}] は 4 軸でない）`);
  }
  if (shape[0] !== 1) {
    throw new Error(`プレビュー近似は batch=1 前提（B=${shape[0]} はチャネル区間が崩れる）`);
  }
  if (shape[1] !== LATENT_RGB_FACTORS.length) {
    throw new Error(
      `プレビュー近似: latent のチャネル数 ${shape[1]} が係数表の行数` +
        ` ${LATENT_RGB_FACTORS.length} と違う`,
    );
  }
  const elements = shape.reduce((a, b) => a * b, 1);
  if (latents.length !== elements) {
    throw new Error(`プレビュー近似: 要素数 ${latents.length} が shape [${shape}] と違う`);
  }
  const [, channels, height, width] = shape;
  const plane = height * width;
  // planar `[3,H,W]`・値域 [-1,1] に組んでから最終画像と同じ postprocess へ渡す（clamp と
  // 8bit 量子化と非有限値の門を 1 本に保つ — 2 経路に分けると規約が独立に動く）。
  const image = new Float32Array(plane * 3);
  for (let channel = 0; channel < 3; channel += 1) {
    const bias = LATENT_RGB_BIAS[channel];
    for (let index = 0; index < plane; index += 1) {
      let value = bias;
      for (let source = 0; source < channels; source += 1) {
        value += latents[source * plane + index] * LATENT_RGB_FACTORS[source][channel];
      }
      image[channel * plane + index] = value;
    }
  }
  return { width, height, rgba: imageToRgba(image, width, height) };
};
