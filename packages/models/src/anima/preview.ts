/**
 * Linear approximation of an intermediate latent → RGB preview (without the VAE).
 *
 * Turns the latent returned by `copyLatents()` of the `denoise-step` event into colour with a
 * linear combination over channels only. The VAE can be loaded only **after** the DiT is released
 * (a VRAM MUST — the module doc of `pipeline.ts`), so this approximation is the only way to show
 * per-step progress as a picture.
 *
 * MUST: this is **not an exact decode**. It is a linear projection that collapses 16 channels into
 * 3, so detail, texture and text are not reproduced (view it as a thumbnail at the collapsed latent
 * resolution, 1/8). Do not judge the correctness of the generated result from this output — the
 * canonical picture is the return value of `generate`.
 */

import { imageToRgba } from "./image.ts";

/**
 * Linear approximation coefficients of 16ch latent → RGB (row = latent channel, column = R,G,B) and
 * the constant term.
 *
 * Source: the `"qwen"` family of
 * https://huggingface.co/spaces/KAD001/Wan2GP/blob/main/shared/RGB_factors.py (ComfyUI-style
 * previewers use the same table).
 *
 * MUST: do not `export`. This is an internal constant that only decides how the preview looks;
 * exposing it would create a "consumer applies it itself" path where the calibration-space mix-up
 * (below) could happen again.
 *
 * ## Measurement of the calibration space (2026-08-24)
 *
 * No primary record of **which latent space** this table was calibrated in was found, so it was
 * decided by measurement. Rig = the local turbo distribution (`models/karume-anima` turbo / default
 * quant), 512×512, seed 0, the manifest's default steps (8): one image was generated, the last
 * step's latent was captured, and two variants — (A) applied directly to the normalised latent /
 * (B) applied directly to the denormalised latent passed through `denormalizeLatents` — were
 * compared with a reference made by 8×8 box-averaging the generated image down to latent
 * resolution.
 *
 * - Pearson correlation (linear output before clamping): A = R 0.9794 / G 0.9742 / B 0.9759 /
 *   luminance 0.9771 (mean 0.9767), B = R 0.9798 / G 0.9743 / B 0.9730 / luminance 0.9771 (mean
 *   0.9761). **Correlation alone does not decide** (difference 0.0006) — denormalisation is a
 *   per-channel affine transform of the latent, so after projection it is only "a projection with
 *   re-weighted coefficients", to which correlation is nearly invariant.
 * - The deciding factor is **how well the absolute values fit**: RMSE against the reference is A
 *   0.0797 / B 0.3614, and the share of pixels that leave the range `[0,1]` and are crushed by the
 *   clamp is A 1.2% / B 51.7%. B blows out to white and does not work as a picture. So **the
 *   calibration space is the normalised latent** (A), which also matches ComfyUI practice.
 * - The same measurement also fixed **the range convention of the output**: the linear output lies
 *   not in `[0,1]` but in **`[-1,1]`, the same as the VAE sample** (treated as `[0,1]`, RMSE 0.5454
 *   / clamp 50.2% — half is crushed). So it goes through exactly the same postprocess as the final
 *   image ({@link imageToRgba}).
 */
const LATENT_RGB_FACTORS: readonly (readonly [number, number, number])[] = [
  [-0.1299, -0.1692, 0.2932],
  [0.0671, 0.0406, 0.0442],
  [0.3568, 0.2548, 0.1747],
  [0.0372, 0.2344, 0.142],
  [0.0313, 0.0189, -0.0328],
  [0.0296, -0.0956, -0.0665],
  [-0.3477, -0.4059, -0.2925],
  [0.0166, 0.1902, 0.1975],
  [-0.0412, 0.0267, -0.1364],
  [-0.1293, 0.074, 0.1636],
  [0.068, 0.3019, 0.1128],
  [0.0032, 0.0581, 0.0639],
  [-0.1251, 0.0927, 0.1699],
  [0.006, -0.0633, 0.0005],
  [0.3477, 0.2275, 0.295],
  [0.1984, 0.0913, 0.1861],
];

/** Constant term paired with {@link LATENT_RGB_FACTORS} (`latent_rgb_factors_bias`). */
const LATENT_RGB_BIAS: readonly [number, number, number] = [-0.1835, -0.0868, -0.336];

/**
 * Approximates one intermediate latent as an RGBA preview. The returned image is at **latent
 * resolution** (1/8 of the original image).
 *
 * MUST: pass the return value of `denoise-step`'s `copyLatents()` **as is** (= the normalised
 * latent); do not pass it through `denormalizeLatents`. The coefficients are calibrated in the
 * normalised space (the measurement record in this module), so passing denormalised values gives a
 * blown-out different picture.
 *
 * MUST: check the **B=1 assumption** at the entry. The per-channel range is derived from
 * `height × width`, so passing a B>1 `[B,C,H,W]` makes the first image's C planes bite into the
 * second — **a silent wrong value** (the length matches, so it passes silently). The same style as
 * `denormalizeLatents`.
 */
export const approximatePreview = (
  latents: Float32Array<ArrayBufferLike>,
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
  // Build planar `[3,H,W]` in range [-1,1], then pass it to the same postprocess as the final image
  // (keeps the clamp, 8-bit quantization and the non-finite check in one place — two paths would
  // let the conventions move independently).
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
