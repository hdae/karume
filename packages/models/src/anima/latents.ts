/**
 * Shaping of latents and condition tensors (outside the IR).
 */

import type { Tensor } from "@karume/runtime";

const f32 = Math.fround;

/**
 * Per-channel latent mean / standard deviation of the VAE
 * (`AutoencoderKLQwenImage.config.latents_mean` / `latents_std`).
 *
 * **Why they are constants on the pipeline side**: these two exist only in the VAE config and are
 * in neither the IR nor the distributed assets (the VAE decoder graph takes an **already
 * denormalised** latent). They are not put in the manifest's `pipelineConfig` either — architecture
 * constants belong to the pipeline implementation, the "keep one derivation source" rule of
 * ADR 0038 §2.
 *
 * MUST: do not place hand-copied numbers without verification. A test pins bit equality with the
 * reference fixture's `latents_mean` / `latents_std`. Using a model that replaces the values fails
 * that one test.
 *
 * MUST: do not hold a TypedArray at module scope (do not run `Float32Array.from` at import) —
 * cross-cutting invariant "zero side effects in every module (no execution at import, no global
 * mutable state)". This stays a `readonly number[]` array literal, and {@link animaLatents} builds
 * the `Float32Array` per call (adding only the cost of creating 16 elements × 2). Consumers
 * (denormalisation in `pipeline.ts`, barrel users) go through that accessor only.
 *
 * Each value is the shortest decimal form of the VAE config's 4-digit decimal (e.g. `-0.7571`)
 * rounded to f32, the same notation as the reference fixture (from torch's f32 tensor). Kept as f64
 * decimals, this array itself would disagree with the f32 actually used in denormalisation.
 */
export const ANIMA_LATENTS_MEAN: readonly number[] = [
  -0.757099986076355, -0.708899974822998, -0.911300003528595, 0.10750000178813934,
  -0.1745000034570694, 0.9653000235557556, -0.1517000049352646, 1.5507999658584595,
  0.41339999437332153, -0.07150000333786011, 0.5516999959945679, -0.36320000886917114,
  -0.19220000505447388, -0.9496999979019165, 0.25029999017715454, -0.2921000123023987,
];

/** Standard deviation paired with {@link ANIMA_LATENTS_MEAN}. */
export const ANIMA_LATENTS_STD: readonly number[] = [
  2.8183999061584473, 1.4541000127792358, 2.327500104904175, 2.6558001041412354, 1.219599962234497,
  1.770799994468689, 2.6052000522613525, 2.0743000507354736, 3.268699884414673, 2.152600049972534,
  2.8652000427246094, 1.5578999519348145, 1.638200044631958, 1.1253000497817993, 2.8250999450683594,
  1.9160000085830688,
];

/**
 * Returns the denormalisation constant pair **as an independent copy per call** (this is what the
 * public surface exposes).
 *
 * MUST: build a new `Float32Array` per call (do not reuse). If a consumer modifies the return
 * value, it reaches neither `generate`'s denormalisation nor the return value of the next call.
 *
 * NOTE: "expose it frozen" does not work — `Object.freeze` throws TypeError on a TypedArray with
 * elements. An accessor that returns a copy is the only form.
 */
export const animaLatents = (): {
  mean: Float32Array<ArrayBuffer>;
  std: Float32Array<ArrayBuffer>;
} => ({
  mean: Float32Array.from(ANIMA_LATENTS_MEAN),
  std: Float32Array.from(ANIMA_LATENTS_STD),
});

/**
 * Per-channel denormalisation of a latent.
 * MUST: do not rewrite as `latents · std` — the reference implementation **builds the reciprocal of
 * std and divides**, so switching to multiplication changes the last digit (bit equality breaks).
 * MUST: check the **B=1 assumption** at the entry. The per-channel range is derived as
 * `length / channels`, so passing a B>1 `[B,C,H,W]` stretches one range over B images, which then
 * get different mean/std each — **a silent wrong value** (the length divides evenly, so it passes
 * silently).
 */
export const denormalizeLatents = (
  latents: Float32Array<ArrayBufferLike>,
  shape: readonly number[],
  mean: Float32Array<ArrayBufferLike>,
  std: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBuffer> => {
  if (shape[0] !== 1) {
    throw new Error(`逆正規化は batch=1 前提（B=${shape[0]} はチャネル区間が崩れる）`);
  }
  if (shape[1] !== mean.length || shape[1] !== std.length) {
    throw new Error(
      `逆正規化: latent のチャネル数 ${shape[1]} が mean ${mean.length} / std ${std.length} と違う`,
    );
  }
  const elements = shape.reduce((a, b) => a * b, 1);
  if (latents.length !== elements) {
    throw new Error(`逆正規化: 要素数 ${latents.length} が shape [${shape}] と違う`);
  }
  const perChannel = latents.length / mean.length;
  const out = new Float32Array(latents.length);
  for (let channel = 0; channel < mean.length; channel += 1) {
    const inverseStd = f32(1 / std[channel]);
    for (let index = 0; index < perChannel; index += 1) {
      const at = channel * perChannel + index;
      out[at] = f32(f32(latents[at] / inverseStd) + mean[channel]);
    }
  }
  return out;
};

/**
 * Zero-pads the conditioner output up to `min_sequence_length` rows (outside the IR).
 *
 * MUST: check the **B=1 assumption** and the row limit at the entry. `[B,T,W]` is copied flat, so
 * with B>1 the second and later images flow into the first one's padding, and it passes silently as
 * long as there are enough rows. Too many rows (T > rows) makes `set` throw RangeError, but that
 * diagnosis says nothing about "which stage it is about", so this fails with a name here.
 * MUST: check the dtype too. Passing an i32 / bool tensor makes `TypedArray.prototype.set`
 * **silently convert the numbers**, and only the values change while shape and length still match
 * (a silent wrong value).
 */
export const padSequence = (hidden: Tensor, rows: number): Float32Array<ArrayBuffer> => {
  if (hidden.dtype !== "f32") {
    throw new Error(`512 パディングは f32 前提（dtype ${hidden.dtype} は黙って数値変換される）`);
  }
  if (hidden.shape[0] !== 1) {
    throw new Error(`512 パディングは batch=1 前提（B=${hidden.shape[0]} は余白へ流れ込む）`);
  }
  if (hidden.shape[1] > rows) {
    throw new Error(`512 パディング: 行数 ${hidden.shape[1]} が上限 ${rows} を超えている`);
  }
  const width = hidden.shape[2];
  const padded = new Float32Array(rows * width);
  padded.set(hidden.data);
  return padded;
};
