/**
 * Tail trim (upstream's `inference_runtime.find_flattening_point` + the cut in `_synthesize`).
 *
 * ## MUST: decide **on z (the latent)**, cut the **waveform**
 *
 * Upstream searches the latent for "the position where the latent's tail becomes flat and near 0",
 * and cuts **the waveform** at that position times `hopLength` in samples. Cutting the latent before
 * decode changes the boundary padding, so the result is not bit-identical to the leading part of a
 * full-length decode (measured in the recon). Decode therefore always runs at full length, and the
 * cut happens once, at the very end.
 *
 * ## The thresholds are the defaults of upstream's `SamplingRequest` (not held as runtime knobs)
 *
 * `tail_window_size` / `tail_std_threshold` / `tail_mean_threshold` are request-side fields
 * upstream, but this implementation only ever uses the defaults, so they are constants (not in
 * `pipelineConfig` = a distribution has no way to declare another value). The golden comparison is
 * the 3 cases of `goldens/dacvae-32dim/host/trim.safetensors` of the test assets.
 */

/** The window looked at the tail (latent frames). */
const WINDOW = 20;
/** Threshold of the population std of the whole window (`WINDOW × latentDim` scalars). */
const STD_THRESHOLD = 0.05;
/** Allowed distance of the whole window's mean from the target value 0. */
const MEAN_THRESHOLD = 0.1;

/**
 * Returns the latent frame position where the tail starts to flatten (`frames` if none is found).
 *
 * As upstream does, it scans from the start **after appending `WINDOW` rows of zero padding to the
 * tail** — thanks to the padding a window always holds `WINDOW` rows of scalars, and the closer a
 * window is to the tail, the more it is diluted with zeros. The scan goes up to
 * `i = frames − 1`; `i = frames` (all padding) is not examined (it would hold unconditionally).
 *
 * MUST: the population standard deviation (torch's `std(unbiased=False)`) — the unbiased one
 * inflates the value more the smaller the window, and the 0.05 threshold test would shift by
 * frames.
 */
export const findFlatteningPoint = (
  latent: Float32Array<ArrayBuffer>,
  frames: number,
  latentDim: number,
): number => {
  if (!Number.isInteger(frames) || frames < 0) {
    throw new Error(`findFlatteningPoint: frames ${frames} is not a non-negative integer`);
  }
  if (!Number.isInteger(latentDim) || latentDim <= 0) {
    throw new Error(`findFlatteningPoint: latentDim ${latentDim} is not a positive integer`);
  }
  if (latent.length !== frames * latentDim) {
    throw new Error(
      `findFlatteningPoint: latent element count ${latent.length} differs from ${frames}×${latentDim}`,
    );
  }
  const count = WINDOW * latentDim;
  for (let start = 0; start < frames; start += 1) {
    // Zero-padding rows contribute to neither the sum nor the sum of squares, so only the real rows
    // need to be walked.
    const end = Math.min(frames, start + WINDOW) * latentDim;
    const from = start * latentDim;
    let sum = 0;
    for (let index = from; index < end; index += 1) sum += latent[index];
    const mean = sum / count;
    if (Math.abs(mean) >= MEAN_THRESHOLD) continue;
    // Two passes (the mean first, then the sum of squared deviations). `E[x²] − E[x]²` can cancel
    // catastrophically into a negative variance in this use, where the std is close to 0 — exactly
    // the region the test is about, so it is not used.
    let squared = 0;
    for (let index = from; index < end; index += 1) {
      const deviation = latent[index] - mean;
      squared += deviation * deviation;
    }
    // Padding rows have the deviation `0 − mean` (upstream also takes the std including the
    // padding).
    const padded = (count - (end - from)) * mean * mean;
    if (Math.sqrt((squared + padded) / count) < STD_THRESHOLD) return start;
  }
  return frames;
};

/**
 * Decides the length (samples) the waveform is cut to (`max_samples` of upstream's `_synthesize`).
 *
 * Only when `flatteningPoint` is 0 is nothing cut — upstream conditions on
 * `flattening_samples > 0`, so that when "flat from the start" = all silent is detected, it does not
 * return 0 samples of audio.
 */
export const trimmedSampleCount = (
  targetSamples: number,
  flatteningPoint: number,
  hopLength: number,
): number => {
  const flatteningSamples = flatteningPoint * hopLength;
  return flatteningSamples > 0 ? Math.min(targetSamples, flatteningSamples) : targetSamples;
};
