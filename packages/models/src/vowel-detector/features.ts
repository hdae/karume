/**
 * Builds the input features of the vowel-detection CRNN (80-dim log-mel + 3 DSP dims = 83) from the
 * waveform. A front stage **outside the graph**; this fills the graph input
 * `features f32 [1, T10, 83]` emitted by `tools/export-recipes/vowel_detector/export.py`.
 *
 * ## Placement (not raised to `src/audio/`)
 *
 * This is the model's **training-time contract** itself (n_fft 512 / win 400 / hop 160 / 80 mel /
 * the Slaney basis shipped as an asset / per-utterance z-normalisation), with no part that can be
 * shared with other families. irodori and sbv2, which **generate** audio, never go through mel, so
 * the only user today is this model (`src/image/preprocess.ts` sits in the shared layer because
 * several families — siglip2, birefnet, depth-anything — go through the same resize → rescale →
 * normalize). Following the rule of not creating abstractions for a single use, it sits under the
 * family.
 *
 * ## The canonical spec is the Python side
 *
 * `training/src/vowel_detector/features.py` + `dsp.py` (upstream `@hdae/vowel-detector`, MIT).
 * Upstream also has a TS implementation of the same port, but **when they disagree Python is
 * right**. The order and dtypes the port must keep are below (`np.fft` runs in f64 even for f32
 * input, so f64 is correct only there):
 *
 * | Step                        | Python                                   | Here            |
 * | --------------------------- | ---------------------------------------- | --------------- |
 * | hann window                 | `np.hanning(401)[:-1]` to f32            | f32 (same)      |
 * | windowing                   | f32 × f32 → f32                          | f32             |
 * | STFT                        | `np.fft.rfft` → **f64**                  | f64 (`fft.ts`)  |
 * | mel multiply-accumulate     | f64 spectrum @ f32 basis → f64           | f64             |
 * | `log(mel + 1e-5)`           | f64 → **rounded to f32**                 | same            |
 * | per-utterance z-norm        | mean/std of an f32 array                 | f64 accumulate  |
 * | autocorrelation             | `np.fft.rfft`/`irfft` → **f64**          | f64             |
 * | RMS, ZCR                    | mean of an f32 array                     | f64 accumulate  |
 *
 * The remaining difference is only that "where numpy reduces pairwise in f32, this accumulates in
 * f64", and the measured max abs diff is **4.77e-7 = 2 ULP from storing the reference in f32** (the
 * derivation is at the top of `tests/vowel_detector_host_test.ts`). Copying f32 pairwise summation
 * would fake a match, so it is not taken (the same judgement as `src/irodori/host/loudness.ts`).
 *
 * ## The mel basis is received from an asset, not baked
 *
 * The `[80, 257]` of `librosa.filters.mel(sr=16000, n_fft=512, n_mels=80, fmin=0, fmax=8000)`
 * (Slaney normalisation) is shipped as one f32 safetensors tensor in the manifest's **`assets`
 * field** (unconditional files independent of the quant choice — ADR 0041 decision 3; kept in
 * `karume/5` too — ADR 0109 decision 4), and the pipeline reads it and passes it to
 * {@link extractFeatures}. The same field and the same convention as sbv2's `style_vectors` /
 * `speaker_embeddings` (`src/sbv2/style.ts`).
 *
 * Rebuilding it from the formula is **not chosen**: upstream too takes the form "export the matrix
 * and share it with JS", and if even one of the definition of the mel scale (Slaney / HTK) or the
 * spelling of the normalisation drifts, the values stay **plausible** while becoming features
 * different from training (80 triangular windows do not look wrong by eye). Carrying that match
 * around as one 82KB asset is the sounder choice.
 */

import { ModelInputError } from "../errors.ts";
import { Fft } from "./fft.ts";

/**
 * Sampling rate of the input waveform. No resampling (a different rate is rejected by the caller).
 */
export const SAMPLE_RATE = 16000;
/**
 * DFT length of the STFT. A power of 2 so radix-2 suffices (a MUST of the Python side's
 * `features.py`).
 */
export const N_FFT = 512;
/**
 * Length of the hann window (25ms). The difference from the DFT length is placed at the frame
 * **centre** (librosa's convention).
 */
export const WIN_LENGTH = 400;
/** Frame advance (10ms). */
export const HOP = 160;
/** Number of mel filters. */
export const N_MELS = 80;
/** Number of columns of the mel basis (`N_FFT / 2 + 1`). */
export const MEL_BINS = N_FFT / 2 + 1;
/** Window of the auxiliary DSP features (32ms). Unlike log-mel, no window function is applied. */
export const DSP_WINDOW = 512;
/** Feature dimension of the graph input (`N_MELS` + voicing + log energy + ZCR). */
export const FEATURE_DIM = N_MELS + 3;

/** Offset that places the window at the frame centre (librosa cuts frames in n_fft units). */
const WIN_OFFSET = (N_FFT - WIN_LENGTH) / 2;
/** Peak search range of the autocorrelation (lag = 16000 / Hz) — from 400Hz down to 60Hz. */
const LAG_MIN = 40;
const LAG_MAX = 267;
/** Offset inside the log (shared by log-mel and log energy). */
const LOG_EPSILON = 1e-5;
/** Offset of the denominator of the per-utterance z-normalisation. */
const STD_EPSILON = 1e-5;
/**
 * Compression of the log energy (divide the ratio to the utterance peak by 10 to align the range).
 */
const LOG_ENERGY_SCALE = 10;

/** The result of {@link extractFeatures}. */
type VowelFeatures = {
  /** Row-major `[frames, FEATURE_DIM]`. Can be passed as is to the graph input `features`. */
  readonly data: Float32Array<ArrayBuffer>;
  /** Frame count on the 10ms grid (the graph's output is half of this with stride 2). */
  readonly frames: number;
};

/**
 * Periodic hann window (equivalent to `np.hanning(length + 1)[:-1]`). Values are rounded to f32.
 */
const hannWindow = (length: number): Float32Array<ArrayBuffer> => {
  const window = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    window[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / length);
  }
  return window;
};

/**
 * The same test as `np.signbit` (**−0 is negative**).
 *
 * MUST: do not settle for `value < 0`. The Python side counts ZCR by comparing neighbours with
 * `np.signbit`, so only when a waveform containing −0 arrives would the crossing count drift and
 * the 82nd feature quietly change. The int16 path of `decodeWav` never produces −0, but IEEE float
 * WAVs and waveforms processed on the host can carry −0 (using the same test as the canonical spec
 * spares rethinking the behaviour there).
 */
const signbit = (value: number): boolean => value < 0 || (value === 0 && 1 / value < 0);

/**
 * Computes the 10ms frame count from the waveform length (sample count) (the `frames` of
 * {@link extractFeatures}).
 *
 * The formula is `floor((samples − N_FFT) / HOP) + 1` (the same counting as librosa's
 * `center=False`). For lengths below `N_FFT` it returns less than 1 (acceptance is decided by the
 * caller — the doc below).
 *
 * MUST: {@link extractFeatures} also counts with this one function. The frame count depends only on
 * the length, so the pipeline checks the operating limit with it **before** building the features
 * (rejecting after building them would run the FFT and the mel transform over every frame of an
 * over-long audio and occupy the thread synchronously for seconds). If the formula were split into
 * 2 places, the earlier decision and the later actual length would disagree.
 */
export const featureFrameCount = (samples: number): number =>
  Math.floor((samples - N_FFT) / HOP) + 1;

/**
 * 16kHz mono waveform → features `[frames, FEATURE_DIM]`.
 *
 * `melBasis` is row-major `[N_MELS, MEL_BINS]` (the asset's f32 safetensors as is).
 *
 * The frame count is `floor((samples − N_FFT) / HOP) + 1` (the same counting as librosa's
 * `center=False`). Under this formula the 512-sample DSP window also **always fits inside the
 * waveform**, so the zero pad of the Python side's `frame_signal` never fires (reads at the
 * boundary do not go outside the indices).
 *
 * A length that yields not even one frame fails loudly (Python returns an empty array and the later
 * `mean()` spreads NaN — failing beats silently building NaN features).
 */
export const extractFeatures = (
  audio: Float32Array<ArrayBufferLike>,
  melBasis: Float32Array<ArrayBuffer>,
): VowelFeatures => {
  if (melBasis.length !== N_MELS * MEL_BINS) {
    throw new Error(
      `extractFeatures: mel 基底が ${melBasis.length} 要素（${N_MELS}×${MEL_BINS} = ${
        N_MELS * MEL_BINS
      } が要る）`,
    );
  }
  const frames = featureFrameCount(audio.length);
  if (frames < 1) {
    // The waveform is the very argument of `detect` the caller passed, and the one remedy is "pass
    // a longer waveform" (ADR 0107 decision 2). The mel basis just above is a number that comes
    // from the asset, so it stays a plain `Error`.
    throw new ModelInputError(
      `extractFeatures: 波形が ${audio.length} サンプルしかない` +
        `（${N_FFT} サンプル = ${(N_FFT / SAMPLE_RATE) * 1000}ms 以上が要る）`,
    );
  }
  for (let index = 0; index < audio.length; index += 1) {
    const value = audio[index];
    // MUST: fail here on non-finite samples — one NaN makes the per-utterance z-normalisation turn
    // every feature into NaN, and after one GPU run it would fail with a message that looks like an
    // asset fault, "the logits are non-finite". The waveform is the caller's argument itself, so
    // it is input-caused (ADR 0107 decision 2).
    if (!Number.isFinite(value)) {
      throw new ModelInputError(
        `extractFeatures: 波形の ${index} 番目のサンプル ${value} が非有限`,
      );
    }
  }

  const window = hannWindow(WIN_LENGTH);
  const melFft = new Fft(N_FFT);
  const dspFft = new Fft(2 * DSP_WINDOW);
  const windowed = new Float32Array(WIN_LENGTH);
  const centered = new Float32Array(DSP_WINDOW);
  const power = new Float64Array(MEL_BINS);
  const autocorrelation = new Float64Array(LAG_MAX + 1);

  const logMel = new Float32Array(frames * N_MELS);
  const voiced = new Float32Array(frames);
  const logEnergy = new Float32Array(frames);
  const zeroCrossingRate = new Float32Array(frames);

  for (let frame = 0; frame < frames; frame += 1) {
    const offset = frame * HOP;
    // --- log-mel (the window applies to the WIN_LENGTH samples at the frame centre) ---
    for (let index = 0; index < WIN_LENGTH; index += 1) {
      windowed[index] = audio[offset + WIN_OFFSET + index] * window[index];
    }
    melFft.powerSpectrum(windowed, power);
    for (let mel = 0; mel < N_MELS; mel += 1) {
      const row = mel * MEL_BINS;
      let sum = 0;
      for (let bin = 0; bin < MEL_BINS; bin += 1) sum += power[bin] * melBasis[row + bin];
      logMel[frame * N_MELS + mel] = Math.log(sum + LOG_EPSILON);
    }

    // --- auxiliary DSP (no window function, only the mean is removed) ---
    let mean = 0;
    for (let index = 0; index < DSP_WINDOW; index += 1) mean += audio[offset + index];
    mean /= DSP_WINDOW;
    let squared = 0;
    let crossings = 0;
    for (let index = 0; index < DSP_WINDOW; index += 1) {
      const sample = audio[offset + index];
      const value = sample - mean;
      centered[index] = value;
      squared += value * value;
      // MUST: count crossings on the **raw waveform** (Python too looks at frames before the mean
      // is removed).
      if (index > 0 && signbit(sample) !== signbit(audio[offset + index - 1])) crossings += 1;
    }
    dspFft.autocorrelation(centered, autocorrelation);
    let peak = Number.NEGATIVE_INFINITY;
    for (let lag = LAG_MIN; lag <= LAG_MAX; lag += 1) {
      peak = Math.max(peak, autocorrelation[lag]);
    }
    voiced[frame] = autocorrelation[0] > 1e-10 ? peak / autocorrelation[0] : 0;
    logEnergy[frame] = Math.log(Math.sqrt(squared / DSP_WINDOW) + LOG_EPSILON);
    zeroCrossingRate[frame] = crossings / (DSP_WINDOW - 1);
  }

  // log-mel is z-normalised by **one scalar for the whole utterance** (neither per column nor per
  // row).
  let sum = 0;
  for (const value of logMel) sum += value;
  const mean = sum / logMel.length;
  let variance = 0;
  for (const value of logMel) variance += (value - mean) * (value - mean);
  const deviation = Math.sqrt(variance / logMel.length) + STD_EPSILON;

  // The log energy is **the ratio to the utterance peak** (0 or below).
  let peakLogEnergy = Number.NEGATIVE_INFINITY;
  for (const value of logEnergy) peakLogEnergy = Math.max(peakLogEnergy, value);

  const data = new Float32Array(frames * FEATURE_DIM);
  for (let frame = 0; frame < frames; frame += 1) {
    const row = frame * FEATURE_DIM;
    for (let mel = 0; mel < N_MELS; mel += 1) {
      data[row + mel] = (logMel[frame * N_MELS + mel] - mean) / deviation;
    }
    data[row + N_MELS] = voiced[frame];
    data[row + N_MELS + 1] = (logEnergy[frame] - peakLogEnergy) / LOG_ENERGY_SCALE;
    data[row + N_MELS + 2] = zeroCrossingRate[frame];
  }
  return { data, frames };
};
