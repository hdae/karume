/**
 * Integrated loudness (LUFS) of ITU-R BS.1770-4. Used only for the LUFS normalization of the
 * reference audio.
 *
 * Upstream goes through `AudioSignal.loudness()` of `descript-audiotools` (= `pyloudnorm`'s Meter
 * rewritten in torch). This is a port of that **implementation's behaviour**, not the general form
 * of the standard — where the standard leaves room for choice (the value when nothing survives
 * gating, the handling of less than 0.5 s, how blocks are counted), it follows upstream's spelling.
 *
 * ## Steps
 *
 * 1. Less than 0.5 s is zero-padded before measuring (`signal_duration < 0.5` of
 *    `AudioSignal.loudness`). The padding is only for the measurement; the original waveform is
 *    what is returned (the caller has nothing to cut)
 * 2. K-weighting = a 2-stage biquad, high_shelf → high_pass ({@link kWeightingFilters})
 * 3. Lay out 0.4 s blocks overlapping by 75% (stride 0.1 s), and take the mean square z per block.
 *    An incomplete trailing block is zero-padded, not discarded (`F = 1 + ceil((T − 19200)/4800)`)
 * 4. Select l = −0.691 + 10·log10(z) with the absolute gate −70 LKFS, derive the relative gate
 *    (−10 LU) from the mean of the remaining z, and compute the integrated loudness from the z mean
 *    of the blocks that **passed both**
 * 5. Clamp from below at −70 LKFS
 *
 * ## MUST: run in f64 (upstream runs `lfilter` in f32)
 *
 * Upstream passes both the coefficients and the data through `torchaudio.functional.lfilter` in
 * f32. Copying that step by step with `Math.fround` is an option, but **it is not taken** — the way
 * rounding enters an IIR's sequential difference equation depends on the library's internal
 * evaluation order (torchaudio normalizes the coefficients before running the difference
 * equation), so a copy would not give bit identity anyway. The difference that remains is of the
 * 1e-5 LU class in refDb (the measurement table of `irodori_reference_test.ts`), relative 1e-6 as a
 * normalization gain — buried in the waveform tolerance. Measuring the difference and putting it
 * into the check's threshold is sounder than staging a fake match by transcription.
 */

/** One biquad stage (the denominator is normalized to `a[0] = 1`). */
type Biquad = {
  readonly b: readonly [number, number, number];
  readonly a: readonly [number, number, number];
};

/** Measurement block length (seconds). */
const BLOCK_SECONDS = 0.4;
/** Block overlap (75% → a stride of 0.1 s). */
const BLOCK_OVERLAP = 0.75;
/** Minimum measurement length (seconds). Anything shorter is zero-padded before measuring. */
const MIN_MEASURE_SECONDS = 0.5;
/** The offset term of BS.1770 (`l = OFFSET + 10·log10(z)`). */
const LOUDNESS_OFFSET = -0.691;
/** Absolute gate (LKFS). Also the lower bound of the return value. */
const ABSOLUTE_GATE_DB = -70;
/** Relative gate (LU — how far below the mean of the blocks that passed the absolute gate). */
const RELATIVE_GATE_LU = -10;

/** Design values of K-weighting stage 1 (high shelf) (RBJ cookbook parameters). */
const SHELF_GAIN_DB = 4.0;
const SHELF_Q = Math.SQRT1_2;
const SHELF_FC_HZ = 1500;
/** Design values of K-weighting stage 2 (high pass). No gain (G = 0dB). */
const HIGHPASS_Q = 0.5;
const HIGHPASS_FC_HZ = 38;

/**
 * High shelf of the RBJ cookbook.
 *
 * MUST: do not bake the coefficients as numbers (build them from the formula). The pipeline does
 * not accept reference audio other than 48kHz, but hard-coded coefficients would let the error
 * "measuring another rate with 48kHz coefficients" through **silently**.
 * The values at 48kHz themselves are pinned by `irodori_reference_test.ts` (catches a mistake in
 * copying the formula).
 */
const highShelf = (sampleRate: number, gainDb: number, q: number, fc: number): Biquad => {
  const amplitude = Math.pow(10, gainDb / 40);
  const w0 = 2 * Math.PI * (fc / sampleRate);
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const root = Math.sqrt(amplitude);
  const b0 = amplitude * (amplitude + 1 + (amplitude - 1) * cos + 2 * root * alpha);
  const b1 = -2 * amplitude * (amplitude - 1 + (amplitude + 1) * cos);
  const b2 = amplitude * (amplitude + 1 + (amplitude - 1) * cos - 2 * root * alpha);
  const a0 = amplitude + 1 - (amplitude - 1) * cos + 2 * root * alpha;
  const a1 = 2 * (amplitude - 1 - (amplitude + 1) * cos);
  const a2 = amplitude + 1 - (amplitude - 1) * cos - 2 * root * alpha;
  return { b: [b0 / a0, b1 / a0, b2 / a0], a: [1, a1 / a0, a2 / a0] };
};

/** High pass of the RBJ cookbook (2nd order). */
const highPass = (sampleRate: number, q: number, fc: number): Biquad => {
  const w0 = 2 * Math.PI * (fc / sampleRate);
  const alpha = Math.sin(w0) / (2 * q);
  const cos = Math.cos(w0);
  const b0 = (1 + cos) / 2;
  const a0 = 1 + alpha;
  return {
    b: [b0 / a0, -(1 + cos) / a0, b0 / a0],
    a: [1, (-2 * cos) / a0, (1 - alpha) / a0],
  };
};

/** The 2 stages of K-weighting (applied in the order high shelf → high pass). */
export const kWeightingFilters = (sampleRate: number): readonly [Biquad, Biquad] => [
  highShelf(sampleRate, SHELF_GAIN_DB, SHELF_Q, SHELF_FC_HZ),
  highPass(sampleRate, HIGHPASS_Q, HIGHPASS_FC_HZ),
];

/**
 * Applies a biquad by its sequential difference equation (the counterpart of `lfilter` — zero
 * initial state).
 *
 * MUST: do not substitute the frequency domain or an FIR approximation. Upstream is an exact IIR,
 * and an approximation shifts the LUFS by about 0.1 LU (the normalization gain moves by 1% = the
 * waveform differs in every sample).
 */
const applyBiquad = (
  samples: Float64Array<ArrayBuffer>,
  filter: Biquad,
): Float64Array<ArrayBuffer> => {
  const [b0, b1, b2] = filter.b;
  const [, a1, a2] = filter.a;
  const out = new Float64Array(samples.length) as Float64Array<ArrayBuffer>;
  let x1 = 0;
  let x2 = 0;
  let y1 = 0;
  let y2 = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const x0 = samples[i];
    const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    out[i] = y0;
    x2 = x1;
    x1 = x0;
    y2 = y1;
    y1 = y0;
  }
  return out;
};

/**
 * Returns the integrated loudness (LUFS / LKFS). Mono only (channel weight G = 1).
 *
 * When not a single block passes the gates, it returns −70 (upstream reaches the same value through
 * `nan_to_num` → `log10(0)` → `-inf` → `maximum(-70)`).
 */
export const integratedLoudness = (
  samples: Float32Array<ArrayBuffer>,
  sampleRate: number,
): number => {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new RangeError(`integratedLoudness: sample rate ${sampleRate} is not a positive integer`);
  }
  if (samples.length === 0) throw new RangeError("integratedLoudness: the waveform is empty");
  // Below 0.5 s, zeros are appended **only for the measurement** (upstream
  // `AudioSignal.loudness`).
  const minimum = Math.round(MIN_MEASURE_SECONDS * sampleRate);
  const measured = new Float64Array(Math.max(samples.length, minimum)) as Float64Array<ArrayBuffer>;
  measured.set(samples);

  let filtered = measured;
  for (const filter of kWeightingFilters(sampleRate)) filtered = applyBiquad(filtered, filter);

  // MUST: blocks are counted as in `julius.core.unfold` (which audiotools uses instead of
  // `torch.nn.Unfold`) — **an incomplete trailing block is not discarded but zero-padded and
  // counted**. Misreading it as torch's Unfold shifts the LUFS by 0.07 LU only for reference audio
  // whose length is not a multiple of the stride (measured — the golden's ref-odd is the only case
  // that tells 27 blocks from 28).
  // The padding is added **after the filters** (it is inside `_unfold`, so the IIR tail does not
  // extend into it).
  const kernel = Math.trunc(BLOCK_SECONDS * sampleRate);
  const stride = Math.trunc(BLOCK_SECONDS * sampleRate * (1 - BLOCK_OVERLAP));
  // MUST: derived by floor, so at a tiny rate the window itself vanishes — stride 0 makes the
  // block count Infinity (an opaque allocation error), kernel 0 makes `squared / kernel` NaN. Both
  // mean "this rate cannot be measured", so the minimum of the derived values fails explicitly.
  if (kernel < 1 || stride < 1) {
    throw new RangeError(
      `integratedLoudness: the loudness window cannot be derived at sample rate ${sampleRate}` +
        ` (${BLOCK_SECONDS} s = ${kernel} samples / stride ${stride} samples — each must be >= 1)`,
    );
  }
  const blocks = Math.ceil((Math.max(filtered.length, kernel) - kernel) / stride) + 1;
  const power = new Float64Array(blocks);
  for (let block = 0; block < blocks; block += 1) {
    const from = block * stride;
    const end = Math.min(from + kernel, filtered.length);
    let squared = 0;
    for (let i = from; i < end; i += 1) squared += filtered[i] * filtered[i];
    power[block] = squared / kernel;
  }

  // Absolute gate → relative gate from its mean → mean of the blocks that passed both.
  const level = (mean: number): number => LOUDNESS_OFFSET + 10 * Math.log10(mean);
  const gatedMean = (threshold: number, relative: number): number => {
    let sum = 0;
    let count = 0;
    for (const value of power) {
      const decibel = level(value);
      if (decibel > threshold && decibel > relative) {
        sum += value;
        count += 1;
      }
    }
    return count === 0 ? 0 : sum / count;
  };
  const absoluteMean = gatedMean(ABSOLUTE_GATE_DB, Number.NEGATIVE_INFINITY);
  const relativeGate = level(absoluteMean) + RELATIVE_GATE_LU;
  const loudness = level(gatedMean(ABSOLUTE_GATE_DB, relativeGate));
  return Math.max(loudness, ABSOLUTE_GATE_DB);
};
