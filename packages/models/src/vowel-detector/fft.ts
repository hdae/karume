/**
 * Iterative radix-2 FFT. It outputs only the 2 quantities the vowel detector's feature extraction
 * needs — the **power spectrum** and the **autocorrelation** of a real signal.
 *
 * ## Why under the family and not in `src/audio/`
 *
 * It is not a general FFT (real signals, powers of 2, only the spectrum and the autocorrelation,
 * resident scratch). The only user is {@link ./features.ts}, and there is no ground for raising it
 * to the shared layer (`src/audio/wav.ts` sits in the shared layer because "audio models in
 * general read and write WAV"). It can be raised when a second user appears.
 *
 * ## MUST: run the transform in f64 (the upstream TS implementation uses f32)
 *
 * The canonical spec is the Python side (`features.py` / `dsp.py`), which goes through `np.fft` =
 * **always f64** (numpy's FFT promotes even f32 input to double precision). The upstream TS
 * implementation runs its butterflies in Float32Array and is that much further from the canonical
 * spec; its max abs diff against the Python reference is **5.42e-6** (measured). Making this f64
 * aligns the dtype flow with Python step by step and shrinks it to **4.77e-7 = 2 ULP from storing
 * the reference in f32** (= numerically at the floor. The derivation is at the top of
 * `tests/vowel_detector_host_test.ts`).
 *
 * The choice of **computing at the same precision as the canonical spec and measuring the
 * difference** rather than copying its rounding to fake a match is the same as in
 * `src/irodori/host/loudness.ts` (there, conversely, upstream is an f32 IIR, and copying it would
 * not give a bit match either, so f64 was chosen).
 */

/** Bit-reversal table of size `size` (for the DIT input reordering). */
const bitReverseTable = (size: number): Uint32Array<ArrayBuffer> => {
  const bits = Math.log2(size);
  const table = new Uint32Array(size);
  for (let index = 0; index < size; index += 1) {
    let reversed = 0;
    for (let bit = 0; bit < bits; bit += 1) {
      reversed |= ((index >> bit) & 1) << (bits - 1 - bit);
    }
    table[index] = reversed;
  }
  return table;
};

/**
 * FFT of transform length `size` (a power of 2). It carries the twiddle factors and the scratch,
 * so it is not rebuilt per frame.
 *
 * The only state is the scratch (there is no point carrying it over between calls).
 * `powerSpectrum` / `autocorrelation` clobber each other's scratch, so each call takes its result
 * fully into `out`.
 */
export class Fft {
  readonly size: number;
  readonly #reversed: Uint32Array<ArrayBuffer>;
  readonly #cos: Float64Array<ArrayBuffer>;
  readonly #sin: Float64Array<ArrayBuffer>;
  readonly #re: Float64Array<ArrayBuffer>;
  readonly #im: Float64Array<ArrayBuffer>;
  readonly #scratch: Float64Array<ArrayBuffer>;

  constructor(size: number) {
    if (!Number.isInteger(size) || size < 2 || (size & (size - 1)) !== 0) {
      throw new RangeError(`Fft: transform length ${size} is not a power of 2 >= 2`);
    }
    this.size = size;
    this.#reversed = bitReverseTable(size);
    this.#cos = new Float64Array(size / 2);
    this.#sin = new Float64Array(size / 2);
    for (let index = 0; index < size / 2; index += 1) {
      this.#cos[index] = Math.cos((-2 * Math.PI * index) / size);
      this.#sin[index] = Math.sin((-2 * Math.PI * index) / size);
    }
    this.#re = new Float64Array(size);
    this.#im = new Float64Array(size);
    this.#scratch = new Float64Array(size);
  }

  /**
   * Writes the power spectrum `|X[k]|²` of a real signal (length ≤ `size`, the shortfall
   * zero-padded at the **end**) to `out[0..size/2]` (the same order as
   * `np.abs(np.fft.rfft(x, n=size))**2`).
   */
  powerSpectrum(input: Float32Array<ArrayBuffer>, out: Float64Array<ArrayBuffer>): void {
    const bins = this.size / 2 + 1;
    if (input.length > this.size) {
      throw new RangeError(
        `Fft: input of ${input.length} points exceeds the transform length ${this.size}`,
      );
    }
    if (out.length < bins) {
      throw new RangeError(
        `Fft: output of ${out.length} points is fewer than the ${bins} rfft bins`,
      );
    }
    this.#load(input);
    this.#butterflies();
    for (let bin = 0; bin < bins; bin += 1) {
      out[bin] = this.#re[bin] * this.#re[bin] + this.#im[bin] * this.#im[bin];
    }
  }

  /**
   * Writes the linear autocorrelation of a real signal (lag 0..`out.length − 1`) to `out`.
   *
   * MUST: the transform length is **at least twice** the signal length (the zero-padding condition
   * under which circular correlation equals linear correlation).
   * MUST: lags are below the signal length (`out.length ≤ input.length`). Lags at or above the
   * signal length are 0 in linear autocorrelation, and in the inverse transform's result the
   * values of negative lags wrap around into indices above "transform length − signal length"
   * (passing `[1, 2]` with transform length 4 gives 0 at index 2, as linear correlation says, but
   * the 2 of lag −1 at index 3). The two are not told apart; lags are limited to below the signal
   * length. The power spectrum is real and even-symmetric, so the inverse transform is obtained by
   * applying the same butterflies and dividing by `size` (equivalent to `np.fft.irfft(S·conj(S))`).
   */
  autocorrelation(input: Float32Array<ArrayBuffer>, out: Float64Array<ArrayBuffer>): void {
    if (input.length * 2 > this.size) {
      throw new RangeError(
        `Fft: autocorrelation requires a transform length of at least twice the signal length ${input.length} (only ${this.size})`,
      );
    }
    if (out.length > input.length) {
      throw new RangeError(
        `Fft: lag ${
          out.length - 1
        } is >= the signal length ${input.length} (it is 0 linearly, and negative lags wrap around into` +
          " indices beyond transform length − signal length — they are not told apart, so lags stay below the signal length)",
      );
    }
    this.#load(input);
    this.#butterflies();
    for (let bin = 0; bin < this.size; bin += 1) {
      this.#scratch[bin] = this.#re[bin] * this.#re[bin] + this.#im[bin] * this.#im[bin];
    }
    this.#load(this.#scratch);
    this.#butterflies();
    for (let lag = 0; lag < out.length; lag += 1) out[lag] = this.#re[lag] / this.size;
  }

  /** Loads the input in bit-reversed order (missing indices are zero = zero padding at the end). */
  #load(values: ArrayLike<number>): void {
    for (let index = 0; index < this.size; index += 1) {
      const source = this.#reversed[index];
      this.#re[index] = source < values.length ? values[source] : 0;
      this.#im[index] = 0;
    }
  }

  /** Transforms the bit-reversed `#re` / `#im` in place (DIT). */
  #butterflies(): void {
    for (let span = 2; span <= this.size; span <<= 1) {
      const half = span >> 1;
      const step = this.size / span;
      for (let start = 0; start < this.size; start += span) {
        for (let offset = 0, twiddle = 0; offset < half; offset += 1, twiddle += step) {
          const lower = start + offset;
          const upper = lower + half;
          const re = this.#re[upper] * this.#cos[twiddle] - this.#im[upper] * this.#sin[twiddle];
          const im = this.#re[upper] * this.#sin[twiddle] + this.#im[upper] * this.#cos[twiddle];
          this.#re[upper] = this.#re[lower] - re;
          this.#im[upper] = this.#im[lower] - im;
          this.#re[lower] += re;
          this.#im[lower] += im;
        }
      }
    }
  }
}
