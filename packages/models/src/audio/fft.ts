/**
 * Iterative radix-2 FFT. It outputs only the quantity its user needs — the **power spectrum** of a
 * real signal.
 *
 * ## Why in `src/audio/`
 *
 * It is not a general FFT (real signals, powers of 2, only the power spectrum, resident scratch).
 * It is an audio primitive rather than a part of one family, so it sits in the shared layer; its
 * user is the log-mel front end of the EmbeddingGemma family's audio tower
 * (`src/embeddinggemma/audio.ts`, the power spectrum of a windowed frame of f32 values, transformed
 * in f64).
 *
 * ## MUST: run the transform in f64
 *
 * numpy's FFT promotes even f32 input to double precision, so a canonical spec written against
 * `np.fft` is f64. The EmbeddingGemma front end's upstream runs its FFT in f32 (complex64) instead;
 * there the f64 transform of the same frames is at most one ulp of a log-mel value from upstream's
 * (`src/embeddinggemma/audio.ts` — the frames themselves are upstream's f32 product, reproduced
 * exactly, since that product is upstream's arithmetic and not an implementation detail), and the
 * front end's digests pin the bits of upstream's arithmetic with an f64 FFT.
 *
 * The choice of **computing at a precision at least that of the canonical spec and measuring the
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
 * The only state is the scratch (there is no point carrying it over between calls), so each call
 * takes its result fully into `out`.
 */
export class Fft {
  readonly size: number;
  readonly #reversed: Uint32Array<ArrayBuffer>;
  readonly #cos: Float64Array<ArrayBuffer>;
  readonly #sin: Float64Array<ArrayBuffer>;
  readonly #re: Float64Array<ArrayBuffer>;
  readonly #im: Float64Array<ArrayBuffer>;

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
  }

  /**
   * Writes the power spectrum `|X[k]|²` of a real signal (length ≤ `size`, the shortfall
   * zero-padded at the **end**) to `out[0..size/2]` (the same order as
   * `np.abs(np.fft.rfft(x, n=size))**2`). The signal may be f32 or f64; the transform is f64
   * either way.
   */
  powerSpectrum(
    input: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>,
    out: Float64Array<ArrayBuffer>,
  ): void {
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
