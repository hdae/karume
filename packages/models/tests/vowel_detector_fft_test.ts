// Checks of the vowel detector's FFT (`src/vowel-detector/fft.ts`). Needs neither a GPU nor real
// assets.
//
// Today the only path that reaches this class is the one Python parity test through
// `extractFeatures`, and none of the four `RangeError`s is exercised. Moreover "the values are right"
// is only seen through the composed result of the feature extraction; the definitions of
// `powerSpectrum` / `autocorrelation` on their own (the same layout and the same normalisation as
// `np.fft`) are not pinned. This file pins three things:
//
// (1) **Outside the accepted set** (a transform length that is not a power of 2 / an input longer than
//     the transform / an output smaller than the rfft bin count / a transform too short for the
//     autocorrelation / an autocorrelation lag at or past the signal length) everything fails with
//     `RangeError`.
// (2) **Agreement with a naive DFT** (rtol 1e-12). If the butterfly, the bit reversal or the sign of
//     the twiddle factors breaks, the features just become "another plausible value" and pass to the
//     end with the right shape and range.
// (3) **The values stay right after zero padding** (signal length < transform length: power and
//     linear autocorrelation as defined).
//     NOTE: the **direction** of the padding (at the end / at the start) cannot be pinned here.
//     Padding at the start is only a cyclic shift and changes neither the power spectrum nor the
//     autocorrelation (`[1,2,0,0]` and `[0,0,1,2]` both have power `[9,5,1]` and both have the
//     autocorrelation `[5,2]` over the first 2 lags). The direction is not observable through the
//     public API, so no API is added for it.

import { test } from "vite-plus/test";
import { assertAlmostEquals, assertThrows } from "@std/assert";
import { Fft } from "../src/vowel-detector/fft.ts";

/** `|X[k]|²` of a naive DFT (zero padding at the end, k = 0..size/2). */
const naivePowerSpectrum = (input: Float32Array, size: number): number[] => {
  const bins: number[] = [];
  for (let k = 0; k <= size / 2; k += 1) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < size; n += 1) {
      const value = n < input.length ? input[n] : 0;
      const angle = (-2 * Math.PI * k * n) / size;
      re += value * Math.cos(angle);
      im += value * Math.sin(angle);
    }
    bins.push(re * re + im * im);
  }
  return bins;
};

/** Naive linear autocorrelation `sum_n x[n]·x[n+lag]` (0 outside the zero padding). */
const naiveAutocorrelation = (input: Float32Array, lags: number): number[] => {
  const out: number[] = [];
  for (let lag = 0; lag < lags; lag += 1) {
    let sum = 0;
    for (let n = 0; n + lag < input.length; n += 1) sum += input[n] * input[n + lag];
    out.push(sum);
  }
  return out;
};

/** Deterministic pseudo-random values (fixed seed — independent of the global RNG). */
const pseudoRandom = (length: number, seed: number): Float32Array<ArrayBuffer> => {
  const values = new Float32Array(length);
  let state = seed;
  for (let index = 0; index < length; index += 1) {
    state = (state * 1103515245 + 12345) % 2147483648;
    values[index] = state / 2147483648 - 0.5;
  }
  return values;
};

test("Fft: a transform length that is not a power of 2 cannot be constructed", () => {
  for (const size of [3, 1, 0, -4, 2.5]) {
    assertThrows(() => new Fft(size), RangeError, "is not a power of 2 >= 2", `size=${size}`);
  }
});

test("Fft.powerSpectrum: an input longer than the transform length fails (no silent truncation)", () => {
  assertThrows(
    () => new Fft(8).powerSpectrum(new Float32Array(9), new Float64Array(5)),
    RangeError,
    "exceeds the transform length 8",
  );
});

test("Fft.powerSpectrum: an output smaller than the rfft bin count fails", () => {
  // `bins = size/2 + 1` = 5 (the same layout as `np.fft.rfft`).
  assertThrows(
    () => new Fft(8).powerSpectrum(new Float32Array(4), new Float64Array(4)),
    RangeError,
    "is fewer than the 5 rfft bins",
  );
});

test("Fft.autocorrelation: a transform length below twice the signal length fails (the condition under which it turns into a cyclic correlation)", () => {
  assertThrows(
    () => new Fft(8).autocorrelation(new Float32Array(5), new Float64Array(4)),
    RangeError,
    "requires a transform length of at least twice the signal length",
  );
});

test("Fft.autocorrelation: asking for a lag at or past the signal length fails (linearly 0, or an index a negative lag wraps into)", () => {
  // Running `[1, 2]` through transform length 4 gives the inverse transform [5, 2, 0, 2]: index 2 is 0
  // as in the linear case, index 3 is the value of lag −1 (0 if linear). The two are not told apart;
  // it fails at lag 2.
  assertThrows(
    () => new Fft(4).autocorrelation(Float32Array.from([1, 2]), new Float64Array(3)),
    RangeError,
    "lag 2 is >= the signal length 2",
  );
  // Boundary: lags up to signal length − 1 (out.length = signal length) pass as a linear
  // autocorrelation.
  const out = new Float64Array(2);
  new Fft(4).autocorrelation(Float32Array.from([1, 2]), out);
  assertAlmostEquals(out[0], 5, 1e-12);
  assertAlmostEquals(out[1], 2, 1e-12);
});

test("Fft.powerSpectrum: matches a naive DFT (a real signal of length 8)", () => {
  const size = 8;
  const input = pseudoRandom(size, 20260904);
  const out = new Float64Array(size / 2 + 1);
  new Fft(size).powerSpectrum(input, out);
  const expected = naivePowerSpectrum(input, size);
  for (let bin = 0; bin < expected.length; bin += 1) {
    // Relative 1e-12 — a width that leaves only the rounding difference between the f64 butterfly
    // and the naive DFT.
    assertAlmostEquals(
      out[bin],
      expected[bin],
      Math.abs(expected[bin]) * 1e-12 + 1e-15,
      `bin ${bin}`,
    );
  }
});

test("Fft.autocorrelation: matches a naive linear autocorrelation (length 4, transform length 8)", () => {
  const input = pseudoRandom(4, 20260905);
  const out = new Float64Array(4);
  new Fft(8).autocorrelation(input, out);
  const expected = naiveAutocorrelation(input, out.length);
  for (let lag = 0; lag < expected.length; lag += 1) {
    assertAlmostEquals(
      out[lag],
      expected[lag],
      Math.abs(expected[lag]) * 1e-12 + 1e-15,
      `lag ${lag}`,
    );
  }
});

test("Fft: power and linear autocorrelation stay right when a signal shorter than the transform is zero-padded", () => {
  // Running `[1, 0]` through transform length 4 gives 1 in every bin (the power of a unit impulse).
  const out = new Float64Array(3);
  new Fft(4).powerSpectrum(Float32Array.from([1, 0]), out);
  for (let bin = 0; bin < out.length; bin += 1) {
    assertAlmostEquals(out[bin], 1, 1e-12, `bin ${bin}`);
  }

  // The linear autocorrelation of `[1, 2]` is lag 0 = 5, lag 1 = 2 (independent of the padding
  // direction — the NOTE under (3) at the top).
  const lags = new Float64Array(2);
  new Fft(4).autocorrelation(Float32Array.from([1, 2]), lags);
  assertAlmostEquals(lags[0], 5, 1e-12);
  assertAlmostEquals(lags[1], 2, 1e-12);
});
