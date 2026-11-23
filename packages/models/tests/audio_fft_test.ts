// Checks of the shared FFT (`src/audio/fft.ts`). Needs neither a GPU nor real assets.
//
// The pipeline-level tests reach this class only through a composed result (the EmbeddingGemma
// audio front end's frames), and none of the three `RangeError`s is exercised there. Moreover
// "the values are right" is only seen through that composition; the definition of `powerSpectrum`
// on its own (the same layout and the same normalisation as `np.fft`) is not pinned. This file
// pins three things:
//
// (1) **Outside the accepted set** (a transform length that is not a power of 2 / an input longer than
//     the transform / an output smaller than the rfft bin count) everything fails with
//     `RangeError`.
// (2) **Agreement with a naive DFT** (rtol 1e-12). If the butterfly, the bit reversal or the sign of
//     the twiddle factors breaks, the features just become "another plausible value" and pass to the
//     end with the right shape and range.
// (3) **The values stay right after zero padding** (signal length < transform length: power as
//     defined).
//     NOTE: the **direction** of the padding (at the end / at the start) cannot be pinned here.
//     Padding at the start is only a cyclic shift and does not change the power spectrum
//     (`[1,2,0,0]` and `[0,0,1,2]` both have power `[9,5,1]`). The direction is not observable
//     through the public API, so no API is added for it.

import { test } from "vite-plus/test";
import { assertAlmostEquals, assertThrows } from "@std/assert";
import { Fft } from "../src/audio/fft.ts";

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

test("Fft: the power stays right when a signal shorter than the transform is zero-padded", () => {
  // Running `[1, 0]` through transform length 4 gives 1 in every bin (the power of a unit impulse).
  const out = new Float64Array(3);
  new Fft(4).powerSpectrum(Float32Array.from([1, 0]), out);
  for (let bin = 0; bin < out.length; bin += 1) {
    assertAlmostEquals(out[bin], 1, 1e-12, `bin ${bin}`);
  }
});
