/**
 * The f64 oracle of the GEMV sweep on the host, and a GPU output's error against it in units of
 * the band of `linear`.
 *
 * The oracle reads the same seeded word streams the sweep's buffers were filled from
 * (`resourceWordStream` of harness.ts — one stream per resource, in buffer order), so it never holds
 * a whole weight: the streams are walked once, output column by output column. A weight element is
 * restored as the kernels restore it — `f32(q) × scale` rounded to f32 (i4: `u − 8`, the low nibble
 * first; i8: the signed byte, byte 0 first; i2: `code − 2`, the low two bits first) — and the
 * products and their sum are taken in f64, the bias added last.
 */

import type { Tolerance } from "../reference/allclose.ts";
import { type GemvSweepStorage, isFloatGemvStorage } from "./gemv-cases.ts";

/** The shape the oracle evaluates. */
export type GemvOracleShape = {
  readonly storage: GemvSweepStorage;
  readonly group?: number;
  readonly n: number;
  readonly k: number;
};

/** The u32 word streams of a case's inputs, each in its buffer's order. */
export type GemvInputStreams = {
  /** `x[rows][k]` as f32 bits. */
  readonly x: () => number;
  /** The packed weight `[n][k]`. */
  readonly weight: () => number;
  /** `bias[n]` as f32 bits. */
  readonly bias: () => number;
  /** The scales as f32 bits: `[n][k / group]` for i4, `[n]` for i2 / i8; never read for a float storage. */
  readonly scale: () => number;
};

/** The bits of one element. */
const bitsOf = (storage: GemvSweepStorage): number =>
  storage === "i2" ? 2 : storage === "i4" ? 4 : storage === "i8" ? 8 : storage === "f32" ? 32 : 16;

/** Elements per u32 weight word. */
const elementsPerWord = (storage: GemvSweepStorage): number => 32 / bitsOf(storage);

/** The integer value of element `lane` of a weight word (i2 / i4 / i8). */
const decode = (storage: "i2" | "i4" | "i8", word: number, lane: number): number => {
  const bits = bitsOf(storage);
  const code = (word >>> (lane * bits)) & ((1 << bits) - 1);
  if (storage === "i8") return (code << 24) >> 24;
  return code - (1 << (bits - 1));
};

/** The value of an IEEE 754 binary16 bit pattern (exact in f64). */
const f16Of = (half: number): number => {
  const sign = half & 0x8000 ? -1 : 1;
  const exponent = (half >>> 10) & 0x1f;
  const fraction = half & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
};

/**
 * The value of element `lane` of a float-storage weight word, as the kernels restore it: f32 the word
 * itself; bf16 the half as the high 16 bits of an f32, the low half first; f16 the half-float, the
 * low half first. Each is exact in f32, so no rounding enters.
 */
const floatElement = (
  storage: "bf16" | "f16" | "f32",
  word: number,
  lane: number,
  f32: (bits: number) => number,
): number => {
  if (storage === "f32") return f32(word);
  const half = (word >>> (lane * 16)) & 0xffff;
  return storage === "bf16" ? f32((half << 16) >>> 0) : f16Of(half);
};

/** Reads one f32 from its u32 bits. */
const f32Of = (): ((bits: number) => number) => {
  const words = new Uint32Array(1);
  const floats = new Float32Array(words.buffer);
  return (bits) => {
    words[0] = bits;
    return floats[0];
  };
};

/** Waits for the next macrotask, so a long evaluation does not hold the page. */
const nextTask = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/** Elements (columns × k × rows) evaluated between two pauses. */
const SLICE_ELEMENTS = 1 << 24;

/**
 * The oracle's outputs `[rows][n]` (f64): `Σ_k x[r,k] · w[c,k] + bias[c]`, the weight restored in f32
 * as the kernels restore it. The streams are consumed: `x` for `rows · k` words, the weight, the
 * scale (an integer storage's only) and the bias for the whole of `[n, k]`. `pause` is awaited
 * between slices of columns.
 */
export const gemvOracle = async (
  shape: GemvOracleShape,
  rows: number,
  streams: GemvInputStreams,
  pause: () => Promise<void> = nextTask,
): Promise<Float64Array<ArrayBuffer>> => {
  const { storage, n, k } = shape;
  const perWord = elementsPerWord(storage);
  const group = storage === "i4" ? shape.group : k;
  if (group === undefined || k % perWord !== 0 || k % group !== 0) {
    throw new Error(`The GEMV oracle needs k a multiple of ${perWord} and of the group (k ${k})`);
  }
  const f32 = f32Of();
  const x = new Float64Array(rows * k);
  for (let index = 0; index < x.length; index += 1) x[index] = f32(streams.x());
  const out = new Float64Array(rows * n);
  const sums = new Float64Array(rows);
  const groupsPerRow = k / group;
  const columnsPerSlice = Math.max(1, Math.floor(SLICE_ELEMENTS / Math.max(1, k * rows)));
  for (let column = 0; column < n; column += 1) {
    if (column > 0 && column % columnsPerSlice === 0) await pause();
    sums.fill(0);
    for (let groupIndex = 0; groupIndex < groupsPerRow; groupIndex += 1) {
      const scale = isFloatGemvStorage(storage) ? 1 : f32(streams.scale());
      const start = groupIndex * group;
      for (let element = start; element < start + group; element += perWord) {
        const word = streams.weight();
        for (let lane = 0; lane < perWord; lane += 1) {
          const weight = isFloatGemvStorage(storage)
            ? floatElement(storage, word, lane, f32)
            : Math.fround(decode(storage, word, lane) * scale);
          const at = element + lane;
          for (let row = 0; row < rows; row += 1) sums[row] += x[row * k + at] * weight;
        }
      }
    }
    const bias = f32(streams.bias());
    for (let row = 0; row < rows; row += 1) out[row * n + column] = sums[row] + bias;
  }
  return out;
};

/** A GPU output's distance from the oracle. */
export type BandReading = {
  /** The largest `|actual − oracle|` over the finite outputs (absent when every output is non-finite). */
  readonly maxAbsError?: number;
  /**
   * The largest `|actual − oracle| / (atol + rtol · |oracle|)`: up to 1 inside the band. Absent when
   * a non-zero error meets a band of zero width, or when an output is not finite.
   */
  readonly bandUse?: number;
  readonly withinBand: boolean;
  /** Outputs that are NaN or infinite (each fails the band). */
  readonly nonFiniteOutputs: number;
};

/** Reads `actual` (the first `oracle.length` outputs of a row-major `[m][n]`) against the oracle. */
export const bandAgainstOracle = (
  actual: Float32Array<ArrayBuffer>,
  oracle: Float64Array<ArrayBuffer>,
  tolerance: Tolerance,
): BandReading => {
  if (actual.length < oracle.length) {
    throw new Error(`The output has ${actual.length} values, the oracle ${oracle.length}`);
  }
  let maxAbsError: number | undefined;
  let bandUse = 0;
  let nonFiniteOutputs = 0;
  for (let index = 0; index < oracle.length; index += 1) {
    const value = actual[index];
    if (!Number.isFinite(value)) {
      nonFiniteOutputs += 1;
      continue;
    }
    const expected = oracle[index];
    const error = Math.abs(value - expected);
    maxAbsError = Math.max(maxAbsError ?? 0, error);
    const width = tolerance.atol + tolerance.rtol * Math.abs(expected);
    if (error > 0) bandUse = Math.max(bandUse, width > 0 ? error / width : Infinity);
  }
  const finiteUse = Number.isFinite(bandUse) && nonFiniteOutputs === 0;
  return {
    ...(maxAbsError === undefined ? {} : { maxAbsError }),
    ...(finiteUse ? { bandUse } : {}),
    withinBand: finiteUse && bandUse <= 1,
    nonFiniteOutputs,
  };
};
