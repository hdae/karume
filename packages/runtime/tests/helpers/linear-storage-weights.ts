/**
 * Linear weights `[n, k]` in every storage the GEMM skeleton reads (f32 / f16 / bf16 / i8 / i4 with
 * its group length / i2), as the kernels bind them, with every element as the kernels decode it —
 * the input of an f64 evaluation that checks a kernel against the op's meaning over the weights the
 * GPU actually reads.
 *
 * The layouts are written out here from the storage rules (src/kernels/weight-storage.ts documents
 * them), not taken from the runtime's decoders: two halves of the flat index per f16 / bf16 word
 * (element 2i in the low half), four i8 codes per word (element i in byte i), eight i4 codes per
 * word (element 2i in the low nibble of byte i, stored `q + 8`), sixteen i2 codes per word (element
 * i at bits 2·(i mod 16), stored `q + 2`). An integer element decodes as `f32(q) · scale` rounded
 * once in f32, the multiply the kernels run; the scale is per output channel for i8 and i2 and per
 * group of `groupSize` elements along k for i4.
 *
 * Browser-safe (no `node:` import): `@karume/kernel-checks` builds its fixtures from it.
 */

import type { WeightStorage } from "../../src/kernels/weight-storage.ts";
import { f32ToBf16Bits } from "./bf16.ts";
import { f32ToF16Bits } from "./f16.ts";

/** A weight storage and, for i4, its group length. */
export type LinearWeightStorage =
  | { readonly storage: "f32" | "f16" | "bf16" | "i8" | "i2"; readonly groupSize?: undefined }
  | { readonly storage: "i4"; readonly groupSize: number };

export type EncodedLinearWeight = LinearWeightStorage & {
  /** The weight buffer's words as the kernels bind them (padded to 16 bytes). */
  readonly words: Uint32Array<ArrayBuffer>;
  /** The scale buffer of i8 / i4 / i2 (`[n]`, or `[n, k / groupSize]` for i4); absent otherwise. */
  readonly scale: Float32Array<ArrayBuffer> | undefined;
  /** Every element as the kernels decode it, `[n, k]` row-major. */
  readonly values: Float32Array<ArrayBuffer>;
};

/**
 * The fixture of the weights: `irregular` values of mixed signs, about ±0.5, so the sums cancel and
 * round; `exact` small integers (integer codes with power-of-two scales), so every product and every
 * partial sum of integer activations is exact in f32 and no order of the additions can change it.
 */
export type WeightFixture = "irregular" | "exact";

/** Words of a byte array, padded to a whole number of 16-byte quads. */
const wordsOf = (bytes: Uint8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const padded = new Uint8Array(Math.max(16, Math.ceil(bytes.length / 16) * 16));
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
};

/** The value of f16 bits (exact). */
const f16Value = (bits: number): number => {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

/** The value of bf16 bits (the upper half of an f32 — exact). */
const bf16Value = (bits: number): number =>
  new Float32Array(new Uint32Array([bits << 16]).buffer)[0];

/** The float value of element `i` of channel `column` (before any rounding to the storage). */
const floatValue = (fixture: WeightFixture, i: number): number =>
  fixture === "exact" ? ((i * 11) % 7) - 3 : Math.cos(i * 0.011 + 0.3) * 0.5;

/** The integer code of element `i` (in the storage's range) and the scales of its fixture. */
const integerCode = (storage: "i8" | "i4" | "i2", fixture: WeightFixture, i: number): number => {
  if (fixture === "exact") return storage === "i2" ? ((i * 11) % 4) - 2 : ((i * 11) % 7) - 3;
  const unit = Math.cos(i * 0.011 + 0.3);
  if (storage === "i8") return Math.max(-127, Math.min(127, Math.round(unit * 127)));
  if (storage === "i4") return Math.max(-7, Math.min(7, Math.round(unit * 7)));
  return Math.max(-2, Math.min(1, Math.floor(unit * 2)));
};

/**
 * The scale of channel `column` (and, for i4, group `group`): powers of two for the exact fixture,
 * values off powers of two for the irregular one (decoded weights about ±0.5).
 */
const scaleOf = (
  storage: "i8" | "i4" | "i2",
  fixture: WeightFixture,
  column: number,
  group: number,
): number => {
  if (fixture === "exact") return storage === "i4" ? 2 ** ((column + group) % 2) : 1;
  const spread = 0.5 + ((column * 7 + group * 3) % 5) / 8;
  return Math.fround((storage === "i8" ? 0.5 / 127 : storage === "i4" ? 0.5 / 7 : 0.25) * spread);
};

/** The weights `[n, k]` of a storage and fixture. */
export const encodeLinearWeight = (
  { storage, groupSize }: LinearWeightStorage,
  n: number,
  k: number,
  fixture: WeightFixture,
): EncodedLinearWeight => {
  const count = n * k;
  const values = new Float32Array(count);
  if (storage === "f32") {
    for (let i = 0; i < count; i += 1) values[i] = floatValue(fixture, i);
    return {
      storage,
      words: wordsOf(new Uint8Array(values.slice().buffer)),
      scale: undefined,
      values,
    };
  }
  if (storage === "f16" || storage === "bf16") {
    const bytes = new Uint8Array(count * 2);
    for (let i = 0; i < count; i += 1) {
      const bits =
        storage === "f16"
          ? f32ToF16Bits(floatValue(fixture, i))
          : f32ToBf16Bits(floatValue(fixture, i));
      bytes[2 * i] = bits & 0xff;
      bytes[2 * i + 1] = bits >>> 8;
      values[i] = storage === "f16" ? f16Value(bits) : bf16Value(bits);
    }
    return { storage, words: wordsOf(bytes), scale: undefined, values };
  }
  if (storage === "i4" && (groupSize === undefined || k % groupSize !== 0)) {
    throw new Error(`encodeLinearWeight: i4 needs a group length that divides k = ${k}`);
  }
  const group = storage === "i4" ? groupSize : k;
  const groups = k / group;
  const scale = new Float32Array(storage === "i4" ? n * groups : n);
  for (let column = 0; column < n; column += 1) {
    for (let g = 0; g < (storage === "i4" ? groups : 1); g += 1) {
      scale[column * (storage === "i4" ? groups : 1) + g] = scaleOf(storage, fixture, column, g);
    }
  }
  const perByte = storage === "i8" ? 1 : storage === "i4" ? 2 : 4;
  const bytes = new Uint8Array(Math.ceil(count / perByte));
  for (let i = 0; i < count; i += 1) {
    const column = Math.floor(i / k);
    const q = integerCode(storage, fixture, i);
    const channelScale =
      storage === "i4" ? scale[column * groups + Math.floor((i % k) / group)] : scale[column];
    values[i] = Math.fround(q * channelScale);
    if (storage === "i8") bytes[i] = q & 0xff;
    else if (storage === "i4") bytes[i >> 1] |= (q + 8) << ((i & 1) * 4);
    else bytes[i >> 2] |= (q + 2) << ((i & 3) * 2);
  }
  return storage === "i4"
    ? { storage, groupSize: group, words: wordsOf(bytes), scale, values }
    : { storage, words: wordsOf(bytes), scale, values };
};

/** The name of a storage in a case label (`i4 g32`). */
export const storageLabel = ({ storage, groupSize }: LinearWeightStorage): string =>
  storage === "i4" ? `i4 g${groupSize}` : storage;

/** Whether a storage binds a scale buffer (the kernels' rule). */
export const hasWeightScale = (storage: WeightStorage): boolean =>
  storage === "i8" || storage === "i4" || storage === "i2";
