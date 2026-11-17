/**
 * The encoders — the exporter's storage conversions, bit for bit, from f32 source values:
 *
 * - `int8-sym`: per channel of the row axis, `scale = max(fl32(amax / 127), f32 tiny)` and
 *   `q = clamp(round_half_even(fl32(w / scale)), ±127)` (`channel_scale` and `quantize_to_int8` in
 *   `tools/exporter/src/karume/quantize.py`; `_convert_for_storage` in `emit.py` stores those codes
 *   in the weight's own memory order and the scale as one f32 per channel). `fl32(a / b)` is
 *   `Math.fround(a / b)`: for two f32 operands the double-precision quotient rounded once more to f32
 *   is the correctly rounded f32 quotient (53 ≥ 2·24 + 2), which is torch's f32 division.
 * - `bf16`: the value itself, or an error — the exporter stores a weight as bf16 only when it is
 *   exactly a bf16 value (`_is_bf16_exact` in `emit.py`, a `torch.equal`, which no NaN passes);
 *   nothing is rounded here.
 * - `f32`: the value itself (every source dtype is exact in f32).
 *
 * Each encoder takes the whole tensor and returns its whole payload; the pieces of a split tensor
 * are row ranges of it.
 *
 * SPEC: docs/spec/convert.md#the-encodings
 */

import type { CodecName } from "@karume/runtime";
import { ConvertError } from "./errors.ts";

/** One side of the int8 range; −128 is not used (`INT8_MAX` in `quantize.py`). */
const INT8_MAX = 127;

/**
 * The smallest scale, `torch.finfo(torch.float32).tiny` (the clamp of `channel_scale`): an all-zero
 * channel gets it and its codes are 0.
 */
const F32_TINY = 2 ** -126;

/** The target codecs this package encodes; a plan that asks for another is refused. */
export const ENCODED_CODECS: readonly CodecName[] = ["int8-sym", "bf16", "f32"];

/** The payload of one initializer, and of its scale when it is quantized. */
export type Encoded = {
  readonly data: Uint8Array<ArrayBuffer>;
  readonly scale?: Uint8Array<ArrayBuffer>;
};

const bytesOf = (
  array: Float32Array<ArrayBuffer> | Int8Array<ArrayBuffer> | Uint16Array<ArrayBuffer>,
): Uint8Array<ArrayBuffer> => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);

/** `torch.round` (half to even) of an f32 value; `Math.round` rounds a half up. */
const roundHalfEven = (value: number): number => {
  const rounded = Math.round(value);
  return rounded - value === 0.5 && (rounded & 1) !== 0 ? rounded - 1 : rounded;
};

/**
 * `int8-sym`: one f32 scale per index of `shape[rowAxis]` over everything else, the codes in the
 * weight's memory order. Row axis 0 is a row of the storage fold; row axis 1 is the channel of a
 * transposed layout (`[Cin, Cout, K]` of `ConvTranspose1d`), its elements strided through memory.
 *
 * A non-finite value is refused: the exporter's round trip (`q · scale == w`) fails on one, so it
 * writes no such weight.
 */
export const encodeInt8Sym = (
  values: Float32Array<ArrayBuffer>,
  shape: readonly number[],
  rowAxis: 0 | 1,
  where: string,
): Encoded => {
  const channels = shape[rowAxis];
  const outer = rowAxis === 0 ? 1 : shape[0];
  const inner = values.length / (outer * channels);
  const amax = new Float64Array(channels);
  for (let o = 0, index = 0; o < outer; o += 1) {
    for (let c = 0; c < channels; c += 1) {
      let max = amax[c];
      for (let i = 0; i < inner; i += 1, index += 1) {
        const value = values[index];
        if (!Number.isFinite(value)) {
          throw new ConvertError(
            `${where}: element ${index} is ${value}; int8-sym quantizes finite values only`,
          );
        }
        const magnitude = Math.abs(value);
        if (magnitude > max) max = magnitude;
      }
      amax[c] = max;
    }
  }
  const scale = new Float32Array(channels);
  for (let c = 0; c < channels; c += 1) {
    const step = Math.fround(amax[c] / INT8_MAX);
    scale[c] = step < F32_TINY ? F32_TINY : step;
  }
  const codes = new Int8Array(values.length);
  for (let o = 0, index = 0; o < outer; o += 1) {
    for (let c = 0; c < channels; c += 1) {
      const step = scale[c];
      for (let i = 0; i < inner; i += 1, index += 1) {
        const q = roundHalfEven(Math.fround(values[index] / step));
        codes[index] = q > INT8_MAX ? INT8_MAX : q < -INT8_MAX ? -INT8_MAX : q;
      }
    }
  }
  return { data: bytesOf(codes), scale: bytesOf(scale) };
};

/**
 * `bf16`: the upper half of each f32 value. A value whose lower half is not zero is not a bf16
 * value, and a NaN is refused like the exporter refuses it; both are errors naming the element.
 */
export const encodeBf16 = (
  values: Float32Array<ArrayBuffer>,
  where: string,
): Uint8Array<ArrayBuffer> => {
  const bits = new Uint32Array(values.buffer, values.byteOffset, values.length);
  const out = new Uint16Array(values.length);
  for (let i = 0; i < bits.length; i += 1) {
    const word = bits[i];
    if ((word & 0x7fffffff) > 0x7f800000) {
      throw new ConvertError(`${where}: element ${i} is NaN, which bf16 storage does not take`);
    }
    if ((word & 0xffff) !== 0) {
      throw new ConvertError(
        `${where}: element ${i} (${values[i]}) is not a bf16 value (bf16 storage takes values as they are and does not round them)`,
      );
    }
    out[i] = word >>> 16;
  }
  return bytesOf(out);
};

/** Encodes one initializer's values to the target codec (one of {@link ENCODED_CODECS}). */
export const encodeValues = (
  codec: CodecName,
  values: Float32Array<ArrayBuffer>,
  shape: readonly number[],
  rowAxis: 0 | 1,
  where: string,
): Encoded => {
  switch (codec) {
    case "int8-sym":
      return encodeInt8Sym(values, shape, rowAxis, where);
    case "bf16":
      return { data: encodeBf16(values, where) };
    case "f32":
      return { data: bytesOf(values) };
    default:
      throw new ConvertError(`${where}: codec '${codec}' is not one this package encodes`);
  }
};
