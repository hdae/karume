// Builder helpers for the f16 storage tests (host-side f32 → f16 encoding).
//
// The encoder is **test-only** (in production the exporter — the Python side — encodes, and the
// runtime only decodes). This is a tool for building "weights that hold only values f16 can
// represent", and its role is to reproduce ADR 0006's fake-quant on the host:
//   bytes  = the quantized storage bytes (handed to the GPU / load path)
//   values = those bytes turned back into f32 **by the production decodeF16** (the expected
//            values the CPU reference uses)
// MUST: the expected values come from decodeF16, not from the encoder. If both sides came from the
// encoder, an encoder bug would land in the expected values too and the test would be
// tautological.

import { decodeF16 } from "../../src/format/f16.ts";

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/**
 * f32 → f16 16-bit pattern (round-to-nearest-even).
 * Overflow goes to ±Inf; underflow goes through the subnormals down to ±0.
 */
export const f32ToF16Bits = (value: number): number => {
  F32[0] = value;
  const bits = U32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  // Inf / NaN: all-ones exponent. NaN keeps only the quiet bit (the payload is not preserved)
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa !== 0 ? 0x200 : 0);
  const shifted = exponent - 127 + 15;
  if (shifted >= 0x1f) return sign | 0x7c00;
  if (shifted <= 0) {
    // Subnormal range (set the implicit 1, then shift right). -11 and below is ±0
    if (shifted < -10) return sign;
    const full = mantissa | 0x800000;
    const shift = 14 - shifted;
    const truncated = full >>> shift;
    const remainder = full & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    const roundUp = remainder > halfway || (remainder === halfway && (truncated & 1) === 1);
    return sign | (truncated + (roundUp ? 1 : 0));
  }
  const truncated = (shifted << 10) | (mantissa >>> 13);
  const remainder = mantissa & 0x1fff;
  const roundUp = remainder > 0x1000 || (remainder === 0x1000 && (truncated & 1) === 1);
  // A carry propagates from the mantissa into the exponent as is (0x7bff + 1 = 0x7c00 = Inf)
  return sign | (truncated + (roundUp ? 1 : 0));
};

/** f16 16-bit patterns → little-endian bytes. */
export const f16BytesFromBits = (bits: readonly number[]): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(bits.length * 2);
  const view = new DataView(bytes.buffer);
  bits.forEach((pattern, index) => view.setUint16(index * 2, pattern & 0xffff, true));
  return bytes;
};

export type QuantizedWeight = {
  /** The raw f16 bytes stored in safetensors. */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** The values after quantization (fake-quant — the CPU reference uses these as the weights). */
  readonly values: Float32Array<ArrayBuffer>;
};

/** Rounds a sequence of f32 to f16 and returns the storage bytes and the rounded values. */
export const quantizeF16 = (values: ArrayLike<number>): QuantizedWeight => {
  const bits = Array.from({ length: values.length }, (_, i) => f32ToF16Bits(values[i]));
  const bytes = f16BytesFromBits(bits);
  return { bytes, values: decodeF16(bytes) };
};
