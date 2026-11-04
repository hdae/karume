// Builder helpers for the bf16 storage tests (host-side f32 → bf16 encoding).
//
// The encoder is **test-only** (in production the exporter — the Python side — encodes, and the
// runtime only decodes). Same role as `helpers/f16.ts` for f16: it reproduces ADR 0006's fake-quant
// on the host and returns the pair
//   bytes  = the quantized storage bytes (handed to the GPU / load path)
//   values = those bytes turned back into f32 **by the production decodeBf16** (the expected
//            values the CPU reference uses).
// MUST: the expected values come from decodeBf16, not from the encoder (if both sides came from
// the encoder, an encoder bug would land in the expected values too and the test would be
// tautological).

import { decodeBf16 } from "../../src/format/bf16.ts";

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/**
 * f32 → bf16 16-bit pattern (round-to-nearest-even — the same rounding as torch's
 * `Tensor.to(bfloat16)`). bf16 is the upper 16 bits of f32, so rounding away the lower 16 bits is
 * enough (the exponent range is the same, so nothing overflows). NaN comes back with the quiet bit
 * set (the payload is not preserved).
 */
export const f32ToBf16Bits = (value: number): number => {
  F32[0] = value;
  const bits = U32[0];
  if (Number.isNaN(value)) return ((bits >>> 16) | 0x0040) & 0xffff;
  const lower = bits & 0xffff;
  const upper = bits >>> 16;
  const roundUp = lower > 0x8000 || (lower === 0x8000 && (upper & 1) === 1);
  return (upper + (roundUp ? 1 : 0)) & 0xffff;
};

/** bf16 16-bit patterns → little-endian bytes. */
export const bf16BytesFromBits = (bits: readonly number[]): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(bits.length * 2);
  const view = new DataView(bytes.buffer);
  bits.forEach((pattern, index) => view.setUint16(index * 2, pattern & 0xffff, true));
  return bytes;
};

export type QuantizedWeight = {
  /** The raw bf16 bytes stored in the container. */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** The values after quantization (fake-quant — the CPU reference uses these as the weights). */
  readonly values: Float32Array<ArrayBuffer>;
};

/** Rounds a sequence of f32 to bf16 and returns the storage bytes and the rounded values. */
export const quantizeBf16 = (values: ArrayLike<number>): QuantizedWeight => {
  const bits = Array.from({ length: values.length }, (_, i) => f32ToBf16Bits(values[i]));
  const bytes = bf16BytesFromBits(bits);
  return { bytes, values: decodeBf16(bytes) };
};
