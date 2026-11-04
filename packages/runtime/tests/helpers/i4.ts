// Builders for i4 storage tests (host-side f32 → symmetric int4 grouped along K, encoded + packed).
//
// The encoding is **test-only** (in production the exporter encodes, on the Python side in
// `karume/emit.py: pack_int4`, and the runtime only decodes). The pack order "element 2i = low
// nibble / 2i+1 = high nibble, stored value u = q + 8" is the exporter's spec, **written out here**
// — taking it from the implementation (format/i4.ts) would make the round trip tautological (the
// same rule as helpers/i8.ts).
//   bytes  = raw bytes of the packed 4-bit values (passed to the GPU / load path)
//   scale  = scale per group (carried in the distribution form as an F32 tensor of group shape)
//   values = those bytes decoded back to f32 **by the production decodeI4** (the expected values
//            the CPU reference uses)

import { decodeI4 } from "../../src/format/i4.ts";
import { F32_TINY } from "./i8.ts";

export type QuantizedI4 = {
  /** Raw I4 bytes to put in safetensors (numel / 2 bytes). */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /**
   * The quantized **integer values** `q ∈ [−7, 7]` (flat, before packing). This is the form the
   * w4a8 CPU reference (`referenceLinearW4a8`) takes, built in the same loop as `bytes` —
   * re-deriving it from the bytes would put a second copy of the nibble-order spec in the tests.
   */
  readonly q: Int8Array<ArrayBuffer>;
  /** Scale per group (flat = row × group order). */
  readonly scale: Float32Array<ArrayBuffer>;
  /**
   * Group shape of the scale tensor (rank 2 whatever the input rank =
   * `[shape[0], row length / g]`).
   */
  readonly scaleShape: readonly number[];
  /** The quantized values (fake-quant — the CPU reference uses these as the weights). */
  readonly values: Float32Array<ArrayBuffer>;
};

/**
 * Fake-quant + pack of symmetric int4 grouped along K (ADR 0069 decisions 3 / 4).
 *
 * `scale = max(amax_group / 7, tiny)` / `q = round(w / scale)` **clamped to ±7**.
 * Rounding is half-away-from-zero (the same as the i8 helper — the expected values come from
 * decodeI4, so this does not affect what the runtime is checked against).
 *
 * Rows are **the leading dimension, with the rest flattened** (the `channel_rows` rule — the same
 * as the last dimension for linear `[O,I]` / embedding `[V,D]`, `Cin·K` for conv1d `[O,Cin,K]`).
 * The scale is rank 2 whatever the input rank. MUST: written out here, not taken from the
 * implementation (format/i4.ts) (taking it would make the round trip tautological — the same rule
 * as helpers/i8.ts).
 */
export const quantizeI4 = (
  values: ArrayLike<number>,
  shape: readonly number[],
  groupSize: number,
): QuantizedI4 => {
  const rows = shape[0];
  const width = values.length / rows;
  if (width % groupSize !== 0) {
    throw new Error(`quantizeI4: row length ${width} is not divisible by group_size ${groupSize}`);
  }
  const groups = width / groupSize;

  const scale = new Float32Array(rows * groups);
  for (let i = 0; i < values.length; i += 1) {
    const row = Math.floor(i / width);
    const group = Math.floor((i % width) / groupSize);
    const slot = row * groups + group;
    scale[slot] = Math.max(scale[slot], Math.abs(values[i]));
  }
  for (let slot = 0; slot < scale.length; slot += 1) {
    scale[slot] = Math.max(Math.fround(scale[slot] / 7), F32_TINY);
  }

  // pack: element 2i = low nibble / 2i+1 = high nibble, u = q + 8 (source of truth = pack_int4 in
  // emit.py)
  const bytes = new Uint8Array(values.length / 2);
  const q = new Int8Array(values.length);
  for (let i = 0; i < values.length; i += 1) {
    const row = Math.floor(i / width);
    const group = Math.floor((i % width) / groupSize);
    const ratio = values[i] / scale[row * groups + group];
    const rounded = Math.sign(ratio) * Math.round(Math.abs(ratio));
    q[i] = Math.max(-7, Math.min(7, rounded));
    bytes[i >> 1] |= (i & 1) === 1 ? (q[i] + 8) << 4 : q[i] + 8;
  }
  const scaleShape = [rows, groups];
  return {
    bytes,
    q,
    scale,
    scaleShape,
    values: decodeI4(bytes, shape, scale, scaleShape, groupSize),
  };
};
