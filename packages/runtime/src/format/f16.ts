/**
 * IEEE 754 binary16 → f32 decode (for the ineligible slots of ADR 0018 / the host mirror of the
 * GPU-side `unpack2x16float`).
 *
 * MUST: decoding lives in this one place only. If the load path and the test's expected values
 * interpret f16 separately, the test becomes "both sides wrong in the same way" and differences in
 * subnormal or NaN handling go unnoticed.
 *
 * f16 → f32 is **always exact** (both the exponent range and the mantissa fit in f32). It therefore
 * MUST match the GPU-side `unpack2x16float` bit for bit, and a real-GPU test pins this for all
 * 65,536 patterns (only NaN payload propagation is outside WGSL's guarantees, so it is checked as
 * "NaN on both sides").
 */

/** Bytes per f16 element. */
const F16_BYTES = 2;

/**
 * f16 16-bit pattern → f32 number (only the low 16 bits are read).
 *
 * - Exponent 0: ±0 (mantissa 0) and subnormals. `mantissa · 2^-24` is exact in f32.
 * - Exponent 31: ±Inf (mantissa 0) and NaN. **NaN payloads are not kept** (WGSL does not guarantee
 *   them either).
 * - Otherwise: `(1 + mantissa/1024) · 2^(exp-15)`.
 */
export const f16BitsToF32 = (bits: number): number => {
  const pattern = bits & 0xffff;
  const sign = (pattern & 0x8000) !== 0 ? -1 : 1;
  const exponent = (pattern >> 10) & 0x1f;
  const mantissa = pattern & 0x3ff;
  if (exponent === 0) {
    // Mantissa 0 gives a signed zero (-1 * 0 = -0 — kept in a form Object.is can tell apart)
    return sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

/**
 * Decodes f16 bytes (the raw bytes of a container block — little endian) into an f32 array.
 * The merge layer's declaration check has already ensured the length is a multiple of 2.
 *
 * MUST: read one element at a time through a DataView. A `Uint16Array` view requires a 2-byte
 * aligned byteOffset, so it cannot be created at some positions in the file.
 */
export const decodeF16 = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const count = bytes.byteLength / F16_BYTES;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = f16BitsToF32(view.getUint16(i * F16_BYTES, true));
  }
  return out;
};

/**
 * f32 → f16 → f32 **rounding** (IEEE 754 binary16 round-to-nearest-even).
 *
 * The source of truth for the numerics contract of the f16 **compute** variants (ADR 0028). The
 * kernels round once with `f16()` right before writing to the shared tile, so "the f32 variant with
 * its inputs rounded by this function" and the f16 variant should **match bit for bit** — this
 * function is the oracle that checks that equivalence (tests/gpu_f16_compute_test.ts).
 *
 * MUST: use `Math.f16round` as is, as the f16 counterpart of `Math.fround` (a Web standard API —
 * ADR 0002). Rewriting it with bit operations gets one of subnormals (steps of 2^-24), overflow
 * (65520 and above become Inf) or round-half-to-even wrong, and real data rarely hits them, so it
 * goes unnoticed.
 * NOTE: the other direction ({@link f16BitsToF32}) is **exact**, so it has no notion of rounding.
 * Rounding happens in this one direction only.
 */
export const roundToF16 = (value: number): number => Math.f16round(value);

/**
 * Aligned bytes for uploading an f16 payload to the GPU as is (ADR 0018).
 *
 * An odd element count is not a multiple of 4 bytes and fails `queue.writeBuffer` validation. The
 * payload is aligned by **zero-filling the last 2 bytes** — kernel reads stop at the element count,
 * so values are not affected (when the last element is in the low half of a word, the high half is
 * never read).
 *
 * When already aligned, the original view is returned without a copy (weights are GB-sized, and an
 * unconditional copy shows up in measurements).
 */
export const alignF16Payload = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  if (bytes.byteLength % 4 === 0) return bytes;
  const padded = new Uint8Array(bytes.byteLength + F16_BYTES);
  padded.set(bytes);
  return padded;
};
