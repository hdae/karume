/**
 * bfloat16 → f32 decode (for ineligible slots of bf16 storage / the host mirror of the GPU-side
 * `bitcast<f32>(bits << 16)` — ADR 0131).
 *
 * bf16 is **exactly the upper 16 bits of an f32** (sign 1, exponent 8, mantissa 7). Decoding to f32
 * only fills the lower 16 bits with 0; unlike f16 there is no exponent-range conversion and no
 * subnormal reinterpretation — it is always exact and MUST match the GPU-side bitcast bit for bit
 * (a real-GPU test pins all 65,536 patterns. NaN means "NaN on both sides"; for patterns below
 * 2^-126, which are f32 subnormals, the GPU may flush to 0 — WGSL does not guarantee that
 * subnormals are preserved).
 *
 * MUST: decoding lives in this one place only (the same reason as f16's {@link "./f16.ts"} — if the
 * load path and the test's expected values interpret it separately, the same error lands on both
 * sides and the test becomes tautological).
 */

/** Bytes per bf16 element. */
const BF16_BYTES = 2;

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/** bf16 16-bit pattern → f32 number (only the low 16 bits are read; NaN payloads are not kept). */
export const bf16BitsToF32 = (bits: number): number => {
  U32[0] = (bits & 0xffff) << 16;
  return F32[0];
};

/**
 * Decodes bf16 bytes (little endian) into an f32 array. The merge layer's declaration check has
 * already ensured the length is a multiple of 2.
 *
 * MUST: read one element at a time through a DataView (a `Uint16Array` view requires 2-byte
 * alignment, so it cannot be created at some positions in the file — the same as f16).
 */
export const decodeBf16 = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const count = bytes.byteLength / BF16_BYTES;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = bf16BitsToF32(view.getUint16(i * BF16_BYTES, true));
  }
  return out;
};
