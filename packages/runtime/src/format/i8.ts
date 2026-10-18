/**
 * CPU-side decode of i8 storage (per-channel symmetric int8) (for the ineligible slots of ADR 0019 /
 * the host mirror of the GPU-side `unpack4xI8` + scale multiplication).
 *
 * MUST: decoding lives in this one place only. If the load path and the test's expected values
 * interpret i8 separately, the test becomes "both sides wrong in the same way" and a mix-up of the
 * sign (−128 to 127) or of the scale broadcast goes unnoticed.
 *
 * Why it is bit-identical: the decoded value is `f32(q) · s` with **one f32 rounding**, the same
 * operation as the GPU-side `f32(unpack4xI8(...)[…]) * scale`. `q` is an integer, so it is exact in
 * f32, and the product is exact in f64 (24 + 24 bits), so the single `Math.fround` rounding equals
 * the correct rounding of an f32 multiplication. The eligible path (GPU-resident) and the
 * ineligible path (CPU decode) therefore produce **bit-identical values**.
 */

/**
 * Element count (the format layer does not depend on src/ops.ts — keeps this one line so the layer
 * dependencies stay one-directional).
 */
const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/**
 * The decode / alignment contract was broken (mismatched element count, scale shape that cannot
 * broadcast).
 */
export class I8Error extends Error {
  override readonly name = "I8Error";
}

/**
 * Aligned bytes for uploading an i8 payload to the GPU as is (ADR 0019).
 *
 * An element count that is not a multiple of 4 fails `queue.writeBuffer` validation. The payload is
 * aligned by **zero-filling the tail up to a 4-byte boundary** — kernel reads stop at the element
 * count, so values are not affected (the unused lanes of the last word are never read).
 *
 * When already aligned, the original view is returned without a copy (weights are GB-sized, and an
 * unconditional copy shows up in measurements).
 */
export const alignI8Payload = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const remainder = bytes.byteLength % 4;
  if (remainder === 0) return bytes;
  const padded = new Uint8Array(bytes.byteLength + (4 - remainder));
  padded.set(bytes);
  return padded;
};

/**
 * Strides mapping a flat weight index → a flat scale index, built from the scale's keepdim
 * broadcast shape. An axis of length 1 has stride 0 (the same scale is handed out along it).
 *
 * MUST: the rank and every axis are checked here (do not rely on the caller's checks). Silently
 * letting through a scale whose rank differs from `weight` applies misindexed scales to every
 * element, a silent wrong value.
 */
const scaleStrides = (
  shape: readonly number[],
  scaleShape: readonly number[],
): readonly number[] => {
  if (scaleShape.length !== shape.length) {
    throw new I8Error(
      `scale の rank ${scaleShape.length} が重み [${shape.join(",")}] と違う（keepdim 形が要る）`,
    );
  }
  const strides = new Array<number>(shape.length).fill(0);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    const extent = scaleShape[axis];
    if (extent !== 1 && extent !== shape[axis]) {
      throw new I8Error(
        `scale [${scaleShape.join(",")}] の軸 ${axis} が重み [${shape.join(
          ",",
        )}] へ broadcast できない`,
      );
    }
    strides[axis] = extent === 1 ? 0 : stride;
    stride *= extent;
  }
  return strides;
};

/**
 * Decodes i8 bytes (the raw bytes of a container block = signed 8-bit) into f32 with a per-channel
 * scale.
 *
 * `scaleShape` is a keepdim broadcast shape of the same rank as the weight (a shape like
 * `[Cout,1,1]`).
 */
export const decodeI8 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
): Float32Array<ArrayBuffer> => {
  const count = numel(shape);
  if (bytes.byteLength !== count) {
    throw new I8Error(
      `i8 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う`,
    );
  }
  if (scale.length !== numel(scaleShape)) {
    throw new I8Error(
      `scale の要素数 ${scale.length} が shape [${scaleShape.join(",")}] と合わない`,
    );
  }
  const strides = scaleStrides(shape, scaleShape);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  // Coordinates carry over from the last axis (folds the multi-dimensional nested loops into one
  // scan).
  const coord = new Array<number>(shape.length).fill(0);
  let scaleIndex = 0;
  for (let i = 0; i < count; i += 1) {
    // MUST: round the product only once (Math.fround). The GPU side does one f32 multiplication;
    // storing the f64 value would end up as the same value through the Float32Array assignment,
    // but "one rounding" is written here as the intent (it stops holding if the multiplication is
    // moved outside the reduction).
    out[i] = Math.fround(view.getInt8(i) * scale[scaleIndex]);
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
      coord[axis] += 1;
      scaleIndex += strides[axis];
      if (coord[axis] < shape[axis]) break;
      scaleIndex -= strides[axis] * coord[axis];
      coord[axis] = 0;
    }
  }
  return out;
};
