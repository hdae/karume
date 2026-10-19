/** CPU decode of packed INT2 with one scale per row (ADR 0097). */
export class I2Error extends Error {
  override readonly name = "I2Error";
}

/** Logical shape [rows, width]. Every row is u32-aligned; implicit tail padding is forbidden. */
export const isI2Shape = (shape: readonly (number | string)[]): boolean =>
  shape.length === 2 &&
  shape.every((dim) => typeof dim === "number" && Number.isSafeInteger(dim) && dim > 0) &&
  Number(shape[1]) % 16 === 0;

/** From the low 2 bits upward, u=q+2. Each multiplication by the scale is rounded to f32. */
export const decodeI2 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
): Float32Array<ArrayBuffer> => {
  if (!isI2Shape(shape)) {
    throw new I2Error(
      `i2 shape [${shape.join(",")}] must be positive rank 2 with a row length that is a multiple of 16`,
    );
  }
  const [rows, width] = shape;
  const count = rows * width;
  if (bytes.byteLength !== count / 4) {
    throw new I2Error(
      `i2 payload of ${bytes.byteLength} bytes does not match shape [${shape.join(",")}]`,
    );
  }
  if (
    scaleShape.length !== 2 ||
    scaleShape[0] !== rows ||
    scaleShape[1] !== 1 ||
    scale.length !== rows
  ) {
    throw new I2Error(`i2 scale [${scaleShape.join(",")}] must be [${rows},1]`);
  }
  const result = new Float32Array(count);
  for (let row = 0; row < rows; row++) {
    const s = scale[row];
    for (let col = 0; col < width; col++) {
      const i = row * width + col;
      const q = ((bytes[i >> 2] >> ((i & 3) * 2)) & 3) - 2;
      result[i] = Math.fround(q * s);
    }
  }
  return result;
};
