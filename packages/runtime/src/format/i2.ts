/** 行ごとの scale を持つ packed INT2 の CPU 展開（ADR 0097）。 */
export class I2Error extends Error {
  override readonly name = "I2Error";
}

/** 論理形 [rows, width]。各行を u32 整列させ、末尾の暗黙パディングを禁止する。 */
export const isI2Shape = (shape: readonly (number | string)[]): boolean =>
  shape.length === 2 && shape.every((dim) =>
    typeof dim === "number" && Number.isSafeInteger(dim) && dim > 0
  ) && Number(shape[1]) % 16 === 0;

/** 下位 2bit から順に u=q+2。scale の乗算ごとに f32 へ丸める。 */
export const decodeI2 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
): Float32Array<ArrayBuffer> => {
  if (!isI2Shape(shape)) {
    throw new I2Error(`i2 の shape [${shape.join(",")}] は正の rank 2・行長は16の倍数が必要`);
  }
  const [rows, width] = shape;
  const count = rows * width;
  if (bytes.byteLength !== count / 4) {
    throw new I2Error(
      `i2 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う`,
    );
  }
  if (
    scaleShape.length !== 2 || scaleShape[0] !== rows || scaleShape[1] !== 1 ||
    scale.length !== rows
  ) {
    throw new I2Error(`i2 の scale [${scaleShape.join(",")}] は [${rows},1] が必要`);
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
