// テスト用の safetensors 組み立て（rope 素表の読み手を叩くためだけの最小ヘルパ）。
// 期待する「正しい形」を 1 箇所に置き、異常系は各テストがそこから 1 点だけ壊す。

export const f32Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> => {
  const array = Float32Array.from(values);
  return new Uint8Array(array.buffer);
};

export type TensorSpec = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly data: Uint8Array;
};

/** 隙間なく詰めた整合ファイルを作る（データ節は 8 バイト境界に置く — safetensors の慣例）。 */
export const buildSafetensors = (tensors: readonly TensorSpec[]): ArrayBuffer => {
  const header: Record<string, unknown> = {};
  let offset = 0;
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + tensor.data.length],
    };
    offset += tensor.data.length;
  }
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const headerLength = headerBytes.length + ((8 - (headerBytes.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength + offset);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(headerBytes, 8);
  bytes.fill(0x20, 8 + headerBytes.length, 8 + headerLength);
  let cursor = 8 + headerLength;
  for (const tensor of tensors) {
    bytes.set(tensor.data, cursor);
    cursor += tensor.data.length;
  }
  return buffer;
};
