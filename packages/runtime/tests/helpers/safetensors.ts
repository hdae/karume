// safetensors リーダ（`src/format/safetensors.ts`）のテスト用に、検査済みの / わざと壊した
// safetensors バイナリを組むヘルパ。
//
// Karume の配布形は容器（`krm` / `krg`）に移ったので、ここが作るのは**付帯資産の読み手**が
// 相手にする素の safetensors だけである（ADR 0038 §2 の extras・golden io）。テストが期待する
// 「正しい形」を 1 箇所に置き、異常系は各テストがそこから 1 点だけ壊す。

export const f32Bytes = (values: ArrayLike<number>): Uint8Array<ArrayBuffer> => {
  const array = Float32Array.from(values);
  return new Uint8Array(array.buffer);
};

/** ヘッダ JSON をテキストのまま詰める（壊れた JSON を作るための入口）。 */
export const packSafetensorsRaw = (headerText: string, data: Uint8Array): ArrayBuffer => {
  const headerBytes = new TextEncoder().encode(headerText);
  // データ節を 8 バイト境界に置く（safetensors の慣例。整列検査の前提でもある）
  const headerLength = headerBytes.length + ((8 - (headerBytes.length % 8)) % 8);
  const buffer = new ArrayBuffer(8 + headerLength + data.length);
  const bytes = new Uint8Array(buffer);
  new DataView(buffer).setBigUint64(0, BigInt(headerLength), true);
  bytes.set(headerBytes, 8);
  bytes.fill(0x20, 8 + headerBytes.length, 8 + headerLength);
  bytes.set(data, 8 + headerLength);
  return buffer;
};

/** ヘッダ表をそのまま詰める（宣言と実データの不整合を作れるよう検査しない）。 */
export const packSafetensors = (header: Record<string, unknown>, data: Uint8Array): ArrayBuffer =>
  packSafetensorsRaw(JSON.stringify(header), data);

export type TensorSpec = {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly data: Uint8Array;
};

/** 隙間なく詰めた整合ファイルを作る。 */
export const buildSafetensors = (
  tensors: readonly TensorSpec[],
  metadata?: Record<string, string>,
): ArrayBuffer => {
  const header: Record<string, unknown> = {};
  if (metadata !== undefined) header["__metadata__"] = metadata;
  let offset = 0;
  for (const tensor of tensors) {
    header[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [offset, offset + tensor.data.length],
    };
    offset += tensor.data.length;
  }
  const data = new Uint8Array(offset);
  let cursor = 0;
  for (const tensor of tensors) {
    data.set(tensor.data, cursor);
    cursor += tensor.data.length;
  }
  return packSafetensors(header, data);
};
