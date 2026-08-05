// format 層テストの組み立てヘルパ（safetensors のバイナリ / 最小の正常系グラフ）。
// テストが期待する「正しい形」を 1 箇所に置き、異常系は各テストがそこから 1 点だけ壊す。

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

/** グラフ JSON の可変な写し（テストが 1 点だけ壊せるよう型は緩く持つ）。 */
export type GraphJson = {
  format: string;
  version: number;
  requires: { ops: string[] };
  symbols: string[];
  inputs: { name: string; dtype: string; shape: (number | string)[] }[];
  outputs: string[];
  initializers: Record<string, { tensor: string; storage: Record<string, unknown> }>;
  values: Record<string, { dtype: string; shape: (number | string)[] }>;
  nodes: { op: string; ins: string[]; outs: string[]; attrs: unknown }[];
};

/** 最小の正常系グラフ: y = x·w + b（x: T×4 → y: T×3）。 */
export const baseGraph = (): GraphJson => ({
  format: "karume-ir",
  version: 1,
  requires: { ops: ["matmul", "add"] },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", 4] }],
  outputs: ["y"],
  initializers: {
    w: { tensor: "enc.w", storage: { dtype: "f32" } },
    b: { tensor: "enc.b", storage: { dtype: "f32" } },
  },
  values: {
    w: { dtype: "f32", shape: [4, 3] },
    b: { dtype: "f32", shape: [3] },
    h: { dtype: "f32", shape: ["T", 3] },
    y: { dtype: "f32", shape: ["T", 3] },
  },
  nodes: [
    { op: "matmul", ins: ["x", "w"], outs: ["h"], attrs: {} },
    { op: "add", ins: ["h", "b"], outs: ["y"], attrs: {} },
  ],
});

/** baseGraph に対応する重みを積んだ正常系ファイル。 */
export const baseModelBuffer = (
  graph: GraphJson = baseGraph(),
  tensors: readonly TensorSpec[] = [
    { name: "enc.w", dtype: "F32", shape: [4, 3], data: f32Bytes(new Array(12).fill(0.5)) },
    { name: "enc.b", dtype: "F32", shape: [3], data: f32Bytes([1, 2, 3]) },
  ],
): ArrayBuffer => buildSafetensors(tensors, { karume_ir: JSON.stringify(graph) });
