/**
 * i4（K 方向 group symmetric packed 4bit）格納の CPU 側展開（ADR 0069 の適格外スロット用 /
 * GPU 側 `unpack4xU8` + マスク／シフト + scale 乗算のホスト鏡像）。
 *
 * MUST: 展開はこの 1 箇所だけに置く（format/i8.ts と同文の理由 — ロード経路とテストが別々に
 * i4 を解釈すると、pack 順〈要素 2i = 下位 nibble / 2i+1 = 上位・格納値 u = q + 8〉の
 * 取り違えが沈黙する）。pack 順の正本はエクスポータ `karume/emit.py: pack_int4`。
 *
 * ビット一致の根拠: 復元値は `f32(q) · s` の **f32 丸め 1 回**（`q ∈ [−7,7]` は f32 で厳密・
 * i8.ts と同じ論証）。エクスポータの逆変換門（`dequant(unpack(pack(w)))` のビット一致 —
 * ADR 0069 決定 4 ③）も同じ式で張られているため、TS / Python / WGSL の 3 実装が同じ数を出す。
 */

/** 要素数（format 層は src/ops.ts に依存しない — i8.ts と同じ理由でこの 1 行だけ持つ）。 */
const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** 展開で契約が破れた（バイト長・group 形 scale・group 長の不一致）。 */
export class I4Error extends Error {
  override readonly name = "I4Error";
}

/**
 * packed 4bit のバイト列を group 形 scale で f32 へ展開する。
 *
 * `scaleShape` は重みと**同 rank・最終次元だけ group 数**（`lastDim / groupSize`）の group 形
 * （ADR 0069 決定 3 — i8 の keepdim broadcast 形とは受理集合が交わらない別物。取り違えは
 * ここで fail loudly にする — 黙って broadcast 解釈すると group scale が 1 チャネル 1 値として
 * 配られる沈黙誤値になる）。
 */
export const decodeI4 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
  groupSize: number,
): Float32Array<ArrayBuffer> => {
  if (shape.length === 0) {
    throw new I4Error("i4 の重みに量子化軸が無い（rank 0）");
  }
  const count = numel(shape);
  if (count % 2 !== 0 || bytes.byteLength * 2 !== count) {
    throw new I4Error(
      `i4 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う` +
        `（numel / 2 = ${count / 2} バイトが要る）`,
    );
  }
  const lastDim = shape[shape.length - 1];
  if (!Number.isInteger(groupSize) || groupSize < 1 || lastDim % groupSize !== 0) {
    throw new I4Error(`group_size ${groupSize} が量子化軸 ${lastDim} を割り切らない`);
  }
  const groups = lastDim / groupSize;
  const expected = [...shape.slice(0, -1), groups];
  if (
    scaleShape.length !== shape.length ||
    scaleShape.some((dim, axis) => dim !== expected[axis])
  ) {
    throw new I4Error(
      `scale [${scaleShape.join(",")}] が重み [${shape.join(",")}] の group 形 [${
        expected.join(",")
      }]（group_size=${groupSize}）でない`,
    );
  }
  if (scale.length !== numel(scaleShape)) {
    throw new I4Error(
      `scale の要素数 ${scale.length} が shape [${scaleShape.join(",")}] と合わない`,
    );
  }
  const out = new Float32Array(count);
  const rows = lastDim === 0 ? 0 : count / lastDim;
  let i = 0;
  for (let row = 0; row < rows; row += 1) {
    const scaleBase = row * groups;
    for (let group = 0; group < groups; group += 1) {
      const s = scale[scaleBase + group];
      for (let element = 0; element < groupSize; element += 1, i += 1) {
        const byte = bytes[i >> 1];
        const u = (i & 1) === 1 ? byte >> 4 : byte & 0x0f;
        // MUST: 積の丸めは 1 回だけ（i8.ts と同文 — GPU 側は f32 の乗算 1 回）。offset 8 は
        // 「zero_point 省略時の既定」（ADR 0069 決定 3 の予約 2）で、pack 側 emit.py と対。
        out[i] = Math.fround((u - 8) * s);
      }
    }
  }
  return out;
};
