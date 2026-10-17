/**
 * bfloat16 → f32 の展開（bf16 格納の適格外スロット用 / GPU 側の `bitcast<f32>(bits << 16)` の
 * ホスト鏡像 — ADR 0131）。
 *
 * bf16 は **f32 の上位 16 bit そのもの**（符号 1・指数 8・仮数 7）。f32 への展開は下位 16 bit を
 * 0 で埋めるだけで、f16 と違って指数域の変換も subnormal の読み替えも無い — 常に厳密で、GPU 側の
 * bitcast とビット単位で一致する MUST（実 GPU テストが全 65,536 パターンで固定する。NaN は
 * 「双方 NaN」、f32 の subnormal に当たる 2^-126 未満のパターンは GPU が 0 へ flush してよい —
 * WGSL は subnormal の保持を保証しない）。
 *
 * MUST: 展開はこの 1 箇所だけに置く（f16 の {@link "./f16.ts"} と同じ理由 — ロード経路と
 * テストの期待値が別々に解釈すると、同じ誤りが両側に乗って恒真化する）。
 */

/** bf16 の 1 要素あたりのバイト数。 */
const BF16_BYTES = 2;

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/** bf16 の 16bit パターン → f32 の数値（下位 16bit だけを見る。NaN のペイロードは保存しない）。 */
export const bf16BitsToF32 = (bits: number): number => {
  U32[0] = (bits & 0xffff) << 16;
  return F32[0];
};

/**
 * bf16 のバイト列（リトルエンディアン）を f32 配列へ展開する。長さが 2 の倍数であることは
 * 合流層の宣言検査が済ませている。
 *
 * MUST: 読みは DataView で 1 要素ずつ（`Uint16Array` の view は 2 バイト整列を要求するため、
 * ファイル中の位置によっては張れない — f16 と同じ）。
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
