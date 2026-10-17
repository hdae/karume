// bf16 格納テストの組み立てヘルパ（ホスト側の f32 → bf16 符号化）。
//
// 符号化は**テスト専用**（本番はエクスポータ = Python 側の担当で、ランタイムは decode しか
// 持たない）。f16 の `helpers/f16.ts` と同じ役目: ADR 0006 の fake-quant をホストで再現し、
//   bytes  = 量子化した格納バイト列（GPU / ロード経路に渡す）
//   values = そのバイト列を**本番の decodeBf16 で**戻した f32 列（CPU 参照が使う期待値）
// を組で返す。MUST: 期待値は符号化器ではなく decodeBf16 から作る（両側を符号化器で作ると、
// 符号化器の誤りが期待値にも同じだけ乗って恒真化する）。

import { decodeBf16 } from "../../src/format/bf16.ts";

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/**
 * f32 → bf16 の 16bit パターン（round-to-nearest-even — torch の `Tensor.to(bfloat16)` と同じ
 * 丸め）。bf16 は f32 の上位 16 bit なので、下位 16 bit を丸め込むだけでよい（指数域は同じで、
 * オーバーフローは起きない）。NaN は quiet ビットを立てて返す（ペイロードは保存しない）。
 */
export const f32ToBf16Bits = (value: number): number => {
  F32[0] = value;
  const bits = U32[0];
  if (Number.isNaN(value)) return ((bits >>> 16) | 0x0040) & 0xffff;
  const lower = bits & 0xffff;
  const upper = bits >>> 16;
  const roundUp = lower > 0x8000 || (lower === 0x8000 && (upper & 1) === 1);
  return (upper + (roundUp ? 1 : 0)) & 0xffff;
};

/** bf16 の 16bit パターン列 → リトルエンディアンのバイト列。 */
export const bf16BytesFromBits = (bits: readonly number[]): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(bits.length * 2);
  const view = new DataView(bytes.buffer);
  bits.forEach((pattern, index) => view.setUint16(index * 2, pattern & 0xffff, true));
  return bytes;
};

export type QuantizedWeight = {
  /** 容器に載せる bf16 の生バイト列。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** 量子化後の値（fake-quant — CPU 参照はこちらを重みとして使う）。 */
  readonly values: Float32Array<ArrayBuffer>;
};

/** f32 の並びを bf16 へ丸め、格納バイト列と丸め後の値を組で返す。 */
export const quantizeBf16 = (values: ArrayLike<number>): QuantizedWeight => {
  const bits = Array.from({ length: values.length }, (_, i) => f32ToBf16Bits(values[i]));
  const bytes = bf16BytesFromBits(bits);
  return { bytes, values: decodeBf16(bytes) };
};
