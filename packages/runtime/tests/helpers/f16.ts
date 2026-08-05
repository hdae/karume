// f16 格納テストの組み立てヘルパ（ホスト側の f32 → f16 符号化）。
//
// 符号化は**テスト専用**（本番はエクスポータ = Python 側の担当で、ランタイムは decode しか
// 持たない）。ここは「f16 で表せる値だけを含む重み」を作るための道具で、ADR 0006 の
// fake-quant をホストで再現する役目を持つ:
//   bytes  = 量子化した格納バイト列（GPU / ロード経路に渡す）
//   values = そのバイト列を**本番の decodeF16 で**戻した f32 列（CPU 参照が使う期待値）
// MUST: 期待値は符号化器ではなく decodeF16 から作る。両側を符号化器で作ると、符号化器の
// 誤りが期待値にも同じだけ乗って恒真化する。

import { decodeF16 } from "../../src/format/f16.ts";

const F32 = new Float32Array(1);
const U32 = new Uint32Array(F32.buffer);

/**
 * f32 → f16 の 16bit パターン（round-to-nearest-even）。
 * オーバーフローは ±Inf、アンダーフローは subnormal を経て ±0 へ落ちる。
 */
export const f32ToF16Bits = (value: number): number => {
  F32[0] = value;
  const bits = U32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  // Inf / NaN は指数部を全 1 に。NaN は quiet ビットだけ残す（ペイロードは保存しない）
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa !== 0 ? 0x200 : 0);
  const shifted = exponent - 127 + 15;
  if (shifted >= 0x1f) return sign | 0x7c00;
  if (shifted <= 0) {
    // subnormal 域（暗黙の 1 を立ててから右シフトで詰める）。-11 以下は ±0
    if (shifted < -10) return sign;
    const full = mantissa | 0x800000;
    const shift = 14 - shifted;
    const truncated = full >>> shift;
    const remainder = full & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    const roundUp = remainder > halfway || (remainder === halfway && (truncated & 1) === 1);
    return sign | (truncated + (roundUp ? 1 : 0));
  }
  const truncated = (shifted << 10) | (mantissa >>> 13);
  const remainder = mantissa & 0x1fff;
  const roundUp = remainder > 0x1000 || (remainder === 0x1000 && (truncated & 1) === 1);
  // 桁上がりは仮数から指数へそのまま伝播する（0x7bff + 1 = 0x7c00 = Inf）
  return sign | (truncated + (roundUp ? 1 : 0));
};

/** f16 の 16bit パターン列 → リトルエンディアンのバイト列。 */
export const f16BytesFromBits = (bits: readonly number[]): Uint8Array<ArrayBuffer> => {
  const bytes = new Uint8Array(bits.length * 2);
  const view = new DataView(bytes.buffer);
  bits.forEach((pattern, index) => view.setUint16(index * 2, pattern & 0xffff, true));
  return bytes;
};

export type QuantizedWeight = {
  /** safetensors に載せる f16 の生バイト列。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** 量子化後の値（fake-quant — CPU 参照はこちらを重みとして使う）。 */
  readonly values: Float32Array<ArrayBuffer>;
};

/** f32 の並びを f16 へ丸め、格納バイト列と丸め後の値を組で返す。 */
export const quantizeF16 = (values: ArrayLike<number>): QuantizedWeight => {
  const bits = Array.from({ length: values.length }, (_, i) => f32ToF16Bits(values[i]));
  const bytes = f16BytesFromBits(bits);
  return { bytes, values: decodeF16(bytes) };
};
