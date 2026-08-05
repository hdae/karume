// i8 格納テストの組み立てヘルパ（ホスト側の f32 → per-channel int8 符号化）。
//
// 符号化は**テスト専用**（本番はエクスポータ = Python 側の担当で、ランタイムは decode しか
// 持たない）。ここは「i8 + per-channel scale で厳密に表せる重み」を作るための道具で、
// ADR 0006 の fake-quant をホストで再現する役目を持つ:
//   bytes  = 量子化した格納バイト列（GPU / ロード経路に渡す）
//   scale  = チャネルごとの scale（keepdim 形の F32 テンソルとして配布形に載せる）
//   values = そのバイト列を**本番の decodeI8 で**戻した f32 列（CPU 参照が使う期待値）
// MUST: 期待値は符号化器ではなく decodeI8 から作る。両側を符号化器で作ると、符号化器の
// 誤りが期待値にも同じだけ乗って恒真化する。

import { decodeI8 } from "../../src/format/i8.ts";

/** f32 の最小 normal（`torch.finfo(float32).tiny` — scale の下限 clamp に使う）。 */
export const F32_TINY = 1.1754943508222875e-38;

/** 符号付き 8bit の並び → 生バイト列（下位 8bit だけを見る）。 */
export const i8BytesFrom = (values: readonly number[]): Uint8Array<ArrayBuffer> =>
  Uint8Array.from(values, (value) => value & 0xff);

export type QuantizedI8 = {
  /** safetensors に載せる I8 の生バイト列。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** チャネルごとの scale（平坦 = チャネル順）。 */
  readonly scale: Float32Array<ArrayBuffer>;
  /** scale テンソルの keepdim 形 shape。 */
  readonly scaleShape: readonly number[];
  /** 量子化後の値（fake-quant — CPU 参照はこちらを重みとして使う）。 */
  readonly values: Float32Array<ArrayBuffer>;
};

/**
 * per-channel symmetric int8 の fake-quant（ADR 0019）。
 *
 * `scale = max(amax / 127, tiny)` / `q = round(w / scale)` を **±127 に閉じる**
 * （−128 を使わないので最大絶対値要素が厳密復元される）。丸めは 0 から遠い側への
 * half-away-from-zero で、torch 側（half-to-even）と厳密には違うが、**期待値は decodeI8 から
 * 作る**のでランタイムの検証には影響しない（本番の符号化器はエクスポータ側）。
 */
export const quantizeI8 = (
  values: ArrayLike<number>,
  shape: readonly number[],
  axis: number,
): QuantizedI8 => {
  const channels = shape[axis];
  const inner = shape.slice(axis + 1).reduce((count, dim) => count * dim, 1);
  const channelOf = (index: number): number => Math.floor(index / inner) % channels;

  const amax = new Float32Array(channels);
  for (let i = 0; i < values.length; i += 1) {
    const channel = channelOf(i);
    amax[channel] = Math.max(amax[channel], Math.abs(values[i]));
  }
  const scale = new Float32Array(channels);
  for (let c = 0; c < channels; c += 1) scale[c] = Math.max(Math.fround(amax[c] / 127), F32_TINY);

  const quantized: number[] = [];
  for (let i = 0; i < values.length; i += 1) {
    const ratio = values[i] / scale[channelOf(i)];
    const rounded = Math.sign(ratio) * Math.round(Math.abs(ratio));
    quantized.push(Math.max(-127, Math.min(127, rounded)));
  }
  const bytes = i8BytesFrom(quantized);
  const scaleShape = shape.map((_, index) => (index === axis ? channels : 1));
  return { bytes, scale, scaleShape, values: decodeI8(bytes, shape, scale, scaleShape) };
};
