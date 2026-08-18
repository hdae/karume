// i4 格納テストの組み立てヘルパ（ホスト側の f32 → K 方向 group 対称 int4 の符号化 + pack）。
//
// 符号化は**テスト専用**（本番はエクスポータ = Python 側 `karume/emit.py: pack_int4` の担当で、
// ランタイムは decode しか持たない）。pack 順「要素 2i = 下位 nibble / 2i+1 = 上位・
// 格納値 u = q + 8」はエクスポータの仕様をここで**書き下す** — 実装（format/i4.ts）から
// 引くと往復が恒真化する（helpers/i8.ts と同じ規律）。
//   bytes  = packed 4bit の生バイト列（GPU / ロード経路に渡す）
//   scale  = group ごとの scale（group 形の F32 テンソルとして配布形に載せる）
//   values = そのバイト列を**本番の decodeI4 で**戻した f32 列（CPU 参照が使う期待値）

import { decodeI4 } from "../../src/format/i4.ts";
import { F32_TINY } from "./i8.ts";

export type QuantizedI4 = {
  /** safetensors に載せる I4 の生バイト列（numel / 2 バイト）。 */
  readonly bytes: Uint8Array<ArrayBuffer>;
  /** group ごとの scale（平坦 = 行 × group 順）。 */
  readonly scale: Float32Array<ArrayBuffer>;
  /** scale テンソルの group 形 shape（重みと同 rank・最終次元 = group 数）。 */
  readonly scaleShape: readonly number[];
  /** 量子化後の値（fake-quant — CPU 参照はこちらを重みとして使う）。 */
  readonly values: Float32Array<ArrayBuffer>;
};

/**
 * K 方向 group symmetric int4 の fake-quant + pack（ADR 0069 決定 3 / 4）。
 *
 * `scale = max(amax_group / 7, tiny)` / `q = round(w / scale)` を **±7 に閉じる**。
 * 丸めは half-away-from-zero（i8 ヘルパと同じ — 期待値は decodeI4 から作るので
 * ランタイムの検証には影響しない）。
 */
export const quantizeI4 = (
  values: ArrayLike<number>,
  shape: readonly number[],
  groupSize: number,
): QuantizedI4 => {
  const lastDim = shape[shape.length - 1];
  if (lastDim % groupSize !== 0) {
    throw new Error(`quantizeI4: 量子化軸 ${lastDim} が group_size ${groupSize} で割り切れない`);
  }
  const groups = lastDim / groupSize;
  const rows = values.length / lastDim;

  const scale = new Float32Array(rows * groups);
  for (let i = 0; i < values.length; i += 1) {
    const row = Math.floor(i / lastDim);
    const group = Math.floor((i % lastDim) / groupSize);
    const slot = row * groups + group;
    scale[slot] = Math.max(scale[slot], Math.abs(values[i]));
  }
  for (let slot = 0; slot < scale.length; slot += 1) {
    scale[slot] = Math.max(Math.fround(scale[slot] / 7), F32_TINY);
  }

  // pack: 要素 2i = 下位 nibble / 2i+1 = 上位・u = q + 8（正本 = emit.py の pack_int4）
  const bytes = new Uint8Array(values.length / 2);
  for (let i = 0; i < values.length; i += 1) {
    const row = Math.floor(i / lastDim);
    const group = Math.floor((i % lastDim) / groupSize);
    const ratio = values[i] / scale[row * groups + group];
    const rounded = Math.sign(ratio) * Math.round(Math.abs(ratio));
    const u = Math.max(-7, Math.min(7, rounded)) + 8;
    bytes[i >> 1] |= (i & 1) === 1 ? u << 4 : u;
  }
  const scaleShape = [...shape.slice(0, -1), groups];
  return { bytes, scale, scaleShape, values: decodeI4(bytes, shape, scale, scaleShape, groupSize) };
};
