/**
 * i8（per-channel symmetric int8）格納の CPU 側展開（ADR 0019 の適格外スロット用 /
 * GPU 側 `unpack4xI8` + scale 乗算のホスト鏡像）。
 *
 * MUST: 展開はこの 1 箇所だけに置く。ロード経路とテストの期待値が別々に i8 を解釈すると、
 * 「両方同じ間違い方をしている」テストになって符号（−128〜127）や scale の broadcast の
 * 取り違えが沈黙する。
 *
 * ビット一致の根拠: 復元値は `f32(q) · s` の **f32 丸め 1 回**で、GPU 側の
 * `f32(unpack4xI8(...)[…]) * scale` と同じ演算。`q` は整数なので f32 で厳密、積は f64 では
 * 厳密（24 + 24 ビット）なので `Math.fround` の 1 回丸めが f32 乗算の正しい丸めと一致する。
 * したがって適格経路（GPU 常駐）と適格外経路（CPU 展開）は**ビット単位で同じ値**を出す。
 */

/**
 * 要素数（format 層は src/ops.ts に依存しない — 層の向きを 1 本に保つため、この 1 行だけ持つ）。
 */
const numel = (shape: readonly number[]): number => shape.reduce((count, dim) => count * dim, 1);

/** 展開・整列で契約が破れた（要素数の不一致・scale の broadcast 不能形）。 */
export class I8Error extends Error {
  override readonly name = "I8Error";
}

/**
 * i8 ペイロードを GPU へ生のまま上げるときの整列済みバイト列（ADR 0019）。
 *
 * 要素数が 4 の倍数でないと `queue.writeBuffer` の検証で落ちる。**末尾を 4 バイト境界まで
 * ゼロ詰め**して整列させる — カーネルの読み出しは要素数で打ち切るので値には影響しない
 * （最終語の未使用レーンは読まれない）。
 *
 * 整列済みならコピーせず元の view を返す（重みは GB 級で、無条件コピーは実測に響く）。
 */
export const alignI8Payload = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  const remainder = bytes.byteLength % 4;
  if (remainder === 0) return bytes;
  const padded = new Uint8Array(bytes.byteLength + (4 - remainder));
  padded.set(bytes);
  return padded;
};

/**
 * scale の keepdim broadcast 形から、重みの平坦添字 → scale の平坦添字を作る stride。
 * 長さ 1 の軸は stride 0（その軸では同じ scale を配る）。
 *
 * MUST: rank と各軸の一致はここで見る（呼び出し側の検査を当てにしない）。`weight` と
 * 別 rank の scale を黙って通すと、添字がずれた scale が全要素に掛かる沈黙誤値になる。
 */
const scaleStrides = (
  shape: readonly number[],
  scaleShape: readonly number[],
): readonly number[] => {
  if (scaleShape.length !== shape.length) {
    throw new I8Error(
      `scale の rank ${scaleShape.length} が重み [${shape.join(",")}] と違う（keepdim 形が要る）`,
    );
  }
  const strides = new Array<number>(shape.length).fill(0);
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
    const extent = scaleShape[axis];
    if (extent !== 1 && extent !== shape[axis]) {
      throw new I8Error(
        `scale [${scaleShape.join(",")}] の軸 ${axis} が重み [${
          shape.join(",")
        }] へ broadcast できない`,
      );
    }
    strides[axis] = extent === 1 ? 0 : stride;
    stride *= extent;
  }
  return strides;
};

/**
 * i8 のバイト列（safetensors の生バイト = 符号付き 8bit）を per-channel scale で f32 へ展開する。
 *
 * `scaleShape` は重みと同 rank の keepdim broadcast 形（`[Cout,1,1]` のような形）。
 */
export const decodeI8 = (
  bytes: Uint8Array<ArrayBuffer>,
  shape: readonly number[],
  scale: Float32Array<ArrayBuffer>,
  scaleShape: readonly number[],
): Float32Array<ArrayBuffer> => {
  const count = numel(shape);
  if (bytes.byteLength !== count) {
    throw new I8Error(
      `i8 ペイロード ${bytes.byteLength} バイトが shape [${shape.join(",")}] と違う`,
    );
  }
  if (scale.length !== numel(scaleShape)) {
    throw new I8Error(
      `scale の要素数 ${scale.length} が shape [${scaleShape.join(",")}] と合わない`,
    );
  }
  const strides = scaleStrides(shape, scaleShape);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  // 座標は最終軸から桁上がりさせる（多次元の入れ子ループを 1 本の走査へ畳む）。
  const coord = new Array<number>(shape.length).fill(0);
  let scaleIndex = 0;
  for (let i = 0; i < count; i += 1) {
    // MUST: 積の丸めは 1 回だけ（Math.fround）。GPU 側は f32 の乗算 1 回なので、
    // f64 のまま格納すると Float32Array への代入で結果的に同じ値になるが、意図として
    // 「1 回丸め」をここに書いておく（縮約の外で掛ける形へ変えると成立しなくなる）。
    out[i] = Math.fround(view.getInt8(i) * scale[scaleIndex]);
    for (let axis = shape.length - 1; axis >= 0; axis -= 1) {
      coord[axis] += 1;
      scaleIndex += strides[axis];
      if (coord[axis] < shape[axis]) break;
      scaleIndex -= strides[axis] * coord[axis];
      coord[axis] = 0;
    }
  }
  return out;
};
