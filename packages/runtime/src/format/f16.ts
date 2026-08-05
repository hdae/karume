/**
 * IEEE 754 binary16 → f32 の展開（ADR 0018 の適格外スロット用 / GPU 側 `unpack2x16float` の
 * ホスト鏡像）。
 *
 * MUST: 展開はこの 1 箇所だけに置く。ロード経路とテストの期待値が別々に f16 を解釈すると、
 * 「両方同じ間違い方をしている」テストになって subnormal や NaN の扱いの差が沈黙する。
 *
 * f16 → f32 は**常に厳密**（指数域も仮数も f32 に収まる）。したがって GPU 側の
 * `unpack2x16float` とはビット単位で一致する MUST で、実 GPU テストが全 65,536 パターンで
 * これを固定する（NaN のペイロード伝播だけは WGSL の保証外なので「双方 NaN」で見る）。
 */

/** f16 の 1 要素あたりのバイト数。 */
export const F16_BYTES = 2;

/**
 * f16 の 16bit パターン → f32 の数値（下位 16bit だけを見る）。
 *
 * - 指数 0: ±0（仮数 0）と subnormal。`mantissa · 2^-24` は f32 で厳密。
 * - 指数 31: ±Inf（仮数 0）と NaN。**NaN のペイロードは保存しない**（WGSL 側も保証しない）。
 * - それ以外: `(1 + mantissa/1024) · 2^(exp-15)`。
 */
export const f16BitsToF32 = (bits: number): number => {
  const pattern = bits & 0xffff;
  const sign = (pattern & 0x8000) !== 0 ? -1 : 1;
  const exponent = (pattern >> 10) & 0x1f;
  const mantissa = pattern & 0x3ff;
  if (exponent === 0) {
    // 仮数 0 なら符号付きゼロ（-1 * 0 = -0 — Object.is で区別できる形を保つ）
    return sign * mantissa * 2 ** -24;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

/**
 * f16 のバイト列（safetensors の生バイト — リトルエンディアン）を f32 配列へ展開する。
 * 長さが 2 の倍数であることは safetensors の宣言検査が済ませている。
 *
 * MUST: 読みは DataView で 1 要素ずつ。`Uint16Array` の view は byteOffset の 2 バイト整列を
 * 要求するため、ファイル中の位置によっては張れない。
 */
export const decodeF16 = (bytes: Uint8Array<ArrayBuffer>): Float32Array<ArrayBuffer> => {
  const count = bytes.byteLength / F16_BYTES;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    out[i] = f16BitsToF32(view.getUint16(i * F16_BYTES, true));
  }
  return out;
};

/**
 * f32 → f16 → f32 の**丸め**（IEEE 754 binary16 の round-to-nearest-even）。
 *
 * f16 **計算**変種（ADR 0028）の数値契約の正本。カーネルは共有タイルへ書く直前に `f16()` で
 * 1 度だけ丸めるので、「入力をこの関数で丸めた f32 変種」と f16 変種の出力は**ビット単位で
 * 一致する**はず — その等価性を検証するオラクルがこの関数（tests/gpu_f16_compute_test.ts）。
 *
 * MUST: `Math.fround` の f16 版として `Math.f16round` をそのまま使う（Web 標準 API — ADR
 * 0002）。ビット演算で書き直すと subnormal（2^-24 刻み）・オーバフロー（65520 以上が Inf）・
 * 同点の偶数丸めのどれかを取り違え、しかも実データでは滅多に踏まないまま沈黙する。
 * NOTE: 逆方向（{@link f16BitsToF32}）は**厳密**なので丸めの概念が無い。丸めが起きるのは
 * こちら 1 方向だけ。
 */
export const roundToF16 = (value: number): number => Math.f16round(value);

/**
 * f16 ペイロードを GPU へ生のまま上げるときの整列済みバイト列（ADR 0018）。
 *
 * 要素数が奇数だと 4 バイトの倍数にならず `queue.writeBuffer` の検証で落ちる。**末尾 2 バイトを
 * ゼロ詰め**して整列させる — カーネルの読み出しは要素数で打ち切るので値には影響しない
 * （最終要素が語の下位半分に来るとき、上位半分は読まれない）。
 *
 * 整列済みならコピーせず元の view を返す（重みは GB 級で、無条件コピーは実測に響く）。
 */
export const alignF16Payload = (bytes: Uint8Array<ArrayBuffer>): Uint8Array<ArrayBuffer> => {
  if (bytes.byteLength % 4 === 0) return bytes;
  const padded = new Uint8Array(bytes.byteLength + F16_BYTES);
  padded.set(bytes);
  return padded;
};
