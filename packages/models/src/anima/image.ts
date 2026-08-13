/**
 * VAE decode 出力（planar `[1,3,H,W]`・値域 `[-1,1]`）→ インターリーブ RGBA 8bit。
 *
 * 規則は `VaeImageProcessor.postprocess` と同一: `v = clamp(x/2 + 0.5, 0, 1)` を
 * `round(v·255)` で 8bit にする。
 *
 * NOTE: **clamp を二重に掛けているのは意図的**（無害）。`AutoencoderKLQwenImage._decode` の
 * `clamp(-1,1)` は IR 側に焼き込まれているので実行経路では飽和済みだが、この関数は
 * 「[-1,1] の平面画像」を入口の契約として持つ。
 * MUST: `[0,1]` 変換の**前**に clamp を移さない（postprocess 由来の clamp と取り違えると
 * 負値が 0 に潰れてから +0.5 され、暗部が中間灰に張り付く）。
 *
 * DOM に依存しない（`ImageData` を作らずバイト列を返す）ので、そのままブラウザの
 * `putImageData` にも PNG 書き出しにも渡せる。
 */
export const imageToRgba = (
  image: Float32Array,
  width: number,
  height: number,
): Uint8ClampedArray<ArrayBuffer> => {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`画像サイズ ${width}×${height} が正の整数でない`);
  }
  const plane = width * height;
  if (image.length !== plane * 3) {
    throw new Error(`画像の要素数 ${image.length} が 3×${width}×${height} と違う`);
  }
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let index = 0; index < plane; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const raw = image[channel * plane + index];
      // MUST: 非有限値を黙って返さない。RGBA 量子化（`Uint8ClampedArray` 代入）は NaN を 0、
      // ±Inf を 0/255 という「正常な画素」へ変換してしまうので、ここが最後の検査点。
      if (!Number.isFinite(raw)) {
        const x = index % width;
        const y = Math.floor(index / width);
        throw new Error(`画素 (x=${x}, y=${y}) の channel ${channel} が非有限値`);
      }
      const value = Math.min(1, Math.max(0, raw / 2 + 0.5));
      rgba[index * 4 + channel] = Math.round(value * 255);
    }
    rgba[index * 4 + 3] = 255;
  }
  return rgba;
};
