/**
 * 生成した動画（`[3, F, H, W]`・値域 `[-1, 1]`）の 1 フレーム → インターリーブ RGBA 8bit。
 *
 * 規則は anima の `imageToRgba`（`VaeImageProcessor.postprocess` と同一）と同じ:
 * `v = clamp(x/2 + 0.5, 0, 1)` を `round(v·255)` で 8bit にする（ADR 0118 決定 8 の「uint8 化は Anima の
 * 画像と同じ規則」）。家族をまたいで import しないので写しを持つ（片方の変更がもう片方の sha256 の行を
 * 黙って動かさないように）。
 *
 * MUST: `[0,1]` 変換の**前**に clamp を移さない（負値が 0 に潰れてから +0.5 され、暗部が中間灰に
 * 張り付く）。
 */

import type { GeneratedVideo } from "./pipeline.ts";

export const wanFrameToRgba = (
  video: GeneratedVideo,
  frame: number,
): Uint8ClampedArray<ArrayBuffer> => {
  const { frames, width, height, data } = video;
  if (!Number.isInteger(frame) || frame < 0 || frame >= frames) {
    throw new RangeError(`フレーム ${frame} が [0, ${frames}) の外`);
  }
  const plane = width * height;
  if (data.length !== 3 * frames * plane) {
    throw new Error(`動画の要素数 ${data.length} が [3, ${frames}, ${height}, ${width}] と違う`);
  }
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let channel = 0; channel < 3; channel += 1) {
    const base = (channel * frames + frame) * plane;
    for (let index = 0; index < plane; index += 1) {
      const raw = data[base + index];
      // MUST: 非有限値を黙って画素にしない（`Uint8ClampedArray` は NaN を 0 にする）。
      if (!Number.isFinite(raw)) {
        throw new Error(
          `フレーム ${frame} の画素 (x=${index % width}, y=${Math.floor(index / width)}) の ` +
            `channel ${channel} が非有限値`,
        );
      }
      rgba[index * 4 + channel] = Math.round(Math.min(1, Math.max(0, raw / 2 + 0.5)) * 255);
    }
  }
  for (let index = 0; index < plane; index += 1) rgba[index * 4 + 3] = 255;
  return rgba;
};
