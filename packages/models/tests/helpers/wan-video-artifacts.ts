/**
 * Wan の生成動画の実物（全フレームの PNG と一覧図）を結果の席へ書く口（利用者の視認の素材 — ADR 0121 段 6）。
 *
 * NOTE: 中身は 2.1 の e2e（`e2e_wan_pipeline_test.ts`）の一覧図と PNG の書き出しの逐語の写し。2.1 の e2e 側の写しは
 * 触らない（2.1 のレーンの凍結 — ADR 0121 段 6 の裁定）ので、重複の解消は範囲外として別に扱う。
 */

import type { Results } from "../../../runtime/tests/helpers/results.ts";
import { encodePng } from "../../mod.ts";
import { type GeneratedVideo, wanFrameToRgba } from "../../wan.ts";

/** 全フレームを `<id>-frame-NN.png` として結果の席へ書く（NN は 0 始まり・2 桁以上）。 */
export const writeWanFrames = async (
  results: Pick<Results, "artifact">,
  id: string,
  video: GeneratedVideo,
): Promise<void> => {
  for (let frame = 0; frame < video.frames; frame += 1) {
    await Deno.writeFile(
      results.artifact(`${id}-frame-${String(frame).padStart(2, "0")}.png`),
      await encodePng(wanFrameToRgba(video, frame), video.width, video.height),
    );
  }
};

/** 一覧図（`rows × cols` のマス・各マスはフレームを `factor` 分の 1 に箱平均で縮めたもの）。 */
export const contactSheet = (
  video: GeneratedVideo,
  rows: number,
  cols: number,
  factor: number,
): {
  readonly rgba: Uint8ClampedArray<ArrayBuffer>;
  readonly width: number;
  readonly height: number;
} => {
  const cellWidth = video.width / factor;
  const cellHeight = video.height / factor;
  const width = cellWidth * cols;
  const height = cellHeight * rows;
  const rgba = new Uint8ClampedArray(width * height * 4);
  const cells = rows * cols;
  for (let cell = 0; cell < cells; cell += 1) {
    // 先頭と末尾のフレームを含めて等間隔に引く（33 / 81 枚から 32 マス）。
    const frame = Math.round((cell * (video.frames - 1)) / (cells - 1));
    const source = wanFrameToRgba(video, frame);
    const left = (cell % cols) * cellWidth;
    const top = Math.floor(cell / cols) * cellHeight;
    for (let y = 0; y < cellHeight; y += 1) {
      for (let x = 0; x < cellWidth; x += 1) {
        const at = ((top + y) * width + left + x) * 4;
        for (let channel = 0; channel < 3; channel += 1) {
          let sum = 0;
          for (let dy = 0; dy < factor; dy += 1) {
            for (let dx = 0; dx < factor; dx += 1) {
              sum += source[((y * factor + dy) * video.width + x * factor + dx) * 4 + channel];
            }
          }
          rgba[at + channel] = Math.round(sum / (factor * factor));
        }
        rgba[at + 3] = 255;
      }
    }
  }
  return { rgba, width, height };
};
