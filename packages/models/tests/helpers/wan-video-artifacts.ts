/**
 * Writes the artifacts of a generated Wan video (every frame as PNG plus a contact sheet) to the
 * results directory (the material for the maintainer's visual check — ADR 0121 step 6).
 *
 * NOTE: the contents are a verbatim copy of the contact-sheet and PNG writing of the 2.1 e2e
 * (`e2e_wan_pipeline_test.ts`). The copy on the 2.1 e2e side is not touched (the 2.1 lane is frozen
 * — ruling of ADR 0121 step 6), so removing the duplication is out of scope and handled separately.
 */

import { writeFile } from "@karume/test-support/fs";
import type { Results } from "../../../runtime/tests/helpers/results.ts";
import { encodePng } from "../../mod.ts";
import { type GeneratedVideo, frameToRgba } from "../../wan.ts";

/**
 * Writes every frame to the results directory as `<id>-frame-NN.png` (NN starts at 0, two digits or
 * more).
 */
export const writeWanFrames = async (
  results: Pick<Results, "artifact">,
  id: string,
  video: GeneratedVideo,
): Promise<void> => {
  for (let frame = 0; frame < video.frames; frame += 1) {
    await writeFile(
      results.artifact(`${id}-frame-${String(frame).padStart(2, "0")}.png`),
      await encodePng(frameToRgba(video, frame), video.width, video.height),
    );
  }
};

/**
 * A contact sheet (a `rows × cols` grid; each cell is a frame box-averaged down by a factor of
 * `factor`).
 */
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
    // Picks frames at even intervals, including the first and the last (32 cells from 33 / 81
    // frames).
    const frame = Math.round((cell * (video.frames - 1)) / (cells - 1));
    const source = frameToRgba(video, frame);
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
