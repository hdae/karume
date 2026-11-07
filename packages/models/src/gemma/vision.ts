/**
 * Gemma 4 12B's image preprocessing: RGB8 → the merged patches and positions the vision embedder
 * reads (`Gemma4UnifiedImageProcessor`, transformers 5.14.1). Byte-identical to it
 * (`gemma4_vision_preprocess_test.ts`).
 *
 * The steps: the aspect-preserving size rule ({@link imageTargetSize}), torch's uint8 bicubic resize
 * (`image/torch-resample.ts`), the rescale, and the split into 48×48 raster blocks. Upstream reaches
 * those blocks by patchifying at 16 and merging 3×3; the result is the same raster crop, in the same
 * order, so no 16-pixel step exists here. Its padding to the budget is dropped too: the embedder's
 * row count is the image's own.
 */

import { ModelInputError } from "../errors.ts";
import { assertRgb8, type Rgb8Image } from "../image/preprocess.ts";
import { resizeRgb8TorchBicubic } from "../image/torch-resample.ts";

/** The side of one soft token in pixels (upstream `patch_size · pooling_kernel_size`). */
const SOFT_TOKEN_SIDE = 48;

/** Upstream's `pooling_kernel_size²`: the 16-pixel patches of one soft token. */
const PATCHES_PER_SOFT_TOKEN = 9;

/** Pixels of one 16-pixel patch (upstream `patch_size²`). */
const PATCH_AREA = 256;

/** The elements of one soft token's pixel row (48 · 48 · 3). */
export const IMAGE_ROW_ELEMENTS = SOFT_TOKEN_SIDE * SOFT_TOKEN_SIDE * 3;

/** The token budgets upstream accepts (`max_soft_tokens`). */
const SOFT_TOKEN_BUDGETS: readonly number[] = [70, 140, 280, 560, 1120];

/** Upstream's default, and the only budget a pipeline uses so far (ADR 0144 decision 5). */
export const DEFAULT_SOFT_TOKEN_BUDGET = 280;

/** Upstream's `1 / 255` as the f32 it multiplies by. */
const RESCALE = Math.fround(1 / 255);

/**
 * Upstream's `get_aspect_ratio_preserving_size`: the largest multiples of 48 that keep the aspect and
 * stay within the budget, with its rule for a side that rounds to 0.
 */
export const imageTargetSize = (
  height: number,
  width: number,
  budget: number = DEFAULT_SOFT_TOKEN_BUDGET,
): { readonly height: number; readonly width: number } => {
  if (!SOFT_TOKEN_BUDGETS.includes(budget)) {
    throw new ModelInputError(
      `Soft token budget ${budget} is not one of ${SOFT_TOKEN_BUDGETS.join(" / ")}`,
    );
  }
  const maxPatches = budget * PATCHES_PER_SOFT_TOKEN;
  const targetArea = maxPatches * PATCH_AREA;
  const factor = Math.sqrt(targetArea / (height * width));
  let targetHeight = Math.floor((factor * height) / SOFT_TOKEN_SIDE) * SOFT_TOKEN_SIDE;
  let targetWidth = Math.floor((factor * width) / SOFT_TOKEN_SIDE) * SOFT_TOKEN_SIDE;
  if (targetHeight === 0 && targetWidth === 0) {
    throw new ModelInputError(`Image ${width}×${height} resizes to nothing`);
  }
  const maxSide = Math.floor(maxPatches / PATCHES_PER_SOFT_TOKEN) * SOFT_TOKEN_SIDE;
  if (targetHeight === 0) {
    targetHeight = SOFT_TOKEN_SIDE;
    targetWidth = Math.min(Math.floor(width / height) * SOFT_TOKEN_SIDE, maxSide);
  } else if (targetWidth === 0) {
    targetWidth = SOFT_TOKEN_SIDE;
    targetHeight = Math.min(Math.floor(height / width) * SOFT_TOKEN_SIDE, maxSide);
  }
  if (targetHeight * targetWidth > targetArea) {
    throw new ModelInputError(
      `Image ${width}×${height} resizes to ${targetWidth}×${targetHeight}, over the budget of ${budget} soft tokens`,
    );
  }
  return { height: targetHeight, width: targetWidth };
};

/** One image's rows for the embedder. */
export type ImageRows = {
  /** The soft tokens of the image (N). */
  readonly softTokens: number;
  /** `[N, 6912]`: each row one 48×48 block in raster order, (y, x, channel), values in [0, 1]. */
  readonly pixels: Float32Array<ArrayBuffer>;
  /** `[N, 2]`: (column, row) of each block, in blocks. */
  readonly positions: Int32Array<ArrayBuffer>;
};

/** The preprocessing of one image. */
export const imageRows = (
  image: Rgb8Image,
  budget: number = DEFAULT_SOFT_TOKEN_BUDGET,
): ImageRows => {
  assertRgb8(image);
  const target = imageTargetSize(image.height, image.width, budget);
  const data =
    target.height === image.height && target.width === image.width
      ? image.data
      : resizeRgb8TorchBicubic(image, target.width, target.height).data;
  const columns = target.width / SOFT_TOKEN_SIDE;
  const rows = target.height / SOFT_TOKEN_SIDE;
  const softTokens = columns * rows;
  const pixels = new Float32Array(softTokens * IMAGE_ROW_ELEMENTS);
  const positions = new Int32Array(softTokens * 2);
  let at = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      positions[((row * columns + column) * 2) | 0] = column;
      positions[((row * columns + column) * 2 + 1) | 0] = row;
      for (let y = 0; y < SOFT_TOKEN_SIDE; y++) {
        const start = ((row * SOFT_TOKEN_SIDE + y) * target.width + column * SOFT_TOKEN_SIDE) * 3;
        for (let k = 0; k < SOFT_TOKEN_SIDE * 3; k++) {
          pixels[at++] = (data[start + k] ?? 0) * RESCALE;
        }
      }
    }
  }
  return { softTokens, pixels, positions };
};
