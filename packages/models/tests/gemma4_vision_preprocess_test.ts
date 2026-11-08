/**
 * Gemma 4 12B's image preprocessing against the upstream processor, byte for byte.
 *
 * The fixture (`tools/export-recipes/gemma4/tests/vision_fixture.py`) holds upstream's sizes and
 * SHA-256 over images both sides build from one seed.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vite-plus/test";

import {
  imageRows,
  imageTargetSize,
  maxImageSoftTokens,
  SOFT_TOKEN_BUDGETS,
} from "../src/gemma/vision.ts";
import { ModelInputError } from "../src/errors.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { resizeRgb8TorchBicubic } from "../src/image/torch-resample.ts";

type ProcessorCase = {
  readonly height: number;
  readonly width: number;
  readonly budget: number;
  readonly seed: number;
  readonly softTokens: number;
  readonly pixelsSha256: string;
  readonly positionsSha256: string;
};

type ResizeCase = {
  readonly height: number;
  readonly width: number;
  readonly targetHeight: number;
  readonly targetWidth: number;
  readonly seed: number;
  readonly sha256: string;
};

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/gemma4-vision/preprocess.json", import.meta.url), "utf8"),
) as { readonly processor: readonly ProcessorCase[]; readonly resize: readonly ResizeCase[] };

/** MUST: the same arithmetic as `synthetic_rgb` in the fixture's generator. */
const syntheticRgb = (height: number, width: number, seed: number): Rgb8Image => {
  const data = new Uint8Array(height * width * 3);
  let state = seed >>> 0;
  let at = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const noise = (state >>> 24) % 64;
        const gradient =
          (Math.floor((x * (c + 1) * 191) / width) + Math.floor((y * (3 - c) * 127) / height)) %
          192;
        data[at++] = gradient + noise;
      }
    }
  }
  return { data, width, height };
};

const sha256 = async (bytes: ArrayBufferView<ArrayBuffer>): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Buffer.from(digest).toString("hex");
};

describe("resizeRgb8TorchBicubic", () => {
  it.each(fixture.resize)(
    "matches F.resize byte for byte from $height×$width to $targetHeight×$targetWidth",
    async ({ height, width, targetHeight, targetWidth, seed, sha256: expected }) => {
      const resized = resizeRgb8TorchBicubic(
        syntheticRgb(height, width, seed),
        targetWidth,
        targetHeight,
      );
      expect(resized.width).toBe(targetWidth);
      expect(resized.height).toBe(targetHeight);
      expect(await sha256(new Uint8Array(resized.data))).toBe(expected);
    },
  );

  it("returns a copy when no axis changes", () => {
    const image = syntheticRgb(8, 8, 1);
    const resized = resizeRgb8TorchBicubic(image, 8, 8);
    expect(resized.data).toEqual(image.data);
    expect(resized.data).not.toBe(image.data);
  });
});

describe("imageRows", () => {
  it.each(fixture.processor)(
    "matches the processor byte for byte at $height×$width, budget $budget",
    async ({ height, width, budget, seed, softTokens, pixelsSha256, positionsSha256 }) => {
      const rows = imageRows(syntheticRgb(height, width, seed), budget);
      expect(rows.softTokens).toBe(softTokens);
      expect(await sha256(rows.pixels)).toBe(pixelsSha256);
      expect(await sha256(rows.positions)).toBe(positionsSha256);
    },
  );

  it("reads the pixels of a raster block, in that block's position", () => {
    const image = syntheticRgb(96, 96, 7);
    const size = imageTargetSize(96, 96);
    // 96² upscales, so take the block from the image the rows are built from.
    const resized = resizeRgb8TorchBicubic(image, size.width, size.height);
    const rows = imageRows(image);
    const columns = size.width / 48;
    const block = columns + 2;
    expect(Array.from(rows.positions.subarray(block * 2, block * 2 + 2))).toEqual([2, 1]);
    const firstPixel = (48 * size.width + 2 * 48) * 3;
    expect(rows.pixels[block * 6912]).toBe(
      Math.fround((resized.data[firstPixel] ?? 0) * Math.fround(1 / 255)),
    );
  });

  it("refuses a budget upstream does not accept", () => {
    expect(() => imageTargetSize(100, 100, 281)).toThrow(ModelInputError);
  });
});

describe("maxImageSoftTokens", () => {
  it.each(SOFT_TOKEN_BUDGETS)("is reached by an image at budget %i, and never passed", (budget) => {
    const most = maxImageSoftTokens(budget);
    // An aspect whose 48-pixel blocks multiply to the budget (70 = 7 × 10, …, 1120 = 28 × 40).
    const columns = [10, 14, 20, 28, 40][SOFT_TOKEN_BUDGETS.indexOf(budget)];
    const filling = imageTargetSize((budget / columns) * 48, columns * 48, budget);
    expect((filling.height / 48) * (filling.width / 48)).toBe(most);
    for (const [height, width] of [
      [1, 1],
      [97, 3001],
      [4000, 3000],
      [48, 4800],
      [1080, 1920],
    ]) {
      const size = imageTargetSize(height, width, budget);
      expect((size.height / 48) * (size.width / 48)).toBeLessThanOrEqual(most);
    }
  });

  it("refuses a budget upstream does not accept", () => {
    expect(() => maxImageSoftTokens(281)).toThrow(ModelInputError);
  });
});
