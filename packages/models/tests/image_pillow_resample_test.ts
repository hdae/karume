// The **preconditions** of the shared core of Pillow's 8-bit separable resample
// (`src/image/pillow-resample.ts`). Value equality is pinned by each caller's fixture (BiRefNet's
// `birefnet_matte_test.ts`, Wan's LANCZOS fixture), so this only checks that a mismatch in the
// input, the output size or the window fails with `RangeError` instead of being silently read
// part-way (no GPU, no assets).

import { assertEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  PILLOW_BICUBIC,
  PILLOW_LANCZOS,
  resampleU8Window,
  resizeU8PlaneBicubic,
} from "../src/image/pillow-resample.ts";

describe("preconditions of resizeU8PlaneBicubic", () => {
  it("throws RangeError when the plane's length does not match the size (shorter or longer)", () => {
    assertThrows(
      () => resizeU8PlaneBicubic(new Uint8Array(5), 3, 2, 4, 4),
      RangeError,
      "BICUBIC: input 3×2×1 and length 5 do not match",
    );
    assertThrows(
      () => resizeU8PlaneBicubic(new Uint8Array(7), 3, 2, 4, 4),
      RangeError,
      "and length 7 do not match",
    );
  });

  it("throws RangeError when an input dimension is not a positive integer (even when the length matches the product)", () => {
    // 0 × n has a product matching length 0, so a length check alone would let it through.
    assertThrows(() => resizeU8PlaneBicubic(new Uint8Array(0), 0, 3, 2, 2), RangeError, "input");
    assertThrows(() => resizeU8PlaneBicubic(new Uint8Array(3), 1.5, 2, 2, 2), RangeError, "input");
  });

  it("throws RangeError when an output dimension is not a positive integer", () => {
    for (const [width, height] of [
      [0, 2],
      [2, -1],
      [2.5, 2],
    ] as const) {
      assertThrows(
        () => resizeU8PlaneBicubic(new Uint8Array(6), 3, 2, width, height),
        RangeError,
        `BICUBIC: output size ${width}×${height} is not a positive integer`,
      );
    }
  });

  it("accepts an input meeting the preconditions (the control showing the checks above do not always fail)", () => {
    const out = resizeU8PlaneBicubic(new Uint8Array(6).fill(200), 3, 2, 5, 4);
    assertEquals(out.length, 20);
    assertEquals([...new Set(out)], [200]);
  });
});

describe("preconditions of resampleU8Window", () => {
  const image = { data: new Uint8Array(4 * 3 * 3), width: 4, height: 3, channels: 3 };

  it("throws RangeError when the window is not inside the output (overhang, negative coordinates, empty window)", () => {
    for (const window of [
      { left: 6, top: 0, width: 2, height: 2 },
      { left: 0, top: 4, width: 2, height: 2 },
      { left: -1, top: 0, width: 2, height: 2 },
      { left: 0, top: 0, width: 0, height: 2 },
    ]) {
      assertThrows(
        () => resampleU8Window(image, 7, 5, window, PILLOW_LANCZOS),
        RangeError,
        "LANCZOS: window",
      );
    }
  });

  it("throws RangeError when the channel count does not match the pixel array's length", () => {
    assertThrows(
      () =>
        resampleU8Window(
          { ...image, channels: 1 },
          7,
          5,
          { left: 0, top: 0, width: 7, height: 5 },
          PILLOW_BICUBIC,
        ),
      RangeError,
      "BICUBIC: input 4×3×1 and length 36 do not match",
    );
  });
});
