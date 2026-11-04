// The host side of preprocessing the condition image of Wan2.2 TI2V's I2V (ADR 0121 step 9a — no GPU, no assets).
//
// Six things are checked:
//
// - **LANCZOS** (`src/image/lanczos.ts`): matches Pillow's `Image.resize(..., LANCZOS)` exactly in uint8 (the git-tracked
//   fixture `fixtures/wan-i2v/lanczos.{json,safetensors}` — written by the recipe's `wan/lanczos_fixture.py`; when it is
//   missing the result is red, not SKIP). Plus 2 points independent of how the fixture is made (an identity size is a
//   copy, and upscaling a step saturates at 0 / 255 without wrapping around mod 256).
// - **Choosing the output size**: chosen from the accepted set by the official comparison `max(r/rc, rc/r)`, with a tie
//   (a square) going to landscape. A form measuring the absolute difference gives a different result at 1100×1000 (a
//   table row turns that mix-up red).
// - **The size after scaling**: Python's `round` (.5 to even) — 752.5 → 752, 757.5 → 758.
// - **Fitting the size**: crop scales to the covering size, then cuts out the centre at `(rw − ow) // 2` (floor). stretch
//   fits the whole image.
// - **`[-1, 1]` and patchify**: matches the official `to_tensor → sub(0.5) → div(0.5)` in f32, and patchify has the same
//   layout as upstream's `permute(0,1,6,4,2,3,5)` (a mix-up of the height and width sub-indices turns red under fault
//   injection).
// - **Normalisation of the encoder output**: `f32(f32(mu − mean) · f32(1 / f32(std)))` (diverging from the division form
//   and from the form with an f64 reciprocal).
//
// Bit-exactness of the real-image chain (PNG → … → the encoder input) against the golden is in
// `e2e_wan_ti2v_i2v_preprocess_real_test.ts`.

import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { parseSafetensors, type SafetensorsFile } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import {
  resizeRgb8Lanczos,
  resizeRgb8LanczosWindow,
  type Rgb8Window,
} from "../src/image/lanczos.ts";
import { normalizeToNchw, type Rgb8Image } from "../src/image/preprocess.ts";
import { WAN22_TI2V_GENERATION } from "../src/wan/descriptor.ts";
import {
  fitWanI2vImage,
  patchifyWanImage,
  preprocessWanI2vImage,
  selectWanI2vSize,
  wanI2vCoverSize,
  type I2vFit,
  wanI2vPixels,
} from "../src/wan/i2v-preprocess.ts";
import {
  denormalizeWanLatents,
  normalizeWanLatents,
  WAN22_LATENTS_MEAN,
  WAN22_LATENTS_STD,
} from "../src/wan/latents.ts";
import { unpatchifyWanVaeFrames } from "../src/wan/vae-tiles.ts";
import { WAN_I2V_LANCZOS_FIXTURE } from "./helpers/wan-i2v-image.ts";
import { firstBitMismatch } from "./helpers/wan-ti2v-dit.ts";
import { readBuffer, viewOf } from "./helpers/safetensors.ts";

const CHANNELS = 3;

/** Builds RGB8 by deciding a value per pixel (`pixel(x, y)` is [r, g, b]). */
const synthetic = (
  width: number,
  height: number,
  pixel: (x: number, y: number) => readonly [number, number, number],
): Rgb8Image => {
  const data = new Uint8Array(width * height * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data.set(pixel(x, y), (y * width + x) * CHANNELS);
    }
  }
  return { data, width, height };
};

/** An image with the coordinates encoded in the pixel values (the crop position can be read back from the values). */
const coordinateImage = (width: number, height: number): Rgb8Image =>
  synthetic(width, height, (x, y) => [x & 255, y & 255, (x >> 8) | ((y >> 8) << 4)]);

const pixelAt = (image: Rgb8Image, x: number, y: number): readonly number[] => {
  const at = (y * image.width + x) * CHANNELS;
  return [...image.data.subarray(at, at + CHANNELS)];
};

/** A rectangular crop (a copy of Pillow's `crop` — a reference independent of the window computation). */
const cropOf = (image: Rgb8Image, { left, top, width, height }: Rgb8Window): Rgb8Image => {
  const data = new Uint8Array(width * height * CHANNELS);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      data.set(pixelAt(image, left + x, top + y), (y * width + x) * CHANNELS);
    }
  }
  return { data, width, height };
};

// ---- LANCZOS ---------------------------------------------------------------------------------------------

/** One row of the fixture's case table (`cases` of `lanczos.json` — the pixels are on the safetensors side). */
type LanczosCase = {
  readonly name: string;
  readonly why: string;
  readonly height: number;
  readonly width: number;
  readonly outHeight: number;
  readonly outWidth: number;
};

type LanczosFixture = {
  readonly pillow: string;
  readonly cases: readonly LanczosCase[];
  readonly tensors: SafetensorsFile;
};

const isLanczosCase = (value: unknown): value is LanczosCase => {
  if (typeof value !== "object" || value === null) return false;
  const row = value as Record<string, unknown>;
  return (
    typeof row.name === "string" &&
    typeof row.why === "string" &&
    [row.height, row.width, row.outHeight, row.outWidth].every(
      (size) => Number.isInteger(size) && (size as number) > 0,
    )
  );
};

/**
 * Reads the fixture (the case table JSON + the pixel safetensors). Fails loudly when its shape departs from the doc of
 * `helpers/wan-i2v-image.ts`. MUST: throw when it is missing (it is a git-tracked fixture, so a missing one is a mix-up
 * or a write failure — rounding it to SKIP would turn "nobody checks the match with Pillow" green).
 */
const readLanczosFixture = async (): Promise<LanczosFixture> => {
  const raw = JSON.parse(await readTextFile(WAN_I2V_LANCZOS_FIXTURE)) as Record<string, unknown>;
  const source = raw.source as Record<string, unknown> | undefined;
  if (
    typeof source?.pillow !== "string" ||
    typeof raw.tensors !== "string" ||
    !Array.isArray(raw.cases) ||
    !raw.cases.every(isLanczosCase)
  ) {
    throw new Error(
      `${WAN_I2V_LANCZOS_FIXTURE.pathname}: the case table's shape is not {source.pillow, tensors, cases}`,
    );
  }
  const tensors = parseSafetensors(await readBuffer(new URL(raw.tensors, WAN_I2V_LANCZOS_FIXTURE)));
  return { pillow: source.pillow, cases: raw.cases, tensors };
};

/** The pixels of a case (U8 `[height, width, 3]` — a shape mismatch fails loudly). */
const pixelsOf = (
  fixture: LanczosFixture,
  key: string,
  height: number,
  width: number,
): Uint8Array<ArrayBuffer> => {
  const view = viewOf(fixture.tensors, key, "the LANCZOS fixture");
  assertEquals(view.dtype, "U8", `dtype of ${key}`);
  assertEquals([...view.shape], [height, width, CHANNELS], `shape of ${key}`);
  return new Uint8Array(
    fixture.tensors.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength),
  );
};

describe("LANCZOS (a verbatim port of Pillow's 8-bit path)", () => {
  it("every fixture case matches Pillow exactly in uint8", async () => {
    const fixture = await readLanczosFixture();
    const failures: string[] = [];
    for (const { name, height, width, outHeight, outWidth } of fixture.cases) {
      const input: Rgb8Image = {
        data: pixelsOf(fixture, `${name}.input`, height, width),
        width,
        height,
      };
      const expected = pixelsOf(fixture, `${name}.resized`, outHeight, outWidth);
      const actual = resizeRgb8Lanczos(input, outWidth, outHeight);
      let mismatches = 0;
      let first = -1;
      for (let index = 0; index < expected.length; index += 1) {
        if (actual.data[index] === expected[index]) continue;
        mismatches += 1;
        if (first < 0) first = index;
      }
      if (mismatches > 0) {
        failures.push(
          `${name}: ${mismatches} / ${expected.length} samples differ` +
            ` (the first is ${first}: ${actual.data[first]} vs ${expected[first]})`,
        );
      }
    }
    assertEquals(
      failures,
      [],
      `Pillow ${fixture.pillow}: of ${fixture.cases.length} cases, ${failures.length} differ`,
    );
  });

  it("the fixture includes downscaling (the path where the support widens), upscaling and a change on one axis only, and records the Pillow version", async () => {
    const fixture = await readLanczosFixture();
    const { cases } = fixture;
    assert(
      cases.some((row) => row.outWidth < row.width || row.outHeight < row.height),
      "no downscaling case (the path common in real inputs is not checked against Pillow)",
    );
    assert(
      cases.some((row) => row.outWidth > row.width || row.outHeight > row.height),
      "no upscaling case",
    );
    // NOTE: skipping the pass on an axis whose size does not change cannot be told apart in the output (the coefficients
    // of n → n are 2^22 at the centre and 0 elsewhere, so going through the pass gives the same values as a copy). What
    // this case pins is scaling on one axis only.
    assert(
      cases.some((row) => (row.outWidth === row.width) !== (row.outHeight === row.height)),
      "no case scaling one axis only (horizontal only, vertical only) (scaling on one axis only is not checked against Pillow)",
    );
    assertEquals(new Set(cases.map(({ name }) => name)).size, cases.length, "duplicate case names");
    assert(fixture.pillow.length > 0, "the Pillow version is empty");
  });

  it("when the size does not change it skips the pass and returns a copy of the input (the same values, another array)", () => {
    const image = coordinateImage(23, 17);
    const same = resizeRgb8Lanczos(image, 23, 17);
    assertEquals([...same.data], [...image.data]);
    assert(
      same.data.buffer !== image.data.buffer,
      "returned the same array as the input (shared with the caller's image)",
    );
  });

  it("computing only the window gives the same uint8 values as scaling the whole and then cropping (both axes, one axis, identity, the window's edge)", () => {
    const table: readonly (readonly [number, number, number, number, Rgb8Window])[] = [
      // [input width, height, scaled width, height, window]
      [13, 9, 50, 31, { left: 7, top: 4, width: 30, height: 20 }], // upscaling on both axes
      [73, 41, 29, 17, { left: 3, top: 5, width: 20, height: 9 }], // downscaling on both axes (the support widens)
      [40, 12, 23, 12, { left: 2, top: 3, width: 15, height: 6 }], // horizontal only (vertically a row crop)
      [12, 40, 12, 57, { left: 4, top: 10, width: 5, height: 30 }], // vertical only (horizontally a column crop)
      [21, 13, 21, 13, { left: 3, top: 2, width: 9, height: 7 }], // identity (a crop on both)
      [9, 50, 31, 13, { left: 30, top: 12, width: 1, height: 1 }], // the one pixel at the bottom-right corner
      [50, 9, 13, 31, { left: 0, top: 0, width: 13, height: 31 }], // window = the whole
    ];
    for (const [width, height, outWidth, outHeight, window] of table) {
      const image = synthetic(width, height, (x, y) => [
        (x * 37 + y * 11) & 255,
        (x * y) & 255,
        ((x ^ y) * 5) & 255,
      ]);
      const whole = resizeRgb8Lanczos(image, outWidth, outHeight);
      const windowed = resizeRgb8LanczosWindow(image, outWidth, outHeight, window);
      assertEquals([windowed.width, windowed.height], [window.width, window.height]);
      assertEquals(
        [...windowed.data],
        [...cropOf(whole, window).data],
        `${width}×${height} → ${outWidth}×${outHeight}, window ${JSON.stringify(window)}`,
      );
    }
  });

  it("a window reaching outside the output is a RangeError (an internal precondition)", () => {
    const image = coordinateImage(4, 4);
    for (const window of [
      { left: 5, top: 0, width: 4, height: 8 },
      { left: 0, top: -1, width: 8, height: 8 },
      { left: 0, top: 0, width: 0, height: 8 },
      { left: 0.5, top: 0, width: 4, height: 8 },
    ]) {
      assertThrows(() => resizeRgb8LanczosWindow(image, 8, 8, window), RangeError, "window");
    }
  });

  it("upscaling a step goes past the input's value range (5..250) and saturates at 0 / 255, without wrapping around mod 256", () => {
    // The expected values are a copy of the output of Pillow 12.3.0's `Image.resize((32, 1), Image.LANCZOS)` (the trough
    // of the negative lobe sticks to 0 on the 5 side, the peak to 255 on the 250 side). Storing into a `Uint8Array`
    // without the saturation wraps the trough around to about 250.
    const image = synthetic(8, 1, (x) => {
      const value = x < 4 ? 5 : 250;
      return [value, value, value];
    });
    const resized = resizeRgb8Lanczos(image, 32, 1);
    const red = Array.from({ length: 32 }, (_, x) => resized.data[x * CHANNELS]);
    assertEquals(
      red,
      [
        5, 5, 5, 5, 5, 5, 5, 9, 13, 10, 0, 0, 0, 0, 28, 91, 164, 227, 255, 255, 255, 255, 245, 242,
        246, 250, 250, 250, 250, 250, 250, 250,
      ],
    );
  });

  it("an output size that is not a positive integer is a RangeError (an internal precondition), and an image length mismatch is a ModelInputError", () => {
    const image = coordinateImage(4, 4);
    assertThrows(() => resizeRgb8Lanczos(image, 0, 4), RangeError);
    assertThrows(() => resizeRgb8Lanczos(image, 4.5, 4), RangeError);
    assertThrows(
      () => resizeRgb8Lanczos({ data: new Uint8Array(10), width: 4, height: 4 }, 8, 8),
      ModelInputError,
    );
  });
});

// ---- Choosing the output size ----------------------------------------------------------------------------------

describe("choosing the output size (the accepted set × the official comparison max(r/rc, rc/r), a tie goes to landscape)", () => {
  const blank = (width: number, height: number): Rgb8Image => ({
    data: new Uint8Array(width * height * CHANNELS),
    width,
    height,
  });

  it("chooses the accepted size closest to the image's aspect ratio", () => {
    const table: readonly (readonly [number, number, number, number])[] = [
      [832, 480, 1280, 704],
      [480, 832, 704, 1280],
      [1920, 1080, 1280, 704],
      [1080, 1920, 704, 1280],
      // A square is an exact tie between 2 candidates → landscape.
      [512, 512, 1280, 704],
      // r = 1.1: the ratio comparison gives landscape (1.65 vs 2.0), the absolute difference portrait (0.72 vs 0.55) — the
      // row that turns a mix-up of the measure red.
      [1100, 1000, 1280, 704],
      [1000, 1001, 704, 1280],
    ];
    for (const [width, height, expectedWidth, expectedHeight] of table) {
      assertEquals(
        selectWanI2vSize(blank(width, height), {}, WAN22_TI2V_GENERATION),
        { width: expectedWidth, height: expectedHeight },
        `${width}×${height}`,
      );
    }
  });

  it("the landscape tie-break does not depend on the order of the accepted set (a square is landscape even with portrait listed first)", () => {
    const reversed = { acceptedSizes: [...WAN22_TI2V_GENERATION.acceptedSizes].reverse() };
    assertEquals(selectWanI2vSize(blank(512, 512), {}, reversed), { width: 1280, height: 704 });
  });

  it("explicit sizes take precedence over the aspect ratio, and with only one given, the choice is among candidates matching that field", () => {
    const landscape = blank(832, 480);
    assertEquals(selectWanI2vSize(landscape, { width: 704, height: 1280 }, WAN22_TI2V_GENERATION), {
      width: 704,
      height: 1280,
    });
    assertEquals(selectWanI2vSize(landscape, { width: 704 }, WAN22_TI2V_GENERATION), {
      width: 704,
      height: 1280,
    });
    assertEquals(selectWanI2vSize(blank(480, 832), { height: 704 }, WAN22_TI2V_GENERATION), {
      width: 1280,
      height: 704,
    });
  });

  it("an explicit size outside the accepted set is a ModelInputError", () => {
    const image = blank(832, 480);
    for (const [requested, message] of [
      [{ width: 832, height: 480 }, "832×480 is not in the accepted set (1280×704 / 704×1280)"],
      [{ width: 1280, height: 1280 }, "1280×1280 is not in the accepted set"],
      [{ width: 1000 }, "width 1000 is not in the accepted set"],
      [{ height: 1000 }, "height 1000 is not in the accepted set"],
    ] as const) {
      assertThrows(
        () => selectWanI2vSize(image, requested, WAN22_TI2V_GENERATION),
        ModelInputError,
        message,
      );
    }
  });

  it("a mismatch in the image itself (its length) is a ModelInputError", () => {
    assertThrows(
      () =>
        selectWanI2vSize(
          { data: new Uint8Array(5), width: 4, height: 4 },
          {},
          WAN22_TI2V_GENERATION,
        ),
      ModelInputError,
    );
  });
});

// ---- The size after scaling (Python's round) ----------------------------------------------------------------------

describe("the size after crop's scaling (scale = max(ow/iw, oh/ih) → Python's round)", () => {
  const landscape = { width: 1280, height: 704 };

  it("exactly .5 rounds to even (752.5 → 752, 757.5 → 758)", () => {
    // JS's Math.round would give 753 (.5 goes toward +∞) — this row turns a rounding mix-up red.
    assertEquals(Math.round(301 * 2.5), 753);
    assertEquals(wanI2vCoverSize({ width: 512, height: 301 }, landscape), {
      width: 1280,
      height: 752,
    });
    assertEquals(wanI2vCoverSize({ width: 512, height: 303 }, landscape), {
      width: 1280,
      height: 758,
    });
  });

  it("takes the covering scale (the axis that set the scale hits the target exactly, the other is at least the target)", () => {
    assertEquals(wanI2vCoverSize({ width: 832, height: 480 }, landscape), {
      width: 1280,
      height: 738,
    });
    assertEquals(wanI2vCoverSize({ width: 480, height: 832 }, { width: 704, height: 1280 }), {
      width: 738,
      height: 1280,
    });
    assertEquals(wanI2vCoverSize({ width: 1280, height: 800 }, landscape), {
      width: 1280,
      height: 800,
    });
    assertEquals(wanI2vCoverSize({ width: 3000, height: 1000 }, landscape), {
      width: 2112,
      height: 704,
    });
  });
});

// ---- Fitting the size ------------------------------------------------------------------------------------------

describe("fitting the size (crop = cover, then cut out the centre / stretch = scale the whole directly)", () => {
  const landscape = { width: 1280, height: 704 };

  it("crop's cut-out position is (rw − ow) // 2 (the remainder goes right / down — floor)", () => {
    // An axis at scale 1 skips the LANCZOS pass, so the output pixels are the input pixels themselves (the position can be
    // read back from the coordinates).
    for (const [width, height, left, top] of [
      [1280, 800, 0, 48],
      [1280, 805, 0, 50], // (805 − 704) / 2 = 50.5 → 50
      [1400, 704, 60, 0],
      [1401, 704, 60, 0], // 60.5 → 60
    ] as const) {
      const image = coordinateImage(width, height);
      const fitted = fitWanI2vImage(image, landscape, "crop");
      assertEquals([fitted.width, fitted.height], [1280, 704]);
      for (const [x, y] of [
        [0, 0],
        [1279, 0],
        [0, 703],
        [1279, 703],
        [640, 352],
      ] as const) {
        assertEquals(
          pixelAt(fitted, x, y),
          pixelAt(image, x + left, y + top),
          `${width}×${height} at (${x}, ${y})`,
        );
      }
    }
  });

  it("crop gives the same uint8 values as scaling the whole to the covering size and then cutting out the centre (an extreme aspect ratio 64×4 → covering size 11,264×704)", () => {
    const image = synthetic(64, 4, (x, y) => [
      (x * 13 + y * 71) & 255,
      (x * 7) & 255,
      (y * 97) & 255,
    ]);
    const cover = wanI2vCoverSize(image, landscape);
    assertEquals(cover, { width: 11264, height: 704 });
    const whole = resizeRgb8Lanczos(image, cover.width, cover.height);
    const expected = cropOf(whole, {
      left: (cover.width - landscape.width) / 2,
      top: 0,
      width: landscape.width,
      height: landscape.height,
    });
    assertEquals([...fitWanI2vImage(image, landscape, "crop").data], [...expected.data]);
  });

  it("an image with an extreme aspect ratio (4096×4 → covering size 720,896×704) also computes and cuts out only the output window", () => {
    // NOTE: the cost bound itself cannot be made a check (reverting to building the whole covering size, this it still ends
    // green after a few minutes and 1.5 GB — it does not turn red). What this pins is the size, and that the window path
    // runs for an edge window too (the centre of the covering size). The values are compared with the whole scaling by
    // the 64×4 it above.
    const image = synthetic(4096, 4, (x, y) => [x & 255, (x >> 4) & 255, y * 60]);
    const fitted = fitWanI2vImage(image, landscape, "crop");
    assertEquals([fitted.width, fitted.height], [1280, 704]);
    assertEquals(fitted.data.length, 1280 * 704 * CHANNELS);
  });

  it("stretch fits the whole without keeping the aspect ratio, keeping the top band that crop drops", () => {
    // Only the top 8 rows of 1280×800 are red. crop drops 48 rows at the top and bottom, so no red remains; stretch
    // shrinks 800 → 704 and keeps it.
    const image = synthetic(1280, 800, (_, y) => (y < 8 ? [255, 0, 0] : [128, 128, 128]));
    const cropped = fitWanI2vImage(image, landscape, "crop");
    const stretched = fitWanI2vImage(image, landscape, "stretch");
    assertEquals([stretched.width, stretched.height], [1280, 704]);
    const reddish = (fitted: Rgb8Image): boolean =>
      Array.from({ length: fitted.width * fitted.height }, (_, at) => at * CHANNELS).some(
        (at) => fitted.data[at] > 200 && fitted.data[at + 1] < 60,
      );
    assert(!reddish(cropped), "the top band remained with crop");
    assert(reddish(stretched), "the top band disappeared with stretch");
    const [r, g] = pixelAt(stretched, 640, 0);
    assert(r > 200 && g < 60, `the top of stretch is not red (${r}, ${g})`);
  });

  it("an unknown fit fails loudly (a plain Error — a misspelt name)", () => {
    assertThrows(
      () => fitWanI2vImage(coordinateImage(4, 4), landscape, "cover" as never),
      Error,
      "is not supported",
    );
  });
});

// ---- [-1, 1] and patchify ---------------------------------------------------------------------------------

describe("[-1, 1] pixels (f32(2·f32(x/255) − 1), planar [3, H, W])", () => {
  // A 256×1 image with x = 0..255 in R, 255 − x in G and x in B.
  const ramp = synthetic(256, 1, (x) => [x, 255 - x, x]);

  it("matches both the official to_tensor → sub(0.5) → div(0.5) and diffusers' 2·(x/255) − 1 in f32, with the channels laid out planar", () => {
    const pixels = wanI2vPixels(ramp);
    assertEquals(pixels.length, 3 * 256);
    const official = (value: number): number =>
      Math.fround(Math.fround(Math.fround(value / 255) - 0.5) / 0.5);
    // diffusers (the reference for stretch): `np.float32(x) / 255.0` → `2.0 * images - 1.0` (both f32 operations).
    const diffusers = (value: number): number =>
      Math.fround(Math.fround(2 * Math.fround(value / 255)) - 1);
    for (let x = 0; x < 256; x += 1) {
      assertEquals(pixels[x], official(x), `R ${x}`);
      assertEquals(pixels[x], diffusers(x), `R ${x} (the diffusers form)`);
      assertEquals(pixels[256 + x], official(255 - x), `G ${x}`);
      assertEquals(pixels[512 + x], official(x), `B ${x}`);
    }
    assertEquals([pixels[0], pixels[255]], [-1, 1]);
  });

  it("the last bit differs from the folded form normalizeToNchw (mean = std = 0.5) (a mix-up from reusing it turns red)", () => {
    const pixels = wanI2vPixels(ramp);
    const folded = normalizeToNchw(ramp, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    const differs = Array.from({ length: 256 }, (_, x) => x).filter((x) => pixels[x] !== folded[x]);
    assert(
      differs.length > 0,
      "every value matched the folded form (a formula mix-up is not pinned)",
    );
  });
});

/**
 * Copies torch's `view` → `permute` → `contiguous` through strides (a formulation independent of the implementation's
 * nested loops). `viewShape` is the row-major shape, `order` the permute's axis order.
 */
const viewPermute = (
  data: Float32Array,
  viewShape: readonly number[],
  order: readonly number[],
): Float32Array<ArrayBuffer> => {
  const strides = viewShape.map((_, axis) =>
    viewShape.slice(axis + 1).reduce((product, dim) => product * dim, 1),
  );
  const outShape = order.map((axis) => viewShape[axis]);
  const out = new Float32Array(data.length);
  const index = outShape.map(() => 0);
  for (let to = 0; to < out.length; to += 1) {
    out[to] = data[order.reduce((from, axis, k) => from + index[k] * strides[axis], 0)];
    for (let k = outShape.length - 1; k >= 0; k -= 1) {
      index[k] += 1;
      if (index[k] < outShape[k]) break;
      index[k] = 0;
    }
  }
  return out;
};

describe("patchify (upstream diffusers' patchify — permute(0,1,6,4,2,3,5), p = 2)", () => {
  // A [3, 6, 10] whose height and width differ and whose elements all have distinct values (a mix-up always shows in the
  // values).
  const height = 6;
  const width = 10;
  const pixels = Float32Array.from({ length: 3 * height * width }, (_, index) => index);
  const p = 2;
  /** Upstream: `x [1, c, 1, H, W]` → `view(1, c, 1, H/p, p, W/p, p)` → `permute(0, 1, 6, 4, 2, 3, 5)`. */
  const viewShape = [1, 3, 1, height / p, p, width / p, p];

  it("the same layout as upstream's view → permute (channel = c·p² + width sub-index·p + height sub-index)", () => {
    const actual = patchifyWanImage(pixels, [3, height, width], p);
    assertEquals([...actual], [...viewPermute(pixels, viewShape, [0, 1, 6, 4, 2, 3, 5])]);
  });

  it("fault injection: diverges from the layout with the height and width sub-indices swapped (permute(0,1,4,6,…))", () => {
    const actual = patchifyWanImage(pixels, [3, height, width], p);
    const swapped = viewPermute(pixels, viewShape, [0, 1, 4, 6, 2, 3, 5]);
    assertNotEquals([...actual], [...swapped]);
  });

  it("unpatchifyWanVaeFrames (the inverse at the VAE output) restores the original", () => {
    const patched = patchifyWanImage(pixels, [3, height, width], p);
    const restored = unpatchifyWanVaeFrames(patched, [12, 1, height / p, width / p], p);
    assertEquals([...restored], [...pixels]);
  });

  it("a size not divisible by the factor and a length mismatch fail loudly", () => {
    assertThrows(
      () => patchifyWanImage(new Float32Array(3 * 5 * 4), [3, 5, 4], 2),
      Error,
      "is not divisible by",
    );
    assertThrows(() => patchifyWanImage(new Float32Array(7), [3, 2, 2], 2), Error, "element count");
  });
});

// ---- The preprocessing chain ------------------------------------------------------------------------------------------

describe("the preprocessing chain (choosing the size → fitting the size → [-1, 1] → patchify)", () => {
  it("the encoder input is [12, 1, H/2, W/2], the size is chosen by the aspect ratio, and fit defaults to crop", () => {
    const image = coordinateImage(832, 480);
    const input = preprocessWanI2vImage(image, {}, WAN22_TI2V_GENERATION);
    assertEquals([input.width, input.height], [1280, 704]);
    assertEquals(input.shape, [12, 1, 352, 640]);
    assertEquals(input.pixels.length, 12 * 352 * 640);
    const crop = preprocessWanI2vImage(image, { fit: "crop" }, WAN22_TI2V_GENERATION);
    assertEquals(firstBitMismatch(input.pixels, crop.pixels), -1);
    const expected = patchifyWanImage(
      wanI2vPixels(fitWanI2vImage(image, { width: 1280, height: 704 }, "crop")),
      [3, 704, 1280],
      2,
    );
    assertEquals(firstBitMismatch(input.pixels, expected), -1);

    const portrait = preprocessWanI2vImage(coordinateImage(480, 832), {}, WAN22_TI2V_GENERATION);
    assertEquals(portrait.shape, [12, 1, 640, 352]);
  });

  it("fit = stretch gives different pixels from crop (the field takes effect inside the chain)", () => {
    const image = coordinateImage(832, 480);
    const fits: readonly I2vFit[] = ["crop", "stretch"];
    const [crop, stretch] = fits.map((fit) =>
      preprocessWanI2vImage(image, { fit }, WAN22_TI2V_GENERATION),
    );
    assertEquals(crop.shape, stretch.shape);
    assertNotEquals(firstBitMismatch(crop.pixels, stretch.pixels), -1);
  });
});

// ---- Normalisation of the encoder output ------------------------------------------------------------------------------

describe("normalisation of the encoder output (upstream (mu − mean) · f32(1 / f32(std)))", () => {
  const stats = WAN22_TI2V_GENERATION.latents;
  const perChannel = 4096;
  const channels = WAN22_LATENTS_MEAN.length;
  const mu = Float32Array.from({ length: channels * perChannel }, (_, index) =>
    Math.fround(Math.sin(index) * 2),
  );

  it("the value is f32(f32(mu − mean) · f32(1 / f32(std))) (the channels on axis 0)", () => {
    const got = normalizeWanLatents(mu, stats);
    for (let channel = 0; channel < channels; channel += 1) {
      const inverse = Math.fround(1 / Math.fround(WAN22_LATENTS_STD[channel]));
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        assertEquals(
          got[at],
          Math.fround(Math.fround(mu[at] - mean) * inverse),
          `${channel}/${index}`,
        );
      }
    }
  });

  it("fault injection: diverges from dividing by std and from a reciprocal built in f64 from the config's decimal values", () => {
    const got = normalizeWanLatents(mu, stats);
    let differsFromDivision = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        const divided = Math.fround(Math.fround(mu[at] - mean) / WAN22_LATENTS_STD[channel]);
        if (divided !== got[at]) differsFromDivision += 1;
      }
    }
    assert(
      differsFromDivision > 0,
      "every value matched dividing by std (the multiplication order is not pinned)",
    );
    // The config values are decimals of at most 4 digits (the convention of `latents.ts`), so toFixed(4) recovers the
    // decimal value.
    let differsFromDecimalInverse = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const mean = Math.fround(WAN22_LATENTS_MEAN[channel]);
      const inverse = Math.fround(1 / Number(WAN22_LATENTS_STD[channel].toFixed(4)));
      for (let index = 0; index < perChannel; index += 1) {
        const at = channel * perChannel + index;
        const fromDecimal = Math.fround(Math.fround(mu[at] - mean) * inverse);
        if (fromDecimal !== got[at]) differsFromDecimalInverse += 1;
      }
    }
    assert(
      differsFromDecimalInverse > 0,
      "every value matched the form with a decimal reciprocal in f64 (how the reciprocal is built is not pinned)",
    );
  });

  it("round-trips with the same statistics table and the same reciprocal as the denormalisation (the round-trip difference is within a few ulp of rounding)", () => {
    const restored = denormalizeWanLatents(normalizeWanLatents(mu, stats), stats);
    for (let index = 0; index < mu.length; index += 1) {
      assert(Math.abs(restored[index] - mu[index]) <= 1e-6 * Math.max(1, Math.abs(mu[index])));
    }
  });

  it("a length not divisible by the channel count and statistics with different counts fail loudly", () => {
    assertThrows(
      () => normalizeWanLatents(new Float32Array(49), stats),
      Error,
      "Normalization: element count 49",
    );
    assertThrows(
      () =>
        normalizeWanLatents(new Float32Array(48), { mean: stats.mean, std: stats.std.slice(1) }),
      Error,
      "Normalization: mean count 48 and std count 47 differ",
    );
  });
});
