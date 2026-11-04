// Behaviour tests of shaping the latent / conditioning tensors and converting to RGBA. Pure
// functions that need neither a GPU nor assets.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertNotStrictEquals,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import {
  ANIMA_LATENTS_MEAN,
  ANIMA_LATENTS_STD,
  latentStats,
  denormalizeLatents,
  padSequence,
} from "../src/anima/latents.ts";
import { imageToRgba } from "../src/anima/image.ts";

/**
 * The reference fixture (`fixtures/anima-latents/parity.json` — its generator is described in the
 * file's `_doc`).
 */
type LatentsFixture = {
  readonly mean: readonly number[];
  readonly std: readonly number[];
};

const FIXTURE_PATH = new URL("./fixtures/anima-latents/parity.json", import.meta.url);
const fixture = JSON.parse(await readTextFile(FIXTURE_PATH)) as LatentsFixture;

test("denormalizeLatents: rejects B>1 and a channel-count mismatch", () => {
  const latents = new Float32Array(2 * 2 * 1 * 1);
  const mean = Float32Array.from([0, 0]);
  const std = Float32Array.from([1, 1]);
  assertThrows(
    () => denormalizeLatents(latents, [2, 2, 1, 1], mean, std),
    Error,
    "assumes batch=1",
  );
  assertThrows(
    () => denormalizeLatents(new Float32Array(3), [1, 3, 1, 1], mean, std),
    Error,
    "channel count",
  );
  assertThrows(
    () => denormalizeLatents(new Float32Array(5), [1, 2, 1, 1], mean, std),
    Error,
    "element count",
  );
});

test("denormalizeLatents: divides by std's reciprocal (rewriting it as a multiplication changes the last digit)", () => {
  const latents = Float32Array.from([1, 1]);
  const mean = Float32Array.from([0, 0]);
  const std = Float32Array.from([3, 7]);
  const got = denormalizeLatents(latents, [1, 2, 1, 1], mean, std);
  for (const [index, scale] of [3, 7].entries()) {
    assertEquals(got[index], Math.fround(1 / Math.fround(1 / scale)), `channel ${index}`);
  }
  // Shows that it **actually differs** from the multiplication form (if it cannot be shown, the
  // MUST means nothing).
  assert(
    got[0] !== Math.fround(1 * 3) || got[1] !== Math.fround(1 * 7),
    "dividing by the reciprocal and multiplying agree on every channel, so this invariant is not pinned",
  );
});

test("demo constants: latents_mean / latents_std are bit-identical to the reference fixture", () => {
  // The substance of the MUST in `latents.ts` (no hand-copied number without validation). These 32
  // values exist only in the VAE config and are in neither the IR nor the distributed assets, so
  // this is the only check that catches a copying mistake (the PNG sha256 check on a real GPU runs
  // only in the reference environment that has both the assets and a GPU). A plain f32 comparison
  // is enough — the fixture's values come from torch f32 tensors and `Float32Array.from` rounds
  // them to the same f32.
  assertEquals([...ANIMA_LATENTS_MEAN], [...Float32Array.from(fixture.mean)]);
  assertEquals([...ANIMA_LATENTS_STD], [...Float32Array.from(fixture.std)]);
});

test("demo constants: latents_mean / latents_std come as a pair of 16 channels", () => {
  // The values themselves are pinned by the bit identity above. This one stops, structurally, the
  // case where "only one of them was replaced" would slip through.
  assertEquals(ANIMA_LATENTS_MEAN.length, 16);
  assertEquals(ANIMA_LATENTS_STD.length, ANIMA_LATENTS_MEAN.length);
  assert(
    ANIMA_LATENTS_STD.every((value) => value > 0),
    "std has a non-positive value",
  );
});

test("latentStats: each call returns an independent copy (a consumer's writes do not reach the original)", () => {
  // This accessor is all the public surface exposes. Handing out the original would let a consumer
  // that rewrites a single element silently change the denormalisation of every later `generate`
  // (global mutable state).
  const first = latentStats();
  assertEquals([...first.mean], [...ANIMA_LATENTS_MEAN]);
  assertEquals([...first.std], [...ANIMA_LATENTS_STD]);

  const meanBefore = ANIMA_LATENTS_MEAN[0];
  const stdBefore = ANIMA_LATENTS_STD[0];
  first.mean[0] = 999;
  first.std[0] = 999;

  // Neither the internal original nor a copy fetched again changes.
  assertEquals(ANIMA_LATENTS_MEAN[0], meanBefore);
  assertEquals(ANIMA_LATENTS_STD[0], stdBefore);
  const second = latentStats();
  assertEquals(second.mean[0], meanBefore);
  assertEquals(second.std[0], stdBefore);
  assertNotStrictEquals(second.mean, first.mean);
  assertNotStrictEquals(second.std, first.std);
});

test("padSequence: padding is zero, B>1 and too many rows are rejected", () => {
  const hidden = { dtype: "f32", shape: [1, 2, 3], data: Float32Array.from([1, 2, 3, 4, 5, 6]) };
  const padded = padSequence(hidden as never, 4);
  assertEquals([...padded], [1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0]);
  assertThrows(() => padSequence(hidden as never, 1), Error, "upper bound");
  assertThrows(
    () => padSequence({ dtype: "f32", shape: [2, 1, 3], data: new Float32Array(6) } as never, 4),
    Error,
    "assumes batch=1",
  );
  // For i32 / bool, `set` silently converts the numbers (shape and length still match, only the
  // values are wrong).
  assertThrows(
    () => padSequence({ dtype: "i32", shape: [1, 2, 3], data: new Int32Array(6) } as never, 4),
    Error,
    "assumes f32",
  );
});

test("imageToRgba: [-1,1] → [0,255], planar to interleaved, alpha 255", () => {
  // A 1×2 image. The R plane / G plane / B plane come in this order.
  const image = Float32Array.from([
    -1,
    1, // R
    0,
    -2, // G (-2 is clamped)
    2,
    0.5, // B (2 is clamped)
  ]);
  const rgba = imageToRgba(image, 2, 1);
  assertEquals([...rgba], [0, 128, 255, 255, 255, 0, 191, 255]);
});

test("imageToRgba: rejects a mismatch between the element count and the size", () => {
  assertThrows(() => imageToRgba(new Float32Array(5), 2, 1), Error, "element count");
  assertThrows(() => imageToRgba(new Float32Array(6), 0, 1), RangeError);
});

test("imageToRgba: a dimension that is not a positive integer is a RangeError even when the element count matches (the same entry contract as wan.frameToRgba)", () => {
  // Every one of them satisfies `image.length === 3·width·height` (a shape that a length check
  // alone lets through).
  assertThrows(
    () => imageToRgba(new Float32Array(3), 0.5, 2),
    RangeError,
    "not a positive integer",
  );
  assertThrows(
    () => imageToRgba(new Float32Array(3), -1, -1),
    RangeError,
    "not a positive integer",
  );
});

test("imageToRgba: rejects a non-finite value with its coordinates (no silent conversion to a black pixel)", () => {
  // Three planes of 2×2. The base is all finite (with out-of-range values that get clamped mixed
  // in), and a single element is broken.
  const base = new Float32Array(2 * 2 * 3);
  base.set([-2, -1, 0, 1], 0); // R
  base.set([1, 0, -1, -2], 4); // G
  base.set([0, 1, -2, -1], 8); // B
  const spoil = (offset: number, value: number): Float32Array<ArrayBuffer> => {
    const image = base.slice();
    image[offset] = value;
    return image;
  };
  // The base does not fail — what is checked is "finite", not "in range".
  assertEquals([...imageToRgba(base, 2, 2).slice(0, 4)], [0, 255, 128, 255]);

  // The broken position moves per plane, to see that the coordinates point at the actual pixel
  // rather than a fixed wording.
  const nan = assertThrows(() => imageToRgba(spoil(3, NaN), 2, 2), Error);
  assertStringIncludes(nan.message, "x=1");
  assertStringIncludes(nan.message, "y=1");
  assertStringIncludes(nan.message, "channel 0");

  const positiveInfinity = assertThrows(() => imageToRgba(spoil(5, Infinity), 2, 2), Error);
  assertStringIncludes(positiveInfinity.message, "x=1");
  assertStringIncludes(positiveInfinity.message, "y=0");
  assertStringIncludes(positiveInfinity.message, "channel 1");

  const negativeInfinity = assertThrows(() => imageToRgba(spoil(10, -Infinity), 2, 2), Error);
  assertStringIncludes(negativeInfinity.message, "x=0");
  assertStringIncludes(negativeInfinity.message, "y=1");
  assertStringIncludes(negativeInfinity.message, "channel 2");
});

test("imageToRgba: runs row-major even when non-square (H≠W)", () => {
  // 2×3 (width 2 / height 3). A transposed implementation keeps the same length and changes only
  // the pixel order, so **each plane gets different values** to catch it by position (a mix-up a
  // square image cannot detect).
  const plane = Float32Array.from([-1, 1, 1, -1, -1, 1]);
  const image = new Float32Array(18);
  image.set(plane, 0); // R
  image.set(plane, 6); // G
  image.set(plane, 12); // B
  const rgba = imageToRgba(image, 2, 3);
  assertEquals(rgba.length, 2 * 3 * 4);
  // Pixel (x=1, y=0) is plane[1] = 1 → 255, pixel (x=0, y=1) is plane[2] = 1 → 255.
  assertEquals([...rgba.slice(4, 8)], [255, 255, 255, 255]);
  assertEquals([...rgba.slice(8, 12)], [255, 255, 255, 255]);
  // Pixel (x=0, y=0) is plane[0] = -1 → 0.
  assertEquals([...rgba.slice(0, 4)], [0, 0, 0, 255]);
});
