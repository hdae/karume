// The `pipelineConfig` schema of `birefnet/2` (`src/birefnet/config.ts`). The **writing side** is
// `tools/export-recipes/birefnet/distribution.py`, whose pytest pins the same field names from the
// other direction (if a field name moved on one side only, the distribution would build but loading
// would fail in parse).
//
// It checks four points:
//
// ① Everything **outside** the accepted set is rejected (unknown keys, missing fields, out-of-range
//   values). No silent degrading to a default.
// ② A 0 in `imageStd` is rejected — a division by 0 throws nothing and makes `pixel_values` of
//   `±Infinity`, and the graph matches only in shape while emitting NaN.
// ③ `interpolation` is a **declaration**, not a branch, and its accepted set is a single value (the
//   upstream `handler.py` runs torchvision Resize with its default interpolation = bilinear — the
//   preprocessing layer also has bicubic).
// ④ `postprocess` declares a member of a closed set (`MATTE_PROCEDURES`); it is required and has no
//   default (ADR 0125). Outside the set, both misspellings and Object.prototype names are rejected.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";
import { MATTE_PROCEDURES } from "../src/birefnet/matte.ts";

/**
 * The `pipelineConfig` of `models/karume-birefnet-hr/karume.json` (the shape rebuilt for
 * `birefnet/2`, 6 fields).
 */
const CONFIG: Record<string, unknown> = {
  imageWidth: 1024,
  imageHeight: 1024,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
  postprocess: "u8-pil-bicubic",
};

const withPatch = (patch: Record<string, unknown>): Record<string, unknown> => ({
  ...CONFIG,
  ...patch,
});

const without = (key: string): Record<string, unknown> => {
  const copy = { ...CONFIG };
  delete copy[key];
  return copy;
};

test("reads the distribution's pipelineConfig as is (6 fields, values unchanged)", () => {
  const config = parseBirefnetPipelineConfig(CONFIG);
  assertEquals(config.imageWidth, 1024);
  assertEquals(config.imageHeight, 1024);
  assertEquals(config.imageMean, [0.485, 0.456, 0.406]);
  assertEquals(config.imageStd, [0.229, 0.224, 0.225]);
  assertEquals(config.interpolation, "bilinear");
  assertEquals(config.postprocess, "u8-pil-bicubic");
});

test("rejects an unknown key (a misspelling does not silently degrade to a default)", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ image_mean: [0.5, 0.5, 0.5] })),
    Error,
    "unknown key 'image_mean'",
  );
});

for (const key of Object.keys(CONFIG)) {
  test(`rejects a missing field: ${key}`, () => {
    assertThrows(() => parseBirefnetPipelineConfig(without(key)), Error, `pipelineConfig.${key}`);
  });
}

test("dimensions are positive integers only (rejects 0 / fractions / strings)", () => {
  for (const bad of [0, -1024, 1024.5, "1024"]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ imageWidth: bad })),
      Error,
      "imageWidth",
    );
  }
});

test("rejects a 0 in imageStd (a division by 0 quietly makes ±Infinity)", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ imageStd: [0.229, 0, 0.225] })),
    Error,
    "imageStd",
  );
});

test("a distribution whose imageMean / imageStd pair overflows f32 after normalisation is rejected as a broken asset", () => {
  // Each field is finite, yet the quotient overflows f32. Unless construction rejects it, every
  // generation fails with the input-caused type.
  const error = assertThrows(
    () => parseBirefnetPipelineConfig({ ...CONFIG, imageStd: [1e-50, 0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean / imageStd: channel 0: the normalized range",
  );
  assert(!(error instanceof ModelInputError), "a broken asset was classified as input-caused");
});

test("channel constants are arrays of three numbers only", () => {
  for (const bad of [[0.485, 0.456], [0.485, 0.456, 0.406, 0.5], [0.485, "0.456", 0.406], 0.485]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ imageMean: bad })),
      Error,
      "imageMean",
    );
  }
});

test("imageMean accepts negative values too (it is a translation, so the range is not narrowed)", () => {
  assertEquals(
    parseBirefnetPipelineConfig(withPatch({ imageMean: [-0.5, 0, 0.5] })).imageMean,
    [-0.5, 0, 0.5],
  );
});

test("rejects non-finite constants (every pixel would become NaN)", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ imageMean: [0.485, Number.NaN, 0.406] })),
    Error,
    "imageMean",
  );
  assertThrows(
    () =>
      parseBirefnetPipelineConfig(withPatch({ imageStd: [0.229, Number.POSITIVE_INFINITY, 0.2] })),
    Error,
    "imageStd",
  );
});

test("does not accept an unsupported interpolation (no silently passing it as bilinear)", () => {
  assertThrows(
    () => parseBirefnetPipelineConfig(withPatch({ interpolation: "bicubic" })),
    Error,
    "only 'bilinear'",
  );
});

test("accepts every member of the closed set of postprocess procedures as declared", () => {
  // The distributions of both series (HR = u8-pil-bicubic, Lucida = float-bilinear) read with the
  // same reader.
  for (const procedure of MATTE_PROCEDURES) {
    assertEquals(
      parseBirefnetPipelineConfig(withPatch({ postprocess: procedure })).postprocess,
      procedure,
    );
  }
});

test("rejects postprocess procedures outside the set (misspellings, letter case, Object.prototype names, non-string values)", () => {
  // Silently degrading to one series' procedure would leave the boundary α produced by a procedure
  // that differs from upstream, unnoticed.
  for (const bad of ["bicubic", "U8-PIL-BICUBIC", "float_bilinear", "", "toString", 1, null]) {
    assertThrows(
      () => parseBirefnetPipelineConfig(withPatch({ postprocess: bad })),
      Error,
      "pipelineConfig.postprocess: unknown postprocessing procedure",
    );
  }
});
