// The `pipelineConfig` schema of `depth-anything/1` (`src/depth-anything/config.ts`). The **writing side** is
// `tools/export-recipes/depth_anything/distribution.py`, whose pytest pins the same field names from the other
// direction (if a field name moves on one side only, the distribution is produced but the load fails in parse).
//
// Three things are checked:
//
// ① Everything **outside** the accepted set is rejected (unknown keys, missing fields, out-of-range values).
//    Nothing silently falls back to a default.
// ② `imageStd` of 0 is rejected — a division by 0 throws nothing and produces `±Infinity` `pixel_values`, and
//    the graph emits NaN while only the shape matches.
// ③ The accepted set of `interpolation` is **the single value `bicubic`** (the **opposite** of BiRefNet /
//    SigLIP2's `bilinear`). DA-V2's upstream is `resample: 3` = PIL's BICUBIC, and passing a distribution that
//    declares bilinear shifts `pixel_values` by 34 times one uint8 LSB. The accepted set differing per family is
//    the very reason the schema is not shared, so both directions are pinned.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";

/** The actual `pipelineConfig` of `small` in `models/karume-depth-anything-v2/karume.json` (5 fields). */
const CONFIG: Record<string, unknown> = {
  imageWidth: 518,
  imageHeight: 518,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
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

test("reads the distribution's pipelineConfig as is (5 fields, values unchanged)", () => {
  const config = parseDepthAnythingPipelineConfig(CONFIG);
  assertEquals(config.imageWidth, 518);
  assertEquals(config.imageHeight, 518);
  assertEquals(config.imageMean, [0.485, 0.456, 0.406]);
  assertEquals(config.imageStd, [0.229, 0.224, 0.225]);
  assertEquals(config.interpolation, "bicubic");
});

test("rejects unknown keys (a misspelling does not silently fall back to a default)", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ image_mean: [0.5, 0.5, 0.5] })),
    Error,
    "unknown key 'image_mean'",
  );
});

test("also rejects leftover fields of the upstream config (keep_aspect_ratio / ensure_multiple_of)", () => {
  // The upstream `preprocessor_config.json` has these 2 fields, but the exported graph accepts only one square
  // size, so the declaration has no place for them. A distribution that copied them through stops here.
  for (const key of ["keepAspectRatio", "ensureMultipleOf"]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ [key]: 14 })),
      Error,
      `unknown key '${key}'`,
    );
  }
});

for (const key of Object.keys(CONFIG)) {
  test(`rejects a missing field: ${key}`, () => {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(without(key)),
      Error,
      `pipelineConfig.${key}`,
    );
  });
}

test("sizes are positive integers only (rejects 0 / fractions / strings)", () => {
  for (const bad of [0, -518, 518.5, "518"]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ imageWidth: bad })),
      Error,
      "imageWidth",
    );
  }
});

test("rejects imageStd of 0 (a division by 0 silently produces ±Infinity)", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ imageStd: [0.229, 0, 0.225] })),
    Error,
    "imageStd",
  );
});

test("rejects as invalid assets a distribution whose imageMean / imageStd pair overflows f32 after normalization", () => {
  // Each field is finite, but the quotient overflows f32. Unless it is rejected at build time, every generation fails with an input-caused type.
  const error = assertThrows(
    () => parseDepthAnythingPipelineConfig({ ...CONFIG, imageStd: [1e-50, 0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean / imageStd: channel 0: the normalized range",
  );
  assert(
    !(error instanceof ModelInputError),
    "invalid assets were classified as caused by the input",
  );
});

test("channel constants are arrays of 3 numbers only", () => {
  for (const bad of [[0.485, 0.456], [0.485, 0.456, 0.406, 0.5], [0.485, "0.456", 0.406], 0.485]) {
    assertThrows(
      () => parseDepthAnythingPipelineConfig(withPatch({ imageMean: bad })),
      Error,
      "imageMean",
    );
  }
});

test("imageMean also accepts negative values (it is a shift, so the range is not narrowed)", () => {
  assertEquals(
    parseDepthAnythingPipelineConfig(withPatch({ imageMean: [-0.5, 0, 0.5] })).imageMean,
    [-0.5, 0, 0.5],
  );
});

test("rejects non-finite constants (every pixel would become NaN)", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ imageMean: [0.485, Number.NaN, 0.406] })),
    Error,
    "imageMean",
  );
  assertThrows(
    () =>
      parseDepthAnythingPipelineConfig(
        withPatch({ imageStd: [0.229, Number.POSITIVE_INFINITY, 0.2] }),
      ),
    Error,
    "imageStd",
  );
});

test("does not accept a distribution that declares bilinear (the opposite of BiRefNet / SigLIP2)", () => {
  assertThrows(
    () => parseDepthAnythingPipelineConfig(withPatch({ interpolation: "bilinear" })),
    Error,
    "only 'bicubic'",
  );
});
