// **Construction guards** and **preprocessing wiring** of `Pipeline`. Only what needs neither a GPU
// nor real assets (the real-GPU comparison belongs to `packages/runtime/tests/e2e_siglip2_test.ts`
// — not duplicated).
//
// Three points are pinned:
//
// ① `fromAssets` rejects **manifest contract violations before acquiring a GPU** (the MUST stated
//  by
//    `openSiglip2State` in `src/siglip2/pipeline.ts`). The observation set-up is the same as on the
//    irodori side — **the containers are present in every case**, and the order of the checks
//    itself is pinned by two things:
//     - a contract-violation case fails with that violation's message = the manifest check comes
//       first even with all assets present
//     - a correct manifest + an empty Record fails with "component 'vision' has no container" (the
//       entry diagnostic).
//    NOTE: the containers are opened **before** admission (`assetComponentOpener` returns a
//    synchronous supply, so it opens every component first — the seam of ADR 0109). So the current
//    contract is not "fails before opening the assets" but "fails at the manifest check even with
//    all assets present".
//
// ② The **rejection paths** of the comparison with the graph declarations (`assertStaticDim` /
//    `assertOutputDim`). Inside `fromAssets` these can only be reached with a real GPU and real
//    assets, so the checks are called directly (`tests/helpers/stub-model.ts` builds a
//    declaration-only `GraphOwner`). If a check's own spelling or axis number drifted, it would
//    stay green as long as only the normal cases ran, so **rejecting a broken declaration by name**
//    is exercised every time.
//
// ③ The preprocessing wiring (`preprocessPixelValues`) runs `resizeRgb8` → `normalizeToNchw` as
//    `pipelineConfig` declares. **The reference is a fixture whose source of truth is Python**
//    (`fixtures/image-preprocess/parity.json` — the checks of the preprocessing layer itself are in
//    `image_preprocess_test.ts`), so all this looks at is whether the six declared fields reach the
//    right argument positions. A non-square case is used because a mix-up of `imageWidth` /
//    `imageHeight` is **undetectable in principle** with a square.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseSiglip2PipelineConfig } from "../src/siglip2/config.ts";
import { ModelInputError } from "../src/errors.ts";
import {
  assertOutputDim,
  assertStaticDim,
  preprocessPixelValues,
  Pipeline,
} from "../src/siglip2/pipeline.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";

/**
 * A declaration-only `vision` container (reproduces what the family's checks read; never executed).
 */
const VISION = partAssets(
  "vision",
  await tensorlessContainer("vision", {
    inputs: [{ name: "pixel_values", shape: [1, 3, 224, 224] }],
    output: { name: "pooler_output", shape: [1, 768] },
  }),
);

/** The real `pipelineConfig` of `base` in `models/karume-siglip2/karume.json` (six fields). */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 224,
  imageHeight: 224,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  hiddenDim: 768,
  interpolation: "bilinear",
};

/**
 * Skeleton of the distribution (only the fields the checks need). `patch` overrides the contents of
 * `models["base"]`.
 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.1.0",
    defaultModel: "base",
    models: {
      base: {
        pipeline: "siglip2/1",
        weights: { vision: { f32: declaredContainer("vision/model.f32") } },
        // Running needs only one graph (no tokenizer and no tables).
        assets: {},
        quants: { f32: { weights: { vision: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

/** A manifest with only `pipelineConfig` replaced (the rest is the skeleton). */
const withConfig = (config: Record<string, unknown>): string =>
  manifestText({ pipelineConfig: config });

test("fromAssets: an unknown model fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }, { model: "nope" }),
    Error,
    "model 'nope' is not in the manifest",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not siglip2", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "anima/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }),
    Error,
    "the manifest's pipeline is 'anima/1'",
  );
});

test("fromAssets: an unknown major fails loudly (models owns the check — ADR 0038 §1)", async () => {
  // The only check that stops the silent degradation "old implementation × new repository". hub
  // does not check the major.
  const manifest = parseManifest(manifestText({ pipeline: "siglip2/2" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }),
    Error,
    "the major of pipeline 'siglip2/2' is not supported",
  );
});

test("fromAssets: an unknown quant fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }, { quant: "nope" }),
    Error,
    "quant 'nope' is not in the manifest",
  );
});

test("fromAssets: an unknown pipelineConfig key fails at construction", async () => {
  // No path where a misspelling (`image_mean` for `imageMean`) silently falls back to the default.
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, image_mean: [0.5, 0.5, 0.5] }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }),
    Error,
    "pipelineConfig: unknown key 'image_mean'",
  );
});

test("fromAssets: a missing pipelineConfig field fails at construction", async () => {
  // If a missing field were filled with a default, preprocessing would pass with the host alone
  // holding a different number (the MUST of config.ts).
  const { hiddenDim: _dropped, ...missing } = PIPELINE_CONFIG;
  const manifest = parseManifest(withConfig(missing));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }),
    Error,
    "pipelineConfig.hiddenDim: missing",
  );
});

test("fromAssets: does not accept a distribution that declares an interpolation other than bilinear", async () => {
  // This distribution's upstream resample is 2 (PIL's BILINEAR). The preprocessing layer also
  // implements bicubic, but widening the accepted set is a separate decision, and silently running
  // bicubic as bilinear would execute with resize values off by up to 47/255 (the MUST of
  // config.ts).
  const manifest = parseManifest(withConfig({ ...PIPELINE_CONFIG, interpolation: "bicubic" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: VISION }),
    Error,
    "pipelineConfig.interpolation: this implementation supports only 'bilinear'",
  );
});

test("fromAssets: fails with both spellings when a component's container is missing (the entry diagnostic)", async () => {
  // The flip side of the seven cases above. A Record without the containers fails on **the missing
  // component** (not with a manifest message) = the contrapositive showing that the cases above are
  // not tripped by missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'vision' has no container",
  );
});

// ---- The pipelineConfig schema (calling the check directly) ----------------

test("parseSiglip2PipelineConfig: reads the six real fields as they are", () => {
  const config = parseSiglip2PipelineConfig(PIPELINE_CONFIG);
  assertEquals(config.imageWidth, 224);
  assertEquals(config.imageHeight, 224);
  assertEquals(config.imageMean, [0.5, 0.5, 0.5]);
  assertEquals(config.imageStd, [0.5, 0.5, 0.5]);
  assertEquals(config.hiddenDim, 768);
  assertEquals(config.interpolation, "bilinear");
});

test("parseSiglip2PipelineConfig: fails when std contains 0 (division by zero raises no exception)", () => {
  // It is the divisor of `(x − mean·255) / (std·255)`, so 0 silently produces ±Infinity
  // pixel_values.
  assertThrows(
    () => parseSiglip2PipelineConfig({ ...PIPELINE_CONFIG, imageStd: [0.5, 0, 0.5] }),
    Error,
    "pipelineConfig.imageStd: has an element that is not a positive finite number",
  );
});

test("a distribution whose imageMean / imageStd pair overflows f32 after normalisation fails as an asset fault", () => {
  // Each field is finite, but the quotient overflows f32. Not rejected at construction, it would
  // fail with an input-error type on every generation.
  const error = assertThrows(
    () => parseSiglip2PipelineConfig({ ...PIPELINE_CONFIG, imageStd: [1e-50, 0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean / imageStd: channel 0: the normalized range",
  );
  assert(!(error instanceof ModelInputError), "an asset fault was classified as an input error");
});

test("parseSiglip2PipelineConfig: fails when mean / std do not have 3 elements", () => {
  assertThrows(
    () => parseSiglip2PipelineConfig({ ...PIPELINE_CONFIG, imageMean: [0.5, 0.5] }),
    Error,
    "pipelineConfig.imageMean: not an array of length 3",
  );
});

// ---- Comparison with the graph declarations (rejection paths) ---------------

/** The same declarations as the real distribution (base). `patch` breaks one point. */
const visionGraph = (
  patch: {
    readonly inputShape?: readonly (number | string)[];
    readonly outputShape?: number[];
  } = {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 224, 224] }],
    outputs: ["pooler_output"],
    values: { pooler_output: patch.outputShape ?? [1, 768] },
  });

test("assertStaticDim: a graph matching the declaration passes, and a different resolution fails by name", () => {
  // The shape of so400m (384²) assets placed where base (224²) belongs. Preprocessing resizes to
  // the declared size, so the host side passes to the end and the failure comes from Session's
  // shape check = nobody can tell which side is right.
  assertStaticDim(visionGraph(), "pixel_values", 2, 224, "imageHeight");
  const error = assertThrows(
    () =>
      assertStaticDim(
        visionGraph({ inputShape: [1, 3, 384, 384] }),
        "pixel_values",
        2,
        224,
        "imageHeight",
      ),
    Error,
    "siglip2: imageHeight — graph input 'pixel_values' axis 2 is 384",
  );
  assert(error.message.includes("pipelineConfig has 224"), error.message);
});

test("assertStaticDim: fails when the input name itself is missing", () => {
  assertThrows(
    () => assertStaticDim(visionGraph(), "pixel", 2, 224, "imageHeight"),
    Error,
    "siglip2: graph input 'pixel' is missing (imageHeight)",
  );
});

test("assertOutputDim: a wrongly declared embedding dimension fails by name", () => {
  // hiddenDim is used by neither preprocessing nor execution, so even when it is off the embeddings
  // come out as if nothing were wrong.
  assertOutputDim(visionGraph(), 1, 768, "hiddenDim");
  assertThrows(
    () => assertOutputDim(visionGraph({ outputShape: [1, 1152] }), 1, 768, "hiddenDim"),
    Error,
    "siglip2: hiddenDim — graph output 'pooler_output' axis 1 is 1152",
  );
});

test("assertOutputDim: fails when the output name has no declaration in values", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 224, 224] }],
    outputs: ["pooler_output"],
    values: {},
  });
  assertThrows(
    () => assertOutputDim(model, 1, 768, "hiddenDim"),
    Error,
    "siglip2: graph output 'pooler_output' has no declaration (hiddenDim)",
  );
});

// ---- Preprocessing wiring (against the fixture whose source of truth is Python) ----

type ParityCase = {
  readonly name: string;
  readonly height: number;
  readonly width: number;
  readonly outHeight: number;
  readonly outWidth: number;
  readonly input: number[];
  readonly pixelValues: number[];
};

type Fixture = {
  readonly constants: {
    readonly imageMean: [number, number, number];
    readonly imageStd: [number, number, number];
  };
  readonly cases: ParityCase[];
};

const FIXTURE_PATH = new URL("./fixtures/image-preprocess/parity.json", import.meta.url);
const fixture = JSON.parse(await readTextFile(FIXTURE_PATH)) as Fixture;

/** A non-square case shrunk 4× vertically and enlarged 3× horizontally (24×5 → 6×15). */
const MIXED_AXIS = "mixed-axis";

/**
 * The tolerance for every path. Derived at the top of `image_preprocess_test.ts` (1 uint8 LSB
 * converted to f32).
 */
const PIXEL_ATOL = 7.85e-3;

const parityCase = (name: string): ParityCase => {
  const found = fixture.cases.find((entry) => entry.name === name);
  assert(found !== undefined, `the fixture has no '${name}' (the generation script changed)`);
  return found;
};

const imageOf = (entry: ParityCase): Rgb8Image => ({
  data: Uint8Array.from(entry.input),
  width: entry.width,
  height: entry.height,
});

/**
 * A `pipelineConfig` declaring the case's sizes and constants as is (hiddenDim does not affect the
 * wiring).
 */
const configFor = (entry: ParityCase, swapped = false) =>
  parseSiglip2PipelineConfig({
    imageWidth: swapped ? entry.outHeight : entry.outWidth,
    imageHeight: swapped ? entry.outWidth : entry.outHeight,
    imageMean: fixture.constants.imageMean,
    imageStd: fixture.constants.imageStd,
    hiddenDim: 768,
    interpolation: "bilinear",
  });

test("preprocessing wiring: matches the Python source of truth with the declared size, mean and std (non-square)", () => {
  const entry = parityCase(MIXED_AXIS);
  const got = preprocessPixelValues(configFor(entry), imageOf(entry));
  assertEquals(got.length, entry.pixelValues.length);
  for (let index = 0; index < got.length; index += 1) {
    assertAlmostEquals(got[index], entry.pixelValues[index], PIXEL_ATOL, `sample ${index}`);
  }
});

test("preprocessing wiring: swapping imageWidth / imageHeight gives a different result", () => {
  // Ensures the case above is not vacuous — transposing a non-square declaration changes even the
  // element count (`resizeRgb8` takes `(width, height)` in that order).
  const entry = parityCase(MIXED_AXIS);
  assert(
    entry.outWidth !== entry.outHeight,
    "a transpose is undetectable without a non-square case",
  );
  const swapped = preprocessPixelValues(configFor(entry, true), imageOf(entry));
  assertEquals(
    swapped.length,
    entry.pixelValues.length,
    "the element count is the same when transposed",
  );
  let matched = 0;
  for (let index = 0; index < swapped.length; index += 1) {
    if (Math.abs(swapped[index] - entry.pixelValues[index]) <= PIXEL_ATOL) matched += 1;
  }
  assert(
    matched < swapped.length,
    "every sample matched even when transposed (the check has no effect)",
  );
});
