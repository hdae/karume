// The **build guards** and the **pre- and post-processing wiring** of `Pipeline`. Only the part that needs
// neither a GPU nor real assets is checked here (the real-GPU comparison is owned by
// `packages/runtime/tests/e2e_depth_anything_real_test.ts` (golden inputs) and `e2e_depth_anything_real_test.ts`
// (preprocessing parity on real images and discrimination) — not duplicated).
//
// Four things are pinned:
//
// ① `fromAssets` rejects **manifest contract violations before acquiring the GPU** (the MUST stated by
//    `openDepthAnythingState` in `src/depth-anything/pipeline.ts`). The observation setup is the same as on the
//    SigLIP2 / BiRefNet side — **every case has the containers in place**, and
//     - a contract-violation case fails with "that violation's message" = the manifest check comes first even
//       with the assets in place
//     - a correct manifest + an empty Record fails with `component 'depth' has no container` (the diagnostic of
//       the entry point)
//    together pin the order of the checks itself.
//    NOTE: the containers are opened **before** admission (`assetComponentOpener` returns a synchronous opener,
//    so it opens every component first — the seam of ADR 0109).
//
// ② The **rejection paths** of the comparison with the graph declaration (`assertStaticDim` /
//    `assertDepthShape`). Inside `fromAssets` they cannot be reached without a real GPU and real assets, so the
//    checks are called directly (`tests/helpers/stub-model.ts` builds a declaration-only `GraphOwner`). Even if
//    the axis numbers or expected shapes of the checks themselves drift, they stay green as long as only the
//    normal path runs, so **rejecting a broken declaration by name** is exercised every time.
//
// ③ The preprocessing wiring (`preprocessPixelValues`) runs `resizeRgb8` (**bicubic**) → `normalizeToNchw` as
//    declared in `pipelineConfig`. **There is no Python-sourced fixture for the bicubic branch yet** (the
//    bilinear branch's `fixtures/image-preprocess/parity.json` uses a different filter and cannot serve as a
//    reference), so only invariants that need no assets and **can fail** are checked here:
//     - for a uniform image, "do the declared constants reach that channel" can be stated in closed form
//     - transposing a non-square declaration changes the values
//     - **the result does not match the one run through bilinear** (it would match if it had fallen back to
//       the default)
//    The correctness of bicubic itself is owned by ① the kernel-signature test in `image_preprocess_test.ts`
//    and ② the E2E that compares with `DPTImageProcessor` on every run in an environment with real assets.
//
// ④ Postprocessing (`resampleDepth`) **does not touch the values**. The decision not to fold the depth into
//    `[0, 1]` (the module doc of `pipeline.ts`) would still pass with matching shape and non-negativity if it
//    were folded, so a check of the "fails if folded" form is in place.

import { test } from "vite-plus/test";
import { assert, assertAlmostEquals, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";
import {
  assertDepthShape,
  assertStaticDim,
  Pipeline,
  preprocessPixelValues,
  resampleDepth,
} from "../src/depth-anything/pipeline.ts";
import { normalizeToNchw, resizeRgb8, type Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";

/** A declaration-only `depth` container (reproduces the shape the family's checks read; never runs). */
const COMPONENT = partAssets(
  "depth",
  await tensorlessContainer("depth", {
    inputs: [{ name: "pixel_values", shape: [1, 3, 518, 518] }],
    output: { name: "y", shape: [1, 518, 518] },
  }),
);

/** The actual `pipelineConfig` of `small` in `models/karume-depth-anything-v2/karume.json` (5 fields). */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 518,
  imageHeight: 518,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
};

/** The skeleton of a distribution (only the fields the checks need). `patch` overrides the contents of `models["small"]`. */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.2.2",
    defaultModel: "small",
    models: {
      small: {
        pipeline: "depth-anything/1",
        weights: { depth: { f32: declaredContainer("small/depth/model.f32") } },
        // Running needs only one graph (no tokenizer, no tables).
        assets: {},
        quants: { f32: { weights: { depth: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

test("fromAssets: rejects a model that does not exist, listing the available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { model: "nope" }),
    Error,
    "model 'nope' is not in the manifest",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not depth-anything", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "birefnet/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "the manifest's pipeline is 'birefnet/1'",
  );
});

test("fromAssets: an unknown major fails loudly (the check is the models side's job — ADR 0038 §1)", async () => {
  // The only check that stops the silent degradation of "an old implementation × a new repo". The hub does not check the major.
  const manifest = parseManifest(manifestText({ pipeline: "depth-anything/2" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "the major of pipeline 'depth-anything/2' is not supported",
  );
});

test("fromAssets: rejects a quant that does not exist, listing the available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { quant: "nope" }),
    Error,
    "quant 'nope' is not in the manifest",
  );
});

test("fromAssets: an unknown key in pipelineConfig fails at build time", async () => {
  // No path where a misspelling (`image_mean` for `imageMean`) silently falls back to a default.
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, image_mean: [0.5, 0.5, 0.5] } }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig: unknown key 'image_mean'",
  );
});

test("fromAssets: does not accept a distribution that declares an interpolation other than bicubic", async () => {
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, interpolation: "bilinear" } }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig.interpolation: this implementation supports only 'bicubic'",
  );
});

test("fromAssets: without the component's container it fails with both spellings (the entry point's diagnostic)", async () => {
  // The flip side of the 6 cases above. A Record without the containers fails on the **missing component** (not
  // on a manifest message) = the contrapositive showing that the cases above are not caught up in missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'depth' has no container",
  );
});

// ---- Comparison with the graph declaration (rejection paths) ------------------------------------------

/** The same declaration as the real distribution (small). `patch` breaks exactly one point. */
const depthGraph = (
  patch: {
    readonly inputShape?: readonly (number | string)[];
    readonly outputShape?: readonly (number | string)[];
  } = {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 518, 518] }],
    outputs: ["depth"],
    values: { depth: patch.outputShape ?? [1, 518, 518] },
  });

const depthConfig = parseDepthAnythingPipelineConfig(PIPELINE_CONFIG);

test("assertStaticDim: a graph as declared passes, and a different pretraining resolution fails by name", () => {
  // DINOv2's position embeddings are tied to the patch count, so a failure here is always **mixed-up assets**.
  assertStaticDim(depthGraph(), "pixel_values", 2, depthConfig.imageHeight, "imageHeight");
  const error = assertThrows(
    () =>
      assertStaticDim(
        depthGraph({ inputShape: [1, 3, 384, 384] }),
        "pixel_values",
        2,
        depthConfig.imageHeight,
        "imageHeight",
      ),
    Error,
    "depth-anything: imageHeight — graph input 'pixel_values' axis 2 is 384",
  );
  assert(error.message.includes("pipelineConfig has 518"), error.message);
});

test("assertStaticDim: rejects when the input name itself is missing", () => {
  assertThrows(
    () => assertStaticDim(depthGraph(), "pixel", 2, depthConfig.imageHeight, "imageHeight"),
    Error,
    "depth-anything: graph input 'pixel' is missing (imageHeight)",
  );
});

test("assertDepthShape: passes when as declared, and rejects an output that still has the channel axis", () => {
  // `[1, 1, S, S]` is the shape of a different export that outputs up to an intermediate stage, and the element count cannot tell it from `[1, S, S]`.
  assertDepthShape(depthGraph(), depthConfig, "depth map shape");
  assertThrows(
    () =>
      assertDepthShape(
        depthGraph({ outputShape: [1, 1, 518, 518] }),
        depthConfig,
        "depth map shape",
      ),
    Error,
    "expected [1, 518, 518]",
  );
});

test("assertDepthShape: also rejects a shorter declaration whose leading axes match (a per-axis comparison alone would pass it)", () => {
  // Every declared axis of `[1, 518]` matches the expectation, so a check that ignores `shape.length` lets it through.
  assertThrows(
    () => assertDepthShape(depthGraph({ outputShape: [1, 518] }), depthConfig, "depth map shape"),
    Error,
    "graph output 'depth' has shape [1, 518]",
  );
});

test("assertDepthShape: rejects when the output name has no declaration in values", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 518, 518] }],
    outputs: ["depth"],
    values: {},
  });
  assertThrows(
    () => assertDepthShape(model, depthConfig, "depth map shape"),
    Error,
    "depth-anything: graph output 'depth' has no declaration (depth map shape)",
  );
});

// ---- Preprocessing wiring (bicubic + ImageNet statistics) ----------------------------------

const IMAGE_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
const IMAGE_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

/** A `pipelineConfig` that declares the case's size as is (the constants are the real distribution's). */
const configFor = (width: number, height: number) =>
  parseDepthAnythingPipelineConfig({
    imageWidth: width,
    imageHeight: height,
    imageMean: IMAGE_MEAN,
    imageStd: IMAGE_STD,
    interpolation: "bicubic",
  });

/** RGB8 filled with a different value per channel (a wrong order of the statistics shows in the values). */
const solid = (
  width: number,
  height: number,
  color: readonly [number, number, number],
): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let index = 0; index < width * height; index += 1) {
    for (let channel = 0; channel < 3; channel += 1) data[index * 3 + channel] = color[channel];
  }
  return { data, width, height };
};

/** A checkerboard + a diagonal ramp (just enough structure for bicubic and bilinear to give different values). */
const textured = (width: number, height: number): Rgb8Image => {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 3;
      data[index] = (x + y) % 2 === 0 ? 250 : 5;
      data[index + 1] = Math.round((255 * x) / Math.max(1, width - 1));
      data[index + 2] = Math.round((255 * y) / Math.max(1, height - 1));
    }
  }
  return { data, width, height };
};

test("preprocessing wiring: for a uniform image the declared mean / std are applied per channel", () => {
  // The resize weights sum to 1, so a uniform input stays uniform and only normalization remains. R, G and B have
  // different values, so a wrong order of the statistics (BGR / mean and std swapped) fails here.
  const color = [10, 200, 90] as const;
  const got = preprocessPixelValues(configFor(4, 2), solid(3, 5, color));
  assertEquals(got.length, 3 * 4 * 2);
  for (let channel = 0; channel < 3; channel += 1) {
    const expected = (color[channel] - IMAGE_MEAN[channel] * 255) / (IMAGE_STD[channel] * 255);
    for (let index = 0; index < 4 * 2; index += 1) {
      assertAlmostEquals(got[channel * 8 + index], expected, 1e-5, `channel ${channel}`);
    }
  }
});

test("preprocessing wiring: swapping imageWidth / imageHeight gives a different result", () => {
  // Ensures the 2 declared fields reach the right argument positions (`resizeRgb8` takes `(width, height)`). A
  // square cannot detect this in principle, so a non-square size is used.
  const image = textured(12, 9);
  const got = preprocessPixelValues(configFor(6, 4), image);
  const swapped = preprocessPixelValues(configFor(4, 6), image);
  assertEquals(got.length, swapped.length, "the element count is the same after transposing");
  let matched = 0;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] === swapped[index]) matched += 1;
  }
  assert(matched < got.length, "every sample matched after transposing (the check has no effect)");
});

test("preprocessing wiring: runs through bicubic (has not fallen back to the default bilinear)", () => {
  // The default of `resizeRgb8`'s `filter` argument is bilinear (the value of the 2 existing families), so
  // forgetting to pass it passes the type check, and `pixel_values` drift with shape and range still matching
  // (measured 0.59 = 34 times one uint8 LSB). **Not matching** the same path built with bilinear is the evidence
  // of the filter choice.
  const image = textured(12, 9);
  const config = configFor(6, 4);
  const got = preprocessPixelValues(config, image);
  const bilinear = normalizeToNchw(resizeRgb8(image, 6, 4, "bilinear"), IMAGE_MEAN, IMAGE_STD);
  assertEquals(got.length, bilinear.length);
  let differing = 0;
  for (let index = 0; index < got.length; index += 1) {
    if (got[index] !== bilinear[index]) differing += 1;
  }
  assert(
    differing > 0,
    "the bicubic setting gave the same values as bilinear (it has fallen back to the default)",
  );
});

// ---- Postprocessing (depth map → original resolution, values untouched) ----------------------------

test("postprocessing: at an identity size the values pass through as is (no rounding, no normalization)", () => {
  const depth = Float32Array.from([0, 0.25, 1.5, 4.75, 2, 0]);
  const map = resampleDepth(depth, 3, 2, 3, 2);
  assertEquals(map.width, 3);
  assertEquals(map.height, 2);
  assertEquals([...map.data], [...depth]);
});

test("postprocessing: the returned size is the **input image's** (not the graph's)", () => {
  const map = resampleDepth(new Float32Array(4 * 4), 4, 4, 7, 3);
  assertEquals(map.width, 7);
  assertEquals(map.height, 3);
  assertEquals(map.data.length, 21);
});

test("postprocessing: keeps the scale of relative depth (does not fold into [0, 1] — a public API decision)", () => {
  // Adding normalization would still pass with matching shape and non-negativity, so this is pinned in the
  // "fails if folded" form. A min-max folded map always contains 0 and 1, and the top value 4.75 would not survive.
  const depth = Float32Array.from([2, 2.5, 3, 4.75]);
  const map = resampleDepth(depth, 4, 1, 4, 1);
  assertEquals(map.data[0], 2, "the bottom value dropped to 0 (min-max normalization is applied)");
  assertEquals(map.data[3], 4.75, "the top value dropped to 1 (min-max normalization is applied)");
});

test("postprocessing: stays non-negative when upscaled (non-negative weights summing to 1)", () => {
  // The property "the far background is exactly 0" produced by the ReLU at the end of the head must not dip
  // below 0 in postprocessing. Swapping in a kernel with negative lobes (bicubic) makes this fail — the only
  // point where the module doc's "the step back to the original resolution is bilinear" shows in the values.
  const depth = Float32Array.from([0, 0, 5, 5]);
  const map = resampleDepth(depth, 4, 1, 16, 1);
  for (const value of map.data) assert(value >= 0, `negative depth ${value} appeared`);
  assert(map.data[15] > map.data[0], "monotonicity was lost (left and right are swapped)");
});

test("postprocessing: a horizontal-only gradient stays horizontal-only when upscaled (detects swapped axes)", () => {
  const width = 6;
  const height = 4;
  const plane = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) plane[y * width + x] = x / (width - 1);
  }
  const map = resampleDepth(plane, width, height, 3, 9);
  for (let y = 1; y < 9; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      assertAlmostEquals(map.data[y * 3 + x], map.data[x], 1e-7, `row ${y} column ${x}`);
    }
  }
  assert(map.data[0] < map.data[2], "horizontal monotonicity was lost (the axes are swapped)");
});

test("postprocessing: non-finite depth fails before the resize with pixel coordinates (no all-black map)", () => {
  // The visualization recipe in the module doc scans with `value < min` / `value > max`, so NaN slips past the
  // comparisons, `span` becomes non-positive, and **every pixel 0 = a uniformly black depth map** comes out
  // looking like a normal value. This has a check point of the same form, word for word, as BiRefNet's
  // `matteFromLogits`.
  const width = 4;
  const height = 3;
  for (const [index, bad] of [
    [6, Number.NaN],
    [11, Number.POSITIVE_INFINITY],
  ] as const) {
    const plane = new Float32Array(width * height).fill(1);
    plane[index] = bad;
    // Pins that the coordinates appear in the message — a check on the element count alone would stay green even
    // if the check moved **after** the resize (it would look after NaN spreads to neighbours, and the source could
    // not be located).
    assertThrows(
      () => resampleDepth(plane, width, height, width, height),
      Error,
      `depth-anything: depth value (x=${index % width}, y=${Math.floor(index / width)}) is non-finite`,
    );
    // The same coordinates for a call that upscales (the path where `resizePlaneF32` spreads NaN through weighted sums).
    assertThrows(
      () => resampleDepth(plane, width, height, 9, 7),
      Error,
      `(x=${index % width}, y=${Math.floor(index / width)})`,
    );
  }
  // Counterpart: a finite plane passes as is (shows the check is not always true).
  const clean = Float32Array.from([0, 0.25, 1.5, 4.75, 2, 0]);
  assertEquals([...resampleDepth(clean, 3, 2, 3, 2).data], [...clean]);
});
