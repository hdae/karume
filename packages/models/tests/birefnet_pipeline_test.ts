// The **construction guards** and the **wiring of pre- and postprocessing** of `Pipeline`. Only the
// part that needs neither a GPU nor real assets (the comparison on a real GPU belongs to
// `packages/runtime/tests/e2e_birefnet_test.ts` (golden input) and `e2e_birefnet_real_test.ts`
// (preprocessing parity and discrimination on real images) — not duplicated here).
//
// It covers four points:
//
// ① `fromAssets` rejects **manifest contract violations before acquiring a GPU** (the MUST stated
//   by `openBirefnetState` in `src/birefnet/pipeline.ts`). The observation setup is the same as on
//   the SigLIP2 side — **every case has all the containers**, and the order of the checks itself is
//   pinned by two things:
//     - a contract-violation case fails with "that violation's wording" = the manifest check comes
//       first even when the assets are all there
//     - a valid manifest + an empty Record fails with `component 'matte' has no container` (the
//       opener's diagnostic).
//    NOTE: the containers are opened **before** admission (`assetComponentOpener` opens every
//    component first so that it can return a synchronous opener — the seam of ADR 0109).
//
// ② The **rejection paths** of the comparison with the graph declaration (`assertStaticDim` /
//   `assertMatteShape`). Inside `fromAssets` they cannot be reached without a real GPU and real
//   assets, so the checks are called directly (`tests/helpers/stub-model.ts` builds a
//   declaration-only `GraphOwner`). Even if a check's own axis numbers or expected shapes drifted,
//   it would stay green as long as only the normal path runs, so **rejecting a broken declaration
//   by name** is exercised every time.
//
// ③ The preprocessing wiring (`preprocessPixelValues`) goes through `resizeRgb8` →
//   `normalizeToNchw` as the `pipelineConfig` declares. **The reference is the fixture of the
//   Python source of truth** (`fixtures/image-preprocess/parity.json`), so all this checks is
//   "whether the four declared fields reach the right argument positions". A non-square case is
//   used because a mix-up of `imageWidth` / `imageHeight` is **undetectable in principle** with a
//   square.
//
// ④ The **stage order and quantisation** of the postprocessing (`matteFromLogits` —
//   `src/birefnet/matte.ts`), and that `segment` applies the procedure the distribution declares
//   (`pipelineConfig.postprocess`). Applying sigmoid before resize is the same as both series'
//   upstreams, and it holds for both procedures (HR = `u8-pil-bicubic`, Lucida = `float-bilinear`).
//   The quantisation (round half up) differs from both series' upstreams (truncation) (ADR 0125 —
//   the actual procedures of both upstreams are in the module doc of `matte.ts`). Swapping the
//   stage order still passes with matching shape and range (a wrong order only shifts α by a few
//   tens), so there is an asymmetric case that fails when the order is swapped.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import {
  assert,
  assertAlmostEquals,
  assertEquals,
  assertNotEquals,
  assertRejects,
  assertThrows,
} from "@std/assert";
import { parseManifest } from "@karume/hub";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";
import {
  assertMatteShape,
  assertStaticDim,
  Pipeline,
  preprocessPixelValues,
} from "../src/birefnet/pipeline.ts";
import { MATTE_PROCEDURES, matteFromLogits, type MatteProcedure } from "../src/birefnet/matte.ts";
import { resizePlaneF32, type Rgb8Image } from "../src/image/preprocess.ts";
import { stubModel } from "./helpers/stub-model.ts";
import { fakeDevice, fakeGpuContext } from "../../runtime/tests/helpers/fake-gpu.ts";
import { declaredContainer, partAssets, tensorlessContainer } from "./helpers/container-fixture.ts";

/**
 * A declaration-only `matte` container (reproduced up to the shape the family's checks read; it is
 * never run).
 */
const COMPONENT = partAssets(
  "matte",
  await tensorlessContainer("matte", {
    inputs: [{ name: "pixel_values", shape: [1, 3, 1024, 1024] }],
    output: { name: "y", shape: [1, 1, 1024, 1024] },
  }),
);

/**
 * The `pipelineConfig` of `models/karume-birefnet-hr/karume.json` (the shape rebuilt for
 * `birefnet/2`, 6 fields).
 */
const PIPELINE_CONFIG: Record<string, unknown> = {
  imageWidth: 1024,
  imageHeight: 1024,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
  postprocess: "u8-pil-bicubic",
};

/**
 * The skeleton of the distribution (only the fields the checks need). `patch` overwrites the
 * contents of `models["hr"]`.
 */
const manifestText = (patch: Record<string, unknown> = {}): string =>
  JSON.stringify({
    format: "karume/5",
    generator: "karume/0.2.2",
    defaultModel: "hr",
    models: {
      hr: {
        pipeline: "birefnet/2",
        weights: { matte: { f32: declaredContainer("hr/matte/model.f32") } },
        // Running it needs only one graph (no tokenizer, no tables).
        assets: {},
        quants: { f32: { weights: { matte: "f32" }, session: {} } },
        defaultQuant: "f32",
        pipelineConfig: PIPELINE_CONFIG,
        ...patch,
      },
    },
  });

const emptyAssets = {} as Record<string, Uint8Array<ArrayBuffer>>;

test("fromAssets: a nonexistent model fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { model: "nope" }),
    Error,
    "model 'nope' is not in the manifest",
  );
});

test("fromAssets: rejects a manifest whose pipeline contract name is not birefnet", async () => {
  const manifest = parseManifest(manifestText({ pipeline: "siglip2/1" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "the manifest's pipeline is 'siglip2/1'",
  );
});

test("fromAssets: an unknown major fails loudly (checking it is models' responsibility — ADR 0038 §1)", async () => {
  // The only check that stops the silent degradation of "old implementation × new repository". The
  // hub does not check the major.
  const manifest = parseManifest(manifestText({ pipeline: "birefnet/3" }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "major of pipeline 'birefnet/3' is not supported",
  );
});

test("fromAssets: rejects the old major too (birefnet/1 — a distribution that declares no postprocessing procedure)", async () => {
  // A major-1 distribution has no `postprocess`. Without the major check it would only fail on the
  // missing pipelineConfig field, and the remedy "rebuild the distribution" would not be readable
  // from the message.
  const legacy: Record<string, unknown> = { ...PIPELINE_CONFIG };
  delete legacy.postprocess;
  const manifest = parseManifest(manifestText({ pipeline: "birefnet/1", pipelineConfig: legacy }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "the major of pipeline 'birefnet/1' is not supported (this implementation reads birefnet/2)",
  );
});

test("fromAssets: a nonexistent quant fails with the list of available ones", async () => {
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }, { quant: "nope" }),
    Error,
    "quant 'nope' is not in the manifest",
  );
});

test("fromAssets: an unknown pipelineConfig key fails at construction", async () => {
  // No path where a misspelling (`image_mean` for `imageMean`) silently degrades to a default.
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, image_mean: [0.5, 0.5, 0.5] } }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig: unknown key 'image_mean'",
  );
});

test("fromAssets: does not accept a distribution that declares an interpolation other than bilinear", async () => {
  const manifest = parseManifest(
    manifestText({ pipelineConfig: { ...PIPELINE_CONFIG, interpolation: "bicubic" } }),
  );
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig.interpolation: this implementation supports only 'bilinear'",
  );
});

test("fromAssets: a pipelineConfig that declares no postprocessing procedure fails at construction", async () => {
  const missing: Record<string, unknown> = { ...PIPELINE_CONFIG };
  delete missing.postprocess;
  const manifest = parseManifest(manifestText({ pipelineConfig: missing }));
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: COMPONENT }),
    Error,
    "pipelineConfig.postprocess: missing",
  );
});

test("fromAssets: without a component's container it fails with the spellings of both forms (the opener's diagnostic)", async () => {
  // The flip side of the eight cases above. A Record without all the containers fails on **the
  // missing component** (not with the manifest's wording) = the contrapositive showing that the
  // cases above are not dragged along by missing assets.
  const manifest = parseManifest(manifestText());
  await assertRejects(
    () => Pipeline.fromAssets({ manifest, assets: emptyAssets }),
    Error,
    "component 'matte' has no container",
  );
});

// ---- Comparison with the graph declaration (rejection paths) ----------------------------------

/** The same declaration as the real distribution (hr). `patch` breaks a single point. */
const matteGraph = (
  patch: {
    readonly inputShape?: readonly (number | string)[];
    readonly outputShape?: readonly (number | string)[];
  } = {},
) =>
  stubModel({
    inputs: [{ name: "pixel_values", shape: patch.inputShape ?? [1, 3, 1024, 1024] }],
    outputs: ["matte"],
    values: { matte: patch.outputShape ?? [1, 1, 1024, 1024] },
  });

const birefnetConfig = parseBirefnetPipelineConfig(PIPELINE_CONFIG);

test("assertStaticDim: a graph as declared passes, and a resolution mismatch fails by name", () => {
  // A 2048² series placed in the 1024² slot. The preprocessing resizes to the declared dimensions,
  // so the host side goes through to the end, and what fails is the Session's shape check = which
  // side is right cannot be read.
  assertStaticDim(matteGraph(), "pixel_values", 3, birefnetConfig.imageWidth, "imageWidth");
  const error = assertThrows(
    () =>
      assertStaticDim(
        matteGraph({ inputShape: [1, 3, 2048, 2048] }),
        "pixel_values",
        3,
        birefnetConfig.imageWidth,
        "imageWidth",
      ),
    Error,
    "birefnet: imageWidth — graph input 'pixel_values' axis 3 is 2048",
  );
  assert(error.message.includes("pipelineConfig has 1024"), error.message);
});

test("assertStaticDim: rejects a missing input name", () => {
  assertThrows(
    () => assertStaticDim(matteGraph(), "pixel", 3, birefnetConfig.imageWidth, "imageWidth"),
    Error,
    "birefnet: graph input 'pixel' is missing (imageWidth)",
  );
});

test("assertMatteShape: an output baked as multi-scale fails on its shape, not its element count", () => {
  // `[1, 3, S, S]` is an export with the intermediate predictions. A check that looks only at the
  // element count would let **other values through as α**.
  assertMatteShape(matteGraph(), birefnetConfig, "matte shape");
  assertThrows(
    () =>
      assertMatteShape(
        matteGraph({ outputShape: [1, 3, 1024, 1024] }),
        birefnetConfig,
        "matte shape",
      ),
    Error,
    "expected [1, 1, 1024, 1024]",
  );
});

test("assertMatteShape: rejects a shorter declaration whose leading axes match (a per-axis comparison alone would pass it)", () => {
  // In `[1, 1, 1024]` every declared axis matches the expectation, so a check that does not look at
  // `shape.length` lets it through (the very reason to check the rank).
  assertThrows(
    () =>
      assertMatteShape(matteGraph({ outputShape: [1, 1, 1024] }), birefnetConfig, "matte shape"),
    Error,
    "graph output 'matte' has shape [1, 1, 1024]",
  );
});

test("assertMatteShape: rejects an output name that has no declaration in values", () => {
  const model = stubModel({
    inputs: [{ name: "pixel_values", shape: [1, 3, 1024, 1024] }],
    outputs: ["matte"],
    values: {},
  });
  assertThrows(
    () => assertMatteShape(model, birefnetConfig, "matte shape"),
    Error,
    "birefnet: graph output 'matte' has no declaration (matte shape)",
  );
});

// ---- Preprocessing wiring (against the fixture of the Python source of truth) ----------------

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

/** A non-square case downscaled 4× vertically and upscaled 3× horizontally (24×5 → 6×15). */
const MIXED_AXIS = "mixed-axis";

/**
 * The tolerance of every path. Derived at the top of `image_preprocess_test.ts` (1 LSB of uint8 in
 * f32).
 */
const PIXEL_ATOL = 7.85e-3;

const parityCase = (name: string): ParityCase => {
  const found = fixture.cases.find((entry) => entry.name === name);
  assert(found !== undefined, `the fixture has no '${name}' (the generator script changed)`);
  return found;
};

const imageOf = (entry: ParityCase): Rgb8Image => ({
  data: Uint8Array.from(entry.input),
  width: entry.width,
  height: entry.height,
});

/** A `pipelineConfig` that declares the case's dimensions and constants as they are. */
const configFor = (entry: ParityCase, swapped = false) =>
  parseBirefnetPipelineConfig({
    imageWidth: swapped ? entry.outHeight : entry.outWidth,
    imageHeight: swapped ? entry.outWidth : entry.outHeight,
    imageMean: fixture.constants.imageMean,
    imageStd: fixture.constants.imageStd,
    interpolation: "bilinear",
    // The preprocessing does not read the postprocessing declaration (the field is required, so the
    // real HR value is used).
    postprocess: "u8-pil-bicubic",
  });

test("preprocessing wiring: matches the Python source of truth with the declared dimensions, mean and std (non-square)", () => {
  const entry = parityCase(MIXED_AXIS);
  const got = preprocessPixelValues(configFor(entry), imageOf(entry));
  assertEquals(got.length, entry.pixelValues.length);
  for (let index = 0; index < got.length; index += 1) {
    assertAlmostEquals(got[index], entry.pixelValues[index], PIXEL_ATOL, `sample ${index}`);
  }
});

test("preprocessing wiring: swapping imageWidth / imageHeight gives a different result", () => {
  // The guarantee that the case above is not vacuous — transposing a non-square declaration changes
  // the values (`resizeRgb8` takes `(width, height)` in that order).
  const entry = parityCase(MIXED_AXIS);
  assert(
    entry.outWidth !== entry.outHeight,
    "a transpose cannot be detected without a non-square case",
  );
  const swapped = preprocessPixelValues(configFor(entry, true), imageOf(entry));
  assertEquals(
    swapped.length,
    entry.pixelValues.length,
    "the element count is the same after transposing",
  );
  let matched = 0;
  for (let index = 0; index < swapped.length; index += 1) {
    if (Math.abs(swapped[index] - entry.pixelValues[index]) <= PIXEL_ATOL) matched += 1;
  }
  assert(
    matched < swapped.length,
    "every sample matched even after transposing (the check has no effect)",
  );
});

// ---- Postprocessing (logit → sigmoid → original resolution → 8-bit α) ------------------------

/** 1 LSB after rounding to 8 bits. */
const ALPHA_LSB = 1;

test("postprocessing: an identity resize quantises the sigmoid to 8 bits as is (both procedures)", () => {
  // logit 0 → 0.5 → 127.5 → rounded to nearest gives 128 (truncation would give 127 — the point
  // where the decision to round rather than truncate like both upstreams (HR's ToPILImage, Lucida's
  // `astype("uint8")`) shows in the value). `u8-pil-bicubic` skips the resample of an axis whose
  // size is unchanged, so at identity both procedures give the same value.
  for (const procedure of MATTE_PROCEDURES) {
    const matte = matteFromLogits(Float32Array.from([0, 40, -40, 1]), 2, 2, 2, 2, procedure);
    assertEquals(matte.width, 2);
    assertEquals(matte.height, 2);
    assertEquals(
      [...matte.data],
      [128, 255, 0, Math.round((1 / (1 + Math.E ** -1)) * 255)],
      procedure,
    );
  }
});

test("postprocessing: the result has the **input image's** dimensions (not the graph's; both procedures)", () => {
  for (const procedure of MATTE_PROCEDURES) {
    const matte = matteFromLogits(new Float32Array(4 * 4), 4, 4, 7, 3, procedure);
    assertEquals(matte.width, 7, procedure);
    assertEquals(matte.height, 3, procedure);
    assertEquals(matte.data.length, 21, procedure);
  }
});

test("postprocessing: sigmoid is applied **before** resize (swapping the order fails)", () => {
  // A 2×1 → 1×1 downscale gives the two points equal weight with either kernel, so the resize is
  // exactly the mean of the two points.
  //   sigmoid first (float-bilinear): (σ(0) + σ(10)) / 2 = (0.5 + 0.9999546) / 2 = 0.7499773 → 191
  //   sigmoid first (u8-pil-bicubic): in 8 bits (128 + 255) / 2 = 191.5 → 192 (fixed-point
  //                                   round half up)
  //   resize first                  : σ((0 + 10) / 2) = σ(5) = 0.9933 → 253
  // They are more than 60 apart, so a wrong order cannot produce a "plausible" matte.
  const expected: Readonly<Record<MatteProcedure, number>> = {
    "float-bilinear": 191,
    "u8-pil-bicubic": 192,
  };
  for (const procedure of MATTE_PROCEDURES) {
    const matte = matteFromLogits(Float32Array.from([0, 10]), 2, 1, 1, 1, procedure);
    assertEquals([...matte.data], [expected[procedure]], procedure);
  }
});

test("resizePlaneF32: the same dimensions pass the values through as is (no rounding, no degradation)", () => {
  const plane = Float32Array.from([0.1, 0.25, 0.5, 0.75, 0.9, 1]);
  const out = resizePlaneF32(plane, 3, 2, 3, 2);
  assertEquals([...out], [...plane]);
});

test("resizePlaneF32: a uniform plane stays uniform when resized (the weights sum to 1)", () => {
  const plane = new Float32Array(5 * 7).fill(0.375);
  for (const [width, height] of [
    [3, 2],
    [11, 13],
    [5, 7],
  ] as const) {
    const out = resizePlaneF32(plane, 5, 7, width, height);
    assertEquals(out.length, width * height);
    for (const value of out) assertAlmostEquals(value, 0.375, 1e-6);
  }
});

test("resizePlaneF32: the downscale weights match Pillow's support (compared with a hand calculation)", () => {
  // A 4 → 2 downscale: scale = 2, so the support stretches 2×, and the weights of output 0 are
  // triangle(-0.25) / triangle(0.25) / triangle(0.75) = 0.75 / 0.75 / 0.25 normalised by the sum
  // 1.75. Output 1 is the mirrored order. A reversal cannot be detected without an **asymmetric
  // input**.
  const out = resizePlaneF32(Float32Array.from([0, 0, 0, 1]), 4, 1, 2, 1);
  assertAlmostEquals(out[0], 0, 1e-7);
  assertAlmostEquals(out[1], 0.75 / 1.75, 1e-7);
});

test("resizePlaneF32: a horizontal-only gradient stays horizontal-only after resize (detects an axis mix-up)", () => {
  const width = 6;
  const height = 4;
  const plane = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) plane[y * width + x] = x / (width - 1);
  }
  const out = resizePlaneF32(plane, width, height, 3, 9);
  for (let y = 1; y < 9; y += 1) {
    for (let x = 0; x < 3; x += 1) {
      assertAlmostEquals(out[y * 3 + x], out[x], 1e-7, `row ${y} column ${x}`);
    }
  }
  assert(out[0] < out[2], "horizontal monotonicity was lost (the axes are swapped)");
});

test("resizePlaneF32: rejects a length mismatch (no silently reading part of it)", () => {
  assertThrows(() => resizePlaneF32(new Float32Array(5), 3, 2, 3, 2), Error, "Plane length 5");
  assertThrows(() => resizePlaneF32(new Float32Array(6), 3, 2, 0, 2), RangeError, "Output width");
});

test("postprocessing: α uses the whole 8-bit range (the saturated side is not crushed by 1 LSB)", () => {
  // Guards against vacuity — avoids the shape where the cases above would all pass even around 128.
  for (const procedure of MATTE_PROCEDURES) {
    const matte = matteFromLogits(Float32Array.from([-20, -2, 2, 20]), 4, 1, 4, 1, procedure);
    assertEquals(matte.data[0], 0, procedure);
    assertEquals(matte.data[3], 255, procedure);
    assert(
      matte.data[1] > 0 && matte.data[1] < 128 - ALPHA_LSB,
      `σ(−2) does not land in the middle (${procedure})`,
    );
    assert(
      matte.data[2] > 128 + ALPHA_LSB && matte.data[2] < 255,
      `σ(2) does not land in the middle (${procedure})`,
    );
  }
});

// ---- Postprocessing wiring of segment (a fake Session lets it run without a GPU) ----------------
//
// Pins that the procedure `segment` applies to the logits follows the distribution's declaration
// (`pipelineConfig.postprocess`) (ADR 0125 — the distribution declares each series' procedure).
// Both declarations run on an input whose values differ per declaration, so if the pipeline ignored
// the declaration and stuck to one procedure, one of the declarations would fail. A fake Session
// and a fake GPU suffice (only `run` is touched), so the state is built directly and passed to the
// private constructor (the same approach as `vowel_detector_pipeline_test.ts`).

/**
 * The graph's dimensions (non-square) and the input image's dimensions (upscaling — the direction
 * in which the two procedures' values differ).
 */
const FAKE_GRAPH = { width: 6, height: 5 };
const FAKE_IMAGE = { width: 13, height: 9 };

/** Logits with a step and a slope (an input on which the two procedures' values differ). */
const fakeLogits = (): Float32Array<ArrayBuffer> =>
  Float32Array.from(
    { length: FAKE_GRAPH.width * FAKE_GRAPH.height },
    (_, index) => (index % FAKE_GRAPH.width < 3 ? -9 : 9) + (index % 7) - 3,
  );

/** A pipeline with a fake Session (`run` returns fixed logits) that declares `postprocess`. */
const pipelineWithFakeSession = (
  logits: Float32Array<ArrayBuffer>,
  postprocess: MatteProcedure,
): Pipeline => {
  const state = {
    gpu: fakeGpuContext(fakeDevice()),
    ownsGpu: false,
    config: parseBirefnetPipelineConfig({
      ...PIPELINE_CONFIG,
      imageWidth: FAKE_GRAPH.width,
      imageHeight: FAKE_GRAPH.height,
      postprocess,
    }),
    matte: { graph: { outputs: ["matte"] } },
    session: {
      run: () =>
        Promise.resolve({
          matte: { dtype: "f32", shape: [1, 1, FAKE_GRAPH.height, FAKE_GRAPH.width], data: logits },
        }),
      diagnostics: () => ({}),
      dispose: () => Promise.resolve(),
    },
  };
  return Reflect.construct(Pipeline, [state]);
};

test("segment: the procedure applied to the logits follows the distribution's declaration (bit-identical to matteFromLogits with the explicit procedure)", async () => {
  const logits = fakeLogits();
  const image: Rgb8Image = {
    data: new Uint8Array(FAKE_IMAGE.width * FAKE_IMAGE.height * 3).fill(128),
    ...FAKE_IMAGE,
  };
  const procedure = (name: MatteProcedure) => [
    ...matteFromLogits(
      logits,
      FAKE_GRAPH.width,
      FAKE_GRAPH.height,
      FAKE_IMAGE.width,
      FAKE_IMAGE.height,
      name,
    ).data,
  ];
  // Guards against vacuity — if the two procedures gave the same value on this input, a mix-up of
  // procedures could not be caught.
  assertNotEquals(procedure("u8-pil-bicubic"), procedure("float-bilinear"));

  for (const declared of MATTE_PROCEDURES) {
    const matte = await pipelineWithFakeSession(logits, declared).segment(image);
    assertEquals([matte.width, matte.height], [FAKE_IMAGE.width, FAKE_IMAGE.height], declared);
    assertEquals([...matte.data], procedure(declared), declared);
  }
});
