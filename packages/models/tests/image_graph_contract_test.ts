/**
 * The pre-build checks of the 3 image families (siglip2 / birefnet / depth-anything) look not only
 * at axis sizes but also at **the input dtype** and **the shape of SigLIP2's output**. Needs
 * neither a GPU nor real assets.
 *
 * A check that looks only at axis sizes lets through a graph with the same shape but another dtype
 * (an i32 `pixel_values`) and a SigLIP2 output built with batch 2 (`[2, hiddenDim]`). Going on from
 * there, it fails after creating the GPU and Session and reading the weights (for SigLIP2's output,
 * after one run), with a message that does not point at the mixed-up asset.
 *
 * The observation setup: a container with graph declarations only is passed to `fromAssets`, and no
 * `gpu` is passed. The family checks run in the admission **before** the GPU is acquired, so a
 * failure here never touches the GPU. A graph that slips past the checks makes the build proceed to
 * `acquireGpu` and either fail with another message (no GPU, the Session's checks) or pass.
 */

import { assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type IrDeclaration, parseIrDeclarationValue } from "@karume/runtime";
import { type Manifest, parseManifest } from "@karume/hub";
import * as birefnet from "../birefnet.ts";
import * as depthAnything from "../depth-anything.ts";
import * as siglip2 from "../siglip2.ts";
import { declaredContainer, partAssets, writeContainer } from "./helpers/container-fixture.ts";

/** The built resolution (square, 2×2). */
const SIZE = 2;

/**
 * A declaration with one input and one output (it is never run, so what it computes does not
 * matter). So that the IR's op contracts hold whatever the input dtype, it first `cast`s to f32 and
 * then `reshape`s to the output shape. When the output dtype is not f32, it `cast`s to that dtype
 * at the end.
 */
const declaration = (
  inputDtype: string,
  output: string,
  outputShape: readonly number[],
  outputDtype = "f32",
): IrDeclaration => {
  const reshaped = outputDtype === "f32" ? output : "reshaped";
  return parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["cast", "reshape"] },
    symbols: [],
    inputs: [{ name: "pixel_values", dtype: inputDtype, shape: [1, 3, SIZE, SIZE] }],
    outputs: [output],
    initializers: {},
    values: {
      as_f32: { dtype: "f32", shape: [1, 3, SIZE, SIZE] },
      [reshaped]: { dtype: "f32", shape: outputShape },
      [output]: { dtype: outputDtype, shape: outputShape },
    },
    states: {},
    nodes: [
      { op: "cast", ins: ["pixel_values"], outs: ["as_f32"], attrs: { to: "f32" } },
      { op: "reshape", ins: ["as_f32"], outs: [reshaped], attrs: {} },
      ...(outputDtype === "f32"
        ? []
        : [{ op: "cast", ins: [reshaped], outs: [output], attrs: { to: outputDtype } }]),
    ],
  });
};

/**
 * Writes a single-component container and folds it into the full-surface key (`<component>[i]`).
 */
const assetsOf = async (key: string, graph: IrDeclaration) =>
  partAssets(
    key,
    await writeContainer({
      graphs: { [key]: graph },
      consts: [],
      weights: [],
      assets: [],
      provenance: { license: "test" },
    }),
  );

const manifestOf = (pipeline: string, key: string, config: Record<string, unknown>): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "test",
      models: {
        test: {
          pipeline,
          weights: { [key]: { f32: declaredContainer(`${key}/model.f32`) } },
          assets: {},
          quants: { f32: { weights: { [key]: "f32" }, session: {} } },
          defaultQuant: "f32",
          pipelineConfig: config,
        },
      },
    }),
  );

const IMAGE_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
};

const HIDDEN = 12;

const buildSiglip2 = async (
  inputDtype: string,
  outputShape: readonly number[],
  outputDtype = "f32",
) =>
  await siglip2.Pipeline.fromAssets({
    manifest: manifestOf("siglip2/1", "vision", {
      ...IMAGE_CONFIG,
      hiddenDim: HIDDEN,
      interpolation: "bilinear",
    }),
    assets: await assetsOf(
      "vision",
      declaration(inputDtype, "pooler_output", outputShape, outputDtype),
    ),
  });

describe("pre-build checks of the 3 image families", () => {
  describe("when the graph's pixel_values is not f32", () => {
    it("SigLIP2 rejects the input dtype by name before acquiring the GPU", async () => {
      await assertRejects(
        () => buildSiglip2("i32", [1, HIDDEN]),
        Error,
        "siglip2: preprocessing's pixel_values — graph input 'pixel_values' dtype is i32, the preprocessing writes f32",
      );
    });

    it("BiRefNet rejects the input dtype by name before acquiring the GPU", async () => {
      await assertRejects(
        async () =>
          await birefnet.Pipeline.fromAssets({
            manifest: manifestOf("birefnet/2", "matte", {
              ...IMAGE_CONFIG,
              interpolation: "bilinear",
              postprocess: "u8-pil-bicubic",
            }),
            assets: await assetsOf("matte", declaration("i32", "logits", [1, 1, SIZE, SIZE])),
          }),
        Error,
        "birefnet: preprocessing's pixel_values — graph input 'pixel_values' dtype is i32",
      );
    });

    it("Depth Anything rejects the input dtype by name before acquiring the GPU", async () => {
      await assertRejects(
        async () =>
          await depthAnything.Pipeline.fromAssets({
            manifest: manifestOf("depth-anything/1", "depth", {
              ...IMAGE_CONFIG,
              interpolation: "bicubic",
            }),
            assets: await assetsOf("depth", declaration("i32", "predicted_depth", [1, SIZE, SIZE])),
          }),
        Error,
        "depth-anything: preprocessing's pixel_values — graph input 'pixel_values' dtype is i32",
      );
    });
  });

  describe("when the graph output is not f32 (the postprocessing reads f32)", () => {
    it("BiRefNet rejects the output dtype by name before acquiring the GPU", async () => {
      await assertRejects(
        async () =>
          await birefnet.Pipeline.fromAssets({
            manifest: manifestOf("birefnet/2", "matte", {
              ...IMAGE_CONFIG,
              interpolation: "bilinear",
              postprocess: "u8-pil-bicubic",
            }),
            assets: await assetsOf(
              "matte",
              declaration("f32", "logits", [1, 1, SIZE, SIZE], "i32"),
            ),
          }),
        Error,
        "birefnet: matte shape — graph output 'logits' dtype is i32, the postprocessing reads f32",
      );
    });

    it("Depth Anything rejects the output dtype by name before acquiring the GPU", async () => {
      await assertRejects(
        async () =>
          await depthAnything.Pipeline.fromAssets({
            manifest: manifestOf("depth-anything/1", "depth", {
              ...IMAGE_CONFIG,
              interpolation: "bicubic",
            }),
            assets: await assetsOf(
              "depth",
              declaration("f32", "predicted_depth", [1, SIZE, SIZE], "i32"),
            ),
          }),
        Error,
        "depth-anything: depth map shape — graph output 'predicted_depth' dtype is i32, the postprocessing reads f32",
      );
    });
  });

  describe("when SigLIP2's output is not [1, hiddenDim]", () => {
    it("rejects an output built with batch 2 before acquiring the GPU", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [2, HIDDEN]),
        Error,
        "siglip2: pooler_output — graph output 'pooler_output' is f32 [2, 12], expected f32 [1, hiddenDim]",
      );
    });

    // Axis 1 is set to hiddenDim so that it passes the existing axis check (assertOutputDim).
    // Otherwise it would fail at the axis check even without the rank check, and the rank check
    // could not be pinned.
    it("rejects a rank-3 output before acquiring the GPU", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [1, HIDDEN, 1]),
        Error,
        "siglip2: pooler_output — graph output 'pooler_output' is f32 [1, 12, 1], expected f32 [1, hiddenDim]",
      );
    });

    it("rejects a non-f32 output before acquiring the GPU", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [1, HIDDEN], "i32"),
        Error,
        "siglip2: pooler_output — graph output 'pooler_output' is i32 [1, 12], expected f32 [1, hiddenDim]",
      );
    });
  });
});
