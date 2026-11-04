/**
 * The `requiredLimits` check on the path where `fromAssets` omits `gpu` (the pipeline acquires its
 * own device) (real GPU).
 *
 * A distribution's quant declares the minimum device limits it needs (`requiredLimits` — ADR 0038
 * §7). On a shared GPU each family's admission checks them, and `fromPretrained` checks them
 * against the adapter's values before downloading, but the self-acquisition of `fromAssets` has no
 * point before acquisition to check them. Unless the limits of the device the build acquired are
 * checked, a shortfall in the declaration is delayed until the actual buffer checks at Session
 * build (with a low-level message), and when the declaration is stricter than the real demand the
 * check never happens at all.
 *
 * Here a **declaration no real machine can satisfy** (`Number.MAX_SAFE_INTEGER` — the largest value
 * the manifest accepts) is set, 4 families are built from declaration-only containers (no weights),
 * and they are expected to fail with the limits message. Every family check (graph declarations,
 * pipelineConfig) is made to pass, so the limits are the only reason to fail.
 *
 * NOTE: releasing the self-acquired device (the cleanup on failure) is not observed because
 * `acquireGpu` cannot be injected. The cleanup sits in the same catch as the feature check.
 */

import { assertRejects } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { type Manifest, parseManifest } from "@karume/hub";
import * as birefnet from "../birefnet.ts";
import * as depthAnything from "../depth-anything.ts";
import * as siglip2 from "../siglip2.ts";
import * as vowelDetector from "../vowel-detector.ts";
import { FEATURE_DIM, MEL_BINS, N_MELS, SAMPLE_RATE } from "../src/vowel-detector/features.ts";
import { LIPSYNC_CLASSES } from "../src/vowel-detector/postprocess.ts";
import {
  declaredContainer,
  partAssets,
  tensorlessContainer,
  type TensorlessGraphSpec,
} from "./helpers/container-fixture.ts";
import { writeSafetensors } from "./helpers/safetensors-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** The baked resolution (square 2×2). */
const SIZE = 2;

/** A declaration no real machine can satisfy. */
const UNSATISFIABLE = { maxBufferSize: Number.MAX_SAFE_INTEGER };

/** The message of the limits check (`assertRequiredLimitsSatisfied`). */
const LIMITS_MESSAGE = "this GPU does not meet the required device limits";

/** The skeleton of a distribution form (one component, one quant, unsatisfiable limits). */
const manifestOf = (
  pipeline: string,
  key: string,
  config: Record<string, unknown>,
  assets: Record<string, unknown> = {},
): Manifest =>
  parseManifest(
    JSON.stringify({
      format: "karume/5",
      generator: "karume/0.1.0",
      defaultModel: "test",
      models: {
        test: {
          pipeline,
          weights: { [key]: { f32: declaredContainer(`${key}/model.f32`) } },
          assets,
          quants: {
            f32: { weights: { [key]: "f32" }, session: {}, requiredLimits: UNSATISFIABLE },
          },
          defaultQuant: "f32",
          pipelineConfig: config,
        },
      },
    }),
  );

/** Folds one declaration-only container into full-surface keys (`<component>[i]`). */
const containerOf = async (key: string, spec: TensorlessGraphSpec) =>
  partAssets(key, await tensorlessContainer(key, spec));

/** The declaration of an image graph that takes `pixel_values [1,3,2,2]`. */
const imageGraph = (output: string, shape: readonly number[]): TensorlessGraphSpec => ({
  inputs: [{ name: "pixel_values", shape: [1, 3, SIZE, SIZE] }],
  output: { name: output, shape },
});

const IMAGE_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
};

/** One table row (built on the self-acquisition path). */
type Family = {
  readonly name: string;
  readonly build: () => Promise<{ dispose: () => Promise<void> }>;
};

const families: readonly Family[] = [
  {
    name: "siglip2.Pipeline",
    build: async () =>
      await siglip2.Pipeline.fromAssets({
        manifest: manifestOf("siglip2/1", "vision", {
          ...IMAGE_CONFIG,
          hiddenDim: 12,
          interpolation: "bilinear",
        }),
        assets: await containerOf("vision", imageGraph("pooler_output", [1, 12])),
      }),
  },
  {
    name: "birefnet.Pipeline",
    build: async () =>
      await birefnet.Pipeline.fromAssets({
        manifest: manifestOf("birefnet/2", "matte", {
          ...IMAGE_CONFIG,
          interpolation: "bilinear",
          postprocess: "u8-pil-bicubic",
        }),
        assets: await containerOf("matte", imageGraph("logits", [1, 1, SIZE, SIZE])),
      }),
  },
  {
    name: "depthAnything.Pipeline",
    build: async () =>
      await depthAnything.Pipeline.fromAssets({
        manifest: manifestOf("depth-anything/1", "depth", {
          ...IMAGE_CONFIG,
          interpolation: "bicubic",
        }),
        assets: await containerOf("depth", imageGraph("predicted_depth", [1, SIZE, SIZE])),
      }),
  },
  {
    name: "vowelDetector.Pipeline",
    build: async () =>
      await vowelDetector.Pipeline.fromAssets({
        manifest: manifestOf(
          "vowel-detector/1",
          "crnn",
          {
            sampleRate: SAMPLE_RATE,
            featureDim: FEATURE_DIM,
            classes: [...LIPSYNC_CLASSES],
            minFrames: 4,
            maxFrames: 60_000,
          },
          {
            mel_basis: {
              path: "crnn/features/mel-basis.safetensors",
              size: 16,
              sha256: "a".repeat(64),
            },
          },
        ),
        assets: {
          ...(await containerOf("crnn", {
            symbols: ["T"],
            inputs: [{ name: "features", shape: [1, "2T", FEATURE_DIM] }],
            output: { name: "logits", shape: [1, "T", LIPSYNC_CLASSES.length] },
          })),
          mel_basis: writeSafetensors(
            new Map([
              [
                "mel_basis",
                {
                  dtype: "F32" as const,
                  shape: [N_MELS, MEL_BINS],
                  data: new Float32Array(N_MELS * MEL_BINS),
                },
              ],
            ]),
            {},
          ),
        },
      }),
  },
];

describe.skipIf(!GPU_AVAILABLE)("fromAssets's own GPU acquisition (real GPU)", () => {
  for (const family of families) {
    it(`${family.name}: the build fails when the acquired device does not meet the declared limits`, async () => {
      await assertRejects(
        async () => {
          // If it passes (= before the fix), just clean up and let the assertRejects below fail.
          const pipeline = await family.build();
          await pipeline.dispose();
        },
        Error,
        LIMITS_MESSAGE,
      );
    });
  }
});
