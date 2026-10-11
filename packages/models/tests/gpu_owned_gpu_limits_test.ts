/**
 * `fromAssets` で `gpu` を省いた（パイプラインが自前で device を取る）経路の `requiredLimits`
 * 照合（実 GPU）。
 *
 * 配布形の quant は満たすべき device limit の最小値（`requiredLimits` — ADR 0038 §7）を宣言する。
 * 共有 GPU は各家族の admission が、`fromPretrained` はダウンロード前にアダプタ値で見るが、
 * `fromAssets` の自前取得には取得前の席が無い。構築が取った device の limits を見ないと、宣言の
 * 不足は Session 構築時の実バッファ検査まで遅れ（低水準の文言になる）、宣言が実需要より厳しい
 * 場合は照合自体が一度も行われない。
 *
 * ここでは**どの実機も満たせない宣言**（`Number.MAX_SAFE_INTEGER` — manifest が受ける最大値）を
 * 置き、宣言だけの容器（重みを持たない）で 4 家族を組んで、limits の文言で落ちることを見る。
 * 家族の門（グラフ宣言・pipelineConfig）は全部通る形にしてあるので、落ちる理由は limits だけ。
 *
 * NOTE: 自前で取った device の破棄（失敗時の後始末）は `acquireGpu` を注入できないので観測して
 * いない。後始末は feature 検査と同じ catch に乗っている。
 */

import { assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type Manifest, parseManifest } from "@karume/hub";
import { BirefnetPipeline } from "../src/birefnet/pipeline.ts";
import { DepthAnythingPipeline } from "../src/depth-anything/pipeline.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { VowelDetectorPipeline } from "../src/vowel-detector/pipeline.ts";
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

/** 焼いた解像度（正方 2×2）。 */
const SIZE = 2;

/** どの実機も満たせない宣言。 */
const UNSATISFIABLE = { maxBufferSize: Number.MAX_SAFE_INTEGER };

/** limits の門の文言（`assertRequiredLimitsSatisfied`）。 */
const LIMITS_MESSAGE = "が要求する device limit をこの GPU が満たさない";

/** 配布形の骨格（部品 1 本・quant 1 つ・満たせない limits）。 */
const manifestOf = (
  pipeline: string,
  key: string,
  config: Record<string, unknown>,
  assets: Record<string, unknown> = {},
): Manifest =>
  parseManifest(JSON.stringify({
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
  }));

/** 宣言だけの容器 1 本を全量面のキー（`<部品>[i]`）へ畳む。 */
const containerOf = async (key: string, spec: TensorlessGraphSpec) =>
  partAssets(key, await tensorlessContainer(key, spec));

/** `pixel_values [1,3,2,2]` を受ける画像グラフの宣言。 */
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

/** 表の 1 行（自前取得の経路で組む）。 */
type Family = {
  readonly name: string;
  readonly build: () => Promise<{ dispose: () => Promise<void> }>;
};

const families: readonly Family[] = [
  {
    name: "Siglip2Pipeline",
    build: async () =>
      await Siglip2Pipeline.fromAssets({
        manifest: manifestOf("siglip2/1", "vision", {
          ...IMAGE_CONFIG,
          hiddenDim: 12,
          interpolation: "bilinear",
        }),
        assets: await containerOf("vision", imageGraph("pooler_output", [1, 12])),
      }),
  },
  {
    name: "BirefnetPipeline",
    build: async () =>
      await BirefnetPipeline.fromAssets({
        manifest: manifestOf("birefnet/1", "matte", { ...IMAGE_CONFIG, interpolation: "bilinear" }),
        assets: await containerOf("matte", imageGraph("logits", [1, 1, SIZE, SIZE])),
      }),
  },
  {
    name: "DepthAnythingPipeline",
    build: async () =>
      await DepthAnythingPipeline.fromAssets({
        manifest: manifestOf("depth-anything/1", "depth", {
          ...IMAGE_CONFIG,
          interpolation: "bicubic",
        }),
        assets: await containerOf("depth", imageGraph("predicted_depth", [1, SIZE, SIZE])),
      }),
  },
  {
    name: "VowelDetectorPipeline",
    build: async () =>
      await VowelDetectorPipeline.fromAssets({
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
          ...await containerOf("crnn", {
            symbols: ["T"],
            inputs: [{ name: "features", shape: [1, "2T", FEATURE_DIM] }],
            output: { name: "logits", shape: [1, "T", LIPSYNC_CLASSES.length] },
          }),
          mel_basis: writeSafetensors(
            new Map([["mel_basis", {
              dtype: "F32" as const,
              shape: [N_MELS, MEL_BINS],
              data: new Float32Array(N_MELS * MEL_BINS),
            }]]),
            {},
          ),
        },
      }),
  },
];

describe({
  name: "fromAssets の自前 GPU 取得（実 GPU）",
  // MUST: アダプタ無し環境は明示 SKIP（ADR 0005）。自前取得の経路は acquireGpu を踏むので、
  // アダプタが無いと GpuUnavailableError に化けて門が見えない。
  ignore: !GPU_AVAILABLE,
  fn: () => {
    for (const family of families) {
      it(`${family.name}: 取った device が宣言の limits を満たさなければ構築で落とす`, async () => {
        await assertRejects(
          async () => {
            // 通ってしまったら（= 修正前）後始末だけして、下の assertRejects に落とさせる。
            const pipeline = await family.build();
            await pipeline.dispose();
          },
          Error,
          LIMITS_MESSAGE,
        );
      });
    }
  },
});
