/**
 * 画像 3 家族（siglip2 / birefnet / depth-anything）の構築前検査が、軸の寸法だけでなく
 * **入力 dtype** と **SigLIP2 の出力の形**も見ること。GPU も実資産も要らない。
 *
 * 軸の寸法だけを見る検査は、同じ形で dtype が違うグラフ（i32 の `pixel_values`）や batch 2 で
 * 焼いた SigLIP2 の出力（`[2, hiddenDim]`）を通す。そのまま進むと GPU と Session を作って重みを
 * 読んだ後（SigLIP2 の出力は 1 回実行した後）に、資産の取り違えを指さない文言で落ちる。
 *
 * 観測の仕掛け: グラフ宣言だけの容器を `fromAssets` に渡し、`gpu` は渡さない。家族の門は GPU を
 * 取る**前**の admission で見るので、ここで落ちれば GPU に触れない。門をすり抜けると構築が
 * `acquireGpu` へ進み、別の文言（GPU 不在・Session の検査）で落ちるか通ってしまう。
 */

import { assertRejects } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import { type IrDeclaration, parseIrDeclarationValue } from "@karume/runtime";
import { type Manifest, parseManifest } from "@karume/hub";
import { BirefnetPipeline } from "../src/birefnet/pipeline.ts";
import { DepthAnythingPipeline } from "../src/depth-anything/pipeline.ts";
import { Siglip2Pipeline } from "../src/siglip2/pipeline.ts";
import { declaredContainer, partAssets, writeContainer } from "./helpers/container-fixture.ts";

/** 焼いた解像度（正方 2×2）。 */
const SIZE = 2;

/**
 * 入力 1 本・出力 1 本の宣言（実行しないので演算の中身は問わない）。入力 dtype を変えても IR の
 * op 契約を満たすよう、先頭で f32 へ `cast` してから出力の形へ `reshape` する。出力 dtype が
 * f32 でないときは、末尾でその dtype へ `cast` する。
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

/** 部品 1 本の容器を書いて、全量面のキー（`<部品>[i]`）へ畳む。 */
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
  parseManifest(JSON.stringify({
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
  }));

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
  await Siglip2Pipeline.fromAssets({
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

describe("画像 3 家族の構築前検査", () => {
  describe("pixel_values が f32 でないグラフを受けたとき", () => {
    it("SigLIP2 は GPU を取る前に入力 dtype を名指しで落とす", async () => {
      await assertRejects(
        () => buildSiglip2("i32", [1, HIDDEN]),
        Error,
        "siglip2: 前処理の pixel_values — グラフ入力 'pixel_values' の dtype が i32、前処理が書くのは f32",
      );
    });

    it("BiRefNet は GPU を取る前に入力 dtype を名指しで落とす", async () => {
      await assertRejects(
        async () =>
          await BirefnetPipeline.fromAssets({
            manifest: manifestOf("birefnet/2", "matte", {
              ...IMAGE_CONFIG,
              interpolation: "bilinear",
              postprocess: "u8-pil-bicubic",
            }),
            assets: await assetsOf("matte", declaration("i32", "logits", [1, 1, SIZE, SIZE])),
          }),
        Error,
        "birefnet: 前処理の pixel_values — グラフ入力 'pixel_values' の dtype が i32",
      );
    });

    it("Depth Anything は GPU を取る前に入力 dtype を名指しで落とす", async () => {
      await assertRejects(
        async () =>
          await DepthAnythingPipeline.fromAssets({
            manifest: manifestOf("depth-anything/1", "depth", {
              ...IMAGE_CONFIG,
              interpolation: "bicubic",
            }),
            assets: await assetsOf(
              "depth",
              declaration("i32", "predicted_depth", [1, SIZE, SIZE]),
            ),
          }),
        Error,
        "depth-anything: 前処理の pixel_values — グラフ入力 'pixel_values' の dtype が i32",
      );
    });
  });

  describe("SigLIP2 の出力が [1, hiddenDim] でないグラフを受けたとき", () => {
    it("batch 2 で焼いた出力は GPU を取る前に落とす", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [2, HIDDEN]),
        Error,
        "siglip2: pooler_output — グラフ出力 'pooler_output' が f32 [2, 12]、期待は f32 の [1, hiddenDim]",
      );
    });

    // 軸 1 を hiddenDim に揃え、既存の軸検査（assertOutputDim）を通る形にする。こうしないと
    // rank の検査が無くても軸検査で落ち、rank の検査を縛れない。
    it("rank 3 の出力は GPU を取る前に落とす", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [1, HIDDEN, 1]),
        Error,
        "siglip2: pooler_output — グラフ出力 'pooler_output' が f32 [1, 12, 1]、期待は f32 の [1, hiddenDim]",
      );
    });

    it("f32 でない出力は GPU を取る前に落とす", async () => {
      await assertRejects(
        () => buildSiglip2("f32", [1, HIDDEN], "i32"),
        Error,
        "siglip2: pooler_output — グラフ出力 'pooler_output' が i32 [1, 12]、期待は f32 の [1, hiddenDim]",
      );
    });
  });
});
