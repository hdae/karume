/**
 * 画像 3 家族（siglip2 / birefnet / depth-anything）の**公開動詞と dispose の契約**（実 GPU）。
 *
 * 部品ごとの純関数（前処理・後処理）は各家族の単体テストが縛っているが、`fromAssets` → 動詞 →
 * 返り値の通しと、所有権・直列化の契約は実資産 e2e（Session を直接張る）も通らない。ここでは
 * 重みを持たない小さなグラフ（解像度 2×2）を容器に焼いて、3 家族を同じ表で回す:
 *
 * - 動詞（`embed` / `segment` / `estimate`）が「前処理 → グラフ 1 回 → 後処理」を結線し、
 *   `onRunDiagnostics` を 1 呼び出しにつき 1 回呼ぶ。birefnet / depth-anything は**入力画像の
 *   寸法**（焼いた解像度ではない）で返す。
 * - 並行に呼んでも両方が正しい値を返す（1 つの Session を順に使う）。
 * - `dispose` は in-flight の呼び出しを待ってから畳み（flush-before-destroy）、2 度目も同じ完了を
 *   返す。dispose 後の呼び出しは fail loudly で拒む。
 * - 渡した共有 GPU は dispose しても破棄しない（同じ GPU で次のパイプラインが動く）。
 *
 * 期待値はホスト側の同じ純関数（各家族の `preprocessPixelValues` と後処理）から組むので、ここが
 * 縛るのは結線と契約だけ（前後処理そのものの正しさは単体テストと Python 正本のフィクスチャ）。
 * グラフの演算は語彙内の `reshape` / `add` / `sum` だけで組む:
 *
 * - siglip2 `vision`: `pixel_values [1,3,2,2]` → reshape `[1,12]` → `x + x`（= `hiddenDim` 12）
 * - birefnet `matte`: チャネル軸の `sum` → `[1,2,2]` → reshape `[1,1,2,2]`
 * - depth-anything `depth`: チャネル軸の `sum` → `[1,2,2]`
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "@std/testing/bdd";
import {
  acquireGpu,
  type GpuContext,
  type IrDeclaration,
  parseIrDeclarationValue,
  type SessionDiagnostics,
} from "@karume/runtime";
import { parseManifest } from "@karume/hub";
import { parseBirefnetPipelineConfig } from "../src/birefnet/config.ts";
import {
  BirefnetPipeline,
  matteFromLogits,
  preprocessPixelValues as birefnetPixelValues,
} from "../src/birefnet/pipeline.ts";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";
import {
  DepthAnythingPipeline,
  preprocessPixelValues as depthPixelValues,
  resampleDepth,
} from "../src/depth-anything/pipeline.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { parseSiglip2PipelineConfig } from "../src/siglip2/config.ts";
import {
  preprocessPixelValues as siglip2PixelValues,
  Siglip2Pipeline,
} from "../src/siglip2/pipeline.ts";
import { declaredContainer, partAssets, writeContainer } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** グラフに焼く解像度（正方 2×2）。 */
const SIZE = 2;
const CHANNELS = 3;
const PLANE = SIZE * SIZE;

/**
 * 入力画像。焼いた解像度と違う**非正方**の寸法にする — 後処理が「入力画像の寸法へ戻す」ことと、
 * 幅と高さの取り違えを同時に見るため。
 */
const IMAGE: Rgb8Image = {
  width: 4,
  height: 3,
  data: Uint8Array.from({ length: 4 * 3 * CHANNELS }, (_, index) => (index * 37 + 11) % 256),
};

const f32 = (shape: readonly number[]) => ({ dtype: "f32", shape });

/** `pixel_values [1,3,2,2]` を受ける 1 入力 1 出力の宣言。 */
const declaration = (
  ops: readonly string[],
  output: string,
  values: Record<string, { dtype: string; shape: readonly number[] }>,
  nodes: readonly Record<string, unknown>[],
): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops },
    symbols: [],
    inputs: [{ name: "pixel_values", ...f32([1, CHANNELS, SIZE, SIZE]) }],
    outputs: [output],
    initializers: {},
    values,
    states: {},
    nodes,
  });

/** チャネル軸（dim 1）の総和。GPU 側は `sum` の縮約順なので、比較は許容差つきで行う。 */
const channelSum = (pixelValues: Float32Array): Float32Array => {
  const out = new Float32Array(PLANE);
  for (let index = 0; index < PLANE; index += 1) {
    let sum = 0;
    for (let channel = 0; channel < CHANNELS; channel += 1) {
      sum += pixelValues[channel * PLANE + index];
    }
    out[index] = sum;
  }
  return out;
};

/** 家族を問わない動詞の返り値（寸法を返さない siglip2 は undefined）。 */
type Output = {
  readonly values: readonly number[];
  readonly width?: number;
  readonly height?: number;
};

/** 家族を問わないパイプラインの把手。 */
type Handle = {
  readonly run: (image: Rgb8Image) => Promise<Output>;
  readonly dispose: () => Promise<void>;
};

type BuildOptions = {
  readonly gpu: GpuContext;
  readonly onRunDiagnostics: (diagnostics: SessionDiagnostics) => void;
};

/** 表の 1 行。 */
type Family = {
  readonly name: string;
  readonly build: (options: BuildOptions) => Promise<Handle>;
  /** ホスト側の純関数で組んだ期待値。 */
  readonly expected: (image: Rgb8Image) => Output;
  /** 要素ごとの許容差（siglip2 は `x + x` なので厳密一致）。 */
  readonly tolerance: number;
  /** dispose 後の拒否の文言。 */
  readonly disposedMessage: string;
};

/** 配布形の骨格（部品 1 本・quant 1 つ）。 */
const manifestOf = (pipeline: string, key: string, config: Record<string, unknown>) =>
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

const SIGLIP2_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  hiddenDim: CHANNELS * PLANE,
  interpolation: "bilinear",
};

const BIREFNET_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bilinear",
};

const DEPTH_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
};

const siglip2: Family = {
  name: "Siglip2Pipeline.embed",
  build: async (options) => {
    const pipeline = await Siglip2Pipeline.fromAssets({
      manifest: manifestOf("siglip2/1", "vision", SIGLIP2_CONFIG),
      assets: await assetsOf(
        "vision",
        declaration(["reshape", "add"], "pooler_output", {
          flat: f32([1, CHANNELS * PLANE]),
          pooler_output: f32([1, CHANNELS * PLANE]),
        }, [
          { op: "reshape", ins: ["pixel_values"], outs: ["flat"], attrs: {} },
          { op: "add", ins: ["flat", "flat"], outs: ["pooler_output"], attrs: {} },
        ]),
      ),
    }, options);
    return {
      run: async (image) => ({ values: Array.from(await pipeline.embed(image)) }),
      dispose: () => pipeline.dispose(),
    };
  },
  expected: (image) => ({
    values: Array.from(
      siglip2PixelValues(parseSiglip2PipelineConfig(SIGLIP2_CONFIG), image),
      (value) => 2 * value,
    ),
  }),
  tolerance: 0,
  disposedMessage: "dispose 済みでは埋め込めない",
};

const birefnet: Family = {
  name: "BirefnetPipeline.segment",
  build: async (options) => {
    const pipeline = await BirefnetPipeline.fromAssets({
      manifest: manifestOf("birefnet/1", "matte", BIREFNET_CONFIG),
      assets: await assetsOf(
        "matte",
        declaration(["sum", "reshape"], "logits", {
          summed: f32([1, SIZE, SIZE]),
          logits: f32([1, 1, SIZE, SIZE]),
        }, [
          { op: "sum", ins: ["pixel_values"], outs: ["summed"], attrs: { dim: 1 } },
          { op: "reshape", ins: ["summed"], outs: ["logits"], attrs: {} },
        ]),
      ),
    }, options);
    return {
      run: async (image) => {
        const matte = await pipeline.segment(image);
        return { values: Array.from(matte.data), width: matte.width, height: matte.height };
      },
      dispose: () => pipeline.dispose(),
    };
  },
  expected: (image) => {
    const logits = channelSum(
      birefnetPixelValues(parseBirefnetPipelineConfig(BIREFNET_CONFIG), image),
    );
    const matte = matteFromLogits(logits, SIZE, SIZE, image.width, image.height);
    return { values: Array.from(matte.data), width: matte.width, height: matte.height };
  },
  // 8bit へ丸める直前の値が縮約順の差で境界をまたぐと 1 LSB ずれうる。
  tolerance: 1,
  disposedMessage: "dispose 済みでは切り抜けない",
};

const depthAnything: Family = {
  name: "DepthAnythingPipeline.estimate",
  build: async (options) => {
    const pipeline = await DepthAnythingPipeline.fromAssets({
      manifest: manifestOf("depth-anything/1", "depth", DEPTH_CONFIG),
      assets: await assetsOf(
        "depth",
        declaration(["sum"], "predicted_depth", { predicted_depth: f32([1, SIZE, SIZE]) }, [
          { op: "sum", ins: ["pixel_values"], outs: ["predicted_depth"], attrs: { dim: 1 } },
        ]),
      ),
    }, options);
    return {
      run: async (image) => {
        const depth = await pipeline.estimate(image);
        return { values: Array.from(depth.data), width: depth.width, height: depth.height };
      },
      dispose: () => pipeline.dispose(),
    };
  },
  expected: (image) => {
    const depth = channelSum(
      depthPixelValues(parseDepthAnythingPipelineConfig(DEPTH_CONFIG), image),
    );
    const map = resampleDepth(depth, SIZE, SIZE, image.width, image.height);
    return { values: Array.from(map.data), width: map.width, height: map.height };
  },
  tolerance: 1e-5,
  disposedMessage: "dispose 済みでは推定できない",
};

const assertOutput = (actual: Output, expected: Output, tolerance: number): void => {
  assertEquals(actual.width, expected.width, "幅が入力画像と違う");
  assertEquals(actual.height, expected.height, "高さが入力画像と違う");
  assertEquals(actual.values.length, expected.values.length);
  for (const [index, value] of actual.values.entries()) {
    const want = expected.values[index];
    assert(
      Math.abs(value - want) <= tolerance,
      `要素 ${index}: 実測 ${value} / 期待 ${want}（許容 ${tolerance}）`,
    );
  }
};

const withGpu = async (fn: (gpu: GpuContext) => Promise<void>): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    await fn(gpu);
  } finally {
    gpu.destroy();
  }
};

for (const family of [siglip2, birefnet, depthAnything]) {
  describe({
    name: `${family.name}（実 GPU・小グラフ）`,
    ignore: !GPU_AVAILABLE,
    fn: () => {
      it("前処理 → グラフ 1 回 → 後処理を結線し、診断を 1 回渡す", async () => {
        await withGpu(async (gpu) => {
          const observed: SessionDiagnostics[] = [];
          const handle = await family.build({
            gpu,
            onRunDiagnostics: (diagnostics) => observed.push(diagnostics),
          });
          try {
            assertOutput(await handle.run(IMAGE), family.expected(IMAGE), family.tolerance);
            assertEquals(observed.length, 1);
            assert(observed[0].lastRun !== undefined, "run の後の診断でない");
          } finally {
            await handle.dispose();
          }
        });
      });

      it("並行に呼んでも両方が正しい値を返す（1 つの Session を順に使う）", async () => {
        await withGpu(async (gpu) => {
          let calls = 0;
          const handle = await family.build({ gpu, onRunDiagnostics: () => calls += 1 });
          try {
            const flipped: Rgb8Image = { ...IMAGE, data: IMAGE.data.map((byte) => 255 - byte) };
            const [first, second] = await Promise.all([handle.run(IMAGE), handle.run(flipped)]);
            assertOutput(first, family.expected(IMAGE), family.tolerance);
            assertOutput(second, family.expected(flipped), family.tolerance);
            assertEquals(calls, 2);
          } finally {
            await handle.dispose();
          }
        });
      });

      it("dispose は in-flight の呼び出しを待って畳み、2 度目も同じ完了を返す", async () => {
        await withGpu(async (gpu) => {
          const handle = await family.build({ gpu, onRunDiagnostics: () => {} });
          const running = handle.run(IMAGE);
          const disposal = handle.dispose();
          assertStrictEquals(handle.dispose(), disposal);
          // 先に受けた呼び出しは dispose の巻き添えにならない（Session を畳むのはその後）。
          assertOutput(await running, family.expected(IMAGE), family.tolerance);
          await disposal;
        });
      });

      it("dispose 後の呼び出しは fail loudly で拒む", async () => {
        await withGpu(async (gpu) => {
          const handle = await family.build({ gpu, onRunDiagnostics: () => {} });
          await handle.dispose();
          await assertRejects(() => handle.run(IMAGE), Error, family.disposedMessage);
        });
      });

      it("渡した共有 GPU は dispose しても破棄しない", async () => {
        await withGpu(async (gpu) => {
          const first = await family.build({ gpu, onRunDiagnostics: () => {} });
          await first.dispose();
          // 同じ GPU で次のパイプラインが張れて動く = 前のパイプラインが device を壊していない。
          const second = await family.build({ gpu, onRunDiagnostics: () => {} });
          try {
            assertOutput(await second.run(IMAGE), family.expected(IMAGE), family.tolerance);
          } finally {
            await second.dispose();
          }
        });
      });
    },
  });
}
