/**
 * The **public verbs and the dispose contract** of the three image families (siglip2 / birefnet /
 * depth-anything) (real GPU).
 *
 * Each family's unit tests pin the per-component pure functions (preprocessing, postprocessing),
 * but neither they nor the real-asset e2e tests (which build Sessions directly) go through
 * `fromAssets` → verb → return value end to end, or through the ownership and serialization
 * contracts. Here a small weightless graph (resolution 2×2) is baked into a container and the three
 * families run through the same table:
 *
 * - The verb (`embed` / `segment` / `estimate`) wires "preprocess → one graph run → postprocess"
 *   and calls `onRunDiagnostics` once per call. birefnet / depth-anything return at the **input
 *   image's size** (not the baked resolution).
 * - Concurrent calls both return correct values (one Session used in turn).
 * - `dispose` waits for in-flight calls before tearing down (flush-before-destroy), and a second
 *   call returns the same completion. A call after dispose fails loudly.
 * - A shared GPU that was passed in is not destroyed by dispose (the next pipeline runs on the
 *   same GPU).
 *
 * The expected values are built from the same host-side pure functions (each family's
 * `preprocessPixelValues` and postprocessing), so this file pins only the wiring and the contracts
 * (the correctness of the pre- and postprocessing itself is covered by the unit tests and the
 * fixtures from the Python source of truth). The graphs use only the in-vocabulary ops `reshape` /
 * `add` / `sum`:
 *
 * - siglip2 `vision`: `pixel_values [1,3,2,2]` → reshape `[1,12]` → `x + x` (= `hiddenDim` 12)
 * - birefnet `matte`: `sum` over the channel axis → `[1,2,2]` → reshape `[1,1,2,2]`
 * - depth-anything `depth`: `sum` over the channel axis → `[1,2,2]`
 */

import { assert, assertEquals, assertRejects, assertStrictEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  acquireGpu,
  type GpuContext,
  type IrDeclaration,
  parseIrDeclarationValue,
  type SessionDiagnostics,
} from "@karume/runtime";
import { parseManifest } from "@karume/hub";
import {
  type BirefnetPipelineConfig,
  parseBirefnetPipelineConfig,
} from "../src/birefnet/config.ts";
import * as birefnet from "../birefnet.ts";
import { preprocessPixelValues as birefnetPixelValues } from "../src/birefnet/pipeline.ts";
import { matteFromLogits } from "../src/birefnet/matte.ts";
import { parseDepthAnythingPipelineConfig } from "../src/depth-anything/config.ts";
import * as depthAnything from "../depth-anything.ts";
import {
  preprocessPixelValues as depthPixelValues,
  resampleDepth,
} from "../src/depth-anything/pipeline.ts";
import type { Rgb8Image } from "../src/image/preprocess.ts";
import { parseSiglip2PipelineConfig } from "../src/siglip2/config.ts";
import * as siglip2 from "../siglip2.ts";
import { preprocessPixelValues as siglip2PixelValues } from "../src/siglip2/pipeline.ts";
import { declaredContainer, partAssets, writeContainer } from "./helpers/container-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/** The resolution baked into the graphs (square 2×2). */
const SIZE = 2;
const CHANNELS = 3;
const PLANE = SIZE * SIZE;

/**
 * The input image. Its size is **non-square** and differs from the baked resolution — to check both
 * that postprocessing "returns to the input image's size" and that width and height are not
 * swapped.
 */
const IMAGE: Rgb8Image = {
  width: 4,
  height: 3,
  data: Uint8Array.from({ length: 4 * 3 * CHANNELS }, (_, index) => (index * 37 + 11) % 256),
};

const f32 = (shape: readonly number[]) => ({ dtype: "f32", shape });

/** A one-input, one-output declaration that takes `pixel_values [1,3,2,2]`. */
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

/**
 * The sum over the channel axis (dim 1). The GPU side follows `sum`'s reduction order, so its
 * rounding can differ from the CPU's. depth is compared with a tolerance; birefnet matches exactly,
 * with a check of the rounding-boundary premise ({@link assertNoRoundingTie}).
 */
const channelSum = (pixelValues: Float32Array): Float32Array<ArrayBuffer> => {
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

/** A family-independent verb result (undefined size for siglip2, which returns none). */
type Output = {
  readonly values: readonly number[];
  readonly width?: number;
  readonly height?: number;
};

/** A family-independent pipeline handle. */
type Handle = {
  readonly run: (image: Rgb8Image) => Promise<Output>;
  readonly dispose: () => Promise<void>;
};

type BuildOptions = {
  readonly gpu: GpuContext;
  readonly onRunDiagnostics: (diagnostics: SessionDiagnostics) => void;
};

/** One table row. */
type Family = {
  readonly name: string;
  readonly build: (options: BuildOptions) => Promise<Handle>;
  /** The expected value built with the host-side pure functions. */
  readonly expected: (image: Rgb8Image) => Output;
  /** The per-element tolerance (siglip2 is `x + x`, so it matches exactly). */
  readonly tolerance: number;
  /** The message of the rejection after dispose. */
  readonly disposedMessage: string;
};

/** The skeleton of a distribution form (one component, one quant). */
const manifestOf = (pipeline: string, key: string, config: Record<string, unknown>) =>
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

/** Writes a one-component container and folds it into full-surface keys (`<component>[i]`). */
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
  postprocess: "u8-pil-bicubic",
};

/**
 * The upper bound of the difference between the GPU and CPU logits (the sum of the 3 channel terms)
 * (derivation: note 1 on the tolerance of the birefnet row).
 *
 * After normalization, |x| per channel is at most `max(mean, 1 − mean) / std` (pixels are in [0,
 * 1]).
 */
const birefnetLogitDiffBound = (config: BirefnetPipelineConfig): number => {
  let absSum = 0;
  for (let channel = 0; channel < CHANNELS; channel += 1) {
    const mean = config.imageMean[channel];
    absSum += Math.max(mean, 1 - mean) / config.imageStd[channel];
  }
  return 3 * 2 ** -24 * absSum;
};

/**
 * The premise of tolerance 0 (note 4 of the birefnet row): every logit is farther from the rounding
 * boundary of the 8-bit plane (`255σ = k + 0.5`) than the upper bound of the GPU–CPU difference. If
 * that breaks, fail without widening the tolerance (if the input changed, redo the derivation).
 */
const assertNoRoundingTie = (logits: Float32Array, config: BirefnetPipelineConfig): void => {
  // σ' ≤ 1/4, so the difference of 255σ is at most 255/4 times the logit difference (note 2).
  const bound = (255 / 4) * birefnetLogitDiffBound(config);
  for (const [index, logit] of logits.entries()) {
    const scaled = (1 / (1 + Math.exp(-logit))) * 255;
    const distance = Math.abs(scaled - (Math.floor(scaled) + 0.5));
    assert(
      distance > bound,
      `logit ${index}: 255σ = ${scaled} is ${distance} from the rounding boundary (within the bound ${bound}) — ` +
        "the 8-bit plane can differ between GPU and CPU, so the premise of tolerance 0 does not hold",
    );
  }
};

const DEPTH_CONFIG: Record<string, unknown> = {
  imageWidth: SIZE,
  imageHeight: SIZE,
  imageMean: [0.485, 0.456, 0.406],
  imageStd: [0.229, 0.224, 0.225],
  interpolation: "bicubic",
};

const siglip2Family: Family = {
  name: "siglip2.Pipeline.embed",
  build: async (options) => {
    const pipeline = await siglip2.Pipeline.fromAssets(
      {
        manifest: manifestOf("siglip2/1", "vision", SIGLIP2_CONFIG),
        assets: await assetsOf(
          "vision",
          declaration(
            ["reshape", "add"],
            "pooler_output",
            {
              flat: f32([1, CHANNELS * PLANE]),
              pooler_output: f32([1, CHANNELS * PLANE]),
            },
            [
              { op: "reshape", ins: ["pixel_values"], outs: ["flat"], attrs: {} },
              { op: "add", ins: ["flat", "flat"], outs: ["pooler_output"], attrs: {} },
            ],
          ),
        ),
      },
      options,
    );
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
  disposedMessage: "cannot embed after dispose",
};

const birefnetFamily: Family = {
  name: "birefnet.Pipeline.segment",
  build: async (options) => {
    const pipeline = await birefnet.Pipeline.fromAssets(
      {
        manifest: manifestOf("birefnet/2", "matte", BIREFNET_CONFIG),
        assets: await assetsOf(
          "matte",
          declaration(
            ["sum", "reshape"],
            "logits",
            {
              summed: f32([1, SIZE, SIZE]),
              logits: f32([1, 1, SIZE, SIZE]),
            },
            [
              { op: "sum", ins: ["pixel_values"], outs: ["summed"], attrs: { dim: 1 } },
              { op: "reshape", ins: ["summed"], outs: ["logits"], attrs: {} },
            ],
          ),
        ),
      },
      options,
    );
    return {
      run: async (image) => {
        const matte = await pipeline.segment(image);
        return { values: Array.from(matte.data), width: matte.width, height: matte.height };
      },
      dispose: () => pipeline.dispose(),
    };
  },
  expected: (image) => {
    const config = parseBirefnetPipelineConfig(BIREFNET_CONFIG);
    const logits = channelSum(birefnetPixelValues(config, image));
    assertNoRoundingTie(logits, config);
    const matte = matteFromLogits(
      logits,
      SIZE,
      SIZE,
      image.width,
      image.height,
      config.postprocess,
    );
    return { values: Array.from(matte.data), width: matte.width, height: matte.height };
  },
  // Derivation of tolerance 0. This test does not read a distribution form (a 2×2 small graph and
  // an in-place manifest), so the declared method is chosen here — HR's `u8-pil-bicubic` (the side
  // where the rounding to 8 bits comes before the resample, and a rounding split can spread through
  // bicubic's negative lobes).
  //  1. Logit difference: the input (`pixel_values` in f32) is identical on GPU and CPU. The CPU
  //     sums in f64 and rounds to f32 once (error ≤ u·|S|); the GPU sums the 3 terms in f32 (error
  //     ≤ 2u·Σ|x| in any order). With u = 2^-24, |Δlogit| ≤ 3u·Σ|x| ≤ 1.31e-6
  //     ({@link birefnetLogitDiffBound} — Σ|x| ≤ 7.32 with these constants).
  //  2. 8-bit plane: σ' ≤ 1/4, so |Δ(255σ)| ≤ 8.4e-5. Both sides compute σ with the same host
  //     function. `Math.round(255σ)` can split only when 255σ is within that of k + 0.5, and then
  //     by one step.
  //  3. Pillow's two BICUBIC passes (horizontal 2 → 4, vertical 2 → 3, rounding to 8 bits
  //     between the passes): the output change is at most the ceiling of "input change × the sum of
  //     the absolute coefficients". The maximum sum of the absolute coefficients is 1.18
  //     horizontally (1.088 / −0.088) and 1.13 vertically (1.066 / −0.066), so 1 step → 2 steps
  //     horizontally → 3 steps vertically. Without the premise, the worst case is this bound of 3
  //     steps.
  //  4. In this test's inputs (`IMAGE` and its inversion) 255σ is at least 0.105 from k + 0.5 in
  //     all 8 samples (more than 1,000 times the bound of 2), so the plane is identical on GPU and
  //     CPU. The resample is integer arithmetic, so the output is bit-identical too → tolerance 0.
  //     The premise is checked every time by `expected` ({@link assertNoRoundingTie}).
  tolerance: 0,
  disposedMessage: "cannot segment after dispose",
};

const depthAnythingFamily: Family = {
  name: "depthAnything.Pipeline.estimate",
  build: async (options) => {
    const pipeline = await depthAnything.Pipeline.fromAssets(
      {
        manifest: manifestOf("depth-anything/1", "depth", DEPTH_CONFIG),
        assets: await assetsOf(
          "depth",
          declaration(["sum"], "predicted_depth", { predicted_depth: f32([1, SIZE, SIZE]) }, [
            { op: "sum", ins: ["pixel_values"], outs: ["predicted_depth"], attrs: { dim: 1 } },
          ]),
        ),
      },
      options,
    );
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
  disposedMessage: "cannot estimate after dispose",
};

const assertOutput = (actual: Output, expected: Output, tolerance: number): void => {
  assertEquals(actual.width, expected.width, "the width differs from the input image");
  assertEquals(actual.height, expected.height, "the height differs from the input image");
  assertEquals(actual.values.length, expected.values.length);
  for (const [index, value] of actual.values.entries()) {
    const want = expected.values[index];
    assert(
      Math.abs(value - want) <= tolerance,
      `element ${index}: actual ${value} / expected ${want} (tolerance ${tolerance})`,
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

for (const family of [siglip2Family, birefnetFamily, depthAnythingFamily]) {
  describe.skipIf(!GPU_AVAILABLE)(`${family.name} (real GPU, small graph)`, () => {
    it("wires preprocess → one graph run → postprocess and hands over diagnostics once", async () => {
      await withGpu(async (gpu) => {
        const observed: SessionDiagnostics[] = [];
        const handle = await family.build({
          gpu,
          onRunDiagnostics: (diagnostics) => observed.push(diagnostics),
        });
        try {
          assertOutput(await handle.run(IMAGE), family.expected(IMAGE), family.tolerance);
          assertEquals(observed.length, 1);
          assert(observed[0].lastRun !== undefined, "not the diagnostics after the run");
        } finally {
          await handle.dispose();
        }
      });
    });

    it("concurrent calls both return correct values (one Session used in turn)", async () => {
      await withGpu(async (gpu) => {
        let calls = 0;
        const handle = await family.build({ gpu, onRunDiagnostics: () => (calls += 1) });
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

    it("dispose waits for the in-flight call before tearing down, and a second call returns the same completion", async () => {
      await withGpu(async (gpu) => {
        const handle = await family.build({ gpu, onRunDiagnostics: () => {} });
        const running = handle.run(IMAGE);
        const disposal = handle.dispose();
        assertStrictEquals(handle.dispose(), disposal);
        // A call accepted earlier is not caught up in the dispose (the Session is torn down after
        // it).
        assertOutput(await running, family.expected(IMAGE), family.tolerance);
        await disposal;
      });
    });

    it("a call after dispose fails loudly", async () => {
      await withGpu(async (gpu) => {
        const handle = await family.build({ gpu, onRunDiagnostics: () => {} });
        await handle.dispose();
        await assertRejects(() => handle.run(IMAGE), Error, family.disposedMessage);
      });
    });

    it("dispose does not destroy the shared GPU that was passed in", async () => {
      await withGpu(async (gpu) => {
        const first = await family.build({ gpu, onRunDiagnostics: () => {} });
        await first.dispose();
        // The next pipeline builds and runs on the same GPU = the previous pipeline did not break
        // the device.
        const second = await family.build({ gpu, onRunDiagnostics: () => {} });
        try {
          assertOutput(await second.run(IMAGE), family.expected(IMAGE), family.tolerance);
        } finally {
          await second.dispose();
        }
      });
    });
  });
}
