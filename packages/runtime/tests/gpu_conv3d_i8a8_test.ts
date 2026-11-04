// Real-GPU checks of the w8a8 variant of conv3d (`convCompute: "a8"` — ADR 0133).
//
// 1. **atol=0 against the TS mirror** (`referenceConv3dI8a8` — the integer dot product per tap and
//    the fma flush in the same order. Class E). The shapes are shrunk chunk-graph shapes + tile edges
//    (Cout > 128 gives 2 row tiles, scalar write-out for N % 4 ≠ 0, a K remainder for Cin % 16 ≠ 0)
//    + padding on 3 axes + stride ≠ 1 + dilation ≠ 1 + Kt ≠ Kh ≠ Kw.
// 2. **The dp4a version and the emulated version agree at atol=0** (the extension changes only the
//    speed — the conv-side detector for the same claim as linear's).
// 3. **The values split from the default (f32 implicit GEMM)** (evidence that activation quantization
//    takes effect — if they matched, the path did not switch).
// 4. **An ineligible node falls to the f32 path, and the key shows it** (Cin % 4 ≠ 0).
// 5. The mirror's **tap order** is ascending Kt, Kh, Kw (a mirror with Kh / Kw swapped splits from
//    the GPU — prevents the check from becoming vacuous).
//
// MUST: compare with `assertExact` (an exhaustive `===` comparison). Admitting a tolerance would erase
// the class E claim.
// MUST: check the margin from the activation's rounding boundaries (`quantizeRowsTieMargin` — measured
// on the transposed [pixel, Cin]) (GPU division is allowed up to 2.5 ULP, so data without a margin
// shifts by ±1 step — the same as linear).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { type Conv3dDims, conv3dIgemmKey, conv3dUsesVec4 } from "../src/kernels/conv3d.ts";
import {
  CONV3D_I8A8_QUANTIZE_GEOMETRY,
  conv3dI8a8Eligible,
  conv3dI8a8Key,
  conv3dI8a8UsesVec4,
} from "../src/kernels/conv3d-i8a8.ts";
import { conv2dIgemmMTile } from "../src/kernels/conv2d.ts";
import { DEFAULT_GEOMETRY_PROFILE, type GeometryProfile } from "../src/kernels/geometry-profile.ts";
import type { I8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { quantizeRowsChannelMajorKey } from "../src/kernels/quantize-rows.ts";
import { referenceConv3dI8a8 } from "../src/reference/conv3d-i8a8.ts";
import { quantizeRowsTieMargin } from "../src/reference/i8a8.ts";
import {
  createSessionFromContainer,
  I8A8_DOT,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { sessionGeometryProfile } from "../src/runtime/geometry-selection.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** Signed data whose magnitude differs per pixel (so that the per-pixel scale is not the same for every pixel). */
const SIGNED = (i: number): number => ((i % 29) - 14) * 0.3717 + 0.0131;
/** Weights whose magnitude differs per row (= output channel) (with the same scale on every row an axis mix-up would not show in the values). */
const WEIGHT = (i: number, rowLength: number): number =>
  (((i * 7) % 23) - 11) * 0.0625 * (1 + (Math.floor(i / rowLength) % 5) * 0.5);
const BIAS = (i: number): number => ((i % 11) - 5) * 0.25 + 0.0625;
const TIE_MARGIN = 1e-3;

const assertExact = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  for (let i = 0; i < actual.length; i += 1) {
    assert(
      actual[i] === expected[i],
      `${where}[${i}]: GPU ${actual[i]} / reference ${expected[i]}`,
    );
  }
};

type Triple = readonly [number, number, number];
const ONE: Triple = [1, 1, 1];

type Conv3dCase = {
  readonly name: string;
  readonly channelsIn: number;
  readonly channelsOut: number;
  /** Input [T, H, W]. */
  readonly input: Triple;
  readonly kernel: Triple;
  readonly stride: Triple;
  readonly padding: Triple;
  readonly dilation: Triple;
  /** The write-out variant the case should exercise (stated explicitly so that a predicate mix-up cannot hide behind "both on the same variant"). */
  readonly expectVec4: boolean;
  /**
   * Input generation (SIGNED by default). For shapes with a small Cin the per-pixel scale is set by
   * 4–6 values, and with SIGNED x/s drifts toward rounding boundaries, so the generator is replaced by
   * one whose ratios are known fractions of 127 (a shape that passes the margin check).
   */
  readonly x?: (channel: number, pixel: number) => number;
};

/** Input that rotates the per-pixel maximum channel so that the ratios fall on 127·{1, 0.8, −0.6, 0.4, …}. */
const fractionsOf =
  (fractions: readonly number[]) =>
  (channel: number, pixel: number): number =>
    (0.5 + (pixel % 13) * 0.37) * fractions[(channel + pixel) % fractions.length];

const outExtent = (
  input: number,
  kernel: number,
  stride: number,
  padding: number,
  dilation: number,
): number => Math.floor((input + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

const dimsOf = (testCase: Conv3dCase): Conv3dDims => ({
  channelsIn: testCase.channelsIn,
  channelsOut: testCase.channelsOut,
  timeIn: testCase.input[0],
  heightIn: testCase.input[1],
  widthIn: testCase.input[2],
  timeOut: outExtent(
    testCase.input[0],
    testCase.kernel[0],
    testCase.stride[0],
    testCase.padding[0],
    testCase.dilation[0],
  ),
  heightOut: outExtent(
    testCase.input[1],
    testCase.kernel[1],
    testCase.stride[1],
    testCase.padding[1],
    testCase.dilation[1],
  ),
  widthOut: outExtent(
    testCase.input[2],
    testCase.kernel[2],
    testCase.stride[2],
    testCase.padding[2],
    testCase.dilation[2],
  ),
  kernelT: testCase.kernel[0],
  kernelH: testCase.kernel[1],
  kernelW: testCase.kernel[2],
  strideT: testCase.stride[0],
  strideH: testCase.stride[1],
  strideW: testCase.stride[2],
  paddingT: testCase.padding[0],
  paddingH: testCase.padding[1],
  paddingW: testCase.padding[2],
  dilationT: testCase.dilation[0],
  dilationH: testCase.dilation[1],
  dilationW: testCase.dilation[2],
  groups: 1,
});

const CASES: readonly Conv3dCase[] = [
  {
    // A shrunk version of the chunk graph's conv_in (48→1024 k3, temporal padding 0, spatial same).
    // Cin = 48 is a multiple of 16 (no K remainder within a tap); N = 1·6·8 = 48 (v4)
    name: "shrunk chunk conv_in 48→96 k3 [48,3,6,8]",
    channelsIn: 48,
    channelsOut: 96,
    input: [3, 6, 8],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [0, 1, 1],
    dilation: ONE,
    expectVec4: true,
  },
  {
    // Cout = 130 > tile edge 128 (2 row tiles, the second with only 2 rows) x 1×1×1 (1 tap = 1 flush) x
    // N = 2·5·7 = 70 (scalar write-out) x Cin = 8 (2 packs = 2 of the 4 packs of one K tile)
    name: "row-tile edge 130 output channels 1×1×1 [8,2,5,7]",
    channelsIn: 8,
    channelsOut: 130,
    input: [2, 5, 7],
    kernel: [1, 1, 1],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: ONE,
    expectVec4: false,
  },
  {
    // stride ≠ 1 on all 3 axes, asymmetric; Kt ≠ Kh ≠ Kw; asymmetric padding; Cin = 20 (not a multiple
    // of 16 -> the second K tile of a tap holds 1 of 4 packs). T: (5+2−2−1)/2+1 = 3 / H: (9+2−1−1)/2+1
    // = 5 / W: (10−3−1)/3+1 = 3 -> N = 45 (scalar)
    name: "stride=[2,2,3] Kt≠Kh≠Kw Cin 20 [20,5,9,10] * W[6,20,3,2,4]",
    channelsIn: 20,
    channelsOut: 6,
    input: [5, 9, 10],
    kernel: [3, 2, 4],
    stride: [2, 2, 3],
    padding: [1, 1, 0],
    dilation: ONE,
    expectVec4: false,
  },
  {
    // dilation ≠ 1 on all 3 axes; Cin = 4 (1 pack = 1 of the 4 packs of one K tile); N = 5·6·8 = 240
    // (4 column tiles of 64, v4)
    name: "dilation=[3,3,2] Cin 4 [4,8,12,10] * W[6,4,2,3,2]",
    channelsIn: 4,
    channelsOut: 6,
    input: [8, 12, 10],
    kernel: [2, 3, 2],
    stride: ONE,
    padding: [0, 0, 0],
    dilation: [3, 3, 2],
    expectVec4: true,
    x: fractionsOf([1, 0.8, -0.6, 0.4]),
  },
  {
    // padding 1 on every axis (exercises the out-of-range 0 on all 3 axes — the detector that
    // out-of-range taps fold with scale 0). N = 4·5·7 = 140 (3 column tiles, v4)
    name: "padding 1 on every axis [12,4,5,7] * W[7,12,3,3,3]",
    channelsIn: 12,
    channelsOut: 7,
    input: [4, 5, 7],
    kernel: [3, 3, 3],
    stride: ONE,
    padding: [1, 1, 1],
    dilation: ONE,
    expectVec4: true,
  },
];

type Prepared = {
  readonly dims: Conv3dDims;
  readonly model: Promise<BoundContainer>;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
  /** The mirror's input (used for the fault injection into the tap order). */
  readonly mirror: {
    readonly x: Float32Array<ArrayBuffer>;
    readonly weight: Float32Array<ArrayBuffer>;
    readonly weightScale: Float32Array<ArrayBuffer>;
    readonly bias: Float32Array<ArrayBuffer>;
  };
};

/** Declaration and supply of a single `conv3d(x, w, b)` (w is i8 + per-channel scale). */
const prepare = (testCase: Conv3dCase): Prepared => {
  const dims = dimsOf(testCase);
  const { channelsIn, channelsOut } = dims;
  const xShape = [channelsIn, dims.timeIn, dims.heightIn, dims.widthIn];
  const wShape = [channelsOut, channelsIn, dims.kernelT, dims.kernelH, dims.kernelW];
  const rowLength = channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  const weight = fill(wShape, (i) => WEIGHT(i, rowLength));
  assert(weight.dtype === "f32");
  const quantized = quantizeI8(weight.data, wShape, 0);
  const bias = fill([channelsOut], BIAS);
  assert(bias.dtype === "f32");
  const pixels = dims.timeIn * dims.heightIn * dims.widthIn;
  const generate = testCase.x;
  const x = fill(
    xShape,
    generate === undefined ? SIGNED : (i) => generate(Math.floor(i / pixels), i % pixels),
  );
  assert(x.dtype === "f32");
  // The margin from the rounding boundaries is measured on the transposed [pixel, Cin] (the order the kernel quantizes in)
  const transposed = new Float32Array(pixels * channelsIn);
  for (let ci = 0; ci < channelsIn; ci += 1) {
    for (let pixel = 0; pixel < pixels; pixel += 1) {
      transposed[pixel * channelsIn + ci] = x.data[ci * pixels + pixel];
    }
  }
  const margin = quantizeRowsTieMargin(transposed, pixels, channelsIn);
  assert(
    margin > TIE_MARGIN,
    `${testCase.name}: the margin from the rounding boundaries is only ${margin}`,
  );
  // The mirror takes the quantized integer values as its weights (fake-quant value ÷ scale — exact integers)
  const weightInts = new Float32Array(quantized.values.length);
  for (let i = 0; i < weightInts.length; i += 1) {
    const channel = Math.floor(i / rowLength);
    weightInts[i] = Math.round(quantized.values[i] / quantized.scale[channel]);
  }
  const mirror = { x: x.data, weight: weightInts, weightScale: quantized.scale, bias: bias.data };
  // Ineligible shapes (Cin % 4 ≠ 0) are outside the mirror (only that they fall to the f32 path is checked)
  const expected = conv3dI8a8Eligible(dims, "i8")
    ? referenceConv3dI8a8({ ...mirror, dims })
    : new Float32Array(0);
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["conv3d"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: xShape }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: wShape },
      b: { dtype: "f32", shape: [channelsOut] },
      y: { dtype: "f32", shape: [channelsOut, dims.timeOut, dims.heightOut, dims.widthOut] },
    },
    nodes: [
      {
        op: "conv3d",
        ins: ["x", "w", "b"],
        outs: ["y"],
        attrs: {
          stride: [...testCase.stride],
          padding: [...testCase.padding],
          dilation: [...testCase.dilation],
          groups: 1,
        },
      },
    ],
  };
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    },
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int8-sym",
        groupSize: perChannelGroupSize(rowLength),
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    },
  ];
  return {
    dims,
    model: openModelBytes(declaration, tensors),
    x: { dtype: "f32", shape: xShape, data: x.data },
    expected,
    mirror,
  };
};

const run = async (
  gpu: GpuContext,
  prepared: Prepared,
  options: SessionOptions,
): Promise<{ readonly y: Tensor; readonly keys: string[]; readonly profile: string }> => {
  const session = await createSessionFromContainer(gpu, await prepared.model, GRAPH_NAME, options);
  try {
    const outputs = await session.run({ x: prepared.x });
    return {
      y: outputs["y"],
      keys: requireCensus(session.diagnostics().lastRunPipelines, "conv3d")
        .map((row) => row.key)
        .sort(),
      profile: session.diagnostics().geometryProfile,
    };
  } finally {
    await session.dispose();
  }
};

/**
 * The two keys that must run (per-pixel quantisation read straight from the channel-major input →
 * the integer dot-product implicit GEMM; no transposing copy — ADR 0133 addendum 2026-10-26). The
 * GEMM's geometry is the Session's profile's `i8a8.conv3d` (ADR 0115 addendum 2026-10-26).
 */
const expectedKeys = (dims: Conv3dDims, dp4a: boolean, geometry: I8a8Geometry): string[] =>
  [
    quantizeRowsChannelMajorKey(CONV3D_I8A8_QUANTIZE_GEOMETRY),
    conv3dI8a8Key(
      conv3dI8a8UsesVec4(dims.timeOut * dims.heightOut * dims.widthOut),
      dp4a,
      geometry,
    ),
  ].sort();

/**
 * The conv3d geometry the Session behind `ran` used: taken from the one selection path
 * (`sessionGeometryProfile` — the injected table, otherwise the embedded one for the adapter), and
 * checked against the profile id that Session reports, so the expected key follows whatever table
 * the Session actually ran.
 */
const ranConv3dGeometry = (gpu: GpuContext, ran: { readonly profile: string }): I8a8Geometry => {
  const profile = sessionGeometryProfile(gpu);
  assertEquals(ran.profile, profile.id, "the profile the Session reports");
  return profile.i8a8.conv3d;
};

test.skipIf(!GPU_AVAILABLE)(
  "convCompute:'a8' conv3d matches the TS mirror at atol=0 (shrunk chunk shape / tile edges / K remainder / 3-axis padding / stride / dilation, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const prepared = prepare(testCase);
        const { dims } = prepared;
        assert(conv3dI8a8Eligible(dims, "i8"), `${testCase.name}: ineligible shape`);
        assertEquals(
          conv3dI8a8UsesVec4(dims.timeOut * dims.heightOut * dims.widthOut),
          testCase.expectVec4,
          `${testCase.name}: the write-out variant exercised differs from the expected one`,
        );
        const actual = await run(gpu, prepared, { convCompute: "a8" });
        assertEquals(
          actual.y.shape,
          [dims.channelsOut, dims.timeOut, dims.heightOut, dims.widthOut],
          testCase.name,
        );
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // The vacuity check: if the output is constant, "a match" verifies nothing
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: the output is constant`);
        assertEquals(
          actual.keys,
          expectedKeys(dims, true, ranConv3dGeometry(gpu, actual)),
          `${testCase.name}: the keys that ran`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8: the dot4I8Packed version and the emulated version agree at atol=0 (the extension changes only the speed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const prepared = prepare(testCase);
        const results: Partial<Record<"dp4a" | "emu", Tensor>> = {};
        for (const dot of ["dp4a", "emu"] as const) {
          const actual = await run(gpu, prepared, { convCompute: "a8", [I8A8_DOT]: dot });
          assertEquals(
            actual.keys,
            expectedKeys(prepared.dims, dot === "dp4a", ranConv3dGeometry(gpu, actual)),
            `${testCase.name}: the key of ${dot}`,
          );
          results[dot] = actual.y;
        }
        assert(results.dp4a !== undefined && results.emu !== undefined);
        assertExact(results.emu.data, results.dp4a.data, `${testCase.name}: emu vs dp4a`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A non-default conv3d geometry: K tile 32 (so Cin % 32 != 0 leaves a K tail inside a tap) and a
 * 64×64 tile with 4×8 outputs per thread — differs from the default and from every embedded table.
 */
const ALTERNATE_CONV3D: I8a8Geometry = { regM: 4, regN: 8, wgX: 8, wgY: 16, tileK: 32 };

const withConv3d = (geometry: I8a8Geometry): GeometryProfile => ({
  ...DEFAULT_GEOMETRY_PROFILE,
  id: "conv3d-alternate",
  i8a8: { ...DEFAULT_GEOMETRY_PROFILE.i8a8, conv3d: geometry },
});

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8 runs the injected profile's i8a8.conv3d: the default and a non-default geometry both match the TS mirror at atol=0 under their own key (real GPU)",
  async () => {
    for (const profile of [DEFAULT_GEOMETRY_PROFILE, withConv3d(ALTERNATE_CONV3D)]) {
      const gpu = await acquireGpu({ geometryProfile: profile });
      try {
        for (const testCase of CASES) {
          const prepared = prepare(testCase);
          const where = `${profile.id} ${testCase.name}`;
          const actual = await run(gpu, prepared, { convCompute: "a8" });
          assertExact(actual.y.data, prepared.expected, where);
          assertEquals(actual.profile, profile.id, `${where}: the profile the Session reports`);
          assertEquals(
            actual.keys,
            expectedKeys(prepared.dims, true, profile.i8a8.conv3d),
            `${where}: the keys that ran`,
          );
        }
      } finally {
        gpu.destroy();
      }
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8 takes effect only when opted in, and its values split from the default f32 implicit GEMM (evidence of activation quantization, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const testCase = CASES[0];
      const prepared = prepare(testCase);
      const { dims } = prepared;
      const baseline = await run(gpu, prepared, {});
      const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
      assertEquals(
        baseline.keys,
        [
          conv3dIgemmKey(
            "i8",
            conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW),
            conv2dIgemmMTile(dims.channelsOut),
          ),
        ],
        "the default key (a single f32 implicit GEMM)",
      );
      const a8 = await run(gpu, prepared, { convCompute: "a8" });
      // MUST: the default's activations stay f32, so its values do not match the w8a8 mirror (if they
      // matched, activation quantization would not be taking effect)
      assertNotEquals(
        [...baseline.y.data],
        [...a8.y.data],
        "a8 has the same values as the default",
      );
      assertExact(a8.y.data, prepared.expected, "a8");
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv3d i8a8: a node with Cin % 4 ≠ 0 falls to the f32 path even when opted in, and the key shows it (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const testCase: Conv3dCase = {
        ...CASES[3],
        name: "Cin 6",
        channelsIn: 6,
        x: fractionsOf([1, 0.8, -0.6, 0.4, -0.2, 0.9]),
      };
      const dims = dimsOf(testCase);
      assert(!conv3dI8a8Eligible(dims, "i8"));
      // prepare has the margin check, but ineligible shapes are not compared against the mirror, so only the check is passed
      const prepared = prepare(testCase);
      const actual = await run(gpu, prepared, { convCompute: "a8" });
      const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
      assertEquals(actual.keys, [
        conv3dIgemmKey(
          "i8",
          conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW),
          conv2dIgemmMTile(dims.channelsOut),
        ),
      ]);
    } finally {
      gpu.destroy();
    }
  },
);

test("the conv3d i8a8 mirror folds taps in ascending Kt, Kh, Kw order (a mirror with Kh / Kw swapped splits)", () => {
  // For a shape with Kh ≠ Kw, passing the weights with (kh, kw) swapped changes the values (a symmetric kernel would make this always hold)
  const testCase = CASES[2];
  const prepared = prepare(testCase);
  const { dims, mirror } = prepared;
  const swapped = new Float32Array(mirror.weight.length);
  const { channelsOut, channelsIn, kernelT, kernelH, kernelW } = dims;
  for (let co = 0; co < channelsOut; co += 1) {
    for (let ci = 0; ci < channelsIn; ci += 1) {
      for (let kt = 0; kt < kernelT; kt += 1) {
        for (let kh = 0; kh < kernelH; kh += 1) {
          for (let kw = 0; kw < kernelW; kw += 1) {
            const from = (((co * channelsIn + ci) * kernelT + kt) * kernelH + kh) * kernelW + kw;
            // (kh, kw) -> (kw % Kh, kh % Kw): the kernel is not square, so this is not a bijection; only that the values move is checked
            const to =
              (((co * channelsIn + ci) * kernelT + kt) * kernelH + (kw % kernelH)) * kernelW +
              (kh % kernelW);
            swapped[to] = mirror.weight[from];
          }
        }
      }
    }
  }
  const other = referenceConv3dI8a8({ ...mirror, weight: swapped, dims });
  assertNotEquals([...other], [...prepared.expected]);
});
