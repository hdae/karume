/**
 * The f32 GEMM skeleton forms (`GemmGeometry.form`: `"tb-un"` / `"tas-tb-un"`) and the counter
 * decode of conv2d's implicit GEMM (`GemmGeometry.decode: "counter"`) — perf-ledger K-88, optional
 * geometry fields — against the same geometry without the field, as Uint32 words, on a real GPU.
 *
 * Every form runs the same multiply-adds in the same order per output element and reads the same
 * elements, so the output must be the very same words: any difference is a thread that stored the
 * wrong quad, a tile slot filled from the wrong place, a K step taken twice or skipped, or a decode
 * that read another element of x. The kernels are dispatched directly (the WGSL the generator
 * returns, pipelines cached by **the final WGSL string**), so that the fault injection below — text
 * mutations of the generated WGSL, not switches in the generator — runs its own string (the MUST
 * of tests/helpers/rank-dispatch.ts).
 *
 * Geometries: every distinct `gemmRows` geometry and every distinct `conv2d` geometry of the
 * default profile and the embedded tables (the adapter's table among them). Shapes: linear /
 * matmul / bmm with row tails (m and n off the tile edges, m = 1, a partial K tile, a scalar shape
 * where `k % 4 != 0` — there the form must not change the WGSL at all); conv2d 3×3 with padding at
 * every edge, stride 1 and stride_h 2, 1×1, 5×5, 7×7, dilation 2, 1×3 / 3×1, no padding, channel
 * counts off the tile, below one tile, several workgroups and a batch (conv2d's implicit GEMM has
 * no groups — groups > 1 runs the direct kernel). Weights: f32 / f16 / bf16 / i8 (and i4 g32 for
 * linear). Inputs: seeded values in [-1, 1) with ±0 and subnormals sprinkled in, and an
 * adversarial set that adds ±Inf and NaN with a payload at a few positions.
 *
 * Without an adapter every case SKIPs explicitly (ADR 0005).
 */

import { assert, assertEquals, assertNotEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { bmmParams, bmmWgsl } from "../src/kernels/bmm.ts";
import {
  type Conv2dDims,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dUsesVec4,
} from "../src/kernels/conv2d.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { type GemmGeometry, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import { BUILTIN_GEOMETRY_PROFILES } from "../src/kernels/geometry-profiles/index.ts";
import { linearParams, linearWgsl } from "../src/kernels/linear.ts";
import { matmulParams, matmulWgsl } from "../src/kernels/matmul.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

const FORMS = ["tb-un", "tas-tb-un"] as const;

const distinct = (geometries: readonly GemmGeometry[]): GemmGeometry[] => {
  const seen = new Set<string>();
  return geometries.filter((geometry) => {
    const key = JSON.stringify(geometry);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const PROFILES = [DEFAULT_GEOMETRY_PROFILE, ...BUILTIN_GEOMETRY_PROFILES];

/** Every `gemmRows` geometry of the default profile and the embedded tables. */
const GEMM_GEOMETRIES = distinct(
  PROFILES.flatMap((profile) => profile.gemmRows.map((rule) => rule.geometry)),
);

/** Every `conv2d` geometry of the default profile and the embedded tables. */
const CONV_GEOMETRIES = distinct(
  PROFILES.flatMap((profile) => [profile.conv2d.rows64, profile.conv2d.rows32]),
);

const name = (geometry: GemmGeometry): string =>
  `r${geometry.regM}x${geometry.regN}w${geometry.wgX}x${geometry.wgY}`;

// ── Seeded inputs ──────────────────────────────────────────────────────────────────────────────

const xorshift = (seed: number): (() => number) => {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state ^ (state << 13)) >>> 0;
    state = (state ^ (state >>> 17)) >>> 0;
    state = (state ^ (state << 5)) >>> 0;
    return state;
  };
};

type Mode = "random" | "adversarial";

/** ±0 and ± subnormal f32 bit patterns (sprinkled in both modes — they poison no row). */
const QUIET_F32 = [0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00012345];
/** ±Inf and NaN with a payload (only in the adversarial mode, at a few positions). */
const LOUD_F32 = [0x7f800000, 0xff800000, 0x7fc0beef, 0xffc01234];

/**
 * f32 bit patterns: values in [-1, 1), one in 23 a ±0 / subnormal, and in the adversarial mode
 * `loud` positions (spread over the array) holding ±Inf / NaN with a payload — by default 4 in an
 * array of at least 4,096 elements and none in a smaller one (a few poisoned rows leave the rest of
 * the output informative; in a tiny or single-row input they would poison all of it).
 */
const f32Words = (
  count: number,
  seed: number,
  mode: Mode,
  loud = count >= 4096 ? 4 : 0,
): Uint32Array<ArrayBuffer> => {
  const next = xorshift(seed);
  const out = new Uint32Array(Math.max(4, count));
  const bits = new Float32Array(1);
  const word = new Uint32Array(bits.buffer);
  for (let index = 0; index < count; index += 1) {
    const r = next();
    if (r % 23 === 0) {
      out[index] = QUIET_F32[(r >>> 8) % QUIET_F32.length];
    } else {
      bits[0] = (r / 2 ** 32) * 2 - 1;
      out[index] = word[0];
    }
  }
  if (mode === "adversarial" && count > 0) {
    for (let slot = 0; slot < loud; slot += 1) {
      out[next() % count] = LOUD_F32[slot % LOUD_F32.length];
    }
  }
  return out;
};

/**
 * The weight words of `storage` for `count` logical elements (f32 / bf16 from the f32 patterns
 * above, f16 from its own patterns, i8 / i4 from random bytes).
 */
const weightWords = (
  storage: WeightStorage,
  count: number,
  seed: number,
  mode: Mode,
): Uint32Array<ArrayBuffer> => {
  switch (storage) {
    case "f32":
      return f32Words(count, seed, mode, count >= 512 ? 2 : 0);
    case "bf16": {
      const source = f32Words(count, seed, mode, count >= 512 ? 2 : 0);
      const out = new Uint32Array(Math.max(4, Math.ceil(count / 2)));
      for (let index = 0; index < count; index += 1) {
        out[index >> 1] |= (source[index] >>> 16) << ((index & 1) * 16);
      }
      return out;
    }
    case "f16": {
      const next = xorshift(seed);
      // Normal halves |v| ∈ [2^-4, 1) with a random sign; ±0 / subnormal halves sprinkled in, and
      // in the adversarial mode a few ±Inf / NaN halves
      const quiet = [0x0000, 0x8000, 0x0001, 0x83ff];
      const loud = [0x7c00, 0xfc00, 0x7e01];
      const halves = new Uint16Array(count);
      for (let index = 0; index < count; index += 1) {
        const r = next();
        halves[index] =
          r % 29 === 0
            ? quiet[(r >>> 8) % quiet.length]
            : (((r >>> 12) & 1) << 15) | ((11 + (r & 3)) << 10) | ((r >>> 2) & 0x3ff);
      }
      if (mode === "adversarial" && count >= 512) {
        for (let slot = 0; slot < 2; slot += 1) halves[next() % count] = loud[slot % loud.length];
      }
      const out = new Uint32Array(Math.max(4, Math.ceil(count / 2)));
      for (let index = 0; index < count; index += 1) {
        out[index >> 1] |= halves[index] << ((index & 1) * 16);
      }
      return out;
    }
    case "i8":
    case "i4": {
      const next = xorshift(seed);
      const out = new Uint32Array(Math.max(4, Math.ceil(count / (storage === "i8" ? 4 : 8))));
      for (let index = 0; index < out.length; index += 1) out[index] = next();
      return out;
    }
    case "i2":
      throw new Error("i2 weights are not part of this check");
  }
};

/** Positive f32 scales near 1/127 (the per-channel / group scales of i8 / i4). */
const scaleWords = (count: number, seed: number): Float32Array<ArrayBuffer> => {
  const next = xorshift(seed);
  const out = new Float32Array(Math.max(4, count));
  for (let index = 0; index < count; index += 1) out[index] = (0.5 + next() / 2 ** 32) / 127;
  return out;
};

// ── Dispatch ───────────────────────────────────────────────────────────────────────────────────

type Bench = { readonly gpu: GpuContext; readonly pipelines: Map<string, GPUComputePipeline> };

const pipelineOf = (bench: Bench, wgsl: string): GPUComputePipeline => {
  const hit = bench.pipelines.get(wgsl);
  if (hit !== undefined) return hit;
  const { device } = bench.gpu;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: wgsl }), entryPoint: "main" },
  });
  bench.pipelines.set(wgsl, pipeline);
  return pipeline;
};

const upload = (bench: Bench, data: ArrayBufferView<ArrayBuffer>, usage = STORAGE): GPUBuffer => {
  const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const buffer = bench.gpu.device.createBuffer({ size, usage });
  bench.gpu.device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

/** The bindings of one dispatch (index = binding; the output among them) and its size. */
type Inputs = {
  readonly buffers: readonly GPUBuffer[];
  readonly out: GPUBuffer;
  readonly outBytes: number;
};

const withOutput = (
  bench: Bench,
  before: readonly GPUBuffer[],
  outBytes: number,
  after: readonly GPUBuffer[],
): Inputs => {
  const out = bench.gpu.device.createBuffer({
    size: Math.max(16, Math.ceil(outBytes / 16) * 16),
    usage: STORAGE,
  });
  return { buffers: [...before, out, ...after], out, outBytes };
};

const destroyInputs = (inputs: Inputs): void => {
  for (const buffer of inputs.buffers) buffer.destroy();
};

/** Runs one WGSL over the inputs (the output reset to 0 first) and reads the output words back. */
const run = async (
  bench: Bench,
  inputs: Inputs,
  wgsl: string,
  workgroups: readonly [number, number, number],
): Promise<Uint32Array<ArrayBuffer>> => {
  const { device } = bench.gpu;
  device.pushErrorScope("validation");
  const pipeline = pipelineOf(bench, wgsl);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: inputs.buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const staging = device.createBuffer({
    size: inputs.out.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.clearBuffer(inputs.out);
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(...workgroups);
  pass.end();
  encoder.copyBufferToBuffer(inputs.out, 0, staging, 0, inputs.out.size);
  device.queue.submit([encoder.finish()]);
  const error = await device.popErrorScope();
  assertEquals(error, null, `validation: ${error?.message ?? ""}`);
  await staging.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(staging.getMappedRange().slice(0, inputs.outBytes));
  staging.unmap();
  staging.destroy();
  return out;
};

/** The words where two outputs differ (0 = identical). */
const differing = (left: Uint32Array, right: Uint32Array): number => {
  assertEquals(left.length, right.length, "output lengths");
  let count = 0;
  for (let index = 0; index < left.length; index += 1) if (left[index] !== right[index]) count += 1;
  return count;
};

/** The first differing index and its two words, for the failure message. */
const firstDifference = (left: Uint32Array, right: Uint32Array): string => {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) {
      return `[${index}] ${left[index].toString(16)} vs ${right[index].toString(16)}`;
    }
  }
  return "none";
};

/** The comparison is not vacuous: the reference output varies and is mostly finite. */
const assertInformative = (expected: Uint32Array, label: string): void => {
  const sample = expected.subarray(0, 8192);
  assert(new Set(sample).size > 1, `${label}: constant output`);
  let finite = 0;
  for (const word of sample) if ((word & 0x7f800000) !== 0x7f800000) finite += 1;
  assert(finite * 2 > sample.length, `${label}: mostly non-finite output`);
};

/** The non-finite words of an output (the adversarial inputs must reach the comparison). */
const nonFinite = (words: Uint32Array): number => {
  let count = 0;
  for (const word of words) if ((word & 0x7f800000) === 0x7f800000) count += 1;
  return count;
};

const withBench = async (body: (bench: Bench) => Promise<void>): Promise<void> => {
  const gpu = await acquireGpu();
  try {
    await body({ gpu, pipelines: new Map() });
  } finally {
    gpu.destroy();
  }
};

const withoutFields = ({ form: _form, decode: _decode, ...geometry }: GemmGeometry): GemmGeometry =>
  geometry;

// ── linear / matmul / bmm ──────────────────────────────────────────────────────────────────────

type GemmShape = {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly batch?: number;
};

const GEMM_SHAPES: readonly GemmShape[] = [
  { name: "m300 n1536 k1536", m: 300, n: 1536, k: 1536 },
  { name: "m1 n2048 k1536", m: 1, n: 2048, k: 1536 },
  { name: "m77 n136 k1028 (k % 16 = 4)", m: 77, n: 136, k: 1028 },
  { name: "m77 n136 k1024", m: 77, n: 136, k: 1024 },
  { name: "m129 n68 k48", m: 129, n: 68, k: 48 },
  { name: "m200 n1000 k256", m: 200, n: 1000, k: 256 },
  { name: "m33 n70 k207 (scalar: k % 4 = 3)", m: 33, n: 70, k: 207 },
];

const BMM_SHAPES: readonly GemmShape[] = [
  { name: "b3 m77 n136 k1028", batch: 3, m: 77, n: 136, k: 1028 },
  { name: "b2 m129 n68 k48", batch: 2, m: 129, n: 68, k: 48 },
  { name: "b16 m64 n64 k128", batch: 16, m: 64, n: 64, k: 128 },
  { name: "b2 m33 n70 k207 (scalar)", batch: 2, m: 33, n: 70, k: 207 },
];

/** The linear weight storages (i4 at group 32 on the shapes whose k is whole groups). */
const LINEAR_WEIGHTS: readonly WeightStorage[] = ["f32", "f16", "bf16", "i8", "i4"];

const linearInputs = (
  bench: Bench,
  shape: GemmShape,
  weight: WeightStorage,
  mode: Mode,
): Inputs => {
  const { m, n, k } = shape;
  const seed = m * 31 + n * 7 + k + weight.length * 1009;
  const scales = weight === "i8" ? n : weight === "i4" ? n * (k / 32) : 0;
  return withOutput(
    bench,
    [
      upload(bench, linearParams(m, n, k), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      upload(bench, f32Words(m * k, seed + 1, mode)),
      upload(bench, weightWords(weight, n * k, seed + 2, mode)),
      upload(bench, f32Words(n, seed + 3, "random")),
    ],
    m * n * 4,
    scales === 0 ? [] : [upload(bench, scaleWords(scales, seed + 4))],
  );
};

const gemmInputs = (bench: Bench, shape: GemmShape, mode: Mode): Inputs => {
  const { m, n, k } = shape;
  const batch = shape.batch ?? 1;
  const seed = m * 13 + n * 5 + k * 3 + batch;
  const params =
    shape.batch === undefined ? matmulParams(m, n, k) : bmmParams(m, n, k, shape.batch);
  return withOutput(
    bench,
    [
      upload(bench, params, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      upload(bench, f32Words(batch * m * k, seed + 1, mode)),
      upload(bench, f32Words(batch * k * n, seed + 2, mode, batch * k * n >= 512 ? 2 : 0)),
    ],
    batch * m * n * 4,
    [],
  );
};

const gemmWorkgroups = (
  shape: GemmShape,
  geometry: GemmGeometry,
): readonly [number, number, number] => [
  Math.ceil(shape.n / gemmTileN(geometry)),
  Math.ceil(shape.m / gemmTileM(geometry)),
  shape.batch ?? 1,
];

const linearWgslOf = (shape: GemmShape, weight: WeightStorage, geometry: GemmGeometry): string =>
  linearWgsl(
    weight,
    gemmUsesVec4(shape.k, shape.n),
    "f32",
    shape.m,
    weight === "i4" ? 32 : undefined,
    geometry,
  );

// ── conv2d ─────────────────────────────────────────────────────────────────────────────────────

type ConvShape = { readonly name: string; readonly dims: Conv2dDims };

const conv = (
  label: string,
  shape: {
    readonly batch?: number;
    readonly cin: number;
    readonly cout: number;
    readonly h: number;
    readonly w: number;
    readonly kernel: readonly [number, number];
    readonly stride?: readonly [number, number];
    readonly padding: readonly [number, number];
    readonly dilation?: readonly [number, number];
  },
): ConvShape => {
  const [kernelH, kernelW] = shape.kernel;
  const [strideH, strideW] = shape.stride ?? [1, 1];
  const [paddingH, paddingW] = shape.padding;
  const [dilationH, dilationW] = shape.dilation ?? [1, 1];
  const out = (size: number, kernel: number, stride: number, padding: number, dilation: number) =>
    Math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;
  return {
    name: label,
    dims: {
      batch: shape.batch ?? 1,
      channelsIn: shape.cin,
      channelsOut: shape.cout,
      heightIn: shape.h,
      widthIn: shape.w,
      heightOut: out(shape.h, kernelH, strideH, paddingH, dilationH),
      widthOut: out(shape.w, kernelW, strideW, paddingW, dilationW),
      kernelH,
      kernelW,
      strideH,
      strideW,
      paddingH,
      paddingW,
      dilationH,
      dilationW,
      groups: 1,
    },
  };
};

const CONV_SHAPES: readonly ConvShape[] = [
  conv("3x3 p1 Cin8 Cout128 20x20", {
    cin: 8,
    cout: 128,
    h: 20,
    w: 20,
    kernel: [3, 3],
    padding: [1, 1],
  }),
  conv("3x3 p1 Cin12 Cout40 12x16 b3 (K 108, Cout below a tile)", {
    batch: 3,
    cin: 12,
    cout: 40,
    h: 12,
    w: 16,
    kernel: [3, 3],
    padding: [1, 1],
  }),
  conv("3x3 p1 Cin16 Cout96 32x32 b2 (several workgroups)", {
    batch: 2,
    cin: 16,
    cout: 96,
    h: 32,
    w: 32,
    kernel: [3, 3],
    padding: [1, 1],
  }),
  conv("3x3 p1 Cin4 Cout16 4x4 (below one tile)", {
    cin: 4,
    cout: 16,
    h: 4,
    w: 4,
    kernel: [3, 3],
    padding: [1, 1],
  }),
  conv("3x3 stride_h 2 p1 Cin8 Cout64 17x24", {
    cin: 8,
    cout: 64,
    h: 17,
    w: 24,
    kernel: [3, 3],
    stride: [2, 1],
    padding: [1, 1],
  }),
  conv("3x3 p0 Cin8 Cout64 18x18", {
    cin: 8,
    cout: 64,
    h: 18,
    w: 18,
    kernel: [3, 3],
    padding: [0, 0],
  }),
  conv("3x3 dilation 2 p2 Cin8 Cout64 20x20", {
    cin: 8,
    cout: 64,
    h: 20,
    w: 20,
    kernel: [3, 3],
    padding: [2, 2],
    dilation: [2, 2],
  }),
  conv("1x1 Cin64 Cout72 16x16", {
    cin: 64,
    cout: 72,
    h: 16,
    w: 16,
    kernel: [1, 1],
    padding: [0, 0],
  }),
  conv("5x5 p2 Cin8 Cout64 24x40", {
    cin: 8,
    cout: 64,
    h: 24,
    w: 40,
    kernel: [5, 5],
    padding: [2, 2],
  }),
  conv("7x7 p3 Cin4 Cout32 32x32", {
    cin: 4,
    cout: 32,
    h: 32,
    w: 32,
    kernel: [7, 7],
    padding: [3, 3],
  }),
  conv("1x3 p(0,1) Cin16 Cout64 8x16", {
    cin: 16,
    cout: 64,
    h: 8,
    w: 16,
    kernel: [1, 3],
    padding: [0, 1],
  }),
  conv("3x1 p(1,0) Cin16 Cout64 8x16", {
    cin: 16,
    cout: 64,
    h: 8,
    w: 16,
    kernel: [3, 1],
    padding: [1, 0],
  }),
  conv("3x3 stride 2 p1 Cin8 Cout64 17x24 (scalar: stride_w 2)", {
    cin: 8,
    cout: 64,
    h: 17,
    w: 24,
    kernel: [3, 3],
    stride: [2, 2],
    padding: [1, 1],
  }),
  conv("3x3 p1 Cin2 Cout64 16x16 (scalar: K 18)", {
    cin: 2,
    cout: 64,
    h: 16,
    w: 16,
    kernel: [3, 3],
    padding: [1, 1],
  }),
];

const CONV_WEIGHTS: readonly WeightStorage[] = ["f32", "f16", "bf16", "i8"];

const convFlat = (dims: Conv2dDims): number => dims.channelsIn * dims.kernelH * dims.kernelW;

const convUsesVec4 = (dims: Conv2dDims): boolean =>
  conv2dUsesVec4(convFlat(dims), dims.widthOut, dims.strideW);

const convInputs = (bench: Bench, shape: ConvShape, weight: WeightStorage, mode: Mode): Inputs => {
  const { dims } = shape;
  const seed = dims.channelsIn * 101 + dims.channelsOut * 7 + dims.heightIn * 3 + dims.kernelW;
  const pixelsOut = dims.heightOut * dims.widthOut;
  return withOutput(
    bench,
    [
      upload(bench, conv2dIgemmParams(dims), GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      upload(
        bench,
        f32Words(dims.batch * dims.channelsIn * dims.heightIn * dims.widthIn, seed + 1, mode),
      ),
      upload(bench, weightWords(weight, dims.channelsOut * convFlat(dims), seed + 2, mode)),
      upload(bench, f32Words(dims.channelsOut, seed + 3, "random")),
    ],
    dims.batch * dims.channelsOut * pixelsOut * 4,
    weight === "i8" ? [upload(bench, scaleWords(dims.channelsOut, seed + 4))] : [],
  );
};

const convWorkgroups = (
  dims: Conv2dDims,
  geometry: GemmGeometry,
): readonly [number, number, number] => [
  Math.ceil((dims.heightOut * dims.widthOut) / gemmTileN(geometry)),
  Math.ceil(dims.channelsOut / gemmTileM(geometry)),
  dims.batch,
];

const convWgslOf = (dims: Conv2dDims, weight: WeightStorage, geometry: GemmGeometry): string =>
  conv2dIgemmWgsl(weight, convUsesVec4(dims), conv2dIgemmMTile(dims.channelsOut), geometry);

/** The conv2d variants: each form, the decode alone, and the decode with each form. */
const CONV_VARIANTS: readonly Pick<GemmGeometry, "form" | "decode">[] = [
  ...FORMS.map((form) => ({ form })),
  { decode: "counter" },
  ...FORMS.map((form) => ({ form, decode: "counter" as const })),
];

const variantName = ({ form, decode }: Pick<GemmGeometry, "form" | "decode">): string =>
  [form, decode].filter((part) => part !== undefined).join("+");

const MODES: readonly Mode[] = ["random", "adversarial"];

describe.skipIf(!GPU_AVAILABLE)(
  "the f32 skeleton forms and the conv2d counter decode return the words of the same geometry without them (real GPU)",
  () => {
    it("linear (f32 / f16 / bf16 / i8 / i4 g32 weights): each form on every profile geometry × shape × input set", async () => {
      await withBench(async (bench) => {
        let compared = 0;
        let scalarSame = 0;
        let loud = 0;
        for (const mode of MODES) {
          for (const shape of GEMM_SHAPES) {
            for (const weight of LINEAR_WEIGHTS) {
              if (weight === "i4" && shape.k % 32 !== 0) continue;
              const inputs = linearInputs(bench, shape, weight, mode);
              try {
                for (const base of GEMM_GEOMETRIES) {
                  const plain = linearWgslOf(shape, weight, base);
                  const expected = await run(bench, inputs, plain, gemmWorkgroups(shape, base));
                  const label = `linear ${weight} ${shape.name} ${name(base)} ${mode}`;
                  assertInformative(expected, label);
                  if (mode === "adversarial") loud += nonFinite(expected);
                  for (const form of FORMS) {
                    const wgsl = linearWgslOf(shape, weight, { ...base, form });
                    if (!gemmUsesVec4(shape.k, shape.n)) {
                      // The scalar kernel keeps the original form (the codegen test pins the key)
                      assertEquals(wgsl, plain, `${label} ${form}: the scalar WGSL moved`);
                      scalarSame += 1;
                      continue;
                    }
                    const actual = await run(bench, inputs, wgsl, gemmWorkgroups(shape, base));
                    assertEquals(
                      differing(actual, expected),
                      0,
                      `${label} ${form}: first difference ${firstDifference(actual, expected)}`,
                    );
                    compared += 1;
                  }
                }
              } finally {
                destroyInputs(inputs);
              }
            }
          }
        }
        console.log(
          `[karume] K-88 linear: ${compared} form comparisons identical, ${scalarSame} scalar kernels unchanged (${GEMM_GEOMETRIES.length} geometries, ${loud} non-finite reference words)`,
        );
        assert(compared > 0 && scalarSame > 0 && loud > 0);
      });
    });

    it("matmul and bmm: each form on every profile geometry × shape × input set", async () => {
      await withBench(async (bench) => {
        let compared = 0;
        let loud = 0;
        for (const mode of MODES) {
          for (const shape of [
            ...GEMM_SHAPES.filter((entry) => entry.m * entry.k <= 1 << 20),
            ...BMM_SHAPES,
          ]) {
            const isBmm = shape.batch !== undefined;
            const v4 = gemmUsesVec4(shape.k, shape.n);
            const wgslOf = (geometry: GemmGeometry): string =>
              isBmm ? bmmWgsl(v4, shape.m, undefined, geometry) : matmulWgsl(v4, shape.m, geometry);
            const inputs = gemmInputs(bench, shape, mode);
            try {
              for (const base of GEMM_GEOMETRIES) {
                const plain = wgslOf(base);
                const expected = await run(bench, inputs, plain, gemmWorkgroups(shape, base));
                const label = `${isBmm ? "bmm" : "matmul"} ${shape.name} ${name(base)} ${mode}`;
                assertInformative(expected, label);
                if (mode === "adversarial") loud += nonFinite(expected);
                for (const form of FORMS) {
                  const wgsl = wgslOf({ ...base, form });
                  if (!v4) {
                    assertEquals(wgsl, plain, `${label} ${form}: the scalar WGSL moved`);
                    continue;
                  }
                  const actual = await run(bench, inputs, wgsl, gemmWorkgroups(shape, base));
                  assertEquals(
                    differing(actual, expected),
                    0,
                    `${label} ${form}: first difference ${firstDifference(actual, expected)}`,
                  );
                  compared += 1;
                }
              }
            } finally {
              destroyInputs(inputs);
            }
          }
        }
        console.log(
          `[karume] K-88 matmul / bmm: ${compared} form comparisons identical (${loud} non-finite reference words)`,
        );
        assert(compared > 0 && loud > 0);
      });
    });

    it("conv2d (f32 / f16 / bf16 / i8 weights): each form, the decode alone and with each form, on every profile geometry × shape × input set", async () => {
      await withBench(async (bench) => {
        let compared = 0;
        let scalarSame = 0;
        let loud = 0;
        for (const mode of MODES) {
          for (const shape of CONV_SHAPES) {
            const v4 = convUsesVec4(shape.dims);
            for (const weight of CONV_WEIGHTS) {
              const inputs = convInputs(bench, shape, weight, mode);
              try {
                for (const base of CONV_GEOMETRIES) {
                  const plain = convWgslOf(shape.dims, weight, base);
                  const workgroups = convWorkgroups(shape.dims, base);
                  const expected = await run(bench, inputs, plain, workgroups);
                  const label = `conv2d ${weight} ${shape.name} ${name(base)} ${mode}`;
                  assertInformative(expected, label);
                  if (mode === "adversarial") loud += nonFinite(expected);
                  for (const variant of CONV_VARIANTS) {
                    const wgsl = convWgslOf(shape.dims, weight, { ...base, ...variant });
                    if (!v4) {
                      assertEquals(
                        wgsl,
                        plain,
                        `${label} ${variantName(variant)}: the scalar WGSL moved`,
                      );
                      scalarSame += 1;
                      continue;
                    }
                    const actual = await run(bench, inputs, wgsl, workgroups);
                    assertEquals(
                      differing(actual, expected),
                      0,
                      `${label} ${variantName(variant)}: first difference ${firstDifference(
                        actual,
                        expected,
                      )}`,
                    );
                    compared += 1;
                  }
                }
              } finally {
                destroyInputs(inputs);
              }
            }
          }
        }
        console.log(
          `[karume] K-88 conv2d: ${compared} variant comparisons identical, ${scalarSame} scalar kernels unchanged (${CONV_GEOMETRIES.length} geometries, ${loud} non-finite reference words)`,
        );
        assert(compared > 0 && scalarSame > 0 && loud > 0);
      });
    });
  },
);

// ── Fault injection ────────────────────────────────────────────────────────────────────────────

/** Applies `pattern` once and fails if it did not match (a no-op mutation would be green). */
const replaceOnce = (wgsl: string, pattern: RegExp | string, replacement: string): string => {
  const mutated = wgsl.replace(pattern, replacement);
  assertNotEquals(mutated, wgsl, `the mutation ${String(pattern)} did not apply`);
  return mutated;
};

/** A text mutation of the generated WGSL (the fault the comparison must see). */
type Mutation = {
  readonly name: string;
  readonly mutate: (wgsl: string) => string;
  /** The shapes on which the fault MUST be seen (it is logged on every shape). */
  readonly pinned: readonly string[];
};

/** The conv2d variant every conv2d fault mutates: both forms' parts and the decode. */
const CONV_FAULT_GEOMETRY: GemmGeometry = {
  ...DEFAULT_GEOMETRY_PROFILE.conv2d.rows64,
  form: "tas-tb-un",
  decode: "counter",
};

const CONV_FAULT_SHAPES: readonly ConvShape[] = CONV_SHAPES.filter(({ dims }) =>
  convUsesVec4(dims),
);

const PRIMARY_CONV = "3x3 p1 Cin8 Cout128 20x20";

const CONV_MUTATIONS: readonly Mutation[] = [
  {
    // The counter of slot 0 loses the carry from kw into kh (it wraps kw and keeps kh)
    name: "counter: a dropped kw → kh carry (slot 0)",
    mutate: (wgsl) =>
      replaceOnce(
        wgsl,
        "if (c_kw0 >= dims.kernel_w) { c_kw0 = c_kw0 - dims.kernel_w; c_kh0 = c_kh0 + 1u; }",
        "if (c_kw0 >= dims.kernel_w) { c_kw0 = c_kw0 - dims.kernel_w; }",
      ),
    pinned: [PRIMARY_CONV],
  },
  {
    // The kw digit of the K tile's step is taken modulo the wrong radix (kernel_h): only a kernel
    // with kernel_h ≠ kernel_w can see it
    name: "counter: the kw step modulo kernel_h",
    mutate: (wgsl) =>
      replaceOnce(
        wgsl,
        "let step_kw = step_r % dims.kernel_w;",
        "let step_kw = step_r % dims.kernel_h;",
      ),
    pinned: ["1x3 p(0,1) Cin16 Cout64 8x16", "3x1 p(1,0) Cin16 Cout64 8x16"],
  },
  {
    // The edge path no longer zeroes a quad whose input row is outside the image (it reads the
    // neighbouring row of x instead)
    name: "edge: the dropped row check of xcol4",
    mutate: (wgsl) =>
      replaceOnce(
        wgsl,
        "  if (iy < 0 || u32(iy) >= dims.height_in) {\n    return vec4<f32>(0.0);\n  }\n",
        "",
      ),
    pinned: [PRIMARY_CONV],
  },
  {
    // The edge path no longer zeroes an element left or right of the image
    name: "edge: the dropped column check of xcol_row",
    mutate: (wgsl) =>
      replaceOnce(wgsl, "  if (ix < 0 || u32(ix) >= dims.width_in) {\n    return 0.0;\n  }\n", ""),
    pinned: [PRIMARY_CONV],
  },
  {
    // Slot 0's contiguous path takes a quad that crosses the right edge of the image
    name: "edge: the contiguous-path guard without the quad's last column (slot 0)",
    mutate: (wgsl) =>
      replaceOnce(
        wgsl,
        "ix0 >= 0 && u32(ix0) + 3u < dims.width_in) {",
        "ix0 >= 0 && u32(ix0) < dims.width_in) {",
      ),
    pinned: [PRIMARY_CONV],
  },
];

/** Skeleton faults, applied to the conv2d fault variant and to linear at the default geometry. */
const SKELETON_MUTATIONS = (tileM: number, wgX: number): readonly Mutation[] => [
  {
    // The store maps a thread's second column quad next to its first (the original form) while
    // the update read it a workgroup row apart
    name: "skeleton: the store's quad 1 adjacent to quad 0 (TB)",
    mutate: (wgsl) => replaceOnce(wgsl, `let ocq1 = ocq0 + ${wgX}u;`, "let ocq1 = ocq0 + 1u;"),
    pinned: [],
  },
  {
    // Component y of A slot 0 stored row-major in the k-major tile (TAs)
    name: "skeleton: A slot 0 lane y at the row-major offset (TAs)",
    mutate: (wgsl) =>
      replaceOnce(wgsl, `sa[sa_base0 + ${tileM}u] = av0.y;`, "sa[sa_base0 + 1u] = av0.y;"),
    pinned: [],
  },
  {
    // One unrolled block repeats the previous K step and skips its own (UN)
    name: "skeleton: unrolled block 5 runs K step 4 (UN)",
    mutate: (wgsl) => replaceOnce(wgsl, "      let kk = 5u;\n", "      let kk = 4u;\n"),
    pinned: [],
  },
];

const LINEAR_FAULT_GEOMETRY: GemmGeometry = {
  ...DEFAULT_GEOMETRY_PROFILE.gemmRows[DEFAULT_GEOMETRY_PROFILE.gemmRows.length - 1].geometry,
  form: "tas-tb-un",
};

const LINEAR_FAULT_SHAPES = GEMM_SHAPES.filter((shape) => gemmUsesVec4(shape.k, shape.n));

describe.skipIf(!GPU_AVAILABLE)(
  "fault injection: each mutation of the counter decode, its edge path or the skeleton forms turns the comparison red (real GPU)",
  () => {
    it("conv2d: two counter faults, three edge faults and three skeleton faults, each red on its pinned shapes", async () => {
      const mutations = [
        ...CONV_MUTATIONS,
        ...SKELETON_MUTATIONS(gemmTileM(CONV_FAULT_GEOMETRY), CONV_FAULT_GEOMETRY.wgX).map(
          (mutation) => ({ ...mutation, pinned: [PRIMARY_CONV] }),
        ),
      ];
      await withBench(async (bench) => {
        const caught = new Map<string, string[]>();
        for (const shape of CONV_FAULT_SHAPES) {
          const inputs = convInputs(bench, shape, "f32", "random");
          try {
            const base = withoutFields(CONV_FAULT_GEOMETRY);
            const workgroups = convWorkgroups(shape.dims, base);
            const expected = await run(
              bench,
              inputs,
              convWgslOf(shape.dims, "f32", base),
              workgroups,
            );
            const wgsl = convWgslOf(shape.dims, "f32", CONV_FAULT_GEOMETRY);
            // Control: the unmutated variant is identical
            assertEquals(differing(await run(bench, inputs, wgsl, workgroups), expected), 0);
            for (const mutation of mutations) {
              const count = differing(
                await run(bench, inputs, mutation.mutate(wgsl), workgroups),
                expected,
              );
              const entry = caught.get(mutation.name) ?? [];
              entry.push(`${shape.name}: ${count}/${expected.length}`);
              caught.set(mutation.name, entry);
              if (mutation.pinned.includes(shape.name)) {
                assert(count > 0, `${mutation.name} is not seen on ${shape.name}`);
              }
            }
          } finally {
            destroyInputs(inputs);
          }
        }
        for (const [mutation, shapes] of caught) {
          console.log(
            `[karume] K-88 conv2d fault '${mutation}' differing words: ${shapes.join("; ")}`,
          );
        }
        assertEquals(
          [...caught.keys()],
          mutations.map((mutation) => mutation.name),
        );
      });
    });

    it("linear: the three skeleton faults, each red on its pinned shapes (logged on every v4 shape)", async () => {
      const mutations = SKELETON_MUTATIONS(
        gemmTileM(LINEAR_FAULT_GEOMETRY),
        LINEAR_FAULT_GEOMETRY.wgX,
      ).map((mutation) => ({ ...mutation, pinned: ["m300 n1536 k1536", "m200 n1000 k256"] }));
      await withBench(async (bench) => {
        const caught = new Map<string, string[]>();
        for (const shape of LINEAR_FAULT_SHAPES) {
          const inputs = linearInputs(bench, shape, "f32", "random");
          try {
            const base = withoutFields(LINEAR_FAULT_GEOMETRY);
            const workgroups = gemmWorkgroups(shape, base);
            const expected = await run(bench, inputs, linearWgslOf(shape, "f32", base), workgroups);
            const wgsl = linearWgslOf(shape, "f32", LINEAR_FAULT_GEOMETRY);
            assertEquals(differing(await run(bench, inputs, wgsl, workgroups), expected), 0);
            for (const mutation of mutations) {
              const count = differing(
                await run(bench, inputs, mutation.mutate(wgsl), workgroups),
                expected,
              );
              const entry = caught.get(mutation.name) ?? [];
              entry.push(`${shape.name}: ${count}/${expected.length}`);
              caught.set(mutation.name, entry);
              if (mutation.pinned.includes(shape.name)) {
                assert(count > 0, `${mutation.name} is not seen on ${shape.name}`);
              }
            }
          } finally {
            destroyInputs(inputs);
          }
        }
        for (const [mutation, shapes] of caught) {
          console.log(
            `[karume] K-88 linear fault '${mutation}' differing words: ${shapes.join("; ")}`,
          );
        }
        assertEquals(
          [...caught.keys()],
          mutations.map((mutation) => mutation.name),
        );
      });
    });
  },
);
