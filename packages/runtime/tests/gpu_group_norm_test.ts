// The group_norm kernel's four-point check (ADR 0110 decision 3 — ADR 0138 decision 7) and the
// large-group check at the MOSS decoder's area cap.
//
// ① f64 band: the Session's output against a float64 two-pass group norm of the same f32 inputs,
//    in the error unit e (below). The oracle derives its rows and channels from the shape and the
//    group count itself (it does not use the kernel's layout function). The band is this test's
//    own, stricter than the op's COMPOSITE tolerance class: the decision cases' worst × 4, far
//    below the theoretical bound of the summation depth. It holds on separate acceptance cases,
//    two of which have a group variance of the order of eps (the cases where eps's place and
//    value matter).
// ② A/B against the decomposed reference path (small shapes): reshape → layer_norm (ones / zeros,
//    one workgroup per row, its own two-pass order) → reshape → mul → add on the same GPU, in the
//    same unit, under its own band (≥ 4 × the measured worst, ≤ 2 × the f64 band — the ADR 0110
//    decision 4 rule for an A/B band).
// ③ structure, exact: two runs and a device with the per-dimension workgroup limit lowered to 3
//    (every dispatch grid-strides) are Uint32-identical; a zero-variance group of a power-of-two
//    length gives exactly the bias; a NaN poisons its own group only.
// ④ reverse fault injection (the check is not vacuous): kernels with the one-pass variance
//    `E[x²] − E[x]²`, with the last partial chunk dropped, with the affine read from the wrong
//    group, with eps outside the square root or with eps × 10 land outside the f64 band by more
//    than 2× (the ADR 0118 addendum margin). The one-pass fault runs on an input with mean +100 —
//    the cancellation the two-pass discipline exists for; the eps faults run on an input whose
//    group variance is of the order of eps (at a variance of O(1) eps moves only a few ULPs).
// The estimate: the runtime estimate's workspace of a group_norm graph equals the plan backing the
// Session builds (the two in-node temporaries are counted with the recipe builder's lifetimes).
//
// The error unit: e = max_i |gpu_i − ref_i| / (ε · (rstd_g · M_g · W + B)) with ε = 2^-23, the
// group g of element i, M_g = max |x| over g, W = max |weight|, B = max |bias|. A group's outputs
// carry the rounding of its mean (∝ ε·M_g·rstd_g·|w|), of its variance (∝ ε·|y − b|) and of the
// affine (∝ ε·|y|); e counts them in ULPs of that scale, so groups with a large mean or a large
// spread are judged on the same footing.
//
// Large group (runs in the default GPU lane — about 10 s, 2 × 2.1 GB of host arrays and about
// 4.3 GB of VRAM on the RTX 3080 Ti): `[1, 192, 1696, 1648]` in 32 groups — 6 channels × 2,795,008
// pixels = 16,770,048 elements per group, the largest GroupNorm of the MOSS decoder at its area cap
// (192·4 B per pixel ≤ the 2 GiB binding). Its partial stage has 32 × 16,377 = 524,064 workgroups,
// so the grid-stride wraps 8 times over the real element count at the device's own limit (65,535 —
// the grid-stride test reaches the wrap only by lowering the limit). The input has mean +100 in one
// half of the groups. Skipped explicitly, with the reason, on a device whose binding limit cannot
// hold the tensor.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { BUFFER_USAGE as U, MAP_MODE } from "../src/gpu/webgpu-constants.ts";
import {
  GROUP_NORM_APPLY_WGSL,
  GROUP_NORM_PARTIAL_WGSL,
  GROUP_NORM_TOTAL_WGSL,
  GROUP_NORM_WORKGROUP_SIZE,
  groupNormApplyParams,
  type GroupNormLayout,
  groupNormLayout,
  groupNormPartialParams,
  groupNormTotalParams,
} from "../src/kernels/group-norm.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { openGraph } from "./helpers/estimate-graphs.ts";
import { type DeclarationJson, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

const EPSILON = 2 ** -23;
const EPS = 1e-6;

/**
 * The f64 band of e (①) — this test's own band, not an ADR 0110 decision 4 rule (R's band would be
 * the op's COMPOSITE class of `helpers/op-tolerance.ts`, which this one is far inside). Derived
 * from the six decision cases below on an RTX 3080 Ti (Node + Dawn, 2026-10-27): measured e 1.290
 * / 0.940 / 1.117 / 1.117 / 1.363 / 1.363, worst 1.363 (`[1,96,128,96]`) → × 4 = 5.45 → 5.5 (two
 * significant digits, rounded up). The theoretical bound for the summation depth of these shapes
 * (D = 4 sequential + 8 tree + ⌈P/256⌉ sequential + 8 tree + 1 scale = 22, e ≲ 2D + 10 = 54) is
 * far above it. The acceptance cases then measured 1.258 / 1.002 / 1.277 and the large group 1.476
 * (D = 85, bound 180) — all within the band; the band was not changed after seeing them. The two
 * eps-scale acceptance cases (group variance about eps) were added later and measured 0.927 /
 * 1.055, also within the unchanged band.
 */
const F64_BAND = 5.5;

/**
 * The A/B band of e against the decomposed reference (②): measured 0.490 … 1.702 over all nine
 * cases (the ADR 0110 rule for an A/B band takes the worst measured), 1.702 × 4 = 6.81 → 7, below
 * the cap of 2 × the f64 band (11).
 */
const AB_BAND = 7;

/** The fault-injection margin (a fault must land outside 2 × the f64 band). */
const FAULT_MARGIN = 2;

type Case = {
  readonly name: string;
  readonly shape: readonly number[];
  readonly groups: number;
  /** Added to every element of the odd groups (the even groups stay centred). */
  readonly offset: number;
  /** Multiplies every element's spread (default 1 — a group variance of O(0.1 … 1)). */
  readonly spread?: number;
};

/**
 * Group variances of the order of eps (spread × 1e-3: variance ≈ 1e-7 … 1.3e-6 against eps =
 * 1e-6), where `1 / sqrt(var + eps)` and `1 / (sqrt(var) + eps)` or another eps differ by tens of
 * percent — at the other cases' O(1) variances eps moves the output by a few ULPs only.
 */
const EPS_SCALE_CASES: readonly Case[] = [
  {
    name: "[1,64,16,16] G32 variance ~ eps",
    shape: [1, 64, 16, 16],
    groups: 32,
    offset: 0,
    spread: 1e-3,
  },
  {
    name: "[1,192,40,40] G32 variance ~ eps, mean +0.01",
    shape: [1, 192, 40, 40],
    groups: 32,
    offset: 0.01,
    spread: 1e-3,
  },
];

const DECISION_CASES: readonly Case[] = [
  { name: "[2,64,9,11] G32 (groups of 198)", shape: [2, 64, 9, 11], groups: 32, offset: 0 },
  { name: "[2,64,9,11] G32 mean +100", shape: [2, 64, 9, 11], groups: 32, offset: 100 },
  { name: "[1,192,40,40] G32 (10 chunks)", shape: [1, 192, 40, 40], groups: 32, offset: 0 },
  { name: "[1,192,40,40] G32 mean +100", shape: [1, 192, 40, 40], groups: 32, offset: 100 },
  { name: "[1,96,128,96] G32 (36 chunks)", shape: [1, 96, 128, 96], groups: 32, offset: 0 },
  { name: "[1,96,128,96] G32 mean +100", shape: [1, 96, 128, 96], groups: 32, offset: 100 },
];

const ACCEPT_CASES: readonly Case[] = [
  { name: "[1,384,32,48] G32 mean -50", shape: [1, 384, 32, 48], groups: 32, offset: -50 },
  { name: "[3,12,7,5] G4 mean +100", shape: [3, 12, 7, 5], groups: 4, offset: 100 },
  { name: "[1,384,4096] G32 mean +3 (rank 3)", shape: [1, 384, 4096], groups: 32, offset: 3 },
  ...EPS_SCALE_CASES,
];

/** A deterministic hash in [-1, 1) (no Math.random — failures must reproduce). */
const unit = (index: number, salt: number): number => {
  let h = Math.imul(index ^ salt, 0x9e3779b1) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b) >>> 0;
  h = (h ^ (h >>> 13)) >>> 0;
  return h / 2 ** 31 - 1;
};

type Inputs = {
  readonly x: Float32Array<ArrayBuffer>;
  readonly weight: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
};

/**
 * The oracle's own row view of `x[N,C,…]` in `groups` groups (①'s reference, the inputs and the
 * error unit): computed here from the shape, not taken from the kernel's `groupNormLayout`, so a
 * wrong layout in the kernel module cannot also move the reference.
 */
type Rows = {
  readonly rows: number;
  readonly length: number;
  readonly inner: number;
  readonly groups: number;
  readonly perGroup: number;
  readonly count: number;
};

const rowsOf = (shape: readonly number[], groups: number): Rows => {
  const [batch, channels] = shape;
  const inner = shape.slice(2).reduce((count, dim) => count * dim, 1);
  const perGroup = channels / groups;
  assert(Number.isInteger(perGroup), `${groups} groups do not divide ${channels} channels`);
  const length = perGroup * inner;
  return { rows: batch * groups, length, inner, groups, perGroup, count: batch * channels * inner };
};

/** The channel of element `index` (row-major `x[N,C,…]`). */
const channelOf = (index: number, layout: Rows): number =>
  Math.floor(index / layout.inner) % (layout.groups * layout.perGroup);

/** The inputs of one case: x in [-1, 1) scaled per channel (×0.5 … ×2) plus the group offset. */
const inputsOf = (testCase: Case): Inputs => {
  const layout = rowsOf(testCase.shape, testCase.groups);
  const channels = testCase.shape[1];
  const scale = testCase.spread ?? 1;
  const x = new Float32Array(layout.count);
  for (let row = 0; row < layout.rows; row += 1) {
    const offset = row % 2 === 1 ? testCase.offset : 0;
    const base = row * layout.length;
    for (let i = 0; i < layout.length; i += 1) {
      const channel = channelOf(base + i, layout);
      const spread = scale * (0.5 + (1.5 * ((channel * 37) % 11)) / 10);
      x[base + i] = offset + spread * unit(base + i, 0x5eed);
    }
  }
  const weight = new Float32Array(channels);
  const bias = new Float32Array(channels);
  for (let c = 0; c < channels; c += 1) {
    weight[c] = 0.4 + 1.2 * (0.5 + 0.5 * unit(c, 0x77));
    bias[c] = 0.5 * unit(c, 0x99);
  }
  return { x, weight, bias };
};

type Reference = {
  /** The f64 output (not rounded). */
  readonly out: (index: number) => number;
  /** e's denominator per row: rstd_g · M_g · W + B. */
  readonly scale: Float64Array<ArrayBuffer>;
};

/** The float64 two-pass group norm of the f32 inputs (①'s oracle — no padding, no tree). */
const reference = (inputs: Inputs, layout: Rows): Reference => {
  const { x, weight, bias } = inputs;
  const mean = new Float64Array(layout.rows);
  const rstd = new Float64Array(layout.rows);
  const scale = new Float64Array(layout.rows);
  const maxWeight = weight.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
  const maxBias = bias.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
  for (let row = 0; row < layout.rows; row += 1) {
    const base = row * layout.length;
    let sum = 0;
    let magnitude = 0;
    for (let i = 0; i < layout.length; i += 1) {
      sum += x[base + i];
      magnitude = Math.max(magnitude, Math.abs(x[base + i]));
    }
    mean[row] = sum / layout.length;
    let squares = 0;
    for (let i = 0; i < layout.length; i += 1) {
      const deviation = x[base + i] - mean[row];
      squares += deviation * deviation;
    }
    rstd[row] = 1 / Math.sqrt(squares / layout.length + Math.fround(EPS));
    scale[row] = rstd[row] * magnitude * maxWeight + maxBias;
  }
  return {
    out: (index) => {
      const row = Math.floor(index / layout.length);
      const channel = channelOf(index, layout);
      return (x[index] - mean[row]) * rstd[row] * weight[channel] + bias[channel];
    },
    scale,
  };
};

/** e of `actual` against the f64 reference (module doc). */
const errorUnits = (actual: Float32Array<ArrayBuffer>, ref: Reference, layout: Rows): number => {
  let worst = 0;
  for (let index = 0; index < actual.length; index += 1) {
    const row = Math.floor(index / layout.length);
    const e = Math.abs(actual[index] - ref.out(index)) / (EPSILON * ref.scale[row]);
    if (!(e <= worst)) worst = e; // NaN wins
  }
  return worst;
};

/** e between two GPU outputs, in the same unit (②). */
const abUnits = (
  left: Float32Array<ArrayBuffer>,
  right: Float32Array<ArrayBuffer>,
  ref: Reference,
  layout: Rows,
): number => {
  let worst = 0;
  for (let index = 0; index < left.length; index += 1) {
    const row = Math.floor(index / layout.length);
    const e = Math.abs(left[index] - right[index]) / (EPSILON * ref.scale[row]);
    if (!(e <= worst)) worst = e;
  }
  return worst;
};

const groupNormGraph = (shape: readonly number[], groups: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["group_norm"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [...shape] },
    { name: "w", dtype: "f32", shape: [shape[1]] },
    { name: "b", dtype: "f32", shape: [shape[1]] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [...shape] } },
  nodes: [
    {
      op: "group_norm",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { num_groups: groups, eps: EPS },
    },
  ],
});

/** ②'s reference path: the decomposition through the existing layer_norm and broadcasts. */
const decomposedGraph = (shape: readonly number[], groups: number): DeclarationJson => {
  const layout = rowsOf(shape, groups);
  const [batch, channels] = shape;
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["add", "layer_norm", "mul", "reshape"] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: [...shape] },
      { name: "ones", dtype: "f32", shape: [layout.length] },
      { name: "zeros", dtype: "f32", shape: [layout.length] },
      { name: "w", dtype: "f32", shape: [channels, 1] },
      { name: "b", dtype: "f32", shape: [channels, 1] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      rows: { dtype: "f32", shape: [batch, groups, layout.length] },
      normed: { dtype: "f32", shape: [batch, groups, layout.length] },
      flat: { dtype: "f32", shape: [batch, channels, layout.inner] },
      scaled: { dtype: "f32", shape: [batch, channels, layout.inner] },
      shifted: { dtype: "f32", shape: [batch, channels, layout.inner] },
      y: { dtype: "f32", shape: [...shape] },
    },
    nodes: [
      { op: "reshape", ins: ["x"], outs: ["rows"], attrs: {} },
      {
        op: "layer_norm",
        ins: ["rows", "ones", "zeros"],
        outs: ["normed"],
        attrs: { normalized_shape: [layout.length], eps: EPS },
      },
      { op: "reshape", ins: ["normed"], outs: ["flat"], attrs: {} },
      { op: "mul", ins: ["flat", "w"], outs: ["scaled"], attrs: {} },
      { op: "add", ins: ["scaled", "b"], outs: ["shifted"], attrs: {} },
      { op: "reshape", ins: ["shifted"], outs: ["y"], attrs: {} },
    ],
  };
};

const f32 = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/** Runs `graph` `times` times on one Session; returns each output and the last census. */
const runGraph = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  inputs: Readonly<Record<string, Tensor>>,
  times = 1,
): Promise<{
  readonly outputs: readonly Float32Array<ArrayBuffer>[];
  readonly census: ReturnType<typeof requireCensus>;
}> => {
  const session = await createSessionFromContainer(
    gpu,
    await openModelBytes(graph, []),
    GRAPH_NAME,
  );
  try {
    const outputs: Float32Array<ArrayBuffer>[] = [];
    for (let run = 0; run < times; run += 1) {
      const result = await session.run(inputs);
      const y = result["y"];
      if (y.dtype !== "f32") throw new Error(`y is ${y.dtype}`);
      outputs.push(y.data);
    }
    return { outputs, census: requireCensus(session.diagnostics().lastRunPipelines, "group_norm") };
  } finally {
    await session.dispose();
  }
};

const runGroupNorm = async (
  gpu: GpuContext,
  testCase: Pick<Case, "shape" | "groups">,
  inputs: Inputs,
  times = 1,
) =>
  runGraph(
    gpu,
    groupNormGraph(testCase.shape, testCase.groups),
    {
      x: f32(testCase.shape, inputs.x),
      w: f32([testCase.shape[1]], inputs.weight),
      b: f32([testCase.shape[1]], inputs.bias),
    },
    times,
  );

const runDecomposed = async (gpu: GpuContext, testCase: Case, inputs: Inputs) => {
  const layout = rowsOf(testCase.shape, testCase.groups);
  const channels = testCase.shape[1];
  return runGraph(gpu, decomposedGraph(testCase.shape, testCase.groups), {
    x: f32(testCase.shape, inputs.x),
    ones: f32([layout.length], new Float32Array(layout.length).fill(1)),
    zeros: f32([layout.length], new Float32Array(layout.length)),
    w: f32([channels, 1], inputs.weight),
    b: f32([channels, 1], inputs.bias),
  });
};

const bitsEqual = (left: Float32Array<ArrayBuffer>, right: Float32Array<ArrayBuffer>): boolean => {
  const a = new Uint32Array(left.buffer, left.byteOffset, left.length);
  const b = new Uint32Array(right.buffer, right.byteOffset, right.length);
  return a.length === b.length && a.every((value, index) => value === b[index]);
};

type Kernels = { readonly partial: string; readonly total: string; readonly apply: string };

const REAL_KERNELS: Kernels = {
  partial: GROUP_NORM_PARTIAL_WGSL,
  total: GROUP_NORM_TOTAL_WGSL,
  apply: GROUP_NORM_APPLY_WGSL,
};

/** One replacement in a kernel's text (fails loudly when the text has moved). */
const swap = (wgsl: string, from: string, to: string): string => {
  assertEquals(wgsl.split(from).length, 2, `fault injection anchor '${from}'`);
  return wgsl.replace(from, to);
};

/**
 * The five dispatches of one group_norm node with the given kernel texts (④'s harness — the
 * Session cannot run an altered kernel). Bindings and params are the recipe builder's.
 */
const runDirect = async (
  gpu: GpuContext,
  kernels: Kernels,
  layout: GroupNormLayout,
  inputs: Inputs,
): Promise<Float32Array<ArrayBuffer>> => {
  const device = gpu.device;
  const buffers: GPUBuffer[] = [];
  let values: Float32Array<ArrayBuffer> | undefined;
  let validation: GPUError | null = null;
  device.pushErrorScope("validation");
  try {
    const buffer = (
      data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer> | number,
      usage = U.STORAGE | U.COPY_DST | U.COPY_SRC,
    ): GPUBuffer => {
      const value = device.createBuffer({
        size: typeof data === "number" ? data : data.byteLength,
        usage,
      });
      buffers.push(value);
      if (typeof data !== "number") device.queue.writeBuffer(value, 0, data);
      return value;
    };
    const uniform = (data: Uint32Array<ArrayBuffer>) => buffer(data, U.UNIFORM | U.COPY_DST);
    const x = buffer(inputs.x);
    const weight = buffer(inputs.weight);
    const bias = buffer(inputs.bias);
    const partials = buffer(layout.rows * layout.chunks * 4);
    const stats = buffer(layout.rows * 2 * 4);
    const out = buffer(layout.count * 4);
    const staging = buffer(layout.count * 4, U.MAP_READ | U.COPY_DST);
    const pipeline = (code: string) =>
      device.createComputePipelineAsync({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
    const [partial, total, apply] = await Promise.all([
      pipeline(kernels.partial),
      pipeline(kernels.total),
      pipeline(kernels.apply),
    ]);
    const group = (target: GPUComputePipeline, resources: readonly GPUBuffer[]) =>
      device.createBindGroup({
        layout: target.getBindGroupLayout(0),
        entries: resources.map((resource, binding) => ({
          binding,
          resource: { buffer: resource },
        })),
      });
    const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
    const encoder = device.createCommandEncoder();
    const dispatch = (
      target: GPUComputePipeline,
      resources: readonly GPUBuffer[],
      count: number,
    ) => {
      const pass = encoder.beginComputePass();
      pass.setPipeline(target);
      pass.setBindGroup(0, group(target, resources));
      pass.dispatchWorkgroups(Math.min(limit, count));
      pass.end();
    };
    for (const pass of [0, 1] as const) {
      dispatch(
        partial,
        [uniform(groupNormPartialParams(layout, pass)), x, stats, partials],
        layout.rows * layout.chunks,
      );
      dispatch(
        total,
        [uniform(groupNormTotalParams(layout, pass, EPS)), partials, stats],
        layout.rows,
      );
    }
    dispatch(
      apply,
      [uniform(groupNormApplyParams(layout)), x, stats, weight, bias, out],
      Math.ceil(layout.count / GROUP_NORM_WORKGROUP_SIZE),
    );
    encoder.copyBufferToBuffer(out, 0, staging, 0, layout.count * 4);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(MAP_MODE.READ);
    values = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
  } finally {
    await device.queue.onSubmittedWorkDone();
    validation = await device.popErrorScope();
    for (const resource of buffers) resource.destroy();
  }
  if (validation) throw new Error(validation.message);
  if (values === undefined) throw new Error("no readback");
  return values;
};

test.skipIf(!GPU_AVAILABLE)(
  "group_norm ①: the output is within the f64 band on decision and acceptance cases (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [role, cases] of [
        ["decision", DECISION_CASES],
        ["accept", ACCEPT_CASES],
      ] as const) {
        for (const testCase of cases) {
          const layout = rowsOf(testCase.shape, testCase.groups);
          const inputs = inputsOf(testCase);
          const { outputs, census } = await runGroupNorm(gpu, testCase, inputs);
          // The census: the node ran as five dispatches of the three kernels.
          assertEquals(
            countDispatches(census, (key) => key.startsWith("group_norm_")),
            5,
          );
          const e = errorUnits(outputs[0], reference(inputs, layout), layout);
          console.info(`[group_norm ①] ${role} ${testCase.name}: e = ${e.toFixed(3)}`);
          assert(e <= F64_BAND, `${testCase.name}: e = ${e} is outside the f64 band ${F64_BAND}`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm ②: the output agrees with the decomposed layer_norm path within the A/B band (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of [...DECISION_CASES, ...ACCEPT_CASES]) {
        const layout = rowsOf(testCase.shape, testCase.groups);
        const inputs = inputsOf(testCase);
        const fused = await runGroupNorm(gpu, testCase, inputs);
        const decomposed = await runDecomposed(gpu, testCase, inputs);
        assertEquals(
          countDispatches(decomposed.census, (key) => key.startsWith("group_norm_")),
          0,
        );
        const e = abUnits(
          fused.outputs[0],
          decomposed.outputs[0],
          reference(inputs, layout),
          layout,
        );
        console.info(`[group_norm ②] ${testCase.name}: e(A/B) = ${e.toFixed(3)}`);
        assert(e <= AB_BAND, `${testCase.name}: e(A/B) = ${e} is outside the A/B band ${AB_BAND}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm ③: runs and dispatch geometries are bit-identical; zero variance and NaN are exact (real GPU)",
  async () => {
    const testCase = DECISION_CASES[3];
    const inputs = inputsOf(testCase);
    const gpu = await acquireGpu();
    let natural: readonly Float32Array<ArrayBuffer>[];
    try {
      natural = (await runGroupNorm(gpu, testCase, inputs, 2)).outputs;
      assert(bitsEqual(natural[0], natural[1]), "two runs of one Session differ");

      // A zero-variance group of a power-of-two length (2 channels × 16 = 32): the sum and 1/32
      // are exact, so x − mean is exactly 0 and the output is exactly the bias.
      const flat = { shape: [1, 4, 4, 4], groups: 2 };
      const constant: Inputs = {
        x: Float32Array.from({ length: 64 }, (_, i) => (i < 32 ? 3 : -7)),
        weight: Float32Array.from([1.5, -0.5, 2, 0.25]),
        bias: Float32Array.from([0.125, -1, 3, 0.5]),
      };
      const [constantOut] = (await runGroupNorm(gpu, flat, constant)).outputs;
      for (let i = 0; i < 64; i += 1) {
        assertEquals(
          constantOut[i],
          constant.bias[Math.floor(i / 16)],
          `zero variance, element ${i}`,
        );
      }

      // A NaN poisons its own group (both statistics) and no other.
      const poisoned: Inputs = {
        x: Float32Array.from({ length: 64 }, (_, i) => (i === 37 ? Number.NaN : unit(i, 3))),
        weight: constant.weight,
        bias: constant.bias,
      };
      const [poisonedOut] = (await runGroupNorm(gpu, flat, poisoned)).outputs;
      for (let i = 0; i < 64; i += 1) {
        assertEquals(Number.isNaN(poisonedOut[i]), i >= 32, `NaN propagation, element ${i}`);
      }
      // +Inf makes its group's mean Inf and every deviation Inf − Inf = NaN — that group only.
      const infinite: Inputs = {
        ...poisoned,
        x: poisoned.x.map((v, i) => (i === 5 ? Infinity : i === 37 ? 0.5 : v)),
      };
      const [infiniteOut] = (await runGroupNorm(gpu, flat, infinite)).outputs;
      for (let i = 0; i < 64; i += 1) {
        assertEquals(Number.isNaN(infiniteOut[i]), i < 32, `Inf propagation, element ${i}`);
      }
    } finally {
      gpu.destroy();
    }
    // Every dispatch grid-strides: with 3 workgroups per dimension the partial stage (320 chunks),
    // the total stage (32 rows) and the apply stage all wrap, and the bits do not move.
    const narrow = await acquireGpu({ [LIMIT_CAPS]: { maxComputeWorkgroupsPerDimension: 3 } });
    try {
      assertEquals(narrow.limits.maxComputeWorkgroupsPerDimension, 3);
      const [wrapped] = (await runGroupNorm(narrow, testCase, inputs)).outputs;
      assert(bitsEqual(wrapped, natural[0]), "the output depends on the dispatch geometry");
    } finally {
      narrow.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm ④: injected faults land outside the f64 band (real GPU)",
  async () => {
    // The mean-+100 input for the faults of the sums and the affine; the eps faults need a group
    // variance of the order of eps (module doc ④).
    const offsetCase = DECISION_CASES[3];
    const epsCase = EPS_SCALE_CASES[1];
    const faults: readonly (readonly [string, Kernels, Case])[] = [
      [
        "one-pass variance E[x²] − E[x]²",
        {
          ...REAL_KERNELS,
          partial: swap(
            GROUP_NORM_PARTIAL_WGSL,
            "let d = v - mean;\n          acc = acc + d * d;",
            "acc = acc + v * v;",
          ),
          total: swap(
            GROUP_NORM_TOTAL_WGSL,
            "1.0 / sqrt(scratch[0u] * params.inv_length + params.eps)",
            "1.0 / sqrt(max(scratch[0u] * params.inv_length - stats[2u * row] * stats[2u * row], 0.0) + params.eps)",
          ),
        },
        offsetCase,
      ],
      [
        "the last partial chunk of each row dropped",
        {
          ...REAL_KERNELS,
          total: swap(
            GROUP_NORM_TOTAL_WGSL,
            "while (j < params.chunks) {",
            "while (j + 1u < params.chunks) {",
          ),
        },
        offsetCase,
      ],
      [
        "the affine read from the wrong group",
        {
          ...REAL_KERNELS,
          apply: swap(
            GROUP_NORM_APPLY_WGSL,
            "(row % params.groups) * params.per_group",
            "((row + 1u) % params.groups) * params.per_group",
          ),
        },
        offsetCase,
      ],
      [
        "eps outside the square root",
        {
          ...REAL_KERNELS,
          total: swap(
            GROUP_NORM_TOTAL_WGSL,
            "1.0 / sqrt(scratch[0u] * params.inv_length + params.eps)",
            "1.0 / (sqrt(scratch[0u] * params.inv_length) + params.eps)",
          ),
        },
        epsCase,
      ],
      [
        "eps × 10",
        {
          ...REAL_KERNELS,
          total: swap(
            GROUP_NORM_TOTAL_WGSL,
            "1.0 / sqrt(scratch[0u] * params.inv_length + params.eps)",
            "1.0 / sqrt(scratch[0u] * params.inv_length + 10.0 * params.eps)",
          ),
        },
        epsCase,
      ],
    ];
    const gpu = await acquireGpu();
    try {
      for (const testCase of [offsetCase, epsCase]) {
        // The harness itself reproduces the Session (bit for bit — the same kernels, params and
        // dispatch counts), so a fault's distance is the fault's alone.
        const inputs = inputsOf(testCase);
        const layout = groupNormLayout(testCase.shape, testCase.groups);
        const real = await runDirect(gpu, REAL_KERNELS, layout, inputs);
        const [session] = (await runGroupNorm(gpu, testCase, inputs)).outputs;
        assert(bitsEqual(real, session), `${testCase.name}: the direct harness differs`);
      }
      for (const [name, kernels, testCase] of faults) {
        const inputs = inputsOf(testCase);
        const rows = rowsOf(testCase.shape, testCase.groups);
        const layout = groupNormLayout(testCase.shape, testCase.groups);
        const e = errorUnits(
          await runDirect(gpu, kernels, layout, inputs),
          reference(inputs, rows),
          rows,
        );
        console.info(`[group_norm ④] ${name}: e = ${e.toExponential(3)}`);
        assert(
          e > FAULT_MARGIN * F64_BAND,
          `${name}: e = ${e} is not outside ${FAULT_MARGIN} × the f64 band (the check is vacuous)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/** The large group at the MOSS decoder's area cap (module doc). */
const LARGE: Case = {
  name: "[1,192,1696,1648] G32 (16,770,048 per group) mean +100 on odd groups",
  shape: [1, 192, 1696, 1648],
  groups: 32,
  offset: 100,
};

test.skipIf(!GPU_AVAILABLE)(
  "group_norm large group: 16.8M elements per group past the 65,535-workgroup wrap, within the f64 band (real GPU)",
  async (context) => {
    const layout = groupNormLayout(LARGE.shape, LARGE.groups);
    const rows = rowsOf(LARGE.shape, LARGE.groups);
    const bytes = rows.count * 4;
    const gpu = await acquireGpu();
    try {
      const limit = Math.min(gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize);
      // The limits are known only once the device is acquired, so the skip is decided here.
      context.skip(
        bytes > limit,
        `${bytes} B does not fit this device's binding / buffer limit ${limit} B`,
      );
      // The wrap is real: the partial stage needs more workgroups than one dimension holds.
      assert(layout.rows * layout.chunks > gpu.limits.maxComputeWorkgroupsPerDimension);
      const inputs = inputsOf(LARGE);
      const { outputs, census } = await runGroupNorm(gpu, LARGE, inputs);
      assertEquals(
        countDispatches(census, (key) => key.startsWith("group_norm_")),
        5,
      );
      const e = errorUnits(outputs[0], reference(inputs, rows), rows);
      console.info(
        `[group_norm large] ${layout.rows} rows × ${layout.chunks} chunks = ${layout.rows * layout.chunks} partial workgroups: e = ${e.toFixed(3)}`,
      );
      assert(e <= F64_BAND, `large group: e = ${e} is outside the f64 band ${F64_BAND}`);
    } finally {
      gpu.destroy();
    }
  },
  600_000,
);

test.skipIf(!GPU_AVAILABLE)(
  "group_norm: the estimate's workspace equals the Session's plan backing, temporaries included (real GPU)",
  async () => {
    // Nothing fuses in a lone group_norm node, so the estimate's workspace is the slot table's
    // total — the plan backing — exactly; the two in-node temporaries (partial sums and the
    // per-row statistics) are part of both.
    const testCase = DECISION_CASES[5];
    const inputs = inputsOf(testCase);
    const model = openGraph(groupNormGraph(testCase.shape, testCase.groups));
    const gpu = await acquireGpu();
    try {
      const { scenarios } = model.estimate({
        maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
      });
      assertEquals(
        scenarios.map((scenario) => scenario.name),
        ["run"],
      );
      const session = await model.createContainerSession(gpu);
      try {
        // The slot backing is built on the second run of one signature (the first is the arena's).
        for (let step = 0; step < 2; step += 1) {
          await session.run({
            x: f32(testCase.shape, inputs.x),
            w: f32([testCase.shape[1]], inputs.weight),
            b: f32([testCase.shape[1]], inputs.bias),
          });
        }
        assertEquals(session.diagnostics().planBacking.residentBytes, scenarios[0].workspaceBytes);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
