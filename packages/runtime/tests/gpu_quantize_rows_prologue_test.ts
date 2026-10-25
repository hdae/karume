// The per-token quantize with an elementwise prologue (perf-ledger F-4 — `quantizeRowsPrologueWgsl`
// in src/kernels/quantize-rows.ts) against the pair it replaces in an a8 linear's step ①,
// "elementwise op → quantize_rows", both from the production generators (the same keys and WGSL a
// Session compiles), for each op the fusion pass accepts (`gelu_tanh` and the erf `gelu`): `xq`
// and `xs` are compared as u32 words.
//
// The fused kernel keeps `y = op(x)` in registers and quantizes it without a rounding barrier,
// with a different element-to-lane assignment for the row maximum. Whether that leaves every bit of
// the pair in place is a measured fact on a real backend, not a WGSL guarantee, so it is visible
// only here. No adapter = explicit SKIP.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
  type PrologueUnaryOp,
} from "../src/codegen/elementwise.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsPrologueKey,
  quantizeRowsPrologueWgsl,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** The quantized activations (four i8 per u32 word) and the per-token scales, as raw words. */
type Quantized = {
  readonly xq: Uint32Array<ArrayBuffer>;
  readonly xs: Uint32Array<ArrayBuffer>;
};

const readback = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  words: number,
): Promise<Uint32Array<ArrayBuffer>> => {
  const size = Math.max(4, words * 4);
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return new Uint32Array(copy, 0, words);
};

/** The ops the fusion pass folds into the linear's step ① (`LINEAR_PROLOGUE_OPS`). */
const OPS = ["gelu_tanh", "gelu"] as const satisfies readonly PrologueUnaryOp[];

/**
 * Runs `x [1, rows, dim]` through either the pair (the elementwise `op` the DiT's FFN runs at rank
 * 3, then the quantize_rows geometry the i8a8 linear picks for `dim`) or the prologue form (one row
 * per workgroup; `groups` clamps the workgroup count to walk the grid-stride row loop).
 */
const quantize = async (
  gpu: GpuContext,
  op: PrologueUnaryOp,
  x: Float32Array<ArrayBuffer>,
  rows: number,
  dim: number,
  form: { readonly kind: "pair" } | { readonly kind: "prologue"; readonly groups?: number },
): Promise<Quantized> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const src = arena.allocHostWritten(Math.max(16, x.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(src, 0, x);
    const xq = arena.allocRegion(Math.max(4, rows * dim));
    const xs = arena.allocRegion(Math.max(4, rows * 4));
    const run = async (
      key: string,
      wgsl: string,
      params: Uint32Array<ArrayBuffer>,
      usage: number,
      buffers: readonly GPUBuffer[],
      groups: number,
    ): Promise<void> => {
      const { pipeline, layout } = await cache.get(key, wgsl);
      const uniform = arena.allocHostWritten(Math.max(16, params.byteLength), usage);
      gpu.device.queue.writeBuffer(uniform, 0, params);
      const bindGroup = gpu.device.createBindGroup({
        layout,
        entries: [uniform, ...buffers].map((buffer, binding) => ({
          binding,
          resource: { buffer },
        })),
      });
      scheduler.dispatch(pipeline, bindGroup, [groups, 1, 1], key);
    };
    const quantizeParams = quantizeRowsParams(rows, dim);
    if (form.kind === "pair") {
      const spec: ElementwiseSpec = { op, rank: 3, dtype: "f32" };
      const shape = [1, rows, dim];
      const y = arena.allocRegion(Math.max(4, x.byteLength));
      await run(
        elementwiseKey(spec),
        elementwiseWgsl(spec),
        elementwiseParams(spec, shape, [shape], []),
        STORAGE_IN,
        [src, y],
        gridStrideWorkgroups(rows * dim, ELEMENTWISE_WORKGROUP_SIZE, limit),
      );
      const geometry = quantizeRowsGeometry(dim);
      await run(
        quantizeRowsKey(geometry),
        quantizeRowsWgsl(geometry),
        quantizeParams,
        UNIFORM_IN,
        [y, xq, xs],
        gridStrideWorkgroups(rows, geometry.rowsPerGroup, limit),
      );
    } else {
      await run(
        quantizeRowsPrologueKey(op, dim),
        quantizeRowsPrologueWgsl(op, dim),
        quantizeParams,
        UNIFORM_IN,
        [src, xq, xs],
        form.groups ?? gridStrideWorkgroups(rows, 1, limit),
      );
    }
    await scheduler.flush();
    return {
      xq: await readback(gpu.device, xq, (rows * dim) / 4),
      xs: await readback(gpu.device, xs, rows),
    };
  } finally {
    await arena.destroy();
  }
};

/** The first differing word, or `undefined` (a full assertEquals on 1M words is unreadable). */
const firstDifference = (actual: Uint32Array, expected: Uint32Array): string | undefined => {
  if (actual.length !== expected.length) return `length ${actual.length} vs ${expected.length}`;
  for (let i = 0; i < actual.length; i += 1) {
    if (actual[i] !== expected[i]) {
      return `word ${i}: 0x${actual[i].toString(16)} vs 0x${expected[i].toString(16)}`;
    }
  }
  return undefined;
};

/**
 * Every word of `xq` and `xs` matches exactly, NaN scales included.
 *
 * NOTE: the NaN scales are compared exactly because they matched exactly where measured (Vulkan /
 * NVIDIA, 2026-10-25): the arithmetic of both ops (`0.5 · x · (…)` in each) turns every input NaN
 * payload into the backend's canonical NaN (0x7fffffff), so the `nan_max` fold — whose element-to-lane assignment the fused
 * kernel changes — has only one payload to keep. The contract of the fused kernels is NaN by
 * classification (`adaln-norm.ts`); on a backend that keeps payloads through the arithmetic, compare
 * the NaN scales by classification (never the codes).
 */
const assertSameWords = (actual: Quantized, expected: Quantized, label: string): void => {
  assertEquals(firstDifference(actual.xq, expected.xq), undefined, `${label}: xq`);
  assertEquals(firstDifference(actual.xs, expected.xs), undefined, `${label}: xs`);
};

/** A seeded normal(0, 1) stream (Box–Muller over mulberry32) — reproducible on every run. */
const normals = (count: number, seed: number): Float32Array<ArrayBuffer> => {
  let state = seed >>> 0;
  const uniform = (): number => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - uniform()));
    const angle = 2 * Math.PI * uniform();
    out[i] = radius * Math.cos(angle);
    if (i + 1 < count) out[i + 1] = radius * Math.sin(angle);
  }
  return out;
};

/** The row lengths: K 4 (one quad), a remainder of one quad past four lanes' slots, Wan2.2's
 * up-projection input (3072) and down-projection input (14,336), DeBERTa's (4096), anima's (8192),
 * Wan2.1's (8960), and the bound. */
const DIMS = [4, 1028, 3072, 4096, 8192, 8960, 14336, 16384] as const;
const ROWS = [1, 7, 300] as const;

test.skipIf(!GPU_AVAILABLE)(
  "the gelu_tanh / erf gelu prologue quantize matches elementwise op → quantize_rows word for word on normal()·2 rows (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of OPS) {
        for (const dim of DIMS) {
          for (const rows of ROWS) {
            const x = normals(rows * dim, dim * 1000 + rows).map((value) => value * 2);
            const label = `${op} K=${dim} rows=${rows}`;
            const pair = await quantize(gpu, op, x, rows, dim, { kind: "pair" });
            const fused = await quantize(gpu, op, x, rows, dim, { kind: "prologue" });
            assertSameWords(fused, pair, label);
            // The rows are not degenerate (otherwise every word could agree trivially).
            assertEquals(new Set(pair.xs).size, rows, `${label}: one scale per row`);
            assertEquals(
              pair.xq.some((word) => word !== 0),
              true,
              `${label}: non-zero codes`,
            );
          }
        }
        // The grid-stride row loop: two workgroups walk 300 rows (150 each).
        const dim = 14336;
        const rows = 300;
        const x = normals(rows * dim, 7).map((value) => value * 2);
        assertSameWords(
          await quantize(gpu, op, x, rows, dim, { kind: "prologue", groups: 2 }),
          await quantize(gpu, op, x, rows, dim, { kind: "pair" }),
          `${op} K=14336 rows=300 over 2 workgroups`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

const bits = (word: number): number => new Float32Array(new Uint32Array([word]).buffer)[0];

/** Row indices of {@link specialRows} that an assertion below names. */
const ERF_UNDERFLOW_ROW = 11;

/**
 * Rows of special values, each placed at the start of the row, at a lane-slot boundary (quad 256 =
 * element 1024, slot 1 of lane 0) when the row has one, and at the end. What each value does differs
 * per op — `gelu_tanh(x) = 0.5·x·(1 + tanh_stable(√(2/π)·(x + 0.044715·x³)))`, the erf
 * `gelu(x) = 0.5·x·(1 + erf_approx(x·(1/√2)))` with `erf_approx = s·(1 − p(t)·t·exp(−a²))`,
 * `a = |x|/√2`, `t = 1/(1 + 0.3275911·a)` (codegen/elementwise.ts):
 *
 * - an all-zero row and a row of −0: both ops give ±0 (`0.5·(±0)·(…)`; erf_approx(±0) is
 *   `sign(±0)·(…) = 0`), so amax is 0 and s = f32 tiny, every code 0;
 * - subnormals of both signs among normal values (the inner products may flush; the pair and the
 *   fused form must flush alike);
 * - ±2e19, which overflows inside both ops: gelu_tanh's `x³` is inf and the stable tanh clamps it to
 *   ±1; the erf form's `a²` is inf, so `exp(−a²) = 0` and erf_approx = ±1 exactly. Either way
 *   op(2e19) = 2e19 and op(−2e19) = −0, so the row's scale is finite;
 * - rounding ties of `y / s`: both ops are the identity at x ≥ 10 — gelu_tanh because tanh saturates
 *   to 1, the erf form because `p(t)·t·exp(−a²) < 2^−25` there (already from x ≈ 6), so
 *   `1 − …` rounds to 1 and `0.5·x·2 = x`. With amax 127 the scale is exactly 1, with amax 254
 *   exactly 2, so 63.5 / 10.5 / 127 / 21 land on half-integers (the lattice rounds them to even);
 * - NaN (both ops keep the factor `x` outside, so `0.5·NaN·(…)` is NaN); +inf (gelu_tanh: tanh 1;
 *   erf: `t = 0`, so erf_approx = 1 — y = inf, s = inf in both); −inf (gelu_tanh: `−inf·0`; erf:
 *   erf_approx = −1, `1 + erf = 0`, `−inf·0` — NaN in both); two different NaN payloads in one row;
 * - the erf form's negative tail, every element in −5.5..−2.5: `1 + erf_approx` cancels from about
 *   `2^−6` down to a few ulps of 1 (`2^−24`), the region where the erf form is least accurate
 *   (gelu_tanh's `1 + tanh` cancels the same way); a row of non-trivial small codes;
 * - every element in −15..−6: there the erf form's `p(t)·t·exp(−a²)` is at most 2e-9 (< 2^−25), so
 *   erf_approx = −1 exactly and y = −0 for the whole row (s = f32 tiny — asserted for the erf form
 *   only: gelu_tanh's value there rests on the backend's `tanh` reaching −1, which WGSL does not
 *   fix).
 */
const specialRows = (
  dim: number,
): { readonly x: Float32Array<ArrayBuffer>; readonly rows: number } => {
  const rows: (readonly number[])[] = [
    [0, 0, 0],
    [bits(0x80000000), bits(0x80000000), bits(0x80000000)],
    [bits(0x00000001), bits(0x807fffff), bits(0x00400000)],
    [2e19, -2e19, 1.5],
    [127, 63.5, 10.5],
    [254, 127, 21],
    [Number.NaN, 1.25, -0.5],
    [Number.POSITIVE_INFINITY, 3.5, -2],
    [Number.NEGATIVE_INFINITY, 0.75, 2],
    [bits(0x7fc01234), bits(0xffc00001), 0.5],
    [-2.5, -4, -5.5],
    [-6, -10, -15],
  ];
  /** The rows that sweep a range instead of sitting on a normal() background. */
  const sweeps: ReadonlyMap<number, readonly [number, number]> = new Map([
    [10, [-5.5, -2.5]],
    [ERF_UNDERFLOW_ROW, [-15, -6]],
  ]);
  const x = new Float32Array(rows.length * dim);
  const positions = [0, ...(dim > 1024 ? [1024] : []), dim - 1];
  rows.forEach((values, row) => {
    const base = row * dim;
    const sweep = sweeps.get(row);
    if (sweep !== undefined) {
      // Spread over the range in a scrambled order, so neighbouring lanes see different values.
      const [lo, hi] = sweep;
      for (let i = 0; i < dim; i += 1) x[base + i] = lo + ((hi - lo) * ((i * 37) % dim)) / dim;
    } else if (row >= 2) {
      // A background of small finite values (zero for the all-zero / −0 rows).
      const background = normals(dim, row + dim).map((value) => value * 0.25);
      x.set(background, base);
    } else if (row === 1) {
      x.fill(bits(0x80000000), base, base + dim);
    }
    positions.forEach((position, at) => {
      x[base + position] = values[at % values.length];
    });
    // The tie rows put every tie value on each position, so ties sit in different lane slots.
    if (row === 4 || row === 5) {
      for (let i = 0; i < dim; i += 97) x[base + i] = values[(i / 97) % values.length];
    }
  });
  return { x, rows: rows.length };
};

test.skipIf(!GPU_AVAILABLE)(
  "the gelu_tanh / erf gelu prologue quantize matches the pair on special values (zero / −0 / subnormal / ±2e19 / ties / NaN / ±inf / erf's negative tail and underflow), NaN scales included (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const op of OPS) {
        for (const dim of DIMS) {
          const { x, rows } = specialRows(dim);
          const label = `${op} special K=${dim}`;
          const pair = await quantize(gpu, op, x, rows, dim, { kind: "pair" });
          const fused = await quantize(gpu, op, x, rows, dim, { kind: "prologue" });
          assertSameWords(fused, pair, label);
          // The rows hit what they are meant to: the zero rows have scale f32 tiny, the ±2e19 row a
          // finite scale, the tie rows scale exactly 1 and 2, and the NaN / ±inf rows a non-finite
          // scale.
          const scale = new Float32Array(pair.xs.buffer);
          const tiny = 1.1754943508222875e-38;
          assertEquals(scale[0], tiny, `${label}: zero row scale`);
          assertEquals(scale[1], tiny, `${label}: −0 row scale`);
          assertEquals(
            Number.isFinite(scale[3]) && scale[3] > 1e17,
            true,
            `${label}: ±2e19 row scale ${scale[3]}`,
          );
          assertEquals(scale[4], 1, `${label}: amax 127 row scale`);
          assertEquals(scale[5], 2, `${label}: amax 254 row scale`);
          for (const row of [6, 8, 9]) {
            assertEquals(Number.isNaN(scale[row]), true, `${label}: NaN scale of row ${row}`);
          }
          assertEquals(scale[7], Number.POSITIVE_INFINITY, `${label}: +inf row scale`);
          if (op === "gelu") {
            assertEquals(scale[ERF_UNDERFLOW_ROW], tiny, `${label}: the underflow row is all −0`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
