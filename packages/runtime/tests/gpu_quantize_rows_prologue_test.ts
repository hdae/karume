// The per-token quantize with an elementwise prologue (perf-ledger F-4 — `quantizeRowsPrologueWgsl`
// in src/kernels/quantize-rows.ts) against the pair it replaces in an a8 linear's step ①,
// "elementwise gelu_tanh → quantize_rows", both from the production generators (the same keys and
// WGSL a Session compiles): `xq` and `xs` are compared as u32 words.
//
// The fused kernel keeps `y = gelu_tanh(x)` in registers and quantizes it without a rounding barrier,
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

/**
 * Runs `x [1, rows, dim]` through either the pair (the elementwise gelu_tanh the DiT's FFN runs at
 * rank 3, then the quantize_rows geometry the i8a8 linear picks for `dim`) or the prologue form
 * (one row per workgroup; `groups` clamps the workgroup count to walk the grid-stride row loop).
 */
const quantize = async (
  gpu: GpuContext,
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
      const spec: ElementwiseSpec = { op: "gelu_tanh", rank: 3, dtype: "f32" };
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
        quantizeRowsPrologueKey("gelu_tanh", dim),
        quantizeRowsPrologueWgsl("gelu_tanh", dim),
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
 * NVIDIA, 2026-10-25): gelu_tanh's arithmetic turns every input NaN payload into the backend's
 * canonical NaN (0x7fffffff), so the `nan_max` fold — whose element-to-lane assignment the fused
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
 * up-projection input (3072) and down-projection input (14,336), Wan2.1's (8960), and the bound. */
const DIMS = [4, 1028, 3072, 8960, 14336, 16384] as const;
const ROWS = [1, 7, 300] as const;

test.skipIf(!GPU_AVAILABLE)(
  "the gelu_tanh prologue quantize matches elementwise gelu_tanh → quantize_rows word for word on normal()·2 rows (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const dim of DIMS) {
        for (const rows of ROWS) {
          const x = normals(rows * dim, dim * 1000 + rows).map((value) => value * 2);
          const label = `K=${dim} rows=${rows}`;
          const pair = await quantize(gpu, x, rows, dim, { kind: "pair" });
          const fused = await quantize(gpu, x, rows, dim, { kind: "prologue" });
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
        await quantize(gpu, x, rows, dim, { kind: "prologue", groups: 2 }),
        await quantize(gpu, x, rows, dim, { kind: "pair" }),
        "K=14336 rows=300 over 2 workgroups",
      );
    } finally {
      gpu.destroy();
    }
  },
);

const bits = (word: number): number => new Float32Array(new Uint32Array([word]).buffer)[0];

/**
 * Rows of special values, each placed at the start of the row, at a lane-slot boundary (quad 256 =
 * element 1024, slot 1 of lane 0) when the row has one, and at the end:
 *
 * - an all-zero row (s = f32 tiny, every code 0) and a row of −0;
 * - subnormals of both signs among normal values;
 * - ±2e19, where `x³` overflows inside gelu_tanh (the stable tanh must still give 1 / −1);
 * - rounding ties of `y / s`: for x ≥ 10 gelu_tanh(x) = x exactly (tanh saturates to 1), and with
 *   amax 127 the scale is exactly 1, with amax 254 exactly 2, so 63.5 / 10.5 / 127 / 21 land on
 *   half-integers (the lattice rounds them to even);
 * - NaN, +inf (y = inf, s = inf), −inf (gelu_tanh(−inf) = −inf · 0 = NaN), and two different NaN
 *   payloads in one row.
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
  ];
  const x = new Float32Array(rows.length * dim);
  const positions = [0, ...(dim > 1024 ? [1024] : []), dim - 1];
  rows.forEach((values, row) => {
    const base = row * dim;
    // A background of small finite values (zero for the all-zero / −0 rows).
    if (row >= 2) {
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
  "the gelu_tanh prologue quantize matches the pair on special values (zero / −0 / subnormal / ±2e19 / ties / NaN / ±inf), NaN scales included (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const dim of DIMS) {
        const { x, rows } = specialRows(dim);
        const label = `special K=${dim}`;
        const pair = await quantize(gpu, x, rows, dim, { kind: "pair" });
        const fused = await quantize(gpu, x, rows, dim, { kind: "prologue" });
        assertSameWords(fused, pair, label);
        // The rows hit what they are meant to: the zero rows have scale f32 tiny, the tie rows have
        // scale exactly 1 and 2, and the NaN / ±inf rows have a non-finite scale.
        const scale = new Float32Array(pair.xs.buffer);
        assertEquals(scale[0], 1.1754943508222875e-38, `${label}: zero row scale`);
        assertEquals(scale[1], 1.1754943508222875e-38, `${label}: −0 row scale`);
        assertEquals(scale[4], 1, `${label}: amax 127 row scale`);
        assertEquals(scale[5], 2, `${label}: amax 254 row scale`);
        for (const row of [6, 8, 9]) {
          assertEquals(Number.isNaN(scale[row]), true, `${label}: NaN scale of row ${row}`);
        }
        assertEquals(scale[7], Number.POSITIVE_INFINITY, `${label}: +inf row scale`);
      }
    } finally {
      gpu.destroy();
    }
  },
);
