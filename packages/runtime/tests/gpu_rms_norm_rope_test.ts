// The fused rms_norm → pair RoPE kernel (perf-ledger F-3 stage 3 — `RMS_NORM_ROPE_WGSL` in
// src/kernels/rms-norm-rope.ts) against the pair it replaces in the Wan DiT's q / k prologue,
// "rms_norm (the 256-wide workgroup form) → rope:v1:pair:bshd" with the normalised row bound twice
// (as `x` and as `pairs`, the way the two reshape aliases bind it), both from the production
// generators (the same keys and WGSL a Session compiles): the outputs are compared as u32 words.
//
// The fused kernel recomputes each normalised value from `x` instead of reading the stored row, so
// whether every bit of the pair stays in place is a measured fact on a real backend, not a WGSL
// guarantee, and it is visible only here. No adapter = explicit SKIP.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { RMS_NORM_KEY, RMS_NORM_WGSL, rmsNormParams } from "../src/kernels/rms-norm.ts";
import {
  RMS_NORM_ROPE_KEY,
  RMS_NORM_ROPE_WGSL,
  rmsNormRopeParams,
} from "../src/kernels/rms-norm-rope.ts";
import {
  ROPE_PAIR_BSHD_KEY,
  ROPE_WORKGROUP_SIZE,
  ropeParams,
  ropeWgsl,
} from "../src/kernels/rope.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** The Wan DiT's eps (`attn1.norm_q` / `norm_k`). */
const EPS = 1e-6;

/** One case: `x [S, H·D]`, `weight [H·D]`, `cos` / `sin [S, D]` (the tables `[1,S,1,D]`). */
type Inputs = {
  readonly sequence: number;
  readonly heads: number;
  readonly headDim: number;
  readonly x: Float32Array<ArrayBuffer>;
  readonly weight: Float32Array<ArrayBuffer>;
  readonly cos: Float32Array<ArrayBuffer>;
  readonly sin: Float32Array<ArrayBuffer>;
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
 * Runs one case through the pair (the plain rms_norm, then the pair RoPE reading its output as `x`
 * and `pairs`) or the fused kernel (one row per workgroup; `groups` clamps the workgroup count to
 * walk the grid-stride row loop) and returns the RoPE output's words.
 */
const run = async (
  gpu: GpuContext,
  inputs: Inputs,
  form: { readonly kind: "pair" } | { readonly kind: "fused"; readonly groups?: number },
): Promise<Uint32Array<ArrayBuffer>> => {
  const { sequence, heads, headDim } = inputs;
  const dim = heads * headDim;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const upload = (data: Float32Array<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(16, data.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const [x, weight, cos, sin] = [inputs.x, inputs.weight, inputs.cos, inputs.sin].map(upload);
    const out = arena.allocRegion(Math.max(4, sequence * dim * 4));
    const dispatch = async (
      key: string,
      wgsl: string,
      params: Uint32Array<ArrayBuffer>,
      buffers: readonly GPUBuffer[],
      groups: number,
    ): Promise<void> => {
      const { pipeline, layout } = await cache.get(key, wgsl);
      const uniform = arena.allocHostWritten(Math.max(16, params.byteLength), UNIFORM_IN);
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
    if (form.kind === "pair") {
      const normalised = arena.allocRegion(Math.max(4, sequence * dim * 4));
      await dispatch(
        RMS_NORM_KEY,
        RMS_NORM_WGSL,
        rmsNormParams(sequence, dim, EPS),
        [x, weight, normalised],
        gridStrideWorkgroups(sequence, 1, limit),
      );
      await dispatch(
        ROPE_PAIR_BSHD_KEY,
        ropeWgsl("bshd", "pair"),
        ropeParams(sequence * dim, heads, headDim, "bshd"),
        [normalised, normalised, cos, sin, out],
        gridStrideWorkgroups(sequence * dim, ROPE_WORKGROUP_SIZE, limit),
      );
    } else {
      await dispatch(
        RMS_NORM_ROPE_KEY,
        RMS_NORM_ROPE_WGSL,
        rmsNormRopeParams(sequence, dim, EPS, headDim),
        [x, weight, cos, sin, out],
        form.groups ?? gridStrideWorkgroups(sequence, 1, limit),
      );
    }
    await scheduler.flush();
    return await readback(gpu.device, out, sequence * dim);
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

const bits = (word: number): number => new Float32Array(new Uint32Array([word]).buffer)[0];

/** normal()·2 for every input (the adoption measurement's first set). */
const normalInputs = (sequence: number, heads: number, headDim: number, seed: number): Inputs => {
  const dim = heads * headDim;
  const scaled = (count: number, offset: number): Float32Array<ArrayBuffer> =>
    normals(count, seed + offset).map((value) => value * 2);
  return {
    sequence,
    heads,
    headDim,
    x: scaled(sequence * dim, 0),
    weight: scaled(dim, 1),
    cos: scaled(sequence * headDim, 2),
    sin: scaled(sequence * headDim, 3),
  };
};

/** Row indices of {@link specialRows} that an assertion below names. */
const ZERO_ROW = 0;
const NEGATIVE_ZERO_ROW = 1;
const OVERFLOW_ROW = 3;
const NAN_ROW = 5;

/**
 * Rows of special values in `x` (weights and tables normal()·2), each value placed at the start of
 * the row, past the first 256-element chunk and at the end:
 *
 * - an all-zero row and a row of −0 (the sum of squares is 0, so `inv = rsqrt(eps)` and every
 *   normalised value is ±0);
 * - subnormals of both signs among normal values (both kernels flush or keep them alike);
 * - ±2e19 among normal values: `v · v` overflows, the sum is +inf and `inv = 0`, so every
 *   normalised value of the row is ±0;
 * - values near 1e-20 (their squares underflow to 0: `inv = rsqrt(eps)` scales them up);
 * - a NaN (the sum is NaN, so the whole row is NaN); +inf and −inf (the sum is +inf, `inv = 0`, and
 *   `±inf · 0` is NaN at that element only); two different NaN payloads in one row.
 */
const specialRows = (heads: number, headDim: number): Inputs => {
  const dim = heads * headDim;
  const rows: (readonly number[])[] = [
    [0, 0, 0],
    [bits(0x80000000), bits(0x80000000), bits(0x80000000)],
    [bits(0x00000001), bits(0x807fffff), bits(0x00400000)],
    [2e19, -2e19, 1.5],
    [1e-20, -3e-20, 2e-20],
    [Number.NaN, 1.25, -0.5],
    [Number.POSITIVE_INFINITY, 3.5, -2],
    [Number.NEGATIVE_INFINITY, 0.75, 2],
    [bits(0x7fc01234), bits(0xffc00001), 0.5],
  ];
  const x = new Float32Array(rows.length * dim);
  const positions = [0, 257, dim - 1];
  rows.forEach((values, row) => {
    const base = row * dim;
    if (row === NEGATIVE_ZERO_ROW) x.fill(bits(0x80000000), base, base + dim);
    else if (row === 4)
      x.set(
        normals(dim, row).map((value) => value * 1e-20),
        base,
      );
    else if (row !== ZERO_ROW)
      x.set(
        normals(dim, row + dim).map((value) => value * 2),
        base,
      );
    positions.forEach((position, at) => {
      x[base + position] = values[at];
    });
  });
  const tables = normalInputs(rows.length, 1, headDim, 77);
  return {
    sequence: rows.length,
    heads,
    headDim,
    x,
    weight: normals(dim, 5).map((value) => value * 2),
    cos: tables.cos,
    sin: tables.sin,
  };
};

/** The Wan2.2 DiT (24 × 128 = 3,072), the Wan2.1 DiT (12 × 128 = 1,536) and a partial last chunk. */
const SHAPES = [
  [24, 128],
  [12, 128],
  [3, 128],
] as const;

test.skipIf(!GPU_AVAILABLE)(
  "the fused rms_norm → pair RoPE kernel matches rms_norm → rope:v1:pair:bshd word for word on normal()·2 rows at the Wan shapes (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, headDim] of SHAPES) {
        for (const sequence of [1, 7, 300]) {
          const label = `H=${heads} D=${headDim} S=${sequence}`;
          const inputs = normalInputs(sequence, heads, headDim, heads * 1000 + sequence);
          const pair = await run(gpu, inputs, { kind: "pair" });
          const fused = await run(gpu, inputs, { kind: "fused" });
          assertEquals(firstDifference(fused, pair), undefined, label);
          // The output is not degenerate (otherwise every word could agree trivially).
          assertEquals(new Set(pair).size > pair.length / 2, true, `${label}: distinct words`);
        }
      }
      // The grid-stride row loop: two workgroups walk 301 rows (151 and 150).
      const inputs = normalInputs(301, 24, 128, 11);
      assertEquals(
        firstDifference(
          await run(gpu, inputs, { kind: "fused", groups: 2 }),
          await run(gpu, inputs, { kind: "pair" }),
        ),
        undefined,
        "H=24 D=128 S=301 over 2 workgroups",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the fused rms_norm → pair RoPE kernel matches the pair word for word on special values (zero / −0 / subnormal / ±2e19 / underflowing squares / NaN / ±inf rows, special weights and tables), NaN payloads included (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [heads, headDim] of SHAPES) {
        const dim = heads * headDim;
        const label = `special H=${heads} D=${headDim}`;
        const inputs = specialRows(heads, headDim);
        const pair = await run(gpu, inputs, { kind: "pair" });
        assertEquals(
          firstDifference(await run(gpu, inputs, { kind: "fused" }), pair),
          undefined,
          label,
        );
        // The rows hit what they are meant to: the zero, −0 and overflowing rows are all ±0, the NaN
        // row all NaN.
        const value = new Float32Array(pair.buffer);
        const row = (index: number): Float32Array => value.subarray(index * dim, (index + 1) * dim);
        for (const index of [ZERO_ROW, NEGATIVE_ZERO_ROW, OVERFLOW_ROW]) {
          assertEquals(
            row(index).every((element) => element === 0),
            true,
            `${label}: row ${index} is ±0`,
          );
        }
        assertEquals(row(NAN_ROW).every(Number.isNaN), true, `${label}: the NaN row`);

        // Special weights and tables on normal()·2 rows: −0, a subnormal, a huge weight (the product
        // overflows) and a NaN weight (its column and its pair partner turn NaN in every row); −0,
        // NaN and ±inf table entries at three tokens.
        const tables = normalInputs(7, heads, headDim, 3 * dim);
        const weight = Float32Array.from(tables.weight);
        weight.set([bits(0x80000000), 1e-40, 3e38], 5);
        weight[100] = Number.NaN;
        const cos = Float32Array.from(tables.cos);
        const sin = Float32Array.from(tables.sin);
        cos.set([bits(0x80000000), Number.NaN, Number.POSITIVE_INFINITY], headDim + 3);
        sin.set([Number.NEGATIVE_INFINITY, bits(0x80000000), Number.NaN], 4 * headDim + 10);
        const weights = { ...tables, weight, cos, sin };
        assertEquals(
          firstDifference(
            await run(gpu, weights, { kind: "fused" }),
            await run(gpu, weights, { kind: "pair" }),
          ),
          undefined,
          `${label}: special weights and tables`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);
