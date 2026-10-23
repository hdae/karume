// The four-point check (ADR 0110 decision 3) of the online int8 attention kernel (ADR 0132,
// src/kernels/attention-online-i8a8.ts), class Q:
//
//   ① the band against the exact (f64) attention of the dequantised inputs — the error model:
//      |ΔO[d]| ≤ (0.5/127) · Σ_j |ṽ_j[d]| / l (one P̃ lattice step per weight) + f32 noise
//   ② the A/B band against the three-pass a8 path it replaces (the TS references of ①QK / ② / ③PV
//      composed) — both carry one lattice step, so twice the model
//   ③ structural: dp4a and its emulation bit-identical; the subgroup and workgroup-memory
//      reduction forms bit-identical; a run split into row windows bit-identical to one window;
//      two dispatches of the same input bit-identical; P̃ read out through an identity V is within
//      ±1 step of the ③PV reference (never ±2); the TS mirror within one lattice step
//   ④ fault injection: dropping the rescale of O and truncating the tile row max turn ① red
//
// The quantised inputs are built on the host with the TS references (the GPU kernels that do it in
// the Session path have their own tests), so the kernel is the only thing under test here.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Eligible,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
  attentionOnlineI8a8WorkgroupStorageBytes,
} from "../src/kernels/attention-online-i8a8.ts";
import {
  quantizeRowsReference,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
} from "../src/reference/i8a8.ts";
import {
  referenceAttentionOnlineI8a8,
  transposeValues,
} from "../src/reference/attention-online.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const P_LATTICE_MAX = 127;
/** f32 noise allowed on top of the lattice model (accumulation over up to 1,000 keys of O(1) values). */
const F32_NOISE = 2e-5;

type Shape = {
  readonly name: string;
  /** B·H folded; ≥ 2 with per-head statistics that differ is the only detector of the batch base. */
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Shapes: every eligible depth, a single key tile (the P̃ read-out shape), key tiles with a
 * remainder (200 = 3 × 64 + 8), row windows with a remainder (150 = 2 × 64 + 22), the cross
 * attention shape of the Wan DiT (512 keys), and a long key axis.
 */
const SHAPES: readonly Shape[] = [
  { name: "single tile D64", batch: 2, m: 64, n: 64, d: 64 },
  { name: "remainders D128", batch: 3, m: 150, n: 200, d: 128 },
  { name: "cross D128", batch: 2, m: 70, n: 512, d: 128 },
  { name: "long keys D64", batch: 2, m: 100, n: 1000, d: 64 },
];

// ---------------------------------------------------------------------------
// Inputs (deterministic, per-head statistics differ)
// ---------------------------------------------------------------------------

const seeded = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return state / 0x1_0000_0000;
  };
};

const gaussian = (next: () => number): number => {
  const u = next() || 1e-9;
  const v = next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

type Inputs = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
  readonly scale: number;
};

const makeInputs = (shape: Shape, seed: number): Inputs => {
  const { batch, m, n, d } = shape;
  const next = seeded(seed);
  const q = new Float32Array(batch * m * d);
  const k = new Float32Array(batch * n * d);
  const v = new Float32Array(batch * n * d);
  for (let b = 0; b < batch; b += 1) {
    const gain = 0.6 + 0.5 * b;
    for (let i = 0; i < m * d; i += 1) q[b * m * d + i] = gaussian(next) * gain;
    for (let i = 0; i < n * d; i += 1) k[b * n * d + i] = gaussian(next) * (1.2 - 0.3 * b);
    for (let i = 0; i < n * d; i += 1) v[b * n * d + i] = gaussian(next) * (0.8 + 0.4 * b);
  }
  // √(1/√D): the half scale of the op's scale 1/√D; sharp enough that a few keys dominate a row
  return { q, k, v, scale: Math.sqrt(1 / Math.sqrt(d)) * 1.5 };
};

/** Pack flat i8 rows four per u32 (component i in byte i — the order of `pack4xI8`). */
const packWords = (values: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(values.length / 4);
  for (let w = 0; w < words.length; w += 1) {
    words[w] =
      ((values[w * 4] & 0xff) |
        ((values[w * 4 + 1] & 0xff) << 8) |
        ((values[w * 4 + 2] & 0xff) << 16) |
        ((values[w * 4 + 3] & 0xff) << 24)) >>>
      0;
  }
  return words;
};

type Quantised = {
  readonly qq: Uint32Array<ArrayBuffer>;
  readonly qs: Float32Array<ArrayBuffer>;
  readonly kq: Uint32Array<ArrayBuffer>;
  readonly ks: Float32Array<ArrayBuffer>;
  readonly vq: Uint32Array<ArrayBuffer>;
  readonly vs: Float32Array<ArrayBuffer>;
  /** The flat i8 of Vᵀ `[batch, d, n]` and the dequantised values for the references. */
  readonly vtInt: Int8Array<ArrayBuffer>;
  readonly qDeq: Float64Array<ArrayBuffer>;
  readonly kDeq: Float64Array<ArrayBuffer>;
  readonly vtDeq: Float64Array<ArrayBuffer>;
};

const quantise = (shape: Shape, inputs: Inputs): Quantised => {
  const { batch, m, n, d } = shape;
  const query = quantizeRowsReference(inputs.q, batch * m, d);
  const key = quantizeRowsReference(inputs.k, batch * n, d);
  const values = quantizeRowsReference(transposeValues(inputs.v, batch, n, d), batch * d, n);
  const dequant = (
    rows: { q: Int8Array; scale: Float32Array },
    dim: number,
  ): Float64Array<ArrayBuffer> => {
    const out = new Float64Array(rows.q.length);
    for (let i = 0; i < out.length; i += 1) out[i] = rows.q[i] * rows.scale[Math.floor(i / dim)];
    return out;
  };
  return {
    qq: packWords(query.q),
    qs: query.scale,
    kq: packWords(key.q),
    ks: key.scale,
    vq: packWords(values.q),
    vs: values.scale,
    vtInt: values.q,
    qDeq: dequant(query, d),
    kDeq: dequant(key, d),
    vtDeq: dequant(values, n),
  };
};

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

type Exact = {
  readonly o: Float64Array<ArrayBuffer>;
  /** The error-model bound per output element: (0.5/127) · Σ_j |ṽ_j[d]| / l + f32 noise. */
  readonly bound: Float64Array<ArrayBuffer>;
};

/** Exact attention (f64) of the dequantised q / k / v — the oracle of the contract (class Q). */
const exactAttention = (shape: Shape, quant: Quantised, scale: number): Exact => {
  const { batch, m, n, d } = shape;
  const o = new Float64Array(batch * m * d);
  const bound = new Float64Array(batch * m * d);
  const weights = new Float64Array(n);
  const absV = new Float64Array(d);
  for (let b = 0; b < batch; b += 1) {
    for (let col = 0; col < d; col += 1) {
      let sum = 0;
      for (let j = 0; j < n; j += 1) sum += Math.abs(quant.vtDeq[(b * d + col) * n + j]);
      absV[col] = sum;
    }
    for (let row = 0; row < m; row += 1) {
      let max = -Infinity;
      for (let j = 0; j < n; j += 1) {
        let s = 0;
        for (let i = 0; i < d; i += 1) {
          s +=
            quant.qDeq[(b * m + row) * d + i] * scale * (quant.kDeq[(b * n + j) * d + i] * scale);
        }
        weights[j] = s;
        if (s > max) max = s;
      }
      let l = 0;
      for (let j = 0; j < n; j += 1) {
        weights[j] = Math.exp(weights[j] - max);
        l += weights[j];
      }
      for (let col = 0; col < d; col += 1) {
        let acc = 0;
        for (let j = 0; j < n; j += 1) acc += weights[j] * quant.vtDeq[(b * d + col) * n + j];
        o[(b * m + row) * d + col] = acc / l;
        bound[(b * m + row) * d + col] = (0.5 / P_LATTICE_MAX) * (absV[col] / l) + F32_NOISE;
      }
    }
  }
  return { o, bound };
};

/** The three-pass a8 path in its TS references: ①QK i8a8 → ② row statistics (f32) → ③PV i8a8. */
const threePassA8 = (shape: Shape, inputs: Inputs, quant: Quantised): Float32Array<ArrayBuffer> => {
  const { batch, m, n, d } = shape;
  const s = referenceAttentionQkI8a8({
    q: inputs.q,
    k: inputs.k,
    batch,
    m,
    n,
    d,
    scale: inputs.scale,
  });
  const rowMax = new Float32Array(batch * m);
  const rowInv = new Float32Array(batch * m);
  for (let row = 0; row < batch * m; row += 1) {
    let max = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < n; j += 1) if (s[row * n + j] > max) max = s[row * n + j];
    let sum = 0;
    for (let j = 0; j < n; j += 1)
      sum = Math.fround(sum + Math.fround(Math.exp(s[row * n + j] - max)));
    rowMax[row] = max;
    rowInv[row] = Math.fround(1 / sum);
  }
  const qp = referenceAttentionPvQuant(s, rowMax, batch * m, n);
  return referenceAttentionPvI8a8Core({
    qp,
    vq: quant.vtInt,
    rowInv,
    vs: quant.vs,
    batch,
    m,
    n,
    d,
  });
};

// ---------------------------------------------------------------------------
// The raw dispatch
// ---------------------------------------------------------------------------

type RunOptions = {
  readonly dp4a: boolean;
  readonly reduce: AttentionOnlineReduce;
  /** Row windows as `[offset, rows]`; one window over all rows when omitted. */
  readonly windows?: readonly (readonly [number, number])[];
  /** Fault injection: rewrites the WGSL (the key gets the label so the cache does not reuse it). */
  readonly mutate?: { readonly label: string; readonly apply: (wgsl: string) => string };
};

const runOnline = async (
  gpu: GpuContext,
  shape: Shape,
  quant: Quantised,
  scale: number,
  options: RunOptions,
): Promise<Float32Array<ArrayBuffer>> => {
  const { batch, m, n, d } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const host = (source: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    const qq = host(quant.qq);
    const qs = host(quant.qs);
    const kq = host(quant.kq);
    const ks = host(quant.ks);
    const vq = host(quant.vq);
    const vs = host(quant.vs);
    const out = arena.allocRegion(batch * m * d * 4);
    let wgsl = attentionOnlineI8a8Wgsl(options.dp4a, d, options.reduce);
    let key = attentionOnlineI8a8Key(options.dp4a, d, options.reduce);
    if (options.mutate !== undefined) {
      const mutated = options.mutate.apply(wgsl);
      assert(
        mutated !== wgsl,
        `fault injection '${options.mutate.label}' found nothing to rewrite`,
      );
      wgsl = mutated;
      key = `${key}:mutated:${options.mutate.label}`;
    }
    const { pipeline, layout } = await cache.get(key, wgsl);
    const windows = options.windows ?? [[0, m] as const];
    for (const [offset, rows] of windows) {
      const params = arena.allocHostWritten(16, UNIFORM_IN);
      gpu.device.queue.writeBuffer(params, 0, attentionOnlineI8a8Params(m, n, offset, scale));
      scheduler.dispatch(
        pipeline,
        gpu.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: qq } },
            { binding: 2, resource: { buffer: kq } },
            { binding: 3, resource: { buffer: vq } },
            { binding: 4, resource: { buffer: out } },
            { binding: 5, resource: { buffer: qs } },
            { binding: 6, resource: { buffer: ks } },
            { binding: 7, resource: { buffer: vs } },
          ],
        }),
        [Math.ceil(rows / ATTENTION_ONLINE_ROWS), batch, 1],
        key,
      );
    }
    await scheduler.flush();
    const bytes = batch * m * d * 4;
    const staging = gpu.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = gpu.device.createCommandEncoder();
    encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
    gpu.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    staging.destroy();
    return new Float32Array(copy);
  } finally {
    await arena.destroy();
  }
};

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

type BandReport = {
  readonly worstRatio: number;
  readonly worstIndex: number;
  readonly nonFinite: number;
};

/** |actual − expected| ≤ factor · bound, element by element; returns the worst share of the bound used. */
const withinModel = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  bound: ArrayLike<number>,
  factor: number,
): BandReport => {
  let worstRatio = 0;
  let worstIndex = -1;
  let nonFinite = 0;
  for (let i = 0; i < actual.length; i += 1) {
    if (!Number.isFinite(actual[i])) {
      nonFinite += 1;
      continue;
    }
    const ratio = Math.abs(actual[i] - expected[i]) / (factor * bound[i]);
    if (ratio > worstRatio) {
      worstRatio = ratio;
      worstIndex = i;
    }
  }
  return { worstRatio, worstIndex, nonFinite };
};

const assertBand = (report: BandReport, where: string): void => {
  assertEquals(report.nonFinite, 0, `${where}: non-finite outputs`);
  assert(
    report.worstRatio <= 1,
    `${where}: the error exceeds the model (worst ${report.worstRatio.toFixed(3)} × bound at ${report.worstIndex})`,
  );
};

const assertBitIdentical = (a: Float32Array, b: Float32Array, where: string): void => {
  const ua = new Uint32Array(a.buffer, a.byteOffset, a.length);
  const ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
  assertEquals(ua.length, ub.length, `${where}: length`);
  for (let i = 0; i < ua.length; i += 1) {
    if (ua[i] !== ub[i]) throw new Error(`${where}: bits differ at ${i} (${a[i]} vs ${b[i]})`);
  }
};

const subgroupsSupported = async (): Promise<boolean> => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return false;
  return (
    adapter.features.has("subgroups") &&
    adapter.features.has("subgroup-size-control") &&
    new Set(navigator.gpu.wgslLanguageFeatures).has("subgroup_id")
  );
};

test("eligibility and workgroup storage of the online kernel", () => {
  assert(attentionOnlineI8a8Eligible(64, 64));
  assert(attentionOnlineI8a8Eligible(128, 512));
  assert(!attentionOnlineI8a8Eligible(256, 1000), "D above the maximum (the tiles exceed 48 KiB)");
  assert(!attentionOnlineI8a8Eligible(96, 64), "D % 64 != 0");
  assert(!attentionOnlineI8a8Eligible(128, 66), "N % 4 != 0");
  // D = 128: q 64 × 33 + k 64 × 33 + Vᵀ 16 × 128 + P̃ 64 × 17 words (+ the reduce scratch)
  assertEquals(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "sg32"),
    (64 * 33 + 64 * 33 + 16 * 128 + 64 * 17) * 4,
  );
  assertEquals(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "wg"),
    (64 * 33 + 64 * 33 + 16 * 128 + 64 * 17 + 64 * 16) * 4,
  );
  assert(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "sg32") < 32768,
    "the subgroup form fits 32 KiB",
  );
});

describe.skipIf(!GPU_AVAILABLE)("online int8 attention (ADR 0132) — the four-point check", () => {
  for (const shape of SHAPES) {
    it(`${shape.name}: ① error model, ② A/B vs three-pass a8, ③ structure (wg)`, async () => {
      const gpu = await acquireGpu();
      try {
        const inputs = makeInputs(shape, 7 + shape.d);
        const quant = quantise(shape, inputs);
        const exact = exactAttention(shape, quant, inputs.scale);
        const dp4a = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "wg" });
        const model = withinModel(dp4a, exact.o, exact.bound, 1);
        console.log(
          `[attention_online] ${shape.name}: ${(100 * model.worstRatio).toFixed(1)}% of the error model used`,
        );
        assertBand(model, `${shape.name} ① vs exact`);
        // ② both paths carry one lattice step each
        const threePass = threePassA8(shape, inputs, quant);
        assertBand(
          withinModel(dp4a, threePass, exact.bound, 2),
          `${shape.name} ② vs three-pass a8`,
        );
        // ③ the TS mirror (ULP-level differences of exp / division move P̃ by at most one step)
        const mirror = referenceAttentionOnlineI8a8({
          ...inputs,
          batch: shape.batch,
          m: shape.m,
          n: shape.n,
          d: shape.d,
        });
        assertBand(withinModel(dp4a, mirror, exact.bound, 2), `${shape.name} ③ vs the TS mirror`);
        // ③ emulated dot product, a second dispatch, row windows: bit-identical
        const emulated = await runOnline(gpu, shape, quant, inputs.scale, {
          dp4a: false,
          reduce: "wg",
        });
        assertBitIdentical(emulated, dp4a, `${shape.name} ③ dp4a vs emulation`);
        const again = await runOnline(gpu, shape, quant, inputs.scale, {
          dp4a: true,
          reduce: "wg",
        });
        assertBitIdentical(again, dp4a, `${shape.name} ③ determinism`);
        if (shape.m > ATTENTION_ONLINE_ROWS) {
          const split = await runOnline(gpu, shape, quant, inputs.scale, {
            dp4a: true,
            reduce: "wg",
            windows: [
              [0, ATTENTION_ONLINE_ROWS],
              [ATTENTION_ONLINE_ROWS, shape.m - ATTENTION_ONLINE_ROWS],
            ],
          });
          assertBitIdentical(split, dp4a, `${shape.name} ③ row windows`);
        }
      } finally {
        gpu.destroy();
      }
    });
  }

  it("③ the subgroup form is bit-identical to the workgroup-memory form (skips without subgroups)", async (context) => {
    if (!(await subgroupsSupported())) context.skip();
    const gpu = await acquireGpu({ subgroups: true });
    try {
      for (const shape of SHAPES) {
        const inputs = makeInputs(shape, 7 + shape.d);
        const quant = quantise(shape, inputs);
        const wg = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "wg" });
        const sg = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "sg32" });
        assertBitIdentical(sg, wg, `${shape.name} ③ sg32 vs wg`);
      }
    } finally {
      gpu.destroy();
    }
  });

  it("③ P̃ read out through an identity V is within ±1 step of the ③PV reference", async () => {
    // One key tile and D = N: the running max is the row max, so P̃ equals ③PV's lattice up to
    // the ULP differences of exp. v = I per head makes vq = 127·I and vs = 1/127, so
    // O[j] = P̃[j] · (1/127) / l and P̃ = round(O · l · 127).
    const shape = SHAPES[0];
    const { batch, m, n, d } = shape;
    const inputs = makeInputs(shape, 11);
    const identity = new Float32Array(batch * n * d);
    for (let b = 0; b < batch; b += 1)
      for (let i = 0; i < n; i += 1) identity[(b * n + i) * d + i] = 1;
    const quant = quantise(shape, { ...inputs, v: identity });
    const s = referenceAttentionQkI8a8({
      q: inputs.q,
      k: inputs.k,
      batch,
      m,
      n,
      d,
      scale: inputs.scale,
    });
    const rowMax = new Float32Array(batch * m);
    const rowSum = new Float64Array(batch * m);
    for (let row = 0; row < batch * m; row += 1) {
      let max = Number.NEGATIVE_INFINITY;
      for (let j = 0; j < n; j += 1) if (s[row * n + j] > max) max = s[row * n + j];
      rowMax[row] = max;
      let sum = 0;
      for (let j = 0; j < n; j += 1) sum += Math.exp(s[row * n + j] - max);
      rowSum[row] = sum;
    }
    const expected = referenceAttentionPvQuant(s, rowMax, batch * m, n);
    const gpu = await acquireGpu();
    try {
      const o = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "wg" });
      let offByOne = 0;
      for (let row = 0; row < batch * m; row += 1) {
        for (let j = 0; j < n; j += 1) {
          const observed = Math.round(o[row * d + j] * rowSum[row] * P_LATTICE_MAX);
          const delta = Math.abs(observed - expected[row * n + j]);
          assert(
            delta <= 1,
            `P̃[${row},${j}]: observed ${observed}, reference ${expected[row * n + j]}`,
          );
          if (delta === 1) offByOne += 1;
        }
      }
      console.log(
        `[attention_online] P̃ read-out: ${offByOne} of ${batch * m * n} elements off by one step`,
      );
    } finally {
      gpu.destroy();
    }
  });

  it("④ fault injection: a missing rescale and a truncated row max leave the model", async () => {
    const shape = SHAPES[1];
    const inputs = makeInputs(shape, 7 + shape.d);
    const quant = quantise(shape, inputs);
    const exact = exactAttention(shape, quant, inputs.scale);
    const gpu = await acquireGpu();
    try {
      const faults = [
        {
          label: "no-rescale",
          apply: (wgsl: string) =>
            wgsl.replace(/o(\d)_(\d) = o\1_\2 \* alpha\1;/g, "o$1_$2 = o$1_$2;"),
        },
        {
          label: "half-row-max",
          apply: (wgsl: string) =>
            wgsl.replace(/i < 16u; i = i \+ 1u\) \{ mx = max/g, "i < 8u; i = i + 1u) { mx = max"),
        },
      ];
      for (const fault of faults) {
        const o = await runOnline(gpu, shape, quant, inputs.scale, {
          dp4a: true,
          reduce: "wg",
          mutate: fault,
        });
        const report = withinModel(o, exact.o, exact.bound, 1);
        assert(
          report.nonFinite > 0 || report.worstRatio > 1,
          `fault '${fault.label}' stayed inside the model (worst ${report.worstRatio.toFixed(3)}) — the check is vacuous`,
        );
      }
    } finally {
      gpu.destroy();
    }
  });
});
