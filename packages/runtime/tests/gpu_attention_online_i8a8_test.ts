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
// The key valid length (ADR 0132 addendum 2026-10-27): shapes with N % 4 = 1, 2, 3 run the
// `padded` key axis (Vᵀ rows of ceil(N/4) words). ① compares them with the exact attention over
// exactly N keys — a reference without any padding, so it is independent of how the padding is
// made — and ③ with the TS mirror (which walks any N with −3.0e38). They run on the dense inputs
// and on two spike-V inputs ({@link spikeInputs}), whose tight band catches a Vᵀ word read from
// the wrong place at every N (the dense band grows with N). On top: the padding bytes filled with
// 0 and with 127 give Uint32-identical outputs (the primary detector of a mask bound past N), and
// three faults — the mask bound at ceil(N/4)·4, the whole form's row length floor(N/4), the last
// Vᵀ word of each row dropped — turn ① red on the spike-V inputs.
//
// The quantised inputs are built on the host with the TS references (the GPU kernels that do it in
// the Session path have their own tests), so the kernel is the only thing under test here.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertThrows } from "@std/assert";
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
  attentionOnlineKeyWords,
  attentionOnlineValueRowWords,
} from "../src/kernels/attention-online-i8a8.ts";
import { quantizeRowsGeometry, quantizeRowsKey } from "../src/kernels/quantize-rows.ts";
import {
  attentionI8a8TempBytes,
  attentionValueQuantization,
} from "../src/runtime/recipe-builders/attention.ts";
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
/**
 * f32 noise allowed on top of the lattice model (accumulation over up to 4,106 keys of O(1)
 * values). Measured sufficient, not proven: the sharp spike inputs leave under 3% of the bound.
 */
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

/**
 * Key lengths that are not multiples of 4 (the `padded` key axis): N % 4 = 1 in a two-tile
 * D = 64 shape, N % 4 = 2 at the MOSS-Image 1024² negative branch's S₋ = 4,096 + 10, N % 4 = 3
 * with row and key tile remainders.
 */
const PADDED_SHAPES: readonly Shape[] = [
  { name: "N%4=1 D64", batch: 2, m: 70, n: 65, d: 64 },
  { name: "N%4=2 D128 (S₋ 4,106)", batch: 2, m: 64, n: 4106, d: 128 },
  { name: "N%4=3 D128", batch: 3, m: 150, n: 203, d: 128 },
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

/**
 * Spike-V inputs (the padded shapes): V holds one key per column at ±(0.5 + b) and 0 elsewhere,
 * so the error model's Σ_j |ṽ_j[d]| is one key's and its band is tight — with the dense V of
 * {@link makeInputs} the band grows with Σ_j |ṽ_j| over all N keys: at N = 4,106 the whole form's
 * row length floor(N/4) left it at 1.33× and a dropped last Vᵀ word at 0.15× (2026-10-27), against
 * 249.6× and 86.3× on the sharp spike. Columns
 * 0–3 put their spike on the last valid keys N − 1 … N − 4 (the padded last word and the word
 * before it); the other columns spread theirs over the axis. `sharp` keeps the dense inputs' q / k
 * and scale (the keys' weights differ, so a key read from the wrong place changes O); `flat` divides
 * the scale by 10, so every key keeps a P̃ of many lattice steps and a key let in past N weighs in
 * (a sharp row max would round a padding key's P̃, exp(0 − m), to 0).
 */
const spikeInputs = (shape: Shape, softmax: "sharp" | "flat"): Inputs => {
  const { batch, n, d } = shape;
  const dense = makeInputs(shape, 7 + d);
  const v = new Float32Array(batch * n * d);
  for (let b = 0; b < batch; b += 1) {
    for (let col = 0; col < d; col += 1) {
      const key = col < 4 ? Math.max(0, n - 1 - col) : (col * 7919 + b * 31) % n;
      v[(b * n + key) * d + col] = (col % 2 === 0 ? 1 : -1) * (0.5 + b);
    }
  }
  return { ...dense, v, scale: softmax === "flat" ? dense.scale * 0.1 : dense.scale };
};

type FourPointCase = {
  readonly shape: Shape;
  /** Appended to the shape's name in the test name and the failure messages. */
  readonly label: string;
  readonly make: () => Inputs;
};

/** Every shape on the dense inputs; the padded shapes also on both spike-V inputs. */
const FOUR_POINT_CASES: readonly FourPointCase[] = [
  ...[...SHAPES, ...PADDED_SHAPES].map((shape) => ({
    shape,
    label: "",
    make: () => makeInputs(shape, 7 + shape.d),
  })),
  ...PADDED_SHAPES.flatMap((shape) =>
    (["sharp", "flat"] as const).map((softmax) => ({
      shape,
      label: ` (spike V, ${softmax} softmax)`,
      make: () => spikeInputs(shape, softmax),
    })),
  ),
];

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

/**
 * Pack `rows` i8 rows of `dim` elements into ceil(dim/4) words each, the bytes past element
 * `dim − 1` set to `padByte` (the layout of the `padded` key axis; `dim % 4 == 0` is
 * {@link packWords}).
 */
const packPaddedRows = (
  values: Int8Array<ArrayBuffer>,
  rows: number,
  dim: number,
  padByte: number,
): Uint32Array<ArrayBuffer> => {
  const words = Math.ceil(dim / 4);
  const padded = new Int8Array(rows * words * 4).fill(padByte);
  for (let row = 0; row < rows; row += 1) {
    padded.set(values.subarray(row * dim, (row + 1) * dim), row * words * 4);
  }
  return packWords(padded);
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

/** `padByte` fills the padding bytes of Vᵀ's last word when N % 4 != 0 (the producer writes 0). */
const quantise = (shape: Shape, inputs: Inputs, padByte = 0): Quantised => {
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
    vq: packPaddedRows(values.q, batch * d, n, padByte),
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
    // The key-axis form production takes for this N (`padded` when N % 4 != 0)
    const keyWords = attentionOnlineKeyWords(n);
    let wgsl = attentionOnlineI8a8Wgsl(options.dp4a, d, options.reduce, "bhsd", keyWords);
    let key = attentionOnlineI8a8Key(options.dp4a, d, options.reduce, "bhsd", keyWords);
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
      gpu.device.queue.writeBuffer(
        params,
        0,
        attentionOnlineI8a8Params(m, n, offset, scale, keyWords),
      );
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
  // ADR 0132 addendum 2026-10-27: N % 4 != 0 runs with a padded Vᵀ (before it, (128, 66) was not
  // eligible); a key axis needs at least one key
  assert(attentionOnlineI8a8Eligible(128, 66), "N % 4 != 0 takes the padded key axis");
  assert(attentionOnlineI8a8Eligible(64, 4106), "N % 4 != 0 takes the padded key axis");
  assert(!attentionOnlineI8a8Eligible(128, 0), "no key");
  assertEquals(attentionOnlineKeyWords(512), "whole");
  assertEquals(
    [65, 66, 67].map((n) => attentionOnlineKeyWords(n)),
    ["padded", "padded", "padded"],
  );
  // D = 128, in vec4 words: q 64 × 9 + k 64 × 9 + Vᵀ 16 × 33 + P̃ 16 × 17 (+ the reduce scratch, 64 × 16 f32)
  assertEquals(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "sg32"),
    (64 * 9 + 64 * 9 + 16 * 33 + 16 * 17) * 16,
  );
  assertEquals(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "wg"),
    (64 * 9 + 64 * 9 + 16 * 33 + 16 * 17) * 16 + 64 * 16 * 4,
  );
  assert(
    attentionOnlineI8a8WorkgroupStorageBytes(128, "sg32") < 32768,
    "the subgroup form fits 32 KiB",
  );
});

test("the key-axis form ties the Vᵀ row length, its producer and the uniform to N", () => {
  // Words per Vᵀ row (by hand): 512 / 4 = 128, ceil(65 / 4) = 17, ceil(4,106 / 4) = 1,027
  assertEquals([512, 65, 4106].map(attentionOnlineValueRowWords), [128, 17, 1027]);
  // vq = B·D·words·4 bytes: 2 · 128 · 51 · 4 at N = 203 (51 = ceil(203 / 4)), B·D·N at N = 512
  assertEquals(attentionI8a8TempBytes({ batch: 2, rows: 10, cols: 203, depth: 128 }).pv.vq, 52224);
  assertEquals(attentionI8a8TempBytes({ batch: 2, rows: 10, cols: 512, depth: 128 }).pv.vq, 131072);
  // The producer: the plain quantize_rows at N % 4 == 0 (its key unchanged), the padded one else
  assertEquals(
    attentionValueQuantization(256, 512, "whole").key,
    quantizeRowsKey(quantizeRowsGeometry(512)),
  );
  assert(attentionValueQuantization(256, 203, "padded").key.endsWith(":pad"));
  // A reader of the other form fails loudly at the producer — ③PV passes "whole" (it needs
  // N % 4 == 0), so an N it cannot read never gets a padded Vᵀ
  assertThrows(() => attentionValueQuantization(256, 43, "whole"), Error, "N = 43");
  assertThrows(() => attentionValueQuantization(256, 512, "padded"), Error, "N = 512");
  // The uniform: N must be a key length of the form the pipeline was compiled with
  assertEquals(attentionOnlineI8a8Params(64, 203, 0, 0.5, "padded")[1], 203);
  assertThrows(() => attentionOnlineI8a8Params(64, 203, 0, 0.5, "whole"), Error, "N = 203");
  assertThrows(() => attentionOnlineI8a8Params(64, 512, 0, 0.5, "padded"), Error, "N = 512");
});

test("the declared workgroup storage is what the WGSL declares (the route decides from it)", () => {
  const elementBytes: Record<string, number> = { "vec4<u32>": 16, f32: 4, u32: 4 };
  for (const dp4a of [true, false]) {
    for (const depth of [64, 128]) {
      for (const reduce of ["sg32", "wg"] as const) {
        const where = `${dp4a ? "dp4a" : "emu"} d${depth} ${reduce}`;
        const wgsl = attentionOnlineI8a8Wgsl(dp4a, depth, reduce);
        const declarations = [
          ...wgsl.matchAll(/var<workgroup> \w+: array<(vec4<u32>|f32|u32), (\d+)>;/g),
        ];
        // Every workgroup declaration must be one the sum below understands.
        assertEquals(
          declarations.length,
          (wgsl.match(/var<workgroup>/g) ?? []).length,
          `${where}: every workgroup declaration is counted`,
        );
        const declared = declarations
          .map(([, type, count]) => Math.ceil((elementBytes[type] * Number(count)) / 16) * 16)
          .reduce((sum, bytes) => sum + bytes, 0);
        assertEquals(attentionOnlineI8a8WorkgroupStorageBytes(depth, reduce), declared, where);
      }
    }
  }
});

describe.skipIf(!GPU_AVAILABLE)("online int8 attention (ADR 0132) — the four-point check", () => {
  for (const { shape, label, make } of FOUR_POINT_CASES) {
    const where = `${shape.name}${label}`;
    it(`${where}: ① error model, ② A/B vs three-pass a8, ③ structure (wg)`, async () => {
      const gpu = await acquireGpu();
      try {
        const inputs = make();
        const quant = quantise(shape, inputs);
        const exact = exactAttention(shape, quant, inputs.scale);
        const dp4a = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "wg" });
        const model = withinModel(dp4a, exact.o, exact.bound, 1);
        console.log(
          `[attention_online] ${where}: ${(100 * model.worstRatio).toFixed(1)}% of the error model used`,
        );
        assertBand(model, `${where} ① vs exact`);
        // ② both paths carry one lattice step each
        const threePass = threePassA8(shape, inputs, quant);
        assertBand(withinModel(dp4a, threePass, exact.bound, 2), `${where} ② vs three-pass a8`);
        // ③ the TS mirror (ULP-level differences of exp / division move P̃ by at most one step)
        const mirror = referenceAttentionOnlineI8a8({
          ...inputs,
          batch: shape.batch,
          m: shape.m,
          n: shape.n,
          d: shape.d,
        });
        assertBand(withinModel(dp4a, mirror, exact.bound, 2), `${where} ③ vs the TS mirror`);
        // ③ emulated dot product, a second dispatch, row windows: bit-identical
        const emulated = await runOnline(gpu, shape, quant, inputs.scale, {
          dp4a: false,
          reduce: "wg",
        });
        assertBitIdentical(emulated, dp4a, `${where} ③ dp4a vs emulation`);
        const again = await runOnline(gpu, shape, quant, inputs.scale, {
          dp4a: true,
          reduce: "wg",
        });
        assertBitIdentical(again, dp4a, `${where} ③ determinism`);
        if (shape.m > ATTENTION_ONLINE_ROWS) {
          const split = await runOnline(gpu, shape, quant, inputs.scale, {
            dp4a: true,
            reduce: "wg",
            windows: [
              [0, ATTENTION_ONLINE_ROWS],
              [ATTENTION_ONLINE_ROWS, shape.m - ATTENTION_ONLINE_ROWS],
            ],
          });
          assertBitIdentical(split, dp4a, `${where} ③ row windows`);
        }
      } finally {
        gpu.destroy();
      }
    });
  }

  for (const { shape, label, make } of FOUR_POINT_CASES) {
    const where = `${shape.name}${label}`;
    it(`${where}: ③ the subgroup form is bit-identical to the workgroup-memory form (skips without subgroups)`, async (context) => {
      if (!(await subgroupsSupported())) context.skip();
      const gpu = await acquireGpu({ subgroups: true });
      try {
        const inputs = make();
        const quant = quantise(shape, inputs);
        const wg = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "wg" });
        const sg = await runOnline(gpu, shape, quant, inputs.scale, { dp4a: true, reduce: "sg32" });
        assertBitIdentical(sg, wg, `${where} ③ sg32 vs wg`);
      } finally {
        gpu.destroy();
      }
    });
  }

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

  for (const shape of PADDED_SHAPES) {
    it(`${shape.name}: the padded key axis — padding bytes 0 and 127 give Uint32-identical outputs`, async () => {
      // The primary detector of a mask bound past N (a padding key whose P̃ is not 0 moves O by
      // its byte): the mask bound at ceil(N/4)·4, patched into the generator, turned it red at every
      // padded shape (2026-10-27). With the producer's zero bytes a leaked key enters l only, which
      // this check cannot see: the f64 band sees it where it weighs (N = 65: 1.12× the bound) but
      // not at N = 4,106 (measured 3.6% of the model on the dense inputs, 8.1% on the flat spike).
      const gpu = await acquireGpu();
      try {
        const inputs = makeInputs(shape, 7 + shape.d);
        const zero = quantise(shape, inputs, 0);
        const filled = quantise(shape, inputs, 127);
        // The two inputs differ exactly in the padding bytes (else this compares a run with itself)
        const differing = zero.vq.filter((word, i) => word !== filled.vq[i]).length;
        assertEquals(differing, shape.batch * shape.d, `${shape.name}: one padded word per Vᵀ row`);
        const a = await runOnline(gpu, shape, zero, inputs.scale, { dp4a: true, reduce: "wg" });
        const b = await runOnline(gpu, shape, filled, inputs.scale, { dp4a: true, reduce: "wg" });
        assertBitIdentical(b, a, `${shape.name}: padding 127 vs 0`);
      } finally {
        gpu.destroy();
      }
    });
  }

  /**
   * Faults of the padded key axis, each on the spike-V inputs that expose it (see
   * {@link spikeInputs}): the mask bound moved from N to ceil(N/4)·4 (the padding keys enter the
   * softmax — under the fault they score 0, since their k rows load as zero, so they need the flat
   * softmax; the padding bytes are 127, the value the fault reads into O), and the two row-length
   * faults of Vᵀ — the whole form's floor(N/4) (every row from the second on read at the wrong
   * place) and the last word of each row dropped (only the keys from 4·(ceil(N/4) − 1) on lost).
   */
  const PADDED_FAULTS: readonly {
    readonly label: string;
    readonly softmax: "sharp" | "flat";
    readonly apply: (wgsl: string) => string;
  }[] = [
    {
      label: "mask-at-padded-length",
      softmax: "flat",
      apply: (wgsl) =>
        wgsl.replace(/let kok = vec4<bool>\([^;]*\);/, (line) =>
          line.replaceAll("< dims.n", "< n4 * 4u"),
        ),
    },
    {
      label: "row-length-floor",
      softmax: "sharp",
      apply: (wgsl) => wgsl.replace("let n4 = (dims.n + 3u) / 4u;", "let n4 = dims.n / 4u;"),
    },
    {
      label: "last-word-dropped",
      softmax: "sharp",
      apply: (wgsl) => wgsl.replaceAll("< n4)", "< n4 - 1u)"),
    },
  ];
  for (const shape of PADDED_SHAPES) {
    for (const fault of PADDED_FAULTS) {
      it(`${shape.name}: ④ fault injection '${fault.label}' leaves the model (spike V, ${fault.softmax} softmax)`, async () => {
        const gpu = await acquireGpu();
        try {
          const inputs = spikeInputs(shape, fault.softmax);
          const quant = quantise(shape, inputs, 127);
          const exact = exactAttention(shape, quant, inputs.scale);
          // The control: the kernel itself stays inside the model on these inputs.
          const clean = await runOnline(gpu, shape, quant, inputs.scale, {
            dp4a: true,
            reduce: "wg",
          });
          assertBand(withinModel(clean, exact.o, exact.bound, 1), `${shape.name} control`);
          const o = await runOnline(gpu, shape, quant, inputs.scale, {
            dp4a: true,
            reduce: "wg",
            mutate: fault,
          });
          const report = withinModel(o, exact.o, exact.bound, 1);
          console.log(
            `[attention_online] ${shape.name}: fault '${fault.label}' uses ${report.worstRatio.toFixed(1)} × the error model`,
          );
          assert(
            report.nonFinite > 0 || report.worstRatio > 1,
            `fault '${fault.label}' stayed inside the model at ${shape.name} (worst ${report.worstRatio.toFixed(3)}) — the check is vacuous`,
          );
        } finally {
          gpu.destroy();
        }
      });
    }
  }
});
