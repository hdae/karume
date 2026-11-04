// The execution path of the **i8a8 variant** of fused attention ③PV (P̃ = round(127·exp(S−m))
// without materialization × per-column i8 of V). This file also holds the ①QK i8a8 variant (q / k
// both per-token i8, integer dot product) and the Session-path route checks for both stages.
//
// Design = docs/research/2026-08-04-attention-a8-design.md §2.2 / §2.3 / §4.2, implementation =
// src/codegen/strided.ts (Vᵀ, reused unmodified) + src/kernels/quantize-rows.ts (unmodified) +
// src/kernels/attention-i8a8.ts, opt-in = `SessionOptions.attentionCompute: "a8"` (default "f32").
//
// ## The numeric contract this file pins (decisively different from ①QK — design §4.2)
//
// **A naive atol=0 does not hold** for O of ③PV. The quantized values on the A side are
// `round(127·exp(S−m))`; `exp` is not guaranteed to match bit for bit between WGSL and JS, and
// `127·exp` is densely distributed over [0,127], so some elements always land on a rounding
// boundary. So the contract is **split in 2 steps**:
//
// (a) **atol=0** against **the pure function from the integers on**
//     (`referenceAttentionPvI8a8Core`). The `qP` the GPU used cannot be observed (P is not
//     materialized), so **another run with V set to the identity matrix serves as "a reader of
//     qP"** (it gives vq = 127·I / vs = 1/127, so O can only be 127·qP·prow·vs). vq / vs are read
//     back from the temporary buffers as is.
// (a2) **The quantization of Vᵀ** (strided permute + quantize_rows) is atol=0 against the TS
//     reference. The only mechanical detector of a mixed-up permute axis and of the per-column
//     quantization.
// (b) **The generation of `qP`** is a separate check: mismatches with the TS reference are **always
//     ±1 step** (a single ±2 is an implementation bug), and the mismatch rate itself is recorded.
//     The form chosen so that the check is not a tautology.
// (c) dp4a and the emulation are atol=0 (whether the extension is present only affects speed).
//
// ## The ①QK contract (design §2.1 / §4.1)
//
// **For S (the output of ①QK), the GPU and the TS reference match at atol=0**. It rests on the same
// 2 points as linear's w8a8 (tests/gpu_i8a8_test.ts) — "the dot product is an exact i32 sum (order
// independent)" and "the floating-point operations are a few per output element, all
// round-to-nearest-even". So however the tiling, the zero fill of the K remainder or the shared
// memory layout change, the values should not move by a single bit; if they move, the
// implementation is wrong. S is an intermediate buffer on the Session path, so the ①QK raw-dispatch
// section below is the oracle for S; the Session-path section only checks route selection and
// degradation.
//
// ## What this iteration deliberately does not verify
//
// - Adding it to the grid-stride degeneration harness (tests/gpu_gridstride_test.ts) is **not
//   needed**. The new kernel covers the whole domain as "1 workgroup = 1 output tile" (exceeding
//   the limit is the fail loudly of `tiledWorkgroups`). The `quantize_rows` and `strided` that run
//   with it are already covered.
// - Bit identity with the decomposed path (tests/gpu_attention_parity_test.ts) cannot hold for i8a8
//   in principle. That test stays as the check dedicated to the f32 / f16 variants (separate keys,
//   so they cannot mix structurally).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { gridStrideWorkgroups, tiledWorkgroups } from "../src/codegen/dispatch.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../src/codegen/strided.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import { gemmUsesVec4 } from "../src/kernels/gemm.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { LINEAR_I8A8_MAX_K } from "../src/kernels/linear-i8a8.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../src/kernels/attention-i8a8.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import {
  quantizeRowsReference,
  quantizeRowsTieMargin,
  referenceAttentionPvI8a8Core,
  referenceAttentionPvQuant,
  referenceAttentionQkI8a8,
} from "../src/reference/i8a8.ts";
import {
  createSessionFromContainer,
  I8A8_DOT,
  type I8a8Dot,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { fill, GRAPH_NAME, openGraphModel, singleOpDeclaration } from "./helpers/model-fixture.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * The margin from the rounding boundaries that allows claiming atol=0 (at least 50 times the 2.5
 * ULP ≈ 1.9e-5).
 */
const TIE_MARGIN = 1e-3;

/**
 * The P̃ grid and `1/127` (**used only for the arithmetic of the observation** — the reference side
 * holds its constants independently in src/reference/i8a8.ts).
 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

/**
 * Exhaustive atol=0 comparison. It uses `===`, so ±0 are treated as equal, but otherwise not even a
 * 1-bit difference is allowed.
 */
const assertExact = (
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  where: string,
): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  for (let i = 0; i < actual.length; i += 1) {
    if (Number.isNaN(expected[i])) {
      assert(Number.isNaN(actual[i]), `${where}[${i}]: ${actual[i]} against a reference NaN`);
      continue;
    }
    assert(
      actual[i] === expected[i],
      `${where}[${i}]: GPU ${actual[i]} / reference ${expected[i]}`,
    );
  }
};

// ---------------------------------------------------------------------------
// Shapes and data
// ---------------------------------------------------------------------------

type PvShape = {
  readonly name: string;
  /** The batch axis folding B·H. **2 or more is the only detector of the batch base**. */
  readonly batch: number;
  /** Output rows = M. */
  readonly m: number;
  /** Reduction axis = N (`N % 4 == 0` is the i8a8 eligibility condition). */
  readonly n: number;
  /** Output columns = D (`D % 4 == 0` is the condition for the v4 write-out). */
  readonly d: number;
};

/**
 * Which shapes are hit.
 *
 * MUST: include shapes with **B·H ≥ 2 and different statistics and patterns per head** (the only
 * detector of the batch base — changing only the magnitude per head yields the same integers after
 * quantization, and a mixed-up base on the payload side does not show in the values. A lesson of
 * iteration 1).
 * MUST: **make m / d cross the tile edge 64** (the detector of mixing up the row inv and the column
 * scale).
 * MUST: **pair the v4 path (D%4==0) with the scalar path** (the write-out of O is a different
 * formula per variant).
 * MUST: include **N % 16 != 0** (a remainder of the K tile 16 = the detector that the zero fill of
 * the P̃ tile takes effect. N=20 is k4=5, with only 1 of the 4 packs of the 2nd tile valid).
 */
const PV_SHAPES: readonly PvShape[] = [
  // v4 / m spans 2 tiles of 64 / N=68 is k4=17, 5 K tiles (remainder of 1 pack)
  { name: "v4 B·H6 M65 N68 D20", batch: 6, m: 65, n: 68, d: 20 },
  // Scalar (D % 4 != 0) fitting in 1 tile + 2 heads / N=20 is a remainder tile
  { name: "scalar B·H2 M17 N20 D19", batch: 2, m: 17, n: 20, d: 19 },
  // 9 K tiles (k4=33, remainder of 1 pack) / D=5 is scalar / M=70 gives 2 row tiles
  { name: "scalar long-K B·H2 M70 N132 D5", batch: 2, m: 70, n: 132, d: 5 },
  // The smallest shape fitting in 1 tile (the path where every guard stays true)
  { name: "v4 small B·H1 M3 N8 D4", batch: 1, m: 3, n: 8, d: 4 },
  // A measured shape (DiT self-attention) — **2 column tiles** (D=128) hit the binding of the
  // column scale
  { name: "DiT shape B·H4 M64 N64 D128", batch: 4, m: 64, n: 64, d: 128 },
];

/**
 * Raw values of S (deterministic — no random numbers). **The pattern changes** per head / row, so
 * P̃ spreads within a row from 0 (elements that vanish in rounding) to 127 (the row maximum).
 *
 * MUST: make the steps **dense** (with only "clean" values like steps of `0.5`, `127·exp` never
 * lands near a rounding boundary and check (b) becomes effectively a tautology — measured: 0 of
 * 62,088 elements). 2 terms with incommensurate steps are layered to fill the range of 15 finely.
 */
const scoreAt = (b: number, row: number, col: number): number => {
  const index = col * 7 + row * 3 + b * 5;
  return ((index % 23) - 11) * 0.6373 + ((index * 13) % 11) * 0.1237 - 0.4291;
};

/**
 * Raw values of V. Built so that **the statistics are skewed along both the column (D) and the row
 * (N) direction** — mistaking the per-column scale for per-token (per row n) shows in the values
 * only when this skew is present (design §2.3). The pattern is shifted per head because a
 * difference in magnitude alone makes the quantized integers match, and a mixed-up payload base
 * could not be detected (a lesson of iteration 1).
 */
const valueAt = (b: number, row: number, col: number, d: number): number => {
  // MUST: do not make the steps commensurate with the scale grid (if `x/s` lands on a half-integer,
  // the GPU division being allowed up to 2.5 ULP makes the quantized value wobble by ±1 step, and
  // the atol=0 of (a2) "happens to fail"). The margin is measured every time with
  // {@link quantizeRowsTieMargin} and made a check.
  const base = ((((row * d + col + b * 11) * 5) % 17) - 8) * 0.2731 + 0.1237;
  const colGain = 0.04 * Math.pow(3.7, col % 5);
  const rowGain = 0.25 + (row % 6) * 0.83;
  return base * colGain * rowGain * (1 + 0.53 * b);
};

type PreparedPv = {
  readonly s: Float32Array<ArrayBuffer>;
  /** Row statistics `[batch·m, 2]` (`[0]` = the row maximum / `[1]` = `1/Σexp(S−m)`). */
  readonly stats: Float32Array<ArrayBuffer>;
  readonly rowInv: Float32Array<ArrayBuffer>;
  readonly v: Float32Array<ArrayBuffer>;
  /** Vᵀ`[batch, d, n]` (the content the GPU's strided permute should produce). */
  readonly vt: Float32Array<ArrayBuffer>;
  /** The margin of Vᵀ's `x/s` from the rounding boundaries. */
  readonly margin: number;
};

const preparePv = (shape: PvShape): PreparedPv => {
  const { batch, m, n, d } = shape;
  const s = new Float32Array(batch * m * n);
  const stats = new Float32Array(batch * m * 2);
  const rowInv = new Float32Array(batch * m);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      let max = Number.NEGATIVE_INFINITY;
      for (let col = 0; col < n; col += 1) {
        const value = Math.fround(scoreAt(b, row, col));
        s[(b * m + row) * n + col] = value;
        if (value > max) max = value;
      }
      let sum = 0;
      for (let col = 0; col < n; col += 1) {
        sum = Math.fround(sum + Math.fround(Math.exp(s[(b * m + row) * n + col] - max)));
      }
      const inv = Math.fround(1 / sum);
      stats[(b * m + row) * 2] = max;
      stats[(b * m + row) * 2 + 1] = inv;
      rowInv[b * m + row] = inv;
    }
  }
  const v = new Float32Array(batch * n * d);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < d; col += 1) {
        v[(b * n + row) * d + col] = valueAt(b, row, col, d);
      }
    }
  }
  const vt = transpose(v, batch, n, d);
  return { s, stats, rowInv, v, vt, margin: quantizeRowsTieMargin(vt, batch * d, n) };
};

/**
 * `[batch, n, d]` → `[batch, d, n]` (the job the existing strided permute does on the GPU side).
 */
const transpose = (
  v: ArrayLike<number>,
  batch: number,
  n: number,
  d: number,
): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * d * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col < d; col += 1) {
        out[(b * d + col) * n + row] = v[(b * n + row) * d + col];
      }
    }
  }
  return out;
};

/**
 * An identity matrix per head `[batch, n, n]` (the reader of qP — {@link observeQuantizedP} below).
 */
const identityV = (batch: number, n: number): Float32Array<ArrayBuffer> => {
  const out = new Float32Array(batch * n * n);
  for (let b = 0; b < batch; b += 1) {
    for (let i = 0; i < n; i += 1) out[(b * n + i) * n + i] = 1;
  }
  return out;
};

// ---------------------------------------------------------------------------
// Raw dispatches (the executor's 3 dispatches of ③PV i8a8, reading back the temporary buffers too)
// ---------------------------------------------------------------------------

const readbackBytes = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  bytes: number,
): Promise<Uint8Array<ArrayBuffer>> => {
  const size = Math.max(4, bytes);
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
  return new Uint8Array(copy, 0, bytes);
};

type PvRun = {
  readonly o: Float32Array<ArrayBuffer>;
  /**
   * The quantized Vᵀ the GPU used (flat i8 of `[batch, d, n]` — the 4-per-word packing read as is).
   */
  readonly vq: Int8Array<ArrayBuffer>;
  /** The row scale of Vᵀ the GPU used (`[batch·d]`). */
  readonly vs: Float32Array<ArrayBuffer>;
};

/**
 * Runs the 3 dispatches `strided` (Vᵀ) → `quantize_rows` → i8a8 GEMM raw. S and the row statistics
 * are **supplied directly by the test** (the contract of ③ is "take S and stats and write O", so ①
 * and ② need not run).
 */
const runPvI8a8 = async (
  gpu: GpuContext,
  shape: { readonly batch: number; readonly m: number; readonly n: number; readonly d: number },
  input: {
    readonly s: Float32Array<ArrayBuffer>;
    readonly stats: Float32Array<ArrayBuffer>;
    readonly v: Float32Array<ArrayBuffer>;
  },
  dot: I8a8Dot,
  where: string,
): Promise<PvRun> => {
  const { batch, m, n, d } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const host = (source: Float32Array<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    const scores = host(input.s);
    const stats = host(input.stats);
    const v = host(input.v);

    // (a) v[batch,N,D] → vT[batch,D,N] (the existing strided read-copy family = permute)
    const stridedSpec = { dtype: "f32" } as const;
    const permuteKey = stridedKey(stridedSpec);
    const { pipeline: permutePipeline, layout: permuteLayout } = await cache.get(
      permuteKey,
      stridedWgsl(stridedSpec),
    );
    const permuteParams = arena.allocHostWritten(40, STORAGE_IN);
    gpu.device.queue.writeBuffer(
      permuteParams,
      0,
      stridedParams([batch, d, n], permuteSrcStrides([batch, n, d], [0, 2, 1]), 0),
    );
    const vt = arena.allocRegion(Math.max(4, batch * d * n * 4));
    scheduler.dispatch(
      permutePipeline,
      gpu.device.createBindGroup({
        layout: permuteLayout,
        entries: [
          { binding: 0, resource: { buffer: permuteParams } },
          { binding: 1, resource: { buffer: v } },
          { binding: 2, resource: { buffer: vt } },
        ],
      }),
      [gridStrideWorkgroups(batch * d * n, STRIDED_WORKGROUP_SIZE, limit), 1, 1],
      permuteKey,
    );

    // (b) Per-row quantization of Vᵀ (= per-column quantization of V)
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    const quantizeParams = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(quantizeParams, 0, quantizeRowsParams(batch * d, n));
    const vq = arena.allocRegion(Math.max(4, batch * d * n));
    const vs = arena.allocRegion(Math.max(4, batch * d * 4));
    scheduler.dispatch(
      quantizePipeline,
      gpu.device.createBindGroup({
        layout: quantizeLayout,
        entries: [
          { binding: 0, resource: { buffer: quantizeParams } },
          { binding: 1, resource: { buffer: vt } },
          { binding: 2, resource: { buffer: vq } },
          { binding: 3, resource: { buffer: vs } },
        ],
      }),
      [gridStrideWorkgroups(batch * d, 1, limit), 1, 1],
      QUANTIZE_ROWS_KEY,
    );

    // (c) The integer dot-product GEMM (P̃ is produced by the A tile fill)
    const v4 = attentionPvI8a8UsesVec4(d);
    const dp4a = dot === "dp4a";
    const pvGeometry = defaultI8a8Geometry("attention_pv");
    const pipelineKey = attentionPvI8a8Key(v4, dp4a);
    const { pipeline, layout } = await cache.get(pipelineKey, attentionPvI8a8Wgsl(v4, dp4a));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionPvI8a8Params(m, d, n));
    const out = arena.allocRegion(Math.max(4, batch * m * d * 4));
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: scores } },
          { binding: 2, resource: { buffer: vq } },
          { binding: 3, resource: { buffer: stats } },
          { binding: 4, resource: { buffer: out } },
          { binding: ATTENTION_PV_V_SCALE_BINDING, resource: { buffer: vs } },
        ],
      }),
      [
        // MUST: derive the edges from ③PV's own geometry (a default distinct from ①QK — the same
        // resolution point as the key and the generated code)
        tiledWorkgroups(d, i8a8TileN(pvGeometry), limit, where),
        tiledWorkgroups(m, i8a8TileM(pvGeometry), limit, where),
        tiledWorkgroups(batch, 1, limit, where),
      ],
      pipelineKey,
    );
    await scheduler.flush();

    const outBytes = await readbackBytes(gpu.device, out, batch * m * d * 4);
    const vqBytes = await readbackBytes(gpu.device, vq, batch * d * n);
    const vsBytes = await readbackBytes(gpu.device, vs, batch * d * 4);
    return {
      o: new Float32Array(outBytes.buffer, 0, batch * m * d),
      // pack4xI8 places component i into byte i, so the flat i8 order is the byte sequence itself
      vq: new Int8Array(vqBytes.buffer, 0, batch * d * n),
      vs: new Float32Array(vsBytes.buffer, 0, batch * d),
    };
  } finally {
    await arena.destroy();
  }
};

/**
 * Observes the `qP` the GPU produced in the A tile.
 *
 * Making V an identity matrix per head gives `vq = 127·I` / `vs = 1/127`, so
 * `O[m,j] = f32(127·qP[m,j]) · (prow · vs[j])` = **a reader of qP**.
 * MUST: assert the premise (vq diagonal 127, off-diagonal 0) before dividing — dividing with the
 * premise broken shifts the observed values quietly, and the atol=0 of (a) "happens to pass".
 */
const observeQuantizedP = (
  run: PvRun,
  shape: { readonly batch: number; readonly m: number; readonly n: number },
  rowInv: ArrayLike<number>,
): Int8Array<ArrayBuffer> => {
  const { batch, m, n } = shape;
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < n; row += 1) {
      assertEquals(
        run.vs[b * n + row],
        INV_P_ABS_MAX,
        `column scale of the identity matrix [${b},${row}]`,
      );
      for (let col = 0; col < n; col += 1) {
        assertEquals(
          run.vq[(b * n + row) * n + col],
          row === col ? P_ABS_MAX : 0,
          `quantized value of the identity matrix [${b},${row},${col}]`,
        );
      }
    }
  }
  const qp = new Int8Array(batch * m * n);
  for (let b = 0; b < batch; b += 1) {
    for (let row = 0; row < m; row += 1) {
      const prow = Math.fround(rowInv[b * m + row] * INV_P_ABS_MAX);
      for (let col = 0; col < n; col += 1) {
        const combined = Math.fround(prow * run.vs[b * n + col]);
        const raw = run.o[(b * m + row) * n + col] / (P_ABS_MAX * combined);
        const rounded = Math.round(raw);
        assert(
          Math.abs(raw - rounded) < 1e-3 && rounded >= 0 && rounded <= P_ABS_MAX,
          `the qP readout is not an integer [${b},${row},${col}]: ${raw}`,
        );
        qp[(b * m + row) * n + col] = rounded;
      }
    }
  }
  return qp;
};

// ---------------------------------------------------------------------------
// Margin from the rounding boundaries (the check that Vᵀ's quantization may be compared at atol=0)
// ---------------------------------------------------------------------------

/**
 * The dispatch count of `quantize_rows` (the small-D variants split keys per geometry, so they are
 * grouped by prefix).
 */
const quantizeRowsDispatches = (byKey: ReadonlyMap<string, number>): number =>
  [...byKey]
    .filter(([key]) => key.startsWith(QUANTIZE_ROWS_KEY))
    .reduce((sum, [, count]) => sum + count, 0);

test("attention_pv i8a8: Vᵀ of the shapes stays well away from the rounding boundaries (premise of the vq comparison)", () => {
  // The GPU division (`x/s`) is allowed up to 2.5 ULP, so an element whose `x/s` is near a
  // half-integer can wobble by ±1 step. Unless the margin is **measured every time** and made a
  // check, the atol=0 of (a2) "merely happens to pass".
  let worst = Number.POSITIVE_INFINITY;
  for (const shape of PV_SHAPES) {
    const { margin } = preparePv(shape);
    assert(
      margin > TIE_MARGIN,
      `${shape.name}: the margin from the rounding boundaries is only ${margin}`,
    );
    worst = Math.min(worst, margin);
  }
  assert(worst > TIE_MARGIN, `worst margin ${worst}`);
});

// ---------------------------------------------------------------------------
// (a) atol=0 against the pure function from the integers on / (a2) atol=0 of the quantization of Vᵀ
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "O of attention_pv i8a8 matches the TS reference (the pure function split just before the integers) at atol=0 (v4 / scalar / tile edge / K remainder / B·H≥2, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of PV_SHAPES) {
        const { batch, m, n, d } = shape;
        const prepared = preparePv(shape);
        const run = await runPvI8a8(gpu, shape, prepared, "dp4a", shape.name);

        // (a2) Vᵀ (the materialized permute) and its per-column quantization are bit-identical to
        // the TS reference. **The only mechanical detector of a mixed-up permute axis and of going
        // per-token**.
        const expectedV = quantizeRowsReference(prepared.vt, batch * d, n);
        assertExact(run.vq, expectedV.q, `${shape.name}: quantized values of Vᵀ`);
        assertExact(run.vs, expectedV.scale, `${shape.name}: column scale of Vᵀ`);
        // Tautology guard: if the column scales are all the same, "pulling by column" verifies
        // nothing
        assert(
          new Set([...run.vs]).size > Math.max(2, batch),
          `${shape.name}: the column scales are not spread out (the axis detector does not hold)`,
        );

        // (a) Observe the qP the GPU used, then compare against the integer-only reference
        const observed = await runPvI8a8(
          gpu,
          { batch, m, n, d: n },
          { s: prepared.s, stats: prepared.stats, v: identityV(batch, n) },
          "dp4a",
          `${shape.name} (qP observation)`,
        );
        const qp = observeQuantizedP(observed, shape, prepared.rowInv);
        const expected = referenceAttentionPvI8a8Core({
          qp,
          vq: run.vq,
          rowInv: prepared.rowInv,
          vs: run.vs,
          batch,
          m,
          n,
          d,
        });
        assertExact(run.o, expected, shape.name);

        // Tautology guard: if O is constant, "equal" verifies nothing
        assert(new Set([...run.o]).size > 1, `${shape.name}: O is constant`);
        // qP reaches both ends 0 and 127 (elements vanishing in rounding and the row maximum
        // coexist)
        assert([...qp].includes(P_ABS_MAX), `${shape.name}: qP has no 127`);
        assert([...qp].includes(0), `${shape.name}: qP has no 0`);
        // The statistics differ per head (the very premise for the batch-base detector to hold)
        if (batch > 1) {
          const head0 = run.o.slice(0, m * d);
          const head1 = run.o.slice(m * d, 2 * m * d);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${shape.name}: O of head 0 and head 1 is identical (a mixed-up base would not show in the values)`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (b) The generation of qP (an implementation difference of exp — atol=0 cannot hold in principle)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "attention_pv i8a8: qP always matches the TS reference within ±1 step (an implementation difference of exp, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    let total = 0;
    let mismatched = 0;
    try {
      for (const shape of PV_SHAPES) {
        const { batch, m, n } = shape;
        const prepared = preparePv(shape);
        const observed = await runPvI8a8(
          gpu,
          { batch, m, n, d: n },
          { s: prepared.s, stats: prepared.stats, v: identityV(batch, n) },
          "dp4a",
          `${shape.name} (qP observation)`,
        );
        const qp = observeQuantizedP(observed, shape, prepared.rowInv);
        const rowMax = new Float32Array(batch * m);
        for (let row = 0; row < batch * m; row += 1) rowMax[row] = prepared.stats[row * 2];
        const expected = referenceAttentionPvQuant(prepared.s, rowMax, batch * m, n);
        for (let i = 0; i < qp.length; i += 1) {
          const diff = qp[i] - expected[i];
          // MUST: ±2 steps or more is an implementation bug even once (a few-ULP difference of exp
          // cannot reach it)
          assert(
            Math.abs(diff) <= 1,
            `${shape.name}: qP[${i}] is GPU ${qp[i]} / reference ${expected[i]} (±2 steps or more)`,
          );
          if (diff !== 0) mismatched += 1;
          total += 1;
        }
      }
    } finally {
      gpu.destroy();
    }
    // A record of the measurement (design §4.2's "the mismatch rate itself becomes the record"). An
    // upper bound is set too so it is not a tautology
    const rate = mismatched / total;
    console.log(
      `GPU/TS mismatch rate of qP: ${mismatched} / ${total} = ${(rate * 100).toFixed(4)}% (all ±1 step)`,
    );
    assert(
      rate < 0.02,
      `the qP mismatch rate ${rate} exceeded 2% (large for an exp implementation difference)`,
    );
  },
);

// ---------------------------------------------------------------------------
// (c) dp4a and the emulation
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "attention_pv i8a8: the dot4I8Packed version and the emulated version match at atol=0 (the extension only affects speed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of PV_SHAPES) {
        const prepared = preparePv(shape);
        const dp4a = await runPvI8a8(gpu, shape, prepared, "dp4a", shape.name);
        const emu = await runPvI8a8(gpu, shape, prepared, "emu", shape.name);
        assertExact(emu.o, dp4a.o, `${shape.name}: dp4a vs emulation`);
        assert(new Set([...dp4a.o]).size > 1, `${shape.name}: O is constant`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// ①QK i8a8 — raw dispatch (reads S directly; the atol=0 oracle for S)
// ---------------------------------------------------------------------------

/**
 * Deterministic data sequences (no random numbers — failures would not reproduce).
 *
 * MUST: do not make the steps commensurate with the scale grid (if `x/s` lands on a half-integer,
 * the GPU division being allowed up to 2.5 ULP makes the quantized value wobble by ±1 step, and
 * atol=0 starts to "happen to fail"). The margin is measured every time with
 * {@link quantizeRowsTieMargin} and made a check.
 */
const QUERY = (i: number): number => (((i * 3) % 29) - 14) * 0.3717 + 0.0419;
const KEY = (i: number): number => (((i * 3) % 41) - 20) * 0.2917 - 0.0173;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/**
 * The half scale (`√scale_factor` of torch's math decomposition) — the value per the contract of
 * ADR 0023.
 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

type QkShape = {
  readonly name: string;
  /** The batch axis folding B·H. **2 or more is the only detector of the batch base**. */
  readonly batch: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Which shapes are hit.
 *
 * MUST: include shapes with **B·H ≥ 2 and different statistics per head** (the only detector of the
 * batch base — changing only the magnitude per head yields the same integers after quantization,
 * and a mixed-up base on the payload side does not show in the values. Change **the pattern
 * itself**).
 * MUST: **make m / n cross the tile edge 64** (the detector of mixing up the row scale and the
 * column scale).
 * MUST: **pair the v4 path with the scalar path** (the write-out of S is a different formula per
 * variant).
 * MUST: include **D % 16 != 0** (the detector that the zero fill of the K remainder takes effect).
 */
const QK_SHAPES: readonly QkShape[] = [
  // v4 (N % 4 == 0) / m and n both span 2 tiles of 64 / D=20 is k4=5, 2 K tiles (remainder of 1
  // pack)
  { name: "v4 B·H6 M65 N68 D20", batch: 6, m: 65, n: 68, d: 20 },
  // Scalar (N % 4 != 0) fitting in 1 tile + 2 heads
  { name: "scalar B·H2 M17 N19 D20", batch: 2, m: 17, n: 19, d: 20 },
  // 9 K tiles (k4=33, remainder of 1 pack) / n=5 is scalar / m=70 gives 2 row tiles
  { name: "scalar long-K B·H2 M70 N5 D132", batch: 2, m: 70, n: 5, d: 132 },
  // The smallest shape fitting in 1 tile (the path where every guard stays true)
  { name: "v4 small B·H1 M3 N8 D4", batch: 1, m: 3, n: 8, d: 4 },
  // A measured shape (DiT self-attention) — sits exactly on the tile edges
  { name: "DiT shape B·H4 M64 N64 D128", batch: 4, m: 64, n: 64, d: 128 },
];

/**
 * Data with **both the pattern and the magnitude** changed per head (the premise of the batch-base
 * detector). Changing only the magnitude makes the quantized integers identical across heads, and a
 * mixed-up payload base would not show in the values.
 */
const headVaried = (
  base: (index: number) => number,
  batch: number,
  rows: number,
  d: number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(batch * rows * d);
  const stride = rows * d;
  for (let head = 0; head < batch; head += 1) {
    for (let i = 0; i < stride; i += 1) {
      data[head * stride + i] = base(i + head * 37) * (1 + head * 0.37);
    }
  }
  return data;
};

type PreparedQk = {
  readonly q: Float32Array<ArrayBuffer>;
  readonly k: Float32Array<ArrayBuffer>;
  readonly scale: number;
  readonly expected: Float32Array<ArrayBuffer>;
  /** The margin of `x/s` from the rounding boundaries (the minimum over q / k). */
  readonly margin: number;
};

const prepareQk = (shape: QkShape): PreparedQk => {
  const { batch, m, n, d } = shape;
  const q = headVaried(QUERY, batch, m, d);
  const k = headVaried(KEY, batch, n, d);
  const scale = halfScale(d);
  const margin = Math.min(
    quantizeRowsTieMargin(q, batch * m, d),
    quantizeRowsTieMargin(k, batch * n, d),
  );
  const expected = referenceAttentionQkI8a8({ q, k, batch, m, n, d, scale });
  return { q, k, scale, expected, margin };
};

/**
 * Runs raw the same 3 dispatches as the executor's ①QK i8a8 (`quantize_rows` ×2 → i8a8 GEMM) and
 * **reads S back as is**. On the Session path S is an intermediate buffer and cannot be observed.
 */
const runAttentionQkI8a8 = async (
  gpu: GpuContext,
  shape: QkShape,
  prepared: PreparedQk,
  dot: I8a8Dot,
  geometry: I8a8Geometry = defaultI8a8Geometry("attention_qk"),
): Promise<Float32Array<ArrayBuffer>> => {
  const { batch, m, n, d } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const { pipeline: quantizePipeline, layout: quantizeLayout } = await cache.get(
      QUANTIZE_ROWS_KEY,
      QUANTIZE_ROWS_WGSL,
    );
    const quantize = (
      source: Float32Array<ArrayBuffer>,
      rows: number,
    ): { readonly payload: GPUBuffer; readonly scales: GPUBuffer } => {
      const params = arena.allocHostWritten(16, UNIFORM_IN);
      gpu.device.queue.writeBuffer(params, 0, quantizeRowsParams(rows, d));
      const src = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(src, 0, source);
      const payload = arena.allocRegion(Math.max(4, rows * d));
      const scales = arena.allocRegion(Math.max(4, rows * 4));
      const bindGroup = gpu.device.createBindGroup({
        layout: quantizeLayout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: payload } },
          { binding: 3, resource: { buffer: scales } },
        ],
      });
      scheduler.dispatch(
        quantizePipeline,
        bindGroup,
        [gridStrideWorkgroups(rows, 1, limit), 1, 1],
        QUANTIZE_ROWS_KEY,
      );
      return { payload, scales };
    };
    const query = quantize(prepared.q, batch * m);
    const key = quantize(prepared.k, batch * n);

    const v4 = attentionQkI8a8UsesVec4(n);
    const dp4a = dot === "dp4a";
    const pipelineKey = attentionQkI8a8Key(v4, dp4a, "f32", geometry);
    const { pipeline, layout } = await cache.get(
      pipelineKey,
      attentionQkI8a8Wgsl(v4, dp4a, "f32", geometry),
    );
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionQkI8a8Params(m, n, d, prepared.scale));
    const scores = arena.allocRegion(Math.max(4, batch * m * n * 4));
    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: query.payload } },
        { binding: 2, resource: { buffer: key.payload } },
        { binding: 3, resource: { buffer: scores } },
        { binding: ATTENTION_QK_Q_SCALE_BINDING, resource: { buffer: query.scales } },
        { binding: ATTENTION_QK_K_SCALE_BINDING, resource: { buffer: key.scales } },
      ],
    });
    // MUST: derive the dispatch edges from **this kernel's geometry** (the same one passed to the
    // key and the generated code). Passing a constant disagrees with the actual tile edge, and on
    // the smaller side tiles go missing as a silent wrong value.
    scheduler.dispatch(
      pipeline,
      bindGroup,
      [
        tiledWorkgroups(n, i8a8TileN(geometry), limit, shape.name),
        tiledWorkgroups(m, i8a8TileM(geometry), limit, shape.name),
        tiledWorkgroups(batch, 1, limit, shape.name),
      ],
      pipelineKey,
    );
    await scheduler.flush();
    const bytes = await readbackBytes(gpu.device, scores, batch * m * n * 4);
    return new Float32Array(bytes.buffer, 0, batch * m * n);
  } finally {
    await arena.destroy();
  }
};

test("attention_qk i8a8: q / k of the shapes stay well away from the rounding boundaries (premise of atol=0)", () => {
  // The GPU division is allowed up to 2.5 ULP, so an element whose `x/s` is near a half-integer can
  // wobble by ±1 step. Unless the margin is **measured every time** and made a check, atol=0
  // "merely happens to pass".
  let worst = Number.POSITIVE_INFINITY;
  for (const shape of QK_SHAPES) {
    const { margin } = prepareQk(shape);
    assert(
      margin > TIE_MARGIN,
      `${shape.name}: the margin from the rounding boundaries is only ${margin}`,
    );
    worst = Math.min(worst, margin);
  }
  // The worst margin over all shapes (changing the data sequences moves this value = the check
  // tells you)
  assert(worst > TIE_MARGIN, `worst margin ${worst}`);
});

test.skipIf(!GPU_AVAILABLE)(
  "S of attention_qk i8a8 matches the TS reference at atol=0 (v4 / scalar / tile edge / K remainder / B·H≥2, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of QK_SHAPES) {
        const prepared = prepareQk(shape);
        const actual = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
        assertExact(actual, prepared.expected, shape.name);
        // Tautology guard: if S is constant, "equal" verifies nothing
        assert(new Set([...actual]).size > 1, `${shape.name}: S is constant`);
        // The statistics differ per head (the very premise for the batch-base detector to hold)
        if (shape.batch > 1) {
          const head0 = actual.slice(0, shape.m * shape.n);
          const head1 = actual.slice(shape.m * shape.n, 2 * shape.m * shape.n);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${shape.name}: S of head 0 and head 1 is identical (a mixed-up base would not show in the values)`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "attention_qk i8a8: S matches the TS reference at atol=0 even on geometries with 5 or more K-side fill slots (real GPU)",
  async () => {
    // Detector of the defect (2026-09-27) where the fill's value variable shadowed the
    // function-scope `k4` (the K pack count). On geometries where the K-side slots one thread fills
    // = regN · tileK / 4 / wgY is 5 or more, the 5th and later columns silently became 0. The
    // default geometry (2 slots) cannot hit it, so 8 / 16-slot geometries are used.
    const geometries: readonly I8a8Geometry[] = [
      { regM: 4, regN: 4, wgX: 8, wgY: 4, tileK: 32 },
      { regM: 8, regN: 8, wgX: 8, wgY: 4, tileK: 32 },
    ];
    for (const geometry of geometries) {
      const slots = (geometry.regN * (geometry.tileK / 4)) / geometry.wgY;
      assert(
        slots >= 5,
        `${i8a8GeometryKeyPart(geometry, false)}: ${slots} fill slots do not hit the defect`,
      );
    }
    const gpu = await acquireGpu();
    try {
      for (const geometry of geometries) {
        for (const dot of ["dp4a", "emu"] as const) {
          for (const shape of QK_SHAPES) {
            const prepared = prepareQk(shape);
            const actual = await runAttentionQkI8a8(gpu, shape, prepared, dot, geometry);
            assertExact(
              actual,
              prepared.expected,
              `${shape.name} / ${i8a8GeometryKeyPart(geometry, false)} / ${dot}`,
            );
            assert(new Set([...actual]).size > 1, `${shape.name}: S is constant`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "attention_qk i8a8: a NaN row propagates to the whole row, a NaN column to the whole column, and a zero row is exactly 0 (real GPU)",
  async () => {
    const shape: QkShape = { name: "nan/zero B·H2 M8 N12 D20", batch: 2, m: 8, n: 12, d: 20 };
    const { batch, m, n, d } = shape;
    const q = headVaried(QUERY, batch, m, d);
    const k = headVaried(KEY, batch, n, d);
    // NaN in q row 2 of head 1, q row 5 all zero (tiny floor), NaN in k row (= output column) 3 of
    // head 0
    const nanRow = 2;
    const zeroRow = 5;
    const nanCol = 3;
    q[(1 * m + nanRow) * d + d - 3] = Number.NaN;
    for (let i = 0; i < d; i += 1) q[(1 * m + zeroRow) * d + i] = 0;
    // oxlint-disable-next-line oxc/erasing-op -- `0 * n` names head 0 in the (head * n + row) index
    k[(0 * n + nanCol) * d + 1] = Number.NaN;
    const scale = halfScale(d);
    const expected = referenceAttentionQkI8a8({ q, k, batch, m, n, d, scale });
    // quantizeRowsTieMargin skips the ratios of non-finite rows, so the margin check still applies
    const margin = Math.min(
      quantizeRowsTieMargin(q, batch * m, d),
      quantizeRowsTieMargin(k, batch * n, d),
    );
    assert(
      margin > TIE_MARGIN,
      `${shape.name}: the margin from the rounding boundaries is only ${margin}`,
    );
    const prepared: PreparedQk = { q, k, scale, expected, margin };

    const gpu = await acquireGpu();
    let actual: Float32Array<ArrayBuffer>;
    try {
      actual = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
    } finally {
      gpu.destroy();
    }
    assertExact(actual, expected, shape.name);
    // A NaN row is NaN in **every column** (the scale becomes NaN and dequant spreads it over the
    // whole row)
    for (let col = 0; col < n; col += 1) {
      assert(Number.isNaN(actual[(1 * m + nanRow) * n + col]), `NaN row col=${col}`);
    }
    // A NaN column is NaN in **every row** (propagation on the column-scale side — rows and columns
    // take different paths)
    for (let row = 0; row < m; row += 1) {
      // oxlint-disable-next-line oxc/erasing-op -- `0 * m` names head 0 in the (head * m + row) index
      assert(Number.isNaN(actual[(0 * m + row) * n + nanCol]), `NaN column row=${row}`);
    }
    // The neighbouring head is intact (q's NaN row does not cross heads — the NaN column is on head
    // 0's side by design)
    for (let col = 0; col < n; col += 1) {
      if (col === nanCol) continue;
      // oxlint-disable-next-line oxc/erasing-op -- `0 * m` names head 0 in the (head * m + row) index
      assert(Number.isFinite(actual[(0 * m + nanRow) * n + col]), `adjacent head col=${col}`);
    }
    // An all-zero row has q = 0, s = tiny, so S is exactly 0 too (±0 treated as equal). The NaN
    // column is on head 0's side, so this row of head 1 is finite and 0 in every column
    for (let col = 0; col < n; col += 1) {
      assertEquals(actual[(1 * m + zeroRow) * n + col], 0, `zero row col=${col}`);
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "attention_qk i8a8: the dot4I8Packed version and the emulated version match at atol=0 (the extension only affects speed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // This machine enumerates packed_4x8_integer_dot_product, so **both variants can actually
      // run**. This one test is the attention-side mechanical detector of the claim "the emulation
      // is numerically identical".
      for (const shape of QK_SHAPES) {
        const prepared = prepareQk(shape);
        const dp4a = await runAttentionQkI8a8(gpu, shape, prepared, "dp4a");
        const emu = await runAttentionQkI8a8(gpu, shape, prepared, "emu");
        assertExact(emu, dp4a, `${shape.name}: dp4a vs emulation`);
        // Both match the TS reference (closes off the case where both are off by the same amount)
        assertExact(emu, prepared.expected, `${shape.name}: emulation vs reference`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Session path (attentionCompute: "a8") — per-stage eligibility and mixing
// ---------------------------------------------------------------------------

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  /** The pipeline keys that ran (census — independent of timing). */
  readonly entries: readonly { readonly key: string; readonly dispatchCount: number }[];
};

const runAttention = async (
  gpu: GpuContext,
  shape: { readonly b: number; readonly h: number; readonly m: number; readonly n: number },
  d: number,
  options: SessionOptions,
): Promise<RunResult> => {
  const { b, h, m, n } = shape;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const graph = singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
    attrs: { scale: halfScale(d) },
  });
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(graph),
    GRAPH_NAME,
    options,
  );
  try {
    const outputs = await session.run({ x0: q, x1: k, x2: v });
    return {
      output: (outputs["y"] as Tensor).data as Float32Array<ArrayBuffer>,
      entries: requireCensus(session.diagnostics().lastRunPipelines, "attention"),
    };
  } finally {
    await session.dispose();
  }
};

const relativeRms = (actual: ArrayLike<number>, expected: ArrayLike<number>): number => {
  let diff = 0;
  let norm = 0;
  for (let i = 0; i < actual.length; i += 1) {
    diff += (actual[i] - expected[i]) ** 2;
    norm += expected[i] ** 2;
  }
  return Math.sqrt(diff / norm);
};

test.skipIf(!GPU_AVAILABLE)(
  "attentionCompute:'a8' makes ①QK and ③PV integer dot products while ② row statistics runs in f32 (real GPU)",
  async () => {
    // MUST: run on **the default device that does not request shader-f16** (i8a8 is outside the
    // feature gate).
    const gpu = await acquireGpu();
    try {
      const shape = { b: 2, h: 3, m: 65, n: 68 };
      const d = 20;
      const i8a8 = await runAttention(gpu, shape, d, { attentionCompute: "a8" });
      const plain = await runAttention(gpu, shape, d, {});
      assert(
        [...i8a8.output].some((value, index) => value !== plain.output[index]),
        "the i8a8 output does not differ by a single bit from the f32 path (the opt-in has no effect)",
      );
      assert(
        new Set([...i8a8.output]).size > 1,
        "the output is constant (the comparison is a tautology)",
      );
      // A coarse check that the whole path is wired correctly (silent wrong values such as a
      // mixed-up scale axis of V miss by orders of magnitude. The numeric contract itself is held
      // at atol=0 by the raw dispatches above). Measured 6.09e-2 (deterministic data, so the value
      // does not move) against a check of 1e-1. This shape has a wide range of S and a sharp
      // softmax, so the peak/rms of P̃ is large = the error is large too, as design §2.2's theory
      // says
      const rms = relativeRms(i8a8.output, plain.output);
      assert(rms < 0.1, `the relative RMS of i8a8 vs f32 is ${rms} (suspect the wiring)`);

      // The knob of the integer dot-product variant affects **both stages** of attention (the same
      // single knob as linear)
      const emu = await runAttention(gpu, shape, d, {
        attentionCompute: "a8",
        [I8A8_DOT]: "emu",
      });
      assertExact(emu.output, i8a8.output, "dp4a vs emulation (Session path)");
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). Which variant actually ran is visible only from keys (both the
 * i8a8 degradation and dp4a ↔ emulation keep the values bit-identical and only change the path).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too). An empty census FAILs unconditionally (guarding with "if entries is empty, look at
 * nothing" turns green without a single check running — `requireCensus` in
 * tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "the variant choice of attentionCompute:'a8' (①③ i8a8 / ② row statistics f32 / dp4a ↔ emulation) is checked through keys (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const shape = { b: 2, h: 3, m: 65, n: 68 };
      const d = 20;
      const i8a8 = await runAttention(gpu, shape, d, { attentionCompute: "a8" });
      assert(i8a8.entries.length > 0, "the breakdown is empty (the key check is a no-op)");
      const byKey = new Map(i8a8.entries.map((entry) => [entry.key, entry.dispatchCount]));
      // ①QK / ③PV are both the i8a8 variant, and not a single f32 variant runs
      assertEquals(
        byKey.get(
          attentionQkI8a8Key(
            attentionQkI8a8UsesVec4(shape.n),
            true,
            "f32",
            profile.i8a8.attentionQk,
          ),
        ),
        1,
      );
      assertEquals(
        byKey.get(
          attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), true, "f32", profile.i8a8.attentionPv),
        ),
        1,
      );
      for (const v4 of [false, true]) {
        assertEquals(
          byKey.has(attentionQkKey(v4, "f32", "f32", false, false, false, profile.attention.qk)),
          false,
          "the f32 attention_qk is still there",
        );
        assertEquals(
          byKey.has(attentionPvKey(v4, "f32", "f32", false, false, profile.attention.pv)),
          false,
          "the f32 attention_pv is still there",
        );
      }
      // ② row statistics stays f32 (the storage form of S stays f32 too — design §4.3's denominator
      // quantization was not adopted).
      // regcache (reading S once) is generated per dim, so epc appears in the key (the values are
      // bit-identical)
      assertEquals(
        byKey.get(attentionStatsKey("f32", "f32", attentionStatsRegCache(shape.n))),
        1,
        "② row statistics stays f32",
      );
      // Quantization runs 3 times for q / k / Vᵀ, the permute once for Vᵀ (7 dispatches for the
      // whole node)
      assertEquals(quantizeRowsDispatches(byKey), 3, "quantize_rows is not the 3 of q / k / v");
      assertEquals(
        byKey.get(stridedKey({ dtype: "f32" })),
        1,
        "the permute of Vᵀ is not exactly 1",
      );
      assertEquals(
        [...byKey.values()].reduce((sum, count) => sum + count, 0),
        7,
        "1 node = 7 dispatches",
      );

      const emu = await runAttention(gpu, shape, d, {
        attentionCompute: "a8",
        [I8A8_DOT]: "emu",
      });
      assert(
        emu.entries.length > 0,
        "the emulated side's breakdown is empty (the key check is a no-op)",
      );
      const keys = new Set(emu.entries.map((entry) => entry.key));
      assertEquals(
        keys.has(
          attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), false, "f32", profile.i8a8.attentionPv),
        ),
        true,
      );
      assertEquals(
        keys.has(
          attentionPvI8a8Key(attentionPvI8a8UsesVec4(d), true, "f32", profile.i8a8.attentionPv),
        ),
        false,
      );
      // ①QK follows the same knob
      const qkV4 = attentionQkI8a8UsesVec4(shape.n);
      assertEquals(
        keys.has(attentionQkI8a8Key(qkV4, false, "f32", profile.i8a8.attentionQk)),
        true,
        "key of the emulated variant",
      );
      assertEquals(
        keys.has(attentionQkI8a8Key(qkV4, true, "f32", profile.i8a8.attentionQk)),
        false,
        "the dp4a variant is still there",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the degradation of both stages under attentionCompute:'a8' (D%4 / N%4 both ineligible) is bit-identical to the f32 path (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      // The degradation is silent (the detectors are 2: keys and values — the key side is held by
      // the census below)
      const both = { b: 1, h: 2, m: 17, n: 19 };
      const degraded = await runAttention(gpu, both, 13, { attentionCompute: "a8" });
      const plain = await runAttention(gpu, both, 13, {});
      assertExact(
        degraded.output,
        plain.output,
        "the degradation of both stages does not match the f32 path",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Independent per-stage eligibility (mixing) does not show in the values, so keys are the only
 * detector. The reason to fail an empty census is the same as the census above.
 */
test.skipIf(!GPU_AVAILABLE)(
  "eligibility under attentionCompute:'a8' is independent per stage, producing a mix where only one stage degrades to f32 (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      // ① N % 4 != 0 → only ③PV is f32 (①QK stays i8a8 with D=20)
      const mixedPv = { b: 1, h: 2, m: 17, n: 19 };
      const pvF32 = await runAttention(gpu, mixedPv, 20, { attentionCompute: "a8" });
      assert(pvF32.entries.length > 0, "the breakdown is empty (the key check is a no-op)");
      const pvKeys = new Map(pvF32.entries.map((entry) => [entry.key, entry.dispatchCount]));
      assertEquals(
        pvKeys.get(
          attentionQkI8a8Key(attentionQkI8a8UsesVec4(19), true, "f32", profile.i8a8.attentionQk),
        ),
        1,
      );
      assertEquals(
        pvKeys.get(
          attentionPvKey(gemmUsesVec4(19, 20), "f32", "f32", false, false, profile.attention.pv),
        ),
        1,
        "③PV degrades to f32",
      );
      for (const v4 of [false, true]) {
        assertEquals(
          pvKeys.has(attentionPvI8a8Key(v4, true, "f32", profile.i8a8.attentionPv)),
          false,
        );
      }
      // Quantization runs only twice for q / k, and the permute of Vᵀ does not run
      assertEquals(quantizeRowsDispatches(pvKeys), 2, "quantize_rows is not the 2 of q / k");
      assertEquals(pvKeys.has(stridedKey({ dtype: "f32" })), false, "the permute of Vᵀ runs");

      // ② D % 4 != 0 → only ①QK is f32 (③PV stays i8a8 with N=20) — the mix in the opposite
      // direction
      const mixedQk = { b: 1, h: 2, m: 17, n: 20 };
      const qkF32 = await runAttention(gpu, mixedQk, 13, { attentionCompute: "a8" });
      assert(qkF32.entries.length > 0, "the breakdown is empty (the key check is a no-op)");
      const qkKeys = new Map(qkF32.entries.map((entry) => [entry.key, entry.dispatchCount]));
      assertEquals(
        qkKeys.get(
          attentionQkKey(
            gemmUsesVec4(13, 20),
            "f32",
            "f32",
            false,
            false,
            false,
            profile.attention.qk,
          ),
        ),
        1,
        "①QK degrades to f32",
      );
      assertEquals(
        qkKeys.get(
          attentionPvI8a8Key(attentionPvI8a8UsesVec4(13), true, "f32", profile.i8a8.attentionPv),
        ),
        1,
      );
      for (const v4 of [false, true]) {
        assertEquals(
          qkKeys.has(attentionQkI8a8Key(v4, true, "f32", profile.i8a8.attentionQk)),
          false,
        );
      }
      // Quantization runs only once for Vᵀ (q / k are read as f32)
      assertEquals(quantizeRowsDispatches(qkKeys), 1, "quantize_rows is not the 1 of Vᵀ");
      assertEquals(
        qkKeys.get(stridedKey({ dtype: "f32" })),
        1,
        "the permute of Vᵀ is not exactly 1",
      );

      // ③ A shape meeting neither runs neither the prologue nor the epilogue (the values side is
      // held by the test above)
      const both = { b: 1, h: 2, m: 17, n: 19 };
      const degraded = await runAttention(gpu, both, 13, { attentionCompute: "a8" });
      assert(degraded.entries.length > 0, "the breakdown is empty (the key check is a no-op)");
      const keys = new Set(degraded.entries.map((entry) => entry.key));
      assertEquals(
        [...keys].some((key) => key.startsWith(QUANTIZE_ROWS_KEY)),
        false,
        "quantization runs despite the degradation",
      );
      assertEquals(
        keys.has(stridedKey({ dtype: "f32" })),
        false,
        "the permute runs despite the degradation",
      );
      assertEquals(
        keys.has(attentionQkKey(false, "f32", "f32", false, false, false, profile.attention.qk)),
        true,
        "the f32 attention_qk did not run",
      );
      for (const dp4a of [false, true]) {
        assertEquals(
          keys.has(attentionQkI8a8Key(false, dp4a, "f32", profile.i8a8.attentionQk)),
          false,
          "i8a8 runs despite the degradation",
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "attention_pv i8a8: fails loudly when N exceeds the i32 reduction limit (2^17) (no silent rollback, real GPU)",
  async () => {
    // Just outside the limit (2^17 + 4, a multiple of 4). |acc| ≤ N·127² fits in i32, but the limit
    // errs on the safe side.
    const n = LINEAR_I8A8_MAX_K + 4;
    const shape = { b: 1, h: 1, m: 1, n };
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => runAttention(gpu, shape, 4, { attentionCompute: "a8" }),
        ExecutionError,
        "i32 reduction guard",
      );
      // On the default f32 path the same shape runs normally (the limit belongs only to the i8a8
      // path)
      const baseline = await runAttention(gpu, shape, 4, {});
      assertEquals(baseline.output.length, 4);
    } finally {
      gpu.destroy();
    }
  },
);

// ①QK has its own D gate, a separate branch from the ③PV N gate above.
test.skipIf(!GPU_AVAILABLE)(
  "attention i8a8: fails loudly when D exceeds the i32 reduction limit (2^17) (no silent rollback, real GPU)",
  async () => {
    // Just outside the limit (2^17 + 4, a multiple of 4). |acc| ≤ D·127² fits in i32, but the limit
    // errs on the safe side.
    const d = LINEAR_I8A8_MAX_K + 4;
    const shape = { b: 1, h: 1, m: 1, n: 4 };
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => runAttention(gpu, shape, d, { attentionCompute: "a8" }),
        ExecutionError,
        "i32 reduction guard",
      );
      // On the default f32 path the same shape runs normally (the limit belongs only to the i8a8
      // path)
      const baseline = await runAttention(gpu, shape, d, {});
      assertEquals(baseline.output.length, d);
    } finally {
      gpu.destroy();
    }
  },
);
