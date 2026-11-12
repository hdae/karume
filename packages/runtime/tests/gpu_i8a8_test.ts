// Execution path of w8a8 / w4a8 linear (integer dot products of per-token i8 activations × i8
// per-channel / i4 group weights).
//
// Design = docs/research/2026-08-03-dp4a-w8a8-design.md (w4a8: perf-ledger Q-8), implementation =
// src/kernels/quantize-rows.ts + src/kernels/linear-i8a8.ts, opt-in =
// `SessionOptions.linearCompute: "a8"` (default "f32"; the weight storage is a separate axis).
//
// ## The numeric contract this file pins
//
// **The GPU and the TS reference match with atol=0.** This contract is of a different kind from the
// other kernels' (where the f32 reduction order varies by implementation), and it rests on only two
// points: "the dot product is an exact i32 sum (order independent)" + "there are only 3 floating
// point operations per output element (f32(acc) / xs·wscale / fma)". So however the tile split,
// the zero padding of the K remainder or the shared-memory layout changes, the values should not
// move by a single bit, and if they move it is an implementation error — there is structurally no
// room for a tolerance to absorb.
// w4a8 has the same property on a shifted basis (only the inside of a group is exact i32, and there
// are `k/g + 1` floating point operations) — see each section's docstring.
//
// MUST: compare with {@link assertExact} (an exhaustive `===` comparison), not `compareTensors`.
// allclose fails every non-finite value, so it cannot check the propagation of NaN rows (part of
// the design).
//
// NOTE: the known-value checks of the reference oracle itself (src/reference/i8a8.ts) are in
// tests/i8a8_reference_test.ts, **CPU only**. The comparison here lets through "the reference and
// the GPU broke in the same direction", so the reference's properties sit on the side that runs
// without an adapter (if that fails, the comparison has lost its footing).
//
// ## Handling of non-finite values (outside the contract)
//
// A NaN row becomes **NaN across the whole row** through `xs[row] = NaN` (the same granularity as
// the f32 path). An Inf row gives `xs[row] = Inf`, and the output is some of ±Inf / NaN — **the
// sign does not match the f32 path** (recorded in docs/limitations.md). The quantized values `xq`
// themselves are **outside the contract** when the row scale is non-finite, since
// `vec4<i32>(NaN)` is an indeterminate value (the comparison covers only the scale and the final
// output).

import { test } from "vite-plus/test";
import {
  assert,
  assertEquals,
  assertRejects,
  assertStringIncludes,
  assertThrows,
} from "@std/assert";
import { gridStrideWorkgroups } from "../src/codegen/dispatch.ts";
import { perChannelGroupSize } from "../src/format/container/codecs.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  type QuantizeRowsGeometry,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsPaddedGeometry,
  quantizeRowsPaddedKey,
  quantizeRowsPaddedParams,
  quantizeRowsPaddedWgsl,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import {
  dp4aAvailable,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8UsesVec4,
} from "../src/kernels/linear-i8a8.ts";
import {
  quantizeRowsReference,
  quantizeRowsTieMargin,
  referenceLinearI8a8,
  referenceLinearW4a8,
  roundTiesToEven,
} from "../src/reference/i8a8.ts";
import {
  createSessionFromContainer,
  I8A8_DOT,
  type I8a8Dot,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { gemmRowsGeometry, i8a8LinearGeometry } from "../src/kernels/geometry-profile.ts";
import { linearI8a8Geometry } from "../src/runtime/recipe-builders/linear.ts";
import { linearKey } from "../src/kernels/linear.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * Deterministic data sequences (no random numbers — a failure would not reproduce).
 *
 * MUST: the activation step **must not be commensurate with the scale grid**. A plain sequence such
 * as steps of `0.75` puts `x/s` exactly on half-integers (e.g. 127·3/6 = 63.5), and since GPU
 * division is allowed up to 2.5 ULP, the quantized value wobbles by ±1 step and atol=0 "fails by
 * chance". The margin from the boundaries is measured every time with
 * {@link quantizeRowsTieMargin} and made a check (if this sequence changes, the check says so).
 */
const SIGNED = (i: number): number => ((i % 29) - 14) * 0.3717 + 0.0131;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/**
 * The margin from the rounding boundaries at which atol=0 may be claimed (over 50 times 2.5 ULP ≈
 * 1.9e-5).
 */
const TIE_MARGIN = 1e-3;

/**
 * Exhaustive atol=0 comparison. `===` treats ±0 as equal, but otherwise not even a 1-bit difference
 * is allowed. Where the reference is NaN it only checks that "the measured value is NaN too" (the
 * identity of the value is outside the contract).
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
// Rounding tie rule (the GPU-independent side)
// ---------------------------------------------------------------------------

test("roundTiesToEven rounds half to even (pins the points where it differs from Math.round's half-up)", () => {
  const cases: readonly (readonly [number, number])[] = [
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [3.5, 4],
    [-0.5, 0],
    [-1.5, -2],
    [-2.5, -2],
    [0.49999997, 0],
    [1.4999999, 1],
    [126.5, 126],
    [127.5, 128],
  ];
  for (const [value, expected] of cases) {
    assertEquals(roundTiesToEven(value), expected, `round(${value})`);
  }
  // MUST: Math.round cannot stand in (on a tie it goes toward +∞)
  assertEquals(Math.round(0.5), 1);
  assertEquals(Math.round(2.5), 3);
  // Non-finite values pass through as is (clamp takes them and pushes them to ±127)
  assertEquals(roundTiesToEven(Number.POSITIVE_INFINITY), Number.POSITIVE_INFINITY);
  assertEquals(Number.isNaN(roundTiesToEven(Number.NaN)), true);
});

// ---------------------------------------------------------------------------
// quantize_rows alone (raw dispatch)
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

type QuantizeResult = {
  /** i8 values unpacked from the 4-per-word packing (flat `[rows, dim]`). */
  readonly q: Int8Array<ArrayBuffer>;
  readonly scale: Float32Array<ArrayBuffer>;
};

/**
 * Runs `quantize_rows` as a single dispatch (narrowing `groups` also exercises the reduced path).
 */
const runQuantizeRows = async (
  gpu: GpuContext,
  x: Float32Array<ArrayBuffer>,
  rows: number,
  dim: number,
  groups?: number,
  // The default is the original form (1 row = 1 workgroup). The small-D variant's check passes the
  // geometry explicitly.
  geometry: QuantizeRowsGeometry = { lanesPerRow: 256, rowsPerGroup: 1 },
): Promise<QuantizeResult> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const key = quantizeRowsKey(geometry);
  try {
    const { pipeline, layout } = await cache.get(key, quantizeRowsWgsl(geometry));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, quantizeRowsParams(rows, dim));
    const src = arena.allocHostWritten(Math.max(4, x.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(src, 0, x);
    const xq = arena.allocRegion(Math.max(4, rows * dim));
    const xs = arena.allocRegion(Math.max(4, rows * 4));
    const bindGroup = gpu.device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: xq } },
        { binding: 3, resource: { buffer: xs } },
      ],
    });
    scheduler.dispatch(
      pipeline,
      bindGroup,
      [
        groups ??
          gridStrideWorkgroups(
            rows,
            geometry.rowsPerGroup,
            gpu.limits.maxComputeWorkgroupsPerDimension,
          ),
        1,
        1,
      ],
      key,
    );
    await scheduler.flush();
    const qBytes = await readbackBytes(gpu.device, xq, rows * dim);
    const scaleBytes = await readbackBytes(gpu.device, xs, rows * 4);
    return {
      q: new Int8Array(qBytes.buffer, 0, rows * dim),
      scale: new Float32Array(scaleBytes.buffer, 0, rows),
    };
  } finally {
    await arena.destroy();
  }
};

/** One `quantize_rows` launch of any form, read back as raw words (`xq`) and scale bits (`xs`). */
const runQuantizeLaunch = async (
  gpu: GpuContext,
  launch: {
    readonly key: string;
    readonly wgsl: string;
    readonly params: Uint32Array<ArrayBuffer>;
    readonly rowsPerGroup: number;
  },
  x: Float32Array<ArrayBuffer>,
  rows: number,
  wordsPerRow: number,
  groups?: number,
): Promise<{
  readonly words: Uint32Array<ArrayBuffer>;
  readonly scaleBits: Uint32Array<ArrayBuffer>;
}> => {
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  try {
    const { pipeline, layout } = await cache.get(launch.key, launch.wgsl);
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, launch.params);
    const src = arena.allocHostWritten(Math.max(4, x.byteLength), STORAGE_IN);
    gpu.device.queue.writeBuffer(src, 0, x);
    const xq = arena.allocRegion(rows * wordsPerRow * 4);
    const xs = arena.allocRegion(rows * 4);
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: src } },
          { binding: 2, resource: { buffer: xq } },
          { binding: 3, resource: { buffer: xs } },
        ],
      }),
      [
        groups ??
          gridStrideWorkgroups(
            rows,
            launch.rowsPerGroup,
            gpu.limits.maxComputeWorkgroupsPerDimension,
          ),
        1,
        1,
      ],
      launch.key,
    );
    await scheduler.flush();
    const words = await readbackBytes(gpu.device, xq, rows * wordsPerRow * 4);
    const scales = await readbackBytes(gpu.device, xs, rows * 4);
    return {
      words: new Uint32Array(words.buffer, 0, rows * wordsPerRow),
      scaleBits: new Uint32Array(scales.buffer, 0, rows),
    };
  } finally {
    await arena.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "quantize_rows padded form (ADR 0132 addendum 2026-10-27): word-for-word the plain form on zero-extended rows",
  async () => {
    // Extending a row of `dim` elements with zeros to 4·ceil(dim/4) changes neither its amax (|0|
    // is the identity of the max) nor the lattice of its elements, and the zeros quantise to the
    // byte 0. So the padded form over the original rows must write exactly the words and scale bits
    // of the plain form over the extended rows — a GPU-to-GPU comparison at atol 0 (the same
    // division on both sides). The padding bytes are then 0 by the word equality; the loop below
    // states it on its own.
    const gpu = await acquireGpu();
    try {
      for (const dim of [5, 65, 125, 203, 4106]) {
        const words = Math.ceil(dim / 4);
        const extended = words * 4;
        const rows = 1_003;
        const x = new Float32Array(rows * dim);
        const xExtended = new Float32Array(rows * extended);
        for (let row = 0; row < rows; row += 1) {
          for (let i = 0; i < dim; i += 1) {
            const value =
              row % 97 === 3
                ? 0
                : SIGNED(row * dim + i) * (1 + (row % 11) * 0.75) * (row % 2 ? -1 : 1);
            x[row * dim + i] = value;
            xExtended[row * extended + i] = value;
          }
        }
        const plainGeometry = quantizeRowsGeometry(extended);
        const plain = await runQuantizeLaunch(
          gpu,
          {
            key: quantizeRowsKey(plainGeometry),
            wgsl: quantizeRowsWgsl(plainGeometry),
            params: quantizeRowsParams(rows, extended),
            rowsPerGroup: plainGeometry.rowsPerGroup,
          },
          xExtended,
          rows,
          words,
        );
        const geometry = quantizeRowsPaddedGeometry(dim);
        assertEquals(geometry, plainGeometry, `dim ${dim}: the geometry of the extended row`);
        for (const groups of [undefined, 2]) {
          const where = `dim ${dim} r${geometry.rowsPerGroup}w${geometry.lanesPerRow} groups ${groups}`;
          const padded = await runQuantizeLaunch(
            gpu,
            {
              key: quantizeRowsPaddedKey(geometry),
              wgsl: quantizeRowsPaddedWgsl(geometry),
              params: quantizeRowsPaddedParams(rows, dim),
              rowsPerGroup: geometry.rowsPerGroup,
            },
            x,
            rows,
            words,
            groups,
          );
          assertEquals(
            Array.from(padded.scaleBits),
            Array.from(plain.scaleBits),
            `${where}: scale bits`,
          );
          assertEquals(Array.from(padded.words), Array.from(plain.words), `${where}: words`);
          // The padding bytes of every row's last word are 0
          const padMask = (0xffffffff << (8 * (dim % 4))) >>> 0;
          for (let row = 0; row < rows; row += 1) {
            const last = padded.words[row * words + words - 1];
            if ((last & padMask) !== 0)
              throw new Error(`${where}: row ${row} padding bytes ${last.toString(16)}`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test("quantize_rows padded form rejects row lengths the plain form owns", () => {
  for (const dim of [0, 4, 128, 4096]) {
    assertThrows(() => quantizeRowsPaddedGeometry(dim), Error, "not a multiple of 4");
    assertThrows(() => quantizeRowsPaddedParams(1, dim), Error, "not a multiple of 4");
  }
});

test.skipIf(!GPU_AVAILABLE)(
  "quantize_rows: WGSL's round rounds half to even (3 tie points measured and pinned in the kernel, real GPU)",
  async () => {
    // With amax = 127, s = f32(127 · (1/127)) is exactly 1.0 and x/s is the input itself
    // (division by 1.0 has no room for 2.5 ULP and is **exact**, so the tie rule alone is
    // isolated).
    // Where the ties (±0.5 / ±1.5 / 2.5 / ±3.5) round to can be observed directly as quantized
    // values.
    const values = [127, 0.5, 1.5, -0.5, -1.5, 2.5, 3.5, -3.5];
    const x = Float32Array.from(values);
    const gpu = await acquireGpu();
    try {
      const actual = await runQuantizeRows(gpu, x, 1, values.length);
      assertEquals(
        actual.scale[0],
        1,
        "s is exactly 1.0 (a form where the ties sit on the grid boundaries)",
      );
      // Round half to even: 0.5→0 / 1.5→2 / -0.5→-0 / -1.5→-2 / 2.5→2 / 3.5→4 / -3.5→-4
      assertEquals([...actual.q], [127, 0, 2, 0, -2, 2, 4, -4]);
      // Also pins that it is the same sequence as the TS reference (roundTiesToEven)
      assertEquals([...actual.q], [...quantizeRowsReference(x, 1, values.length).q]);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "quantize_rows: the small-D variant is bit-identical to the original form in both q and scale (4 geometries × the reduced path, real GPU)",
  async () => {
    // The geometry follows from dim: 128 → 8 rows × 32 lanes / 64 → 16 × 16 / 20 → 32 × 8 /
    // 8 → 128 × 2.
    // Each row has a different magnitude (with the same scale in every row, the error "uses another
    // row's amax" does not show in the values).
    // The row count is not a multiple of rowsPerGroup, so the path of the trailing "group without
    // rows" is also exercised.
    const gpu = await acquireGpu();
    try {
      for (const dim of [128, 64, 20, 8]) {
        const geometry = quantizeRowsGeometry(dim);
        assertEquals(geometry.lanesPerRow * geometry.rowsPerGroup, 256);
        assertEquals(
          geometry.lanesPerRow >= dim / 4,
          true,
          `dim ${dim}: lane width below the quad count`,
        );
        const rows = 1_003;
        const x = new Float32Array(rows * dim);
        for (let row = 0; row < rows; row += 1) {
          for (let i = 0; i < dim; i += 1) {
            x[row * dim + i] = row % 97 === 3 ? 0 : SIGNED(row * dim + i) * (1 + (row % 11) * 0.75);
          }
        }
        const baseline = await runQuantizeRows(gpu, x, rows, dim);
        // Comparing variants with each other is GPU against GPU (the same division), so no tie
        // margin is needed. Only the comparison with the TS reference is limited to forms with a
        // margin (short rows have coarse ratios, and with this sequence dim 8 sits near a
        // boundary).
        if (quantizeRowsTieMargin(x, rows, dim) > TIE_MARGIN) {
          const reference = quantizeRowsReference(x, rows, dim);
          assertEquals([...baseline.q], [...reference.q], `dim ${dim}: original-form q`);
        }
        // Both the natural count and the reduced dispatch (grid-stride over 2 workgroups).
        for (const groups of [undefined, 2]) {
          const grouped = await runQuantizeRows(gpu, x, rows, dim, groups, geometry);
          const label = `dim ${dim} r${geometry.rowsPerGroup}w${geometry.lanesPerRow} groups ${groups}`;
          assertEquals([...grouped.q], [...baseline.q], `${label}: q`);
          assertEquals(
            Array.from(new Uint32Array(grouped.scale.buffer)),
            Array.from(new Uint32Array(baseline.scale.buffer)),
            `${label}: bits of the scale`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "quantize_rows matches the TS reference (exact recovery of the absolute maximum / all-zero rows / per-row scales / remainders, real GPU)",
  async () => {
    // dim is a multiple of 4 but **neither a multiple of 256 (the workgroup width) nor of 16**
    // (exercises both the path where 1 thread handles several quads and the path with threads that
    // handle none).
    const rows = 7;
    const dim = 20;
    const x = new Float32Array(rows * dim);
    for (let row = 0; row < rows; row += 1) {
      for (let i = 0; i < dim; i += 1) {
        // Each row has a different magnitude (independence of the per-row scales — if all rows were
        // the same, a row mix-up would not show in the values)
        x[row * dim + i] = row === 3 ? 0 : SIGNED(row * dim + i) * (1 + row * 3);
      }
    }
    // Checks that the data allows claiming atol=0 (WGSL division is allowed up to 2.5 ULP)
    assert(
      quantizeRowsTieMargin(x, rows, dim) > TIE_MARGIN,
      `the margin from the rounding boundaries is only ${quantizeRowsTieMargin(x, rows, dim)}`,
    );
    const gpu = await acquireGpu();
    let actual: QuantizeResult;
    try {
      actual = await runQuantizeRows(gpu, x, rows, dim);
    } finally {
      gpu.destroy();
    }
    const expected = quantizeRowsReference(x, rows, dim);
    assertExact(actual.scale, expected.scale, "quantize_rows scale");
    assertExact(actual.q, expected.q, "quantize_rows q");
    // Independence of the per-row scales (if they were all the same value, the checks below would
    // be vacuous)
    assertEquals(new Set([...actual.scale]).size >= rows - 1, true, "a different scale per row");
    // An all-zero row has s = f32 tiny and q = 0 (0 · tiny = 0 recovers it exactly)
    assertEquals(
      actual.scale[3],
      1.1754943508222875e-38,
      "the scale of an all-zero row is f32 tiny",
    );
    for (let i = 0; i < dim; i += 1)
      assertEquals(actual.q[3 * dim + i], 0, `q[${i}] of the all-zero row`);
    // The element with the largest absolute value lands exactly on ±127 (a direct consequence of
    // the rule not to use −128)
    for (let row = 0; row < rows; row += 1) {
      if (row === 3) continue;
      let amax = 0;
      for (let i = 0; i < dim; i += 1) amax = Math.max(amax, Math.abs(x[row * dim + i]));
      let sawExtreme = false;
      for (let i = 0; i < dim; i += 1) {
        if (Math.abs(x[row * dim + i]) === amax) {
          assertEquals(Math.abs(actual.q[row * dim + i]), 127, `absolute maximum of row ${row}`);
          sawExtreme = true;
        }
      }
      assert(sawExtreme, `row ${row} has no element with the largest absolute value`);
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "quantize_rows propagates NaN / Inf to the row scale (a plain max would lose them, real GPU)",
  async () => {
    // Row 0 = normal / row 1 = NaN at the end / row 2 = +Inf in the middle / row 3 = −Inf in the
    // middle
    const rows = 4;
    const dim = 8;
    const x = new Float32Array(rows * dim);
    for (let i = 0; i < rows * dim; i += 1) x[i] = SIGNED(i);
    x[1 * dim + dim - 1] = Number.NaN;
    x[2 * dim + 3] = Number.POSITIVE_INFINITY;
    x[3 * dim + 3] = Number.NEGATIVE_INFINITY;
    const gpu = await acquireGpu();
    let actual: QuantizeResult;
    try {
      actual = await runQuantizeRows(gpu, x, rows, dim);
    } finally {
      gpu.destroy();
    }
    const expected = quantizeRowsReference(x, rows, dim);
    // Exhaustive comparison including the non-finite rows (assertExact only checks "measured is NaN
    // too" where the reference is NaN)
    assertExact(actual.scale, expected.scale, "quantize_rows scale");
    assertEquals(Number.isFinite(actual.scale[0]), true, "the scale of the normal row is finite");
    // MUST: NaN goes to the row scale (a plain max swallows the NaN and gives a finite scale)
    assertEquals(Number.isNaN(actual.scale[1]), true, "the scale of the NaN row");
    assertEquals(actual.scale[2], Number.POSITIVE_INFINITY, "the scale of the +Inf row");
    // |−Inf| = +Inf, so the absolute maximum is +Inf (no sign remains in the scale)
    assertEquals(actual.scale[3], Number.POSITIVE_INFINITY, "the scale of the −Inf row");
    // The quantized values of the normal row match the reference (q of the non-finite rows is
    // outside the contract — top of the file)
    assertExact(actual.q.slice(0, dim), expected.q.slice(0, dim), "quantize_rows q (normal row)");
  },
);

// ---------------------------------------------------------------------------
// Session path (linearCompute: "a8")
// ---------------------------------------------------------------------------

type LinearCase = {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  /** SIGNED by default. Replaced only for the NaN / zero-row cases. */
  readonly x?: (index: number, m: number, k: number) => number;
};

/** Declaration and supply of one `linear(x, w, b)` (w is i8 + per-channel scale). */
const i8LinearParts = (
  m: number,
  n: number,
  k: number,
  quantized: ReturnType<typeof quantizeI8>,
  bias: FilledTensor,
): { readonly declaration: DeclarationJson; readonly tensors: readonly TensorInput[] } => ({
  declaration: {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  },
  tensors: [
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
      // The per-channel groupSize is looked up in the ledger (`codecs.ts`) (row length k; 1 for the
      // degenerate k = 0).
      encoding: {
        codec: "int8-sym",
        groupSize: perChannelGroupSize(k),
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    },
  ],
});

const i8LinearModel = (
  m: number,
  n: number,
  k: number,
  quantized: ReturnType<typeof quantizeI8>,
  bias: FilledTensor,
): Promise<OpenedContainer> => {
  const { declaration, tensors } = i8LinearParts(m, n, k, quantized, bias);
  return openModelBytes(declaration, tensors);
};

type PreparedLinear = {
  readonly model: Promise<BoundContainer>;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
};

const prepareLinear = (testCase: LinearCase): PreparedLinear => {
  const { m, n, k } = testCase;
  // Weights whose magnitude differs per row (= output channel). With the same scale in every row,
  // an axis mix-up does not show in the values.
  const weight = fill([n, k], (i) => POSITIVE(i) * (1 + (Math.floor(i / k) % 7) * 0.25));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], SIGNED);
  const x = fill([m, k], (i) => (testCase.x ?? ((index) => SIGNED(index)))(i, m, k));
  // Checks that the data allows claiming atol=0 (the MUST of {@link SIGNED} at the top of the file)
  const margin = quantizeRowsTieMargin(x.data, m, k);
  if (!(margin > TIE_MARGIN)) {
    throw new Error(`${testCase.name}: the margin from the rounding boundaries is only ${margin}`);
  }
  const expected = referenceLinearI8a8({
    x: x.data,
    weight: new Int8Array(quantized.bytes.buffer, quantized.bytes.byteOffset, n * k),
    weightScale: quantized.scale,
    bias: bias.data,
    m,
    n,
    k,
  });
  return { model: i8LinearModel(m, n, k, quantized, bias), x, expected };
};

const runLinear = async (
  gpu: GpuContext,
  prepared: PreparedLinear,
  options: SessionOptions,
): Promise<{ readonly y: Tensor; readonly pipelineCount: number; readonly keys: string[] }> => {
  const session = await createSessionFromContainer(gpu, await prepared.model, GRAPH_NAME, options);
  try {
    const outputs = await session.run({ x: prepared.x });
    const diagnostics = session.diagnostics();
    return {
      y: outputs["y"],
      pipelineCount: diagnostics.pipelineCount,
      keys: requireCensus(diagnostics.lastRunPipelines, "linear")
        .map((row) => row.key)
        .sort(),
    };
  } finally {
    await session.dispose();
  }
};

/**
 * How the shapes are split up.
 *
 * MUST: **keep the v4 path and the scalar path as a pair** (the write guard is a different
 * expression per variant).
 * MUST: **m / n cross the tile edge 64** (collapsed below 1 tile, the guard is never false).
 * MUST: include **k % 16 != 0** (a detector that the zero padding of the K remainder works. As
 * detection limit ① of ADR 0022 says, though, a regression that zero-pads only one of A/B cancels
 * out and is not visible).
 */
const LINEAR_CASES: readonly LinearCase[] = [
  // v4 (n % 4 == 0) / m = 65 gives 2 row tiles / n = 68 gives 2 column tiles (the last tile has
  // 1 valid quad out of 16) / k = 20 is k4 = 5, 2 K tiles (the second holds 1 pack of 4)
  { name: "i8a8 v4 [65,20] × W[68,20]", m: 65, n: 68, k: 20 },
  // Scalar (n % 4 != 0)
  { name: "i8a8 scalar [17,20] × W[19,20]", m: 17, n: 19, k: 20 },
  // 9 K tiles (k4 = 33) with 1 pack in the last tile / n = 5 is scalar / m = 70 gives 2 row tiles
  { name: "i8a8 scalar long K [70,132] × W[5,132]", m: 70, n: 5, k: 132 },
  // A small shape that fits in 1 tile (the path where every guard stays true)
  { name: "i8a8 v4 small [3,4] × W[8,4]", m: 3, n: 8, k: 4 },
];

test.skipIf(!GPU_AVAILABLE)(
  "linear with linearCompute:'a8' matches the TS reference with atol=0 (v4 / scalar / tile edges / K remainder, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const testCase of LINEAR_CASES) {
        const prepared = prepareLinear(testCase);
        const actual = await runLinear(gpu, prepared, { linearCompute: "a8" });
        assertEquals(actual.y.shape, [testCase.m, testCase.n], testCase.name);
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // The check against a vacuous pass: with a constant output a "match" checks nothing
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: constant output`);
        // It really goes through the i8a8 path (2 pipelines: quantize_rows + i8a8 GEMM)
        assertEquals(actual.pipelineCount, 2, `${testCase.name}: pipeline count`);
        assertEquals(
          actual.keys,
          [
            linearI8a8Key(
              linearI8a8UsesVec4(testCase.n),
              true,
              i8a8LinearGeometry(profile, testCase.m),
            ),
            quantizeRowsKey(quantizeRowsGeometry(testCase.k)),
          ].sort(),
          `${testCase.name}: pipeline keys that ran`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "i8a8: a NaN row propagates across the whole row, and zero rows and negative values match exactly (real GPU)",
  async () => {
    // One NaN in row 2, row 5 all zeros, the rest mixed signs
    const nanRow = 2;
    const zeroRow = 5;
    const prepared = prepareLinear({
      name: "nan/zero",
      m: 8,
      n: 12,
      k: 20,
      x: (index, _m, k) => {
        const row = Math.floor(index / k);
        if (row === zeroRow) return 0;
        if (row === nanRow && index % k === k - 3) return Number.NaN;
        return SIGNED(index);
      },
    });
    const gpu = await acquireGpu();
    let y: Tensor;
    try {
      y = (await runLinear(gpu, prepared, { linearCompute: "a8" })).y;
    } finally {
      gpu.destroy();
    }
    assertExact(y.data, prepared.expected, "NaN / zero rows");
    // **Every column** of the NaN row is NaN (the same granularity as the f32 path — the reduction
    // spans the whole row)
    for (let col = 0; col < 12; col += 1) {
      assert(Number.isNaN(y.data[nanRow * 12 + col]), `NaN row col=${col}`);
    }
    // The next row is intact (the propagation does not cross rows)
    for (let col = 0; col < 12; col += 1) {
      assert(Number.isFinite(y.data[(nanRow + 1) * 12 + col]), `adjacent row col=${col}`);
    }
    // A zero row is the bias itself (q = 0, s = tiny, so f32(0)·(tiny·ws) + bias = bias)
    for (let col = 0; col < 12; col += 1) {
      assertEquals(
        y.data[zeroRow * 12 + col],
        prepared.expected[zeroRow * 12 + col],
        `zero row col=${col}`,
      );
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "i8a8: the dot4I8Packed version and the emulated version match with atol=0 (the extension only affects speed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      // This machine lists packed_4x8_integer_dot_product, so **both variants can actually run**.
      // This one test is the linear-side mechanical detector for the claim "the emulation is
      // numerically identical" (design §4.4; the QK side of the fused attention has the same form
      // in tests/gpu_attention_pv_i8a8_test.ts).
      const dots: readonly I8a8Dot[] = ["dp4a", "emu"];
      for (const testCase of LINEAR_CASES) {
        const prepared = prepareLinear(testCase);
        const results: Record<string, Tensor> = {};
        for (const dot of dots) {
          const options: SessionOptions = { linearCompute: "a8", [I8A8_DOT]: dot };
          const actual = await runLinear(gpu, prepared, options);
          results[dot] = actual.y;
          assertEquals(
            actual.keys.includes(
              linearI8a8Key(
                linearI8a8UsesVec4(testCase.n),
                dot === "dp4a",
                i8a8LinearGeometry(profile, testCase.m),
              ),
            ),
            true,
            `${testCase.name}: ${dot} key`,
          );
        }
        assertExact(results["emu"].data, results["dp4a"].data, `${testCase.name}: dp4a vs emu`);
        // Both match the TS reference (blocks the case where both are off by the same amount)
        assertExact(results["emu"].data, prepared.expected, `${testCase.name}: emu vs reference`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Runs `body` with `console.warn` replaced and always restores it (returns the emitted text as is).
 */
const captureWarnings = async (body: () => Promise<void>): Promise<string[]> => {
  const original = console.warn;
  const warnings: string[] = [];
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map((arg) => String(arg)).join(" "));
  };
  try {
    await body();
  } finally {
    console.warn = original;
  }
  return warnings;
};

test.skipIf(!GPU_AVAILABLE)(
  "i8a8: requesting a8 on a device without dp4a warns exactly once per device, and does not warn on a device with it (real GPU)",
  async () => {
    // ADR 0128 decision 5 — missing dp4a support is neither a fallback nor an automatic sibling
    // preset, but a warning. The unit tests of the decision (src/runtime/a8-emulation-warning.ts)
    // and of "once" (GpuContext.warnOnce) are in tests/a8_emulation_warning_test.ts. This checks
    // that both come out of a real Session build.
    const gpu = await acquireGpu();
    try {
      const smallCase: LinearCase = { name: "a8 warning [3,4] × W[8,4]", m: 3, n: 8, k: 4 };
      const prepared = prepareLinear(smallCase);
      // This machine lists dp4a (the comparison of both variants above assumes it) — on a device
      // with it, it passes silently.
      assert(dp4aAvailable(gpu.wgslLanguageFeatures), "assumes this machine lists dp4a");
      const quiet = await captureWarnings(async () => {
        await runLinear(gpu, prepared, { linearCompute: "a8" });
      });
      assertEquals(
        quiet.filter((warning) => warning.includes("dp4a")),
        [],
        "no warning on a device with dp4a",
      );
      // Wrap the same real device in a GpuContext that "does not list dp4a" — only the capability
      // surface changes, and the build really runs with the emulated kernel (the reason for the
      // warning and the actual run agree).
      const stripped = new GpuContext(gpu.device, gpu.adapterInfo, gpu.limits, new Set());
      const profile = deviceGeometryProfile(stripped);
      const warnings = await captureWarnings(async () => {
        const first = await runLinear(stripped, prepared, { linearCompute: "a8" });
        assertEquals(
          first.keys.includes(
            linearI8a8Key(
              linearI8a8UsesVec4(smallCase.n),
              false,
              i8a8LinearGeometry(profile, smallCase.m),
            ),
          ),
          true,
          "runs with the emulated version's key",
        );
        assertExact(first.y.data, prepared.expected, "emu vs reference");
        // A second a8 Session on the same device does not warn, nor does a Session that does not
        // use a8
        await runLinear(stripped, prepared, { linearCompute: "a8" });
        await runLinear(stripped, prepared, {});
      });
      assertEquals(
        warnings.length,
        1,
        `one warning per device (emitted: ${JSON.stringify(warnings)})`,
      );
      assertStringIncludes(warnings[0] ?? "", 'linearCompute: "a8"');
      assertStringIncludes(warnings[0] ?? "", "pass quant explicitly");
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "i8a8 takes effect only with opt-in × i8 resident × k%4==0 (the default stays on the original path, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      // ① The default (linearCompute omitted) uses the original f32 path even with i8 assets (just
      //   1 linear)
      const prepared = prepareLinear({ name: "default", m: 17, n: 19, k: 20 });
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.pipelineCount, 1, "the default has 1 pipeline (no quantize_rows)");
      // What to check is that "just 1 f32 linear (wi8 storage) ran, not i8a8". The spelling of
      // the key itself is pinned by tests/geometry_profile_test.ts, so here it is looked up from
      // the same shape (m=17 → the small-M tile geometry bucket / n=19 → the scalar variant).
      assertEquals(
        baseline.keys,
        [linearKey("i8", false, "f32", 17, undefined, gemmRowsGeometry(profile, 17))],
        "default key",
      );
      // The default values are w8 (only the weights quantized), so they **do not match** the w8a8
      // reference (if they did, the activation quantization would have no effect)
      const i8a8 = await runLinear(gpu, prepared, { linearCompute: "a8" });
      assertEquals(
        [...baseline.y.data].some((value, index) => value !== i8a8.y.data[index]),
        true,
        "activation quantization did not change the output (the opt-in has no effect)",
      );

      // ② k % 4 != 0 falls to the original path even with opt-in (the word-boundary condition of
      //   the i8 payload)
      const odd = prepareLinear({ name: "k=7", m: 5, n: 6, k: 7 });
      const oddRun = await runLinear(gpu, odd, { linearCompute: "a8" });
      assertEquals(oddRun.pipelineCount, 1, "k % 4 != 0 uses the f32 path");
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The degenerate K=0 shape (in_features 0). The contract (linear in src/ops.ts) allows it, but it
 * satisfies `k % 4 == 0`, so if the selection predicate did not look at `k > 0` it would enter the
 * i8a8 path and collide with `dim >= 1` of ① `quantize_rows`, giving **a CodegenError specific to
 * i8a8** (there are no activations to quantize at all, so as an i8a8 check the exception means
 * nothing).
 *
 * What is pinned is that "**opt-in does not change the K=0 path**" — it fails with **the same**
 * failure as the default session. K=0 itself also fails on the default f32 path with a minimum-size
 * violation of a 0-byte binding (a separate cause unrelated to linearCompute), so this only checks
 * that the 2 paths agree.
 */
test.skipIf(!GPU_AVAILABLE)(
  "i8a8: a K=0 i8 resident linear takes the same path as the default even with opt-in (no i8a8-specific failure, real GPU)",
  async () => {
    const m = 3;
    const n = 8;
    const bias = fill([n], SIGNED);
    // The weights have 0 elements ([n,0]). The scale only keeps its per-channel shape (the
    // reduction is empty, so it is never read).
    const quantized = quantizeI8(new Float32Array(0), [n, 0], 0);
    // NOTE: a `krm` cannot hold a 0-element weight (the writer does not make a 0-length block —
    // container-v1 §4.1). The subject is exercising the path of the degenerate shape, so the supply
    // is passed as an in-memory container.
    const { declaration, tensors } = i8LinearParts(m, n, 0, quantized, bias);
    const model = memoryModel(declaration, tensors);
    const x = fill([m, 0], SIGNED);
    const runK0 = async (gpu: GpuContext, options: SessionOptions): Promise<Error> => {
      const session = await createSessionFromContainer(gpu, model, GRAPH_NAME, options);
      try {
        return await assertRejects(() => session.run({ x }), Error);
      } finally {
        await session.dispose();
      }
    };
    const gpu = await acquireGpu();
    try {
      const baseline = await runK0(gpu, {});
      const optIn = await runK0(gpu, { linearCompute: "a8" });
      assertEquals(optIn.name, baseline.name, "K=0: opt-in changed the kind of exception");
      assertEquals(
        optIn.message,
        baseline.message,
        "K=0: opt-in changed the reason for the failure",
      );
      // The check against a vacuous pass: it is not the ① check that fires when the predicate picks
      // up K=0
      assert(!optIn.message.includes("quantize_rows"), `K=0: entered ① of i8a8 (${optIn.message})`);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "i8a8: fail loudly when k exceeds the limit of the i32 reduction (2^17) (no silent wrap-around, real GPU)",
  async () => {
    // Just outside the limit (2^17 + 4, a multiple of 4). |acc| ≤ k·127·128 fits in i32, but the
    // limit errs on the safe side.
    const k = LINEAR_I8A8_MAX_K + 4;
    const prepared = prepareLinear({ name: "overflow gate", m: 1, n: 4, k });
    const gpu = await acquireGpu();
    try {
      const session = await createSessionFromContainer(gpu, await prepared.model, GRAPH_NAME, {
        linearCompute: "a8",
      });
      try {
        await assertRejects(
          () => session.run({ x: prepared.x }),
          ExecutionError,
          "i32 reduction guard",
        );
      } finally {
        await session.dispose();
      }
      // On the default f32 path the same model runs normally (the limit belongs only to the i8a8
      // path)
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.y.shape, [1, 4]);
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// w4a8 (i4 resident weights × per-token i8 activations — perf-ledger Q-8)
// ---------------------------------------------------------------------------

/**
 * The numeric contract this section pins is **the same atol=0** as the i8 variant; only the basis
 * differs: inside a group an exact i32 dot product, a single flush to f32 only at group boundaries,
 * and xs applied once at the end.
 * So the floating point operations are fixed at `k/g + 1` per output element, and whatever changes
 * among the tile split, the fill assignment or dp4a / emu, the values do not move by a single bit.
 *
 * MUST: do **not use the f32 i4 path (an f32 GEMM of the fake-quant weights)** for the expected
 * values. Its activations stay f32, so if it matched, the activation quantization would have no
 * effect — a vacuous pass by construction. The check "values differ from the default path" below
 * backs this up.
 * MUST: the weights **alternate the sign of adjacent elements** (a symmetric pattern matches even
 * with the nibble halves swapped — ADR 0069 decision 4 ①; the only value-side detector for the
 * i4lanes order).
 * MUST: the weights **vary in amplitude per group** (with the same scale in every group, a mix-up
 * of the group index [swapping row/group, a wrong shift] never shows in the values).
 */
const groupVarying =
  (cols: number, groupSize: number) =>
  (i: number): number => {
    const group = Math.floor((i % cols) / groupSize);
    const row = Math.floor(i / cols);
    const base = (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);
    return base * (1 + group * 0.75 + (row % 5) * 0.25);
  };

/** A graph with one `linear(x, w, b)` (w is i4 + group scale). */
const i4LinearModel = (
  m: number,
  n: number,
  k: number,
  groupSize: number,
  quantized: ReturnType<typeof quantizeI4>,
  bias: FilledTensor,
): Promise<OpenedContainer> => {
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [m, k] }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      y: { dtype: "f32", shape: [m, n] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return openModelBytes(declaration, [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes([...bias.data]),
      encoding: { codec: "f32" },
    },
  ]);
};

type W4a8Case = {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly groupSize: number;
};

/**
 * How the shapes are split up.
 *
 * MUST: **keep the v4 path and the scalar path as a pair** (the write guard is a different
 * expression per variant).
 * MUST: **m / n cross the tile edges (128 × 64)** (collapsed below 1 tile, the guard is never
 * false).
 * MUST: include **shapes where the group changes at least twice within a row** (with 1 group per
 * row there is only one flush, and neither the group index nor the folding order is checked).
 * MUST: have **both group = K tile (g16) and 2 tiles per group (g32)** (blocks an error that passes
 * only when the inner loop's bound is 1 [forgetting to multiply by gbase]).
 */
const W4A8_CASES: readonly W4a8Case[] = [
  // v4 / m = 130 gives 2 row tiles, n = 68 gives 2 column tiles (the last tile has 1 valid quad out
  // of 16)
  // / k = 96 g32 = 3 groups per row, 1 group spans 2 K tiles
  { name: "w4a8 v4 [130,96] × W[68,96] g32", m: 130, n: 68, k: 96, groupSize: 32 },
  // Scalar (n % 4 != 0) / m = 129 and n = 67 both cross the tile edges / k = 48 g16 = 3 groups per
  // row, **1 group is exactly 1 K tile**
  { name: "w4a8 scalar [129,48] × W[67,48] g16", m: 129, n: 67, k: 48, groupSize: 16 },
  // A small shape that fits in 1 tile (the path where every guard stays true), 1 group per row
  { name: "w4a8 v4 small [3,32] × W[8,32] g32", m: 3, n: 8, k: 32, groupSize: 32 },
  // Small scalar / k = 96 g32 = 3 groups per row (a control with only the group layout matching the
  // v4 A)
  { name: "w4a8 scalar small [17,96] × W[19,96] g32", m: 17, n: 19, k: 96, groupSize: 32 },
];

type PreparedW4a8 = {
  readonly model: Promise<BoundContainer>;
  readonly x: Tensor;
  readonly expected: Float32Array<ArrayBuffer>;
};

const prepareW4a8 = (
  testCase: W4a8Case,
  xValue: (index: number, m: number, k: number) => number = SIGNED,
): PreparedW4a8 => {
  const { m, n, k, groupSize } = testCase;
  const weight = fill([n, k], groupVarying(k, groupSize));
  const quantized = quantizeI4(weight.data, [n, k], groupSize);
  // The check against a vacuous pass: unless adjacent elements alternate in sign, a nibble mix-up
  // does not show in the values
  let alternating = 0;
  for (let i = 0; i + 1 < quantized.q.length; i += 1) {
    if (quantized.q[i] > 0 && quantized.q[i + 1] < 0) alternating += 1;
  }
  if (alternating * 4 < quantized.q.length) {
    throw new Error(`${testCase.name}: only ${alternating} adjacent pairs alternate in sign`);
  }
  const bias = fill([n], SIGNED);
  const x = fill([m, k], (index) => xValue(index, m, k));
  // Checks that the data allows claiming atol=0 (the MUST of {@link SIGNED} at the top of the
  // file).
  // Non-finite and all-zero rows do not round, so they are replaced out of scope (non-finite values
  // pass while the check stays).
  const gated = Float32Array.from(x.data);
  for (let row = 0; row < m; row += 1) {
    const slice = gated.subarray(row * k, row * k + k);
    const exempt =
      slice.some((value) => !Number.isFinite(value)) || slice.every((value) => value === 0);
    if (exempt) {
      for (let i = 0; i < k; i += 1) slice[i] = SIGNED(row * k + i);
    }
  }
  const margin = quantizeRowsTieMargin(gated, m, k);
  if (!(margin > TIE_MARGIN)) {
    throw new Error(`${testCase.name}: the margin from the rounding boundaries is only ${margin}`);
  }
  const expected = referenceLinearW4a8({
    x: x.data,
    weight: quantized.q,
    weightScale: quantized.scale,
    bias: bias.data,
    m,
    n,
    k,
    groupSize,
  });
  return { model: i4LinearModel(m, n, k, groupSize, quantized, bias), x, expected };
};

test.skipIf(!GPU_AVAILABLE)(
  "linearCompute:'a8' × i4 resident (w4a8) matches the TS reference with atol=0 (v4 / scalar / tile edges / 2 group sizes, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const testCase of W4A8_CASES) {
        const prepared = prepareW4a8(testCase);
        const actual = await runLinear(gpu, prepared, { linearCompute: "a8" });
        assertEquals(actual.y.shape, [testCase.m, testCase.n], testCase.name);
        assertExact(actual.y.data, prepared.expected, testCase.name);
        // The check against a vacuous pass: with a constant output a "match" checks nothing
        assert(new Set([...actual.y.data]).size > 1, `${testCase.name}: constant output`);
        // It really goes through the w4a8 path (2 pipelines: quantize_rows + w4a8 GEMM)
        assertEquals(actual.pipelineCount, 2, `${testCase.name}: pipeline count`);
        const key = linearI8a8Key(
          linearI8a8UsesVec4(testCase.n),
          true,
          linearI8a8Geometry(profile, testCase.m, "i4", testCase.groupSize),
          "i4",
          testCase.groupSize,
        );
        assertEquals(
          actual.keys,
          [key, quantizeRowsKey(quantizeRowsGeometry(testCase.k))].sort(),
          `${testCase.name}: pipeline keys that ran`,
        );
        // The diagnostics show "i4 resident × its group length" (ADR 0021)
        assertEquals(
          key.endsWith(`:wi4g${testCase.groupSize}`),
          true,
          `${testCase.name}: discriminator`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "w4a8: a NaN row propagates across the whole row, and zero rows and negative values match exactly (real GPU)",
  async () => {
    // The propagation of non-finite values **does not rest on the same basis** as i8a8 — w4a8
    // cannot fold `xs · wscale` early (wscale changes per group) and leaves `xs` for the single
    // final fma.
    // So where `xs = ±Inf` and a group reduction is 0, a branch like `fma(0, Inf, bias)` could
    // diverge, and the "propagation of NaN rows (part of the design)" that the top of the file
    // states as the overall contract is pinned separately for the new variant too.
    const testCase = W4A8_CASES[0];
    const { m, n } = testCase;
    const nanRow = 2;
    const zeroRow = 5;
    const prepared = prepareW4a8(testCase, (index, _m, cols) => {
      const row = Math.floor(index / cols);
      if (row === zeroRow) return 0;
      if (row === nanRow && index % cols === cols - 3) return Number.NaN;
      return SIGNED(index);
    });

    const gpu = await acquireGpu();
    let y: Tensor;
    try {
      y = (await runLinear(gpu, prepared, { linearCompute: "a8" })).y;
    } finally {
      gpu.destroy();
    }

    assertExact(y.data, prepared.expected, "w4a8 NaN / zero rows");
    for (let col = 0; col < n; col += 1) {
      assert(Number.isNaN(y.data[nanRow * n + col]), `NaN row col=${col}`);
      assert(Number.isFinite(y.data[(nanRow + 1) * n + col]), `adjacent row col=${col}`);
      assertEquals(
        y.data[zeroRow * n + col],
        prepared.expected[zeroRow * n + col],
        `zero row col=${col}`,
      );
    }
    // The check against a vacuous pass: if everything outside the NaN and zero rows is constant, a
    // "match" checks nothing.
    const others = [...y.data].filter((_value, index) => {
      const row = Math.floor(index / n);
      return row !== nanRow && row !== zeroRow;
    });
    assert(new Set(others).size > 1, "constant outside the NaN / zero rows");
    assertEquals(y.shape, [m, n]);
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "w4a8: the dot4I8Packed version and the emulated version match with atol=0 (the i4 lane expansion does not depend on the dot-product variant, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const dots: readonly I8a8Dot[] = ["dp4a", "emu"];
      for (const testCase of W4A8_CASES) {
        const prepared = prepareW4a8(testCase);
        const results: Record<string, Tensor> = {};
        for (const dot of dots) {
          const options: SessionOptions = { linearCompute: "a8", [I8A8_DOT]: dot };
          const actual = await runLinear(gpu, prepared, options);
          results[dot] = actual.y;
          assertEquals(
            actual.keys.includes(
              linearI8a8Key(
                linearI8a8UsesVec4(testCase.n),
                dot === "dp4a",
                linearI8a8Geometry(profile, testCase.m, "i4", testCase.groupSize),
                "i4",
                testCase.groupSize,
              ),
            ),
            true,
            `${testCase.name}: ${dot} key`,
          );
        }
        assertExact(results["emu"].data, results["dp4a"].data, `${testCase.name}: dp4a vs emu`);
        // Both match the TS reference (blocks the case where both are off by the same amount)
        assertExact(results["emu"].data, prepared.expected, `${testCase.name}: emu vs reference`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "w4a8 takes effect only with opt-in, and its values differ from the default f32 i4 path (backing for the activation quantization, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const testCase = W4A8_CASES[3];
      const prepared = prepareW4a8(testCase);
      // ① The default (linearCompute omitted) uses the original f32 compute path even with i4
      //   assets (just 1 linear)
      const baseline = await runLinear(gpu, prepared, {});
      assertEquals(baseline.pipelineCount, 1, "the default has 1 pipeline (no quantize_rows)");
      assertEquals(
        baseline.keys,
        [
          linearKey(
            "i4",
            false,
            "f32",
            testCase.m,
            testCase.groupSize,
            gemmRowsGeometry(profile, testCase.m),
          ),
        ],
        "default key",
      );
      // MUST: the default values keep f32 activations, so they **do not match** the w4a8 reference.
      // If they did, the activation quantization would have no effect = the same vacuous pass as
      // using the f32 i4 path for the expected values.
      const w4a8 = await runLinear(gpu, prepared, { linearCompute: "a8" });
      assertEquals(
        [...baseline.y.data].some((value, index) => value !== w4a8.y.data[index]),
        true,
        "activation quantization did not change the output (the opt-in has no effect)",
      );
    } finally {
      gpu.destroy();
    }
  },
);
