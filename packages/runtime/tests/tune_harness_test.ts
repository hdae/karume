import { test } from "vite-plus/test";
import { assert, assertEquals, assertMatch, assertThrows } from "@std/assert";
import { acquireGpu } from "../mod.ts";
import { parseStorageRoles } from "../src/gpu/pipeline-cache.ts";
import { gemmMTileGeometry } from "../src/kernels/gemm.ts";
import { defaultGemmGeometry } from "../src/kernels/gemm-geometry.ts";
import { conv3dI8a8Kernel, conv3dI8a8Key } from "../src/kernels/conv3d-i8a8.ts";
import { defaultI8a8Geometry } from "../src/kernels/i8a8-geometry.ts";
import { linearWgsl } from "../src/kernels/linear.ts";
import { calibrateReps, TARGET_PASS_MS } from "../src/tune/measurement.ts";
import {
  type AttentionCase,
  type BmmCase,
  type Conv2dCase,
  type Conv3dI8a8Case,
  type LinearCase,
  type MatmulCase,
  SWEEP_CASES,
  type SweepCase,
} from "../src/tune/cases.ts";
import {
  conv2dCandidate,
  gemmCandidate,
  type GeometryCandidate,
  i8a8Candidate,
} from "../src/tune/geometries.ts";
import {
  assertBindingRoles,
  casePlan,
  type CaseResult,
  createSweepContext,
  destroySweepContext,
  joinChunkDigests,
  resourceWordStream,
  SWEEP_MAX_REPS,
  sweepCase,
} from "../src/tune/harness.ts";

// The shape table's DiT shapes take a few ms to a few hundred ms each, so the GPU tests run a small
// shape on the same path (the smallest round shape that lands in the 128×128 default-geometry
// bucket at M ≥ 513).
const LINEAR: LinearCase = {
  id: "test-linear-m1024-n512-k256",
  op: "linear",
  m: 1024,
  n: 512,
  k: 256,
  censusCount: 1,
  source: "test",
};
const I8A8_LINEAR: LinearCase = {
  ...LINEAR,
  id: "test-i8a8-linear-m1024-n512-k256",
  op: "i8a8-linear",
};

/**
 * A small fused-attention shape (B·H 2, M = N = 128, D 128 — one tile of the 128×128 default
 * geometry).
 */
const attentionCase = (op: AttentionCase["op"], stage: AttentionCase["stage"]): AttentionCase => ({
  id: `test-${op}-${stage}-bh2-m128-n128-d128`,
  op,
  stage,
  batchHeads: 2,
  m: 128,
  n: 128,
  d: 128,
  scale: 0.2973017692565918,
  score: op === "attention" ? "f32" : "f16",
  censusCount: 1,
  source: "test",
});

/**
 * A small conv2d shape (Cin = Cout = 64, 16², 3×3 — the production geometry with m tiles of 64
 * rows).
 */
const CONV2D: Conv2dCase = {
  id: "test-conv2d-c64-16x16",
  op: "conv2d",
  channelsIn: 64,
  channelsOut: 64,
  height: 16,
  width: 16,
  censusCount: 1,
  source: "test",
};

/**
 * A small conv3d w8a8 case: Cout 80 (not a multiple of any tile edge), Cin 36 (Cin % 16 != 0 — a
 * K tail in each tap), 3×3×3 with spatial padding 1 over [3, 9, 10] → [1, 9, 10] (P = 90, the scalar
 * store).
 */
const CONV3D: Conv3dI8a8Case = {
  id: "test-i8a8-conv3d-cin36-cout80-k333-in3x9x10",
  op: "i8a8-conv3d",
  dims: {
    channelsIn: 36,
    channelsOut: 80,
    timeIn: 3,
    heightIn: 9,
    widthIn: 10,
    timeOut: 1,
    heightOut: 9,
    widthOut: 10,
    kernelT: 3,
    kernelH: 3,
    kernelW: 3,
    strideT: 1,
    strideH: 1,
    strideW: 1,
    paddingT: 0,
    paddingH: 1,
    paddingW: 1,
    dilationT: 1,
    dilationH: 1,
    dilationW: 1,
    groups: 1,
  },
  censusCount: 1,
  source: "test",
};

/**
 * CONV3D without padding over [3, 5, 6] → [1, 3, 4]: P_out = 12 (a multiple of 4, not of 8) and
 * P_in = 90 (not a multiple of 4) — the store is vec4, decided by the output pixels.
 */
const CONV3D_VEC4_OUT: Conv3dI8a8Case = {
  ...CONV3D,
  id: "test-i8a8-conv3d-cin36-cout80-k333-in3x5x6",
  dims: {
    ...CONV3D.dims,
    heightIn: 5,
    widthIn: 6,
    heightOut: 3,
    widthOut: 4,
    paddingH: 0,
    paddingW: 0,
  },
};

/**
 * CONV3D without padding over [3, 5, 4] → [1, 3, 2]: P_in = 60 (a multiple of 4) and P_out = 6
 * (not) — the store is scalar, decided by the output pixels.
 */
const CONV3D_SCALAR_OUT: Conv3dI8a8Case = {
  ...CONV3D_VEC4_OUT,
  id: "test-i8a8-conv3d-cin36-cout80-k333-in3x5x4",
  dims: { ...CONV3D_VEC4_OUT.dims, widthIn: 4, widthOut: 2 },
};

/** 64×64 / 256 threads (the second entry of quick). */
const GEMM_64X64 = gemmCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 });
const I8A8_64X64 = i8a8Candidate({ regM: 4, regN: 4, wgX: 16, wgY: 16, tileK: 16 });
const CONV2D_64X64 = conv2dCandidate({ regM: 4, regN: 4, wgX: 16, wgY: 16 });

const adapter = navigator.gpu === undefined ? null : await navigator.gpu.requestAdapter();
const timestampQuery = adapter !== null && adapter.features.has("timestamp-query");

/**
 * Asserts that 2 rows (the default → another geometry) were measured with matching output digests,
 * and that the ratio of re-measuring the default geometry at the end of the case was kept. The key
 * contains the geometry's name (the name is spelled without v4, so `v4` is removed from the key
 * before looking).
 */
const assertMeasuredPair = (result: CaseResult, defaultName: string, other: string): void => {
  const { rows, summary } = result;
  assertEquals(rows.length, 2);
  const [first, second] = rows;
  assertEquals(first.isDefault, true);
  assertEquals(first.geometry, defaultName);
  assertEquals(second.geometry, other);
  for (const row of rows) {
    assertEquals(row.error, undefined, row.error);
    assert((row.perDispatch ?? 0) > 0, `${row.geometry}: ${row.perDispatch}`);
    assert((row.reps ?? 0) >= 1);
    assertMatch(row.outputSha256 ?? "", /^[0-9a-f]{64}$/);
    const key = (row.key ?? "").replaceAll("v4", "");
    assert(key.includes(row.geometry), `${row.key} does not contain ${row.geometry}`);
  }
  assertEquals(second.identicalToDefault, true);
  assert((second.speedupVsDefault ?? 0) > 0);
  assertEquals(summary.caseId, first.caseId);
  assertEquals(summary.defaultRepeatError, undefined, summary.defaultRepeatError);
  assert((summary.defaultRepeat?.perDispatch ?? 0) > 0);
  assert((summary.defaultRepeat?.driftRatio ?? 0) > 0);
};

/** Runs 1 case on the real GPU with the default + 1 geometry (rounds 1). */
const sweepPair = async (
  target: SweepCase,
  other: GeometryCandidate,
  gpuTiming = timestampQuery,
): Promise<CaseResult> => {
  const gpu = await acquireGpu(gpuTiming ? { gpuTiming: true } : {});
  const context = await createSweepContext(gpu, "deno-raw-tick");
  try {
    assertEquals(context.unit, gpuTiming ? "deno-raw-tick" : "wall");
    return await sweepCase(context, target, [other], { rounds: 1 });
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

test("the binding table passes the comparison with the WGSL storage roles (count and the position of the write target)", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertBindingRoles(
    { key: "linear", bindings: ["x", "w", "bias", "out"], writes: ["out"] },
    roles,
  );
});

test("fails when the binding table's write-target position differs from the WGSL (detects swapped positional bindings)", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertThrows(
    () =>
      assertBindingRoles(
        { key: "linear", bindings: ["x", "w", "out", "bias"], writes: ["out"] },
        roles,
      ),
    Error,
    "write bindings",
  );
});

test("the binding tables of every matmul / bmm case agree with the storage roles of the default geometry's WGSL", () => {
  const dense = SWEEP_CASES.filter((target) => target.op === "matmul" || target.op === "bmm");
  assertEquals(new Set(dense.map((target) => target.op)), new Set(["matmul", "bmm"]));
  for (const target of dense) {
    const plan = casePlan(target, 65535, false);
    const launch = plan.launch(plan.defaultCandidate);
    assertBindingRoles(launch, parseStorageRoles(launch.wgsl));
    // Every binding name is in the resource table, and the output is the write target
    const names = new Set(plan.resources.map((spec) => spec.name));
    for (const name of launch.bindings) assert(names.has(name), `${target.id}: ${name}`);
    assert(launch.writes.includes(plan.output), target.id);
  }
});

test("every conv3d w8a8 case's binding table fits the WGSL's storage roles (both dp4a variants), and the resources are the dims' sizes", () => {
  const conv3d = SWEEP_CASES.flatMap((target) => (target.op === "i8a8-conv3d" ? [target] : []));
  assert(conv3d.length > 0);
  for (const target of [...conv3d, CONV3D]) {
    const { dims } = target;
    for (const dp4a of [false, true]) {
      const plan = casePlan(target, 65535, dp4a);
      const launch = plan.launch(plan.defaultCandidate);
      assertBindingRoles(launch, parseStorageRoles(launch.wgsl));
      assertEquals(launch.bindings, ["xq", "w", "bias", "out", "wscale", "xscale"], target.id);
      assertEquals(launch.writes, ["out"], target.id);
      assertEquals(plan.output, "out", target.id);
    }
    const pixelsIn = dims.timeIn * dims.heightIn * dims.widthIn;
    const pixelsOut = dims.timeOut * dims.heightOut * dims.widthOut;
    const taps = dims.kernelT * dims.kernelH * dims.kernelW;
    const sizes = Object.fromEntries(
      casePlan(target, 65535, true).resources.map((spec) => [spec.name, spec.bytes]),
    );
    assertEquals(
      sizes,
      {
        xq: pixelsIn * dims.channelsIn,
        xscale: pixelsIn * 4,
        w: Math.ceil((dims.channelsOut * taps * dims.channelsIn) / 4) * 4,
        wscale: dims.channelsOut * 4,
        bias: dims.channelsOut * 4,
        out: dims.channelsOut * pixelsOut * 4,
      },
      target.id,
    );
  }
});

test("every conv3d w8a8 case launches the shared kernel function's variant, which is vec4 exactly when P_out is a multiple of 4 (store variant against an independent rule, key against conv3dI8a8Key)", () => {
  const conv3d = SWEEP_CASES.flatMap((target) => (target.op === "i8a8-conv3d" ? [target] : []));
  assert(conv3d.length > 0);
  // CONV3D has 90 output pixels (not a multiple of 4), so the scalar store is covered too. The
  // sweep cases all have P_out % 8 == 0, so CONV3D_VEC4_OUT (P_out % 8 != 0, P_in % 4 != 0) and
  // CONV3D_SCALAR_OUT (P_in % 4 == 0, P_out % 4 != 0) separate the output pixel count from the
  // input's and the multiple of 4 from that of 8
  const targets = [...conv3d, CONV3D, CONV3D_VEC4_OUT, CONV3D_SCALAR_OUT];
  assert(targets.some(({ dims }) => (dims.timeOut * dims.heightOut * dims.widthOut) % 4 !== 0));
  const candidates = [
    i8a8Candidate(defaultI8a8Geometry("conv3d")),
    i8a8Candidate({ regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 }),
  ];
  for (const target of targets) {
    const { dims } = target;
    // The store variant derived here independently: vec4 exactly when P_out is a multiple of 4
    const v4 = (dims.timeOut * dims.heightOut * dims.widthOut) % 4 === 0;
    for (const dp4a of [false, true]) {
      const plan = casePlan(target, 65535, dp4a);
      for (const candidate of candidates) {
        const where = `${target.id} ${candidate.name} dp4a=${dp4a}`;
        const launch = plan.launch(candidate);
        assertEquals(conv3dI8a8Kernel(dims, dp4a, candidate.geometry).v4, v4, where);
        assertEquals(launch.key, conv3dI8a8Key(v4, dp4a, candidate.geometry), where);
      }
    }
  }
});

test("fails when the binding table's count differs from the WGSL", () => {
  const roles = parseStorageRoles(linearWgsl("f16", true, "f32", 1024));
  assertThrows(
    () =>
      assertBindingRoles({ key: "linear", bindings: ["x", "w", "out"], writes: ["out"] }, roles),
    Error,
    "storage bindings",
  );
});

test("even for a small case of 15 µs per dispatch, the sweep's iterations go beyond opbench's cap of 1024 so a pass reaches the target length", () => {
  const nsPerDispatch = 15e3;
  const reps = calibrateReps(nsPerDispatch, TARGET_PASS_MS, SWEEP_MAX_REPS);
  assertEquals(reps, 5334);
  assert(reps > 1024);
  assert(reps * nsPerDispatch >= TARGET_PASS_MS * 1e6);
});

test.skipIf(adapter === null)(
  "real GPU: measuring f32 linear with the default and 64×64/256 gives positive times for both and matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(LINEAR, GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring i8a8 linear with the default and 64×64/256 gives positive times for both and matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(I8A8_LINEAR, I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("linear")).name,
      I8A8_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: a device without timestamp-query runs the same protocol on the wall clock (wall)",
  async () => {
    const result = await sweepPair(LINEAR, GEMM_64X64, false);
    assertMeasuredPair(result, gemmCandidate(defaultGemmGeometry()).name, GEMM_64X64.name);
    // With the unit wall, the round values are the wall-clock times themselves
    for (const row of result.rows) assertEquals(row.rounds, row.wallRounds);
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring ①QK of the f32 fused attention with the default and 64×64/256 gives matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("attention", "qk"), GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring ③PV of the f32 fused attention (preceded by ①QK → ② row stats) with the default and 64×64/256 gives matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("attention", "pv"), GEMM_64X64),
      gemmCandidate(defaultGemmGeometry()).name,
      GEMM_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring ①QK of the i8a8 fused attention with the default and 64×64/256 gives matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("i8a8-attention", "qk"), I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("attention_qk")).name,
      I8A8_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring ③PV of the i8a8 fused attention with the default and 64×64/256 gives matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(attentionCase("i8a8-attention", "pv"), I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("attention_pv")).name,
      I8A8_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: conv3d w8a8 measured with the default and 64×64/256 gives positive times and the same output",
  async () => {
    assertMeasuredPair(
      await sweepPair(CONV3D, I8A8_64X64),
      i8a8Candidate(defaultI8a8Geometry("conv3d")).name,
      I8A8_64X64.name,
    );
  },
);

test.skipIf(adapter === null)(
  "real GPU: measuring conv2d's implicit GEMM with the default and 64×64/256 gives matching outputs",
  async () => {
    assertMeasuredPair(
      await sweepPair(CONV2D, CONV2D_64X64),
      conv2dCandidate(gemmMTileGeometry(64)).name,
      CONV2D_64X64.name,
    );
  },
);

// Agreement between geometries cannot detect the case where "every geometry makes the same mistake
// (swapped bindings, wrong params)", so the output of linear's default geometry is also compared
// with a CPU reference. The reference's rounding order follows the contract of src/kernels/gemm.ts
// (K ascending, sequential accumulation `acc + a * b` into acc, no explicit fma, bias once at the
// end), and bit identity is required.

/** f16 bits → value (normal and subnormal numbers; the sweep's inputs are only normal numbers). */
const f16ToNumber = (bits: number): number => {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

/** The neighbouring f32 (towards +∞ when `up`). */
const adjacentF32 = (value: number, up: boolean): number => {
  const bits = new Uint32Array(new Float32Array([value]).buffer);
  if (value === 0) {
    bits[0] = 1;
    const tiny = new Float32Array(bits.buffer)[0];
    return up ? tiny : -tiny;
  }
  bits[0] += value > 0 === up ? 1 : -1;
  return new Float32Array(bits.buffer)[0];
};

/**
 * An f32 fma (`a·b + c` rounded only once). A product of two f32 values is exact in f64, so the
 * error of the sum is taken with TwoSum, and only when the value lands on an f32 rounding midpoint
 * does the error's sign decide the direction. For diagnostics on failure (the contract does not
 * make fma explicit, but the failure message can then tell whether the backend contracted).
 */
const fmaF32 = (a: number, b: number, c: number): number => {
  const product = a * b;
  const sum = product + c;
  const virtual = sum - product;
  const error = product - (sum - virtual) + (c - virtual);
  const rounded = Math.fround(sum);
  if (error === 0 || rounded === sum) return rounded;
  const other = adjacentF32(rounded, rounded < sum);
  if ((rounded + other) / 2 !== sum) return rounded;
  return error > 0 ? Math.max(rounded, other) : Math.min(rounded, other);
};

const digestOf = async (values: Float32Array<ArrayBuffer>): Promise<string> =>
  await joinChunkDigests([await crypto.subtle.digest("SHA-256", new Uint8Array(values.buffer))]);

/** Builds one resource-table entry as the same words the harness wrote to the GPU. */
const resourceWords = (target: SweepCase, name: string): Uint32Array<ArrayBuffer> => {
  const specs = casePlan(target, 65535, false).resources;
  const index = specs.findIndex((spec) => spec.name === name);
  const spec = specs[index];
  if (spec === undefined || spec.fill === "none") throw new Error(`resource ${name} is not filled`);
  const words = new Uint32Array(Math.max(4, Math.ceil(spec.bytes / 4) * 4) / 4);
  const next = resourceWordStream(spec.fill, index);
  for (let at = 0; at < words.length; at += 1) words[at] = next();
  return words;
};

test.skipIf(adapter === null)(
  "real GPU: the output of f32 linear's default geometry is bit-identical to a CPU reference (sequential multiply-add in K order + bias at the end, either uncontracted or fma-contracted)",
  async () => {
    const target: LinearCase = {
      id: "test-linear-cpu-m64-n64-k64",
      op: "linear",
      m: 64,
      n: 64,
      k: 64,
      censusCount: 1,
      source: "test",
    };
    const { m, n, k } = target;
    const x = new Float32Array(resourceWords(target, "x").buffer);
    const weightWords = resourceWords(target, "w");
    const bias = new Float32Array(resourceWords(target, "bias").buffer);
    // Two f16 weights per word (low half = even index — kernels/weight-storage.ts)
    const weight = (index: number): number =>
      f16ToNumber((weightWords[index >> 1] >>> ((index & 1) * 16)) & 0xffff);
    const expected = new Float32Array(m * n);
    // A reference in the same order with only the multiply-add contracted to fma. The contract of
    // src/kernels/gemm.ts is "`acc + a * b` as written", and whether it is contracted to fma is up
    // to the backend (B570 / Vulkan contracts, RTX / Vulkan did not — the environment-dependent
    // axis of docs/limitations.md). Swapped bindings, params or K order match neither reference, so
    // "bit identical to one of the two" keeps this test's purpose (detecting wiring mistakes).
    const contracted = new Float32Array(m * n);
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        let fused = 0;
        for (let at = 0; at < k; at += 1) {
          const a = x[row * k + at];
          const b = weight(col * k + at);
          acc = Math.fround(acc + Math.fround(a * b));
          fused = fmaF32(a, b, fused);
        }
        expected[row * n + col] = Math.fround(acc + bias[col]);
        contracted[row * n + col] = Math.fround(fused + bias[col]);
      }
    }
    const reference = await digestOf(expected);
    const contractedReference = await digestOf(contracted);
    const gpu = await acquireGpu(timestampQuery ? { gpuTiming: true } : {});
    const context = await createSweepContext(gpu, "deno-raw-tick");
    try {
      const { rows } = await sweepCase(context, target, [], { rounds: 1 });
      assertEquals(rows.length, 1);
      assertEquals(rows[0].error, undefined, rows[0].error);
      const digest = rows[0].outputSha256;
      assert(
        digest === reference || digest === contractedReference,
        `output ${digest} matches neither the uncontracted CPU reference ${reference} nor the fma-contracted CPU reference ${contractedReference} (one of the bindings, params or K order is wrong)`,
      );
    } finally {
      destroySweepContext(context);
      gpu.destroy();
    }
  },
);

/**
 * The two CPU references (uncontracted / fma-contracted) of a dense GEMM (matmul / bmm — no bias,
 * B side `[K,N]`). The rounding order is the same as linear's reference (sequential multiply-add in
 * K order). bmm lays `batch` matrices out contiguously.
 */
const denseReferences = async (
  target: MatmulCase | BmmCase,
): Promise<{ readonly plain: string; readonly contracted: string }> => {
  const batch = target.op === "bmm" ? target.batch : 1;
  const { m, n, k } = target;
  const a = new Float32Array(resourceWords(target, "a").buffer);
  const b = new Float32Array(resourceWords(target, "b").buffer);
  const plain = new Float32Array(batch * m * n);
  const contracted = new Float32Array(batch * m * n);
  for (let z = 0; z < batch; z += 1) {
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < n; col += 1) {
        let acc = 0;
        let fused = 0;
        for (let at = 0; at < k; at += 1) {
          const left = a[(z * m + row) * k + at];
          const right = b[(z * k + at) * n + col];
          acc = Math.fround(acc + Math.fround(left * right));
          fused = fmaF32(left, right, fused);
        }
        plain[(z * m + row) * n + col] = acc;
        contracted[(z * m + row) * n + col] = fused;
      }
    }
  }
  return { plain: await digestOf(plain), contracted: await digestOf(contracted) };
};

/**
 * Measures the 2 rows of the default geometry and 64×64/256, and asserts that the default's output
 * is bit-identical to one of the CPU references and that the other row matches the default (swapped
 * bindings, params or dispatch disagree with the reference).
 */
const assertDenseMatchesReference = async (target: MatmulCase | BmmCase): Promise<void> => {
  const { plain, contracted } = await denseReferences(target);
  const gpu = await acquireGpu(timestampQuery ? { gpuTiming: true } : {});
  const context = await createSweepContext(gpu, "deno-raw-tick");
  try {
    const { rows } = await sweepCase(context, target, [GEMM_64X64], { rounds: 1 });
    assertEquals(rows.length, 2);
    for (const row of rows) assertEquals(row.error, undefined, row.error);
    const digest = rows[0].outputSha256;
    assert(
      digest === plain || digest === contracted,
      `${target.op}: output ${digest} matches neither the uncontracted CPU reference ${plain} nor the fma-contracted CPU reference ${contracted} (one of the bindings, params or dispatch is wrong)`,
    );
    assertEquals(rows[1].identicalToDefault, true);
  } finally {
    destroySweepContext(context);
    gpu.destroy();
  }
};

/** One i8 lane of a packed word (lane j = byte j, sign-extended — the order of `unpack4xI8`). */
const i8Lane = (words: Uint32Array<ArrayBuffer>, index: number): number =>
  ((words[index >> 2] >>> ((index & 3) * 8)) << 24) >> 24;

test.skipIf(adapter === null)(
  "real GPU: the conv3d w8a8 default geometry's output is bit-identical to a CPU reference over the harness's inputs (per-tap exact i32, fma flush per tap in tap order, final fma)",
  async () => {
    const target = CONV3D;
    const { dims } = target;
    const xq = resourceWords(target, "xq");
    const xscale = new Float32Array(resourceWords(target, "xscale").buffer);
    const w = resourceWords(target, "w");
    const wscale = new Float32Array(resourceWords(target, "wscale").buffer);
    const bias = new Float32Array(resourceWords(target, "bias").buffer);
    const taps = dims.kernelT * dims.kernelH * dims.kernelW;
    const pixelsOut = dims.timeOut * dims.heightOut * dims.widthOut;
    const expected = new Float32Array(dims.channelsOut * pixelsOut);
    for (let co = 0; co < dims.channelsOut; co += 1) {
      for (let n = 0; n < pixelsOut; n += 1) {
        const ot = Math.floor(n / (dims.heightOut * dims.widthOut));
        const oy = Math.floor(n / dims.widthOut) % dims.heightOut;
        const ox = n % dims.widthOut;
        let accf = 0;
        for (let tap = 0; tap < taps; tap += 1) {
          const kt = Math.floor(tap / (dims.kernelH * dims.kernelW));
          const kh = Math.floor(tap / dims.kernelW) % dims.kernelH;
          const kw = tap % dims.kernelW;
          const it = ot + kt - dims.paddingT;
          const iy = oy + kh - dims.paddingH;
          const ix = ox + kw - dims.paddingW;
          if (it < 0 || it >= dims.timeIn || iy < 0 || iy >= dims.heightIn) continue;
          if (ix < 0 || ix >= dims.widthIn) continue;
          const pixel = (it * dims.heightIn + iy) * dims.widthIn + ix;
          // The tap-major weight [Cout][taps][Cin] and the pixel-major activation [P][Cin]
          let acci = 0;
          for (let ci = 0; ci < dims.channelsIn; ci += 1) {
            acci +=
              i8Lane(xq, pixel * dims.channelsIn + ci) *
              i8Lane(w, (co * taps + tap) * dims.channelsIn + ci);
          }
          accf = fmaF32(Math.fround(acci), xscale[pixel], accf);
        }
        expected[co * pixelsOut + n] = fmaF32(accf, wscale[co], bias[co]);
      }
    }
    const reference = await digestOf(expected);
    const gpu = await acquireGpu(timestampQuery ? { gpuTiming: true } : {});
    const context = await createSweepContext(gpu, "deno-raw-tick");
    try {
      const { rows } = await sweepCase(context, target, [I8A8_64X64], { rounds: 1 });
      assertEquals(rows.length, 2);
      for (const row of rows) assertEquals(row.error, undefined, row.error);
      assertEquals(
        rows[0].outputSha256,
        reference,
        "the default geometry's output differs from the CPU reference (a binding, params or dispatch is wrong)",
      );
      assertEquals(rows[1].identicalToDefault, true);
    } finally {
      destroySweepContext(context);
      gpu.destroy();
    }
  },
);

test.skipIf(adapter === null)(
  "real GPU: the output of f32 matmul is bit-identical to a CPU reference (sequential multiply-add in K order, either uncontracted or fma-contracted)",
  async () => {
    // M / N / K all differ (a mix-up of m, n and k shows up as a dimension mismatch)
    await assertDenseMatchesReference({
      id: "test-matmul-cpu-m64-n48-k32",
      op: "matmul",
      m: 64,
      n: 48,
      k: 32,
      censusCount: 0,
      mirrorOf: "test",
      source: "test",
    });
  },
);

test.skipIf(adapter === null)(
  "real GPU: the output of f32 bmm (batch 3) is bit-identical to the CPU reference (a batch mix-up disagrees with the reference)",
  async () => {
    // The inputs are a pseudo-random sequence per resource, so the values differ per batch — a
    // wrong z offset (every batch reading matrix 0, overlapping write targets, unwritten entries
    // left at 0) does not match the reference
    await assertDenseMatchesReference({
      id: "test-bmm-cpu-b3-m64-n48-k32",
      op: "bmm",
      batch: 3,
      m: 64,
      n: 48,
      k: 32,
      censusCount: 1,
      source: "test",
    });
  },
);
