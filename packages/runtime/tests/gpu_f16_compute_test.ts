// The numeric contract of the f16 **compute** variants (ADR 0028 — `"f16"` of
// `SessionOptions.attentionCompute` / `linearCompute`). **A separate axis from f16 weight storage (ADR
// 0018)**: this one lowers the shared tiles to f16.
//
// ## The numeric contract this file pins: atol=0
//
// The kernel rounds to f16 exactly once, just before writing into the shared tile, and in the inner
// loop widens to f32 (the widening is exact) and accumulates in f32. Therefore
//
//     f16 variant output ≡ f32 variant output on inputs rounded to f16     … not one bit differs
//
// should hold, and there is structurally no room for a tolerance to absorb anything (a contract of
// the same quality as w8a8's atol=0).
//
// ## How the oracle is built (the host does only the rounding; the f32 reduction is left to the GPU)
//
// MUST: do not rewrite the f32 reduction in JS. WGSL may fuse `a + b * c` into an FFMA (a single
// rounding), and the implementation of `exp` is not specified either, so an f32 reference written on
// the CPU can split by **the driver's latitude**. That difference has nothing to do with the
// correctness of the f16 variant, so bringing it into the contract would break atol=0. Instead,
// **the f32 variant itself is the oracle, and the host does only the rounding ({@link roundToF16})**:
//
// | Target    | Measured                               | Oracle                                                              |
// | --------- | -------------------------------------- | ------------------------------------------------------------------- |
// | linear    | `linearCompute:"f16"` × plain x / w    | default × x / w **rounded to f16**                                  |
// | attention | `attentionCompute:"f16"` × plain q/k/v | ① bmm(f16(q·s), f16(k·s)ᵀ) → ② softmax(f16(S)) → ③ bmm(f16(P), f16(v)) |
//
// The 3 stages of attention are exactly the decomposed path before fusion (the path whose bit
// identity tests/gpu_attention_parity_test.ts already pins), and the host inserts **only the 3 points
// where rounding happens**. This rests directly on ADR 0023's invariant "② matches softmax passes
// ①② verbatim / ③'s A element is the same expression as the output of softmax pass ③" — if it
// breaks, this turns red.
//
// MUST: do not let the check become vacuous. Every case confirms that **the f16 variant's output
// differs from the f32 variant's** (an implementation in which no rounding ever happens would still
// match the oracle).
//
// No assets needed. SKIPs on an adapter that does not list `shader-f16` (the checks of the default
// f32 path are unaffected).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { f16BitsToF32, roundToF16 } from "../src/format/f16.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import { gemmRowsGeometry } from "../src/kernels/geometry-profile.ts";
import { linearKey } from "../src/kernels/linear.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { OpenedContainer } from "../src/format/container/open.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { f32ToF16Bits, quantizeF16 } from "./helpers/f16.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { quantizeI8 } from "./helpers/i8.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

// ---------------------------------------------------------------------------
// The rounding reference (GPU-independent)
// ---------------------------------------------------------------------------

test("roundToF16 matches an independently implemented RTNE encoder over the whole f32 input range (cross-check of the rounding reference)", () => {
  // MUST: compare 2 implementations. With only one, "both wrong the same way" cannot be detected
  // (tests/helpers/f16.ts is a hand-written bit-manipulation implementation; src/format/f16.ts is
  // Math.f16round).
  //
  // MUST: the inputs are aligned to **f32 values** (already `Math.fround`ed). The bit implementation
  // first drops a double to f32 and then rounds to f16, so passing a double not representable in f32
  // gives a **double rounding** difference (e.g. 65519.999 is 65504 as a double, but through f32 it is
  // a tie that goes to the even side = Inf). The kernel always rounds f32 values, so only the f32-input
  // side matters to the contract.
  const viaBits = (value: number): number => f16BitsToF32(f32ToF16Bits(value));
  const cases: readonly number[] = [
    0,
    -0,
    1,
    // The tie (1 + 2^-11) goes to the even side 1.0 / 1 + 3·2^-12 goes up
    1 + 2 ** -11,
    1 + 3 * 2 ** -12,
    -(1 + 2 ** -11),
    0.1,
    -0.1,
    1 / 3,
    65504, // the largest finite f16 value
    65519, // still falls to 65504
    65520, // tie -> even side (mantissa 0) = Inf
    65536,
    -65536,
    2 ** -14, // smallest normal
    2 ** -24, // smallest subnormal
    2 ** -25, // half -> 0 by round-to-even
    3 * 2 ** -25, // the upper side of a subnormal tie
    1e-9,
    Infinity,
    -Infinity,
  ].map(Math.fround);
  for (const value of cases) {
    assertEquals(
      Object.is(roundToF16(value), viaBits(value)),
      true,
      `roundToF16(${value}) = ${roundToF16(value)} / bit implementation = ${viaBits(value)}`,
    );
  }
  // Compare the whole picture too with 3 sweeps of different steps (rejects an implementation that is right only at the boundaries)
  for (let i = -20000; i < 20000; i += 7) {
    for (const step of [1e-4, 3.7e-2, 11.3]) {
      const value = Math.fround(i * step);
      assert(Object.is(roundToF16(value), viaBits(value)), `roundToF16(${value}) split`);
    }
  }
  assert(Number.isNaN(roundToF16(Number.NaN)));
});

// ---------------------------------------------------------------------------
// Real GPU (f16 compute variants)
// ---------------------------------------------------------------------------

const F16_GPU = GPU_AVAILABLE && SHADER_F16_AVAILABLE;

/** Deterministic data sequences (no random numbers). MUST: use steps not exactly representable in f16. */
const XS = (i: number): number => (((i * 7) % 23) - 11) * 0.1713 + 0.0271;
const WS = (i: number): number => (((i * 11) % 19) - 9) * 0.2317 - 0.0139;
const BS = (i: number): number => ((i % 5) - 2) * 0.4271;
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.1731;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.2317;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119;

/** Half scale (`√scale_factor` of torch's math decomp) — the value ADR 0023's contract prescribes. */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** The f32 bit pattern (a form in which both a ±0 difference and a last-ulp difference are visible). */
const bits = (data: Float32Array): Uint32Array =>
  new Uint32Array(data.buffer, data.byteOffset, data.length);

const assertBitEqual = (actual: Float32Array, expected: Float32Array, where: string): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  const a = bits(actual);
  const e = bits(expected);
  const mismatches: string[] = [];
  for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
    if (a[i] !== e[i]) mismatches.push(`[${i}] ${actual[i]} vs ${expected[i]}`);
  }
  assertEquals(mismatches, [], `${where}: the bit patterns differ (${mismatches.join(" / ")})`);
};

/** Confirms that 2 results **differ** (guards against vacuity: the rounding actually takes effect). */
const assertDiffers = (a: Float32Array, b: Float32Array, where: string): void => {
  const x = bits(a);
  const y = bits(b);
  let differ = false;
  for (let i = 0; i < x.length; i += 1) {
    if (x[i] !== y[i]) {
      differ = true;
      break;
    }
  }
  assert(
    differ,
    `${where}: the f16 variant's output does not differ from the f32 variant's by a single bit (the rounding has no effect)`,
  );
  assert(
    new Set(x).size > 1,
    `${where}: the output is constant (the comparison has become vacuous)`,
  );
};

const roundedCopy = (data: Float32Array): Float32Array<ArrayBuffer> =>
  Float32Array.from(data, roundToF16);

const roundedTensor = (tensor: FilledTensor): FilledTensor => ({
  dtype: "f32",
  shape: tensor.shape,
  data: roundedCopy(tensor.data as Float32Array),
});

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  /** The pipeline keys that ran (census — independent of timing). */
  readonly keys: readonly string[];
};

const runModel = async (
  gpu: GpuContext,
  opened: OpenedContainer,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
): Promise<RunResult> => {
  const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME, options);
  try {
    const outputs = await session.run(inputs);
    const tensor = outputs[opened.graphs[GRAPH_NAME].declaration.outputs[0]];
    return {
      output: tensor.data as Float32Array<ArrayBuffer>,
      keys: requireCensus(session.diagnostics().lastRunPipelines, "run").map((row) => row.key),
    };
  } finally {
    await session.dispose();
  }
};

const runGraph = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  inputs: Readonly<Record<string, Tensor>>,
  options: SessionOptions = {},
  tensors: readonly TensorInput[] = [],
): Promise<RunResult> => await runModel(gpu, await openModelBytes(graph, tensors), inputs, options);

// ---------------------------------------------------------------------------
// linear
// ---------------------------------------------------------------------------

/** A single `linear(x, w, b)` node (the weight is **a graph input** = the f32-storage path). */
const linearInputGraph = (m: number, n: number, k: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [
    { name: "x", dtype: "f32", shape: [m, k] },
    { name: "w", dtype: "f32", shape: [n, k] },
    { name: "b", dtype: "f32", shape: [n] },
  ],
  outputs: ["y"],
  initializers: {},
  values: { y: { dtype: "f32", shape: [m, n] } },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** A single `linear(x, w, b)` node (the weight is an f16-stored initializer = the pair f16 storage x f16 compute). */
const linearWf16Graph = (m: number, n: number, k: number): DeclarationJson => ({
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
});

/**
 * The choice of shapes (following the existing GEMM parity tests).
 *
 * MUST: have **both** the v4 path and the scalar path, and include one case that spans the 64 tiles in
 * both rows and columns. The parity and remainder of K are where zero fill happens at the edge of the
 * shared tile, so always include `k % 16 ≠ 0`.
 */
const LINEAR_SHAPES: readonly {
  readonly name: string;
  readonly m: number;
  readonly n: number;
  readonly k: number;
}[] = [
  // v4 (k%4 && n%4), spanning the 64 tiles in both m and n
  { name: "v4 m65 n68 k20", m: 65, n: 68, k: 20 },
  // The scalar variant (neither k nor n is a multiple of 4) + a K tile remainder
  { name: "scalar m70 n23 k37", m: 70, n: 23, k: 37 },
  // v4 spanning many K tiles (rounding error piles up in the reduction)
  { name: "v4 m9 n64 k132", m: 9, n: 64, k: 132 },
  // k is the odd number 1 (the K remainder zero fill is 15 words)
  { name: "scalar m5 n7 k1", m: 5, n: 7, k: 1 },
];

test.skipIf(!F16_GPU)(
  "the linear f16 compute variant matches 'the f32 variant with inputs rounded to f16' bit for bit (real GPU)",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of LINEAR_SHAPES) {
        const { name, m, n, k } = shape;
        const x = fill([m, k], XS);
        const w = fill([n, k], WS);
        const b = fill([n], BS);
        const graph = linearInputGraph(m, n, k);

        // Measured: plain inputs x f16 compute
        const actual = await runGraph(gpu, graph, { x, w, b }, { linearCompute: "f16" });
        // Oracle: inputs rounded to f16 x the default f32 compute (the bias stays f32 — not rounded)
        const oracle = await runGraph(gpu, graph, {
          x: roundedTensor(x),
          w: roundedTensor(w),
          b,
        });
        // Control: plain inputs x the default f32 compute (to confirm that the rounding takes effect)
        const plain = await runGraph(gpu, graph, { x, w, b });

        assertBitEqual(actual.output, oracle.output, `linear ${name}`);
        assertDiffers(actual.output, plain.output, `linear ${name}`);

        const v4 = k % 4 === 0 && n % 4 === 0;
        // MUST: pass **m** to the expected key too (the executor resolves the geometry from the row count
        // with linearKey(..., m, ...) — src/runtime/recipe-builder.ts). Leaving it out gives the spelling of
        // the default geometry, and the comparison would be against a key with a different geometry.
        assertEquals(
          actual.keys.filter(
            (key) =>
              key === linearKey("f32", v4, "f16", m, undefined, gemmRowsGeometry(profile, m)),
          ).length,
          1,
          `linear ${name}: the f16 variant's key did not run (${actual.keys.join(", ")})`,
        );
        assertEquals(
          actual.keys.filter(
            (key) =>
              key === linearKey("f32", v4, "f32", m, undefined, gemmRowsGeometry(profile, m)),
          ).length,
          0,
          `linear ${name}: an f32 variant's key remains`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!F16_GPU)(
  "the pair f16 weight storage x f16 compute round-trips identically (unpack2x16float values return exactly to f16)",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of LINEAR_SHAPES) {
        const { name, m, n, k } = shape;
        const x = fill([m, k], XS);
        const raw = Float32Array.from({ length: n * k }, (_, i) => WS(i));
        const weight = quantizeF16(raw);
        const bias = Float32Array.from({ length: n }, (_, i) => BS(i));
        const graph = linearWf16Graph(m, n, k);
        const tensors: readonly TensorInput[] = [
          {
            graph: GRAPH_NAME,
            initializer: "b",
            bytes: f32Bytes(bias),
            encoding: { codec: "f32" },
          },
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: weight.bytes,
            encoding: { codec: "f16" },
          },
        ];

        const actual = await runGraph(gpu, graph, { x }, { linearCompute: "f16" }, tensors);
        // The weights are already on the f16 grid, so only x is rounded (the round-trip identity claim itself)
        const oracle = await runGraph(gpu, graph, { x: roundedTensor(x) }, {}, tensors);
        const plain = await runGraph(gpu, graph, { x }, {}, tensors);

        assertBitEqual(actual.output, oracle.output, `linear wf16 ${name}`);
        assertDiffers(actual.output, plain.output, `linear wf16 ${name}`);

        const v4 = k % 4 === 0 && n % 4 === 0;
        assertEquals(
          actual.keys.filter(
            (key) =>
              key === linearKey("f16", v4, "f16", m, undefined, gemmRowsGeometry(profile, m)),
          ).length,
          1,
          `linear wf16 ${name}: the f16 compute key did not run`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Fused attention
// ---------------------------------------------------------------------------

type AttentionShape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/** MUST: the same set of shapes as the existing f32 parity test (folding of B and H, remainders, v4 selection). */
const ATTENTION_SHAPES: readonly AttentionShape[] = [
  { name: "all distinct B2 H3 M5 N11 D7", b: 2, h: 3, m: 5, n: 11, d: 7 },
  { name: "remainders B3 H1 M17 N19 D13", b: 3, h: 1, m: 17, n: 19, d: 13 },
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, m: 68, n: 20, d: 12 },
  { name: "mixed B2 H2 M9 N8 D6", b: 2, h: 2, m: 9, n: 8, d: 6 },
  { name: "DiT shape B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

const tensorOf = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

/**
 * The oracle of the 3-stage decomposed path. The host inserts only the points where rounding happens
 * (rounding sequences 1 / 4 of ADR 0028):
 *
 * 1. `f16(q·scale)` and `f16(k·scale)` — the tile fill of ①QK. **The scale multiplication is done in
 *    f32 first**, then rounded (the half-scale contract is ADR 0023). k is transposed on the host from
 *    `[n,d]` to `[d,n]` (the same layout as the kernel's "transpose in shared memory").
 * 2. `f16(S)` — the write-out of ①QK (S is handed over in f16).
 * 3. `f16(P)` and `f16(v)` — the tile fill of ③PV. `P = exp(S−m)·inv` is exactly the output of softmax
 *    pass ③ (reason 4 why it holds, ADR 0023).
 */
const attentionOracle = async (
  gpu: GpuContext,
  shape: AttentionShape,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scale: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;

  // ① q·scale to f16 / k·scale transposed to [heads, d, n] and to f16
  const qs = Float32Array.from(q, (value) => roundToF16(Math.fround(value * scale)));
  const kt = new Float32Array(heads * d * n);
  for (let head = 0; head < heads; head += 1) {
    for (let col = 0; col < n; col += 1) {
      for (let depth = 0; depth < d; depth += 1) {
        kt[(head * d + depth) * n + col] = roundToF16(
          Math.fround(k[(head * n + col) * d + depth] * scale),
        );
      }
    }
  }
  const scores = await runGraph(
    gpu,
    singleOpDeclaration(
      "bmm",
      [
        [heads, m, d],
        [heads, d, n],
      ],
      [[heads, m, n]],
    ),
    { x0: tensorOf([heads, m, d], qs), x1: tensorOf([heads, d, n], kt) },
  );

  // ② S to f16 (the rounding of ①'s write-out) -> the row statistics and P are produced by softmax as is
  const probs = await runGraph(
    gpu,
    singleOpDeclaration("softmax", [[heads, m, n]], [[heads, m, n]], { attrs: { dim: 2 } }),
    { x0: tensorOf([heads, m, n], roundedCopy(scores.output)) },
  );

  // ③ P and v to f16 (the rounding of ③'s tile fill)
  const out = await runGraph(
    gpu,
    singleOpDeclaration(
      "bmm",
      [
        [heads, m, n],
        [heads, n, d],
      ],
      [[heads, m, d]],
    ),
    {
      x0: tensorOf([heads, m, n], roundedCopy(probs.output)),
      x1: tensorOf([heads, n, d], roundedCopy(v as Float32Array<ArrayBuffer>)),
    },
  );
  return out.output;
};

test.skipIf(!F16_GPU)(
  "the fused attention f16 variant matches the decomposed path + 3 rounding points bit for bit (real GPU)",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of ATTENTION_SHAPES) {
        const { name, b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const graph = singleOpDeclaration(
          "attention",
          [q.shape, k.shape, v.shape],
          [[b, h, m, d]],
          {
            attrs: { scale },
          },
        );

        const actual = await runGraph(
          gpu,
          graph,
          { x0: q, x1: k, x2: v },
          {
            attentionCompute: "f16",
          },
        );
        const plain = await runGraph(gpu, graph, { x0: q, x1: k, x2: v });
        const oracle = await attentionOracle(
          gpu,
          shape,
          q.data as Float32Array,
          k.data as Float32Array,
          v.data as Float32Array,
          scale,
        );

        assertBitEqual(actual.output, oracle, `attention ${name}`);
        assertDiffers(actual.output, plain.output, `attention ${name}`);

        const qkV4 = d % 4 === 0 && n % 4 === 0;
        const pvV4 = n % 4 === 0 && d % 4 === 0;
        const running = new Set(actual.keys);
        // MUST: pass regcache to the expected key of ② row statistics too (the executor derives
        // `attentionStatsRegCache(n)` from the column count and puts it into the key). Leaving it out gives
        // a spelling without `:rc`, which always disagrees with the key that ran.
        const regCache = attentionStatsRegCache(n);
        for (const key of [
          attentionQkKey(qkV4, "f16", "f32", false, false, false, profile.attention.qk),
          attentionStatsKey("f16", "f32", regCache),
          attentionPvKey(pvV4, "f16", "f32", false, false, profile.attention.pv),
        ]) {
          assert(running.has(key), `attention ${name}: '${key}' did not run`);
        }
        // The 3 kernels switch **together** (the condition for the storage form of S to agree between writer and reader)
        for (const key of [
          attentionQkKey(qkV4, "f32", "f32", false, false, false, profile.attention.qk),
          attentionStatsKey("f32", "f32", regCache),
          attentionPvKey(pvV4, "f32", "f32", false, false, profile.attention.pv),
        ]) {
          assert(!running.has(key), `attention ${name}: f32 variant '${key}' remains`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// fail loudly (do not silently fall to f32)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "a request for f16 compute fails at Session build on a device without shader-f16 (real GPU)",
  async () => {
    // MUST: create a device that **does not request shaderF16** (the default acquireGpu).
    const gpu = await acquireGpu();
    try {
      assertEquals(
        gpu.shaderF16Enabled,
        false,
        "the default acquireGpu does not request shader-f16",
      );
      const opened = await openModelBytes(linearInputGraph(4, 8, 4), []);
      for (const options of [{ linearCompute: "f16" }, { attentionCompute: "f16" }] as const) {
        const error = await assertRejects(
          () => createSessionFromContainer(gpu, opened, GRAPH_NAME, options),
          ExecutionError,
        );
        assert(
          error.message.includes("acquireGpu({ shaderF16: true })"),
          `the guidance is insufficient: ${error.message}`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!F16_GPU)(
  "i8-resident weights x linearCompute 'f16' (w8a16) fail loudly (real GPU)",
  async () => {
    const m = 4;
    const n = 8;
    const k = 8;
    const raw = Float32Array.from({ length: n * k }, (_, i) => WS(i));
    const weight = quantizeI8(raw, [n, k], 0);
    const graph: DeclarationJson = {
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
    const tensors: readonly TensorInput[] = [
      {
        graph: GRAPH_NAME,
        initializer: "w",
        bytes: weight.bytes,
        // per-channel (row length = K is the group length, groups is 1). `rowAxis` is the channel axis 0 of
        // linear's weight (`WEIGHT_CHANNEL_AXES` in src/ops/names.ts).
        encoding: {
          codec: "int8-sym",
          groupSize: k,
          scale: { dtype: "f32", bytes: f32Bytes(weight.scale) },
        },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes(Float32Array.from({ length: n }, (_, i) => BS(i))),
        encoding: { codec: "f32" },
      },
    ];
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      const opened = await openModelBytes(graph, tensors);
      const x = fill([m, k], XS);
      // MUST: i8a8 and f32 pass (only the w8a16 pair is rejected)
      await runModel(gpu, opened, { x }, {});
      await runModel(gpu, opened, { x }, { linearCompute: "a8" });
      const error = await assertRejects(
        () => runModel(gpu, opened, { x }, { linearCompute: "f16" }),
        ExecutionError,
      );
      assert(error.message.includes("w8a16"), `the guidance is insufficient: ${error.message}`);
    } finally {
      gpu.destroy();
    }
  },
);
