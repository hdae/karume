// The execution path of the **f16 storage** variant of the attention score S (option γ, phase 1 —
// `SessionOptions.attentionScoreStorage`). Implementation = src/kernels/score-storage.ts (the
// source of truth for the parts) + src/kernels/gemm.ts / src/kernels/attention.ts /
// src/kernels/attention-i8a8.ts (insertion points) + src/runtime/executor.ts (eligibility and the
// allocation width of S). Design = docs/research/2026-08-04-intermediate-f16-design.md §3.4②.
//
// ## The numeric contract this file pins: atol=0
//
// s16 packs S into `array<u32>` with `pack2x16float`, 2 elements per word. **Rounding happens only
// once, at storage**, and the reader's `unpack2x16float` is exact, so
//
//     s16 output ≡ output of the f32 variant with S rounded to f16 on the host  … no bit differs
//
// should hold (the arithmetic stays f32, so it is an identity by definition — a stronger contract
// than the f16 **tile compute** of ADR 0028). There is structurally no room to absorb anything in a
// tolerance.
//
// ## How the oracle is built (the host only rounds; the GPU does the f32 reductions)
//
// MUST: do not rewrite the f32 reductions in JS (WGSL may fuse `a + b*c` into FFMA and the
// implementation of `exp` is not specified — the same reason as the oracle design of ADR 0028).
// Instead, **the decomposed path itself is the oracle, and the host does only the one rounding of
// S**:
//
//     ① bmm(q·scale, (k·scale)ᵀ) → S      … bit-identical to fused ①QK
//        (the path pinned by tests/gpu_attention_parity_test.ts)
//     ② softmax(S **rounded to f16**) → P   … same formulas and order as m / inv / P of fused ②③
//     ③ bmm(P, v) → O
//
// The only difference from the oracle of ADR 0028 is that **there is 1 rounding point instead of
// 3**. q / k / P / v stay raw = exactly the skeleton of option γ, "only the storage goes f16, the
// compute stays f32".
//
// MUST: do not make it tautological. Every case confirms **the s16 output differs from the f32
// variant** (an implementation where no rounding ever happens would also match the oracle).
//
// ## What this iteration deliberately does not verify
//
// - **The s16 variant of ② row statistics** has been added to the grid-stride degeneration harness
//   (tests/gpu_gridstride_test.ts) (the family of row-wise grid-stride kernels grew by one, so the
//   family's MUST applies). ① and ③ are tiled, a different category (their safety net is the fail
//   loudly of `tiledWorkgroups` and the poison-value injection of full-write).
// - s16 is **not wired** into the scalar path (`D % 4 != 0` or `N % 4 != 0`). Ineligible shapes
//   silently degrade to f32 storage, and the detector is the single pipeline-key check of the
//   "degradation in 2 directions" test below.
//
// ## Detection limits (measured by fault injection — 2026-08-04)
//
// **The even/odd selection of the scalar read `score_at` in ② row statistics cannot be detected by
// the numeric checks**. An injection that swaps the 2 branches of `select` (reading element `i^1`
// instead of `i`) leaves both this file and tests/gpu_gridstride_test.ts green — the eligibility
// condition imposes `N % 4 == 0`, so a row always starts at an even index and the swap is only **a
// permutation within each pair of a row** (`amax` is permutation-invariant, and `Σexp` also matched
// to the last ulp on these shapes). ADR 0018's "trap of taking even/odd from the relative index
// within a row" is, for s16, **made harmless** by the eligibility condition. The working detector
// is the **WGSL snapshot** (tests/fixtures/wgsl/attention_stats_s16.wgsl) alone, and that pins the
// intended structure; it is not a numeric check. The other 5 (the packing order of 2 words / the
// 2nd word of a quad read / `D % 4` of the eligibility / the key's word / silent expiry of the
// opt-in) are all caught red by this file or the E2E.
//
// No assets needed. **`shader-f16` is not needed either** (core WGSL's pack2x16float — the
// execution-side evidence that ADR 0030 decision 1 is intact). Environments without an adapter SKIP
// explicitly.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { tiledWorkgroups } from "../src/codegen/dispatch.ts";
import { roundToF16 } from "../src/format/f16.ts";
import { RunArena } from "../src/gpu/arena.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { PipelineCache } from "../src/gpu/pipeline-cache.ts";
import { SubmitScheduler } from "../src/gpu/submit.ts";
import {
  attentionPvKey,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsRegCache,
} from "../src/kernels/attention.ts";
import {
  ATTENTION_PV_V_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
} from "../src/kernels/attention-i8a8.ts";
import { defaultGemmGeometry, gemmTileM, gemmTileN } from "../src/kernels/gemm-geometry.ts";
import { defaultI8a8Geometry, i8a8TileM, i8a8TileN } from "../src/kernels/i8a8-geometry.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { referenceAttentionPvQuant } from "../src/reference/i8a8.ts";
import { stridedKey } from "../src/codegen/strided.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { f32ToF16Bits } from "./helpers/f16.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import {
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const STORAGE_IN = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const UNIFORM_IN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/**
 * The half scale (`√scale_factor` of torch's math decomposition) — the value per the contract of
 * ADR 0023.
 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** The f32 bits (a form in which both a ±0 difference and a 1-ulp difference are visible). */
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
  assertEquals(mismatches, [], `${where}: bits differ (${mismatches.join(" / ")})`);
};

/**
 * Confirms 2 results **differ** (guarding against a tautology: the rounding actually takes effect).
 */
const assertDiffers = (a: Float32Array, b: Float32Array, where: string): void => {
  const x = bits(a);
  const y = bits(b);
  assert(
    [...x].some((value, index) => value !== y[index]),
    `${where}: the s16 output does not differ by a single bit from f32 storage (the rounding has no effect)`,
  );
  assert(new Set(x).size > 1, `${where}: the output is constant (the comparison is a tautology)`);
};

const roundedCopy = (data: Float32Array): Float32Array<ArrayBuffer> =>
  Float32Array.from(data, roundToF16);

/** Packs S into the same storage form as the GPU (2 elements per word, like `pack2x16float`). */
const packScores = (scores: Float32Array): Uint32Array<ArrayBuffer> => {
  assertEquals(
    scores.length % 2,
    0,
    "s16-eligible shapes have N % 4 == 0, so the element count is always even",
  );
  const words = new Uint32Array(scores.length / 2);
  for (let i = 0; i < words.length; i += 1) {
    // MUST: the lower half is the even index (the x side of `pack2x16float(vec2(x, y))`). Swapping
    // it shifts the GPU's read by 1 element, and only the row edges go quietly wrong
    words[i] =
      (f32ToF16Bits(scores[2 * i]) & 0xffff) | ((f32ToF16Bits(scores[2 * i + 1]) & 0xffff) << 16);
  }
  return words;
};

// ---------------------------------------------------------------------------
// Shapes and data
// ---------------------------------------------------------------------------

type AttentionShape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Eligible shapes (`D % 4 == 0 && N % 4 == 0` = the condition for the writer ①QK to take the v4
 * path).
 *
 * MUST: include shapes with **B·H ≥ 2 and a different pattern per head** (the only detector of the
 * batch base — detection limit 5 of ADR 0030).
 * MUST: include shapes where **m / n cross the tile edge 64** (the detector that the write-out of
 * an edge tile fits exactly into 2 words).
 * MUST: include **N % 16 != 0** (a remainder of ③PV's K tile 16 = the shape where the zero fill on
 * the S side matters).
 */
const S16_SHAPES: readonly AttentionShape[] = [
  // B·H = 6 / m spans 2 tiles of 64 / N=68 is k4=17, 5 K tiles (remainder of 1 quad)
  { name: "B2 H3 M65 N68 D20", b: 2, h: 3, m: 65, n: 68, d: 20 },
  // N=20 leaves a K tile remainder (k4=5), D=12 gives 3 column quads
  { name: "B3 H1 M17 N20 D12", b: 3, h: 1, m: 17, n: 20, d: 12 },
  // The smallest shape fitting in 1 tile (the path where every guard stays true)
  { name: "B1 H2 M3 N8 D4", b: 1, h: 2, m: 3, n: 8, d: 4 },
  // n spans 2 tiles of 64 (a row of S splits across several workgroups)
  { name: "B2 H2 M9 N132 D8", b: 2, h: 2, m: 9, n: 132, d: 8 },
  // A measured shape (a shrunk DiT self-attention)
  { name: "DiT shape B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

/**
 * Ineligible shapes (s16 silently degrades to f32 storage). Both directions (D side / N side) are
 * covered.
 */
const DEGRADED_SHAPES: readonly (AttentionShape & { readonly why: string })[] = [
  { name: "D%4!=0 B1 H2 M17 N20 D13", why: "D", b: 1, h: 2, m: 17, n: 20, d: 13 },
  { name: "N%4!=0 B1 H2 M17 N19 D20", why: "N", b: 1, h: 2, m: 17, n: 19, d: 20 },
];

/**
 * Deterministic data sequences (no random numbers).
 *
 * MUST: use steps not exactly representable in f16 (with only representable values no rounding ever
 * happens and the oracle match becomes a tautology). The periods are mutually incommensurate, so
 * they do not align with the head boundary (`m·d` elements) and **the pattern changes per head**.
 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.1731 + 0.0271;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.2317 - 0.0139;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.3119 + 0.0413;

const tensorOf = (shape: readonly number[], data: Float32Array<ArrayBuffer>): Tensor => ({
  dtype: "f32",
  shape,
  data,
});

// ---------------------------------------------------------------------------
// Execution through the Session path (collecting the diagnostics' pipeline keys too)
// ---------------------------------------------------------------------------

type RunResult = {
  readonly output: Float32Array<ArrayBuffer>;
  readonly entries: readonly { readonly key: string; readonly dispatchCount: number }[];
  /**
   * The intermediate-buffer record of the last run (the only observation point where halving S is
   * visible).
   */
  readonly peakTransientBytes: number;
};

const runAttention = async (
  gpu: GpuContext,
  shape: AttentionShape,
  options: SessionOptions,
): Promise<RunResult> => {
  const { b, h, m, n, d } = shape;
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
    const diagnostics = session.diagnostics();
    return {
      output: outputs["y"].data as Float32Array<ArrayBuffer>,
      entries: requireCensus(diagnostics.lastRunPipelines, "attention"),
      peakTransientBytes: diagnostics.lastRun?.peakTransientBytes ?? 0,
    };
  } finally {
    await session.dispose();
  }
};

const runGraph = async (
  gpu: GpuContext,
  graph: ReturnType<typeof singleOpDeclaration>,
  inputs: Readonly<Record<string, Tensor | FilledTensor>>,
): Promise<Float32Array<ArrayBuffer>> => {
  const session = await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME);
  try {
    const outputs = await session.run(inputs);
    return outputs["y"].data as Float32Array<ArrayBuffer>;
  } finally {
    await session.dispose();
  }
};

/**
 * The oracle of the 3-stage decomposed path. **Rounding happens only at the one point of S** (the
 * design at the top).
 *
 * `q·scale` / `k·scale` stay f32 (the half-scale contract is ADR 0023). k is transposed `[n,d]` →
 * `[d,n]` on the host (the same layout as the kernel's "transpose in shared memory").
 */
const s16Oracle = async (
  gpu: GpuContext,
  shape: AttentionShape,
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  scale: number,
): Promise<Float32Array<ArrayBuffer>> => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;
  const qs = Float32Array.from(q, (value) => Math.fround(value * scale));
  const kt = new Float32Array(heads * d * n);
  for (let head = 0; head < heads; head += 1) {
    for (let col = 0; col < n; col += 1) {
      for (let depth = 0; depth < d; depth += 1) {
        kt[(head * d + depth) * n + col] = Math.fround(k[(head * n + col) * d + depth] * scale);
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
  // The only rounding (corresponds to the write-out of ①QK = pack2x16float)
  const probs = await runGraph(
    gpu,
    singleOpDeclaration("softmax", [[heads, m, n]], [[heads, m, n]], { attrs: { dim: 2 } }),
    { x0: tensorOf([heads, m, n], roundedCopy(scores)) },
  );
  return await runGraph(
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
      x0: tensorOf([heads, m, n], probs),
      x1: tensorOf([heads, n, d], v as Float32Array<ArrayBuffer>),
    },
  );
};

// ---------------------------------------------------------------------------
// (1) The direct check that the storage rounding = pack2x16float is RTE
// ---------------------------------------------------------------------------

/**
 * f32 values that target the rounding boundaries (f16 ties). The same sequence as the reference
 * test of `Math.f16round` (tests/gpu_f16_compute_test.ts) — that one compares WGSL's `f16()`, this
 * one `pack2x16float`.
 *
 * MUST: apply `Math.fround` (passing a double not representable in f32 shows a **double rounding**
 * difference).
 */
const TIE_VALUES: readonly number[] = [
  0,
  1,
  1 + 2 ** -11, // tie → even side 1.0
  1 + 3 * 2 ** -12, // the upper side of a tie
  -(1 + 2 ** -11),
  0.1,
  -0.1,
  1 / 3,
  65504, // the largest finite f16
  65519, // still drops to 65504
  65520, // tie → even side (mantissa 0) = Inf
  -65536,
  2 ** -14, // smallest normal
  2 ** -24, // smallest subnormal
  2 ** -25, // half → 0 by round-to-even
  3 * 2 ** -25, // the upper side of a subnormal tie
  5 * 2 ** -25,
  1e-9,
  1024 + 0.5,
  1024 + 1.5,
  2048 + 1,
  2048 + 3,
].map(Math.fround);

/**
 * f16 16-bit pattern → f32 (a readback independent of production {@link "../src/format/f16.ts"}).
 */
const decodeHalf = (pattern: number): number => {
  const sign = (pattern & 0x8000) !== 0 ? -1 : 1;
  const exponent = (pattern >> 10) & 0x1f;
  const mantissa = pattern & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 0x1f) return mantissa === 0 ? sign * Infinity : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
};

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

/**
 * The dispatch count of `quantize_rows` (the small-D variants split keys per geometry, so they are
 * grouped by prefix).
 */
const quantizeRowsDispatches = (byKey: ReadonlyMap<string, number>): number =>
  [...byKey]
    .filter(([key]) => key.startsWith(QUANTIZE_ROWS_KEY))
    .reduce((sum, [, count]) => sum + count, 0);

test.skipIf(!GPU_AVAILABLE)(
  "the s16 write-out of ①QK is bit-identical to Math.f16round (the direct check that pack2x16float rounds RTE, real GPU)",
  async () => {
    // With `scale = 1` and k made of unit vectors, `S[m][n] = q[m][n % D]` comes out **exactly**
    // (only products with 0 and additions of 0 mix in, so the dot product has zero rounding). Each
    // element of S is therefore one of the tie values above, and all that remains is whether the
    // written 16 bits match Math.f16round.
    const m = 32;
    const n = 8;
    const d = 4;
    const q = new Float32Array(m * d);
    for (let row = 0; row < m; row += 1) {
      for (let col = 0; col < d; col += 1) {
        q[row * d + col] = TIE_VALUES[(row * d + col) % TIE_VALUES.length];
      }
    }
    const k = new Float32Array(n * d);
    for (let col = 0; col < n; col += 1) k[col * d + (col % d)] = 1;

    const gpu = await acquireGpu();
    // MUST: run on **the default device that does not request shader-f16** (s16 is core WGSL)
    assertEquals(gpu.shaderF16Enabled, false, "the default acquireGpu does not request shader-f16");
    const scheduler = new SubmitScheduler(gpu);
    const cache = new PipelineCache(gpu.device);
    const arena = new RunArena(gpu.device, () => scheduler.flush());
    try {
      const host = (source: Float32Array<ArrayBuffer>): GPUBuffer => {
        const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
        gpu.device.queue.writeBuffer(buffer, 0, source);
        return buffer;
      };
      const params = arena.allocHostWritten(16, UNIFORM_IN);
      gpu.device.queue.writeBuffer(params, 0, attentionQkParams(m, n, d, 1));
      const scores = arena.allocRegion(m * n * 2);
      const key = attentionQkKey(true, "f32", "f16");
      const { pipeline, layout } = await cache.get(key, attentionQkWgsl(true, "f32", "f16"));
      const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
      scheduler.dispatch(
        pipeline,
        gpu.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: host(q) } },
            { binding: 2, resource: { buffer: host(k) } },
            { binding: 3, resource: { buffer: scores } },
          ],
        }),
        [
          // ①QK of fused attention does not go through the row-count bucket = fixed default
          // geometry (the same derivation as the executor)
          tiledWorkgroups(n, gemmTileN(defaultGemmGeometry()), limit, "s16 rounding check ①QK"),
          tiledWorkgroups(m, gemmTileM(defaultGemmGeometry()), limit, "s16 rounding check ①QK"),
          1,
        ],
        key,
      );
      await scheduler.flush();

      const raw = await readbackBytes(gpu.device, scores, m * n * 2);
      const words = new Uint32Array(raw.buffer, 0, (m * n) / 2);
      let rounded = 0;
      for (let index = 0; index < m * n; index += 1) {
        const source = q[Math.floor(index / n) * d + ((index % n) % d)];
        const word = words[index >> 1];
        const got = decodeHalf((index & 1) === 1 ? word >>> 16 : word & 0xffff);
        const want = roundToF16(source);
        assert(
          Object.is(got, want) || (Number.isNaN(got) && Number.isNaN(want)),
          `S[${index}]: GPU ${got} / Math.f16round(${source}) = ${want}`,
        );
        if (!Object.is(got, source)) rounded += 1;
      }
      // Tautology guard: if not a single element is rounded, "equal" verifies nothing
      assert(rounded > 0, `no rounding happened at all (${rounded} / ${m * n})`);
    } finally {
      await arena.destroy();
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (2) atol=0 (the s16 variant ≡ the f32 variant with S rounded to f16 on the host)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "attentionScoreStorage:'f16' is bit-for-bit equal to \"the f32 variant with S rounded to f16\" (v4 / tile edge / K remainder / B·H≥2, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of S16_SHAPES) {
        const { name, b, h, m, n, d } = shape;
        assert(attentionScoreUsesF16(d, n), `${name}: should be an eligible shape`);
        const scale = halfScale(d);
        const actual = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        const plain = await runAttention(gpu, shape, {});
        const oracle = await s16Oracle(
          gpu,
          shape,
          fill([b, h, m, d], QUERY).data as Float32Array,
          fill([b, h, n, d], KEY).data as Float32Array,
          fill([b, h, n, d], VALUE).data as Float32Array,
          scale,
        );

        assertBitEqual(actual.output, oracle, `attention s16 ${name}`);
        assertDiffers(actual.output, plain.output, `attention s16 ${name}`);
        // Values differ per head (the very premise for the batch-base detector to hold)
        if (b * h > 1) {
          const head0 = actual.output.slice(0, m * d);
          const head1 = actual.output.slice(m * d, 2 * m * d);
          assert(
            [...head0].some((value, index) => value !== head1[index]),
            `${name}: the outputs of head 0 and head 1 are identical (a mixed-up base would not show in the values)`,
          );
        }
        // The allocation of S is halved (the core value of option γ — visible in RunArena's
        // measurement).
        // MUST: do not wrap the difference comparison in a condition (peakTransientBytes is always
        // positive on a real run regardless of the timing feature — dropping to 0 is itself a
        // regression of the diagnostics, and the check would vanish silently).
        assert(plain.peakTransientBytes > 0, `${name}: the peakTransient observation is a no-op`);
        assertEquals(
          plain.peakTransientBytes - actual.peakTransientBytes,
          b * h * m * n * 2,
          `${name}: the peakTransient difference does not match halving S`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). Sees through keys that the 3 kernels switched to s16
 * **together**.
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too). An empty census FAILs unconditionally (guarding with "if entries is empty, look at
 * nothing" turns green without a single check running — `requireCensus` in
 * tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "attentionScoreStorage:'f16' switches ①QK / ② row statistics / ③PV to s16 together (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      for (const shape of S16_SHAPES) {
        const { name, n } = shape;
        const actual = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        assert(
          actual.entries.length > 0,
          `${name}: the breakdown is empty (the key check is a no-op)`,
        );
        const running = new Map(actual.entries.map((entry) => [entry.key, entry.dispatchCount]));
        // The 3 kernels switch **together** (the condition for writer and readers to agree on the
        // storage form of S)
        for (const key of [
          attentionQkKey(true, "f32", "f16", false, false, false, profile.attention.qk),
          attentionStatsKey("f32", "f16", attentionStatsRegCache(n)),
          attentionPvKey(true, "f32", "f16", false, false, profile.attention.pv),
        ]) {
          assertEquals(running.get(key), 1, `${name}: '${key}' did not run exactly once`);
        }
        for (const key of [
          attentionQkKey(true, "f32", "f32", false, false, false, profile.attention.qk),
          attentionStatsKey("f32", "f32", attentionStatsRegCache(n)),
          attentionPvKey(true, "f32", "f32", false, false, profile.attention.pv),
        ]) {
          assertEquals(running.has(key), false, `${name}: the f32-storage '${key}' is still there`);
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (3) The 2 directions of eligibility (eligible → s16 keys / ineligible → f32 keys + bit identity)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "ineligible shapes (D%4 / N%4) of attentionScoreStorage:'f16' silently degrade to f32 storage (the detector is the key check, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of DEGRADED_SHAPES) {
        const { name, d, n } = shape;
        assertEquals(attentionScoreUsesF16(d, n), false, `${name}: should be an ineligible shape`);
        const requested = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        const plain = await runAttention(gpu, shape, {});
        // The degradation is **silent** (values are bit-identical to the f32 path — no rounding
        // ever happens)
        assertBitEqual(
          requested.output,
          plain.output,
          `${name}: the degradation does not match the f32 path`,
        );
        assert(new Set(bits(plain.output)).size > 1, `${name}: the output is constant`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The only detector of the degradation of ineligible shapes (the key check). The reason for the
 * unconditional assert is the same as the census above.
 */
test.skipIf(!GPU_AVAILABLE)(
  "ineligible shapes emit no s16 key at all and run the f32-storage ② row statistics (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of DEGRADED_SHAPES) {
        const { name, why, n } = shape;
        const requested = await runAttention(gpu, shape, { attentionScoreStorage: "f16" });
        assert(
          requested.entries.length > 0,
          `${name}: the breakdown is empty (the key check is a no-op)`,
        );
        const keys = new Set(requested.entries.map((entry) => entry.key));
        for (const key of [...keys]) {
          assertEquals(key.endsWith(":s16"), false, `${name} (${why} side): an s16 key ran`);
        }
        assertEquals(
          keys.has(attentionStatsKey("f32", "f32", attentionStatsRegCache(n))),
          true,
          `${name}: ② row statistics with f32 storage`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (4) fail loudly (the combination with c16 is not silently interpreted as either one)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "attentionScoreStorage 'f16' × attentionCompute 'f16' fails at Session construction (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const graph = singleOpDeclaration(
        "attention",
        [
          [1, 1, 4, 4],
          [1, 1, 4, 4],
          [1, 1, 4, 4],
        ],
        [[1, 1, 4, 4]],
        { attrs: { scale: halfScale(4) } },
      );
      const opened = await openGraphModel(graph);
      const error = await assertRejects(
        () =>
          createSessionFromContainer(gpu, opened, GRAPH_NAME, {
            attentionCompute: "f16",
            attentionScoreStorage: "f16",
          }),
        ExecutionError,
      );
      assert(
        error.message.includes("cannot be given together"),
        `the guidance is insufficient: ${error.message}`,
      );
      // MUST: **s16 alone passes on a device without shader-f16** (the same discipline as ADR 0030
      // decision 1 — if this fails, the very premise of option γ, "independent of the feature",
      // collapses)
      const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME, {
        attentionScoreStorage: "f16",
      });
      await session.dispose();
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (5) The main combination (i8a8 × s16) — a key check that they are orthogonal
// ---------------------------------------------------------------------------

const A8_S16_SHAPE: AttentionShape = { name: "B2 H3 M65 N68 D20", b: 2, h: 3, m: 65, n: 68, d: 20 };

test.skipIf(!GPU_AVAILABLE)(
  "attentionCompute:'a8' × attentionScoreStorage:'f16' are orthogonal and both take effect (no shader-f16 needed, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const shape = A8_S16_SHAPE;
      const both = await runAttention(gpu, shape, {
        attentionCompute: "a8",
        attentionScoreStorage: "f16",
      });
      const i8a8Only = await runAttention(gpu, shape, { attentionCompute: "a8" });
      // The rounding of S adds one step, so the values move (the numeric-side evidence that the
      // opt-in takes effect)
      assertDiffers(both.output, i8a8Only.output, "i8a8 × s16");
      // S is halved (the same single allocation works with i8a8 too). Unconditional for the same
      // reason as above.
      assert(i8a8Only.peakTransientBytes > 0, "the peakTransient observation is a no-op");
      assertEquals(
        i8a8Only.peakTransientBytes - both.peakTransientBytes,
        shape.b * shape.h * shape.m * shape.n * 2,
        "the peakTransient difference does not match halving S",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The only detector of orthogonality (3 axes taking effect together). The reason for the
 * unconditional assert is the same as the census above.
 */
test.skipIf(!GPU_AVAILABLE)(
  "attentionCompute:'a8' × attentionScoreStorage:'f16' runs ①③ as i8a8 and only ② with the s16 key (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const shape = A8_S16_SHAPE;
      const both = await runAttention(gpu, shape, {
        attentionCompute: "a8",
        attentionScoreStorage: "f16",
      });
      assert(both.entries.length > 0, "the breakdown is empty (the key check is a no-op)");
      const byKey = new Map(both.entries.map((entry) => [entry.key, entry.dispatchCount]));
      // ①QK / ③PV stay i8a8, only S has f16 storage (3 axes take effect together)
      assertEquals(byKey.get(attentionQkI8a8Key(true, true, "f16", profile.i8a8.attentionQk)), 1);
      assertEquals(byKey.get(attentionPvI8a8Key(true, true, "f16", profile.i8a8.attentionPv)), 1);
      // ② row statistics stays f32 compute and reads s16
      assertEquals(byKey.get(attentionStatsKey("f32", "f16", attentionStatsRegCache(shape.n))), 1);
      // No f32-storage variant remains
      for (const key of [
        attentionQkI8a8Key(true, true, "f32", profile.i8a8.attentionQk),
        attentionPvI8a8Key(true, true, "f32", profile.i8a8.attentionPv),
        attentionStatsKey("f32", "f32", attentionStatsRegCache(shape.n)),
      ]) {
        assertEquals(byKey.has(key), false, `the f32-storage '${key}' is still there`);
      }
      // The i8a8 prologue (quantization of q / k / Vᵀ and the permute of Vᵀ) is unaffected by s16
      assertEquals(quantizeRowsDispatches(byKey), 3, "quantize_rows is not the 3 of q / k / v");
      assertEquals(
        byKey.get(stridedKey({ dtype: "f32" })),
        1,
        "the permute of Vᵀ is not exactly 1",
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// (6) qP of ③PV i8a8 re-derived with s16 (ADR 0030's ±1 step + mismatch-rate check, s16 version)
// ---------------------------------------------------------------------------

/**
 * The P̃ grid and `1/127` (**used only for the arithmetic of the observation** — the reference
 * side's constants live in src/reference/i8a8.ts).
 */
const P_ABS_MAX = 127;
const INV_P_ABS_MAX = Math.fround(1 / P_ABS_MAX);

type QpShape = {
  readonly name: string;
  readonly batch: number;
  readonly m: number;
  readonly n: number;
};

/** MUST: `N % 4 == 0` (the eligibility of ③PV i8a8) and B·H ≥ 2 (the batch-base detector). */
const QP_SHAPES: readonly QpShape[] = [
  { name: "B·H6 M65 N68", batch: 6, m: 65, n: 68 },
  { name: "B·H2 M17 N20", batch: 2, m: 17, n: 20 },
];

/** Raw values of S (deterministic). The pattern changes per row, so P̃ spreads from 0 to 127. */
const scoreAt = (b: number, row: number, col: number): number => {
  const index = col * 7 + row * 3 + b * 5;
  return ((index % 23) - 11) * 0.6373 + ((index * 13) % 11) * 0.1237 - 0.4291;
};

/**
 * Runs ③PV i8a8 **as a single raw dispatch** (the test supplies S and the row statistics; the
 * quantized Vᵀ is built on the host). Making V an identity matrix per head **structurally** gives
 * `vq = 127·I` / `vs = 1/127`, so `O[m,j] = f32(127·qP[m,j]) · (prow · vs[j])` becomes **a reader
 * of qP**.
 *
 * MUST: the premise (vq diagonal 127, vs 1/127) is pinned by tests/gpu_attention_pv_i8a8_test.ts
 * through an actual run of `quantize_rows`. Here those values are rebuilt on the host and only the
 * storage form of S changes.
 */
const observeQpS16 = async (
  gpu: GpuContext,
  shape: QpShape,
  packed: Uint32Array<ArrayBuffer>,
  stats: Float32Array<ArrayBuffer>,
  rowInv: Float32Array<ArrayBuffer>,
): Promise<Int8Array<ArrayBuffer>> => {
  const { batch, m, n } = shape;
  const scheduler = new SubmitScheduler(gpu);
  const cache = new PipelineCache(gpu.device);
  const arena = new RunArena(gpu.device, () => scheduler.flush());
  const limit = gpu.limits.maxComputeWorkgroupsPerDimension;
  try {
    const host = (source: ArrayBufferView<ArrayBuffer>): GPUBuffer => {
      const buffer = arena.allocHostWritten(Math.max(4, source.byteLength), STORAGE_IN);
      gpu.device.queue.writeBuffer(buffer, 0, source);
      return buffer;
    };
    // Vᵀ = the quantized identity matrix (row = (b, d), N consecutive i8 packed 4 per word)
    const vq = new Int8Array(batch * n * n);
    const vs = new Float32Array(batch * n);
    for (let head = 0; head < batch; head += 1) {
      for (let row = 0; row < n; row += 1) {
        vq[(head * n + row) * n + row] = P_ABS_MAX;
        vs[head * n + row] = INV_P_ABS_MAX;
      }
    }
    const out = arena.allocRegion(Math.max(4, batch * m * n * 4));
    const params = arena.allocHostWritten(16, UNIFORM_IN);
    gpu.device.queue.writeBuffer(params, 0, attentionPvI8a8Params(m, n, n));
    const v4 = attentionPvI8a8UsesVec4(n);
    const pvGeometry = defaultI8a8Geometry("attention_pv");
    const key = attentionPvI8a8Key(v4, true, "f16");
    const { pipeline, layout } = await cache.get(key, attentionPvI8a8Wgsl(v4, true, "f16"));
    scheduler.dispatch(
      pipeline,
      gpu.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: { buffer: host(packed) } },
          { binding: 2, resource: { buffer: host(new Uint8Array(vq.buffer)) } },
          { binding: 3, resource: { buffer: host(stats) } },
          { binding: 4, resource: { buffer: out } },
          { binding: ATTENTION_PV_V_SCALE_BINDING, resource: { buffer: host(vs) } },
        ],
      }),
      [
        // ③PV uses the i8a8-side geometry (the default where tiling N = D into 1 tile wins —
        // distinct from ①QK)
        tiledWorkgroups(n, i8a8TileN(pvGeometry), limit, `${shape.name} ③PV i8a8 s16`),
        tiledWorkgroups(m, i8a8TileM(pvGeometry), limit, `${shape.name} ③PV i8a8 s16`),
        tiledWorkgroups(batch, 1, limit, `${shape.name} ③PV i8a8 s16`),
      ],
      key,
    );
    await scheduler.flush();
    const bytes = await readbackBytes(gpu.device, out, batch * m * n * 4);
    const o = new Float32Array(bytes.buffer, 0, batch * m * n);
    const qp = new Int8Array(batch * m * n);
    for (let head = 0; head < batch; head += 1) {
      for (let row = 0; row < m; row += 1) {
        const prow = Math.fround(rowInv[head * m + row] * INV_P_ABS_MAX);
        for (let col = 0; col < n; col += 1) {
          const combined = Math.fround(prow * INV_P_ABS_MAX);
          const value = o[(head * m + row) * n + col] / (P_ABS_MAX * combined);
          const nearest = Math.round(value);
          // MUST: dividing with the premise broken shifts the observed values quietly (the check
          // becomes a tautology)
          assert(
            Math.abs(value - nearest) < 1e-3 && nearest >= 0 && nearest <= P_ABS_MAX,
            `the qP readout is not an integer [${head},${row},${col}]: ${value}`,
          );
          qp[(head * m + row) * n + col] = nearest;
        }
      }
    }
    return qp;
  } finally {
    await arena.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  '③PV i8a8 × s16: qP always matches within ±1 step the TS reference built from "S rounded to f16" (real GPU)',
  async () => {
    const gpu = await acquireGpu();
    let total = 0;
    let mismatched = 0;
    try {
      for (const shape of QP_SHAPES) {
        const { batch, m, n } = shape;
        // MUST: the reference also reads the **rounded** S (the storage form of S changed, so
        // unless the row statistics and qP are built from the rounded values it becomes "a rounding
        // detector", not "the s16 check")
        const raw = new Float32Array(batch * m * n);
        for (let head = 0; head < batch; head += 1) {
          for (let row = 0; row < m; row += 1) {
            for (let col = 0; col < n; col += 1) {
              raw[(head * m + row) * n + col] = Math.fround(scoreAt(head, row, col));
            }
          }
        }
        const rounded = roundedCopy(raw);
        const stats = new Float32Array(batch * m * 2);
        const rowInv = new Float32Array(batch * m);
        const rowMax = new Float32Array(batch * m);
        for (let row = 0; row < batch * m; row += 1) {
          let max = Number.NEGATIVE_INFINITY;
          for (let col = 0; col < n; col += 1) max = Math.max(max, rounded[row * n + col]);
          let sum = 0;
          for (let col = 0; col < n; col += 1) {
            sum = Math.fround(sum + Math.fround(Math.exp(rounded[row * n + col] - max)));
          }
          const inv = Math.fround(1 / sum);
          stats[row * 2] = max;
          stats[row * 2 + 1] = inv;
          rowInv[row] = inv;
          rowMax[row] = max;
        }
        const qp = await observeQpS16(gpu, shape, packScores(rounded), stats, rowInv);
        const expected = referenceAttentionPvQuant(rounded, rowMax, batch * m, n);
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
        // Tautology guard: if qP does not reach both ends, the grid is not verified at all
        assert([...qp].includes(P_ABS_MAX), `${shape.name}: qP has no 127`);
        assert([...qp].includes(0), `${shape.name}: qP has no 0`);
      }
    } finally {
      gpu.destroy();
    }
    // A record of the measurement (the value of ADR 0030's check re-derived with s16). An upper
    // bound is set too so it is not a tautology
    const rate = mismatched / total;
    console.log(
      `GPU/TS mismatch rate of qP (S stored as f16): ${mismatched} / ${total} = ${(
        rate * 100
      ).toFixed(4)}% (all ±1 step)`,
    );
    assert(
      rate < 0.02,
      `the qP mismatch rate ${rate} exceeded 2% (large for an exp implementation difference)`,
    );
  },
);

// ---------------------------------------------------------------------------
// (7) Propagation of NaN rows (non-finite handling does not move when the storage form changes)
// ---------------------------------------------------------------------------

test.skipIf(!GPU_AVAILABLE)(
  "with s16 too, a row containing NaN propagates NaN as is (the same rows become NaN as with f32 storage, real GPU)",
  async () => {
    const shape: AttentionShape = { name: "NaN row B1 H2 M8 N8 D4", b: 1, h: 2, m: 8, n: 8, d: 4 };
    const { b, h, m, n, d } = shape;
    const q = fill([b, h, m, d], QUERY);
    const k = fill([b, h, n, d], KEY);
    const v = fill([b, h, n, d], VALUE);
    // Make only row 3 of head 1 NaN (one whole row of S becomes NaN)
    (q.data as Float32Array)[(1 * m + 3) * d] = Number.NaN;
    const graph = singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
      attrs: { scale: halfScale(d) },
    });
    const gpu = await acquireGpu();
    try {
      const run = async (options: SessionOptions): Promise<Float32Array> => {
        const session = await createSessionFromContainer(
          gpu,
          await openGraphModel(graph),
          GRAPH_NAME,
          options,
        );
        try {
          return (await session.run({ x0: q, x1: k, x2: v }))["y"].data as Float32Array;
        } finally {
          await session.dispose();
        }
      };
      const s16 = await run({ attentionScoreStorage: "f16" });
      const plain = await run({});
      const nanRows = (data: Float32Array): readonly number[] =>
        Array.from({ length: b * h * m }, (_, row) => row).filter((row) =>
          [...data.slice(row * d, (row + 1) * d)].some(Number.isNaN),
        );
      // The contaminated rows are exactly the same as with f32 storage (pack2x16float does not
      // squash NaN into 0 or a finite value)
      assertEquals(nanRows(s16), nanRows(plain), "the set of NaN rows differs from f32 storage");
      assertEquals(nanRows(s16), [1 * m + 3], "NaN is not confined to 1 row");
      // No uncontaminated row is NaN (if every row were wiped out, the comparison would be a
      // tautology)
      assert([...s16].filter(Number.isNaN).length === d, "NaN spreads beyond one row");
    } finally {
      gpu.destroy();
    }
  },
);
