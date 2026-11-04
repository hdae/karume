// **Bit-identity check** of fused attention (ADR 0023). The most important test of this work.
//
// For the same random inputs,
//
//   ① the decomposed path of existing GPU ops (mul → permute → mul → bmm → softmax → expand → bmm)
//   ② a single `attention` op node
//
// are run on **the same real GPU** and the outputs are compared as **f32 bits**. The point is the
// exact equality of `Uint32Array`, not allclose: a change in the rounding sequence that hides
// inside a tolerance (e.g. a "full scale" rewrite that multiplies the scale once after the dot
// product) can only be detected here.
//
// MUST: do not make it tautological. ① and ② run **separate graphs** in separate Sessions (calling
// the same kernel twice would always match). The node sequence of the decomposed path is built in
// the same order with the same ops as the measured Anima DiT graph (#60–#74 of the design recon
// §1.1).
//
// MUST: B / H / M / N / D run over several shapes (with remainders). The kernel folds B and H into
// one batch axis, so with B=1 alone a mixed-up axis does not show in the values.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { attentionPvKey, attentionQkKey } from "../src/kernels/attention.ts";
import { deviceGeometryProfile } from "./helpers/geometry.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/**
 * The half scale (`√scale_factor` of torch's math decomposition). The value derived from D per the
 * contract.
 */
const halfScale = (depth: number): number => Math.fround(Math.sqrt(1 / Math.sqrt(depth)));

/** Deterministic data sequences (no random numbers — failures would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

type Shape = {
  readonly name: string;
  readonly b: number;
  readonly h: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

/**
 * The graph of the decomposed path (torch SDPA's math decomposition mapped to IR).
 *
 * `sc` is an input passing the half scale as 1 element and corresponds to `const.6953fe58410d6c34`
 * of the real graph (the right-aligned broadcast of `mul` is the same for an initializer and an
 * input). The identity `expand` is included so that it goes through the same copy as the 56 on the
 * P side of the real graph, reproducing "the one buffer the fusion removes".
 *
 * `masked` is the form with an additive mask (ADR 0023 revision). The point is to write it as **a
 * single `add` after S is materialized**; the fused side does the same addition in the write-out
 * epilogue of ①QK (same rounding position and count = the basis of the bit identity).
 */
const decomposedGraph = (shape: Shape, masked = false): DeclarationJson => {
  const { b, h, m, n, d } = shape;
  const heads = b * h;
  return {
    format: "karume-ir",
    version: 2,
    requires: {
      ops: ["mul", "permute", "reshape", "bmm", "softmax", "expand", ...(masked ? ["add"] : [])],
    },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [b, h, m, d] },
      { name: "k", dtype: "f32", shape: [b, h, n, d] },
      { name: "v", dtype: "f32", shape: [b, h, n, d] },
      { name: "sc", dtype: "f32", shape: [1] },
      ...(masked ? [{ name: "mk", dtype: "f32" as const, shape: [1, 1, m, n] }] : []),
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      qs: { dtype: "f32", shape: [b, h, m, d] },
      kt: { dtype: "f32", shape: [b, h, d, n] },
      kts: { dtype: "f32", shape: [b, h, d, n] },
      qs3: { dtype: "f32", shape: [heads, m, d] },
      kts3: { dtype: "f32", shape: [heads, d, n] },
      scores3: { dtype: "f32", shape: [heads, m, n] },
      scores4: { dtype: "f32", shape: [b, h, m, n] },
      ...(masked ? { scoresMasked: { dtype: "f32" as const, shape: [b, h, m, n] } } : {}),
      probs: { dtype: "f32", shape: [b, h, m, n] },
      probsExpanded: { dtype: "f32", shape: [b, h, m, n] },
      probs3: { dtype: "f32", shape: [heads, m, n] },
      v3: { dtype: "f32", shape: [heads, n, d] },
      out3: { dtype: "f32", shape: [heads, m, d] },
      y: { dtype: "f32", shape: [b, h, m, d] },
    },
    nodes: [
      // #60 mul q × scale (the q side of the half scale)
      { op: "mul", ins: ["q", "sc"], outs: ["qs"], attrs: {} },
      // #61 permute k → kᵀ
      { op: "permute", ins: ["k"], outs: ["kt"], attrs: { dims: [0, 1, 3, 2] } },
      // #62 mul kᵀ × scale (the k side of the half scale — the same constant)
      { op: "mul", ins: ["kt"], outs: ["kts"], attrs: {} },
      { op: "reshape", ins: ["qs"], outs: ["qs3"], attrs: {} },
      { op: "reshape", ins: ["kts"], outs: ["kts3"], attrs: {} },
      // #67 bmm → S
      { op: "bmm", ins: ["qs3", "kts3"], outs: ["scores3"], attrs: {} },
      { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
      // Additive mask (one addition after S is materialized — [1,1,M,N] spreads to B·H by
      // right-aligned broadcast)
      ...(masked ? [{ op: "add", ins: ["scores4", "mk"], outs: ["scoresMasked"], attrs: {} }] : []),
      // #69 softmax dim=-1
      {
        op: "softmax",
        ins: [masked ? "scoresMasked" : "scores4"],
        outs: ["probs"],
        attrs: {
          dim: 3,
        },
      },
      // #70 identity expand (a full copy of P — the one buffer the fusion removes)
      { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
      { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
      { op: "reshape", ins: ["v"], outs: ["v3"], attrs: {} },
      // #74 bmm → O
      { op: "bmm", ins: ["probs3", "v3"], outs: ["out3"], attrs: {} },
      { op: "reshape", ins: ["out3"], outs: ["y"], attrs: {} },
    ],
  };
};

/**
 * Both `mul`s of the graph above take the same scale input (the ins of the k side are added
 * afterwards).
 */
const withScaleInput = (graph: DeclarationJson): DeclarationJson => {
  for (const node of graph.nodes) {
    if (node.op === "mul" && node.ins.length === 1) node.ins.push("sc");
  }
  return graph;
};

const run = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<Tensor> => {
  const session = await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME);
  try {
    return (await session.run(inputs))["y"];
  } finally {
    await session.dispose();
  }
};

/** The f32 bits (a form in which both `0.0` vs `-0.0` and a 1-ulp difference are visible). */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const SHAPES: readonly Shape[] = [
  // A cross-attention shape where B / H / M / N / D all differ (a mixed-up axis shows in the
  // values)
  { name: "all-distinct B2 H3 M5 N11 D7", b: 2, h: 3, m: 5, n: 11, d: 7 },
  // Remainders on every tile (M % 64 ≠ 0 / N % 64 ≠ 0 / D % 4 ≠ 0 = scalar variant)
  { name: "remainders B3 H1 M17 N19 D13", b: 3, h: 1, m: 17, n: 19, d: 13 },
  // The v4 path (① needs D%4 && N%4, ③ needs N%4 && D%4) spanning 2 row tiles
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, m: 68, n: 20, d: 12 },
  // A shape where ① is v4 and ③ drops to scalar (pins that ① and ③ pick their variants differently)
  { name: "mixed B2 H2 M9 N8 D6", b: 2, h: 2, m: 9, n: 8, d: 6 },
  // N > 256 = a shape that hits epc ≥ 2 of the regcache variant of ② attention_stats on a real GPU
  // (epc = ceil(N / 256) — `attentionStatsRegCache`, src/kernels/attention.ts:147-150). With epc=1
  // shapes alone, the fold order over 2 or more statically unrolled register slots does not show in
  // the values.
  { name: "regcache epc2 B1 H2 M64 N512 D128", b: 1, h: 2, m: 64, n: 512, d: 128 },
  // The same regcache but with a remainder (N % 256 ≠ 0 partially fills the last slot) and the
  // scalar variant (N % 4 ≠ 0)
  { name: "regcache epc3 B1 H2 M17 N598 D64", b: 1, h: 2, m: 17, n: 598, d: 64 },
  // A shrunk DiT self-attention shape (D=128, M/N both multiples of 64 = the real model's path)
  { name: "DiT shape B1 H4 M64 N64 D128", b: 1, h: 4, m: 64, n: 64, d: 128 },
];

/**
 * The measured band mask (built like EmbeddingGemma's bidirectional sliding window).
 *
 * MUST: **every row has at least one unmasked column** (if a whole row is masked, amax becomes −inf
 * and `exp(−inf − (−inf))` becomes NaN — the fused and decomposed sides produce the same NaN, so
 * the bit comparison passes, but it verifies nothing). A symmetric band always leaves its centre
 * unmasked, so this condition holds by construction.
 */
const bandMask = (
  m: number,
  n: number,
  width: number,
  blocked: number,
): Float32Array<ArrayBuffer> => {
  const data = new Float32Array(m * n);
  for (let row = 0; row < m; row += 1) {
    // Even with M ≠ N (cross-attention shape), centre on "the column corresponding to the row"
    const center = Math.floor((row * n) / m);
    for (let col = 0; col < n; col += 1) {
      data[row * n + col] = Math.abs(col - center) <= width ? 0 : blocked;
    }
  }
  return data;
};

test.skipIf(!GPU_AVAILABLE)(
  "a single attention node's output is **bit-for-bit equal** to the decomposed path (bmm/softmax/bmm) (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const sc = fill([1], () => scale);

        const fused = await run(
          gpu,
          singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
            attrs: { scale },
          }),
          { x0: q, x1: k, x2: v },
        );
        const split = await run(gpu, withScaleInput(decomposedGraph(shape)), { q, k, v, sc });

        assertEquals(fused.shape, split.shape, shape.name);
        const a = bits(fused);
        const c = bits(split);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
          if (a[i] !== c[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches,
          [],
          `${shape.name}: bits differ from the decomposed path (first mismatches: ${mismatches
            .map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`)
            .join(" / ")})`,
        );
        // Tautology guard: if the output is all one value, "equal" verifies nothing
        assert(
          new Set(a).size > 1,
          `${shape.name}: the output is constant (bit equality is a tautology)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **Bit-identity check of the additive-mask version** (ADR 0023 revision). The fused side only adds
 * `S' = fl(S + mask[m·N+n])` in the write-out epilogue of ①QK, so the rounding position and count
 * match the decomposed side's `bmm` (S materialized) → `add` (mask) — it stops matching only when
 * an index (row/column, batch base) is mixed up, and that always shows in the bit comparison
 * instead of hiding inside a tolerance.
 *
 * The mask values are **a mix of 0 and large negative values** (the measured band mask); only the
 * last shape goes through `-Infinity` (the form in which the exporter folds in −inf).
 */
test.skipIf(!GPU_AVAILABLE)(
  "attention output with an additive mask is bit-for-bit equal to the decomposed path (bmm → add → softmax → bmm) (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const [index, shape] of SHAPES.entries()) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const scale = halfScale(d);
        const sc = fill([1], () => scale);
        // Only the last shape is −inf (the rest is the large negative value of the measured mask —
        // the same magnitude as masked_fill's fill value)
        const blocked = index === SHAPES.length - 1 ? -Infinity : -3.4028234663852886e38;
        const mask = {
          dtype: "f32" as const,
          shape: [1, 1, m, n],
          data: bandMask(m, n, 2, blocked),
        };
        const where = `${shape.name} (blocked=${blocked})`;

        const fused = await run(
          gpu,
          singleOpDeclaration(
            "attention",
            [q.shape, k.shape, v.shape, mask.shape],
            [[b, h, m, d]],
            {
              attrs: { scale },
            },
          ),
          { x0: q, x1: k, x2: v, x3: mask },
        );
        const split = await run(gpu, withScaleInput(decomposedGraph(shape, true)), {
          q,
          k,
          v,
          sc,
          mk: mask,
        });

        assertEquals(fused.shape, split.shape, where);
        const a = bits(fused);
        const c = bits(split);
        const mismatches: number[] = [];
        for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
          if (a[i] !== c[i]) mismatches.push(i);
        }
        assertEquals(
          mismatches,
          [],
          `${where}: bits differ from the decomposed path (first mismatches: ${mismatches
            .map((i) => `[${i}] ${fused.data[i]} vs ${split.data[i]}`)
            .join(" / ")})`,
        );
        // Tautology guard ①: if the output is constant, "equal" verifies nothing
        assert(
          new Set(a).size > 1,
          `${where}: the output is constant (bit equality is a tautology)`,
        );
        // Tautology guard ②: the mask affects the result (values **differ** from the same input
        // without the mask). A mixed-up index changes the values too, but what this guard protects
        // against is "the mask being ignored entirely".
        const unmasked = await run(
          gpu,
          singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
            attrs: { scale },
          }),
          { x0: q, x1: k, x2: v },
        );
        const plain = bits(unmasked);
        assert(
          a.some((word, i) => word !== plain[i]),
          `${where}: the outputs with and without the mask are identical (the mask has no effect)`,
        );
        // No NaN appears (the band mask leaves an unmasked column in every row — the measured
        // confirmation of the MUST above)
        assert(
          fused.data.every((value) => Number.isFinite(value)),
          `${where}: non-finite values in the output (a whole row is masked)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Wiring of the Session path (ADR 0023 revision). The mask adds only **one binding of ①QK and one
 * word of the key**; there stay 3 dispatches (② and ③ do not know the mask exists).
 *
 * The i8a8 ①QK is a separate kernel without an epilogue, so the combination **fails loudly**
 * (silently degrading to f32 would be the silent "asked for i8a8 but it has no effect", and
 * dropping the mask would break the values).
 */
const MASK_WIRING_SHAPE = { b: 1, h: 2, m: 8, n: 8, d: 8 };

/**
 * The graph and inputs of the masked 1-node attention (the wiring test and the census hit the same
 * shape).
 */
const maskedAttention = (): {
  graph: ReturnType<typeof singleOpDeclaration>;
  inputs: Record<string, Tensor>;
} => {
  const { b, h, m, n, d } = MASK_WIRING_SHAPE;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, h, n, d], KEY);
  const v = fill([b, h, n, d], VALUE);
  const mask = { dtype: "f32" as const, shape: [1, 1, m, n], data: bandMask(m, n, 2, -1e30) };
  const graph = singleOpDeclaration(
    "attention",
    [q.shape, k.shape, v.shape, mask.shape],
    [[b, h, m, d]],
    { attrs: { scale: halfScale(d) } },
  );
  return { graph, inputs: { x0: q, x1: k, x2: v, x3: mask } };
};

test.skipIf(!GPU_AVAILABLE)(
  "an additive mask cannot be combined with i8a8 (no silent degradation, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const { graph, inputs } = maskedAttention();
      // The combination with i8a8 fails loudly (no degradation)
      const i8a8 = await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME, {
        attentionCompute: "a8",
      });
      try {
        await assertRejects(() => i8a8.run(inputs), ExecutionError, "attentionCompute 'a8'");
      } finally {
        await i8a8.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). Whether the mask variant actually ran and whether dispatches
 * did not grow is visible only from the keys and the counts (dropping the mask entirely still
 * passes the shapes).
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too). An empty census FAILs unconditionally (guarding with "if entries is empty, look at
 * nothing" turns green without a single check running — `requireCensus` in
 * tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "an additive mask adds only to the key and bindings of ①QK (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const { graph, inputs } = maskedAttention();
      const session = await createSessionFromContainer(
        gpu,
        await openGraphModel(graph),
        GRAPH_NAME,
      );
      try {
        await session.run(inputs);
        const entries = requireCensus(session.diagnostics().lastRunPipelines, "masked attention");
        const keys = entries.map((entry) => entry.key);
        // 1 node = still 3 dispatches (the mask adds no dispatch)
        assertEquals(
          entries.reduce((sum, entry) => sum + entry.dispatchCount, 0),
          3,
        );
        assertEquals(
          keys.includes(
            attentionQkKey(true, "f32", "f32", true, false, false, profile.attention.qk),
          ),
          true,
          "mask variant",
        );
        assertEquals(
          keys.includes(
            attentionQkKey(true, "f32", "f32", false, false, false, profile.attention.qk),
          ),
          false,
          "the unmasked ①QK is still there",
        );
        // ② and ③ have the same keys as without the mask (same generated code too = pinned by
        // snapshots)
        assertEquals(
          keys.includes(attentionPvKey(true, "f32", "f32", false, false, profile.attention.pv)),
          true,
          "③PV does not change with the mask",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The rejection of mask × `attentionCompute: "a8"` **does not depend on D / N being multiples of
 * 4** (ADR 0023 / the "unconditional fail loudly" of docs/limitations.md). Deciding the rejection
 * by per-stage eligibility (① needs `D % 4`, ③ needs `N % 4`) lets only the `D % 4 != 0` shapes
 * through, which then run as **a mix of the f32 ①QK and the i8a8 ③PV** — even if the values are
 * correct, the contract "cannot be combined" is broken.
 *
 * Tautology guard: **the same shape without the mask** runs with i8a8 (pins that the cause of the
 * rejection is the mask, not "that shape does not run with i8a8").
 */
test.skipIf(!GPU_AVAILABLE)(
  "mask × attentionCompute 'a8' is rejected consistently regardless of D%4 / N%4 (real GPU)",
  async () => {
    // The 3 combinations of ①QK eligibility (D%4) × ③PV eligibility (N%4). true×true was rejected
    // already; the other 2 slipped through (false×true = mixed / false×false = silent degradation
    // to f32 on every stage).
    const shapes: readonly Shape[] = [
      { name: "both ①③ eligible D8 N8", b: 1, h: 2, m: 8, n: 8, d: 8 },
      { name: "only ③ eligible D13 N20", b: 1, h: 2, m: 8, n: 20, d: 13 },
      { name: "both ①③ ineligible D13 N19", b: 1, h: 2, m: 8, n: 19, d: 13 },
    ];
    const gpu = await acquireGpu();
    try {
      for (const shape of shapes) {
        const { b, h, m, n, d } = shape;
        const q = fill([b, h, m, d], QUERY);
        const k = fill([b, h, n, d], KEY);
        const v = fill([b, h, n, d], VALUE);
        const mask = { dtype: "f32" as const, shape: [1, 1, m, n], data: bandMask(m, n, 2, -1e30) };
        const attrs = { scale: halfScale(d) };

        const masked = await createSessionFromContainer(
          gpu,
          await openGraphModel(
            singleOpDeclaration(
              "attention",
              [q.shape, k.shape, v.shape, mask.shape],
              [[b, h, m, d]],
              {
                attrs,
              },
            ),
          ),
          GRAPH_NAME,
          { attentionCompute: "a8" },
        );
        try {
          // Same wording and type as the rejection on the D%4==0 path (no separate rejection per
          // branch)
          await assertRejects(
            () => masked.run({ x0: q, x1: k, x2: v, x3: mask }),
            ExecutionError,
            "attention with an additive mask cannot be combined with attentionCompute 'a8'",
          );
        } finally {
          await masked.dispose();
        }

        const plain = await createSessionFromContainer(
          gpu,
          await openGraphModel(
            singleOpDeclaration("attention", [q.shape, k.shape, v.shape], [[b, h, m, d]], {
              attrs,
            }),
          ),
          GRAPH_NAME,
          { attentionCompute: "a8" },
        );
        try {
          const y = (await plain.run({ x0: q, x1: k, x2: v }))["y"];
          assertEquals(y.shape, [b, h, m, d], shape.name);
          assert(
            y.data.every((value) => Number.isFinite(value)),
            `${shape.name}: non-finite values in the non-masked i8a8 output`,
          );
        } finally {
          await plain.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
