// Real-GPU comparison of the memory-requirement estimator (ADR 0070 decision 5) (the GPU-independent
// checks are in estimate_test.ts).
//
// Exact equality is claimed for the 2 categories the diagnostics measure (compressed residency,
// expansion) and the state capacity, and for the slot tables of small graphs whose every step the
// estimate replays as execution allocates it: a states-form attention, a single fused attention op
// node (its in-node temporaries — S, the row statistics, the i8a8 quantization temporaries — come
// from the same pure functions as execution, ADR 0124 decision 7), an attention with absorbed
// permutes, and one window of each fusion rule, built-in or external (the estimate walks the same
// fusion plan's step list and maps each fused step from the same device-free bookkeeping,
// fused-step-layout.ts). A plain node is planned as one dispatch reading its inputs and writing its
// outputs, so a graph whose kernels bind narrower sets can be placed differently; and the node
// temporaries the estimator's unaccounted field names are not counted. Without an adapter the tests
// SKIP explicitly.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu, LIMIT_CAPS } from "../src/gpu/device.ts";
import { attentionOnlineI8a8Eligible } from "../src/kernels/attention-online-i8a8.ts";
import type { PreparedModel, SessionOptions } from "../src/runtime/executor.ts";
import { planRowBlocks } from "../src/runtime/fusion.ts";
import { planStateAttention } from "../src/runtime/state-attention-plan.ts";
import { attentionDeviceFacts, attentionRoute } from "../src/runtime/attention-route.ts";
import {
  bothScenarios,
  decomposedAttentionGraph,
  f16Zeros,
  openGraph,
  stateAttentionGraph,
  stateModel,
  weight,
} from "./helpers/estimate-graphs.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import {
  a8LinearAfterDeadRegion,
  a8LinearCase,
  FUSED_WINDOW_CASES,
} from "./helpers/fused-window-graphs.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI8 } from "./helpers/i8.ts";

/**
 * The only measured check that the formulas for S / the row statistics come from the same source
 * (`planStateAttention`) as recipe-builder.
 *
 * No fusion applies to this graph at all (nodes that touch states do not grab a window — ADR 0067
 * decision 5b), so `workspaceBytes` **equals exactly** the slot table's total bytes =
 * `planBacking.residentBytes`. Both share the formula itself (= it cannot split here), but if **the
 * inputs passed** to the shared function (`B·H`, the window, where the capacity comes from) or the
 * rules by which the execution phase places the returned bytes and reuses size classes drift from
 * the estimator's copy, this splits without any exception.
 *
 * It is called for the 2 variants full / sliding (the 2 tests below) — `colCap` is the only field
 * whose formula differs by variant, so with only one of them the other side of the branch would stay
 * unchecked.
 *
 * Passing `limitCap` restricts the device's `maxStorageBufferBindingSize` and **splits the rows into
 * several blocks** (the estimator has no entry for `ROW_BLOCK_SPLIT`, so the only levers to steer the
 * block count are the shape and the limit). The 2 rules that apply only with several blocks — pool
 * reuse across blocks, and 2 size classes when a block one row narrower from the remainder is mixed
 * in — are never exercised by a call without the restriction.
 * MUST: when restricted, pass `expectedBlocks` to pin the block count first (going green on a shape
 * that fell to 1 block would not count as seeing "agreement over several blocks").
 */
const assertPlanBackingMatchesEstimate = async (variant: {
  readonly capacity: number;
  readonly window?: number;
  readonly heads?: number;
  readonly chunkLength?: number;
  readonly limitCap?: number;
  readonly expectedBlocks?: number;
}): Promise<void> => {
  const heads = variant.heads ?? 4;
  const chunkLength = variant.chunkLength ?? 4;
  const gpu = await acquireGpu(
    variant.limitCap === undefined
      ? {}
      : { [LIMIT_CAPS]: { maxStorageBufferBindingSize: variant.limitCap } },
  );
  try {
    const limit = gpu.limits.maxStorageBufferBindingSize;
    if (variant.limitCap !== undefined) {
      assertEquals(
        limit,
        variant.limitCap,
        "requiredLimits is not restricted (the check would be vacuous)",
      );
    }
    if (variant.expectedBlocks !== undefined) {
      // Pin the block count with the very pure function that execution and the estimate share (makes the premise visible).
      const blocks = planStateAttention(
        {
          batchHeads: heads,
          chunkRows: chunkLength,
          capacity: variant.capacity,
          window: variant.window,
        },
        limit,
      ).blocks;
      assertEquals(blocks.length, variant.expectedBlocks, "number of row blocks");
    }
    const model = openGraph(stateAttentionGraph(variant.window, heads));
    const generation = { chunkLength, bindings: { C: variant.capacity } };
    const { prefill } = bothScenarios(
      model.estimate({ generation, maxStorageBufferBindingSize: limit }),
    );
    const session = await model.createContainerSession(gpu);
    try {
      const context = await session.createGenerationContext(generation);
      try {
        // The slot backing is built on the second run with the same signature (the first run takes the arena path).
        for (let step = 0; step < 2; step += 1) {
          await session.run(
            {
              q: fill([1, heads, chunkLength, 8], (i) => ((i % 5) - 2) / 4),
              k: fill([1, 2, chunkLength, 8], (i) => ((i % 3) - 1) / 4),
              v: fill([1, 2, chunkLength, 8], (i) => ((i % 7) - 3) / 4),
            },
            {},
            { context, queryLength: chunkLength },
          );
        }
      } finally {
        await context.dispose();
      }
      assertEquals(session.diagnostics().planBacking.residentBytes, prefill.workspaceBytes);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a states-form attention exactly match the execution plan's slot table (full variant, real GPU)",
  () => assertPlanBackingMatchesEstimate({ capacity: 16 }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a states-form attention exactly match the execution plan's slot table (sliding variant, real GPU)",
  () => assertPlanBackingMatchesEstimate({ capacity: 8, window: 8 }),
);

/**
 * Agreement of the 2 implementations with **several** row blocks. With H=64 / Hkv=2 / D=8 / C=16, one
 * row of S is 64·16·4 = 4096B while a state slot is 2·16·8·4 = 1024B, so restricting the limit to
 * 8192B gives the shape "the slots can be bound but 4 rows of S cannot", and the rows always split
 * into blocks.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a states-form attention exactly match the slot table with several blocks too (full variant, real GPU)",
  () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 4,
      limitCap: 8192,
      expectedBlocks: 2,
    }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a states-form attention exactly match the slot table with a remainder block too (real GPU)",
  () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      chunkLength: 5,
      // The smallest restriction under which the output o [1,64,5,8] = 10240B fits the binding limit (S has 4096B per row -> 2 rows x 3 blocks)
      limitCap: 10240,
      expectedBlocks: 3,
    }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a states-form attention exactly match the slot table with several blocks too (sliding variant, real GPU)",
  () =>
    assertPlanBackingMatchesEstimate({
      capacity: 16,
      heads: 64,
      window: 8,
      chunkLength: 4,
      // A restriction under which the output o [1,64,4,8] = 8192B fits the binding limit (S has 64×11×4 = 2816B per row -> 2 rows x 2 blocks)
      limitCap: 8192,
      expectedBlocks: 2,
    }),
);

/**
 * Dimensions of the decomposed attention. One row of scores = `heads·keys·4` = 512B, the full width
 * = 32KiB, and q / kᵀ / v / y are 4096B each (within the binding limit under every restriction below
 * — only the row blocks of the scores split).
 */
const DECOMPOSED = { heads: 2, queries: 64, keys: 64, headDim: 8 } as const;

/**
 * The measured check that the total intermediates of a row-block attention (ADR 0060) window exactly
 * match the execution plan's slot table (ADR 0093 addendum 2026-10-10).
 *
 * The estimate calls the same fusion plan (`planFusions`) as the execution phase, replaces the window
 * with the fused allocation program and passes it to the same packer. The allocation program is a
 * copy of recipe-builder's bookkeeping (the outputs' allocation, the temporaries' lifetimes, the reads
 * and writes taken from the WGSL storage declarations, the release of external inputs), so if the copy
 * drifts from the execution phase, this splits without any exception. Every value other than the
 * graph output `y` is inside the window, so the slot table is decided by a single fused step (no
 * approximation from outside the window mixes in).
 *
 * `limitCap` restricts the binding limit and splits the rows into 1 / several / uneven blocks (with 1
 * block the hand-out across blocks is not exercised; without uneven blocks the 2 size classes of
 * temporaries with different widths are not).
 * MUST: pin the block count first with `expectedBlocks` (going green with an unexpected block count
 * would not count as seeing the intended shape).
 */
const assertRowBlockAttentionMatchesEstimate = async (variant: {
  readonly limitCap?: number;
  readonly expectedBlocks: number;
}): Promise<void> => {
  const { heads, queries, keys, headDim } = DECOMPOSED;
  const gpu = await acquireGpu(
    variant.limitCap === undefined
      ? {}
      : { [LIMIT_CAPS]: { maxStorageBufferBindingSize: variant.limitCap } },
  );
  try {
    const limit = gpu.limits.maxStorageBufferBindingSize;
    if (variant.limitCap !== undefined) {
      assertEquals(
        limit,
        variant.limitCap,
        "requiredLimits is not restricted (the check would be vacuous)",
      );
    }
    // Pin the block count with the very pure function that execution and the estimate share (makes the premise visible).
    assertEquals(
      planRowBlocks(queries, heads * keys * 4, limit).length,
      variant.expectedBlocks,
      "number of row blocks",
    );
    const model = openGraph(decomposedAttentionGraph(DECOMPOSED));
    const { scenarios } = model.estimate({ maxStorageBufferBindingSize: limit });
    assertEquals(
      scenarios.map((scenario) => scenario.name),
      ["run"],
    );
    const session = await model.createContainerSession(gpu);
    try {
      // The slot backing is built on the second run with the same signature (the first run takes the arena path).
      for (let step = 0; step < 2; step += 1) {
        await session.run({
          q: fill([heads, queries, headDim], (i) => ((i % 5) - 2) / 4),
          kt: fill([heads, headDim, keys], (i) => ((i % 3) - 1) / 4),
          mask: fill([1, 1, 1, keys], (i) => (i % 4 === 3 ? -1 : 0)),
          v: fill([1, heads, keys, headDim], (i) => ((i % 7) - 3) / 4),
        });
      }
      assertEquals(session.diagnostics().planBacking.residentBytes, scenarios[0].workspaceBytes);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a row-block attention exactly match the slot table with 1 block (real GPU)",
  () => assertRowBlockAttentionMatchesEstimate({ expectedBlocks: 1 }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a row-block attention exactly match the slot table with several blocks too (real GPU)",
  () => assertRowBlockAttentionMatchesEstimate({ limitCap: 8192, expectedBlocks: 4 }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a row-block attention exactly match the slot table with uneven blocks too (real GPU)",
  () => assertRowBlockAttentionMatchesEstimate({ limitCap: 6144, expectedBlocks: 6 }),
);

/**
 * The shape of a graph of one fused attention op node (no states field — ADR 0023). A shrunk DiT
 * self-attention; D % 4 == 0 and N % 4 == 0, so with a8 both ①③ are i8a8 and s16 is eligible too
 * (all the quantization temporaries and permutes appear).
 */
const FUSED = { b: 1, h: 4, m: 64, n: 64, d: 128 } as const;

/**
 * The measured check that the estimate counts the fused attention op's in-node temporaries the same
 * way as execution (ADR 0124 decision 7).
 *
 * The graph is a single attention node (no other fusion and no alias apply — the same condition as
 * the states form above), so `workspaceBytes` **equals exactly** the slot table's total bytes =
 * `planBacking.residentBytes`. The estimate copies the allocation and dispatch sequence of execution
 * (`buildAttention`), so if any of the temporaries' lifetimes, the read / write directions or the size
 * of the quantization temporaries drifts from the copy, this splits without any exception. The shapes
 * under a budget (`attentionRowBlockMaxBytes`) also check that the block count is derived the same way
 * as in execution.
 * MUST: the shapes under a budget pin the block count first (`expectedBlocks` — going green on a shape
 * that fell to 1 block would not count as seeing "agreement over several blocks"). The block count is
 * observed by a census that does not depend on timing (the number of dispatches of the row window's
 * ①QK key).
 */
const assertFusedAttentionMatchesEstimate = async (variant: {
  readonly options: SessionOptions;
  readonly mask?: boolean;
  readonly expectedBlocks: number;
  /** The online kernel (ADR 0132): the expected dispatch count (= row windows); the estimate then needs the device facts. */
  readonly expectedOnlineWindows?: number;
  /** Acquire with subgroups "available" (the online kernel's sg32 form on devices that have them). */
  readonly subgroups?: "available";
  /** The graph shape (default {@link FUSED}). */
  readonly shape?: {
    readonly b: number;
    readonly h: number;
    readonly m: number;
    readonly n: number;
    readonly d: number;
  };
}): Promise<void> => {
  const { b, h, m, n, d } = variant.shape ?? FUSED;
  const shapes = [
    [b, h, m, d],
    [b, h, n, d],
    [b, h, n, d],
    ...(variant.mask ? [[1, 1, m, n]] : []),
  ];
  const model = openGraph(
    singleOpDeclaration("attention", shapes, [[b, h, m, d]], {
      attrs: { scale: 1 / Math.sqrt(d) },
    }),
  );
  const gpu = await acquireGpu(
    variant.subgroups === undefined ? {} : { subgroups: variant.subgroups },
  );
  try {
    const { scenarios } = model.estimate({
      maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
      maxBufferSize: gpu.limits.maxBufferSize,
      ...(variant.expectedOnlineWindows === undefined
        ? {}
        : { attentionDevice: attentionDeviceFacts(gpu) }),
      ...variant.options,
    });
    assertEquals(
      scenarios.map((scenario) => scenario.name),
      ["run"],
    );
    const session = await model.createContainerSession(gpu, variant.options);
    try {
      const inputs = Object.fromEntries(
        shapes.map((shape, index) => [
          `x${index}`,
          fill(shape, (i) => (index === 3 ? (i % 4 === 3 ? -1 : 0) : ((i % (5 + index)) - 2) / 4)),
        ]),
      );
      // The slot backing is built on the second run with the same signature (the first run takes the arena path).
      for (let step = 0; step < 2; step += 1) await session.run(inputs);
      const diagnostics = session.diagnostics();
      const windowedQk = (diagnostics.lastRunPipelines ?? []).filter(
        ({ key }) => key.startsWith("attention_qk") && key.includes(":rwa"),
      );
      assertEquals(
        windowedQk.reduce((total, { dispatchCount }) => total + dispatchCount, 0),
        variant.expectedBlocks === 1 ? 0 : variant.expectedBlocks,
        "number of row blocks (1 block sets up no row window)",
      );
      if (variant.expectedOnlineWindows !== undefined) {
        // The online route: one dispatch per row window, and none of the three-pass kernels.
        const pipelines = diagnostics.lastRunPipelines ?? [];
        const count = (family: string): number =>
          pipelines
            .filter(({ key }) => key.startsWith(family))
            .reduce((total, { dispatchCount }) => total + dispatchCount, 0);
        assertEquals(count("attention_online"), variant.expectedOnlineWindows, "online windows");
        assertEquals(
          count("attention_qk") + count("attention_stats") + count("attention_pv"),
          0,
          "the three-pass kernels must not run on the online route",
        );
      }
      assertEquals(diagnostics.planBacking.residentBytes, scenarios[0].workspaceBytes);
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a fused attention op exactly match the slot table (f32, no budget, real GPU)",
  () => assertFusedAttentionMatchesEstimate({ options: {}, expectedBlocks: 1 }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a fused attention op exactly match the slot table when split by a budget too (f32 + mask, real GPU)",
  () =>
    assertFusedAttentionMatchesEstimate({
      options: { attentionRowBlockMaxBytes: 16 * 1024 },
      mask: true,
      expectedBlocks: 4,
    }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the total intermediates of a fused attention op exactly match the slot table with a8 + s16 split by a budget too (real GPU)",
  () =>
    assertFusedAttentionMatchesEstimate({
      options: {
        attentionCompute: "a8",
        attentionScoreStorage: "f16",
        attentionRowBlockMaxBytes: 16 * 1024,
      },
      expectedBlocks: 2,
    }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the workspace of a fused attention op matches the slot table on the online route (a8 + none, one window, real GPU)",
  () =>
    assertFusedAttentionMatchesEstimate({
      options: { attentionCompute: "a8", attentionScoreStorage: "none" },
      expectedBlocks: 1,
      expectedOnlineWindows: 1,
      subgroups: "available",
    }),
);

test.skipIf(!GPU_AVAILABLE)(
  "the workspace of a fused attention op matches the slot table on the online route over two row windows (real GPU)",
  () =>
    // 2⁴⁰ / (65,536 keys · 128 · 2) = 65,536 rows per window; 65,568 rows make two windows
    assertFusedAttentionMatchesEstimate({
      options: { attentionCompute: "a8", attentionScoreStorage: "none" },
      expectedBlocks: 1,
      expectedOnlineWindows: 2,
      subgroups: "available",
      shape: { b: 1, h: 1, m: 65568, n: 65536, d: 128 },
    }),
);

/**
 * `permute ×3 → attention → permute` on token-major q / k / v `[B, S, H, D]` (the Wan DiT form): on an
 * a8 seat the three input permutes are absorbed by the attention (perf-ledger F-3 A4 / F-8), and on
 * the online route the output permute too, so the workspace is the attention step's — its output
 * (token-major where it absorbed the output permute), the token-major inputs it now reads and the
 * temporaries of its route (the head-major copies of `attentionHeadMajorCopies` included) — and,
 * off the online route, the output permute's.
 */
const permutedAttentionModel = (shape: {
  readonly b: number;
  readonly h: number;
  readonly s: number;
  readonly n: number;
  readonly d: number;
}): PreparedModel => {
  const { b, h, s, n, d } = shape;
  const dtype = "f32";
  const token = { q: [b, s, h, d], k: [b, n, h, d], v: [b, n, h, d] } as const;
  const head = { q: [b, h, s, d], k: [b, h, n, d], v: [b, h, n, d] } as const;
  const operands = ["q", "k", "v"] as const;
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["attention", "permute"] },
    symbols: [],
    inputs: operands.map((name) => ({ name: `${name}t`, dtype, shape: [...token[name]] })),
    outputs: ["ot"],
    initializers: {},
    values: {
      ...Object.fromEntries(operands.map((name) => [name, { dtype, shape: [...head[name]] }])),
      o: { dtype, shape: [b, h, s, d] },
      ot: { dtype, shape: [b, s, h, d] },
    },
    nodes: [
      ...operands.map((name) => ({
        op: "permute",
        ins: [`${name}t`],
        outs: [name],
        attrs: { dims: [0, 2, 1, 3] },
      })),
      { op: "attention", ins: ["q", "k", "v"], outs: ["o"], attrs: { scale: 1 / Math.sqrt(d) } },
      { op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } },
    ],
  };
  return openGraph(graph);
};

test.skipIf(!GPU_AVAILABLE)(
  "an attention that absorbed its permutes (perf-ledger F-3 / F-8) has the workspace of its slot table on the a8 routes: online, s16 and either stage on f32 (real GPU)",
  async () => {
    const variants: readonly {
      readonly name: string;
      readonly options: SessionOptions;
      readonly shape: Parameters<typeof permutedAttentionModel>[0];
      /** The variant must take the online route (and absorb the output permute) on this device. */
      readonly online?: true;
    }[] = [
      // ①QK on f32 (D % 4 != 0): copies of q / k, the transposed V
      {
        name: "a8, ①QK on f32",
        options: { attentionCompute: "a8" },
        shape: { b: 2, h: 3, s: 37, n: 44, d: 14 },
      },
      // ③PV on f32 (N % 4 != 0): the token-major q / k quantised, a copy of V
      {
        name: "a8, ③PV on f32",
        options: { attentionCompute: "a8" },
        shape: { b: 2, h: 3, s: 37, n: 43, d: 16 },
      },
      // the quantisation reads q / k / v token-major; no copy
      {
        name: "a8 + s16",
        options: { attentionCompute: "a8", attentionScoreStorage: "f16" },
        shape: { b: 2, h: 3, s: 37, n: 44, d: 16 },
      },
      // no copy at all, and O stored token-major (the output permute absorbed)
      {
        name: "online",
        options: { attentionCompute: "a8", attentionScoreStorage: "none" },
        shape: { b: 2, h: 3, s: 70, n: 68, d: 64 },
        online: true,
      },
      // M ≫ N: O (256 KiB) outgrows the attention's temporaries (74.6 KiB), so a head-major O
      // next to the permute's output would raise the region total — the estimate has to replay the
      // absorbed output permute to meet the slot table
      {
        name: "online, O larger than the temporaries",
        options: { attentionCompute: "a8", attentionScoreStorage: "none" },
        shape: { b: 1, h: 2, s: 512, n: 8, d: 64 },
        online: true,
      },
    ];
    const gpu = await acquireGpu({ subgroups: "available" });
    try {
      for (const variant of variants) {
        const { d, n } = variant.shape;
        const route = attentionRoute(
          {
            attentionCompute: variant.options.attentionCompute ?? "f32",
            attentionScoreStorage: variant.options.attentionScoreStorage ?? "f32",
          },
          d,
          n,
          attentionDeviceFacts(gpu),
        );
        if (variant.online && route.online === undefined) {
          // The shape is one the online kernel admits, so only the device rules it out.
          assert(attentionOnlineI8a8Eligible(d, n), `${variant.name}: an online-eligible shape`);
          console.warn(
            `[karume] SKIP ${variant.name}: the device holds neither form of the online kernel (sg32 / wg) at D ${d}`,
          );
          continue;
        }
        assertEquals(route.online !== undefined, variant.online === true, `${variant.name}: route`);
        const model = permutedAttentionModel(variant.shape);
        const { scenarios } = model.estimate({
          maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
          maxBufferSize: gpu.limits.maxBufferSize,
          attentionDevice: attentionDeviceFacts(gpu),
          ...variant.options,
        });
        const session = await model.createContainerSession(gpu, variant.options);
        try {
          const { b, h, s } = variant.shape;
          const inputs = {
            qt: fill([b, s, h, d], (i) => Math.sin(i * 0.3)),
            kt: fill([b, n, h, d], (i) => Math.cos(i * 0.2)),
            vt: fill([b, n, h, d], (i) => ((i % 9) - 4) / 4),
          };
          // The slot backing is built on the second run of one signature (the first runs on the arena).
          for (let step = 0; step < 2; step += 1) await session.run(inputs);
          const diagnostics = session.diagnostics();
          // The output permute is absorbed where the route is the online kernel (perf-ledger F-3 A4).
          assertEquals(
            diagnostics.lastRunFusions?.attentionPermute,
            variant.online ? 4 : 3,
            `${variant.name}: absorbed`,
          );
          assertEquals(
            diagnostics.planBacking.residentBytes,
            scenarios[0].workspaceBytes,
            `${variant.name}: slot table vs estimate`,
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Real-GPU comparison
// ---------------------------------------------------------------------------

/**
 * A model in which all 3 fields are non-zero — an i8 embedding table (eligible = stays resident
 * compressed), an f16 mul operand (ineligible = expanded to f32) and an f32 add operand (stays
 * resident uncompressed).
 *
 * For the 2 fields compressed / expanded exact equality with the diagnostic `storage` can be claimed;
 * f32 does not appear in the diagnostics (the very reason `weights.uncompressedBytes` is a separate
 * field), so it is compared against a hand-computed constant.
 */
const gpuWeightModel = (): PreparedModel => {
  const table = fill([5, 3], (i) => (i % 7) - 3);
  const quantized = quantizeI8(table.data, [5, 3], 0);
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding", "mul", "add"] },
    symbols: [],
    inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
    outputs: ["y"],
    initializers: {
      w: {},
      g: {},
      c: {},
    },
    values: {
      w: { dtype: "f32", shape: [5, 3] },
      g: { dtype: "f32", shape: [3] },
      c: { dtype: "f32", shape: [3] },
      e: { dtype: "f32", shape: [2, 3] },
      h: { dtype: "f32", shape: [2, 3] },
      y: { dtype: "f32", shape: [2, 3] },
    },
    nodes: [
      { op: "embedding", ins: ["w", "ids"], outs: ["e"], attrs: { padding_idx: -1 } },
      { op: "mul", ins: ["e", "g"], outs: ["h"], attrs: {} },
      { op: "add", ins: ["h", "c"], outs: ["y"], attrs: {} },
    ],
  };
  return openGraph(graph, [
    weight("c", f32Bytes([1, 2, 3])),
    weight("g", f16Zeros(3), { codec: "f16" }),
    // per-channel (row length 3, 1 group) — the scale bytes are the same as v1's `[5,1]`.
    weight("w", quantized.bytes, {
      codec: "int8-sym",
      groupSize: 3,
      scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
    }),
  ]);
};

test.skipIf(!GPU_AVAILABLE)(
  "the estimator's weights and state exactly match the measured diagnostics (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const weightModel = gpuWeightModel();
      const estimate = weightModel.estimate();
      const session = await weightModel.createContainerSession(gpu);
      try {
        await session.run({ ids: fill([2], (i) => i, "i32") });
        const storage = session.diagnostics().storage;
        const { weights } = estimate.resident;
        assertEquals(weights.compressedBytes, storage.residentCompressedBytes);
        assertEquals(weights.expandedBytes, storage.hostExpandedBytes);
        // The f32 c does not appear in the diagnostics (ADR 0006's storage diagnostics cover only low-precision storage) — hand-computed 3×4
        assertEquals(weights.uncompressedBytes, 12);
      } finally {
        await session.dispose();
      }

      const model = stateModel();
      const generation = { chunkLength: 4, bindings: { C: 8 } };
      const stateEstimate = model.estimate({ bindings: { T: 2 }, generation });
      const stateSession = await model.createContainerSession(gpu);
      try {
        const context = await stateSession.createGenerationContext(generation);
        assertEquals(
          stateEstimate.resident.stateBytes,
          stateSession.diagnostics().stateBacking.residentBytes,
        );
        await context.dispose();
      } finally {
        await stateSession.dispose();
      }
      // workspaceBytes is compared on the graphs below, each built for one accounting path.
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * `linear(gelu_tanh(x), w, b)` (`gelu = true`) or `linear(x, w, b)` with `w` resident as i8 — the
 * pair the linear prologue (perf-ledger F-4) folds under `linearCompute: "a8"`, and the graph the
 * fold leaves behind.
 */
const geluLinearModel = (gelu: boolean): PreparedModel => {
  const [rows, k, n] = [4, 64, 16];
  const row = [1, rows, k];
  const quantized = quantizeI8(fill([n, k], (i) => Math.sin(i * 0.61)).data, [n, k], 0);
  const graph: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: gelu ? ["gelu_tanh", "linear"] : ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: row }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [n, k] },
      b: { dtype: "f32", shape: [n] },
      ...(gelu ? { g: { dtype: "f32", shape: row } } : {}),
      y: { dtype: "f32", shape: [1, rows, n] },
    },
    nodes: [
      ...(gelu ? [{ op: "gelu_tanh", ins: ["x"], outs: ["g"], attrs: {} }] : []),
      { op: "linear", ins: [gelu ? "g" : "x", "w", "b"], outs: ["y"], attrs: {} },
    ],
  };
  return openGraph(graph, [
    weight("b", f32Bytes(new Array(n).fill(0.5))),
    weight("w", quantized.bytes, {
      codec: "int8-sym",
      groupSize: k,
      scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
    }),
  ]);
};

test.skipIf(!GPU_AVAILABLE)(
  "a gelu_tanh folded into an a8 linear (perf-ledger F-4) leaves no output in the estimate or in the slot table: both equal the linear alone (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const limits = {
        maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
        maxBufferSize: gpu.limits.maxBufferSize,
      };
      const pair = geluLinearModel(true);
      const alone = geluLinearModel(false);
      const workspace = (model: PreparedModel, options: SessionOptions): number =>
        model.estimate({ ...limits, ...options }).scenarios[0].workspaceBytes;
      // The estimate sees the fold only with the Session's linearCompute.
      assertEquals(
        workspace(pair, { linearCompute: "a8" }),
        workspace(alone, { linearCompute: "a8" }),
      );
      assertEquals(
        workspace(pair, {}) > workspace(pair, { linearCompute: "a8" }),
        true,
        "without a8 the gelu output is counted",
      );
      // The real plan: the slot table of the folded pair is the linear's alone (xq, xs and y).
      const residentBytes = async (model: PreparedModel): Promise<number> => {
        const session = await model.createContainerSession(gpu, { linearCompute: "a8" });
        try {
          const x = fill([1, 4, 64], (i) => Math.cos(i * 0.3) * 2);
          // The slot backing is built on the second run of one signature (the first runs on the arena).
          for (let step = 0; step < 2; step += 1) await session.run({ x });
          assertEquals(
            session.diagnostics().lastRunFusions?.linearPrologue,
            model === pair ? 1 : 0,
            "linearPrologue",
          );
          return session.diagnostics().planBacking.residentBytes;
        } finally {
          await session.dispose();
        }
      };
      assertEquals(await residentBytes(pair), await residentBytes(alone));
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The estimate's post-fusion accounting of every rule against the slot table a Session builds for
 * the same graph (helpers/fused-window-graphs.ts — one window per rule, both external test rules,
 * and the a8 linear's two node temporaries with and without a folded producer;
 * estimate_fused_steps_test.ts pins the same graphs to hand-derived numbers). The Session gets the
 * options the window needs, the estimate the same options and the device's limits.
 */
test.skipIf(!GPU_AVAILABLE)(
  "every rule's fused window and the a8 linear have the workspace of their slot table: the ten built-in rules, two external ones and the a8 linear with and without a folded producer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const limits = {
        maxStorageBufferBindingSize: gpu.limits.maxStorageBufferBindingSize,
        maxBufferSize: gpu.limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension: gpu.limits.maxComputeWorkgroupsPerDimension,
      };
      for (const make of [
        ...FUSED_WINDOW_CASES,
        () => a8LinearCase(false),
        () => a8LinearCase(true),
        a8LinearAfterDeadRegion,
      ]) {
        const fused = make();
        const model = openGraph(fused.graph, fused.tensors);
        const { scenarios } = model.estimate({
          ...limits,
          ...(fused.bindings === undefined ? {} : { bindings: fused.bindings }),
          ...fused.options,
        });
        const session = await model.createContainerSession(gpu, fused.options);
        try {
          // The slot backing is built on the second run of one signature (the first runs on the arena).
          for (let step = 0; step < 2; step += 1) await session.run(fused.inputs);
          const diagnostics = session.diagnostics();
          const counts: Readonly<Record<string, number | undefined>> =
            diagnostics.lastRunFusions ?? {};
          assertEquals(counts[fused.counter], fused.windows, `${fused.name}: windows taken`);
          assertEquals(
            diagnostics.planBacking.residentBytes,
            scenarios[0].workspaceBytes,
            `${fused.name}: slot table vs estimate`,
          );
        } finally {
          await session.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);
