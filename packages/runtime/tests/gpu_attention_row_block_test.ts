/**
 * Real-GPU checks of **query row-block execution** in fused attention (ADR 0023). The same 2 as the
 * row blocks of the decomposed path (fusion rule `rowBlockAttention` —
 * tests/gpu_row_block_attention_test.ts) (① forced-split parity / ② portability), plus 2 that only
 * the fused path has (slot derivation of hit runs, census of the row-window keys) — 4 in all.
 *
 * ① **Forced-split parity**: with the same inputs and the same Session settings, run
 *    - the default (= 1 block on a machine with headroom under the limit = exactly the path before
 *      splitting)
 *    - the form forced to 2 / 3 / 5 blocks with `ROW_BLOCK_SPLIT`
 *    and compare them as **f32 bits**. The point is the exact equality of `Uint32Array`, not
 *    allclose — row blocking only changes "which workgroup handles which row", so a move of even 1
 *    ulp means the K reduction order or the rounding sequence (or a row / statistics index) moved.
 *    The variants (s16 storage / i8a8 / additive mask / GQA) are run across because the row window
 *    enters **a different side per stage** (the A side = q for ①QK, the C side = O for ③PV), so an
 *    implementation that forgets one side survives as "the values are right but one variant
 *    breaks".
 *
 * ② **Portability check**: on a device with `maxStorageBufferBindingSize` narrowed to the WebGPU
 *    core default (128MiB), a shape whose S exceeds 128MiB passes. As a pair, **forcing
 *    `ROW_BLOCK_SPLIT: 1` on the same device and shape fails loudly** (a shape that does not fail
 *    being green is no evidence that row blocking works).
 *
 * ③ **Budget-split parity** (ADR 0124): a split finer than the binding limit made by
 *    `SessionOptions.attentionRowBlockMaxBytes` equals the unbudgeted run (1 block) as Uint32. The
 *    2 f16-compute variants (which need `shader-f16`) run both the forced split and the budget
 *    split in a separate `test` (the variant enumeration of ADR 0124 decision 4).
 *
 * MUST: do not make it tautological. ① also checks that the intermediate peak
 * (`peakTransientBytes`) **shrinks monotonically** with the block count — without it, "the split
 * never fired and the same path was hit twice" is always green. ③ also checks that the budgeted
 * block count (the dispatch count of the row-window keys) is at least 2 and that the peak is
 * strictly smaller than unbudgeted (a run where the budget never reached the derivation and ran as
 * 1 block is not green). Every time, the output is also confirmed not to be constant.
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { planRowBlocks } from "../src/runtime/fusion.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import {
  createSessionFromContainer,
  ROW_BLOCK_SPLIT,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import {
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openGraphModel,
  singleOpDeclaration,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE, SHADER_F16_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

/** The storage binding limit of the WebGPU core defaults. The portability check reproduces it. */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

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
  /** Head count of K / V (GQA if it differs from `h` — ADR 0067). */
  readonly kv: number;
  readonly m: number;
  readonly n: number;
  readonly d: number;
};

type Variant = {
  readonly name: string;
  readonly options: SessionOptions;
  /**
   * Whether to add an additive mask `[1,1,M,N]` (a mask **differing per row** = a detector of the
   * row offset).
   */
  readonly mask?: boolean;
  /**
   * Whether it can run on a GQA shape (`h != kv`) (i8a8 cannot be combined with GQA — ADR 0067
   * decision 3).
   */
  readonly gqa?: boolean;
};

/**
 * Across the variants. The row window enters **different places** in ①QK (the A side = rows of q /
 * mask) and ③PV (the C side = O), and with i8a8 it becomes yet another kernel (reading the
 * quantized q with the full-M stride), so each path is run once.
 */
const VARIANTS: readonly Variant[] = [
  { name: "f32", options: {}, gqa: true },
  { name: "f32 + mask", options: {}, mask: true, gqa: true },
  { name: "s16", options: { attentionScoreStorage: "f16" }, gqa: true },
  { name: "a8", options: { attentionCompute: "a8" } },
  { name: "a8 + s16", options: { attentionCompute: "a8", attentionScoreStorage: "f16" } },
];

/**
 * The f16-compute variants (ADR 0028 — another storage form passing S as `array<f16>`). They need
 * `shader-f16`, so they run in a separate `test` that takes the device of
 * `acquireGpu({ shaderF16: true })` (the device of {@link VARIANTS} is taken without the feature).
 * They can be combined with the mask and GQA.
 */
const F16_VARIANTS: readonly Variant[] = [
  { name: "f16", options: { attentionCompute: "f16" }, gqa: true },
  { name: "f16 + mask", options: { attentionCompute: "f16" }, mask: true, gqa: true },
];

const SHAPES: readonly Shape[] = [
  // Remainders on every axis (D % 4 ≠ 0 / N % 4 ≠ 0 = scalar variant for both ①③; the path where
  // s16 and i8a8 both degrade). M = 17 splits with a remainder for each of 2 / 3 / 5 blocks.
  { name: "remainders B2 H3 M17 N19 D13", b: 2, h: 3, kv: 3, m: 17, n: 19, d: 13 },
  // The v4 path (D % 4 == 0 && N % 4 == 0) with M spanning 2 row tiles.
  { name: "v4 B1 H2 M68 N20 D12", b: 1, h: 2, kv: 2, m: 68, n: 20, d: 12 },
  // A shrunk DiT self-attention shape (D = 128, M / N both multiples of 64 = the real model's
  // path).
  { name: "DiT shape B1 H4 M64 N64 D128", b: 1, h: 4, kv: 4, m: 64, n: 64, d: 128 },
  // GQA (r = 4). The row window sits on top of the kv-head mapping in both ①QK and ③PV.
  { name: "GQA B1 H8 Hkv2 M40 N24 D16", b: 1, h: 8, kv: 2, m: 40, n: 24, d: 16 },
];

/** Block counts to force (1 = same as the default, 2/3 split evenly, 5 splits with a remainder). */
const SPLITS: readonly number[] = [2, 3, 5];

/**
 * The measured band mask (the centre moves per row = **a mixed-up row offset always changes the
 * values**).
 *
 * MUST: **every row has at least one unmasked column** (if a whole row is −inf, amax becomes −inf
 * and `exp(−inf −(−inf))` is NaN — the split and unsplit sides produce the same NaN, so the bit
 * comparison passes, but it verifies nothing). A symmetric band always leaves its centre unmasked.
 */
const bandMask = (m: number, n: number): FilledTensor => {
  const width = Math.max(1, Math.floor(n / 4));
  return fill([1, 1, m, n], (index) => {
    const row = Math.floor(index / n);
    const col = index % n;
    const center = Math.floor((row * n) / m);
    return Math.abs(col - center) <= width ? 0 : Number.NEGATIVE_INFINITY;
  });
};

/**
 * The output and the peak of the intermediate buffers that run actually allocated (where the effect
 * of row blocking is observed).
 */
type RunResult = {
  /**
   * The outputs per run (from the 2nd on, hit runs of the derived plan — a different binding path).
   */
  readonly outputs: readonly Tensor[];
  readonly output: Tensor;
  /** The intermediate peak of **the 1st run** (the arena path). */
  readonly peakTransientBytes: number;
  /** Dispatch counts per pipeline key (census — independent of timing). */
  readonly dispatchCounts: ReadonlyMap<string, number>;
};

const runAttention = async (
  gpu: GpuContext,
  shape: Shape,
  variant: Variant,
  split?: number,
  runs = 1,
): Promise<RunResult> => {
  const { b, h, kv, m, n, d } = shape;
  const q = fill([b, h, m, d], QUERY);
  const k = fill([b, kv, n, d], KEY);
  const v = fill([b, kv, n, d], VALUE);
  const mask = variant.mask === true ? bandMask(m, n) : undefined;
  const graph = singleOpDeclaration(
    "attention",
    [q.shape, k.shape, v.shape, ...(mask === undefined ? [] : [mask.shape])],
    [[b, h, m, d]],
    { attrs: { scale: halfScale(d) } },
  );
  const options: SessionOptions = {
    ...variant.options,
    ...(split === undefined ? {} : { [ROW_BLOCK_SPLIT]: split }),
  };
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(graph),
    GRAPH_NAME,
    options,
  );
  try {
    const inputs: Record<string, FilledTensor> = { x0: q, x1: k, x2: v };
    if (mask !== undefined) inputs["x3"] = mask;
    const outputs: Tensor[] = [];
    let peakTransientBytes = 0;
    let dispatchCounts = new Map<string, number>();
    for (let run = 0; run < runs; run += 1) {
      outputs.push((await session.run(inputs))["y"]);
      const diagnostics = session.diagnostics();
      // MUST: the peak and the breakdown are those of **the 1st run** (the arena path). Hit runs
      // reuse the resident backing and allocate no temporaries, so overwriting them would make the
      // observation a no-op.
      if (run > 0) continue;
      peakTransientBytes = diagnostics.lastRun?.peakTransientBytes ?? 0;
      dispatchCounts = new Map(
        requireCensus(diagnostics.lastRunPipelines, "attention").map((row) => [
          row.key,
          row.dispatchCount,
        ]),
      );
    }
    return { outputs, output: outputs[0], peakTransientBytes, dispatchCounts };
  } finally {
    await session.dispose();
  }
};

/** The f32 bits (a form in which both `0.0` vs `-0.0` and a 1-ulp difference are visible). */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const assertSameBits = (actual: Tensor, expected: Tensor, label: string): void => {
  assertEquals(actual.shape, expected.shape, label);
  const a = bits(actual);
  const e = bits(expected);
  const mismatches: number[] = [];
  for (let i = 0; i < a.length && mismatches.length < 4; i += 1) {
    if (a[i] !== e[i]) mismatches.push(i);
  }
  assertEquals(
    mismatches,
    [],
    `${label}: bits differ (first mismatches: ${mismatches
      .map((i) => `[${i}] ${actual.data[i]} vs ${expected.data[i]}`)
      .join(" / ")})`,
  );
  // Tautology guard: if the output is constant, "equal" verifies nothing.
  assert(new Set(a).size > 1, `${label}: the output is constant (bit equality is a tautology)`);
};

/**
 * Lists only the shapes a variant can run (GQA shapes only with variants combinable with GQA — i8a8
 * × GQA fails loudly).
 */
const shapeVariants = (
  variants: readonly Variant[],
): readonly { readonly shape: Shape; readonly variant: Variant }[] =>
  SHAPES.flatMap((shape) =>
    variants
      .filter((variant) => shape.h === shape.kv || variant.gqa === true)
      .map((variant) => ({ shape, variant })),
  );

/**
 * ① Forced-split parity (compares 2 / 3 / 5 blocks against the 1-block run as Uint32 and checks the
 * peak decreases monotonically).
 */
const assertForcedSplitParity = async (
  gpu: GpuContext,
  variants: readonly Variant[],
): Promise<void> => {
  for (const { shape, variant } of shapeVariants(variants)) {
    const label = `${shape.name} / ${variant.name}`;
    // The reference = before splitting (on a real machine with headroom under the limit, 1 block =
    // exactly the existing path without a row window).
    const reference = await runAttention(gpu, shape, variant);
    let previousPeak = reference.peakTransientBytes;
    assert(previousPeak > 0, `${label}: the intermediate peak is 0 (the observation is a no-op)`);
    for (const split of SPLITS) {
      const blocked = await runAttention(gpu, shape, variant, split);
      assertSameBits(blocked.output, reference.output, `${label} / ${split} blocks`);
      // More blocks shrink the materialized width of S (the observation that **the materialized
      // width shrinks** even with the same values — if it is not monotonic, the temporary lifetime
      // declarations or the pool reuse are broken).
      assert(
        blocked.peakTransientBytes < previousPeak,
        `${label} / ${split} blocks: the intermediate peak ${blocked.peakTransientBytes}B did not shrink from ${previousPeak}B`,
      );
      previousPeak = blocked.peakTransientBytes;
    }
  }
};

/**
 * The budget used for the budget split (ADR 0124). The value giving ⌈M/3⌉ rows per block when S is
 * stored in 2 B — the 2 B variants (s16 / f16) split into 3 blocks, the 4 B variants (f32 / a8)
 * into more (2 or more on every shape). Every shape has N ≥ 4, so one row of scores is larger than
 * the 8 B of row statistics and the block count is decided by S alone.
 */
const budgetFor = (shape: Shape): number =>
  shape.b * shape.h * shape.n * 2 * Math.ceil(shape.m / 3);

/**
 * ③ Budget-split parity. Equal to the unbudgeted run (1 block) as Uint32, the budgeted block count
 * (the dispatch count of the row-window keys ①QK / ③PV — a census independent of timing) is at
 * least 2, and the peak is strictly smaller than unbudgeted.
 */
const assertBudgetSplitParity = async (
  gpu: GpuContext,
  variants: readonly Variant[],
): Promise<void> => {
  for (const { shape, variant } of shapeVariants(variants)) {
    const budget = budgetFor(shape);
    const label = `${shape.name} / ${variant.name} / budget ${budget}B`;
    const reference = await runAttention(gpu, shape, variant);
    assertEquals(
      [...reference.dispatchCounts.keys()].filter((key) => key.includes(":rw")),
      [],
      `${label}: row-window keys are set without a budget (the reference is not 1 block)`,
    );
    const budgeted = await runAttention(gpu, shape, {
      ...variant,
      options: { ...variant.options, attentionRowBlockMaxBytes: budget },
    });
    assertSameBits(budgeted.output, reference.output, label);
    const windowed = [...budgeted.dispatchCounts].filter(([key]) => key.includes(":rw"));
    assertEquals(windowed.length, 2, `${label}: the row-window keys are the 2 of ①QK and ③PV`);
    const blocks = windowed[0][1];
    assert(
      blocks >= 2,
      `${label}: the budgeted block count ${blocks} is below 2 (the budget did not reach the derivation)`,
    );
    assertEquals(windowed[1][1], blocks, `${label}: the block counts of ①QK and ③PV disagree`);
    assert(
      budgeted.peakTransientBytes < reference.peakTransientBytes,
      `${label}: the intermediate peak ${budgeted.peakTransientBytes}B did not shrink from the unbudgeted ${reference.peakTransientBytes}B`,
    );
  }
};

test.skipIf(!GPU_AVAILABLE)(
  "row-block execution of fused attention is **bit-for-bit equal** to the 1-block run whatever the block count (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      await assertForcedSplitParity(gpu, VARIANTS);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "fused attention split by the budget attentionRowBlockMaxBytes is **bit-for-bit equal** to the 1-block run, with more blocks and a smaller peak (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      await assertBudgetSplitParity(gpu, VARIANTS);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE || !SHADER_F16_AVAILABLE)(
  "row-block execution of f16 compute (f16 / f16 + mask, including GQA shapes) is **bit-for-bit equal** to the 1-block run under both the forced split and the budget split (real GPU, shader-f16)",
  async () => {
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      await assertForcedSplitParity(gpu, F16_VARIANTS);
      await assertBudgetSplitParity(gpu, F16_VARIANTS);
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Even on **hit runs** (the derived plan = resident backing + baked bind groups —
 * `#activateBacking` of src/runtime/executor.ts), the row-block temporaries get the same slot
 * assignment as on a miss run.
 *
 * Row blocking changes **the count and lifetimes** of the temporaries within a step, so if slot
 * derivation (`derivePlanSlots`) disagrees with the arena path, it becomes the silent wrong value
 * "only the 2nd run writes to another buffer" — no exception is thrown, so only a bit comparison
 * can see it.
 */
test.skipIf(!GPU_AVAILABLE)(
  "row-block execution is bit-for-bit equal between hit runs (the derived plan) and miss runs (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const shape = SHAPES[2];
      for (const variant of [VARIANTS[0], VARIANTS[4]]) {
        const reference = await runAttention(gpu, shape, variant);
        // Run with an uneven split (M=64 → 22/21/21) — an even split reuses slots across blocks, so
        // a layout with 2 slot sizes side by side more easily exposes a derivation mismatch.
        const blocked = await runAttention(gpu, shape, variant, 3, 2);
        assertEquals(blocked.outputs.length, 2);
        for (const [index, output] of blocked.outputs.entries()) {
          assertSameBits(
            output,
            reference.output,
            `${shape.name} / ${variant.name} / run ${index}`,
          );
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * **census** (ADR 0058 decision 4). The keys of the row-window variants (`:rwa` / `:rwc`)
 * **actually run** for n ≥ 2, their dispatch count equals the block count exactly, and none is set
 * for 1 block.
 *
 * MUST: the source is `lastRunPipelines`, which does not depend on timing (runs on machines without
 * timing too — an empty census is rejected by `requireCensus` of tests/helpers/pipeline-census.ts).
 */
test.skipIf(!GPU_AVAILABLE)(
  "row-window keys are set only for n ≥ 2 and their dispatch count equals the block count (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const shape = SHAPES[2];
      for (const variant of VARIANTS) {
        const label = `${shape.name} / ${variant.name}`;
        const single = await runAttention(gpu, shape, variant);
        assert(
          single.dispatchCounts.size > 0,
          `${label}: the breakdown is empty (the key check is a no-op)`,
        );
        assertEquals(
          [...single.dispatchCounts.keys()].filter((key) => key.includes(":rw")),
          [],
          `${label}: row-window keys are set for 1 block`,
        );
        for (const split of SPLITS) {
          const blocked = await runAttention(gpu, shape, variant, split);
          const windowed = [...blocked.dispatchCounts].filter(([key]) => key.includes(":rw"));
          assertEquals(
            windowed.length,
            2,
            `${label} / ${split} blocks: the row-window keys are the 2 of ①QK and ③PV (actually ${windowed
              .map(([key]) => key)
              .join(" / ")})`,
          );
          for (const [key, count] of windowed) {
            assertEquals(count, split, `${label} / ${split} blocks: dispatch count of ${key}`);
          }
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The shape of the portability check. Lifts `S = B·H·M·N·4` above 128MiB with a shape that **needs
 * no real assets** (4 × 2048 × 4200 × 4 = 137,625,600B). D is narrowed to 8, so the GEMM work
 * itself is small; the one thing checked is "whether it avoids materializing an intermediate above
 * the limit".
 */
const OVERSIZE: Shape = {
  name: "over-limit B1 H4 M2048 N4200 D8",
  b: 1,
  h: 4,
  kv: 4,
  m: 2048,
  n: 4200,
  d: 8,
};

test.skipIf(!GPU_AVAILABLE)(
  "on a device narrowed to maxStorageBufferBindingSize=128MiB, fused attention whose S exceeds the limit passes through row blocks (real GPU)",
  async () => {
    const { b, h, m, n } = OVERSIZE;
    const scoreBytes = b * h * m * n * 4;
    assert(
      scoreBytes > CORE_STORAGE_BINDING_LIMIT,
      `S of the synthetic graph, ${scoreBytes}B, does not exceed the core default limit (the check would be a no-op)`,
    );
    // With **the same pure function** the execution reads, first pin that this shape splits into 2
    // blocks and one block fits inside the limit (so the GPU side going green is not "by chance").
    const blocks = planRowBlocks(m, b * h * n * 4, CORE_STORAGE_BINDING_LIMIT);
    assertEquals(blocks.length, 2, "the smallest block count that fits the limit");
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits is not narrowed (if not narrowed, the check sees nothing)",
      );
      const blocked = await runAttention(gpu, OVERSIZE, VARIANTS[0]);
      assertEquals(blocked.output.shape, [b, h, m, OVERSIZE.d]);
      assert(
        new Set(bits(blocked.output)).size > 1,
        "the output is constant (the check is a no-op)",
      );

      // Proof that the check has teeth: on **the same device and shape**, forcing 1 block
      // materializes S whole and so fails loudly. If this stays green, the green above is not
      // thanks to row blocking.
      await assertRejects(
        () => runAttention(gpu, OVERSIZE, VARIANTS[0], 1),
        ExecutionError,
        "does not fit the limit",
      );
    } finally {
      gpu.destroy();
    }
  },
);
