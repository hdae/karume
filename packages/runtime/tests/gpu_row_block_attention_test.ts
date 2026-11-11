/**
 * Real-GPU checks for the **row-block execution** of decomposed attention (fusion rule
 * `rowBlockAttention` — src/runtime/fusion.ts). There are three, and each holds only on real
 * hardware.
 *
 * ① **Forced-split parity**: for the same input, run
 *    - the **plain 9-node sequence** of a spelling the matcher cannot catch (only one zero-dispatch
 *      alias inserted between bmm and reshape)
 *    - the fusion in 1 block (the default = identical to the plain 4-dispatch sequence)
 *    - the fusion in n blocks (`ROW_BLOCK_SPLIT` forces 2 / 3 / an uneven split)
 *    and compare them as **f32 bit patterns**. The point is exact `Uint32Array` equality rather
 *    than allclose — row blocking only changes "which workgroup handles which row", so a 1 ulp
 *    change means the K reduction order or the rounding sequence moved.
 *
 * ② **Portability check**: on a **device with `maxStorageBufferBindingSize` capped** to the WebGPU
 *    core default (128MiB) (`LIMIT_CAPS`), a synthetic graph whose S exceeds 128MiB passes. As the
 *    pair, also check that **the same graph in a spelling the matcher cannot catch fails on the
 *    same device** (passing a shape that never fails is no evidence that row blocking is in
 *    effect).
 *
 * ③ **Budget-split parity** (ADR 0124): the fusion split by the budget `attentionRowBlockMaxBytes`
 *    matches the plain 9-node sequence in Uint32 (the block count and the peak are checked too).
 *
 * MUST: do not let this become a tautology. ① runs **separate graphs / separate Sessions** (running
 * the same path twice always agrees). Every run also confirms that the output is not constant.
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import { checkFusionTwin } from "@karume/kernel-checks";
import { ExecutionError } from "../src/runtime/plan.ts";
import { acquireGpu, type GpuContext, LIMIT_CAPS } from "../src/gpu/device.ts";
import { planRowBlocks } from "../src/runtime/fusion.ts";
import {
  createSessionFromContainer,
  ROW_BLOCK_SPLIT,
  type Session,
  type Tensor,
} from "../src/runtime/executor.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openGraphModel,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckCase } from "./helpers/kernel-check-assert.ts";
import { countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

/** The WebGPU core default storage binding limit. The portability check reproduces it. */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

type Shape = {
  readonly name: string;
  /** The batch axis with B·H folded. */
  readonly heads: number;
  /** The query row count M. */
  readonly queries: number;
  /** The key column count N. */
  readonly keys: number;
  /** The head width D. */
  readonly headDim: number;
};

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/**
 * The `[1,1,1,N]` additive mask (the same "per column" broadcast as the observed shape).
 *
 * One −inf in every 4 columns exercises safe_softmax's −inf path. MUST: do not block every column —
 * safe_softmax writes all zeros for an all −inf row, so the bit comparison becomes the tautology
 * "0 equals 0".
 */
const columnMask = (keys: number): FilledTensor =>
  fill([1, 1, 1, keys], (i) => (i % 4 === 3 ? Number.NEGATIVE_INFINITY : 0));

/**
 * The observed spelling of decomposed attention (exactly the 9-node window of the irodori DiT).
 *
 * `interpose` only inserts one zero-dispatch alias between bmm and reshape. It breaks only the
 * window's adjacency condition without changing the values or the physical dispatch count, so it
 * becomes **the reference on the same backend** (the path before fusion).
 */
const attentionGraph = (shape: Shape, interpose = false): DeclarationJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = interpose ? "scores3_alias" : "scores3";
  const nodes: DeclarationJson["nodes"] = [
    { op: "bmm", ins: ["q", "kt"], outs: [qkOut], attrs: {} },
    ...(interpose ? [{ op: "reshape", ins: [qkOut], outs: ["scores3"], attrs: {} }] : []),
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    { op: "add", ins: ["scores4", "mask"], outs: ["masked"], attrs: {} },
    { op: "safe_softmax", ins: ["masked"], outs: ["probs"], attrs: { dim: 3 } },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["probs3", "v3"], outs: ["y"], attrs: {} },
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      { name: "kt", dtype: "f32", shape: [heads, headDim, keys] },
      { name: "mask", dtype: "f32", shape: [1, 1, 1, keys] },
      { name: "v", dtype: "f32", shape: [1, heads, keys, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      ...(interpose ? { scores3_alias: { dtype: "f32", shape: scores3 } } : {}),
      scores3: { dtype: "f32", shape: scores3 },
      scores4: { dtype: "f32", shape: scores4 },
      masked: { dtype: "f32", shape: scores4 },
      probs: { dtype: "f32", shape: scores4 },
      probsExpanded: { dtype: "f32", shape: scores4 },
      probs3: { dtype: "f32", shape: scores3 },
      vExpanded: { dtype: "f32", shape: [1, heads, keys, headDim] },
      v3: { dtype: "f32", shape: [heads, keys, headDim] },
      y: { dtype: "f32", shape: [heads, queries, headDim] },
    },
    nodes,
  };
};

const attentionInputs = (shape: Shape): Readonly<Record<string, FilledTensor>> => ({
  q: fill([shape.heads, shape.queries, shape.headDim], QUERY),
  kt: fill([shape.heads, shape.headDim, shape.keys], KEY),
  mask: columnMask(shape.keys),
  v: fill([1, shape.heads, shape.keys, shape.headDim], VALUE),
});

/**
 * The output and the peak of intermediate buffers that run actually allocated (where the effect of
 * row blocking is observed).
 */
type RunResult = {
  readonly output: Tensor;
  readonly peakTransientBytes: number;
  /**
   * The dispatch count of the row-window QK (`:rwa`) = the block count when there are 2 or more
   * (0 with 1 block — a census that does not depend on timing).
   */
  readonly windowedQkDispatches: number;
};

const open = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  split?: number,
  budget?: number,
): Promise<Session> =>
  await createSessionFromContainer(gpu, await openGraphModel(graph), GRAPH_NAME, {
    ...(split === undefined ? {} : { [ROW_BLOCK_SPLIT]: split }),
    ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
  });

/**
 * What the Session's last run allocated. MUST: read after its first run — a later run on the slot
 * backing allocates no intermediate buffer from the arena, so its peak reads 0.
 */
const allocated = (session: Session): Omit<RunResult, "output"> => {
  const diagnostics = session.diagnostics();
  return {
    peakTransientBytes: diagnostics.lastRun?.peakTransientBytes ?? 0,
    windowedQkDispatches: countDispatches(
      requireCensus(diagnostics.lastRunPipelines, "decomposed attention"),
      (key) => key.includes(":rwa"),
    ),
  };
};

/** Runs `session` once and reads what that run allocated. */
const observe = async (
  session: Session,
  inputs: Readonly<Record<string, FilledTensor>>,
): Promise<RunResult> => {
  const output = (await session.run(inputs))["y"];
  return { output, ...allocated(session) };
};

const run = async (
  gpu: GpuContext,
  graph: DeclarationJson,
  inputs: Readonly<Record<string, FilledTensor>>,
  split?: number,
  budget?: number,
): Promise<RunResult> => {
  const session = await open(gpu, graph, split, budget);
  try {
    return await observe(session, inputs);
  } finally {
    await session.dispose();
  }
};

/**
 * The f32 bit patterns (a form where both `0.0` vs `-0.0` and a last-place 1 ulp difference show).
 */
const bits = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

/**
 * The fused side's row-block run against the plain 9-node sequence: the bit patterns, the
 * rowBlockAttention counter on both sides (checkFusionTwin, NaN by payload), the output shape and a
 * check against tautology (if the output is constant, "equal" verifies nothing), on a run of the
 * fused side of its own (the harness returns a case). Returns that output with what the harness's
 * run — the fused Session's first — allocated.
 */
const assertSameBits = async (
  gpu: GpuContext,
  fused: Session,
  reference: { readonly session: Session; readonly output: Tensor },
  inputs: Readonly<Record<string, FilledTensor>>,
  label: string,
): Promise<RunResult> => {
  assertCheckCase(
    await checkFusionTwin(gpu, {
      label,
      fused,
      plain: reference.session,
      feeds: inputs,
      counter: "rowBlockAttention",
      hits: 1,
      nan: "payload",
    }),
  );
  const firstRun = allocated(fused);
  const output = (await fused.run(inputs))["y"];
  assertEquals(output.shape, reference.output.shape, label);
  assert(
    new Set(bits(output)).size > 1,
    `${label}: the output is constant (bit equality is a tautology)`,
  );
  return { output, ...firstRun };
};

const SHAPES: readonly Shape[] = [
  // Every axis uneven (none of M / N / D is a multiple of 4 = both bmm take the scalar variant)
  { name: "uneven H3 M17 N19 D13", heads: 3, queries: 17, keys: 19, headDim: 13 },
  // On the v4 path (D % 4 == 0 && N % 4 == 0), M spans two row tiles. Splitting also crosses the
  // geometry buckets (M ≤ 64 / ≤ 512), so this confirms "the values do not move when the geometry
  // changes".
  { name: "v4 H2 M68 N20 D12", heads: 2, queries: 68, keys: 20, headDim: 12 },
  // A shrunken DiT self-attention (D = 128, M / N both multiples of 64 = the real model's path)
  { name: "DiT shape H4 M64 N64 D128", heads: 4, queries: 64, keys: 64, headDim: 128 },
];

/** The forced block counts (1 = the same shape as the default, 2/3 split evenly, 5 unevenly). */
const SPLITS: readonly number[] = [1, 2, 3, 5];

test.skipIf(!GPU_AVAILABLE)(
  "row-block execution is **bit-identical** to the plain 9-node sequence at every block count (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const inputs = attentionInputs(shape);
        // The reference = a spelling the matcher cannot catch (the plain 9-node sequence that
        // never fuses).
        const referenceSession = await open(gpu, attentionGraph(shape, true));
        try {
          const reference = {
            session: referenceSession,
            output: (await observe(referenceSession, inputs)).output,
          };
          let previousPeak = Number.POSITIVE_INFINITY;
          for (const split of SPLITS) {
            const fusedSession = await open(gpu, attentionGraph(shape), split);
            try {
              const label = `${shape.name} / ${split} blocks`;
              const fused = await assertSameBits(gpu, fusedSession, reference, inputs, label);
              // More blocks lower the intermediate peak (the observation point that **the
              // materialized width shrinks** even with identical values — if this is not
              // monotonic, a temporary's lifetime declaration or the pool reuse is broken).
              assert(
                fused.peakTransientBytes < previousPeak,
                `${label}: the intermediate peak ${fused.peakTransientBytes}B did not shrink from ${previousPeak}B`,
              );
              previousPeak = fused.peakTransientBytes;
            } finally {
              await fusedSession.dispose();
            }
          }
        } finally {
          await referenceSession.dispose();
        }
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Budget-split parity (ADR 0124). The fusion split more finely than the binding limit by
 * `SessionOptions.attentionRowBlockMaxBytes` matches the plain 9-node sequence in Uint32. The
 * budget is set to ⌈M/3⌉ rows per block (3 blocks for every shape — 17 rows are 6 + 6 + 5, 68 rows
 * are 23 + 23 + 22, 64 rows are 22 + 21 + 21).
 *
 * MUST: do not let this become a tautology. Check at once that the budgeted block count (the
 * dispatch count of the row-window QK) is the independently counted 3, and that the peak is
 * strictly below the unbudgeted fusion (1 block) (a run in 1 block where the budget never reached
 * the fusion's context must not pass).
 */
test.skipIf(!GPU_AVAILABLE)(
  "row-block execution split by the attentionRowBlockMaxBytes budget is **bit-identical** to the plain 9-node sequence, with more blocks and a smaller peak (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of SHAPES) {
        const inputs = attentionInputs(shape);
        const budget = shape.heads * shape.keys * 4 * Math.ceil(shape.queries / 3);
        const label = `${shape.name} / budget ${budget}B`;
        const referenceSession = await open(gpu, attentionGraph(shape, true));
        const budgetedSession = await open(gpu, attentionGraph(shape), undefined, budget);
        let budgeted: RunResult;
        let single: RunResult;
        try {
          const reference = {
            session: referenceSession,
            output: (await observe(referenceSession, inputs)).output,
          };
          single = await run(gpu, attentionGraph(shape), inputs);
          assertEquals(
            single.windowedQkDispatches,
            0,
            `${label}: the unbudgeted run is not 1 block`,
          );
          budgeted = await assertSameBits(gpu, budgetedSession, reference, inputs, label);
        } finally {
          await referenceSession.dispose();
          await budgetedSession.dispose();
        }
        assertEquals(budgeted.windowedQkDispatches, 3, `${label}: the budgeted block count`);
        assert(
          budgeted.peakTransientBytes < single.peakTransientBytes,
          `${label}: the intermediate peak ${budgeted.peakTransientBytes}B did not shrink from the unbudgeted ${single.peakTransientBytes}B`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The synthetic graph of the portability check. It raises `S = H·M·N·4` above 128MiB **without
 * real assets** (4 × 2048 × 4200 × 4 = 137,625,600B). D is cut down to 8, so the GEMM work itself
 * is small; the only thing checked is "whether an intermediate above the limit can avoid being
 * materialized".
 */
const OVERSIZE: Shape = {
  name: "over the limit H4 M2048 N4200 D8",
  heads: 4,
  queries: 2048,
  keys: 4200,
  headDim: 8,
};

test.skipIf(!GPU_AVAILABLE)(
  "on a device capped to maxStorageBufferBindingSize=128MiB, a graph whose S exceeds the limit passes with row blocks (real GPU)",
  async () => {
    const scoreBytes = OVERSIZE.heads * OVERSIZE.queries * OVERSIZE.keys * 4;
    assert(
      scoreBytes > CORE_STORAGE_BINDING_LIMIT,
      `the synthetic graph's S ${scoreBytes}B does not exceed the core default limit (the check would never fire)`,
    );
    // With **the same pure function** that execution reads, first pin that this shape splits into
    // 2 blocks and that each block fits within the limit (so the GPU side does not pass "by
    // chance").
    const blocks = planRowBlocks(
      OVERSIZE.queries,
      OVERSIZE.heads * OVERSIZE.keys * 4,
      CORE_STORAGE_BINDING_LIMIT,
    );
    assertEquals(blocks.length, 2, "the smallest block count that fits the limit");
    for (const block of blocks) {
      assert(
        block.rows * OVERSIZE.heads * OVERSIZE.keys * 4 <= CORE_STORAGE_BINDING_LIMIT,
        `block ${block.offset}+${block.rows} does not fit the limit`,
      );
    }
    const gpu = await acquireGpu({
      [LIMIT_CAPS]: { maxStorageBufferBindingSize: CORE_STORAGE_BINDING_LIMIT },
    });
    try {
      assertEquals(
        gpu.limits.maxStorageBufferBindingSize,
        CORE_STORAGE_BINDING_LIMIT,
        "requiredLimits is not capped (without the cap the check verifies nothing)",
      );
      const inputs = attentionInputs(OVERSIZE);
      const fused = await run(gpu, attentionGraph(OVERSIZE), inputs);
      assertEquals(fused.output.shape, [OVERSIZE.heads, OVERSIZE.queries, OVERSIZE.headDim]);
      assert(
        new Set(bits(fused.output)).size > 1,
        "the output is constant (the check never fires)",
      );

      // Proof that the check has teeth: on **the same device with the same shape**, breaking the
      // window down to the plain 9-node sequence tries to materialize all of S and fails. If this
      // stayed green, the green above would not be thanks to row blocking. What fails it is the
      // plan-time limit preflight (ADR 0093 decision 5 — before GPU validation, it lists every
      // intermediate above the limit with its node name).
      await assertRejects(
        () => run(gpu, attentionGraph(OVERSIZE, true), inputs),
        ExecutionError,
        `intermediate buffers exceed the device limits`,
      );
    } finally {
      gpu.destroy();
    }
  },
);
