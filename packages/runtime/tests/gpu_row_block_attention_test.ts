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
 *
 * The parity cases of ① and ③ live in @karume/kernel-checks (checkRowBlockAttentionTwins), which
 * the playground's /kernel-check page runs on the browser's backend too; the peaks, the block
 * counts, the output shapes, the non-constant outputs and ② stay here.
 */

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import {
  checkRowBlockAttentionTwins,
  ROW_BLOCK_ATTENTION_TWIN_CASES,
  ROW_BLOCK_SHAPES,
  ROW_BLOCK_SPLITS,
  rowBlockAttentionGraph,
  rowBlockAttentionInputs,
  rowBlockBudget,
  type RowBlockShape,
} from "@karume/kernel-checks";
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
  type FilledTensor,
  GRAPH_NAME,
  openGraphModel,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import { countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

/** The WebGPU core default storage binding limit. The portability check reproduces it. */
const CORE_STORAGE_BINDING_LIMIT = 128 * 1024 * 1024;

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
 * The first run of a fused Session (what it allocated) with the output shape of the reference and a
 * check against tautology (if the output is constant, "equal" verifies nothing).
 */
const observeFused = async (
  fused: Session,
  reference: Tensor,
  inputs: Readonly<Record<string, FilledTensor>>,
  label: string,
): Promise<RunResult> => {
  const result = await observe(fused, inputs);
  assertEquals(result.output.shape, reference.shape, label);
  assert(
    new Set(bits(result.output)).size > 1,
    `${label}: the output is constant (bit equality is a tautology)`,
  );
  return result;
};

test.skipIf(!GPU_AVAILABLE)(
  "row-block execution is **bit-identical** to the plain 9-node sequence at every forced block count and split by the attentionRowBlockMaxBytes budget (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkRowBlockAttentionTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(ROW_BLOCK_ATTENTION_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "row-block execution lowers the intermediate peak at every further forced block, with the reference's shape and a non-constant output (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of ROW_BLOCK_SHAPES) {
        const inputs = rowBlockAttentionInputs(shape);
        const reference = await run(gpu, rowBlockAttentionGraph(shape, true), inputs);
        let previousPeak = Number.POSITIVE_INFINITY;
        for (const split of ROW_BLOCK_SPLITS) {
          const fusedSession = await open(gpu, rowBlockAttentionGraph(shape), split);
          try {
            const label = `${shape.name} / ${split} blocks`;
            const fused = await observeFused(fusedSession, reference.output, inputs, label);
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
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * Budget-split parity (ADR 0124) is a case of the shared check above; here the budget's effect.
 * The budget is set to ⌈M/3⌉ rows per block (3 blocks for every shape — 17 rows are 6 + 6 + 5, 68
 * rows are 23 + 23 + 22, 64 rows are 22 + 21 + 21).
 *
 * MUST: do not let this become a tautology. Check at once that the budgeted block count (the
 * dispatch count of the row-window QK) is the independently counted 3, and that the peak is
 * strictly below the unbudgeted fusion (1 block) (a run in 1 block where the budget never reached
 * the fusion's context must not pass).
 */
test.skipIf(!GPU_AVAILABLE)(
  "row-block execution split by the attentionRowBlockMaxBytes budget runs more blocks with a smaller peak (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const shape of ROW_BLOCK_SHAPES) {
        const inputs = rowBlockAttentionInputs(shape);
        const budget = rowBlockBudget(shape);
        const label = `${shape.name} / budget ${budget}B`;
        const reference = await run(gpu, rowBlockAttentionGraph(shape, true), inputs);
        const single = await run(gpu, rowBlockAttentionGraph(shape), inputs);
        assertEquals(single.windowedQkDispatches, 0, `${label}: the unbudgeted run is not 1 block`);
        const budgetedSession = await open(gpu, rowBlockAttentionGraph(shape), undefined, budget);
        let budgeted: RunResult;
        try {
          budgeted = await observeFused(budgetedSession, reference.output, inputs, label);
        } finally {
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
const OVERSIZE: RowBlockShape = {
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
      const inputs = rowBlockAttentionInputs(OVERSIZE);
      const fused = await run(gpu, rowBlockAttentionGraph(OVERSIZE), inputs);
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
        () => run(gpu, rowBlockAttentionGraph(OVERSIZE, true), inputs),
        ExecutionError,
        `intermediate buffers exceed the device limits`,
      );
    } finally {
      gpu.destroy();
    }
  },
);
