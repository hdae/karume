/**
 * Per-op-family derivation — the matrix-product family (matmul, bmm, linear, embedding,
 * expert_linear).
 *
 * linear is built from data: the selection names its strategy (src/runtime/linear-strategies.ts),
 * the strategy's dispatch program says its kernels, uniforms, bindings, grids and temporaries
 * without a device (src/runtime/linear-program.ts), and `emitLinearProgram` here turns the program
 * into the step's dispatches. The other ops of the family build their single dispatch directly.
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state, params writes, the weights' storage form and scale bindings) are reached only through
 * {@link RecipeBuildFace}.
 */

import type { BindingRecipe, BindingSource, StepRecipeBuilder, TempSource } from "../recipe.ts";
import {
  EMBEDDING_SCALE_BINDING,
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../../kernels/embedding.ts";
import {
  EXPERT_LINEAR_SCALE_BINDING,
  EXPERT_LINEAR_TABLE_BINDING,
  expertLinearDispatch,
  expertLinearKey,
  expertLinearParams,
  expertLinearWgsl,
} from "../../kernels/expert-linear.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { gemmUsesVec4 } from "../../kernels/gemm.ts";
import type { LinearPrologue, PackedActivations } from "../fusion.ts";
import { type LinearProgram, linearProgramFor, type ProgramOperand } from "../linear-program.ts";
import { selectLinearStrategy } from "../linear-strategies.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../../kernels/bmm.ts";
import { gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { gemmRowsGeometry } from "../../kernels/geometry-profile.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../../kernels/matmul.ts";
import { numel } from "../../ops.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

export const buildMatmul = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [a, b] = step.inputShapes;
  const [m, k] = a;
  const n = b[1];
  // v4 (vec4 reads and writes) is one bit derived from the shape. It is evaluated at derivation
  // and passed to both the key and the WGSL (same key ⇔ same bytes is preserved).
  const v4 = gemmUsesVec4(k, n);
  // MUST: the tile geometry is the bucket of the row count M (the adapter's profile —
  // src/kernels/geometry-profile.ts). Pass the geometry taken once **unchanged** to all three of
  // key, WGSL and dispatch — if even one gets a different geometry, output tiles go missing
  // silently.
  const geometry = gemmRowsGeometry(face.state.geometryProfile, m);
  const key = matmulKey(v4, m, geometry);
  const { pipeline, layout, roles } = await face.state.cache.get(key, matmulWgsl(v4, m, geometry));
  const params = face.writeParams(matmulParams(m, n, k), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `matmul [${a.join(",")}] × [${b.join(",")}]`;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      1,
    ],
  });
};

/**
 * Batched matmul (rank 3). Besides the two tile axes, **the batch goes on the z axis**.
 * MUST: the same "one workgroup = one tile" as matmul, so exceeding the limit fails loudly on all
 * three axes.
 */
export const buildBmm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [a, b] = step.inputShapes;
  const [batch, m, k] = a;
  const n = b[2];
  const v4 = gemmUsesVec4(k, n);
  // The geometry bucket is **the m of one matrix** (the batch is on the z axis, independent of the
  // tile geometry). Key, WGSL and dispatch get the same single return value (the same discipline
  // as matmul).
  const geometry = gemmRowsGeometry(face.state.geometryProfile, m);
  const key = bmmKey(v4, m, undefined, geometry);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    bmmWgsl(v4, m, undefined, geometry),
  );
  const params = face.writeParams(bmmParams(m, n, k, batch), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `bmm [${a.join(",")}] × [${b.join(",")}]`;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      // One workgroup = one batch. Clamping here too would leave whole batches unwritten.
      tiledWorkgroups(batch, 1, limit, where),
    ],
  });
};

/**
 * linear (a fused op — ADR 0012). Flattens the leading dimensions and lowers to the 2D GEMM
 * `[m,k] × [k,n] + bias`. The kernel reads the weight in its transposed `[n,k]` layout as is (no
 * transposing copy).
 *
 * Which kernel computes it is decided in one place, {@link selectLinearStrategy}
 * (src/runtime/linear-strategies.ts); its dispatches are the strategy's program
 * ({@link linearProgramFor} in src/runtime/linear-program.ts) — this function only emits that.
 * MUST: the same "one workgroup = one tile" as matmul, so exceeding the limit fails loudly.
 */
export const buildLinear = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  packed: PackedActivations | undefined,
  /** The producer folded into step ① of the i8a8 path (perf-ledger F-4); `binds[0]` is its input. */
  prologue: LinearPrologue | undefined,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const n = weight[0];
  const k = weight[1];
  const m = numel(x.slice(0, -1));
  const weightStorage = face.weightStorage(step);
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  const choice = selectLinearStrategy({
    plan: step,
    weightStorage,
    groupSize,
    knobs: face.state,
    linearGemv: face.state.geometryProfile.linearGemv,
    packed,
    prologue,
  });
  const program = linearProgramFor(choice, {
    x,
    weight,
    m,
    n,
    k,
    weightStorage,
    groupSize,
    packedScale: packed?.scale,
    prologue,
    geometryProfile: face.state.geometryProfile,
    linearI8a8Dot: face.state.linearI8a8Dot,
  });
  await emitLinearProgram(face, step, binds, outs, builder, program);
};

/**
 * Turns a strategy's {@link LinearProgram} into the step's dispatches: the temporaries are allocated
 * before the first dispatch and released after the last, each dispatch's pipeline comes from the
 * Session's cache under its key, the weight's scale is bound where its storage has one, and each grid
 * is checked against the device limit.
 *
 * MUST: a temporary larger than the device's storage binding limit fails here — binding it would be a
 * validation error at the first run, with nothing naming the linear.
 */
const emitLinearProgram = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  program: LinearProgram,
): Promise<void> => {
  const { limits } = face.state.gpu;
  const limit = limits.maxComputeWorkgroupsPerDimension;
  const temps: TempSource[] = program.temps.map((byteLength) => {
    if (byteLength > limits.maxStorageBufferBindingSize) {
      throw new ExecutionError(
        `${program.where}: a temporary of ${byteLength} bytes exceeds the device's storage binding limit ` +
          `${limits.maxStorageBufferBindingSize}`,
      );
    }
    return builder.allocTemp(byteLength);
  });
  const sourceOf = (operand: Exclude<ProgramOperand, "weightScale">): BindingSource => {
    if (operand === "x") return binds[0];
    if (operand === "weight") return binds[1];
    if (operand === "bias") return binds[2];
    if (operand === "out") return outs[0];
    return temps[operand.temp];
  };
  for (const { key, wgsl, params, bindings, grid } of program.dispatches) {
    const { pipeline, layout, roles } = await face.state.cache.get(key, wgsl());
    builder.dispatch({
      key,
      pipeline,
      layout,
      roles,
      params: face.writeParams(params, PARAMS_UNIFORM_USAGE),
      // The weight's scale is bound only where its storage has a scale buffer (none for f32 / f16 /
      // bf16 — `weightScaleBindings` returns no entry), the condition the WGSL declares it on.
      bindings: bindings.flatMap(({ binding, operand }): readonly BindingRecipe[] =>
        operand === "weightScale"
          ? face.weightScaleBindings(step, binding)
          : [{ binding, source: sourceOf(operand) }],
      ),
      workgroups:
        grid.kind === "tiled"
          ? [
              tiledWorkgroups(grid.axes[0].items, grid.axes[0].tile, limit, program.where),
              tiledWorkgroups(grid.axes[1].items, grid.axes[1].tile, limit, program.where),
              tiledWorkgroups(grid.axes[2].items, grid.axes[2].tile, limit, program.where),
            ]
          : [gridStrideWorkgroups(grid.items, grid.workgroupSize, limit), 1, 1],
    });
  }
  // MUST: return the temporaries at the node boundary (the arena's invariant), last allocated first.
  for (const temp of temps.toReversed()) builder.releaseTemp(temp);
};

/**
 * embedding (row gather). Out-of-range indices follow the ruling in src/kernels/embedding.ts (the
 * GPU poisons with NaN / the CPU reference throws). padding_idx in attrs has no effect on forward,
 * so it is not passed to the kernel.
 */
export const buildEmbedding = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const weight = step.inputShapes[0];
  const count = numel(step.outputs[0].shape);
  const weightStorage = face.weightStorage(step);
  // i4 bakes the group length into the WGSL (paired with the key's g part — the same discipline as
  // linear, ADR 0069).
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  const key = embeddingKey(weightStorage, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    embeddingWgsl(weightStorage, groupSize),
  );
  const params = face.writeParams(
    embeddingParams(count, weight[1], weight[0], groupSize),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    count,
    EMBEDDING_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
      ...face.weightScaleBindings(step, EMBEDDING_SCALE_BINDING),
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * expert_linear (ADR 0145 decisions 2 and 4): one dispatch of the GEMV-form kernel, one thread per
 * output column of one pick row, the expert id read from `index` on the GPU. The T · k rows
 * grid-stride along y, so they may exceed `maxComputeWorkgroupsPerDimension`; T is the run's
 * resolved length (the recipe is derived per binding, as for every op with a symbolic leading dim).
 *
 * MUST: bind as the WGSL declares (src/kernels/expert-linear.ts): 0 dims (`expertLinearParams`),
 * 1 x, 2 index, 3 w, 4 out, and the step table at {@link EXPERT_LINEAR_SCALE_BINDING} for i4 /
 * i4f16 only. MUST: tokens and picks come from index, n / k / E from the weight (the shape rule has
 * matched x against them); the form of x (shared or per pick) is its rank.
 *
 * Under the expert cache (ADR 0146 decision 2) the weight is a pool of K slots: the pooled kernel
 * runs with K in its params and the index's slot table at {@link EXPERT_LINEAR_TABLE_BINDING}. E stays
 * the declaration's (the table's length). Without the cache the key, WGSL, params and bindings are
 * the resident form's, unchanged.
 */
export const buildExpertLinear = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, index, weight] = step.inputShapes;
  const [tokens, picks] = index;
  const [experts, n, k] = weight;
  const storage = face.expertLinearStorage(step);
  // The group length is baked into the WGSL (paired with the key's g part, as linear's i4).
  const groupSize = storage === "f32" ? undefined : face.weightGroupSize(step);
  const pool = face.expertTable(step);
  const pooled = pool !== undefined;
  const key = expertLinearKey(storage, groupSize, pooled);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    expertLinearWgsl(storage, groupSize, pooled),
  );
  const params = face.writeParams(
    expertLinearParams(
      storage,
      {
        tokens,
        picks,
        n,
        k,
        experts,
        sharedX: x.length === 2,
        ...(pool === undefined ? {} : { slots: pool.slots }),
      },
      groupSize,
    ),
    PARAMS_UNIFORM_USAGE,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: binds[2] },
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, EXPERT_LINEAR_SCALE_BINDING),
      ...(pool === undefined
        ? []
        : [
            {
              binding: EXPERT_LINEAR_TABLE_BINDING,
              source: { kind: "resident", buffer: pool.table } as const,
            },
          ]),
    ],
    workgroups: expertLinearDispatch(
      tokens * picks,
      n,
      face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
    ),
  });
};
