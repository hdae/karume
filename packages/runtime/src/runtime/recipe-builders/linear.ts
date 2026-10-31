/**
 * Per-op-family derivation — the matrix-product family (matmul, bmm, linear with its GEMV / i8a8
 * variants, embedding).
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state, params writes, the weights' storage form and scale bindings) are reached only through
 * {@link RecipeBuildFace}.
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import {
  EMBEDDING_SCALE_BINDING,
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../../kernels/embedding.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { type GemmCompute, gemmUsesVec4 } from "../../kernels/gemm.ts";
import {
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../../kernels/linear-i8a8.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvPackedEligible,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelPackedKey,
  linearGemvParallelPackedParams,
  linearGemvParallelPackedWgsl,
  linearGemvPackedStagedEligible,
  linearGemvParallelPackedStagedKey,
  linearGemvParallelPackedStagedWgsl,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
  linearGemvStagedEligible,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
  linearGemvUnit,
  linearGemvWgsl,
} from "../../kernels/linear-gemv.ts";
import { LINEAR_SCALE_BINDING, linearKey, linearParams, linearWgsl } from "../../kernels/linear.ts";
import type { LinearPrologue, PackedActivations } from "../fusion.ts";
import { linearTakesI8a8 } from "../linear-route.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../../kernels/bmm.ts";
import { i8a8TileM, i8a8TileN } from "../../kernels/i8a8-geometry.ts";
import { gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { gemmRowsGeometry } from "../../kernels/geometry-profile.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../../kernels/matmul.ts";
import { numel } from "../../ops.ts";
import {
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsPrologueEligible,
  quantizeRowsPrologueKey,
  quantizeRowsPrologueWgsl,
  quantizeRowsWgsl,
} from "../../kernels/quantize-rows.ts";
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
  const params = face.writeParams(bmmParams(m, n, k), PARAMS_UNIFORM_USAGE);
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
  if (weightStorage === "i2" && face.state.linearCompute !== "f32") {
    throw new ExecutionError("linear: i2 residency supports only linearCompute 'f32' (ADR 0097)");
  }
  // If a linear declared to receive packed activations (ADR 0105) does not land on the parallel
  // GEMV, f32 words flow into the `vec4<u32>` binding = a silent wrong value with no exception.
  // The pairing is accepted in one place, fusion.ts, but it is checked here too **with the same
  // predicate** (fail loudly).
  if (packed !== undefined) {
    const eligible =
      packed.role === "read" &&
      face.state.linearGemvReduce === "parallel" &&
      face.state.linearCompute === "f32" &&
      linearGemvPackedEligible(
        weightStorage,
        m,
        n,
        k,
        weightStorage === "i4" ? face.weightGroupSize(step) : undefined,
      ) !== undefined;
    if (!eligible) {
      throw new ExecutionError(
        `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
          "declared as packed activations (ADR 0105) but does not land on the parallel GEMV",
      );
    }
  }
  // The integer dot-product path (opt-in × integer residency × k > 0 × k % 4 == 0). The predicate
  // and its MUSTs live in one function shared with the fusion pass (src/runtime/linear-route.ts).
  const i8a8 = linearTakesI8a8(face.state.linearCompute, weightStorage, k);
  // If a linear declared to absorb its producer (perf-ledger F-4) does not take the i8a8 path with
  // a row the prologue kernel accepts, the producer's op would never run = a silent wrong value. The
  // fold is accepted in one place, fusion.ts, but it is checked here too **with the same predicate**
  // (fail loudly).
  if (prologue !== undefined && !(i8a8 && quantizeRowsPrologueEligible(k))) {
    throw new ExecutionError(
      `linear [${x.join(",")}] × [${weight.join(",")}]: declared with a ${prologue.op} prologue` +
        " (perf-ledger F-4) but does not take the i8a8 path with a prologue-eligible row",
    );
  }
  if (i8a8) {
    await buildLinearI8a8(face, step, binds, outs, builder, m, n, k, weightStorage, prologue);
    return;
  }
  // The f16 compute variant (ADR 0028). **It cannot be combined with i8-resident weights** (w8a16
  // is not implemented), so it fails here instead of silently dropping to f32 — dropping would give
  // "f16 has no effect on the i8 layers only", which nobody notices without looking at the
  // diagnostic keys.
  const compute: GemmCompute = face.state.linearCompute === "f16" ? "f16" : "f32";
  if (compute === "f16" && weightStorage === "i8") {
    throw new ExecutionError(
      `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
        "linearCompute 'f16' cannot be combined with i8-resident weights (w8a16 is not implemented — ADR 0028). " +
        "Set linearCompute to 'a8', or store this weight as f32 / f16",
    );
  }
  // The same for i4 (w4a16 is not implemented — ADR 0069; the reason for not silently dropping to
  // f32 compute is the same as for i8).
  if (compute === "f16" && weightStorage === "i4") {
    throw new ExecutionError(
      `linear [${x.join(",")}] × [${weight.join(",")}]: ` +
        "linearCompute 'f16' cannot be combined with i4-resident weights (w4a16 is not implemented — ADR 0069). " +
        "Set linearCompute to 'a8', or store this weight as f32 / f16",
    );
  }
  const v4 = gemmUsesVec4(k, n);
  // For i4 the group length affects both the key and the WGSL (the baked-in shift) (ADR 0069 —
  // codegen determinism: same key → byte-identical WGSL).
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  // **Small M (1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS) branches to the GEMV family** (ADR 0082 —
  // perf-ledger K-11 / i8 is K-16 / row blocks for M ≥ 2 are K-21). In the bucket M ≤ 64 (M16N16)
  // the default GEMM skeleton has only 4 of its 64 threads write output, and the double barrier
  // per K tile of 16 exposes the weight-read latency serially (proportional to k, independent of
  // n, the same with L2-resident shapes — not bandwidth starvation). This cost does not depend on
  // M (the GEMM path stays at 65–67 ms for M=2–16 — research 2026-09-07 §7.2). On gemma4 E2B
  // decode: ×8.45 against the default, ×5.0 at M=8, ×2.8 at M=32, ×2.7 at M=64 (census-weighted —
  // research 2026-09-07-gemv-rows-k21).
  // MUST: the reduction order, the spelling of the multiply-add and the order of adding bias are
  // the same as the default path = **bit-identical** (the numeric contract of
  // src/kernels/linear-gemv.ts; the check is tests/gpu_linear_gemv_test.ts).
  // The eligibility conditions:
  // - `1 ≤ m ≤ LINEAR_GEMV_MAX_ROWS` — the range measured with one thread owning the whole
  //   reduction of one output column (× row block). Outside the limit, the default GEMM skeleton
  //   stays.
  // - `i2` / `i4` / `i8` storage × `f32` compute. f16 / f32 storage is taken by the M=1-only
  //   check below. i8 × f16 compute (w8a16) has already failed above.
  // - `groupSize % <unit> === 0` (i4 only) — the condition that the 32 elements of one weight word
  //   do not cross a group (crossing would need more than one scale per word, a silent wrong
  //   value). An i8 scale is one per output channel, so there is nothing to cross.
  // - `k % <unit> === 0` — 16 B alignment of the `vec4<u32>` weight binding (the unit depends on
  //   storage = 32 elements for i4 / 16 for i8). For i4 it follows from the declaration layer's
  //   "row length is a multiple of group_size" (ADR 0069 decision 2), but it is restated as a
  //   binding requirement.
  // - `v4` is **not** a GEMV requirement (the output is written as scalars, so n need not divide),
  //   but every swept real shape was v4, so the eligibility stays inside the measured range. M=1
  //   with n % 4 != 0 keeps the default scalar variant (same values, only the speed as before).
  // f16 / bf16 / f32 storage also go to the same skeleton, at M=1 only (ADR 0082 addenda 6 and 7;
  // bf16 is ADR 0131). Compute stays f32; one weight word = 8 elements for f16 / bf16, 4 for f32.
  // Row blocks and f16 compute are outside this variant's acceptance range.
  if (
    m === 1 &&
    (weightStorage === "f16" || weightStorage === "bf16" || weightStorage === "f32") &&
    compute === "f32" &&
    v4 &&
    k > 0 &&
    k % linearGemvUnit(weightStorage) === 0
  ) {
    await buildLinearGemv(
      face,
      step,
      binds,
      outs,
      builder,
      weightStorage,
      m,
      n,
      k,
      undefined,
      undefined,
    );
    return;
  }
  const i4Unit = linearGemvUnit("i4");
  const gemvRows = m >= 1 && m <= LINEAR_GEMV_MAX_ROWS;
  if (
    gemvRows &&
    weightStorage === "i2" &&
    compute === "f32" &&
    v4 &&
    k % linearGemvUnit("i2") === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i2", m, n, k, undefined, packed);
    return;
  }

  if (
    gemvRows &&
    weightStorage === "i4" &&
    compute === "f32" &&
    v4 &&
    groupSize !== undefined &&
    groupSize % i4Unit === 0 &&
    k % i4Unit === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i4", m, n, k, groupSize, packed);
    return;
  }
  // i8 storage (lm_head — perf-ledger K-16). The same family and variant as i4; the only
  // differences are the element count per word (16) and how the scale is read (one per output
  // channel = bound once, outside the reduction).
  if (
    gemvRows &&
    weightStorage === "i8" &&
    compute === "f32" &&
    v4 &&
    k % linearGemvUnit("i8") === 0
  ) {
    await buildLinearGemv(face, step, binds, outs, builder, "i8", m, n, k, undefined, packed);
    return;
  }
  // MUST: the tile geometry is the bucket of the flattened row count m (the adapter's profile —
  // src/kernels/geometry-profile.ts). Pass the geometry taken once **unchanged** to key, WGSL and
  // dispatch.
  const geometry = gemmRowsGeometry(face.state.geometryProfile, m);
  const key = linearKey(weightStorage, v4, compute, m, groupSize, geometry);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    linearWgsl(weightStorage, v4, compute, m, groupSize, geometry),
  );
  const params = face.writeParams(linearParams(m, n, k), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `linear [${x.join(",")}] × [${weight.join(",")}]`;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      1,
    ],
  });
};

/**
 * linear's **GEMV family** (1 ≤ M ≤ LINEAR_GEMV_MAX_ROWS × i2 / i4 / i8 weights, and M=1 ×
 * f16 / f32 weights — ADR 0082 (f16 / f32 in addenda 6 and 7); i2 is ADR 0097).
 *
 * Bindings, uniform and output storage are the same as the default path; only "which thread owns
 * which output" changes (one thread = one output column (× row block for M ≥ 2), no shared tile
 * and no barrier). The exception is the explicitly requested linearGemvReduce: parallel (ADR 0098),
 * which splits K for the eligible shapes.
 * The K reduction order per output element stays sequential in ascending k, so it is
 * **bit-identical** (the numeric contract of src/kernels/linear-gemv.ts).
 * MUST: M=1 is the M=1 variant (decode output does not move); M ≥ 2 is the row-block variant, whose
 * row count `rows` is the pure function `defaultLinearGemvRowsVariant` of (storage, m, n, the
 * parallelism target) (part of the key). The target is a static knob fixed at Session creation
 * (`SessionOptions.linearGemvRowsThreadTarget`).
 * MUST: pass `groupSize` only for i4 (i8 has no groups — the kernel checks the pairing).
 * MUST: one thread, one output (column × row block), so the dispatch is
 * `[ceil(n / cols), ceil(m / rows), 1]`.
 * It is not grid-stride, so exceeding the limit fails loudly (the same discipline as the default
 * path).
 */
const buildLinearGemv = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  storage: WeightStorage,
  m: number,
  n: number,
  k: number,
  groupSize: number | undefined,
  /** The packed int8 activation pair (ADR 0105). Can be set only for the parallel variant. */
  packed: PackedActivations | undefined,
): Promise<void> => {
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const [x, weight] = step.inputShapes;
  const where = `linear gemv [${x.join(",")}] × [${weight.join(",")}]`;
  // A choice that changes numerics happens only when explicitly requested. The reference path's
  // choice and WGSL stay as before.
  const subgroup = face.state.linearGemvReduce === "parallel-subgroup32";
  const lanes =
    face.state.linearGemvReduce !== "sequential"
      ? linearGemvParallelLanes(storage, m, n, k, groupSize)
      : undefined;
  const rowsVariant =
    m === 1 || lanes !== undefined
      ? undefined
      : defaultLinearGemvRowsVariant(storage, m, n, face.state.linearGemvRowsThreadTarget);
  const variant = rowsVariant ?? defaultLinearGemvVariant({ storage, n, k });
  const rows = rowsVariant?.rows ?? 1;
  // Only the parallel (non-subgroup) variant has a packed activation variant. Having passed
  // `buildLinear`'s guard, this is always `lanes !== undefined && !subgroup`.
  const packedScale = packed === undefined ? undefined : packed.scale;
  // The staged-activation variants (K-90) are bit-identical to their unstaged bodies, so they are
  // taken wherever the measured table marks the shape, without a separate opt-in. A packed shape
  // reads its own predicate, a narrowing of the packed one.
  const staged =
    lanes !== undefined &&
    !subgroup &&
    (packedScale === undefined
      ? linearGemvStagedEligible(storage, m, n, k, groupSize)
      : linearGemvPackedStagedEligible(storage, m, n, k, groupSize)) !== undefined;
  const key =
    lanes !== undefined
      ? subgroup
        ? linearGemvSubgroupKey(storage, groupSize, lanes)
        : packedScale !== undefined
          ? staged
            ? linearGemvParallelPackedStagedKey(storage, groupSize, lanes)
            : linearGemvParallelPackedKey(storage, groupSize, lanes)
          : staged
            ? linearGemvParallelStagedKey(storage, groupSize, lanes)
            : linearGemvParallelKey(storage, groupSize, lanes)
      : rowsVariant === undefined
        ? linearGemvKey(storage, groupSize, variant)
        : linearGemvRowsKey(storage, groupSize, rowsVariant);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    lanes !== undefined
      ? subgroup
        ? linearGemvSubgroupWgsl(storage, groupSize, lanes)
        : packedScale !== undefined
          ? staged
            ? linearGemvParallelPackedStagedWgsl(storage, groupSize, lanes)
            : linearGemvParallelPackedWgsl(storage, groupSize, lanes)
          : staged
            ? linearGemvParallelStagedWgsl(storage, groupSize, lanes)
            : linearGemvParallelWgsl(storage, groupSize, lanes)
      : rowsVariant === undefined
        ? linearGemvWgsl(storage, groupSize, variant)
        : linearGemvRowsWgsl(storage, groupSize, rowsVariant),
  );
  // The uniform is the same three words as the default path (no separate binding layout). The
  // divisibility check lives in one place on the family side. Only the packed variant appends one
  // word, the activation scale, to Dims.
  const params = face.writeParams(
    packedScale === undefined
      ? linearGemvParams(storage, m, n, k, groupSize)
      : linearGemvParallelPackedParams(storage, m, n, k, packedScale, groupSize),
    PARAMS_UNIFORM_USAGE,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(n, lanes === undefined ? variant.cols : 128 / lanes, limit, where),
      tiledWorkgroups(m, rows, limit, where),
      1,
    ],
  });
};

/**
 * The kernel of the i8a8 path's step ①: the plain per-token quantization (one row = one workgroup,
 * or the small-row geometry), or — when the fusion pass folded a producer into this linear — the
 * prologue form, which keeps one row per workgroup in registers (perf-ledger F-4).
 */
const quantizeStep = (
  k: number,
  prologue: LinearPrologue | undefined,
): { readonly key: string; readonly wgsl: string; readonly rowsPerGroup: number } => {
  if (prologue !== undefined) {
    return {
      key: quantizeRowsPrologueKey(prologue.op, k),
      wgsl: quantizeRowsPrologueWgsl(prologue.op, k),
      rowsPerGroup: 1,
    };
  }
  const geometry = quantizeRowsGeometry(k);
  return {
    key: quantizeRowsKey(geometry),
    wgsl: quantizeRowsWgsl(geometry),
    rowsPerGroup: geometry.rowsPerGroup,
  };
};

/**
 * linear's **w8a8 / w4a8 variant** (opt-in — {@link SessionOptions.linearCompute}).
 * **One node = two dispatches** (the same "one node in several dispatches" treatment as fused
 * attention):
 *
 * ① `quantize_rows` (activations to per-token i8, grid-stride over rows) → ② the integer
 * dot-product GEMM (one workgroup = one output tile, so exceeding the limit fails loudly). ① is the
 * same whatever the weights' storage form. With a {@link LinearPrologue} (the producer folded by the
 * fusion pass — perf-ledger F-4), ① quantizes `op(x)` of the re-pointed input instead (same bindings
 * and params, and the same values as "elementwise op → quantize_rows").
 *
 * MUST: the temporary buffers (`xq` / `xs`) are declared → released at the end of the node. That
 * closes the plan's reference counting, and even on a failure path `arena.destroy()` collects the
 * whole region.
 * MUST: the i32 overflow guard fails loudly. Letting it pass would garble values, sign included,
 * through i32 wrap-around.
 * The guard's axis depends on the storage form — for i8 it is **k** (the whole reduction is one
 * i32), for i4 **the group length** (flushing is per group, so only one group's worth sits in an
 * i32).
 */
const buildLinearI8a8 = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  m: number,
  n: number,
  k: number,
  weightStorage: WeightStorage,
  prologue: LinearPrologue | undefined,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const where = `linear i8a8 [${x.join(",")}] × [${weight.join(",")}]`;
  // For i4 the group length affects both the key and the WGSL (the baked-in shift) (ADR 0069).
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  if (groupSize === undefined && k > LINEAR_I8A8_MAX_K) {
    throw new ExecutionError(
      `${where}: k=${k} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K} of the i8a8 path` +
        " (set linearCompute to 'f32', or take this linear off i8 residency)",
    );
  }
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // The quantized activations `xq` (four i8 packed per word) and the per-token scale `xs`.
  // Temporary regions closed inside the node.
  const xq = builder.allocTemp(Math.max(4, m * (k / 4) * 4));
  const xs = builder.allocTemp(Math.max(4, m * 4));

  // ① Per-token quantization of the activations (grid-stride over rows). With a prologue the same
  // quantization of `op(x)`, `x` being the folded producer's input (`binds[0]`); the bindings and
  // params are the plain form's.
  const quantize = quantizeStep(k, prologue);
  const {
    pipeline: quantizePipeline,
    layout: quantizeLayout,
    roles: quantizeRoles,
  } = await face.state.cache.get(quantize.key, quantize.wgsl);
  builder.dispatch({
    key: quantize.key,
    pipeline: quantizePipeline,
    layout: quantizeLayout,
    roles: quantizeRoles,
    params: face.writeParams(quantizeRowsParams(m, k), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: xq },
      { binding: 3, source: xs },
    ],
    workgroups: [gridStrideWorkgroups(m, quantize.rowsPerGroup, limit), 1, 1],
  });

  // ② The integer dot-product GEMM. The tile geometry is the per-op field of the adapter's profile
  // (src/kernels/geometry-profile.ts) — it is part of the key, so "same key → byte-identical WGSL"
  // holds.
  const v4 = linearI8a8UsesVec4(n);
  const geometry = face.state.geometryProfile.i8a8.linear;
  const dp4a = face.state.linearI8a8Dot === "dp4a";
  // MUST: look up both key and wgsl with **the actual resident form** (`weightStorage`). Fixing it
  // to i8 would hand out i4's group scales as per-channel scales, a silent wrong value (the numeric
  // contracts differ — ADR 0076).
  const key = linearI8a8Key(v4, dp4a, geometry, weightStorage, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    linearI8a8Wgsl(v4, dp4a, geometry, weightStorage, groupSize),
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(linearI8a8Params(m, n, k, groupSize), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: xq },
      { binding: 2, source: binds[1] },
      { binding: 3, source: binds[2] },
      { binding: 4, source: outs[0] },
      // Only **whether** it is bound depends on the resident form (i8 / i4) — the buffer itself is
      // looked up by initializer name, so it is the same for i8 / i4. The **interpretation** of the
      // scale (per-channel or group) is decided by the discriminator of `linearI8a8Wgsl` /
      // `linearI8a8Key` above, which is the real guard against silent wrong values.
      ...face.weightScaleBindings(step, LINEAR_SCALE_BINDING),
      { binding: LINEAR_ACT_SCALE_BINDING, source: xs },
    ],
    workgroups: [
      tiledWorkgroups(n, i8a8TileN(geometry), limit, where),
      tiledWorkgroups(m, i8a8TileM(geometry), limit, where),
      1,
    ],
  });

  // MUST: return the temporary buffers at the node boundary (the arena's invariant).
  builder.releaseTemp(xs);
  builder.releaseTemp(xq);
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
