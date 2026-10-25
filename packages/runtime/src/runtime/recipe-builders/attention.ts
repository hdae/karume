/**
 * Per-op-family derivation — every form of attention (fused attention with its i8a8 helpers, the
 * states form / readonly states form, state_append).
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state and params writes) are reached only through {@link RecipeBuildFace}.
 */

import {
  ATTENTION_PV_V_SCALE_BINDING,
  ATTENTION_QK_K_SCALE_BINDING,
  ATTENTION_QK_Q_SCALE_BINDING,
  attentionPvI8a8Key,
  attentionPvI8a8Params,
  attentionPvI8a8UsesVec4,
  attentionPvI8a8Wgsl,
  attentionQkI8a8Key,
  attentionQkI8a8Params,
  attentionQkI8a8UsesVec4,
  attentionQkI8a8Wgsl,
} from "../../kernels/attention-i8a8.ts";
import {
  ATTENTION_ONLINE_ROWS,
  type AttentionOnlineOutput,
  type AttentionOnlineReduce,
  attentionOnlineI8a8Key,
  attentionOnlineI8a8Params,
  attentionOnlineI8a8Wgsl,
} from "../../kernels/attention-online-i8a8.ts";
import {
  ATTENTION_QK_MASK_BINDING,
  type GemmCompute,
  gemmUsesVec4,
  statePvTiledWgsl,
  stateQkTiledWgsl,
} from "../../kernels/gemm.ts";
import {
  ATTENTION_STATS_STRIDE,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsRegCache,
  attentionStatsWgsl,
} from "../../kernels/attention.ts";
import type { BindingSource, StepRecipeBuilder, TempSource } from "../recipe.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { LINEAR_I8A8_MAX_K } from "../../kernels/linear-i8a8.ts";
import {
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
} from "../../codegen/strided.ts";
import { type MaterialisedScoreStorage, scoreStorageBytes } from "../../kernels/score-storage.ts";
import { type AttentionRoute, attentionRoute, type AttentionSeats } from "../attention-route.ts";
import { attentionScale, numel, stateWindow } from "../../ops.ts";
import { gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { type I8a8Geometry, i8a8TileM, i8a8TileN } from "../../kernels/i8a8-geometry.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import {
  type AttentionTokenMajor,
  planRowBlocks,
  type RowBlockLimit,
  rowBlockLimit,
} from "../fusion.ts";
import { planStateAttention } from "../state-attention-plan.ts";
import {
  quantizeRowsBshdKey,
  quantizeRowsBshdParams,
  quantizeRowsBshdWgsl,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../../kernels/quantize-rows.ts";
import {
  TRANSPOSE_TILED_KEY,
  TRANSPOSE_TILED_TILE,
  TRANSPOSE_TILED_WGSL,
  transposeTiledParams,
} from "../../kernels/transpose-tiled.ts";
import {
  stateAppendKey,
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../../kernels/state-append.ts";
import {
  stateAttentionParams,
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  statePvParallelReadonlyWgsl,
  statePvParallelReadonlyWorkgroups,
  statePvParallelWgsl,
  statePvParallelWorkgroups,
  statePvTiledEligible,
  statePvTiledKey,
  statePvTiledParams,
  statePvTiledWorkgroups,
  statePvWgsl,
  statePvWorkgroups,
  stateQkKey,
  stateQkParallelEligible,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateQkParallelReadonlyWgsl,
  stateQkParallelReadonlyWorkgroups,
  stateQkParallelWgsl,
  stateQkParallelWorkgroups,
  stateQkTiledEligible,
  stateQkTiledKey,
  stateQkTiledParams,
  stateQkTiledWorkgroups,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateStatsKey,
  stateStatsParams,
  stateStatsReadonlyKey,
  stateStatsReadonlyWgsl,
  stateStatsReadonlyWorkgroups,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../../kernels/state-attention.ts";
import { stateStatsPvKey, stateStatsPvWgsl } from "../../kernels/state-attention-stats-pv.ts";
import type { StorageRoles } from "../../gpu/pipeline-cache.ts";
import type { ComputePrecision } from "../session-types.ts";
import type { RecipeBuildFace, StateBuildContext } from "../recipe-builder.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/**
 * The resolved shape of fused attention (batch axis with B·H folded / query rows M / key columns
 * N / head width D).
 */
export type AttentionShape = {
  readonly batch: number;
  readonly rows: number;
  readonly cols: number;
  readonly depth: number;
};

/** The resolved pipeline of a stage (①QK / ③PV). */
type AttentionStagePipeline = {
  readonly key: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
  readonly roles: StorageRoles;
};

/**
 * The execution stage of ①QK. Resolved **once, outside the row-block loop** (the i8a8 quantization
 * of the column side does not depend on the block, and the row side's total does not change, so
 * putting it in the loop would only multiply the work by the block count).
 * The temporaries (`qq` / `qs` / `kq` / `ks`) live across the loop, so they are released at the
 * end of the node.
 */
type AttentionQkStage =
  | (AttentionStagePipeline & { readonly kind: "dense" })
  | (AttentionStagePipeline & {
      readonly kind: "i8a8";
      readonly geometry: I8a8Geometry;
      readonly qq: TempSource;
      readonly qs: TempSource;
      readonly kq: TempSource;
      readonly ks: TempSource;
    });

/** The execution stage of ③PV (as for ①QK — `vt` / `vq` / `vs` live across the loop). */
type AttentionPvStage =
  | (AttentionStagePipeline & { readonly kind: "dense" })
  | (AttentionStagePipeline & {
      readonly kind: "i8a8";
      readonly geometry: I8a8Geometry;
      readonly vt: TempSource;
      readonly vq: TempSource;
      readonly vs: TempSource;
    });

/**
 * Which operands the a8 quantisation reads token-major (the absorbed permutes it honours itself —
 * perf-ledger F-3 / F-8, {@link attentionHeadMajorCopies} for the rest), with the head count those
 * reads need and the node's label for failures.
 */
type AttentionLayout = {
  readonly q: boolean;
  readonly k: boolean;
  readonly v: boolean;
  /** `H` (the q side's — a8 rejects GQA, so k / v have the same head count). */
  readonly heads: number;
  readonly where: string;
};

/** The temporaries of one row block of plain attention (allocation order — released in reverse). */
export type AttentionBlock = {
  /** The first query row of the block (the row window's `row_offset`). */
  readonly offset: number;
  /** The block's row count. */
  readonly rows: number;
  /** Score S = `B·H × rows × N × storage width` bytes. */
  readonly scoreBytes: number;
  /** Row statistics = `B·H × rows × ATTENTION_STATS_STRIDE × 4` bytes. */
  readonly statsBytes: number;
};

/**
 * The query-row blocks of plain attention (no states field) and the bytes of one block's
 * temporaries (a pure function).
 *
 * An even split into the smallest count where **both** one S block and one row-statistics block fit
 * `limit` (the device's granted `maxStorageBufferBindingSize`, or the result of
 * {@link rowBlockLimit} when a budget applies); there is no run-time autotuning (ADR 0022). The
 * limit is per temporary, not the sum of S and row statistics (ADR 0124 decision 1).
 *
 * MUST: the count is decided by the **larger** of the per-row sizes of the row-proportional
 * temporaries (S and row statistics). Deciding by S alone would, for shapes where one row of scores
 * is smaller than two words of row statistics (N = 1 in f32, N < 4 in f16), let S fit while one
 * row-statistics block exceeds the limit, rejecting at allocation a shape that splitting would
 * pass. The i8a8 quantization temporaries are allocated outside the block loop, so they are not
 * included here.
 *
 * @param forced States the row-block count explicitly (the entry point for the test-only
 *   `ROW_BLOCK_SPLIT`).
 */
export const planAttentionBlocks = (
  extent: {
    /** `B·H` (the axis with batch and heads folded). */
    readonly batch: number;
    /** Query rows `M`. */
    readonly rows: number;
    /** Key columns `N`. */
    readonly cols: number;
    /** Bytes per element of S. */
    readonly scoreBytes: number;
  },
  limit: number | RowBlockLimit,
  forced?: number,
): readonly AttentionBlock[] => {
  const { batch, rows, cols, scoreBytes } = extent;
  const statsRowBytes = ATTENTION_STATS_STRIDE * 4;
  const rowBytes = batch * Math.max(cols * scoreBytes, statsRowBytes);
  return planRowBlocks(rows, rowBytes, limit, forced).map((block) => ({
    offset: block.offset,
    rows: block.rows,
    scoreBytes: batch * block.rows * cols * scoreBytes,
    statsBytes: batch * block.rows * statsRowBytes,
  }));
};

/**
 * The work budget of one online-attention dispatch (ADR 0132 decision 2), in multiply-accumulates:
 * `B·H · window rows · N · D · 2` (the score and the P̃·V product each take D per key). 2⁴⁰ gives
 * 6,528-row windows at the Wan2.2 121-frame shape (24 heads × 27,280 keys × D = 128 — 5 windows per
 * node, 2,448 workgroups each), about 70 ms per dispatch on the RTX 3080 Ti, well under the 1 s
 * submit gate. Smaller windows cost occupancy: 768-row windows (288 workgroups on 80 SMs) ran 15%
 * slower than one dispatch in the spike of 2026-10-23, 6,528-row windows within 1%.
 */
export const ATTENTION_ONLINE_WINDOW_MACS = 2 ** 40;

/** One window of query rows of the online kernel (the dispatch's `row_offset` and its row count). */
export type AttentionOnlineWindow = { readonly offset: number; readonly rows: number };

/**
 * The query-row windows of the online kernel (a pure function — ADR 0132 decision 2). Not a capacity
 * device (the kernel has no S temporary) but a dispatch-size one: each window holds at most
 * {@link ATTENTION_ONLINE_WINDOW_MACS} of work, rounded down to a multiple of
 * {@link ATTENTION_ONLINE_ROWS} rows, and never fewer than {@link ATTENTION_ONLINE_ROWS} rows (a
 * key axis so long that one workgroup row exceeds the budget still runs, one workgroup row per
 * window). The last window takes the remainder.
 *
 * MUST: every window but the last is a multiple of {@link ATTENTION_ONLINE_ROWS}. The kernel's
 * workgroups cover 64 rows from `row_offset` and clamp only at M, so a window ending mid-workgroup
 * would also compute (and write) the first rows of the next window.
 * MUST: execution ({@link buildAttention}) and the estimate (estimate.ts — ADR 0124 decision 7) take
 * the windows from this one function.
 */
export const planAttentionOnlineWindows = (
  rows: number,
  extent: {
    /** `B·H` (the axis with batch and heads folded). */
    readonly batch: number;
    /** Key columns `N`. */
    readonly cols: number;
    /** Head width `D`. */
    readonly depth: number;
  },
): readonly AttentionOnlineWindow[] => {
  const groupMacs = extent.batch * extent.cols * extent.depth * 2 * ATTENTION_ONLINE_ROWS;
  const windowRows =
    Math.max(1, Math.floor(ATTENTION_ONLINE_WINDOW_MACS / groupMacs)) * ATTENTION_ONLINE_ROWS;
  const windows: AttentionOnlineWindow[] = [];
  for (let offset = 0; offset < rows; offset += windowRows) {
    windows.push({ offset, rows: Math.min(windowRows, rows - offset) });
  }
  return windows;
};

/**
 * Bytes per element of fused attention's S (a pure function). f16 compute holds S as `array<f16>`
 * (2 B); otherwise it is S's storage form (`route.score` decided by {@link attentionRoute} — 2 B
 * for s16, 4 B for f32).
 *
 * MUST: execution ({@link buildAttention}) and the estimate (estimate.ts — ADR 0124 decision 7)
 * take it from this one function.
 */
export const attentionScoreElementBytes = (
  attentionCompute: ComputePrecision,
  route: { readonly score: MaterialisedScoreStorage },
): number => (attentionCompute === "f16" ? 2 : scoreStorageBytes(route.score));

/**
 * Bytes of fused attention's i8a8 quantization temporaries (a pure function). The ①QK side has
 * `qq` / `qs` / `kq` / `ks`, the ③PV side `vt` / `vq` / `vs`. All are allocated outside the
 * row-block loop and live until the end of the node.
 *
 * MUST: execution (`prepareAttentionQkI8a8` / `prepareAttentionPvI8a8`) and the estimate
 * (estimate.ts — ADR 0124 decision 7) take them from this one function (rewriting the formula would
 * let the estimate keep claiming a different number against an implementation where only one side
 * was fixed). i8a8 cannot be combined with GQA ({@link fusedAttentionViolation}), so the k / v side
 * is also counted with the q side's `B·H`.
 */
export const attentionI8a8TempBytes = (
  shape: AttentionShape,
): {
  readonly qk: {
    readonly qq: number;
    readonly qs: number;
    readonly kq: number;
    readonly ks: number;
  };
  readonly pv: { readonly vt: number; readonly vq: number; readonly vs: number };
} => {
  const { batch, rows, cols, depth } = shape;
  return {
    // The quantized q / k (four i8 packed per word) and the per-token scales.
    qk: {
      qq: Math.max(4, batch * rows * depth),
      qs: Math.max(4, batch * rows * 4),
      kq: Math.max(4, batch * cols * depth),
      ks: Math.max(4, batch * cols * 4),
    },
    // Vᵀ (f32 — the strided permute of a head-major V, or the tiled transpose of a token-major one)
    // and its quantization result.
    pv: {
      vt: Math.max(4, batch * depth * cols * 4),
      vq: Math.max(4, batch * depth * cols),
      vs: Math.max(4, batch * depth * 4),
    },
  };
};

/**
 * The permutation between a token-major `[B, S, H, D]` tensor and its head-major `[B, H, S, D]` form
 * (its own inverse) — the `[0, 2, 1, 3]` permute a fused attention absorbs (perf-ledger F-3 A4 /
 * F-8, {@link AttentionTokenMajor}).
 */
const HEAD_SWAP: readonly number[] = [0, 2, 1, 3];

/** A shape with its two middle axes swapped (token-major ↔ head-major). */
const swapHeads = (shape: readonly number[]): readonly number[] =>
  HEAD_SWAP.map((axis) => shape[axis]);

/** The token-major flags of q / k / v, by operand slot (`undefined` = all head-major). */
const tokenMajorOperands = (
  tokenMajor: AttentionTokenMajor | undefined,
): readonly [boolean, boolean, boolean] => [
  tokenMajor?.q === true,
  tokenMajor?.k === true,
  tokenMajor?.v === true,
];

/**
 * The head-major shapes of a fused attention node's q / k / v (a pure function): a token-major
 * operand ({@link AttentionTokenMajor}) arrives with the absorbed permute's input shape `[B, S, H, D]`,
 * and every derivation of the node (B·H, M, N, D, the guards and the estimate) reads the operand's
 * `[B, H, S, D]`.
 */
export const attentionHeadMajorShapes = (
  inputShapes: readonly (readonly number[])[],
  tokenMajor: AttentionTokenMajor | undefined,
): readonly [readonly number[], readonly number[], readonly number[]] => {
  const [q, k, v] = tokenMajorOperands(tokenMajor).map((swapped, slot) =>
    swapped ? swapHeads(inputShapes[slot]) : inputShapes[slot],
  );
  return [q, k, v];
};

/**
 * The head-major copies a fused attention node makes as node temporaries for its token-major operands
 * (the permutes the fusion pass let it absorb — {@link AttentionTokenMajor}), in bytes per operand
 * (`undefined` = no copy). A pure function.
 *
 * The a8 quantisation reads a token-major q / k (①QK on `i8a8`) or V (③PV on `i8a8`) itself, and the
 * online kernel stores O token-major; every other token-major operand — a dense stage (the f32 / f16
 * presets, or a stage the shape keeps off i8a8) and O on the three-dispatch route — is copied with
 * the strided permute the absorbed node ran, so that route's work and values are the graph's byte for
 * byte (the copy of an input before the first stage reads it, the copy of O after the last ③PV).
 *
 * MUST: execution ({@link buildAttention}) and the estimate (estimate.ts — ADR 0124 decision 7) take
 * them from this one function.
 */
export const attentionHeadMajorCopies = (
  tokenMajor: AttentionTokenMajor | undefined,
  route: AttentionRoute,
  shapes: {
    /** The node's bound inputs (a token-major operand has its token-major shape). */
    readonly inputs: readonly (readonly number[])[];
    /** The node's bound output (token-major with `o`). */
    readonly output: readonly number[];
  },
): Readonly<Record<"q" | "k" | "v" | "o", number | undefined>> => {
  const [q, k, v] = tokenMajorOperands(tokenMajor);
  const bytes = (copied: boolean, shape: readonly number[]): number | undefined =>
    copied ? Math.max(4, numel(shape) * 4) : undefined;
  return {
    q: bytes(q && route.qk !== "i8a8", shapes.inputs[0]),
    k: bytes(k && route.qk !== "i8a8", shapes.inputs[1]),
    v: bytes(v && route.pv !== "i8a8", shapes.inputs[2]),
    o: bytes(tokenMajor?.o === true && route.online === undefined, shapes.output),
  };
};

/**
 * The strided `[0, 2, 1, 3]` permute of an `srcShape` tensor from `source` into `target` — the same
 * key, params and workgroups as `buildStridedCopy` (recipe-builders/layout.ts) gives the permute node
 * a fused attention absorbed, so a route that cannot read or write token-major makes the very copy
 * the graph made.
 */
const permuteHeads = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  source: BindingSource,
  target: BindingSource,
  srcShape: readonly number[],
): Promise<void> => {
  const spec = { dtype: "f32" } as const;
  const key = stridedKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(key, stridedWgsl(spec));
  const outShape = swapHeads(srcShape);
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(
      stridedParams(outShape, permuteSrcStrides(srcShape, HEAD_SWAP), 0),
      PARAMS_STORAGE_USAGE,
    ),
    bindings: [
      { binding: 1, source },
      { binding: 2, source: target },
    ],
    workgroups: [
      gridStrideWorkgroups(
        numel(outShape),
        STRIDED_WORKGROUP_SIZE,
        face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
      ),
      1,
      1,
    ],
  });
};

/**
 * The combination guard of fused attention (the violation message — `undefined` when there is
 * none). Execution ({@link buildAttention}) checks it before taking temporaries, and the estimate
 * (estimate.ts — ADR 0124 decision 7) before counting, through the same function (so that the
 * estimate does not return a number for a combination execution rejects).
 *
 * - mask × a8: the i8a8 variant of ①QK has no mask epilogue (ADR 0023 / docs/limitations.md).
 * - GQA × a8: not supported (ADR 0067 decision 3).
 * - The i8a8 i32 reduction guard: when D (for ①) or N (for ③) exceeds {@link LINEAR_I8A8_MAX_K},
 *   the i32 wraps around.
 *
 * MUST: mask × a8 and GQA × a8 are checked against **the requested mode** (`i8a8`) — checking
 * `route` (which includes the D % 4 / N % 4 eligibility) would let only the shapes that fall back
 * slip past the rejection and run as a mix of f32 ①QK and i8a8 ③PV.
 * The reduction guard applies only to the stages that actually take i8a8 (`route`).
 */
export const fusedAttentionViolation = (
  where: string,
  shape: {
    /** Head count `H` on the q side. */
    readonly heads: number;
    /** Head count `Hkv` on the k / v side (GQA when it differs from `H`). */
    readonly kvHeads: number;
    readonly depth: number;
    readonly cols: number;
  },
  request: {
    readonly i8a8: boolean;
    readonly mask: boolean;
    readonly route: Pick<ReturnType<typeof attentionRoute>, "qk" | "pv">;
  },
): string | undefined => {
  const { heads, kvHeads, depth, cols } = shape;
  const { i8a8, mask, route } = request;
  if (mask && i8a8) {
    return (
      `${where}: attention with an additive mask cannot be combined with attentionCompute 'a8'` +
      " (the i8a8 variant of ①QK has no mask epilogue — set attentionCompute to 'f32' or " +
      "'f16')"
    );
  }
  const kvRepeat = heads / kvHeads;
  // DECIDED: GQA × i8a8 starts as fail loudly (docs/decisions/0067 decision 3). i8a8 has five
  // head bases (src/kernels/attention-i8a8.ts), and the quantization and allocation of K / V also
  // assume `B·H`, so passing a GQA shape through would mix up the quantization buffers.
  // MUST: do not drop silently to the f32 path (ADR 0058 decision 3 — performance would change
  // silently).
  if (kvRepeat > 1 && i8a8) {
    return (
      `${where}: GQA (H=${heads} / Hkv=${kvHeads} → r=${kvRepeat}) × i8a8 is not supported` +
      " (ADR 0067 decision 3, support planned — set attentionCompute to 'f32' or 'f16')"
    );
  }
  // MUST: the reduction guard fails loudly (letting it pass would garble values, sign included,
  // through i32 wrap-around. Against the measured shapes D ≤ 384 and N ≤ 16,384 the guard has
  // orders of magnitude of headroom, but without it there is nothing to catch a regression).
  if (route.qk === "i8a8" && depth > LINEAR_I8A8_MAX_K) {
    return (
      `${where}: D=${depth} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K} of the i8a8 path` +
      " (set attentionCompute to 'f32')"
    );
  }
  if (route.pv === "i8a8" && cols > LINEAR_I8A8_MAX_K) {
    return (
      `${where}: N=${cols} exceeds the i32 reduction guard ${LINEAR_I8A8_MAX_K} of the i8a8 path` +
      " (set attentionCompute to 'f32')"
    );
  }
  return undefined;
};

/**
 * The guard for combining the states form / readonly states form of attention with numerics
 * variants (the violation message — `undefined` when there is none). Execution
 * ({@link buildStateAttention} / {@link buildReadonlyStateAttention}) and the estimate
 * (estimate.ts — ADR 0124 decision 7) check it through the same function.
 *
 * DECIDED: numerics variant × states form starts as fail loudly (ADR 0058 decision 3 — "an
 * unimplemented combination fails loudly instead of falling back". Running silently in f32 would
 * erase the meaning of the opt-in from both the diagnostics and the numbers).
 */
export const stateAttentionVariantViolation = (
  where: string,
  form: "states" | "readonly",
  seats: AttentionSeats,
): string | undefined => {
  if (seats.attentionCompute !== "f32") {
    return form === "states"
      ? `${where}: states-form attention cannot be combined with attentionCompute '${seats.attentionCompute}'` +
          " (only the separate f32 kernel family — ADR 0067 decisions 3 / 4)"
      : `${where}: readonly attention cannot be combined with attentionCompute ` +
          `'${seats.attentionCompute}' (only the separate f32 kernel family)`;
  }
  if (seats.attentionScoreStorage !== "f32") {
    const subject = form === "states" ? "states-form attention" : "readonly attention";
    return (
      `${where}: ${subject} cannot be combined with attentionScoreStorage ` +
      `'${seats.attentionScoreStorage}' (S is stored as f32 only)`
    );
  }
  return undefined;
};

/**
 * Fused attention (ADR 0023). **One node = three dispatches per row block** (the same "one node in
 * several dispatches" treatment as `cat`; full-write holds **for the node's output O**):
 *
 * ① QK gemm (materialises S; the scale is applied to both q and k when filling tiles) → ② row
 * statistics (m and 1/Σexp) → ③ PV gemm (evaluates `exp(S−m)·inv` when filling the A tile = P is
 * not materialised).
 *
 * ## Block execution over query rows (bounding the materialised width of S)
 *
 * S takes `B·H · M · N · storage width` bytes and grows **quadratically in the symbolic dimension
 * S** (on a 1824×1248 DiT, S = 8892 → 2.53GB even with s16 = beyond D3D12's fixed 2GiB limit). So
 * the query rows are split evenly by {@link planRowBlocks} (ADR 0060 / the same pure function as
 * the states-form attention) into "the smallest count where each block fits
 * `maxStorageBufferBindingSize`", and ①②③ run per block. There is no run-time autotuning
 * (ADR 0022) — the count is decided from the device's granted limit and the resolved shapes only.
 *
 * The basis of bit identity is the same as the decomposed path's row blocks (`rowBlockAttention` in
 * fusion.ts): ①③ change **only the assignment of rows** (`GemmRowWindow` in
 * {@link "../../kernels/gemm.ts"}), and ② and ③'s `exp(S−m)·inv` are **closed within a row**, so
 * neither the op sequence nor the order of rounding per row moves.
 * MUST: **with n = 1, no row window is set** — the key, uniform, generated WGSL and dispatch list
 * are exactly the same as before splitting (the existing snapshots and the bit-identity check are
 * its detectors).
 *
 * The optional fourth input `mask[1,1,M,N]` (additive) **only adds one binding to ①**; neither the
 * dispatch count nor the paths of ②③ change (S comes out already masked).
 *
 * GQA (`H % Hkv == 0` — ADR 0067) is an axis of **one key word and one uniform word of ①③ only**;
 * neither the dispatch count nor the allocation changes (only the K / V base maps to the kv-head
 * via `wid.z / r`). Its combination with i8a8 is not supported and fails loudly (decision 3).
 *
 * In the i8a8 variant (opt-in) **the four quantization dispatches move outside the block loop**,
 * and the loop keeps three dispatches (the ① and ③ GEMMs switch to i8a8 kernels). **Eligibility is
 * decided per stage independently** (① needs `D % 4 == 0`, ③ needs `N % 4 == 0`), so a **mix**
 * with only one of them in i8a8 can occur — only the stage that does not qualify falls back to the
 * f32 path **silently** (the same manner as linear's `k % 4`; the fallback shows only in the
 * diagnostic pipeline key — the family check detects it with the full-eligibility check
 * `assertAttentionSeatsComplete`, ADR 0058 addendum of 2026-10-10).
 *
 * With `attentionScoreStorage: "none"` (ADR 0132) and a route that admits the online kernel, the
 * node is instead the same four quantization dispatches plus one online-kernel dispatch per window
 * of query rows ({@link buildAttentionOnline}) — no S, no row statistics, no row blocks.
 *
 * ## Token-major operands (the absorbed permutes — perf-ledger F-3 A4 / F-8)
 *
 * When the fusion pass let the node absorb the `[0, 2, 1, 3]` permutes around it
 * ({@link AttentionTokenMajor} — decided without the route), the absorbed operands arrive token-major
 * `[B, S, H, D]` and, with `o`, O leaves token-major `[B, M, H, D]`. Every route honours it, so the
 * decision never needs the route (`attentionRoute` stays the one decision):
 *
 * - the a8 quantisation (shared by the online and the three-dispatch routes — the MUST of
 *   {@link buildAttentionOnline}) reads a token-major q / k with the token-major `quantize_rows`
 *   and builds Vᵀ from a token-major V with the tiled transpose (one transpose per batch,
 *   `[N, H·D]` → `[H·D, N]`), instead of the strided copy;
 * - the online kernel stores O token-major;
 * - anything else (a dense stage, O on the three-dispatch route) gets the strided permute the absorbed
 *   node ran, into a node temporary ({@link attentionHeadMajorCopies}) — the graph's work byte for
 *   byte.
 *
 * MUST: ① and ③ are one workgroup = one tile, so exceeding the limit fails loudly on all three axes
 * ({@link tiledWorkgroups}). Clamping would drop tiles and leave part of O unwritten (with a
 * reallocated buffer, the previous values) with no exception. Only ② is grid-stride over rows.
 * MUST: the temporary buffers (S / row statistics) are allocated and returned **per block** (taking
 * them for all blocks at once would defeat the point of staying under the limit). Only the
 * quantization temporaries live across the loop, returned at the end of the node in reverse
 * allocation order. That closes the plan's reference counting, and even on a failure path
 * `arena.destroy()` collects the whole region (allocation and destruction in one place — ADR 0004).
 */
export const buildAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  /** The permutes the node absorbed (perf-ledger F-3 A4 / F-8 — accepted in one place, fusion.ts). */
  tokenMajor: AttentionTokenMajor | undefined,
): Promise<void> => {
  // The operands' head-major shapes (a token-major operand is the absorbed permute's input).
  const tokenMajorIns = tokenMajorOperands(tokenMajor);
  const [q, k, v] = attentionHeadMajorShapes(step.inputShapes, tokenMajor);
  // B and H fold into one batch axis (the contract is rank-4 head-first; B matches across all
  // three, H is the q side's).
  const batch = q[0] * q[1];
  const rows = q[2];
  const cols = k[2];
  const depth = q[3];
  const scale = attentionScale(step.node.attrs, `nodes (${step.node.op})`);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `attention [${q.join(",")}] × [${k.join(",")}] × [${v.join(",")}]`;

  // GQA (divisible broadcast — ADR 0067 decisions 1 / 2). `r = H / Hkv` is **a derived value** (no
  // attrs field); the shape check (src/ops/shapes.ts) guarantees divisibility. Allocation, dispatch
  // and S / O / row statistics all keep the q side's `B·H`; **only the reads of K / V** map to the
  // kv-head.
  // SHOULD (re-graded 2026-10-18 from MUST — ADR 0067 decision 2): with `r = 1`, do not set the GQA
  // bit (the key and the generated WGSL are byte-identical).
  const kvHeads = k[1];
  const kvRepeat = q[1] / kvHeads;
  const gqa = kvRepeat > 1;

  // The f16 compute variant (ADR 0028). The three kernels switch **together** — S's storage form
  // has to match between ①, which writes it, and ②③, which read it, so a per-stage mix cannot
  // occur.
  const compute: GemmCompute = face.state.attentionCompute === "f16" ? "f16" : "f32";
  // The i8a8 variant (design §7 waves 1 + 2 — research 2026-08-04-attention-a8-design). **② row
  // statistics stay f32** (S's storage form also stays f32, so the S that ②③ read has the same form
  // whichever path ① wrote it with).
  // S's **storage form** (option γ wave 1 — a second axis orthogonal to the compute form).
  // The per-stage eligibility (① by D, ③ by N, independently) and the eligibility of S's storage
  // form (all three kernels together) are owned by `attentionRoute`
  // (src/runtime/attention-route.ts); ineligible stages fall back to the f32 path **silently**
  // (ADR 0058 decision 3 — detected by the family check's full-eligibility check).
  // MUST: do not write the decision back here (the acceptance-set tests go through the same
  // function, so the decisions cannot drift apart).
  const i8a8 = face.state.attentionCompute === "a8";
  const route = attentionRoute(face.state, depth, cols, face.state.attentionDevice);

  // The additive mask (an optional fourth input — the contract is in src/ops.ts). An axis of ①'s
  // epilogue only; to ②③ it looks like "S is already masked".
  const mask = binds[3];
  // MUST: the combination guards (mask × a8, GQA × a8, the i8a8 i32 reduction) fail loudly **before
  // any temporary buffer is taken**. i8a8's ①QK is a separate kernel
  // (src/kernels/attention-i8a8.ts) without an epilogue; dropping silently to the f32 path would
  // give the silence of "asked for i8a8 and it has no effect", and dropping the mask would break
  // the values. The decision is the one function shared with the estimate
  // ({@link fusedAttentionViolation} — the rule of checking the requested mode lives there too).
  const violation = fusedAttentionViolation(
    where,
    { heads: q[1], kvHeads, depth, cols },
    { i8a8, mask: mask !== undefined, route },
  );
  if (violation !== undefined) throw new ExecutionError(violation);

  // The online kernel (`attentionScoreStorage: "none"` — ADR 0132): the route chose it for this
  // shape and device, so S is never materialised and none of the three dispatches below run. The
  // guards above apply unchanged (the requested mode is a8).
  // The token-major operands this route cannot read or write itself, copied to and from head-major
  // node temporaries (the one function shared with the estimate). The online route has none.
  const copies = attentionHeadMajorCopies(tokenMajor, route, {
    inputs: step.inputShapes,
    output: step.outputs[0].shape,
  });
  const layout: AttentionLayout = {
    q: tokenMajorIns[0] && copies.q === undefined,
    k: tokenMajorIns[1] && copies.k === undefined,
    v: tokenMajorIns[2] && copies.v === undefined,
    heads: q[1],
    where,
  };
  if (route.online !== undefined) {
    await buildAttentionOnline(
      face,
      builder,
      binds,
      outs[0],
      { batch, rows, cols, depth },
      { scale, reduce: route.online, where, output: tokenMajor?.o === true ? "bshd" : "bhsd" },
      layout,
    );
    return;
  }
  const scoreStorage = route.score;
  // The head-major copies of the token-major inputs the stages below read as stored (before the first
  // stage — the permute the absorbed node ran), and the head-major O that ③PV writes before the
  // permute out. MUST: allocated before the stage temporaries and released after them, at the end of
  // the node (reverse allocation order — ADR 0004).
  const inputCopies = [copies.q, copies.k, copies.v].map((byteLength, slot) => {
    return byteLength === undefined ? undefined : { byteLength, source: binds[slot], slot };
  });
  const headMajorIns: (TempSource | undefined)[] = [];
  for (const copy of inputCopies) {
    if (copy === undefined) {
      headMajorIns.push(undefined);
      continue;
    }
    const temp = builder.allocTemp(copy.byteLength);
    await permuteHeads(face, builder, copy.source, temp, step.inputShapes[copy.slot]);
    headMajorIns.push(temp);
  }
  // What the dense stages bind for q / k / v (the copy when one was made).
  const operands = binds.map((source, slot) => headMajorIns[slot] ?? source);

  // Block split over query rows (the same pure function as ADR 0060 — the same manner as the
  // states-form attention). An even split into the smallest count where one S block = `B·H · block
  // · N · storage width` bytes and one row-statistics block fit the smaller of the storage binding
  // limit and the budget (`attentionRowBlockMaxBytes` — ADR 0124) ({@link planAttentionBlocks});
  // there is no run-time autotuning (ADR 0022).
  // A shape where even one row does not fit fails loudly.
  // MUST: **a machine with n = 1 is exactly the same as before splitting** (no row window, so
  // neither the key, the uniform, the WGSL nor the dispatch list moves by a byte). The existing
  // snapshots and the bit-identity check are its detectors.
  const scoreBytes = attentionScoreElementBytes(face.state.attentionCompute, route);
  const blocks = planAttentionBlocks(
    { batch, rows, cols, scoreBytes },
    rowBlockLimit(
      face.state.gpu.limits.maxStorageBufferBindingSize,
      face.state.attentionRowBlockMaxBytes,
    ),
    face.state.rowBlockSplit,
  );
  const windowed = blocks.length > 1;

  // The geometry does not depend on the block's row count; it is a field of the adapter's profile
  // (src/kernels/geometry-profile.ts), so the ①③ pipelines are shared by every block (blocks differ
  // only in the uniform's `row_offset` / `m`).
  // ① and ③ **may use different geometries** (for the same reason the i8a8 variant always had a
  // geometry per stage): the stages exchange only the materialised S and row statistics; ①'s
  // dispatch (with ①'s geometry) writes every element of S, and ②③ only read it by row and column
  // index — nothing shares tile edges across stages.
  // MUST: pass the geometry taken once per stage **unchanged** to all three of that stage's key,
  // WGSL and dispatch.
  const qkGeometry = face.state.geometryProfile.attention.qk;
  const pvGeometry = face.state.geometryProfile.attention.pv;
  const hasMask = mask !== undefined;
  const qkV4 = gemmUsesVec4(depth, cols);
  const qkKey = attentionQkKey(qkV4, compute, scoreStorage, hasMask, gqa, windowed, qkGeometry);
  // The i8a8 quantization (①'s k / ③'s Vᵀ = **the column side**) does not depend on the block, so
  // it runs **once, outside the loop**.
  // MUST: do not move it into the loop (same values, work multiplied by the block count — a pure
  // performance regression).
  // The q side (qq / qs) is also work proportional to the rows with an unchanged total, so it too
  // stays outside the loop, and ①QK reads it from the full-M stride through the row window
  // (quantizing per block would need a row gather over the full-M stride in `quantize_rows`,
  // touching the one kernel shared with linear).
  // NOTE: the quantization temporaries live across the loop, so even with n = 1 more is alive at
  // once. The increase is `O(B·H·(M+N)·D)`, one order lower than S's `O(B·H·M·N)` (measured = anima
  // DiT self-attention at S=8892: 122.8MiB against 2,413.0MiB for one S = +5.1%). **What the limit
  // sees is the size of each buffer**, so the peak against the binding limit stays one S block.
  const qk: AttentionQkStage =
    route.qk === "i8a8"
      ? await prepareAttentionQkI8a8(
          face,
          builder,
          binds,
          scoreStorage,
          { batch, rows, cols, depth },
          windowed,
          layout,
        )
      : {
          kind: "dense",
          key: qkKey,
          ...(await face.state.cache.get(
            qkKey,
            attentionQkWgsl(qkV4, compute, scoreStorage, hasMask, gqa, windowed, qkGeometry),
          )),
        };
  const pvV4 = gemmUsesVec4(cols, depth);
  const pvKey = attentionPvKey(pvV4, compute, scoreStorage, gqa, windowed, pvGeometry);
  const pv: AttentionPvStage =
    route.pv === "i8a8"
      ? await prepareAttentionPvI8a8(
          face,
          builder,
          binds,
          scoreStorage,
          { batch, rows, cols, depth },
          windowed,
          layout,
        )
      : {
          kind: "dense",
          key: pvKey,
          ...(await face.state.cache.get(
            pvKey,
            attentionPvWgsl(pvV4, compute, scoreStorage, gqa, windowed, pvGeometry),
          )),
        };
  // ② Row statistics — one row = one workgroup, grid-stride over rows (the same shape as softmax).
  // The regcache variant that reads S only once is generated per dim, so `epc` adds a key axis
  // (both give bit-identical values — src/kernels/attention.ts). Blocking keeps it **closed within
  // a row**, so the plain kernel runs as is on block-relative S and row statistics (no row offset
  // needed).
  const statsRegCache = attentionStatsRegCache(cols);
  const statsKey = attentionStatsKey(compute, scoreStorage, statsRegCache);
  const stats = await face.state.cache.get(
    statsKey,
    attentionStatsWgsl(compute, scoreStorage, statsRegCache),
  );
  // Where ③PV writes O: the node's output, or the head-major temporary a token-major O is permuted
  // out of after the loop.
  const headMajorOut = copies.o === undefined ? undefined : builder.allocTemp(copies.o);
  const pvOut = headMajorOut ?? outs[0];

  for (const block of blocks) {
    // S[batch, block, N] and row statistics [batch·block, 2]. O is already allocated by the
    // execution phase, so the peak is O + S + statistics (one buffer less than the decomposed
    // path's three: S + P + the identity expand).
    // In the f16 variants (`:c16`'s array<f16> / s16's pack2x16float) S takes half the bytes
    // (1,073.7MB → 536.9MB on a 1024px DiT).
    // MUST: temporaries are allocated and returned **per block** (taking them for all blocks at
    // once would defeat the point of staying under the limit — the same discipline as the
    // states-form attention).
    const scores = builder.allocTemp(block.scoreBytes);
    const rowStats = builder.allocTemp(block.statsBytes);
    // The two row-window words (not passed for n = 1 = the uniform's bytes stay as before).
    const window = windowed ? { offset: block.offset, rowsFull: rows } : undefined;

    // ① QK gemm — the reduction dimension is D, the output columns are N. The row window `"a"` only
    // reads q from row `row_offset` of the full-M stride; S is written as a block from row 0.
    if (qk.kind === "i8a8") {
      builder.dispatch({
        key: qk.key,
        pipeline: qk.pipeline,
        layout: qk.layout,
        roles: qk.roles,
        params: face.writeParams(
          attentionQkI8a8Params(block.rows, cols, depth, scale, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: qk.qq },
          { binding: 2, source: qk.kq },
          { binding: 3, source: scores },
          { binding: ATTENTION_QK_Q_SCALE_BINDING, source: qk.qs },
          { binding: ATTENTION_QK_K_SCALE_BINDING, source: qk.ks },
        ],
        workgroups: [
          tiledWorkgroups(cols, i8a8TileN(qk.geometry), limit, `${where} ①QK i8a8`),
          tiledWorkgroups(block.rows, i8a8TileM(qk.geometry), limit, `${where} ①QK i8a8`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK i8a8`),
        ],
      });
    } else {
      builder.dispatch({
        key: qk.key,
        pipeline: qk.pipeline,
        layout: qk.layout,
        roles: qk.roles,
        params: face.writeParams(
          attentionQkParams(block.rows, cols, depth, scale, gqa ? kvRepeat : undefined, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: operands[0] },
          { binding: 2, source: operands[1] },
          { binding: 3, source: scores },
          ...(mask === undefined ? [] : [{ binding: ATTENTION_QK_MASK_BINDING, source: mask }]),
        ],
        workgroups: [
          tiledWorkgroups(cols, gemmTileN(qkGeometry), limit, `${where} ①QK`),
          tiledWorkgroups(block.rows, gemmTileM(qkGeometry), limit, `${where} ①QK`),
          tiledWorkgroups(batch, 1, limit, `${where} ①QK`),
        ],
      });
    }

    // ② Row statistics (block-relative — closed within a row, so cutting rows does not change a
    // row's op sequence).
    builder.dispatch({
      key: statsKey,
      pipeline: stats.pipeline,
      layout: stats.layout,
      roles: stats.roles,
      params: face.writeParams(
        attentionStatsParams(batch * block.rows, cols, statsRegCache),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: rowStats },
      ],
      workgroups: [gridStrideWorkgroups(batch * block.rows, 1, limit), 1, 1],
    });

    // ③ PV gemm — the reduction dimension is N, the output columns are D. The row window `"c"` only
    // writes O to row `row_offset` of the full-M stride; S and row statistics are read
    // block-relative.
    if (pv.kind === "i8a8") {
      builder.dispatch({
        key: pv.key,
        pipeline: pv.pipeline,
        layout: pv.layout,
        roles: pv.roles,
        params: face.writeParams(
          attentionPvI8a8Params(block.rows, depth, cols, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: pv.vq },
          { binding: 3, source: rowStats },
          { binding: 4, source: pvOut },
          { binding: ATTENTION_PV_V_SCALE_BINDING, source: pv.vs },
        ],
        workgroups: [
          tiledWorkgroups(depth, i8a8TileN(pv.geometry), limit, `${where} ③PV i8a8`),
          tiledWorkgroups(block.rows, i8a8TileM(pv.geometry), limit, `${where} ③PV i8a8`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV i8a8`),
        ],
      });
    } else {
      builder.dispatch({
        key: pv.key,
        pipeline: pv.pipeline,
        layout: pv.layout,
        roles: pv.roles,
        params: face.writeParams(
          attentionPvParams(block.rows, depth, cols, gqa ? kvRepeat : undefined, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: operands[2] },
          { binding: 3, source: rowStats },
          { binding: 4, source: pvOut },
        ],
        workgroups: [
          tiledWorkgroups(depth, gemmTileN(pvGeometry), limit, `${where} ③PV`),
          tiledWorkgroups(block.rows, gemmTileM(pvGeometry), limit, `${where} ③PV`),
          tiledWorkgroups(batch, 1, limit, `${where} ③PV`),
        ],
      });
    }

    // MUST: return the temporary buffers at the node boundary (the plan's invariant — missing it
    // fails the closure check, or leaves them out of reallocation and overstates the peak). Reverse
    // allocation order.
    builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }

  // A token-major O: the permute the absorbed output node ran, from the head-major temporary.
  if (headMajorOut !== undefined) {
    await permuteHeads(face, builder, headMajorOut, outs[0], swapHeads(step.outputs[0].shape));
    builder.releaseTemp(headMajorOut);
  }
  // MUST: the quantization temporaries are also returned in reverse allocation order (this closes
  // what `#prepare*` took — allocation and destruction pair up inside the node. ADR 0004).
  if (pv.kind === "i8a8") {
    builder.releaseTemp(pv.vs);
    builder.releaseTemp(pv.vq);
    builder.releaseTemp(pv.vt);
  }
  if (qk.kind === "i8a8") {
    builder.releaseTemp(qk.ks);
    builder.releaseTemp(qk.kq);
    builder.releaseTemp(qk.qs);
    builder.releaseTemp(qk.qq);
  }
  for (const temp of [...headMajorIns].reverse()) {
    if (temp !== undefined) builder.releaseTemp(temp);
  }
};

/**
 * The name of the slot a key of the `states` field points to, and its capacity shape with bindings
 * resolved.
 *
 * The shape's validity (`[B,Hkv,C,D]`, B / Hkv / D matching ins, `window ≤ C`) is settled by the
 * shape layer (src/ops/shapes.ts), so failing to look it up here is a broken runtime-internal
 * invariant (including the path where the derivation phase hit a state node without
 * `stateShapes`).
 */
const stateSlot = (
  step: NodePlan,
  key: string,
  states: StateBuildContext,
  where: string,
): { readonly name: string; readonly shape: readonly number[] } => {
  const name = step.node.states[key];
  if (name === undefined) throw new ExecutionError(`${where}: no '${key}' in the states field`);
  const shape = states.shapes.get(name);
  if (shape === undefined) {
    throw new ExecutionError(
      `${where}: state slot '${name}' has no resolved capacity` +
        " (a node that references state cannot be built for a run without a GenerationContext)",
    );
  }
  return { name, shape };
};

/**
 * States-form attention (ADR 0067 decisions 4 / 6 / 7). **One node = three dispatches per row
 * block**, with a kernel family completely separate from fused attention
 * (src/kernels/state-attention.ts).
 *
 * ①QK (materialises S for the live columns only) → ② row statistics (identity −inf, empty-row
 * guard) → ③PV (P not materialised).
 * K / V come from two places (logical col < `pastLength` from **the slot**, the rest from this
 * step's `ins`), and the scan range is a run-time value, so **only ①QK's dispatch count is computed
 * from the logical length** (②③ depend only on the output-side shape; the live scan happens inside
 * the invocation).
 *
 * MUST: the column capacity `colCap`, the row-block split and the temporaries' byte formulas come
 * only from {@link planStateAttention} (the one function shared with the estimate `estimate.ts`).
 * An even split into the smallest count where one S block = `B·H · block · colCap · 4` bytes fits
 * `maxStorageBufferBindingSize` (with a budget, the smaller of it and the budget — ADR 0124); there
 * is no run-time autotuning
 * (ADR 0022 / ADR 0060). A shape where even one row does not fit fails loudly.
 * MUST: temporaries (S / row statistics) are allocated and returned **per block** (taking them for
 * all blocks at once would defeat the point of staying under the limit).
 */
export const buildStateAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const [q, insK] = step.inputShapes;
  const where = `attention (states) [${q.join(",")}] × [${insK.join(",")}]`;
  const [batch, heads, chunkRows, depth] = q;
  const kvRepeat = heads / insK[1];
  const gqa = kvRepeat > 1;
  const kSlot = stateSlot(step, "k", states, where);
  const vSlot = stateSlot(step, "v", states, where);
  // The shape layer has already established that the k / v slots have the same shape (the capacity
  // can come from either).
  const capacity = kSlot.shape[2];
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const scale = attentionScale(step.node.attrs, where);
  // Numerics variant × states form fails loudly (ADR 0058 decision 3 — the decision is
  // {@link stateAttentionVariantViolation}, shared with the estimate). It is checked **before any
  // temporary buffer is taken** (the same place as the GQA × i8a8 guard).
  const variantViolation = stateAttentionVariantViolation(where, "states", face.state);
  if (variantViolation !== undefined) throw new ExecutionError(variantViolation);
  // Material for the pre-run check (ADR 0066 decision 4 / ADR 0067 decision 4 ④). sliding is a
  // ring, so it is not subject to the capacity check.
  states.chunkRows.add(chunkRows);
  if (!sliding) {
    states.fullCapacities.set(kSlot.name, capacity);
    states.fullCapacities.set(vSlot.name, capacity);
  }

  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const batchHeads = batch * heads;
  // Only the execution plan passes `rowBlockSplit` (test-only `ROW_BLOCK_SPLIT`), which states the
  // row-block count — the estimate has no entry point for it and claims only the default even
  // split.
  const { colCap, fusedStatsPv, blocks } = planStateAttention(
    {
      batchHeads,
      chunkRows,
      capacity,
      window,
      fuseStatsPv: face.state.stateAttentionReduce === "parallel-fused",
    },
    rowBlockLimit(
      face.state.gpu.limits.maxStorageBufferBindingSize,
      face.state.attentionRowBlockMaxBytes,
    ),
    face.state.rowBlockSplit,
  );

  // The reduction forms of ①QK / ③PV share **the same** opt-in option (ADR 0058; the 2026-09-06
  // ruling kept a single knob).
  // ①' / ③' both have bindings and params identical to the reference path; only three things
  // change: key, WGSL and the workgroup computation (the key's `:par` is the census check's
  // marker).
  const parallel = face.state.stateAttentionReduce !== "sequential";
  // Only ①' has an **eligibility condition** on the option: it is chosen only when this plan's `M`
  // (= chunkRows) is 8 or less (decode's M=1 and speculative verify's M ≤ 8).
  // WHY: ①' shortens the latency of one invocation's D-serial loop, and that is the bottleneck only
  // in decode (M=1), where the active invocations are just "live columns × 1 row". In prefill
  // (M=768) ① is already filled with rows × columns, and ①' only drops the row tile width from 4 to
  // 1, stacking 4× the workgroups and barriers. Measured (2026-09-06): decode gained ×1.55–1.72,
  // while prefill's ①QK GPU time grew 1.5–1.9× and prefill as a whole regressed by +30–60%. verify
  // (M=4) was neutral (2026-09-09).
  // The condition **stays inside the measured range** (the same discipline as ADR 0082 decision 4).
  // On the ③' side only plans with `M < 16` take it (`M ≥ 16` takes ③ₜ whatever the option — see
  // below), so one option keeps covering two stages with different ranges.
  // MUST: the decision inputs are only static values fixed at plan time (`chunkRows` comes from the
  // declared shape) — branching on the run-time logical length would make the same plan key point
  // at a different pipeline per run.
  const qkParallel = parallel && stateQkParallelEligible(chunkRows);
  // ①ₜ (K row-tile sharing in the GEMM skeleton — perf-ledger K-13) is **bit-identical to ①**, so
  // it does not depend on the option and is chosen by default for plans with `M ≥ 16` (the
  // eligibility condition and the WHY are in `stateQkTiledEligible`). The two eligibility
  // conditions do not overlap (1 < 16), so this priority order exists only for readability.
  // MUST: **this is the only place** that holds the priority order of the three ①QK paths (the four
  // points key, WGSL, params and workgroups line up here). Scattering it would allow pairs like
  // "key is ①ₜ, dispatch uses ①'s edges", with tiles missing and no exception.
  const qkTiled = !qkParallel && stateQkTiledEligible(chunkRows);
  // ③ₜ (V row-tile sharing in the GEMM skeleton — perf-ledger K-13 step 2) is also **bit-identical
  // to ③**, so it does not depend on the option and is chosen by default for plans with `M ≥ 16`.
  // Unlike ①ₜ, the point here is that **it takes priority over the option**: a prefill plan whose
  // option is `"parallel"` gets ③ₜ (= the reference path's values), not ③'.
  // WHY: ③''s gain is confined to decode, and on the prefill side K-12 measured it within noise,
  // while ③ₜ cuts traffic. Being able to put a bit-identical path as the default keeps the numeric
  // contract simpler.
  // MUST: **this is also the only place** that holds the priority order of the three ③PV paths (as
  // for ①QK — the four points key, WGSL, params and workgroups line up here).
  const pvTiled = statePvTiledEligible(chunkRows);
  const pvParallel = parallel && !pvTiled;
  const qkKey = qkParallel
    ? stateQkParallelKey(sliding, gqa)
    : qkTiled
      ? stateQkTiledKey(sliding, gqa, chunkRows)
      : stateQkKey(sliding, gqa);
  const statsKey = stateStatsKey(sliding);
  const pvKey = fusedStatsPv
    ? stateStatsPvKey(sliding, gqa)
    : pvTiled
      ? statePvTiledKey(sliding, gqa, chunkRows)
      : pvParallel
        ? statePvParallelKey(sliding, gqa)
        : statePvKey(sliding, gqa);
  const qk = await face.state.cache.get(
    qkKey,
    qkParallel
      ? stateQkParallelWgsl(sliding, gqa)
      : qkTiled
        ? stateQkTiledWgsl(sliding, gqa, chunkRows)
        : stateQkWgsl(sliding, gqa),
  );
  const stats = fusedStatsPv
    ? undefined
    : await face.state.cache.get(statsKey, stateStatsWgsl(sliding));
  const pv = await face.state.cache.get(
    pvKey,
    fusedStatsPv
      ? stateStatsPvWgsl(sliding, gqa)
      : pvTiled
        ? statePvTiledWgsl(sliding, gqa, chunkRows)
        : pvParallel
          ? statePvParallelWgsl(sliding, gqa)
          : statePvWgsl(sliding, gqa),
  );

  for (const block of blocks) {
    // **The only source** of the static geometry (eligible for the content-addressed cache — blocks
    // differ only in rowOffset / rowsBlock). params for each stage are built from here.
    const geometry = {
      rowsBlock: block.rows,
      rowOffset: block.offset,
      chunkRows,
      depth,
      kvRepeat,
      window,
      capacity,
      colCap,
      scale,
    };
    // The params with the word order shared by ①③. **A tiled stage reads none of its words**, so a
    // plan with both stages tiled builds none (it would be a dead uniform nobody binds).
    let shared: GPUBuffer | undefined;
    const sharedParams = (): GPUBuffer =>
      (shared ??= face.writeParams(stateAttentionParams(geometry), PARAMS_UNIFORM_USAGE));
    // ①ₜ / ③ₜ put the skeleton's `Dims` (first three words m / n / k) at binding 0, so they rebuild
    // **the same static geometry in a different word order** (③ₜ does not read `neg_inf` / `scale`,
    // so the word count differs too). The word order shared by ①③ does not move — ③ and the
    // estimate read the same table.
    const qkParams = qkTiled
      ? face.writeParams(stateQkTiledParams(geometry), PARAMS_UNIFORM_USAGE)
      : sharedParams();
    const pvParams = pvTiled
      ? face.writeParams(statePvTiledParams(geometry), PARAMS_UNIFORM_USAGE)
      : sharedParams();
    const dispatchGeometry = {
      batchHeads,
      rowsBlock: block.rows,
      rowOffset: block.offset,
      depth,
      window,
    };
    const scores = builder.allocTemp(block.scoreBytes);
    const rowStats = fusedStatsPv ? undefined : builder.allocTemp(block.statsBytes);

    // ①QK — **only the workgroup count is computed from the logical length** (the very mechanism of
    // work ∝ Q × (effective past + Q) — the acceptance condition of ADR 0066 decision 3).
    builder.dispatch({
      key: qkKey,
      pipeline: qk.pipeline,
      layout: qk.layout,
      roles: qk.roles,
      params: qkParams,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: binds[1] },
        { binding: 3, source: { kind: "state", name: kSlot.name } },
        { binding: 4, source: scores },
        { binding: 5, source: { kind: "lengths" } },
      ],
      workgroups: (past, query) =>
        qkParallel
          ? stateQkParallelWorkgroups(dispatchGeometry, past, query, limit, `${where} ①QK`)
          : qkTiled
            ? stateQkTiledWorkgroups(
                dispatchGeometry,
                chunkRows,
                past,
                query,
                limit,
                `${where} ①QK`,
              )
            : stateQkWorkgroups(dispatchGeometry, past, query, limit, `${where} ①QK`),
    });

    if (stats !== undefined && rowStats !== undefined) {
      // ② Row statistics — one row = one workgroup, grid-stride over rows (the live scan is inside
      // the row loop). It covers only the active rows, so the workgroup count is computed from the
      // logical length as for ①.
      builder.dispatch({
        key: statsKey,
        pipeline: stats.pipeline,
        layout: stats.layout,
        roles: stats.roles,
        params: face.writeParams(
          stateStatsParams(batchHeads, block.rows, block.offset, colCap, window),
          PARAMS_UNIFORM_USAGE,
        ),
        bindings: [
          { binding: 1, source: scores },
          { binding: 2, source: rowStats },
          { binding: 3, source: { kind: "lengths" } },
        ],
        workgroups: (_past, query) =>
          stateStatsWorkgroups(dispatchGeometry, query, limit, `${where} ②stats`),
      });
    }
    // ③PV — the output is **all** `rowsBlock` rows from `rowOffset` (pad rows are full-written
    // too).
    builder.dispatch({
      key: pvKey,
      pipeline: pv.pipeline,
      layout: pv.layout,
      roles: pv.roles,
      params: pvParams,
      bindings: [
        { binding: 1, source: scores },
        ...(rowStats === undefined ? [] : [{ binding: 2, source: rowStats }]),
        { binding: 3, source: binds[2] },
        { binding: 4, source: { kind: "state", name: vSlot.name } },
        { binding: 5, source: outs[0] },
        { binding: 6, source: { kind: "lengths" } },
      ],
      workgroups: pvTiled
        ? statePvTiledWorkgroups(dispatchGeometry, chunkRows, limit, `${where} ③PV`)
        : pvParallel
          ? statePvParallelWorkgroups(dispatchGeometry, limit, `${where} ③PV`)
          : statePvWorkgroups(dispatchGeometry, limit, `${where} ③PV`),
    });

    // MUST: return in reverse allocation order (the same order as the plan's replay).
    if (rowStats !== undefined) builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }
};

/**
 * **Readonly states-form attention** (ADR 0096 stage 2 §1.2 / §2.3 — the form where the drafter
 * only reads the lender's KV). It differs from `buildStateAttention` in four points only; the
 * skeleton (①QK → ② row statistics → ③PV, three dispatches × row blocks) is the same:
 *
 * 1. **ins is q alone** (no k/v of this step) — three separate kernels whose bindings each shift up
 *    by one
 * 2. `M = 1` is fixed (a MUST of the shape layer), so there is always one row block and the
 *    reduction variants' eligibility conditions are not consulted (①' / ③' are **always** chosen
 *    whatever the option — `M=1` is exactly their eligibility condition)
 * 3. **The live formula differs** (`[P−min(P,W), P)` — this step's share is not added), so ①②③ all
 *    use readonly-only keys (`:ro`)
 * 4. `states.chunkRows` / `fullCapacities` are **not registered** — the borrower has no
 *    `state_append` at all (§1.1), and a run that writes nothing needs no "does it fit the
 *    capacity" check. Checking the full `P+Q ≤ C` on the borrower too would reject, on the
 *    borrower's run only, a "P that is legitimate for the lender" in the slots other than the
 *    sliding ones (which the lender cycles as a ring)
 *
 * MUST: `colCap` and the row-block split come only from {@link planStateAttention} (the one
 * function shared with the estimate) (the same discipline as the states form). It is given
 * `M = 1`, so the sliding column capacity becomes exactly `W`, matching the readonly live limit
 * `min(P, W)`.
 */
export const buildReadonlyStateAttention = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const q = step.inputShapes[0];
  const where = `attention (readonly) [${q.join(",")}]`;
  // q is `[B,H,1,D]` (M = 1 is guaranteed by the shape layer — ADR 0096 stage 2 §1.2).
  const [batch, heads, , depth] = q;
  const kSlot = stateSlot(step, "k", states, where);
  const vSlot = stateSlot(step, "v", states, where);
  // The shape layer has already established that the k / v slots have the same shape (the capacity
  // can come from either).
  const capacity = kSlot.shape[2];
  const kvRepeat = heads / kSlot.shape[1];
  const gqa = kvRepeat > 1;
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const scale = attentionScale(step.node.attrs, where);
  // Numerics variant × states form fails loudly (the same guard as `buildStateAttention` — ADR 0058
  // decision 3).
  const variantViolation = stateAttentionVariantViolation(where, "readonly", face.state);
  if (variantViolation !== undefined) throw new ExecutionError(variantViolation);

  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const batchHeads = batch * heads;
  const { colCap, blocks } = planStateAttention(
    { batchHeads, chunkRows: 1, capacity, window },
    rowBlockLimit(
      face.state.gpu.limits.maxStorageBufferBindingSize,
      face.state.attentionRowBlockMaxBytes,
    ),
    face.state.rowBlockSplit,
  );
  const qkKey = stateQkParallelReadonlyKey(sliding, gqa);
  const statsKey = stateStatsReadonlyKey(sliding);
  const pvKey = statePvParallelReadonlyKey(sliding, gqa);
  const qk = await face.state.cache.get(qkKey, stateQkParallelReadonlyWgsl(sliding, gqa));
  const stats = await face.state.cache.get(statsKey, stateStatsReadonlyWgsl(sliding));
  const pv = await face.state.cache.get(pvKey, statePvParallelReadonlyWgsl(sliding, gqa));

  for (const block of blocks) {
    const geometry = {
      rowsBlock: block.rows,
      rowOffset: block.offset,
      chunkRows: 1,
      depth,
      kvRepeat,
      window,
      capacity,
      colCap,
      scale,
    };
    // ①③ use the same params word order (the readonly kernels share the states form's `Params`
    // struct).
    const params = face.writeParams(stateAttentionParams(geometry), PARAMS_UNIFORM_USAGE);
    const dispatchGeometry = {
      batchHeads,
      rowsBlock: block.rows,
      rowOffset: block.offset,
      depth,
      window,
    };
    const scores = builder.allocTemp(block.scoreBytes);
    const rowStats = builder.allocTemp(block.statsBytes);

    // ①'QK(ro) — bindings are [params, q, slot_k, s, lengths]. **Only the workgroup count is
    // computed from the logical length** (`P = 0` means live 0 and a column axis of 0 = the
    // dispatch itself is not recorded).
    builder.dispatch({
      key: qkKey,
      pipeline: qk.pipeline,
      layout: qk.layout,
      roles: qk.roles,
      params,
      bindings: [
        { binding: 1, source: binds[0] },
        { binding: 2, source: { kind: "state", name: kSlot.name } },
        { binding: 3, source: scores },
        { binding: 4, source: { kind: "lengths" } },
      ],
      workgroups: (past) =>
        stateQkParallelReadonlyWorkgroups(dispatchGeometry, past, limit, `${where} ①'QK(ro)`),
    });

    // ② Row statistics (ro) — bindings are [params, s, stats, lengths]. The row count is fixed at
    // `Q = 1`, so it is static.
    builder.dispatch({
      key: statsKey,
      pipeline: stats.pipeline,
      layout: stats.layout,
      roles: stats.roles,
      params: face.writeParams(
        stateStatsParams(batchHeads, block.rows, block.offset, colCap, window),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: rowStats },
        { binding: 3, source: { kind: "lengths" } },
      ],
      workgroups: stateStatsReadonlyWorkgroups(dispatchGeometry, limit, `${where} ②stats(ro)`),
    });

    // ③'PV(ro) — bindings are [params, s, stats, slot_v, out, lengths]. The output full-writes the
    // whole row block (empty rows = P 0 are exactly 0).
    builder.dispatch({
      key: pvKey,
      pipeline: pv.pipeline,
      layout: pv.layout,
      roles: pv.roles,
      params,
      bindings: [
        { binding: 1, source: scores },
        { binding: 2, source: rowStats },
        { binding: 3, source: { kind: "state", name: vSlot.name } },
        { binding: 4, source: outs[0] },
        { binding: 5, source: { kind: "lengths" } },
      ],
      workgroups: statePvParallelReadonlyWorkgroups(dispatchGeometry, limit, `${where} ③'PV(ro)`),
    });

    // MUST: return in reverse allocation order (the same order as the plan's replay).
    builder.releaseTemp(rowStats);
    builder.releaseTemp(scores);
  }
};

/**
 * `state_append` (a single-purpose effect op that writes this step's k / v into the slot — ADR 0067
 * decision 5).
 * **One node = one dispatch, zero outputs** (`StepRecipe.outputs` is an empty list).
 *
 * MUST: the workgroup count is computed from `queryLength` (not proportional to the capacity `C` —
 * ADR 0066 decision 3).
 * Only the first `queryLength` rows are written; pad rows are outside the kernel's index space.
 */
export const buildStateAppend = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  builder: StepRecipeBuilder,
  states: StateBuildContext,
): Promise<void> => {
  const x = step.inputShapes[0];
  const where = `state_append [${x.join(",")}]`;
  const slot = stateSlot(step, "slot", states, where);
  const capacity = slot.shape[2];
  const window = stateWindow(step.node.attrs, where) ?? 0;
  const sliding = stateSliding(window);
  const geometry = {
    kvPlanes: x[0] * x[1],
    chunkRows: x[2],
    depth: x[3],
    capacity,
    window,
  };
  states.chunkRows.add(x[2]);
  if (!sliding) states.fullCapacities.set(slot.name, capacity);

  const key = stateAppendKey(sliding);
  const { pipeline, layout, roles } = await face.state.cache.get(key, stateAppendWgsl(sliding));
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(stateAppendParams(geometry), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: { kind: "state", name: slot.name } },
      { binding: 3, source: { kind: "lengths" } },
    ],
    workgroups: (_past, query) => stateAppendWorkgroups(geometry, query, limit, where),
  });
};

/**
 * The **front stage** of the **i8a8 variant** of fused attention's ①QK (opt-in —
 * {@link SessionOptions.attentionCompute}). ① grows from one dispatch to **three** (with n row
 * blocks, 2 + 3n for the whole node):
 *
 * (a) `quantize_rows` (q to per-token i8) → (b) `quantize_rows` (k to per-token i8) →
 * (c) the i8a8 GEMM (integer dot product + dequant — it runs **per block**, so the caller owns it).
 *
 * The quantization kernel is **the same single kernel** as linear's w8a8 (it shares
 * `QUANTIZE_ROWS_KEY` — the reduction axis D is innermost-contiguous for both q and k, so per-token
 * quantization with row = token gives exactly the required form). Diagnostics sum it with linear's
 * activation quantization, so the breakdown is guaranteed by the E2E key-count check.
 *
 * MUST: both (a) and (b) run **outside the row-block loop** (= over all M / all N once). The k side
 * does not depend on the block, so multiplying by the block count is a pure loss; the q side's
 * total does not change, but running it per block would need `quantize_rows` to gather rows over
 * the full-M stride (touching the one kernel shared with linear).
 * Instead (c) reads `qq` / `qs` from the full-M stride through **the A-side row window**.
 * MUST: the temporary buffers (`qq` / `qs` / `kq` / `ks`) are declared here and released **at the
 * end of the node** (by the caller) in reverse allocation order.
 * NOTE: the i32 reduction guard on `D` is checked by the caller's {@link fusedAttentionViolation}
 * before any temporary is taken (the one function shared with the estimate).
 */
const prepareAttentionQkI8a8 = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  scoreStorage: MaterialisedScoreStorage,
  shape: AttentionShape,
  rowWindow: boolean,
  operandLayout: AttentionLayout,
): Promise<AttentionQkStage> => {
  const quantized = await quantizeAttentionQk(face, builder, binds, shape, operandLayout);

  // (c) The integer dot-product GEMM (the half scales go to both q and k on the dequant side —
  // design §2.1). The geometry is chosen **separately** from ③PV (only ③ wins with a single N = D
  // tile — measured). It comes from the per-op field of the profile
  // (src/kernels/geometry-profile.ts). The dispatch is per block, so this only resolves it and
  // returns it to the caller.
  const v4 = attentionQkI8a8UsesVec4(shape.cols);
  const dp4a = face.state.attentionI8a8Dot === "dp4a";
  const geometry = face.state.geometryProfile.i8a8.attentionQk;
  const key = attentionQkI8a8Key(v4, dp4a, scoreStorage, geometry, rowWindow);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    attentionQkI8a8Wgsl(v4, dp4a, scoreStorage, geometry, rowWindow),
  );
  return { kind: "i8a8", key, pipeline, layout, roles, geometry, ...quantized };
};

/** The quantized q / k (four i8 packed per word) and their per-token scales — temporaries the caller releases at the end of the node. */
type AttentionQkQuantized = {
  readonly qq: TempSource;
  readonly qs: TempSource;
  readonly kq: TempSource;
  readonly ks: TempSource;
};

/**
 * (a)(b) of {@link prepareAttentionQkI8a8}: allocates `qq` / `qs` / `kq` / `ks` and dispatches the
 * per-token quantization of q and k. Shared with the online kernel ({@link buildAttentionOnline} —
 * ADR 0132 keeps the a8 quantization), so both forms read the same dispatches.
 *
 * A token-major q / k (`layout` — the node absorbed its permute, perf-ledger F-3) is read with the
 * token-major form of `quantize_rows` (rows from `[B, S, H, D]`, outputs head-major exactly as
 * "permute → quantize_rows" writes them), so the outputs and everything downstream are unchanged.
 */
const quantizeAttentionQk = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  shape: AttentionShape,
  layout: AttentionLayout,
): Promise<AttentionQkQuantized> => {
  const { batch, rows, cols, depth } = shape;
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // The quantized q / k (four i8 packed per word) and the per-token scales. Temporary regions
  // closed inside the node; the q side's scale is **per row** and the k side's **per output
  // column** (two readings of the same per-token quantization).
  // The sizes come from the formula shared with the estimate ({@link attentionI8a8TempBytes}).
  const bytes = attentionI8a8TempBytes(shape).qk;
  const qq = builder.allocTemp(bytes.qq);
  const qs = builder.allocTemp(bytes.qs);
  const kq = builder.allocTemp(bytes.kq);
  const ks = builder.allocTemp(bytes.ks);

  // (a)(b) Per-token quantization of the activations (grid-stride over rows. D is short, so this is
  // the small-D variant that packs several rows into one workgroup — `quantizeRowsGeometry`,
  // perf-ledger P-1)
  const quantizeGeometry = quantizeRowsGeometry(depth);
  // `seq` = the token axis of a token-major operand (`undefined` = head-major rows, the plain form).
  const quantize = async (
    source: BindingSource,
    payload: TempSource,
    scales: TempSource,
    count: number,
    seq: number | undefined,
  ): Promise<void> => {
    const quantizeKey =
      seq === undefined ? quantizeRowsKey(quantizeGeometry) : quantizeRowsBshdKey(quantizeGeometry);
    const {
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      roles: quantizeRoles,
    } = await face.state.cache.get(
      quantizeKey,
      seq === undefined
        ? quantizeRowsWgsl(quantizeGeometry)
        : quantizeRowsBshdWgsl(quantizeGeometry),
    );
    builder.dispatch({
      key: quantizeKey,
      pipeline: quantizePipeline,
      layout: quantizeLayout,
      roles: quantizeRoles,
      params: face.writeParams(
        seq === undefined
          ? quantizeRowsParams(count, depth)
          : quantizeRowsBshdParams(count, depth, seq, layout.heads),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source },
        { binding: 2, source: payload },
        { binding: 3, source: scales },
      ],
      workgroups: [gridStrideWorkgroups(count, quantizeGeometry.rowsPerGroup, limit), 1, 1],
    });
  };
  await quantize(binds[0], qq, qs, batch * rows, layout.q ? rows : undefined);
  await quantize(binds[1], kq, ks, batch * cols, layout.k ? cols : undefined);
  return { qq, qs, kq, ks };
};

/**
 * The **front stage** of the **i8a8 variant** of fused attention's ③PV (opt-in —
 * {@link SessionOptions.attentionCompute}). ③ grows from one dispatch to **three** (with ① also
 * i8a8 and n row blocks, 4 + 3n for the whole node):
 *
 * (a) `strided` (the permute v`[B·H,N,D]` → Vᵀ`[B·H,D,N]`) → (b) `quantize_rows` (quantize Vᵀ
 * with row = `(b,h,d)`) → (c) the i8a8 GEMM (integer dot product + dequant — it runs **per block**,
 * so the caller owns it).
 *
 * The point of this sequence is that **per-column quantization comes without writing a single new
 * kernel** (design §2.3): Vᵀ's "rows" are `(b,h,d)`, so `quantize_rows`'s per-token quantization
 * becomes **V's per-column scale (amax over all of N)** as is, and at the same time it yields the
 * **N-contiguous packing** dp4a requires. MUST: V MUST NOT be quantized without transposing it —
 * the scale would vary along the reduction axis n and break the premise of the `f32(acc)·s` form
 * (s independent of n) (a wrong value with no exception).
 *
 * The P̃ side does not go through the quantization kernel (filling the A tile produces
 * `round(127·exp(S−m))`), so neither dispatches nor temporary buffers increase — ② row statistics
 * stay f32 and do not change by a byte.
 *
 * MUST: (a)(b) run **outside the row-block loop** (Vᵀ is the column side = independent of the
 * block, so putting it in the loop would multiply the permute and the quantization by the block
 * count — a pure performance regression). Only S and row statistics are block-relative, and (c)
 * writes only O's rows to the full-M stride through **the C-side row window**.
 * MUST: the temporary buffers (`vt` / `vq` / `vs`) are declared here and released **at the end of
 * the node** (by the caller) in reverse allocation order.
 * NOTE: the i32 reduction guard on `N` (|acc| ≤ N·127²) is checked by the caller's
 * {@link fusedAttentionViolation} before any temporary is taken (the one function shared with the
 * estimate).
 */
const prepareAttentionPvI8a8 = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  scoreStorage: MaterialisedScoreStorage,
  shape: AttentionShape,
  rowWindow: boolean,
  operandLayout: AttentionLayout,
): Promise<AttentionPvStage> => {
  const quantized = await quantizeAttentionPv(face, builder, binds, shape, operandLayout);

  // (c) The integer dot-product GEMM (P̃ is produced by filling the A tile = stays unmaterialised).
  // The geometry is the per-op field of the profile. The dispatch is per block, so this only
  // resolves it and returns it to the caller.
  const v4 = attentionPvI8a8UsesVec4(shape.depth);
  const dp4a = face.state.attentionI8a8Dot === "dp4a";
  const geometry = face.state.geometryProfile.i8a8.attentionPv;
  const key = attentionPvI8a8Key(v4, dp4a, scoreStorage, geometry, rowWindow);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    attentionPvI8a8Wgsl(v4, dp4a, scoreStorage, geometry, rowWindow),
  );
  return { kind: "i8a8", key, pipeline, layout, roles, geometry, ...quantized };
};

/** Vᵀ (f32, the materialised permute), its quantization and the per-column scales — temporaries the caller releases at the end of the node. */
type AttentionPvQuantized = {
  readonly vt: TempSource;
  readonly vq: TempSource;
  readonly vs: TempSource;
};

/**
 * (a)(b) of {@link prepareAttentionPvI8a8}: allocates `vt` / `vq` / `vs` and dispatches the permute
 * to Vᵀ and its quantization. Shared with the online kernel ({@link buildAttentionOnline} — ADR 0132
 * keeps the a8 quantization), so both forms read the same dispatches.
 *
 * A token-major V (`layout` — the node absorbed its permute, perf-ledger F-8) is `[N, H·D]` per batch,
 * whose transpose `[H·D, N]` is Vᵀ `[B·H, D, N]` as is: (a) is then one tiled transpose per batch
 * instead of the strided copy (both are copies, so Vᵀ is the same bits).
 */
const quantizeAttentionPv = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  shape: AttentionShape,
  layout: AttentionLayout,
): Promise<AttentionPvQuantized> => {
  const { batch, cols, depth } = shape;
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;

  // Vᵀ (f32, the materialised permute or transpose) and its quantization result. All are temporary
  // regions closed inside the node.
  // The sizes come from the formula shared with the estimate ({@link attentionI8a8TempBytes}).
  const bytes = attentionI8a8TempBytes(shape).pv;
  const vt = builder.allocTemp(bytes.vt);
  const vq = builder.allocTemp(bytes.vq);
  const vs = builder.allocTemp(bytes.vs);

  if (layout.v) {
    // (a) A token-major v[B,N,H·D] → Vᵀ[B,H·D,N] = Vᵀ[B·H,D,N]: one 32×32 tile per workgroup, the
    // batch on z. MUST: tiled on every axis (one workgroup = one block — clamping would leave part of
    // Vᵀ unwritten without an exception).
    const {
      pipeline,
      layout: bindLayout,
      roles,
    } = await face.state.cache.get(TRANSPOSE_TILED_KEY, TRANSPOSE_TILED_WGSL);
    const channels = layout.heads * depth;
    const where = `${layout.where} Vᵀ transpose`;
    builder.dispatch({
      key: TRANSPOSE_TILED_KEY,
      pipeline,
      layout: bindLayout,
      roles,
      params: face.writeParams(transposeTiledParams(cols, channels), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: binds[2] },
        { binding: 2, source: vt },
      ],
      workgroups: [
        tiledWorkgroups(channels, TRANSPOSE_TILED_TILE, limit, where),
        tiledWorkgroups(cols, TRANSPOSE_TILED_TILE, limit, where),
        tiledWorkgroups(batch / layout.heads, 1, limit, where),
      ],
    });
  } else {
    await transposeHeadMajorValues(face, builder, binds[2], vt, shape);
  }

  // (b) Quantize Vᵀ (row = (b,h,d), row length N — the per-column scale and the N-contiguous
  // packing come out together)
  const quantizeGeometry = quantizeRowsGeometry(cols);
  const quantizeKey = quantizeRowsKey(quantizeGeometry);
  const {
    pipeline: quantizePipeline,
    layout: quantizeLayout,
    roles: quantizeRoles,
  } = await face.state.cache.get(quantizeKey, quantizeRowsWgsl(quantizeGeometry));
  builder.dispatch({
    key: quantizeKey,
    pipeline: quantizePipeline,
    layout: quantizeLayout,
    roles: quantizeRoles,
    params: face.writeParams(quantizeRowsParams(batch * depth, cols), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: vt },
      { binding: 2, source: vq },
      { binding: 3, source: vs },
    ],
    workgroups: [gridStrideWorkgroups(batch * depth, quantizeGeometry.rowsPerGroup, limit), 1, 1],
  });
  return { vt, vq, vs };
};

/** (a) of {@link quantizeAttentionPv} for a head-major V: the strided permute to Vᵀ. */
const transposeHeadMajorValues = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  source: BindingSource,
  vt: TempSource,
  shape: AttentionShape,
): Promise<void> => {
  const { batch, cols, depth } = shape;
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  // (a) v[B·H,N,D] → Vᵀ[B·H,D,N] (the existing strided-read copy family — the permute itself).
  // MUST: build the strides from the contiguous strides of the **input** `[B·H,N,D]` (building them
  // from the output shape matches only when D == N. The measured shapes have D != N, so it would
  // show, but the unit test is the proper detector).
  const stridedSpec = { dtype: "f32" } as const;
  const permuteKey = stridedKey(stridedSpec);
  const {
    pipeline: permutePipeline,
    layout: permuteLayout,
    roles: permuteRoles,
  } = await face.state.cache.get(permuteKey, stridedWgsl(stridedSpec));
  builder.dispatch({
    key: permuteKey,
    pipeline: permutePipeline,
    layout: permuteLayout,
    roles: permuteRoles,
    params: face.writeParams(
      stridedParams([batch, depth, cols], permuteSrcStrides([batch, cols, depth], [0, 2, 1]), 0),
      PARAMS_STORAGE_USAGE,
    ),
    bindings: [
      { binding: 1, source },
      { binding: 2, source: vt },
    ],
    workgroups: [gridStrideWorkgroups(batch * depth * cols, STRIDED_WORKGROUP_SIZE, limit), 1, 1],
  });
};

/**
 * The online kernel's node (ADR 0132 decision 2 — the `online` route of {@link buildAttention}):
 * the four quantization dispatches of a8 (q and k per token, Vᵀ through the permute) followed by
 * one `attention_online` dispatch per window of query rows ({@link planAttentionOnlineWindows}).
 * No S and no row statistics exist.
 *
 * MUST: the quantization is the a8 path's own dispatches ({@link quantizeAttentionQk} /
 * {@link quantizeAttentionPv}) — the kernel's contract is "the a8 quantization with S never
 * stored", and a second spelling would let the two forms drift apart.
 * MUST: `scale` is the op's half scale exactly as the a8 ①QK receives it (applied to both q and k
 * on the dequant side — ADR 0023).
 * MUST: the temporaries are released at the end of the node in reverse allocation order (ADR 0004 —
 * the same closure as the a8 path).
 * With a token-major O (`run.output` `bshd` — the node absorbed the permute after it, perf-ledger F-3
 * A4) the kernel stores O token-major directly; only the store address changes.
 */
const buildAttentionOnline = async (
  face: RecipeBuildFace,
  builder: StepRecipeBuilder,
  binds: readonly BindingSource[],
  out: BindingSource,
  shape: AttentionShape,
  run: {
    readonly scale: number;
    readonly reduce: AttentionOnlineReduce;
    readonly where: string;
    readonly output: AttentionOnlineOutput;
  },
  layout: AttentionLayout,
): Promise<void> => {
  const { batch, rows, cols, depth } = shape;
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `${run.where} online`;
  const qk = await quantizeAttentionQk(face, builder, binds, shape, layout);
  const pv = await quantizeAttentionPv(face, builder, binds, shape, layout);
  // The integer dot-product variant follows the same decision as the a8 kernels (the per-device
  // canary — SessionState.attentionI8a8Dot); both variants are bit-identical.
  const dp4a = face.state.attentionI8a8Dot === "dp4a";
  const key = attentionOnlineI8a8Key(dp4a, depth, run.reduce, run.output);
  const {
    pipeline,
    layout: bindLayout,
    roles,
  } = await face.state.cache.get(key, attentionOnlineI8a8Wgsl(dp4a, depth, run.reduce, run.output));
  for (const window of planAttentionOnlineWindows(rows, { batch, cols, depth })) {
    builder.dispatch({
      key,
      pipeline,
      layout: bindLayout,
      roles,
      params: face.writeParams(
        attentionOnlineI8a8Params(
          rows,
          cols,
          window.offset,
          run.scale,
          run.output === "bshd" ? layout.heads : undefined,
        ),
        PARAMS_UNIFORM_USAGE,
      ),
      bindings: [
        { binding: 1, source: qk.qq },
        { binding: 2, source: qk.kq },
        { binding: 3, source: pv.vq },
        { binding: 4, source: out },
        { binding: 5, source: qk.qs },
        { binding: 6, source: qk.ks },
        { binding: 7, source: pv.vs },
      ],
      // MUST: tiled on both axes (one workgroup = 64 rows of one head) — clamping would leave rows
      // of O unwritten without an exception (the same rule as ①③).
      workgroups: [
        tiledWorkgroups(window.rows, ATTENTION_ONLINE_ROWS, limit, where),
        tiledWorkgroups(batch, 1, limit, where),
        1,
      ],
    });
  }
  builder.releaseTemp(pv.vs);
  builder.releaseTemp(pv.vq);
  builder.releaseTemp(pv.vt);
  builder.releaseTemp(qk.ks);
  builder.releaseTemp(qk.kq);
  builder.releaseTemp(qk.qs);
  builder.releaseTemp(qk.qq);
};
