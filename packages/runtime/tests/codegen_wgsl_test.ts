import { readDirSync, readTextFile, writeTextFile } from "@karume/test-support/fs";
import { describe, it, test } from "vite-plus/test";
import { stateStatsPvWgsl } from "../src/kernels/state-attention-stats-pv.ts";
import { rmsNormSubgroupWgsl } from "../src/kernels/rms-norm-subgroup.ts";
import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  STATIC_QUANTIZE_PACKED_WGSL,
  STATIC_QUANTIZE_WGSL,
} from "../src/kernels/static-quantize.ts";
import { geluMulStaticQuantizeWgsl } from "../src/kernels/gelu-mul-static-quantize.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../src/codegen/elementwise.ts";
import { CodegenError } from "../src/codegen/errors.ts";
import { GRU_SCAN_MAX_HIDDEN } from "../src/codegen/limits.ts";
import { IS_NAN_BITS_WGSL, TANH_SATURATION } from "../src/codegen/numerics-wgsl.ts";
import {
  AXIS_REDUCE_SPLIT,
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceOutputsPerGroup,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  type ReduceSpec,
  reduceWgsl,
} from "../src/codegen/reduce.ts";
import {
  catOutOffset,
  catOutStrides,
  expandSrcStrides,
  permuteSrcStrides,
  sliceSrcOffset,
  sliceSrcStrides,
  STRIDED_RANK,
  stridedKey,
  stridedParams,
  type StridedSpec,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../src/codegen/strided.ts";
import { type IrDtype, SEMANTIC_DTYPES } from "../src/format/ir.ts";
import {
  ARGMAX_KEY,
  ARGMAX_NEG_INF_BITS,
  ARGMAX_SPLIT_MERGE_WGSL,
  ARGMAX_SPLIT_PARTIAL_WGSL,
  ARGMAX_WGSL,
  ARGMAX_WORKGROUP_SIZE,
  argmaxParams,
  TOPK_ONE_SPLIT_MERGE_KEY,
  topkOneSplitMergeWgsl,
} from "../src/kernels/argmax.ts";
import { bmmKey, bmmParams, bmmRowWindowParams, bmmWgsl } from "../src/kernels/bmm.ts";
import {
  statePvParallelReadonlyWgsl,
  statePvParallelWgsl,
  statePvWgsl,
  stateQkFoldedWgsl,
  stateQkParallelReadonlyWgsl,
  stateQkParallelWgsl,
  stateQkWgsl,
  stateStatsReadonlyWgsl,
  stateStatsWgsl,
} from "../src/kernels/state-attention.ts";
import { stateAppendWgsl } from "../src/kernels/state-append.ts";
import {
  assertTopkK,
  TOPK_CORE_LIMIT_MAX_K,
  TOPK_NEG_INF_BITS,
  TOPK_WORKGROUP_SIZE,
  topkKey,
  topkMaxK,
  topkParams,
  topkWgsl,
  topkWorkgroupStorageBytes,
} from "../src/kernels/topk.ts";
import {
  ATTENTION_QK_MASK_BINDING,
  GEMM_MTILE_SMALL,
  gemmUsesVec4,
  statePvTiledWgsl,
  stateQkTiledWgsl,
} from "../src/kernels/gemm.ts";
import { defaultGemmGeometry, GEMM_TILE } from "../src/kernels/gemm-geometry.ts";
import {
  CONV1D_SCALE_BINDING,
  conv1dIgemmKey,
  conv1dIgemmParams,
  conv1dIgemmWgsl,
  conv1dKey,
  conv1dParams,
  conv1dUsesVec4,
  conv1dWgsl,
} from "../src/kernels/conv1d.ts";
import {
  CONV2D_SCALE_BINDING,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dKey,
  conv2dParams,
  conv2dUsesVec4,
  conv2dWgsl,
} from "../src/kernels/conv2d.ts";
import {
  CONV3D_SCALE_BINDING,
  conv3dIgemmKey,
  conv3dIgemmParams,
  conv3dIgemmWgsl,
  conv3dUsesVec4,
} from "../src/kernels/conv3d.ts";
import {
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../src/kernels/conv-transpose1d.ts";
import { CUMSUM_KEY, CUMSUM_WGSL, cumsumParams } from "../src/kernels/cumsum.ts";
import { FLIP_KEY, FLIP_WGSL, flipParams } from "../src/kernels/flip.ts";
import { PAD_KEY, PAD_WGSL, padParams } from "../src/kernels/pad.ts";
import {
  upsample2xParams,
  UPSAMPLE_2X_KEY,
  UPSAMPLE_2X_WGSL,
  UPSAMPLE_2X_WORKGROUP_SIZE,
} from "../src/kernels/upsample2x.ts";
import {
  UPSAMPLE_BILINEAR2D_KEY,
  UPSAMPLE_BILINEAR2D_WGSL,
  UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
  upsampleBilinear2dParams,
} from "../src/kernels/upsample-bilinear2d.ts";
import {
  GROUP_NORM_APPLY_KEY,
  GROUP_NORM_APPLY_WGSL,
  GROUP_NORM_CHUNK,
  GROUP_NORM_PARTIAL_KEY,
  GROUP_NORM_PARTIAL_WGSL,
  GROUP_NORM_TOTAL_KEY,
  GROUP_NORM_TOTAL_WGSL,
  GROUP_NORM_WORKGROUP_SIZE,
  groupNormApplyParams,
  groupNormLayout,
  groupNormPartialParams,
  groupNormTempBytes,
  groupNormTotalParams,
} from "../src/kernels/group-norm.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_OOB_BITS,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../src/kernels/deform-conv2d.ts";
import {
  GRU_SCAN_WORKGROUP_SIZE,
  type GruScanDirection,
  gruScanKey,
  gruScanParams,
  gruScanWgsl,
} from "../src/kernels/gru-scan.ts";
import { gatedDeltaScanWgsl } from "../src/kernels/gated-delta-scan.ts";
import {
  SILU_WORKGROUP_SIZE,
  siluKey,
  type SiluMulOrder,
  siluParams,
  siluWgsl,
} from "../src/kernels/silu.ts";
import {
  EMBEDDING_OOB_BITS,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../src/kernels/embedding.ts";
import { GATHER_KEY, GATHER_OOB_BITS, GATHER_WGSL, gatherParams } from "../src/kernels/gather.ts";
import {
  LAYER_NORM_AFFINE_WGSL,
  LAYER_NORM_KEY,
  LAYER_NORM_ROW_STATS_WGSL,
  LAYER_NORM_WGSL,
  LAYER_NORM_WORKGROUP_SIZE,
  layerNormParams,
} from "../src/kernels/layer-norm.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_MODULATION_PACK_WGSL,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
  ADALN_NORM_WGSL,
  ADALN_NORM_WORKGROUP_SIZE,
  adalnModulationPackParams,
  adalnNormParams,
  adalnNormWgsl,
} from "../src/kernels/adaln-norm.ts";
import {
  GATED_RESIDUAL_KEY,
  GATED_RESIDUAL_WGSL,
  GATED_RESIDUAL_WORKGROUP_SIZE,
  gatedResidualParams,
} from "../src/kernels/gated-residual.ts";
import {
  RMS_NORM_128_KEY,
  RMS_NORM_128_WGSL,
  RMS_NORM_KEY,
  RMS_NORM_WGSL,
  RMS_NORM_PREFETCH_MAX_DIM,
  RMS_NORM_WORKGROUP_SIZE,
  rmsNormAddKey,
  rmsNormAddWgsl,
  rmsNormInvWgsl,
  rmsNormParams,
  rmsNormPrefetchTrip,
  rmsNormWorkgroupKey,
  rmsNormWorkgroupWgsl,
} from "../src/kernels/rms-norm.ts";
import {
  ROPE_BSHD_KEY,
  ROPE_KEY,
  ROPE_PAIR_BSHD_KEY,
  ROPE_WORKGROUP_SIZE,
  ropeParams,
  ropeWgsl,
} from "../src/kernels/rope.ts";
import { RMS_NORM_ROPE_WGSL } from "../src/kernels/rms-norm-rope.ts";
import { linearKey, linearParams, linearWgsl } from "../src/kernels/linear.ts";
import {
  linearGemvKey,
  linearGemvParallelPackedWgsl,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
  linearGemvRowsKey,
  type LinearGemvRowsVariant,
  linearGemvRowsWgsl,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeWgsl,
  linearGemvSubgroupWgsl,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import {
  DP4A_WGSL_FEATURE,
  dp4aAvailable,
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  LINEAR_W4A8_MAX_GROUP,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../src/kernels/linear-i8a8.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  QUANTIZE_ROWS_WORKGROUP_SIZE,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsBshdWgsl,
  quantizeRowsChannelMajorKey,
  quantizeRowsChannelMajorWgsl,
  quantizeRowsChannelMajorWorkgroups,
  quantizeRowsPaddedGeometry,
  quantizeRowsPaddedWgsl,
  quantizeRowsPrologueWgsl,
  quantizeRowsWgsl,
} from "../src/kernels/quantize-rows.ts";
import { WEIGHT_STORAGES } from "../src/kernels/weight-storage.ts";
import { MASKED_FILL_KEY, MASKED_FILL_WGSL, maskedFillParams } from "../src/kernels/masked-fill.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../src/kernels/matmul.ts";
import {
  SAFE_SOFTMAX_KEY,
  SAFE_SOFTMAX_NEG_INF_BITS,
  SAFE_SOFTMAX_WGSL,
  SOFTMAX_KEY,
  SOFTMAX_WGSL,
  SOFTMAX_WORKGROUP_SIZE,
  softmaxParams,
} from "../src/kernels/softmax.ts";
import {
  ATTENTION_STATS_KEY,
  ATTENTION_STATS_REG_CACHE_MAX,
  ATTENTION_STATS_WGSL,
  ATTENTION_STATS_WORKGROUP_SIZE,
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
} from "../src/kernels/attention.ts";
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
} from "../src/kernels/attention-i8a8.ts";
import { attentionOnlineI8a8Wgsl } from "../src/kernels/attention-online-i8a8.ts";
import { TRANSPOSE_TILED_WGSL } from "../src/kernels/transpose-tiled.ts";
import { CONV3D_I8A8_QUANTIZE_GEOMETRY, conv3dI8a8Wgsl } from "../src/kernels/conv3d-i8a8.ts";
import { DEFAULT_GEOMETRY_PROFILE } from "../src/kernels/geometry-profile.ts";
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { BINARY_OPS, OP_CONTRACTS, REDUCE_OPS, UNARY_OPS, WHERE_OP } from "../src/ops.ts";

const fixtureUrl = (name: string): URL => new URL(`./fixtures/wgsl/${name}`, import.meta.url);
const fixture = (name: string): Promise<string> => readTextFile(fixtureUrl(name));
// `KARUME_WGSL_SNAPSHOT=write` rewrites every snapshot from the current generators before the
// comparison (the same shape as `KARUME_REFERENCE=write` for the sha rows). It exists for commits
// that change only the comments inside the WGSL templates (ADR 0130 step 2-9d): the diff of the
// fixtures then has to be comment lines only, which the commit's review checks.
const WRITE_SNAPSHOTS = process.env.KARUME_WGSL_SNAPSHOT === "write";

// The 5 fused kernels have weight-storage variants (ADR 0018). The existing invariants apply to the
// **w=f32 variant**, so the old names are bound to that variant (the f16 variants are covered by
// their own tests). The 3 GEMM ops additionally have 2 variants by the shape-derived v4 flag
// (src/kernels/gemm.ts) — the old names are bound to the **scalar variant**, and each test lists
// the v4 side explicitly.
const LINEAR_WGSL = linearWgsl("f32", false);
const MATMUL_WGSL = matmulWgsl(false);
const BMM_WGSL = bmmWgsl(false);
/**
 * The Cartesian product used to iterate mechanically over every generator input of the GEMM
 * skeleton (7 ops × weight storage × v4). Fused attention's ①③ (ADR 0023) and the implicit GEMM of
 * conv1d / conv2d (ADR 0024) share the same skeleton, so the checks of the inner-product loop, the
 * tile edges and the keys apply to them automatically as well.
 */
const GEMM_VARIANTS: readonly (readonly [string, string, string])[] = [false, true].flatMap(
  (v4) => [
    [`matmul${v4 ? " v4" : ""}`, matmulKey(v4), matmulWgsl(v4)] as const,
    [`bmm${v4 ? " v4" : ""}`, bmmKey(v4), bmmWgsl(v4)] as const,
    [`attention_qk${v4 ? " v4" : ""}`, attentionQkKey(v4), attentionQkWgsl(v4)] as const,
    [`attention_pv${v4 ? " v4" : ""}`, attentionPvKey(v4), attentionPvWgsl(v4)] as const,
    ...WEIGHT_STORAGES.flatMap((weight) => [
      [
        `linear ${weight}${v4 ? " v4" : ""}`,
        linearKey(weight, v4),
        linearWgsl(weight, v4),
      ] as const,
      [
        `conv1d igemm ${weight}${v4 ? " v4" : ""}`,
        conv1dIgemmKey(weight, v4),
        conv1dIgemmWgsl(weight, v4),
      ] as const,
      [
        `conv2d igemm ${weight}${v4 ? " v4" : ""}`,
        conv2dIgemmKey(weight, v4),
        conv2dIgemmWgsl(weight, v4),
      ] as const,
      [
        `conv3d igemm ${weight}${v4 ? " v4" : ""}`,
        conv3dIgemmKey(weight, v4),
        conv3dIgemmWgsl(weight, v4),
      ] as const,
    ]),
    // i4 is **limited to linear / embedding** (ADR 0069 decision 5 + the embedding addendum) — it
    // is not in the conv product (the binding path of the group scale exists only in linear's fill
    // and the embedding kernel; passing it to conv gives invalid WGSL). The group is represented by
    // the default 32 (addendum 1) — another group only changes the g part of the key and the baked
    // shift as a pair.
    [
      `linear i4 g32${v4 ? " v4" : ""}`,
      linearKey("i4", v4, "f32", undefined, 32),
      linearWgsl("i4", v4, "f32", undefined, 32),
    ] as const,
  ],
);
const EMBEDDING_WGSL = embeddingWgsl("f32");
const CONV1D_WGSL = conv1dWgsl("f32");
const CONV1D_KEY = conv1dKey("f32");
const CONV2D_WGSL = conv2dWgsl("f32");
const CONV2D_KEY = conv2dKey("f32");
const CONV_TRANSPOSE1D_WGSL = convTranspose1dWgsl("f32");

/**
 * Every (op, dtype) pair the contract table allows (the elementwise space codegen can generate).
 * The dtype is that of the **value slot** — where's condition slot (fixed to bool) is not a
 * generator input, so it is taken from the accepted set of the slot that holds "the value side",
 * not slot 0.
 */
const elementwiseDtypes = (): readonly (readonly [string, IrDtype])[] =>
  [...UNARY_OPS, ...BINARY_OPS, WHERE_OP].flatMap((op) => {
    const contract = OP_CONTRACTS.get(op);
    if (contract === undefined) return [];
    const slots = contract.slotDtypes;
    const accept = slots.kind === "uniform" ? slots.accept : slots.slots[slots.slots.length - 1];
    return accept.map((dtype): [string, IrDtype] => [op, dtype]);
  });

test("generated WGSL matches the snapshots byte for byte (pins codegen determinism)", async () => {
  const cases: readonly (readonly [string, string])[] = [
    ["static_quantize.wgsl", STATIC_QUANTIZE_WGSL],
    ["static_quantize_packed.wgsl", STATIC_QUANTIZE_PACKED_WGSL],
    // gelu_tanh → mul → SRQ (perf-ledger K-48): both operand orders — the product text differs.
    ["gelu_mul_static_quantize_gelu_other.wgsl", geluMulStaticQuantizeWgsl("gelu-other")],
    ["gelu_mul_static_quantize_other_gelu.wgsl", geluMulStaticQuantizeWgsl("other-gelu")],
    ["elementwise_relu_r1.wgsl", elementwiseWgsl({ op: "relu", rank: 1, dtype: "f32" })],
    ["elementwise_sigmoid_r1.wgsl", elementwiseWgsl({ op: "sigmoid", rank: 1, dtype: "f32" })],
    ["elementwise_gelu_r2.wgsl", elementwiseWgsl({ op: "gelu", rank: 2, dtype: "f32" })],
    // The saturating-clamp version of tanh (TANH_STABLE_WGSL in src/codegen/elementwise.ts). The
    // condition is **placing the 2 ops as a pair**: the clamp only changes "the argument band", so
    // a state where only one of them went through (only inside gelu_tanh / only unary tanh) slips
    // through a value comparison. The same enumeration also checks that elementwise_gelu_r2.wgsl
    // above (the erf form) does not move by a single byte.
    ["elementwise_tanh_r1.wgsl", elementwiseWgsl({ op: "tanh", rank: 1, dtype: "f32" })],
    ["elementwise_gelu_tanh_r1.wgsl", elementwiseWgsl({ op: "gelu_tanh", rank: 1, dtype: "f32" })],
    ["elementwise_add_r3.wgsl", elementwiseWgsl({ op: "add", rank: 3, dtype: "f32" })],
    ["elementwise_mul_i32_r2.wgsl", elementwiseWgsl({ op: "mul", rank: 2, dtype: "i32" })],
    [
      "elementwise_bitwise_not_bool_r1.wgsl",
      elementwiseWgsl({ op: "bitwise_not", rank: 1, dtype: "bool" }),
    ],
    [
      "elementwise_cast_f32_i32_r1.wgsl",
      elementwiseWgsl({ op: "cast", rank: 1, dtype: "f32", to: "i32" }),
    ],
    [
      "elementwise_cast_i32_bool_r2.wgsl",
      elementwiseWgsl({ op: "cast", rank: 2, dtype: "i32", to: "bool" }),
    ],
    ["elementwise_log1p_r1.wgsl", elementwiseWgsl({ op: "log1p", rank: 1, dtype: "f32" })],
    ["elementwise_clamp_r2.wgsl", elementwiseWgsl({ op: "clamp", rank: 2, dtype: "f32" })],
    ["elementwise_clamp_min_r1.wgsl", elementwiseWgsl({ op: "clamp_min", rank: 1, dtype: "f32" })],
    [
      "elementwise_leaky_relu_r1.wgsl",
      elementwiseWgsl({ op: "leaky_relu", rank: 1, dtype: "f32" }),
    ],
    ["elementwise_ge_scalar_r2.wgsl", elementwiseWgsl({ op: "ge_scalar", rank: 2, dtype: "f32" })],
    ["elementwise_ge_r3.wgsl", elementwiseWgsl({ op: "ge", rank: 3, dtype: "f32" })],
    [
      "elementwise_bitwise_and_bool_r1.wgsl",
      elementwiseWgsl({ op: "bitwise_and", rank: 1, dtype: "bool" }),
    ],
    ["elementwise_where_r3.wgsl", elementwiseWgsl({ op: WHERE_OP, rank: 3, dtype: "f32" })],
    ["reduce_sum.wgsl", reduceWgsl({ op: "sum", dtype: "f32" })],
    ["reduce_sum_bool.wgsl", reduceWgsl({ op: "sum", dtype: "bool" })],
    ["reduce_amax.wgsl", reduceWgsl({ op: "amax", dtype: "f32" })],
    // amax / amin differ only in the fold function (both NaN-propagation forms are frozen)
    ["reduce_amin.wgsl", reduceWgsl({ op: "amin", dtype: "f32" })],
    // The axis reduce (any dimension but the last), a separate variant at the production split.
    // It reproduces the row reduce's order with per-thread bit-reversed carry stacks over aligned
    // subtrees plus pairwise root combines; the bytes are frozen because a moved order breaks the
    // bit identity. The dtype variant (bool → i32) and the NaN-propagating form (amax) sit beside it.
    ["reduce_axis_sum.wgsl", axisReduceWgsl({ op: "sum", dtype: "f32" })],
    ["reduce_axis_sum_bool.wgsl", axisReduceWgsl({ op: "sum", dtype: "bool" })],
    ["reduce_axis_amax.wgsl", axisReduceWgsl({ op: "amax", dtype: "f32" })],
    // amin is placed as a pair too (the same reason as the row reduce's reduce_amax / reduce_amin —
    // there are two NaN-propagation forms, nan_max / nan_min, and freezing only one lets a swapped
    // injection slip through).
    ["reduce_axis_amin.wgsl", axisReduceWgsl({ op: "amin", dtype: "f32" })],
    ["cumsum.wgsl", CUMSUM_WGSL],
    // argmax (ADR 0068 decision 2). The condition is **placing it as a pair with the 3 reduces**:
    // it has the same skeleton as the row reduce, so the worst accident is "adding a new family
    // moved the bytes of an existing reduce" (the existing 3 + the 3 axis variants are its
    // detector).
    ["argmax.wgsl", ARGMAX_WGSL],
    // The two-phase form (long rows — MTP step 4-B ③). partial / merge are placed as a pair (if
    // only one moves, "the per-segment max element moved but merge is the old form" passes under
    // the same key, and the bit-identity claim breaks silently).
    ["argmax_split_partial.wgsl", ARGMAX_SPLIT_PARTIAL_WGSL],
    ["argmax_split_merge.wgsl", ARGMAX_SPLIT_MERGE_WGSL],
    ["topk_one_split_merge.wgsl", topkOneSplitMergeWgsl()],
    // topk (ADR 0068 decision 3). The condition is **placing k=1 and the general form as a pair** —
    // with k=1 the block's tail (weakest) and head are the same word and the insertion loop never
    // runs, a degenerate form, so pinning only the general form would miss a baked `k-1` that is
    // off by one (k=1 is also the counterpart of the comparison check that returns the same answer
    // as argmax).
    ["topk_k1.wgsl", topkWgsl(1)],
    ["topk_k4.wgsl", topkWgsl(4)],
    // The 3 GEMM ops have 2 variants (scalar / vec4) by the shape-derived v4 flag. The condition is
    // **listing both sides**: pinning only one would miss the fallback for ragged shapes silently
    // breaking.
    ["matmul.wgsl", MATMUL_WGSL],
    ["matmul_v4.wgsl", matmulWgsl(true)],
    ["bmm.wgsl", BMM_WGSL],
    ["bmm_v4.wgsl", bmmWgsl(true)],
    // bmm's **row-window variants** (row-block execution of decomposed attention —
    // `rowBlockAttention` in src/runtime/fusion.ts). The condition is placing the 4 of A side / C
    // side × v4 **as a pair with the 2 plain bmm**: the worst accident is adding the row window
    // moving the plain bmm's bytes (a machine with one row block runs the plain bmm as is, so if it
    // moves, the existing WAV / PNG checks all move).
    ["bmm_rowwin_a.wgsl", bmmWgsl(false, undefined, "a")],
    ["bmm_rowwin_a_v4.wgsl", bmmWgsl(true, undefined, "a")],
    ["bmm_rowwin_c.wgsl", bmmWgsl(false, undefined, "c")],
    ["bmm_rowwin_c_v4.wgsl", bmmWgsl(true, undefined, "c")],
    // The 3 fused attention kernels (ADR 0023). ①③ are variants of the GEMM skeleton, so they are
    // placed as a pair with v4.
    ["attention_qk.wgsl", attentionQkWgsl(false)],
    ["attention_qk_v4.wgsl", attentionQkWgsl(true)],
    ["attention_pv.wgsl", attentionPvWgsl(false)],
    ["attention_pv_v4.wgsl", attentionPvWgsl(true)],
    ["attention_stats.wgsl", ATTENTION_STATS_WGSL],
    // ①QK's **additive-mask variants** (ADR 0023 revision). The main purpose of this enumeration is
    // less adding these 2 than checking that **attention_qk*.wgsl / attention_pv*.wgsl /
    // attention_stats*.wgsl above do not move by a single byte** (②③ never look at the mask).
    ["attention_qk_mask.wgsl", attentionQkWgsl(false, "f32", "f32", true)],
    ["attention_qk_mask_v4.wgsl", attentionQkWgsl(true, "f32", "f32", true)],
    // **GQA variants** (broadcast by exact division — ADR 0067 decisions 1 / 2). The condition is
    // placing ①③ **as 4 in pairs with v4**; the main purpose of the enumeration is less adding
    // these 4 than **attention_qk*.wgsl / attention_pv*.wgsl above not moving by a single byte** (=
    // the
    // MUST of decision 2 that `r = 1` is byte-identical).
    ["attention_qk_gqa.wgsl", attentionQkWgsl(false, "f32", "f32", false, true)],
    ["attention_qk_gqa_v4.wgsl", attentionQkWgsl(true, "f32", "f32", false, true)],
    ["attention_pv_gqa.wgsl", attentionPvWgsl(false, "f32", "f32", true)],
    ["attention_pv_gqa_v4.wgsl", attentionPvWgsl(true, "f32", "f32", true)],
    // **Row-window variants** (block execution over query rows — `buildAttention` in
    // src/runtime/recipe-builders/attention.ts). ①QK's A side (q) and ③PV's C side (O) get the
    // full-M stride + a row offset. The condition is **placing the masked forms as a pair**,
    // because the mask's **row** index is the only thing in the row window that leaves the base
    // arithmetic (S is block-relative, the mask is the full-M tensor). The main purpose of the
    // enumeration is less adding these 6 than **attention_qk*.wgsl / attention_pv*.wgsl above not
    // moving by a single byte** (= n=1 sets up no row window, so it is fully identical to before
    // the split).
    ["attention_qk_rwa.wgsl", attentionQkWgsl(false, "f32", "f32", false, false, true)],
    ["attention_qk_rwa_v4.wgsl", attentionQkWgsl(true, "f32", "f32", false, false, true)],
    ["attention_qk_mask_rwa.wgsl", attentionQkWgsl(false, "f32", "f32", true, false, true)],
    ["attention_qk_mask_rwa_v4.wgsl", attentionQkWgsl(true, "f32", "f32", true, false, true)],
    ["attention_pv_rwc.wgsl", attentionPvWgsl(false, "f32", "f32", false, true)],
    ["attention_pv_rwc_v4.wgsl", attentionPvWgsl(true, "f32", "f32", false, true)],
    ["gather.wgsl", GATHER_WGSL],
    ["strided_copy_f32.wgsl", stridedWgsl({ dtype: "f32" })],
    ["strided_copy_i32.wgsl", stridedWgsl({ dtype: "i32" })],
    ["strided_copy_bool.wgsl", stridedWgsl({ dtype: "bool" })],
    // The write family is cat-only, and its contract is f32-only, so only f32 can be generated (ADR
    // 0014)
    ["strided_write_f32.wgsl", stridedWriteWgsl({ dtype: "f32" })],
    ["pad.wgsl", PAD_WGSL],
    ["upsample2x.wgsl", UPSAMPLE_2X_WGSL],
    // Bilinear resample (layer 1 op). It is a separate kernel and key from the layer 3 fusion rule
    // `upsample2x` (bitwise nearest replication), and the condition is **listing both** — when only
    // one is fixed, this enumeration checks mechanically that the other's bytes did not move.
    ["upsample_bilinear2d.wgsl", UPSAMPLE_BILINEAR2D_WGSL],
    // group_norm (Core ATen layer — ADR 0138 decision 7): the three kernels of its five dispatches.
    ["group_norm_partial.wgsl", GROUP_NORM_PARTIAL_WGSL],
    ["group_norm_total.wgsl", GROUP_NORM_TOTAL_WGSL],
    ["group_norm_apply.wgsl", GROUP_NORM_APPLY_WGSL],
    // DCNv2 (layer 1' op — ADR 0055). The condition is **listing it as a pair with conv2d's direct
    // kernel (conv2d_*.wgsl below)**: the degenerate bit match (offset 0, mask 1 → conv2d) depends
    // on the two having exactly the same reduction order — the worst accident is only one side's
    // bytes moving.
    ["deform_conv2d.wgsl", DEFORM_CONV2D_WGSL],
    // The GRU hidden-side scan (layer 2 op — ADR 0056). The condition is **listing both
    // directions**: only this pair shows that nothing but the one line of scan order differs. That
    // the rounding barrier (a workgroup memory round trip) remains in the output is also frozen
    // here, together with the invariant test below.
    ["gru_scan_forward.wgsl", gruScanWgsl("forward")],
    ["gru_scan_reverse.wgsl", gruScanWgsl("reverse")],
    // The gated delta rule scan (ADR 0135): one kernel, one geometry. The snapshot pins the measured
    // form (a transcription of the spike's kernel — the module doc lists the textual differences).
    ["gated_delta_scan.wgsl", gatedDeltaScanWgsl()],
    ["silu_x_sigmoid.wgsl", siluWgsl("x-sigmoid")],
    ["silu_sigmoid_x.wgsl", siluWgsl("sigmoid-x")],
    ["flip.wgsl", FLIP_WGSL],
    ["rope.wgsl", ropeWgsl("bhsd")],
    ["rope_bshd.wgsl", ropeWgsl("bshd")],
    // The pair rotation (Wan's DiT) is a third variant of the same template: the two half-split
    // snapshots above stay byte-identical next to it.
    ["rope_pair_bshd.wgsl", ropeWgsl("bshd", "pair")],
    // rms_norm → pair RoPE in one kernel (perf-ledger F-3); it shares the rms_norm and RoPE text, so
    // the plain forms around it must not move.
    ["rms_norm_rope.wgsl", RMS_NORM_ROPE_WGSL],
    ["linear.wgsl", LINEAR_WGSL],
    ["linear_v4.wgsl", linearWgsl("f32", true)],
    // One representative of the skinny-M geometry (M ≤ 64 → M16N16 — the sweep-confirmed value in
    // src/kernels/gemm-geometry.ts). The condition is **placing it as a pair with the default side
    // (the 2 above)**: the worst accident is adding a bucket moving the default's bytes. The 3 ops
    // share the skeleton, so linear alone is enough as the representative.
    ["linear_m16_v4.wgsl", linearWgsl("f32", true, "f32", 4)],
    ["layer_norm.wgsl", LAYER_NORM_WGSL],
    // adaLN fusion (the 4th of ADR 0040). The condition is **placing it as a pair with the plain
    // layer_norm**: they share the row-statistics and affine text, so the worst accident is only
    // one side's bytes moving.
    ["adaln_norm.wgsl", ADALN_NORM_WGSL],
    // The per-token variant (Wan2.2) comes from the same template as adaln_norm.wgsl; the pair
    // pins both byte sequences, and the pack dispatch that feeds it sits next to them.
    ["adaln_norm_token.wgsl", adalnNormWgsl("token")],
    ["adaln_modulation_pack.wgsl", ADALN_MODULATION_PACK_WGSL],
    ["gated_residual.wgsl", GATED_RESIDUAL_WGSL],
    ["rms_norm.wgsl", RMS_NORM_WGSL],
    ["rms_norm_128.wgsl", RMS_NORM_128_WGSL],
    // The prefetch form (v2, perf-ledger K-99) at its edge trip count and at Gemma 4 E2B's width;
    // v1 above stays the text of every other width. The v1 add forms it replaced are kept in
    // fixtures/wgsl-baseline/ for the GPU identity test (gpu_rms_norm_prefetch_test.ts).
    ["rms_norm_trip1.wgsl", rmsNormWorkgroupWgsl(256)],
    ["rms_norm_trip6.wgsl", rmsNormWorkgroupWgsl(1536)],
    // RMS→add fusion (ADR 0099). The condition is **placing it as a pair with the plain rms_norm**;
    // the bytes freeze that the rounding barrier (an integer XOR with a uniform 0) remains in the
    // output for both orders (if it were simplified to a constant-expression bitcast round trip,
    // only the fused side's rounding would change).
    ["rms_norm_add_nr.wgsl", rmsNormAddWgsl("norm-residual", 1536)],
    ["rms_norm_add_rn.wgsl", rmsNormAddWgsl("residual-norm", 1536)],
    ["rms_norm_subgroup32.wgsl", rmsNormSubgroupWgsl()],
    ["rms_norm_add_subgroup32_nr.wgsl", rmsNormSubgroupWgsl("norm-residual")],
    ["rms_norm_add_subgroup32_rn.wgsl", rmsNormSubgroupWgsl("residual-norm")],
    ["softmax.wgsl", SOFTMAX_WGSL],
    // The safe_softmax variant (ADR 0044). The condition is **placing it as a pair with the plain
    // softmax**: both come from the same generator function (if the reduction order of ②③ moves by
    // a single word, bit identity with the decomposed path breaks).
    ["safe_softmax.wgsl", SAFE_SOFTMAX_WGSL],
    ["embedding.wgsl", EMBEDDING_WGSL],
    ["masked_fill.wgsl", MASKED_FILL_WGSL],
    ["conv1d.wgsl", CONV1D_WGSL],
    ["conv2d.wgsl", CONV2D_WGSL],
    ["conv_transpose1d.wgsl", CONV_TRANSPOSE1D_WGSL],
    // The f16 / i8 weight-storage variants (ADR 0018 / 0019). Placed as a pair with the 5 above,
    // the same enumeration checks mechanically that **the f32 side's bytes have not moved** (the
    // worst accident is adding a variant moving an existing one).
    ["linear_wf16.wgsl", linearWgsl("f16", false)],
    ["linear_wf16_v4.wgsl", linearWgsl("f16", true)],
    // The bf16 weight-storage variants (ADR 0131) — placed as a pair with the f16 variants (only
    // the unpack changes to a bitcast).
    ["linear_wbf16.wgsl", linearWgsl("bf16", false)],
    ["linear_wbf16_v4.wgsl", linearWgsl("bf16", true)],
    ["embedding_wf16.wgsl", embeddingWgsl("f16")],
    ["embedding_wbf16.wgsl", embeddingWgsl("bf16")],
    ["conv1d_wf16.wgsl", conv1dWgsl("f16")],
    ["conv1d_wbf16.wgsl", conv1dWgsl("bf16")],
    ["conv2d_wf16.wgsl", conv2dWgsl("f16")],
    ["conv2d_wbf16.wgsl", conv2dWgsl("bf16")],
    ["conv_transpose1d_wf16.wgsl", convTranspose1dWgsl("f16")],
    ["conv_transpose1d_wbf16.wgsl", convTranspose1dWgsl("bf16")],
    ["linear_wi8.wgsl", linearWgsl("i8", false)],
    ["linear_wi8_v4.wgsl", linearWgsl("i8", true)],
    // i4 is limited to linear / embedding (ADR 0069) — the variant with the shift of group 32 (the
    // default) baked in. The embedding version is **placed as a pair with linear** (the scale is
    // fetched by a separate implementation [per quad in the fill / per element in the kernel
    // body], so an output where only one side moved still builds normally).
    ["linear_wi2.wgsl", linearWgsl("i2", false)],
    ["linear_wi2_v4.wgsl", linearWgsl("i2", true)],
    ["linear_gemv_wi2.wgsl", linearGemvWgsl("i2")],
    ["linear_gemv_parallel_wi2_l2.wgsl", linearGemvParallelWgsl("i2", undefined, 2)],
    ["linear_gemv_static_quantize_wi2_l2.wgsl", linearGemvStaticQuantizeWgsl("i2", undefined, 2)],
    ["linear_gemv_static_quantize_g512_l4.wgsl", linearGemvStaticQuantizeWgsl("i4", 512, 4)],
    ["linear_gemv_static_quantize_wi8_l32.wgsl", linearGemvStaticQuantizeWgsl("i8", undefined, 32)],
    // The packed int8 activation variants (ADR 0105) — 6 that differ only in the activation binding
    // and its restore.
    ["linear_gemv_parallel_packed_wi2_l2.wgsl", linearGemvParallelPackedWgsl("i2", undefined, 2)],
    ["linear_gemv_parallel_packed_g512_l4.wgsl", linearGemvParallelPackedWgsl("i4", 512, 4)],
    ["linear_gemv_parallel_packed_wi8_l16.wgsl", linearGemvParallelPackedWgsl("i8", undefined, 16)],
    // The staged-activation variant (K-90): the same K loop in rounds, x read from workgroup memory.
    ["linear_gemv_parallel_staged_g32_l32.wgsl", linearGemvParallelStagedWgsl("i4", 32, 32)],
    ["linear_gemv_parallel_staged_wi8_l32.wgsl", linearGemvParallelStagedWgsl("i8", undefined, 32)],
    ["linear_gemv_parallel_staged_wi2_l4.wgsl", linearGemvParallelStagedWgsl("i2", undefined, 4)],
    [
      "linear_gemv_static_quantize_packed_wi2_l2.wgsl",
      linearGemvStaticQuantizePackedWgsl("i2", undefined, 2),
    ],
    [
      "linear_gemv_static_quantize_packed_g512_l4.wgsl",
      linearGemvStaticQuantizePackedWgsl("i4", 512, 4),
    ],
    [
      "linear_gemv_static_quantize_packed_wi8_l32.wgsl",
      linearGemvStaticQuantizePackedWgsl("i8", undefined, 32),
    ],
    ["linear_gemv_parallel_g512_l4.wgsl", linearGemvParallelWgsl("i4", 512, 4)],
    ["linear_gemv_parallel_wi8_l16.wgsl", linearGemvParallelWgsl("i8", undefined, 16)],
    ["linear_gemv_subgroup_wi2_l2.wgsl", linearGemvSubgroupWgsl("i2", undefined, 2)],
    ["linear_gemv_subgroup_g512_l4.wgsl", linearGemvSubgroupWgsl("i4", 512, 4)],
    ["linear_gemv_subgroup_wi8_l16.wgsl", linearGemvSubgroupWgsl("i8", undefined, 16)],
    [
      "linear_gemv_r4_wi2.wgsl",
      linearGemvRowsWgsl("i2", undefined, { cols: 32, unroll: 4, rows: 4 }),
    ],
    ["embedding_wi2.wgsl", embeddingWgsl("i2")],
    ["linear_wi4.wgsl", linearWgsl("i4", false, "f32", undefined, 32)],
    ["linear_wi4_v4.wgsl", linearWgsl("i4", true, "f32", undefined, 32)],
    // linear's **GEMV family** (M=1 × i4 / i8 — ADR 0082). It is a separate kernel that does not
    // share the skeleton, so the main purpose of the enumeration is less adding these 3 than **the
    // linear_wi4* / linear_wi8* above not moving by a single byte** (= the default path for M>1 is
    // fully identical to before the branch).
    // MUST: place the 2 groups **as a pair**. With g32 the scale index `(unit · 32) >> shift`
    // reduces to an identity, so only the g64 side shows that the baked shift actually takes
    // effect.
    // MUST: place the i8 side **as a pair** too (perf-ledger K-16). i4 and i8 are separate branches
    // in the element count per word (32 / 16) and in how the scale is fetched (a group per word /
    // one channel scale outside the reduction), so an output where only one side moved still
    // builds normally.
    ["linear_gemv_g32.wgsl", linearGemvWgsl("i4", 32)],
    ["linear_gemv_g64.wgsl", linearGemvWgsl("i4", 64)],
    ["linear_gemv_wi8.wgsl", linearGemvWgsl("i8")],
    ["linear_gemv_wf16.wgsl", linearGemvWgsl("f16")],
    ["linear_gemv_wbf16.wgsl", linearGemvWgsl("bf16")],
    ["linear_gemv_f32.wgsl", linearGemvWgsl("f32")],
    // The family's **row-block variants** (M ≥ 2 — ADR 0082 addendum 5 / perf-ledger K-21). The
    // condition is placing them **as a pair with the 3 M=1 ones above**: only this pair shows that
    // the row-block change does not move the decode (M=1) output by a single byte.
    // MUST: place `r1` — rows=1 is "the M=1 kernel laid M times along y" but is **separate text**
    // (it adds the row head `xr0` to index x, and the write-back gets a `row0 < dims.m` guard). If
    // it degenerated to the same text, a dispatch with M ≥ 2 would write only one row, a silent
    // wrong value.
    // MUST: place the 2 groups as a pair (same wording as the M=1 side — with g32 the scale index
    // reduces to an identity, so only the g64 side shows that the baked shift takes effect).
    // MUST: place the i8 side too (a separate branch of 16 elements per word × a channel scale — a
    // green on one side is no evidence for the other).
    ["linear_gemv_r1_g32.wgsl", linearGemvRowsWgsl("i4", 32, { cols: 32, unroll: 4, rows: 1 })],
    ["linear_gemv_r4_g32.wgsl", linearGemvRowsWgsl("i4", 32, { cols: 32, unroll: 4, rows: 4 })],
    ["linear_gemv_r4_g64.wgsl", linearGemvRowsWgsl("i4", 64, { cols: 32, unroll: 4, rows: 4 })],
    [
      "linear_gemv_r4_wi8.wgsl",
      linearGemvRowsWgsl("i8", undefined, {
        cols: 32,
        unroll: 4,
        rows: 4,
      }),
    ],
    ["embedding_wi4.wgsl", embeddingWgsl("i4", 32)],
    ["embedding_wi8.wgsl", embeddingWgsl("i8")],
    ["conv1d_wi8.wgsl", conv1dWgsl("i8")],
    ["conv2d_wi8.wgsl", conv2dWgsl("i8")],
    ["conv_transpose1d_wi8.wgsl", convTranspose1dWgsl("i8")],
    // conv2d's implicit GEMM (ADR 0024) has 6 variants: storage 3 × v4. The condition is listing
    // **both** it and the direct kernel (conv2d*.wgsl above); with only one, one side of the
    // dispatch choice could move silently without anyone noticing.
    ["conv2d_igemm.wgsl", conv2dIgemmWgsl("f32", false)],
    ["conv2d_igemm_v4.wgsl", conv2dIgemmWgsl("f32", true)],
    ["conv2d_igemm_wf16.wgsl", conv2dIgemmWgsl("f16", false)],
    ["conv2d_igemm_wf16_v4.wgsl", conv2dIgemmWgsl("f16", true)],
    ["conv2d_igemm_wbf16.wgsl", conv2dIgemmWgsl("bf16", false)],
    ["conv2d_igemm_wbf16_v4.wgsl", conv2dIgemmWgsl("bf16", true)],
    ["conv2d_igemm_wi8.wgsl", conv2dIgemmWgsl("i8", false)],
    ["conv2d_igemm_wi8_v4.wgsl", conv2dIgemmWgsl("i8", true)],
    // The 32-row m-tile variants (next to ADR 0024). The condition is **listing them as a pair with
    // the 64-row versions**: the worst accident is adding the variant moving the 64-row side's
    // bytes (the 6 above are the detector).
    ["conv2d_igemm_m32.wgsl", conv2dIgemmWgsl("f32", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_v4.wgsl", conv2dIgemmWgsl("f32", true, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wf16.wgsl", conv2dIgemmWgsl("f16", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wf16_v4.wgsl", conv2dIgemmWgsl("f16", true, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wbf16.wgsl", conv2dIgemmWgsl("bf16", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wbf16_v4.wgsl", conv2dIgemmWgsl("bf16", true, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wi8.wgsl", conv2dIgemmWgsl("i8", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wi8_v4.wgsl", conv2dIgemmWgsl("i8", true, GEMM_MTILE_SMALL)],
    // conv1d's implicit GEMM (the 1D version of ADR 0024) has 12 variants at the same granularity.
    // The condition is **listing them as a pair with both conv1d*.wgsl above (the direct kernel)
    // and conv2d_igemm*.wgsl**: the worst accident is generalizing the fragments to 1D moving the
    // 2D side's bytes.
    ["conv1d_igemm.wgsl", conv1dIgemmWgsl("f32", false)],
    ["conv1d_igemm_v4.wgsl", conv1dIgemmWgsl("f32", true)],
    ["conv1d_igemm_wf16.wgsl", conv1dIgemmWgsl("f16", false)],
    ["conv1d_igemm_wf16_v4.wgsl", conv1dIgemmWgsl("f16", true)],
    ["conv1d_igemm_wbf16.wgsl", conv1dIgemmWgsl("bf16", false)],
    ["conv1d_igemm_wbf16_v4.wgsl", conv1dIgemmWgsl("bf16", true)],
    ["conv1d_igemm_wi8.wgsl", conv1dIgemmWgsl("i8", false)],
    ["conv1d_igemm_wi8_v4.wgsl", conv1dIgemmWgsl("i8", true)],
    ["conv1d_igemm_m32.wgsl", conv1dIgemmWgsl("f32", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_v4.wgsl", conv1dIgemmWgsl("f32", true, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wf16.wgsl", conv1dIgemmWgsl("f16", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wf16_v4.wgsl", conv1dIgemmWgsl("f16", true, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wbf16.wgsl", conv1dIgemmWgsl("bf16", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wbf16_v4.wgsl", conv1dIgemmWgsl("bf16", true, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wi8.wgsl", conv1dIgemmWgsl("i8", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wi8_v4.wgsl", conv1dIgemmWgsl("i8", true, GEMM_MTILE_SMALL)],
    // conv1d's i4 variant (ADR 0069 decision 5's conv1d addendum — the shift of group 32 baked in).
    // The condition is **placing it as a pair with linear_wi4***: i4's group scale is a separate
    // implementation, on the B side in linear and on the A side in conv1d (`fillBLinear` /
    // `fillAConv`) — an output where only one side moved still builds normally.
    // MUST: there is no i4 counterpart on the conv2d side (there is no execution path at all; the
    // generator entry rejects it).
    ["conv1d_igemm_wi4.wgsl", conv1dIgemmWgsl("i4", false, undefined, 32)],
    ["conv1d_igemm_wi4_v4.wgsl", conv1dIgemmWgsl("i4", true, undefined, 32)],
    ["conv1d_igemm_m32_wi4.wgsl", conv1dIgemmWgsl("i4", false, GEMM_MTILE_SMALL, 32)],
    ["conv1d_igemm_m32_wi4_v4.wgsl", conv1dIgemmWgsl("i4", true, GEMM_MTILE_SMALL, 32)],
    // conv3d's implicit GEMM (ADR 0118 decision 1 — unbatched, the 3D version of ADR 0024) has 12
    // variants at the same granularity as conv1d / conv2d (storage 3 × v4 × m tile 64 / 32). The
    // condition is **listing them as a pair with conv1d_igemm* / conv2d_igemm***: the worst
    // accident is adding the 3D fragments (DIMS_EXTRA / xcol / kDecode / the B fill) to the shared
    // skeleton moving the bytes of the 2 existing families (the 24 above are its detector).
    ["conv3d_igemm.wgsl", conv3dIgemmWgsl("f32", false)],
    ["conv3d_igemm_v4.wgsl", conv3dIgemmWgsl("f32", true)],
    ["conv3d_igemm_wf16.wgsl", conv3dIgemmWgsl("f16", false)],
    ["conv3d_igemm_wf16_v4.wgsl", conv3dIgemmWgsl("f16", true)],
    ["conv3d_igemm_wbf16.wgsl", conv3dIgemmWgsl("bf16", false)],
    ["conv3d_igemm_wbf16_v4.wgsl", conv3dIgemmWgsl("bf16", true)],
    ["conv3d_igemm_wi8.wgsl", conv3dIgemmWgsl("i8", false)],
    ["conv3d_igemm_wi8_v4.wgsl", conv3dIgemmWgsl("i8", true)],
    ["conv3d_igemm_m32.wgsl", conv3dIgemmWgsl("f32", false, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_v4.wgsl", conv3dIgemmWgsl("f32", true, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wf16.wgsl", conv3dIgemmWgsl("f16", false, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wf16_v4.wgsl", conv3dIgemmWgsl("f16", true, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wbf16.wgsl", conv3dIgemmWgsl("bf16", false, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wbf16_v4.wgsl", conv3dIgemmWgsl("bf16", true, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wi8.wgsl", conv3dIgemmWgsl("i8", false, GEMM_MTILE_SMALL)],
    ["conv3d_igemm_m32_wi8_v4.wgsl", conv3dIgemmWgsl("i8", true, GEMM_MTILE_SMALL)],
    // w8a8 (i8 activations + integer dot product). The condition is placing **both the dp4a version
    // and the emulated version**: the claim "same numbers, only the speed differs" presupposes that
    // the two outputs exist separately.
    ["quantize_rows.wgsl", QUANTIZE_ROWS_WGSL],
    // The small-D variant (perf-ledger P-1 — 8 rows × 32 lanes used by the QK prologue at D=128).
    // The condition is **placing it as a pair with the existing form above**: the worst accident is
    // adding the variant moving the existing form's bytes.
    ["quantize_rows_r8w32.wgsl", quantizeRowsWgsl(quantizeRowsGeometry(128))],
    // The geometry takes 9 values determined by dim (lanesPerRow is a power of 2 in 1..256), and
    // the callers' dim is a **runtime shape** (linear's k / attention's head width / the runtime
    // key count). The two degenerate ends and the shipped head width 256 are frozen: dim 4 is the
    // form where the tree reduce runs no level at all (lanesPerRow 1, rowsPerGroup 256), and dim
    // 512 is r2w128, near the other end.
    ["quantize_rows_r256w1.wgsl", quantizeRowsWgsl(quantizeRowsGeometry(4))],
    ["quantize_rows_r4w64.wgsl", quantizeRowsWgsl(quantizeRowsGeometry(256))],
    ["quantize_rows_r2w128.wgsl", quantizeRowsWgsl(quantizeRowsGeometry(512))],
    // The prologue form (perf-ledger F-4): the Wan2.2 down projection's row (K 14,336 = 14 vec4 per
    // lane) and the bound's edge (16 per lane). The plain forms above must not move with it.
    ["quantize_rows_gelu_tanh_reg14.wgsl", quantizeRowsPrologueWgsl("gelu_tanh", 14336)],
    ["quantize_rows_gelu_tanh_reg16.wgsl", quantizeRowsPrologueWgsl("gelu_tanh", 16384)],
    // The erf form at anima's DiT FFN down projection (K 8,192 = 8 vec4 per lane).
    ["quantize_rows_gelu_reg8.wgsl", quantizeRowsPrologueWgsl("gelu", 8192)],
    // The token-major read of the attention's q / k (perf-ledger F-3): the D = 128 geometry the
    // attention uses, and the one-row-per-workgroup end. The plain forms above must not move.
    ["quantize_rows_r8w32_bshd.wgsl", quantizeRowsBshdWgsl(quantizeRowsGeometry(128))],
    ["quantize_rows_r1w256_bshd.wgsl", quantizeRowsBshdWgsl(quantizeRowsGeometry(1024))],
    // The padded form (ADR 0132 addendum 2026-10-27 — the online attention's Vᵀ at N % 4 != 0):
    // the MOSS-Image 1024² negative branch's N = 4,106 (one row per workgroup) and a short row
    // (N = 125, 8 rows × 32 lanes). The plain forms above must not move.
    ["quantize_rows_r1w256_pad.wgsl", quantizeRowsPaddedWgsl(quantizeRowsPaddedGeometry(4106))],
    ["quantize_rows_r8w32_pad.wgsl", quantizeRowsPaddedWgsl(quantizeRowsPaddedGeometry(125))],
    // The channel-major read of conv3d w8a8's activation (perf-ledger V-6) at the production
    // geometry. The plain forms above must not move.
    [
      "quantize_rows_cm_p16l16b16.wgsl",
      quantizeRowsChannelMajorWgsl(CONV3D_I8A8_QUANTIZE_GEOMETRY),
    ],
    // The tiled transpose that builds Vᵀ from a token-major V (perf-ledger F-8).
    ["transpose_tiled.wgsl", TRANSPOSE_TILED_WGSL],
    ["linear_i8a8.wgsl", linearI8a8Wgsl(false, true)],
    ["linear_i8a8_v4.wgsl", linearI8a8Wgsl(true, true)],
    ["linear_i8a8_emu.wgsl", linearI8a8Wgsl(false, false)],
    ["linear_i8a8_emu_v4.wgsl", linearI8a8Wgsl(true, false)],
    // w4a8 (resident i4 weights × per-token i8 activations, perf-ledger Q-8). The condition is
    // **placing them as a pair with the 4 linear_i8a8* above**: the worst accident is adding these
    // 4 moving the i8 side's bytes (the i8 path not changing by a single byte is the very
    // implementation contract of w4a8). The K loop becomes 2 levels, "groups outside × tiles
    // inside", so this also freezes the shared fragments (the inner-product loop / the A-side fill)
    // going in with their indentation one level shallower.
    ["linear_w4a8_g32.wgsl", linearI8a8Wgsl(false, true, undefined, "i4", 32)],
    ["linear_w4a8_g32_v4.wgsl", linearI8a8Wgsl(true, true, undefined, "i4", 32)],
    ["linear_w4a8_g32_emu.wgsl", linearI8a8Wgsl(false, false, undefined, "i4", 32)],
    ["linear_w4a8_g32_emu_v4.wgsl", linearI8a8Wgsl(true, false, undefined, "i4", 32)],
    // The i8a8 variants of fused attention ①QK / ③PV (iterations 1 / 2 of the design's §7). The
    // condition is **placing them as a pair with the f32 / :c16 variants above**: the worst
    // accident is adding the variants moving the default path's bytes.
    ["attention_qk_i8a8.wgsl", attentionQkI8a8Wgsl(false, true)],
    ["attention_qk_i8a8_v4.wgsl", attentionQkI8a8Wgsl(true, true)],
    ["attention_qk_i8a8_emu.wgsl", attentionQkI8a8Wgsl(false, false)],
    ["attention_qk_i8a8_emu_v4.wgsl", attentionQkI8a8Wgsl(true, false)],
    ["attention_pv_i8a8.wgsl", attentionPvI8a8Wgsl(false, true)],
    ["attention_pv_i8a8_v4.wgsl", attentionPvI8a8Wgsl(true, true)],
    ["attention_pv_i8a8_emu.wgsl", attentionPvI8a8Wgsl(false, false)],
    ["attention_pv_i8a8_emu_v4.wgsl", attentionPvI8a8Wgsl(true, false)],
    // The i8a8 **row-window variants** (one each, with the v4 side of the measured shape as the
    // representative — they differ from the scalar side only in the unit of base, and that choice
    // is covered by the 4 plain ones above and the structure test below). ①QK gets the full-M
    // stride on 2 bindings, the quantized q and its row scale; ③PV only on 1, O.
    ["attention_qk_i8a8_rwa_v4.wgsl", attentionQkI8a8Wgsl(true, true, "f32", undefined, true)],
    ["attention_pv_i8a8_rwc_v4.wgsl", attentionPvI8a8Wgsl(true, true, "f32", undefined, true)],
    // The f16 **compute** variants (ADR 0028). The condition is placing them as a pair with the f32
    // variants above: the worst accident is adding the variants moving the f32 side's bytes (the
    // default path's whole numeric contract rests on it).
    // The vocabulary is `c16`, separate from `wf16` for f16 storage — `linear_wf16_c16*` is "f16
    // weight storage × f16 compute".
    ["attention_qk_c16.wgsl", attentionQkWgsl(false, "f16")],
    ["attention_qk_c16_v4.wgsl", attentionQkWgsl(true, "f16")],
    ["attention_pv_c16.wgsl", attentionPvWgsl(false, "f16")],
    ["attention_pv_c16_v4.wgsl", attentionPvWgsl(true, "f16")],
    ["attention_stats_c16.wgsl", attentionStatsWgsl("f16")],
    ["linear_c16.wgsl", linearWgsl("f32", false, "f16")],
    ["linear_c16_v4.wgsl", linearWgsl("f32", true, "f16")],
    ["linear_wf16_c16.wgsl", linearWgsl("f16", false, "f16")],
    ["linear_wf16_c16_v4.wgsl", linearWgsl("f16", true, "f16")],
    ["linear_wbf16_c16.wgsl", linearWgsl("bf16", false, "f16")],
    ["linear_wbf16_c16_v4.wgsl", linearWgsl("bf16", true, "f16")],
    // The **f16 storage** variants of S (option γ iteration 1 —
    // `SessionOptions.attentionScoreStorage`). The vocabulary is `s16`, separate from weight
    // storage `wf16` / compute `c16`, and **only the v4 side exists** (a partial write on the
    // scalar path would be an RMW on the same u32 word, so the generator entry rejects it — the
    // fail-loudly test below).
    // MUST: place them as a pair with the f32 / c16 / i8a8 variants above (the worst accident is
    // adding variants moving the default path's bytes, and this enumeration is what watches it).
    ["attention_qk_s16_v4.wgsl", attentionQkWgsl(true, "f32", "f16")],
    ["attention_pv_s16_v4.wgsl", attentionPvWgsl(true, "f32", "f16")],
    ["attention_stats_s16.wgsl", attentionStatsWgsl("f32", "f16")],
    ["attention_qk_i8a8_s16_v4.wgsl", attentionQkI8a8Wgsl(true, true, "f16")],
    ["attention_qk_i8a8_emu_s16_v4.wgsl", attentionQkI8a8Wgsl(true, false, "f16")],
    ["attention_pv_i8a8_s16_v4.wgsl", attentionPvI8a8Wgsl(true, true, "f16")],
    ["attention_pv_i8a8_emu_s16_v4.wgsl", attentionPvI8a8Wgsl(true, false, "f16")],
    // The online int8 attention (ADR 0132 — `attentionScoreStorage: "none"`): one dispatch per
    // row window, S never stored. Both reduction forms are frozen (they MUST stay bit-identical,
    // so a change to one of them shows up only here), the emulated dot product and both depths.
    ["attention_online_i8a8_d128_sg32.wgsl", attentionOnlineI8a8Wgsl(true, 128, "sg32")],
    // The token-major O store (perf-ledger F-3 A4); the head-major forms around it must not move.
    [
      "attention_online_i8a8_d128_sg32_obshd.wgsl",
      attentionOnlineI8a8Wgsl(true, 128, "sg32", "bshd"),
    ],
    ["attention_online_i8a8_d128_wg.wgsl", attentionOnlineI8a8Wgsl(true, 128, "wg")],
    ["attention_online_i8a8_d128_emu_sg32.wgsl", attentionOnlineI8a8Wgsl(false, 128, "sg32")],
    ["attention_online_i8a8_d64_sg32.wgsl", attentionOnlineI8a8Wgsl(true, 64, "sg32")],
    // The padded key axis (ADR 0132 addendum 2026-10-27 — N % 4 != 0) with the token-major O
    // (both key fragments in one key). The whole-word forms above must not move.
    [
      "attention_online_i8a8_d128_sg32_obshd_kpad.wgsl",
      attentionOnlineI8a8Wgsl(true, 128, "sg32", "bshd", "padded"),
    ],
    // The padded key axis on the other axes of the key: head-major O, the wg form and D = 64.
    [
      "attention_online_i8a8_d64_wg_kpad.wgsl",
      attentionOnlineI8a8Wgsl(true, 64, "wg", "bhsd", "padded"),
    ],
    // The w8a8 variants of conv3d (ADR 0133 — `convCompute: "a8"`): of the 2 output variants × 2
    // dot-product variants, 3 are frozen (scalar × emulated only differs as the combination, so it
    // is left out).
    // The default profile's conv3d geometry (the geometry has no default argument)
    ["conv3d_i8a8_v4.wgsl", conv3dI8a8Wgsl(true, true, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d)],
    ["conv3d_i8a8.wgsl", conv3dI8a8Wgsl(false, true, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d)],
    ["conv3d_i8a8_emu_v4.wgsl", conv3dI8a8Wgsl(true, false, DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d)],
    // ② the **regcache variant** of the row statistics (reads S only once and keeps it in
    // registers). The generation depends on dim, so `epc` is added to the key axes — the measured
    // self shape (dim 4096 → 16 elements per thread) is frozen.
    // MUST: place it as a pair with the two-read version above (the values are bit-identical, so
    // the difference shows only on the output side).
    ["attention_stats_rc16.wgsl", attentionStatsWgsl("f32", "f32", 16)],
    // The 3 kernels + `state_append` of the **states-form attention** (ADR 0067 decisions 4–7). It
    // is a **separate family** that shares not a single byte with the existing fused attention (two
    // sources of K / V, the identity is −inf, the scan range is a runtime value), so the main
    // purpose of this enumeration is less adding these 12 than **every snapshot of attention_* /
    // bmm_* / the GEMM skeleton above not moving by a single byte**.
    // MUST: all 4 families place full / sliding as a pair (only sliding has the lower-bound
    // predicate and the ring mapping, so pinning one side alone lets both "full grew a lower bound"
    // and "the lower bound vanished from sliding" slip through the snapshots). ①③ also place the
    // GQA variants as a pair.
    ["attention_state_qk.wgsl", stateQkWgsl(false, false)],
    ["attention_state_qk_gqa.wgsl", stateQkWgsl(false, true)],
    ["attention_state_qk_sliding.wgsl", stateQkWgsl(true, false)],
    ["attention_state_qk_sliding_gqa.wgsl", stateQkWgsl(true, true)],
    // The ①' D-parallel reduction variant (opt-in — **the same option** as ③',
    // `stateAttentionReduce: "parallel"`, perf-ledger K-14).
    ["attention_state_qk_par.wgsl", stateQkParallelWgsl(false, false)],
    ["attention_state_qk_par_gqa.wgsl", stateQkParallelWgsl(false, true)],
    ["attention_state_qk_par_sliding.wgsl", stateQkParallelWgsl(true, false)],
    ["attention_state_qk_par_sliding_gqa.wgsl", stateQkParallelWgsl(true, true)],
    // The folded ①' (perf-ledger K-55 — the F = r query heads of one kv plane per workgroup, full GQA
    // only). Both generated F are frozen (F = 8 is selected; F = 4 is kept for perf-ledger K-95): F
    // changes the accumulators, the shared storage and the writers, so one F passing says nothing
    // about the other.
    ["attention_state_qk_par_fold4_gqa.wgsl", stateQkFoldedWgsl(4)],
    ["attention_state_qk_par_fold8_gqa.wgsl", stateQkFoldedWgsl(8)],
    // The ①ₜ K-row tile-sharing variant (**the default path**, perf-ledger K-13). It is the only
    // kernel that plugs states fragments into the GEMM skeleton, so it is the detector of "adding
    // fragments to the skeleton does not move the generated bytes of the existing 6 ops by a single
    // byte" (attention_* / bmm_* / linear_* / conv*_ above are the reference).
    // MUST: place **2 geometry buckets** (the smallest M16N16 and the default M128N128) — the
    // geometry should change only the work assignment, so this pins that the store's guard
    // structure and the inner-product loop do not change between them.
    ["attention_state_qk_tiled.wgsl", stateQkTiledWgsl(false, false, 16)],
    ["attention_state_qk_tiled_gqa.wgsl", stateQkTiledWgsl(false, true, 16)],
    ["attention_state_qk_tiled_sliding.wgsl", stateQkTiledWgsl(true, false, 16)],
    ["attention_state_qk_tiled_sliding_gqa.wgsl", stateQkTiledWgsl(true, true, 16)],
    ["attention_state_qk_tiled_m768.wgsl", stateQkTiledWgsl(false, false, 768)],
    ["attention_state_qk_tiled_m768_sliding_gqa.wgsl", stateQkTiledWgsl(true, true, 768)],
    // The fused stats + PV (ADR 0102) and ③' below are v2 (the unrolled KV loop — perf-ledger K-91).
    // The v1 WGSL they must stay word-for-word identical to is kept in fixtures/wgsl-baseline/ for
    // the GPU identity test (gpu_state_attention_fold_test.ts); it is not a snapshot of a generator,
    // so it is outside this enumeration and its directory check.
    ["attention_state_stats_pv.wgsl", stateStatsPvWgsl(false, false)],
    ["attention_state_stats_pv_gqa.wgsl", stateStatsPvWgsl(false, true)],
    ["attention_state_stats_pv_sliding.wgsl", stateStatsPvWgsl(true, false)],
    ["attention_state_stats_pv_sliding_gqa.wgsl", stateStatsPvWgsl(true, true)],
    ["attention_state_stats.wgsl", stateStatsWgsl(false)],
    ["attention_state_stats_sliding.wgsl", stateStatsWgsl(true)],
    ["attention_state_pv.wgsl", statePvWgsl(false, false)],
    ["attention_state_pv_gqa.wgsl", statePvWgsl(false, true)],
    ["attention_state_pv_sliding.wgsl", statePvWgsl(true, false)],
    ["attention_state_pv_sliding_gqa.wgsl", statePvWgsl(true, true)],
    // The ③' KV-parallel reduction variant (opt-in — `stateAttentionReduce: "parallel"`,
    // perf-ledger K-12).
    ["attention_state_pv_par.wgsl", statePvParallelWgsl(false, false)],
    ["attention_state_pv_par_gqa.wgsl", statePvParallelWgsl(false, true)],
    ["attention_state_pv_par_sliding.wgsl", statePvParallelWgsl(true, false)],
    ["attention_state_pv_par_sliding_gqa.wgsl", statePvParallelWgsl(true, true)],
    // The ③ₜ V-row tile-sharing variant (**the default path**, perf-ledger K-13 step 2). It is the
    // second kernel on the same skeleton as ①ₜ, so it also serves as the detector of
    // "parameterizing the skeleton's fragments (the A bound, the K loop bound) does not move the
    // generated bytes of the existing 6 ops and ①ₜ by a single byte".
    // MUST: place **2 geometry buckets** (the smallest M16N16 and the default M128N128) — same
    // wording as ①ₜ: this pins that the store's guard structure and the inner-product loop do not
    // change between them.
    ["attention_state_pv_tiled.wgsl", statePvTiledWgsl(false, false, 16)],
    ["attention_state_pv_tiled_gqa.wgsl", statePvTiledWgsl(false, true, 16)],
    ["attention_state_pv_tiled_sliding.wgsl", statePvTiledWgsl(true, false, 16)],
    ["attention_state_pv_tiled_sliding_gqa.wgsl", statePvTiledWgsl(true, true, 16)],
    ["attention_state_pv_tiled_m768.wgsl", statePvTiledWgsl(false, false, 768)],
    ["attention_state_pv_tiled_m768_sliding_gqa.wgsl", statePvTiledWgsl(true, true, 768)],
    // The 3 readonly variants (ADR 0096 step 2 §1.2 — the form where a drafter only reads the
    // lender's KV).
    // MUST: choose the 3 so that they cover the **2 live branches** (sliding / full), **the window
    // predicate** (only ①' has it) and **the GQA kv plane** — placing the full product per variant
    // would bury the difference of the readonly-specific fragments (`column_base` / `live_columns`
    // / `in_window`) under the volume of snapshots. ①' is sliding + GQA (the only combination that
    // shows both the window predicate and the kv plane), ② is full (the `live = past` side), ③' is
    // sliding without GQA (the remaining 2 branches).
    ["attention_state_qk_par_ro_sliding_gqa.wgsl", stateQkParallelReadonlyWgsl(true, true)],
    ["attention_state_stats_ro.wgsl", stateStatsReadonlyWgsl(false)],
    ["attention_state_pv_par_ro_sliding.wgsl", statePvParallelReadonlyWgsl(true, false)],
    ["state_append.wgsl", stateAppendWgsl(false)],
    ["state_append_sliding.wgsl", stateAppendWgsl(true)],
  ];
  for (const [name, generated] of cases) {
    if (WRITE_SNAPSHOTS) await writeTextFile(fixtureUrl(name), generated);
    assertEquals(generated, await fixture(name), name);
  }
  // MUST: match the snapshot enumeration against the directory's contents. A .wgsl left out of the
  // enumeration becomes a dead pin "present but never compared"; conversely a name not in the
  // directory fails at readTextFile above (= a state with only one side added is blocked in both
  // directions).
  assertEquals(
    [...readDirSync(new URL("./fixtures/wgsl/", import.meta.url))]
      .map((entry) => entry.name)
      .sort(),
    cases.map(([name]) => name).sort(),
    "the contents of fixtures/wgsl disagree with the snapshot enumeration",
  );
});

test("the same generator input always yields identical WGSL (every op × dtype × rank)", () => {
  for (const [op, dtype] of elementwiseDtypes()) {
    for (const rank of [1, 2, 3, 4]) {
      const spec = { op, rank, dtype } as ElementwiseSpec;
      assertEquals(elementwiseWgsl(spec), elementwiseWgsl(spec), `${op}:${dtype}:r${rank}`);
    }
  }
  for (const from of SEMANTIC_DTYPES) {
    for (const to of SEMANTIC_DTYPES) {
      const spec: ElementwiseSpec = { op: "cast", rank: 2, dtype: from, to };
      assertEquals(elementwiseWgsl(spec), elementwiseWgsl(spec), `cast:${from}>${to}`);
    }
  }
  for (const op of REDUCE_OPS) {
    for (const dtype of OP_CONTRACTS.get(op)?.dtypes ?? []) {
      assertEquals(reduceWgsl({ op, dtype }), reduceWgsl({ op, dtype }), `${op}:${dtype}`);
      assertEquals(
        axisReduceWgsl({ op, dtype }),
        axisReduceWgsl({ op, dtype }),
        `${op}:${dtype}:axis`,
      );
    }
  }
  for (const dtype of SEMANTIC_DTYPES) {
    assertEquals(stridedWgsl({ dtype }), stridedWgsl({ dtype }), `strided:${dtype}`);
    assertEquals(
      stridedWriteWgsl({ dtype }),
      stridedWriteWgsl({ dtype }),
      `strided_write:${dtype}`,
    );
  }
  // The GEMV family (ADR 0082) is a separate kernel that does not share the skeleton; its generator
  // inputs are only storage and group length (it takes no v4 — the output is written as scalars).
  // That baking the shift holds no state is pinned over 3 groups.
  for (const groupSize of [32, 64, 128]) {
    assertEquals(
      linearGemvWgsl("i4", groupSize),
      linearGemvWgsl("i4", groupSize),
      `linear gemv:g=${groupSize}`,
    );
  }
  assertEquals(linearGemvWgsl("i8"), linearGemvWgsl("i8"), "linear gemv:i8");
  assertEquals(linearGemvWgsl("f16"), linearGemvWgsl("f16"), "linear gemv:f16");
  assertEquals(linearGemvWgsl("f32"), linearGemvWgsl("f32"), "linear gemv:f32");
  // The row-block variants only add `rows` to the generator inputs (rows comes from a pure function
  // of (storage, m, n), and the generator bakes the number it receives as is). The unroll count
  // changes with rows, so the absence of state is pinned across heights.
  for (const rows of [1, 2, 4, 16]) {
    const i4: LinearGemvRowsVariant = { cols: 32, unroll: 4, rows };
    assertEquals(
      linearGemvRowsWgsl("i4", 32, i4),
      linearGemvRowsWgsl("i4", 32, i4),
      `linear gemv rows:i4:r${rows}`,
    );
    assertEquals(
      linearGemvRowsWgsl("i8", undefined, i4),
      linearGemvRowsWgsl("i8", undefined, i4),
      `linear gemv rows:i8:r${rows}`,
    );
  }
  // MUST: put a limit on the generated text of the largest selectable form (i4 8 rows / i8 16 rows
  // = the ceiling of 256 elements per word). The text size is directly the shader's parse and
  // validation cost (superlinear in text size in naga); this is the detector that stops a
  // regression where 8 rows at 73 KB ≈ 55 ms silently goes back to 100 KB / 200 ms (perf-ledger
  // K-21, research 2026-09-07-gemv-rows-k21 §6).
  for (const [label, text] of [
    ["i4 r8", linearGemvRowsWgsl("i4", 32, { cols: 32, unroll: 4, rows: 8 })],
    ["i8 r16", linearGemvRowsWgsl("i8", undefined, { cols: 32, unroll: 4, rows: 16 })],
  ] as const) {
    assert(
      text.length <= 80_000,
      `linear gemv rows ${label}: the generated text of ${text.length} characters exceeds the limit of 80,000`,
    );
  }
  // MUST: the rows=1 row block is **separate text** from the M=1 variant (if they were the same,
  // the y tiling would be gone).
  assertNotEquals(
    linearGemvRowsWgsl("i4", 32, { cols: 32, unroll: 4, rows: 1 }),
    linearGemvWgsl("i4", 32),
    "linear gemv:r1 and the M=1 variant are identical text",
  );
  // GEMM shares one skeleton across 6 ops × weight storage × v4 (pins that generation holds no
  // state)
  for (const v4 of [false, true]) {
    assertEquals(matmulWgsl(v4), matmulWgsl(v4), `matmul:v4=${v4}`);
    assertEquals(bmmWgsl(v4), bmmWgsl(v4), `bmm:v4=${v4}`);
    assertEquals(attentionQkWgsl(v4), attentionQkWgsl(v4), `attention_qk:v4=${v4}`);
    assertEquals(attentionPvWgsl(v4), attentionPvWgsl(v4), `attention_pv:v4=${v4}`);
    assertEquals(
      linearWgsl("i4", v4, "f32", undefined, 32),
      linearWgsl("i4", v4, "f32", undefined, 32),
      `linear:i4:v4=${v4}`,
    );
    for (const weight of WEIGHT_STORAGES) {
      assertEquals(linearWgsl(weight, v4), linearWgsl(weight, v4), `linear:${weight}:v4=${v4}`);
      assertEquals(
        conv2dIgemmWgsl(weight, v4),
        conv2dIgemmWgsl(weight, v4),
        `conv2d igemm:${weight}:v4=${v4}`,
      );
      assertEquals(
        conv2dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL),
        conv2dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL),
        `conv2d igemm m32:${weight}:v4=${v4}`,
      );
      assertEquals(
        conv1dIgemmWgsl(weight, v4),
        conv1dIgemmWgsl(weight, v4),
        `conv1d igemm:${weight}:v4=${v4}`,
      );
      assertEquals(
        conv1dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL),
        conv1dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL),
        `conv1d igemm m32:${weight}:v4=${v4}`,
      );
    }
    for (const dp4a of [false, true]) {
      assertEquals(
        linearI8a8Wgsl(v4, dp4a),
        linearI8a8Wgsl(v4, dp4a),
        `linear i8a8:v4=${v4}:dp4a=${dp4a}`,
      );
      for (const groupSize of [16, 32, 64]) {
        assertEquals(
          linearI8a8Wgsl(v4, dp4a, undefined, "i4", groupSize),
          linearI8a8Wgsl(v4, dp4a, undefined, "i4", groupSize),
          `linear w4a8:v4=${v4}:dp4a=${dp4a}:g=${groupSize}`,
        );
      }
      assertEquals(
        attentionQkI8a8Wgsl(v4, dp4a),
        attentionQkI8a8Wgsl(v4, dp4a),
        `attention_qk i8a8:v4=${v4}:dp4a=${dp4a}`,
      );
      assertEquals(
        attentionPvI8a8Wgsl(v4, dp4a),
        attentionPvI8a8Wgsl(v4, dp4a),
        `attention_pv i8a8:v4=${v4}:dp4a=${dp4a}`,
      );
    }
  }
  // The generator functions the snapshots above call at only one point (a form that **returns a
  // different string the second time** = module-scope mutable state or a cache is in principle
  // invisible to one comparison at one point). The surface where "zero side effects in every
  // module" and "same key → byte-identical WGSL" apply directly.
  for (const dim of [4, 128, 256, 1024]) {
    const geometry = quantizeRowsGeometry(dim);
    assertEquals(
      quantizeRowsWgsl(geometry),
      quantizeRowsWgsl(geometry),
      `quantize_rows:dim=${dim}`,
    );
  }
  for (const dim of [5, 125, 4106]) {
    const geometry = quantizeRowsPaddedGeometry(dim);
    assertEquals(
      quantizeRowsPaddedWgsl(geometry),
      quantizeRowsPaddedWgsl(geometry),
      `quantize_rows padded:dim=${dim}`,
    );
  }
  for (const pixels of [4, 8, 16, 32, 64]) {
    assertEquals(
      quantizeRowsChannelMajorWgsl({ pixels }),
      quantizeRowsChannelMajorWgsl({ pixels }),
      `quantize_rows channel-major:p${pixels}`,
    );
  }
  for (const heads of [4, 8]) {
    assertEquals(stateQkFoldedWgsl(heads), stateQkFoldedWgsl(heads), `state_qk_fold:F=${heads}`);
  }
  for (const sliding of [false, true]) {
    assertEquals(
      stateStatsWgsl(sliding),
      stateStatsWgsl(sliding),
      `state_stats:sliding=${sliding}`,
    );
    assertEquals(
      stateAppendWgsl(sliding),
      stateAppendWgsl(sliding),
      `state_append:sliding=${sliding}`,
    );
    for (const gqa of [false, true]) {
      const where = `sliding=${sliding}:gqa=${gqa}`;
      assertEquals(stateQkWgsl(sliding, gqa), stateQkWgsl(sliding, gqa), `state_qk:${where}`);
      assertEquals(statePvWgsl(sliding, gqa), statePvWgsl(sliding, gqa), `state_pv:${where}`);
      assertEquals(
        statePvParallelWgsl(sliding, gqa),
        statePvParallelWgsl(sliding, gqa),
        `state_pv_par:${where}`,
      );
      assertEquals(
        stateQkParallelWgsl(sliding, gqa),
        stateQkParallelWgsl(sliding, gqa),
        `state_qk_par:${where}`,
      );
      assertEquals(
        stateStatsPvWgsl(sliding, gqa),
        stateStatsPvWgsl(sliding, gqa),
        `state_stats_pv:${where}`,
      );
      for (const rows of [16, 100, 768]) {
        assertEquals(
          stateQkTiledWgsl(sliding, gqa, rows),
          stateQkTiledWgsl(sliding, gqa, rows),
          `state_qk_tiled:${where}:M=${rows}`,
        );
        assertEquals(
          statePvTiledWgsl(sliding, gqa, rows),
          statePvTiledWgsl(sliding, gqa, rows),
          `state_pv_tiled:${where}:M=${rows}`,
        );
      }
    }
  }
  for (const epc of [undefined, 16]) {
    assertEquals(
      attentionStatsWgsl("f32", "f32", epc),
      attentionStatsWgsl("f32", "f32", epc),
      `attention_stats:epc=${epc}`,
    );
  }
  for (const k of [1, 4, TOPK_CORE_LIMIT_MAX_K]) {
    assertEquals(topkWgsl(k), topkWgsl(k), `topk:k=${k}`);
  }
  for (const direction of ["forward", "reverse"] as const) {
    assertEquals(gruScanWgsl(direction), gruScanWgsl(direction), `gru_scan:${direction}`);
  }
  assertEquals(gatedDeltaScanWgsl(), gatedDeltaScanWgsl(), "gated_delta_scan");
  for (const order of ["x-sigmoid", "sigmoid-x"] as const) {
    assertEquals(siluWgsl(order), siluWgsl(order), `silu:${order}`);
  }
  assertEquals(ropeWgsl("bshd", "pair"), ropeWgsl("bshd", "pair"), "rope:pair");
  assertEquals(adalnNormWgsl("token"), adalnNormWgsl("token"), "adaln_norm:token");
  for (const dim of [256, 1536, 2560, RMS_NORM_PREFETCH_MAX_DIM]) {
    assertEquals(rmsNormWorkgroupWgsl(dim), rmsNormWorkgroupWgsl(dim), `rms_norm:${dim}`);
    for (const order of ["norm-residual", "residual-norm"] as const) {
      assertEquals(
        rmsNormAddWgsl(order, dim),
        rmsNormAddWgsl(order, dim),
        `rms_norm_add:${order}:${dim}`,
      );
    }
  }
  assertEquals(embeddingWgsl("i4", 32), embeddingWgsl("i4", 32), "embedding:i4:g32");
  for (const weight of WEIGHT_STORAGES) {
    assertEquals(embeddingWgsl(weight), embeddingWgsl(weight), `embedding:${weight}`);
    assertEquals(conv1dWgsl(weight), conv1dWgsl(weight), `conv1d:${weight}`);
    assertEquals(conv2dWgsl(weight), conv2dWgsl(weight), `conv2d:${weight}`);
    assertEquals(
      convTranspose1dWgsl(weight),
      convTranspose1dWgsl(weight),
      `conv_transpose1d:${weight}`,
    );
  }
});

test("pipeline keys are unique per generator input (no two kernels map to the same key)", () => {
  const keys: string[] = [
    // The product of the 3 GEMM ops × weight storage × v4 (the enumeration is kept as one so
    // nothing is missed)
    ...GEMM_VARIANTS.map(([, key]) => key),
    GATHER_KEY,
    LAYER_NORM_KEY,
    RMS_NORM_KEY,
    RMS_NORM_128_KEY,
    // The prefetch form bakes its trip count into the WGSL, so each width is its own key.
    ...[256, 512, 1536, 2560, RMS_NORM_PREFETCH_MAX_DIM].flatMap((dim) => [
      rmsNormWorkgroupKey(dim),
      rmsNormAddKey("norm-residual", dim),
      rmsNormAddKey("residual-norm", dim),
    ]),
    SOFTMAX_KEY,
    SAFE_SOFTMAX_KEY,
    ATTENTION_STATS_KEY,
    MASKED_FILL_KEY,
    CUMSUM_KEY,
    // argmax is a **separate pipeline** from the reduce family (the identity is −inf, it carries
    // (value, index) pairs). On a collision one WGSL is assigned to the other's dispatch, and since
    // the bind surface (i32 output) differs, a different bit pattern is read without any exception.
    ARGMAX_KEY,
    // topk is a **separate pipeline per k** (k is baked into the WGSL, so the array length and the
    // round count change). Without k in the key, the pipeline built for the first k is handed to a
    // dispatch with another k, and an output with a different number of written entries (the rest
    // left over from the previous run) is read without any exception. It does not collide with
    // argmax either.
    TOPK_ONE_SPLIT_MERGE_KEY,
    topkKey(1),
    topkKey(4),
    PAD_KEY,
    FLIP_KEY,
    // The 2 upsample kernels are **listed as a pair** (the layer 3 fusion rule `upsample2x` =
    // bitwise nearest replication / the layer 1 op `upsample_bilinear2d`). Their similar names make
    // the accident of mapping them to the same key likely, and when it happens nearest's IR runs on
    // the bilinear pipeline.
    UPSAMPLE_2X_KEY,
    UPSAMPLE_BILINEAR2D_KEY,
    DEFORM_CONV2D_KEY,
    // The 5 fused kernels have their weight-storage variants (ADR 0018). The WGSL bodies differ, so
    // if variants map to the same key one silently runs on the other's pipeline (linear is covered,
    // with v4, by the GEMM product above).
    ...WEIGHT_STORAGES.flatMap((weight) => [
      embeddingKey(weight),
      conv1dKey(weight),
      conv2dKey(weight),
      convTranspose1dKey(weight),
    ]),
    // conv2d's 32-row m-tile variant (next to ADR 0024). The condition is a **key separate from the
    // 64-row version**: if they map to the same key one WGSL runs on the other's pipeline (the
    // workgroup shape differs, so it is an immediate wrong value rather than a silent no-op).
    ...WEIGHT_STORAGES.flatMap((weight) =>
      [false, true].flatMap((v4) => [
        conv2dIgemmKey(weight, v4, GEMM_MTILE_SMALL),
        conv1dIgemmKey(weight, v4, GEMM_MTILE_SMALL),
        conv3dIgemmKey(weight, v4, GEMM_MTILE_SMALL),
      ]),
    ),
    // i4 has a **separate key per group length** (the shift is baked into the WGSL — ADR 0069).
    // Without the g part in the key, the group 32 pipeline runs on group 64 assets and the scale
    // index is off, a silent wrong value (linear g32's basic form is covered by the GEMM product
    // above).
    ...[false, true].flatMap((v4) => [
      linearKey("i4", v4, "f32", undefined, 64),
      linearKey("i4", v4, "f32", undefined, 128),
    ]),
    // The GEMV family (ADR 0082) is **separate per `linear:` family name** (the binding element
    // type and the workgroup shape differ, so on a collision one WGSL is handed to the other's
    // dispatch, an immediate wrong value). A separate key per group length is the same discipline
    // as i4's default path (the shift is baked into the WGSL — ADR 0069).
    ...[32, 64, 128].map((groupSize) => linearGemvKey("i4", groupSize)),
    // The storage (i4 / i8) is in the key too — the element count per word differs, 32 / 16, so a
    // collision gives a silent wrong value where the reduction runs over half / double the row
    // (perf-ledger K-16).
    linearGemvKey("i8"),
    linearGemvKey("f16"),
    linearGemvKey("f32"),
    // The variant (column count / prefetch count) is in the key too — without it, the pipeline
    // built for the first shape runs on another shape's dispatch, a silent wrong value from the
    // workgroup-shape mismatch.
    linearGemvKey("i4", 32, { cols: 64, unroll: 4 }),
    linearGemvKey("i4", 32, { cols: 32, unroll: 2 }),
    linearGemvKey("i8", undefined, { cols: 64, unroll: 4 }),
    // The row-block variants (perf-ledger K-21) put `r<rows>` in the key too — without it, rows 4
    // assets are handed to a rows 16 dispatch, the accumulator count and the y tile height
    // disagree, an immediate wrong value. The same enumeration checks that `r1` (= **separate
    // text** from the M=1 variant `c32u4` above) does not collide with the M=1 key.
    ...[1, 2, 4, 16].flatMap((rows) => [
      linearGemvRowsKey("i4", 32, { cols: 32, unroll: 4, rows }),
      linearGemvRowsKey("i4", 64, { cols: 32, unroll: 4, rows }),
      linearGemvRowsKey("i8", undefined, { cols: 32, unroll: 4, rows }),
    ]),
    ...[16, 32, 64].map((groupSize) => embeddingKey("i4", groupSize)),
    // w8a8: the 4 of v4 × integer dot-product variants + the activation quantization. The condition
    // is **separate keys for dp4a and emulation**: on the same key, diagnostics could not tell
    // which one ran (design §4.4-5).
    ...[false, true].flatMap((v4) => [linearI8a8Key(v4, false), linearI8a8Key(v4, true)]),
    // w4a8 has a **separate key per storage discriminator + group length** (g is baked into the
    // WGSL as a shift). Without the g part, the group 32 pipeline runs on group 64 assets and the
    // scale index is off, a silent wrong value. The condition includes not colliding with the i8
    // key of the same geometry and the same v4.
    ...[false, true].flatMap((v4) =>
      [16, 32, 64].flatMap((groupSize) => [
        linearI8a8Key(v4, false, undefined, "i4", groupSize),
        linearI8a8Key(v4, true, undefined, "i4", groupSize),
      ]),
    ),
    // The i8a8 variants of fused attention ①QK / ③PV follow the same discipline
    // (the 3 lines f32 / :c16 / i8a8 × the 2 stages are all separate keys)
    ...[false, true].flatMap((v4) => [
      attentionQkI8a8Key(v4, false),
      attentionQkI8a8Key(v4, true),
      attentionPvI8a8Key(v4, false),
      attentionPvI8a8Key(v4, true),
    ]),
    QUANTIZE_ROWS_KEY,
    // Every geometry of the channel-major form is its own WGSL, so its own key
    ...[4, 8, 16, 32, 64].map((pixels) => quantizeRowsChannelMajorKey({ pixels })),
    // The fusion kernels of the DiT rules: each variant (rotation / modulation source) has its own
    // key, so a pipeline built for one form is never handed the other form's bind surface.
    ROPE_KEY,
    ROPE_BSHD_KEY,
    ROPE_PAIR_BSHD_KEY,
    ADALN_NORM_KEY,
    ADALN_NORM_TOKEN_KEY,
    ADALN_MODULATION_PACK_KEY,
    GATED_RESIDUAL_KEY,
  ];
  for (const [op, dtype] of elementwiseDtypes()) {
    for (const rank of [1, 2, 3, 4]) {
      keys.push(elementwiseKey({ op, rank, dtype } as ElementwiseSpec));
    }
  }
  for (const from of SEMANTIC_DTYPES) {
    for (const to of SEMANTIC_DTYPES) {
      for (const rank of [1, 2]) keys.push(elementwiseKey({ op: "cast", rank, dtype: from, to }));
    }
  }
  for (const op of REDUCE_OPS) {
    for (const dtype of OP_CONTRACTS.get(op)?.dtypes ?? []) {
      keys.push(reduceKey({ op, dtype }));
      // The axis variants are a **separate pipeline** from the row reduce (on a collision one WGSL
      // is assigned to the other's dispatch and passes with the reduction axis mixed up, without
      // any exception)
      keys.push(axisReduceKey({ op, dtype }));
      // every split is its own WGSL, so its own key (a split missing from the key would hand one
      // split's pipeline to another split's dispatch grid)
      for (const split of [1, 2, 4, 8, 32, 64, 128, 256])
        keys.push(axisReduceKey({ op, dtype }, split));
    }
  }
  for (const dtype of SEMANTIC_DTYPES) {
    keys.push(stridedKey({ dtype }));
    // The read family and the write family have different WGSL (opposite traversal) — their keys
    // must not collide for the same element type
    keys.push(stridedWriteKey({ dtype }));
  }
  assertEquals(new Set(keys).size, keys.length);
});

// permute / expand are one kernel family (ADR 0011). The WGSL is the same, so there is no separate
// key per op; only the element type is an axis of separate pipelines.
test("strided copy is a separate kernel per element type, and the fixed rank is in the key", () => {
  assertEquals(stridedWgsl({ dtype: "f32" }).includes("array<f32>"), true);
  assertEquals(stridedWgsl({ dtype: "i32" }).includes("array<i32>"), true);
  // bool is stored as u32 0/1 (ADR 0009)
  assertEquals(stridedWgsl({ dtype: "bool" }).includes("array<u32>"), true);
  assertNotEquals(stridedKey({ dtype: "i32" }), stridedKey({ dtype: "bool" }));
  for (const dtype of SEMANTIC_DTYPES) {
    assertEquals(stridedKey({ dtype }).includes(`r${STRIDED_RANK}`), true);
    // The output is always contiguous, so only the src side is reindexed
    assertEquals(stridedWgsl({ dtype }).includes("out[i] = src[src_index];"), true);
  }
});

test("permute's strides are the axis permutation, and expand's replicated axes have stride 0", () => {
  // The strides of a contiguous [2,3,4] are [12,4,1]. dims=[0,2,1] has output axis 1 take input
  // axis 2.
  assertEquals(permuteSrcStrides([2, 3, 4], [0, 2, 1]), [12, 1, 4]);
  assertEquals(permuteSrcStrides([1, 16, 5, 64], [0, 2, 1, 3]), [5120, 64, 320, 1]);
  // MUST: include one cycle of length 3 or more. Every permutation measured in practice is an
  // involution (inverse permutation = itself), so the mistake of building the stride table from
  // the inverse permutation cannot be detected with an involution. The inverse of [1,2,0] is
  // [2,0,1].
  assertEquals(permuteSrcStrides([2, 3, 4], [1, 2, 0]), [4, 1, 12]);
  assertEquals(permuteSrcStrides([2, 3, 4], [2, 0, 1]), [1, 12, 4]);
  // Measured shapes: a gather index [1,T,T] → [16,T,T] (the head replicated) and a bool mask
  // [1,T,1] → [1,T,1024]
  assertEquals(expandSrcStrides([1, 6, 6], [16, 6, 6]), [0, 6, 1]);
  assertEquals(expandSrcStrides([1, 6, 1], [1, 6, 1024]), [0, 1, 0]);
  // When the rank grows, the leading axes also get stride 0
  assertEquals(expandSrcStrides([4], [3, 4]), [0, 1]);
});

test("strided params pad a missing rank on the left with 1 / 0", () => {
  // Output [3,4] (rank2) → dims are [1,1,3,4], strides are [0,0,...]
  const params = stridedParams([3, 4], [1, 3], 0);
  assertEquals(params.length, 2 + 2 * STRIDED_RANK);
  assertEquals([...params.slice(0, 5)], [12, 1, 1, 3, 4]);
  assertEquals([...params.slice(5, 9)], [0, 0, 1, 3]);
  assertEquals(params[9], 0, "the offset of permute / expand is always 0");
  assertEquals(
    [...stridedParams([2, 3, 4, 5], [60, 20, 5, 1], 7).slice(1)],
    [2, 3, 4, 5, 60, 20, 5, 1, 7],
  );
});

// ADR 0014: slice reuses the existing read family, and the only variable point is **one word of
// offset in params**.
test("a slice reads at the input's contiguous strides with start × that axis's stride", () => {
  assertEquals(sliceSrcStrides([2, 3, 4]), [12, 4, 1]);
  // The same shape as enc_p's m_p / logs_p split (taking the second half of the channel axis)
  assertEquals(sliceSrcOffset([1, 8, 5], 1, 4), 20);
  assertEquals(sliceSrcOffset([1, 8, 5], 1, 0), 0, "a slice from the start has offset 0");
  // A slice of the last dimension has stride 1, so offset = start itself
  assertEquals(sliceSrcOffset([3, 7], 1, 2), 2);
  // The end (exactly the axis length) is valid as the start of an empty slice
  assertEquals(sliceSrcOffset([3, 7], 0, 3), 21);
  assertThrows(
    () => sliceSrcOffset([3, 7], 2, 0),
    CodegenError,
    undefined,
    "axis outside the rank",
  );
  assertThrows(
    () => sliceSrcOffset([3, 7], 0, 4),
    CodegenError,
    undefined,
    "start beyond the axis length",
  );
  assertThrows(() => sliceSrcOffset([3, 7], 0, -1), CodegenError);
});

// ADR 0014: cat is the write family. A wrong offset writes part of the output twice and leaves the
// rest unwritten (holding the previous values when the pool reuses the buffer).
test("cat writes at the output's contiguous strides with the preceding inputs' length × that axis's stride", () => {
  assertEquals(catOutStrides([2, 6, 4]), [24, 4, 1]);
  // The same shape as coupling reverse's 96/96 split: the second input is written from element 96
  // of axis 1
  assertEquals(catOutOffset([1, 8, 5], 1, 0), 0);
  assertEquals(catOutOffset([1, 8, 5], 1, 4), 20);
  assertEquals(
    catOutOffset([1, 8, 5], 1, 8),
    40,
    "the position after everything is written (the right end of the coverage check)",
  );
  assertThrows(() => catOutOffset([1, 8, 5], 3, 0), CodegenError);
  assertThrows(
    () => catOutOffset([1, 8, 5], 1, 9),
    CodegenError,
    undefined,
    "beyond the axis length",
  );
});

test("strided write params carry the input dims and the output strides in the same layout", () => {
  // Input [3,4] (rank2) → dims are left-padded to [1,1,3,4], strides are left-padded with 0 too
  const params = stridedWriteParams([3, 4], [7, 1], 21);
  assertEquals(params.length, 2 + 2 * STRIDED_RANK);
  assertEquals(
    [...params.slice(0, 5)],
    [12, 1, 1, 3, 4],
    "n is the element count of the **input**",
  );
  assertEquals([...params.slice(5, 9)], [0, 0, 7, 1]);
  assertEquals(params[9], 21, "offset of the write start");
  assertThrows(() => stridedWriteParams([], [], 0), CodegenError);
  assertThrows(() => stridedWriteParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => stridedWriteParams([3, 4], [4, 1], -1), CodegenError);
});

test("pad / flip params carry the 3-way split into rows and the axis as is", () => {
  // Padding [2,3] on the last dimension by [1,2] → output [2,6], n=12
  assertEquals([...padParams(2, 3, 1, 2)], [12, 6, 3, 1]);
  // A zero-width pad (identity copy) can be expressed too
  assertEquals([...padParams(4, 5, 0, 0)], [20, 5, 5, 0]);
  assertThrows(() => padParams(2, 3, -1, 0), CodegenError);
  assertThrows(() => padParams(2, 3, 0, 1.5), CodegenError);
  // Flipping axis 1 of [2,3,4] → outer=2 / len=3 / inner=4
  assertEquals([...flipParams(2, 3, 4)], [24, 3, 4, 0]);
  // Flipping the last axis has inner = 1
  assertEquals([...flipParams(6, 4, 1)], [24, 4, 1, 0]);
  assertThrows(() => flipParams(2, -1, 4), CodegenError);
});

test("strided codegen fails loudly on generator inputs outside the contract", () => {
  const foreign: { dtype: string } = { dtype: "f16" };
  assertThrows(() => stridedWgsl(foreign as StridedSpec), CodegenError);
  assertThrows(() => stridedKey(foreign as StridedSpec), CodegenError);
  // rank 0 / beyond STRIDED_RANK
  assertThrows(() => stridedParams([], [], 0), CodegenError);
  assertThrows(() => stridedParams([2, 2, 2, 2, 2], [16, 8, 4, 2, 1], 0), CodegenError);
  assertThrows(() => stridedParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => stridedParams([3, 4], [1, 3], -1), CodegenError);
  assertThrows(() => permuteSrcStrides([2, 3], [0, 1, 2]), CodegenError);
  assertThrows(() => permuteSrcStrides([2, 3], [0, 5]), CodegenError);
  assertThrows(() => expandSrcStrides([2, 3], [3]), CodegenError);
});

// The element type is embedded in the bindings' array<T>. Without it in the key, a pipeline
// compiled for f32 is assigned under the same key to an i32 dispatch, and the bit reinterpretation
// passes without any exception.
test("the element type appears in both the WGSL and the key (f32 and i32 are separate kernels)", () => {
  const f32 = elementwiseWgsl({ op: "mul", rank: 2, dtype: "f32" });
  const i32 = elementwiseWgsl({ op: "mul", rank: 2, dtype: "i32" });
  assertNotEquals(f32, i32);
  assertEquals(f32.includes("array<f32>"), true);
  assertEquals(i32.includes("array<i32>"), true);
  assertNotEquals(
    elementwiseKey({ op: "mul", rank: 2, dtype: "f32" }),
    elementwiseKey({ op: "mul", rank: 2, dtype: "i32" }),
  );
  // bool is stored as u32 0/1 (storage buffers have no 1-bit type — ADR 0009)
  assertEquals(
    elementwiseWgsl({ op: "bitwise_not", rank: 1, dtype: "bool" }).includes("array<u32>"),
    true,
  );
  // cast has different element types for input and output (putting only one in the key collides)
  const widen = elementwiseWgsl({ op: "cast", rank: 1, dtype: "bool", to: "f32" });
  assertEquals(widen.includes("read> in0: array<u32>"), true);
  assertEquals(widen.includes("read_write> out: array<f32>"), true);
  assertNotEquals(
    elementwiseKey({ op: "cast", rank: 1, dtype: "bool", to: "f32" }),
    elementwiseKey({ op: "cast", rank: 1, dtype: "f32", to: "bool" }),
  );
});

// The contract (src/ops.ts): f32 → integer truncates toward 0, x → bool is x != 0.
test("cast's value expression follows the contract (truncation by i32() / truthiness by != 0)", () => {
  assertEquals(
    elementwiseWgsl({ op: "cast", rank: 1, dtype: "f32", to: "i32" }).includes("out[i] = i32(v0);"),
    true,
  );
  assertEquals(
    elementwiseWgsl({ op: "cast", rank: 1, dtype: "f32", to: "bool" }).includes(
      "out[i] = select(0u, 1u, v0 != 0.0);",
    ),
    true,
  );
  assertEquals(
    elementwiseWgsl({ op: "cast", rank: 1, dtype: "i32", to: "bool" }).includes(
      "out[i] = select(0u, 1u, v0 != 0);",
    ),
    true,
  );
});

test("the pipeline key includes the workgroup size embedded in the WGSL", () => {
  for (const rank of [1, 2, 3, 4]) {
    const key = elementwiseKey({ op: "relu", rank, dtype: "f32" });
    assertEquals(key.includes(`wg${ELEMENTWISE_WORKGROUP_SIZE}`), true, key);
  }
});

test("elementwise generated WGSL injects helper functions only into the ops that use them", () => {
  const gelu = elementwiseWgsl({ op: "gelu", rank: 1, dtype: "f32" });
  const relu = elementwiseWgsl({ op: "relu", rank: 1, dtype: "f32" });
  assertEquals(gelu.includes("fn erf_approx"), true);
  assertEquals(gelu.includes("fn sigmoid_stable"), false);
  assertEquals(relu.includes("fn erf_approx"), false);
  // The tanh form does not pull in erf's approximation. The literal √(2/π) is pinned here (mixing
  // up the 2 gelus would still match the shape and the dtype).
  // MUST: the inside calls the saturating-clamp version (TANH_STABLE_WGSL), not the built-in
  // `tanh` — going back to the plain tanh, the exp-based implementation gives a silent NaN for
  // pre-activations x ≳ 10.05.
  const geluTanh = elementwiseWgsl({ op: "gelu_tanh", rank: 1, dtype: "f32" });
  assertEquals(geluTanh.includes("fn erf_approx"), false);
  assertEquals(
    geluTanh.includes("tanh_stable(0.7978845608028654 * (v0 + 0.044715 * v0 * v0 * v0))"),
    true,
  );
  // The dependency between helper functions (tanh_stable → is_nan_bits) is closed. is_nan_bits does
  // not appear in the value expression's text, so if injection stopped after one level, the WGSL
  // would come out with an undefined function.
  assertEquals(geluTanh.includes("fn tanh_stable"), true);
  assertEquals(geluTanh.includes("fn is_nan_bits"), true);
  // sin passes straight through to the WGSL built-in. This pins that it was not replaced with a
  // polynomial approximation (the erf form's precedent) and that no NaN shell (the bit-pattern
  // test from {@link nanGuard}) slipped in.
  const sin = elementwiseWgsl({ op: "sin", rank: 1, dtype: "f32" });
  assertEquals(sin.includes("out[i] = sin(v0);"), true);
  assertEquals(sin.includes("fn erf_approx"), false);
  assertEquals(sin.includes("fn is_nan_bits"), false);
  // MUST: the naive 1/(1+exp(-x)) overflows into an indeterminate form for x ≲ -88
  assertEquals(
    elementwiseWgsl({ op: "sigmoid", rank: 1, dtype: "f32" }).includes("exp(-abs(x))"),
    true,
  );
});

/**
 * Permanently pins that the saturating-clamp threshold of tanh ({@link TANH_SATURATION} in
 * src/codegen/elementwise.ts) meets the 2 selection conditions **numerically**. Moving the constant
 * breaks this immediately.
 *
 * 1. **The bit-identity condition**: f32 tanh already rounds to exactly ±1.0 at the threshold.
 *    If this breaks, the clamp changes values = the claim "bit-identical to an IEEE-faithful
 *    implementation" no longer holds.
 * 2. **The overflow-avoidance condition**: `exp(2·threshold)` is far enough from the f32 maximum.
 *    If this breaks, the exp-based implementation returns Inf/Inf = NaN even with the clamp (the
 *    very purpose of the fix).
 *
 * MUST: do not make it tautological — **below** the threshold condition 1 actually fails (a
 * control is placed below).
 */
test('tanh\'s saturating-clamp threshold lies in the band "already saturated × far from overflow"', () => {
  // Condition 1: f32 tanh is exactly ±1.0 both at the threshold and at its negation.
  assertEquals(Math.fround(Math.tanh(TANH_SATURATION)), 1);
  assertEquals(Math.fround(Math.tanh(-TANH_SATURATION)), -1);
  // Control: below f32's saturation point (measured 9.011) it does not round to 1.0 — the 2 above
  // are not tautological.
  assertEquals(Math.fround(Math.tanh(9)) < 1, true, `${Math.fround(Math.tanh(9))}`);
  // Condition 2: the intermediate value of the exp-based implementation
  // ((exp(2x)−1)/(exp(2x)+1)) is orders of magnitude away from the f32 maximum 3.4028e38. What the
  // plain gelu_tanh hit was pre-activation 11.45 → inner argument 62.7 → exp(125.4) = 2.8e54
  // (= overflow).
  assertEquals(Math.exp(2 * TANH_SATURATION) < 1e30, true);
  // Control: without the clamp, arguments beyond the f32 maximum do exist (44.36 is the boundary).
  assertEquals(Math.exp(2 * 44.4) > 3.4028234663852886e38, true);
});

// The math ops added in M1-P3 iteration 3. Kernel-specific invariants are pinned from the output
// side (value comparison is in tests/gpu_ops_test.ts, the contract in
// tests/ops_conformance_test.ts).
test("the value expressions of where / comparisons / clamp / leaky_relu generate the contracted forms", () => {
  // MUST: torch's where(cond, a, b) = cond ? a : b. WGSL's select(f, t, cond) has the arguments in
  // the opposite order, so a transcription mistake swaps the branches while the shape and the
  // dtype still match.
  const where = elementwiseWgsl({ op: WHERE_OP, rank: 2, dtype: "f32" });
  assertEquals(where.includes("out[i] = select(v2, v1, v0 != 0u);"), true);
  // Only the condition slot is u32 (bool); the value slots and the output are f32
  assertEquals(where.includes("read> in0: array<u32>;"), true);
  assertEquals(where.includes("read> in1: array<f32>;"), true);
  assertEquals(where.includes("read> in2: array<f32>;"), true);
  assertEquals(where.includes("read_write> out: array<f32>;"), true);

  // Comparisons read f32 and write bool (u32 0/1)
  const ge = elementwiseWgsl({ op: "ge", rank: 1, dtype: "f32" });
  assertEquals(ge.includes("read> in0: array<f32>;"), true);
  assertEquals(ge.includes("read_write> out: array<u32>;"), true);
  assertEquals(ge.includes("out[i] = select(0u, 1u, v0 >= v1);"), true);
  // The boundary (the equality point) of ge and gt_scalar can only be told apart by the generated
  // expression's operator
  assertEquals(
    elementwiseWgsl({ op: "ge_scalar", rank: 1, dtype: "f32" }).includes(
      "out[i] = select(0u, 1u, v0 >= s0);",
    ),
    true,
  );
  assertEquals(
    elementwiseWgsl({ op: "gt_scalar", rank: 1, dtype: "f32" }).includes(
      "out[i] = select(0u, 1u, v0 > s0);",
    ),
    true,
  );
  assertEquals(
    elementwiseWgsl({ op: "le_scalar", rank: 1, dtype: "f32" }).includes(
      "out[i] = select(0u, 1u, v0 <= s0);",
    ),
    true,
  );

  // MUST: leaky_relu is the select form (`max(x, s·x)` does not work because WGSL's max does not
  // guarantee NaN propagation — ADR 0015). The slope is s0 from params and is not baked into the
  // WGSL.
  const leaky = elementwiseWgsl({ op: "leaky_relu", rank: 1, dtype: "f32" });
  assertEquals(leaky.includes("out[i] = select(s0 * v0, v0, v0 >= 0.0);"), true);
  assertEquals(leaky.includes("max("), false);
  assertEquals(leaky.includes("let s0 = bitcast<f32>(params[3u]);"), true);

  // MUST: clamp does not use the built-in clamp either (lo > hi is implementation-defined). NaN is
  // propagated by a bit-pattern shell — the inner 2-level select cannot be relied on, since the
  // compiler folds it into the max idiom.
  const clamp = elementwiseWgsl({ op: "clamp", rank: 1, dtype: "f32" });
  assertEquals(
    clamp.includes(
      "out[i] = select(select(select(v0, s0, v0 < s0), s1, v0 > s1), v0, is_nan_bits(v0));",
    ),
    true,
  );
  assertEquals(clamp.includes("clamp("), false);

  // The logical AND of bool is written as the boolean &&, not a bitwise op (it stays boolean even
  // for stored values other than 0/1)
  assertEquals(
    elementwiseWgsl({ op: "bitwise_and", rank: 1, dtype: "bool" }).includes(
      "out[i] = select(0u, 1u, (v0 != 0u) && (v1 != 0u));",
    ),
    true,
  );
});

/**
 * The NaN-propagation shell (a bit-pattern test). Pins on the output side that the test **is not
 * done with a floating-point comparison** — the shader compiler folds the comparison form into
 * the `max` idiom, and the driver's `max` swallows NaN (the propagation itself on a real GPU is
 * covered by tests/gpu_ops_test.ts).
 */
test("clamp / clamp_min / relu / amax / amin test for NaN by bit pattern, not by comparison", () => {
  const bits = "return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;";
  for (const op of ["clamp", "clamp_min", "relu"] as const) {
    const wgsl = elementwiseWgsl({ op, rank: 1, dtype: "f32" });
    assertEquals(wgsl.includes(`fn is_nan_bits(x: f32) -> bool {\n  ${bits}\n}`), true, op);
    // The shell has the form "if NaN, return the value as is" (the inner expression handles only
    // the non-NaN path)
    assertEquals(wgsl.includes(", v0, is_nan_bits(v0));"), true, op);
  }
  // The injection discipline of helper functions (same as erf / log1p) — it does not leak into ops
  // that need no shell
  for (const op of ["leaky_relu", "abs", "neg"] as const) {
    const wgsl = elementwiseWgsl({ op, rank: 1, dtype: "f32" });
    assertEquals(wgsl.includes("is_nan_bits"), false, op);
  }
  // MUST: the reduction has 2 levels (one thread's scan + the workgroup tree). Guarding only one
  // still lets NaN be swallowed by the identity (±F32_MAX) at the other level, so both levels must
  // go through the same function.
  for (const [op, fn] of [
    ["amax", "nan_max"],
    ["amin", "nan_min"],
  ] as const) {
    const wgsl = reduceWgsl({ op, dtype: "f32" });
    assertEquals(wgsl.includes(bits), true, op);
    assertEquals(wgsl.includes(`acc = ${fn}(acc, x[base + i]);`), true, op);
    assertEquals(
      wgsl.includes(`scratch[lid] = ${fn}(scratch[lid], scratch[lid + stride]);`),
      true,
      op,
    );
    // The axis variant has three stages (the per-slot scan, the carry stack, the cross-subtree
    // combine) and every one goes through the same function.
    const axis = axisReduceWgsl({ op, dtype: "f32" });
    assertEquals(axis.includes(bits), true, `${op}:axis`);
    assertEquals(axis.includes(`v = ${fn}(v, x[base + c * inner]);`), true, `${op}:axis`);
    assertEquals(axis.includes(`v = ${fn}(acc[k], v);`), true, `${op}:axis`);
    const outputs = axisReduceOutputsPerGroup();
    assertEquals(
      axis.includes(`roots[lid.x] = ${fn}(roots[lid.x], roots[lid.x + ${outputs}u]);`),
      true,
      `${op}:axis`,
    );
  }
  assertEquals(reduceWgsl({ op: "sum", dtype: "f32" }).includes("is_nan_bits"), false);
  assertEquals(axisReduceWgsl({ op: "sum", dtype: "f32" }).includes("is_nan_bits"), false);
});

/**
 * The axis reduce is bit-identical to the row reduce only while several numbers hold together: the
 * same workgroup size in both variants, leaves × split = 256, a carry stack of log2(leaves) + 1
 * levels, a bit-reversal shift of 32 − log2(256), and log2(split) cross-subtree levels whose left
 * operand is the lower subtree. None of them raises an exception when it drifts (too few levels
 * read a level never written; a wrong shift or a swapped operand only changes the rounding of sum),
 * so they are checked arithmetically on the generated text. Unlike the GPU parity test
 * (tests/gpu_reduce_axis_parity_test.ts) this runs without an adapter; the full tree is replayed in
 * tests/reduce_axis_tree_test.ts.
 */
test("axis reduce: thread mapping, stack depth, shift and combine levels follow from size and split", () => {
  const size = AXIS_REDUCE_WORKGROUP_SIZE;
  const bits = Math.log2(size);
  assertEquals(Number.isInteger(bits), true, "the row reduce's tree needs a power of 2 too");
  for (const split of [1, 2, 4, 8, 16, 32, 64, 128, 256]) {
    const axis = axisReduceWgsl({ op: "sum", dtype: "f32" }, split);
    const leaves = size / split;
    const leafBits = Math.log2(leaves);
    const where = `split ${split}`;
    assertEquals(leaves * split, size, where);
    assertEquals(axisReduceOutputsPerGroup(split), leaves, where);
    assertEquals(axis.includes(`@workgroup_size(${size})`), true, where);
    assertEquals(axis.includes(`let o = lid.x % ${leaves}u;`), true, where);
    assertEquals(axis.includes(`let s = lid.x / ${leaves}u;`), true, where);
    assertEquals(axis.includes(`for (var j = 0u; j < ${leaves}u;`), true, where);
    assertEquals(
      axis.includes(`let slot = reverseBits(s * ${leaves}u + j) >> ${32 - bits}u;`),
      true,
      where,
    );
    assertEquals(axis.includes(`c = c + ${size}u;`), true, where);
    assertEquals(axis.includes(`var acc: array<f32, ${leafBits + 1}>;`), true, where);
    assertEquals(axis.includes(`root = acc[${leafBits}u];`), true, where);
    // log2(split) levels, w = 1, 2, 4, …: subtree s (s % 2w == 0) adds subtree s + w on its right
    const levels = [...axis.matchAll(/if \(s % (\d+)u == 0u\) \{\n\s*(.*)\n/g)].map((match) => [
      Number(match[1]),
      match[2],
    ]);
    assertEquals(levels.length, Math.log2(split), where);
    for (const [index, [modulus, line]] of levels.entries()) {
      const width = 2 ** index;
      assertEquals(modulus, 2 * width, where);
      assertEquals(line, `roots[lid.x] = roots[lid.x] + roots[lid.x + ${width * leaves}u];`, where);
    }
    assertEquals(axis.includes(`tb = tb + nwg.x * ${leaves}u;`), true, where);
  }
  // the row reduce has the same size (the leaf order matches only at equal sizes)
  const row = reduceWgsl({ op: "sum", dtype: "f32" });
  assertEquals(row.includes(`@workgroup_size(${size})`), true);
  assertEquals(row.includes(`i = i + ${size}u;`), true);
});

/**
 * The barriers of the axis kernel, in order. Level `w` reads `roots[lid.x + w·outputs]`, a slot
 * another lane wrote, so the roots store and every combine level MUST be followed by a barrier
 * before the next cross-lane read. A value comparison can miss a dropped one on a GPU whose
 * subgroups happen to cover both lanes (a 32-wide warp holds both lanes of the first level at the
 * production split), so the order is checked on the generated text. After the last level only lanes with `s == 0` read
 * `roots[o]`, their own slot, so the block ends without a barrier.
 */
const axisBarrierSequence = (wgsl: string): readonly string[] =>
  wgsl.split("\n").flatMap((line) => {
    const statement = line.trim();
    if (statement === "workgroupBarrier();") return ["barrier"];
    if (statement === "roots[lid.x] = root;") return ["store"];
    if (statement.startsWith("if (s % ")) return ["level"];
    if (statement === "out[t] = roots[o];") return ["out"];
    return [];
  });

test("axis reduce: a barrier follows the roots store and every combine level, and only those", () => {
  const expected = (split: number): readonly string[] => [
    "store",
    "barrier",
    ...Array.from({ length: Math.log2(split) }, () => ["level", "barrier"]).flat(),
    "out",
  ];
  for (const split of [2, AXIS_REDUCE_SPLIT, 256]) {
    const wgsl = axisReduceWgsl({ op: "sum", dtype: "f32" }, split);
    assertEquals(axisBarrierSequence(wgsl), expected(split), `split ${split}`);
  }
  // The check fails when the barrier after the store, or one after a level, is dropped.
  const wgsl = axisReduceWgsl({ op: "sum", dtype: "f32" }, AXIS_REDUCE_SPLIT);
  const store = "    roots[lid.x] = root;\n    workgroupBarrier();\n";
  assertEquals(wgsl.split(store).length, 2, "the store line moved (the fault would not apply)");
  assertNotEquals(
    axisBarrierSequence(wgsl.replace(store, "    roots[lid.x] = root;\n")),
    expected(AXIS_REDUCE_SPLIT),
  );
  const level = "    }\n    workgroupBarrier();\n    if (s % ";
  assert(wgsl.includes(level), "the level layout moved (the fault would not apply)");
  assertNotEquals(
    axisBarrierSequence(wgsl.replace(level, "    }\n    if (s % ")),
    expected(AXIS_REDUCE_SPLIT),
  );
});

test("axis reduce rejects a split that is not a power of 2 up to the workgroup size", () => {
  for (const split of [0, 3, 6, 24, 0.5, 512, -2, Number.NaN]) {
    assertThrows(
      () => axisReduceWgsl({ op: "sum", dtype: "f32" }, split),
      CodegenError,
      undefined,
      `${split}`,
    );
    assertThrows(
      () => axisReduceKey({ op: "sum", dtype: "f32" }, split),
      CodegenError,
      undefined,
      `${split}`,
    );
    assertThrows(() => axisReduceOutputsPerGroup(split), CodegenError, undefined, `${split}`);
  }
});

describe("quantize_rows channel-major geometry", () => {
  it("rejects pixels per workgroup that are not a power of 2 in [4, 64], on every entry point", () => {
    for (const pixels of [0, 1, 2, 3, 12, 24, 48, 128, 256, -4, 0.5, 16.5, Number.NaN]) {
      const geometry = { pixels };
      assertThrows(
        () => quantizeRowsChannelMajorKey(geometry),
        CodegenError,
        undefined,
        `${pixels}`,
      );
      assertThrows(
        () => quantizeRowsChannelMajorWgsl(geometry),
        CodegenError,
        undefined,
        `${pixels}`,
      );
      assertThrows(
        () => quantizeRowsChannelMajorWorkgroups(1024, geometry, 65535),
        CodegenError,
        undefined,
        `${pixels}`,
      );
    }
  });

  it("derives lanes = 256 / pixels and blockWords = max(16, lanes) into the key and the stage", () => {
    const expected: readonly (readonly [number, number, number])[] = [
      [4, 64, 64],
      [8, 32, 32],
      [16, 16, 16],
      [32, 8, 16],
      [64, 4, 16],
    ];
    for (const [pixels, lanes, blockWords] of expected) {
      assertEquals(
        quantizeRowsChannelMajorKey({ pixels }),
        `quantize_rows:v1:f32>i8:pertoken:wg256:cm:p${pixels}l${lanes}b${blockWords}`,
      );
      const wgsl = quantizeRowsChannelMajorWgsl({ pixels });
      assert(wgsl.includes(`let px = lid % ${pixels}u;`), `p${pixels}: pixel index`);
      assert(wgsl.includes(`c = c + ${lanes}u;`), `p${pixels}: lane stride`);
      assert(
        wgsl.includes(`var<workgroup> staged: array<u32, ${pixels * blockWords}>;`),
        `p${pixels}: stage size`,
      );
      assert(wgsl.includes(`w0 = w0 + ${blockWords}u;`), `p${pixels}: word block step`);
    }
  });
});

/**
 * The barriers of the channel-major quantiser and the shared-memory accesses around them, in order.
 * Lane `l` of pixel `px` folds the partials the other lanes wrote, and the stage written by one
 * thread is drained by another, so a barrier MUST separate the partials store from their fold, the
 * stage fill from its drain, and the drain from the next word block's fill (or the next pixel
 * block's partials store). A value comparison can miss a dropped one when the threads involved
 * happen to sit in one warp, so the order is checked on the generated text.
 */
const channelMajorBarrierSequence = (wgsl: string): readonly string[] =>
  wgsl.split("\n").flatMap((line) => {
    const statement = line.trim();
    if (statement === "workgroupBarrier();") return ["barrier"];
    if (statement === "partial[lid] = acc;") return ["partials store"];
    if (statement === "var amax = partial[px];") return ["partials fold"];
    if (statement.startsWith("staged[px * ")) return ["stage fill"];
    if (statement === "xq[wp * quads + wq] = staged[t];") return ["stage drain"];
    if (statement.startsWith("w0 = w0 + ")) return ["next word block"];
    return [];
  });

test("quantize_rows channel-major: a barrier follows the partials store, the stage fill and every word block", () => {
  const expected = [
    "partials store",
    "barrier",
    "partials fold",
    "stage fill",
    "barrier",
    "stage drain",
    "barrier",
    "next word block",
  ];
  for (const pixels of [4, 8, 16, 32, 64]) {
    assertEquals(
      channelMajorBarrierSequence(quantizeRowsChannelMajorWgsl({ pixels })),
      expected,
      `p${pixels}`,
    );
  }
  // The check fails when any one of the three barriers is dropped.
  const wgsl = quantizeRowsChannelMajorWgsl(CONV3D_I8A8_QUANTIZE_GEOMETRY);
  for (const [name, withBarrier, without] of [
    [
      "after the partials store",
      "    partial[lid] = acc;\n    workgroupBarrier();\n",
      "    partial[lid] = acc;\n",
    ],
    [
      "after the stage fill",
      "      }\n      workgroupBarrier();\n      var t = lid;\n",
      "      }\n      var t = lid;\n",
    ],
    [
      "at the end of the word block",
      "      workgroupBarrier();\n      w0 = w0 + ",
      "      w0 = w0 + ",
    ],
  ] as const) {
    assertEquals(wgsl.split(withBarrier).length, 2, `${name}: the layout moved (no fault)`);
    assertNotEquals(
      channelMajorBarrierSequence(wgsl.replace(withBarrier, without)),
      expected,
      `dropping the barrier ${name} is not detected`,
    );
  }
});

// MUST: the naive log(1 + x) loses its significant digits for |x| ≪ 1. The output pins that the
// corrected formula is in place (the numeric measurement is the CPU reference comparison in
// tests/gpu_ops_test.ts).
test("log1p injects a helper that switches between a series and log, and it does not leak into other ops", () => {
  const log1p = elementwiseWgsl({ op: "log1p", rank: 1, dtype: "f32" });
  assertEquals(log1p.includes("fn log1p_series(x: f32) -> f32 {"), true);
  // The Horner form of the coefficients 1/k (the first 3 terms and the last term) and the switch
  // point
  assertEquals(log1p.includes("let series = x * (1.0 - x * (0.5 - x * (0.3333333333333333"), true);
  assertEquals(log1p.includes("x * (0.1)"), true);
  assertEquals(log1p.includes("return select(log(1.0 + x), series, abs(x) < 0.25);"), true);
  assertEquals(log1p.includes("out[i] = log1p_series(v0);"), true);
  // The helper goes only into the ops that use it (the same discipline as erf / sigmoid)
  assertEquals(log1p.includes("fn erf_approx"), false);
  assertEquals(elementwiseWgsl({ op: "log", rank: 1, dtype: "f32" }).includes("log1p"), false);
  assertEquals(elementwiseWgsl({ op: "gelu", rank: 1, dtype: "f32" }).includes("log1p"), false);
});

// MUST: the direction of accumulation can only be detected by value (neither the shape nor the
// element count changes). On the output side, the order "add the previous element, then write" is
// pinned.
test("cumsum is a prefix sum with 1 invocation = 1 row, looping over rows grid-stride", () => {
  assertEquals(CUMSUM_WGSL.includes("acc = acc + x[base + j];"), true);
  assertEquals(CUMSUM_WGSL.includes("out[base + j] = acc;"), true);
  // The row loop is grid-stride (the row count can exceed the 1-D dispatch limit)
  assertEquals(CUMSUM_WGSL.includes("var row = gid.x;"), true);
  assertEquals(CUMSUM_WGSL.includes("row = row + stride;"), true);
  assertEquals([...cumsumParams(7, 3)], [7, 3, 0, 0]);
  assertEquals(cumsumParams(7, 3).byteLength, 16);
  assertThrows(() => cumsumParams(-1, 3), CodegenError);
  assertThrows(() => cumsumParams(1.5, 3), CodegenError);
});

/**
 * argmax (ADR 0068 decision 2). It has 3 contracts that **cannot be detected by value**, so they
 * are pinned from the output side:
 *
 * 1. **The tie-break direction** — rewriting `ib < ia` (smallest index) to `>` still matches the
 *    shape and the dtype. It also checks that both children of the tree use the same predicate (if
 *    only one had the opposite direction, "the winner changes with the tree's shape" = the result
 *    would vary with the dispatch count).
 * 2. **The identity is −inf** — it has not gone back to a finite sentinel (the reduce family's
 *    `-F32_MAX`). The bit pattern is carried in the 3rd word of params (avoiding a const-expression
 *    inf — the same discipline as softmax).
 * 3. **The NaN test is by bit pattern** — going back to a plain comparison makes NaN lose silently,
 *    an inconsistency within the family where `amax` gives NaN and `argmax` another element.
 */
test("argmax has a comparison that keeps the smallest index, a −inf identity and a bit-pattern NaN test", () => {
  assertEquals(ARGMAX_KEY, `argmax:v2:f32>i32:lastdim:minindex:wg${ARGMAX_WORKGROUP_SIZE}`);
  assertEquals(ARGMAX_WORKGROUP_SIZE, 256);
  // ① Tie-break: on equal values the smaller index wins (one predicate of lexicographic order)
  assertEquals(ARGMAX_WGSL.includes("return kb > ka || (kb == ka && ib < ia);"), true);
  // The predicate is defined in one place, and both the scan and the tree call the same function
  // (structure rules out a swapped direction)
  assertEquals(ARGMAX_WGSL.includes("fn rank_key_beats(kb: u32, ib: u32, ka: u32, ia: u32)"), true);
  assertEquals(ARGMAX_WGSL.includes("if (rank_key_beats(v, i, best, best_at)) {"), true);
  assertEquals(
    ARGMAX_WGSL.includes(
      "if (rank_key_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
    ),
    true,
  );
  // ② The identity is −inf (via params) + the index sentinel is dim (the reason an all −inf row
  // gives index 0)
  assertEquals(ARGMAX_WGSL.includes("let neg_inf = f32_rank_key(params.neg_inf);"), true);
  assertEquals(ARGMAX_WGSL.includes("var best = neg_inf;"), true);
  assertEquals(ARGMAX_WGSL.includes("var best_at = dim;"), true);
  // MUST: the reduce family's finite sentinel has not slipped in
  assertEquals(ARGMAX_WGSL.includes("3.402823466e38"), false);
  // ③ NaN becomes the largest common key, and ±0 ties (on the same key, the smallest index)
  assertEquals(ARGMAX_WGSL.includes("let magnitude = bits & 0x7fffffffu;"), true);
  assertEquals(
    ARGMAX_WGSL.includes("return select(ordered, 0xffffffffu, magnitude > 0x7f800000u);"),
    true,
  );
  assertEquals(ARGMAX_WGSL.includes("let zeroed = select(bits, 0u, magnitude == 0u);"), true);
  // The output is an i32 index (written as f32 it would round to the neighbouring token for
  // vocabularies beyond 2^24)
  assertEquals(ARGMAX_WGSL.includes("read_write> out: array<i32>;"), true);
  assertEquals(ARGMAX_WGSL.includes("out[row] = i32(scratch_index[0u]);"), true);
  // The row loop is grid-stride (advancing by workgroup_id × num_workgroups, as in the row reduce)
  assertEquals(ARGMAX_WGSL.includes("var row = wid.x;"), true);
  assertEquals(ARGMAX_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals([...argmaxParams(7, 3)], [7, 3, ARGMAX_NEG_INF_BITS, 0]);
  assertEquals(argmaxParams(7, 3).byteLength, 16, "16-byte alignment of the uniform struct");
  assertThrows(() => argmaxParams(7, 0), CodegenError, "dim must be a positive integer");
  assertThrows(() => argmaxParams(-1, 3), CodegenError);
  assertThrows(() => argmaxParams(7, 1.5), CodegenError);
});

/**
 * topk (ADR 0068 decision 3). Pins from the output side the contracts that **cannot be detected
 * by value**:
 *
 * 1. **The two-phase form** (lane-local top-k → a k-round tournament merge) — neither an argsort
 *    over the whole vocabulary nor "k row reduces". The latter reads the row k times, the costly
 *    side the ADR avoids.
 * 2. **The tie-break direction and the shared predicate** — rewriting `ib < ia` (smallest index)
 *    to `>` still matches the shape and the dtype. The scan, the tree and the cross-round step
 *    calling **the same single predicate** is why the order is "values descending, ties by index
 *    ascending" regardless of the combination order.
 * 3. **The identity is −inf + the sentinel index = dim** — it has not gone back to a finite
 *    sentinel (the reduce family's `-F32_MAX`). The bit pattern is carried in the 3rd word of
 *    params.
 * 4. **The NaN test is by bit pattern** — going back to a plain comparison makes NaN lose silently.
 * 5. **Only the owner of the winning element advances its cursor** (`won % W == lid`) — if this
 *    breaks, the same element comes out k times / another element is skipped, while the shape and
 *    the dtype still match.
 * 6. **The implementation limit of k is decided statically from workgroup storage** (it does not
 *    degrade).
 */
test("topk is two-phase (lane-local top-k → tournament merge) and shares a predicate that keeps the smallest index", () => {
  assertEquals(TOPK_WORKGROUP_SIZE, 32);
  assertEquals(topkKey(4), `topk:v2:f32+i32:lastdim:desc:minindex:k4:wg${TOPK_WORKGROUP_SIZE}`);
  // The key includes k (a separate WGSL per k)
  assertEquals(topkKey(1) === topkKey(2), false);
  const wgsl = topkWgsl(4);
  // ① The 2 outputs (the f32 value bits + an i32 index) are written to separate bindings
  assertEquals(wgsl.includes("@binding(2) var<storage, read_write> values: array<u32>;"), true);
  assertEquals(wgsl.includes("@binding(3) var<storage, read_write> indices: array<i32>;"), true);
  // ① Phase 1 = per-lane candidate blocks (k words × W lanes) / phase 2 = the head of a W-way
  // tournament
  assertEquals(wgsl.includes("var<workgroup> cand_value: array<u32, 128>;"), true);
  assertEquals(wgsl.includes("var<workgroup> cand_index: array<u32, 128>;"), true);
  assertEquals(wgsl.includes("var<workgroup> head_value: array<u32, 32>;"), true);
  assertEquals(wgsl.includes("var<workgroup> head_index: array<u32, 32>;"), true);
  // ① The row is read only once (there is one scan loop; the k loop exists only on the merge side)
  assertEquals((wgsl.match(/let v = f32_rank_key\(x\[base \+ i\]\);/g) ?? []).length, 1);
  assertEquals(wgsl.includes("let block = lid * 4u;"), true);
  // ② The predicate is defined in one place, and both the scan and the tree call the same function
  // (structure rules out a swapped direction)
  assertEquals(wgsl.includes("fn rank_key_beats(kb: u32, ib: u32, ka: u32, ia: u32)"), true);
  assertEquals(wgsl.includes("return kb > ka || (kb == ka && ib < ia);"), true);
  assertEquals(
    wgsl.includes("if (rank_key_beats(v, i, cand_value[block + 3u], cand_index[block + 3u])) {"),
    true,
  );
  assertEquals(
    wgsl.includes(
      "while (s > 0u && rank_key_beats(v, i, cand_value[block + s - 1u], cand_index[block + s - 1u])) {",
    ),
    true,
  );
  assertEquals(
    wgsl.includes("if (rank_key_beats(other, other_at, head_value[lid], head_index[lid])) {"),
    true,
  );
  // ③ The identity is −inf (via params) + the sentinel index = dim (the reason the answer is
  // defined even for an all −inf row)
  assertEquals(wgsl.includes("let neg_inf = f32_rank_key(params.neg_inf);"), true);
  assertEquals(wgsl.includes("cand_value[block + s] = neg_inf;"), true);
  assertEquals(wgsl.includes("cand_index[block + s] = dim;"), true);
  // MUST: the reduce family's finite sentinel has not slipped in
  assertEquals(wgsl.includes("3.402823466e38"), false);
  // ④ NaN maps to the largest common key, ±0 to a tie (the same body as argmax)
  assertEquals(wgsl.includes("let magnitude = bits & 0x7fffffffu;"), true);
  assertEquals(
    wgsl.includes("return select(ordered, 0xffffffffu, magnitude > 0x7f800000u);"),
    true,
  );
  assertEquals(wgsl.includes("let zeroed = select(bits, 0u, magnitude == 0u);"), true);
  assertEquals(
    ARGMAX_WGSL.includes("return kb > ka || (kb == ka && ib < ia);"),
    true,
    "the same predicate body as argmax (the premise of k=1 matching argmax)",
  );
  // ⑤ The output is k words per row, and only the owner of the winning element advances its cursor
  assertEquals(wgsl.includes("values[row * 4u + r] = x[base + won];"), true);
  assertEquals(wgsl.includes("indices[row * 4u + r] = i32(won);"), true);
  assertEquals(wgsl.includes("if (won % 32u == lid) {"), true);
  assertEquals(wgsl.includes("cursor = cursor + 1u;"), true);
  // The row loop is grid-stride (advancing by workgroup_id × num_workgroups, as in argmax / the row
  // reduce)
  assertEquals(wgsl.includes("var row = wid.x;"), true);
  assertEquals(wgsl.includes("row = row + nwg.x;"), true);
  // The degenerate form k=1: the block's tail (weakest) and head are the same word, and the
  // insertion loop never runs
  assertEquals(
    topkWgsl(1).includes(
      "if (rank_key_beats(v, i, cand_value[block + 0u], cand_index[block + 0u])) {",
    ),
    true,
  );

  // ⑥ The implementation limit is decided statically from workgroup storage (8·W·(k+1) bytes)
  assertEquals(topkWorkgroupStorageBytes(1), 8 * 32 * 2);
  assertEquals(topkWorkgroupStorageBytes(63), 16384);
  assertEquals(topkMaxK(16384), TOPK_CORE_LIMIT_MAX_K);
  assertEquals(TOPK_CORE_LIMIT_MAX_K, 63);
  // The limit moves with the device limit (the requested limit becomes the limit as is)
  assertEquals(topkMaxK(32768), 127);
  assertEquals(topkMaxK(1024), 3);
  assertTopkK(63, 16384, "t");
  // MUST: an excess is not degraded but fails loudly **with the limit value** (a form where the
  // user can tell from the diagnostic alone how far to lower k)
  assertThrows(
    () => assertTopkK(64, 16384, "t"),
    CodegenError,
    "exceeds the implementation limit 63",
  );
  assertThrows(() => assertTopkK(4, 1024, "t"), CodegenError, "exceeds the implementation limit 3");
  assertThrows(() => assertTopkK(0, 16384, "t"), CodegenError, "k must be a positive integer");
  assertThrows(() => assertTopkK(1.5, 16384, "t"), CodegenError);

  assertEquals([...topkParams(7, 3)], [7, 3, TOPK_NEG_INF_BITS, 0]);
  assertEquals(topkParams(7, 3).byteLength, 16, "16-byte alignment of the uniform struct");
  // MUST: k is not put into params (it is baked into the WGSL — structure forbids keeping it in two
  // places)
  assertEquals(topkParams(7, 3).length, 4);
  assertThrows(() => topkParams(7, 0), CodegenError, "dim must be a positive integer");
  assertThrows(() => topkParams(-1, 3), CodegenError);
});

test("half-split RoPE rounds the products through workgroup u32 and adds them after a uniform barrier", () => {
  const ROPE_WGSL = ropeWgsl("bhsd");
  assertEquals(ROPE_KEY, `rope:v1:half:f32:wg${ROPE_WORKGROUP_SIZE}`);
  assertEquals(ROPE_WORKGROUP_SIZE, 256);
  assertEquals(ROPE_WGSL.includes("var<workgroup> products: array<vec2<u32>, 256>;"), true);
  assertEquals(
    ROPE_WGSL.includes(
      "rotated_bits = bitcast<u32>(x[row_base + d + params.half_dim]) ^ 0x80000000u;",
    ),
    true,
  );
  assertEquals(ROPE_WGSL.includes("products[lid] = vec2<u32>(direct_bits, cross_bits);"), true);
  assertEquals(ROPE_WGSL.match(/workgroupBarrier\(\);/g)?.length, 2);
  assertEquals(ROPE_WGSL.includes("out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);"), true);
  assertEquals([...ropeParams(2 * 3 * 128, 3, 128)], [768, 3, 128, 64]);
  assertEquals(ropeParams(768, 3, 128).byteLength, 16);
  assertThrows(() => ropeParams(768, 0, 128), CodegenError, "sequence");
  assertThrows(() => ropeParams(768, 3, 127), CodegenError, "headDim");
  assertThrows(() => ropeParams(767, 3, 128), CodegenError, "whole number of rows");
  assertThrows(() => ropeParams(0x1_0000_0000, 1, 128), CodegenError, "u32");
});

test("NCHW 2x upsample writes 2x2 per input element and checks the row bounds in params", () => {
  assertEquals(UPSAMPLE_2X_KEY, `upsample2x:v1:nchw:f32:wg${UPSAMPLE_2X_WORKGROUP_SIZE}`);
  assertEquals(UPSAMPLE_2X_WORKGROUP_SIZE, 256);
  assertEquals(
    UPSAMPLE_2X_WGSL.includes("let base = 2u * row * params.out_width + 2u * col;"),
    true,
  );
  // Bitwise replication, so both input and output are u32 views (NaN payloads / subnormals / ±0 are
  // not normalized)
  assertEquals(UPSAMPLE_2X_WGSL.includes("x: array<u32>"), true);
  assertEquals(UPSAMPLE_2X_WGSL.includes("out: array<u32>"), true);
  assertEquals(UPSAMPLE_2X_WGSL.includes("out[base + params.out_width + 1u] = value;"), true);
  assertEquals([...upsample2xParams(24, 3)], [24, 3, 6, 0]);
  assertEquals(upsample2xParams(24, 3).byteLength, 16);
  assertThrows(() => upsample2xParams(-1, 3), CodegenError, "u32");
  assertThrows(() => upsample2xParams(24, 0), CodegenError, "width");
  assertThrows(() => upsample2xParams(25, 3), CodegenError, "whole number of rows");
  assertThrows(() => upsample2xParams(0x4000_0000, 1), CodegenError, "4 * n");
  assertThrows(() => upsample2xParams(0, 0x8000_0000), CodegenError, "2 * width");
});

test("bilinear resample has 1 output element = 1 thread and decides the last tap by an integer comparison", () => {
  assertEquals(
    UPSAMPLE_BILINEAR2D_KEY,
    `upsample_bilinear2d:v2:nchw:f32:align_corners:wg${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}`,
  );
  assertEquals(UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE, 256);
  // MUST: the host divides the scale and carries it in params (WGSL's f32 division is allowed up to
  // 2.5 ULP, and when the last source coordinate falls slightly below the input's last index, the
  // exact match at the edge breaks in a backend-dependent way). The only division left in the
  // shader is the u32 index decomposition = having not a single f32 division is that declaration.
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("  scale_h: f32,"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("params.scale_h * f32(oy)"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("params.scale_w * f32(ox)"), true);
  assertEquals(/f32\([^)]*\) *\//.test(UPSAMPLE_BILINEAR2D_WGSL), false, "an f32 division remains");
  // MUST: the last-tap special case is an **integer comparison** (f32 min/max swallow NaN — ADR
  // 0020)
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("if (y0 + 1u < params.in_h) {"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("if (x0 + 1u < params.in_w) {"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("clamp("), false, "no clamp on coordinates");
  // The expression tree has H outside and W inside (the same form as torch's CUDA kernel — moving
  // the parentheses changes the rounding sequence)
  assertEquals(
    UPSAMPLE_BILINEAR2D_WGSL.includes("out[i] = ly0 * (lx0 * x[row0 + x0] + lx1 * x[row0 + x1])"),
    true,
  );
  const params = upsampleBilinear2dParams(2 * 3 * 7 * 9, 4, 5, 7, 9);
  assertEquals([...params.slice(0, 5)], [378, 4, 5, 7, 9]);
  // scale = (in − 1) / (out − 1) in f32 (both 4→7 and 5→9 are exactly 0.5)
  assertEquals([...new Float32Array(params.buffer).slice(5, 7)], [0.5, 0.5]);
  assertEquals(params[7], 0, "the reserved word is 0");
  // An axis of output length 1 has scale 0 (the special case in torch's area_pixel_compute_scale)
  assertEquals(
    [...new Float32Array(upsampleBilinear2dParams(2, 6, 7, 1, 1).buffer).slice(5, 7)],
    [0, 0],
  );
  // An axis of input length 1 has scale 0 (the form where the last-tap special case fires for every
  // output)
  assertEquals(
    [...new Float32Array(upsampleBilinear2dParams(12, 1, 6, 4, 3).buffer).slice(5, 7)],
    [0, 2.5],
  );
  // The uniform struct is aligned to 16 bytes (5 u32 words + 2 f32 words + 1 reserved word = 32
  // bytes)
  assertEquals(params.byteLength, 32);
  assertThrows(() => upsampleBilinear2dParams(-1, 4, 5, 7, 9), CodegenError, "u32");
  // A spatial axis of 0 makes `in_size - 1u` on the WGSL side wrap around (a silent out-of-range
  // read)
  assertThrows(() => upsampleBilinear2dParams(63, 0, 5, 7, 9), CodegenError, "heightIn");
  assertThrows(() => upsampleBilinear2dParams(63, 4, 0, 7, 9), CodegenError, "widthIn");
  assertThrows(() => upsampleBilinear2dParams(0, 4, 5, 0, 9), CodegenError, "heightOut");
  assertThrows(() => upsampleBilinear2dParams(0, 4, 5, 7, 0), CodegenError, "widthOut");
  // An n that is not a whole multiple of the output plane throws off the plane's index
  // decomposition
  assertThrows(() => upsampleBilinear2dParams(64, 4, 5, 7, 9), CodegenError, "integer multiple");
});

test("DCNv2 writes the range test in the positive form and separates and propagates NaN by bit pattern", () => {
  assertEquals(
    DEFORM_CONV2D_KEY,
    `deform_conv2d:v2:nchw:f32:dcnv2:wg${DEFORM_CONV2D_WORKGROUP_SIZE}`,
  );
  assertEquals(DEFORM_CONV2D_WORKGROUP_SIZE, 256);
  // MUST: the range test is in the positive form (NaN falls to the false side). Rewriting it in the
  // negative form (`<= -1 ||`) makes NaN fall "in range" and go straight to the undefined
  // i32(floor(NaN)).
  assertEquals(
    DEFORM_CONV2D_WGSL.includes(
      "let inside = sy > -1.0 && sy < f32(dims.height_in)\n    && sx > -1.0 && sx < f32(dims.width_in);",
    ),
    true,
  );
  // MUST: NaN is tested by bit pattern and **propagated** (the positive form alone drops it to a
  // zero contribution). The test injects the one canonical shared fragment (ADR 0020 — a copy would
  // be left behind when the canonical one is revised).
  assertEquals(
    DEFORM_CONV2D_WGSL.includes(IS_NAN_BITS_WGSL),
    true,
    "the shared fragment is not injected",
  );
  assertEquals(DEFORM_CONV2D_WGSL.includes("if (is_nan_bits(sy) || is_nan_bits(sx)) {"), true);
  assertEquals(DEFORM_CONV2D_WGSL.includes("fn is_nan("), false, "a copy of the NaN test remains");
  assertEquals(DEFORM_CONV2D_WGSL.includes("return bitcast<f32>(dims.oob);"), true);
  // MUST: the NaN bit pattern is carried in params (some implementations may turn a constant
  // expression `bitcast<f32>(0x…)` into a shader creation error as "the const-expression is NaN" —
  // the same discipline as gather / embedding).
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN is built with a constant expression (carry it via params)",
  );
  // MUST: no clamp on coordinates (out of bounds is zero-filled, not clamped)
  assertEquals(DEFORM_CONV2D_WGSL.includes("clamp("), false, "no clamp on coordinates");
  // MUST: even channels = y / odd = x
  assertEquals(DEFORM_CONV2D_WGSL.includes("offsets[offset_base + 2u * tap * plane_out]"), true);
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("offsets[offset_base + (2u * tap + 1u) * plane_out]"),
    true,
  );
  // MUST: the mask comes after interpolation and before the weight (`(m · v) · w`), and the
  // reduction order is the same as conv2d, (ic, kh, kw)
  assertEquals(
    DEFORM_CONV2D_WGSL.includes(
      "acc = acc + (m * deform_sample(x_base, sy, sx)) * w[w_base + tap];",
    ),
    true,
  );
  assertEquals(
    DEFORM_CONV2D_WGSL.indexOf("for (var ic = 0u;") <
      DEFORM_CONV2D_WGSL.indexOf("for (var kh = 0u;"),
    true,
    "the ic loop is outside kh (the same reduction order as conv2d)",
  );
  const params = deformConv2dParams({
    batch: 1,
    channelsIn: 3,
    channelsOut: 5,
    heightIn: 4,
    widthIn: 5,
    heightOut: 4,
    widthOut: 4,
    kernelH: 3,
    kernelW: 2,
    paddingH: 1,
    paddingW: 0,
  });
  assertEquals([...params], [80, 1, 3, 5, 4, 5, 4, 4, 3, 2, 1, 0, DEFORM_CONV2D_OOB_BITS, 0, 0, 0]);
  // The uniform struct is aligned to 16 bytes (13 words of content get 16 words = 64 bytes)
  assertEquals(params.byteLength, 64);
  const dims = {
    batch: 1,
    channelsIn: 3,
    channelsOut: 5,
    heightIn: 4,
    widthIn: 5,
    heightOut: 4,
    widthOut: 4,
    kernelH: 3,
    kernelW: 2,
    paddingH: 1,
    paddingW: 0,
  };
  assertThrows(
    () => deformConv2dParams({ ...dims, paddingH: -1 }),
    CodegenError,
    "non-negative u32 integer",
  );
  // A spatial length of 0 makes every tap fall into the out-of-range test, a silent wrong value
  // where the output is all bias
  assertThrows(() => deformConv2dParams({ ...dims, heightIn: 0 }), CodegenError, "height_in");
  assertThrows(() => deformConv2dParams({ ...dims, widthIn: 0 }), CodegenError, "width_in");
  assertThrows(() => deformConv2dParams({ ...dims, kernelW: 0 }), CodegenError, "kernel_w");
  assertThrows(() => deformConv2dParams({ ...dims, batch: 0 }), CodegenError, "batch");
});

// The GRU hidden-side scan (ADR 0056). This op's bit-identity check
// (tests/gpu_gru_scan_parity_test.ts) depends entirely on "the update formula is a verbatim copy
// of the decomposed form" and "fma contraction is stopped by rounding barriers", so the output
// side pins those 2 points too (a form that turns red even without a real GPU).
test("gru_scan writes the update formula verbatim from the decomposed form and puts a rounding barrier between mul and add", () => {
  assertEquals(GRU_SCAN_WORKGROUP_SIZE, 256);
  assertEquals(GRU_SCAN_MAX_HIDDEN, 256);
  assertEquals(
    gruScanKey("forward"),
    `gru_scan:v2:f32:forward:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`,
  );
  assertEquals(
    gruScanKey("reverse"),
    `gru_scan:v2:f32:reverse:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`,
  );
  assertNotEquals(gruScanKey("forward"), gruScanKey("reverse"));
  const invalid = "backward" as GruScanDirection;
  assertThrows(() => gruScanKey(invalid), CodegenError, "invalid scan direction");
  assertThrows(() => gruScanWgsl(invalid), CodegenError, "invalid scan direction");

  const forward = gruScanWgsl("forward");
  const reverse = gruScanWgsl("reverse");
  for (const [where, wgsl] of [
    ["forward", forward],
    ["reverse", reverse],
  ] as const) {
    // MUST: the reduction is sequential in ascending k, and its text matches GEMM (`acc = acc + a *
    // b`)
    assertEquals(wgsl.includes("acc_r = acc_r + w_hh[row_r + k] * hk;"), true, where);
    // MUST: the bias is last (copying the conv family's bias-first breaks bit identity with linear)
    assertEquals(
      wgsl.includes("sigmoid_stable((acc_r + b_hh[lid]) + gi[gi_base + lid])"),
      true,
      where,
    );
    // MUST: the candidate gate uses tanh_stable, not the plain tanh (silent NaN under Metal
    // fast-math — v2)
    assertEquals(wgsl.includes("cand = tanh_stable(gi[gi_base + hidden * 2u + lid]"), true, where);
    assertEquals(wgsl.includes("fn tanh_stable(x: f32)"), true, where);
    // MUST: rounding barriers ① (the product of n) and ② (the products of the update formula)
    // remain as workgroup memory round trips
    assertEquals(wgsl.includes("stage[lid] = bitcast<u32>(gh_n * gate_r);"), true, where);
    assertEquals(
      wgsl.includes("stage[lid] = bitcast<u32>((h_prev - cand) * gate_z);"),
      true,
      where,
    );
    assertEquals(wgsl.includes("let h_next = bitcast<f32>(stage[lid]) + cand;"), true, where);
    // MUST: do not rewrite it as `(1 − z)·n + z·h` (equivalent but a different rounding sequence —
    // ADR 0056 decision 3)
    assertEquals(wgsl.includes("1.0 - gate_z"), false, where);
    assertEquals(wgsl.includes("var<workgroup> h_shared: array<f32, 256>;"), true, where);
  }
  // MUST: the 2 directions differ only in the one line of scan order (the output's time index is
  // `t` in both)
  assertEquals(forward.includes("let t = step;"), true);
  assertEquals(reverse.includes("let t = dims.time - 1u - step;"), true);
  assertEquals(
    forward.replaceAll("let t = step;", "").replaceAll("(forward,", "(reverse,"),
    reverse.replaceAll("let t = dims.time - 1u - step;", ""),
    "the 2 directions differ in more than the one line of scan order and the header",
  );

  assertEquals([...gruScanParams({ time: 5, batch: 2, hidden: 128 })], [5, 2, 128, 0]);
  // The uniform struct is aligned to 16 bytes (4 words are allocated even for 3 words of content)
  assertEquals(gruScanParams({ time: 5, batch: 2, hidden: 128 }).byteLength, 16);
  assertThrows(
    () => gruScanParams({ time: -1, batch: 2, hidden: 4 }),
    CodegenError,
    "time must be a non-negative u32 integer",
  );
  assertThrows(
    () => gruScanParams({ time: 5, batch: 2, hidden: 0 }),
    CodegenError,
    "hidden must be a positive integer",
  );
  // MUST: exceeding the limit fails loudly (degrading silently would write h_shared out of range)
  assertThrows(
    () => gruScanParams({ time: 5, batch: 2, hidden: GRU_SCAN_MAX_HIDDEN + 1 }),
    CodegenError,
    "exceeds the limit 256",
  );
});

test("SiLU keeps the f32 storage boundary of sigmoid with u32 workgroup staging and keeps the mul order in the key", () => {
  assertEquals(SILU_WORKGROUP_SIZE, 256);
  assertEquals(siluKey("x-sigmoid"), `silu:v1:x-sigmoid:f32:wg${SILU_WORKGROUP_SIZE}`);
  assertEquals(siluKey("sigmoid-x"), `silu:v1:sigmoid-x:f32:wg${SILU_WORKGROUP_SIZE}`);
  assertNotEquals(siluKey("x-sigmoid"), siluKey("sigmoid-x"));
  const invalid = "foreign-order" as SiluMulOrder;
  assertThrows(() => siluKey(invalid), CodegenError, "invalid input order");
  assertThrows(() => siluWgsl(invalid), CodegenError, "invalid input order");

  const xs = siluWgsl("x-sigmoid");
  const sx = siluWgsl("sigmoid-x");
  for (const [where, wgsl] of [
    ["x-sigmoid", xs],
    ["sigmoid-x", sx],
  ] as const) {
    assertEquals(wgsl.includes("var<workgroup> sigmoid_bits: array<u32, 256>;"), true, where);
    assertEquals(
      wgsl.includes("sigmoid_bits[lid] = bitcast<u32>(sigmoid_stable(x_for_sigmoid));"),
      true,
      where,
    );
    assertEquals(
      wgsl.includes("let sigmoid_after_store = bitcast<f32>(sigmoid_bits[lid]);"),
      true,
      where,
    );
    assertEquals(wgsl.includes("let blocks = params.n / 256u +"), true, where);
    assertEquals(wgsl.includes("select(0u, 1u, params.n % 256u != 0u);"), true, where);
    assertEquals(wgsl.includes("while (block < blocks) {"), true, where);
    assertEquals(wgsl.match(/workgroupBarrier\(\);/g)?.length, 2, where);
    assertEquals(wgsl.includes("let x_for_mul = x[i];"), true, where);
    // Shares the stable formula of the primitive sigmoid. Prevents a regression to the naive
    // exp(-x) form.
    assertEquals(wgsl.includes("let t = exp(-abs(x));"), true, where);
  }
  assertEquals(xs.includes("out[i] = x_for_mul * sigmoid_after_store;"), true);
  assertEquals(sx.includes("out[i] = sigmoid_after_store * x_for_mul;"), true);

  assertEquals([...siluParams(257)], [257, 0, 0, 0]);
  assertEquals(siluParams(0).byteLength, 16);
  assertThrows(() => siluParams(-1), CodegenError, "u32");
  assertThrows(() => siluParams(1.5), CodegenError, "u32");
  assertThrows(() => siluParams(0x1_0000_0000), CodegenError, "u32");
});

test("adaLN shares the normalization body with the plain layer_norm and splits modulation and products into 2 levels with u32 staging", () => {
  assertEquals(ADALN_NORM_WORKGROUP_SIZE, 256);
  // MUST: the same width as the plain layer_norm (since they share the body, moving only one breaks
  // the WGSL — the MUST in src/kernels/adaln-norm.ts). The derivation is exposed to the check.
  assertEquals(ADALN_NORM_WORKGROUP_SIZE, LAYER_NORM_WORKGROUP_SIZE);
  assertEquals(ADALN_NORM_KEY, `adaln_norm:v1:lastdim:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`);
  assertNotEquals(ADALN_NORM_KEY, LAYER_NORM_KEY);

  // MUST: the row statistics (2 passes / population variance) and affine are the **same string**.
  // Writing them separately would split the reduction order between the plain one and the fused
  // version and break the bit match on finite values.
  assertEquals(
    LAYER_NORM_WGSL.includes(LAYER_NORM_ROW_STATS_WGSL),
    true,
    "shared on the plain side",
  );
  assertEquals(
    ADALN_NORM_WGSL.includes(LAYER_NORM_ROW_STATS_WGSL),
    true,
    "shared on the fused side",
  );
  assertEquals(
    LAYER_NORM_WGSL.includes(`out[base + o] = ${LAYER_NORM_AFFINE_WGSL};`),
    true,
    "affine on the plain side",
  );
  assertEquals(ADALN_NORM_WGSL.includes(`(${LAYER_NORM_AFFINE_WGSL}) * modulation`), true);

  // 2 levels of rounding barriers (`1 + scale` and `t * s`). The buffers are 6 in + 1 out.
  assertEquals(ADALN_NORM_WGSL.includes("var<workgroup> staged: array<u32, 256>;"), true);
  assertEquals(
    ADALN_NORM_WGSL.includes("staged[lid] = bitcast<u32>(scale_vec[o] + one[0u]);"),
    true,
  );
  assertEquals(ADALN_NORM_WGSL.includes("let modulation = bitcast<f32>(staged[lid]);"), true);
  assertEquals(
    ADALN_NORM_WGSL.includes("out[base + o] = bitcast<f32>(staged[lid]) + shift[o];"),
    true,
  );
  assertEquals(ADALN_NORM_WGSL.includes("@binding(7) var<storage, read_write> out"), true);
  assertEquals(ADALN_NORM_WGSL.includes("@binding(8)"), false, "at most 7 storage bindings");
  // The output is a block loop (workgroup-uniform). With the `o = lid` while loop, no barrier can
  // be placed.
  assertEquals(ADALN_NORM_WGSL.includes("while (block < blocks) {"), true);

  // params use the same generator as the plain layer_norm (rows / dim / eps).
  assertEquals([...adalnNormParams(4, 8, 1e-6)], [...layerNormParams(4, 8, 1e-6)]);
  assertEquals(adalnNormParams(4, 8, 1e-6).byteLength, 16);
  assertThrows(() => adalnNormParams(4, 0, 1e-6), CodegenError, "dim");
  assertThrows(() => adalnNormParams(4, 8, 0), CodegenError, "eps");
});

test("elementwise params put scalar attrs at the end as f32 bit patterns", () => {
  // Output [2,3] (rank2) + 1 input → the scalars follow the 2 words of dims and the 2 words of
  // strides
  const clamp: ElementwiseSpec = { op: "clamp", rank: 2, dtype: "f32" };
  const params = elementwiseParams(clamp, [2, 3], [[2, 3]], [-1.5, 2.25]);
  assertEquals(params.length, 1 + 2 + 2 + 2);
  assertEquals([...params.slice(0, 5)], [6, 2, 3, 3, 1]);
  const floats = new Float32Array(params.buffer);
  assertEquals(floats[5], -1.5);
  assertEquals(floats[6], 2.25);
  // The default is 0 scalars (the params layout of existing ops does not change)
  assertEquals(elementwiseParams({ op: "neg", rank: 2, dtype: "f32" }, [2, 3], [[2, 3]]).length, 5);
  assertThrows(
    () => elementwiseParams({ op: "clamp_min", rank: 1, dtype: "f32" }, [2], [[2]], [Number.NaN]),
    CodegenError,
  );
  // MUST: the finiteness test is as f32 (the destination is an f32 word, so 1e39, finite in f64,
  // silently becomes +Inf)
  assertThrows(
    () => elementwiseParams({ op: "clamp_min", rank: 1, dtype: "f32" }, [2], [[2]], [1e39]),
    CodegenError,
    "is not finite as f32",
  );
  // A value that rounds normally to f32 passes (the counterpart showing the check is not too wide)
  assertEquals(
    new Float32Array(
      elementwiseParams({ op: "clamp_min", rank: 1, dtype: "f32" }, [2], [[2]], [0.1]).buffer,
    )[3],
    Math.fround(0.1),
  );
  // MUST: the input count is matched against the arity at generation. A mismatch shifts the write
  // positions of `1 + rank + arity·rank + index` and the scalars on the WGSL side, a silent wrong
  // value from an out-of-range read.
  assertThrows(
    () => elementwiseParams({ op: "clamp", rank: 1, dtype: "f32" }, [4], [[4], [4]], [0, 1]),
    CodegenError,
    "arity",
  );
  // The scalar count is checked at the same single place (an excess used to be dropped silently)
  assertThrows(() => elementwiseParams(clamp, [2, 3], [[2, 3]], [-1.5]), CodegenError, "scalars");
  // The rank is also canonical on the spec side (not derived from the output shape alone)
  assertThrows(() => elementwiseParams(clamp, [6], [[6]], [-1.5, 2.25]), CodegenError, "rank");
});

test("elementwise params set the stride of a right-aligned broadcast to 0", () => {
  // For an output [2,3,4], in0=[2,3,4] (contiguous), in1=[3,1] (right-aligned to [1,3,1])
  const params = elementwiseParams(
    { op: "add", rank: 3, dtype: "f32" },
    [2, 3, 4],
    [
      [2, 3, 4],
      [3, 1],
    ],
  );
  assertEquals([...params.slice(0, 4)], [24, 2, 3, 4]);
  assertEquals([...params.slice(4, 7)], [12, 4, 1]);
  assertEquals([...params.slice(7, 10)], [0, 1, 0]);
});

test("elementwise params absorb a scalar input by right alignment", () => {
  const params = elementwiseParams({ op: "add", rank: 1, dtype: "f32" }, [5], [[5], [1]]);
  assertEquals([...params], [5, 5, 1, 0]);
});

/**
 * MUST: each input axis is either "equal to the output" or "1". The stride construction assumes
 * "contiguous stride unless 1", so passing a non-broadcastable pair silently produces params that
 * point outside **the input buffer** (WebGPU's bounded access returns a clamped value without an
 * exception). This test is the only detector for when the contract layer (broadcastShapes in
 * src/ops/shapes.ts) is skipped.
 */
test("elementwise params reject axis lengths that cannot broadcast", () => {
  assertThrows(
    () => elementwiseParams({ op: "neg", rank: 1, dtype: "f32" }, [5], [[3]]),
    CodegenError,
    "matches neither output 5 nor 1",
  );
  assertThrows(
    () =>
      elementwiseParams(
        { op: "add", rank: 2, dtype: "f32" },
        [2, 3],
        [
          [2, 3],
          [2, 4],
        ],
      ),
    CodegenError,
    "matches neither output 3 nor 1",
  );
  // A conforming shape does not move a single word (the counterpart showing the check is not too
  // wide)
  assertEquals(
    [
      ...elementwiseParams(
        { op: "add", rank: 2, dtype: "f32" },
        [2, 3],
        [
          [2, 3],
          [1, 3],
        ],
      ),
    ],
    [6, 2, 3, 3, 1, 0, 1],
  );
});

test("params passed as uniform meet 16-byte alignment", () => {
  // In WGSL's uniform address space a struct's alignment is 16 bytes
  assertEquals(reduceParams(7, 3).byteLength, 16);
  assertEquals(matmulParams(2, 3, 4).byteLength, 16);
  assertEquals(bmmParams(2, 3, 4, 1).byteLength, 16);
  assertEquals(gatherParams(12, 3, 9).byteLength, 16);
  assertEquals(padParams(2, 3, 1, 2).byteLength, 16);
  // flip's Params are 3 words, but 16 bytes are allocated (the uniform struct's alignment)
  assertEquals(flipParams(2, 3, 4).byteLength, 16);
  assertEquals(axisReduceParams(35, 384, 5).byteLength, 16);
  assertEquals([...reduceParams(7, 3).slice(0, 2)], [7, 3]);
  // The axis variant has the 3 words {out_count, axis_len, inner} (the tail is padding for
  // alignment)
  assertEquals([...axisReduceParams(35, 384, 5)], [35, 384, 5, 0]);
  assertEquals([...matmulParams(2, 3, 4).slice(0, 3)], [2, 3, 4]);
  // bmm also has 3 words {m,n,k}. The batch count is derived from wid.z, so it is not in the
  // uniform (it would be a dead field the WGSL never reads)
  assertEquals([...bmmParams(2, 3, 4, 1)], [2, 3, 4, 0]);
  // The tail is the NaN bit pattern written to out-of-range indices (WGSL has no NaN literal, so it
  // is carried in params)
  assertEquals([...gatherParams(12, 3, 9)], [12, 3, 9, GATHER_OOB_BITS]);
  assertEquals(Number.isNaN(new Float32Array(new Uint32Array([GATHER_OOB_BITS]).buffer)[0]), true);
});

// MUST: bmm is a **separate kernel with a separate key** from matmul (the skeleton is shared, but
// the batch-axis fragments differ).
test("bmm is a separate kernel from matmul and has a batch-axis offset", () => {
  for (const v4 of [false, true]) {
    const where = `v4=${v4}`;
    assertNotEquals(bmmKey(v4), matmulKey(v4), where);
    assertNotEquals(bmmWgsl(v4), matmulWgsl(v4), where);
    // The batch comes from the workgroup's z axis, and all 3 buffers are read and written at an
    // offset of one matrix.
    // MUST: with v4 the base is also **in quads** (`m * k4`) — built in elements, batch ≥ 2 would
    // read and write the neighbouring batch, a wrong value without an exception (the GPU case at
    // the tile boundary is the only detector).
    const unit = (dim: string) => (v4 ? `${dim}4` : `dims.${dim}`);
    assertEquals(bmmWgsl(v4).includes(`let abase = wid.z * dims.m * ${unit("k")};`), true, where);
    assertEquals(bmmWgsl(v4).includes(`let bbase = wid.z * dims.k * ${unit("n")};`), true, where);
    assertEquals(bmmWgsl(v4).includes(`let cbase = wid.z * dims.m * ${unit("n")};`), true, where);
    // All 3 buffers are accessed through base (blocks the mistake of only one going back to a bare
    // index). The index suffix is the number of the fill slot / output row (geometries where one
    // thread holds several rows — src/kernels/gemm-geometry.ts)
    assertEquals(bmmWgsl(v4).includes("abase + arow0 *"), true, where);
    assertEquals(bmmWgsl(v4).includes("bbase + brow0 *"), true, where);
    assertEquals(bmmWgsl(v4).includes("cbase + orow0 *"), true, where);
    // The batch concept has not leaked into the matmul side (no key collision and no change in the
    // output)
    assertEquals(matmulWgsl(v4).includes("wid.z"), false, where);
    assertEquals(matmulWgsl(v4).includes("base + arow"), false, where);
  }
});

/**
 * The output of fused attention (ADR 0023). **Bit identity is the core of the design**, so the 4
 * points that build its premise (the half scale on both q/k sides / the row statistics being
 * verbatim identical to passes ①② of softmax / ③ evaluating `exp(S−m)·inv` per component / the
 * row-statistics index including the batch base) are pinned on the output side.
 */
test("the 3 fused attention kernels generate a form bit-identical to the decomposed path", async () => {
  for (const v4 of [false, true]) {
    const qk = attentionQkWgsl(v4);
    const where = `v4=${v4}`;
    // MUST: the scale goes on **both the q side and the k side** (the half-scale contract). On one
    // side only, the values are off by a factor of √; multiplying once after the inner product
    // changes the rounding sequence and breaks bit identity.
    // MUST: it applies to **every** fill slot (the geometry has one thread fill several columns, so
    // applying it only to slot 0 puts half of the thread's share off by √). The slot count is 2
    // (the default geometry — the geometry test below)
    assertEquals(qk.includes("wv0 = wv0 * dims.scale;"), true, `${where}: k-side scale`);
    assertEquals(qk.includes("wv1 = wv1 * dims.scale;"), true, `${where}: k-side scale slot 1`);
    assertEquals(
      (qk.match(/wv(\d+) = wv\1 \* dims\.scale;/g) ?? []).length,
      qk.split("var wv").length - 1,
      `${where}: the k-side scale is exactly the slot count`,
    );
    assertEquals(
      qk.includes("raw0.x * dims.scale") || qk.includes("q[arow_base0 + ak0] * dims.scale"),
      true,
      `${where}: q-side scale`,
    );
    // scale is the 4th word of the uniform (not baked into the WGSL — pipelines would multiply by
    // value)
    assertEquals(qk.includes("  scale: f32,\n"), true, where);
    // k is read as [N,D] as is (the output-side evidence that the old path's permute(kᵀ) is gone)
    assertEquals(qk.includes("let krow_base0 = bbase + wcol0 * dims.k;"), true, where);
    // MUST: the base of the transposed read is **in elements** even with v4 (in quads it would read
    // at a quarter of the position)
    assertEquals(qk.includes("let bbase = wid.z * dims.n * dims.k;"), true, where);

    const pv = attentionPvWgsl(v4);
    // MUST: the A tile does not materialize P and evaluates `exp(S − m) · inv` as a **per-component
    // scalar expression** (it does not group into a `vec4` and apply `exp` — the vector versions of
    // transcendental functions are implementation-defined).
    // MUST: the statistics are **per fill slot** (if a thread fills different rows, their row
    // statistics differ too). Using slot 0's statistics for the second row quietly normalizes only
    // the second row by another row.
    for (const slot of [0, 1]) {
      assertEquals(
        pv.split(`- row_max${slot}) * row_inv${slot}`).length - 1,
        4,
        `${where}: the normalization expression is exactly the 4 components of slot ${slot}`,
      );
    }
    assertEquals(pv.includes("exp(vec4"), false, `${where}: exp is not grouped into a vector`);
    // MUST: the row-statistics index includes the batch base (blocks the mistake of every batch
    // using batch 0's statistics when B·H ≥ 2)
    assertEquals(pv.includes("let rbase = wid.z * dims.m;"), true, where);
    assertEquals(
      pv.includes("let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);"),
      true,
      where,
    );
    assertEquals(
      pv.includes("let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);"),
      true,
      where,
    );
    // ①③ are separate kernels with separate keys (the skeleton is shared but the fill fragments
    // differ)
    assertNotEquals(qk, pv, where);
    assertNotEquals(attentionQkKey(v4), attentionPvKey(v4), where);
    assertNotEquals(qk, bmmWgsl(v4), where);
    assertNotEquals(pv, bmmWgsl(v4), where);
  }

  // MUST: ② is passes ①② of the current softmax cut out **verbatim**. If the tree-reduction levels,
  // the scan order or `1.0 / Σ` change, bit identity breaks, so it is matched fragment by fragment
  // against the softmax body (the only mechanical detector of a corrupted copy). From v2 on, the
  // identity and the empty-row guard are shared with **the safe_softmax side** (bit-identical to
  // the plain form on finite rows — attention.ts reason 3).
  for (const shared of [
    "    var i = lid;\n    while (i < dim) {",
    "    var stride = 128u;",
    "        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);",
    "    var stride2 = 128u;",
    "        scratch[lid] = scratch[lid] + scratch[lid + stride2];",
    "    let inv = 1.0 / scratch[0u];",
  ]) {
    assertEquals(
      SOFTMAX_WGSL.includes(shared),
      true,
      `fragment missing on the softmax side: ${shared}`,
    );
    assertEquals(
      ATTENTION_STATS_WGSL.includes(shared),
      true,
      `missing on the attention_stats side: ${shared}`,
    );
  }
  // The 3 guards shared with safe_softmax (identity −inf, the empty-row test, amax's select)
  for (const guard of [
    "let neg_inf = bitcast<f32>(params.neg_inf);",
    "    var hi = neg_inf;",
    "    let row_max = scratch[0u];",
    "    let empty = row_max == neg_inf;",
    "    let amax = select(row_max, 0.0, empty);",
  ]) {
    assertEquals(
      SAFE_SOFTMAX_WGSL.includes(guard),
      true,
      `missing on the safe_softmax side: ${guard}`,
    );
    assertEquals(
      ATTENTION_STATS_WGSL.includes(guard),
      true,
      `missing on the attention_stats side: ${guard}`,
    );
  }
  // The scan body matches down to the expression that reads the row's elements (`x[base + …]` only
  // becomes `s[base + …]`)
  assertEquals(ATTENTION_STATS_WGSL.includes("acc = acc + exp(s[base + j] - amax);"), true);
  assertEquals(SOFTMAX_WGSL.includes("acc = acc + exp(x[base + j] - amax);"), true);
  // ② does not write the row back (the substance of going from 3 passes to 1 — the output is only
  // 2 words per row).
  // An empty row is (0.0, 0.0) (amax is already selected; only inv is settled here)
  assertEquals(ATTENTION_STATS_WGSL.includes("stats[row * 2u] = amax;"), true);
  assertEquals(
    ATTENTION_STATS_WGSL.includes("stats[row * 2u + 1u] = select(inv, 0.0, empty);"),
    true,
  );
  // The row direction is grid-stride (a target of the degenerate harness —
  // tests/gpu_gridstride_test.ts)
  assertEquals(ATTENTION_STATS_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals(
    ATTENTION_STATS_KEY,
    `attention_stats:v2:f32:lastdim:safe:wg${ATTENTION_STATS_WORKGROUP_SIZE}`,
  );
  assertEquals(ATTENTION_STATS_WORKGROUP_SIZE, 256);

  // params: only ① puts the f32 scale in the 4th word
  const qkParams = attentionQkParams(5, 7, 4, 0.2973017692565918);
  assertEquals(qkParams.byteLength, 16);
  assertEquals([...qkParams.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkParams.buffer)[3], Math.fround(0.2973017692565918));
  assertEquals([...attentionPvParams(5, 4, 7)], [5, 4, 7, 0]);
  assertEquals([...attentionStatsParams(30, 7)], [30, 7, 0xff800000, 0]);
  assertEquals(attentionStatsParams(30, 7).byteLength, 16);
  assertThrows(() => attentionQkParams(5, 7, 4, Number.NaN), CodegenError);
  assertThrows(() => attentionQkParams(-1, 7, 4, 1), CodegenError);
  // MUST: the finiteness test of the scale is also as f32 (1e39 becomes +Inf in an f32 word, and `0
  // * Inf = NaN`)
  assertThrows(
    () => attentionQkParams(5, 7, 4, 1e39),
    CodegenError,
    "scale must be a finite number as f32",
  );
  assertThrows(() => attentionStatsParams(30, 0), CodegenError);
  assertThrows(() => attentionStatsParams(-1, 7), CodegenError);
  // MUST: the regcache variant checks `dim <= epc · 256` here too (duplicated with the generator,
  // but a check that also blocks the path of calling the kernel directly). Elements beyond it enter
  // neither max nor Σ, a silent wrong value.
  assertEquals([...attentionStatsParams(4, 4096, 16)], [4, 4096, 0xff800000, 0]);
  assertThrows(
    () => attentionStatsParams(4, 8192, 16),
    CodegenError,
    "covered by the regcache variant",
  );
  assertThrows(
    () => attentionStatsParams(4, 4096, ATTENTION_STATS_REG_CACHE_MAX + 1),
    CodegenError,
    "limit",
  );
  // MUST: integrality is checked before the coverage comparison. The generator truncates a
  // non-integer length and unrolls `1.5` as 1 slot (256 elements), while the coverage computation
  // claims 1.5 × 256 = 384, so without this check 44 elements of dim 300 silently drop out of the
  // reduction.
  assertThrows(() => attentionStatsParams(4, 300, 1.5), CodegenError, "is not an integer");
  assertThrows(() => attentionStatsParams(4, 100, 0.5), CodegenError, "is not an integer");
  // The normal integer form (where no truncation happens) passes
  assertEquals([...attentionStatsParams(4, 300, 2)], [4, 300, 0xff800000, 0]);
  await Promise.resolve();
});

/**
 * The additive-mask variants (ADR 0023 revision). **The ground for bit identity is "the only extra
 * rounding is the one add at the write"**, so the output side pins the form of ①'s epilogue and
 * the byte invariance of ②③.
 */
test("attention's additive mask changes only ①QK's epilogue and does not move ②③ by a single byte", () => {
  for (const v4 of [false, true]) {
    const where = `v4=${v4}`;
    const plain = attentionQkWgsl(v4);
    const masked = attentionQkWgsl(v4, "f32", "f32", true);
    assertNotEquals(plain, masked, where);
    // The bindings grow by exactly one, after S (the variant without a mask does not declare
    // binding 4)
    assertEquals(plain.includes("mask"), false, `${where}: no mask appears in the unmasked form`);
    assertEquals(
      masked.includes(`@group(0) @binding(${ATTENTION_QK_MASK_BINDING}) var<storage, read> mask:`),
      true,
      where,
    );
    // MUST: neither the reduction loop nor the shared tile fill changes by a single character (the
    // difference is only "the header comment + the mask binding + the add at the write"). It is
    // pinned in a form where the line-count difference = the number of add lines.
    const addLines = masked.split("\n").filter((line) => line.includes("let sv = "));
    assertEquals(
      masked.split("\n").length - plain.split("\n").length,
      addLines.length + 2 + (v4 ? 0 : 8),
      `${where}: the added lines are only the mask adds + 2 binding lines (8 per-row mbase lines for scalar)`,
    );
    // MUST: **no batch base is added** to the mask index (the contract is a [1,1,M,N] broadcast).
    // The output side has `cbase +` / `obase`, so the point is that the two are not the same index
    // expression.
    for (const line of addLines) {
      assertEquals(line.includes("cbase"), false, `${where}: batch base in the mask: ${line}`);
      assertEquals(line.includes("obase"), false, `${where}: output base in the mask: ${line}`);
    }
    assertEquals(
      masked.includes(
        v4
          ? "let sv = acc0_0 + mask[orow0 * n4 + ocq0];"
          : "let sv = acc0_0.x + mask[mbase + ocol];",
      ),
      true,
      `${where}: the epilogue's form`,
    );
    // MUST: the add happens only once, right before the write (1 output element = 1 add — unrolled
    // per slot)
    assertEquals(
      addLines.length,
      (plain.match(/\n\s+s\[/g) ?? []).length,
      `${where}: number of adds = number of writes`,
    );
    // ③PV does not look at the mask (S is already masked) — neither the key nor the output changes
    assertEquals(attentionPvWgsl(v4), attentionPvWgsl(v4), where);
    assertEquals(attentionPvWgsl(v4).includes("mask"), false, `${where}: ③PV has no mask`);
  }
  // ② The row statistics have no mask axis at all (one key and one WGSL)
  assertEquals(ATTENTION_STATS_WGSL.includes("mask"), false);
  // The key gains exactly one word after s16 (existing keys do not move by a single character)
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(
    attentionQkKey(true, "f32", "f32", true),
    "attention_qk:v1:f32:reg128x128r8x8w16v4:mask",
  );
  assertEquals(
    attentionQkKey(true, "f32", "f16", true),
    "attention_qk:v1:f32:reg128x128r8x8w16v4:s16:mask",
  );
  assertEquals(
    attentionQkKey(false, "f16", "f32", true),
    "attention_qk:v1:f32:reg128x128r8x8w16:c16:mask",
  );
  // One configuration never has 2 keys (every combination of mask × v4 × storage is unique)
  const keys = [false, true].flatMap((v4) =>
    [false, true].map((mask) => attentionQkKey(v4, "f32", "f32", mask)),
  );
  assertEquals(new Set(keys).size, keys.length);
});

/**
 * **Cancels** the difference of a GQA variant to get back the plain variant (ADR 0067 decision 2's
 * "the difference is base arithmetic only").
 *
 * The difference is only ① the one `kv_repeat` word of the uniform ② the 2 lines of GQA comment
 * ③ the one base line of K / V ④ one word of the header comment. **Pinning by being able to undo
 * it** is the point; a form that only checks that the line counts match would pass even if the
 * contents of the reduction loop were swapped.
 */
const undoGqa = (text: string, side: "k" | "v", mapped: string, plain: string): string =>
  text
    .split("\n")
    .filter(
      (line) =>
        line !== "  kv_repeat: u32," &&
        !line.includes("GQA (ADR 0067 decision 2)") &&
        !line.includes("q / S / O / row statistics stay on the q-head"),
    )
    .map((line) => (line === mapped ? plain : line))
    .join("\n")
    .replace(`, GQA (${side} is mapped to the kv-head by integer division)`, "");

/**
 * GQA (broadcast by exact division — ADR 0067 decisions 1 / 2). Pins on the output side that **the
 * output and the key for `r = 1` are byte-identical** (the MUST of adding the option while keeping
 * the 133 existing snapshots and the bit-identity checks frozen), and that the difference of a GQA
 * variant is confined to "the one base line of K / V + one uniform word".
 */
test("attention's GQA variants change only the one base line of K / V and one uniform word", () => {
  for (const v4 of [false, true]) {
    // They **can be set together** with the other axes of storage / compute / mask, so the
    // difference is checked per combination (s16 exists only on the v4 side —
    // src/kernels/score-storage.ts).
    const qkVariants = [
      ["plain", (gqa: boolean) => attentionQkWgsl(v4, "f32", "f32", false, gqa)] as const,
      ["mask", (gqa: boolean) => attentionQkWgsl(v4, "f32", "f32", true, gqa)] as const,
      ["c16", (gqa: boolean) => attentionQkWgsl(v4, "f16", "f32", false, gqa)] as const,
      ...(v4
        ? [["s16", (gqa: boolean) => attentionQkWgsl(v4, "f32", "f16", false, gqa)] as const]
        : []),
    ];
    for (const [label, generate] of qkVariants) {
      const where = `①QK ${label} v4=${v4}`;
      const plain = generate(false);
      const gqa = generate(true);
      assertNotEquals(plain, gqa, where);
      // The plain variant has not a single character of GQA trace (= the r=1 path has not moved by
      // a single byte)
      assertEquals(plain.includes("kv_repeat"), false, `${where}: kv_repeat in the plain form`);
      assertEquals(plain.includes("GQA"), false, `${where}: GQA in the plain form`);
      // The difference is only the one base line of k (+ one uniform word + comments)
      assertEquals(
        gqa.includes("  let bbase = (wid.z / dims.kv_repeat) * dims.n * dims.k;"),
        true,
        `${where}: the base of k is not mapped to the kv head`,
      );
      assertEquals(
        undoGqa(
          gqa,
          "k",
          "  let bbase = (wid.z / dims.kv_repeat) * dims.n * dims.k;",
          "  let bbase = wid.z * dims.n * dims.k;",
        ),
        plain,
        `${where}: the difference goes beyond base arithmetic + one uniform word`,
      );
      // MUST: q (A) and S (C) stay on the q head (only K / V are mapped — ADR 0067 decision 2)
      for (const line of gqa.split("\n")) {
        if (line.includes("let abase") || line.includes("let cbase")) {
          assertEquals(
            line.includes("kv_repeat"),
            false,
            `${where}: the q / S side is mapped: ${line}`,
          );
        }
      }
      // The uniform field comes **after** scale (the order pairs with attentionQkParams)
      assertEquals(
        gqa.includes("  scale: f32,\n  kv_repeat: u32,\n}"),
        true,
        `${where}: field order`,
      );
    }
    const pvVariants = [
      ["plain", (gqa: boolean) => attentionPvWgsl(v4, "f32", "f32", gqa)] as const,
      ["c16", (gqa: boolean) => attentionPvWgsl(v4, "f16", "f32", gqa)] as const,
      ...(v4 ? [["s16", (gqa: boolean) => attentionPvWgsl(v4, "f32", "f16", gqa)] as const] : []),
    ];
    for (const [label, generate] of pvVariants) {
      const where = `③PV ${label} v4=${v4}`;
      const plain = generate(false);
      const gqa = generate(true);
      assertNotEquals(plain, gqa, where);
      assertEquals(plain.includes("kv_repeat"), false, `${where}: kv_repeat in the plain form`);
      // MUST: with v4 the base is **in quads** (in elements it would read at a quarter of the
      // position)
      const nUnit = v4 ? "n4" : "dims.n";
      assertEquals(
        gqa.includes(`  let bbase = (wid.z / dims.kv_repeat) * dims.k * ${nUnit};`),
        true,
        `${where}: the base of v is not mapped to the kv head`,
      );
      assertEquals(
        undoGqa(
          gqa,
          "v",
          `  let bbase = (wid.z / dims.kv_repeat) * dims.k * ${nUnit};`,
          `  let bbase = wid.z * dims.k * ${nUnit};`,
        ),
        plain,
        `${where}: the difference goes beyond base arithmetic + one uniform word`,
      );
      // MUST: both the row statistics (rbase) and the output (cbase) stay on the q head (S stays
      // [B·H,M,N])
      for (const line of gqa.split("\n")) {
        if (
          line.includes("let rbase") ||
          line.includes("let cbase") ||
          line.includes("let abase")
        ) {
          assertEquals(
            line.includes("kv_repeat"),
            false,
            `${where}: the q side is mapped: ${line}`,
          );
        }
      }
    }
    // bmm has no GQA axis (it lives next to ③PV's base expression, so a leak would silently break
    // the "exact batch match" contract [ADR 0022 / 0023])
    assertEquals(bmmWgsl(v4).includes("kv_repeat"), false, `bmm v4=${v4}`);
    assertEquals(matmulWgsl(v4).includes("kv_repeat"), false, `matmul v4=${v4}`);
    assertEquals(linearWgsl("f32", v4).includes("kv_repeat"), false, `linear v4=${v4}`);
  }
  // ② The row statistics have no GQA axis at all (S stays the q-head [B·H, M, N])
  assertEquals(ATTENTION_STATS_WGSL.includes("kv_repeat"), false);

  // The key gains exactly one word after mask (the existing r=1 keys do not move by a single
  // character)
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(
    attentionQkKey(true, "f32", "f32", false, true),
    "attention_qk:v1:f32:reg128x128r8x8w16v4:gqa",
  );
  assertEquals(
    attentionQkKey(true, "f32", "f16", true, true),
    "attention_qk:v1:f32:reg128x128r8x8w16v4:s16:mask:gqa",
  );
  assertEquals(attentionPvKey(true), "attention_pv:v1:f32:reg128x128r8x8w16v4");
  assertEquals(
    attentionPvKey(true, "f32", "f32", true),
    "attention_pv:v1:f32:reg128x128r8x8w16v4:gqa",
  );
  // One configuration never has 2 keys (every combination of v4 × mask × storage × GQA is unique)
  const allKeys = [false, true].flatMap((v4) =>
    [false, true].flatMap((mask) =>
      [false, true].flatMap((gqa) => [
        attentionQkKey(v4, "f32", "f32", mask, gqa),
        attentionPvKey(v4, "f32", "f32", gqa),
      ]),
    ),
  );
  assertEquals(
    new Set(allKeys).size,
    2 * 2 * 2 + 2 * 2,
    "① unique over mask×gqa×v4 / ③ over gqa×v4",
  );

  // params: `r` is in the uniform (not in the key). For ① it is the 5th word (alignment grows it to
  // 32 B), for ③ the 4th word (it stays 16 B), and **the bytes when it is omitted are as before**.
  assertEquals(attentionQkParams(5, 7, 4, 0.5).byteLength, 16);
  const qkGqa = attentionQkParams(5, 7, 4, 0.5, 8);
  assertEquals(qkGqa.byteLength, 32);
  assertEquals([...qkGqa.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkGqa.buffer)[3], 0.5);
  assertEquals(qkGqa[4], 8);
  assertEquals([...attentionPvParams(5, 4, 7, 8)], [5, 4, 7, 8]);
  assertEquals(attentionPvParams(5, 4, 7, 8).byteLength, 16);
  // MUST: r = 0 fails loudly (WGSL's u32 division by zero does not trap and gives a silent wrong
  // value)
  assertThrows(() => attentionQkParams(5, 7, 4, 0.5, 0), CodegenError, "kv_repeat");
  assertThrows(() => attentionPvParams(5, 4, 7, 0), CodegenError, "kv_repeat");
  assertThrows(() => attentionQkParams(5, 7, 4, 0.5, 1.5), CodegenError, "kv_repeat");
});

/**
 * The structure of the **row-window variants** (block execution over query rows — `buildAttention`
 * in src/runtime/recipe-builders/attention.ts). The snapshots freeze only 8 representatives, so
 * "which bases move and which must not" is checked mechanically here over every variant (v4 ×
 * stage × i8a8).
 *
 * MUST: only **one side** moves (①QK's A = q / ③PV's C = O). The other side and the row statistics
 * stay block-relative; a transcription mistake leaves part of S unwritten or gives a silent wrong
 * value multiplied by another block's row statistics.
 * MUST: only the mask's **row** needs `row_offset` outside the base arithmetic (S is a
 * block-relative tensor, the mask is the full-M tensor).
 */
test("the row-window variants of fused attention change only one side's base and the mask's row index", () => {
  for (const v4 of [false, true]) {
    const kUnit = v4 ? "k4" : "dims.k";
    const nUnit = v4 ? "n4" : "dims.n";
    // ①QK: only A (q) gets the full-M stride + row offset. C (S) stays block-relative.
    const qkPlain = attentionQkWgsl(v4);
    const qkWindow = attentionQkWgsl(v4, "f32", "f32", false, false, true);
    assertEquals(qkPlain.includes("row_offset"), false, `①QK plain v4=${v4}`);
    assertEquals(
      qkWindow.includes(
        `  let abase = wid.z * dims.rows_full * ${kUnit} + dims.row_offset * ${kUnit};`,
      ),
      true,
      `①QK row window v4=${v4}: the base of q is not the full-M stride`,
    );
    assertEquals(
      qkWindow.includes(`  let cbase = wid.z * dims.m * ${nUnit};`),
      true,
      `①QK row window v4=${v4}: S is not block-relative`,
    );
    // ③PV: only C (O) gets the full-M stride + row offset. A (S) and the row statistics stay
    // block-relative.
    const pvPlain = attentionPvWgsl(v4);
    const pvWindow = attentionPvWgsl(v4, "f32", "f32", false, true);
    assertEquals(pvPlain.includes("row_offset"), false, `③PV plain v4=${v4}`);
    assertEquals(
      pvWindow.includes(
        `  let cbase = wid.z * dims.rows_full * ${nUnit} + dims.row_offset * ${nUnit};`,
      ),
      true,
      `③PV row window v4=${v4}: the base of O is not the full-M stride`,
    );
    for (const prefix of ["  let abase", "  let rbase"]) {
      const line = pvWindow.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `③PV row window v4=${v4}: ${line}`);
    }
    // The mask is the full-M tensor, so only its row is indexed with `row_offset` added (the
    // columns are the same N as S).
    const maskPlain = attentionQkWgsl(v4, "f32", "f32", true);
    const maskWindow = attentionQkWgsl(v4, "f32", "f32", true, false, true);
    assertEquals(
      maskPlain.includes("mask[(orow"),
      false,
      `①QK mask plain v4=${v4}: an offset is in the row`,
    );
    assertEquals(
      maskWindow.includes(
        v4
          ? "mask[(orow0 + dims.row_offset) * n4 + ocq0]"
          : "let mbase = (orow0 + dims.row_offset)",
      ),
      true,
      `①QK mask row window v4=${v4}: the mask's row is not a full-M index`,
    );
    // i8a8: ①QK has 2 — the quantized q and **its row scale** (mapping only one multiplies by
    // another row's scale, a silent wrong value). The k / S side does not depend on the block, so
    // it is left alone.
    const qkI8a8Plain = attentionQkI8a8Wgsl(v4, true);
    const qkI8a8Window = attentionQkI8a8Wgsl(v4, true, "f32", undefined, true);
    assertEquals(qkI8a8Plain.includes("row_offset"), false, `①QK i8a8 plain v4=${v4}`);
    assertEquals(
      qkI8a8Window.includes("  let qbase = wid.z * dims.rows_full * k4 + dims.row_offset * k4;") &&
        qkI8a8Window.includes("  let qsbase = wid.z * dims.rows_full + dims.row_offset;"),
      true,
      `①QK i8a8 row window v4=${v4}: the bases of the q payload and the row scale do not move as a pair`,
    );
    for (const prefix of ["  let kbase", "  let ksbase", "  let sbase"]) {
      const line = qkI8a8Window.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `①QK i8a8 row window v4=${v4}: ${line}`);
    }
    // i8a8 ③PV has only 1, O (S / the row statistics / Vᵀ are block-relative).
    const pvI8a8Plain = attentionPvI8a8Wgsl(v4, true);
    const pvI8a8Window = attentionPvI8a8Wgsl(v4, true, "f32", undefined, true);
    assertEquals(pvI8a8Plain.includes("row_offset"), false, `③PV i8a8 plain v4=${v4}`);
    assertEquals(
      pvI8a8Window.includes(
        `  let obase = wid.z * dims.rows_full * ${nUnit} + dims.row_offset * ${nUnit};`,
      ),
      true,
      `③PV i8a8 row window v4=${v4}: the base of O is not the full-M stride`,
    );
    for (const prefix of ["  let sbase", "  let rbase", "  let vbase", "  let vsbase"]) {
      const line = pvI8a8Window.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `③PV i8a8 row window v4=${v4}: ${line}`);
    }
  }
  // ② The row statistics have no row-window axis at all (S and the statistics are block-relative
  // and closed within a row).
  assertEquals(ATTENTION_STATS_WGSL.includes("row_offset"), false);

  // The key gains one word **after** GQA (the existing n=1 keys do not move by a single character).
  assertEquals(
    attentionQkKey(true, "f32", "f16", true, true, true),
    "attention_qk:v1:f32:reg128x128r8x8w16v4:s16:mask:gqa:rwa",
  );
  assertEquals(
    attentionPvKey(true, "f32", "f32", true, true),
    "attention_pv:v1:f32:reg128x128r8x8w16v4:gqa:rwc",
  );
  assertEquals(
    attentionQkI8a8Key(true, true, "f16", undefined, true).endsWith(":s16:rwa"),
    true,
    "the row-window word of ①QK i8a8 is at the end",
  );
  assertEquals(
    attentionPvI8a8Key(true, true, "f16", undefined, true).endsWith(":s16:rwc"),
    true,
    "the row-window word of ③PV i8a8 is at the end",
  );
  // One configuration never has 2 keys (every combination of v4 × mask × gqa × row window is
  // unique).
  const allKeys = [false, true].flatMap((v4) =>
    [false, true].flatMap((mask) =>
      [false, true].flatMap((gqa) =>
        [false, true].flatMap((window) => [
          attentionQkKey(v4, "f32", "f32", mask, gqa, window),
          attentionPvKey(v4, "f32", "f32", gqa, window),
        ]),
      ),
    ),
  );
  assertEquals(
    new Set(allKeys).size,
    2 * 2 * 2 * 2 + 2 * 2 * 2,
    "① over mask×gqa×window×v4 / ③ over gqa×window×v4",
  );

  // params: the 2 row-window words are **at the end of the extra fields** (the word position of GQA
  // does not move with the row window).
  const window = { offset: 2, rowsFull: 9 } as const;
  assertEquals(
    attentionQkParams(5, 7, 4, 0.5).byteLength,
    16,
    "without a window it is 16B as before",
  );
  const qkWindowed = attentionQkParams(5, 7, 4, 0.5, undefined, window);
  assertEquals(qkWindowed.byteLength, 32);
  assertEquals([...qkWindowed.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkWindowed.buffer)[3], 0.5);
  assertEquals([qkWindowed[4], qkWindowed[5]], [2, 9]);
  const qkBoth = attentionQkParams(5, 7, 4, 0.5, 8, window);
  assertEquals(
    [qkBoth[4], qkBoth[5], qkBoth[6]],
    [8, 2, 9],
    "kv_repeat comes first even together with GQA",
  );
  const pvWindowed = attentionPvParams(5, 4, 7, undefined, window);
  assertEquals(pvWindowed.byteLength, 32);
  assertEquals([...pvWindowed.slice(0, 5)], [5, 4, 7, 2, 9]);
  assertEquals([...attentionPvParams(5, 4, 7, 8, window).slice(0, 6)], [5, 4, 7, 8, 2, 9]);
  const qkI8a8Windowed = attentionQkI8a8Params(5, 12, 4, 0.5, window);
  assertEquals(qkI8a8Windowed.byteLength, 32);
  assertEquals([qkI8a8Windowed[4], qkI8a8Windowed[5]], [2, 9]);
  assertEquals(new Float32Array(qkI8a8Windowed.buffer)[3], 0.5);
  const pvI8a8Windowed = attentionPvI8a8Params(5, 4, 8, window);
  assertEquals([...pvI8a8Windowed.slice(0, 5)], [5, 4, 8, 2, 9]);
  assertEquals(
    attentionQkI8a8Params(5, 12, 4, 0.5).byteLength,
    16,
    "without a window it is 16B as before",
  );
  assertEquals(
    attentionPvI8a8Params(5, 4, 8).byteLength,
    16,
    "without a window it is 16B as before",
  );
  // MUST: a combination whose block goes beyond the full M fails loudly (passing silently would
  // read and write the neighbouring batch).
  for (const build of [
    () => attentionQkParams(5, 7, 4, 0.5, undefined, { offset: 5, rowsFull: 9 }),
    () => attentionPvParams(5, 4, 7, undefined, { offset: 5, rowsFull: 9 }),
    () => attentionQkI8a8Params(8, 12, 4, 0.5, { offset: 4, rowsFull: 9 }),
    () => attentionPvI8a8Params(8, 4, 8, { offset: 4, rowsFull: 9 }),
  ]) {
    assertThrows(build, CodegenError, "exceed the full M");
  }
});

/**
 * bmm's row-window variants (row-block execution of decomposed attention — src/runtime/fusion.ts).
 * They use **the same 2 words and the same check** as the 4 of fused attention, but had no
 * observation point on the params side, and the only detector was
 * tests/gpu_row_block_attention_test.ts via a real GPU (which passes through without an adapter).
 */
test("bmm's row-window params are 5 words in 32 bytes and fail loudly on an overflowing window", () => {
  const params = bmmRowWindowParams(4, 7, 8, 2, 9, "a", 1);
  assertEquals([...params.slice(0, 5)], [4, 7, 8, 2, 9]);
  // 5 words = 20 bytes are rounded up to 32 bytes by the uniform struct's 16-byte alignment
  assertEquals(params.byteLength, 32);
  assertEquals(bmmParams(4, 7, 8, 1).byteLength, 16, "without a window it is 16B as before");
  // MUST: a combination whose block goes beyond the full M fails loudly (passing silently would
  // read and write the neighbouring batch)
  assertThrows(() => bmmRowWindowParams(5, 7, 8, 5, 9, "a", 1), CodegenError, "exceed the full M");
  assertThrows(() => bmmRowWindowParams(4, 7, 8, -1, 9, "a", 1), CodegenError, "row_offset");
  // The word positions and the WGSL's reading point are a pair (whether the A side = input rows or
  // the C side = output rows has the full-M stride)
  assertEquals(
    bmmWgsl(true, 4, "a").includes(
      "  let abase = wid.z * dims.rows_full * k4 + dims.row_offset * k4;",
    ),
    true,
  );
  assertEquals(
    bmmWgsl(true, 4, "c").includes(
      "  let cbase = wid.z * dims.rows_full * n4 + dims.row_offset * n4;",
    ),
    true,
  );
});

/**
 * The tile constants of the GEMM skeleton. **The typical silent wrong value** is a mismatch between
 * "the edge on the WGSL side / the edge on the dispatch side", so the edges embedded in the output
 * and the geometry are matched in the same test (the executor derives the dispatch edges from the
 * geometry with `gemmTileM` / `gemmTileN`, so a mismatch can only be an accident on the generator
 * side).
 */
test("the tile edges, keys and TS constants of the 8 GEMM-skeleton ops match the default geometry", () => {
  // The reference value of conv's m-tile heuristic (not the actual tile edge — the MUST in
  // gemm-geometry.ts)
  assertEquals(GEMM_TILE, 64);
  // The default geometry (src/kernels/gemm-geometry.ts). It **affects both the keys and the
  // output**, so an unintended difference stops here
  assertEquals(defaultGemmGeometry(), { regM: 8, regN: 8, wgX: 16, wgY: 16 });
  for (const [where, key, wgsl] of GEMM_VARIANTS) {
    // The origin of the output tile (row / column) and the workgroup. Only conv1d / conv2d limit
    // the m tile to 64 (the output channel counts are small — tileN and the register block are
    // shared by every op)
    const conv = where.startsWith("conv");
    const tileM = conv ? 64 : 128;
    assertEquals(wgsl.includes(`let arow0 = wid.y * ${tileM}u + ar;`), true, where);
    assertEquals(wgsl.includes(`let orow0 = wid.y * ${tileM}u + lid.y * 8u;`), true, where);
    assertEquals(wgsl.includes(`@compute @workgroup_size(16, ${tileM / 8})`), true, where);
    // One thread 8×8 = 64 registers (the static unroll `acc{row}_{column quad}` is shared by every
    // op). Shared memory is sa ${tileM}×16 f32 + sb 8,192 B (16 KB in total for the 128-row f32
    // version = the WebGPU default limit)
    assertEquals(wgsl.includes("var acc7_1 = vec4<f32>("), true, where);
    assertEquals(wgsl.includes("acc["), false, `${where}: dynamic index on acc`);
    assertEquals(wgsl.includes(`var<workgroup> sa: array<f32, ${tileM * 16}>;`), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
    // MUST: the generator parameters are in the key (changing the tile edges **and the geometry**
    // gives a separate key). Only conv's implicit GEMM has the discriminator `igemm` (the name says
    // its line differs from the direct kernel), and for the geometry the workgroup shape (`wg16x8`)
    // has the same discriminating power.
    const tilePart = conv ? "igemm" : "reg";
    assertEquals(key.includes(`${tilePart}${tileM}x128`), true, where);
    assertEquals(key.includes(conv ? "wg16x8" : "r8x8w16"), true, where);
    // The protection of edge-channel reads bumps the version of conv and i8 linear. The storage
    // formats whose WGSL did not change and attention / matmul / bmm keep their version.
    const version =
      where.startsWith("attention") || where.startsWith("conv3d")
        ? ":v1:"
        : where.startsWith("conv1d")
          ? ":v4:"
          : where.startsWith("conv2d") || where.startsWith("linear i8")
            ? ":v3:"
            : ":v2:";
    assertEquals(key.includes(version), true, where);
    // No trace of the old 16×16 tile remains (revising the key and replacing the whole output go
    // together)
    assertEquals(key.includes("tile16"), false, where);
  }
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionPvKey(false), "attention_pv:v1:f32:reg128x128r8x8w16");
  // The v4 discriminator is in the key (one bit derived from the shape separates pipelines)
  assertEquals(matmulKey(true), "matmul:v2:f32:reg128x128r8x8w16v4");
  assertEquals(matmulKey(false), "matmul:v2:f32:reg128x128r8x8w16");
  assertEquals(bmmKey(true), "bmm:v2:f32:reg128x128r8x8w16v4");
  assertEquals(linearKey("i8", true), "linear:v3:f32:reg128x128r8x8w16v4:wi8");
  assertEquals(linearKey("f16", false), "linear:v2:f32:reg128x128r8x8w16:wf16");
  // conv1d / conv2d have 2 lines (implicit GEMM / direct kernel), and the direct side's key does
  // not move
  assertEquals(conv2dIgemmKey("f32", true), "conv2d:v3:f32:igemm64x128v4:wg16x8");
  assertEquals(conv2dIgemmKey("i8", false), "conv2d:v3:f32:igemm64x128:wg16x8:wi8");
  assertEquals(conv2dKey("f32"), "conv2d:v1:f32:direct:wg256");
  assertEquals(conv1dIgemmKey("f32", true), "conv1d:v4:f32:igemm64x128v4:wg16x8");
  assertEquals(conv1dIgemmKey("i8", false), "conv1d:v4:f32:igemm64x128:wg16x8:wi8");
  assertEquals(conv1dKey("f32"), "conv1d:v2:f32:direct:wg256");
});

/**
 * conv2d's **32-row m-tile variant** (next to ADR 0024).
 *
 * MUST: it changes only "which workgroup handles which output". The K reduction order of one
 * output element (K tile 16 ascending), the rounding sequence, bias-first and zero fill are held by
 * the skeleton shared with the 64-row version, so here **only the tile geometry and the work
 * assignment** are pinned (numeric agreement is covered by the bitwise comparison on a real GPU —
 * tests/gpu_conv2d_parity_test.ts).
 */
test("conv2d's 32-row m-tile variant changes only the geometry (n tile 128 and the 8×8 registers unchanged)", () => {
  assertEquals(GEMM_MTILE_SMALL, 32);
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = conv2dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL);
      const where = `conv2d igemm m32 ${weight} v4=${v4}`;
      // The m tile is 32 rows (3 places: the A tile's row origin, the store's row origin,
      // bias-first's row origin)
      assertEquals(wgsl.includes("let arow0 = wid.y * 32u + ar;"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 32u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let bias0 = wid.y * 32u + lid.y * 8u;"), true, where);
      // The n tile is 128, the same as the default geometry (the detector that the variant is on
      // the m side only)
      assertEquals(
        wgsl.includes(v4 ? "let bc4 = wid.x * 32u + bcq;" : "let bcol = wid.x * 128u + bcq * 4u;"),
        true,
        where,
      );
      // The workgroup is 16×4 = 64 threads. **One thread 8×8 is unchanged** = the arithmetic
      // density of the inner product does not drop (keeping the workgroup by trimming one thread's
      // outputs would lower the density)
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 4)"), true, where);
      assertEquals(
        wgsl.includes("var acc7_0 = vec4<f32>(bias[min(bias0 + 7u, dims.m - 1u)]);"),
        true,
        where,
      );
      // The shared A shrinks to 32 rows (sb stays 512 since the n tile is the same)
      assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
      // 64 threads fill a B tile of 512 quads, so 8 slots. Each slot's k row shifts by 2
      assertEquals(wgsl.includes("let bk0 = tid / 32u;"), true, where);
      assertEquals(wgsl.includes("let bk7 = bk0 + 14u;"), true, where);
      assertEquals(wgsl.includes("let bk8 ="), false, `${where}: exactly 8 slots`);
      // The canonical inner-product loop is in one place in the skeleton (not a single character
      // different from the 64-row version)
      assertEquals(
        wgsl.includes("acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;"),
        true,
        where,
      );
    }
  }
  // The 64-row version fills the B tile with 4 slots over 128 threads (the 32-row version's 8 slots
  // have not leaked out)
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk3 = bk0 + 12u;"), true);
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk4 ="), false);
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk0 = tid / 32u;"), true);
  // The key is a separate line (the tile shape is a generator parameter, so it is in the key)
  assertEquals(conv2dIgemmKey("f32", true, 32), "conv2d:v3:f32:igemm32x128v4:wg16x4");
  assertEquals(conv2dIgemmKey("i8", false, 32), "conv2d:v3:f32:igemm32x128:wg16x4:wi8");
});

/**
 * The m-tile selection predicate. **The boundary is `M % 64 <= 32`**.
 *
 * MUST: with `< 32`, the census's Cout=96 (`96 % 64 == 32` — 32.4% of the MACs of the 1024 series)
 * would stay on 64 rows, and the very motivation for this variant would vanish. With
 * `M % 64 == 0` 64 rows waste nothing, so splitting into 32 rows only loses by reading the B tile
 * (x's implicit gather) twice.
 */
test("conv2dIgemmMTile picks 32 rows only when M%64 is 1..32", () => {
  // Real shapes from the census (VAE decoder)
  assertEquals(conv2dIgemmMTile(96), 32, "Cout=96 (96%64==32 — the main target of this variant)");
  assertEquals(conv2dIgemmMTile(3), 32, "Cout=3 (waste 21.33× → 10.67×)");
  assertEquals(conv2dIgemmMTile(16), 32, "Cout=16 (waste 4× → 2×)");
  assertEquals(conv2dIgemmMTile(192), 64, "Cout=192 (no waste with 64 rows)");
  assertEquals(conv2dIgemmMTile(384), 64);
  assertEquals(conv2dIgemmMTile(1152), 64);
  // Boundary
  assertEquals(conv2dIgemmMTile(64), 64, "M%64==0");
  assertEquals(conv2dIgemmMTile(65), 32, "M%64==1");
  assertEquals(conv2dIgemmMTile(32), 32);
  assertEquals(
    conv2dIgemmMTile(33),
    64,
    "M%64==33 (the band where the waste is the same and only B reads grow)",
  );
  assertEquals(conv2dIgemmMTile(1), 32);
  // Monotonicity of waste (the chosen tile always has "strictly less or equal waste")
  for (let m = 1; m <= 256; m += 1) {
    const chosen = conv2dIgemmMTile(m);
    const waste = (tile: number) => (Math.ceil(m / tile) * tile) / m;
    assertEquals(waste(chosen) <= waste(chosen === 64 ? 32 : 64), true, `M=${m}`);
  }
});

/**
 * Pins **from the output side** the 4 MUSTs that conv2d's implicit GEMM (ADR 0024) has to meet
 * through fragment sharing in the skeleton. All of them are mistakes of the kind that "never show
 * at measured shapes", and keeping them paired with the numeric tests (parity / CPU reference) is
 * what gives them meaning.
 */
test("conv2d's implicit GEMM has bias-first / zero fill / row scale in its output", () => {
  for (const v4 of [false, true]) {
    const wgsl = conv2dIgemmWgsl("f32", v4);
    const where = `conv2d igemm v4=${v4}`;
    // MUST ①: the bias is acc's initial value (adding it on the store side changes the rounding to
    // (Σ)+bias)
    assertEquals(wgsl.includes("let bias0 = wid.y * 64u + lid.y * 8u;"), true, where);
    assertEquals(
      wgsl.includes("var acc1_0 = vec4<f32>(bias[min(bias0 + 1u, dims.m - 1u)]);"),
      true,
      where,
    );
    assertEquals(wgsl.includes("+ biasv"), false, `${where}: the bias is added on the store side`);
    assertEquals(
      wgsl.includes("+ bias[ocol"),
      false,
      `${where}: the bias is added on the store side`,
    );
    // MUST ③: out-of-range x is 0 (not read through a clamped index)
    assertEquals(
      wgsl.includes(
        "  if (iy < 0 || u32(iy) >= dims.height_in || ix < 0 || u32(ix) >= dims.width_in) {\n    return 0.0;\n  }",
      ),
      true,
      where,
    );
    assertEquals(wgsl.includes("clamp("), false, `${where}: index clamping slipped in`);
    // Flat k → (ic, kh, kw) (swapping Kh and Kw still matches numerically for a square kernel)
    assertEquals(wgsl.includes("let ic = brow0 / khw;"), true, where);
    assertEquals(
      wgsl.includes("let ky = i32((kr / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);"),
      true,
      where,
    );
    assertEquals(
      wgsl.includes("let kx = i32((kr % dims.kernel_w) * dims.dilation_w) - i32(dims.padding_w);"),
      true,
      where,
    );
    // The batch is the z axis (folding it into the N side would swap the output axes into
    // [Cout][B·H·W])
    assertEquals(wgsl.includes("let xbase = wid.z * dims.channels_in;"), true, where);
    assertEquals(wgsl.includes("let cbase = wid.z * dims.m * "), true, where);
    // No dead uniform fields (height_out can be derived from n / width_out)
    assertEquals(
      wgsl.includes("height_out"),
      false,
      `${where}: there is a Dims field nobody reads`,
    );
  }
  // MUST ④: the i8 scale is per **row** (= output channel). Reusing linear's `wcol` (column) is a
  // silent wrong value
  // MUST: bind a separate name per fill slot (one thread fills several channels, so using slot 0's
  // scale for the second one quietly dequantizes only one of them with another channel's scale)
  const i8Wgsl = conv2dIgemmWgsl("i8", true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[min(arow0, dims.m - 1u)];"), true);
  assertEquals(i8Wgsl.includes("let wscale_v1 = wscale[min(arow1, dims.m - 1u)];"), true);
  assertEquals(i8Wgsl.includes("wscale[wcol"), false, "linear's column scale was brought in");
  // The binding numbers are the same as the direct kernel (the executor binds both with one
  // constant)
  assertEquals(
    i8Wgsl.includes(`@group(0) @binding(${CONV2D_SCALE_BINDING}) var<storage, read> wscale:`),
    true,
  );
  // The A tile is the weights (via weight-storage), the B tile is x's implicit gather
  assertEquals(i8Wgsl.includes("av0 = dequant4(arow_base0 + ak0, wscale_v);"), true);
  assertEquals(i8Wgsl.includes("av1 = dequant4(arow_base1 + ak0, wscale_v1);"), true);
  assertEquals(conv2dIgemmWgsl("f16", false).includes("av0.x = dequant(abase);"), true);
  assertEquals(conv2dIgemmWgsl("f32", false).includes("av0.x = w[abase];"), true);
});

/**
 * The v4 decision (`kFlat % 4 == 0 && Wout % 4 == 0 && strideW == 1` — MUST ② of ADR 0024).
 *
 * MUST: the column-quad condition is **`Wout % 4`**. Written as `N % 4` (= B·Hout·Wout), a
 * Hout=Wout=2 shape picks v4 with quads straddling output rows. Every real shape in the census has
 * Wout a multiple of 4, so it **never shows on a real model**.
 */
test("conv2dUsesVec4 looks at all 3 conditions, kFlat / Wout / strideW", () => {
  assertEquals(conv2dUsesVec4(1728, 512, 1), true, "a real shape from the census");
  assertEquals(conv2dUsesVec4(4, 2, 1), false, "Wout=2 (quads straddle rows even with N%4==0)");
  assertEquals(conv2dUsesVec4(18, 8, 1), false, "kFlat=18 (3×3 with Cin=2)");
  assertEquals(conv2dUsesVec4(24, 8, 3), false, "stride_w=3 (4 columns of x are not contiguous)");
  assertEquals(conv2dUsesVec4(4, 4, 1), true);
});

/**
 * Pins **from the output side** the MUSTs that conv1d's implicit GEMM has to meet through fragment
 * sharing in the skeleton (the same 4 points as the 2D version + the 1D-specific "bring in no 2D
 * concept"). Numeric agreement is covered by the bitwise comparison on a real GPU
 * (tests/gpu_conv1d_parity_test.ts).
 */
test("conv1d's implicit GEMM has bias-first / zero fill / row scale / 1-axis Dims in its output", () => {
  for (const v4 of [false, true]) {
    const wgsl = conv1dIgemmWgsl("f32", v4);
    const where = `conv1d igemm v4=${v4}`;
    // MUST ①: the bias is acc's initial value (adding it on the store side changes the rounding to
    // (Σ)+bias)
    assertEquals(wgsl.includes("let bias0 = wid.y * 64u + lid.y * 8u;"), true, where);
    if (v4) {
      assertEquals(wgsl.includes("if (wid.y < dims.m / 64u) {"), true, where);
      assertEquals(wgsl.includes("    acc1_0 = vec4<f32>(bias[bias0 + 1u]);"), true, where);
      assertEquals(
        wgsl.includes("    acc1_0 = vec4<f32>(bias[min(bias0 + 1u, dims.m - 1u)]);"),
        true,
        where,
      );
    } else {
      assertEquals(
        wgsl.includes("var acc1_0 = vec4<f32>(bias[min(bias0 + 1u, dims.m - 1u)]);"),
        true,
        where,
      );
    }

    assertEquals(wgsl.includes("+ biasv"), false, `${where}: the bias is added on the store side`);
    assertEquals(
      wgsl.includes("+ bias[ocol"),
      false,
      `${where}: the bias is added on the store side`,
    );
    // MUST ③: out-of-range x is 0 (not read through a clamped index)
    assertEquals(
      wgsl.includes(
        "  let ix = i32(n * dims.stride) + kt;\n  if (ix < 0 || u32(ix) >= dims.length_in) {\n    return 0.0;\n  }",
      ),
      true,
      where,
    );
    assertEquals(wgsl.includes("clamp("), false, `${where}: index clamping slipped in`);
    // Flat k → (ic, k) (taking ic innermost keeps the same set of taps but changes only the add
    // order)
    assertEquals(wgsl.includes("let ic = brow0 / dims.kernel;"), true, where);
    assertEquals(
      wgsl.includes("let kt = i32((brow0 % dims.kernel) * dims.dilation) - i32(dims.padding);"),
      true,
      where,
    );
    // The batch is the z axis (folding it into the N side would swap the output axes into
    // [Cout][B·Lout])
    assertEquals(wgsl.includes("let xbase = wid.z * dims.channels_in;"), true, where);
    assertEquals(wgsl.includes("let cbase = wid.z * dims.m * "), true, where);
    // MUST: no dead Dims fields (length_out is dims.n itself). Not a single word of the 2D
    // concepts (H/W, Kh/Kw) has leaked either — the detector of a copy leftover from the 2D version
    for (const dead of ["length_out", "width", "height", "kernel_h", "kernel_w", "stride_h"]) {
      assertEquals(wgsl.includes(dead), false, `${where}: ${dead} has leaked`);
    }
    // The A tile, the inner-product loop and the store are **the same fragments** as the 2D version
    // (a single character of difference means the sharing is broken)
    const conv2d = conv2dIgemmWgsl("f32", v4);
    for (const shared of [
      "  let ar = tid / 4u;",
      "  let arow_base0 = arow0 * dims.k;",
      "  let bias0 = wid.y * 64u + lid.y * 8u;",
      "      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;",
      "  let orow0 = wid.y * 64u + lid.y * 8u;",
    ]) {
      assertEquals(wgsl.includes(shared), true, `${where}: ${shared}`);
      assertEquals(conv2d.includes(shared), true, `conv2d igemm v4=${v4}: ${shared}`);
    }
  }
  // MUST ④: the i8 scale is per **row** (= output channel). Reusing linear's `wcol` (column) is a
  // silent wrong value, and the only detector is the test of a shape with 2 or more m tiles
  // (Cout=96 in gpu_conv1d_parity_test.ts)
  const i8Wgsl = conv1dIgemmWgsl("i8", true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[min(arow0, dims.m - 1u)];"), true);
  assertEquals(i8Wgsl.includes("let wscale_v1 = wscale[min(arow1, dims.m - 1u)];"), true);
  assertEquals(i8Wgsl.includes("wscale[wcol"), false, "linear's column scale was brought in");
  // The binding numbers are the same as the direct kernel (the executor binds both with one
  // constant)
  assertEquals(
    i8Wgsl.includes(`@group(0) @binding(${CONV1D_SCALE_BINDING}) var<storage, read> wscale:`),
    true,
  );
  // The A tile is the weights (via weight-storage), the B tile is x's implicit gather
  assertEquals(i8Wgsl.includes("av0 = dequant4(arow_base0 + ak0, wscale_v);"), true);
  assertEquals(conv1dIgemmWgsl("f16", false).includes("av0.x = dequant(abase);"), true);
  assertEquals(conv1dIgemmWgsl("f32", false).includes("av0.x = w[abase];"), true);
});

/**
 * onv1d's **32-row m-tile variant** (the predicate is shared with conv2d — {@link
 * conv2dIgemmMTile}).
 *
 * UST: it changes only "which workgroup handles which output". The K reduction order of one utput
 * element, the rounding sequence, bias-first and zero fill are held by the skeleton shared ith the
 * 64-row version.
 */
test("conv1d's 32-row m-tile variant changes only the geometry (n tile 128 and the 8×8 registers unchanged)", () => {
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = conv1dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL);
      const where = `conv1d igemm m32 ${weight} v4=${v4}`;
      assertEquals(wgsl.includes("let arow0 = wid.y * 32u + ar;"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 32u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let bias0 = wid.y * 32u + lid.y * 8u;"), true, where);
      // The n tile is 128, the same as the default geometry (the detector that the variant is on
      // the m side only)
      assertEquals(
        wgsl.includes(v4 ? "let bc4 = wid.x * 32u + bcq;" : "let bcol = wid.x * 128u + bcq * 4u;"),
        true,
        where,
      );
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 4)"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
      // 64 threads fill a B tile of 512 quads, so 8 slots
      assertEquals(wgsl.includes("let bk7 = bk0 + 14u;"), true, where);
      assertEquals(wgsl.includes("let bk8 ="), false, `${where}: exactly 8 slots`);
    }
  }
  // The 32-row version's work assignment has not leaked into the 64-row version
  assertEquals(conv1dIgemmWgsl("f32", true).includes("let bk3 = bk0 + 12u;"), true);
  assertEquals(conv1dIgemmWgsl("f32", true).includes("let bk4 ="), false);
  // The key is a separate line (the tile shape is a generator parameter, so it is in the key)
  assertEquals(conv1dIgemmKey("f32", true, 32), "conv1d:v4:f32:igemm32x128v4:wg16x4");
  assertEquals(conv1dIgemmKey("i8", false, 32), "conv1d:v4:f32:igemm32x128:wg16x4:wi8");
});

/**
 * The v4 decision (`kFlat % 4 == 0 && Lout % 4 == 0 && stride == 1`).
 *
 * MUST: look at all 3 conditions. In 1D the output plane is one row, so the distinction between
 * 2D's `Wout % 4` and `N % 4` disappears, but **not substituting `Cin % 4`** and **looking at the
 * stride** are needed for the same reasons as in 2D.
 */
test("conv1dUsesVec4 looks at all 3 conditions, kFlat / Lout / stride", () => {
  assertEquals(conv1dUsesVec4(1024, 512, 1), true, "a real shape of the dacvae family");
  assertEquals(
    conv1dUsesVec4(7, 8, 1),
    false,
    "kFlat=7 (K=7 with Cin=1 — Cin%4 would not catch it)",
  );
  assertEquals(
    conv1dUsesVec4(8, 170, 1),
    false,
    "Lout=170 (the scalar variant of a measured shape)",
  );
  assertEquals(conv1dUsesVec4(8, 8, 2), false, "stride=2 (4 columns of x are not contiguous)");
  assertEquals(conv1dUsesVec4(4, 4, 1), true);
});

/**
 * The v4 decision (`k % 4 == 0 && n % 4 == 0`).
 *
 * MUST: look at **both**. A decision on one alone makes the vec4 binding and the actual byte count
 * disagree for the shapes "only k a multiple of 4 / only n a multiple of 4", a wrong value without
 * an exception.
 */
test("gemmUsesVec4 is true only when both k and n are multiples of 4", () => {
  assertEquals(gemmUsesVec4(20, 64), true);
  assertEquals(gemmUsesVec4(4, 4), true);
  assertEquals(gemmUsesVec4(20, 19), false, "only k is a multiple of 4");
  assertEquals(gemmUsesVec4(19, 20), false, "only n is a multiple of 4");
  assertEquals(gemmUsesVec4(19, 19), false);
  // 0 is divisible by 4 (the decision does not throw even for an empty shape)
  assertEquals(gemmUsesVec4(0, 0), true);
});

/**
 * The output of w8a8 (integer dot product of per-token i8 activations × i8 weights).
 *
 * The numeric contract of this path is **atol=0 between the GPU and the TS reference**, so the 3
 * points that decide where rounding happens (the scale is not made by division / the multiplication
 * order of the dequant / the multiply-add rounds once with `fma`) are pinned on the output side.
 * Each of them is only a 1 ULP difference in numeric tests at measured shapes, and becomes a
 * detector only when paired with the atol=0 comparison.
 */
test("i8a8 linear and quantize_rows have the 3 points that decide where rounding happens in their output", () => {
  // ---- quantize_rows -------------------------------------------------------
  // MUST: the scale is a **multiplication by 1/127, not a division by 127** (WGSL's division is
  // allowed up to 2.5 ULP and is 1 ULP off in measurements too; relying on constant folding
  // disagrees with the reference depending on the driver)
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes(
      "let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));",
    ),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("/ 127"), false, "the scale is made by division");
  // MUST: closed to ±127 (not using −128 = the element with the largest absolute value is restored
  // exactly)
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes("let r = clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0));"),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("pack4xI8Clamp"), false, "relies on a saturating pack");
  assertEquals(QUANTIZE_ROWS_WGSL.includes("xq[qbase + q] = pack4xI8(vec4<i32>(r));"), true);
  // MUST: NaN propagation in absmax is a bit-pattern test (the plain max swallows NaN — ADR 0020)
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes("return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;"),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("acc = nan_max(acc, abs(x[base + i]));"), true);
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes("scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);"),
    true,
  );
  // The row direction is grid-stride (a target of the degenerate harness —
  // tests/gpu_gridstride_test.ts)
  assertEquals(QUANTIZE_ROWS_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals(
    QUANTIZE_ROWS_KEY,
    `quantize_rows:v1:f32>i8:pertoken:wg${QUANTIZE_ROWS_WORKGROUP_SIZE}`,
  );
  assertEquals(QUANTIZE_ROWS_WORKGROUP_SIZE, 256);
  // ---- small-D variant (P-1): the geometry follows from dim; the 3 rounding points and NaN
  // propagation are spelled as in the existing form ----
  assertEquals(quantizeRowsGeometry(1024), { lanesPerRow: 256, rowsPerGroup: 1 });
  assertEquals(quantizeRowsGeometry(128), { lanesPerRow: 32, rowsPerGroup: 8 });
  assertEquals(
    quantizeRowsGeometry(96),
    { lanesPerRow: 32, rowsPerGroup: 8 },
    "quad 24 → width 32",
  );
  assertEquals(quantizeRowsGeometry(4), { lanesPerRow: 1, rowsPerGroup: 256 });
  assertEquals(
    quantizeRowsKey(quantizeRowsGeometry(2048)),
    QUANTIZE_ROWS_KEY,
    "the existing form keeps its key",
  );
  assertEquals(
    quantizeRowsWgsl(quantizeRowsGeometry(2048)),
    QUANTIZE_ROWS_WGSL,
    "the existing form keeps its WGSL",
  );
  assertEquals(quantizeRowsKey(quantizeRowsGeometry(128)), `${QUANTIZE_ROWS_KEY}:r8w32`);
  const grouped = quantizeRowsWgsl(quantizeRowsGeometry(128));
  for (const line of [
    "let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));",
    "let r = clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0));",
    "xq[qbase + q] = pack4xI8(vec4<i32>(r));",
    "acc = nan_max(acc, abs(x[base + i]));",
    "scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);",
    "rowBase = rowBase + nwg.x * 8u;",
  ]) {
    assertEquals(grouped.includes(line), true, `missing in the small-D variant: ${line}`);
  }
  assertEquals(grouped.includes("/ 127"), false);
  assertEquals(grouped.includes("pack4xI8Clamp"), false);
  assertEquals([...quantizeRowsParams(30, 8)], [30, 8, 0, 0]);
  assertEquals(quantizeRowsParams(30, 8).byteLength, 16, "16-byte alignment of the uniform struct");
  assertThrows(() => quantizeRowsParams(30, 6), CodegenError, "multiple of 4");
  assertThrows(() => quantizeRowsParams(-1, 8), CodegenError);
  assertThrows(() => quantizeRowsParams(30, 0), CodegenError);

  // ---- i8a8 GEMM -----------------------------------------------------------
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = linearI8a8Wgsl(v4, dp4a);
      const where = `i8a8 v4=${v4} dp4a=${dp4a}`;
      // The integer dot-product variants differ only in **the one idot** (the structural guarantee
      // behind the claim of numeric identity)
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: occurrences of dot4I8Packed`,
      );
      assertEquals(
        wgsl.split("return dot(unpack4xI8(a), unpack4xI8(b));").length - 1,
        dp4a ? 0 : 1,
        `${where}: occurrences of the emulated dot product`,
      );
      // The dot products are unrolled statically into 8 rows × 2 column quads (8×8 outputs = 64 dot
      // products). **Exactly once per pair**, with row numbers fixed at codegen time — the
      // condition is that no dynamic index remains on the accumulators.
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [
          [0, [0, 1, 2, 3]],
          [1, [4, 5, 6, 7]],
        ] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${lanes
            .map((lane) => `idot(a${i}, b${lane})`)
            .join(", ")});`;
          assertEquals(
            wgsl.split(inner).length,
            2,
            `${where}: dot product of row ${i} quad ${quad}`,
          );
        }
      }
      // MUST: a dynamic index `acc[...]` requires addressable local memory and drops out of
      // registers (notably on Metal). That is the very purpose of the unroll, so this checks that
      // not one remains in the output.
      assertEquals(wgsl.includes("acc["), false, `${where}: a dynamic accumulator index remains`);
      // The shared tiles are [pack][row] / [pack][col] (2-way bank conflicts — rearranged from the
      // prototype).
      // One thread fills 4 of A and 2 of B (128 threads cover tileM 128 / tileN 64)
      assertEquals(wgsl.includes("sa[sa_at] = av0;"), true, where);
      assertEquals(wgsl.includes("sa[sa_at + 96u] = av3;"), true, where);
      assertEquals(wgsl.includes("sb[sb_at] = wv0;"), true, where);
      assertEquals(wgsl.includes("sb[sb_at + 32u] = wv1;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 256>;"), true, where);
      // A ragged K is zero-filled (exact, since dot4I8Packed(0, x) == 0). A check is needed **per
      // fill slot**
      assertEquals(wgsl.includes("if (arow0 < dims.m && apack < k4) {"), true, where);
      assertEquals(wgsl.includes("if (arow3 < dims.m && apack < k4) {"), true, where);
      assertEquals(wgsl.includes("if (wcol1 < dims.n && wpack < k4) {"), true, where);
      // Bindings: 5 = weight scale (shared with linear) / 6 = activation scale (i8a8 only)
      assertEquals(wgsl.includes("@group(0) @binding(5) var<storage, read> wscale:"), true, where);
      assertEquals(
        wgsl.includes(`@group(0) @binding(${LINEAR_ACT_SCALE_BINDING}) var<storage, read> xscale:`),
        true,
        where,
      );
      // MUST: the dequant folds `xs·wscale` first, and the multiply-add is an fma (single
      // rounding). The sequential form `f32(acc)·xs·wscale` moves where rounding happens and breaks
      // atol=0.
      assertEquals(
        wgsl.includes(
          v4
            ? "out[orow0 * n4 + ocq0] = fma(vec4<f32>(acc0_0), xscale[orow0] * ws, biasv);"
            : "out[obase + ocol] = fma(f32(acc0_0.x), xs * wscale[ocol], bias[ocol]);",
        ),
        true,
        `${where}: multiplication order of the dequant`,
      );
      // No plain `a * b + c` remains (a form where whether it is fused depends on the driver)
      assertEquals(wgsl.includes(") + biasv;"), false, `${where}: a plain multiply-add remains`);
      // The tile geometry is specific to i8a8 (separate from the f32 skeleton's 64×64 / 16×16 —
      // the integer reduction does not depend on order, which gives the freedom to choose by
      // measurement). The default is M128N64 r8×8 wg8×16 K16.
      assertEquals(wgsl.includes("@compute @workgroup_size(8, 16)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 128u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
    }
  }
  // The key lets diagnostics tell which variant ran (ADR 0021). It carries the geometry, so the
  // version went up to v4
  assertEquals(linearI8a8Key(true, true), "linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a");
  assertEquals(linearI8a8Key(false, false), "linear:v4:i8a8:tile128x64r8x8w8x16k16:dp4aEmu");
  // MUST: the v4 decision looks at **n only** (k % 4 == 0 is handled by the eligibility decision,
  // so it is not mixed into the condition)
  assertEquals(linearI8a8UsesVec4(68), true);
  assertEquals(linearI8a8UsesVec4(19), false);
  // params are the same 3 words as the f32 skeleton + 16-byte alignment. The k check also applies
  // to params
  assertEquals([...linearI8a8Params(5, 7, 8)], [5, 7, 8, 0]);
  assertEquals(linearI8a8Params(5, 7, 8).byteLength, 16);
  assertThrows(() => linearI8a8Params(5, 7, 7), CodegenError, "multiple of 4");
  assertThrows(
    () => linearI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4),
    CodegenError,
    "i32 reduction guard",
  );
  // The extension check is an enumeration of WGSL language features (not a device feature)
  assertEquals(dp4aAvailable(new Set([DP4A_WGSL_FEATURE])), true);
  assertEquals(dp4aAvailable(new Set(["shader-f16"])), false);
});

/**
 * w4a8 (resident i4 × per-token i8 activations — perf-ledger Q-8). The numeric contract rests on
 * only 3 points, "**the flush exactly at the group boundary**", "where xs is applied, asymmetric to
 * the i8 variant" and "the nibble order", so they are pinned on the side of the output's structure
 * rather than by value (the atol=0 comparison on a real GPU is the w4a8 section of
 * tests/gpu_i8a8_test.ts).
 */
test("w4a8 linear flushes to f32 only at group boundaries and carries xs to the last fma", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = linearI8a8Wgsl(v4, dp4a, undefined, "i4", 32);
      const where = `w4a8 v4=${v4} dp4a=${dp4a}`;
      // MUST: a dynamic index drops out of registers (notably on Metal). This checks the very
      // purpose of the unroll for **both** the i32 and the f32 accumulators.
      assertEquals(wgsl.includes("acc["), false, `${where}: dynamic index on the i32 accumulator`);
      assertEquals(wgsl.includes("accf["), false, `${where}: dynamic index on the f32 accumulator`);
      // The inner K-tile loop is **the constant group length / tileK = 2** (a runtime value could
      // misplace the flush relative to the group boundary). Only the outer one is a uniform runtime
      // value.
      assertEquals(wgsl.includes("for (var gt = 0u; gt < 2u; gt = gt + 1u) {"), true, where);
      assertEquals(wgsl.includes("for (var gi = 0u; gi < groups; gi = gi + 1u) {"), true, where);
      assertEquals(
        wgsl.includes("let groups = dims.k >> 5u;"),
        true,
        `${where}: the shift for g=32`,
      );
      // MUST: the flush happens **exactly once for each of the 16 pairs** of 8 rows × 2 column
      // quads (once per group, neither per tile nor per output — the structural guarantee that
      // rounding happens k/g + 1 times).
      for (let row = 0; row < 8; row += 1) {
        for (const quad of [0, 1]) {
          const flush = `    accf${row}_${quad} = fma(vec4<f32>(acc${row}_${quad}), ws${quad}, accf${row}_${quad});`;
          assertEquals(wgsl.split(flush).length, 2, `${where}: flush of row ${row} quad ${quad}`);
        }
      }
      // The group scale is fetched from the flat [n, k/g] by each column's row head + the group
      // number (regressing to a per-channel reading of a column scale would make the index
      // independent of gi)
      assertEquals(wgsl.includes("let wsb1 = min(ocol + 1u, dims.n - 1u) * groups;"), true, where);
      assertEquals(
        wgsl.includes(
          "let ws0 = vec4<f32>(wscale[wsb0 + gi], wscale[wsb1 + gi], wscale[wsb2 + gi], wscale[wsb3 + gi]);",
        ),
        true,
        where,
      );
      // MUST: nibble extraction is a 16-bit shift (zero dynamic vec4 indices), and the stored value
      // is u = q + 8. The order is the same as dequant4 (element 2i = low / 2i+1 = high).
      assertEquals(
        wgsl.includes(
          "  let half = (w[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;\n" +
            "  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);\n" +
            "  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));",
        ),
        true,
        `${where}: nibble extraction`,
      );
      // The B side uses flat element indices (mixing it up with i8's word index k4 reads at a
      // quarter of the position)
      assertEquals(wgsl.includes("let wrow_base0 = wcol0 * dims.k;"), true, where);
      assertEquals(wgsl.includes("let welem = t * 16u + wp * 4u;"), true, where);
      assertEquals(wgsl.includes("wv0 = i4lanes(wrow_base0 + welem);"), true, where);
      // MUST: xs goes into the last fma (the i8 variant's `xs * wscale` folding does not hold for
      // w4a8)
      assertEquals(
        wgsl.includes(
          v4
            ? "out[orow0 * n4 + ocq0] = fma(accf0_0, vec4<f32>(xscale[orow0]), biasv);"
            : "out[obase + ocol] = fma(accf0_0.x, xs, bias[ocol]);",
        ),
        true,
        `${where}: where xs is applied`,
      );
      assertEquals(
        wgsl.includes("xs * wscale"),
        false,
        `${where}: the i8 variant's folded form remains`,
      );
      // The integer dot-product variants differ in the same one place as i8 (the guarantee that the
      // shared fragment idot is used)
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: occurrences of dot4I8Packed`,
      );
    }
  }
  // g is baked into the WGSL, so the output differs per group (paired with the key's g part — a
  // different output under the same key, or the same output under different keys, would both be
  // entry points for silent wrong values)
  assertNotEquals(
    linearI8a8Wgsl(true, true, undefined, "i4", 32),
    linearI8a8Wgsl(true, true, undefined, "i4", 64),
    "the group length does not appear in the output",
  );
  assertEquals(
    linearI8a8Key(true, true, undefined, "i4", 32),
    "linear:v5:i8a8:tile128x64r8x8w8x16k16v4:dp4a:wi4g32",
  );
  assertEquals(
    linearI8a8Key(false, false, undefined, "i4", 64),
    "linear:v5:i8a8:tile128x64r8x8w8x16k16:dp4aEmu:wi4g64",
  );
  // MUST: the i8 key stays as before with the default argument (the direct check that the existing
  // keys are byte-unchanged)
  assertEquals(linearI8a8Key(true, true, undefined, "i8"), linearI8a8Key(true, true));
  // i4 and the group length go together (only one of them is a wiring bug — i4GroupShift in
  // weight-storage.ts)
  assertThrows(
    () => linearI8a8Wgsl(true, true, undefined, "i4"),
    CodegenError,
    "groupSize must be passed together with i4 weight storage",
  );
  assertThrows(
    () => linearI8a8Wgsl(true, true, undefined, "i8", 32),
    CodegenError,
    "groupSize must be passed together with i4 weight storage",
  );
  // A storage form that does not fit the integer dot product fails loudly (there is no path that
  // passes f32 / f16 at all)
  assertThrows(() => linearI8a8Wgsl(true, true, undefined, "f16"), CodegenError, "must be i8 / i4");
  // MUST: a shape where one K tile (16) straddles a group boundary is rejected at generation —
  // letting it through would mix the weights of 2 groups into one i32 and apply only one of their
  // scales at the flush, a silent wrong value
  assertThrows(
    () => linearI8a8Wgsl(true, true, { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 32 }, "i4", 16),
    CodegenError,
    "is not a multiple of the K tile 32",
  );
  // Positive cases (the counterpart showing the check is not too wide): when the group is a
  // multiple of the K tile it passes, and the inner loop's `tilesPerGroup = group / tileK` is baked
  // as is. The same holds for a geometry with a different tileK.
  assertEquals(
    linearI8a8Wgsl(true, true, { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 32 }, "i4", 32).includes(
      "let gbase = gi * 1u;",
    ),
    true,
    "tileK 32 × group 32 is 1 group = 1 tile",
  );
  assertEquals(
    linearI8a8Wgsl(true, true, undefined, "i4", 16).includes("let gbase = gi * 1u;"),
    true,
    "the default geometry (tileK 16) × group 16 is 1 group = 1 tile",
  );
  assertEquals(
    linearI8a8Wgsl(true, true, undefined, "i4", 64).includes("let gbase = gi * 4u;"),
    true,
    "the default geometry (tileK 16) × group 64 is 1 group = 4 tiles",
  );
  // The params check is on the **group length**, not on k (i8's k check does not apply to w4a8)
  assertEquals([...linearI8a8Params(5, 8, 64, 32)], [5, 8, 64, 0]);
  assertEquals(
    [...linearI8a8Params(5, 8, LINEAR_I8A8_MAX_K + 32, 32)].length,
    4,
    "w4a8 does not apply i8's k check",
  );
  assertThrows(
    () => linearI8a8Params(5, 8, 48, 32),
    CodegenError,
    "is not divisible by group_size 32",
  );
  assertThrows(
    () => linearI8a8Params(5, 8, LINEAR_W4A8_MAX_GROUP * 2, LINEAR_W4A8_MAX_GROUP * 2),
    CodegenError,
    "i32 reduction guard",
  );
});

/**
 * The i8a8 variant of fused attention ①QK (design §2.1 / §4.1, iteration 1).
 *
 * The numeric contract is the same **GPU vs TS reference atol=0** as linear's w8a8, so the 3 points
 * that decide where rounding happens (the half scale goes to the dequant side / `qs'·ks'` is folded
 * first / one single implementation of the integer dot product) and **the units of the batch base**
 * (the scale in rows, the payload in packs) are pinned on the output side.
 */
test("i8a8 attention_qk keeps the half scale on the dequant side and separates the batch base by unit", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = attentionQkI8a8Wgsl(v4, dp4a);
      const where = `attention_qk i8a8 v4=${v4} dp4a=${dp4a}`;
      // The integer dot product is the one implementation shared with linear (only `idot` differs
      // per variant)
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: occurrences of dot4I8Packed`,
      );
      assertEquals(
        wgsl.split("return dot(unpack4xI8(a), unpack4xI8(b));").length - 1,
        dp4a ? 0 : 1,
        `${where}: occurrences of the emulated dot product`,
      );
      // The dot products are unrolled statically into 8 rows × 2 column quads (the same unroll as
      // linear's i8a8). **Exactly once per pair**, with row numbers fixed at codegen time — the
      // condition is that no dynamic index remains on the accumulators.
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [
          [0, [0, 1, 2, 3]],
          [1, [4, 5, 6, 7]],
        ] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${lanes
            .map((lane) => `idot(a${i}, b${lane})`)
            .join(", ")});`;
          assertEquals(
            wgsl.split(inner).length,
            2,
            `${where}: dot product of row ${i} quad ${quad}`,
          );
        }
      }
      // MUST: a dynamic index `acc[...]` requires addressable local memory and drops out of
      // registers. That is the very purpose of the unroll, so this checks that not one remains in
      // the output.
      assertEquals(wgsl.includes("acc["), false, `${where}: a dynamic accumulator index remains`);
      // MUST: the half scale goes to both q / k **on the dequant side, not before quantization**.
      // Moving it to one side or before quantization both give a wrong value without an exception.
      assertEquals(
        wgsl.includes(
          v4
            ? "s[sbase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * ((qscale[qsbase + orow0] * dims.scale) * ks);"
            : "s[obase + ocol] = f32(acc0_0.x) * (qs * (kscale[ksbase + ocol] * dims.scale));",
        ),
        true,
        `${where}: multiplication order of the dequant and position of the half scale`,
      );
      // There is no bias, so no fma is used (the counterpart of linear's `fma(…, bias)`)
      assertEquals(wgsl.includes("fma("), false, `${where}: an fma slipped in`);
      // MUST: the batch base is "payload = in packs / scale = in rows / S = in elements (quads)".
      // Mixing up the units reads and writes the neighbouring head when B·H ≥ 2 (it does not show
      // in a test with B=H=1)
      assertEquals(wgsl.includes("let qbase = wid.z * dims.m * k4;"), true, where);
      assertEquals(wgsl.includes("let kbase = wid.z * dims.n * k4;"), true, where);
      assertEquals(wgsl.includes("let qsbase = wid.z * dims.m;"), true, where);
      assertEquals(wgsl.includes("let ksbase = wid.z * dims.n;"), true, where);
      assertEquals(
        wgsl.includes(`let sbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};`),
        true,
        where,
      );
      // k is read as [N,D] as is (the contiguous direction is D = the pack direction, so no
      // transpose is needed)
      assertEquals(wgsl.includes("let krow_base0 = kbase + wcol0 * k4;"), true, where);
      assertEquals(wgsl.includes("let krow_base1 = kbase + wcol1 * k4;"), true, where);
      // Bindings: 4 = q's row scale / 5 = k's column scale
      assertEquals(
        wgsl.includes(
          `@group(0) @binding(${ATTENTION_QK_Q_SCALE_BINDING}) var<storage, read> qscale:`,
        ),
        true,
        where,
      );
      assertEquals(
        wgsl.includes(
          `@group(0) @binding(${ATTENTION_QK_K_SCALE_BINDING}) var<storage, read> kscale:`,
        ),
        true,
        where,
      );
      // The tile geometry is the same default as linear's i8a8 (M128N64 r8×8 wg8×16 K16). **It
      // differs from ③PV**
      assertEquals(wgsl.includes("@compute @workgroup_size(8, 16)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 128u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 256>;"), true, where);
    }
  }
  // The key follows the precedent of linear:v4:i8a8 (version ++ / dtype field i8a8 / geometry /
  // dot-product variant at the end)
  assertEquals(
    attentionQkI8a8Key(true, true),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a",
  );
  assertEquals(
    attentionQkI8a8Key(false, false),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16:dp4aEmu",
  );
  // MUST: the v4 decision looks at **n only** (D % 4 == 0 is handled by the eligibility decision)
  assertEquals(attentionQkI8a8UsesVec4(68), true);
  assertEquals(attentionQkI8a8UsesVec4(19), false);
  // params are the same 3 words as the f32 variant + the 4th word is the f32 bit pattern of the
  // half scale
  const params = attentionQkI8a8Params(5, 7, 8, 0.5);
  assertEquals([...params.slice(0, 3)], [5, 7, 8]);
  assertEquals(new Float32Array(params.buffer)[3], 0.5);
  assertEquals(params.byteLength, 16);
  assertThrows(() => attentionQkI8a8Params(5, 7, 7, 0.5), CodegenError, "multiple of 4");
  assertThrows(() => attentionQkI8a8Params(5, 7, 8, Number.NaN), CodegenError, "finite");
  assertThrows(
    () => attentionQkI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4, 0.5),
    CodegenError,
    "i32 reduction guard",
  );
});

/**
 * The i8a8 variant of ③PV (design §2.2 / §2.3). The numeric contract rests on only 4 points, so the
 * output side pins all of them — ① P̃ is built per component as `round(127·exp(S−m))` (no
 * division, no amax taken) ② out of range is 0 without applying the conversion ③ the dequant folds
 * `prow·vs` first, and `1/127` is a multiplication ④ the row statistics are `rbase + orow` /
 * `rbase + arow` (including the batch base).
 */
test("i8a8 attention_pv builds P̃ without materializing it and fetches the column scale and the row inv along separate axes", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = attentionPvI8a8Wgsl(v4, dp4a);
      const where = `attention_pv i8a8 v4=${v4} dp4a=${dp4a}`;
      // The integer dot product is the one implementation shared with linear / ①QK (the detector
      // that it was not copied)
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: occurrences of dot4I8Packed`,
      );
      // The dot products are unrolled statically into 8 rows × 2 column quads (the same unroll as
      // ①QK / linear's i8a8). **Exactly once per pair**, and the condition is that no dynamic index
      // remains on the accumulators (paired with the `acc[` check below).
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [
          [0, [0, 1, 2, 3]],
          [1, [4, 5, 6, 7]],
        ] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${lanes
            .map((lane) => `idot(a${i}, b${lane})`)
            .join(", ")});`;
          assertEquals(
            wgsl.split(inner).length,
            2,
            `${where}: dot product of row ${i} quad ${quad}`,
          );
        }
      }
      // MUST: a dynamic index `acc[...]` requires addressable local memory and drops out of
      // registers.
      assertEquals(wgsl.includes("acc["), false, `${where}: a dynamic accumulator index remains`);
      // MUST: P̃ is a **per-component scalar expression**, exactly 2 fill slots × 4 components
      // (it does not group into a vec4 and apply exp)
      assertEquals(
        wgsl.split("round(exp(raw0.").length - 1,
        4,
        `${where}: the quantization expression of P̃ is exactly the 4 components of slot 0`,
      );
      assertEquals(
        wgsl.split("round(exp(raw1.").length - 1,
        4,
        `${where}: the quantization expression of P̃ is exactly the 4 components of slot 1`,
      );
      assertEquals(wgsl.includes("exp(vec4"), false, `${where}: exp is not grouped into a vector`);
      // MUST: the grid is 127 (128 would be a regression of the denominator form). The scale is
      // made by **multiplying** by 1/127 (no division)
      assertEquals(wgsl.includes("- row_max0) * 127.0"), true, `${where}: the grid of P̃`);
      assertEquals(
        wgsl.includes("stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496"),
        true,
        `${where}: prow = inv·(1/127) as a multiplication`,
      );
      assertEquals(wgsl.includes("/ 127"), false, `${where}: 1/127 is made by division`);
      // MUST: out of range in the A tile stays 0 (applying exp(0−m)·127 quietly corrupts ragged
      // tiles)
      assertEquals(
        wgsl.includes("    var av0 = 0u;\n    if (arow0 < dims.m && apack < k4) {"),
        true,
        `${where}: the check of zero fill out of range`,
      );
      assertEquals(
        wgsl.includes("    var av1 = 0u;\n    if (arow1 < dims.m && apack < k4) {"),
        true,
        `${where}: the check of zero fill out of range (slot 1)`,
      );
      // MUST: the dequant folds prow·vs into one f32 first (there is no bias, so no fma)
      assertEquals(
        wgsl.includes(
          v4
            ? "o[obase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * (prow * vs);"
            : "o[orow_base + ocol] = f32(acc0_0.x) * (prow * vscale[vsbase + ocol]);",
        ),
        true,
        `${where}: multiplication order of the dequant`,
      );
      assertEquals(wgsl.includes("fma("), false, `${where}: an fma slipped in`);
      // MUST: in the row statistics the row max uses `arow` (the A-tile fill) and the row inv uses
      // `orow` (the write). Reusing one for the other applies a row of a thread with a different
      // share (they agree only when B·H = M = 1)
      // MUST: the row max is fetched **per fill slot** (one thread fills 2 rows, so reusing one
      // value mixes in the neighbouring row's maximum — a wrong value without an exception)
      assertEquals(
        wgsl.includes("let stat_at0 = select(0u, (rbase + arow0) * 2u, arow0 < dims.m);"),
        true,
        where,
      );
      assertEquals(
        wgsl.includes("let stat_at1 = select(0u, (rbase + arow1) * 2u, arow1 < dims.m);"),
        true,
        where,
      );
      // MUST: the batch base is "S = in quads / vq = in packs / statistics and column scale = in
      // counts"
      assertEquals(wgsl.includes("let sbase = wid.z * dims.m * k4;"), true, where);
      assertEquals(wgsl.includes("let vbase = wid.z * dims.n * k4;"), true, where);
      assertEquals(wgsl.includes("let rbase = wid.z * dims.m;"), true, where);
      assertEquals(wgsl.includes("let vsbase = wid.z * dims.n;"), true, where);
      assertEquals(
        wgsl.includes(`let obase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};`),
        true,
        where,
      );
      // vq is [D,N] with N contiguous (the pack direction), so no transpose is needed
      assertEquals(wgsl.includes("let vrow_base0 = vbase + wcol0 * k4;"), true, where);
      assertEquals(wgsl.includes("let vrow_base3 = vbase + wcol3 * k4;"), true, where);
      // Bindings: 5 = the row scale of Vᵀ (= the column scale of V)
      assertEquals(
        wgsl.includes(
          `@group(0) @binding(${ATTENTION_PV_V_SCALE_BINDING}) var<storage, read> vscale:`,
        ),
        true,
        where,
      );
      // The eligibility condition gives N % 4 == 0, so S can always be read in quads (the v4 flag
      // concerns only the output D side)
      assertEquals(
        wgsl.includes("@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;"),
        true,
        where,
      );
      // MUST: the tile geometry is **separate** from ①QK (only for ③ does a single tile of N = D =
      // 128 win — measured. Bundling them under the assumption of "one best geometry" always leaves
      // one of them behind)
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 8)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 64u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 256>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 512>;"), true, where);
      // Separate from ①QK (the skeleton is shared, but the fill, the dequant and the geometry
      // differ)
      assertNotEquals(wgsl, attentionQkI8a8Wgsl(v4, dp4a), where);
    }
  }
  // The key follows the same convention as ①QK (version ++ / dtype field i8a8 / geometry /
  // dot-product variant at the end)
  assertEquals(
    attentionPvI8a8Key(true, true),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4a",
  );
  assertEquals(
    attentionPvI8a8Key(false, false),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16:dp4aEmu",
  );
  // MUST: the v4 decision looks at **the output columns = D only** (N % 4 == 0 is handled by the
  // eligibility decision)
  assertEquals(attentionPvI8a8UsesVec4(128), true);
  assertEquals(attentionPvI8a8UsesVec4(19), false);
  // params are the same 3 words as the f32 variant (the half scale does not appear in ③)
  const params = attentionPvI8a8Params(5, 7, 8);
  assertEquals([...params], [5, 7, 8, 0]);
  assertEquals(params.byteLength, 16);
  assertThrows(() => attentionPvI8a8Params(5, 7, 7), CodegenError, "multiple of 4");
  assertThrows(
    () => attentionPvI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4),
    CodegenError,
    "i32 reduction guard",
  );
});

/**
 * The i8a8 tile geometry (src/kernels/i8a8-geometry.ts). If the one point that **the geometry is a
 * generator parameter and a key axis** breaks, the determinism "same key → byte-identical WGSL"
 * breaks, so one test pins ① the defaults ② their reflection in the key ③ their reflection in the
 * output ④ fail loudly on combinations that do not divide. The values themselves are exact i32
 * additions, so changing the geometry does not move a single bit (checking that on hardware is
 * tests/gpu_i8a8_test.ts's job).
 */
test("the i8a8 geometry is reflected in the key and the output, and combinations that do not divide fail loudly", () => {
  // ① Defaults: linear / ①QK are M128N64, only ③PV is M64N128 (a single tile of N = D = 128 wins)
  assertEquals(defaultI8a8Geometry("linear"), { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 });
  assertEquals(defaultI8a8Geometry("attention_qk"), defaultI8a8Geometry("linear"));
  assertEquals(defaultI8a8Geometry("attention_pv"), {
    regM: 8,
    regN: 8,
    wgX: 16,
    wgY: 8,
    tileK: 16,
  });
  assertEquals(i8a8TileM(defaultI8a8Geometry("linear")), 128);
  assertEquals(i8a8TileN(defaultI8a8Geometry("linear")), 64);
  assertEquals(i8a8TileM(defaultI8a8Geometry("attention_pv")), 64);
  assertEquals(i8a8TileN(defaultI8a8Geometry("attention_pv")), 128);

  // ② Key: the tile edges alone do not determine the geometry (M128N64 can be built as r8×8
  // wg8×16 or as r16×8 wg8×8), so the register block, the workgroup shape and the K width are in
  // it too
  const alt: I8a8Geometry = { regM: 16, regN: 8, wgX: 8, wgY: 8, tileK: 32 };
  assertEquals(i8a8TileM(alt), 128);
  assertEquals(i8a8TileN(alt), 64);
  assertEquals(i8a8GeometryKeyPart(alt, true), "tile128x64r16x8w8x8k32v4");
  assertNotEquals(
    i8a8GeometryKeyPart(alt, true),
    i8a8GeometryKeyPart(defaultI8a8Geometry("linear"), true),
  );
  for (const [where, keyOf] of [
    ["linear", (geometry: I8a8Geometry) => linearI8a8Key(true, true, geometry)],
    ["attention_qk", (geometry: I8a8Geometry) => attentionQkI8a8Key(true, true, "f32", geometry)],
    ["attention_pv", (geometry: I8a8Geometry) => attentionPvI8a8Key(true, true, "f32", geometry)],
  ] as const) {
    assertNotEquals(
      keyOf(alt),
      keyOf(defaultI8a8Geometry("linear")),
      `${where}: the geometry is not in the key`,
    );
    assertEquals(keyOf(alt).includes("tile128x64r16x8w8x8k32v4"), true, where);
  }

  // ③ Output: the workgroup shape, the shared tiles' byte count and the fill slot count follow the
  // geometry (tileK 32 = 8 packs / 64 threads, so A takes 16 rounds and B 8)
  for (const [where, wgsl] of [
    ["linear", linearI8a8Wgsl(true, true, alt)],
    ["attention_qk", attentionQkI8a8Wgsl(true, true, "f32", alt)],
    ["attention_pv", attentionPvI8a8Wgsl(true, true, "f32", alt)],
  ] as const) {
    assertEquals(wgsl.includes("@compute @workgroup_size(8, 8)"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 1024>;"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 512>;"), true, where);
    assertEquals(wgsl.includes("let sa_at = ap * 128u + ar;"), true, where);
    assertEquals(wgsl.includes("let tiles = (k4 + 7u) / 8u;"), true, where);
    // The fill **exactly covers** tileM / tileN (a word left unreached enters the dot product as 0)
    assertEquals(
      wgsl.includes("sa[sa_at + 120u] = av15;"),
      true,
      `${where}: the last slot of the A fill`,
    );
    assertEquals(wgsl.includes("+ 56u] = "), true, `${where}: the last slot of the B fill`);
    assertEquals(wgsl.includes("acc15_1 = acc15_1 +"), true, `${where}: the unroll over 16 rows`);
    assertEquals(wgsl.includes("acc["), false, `${where}: dynamic accumulator index`);
    assertNotEquals(wgsl, "", where);
  }
  assertNotEquals(linearI8a8Wgsl(true, true, alt), linearI8a8Wgsl(true, true));

  // ④ Combinations that do not divide are rejected at generation (a hole in the shared tile = a
  // wrong value without an exception)
  const bad: readonly [string, I8a8Geometry][] = [
    // regN is not a multiple of 4 (acc cannot be bundled into vec4)
    ["regN", { regM: 8, regN: 6, wgX: 8, wgY: 16, tileK: 16 }],
    // tileK is not a multiple of 4 (i8's packing by 4 does not divide)
    ["tileK", { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 18 }],
    // The thread count does not divide by the K pack count (the fill shares cannot be assigned)
    ["K pack count", { regM: 8, regN: 8, wgX: 3, wgY: 3, tileK: 16 }],
    // The tile edge does not divide by the fill stride (words are left unreached)
    ["fill stride", { regM: 3, regN: 8, wgX: 8, wgY: 16, tileK: 16 }],
    ["positive integer", { regM: 0, regN: 8, wgX: 8, wgY: 16, tileK: 16 }],
  ];
  for (const [message, geometry] of bad) {
    assertThrows(() => linearI8a8Wgsl(true, true, geometry), CodegenError, message);
    assertThrows(() => attentionQkI8a8Wgsl(true, true, "f32", geometry), CodegenError, message);
    assertThrows(() => attentionPvI8a8Wgsl(true, true, "f32", geometry), CodegenError, message);
  }
});

/**
 * ② The **regcache variant** of the row statistics (reads S only once and keeps it in registers).
 *
 * MUST: both the element → thread assignment (`lid + 256·s`) and the reduction order (`s`
 * ascending → a 256-wide tree) are **exactly the same** as the two-read version. If they move,
 * ADR 0023's bit identity (② verbatim matches passes ①② of softmax) breaks, so the output side pins
 * the scan order and the out-of-range handling.
 */
test("the regcache variant of attention_stats reads S only once without changing the assignment or the reduction order", () => {
  const loop = attentionStatsWgsl();
  const cached = attentionStatsWgsl("f32", "f32", 3);
  // The two-read version's output does not move by a single byte (protecting the default path —
  // paired with the snapshots)
  assertEquals(loop, ATTENTION_STATS_WGSL);
  assertEquals(loop.split("s[base + ").length - 1, 2, "the two-read version reads S twice");
  // The regcache version reads S exactly epc times (= once per element)
  assertEquals(cached.split("s[base + ").length - 1, 3, "the regcache version reads S only once");
  // The scan order is `lid + 256·s` ascending (the same sequence as the two-read version's `i +=
  // 256`)
  assertEquals(cached.includes("  let i0 = lid;"), true);
  assertEquals(cached.includes("  let i1 = lid + 256u;"), true);
  assertEquals(cached.includes("  let i2 = lid + 512u;"), true);
  // ① max folds the read values as they are / ② the sum adds `exp(c - amax)` in the same order
  assertEquals(cached.includes("      c0 = s[base + i0];\n      hi = nan_max(hi, c0);"), true);
  assertEquals(
    cached.includes("    if (i2 < dim) {\n      acc = acc + exp(c2 - amax);\n    }"),
    true,
  );
  // MUST: out of range is **not touched** at either level (mixing in 0.0 quietly corrupts both max
  // and the sum)
  assertEquals(
    cached.split("if (i0 < dim) {").length - 1,
    2,
    "both levels need the out-of-range check",
  );
  // The tree reduction and the write are verbatim identical to the two-read version
  for (const shared of [
    "      scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);",
    "      scratch[lid] = scratch[lid] + scratch[lid + stride2];",
    "    let inv = 1.0 / scratch[0u];",
    "      stats[row * 2u] = amax;",
    "    row = row + nwg.x;",
  ]) {
    assertEquals(cached.includes(shared), true, `missing in the regcache version: ${shared}`);
  }
  // dim → epc is one pure function in one place. A shape beyond the limit falls back to the
  // two-read version (the values are the same either way)
  assertEquals(attentionStatsRegCache(1), 1);
  assertEquals(attentionStatsRegCache(256), 1);
  assertEquals(attentionStatsRegCache(257), 2);
  assertEquals(attentionStatsRegCache(512), 2);
  assertEquals(attentionStatsRegCache(4096), 16);
  assertEquals(attentionStatsRegCache(ATTENTION_STATS_REG_CACHE_MAX * 256), 32);
  assertEquals(attentionStatsRegCache(ATTENTION_STATS_REG_CACHE_MAX * 256 + 1), undefined);
  // The key adds `:rc{epc}` at the end (the default = the two-read key does not move by a single
  // character)
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  assertEquals(attentionStatsKey("f32", "f32", 16), `${ATTENTION_STATS_KEY}:rc16`);
  assertEquals(attentionStatsKey("f32", "f16", 2), `${ATTENTION_STATS_KEY}:s16:rc2`);
  assertNotEquals(attentionStatsKey("f32", "f32", 2), attentionStatsKey("f32", "f32", 16));
});

// Ruling (src/kernels/gather.ts): an out-of-range index does not return another element but taints
// with NaN.
test("gather taints out-of-range indices with NaN and reads at a row offset on the normal path", () => {
  assertEquals(GATHER_WGSL.includes("if (pick < 0 || u32(pick) >= dims.src_cols) {"), true);
  assertEquals(GATHER_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
  // MUST: the NaN bit pattern is carried in params (some implementations may turn a constant
  // expression `bitcast<f32>(0x…)` into a shader creation error as "the const-expression is NaN" —
  // the MUST in the kernel doc).
  assertEquals(
    GATHER_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN is built with a constant expression (carry it via params)",
  );
  assertEquals(GATHER_WGSL.includes("out[i] = src[row * dims.src_cols + u32(pick)];"), true);
  // The output is contiguous, so the row is divided out of the flat index (assuming grid-stride)
  assertEquals(GATHER_WGSL.includes("let row = i / dims.cols;"), true);
  assertEquals(GATHER_WGSL.includes("var i = gid.x;"), true);
});

// The fused ops of ADR 0012. Kernel-specific invariants are pinned from the output side.
test("the fused kernels are separate from the existing kernels and generate the contracted forms", () => {
  // linear reads the weights in the transposed [n,k] layout as is (the contiguous direction is k,
  // so 4 contiguous elements along k)
  assertEquals(LINEAR_WGSL.includes("let wrow_base0 = wcol0 * dims.k;"), true);
  assertEquals(LINEAR_WGSL.includes("wv0.x = w[wbase];"), true);
  // MUST: the transpose is done on the shared-memory side (column quad = wc / 4, component = wc %
  // 4). Mixing the two up swaps columns within one tile — only a ragged shape with per-column
  // differing values detects it.
  assertEquals(LINEAR_WGSL.includes("let wsq0 = wc0 / 4u;"), true);
  assertEquals(LINEAR_WGSL.includes("let wsl0 = wc0 % 4u;"), true);
  // MUST: components are written **statically** (not with the dynamic index `sb[i][wsl] = v`). On
  // Metal, writes with wsl != 0 are silently dropped, and 3 of the 4 elements enter the inner
  // product as 0 (the mechanism is storeBTransposed in src/kernels/gemm.ts). Every transposed
  // placement of the 4 arms is pinned — mixing up even one arm swaps columns within one tile.
  assertEquals(LINEAR_WGSL.includes("[wsl]"), false);
  const components = ["x", "y", "z", "w"] as const;
  // 4 arms per fill slot (a geometry where one thread fills several channels)
  for (const slot of [0, 1]) {
    for (const [at, component] of components.entries()) {
      assertEquals(
        LINEAR_WGSL.includes(
          `${at === components.length - 1 ? "default" : `case ${at}u`}: {\n` +
            `        sb[sb_base${slot}].${component} = wv${slot}.x;\n` +
            `        sb[sb_base${slot} + 32u].${component} = wv${slot}.y;\n` +
            `        sb[sb_base${slot} + 64u].${component} = wv${slot}.z;\n` +
            `        sb[sb_base${slot} + 96u].${component} = wv${slot}.w;\n` +
            `      }`,
        ),
        true,
        `linear's transposed B-tile placement (slot ${slot} component ${component})`,
      );
    }
  }
  // The bias is added once at the end (not as the accumulator's initial value — to keep the
  // reduction order). acc is the static unroll `acc{row}_{column quad}`, so the same form repeats
  // per row.
  assertEquals(LINEAR_WGSL.includes("out[obase + ocol] = acc0_0.x + bias[ocol];"), true);
  assertEquals(
    linearWgsl("f32", true).includes(
      "let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);",
    ),
    true,
  );
  assertEquals(linearWgsl("f32", true).includes("out[orow0 * n4 + ocq0] = acc0_0 + biasv;"), true);
  // MUST: the concept of fused ops has not leaked into the output of matmul / bmm
  assertEquals(MATMUL_WGSL.includes("bias"), false);
  assertEquals(BMM_WGSL.includes("bias"), false);
  assertNotEquals(LINEAR_WGSL, MATMUL_WGSL);
  // MUST: the 6 ops share the same inner-product loop (= the same reduction order). The output
  // side pins that it exists in only one place (leaving no room for a copying mistake). The static
  // unroll `acc{row}_{column quad}` only makes the index and the row number static; the shared-tile
  // positions read and the add order are the same — exactly one line per row.
  const staticLoop = (row: number): string =>
    `      acc${row}_0 = acc${row}_0 + sa[(lid.y * 8u + ${row}u) * 16u + kk] * bv0;`;
  for (const [where, , wgsl] of GEMM_VARIANTS) {
    for (const line of [0, 1, 2, 3, 4, 5, 6, 7].map(staticLoop)) {
      assertEquals(
        wgsl.split(line).length,
        2,
        `${where}: the inner-product loop is not in exactly one place`,
      );
    }
    assertEquals(wgsl.includes("let tiles = (dims.k + 15u) / 16u;"), true, where);
  }

  // layer_norm has 2 passes (mean → sum of squared deviations), and the variance is the population
  // variance (divided by N)
  assertEquals(LAYER_NORM_WGSL.includes("let d = x[base + j] - mean;"), true);
  assertEquals(LAYER_NORM_WGSL.includes("let scale = 1.0 / f32(dim);"), true);
  assertEquals(LAYER_NORM_WGSL.includes("sqrt(scratch[0u] * scale + params.eps)"), true);

  // rms_norm **does not subtract the mean** (pins by form that layer_norm's 2 passes were not
  // copied)
  assertEquals(RMS_NORM_WGSL.includes("acc = acc + v * v;"), true);
  assertEquals(RMS_NORM_WGSL.includes("mean"), false, "rms_norm has no concept of a mean");
  assertEquals(RMS_NORM_WGSL.includes("bias"), false, "rms_norm has no bias (arity 2)");
  assertEquals(
    RMS_NORM_WGSL.includes("let inv = inverseSqrt(scratch[0u] * scale + params.eps);"),
    true,
  );
  assertEquals(RMS_NORM_WGSL.includes("out[base + o] = x[base + o] * inv * weight[o];"), true);
  assertNotEquals(RMS_NORM_WGSL, LAYER_NORM_WGSL);
  assertNotEquals(RMS_NORM_KEY, LAYER_NORM_KEY);
  assertEquals(RMS_NORM_KEY, `rms_norm:v1:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}`);
  assertEquals(RMS_NORM_WORKGROUP_SIZE, 256);
  // The prefetch form (v2) runs exactly on the multiples of 256 up to the measured width; every
  // other width keeps v1's key and text, and the add form (which has no v1 path left) refuses them.
  for (const dim of [256, 768, 1280, 1536, 2560, 3072, 4096]) {
    assertEquals(rmsNormPrefetchTrip(dim), dim / 256, `trip ${dim}`);
    assertEquals(rmsNormWorkgroupKey(dim), `rms_norm:v2:f32:lastdim:wg256:trip${dim / 256}`);
    const v2 = rmsNormWorkgroupWgsl(dim);
    // the reduction is v1's text, the value v1's expression on the prefetched words
    assertEquals(v2.includes(rmsNormInvWgsl(256)), true, `reduction ${dim}`);
    assertEquals(v2.split("out[base + o] = xv[t] * inv * wv[t];").length, 2, `value ${dim}`);
    assertEquals(v2.includes(`array<f32, ${dim / 256}>`), true, `registers ${dim}`);
  }
  for (const dim of [1, 128, 129, 255, 257, 1000, 4352, 5120]) {
    assertEquals(rmsNormPrefetchTrip(dim), undefined, `trip ${dim}`);
    if (dim > 128) {
      assertEquals(rmsNormWorkgroupKey(dim), RMS_NORM_KEY, `v1 key ${dim}`);
      assertEquals(rmsNormWorkgroupWgsl(dim), RMS_NORM_WGSL, `v1 text ${dim}`);
    }
    assertThrows(() => rmsNormAddKey("norm-residual", dim), CodegenError, "no prefetch form");
    assertThrows(() => rmsNormAddWgsl("residual-norm", dim), CodegenError, "no prefetch form");
  }
  // the add forms keep the rounding barrier on the prefetched words and read the residual register
  assertEquals(
    rmsNormAddWgsl("norm-residual", 1536).includes(
      "out[base + o] = bitcast<f32>(bitcast<u32>(xv[t] * inv * wv[t]) ^ params.rounding_mask) + rv[t];",
    ),
    true,
  );
  assertEquals(
    rmsNormAddWgsl("residual-norm", 1536).includes(
      "out[base + o] = rv[t] + bitcast<f32>(bitcast<u32>(xv[t] * inv * wv[t]) ^ params.rounding_mask);",
    ),
    true,
  );
  // MUST: the output of layer_norm has no rms_norm concept leaked into it (a premise of the
  // determinism snapshots)
  assertEquals(LAYER_NORM_WGSL.includes("inverseSqrt"), false);

  // MUST: softmax is safe-softmax (subtracting amax). The naive form underflows / overflows on
  // large inputs
  assertEquals(SOFTMAX_WGSL.includes("let amax = scratch[0u];"), true);
  assertEquals(SOFTMAX_WGSL.includes("acc = acc + exp(x[base + j] - amax);"), true);
  assertEquals(SOFTMAX_WGSL.includes("out[base + o] = exp(x[base + o] - amax) * inv;"), true);

  // safe_softmax (ADR 0044) shares softmax's ①②③ **verbatim** and adds only the -inf identity, the
  // empty-row test and the select at the write. If this breaks, parity (bit identity with the
  // decomposed-guard equivalent) breaks silently, so the shared fragments are pinned by matching
  // them against the plain softmax side.
  for (const shared of [
    "      hi = nan_max(hi, x[base + i]);",
    "        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);",
    "      acc = acc + exp(x[base + j] - amax);",
    "        scratch[lid] = scratch[lid] + scratch[lid + stride2];",
    "    let inv = 1.0 / scratch[0u];",
  ]) {
    assertEquals(
      SAFE_SOFTMAX_WGSL.includes(shared),
      true,
      `fragment missing on the safe_softmax side: ${shared}`,
    );
    assertEquals(
      SOFTMAX_WGSL.includes(shared),
      true,
      `fragment missing on the softmax side: ${shared}`,
    );
  }
  // The 3 points the variant adds (-inf identity / empty-row test / the select at the write)
  assertEquals(SAFE_SOFTMAX_WGSL.includes("let neg_inf = bitcast<f32>(params.neg_inf);"), true);
  assertEquals(SAFE_SOFTMAX_WGSL.includes("var hi = neg_inf;"), true);
  assertEquals(SAFE_SOFTMAX_WGSL.includes("let empty = row_max == neg_inf;"), true);
  assertEquals(
    SAFE_SOFTMAX_WGSL.includes(
      "out[base + o] = select(exp(x[base + o] - amax) * inv, 0.0, empty);",
    ),
    true,
  );
  // MUST: not a single word of the empty-row concept has leaked into the plain softmax side (the
  // contract stays as ADR 0044 decision 3)
  assertEquals(SOFTMAX_WGSL.includes("neg_inf"), false);
  assertEquals(SOFTMAX_WGSL.includes("empty"), false);
  assertNotEquals(SAFE_SOFTMAX_KEY, SOFTMAX_KEY);
  assertEquals(SOFTMAX_KEY, `softmax:v2:f32:lastdim:safe:wg${SOFTMAX_WORKGROUP_SIZE}`);
  assertEquals(
    SAFE_SOFTMAX_KEY,
    `safe_softmax:v2:f32:lastdim:safe:emptyrow0:wg${SOFTMAX_WORKGROUP_SIZE}`,
  );
  assertEquals(SOFTMAX_WORKGROUP_SIZE, 256);

  // embedding taints out-of-range indices with NaN (the same ruling as gather)
  assertEquals(EMBEDDING_WGSL.includes("if (pick < 0 || u32(pick) >= dims.vocab) {"), true);
  assertEquals(EMBEDDING_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
  // MUST: the NaN bit pattern is carried in params (the same discipline as gather / deform_conv2d).
  assertEquals(
    EMBEDDING_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN is built with a constant expression (carry it via params)",
  );
  assertEquals(EMBEDDING_WGSL.includes("weight[u32(pick) * dims.hidden + col]"), true);

  // masked_fill reads x contiguously, and only the mask goes through strides (right-aligned
  // broadcast)
  assertEquals(
    MASKED_FILL_WGSL.includes("out[i] = select(x[i], fill, mask[mask_index] != 0u);"),
    true,
  );
  assertEquals(
    MASKED_FILL_WGSL.includes(`let fill = bitcast<f32>(params[${1 + 2 * STRIDED_RANK}u]);`),
    true,
  );

  // conv1d "skips" the padding region (not a form that adds 0)
  assertEquals(CONV1D_WGSL.includes("if (ix >= 0 && u32(ix) < dims.length_in) {"), true);
  assertEquals(CONV1D_WGSL.includes("var acc = bias[oc];"), true);
  assertEquals(
    CONV1D_WGSL.includes("let origin = i32(ox * dims.stride) - i32(dims.padding);"),
    true,
  );
  // dilation applies to the kernel position (the mistake of applying it on the output position ox
  // side cannot be told from stride)
  assertEquals(CONV1D_WGSL.includes("let ix = origin + i32(k * dims.dilation);"), true);
  // The weights' second axis is Cin/groups, not Cin (guards against the mistake of skipping weights
  // in depthwise)
  assertEquals(
    CONV1D_WGSL.includes("let w_base = (oc * in_per_group + ic_rel) * dims.kernel;"),
    true,
  );

  // conv2d keeps the 2 spatial axes independent (pins by form that H/W were not collapsed into one)
  assertEquals(
    CONV2D_WGSL.includes("let origin_y = i32(oy * dims.stride_h) - i32(dims.padding_h);"),
    true,
  );
  assertEquals(
    CONV2D_WGSL.includes("let origin_x = i32(ox * dims.stride_w) - i32(dims.padding_w);"),
    true,
  );
  // The padding region is skipped per axis (H with continue, W with a conditional add)
  assertEquals(CONV2D_WGSL.includes("if (iy < 0 || u32(iy) >= dims.height_in) {"), true);
  assertEquals(CONV2D_WGSL.includes("if (ix >= 0 && u32(ix) < dims.width_in) {"), true);
  // The weights are [Cout, Cin/groups, Kh, Kw] (Kw innermost, the row step is Kw)
  assertEquals(
    CONV2D_WGSL.includes(
      "let w_base = (oc * in_per_group + ic_rel) * dims.kernel_h * dims.kernel_w;",
    ),
    true,
  );
  assertEquals(CONV2D_WGSL.includes("let w_row = w_base + kh * dims.kernel_w;"), true);
  assertEquals(CONV2D_WGSL.includes("acc = acc + x[row_base + u32(ix)] * w[w_row + kw];"), true);
  assertEquals(CONV2D_WGSL.includes("var acc = bias[oc];"), true);
  // Every byte of the output is written (a single write with =, not +=)
  assertEquals(CONV2D_WGSL.includes("out[i] = acc;"), true);
  assertNotEquals(CONV2D_WGSL, CONV1D_WGSL);
  assertNotEquals(CONV2D_KEY, CONV1D_KEY);
  // MUST: no 2D concept has leaked into conv1d's output (preserving the existing snapshots)
  assertEquals(CONV1D_WGSL.includes("width"), false);

  // conv_transpose1d is the gather form (1 output element = the sum of the contributing inputs —
  // full-write), and the valid taps are enumerated by a residue split (perf-ledger K-10). MUST: the
  // remainder is computed only once per output element, and it has not gone back to the form of a
  // full scan over k + a divisibility test (the same tap set, but 7/8 of the measured shape would
  // be judged invalid).
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("let r = shifted % dims.stride;"), true);
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("% dims.stride == 0u"), false);
  // The range of j is closed by 3 inequalities (ix <= L-1 / ix >= 0 / k <= K-1)
  assertEquals(
    CONV_TRANSPOSE1D_WGSL.includes("let j_start = max(0, i32(q) + 1 - i32(dims.length_in));"),
    true,
  );
  assertEquals(
    CONV_TRANSPOSE1D_WGSL.includes(
      "j_end = min(i32((dims.kernel - 1u - r) / dims.stride), i32(q));",
    ),
    true,
  );
  // MUST: j ascending = k ascending (the reduction order is the same as the full-scan-over-k
  // version = the ground for f32 bit identity)
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("let k = r + u32(j) * dims.stride;"), true);
  // The weights are [Cin, Cout, K] (transposed from conv1d's [Cout, Cin, K])
  assertEquals(
    CONV_TRANSPOSE1D_WGSL.includes("let w_base = (ic * dims.channels_out + oc) * dims.kernel;"),
    true,
  );
  // Every byte of the output is written (a single write with =, not +=)
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("out[i] = acc;"), true);
});

/**
 * The f16 weight-storage variants (ADR 0018). Pins from the output side that all 5 differ only in
 * "the weights' binding type" and "one read line", and the rest is the same as the f32 variants.
 *
 * MUST: the pair selection is built from the **flat index** (the very expression the f32 variant
 * passes to `w[...]`). The mistake of taking the parity from a relative index within a row still
 * matches numerically in a test with an even row length.
 */
const dropHeader = (wgsl: string): string => wgsl.slice(wgsl.indexOf("\n") + 1);

test("the w=f16 variants read only the weights through array<u32> + unpack2x16float and are otherwise identical to the f32 variants", () => {
  // (f32 variant, f16 variant, weight array name, the read expression(s) in the f32 variant)
  const variants: readonly (readonly [string, string, string, readonly string[]])[] = [
    [
      LINEAR_WGSL,
      linearWgsl("f16", false),
      "w",
      ["wbase", "wbase + 1u", "wbase + 2u", "wbase + 3u"],
    ],
    [EMBEDDING_WGSL, embeddingWgsl("f16"), "weight", ["u32(pick) * dims.hidden + col"]],
    [CONV1D_WGSL, conv1dWgsl("f16"), "w", ["w_base + k"]],
    [CONV2D_WGSL, conv2dWgsl("f16"), "w", ["w_row + kw"]],
    [CONV_TRANSPOSE1D_WGSL, convTranspose1dWgsl("f16"), "w", ["w_base + k"]],
  ];
  for (const [f32Wgsl, f16Wgsl, name, indices] of variants) {
    const where = `${name} / ${indices.join(" , ")}`;
    assertNotEquals(f16Wgsl, f32Wgsl, where);
    // Only the weights are a u32 binding (input, bias and output stay f32)
    assertEquals(f16Wgsl.includes(`read> ${name}: array<u32>;`), true, where);
    assertEquals(f16Wgsl.includes(`read> ${name}: array<f32>;`), false, where);
    assertEquals(f16Wgsl.includes("read_write> out: array<f32>;"), true, where);
    // The unpack is one unpack2x16float (no optional-feature enable is used — ADR 0018)
    assertEquals(f16Wgsl.includes(`let pair = unpack2x16float(${name}[i >> 1u]);`), true, where);
    assertEquals(f16Wgsl.includes("return select(pair.x, pair.y, (i & 1u) == 1u);"), true, where);
    assertEquals(f16Wgsl.includes("enable f16"), false, where);
    // MUST: the read passes **the same index expression** as the f32 variant to the unpack function
    // (the pair is chosen by the parity of the flat index)
    let rewound = f16Wgsl
      .replace(/\/\/ decode of f16 storage:[\s\S]*?\n}\n\n/, "")
      .replace(`read> ${name}: array<u32>;`, `read> ${name}: array<f32>;`);
    for (const index of indices) {
      assertEquals(f32Wgsl.includes(`${name}[${index}]`), true, `${where}: the f32 variant's read`);
      assertEquals(f16Wgsl.includes(`dequant(${index})`), true, where);
      assertEquals(f16Wgsl.includes(`${name}[${index}]`), false, where);
      // MUST: undo every occurrence (the GEMM skeleton has one thread fill several channels, so the
      // same index expression appears once per fill slot)
      rewound = rewound.replaceAll(`dequant(${index})`, `${name}[${index}]`);
    }
    // These 3 points are the only difference (the proviso in the header comment / the unpack
    // function / one line each for the binding and the read). Applying the inverse mapping to get
    // back exactly the f32 variant pins that "nothing else moved at all".
    assertEquals(dropHeader(rewound), dropHeader(f32Wgsl), where);
  }
});

/**
 * The **quad unpack** the GEMM v4 path uses (it keeps the semantics of ADR 0018 / 0019 and only
 * groups the read instructions).
 *
 * MUST: the quad version depends on "the flat index being a multiple of 4" (not on the row head
 * falling on a word boundary). The scalar path's detectors (f16 = row length not a multiple of 2 /
 * i8 = not a multiple of 4) are intact, and the v4 path avoids that trap but cannot detect it
 * either — **not removing the scalar-side tests** is the condition for detection power.
 */
test("linear's v4 variants read the weights through a quad unpack and are otherwise identical to the f32 v4 variant", () => {
  const f32Wgsl = linearWgsl("f32", true);
  // f32 uses quad indices into a vec4<f32> array (flat index >> 2)
  assertEquals(f32Wgsl.includes("read> w: array<vec4<f32>>;"), true);
  assertEquals(f32Wgsl.includes("wv0 = w[(wrow_base0 + wk0) >> 2u];"), true);

  const f16Wgsl = linearWgsl("f16", true);
  // f16 is 4 elements in 2 u32 words (it does not straddle words). The scalar dequant is not
  // emitted
  assertEquals(f16Wgsl.includes("let lo = unpack2x16float(w[i >> 1u]);"), true);
  assertEquals(f16Wgsl.includes("let hi = unpack2x16float(w[(i >> 1u) + 1u]);"), true);
  assertEquals(f16Wgsl.includes("return vec4<f32>(lo.x, lo.y, hi.x, hi.y);"), true);
  assertEquals(f16Wgsl.includes("fn dequant(i: u32)"), false, "the scalar version is not emitted");
  assertEquals(f16Wgsl.includes("wv0 = dequant4(wrow_base0 + wk0);"), true);

  const i8Wgsl = linearWgsl("i8", true);
  // MUST: the scale is a **per-component f32 multiplication** (vec4 * scalar) — the same rounding
  // as the scalar path (ADR 0019). A form that moves it outside the reduction (acc * scale) would
  // trigger the revision of ADR 0019 and a re-derivation of the tolerance.
  assertEquals(i8Wgsl.includes("return vec4<f32>(unpack4xI8(w[i >> 2u])) * scale;"), true);
  // MUST: bind the scale under a separate name per fill slot (using slot 0's scale for the second
  // one is a silent wrong value where only one of the assigned channels is dequantized with
  // another channel's scale)
  assertEquals(
    i8Wgsl.includes("let wscale_v = wscale[min(wcol0, dims.n - 1u)];"),
    true,
    "the scale is outside the K loop",
  );
  assertEquals(
    i8Wgsl.includes("let wscale_v1 = wscale[min(wcol1, dims.n - 1u)];"),
    true,
    "a separate name per slot",
  );
  assertEquals(i8Wgsl.includes("wv0 = dequant4(wrow_base0 + wk0, wscale_v);"), true);
  assertEquals(i8Wgsl.includes("wv1 = dequant4(wrow_base1 + wk0, wscale_v1);"), true);
  // acc is the static unroll `acc{row}_{column quad}`, so a form that multiplies after the
  // reduction is checked by this name
  assertEquals(
    i8Wgsl.includes("acc0_0 * "),
    false,
    "the scale is not applied outside the reduction",
  );

  // The inverse mapping gets back exactly the f32 v4 variant (nothing else moved at all)
  const scaleArg = (slot: number): string => (slot === 0 ? "wscale_v" : `wscale_v${slot}`);
  for (const [wgsl, call] of [
    [f16Wgsl, (slot: number) => `dequant4(wrow_base${slot} + wk0)`],
    [i8Wgsl, (slot: number) => `dequant4(wrow_base${slot} + wk0, ${scaleArg(slot)})`],
  ] as const) {
    let rewound = wgsl
      .replace(/\n@group\(0\) @binding\(5\) var<storage, read> wscale: array<f32>;\n/, "")
      .replace(/\/\/ quad decode of (f16|i8) storage:[\s\S]*?\n}\n\n/, "")
      .replaceAll(
        /\n {2}\/\/ the output channel's scale is loop-invariant[\s\S]*?\n {2}let wscale_v\d* = wscale\[min\(wcol\d, dims\.n - 1u\)\];/g,
        "",
      )
      .replace("read> w: array<u32>;", "read> w: array<vec4<f32>>;");
    for (const slot of [0, 1]) {
      rewound = rewound.replace(call(slot), `w[(wrow_base${slot} + wk0) >> 2u]`);
    }
    assertEquals(dropHeader(rewound), dropHeader(f32Wgsl), call(0));
  }
});

/**
 * The f16 **compute** variants (ADR 0028). Pins the rounding sequence from the output side — the
 * numeric contract "f16 variant = the f32 variant with its inputs rounded to f16" rests on these
 * 4 points alone:
 *
 * 1. Rounding happens **in one place, the write to the shared tile** (`f16(...)` /
 *    `vec4<f16>(...)`).
 * 2. Widening happens **once at the register load** (a per-MAC `f32(av * bv)` is forbidden — a
 *    negative lesson from the prototype: it doubles the conversions and drops the product to f16
 *    precision).
 * 3. Accumulation is f32 (the static unroll `acc{row}_{column quad}`, with the element type staying
 *    `vec4<f32>`).
 * 4. The uniform is not made f16.
 */
const COMPUTE_F16_VARIANTS: readonly (readonly [string, string])[] = [
  ["attention_qk c16", attentionQkWgsl(false, "f16")],
  ["attention_qk c16 v4", attentionQkWgsl(true, "f16")],
  ["attention_pv c16", attentionPvWgsl(false, "f16")],
  ["attention_pv c16 v4", attentionPvWgsl(true, "f16")],
  ["linear c16", linearWgsl("f32", false, "f16")],
  ["linear c16 v4", linearWgsl("f32", true, "f16")],
  ["linear wf16 c16", linearWgsl("f16", false, "f16")],
  ["linear wf16 c16 v4", linearWgsl("f16", true, "f16")],
];

test("the f16 compute variants make only the shared tiles f16 and confine rounding and widening to one place each", () => {
  for (const [where, wgsl] of COMPUTE_F16_VARIANTS) {
    const lines = wgsl.split("\n");
    // enable is at the head of the module (right after the one comment line = before every global
    // declaration)
    assertEquals(lines[0].startsWith("// karume "), true, where);
    assertEquals(lines[1], "enable f16;", where);
    // ① Only the shared tiles are f16 (halving the bytes is this variant's only mechanism; the edge
    // is the default geometry 128×128)
    assertEquals(wgsl.includes("var<workgroup> sa: array<f16, 2048>;"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f16>, 512>;"), true, where);
    // ③ Accumulation stays f32 (avoids cancellation for large k — changing it turns parity and E2E
    // red)
    assertEquals(wgsl.includes("var acc0_0 = vec4<f32>(0.0);"), true, where);
    // ② Widening happens once at the register load. A per-MAC product conversion (f32(av * bv)) is
    // not emitted
    assertEquals(wgsl.includes("let bv0 = vec4<f32>(sb[kk * 32u + lid.x * 2u]);"), true, where);
    assertEquals(
      wgsl.includes("acc0_0 = acc0_0 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv0;"),
      true,
      where,
    );
    // MUST: the check runs on the body without comments (the comments describe the forbidden form)
    const code = wgsl
      .split("\n")
      .filter((line) => !line.trimStart().startsWith("//"))
      .join("\n");
    assertEquals(code.includes("f32(av"), false, `${where}: per-MAC conversion`);
    // ① Rounding happens only in the write to the shared tile (the A tile wraps all 4 components in
    // f16(...))
    assertEquals(wgsl.includes("sa[sa_base0] = f16(av0.x);"), true, where);
    assertEquals(wgsl.includes("sa[sa_base0 + 3u] = f16(av0.w);"), true, where);
    assertEquals(wgsl.includes("sa[sa_base1] = f16(av1.x);"), true, where);
    // ④ The uniform is not made f16 (uniform array elements are 16B-aligned, and f16 would gain
    // nothing)
    assertEquals(wgsl.includes("var<uniform> dims: Dims;"), true, where);
    assertEquals(/^\s+[mnk]: f16,$/m.test(wgsl), false, `${where}: the uniform is f16`);
  }
  // Fused attention's S is passed as f16 (① writes, ②③ read = the transient is halved)
  assertEquals(
    attentionQkWgsl(true, "f16").includes("s[cbase + orow0 * n4 + ocq0] = vec4<f16>(acc0_0);"),
    true,
  );
  assertEquals(attentionQkWgsl(false, "f16").includes("s[obase + ocol] = f16(acc0_0.x);"), true);
  assertEquals(
    attentionPvWgsl(false, "f16").includes(
      "av0.x = exp(f32(s[arow_base0 + ak0]) - row_max0) * row_inv0;",
    ),
    true,
  );
  // ② The row statistics also widen S to f32 before the reduction (the reduction's rounding
  // sequence is the same as the f32 variant)
  const stats16 = attentionStatsWgsl("f16");
  assertEquals(stats16.includes("var<storage, read> s: array<f16>;"), true);
  assertEquals(stats16.includes("hi = nan_max(hi, f32(s[base + i]));"), true);
  assertEquals(stats16.includes("acc = acc + exp(f32(s[base + j]) - amax);"), true);
  assertEquals(
    stats16.includes("var<workgroup> scratch: array<f32, 256>;"),
    true,
    "the reduction is f32",
  );
});

test("the default (f32 compute) output contains not a single character of f16", () => {
  // f16 storage (ADR 0018) has "f16" in its proviso comment, so this checks the variants with **f32
  // storage too**.
  const defaults: readonly (readonly [string, string])[] = [
    ["attention_qk", attentionQkWgsl(false)],
    ["attention_qk v4", attentionQkWgsl(true)],
    ["attention_pv", attentionPvWgsl(false)],
    ["attention_pv v4", attentionPvWgsl(true)],
    ["attention_stats", ATTENTION_STATS_WGSL],
    ["linear", linearWgsl("f32", false)],
    ["linear v4", linearWgsl("f32", true)],
    ["matmul", MATMUL_WGSL],
    ["bmm", BMM_WGSL],
  ];
  for (const [where, wgsl] of defaults) {
    assertEquals(wgsl.includes("f16"), false, where);
  }
});

test("the compute variant's discriminator is attached only on the f16 side of the key and is a different word from storage's wf16", () => {
  // The compute variant's word comes **after** the geometry discriminator (the word order of the
  // key itself is pinned)
  assertEquals(linearKey("f32", false), "linear:v2:f32:reg128x128r8x8w16");
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg128x128r8x8w16v4:wf16");
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  // f16 compute appends :c16 (a word that does not overlap storage's :wf16 — some combinations set
  // both)
  assertEquals(linearKey("f32", false, "f16"), "linear:v2:f32:reg128x128r8x8w16:c16");
  assertEquals(linearKey("f16", true, "f16"), "linear:v2:f32:reg128x128r8x8w16v4:wf16:c16");
  assertEquals(attentionQkKey(false, "f16"), "attention_qk:v1:f32:reg128x128r8x8w16:c16");
  assertEquals(attentionPvKey(true, "f16"), "attention_pv:v1:f32:reg128x128r8x8w16v4:c16");
  assertEquals(attentionStatsKey("f16"), `${ATTENTION_STATS_KEY}:c16`);
});

// A dynamic index `acc[i]` requires an addressable function-local area, and the accumulators drop
// from registers into local memory. **All 6 ops that share the skeleton** use this static unroll
// on the default path.
test("the GEMM skeleton's accumulators are the static unroll acc{row}_{column quad} and leave no dynamic index", () => {
  for (const [where, , wgsl] of GEMM_VARIANTS) {
    assertEquals(wgsl.includes("var acc = array<"), false, where);
    // MUST: not a single dynamic index remains (if one did, the motivation for the unroll would
    // vanish)
    assertEquals(wgsl.includes("acc["), false, where);
    for (let row = 0; row < 8; row += 1) {
      // The inner-product update of row `row` happens exactly once (K tile 16, kk ascending, the
      // add order the same as before the unroll)
      assertEquals(
        wgsl.split(`acc${row}_0 = acc${row}_0 + sa[(lid.y * 8u + ${row}u) * 16u + kk] * bv0;`)
          .length,
        2,
        `${where}:update${row}`,
      );
      // The write also reads a static name per row
      assertEquals(wgsl.includes(`orow${row} < dims.m`), true, `${where}:store${row}`);
    }
    // There are 2 column quads (regN = 8 = vec4 × 2) — a third one would mean the geometry changed
    assertEquals(wgsl.includes("acc0_1"), true, `${where}: there are not 2 column quads`);
    assertEquals(wgsl.includes("acc0_2"), false, `${where}: there are 3 column quads`);
  }
  // Zero initialization is on every op but bias-first conv2d (a separate test checks conv2d's bias
  // initial value)
  for (const wgsl of [matmulWgsl(true), linearWgsl("f32", false), attentionPvWgsl(true)]) {
    assertEquals(wgsl.includes("var acc0_0 = vec4<f32>(0.0);"), true);
    assertEquals(wgsl.includes("var acc7_0 = vec4<f32>(0.0);"), true);
  }
});

test("i8 weight storage × f16 compute (w8a16) fails loudly at the generator entry", () => {
  assertThrows(() => linearWgsl("i8", true, "f16"), CodegenError, "w8a16");
  // Anything but that combination passes (i8 × f32 compute / f16 storage × f16 compute)
  linearWgsl("i8", true);
  linearWgsl("f16", true, "f16");
});

/**
 * The **f16 storage** variants of S (option γ iteration 1). Pins from the output side the numeric
 * contract "s16 variant = the f32 variant with S rounded to f16 on the host". It holds under 3
 * conditions:
 *
 * 1. Rounding happens **only at the one `pack2x16float`** (`score_store`). The reader's
 *    `unpack2x16float` is exact, so it is not a rounding.
 * 2. **No `enable f16` is emitted** (core WGSL — the output-side evidence that ADR 0030 decision 1,
 *    "i8a8 does not require shader-f16", stays intact when combined with s16).
 * 3. The writer is **the v4 path only** (one thread writes 4 consecutive columns = exactly 2 words,
 *    exclusively).
 */
const SCORE_F16_VARIANTS: readonly (readonly [string, string, "read" | "write"])[] = [
  ["attention_qk s16 v4", attentionQkWgsl(true, "f32", "f16"), "write"],
  ["attention_pv s16 v4", attentionPvWgsl(true, "f32", "f16"), "read"],
  ["attention_stats s16", attentionStatsWgsl("f32", "f16"), "read"],
  ["attention_qk i8a8 s16 v4", attentionQkI8a8Wgsl(true, true, "f16"), "write"],
  ["attention_qk i8a8 emu s16 v4", attentionQkI8a8Wgsl(true, false, "f16"), "write"],
  ["attention_pv i8a8 s16 v4", attentionPvI8a8Wgsl(true, true, "f16"), "read"],
  ["attention_pv i8a8 emu s16 v4", attentionPvI8a8Wgsl(true, false, "f16"), "read"],
];

test("the f16 storage variants of S round only at one pack2x16float and do not require shader-f16", () => {
  for (const [where, wgsl, side] of SCORE_F16_VARIANTS) {
    // ② It stays core WGSL (feature-independent — if this breaks, i8a8 gets tied to the feature)
    assertEquals(wgsl.includes("enable f16;"), false, `${where}: enable f16 is emitted`);
    // The S binding is always a flat u32 array (if it were left as vec4<f32> / f16, the types would
    // not match)
    assertEquals(
      /var<storage, read(_write)?> s: array<u32>;/.test(wgsl),
      true,
      `${where}: S binding`,
    );
    // MUST: the check runs on the body without comments (the comments describe the forbidden form)
    const code = wgsl
      .split("\n")
      .filter((line) => !line.trimStart().startsWith("//"))
      .join("\n");
    // MUST: count only `pack2x16float` without `un` (a partial match that catches unpack would fill
    // the claim "rounding in one place" with the reader's count and make it tautological)
    const packs = code.match(/(?<!un)pack2x16float\(/g) ?? [];
    const unpacks = code.match(/unpack2x16float\(/g) ?? [];
    if (side === "write") {
      // ① Rounding is only the pack of 2 words (the 2 lines of `score_store`). There is not a
      // single read
      assertEquals(packs.length, 2, `${where}: number of pack2x16float`);
      assertEquals(unpacks.length, 0, `${where}: the writer unpacks`);
      assertEquals(
        code.includes("  s[w] = pack2x16float(value.xy);\n  s[w + 1u] = pack2x16float(value.zw);"),
        true,
        `${where}: the 2 words are not written consecutively`,
      );
      // ③ Only the quad write of the v4 path calls score_store (no guard for partial writes
      // remains)
      assertEquals(
        code.includes("score_store(") && !/if \(ocol( \+ \d+u)? < dims\.n\)/.test(code),
        true,
        `${where}: a partial write of the scalar path remains`,
      );
    } else {
      assertEquals(packs.length, 0, `${where}: the reader packs`);
      assertEquals(unpacks.length > 0, true, `${where}: no unpack2x16float`);
    }
  }
  // The reader's index arithmetic keeps the same form as the f32 variant (quads stay quads,
  // elements stay elements)
  assertEquals(
    attentionPvWgsl(true, "f32", "f16").includes(
      "let raw0 = score_quad(arow_base0 + t * 4u + aq);",
    ),
    true,
  );
  assertEquals(
    attentionPvI8a8Wgsl(true, true, "f16").includes("let raw0 = score_quad(arow_base0 + apack);"),
    true,
  );
  assertEquals(
    attentionStatsWgsl("f32", "f16").includes("hi = nan_max(hi, score_at(base + i));"),
    true,
  );
  // The writer keeps the same value expression as the f32 variant (rounding has not slipped into
  // the value computation)
  assertEquals(
    attentionQkWgsl(true, "f32", "f16").includes("score_store(cbase + orow0 * n4 + ocq0, acc0_0);"),
    true,
  );
});

test("the stored-S discriminator s16 goes at the end of the key and is a different word from wf16 / c16", () => {
  // The default (f32 storage) key ends at the geometry discriminator
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionPvKey(true), "attention_pv:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  assertEquals(
    attentionQkI8a8Key(true, true),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a",
  );
  assertEquals(
    attentionPvI8a8Key(true, false),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4aEmu",
  );
  // s16 goes **at the end** (the 3 axes can be set together, so the word order is pinned in one
  // place)
  assertEquals(attentionQkKey(true, "f32", "f16"), "attention_qk:v1:f32:reg128x128r8x8w16v4:s16");
  assertEquals(attentionPvKey(true, "f32", "f16"), "attention_pv:v1:f32:reg128x128r8x8w16v4:s16");
  assertEquals(attentionStatsKey("f32", "f16"), `${ATTENTION_STATS_KEY}:s16`);
  assertEquals(
    attentionQkI8a8Key(true, true, "f16"),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a:s16",
  );
  assertEquals(
    attentionPvI8a8Key(true, false, "f16"),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4aEmu:s16",
  );
  // The 3 words (weight storage wf16 / compute c16 / S storage s16) do not collide with each other
  assertEquals(new Set(["wf16", "c16", "s16"]).size, 3);
  // One configuration never has 2 keys (every attention key that can be generated is unique)
  const keys = [
    ...[false, true].flatMap((v4) => [
      attentionQkKey(v4),
      attentionPvKey(v4),
      attentionQkKey(v4, "f16"),
      attentionPvKey(v4, "f16"),
      ...[true, false].flatMap((dp4a) => [
        attentionQkI8a8Key(v4, dp4a),
        attentionPvI8a8Key(v4, dp4a),
      ]),
    ]),
    attentionQkKey(true, "f32", "f16"),
    attentionPvKey(true, "f32", "f16"),
    ...[true, false].flatMap((dp4a) => [
      attentionQkI8a8Key(true, dp4a, "f16"),
      attentionPvI8a8Key(true, dp4a, "f16"),
    ]),
    attentionStatsKey(),
    attentionStatsKey("f16"),
    attentionStatsKey("f32", "f16"),
  ];
  assertEquals(new Set(keys).size, keys.length, "duplicate attention keys");
});

test('f16 storage of S fails loudly at the generator entry for "the scalar path" and "together with c16"', () => {
  // A partial write on the scalar path would be a read-modify-write on the same u32 word (it does
  // not silently fall back to f32 storage)
  for (const [where, generate] of [
    ["attention_qk", () => attentionQkWgsl(false, "f32", "f16")],
    ["attention_pv", () => attentionPvWgsl(false, "f32", "f16")],
    ["attention_qk i8a8", () => attentionQkI8a8Wgsl(false, true, "f16")],
  ] as const) {
    assertThrows(generate, CodegenError, "is for the v4 path only", where);
  }
  // `:c16` is another storage form that holds S as array<f16> (a redundant and contradictory
  // combination)
  for (const [where, generate] of [
    ["attention_qk", () => attentionQkWgsl(true, "f16", "f16")],
    ["attention_pv", () => attentionPvWgsl(true, "f16", "f16")],
    ["attention_stats", () => attentionStatsWgsl("f16", "f16")],
  ] as const) {
    assertThrows(generate, CodegenError, "cannot be given together", where);
  }
  // The eligibility decision is the very condition under which the writer takes v4 (**both** D%4
  // and N%4)
  for (const [depth, cols, expected] of [
    [4, 8, true],
    [128, 4096, true],
    [4, 6, false],
    [13, 8, false],
    [13, 19, false],
  ] as const) {
    assertEquals(attentionScoreUsesF16(depth, cols), expected, `D=${depth} N=${cols}`);
    // At eligible shapes both the f32 ①QK and the i8a8 ①QK take v4 (the premise that no degenerate
    // mix occurs)
    if (expected) {
      assertEquals(gemmUsesVec4(depth, cols), true, `D=${depth} N=${cols}: f32 ①QK is v4`);
      assertEquals(attentionQkI8a8UsesVec4(cols), true, `D=${depth} N=${cols}: i8a8 ①QK is v4`);
      assertEquals(gemmUsesVec4(cols, depth), true, `D=${depth} N=${cols}: f32 ③PV is v4`);
      assertEquals(attentionPvI8a8UsesVec4(depth), true, `D=${depth} N=${cols}: i8a8 ③PV is v4`);
    }
  }
});

test("the storage discriminator is attached only on the f16 side of the key (the existing f32 keys are byte-unchanged)", () => {
  const pairs: readonly (readonly [string, (weight: "f32" | "f16") => string])[] = [
    ["linear:v2:f32:reg128x128r8x8w16", (weight) => linearKey(weight, false)],
    ["embedding:v1:f32:i32:wg256", embeddingKey],
    ["conv1d:v2:f32:direct:wg256", conv1dKey],
    ["conv2d:v1:f32:direct:wg256", conv2dKey],
    ["conv_transpose1d:v2:f32:gather:wg256", convTranspose1dKey],
  ];
  for (const [expected, key] of pairs) {
    // MUST: the f32 key strings are exactly the same before and after the variants were introduced
    // (pinned by writing the actual keys literally)
    assertEquals(key("f32"), expected);
    assertEquals(key("f16"), `${expected}:wf16`);
  }
  // The storage discriminator comes **after** the v4 discriminator (only linear has the 2 axes)
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg128x128r8x8w16v4:wf16");
  // i4 puts the group length in the key (paired with the baked shift — ADR 0069)
  assertEquals(
    linearKey("i4", true, "f32", undefined, 32),
    "linear:v2:f32:reg128x128r8x8w16v4:wi4g32",
  );
  assertEquals(embeddingKey("i4", 32), "embedding:v1:f32:i32:wg256:wi4g32");
  // The group length is one-to-one with i4 (missing / extra are both wiring bugs — letting it pass
  // silently would set up a key without the g part, and assets of different groups would map to
  // the same pipeline)
  assertThrows(() => linearKey("i4", false), CodegenError, "must be passed together with");
  assertThrows(
    () => linearKey("f32", false, "f32", undefined, 32),
    CodegenError,
    "must be passed together with",
  );
  assertThrows(() => linearKey("i4", false, "f32", undefined, 24), CodegenError, "power of two");
  // The same discipline applies to embedding (i4 has 2 generator entries, so it is restated for
  // both)
  assertThrows(() => embeddingKey("i4"), CodegenError, "must be passed together with");
  assertThrows(() => embeddingKey("i8", 32), CodegenError, "must be passed together with");
  assertThrows(() => embeddingKey("i4", 8), CodegenError, "power of two");
});

/**
 * onv1d's i4 variant (ADR 0069 decision 5's conv1d addendum) is **limited to the implicit GEMM**.
 *
 * The key carries the group part (the same convention as linear — paired with the baked shift) The
 * group length is one-to-one with i4 (missing / extra are both wiring bugs) The direct kernel
 * (groups > 1) and conv2d have no unpack path, and **the generator entry rejects them** — not
 * relying on the eligibility decision (plan.ts) being closed, it also blocks the direct-call path
 */
test("conv1d's i4 is limited to the implicit GEMM, and the key carries the group part", () => {
  // ① The f32 / i8 keys are exactly the same before and after the variant was introduced (pinned by
  // writing the actual keys literally)
  assertEquals(conv1dIgemmKey("f32", false), "conv1d:v4:f32:igemm64x128:wg16x8");
  assertEquals(conv1dIgemmKey("i8", true), "conv1d:v4:f32:igemm64x128v4:wg16x8:wi8");
  assertEquals(
    conv1dIgemmKey("i4", true, undefined, 32),
    "conv1d:v4:f32:igemm64x128v4:wg16x8:wi4g32",
  );
  // A different group length gives a different key (blocks the silent wrong value of the same WGSL
  // running on assets of another group)
  assertNotEquals(
    conv1dIgemmKey("i4", false, GEMM_MTILE_SMALL, 16),
    conv1dIgemmKey("i4", false, GEMM_MTILE_SMALL, 32),
  );
  // ② i4 and the group length are one-to-one (the key and the generation go through the same check)
  assertThrows(() => conv1dIgemmKey("i4", false), CodegenError, "must be passed together with");
  assertThrows(
    () => conv1dIgemmKey("i8", false, GEMM_TILE, 32),
    CodegenError,
    "must be passed together with",
  );
  assertThrows(() => conv1dIgemmKey("i4", false, GEMM_TILE, 24), CodegenError, "power of two");
  assertThrows(() => conv1dIgemmWgsl("i4", false), CodegenError, "must be passed together with");
  // ③ The direct kernel (for groups > 1) does not accept i4 — it emits neither a key nor an output
  assertThrows(() => conv1dKey("i4"), CodegenError, "i4");
  assertThrows(() => conv1dWgsl("i4"), CodegenError, "i4");
  // ③' The same goes for conv2d's implicit GEMM (the A-side unpacker is shared by 1D / 2D, so
  // **generation would go through** — without rejecting it, indices without a group scale would hit
  // the packed bytes)
  assertThrows(() => conv2dIgemmWgsl("i4", false), CodegenError, "i4");
  // The i4 output fetches the group scale per quad in the A-side (weight-side) fill
  const wgsl = conv1dIgemmWgsl("i4", false, undefined, 32);
  assertEquals(wgsl.includes("@group(0) @binding(5) var<storage, read> wscale: array<f32>;"), true);
  assertEquals(wgsl.includes("let ags0 = wscale[arow0 * (dims.k >> 5u) + (ak0 >> 5u)];"), true);
  // MUST: do not bring in linear's (B tile = columns) index — in conv the weights are on the A side
  // (rows = Cout)
  assertEquals(wgsl.includes("wcol"), false);
});

/**
 * he remaining generator entries without an i4 unpack path (conv2d's direct kernel /
 * onv_transpose1d) are also **rejected at the generator entry** (the MUST in weight-storage.ts:
 * each generator function rejects the paths that pass i4 to the remaining generator entries").
 *
 * ithout the rejection, `weightScaleWgsl` returns an empty string for i4, so `dequant(…,
 * wscale_v)` s emitted with `wscale_v` unbound, and the **invalid WGSL with an undefined
 * identifier** breaks n a distant shader compilation (the diagnostic ends up far from the
 * generator entry). The 3 ariants f32 / f16 / i8 pass as before.
 */
test("conv2d direct / conv_transpose1d, which have no i4 unpack path, reject it at the generator entry", () => {
  assertThrows(() => conv2dKey("i4"), CodegenError, "i4");
  assertThrows(() => conv2dWgsl("i4"), CodegenError, "i4");
  assertThrows(() => convTranspose1dKey("i4"), CodegenError, "i4");
  assertThrows(() => convTranspose1dWgsl("i4"), CodegenError, "i4");
  // The check rejects only i4 (the 3 shared variants pass through — keys and outputs keep their
  // existing form)
  for (const storage of WEIGHT_STORAGES) {
    const suffix =
      storage === "f32" ? "" : storage === "f16" ? ":wf16" : storage === "bf16" ? ":wbf16" : ":wi8";
    assertEquals(conv2dKey(storage), `conv2d:v1:f32:direct:wg256${suffix}`);
    assertEquals(convTranspose1dKey(storage), `conv_transpose1d:v2:f32:gather:wg256${suffix}`);
    assertEquals(conv2dWgsl(storage).includes("@compute @workgroup_size(256)"), true, storage);
    assertEquals(
      convTranspose1dWgsl(storage).includes("@compute @workgroup_size(256)"),
      true,
      storage,
    );
  }
});

/**
 * The quad extension of weight-storage is **opt-in** (the 4th / 2nd argument of
 * {@link weightLoaderWgsl} / {@link weightArrayType}).
 *
 * MUST: the outputs of the 4 existing ops (conv1d / conv2d / conv_transpose1d / embedding) do not
 * move by a single byte. Adding `dequant4` unconditionally would move all the fixtures at once and
 * break the identity of the pipeline cache (same key ⇔ same bytes) too.
 */
test("the quad extension of weight-storage does not move the outputs of the 4 existing ops by a single byte", async () => {
  const existing: readonly (readonly [string, string])[] = WEIGHT_STORAGES.flatMap((weight) => {
    const suffix =
      weight === "f32" ? "" : weight === "f16" ? "_wf16" : weight === "bf16" ? "_wbf16" : "_wi8";
    return [
      [`conv1d${suffix}.wgsl`, conv1dWgsl(weight)] as const,
      [`conv2d${suffix}.wgsl`, conv2dWgsl(weight)] as const,
      [`conv_transpose1d${suffix}.wgsl`, convTranspose1dWgsl(weight)] as const,
      [`embedding${suffix}.wgsl`, embeddingWgsl(weight)] as const,
    ];
  });
  assertEquals(existing.length, 16);
  for (const [name, generated] of existing) {
    assertEquals(
      generated,
      await fixture(name),
      `${name}: differs from the bytes before the extension`,
    );
    // The parts of the quad version have not leaked (a leak would also fail the comparison above,
    // but this names the cause)
    assertEquals(generated.includes("dequant4"), false, `${name}: the quad unpack has leaked`);
    assertEquals(generated.includes("vec4<f32>"), false, `${name}: the quad binding has leaked`);
  }
});

test("the params of the fused kernels fail loudly on values outside the contract", () => {
  assertThrows(() => linearParams(-1, 2, 3), CodegenError);
  assertEquals([...linearParams(5, 3, 7)], [5, 3, 7, 0]);

  // eps is carried as an f32 bit pattern (written as u32, the exponent would turn into an integer
  // value)
  const norm = layerNormParams(4, 8, 1e-7);
  assertEquals([norm[0], norm[1]], [4, 8]);
  assertEquals(new Float32Array(norm.buffer)[2], Math.fround(1e-7));
  assertThrows(() => layerNormParams(4, 0, 1e-7), CodegenError);
  assertThrows(() => layerNormParams(4, 8, 0), CodegenError);
  assertThrows(() => layerNormParams(4, 8, Number.POSITIVE_INFINITY), CodegenError);
  // MUST: the value range is checked **as f32** (even when finite and positive in f64, dropping to
  // an f32 word can turn it into ±0 / ±Inf). An overshoot of 1e39 is the entry point of a silent
  // wrong value where only the GPU turns all bias via `1/sqrt(x + Inf) = 0`.
  assertThrows(() => layerNormParams(4, 8, 1e39), CodegenError, "finite positive f32 value");
  assertThrows(() => layerNormParams(4, 8, 1e-50), CodegenError, "finite positive f32 value");
  // A normal eps passes (the counterpart showing the check is not too wide)
  assertEquals(new Float32Array(layerNormParams(4, 8, 1e-5).buffer)[2], Math.fround(1e-5));

  // rms_norm's eps is also an f32 bit pattern (the same convention as layer_norm)
  const rms = rmsNormParams(6, 10, 1e-6);
  assertEquals([rms[0], rms[1]], [6, 10]);
  assertEquals(new Float32Array(rms.buffer)[2], Math.fround(1e-6));
  assertEquals(rms.byteLength % 16, 0);
  assertThrows(() => rmsNormParams(6, 0, 1e-6), CodegenError);
  assertThrows(() => rmsNormParams(6, 10, 0), CodegenError);
  assertThrows(() => rmsNormParams(6, 10, Number.NaN), CodegenError);
  assertThrows(() => rmsNormParams(6, 10, 1e39), CodegenError, "finite positive f32 value");
  assertThrows(() => rmsNormParams(6, 10, 1e-50), CodegenError, "finite positive f32 value");
  assertEquals(new Float32Array(rmsNormParams(6, 10, 1e-5).buffer)[2], Math.fround(1e-5));

  assertEquals([...softmaxParams(4, 9)], [4, 9, 0, 0]);
  assertThrows(() => softmaxParams(4, 0), CodegenError);
  // safe_softmax puts the -inf bit pattern in the 3rd word (WGSL has no infinity literal)
  assertEquals([...softmaxParams(4, 9, true)], [4, 9, SAFE_SOFTMAX_NEG_INF_BITS, 0]);
  assertEquals(new Float32Array(softmaxParams(4, 9, true).buffer)[2], Number.NEGATIVE_INFINITY);
  assertThrows(() => softmaxParams(4, 0, true), CodegenError);

  assertEquals([...embeddingParams(8, 2, 7)], [8, 2, 7, EMBEDDING_OOB_BITS]);
  assertThrows(() => embeddingParams(6, 0, 7), CodegenError);
  assertThrows(() => embeddingParams(-1, 2, 7), CodegenError);

  // A missing rank is left-padded with dims=1 / strides=0 (the same padding as the strided family)
  const filled = maskedFillParams([3, 4], [0, 1], -3.4028234663852886e38);
  assertEquals([...filled.slice(1, 1 + STRIDED_RANK)], [1, 1, 3, 4]);
  assertEquals([...filled.slice(1 + STRIDED_RANK, 1 + 2 * STRIDED_RANK)], [0, 0, 0, 1]);
  assertEquals(new Float32Array(filled.buffer)[1 + 2 * STRIDED_RANK], -3.4028234663852886e38);
  assertThrows(() => maskedFillParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => maskedFillParams([], [], 0), CodegenError);
  assertThrows(() => maskedFillParams([2, 3], [1, 1], Number.NaN), CodegenError);
  // MUST: the finiteness test of the fill value is also as f32 (exactly the smallest finite f32
  // passes, 1e39 fails)
  assertThrows(() => maskedFillParams([2, 3], [1, 1], 1e39), CodegenError, "is not finite as f32");

  const conv = conv1dParams({
    batch: 2,
    channelsIn: 3,
    channelsOut: 4,
    lengthIn: 9,
    lengthOut: 9,
    kernel: 3,
    stride: 1,
    padding: 1,
    dilation: 1,
    groups: 1,
  });
  assertEquals([...conv.slice(0, 11)], [2 * 4 * 9, 2, 3, 4, 9, 9, 3, 1, 1, 1, 1]);
  // The uniform struct is aligned to 16 bytes (12 words are allocated even for 11 words of content)
  assertEquals(conv.byteLength % 16, 0);
  const convParams = (over: Partial<Parameters<typeof conv1dParams>[0]>) =>
    conv1dParams({
      batch: 1,
      channelsIn: 4,
      channelsOut: 4,
      lengthIn: 4,
      lengthOut: 4,
      kernel: 3,
      stride: 1,
      padding: 1,
      dilation: 1,
      groups: 1,
      ...over,
    });
  assertThrows(() => convParams({ stride: 0 }), CodegenError);
  assertThrows(() => convParams({ dilation: 0 }), CodegenError);
  assertThrows(() => convParams({ groups: 0 }), CodegenError);
  // A shape where groups does not divide Cin / Cout also fails at the params layer (the shader's
  // division truncates)
  assertThrows(() => convParams({ groups: 3 }), CodegenError);
  assertThrows(() => convParams({ channelsOut: 6, groups: 4 }), CodegenError);

  // conv1d's implicit GEMM is `{m, n, k}` + 6 geometry words. **n is the output plane of one
  // batch** (the batch is the dispatch's z axis), and `length_out` equals n, so it is not in Dims.
  const conv1Igemm = conv1dIgemmParams({
    batch: 2,
    channelsIn: 6,
    channelsOut: 9,
    lengthIn: 11,
    lengthOut: 4,
    kernel: 3,
    stride: 3,
    padding: 1,
    dilation: 2,
    groups: 1,
  });
  assertEquals(
    [...conv1Igemm.slice(0, 9)],
    [
      9, // m = Cout
      4, // n = Lout (not multiplied by B)
      6 * 3, // k = Cin·K
      6,
      11,
      3,
      3,
      1,
      2,
    ],
  );
  assertEquals(conv1Igemm.byteLength % 16, 0);
  // The contract checks are shared with the direct kernel (the same geometry fails for the same
  // reason) + groups > 1 has its own check
  const conv1IgemmParams = (over: Partial<Parameters<typeof conv1dIgemmParams>[0]>) =>
    conv1dIgemmParams({
      batch: 1,
      channelsIn: 4,
      channelsOut: 4,
      lengthIn: 4,
      lengthOut: 4,
      kernel: 3,
      stride: 1,
      padding: 1,
      dilation: 1,
      groups: 1,
      ...over,
    });
  assertThrows(() => conv1IgemmParams({ stride: 0 }), CodegenError);
  assertThrows(() => conv1IgemmParams({ dilation: 0 }), CodegenError);
  // MUST: groups > 1 is outside the implicit GEMM (the executor sends it to the direct kernel)
  assertThrows(() => conv1IgemmParams({ groups: 2 }), CodegenError);
  assertThrows(() => conv1IgemmParams({ groups: 4 }), CodegenError);

  // conv2d has words for the 2 spatial axes (pins by their order that H and W are not swapped)
  const conv2 = conv2dParams({
    batch: 2,
    channelsIn: 6,
    channelsOut: 9,
    heightIn: 7,
    widthIn: 11,
    heightOut: 5,
    widthOut: 4,
    kernelH: 2,
    kernelW: 3,
    strideH: 1,
    strideW: 3,
    paddingH: 0,
    paddingW: 1,
    dilationH: 2,
    dilationW: 1,
    groups: 3,
  });
  assertEquals(
    [...conv2.slice(0, 17)],
    [2 * 9 * 5 * 4, 2, 6, 9, 7, 11, 5, 4, 2, 3, 1, 3, 0, 1, 2, 1, 3],
  );
  // The uniform struct is aligned to 16 bytes (20 words are allocated even for 17 words of content)
  assertEquals(conv2.byteLength % 16, 0);
  const conv2Params = (over: Partial<Parameters<typeof conv2dParams>[0]>) =>
    conv2dParams({
      batch: 1,
      channelsIn: 4,
      channelsOut: 4,
      heightIn: 4,
      widthIn: 4,
      heightOut: 4,
      widthOut: 4,
      kernelH: 3,
      kernelW: 3,
      strideH: 1,
      strideW: 1,
      paddingH: 1,
      paddingW: 1,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
      ...over,
    });
  // MUST: each axis is checked independently (a shape with 0 on one side only slipping through
  // hangs the GPU)
  assertThrows(() => conv2Params({ strideH: 0 }), CodegenError);
  assertThrows(() => conv2Params({ strideW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ dilationH: 0 }), CodegenError);
  assertThrows(() => conv2Params({ dilationW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ kernelW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ groups: 0 }), CodegenError);
  assertThrows(() => conv2Params({ groups: 3 }), CodegenError);
  assertThrows(() => conv2Params({ channelsOut: 6, groups: 4 }), CodegenError);
  assertThrows(() => conv2Params({ paddingH: -1 }), CodegenError);

  // The implicit GEMM is `{m, n, k}` + 12 geometry words. **n is the output plane of one batch**
  // (the batch is the dispatch's z axis); deriving m·n·k wrongly slips past the shape checks and
  // gives a wrong value.
  const igemm = conv2dIgemmParams({
    batch: 2,
    channelsIn: 6,
    channelsOut: 9,
    heightIn: 7,
    widthIn: 11,
    heightOut: 5,
    widthOut: 4,
    kernelH: 2,
    kernelW: 3,
    strideH: 1,
    strideW: 3,
    paddingH: 0,
    paddingW: 1,
    dilationH: 2,
    dilationW: 1,
    groups: 1,
  });
  assertEquals(
    [...igemm.slice(0, 15)],
    [
      9, // m = Cout
      5 * 4, // n = Hout·Wout (not multiplied by B)
      6 * 2 * 3, // k = Cin·Kh·Kw
      6,
      7,
      11,
      4,
      2,
      3,
      1,
      3,
      0,
      1,
      2,
      1,
    ],
  );
  assertEquals(igemm.byteLength % 16, 0);
  // The contract checks are shared with the direct kernel (the same geometry fails for the same
  // reason) + groups > 1 has its own check
  const igemmParams = (over: Partial<Parameters<typeof conv2dIgemmParams>[0]>) =>
    conv2dIgemmParams({
      batch: 1,
      channelsIn: 4,
      channelsOut: 4,
      heightIn: 4,
      widthIn: 4,
      heightOut: 4,
      widthOut: 4,
      kernelH: 3,
      kernelW: 3,
      strideH: 1,
      strideW: 1,
      paddingH: 1,
      paddingW: 1,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
      ...over,
    });
  assertThrows(() => igemmParams({ strideW: 0 }), CodegenError);
  assertThrows(() => igemmParams({ dilationH: 0 }), CodegenError);
  // MUST: groups > 1 is outside the implicit GEMM (the executor sends it to the direct kernel)
  assertThrows(() => igemmParams({ groups: 2 }), CodegenError);
  assertThrows(() => igemmParams({ groups: 4 }), CodegenError);

  const transpose = convTranspose1dParams({
    batch: 1,
    channelsIn: 3,
    channelsOut: 5,
    lengthIn: 4,
    lengthOut: 8,
    kernel: 4,
    stride: 2,
    padding: 1,
  });
  assertEquals([...transpose.slice(0, 9)], [1 * 5 * 8, 1, 3, 5, 4, 8, 4, 2, 1]);
  assertEquals(transpose.byteLength % 16, 0);
  // MUST: stride 0 makes WGSL's remainder a division by zero (the GPU hang precedent in recon §4)
  assertThrows(
    () =>
      convTranspose1dParams({
        batch: 1,
        channelsIn: 1,
        channelsOut: 1,
        lengthIn: 4,
        lengthOut: 4,
        kernel: 1,
        stride: 0,
        padding: 0,
      }),
    CodegenError,
  );
});

test("the params of bmm / gather fail loudly on values outside the contract", () => {
  assertThrows(() => bmmParams(-1, 2, 3, 1), CodegenError);
  assertThrows(() => bmmParams(1, 2, 1.5, 1), CodegenError);
  assertThrows(() => gatherParams(-1, 2, 3), CodegenError);
  // Zero columns with element count > 0 cannot happen under the shape contract (do not create a
  // division by zero in the kernel)
  assertThrows(() => gatherParams(6, 0, 3), CodegenError);
  assertEquals([...gatherParams(0, 0, 3)], [0, 0, 3, GATHER_OOB_BITS]);
});

test("codegen fails loudly on generator inputs outside the contract", () => {
  assertThrows(() => elementwiseWgsl({ op: "relu", rank: 0, dtype: "f32" }), CodegenError);
  // Ops outside the vocabulary are rejected by the type, so testing the runtime guard needs an
  // explicit narrowing from a wider type
  const foreignElementwise: { op: string; rank: number; dtype: string } = {
    op: "matmul",
    rank: 1,
    dtype: "f32",
  };
  assertThrows(() => elementwiseWgsl(foreignElementwise as ElementwiseSpec), CodegenError);
  // Element types the contract table does not allow (i32 / bool for f32-only ops, f32 for bool-only
  // ops)
  assertThrows(() => elementwiseWgsl({ op: "relu", rank: 1, dtype: "i32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "div", rank: 1, dtype: "i32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "bitwise_not", rank: 1, dtype: "f32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "bitwise_and", rank: 1, dtype: "f32" }), CodegenError);
  // MUST: the **union** of where's accepted sets is {bool, f32}. Looking at the union, the bool for
  // the condition would slip through as a generator input for a value slot, so this pins that it
  // is rejected per slot.
  assertThrows(() => elementwiseWgsl({ op: WHERE_OP, rank: 1, dtype: "bool" }), CodegenError);
  const foreignReduce: { op: string; dtype: string } = { op: "mean", dtype: "f32" };
  assertThrows(() => reduceWgsl(foreignReduce as ReduceSpec), CodegenError);
  // The row reduce's dtype allowance is per op (only sum can count bool)
  assertThrows(() => reduceWgsl({ op: "amax", dtype: "bool" }), CodegenError);
  assertThrows(() => reduceWgsl({ op: "sum", dtype: "i32" }), CodegenError);
  const negR1: ElementwiseSpec = { op: "neg", rank: 1, dtype: "f32" };
  assertThrows(() => elementwiseParams(negR1, [], []), CodegenError);
  assertThrows(() => elementwiseParams(negR1, [4], [[2, 2]]), CodegenError);
  assertThrows(() => matmulParams(-1, 2, 3), CodegenError);
  assertThrows(() => reduceParams(1.5, 2), CodegenError);
  // The axis variants follow the same vocabulary and the same dtype discipline (blocks loosening
  // when copied into another codegen)
  assertThrows(() => axisReduceWgsl(foreignReduce as ReduceSpec), CodegenError);
  assertThrows(() => axisReduceWgsl({ op: "amax", dtype: "bool" }), CodegenError);
  assertThrows(() => axisReduceParams(1.5, 2, 3), CodegenError);
  assertThrows(() => axisReduceParams(4, 2, -1), CodegenError);
});

test("INT2 rejects generating unsupported convolutions and compute precisions", () => {
  const cases = [
    () => conv1dWgsl("i2"),
    () => conv2dWgsl("i2"),
    () => convTranspose1dWgsl("i2"),
    () => conv1dIgemmWgsl("i2", true),
    () => conv2dIgemmWgsl("i2", true),
    () => linearWgsl("i2", true, "f16"),
    () => linearI8a8Wgsl(true, false, undefined, "i2"),
  ];
  for (const generate of cases) assertThrows(generate, CodegenError);
});

test("BSHD RoPE looks up the position table by head count and uses a different key and uniform", () => {
  assertEquals(ROPE_BSHD_KEY, `rope:v1:half:bshd:f32:wg${ROPE_WORKGROUP_SIZE}`);
  assertEquals(ropeWgsl("bshd").includes("let token = row / params.heads;"), true);
  assertEquals([...ropeParams(3 * 5 * 128, 3, 128, "bshd")], [1920, 3, 128, 64]);
  assertThrows(() => ropeParams(768, 0, 128, "bshd"), CodegenError, "heads");
  assertThrows(() => ropeParams(768, 3, 127, "bshd"), CodegenError, "headDim");
  assertThrows(() => ropeParams(769, 3, 128, "bshd"), CodegenError, "whole number of rows");
  assertThrows(() => ropeParams(0, 0x80000000, 128, "bshd"), CodegenError, "u32");
});

/**
 * The **structure** check of conv3d's implicit GEMM (ADR 0118 decision 1 — unbatched, the 3D
 * version of ADR 0024). The numbers (comparison with the CPU reference, the identity checks) are
 * covered by a real GPU (tests/gpu_conv3d_parity_test.ts), so this pins the key, the uniform, the
 * order of the fragments and the generator checks without an adapter.
 */
test("conv3d's implicit GEMM plugs an unbatched 3D gather into the shared skeleton (key, uniform, fragments)", () => {
  // The key is a new line, v1. The edges and the workgroup shape of m tile 64 / 32 come from the
  // geometry (the same discipline as conv1d / conv2d)
  assertEquals(conv3dIgemmKey("f32", true), "conv3d:v1:f32:igemm64x128v4:wg16x8");
  assertEquals(conv3dIgemmKey("f16", false), "conv3d:v1:f32:igemm64x128:wg16x8:wf16");
  assertEquals(
    conv3dIgemmKey("i8", false, GEMM_MTILE_SMALL),
    "conv3d:v1:f32:igemm32x128:wg16x4:wi8",
  );
  // The i8 scale binding has the same number as the implicit GEMM of linear / conv1d / conv2d (the
  // same binding layout)
  assertEquals(CONV3D_SCALE_BINDING, CONV2D_SCALE_BINDING);
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      for (const mTile of [GEMM_TILE, GEMM_MTILE_SMALL]) {
        const wgsl = conv3dIgemmWgsl(weight, v4, mTile);
        const where = `conv3d igemm ${weight} v4=${v4} m${mTile}`;
        // MUST: unbatched — it never reads a batch base (the dispatch's z axis). Copying conv2d's
        // `xbase = wid.z * …` would leave dead arithmetic that stays harmless for a z = 1 dispatch
        // and looks "batch-capable".
        assertEquals(wgsl.includes("wid.z"), false, `${where}: a batch base on the z axis remains`);
        // The uniform is {m,n,k} + 18 geometry words = 21 fields (time_out can be derived from n,
        // so it is not included)
        const struct = wgsl.slice(wgsl.indexOf("struct Dims {"), wgsl.indexOf("}\n@group"));
        assertEquals(struct.split(": u32,").length - 1, 21, `${where}: uniform word count`);
        assertEquals(struct.includes("time_out"), false, `${where}: time_out is a dead field`);
        // bias-first (ADR 0024 decision 3 — adding it on the store side breaks bit identity)
        assertEquals(wgsl.includes(`let bias0 = wid.y * ${mTile}u + lid.y * 8u;`), true, where);
        assertEquals(wgsl.includes("+ biasv"), false, `${where}: the bias is on the store side`);
        // Flat k → (ic, kt, kh, kw), four-level ascending (row-major over the weights [Cout, Cin,
        // Kt, Kh, Kw])
        assertEquals(wgsl.includes("let ic = brow0 / kthw;"), true, where);
        assertEquals(
          wgsl.includes("let kt = i32((kr / khw) * dims.dilation_t) - i32(dims.padding_t);"),
          true,
          where,
        );
        assertEquals(
          wgsl.includes(
            "let ky = i32((ks / dims.kernel_w) * dims.dilation_h) - i32(dims.padding_h);",
          ),
          true,
          where,
        );
        // n → (ot, oy, ox) divides out time by Hout·Wout and rows by Wout
        assertEquals(wgsl.includes("let plane = dims.height_out * dims.width_out;"), true, where);
        assertEquals(
          wgsl.includes("let iy = i32((pixel / dims.width_out) * dims.stride_h) + ky;"),
          true,
          where,
        );
        // Out of range is 0 (not a clamped read — MUST ③ of ADR 0024)
        assertEquals(wgsl.includes("    return 0.0;"), true, where);
        // Only the fast v4 path has the step that divides the quad's first n (no dead let left on
        // the scalar path)
        assertEquals(wgsl.includes("let hw_out ="), v4, where);
      }
    }
  }
  // There is no execution path for i4 / i2 (the A-side unpacker is shared by 1D / 2D / 3D, so the
  // generator entry rejects them)
  assertThrows(() => conv3dIgemmWgsl("i4", false), CodegenError, "i4");
  assertThrows(() => conv3dIgemmWgsl("i2", true), CodegenError, "i2");
});

test("conv3dUsesVec4 looks at all 3 conditions, kFlat / Wout / strideW", () => {
  assertEquals(conv3dUsesVec4(16 * 27, 32, 1), true, "chunk graph conv_in (16→384 k3, latent 32)");
  // MUST: Wout%4 cannot be replaced by Hout·Wout%4 (Hout=2, Wout=2 has N%4==0 but quads straddle
  // rows)
  assertEquals(conv3dUsesVec4(8, 2, 1), false, "Wout=2");
  assertEquals(conv3dUsesVec4(27, 8, 1), false, "kFlat=27 (3×3×3 with Cin=1)");
  assertEquals(conv3dUsesVec4(24, 8, 2), false, "stride_w=2 (4 columns of x are not contiguous)");
  assertEquals(conv3dUsesVec4(4, 4, 1), true);
});

test("conv3d igemm params pack {m,n,k} + 18 geometry words into 24 words and fail loudly outside the contract", () => {
  const base = {
    channelsIn: 3,
    channelsOut: 7,
    timeIn: 6,
    heightIn: 9,
    widthIn: 11,
    timeOut: 2,
    heightOut: 4,
    widthOut: 5,
    kernelT: 3,
    kernelH: 2,
    kernelW: 4,
    strideT: 2,
    strideH: 2,
    strideW: 2,
    paddingT: 0,
    paddingH: 1,
    paddingW: 0,
    dilationT: 1,
    dilationH: 3,
    dilationW: 1,
    groups: 1,
  };
  const params = conv3dIgemmParams(base);
  // The 21 words are packed into 24 words (96 bytes) aligned to 16 bytes (the MUST of uniform
  // struct alignment)
  assertEquals(params.length, 24);
  assertEquals(
    [...params.slice(0, 21)],
    [
      7, // m = Cout
      2 * 4 * 5, // n = Tout·Hout·Wout (the time axis is folded into N)
      3 * 3 * 2 * 4, // k = Cin·Kt·Kh·Kw
      3, // channels_in
      6, // time_in
      9, // height_in
      11, // width_in
      4, // height_out
      5, // width_out
      3, // kernel_t
      2, // kernel_h
      4, // kernel_w
      2, // stride_t
      2, // stride_h
      2, // stride_w
      0, // padding_t
      1, // padding_h
      0, // padding_w
      1, // dilation_t
      3, // dilation_h
      1, // dilation_w
    ],
  );
  assertEquals([...params.slice(21)], [0, 0, 0], "the padding is zero");
  // MUST: each axis is checked independently (a shape with 0 on one side only slipping through
  // hangs the GPU / turns the output all bias)
  for (const key of [
    "strideT",
    "strideH",
    "strideW",
    "dilationT",
    "dilationH",
    "dilationW",
    "kernelT",
    "kernelH",
    "kernelW",
    "timeIn",
    "timeOut",
    "channelsIn",
  ] as const) {
    assertThrows(() => conv3dIgemmParams({ ...base, [key]: 0 }), CodegenError, "positive integer");
  }
  assertThrows(() => conv3dIgemmParams({ ...base, paddingT: -1 }), CodegenError);
  // MUST: groups > 1 is outside the subset implemented on the GPU (ADR 0118 decision 1)
  assertThrows(() => conv3dIgemmParams({ ...base, groups: 3 }), CodegenError, "groups");
  // Values outside u32 fail at the params check (indices and the uniform are u32 — ADR 0118
  // decision 1)
  assertThrows(
    () => conv3dIgemmParams({ ...base, timeOut: 0x10000, heightOut: 0x10000 }),
    CodegenError,
  );
});

test("conv3d igemm params reject an axis whose input length + 2·padding does not fit in i32", () => {
  // From the shape with every axis of length 1, K 1, stride 1, dilation 1, padding 0, only one
  // axis's length and padding are replaced. The output length follows the formula in shapes.ts
  // (K = 1 and stride 1, so input length + 2·padding).
  const unit = {
    channelsIn: 1,
    channelsOut: 1,
    timeIn: 1,
    heightIn: 1,
    widthIn: 1,
    timeOut: 1,
    heightOut: 1,
    widthOut: 1,
    kernelT: 1,
    kernelH: 1,
    kernelW: 1,
    strideT: 1,
    strideH: 1,
    strideW: 1,
    paddingT: 0,
    paddingH: 0,
    paddingW: 0,
    dilationT: 1,
    dilationH: 1,
    dilationW: 1,
    groups: 1,
  };
  const axes = [
    ["timeIn", "paddingT", "timeOut"],
    ["heightIn", "paddingH", "heightOut"],
    ["widthIn", "paddingW", "widthOut"],
  ] as const;
  const padding = 2 ** 30 - 1;
  for (const [lengthKey, paddingKey, outKey] of axes) {
    const withLength = (length: number) => ({
      ...unit,
      [lengthKey]: length,
      [paddingKey]: padding,
      [outKey]: length + 2 * padding,
    });
    // Exactly at the boundary (L + 2p = 2^31 − 1) it passes; one beyond, it fails
    assertEquals(conv3dIgemmParams(withLength(1)).length, 24, `${lengthKey} boundary`);
    assertThrows(
      () => conv3dIgemmParams(withLength(2)),
      CodegenError,
      "i32 maximum",
      `${lengthKey} boundary + 1`,
    );
  }
  // A shape with W length 3 and K 3 where padding and dilation are the u32 maximum. The output
  // length formula holds with Wout = 3, but WGSL's i32(kw·dil) − i32(pad) wraps the true value
  // ±(2^32 − 1) to ∓1, and a read that should be an out-of-range 0 picks up the neighbouring pixel
  // (the form that returns [3,6,5] against the CPU reference [1,2,3]).
  assertThrows(
    () =>
      conv3dIgemmParams({
        ...unit,
        widthIn: 3,
        widthOut: 3,
        kernelW: 3,
        paddingW: 0xffff_ffff,
        dilationW: 0xffff_ffff,
      }),
    CodegenError,
    "axis w",
  );
});

// That params reject shapes whose logical weight element index does not fit in u32 (no GPU needed).
//
// linear's GEMM path, the w4a8 path and embedding fetch the weights by "the flat index of logical
// elements", and the loader (weight-storage.ts) turns it into a word position with `>> 4` for i2
// and `>> 3` for i4. If the product `row × row length` wraps in u32 first, the high bits do not
// come back, and another row's weights are read without an exception. For i2 the storage reaches
// that at about 1 GiB, so the binding-limit check does not stop it. Huge weights are not allocated;
// the accepted set of params is pinned here.

describe("linearParams: the logical element count of the weights [N, K]", () => {
  it("rejects shapes with N·K ≥ 2^32 (with M 65, N 131076, K 32768, row 131072 points at word 0)", () => {
    assertThrows(() => linearParams(65, 131076, 32768), CodegenError, "n * k");
  });

  it("rejects exactly N·K = 2^32 and accepts 2^32 − 16", () => {
    assertThrows(() => linearParams(65, 131072, 32768), CodegenError, "n * k");
    assertEquals([...linearParams(65, 268435455, 16)], [65, 268435455, 16, 0]);
  });
});

describe("linearI8a8Params: the logical element count of the w4a8 (i4 weights × i8 activations) weights [N, K]", () => {
  it("rejects the shape with N·K = 2^32 (the row head wcol·dims.k wraps) and accepts 2^32 − 32", () => {
    assertThrows(() => linearI8a8Params(65, 131072, 32768, 32), CodegenError, "n * k");
    assertEquals([...linearI8a8Params(65, 134217727, 32, 32)], [65, 134217727, 32, 0]);
  });

  it("the i8 variant builds the row head in words, so the logical-element-count check does not reject the same N·K = 2^32", () => {
    assertEquals([...linearI8a8Params(65, 131072, 32768)], [65, 131072, 32768, 0]);
  });
});

describe("embeddingParams: the logical element count of the weights [V, H]", () => {
  it("rejects the shape with V·H = 2^32 (the last row reads the first row's integer codes)", () => {
    assertThrows(() => embeddingParams(32768, 32768, 131072), CodegenError, "vocab * hidden");
  });

  it("accepts V·H = 2^32 − 1", () => {
    assertEquals(embeddingParams(1, 1, 0xffff_ffff)[2], 0xffff_ffff);
  });
});

/** The storage bindings a WGSL declares (the WebGPU default per stage is 8). */
const storageBindingCount = (wgsl: string): number => wgsl.match(/var<storage,/g)?.length ?? 0;

test("pair-form RoPE reads the rotation from its own pairs binding and keeps the shared rounding barrier", () => {
  const wgsl = ropeWgsl("bshd", "pair");
  assertEquals(ROPE_PAIR_BSHD_KEY, `rope:v1:pair:bshd:f32:wg${ROPE_WORKGROUP_SIZE}`);
  assertEquals(wgsl.includes("@group(0) @binding(2) var<storage, read> pairs: array<f32>;"), true);
  assertEquals(wgsl.includes("@binding(5) var<storage, read_write> out"), true);
  assertEquals(storageBindingCount(wgsl), 5);
  // rot[2p] = -pairs[2p+1], rot[2p+1] = pairs[2p] (the parity of the flat index is the slot).
  assertEquals(wgsl.includes("if ((i & 1u) == 0u) {"), true);
  assertEquals(wgsl.includes("rotated_bits = bitcast<u32>(pairs[i + 1u]) ^ 0x80000000u;"), true);
  assertEquals(wgsl.includes("rotated_bits = bitcast<u32>(pairs[i - 1u]);"), true);
  assertEquals(wgsl.includes("half_dim]"), false, "the pair rotation never reads half_dim");
  assertEquals(wgsl.includes("let token = row / params.heads;"), true);
  // The same barrier as the half split: both products through workgroup u32, then the add.
  assertEquals(wgsl.includes("products[lid] = vec2<u32>(direct_bits, cross_bits);"), true);
  assertEquals(wgsl.match(/workgroupBarrier\(\);/g)?.length, 2);
  assertEquals(wgsl.includes("out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);"), true);
  // Only the measured layout exists.
  assertThrows(() => ropeWgsl("bhsd", "pair"), CodegenError, "only for bshd");
});

test("per-token adaLN shares the staged body with the broadcast variant and stays at 7 storage bindings", () => {
  const token = adalnNormWgsl("token");
  assertEquals(
    ADALN_NORM_TOKEN_KEY,
    `adaln_norm:v1:lastdim:token:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`,
  );
  assertEquals(adalnNormWgsl("broadcast"), ADALN_NORM_WGSL);
  assertEquals(token.includes(LAYER_NORM_ROW_STATS_WGSL), true, "shared row statistics");
  assertEquals(token.includes(`(${LAYER_NORM_AFFINE_WGSL}) * modulation`), true, "shared affine");
  // The where of the plain sequence: true → condition (row 0 of the table half), false → generation.
  assertEquals(token.includes("let scale_base = select(dim, 0u, mask[row] != 0u);"), true);
  assertEquals(token.includes("let shift_base = scale_base + 2u * dim;"), true);
  assertEquals(
    token.includes("staged[lid] = bitcast<u32>(modulation_table[scale_base + o] + one[0u]);"),
    true,
  );
  assertEquals(
    token.includes("out[base + o] = bitcast<f32>(staged[lid]) + modulation_table[shift_base + o];"),
    true,
  );
  assertEquals(
    token.match(/workgroupBarrier\(\);/g)?.length,
    ADALN_NORM_WGSL.match(/workgroupBarrier\(\);/g)?.length,
  );
  assertEquals(storageBindingCount(token), 7);
  assertEquals(storageBindingCount(ADALN_NORM_WGSL), 7);
});

test("the adaLN modulation pack copies four vectors as raw u32 words over a grid-stride loop", () => {
  assertEquals(ADALN_MODULATION_PACK_KEY, "adaln_modulation_pack:v1:u32:wg256");
  assertEquals(storageBindingCount(ADALN_MODULATION_PACK_WGSL), 5);
  // u32 bindings only: a copy with no f32 arithmetic keeps every bit pattern.
  assertEquals(ADALN_MODULATION_PACK_WGSL.includes("f32"), false);
  for (const line of [
    "modulation_table[o] = scale_condition[o];",
    "modulation_table[dim + o] = scale_generation[o];",
    "modulation_table[2u * dim + o] = shift_condition[o];",
    "modulation_table[3u * dim + o] = shift_generation[o];",
    "o = o + stride;",
  ]) {
    assertEquals(ADALN_MODULATION_PACK_WGSL.includes(line), true, line);
  }
  assertEquals([...adalnModulationPackParams(3072)], [3072, 0, 0, 0]);
  assertEquals(adalnModulationPackParams(3072).byteLength, 16);
  assertThrows(() => adalnModulationPackParams(0), CodegenError, "dim");
  // The body indexes the table up to 4 · dim - 1.
  assertThrows(() => adalnModulationPackParams(0x4000_0000), CodegenError, "u32");
});

test("the gated residual stages the product through workgroup u32 before adding the residual", () => {
  assertEquals(
    GATED_RESIDUAL_KEY,
    `gated_residual:v1:token:f32:wg${GATED_RESIDUAL_WORKGROUP_SIZE}`,
  );
  assertEquals(GATED_RESIDUAL_WORKGROUP_SIZE, 256);
  assertEquals(storageBindingCount(GATED_RESIDUAL_WGSL), 6);
  assertEquals(
    GATED_RESIDUAL_WGSL.includes(
      "let gate = select(gate_generation[o], gate_condition[o], mask[row] != 0u);",
    ),
    true,
  );
  // mul(y, g) then add(residual, p), with the product rounded through workgroup memory.
  assertEquals(GATED_RESIDUAL_WGSL.includes("products[lid] = bitcast<u32>(y[i] * gate);"), true);
  assertEquals(
    GATED_RESIDUAL_WGSL.includes("out[i] = residual[i] + bitcast<f32>(products[lid]);"),
    true,
  );
  assertEquals(GATED_RESIDUAL_WGSL.match(/workgroupBarrier\(\);/g)?.length, 2);
  assertEquals([...gatedResidualParams(5 * 257, 257)], [1285, 257, 0, 0]);
  assertEquals(gatedResidualParams(1285, 257).byteLength, 16);
  assertThrows(() => gatedResidualParams(1285, 0), CodegenError, "dim");
  assertThrows(() => gatedResidualParams(1286, 257), CodegenError, "whole number of rows");
  assertThrows(() => gatedResidualParams(0x1_0000_0000, 1), CodegenError, "u32");
});

describe("group_norm kernels (ADR 0138 decision 7)", () => {
  it("keys name the version, the chunk and the workgroup size", () => {
    assertEquals(GROUP_NORM_WORKGROUP_SIZE, 256);
    assertEquals(GROUP_NORM_CHUNK, 1024);
    assertEquals(GROUP_NORM_PARTIAL_KEY, "group_norm_partial:v1:f32:chunk1024:wg256");
    assertEquals(GROUP_NORM_TOTAL_KEY, "group_norm_total:v1:f32:wg256");
    assertEquals(GROUP_NORM_APPLY_KEY, "group_norm_apply:v1:f32:wg256");
    assertEquals(storageBindingCount(GROUP_NORM_PARTIAL_WGSL), 3);
    assertEquals(storageBindingCount(GROUP_NORM_TOTAL_WGSL), 2);
    assertEquals(storageBindingCount(GROUP_NORM_APPLY_WGSL), 5);
  });

  it("sums in two passes: the values, then the squared deviations from the stored mean", () => {
    // Pass 0 adds x, pass 1 adds (x - mean)^2 with the mean of the total stage — never x * x
    // (the one-pass E[x^2] - E[x]^2 form cancels for groups with a large mean).
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("acc = acc + v;"), true);
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("let d = v - mean;"), true);
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("acc = acc + d * d;"), true);
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("v * v"), false);
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("mean = stats[2u * row];"), true);
    assertEquals(
      GROUP_NORM_TOTAL_WGSL.includes("stats[2u * row] = scratch[0u] * params.inv_length;"),
      true,
    );
    assertEquals(
      GROUP_NORM_TOTAL_WGSL.includes(
        "stats[2u * row + 1u] = 1.0 / sqrt(scratch[0u] * params.inv_length + params.eps);",
      ),
      true,
    );
    // No division by the element count in the shader (the host rounds 1/length once).
    assertEquals(/f32\(params\.length\)/.test(GROUP_NORM_TOTAL_WGSL), false);
  });

  it("reduces with one fixed tree shape and grid-strides every loop", () => {
    for (const wgsl of [GROUP_NORM_PARTIAL_WGSL, GROUP_NORM_TOTAL_WGSL]) {
      assertEquals(wgsl.includes("var stride = 128u;"), true);
      assertEquals(wgsl.includes("scratch[lid] = scratch[lid] + scratch[lid + stride];"), true);
      assertEquals(wgsl.includes("+ nwg.x;"), true);
    }
    assertEquals(GROUP_NORM_PARTIAL_WGSL.includes("for (var k = 0u; k < 4u; k = k + 1u) {"), true);
    assertEquals(GROUP_NORM_TOTAL_WGSL.includes("j = j + 256u;"), true);
    assertEquals(GROUP_NORM_APPLY_WGSL.includes("let step = nwg.x * 256u;"), true);
    assertEquals(
      GROUP_NORM_APPLY_WGSL.includes(
        "out[e] = (x[e] - stats[2u * row]) * stats[2u * row + 1u] * weight[c] + bias[c];",
      ),
      true,
    );
    assertEquals(
      GROUP_NORM_APPLY_WGSL.includes(
        "let c = (row % params.groups) * params.per_group + o / params.inner;",
      ),
      true,
    );
  });

  it("lays out x[N,C,...] as N·G rows of (C/G)·inner elements", () => {
    const layout = groupNormLayout([2, 192, 3, 5], 32);
    assertEquals(layout, {
      rows: 64,
      length: 90,
      inner: 15,
      groups: 32,
      perGroup: 6,
      chunks: 1,
      count: 5760,
    });
    // The MOSS decoder's largest group at the area cap: 6 channels × 1696·1648 pixels.
    const cap = groupNormLayout([1, 192, 1696, 1648], 32);
    assertEquals(cap.length, 16_770_048);
    assertEquals(cap.chunks, Math.ceil(16_770_048 / 1024));
    assertEquals(cap.rows * cap.chunks > 65_535, true);
    assertEquals(groupNormTempBytes(cap), { partials: 32 * 16_377 * 4, stats: 32 * 2 * 4 });
    assertEquals(groupNormLayout([4, 8], 2).inner, 1);
    assertThrows(() => groupNormLayout([1, 6, 4], 4), CodegenError, "divide");
    assertThrows(() => groupNormLayout([1, 6, 0], 3), CodegenError, "positive");
    assertThrows(() => groupNormLayout([6], 3), CodegenError, "[N,C");
    assertThrows(() => groupNormLayout([1, 4, 65_536, 65_536], 1), CodegenError, "u32");
  });

  it("writes the uniforms with eps and 1/length as f32 bit patterns", () => {
    const layout = groupNormLayout([1, 6, 700], 3);
    assertEquals([...groupNormPartialParams(layout, 1)], [3, 1400, 2, 1]);
    assertEquals(groupNormPartialParams(layout, 0).byteLength, 16);
    const total = groupNormTotalParams(layout, 1, 1e-6);
    assertEquals(total.byteLength, 32);
    assertEquals([...total.slice(0, 3)], [3, 2, 1]);
    const floats = new Float32Array(total.buffer);
    assertEquals(floats[3], Math.fround(1 / 1400));
    assertEquals(floats[4], Math.fround(1e-6));
    assertEquals([...groupNormApplyParams(layout)], [4200, 1400, 700, 3, 2, 0, 0, 0]);
    for (const eps of [0, -1e-6, 1e-50, 1e-40, 1e39, Number.NaN]) {
      assertThrows(() => groupNormTotalParams(layout, 0, eps), CodegenError, "eps");
    }
  });
});
