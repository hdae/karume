import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../src/codegen/elementwise.ts";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  axisReduceKey,
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
  ARGMAX_WGSL,
  ARGMAX_WORKGROUP_SIZE,
  argmaxParams,
} from "../src/kernels/argmax.ts";
import { bmmKey, bmmParams, bmmWgsl } from "../src/kernels/bmm.ts";
import { statePvWgsl, stateQkWgsl, stateStatsWgsl } from "../src/kernels/state-attention.ts";
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
import { ATTENTION_QK_MASK_BINDING, GEMM_MTILE_SMALL, gemmUsesVec4 } from "../src/kernels/gemm.ts";
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
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_OOB_BITS,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../src/kernels/deform-conv2d.ts";
import {
  GRU_SCAN_MAX_HIDDEN,
  GRU_SCAN_WORKGROUP_SIZE,
  type GruScanDirection,
  gruScanKey,
  gruScanParams,
  gruScanWgsl,
} from "../src/kernels/gru-scan.ts";
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
  ADALN_NORM_KEY,
  ADALN_NORM_WGSL,
  ADALN_NORM_WORKGROUP_SIZE,
  adalnNormParams,
} from "../src/kernels/adaln-norm.ts";
import {
  RMS_NORM_KEY,
  RMS_NORM_WGSL,
  RMS_NORM_WORKGROUP_SIZE,
  rmsNormParams,
} from "../src/kernels/rms-norm.ts";
import { ROPE_KEY, ROPE_WGSL, ROPE_WORKGROUP_SIZE, ropeParams } from "../src/kernels/rope.ts";
import { linearKey, linearParams, linearWgsl } from "../src/kernels/linear.ts";
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
  quantizeRowsParams,
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
import {
  defaultI8a8Geometry,
  type I8a8Geometry,
  i8a8GeometryKeyPart,
  i8a8TileM,
  i8a8TileN,
} from "../src/kernels/i8a8-geometry.ts";
import { attentionScoreUsesF16 } from "../src/kernels/score-storage.ts";
import { BINARY_OPS, OP_CONTRACTS, REDUCE_OPS, UNARY_OPS, WHERE_OP } from "../src/ops.ts";

const fixture = (name: string): Promise<string> =>
  Deno.readTextFile(new URL(`./fixtures/wgsl/${name}`, import.meta.url));

// 融合 5 カーネルは重み格納の変種を持つ（ADR 0018）。既存の不変条件は **w=f32 変種**に
// 掛かっているので、従来の名前をその変種に束ねる（f16 変種は専用テストが別に見る）。
// GEMM 3 op は更に形状由来の v4 フラグで 2 変種を持つ（src/kernels/gemm.ts）— 従来の名前は
// **スカラ変種**に束ね、v4 側は各テストで明示的に並べる。
const LINEAR_WGSL = linearWgsl("f32", false);
const MATMUL_WGSL = matmulWgsl(false);
const BMM_WGSL = bmmWgsl(false);
/**
 * GEMM 骨格の全生成入力（7 op × 重み格納 × v4）を機械的に回すための直積。
 * 融合 attention の ①③（ADR 0023）と conv1d / conv2d の implicit GEMM（ADR 0024）も同じ
 * 骨格を共有するので、内積ループ・タイル辺・キーの検査が自動でこちらにも掛かる。
 */
const GEMM_VARIANTS: readonly (readonly [string, string, string])[] = [false, true]
  .flatMap((
    v4,
  ) => [
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
    ]),
    // i4 は **linear / embedding 限定**（ADR 0069 決定 5 + embedding 追補）— conv 系の直積には
    // 入れない（group scale の束縛経路が linear の充填と embedding のカーネルにしか無く、conv へ
    // 渡すと不成立 WGSL になる）。group は既定の 32（追記 1）で代表させる — group 違いはキーの
    // g 部と shift の焼き込みが対で変わるだけ。
    [
      `linear i4 g32${v4 ? " v4" : ""}`,
      linearKey("i4", v4, "f32", undefined, 32),
      linearWgsl("i4", v4, "f32", undefined, 32),
    ] as const,
  ]);
const EMBEDDING_WGSL = embeddingWgsl("f32");
const CONV1D_WGSL = conv1dWgsl("f32");
const CONV1D_KEY = conv1dKey("f32");
const CONV2D_WGSL = conv2dWgsl("f32");
const CONV2D_KEY = conv2dKey("f32");
const CONV_TRANSPOSE1D_WGSL = convTranspose1dWgsl("f32");

/**
 * 契約表が解禁している (op, dtype) の全組（codegen が生成しうる elementwise 空間）。
 * dtype は**値スロット**の側 — where の条件スロット（bool 固定）は生成入力に載らないので、
 * スロット 0 ではなく「値の側」を持つスロットの受理集合から引く。
 */
const elementwiseDtypes = (): readonly (readonly [string, IrDtype])[] =>
  [...UNARY_OPS, ...BINARY_OPS, WHERE_OP].flatMap((op) => {
    const contract = OP_CONTRACTS.get(op);
    if (contract === undefined) return [];
    const slots = contract.slotDtypes;
    const accept = slots.kind === "uniform" ? slots.accept : slots.slots[slots.slots.length - 1];
    return accept.map((dtype): [string, IrDtype] => [op, dtype]);
  });

Deno.test("生成した WGSL がスナップショットとバイト単位で一致する（codegen 決定性の固定）", async () => {
  const cases: readonly (readonly [string, string])[] = [
    ["elementwise_relu_r1.wgsl", elementwiseWgsl({ op: "relu", rank: 1, dtype: "f32" })],
    ["elementwise_sigmoid_r1.wgsl", elementwiseWgsl({ op: "sigmoid", rank: 1, dtype: "f32" })],
    ["elementwise_gelu_r2.wgsl", elementwiseWgsl({ op: "gelu", rank: 2, dtype: "f32" })],
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
    [
      "elementwise_clamp_min_r1.wgsl",
      elementwiseWgsl({ op: "clamp_min", rank: 1, dtype: "f32" }),
    ],
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
    // amax / amin は畳み込み関数だけが違う（NaN 伝播の形を両方とも凍結する）
    ["reduce_amin.wgsl", reduceWgsl({ op: "amin", dtype: "f32" })],
    // 軸 reduce（最終次元以外）の別変種。行 reduce の縮約順序を 1 スレッドの bitrev
    // carry-stack で再現する形をバイト単位で凍結する（順序が動けばビット同一が崩れる）。
    // dtype 変種（bool → i32）と NaN 伝播形（amax）も並べる。
    ["reduce_axis_sum.wgsl", axisReduceWgsl({ op: "sum", dtype: "f32" })],
    ["reduce_axis_sum_bool.wgsl", axisReduceWgsl({ op: "sum", dtype: "bool" })],
    ["reduce_axis_amax.wgsl", axisReduceWgsl({ op: "amax", dtype: "f32" })],
    ["cumsum.wgsl", CUMSUM_WGSL],
    // argmax（ADR 0068 決定 2）。**reduce 3 本と対で置く**のが条件で、行 reduce と同型の
    // 骨格を持つぶん「新族を足したら既存 reduce のバイト列が動いた」が最大の事故
    // （既存 3 本 + 軸変種 3 本がその検出器）。
    ["argmax.wgsl", ARGMAX_WGSL],
    // topk（ADR 0068 決定 3）。**k=1 と一般形を対で置く**のが条件 — k=1 ではブロックの末尾
    // （最弱）と先頭が同じ語になり、挿入ループが 1 度も回らない縮退形なので、一般形だけを
    // 固定すると `k-1` の焼き込みが 1 ずれても気づけない（k=1 は argmax と同じ答えを返す
    // 突合門の相手でもある）。
    ["topk_k1.wgsl", topkWgsl(1)],
    ["topk_k4.wgsl", topkWgsl(4)],
    // GEMM 3 op は形状由来の v4 フラグで 2 変種（スカラ / vec4）を持つ。**両側を並べる**のが
    // 条件で、片方だけ固定すると端数形状のフォールバックが黙って壊れても気づけない。
    ["matmul.wgsl", MATMUL_WGSL],
    ["matmul_v4.wgsl", matmulWgsl(true)],
    ["bmm.wgsl", BMM_WGSL],
    ["bmm_v4.wgsl", bmmWgsl(true)],
    // bmm の**行窓変種**（分解 attention の行ブロック実行 — src/runtime/fusion.ts の
    // `rowBlockAttention`）。A 側 / C 側 × v4 の 4 本を、**素の bmm 2 本と対で置く**のが
    // 条件で、行窓を足したことで素の bmm のバイト列が動くのが最大の事故（行ブロック 1 枚の
    // 機は素の bmm をそのまま撃つので、そこが動けば既存の WAV / PNG 門が丸ごと動く）。
    ["bmm_rowwin_a.wgsl", bmmWgsl(false, undefined, "a")],
    ["bmm_rowwin_a_v4.wgsl", bmmWgsl(true, undefined, "a")],
    ["bmm_rowwin_c.wgsl", bmmWgsl(false, undefined, "c")],
    ["bmm_rowwin_c_v4.wgsl", bmmWgsl(true, undefined, "c")],
    // 融合 attention の 3 カーネル（ADR 0023）。①③ は GEMM 骨格の変種なので v4 と対で置く。
    ["attention_qk.wgsl", attentionQkWgsl(false)],
    ["attention_qk_v4.wgsl", attentionQkWgsl(true)],
    ["attention_pv.wgsl", attentionPvWgsl(false)],
    ["attention_pv_v4.wgsl", attentionPvWgsl(true)],
    ["attention_stats.wgsl", ATTENTION_STATS_WGSL],
    // ①QK の**加算 mask 変種**（ADR 0023 改訂）。この 2 本を足すことより、**上の
    // attention_qk*.wgsl / attention_pv*.wgsl / attention_stats*.wgsl が 1 バイトも
    // 動かない**ことがこの列挙の主目的（②③ は mask を一切見ない）。
    ["attention_qk_mask.wgsl", attentionQkWgsl(false, "f32", "f32", true)],
    ["attention_qk_mask_v4.wgsl", attentionQkWgsl(true, "f32", "f32", true)],
    // **GQA 変種**（整除 broadcast — ADR 0067 決定 1 / 2）。①③ を **v4 と対で 4 本**置くのが
    // 条件で、この 4 本を足すことより **上の attention_qk*.wgsl / attention_pv*.wgsl が
    // 1 バイトも動かない**（= `r = 1` はバイト同一という決定 2 の MUST）が列挙の主目的。
    ["attention_qk_gqa.wgsl", attentionQkWgsl(false, "f32", "f32", false, true)],
    ["attention_qk_gqa_v4.wgsl", attentionQkWgsl(true, "f32", "f32", false, true)],
    ["attention_pv_gqa.wgsl", attentionPvWgsl(false, "f32", "f32", true)],
    ["attention_pv_gqa_v4.wgsl", attentionPvWgsl(true, "f32", "f32", true)],
    // **行窓変種**（クエリ行のブロック実行 — src/runtime/recipe-builder.ts の `#buildAttention`）。
    // ①QK は A 側（q）・③PV は C 側（O）が全 M ストライド + 行オフセットになる。**mask 付きを
    // 対で置く**のが条件で、行窓で唯一 base 算術の外へ出るのが mask の**行**添字だから
    // （S はブロック相対・mask は全 M ぶんの実体）。この 6 本を足すことより **上の
    // attention_qk*.wgsl / attention_pv*.wgsl が 1 バイトも動かない**（= n=1 は行窓を立てない
    // ので分割前と完全に同一）が列挙の主目的。
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
    // 書き族は cat 専用で、契約が f32 専業なので生成されうるのは f32 だけ（ADR 0014）
    ["strided_write_f32.wgsl", stridedWriteWgsl({ dtype: "f32" })],
    ["pad.wgsl", PAD_WGSL],
    ["upsample2x.wgsl", UPSAMPLE_2X_WGSL],
    // 双線形 resample（第 1 層 op）。第 3 層の融合ルール `upsample2x`（nearest のビット複製）
    // とは別カーネル・別キーで、**両方を並べる**のが条件 — 片方だけを直したときに、もう
    // 一方のバイト列が動いていないことがこの列挙で機械確認される。
    ["upsample_bilinear2d.wgsl", UPSAMPLE_BILINEAR2D_WGSL],
    // DCNv2（第 1' 層 op — ADR 0055）。**conv2d の直接カーネル（下の conv2d_*.wgsl）と対で
    // 並べる**のが条件で、退化ビット一致（offset 0・mask 1 → conv2d）は両者の縮約順が
    // 厳密に一致していることに依存する — 片方だけバイト列が動くのが最大の事故。
    ["deform_conv2d.wgsl", DEFORM_CONV2D_WGSL],
    // GRU 隠れ側スキャン（第 2 層 op — ADR 0056）。**2 方向を両方並べる**のが条件で、
    // 走査順の 1 行以外がずれていないことがこの対でしか見えない。丸め障壁（workgroup
    // memory 往復）が生成物に残っていることも、下の不変条件テストと合わせてここで凍結する。
    ["gru_scan_forward.wgsl", gruScanWgsl("forward")],
    ["gru_scan_reverse.wgsl", gruScanWgsl("reverse")],
    ["silu_x_sigmoid.wgsl", siluWgsl("x-sigmoid")],
    ["silu_sigmoid_x.wgsl", siluWgsl("sigmoid-x")],
    ["flip.wgsl", FLIP_WGSL],
    ["rope.wgsl", ROPE_WGSL],
    ["linear.wgsl", LINEAR_WGSL],
    ["linear_v4.wgsl", linearWgsl("f32", true)],
    // skinny-M 幾何（M ≤ 64 → M16N16 — src/kernels/gemm-geometry.ts の掃引確定値）の代表
    // 1 本。**既定側（上 2 本）と対で置く**のが条件で、バケット追加で既定のバイト列が
    // 動くのが最大の事故。3 op は骨格共有なので代表は linear だけでよい。
    ["linear_m16_v4.wgsl", linearWgsl("f32", true, "f32", 4)],
    ["layer_norm.wgsl", LAYER_NORM_WGSL],
    // adaLN 融合（ADR 0040 の 4 本目）。**素の layer_norm と対で置く**のが条件で、
    // 行統計と affine の文字列を共有しているぶん、片方だけバイト列が動くのが最大の事故。
    ["adaln_norm.wgsl", ADALN_NORM_WGSL],
    ["rms_norm.wgsl", RMS_NORM_WGSL],
    ["softmax.wgsl", SOFTMAX_WGSL],
    // safe_softmax 変種（ADR 0044）。**素の softmax と対で置く**のが条件で、両者は同じ
    // 生成関数から出る（②③ の縮約順序が 1 語でもずれれば分解経路とのビット同一が壊れる）。
    ["safe_softmax.wgsl", SAFE_SOFTMAX_WGSL],
    ["embedding.wgsl", EMBEDDING_WGSL],
    ["masked_fill.wgsl", MASKED_FILL_WGSL],
    ["conv1d.wgsl", CONV1D_WGSL],
    ["conv2d.wgsl", CONV2D_WGSL],
    ["conv_transpose1d.wgsl", CONV_TRANSPOSE1D_WGSL],
    // 重み f16 / i8 格納の変種（ADR 0018 / 0019）。上の 5 本と対で置き、**f32 側のバイト列が
    // 動いていない**ことを同じ列挙で機械確認する（変種追加で既存が動くのが最大の事故）。
    ["linear_wf16.wgsl", linearWgsl("f16", false)],
    ["linear_wf16_v4.wgsl", linearWgsl("f16", true)],
    ["embedding_wf16.wgsl", embeddingWgsl("f16")],
    ["conv1d_wf16.wgsl", conv1dWgsl("f16")],
    ["conv2d_wf16.wgsl", conv2dWgsl("f16")],
    ["conv_transpose1d_wf16.wgsl", convTranspose1dWgsl("f16")],
    ["linear_wi8.wgsl", linearWgsl("i8", false)],
    ["linear_wi8_v4.wgsl", linearWgsl("i8", true)],
    // i4 は linear / embedding 限定（ADR 0069）— group 32（既定）の shift を焼いた変種。
    // embedding 版は **linear と対で置く**（scale の引き方が別実装〈充填で quad ごと /
    // カーネル本体で要素ごと〉なので、片方だけ動いた生成物が普通に組み上がる）。
    ["linear_wi4.wgsl", linearWgsl("i4", false, "f32", undefined, 32)],
    ["linear_wi4_v4.wgsl", linearWgsl("i4", true, "f32", undefined, 32)],
    ["embedding_wi4.wgsl", embeddingWgsl("i4", 32)],
    ["embedding_wi8.wgsl", embeddingWgsl("i8")],
    ["conv1d_wi8.wgsl", conv1dWgsl("i8")],
    ["conv2d_wi8.wgsl", conv2dWgsl("i8")],
    ["conv_transpose1d_wi8.wgsl", convTranspose1dWgsl("i8")],
    // conv2d の implicit GEMM（ADR 0024）は格納 3 × v4 の 6 変種。直接カーネル
    // （上の conv2d*.wgsl）と**両方**並べるのが条件で、片方だけだと踏み分けの一方が
    // 黙って動いても気づけない。
    ["conv2d_igemm.wgsl", conv2dIgemmWgsl("f32", false)],
    ["conv2d_igemm_v4.wgsl", conv2dIgemmWgsl("f32", true)],
    ["conv2d_igemm_wf16.wgsl", conv2dIgemmWgsl("f16", false)],
    ["conv2d_igemm_wf16_v4.wgsl", conv2dIgemmWgsl("f16", true)],
    ["conv2d_igemm_wi8.wgsl", conv2dIgemmWgsl("i8", false)],
    ["conv2d_igemm_wi8_v4.wgsl", conv2dIgemmWgsl("i8", true)],
    // 32 行 m タイル変種（ADR 0024 隣接）。**64 行版と対で並べる**のが条件で、
    // 変種追加で 64 行側のバイト列が動くのが最大の事故（上の 6 本が検出器）。
    ["conv2d_igemm_m32.wgsl", conv2dIgemmWgsl("f32", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_v4.wgsl", conv2dIgemmWgsl("f32", true, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wf16.wgsl", conv2dIgemmWgsl("f16", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wf16_v4.wgsl", conv2dIgemmWgsl("f16", true, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wi8.wgsl", conv2dIgemmWgsl("i8", false, GEMM_MTILE_SMALL)],
    ["conv2d_igemm_m32_wi8_v4.wgsl", conv2dIgemmWgsl("i8", true, GEMM_MTILE_SMALL)],
    // conv1d の implicit GEMM（ADR 0024 の 1D 版）も同じ粒度で 12 変種。**上の conv1d*.wgsl
    // （直接カーネル）と conv2d_igemm*.wgsl の両方と対で並べる**のが条件で、断片を
    // 1D へ一般化したときに 2D 側のバイト列が動くのが最大の事故。
    ["conv1d_igemm.wgsl", conv1dIgemmWgsl("f32", false)],
    ["conv1d_igemm_v4.wgsl", conv1dIgemmWgsl("f32", true)],
    ["conv1d_igemm_wf16.wgsl", conv1dIgemmWgsl("f16", false)],
    ["conv1d_igemm_wf16_v4.wgsl", conv1dIgemmWgsl("f16", true)],
    ["conv1d_igemm_wi8.wgsl", conv1dIgemmWgsl("i8", false)],
    ["conv1d_igemm_wi8_v4.wgsl", conv1dIgemmWgsl("i8", true)],
    ["conv1d_igemm_m32.wgsl", conv1dIgemmWgsl("f32", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_v4.wgsl", conv1dIgemmWgsl("f32", true, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wf16.wgsl", conv1dIgemmWgsl("f16", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wf16_v4.wgsl", conv1dIgemmWgsl("f16", true, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wi8.wgsl", conv1dIgemmWgsl("i8", false, GEMM_MTILE_SMALL)],
    ["conv1d_igemm_m32_wi8_v4.wgsl", conv1dIgemmWgsl("i8", true, GEMM_MTILE_SMALL)],
    // conv1d の i4 変種（ADR 0069 決定 5 の conv1d 追補 — group 32 の shift を焼いた形）。
    // **linear_wi4* と対で置く**のが条件で、i4 の group scale は linear が B 側・conv1d が
    // A 側と別実装（`fillBLinear` / `fillAConv`）— 片方だけ動いた生成物が普通に組み上がる。
    // MUST: conv2d 側に対の i4 は無い（実行経路そのものが無く、生成の入口が落とす）。
    ["conv1d_igemm_wi4.wgsl", conv1dIgemmWgsl("i4", false, undefined, 32)],
    ["conv1d_igemm_wi4_v4.wgsl", conv1dIgemmWgsl("i4", true, undefined, 32)],
    ["conv1d_igemm_m32_wi4.wgsl", conv1dIgemmWgsl("i4", false, GEMM_MTILE_SMALL, 32)],
    ["conv1d_igemm_m32_wi4_v4.wgsl", conv1dIgemmWgsl("i4", true, GEMM_MTILE_SMALL, 32)],
    // w8a8（活性 i8 化 + 整数内積）。**dp4a 版とエミュ版の両方**を置くのが条件で、
    // 「数値は同じで速度だけ違う」という主張は生成物が 2 つ別々に存在することが前提になる。
    ["quantize_rows.wgsl", QUANTIZE_ROWS_WGSL],
    ["linear_i8a8.wgsl", linearI8a8Wgsl(false, true)],
    ["linear_i8a8_v4.wgsl", linearI8a8Wgsl(true, true)],
    ["linear_i8a8_emu.wgsl", linearI8a8Wgsl(false, false)],
    ["linear_i8a8_emu_v4.wgsl", linearI8a8Wgsl(true, false)],
    // w4a8（i4 常駐の重み × per-token i8 活性・perf-ledger Q-8）。**上の linear_i8a8* 4 本と
    // 対で置く**のが条件で、この 4 本を足したことで i8 側のバイト列が動くのが最大の事故
    // （i8 経路は 1 バイトも変わらないことが w4a8 の実装契約そのもの）。K ループが
    // 「group 外側 × タイル内側」の 2 段になるので、共有断片（内積ループ / A 側充填）が
    // 1 段浅い字下げのまま入るのも含めてここで凍結する。
    ["linear_w4a8_g32.wgsl", linearI8a8Wgsl(false, true, undefined, "i4", 32)],
    ["linear_w4a8_g32_v4.wgsl", linearI8a8Wgsl(true, true, undefined, "i4", 32)],
    ["linear_w4a8_g32_emu.wgsl", linearI8a8Wgsl(false, false, undefined, "i4", 32)],
    ["linear_w4a8_g32_emu_v4.wgsl", linearI8a8Wgsl(true, false, undefined, "i4", 32)],
    // 融合 attention ①QK / ③PV の i8a8 変種（設計 §7 の波 1 / 波 2）。**上の f32 / :c16
    // 変種と対で置く**のが条件で、変種追加で既定経路のバイト列が動くのが最大の事故。
    ["attention_qk_i8a8.wgsl", attentionQkI8a8Wgsl(false, true)],
    ["attention_qk_i8a8_v4.wgsl", attentionQkI8a8Wgsl(true, true)],
    ["attention_qk_i8a8_emu.wgsl", attentionQkI8a8Wgsl(false, false)],
    ["attention_qk_i8a8_emu_v4.wgsl", attentionQkI8a8Wgsl(true, false)],
    ["attention_pv_i8a8.wgsl", attentionPvI8a8Wgsl(false, true)],
    ["attention_pv_i8a8_v4.wgsl", attentionPvI8a8Wgsl(true, true)],
    ["attention_pv_i8a8_emu.wgsl", attentionPvI8a8Wgsl(false, false)],
    ["attention_pv_i8a8_emu_v4.wgsl", attentionPvI8a8Wgsl(true, false)],
    // i8a8 の**行窓変種**（実測形の v4 側を代表に 1 本ずつ — スカラ側との差は base の単位だけで、
    // その踏み分けは上の素の 4 本と下の構造テストが見る）。①QK は量子化済み q とその行 scale の
    // 2 本、③PV は O の 1 本だけが全 M ストライドになる。
    ["attention_qk_i8a8_rwa_v4.wgsl", attentionQkI8a8Wgsl(true, true, "f32", undefined, true)],
    ["attention_pv_i8a8_rwc_v4.wgsl", attentionPvI8a8Wgsl(true, true, "f32", undefined, true)],
    // f16 **計算**変種（ADR 0028）。上の f32 変種と対で置くのが条件で、変種追加で f32 側の
    // バイト列が動くのが最大の事故（既定経路の数値契約が全部そこに掛かっている）。
    // 語彙は格納 f16 の `wf16` と別の `c16` — `linear_wf16_c16*` は「重み f16 格納 × f16 計算」。
    ["attention_qk_c16.wgsl", attentionQkWgsl(false, "f16")],
    ["attention_qk_c16_v4.wgsl", attentionQkWgsl(true, "f16")],
    ["attention_pv_c16.wgsl", attentionPvWgsl(false, "f16")],
    ["attention_pv_c16_v4.wgsl", attentionPvWgsl(true, "f16")],
    ["attention_stats_c16.wgsl", attentionStatsWgsl("f16")],
    ["linear_c16.wgsl", linearWgsl("f32", false, "f16")],
    ["linear_c16_v4.wgsl", linearWgsl("f32", true, "f16")],
    ["linear_wf16_c16.wgsl", linearWgsl("f16", false, "f16")],
    ["linear_wf16_c16_v4.wgsl", linearWgsl("f16", true, "f16")],
    // S の **f16 格納**変種（案 γ 波 1 — `SessionOptions.attentionScoreStorage`）。語彙は
    // 格納重み `wf16` / 計算 `c16` と別の `s16` で、**v4 側しか存在しない**（スカラ経路の
    // 部分書きは同じ u32 語への RMW になるので生成の入口が落とす — 下の fail loudly テスト）。
    // MUST: 上の f32 / c16 / i8a8 変種と対で置く（変種追加で既定経路のバイト列が動くのが
    // 最大の事故で、それを見ているのがこの列挙そのもの）。
    ["attention_qk_s16_v4.wgsl", attentionQkWgsl(true, "f32", "f16")],
    ["attention_pv_s16_v4.wgsl", attentionPvWgsl(true, "f32", "f16")],
    ["attention_stats_s16.wgsl", attentionStatsWgsl("f32", "f16")],
    ["attention_qk_i8a8_s16_v4.wgsl", attentionQkI8a8Wgsl(true, true, "f16")],
    ["attention_qk_i8a8_emu_s16_v4.wgsl", attentionQkI8a8Wgsl(true, false, "f16")],
    ["attention_pv_i8a8_s16_v4.wgsl", attentionPvI8a8Wgsl(true, true, "f16")],
    ["attention_pv_i8a8_emu_s16_v4.wgsl", attentionPvI8a8Wgsl(true, false, "f16")],
    // ② 行統計の **regcache 変種**（S を 1 回だけ読んでレジスタに残す）。dim 依存の生成なので
    // `epc` がキー軸に増える — 実測形の self（dim 4096 → 16 要素／スレッド）を凍結する。
    // MUST: 上の 2 回読み版と対で置く（値はビット同一である以上、生成物の側でしか差が見えない）。
    ["attention_stats_rc16.wgsl", attentionStatsWgsl("f32", "f32", 16)],
    // **states 形 attention**（ADR 0067 決定 4〜7）の 3 カーネル + `state_append`。既存の融合
    // attention とは 1 バイトも共有しない**別族**（K / V の出どころが 2 つ・identity が −inf・
    // 走査範囲が実行時値）なので、この 12 本を足すことより **上の attention_* / bmm_* /
    // gemm 骨格の全スナップショットが 1 バイトも動かない**ことがこの列挙の主目的。
    // MUST: 変種は 4 族とも full / sliding を対で置く（sliding だけが下限述語と ring 写像を
    // 持つので、片側だけの固定では「full に下限が生えた」も「sliding から下限が消えた」も
    // スナップショットを素通りする）。①③ は GQA 変種も対で置く。
    ["attention_state_qk.wgsl", stateQkWgsl(false, false)],
    ["attention_state_qk_gqa.wgsl", stateQkWgsl(false, true)],
    ["attention_state_qk_sliding.wgsl", stateQkWgsl(true, false)],
    ["attention_state_qk_sliding_gqa.wgsl", stateQkWgsl(true, true)],
    ["attention_state_stats.wgsl", stateStatsWgsl(false)],
    ["attention_state_stats_sliding.wgsl", stateStatsWgsl(true)],
    ["attention_state_pv.wgsl", statePvWgsl(false, false)],
    ["attention_state_pv_gqa.wgsl", statePvWgsl(false, true)],
    ["attention_state_pv_sliding.wgsl", statePvWgsl(true, false)],
    ["attention_state_pv_sliding_gqa.wgsl", statePvWgsl(true, true)],
    ["state_append.wgsl", stateAppendWgsl(false)],
    ["state_append_sliding.wgsl", stateAppendWgsl(true)],
  ];
  for (const [name, generated] of cases) {
    assertEquals(generated, await fixture(name), name);
  }
  // MUST: スナップショットの列挙とディレクトリの中身を突き合わせる。列挙から漏れた
  // .wgsl は「置いてあるのに 1 度も比較されない」死んだ固定になり、逆にディレクトリに
  // 無い名前は上の readTextFile で落ちる（= 片側だけ足した状態を両方向で塞ぐ）。
  assertEquals(
    [...Deno.readDirSync(new URL("./fixtures/wgsl/", import.meta.url))]
      .map((entry) => entry.name).sort(),
    cases.map(([name]) => name).sort(),
    "fixtures/wgsl の中身とスナップショット列挙が食い違う",
  );
});

Deno.test("同じ生成入力からは常に同一の WGSL が出る（全 op × dtype × rank）", () => {
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
  // GEMM は 6 op × 重み格納 × v4 が 1 本の骨格を共有する（生成が状態を持たないことの固定）
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
});

Deno.test("パイプラインキーは生成入力ごとに一意（別カーネルが同じキーに割り当たらない）", () => {
  const keys: string[] = [
    // GEMM 3 op × 重み格納 × v4 の直積（漏れを作らないために列挙を 1 本にまとめてある）
    ...GEMM_VARIANTS.map(([, key]) => key),
    GATHER_KEY,
    LAYER_NORM_KEY,
    RMS_NORM_KEY,
    SOFTMAX_KEY,
    SAFE_SOFTMAX_KEY,
    ATTENTION_STATS_KEY,
    MASKED_FILL_KEY,
    CUMSUM_KEY,
    // argmax は reduce 族と**別のパイプライン**（identity が −inf・(値, index) 対を運ぶ）。
    // 衝突すると片方の WGSL が他方の dispatch へ割り当たり、bind 面（出力 i32）が違うので
    // 例外なしに別のビット列が読まれる。
    ARGMAX_KEY,
    // topk は **k ごとに別パイプライン**（k を WGSL に焼くので配列長とラウンド数が変わる）。
    // k を含めないと最初に組んだ k のパイプラインが別の k の dispatch に配られ、例外なしに
    // 別の本数だけが書かれた出力（残りは前 run の残骸）が読まれる。argmax とも衝突しない。
    topkKey(1),
    topkKey(4),
    PAD_KEY,
    FLIP_KEY,
    // upsample 系 2 本は**対で載せる**（第 3 層の融合ルール `upsample2x` = nearest の
    // ビット複製 / 第 1 層 op の `upsample_bilinear2d`）。名前が似ているぶん同じキーへ
    // 割り当てる事故が起きやすく、起きると nearest の IR が bilinear のパイプラインで走る。
    UPSAMPLE_2X_KEY,
    UPSAMPLE_BILINEAR2D_KEY,
    DEFORM_CONV2D_KEY,
    // 融合 5 カーネルは重み格納の変種ぶん（ADR 0018）。WGSL 本体が違うので、
    // 変種どうしが同じキーに割り当たると片方が黙って他方のパイプラインで走る
    // （linear は上の GEMM 直積が v4 込みで持っている）。
    ...WEIGHT_STORAGES.flatMap((weight) => [
      embeddingKey(weight),
      conv1dKey(weight),
      conv2dKey(weight),
      convTranspose1dKey(weight),
    ]),
    // conv2d の 32 行 m タイル変種（ADR 0024 隣接）。**64 行版と別キー**であることが条件で、
    // 同じキーに割り当たると片方の WGSL が他方のパイプラインで走る（workgroup 形が違うので
    // 沈黙 no-op ではなく即座に誤値になる）。
    ...WEIGHT_STORAGES.flatMap((weight) =>
      [false, true].flatMap((v4) => [
        conv2dIgemmKey(weight, v4, GEMM_MTILE_SMALL),
        conv1dIgemmKey(weight, v4, GEMM_MTILE_SMALL),
      ])
    ),
    // i4 は **group 長ごとに別キー**（shift を WGSL に焼く — ADR 0069）。g 部がキーに
    // 乗っていないと group 32 のパイプラインが group 64 の資産で走り、scale 添字がずれた
    // 沈黙誤値になる（linear g32 の基本形は上の GEMM 直積が持っている）。
    ...[false, true].flatMap((v4) => [
      linearKey("i4", v4, "f32", undefined, 64),
      linearKey("i4", v4, "f32", undefined, 128),
    ]),
    ...[16, 32, 64].map((groupSize) => embeddingKey("i4", groupSize)),
    // w8a8: v4 × 整数内積変種の 4 本 + 活性量子化。**dp4a とエミュを別キーにする**のが条件で、
    // 同じキーに割り当たると診断でどちらが走ったか分からなくなる（設計 §4.4-5）。
    ...[false, true].flatMap((v4) => [linearI8a8Key(v4, false), linearI8a8Key(v4, true)]),
    // w4a8 は **格納判別子 + group 長ごとに別キー**（g は WGSL に shift として焼かれる）。
    // g 部が載っていないと group 32 のパイプラインが group 64 の資産で走り、scale 添字が
    // ずれた沈黙誤値になる。i8 の同じ幾何・同じ v4 のキーとも衝突しないことが条件。
    ...[false, true].flatMap((v4) =>
      [16, 32, 64].flatMap((groupSize) => [
        linearI8a8Key(v4, false, undefined, "i4", groupSize),
        linearI8a8Key(v4, true, undefined, "i4", groupSize),
      ])
    ),
    // 融合 attention ①QK / ③PV の i8a8 変種も同じ規律
    // （f32 / :c16 / i8a8 の 3 系統 × 段の 2 本が全て別キー）
    ...[false, true].flatMap((v4) => [
      attentionQkI8a8Key(v4, false),
      attentionQkI8a8Key(v4, true),
      attentionPvI8a8Key(v4, false),
      attentionPvI8a8Key(v4, true),
    ]),
    QUANTIZE_ROWS_KEY,
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
      // 軸変種は行 reduce と**別のパイプライン**（衝突すると片方の WGSL がもう片方の
      // dispatch に割り当たり、縮約軸を取り違えたまま例外なしで通る）
      keys.push(axisReduceKey({ op, dtype }));
    }
  }
  for (const dtype of SEMANTIC_DTYPES) {
    keys.push(stridedKey({ dtype }));
    // 読み族と書き族は WGSL が違う（走査の向きが逆）— 同じ要素型でキーが衝突しないこと
    keys.push(stridedWriteKey({ dtype }));
  }
  assertEquals(new Set(keys).size, keys.length);
});

// permute / expand は 1 カーネル族（ADR 0011）。同じ WGSL なので op ごとに別キーにはならず、
// 要素型だけが別パイプラインの軸になる。
Deno.test("strided copy は要素型ごとに別カーネルで、rank は固定でキーに載る", () => {
  assertEquals(stridedWgsl({ dtype: "f32" }).includes("array<f32>"), true);
  assertEquals(stridedWgsl({ dtype: "i32" }).includes("array<i32>"), true);
  // bool は u32 の 0/1 で格納する（ADR 0009）
  assertEquals(stridedWgsl({ dtype: "bool" }).includes("array<u32>"), true);
  assertNotEquals(stridedKey({ dtype: "i32" }), stridedKey({ dtype: "bool" }));
  for (const dtype of SEMANTIC_DTYPES) {
    assertEquals(stridedKey({ dtype }).includes(`r${STRIDED_RANK}`), true);
    // 出力は常に連続なので、読み替えるのは src 側だけ
    assertEquals(stridedWgsl({ dtype }).includes("out[i] = src[src_index];"), true);
  }
});

Deno.test("permute の stride は軸の並べ替えで、expand の複製軸は stride 0", () => {
  // 連続 [2,3,4] の stride は [12,4,1]。dims=[0,2,1] は出力軸 1 が入力軸 2 を取る。
  assertEquals(permuteSrcStrides([2, 3, 4], [0, 2, 1]), [12, 1, 4]);
  assertEquals(permuteSrcStrides([1, 16, 5, 64], [0, 2, 1, 3]), [5120, 64, 320, 1]);
  // MUST: 巡回長 3 以上を 1 本持つ。実測の並べ替えは全て対合（逆置換 = 自分自身）なので、
  // stride 表を逆置換で組む誤りが対合の例では検出できない。[1,2,0] の逆は [2,0,1]。
  assertEquals(permuteSrcStrides([2, 3, 4], [1, 2, 0]), [4, 1, 12]);
  assertEquals(permuteSrcStrides([2, 3, 4], [2, 0, 1]), [1, 12, 4]);
  // 実測形: gather 添字 [1,T,T] → [16,T,T]（先頭を複製）と bool マスク [1,T,1] → [1,T,1024]
  assertEquals(expandSrcStrides([1, 6, 6], [16, 6, 6]), [0, 6, 1]);
  assertEquals(expandSrcStrides([1, 6, 1], [1, 6, 1024]), [0, 1, 0]);
  // rank が増える形は先行軸も stride 0
  assertEquals(expandSrcStrides([4], [3, 4]), [0, 1]);
});

Deno.test("strided params は rank 不足を左詰めで 1 / 0 に埋める", () => {
  // 出力 [3,4]（rank2）→ dims は [1,1,3,4]、strides は [0,0,...]
  const params = stridedParams([3, 4], [1, 3], 0);
  assertEquals(params.length, 2 + 2 * STRIDED_RANK);
  assertEquals([...params.slice(0, 5)], [12, 1, 1, 3, 4]);
  assertEquals([...params.slice(5, 9)], [0, 0, 1, 3]);
  assertEquals(params[9], 0, "permute / expand の offset は常に 0");
  assertEquals([...stridedParams([2, 3, 4, 5], [60, 20, 5, 1], 7).slice(1)], [
    2,
    3,
    4,
    5,
    60,
    20,
    5,
    1,
    7,
  ]);
});

// ADR 0014: slice は既存の読み族の流用で、可変点は **params の offset 1 語**だけ。
Deno.test("slice の読み出しは入力の連続 stride と start × その軸の stride", () => {
  assertEquals(sliceSrcStrides([2, 3, 4]), [12, 4, 1]);
  // enc_p の m_p / logs_p 分割と同型（チャネル軸の後半を取る形）
  assertEquals(sliceSrcOffset([1, 8, 5], 1, 4), 20);
  assertEquals(sliceSrcOffset([1, 8, 5], 1, 0), 0, "先頭からの切り出しは offset 0");
  // 最終次元の切り出しは stride 1 なので offset = start そのもの
  assertEquals(sliceSrcOffset([3, 7], 1, 2), 2);
  // 端（軸長ちょうど）は空スライスの開始位置として妥当
  assertEquals(sliceSrcOffset([3, 7], 0, 3), 21);
  assertThrows(() => sliceSrcOffset([3, 7], 2, 0), CodegenError, undefined, "軸が rank の外");
  assertThrows(() => sliceSrcOffset([3, 7], 0, 4), CodegenError, undefined, "start が軸長超過");
  assertThrows(() => sliceSrcOffset([3, 7], 0, -1), CodegenError);
});

// ADR 0014: cat は書き族。offset を取り違えると出力の一部が二重書きされ、残りが未書き込みの
// まま（プール再利用なら前の値）残る。
Deno.test("cat の書き込みは出力の連続 stride と、先行入力の長さ × その軸の stride", () => {
  assertEquals(catOutStrides([2, 6, 4]), [24, 4, 1]);
  // coupling reverse の 96/96 分割と同型: 2 本目は軸 1 の 96 要素目から書く
  assertEquals(catOutOffset([1, 8, 5], 1, 0), 0);
  assertEquals(catOutOffset([1, 8, 5], 1, 4), 20);
  assertEquals(catOutOffset([1, 8, 5], 1, 8), 40, "全部書き終えた位置（被覆検査の右端）");
  assertThrows(() => catOutOffset([1, 8, 5], 3, 0), CodegenError);
  assertThrows(() => catOutOffset([1, 8, 5], 1, 9), CodegenError, undefined, "軸長超過");
});

Deno.test("strided 書き params は入力 dims と出力 strides を同じレイアウトで運ぶ", () => {
  // 入力 [3,4]（rank2）→ dims は左詰めで [1,1,3,4]、strides も左詰めで 0 埋め
  const params = stridedWriteParams([3, 4], [7, 1], 21);
  assertEquals(params.length, 2 + 2 * STRIDED_RANK);
  assertEquals([...params.slice(0, 5)], [12, 1, 1, 3, 4], "n は**入力**の要素数");
  assertEquals([...params.slice(5, 9)], [0, 0, 7, 1]);
  assertEquals(params[9], 21, "書き出し先頭 offset");
  assertThrows(() => stridedWriteParams([], [], 0), CodegenError);
  assertThrows(() => stridedWriteParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => stridedWriteParams([3, 4], [4, 1], -1), CodegenError);
});

Deno.test("pad / flip の params は行と軸の 3 分割をそのまま運ぶ", () => {
  // [2,3] を最終次元に [1,2] で pad → 出力 [2,6]、n=12
  assertEquals([...padParams(2, 3, 1, 2)], [12, 6, 3, 1]);
  // pad 0 幅（恒等コピー）も表現できる
  assertEquals([...padParams(4, 5, 0, 0)], [20, 5, 5, 0]);
  assertThrows(() => padParams(2, 3, -1, 0), CodegenError);
  assertThrows(() => padParams(2, 3, 0, 1.5), CodegenError);
  // [2,3,4] の軸 1 を反転 → outer=2 / len=3 / inner=4
  assertEquals([...flipParams(2, 3, 4)], [24, 3, 4, 0]);
  // 最終軸の反転は inner = 1
  assertEquals([...flipParams(6, 4, 1)], [24, 4, 1, 0]);
  assertThrows(() => flipParams(2, -1, 4), CodegenError);
});

Deno.test("strided codegen は契約外の生成入力を fail loudly にする", () => {
  const foreign: { dtype: string } = { dtype: "f16" };
  assertThrows(() => stridedWgsl(foreign as StridedSpec), CodegenError);
  assertThrows(() => stridedKey(foreign as StridedSpec), CodegenError);
  // rank 0 / STRIDED_RANK 超え
  assertThrows(() => stridedParams([], [], 0), CodegenError);
  assertThrows(() => stridedParams([2, 2, 2, 2, 2], [16, 8, 4, 2, 1], 0), CodegenError);
  assertThrows(() => stridedParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => stridedParams([3, 4], [1, 3], -1), CodegenError);
  assertThrows(() => permuteSrcStrides([2, 3], [0, 1, 2]), CodegenError);
  assertThrows(() => permuteSrcStrides([2, 3], [0, 5]), CodegenError);
  assertThrows(() => expandSrcStrides([2, 3], [3]), CodegenError);
});

// 要素型は bindings の array<T> に埋まる。キーに載っていないと、i32 の dispatch に f32 で
// コンパイル済みのパイプラインが同じキーで割り当たり、ビット列の読み替えが例外なしに通る。
Deno.test("要素型は WGSL とキーの両方に現れる（f32 と i32 が別カーネルになる）", () => {
  const f32 = elementwiseWgsl({ op: "mul", rank: 2, dtype: "f32" });
  const i32 = elementwiseWgsl({ op: "mul", rank: 2, dtype: "i32" });
  assertNotEquals(f32, i32);
  assertEquals(f32.includes("array<f32>"), true);
  assertEquals(i32.includes("array<i32>"), true);
  assertNotEquals(
    elementwiseKey({ op: "mul", rank: 2, dtype: "f32" }),
    elementwiseKey({ op: "mul", rank: 2, dtype: "i32" }),
  );
  // bool は u32 の 0/1 で格納する（ストレージバッファに 1bit 型は無い — ADR 0009）
  assertEquals(
    elementwiseWgsl({ op: "bitwise_not", rank: 1, dtype: "bool" }).includes("array<u32>"),
    true,
  );
  // cast は入力と出力で要素型が違う（片方だけをキーに載せると衝突する）
  const widen = elementwiseWgsl({ op: "cast", rank: 1, dtype: "bool", to: "f32" });
  assertEquals(widen.includes("read> in0: array<u32>"), true);
  assertEquals(widen.includes("read_write> out: array<f32>"), true);
  assertNotEquals(
    elementwiseKey({ op: "cast", rank: 1, dtype: "bool", to: "f32" }),
    elementwiseKey({ op: "cast", rank: 1, dtype: "f32", to: "bool" }),
  );
});

// 契約（src/ops.ts）: f32 → 整数は 0 方向切り捨て、x → bool は x != 0。
Deno.test("cast の値式は契約どおり（i32() の切り捨て / != 0 の真偽化）", () => {
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

Deno.test("パイプラインキーは WGSL に埋まる workgroup サイズを含む", () => {
  for (const rank of [1, 2, 3, 4]) {
    const key = elementwiseKey({ op: "relu", rank, dtype: "f32" });
    assertEquals(key.includes(`wg${ELEMENTWISE_WORKGROUP_SIZE}`), true, key);
  }
});

Deno.test("elementwise の生成 WGSL は補助関数を使う op にだけ注入する", () => {
  const gelu = elementwiseWgsl({ op: "gelu", rank: 1, dtype: "f32" });
  const relu = elementwiseWgsl({ op: "relu", rank: 1, dtype: "f32" });
  assertEquals(gelu.includes("fn erf_approx"), true);
  assertEquals(gelu.includes("fn sigmoid_stable"), false);
  assertEquals(relu.includes("fn erf_approx"), false);
  // tanh 形は組込の tanh だけで書ける — erf の近似式を巻き込まないことと、√(2/π) の
  // リテラルをここで固定する（2 つの gelu を取り違えても shape も dtype も合ってしまう）。
  const geluTanh = elementwiseWgsl({ op: "gelu_tanh", rank: 1, dtype: "f32" });
  assertEquals(geluTanh.includes("fn erf_approx"), false);
  assertEquals(
    geluTanh.includes("tanh(0.7978845608028654 * (v0 + 0.044715 * v0 * v0 * v0))"),
    true,
  );
  // sin は WGSL 組込の素通し。多項式近似（erf 形の前例）へ差し替えられていないことと、
  // NaN 外殻（{@link nanGuard} 由来のビット列判定）が紛れ込んでいないことを固定する。
  const sin = elementwiseWgsl({ op: "sin", rank: 1, dtype: "f32" });
  assertEquals(sin.includes("out[i] = sin(v0);"), true);
  assertEquals(sin.includes("fn erf_approx"), false);
  assertEquals(sin.includes("fn is_nan_bits"), false);
  // MUST: 素朴な 1/(1+exp(-x)) は x ≲ -88 でオーバーフローが indeterminate になる
  assertEquals(
    elementwiseWgsl({ op: "sigmoid", rank: 1, dtype: "f32" }).includes("exp(-abs(x))"),
    true,
  );
});

// M1-P3 波3 で足した数理 op 群。カーネル固有の不変条件を生成物の側から固定する
// （値の突合は tests/gpu_ops_test.ts、契約は tests/ops_contract_test.ts）。
Deno.test("where / 比較 / clamp / leaky_relu の値式が契約どおりの形を生成する", () => {
  // MUST: torch の where(cond, a, b) = cond ? a : b。WGSL の select(f, t, cond) は引数順が
  // 逆なので、写し間違えると分岐が入れ替わったまま shape も dtype も合ってしまう。
  const where = elementwiseWgsl({ op: WHERE_OP, rank: 2, dtype: "f32" });
  assertEquals(where.includes("out[i] = select(v2, v1, v0 != 0u);"), true);
  // 条件スロットだけ u32（bool）で、値スロットと出力は f32
  assertEquals(where.includes("read> in0: array<u32>;"), true);
  assertEquals(where.includes("read> in1: array<f32>;"), true);
  assertEquals(where.includes("read> in2: array<f32>;"), true);
  assertEquals(where.includes("read_write> out: array<f32>;"), true);

  // 比較は f32 を読んで bool（u32 の 0/1）を書く
  const ge = elementwiseWgsl({ op: "ge", rank: 1, dtype: "f32" });
  assertEquals(ge.includes("read> in0: array<f32>;"), true);
  assertEquals(ge.includes("read_write> out: array<u32>;"), true);
  assertEquals(ge.includes("out[i] = select(0u, 1u, v0 >= v1);"), true);
  // ge と gt_scalar の境界（等値点）は生成式の演算子でしか区別できない
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

  // MUST: leaky_relu は select 形（`max(x, s·x)` は WGSL の max が NaN 伝播を保証しない —
  // ADR 0015）。slope は params 由来の s0 で、WGSL には焼かない。
  const leaky = elementwiseWgsl({ op: "leaky_relu", rank: 1, dtype: "f32" });
  assertEquals(leaky.includes("out[i] = select(s0 * v0, v0, v0 >= 0.0);"), true);
  assertEquals(leaky.includes("max("), false);
  assertEquals(leaky.includes("let s0 = bitcast<f32>(params[3u]);"), true);

  // MUST: clamp も組込 clamp を使わない（lo > hi が実装依存）。NaN はビット列判定の外殻で
  // 伝播させる — 内側の 2 段 select はコンパイラが max イディオムへ畳むので当てにならない。
  const clamp = elementwiseWgsl({ op: "clamp", rank: 1, dtype: "f32" });
  assertEquals(
    clamp.includes(
      "out[i] = select(select(select(v0, s0, v0 < s0), s1, v0 > s1), v0, is_nan_bits(v0));",
    ),
    true,
  );
  assertEquals(clamp.includes("clamp("), false);

  // bool の論理積はビット演算ではなく真偽の && で書く（0/1 以外の格納でも真偽になる）
  assertEquals(
    elementwiseWgsl({ op: "bitwise_and", rank: 1, dtype: "bool" }).includes(
      "out[i] = select(0u, 1u, (v0 != 0u) && (v1 != 0u));",
    ),
    true,
  );
});

/**
 * NaN 伝播の外殻（ビット列判定）。**浮動小数の比較で判定していない**ことを生成物の側で
 * 固定する — 比較形はシェーダコンパイラが `max` イディオムへ畳み、ドライバの `max` が
 * NaN を飲む（実 GPU での伝播そのものは tests/gpu_ops_test.ts が見る）。
 */
Deno.test("clamp / clamp_min / relu / amax / amin は NaN を比較ではなくビット列で判定する", () => {
  const bits = "return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;";
  for (const op of ["clamp", "clamp_min", "relu"] as const) {
    const wgsl = elementwiseWgsl({ op, rank: 1, dtype: "f32" });
    assertEquals(wgsl.includes(`fn is_nan_bits(x: f32) -> bool {\n  ${bits}\n}`), true, op);
    // 外殻は「NaN ならその値をそのまま返す」形（内側の式は非 NaN 経路だけを担う）
    assertEquals(wgsl.includes(", v0, is_nan_bits(v0));"), true, op);
  }
  // 補助関数の注入規律（erf / log1p と同じ）— 外殻が要らない op には漏れない
  for (const op of ["leaky_relu", "abs", "neg"] as const) {
    const wgsl = elementwiseWgsl({ op, rank: 1, dtype: "f32" });
    assertEquals(wgsl.includes("is_nan_bits"), false, op);
  }
  // MUST: 縮約は 2 段（1 スレッドの走査 + workgroup の木）。片方だけ守っても NaN は
  // 反対側の段で identity（±F32_MAX）に飲まれるので、両段が同じ関数を通ること。
  for (const [op, fn] of [["amax", "nan_max"], ["amin", "nan_min"]] as const) {
    const wgsl = reduceWgsl({ op, dtype: "f32" });
    assertEquals(wgsl.includes(bits), true, op);
    assertEquals(wgsl.includes(`acc = ${fn}(acc, x[base + i]);`), true, op);
    assertEquals(
      wgsl.includes(`scratch[lid] = ${fn}(scratch[lid], scratch[lid + stride]);`),
      true,
      op,
    );
    // 軸変種も 2 段（スレッド内の走査 + carry-stack）とも同じ関数を通る
    const axis = axisReduceWgsl({ op, dtype: "f32" });
    assertEquals(axis.includes(bits), true, `${op}:axis`);
    assertEquals(axis.includes(`v = ${fn}(v, x[base + c * inner]);`), true, `${op}:axis`);
    assertEquals(axis.includes(`v = ${fn}(acc[k], v);`), true, `${op}:axis`);
  }
  assertEquals(reduceWgsl({ op: "sum", dtype: "f32" }).includes("is_nan_bits"), false);
  assertEquals(axisReduceWgsl({ op: "sum", dtype: "f32" }).includes("is_nan_bits"), false);
});

// MUST: 素朴な log(1 + x) は |x| ≪ 1 で有効桁が消える。補正式が入っていることを生成物で
// 固定する（数値の実測は tests/gpu_ops_test.ts の CPU 参照突合）。
Deno.test("log1p は級数と log を切り替える補助関数を注入し、他の op には漏れない", () => {
  const log1p = elementwiseWgsl({ op: "log1p", rank: 1, dtype: "f32" });
  assertEquals(log1p.includes("fn log1p_series(x: f32) -> f32 {"), true);
  // 係数 1/k の Horner 形（先頭 3 項と末項）と切り替え点
  assertEquals(log1p.includes("let series = x * (1.0 - x * (0.5 - x * (0.3333333333333333"), true);
  assertEquals(log1p.includes("x * (0.1)"), true);
  assertEquals(log1p.includes("return select(log(1.0 + x), series, abs(x) < 0.25);"), true);
  assertEquals(log1p.includes("out[i] = log1p_series(v0);"), true);
  // 補助関数は使う op にだけ入る（erf / sigmoid と同じ規律）
  assertEquals(log1p.includes("fn erf_approx"), false);
  assertEquals(elementwiseWgsl({ op: "log", rank: 1, dtype: "f32" }).includes("log1p"), false);
  assertEquals(elementwiseWgsl({ op: "gelu", rank: 1, dtype: "f32" }).includes("log1p"), false);
});

// MUST: 累積の向きは値でしか検出できない（shape も要素数も変わらない）。生成物の側では
// 「前の要素を足してから書く」順序を固定する。
Deno.test("cumsum は 1 invocation = 1 行の前縁和で、行方向を grid-stride で回す", () => {
  assertEquals(CUMSUM_WGSL.includes("acc = acc + x[base + j];"), true);
  assertEquals(CUMSUM_WGSL.includes("out[base + j] = acc;"), true);
  // 行ループは grid-stride（1 次元の dispatch 上限を超える行数がありうる）
  assertEquals(CUMSUM_WGSL.includes("var row = gid.x;"), true);
  assertEquals(CUMSUM_WGSL.includes("row = row + stride;"), true);
  assertEquals([...cumsumParams(7, 3)], [7, 3, 0, 0]);
  assertEquals(cumsumParams(7, 3).byteLength, 16);
  assertThrows(() => cumsumParams(-1, 3), CodegenError);
  assertThrows(() => cumsumParams(1.5, 3), CodegenError);
});

/**
 * argmax（ADR 0068 決定 2）。**値では検出できない**契約が 3 つあるので、生成物の側から固定する:
 *
 * 1. **タイブレークの向き** — `ib < ia`（最小 index）を `>` に書き換えても shape も dtype も
 *    合ってしまう。木の両子で同じ述語を使っていることも同時に見る（片方だけ向きが違うと
 *    「木の形に依存して勝者が変わる」= 実行結果が dispatch 数で揺れる）。
 * 2. **identity が −inf** — 有限 sentinel（reduce 族の `-F32_MAX`）に戻っていないこと。
 *    ビット列は params 3 語目で運ぶ（const-expression の inf を避ける — softmax と同じ規律）。
 * 3. **NaN 判定がビット列** — 素の比較に戻すと NaN が黙って負け、`amax` は NaN・`argmax` は
 *    別要素という族内の食い違いになる。
 */
Deno.test("argmax は最小 index を保存する比較式・−inf identity・ビット列 NaN 判定を持つ", () => {
  assertEquals(ARGMAX_KEY, `argmax:v1:f32>i32:lastdim:minindex:wg${ARGMAX_WORKGROUP_SIZE}`);
  assertEquals(ARGMAX_WORKGROUP_SIZE, 256);
  // ① タイブレーク: 同値なら小さい index が勝つ（辞書式順序の 1 本の述語）
  assertEquals(ARGMAX_WGSL.includes("return vb > va || (vb == va && ib < ia);"), true);
  // 述語は 1 箇所定義で、走査・木の両方が同じ関数を呼ぶ（向きの取り違えを構造で潰す）
  assertEquals(ARGMAX_WGSL.includes("fn argmax_beats(vb: f32, ib: u32, va: f32, ia: u32)"), true);
  assertEquals(ARGMAX_WGSL.includes("if (argmax_beats(v, i, best, best_at)) {"), true);
  assertEquals(
    ARGMAX_WGSL.includes(
      "if (argmax_beats(other, other_at, scratch_value[lid], scratch_index[lid])) {",
    ),
    true,
  );
  // ② identity は −inf（params 経由）+ index の番兵は dim（全 −inf 行が index 0 になる根拠）
  assertEquals(ARGMAX_WGSL.includes("let neg_inf = bitcast<f32>(params.neg_inf);"), true);
  assertEquals(ARGMAX_WGSL.includes("var best = neg_inf;"), true);
  assertEquals(ARGMAX_WGSL.includes("var best_at = dim;"), true);
  // MUST: reduce 族の有限 sentinel が紛れ込んでいない
  assertEquals(ARGMAX_WGSL.includes("3.402823466e38"), false);
  // ③ NaN はビット列で判定して「最大」として扱う（両方 NaN なら最小 index）
  assertEquals(ARGMAX_WGSL.includes("fn is_nan_bits(x: f32) -> bool {"), true);
  assertEquals(ARGMAX_WGSL.includes("  if (na != nb) {\n    return nb;\n  }"), true);
  assertEquals(ARGMAX_WGSL.includes("  if (na) {\n    return ib < ia;\n  }"), true);
  // 出力は i32 の添字（f32 で書くと語彙 2^24 超で隣の token に丸まる）
  assertEquals(ARGMAX_WGSL.includes("read_write> out: array<i32>;"), true);
  assertEquals(ARGMAX_WGSL.includes("out[row] = i32(scratch_index[0u]);"), true);
  // 行ループは grid-stride（行 reduce と同じ workgroup_id × num_workgroups の送り）
  assertEquals(ARGMAX_WGSL.includes("var row = wid.x;"), true);
  assertEquals(ARGMAX_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals([...argmaxParams(7, 3)], [7, 3, ARGMAX_NEG_INF_BITS, 0]);
  assertEquals(argmaxParams(7, 3).byteLength, 16, "uniform struct の 16 バイト整列");
  assertThrows(() => argmaxParams(7, 0), CodegenError, "dim は正整数");
  assertThrows(() => argmaxParams(-1, 3), CodegenError);
  assertThrows(() => argmaxParams(7, 1.5), CodegenError);
});

/**
 * topk（ADR 0068 決定 3）。**値では検出できない**契約を生成物の側から固定する:
 *
 * 1. **2 相の形**（レーン局所 top-k → k ラウンドのトーナメント merge）— 全語彙 argsort でも
 *    「k 回の行 reduce」でもないこと。後者は行を k 回読むので ADR が避けている高コスト側。
 * 2. **タイブレークの向きと述語の共有** — `ib < ia`（最小 index）を `>` に書き換えても shape も
 *    dtype も合ってしまう。走査・木・ラウンド跨ぎが**同じ 1 本の述語**を呼ぶことが、結合順に
 *    依らず「値降順・同値なら index 昇順」になる根拠。
 * 3. **identity が −inf + 番兵 index = dim** — 有限 sentinel（reduce 族の `-F32_MAX`）に
 *    戻っていないこと。ビット列は params 3 語目で運ぶ。
 * 4. **NaN 判定がビット列** — 素の比較に戻すと NaN が黙って負ける。
 * 5. **カーソルを進めるのは勝った要素の持ち主だけ**（`won % W == lid`）— ここが壊れると同じ
 *    要素が k 回出る / 別の要素が飛ばされるが、shape も dtype も合ったままになる。
 * 6. **k の実装上限が workgroup storage から静的に決まる**こと（縮退しない）。
 */
Deno.test("topk は 2 相（レーン局所 top-k → トーナメント merge）で、最小 index を保存する述語を共有する", () => {
  assertEquals(TOPK_WORKGROUP_SIZE, 32);
  assertEquals(topkKey(4), `topk:v1:f32+i32:lastdim:desc:minindex:k4:wg${TOPK_WORKGROUP_SIZE}`);
  // キーは k を含む（k ごとに別 WGSL）
  assertEquals(topkKey(1) === topkKey(2), false);
  const wgsl = topkWgsl(4);
  // ① 出力 2 本（値 f32 + 添字 i32）を別の束縛で書く
  assertEquals(wgsl.includes("@binding(2) var<storage, read_write> values: array<f32>;"), true);
  assertEquals(wgsl.includes("@binding(3) var<storage, read_write> indices: array<i32>;"), true);
  // ① 相 1 = レーンごとの候補ブロック（k 語 × W レーン）/ 相 2 = W 者トーナメントの先頭
  assertEquals(wgsl.includes("var<workgroup> cand_value: array<f32, 128>;"), true);
  assertEquals(wgsl.includes("var<workgroup> cand_index: array<u32, 128>;"), true);
  assertEquals(wgsl.includes("var<workgroup> head_value: array<f32, 32>;"), true);
  assertEquals(wgsl.includes("var<workgroup> head_index: array<u32, 32>;"), true);
  // ① 行の読み出しは 1 回だけ（走査ループは 1 本・k のループは merge 側にしかない）
  assertEquals((wgsl.match(/let v = x\[base \+ i\];/g) ?? []).length, 1);
  assertEquals(wgsl.includes("let block = lid * 4u;"), true);
  // ② 述語は 1 箇所定義で、走査・木の両方が同じ関数を呼ぶ（向きの取り違えを構造で潰す）
  assertEquals(wgsl.includes("fn topk_beats(vb: f32, ib: u32, va: f32, ia: u32)"), true);
  assertEquals(wgsl.includes("return vb > va || (vb == va && ib < ia);"), true);
  assertEquals(
    wgsl.includes("if (topk_beats(v, i, cand_value[block + 3u], cand_index[block + 3u])) {"),
    true,
  );
  assertEquals(
    wgsl.includes(
      "while (s > 0u && topk_beats(v, i, cand_value[block + s - 1u], cand_index[block + s - 1u])) {",
    ),
    true,
  );
  assertEquals(
    wgsl.includes("if (topk_beats(other, other_at, head_value[lid], head_index[lid])) {"),
    true,
  );
  // ③ identity は −inf（params 経由）+ 番兵 index = dim（全 −inf 行でも答えが定義される根拠）
  assertEquals(wgsl.includes("let neg_inf = bitcast<f32>(params.neg_inf);"), true);
  assertEquals(wgsl.includes("cand_value[block + s] = neg_inf;"), true);
  assertEquals(wgsl.includes("cand_index[block + s] = dim;"), true);
  // MUST: reduce 族の有限 sentinel が紛れ込んでいない
  assertEquals(wgsl.includes("3.402823466e38"), false);
  // ④ NaN はビット列で判定して「最大」として扱う（argmax と同一本文）
  assertEquals(wgsl.includes("fn is_nan_bits(x: f32) -> bool {"), true);
  assertEquals(wgsl.includes("  if (na != nb) {\n    return nb;\n  }"), true);
  assertEquals(wgsl.includes("  if (na) {\n    return ib < ia;\n  }"), true);
  assertEquals(
    ARGMAX_WGSL.includes("return vb > va || (vb == va && ib < ia);"),
    true,
    "argmax と同じ述語本文（k=1 が argmax と一致することの前提）",
  );
  // ⑤ 出力は行ごとに k 語・カーソルを進めるのは勝った要素の持ち主だけ
  assertEquals(wgsl.includes("values[row * 4u + r] = head_value[0u];"), true);
  assertEquals(wgsl.includes("indices[row * 4u + r] = i32(won);"), true);
  assertEquals(wgsl.includes("if (won % 32u == lid) {"), true);
  assertEquals(wgsl.includes("cursor = cursor + 1u;"), true);
  // 行ループは grid-stride（argmax / 行 reduce と同じ workgroup_id × num_workgroups の送り）
  assertEquals(wgsl.includes("var row = wid.x;"), true);
  assertEquals(wgsl.includes("row = row + nwg.x;"), true);
  // k=1 の縮退形: ブロックの末尾（最弱）と先頭が同じ語になり、挿入ループは 1 度も回らない
  assertEquals(
    topkWgsl(1).includes("if (topk_beats(v, i, cand_value[block + 0u], cand_index[block + 0u])) {"),
    true,
  );

  // ⑥ 実装上限は workgroup storage から静的に決まる（8·W·(k+1) バイト）
  assertEquals(topkWorkgroupStorageBytes(1), 8 * 32 * 2);
  assertEquals(topkWorkgroupStorageBytes(63), 16384);
  assertEquals(topkMaxK(16384), TOPK_CORE_LIMIT_MAX_K);
  assertEquals(TOPK_CORE_LIMIT_MAX_K, 63);
  // 上限が device limit で動く（要求した limit がそのまま上限になる）
  assertEquals(topkMaxK(32768), 127);
  assertEquals(topkMaxK(1024), 3);
  assertTopkK(63, 16384, "t");
  // MUST: 超過は縮退させず、**上限値つきで** fail loudly（利用者が k をどこまで下げれば
  // よいか診断だけで分かる形）
  assertThrows(() => assertTopkK(64, 16384, "t"), CodegenError, "実装上限 63 を超える");
  assertThrows(() => assertTopkK(4, 1024, "t"), CodegenError, "実装上限 3 を超える");
  assertThrows(() => assertTopkK(0, 16384, "t"), CodegenError, "k は正整数");
  assertThrows(() => assertTopkK(1.5, 16384, "t"), CodegenError);

  assertEquals([...topkParams(7, 3)], [7, 3, TOPK_NEG_INF_BITS, 0]);
  assertEquals(topkParams(7, 3).byteLength, 16, "uniform struct の 16 バイト整列");
  // MUST: params に k を載せない（WGSL に焼いてある — 二重管理を構造で禁じる）
  assertEquals(topkParams(7, 3).length, 4);
  assertThrows(() => topkParams(7, 0), CodegenError, "dim は正整数");
  assertThrows(() => topkParams(-1, 3), CodegenError);
});

Deno.test("half-split RoPE は積を workgroup u32 へ丸め、一様 barrier 後に加算する", () => {
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
  assertEquals(
    ROPE_WGSL.includes("out[i] = bitcast<f32>(pair.x) + bitcast<f32>(pair.y);"),
    true,
  );
  assertEquals([...ropeParams(2 * 3 * 128, 3, 128)], [768, 3, 128, 64]);
  assertEquals(ropeParams(768, 3, 128).byteLength, 16);
  assertThrows(() => ropeParams(768, 0, 128), CodegenError, "sequence");
  assertThrows(() => ropeParams(768, 3, 127), CodegenError, "headDim");
  assertThrows(() => ropeParams(767, 3, 128), CodegenError, "整数行");
  assertThrows(() => ropeParams(0x1_0000_0000, 1, 128), CodegenError, "u32");
});

Deno.test("NCHW 2x upsample は入力要素ごとに 2x2 を書き、params の行境界を検査する", () => {
  assertEquals(UPSAMPLE_2X_KEY, `upsample2x:v1:nchw:f32:wg${UPSAMPLE_2X_WORKGROUP_SIZE}`);
  assertEquals(UPSAMPLE_2X_WORKGROUP_SIZE, 256);
  assertEquals(
    UPSAMPLE_2X_WGSL.includes("let base = 2u * row * params.out_width + 2u * col;"),
    true,
  );
  // ビット複製なので入出力とも u32 view（NaN payload / subnormal / ±0 が正規化されない）
  assertEquals(UPSAMPLE_2X_WGSL.includes("x: array<u32>"), true);
  assertEquals(UPSAMPLE_2X_WGSL.includes("out: array<u32>"), true);
  assertEquals(UPSAMPLE_2X_WGSL.includes("out[base + params.out_width + 1u] = value;"), true);
  assertEquals([...upsample2xParams(24, 3)], [24, 3, 6, 0]);
  assertEquals(upsample2xParams(24, 3).byteLength, 16);
  assertThrows(() => upsample2xParams(-1, 3), CodegenError, "u32");
  assertThrows(() => upsample2xParams(24, 0), CodegenError, "width");
  assertThrows(() => upsample2xParams(25, 3), CodegenError, "整数行");
  assertThrows(() => upsample2xParams(0x4000_0000, 1), CodegenError, "4 * n");
  assertThrows(() => upsample2xParams(0, 0x8000_0000), CodegenError, "2 * width");
});

Deno.test("双線形 resample は出力 1 要素 = 1 スレッドで、末尾タップを整数比較で決める", () => {
  assertEquals(
    UPSAMPLE_BILINEAR2D_KEY,
    `upsample_bilinear2d:v1:nchw:f32:align_corners:wg${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}`,
  );
  assertEquals(UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE, 256);
  // MUST: scale はホストが割って params で運ぶ（WGSL の f32 除算は 2.5 ULP まで許され、
  // 末尾の源座標が入力の末尾添字をわずかに下回ると端の厳密一致がバックエンド依存で壊れる）。
  // シェーダ側に残る除算は u32 の添字分解だけ = f32 の除算が 1 つも無いことがその宣言。
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("  scale_h: f32,"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("params.scale_h * f32(oy)"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("params.scale_w * f32(ox)"), true);
  assertEquals(/f32\([^)]*\) *\//.test(UPSAMPLE_BILINEAR2D_WGSL), false, "f32 の除算が残っている");
  // MUST: 末尾タップの特例は**整数比較**（f32 の min/max は NaN を飲む — ADR 0020）
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("if (y0 + 1u < params.in_h) {"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("if (x0 + 1u < params.in_w) {"), true);
  assertEquals(UPSAMPLE_BILINEAR2D_WGSL.includes("clamp("), false, "座標の clamp は使わない");
  // 式木は H が外・W が内（torch の CUDA カーネルと同形 — 括弧が動くと丸め列が変わる）
  assertEquals(
    UPSAMPLE_BILINEAR2D_WGSL.includes(
      "out[i] = ly0 * (lx0 * x[row0 + x0] + lx1 * x[row0 + x1])",
    ),
    true,
  );
  const params = upsampleBilinear2dParams(2 * 3 * 7 * 9, 4, 5, 7, 9);
  assertEquals([...params.slice(0, 5)], [378, 4, 5, 7, 9]);
  // scale = (in − 1) / (out − 1) を f32 で（4→7 も 5→9 もちょうど 0.5）
  assertEquals([...new Float32Array(params.buffer).slice(5, 7)], [0.5, 0.5]);
  assertEquals(params[7], 0, "予約語は 0");
  // 出力長 1 の軸は scale 0（torch の area_pixel_compute_scale の特例）
  assertEquals(
    [...new Float32Array(upsampleBilinear2dParams(2, 6, 7, 1, 1).buffer).slice(5, 7)],
    [0, 0],
  );
  // 入力長 1 の軸は scale 0（末尾特例が全出力で発火する形）
  assertEquals(
    [...new Float32Array(upsampleBilinear2dParams(12, 1, 6, 4, 3).buffer).slice(5, 7)],
    [0, 2.5],
  );
  // uniform struct の整列は 16 バイト（u32 5 語 + f32 2 語 + 予約 1 語 = 32 バイト）
  assertEquals(params.byteLength, 32);
  assertThrows(() => upsampleBilinear2dParams(-1, 4, 5, 7, 9), CodegenError, "u32");
  // 空間軸の 0 は WGSL 側の `in_size - 1u` が回り込む（沈黙の範囲外読み出し）
  assertThrows(() => upsampleBilinear2dParams(63, 0, 5, 7, 9), CodegenError, "heightIn");
  assertThrows(() => upsampleBilinear2dParams(63, 4, 0, 7, 9), CodegenError, "widthIn");
  assertThrows(() => upsampleBilinear2dParams(0, 4, 5, 0, 9), CodegenError, "heightOut");
  assertThrows(() => upsampleBilinear2dParams(0, 4, 5, 7, 0), CodegenError, "widthOut");
  // 出力平面の整数倍でない n は平面の添字分解がずれる
  assertThrows(() => upsampleBilinear2dParams(64, 4, 5, 7, 9), CodegenError, "整数倍");
});

Deno.test("DCNv2 は範囲判定を正の形で書き、NaN をビット列で分けて伝播させる", () => {
  assertEquals(
    DEFORM_CONV2D_KEY,
    `deform_conv2d:v1:nchw:f32:dcnv2:wg${DEFORM_CONV2D_WORKGROUP_SIZE}`,
  );
  assertEquals(DEFORM_CONV2D_WORKGROUP_SIZE, 256);
  // MUST: 範囲判定は正の形（NaN は false 側へ落ちる）。負の形（`<= -1 ||`）に書き換えると
  // NaN が「範囲内」に落ちて i32(floor(NaN)) の未定義へ直行する。
  assertEquals(
    DEFORM_CONV2D_WGSL.includes(
      "let inside = sy > -1.0 && sy < f32(dims.height_in)\n    && sx > -1.0 && sx < f32(dims.width_in);",
    ),
    true,
  );
  // MUST: NaN はビット列で判定して**伝播**させる（正の形だけだと 0 寄与に落ちる）。
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u;"),
    true,
  );
  assertEquals(DEFORM_CONV2D_WGSL.includes("return bitcast<f32>(dims.oob);"), true);
  // MUST: NaN のビット列は params で運ぶ（定数式の `bitcast<f32>(0x…)` は「const-expression が
  // NaN」としてシェーダ生成エラーにする実装がありうる — gather / embedding と同じ規律）。
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN を定数式で作っている（params 経由にする）",
  );
  // MUST: 座標の clamp は使わない（境界外は clamp ではなくゼロ埋め）
  assertEquals(DEFORM_CONV2D_WGSL.includes("clamp("), false, "座標の clamp は使わない");
  // MUST: 偶数チャネル = y / 奇数 = x
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("offsets[offset_base + 2u * tap * plane_out]"),
    true,
  );
  assertEquals(
    DEFORM_CONV2D_WGSL.includes("offsets[offset_base + (2u * tap + 1u) * plane_out]"),
    true,
  );
  // MUST: mask は補間の後・重みの前（`(m · v) · w`）で、縮約順は conv2d と同じ (ic, kh, kw)
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
    "ic ループが kh より外（conv2d と同じ縮約順）",
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
  assertEquals(
    [...params],
    [80, 1, 3, 5, 4, 5, 4, 4, 3, 2, 1, 0, DEFORM_CONV2D_OOB_BITS, 0, 0, 0],
  );
  // uniform struct の整列は 16 バイト（13 語ぶんの内容を 16 語 = 64 バイト確保する）
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
    "非負整数",
  );
  // 空間長 0 は全タップが範囲外判定に落ち、出力が bias 一色になる沈黙誤値
  assertThrows(() => deformConv2dParams({ ...dims, heightIn: 0 }), CodegenError, "height_in");
  assertThrows(() => deformConv2dParams({ ...dims, widthIn: 0 }), CodegenError, "width_in");
  assertThrows(() => deformConv2dParams({ ...dims, kernelW: 0 }), CodegenError, "kernel_w");
  assertThrows(() => deformConv2dParams({ ...dims, batch: 0 }), CodegenError, "batch");
});

// GRU 隠れ側スキャン（ADR 0056）。この op のビット同一門（tests/gpu_gru_scan_parity_test.ts）は
// 「更新式が分解形の逐語であること」と「fma 縮約を丸め障壁で止めていること」に全面的に
// 依存しているので、生成物の側でもその 2 点を固定する（実 GPU が無い環境でも赤くなる形）。
Deno.test("gru_scan は更新式を分解形の逐語で書き、mul と add の間に丸め障壁を挟む", () => {
  assertEquals(GRU_SCAN_WORKGROUP_SIZE, 256);
  assertEquals(GRU_SCAN_MAX_HIDDEN, 256);
  assertEquals(
    gruScanKey("forward"),
    `gru_scan:v1:f32:forward:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`,
  );
  assertEquals(
    gruScanKey("reverse"),
    `gru_scan:v1:f32:reverse:wg${GRU_SCAN_WORKGROUP_SIZE}:h${GRU_SCAN_MAX_HIDDEN}`,
  );
  assertNotEquals(gruScanKey("forward"), gruScanKey("reverse"));
  const invalid = "backward" as GruScanDirection;
  assertThrows(() => gruScanKey(invalid), CodegenError, "走査方向が不正");
  assertThrows(() => gruScanWgsl(invalid), CodegenError, "走査方向が不正");

  const forward = gruScanWgsl("forward");
  const reverse = gruScanWgsl("reverse");
  for (const [where, wgsl] of [["forward", forward], ["reverse", reverse]] as const) {
    // MUST: 縮約は k 昇順の逐次で字面も GEMM と同じ（`acc = acc + a * b`）
    assertEquals(wgsl.includes("acc_r = acc_r + w_hh[row_r + k] * hk;"), true, where);
    // MUST: bias は last（conv 系の bias-first を写すと linear とのビット同一が崩れる）
    assertEquals(
      wgsl.includes("sigmoid_stable((acc_r + b_hh[lid]) + gi[gi_base + lid])"),
      true,
      where,
    );
    // MUST: 丸め障壁 ①（n の積）と ②（更新式の積）が workgroup memory 往復で残っている
    assertEquals(wgsl.includes("stage[lid] = bitcast<u32>(gh_n * gate_r);"), true, where);
    assertEquals(
      wgsl.includes("stage[lid] = bitcast<u32>((h_prev - cand) * gate_z);"),
      true,
      where,
    );
    assertEquals(wgsl.includes("let h_next = bitcast<f32>(stage[lid]) + cand;"), true, where);
    // MUST: `(1 − z)·n + z·h` へ書き換えない（同値だが別の丸め列 — ADR 0056 決定 3）
    assertEquals(wgsl.includes("1.0 - gate_z"), false, where);
    assertEquals(wgsl.includes("var<workgroup> h_shared: array<f32, 256>;"), true, where);
  }
  // MUST: 2 方向の差は走査順の 1 行だけ（出力の時間添字は両方とも `t`）
  assertEquals(forward.includes("let t = step;"), true);
  assertEquals(reverse.includes("let t = dims.time - 1u - step;"), true);
  assertEquals(
    forward.replaceAll("let t = step;", "").replaceAll("(forward,", "(reverse,"),
    reverse.replaceAll("let t = dims.time - 1u - step;", ""),
    "走査順の 1 行とヘッダ以外が 2 方向で食い違う",
  );

  assertEquals([...gruScanParams({ time: 5, batch: 2, hidden: 128 })], [5, 2, 128, 0]);
  // uniform struct の整列は 16 バイト（3 語ぶんの内容でも 4 語確保する）
  assertEquals(gruScanParams({ time: 5, batch: 2, hidden: 128 }).byteLength, 16);
  assertThrows(
    () => gruScanParams({ time: -1, batch: 2, hidden: 4 }),
    CodegenError,
    "time は u32 の非負整数",
  );
  assertThrows(
    () => gruScanParams({ time: 5, batch: 2, hidden: 0 }),
    CodegenError,
    "hidden は正整数",
  );
  // MUST: 上限超過は fail loudly（黙って縮退させると h_shared の範囲外書き込みになる）
  assertThrows(
    () => gruScanParams({ time: 5, batch: 2, hidden: GRU_SCAN_MAX_HIDDEN + 1 }),
    CodegenError,
    "上限 256 を超える",
  );
});

Deno.test("SiLU は sigmoid の f32 格納境界を u32 workgroup staging で保ち、mul 順をキーへ残す", () => {
  assertEquals(SILU_WORKGROUP_SIZE, 256);
  assertEquals(siluKey("x-sigmoid"), `silu:v1:x-sigmoid:f32:wg${SILU_WORKGROUP_SIZE}`);
  assertEquals(siluKey("sigmoid-x"), `silu:v1:sigmoid-x:f32:wg${SILU_WORKGROUP_SIZE}`);
  assertNotEquals(siluKey("x-sigmoid"), siluKey("sigmoid-x"));
  const invalid = "foreign-order" as SiluMulOrder;
  assertThrows(() => siluKey(invalid), CodegenError, "入力順が不正");
  assertThrows(() => siluWgsl(invalid), CodegenError, "入力順が不正");

  const xs = siluWgsl("x-sigmoid");
  const sx = siluWgsl("sigmoid-x");
  for (const [where, wgsl] of [["x-sigmoid", xs], ["sigmoid-x", sx]] as const) {
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
    // primitive sigmoid の安定式を共有する。素朴な exp(-x) 形への退行を防ぐ。
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

Deno.test("adaLN は正規化本体を素の layer_norm と共有し、変調と積を u32 staging で 2 段に区切る", () => {
  assertEquals(ADALN_NORM_WORKGROUP_SIZE, 256);
  // MUST: 素の layer_norm と同じ幅（本文を共有している以上、片方だけ動かすと WGSL が壊れる —
  // src/kernels/adaln-norm.ts の MUST）。導出関係を門に出す。
  assertEquals(ADALN_NORM_WORKGROUP_SIZE, LAYER_NORM_WORKGROUP_SIZE);
  assertEquals(ADALN_NORM_KEY, `adaln_norm:v1:lastdim:f32:wg${ADALN_NORM_WORKGROUP_SIZE}`);
  assertNotEquals(ADALN_NORM_KEY, LAYER_NORM_KEY);

  // MUST: 行統計（2 パス / 母分散）と affine は**同一文字列**。別々に書くと素の列と
  // 融合版で縮約順が割れ、有限値のビット一致が崩れる。
  assertEquals(LAYER_NORM_WGSL.includes(LAYER_NORM_ROW_STATS_WGSL), true, "素の側の共有");
  assertEquals(ADALN_NORM_WGSL.includes(LAYER_NORM_ROW_STATS_WGSL), true, "融合側の共有");
  assertEquals(
    LAYER_NORM_WGSL.includes(`out[base + o] = ${LAYER_NORM_AFFINE_WGSL};`),
    true,
    "素の側の affine",
  );
  assertEquals(ADALN_NORM_WGSL.includes(`(${LAYER_NORM_AFFINE_WGSL}) * modulation`), true);

  // 丸め障壁 2 段（`1 + scale` と `t * s`）。バッファは 6 in + 1 out。
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
  assertEquals(ADALN_NORM_WGSL.includes("@binding(8)"), false, "storage は 7 本まで");
  // 出力は block ループ（workgroup 一様）。`o = lid` の while のままでは barrier を置けない。
  assertEquals(ADALN_NORM_WGSL.includes("while (block < blocks) {"), true);

  // params は素の layer_norm と同一の生成器（rows / dim / eps）。
  assertEquals([...adalnNormParams(4, 8, 1e-6)], [...layerNormParams(4, 8, 1e-6)]);
  assertEquals(adalnNormParams(4, 8, 1e-6).byteLength, 16);
  assertThrows(() => adalnNormParams(4, 0, 1e-6), CodegenError, "dim");
  assertThrows(() => adalnNormParams(4, 8, 0), CodegenError, "eps");
});

Deno.test("elementwise params はスカラ attr を末尾に f32 のビット列で載せる", () => {
  // 出力 [2,3]（rank2）+ 入力 1 本 → dims 2 語・strides 2 語のあとにスカラが並ぶ
  const clamp: ElementwiseSpec = { op: "clamp", rank: 2, dtype: "f32" };
  const params = elementwiseParams(clamp, [2, 3], [[2, 3]], [-1.5, 2.25]);
  assertEquals(params.length, 1 + 2 + 2 + 2);
  assertEquals([...params.slice(0, 5)], [6, 2, 3, 3, 1]);
  const floats = new Float32Array(params.buffer);
  assertEquals(floats[5], -1.5);
  assertEquals(floats[6], 2.25);
  // 既定は 0 本（既存 op の params レイアウトは変わらない）
  assertEquals(
    elementwiseParams({ op: "neg", rank: 2, dtype: "f32" }, [2, 3], [[2, 3]]).length,
    5,
  );
  assertThrows(
    () => elementwiseParams({ op: "clamp_min", rank: 1, dtype: "f32" }, [2], [[2]], [Number.NaN]),
    CodegenError,
  );
  // MUST: 入力の本数は生成時のアリティと照合する。食い違うと WGSL 側の
  // `1 + rank + arity·rank + index` とスカラの書き込み位置がずれ、範囲外読みの沈黙誤値になる。
  assertThrows(
    () => elementwiseParams({ op: "clamp", rank: 1, dtype: "f32" }, [4], [[4], [4]], [0, 1]),
    CodegenError,
    "アリティ",
  );
  // スカラの本数も同じ 1 箇所で見る（多すぎるぶんは黙って捨てられていた）
  assertThrows(
    () => elementwiseParams(clamp, [2, 3], [[2, 3]], [-1.5]),
    CodegenError,
    "スカラ",
  );
  // rank も spec 側が正本（出力 shape だけから導かない）
  assertThrows(() => elementwiseParams(clamp, [6], [[6]], [-1.5, 2.25]), CodegenError, "rank");
});

Deno.test("elementwise params は右詰め broadcast の stride を 0 にする", () => {
  // 出力 [2,3,4] に対し in0=[2,3,4]（連続）、in1=[3,1]（右詰めで [1,3,1]）
  const params = elementwiseParams({ op: "add", rank: 3, dtype: "f32" }, [2, 3, 4], [
    [2, 3, 4],
    [3, 1],
  ]);
  assertEquals([...params.slice(0, 4)], [24, 2, 3, 4]);
  assertEquals([...params.slice(4, 7)], [12, 4, 1]);
  assertEquals([...params.slice(7, 10)], [0, 1, 0]);
});

Deno.test("elementwise params はスカラ入力を右詰めで吸収する", () => {
  const params = elementwiseParams({ op: "add", rank: 1, dtype: "f32" }, [5], [[5], [1]]);
  assertEquals([...params], [5, 5, 1, 0]);
});

Deno.test("uniform で渡す params は 16 バイト整列を満たす", () => {
  // WGSL の uniform アドレス空間では struct の整列が 16 バイトになる
  assertEquals(reduceParams(7, 3).byteLength, 16);
  assertEquals(matmulParams(2, 3, 4).byteLength, 16);
  assertEquals(bmmParams(2, 3, 4).byteLength, 16);
  assertEquals(gatherParams(12, 3, 9).byteLength, 16);
  assertEquals(padParams(2, 3, 1, 2).byteLength, 16);
  // flip の Params は 3 語だが、確保は 16 バイト（uniform struct の整列）
  assertEquals(flipParams(2, 3, 4).byteLength, 16);
  assertEquals(axisReduceParams(35, 384, 5).byteLength, 16);
  assertEquals([...reduceParams(7, 3).slice(0, 2)], [7, 3]);
  // 軸変種は {out_count, axis_len, inner} の 3 語（末尾は整列のための余白）
  assertEquals([...axisReduceParams(35, 384, 5)], [35, 384, 5, 0]);
  assertEquals([...matmulParams(2, 3, 4).slice(0, 3)], [2, 3, 4]);
  // bmm も 3 語 {m,n,k}。バッチ数は wid.z から導くので uniform に載せない（載せると
  // WGSL が一度も読まない死んだフィールドになる）
  assertEquals([...bmmParams(2, 3, 4)], [2, 3, 4, 0]);
  // 末尾は範囲外添字に書く NaN のビット列（WGSL に NaN リテラルが無いので params で運ぶ）
  assertEquals([...gatherParams(12, 3, 9)], [12, 3, 9, GATHER_OOB_BITS]);
  assertEquals(Number.isNaN(new Float32Array(new Uint32Array([GATHER_OOB_BITS]).buffer)[0]), true);
});

// MUST: bmm は matmul と**別キーの別カーネル**（骨格は共有だがバッチ軸の断片が違う）。
Deno.test("bmm は matmul と別カーネルで、バッチ軸のオフセットを持つ", () => {
  for (const v4 of [false, true]) {
    const where = `v4=${v4}`;
    assertNotEquals(bmmKey(v4), matmulKey(v4), where);
    assertNotEquals(bmmWgsl(v4), matmulWgsl(v4), where);
    // バッチは workgroup の z 軸から取り、3 バッファとも行列 1 枚ぶんのオフセットで読み書きする。
    // MUST: v4 では base も **quad 単位**（`m * k4`）— 要素単位のまま組むと batch ≥ 2 で
    // 隣のバッチを読み書きし、例外なしの誤値になる（タイル境界の GPU ケースが唯一の検出器）。
    const unit = (dim: string) => v4 ? `${dim}4` : `dims.${dim}`;
    assertEquals(bmmWgsl(v4).includes(`let abase = wid.z * dims.m * ${unit("k")};`), true, where);
    assertEquals(bmmWgsl(v4).includes(`let bbase = wid.z * dims.k * ${unit("n")};`), true, where);
    assertEquals(bmmWgsl(v4).includes(`let cbase = wid.z * dims.m * ${unit("n")};`), true, where);
    // 3 バッファとも base 経由で触る（片方だけ素の添字に戻る誤りを塞ぐ）。添字の接尾辞は
    // 充填スロット / 出力行の番号（1 スレッドが複数行を持つ幾何 — src/kernels/gemm-geometry.ts）
    assertEquals(bmmWgsl(v4).includes("abase + arow0 *"), true, where);
    assertEquals(bmmWgsl(v4).includes("bbase + brow0 *"), true, where);
    assertEquals(bmmWgsl(v4).includes("cbase + orow0 *"), true, where);
    // matmul 側にバッチの概念が漏れていない（キー衝突も生成物の変化も無い）
    assertEquals(matmulWgsl(v4).includes("wid.z"), false, where);
    assertEquals(matmulWgsl(v4).includes("base + arow"), false, where);
  }
});

/**
 * 融合 attention（ADR 0023）の生成物。**ビット同一が設計の核**なので、その前提を作っている
 * 4 点（半スケールが q/k の両側 / 行統計が softmax のパス①② と逐語同一 / ③ が
 * `exp(S−m)·inv` を成分ごとに評価 / 行統計の添字がバッチ base 込み）を生成物の側で固定する。
 */
Deno.test("融合 attention の 3 カーネルは分解経路とビット同一になる形を生成する", async () => {
  for (const v4 of [false, true]) {
    const qk = attentionQkWgsl(v4);
    const where = `v4=${v4}`;
    // MUST: scale は **q 側と k 側の両方**（半スケール契約）。片側だけにすると値が √ 倍ずれ、
    // 内積の後に 1 度だけ掛ける形にすると丸め列が変わってビット同一が壊れる。
    // MUST: 充填スロット**全て**に掛かる（1 スレッドが複数列を埋める幾何なので、スロット 0
    // だけに掛けると担当の半分が √ 倍ずれる）。スロット数は 2（既定幾何 — 下の幾何テスト）
    assertEquals(qk.includes("wv0 = wv0 * dims.scale;"), true, `${where}: k 側の scale`);
    assertEquals(qk.includes("wv1 = wv1 * dims.scale;"), true, `${where}: k 側の scale スロット 1`);
    assertEquals(
      (qk.match(/wv(\d+) = wv\1 \* dims\.scale;/g) ?? []).length,
      qk.split("var wv").length - 1,
      `${where}: k 側の scale はスロットの本数ちょうど`,
    );
    assertEquals(
      qk.includes("raw0.x * dims.scale") || qk.includes("q[arow_base0 + ak0] * dims.scale"),
      true,
      `${where}: q 側の scale`,
    );
    // scale は uniform の 4 語目（WGSL に焼かない — 値の種類でパイプラインが増える）
    assertEquals(qk.includes("  scale: f32,\n"), true, where);
    // k は [N,D] のまま読む（旧経路の permute(kᵀ) が消えることの生成物側の証拠）
    assertEquals(qk.includes("let krow_base0 = bbase + wcol0 * dims.k;"), true, where);
    // MUST: 転置読みの base は v4 でも**要素単位**（quad にすると 4 分の 1 の位置を読む）
    assertEquals(qk.includes("let bbase = wid.z * dims.n * dims.k;"), true, where);

    const pv = attentionPvWgsl(v4);
    // MUST: A タイルは P を実体化せず `exp(S − m) · inv` を**成分ごとのスカラ式**で評価する
    // （`vec4` へまとめて `exp` を掛けない — 超越関数のベクトル版は実装依存）。
    // MUST: 統計は**充填スロットごと**（1 スレッドが埋める行が別なら行統計も別）。
    // スロット 0 の統計を 2 行目にも使うと、2 行目だけが静かに別の行で正規化される。
    for (const slot of [0, 1]) {
      assertEquals(
        pv.split(`- row_max${slot}) * row_inv${slot}`).length - 1,
        4,
        `${where}: 正規化式はスロット ${slot} の 4 成分ぶんちょうど`,
      );
    }
    assertEquals(pv.includes("exp(vec4"), false, `${where}: exp をベクトルへまとめない`);
    // MUST: 行統計の添字はバッチ base 込み（B·H ≥ 2 で全バッチが 0 番の統計を使う誤りを塞ぐ）
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
    // ①③ は別カーネル・別キー（骨格は共有だが充填断片が違う）
    assertNotEquals(qk, pv, where);
    assertNotEquals(attentionQkKey(v4), attentionPvKey(v4), where);
    assertNotEquals(qk, bmmWgsl(v4), where);
    assertNotEquals(pv, bmmWgsl(v4), where);
  }

  // MUST: ② は現行 softmax のパス①② を**逐語で**切り出したもの。ツリー縮約の段・走査順・
  // identity・`1.0 / Σ` のどれかが変わればビット同一が壊れるので、softmax 側の本文と
  // 断片単位で突き合わせる（写し崩れの唯一の機械的検出器）。
  for (
    const shared of [
      "    var hi = -3.402823466e38;",
      "    var i = lid;\n    while (i < dim) {",
      "    var stride = 128u;",
      "        scratch[lid] = max(scratch[lid], scratch[lid + stride]);",
      "    let amax = scratch[0u];",
      "    var stride2 = 128u;",
      "        scratch[lid] = scratch[lid] + scratch[lid + stride2];",
      "    let inv = 1.0 / scratch[0u];",
    ]
  ) {
    assertEquals(SOFTMAX_WGSL.includes(shared), true, `softmax 側に無い断片: ${shared}`);
    assertEquals(
      ATTENTION_STATS_WGSL.includes(shared),
      true,
      `attention_stats 側に無い: ${shared}`,
    );
  }
  // 走査の本体は行の要素を読む式まで一致する（`x[base + …]` が `s[base + …]` になるだけ）
  assertEquals(
    ATTENTION_STATS_WGSL.includes("acc = acc + exp(s[base + j] - amax);"),
    true,
  );
  assertEquals(SOFTMAX_WGSL.includes("acc = acc + exp(x[base + j] - amax);"), true);
  // ② は行を書き戻さない（3 パス → 1 パス化の実体 — 出力は行あたり 2 語だけ）
  assertEquals(ATTENTION_STATS_WGSL.includes("stats[row * 2u] = amax;"), true);
  assertEquals(ATTENTION_STATS_WGSL.includes("stats[row * 2u + 1u] = inv;"), true);
  // 行方向は grid-stride（縮退ハーネスの対象 — tests/gpu_gridstride_test.ts）
  assertEquals(ATTENTION_STATS_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals(
    ATTENTION_STATS_KEY,
    `attention_stats:v1:f32:lastdim:safe:wg${ATTENTION_STATS_WORKGROUP_SIZE}`,
  );
  assertEquals(ATTENTION_STATS_WORKGROUP_SIZE, 256);

  // params: ① だけが 4 語目に f32 の scale を載せる
  const qkParams = attentionQkParams(5, 7, 4, 0.2973017692565918);
  assertEquals(qkParams.byteLength, 16);
  assertEquals([...qkParams.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkParams.buffer)[3], Math.fround(0.2973017692565918));
  assertEquals([...attentionPvParams(5, 4, 7)], [5, 4, 7, 0]);
  assertEquals([...attentionStatsParams(30, 7)], [30, 7, 0, 0]);
  assertEquals(attentionStatsParams(30, 7).byteLength, 16);
  assertThrows(() => attentionQkParams(5, 7, 4, Number.NaN), CodegenError);
  assertThrows(() => attentionQkParams(-1, 7, 4, 1), CodegenError);
  assertThrows(() => attentionStatsParams(30, 0), CodegenError);
  assertThrows(() => attentionStatsParams(-1, 7), CodegenError);
  // MUST: regcache 変種は `dim <= epc · 256` をここでも見る（生成側と二重だが、カーネル
  // 直呼びの経路も塞ぐ門）。超えた要素は max にも Σ にも入らず沈黙誤値になる。
  assertEquals([...attentionStatsParams(4, 4096, 16)], [4, 4096, 0, 0]);
  assertThrows(() => attentionStatsParams(4, 8192, 16), CodegenError, "担当範囲");
  assertThrows(
    () => attentionStatsParams(4, 4096, ATTENTION_STATS_REG_CACHE_MAX + 1),
    CodegenError,
    "上限",
  );
  // MUST: 被覆の突合より先に整数性を見る。生成側は非整数長を切り捨てて `1.5` を 1 スロット
  // （256 要素）で展開する一方、被覆計算は 1.5 × 256 = 384 を名乗るので、この門が無いと
  // dim 300 の 44 要素が縮約から黙って落ちる。
  assertThrows(() => attentionStatsParams(4, 300, 1.5), CodegenError, "整数でない");
  assertThrows(() => attentionStatsParams(4, 100, 0.5), CodegenError, "整数でない");
  // 整数の正常形（切り捨てが起きない形）は通る
  assertEquals([...attentionStatsParams(4, 300, 2)], [4, 300, 0, 0]);
  await Promise.resolve();
});

/**
 * 加算 mask 変種（ADR 0023 改訂）。**ビット同一の根拠は「丸めが増えるのは書き出しの 1 加算
 * だけ」**なので、生成物の側で ① epilogue の形と ②③ のバイト不変を固定する。
 */
Deno.test("attention の加算 mask は ①QK の epilogue だけを変え、②③ を 1 バイトも動かさない", () => {
  for (const v4 of [false, true]) {
    const where = `v4=${v4}`;
    const plain = attentionQkWgsl(v4);
    const masked = attentionQkWgsl(v4, "f32", "f32", true);
    assertNotEquals(plain, masked, where);
    // 束縛は S の次の 1 本だけ増える（mask 無しの変種は binding 4 を宣言しない）
    assertEquals(plain.includes("mask"), false, `${where}: mask 無しに mask が現れない`);
    assertEquals(
      masked.includes(`@group(0) @binding(${ATTENTION_QK_MASK_BINDING}) var<storage, read> mask:`),
      true,
      where,
    );
    // MUST: 縮約ループも共有タイル充填も 1 文字も変わらない（差は「先頭コメント +
    // mask の束縛 + 書き出しの加算」だけ）。行数の差＝加算行の本数になる形で固定する。
    const addLines = masked.split("\n").filter((line) => line.includes("let sv = "));
    assertEquals(
      masked.split("\n").length - plain.split("\n").length,
      addLines.length + 2 + (v4 ? 0 : 8),
      `${where}: 増えた行は mask 加算 + 束縛 2 行（スカラは行ごとの mbase 8 行）ぶんだけ`,
    );
    // MUST: mask の添字に**バッチ base を足さない**（契約は [1,1,M,N] の broadcast）。
    // 出力側は `cbase +` / `obase` を持つので、両者が同じ添字式でないことが要点。
    for (const line of addLines) {
      assertEquals(line.includes("cbase"), false, `${where}: mask にバッチ base: ${line}`);
      assertEquals(line.includes("obase"), false, `${where}: mask に出力 base: ${line}`);
    }
    assertEquals(
      masked.includes(
        v4
          ? "let sv = acc0_0 + mask[orow0 * n4 + ocq0];"
          : "let sv = acc0_0.x + mask[mbase + ocol];",
      ),
      true,
      `${where}: epilogue の形`,
    );
    // MUST: 加算は書き出しの直前 1 回だけ（1 出力要素 = 1 加算 — スロット数ぶん展開される）
    assertEquals(
      addLines.length,
      (plain.match(/\n\s+s\[/g) ?? []).length,
      `${where}: 加算の本数 = 書き出しの本数`,
    );
    // ③PV は mask を見ない（S は既に mask 済み）— キーも生成物も変わらない
    assertEquals(attentionPvWgsl(v4), attentionPvWgsl(v4), where);
    assertEquals(attentionPvWgsl(v4).includes("mask"), false, `${where}: ③PV に mask が無い`);
  }
  // ② 行統計は mask の軸をそもそも持たない（キーも WGSL も 1 本のまま）
  assertEquals(ATTENTION_STATS_WGSL.includes("mask"), false);
  // キーは s16 のさらに後ろに 1 語だけ足る（既存キーは 1 文字も動かない）
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
  // 同一構成が 2 通りのキーを持たない（mask の有無 × v4 × 格納の全数が一意）
  const keys = [false, true].flatMap((v4) =>
    [false, true].map((mask) => attentionQkKey(v4, "f32", "f32", mask))
  );
  assertEquals(new Set(keys).size, keys.length);
});

/**
 * GQA 変種の差分を**打ち消して**素の変種へ戻す（ADR 0067 決定 2 の「差分は base 算術のみ」）。
 *
 * 差分は ①uniform の `kv_repeat` 1 語 ②GQA の説明コメント 2 行 ③K / V の base 1 行
 * ④先頭コメントの 1 語 だけ。**戻せることで固定する**のが要点で、行数の一致だけを見る形に
 * すると縮約ループの中身が入れ替わっても通る。
 */
const undoGqa = (text: string, side: "k" | "v", mapped: string, plain: string): string =>
  text
    .split("\n")
    .filter((line) =>
      line !== "  kv_repeat: u32," &&
      !line.includes("GQA（ADR 0067 決定 2）") &&
      !line.includes("q / S / O / 行統計は q-head のまま")
    )
    .map((line) => line === mapped ? plain : line)
    .join("\n")
    .replace(`, GQA（${side} は kv-head へ整数除算で写す）`, "");

/**
 * GQA（整除 broadcast — ADR 0067 決定 1 / 2）。**`r = 1` の生成物とキーがバイト同一**であること
 * （既存スナップショット 133 本とビット同一門を凍結したまま席を足す MUST）と、GQA 変種の差分が
 * 「K / V の base 1 行 + uniform 1 語」に閉じていることを生成物の側で固定する。
 */
Deno.test("attention の GQA 変種は K / V の base 1 行と uniform 1 語だけを変える", () => {
  for (const v4 of [false, true]) {
    // 格納 / 計算 / mask の他の軸と**同時に立ちうる**ので、組み合わせごとに差分を見る
    // （s16 は v4 側しか存在しない — src/kernels/score-storage.ts）。
    const qkVariants = [
      ["素", (gqa: boolean) => attentionQkWgsl(v4, "f32", "f32", false, gqa)] as const,
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
      // 素の変種に GQA の痕跡が 1 文字も無い（= r=1 経路は 1 バイトも動いていない）
      assertEquals(plain.includes("kv_repeat"), false, `${where}: 素に kv_repeat`);
      assertEquals(plain.includes("GQA"), false, `${where}: 素に GQA`);
      // 差分は k の base 1 行（+ uniform 1 語 + コメント）だけ
      assertEquals(
        gqa.includes("  let bbase = (wid.z / dims.kv_repeat) * dims.n * dims.k;"),
        true,
        `${where}: k の base が kv-head へ写っていない`,
      );
      assertEquals(
        undoGqa(
          gqa,
          "k",
          "  let bbase = (wid.z / dims.kv_repeat) * dims.n * dims.k;",
          "  let bbase = wid.z * dims.n * dims.k;",
        ),
        plain,
        `${where}: 差分が base 算術 + uniform 1 語を超えている`,
      );
      // MUST: q（A）と S（C）は q-head のまま（写すのは K / V だけ — ADR 0067 決定 2）
      for (const line of gqa.split("\n")) {
        if (line.includes("let abase") || line.includes("let cbase")) {
          assertEquals(
            line.includes("kv_repeat"),
            false,
            `${where}: q / S 側が写っている: ${line}`,
          );
        }
      }
      // uniform は scale の**次**（並びは attentionQkParams と対）
      assertEquals(gqa.includes("  scale: f32,\n  kv_repeat: u32,\n}"), true, `${where}: 欄の並び`);
    }
    const pvVariants = [
      ["素", (gqa: boolean) => attentionPvWgsl(v4, "f32", "f32", gqa)] as const,
      ["c16", (gqa: boolean) => attentionPvWgsl(v4, "f16", "f32", gqa)] as const,
      ...(v4 ? [["s16", (gqa: boolean) => attentionPvWgsl(v4, "f32", "f16", gqa)] as const] : []),
    ];
    for (const [label, generate] of pvVariants) {
      const where = `③PV ${label} v4=${v4}`;
      const plain = generate(false);
      const gqa = generate(true);
      assertNotEquals(plain, gqa, where);
      assertEquals(plain.includes("kv_repeat"), false, `${where}: 素に kv_repeat`);
      // MUST: base は v4 で **quad 単位**（要素単位のままだと 4 分の 1 の位置を読む）
      const nUnit = v4 ? "n4" : "dims.n";
      assertEquals(
        gqa.includes(`  let bbase = (wid.z / dims.kv_repeat) * dims.k * ${nUnit};`),
        true,
        `${where}: v の base が kv-head へ写っていない`,
      );
      assertEquals(
        undoGqa(
          gqa,
          "v",
          `  let bbase = (wid.z / dims.kv_repeat) * dims.k * ${nUnit};`,
          `  let bbase = wid.z * dims.k * ${nUnit};`,
        ),
        plain,
        `${where}: 差分が base 算術 + uniform 1 語を超えている`,
      );
      // MUST: 行統計（rbase）も出力（cbase）も q-head のまま（S は [B·H,M,N] のまま）
      for (const line of gqa.split("\n")) {
        if (
          line.includes("let rbase") || line.includes("let cbase") || line.includes("let abase")
        ) {
          assertEquals(line.includes("kv_repeat"), false, `${where}: q 側が写っている: ${line}`);
        }
      }
    }
    // bmm は GQA の軸を持たない（③PV の base 式と同居しているので、漏れると
    // 「バッチ完全一致」の契約〈ADR 0022 / 0023〉が黙って割れる）
    assertEquals(bmmWgsl(v4).includes("kv_repeat"), false, `bmm v4=${v4}`);
    assertEquals(matmulWgsl(v4).includes("kv_repeat"), false, `matmul v4=${v4}`);
    assertEquals(linearWgsl("f32", v4).includes("kv_repeat"), false, `linear v4=${v4}`);
  }
  // ② 行統計は GQA の軸をそもそも持たない（S は q-head の [B·H, M, N] のまま）
  assertEquals(ATTENTION_STATS_WGSL.includes("kv_repeat"), false);

  // キーは mask のさらに後ろに 1 語だけ足る（r=1 の既存キーは 1 文字も動かない）
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
  // 同一構成が 2 通りのキーを持たない（v4 × mask × 格納 × GQA の全数が一意）
  const allKeys = [false, true].flatMap((v4) =>
    [false, true].flatMap((mask) =>
      [false, true].flatMap((gqa) => [
        attentionQkKey(v4, "f32", "f32", mask, gqa),
        attentionPvKey(v4, "f32", "f32", gqa),
      ])
    )
  );
  assertEquals(new Set(allKeys).size, 2 * 2 * 2 + 2 * 2, "①は mask×gqa×v4 / ③は gqa×v4 で一意");

  // params: `r` は uniform（キーには載らない）。①は 5 語目（整列で 32 B へ伸びる）・
  // ③は 4 語目（16 B のまま）で、**省略時のバイト列は従来どおり**。
  assertEquals(attentionQkParams(5, 7, 4, 0.5).byteLength, 16);
  const qkGqa = attentionQkParams(5, 7, 4, 0.5, 8);
  assertEquals(qkGqa.byteLength, 32);
  assertEquals([...qkGqa.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkGqa.buffer)[3], 0.5);
  assertEquals(qkGqa[4], 8);
  assertEquals([...attentionPvParams(5, 4, 7, 8)], [5, 4, 7, 8]);
  assertEquals(attentionPvParams(5, 4, 7, 8).byteLength, 16);
  // MUST: r = 0 は fail loudly（WGSL の u32 ゼロ除算は trap せず沈黙誤値になる）
  assertThrows(() => attentionQkParams(5, 7, 4, 0.5, 0), CodegenError, "kv_repeat");
  assertThrows(() => attentionPvParams(5, 4, 7, 0), CodegenError, "kv_repeat");
  assertThrows(() => attentionQkParams(5, 7, 4, 0.5, 1.5), CodegenError, "kv_repeat");
});

/**
 * **行窓変種**（クエリ行のブロック実行 — src/runtime/recipe-builder.ts の `#buildAttention`）の
 * 構造。スナップショットが凍結するのは 8 本の代表だけなので、「どの base が動き、どれが動いて
 * はいけないか」はここで全変種（v4 × 段 × i8a8）に対して機械確認する。
 *
 * MUST: 動くのは**片側だけ**（①QK は A = q / ③PV は C = O）。反対側と行統計はブロック相対の
 * ままで、写し間違えると S の一部が未書き込みか、別ブロックの行統計を掛けた沈黙誤値になる。
 * MUST: mask の**行**だけが base 算術の外で `row_offset` を要る（S はブロック相対の実体・
 * mask は全 M ぶんの実体）。
 */
Deno.test("融合 attention の行窓変種は片側の base と mask の行添字だけを変える", () => {
  for (const v4 of [false, true]) {
    const kUnit = v4 ? "k4" : "dims.k";
    const nUnit = v4 ? "n4" : "dims.n";
    // ①QK: A（q）だけが全 M ストライド + 行オフセット。C（S）はブロック相対のまま。
    const qkPlain = attentionQkWgsl(v4);
    const qkWindow = attentionQkWgsl(v4, "f32", "f32", false, false, true);
    assertEquals(qkPlain.includes("row_offset"), false, `①QK 素 v4=${v4}`);
    assertEquals(
      qkWindow.includes(
        `  let abase = wid.z * dims.rows_full * ${kUnit} + dims.row_offset * ${kUnit};`,
      ),
      true,
      `①QK 行窓 v4=${v4}: q の base が全 M ストライドになっていない`,
    );
    assertEquals(
      qkWindow.includes(`  let cbase = wid.z * dims.m * ${nUnit};`),
      true,
      `①QK 行窓 v4=${v4}: S がブロック相対でない`,
    );
    // ③PV: C（O）だけが全 M ストライド + 行オフセット。A（S）も行統計もブロック相対のまま。
    const pvPlain = attentionPvWgsl(v4);
    const pvWindow = attentionPvWgsl(v4, "f32", "f32", false, true);
    assertEquals(pvPlain.includes("row_offset"), false, `③PV 素 v4=${v4}`);
    assertEquals(
      pvWindow.includes(
        `  let cbase = wid.z * dims.rows_full * ${nUnit} + dims.row_offset * ${nUnit};`,
      ),
      true,
      `③PV 行窓 v4=${v4}: O の base が全 M ストライドになっていない`,
    );
    for (const prefix of ["  let abase", "  let rbase"]) {
      const line = pvWindow.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `③PV 行窓 v4=${v4}: ${line}`);
    }
    // mask は全 M ぶんの実体なので、行だけ `row_offset` を足して引く（列は S と同じ N）。
    const maskPlain = attentionQkWgsl(v4, "f32", "f32", true);
    const maskWindow = attentionQkWgsl(v4, "f32", "f32", true, false, true);
    assertEquals(
      maskPlain.includes("mask[(orow"),
      false,
      `①QK mask 素 v4=${v4}: 行にオフセットが入っている`,
    );
    assertEquals(
      maskWindow.includes(
        v4
          ? "mask[(orow0 + dims.row_offset) * n4 + ocq0]"
          : "let mbase = (orow0 + dims.row_offset)",
      ),
      true,
      `①QK mask 行窓 v4=${v4}: mask の行が全 M の添字になっていない`,
    );
    // i8a8: ①QK は量子化済み q と**その行 scale**の 2 本（片方だけ写すと別の行の scale が
    // 乗る沈黙誤値）。k / S 側はブロックに依らないので触らない。
    const qkI8a8Plain = attentionQkI8a8Wgsl(v4, true);
    const qkI8a8Window = attentionQkI8a8Wgsl(v4, true, "f32", undefined, true);
    assertEquals(qkI8a8Plain.includes("row_offset"), false, `①QK i8a8 素 v4=${v4}`);
    assertEquals(
      qkI8a8Window.includes("  let qbase = wid.z * dims.rows_full * k4 + dims.row_offset * k4;") &&
        qkI8a8Window.includes("  let qsbase = wid.z * dims.rows_full + dims.row_offset;"),
      true,
      `①QK i8a8 行窓 v4=${v4}: q ペイロードと行 scale の base が対で動いていない`,
    );
    for (const prefix of ["  let kbase", "  let ksbase", "  let sbase"]) {
      const line = qkI8a8Window.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `①QK i8a8 行窓 v4=${v4}: ${line}`);
    }
    // i8a8 ③PV は O の 1 本だけ（S / 行統計 / Vᵀ はブロック相対）。
    const pvI8a8Plain = attentionPvI8a8Wgsl(v4, true);
    const pvI8a8Window = attentionPvI8a8Wgsl(v4, true, "f32", undefined, true);
    assertEquals(pvI8a8Plain.includes("row_offset"), false, `③PV i8a8 素 v4=${v4}`);
    assertEquals(
      pvI8a8Window.includes(
        `  let obase = wid.z * dims.rows_full * ${nUnit} + dims.row_offset * ${nUnit};`,
      ),
      true,
      `③PV i8a8 行窓 v4=${v4}: O の base が全 M ストライドになっていない`,
    );
    for (const prefix of ["  let sbase", "  let rbase", "  let vbase", "  let vsbase"]) {
      const line = pvI8a8Window.split("\n").find((text) => text.startsWith(prefix)) ?? "";
      assertEquals(line.includes("row_offset"), false, `③PV i8a8 行窓 v4=${v4}: ${line}`);
    }
  }
  // ② 行統計は行窓の軸をそもそも持たない（S も統計もブロック相対で、行内で閉じる）。
  assertEquals(ATTENTION_STATS_WGSL.includes("row_offset"), false);

  // キーは GQA の**さらに後ろ**に 1 語（n=1 の既存キーは 1 文字も動かない）。
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
    "①QK i8a8 の行窓語は末尾",
  );
  assertEquals(
    attentionPvI8a8Key(true, true, "f16", undefined, true).endsWith(":s16:rwc"),
    true,
    "③PV i8a8 の行窓語は末尾",
  );
  // 同一構成が 2 通りのキーを持たない（v4 × mask × gqa × 行窓の全数が一意）。
  const allKeys = [false, true].flatMap((v4) =>
    [false, true].flatMap((mask) =>
      [false, true].flatMap((gqa) =>
        [false, true].flatMap((window) => [
          attentionQkKey(v4, "f32", "f32", mask, gqa, window),
          attentionPvKey(v4, "f32", "f32", gqa, window),
        ])
      )
    )
  );
  assertEquals(
    new Set(allKeys).size,
    2 * 2 * 2 * 2 + 2 * 2 * 2,
    "①は mask×gqa×窓×v4 / ③は gqa×窓×v4",
  );

  // params: 行窓の 2 語は**追加欄の末尾**（GQA の語位置を行窓の有無で動かさない）。
  const window = { offset: 2, rowsFull: 9 } as const;
  assertEquals(attentionQkParams(5, 7, 4, 0.5).byteLength, 16, "窓なしは従来どおり 16B");
  const qkWindowed = attentionQkParams(5, 7, 4, 0.5, undefined, window);
  assertEquals(qkWindowed.byteLength, 32);
  assertEquals([...qkWindowed.slice(0, 3)], [5, 7, 4]);
  assertEquals(new Float32Array(qkWindowed.buffer)[3], 0.5);
  assertEquals([qkWindowed[4], qkWindowed[5]], [2, 9]);
  const qkBoth = attentionQkParams(5, 7, 4, 0.5, 8, window);
  assertEquals([qkBoth[4], qkBoth[5], qkBoth[6]], [8, 2, 9], "GQA と併用しても kv_repeat が先");
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
  assertEquals(attentionQkI8a8Params(5, 12, 4, 0.5).byteLength, 16, "窓なしは従来どおり 16B");
  assertEquals(attentionPvI8a8Params(5, 4, 8).byteLength, 16, "窓なしは従来どおり 16B");
  // MUST: ブロックが全 M をはみ出す組は fail loudly（黙って通すと隣のバッチを読み書きする）。
  for (
    const build of [
      () => attentionQkParams(5, 7, 4, 0.5, undefined, { offset: 5, rowsFull: 9 }),
      () => attentionPvParams(5, 4, 7, undefined, { offset: 5, rowsFull: 9 }),
      () => attentionQkI8a8Params(8, 12, 4, 0.5, { offset: 4, rowsFull: 9 }),
      () => attentionPvI8a8Params(8, 4, 8, { offset: 4, rowsFull: 9 }),
    ]
  ) {
    assertThrows(build, CodegenError, "をはみ出す");
  }
});

/**
 * GEMM 骨格のタイル定数。**沈黙誤値の代表格**が「WGSL 側の辺 / dispatch 側の辺」のずれなので、
 * 生成物に埋まった辺と幾何を同じテストで突き合わせる（executor は dispatch の辺を
 * `gemmTileM` / `gemmTileN` で幾何から導くので、両者がずれるのは生成側の事故だけ）。
 */
Deno.test("GEMM 骨格 7 op のタイル辺・キー・TS 定数が既定幾何と一致する", () => {
  // conv の m タイルヒューリスティックの基準値（実タイル辺ではない — gemm-geometry.ts の MUST）
  assertEquals(GEMM_TILE, 64);
  // 既定幾何（src/kernels/gemm-geometry.ts）。**キーと生成物の両方に効く**ので、
  // 変えたつもりの無い差分がここで止まる
  assertEquals(defaultGemmGeometry(), { regM: 8, regN: 8, wgX: 16, wgY: 16 });
  for (const [where, key, wgsl] of GEMM_VARIANTS) {
    // 出力タイルの原点（行 / 列）と workgroup。conv1d / conv2d だけ m タイルを 64 へ抑える
    //（出力チャネル数が小さい — tileN とレジスタブロックは全 op 共通）
    const conv = where.startsWith("conv");
    const tileM = conv ? 64 : 128;
    assertEquals(wgsl.includes(`let arow0 = wid.y * ${tileM}u + ar;`), true, where);
    assertEquals(wgsl.includes(`let orow0 = wid.y * ${tileM}u + lid.y * 8u;`), true, where);
    assertEquals(wgsl.includes(`@compute @workgroup_size(16, ${tileM / 8})`), true, where);
    // 1 スレッド 8×8 = レジスタ 64 本（`acc{行}_{列 quad}` の静的展開が全 op 共通）。
    // 共有は sa ${tileM}×16 f32 + sb 8,192 B（f32 の 128 行版で計 16 KB = WebGPU 既定上限）
    assertEquals(wgsl.includes("var acc7_1 = vec4<f32>("), true, where);
    assertEquals(wgsl.includes("acc["), false, `${where}: acc の動的添字`);
    assertEquals(wgsl.includes(`var<workgroup> sa: array<f32, ${tileM * 16}>;`), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
    // MUST: 生成パラメータはキーに載る（タイル辺**と幾何**を変えれば別キーになる）。conv の
    // implicit GEMM だけは判別子が `igemm`（直接カーネルと系統が違うことを名前で表す）で、
    // 幾何は workgroup 形（`wg16x8`）が同じ判別力を持つ。
    const tilePart = conv ? "igemm" : "reg";
    assertEquals(key.includes(`${tilePart}${tileM}x128`), true, where);
    assertEquals(key.includes(conv ? "wg16x8" : "r8x8w16"), true, where);
    // 版番号は op ごと（既存 3 op は 16×16 からの改版で v2・融合 attention は新設なので v1・
    // conv1d の implicit GEMM は直接カーネルが既に v2 を名乗っているので v3）
    const version = where.startsWith("attention")
      ? ":v1:"
      : where.startsWith("conv1d")
      ? ":v3:"
      : ":v2:";
    assertEquals(key.includes(version), true, where);
    // 旧 16×16 タイルの痕跡が残っていない（キーの改版と生成物の総取り替えは対）
    assertEquals(key.includes("tile16"), false, where);
  }
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionPvKey(false), "attention_pv:v1:f32:reg128x128r8x8w16");
  // v4 判別子はキーに載る（形状から導いた 1 ビットがパイプラインを分ける）
  assertEquals(matmulKey(true), "matmul:v2:f32:reg128x128r8x8w16v4");
  assertEquals(matmulKey(false), "matmul:v2:f32:reg128x128r8x8w16");
  assertEquals(bmmKey(true), "bmm:v2:f32:reg128x128r8x8w16v4");
  assertEquals(linearKey("i8", true), "linear:v2:f32:reg128x128r8x8w16v4:wi8");
  assertEquals(linearKey("f16", false), "linear:v2:f32:reg128x128r8x8w16:wf16");
  // conv1d / conv2d は 2 系統（implicit GEMM / 直接カーネル）で、直接側のキーは動かさない
  assertEquals(conv2dIgemmKey("f32", true), "conv2d:v2:f32:igemm64x128v4:wg16x8");
  assertEquals(conv2dIgemmKey("i8", false), "conv2d:v2:f32:igemm64x128:wg16x8:wi8");
  assertEquals(conv2dKey("f32"), "conv2d:v1:f32:direct:wg256");
  assertEquals(conv1dIgemmKey("f32", true), "conv1d:v3:f32:igemm64x128v4:wg16x8");
  assertEquals(conv1dIgemmKey("i8", false), "conv1d:v3:f32:igemm64x128:wg16x8:wi8");
  assertEquals(conv1dKey("f32"), "conv1d:v2:f32:direct:wg256");
});

/**
 * conv2d の **32 行 m タイル変種**（ADR 0024 隣接）。
 *
 * MUST: 変えるのは「どの workgroup がどの出力を担当するか」だけ。1 出力要素の K 縮約順
 * （K タイル 16 昇順）・丸め列・bias-first・0 埋めは 64 行版と共通の骨格が持つので、
 * ここでは**タイル幾何と担当割りだけ**を固定する（数値の一致は実 GPU のビット比較が見る —
 * tests/gpu_conv2d_parity_test.ts）。
 */
Deno.test("conv2d の 32 行 m タイル変種は幾何だけが変わる（n タイル 128・8×8 レジスタは不変）", () => {
  assertEquals(GEMM_MTILE_SMALL, 32);
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = conv2dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL);
      const where = `conv2d igemm m32 ${weight} v4=${v4}`;
      // m タイルは 32 行（A タイルの行原点・store の行原点・bias-first の行原点の 3 箇所）
      assertEquals(wgsl.includes("let arow0 = wid.y * 32u + ar;"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 32u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let bias0 = wid.y * 32u + lid.y * 8u;"), true, where);
      // n タイルは既定幾何と同じ 128（m 側だけの変種であることの検出器）
      assertEquals(
        wgsl.includes(v4 ? "let bc4 = wid.x * 32u + bcq;" : "let bcol = wid.x * 128u + bcq * 4u;"),
        true,
        where,
      );
      // workgroup は 16×4 = 64 スレッド。**1 スレッド 8×8 は不変** = 内積の演算密度が
      // 落ちない（1 スレッドの出力を削って workgroup を保つ形は密度が下がる）
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 4)"), true, where);
      assertEquals(wgsl.includes("var acc7_0 = vec4<f32>(bias[bias0 + 7u]);"), true, where);
      // 共有 A は 32 行ぶんへ縮む（sb は n タイルが同じなので 512 のまま）
      assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
      // 64 スレッドで 512 quad の B タイルを埋めるので 8 スロット。担当の k 行は 2 ずつずれる
      assertEquals(wgsl.includes("let bk0 = tid / 32u;"), true, where);
      assertEquals(wgsl.includes("let bk7 = bk0 + 14u;"), true, where);
      assertEquals(wgsl.includes("let bk8 ="), false, `${where}: スロットは 8 本ちょうど`);
      // 内積ループの正本は骨格 1 箇所（64 行版と 1 文字も違わない）
      assertEquals(
        wgsl.includes("acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;"),
        true,
        where,
      );
    }
  }
  // 64 行版は 128 スレッドで B タイルを 4 スロット（32 行版の 8 スロットが漏れ出していないこと）
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk3 = bk0 + 12u;"), true);
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk4 ="), false);
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk0 = tid / 32u;"), true);
  // キーは別系統（タイル形は生成パラメータなのでキーに載る）
  assertEquals(conv2dIgemmKey("f32", true, 32), "conv2d:v2:f32:igemm32x128v4:wg16x4");
  assertEquals(conv2dIgemmKey("i8", false, 32), "conv2d:v2:f32:igemm32x128:wg16x4:wi8");
});

/**
 * m タイルの選択述語。**境界は `M % 64 <= 32`**。
 *
 * MUST: `< 32` にすると census の Cout=96（`96 % 64 == 32` — 1024 系列 MAC の 32.4%）が
 * 64 行のまま残り、本変種の動機そのものが消える。`M % 64 == 0` は 64 行が無駄ゼロなので
 * 32 行に割ると B タイル（x の暗黙 gather）を 2 倍読むだけ損。
 */
Deno.test("conv2dIgemmMTile は M%64 が 1..32 のときだけ 32 行を選ぶ", () => {
  // census（VAE decoder）の実形状
  assertEquals(conv2dIgemmMTile(96), 32, "Cout=96（96%64==32 — 本変種の本命）");
  assertEquals(conv2dIgemmMTile(3), 32, "Cout=3（無駄 21.33× → 10.67×）");
  assertEquals(conv2dIgemmMTile(16), 32, "Cout=16（無駄 4× → 2×）");
  assertEquals(conv2dIgemmMTile(192), 64, "Cout=192（64 行で無駄ゼロ）");
  assertEquals(conv2dIgemmMTile(384), 64);
  assertEquals(conv2dIgemmMTile(1152), 64);
  // 境界
  assertEquals(conv2dIgemmMTile(64), 64, "M%64==0");
  assertEquals(conv2dIgemmMTile(65), 32, "M%64==1");
  assertEquals(conv2dIgemmMTile(32), 32);
  assertEquals(conv2dIgemmMTile(33), 64, "M%64==33（無駄は同じで B 読みだけ増える帯）");
  assertEquals(conv2dIgemmMTile(1), 32);
  // 無駄の単調性（選んだタイルが常に「無駄が真に小さいか同じ」であること）
  for (let m = 1; m <= 256; m += 1) {
    const chosen = conv2dIgemmMTile(m);
    const waste = (tile: number) => Math.ceil(m / tile) * tile / m;
    assertEquals(waste(chosen) <= waste(chosen === 64 ? 32 : 64), true, `M=${m}`);
  }
});

/**
 * conv2d の implicit GEMM（ADR 0024）が骨格の断片共有で満たすべき 4 つの MUST を、
 * **生成物の側から**固定する。どれも「実測形では絶対に露見しない」種類の誤りで、
 * 数値テスト（parity / CPU 参照）と対で持つことに意味がある。
 */
Deno.test("conv2d の implicit GEMM は bias-first / 0 埋め / 行 scale を生成物に持つ", () => {
  for (const v4 of [false, true]) {
    const wgsl = conv2dIgemmWgsl("f32", v4);
    const where = `conv2d igemm v4=${v4}`;
    // MUST ①: bias は acc の初期値（store 側で足す形にすると (Σ)+bias で丸めが変わる）
    assertEquals(wgsl.includes("let bias0 = wid.y * 64u + lid.y * 8u;"), true, where);
    assertEquals(wgsl.includes("var acc1_0 = vec4<f32>(bias[bias0 + 1u]);"), true, where);
    assertEquals(wgsl.includes("+ biasv"), false, `${where}: store 側で bias を足している`);
    assertEquals(wgsl.includes("+ bias[ocol"), false, `${where}: store 側で bias を足している`);
    // MUST ③: 範囲外の x は 0（クランプ添字で読まない）
    assertEquals(
      wgsl.includes(
        "  if (iy < 0 || u32(iy) >= dims.height_in || ix < 0 || u32(ix) >= dims.width_in) {\n    return 0.0;\n  }",
      ),
      true,
      where,
    );
    assertEquals(wgsl.includes("clamp("), false, `${where}: 添字のクランプが混ざっている`);
    // 平坦 k → (ic, kh, kw)（Kh と Kw を取り違えると正方カーネルでは数値が一致する）
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
    // バッチは z 軸（N 側へ畳むと出力の軸が [Cout][B·H·W] に入れ替わる）
    assertEquals(wgsl.includes("let xbase = wid.z * dims.channels_in;"), true, where);
    assertEquals(wgsl.includes("let cbase = wid.z * dims.m * "), true, where);
    // 死んだ uniform 欄を持たない（height_out は n / width_out で導ける）
    assertEquals(wgsl.includes("height_out"), false, `${where}: 読まれない Dims 欄がある`);
  }
  // MUST ④: i8 の scale は**行**（= 出力チャネル）。linear の `wcol`（列）流用は沈黙誤値
  // MUST: 充填スロットごとに別名で束ねる（1 スレッドが複数チャネルを埋めるので、
  // スロット 0 の scale を 2 本目にも使うと片方だけが静かに別チャネルの scale で dequant される）
  const i8Wgsl = conv2dIgemmWgsl("i8", true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[arow0];"), true);
  assertEquals(i8Wgsl.includes("let wscale_v1 = wscale[arow1];"), true);
  assertEquals(i8Wgsl.includes("wscale[wcol"), false, "linear の列 scale を持ってきている");
  // 束縛番号は直接カーネルと同じ（executor は 1 本の定数で両方を束ねる）
  assertEquals(
    i8Wgsl.includes(`@group(0) @binding(${CONV2D_SCALE_BINDING}) var<storage, read> wscale:`),
    true,
  );
  // A タイルは重み（weight-storage 経由）・B タイルは x の暗黙 gather
  assertEquals(i8Wgsl.includes("av0 = dequant4(arow_base0 + ak0, wscale_v);"), true);
  assertEquals(i8Wgsl.includes("av1 = dequant4(arow_base1 + ak0, wscale_v1);"), true);
  assertEquals(conv2dIgemmWgsl("f16", false).includes("av0.x = dequant(abase);"), true);
  assertEquals(conv2dIgemmWgsl("f32", false).includes("av0.x = w[abase];"), true);
});

/**
 * v4 判定（`kFlat % 4 == 0 && Wout % 4 == 0 && strideW == 1` — ADR 0024 の MUST ②）。
 *
 * MUST: 列 quad の条件は **`Wout % 4`**。`N % 4`（= B·Hout·Wout）で書くと、Hout=Wout=2 型で
 * quad が出力行をまたいだまま v4 が選ばれる。census の実形状は Wout が全て 4 の倍数なので
 * **実モデルでは絶対に露見しない**。
 */
Deno.test("conv2dUsesVec4 は kFlat / Wout / strideW の 3 条件を全て見る", () => {
  assertEquals(conv2dUsesVec4(1728, 512, 1), true, "census の実形状");
  assertEquals(conv2dUsesVec4(4, 2, 1), false, "Wout=2（N%4==0 でも quad が行をまたぐ）");
  assertEquals(conv2dUsesVec4(18, 8, 1), false, "kFlat=18（Cin=2 の 3×3）");
  assertEquals(conv2dUsesVec4(24, 8, 3), false, "stride_w=3（4 列の x が連続しない）");
  assertEquals(conv2dUsesVec4(4, 4, 1), true);
});

/**
 * conv1d の implicit GEMM が骨格の断片共有で満たすべき MUST を**生成物の側から**固定する
 * （2D 版と同じ 4 点 + 1D 固有の「2 次元の概念を持ち込まない」）。数値の一致は実 GPU の
 * ビット比較が見る（tests/gpu_conv1d_parity_test.ts）。
 */
Deno.test("conv1d の implicit GEMM は bias-first / 0 埋め / 行 scale / 1 軸 Dims を生成物に持つ", () => {
  for (const v4 of [false, true]) {
    const wgsl = conv1dIgemmWgsl("f32", v4);
    const where = `conv1d igemm v4=${v4}`;
    // MUST ①: bias は acc の初期値（store 側で足す形にすると (Σ)+bias で丸めが変わる）
    assertEquals(wgsl.includes("let bias0 = wid.y * 64u + lid.y * 8u;"), true, where);
    assertEquals(wgsl.includes("var acc1_0 = vec4<f32>(bias[bias0 + 1u]);"), true, where);
    assertEquals(wgsl.includes("+ biasv"), false, `${where}: store 側で bias を足している`);
    assertEquals(wgsl.includes("+ bias[ocol"), false, `${where}: store 側で bias を足している`);
    // MUST ③: 範囲外の x は 0（クランプ添字で読まない）
    assertEquals(
      wgsl.includes(
        "  let ix = i32(n * dims.stride) + kt;\n  if (ix < 0 || u32(ix) >= dims.length_in) {\n    return 0.0;\n  }",
      ),
      true,
      where,
    );
    assertEquals(wgsl.includes("clamp("), false, `${where}: 添字のクランプが混ざっている`);
    // 平坦 k → (ic, k)（ic を最内に取ると同じ tap 集合のまま加算順序だけが変わる）
    assertEquals(wgsl.includes("let ic = brow0 / dims.kernel;"), true, where);
    assertEquals(
      wgsl.includes(
        "let kt = i32((brow0 % dims.kernel) * dims.dilation) - i32(dims.padding);",
      ),
      true,
      where,
    );
    // バッチは z 軸（N 側へ畳むと出力の軸が [Cout][B·Lout] に入れ替わる）
    assertEquals(wgsl.includes("let xbase = wid.z * dims.channels_in;"), true, where);
    assertEquals(wgsl.includes("let cbase = wid.z * dims.m * "), true, where);
    // MUST: 死んだ Dims 欄を持たない（length_out は dims.n そのもの）。2 次元の概念
    // （H/W・Kh/Kw）も 1 語たりとも漏れていない — 2D 版からの写し漏れの検出器
    for (const dead of ["length_out", "width", "height", "kernel_h", "kernel_w", "stride_h"]) {
      assertEquals(wgsl.includes(dead), false, `${where}: ${dead} が漏れている`);
    }
    // A タイル・内積ループ・store は 2D 版と**同じ断片**（1 文字でも割れたら共有が壊れている）
    const conv2d = conv2dIgemmWgsl("f32", v4);
    for (
      const shared of [
        "  let ar = tid / 4u;",
        "  let arow_base0 = arow0 * dims.k;",
        "  let bias0 = wid.y * 64u + lid.y * 8u;",
        "      acc0_0 = acc0_0 + sa[(lid.y * 8u + 0u) * 16u + kk] * bv0;",
        "  let orow0 = wid.y * 64u + lid.y * 8u;",
      ]
    ) {
      assertEquals(wgsl.includes(shared), true, `${where}: ${shared}`);
      assertEquals(conv2d.includes(shared), true, `conv2d igemm v4=${v4}: ${shared}`);
    }
  }
  // MUST ④: i8 の scale は**行**（= 出力チャネル）。linear の `wcol`（列）流用は沈黙誤値で、
  // m タイルが 2 枚以上ある形のテスト（gpu_conv1d_parity_test.ts の Cout=96）だけが検出器
  const i8Wgsl = conv1dIgemmWgsl("i8", true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[arow0];"), true);
  assertEquals(i8Wgsl.includes("let wscale_v1 = wscale[arow1];"), true);
  assertEquals(i8Wgsl.includes("wscale[wcol"), false, "linear の列 scale を持ってきている");
  // 束縛番号は直接カーネルと同じ（executor は 1 本の定数で両方を束ねる）
  assertEquals(
    i8Wgsl.includes(`@group(0) @binding(${CONV1D_SCALE_BINDING}) var<storage, read> wscale:`),
    true,
  );
  // A タイルは重み（weight-storage 経由）・B タイルは x の暗黙 gather
  assertEquals(i8Wgsl.includes("av0 = dequant4(arow_base0 + ak0, wscale_v);"), true);
  assertEquals(conv1dIgemmWgsl("f16", false).includes("av0.x = dequant(abase);"), true);
  assertEquals(conv1dIgemmWgsl("f32", false).includes("av0.x = w[abase];"), true);
});

/**
 * conv1d の **32 行 m タイル変種**（述語は conv2d と共有 — {@link conv2dIgemmMTile}）。
 *
 * MUST: 変えるのは「どの workgroup がどの出力を担当するか」だけ。1 出力要素の K 縮約順・
 * 丸め列・bias-first・0 埋めは 64 行版と共通の骨格が持つ。
 */
Deno.test("conv1d の 32 行 m タイル変種は幾何だけが変わる（n タイル 128・8×8 レジスタは不変）", () => {
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = conv1dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL);
      const where = `conv1d igemm m32 ${weight} v4=${v4}`;
      assertEquals(wgsl.includes("let arow0 = wid.y * 32u + ar;"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 32u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let bias0 = wid.y * 32u + lid.y * 8u;"), true, where);
      // n タイルは既定幾何と同じ 128（m 側だけの変種であることの検出器）
      assertEquals(
        wgsl.includes(v4 ? "let bc4 = wid.x * 32u + bcq;" : "let bcol = wid.x * 128u + bcq * 4u;"),
        true,
        where,
      );
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 4)"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 512>;"), true, where);
      // 64 スレッドで 512 quad の B タイルを埋めるので 8 スロット
      assertEquals(wgsl.includes("let bk7 = bk0 + 14u;"), true, where);
      assertEquals(wgsl.includes("let bk8 ="), false, `${where}: スロットは 8 本ちょうど`);
    }
  }
  // 64 行版へ 32 行版の担当割りが漏れていない
  assertEquals(conv1dIgemmWgsl("f32", true).includes("let bk3 = bk0 + 12u;"), true);
  assertEquals(conv1dIgemmWgsl("f32", true).includes("let bk4 ="), false);
  // キーは別系統（タイル形は生成パラメータなのでキーに載る）
  assertEquals(conv1dIgemmKey("f32", true, 32), "conv1d:v3:f32:igemm32x128v4:wg16x4");
  assertEquals(conv1dIgemmKey("i8", false, 32), "conv1d:v3:f32:igemm32x128:wg16x4:wi8");
});

/**
 * v4 判定（`kFlat % 4 == 0 && Lout % 4 == 0 && stride == 1`）。
 *
 * MUST: 3 条件を全て見る。1D では出力平面が 1 行なので 2D の `Wout % 4` と `N % 4` の
 * 区別は消えるが、**`Cin % 4` で代用しない**ことと **stride を見る**ことは 2D と同じ理由で要る。
 */
Deno.test("conv1dUsesVec4 は kFlat / Lout / stride の 3 条件を全て見る", () => {
  assertEquals(conv1dUsesVec4(1024, 512, 1), true, "dacvae 系の実形状");
  assertEquals(conv1dUsesVec4(7, 8, 1), false, "kFlat=7（Cin=1 の K=7 — Cin%4 では捕まらない）");
  assertEquals(conv1dUsesVec4(8, 170, 1), false, "Lout=170（実測形のスカラ変種）");
  assertEquals(conv1dUsesVec4(8, 8, 2), false, "stride=2（4 列の x が連続しない）");
  assertEquals(conv1dUsesVec4(4, 4, 1), true);
});

/**
 * v4 判定（`k % 4 == 0 && n % 4 == 0`）。
 *
 * MUST: **両方**見る。片方だけの判定は「k のみ 4 の倍数 / n のみ 4 の倍数」の形で
 * vec4 束縛と実バイト数が食い違い、例外なしの誤値になる。
 */
Deno.test("gemmUsesVec4 は k と n の両方が 4 の倍数のときだけ真", () => {
  assertEquals(gemmUsesVec4(20, 64), true);
  assertEquals(gemmUsesVec4(4, 4), true);
  assertEquals(gemmUsesVec4(20, 19), false, "k のみ 4 の倍数");
  assertEquals(gemmUsesVec4(19, 20), false, "n のみ 4 の倍数");
  assertEquals(gemmUsesVec4(19, 19), false);
  // 0 は 4 で割り切れる（空 shape でも判定が例外にならない）
  assertEquals(gemmUsesVec4(0, 0), true);
});

/**
 * w8a8（活性 per-token i8 × 重み i8 の整数内積）の生成物。
 *
 * この経路の数値契約は **GPU と TS 参照が atol=0** なので、丸めの位置を決める 3 点
 * （scale を除算で作らない / dequant の乗算順序 / 積和を `fma` で単一丸めにする）は
 * 生成物の側で固定する。どれも実測形の数値テストでは 1 ULP 差にしかならず、
 * atol=0 の突合と対で持って初めて検出器になる。
 */
Deno.test("i8a8 linear と quantize_rows は丸めの位置を決める 3 点を生成物に持つ", () => {
  // ---- quantize_rows -------------------------------------------------------
  // MUST: scale は 127 の**除算ではなく 1/127 との乗算**（WGSL の除算は 2.5 ULP まで許され、
  // 実測でも 1 ULP 割れる。定数畳み込みに頼るとドライバ次第で参照と食い違う）
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes(
      "let s = select(max(amax * 0.007874015748031496, 1.1754943508222875e-38), amax, is_nan_bits(amax));",
    ),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("/ 127"), false, "scale を除算で作っている");
  // MUST: ±127 に閉じる（−128 を使わない = 絶対値最大の要素が厳密に復元される）
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes(
      "let r = clamp(round(v), vec4<f32>(-127.0), vec4<f32>(127.0));",
    ),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("pack4xI8Clamp"), false, "飽和 pack に頼っている");
  assertEquals(QUANTIZE_ROWS_WGSL.includes("xq[qbase + q] = pack4xI8(vec4<i32>(r));"), true);
  // MUST: absmax の NaN 伝播はビット列判定（素の max は NaN を飲む — ADR 0020）
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes("return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u;"),
    true,
  );
  assertEquals(QUANTIZE_ROWS_WGSL.includes("acc = nan_max(acc, abs(x[base + i]));"), true);
  assertEquals(
    QUANTIZE_ROWS_WGSL.includes("scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);"),
    true,
  );
  // 行方向は grid-stride（縮退ハーネスの対象 — tests/gpu_gridstride_test.ts）
  assertEquals(QUANTIZE_ROWS_WGSL.includes("row = row + nwg.x;"), true);
  assertEquals(
    QUANTIZE_ROWS_KEY,
    `quantize_rows:v1:f32>i8:pertoken:wg${QUANTIZE_ROWS_WORKGROUP_SIZE}`,
  );
  assertEquals(QUANTIZE_ROWS_WORKGROUP_SIZE, 256);
  assertEquals([...quantizeRowsParams(30, 8)], [30, 8, 0, 0]);
  assertEquals(quantizeRowsParams(30, 8).byteLength, 16, "uniform struct の 16 バイト整列");
  assertThrows(() => quantizeRowsParams(30, 6), CodegenError, "4 の倍数");
  assertThrows(() => quantizeRowsParams(-1, 8), CodegenError);
  assertThrows(() => quantizeRowsParams(30, 0), CodegenError);

  // ---- i8a8 GEMM -----------------------------------------------------------
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = linearI8a8Wgsl(v4, dp4a);
      const where = `i8a8 v4=${v4} dp4a=${dp4a}`;
      // 整数内積の変種は **idot 1 箇所**だけが違う（数値同一の主張の構造的な担保）
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: dot4I8Packed の出現数`,
      );
      assertEquals(
        wgsl.split("return dot(unpack4xI8(a), unpack4xI8(b));").length - 1,
        dp4a ? 0 : 1,
        `${where}: エミュ内積の出現数`,
      );
      // 内積は 8 行 × 2 列 quad へ静的展開（8×8 出力 = 64 内積）。**組ごとにちょうど 1 回**で、
      // 行番号は codegen 時に確定する — accumulator に動的添字が残っていないことが条件。
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [[0, [0, 1, 2, 3]], [1, [4, 5, 6, 7]]] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${
            lanes.map((lane) => `idot(a${i}, b${lane})`).join(", ")
          });`;
          assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} quad ${quad} の内積`);
        }
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる
      // （Metal で顕著）。展開の目的そのものなので、生成物に 1 つも残っていないことを見る。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // 共有タイルは [pack][row] / [pack][col]（バンク衝突 2-way — プロトタイプからの組み替え）。
      // 1 スレッドが A を 4 本・B を 2 本埋める（tileM 128 / tileN 64 を 128 スレッドで覆う）
      assertEquals(wgsl.includes("sa[sa_at] = av0;"), true, where);
      assertEquals(wgsl.includes("sa[sa_at + 96u] = av3;"), true, where);
      assertEquals(wgsl.includes("sb[sb_at] = wv0;"), true, where);
      assertEquals(wgsl.includes("sb[sb_at + 32u] = wv1;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 256>;"), true, where);
      // K 端数は 0 埋め（dot4I8Packed(0, x) == 0 なので厳密）。**充填スロットごとに**門が要る
      assertEquals(wgsl.includes("if (arow0 < dims.m && apack < k4) {"), true, where);
      assertEquals(wgsl.includes("if (arow3 < dims.m && apack < k4) {"), true, where);
      assertEquals(wgsl.includes("if (wcol1 < dims.n && wpack < k4) {"), true, where);
      // 束縛: 5 = 重み scale（linear と共通）/ 6 = 活性 scale（i8a8 だけ）
      assertEquals(wgsl.includes("@group(0) @binding(5) var<storage, read> wscale:"), true, where);
      assertEquals(
        wgsl.includes(
          `@group(0) @binding(${LINEAR_ACT_SCALE_BINDING}) var<storage, read> xscale:`,
        ),
        true,
        where,
      );
      // MUST: dequant は `xs·wscale` を先に畳み、積和は fma（単一丸め）。逐次形
      // `f32(acc)·xs·wscale` にすると丸めの位置が動いて atol=0 が崩れる。
      assertEquals(
        wgsl.includes(
          v4
            ? "out[orow0 * n4 + ocq0] = fma(vec4<f32>(acc0_0), xscale[orow0] * ws, biasv);"
            : "out[obase + ocol] = fma(f32(acc0_0.x), xs * wscale[ocol], bias[ocol]);",
        ),
        true,
        `${where}: dequant の乗算順序`,
      );
      // 素の `a * b + c` が残っていない（融合するかどうかがドライバ依存になる形）
      assertEquals(wgsl.includes(") + biasv;"), false, `${where}: 素の積和が残っている`);
      // タイル幾何は i8a8 独自（f32 骨格の 64×64 / 16×16 とは別 — 整数縮約が順序非依存だから
      // 実測で選べる自由度になっている）。既定は M128N64 r8×8 wg8×16 K16。
      assertEquals(wgsl.includes("@compute @workgroup_size(8, 16)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 128u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
    }
  }
  // キーは診断でどの変種が走ったか分かる形（ADR 0021）。幾何が載るので世代を v4 へ上げた
  assertEquals(linearI8a8Key(true, true), "linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a");
  assertEquals(linearI8a8Key(false, false), "linear:v4:i8a8:tile128x64r8x8w8x16k16:dp4aEmu");
  // MUST: v4 判定は **n だけ**（k % 4 == 0 は適格判定が担うので条件に混ぜない）
  assertEquals(linearI8a8UsesVec4(68), true);
  assertEquals(linearI8a8UsesVec4(19), false);
  // params は f32 骨格と同じ 3 語 + 16 バイト整列。k の門は params でも見る
  assertEquals([...linearI8a8Params(5, 7, 8)], [5, 7, 8, 0]);
  assertEquals(linearI8a8Params(5, 7, 8).byteLength, 16);
  assertThrows(() => linearI8a8Params(5, 7, 7), CodegenError, "4 の倍数");
  assertThrows(
    () => linearI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4),
    CodegenError,
    "i32 縮約の門",
  );
  // 拡張の判定は WGSL 言語機能の列挙（device feature ではない）
  assertEquals(dp4aAvailable(new Set([DP4A_WGSL_FEATURE])), true);
  assertEquals(dp4aAvailable(new Set(["shader-f16"])), false);
});

/**
 * w4a8（i4 常駐 × per-token i8 活性 — perf-ledger Q-8）。数値契約の担い手は
 * 「**group 境界ちょうどの flush**」「i8 変種と非対称な xs の掛け位置」「nibble の並び」の
 * 3 点しかないので、値ではなく生成物の構造の側で固定する（実 GPU の atol=0 突合は
 * tests/gpu_i8a8_test.ts の w4a8 節）。
 */
Deno.test("w4a8 linear は group 境界でだけ f32 へ flush し、xs を最後の fma まで持ち越す", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = linearI8a8Wgsl(v4, dp4a, undefined, "i4", 32);
      const where = `w4a8 v4=${v4} dp4a=${dp4a}`;
      // MUST: 動的添字はレジスタから落ちる（Metal で顕著）。i32 / f32 の**両方**の
      // accumulator について、展開の目的そのものを見る。
      assertEquals(wgsl.includes("acc["), false, `${where}: i32 accumulator の動的添字`);
      assertEquals(wgsl.includes("accf["), false, `${where}: f32 accumulator の動的添字`);
      // 内側の K タイルループは **group 長 / tileK = 2 の定数**（実行時値だと group 境界と
      // flush の位置がずれうる）。外側だけが uniform の実行時値。
      assertEquals(wgsl.includes("for (var gt = 0u; gt < 2u; gt = gt + 1u) {"), true, where);
      assertEquals(wgsl.includes("for (var gi = 0u; gi < groups; gi = gi + 1u) {"), true, where);
      assertEquals(wgsl.includes("let groups = dims.k >> 5u;"), true, `${where}: g=32 の shift`);
      // MUST: flush は 8 行 × 2 列 quad の **16 組ちょうど 1 回ずつ**（group ごとに 1 回で、
      // タイルごとでも出力ごとでもない — 丸めが k/g + 1 回であることの構造的な担保）。
      for (let row = 0; row < 8; row += 1) {
        for (const quad of [0, 1]) {
          const flush =
            `    accf${row}_${quad} = fma(vec4<f32>(acc${row}_${quad}), ws${quad}, accf${row}_${quad});`;
          assertEquals(wgsl.split(flush).length, 2, `${where}: 行 ${row} quad ${quad} の flush`);
        }
      }
      // group scale は [n, k/g] の平坦を列ごとの行頭 + group 番号で引く（列 scale の per-channel
      // 解釈へ退行すると添字が gi に依らなくなる）
      assertEquals(wgsl.includes("let wsb1 = (ocol + 1u) * groups;"), true, where);
      assertEquals(
        wgsl.includes(
          "let ws0 = vec4<f32>(wscale[wsb0 + gi], wscale[wsb1 + gi], wscale[wsb2 + gi], wscale[wsb3 + gi]);",
        ),
        true,
        where,
      );
      // MUST: nibble 抽出は 16bit のシフト（動的 vec4 添字ゼロ）で、格納値は u = q + 8。
      // 並びは dequant4 と同一（要素 2i = 下位 / 2i+1 = 上位）。
      assertEquals(
        wgsl.includes(
          "  let half = (w[i >> 3u] >> (((i >> 2u) & 1u) * 16u)) & 0xFFFFu;\n" +
            "  let u = vec4<u32>(half, half >> 4u, half >> 8u, half >> 12u) & vec4<u32>(0xFu);\n" +
            "  return pack4xI8(vec4<i32>(u) - vec4<i32>(8));",
        ),
        true,
        `${where}: nibble 抽出`,
      );
      // B 側は平坦要素の添字（i8 の語添字 k4 と取り違えると 1/4 の位置を読む）
      assertEquals(wgsl.includes("let wrow_base0 = wcol0 * dims.k;"), true, where);
      assertEquals(wgsl.includes("let welem = t * 16u + wp * 4u;"), true, where);
      assertEquals(wgsl.includes("wv0 = i4lanes(wrow_base0 + welem);"), true, where);
      // MUST: xs は最後の fma へ（i8 変種の `xs * wscale` 畳みは w4a8 では成立しない）
      assertEquals(
        wgsl.includes(
          v4
            ? "out[orow0 * n4 + ocq0] = fma(accf0_0, vec4<f32>(xscale[orow0]), biasv);"
            : "out[obase + ocol] = fma(accf0_0.x, xs, bias[ocol]);",
        ),
        true,
        `${where}: xs の掛け位置`,
      );
      assertEquals(wgsl.includes("xs * wscale"), false, `${where}: i8 変種の畳み形が残っている`);
      // 整数内積の変種は i8 と同じ 1 箇所だけ（共有断片 idot を使っていることの担保）
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: dot4I8Packed の出現数`,
      );
    }
  }
  // g は WGSL に焼かれるので生成物が group ごとに違う（キーの g 部と対 — 同じキーで違う
  // 生成物、あるいは違うキーで同じ生成物になったらどちらも沈黙誤値の入口）
  assertNotEquals(
    linearI8a8Wgsl(true, true, undefined, "i4", 32),
    linearI8a8Wgsl(true, true, undefined, "i4", 64),
    "group 長が生成物に出ていない",
  );
  assertEquals(
    linearI8a8Key(true, true, undefined, "i4", 32),
    "linear:v4:i8a8:tile128x64r8x8w8x16k16v4:dp4a:wi4g32",
  );
  assertEquals(
    linearI8a8Key(false, false, undefined, "i4", 64),
    "linear:v4:i8a8:tile128x64r8x8w8x16k16:dp4aEmu:wi4g64",
  );
  // MUST: i8 のキーは既定引数で従来のまま（既存キーがバイト不変であることの直接の門）
  assertEquals(linearI8a8Key(true, true, undefined, "i8"), linearI8a8Key(true, true));
  // i4 と group 長は対（片方だけは結線バグ — weight-storage.ts の i4GroupShift）
  assertThrows(
    () => linearI8a8Wgsl(true, true, undefined, "i4"),
    CodegenError,
    "groupSize は重み i4 格納と対で渡す",
  );
  assertThrows(
    () => linearI8a8Wgsl(true, true, undefined, "i8", 32),
    CodegenError,
    "groupSize は重み i4 格納と対で渡す",
  );
  // 整数内積に載らない格納形は fail loudly（f32 / f16 を渡す経路はそもそも無い）
  assertThrows(() => linearI8a8Wgsl(true, true, undefined, "f16"), CodegenError, "i8 / i4 のみ");
  // MUST: 1 枚の K タイル（16）が group 境界を跨ぐ形は生成時に落とす — 通すと 2 group の
  // 重みが 1 つの i32 に混ざり、flush で片方の scale だけが掛かる沈黙誤値になる
  assertThrows(
    () => linearI8a8Wgsl(true, true, { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 32 }, "i4", 16),
    CodegenError,
    "K タイル 32 の倍数でない",
  );
  // params の門は k ではなく **group 長**（i8 の k 門は w4a8 に適用しない）
  assertEquals([...linearI8a8Params(5, 8, 64, 32)], [5, 8, 64, 0]);
  assertEquals(
    [...linearI8a8Params(5, 8, LINEAR_I8A8_MAX_K + 32, 32)].length,
    4,
    "w4a8 では i8 の k 門を適用しない",
  );
  assertThrows(
    () => linearI8a8Params(5, 8, 48, 32),
    CodegenError,
    "group_size 32 で割り切れない",
  );
  assertThrows(
    () => linearI8a8Params(5, 8, LINEAR_W4A8_MAX_GROUP * 2, LINEAR_W4A8_MAX_GROUP * 2),
    CodegenError,
    "i32 縮約の門",
  );
});

/**
 * 融合 attention ①QK の i8a8 変種（設計 §2.1 / §4.1・波 1）。
 *
 * 数値契約は linear の w8a8 と同じ **GPU vs TS 参照 atol=0** なので、丸めの位置を決める
 * 3 点（半スケールを dequant 側へ / `qs'·ks'` を先に畳む / 整数内積の実体は 1 本）と、
 * **バッチ base の単位**（scale は行数単位・ペイロードは pack 単位）を生成物の側で固定する。
 */
Deno.test("i8a8 attention_qk は半スケールを dequant 側に持ち、バッチ base を単位ごとに分ける", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = attentionQkI8a8Wgsl(v4, dp4a);
      const where = `attention_qk i8a8 v4=${v4} dp4a=${dp4a}`;
      // 整数内積の実体は linear と共有の 1 本（`idot` だけが変種で違う）
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: dot4I8Packed の出現数`,
      );
      assertEquals(
        wgsl.split("return dot(unpack4xI8(a), unpack4xI8(b));").length - 1,
        dp4a ? 0 : 1,
        `${where}: エミュ内積の出現数`,
      );
      // 内積は 8 行 × 2 列 quad へ静的展開（linear の i8a8 と同じ展開）。**組ごとにちょうど
      // 1 回**で、行番号は codegen 時に確定する — accumulator に動的添字が残っていないことが条件。
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [[0, [0, 1, 2, 3]], [1, [4, 5, 6, 7]]] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${
            lanes.map((lane) => `idot(a${i}, b${lane})`).join(", ")
          });`;
          assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} quad ${quad} の内積`);
        }
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる。
      // 展開の目的そのものなので、生成物に 1 つも残っていないことを見る。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // MUST: 半スケールは**量子化の前ではなく dequant 側**で q / k の両方へ。
      // 片側化・量子化前への移動はどちらも例外なしの誤値になる。
      assertEquals(
        wgsl.includes(
          v4
            ? "s[sbase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * ((qscale[qsbase + orow0] * dims.scale) * ks);"
            : "s[obase + ocol] = f32(acc0_0.x) * (qs * (kscale[ksbase + ocol] * dims.scale));",
        ),
        true,
        `${where}: dequant の乗算順序と半スケールの位置`,
      );
      // bias が無いので fma は使わない（linear の `fma(…, bias)` と対）
      assertEquals(wgsl.includes("fma("), false, `${where}: fma が混ざっている`);
      // MUST: バッチ base は「ペイロード = pack 単位 / scale = 行数単位 / S = 要素（quad）単位」。
      // 単位を取り違えると B·H ≥ 2 で隣の head を読み書きする（B=H=1 のテストには出ない）
      assertEquals(wgsl.includes("let qbase = wid.z * dims.m * k4;"), true, where);
      assertEquals(wgsl.includes("let kbase = wid.z * dims.n * k4;"), true, where);
      assertEquals(wgsl.includes("let qsbase = wid.z * dims.m;"), true, where);
      assertEquals(wgsl.includes("let ksbase = wid.z * dims.n;"), true, where);
      assertEquals(
        wgsl.includes(`let sbase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};`),
        true,
        where,
      );
      // k は [N,D] のまま読む（連続方向が D = パック方向なので転置が要らない）
      assertEquals(wgsl.includes("let krow_base0 = kbase + wcol0 * k4;"), true, where);
      assertEquals(wgsl.includes("let krow_base1 = kbase + wcol1 * k4;"), true, where);
      // 束縛: 4 = q の行 scale / 5 = k の列 scale
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
      // タイル幾何は linear の i8a8 と同じ既定（M128N64 r8×8 wg8×16 K16）。**③PV とは違う**
      assertEquals(wgsl.includes("@compute @workgroup_size(8, 16)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 128u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 256>;"), true, where);
    }
  }
  // キーは linear:v4:i8a8 の前例に倣う（世代 ++ / dtype 欄 i8a8 / 幾何 / 内積変種は末尾）
  assertEquals(
    attentionQkI8a8Key(true, true),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16v4:dp4a",
  );
  assertEquals(
    attentionQkI8a8Key(false, false),
    "attention_qk:v3:i8a8:tile128x64r8x8w8x16k16:dp4aEmu",
  );
  // MUST: v4 判定は **n だけ**（D % 4 == 0 は適格判定が担う）
  assertEquals(attentionQkI8a8UsesVec4(68), true);
  assertEquals(attentionQkI8a8UsesVec4(19), false);
  // params は f32 変種と同じ 3 語 + 4 語目が半スケールの f32 ビット列
  const params = attentionQkI8a8Params(5, 7, 8, 0.5);
  assertEquals([...params.slice(0, 3)], [5, 7, 8]);
  assertEquals(new Float32Array(params.buffer)[3], 0.5);
  assertEquals(params.byteLength, 16);
  assertThrows(() => attentionQkI8a8Params(5, 7, 7, 0.5), CodegenError, "4 の倍数");
  assertThrows(() => attentionQkI8a8Params(5, 7, 8, Number.NaN), CodegenError, "有限");
  assertThrows(
    () => attentionQkI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4, 0.5),
    CodegenError,
    "i32 縮約の門",
  );
});

/**
 * ③PV の i8a8 変種（設計 §2.2 / §2.3）。数値契約の担い手は 4 点しかないので、生成物の側で
 * 全部固定する — ① P̃ を成分ごとに作る `round(127·exp(S−m))`（除算ゼロ・amax を取らない）
 * ② 範囲外は変換を掛けずに 0 ③ dequant は `prow·vs` を先に畳む・`1/127` は乗算
 * ④ 行統計は `rbase + orow` / `rbase + arow`（バッチ base 込み）。
 */
Deno.test("i8a8 attention_pv は P̃ を非実体化のまま作り、列 scale と行 inv を別の軸で引く", () => {
  for (const v4 of [false, true]) {
    for (const dp4a of [false, true]) {
      const wgsl = attentionPvI8a8Wgsl(v4, dp4a);
      const where = `attention_pv i8a8 v4=${v4} dp4a=${dp4a}`;
      // 整数内積の実体は linear / ①QK と共有の 1 本（書き写していないことの検出器）
      assertEquals(
        wgsl.split("return dot4I8Packed(a, b);").length - 1,
        dp4a ? 1 : 0,
        `${where}: dot4I8Packed の出現数`,
      );
      // 内積は 8 行 × 2 列 quad へ静的展開（①QK / linear の i8a8 と同じ展開）。**組ごとに
      // ちょうど 1 回**で、accumulator に動的添字が残っていないことが条件（下の `acc[` 検査が対）。
      for (let i = 0; i < 8; i += 1) {
        for (const [quad, lanes] of [[0, [0, 1, 2, 3]], [1, [4, 5, 6, 7]]] as const) {
          const inner = `      acc${i}_${quad} = acc${i}_${quad} + vec4<i32>(${
            lanes.map((lane) => `idot(a${i}, b${lane})`).join(", ")
          });`;
          assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} quad ${quad} の内積`);
        }
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // MUST: P̃ は**成分ごとのスカラ式**で、充填スロット 2 本 × 4 成分ぶんちょうど
      // （vec4 へまとめて exp を掛けない）
      assertEquals(
        wgsl.split("round(exp(raw0.").length - 1,
        4,
        `${where}: P̃ の量子化式はスロット 0 の 4 成分ぶんちょうど`,
      );
      assertEquals(
        wgsl.split("round(exp(raw1.").length - 1,
        4,
        `${where}: P̃ の量子化式はスロット 1 の 4 成分ぶんちょうど`,
      );
      assertEquals(wgsl.includes("exp(vec4"), false, `${where}: exp をベクトルへまとめない`);
      // MUST: 格子は 127（128 化は分母型の退行）。scale は 1/127 の**乗算**で作る（除算禁止）
      assertEquals(wgsl.includes("- row_max0) * 127.0"), true, `${where}: P̃ の格子`);
      assertEquals(
        wgsl.includes("stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496"),
        true,
        `${where}: prow = inv·(1/127) の乗算`,
      );
      assertEquals(wgsl.includes("/ 127"), false, `${where}: 1/127 を除算で作っている`);
      // MUST: A タイルの範囲外は 0 のまま（exp(0−m)·127 を掛けると端数タイルが静かに誤る）
      assertEquals(
        wgsl.includes("    var av0 = 0u;\n    if (arow0 < dims.m && apack < k4) {"),
        true,
        `${where}: 範囲外 0 埋めの門`,
      );
      assertEquals(
        wgsl.includes("    var av1 = 0u;\n    if (arow1 < dims.m && apack < k4) {"),
        true,
        `${where}: 範囲外 0 埋めの門（スロット 1）`,
      );
      // MUST: dequant は prow·vs を先に 1 つの f32 へ畳む（bias が無いので fma は無い）
      assertEquals(
        wgsl.includes(
          v4
            ? "o[obase + orow0 * n4 + ocq0] = vec4<f32>(acc0_0) * (prow * vs);"
            : "o[orow_base + ocol] = f32(acc0_0.x) * (prow * vscale[vsbase + ocol]);",
        ),
        true,
        `${where}: dequant の乗算順序`,
      );
      assertEquals(wgsl.includes("fma("), false, `${where}: fma が混ざっている`);
      // MUST: 行統計は行 max が `arow`（A タイル充填）・行 inv が `orow`（書き出し）。
      // 片方をもう片方へ流用すると担当の違うスレッドの行が乗る（B·H = M = 1 でしか一致しない）
      // MUST: 行 max は**充填スロットごと**に引く（1 スレッドが 2 行を埋めるので、1 本で
      // 使い回すと隣の行の最大値が混ざる — 例外の出ない誤値）
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
      // MUST: バッチ base は「S = quad 単位 / vq = pack 単位 / 統計と列 scale = 本数単位」
      assertEquals(wgsl.includes("let sbase = wid.z * dims.m * k4;"), true, where);
      assertEquals(wgsl.includes("let vbase = wid.z * dims.n * k4;"), true, where);
      assertEquals(wgsl.includes("let rbase = wid.z * dims.m;"), true, where);
      assertEquals(wgsl.includes("let vsbase = wid.z * dims.n;"), true, where);
      assertEquals(
        wgsl.includes(`let obase = wid.z * dims.m * ${v4 ? "n4" : "dims.n"};`),
        true,
        where,
      );
      // vq は [D,N] の N 連続（パック方向）なので転置が要らない
      assertEquals(wgsl.includes("let vrow_base0 = vbase + wcol0 * k4;"), true, where);
      assertEquals(wgsl.includes("let vrow_base3 = vbase + wcol3 * k4;"), true, where);
      // 束縛: 5 = Vᵀ の行 scale（= V の列 scale）
      assertEquals(
        wgsl.includes(
          `@group(0) @binding(${ATTENTION_PV_V_SCALE_BINDING}) var<storage, read> vscale:`,
        ),
        true,
        where,
      );
      // 適格条件で N % 4 == 0 なので S は常に quad で読める（v4 フラグは出力 D 側だけの話）
      assertEquals(
        wgsl.includes("@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;"),
        true,
        where,
      );
      // MUST: タイル幾何は ①QK と**別**（③ だけ N = D = 128 の 1 タイル化が勝つ — 実測。
      // 「1 本の最良幾何」を仮定して束ねるとどちらかが必ず劣後する）
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 8)"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 64u + lid.y * 8u;"), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 256>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 512>;"), true, where);
      // ①QK と別物（骨格は共有だが充填も dequant も幾何も違う）
      assertNotEquals(wgsl, attentionQkI8a8Wgsl(v4, dp4a), where);
    }
  }
  // キーは ①QK と同じ規約（世代 ++ / dtype 欄 i8a8 / 幾何 / 内積変種は末尾）
  assertEquals(
    attentionPvI8a8Key(true, true),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16v4:dp4a",
  );
  assertEquals(
    attentionPvI8a8Key(false, false),
    "attention_pv:v3:i8a8:tile64x128r8x8w16x8k16:dp4aEmu",
  );
  // MUST: v4 判定は **出力の列 = D だけ**（N % 4 == 0 は適格判定が担う）
  assertEquals(attentionPvI8a8UsesVec4(128), true);
  assertEquals(attentionPvI8a8UsesVec4(19), false);
  // params は f32 変種と同じ 3 語（半スケールは ③ に登場しない）
  const params = attentionPvI8a8Params(5, 7, 8);
  assertEquals([...params], [5, 7, 8, 0]);
  assertEquals(params.byteLength, 16);
  assertThrows(() => attentionPvI8a8Params(5, 7, 7), CodegenError, "4 の倍数");
  assertThrows(
    () => attentionPvI8a8Params(5, 7, LINEAR_I8A8_MAX_K + 4),
    CodegenError,
    "i32 縮約の門",
  );
});

/**
 * i8a8 のタイル幾何（src/kernels/i8a8-geometry.ts）。**幾何は生成パラメタでありキー軸**という
 * 一点が崩れると「同一キー → バイト同一 WGSL」の決定性が壊れるので、
 * ① 既定値 ② キーへの反映 ③ 生成物への反映 ④ 割り切れない組み合わせの fail loudly
 * を 1 本で固定する。値そのものは i32 の厳密加算なので幾何を変えても 1 ビットも動かない
 * （それを実機で見るのは tests/gpu_i8a8_test.ts 側）。
 */
Deno.test("i8a8 の幾何はキーと生成物に反映され、割り切れない組み合わせは fail loudly", () => {
  // ① 既定: linear / ①QK は M128N64、③PV だけ M64N128（N = D = 128 の 1 タイル化が勝つ）
  assertEquals(defaultI8a8Geometry("linear"), { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 });
  assertEquals(defaultI8a8Geometry("attention_qk"), defaultI8a8Geometry("linear"));
  assertEquals(
    defaultI8a8Geometry("attention_pv"),
    { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
  );
  assertEquals(i8a8TileM(defaultI8a8Geometry("linear")), 128);
  assertEquals(i8a8TileN(defaultI8a8Geometry("linear")), 64);
  assertEquals(i8a8TileM(defaultI8a8Geometry("attention_pv")), 64);
  assertEquals(i8a8TileN(defaultI8a8Geometry("attention_pv")), 128);

  // ② キー: タイル辺だけでは幾何が決まらない（M128N64 は r8×8 wg8×16 とも r16×8 wg8×8 とも
  // 組める）ので、レジスタブロック・workgroup 形・K 幅まで載る
  const alt: I8a8Geometry = { regM: 16, regN: 8, wgX: 8, wgY: 8, tileK: 32 };
  assertEquals(i8a8TileM(alt), 128);
  assertEquals(i8a8TileN(alt), 64);
  assertEquals(i8a8GeometryKeyPart(alt, true), "tile128x64r16x8w8x8k32v4");
  assertNotEquals(
    i8a8GeometryKeyPart(alt, true),
    i8a8GeometryKeyPart(defaultI8a8Geometry("linear"), true),
  );
  for (
    const [where, keyOf] of [
      ["linear", (geometry: I8a8Geometry) => linearI8a8Key(true, true, geometry)],
      ["attention_qk", (geometry: I8a8Geometry) => attentionQkI8a8Key(true, true, "f32", geometry)],
      ["attention_pv", (geometry: I8a8Geometry) => attentionPvI8a8Key(true, true, "f32", geometry)],
    ] as const
  ) {
    assertNotEquals(keyOf(alt), keyOf(defaultI8a8Geometry("linear")), `${where}: 幾何がキーに無い`);
    assertEquals(keyOf(alt).includes("tile128x64r16x8w8x8k32v4"), true, where);
  }

  // ③ 生成物: workgroup 形・共有タイルのバイト数・充填スロット数が幾何どおりに動く
  // （tileK 32 = 8 pack / 64 スレッドなので A は 16 巡・B は 8 巡）
  for (
    const [where, wgsl] of [
      ["linear", linearI8a8Wgsl(true, true, alt)],
      ["attention_qk", attentionQkI8a8Wgsl(true, true, "f32", alt)],
      ["attention_pv", attentionPvI8a8Wgsl(true, true, "f32", alt)],
    ] as const
  ) {
    assertEquals(wgsl.includes("@compute @workgroup_size(8, 8)"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 1024>;"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<u32, 512>;"), true, where);
    assertEquals(wgsl.includes("let sa_at = ap * 128u + ar;"), true, where);
    assertEquals(wgsl.includes("let tiles = (k4 + 7u) / 8u;"), true, where);
    // 充填は tileM / tileN を**ちょうど覆う**（届かない語が残ると 0 のまま内積へ入る）
    assertEquals(wgsl.includes("sa[sa_at + 120u] = av15;"), true, `${where}: A 充填の最終スロット`);
    assertEquals(wgsl.includes("+ 56u] = "), true, `${where}: B 充填の最終スロット`);
    assertEquals(wgsl.includes("acc15_1 = acc15_1 +"), true, `${where}: 16 行ぶんの展開`);
    assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字`);
    assertNotEquals(wgsl, "", where);
  }
  assertNotEquals(linearI8a8Wgsl(true, true, alt), linearI8a8Wgsl(true, true));

  // ④ 割り切れない組み合わせは生成時に落とす（共有タイルの穴 = 例外の出ない誤値になる）
  const bad: readonly [string, I8a8Geometry][] = [
    // regN が 4 の倍数でない（acc の vec4 束ねが組めない）
    ["regN", { regM: 8, regN: 6, wgX: 8, wgY: 16, tileK: 16 }],
    // tileK が 4 の倍数でない（i8 の 4 詰めが割り切れない）
    ["tileK", { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 18 }],
    // スレッド数が K パック数で割り切れない（充填の担当が組めない）
    ["K パック数", { regM: 8, regN: 8, wgX: 3, wgY: 3, tileK: 16 }],
    // タイル辺が充填ストライドで割り切れない（届かない語が残る）
    ["充填ストライド", { regM: 3, regN: 8, wgX: 8, wgY: 16, tileK: 16 }],
    ["正整数", { regM: 0, regN: 8, wgX: 8, wgY: 16, tileK: 16 }],
  ];
  for (const [message, geometry] of bad) {
    assertThrows(() => linearI8a8Wgsl(true, true, geometry), CodegenError, message);
    assertThrows(() => attentionQkI8a8Wgsl(true, true, "f32", geometry), CodegenError, message);
    assertThrows(() => attentionPvI8a8Wgsl(true, true, "f32", geometry), CodegenError, message);
  }
});

/**
 * ② 行統計の **regcache 変種**（S を 1 回だけ読んでレジスタに残す）。
 *
 * MUST: 要素 → スレッドの割当（`lid + 256·s`）も縮約順（`s` 昇順 → 256 幅ツリー）も
 * 2 回読み版と**完全に同一**。ここが動くと ADR 0023 のビット同一（② は softmax のパス①② と
 * 逐語一致）が崩れるので、生成物の側で走査順と範囲外の扱いを固定する。
 */
Deno.test("attention_stats の regcache 変種は割当と縮約順を変えずに S を 1 回だけ読む", () => {
  const loop = attentionStatsWgsl();
  const cached = attentionStatsWgsl("f32", "f32", 3);
  // 2 回読み版の生成物は 1 バイトも動かない（既定経路の保護 — スナップショットと対）
  assertEquals(loop, ATTENTION_STATS_WGSL);
  assertEquals(loop.split("s[base + ").length - 1, 2, "2 回読み版は S を 2 度読む");
  // regcache 版は S の読みがちょうど epc 回（= 1 要素あたり 1 回）
  assertEquals(cached.split("s[base + ").length - 1, 3, "regcache 版は S を 1 度だけ読む");
  // 走査順は `lid + 256·s` の昇順（2 回読み版の `i += 256` と同じ列）
  assertEquals(cached.includes("  let i0 = lid;"), true);
  assertEquals(cached.includes("  let i1 = lid + 256u;"), true);
  assertEquals(cached.includes("  let i2 = lid + 512u;"), true);
  // ① max は読んだ値をそのまま畳む / ② 総和は同じ順で `exp(c - amax)` を足す
  assertEquals(cached.includes("      c0 = s[base + i0];\n      hi = max(hi, c0);"), true);
  assertEquals(
    cached.includes("    if (i2 < dim) {\n      acc = acc + exp(c2 - amax);\n    }"),
    true,
  );
  // MUST: 範囲外はどちらの段でも**触らない**（0.0 を混ぜると max も総和も静かに誤る）
  assertEquals(cached.split("if (i0 < dim) {").length - 1, 2, "範囲外の門が 2 段とも要る");
  // ツリー縮約と書き出しは 2 回読み版と逐語同一
  for (
    const shared of [
      "      scratch[lid] = max(scratch[lid], scratch[lid + stride]);",
      "      scratch[lid] = scratch[lid] + scratch[lid + stride2];",
      "    let inv = 1.0 / scratch[0u];",
      "      stats[row * 2u] = amax;",
      "    row = row + nwg.x;",
    ]
  ) {
    assertEquals(cached.includes(shared), true, `regcache 版に無い: ${shared}`);
  }
  // dim → epc は 1 箇所の純関数。上限を超える形は 2 回読みへ落ちる（値はどちらも同じ）
  assertEquals(attentionStatsRegCache(1), 1);
  assertEquals(attentionStatsRegCache(256), 1);
  assertEquals(attentionStatsRegCache(257), 2);
  assertEquals(attentionStatsRegCache(512), 2);
  assertEquals(attentionStatsRegCache(4096), 16);
  assertEquals(attentionStatsRegCache(ATTENTION_STATS_REG_CACHE_MAX * 256), 32);
  assertEquals(attentionStatsRegCache(ATTENTION_STATS_REG_CACHE_MAX * 256 + 1), undefined);
  // キーは `:rc{epc}` を末尾に足す（既定 = 2 回読みのキーは 1 文字も動かない）
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  assertEquals(attentionStatsKey("f32", "f32", 16), `${ATTENTION_STATS_KEY}:rc16`);
  assertEquals(attentionStatsKey("f32", "f16", 2), `${ATTENTION_STATS_KEY}:s16:rc2`);
  assertNotEquals(attentionStatsKey("f32", "f32", 2), attentionStatsKey("f32", "f32", 16));
});

// 裁定（src/kernels/gather.ts）: 範囲外添字は別要素を返さず NaN で汚染する。
Deno.test("gather は範囲外添字を NaN で汚染し、通常経路は行オフセットで読む", () => {
  assertEquals(GATHER_WGSL.includes("if (pick < 0 || u32(pick) >= dims.src_cols) {"), true);
  assertEquals(GATHER_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
  // MUST: NaN のビット列は params で運ぶ（定数式の `bitcast<f32>(0x…)` は「const-expression が
  // NaN」としてシェーダ生成エラーにする実装がありうる — カーネル doc の MUST）。
  assertEquals(
    GATHER_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN を定数式で作っている（params 経由にする）",
  );
  assertEquals(GATHER_WGSL.includes("out[i] = src[row * dims.src_cols + u32(pick)];"), true);
  // 出力は連続なので行は平坦添字から割る（grid-stride 前提）
  assertEquals(GATHER_WGSL.includes("let row = i / dims.cols;"), true);
  assertEquals(GATHER_WGSL.includes("var i = gid.x;"), true);
});

// ADR 0012 の融合 op。カーネル固有の不変条件を生成物の側から固定する。
Deno.test("融合カーネルは既存カーネルと別物で、契約どおりの形を生成する", () => {
  // linear は重みを [n,k] の転置レイアウトのまま読む（連続方向が k なので k 連続 4 要素）
  assertEquals(LINEAR_WGSL.includes("let wrow_base0 = wcol0 * dims.k;"), true);
  assertEquals(LINEAR_WGSL.includes("wv0.x = w[wbase];"), true);
  // MUST: 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）。2 つを取り違えると
  // 1 タイル内で列が入れ替わる — 列ごとに値が違う端数形状だけが検出器になる。
  assertEquals(LINEAR_WGSL.includes("let wsq0 = wc0 / 4u;"), true);
  assertEquals(LINEAR_WGSL.includes("let wsl0 = wc0 % 4u;"), true);
  // MUST: 成分は**静的**に書く（`sb[i][wsl] = v` の動的インデックスにしない）。Metal では
  // wsl != 0 の書き込みが黙って捨てられ、4 要素中 3 要素が 0 のまま内積へ入る（機序は
  // src/kernels/gemm.ts の storeBTransposed）。4 アームぶんの転置配置を全て固定する —
  // 1 アームでも取り違えると 1 タイル内で列が入れ替わる。
  assertEquals(LINEAR_WGSL.includes("[wsl]"), false);
  const components = ["x", "y", "z", "w"] as const;
  // 充填スロットごとに 4 アーム（1 スレッドが複数チャネルを埋める幾何）
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
        `linear の B タイル転置配置（スロット ${slot} 成分 ${component}）`,
      );
    }
  }
  // 末尾で bias を 1 度だけ足す（accumulator の初期値にはしない — 縮約順序を保つため）。
  // acc は `acc{行}_{列 quad}` の静的展開なので、行ごとに同じ形が並ぶ。
  assertEquals(LINEAR_WGSL.includes("out[obase + ocol] = acc0_0.x + bias[ocol];"), true);
  assertEquals(
    linearWgsl("f32", true).includes(
      "let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);",
    ),
    true,
  );
  assertEquals(linearWgsl("f32", true).includes("out[orow0 * n4 + ocq0] = acc0_0 + biasv;"), true);
  // MUST: matmul / bmm の生成物に融合 op の概念が漏れていない
  assertEquals(MATMUL_WGSL.includes("bias"), false);
  assertEquals(BMM_WGSL.includes("bias"), false);
  assertNotEquals(LINEAR_WGSL, MATMUL_WGSL);
  // MUST: 6 op は同じ内積ループ（= 同じ縮約順序）を共有する。1 箇所しか無いことを
  // 生成物の側から固定する（写し間違いが起きる余地を残さない）。`acc{行}_{列 quad}` の
  // 静的展開は添字と行番号が静的になるだけで、読む共有タイルの位置も加算順序も同一 —
  // 行ごとにちょうど 1 本ずつ並ぶ。
  const staticLoop = (row: number): string =>
    `      acc${row}_0 = acc${row}_0 + sa[(lid.y * 8u + ${row}u) * 16u + kk] * bv0;`;
  for (const [where, , wgsl] of GEMM_VARIANTS) {
    for (const line of [0, 1, 2, 3, 4, 5, 6, 7].map(staticLoop)) {
      assertEquals(wgsl.split(line).length, 2, `${where}: 内積ループが 1 箇所でない`);
    }
    assertEquals(wgsl.includes("let tiles = (dims.k + 15u) / 16u;"), true, where);
  }

  // layer_norm は 2 パス（平均 → 偏差平方和）で、分散は母分散（N 割り）
  assertEquals(LAYER_NORM_WGSL.includes("let d = x[base + j] - mean;"), true);
  assertEquals(LAYER_NORM_WGSL.includes("let scale = 1.0 / f32(dim);"), true);
  assertEquals(LAYER_NORM_WGSL.includes("sqrt(scratch[0u] * scale + params.eps)"), true);

  // rms_norm は**平均を引かない**（layer_norm の 2 パスを写していないことを形で固定する）
  assertEquals(RMS_NORM_WGSL.includes("acc = acc + v * v;"), true);
  assertEquals(RMS_NORM_WGSL.includes("mean"), false, "rms_norm に平均の概念は無い");
  assertEquals(RMS_NORM_WGSL.includes("bias"), false, "rms_norm に bias は無い（アリティ 2）");
  assertEquals(
    RMS_NORM_WGSL.includes("let inv = inverseSqrt(scratch[0u] * scale + params.eps);"),
    true,
  );
  assertEquals(RMS_NORM_WGSL.includes("out[base + o] = x[base + o] * inv * weight[o];"), true);
  assertNotEquals(RMS_NORM_WGSL, LAYER_NORM_WGSL);
  assertNotEquals(RMS_NORM_KEY, LAYER_NORM_KEY);
  assertEquals(RMS_NORM_KEY, `rms_norm:v1:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}`);
  assertEquals(RMS_NORM_WORKGROUP_SIZE, 256);
  // MUST: layer_norm の生成物に rms_norm の概念が漏れていない（決定性スナップショットの前提）
  assertEquals(LAYER_NORM_WGSL.includes("inverseSqrt"), false);

  // MUST: softmax は safe-softmax（amax 減算）。素朴形は大入力で underflow / overflow する
  assertEquals(SOFTMAX_WGSL.includes("let amax = scratch[0u];"), true);
  assertEquals(SOFTMAX_WGSL.includes("acc = acc + exp(x[base + j] - amax);"), true);
  assertEquals(SOFTMAX_WGSL.includes("out[base + o] = exp(x[base + o] - amax) * inv;"), true);

  // safe_softmax（ADR 0044）は softmax の ①②③ を**逐語で**共有し、足すのは identity の
  // -inf 化・空行判定・書き出しの select だけ。ここが崩れると parity（分解ガード相当との
  // ビット同一）が黙って壊れるので、共有断片を素の softmax 側と突き合わせて固定する。
  for (
    const shared of [
      "      hi = max(hi, x[base + i]);",
      "        scratch[lid] = max(scratch[lid], scratch[lid + stride]);",
      "      acc = acc + exp(x[base + j] - amax);",
      "        scratch[lid] = scratch[lid] + scratch[lid + stride2];",
      "    let inv = 1.0 / scratch[0u];",
    ]
  ) {
    assertEquals(SAFE_SOFTMAX_WGSL.includes(shared), true, `safe_softmax 側に無い断片: ${shared}`);
    assertEquals(SOFTMAX_WGSL.includes(shared), true, `softmax 側に無い断片: ${shared}`);
  }
  // 変種が足す 3 点（-inf identity / 空行判定 / 書き出しの select）
  assertEquals(SAFE_SOFTMAX_WGSL.includes("let neg_inf = bitcast<f32>(params.neg_inf);"), true);
  assertEquals(SAFE_SOFTMAX_WGSL.includes("var hi = neg_inf;"), true);
  assertEquals(SAFE_SOFTMAX_WGSL.includes("let empty = row_max == neg_inf;"), true);
  assertEquals(
    SAFE_SOFTMAX_WGSL.includes(
      "out[base + o] = select(exp(x[base + o] - amax) * inv, 0.0, empty);",
    ),
    true,
  );
  // MUST: 素の softmax 側に空行の概念が 1 語も漏れていない（契約は ADR 0044 決定 3 のまま）
  assertEquals(SOFTMAX_WGSL.includes("neg_inf"), false);
  assertEquals(SOFTMAX_WGSL.includes("empty"), false);
  assertNotEquals(SAFE_SOFTMAX_KEY, SOFTMAX_KEY);
  assertEquals(SOFTMAX_KEY, `softmax:v1:f32:lastdim:safe:wg${SOFTMAX_WORKGROUP_SIZE}`);
  assertEquals(
    SAFE_SOFTMAX_KEY,
    `safe_softmax:v1:f32:lastdim:safe:emptyrow0:wg${SOFTMAX_WORKGROUP_SIZE}`,
  );
  assertEquals(SOFTMAX_WORKGROUP_SIZE, 256);

  // embedding は範囲外添字を NaN で汚染する（gather と同じ裁定）
  assertEquals(EMBEDDING_WGSL.includes("if (pick < 0 || u32(pick) >= dims.vocab) {"), true);
  assertEquals(EMBEDDING_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
  // MUST: NaN のビット列は params で運ぶ（gather / deform_conv2d と同じ規律）。
  assertEquals(
    EMBEDDING_WGSL.includes("bitcast<f32>(0x"),
    false,
    "NaN を定数式で作っている（params 経由にする）",
  );
  assertEquals(EMBEDDING_WGSL.includes("weight[u32(pick) * dims.hidden + col]"), true);

  // masked_fill は x を連続で読み、mask だけ stride 経由（右詰め broadcast）
  assertEquals(
    MASKED_FILL_WGSL.includes("out[i] = select(x[i], fill, mask[mask_index] != 0u);"),
    true,
  );
  assertEquals(
    MASKED_FILL_WGSL.includes(`let fill = bitcast<f32>(params[${1 + 2 * STRIDED_RANK}u]);`),
    true,
  );

  // conv1d は padding 域を「読み飛ばす」（0 を足す形にしない）
  assertEquals(CONV1D_WGSL.includes("if (ix >= 0 && u32(ix) < dims.length_in) {"), true);
  assertEquals(CONV1D_WGSL.includes("var acc = bias[oc];"), true);
  assertEquals(
    CONV1D_WGSL.includes("let origin = i32(ox * dims.stride) - i32(dims.padding);"),
    true,
  );
  // dilation はカーネル位置に掛ける（出力位置 ox 側に掛ける誤りは stride と見分けがつかない）
  assertEquals(CONV1D_WGSL.includes("let ix = origin + i32(k * dims.dilation);"), true);
  // 重みの第 2 軸は Cin ではなく Cin/groups（depthwise で重みを読み飛ばす誤りの押さえ）
  assertEquals(
    CONV1D_WGSL.includes("let w_base = (oc * in_per_group + ic_rel) * dims.kernel;"),
    true,
  );

  // conv2d は空間 2 軸を独立に持つ（H/W を 1 本に潰していないことを形で固定する）
  assertEquals(
    CONV2D_WGSL.includes("let origin_y = i32(oy * dims.stride_h) - i32(dims.padding_h);"),
    true,
  );
  assertEquals(
    CONV2D_WGSL.includes("let origin_x = i32(ox * dims.stride_w) - i32(dims.padding_w);"),
    true,
  );
  // padding 域は軸ごとに読み飛ばす（H は continue・W は条件付き加算）
  assertEquals(CONV2D_WGSL.includes("if (iy < 0 || u32(iy) >= dims.height_in) {"), true);
  assertEquals(CONV2D_WGSL.includes("if (ix >= 0 && u32(ix) < dims.width_in) {"), true);
  // 重みは [Cout, Cin/groups, Kh, Kw]（Kw が最内・行送りは Kw）
  assertEquals(
    CONV2D_WGSL.includes(
      "let w_base = (oc * in_per_group + ic_rel) * dims.kernel_h * dims.kernel_w;",
    ),
    true,
  );
  assertEquals(CONV2D_WGSL.includes("let w_row = w_base + kh * dims.kernel_w;"), true);
  assertEquals(CONV2D_WGSL.includes("acc = acc + x[row_base + u32(ix)] * w[w_row + kw];"), true);
  assertEquals(CONV2D_WGSL.includes("var acc = bias[oc];"), true);
  // 出力の全バイトを書く（+= ではなく = の 1 回書き）
  assertEquals(CONV2D_WGSL.includes("out[i] = acc;"), true);
  assertNotEquals(CONV2D_WGSL, CONV1D_WGSL);
  assertNotEquals(CONV2D_KEY, CONV1D_KEY);
  // MUST: conv1d の生成物に 2 次元の概念が漏れていない（既存スナップショットの温存）
  assertEquals(CONV1D_WGSL.includes("width"), false);

  // conv_transpose1d は gather 形（出力 1 要素 = 寄与する入力の総和 — full-write）で、有効 tap は
  // residue 分割で数え上げる（perf-ledger K-10）。MUST: 剰余は出力要素あたり 1 回だけで、
  // k 全数走査 + 割り切れ判定の形に戻っていない（tap 集合は同じでも実測形の 7/8 が無効判定）。
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("let r = shifted % dims.stride;"), true);
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("% dims.stride == 0u"), false);
  // j の範囲は 3 本の不等式（ix <= L-1 / ix >= 0 / k <= K-1）で閉じる
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
  // MUST: j 昇順 = k 昇順（縮約順序が k 全数走査版と同一 = f32 のビット同一の根拠）
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("let k = r + u32(j) * dims.stride;"), true);
  // 重みは [Cin, Cout, K]（conv1d の [Cout, Cin, K] と転置）
  assertEquals(
    CONV_TRANSPOSE1D_WGSL.includes("let w_base = (ic * dims.channels_out + oc) * dims.kernel;"),
    true,
  );
  // 出力の全バイトを書く（+= ではなく = の 1 回書き）
  assertEquals(CONV_TRANSPOSE1D_WGSL.includes("out[i] = acc;"), true);
});

/**
 * 重み f16 格納の変種（ADR 0018）。5 本とも「重みの束縛型」と「読み出し 1 行」だけが違い、
 * 残りは f32 変種と同じであることを生成物の側から固定する。
 *
 * MUST: 対の選択は**平坦添字**（f32 変種が `w[...]` に渡している式そのもの）から作る。
 * 行内の相対添字で偶奇を取る誤りは、行長が偶数のテストでは数値が一致してしまう。
 */
const dropHeader = (wgsl: string): string => wgsl.slice(wgsl.indexOf("\n") + 1);

Deno.test("w=f16 変種は重みだけを array<u32> + unpack2x16float で読み、他は f32 変種と同一", () => {
  // (f32 変種, f16 変種, 重み配列名, f32 変種での読み出し式（複数可）)
  const variants: readonly (readonly [string, string, string, readonly string[]])[] = [
    [LINEAR_WGSL, linearWgsl("f16", false), "w", [
      "wbase",
      "wbase + 1u",
      "wbase + 2u",
      "wbase + 3u",
    ]],
    [EMBEDDING_WGSL, embeddingWgsl("f16"), "weight", ["u32(pick) * dims.hidden + col"]],
    [CONV1D_WGSL, conv1dWgsl("f16"), "w", ["w_base + k"]],
    [CONV2D_WGSL, conv2dWgsl("f16"), "w", ["w_row + kw"]],
    [CONV_TRANSPOSE1D_WGSL, convTranspose1dWgsl("f16"), "w", ["w_base + k"]],
  ];
  for (const [f32Wgsl, f16Wgsl, name, indices] of variants) {
    const where = `${name} / ${indices.join(" , ")}`;
    assertNotEquals(f16Wgsl, f32Wgsl, where);
    // 重みだけが u32 束縛（入力・bias・出力は f32 のまま）
    assertEquals(f16Wgsl.includes(`read> ${name}: array<u32>;`), true, where);
    assertEquals(f16Wgsl.includes(`read> ${name}: array<f32>;`), false, where);
    assertEquals(f16Wgsl.includes("read_write> out: array<f32>;"), true, where);
    // 展開は unpack2x16float 1 本（optional feature の enable は使わない — ADR 0018）
    assertEquals(f16Wgsl.includes(`let pair = unpack2x16float(${name}[i >> 1u]);`), true, where);
    assertEquals(f16Wgsl.includes("return select(pair.x, pair.y, (i & 1u) == 1u);"), true, where);
    assertEquals(f16Wgsl.includes("enable f16"), false, where);
    // MUST: 読み出しは f32 変種と**同じ添字式**を展開関数に渡す（平坦添字の偶奇で対を選ぶ）
    let rewound = f16Wgsl
      .replace(/\/\/ f16 格納の展開:[\s\S]*?\n}\n\n/, "")
      .replace(`read> ${name}: array<u32>;`, `read> ${name}: array<f32>;`);
    for (const index of indices) {
      assertEquals(f32Wgsl.includes(`${name}[${index}]`), true, `${where}: f32 変種の読み出し`);
      assertEquals(f16Wgsl.includes(`dequant(${index})`), true, where);
      assertEquals(f16Wgsl.includes(`${name}[${index}]`), false, where);
      // MUST: 全出現を戻す（GEMM 骨格は 1 スレッドが複数チャネルを埋めるので、同じ添字式が
      // 充填スロットの本数ぶん現れる）
      rewound = rewound.replaceAll(`dequant(${index})`, `${name}[${index}]`);
    }
    // 差はこの 3 点だけ（先頭コメントの但し書き / 展開関数 / 束縛と読み出しの各 1 行）。
    // 逆写像を当てて f32 変種そのものに戻ることで「他は一切動いていない」を固定する。
    assertEquals(dropHeader(rewound), dropHeader(f32Wgsl), where);
  }
});

/**
 * GEMM の v4 経路が使う **quad 展開**（ADR 0018 / 0019 の意味論は保ったまま読み出し命令だけ
 * まとめる）。
 *
 * MUST: quad 版は「平坦添字が 4 の倍数」に依存する（行頭が語境界に来ることへの依存では
 * ない）。スカラ経路の検出器（f16 = 行長が 2 の倍数でない / i8 = 4 の倍数でない）は無傷で、
 * v4 経路はその罠を踏まない代わりに検出もできない — **スカラ側のテストを削らない**ことが
 * 検出力の条件。
 */
Deno.test("linear の v4 変種は重みを quad 展開で読み、他は f32 v4 変種と同一", () => {
  const f32Wgsl = linearWgsl("f32", true);
  // f32 は vec4<f32> 配列の quad 添字（平坦添字 >> 2）
  assertEquals(f32Wgsl.includes("read> w: array<vec4<f32>>;"), true);
  assertEquals(f32Wgsl.includes("wv0 = w[(wrow_base0 + wk0) >> 2u];"), true);

  const f16Wgsl = linearWgsl("f16", true);
  // f16 は u32 2 語で 4 要素（語をまたがない）。スカラ版の dequant は出さない
  assertEquals(f16Wgsl.includes("let lo = unpack2x16float(w[i >> 1u]);"), true);
  assertEquals(f16Wgsl.includes("let hi = unpack2x16float(w[(i >> 1u) + 1u]);"), true);
  assertEquals(f16Wgsl.includes("return vec4<f32>(lo.x, lo.y, hi.x, hi.y);"), true);
  assertEquals(f16Wgsl.includes("fn dequant(i: u32)"), false, "スカラ版は出さない");
  assertEquals(f16Wgsl.includes("wv0 = dequant4(wrow_base0 + wk0);"), true);

  const i8Wgsl = linearWgsl("i8", true);
  // MUST: scale は**成分ごとの f32 乗算**（vec4 * scalar）— スカラ経路と同一の丸め（ADR 0019）。
  // 縮約の外へ出す形（acc * scale）は ADR 0019 の改訂と tolerance 再導出を発火させる。
  assertEquals(i8Wgsl.includes("return vec4<f32>(unpack4xI8(w[i >> 2u])) * scale;"), true);
  // MUST: scale は充填スロットごとに別名で束ねる（スロット 0 の scale を 2 本目にも使うと、
  // 担当チャネルの片方だけが別チャネルの scale で dequant される沈黙誤値になる）
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[wcol0];"), true, "scale は K ループの外");
  assertEquals(i8Wgsl.includes("let wscale_v1 = wscale[wcol1];"), true, "スロットごとに別名");
  assertEquals(i8Wgsl.includes("wv0 = dequant4(wrow_base0 + wk0, wscale_v);"), true);
  assertEquals(i8Wgsl.includes("wv1 = dequant4(wrow_base1 + wk0, wscale_v1);"), true);
  // acc は `acc{行}_{列 quad}` の静的展開なので、縮約後に掛ける形はこの名前で見る
  assertEquals(i8Wgsl.includes("acc0_0 * "), false, "縮約の外で scale を掛けていない");

  // 逆写像で f32 v4 変種そのものに戻る（他は一切動いていない）
  const scaleArg = (slot: number): string => slot === 0 ? "wscale_v" : `wscale_v${slot}`;
  for (
    const [wgsl, call] of [
      [f16Wgsl, (slot: number) => `dequant4(wrow_base${slot} + wk0)`],
      [i8Wgsl, (slot: number) => `dequant4(wrow_base${slot} + wk0, ${scaleArg(slot)})`],
    ] as const
  ) {
    let rewound = wgsl
      .replace(/\n@group\(0\) @binding\(5\) var<storage, read> wscale: array<f32>;\n/, "")
      .replace(/\/\/ (f16|i8) 格納の quad 展開:[\s\S]*?\n}\n\n/, "")
      .replaceAll(
        /\n {2}\/\/ 出力チャネルの scale はループ不変[\s\S]*?\n {2}let wscale_v\d* = wscale\[wcol\d\];/g,
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
 * f16 **計算**変種（ADR 0028）。生成物の側から丸め列を固定する — 数値契約
 * 「f16 変種 = 入力を f16 に丸めた f32 変種」はこの 4 点だけで成立している:
 *
 * 1. 丸めは**共有タイルへ書く 1 箇所**（`f16(...)` / `vec4<f16>(...)`）。
 * 2. 拡幅は**レジスタロード時に 1 回**（MAC ごとの `f32(av * bv)` は禁止 — プロトタイプの
 *    負の教訓で、変換回数が倍になるうえ積が f16 精度に落ちる）。
 * 3. 累積は f32（`acc{行}_{列 quad}` の静的展開で、要素型は `vec4<f32>` のまま）。
 * 4. uniform は f16 にしない。
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

Deno.test("f16 計算変種は共有タイルだけを f16 にし、丸めと拡幅の位置を 1 箇所に閉じる", () => {
  for (const [where, wgsl] of COMPUTE_F16_VARIANTS) {
    const lines = wgsl.split("\n");
    // enable はモジュール先頭（コメント 1 行の直後 = 全ての global 宣言より前）
    assertEquals(lines[0].startsWith("// karume "), true, where);
    assertEquals(lines[1], "enable f16;", where);
    // ① 共有タイルだけが f16（バイト半減がこの変種の唯一の機序。辺は既定幾何 128×128）
    assertEquals(wgsl.includes("var<workgroup> sa: array<f16, 2048>;"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f16>, 512>;"), true, where);
    // ③ 累積は f32 のまま（k 大の桁落ちを避ける — 変えると parity と E2E が赤くなる）
    assertEquals(wgsl.includes("var acc0_0 = vec4<f32>(0.0);"), true, where);
    // ② 拡幅はレジスタロード時に 1 回。MAC ごとの積の変換（f32(av * bv)）は出さない
    assertEquals(wgsl.includes("let bv0 = vec4<f32>(sb[kk * 32u + lid.x * 2u]);"), true, where);
    assertEquals(
      wgsl.includes("acc0_0 = acc0_0 + f32(sa[(lid.y * 8u + 0u) * 16u + kk]) * bv0;"),
      true,
      where,
    );
    // MUST: 判定はコメントを除いた本体で行う（禁じている形をコメントに書いてあるため）
    const code = wgsl.split("\n").filter((line) => !line.trimStart().startsWith("//")).join("\n");
    assertEquals(code.includes("f32(av"), false, `${where}: MAC ごとの変換`);
    // ① 丸めは共有タイルへの書き込みだけ（A タイルは 4 成分とも f16(...) で包む）
    assertEquals(wgsl.includes("sa[sa_base0] = f16(av0.x);"), true, where);
    assertEquals(wgsl.includes("sa[sa_base0 + 3u] = f16(av0.w);"), true, where);
    assertEquals(wgsl.includes("sa[sa_base1] = f16(av1.x);"), true, where);
    // ④ uniform は f16 にしない（uniform の配列要素は 16B 整列で、f16 にする利得も無い）
    assertEquals(wgsl.includes("var<uniform> dims: Dims;"), true, where);
    assertEquals(/^\s+[mnk]: f16,$/m.test(wgsl), false, `${where}: uniform が f16`);
  }
  // 融合 attention の S は f16 で受け渡す（① が書き ②③ が読む = transient 半減）
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
  // ② 行統計も S を f32 へ広げてから縮約する（縮約の丸め列は f32 変種と同一）
  const stats16 = attentionStatsWgsl("f16");
  assertEquals(stats16.includes("var<storage, read> s: array<f16>;"), true);
  assertEquals(stats16.includes("hi = max(hi, f32(s[base + i]));"), true);
  assertEquals(stats16.includes("acc = acc + exp(f32(s[base + j]) - amax);"), true);
  assertEquals(stats16.includes("var<workgroup> scratch: array<f32, 256>;"), true, "縮約は f32");
});

Deno.test("既定（f32 計算）の生成物には f16 が 1 文字も現れない", () => {
  // 格納 f16（ADR 0018）は但し書きコメントに "f16" を含むので、ここは**格納も f32** の変種で見る。
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

Deno.test("計算変種の判別子はキーの f16 側だけに付き、格納の wf16 とは別語になる", () => {
  // 計算変種の語は幾何判別子の**後ろ**に付く（キーの語順そのものが固定対象）
  assertEquals(linearKey("f32", false), "linear:v2:f32:reg128x128r8x8w16");
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg128x128r8x8w16v4:wf16");
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg128x128r8x8w16v4");
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  // f16 計算は末尾に :c16（格納 :wf16 と重ならない語 — 両方同時に立つ組み合わせがある）
  assertEquals(linearKey("f32", false, "f16"), "linear:v2:f32:reg128x128r8x8w16:c16");
  assertEquals(linearKey("f16", true, "f16"), "linear:v2:f32:reg128x128r8x8w16v4:wf16:c16");
  assertEquals(attentionQkKey(false, "f16"), "attention_qk:v1:f32:reg128x128r8x8w16:c16");
  assertEquals(attentionPvKey(true, "f16"), "attention_pv:v1:f32:reg128x128r8x8w16v4:c16");
  assertEquals(attentionStatsKey("f16"), `${ATTENTION_STATS_KEY}:c16`);
});

// `acc[i]` の動的添字はアドレス可能な関数ローカル領域を要求し、accumulator がレジスタから
// ローカルメモリへ落ちる。**骨格を共有する 6 op すべて**が既定経路でこの静的展開になる。
Deno.test("GEMM 骨格の accumulator は acc{行}_{列 quad} の静的展開で、動的添字を残さない", () => {
  for (const [where, , wgsl] of GEMM_VARIANTS) {
    assertEquals(wgsl.includes("var acc = array<"), false, where);
    // MUST: 動的添字が 1 つも残っていない（残ると展開の動機がそのまま消える）
    assertEquals(wgsl.includes("acc["), false, where);
    for (let row = 0; row < 8; row += 1) {
      // 行 row の内積更新はちょうど 1 回（K タイル 16・kk 昇順・加算順序は展開前と同一）
      assertEquals(
        wgsl.split(`acc${row}_0 = acc${row}_0 + sa[(lid.y * 8u + ${row}u) * 16u + kk] * bv0;`)
          .length,
        2,
        `${where}:update${row}`,
      );
      // 書き出しも行ごとに静的な名前を読む
      assertEquals(wgsl.includes(`orow${row} < dims.m`), true, `${where}:store${row}`);
    }
    // 列 quad は 2 本（regN = 8 = vec4 × 2）— 3 本目が生えたら幾何が変わっている
    assertEquals(wgsl.includes("acc0_1"), true, `${where}: 列 quad が 2 本ない`);
    assertEquals(wgsl.includes("acc0_2"), false, `${where}: 列 quad が 3 本ある`);
  }
  // 0 初期化は bias-first の conv2d 以外の全 op（conv2d は別テストが bias 初期値を見る）
  for (const wgsl of [matmulWgsl(true), linearWgsl("f32", false), attentionPvWgsl(true)]) {
    assertEquals(wgsl.includes("var acc0_0 = vec4<f32>(0.0);"), true);
    assertEquals(wgsl.includes("var acc7_0 = vec4<f32>(0.0);"), true);
  }
});

Deno.test("重み i8 格納 × f16 計算（w8a16）は生成の入口で fail loudly", () => {
  assertThrows(
    () => linearWgsl("i8", true, "f16"),
    CodegenError,
    "w8a16",
  );
  // 組み合わせ以外は通る（i8 × f32 計算 / f16 格納 × f16 計算）
  linearWgsl("i8", true);
  linearWgsl("f16", true, "f16");
});

/**
 * S の **f16 格納**変種（案 γ 波 1）。生成物の側から数値契約
 * 「s16 変種 = S をホストで f16 に丸めた f32 変種」を固定する。成立の条件は 3 つ:
 *
 * 1. 丸めは **`pack2x16float` の 1 箇所だけ**（`score_store`）。読み側の `unpack2x16float` は
 *    厳密なので丸めではない。
 * 2. **`enable f16` を出さない**（core WGSL — ADR 0030 決定 1「i8a8 は shader-f16 を要求
 *    しない」が s16 と組んでも無傷であることの生成物側の証拠）。
 * 3. 書き手は **v4 経路だけ**（1 スレッドが 4 連続列 = 2 語ちょうどを排他に書く）。
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

Deno.test("S の f16 格納変種は pack2x16float 1 点だけで丸め、shader-f16 を要求しない", () => {
  for (const [where, wgsl, side] of SCORE_F16_VARIANTS) {
    // ② core WGSL のまま（feature 非依存 — ここが崩れると i8a8 が feature に縛られる）
    assertEquals(wgsl.includes("enable f16;"), false, `${where}: enable f16 が出ている`);
    // S の束縛は必ず u32 の平坦配列（vec4<f32> / f16 のまま残っていたら型が合わない）
    assertEquals(/var<storage, read(_write)?> s: array<u32>;/.test(wgsl), true, `${where}: S 束縛`);
    // MUST: 判定はコメントを除いた本体で（禁じている形をコメントに書いてあるため）
    const code = wgsl.split("\n").filter((line) => !line.trimStart().startsWith("//")).join("\n");
    // MUST: `un` を含まない `pack2x16float` だけを数える（部分一致で unpack を巻き込むと
    // 「丸めが 1 箇所」の主張が読み側の本数で埋まって恒真になる）
    const packs = code.match(/(?<!un)pack2x16float\(/g) ?? [];
    const unpacks = code.match(/unpack2x16float\(/g) ?? [];
    if (side === "write") {
      // ① 丸めは 2 語ぶんの pack だけ（`score_store` の 2 行）。読みは 1 つも無い
      assertEquals(packs.length, 2, `${where}: pack2x16float の本数`);
      assertEquals(unpacks.length, 0, `${where}: 書き手が unpack している`);
      assertEquals(
        code.includes("  s[w] = pack2x16float(value.xy);\n  s[w + 1u] = pack2x16float(value.zw);"),
        true,
        `${where}: 2 語を続けて書いていない`,
      );
      // ③ v4 経路の quad 書き出しだけが score_store を呼ぶ（部分書きのガードが残っていない）
      assertEquals(
        code.includes("score_store(") && !/if \(ocol( \+ \d+u)? < dims\.n\)/.test(code),
        true,
        `${where}: スカラ経路の部分書きが残っている`,
      );
    } else {
      assertEquals(packs.length, 0, `${where}: 読み手が pack している`);
      assertEquals(unpacks.length > 0, true, `${where}: unpack2x16float が無い`);
    }
  }
  // 読み側の添字算術は f32 変種と同じ形のまま（quad は quad・要素は要素）
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
    attentionStatsWgsl("f32", "f16").includes("hi = max(hi, score_at(base + i));"),
    true,
  );
  // 書き手は f32 変種と同じ値式のまま（丸めが値の計算へ紛れ込んでいない）
  assertEquals(
    attentionQkWgsl(true, "f32", "f16").includes(
      "score_store(cbase + orow0 * n4 + ocq0, acc0_0);",
    ),
    true,
  );
});

Deno.test("格納 S の判別子 s16 はキーの末尾に付き、wf16 / c16 と別語になる", () => {
  // 既定（f32 格納）のキーは幾何判別子までで終わる
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
  // s16 は**末尾**（3 つの軸が同時に立ちうるので語順を 1 箇所で固定する）
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
  // 3 語（格納重み wf16 / 計算 c16 / 格納 S s16）は互いに衝突しない
  assertEquals(new Set(["wf16", "c16", "s16"]).size, 3);
  // 同一構成が 2 通りのキーを持たない（生成しうる attention キーの全数が一意）
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
  assertEquals(new Set(keys).size, keys.length, "attention のキーに重複がある");
});

Deno.test("S の f16 格納は「スカラ経路」と「c16 との併用」を生成の入口で fail loudly", () => {
  // スカラ経路の部分書きは同じ u32 語への read-modify-write（黙って f32 格納へ落とさない）
  for (
    const [where, generate] of [
      ["attention_qk", () => attentionQkWgsl(false, "f32", "f16")],
      ["attention_pv", () => attentionPvWgsl(false, "f32", "f16")],
      ["attention_qk i8a8", () => attentionQkI8a8Wgsl(false, true, "f16")],
    ] as const
  ) {
    assertThrows(generate, CodegenError, "v4 経路専用", where);
  }
  // `:c16` は S を array<f16> で持つ別の格納形（冗長かつ矛盾する組）
  for (
    const [where, generate] of [
      ["attention_qk", () => attentionQkWgsl(true, "f16", "f16")],
      ["attention_pv", () => attentionPvWgsl(true, "f16", "f16")],
      ["attention_stats", () => attentionStatsWgsl("f16", "f16")],
    ] as const
  ) {
    assertThrows(generate, CodegenError, "同時に指定できない", where);
  }
  // 適格判定は書き手が v4 を取る条件そのもの（D%4 と N%4 の**両方**）
  for (
    const [depth, cols, expected] of [
      [4, 8, true],
      [128, 4096, true],
      [4, 6, false],
      [13, 8, false],
      [13, 19, false],
    ] as const
  ) {
    assertEquals(attentionScoreUsesF16(depth, cols), expected, `D=${depth} N=${cols}`);
    // 適格な形では f32 の ①QK も i8a8 の ①QK も v4 を取る（縮退の混成が起きない前提）
    if (expected) {
      assertEquals(gemmUsesVec4(depth, cols), true, `D=${depth} N=${cols}: f32 ①QK が v4`);
      assertEquals(attentionQkI8a8UsesVec4(cols), true, `D=${depth} N=${cols}: i8a8 ①QK が v4`);
      assertEquals(gemmUsesVec4(cols, depth), true, `D=${depth} N=${cols}: f32 ③PV が v4`);
      assertEquals(attentionPvI8a8UsesVec4(depth), true, `D=${depth} N=${cols}: i8a8 ③PV が v4`);
    }
  }
});

Deno.test("格納判別子はキーの f16 側だけに付く（既存の f32 キーはバイト単位で不変）", () => {
  const pairs: readonly (readonly [string, (weight: "f32" | "f16") => string])[] = [
    ["linear:v2:f32:reg128x128r8x8w16", (weight) => linearKey(weight, false)],
    ["embedding:v1:f32:i32:wg256", embeddingKey],
    ["conv1d:v2:f32:direct:wg256", conv1dKey],
    ["conv2d:v1:f32:direct:wg256", conv2dKey],
    ["conv_transpose1d:v2:f32:gather:wg256", convTranspose1dKey],
  ];
  for (const [expected, key] of pairs) {
    // MUST: f32 のキー文字列は変種導入の前後で完全に同じ（実キーを直書きして固定する）
    assertEquals(key("f32"), expected);
    assertEquals(key("f16"), `${expected}:wf16`);
  }
  // 格納判別子は v4 判別子の**後ろ**に付く（linear だけ 2 軸を持つ）
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg128x128r8x8w16v4:wf16");
  // i4 は group 長がキーに乗る（shift の焼き込みと対 — ADR 0069）
  assertEquals(
    linearKey("i4", true, "f32", undefined, 32),
    "linear:v2:f32:reg128x128r8x8w16v4:wi4g32",
  );
  assertEquals(embeddingKey("i4", 32), "embedding:v1:f32:i32:wg256:wi4g32");
  // group 長は i4 と 1 対 1（欠け / 余りはどちらも結線バグ — 黙って通すと g 部の無い
  // キーが立って group 違いの資産が同じパイプラインへ割り当たる）
  assertThrows(() => linearKey("i4", false), CodegenError, "対で渡す");
  assertThrows(() => linearKey("f32", false, "f32", undefined, 32), CodegenError, "対で渡す");
  assertThrows(() => linearKey("i4", false, "f32", undefined, 24), CodegenError, "2 冪");
  // 同じ規律が embedding にも掛かる（i4 の生成入口は 2 つあるので両方で言い直す）
  assertThrows(() => embeddingKey("i4"), CodegenError, "対で渡す");
  assertThrows(() => embeddingKey("i8", 32), CodegenError, "対で渡す");
  assertThrows(() => embeddingKey("i4", 8), CodegenError, "2 冪");
});

/**
 * conv1d の i4 変種（ADR 0069 決定 5 の conv1d 追補）は **implicit GEMM 限定**。
 *
 * ① キーに group 部が乗る（linear と同じ流儀 — 焼き込んだ shift と対）
 * ② group 長は i4 と 1 対 1（欠け / 余りはどちらも結線バグ）
 * ③ 直接カーネル（groups > 1）と conv2d には展開経路が無く、**生成の入口が落とす**
 *    — 適格判定（plan.ts）が閉じている前提に乗らず、直呼び経路も塞ぐ
 */
Deno.test("conv1d の i4 は implicit GEMM 限定で、キーに group 部が乗る", () => {
  // ① f32 / i8 のキーは変種導入の前後で完全に同じ（実キーを直書きして固定する）
  assertEquals(conv1dIgemmKey("f32", false), "conv1d:v3:f32:igemm64x128:wg16x8");
  assertEquals(conv1dIgemmKey("i8", true), "conv1d:v3:f32:igemm64x128v4:wg16x8:wi8");
  assertEquals(
    conv1dIgemmKey("i4", true, undefined, 32),
    "conv1d:v3:f32:igemm64x128v4:wg16x8:wi4g32",
  );
  // group 長が違えば別キー（同じ WGSL が group 違いの資産で走る沈黙誤値を塞ぐ）
  assertNotEquals(
    conv1dIgemmKey("i4", false, GEMM_MTILE_SMALL, 16),
    conv1dIgemmKey("i4", false, GEMM_MTILE_SMALL, 32),
  );
  // ② i4 と group 長は 1 対 1（キーも生成も同じ門を通る）
  assertThrows(() => conv1dIgemmKey("i4", false), CodegenError, "対で渡す");
  assertThrows(() => conv1dIgemmKey("i8", false, GEMM_TILE, 32), CodegenError, "対で渡す");
  assertThrows(() => conv1dIgemmKey("i4", false, GEMM_TILE, 24), CodegenError, "2 冪");
  assertThrows(() => conv1dIgemmWgsl("i4", false), CodegenError, "対で渡す");
  // ③ 直接カーネル（groups > 1 用）は i4 を受けない — キーも生成物も出さない
  assertThrows(() => conv1dKey("i4"), CodegenError, "i4");
  assertThrows(() => conv1dWgsl("i4"), CodegenError, "i4");
  // ③' conv2d の implicit GEMM も同様（A 側の展開器は 1D / 2D 共有なので**生成は通ってしまう**
  // — 落とさないと packed バイトに group scale 無しの添字が当たる）
  assertThrows(() => conv2dIgemmWgsl("i4", false), CodegenError, "i4");
  // i4 の生成物は group scale を A 側（重み側）の充填で quad ごとに引く
  const wgsl = conv1dIgemmWgsl("i4", false, undefined, 32);
  assertEquals(wgsl.includes("@group(0) @binding(5) var<storage, read> wscale: array<f32>;"), true);
  assertEquals(wgsl.includes("let ags0 = wscale[arow0 * (dims.k >> 5u) + (ak0 >> 5u)];"), true);
  // MUST: linear 側（B タイル = 列）の添字を持ってこない — conv は重みが A 側（行 = Cout）
  assertEquals(wgsl.includes("wcol"), false);
});

/**
 * i4 の展開経路を持たない残りの生成入口（conv2d の直接カーネル / conv_transpose1d）も
 * **生成の入口で落とす**（weight-storage.ts の「残りの生成入口に i4 を渡す経路は各生成関数が
 * 落とす」MUST）。
 *
 * 落とさないと `weightScaleWgsl` が i4 で空文字を返すぶん `wscale_v` が束縛されないまま
 * `dequant(…, wscale_v)` が出て、**未定義識別子の不成立 WGSL** が遠くのシェーダコンパイルで
 * 割れる（診断が生成の入口から離れる）。f32 / f16 / i8 の 3 変種は従来どおり通る。
 */
Deno.test("i4 の展開経路が無い conv2d 直接 / conv_transpose1d は生成の入口で落とす", () => {
  assertThrows(() => conv2dKey("i4"), CodegenError, "i4");
  assertThrows(() => conv2dWgsl("i4"), CodegenError, "i4");
  assertThrows(() => convTranspose1dKey("i4"), CodegenError, "i4");
  assertThrows(() => convTranspose1dWgsl("i4"), CodegenError, "i4");
  // 門は i4 だけを落とす（共有 3 変種は素通り — キーも生成物も従来の形のまま）
  for (const storage of WEIGHT_STORAGES) {
    const suffix = storage === "f32" ? "" : storage === "f16" ? ":wf16" : ":wi8";
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
 * weight-storage の quad 拡張は **opt-in**（{@link weightLoaderWgsl} / {@link weightArrayType} の
 * 第 4 / 第 2 引数）。
 *
 * MUST: 既存 4 op（conv1d / conv2d / conv_transpose1d / embedding）の生成物は 1 バイトも
 * 動かない。無条件に `dequant4` を足すと fixture が一斉に動き、パイプラインキャッシュの
 * 同一性（同一キー ⇔ 同一バイト列）も崩れる。
 */
Deno.test("weight-storage の quad 拡張は既存 4 op の生成物を 1 バイトも動かさない", async () => {
  const existing: readonly (readonly [string, string])[] = WEIGHT_STORAGES.flatMap((weight) => {
    const suffix = weight === "f32" ? "" : weight === "f16" ? "_wf16" : "_wi8";
    return [
      [`conv1d${suffix}.wgsl`, conv1dWgsl(weight)] as const,
      [`conv2d${suffix}.wgsl`, conv2dWgsl(weight)] as const,
      [`conv_transpose1d${suffix}.wgsl`, convTranspose1dWgsl(weight)] as const,
      [`embedding${suffix}.wgsl`, embeddingWgsl(weight)] as const,
    ];
  });
  assertEquals(existing.length, 12);
  for (const [name, generated] of existing) {
    assertEquals(generated, await fixture(name), `${name}: 拡張前のバイト列と違う`);
    // quad 版の部品が漏れていない（漏れれば上の突合でも落ちるが、原因を名指しできるようにする）
    assertEquals(generated.includes("dequant4"), false, `${name}: quad 展開が漏れている`);
    assertEquals(generated.includes("vec4<f32>"), false, `${name}: quad 束縛が漏れている`);
  }
});

Deno.test("融合カーネルの params は契約外の値を fail loudly にする", () => {
  assertThrows(() => linearParams(-1, 2, 3), CodegenError);
  assertEquals([...linearParams(5, 3, 7)], [5, 3, 7, 0]);

  // eps は f32 のビット列として載る（u32 として書くと指数部が整数値に化ける）
  const norm = layerNormParams(4, 8, 1e-7);
  assertEquals([norm[0], norm[1]], [4, 8]);
  assertEquals(new Float32Array(norm.buffer)[2], Math.fround(1e-7));
  assertThrows(() => layerNormParams(4, 0, 1e-7), CodegenError);
  assertThrows(() => layerNormParams(4, 8, 0), CodegenError);
  assertThrows(() => layerNormParams(4, 8, Number.POSITIVE_INFINITY), CodegenError);

  // rms_norm も eps は f32 のビット列（layer_norm と同じ規約）
  const rms = rmsNormParams(6, 10, 1e-6);
  assertEquals([rms[0], rms[1]], [6, 10]);
  assertEquals(new Float32Array(rms.buffer)[2], Math.fround(1e-6));
  assertEquals(rms.byteLength % 16, 0);
  assertThrows(() => rmsNormParams(6, 0, 1e-6), CodegenError);
  assertThrows(() => rmsNormParams(6, 10, 0), CodegenError);
  assertThrows(() => rmsNormParams(6, 10, Number.NaN), CodegenError);

  assertEquals([...softmaxParams(4, 9)], [4, 9, 0, 0]);
  assertThrows(() => softmaxParams(4, 0), CodegenError);
  // safe_softmax は 3 語目に -inf のビット列を載せる（WGSL に無限大リテラルが無いため）
  assertEquals([...softmaxParams(4, 9, true)], [4, 9, SAFE_SOFTMAX_NEG_INF_BITS, 0]);
  assertEquals(new Float32Array(softmaxParams(4, 9, true).buffer)[2], Number.NEGATIVE_INFINITY);
  assertThrows(() => softmaxParams(4, 0, true), CodegenError);

  assertEquals([...embeddingParams(8, 2, 7)], [8, 2, 7, EMBEDDING_OOB_BITS]);
  assertThrows(() => embeddingParams(6, 0, 7), CodegenError);
  assertThrows(() => embeddingParams(-1, 2, 7), CodegenError);

  // rank 不足は左詰めで dims=1 / strides=0 に埋める（strided 族と同じ埋め方）
  const filled = maskedFillParams([3, 4], [0, 1], -3.4028234663852886e+38);
  assertEquals([...filled.slice(1, 1 + STRIDED_RANK)], [1, 1, 3, 4]);
  assertEquals([...filled.slice(1 + STRIDED_RANK, 1 + 2 * STRIDED_RANK)], [0, 0, 0, 1]);
  assertEquals(new Float32Array(filled.buffer)[1 + 2 * STRIDED_RANK], -3.4028234663852886e+38);
  assertThrows(() => maskedFillParams([3, 4], [1], 0), CodegenError);
  assertThrows(() => maskedFillParams([], [], 0), CodegenError);
  assertThrows(() => maskedFillParams([2, 3], [1, 1], Number.NaN), CodegenError);

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
  // uniform の struct は 16 バイト整列（11 語ぶんの内容でも 12 語確保する）
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
  // groups が Cin / Cout を割り切らない形は params 層でも落ちる（シェーダの除算は切り捨て）
  assertThrows(() => convParams({ groups: 3 }), CodegenError);
  assertThrows(() => convParams({ channelsOut: 6, groups: 4 }), CodegenError);

  // conv1d の implicit GEMM は `{m, n, k}` + 幾何 6 語。**n は 1 バッチぶんの出力平面**
  // （バッチは dispatch の z 軸）で、`length_out` は n と同値なので Dims に持たない。
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
  assertEquals([...conv1Igemm.slice(0, 9)], [
    9, // m = Cout
    4, // n = Lout（B は掛けない）
    6 * 3, // k = Cin·K
    6,
    11,
    3,
    3,
    1,
    2,
  ]);
  assertEquals(conv1Igemm.byteLength % 16, 0);
  // 契約検査は直接カーネルと共有（同じ幾何は同じ理由で落ちる）+ groups > 1 は専用の門
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
  // MUST: groups > 1 は implicit GEMM の対象外（executor は直接カーネルへ流す）
  assertThrows(() => conv1IgemmParams({ groups: 2 }), CodegenError);
  assertThrows(() => conv1IgemmParams({ groups: 4 }), CodegenError);

  // conv2d は空間 2 軸ぶんの語を持つ（H と W を取り違えていないことを並びで固定する）
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
  assertEquals([...conv2.slice(0, 17)], [
    2 * 9 * 5 * 4,
    2,
    6,
    9,
    7,
    11,
    5,
    4,
    2,
    3,
    1,
    3,
    0,
    1,
    2,
    1,
    3,
  ]);
  // uniform の struct は 16 バイト整列（17 語ぶんの内容でも 20 語確保する）
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
  // MUST: 軸ごとに独立に見る（片側だけ 0 の形が素通りすると GPU ハングになる）
  assertThrows(() => conv2Params({ strideH: 0 }), CodegenError);
  assertThrows(() => conv2Params({ strideW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ dilationH: 0 }), CodegenError);
  assertThrows(() => conv2Params({ dilationW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ kernelW: 0 }), CodegenError);
  assertThrows(() => conv2Params({ groups: 0 }), CodegenError);
  assertThrows(() => conv2Params({ groups: 3 }), CodegenError);
  assertThrows(() => conv2Params({ channelsOut: 6, groups: 4 }), CodegenError);
  assertThrows(() => conv2Params({ paddingH: -1 }), CodegenError);

  // implicit GEMM は `{m, n, k}` + 幾何 12 語。**n は 1 バッチぶんの出力平面**（バッチは
  // dispatch の z 軸）で、m·n·k の導出を取り違えると shape 検査を素通りして誤値になる。
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
  assertEquals([...igemm.slice(0, 15)], [
    9, // m = Cout
    5 * 4, // n = Hout·Wout（B は掛けない）
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
  ]);
  assertEquals(igemm.byteLength % 16, 0);
  // 契約検査は直接カーネルと共有（同じ幾何は同じ理由で落ちる）+ groups > 1 は専用の門
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
  // MUST: groups > 1 は implicit GEMM の対象外（executor は直接カーネルへ流す）
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
  // MUST: stride 0 は WGSL の剰余がゼロ除算になる（recon §4 の GPU ハング前例）
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

Deno.test("bmm / gather の params は契約外の値を fail loudly にする", () => {
  assertThrows(() => bmmParams(-1, 2, 3), CodegenError);
  assertThrows(() => bmmParams(1, 2, 1.5), CodegenError);
  assertThrows(() => gatherParams(-1, 2, 3), CodegenError);
  // 列数 0 で要素数 > 0 は shape 契約上ありえない（カーネルの 0 除算を作らない）
  assertThrows(() => gatherParams(6, 0, 3), CodegenError);
  assertEquals([...gatherParams(0, 0, 3)], [0, 0, 3, GATHER_OOB_BITS]);
});

Deno.test("codegen は契約外の生成入力を fail loudly にする", () => {
  assertThrows(() => elementwiseWgsl({ op: "relu", rank: 0, dtype: "f32" }), CodegenError);
  // 語彙外 op は型で弾かれるので、実行時ガードを試すには広い型からの明示的な絞り込みが要る
  const foreignElementwise: { op: string; rank: number; dtype: string } = {
    op: "matmul",
    rank: 1,
    dtype: "f32",
  };
  assertThrows(() => elementwiseWgsl(foreignElementwise as ElementwiseSpec), CodegenError);
  // 契約表が解禁していない要素型（f32 専業の op に i32 / bool 専業の op に f32）
  assertThrows(() => elementwiseWgsl({ op: "relu", rank: 1, dtype: "i32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "div", rank: 1, dtype: "i32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "bitwise_not", rank: 1, dtype: "f32" }), CodegenError);
  assertThrows(() => elementwiseWgsl({ op: "bitwise_and", rank: 1, dtype: "f32" }), CodegenError);
  // MUST: where の受理集合の**和**は {bool, f32}。和で見ると条件用の bool が値スロットの
  // 生成入力として素通りするので、スロット単位で拒否できていることを固定する。
  assertThrows(() => elementwiseWgsl({ op: WHERE_OP, rank: 1, dtype: "bool" }), CodegenError);
  const foreignReduce: { op: string; dtype: string } = { op: "mean", dtype: "f32" };
  assertThrows(() => reduceWgsl(foreignReduce as ReduceSpec), CodegenError);
  // 行 reduce の dtype 解禁は op ごと（bool を数えられるのは sum だけ）
  assertThrows(() => reduceWgsl({ op: "amax", dtype: "bool" }), CodegenError);
  assertThrows(() => reduceWgsl({ op: "sum", dtype: "i32" }), CodegenError);
  const negR1: ElementwiseSpec = { op: "neg", rank: 1, dtype: "f32" };
  assertThrows(() => elementwiseParams(negR1, [], []), CodegenError);
  assertThrows(() => elementwiseParams(negR1, [4], [[2, 2]]), CodegenError);
  assertThrows(() => matmulParams(-1, 2, 3), CodegenError);
  assertThrows(() => reduceParams(1.5, 2), CodegenError);
  // 軸変種も同じ語彙・同じ dtype 規律（別 codegen へ複製した際の緩みを塞ぐ）
  assertThrows(() => axisReduceWgsl(foreignReduce as ReduceSpec), CodegenError);
  assertThrows(() => axisReduceWgsl({ op: "amax", dtype: "bool" }), CodegenError);
  assertThrows(() => axisReduceParams(1.5, 2, 3), CodegenError);
  assertThrows(() => axisReduceParams(4, 2, -1), CodegenError);
});
