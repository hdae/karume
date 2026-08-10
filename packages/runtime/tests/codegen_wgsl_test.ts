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
import { bmmKey, bmmParams, bmmWgsl } from "../src/kernels/bmm.ts";
import { GEMM_MTILE_SMALL, GEMM_TILE, gemmUsesVec4 } from "../src/kernels/gemm.ts";
import { conv1dKey, conv1dParams, conv1dWgsl } from "../src/kernels/conv1d.ts";
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
  layerNormParams,
} from "../src/kernels/layer-norm.ts";
import {
  ADALN_NORM_KEY,
  ADALN_NORM_WGSL,
  ADALN_NORM_WORKGROUP_SIZE,
  adalnNormParams,
} from "../src/kernels/adaln-norm.ts";
import { RMS_NORM_KEY, RMS_NORM_WGSL, rmsNormParams } from "../src/kernels/rms-norm.ts";
import { ROPE_KEY, ROPE_WGSL, ROPE_WORKGROUP_SIZE, ropeParams } from "../src/kernels/rope.ts";
import { linearKey, linearParams, linearWgsl } from "../src/kernels/linear.ts";
import {
  DP4A_WGSL_FEATURE,
  dp4aAvailable,
  LINEAR_ACT_SCALE_BINDING,
  LINEAR_I8A8_MAX_K,
  linearI8a8Key,
  linearI8a8Params,
  linearI8a8UsesVec4,
  linearI8a8Wgsl,
} from "../src/kernels/linear-i8a8.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsParams,
} from "../src/kernels/quantize-rows.ts";
import { WEIGHT_STORAGES } from "../src/kernels/weight-storage.ts";
import { MASKED_FILL_KEY, MASKED_FILL_WGSL, maskedFillParams } from "../src/kernels/masked-fill.ts";
import { matmulKey, matmulParams, matmulWgsl } from "../src/kernels/matmul.ts";
import { SOFTMAX_KEY, SOFTMAX_WGSL, softmaxParams } from "../src/kernels/softmax.ts";
import {
  ATTENTION_STATS_KEY,
  ATTENTION_STATS_WGSL,
  attentionPvKey,
  attentionPvParams,
  attentionPvWgsl,
  attentionQkKey,
  attentionQkParams,
  attentionQkWgsl,
  attentionStatsKey,
  attentionStatsParams,
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
 * GEMM 骨格の全生成入力（6 op × 重み格納 × v4）を機械的に回すための直積。
 * 融合 attention の ①③（ADR 0023）と conv2d の implicit GEMM（ADR 0024）も同じ骨格を
 * 共有するので、内積ループ・タイル辺・キーの検査が自動でこちらにも掛かる。
 * 4 番目の要素は accumulator の持ち方（`true` = `acc0..acc3` の静的展開 = linear だけ）。
 */
const GEMM_VARIANTS: readonly (readonly [string, string, string, boolean])[] = [false, true]
  .flatMap((
    v4,
  ) => [
    [`matmul${v4 ? " v4" : ""}`, matmulKey(v4), matmulWgsl(v4), false] as const,
    [`bmm${v4 ? " v4" : ""}`, bmmKey(v4), bmmWgsl(v4), false] as const,
    [`attention_qk${v4 ? " v4" : ""}`, attentionQkKey(v4), attentionQkWgsl(v4), false] as const,
    [`attention_pv${v4 ? " v4" : ""}`, attentionPvKey(v4), attentionPvWgsl(v4), false] as const,
    ...WEIGHT_STORAGES.flatMap((weight) => [
      [
        `linear ${weight}${v4 ? " v4" : ""}`,
        linearKey(weight, v4),
        linearWgsl(weight, v4),
        true,
      ] as const,
      [
        `conv2d igemm ${weight}${v4 ? " v4" : ""}`,
        conv2dIgemmKey(weight, v4),
        conv2dIgemmWgsl(weight, v4),
        false,
      ] as const,
    ]),
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
    // GEMM 3 op は形状由来の v4 フラグで 2 変種（スカラ / vec4）を持つ。**両側を並べる**のが
    // 条件で、片方だけ固定すると端数形状のフォールバックが黙って壊れても気づけない。
    ["matmul.wgsl", MATMUL_WGSL],
    ["matmul_v4.wgsl", matmulWgsl(true)],
    ["bmm.wgsl", BMM_WGSL],
    ["bmm_v4.wgsl", bmmWgsl(true)],
    // 融合 attention の 3 カーネル（ADR 0023）。①③ は GEMM 骨格の変種なので v4 と対で置く。
    ["attention_qk.wgsl", attentionQkWgsl(false)],
    ["attention_qk_v4.wgsl", attentionQkWgsl(true)],
    ["attention_pv.wgsl", attentionPvWgsl(false)],
    ["attention_pv_v4.wgsl", attentionPvWgsl(true)],
    ["attention_stats.wgsl", ATTENTION_STATS_WGSL],
    ["gather.wgsl", GATHER_WGSL],
    ["strided_copy_f32.wgsl", stridedWgsl({ dtype: "f32" })],
    ["strided_copy_i32.wgsl", stridedWgsl({ dtype: "i32" })],
    ["strided_copy_bool.wgsl", stridedWgsl({ dtype: "bool" })],
    // 書き族は cat 専用で、契約が f32 専業なので生成されうるのは f32 だけ（ADR 0014）
    ["strided_write_f32.wgsl", stridedWriteWgsl({ dtype: "f32" })],
    ["pad.wgsl", PAD_WGSL],
    ["upsample2x.wgsl", UPSAMPLE_2X_WGSL],
    ["silu_x_sigmoid.wgsl", siluWgsl("x-sigmoid")],
    ["silu_sigmoid_x.wgsl", siluWgsl("sigmoid-x")],
    ["flip.wgsl", FLIP_WGSL],
    ["rope.wgsl", ROPE_WGSL],
    ["linear.wgsl", LINEAR_WGSL],
    ["linear_v4.wgsl", linearWgsl("f32", true)],
    ["layer_norm.wgsl", LAYER_NORM_WGSL],
    // adaLN 融合（ADR 0040 の 4 本目）。**素の layer_norm と対で置く**のが条件で、
    // 行統計と affine の文字列を共有しているぶん、片方だけバイト列が動くのが最大の事故。
    ["adaln_norm.wgsl", ADALN_NORM_WGSL],
    ["rms_norm.wgsl", RMS_NORM_WGSL],
    ["softmax.wgsl", SOFTMAX_WGSL],
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
    // w8a8（活性 i8 化 + 整数内積）。**dp4a 版とエミュ版の両方**を置くのが条件で、
    // 「数値は同じで速度だけ違う」という主張は生成物が 2 つ別々に存在することが前提になる。
    ["quantize_rows.wgsl", QUANTIZE_ROWS_WGSL],
    ["linear_i8a8.wgsl", linearI8a8Wgsl(false, true)],
    ["linear_i8a8_v4.wgsl", linearI8a8Wgsl(true, true)],
    ["linear_i8a8_emu.wgsl", linearI8a8Wgsl(false, false)],
    ["linear_i8a8_emu_v4.wgsl", linearI8a8Wgsl(true, false)],
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
    }
    for (const dp4a of [false, true]) {
      assertEquals(
        linearI8a8Wgsl(v4, dp4a),
        linearI8a8Wgsl(v4, dp4a),
        `linear i8a8:v4=${v4}:dp4a=${dp4a}`,
      );
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
  assertEquals(QUANTIZE_ROWS_WGSL, QUANTIZE_ROWS_WGSL);
});

Deno.test("パイプラインキーは生成入力ごとに一意（別カーネルが同じキーに割り当たらない）", () => {
  const keys: string[] = [
    // GEMM 3 op × 重み格納 × v4 の直積（漏れを作らないために列挙を 1 本にまとめてある）
    ...GEMM_VARIANTS.map(([, key]) => key),
    GATHER_KEY,
    LAYER_NORM_KEY,
    RMS_NORM_KEY,
    SOFTMAX_KEY,
    ATTENTION_STATS_KEY,
    MASKED_FILL_KEY,
    CUMSUM_KEY,
    PAD_KEY,
    FLIP_KEY,
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
      [false, true].map((v4) => conv2dIgemmKey(weight, v4, GEMM_MTILE_SMALL))
    ),
    // w8a8: v4 × 整数内積変種の 4 本 + 活性量子化。**dp4a とエミュを別キーにする**のが条件で、
    // 同じキーに割り当たると診断でどちらが走ったか分からなくなる（設計 §4.4-5）。
    ...[false, true].flatMap((v4) => [linearI8a8Key(v4, false), linearI8a8Key(v4, true)]),
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

Deno.test("SiLU は sigmoid の f32 格納境界を u32 workgroup staging で保ち、mul 順をキーへ残す", () => {
  assertEquals(SILU_WORKGROUP_SIZE, 256);
  assertEquals(siluKey("x-sigmoid"), "silu:v1:x-sigmoid:f32:wg256");
  assertEquals(siluKey("sigmoid-x"), "silu:v1:sigmoid-x:f32:wg256");
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
  assertEquals(ADALN_NORM_KEY, "adaln_norm:v1:lastdim:f32:wg256");
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
  const params = elementwiseParams([2, 3], [[2, 3]], [-1.5, 2.25]);
  assertEquals(params.length, 1 + 2 + 2 + 2);
  assertEquals([...params.slice(0, 5)], [6, 2, 3, 3, 1]);
  const floats = new Float32Array(params.buffer);
  assertEquals(floats[5], -1.5);
  assertEquals(floats[6], 2.25);
  // 既定は 0 本（既存 op の params レイアウトは変わらない）
  assertEquals(elementwiseParams([2, 3], [[2, 3]]).length, 5);
  assertThrows(() => elementwiseParams([2], [[2]], [Number.NaN]), CodegenError);
});

Deno.test("elementwise params は右詰め broadcast の stride を 0 にする", () => {
  // 出力 [2,3,4] に対し in0=[2,3,4]（連続）、in1=[3,1]（右詰めで [1,3,1]）
  const params = elementwiseParams([2, 3, 4], [[2, 3, 4], [3, 1]]);
  assertEquals([...params.slice(0, 4)], [24, 2, 3, 4]);
  assertEquals([...params.slice(4, 7)], [12, 4, 1]);
  assertEquals([...params.slice(7, 10)], [0, 1, 0]);
});

Deno.test("elementwise params はスカラ入力を右詰めで吸収する", () => {
  const params = elementwiseParams([5], [[5], [1]]);
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
    // 3 バッファとも base 経由で触る（片方だけ素の添字に戻る誤りを塞ぐ）
    assertEquals(bmmWgsl(v4).includes("abase + arow *"), true, where);
    assertEquals(bmmWgsl(v4).includes("bbase + brow *"), true, where);
    assertEquals(bmmWgsl(v4).includes("cbase + orow *"), true, where);
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
    assertEquals(qk.includes("wv = wv * dims.scale;"), true, `${where}: k 側の scale`);
    assertEquals(
      qk.includes("raw.x * dims.scale") || qk.includes("q[arow_base + ak0] * dims.scale"),
      true,
      `${where}: q 側の scale`,
    );
    // scale は uniform の 4 語目（WGSL に焼かない — 値の種類でパイプラインが増える）
    assertEquals(qk.includes("  scale: f32,\n"), true, where);
    // k は [N,D] のまま読む（旧経路の permute(kᵀ) が消えることの生成物側の証拠）
    assertEquals(qk.includes("let krow_base = bbase + wcol * dims.k;"), true, where);
    // MUST: 転置読みの base は v4 でも**要素単位**（quad にすると 4 分の 1 の位置を読む）
    assertEquals(qk.includes("let bbase = wid.z * dims.n * dims.k;"), true, where);

    const pv = attentionPvWgsl(v4);
    // MUST: A タイルは P を実体化せず `exp(S − m) · inv` を**成分ごとのスカラ式**で評価する
    // （`vec4` へまとめて `exp` を掛けない — 超越関数のベクトル版は実装依存）。
    assertEquals(
      pv.split("- row_max) * row_inv").length - 1,
      4,
      `${where}: 正規化式は 4 成分ぶんちょうど`,
    );
    assertEquals(pv.includes("exp(vec4"), false, `${where}: exp をベクトルへまとめない`);
    // MUST: 行統計の添字はバッチ base 込み（B·H ≥ 2 で全バッチが 0 番の統計を使う誤りを塞ぐ）
    assertEquals(pv.includes("let rbase = wid.z * dims.m;"), true, where);
    assertEquals(
      pv.includes("let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);"),
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
  await Promise.resolve();
});

/**
 * GEMM 骨格のタイル定数。**沈黙誤値の代表格**が「WGSL 側 64 / dispatch 側 16」のずれなので、
 * 生成物に埋まった辺と TS 定数を同じテストで突き合わせる（executor は {@link GEMM_TILE} を
 * そのまま `tiledWorkgroups` へ渡す）。
 */
Deno.test("GEMM 骨格 6 op のタイル辺は 64 で、生成物・キー・TS 定数が一致する", () => {
  assertEquals(GEMM_TILE, 64);
  for (const [where, key, wgsl, staticAcc] of GEMM_VARIANTS) {
    // 出力タイルの原点（行 / 列）と workgroup は骨格を共有する全 op で共通
    assertEquals(wgsl.includes(`let arow = wid.y * ${GEMM_TILE}u + ar;`), true, where);
    assertEquals(wgsl.includes(`let orow0 = wid.y * ${GEMM_TILE}u + lid.y * 4u;`), true, where);
    assertEquals(wgsl.includes("@compute @workgroup_size(16, 16)"), true, where);
    // 1 スレッド 4×4 = レジスタ 16 本（linear だけ acc0..acc3 の静的展開）。
    // 共有は 4,096 B + 4,096 B
    assertEquals(
      wgsl.includes(staticAcc ? "var acc3 = vec4<f32>(0.0);" : "var acc = array<vec4<f32>, 4>("),
      true,
      where,
    );
    assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 1024>;"), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 256>;"), true, where);
    // MUST: 生成パラメータはキーに載る（タイル辺を変えれば別キーになる）。conv2d の
    // implicit GEMM だけは判別子が `igemm`（直接カーネルと系統が違うことを名前で表す）。
    const tilePart = where.startsWith("conv2d") ? "igemm" : "reg";
    assertEquals(key.includes(`${tilePart}${GEMM_TILE}x${GEMM_TILE}`), true, where);
    // 版番号は op ごと（既存 3 op は 16×16 からの改版で v2・融合 attention は新設なので v1）
    assertEquals(key.includes(where.startsWith("attention") ? ":v1:" : ":v2:"), true, where);
    // 旧 16×16 タイルの痕跡が残っていない（キーの改版と生成物の総取り替えは対）
    assertEquals(key.includes("tile16"), false, where);
  }
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg64x64v4");
  assertEquals(attentionPvKey(false), "attention_pv:v1:f32:reg64x64");
  // v4 判別子はキーに載る（形状から導いた 1 ビットがパイプラインを分ける）
  assertEquals(matmulKey(true), "matmul:v2:f32:reg64x64v4");
  assertEquals(matmulKey(false), "matmul:v2:f32:reg64x64");
  assertEquals(bmmKey(true), "bmm:v2:f32:reg64x64v4");
  assertEquals(linearKey("i8", true), "linear:v2:f32:reg64x64v4:wi8");
  assertEquals(linearKey("f16", false), "linear:v2:f32:reg64x64:wf16");
  // conv2d は 2 系統（implicit GEMM / 直接カーネル）で、直接側のキーは動かさない
  assertEquals(conv2dIgemmKey("f32", true), "conv2d:v2:f32:igemm64x64v4:wg16x16");
  assertEquals(conv2dIgemmKey("i8", false), "conv2d:v2:f32:igemm64x64:wg16x16:wi8");
  assertEquals(conv2dKey("f32"), "conv2d:v1:f32:direct:wg256");
});

/**
 * conv2d の **32 行 m タイル変種**（ADR 0024 隣接）。
 *
 * MUST: 変えるのは「どの workgroup がどの出力を担当するか」だけ。1 出力要素の K 縮約順
 * （K タイル 16 昇順）・丸め列・bias-first・0 埋めは 64 行版と共通の骨格が持つので、
 * ここでは**タイル幾何と担当割りだけ**を固定する（数値の一致は実 GPU のビット比較が見る —
 * tests/gpu_conv2d_parity_test.ts）。
 */
Deno.test("conv2d の 32 行 m タイル変種は幾何だけが変わる（n タイル 64・4×4 レジスタは不変）", () => {
  assertEquals(GEMM_MTILE_SMALL, 32);
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = conv2dIgemmWgsl(weight, v4, GEMM_MTILE_SMALL);
      const where = `conv2d igemm m32 ${weight} v4=${v4}`;
      // m タイルは 32 行（A タイルの行原点・store の行原点・bias-first の行原点の 3 箇所）
      assertEquals(wgsl.includes("let arow = wid.y * 32u + ar;"), true, where);
      assertEquals(wgsl.includes("let orow0 = wid.y * 32u + lid.y * 4u;"), true, where);
      assertEquals(wgsl.includes("let bias0 = wid.y * 32u + lid.y * 4u;"), true, where);
      // n タイルは 64 のまま（2048px の n 上限の扱いを動かさない）
      assertEquals(
        wgsl.includes(v4 ? "let bc4 = wid.x * 16u + bcq;" : "let bcol = wid.x * 64u + bcq * 4u;"),
        true,
        where,
      );
      // workgroup は 16×8 = 128 スレッド。**1 スレッド 4×4 は不変** = 内積の
      // 「共有ロード 5 回で 16 MAC」が落ちない（16×16 のまま 2×4 に落とす形は密度が下がる）
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 8)"), true, where);
      assertEquals(wgsl.includes("var acc = array<vec4<f32>, 4>("), true, where);
      // 共有 A は 32 行ぶんへ縮む（sb は n タイルが同じなので 256 のまま）
      assertEquals(wgsl.includes("var<workgroup> sa: array<f32, 512>;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f32>, 256>;"), true, where);
      // 128 スレッドで 256 要素の B タイルを埋めるので 2 パス。担当の k 行は 8 ずつずれる
      assertEquals(wgsl.includes("for (var bp = 0u; bp < 2u; bp = bp + 1u) {"), true, where);
      assertEquals(wgsl.includes("let bk = bp * 8u + bkr;"), true, where);
      // 内積ループの正本は骨格 1 箇所（64 行版と 1 文字も違わない）
      assertEquals(
        wgsl.includes("acc[i] = acc[i] + sa[(lid.y * 4u + i) * 16u + kk] * bv;"),
        true,
        where,
      );
    }
  }
  // 64 行版は 256 スレッド 1 パスのまま（2 パス化が漏れ出していないこと）
  assertEquals(conv2dIgemmWgsl("f32", true).includes("bp"), false);
  assertEquals(conv2dIgemmWgsl("f32", true).includes("let bk = tid / 16u;"), true);
  // キーは別系統（タイル形は生成パラメータなのでキーに載る）
  assertEquals(conv2dIgemmKey("f32", true, 32), "conv2d:v2:f32:igemm32x64v4:wg16x8");
  assertEquals(conv2dIgemmKey("i8", false, 32), "conv2d:v2:f32:igemm32x64:wg16x8:wi8");
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
    assertEquals(wgsl.includes("let bias0 = wid.y * 64u + lid.y * 4u;"), true, where);
    assertEquals(wgsl.includes("vec4<f32>(bias[bias0 + 1u]),"), true, where);
    assertEquals(wgsl.includes("acc[i] + biasv"), false, `${where}: store 側で bias を足している`);
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
    assertEquals(wgsl.includes("let ic = brow / khw;"), true, where);
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
  const i8Wgsl = conv2dIgemmWgsl("i8", true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[arow];"), true);
  assertEquals(i8Wgsl.includes("wscale[wcol]"), false, "linear の列 scale を持ってきている");
  // 束縛番号は直接カーネルと同じ（executor は 1 本の定数で両方を束ねる）
  assertEquals(
    i8Wgsl.includes(`@group(0) @binding(${CONV2D_SCALE_BINDING}) var<storage, read> wscale:`),
    true,
  );
  // A タイルは重み（weight-storage 経由）・B タイルは x の暗黙 gather
  assertEquals(i8Wgsl.includes("av = dequant4(arow_base + ak0, wscale_v);"), true);
  assertEquals(conv2dIgemmWgsl("f16", false).includes("av.x = dequant(abase);"), true);
  assertEquals(conv2dIgemmWgsl("f32", false).includes("av.x = w[abase];"), true);
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
  assertEquals(QUANTIZE_ROWS_KEY, "quantize_rows:v1:f32>i8:pertoken:wg256");
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
      // 内積は 4 行へ静的展開（4×4 出力 × 4 列 = 16 内積）。**行ごとにちょうど 1 回**で、
      // 行番号は codegen 時に確定する — accumulator に動的添字が残っていないことが条件。
      for (let i = 0; i < 4; i += 1) {
        const inner =
          `      acc${i} = acc${i} + vec4<i32>(idot(a${i}, b0), idot(a${i}, b1), idot(a${i}, b2), idot(a${i}, b3));`;
        assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} の内積が 1 箇所でない`);
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる
      // （Metal で顕著）。展開の目的そのものなので、生成物に 1 つも残っていないことを見る。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // 共有タイルは [pack][row] / [pack][col]（バンク衝突 2-way — プロトタイプからの組み替え）
      assertEquals(wgsl.includes("sa[ap * 64u + ar] = av;"), true, where);
      assertEquals(wgsl.includes("sb[wp * 64u + wc] = wv;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 256>;"), true, where);
      // K 端数は 0 埋め（dot4I8Packed(0, x) == 0 なので厳密）
      assertEquals(wgsl.includes("if (arow < dims.m && apack < k4) {"), true, where);
      assertEquals(wgsl.includes("if (wcol < dims.n && wpack < k4) {"), true, where);
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
            ? "out[orow0 * n4 + ocq] = fma(vec4<f32>(acc0), xscale[orow0] * ws, biasv);"
            : "out[obase + ocol] = fma(f32(acc0.x), xs * wscale[ocol], bias[ocol]);",
        ),
        true,
        `${where}: dequant の乗算順序`,
      );
      // 素の `a * b + c` が残っていない（融合するかどうかがドライバ依存になる形）
      assertEquals(wgsl.includes(") + biasv;"), false, `${where}: 素の積和が残っている`);
      // タイル幾何は f32 骨格と同じ（gemm.ts の定数を輸入している）
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 16)"), true, where);
      assertEquals(wgsl.includes(`let orow0 = wid.y * ${GEMM_TILE}u + lid.y * 4u;`), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
    }
  }
  // キーは診断でどの変種が走ったか分かる形（ADR 0021）
  assertEquals(linearI8a8Key(true, true), "linear:v3:i8a8:reg64x64v4:dp4a");
  assertEquals(linearI8a8Key(false, false), "linear:v3:i8a8:reg64x64:dp4aEmu");
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
      // 内積は 4 行へ静的展開（linear の i8a8 と同じ展開）。**行ごとにちょうど 1 回**で、
      // 行番号は codegen 時に確定する — accumulator に動的添字が残っていないことが条件。
      for (let i = 0; i < 4; i += 1) {
        const inner =
          `      acc${i} = acc${i} + vec4<i32>(idot(a${i}, b0), idot(a${i}, b1), idot(a${i}, b2), idot(a${i}, b3));`;
        assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} の内積が 1 箇所でない`);
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる。
      // 展開の目的そのものなので、生成物に 1 つも残っていないことを見る。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // MUST: 半スケールは**量子化の前ではなく dequant 側**で q / k の両方へ。
      // 片側化・量子化前への移動はどちらも例外なしの誤値になる。
      assertEquals(
        wgsl.includes(
          v4
            ? "s[sbase + orow0 * n4 + ocq] = vec4<f32>(acc0) * ((qscale[qsbase + orow0] * dims.scale) * ks);"
            : "s[obase + ocol] = f32(acc0.x) * (qs * (kscale[ksbase + ocol] * dims.scale));",
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
      assertEquals(wgsl.includes("let krow_base = kbase + wcol * k4;"), true, where);
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
      // タイル幾何は f32 骨格・linear の i8a8 と同じ（gemm.ts の定数を輸入している）
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 16)"), true, where);
      assertEquals(wgsl.includes(`let orow0 = wid.y * ${GEMM_TILE}u + lid.y * 4u;`), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 256>;"), true, where);
    }
  }
  // キーは linear:v3:i8a8 の前例に倣う（世代 ++ / dtype 欄 i8a8 / 内積変種は末尾）
  assertEquals(attentionQkI8a8Key(true, true), "attention_qk:v2:i8a8:reg64x64v4:dp4a");
  assertEquals(attentionQkI8a8Key(false, false), "attention_qk:v2:i8a8:reg64x64:dp4aEmu");
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
      // 内積は 4 行へ静的展開（①QK / linear の i8a8 と同じ展開）。**行ごとにちょうど 1 回**で、
      // accumulator に動的添字が残っていないことが条件（下の `acc[` 検査が対）。
      for (let i = 0; i < 4; i += 1) {
        const inner =
          `      acc${i} = acc${i} + vec4<i32>(idot(a${i}, b0), idot(a${i}, b1), idot(a${i}, b2), idot(a${i}, b3));`;
        assertEquals(wgsl.split(inner).length, 2, `${where}: 行 ${i} の内積が 1 箇所でない`);
      }
      // MUST: `acc[...]` の動的添字はアドレス可能なローカル領域を要求し、レジスタから落ちる。
      assertEquals(wgsl.includes("acc["), false, `${where}: accumulator の動的添字が残っている`);
      // MUST: P̃ は**成分ごとのスカラ式**でちょうど 4 本（vec4 へまとめて exp を掛けない）
      assertEquals(
        wgsl.split("round(exp(raw.").length - 1,
        4,
        `${where}: P̃ の量子化式は 4 成分ぶんちょうど`,
      );
      assertEquals(wgsl.includes("exp(vec4"), false, `${where}: exp をベクトルへまとめない`);
      // MUST: 格子は 127（128 化は分母型の退行）。scale は 1/127 の**乗算**で作る（除算禁止）
      assertEquals(wgsl.includes("- row_max) * 127.0"), true, `${where}: P̃ の格子`);
      assertEquals(
        wgsl.includes("stats[(rbase + orow0) * 2u + 1u] * 0.007874015748031496"),
        true,
        `${where}: prow = inv·(1/127) の乗算`,
      );
      assertEquals(wgsl.includes("/ 127"), false, `${where}: 1/127 を除算で作っている`);
      // MUST: A タイルの範囲外は 0 のまま（exp(0−m)·127 を掛けると端数タイルが静かに誤る）
      assertEquals(
        wgsl.includes("    var av = 0u;\n    if (arow < dims.m && apack < k4) {"),
        true,
        `${where}: 範囲外 0 埋めの門`,
      );
      // MUST: dequant は prow·vs を先に 1 つの f32 へ畳む（bias が無いので fma は無い）
      assertEquals(
        wgsl.includes(
          v4
            ? "o[obase + orow0 * n4 + ocq] = vec4<f32>(acc0) * (prow * vs);"
            : "o[orow_base + ocol] = f32(acc0.x) * (prow * vscale[vsbase + ocol]);",
        ),
        true,
        `${where}: dequant の乗算順序`,
      );
      assertEquals(wgsl.includes("fma("), false, `${where}: fma が混ざっている`);
      // MUST: 行統計は行 max が `arow`（A タイル充填）・行 inv が `orow`（書き出し）。
      // 片方をもう片方へ流用すると担当の違うスレッドの行が乗る（B·H = M = 1 でしか一致しない）
      assertEquals(
        wgsl.includes("let stat_at = select(0u, (rbase + arow) * 2u, arow < dims.m);"),
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
      assertEquals(wgsl.includes("let vrow_base = vbase + wcol * k4;"), true, where);
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
      // タイル幾何は f32 骨格・linear / ①QK の i8a8 と同じ
      assertEquals(wgsl.includes("@compute @workgroup_size(16, 16)"), true, where);
      assertEquals(wgsl.includes(`let orow0 = wid.y * ${GEMM_TILE}u + lid.y * 4u;`), true, where);
      assertEquals(wgsl.includes("let tiles = (k4 + 3u) / 4u;"), true, where);
      assertEquals(wgsl.includes("var<workgroup> sa: array<u32, 256>;"), true, where);
      // ①QK と別物（骨格は共有だが充填も dequant も違う）
      assertNotEquals(wgsl, attentionQkI8a8Wgsl(v4, dp4a), where);
    }
  }
  // キーは ①QK と同じ規約（世代 ++ / dtype 欄 i8a8 / 内積変種は末尾）
  assertEquals(attentionPvI8a8Key(true, true), "attention_pv:v2:i8a8:reg64x64v4:dp4a");
  assertEquals(attentionPvI8a8Key(false, false), "attention_pv:v2:i8a8:reg64x64:dp4aEmu");
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

// 裁定（src/kernels/gather.ts）: 範囲外添字は別要素を返さず NaN で汚染する。
Deno.test("gather は範囲外添字を NaN で汚染し、通常経路は行オフセットで読む", () => {
  assertEquals(GATHER_WGSL.includes("if (pick < 0 || u32(pick) >= dims.src_cols) {"), true);
  assertEquals(GATHER_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
  assertEquals(GATHER_WGSL.includes("out[i] = src[row * dims.src_cols + u32(pick)];"), true);
  // 出力は連続なので行は平坦添字から割る（grid-stride 前提）
  assertEquals(GATHER_WGSL.includes("let row = i / dims.cols;"), true);
  assertEquals(GATHER_WGSL.includes("var i = gid.x;"), true);
});

// ADR 0012 の融合 op。カーネル固有の不変条件を生成物の側から固定する。
Deno.test("融合カーネルは既存カーネルと別物で、契約どおりの形を生成する", () => {
  // linear は重みを [n,k] の転置レイアウトのまま読む（連続方向が k なので k 連続 4 要素）
  assertEquals(LINEAR_WGSL.includes("let wrow_base = wcol * dims.k;"), true);
  assertEquals(LINEAR_WGSL.includes("wv.x = w[wbase];"), true);
  // MUST: 共有メモリ側で転置して置く（列 quad = wc / 4・成分 = wc % 4）。2 つを取り違えると
  // 1 タイル内で列が入れ替わる — 列ごとに値が違う端数形状だけが検出器になる。
  assertEquals(LINEAR_WGSL.includes("let wsq = wc / 4u;"), true);
  assertEquals(LINEAR_WGSL.includes("let wsl = wc % 4u;"), true);
  // MUST: 成分は**静的**に書く（`sb[i][wsl] = v` の動的インデックスにしない）。Metal では
  // wsl != 0 の書き込みが黙って捨てられ、4 要素中 3 要素が 0 のまま内積へ入る（機序は
  // src/kernels/gemm.ts の storeBTransposed）。4 アームぶんの転置配置を全て固定する —
  // 1 アームでも取り違えると 1 タイル内で列が入れ替わる。
  assertEquals(LINEAR_WGSL.includes("[wsl]"), false);
  const components = ["x", "y", "z", "w"] as const;
  for (const [at, component] of components.entries()) {
    assertEquals(
      LINEAR_WGSL.includes(
        `${at === components.length - 1 ? "default" : `case ${at}u`}: {\n` +
          `        sb[sb_base].${component} = wv.x;\n` +
          `        sb[sb_base + 16u].${component} = wv.y;\n` +
          `        sb[sb_base + 32u].${component} = wv.z;\n` +
          `        sb[sb_base + 48u].${component} = wv.w;\n` +
          `      }`,
      ),
      true,
      `linear の B タイル転置配置（成分 ${component}）`,
    );
  }
  // 末尾で bias を 1 度だけ足す（accumulator の初期値にはしない — 縮約順序を保つため）。
  // linear の acc は acc0..acc3 の静的展開なので、行ごとに同じ形が 4 本並ぶ。
  assertEquals(LINEAR_WGSL.includes("out[obase + ocol] = acc0.x + bias[ocol];"), true);
  assertEquals(
    linearWgsl("f32", true).includes(
      "let biasv = vec4<f32>(bias[oc], bias[oc + 1u], bias[oc + 2u], bias[oc + 3u]);",
    ),
    true,
  );
  assertEquals(linearWgsl("f32", true).includes("out[orow0 * n4 + ocq] = acc0 + biasv;"), true);
  // MUST: matmul / bmm の生成物に融合 op の概念が漏れていない
  assertEquals(MATMUL_WGSL.includes("bias"), false);
  assertEquals(BMM_WGSL.includes("bias"), false);
  assertNotEquals(LINEAR_WGSL, MATMUL_WGSL);
  // MUST: 3 op は同じ内積ループ（= 同じ縮約順序）を共有する。1 箇所しか無いことを
  // 生成物の側から固定する（写し間違いが起きる余地を残さない）。
  // linear は `acc[i]` を acc0..acc3 へ展開した形（添字と行番号が静的になるだけで、
  // 読む共有タイルの位置も加算順序も同一）— 行ごとにちょうど 1 本ずつ。
  const innerLoop = "        acc[i] = acc[i] + sa[(lid.y * 4u + i) * 16u + kk] * bv;";
  const staticLoop = (i: number): string =>
    `      acc${i} = acc${i} + sa[(lid.y * 4u + ${i}u) * 16u + kk] * bv;`;
  for (const [where, , wgsl, staticAcc] of GEMM_VARIANTS) {
    const lines = staticAcc ? [0, 1, 2, 3].map(staticLoop) : [innerLoop];
    for (const line of lines) {
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
  // MUST: layer_norm の生成物に rms_norm の概念が漏れていない（決定性スナップショットの前提）
  assertEquals(LAYER_NORM_WGSL.includes("inverseSqrt"), false);

  // MUST: softmax は safe-softmax（amax 減算）。素朴形は大入力で underflow / overflow する
  assertEquals(SOFTMAX_WGSL.includes("let amax = scratch[0u];"), true);
  assertEquals(SOFTMAX_WGSL.includes("acc = acc + exp(x[base + j] - amax);"), true);
  assertEquals(SOFTMAX_WGSL.includes("out[base + o] = exp(x[base + o] - amax) * inv;"), true);

  // embedding は範囲外添字を NaN で汚染する（gather と同じ裁定）
  assertEquals(EMBEDDING_WGSL.includes("if (pick < 0 || u32(pick) >= dims.vocab) {"), true);
  assertEquals(EMBEDDING_WGSL.includes("out[i] = bitcast<f32>(dims.oob);"), true);
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

  // conv_transpose1d は gather 形（出力 1 要素 = 寄与する入力の総和 — full-write）
  assertEquals(
    CONV_TRANSPOSE1D_WGSL.includes("if (t >= 0 && u32(t) % dims.stride == 0u) {"),
    true,
  );
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
      rewound = rewound.replace(`dequant(${index})`, `${name}[${index}]`);
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
  assertEquals(f32Wgsl.includes("wv = w[(wrow_base + wk0) >> 2u];"), true);

  const f16Wgsl = linearWgsl("f16", true);
  // f16 は u32 2 語で 4 要素（語をまたがない）。スカラ版の dequant は出さない
  assertEquals(f16Wgsl.includes("let lo = unpack2x16float(w[i >> 1u]);"), true);
  assertEquals(f16Wgsl.includes("let hi = unpack2x16float(w[(i >> 1u) + 1u]);"), true);
  assertEquals(f16Wgsl.includes("return vec4<f32>(lo.x, lo.y, hi.x, hi.y);"), true);
  assertEquals(f16Wgsl.includes("fn dequant(i: u32)"), false, "スカラ版は出さない");
  assertEquals(f16Wgsl.includes("wv = dequant4(wrow_base + wk0);"), true);

  const i8Wgsl = linearWgsl("i8", true);
  // MUST: scale は**成分ごとの f32 乗算**（vec4 * scalar）— スカラ経路と同一の丸め（ADR 0019）。
  // 縮約の外へ出す形（acc * scale）は ADR 0019 の改訂と tolerance 再導出を発火させる。
  assertEquals(i8Wgsl.includes("return vec4<f32>(unpack4xI8(w[i >> 2u])) * scale;"), true);
  assertEquals(i8Wgsl.includes("let wscale_v = wscale[wcol];"), true, "scale は K ループの外");
  assertEquals(i8Wgsl.includes("wv = dequant4(wrow_base + wk0, wscale_v);"), true);
  // linear の acc は acc0..acc3 の静的展開なので、縮約後に掛ける形はこの名前で見る
  assertEquals(i8Wgsl.includes("acc0 * "), false, "縮約の外で scale を掛けていない");

  // 逆写像で f32 v4 変種そのものに戻る（他は一切動いていない）
  for (
    const [wgsl, call] of [[f16Wgsl, "dequant4(wrow_base + wk0)"], [
      i8Wgsl,
      "dequant4(wrow_base + wk0, wscale_v)",
    ]] as const
  ) {
    const rewound = wgsl
      .replace(/\n@group\(0\) @binding\(5\) var<storage, read> wscale: array<f32>;\n/, "")
      .replace(/\/\/ (f16|i8) 格納の quad 展開:[\s\S]*?\n}\n\n/, "")
      .replace(
        /\n {2}\/\/ 出力チャネルの scale はループ不変[\s\S]*?\n {2}let wscale_v = wscale\[wcol\];/,
        "",
      )
      .replace("read> w: array<u32>;", "read> w: array<vec4<f32>>;")
      .replace(call, "w[(wrow_base + wk0) >> 2u]");
    assertEquals(dropHeader(rewound), dropHeader(f32Wgsl), call);
  }
});

/**
 * f16 **計算**変種（ADR 0028）。生成物の側から丸め列を固定する — 数値契約
 * 「f16 変種 = 入力を f16 に丸めた f32 変種」はこの 4 点だけで成立している:
 *
 * 1. 丸めは**共有タイルへ書く 1 箇所**（`f16(...)` / `vec4<f16>(...)`）。
 * 2. 拡幅は**レジスタロード時に 1 回**（MAC ごとの `f32(av * bv)` は禁止 — プロトタイプの
 *    負の教訓で、変換回数が倍になるうえ積が f16 精度に落ちる）。
 * 3. 累積は f32（`vec4<f32>` × 4 — linear だけ `acc0..acc3` の静的展開で、要素型は同じ）。
 * 4. uniform は f16 にしない。
 *
 * 3 番目の要素は accumulator の持ち方（`true` = 静的展開）。
 */
const COMPUTE_F16_VARIANTS: readonly (readonly [string, string, boolean])[] = [
  ["attention_qk c16", attentionQkWgsl(false, "f16"), false],
  ["attention_qk c16 v4", attentionQkWgsl(true, "f16"), false],
  ["attention_pv c16", attentionPvWgsl(false, "f16"), false],
  ["attention_pv c16 v4", attentionPvWgsl(true, "f16"), false],
  ["linear c16", linearWgsl("f32", false, "f16"), true],
  ["linear c16 v4", linearWgsl("f32", true, "f16"), true],
  ["linear wf16 c16", linearWgsl("f16", false, "f16"), true],
  ["linear wf16 c16 v4", linearWgsl("f16", true, "f16"), true],
];

Deno.test("f16 計算変種は共有タイルだけを f16 にし、丸めと拡幅の位置を 1 箇所に閉じる", () => {
  for (const [where, wgsl, staticAcc] of COMPUTE_F16_VARIANTS) {
    const lines = wgsl.split("\n");
    // enable はモジュール先頭（コメント 1 行の直後 = 全ての global 宣言より前）
    assertEquals(lines[0].startsWith("// karume "), true, where);
    assertEquals(lines[1], "enable f16;", where);
    // ① 共有タイルだけが f16（バイト半減がこの変種の唯一の機序）
    assertEquals(wgsl.includes(`var<workgroup> sa: array<f16, ${GEMM_TILE * 16}>;`), true, where);
    assertEquals(wgsl.includes("var<workgroup> sb: array<vec4<f16>, 256>;"), true, where);
    // ③ 累積は f32 のまま（k 大の桁落ちを避ける — 変えると parity と E2E が赤くなる）
    assertEquals(
      wgsl.includes(staticAcc ? "var acc0 = vec4<f32>(0.0);" : "var acc = array<vec4<f32>, 4>("),
      true,
      where,
    );
    // ② 拡幅はレジスタロード時に 1 回。MAC ごとの積の変換（f32(av * bv)）は出さない
    assertEquals(wgsl.includes("let bv = vec4<f32>(sb[kk * 16u + lid.x]);"), true, where);
    assertEquals(
      wgsl.includes(
        staticAcc
          ? "acc0 = acc0 + f32(sa[(lid.y * 4u + 0u) * 16u + kk]) * bv;"
          : "acc[i] = acc[i] + f32(sa[(lid.y * 4u + i) * 16u + kk]) * bv;",
      ),
      true,
      where,
    );
    // MUST: 判定はコメントを除いた本体で行う（禁じている形をコメントに書いてあるため）
    const code = wgsl.split("\n").filter((line) => !line.trimStart().startsWith("//")).join("\n");
    assertEquals(code.includes("f32(av"), false, `${where}: MAC ごとの変換`);
    // ① 丸めは共有タイルへの書き込みだけ（A タイルは 4 成分とも f16(...) で包む）
    assertEquals(wgsl.includes("sa[sa_base] = f16(av.x);"), true, where);
    assertEquals(wgsl.includes("sa[sa_base + 3u] = f16(av.w);"), true, where);
    // ④ uniform は f16 にしない（uniform の配列要素は 16B 整列で、f16 にする利得も無い）
    assertEquals(wgsl.includes("var<uniform> dims: Dims;"), true, where);
    assertEquals(/^\s+[mnk]: f16,$/m.test(wgsl), false, `${where}: uniform が f16`);
  }
  // 融合 attention の S は f16 で受け渡す（① が書き ②③ が読む = transient 半減）
  assertEquals(
    attentionQkWgsl(true, "f16").includes("s[cbase + orow * n4 + ocq] = vec4<f16>(acc[i]);"),
    true,
  );
  assertEquals(attentionQkWgsl(false, "f16").includes("s[obase + ocol] = f16(acc[i].x);"), true);
  assertEquals(
    attentionPvWgsl(false, "f16").includes(
      "av.x = exp(f32(s[arow_base + ak0]) - row_max) * row_inv;",
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
  // 既存の f32 計算キーはバイト単位で不変
  assertEquals(linearKey("f32", false), "linear:v2:f32:reg64x64");
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg64x64v4:wf16");
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg64x64v4");
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  // f16 計算は末尾に :c16（格納 :wf16 と重ならない語 — 両方同時に立つ組み合わせがある）
  assertEquals(linearKey("f32", false, "f16"), "linear:v2:f32:reg64x64:c16");
  assertEquals(linearKey("f16", true, "f16"), "linear:v2:f32:reg64x64v4:wf16:c16");
  assertEquals(attentionQkKey(false, "f16"), "attention_qk:v1:f32:reg64x64:c16");
  assertEquals(attentionPvKey(true, "f16"), "attention_pv:v1:f32:reg64x64v4:c16");
  assertEquals(attentionStatsKey("f16"), `${ATTENTION_STATS_KEY}:c16`);
});

// `acc[i]` の動的添字はアドレス可能な関数ローカル領域を要求し、accumulator がレジスタから
// ローカルメモリへ落ちる。linear（f32 / f16 計算 × 3 格納）は既定経路がこの静的展開になる。
Deno.test("linear の accumulator は acc0..acc3 の静的展開で、他の GEMM op は配列のまま", () => {
  for (const weight of WEIGHT_STORAGES) {
    for (const v4 of [false, true]) {
      const wgsl = linearWgsl(weight, v4);
      const where = `linear ${weight} v4=${v4}`;
      assertEquals(wgsl.includes("var acc = array<"), false, where);
      // MUST: 動的添字が 1 つも残っていない（残ると展開の動機がそのまま消える）
      assertEquals(wgsl.includes("acc["), false, where);
      for (let i = 0; i < 4; i += 1) {
        assertEquals(wgsl.includes(`var acc${i} = vec4<f32>(0.0);`), true, `${where}:init${i}`);
        // 行 i の内積更新はちょうど 1 回（K タイル 16・kk 昇順・加算順序は展開前と同一）
        assertEquals(
          wgsl.split(`acc${i} = acc${i} + sa[(lid.y * 4u + ${i}u) * 16u + kk] * bv;`).length,
          2,
          `${where}:update${i}`,
        );
        // 書き出しも行ごとに静的な名前を読む
        assertEquals(wgsl.includes(`orow${i} < dims.m`), true, `${where}:store${i}`);
      }
    }
  }
  // MUST: 静的展開は linear 限定。他 op へ広げると固定済みの生成バイト列が動く
  for (
    const [where, wgsl] of [
      ["matmul", matmulWgsl(true)],
      ["bmm", bmmWgsl(true)],
      ["attention_qk", attentionQkWgsl(true)],
      ["attention_pv", attentionPvWgsl(true)],
      ["conv2d", conv2dIgemmWgsl("f32", true, GEMM_TILE)],
    ] as const
  ) {
    assertEquals(wgsl.includes("var acc = array<vec4<f32>, 4>("), true, where);
    assertEquals(wgsl.includes("acc[i] = acc[i] + "), true, where);
  }
  // キーとシグネチャは据え置き（変種ではなく既定経路の置き換え）
  assertEquals(linearKey("f32", false), "linear:v2:f32:reg64x64");
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
    attentionPvWgsl(true, "f32", "f16").includes("let raw = score_quad(arow_base + t * 4u + aq);"),
    true,
  );
  assertEquals(
    attentionPvI8a8Wgsl(true, true, "f16").includes("let raw = score_quad(arow_base + apack);"),
    true,
  );
  assertEquals(
    attentionStatsWgsl("f32", "f16").includes("hi = max(hi, score_at(base + i));"),
    true,
  );
  // 書き手は f32 変種と同じ値式のまま（丸めが値の計算へ紛れ込んでいない）
  assertEquals(
    attentionQkWgsl(true, "f32", "f16").includes("score_store(cbase + orow * n4 + ocq, acc[i]);"),
    true,
  );
});

Deno.test("格納 S の判別子 s16 はキーの末尾に付き、wf16 / c16 と別語になる", () => {
  // 既定（f32 格納）のキーは 1 文字も動かない
  assertEquals(attentionQkKey(true), "attention_qk:v1:f32:reg64x64v4");
  assertEquals(attentionPvKey(true), "attention_pv:v1:f32:reg64x64v4");
  assertEquals(attentionStatsKey(), ATTENTION_STATS_KEY);
  assertEquals(attentionQkI8a8Key(true, true), "attention_qk:v2:i8a8:reg64x64v4:dp4a");
  assertEquals(attentionPvI8a8Key(true, false), "attention_pv:v2:i8a8:reg64x64v4:dp4aEmu");
  // s16 は**末尾**（3 つの軸が同時に立ちうるので語順を 1 箇所で固定する）
  assertEquals(attentionQkKey(true, "f32", "f16"), "attention_qk:v1:f32:reg64x64v4:s16");
  assertEquals(attentionPvKey(true, "f32", "f16"), "attention_pv:v1:f32:reg64x64v4:s16");
  assertEquals(attentionStatsKey("f32", "f16"), `${ATTENTION_STATS_KEY}:s16`);
  assertEquals(
    attentionQkI8a8Key(true, true, "f16"),
    "attention_qk:v2:i8a8:reg64x64v4:dp4a:s16",
  );
  assertEquals(
    attentionPvI8a8Key(true, false, "f16"),
    "attention_pv:v2:i8a8:reg64x64v4:dp4aEmu:s16",
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
    ["linear:v2:f32:reg64x64", (weight) => linearKey(weight, false)],
    ["embedding:v1:f32:i32:wg256", embeddingKey],
    ["conv1d:v2:f32:direct:wg256", conv1dKey],
    ["conv2d:v1:f32:direct:wg256", conv2dKey],
    ["conv_transpose1d:v1:f32:gather:wg256", convTranspose1dKey],
  ];
  for (const [expected, key] of pairs) {
    // MUST: f32 のキー文字列は変種導入の前後で完全に同じ（実キーを直書きして固定する）
    assertEquals(key("f32"), expected);
    assertEquals(key("f16"), `${expected}:wf16`);
  }
  // 格納判別子は v4 判別子の**後ろ**に付く（linear だけ 2 軸を持つ）
  assertEquals(linearKey("f16", true), "linear:v2:f32:reg64x64v4:wf16");
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
  assertThrows(() => elementwiseParams([], []), CodegenError);
  assertThrows(() => elementwiseParams([4], [[2, 2]]), CodegenError);
  assertThrows(() => matmulParams(-1, 2, 3), CodegenError);
  assertThrows(() => reduceParams(1.5, 2), CodegenError);
  // 軸変種も同じ語彙・同じ dtype 規律（別 codegen へ複製した際の緩みを塞ぐ）
  assertThrows(() => axisReduceWgsl(foreignReduce as ReduceSpec), CodegenError);
  assertThrows(() => axisReduceWgsl({ op: "amax", dtype: "bool" }), CodegenError);
  assertThrows(() => axisReduceParams(1.5, 2, 3), CodegenError);
  assertThrows(() => axisReduceParams(4, 2, -1), CodegenError);
});
