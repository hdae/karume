/**
 * op 契約テーブル — op ごとに「アリティ / dtype 規則 / attrs スキーマ / 出力 shape 計算」を
 * 1 箇所に持つ。codegen・CPU 参照・executor は全てここを参照する。
 *
 * MUST: ランタイムの capability 宣言（{@link RUNTIME_SUPPORT}）はこの表からのみ導出する。
 * op 名だけの突合は dtype と属性の差を見逃す（対応表に名前だけある op が実行時に落ちる）。
 * MUST: 意味論 dtype の解禁は op ごとに行う（一括解禁しない — ADR 0007 の語彙 allowlist
 * 凍結）。i32 / bool を許すのは実測グラフに現れた形と、その形のために新設した op だけ。
 *
 * 出力 shape の計算は**束縛解決後の数値 shape**に対して行う。記号次元は実行前に必ず具体値へ
 * 解決される（静的形状 — ADR 0004）ため、broadcast 可否のような判定を記号のまま扱う必要が無い。
 */

// 実体は src/ops/ の 4 モジュール（names = op 名と基底 / attrs = attrs スキーマとアクセサ /
// contracts = 契約表と照会 / shapes = shape 計算）。この面は再輸出のみで、公開名は分割前と同一。

export {
  ATTENTION_OP,
  BINARY_OPS,
  BMM_OP,
  CAST_OP,
  CAT_OP,
  CONV1D_OP,
  CONV2D_OP,
  CONV_TRANSPOSE1D_OP,
  CUMSUM_OP,
  DEFORM_CONV2D_OP,
  EMBEDDING_OP,
  EXPAND_OP,
  FLIP_OP,
  GATHER_OP,
  GRU_SCAN_OPS,
  LAYER_NORM_OP,
  LINEAR_OP,
  MASKED_FILL_OP,
  OpContractError,
  PAD_OP,
  PERMUTE_OP,
  REDUCE_OPS,
  RESHAPE_OP,
  RMS_NORM_OP,
  SAFE_SOFTMAX_OP,
  SLICE_OP,
  SOFTMAX_OP,
  SYM_PREFIX_SLICE_OP,
  UNARY_OPS,
  UPSAMPLE_BILINEAR2D_OP,
  WEIGHT_CHANNEL_AXES,
  WEIGHT_SLOTS,
  WHERE_OP,
} from "./ops/names.ts";
export type { BinaryOpName, GruScanOpName, ReduceOpName, UnaryOpName } from "./ops/names.ts";

export {
  attentionScale,
  castTargetDtype,
  catDim,
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  cumsumDim,
  deformConv2dAttrs,
  flipDim,
  layerNormAttrs,
  maskedFillValue,
  padAttrs,
  permuteDims,
  reduceDim,
  rmsNormEps,
  SCALAR_PARAM_ATTRS,
  scalarParamCount,
  sliceAttrs,
  softmaxDim,
  symPrefixSliceAttrs,
  upsampleBilinear2dAttrs,
} from "./ops/attrs.ts";
export type {
  AttrSchema,
  Conv1dAttrs,
  Conv2dAttrs,
  DeformConv2dAttrs,
  PadAttrs,
  PrefixSlice,
  SliceAttrs,
  UpsampleBilinear2dAttrs,
} from "./ops/attrs.ts";

export {
  arityFits,
  assertArity,
  assertDtype,
  assertNodeContract,
  assertSlotDtype,
  attrKeysOf,
  capabilities,
  describeArity,
  IO_DTYPES,
  OP_CONTRACTS,
  outputCountOf,
  outputDtypeOf,
  resolveNodeDtypes,
  resolveOpContract,
  RUNTIME_SUPPORT,
  scalarParamValues,
  slotDtypesOf,
} from "./ops/contracts.ts";
export type { OpContract, OpKind, RuntimeCapabilities, SlotDtypes } from "./ops/contracts.ts";

export { broadcastShapes, computeOutputShape, numel } from "./ops/shapes.ts";
export type { ShapeContext } from "./ops/shapes.ts";
