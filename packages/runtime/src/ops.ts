/**
 * The op contract table — per op, "arity / dtype rules / attrs schema / output shape computation"
 * in one place. codegen, the CPU reference and the executor all consult it.
 *
 * MUST: the runtime's capability declaration ({@link RUNTIME_SUPPORT}) is derived from this table
 * only. Matching op names alone misses dtype and attribute differences (an op that the support
 * table lists by name only fails at run time).
 * MUST: semantic dtypes are enabled per op (never all at once — the vocabulary allowlist of ADR 0007
 * is frozen). i32 / bool are allowed only in the forms that appeared in measured graphs and in the
 * ops created for those forms.
 *
 * Output shapes are computed on **the numeric shape after binding resolution**. Symbolic
 * dimensions are always resolved to concrete values before execution (static shapes — ADR 0004),
 * so a decision such as "can these broadcast" never has to be made on symbols.
 */

// The bodies are the 4 modules of src/ops/ (names = op names and bases / attrs = the attrs schema
// and accessors / contracts = the contract table and queries / shapes = shape computation). This
// surface only re-exports; the public names are the same as before the split. support.ts in the same
// directory (the runtime support table's types and the enumeration check) is outside this surface;
// runtime/executor.ts, runtime/estimate.ts and mod.ts import it directly.

export {
  ARGMAX_OP,
  ATTENTION_OP,
  BINARY_OPS,
  BMM_OP,
  CAST_OP,
  CAT_OP,
  CONV1D_OP,
  CONV2D_OP,
  CONV3D_OP,
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
  STATE_APPEND_OP,
  STATIC_QUANTIZE_OP,
  SYM_PREFIX_SLICE_OP,
  TOPK_OP,
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
  conv3dAttrs,
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
  stateReadonly,
  stateWindow,
  staticQuantizeScale,
  symPrefixSliceAttrs,
  topkK,
  upsampleBilinear2dAttrs,
} from "./ops/attrs.ts";
export type {
  AttrSchema,
  Conv1dAttrs,
  Conv2dAttrs,
  Conv3dAttrs,
  DeformConv2dAttrs,
  PadAttrs,
  PrefixSlice,
  SliceAttrs,
  UpsampleBilinear2dAttrs,
} from "./ops/attrs.ts";

export {
  arityFits,
  assertArity,
  assertAttrKeys,
  assertAttrValues,
  assertDtype,
  assertInputDtypes,
  assertNodeContract,
  assertSlotDtype,
  attrKeysOf,
  capabilities,
  describeArity,
  IO_DTYPES,
  OP_CONTRACTS,
  optionalAttrKeysOf,
  outputCountOf,
  outputDtypeOf,
  resolveNodeDtypes,
  resolveOpContract,
  RUNTIME_SUPPORT,
  scalarParamValues,
  slotDtypesOf,
} from "./ops/contracts.ts";
export type {
  OpContract,
  OpKind,
  RuntimeCapabilities,
  SlotDtypes,
  StateFieldContract,
} from "./ops/contracts.ts";

export { broadcastShapes, computeOutputShape, numel } from "./ops/shapes.ts";
export type { ShapeContext } from "./ops/shapes.ts";
