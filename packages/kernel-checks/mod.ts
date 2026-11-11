/**
 * The kernel identity checks shared by the runtime's node GPU tests and the playground's
 * `/kernel-check` page (one implementation, so the page runs exactly the cases the tests run).
 * Private: the checks drive runtime internals and are not part of any published surface.
 */

export {
  type CheckCase,
  type CheckOptions,
  type CheckPart,
  type CheckProgress,
  type Criterion,
  type ExpectedCases,
  type Requirement,
  subgroup32Available,
  type WordFailure,
} from "./src/result.ts";
export { checkFusionTwin, type FusionTwin } from "./src/fusion-twin.ts";
export {
  checkGatedResidualTwins,
  GATED_RESIDUAL_MASKS,
  GATED_RESIDUAL_SHAPES,
  GATED_RESIDUAL_TWIN_CASES,
  gatedResidualGraph,
  gatedResidualInputs,
} from "./src/fusion-twins/gated-residual.ts";
export {
  checkSiluTwins,
  SILU_ORDERS,
  SILU_SHAPE,
  SILU_TWIN_CASES,
  siluGraph,
  siluInput,
} from "./src/fusion-twins/silu.ts";
export {
  ADALN_GATES,
  ADALN_TOKEN_MASKS,
  ADALN_TOKEN_SHAPES,
  ADALN_TWIN_CASES,
  adalnGraph,
  type AdalnGraphOptions,
  adalnInputs,
  adalnTokenGraph,
  adalnTokenInputs,
  checkAdalnTwins,
} from "./src/fusion-twins/adaln.ts";
export {
  checkRopeTwins,
  ROPE_BSHD_SHAPES,
  ROPE_ORDERS,
  ROPE_PAIR_SHAPES,
  ROPE_TWIN_CASES,
  ropeBshdInputs,
  ropeGraph,
  ropeInputs,
  ropePairGraph,
  ropePairInputs,
  ropePairLabel,
} from "./src/fusion-twins/rope.ts";
export {
  checkUpsample2xTwins,
  UPSAMPLE2X_SHAPE,
  UPSAMPLE2X_TWIN_CASES,
  upsample2xGraph,
  upsample2xInput,
} from "./src/fusion-twins/upsample2x.ts";
export {
  checkRmsNormRopeTwins,
  RMS_NORM_ROPE_OUTPUTS,
  RMS_NORM_ROPE_SHAPES,
  RMS_NORM_ROPE_TWIN_CASES,
  rmsNormRopeInputs,
} from "./src/fusion-twins/rms-norm-rope.ts";
export { checkGemvSubgroup, GEMV_SUBGROUP_CASES } from "./src/gemv-subgroup.ts";
export { checkExpertLinear, EXPERT_LINEAR_CASES } from "./src/expert-linear.ts";
export {
  checkStateBidirectional,
  STATE_BIDIRECTIONAL_CASES,
  stateBidirectionalMutants,
  type StateBidirectionalMutant,
} from "./src/state-bidirectional.ts";
export { checkRmsSubgroup, RMS_SUBGROUP_CASES } from "./src/rms-subgroup.ts";
export {
  benchIntegerGemv,
  checkIntegerGemv,
  GEMV_INTEGER_CASES,
  type IntegerGemvBenchRow,
} from "./src/gemv-integer.ts";
export {
  benchPlanesGemv,
  checkPlanesGemv,
  GEMV_PLANES_CASES,
  GEMV_PLANES_F64_BAND,
  type PlanesGemvBenchRow,
  type PlanesGemvCheckOptions,
} from "./src/gemv-planes.ts";
export {
  checkStateAttentionMutantMatrix,
  checkStateAttentionMutants,
  checkStateAttentionRewrites,
  checkStatePvUnroll,
  checkStateQkFold,
  checkStateStatsPvUnroll,
  type FoldClass,
  type MutantMatrixRow,
  type MutantResult,
  STATE_ATTENTION_CASES,
  STATE_ATTENTION_MUTANTS,
  type StateAttentionRewritesReport,
} from "./src/state-attention-rewrites.ts";
export {
  checkScoreF16Store,
  checkScoreF16TowardZeroStore,
  SCORE_F16_CASES,
  scoreF16StoreMutants,
} from "./src/score-f16.ts";
export {
  checkStateKv16,
  checkStateKv16Mutants,
  STATE_KV16_CASES,
  STATE_KV16_MUTANTS,
  type StateKv16Mutant,
} from "./src/state-kv16.ts";
