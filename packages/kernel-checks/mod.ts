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
export { checkGemvSubgroup, GEMV_SUBGROUP_CASES } from "./src/gemv-subgroup.ts";
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
  checkStateKv16,
  checkStateKv16Mutants,
  STATE_KV16_CASES,
  STATE_KV16_MUTANTS,
  type StateKv16Mutant,
} from "./src/state-kv16.ts";
