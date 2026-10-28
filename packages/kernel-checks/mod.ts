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
