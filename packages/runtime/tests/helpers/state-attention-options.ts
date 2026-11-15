import {
  REORDER_STRATEGIES,
  type SessionOptions,
  type StateAttentionReduce,
} from "../../src/runtime/session-types.ts";

/**
 * The Session options under which states-form attention runs the reduction form `reduce`: the
 * reference kernels at `exact`, the fused parallel reductions at `reorder`, and the parallel
 * reductions without the fold through the private `REORDER_STRATEGIES` (the derivation is
 * `executionKnobDefaults` in src/runtime/session-options-check.ts).
 */
export const stateAttentionReduceOptions = (reduce: StateAttentionReduce): SessionOptions =>
  reduce === "sequential"
    ? {}
    : {
        numerics: { ops: { attention: "reorder" } },
        ...(reduce === "parallel" ? { [REORDER_STRATEGIES]: { stateAttention: "parallel" } } : {}),
      };
