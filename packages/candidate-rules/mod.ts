/**
 * Candidate fusion rules: rules written outside the runtime against `@karume/runtime/kernel`, kept
 * here while they are measured (the session A/B) and checked. Private: nothing here is part of a
 * release, and a rule moves into `packages/models` with its evidence when it is adopted (README).
 */

export { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "./src/gelu-mul.ts";
export { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "./src/ew-chain.ts";
