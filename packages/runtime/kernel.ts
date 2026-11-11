/**
 * Writing a fusion rule outside the runtime (`@karume/runtime/kernel`).
 *
 * **Experimental**: while the package is 0.x, anything this entry exports may change in a minor
 * version (ADR 0148).
 *
 * A fusion rule is the one form of custom kernel the runtime takes: it replaces a run of the graph's
 * **built-in** ops — one node or several — with its own kernel dispatches. No new IR op enters, so
 * the container needs nothing from the rule: without it the same graph runs unfused, and that
 * unfused plan is the reference the rule is checked against. A Session takes rules through
 * `SessionOptions.fusionRules` (fixed at construction, tried before the built-in rules in the order
 * given, counted under their names in `lastRunFusions`); passing a rule is the opt-in.
 *
 * - {@link defineRule} builds a rule from a definition: its name `<namespace>/<rule>`, the leading
 *   ops it can take (`heads`), its contract class (`contract`, ADR 0147 decision 2 — `"E"` for a
 *   rule u32-identical to the unfused plan), who may read the chain's internal values (`internals`),
 *   a matcher that grabs a window of nodes (it reads the nodes and a {@link FusionMatchContext} —
 *   the graph's use counts and outputs, nothing a plan decides), and a build that lowers it to
 *   dispatches: the binds, the output, the step's temporaries and, per dispatch, a key, the WGSL,
 *   the params and the workgroup count. Only its return value is a rule a Session takes. It applies the checks every rule meets
 *   (single-output nodes, no node that touches state, the declared privacy, a window that starts
 *   where it was tried) and checks each step it builds (only the chain's own inputs bound, the
 *   chain's last value written, every dispatch key starting with `ext:<namespace>:`).
 * - {@link allF32} and {@link sameShape} are the predicates the built-in rules use in their
 *   matchers. `defineRule` does not check dtypes: a matcher that writes an f32 kernel checks
 *   {@link allF32} itself.
 * - The WGSL parts keep the unfused rounding. {@link prologueUnaryWgsl} and {@link fusedBinaryWgsl}
 *   are the text the plain elementwise kernels compile; {@link SIGMOID_STABLE_WGSL} and
 *   {@link TANH_STABLE_WGSL} are the overflow-safe functions those kernels call
 *   ({@link IS_NAN_BITS_WGSL} with the tanh, which calls it). {@link stagedLoopWgsl} and
 *   {@link stagedF32} are the rounding barrier: a grid-stride entry point that keeps each value the
 *   unfused sequence stored — at least every product that feeds an add — at its own rounding point,
 *   so the compiler cannot contract across it. That this holds is measured per backend, not
 *   guaranteed by WGSL.
 * - {@link gridStrideWorkgroups} is the workgroup count the runtime dispatches for a `gridStride`
 *   dispatch; {@link CodegenError} is what the WGSL parts throw.
 *
 * What the runtime does and the rule does not: it allocates and releases every buffer, builds the
 * bind groups, caches the pipelines, runs every dispatch under its error scopes and requests the
 * limits. What stays with the rule's author: evidence for its numbers. A class E rule carries a
 * fused-against-unfused u32 check on each backend it ships on (docs/kernel-rules.md); a rule of
 * another class follows ADR 0147. The memory estimate does not see these rules.
 *
 * This entry has no side effects and reads no WebGPU global at import, so it can be imported where
 * WebGPU does not exist (on Node without an implementation, for example).
 *
 * @module
 */

/**
 * Builds a fusion rule written outside the runtime; the value `SessionOptions.fusionRules` takes.
 * Throws `ExecutionError` (of `@karume/runtime`) on a name that is not `<namespace>/<rule>`, on
 * heads that are not a non-empty array of op names, on a contract class or a privacy outside the
 * declared values, on a `match` or `build` that is not a function, and, each time the rule builds a
 * step, on a step that breaks the checks above.
 */
export { defineExternalRule as defineRule } from "./src/runtime/fusion.ts";
/** The parts of a rule's definition, of the step it builds, and of the context its matcher reads. */
export type {
  ContractClass,
  ExternalRuleDefinition as FusionRuleDefinition,
  ExternalRuleName,
  FusedBuild,
  FusedDispatch,
  FusedOperand,
  FusedStep,
  FusedTemp,
  FusedWorkgroups,
  FusionHit,
  FusionMatch,
  FusionMatchContext,
  FusionRule,
} from "./src/runtime/fusion.ts";
/** One node of the planned graph, with its resolved shapes and dtypes (what a matcher reads). */
export type { NodePlan } from "./src/runtime/plan.ts";
/** The predicates of the built-in matchers. */
export { allF32, sameShape } from "./src/runtime/fusion.ts";
/** The plain elementwise kernels' own value text (a fused kernel that copies it is bit-identical only by accident). */
export { fusedBinaryWgsl, prologueUnaryWgsl } from "./src/codegen/elementwise.ts";
export type { FusedBinaryOp, PrologueUnaryOp } from "./src/codegen/elementwise.ts";
/** The overflow-safe sigmoid and tanh of the plain kernels, and the NaN test the tanh calls. */
export {
  IS_NAN_BITS_WGSL,
  SIGMOID_STABLE_WGSL,
  TANH_STABLE_WGSL,
} from "./src/codegen/numerics-wgsl.ts";
/** The rounding barrier of a fused elementwise kernel. */
export { stagedF32, stagedLoopWgsl } from "./src/kernels/rounding-barrier.ts";
export type { StagedLoop, StagedValue } from "./src/kernels/rounding-barrier.ts";
/** The workgroup count of a grid-stride dispatch. */
export { gridStrideWorkgroups } from "./src/codegen/dispatch.ts";
/** The error of the WGSL generation layer. */
export { CodegenError } from "./src/codegen/errors.ts";
