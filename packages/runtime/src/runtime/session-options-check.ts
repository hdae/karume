/**
 * The acceptance conditions of {@link SessionOptions} that do not depend on GPU capabilities (type,
 * spelling, combination, range) and the runtime defaults of omitted execution knobs — pure functions
 * that Session construction (session-build.ts) and the estimate (estimate.ts) read through one path.
 *
 * They are kept apart from the construction module so that the estimate (the pure-function surface
 * that never touches the GPU) has no value import of the construction phase (dependencies run one
 * way: session-build.ts / estimate.ts → here).
 * MUST: bring in no import that touches the GPU, the device or buffers (the estimate's dependency
 * would pull it in).
 */

import type { ScoreStorage } from "../kernels/score-storage.ts";
import { describeInputValue, ExecutionError } from "./plan.ts";
import {
  type ComputePrecision,
  DEFAULT_PLAN_BACKING_BUDGET_BYTES,
  type LinearGemvReduce,
  type SessionOptions,
  STATE_ATTENTION_REDUCES,
  type StateAttentionReduce,
} from "./session-types.ts";

/**
 * The accepted sets of the execution knobs of {@link SessionOptions}.
 *
 * MUST: the container is `Record<union, true>` — adding a value to the union and forgetting to update
 * this turns the missing key red **in the type check** (an array of values would let the added value
 * drop out of the accepted set silently).
 */
const LINEAR_COMPUTES: Readonly<Record<NonNullable<SessionOptions["linearCompute"]>, true>> = {
  f32: true,
  a8: true,
  f16: true,
};
const CONV_COMPUTES: Readonly<Record<NonNullable<SessionOptions["convCompute"]>, true>> = {
  f32: true,
  a8: true,
};
const ATTENTION_COMPUTES: Readonly<Record<ComputePrecision, true>> = {
  f32: true,
  f16: true,
  a8: true,
};
const SCORE_STORAGES: Readonly<Record<ScoreStorage, true>> = { f32: true, f16: true, none: true };
const LINEAR_GEMV_REDUCES: Readonly<Record<LinearGemvReduce, true>> = {
  sequential: true,
  parallel: true,
  "parallel-subgroup32": true,
};

/**
 * Checks the spelling of the execution knobs at the entry of `Session.build` in
 * {@link "./executor.ts"}.
 *
 * MUST: a spelling outside the union fails loudly. Every downstream consumer compares by equality
 * (`=== "a8"` / `=== "f16"` / `=== "parallel"`), so a single wrong character **runs silently on the
 * default (f32 / sequential)** — the opt-in is not applied, yet the result reads as "a8 was
 * measured". The TS types protect only TS callers; JS callers and code that kept a spelling from
 * before a rename (`linearCompute` went `"i8a8"` → `"a8"` in 0.5.0, ADR 0074 decision 3, with no
 * compatibility shim) are outside that net.
 * MUST: list **every violation and fail once** (the same convention as `assertWeightsWithinLimits` —
 * failing one at a time makes the next one appear after each fix, and how many fixes are needed stays
 * unknown to the end).
 * MUST: check the type (string) with `typeof` **before** looking up the accepted set. `Object.hasOwn`
 * converts the value to a property key, so without this check `["a8"]` is accepted as `'a8'` and then
 * misses the strict downstream comparisons, falling back silently to the default.
 * MUST: do not call the user's conversions (`toString` / `Symbol.toPrimitive` / `toJSON`) even in
 * diagnostics. For non-strings only the `typeof` type name is printed — running user code in a
 * diagnostic at the input boundary lets the user's exception escape from `createContainerSession`
 * instead of an `ExecutionError`.
 *
 * It is exported within the package for tests (a pure function that never touches the GPU, so
 * regressions can be exercised without an adapter). It is not on the public surface of `mod.ts`
 * (ADR 0008).
 */
export const assertExecutionKnobs = (
  linearCompute: NonNullable<SessionOptions["linearCompute"]>,
  convCompute: NonNullable<SessionOptions["convCompute"]>,
  attentionCompute: ComputePrecision,
  attentionScoreStorage: ScoreStorage,
  stateAttentionReduce: StateAttentionReduce,
  linearGemvReduce: LinearGemvReduce,
): void => {
  const violation = executionKnobsViolation(
    linearCompute,
    convCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
  );
  if (violation !== undefined) throw new ExecutionError(violation);
};

/** The decision body of {@link assertExecutionKnobs} (returns the violation text; the caller decides whether to throw). */
const executionKnobsViolation = (
  linearCompute: NonNullable<SessionOptions["linearCompute"]>,
  convCompute: NonNullable<SessionOptions["convCompute"]>,
  attentionCompute: ComputePrecision,
  attentionScoreStorage: ScoreStorage,
  stateAttentionReduce: StateAttentionReduce,
  linearGemvReduce: LinearGemvReduce,
): string | undefined => {
  const knobs: readonly (readonly [string, string, Readonly<Record<string, true>>])[] = [
    ["linearCompute", linearCompute, LINEAR_COMPUTES],
    ["convCompute", convCompute, CONV_COMPUTES],
    ["attentionCompute", attentionCompute, ATTENTION_COMPUTES],
    ["attentionScoreStorage", attentionScoreStorage, SCORE_STORAGES],
    ["stateAttentionReduce", stateAttentionReduce, STATE_ATTENTION_REDUCES],
    ["linearGemvReduce", linearGemvReduce, LINEAR_GEMV_REDUCES],
  ];
  const violations = knobs
    .filter(([, value, accepted]) => typeof value !== "string" || !Object.hasOwn(accepted, value))
    .map(
      ([name, value, accepted]) =>
        `  - ${name}: ${
          typeof value === "string" ? JSON.stringify(value) : typeof value
        } (accepted: ${Object.keys(accepted)
          .map((accept) => `'${accept}'`)
          .join(" / ")})`,
    );
  if (violations.length === 0) return undefined;
  return (
    `SessionOptions: execution knobs outside the accepted set: ${violations.length} (no silent fallback to the default):\n` +
    `${violations.join("\n")}\n` +
    "Check the spelling (linearCompute 'i8a8' was renamed to 'a8' in 0.5.0 — ADR 0074 decision 3)"
  );
};

/**
 * The values with runtime defaults filled into omitted execution knobs ({@link sessionOptionsViolation}
 * and `buildSessionState` in session-build.ts read them through this one function — spelling the
 * defaults in two places could split the value that was checked from the value used to build).
 */
export const executionKnobDefaults = (options: SessionOptions) => ({
  rmsNormReduce: options.rmsNormReduce === undefined ? "workgroup" : options.rmsNormReduce,
  linearCompute: options.linearCompute ?? "f32",
  convCompute: options.convCompute ?? "f32",
  attentionCompute: options.attentionCompute ?? "f32",
  attentionScoreStorage: options.attentionScoreStorage ?? "f32",
  stateAttentionReduce: options.stateAttentionReduce ?? "sequential",
  linearGemvReduce: options.linearGemvReduce ?? "sequential",
  planBackingBudgetBytes: options.planBackingBudgetBytes ?? DEFAULT_PLAN_BACKING_BUDGET_BYTES,
});

/**
 * Decides in one pass the acceptance conditions of {@link SessionOptions} that **do not depend on GPU
 * capabilities** (type, spelling, combination, range) and returns the text of the first violation
 * (`undefined` when there is none).
 *
 * Session construction (`createSessionFromContainer` / `PreparedModel.createContainerSession`) calls
 * this function at its entry and throws a violation as `ExecutionError`. It is public for callers
 * that want to borrow the same decision without waiting for construction — upper layers that combine
 * a distribution's declaration with the user's explicit options and want to reject an unacceptable
 * combination **before fetching a single byte of weights**. It returns only the text, not an error
 * type, because the remedy for the same violation depends on its origin (the distribution's
 * declaration or the caller's options), and only the caller, who knows the origin, can choose the
 * type.
 *
 * MUST: conditions that need GPU capabilities (`shader-f16` / subgroups) are not included here — they
 * cannot be decided without the device, so they stay on the construction side. Construction can still
 * fail when this returns `undefined`.
 * MUST: do not call the user's conversions (`toString` / `Symbol.toPrimitive`) in diagnostics (the
 * same rule as {@link assertExecutionKnobs}).
 * NOTE: omitted fields are judged as the runtime defaults (`f32` / `sequential` etc.). A field value
 * such as `null` is also read as the default through `??` (the same reading as construction), so a
 * caller that wants to reject "field present but value invalid" rather than fall back to the default
 * checks the range at its own entry first.
 */
export const sessionOptionsViolation = (options: SessionOptions): string | undefined => {
  if (
    options.fuseLinearStaticQuantize !== undefined &&
    typeof options.fuseLinearStaticQuantize !== "boolean"
  ) {
    return "options.fuseLinearStaticQuantize must be a boolean";
  }
  if (
    options.fuseGeluMulStaticQuantize !== undefined &&
    typeof options.fuseGeluMulStaticQuantize !== "boolean"
  ) {
    return "options.fuseGeluMulStaticQuantize must be a boolean";
  }
  if (options.fuseRmsNormAdd !== undefined && typeof options.fuseRmsNormAdd !== "boolean") {
    return "options.fuseRmsNormAdd must be a boolean";
  }
  if (
    options.packedStaticQuantize !== undefined &&
    typeof options.packedStaticQuantize !== "boolean"
  ) {
    return "options.packedStaticQuantize must be a boolean";
  }
  if (options.integerGemv !== undefined && typeof options.integerGemv !== "boolean") {
    return "options.integerGemv must be a boolean";
  }
  if (options.linearGemvPlanes !== undefined && typeof options.linearGemvPlanes !== "boolean") {
    return "options.linearGemvPlanes must be a boolean";
  }
  const {
    rmsNormReduce,
    linearCompute,
    convCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
    planBackingBudgetBytes,
  } = executionKnobDefaults(options);
  if (rmsNormReduce !== "workgroup" && rmsNormReduce !== "subgroup32") {
    // Do not call the user's conversions in diagnostics (`String(x)` runs `Symbol.toPrimitive` /
    // `toString`, and an object that throws would escape instead of the `ExecutionError`).
    return `options.rmsNormReduce: unsupported value ${
      typeof rmsNormReduce === "string" ? JSON.stringify(rmsNormReduce) : typeof rmsNormReduce
    }`;
  }
  // MUST: the spelling check comes right after the defaults are filled in and before every later
  // check. Once past it, the s16×c16 check and the f16 feature check only need to consider values
  // inside the union.
  const knobs = executionKnobsViolation(
    linearCompute,
    convCompute,
    attentionCompute,
    attentionScoreStorage,
    stateAttentionReduce,
    linearGemvReduce,
  );
  if (knobs !== undefined) return knobs;
  if (
    options.fuseLinearStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    return "fuseLinearStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32";
  }
  // Only the parallel GEMV family has the packed-activation variant (ADR 0105). Falling back silently
  // to the f32 path would make it an option that "does nothing when set", so it is rejected the same
  // way as the fusion.
  if (
    options.packedStaticQuantize === true &&
    (linearGemvReduce !== "parallel" || linearCompute !== "f32")
  ) {
    return "packedStaticQuantize supports only linearGemvReduce: parallel / linearCompute: f32";
  }
  // The integer GEMV reads only packed int8 activations (src/kernels/linear-gemv-integer.ts), so
  // without the packed pairing it could never run: rejected rather than accepted as a no-op. The
  // packed check above already holds packedStaticQuantize to parallel / f32.
  if (options.integerGemv === true && options.packedStaticQuantize !== true) {
    return "integerGemv requires packedStaticQuantize: true (with linearGemvReduce: parallel / linearCompute: f32)";
  }
  if (linearGemvReduce !== "sequential" && linearCompute !== "f32") {
    return `linearGemvReduce: ${linearGemvReduce} supports only linearCompute: f32`;
  }
  // Range checks (which do not read the union) come after the spelling check — the wording matches
  // the same check in estimate.ts.
  if (!Number.isSafeInteger(planBackingBudgetBytes) || planBackingBudgetBytes < 0) {
    return (
      `options.planBackingBudgetBytes ${describeInputValue(
        planBackingBudgetBytes,
      )} must be a non-negative safe` + " integer"
    );
  }
  // The parallelism target of the row-block GEMV is checked the same way (0 / negative / non-integer
  // mean nothing as a "thread-count target", and accepting them silently gives a run with the
  // largest rows chosen).
  const linearGemvRowsThreadTarget = options.linearGemvRowsThreadTarget;
  if (
    linearGemvRowsThreadTarget !== undefined &&
    (!Number.isSafeInteger(linearGemvRowsThreadTarget) || linearGemvRowsThreadTarget < 1)
  ) {
    return (
      `options.linearGemvRowsThreadTarget ${describeInputValue(
        linearGemvRowsThreadTarget,
      )} must be a safe integer` + " of at least 1"
    );
  }
  // The budget per row-block temporary (ADR 0124 decision 5). Only invalid values known at
  // construction fail here — "a budget smaller than a single row" fails, naming the budget, in the
  // plan derivation of the run where the shape is fixed (at construction the shape is still
  // symbolic). The wording matches the same check in estimate.ts.
  const attentionRowBlockMaxBytes = options.attentionRowBlockMaxBytes;
  if (
    attentionRowBlockMaxBytes !== undefined &&
    (!Number.isSafeInteger(attentionRowBlockMaxBytes) || attentionRowBlockMaxBytes < 1)
  ) {
    return `options.attentionRowBlockMaxBytes ${describeInputValue(
      attentionRowBlockMaxBytes,
    )} must be a positive safe integer`;
  }
  // MUST: the storage form of S must resolve to one. `:c16` is **a different form** that holds S as
  // array<f16> (ADR 0028), so when it is given together with s16, fail rather than silently pick one
  // reading (which rounding sequence ran would vanish from both the diagnostics and the numbers).
  if (attentionScoreStorage === "f16" && attentionCompute === "f16") {
    return (
      "attentionScoreStorage 'f16' and attentionCompute 'f16' cannot be given together" +
      " (attentionCompute 'f16' is a different storage form that holds S as array<f16> — " +
      "to halve S without shader-f16, set attentionCompute to 'f32' or 'a8')"
    );
  }
  // MUST: the online kernel exists only in the a8 form (ADR 0132 decision 1). Any other compute form
  // fails loudly — an unimplemented combination is not a fallback (ADR 0058 decision 3); the shape
  // and device fallbacks of an a8 request are the route's (attention-route.ts).
  if (attentionScoreStorage === "none" && attentionCompute !== "a8") {
    return "attentionScoreStorage 'none' (the online kernel) is implemented only with attentionCompute 'a8'";
  }
  return undefined;
};
