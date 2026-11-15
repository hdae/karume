/**
 * The `reorder` strategies whose gain depends on the adapter ({@link ReorderStrategies}).
 *
 * A level of `reorder` (`SessionOptions.numerics`) admits these strategies, but a strategy that
 * was not measured to gain on an adapter is not selected there, and nothing is tuned at run time
 * (ADR 0152 decision 8). No adapter has such a measurement today, so no Session selects them by
 * itself: on the Apple M2 the subgroup GEMV was about 9% slower than the parallel GEMV, the
 * subgroup RMS within the run-to-run spread, and the byte-plane GEMV faster per dispatch but
 * within 0.1% end to end (docs/research/2026-09-13-m2-gemv-subgroup-adoption.md,
 * 2026-09-13-gemv-subgroup.md, 2026-11-03-qat-integer-gemv-m2.md §10); the gain of the subgroup
 * GEMV on the RTX 3080 Ti was measured against the parallel GEMV without packed activations,
 * which it cannot take. They run through the private Session key `REORDER_STRATEGIES` (the kernel
 * checks and tests), which the memory estimate takes too. Per-adapter tables are where a measured
 * gain will list them.
 *
 * Pure and device-free.
 */

/** The adapter-dependent `reorder` strategies; an absent field keeps the portable choice. */
export type ReorderStrategies = {
  /**
   * The parallel GEMV in its subgroup form: the same partition and summation tree as the parallel
   * GEMV through value exchange within 32 lanes, so the same bits. It takes neither packed int8
   * activations nor the linear→static_quantize fusion, so a Session that asks for packed
   * activations keeps the workgroup form.
   */
  readonly linearGemv?: "subgroup32";
  /**
   * The byte-plane GEMV on its measured shapes (i2 weights, f32 activations): the same exact
   * products in another order, the channel scale applied once after the sum.
   */
  readonly linearGemvPlanes?: true;
  /** The RMS reduction through 32-lane subgroups: another summation order than the workgroup form. */
  readonly rmsNorm?: "subgroup32";
};

/** No adapter-dependent strategy: every op kind keeps its portable choice. */
export const NO_REORDER_STRATEGIES: ReorderStrategies = Object.freeze({});

/**
 * The violation of a strategies value from outside the type (`undefined` when acceptable): the
 * test-only Session key and the estimate's fact are read through it.
 */
export const reorderStrategiesViolation = (value: unknown, where: string): string | undefined => {
  if (value === undefined) return undefined;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return `${where} must be an object { linearGemv?, linearGemvPlanes?, rmsNorm? }`;
  }
  const record: Readonly<Record<string, unknown>> = { ...value };
  for (const [key, field] of Object.entries(record)) {
    if (field === undefined) continue;
    if (key === "linearGemv" || key === "rmsNorm") {
      if (field !== "subgroup32") return `${where}.${key} must be "subgroup32" or omitted`;
    } else if (key === "linearGemvPlanes") {
      if (field !== true) return `${where}.linearGemvPlanes must be true or omitted`;
    } else {
      return `${where}.${key} is not a field`;
    }
  }
  return undefined;
};

/**
 * A copy as plain data, each field read once (the rule of `copyNumerics`: the value the check
 * sees is the value used). A value the check rejects by its own type is returned as it is.
 */
export const copyReorderStrategies = (
  strategies: ReorderStrategies | undefined,
): ReorderStrategies | undefined => {
  if (typeof strategies !== "object" || strategies === null || Array.isArray(strategies)) {
    return strategies;
  }
  return { ...strategies };
};
