/**
 * The vocabulary a strategy table states about each kernel strategy (ADR 0152 decisions 2 and 3).
 *
 * NOTE: types only. Not part of the public surface of mod.ts (ADR 0008).
 */

/**
 * How far a strategy's numbers may stand from the reference kernel (ADR 0152 decision 2).
 *
 * - `exact`: the reference kernel, or a kernel checked to give its bits.
 * - `reorder`: only the summation order differs; the error stays inside the op's band.
 * - `approximation`: anything else — it is named, by the Session option value that asks for it.
 */
export type RequirementLevel = "exact" | "reorder" | "approximation";

/**
 * The kind of evidence a strategy carries (ADR 0152 decision 3; the classes of
 * docs/quantization.md Table 2-b spelled out).
 */
export type EvidenceForm = "reference" | "mirror" | "composition" | "reorder" | "approximation";

export type StrategyInfo = {
  readonly level: RequirementLevel;
  readonly evidence: EvidenceForm;
  /** The Session option value that admits the strategy today (`undefined`: no option — the default). */
  readonly admittedBy: string | undefined;
};
