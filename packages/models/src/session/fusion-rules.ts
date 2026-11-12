/**
 * The experimental `fusionRules` pass-through of the image and video pipelines (MOSS-Image, Wan,
 * anima): fusion rules written outside the runtime (`defineRule` of `@karume/runtime/kernel` —
 * ADR 0148), passed unchanged as `SessionOptions.fusionRules` to the Sessions a pipeline names and
 * to the estimates of the same graphs.
 *
 * MUST: absent rules leave every options object as it was (the same object, no `fusionRules` key,
 * no added limit), so a pipeline without the option builds and estimates exactly as before.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state).
 */

import type { EstimateOptions, SessionOptions } from "@karume/runtime";

/** The rules a pipeline passes through (the runtime's own type — not interpreted here). */
export type PipelineFusionRules = SessionOptions["fusionRules"];

/**
 * `options` with the rules added (`options` itself when there are none). The caller's array is
 * passed as is, not copied: the Session copies the list it plans with.
 */
export const withFusionRules = (
  options: SessionOptions,
  rules: PipelineFusionRules,
): SessionOptions => (rules === undefined ? options : { ...options, fusionRules: rules });

/**
 * The estimate's side of the same rules: the rules and the device's
 * `maxComputeWorkgroupsPerDimension`, which a non-empty rule list needs in the estimate (a rule's
 * matcher may read it — `EstimateOptions.maxComputeWorkgroupsPerDimension`), or nothing.
 *
 * MUST: nothing at all without rules (the limit is not added either): the estimate then plans as it
 * did before the option existed. With rules and without the limit the estimate fails loudly (the
 * runtime's check) — the pipelines pass the device's `gpu.limits`, which always carry it.
 */
export const fusionRulesEstimateOptions = (
  rules: PipelineFusionRules,
  limits: { readonly maxComputeWorkgroupsPerDimension?: number },
): Pick<EstimateOptions, "fusionRules" | "maxComputeWorkgroupsPerDimension"> => {
  if (rules === undefined) return {};
  const limit = limits.maxComputeWorkgroupsPerDimension;
  return {
    fusionRules: rules,
    ...(limit === undefined ? {} : { maxComputeWorkgroupsPerDimension: limit }),
  };
};
