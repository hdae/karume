/**
 * Maps the **GPU prerequisites** a `Quant` declares (`gpuFeatures` = GPU features that can only be
 * requested at device creation — ADR 0038 §3 / `requiredLimits` = minimum values of device limits
 * that must be met — ADR 0038 §7) → the request to `acquireGpu`, and the check against the
 * counterpart (an acquired `GpuContext` or the adapter's measured values) (**pipeline-independent
 * shared processing** — the feature side is used the same way by the admission and construction of
 * every family (both `fromAssets` and `fromPretrained`), the limits side by every family's
 * `fromPretrained`).
 *
 * MUST: not exported from the barrel (for the same reason as the co-located `options.ts` — it is an
 * internal mechanism that translates the distribution's declarations into runtime knobs, not a
 * surface the user touches). It sits next to the `session` mapping because both translate **fields
 * of the same `Quant`** into runtime knobs.
 *
 * MUST: the key set of the table **covers** `Required<GpuFeaturesSpec>` — when a feature is added to
 * `GpuFeaturesSpec`, this declaration fails type checking. Every family goes through this one table
 * — if each family read `quant.gpuFeatures?.<field> === true` line by line on its own, then when a
 * field is added **every family would still pass type checking** and silently drop that request
 * (= a silent degradation where execution proceeds on a device lacking a capability the
 * distribution required).
 *
 * MUST: keep "the request (what to pass to `acquireGpu`)" and "the check (is it enabled on the
 * shared GPU)" in the same single entry. Splitting them into separate tables lets a shape where only
 * one side follows (requests but does not check) pass type checking.
 *
 * NOTE: this exhaustive-table shape cannot be reused for `requiredLimits` — features are a total
 * mapping that "knows every field to build the request", whereas limits are a **partial mapping**
 * (it looks only at the limits the distribution wrote and claims nothing about unwritten limits —
 * no field = runs on the WebGPU guaranteed defaults, the semantics of ADR 0089 decision 3).
 * Vocabulary drift itself is pinned on both the hub and the runtime side by
 * `tests/limit_vocabulary_test.ts`.
 */

import type { GpuFeaturesSpec, RequiredLimitsSpec } from "@karume/hub";
import {
  type AcquireGpuOptions,
  type GpuContext,
  readAdapterLimits,
  type RequiredLimits,
  type SessionOptions,
} from "@karume/runtime";

type GpuFeatureEntry = {
  /** Whether the distribution requests this feature. */
  readonly wanted: (spec: GpuFeaturesSpec) => boolean;
  /** The field added to `acquireGpu` when requesting it. */
  readonly acquire: AcquireGpuOptions;
  /** Whether it is actually enabled on the acquired GpuContext. */
  readonly enabled: (gpu: GpuContext) => boolean;
  /** The feature's spelling in diagnostics (the WebGPU-side name). */
  readonly label: string;
  /** How to re-acquire, shown in diagnostics. */
  readonly reacquire: string;
  /**
   * Whether the effective `SessionOptions` require this feature on the device (the runtime's
   * Session construction fails under the same condition — the f16 check in `buildSessionState` of
   * `@karume/runtime`).
   */
  readonly requiredBy: (options: SessionOptions) => boolean;
  /** The field added to the declaration when it is required ({@link sessionGpuFeatures}). */
  readonly declare: GpuFeaturesSpec;
};

const FEATURES: { readonly [K in keyof Required<GpuFeaturesSpec>]: GpuFeatureEntry } = {
  shaderF16: {
    wanted: (spec) => spec.shaderF16 === true,
    acquire: { shaderF16: true },
    enabled: (gpu) => gpu.shaderF16Enabled,
    label: "shader-f16",
    reacquire: "acquireGpu({ shaderF16: true })",
    requiredBy: (options) => options.linearCompute === "f16" || options.attentionCompute === "f16",
    declare: { shaderF16: true },
  },
};

/**
 * Adds to the features the quant declared the features that the **effective** `SessionOptions`
 * (after explicit settings are composed in — `resolveSessionOptions` in `./options.ts`) require.
 *
 * MUST: every family does both the request (`acquireGpu`) and the check (shared GPU, after
 * acquisition) with this return value — including families without an explicit override. Looking
 * only at the quant's declaration (`gpuFeatures`), f16 compute that the declaration does not request
 * (an explicit `linearCompute: "f16"`, or a distribution that wrote it in `session` but left out
 * `gpuFeatures`) makes a self-acquired device lack shader-f16 and fail at Session construction after
 * all the weights were uploaded (with a shared GPU, a capability shortfall that the admission step
 * could have rejected is delayed until after the fetch). "Requested features = declared ∪ the
 * features the effective settings require" is not written separately per family.
 * NOTE: when nothing is added it returns `spec` as is (the same value as looking at the declaration
 * only — that every quant of the distribution mirror has this shape is checked by
 * `packages/e2e/tests/e2e_session_options_mirror_test.ts`).
 */
export const sessionGpuFeatures = (
  spec: GpuFeaturesSpec | undefined,
  options: SessionOptions,
): GpuFeaturesSpec | undefined => {
  let merged = spec;
  for (const feature of Object.values(FEATURES)) {
    const declared = merged !== undefined && feature.wanted(merged);
    if (!declared && feature.requiredBy(options)) merged = { ...merged, ...feature.declare };
  }
  return merged;
};

/**
 * Builds `acquireGpu` options that request only the declared features — plus, when the effective
 * `SessionOptions` declare the online attention (`attentionScoreStorage: "none"`, ADR 0132
 * decision 5), `subgroups: "available"`: the kernel's subgroup form is faster and bit-identical
 * to its portable form, so the features are requested when the adapter advertises them and never
 * required (a default preset still declares no `gpuFeatures` — ADR 0128 C1).
 */
export const toAcquireGpuOptions = (
  spec: GpuFeaturesSpec | undefined,
  options?: SessionOptions,
): AcquireGpuOptions => {
  let acquire: AcquireGpuOptions =
    options?.attentionScoreStorage === "none" ? { subgroups: "available" } : {};
  if (spec === undefined) return acquire;
  for (const feature of Object.values(FEATURES)) {
    if (feature.wanted(spec)) acquire = { ...acquire, ...feature.acquire };
  }
  return acquire;
};

/**
 * Confirms that the declared features are enabled on the acquired GpuContext.
 *
 * MUST: on the path that receives a shared GPU (`options.gpu`), features cannot be requested (they
 * can only be requested at device creation — ADR 0028), so the missing capability is named and
 * rejected **here**. Letting it through either fails only after proceeding to Session construction
 * or silently falls back to another path.
 *
 * @param where Subject of the diagnostic (`"AnimaPipeline: quant 'f16+dit8-a8-attn8-s16'"`).
 */
export const assertGpuFeaturesGranted = (
  spec: GpuFeaturesSpec | undefined,
  gpu: GpuContext,
  where: string,
): void => {
  if (spec === undefined) return;
  for (const feature of Object.values(FEATURES)) {
    if (feature.wanted(spec) && !feature.enabled(gpu)) {
      throw new Error(
        `${where} requires ${feature.label}, but it is not enabled on the passed GpuContext` +
          ` (re-acquire with ${feature.reacquire})`,
      );
    }
  }
};

/**
 * Matches the `requiredLimits` the distribution declared against the GPU-side limits (ADR 0089
 * decision 5).
 *
 * Only **the limits written in the spec** are checked (a partial mapping — the NOTE of the module
 * doc). Shortfalls are listed **all at once** and rejected (fixing them one at a time would never
 * show the full picture of what the environment lacks).
 *
 * @param limits The counterpart. `GpuContext.limits` for a shared GPU, the measured values of
 *   `readAdapterLimits()` on the self-acquiring path — both come from the same formula of
 *   `planRequiredLimits`, so they are the same measure.
 * @param where Subject of the diagnostic (`"AnimaPipeline: quant 'f16'"`).
 */
export const assertRequiredLimitsSatisfied = (
  spec: RequiredLimitsSpec | undefined,
  limits: RequiredLimits,
  where: string,
): void => {
  if (spec === undefined) return;
  // Moved onto an index-signature surface by the type annotation (that the key vocabulary agrees
  // between hub and runtime is the check of `tests/limit_vocabulary_test.ts`; here it only looks up
  // by name).
  const declared: Readonly<Record<string, number | undefined>> = spec;
  const granted: Readonly<Record<string, number | undefined>> = limits;
  const shortfalls: string[] = [];
  for (const name of Object.keys(declared)) {
    const required = declared[name];
    if (required === undefined) continue;
    const actual = granted[name];
    if (actual === undefined) {
      // Vocabulary drift (a name hub accepted that the runtime does not have). Skipping it silently
      // goes back to "a declared limit nobody looks at" = a silent ceiling.
      throw new Error(
        `${where}: the required limit '${name}' is not in the runtime's limits` +
          ` (vocabulary drift between hub and runtime — tests/limit_vocabulary_test.ts)`,
      );
    }
    if (actual < required)
      shortfalls.push(`${name}: required ${required} / this GPU has ${actual}`);
  }
  if (shortfalls.length > 0) {
    throw new Error(
      `${where}: this GPU does not meet the required device limits: ${shortfalls.join(", ")}` +
        ` (choose another quant)`,
    );
  }
};

/**
 * The limits check **before download**, called from the admission step of `fromPretrained`
 * (ADR 0089 decision 5).
 *
 * With a shared GPU (`options.gpu`) it matches that GPU's `limits`; on the self-acquiring path it
 * reads the adapter once ({@link readAdapterLimits} — creates no device and does not keep the
 * adapter) and matches against that. In an environment without a GPU this becomes a
 * `GpuUnavailableError` and **not a single byte of weights is downloaded** (that is why this step
 * exists — previously it failed after transferring several GiB).
 *
 * MUST: read the adapter once even for a distribution without a declaration
 * (`spec === undefined`). What this buys is **only the early detection of a missing GPU**; without a
 * spec the limits comparison is a no-op at the entry of {@link assertRequiredLimitsSatisfied} (the
 * semantics of ADR 0089 decision 3, "no field = runs on the default spec", do not change by a single
 * character). Not reading it would make only distributions without a declaration receive the
 * missing-GPU failure **after downloading every weight** = the very failure shape this step exists to
 * remove would remain (every family's `#build` always goes through `acquireGpu` when `options.gpu` is
 * absent, so `fromPretrained` never completes in an environment without a GPU).
 *
 * NOTE: the extra cost is a single `requestAdapter` round trip (the adapter is not kept, so the risk
 * of it expiring does not grow — {@link readAdapterLimits}). On the path that receives a shared GPU
 * the adapter is not read at all.
 *
 * NOTE: on the shared-GPU path each family's admission function has already run the same check
 * (that one also guards the `fromAssets` surface). It is compared once more here so that every
 * family's `fromPretrained` can write it as **the same one line** — it is a pure numeric comparison,
 * so checking twice has no side effects.
 */
export const assertRequiredLimitsBeforeDownload = async (
  spec: RequiredLimitsSpec | undefined,
  sharedGpu: GpuContext | undefined,
  where: string,
): Promise<void> => {
  const limits = sharedGpu === undefined ? await readAdapterLimits() : sharedGpu.limits;
  assertRequiredLimitsSatisfied(spec, limits, where);
};
