/**
 * The warning for running int8 activations (`linearCompute` / `attentionCompute: "a8"`) on a device
 * without dp4a (DECIDED: ADR 0128 decision 5 — a device without dp4a gets a warning, not a fallback
 * and not an automatic sibling preset).
 *
 * The i8a8 integer dot product uses the dp4a built-in (WGSL language feature
 * `packed_4x8_integer_dot_product`) when it exists and runs emulated otherwise. **Both variants give
 * identical numbers** (src/kernels/linear-i8a8.ts), so this is not a correctness problem but a speed
 * problem only — emulation is never faster than the f32 path, and on Apple's Metal it was measured
 * to be slower (ADR 0115 / ADR 0128 Context). The distributions with a8 in their default
 * (anima / irodori / wan) were chosen by measurement on GPUs with dp4a; a user on a device without
 * dp4a pays the quality cost of activation quantization with no speed gain. Passing silently hides
 * that fact from the user (it shows in the diagnostics' pipeline key `:dp4aEmu`, but people on the
 * default do not read diagnostics).
 *
 * The decision is a pure function so it can be tested without a GPU. Limiting it to once per device
 * is the caller's job (session-build.ts) through `warnOnce` of {@link GpuContext} — it keeps the same
 * single fact from appearing once per Session in families that build one a8 Session per component
 * (anima's text encoder / DiT).
 */

import { DP4A_WGSL_FEATURE, dp4aAvailable } from "../kernels/linear-i8a8.ts";
import type { ComputePrecision } from "./session-types.ts";

/** The `GpuContext.warnOnce` key (the once-per-device unit = the fact "this device has no dp4a"). */
export const A8_EMULATION_WARNING_KEY = "a8-without-dp4a";

/**
 * The warning text when a8 was requested but the device has no dp4a. Otherwise (a8 not used, or dp4a
 * present) undefined.
 *
 * The wording follows the attention canary's warning (session-build.ts) — what happened, and what
 * makes it go away. That an escape route, "a preset with the same weights and no activation
 * quantization", always exists is guaranteed by the mechanical check of distributions
 * (`distribution_gate_test.ts` — ADR 0128 decision 2).
 */
export const a8EmulationWarning = (
  linearCompute: ComputePrecision,
  attentionCompute: ComputePrecision,
  wgslLanguageFeatures: ReadonlySet<string>,
): string | undefined => {
  const requested = [
    ...(linearCompute === "a8" ? ["linearCompute"] : []),
    ...(attentionCompute === "a8" ? ["attentionCompute"] : []),
  ];
  if (requested.length === 0 || dp4aAvailable(wgslLanguageFeatures)) return undefined;
  return (
    `@karume/runtime: この device は dp4a（WGSL 言語機能 ${DP4A_WGSL_FEATURE}）を持たないため、\n` +
    `  int8 活性（${requested.join(" / ")}: "a8"）の整数内積はエミュレーションで走る。\n` +
    `数値は dp4a 版と同一だが、f32 の経路より速くならない（Apple の Metal では遅くなる実測がある）。\n` +
    `速度のために選んだのなら、同じ重みで活性量子化を持たないプリセット（モデルカードの Quants 表で\n` +
    `Compute 列が空のもの）を quant で明示するか、linearCompute / attentionCompute を 'f32' にすること。`
  );
};
