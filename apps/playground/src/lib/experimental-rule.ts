/**
 * The "Experimental fusion rule" control of the image and video pages (`/moss-image`, `/wan22`,
 * `/anima`): one candidate rule of `@karume/candidate-rules`, off by default, that a run passes
 * through its pipeline's experimental `fusionRules` option. The pipeline opens with the rule fixed
 * (a Session takes its rules at construction), so the page keys its held pipeline by the rule in
 * use as well. Pure, so `vp test` covers it.
 *
 * Every rule here is bit-identical to the unfused graph on the backends it was checked on (the u32
 * checks named in packages/candidate-rules/README.md): turning the control on is meant to change the
 * time only, never the output.
 */

import {
  EW_CHAIN_RULE,
  EW_CHAIN_RULE_NAME,
  vaeRmsSiluRule,
  vaeRmsSiluRuleName,
} from "@karume/candidate-rules";
import type { FusionRule } from "@karume/runtime/kernel";

/** One candidate a page offers. */
export type ExperimentalRule = {
  /** The rule's name (its counter in `lastRunFusions` and what the record and the result show). */
  readonly name: string;
  /** What the rule folds, in one line (the control's description). */
  readonly folds: string;
  /** Why the rule must not run on Metal; absent = it may. */
  readonly metalRefusal?: string;
  /** Builds the rule (called once per pipeline opening). */
  readonly build: () => FusionRule;
};

/** `/moss-image`: the elementwise chains of the DiT (perf-ledger F-13). */
export const EW_CHAIN_EXPERIMENT: ExperimentalRule = {
  name: EW_CHAIN_RULE_NAME,
  folds:
    "the runs of f32 elementwise nodes of dit_text, the DiT pair and the VAE in one dispatch each",
  build: () => EW_CHAIN_RULE,
};

/** `/wan22` and `/anima`: the VAE's channel RMS norm with its SiLU (perf-ledger V-10). */
export const VAE_RMS_SILU_EXPERIMENT: ExperimentalRule = {
  name: vaeRmsSiluRuleName(8),
  folds: "the VAE's channel RMS norm and the SiLU after it in one dispatch per site",
  metalRefusal:
    "Not available on Metal: the rule's rounding barrier is measured not to hold there (docs/known-issues.md).",
  build: () => vaeRmsSiluRule(8),
};

/** The adapter architectures WebGPU reports on Apple GPUs (`metal-3` and the like). */
const METAL_ARCHITECTURE = /^metal/i;

/** The platforms whose WebGPU runs on Metal alone (Chrome's and Safari's). */
const APPLE_PLATFORM = /\b(Macintosh|Mac OS X|iPhone|iPad)\b/;

/**
 * Whether this page's WebGPU runs on Metal. WebGPU does not expose the backend, so it is inferred:
 * an Apple platform in the user agent (WebGPU there has no other backend), or an adapter
 * architecture of the `metal` family.
 */
export const runsOnMetal = (userAgent: string, adapterArchitecture: string): boolean =>
  APPLE_PLATFORM.test(userAgent) || METAL_ARCHITECTURE.test(adapterArchitecture);

/** Why the control is disabled on this device (one line), or `undefined` when it may run. */
export const experimentalRuleRefusal = (
  rule: ExperimentalRule,
  metal: boolean,
): string | undefined => (metal ? rule.metalRefusal : undefined);

/**
 * The name of the rule a run passes: the rule's when the control is on and not refused here, else
 * `undefined` (no rule — the pipeline is opened as without the control).
 */
export const experimentalRuleInUse = (
  rule: ExperimentalRule,
  enabled: boolean,
  metal: boolean,
): string | undefined =>
  enabled && experimentalRuleRefusal(rule, metal) === undefined ? rule.name : undefined;

/**
 * The pipeline options of the rule in use: `{ fusionRules: [rule] }`, or nothing at all without one
 * (the option is then absent, not an empty list). Fails loudly on a name that is not `rule`'s.
 */
export const experimentalRuleOptions = (
  rule: ExperimentalRule,
  inUse: string | undefined,
): { readonly fusionRules?: readonly FusionRule[] } => {
  if (inUse === undefined) return {};
  if (inUse !== rule.name) {
    throw new Error(`The page offers ${rule.name}, not ${inUse}`);
  }
  return { fusionRules: [rule.build()] };
};

/** How a record or a result names the rule a run passed (`none` without one). */
export const experimentalRuleLabel = (inUse: string | null | undefined): string =>
  inUse === null || inUse === undefined ? "none" : `${inUse} (experimental)`;
