import { GELU_MUL_RULE, GELU_MUL_RULE_NAME } from "@karume/candidate-rules";
import { describe, expect, it } from "vite-plus/test";

import {
  type ExperimentalRule,
  experimentalRuleInUse,
  experimentalRuleLabel,
  experimentalRuleOptions,
  experimentalRuleRefusal,
  runsOnMetal,
  VAE_RMS_SILU_EXPERIMENT,
} from "../src/lib/experimental-rule.ts";
import { initialForm as animaInitialForm } from "../src/models/anima/settings.ts";

/** A rule with no Metal refusal (no page offers one now), for the branch where Metal may run it. */
const NO_REFUSAL: ExperimentalRule = {
  name: GELU_MUL_RULE_NAME,
  folds: "a test rule",
  build: () => GELU_MUL_RULE,
};

const CHROME_WINDOWS =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36";
const CHROME_LINUX =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36";
const CHROME_MAC =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36";
const SAFARI_IPAD =
  "Mozilla/5.0 (iPad; CPU OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";

describe("runsOnMetal (WebGPU does not expose the backend)", () => {
  it("an Apple platform in the user agent is Metal, whatever the adapter says", () => {
    expect(runsOnMetal(CHROME_MAC, "")).toBe(true);
    expect(runsOnMetal(SAFARI_IPAD, "")).toBe(true);
  });

  it("an adapter of the metal architecture family is Metal on any user agent", () => {
    expect(runsOnMetal("test", "metal-3")).toBe(true);
  });

  it("Windows and Linux with other adapters are not", () => {
    expect(runsOnMetal(CHROME_WINDOWS, "blackwell")).toBe(false);
    expect(runsOnMetal(CHROME_LINUX, "ampere")).toBe(false);
    expect(runsOnMetal(CHROME_LINUX, "")).toBe(false);
  });
});

describe("the experimental rule control", () => {
  it("the VAE rule is refused on Metal with a one-line reason, and allowed elsewhere", () => {
    const refusal = experimentalRuleRefusal(VAE_RMS_SILU_EXPERIMENT, true);
    expect(refusal).toMatch(/Metal/);
    expect(refusal).not.toMatch(/\n/);
    expect(experimentalRuleRefusal(VAE_RMS_SILU_EXPERIMENT, false)).toBeUndefined();
  });

  it("a rule without a Metal refusal is not refused on Metal", () => {
    expect(experimentalRuleRefusal(NO_REFUSAL, true)).toBeUndefined();
  });

  it("names the rule in use only when the control is on and the rule may run here", () => {
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, true, false)).toBe(
      "candidate/vae_rms_silu_g8",
    );
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, true, true)).toBeUndefined();
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, false, false)).toBeUndefined();
    expect(experimentalRuleInUse(NO_REFUSAL, true, true)).toBe(GELU_MUL_RULE_NAME);
  });

  it("without a rule in use the pipeline options carry no fusionRules key at all", () => {
    const options = experimentalRuleOptions(VAE_RMS_SILU_EXPERIMENT, undefined);
    expect(options).toEqual({});
    expect(Object.hasOwn(options, "fusionRules")).toBe(false);
  });

  it("with the rule in use the options carry exactly that rule", () => {
    expect(experimentalRuleOptions(NO_REFUSAL, GELU_MUL_RULE_NAME).fusionRules).toEqual([
      GELU_MUL_RULE,
    ]);
    const vae = experimentalRuleOptions(VAE_RMS_SILU_EXPERIMENT, "candidate/vae_rms_silu_g8");
    expect(vae.fusionRules?.map((rule) => rule.name)).toEqual(["candidate/vae_rms_silu_g8"]);
  });

  it("a name the page does not offer fails loudly", () => {
    expect(() => experimentalRuleOptions(VAE_RMS_SILU_EXPERIMENT, GELU_MUL_RULE_NAME)).toThrow(
      /offers candidate\/vae_rms_silu_g8/,
    );
  });

  it("labels the rule a record names, and none", () => {
    expect(experimentalRuleLabel("candidate/vae_rms_silu_g8")).toBe(
      "candidate/vae_rms_silu_g8 (experimental)",
    );
    expect(experimentalRuleLabel(null)).toBe("none");
    expect(experimentalRuleLabel(undefined)).toBe("none");
  });

  it("is off in the initial form of /anima", () => {
    expect(animaInitialForm("a red fox").experimentalRule).toBe(false);
  });
});
