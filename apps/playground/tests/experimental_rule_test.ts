import { EW_CHAIN_RULE, EW_CHAIN_RULE_NAME } from "@karume/candidate-rules";
import { describe, expect, it } from "vite-plus/test";

import {
  EW_CHAIN_EXPERIMENT,
  experimentalRuleInUse,
  experimentalRuleLabel,
  experimentalRuleOptions,
  experimentalRuleRefusal,
  runsOnMetal,
  VAE_RMS_SILU_EXPERIMENT,
} from "../src/lib/experimental-rule.ts";
import { initialForm as animaInitialForm } from "../src/models/anima/settings.ts";
import { initialForm as mossInitialForm } from "../src/models/moss-image/settings.ts";

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

  it("the DiT rule has no Metal refusal", () => {
    expect(experimentalRuleRefusal(EW_CHAIN_EXPERIMENT, true)).toBeUndefined();
  });

  it("names the rule in use only when the control is on and the rule may run here", () => {
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, true, false)).toBe(
      "candidate/vae_rms_silu_g8",
    );
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, true, true)).toBeUndefined();
    expect(experimentalRuleInUse(VAE_RMS_SILU_EXPERIMENT, false, false)).toBeUndefined();
    expect(experimentalRuleInUse(EW_CHAIN_EXPERIMENT, true, true)).toBe(EW_CHAIN_RULE_NAME);
  });

  it("without a rule in use the pipeline options carry no fusionRules key at all", () => {
    const options = experimentalRuleOptions(EW_CHAIN_EXPERIMENT, undefined);
    expect(options).toEqual({});
    expect(Object.hasOwn(options, "fusionRules")).toBe(false);
  });

  it("with the rule in use the options carry exactly that rule", () => {
    expect(experimentalRuleOptions(EW_CHAIN_EXPERIMENT, EW_CHAIN_RULE_NAME).fusionRules).toEqual([
      EW_CHAIN_RULE,
    ]);
    const vae = experimentalRuleOptions(VAE_RMS_SILU_EXPERIMENT, "candidate/vae_rms_silu_g8");
    expect(vae.fusionRules?.map((rule) => rule.name)).toEqual(["candidate/vae_rms_silu_g8"]);
  });

  it("a name the page does not offer fails loudly", () => {
    expect(() => experimentalRuleOptions(EW_CHAIN_EXPERIMENT, "candidate/vae_rms_silu_g8")).toThrow(
      /offers candidate\/ew_chain/,
    );
  });

  it("labels the rule a record names, and none", () => {
    expect(experimentalRuleLabel(EW_CHAIN_RULE_NAME)).toBe("candidate/ew_chain (experimental)");
    expect(experimentalRuleLabel(null)).toBe("none");
    expect(experimentalRuleLabel(undefined)).toBe("none");
  });

  it("is off in the initial forms of /moss-image and /anima", () => {
    expect(mossInitialForm("a red fox").experimentalRule).toBe(false);
    expect(animaInitialForm("a red fox").experimentalRule).toBe(false);
  });
});
