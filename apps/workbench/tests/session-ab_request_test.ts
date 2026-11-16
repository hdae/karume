import { describe, expect, it } from "vite-plus/test";

import { ParameterError } from "../src/entry.ts";
import {
  presetOf,
  SESSION_AB_DEFAULTS,
  SESSION_AB_PROMPT,
  sessionAbRequest,
} from "../src/session-ab/request.ts";

import { gemmaManifestText, manifestOf } from "./helpers/session-ab-manifests.ts";

/** One model with the presets i4 and i4-fast (the default). */
const MANIFEST = manifestOf(gemmaManifestText(["e2b"]));

describe("the session A/B request", () => {
  describe("the defaults", () => {
    it("are 4 rounds, 96 tokens, a minimum gain of 1%, open per visit and the counts probed", () => {
      expect(SESSION_AB_DEFAULTS).toEqual({
        rounds: 4,
        tokens: 96,
        minGain: 0.01,
        openPerVisit: true,
        probeCounts: true,
      });
    });
  });

  it("names the two arms and the fixed work of every run", () => {
    const request = sessionAbRequest(
      {
        mount: "gemma4-qat",
        model: "26b",
        quant: "i4",
        expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
      },
      { ...SESSION_AB_DEFAULTS, openPerVisit: false, probeCounts: false },
      "candidate/gelu_mul",
    );
    expect(request).toEqual({
      mount: "gemma4-qat",
      model: "26b",
      quant: "i4",
      expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
      rounds: 4,
      tokens: 96,
      minGain: 0.01,
      openPerVisit: false,
      probeCounts: false,
      prompt: SESSION_AB_PROMPT,
      sampler: { temperature: 0, seed: 0 },
      capacity: 1024,
      arms: [
        { name: "base", fusionRules: [] },
        { name: "rule", fusionRules: ["candidate/gelu_mul"] },
      ],
    });
  });

  describe("the preset", () => {
    it("is the model's default preset when none is named, and the named one when the model has it", () => {
      expect(presetOf(MANIFEST, "e2b", "")).toBe("i4-fast");
      expect(presetOf(MANIFEST, "e2b", "i4")).toBe("i4");
    });

    it("refuses a name that is not a preset of the model, listing its presets", () => {
      expect(() => presetOf(MANIFEST, "e2b", "f16")).toThrow(ParameterError);
      expect(() => presetOf(MANIFEST, "e2b", "f16")).toThrow(
        "Parameter preset: 'f16' is not a preset of model 'e2b'; its presets are i4, i4-fast",
      );
    });
  });
});
