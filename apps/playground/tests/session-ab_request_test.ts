import { describe, expect, it } from "vite-plus/test";

import {
  INITIAL_SESSION_AB_FORM,
  parseSessionAbForm,
  SESSION_AB_PROMPT,
  type SessionAbForm,
  sessionAbRequest,
} from "../src/models/session-ab/request.ts";

const chosen: SessionAbForm = { ...INITIAL_SESSION_AB_FORM, backend: "d3d12" };

const errorsOf = (form: SessionAbForm): readonly string[] => {
  const parsed = parseSessionAbForm(form);
  return parsed.ok ? [] : parsed.errors;
};

describe("the session A/B request", () => {
  describe("the defaults", () => {
    it("are 4 rounds, 96 tokens, a minimum gain of 1%, open per visit and counts read", () => {
      expect(parseSessionAbForm(chosen)).toEqual({
        ok: true,
        value: {
          rounds: 4,
          tokens: 96,
          minGain: 0.01,
          openPerVisit: true,
          readCounts: true,
          backend: "d3d12",
        },
      });
    });

    it("name no backend: the form is refused until one is chosen", () => {
      expect(INITIAL_SESSION_AB_FORM.backend).toBe("");
      expect(errorsOf(INITIAL_SESSION_AB_FORM)).toEqual([
        "Choose the backend (d3d12 / metal / vulkan / other).",
      ]);
      expect(errorsOf({ ...chosen, backend: "webgl" })).toHaveLength(1);
      for (const backend of ["d3d12", "metal", "vulkan", "other"]) {
        expect(errorsOf({ ...chosen, backend })).toEqual([]);
      }
    });
  });

  describe("the validation", () => {
    it("takes 3 rounds or more, the runner's minimum", () => {
      expect(errorsOf({ ...chosen, rounds: "3" })).toEqual([]);
      for (const rounds of ["2", "0", "", "3.5", "-4", "four"]) {
        expect(errorsOf({ ...chosen, rounds }), rounds).toEqual(["Rounds must be an integer ≥ 3."]);
      }
    });

    it("takes 2 to 512 tokens: a run of one token has no decode run to count", () => {
      for (const tokens of ["2", "512"]) expect(errorsOf({ ...chosen, tokens })).toEqual([]);
      for (const tokens of ["1", "513", "", "9.5"]) {
        expect(errorsOf({ ...chosen, tokens }), tokens).toEqual([
          "Tokens per run must be an integer from 2 to 512.",
        ]);
      }
    });

    it("takes a minimum gain that is a non-negative number, 0 included", () => {
      for (const minGain of ["0", "0.05", "1"]) {
        expect(errorsOf({ ...chosen, minGain }), minGain).toEqual([]);
      }
      for (const minGain of ["-0.01", "", "1%", "Infinity", "NaN"]) {
        expect(errorsOf({ ...chosen, minGain }), minGain).toHaveLength(1);
      }
    });

    it("reports every problem at once", () => {
      expect(
        errorsOf({ ...INITIAL_SESSION_AB_FORM, rounds: "1", tokens: "0", minGain: "x" }),
      ).toHaveLength(4);
    });
  });

  it("names the two arms and the fixed work of every run", () => {
    const parsed = parseSessionAbForm({ ...chosen, openPerVisit: false, readCounts: false });
    if (!parsed.ok) throw new Error(parsed.errors.join(" "));
    const request = sessionAbRequest(
      {
        mount: "gemma4-qat",
        model: "26b",
        quant: "i4",
        expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
      },
      parsed.value,
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
      readCounts: false,
      backend: "d3d12",
      prompt: SESSION_AB_PROMPT,
      sampler: { temperature: 0, seed: 0 },
      capacity: 1024,
      arms: [
        { name: "base", fusionRules: [] },
        { name: "rule", fusionRules: ["candidate/gelu_mul"] },
      ],
    });
  });
});
