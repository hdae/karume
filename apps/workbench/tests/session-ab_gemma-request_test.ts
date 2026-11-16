import { describe, expect, it } from "vite-plus/test";

import { SESSION_AB_PROMPT, sessionAbRequest } from "../src/session-ab/gemma-request.ts";

describe("the session A/B request", () => {
  it("names the two arms and the fixed work of every run", () => {
    const request = sessionAbRequest(
      {
        mount: "gemma4-qat",
        model: "26b",
        quant: "i4",
        expertCache: { slots: 64, hostBytes: Number.MAX_SAFE_INTEGER },
      },
      { rounds: 4, tokens: 96, minGain: 0.01, openPerVisit: false, probeCounts: false },
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
});
