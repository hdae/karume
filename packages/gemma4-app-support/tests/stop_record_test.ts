import type { ChatStop } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import { stopRecordOf } from "../stop-record.ts";

describe("stopRecordOf", () => {
  it("keeps the stop token of an EOS / stop-token stop", () => {
    const stop: ChatStop = { reason: "eos", token: 106, tokens: 3 };
    expect(stopRecordOf(stop)).toEqual({ reason: "eos", tokens: 3, token: 106 });
  });

  it("keeps the matched stop string and the speculation counters", () => {
    const speculation = {
      cycles: 2,
      draftRuns: 6,
      drafted: 6,
      accepted: 4,
      delivered: 6,
      acceptedHistogram: [0, 0, 2, 0],
    };
    const stop: ChatStop = {
      reason: "stop-string",
      stopString: "END",
      tokens: 7,
      speculation,
    };
    expect(stopRecordOf(stop)).toEqual({
      reason: "stop-string",
      tokens: 7,
      stopString: "END",
      speculation,
    });
  });

  it("has no token for a max-tokens or aborted stop", () => {
    expect(stopRecordOf({ reason: "max-tokens", tokens: 64 })).toEqual({
      reason: "max-tokens",
      tokens: 64,
    });
    expect(stopRecordOf({ reason: "aborted", tokens: 5 })).toEqual({
      reason: "aborted",
      tokens: 5,
    });
  });

  it("keeps the decode run-ahead counters of a run-ahead generation, and has none otherwise", () => {
    const decodeAhead = { depth: 4, batches: 3, steps: 11 };
    expect(stopRecordOf({ reason: "eos", token: 106, tokens: 10, decodeAhead })).toEqual({
      reason: "eos",
      tokens: 10,
      token: 106,
      decodeAhead,
    });
    expect(Object.hasOwn(stopRecordOf({ reason: "max-tokens", tokens: 4 }), "decodeAhead")).toBe(
      false,
    );
  });
});
