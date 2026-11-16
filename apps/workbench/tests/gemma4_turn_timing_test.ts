// Cases copied from the Playground's tests of its Gemma 4 chat page turns (apps/playground/tests/
// gemma4_turns_test.ts), for the function the workbench copied (src/gemma4/turn-timing.ts). The
// copy has no `lastRunAt`, so the first case ends its decode at the stream's end.

import { describe, expect, it } from "vite-plus/test";

import { turnTiming } from "../src/gemma4/turn-timing.ts";

describe("turnTiming", () => {
  it("splits a turn into prefill, time to first token and decode", () => {
    const timing = turnTiming(
      { sentAt: 1000, prefillAt: [1100, 1200], firstTokenAt: 1250, endAt: 2250 },
      { promptTokens: 1000, tokens: 51 },
    );
    expect(timing).toEqual({
      totalSeconds: 1.25,
      prefillSeconds: 0.2,
      prefillTokensPerSecond: 5000,
      ttftSeconds: 0.25,
      decodeTokens: 50,
      decodeSeconds: 1,
      decodeTokensPerSecond: 50,
    });
  });

  it("times the decode from the first token to the stream's end", () => {
    const timing = turnTiming(
      { sentAt: 0, prefillAt: [100], firstTokenAt: 100, endAt: 600 },
      { promptTokens: 10, tokens: 11 },
    );
    expect(timing.decodeSeconds).toBe(0.5);
    expect(timing.decodeTokensPerSecond).toBe(20);
  });

  it("measures no decode when the first draw was a stop token", () => {
    const timing = turnTiming(
      { sentAt: 0, prefillAt: [100], endAt: 120 },
      { promptTokens: 10, tokens: 1 },
    );
    expect(timing).toEqual({
      totalSeconds: 0.12,
      prefillSeconds: 0.1,
      prefillTokensPerSecond: 100,
      decodeTokens: 0,
    });
  });

  it("measures no prefill rate when the turn failed before generating", () => {
    const timing = turnTiming({ sentAt: 0, prefillAt: [], endAt: 5 }, { tokens: 0 });
    expect(timing).toEqual({ totalSeconds: 0.005, decodeTokens: 0 });
  });
});
