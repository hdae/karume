import { describe, expect, it } from "vite-plus/test";

import { rate, turnTiming } from "../turn-timing.ts";

describe("turnTiming", () => {
  it("splits a turn into prefill, time to first token and decode, the decode ending at the last run", () => {
    const timing = turnTiming(
      { sentAt: 1000, prefillAt: [1100, 1200], firstTokenAt: 1250, lastRunAt: 2250, endAt: 2300 },
      { promptTokens: 1000, tokens: 51 },
    );
    expect(timing).toEqual({
      totalSeconds: 1.3,
      prefillSeconds: 0.2,
      prefillTokensPerSecond: 5000,
      ttftSeconds: 0.25,
      decodeTokens: 50,
      decodeSeconds: 1,
      decodeTokensPerSecond: 50,
    });
  });

  it("splits a generation without a run mark (no diagnostics), the decode ending at the stream's end", () => {
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

  it("falls back to the stream's end when no decode run reported", () => {
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

describe("rate", () => {
  it("is undefined for no time or no tokens rather than 0 or Infinity", () => {
    expect(rate(10, 0)).toBeUndefined();
    expect(rate(0, 1)).toBeUndefined();
    expect(rate(10, undefined)).toBeUndefined();
    expect(rate(10, 4)).toBe(2.5);
  });
});
