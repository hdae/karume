import { describe, expect, it } from "vite-plus/test";

import {
  type ChatMarks,
  gpuWindow,
  measureChat,
  type PassTime,
  type TokenMark,
} from "../src/models/gemma4-bench/measure.ts";

/** A pass of `ms` GPU time starting at `atMs` (timestamps in ns, as the tap reads them). */
const pass = (atMs: number, ms: number, dispatches = 10): PassTime => ({
  dispatches,
  beginNs: BigInt(Math.round(atMs * 1e6)),
  endNs: BigInt(Math.round((atMs + ms) * 1e6)),
});

/**
 * A chat sent at 0: prefill in two chunks (20, 30 ms), the first token at 40 ms, then one token every
 * 10 ms (`tokens` marks in all), the stream ending 10 ms after the last mark. With the tap, the
 * prefill ran 3 passes in 2 submits, and every decode step 2 passes (4 + 3 ms GPU) in 1 submit.
 */
const chat = (tokens: number) => {
  const marks: TokenMark[] = [];
  const passes: PassTime[] = [pass(1, 5), pass(10, 5), pass(25, 5)];
  let submits = 2;
  for (let index = 0; index < tokens; index += 1) {
    marks.push({ at: 40 + index * 10, passes: passes.length, submits });
    // The step that draws the next token (after the last mark: the one that draws the stop token).
    passes.push(pass(41 + index * 10, 4), pass(46 + index * 10, 3));
    submits += 1;
  }
  const chatMarks: ChatMarks = {
    sentAt: 0,
    prefillAt: [20, 30],
    tokens: marks,
    endAt: 40 + tokens * 10,
  };
  return { marks: chatMarks, tap: { passes, submits } };
};

describe("measuring a chat", () => {
  describe("from the stream timings", () => {
    it("rates decode over the tokens after the first, from the first token to the stream's end", () => {
      const { marks } = chat(5);
      // stop.tokens 6: five streamed tokens plus the stop token; 5 decode tokens over 50 ms.
      const measured = measureChat({ marks, promptTokens: 21, tokens: 6 });
      expect(measured.decodeTokens).toBe(5);
      expect(measured.decodeMsPerToken).toBe(10);
      expect(measured.decodeTokensPerSecond).toBe(100);
    });

    it("keeps TTFT and prefill apart from decode", () => {
      const measured = measureChat({ marks: chat(5).marks, promptTokens: 21, tokens: 6 });
      expect(measured.ttftMs).toBe(40);
      // 21 tokens to the last prefill chunk at 30 ms.
      expect(measured.prefillTokensPerSecond).toBe(700);
    });

    it("keeps the chat page's statistics alongside", () => {
      const measured = measureChat({ marks: chat(5).marks, promptTokens: 21, tokens: 6 });
      expect(measured.timing).toMatchObject({
        prefillSeconds: 0.03,
        ttftSeconds: 0.04,
        decodeTokens: 5,
        decodeSeconds: 0.05,
        decodeTokensPerSecond: 100,
      });
    });

    it("measures no decode for a chat that drew a single token", () => {
      const measured = measureChat({ marks: chat(1).marks, promptTokens: 21, tokens: 1 });
      expect(measured.decodeTokens).toBe(0);
      expect(measured.decodeMsPerToken).toBeUndefined();
      expect(measured.decodeTokensPerSecond).toBeUndefined();
    });

    it("leaves the tap's figures out when the tap was off", () => {
      expect(
        measureChat({ marks: chat(5).marks, promptTokens: 21, tokens: 6 }).tap,
      ).toBeUndefined();
    });
  });

  describe("from the tap", () => {
    it("splits the passes and submits at the first token into prefill and decode", () => {
      const { marks, tap } = chat(5);
      const figures = measureChat({ marks, promptTokens: 21, tokens: 6, tap }).tap;
      expect(figures?.prefill).toEqual({
        passes: 3,
        dispatches: 30,
        submits: 2,
        busyMs: 15,
        untimedPasses: 0,
      });
      // Five steps after the first token, the one that drew the stop token included.
      expect(figures?.decode).toEqual({
        passes: 10,
        dispatches: 100,
        submits: 5,
        busyMs: 35,
        untimedPasses: 0,
      });
    });

    it("reports GPU busy, counts and the wall − GPU gap per decode token", () => {
      const { marks, tap } = chat(5);
      const perToken = measureChat({ marks, promptTokens: 21, tokens: 6, tap }).tap?.perToken;
      expect(perToken).toEqual({
        gpuBusyMs: 7,
        gpuBusyMsMedian: 7,
        wallMinusGpuMs: 3,
        dispatches: 20,
        passes: 2,
        submits: 1,
      });
    });

    it("takes the per-token median over the token intervals, so one slow step does not move it", () => {
      const { marks, tap } = chat(5);
      // The second decode step's first pass ran 45× slower (180 ms instead of 4).
      const passes = tap.passes.map((recorded, index) => (index === 5 ? pass(51, 180) : recorded));
      const perToken = measureChat({
        marks,
        promptTokens: 21,
        tokens: 6,
        tap: { ...tap, passes },
      }).tap?.perToken;
      expect(perToken?.gpuBusyMsMedian).toBe(7);
      expect(perToken?.gpuBusyMs).toBeCloseTo(42.2, 10);
    });

    it("gives no busy time when any pass of the window went untimed", () => {
      const { marks, tap } = chat(5);
      const passes = tap.passes.map((recorded, index) =>
        index === 7 ? { dispatches: recorded.dispatches } : recorded,
      );
      const figures = measureChat({
        marks,
        promptTokens: 21,
        tokens: 6,
        tap: { ...tap, passes },
      }).tap;
      expect(figures?.decode.busyMs).toBeUndefined();
      expect(figures?.decode.untimedPasses).toBe(1);
      expect(figures?.perToken?.gpuBusyMs).toBeUndefined();
      expect(figures?.perToken?.gpuBusyMsMedian).toBeUndefined();
      expect(figures?.perToken?.wallMinusGpuMs).toBeUndefined();
      // The counts do not need timestamps.
      expect(figures?.perToken?.dispatches).toBe(20);
    });
  });
});

describe("the GPU work of a window", () => {
  it("adds no time for a pair that came back equal or reversed", () => {
    const window = gpuWindow(
      [
        pass(0, 2),
        { dispatches: 1, beginNs: 5n, endNs: 5n },
        { dispatches: 1, beginNs: 9n, endNs: 3n },
      ],
      1,
    );
    expect(window.busyMs).toBe(2);
    expect(window.dispatches).toBe(12);
  });

  it("keeps sub-microsecond precision on timestamps far beyond 2^53 ns", () => {
    const base = 2n ** 62n;
    const window = gpuWindow([{ dispatches: 1, beginNs: base, endNs: base + 1_234_567n }], 0);
    expect(window.busyMs).toBe(1.2346);
  });
});
