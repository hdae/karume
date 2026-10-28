import type { ChatMessage } from "@karume/models/gemma";
import { describe, expect, it } from "vite-plus/test";

import {
  contextUsageText,
  deriveTranscript,
  historyOf,
  notKeptNotice,
  rate,
  type SentTurn,
  stopLabel,
  type TranscriptEntry,
  turnSummary,
  turnTiming,
} from "../src/models/gemma4/turns.ts";

const user = (content: string): ChatMessage => ({ role: "user", content });
const assistant = (content: string): ChatMessage => ({ role: "assistant", content });

/** Ids from 100 up, for history entries that line up with nothing the page sent. */
const idSource = () => {
  let next = 100;
  return () => (next += 1);
};

const sent = (turn: number, text: string, reply: string): SentTurn => ({
  turn,
  text,
  reply,
  userId: turn * 2 + 1,
  assistantId: turn * 2 + 2,
});

/** The transcript after two kept turns ("Hi" → "Hello.", "Again" → "Hello again."). */
const twoTurns = (): readonly TranscriptEntry[] => {
  const first = deriveTranscript({
    previous: [],
    history: [user("Hi"), assistant("Hello.")],
    sent: sent(0, "Hi", "Hello."),
    dropped: 0,
    newId: idSource(),
  });
  return deriveTranscript({
    previous: first.entries,
    history: [user("Hi"), assistant("Hello."), user("Again"), assistant("Hello again.")],
    sent: sent(1, "Again", "Hello again."),
    dropped: 0,
    newId: idSource(),
  }).entries;
};

const shape = (entries: readonly TranscriptEntry[]) =>
  entries.map(({ role, text, inContext, turn }) => ({ role, text, inContext, turn }));

describe("historyOf", () => {
  it("leaves out the system message the session puts first", () => {
    expect(historyOf([{ role: "system", content: "Be brief." }, user("Hi")])).toEqual([user("Hi")]);
  });
});

describe("deriveTranscript (the transcript rebuilt from the session's history)", () => {
  it("attributes a kept turn's message and reply to the turn", () => {
    const { entries, kept } = deriveTranscript({
      previous: [],
      history: [user("Hi"), assistant("")],
      sent: sent(0, "Hi", ""),
      dropped: 0,
      newId: idSource(),
    });
    expect(kept).toBe(true);
    expect(entries).toEqual([
      { id: 1, role: "user", text: "Hi", inContext: true, turn: 0 },
      { id: 2, role: "assistant", text: "", inContext: true, turn: 0 },
    ]);
  });

  it("adds nothing when the session did not keep the turn", () => {
    const { entries, kept } = deriveTranscript({
      previous: twoTurns(),
      history: [user("Hi"), assistant("Hello."), user("Again"), assistant("Hello again.")],
      sent: sent(2, "Third", ""),
      dropped: 0,
      newId: idSource(),
    });
    expect(kept).toBe(false);
    expect(entries).toEqual(twoTurns());
  });

  it("tells a repeated turn from a rolled-back one by the lengths, not by the text", () => {
    const previous = deriveTranscript({
      previous: [],
      history: [user("hi"), assistant("")],
      sent: sent(0, "hi", ""),
      dropped: 0,
      newId: idSource(),
    }).entries;
    const rolledBack = deriveTranscript({
      previous,
      history: [user("hi"), assistant("")],
      sent: sent(1, "hi", ""),
      dropped: 0,
      newId: idSource(),
    });
    expect(rolledBack.kept).toBe(false);
    expect(rolledBack.entries).toEqual(previous);
    const repeated = deriveTranscript({
      previous,
      history: [user("hi"), assistant(""), user("hi"), assistant("")],
      sent: sent(1, "hi", ""),
      dropped: 0,
      newId: idSource(),
    });
    expect(repeated.kept).toBe(true);
    expect(repeated.entries.map(({ turn }) => turn)).toEqual([0, 0, 1, 1]);
  });

  it("keeps the entries the overflow policy dropped, out of context and in order", () => {
    const { entries, kept } = deriveTranscript({
      previous: twoTurns(),
      history: [user("Again"), assistant("Hello again."), user("Third"), assistant("Sure.")],
      sent: sent(2, "Third", "Sure."),
      dropped: 2,
      newId: idSource(),
    });
    expect(kept).toBe(true);
    expect(shape(entries)).toEqual([
      { role: "user", text: "Hi", inContext: false, turn: 0 },
      { role: "assistant", text: "Hello.", inContext: false, turn: 0 },
      { role: "user", text: "Again", inContext: true, turn: 1 },
      { role: "assistant", text: "Hello again.", inContext: true, turn: 1 },
      { role: "user", text: "Third", inContext: true, turn: 2 },
      { role: "assistant", text: "Sure.", inContext: true, turn: 2 },
    ]);
    const later = deriveTranscript({
      previous: entries,
      history: [user("Third"), assistant("Sure."), user("Fourth"), assistant("Ok.")],
      sent: sent(3, "Fourth", "Ok."),
      dropped: 2,
      newId: idSource(),
    });
    expect(shape(later.entries).map(({ text, inContext }) => [text, inContext])).toEqual([
      ["Hi", false],
      ["Hello.", false],
      ["Again", false],
      ["Hello again.", false],
      ["Third", true],
      ["Sure.", true],
      ["Fourth", true],
      ["Ok.", true],
    ]);
  });

  it("marks dropped entries even when the turn itself was then not kept", () => {
    const { entries, kept } = deriveTranscript({
      previous: twoTurns(),
      history: [user("Again"), assistant("Hello again.")],
      sent: sent(2, "Third", ""),
      dropped: 2,
      newId: idSource(),
    });
    expect(kept).toBe(false);
    expect(entries.map(({ inContext }) => inContext)).toEqual([false, false, true, true]);
  });

  it("shows the history's text even where it does not line up with what the page sent", () => {
    const { entries } = deriveTranscript({
      previous: [],
      history: [user("Hi"), assistant("Hello!")],
      sent: sent(0, "Hi", "Hello."),
      dropped: 0,
      newId: idSource(),
    });
    expect(entries.at(-1)).toEqual({
      id: 101,
      role: "assistant",
      text: "Hello!",
      inContext: true,
    });
  });
});

describe("notKeptNotice", () => {
  it("says why the message is not in the history", () => {
    expect(notKeptNotice(3, { reason: "aborted", tokens: 0 })).toBe(
      "Turn 3 was stopped before any text, so the session did not add it to the history. The message is back in the input.",
    );
    expect(notKeptNotice(1, undefined)).toMatch(/^Turn 1 failed before any text/);
    expect(notKeptNotice(2, { reason: "stop-string", tokens: 2, stopString: "END" })).toMatch(
      /^Turn 2 ended without any text \(stop string "END"\)/,
    );
  });
});

describe("turnTiming", () => {
  it("splits a turn into prefill, time to first token and decode", () => {
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

describe("labels", () => {
  it("names each stop reason", () => {
    expect(stopLabel({ reason: "eos", tokens: 3 })).toBe("end of turn");
    expect(stopLabel({ reason: "max-tokens", tokens: 3 })).toBe("max new tokens reached");
    expect(stopLabel({ reason: "aborted", tokens: 3 })).toBe("stopped");
    expect(stopLabel({ reason: "stop-string", tokens: 3, stopString: "END" })).toBe(
      'stop string "END"',
    );
  });

  it("summarizes only the measured figures of a turn", () => {
    const timing = turnTiming(
      { sentAt: 0, prefillAt: [200], firstTokenAt: 250, lastRunAt: 1250, endAt: 1300 },
      { promptTokens: 20, tokens: 21 },
    );
    expect(turnSummary({ reason: "eos", tokens: 21 }, 20, timing)).toBe(
      "end of turn · 21 tokens · prefill 20 tokens at 100 tokens/s · TTFT 0.25 s · decode 20 tokens/s · 1.30 s",
    );
    expect(
      turnSummary(
        undefined,
        undefined,
        turnTiming({ sentAt: 0, prefillAt: [], endAt: 10 }, { tokens: 0 }),
      ),
    ).toBe("failed · 0.01 s");
  });

  it("says whether the context is the carried KV or the re-rendered history", () => {
    expect(contextUsageText({ kind: "kv", used: 1024, capacity: 4096 })).toBe(
      "1024 / 4096 positions (25.0%, KV carried to the next turn)",
    );
    expect(contextUsageText({ kind: "history", tokens: 512, capacity: 4096 })).toMatch(
      /^512 \/ 4096 tokens \(12\.5%, the history is re-rendered/,
    );
  });
});
