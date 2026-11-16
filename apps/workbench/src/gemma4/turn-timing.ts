/**
 * The statistics of one Gemma 4 chat from its `performance.now()` marks. Copied from the
 * Playground's chat page (`apps/playground/src/models/gemma4/turns.ts`, `turnTiming`), which keeps
 * its own: the workbench and the Playground share no module. The workbench builds no pipeline with
 * `onRunDiagnostics`, so the decode window always ends at the stream's end (the chat page's
 * `lastRunAt` is left out). Pure, so `vp test` covers it.
 */

import { seconds as secondsOf } from "../record.ts";

/** Times of one chat, from `performance.now()`. */
export type TurnMarks = {
  /** When `chat` was called. */
  readonly sentAt: number;
  /** When each prefill chunk was committed (`onPrefill`), in order. */
  readonly prefillAt: readonly number[];
  /** When the first generated token arrived (`onToken`; none when the first draw was a stop token). */
  readonly firstTokenAt?: number;
  /** When the text stream ended. */
  readonly endAt: number;
};

/** The statistics of one chat (seconds at millisecond precision, rates at 0.1 tokens/s). */
export type TurnTiming = {
  readonly totalSeconds: number;
  /** From `chat` to the last committed prefill chunk (includes the sequence allocation). */
  readonly prefillSeconds?: number;
  readonly prefillTokensPerSecond?: number;
  /** Time to first token: from `chat` to the first generated token. */
  readonly ttftSeconds?: number;
  /** Tokens drawn after the first one (the first is sampled from the prefill's logits). */
  readonly decodeTokens: number;
  /** From the first token to the stream's end. */
  readonly decodeSeconds?: number;
  readonly decodeTokensPerSecond?: number;
};

/** `count / seconds` at 0.1 precision; `undefined` when it cannot be measured (no time, no tokens). */
const rate = (count: number, seconds: number | undefined): number | undefined => {
  if (seconds === undefined || !(seconds > 0) || !(count > 0)) return undefined;
  return Number((count / seconds).toFixed(1));
};

/**
 * The statistics of one chat. `tokens` is the stop's `tokens` (the generated count, a stop token
 * counted as one); `promptTokens` is the prompt passed to the sequence (`undefined` when the chat
 * failed before generating).
 */
export const turnTiming = (
  marks: TurnMarks,
  counts: { readonly promptTokens?: number; readonly tokens: number },
): TurnTiming => {
  const prefillEnd = marks.prefillAt.at(-1) ?? marks.firstTokenAt;
  const prefillSeconds =
    prefillEnd === undefined ? undefined : secondsOf(prefillEnd - marks.sentAt);
  const decodeTokens = Math.max(0, counts.tokens - 1);
  const decodeSeconds =
    marks.firstTokenAt === undefined || decodeTokens === 0
      ? undefined
      : secondsOf(marks.endAt - marks.firstTokenAt);
  const prefillTokensPerSecond =
    counts.promptTokens === undefined ? undefined : rate(counts.promptTokens, prefillSeconds);
  const decodeTokensPerSecond = rate(decodeTokens, decodeSeconds);
  return {
    totalSeconds: secondsOf(marks.endAt - marks.sentAt),
    ...(prefillSeconds === undefined ? {} : { prefillSeconds }),
    ...(prefillTokensPerSecond === undefined ? {} : { prefillTokensPerSecond }),
    ...(marks.firstTokenAt === undefined
      ? {}
      : { ttftSeconds: secondsOf(marks.firstTokenAt - marks.sentAt) }),
    decodeTokens,
    ...(decodeSeconds === undefined ? {} : { decodeSeconds }),
    ...(decodeTokensPerSecond === undefined ? {} : { decodeTokensPerSecond }),
  };
};
