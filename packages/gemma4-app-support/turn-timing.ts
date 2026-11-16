/**
 * `@karume/gemma4-app-support/turn-timing` — the statistics of one Gemma 4 generation (a turn of the
 * playground's chat page, a chat of the workbench's decode bench) from its `performance.now()`
 * marks. Pure, so `vp test` covers it.
 *
 * The one difference between the two: only a pipeline built with `onRunDiagnostics` reports when its
 * last decode run ended (the chat page's diagnostics; the workbench builds none). With that mark
 * (`lastRunAt`) the decode window ends there; without it, at the stream's end.
 */

/** Times of one generation, from `performance.now()`. */
export type TurnMarks = {
  /** When it was sent (`send` of a `ChatSession`, `chat` of the pipeline). */
  readonly sentAt: number;
  /** When each prefill chunk was committed (`onPrefill`), in order. */
  readonly prefillAt: readonly number[];
  /** When the first generated token arrived (`onToken`; none when the first draw was a stop token). */
  readonly firstTokenAt?: number;
  /**
   * When the last decode / draft / verify run reported (`onRunDiagnostics` — in the synchronous
   * section right after the run's outputs were read, so it includes the run that drew the stop token).
   * Only when the pipeline was built with diagnostics; without them the decode ends at `endAt`.
   */
  readonly lastRunAt?: number;
  /** When the text stream ended. */
  readonly endAt: number;
};

/** The statistics of one generation (seconds at millisecond precision, rates at 0.1 tokens/s). */
export type TurnTiming = {
  readonly totalSeconds: number;
  /**
   * From the send to the last committed prefill chunk (includes the session's overflow handling, if
   * any, and the sequence allocation).
   */
  readonly prefillSeconds?: number;
  readonly prefillTokensPerSecond?: number;
  /** Time to first token: from the send to the first generated token. */
  readonly ttftSeconds?: number;
  /** Tokens drawn after the first one (the first is sampled from the prefill's logits). */
  readonly decodeTokens: number;
  /** From the first token to the last decode run (with `lastRunAt`) or the stream's end (without). */
  readonly decodeSeconds?: number;
  readonly decodeTokensPerSecond?: number;
};

const secondsOf = (milliseconds: number): number => Number((milliseconds / 1000).toFixed(3));

/** `count / seconds` at 0.1 precision; `undefined` when it cannot be measured (no time, no tokens). */
export const rate = (count: number, seconds: number | undefined): number | undefined => {
  if (seconds === undefined || !(seconds > 0) || !(count > 0)) return undefined;
  return Number((count / seconds).toFixed(1));
};

/**
 * The statistics of one generation. `tokens` is the stop's `tokens` (the generated count, a stop
 * token counted as one); `promptTokens` is the prompt passed to the sequence (`undefined` when it
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
      : secondsOf((marks.lastRunAt ?? marks.endAt) - marks.firstTokenAt);
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
