/**
 * The figures of one benchmark chat, from the marks the runner takes and (with the tap on) the
 * passes and submits the tap recorded. Pure, so `vp test` covers it.
 *
 * The decode window runs from the first generated token (`onToken`) to the end of the text stream,
 * over `tokens − 1` decode tokens — the first token is sampled from the prefill's logits, and a stop
 * token counts as one (the chat page's `turnTiming`, whose figures the record also keeps). The tap's
 * decode window starts at the passes and submits recorded when the first token arrived and runs to
 * the end of the chat, so it includes the run that drew a stop token, as the wall-clock window does.
 * An encode that runs ahead of its token moves at most one step across a boundary.
 */

import { type TurnTiming, turnTiming } from "@/models/gemma4/turns";

/** What the tap had recorded when a token arrived. */
export type TokenMark = {
  /** `performance.now()` at `onToken`. */
  readonly at: number;
  /** Passes recorded so far (0 with the tap off). */
  readonly passes: number;
  /** Submits recorded so far (0 with the tap off). */
  readonly submits: number;
};

/** The marks of one chat (`performance.now()`). */
export type ChatMarks = {
  /** When `chat` was called. */
  readonly sentAt: number;
  /** When each prefill chunk was committed (`onPrefill`), in order. */
  readonly prefillAt: readonly number[];
  /** One per generated token except a stop token, in order. */
  readonly tokens: readonly TokenMark[];
  /** When the text stream ended. */
  readonly endAt: number;
};

/** One pass as the tap recorded it. */
export type PassTime = {
  readonly dispatches: number;
  readonly beginNs?: bigint;
  readonly endNs?: bigint;
};

/** What the tap recorded over one chat (read after the chat). */
export type TapCapture = { readonly passes: readonly PassTime[]; readonly submits: number };

/** The GPU work of one stretch of the chat. */
export type GpuWindow = {
  readonly passes: number;
  readonly dispatches: number;
  readonly submits: number;
  /** Σ(end − begin) over the passes; absent when any pass is untimed. */
  readonly busyMs?: number;
  /** Passes without a timestamp pair (no timestamp-query, or past the tap's slots). */
  readonly untimedPasses: number;
};

/** The tap's figures per decode token (the decode window ÷ the decode tokens). */
export type PerTokenTap = {
  readonly gpuBusyMs?: number;
  /** The median over the token-to-token intervals (robust to a transient slow kernel). */
  readonly gpuBusyMsMedian?: number;
  /** Decode wall ms per token − GPU busy ms per token: host time the GPU does not overlap. */
  readonly wallMinusGpuMs?: number;
  readonly dispatches: number;
  readonly passes: number;
  readonly submits: number;
};

export type TapFigures = {
  readonly prefill: GpuWindow;
  readonly decode: GpuWindow;
  /** Absent without decode tokens. */
  readonly perToken?: PerTokenTap;
};

export type ChatMeasurement = {
  /** The chat page's statistics (seconds at ms precision, rates at 0.1 tokens/s). */
  readonly timing: TurnTiming;
  readonly decodeTokens: number;
  /** Unrounded figures, which the summary aggregates (absent when not measurable). */
  readonly decodeMsPerToken?: number;
  readonly decodeTokensPerSecond?: number;
  readonly ttftMs?: number;
  readonly prefillTokensPerSecond?: number;
  /** Absent with the tap off. */
  readonly tap?: TapFigures;
};

/** Four decimals: the records stay readable and diff-friendly. */
export const round4 = (value: number): number => Number(value.toFixed(4));

const NS_PER_MS = 1_000_000n;

/** The GPU work of `passes` (`submits` is counted by the caller). */
export const gpuWindow = (passes: readonly PassTime[], submits: number): GpuWindow => {
  let busyNs = 0n;
  let dispatches = 0;
  let untimedPasses = 0;
  for (const pass of passes) {
    dispatches += pass.dispatches;
    if (pass.beginNs === undefined || pass.endNs === undefined) {
      untimedPasses += 1;
      continue;
    }
    // A pair can come back equal (timestamps quantized) or out of order; neither adds time.
    if (pass.endNs > pass.beginNs) busyNs += pass.endNs - pass.beginNs;
  }
  // Whole milliseconds as a bigint, the rest as a fraction: no precision is lost converting.
  const busyMs = Number(busyNs / NS_PER_MS) + Number(busyNs % NS_PER_MS) / 1e6;
  return {
    passes: passes.length,
    dispatches,
    submits,
    ...(untimedPasses === 0 ? { busyMs: round4(busyMs) } : {}),
    untimedPasses,
  };
};

/** The median of a non-empty list (an even count takes the mean of the middle two). */
export const median = (values: readonly number[]): number | undefined => {
  if (values.length === 0) return undefined;
  const sorted = values.toSorted((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const tapFigures = (
  tap: TapCapture,
  marks: ChatMarks,
  decodeTokens: number,
  decodeMsPerToken: number | undefined,
): TapFigures => {
  const first = marks.tokens[0];
  const prefill = gpuWindow(
    tap.passes.slice(0, first?.passes ?? tap.passes.length),
    first?.submits ?? tap.submits,
  );
  const decode =
    first === undefined
      ? gpuWindow([], 0)
      : gpuWindow(tap.passes.slice(first.passes), tap.submits - first.submits);
  if (decodeTokens === 0) return { prefill, decode };
  const intervals: number[] = [];
  for (let index = 1; index < marks.tokens.length; index += 1) {
    const interval = gpuWindow(
      tap.passes.slice(marks.tokens[index - 1].passes, marks.tokens[index].passes),
      0,
    );
    if (interval.busyMs !== undefined) intervals.push(interval.busyMs);
  }
  const gpuBusyMs = decode.busyMs === undefined ? undefined : decode.busyMs / decodeTokens;
  // Only when every interval was timed (a partly timed median would pick from a subset).
  const gpuBusyMsMedian =
    intervals.length === marks.tokens.length - 1 ? median(intervals) : undefined;
  return {
    prefill,
    decode,
    perToken: {
      ...(gpuBusyMs === undefined ? {} : { gpuBusyMs: round4(gpuBusyMs) }),
      ...(gpuBusyMsMedian === undefined ? {} : { gpuBusyMsMedian: round4(gpuBusyMsMedian) }),
      ...(gpuBusyMs === undefined || decodeMsPerToken === undefined
        ? {}
        : { wallMinusGpuMs: round4(decodeMsPerToken - gpuBusyMs) }),
      dispatches: round4(decode.dispatches / decodeTokens),
      passes: round4(decode.passes / decodeTokens),
      submits: round4(decode.submits / decodeTokens),
    },
  };
};

/**
 * The figures of one chat. `tokens` is the stop's `tokens` (generated tokens, a stop token counted
 * as one); `promptTokens` is the rendered prompt's length; `tap` is absent with the tap off.
 */
export const measureChat = (input: {
  readonly marks: ChatMarks;
  readonly promptTokens: number;
  readonly tokens: number;
  readonly tap?: TapCapture;
}): ChatMeasurement => {
  const { marks, promptTokens, tokens, tap } = input;
  const firstTokenAt = marks.tokens[0]?.at;
  const timing = turnTiming(
    {
      sentAt: marks.sentAt,
      prefillAt: marks.prefillAt,
      ...(firstTokenAt === undefined ? {} : { firstTokenAt }),
      endAt: marks.endAt,
    },
    { promptTokens, tokens },
  );
  const decodeTokens = timing.decodeTokens;
  const decodeMs = firstTokenAt === undefined ? undefined : marks.endAt - firstTokenAt;
  const decodeMsPerToken =
    decodeMs === undefined || decodeTokens === 0 || !(decodeMs > 0)
      ? undefined
      : decodeMs / decodeTokens;
  const prefillEnd = marks.prefillAt.at(-1) ?? firstTokenAt;
  const prefillMs = prefillEnd === undefined ? undefined : prefillEnd - marks.sentAt;
  return {
    timing,
    decodeTokens,
    ...(decodeMsPerToken === undefined
      ? {}
      : {
          decodeMsPerToken: round4(decodeMsPerToken),
          decodeTokensPerSecond: round4(1000 / decodeMsPerToken),
        }),
    ...(firstTokenAt === undefined ? {} : { ttftMs: round4(firstTokenAt - marks.sentAt) }),
    ...(prefillMs === undefined || !(prefillMs > 0) || promptTokens === 0
      ? {}
      : { prefillTokensPerSecond: round4((promptTokens * 1000) / prefillMs) }),
    ...(tap === undefined ? {} : { tap: tapFigures(tap, marks, decodeTokens, decodeMsPerToken) }),
  };
};
