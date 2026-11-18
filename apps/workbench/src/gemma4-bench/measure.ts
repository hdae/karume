/**
 * The figures of one benchmark chat, from the marks the runner takes and (with the tap on) the
 * passes and submits the tap recorded. Pure, so `vp test` covers it.
 *
 * The decode window runs from the first generated token (`onToken`) to the end of the text stream,
 * over `tokens − 1` decode tokens — the first token is sampled from the prefill's logits, and a stop
 * token counts as one (`turnTiming`, whose figures the record also keeps). The tap's
 * decode window starts at the passes and submits recorded when the first token arrived and runs to
 * the end of the chat, so it includes the run that drew a stop token, as the wall-clock window does.
 * An encode that runs ahead of its token moves at most one step across a boundary.
 *
 * With the tap split (one timed pass per dispatch, labelled with its pipeline key), the decode window
 * is also broken down per key ({@link KeyPerToken}).
 *
 * With the tap in the host mode (wall-clock marks at the WebGPU calls), every decode token's
 * interval is cut into the host's segments instead ({@link HostFigures}).
 */

import { type TurnTiming, turnTiming } from "@karume/gemma4-app-support/turn-timing";

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
  /** The pipeline label of its dispatch (split only). */
  readonly label?: string;
  readonly beginNs?: bigint;
  readonly endNs?: bigint;
};

/** What the tap recorded over one chat (read after the chat). */
export type TapCapture = {
  readonly passes: readonly PassTime[];
  readonly submits: number;
  /** Whether the tap ran split (one pass per dispatch): the decode window is then broken down per key. */
  readonly split?: boolean;
};

/** One pipeline key's share of the decode window, per decode token. */
export type KeyPerToken = {
  /** The pipeline label ({@link UNLABELED_KEY} for passes without one). */
  readonly key: string;
  readonly dispatches: number;
  /** GPU ms per token; absent when any of the key's passes went untimed. */
  readonly gpuMs?: number;
};

/** The key of passes the tap recorded without a pipeline label (no dispatch in the pass). */
export const UNLABELED_KEY = "(unlabeled)";

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
  /**
   * The decode window per pipeline key, GPU ms per token descending (split only; absent without
   * decode tokens). Its GPU ms sum to `perToken.gpuBusyMs` up to rounding.
   */
  readonly perKey?: readonly KeyPerToken[];
};

/** One WebGPU call the tap's host mode marked (`HostMark` of `tap.ts`, on the marks' clock). */
export type HostCall = {
  readonly kind:
    | "write"
    | "encoder"
    | "submit"
    | "submitted"
    | "map"
    | "mapped"
    | "range"
    | "unmap";
  readonly at: number;
  /** `range` only: the bytes of the mapped range. */
  readonly bytes?: number;
};

/** What the host mode recorded over one chat. */
export type HostCapture = {
  /** In call order. */
  readonly calls: readonly HostCall[];
  /**
   * When the chat's signal was checked (`throwIfAborted`), in order; empty for a chat sent without a
   * signal. The generation loop checks it at the boundary before a run — after a yield to the
   * event loop only when the run before it did not let the event loop turn — and again right
   * before it issues the run, so the first and the last check before a token's first GPU call
   * bound the boundary's check and the derived inputs.
   */
  readonly signalChecks: readonly number[];
};

/**
 * The segments of a decode token's interval (from the previous token's `onToken` to its own), in
 * order; they add up to the interval.
 *
 * - `between`: to the signal's first check — the previous token's delivery to the consumer and the
 *   generation loop's yield to the event loop where its boundary takes one. Without a signal there
 *   is no check and no yield, and it runs to the first GPU call: the delivery and the derived
 *   inputs.
 * - `derive`: from the signal's first check to its last before the first GPU call — the inputs the
 *   pipeline derives on the host (the PLE rows). 0 without a signal (it is in `between`).
 * - `start`: to the return of the interval's first submit — preparing the run, the queue writes and
 *   the first chunk of passes. The GPU has no work of this token until it ends.
 * - `encode`: to the return of the interval's last submit — the remaining chunks, encoded while the
 *   GPU runs the ones before them.
 * - `wait`: to the settlement of the last read-back map — the GPU finishing the run.
 * - `read`: to the last `unmap` — the mapped ranges copied to the host.
 * - `draw`: to `onToken` — the selection or sampling of the token and its way to the consumer.
 *
 * An interval without a submit (a token handed out of a run-ahead batch already read back) is all
 * `between`.
 */
export const HOST_SEGMENTS = [
  "between",
  "derive",
  "start",
  "encode",
  "wait",
  "read",
  "draw",
] as const;
export type HostSegment = (typeof HOST_SEGMENTS)[number];

/** The host's figures per decode token. */
export type HostFigures = {
  /** Decode intervals cut (the decode tokens; absent figures when 0). */
  readonly intervals: number;
  /** Mean ms per decode token of every segment; they add up to the decode ms per token. */
  readonly meanMs: Readonly<Record<HostSegment, number>>;
  /** The median over the intervals of every segment (robust to one slow token). */
  readonly medianMs: Readonly<Record<HostSegment, number>>;
  /** Calls per decode token. */
  readonly submits: number;
  readonly queueWrites: number;
  readonly maps: number;
  /** Bytes of the mapped ranges the host asked for, per decode token. */
  readonly mappedBytes: number;
};

export type ChatMeasurement = {
  /** `turnTiming`'s statistics (seconds at ms precision, rates at 0.1 tokens/s). */
  readonly timing: TurnTiming;
  readonly decodeTokens: number;
  /** Unrounded figures, which the summary aggregates (absent when not measurable). */
  readonly decodeMsPerToken?: number;
  readonly decodeTokensPerSecond?: number;
  /**
   * The median of the intervals between consecutive delivered tokens (absent with fewer than two):
   * what a token usually takes, unmoved by the few that hit a transiently slow kernel — the figure
   * to compare two arms by when their difference is a few percent. Not a rate: tokens handed out of
   * one run-ahead batch are no interval apart, so it says nothing about an arm that runs ahead.
   */
  readonly tokenIntervalMsMedian?: number;
  readonly ttftMs?: number;
  readonly prefillTokensPerSecond?: number;
  /** Absent with the tap off and in the host mode. */
  readonly tap?: TapFigures;
  /** The host's segments (the tap's host mode only; absent without decode tokens). */
  readonly host?: HostFigures;
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

/**
 * The passes per pipeline key, per decode token: dispatches, and GPU ms when every pass of the key
 * was timed. GPU ms descending (an untimed key last), then by key.
 */
export const perKeyFigures = (
  passes: readonly PassTime[],
  decodeTokens: number,
): readonly KeyPerToken[] => {
  const byKey = new Map<string, PassTime[]>();
  for (const pass of passes) {
    const key = pass.label === undefined || pass.label === "" ? UNLABELED_KEY : pass.label;
    const group = byKey.get(key);
    if (group === undefined) byKey.set(key, [pass]);
    else group.push(pass);
  }
  const figures = [...byKey].map(([key, group]): KeyPerToken => {
    const window = gpuWindow(group, 0);
    return {
      key,
      dispatches: round4(window.dispatches / decodeTokens),
      ...(window.busyMs === undefined ? {} : { gpuMs: round4(window.busyMs / decodeTokens) }),
    };
  });
  return figures.toSorted(
    (left, right) => (right.gpuMs ?? -1) - (left.gpuMs ?? -1) || left.key.localeCompare(right.key),
  );
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
  const perKey =
    tap.split === true && first !== undefined
      ? perKeyFigures(tap.passes.slice(first.passes), decodeTokens)
      : undefined;
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
    ...(perKey === undefined ? {} : { perKey }),
  };
};

const zeroSegments = (): Record<HostSegment, number> => ({
  between: 0,
  derive: 0,
  start: 0,
  encode: 0,
  wait: 0,
  read: 0,
  draw: 0,
});

/** The last element of `values` that `accept` takes, scanning from the end. */
const lastWhere = <Value>(
  values: readonly Value[],
  accept: (value: Value) => boolean,
): Value | undefined => {
  for (let index = values.length - 1; index >= 0; index -= 1) {
    if (accept(values[index])) return values[index];
  }
  return undefined;
};

/** One interval `(from, to]` cut into its segments ({@link HOST_SEGMENTS}). */
const cutInterval = (
  from: number,
  to: number,
  calls: readonly HostCall[],
  checks: readonly number[],
): Record<HostSegment, number> => {
  const segments = zeroSegments();
  const submitted = lastWhere(calls, ({ kind }) => kind === "submitted");
  const first = calls.find(
    ({ kind }) => kind === "write" || kind === "encoder" || kind === "submit",
  );
  if (submitted === undefined || first === undefined) {
    segments.between = to - from;
    return segments;
  }
  const before = checks.filter((at) => at <= first.at);
  const firstCheck = before.at(0);
  const lastCheck = before.at(-1);
  const issueFrom = lastCheck ?? first.at;
  segments.between = (firstCheck ?? first.at) - from;
  segments.derive = firstCheck === undefined ? 0 : issueFrom - firstCheck;
  // The first submit's return is at or after `first` (a submit is one of the calls `first` is the
  // earliest of) and at or before the last one's.
  const started = calls.find(({ kind }) => kind === "submitted") ?? submitted;
  segments.start = started.at - issueFrom;
  segments.encode = submitted.at - started.at;
  // Each later mark is looked for after the one before it, so the segments stay in order.
  const mapped = lastWhere(calls, ({ kind, at }) => kind === "mapped" && at >= submitted.at);
  const waitEnd = mapped?.at ?? submitted.at;
  segments.wait = waitEnd - submitted.at;
  const unmapped = lastWhere(calls, ({ kind, at }) => kind === "unmap" && at >= waitEnd);
  const readEnd = unmapped?.at ?? waitEnd;
  segments.read = readEnd - waitEnd;
  segments.draw = to - readEnd;
  return segments;
};

/**
 * The host's figures of the decode window: one interval per decode token — between consecutive
 * tokens, and from the last token to the end of the chat when a run drew a token that was not
 * delivered (a stop token). The time after the last delivered token of a chat that ended at its
 * token limit (the stream closing) is in no interval. `undefined` without decode tokens.
 */
export const hostFigures = (
  host: HostCapture,
  marks: ChatMarks,
  decodeTokens: number,
): HostFigures | undefined => {
  const bounds = marks.tokens.map(({ at }) => at);
  if (bounds.length - 1 < decodeTokens) bounds.push(marks.endAt);
  const intervals = bounds.length - 1;
  if (decodeTokens === 0 || intervals < 1) return undefined;
  const perInterval: Record<HostSegment, number>[] = [];
  let call = 0;
  let check = 0;
  const { calls, signalChecks } = host;
  // The prefill's calls and checks come before the first token.
  while (call < calls.length && calls[call].at <= bounds[0]) call += 1;
  while (check < signalChecks.length && signalChecks[check] <= bounds[0]) check += 1;
  const windowFrom = call;
  for (let index = 1; index < bounds.length; index += 1) {
    const callFrom = call;
    const checkFrom = check;
    while (call < calls.length && calls[call].at <= bounds[index]) call += 1;
    while (check < signalChecks.length && signalChecks[check] <= bounds[index]) check += 1;
    perInterval.push(
      cutInterval(
        bounds[index - 1],
        bounds[index],
        calls.slice(callFrom, call),
        signalChecks.slice(checkFrom, check),
      ),
    );
  }
  const meanMs = zeroSegments();
  const medianMs = zeroSegments();
  for (const segment of HOST_SEGMENTS) {
    const values = perInterval.map((segments) => segments[segment]);
    meanMs[segment] = round4(values.reduce((sum, value) => sum + value, 0) / intervals);
    medianMs[segment] = round4(median(values) ?? 0);
  }
  const windowCalls = calls.slice(windowFrom, call);
  const count = (kind: HostCall["kind"]): number =>
    round4(windowCalls.filter((item) => item.kind === kind).length / intervals);
  return {
    intervals,
    meanMs,
    medianMs,
    submits: count("submit"),
    queueWrites: count("write"),
    maps: count("map"),
    mappedBytes: round4(windowCalls.reduce((sum, item) => sum + (item.bytes ?? 0), 0) / intervals),
  };
};

/**
 * The figures of one chat. `tokens` is the stop's `tokens` (generated tokens, a stop token counted
 * as one); `promptTokens` is the rendered prompt's length; `tap` is absent with the tap off and in
 * the host mode, `host` present only in the host mode.
 */
export const measureChat = (input: {
  readonly marks: ChatMarks;
  readonly promptTokens: number;
  readonly tokens: number;
  readonly tap?: TapCapture;
  readonly host?: HostCapture;
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
  const host = input.host === undefined ? undefined : hostFigures(input.host, marks, decodeTokens);
  const intervalMedian = median(
    marks.tokens.slice(1).map((token, index) => token.at - marks.tokens[index].at),
  );
  return {
    timing,
    decodeTokens,
    ...(decodeMsPerToken === undefined
      ? {}
      : {
          decodeMsPerToken: round4(decodeMsPerToken),
          decodeTokensPerSecond: round4(1000 / decodeMsPerToken),
        }),
    ...(intervalMedian === undefined ? {} : { tokenIntervalMsMedian: round4(intervalMedian) }),
    ...(firstTokenAt === undefined ? {} : { ttftMs: round4(firstTokenAt - marks.sentAt) }),
    ...(prefillMs === undefined || !(prefillMs > 0) || promptTokens === 0
      ? {}
      : { prefillTokensPerSecond: round4((promptTokens * 1000) / prefillMs) }),
    ...(tap === undefined ? {} : { tap: tapFigures(tap, marks, decodeTokens, decodeMsPerToken) }),
    ...(host === undefined ? {} : { host }),
  };
};
