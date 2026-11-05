/**
 * The chat page's transcript and per-turn statistics. Pure, so `vp test` covers it.
 *
 * The transcript is derived from the session's history (`session.turns`), not mirrored: after every
 * turn the in-context entries are rebuilt from that history, so the bubbles always show exactly what
 * the session will send next. The page keeps only what the history does not hold: an id per entry,
 * the turn an entry belongs to (for its statistics), and the entries the overflow policy dropped
 * (shown dimmed).
 */

import type { ChatJsonObject, ChatMessage, ChatStop } from "@karume/models/gemma";

/** One tool call of an assistant entry: the function and its arguments as JSON text. */
export type TranscriptToolCall = {
  readonly id?: string;
  readonly name: string;
  readonly arguments: string;
};

export type TranscriptEntry = {
  readonly id: number;
  readonly role: ChatMessage["role"];
  readonly text: string;
  /** An assistant entry's thought (`reasoning` of the history message). */
  readonly thinking?: string;
  /** The calls of a tool-calling assistant entry. */
  readonly toolCalls?: readonly TranscriptToolCall[];
  /** A tool result's tool: the name of the call it answers (absent when that cannot be resolved). */
  readonly toolName?: string;
  /** Whether the session's history still holds the entry (false = dropped by the overflow policy). */
  readonly inContext: boolean;
  /** Index of the turn (its record) the entry came from; absent when it could not be attributed. */
  readonly turn?: number;
};

/**
 * What one turn sent: a user message (`text`), or the results of the tool calls the previous turn
 * ended on (`toolResults`, the contents in the order sent — a continuation of the same model turn).
 */
export type SentTurn =
  | {
      readonly turn: number;
      readonly text: string;
      /**
       * The streamed reply when it is the text the history keeps (a plain session: one assistant
       * entry, matched by text). `undefined` when the session streams the structure spellings (tools
       * or thinking): every entry after the message is then the turn's, by position.
       */
      readonly reply: string | undefined;
      /** Ids for the turn's user entry and its first reply entry. */
      readonly userId: number;
      readonly assistantId: number;
    }
  | { readonly turn: number; readonly toolResults: readonly string[] };

/** The session's history without the system message (the transcript does not show it). */
export const historyOf = (turns: readonly ChatMessage[]): readonly ChatMessage[] =>
  turns.filter((turn) => turn.role !== "system");

/**
 * Whether the session waits for tool results: its last history entry is an assistant message with
 * tool calls (`send` then throws until `sendToolResults` answers them).
 */
export const awaitingToolResults = (turns: readonly ChatMessage[]): boolean => {
  const last = turns.at(-1);
  return last?.role === "assistant" && (last.tool_calls?.length ?? 0) > 0;
};

const argumentsText = (args: ChatJsonObject | null): string => JSON.stringify(args ?? {});

/** What an entry shows of a history message (the history is the source of truth for all of it). */
const viewOf = (
  message: ChatMessage,
  history: readonly ChatMessage[],
): Pick<TranscriptEntry, "role" | "text" | "thinking" | "toolCalls" | "toolName"> => {
  switch (message.role) {
    case "assistant": {
      const thinking = message.reasoning ?? message.reasoning_content ?? "";
      const calls = message.tool_calls ?? [];
      return {
        role: message.role,
        text: message.content,
        ...(thinking === "" ? {} : { thinking }),
        ...(calls.length === 0
          ? {}
          : {
              toolCalls: calls.map((call) => ({
                ...(call.id === undefined ? {} : { id: call.id }),
                name: call.function.name,
                arguments: argumentsText(call.function.arguments),
              })),
            }),
      };
    }
    case "tool": {
      const named =
        message.tool_call_id === undefined
          ? undefined
          : history
              .flatMap((entry) => (entry.role === "assistant" ? (entry.tool_calls ?? []) : []))
              .find((call) => call.id === message.tool_call_id)?.function.name;
      const toolName = named ?? message.name;
      return {
        role: message.role,
        text: message.content,
        ...(toolName === undefined || toolName === "" ? {} : { toolName }),
      };
    }
    default:
      return { role: message.role, text: message.content };
  }
};

/**
 * What a history entry is matched against: a previous entry, or one of the turn's. `role: "reply"`
 * with no `text` matches any assistant or tool entry (a reply of a session that streams structure);
 * `id: undefined` takes a fresh id when it lines up.
 */
type Candidate = {
  readonly id: number | undefined;
  readonly role: ChatMessage["role"] | "reply";
  readonly text: string | undefined;
  readonly turn: number | undefined;
};

const lineUp = (candidate: Candidate | undefined, message: ChatMessage): candidate is Candidate => {
  if (candidate === undefined) return false;
  if (candidate.role === "reply") return message.role === "assistant" || message.role === "tool";
  return candidate.role === message.role && candidate.text === message.content;
};

/** The turn's own candidates, for `added` new history entries. */
const turnCandidates = (sent: SentTurn, added: number): readonly Candidate[] => {
  const replies = (count: number, firstId: number | undefined): Candidate[] =>
    Array.from({ length: Math.max(0, count) }, (_, index) => ({
      id: index === 0 ? firstId : undefined,
      role: "reply",
      text: undefined,
      turn: sent.turn,
    }));
  if ("toolResults" in sent) {
    return [
      ...sent.toolResults.map((content): Candidate => ({
        id: undefined,
        role: "tool",
        text: content,
        turn: sent.turn,
      })),
      ...replies(added - sent.toolResults.length, undefined),
    ];
  }
  const user: Candidate = { id: sent.userId, role: "user", text: sent.text, turn: sent.turn };
  return sent.reply === undefined
    ? [user, ...replies(added - 1, sent.assistantId)]
    : [user, { id: sent.assistantId, role: "assistant", text: sent.reply, turn: sent.turn }];
};

/**
 * The transcript after a turn, from the session's history after it.
 *
 * `dropped` is what the overflow policy removed during the turn (observed in the policy: its input
 * length minus its output length). With it the lengths alone tell whether the session kept the turn:
 * the history then holds the previous in-context entries minus `dropped`, plus what the turn sent and
 * its replies when kept, or neither when the session treated the turn as never having happened.
 * Each history entry is attributed, aligned from the end, to the entry it came from (by role and
 * text — what the page sent is matched by its text; a reply by its streamed text in a plain session,
 * by position in one that streams structure); the history's content is what is shown either way,
 * and an entry that does not line up keeps no turn. Previous in-context entries in front of the
 * aligned part left the history: they stay, marked out of context.
 */
export const deriveTranscript = (input: {
  readonly previous: readonly TranscriptEntry[];
  readonly history: readonly ChatMessage[];
  readonly sent: SentTurn;
  readonly dropped: number;
  /** A fresh id, for a history entry that lines up with nothing the page sent. */
  readonly newId: () => number;
}): { readonly entries: readonly TranscriptEntry[]; readonly kept: boolean } => {
  const { previous, history, sent, dropped, newId } = input;
  const archived = previous.filter((entry) => !entry.inContext);
  const inContext = previous.filter((entry) => entry.inContext);
  const added = history.length - (inContext.length - dropped);
  const kept = added > 0;
  const candidates: readonly Candidate[] = [
    ...inContext.map((entry): Candidate => ({
      id: entry.id,
      role: entry.role,
      text: entry.text,
      turn: entry.turn,
    })),
    ...(kept ? turnCandidates(sent, added) : []),
  ];
  const offset = candidates.length - history.length;
  const context = history.map((message, index): TranscriptEntry => {
    const candidate = candidates[offset + index];
    const view = viewOf(message, history);
    return lineUp(candidate, message)
      ? {
          id: candidate.id ?? newId(),
          ...view,
          inContext: true,
          ...(candidate.turn === undefined ? {} : { turn: candidate.turn }),
        }
      : { id: newId(), ...view, inContext: true };
  });
  const left = inContext
    .slice(0, Math.min(inContext.length, Math.max(0, offset)))
    .map((entry) => ({ ...entry, inContext: false }));
  return { entries: [...archived, ...left, ...context], kept };
};

/** Times of one turn, from `performance.now()`. */
export type TurnMarks = {
  /** When `send` was called. */
  readonly sentAt: number;
  /** When each prefill chunk was committed (`onPrefill`), in order. */
  readonly prefillAt: readonly number[];
  /** When the first generated token arrived (`onToken`; none when the first draw was a stop token). */
  readonly firstTokenAt?: number;
  /**
   * When the last decode / draft / verify run reported (`onRunDiagnostics` — in the synchronous
   * section right after the run's outputs were read, so it includes the run that drew the stop token).
   * Only when the pipeline was loaded with diagnostics; without them the decode ends at `endAt`.
   */
  readonly lastRunAt?: number;
  /** When the text stream ended. */
  readonly endAt: number;
};

/** The statistics of one turn (seconds at millisecond precision, rates at 0.1 tokens/s). */
export type TurnTiming = {
  readonly totalSeconds: number;
  /** From `send` to the last committed prefill chunk (includes overflow handling and the sequence allocation). */
  readonly prefillSeconds?: number;
  readonly prefillTokensPerSecond?: number;
  /** Time to first token: from `send` to the first generated token. */
  readonly ttftSeconds?: number;
  /** Tokens drawn after the first one (the first is sampled from the prefill's logits). */
  readonly decodeTokens: number;
  /** From the first token to the last decode run (with diagnostics) or the stream's end (without). */
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
 * The statistics of one turn. `tokens` is the stop's `tokens` (the generated count, a stop token
 * counted as one); `promptTokens` is the prompt the session passed to the sequence (`undefined` when
 * the turn failed before generating).
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

/** How a turn stopped, as the stop record keeps it (the reason, and the matched stop string). */
export type StopFacts = {
  readonly reason: ChatStop["reason"];
  readonly tokens: number;
  readonly stopString?: string;
};

/** How a turn stopped, in a few words. */
export const stopLabel = (stop: StopFacts): string => {
  switch (stop.reason) {
    case "eos":
      return "end of turn";
    case "stop-token":
      return "stop token";
    case "stop-string":
      return `stop string "${stop.stopString ?? ""}"`;
    case "max-tokens":
      return "max new tokens reached";
    case "aborted":
      return "stopped";
    case "closed":
      return "closed";
  }
};

/**
 * What the page says after a turn the session did not keep: one that produced no text and did not
 * close with EOS leaves neither its message nor a reply in the history (`stop` = `undefined`: the
 * turn failed).
 */
export const notKeptNotice = (turnNumber: number, stop: StopFacts | undefined): string => {
  const what =
    stop === undefined
      ? "failed before any text (the error is shown below)"
      : stop.reason === "aborted"
        ? "was stopped before any text"
        : `ended without any text (${stopLabel(stop)})`;
  return `Turn ${turnNumber} ${what}, so the session did not add it to the history. The message is back in the input.`;
};

/** The footer line of an assistant turn (only the figures that were measured). */
export const turnSummary = (
  stop: StopFacts | undefined,
  promptTokens: number | undefined,
  timing: TurnTiming,
): string =>
  [
    stop === undefined ? "failed" : stopLabel(stop),
    stop === undefined ? undefined : `${stop.tokens} tokens`,
    promptTokens === undefined
      ? undefined
      : `prefill ${promptTokens} tokens${
          timing.prefillTokensPerSecond === undefined
            ? ""
            : ` at ${timing.prefillTokensPerSecond} tokens/s`
        }`,
    timing.ttftSeconds === undefined ? undefined : `TTFT ${timing.ttftSeconds.toFixed(2)} s`,
    timing.decodeTokensPerSecond === undefined
      ? undefined
      : `decode ${timing.decodeTokensPerSecond} tokens/s`,
    `${timing.totalSeconds.toFixed(2)} s`,
  ]
    .filter((part) => part !== undefined)
    .join(" · ");

/** How full the context is: the KV the session holds, or the history it re-renders on the next send. */
export type ContextUsage =
  | { readonly kind: "kv"; readonly used: number; readonly capacity: number }
  | { readonly kind: "history"; readonly tokens: number; readonly capacity: number };

export const contextUsageText = (usage: ContextUsage): string => {
  const percent = (value: number): string => `${((value / usage.capacity) * 100).toFixed(1)}%`;
  return usage.kind === "kv"
    ? `${usage.used} / ${usage.capacity} positions (${percent(usage.used)}, KV carried to the next turn)`
    : `${usage.tokens} / ${usage.capacity} tokens (${percent(usage.tokens)}, the history is re-rendered on the next send)`;
};
