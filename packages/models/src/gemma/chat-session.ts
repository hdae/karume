/**
 * `ChatSession` — the middle layer for carrying **an ordinary multi-turn chat**.
 *
 * The surface has three layers. From the lowest:
 *
 * 1. `Pipeline.sequence()` (ADR 0083 decisions 1–4) — the caller runs token ids and the KV
 *    lifetime itself
 * 2. **this** — holds the mapping between the conversation (a list of `ChatMessage`) and the
 *    KV, and truncates with an injected policy on overflow
 * 3. `Pipeline.chat()` — one turn = one sequence (past turns are re-rendered every time)
 *
 * With only 3, prefill becomes O(n²) as the conversation grows, and avoiding that drops straight
 * into the world of 1 (token ids / `<bos>` / the uncommitted frontier). This layer holds **only that
 * gap**: generation itself rides on 1, and rendering, decoding and stop strings ride on the same
 * implementation as `chat()` ({@link decodeChatChunks} / {@link chatStreamOf}) — the
 * `ChatStream` that `send` returns has the same type and the same contract as the return value
 * of `chat()`.
 *
 * ## When the KV is carried over
 *
 * Only when the previous turn **closed with the distribution's EOS** (`GenerationStop.reason ===
 * "eos"`) does the next turn carry on with a delta ({@link chatTurn}). The closing token was
 * emitted by generation and is held by the sequence as the uncommitted frontier (`pendingToken`);
 * the next `generate` prepends it to the prompt (ADR 0083 decision 4). After a turn cut off by
 * max-tokens / a full context (`capacity`) / a stop string / an interruption the model turn is not
 * closed, so the sequence is discarded and re-rendered from the history.
 *
 * NOTE: the distribution's EOS set contains `<eos>` and `<|tool_response>` as well
 * (`gemma4StopTokens`). Carrying over a turn closed by something other than `<turn|>` without a
 * tool call makes the spelling differ from the upstream template by one token, but the conversation
 * loses nothing (the delta always starts "right after the closing token").
 *
 * ## Tools, tool calls and thinking (ADR 0084 addendum 2026-11-04)
 *
 * {@link ChatSessionOptions.tools} and {@link ChatSessionOptions.enableThinking} are fixed for the
 * session: they shape the system block at the start of the conversation and the tail of every
 * generation prompt. Every turn that closes with an EOS token is parsed
 * (`parseChatResponse`, the reply of {@link ChatSessionStream.reply}); a turn that stopped at
 * `<|tool_response>` after a call ends `"tool-call"`, and {@link ChatSession.sendToolResults}
 * continues **the same model turn** with the results (the delta of `chatToolResults` — the
 * `<|tool_response>` that stopped generation is the pending token, as `<turn|>` is for a closed
 * turn). The history records what the parser read, in the template's input shape: the thought as
 * `reasoning`, the calls as `tool_calls` (with session-assigned ids), each result as a role `tool`
 * message; content the model wrote **before** its calls becomes its own assistant message in front
 * of the tool-calling one, because the template writes a message's content after its results.
 *
 * ## The identity between the KV and a whole re-render
 *
 * With `P₀ = chatPrompt(h₀, o)` the first prompt (`h₀` the history then, `o` the session's
 * template options), `Gⱼ` / `sⱼ` the ids and the stop token generated in step `j`, and `Dⱼ` the
 * delta of step `j` — `chatTurn(m, o)` after a turn closed by `<turn|>`, `chatToolResults(calls,
 * results, o)` after a tool call — the KV of step `k` holds
 *
 * ```text
 * P₀ ⧺ G₀ ⧺ s₀ ⧺ D₁ ⧺ G₁ ⧺ s₁ ⧺ … ⧺ Dₖ  =  chatPrompt(hₖ, o)
 * ```
 *
 * token for token (`hₖ` = {@link ChatSession.turns} once step `k`'s messages are pushed, before it
 * generates), whenever each `Gⱼ` is the encoding of the canonical rendering of what step `j`
 * recorded. It differs, by design, in
 * three places — all of them the template's own spellings; only the whole re-render matches
 * `apply_chat_template`:
 *
 * 1. `"gemma4-unified"` (gemma-4-12B-it) with thinking off: every generation prompt that opens a
 *    model turn ends with an empty thought channel, which the KV keeps at the start of each past
 *    model turn and a re-render drops (the NOTE of `renderGemma4ChatTurn` in `./text/chat.ts`).
 * 2. Thinking on: a re-render drops the thought of the model turns before the last user message
 *    (the template's gate); the KV keeps every thought.
 * 3. What the parser normalizes: content and thoughts are stripped (a thought is re-rendered as
 *    `<|channel>thought\n…\n<channel|>`), argument keys come back in the template's sorted order
 *    and values in its spelling, and content written after a call (before `<|tool_response>`) is
 *    re-rendered before it.
 *
 * The check is `gemma4_chat_session_test.ts` (a multi-step tool conversation for both dialects).
 * Images do not move the identity: a render places an image from its size alone, so its soft ids are
 * the same on every render.
 *
 * ## Overflow handling is **injectable** (the revision of ADR 0083 decision 10)
 *
 * The low-level surface still says "truncation is the host's job" and ends by throwing
 * `GenerationCapacityError`. This layer has a default policy ({@link dropOldestTurns} = drop the
 * oldest user / assistant pair, keep system), replaceable as a whole through
 * {@link ChatSessionOptions.onOverflow}. A host that passes a throwing function can handle it
 * on its side as before.
 *
 * A turn without `maxNewTokens` (neither per turn nor per session) generates until a stop token or
 * until the conversation is full. Its up-front check keeps {@link ChatSessionOptions.reserveNewTokens}
 * free, which is what makes a reply possible at all after a long history; the reserve does not limit
 * the reply, which may run past it up to the limit and then closes as `capacity`.
 *
 * ## Branching ({@link ChatSession.fork} — ADR 0083 addendum 2026-10-11, ruling 8)
 *
 * Returns an independent session with the same settings and the same history. A conversation that
 * carries its KV is copied KV and all with `GenerationSequence.fork`; a conversation with an empty KV
 * (nothing sent yet, or the previous turn did not close with EOS) copies only the history (no GPU
 * copy — the branch's first `send` re-renders the whole, like the original). Regeneration is written
 * as this use rather than a `regenerate` (the doc of {@link ChatSession.fork}).
 *
 * ## Images (ADR 0144)
 *
 * `send` takes content parts when the host has an embedder ({@link ChatSessionHost.vision}). A history
 * that can be re-rendered from the start decides three things:
 *
 * - **One copy of the pixels.** `send` copies each image once (the rule "copied at issue"). That frozen
 *   object is the image's identity, and the history copies — `turns`, an overflow policy's input,
 *   {@link ChatSession.fork} — hold references to it ({@link copyMessages}); `structuredClone` would
 *   copy the pixels on every read.
 * - **One embedding per image.** The rows are kept next to the history, keyed by that object, so a whole
 *   re-render or a branch runs no embedder. They are released with the last entry that holds the image.
 * - **Equality looks at the pixels** ({@link sameValue}), because a policy may return deep copies. Two
 *   different images never compare equal; such a copy embeds once more, as the new image it is.
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import { closeableGenerator } from "../concurrency/closeable-generator.ts";
import { ModelInputError } from "../errors.ts";
import type { GenerationProgram } from "../generation/program.ts";
import { type SamplerSpec, snapshotSpec } from "../generation/sampler.ts";
import {
  assertGenerationRequestValues,
  type GenerationCapacityDetail,
  GenerationCapacityError,
  type GenerationSequence,
  type GenerationSoftBlock,
  type GenerationStream,
} from "../generation/sequence.ts";
import { createStopStringFilter } from "../text/detokenizer.ts";
import type { Rgb8Image } from "../image/preprocess.ts";
import type { DefaultSampler, PipelineVisionConfig } from "./config.ts";
import {
  chatStreamOf,
  completeChatTurn,
  decodeChatChunks,
  type ChatStop,
  type ChatStream,
  type PrefillProgress,
} from "./chat-turn.ts";
import type { SequenceOptions } from "./pipeline.ts";
import {
  assertChatTemplateOptions,
  type ChatAssistantMessage,
  type ChatContentPart,
  chatImagePrompt,
  type ChatImagePrompt,
  chatImageTurn,
  type ChatMessage,
  type ChatPromptImage,
  chatPrompt,
  type ChatTemplateOptions,
  type ChatTool,
  type ChatToolCall,
  type ChatToolMessage,
  chatToolResults,
  chatTurn,
  renderGemma4ToolResults,
} from "./text/chat.ts";
import {
  type ChatResponse,
  openPrefillOf,
  type ParsedToolCall,
  parseChatResponse,
} from "./text/chat-response.ts";
import type { Tokenizer } from "./text/tokenizer.ts";
import type { ImageEmbedding } from "./vision.ts";

/**
 * The surface the session reads from the pipeline ({@link Pipeline} satisfies it as is).
 *
 * The surface is narrowed to show in the type that this layer **uses nothing inside the public
 * surface** (a consumer could write the same thing — its value as a copyable example lies here).
 * That tests can look at the conversation assembly alone without a real GPU follows from the same
 * property.
 */
export type ChatSessionHost = {
  readonly tokenizer: Tokenizer;
  readonly program: GenerationProgram;
  readonly defaultSampler: DefaultSampler | undefined;
  sequence(options?: SequenceOptions): Promise<GenerationSequence>;
  /**
   * The image input's declaration: present on a pipeline opened with `vision: true`. Without it the
   * session refuses content parts.
   */
  readonly vision?: PipelineVisionConfig | undefined;
  /** One image → its rows (needed with {@link ChatSessionHost.vision}). */
  embedImage?(image: Rgb8Image): Promise<ImageEmbedding>;
};

/**
 * The context passed to overflow handling ("the current conversation" and "how much is missing").
 *
 * `capacity` / `needed` use **the same measure** (the number of logical positions the conversation
 * occupies), and `needed > capacity` is why it was called. `capacity` is **whichever of two limits
 * bit first** (the smaller of the state capacity this session allocated and the position limit the
 * model declares); either way the remedy is the same (shorten the conversation).
 */
export type ChatOverflow = {
  /** The current history (if a system message is declared, it comes first). */
  readonly turns: readonly ChatMessage[];
  /** The system message the session declared (`undefined` = none declared). */
  readonly system: string | undefined;
  /** The upper limit of logical positions this conversation can use. */
  readonly capacity: number;
  /**
   * The number of logical positions needed to pass this turn (prompt + `maxNewTokens` − 1; a turn
   * without `maxNewTokens` counts {@link ChatSessionOptions.reserveNewTokens} in its place).
   */
  readonly needed: number;
};

/**
 * A function that returns **a new history** on overflow (default {@link dropOldestTurns}).
 *
 * MUST: the returned history is **shorter** than the `turns` passed in. Returning the same length
 * (= nothing can be done) makes the session fail with `GenerationCapacityError` — a retry that does
 * not shrink can only loop forever, and would also crush the fact that "the conversation does not
 * fit". A throw reaches the caller as is.
 *
 * MUST: **keep the last entries being answered now** (the message just sent, or every tool result
 * just sent). A history that dropped them passes into rendering without an exception (`chatPrompt`
 * does not constrain the last role), and an answer gets attached to "a conversation with no
 * question". An implementation that returns copies (`turns.map((turn) => ({ ...turn }))`) is
 * legitimate; the check compares values. A history that splits a tool-calling message from its
 * results fails loudly when it is rendered.
 */
export type ChatOverflowPolicy = (
  context: ChatOverflow,
) => readonly ChatMessage[] | Promise<readonly ChatMessage[]>;

/** The settings of one session (per-turn options are {@link ChatTurnOptions}). */
export type ChatSessionOptions = {
  /** The system message placed at the start of the conversation (none when omitted). */
  readonly system?: string;
  /**
   * The upper limit of tokens generated in one turn (can be overridden per turn).
   *
   * Omitted = a turn without its own value generates until a stop token or until the conversation
   * is full ({@link ChatSessionOptions.reserveNewTokens} is what its overflow check keeps free).
   */
  readonly maxNewTokens?: number;
  /**
   * The room the up-front overflow check keeps free for a turn without `maxNewTokens` (a positive
   * integer, default 1024): the history is dropped ({@link ChatSessionOptions.onOverflow}) until
   * prompt + reserve fits.
   *
   * It does **not** limit the generated length: such a turn may run past the reserve up to the limit,
   * and it is never passed to generation. A capacity smaller than prompt + reserve fails loudly
   * (`GenerationCapacityError`, naming this option) rather than clamping the reserve, so a small
   * `capacity` needs a matching value here.
   *
   * MUST: not together with a session {@link ChatSessionOptions.maxNewTokens} (fails loudly at
   * construction — with a session value every turn has a limit, so the reserve could never apply). A
   * per-turn `maxNewTokens` overrides the reserve for that turn.
   */
  readonly reserveNewTokens?: number;
  /**
   * The capacity this session allocates (default: the distribution's default =
   * `host.program.capacity`).
   *
   * One session = one conversation, so this is "how much KV to take for this conversation". The
   * overflow decision (the line at which {@link ChatSessionOptions.onOverflow} is called) uses
   * this value too — if the truncation criterion and the capacity actually allocated were different
   * numbers, a turn that should pass would be dropped, or the reverse.
   *
   * MUST: the same value is used when the sequence is rebuilt (if the capacity changed mid-
   * conversation, the truncation decision would use a different measure per turn).
   */
  readonly capacity?: number;
  /**
   * This session's sampling options. When omitted, the distribution's declaration
   * ({@link ChatSessionHost.defaultSampler}), and failing that temperature 0 (greedy).
   */
  readonly sampler?: SamplerSpec;
  /** Rebuilds the history when capacity runs out (default {@link dropOldestTurns}). */
  readonly onOverflow?: ChatOverflowPolicy;
  /**
   * Whether this conversation uses speculative decoding (default = `true` if the pipeline has a
   * drafter = with the self-tuning gate; the values mean the same as
   * {@link SequenceOptions.speculative}, and `"always"` speculates always, with no gate).
   *
   * This is a **per-session** option, not per-turn — this layer reuses one sequence across many
   * turns, and the speculation borrower context is bound to the sequence's lifetime (switching per
   * turn would mean rebuilding the sequence = discarding the KV). The gate's moving average has the
   * same lifetime, so it keeps measuring across turns.
   */
  readonly speculative?: boolean | "always";
  /**
   * The function declarations of this conversation (the OpenAI shape — `gemma.ChatTool`).
   *
   * Fixed for the session: the declarations live in the system block at the start of the
   * conversation, so changing them would mean re-rendering everything. Checked at construction.
   */
  readonly tools?: readonly ChatTool[];
  /**
   * Thinking mode (default off) — `<|think|>` in the system block, and the model writes a thought
   * channel before its answer. Fixed for the session (it shapes the system block and every
   * generation prompt).
   */
  readonly enableThinking?: boolean;
};

/**
 * How one turn of a session ended ({@link ChatSessionStream.reply}).
 *
 * - `"turn"` — the turn closed with an EOS token and holds no tool call; `message` is what the
 *   parser read (`content`, `thinking`).
 * - `"tool-call"` — the turn stopped at `<|tool_response>` after one or more calls; `message`
 *   carries them (each with a session-assigned `id`). The session waits for
 *   {@link ChatSession.sendToolResults}.
 * - `"cut-off"` — the turn did not close (max tokens, a full context, a stop string, a host stop
 *   token, `break`, an interruption): nothing is parsed, the streamed text is what the history keeps
 *   and the next turn re-renders it.
 */
export type ChatReply =
  | { readonly end: "turn"; readonly message: ChatResponse; readonly stop: ChatStop }
  | {
      readonly end: "tool-call";
      readonly message: ChatResponse & { readonly tool_calls: readonly ParsedToolCall[] };
      readonly stop: ChatStop;
    }
  | { readonly end: "cut-off"; readonly stop: ChatStop };

/**
 * The stream of one session turn: the {@link ChatStream} of `Pipeline.chat` plus the parsed reply.
 *
 * MUST: **drain the stream (or close it with `break`) before awaiting `reply`** — `reply` settles
 * at the end of the stream, so awaiting it first waits forever. It rejects with the turn's failure,
 * like `done`, and also — while `done` resolves — when the model's closed turn is malformed (a
 * `ModelInputError`: a broken call, tool calls closed by `<turn|>`, a call in a session without
 * tools, `<|tool_response>` without a call in a session with tools; the history then keeps the raw
 * text and the KV is rebuilt on the next turn) or when the session was disposed during the turn (a
 * plain `Error` — a call from a torn-down turn must not be run). The message is a deep-frozen copy.
 */
export type ChatSessionStream = ChatStream & { readonly reply: Promise<ChatReply> };

/**
 * Options that apply to one turn only (the same vocabulary and meaning as the request of
 * {@link Pipeline.chat}).
 *
 * MUST: {@link ChatSession.send} **copies** the contents **at issue** (ADR 0083 addendum
 * 2026-09-02) — rewriting them after the returned stream has started draining has no effect on the
 * running turn.
 */
export type ChatTurnOptions = {
  /**
   * Defaults to the session's {@link ChatSessionOptions.maxNewTokens}; with neither, the turn
   * generates until a stop token or until the conversation is full.
   */
  readonly maxNewTokens?: number;
  /** Additional stop tokens for this turn only (unioned with the distribution's EOS set). */
  readonly stopTokens?: readonly number[];
  /** Stop strings for this turn only (checked on the decoded text). */
  readonly stopStrings?: readonly string[];
  /** Defaults to the session's sampler. */
  readonly sampler?: SamplerSpec;
  /**
   * Prefill progress (each time one chunk is committed — the same meaning as
   * `ChatOptions.onPrefill`).
   *
   * The longer the conversation, the longer the prefill of "a turn that cannot carry the KV and
   * re-renders the whole", so an entry point that reports the silent time is needed.
   */
  readonly onPrefill?: (progress: PrefillProgress) => void;
  /**
   * Notification of generated tokens excluding stop tokens. Called synchronously, before decoding
   * and before stop-string holding. Includes special tokens that emit no text. Exceptions propagate
   * to the turn. The function is copied at issue.
   */
  readonly onToken?: (id: number) => void;
  /** Interruption (throws `signal.reason` as is — ADR 0083 decision 5). */
  readonly signal?: AbortSignal;
};

/** The default of {@link ChatSessionOptions.reserveNewTokens}. */
const DEFAULT_RESERVE_NEW_TOKENS = 1024;

/**
 * The default overflow handling — **drop the oldest exchange** (the system message is kept): the
 * entries from the first one after the system message up to the next user message — a user /
 * assistant pair in a plain conversation, the question with its tool calls, results and answer in
 * a tool conversation.
 *
 * A whole exchange is dropped at a time because dropping only part of it leaves "a conversation
 * with only the answer", "only the question", or a tool result without its call — a broken context
 * to show the model (the last one fails loudly when rendered). When nothing can be dropped (the
 * exchange being answered now is all there is), `turns` is returned as is (= the session fails with
 * `GenerationCapacityError` — this exchange alone does not fit).
 */
export const dropOldestTurns: ChatOverflowPolicy = ({ turns, system }): readonly ChatMessage[] => {
  const floor = system === undefined ? 0 : 1;
  // The exchange being answered now (from the last user message on) is never dropped.
  const next = turns.findIndex((turn, index) => index > floor && turn.role === "user");
  if (next < 0) return turns;
  return [...turns.slice(0, floor), ...turns.slice(next)];
};

/**
 * Whether this turn hits a limit (returns the actual values if it does).
 *
 * Applies **the same two formulas** as `GenerationSequence`'s budget check (`assertBudget` in
 * `sequence.ts`), not right before the run but **right after the prompt is built**. Truncation is
 * more naturally decided "before sending" than "after an exception", and `GenerationSequence.used`
 * is public for that (ADR 0083 addendum 2026-08-31).
 *
 * NOTE: the split of the fields is off by one from the sequence layer — this layer's `pastLength` is
 * `used` (including the uncommitted frontier), and `promptLength` is the length of the prompt this
 * layer passes. The total and `maxNewTokens` (= the limit that would pass now) both match the
 * sequence layer's fields. A turn without `maxNewTokens` is checked with the reserve in its place.
 *
 * MUST: `capacity` is **the capacity this session allocated** (not `program.capacity` = the
 * distribution's default). Deciding truncation on a number other than the capacity actually
 * allocated would drop a turn that passes, or send a turn that fails.
 */
const overflowOf = (
  program: GenerationProgram,
  capacity: number,
  used: number,
  promptLength: number,
  maxNewTokens: number,
): GenerationCapacityDetail | undefined => {
  const detail = (
    constraint: GenerationCapacityDetail["constraint"],
    limit: number,
  ): GenerationCapacityDetail => ({
    constraint,
    pastLength: used,
    promptLength,
    requestedNewTokens: maxNewTokens,
    limit,
    maxNewTokens: limit - used - promptLength + 1,
  });
  const peak = used + promptLength + maxNewTokens - 1;
  // Same order as the sequence layer (a turn that hits both names the same constraint).
  if (peak - 1 >= program.maxPosition) return detail("maxPosition", program.maxPosition);
  if (peak > capacity) return detail("capacity", capacity);
  return undefined;
};

/**
 * The number of logical positions the conversation occupies ({@link ChatOverflow.needed}).
 */
const neededOf = (detail: GenerationCapacityDetail): number =>
  detail.pastLength + detail.promptLength + detail.requestedNewTokens - 1;

/**
 * Message equality (the last-entry check of `#shrink` and the rewind of `#finish` use **the same
 * comparison**).
 *
 * MUST: do not compare by reference identity. An overflow policy may return copies such as
 * `turns.map((turn) => ({ ...turn }))` (what it receives is a frozen copy, so an implementation that
 * copies and returns is legitimate). Comparing by identity would make that legitimate policy either
 * "judged to have dropped the last entry while keeping it" (the check side) or "silently disable the
 * rewind of a turn that emitted no character" (the `#finish` side).
 */
const sameTurn = (left: ChatMessage, right: ChatMessage): boolean => sameValue(left, right);

/**
 * Structural equality of what messages hold: plain objects, arrays, scalars, and an image's pixels
 * (byte by byte, because an array of bytes is not a JSON-like value to walk key by key).
 */
const sameValue = (left: unknown, right: unknown): boolean => {
  if (left === right) return true;
  if (typeof left !== "object" || typeof right !== "object" || left === null || right === null) {
    return false;
  }
  if (left instanceof Uint8Array || right instanceof Uint8Array) {
    if (!(left instanceof Uint8Array && right instanceof Uint8Array)) return false;
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
      if (left[index] !== right[index]) return false;
    }
    return true;
  }
  if (Array.isArray(left) !== Array.isArray(right)) return false;
  const leftRecord = left as Readonly<Record<string, unknown>>;
  const rightRecord = right as Readonly<Record<string, unknown>>;
  const keys = Object.keys(leftRecord);
  if (keys.length !== Object.keys(rightRecord).length) return false;
  return keys.every(
    (key) => Object.hasOwn(rightRecord, key) && sameValue(leftRecord[key], rightRecord[key]),
  );
};

/** Whether `turns` ends with exactly `tail` (by value — {@link sameTurn}). */
const endsWith = (turns: readonly ChatMessage[], tail: readonly ChatMessage[]): boolean =>
  turns.length >= tail.length &&
  tail.every((message, index) => sameTurn(turns[turns.length - tail.length + index], message));

/**
 * Freezes every object under `value`. An image's pixels stay as they are: a typed array with elements
 * cannot be frozen.
 */
const deepFreeze = (value: unknown): void => {
  if (typeof value !== "object" || value === null || ArrayBuffer.isView(value)) return;
  Object.values(value).forEach(deepFreeze);
  Object.freeze(value);
};

/** A deep copy whose every object is frozen (the history handed out cannot reach the internal one). */
const frozenCopy = <T>(value: T): T => {
  const copy = structuredClone(value);
  deepFreeze(copy);
  return copy;
};

/**
 * Copies of history entries that share the images: a content array is copied part by part with the
 * image object in place, anything else goes through `structuredClone`.
 */
const copyMessages = (messages: readonly ChatMessage[]): ChatMessage[] =>
  messages.map((message) =>
    message.role === "user" && typeof message.content !== "string"
      ? { ...message, role: "user", content: message.content.map((part) => ({ ...part })) }
      : structuredClone(message),
  );

/** The session's own copy of one part (the MUST of {@link ChatSession.send}). */
const snapshotPart = (part: ChatContentPart): ChatContentPart => {
  // Anything else is not a part: the check that follows names it, and it needs no copy to be refused.
  if (typeof part !== "object" || part === null) return part;
  if (part.type !== "image" || !(part.image?.data instanceof Uint8Array)) return { ...part };
  const { data, width, height } = part.image;
  // MUST: `new Uint8Array(data)`, not `data.slice()`: a Node Buffer is a Uint8Array whose `slice` is
  // a view of the same memory.
  return { ...part, image: Object.freeze({ data: new Uint8Array(data), width, height }) };
};

/** The tool calls of a history entry (none for anything but a tool-calling assistant message). */
const callsOf = (message: ChatMessage | undefined): readonly ChatToolCall[] =>
  message?.role === "assistant" ? (message.tool_calls ?? []) : [];

/**
 * The history entries of a parsed closed turn (the template's input shape — the doc of
 * {@link ChatSession}, "Tools, tool calls and thinking").
 *
 * The calls get session-assigned ids (`call_1`, `call_2`, … from `firstId` — unique within the
 * session, so a result can never bind to an earlier call of the same id); the template does not
 * render ids.
 */
const entriesOf = (
  parsed: ChatResponse,
  firstId: number,
): { readonly entries: ChatAssistantMessage[]; readonly calls: ParsedToolCall[] } => {
  const reasoning = parsed.thinking === undefined ? {} : { reasoning: parsed.thinking };
  const calls = (parsed.tool_calls ?? []).map((call, index): ParsedToolCall => ({
    ...call,
    id: `call_${firstId + index}`,
  }));
  if (calls.length === 0) {
    return { entries: [{ role: "assistant", content: parsed.content ?? "", ...reasoning }], calls };
  }
  if (parsed.content === undefined) {
    return {
      entries: [{ role: "assistant", content: "", ...reasoning, tool_calls: calls }],
      calls,
    };
  }
  return {
    entries: [
      { role: "assistant", content: parsed.content, ...reasoning },
      { role: "assistant", content: "", tool_calls: calls },
    ],
    calls,
  };
};

/** One turn's request, read and copied at issue ({@link ChatSession} `#requestOf`). */
type TurnRequest = {
  /** `undefined` = until a stop token or the limit. */
  readonly maxNewTokens: number | undefined;
  readonly sampler: SamplerSpec | undefined;
  readonly stopTokens: number[] | undefined;
  readonly signal: AbortSignal | undefined;
  readonly onPrefill: ((progress: PrefillProgress) => void) | undefined;
  readonly onToken: ((id: number) => void) | undefined;
  readonly stopStrings: ReturnType<typeof createStopStringFilter>;
  readonly detokenizer: ReturnType<Tokenizer["createDetokenizer"]>;
};

/**
 * Checks that the results answer every call of the turn exactly once (OpenAI's contract — a call
 * left unanswered, or answered twice, would put a turn the model never saw into the history).
 */
const assertOneResultPerCall = (
  calls: readonly ChatToolCall[],
  results: readonly ChatToolMessage[],
): void => {
  const where = "gemma.ChatSession.sendToolResults";
  const answered = new Map<ChatToolCall, number>();
  results.forEach((result, index) => {
    const named =
      result.tool_call_id === undefined
        ? calls.filter((call) => call.function.name === result.name)
        : calls.filter((call) => call.id === result.tool_call_id);
    if (named.length !== 1) {
      throw new ModelInputError(
        `${where}: results[${index}] matches ${named.length} of the turn's calls (name the call by ` +
          `tool_call_id — one result per call)`,
      );
    }
    answered.set(named[0], (answered.get(named[0]) ?? 0) + 1);
  });
  const label = (call: ChatToolCall): string => `${call.id ?? "?"} (${call.function.name})`;
  const missing = calls.filter((call) => !answered.has(call));
  const twice = calls.filter((call) => (answered.get(call) ?? 0) > 1);
  if (missing.length > 0 || twice.length > 0) {
    throw new ModelInputError(
      `${where}: every call of the turn needs exactly one result` +
        (missing.length > 0 ? ` — unanswered: ${missing.map(label).join(", ")}` : "") +
        (twice.length > 0 ? ` — answered more than once: ${twice.map(label).join(", ")}` : ""),
    );
  }
};

/**
 * A session holding one conversation (history + KV).
 *
 * ```ts
 * await using pipeline = await Pipeline.fromPretrained(ref);
 * await using session = new ChatSession(pipeline, { maxNewTokens: 256 });
 * let reply = "";
 * for await (const chunk of session.send("Name a color.")) reply += chunk; // streamed text
 * console.log(await session.send("Another one.").text());
 * ```
 *
 * With tools, a turn can end on a tool call; the results continue the same model turn:
 *
 * ```ts
 * await using session = new ChatSession(pipeline, { maxNewTokens: 256, tools: [weather] });
 * let turn = session.send("What is the weather in Paris?");
 * await turn.text();
 * let reply = await turn.reply;
 * while (reply.end === "tool-call") {
 *   const results = reply.message.tool_calls.map((call) => ({
 *     role: "tool" as const,
 *     tool_call_id: call.id,
 *     content: run(call.function.name, call.function.arguments), // the host runs the tool
 *   }));
 *   turn = session.sendToolResults(results);
 *   await turn.text();
 *   reply = await turn.reply;
 * }
 * ```
 *
 * MUST: only one turn can run at a time (a second `send` throws **synchronously**). The history is
 * stacked in order as one conversation, so two generations pushing onto the same history would put a
 * conversation with "a question without an answer" in between into the KV — a mix-up that raises no
 * exception, so the entry point blocks it.
 *
 * MUST: an issued stream is **drained or closed with `break`**. The turn's wrap-up (appending to the
 * history, deciding whether to carry the KV, accepting the next `send`) runs at the end of the
 * stream, so a stream discarded without draining leaves the session stuck in that turn.
 */
export class ChatSession {
  readonly #host: ChatSessionHost;
  readonly #system: string | undefined;
  readonly #maxNewTokens: number | undefined;
  /** What the overflow check keeps free for a turn without `maxNewTokens` (never sent to generate). */
  readonly #reserveNewTokens: number;
  readonly #sampler: SamplerSpec | undefined;
  readonly #onOverflow: ChatOverflowPolicy;
  /**
   * The capacity this session allocates (unchanged when the sequence is rebuilt — also the measure
   * of the overflow decision).
   */
  readonly #capacity: number;
  /**
   * Whether to speculate (`undefined` = leave it to the pipeline's default — unchanged when the
   * sequence is rebuilt).
   */
  readonly #speculative: boolean | "always" | undefined;
  /** The template options of every render (a frozen copy — fixed for the session). */
  readonly #template: ChatTemplateOptions;
  /**
   * The conversation history (this layer's only mutable state — the sequence holds no transcript).
   */
  #turns: ChatMessage[];
  /**
   * The rows of the images the history holds, keyed by the image object ("Images" of the module doc).
   * Derived data the embedder is too slow to derive again: {@link ChatSession.#pruneEmbeddings} keeps
   * its keys inside the history.
   */
  #embeddings = new Map<Rgb8Image, ImageEmbedding>();
  /** The current sequence (`undefined` = rebuild on the next turn). */
  #sequence: GenerationSequence | undefined;
  /** The number of history entries in that sequence's KV (0 = empty = re-render the whole). */
  #committed = 0;
  /** A turn is running (issued, and its stream has not ended yet). */
  #busy = false;
  /**
   * The number of the next session-assigned call id (`call_<n>`). Not derivable from the history:
   * overflow handling drops calls, and their ids must not come back (a host may still hold them).
   */
  #nextCallId = 1;
  /** The one dispose. **Not undefined means "disposed"** (no separate derived state). */
  #disposal: Promise<void> | undefined;

  constructor(host: ChatSessionHost, options: ChatSessionOptions) {
    this.#host = host;
    this.#system = options.system;
    this.#maxNewTokens = options.maxNewTokens;
    if (options.reserveNewTokens !== undefined) {
      if (options.maxNewTokens !== undefined) {
        throw new ModelInputError(
          "gemma.ChatSession: reserveNewTokens cannot be combined with a session maxNewTokens " +
            "(every turn then has a limit, so the reserve could never apply)",
        );
      }
      if (!Number.isSafeInteger(options.reserveNewTokens) || options.reserveNewTokens < 1) {
        throw new ModelInputError(
          `gemma.ChatSession: reserveNewTokens ${options.reserveNewTokens} is not a positive integer`,
        );
      }
    }
    this.#reserveNewTokens = options.reserveNewTokens ?? DEFAULT_RESERVE_NEW_TOKENS;
    this.#sampler = options.sampler;
    this.#onOverflow = options.onOverflow ?? dropOldestTurns;
    this.#capacity = options.capacity ?? host.program.capacity;
    this.#speculative = options.speculative;
    // MUST: check the declarations **at construction** — they are rendered only into the first
    // prompt, so a bad one would otherwise surface on the first `send`, after the history moved.
    const template: ChatTemplateOptions = {
      ...(options.tools === undefined ? {} : { tools: options.tools }),
      ...(options.enableThinking === undefined ? {} : { enableThinking: options.enableThinking }),
    };
    assertChatTemplateOptions(template, "gemma.ChatSession");
    this.#template = frozenCopy(template);
    // MUST: check the capacity relations **at construction** (the same two formulas as
    // `createGenerationSequence` in `sequence.ts`). This layer uses `#capacity` as the measure of
    // the overflow decision **before** the sequence is allocated, so leaving the check to the
    // sequence side alone would, for a long turn, have `#shrink` actually truncate the history
    // before the capacity error, and the real cause (a declaration mistake) would appear in no
    // diagnostic.
    if (!Number.isSafeInteger(this.#capacity) || this.#capacity < 1) {
      throw new ModelInputError(
        `gemma.ChatSession: capacity ${this.#capacity} is not a positive integer`,
      );
    }
    if (this.#capacity < host.program.chunkLength) {
      throw new ModelInputError(
        `gemma.ChatSession: capacity ${this.#capacity} is below chunkLength ` +
          `${host.program.chunkLength} (not even one chunk fits)`,
      );
    }
    if (this.#capacity > host.program.maxPosition) {
      throw new ModelInputError(
        `gemma.ChatSession: capacity ${this.#capacity} exceeds maxPosition ` +
          `${host.program.maxPosition} (a conversation filling the capacity would look up positions beyond the model's position limit)`,
      );
    }
    this.#turns = options.system === undefined ? [] : [{ role: "system", content: options.system }];
  }

  /**
   * The conversation so far (if a system message is declared, it comes first) — in the template's
   * input shape, assistant tool calls and role `tool` results included.
   *
   * Returns a deep-frozen **copy** — exposing the internal array as is would let the consumer's
   * `push` / `splice` (or a write into a call's arguments) silently break the mapping to the KV
   * ("When the KV is carried over" of {@link ChatSession}). Images are the session's own objects, not
   * copies; their pixels are not read again once embedded.
   */
  get turns(): readonly ChatMessage[] {
    const copy = copyMessages(this.#turns);
    deepFreeze(copy);
    return copy;
  }

  /**
   * Sends one user message and streams **committed text pieces** (the same stream, the same stop
   * reason and the same `text()` as {@link Pipeline.chat}, plus the parsed
   * {@link ChatSessionStream.reply}).
   *
   * The message is a string, or content parts on a host with an embedder
   * ({@link ChatSessionHost.vision}). MUST: each image is **copied at issue** (one copy, for the life
   * of its history entry — "Images" of the module doc), and refused parts fail synchronously with
   * `ModelInputError`: an empty array, audio or video, a text that encodes to an image token. Embedding
   * happens when the stream starts draining, once per image.
   *
   * If the previous turn closed with EOS, the KV is carried over and **only the delta** is streamed;
   * otherwise the sequence is rebuilt and the history re-rendered. A turn that does not fit goes to
   * overflow handling **before sending** ({@link ChatSessionOptions.onOverflow}).
   *
   * NOTE: in a session with tools or thinking the pieces keep the special spellings
   * (`<|channel>thought\n…<channel|>`, `<|tool_call>call:…<tool_call|>`, `<|"|>`) — without them a
   * thought would run into the answer and a call would lose its string delimiters. A plain session
   * streams the text without them, as `Pipeline.chat` does.
   *
   * MUST: after a turn that ended on a tool call, the results come first
   * ({@link ChatSession.sendToolResults}) — the model turn is still open, and a user message there
   * cannot be rendered.
   */
  send(
    message: string | readonly ChatContentPart[],
    options: ChatTurnOptions = {},
  ): ChatSessionStream {
    const pending = this.#userMessageOf(message);
    const request = this.#requestOf(options);
    // MUST: from here to `#busy = true` nothing the caller controls runs (every caller input was
    // read and copied above) — a getter that called `send` in between would issue two turns.
    this.#assertIdle("send");
    if (callsOf(this.#turns.at(-1)).length > 0) {
      throw new Error(
        "gemma.ChatSession: the previous turn ended on a tool call — send its results with " +
          "sendToolResults first (the model turn is still open)",
      );
    }
    return this.#issue([pending], request);
  }

  /**
   * Sends the results of the tool calls the previous turn ended on, and streams the model's
   * continuation of **the same model turn** (another call, or the answer).
   *
   * MUST: **exactly one result per call** of that turn (OpenAI's contract), each naming its call by
   * `tool_call_id` (the `id` of `reply.message.tool_calls` — unique within the session) or by
   * `name` when no other call of the turn has it. A missing, extra or ambiguous result throws
   * `ModelInputError` synchronously, before the history moves. The results are **copied** first and
   * the copy is what is checked and recorded.
   *
   * MUST: `content` is a string here (serialize a structured result) — the history keeps the
   * typed shape of {@link ChatToolMessage}, which `turns` hands out.
   */
  sendToolResults(
    results: readonly ChatToolMessage[],
    options: ChatTurnOptions = {},
  ): ChatSessionStream {
    const request = this.#requestOf(options);
    const copied: ChatToolMessage[] = structuredClone([...results]);
    // MUST: from here to `#busy = true` nothing the caller controls runs (the checks read the copy).
    this.#assertIdle("sendToolResults");
    const calls = callsOf(this.#turns.at(-1));
    if (calls.length === 0) {
      throw new Error(
        "gemma.ChatSession: no tool call is waiting for results (sendToolResults continues a turn " +
          "that ended on a tool call)",
      );
    }
    // The same checks as the render (shape and name resolution), so nothing invalid enters the
    // history.
    renderGemma4ToolResults(calls, copied, this.#template);
    copied.forEach((result, index) => {
      const content: unknown = result.content;
      if (typeof content !== "string") {
        throw new ModelInputError(
          `gemma.ChatSession.sendToolResults: results[${index}].content is not a string ` +
            `(${JSON.stringify(content)} — serialize the result)`,
        );
      }
    });
    assertOneResultPerCall(calls, copied);
    return this.#issue(copied, request);
  }

  /** Reads, copies and checks the message of {@link ChatSession.send}. */
  #userMessageOf(message: string | readonly ChatContentPart[]): ChatMessage {
    const { tokenizer, vision } = this.#host;
    if (typeof message === "string") {
      const text: ChatMessage = { role: "user", content: message };
      // On an image-capable host a spelled image token is refused here too, before the history moves.
      if (vision !== undefined) chatImageTurn(tokenizer, text, vision, this.#template);
      return text;
    }
    if (!Array.isArray(message)) {
      throw new ModelInputError(
        `gemma.ChatSession.send: the message is not a string or a content-part array (${typeof message})`,
      );
    }
    if (vision === undefined || this.#host.embedImage === undefined) {
      throw new ModelInputError(
        "gemma.ChatSession.send: content parts need a pipeline opened with vision: true, and this " +
          "one has no image embedder",
      );
    }
    // `[...]` turns a hole of a sparse array into `undefined`, which the check below refuses.
    const parts: ChatMessage = { role: "user", content: [...message].map(snapshotPart) };
    chatImageTurn(tokenizer, parts, vision, this.#template);
    return parts;
  }

  /** Throws synchronously when no turn can be issued (disposed, or a turn is running). */
  #assertIdle(entry: string): void {
    if (this.#disposal !== undefined) {
      throw new Error(`gemma.ChatSession: cannot ${entry} after dispose`);
    }
    if (this.#busy) {
      throw new Error(
        "gemma.ChatSession: the previous turn has not finished yet" +
          " (1 session = 1 generation — drain the issued stream or close it with break)",
      );
    }
  }

  /**
   * Reads, copies and checks one turn's options — **before** the session's state is looked at, so
   * no caller code (a getter, a sampler object) runs between the idle check and `#busy = true`.
   */
  #requestOf(options: ChatTurnOptions): TurnRequest {
    // MUST: the request is **copied at issue** (ADR 0083 addendum 2026-09-02). The body (an async
    // generator) does not run until the first `next()`, so without reading everything here it would
    // run with "the values at the time draining started".
    // Only `undefined` falls through: a `null` must reach the check below, not read as "omitted".
    const turnMaxNewTokens = options.maxNewTokens;
    const maxNewTokens = turnMaxNewTokens === undefined ? this.#maxNewTokens : turnMaxNewTokens;
    // The sampling options are copied here as well, like the arrays (the one copy point is
    // {@link snapshotSpec} in `sampler.ts` — `createSampler` copies at the first `next()`, so that
    // alone leaves the window between issue and consumption).
    const chosen = options.sampler ?? this.#sampler ?? this.#host.defaultSampler;
    const sampler = chosen === undefined ? undefined : snapshotSpec(chosen);
    const stopTokens = options.stopTokens === undefined ? undefined : [...options.stopTokens];
    const signal = options.signal;
    const onPrefill = options.onPrefill;
    const onToken = options.onToken;
    // The stop-string state machine is also built here (the option check and copy happen inside it =
    // the accepted set fails synchronously).
    const stopStrings = createStopStringFilter(options.stopStrings ?? []);
    const structured =
      (this.#template.tools?.length ?? 0) > 0 || this.#template.enableThinking === true;
    const detokenizer = this.#host.tokenizer.createDetokenizer({
      skipSpecialTokens: !structured,
    });
    // MUST: check the value ranges of the accepted set **before moving the session's state** (the
    // source of truth of the check is the one in `sequence.ts`). Placed later, an invalid request
    // would push a message onto the history and set `#busy`, "not failing until draining starts" —
    // from the caller's view, one mistake would make the session unusable.
    assertGenerationRequestValues(this.#host.program.vocabSize, {
      maxNewTokens,
      ...(stopTokens === undefined ? {} : { stopTokens }),
      ...(sampler === undefined ? {} : { sampler }),
    });
    return {
      maxNewTokens,
      sampler,
      stopTokens,
      signal,
      onPrefill,
      onToken,
      stopStrings,
      detokenizer,
    };
  }

  /** Issues one turn for the messages just sent (`pending` — one user message, or the results). */
  #issue(pending: readonly ChatMessage[], request: TurnRequest): ChatSessionStream {
    const {
      maxNewTokens,
      sampler,
      stopTokens,
      signal,
      onPrefill,
      onToken,
      stopStrings,
      detokenizer,
    } = request;
    this.#turns.push(...pending);
    this.#busy = true;

    let settle!: (stop: ChatStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<ChatStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // A secondary notification path, so not an unhandled rejection even when nobody reads it.
    done.catch(() => {});
    let settleReply!: (reply: ChatReply) => void;
    let failReply!: (error: unknown) => void;
    const reply = new Promise<ChatReply>((resolve, reject) => {
      settleReply = resolve;
      failReply = reject;
    });
    reply.catch(() => {});

    const prepare = this.#prepare.bind(this);
    const finish = this.#finish.bind(this);
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      /** The matched stop string (this layer's stop reason). */
      let matched: string | undefined;
      /** The text streamed in this turn (what a cut-off turn leaves in the history). */
      let text = "";
      /** The generated ids (stop tokens excluded — what the parser reads, with `done.token`). */
      const generated: number[] = [];
      let prompt: readonly number[] = [];
      try {
        const prepared = await prepare(maxNewTokens, signal);
        prompt = prepared.prompt;
        stream = prepared.sequence.generate({
          prompt: prepared.prompt,
          ...(prepared.softBlocks.length === 0 ? {} : { softBlocks: prepared.softBlocks }),
          ...(maxNewTokens === undefined ? {} : { maxNewTokens }),
          ...(stopTokens === undefined ? {} : { stopTokens }),
          ...(sampler === undefined ? {} : { sampler }),
          ...(signal === undefined ? {} : { signal }),
        });
        const parts = decodeChatChunks(stream, detokenizer, stopStrings, onPrefill, (id) => {
          generated.push(id);
          onToken?.(id);
        });
        try {
          for (;;) {
            const step = await parts.next();
            if (step.done === true) {
              matched = step.value;
              break;
            }
            text += step.value;
            yield step.value;
          }
        } finally {
          // MUST: pass the consumer's `break` inward (the same shape as `yield*` delegation).
          // Otherwise the event stream's `finally` does not run, and the sequence stays running.
          await parts.return(undefined);
        }
      } catch (error) {
        failure = { error };
        // MUST: throw as is, unwrapped (ADR 0083 decision 5 — the consumer can identify its own
        // interruption with `error === signal.reason`).
        throw error;
      } finally {
        await completeSessionTurn({
          stream,
          matched,
          failure,
          settle,
          fail,
          settleReply,
          failReply,
          finish: (stop) => finish(pending, { text, generated, prompt }, stop),
        });
      }
    };

    const stream = chatStreamOf(
      closeableGenerator(chunks(), (failure) => {
        // prepare has not run, so only the provisionally added messages are removed. The previous
        // turn's KV can still be carried over.
        this.#turns.splice(this.#turns.length - pending.length, pending.length);
        this.#busy = false;
        if (failure !== undefined) {
          fail(failure.error);
          failReply(failure.error);
        } else {
          const stop: ChatStop = { reason: "closed", tokens: 0 };
          settle(stop);
          settleReply({ end: "cut-off", stop });
        }
      }),
      done,
    );
    return Object.assign(stream, { reply });
  }

  /**
   * Releases. The sequence is closed after the running turn (`dispose` goes on the sequence's own
   * serialization chain, so flush-before-destroy is held there). Later calls return the same
   * completion.
   *
   * MUST: the pipeline is not closed (its owner is whoever passed it).
   */
  dispose(): Promise<void> {
    this.#disposal ??= this.#releaseSequence();
    return this.#disposal;
  }

  /** `await using` support (Explicit Resource Management) — an alias of {@link dispose}. */
  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  /**
   * Returns an **independent** session with the same settings and the same history (branching the
   * conversation). The returned session is owned by the caller, which also calls its `dispose()`
   * (there is no parent-child dependency on the original; the original may be disposed first).
   *
   * Regeneration is written as this use — keep the base session without advancing it, and on each
   * press `fork()` and `send` on the branch (the base stays, so it can be retried any number of
   * times):
   *
   * ```ts
   * const sampler = { temperature: 1, topP: 0.95 } satisfies SamplerSpec;
   * await session.send("Name a color.", { sampler }).text();
   * for (let attempt = 1; attempt <= 3; attempt += 1) {
   *   const retry = await session.fork(); // keep the base (session) as is
   *   // Change the seed per attempt (the same seed returns the same answer — see the note below).
   *   console.log(await retry.send("Another one.", { sampler: { ...sampler, seed: attempt } }).text());
   *   await retry.dispose(); // if it is not good, discard it; the next attempt forks the base again
   * }
   * ```
   *
   * Note: **the same seed gives the same answer**. The sampling random sequence is rebuilt on each
   * `send` from the sampler's `seed` (default 0), and the copied KV is bit-identical to the original,
   * so a regeneration that does not change the seed returns the same token sequence every time
   * (temperature 0 is the same regardless of seed).
   *
   * A conversation that carries its KV is copied whole at the same capacity, so the base + one
   * branch roughly doubles the KV VRAM. A conversation with an empty KV (nothing sent yet, or the
   * previous turn did not close with the distribution's EOS) makes no GPU copy and copies only the
   * history.
   *
   * When disposed, or when a turn is running (the stream `send` returned has not ended yet), it
   * throws **synchronously** (the same discipline as `send`).
   *
   * MUST: copying the history and "the number of entries in the KV" and calling `sequence.fork()`
   * happen **in the calling synchronous section**. Deferred, a `send` issued right after would push a
   * message onto the original history first, and "a question without an answer" would mix into the
   * branch (a mix-up that raises no exception).
   */
  fork(): Promise<ChatSession> {
    if (this.#disposal !== undefined) {
      throw new Error("gemma.ChatSession: cannot fork after dispose");
    }
    if (this.#busy) {
      throw new Error(
        "gemma.ChatSession: the previous turn has not finished yet" +
          " (fork only between turns — drain the issued stream or close it with break)",
      );
    }
    // Copy along with the KV-to-history mapping (`#committed`). Between turns, "no sequence ⇔ the KV
    // is empty (`#committed === 0`)", so the KV copy is made only when there is a sequence.
    const turns = copyMessages(this.#turns);
    const committed = this.#committed;
    const copying = this.#sequence?.fork();
    const branch = new ChatSession(this.#host, {
      ...(this.#maxNewTokens === undefined
        ? { reserveNewTokens: this.#reserveNewTokens }
        : { maxNewTokens: this.#maxNewTokens }),
      capacity: this.#capacity,
      onOverflow: this.#onOverflow,
      ...(this.#system === undefined ? {} : { system: this.#system }),
      ...(this.#sampler === undefined ? {} : { sampler: this.#sampler }),
      ...(this.#speculative === undefined ? {} : { speculative: this.#speculative }),
      ...this.#template,
    });
    branch.#turns = turns;
    // The rows are shared with the images: neither session writes them.
    branch.#embeddings = new Map(this.#embeddings);
    branch.#nextCallId = this.#nextCallId;
    if (copying === undefined) return Promise.resolve(branch);
    return copying.then((sequence) => {
      branch.#sequence = sequence;
      branch.#committed = committed;
      return branch;
    });
  }

  /**
   * This turn's prompt when the KV is carried over: the delta of the messages after the committed
   * ones — one message after a closed turn (`chatTurn`), or the results after a tool call
   * (`chatToolResults`).
   *
   * MUST: anything else is a broken mapping between the KV and the history — streaming a delta
   * while out of step would silently lose part of the conversation, so the premise of carrying over
   * is itself a check.
   */
  #delta(): ChatImagePrompt {
    const pending = this.#turns.slice(this.#committed);
    const owner = this.#turns[this.#committed - 1];
    const { tokenizer, vision } = this.#host;
    if (pending.length > 0 && pending.every((message) => message.role === "tool")) {
      return {
        prompt: chatToolResults(
          tokenizer,
          callsOf(owner),
          pending as ChatToolMessage[],
          this.#template,
        ),
        images: [],
      };
    }
    if (pending.length === 1 && callsOf(owner).length === 0) {
      return vision === undefined
        ? { prompt: chatTurn(tokenizer, pending[0], this.#template), images: [] }
        : chatImageTurn(tokenizer, pending[0], vision, this.#template);
    }
    throw new Error(
      `gemma.ChatSession internal inconsistency: the KV holds ${this.#committed} turns and the ` +
        `history ${this.#turns.length}, so the delta would be ` +
        (pending.length === 0
          ? "empty (nothing was sent)"
          : `${pending.map((message) => message.role).join(", ")} after ${owner?.role ?? "nothing"}`) +
        ` — neither one message after a closed turn nor the results of a tool call`,
    );
  }

  /** The whole history as a prompt (the turn starts from an empty KV). */
  #renderWhole(): ChatImagePrompt {
    const { tokenizer, vision } = this.#host;
    return vision === undefined
      ? { prompt: chatPrompt(tokenizer, this.#turns, this.#template), images: [] }
      : chatImagePrompt(tokenizer, this.#turns, vision, this.#template);
  }

  /**
   * The soft blocks of a rendered prompt: each image's rows from {@link ChatSession.#embeddings}, or
   * from the embedder the first time the image is rendered.
   *
   * MUST: the render's count equals the embedding's. The prompt's rows were reserved from the image's
   * size and the rows came from its pixels, and a difference would fill another row count than the
   * block holds.
   */
  async #softBlocksOf(
    images: readonly ChatPromptImage[],
    signal: AbortSignal | undefined,
  ): Promise<GenerationSoftBlock[]> {
    const blocks: GenerationSoftBlock[] = [];
    for (const { image, start, softTokens } of images) {
      let embedding = this.#embeddings.get(image);
      if (embedding === undefined) {
        // An abort is left to the sequence, which settles the turn as aborted before its first run,
        // as it does for a turn without images; the rows it needs are then none.
        if (signal?.aborted === true) return [];
        if (this.#host.embedImage === undefined) {
          throw new Error("gemma.ChatSession internal inconsistency: an image without an embedder");
        }
        embedding = await this.#host.embedImage(image);
        this.#embeddings.set(image, embedding);
      }
      if (embedding.softTokens !== softTokens) {
        throw new Error(
          `gemma.ChatSession internal inconsistency: the render reserved ${softTokens} soft tokens ` +
            `for an image the embedder returned ${embedding.softTokens} rows for`,
        );
      }
      blocks.push({ start, rows: embedding.rows });
    }
    return blocks;
  }

  /** Releases the rows of the images no history entry holds any more. */
  #pruneEmbeddings(): void {
    if (this.#embeddings.size === 0) return;
    const held = new Set<Rgb8Image>();
    for (const message of this.#turns) {
      if (message.role !== "user" || typeof message.content === "string") continue;
      for (const part of message.content) if (part.type === "image") held.add(part.image);
    }
    for (const image of this.#embeddings.keys()) {
      if (!held.has(image)) this.#embeddings.delete(image);
    }
  }

  /**
   * Prepares this turn's sequence and prompt (running overflow handling first if it does not fit).
   *
   * MUST: the overflow decision happens **before allocating** the sequence. All it needs is `used`,
   * and when `#committed` is 0 the KV is empty = `used` is 0, so it can be decided without
   * allocating. In the reverse order, "a turn that truncates allocates a capacity's worth of KV once
   * and immediately discards it" (hit on the first turn of a conversation, and when the previous turn
   * closed by max-tokens / a full context / a stop string / an interruption / a failure = not rare).
   *
   * MUST: retries continue only when the history **shrank** (the check in `#shrink`), so the number
   * of attempts does not exceed the number of messages at entry — the guard below exists to fail by
   * name when that invariant breaks, and is never hit while it holds.
   */
  async #prepare(
    maxNewTokens: number | undefined,
    signal: AbortSignal | undefined,
  ): Promise<{
    readonly sequence: GenerationSequence;
    readonly prompt: number[];
    readonly softBlocks: readonly GenerationSoftBlock[];
  }> {
    const attempts = this.#turns.length;
    for (let attempt = 0; ; attempt += 1) {
      // MUST: re-check after `await` (`send` checks at issue, but the body starts running later).
      // Draining an issued stream after disposing only the session would otherwise take a sequence
      // with fresh capacity from a live pipeline.
      if (this.#disposal !== undefined) {
        throw new Error("gemma.ChatSession: cannot generate after dispose");
      }
      const held = this.#sequence;
      // `used` is a derived value (`#committed === 0` ⇔ the KV is empty ⇔ `used === 0`). Assuming 0
      // while the equivalence is broken would make the overflow decision looser than reality, so a
      // guard that fails the break by name sits next to it.
      let used = 0;
      if (this.#committed !== 0) {
        if (held === undefined) {
          throw new Error(
            `gemma.ChatSession internal inconsistency: the KV holds ${this.#committed} turns but there is no sequence`,
          );
        }
        used = held.used;
      } else if (held !== undefined && held.used !== 0) {
        throw new Error(
          `gemma.ChatSession internal inconsistency: the KV is empty but the sequence's used is ${held.used}`,
        );
      }
      const { prompt, images } = this.#committed === 0 ? this.#renderWhole() : this.#delta();
      const detail = overflowOf(
        this.#host.program,
        this.#capacity,
        used,
        prompt.length,
        maxNewTokens ?? this.#reserveNewTokens,
      );
      if (detail === undefined) {
        // After the overflow decision (an image of a dropped turn is not embedded for nothing) and
        // before the allocation (a failing embedder takes no KV).
        const softBlocks = await this.#softBlocksOf(images, signal);
        if (this.#disposal !== undefined) {
          throw new Error("gemma.ChatSession: cannot generate after dispose");
        }
        if (held !== undefined) return { sequence: held, prompt, softBlocks };
        const created = await this.#host.sequence({
          capacity: this.#capacity,
          // When unspecified, the field is not passed at all (fall back to the pipeline's default =
          // speculate if there is a drafter).
          ...(this.#speculative === undefined ? {} : { speculative: this.#speculative }),
        });
        // MUST: check once more after `await` (the same shape as `Pipeline.sequence`). An
        // instance disposed mid-allocation is closed neither by `dispose()` (`#sequence` is still
        // undefined) nor by the eos branch of `#finish` (which does not go through
        // `#releaseSequence`) — this re-check is the only thing that closes the window.
        if (this.#disposal !== undefined) {
          await created.dispose();
          throw new Error("gemma.ChatSession: cannot generate after dispose");
        }
        this.#sequence = created;
        this.#committed = 0;
        return { sequence: created, prompt, softBlocks };
      }
      if (attempt >= attempts) {
        throw new Error(
          `gemma.ChatSession internal inconsistency: overflow handling did not converge in ${attempts} attempts`,
        );
      }
      await this.#shrink(detail, maxNewTokens === undefined ? "reserveNewTokens" : "maxNewTokens");
    }
  }

  /**
   * Runs overflow handling once (replaces the history and discards the sequence whose mapping to the
   * KV broke). `knob` is the option that sets the room this turn could not keep.
   */
  async #shrink(
    detail: GenerationCapacityDetail,
    knob: "maxNewTokens" | "reserveNewTokens",
  ): Promise<void> {
    const before = this.#turns.length;
    // The messages being answered now (one user message, or the results just sent).
    const asked = this.#turns.slice(
      this.#committed === 0 ? pendingStart(this.#turns) : this.#committed,
    );
    const needed = neededOf(detail);
    const next = await this.#onOverflow({
      turns: this.turns,
      system: this.#system,
      capacity: detail.limit,
      needed,
    });
    // MUST: the messages being answered now (the last entries) remain. Letting them through
    // dropped would make `#prepare` re-render "a conversation with no question", and since
    // `chatPrompt` does not constrain the last role either, only an answer would be committed
    // with **zero exceptions**. The comparison is by value ({@link sameTurn}) — a legitimate policy
    // that returns copies is not rejected.
    if (!endsWith(next, asked)) {
      throw new Error(
        `gemma.ChatSession: overflow handling dropped the current utterance (the last one)` +
          ` (history ${before} turns → ${next.length} — keep the utterance being answered now)`,
      );
    }
    if (next.length >= before) {
      throw new GenerationCapacityError(
        `The conversation does not fit: the ${detail.constraint} limit is ${detail.limit}, but ${needed} are needed` +
          ` (overflow handling did not shrink the history of ${before} turns — shorten the utterance or lower` +
          ` ${knob}. At this length ${knob} ≤ ${detail.maxNewTokens})`,
        detail,
      );
    }
    this.#turns = copyMessages(next);
    this.#pruneEmbeddings();
    // A truncated history is re-rendered from the start, so the context is returned and rebuilt.
    await this.#releaseSequence();
  }

  /**
   * The turn's wrap-up (appending to the history, deciding whether to carry the KV, accepting the
   * next `send`) → the parsed reply, or the reason the closed turn could not be parsed.
   */
  async #finish(
    pending: readonly ChatMessage[],
    output: {
      readonly text: string;
      readonly generated: readonly number[];
      readonly prompt: readonly number[];
    },
    stop: ChatStop | undefined,
  ): Promise<{ readonly reply: ChatReply } | { readonly malformed: unknown } | undefined> {
    try {
      const tokenizer = this.#host.tokenizer;
      // A thought channel the prompt opened (after a result with thinking on) belongs to this
      // turn's text: a record without it would let the thought through strip_thinking as content.
      const opener = openPrefillOf(tokenizer.decode(output.prompt, { skipSpecialTokens: false }));
      if (stop?.reason === "eos" && this.#disposal === undefined) {
        // Closed with the distribution's EOS = the model turn is closed (or stopped at a tool call).
        let parsed: ChatResponse | undefined;
        let malformed: unknown;
        try {
          parsed = parseChatResponse(tokenizer, [...output.generated, stop.token], output.prompt);
        } catch (error) {
          malformed = error;
        }
        const calledTool = (parsed?.tool_calls?.length ?? 0) > 0;
        const declared = (this.#template.tools?.length ?? 0) > 0;
        const toolResponse = tokenizer.addedTokenId("<|tool_response>");
        if (calledTool && !declared) {
          // Waiting for results nobody can produce would block the conversation.
          malformed = new ModelInputError(
            "gemma.ChatSession: the model wrote a tool call, but the session declares no tools",
          );
        } else if (calledTool && stop.token !== toolResponse) {
          malformed = new ModelInputError(
            `gemma.ChatSession: the model wrote tool calls but closed its turn with ` +
              `${tokenizer.decode([stop.token], { skipSpecialTokens: false })} instead of ` +
              `<|tool_response> (the results could not continue that turn)`,
          );
        } else if (parsed !== undefined && !calledTool && declared && stop.token === toolResponse) {
          // With tools, <|tool_response> opens a result; without a call none follows, and carrying
          // the KV would leave that spelling open in it.
          malformed = new ModelInputError(
            "gemma.ChatSession: the model closed its turn with <|tool_response> but wrote no call",
          );
        }
        if (parsed !== undefined && malformed === undefined) {
          if (this.#sequence === undefined) {
            throw new Error(
              "gemma.ChatSession internal inconsistency: a closed turn without a sequence to carry",
            );
          }
          const { entries, calls } = entriesOf(parsed, this.#nextCallId);
          this.#nextCallId += calls.length;
          // The history and the reply get copies of their own: a host writing into the reply's
          // arguments would otherwise rewrite the history the next re-render sends (the KV holds
          // what the model wrote). The entries are pushed even when the body is empty (the KV holds
          // an empty model turn, so without them the re-render would be a different conversation).
          this.#turns.push(...structuredClone(entries));
          this.#committed = this.#turns.length;
          return {
            reply: calledTool
              ? { end: "tool-call", message: frozenCopy({ ...parsed, tool_calls: calls }), stop }
              : { end: "turn", message: frozenCopy(parsed), stop },
          };
        }
        // A closed turn the parser rejects: the history keeps what the model wrote, with its
        // spellings (a re-render shows the model its own output), and the KV is rebuilt from it.
        this.#turns.push({
          role: "assistant",
          content: opener + tokenizer.decode(output.generated, { skipSpecialTokens: false }),
        });
        await this.#releaseSequence();
        return { malformed };
      }
      // max-tokens / capacity / a stop string / an interruption / a failure leave the model turn
      // unclosed = the premise of a delta is missing. So does a dispose during the turn: its result is not
      // delivered (a host must not run a call from a turn that was torn down).
      if (output.text !== "") {
        this.#turns.push({ role: "assistant", content: opener + output.text });
      }
      // A turn that emitted no character is **treated as never having happened** (no question
      // without an answer is left in the history). The comparison is by value ({@link sameTurn}) —
      // when overflow handling copies the history, references break.
      else if (endsWith(this.#turns, pending)) {
        this.#turns.splice(this.#turns.length - pending.length, pending.length);
      }
      await this.#releaseSequence();
      if (stop?.reason === "eos") {
        return {
          malformed: new Error(
            "gemma.ChatSession: the session was disposed during the turn — its result is not " +
              "delivered (a call from it must not be run)",
          ),
        };
      }
      return stop === undefined ? undefined : { reply: { end: "cut-off", stop } };
    } finally {
      this.#pruneEmbeddings();
      this.#busy = false;
    }
  }

  /**
   * Returns the context to prepare for a rebuild (always passed through when the KV-to-history
   * mapping breaks).
   */
  async #releaseSequence(): Promise<void> {
    const held = this.#sequence;
    this.#sequence = undefined;
    this.#committed = 0;
    await held?.dispose();
  }
}

/**
 * Where the messages being answered now start in a history whose KV is empty: the trailing tool
 * results, or the last message.
 */
const pendingStart = (turns: readonly ChatMessage[]): number => {
  let start = turns.length - 1;
  while (start > 0 && turns[start].role === "tool" && turns[start - 1].role === "tool") start -= 1;
  return start;
};

/**
 * The end of one session turn: {@link completeChatTurn} (the stop reason, the clean-up through
 * `finish`, the lease) and then the reply.
 *
 * NOTE: a separate function so that no control-flow statement sits in the caller's `finally`
 * (`no-unsafe-finally` — the same reason as `closeChatTurn`).
 */
const completeSessionTurn = async (options: {
  readonly stream: GenerationStream | undefined;
  readonly matched: string | undefined;
  readonly failure: { readonly error: unknown } | undefined;
  readonly settle: (stop: ChatStop) => void;
  readonly fail: (error: unknown) => void;
  readonly settleReply: (reply: ChatReply) => void;
  readonly failReply: (error: unknown) => void;
  readonly finish: (
    stop: ChatStop | undefined,
  ) => Promise<{ readonly reply: ChatReply } | { readonly malformed: unknown } | undefined>;
}): Promise<void> => {
  let outcome: { readonly reply: ChatReply } | { readonly malformed: unknown } | undefined;
  try {
    await completeChatTurn({
      where: "gemma.ChatSession.send",
      stream: options.stream,
      matched: options.matched,
      failure: options.failure,
      settle: options.settle,
      fail: options.fail,
      cleanup: async (stop) => {
        outcome = await options.finish(stop);
      },
    });
  } catch (error) {
    options.failReply(error);
    throw error;
  }
  if (outcome === undefined) {
    // `finish` returns nothing only without a stop reason, and then `completeChatTurn` throws.
    const error = new Error(
      "gemma.ChatSession internal inconsistency: the turn ended without an outcome",
    );
    options.failReply(error);
    throw error;
  } else if ("reply" in outcome) {
    options.settleReply(outcome.reply);
  } else {
    options.failReply(outcome.malformed);
  }
};
