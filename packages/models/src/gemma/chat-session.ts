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
 * max-tokens / a stop string / an interruption the model turn is not closed, so the sequence is
 * discarded and re-rendered from the history.
 *
 * NOTE: the distribution's EOS set contains `<eos>` and `<|tool_response>` as well
 * (`gemma4StopTokens`). Carrying over a turn closed by something other than `<turn|>` makes the
 * spelling differ from the upstream template by one token, but the conversation loses nothing (the
 * delta always starts "right after the closing token").
 *
 * NOTE: in the `"gemma4-unified"` chat dialect (gemma-4-12B-it) every generation prompt ends with
 * an empty thought channel, so a carried-over conversation keeps it at the start of each past model
 * turn, where a whole re-render (after overflow handling or a cut-off turn) drops it as the upstream
 * template does (the NOTE of `renderGemma4ChatTurn` in `./text/chat.ts`).
 *
 * ## Overflow handling is **injectable** (the revision of ADR 0083 decision 10)
 *
 * The low-level surface still says "truncation is the host's job" and ends by throwing
 * `GenerationCapacityError`. This layer has a default policy ({@link dropOldestTurns} = drop the
 * oldest user / assistant pair, keep system), replaceable as a whole through
 * {@link ChatSessionOptions.onOverflow}. A host that passes a throwing function can handle it
 * on its side as before.
 *
 * ## Branching ({@link ChatSession.fork} — ADR 0083 addendum 2026-10-11, ruling 8)
 *
 * Returns an independent session with the same settings and the same history. A conversation that
 * carries its KV is copied KV and all with `GenerationSequence.fork`; a conversation with an empty KV
 * (nothing sent yet, or the previous turn did not close with EOS) copies only the history (no GPU
 * copy — the branch's first `send` re-renders the whole, like the original). Regeneration is written
 * as this use rather than a `regenerate` (the doc of {@link ChatSession.fork}).
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
  type GenerationStream,
} from "../generation/sequence.ts";
import { createStopStringFilter } from "../text/detokenizer.ts";
import type { DefaultSampler } from "./config.ts";
import {
  chatStreamOf,
  completeChatTurn,
  decodeChatChunks,
  type ChatStop,
  type ChatStream,
  type PrefillProgress,
} from "./chat-turn.ts";
import type { SequenceOptions } from "./pipeline.ts";
import { type ChatMessage, chatPrompt, chatTurn } from "./text/chat.ts";
import type { Tokenizer } from "./text/tokenizer.ts";

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
  /** The number of logical positions needed to pass this turn (prompt + `maxNewTokens` − 1). */
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
 * MUST: **keep the last entry (the message being answered now)**. A history that dropped it passes
 * into rendering without an exception (`chatPrompt` does not constrain the last role), and an
 * answer gets attached to "a conversation with no question". An implementation that returns copies
 * (`turns.map((turn) => ({ ...turn }))`) is legitimate; the check compares values.
 */
export type ChatOverflowPolicy = (
  context: ChatOverflow,
) => readonly ChatMessage[] | Promise<readonly ChatMessage[]>;

/** The settings of one session (per-turn options are {@link ChatTurnOptions}). */
export type ChatSessionOptions = {
  /** The system message placed at the start of the conversation (none when omitted). */
  readonly system?: string;
  /** The upper limit of tokens generated in one turn (can be overridden per turn). */
  readonly maxNewTokens: number;
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
};

/**
 * Options that apply to one turn only (the same vocabulary and meaning as the request of
 * {@link Pipeline.chat}).
 *
 * MUST: {@link ChatSession.send} **copies** the contents **at issue** (ADR 0083 addendum
 * 2026-09-02) — rewriting them after the returned stream has started draining has no effect on the
 * running turn.
 */
export type ChatTurnOptions = {
  /** Defaults to the session's {@link ChatSessionOptions.maxNewTokens}. */
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

/**
 * The default overflow handling — **drop the oldest user / assistant pair** (the system message is
 * kept).
 *
 * Two entries are dropped at a time because dropping only one leaves "a conversation with only the
 * answer" or "a conversation with only the question", a broken context to show the model. When
 * nothing can be dropped, `turns` is returned as is (= the session fails with
 * `GenerationCapacityError` — this one message alone does not fit).
 */
export const dropOldestTurns: ChatOverflowPolicy = ({ turns, system }): readonly ChatMessage[] => {
  const floor = system === undefined ? 0 : 1;
  // The last entry (= the message being answered now) is never dropped.
  const droppable = turns.length - floor - 1;
  if (droppable <= 0) return turns;
  return [...turns.slice(0, floor), ...turns.slice(floor + Math.min(2, droppable))];
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
 * sequence layer's fields.
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
const sameTurn = (left: ChatMessage, right: ChatMessage): boolean =>
  left.role === right.role && left.content === right.content;

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
  readonly #maxNewTokens: number;
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
  /**
   * The conversation history (this layer's only mutable state — the sequence holds no transcript).
   */
  #turns: ChatMessage[];
  /** The current sequence (`undefined` = rebuild on the next turn). */
  #sequence: GenerationSequence | undefined;
  /** The number of history entries in that sequence's KV (0 = empty = re-render the whole). */
  #committed = 0;
  /** A turn is running (issued, and its stream has not ended yet). */
  #busy = false;
  /** The one dispose. **Not undefined means "disposed"** (no separate derived state). */
  #disposal: Promise<void> | undefined;

  constructor(host: ChatSessionHost, options: ChatSessionOptions) {
    this.#host = host;
    this.#system = options.system;
    this.#maxNewTokens = options.maxNewTokens;
    this.#sampler = options.sampler;
    this.#onOverflow = options.onOverflow ?? dropOldestTurns;
    this.#capacity = options.capacity ?? host.program.capacity;
    this.#speculative = options.speculative;
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
   * The conversation so far (if a system message is declared, it comes first).
   *
   * Returns a frozen **copy** — exposing the internal array as is would let the consumer's `push` /
   * `splice` silently break the mapping to the KV ("When the KV is carried over" of
   * {@link ChatSession}).
   */
  get turns(): readonly ChatMessage[] {
    return Object.freeze(this.#turns.map((turn) => Object.freeze({ ...turn })));
  }

  /**
   * Sends one message and streams **committed text pieces** (the same stream, the same stop reason
   * and the same `text()` as {@link Pipeline.chat}).
   *
   * If the previous turn closed with EOS, the KV is carried over and **only the delta** is streamed;
   * otherwise the sequence is rebuilt and the history re-rendered. A turn that does not fit goes to
   * overflow handling **before sending** ({@link ChatSessionOptions.onOverflow}).
   */
  send(text: string, options: ChatTurnOptions = {}): ChatStream {
    if (this.#disposal !== undefined) {
      throw new Error("gemma.ChatSession: cannot generate after dispose");
    }
    if (this.#busy) {
      throw new Error(
        "gemma.ChatSession: the previous turn has not finished yet" +
          " (1 session = 1 generation — drain the issued stream or close it with break)",
      );
    }
    // MUST: the request is **copied at issue** (ADR 0083 addendum 2026-09-02). The body (an async
    // generator) does not run until the first `next()`, so without reading everything here it would
    // run with "the values at the time draining started".
    const maxNewTokens = options.maxNewTokens ?? this.#maxNewTokens;
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
    const detokenizer = this.#host.tokenizer.createDetokenizer();
    // MUST: check the value ranges of the accepted set **before moving the session's state** (the
    // source of truth of the check is the one in `sequence.ts`). Placed later, an invalid request
    // would push a message onto the history and set `#busy`, "not failing until draining starts" —
    // from the caller's view, one mistake would make the session unusable.
    assertGenerationRequestValues(this.#host.program.vocabSize, {
      maxNewTokens,
      ...(stopTokens === undefined ? {} : { stopTokens }),
      ...(sampler === undefined ? {} : { sampler }),
    });

    const asked: ChatMessage = { role: "user", content: text };
    this.#turns.push(asked);
    this.#busy = true;

    let settle!: (stop: ChatStop) => void;
    let fail!: (error: unknown) => void;
    const done = new Promise<ChatStop>((resolve, reject) => {
      settle = resolve;
      fail = reject;
    });
    // A secondary notification path, so not an unhandled rejection even when nobody reads it.
    done.catch(() => {});

    const prepare = this.#prepare.bind(this);
    const finish = this.#finish.bind(this);
    const chunks = async function* (): AsyncGenerator<string, void, undefined> {
      let stream: GenerationStream | undefined;
      let failure: { readonly error: unknown } | undefined;
      /** The matched stop string (this layer's stop reason). */
      let matched: string | undefined;
      /** The text streamed in this turn (the assistant entry pushed onto the history). */
      let reply = "";
      try {
        const { sequence, prompt } = await prepare(maxNewTokens);
        stream = sequence.generate({
          prompt,
          maxNewTokens,
          ...(stopTokens === undefined ? {} : { stopTokens }),
          ...(sampler === undefined ? {} : { sampler }),
          ...(signal === undefined ? {} : { signal }),
        });
        const parts = decodeChatChunks(stream, detokenizer, stopStrings, onPrefill, onToken);
        try {
          for (;;) {
            const step = await parts.next();
            if (step.done === true) {
              matched = step.value;
              break;
            }
            reply += step.value;
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
        await completeChatTurn({
          where: "gemma.ChatSession.send",
          stream,
          matched,
          failure,
          settle,
          fail,
          cleanup: (stop) => finish(asked, reply, stop),
        });
      }
    };

    return chatStreamOf(
      closeableGenerator(chunks(), (failure) => {
        // prepare has not run, so only the provisionally added message is removed. The previous
        // turn's KV can still be carried over.
        this.#turns.pop();
        this.#busy = false;
        if (failure !== undefined) fail(failure.error);
        else settle({ reason: "closed", tokens: 0 });
      }),
      done,
    );
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
    const turns = this.#turns.map((turn) => ({ ...turn }));
    const committed = this.#committed;
    const copying = this.#sequence?.fork();
    const branch = new ChatSession(this.#host, {
      maxNewTokens: this.#maxNewTokens,
      capacity: this.#capacity,
      onOverflow: this.#onOverflow,
      ...(this.#system === undefined ? {} : { system: this.#system }),
      ...(this.#sampler === undefined ? {} : { sampler: this.#sampler }),
      ...(this.#speculative === undefined ? {} : { speculative: this.#speculative }),
    });
    branch.#turns = turns;
    if (copying === undefined) return Promise.resolve(branch);
    return copying.then((sequence) => {
      branch.#sequence = sequence;
      branch.#committed = committed;
      return branch;
    });
  }

  /**
   * Prepares this turn's sequence and prompt (running overflow handling first if it does not fit).
   *
   * MUST: the overflow decision happens **before allocating** the sequence. All it needs is `used`,
   * and when `#committed` is 0 the KV is empty = `used` is 0, so it can be decided without
   * allocating. In the reverse order, "a turn that truncates allocates a capacity's worth of KV once
   * and immediately discards it" (hit on the first turn of a conversation, and when the previous turn
   * closed by max-tokens / a stop string / an interruption / a failure = not rare).
   *
   * MUST: retries continue only when the history **shrank** (the check in `#shrink`), so the number
   * of attempts does not exceed the number of messages at entry — the guard below exists to fail by
   * name when that invariant breaks, and is never hit while it holds.
   */
  async #prepare(
    maxNewTokens: number,
  ): Promise<{ readonly sequence: GenerationSequence; readonly prompt: number[] }> {
    const attempts = this.#turns.length;
    for (let attempt = 0; ; attempt += 1) {
      // MUST: re-check after `await` (`send` checks at issue, but the body starts running later).
      // Draining an issued stream after disposing only the session would otherwise take a sequence
      // with fresh capacity from a live pipeline.
      if (this.#disposal !== undefined) {
        throw new Error("gemma.ChatSession: cannot generate after dispose");
      }
      const held = this.#sequence;
      // Carry on with a delta only when "exactly the last message is uncommitted". Streaming a delta
      // while out of step would silently lose part of the conversation, so the premise of carrying
      // over is itself a check.
      if (this.#committed !== 0 && this.#committed !== this.#turns.length - 1) {
        throw new Error(
          `gemma.ChatSession internal inconsistency: the KV holds ${this.#committed} turns but the history holds ` +
            `${this.#turns.length}`,
        );
      }
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
      const prompt =
        this.#committed === 0
          ? chatPrompt(this.#host.tokenizer, this.#turns)
          : chatTurn(this.#host.tokenizer, this.#turns[this.#turns.length - 1]);
      const detail = overflowOf(
        this.#host.program,
        this.#capacity,
        used,
        prompt.length,
        maxNewTokens,
      );
      if (detail === undefined) {
        if (held !== undefined) return { sequence: held, prompt };
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
        return { sequence: created, prompt };
      }
      if (attempt >= attempts) {
        throw new Error(
          `gemma.ChatSession internal inconsistency: overflow handling did not converge in ${attempts} attempts`,
        );
      }
      await this.#shrink(detail);
    }
  }

  /**
   * Runs overflow handling once (replaces the history and discards the sequence whose mapping to the
   * KV broke).
   */
  async #shrink(detail: GenerationCapacityDetail): Promise<void> {
    const before = this.#turns.length;
    const asked = this.#turns[before - 1];
    const needed = neededOf(detail);
    const next = await this.#onOverflow({
      turns: this.turns,
      system: this.#system,
      capacity: detail.limit,
      needed,
    });
    // MUST: the message being answered now (the last entry) remains. Letting it through dropped
    // would make `#prepare` re-render "a conversation with no question", and since
    // `chatPrompt` does not constrain the last role either, only an answer would be committed
    // with **zero exceptions**. The comparison is by value ({@link sameTurn}) — a legitimate policy
    // that returns copies is not rejected.
    const last = next.at(-1);
    if (last === undefined || !sameTurn(last, asked)) {
      throw new Error(
        `gemma.ChatSession: overflow handling dropped the current utterance (the last one)` +
          ` (history ${before} turns → ${next.length} — keep the utterance being answered now)`,
      );
    }
    if (next.length >= before) {
      throw new GenerationCapacityError(
        `The conversation does not fit: the ${detail.constraint} limit is ${detail.limit}, but ${needed} are needed` +
          ` (overflow handling did not shrink the history of ${before} turns — shorten the utterance or lower` +
          ` maxNewTokens. At this length maxNewTokens ≤ ${detail.maxNewTokens})`,
        detail,
      );
    }
    this.#turns = next.map((turn) => ({ ...turn }));
    // A truncated history is re-rendered from the start, so the context is returned and rebuilt.
    await this.#releaseSequence();
  }

  /**
   * The turn's wrap-up (appending to the history, deciding whether to carry the KV, accepting the
   * next `send`).
   */
  async #finish(asked: ChatMessage, reply: string, stop: ChatStop | undefined): Promise<void> {
    try {
      if (stop?.reason === "eos") {
        // Closed with the distribution's EOS = the model turn is closed. The entry is pushed even
        // when the body is empty (the KV holds an empty model turn, so without it the re-render would
        // be a different conversation from the KV).
        this.#turns.push({ role: "assistant", content: reply });
        this.#committed = this.#turns.length;
        return;
      }
      // max-tokens / a stop string / an interruption / a failure leave the model turn unclosed = the
      // premise of a delta is missing.
      if (reply !== "") this.#turns.push({ role: "assistant", content: reply });
      else {
        // A turn that emitted no character is **treated as never having happened** (no question
        // without an answer is left in the history). The comparison is by value ({@link sameTurn}) —
        // when overflow handling copies the history, references break.
        const last = this.#turns.at(-1);
        if (last !== undefined && sameTurn(last, asked)) this.#turns.pop();
      }
      await this.#releaseSequence();
    } finally {
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
