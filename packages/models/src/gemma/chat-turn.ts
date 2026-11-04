/**
 * The **conversion and cleanup** of one chat turn (the parts {@link Pipeline.chat} and
 * `ChatSession.send` share) — generation events → committed text pieces, building the stop
 * reason, closing the turn, and shaping the per-run observation callback.
 *
 * This module **owns** no Session, no serialization chain and no GPU (ownership and queueing live in
 * `./pipeline.ts`). It only turns the event stream it receives into text and keeps the cleanup order
 * on one path, so the stop-string contract and the call rules of the observation callback can be
 * pinned without a real GPU (`tests/gemma4_chat_test.ts`).
 *
 * MUST: every module is side-effect free (no import-time execution, no global mutable state —
 * CLAUDE.md).
 */

import type {
  GenerationEvent,
  GenerationRunPhase,
  GenerationDecodeAhead,
  GenerationSpeculation,
  GenerationStop,
  GenerationStream,
} from "../generation/sequence.ts";
import type { StopStringFilter, StreamingDetokenizer } from "../text/detokenizer.ts";

/**
 * One prefill progress notification (`chunk / chunks` is the progress as is).
 *
 * `chunk` is the number of **committed** chunks (1-based), with the same meaning and the same number
 * as the `prefill` of `GenerationEvent` (this layer is the text surface, so it does not expose the
 * event type itself).
 */
export type PrefillProgress = {
  readonly chunk: number;
  readonly chunks: number;
};

/**
 * "Which run this notification is" as received by the observation callback
 * ({@link PipelineOptions.onRunDiagnostics}).
 *
 * The callback receives **one notification per run**, and this value says what that run was (it is
 * not a count of the order). All numbers are 1-based — the `chunk` / `chunks` of `prefill` are the
 * same numbers as the `prefill` of `GenerationEvent` (committed chunk count), the `step` of `decode`
 * is the number of the decode run **within that turn**, and the `cycle` of `draft` / `verify` is the
 * speculation cycle number (the two runs of one cycle carry the same number).
 *
 * MUST: the receiver branches on this value, not on the number of notifications — for a prompt
 * split into several chunks, "only the first notification is prefill" does not hold.
 * MUST: this is the generation surface's `GenerationRunPhase` **itself** (no copied type — a copy
 * would keep type-checking when only one side gains a branch). All this layer adds is "which
 * Session's diagnostics to read".
 */
export type RunPhase = GenerationRunPhase;

/**
 * The stop reason of one chat turn.
 *
 * The sequence layer's reasons (`eos` / `stop-token` / `max-tokens` / `aborted` / `closed`) plus the
 * one only this layer can decide (a match of {@link ChatOptions.stopStrings}). `tokens` means
 * the same as in the sequence layer (the number of tokens the turn generated — a stop token counts
 * as 1); for `stop-string` it counts **up to the token that emitted the piece containing the stop
 * string**.
 */
export type ChatStop =
  | GenerationStop
  | {
      readonly reason: "stop-string";
      /** The matched stop string (not included in the output). */
      readonly stopString: string;
      readonly tokens: number;
      /**
       * The speculation accounting (`GenerationStop.speculation` copied as is — present only for
       * turns of a speculative sequence).
       *
       * MUST: copy it. This branch rebuilds the object to replace the reason, so without the copy
       * "only turns closed by a stop string lose the accounting" (a loss that raises no exception).
       */
      readonly speculation?: GenerationSpeculation;
      /**
       * The run-ahead counters (`GenerationStop.decodeAhead` copied as is — present only for turns
       * of a sequence built with the run-ahead). MUST: copy it, for the same reason as `speculation`.
       */
      readonly decodeAhead?: GenerationDecodeAhead;
    };

/**
 * A stream of text pieces (drained with `for await`) + the stop reason + a whole-text entry point.
 *
 * The pieces are only what the streaming detokenizer has **committed** (ADR 0084 decision 4); a
 * byte_fallback in progress is carried over to the next token. Concatenated they equal
 * `decode(all token ids)` (except for a turn cut by a stop string, which ends just before it).
 *
 * MUST: **a stream can be consumed in only one way** — mixing iteration (`for await`) with
 * {@link ChatStream.text}, or iterating twice, throws synchronously. Generation runs only once,
 * so a second consumer would receive only "the rest" (only the side that drained first has the
 * body) — a mix-up that raises no exception, so the entry point blocks it.
 *
 * MUST: `done` is a **secondary** notification path (the same discipline as
 * `GenerationStream.done`) — the primary path for a failure is the iterable throwing; `done` only
 * rejects with the same exception.
 */
export type ChatStream = AsyncIterable<string> & {
  readonly done: Promise<ChatStop>;
  /**
   * The whole text, drained and concatenated (the entry point for callers that do not need
   * incremental display).
   *
   * It drains the same stream in the same order as iteration, so the result of `text()` equals
   * "all pieces concatenated". When the stop reason is needed, also read
   * {@link ChatStream.done} (after `text()` is fine).
   */
  text(): Promise<string>;
};

/**
 * Generation events → **committed text pieces** (decode → stop-string check). The return value is
 * the matched stop string (`undefined` = the stream ended without stopping).
 *
 * MUST: stopping on a stop string exits with `return` — leaving a `for await` calls the event
 * stream's `return()`, so the sequence is closed with **the same cleanup as an interruption
 * (`break`)**. Writing a separate way to close it would split the KV committed consistency (the
 * uncommitted frontier of 1 token) across two implementations.
 *
 * NOTE: unlike a stop token (sequence layer), a stop string can only be decided **after** decoding —
 * one stop string can be split across several tokens, and one token can span the end of a stop
 * string and the following text. That is why the check lives in two layers (ADR 0083 addendum
 * 2026-09-02).
 *
 * NOTE: this is an **internal entry point** not exposed in the barrel (`mod.ts` / `./gemma`) (the
 * public entry is {@link Pipeline.chat} only). It is exported because at this unit the
 * stop-string contract can be pinned without a real GPU (`tests/gemma4_chat_test.ts`).
 */
export const decodeChatChunks = async function* (
  events: AsyncIterable<GenerationEvent>,
  detokenizer: StreamingDetokenizer,
  stopStrings: StopStringFilter,
  onPrefill?: (progress: PrefillProgress) => void,
  onToken?: (id: number) => void,
): AsyncGenerator<string, string | undefined, undefined> {
  for await (const event of events) {
    if (event.kind === "prefill") {
      // The text surface has no prefill piece (no character of the body has been emitted yet), so
      // only the progress goes to the observer. Exceptions are not caught (fail loudly — an error in
      // the caller's callback is not swallowed).
      onPrefill?.({ chunk: event.chunk, chunks: event.chunks });
      continue;
    }
    onToken?.(event.id);
    const chunk = stopStrings.push(detokenizer.push(event.id));
    if (chunk.text !== "") yield chunk.text;
    if (chunk.matched !== undefined) return chunk.matched;
  }
  // The detokenizer's carry-over (a byte_fallback run), once committed, also goes through the check
  // — the last character of a stop string can sit inside that run.
  const tail = stopStrings.push(detokenizer.finish());
  if (tail.text !== "") yield tail.text;
  if (tail.matched !== undefined) return tail.matched;
  // A turn that ended without stopping finally emits what was held back as a prefix (not a single
  // character is dropped — holding back is a delay for the check, not a truncation of the output).
  const held = stopStrings.finish();
  if (held !== "") yield held;
  return undefined;
};

/**
 * A piece generator + a stop reason → the public {@link ChatStream} (an entry point that can
 * be **consumed in only one way**).
 *
 * MUST: iteration and {@link ChatStream.text} drain **the same generator** (no second path) —
 * generation runs only once, so if the whole-text entry point had its own loop, "the result depends
 * on which way it was read" could be written. The second consumer would silently get an empty
 * result without an exception, so the entry point blocks it.
 *
 * NOTE: like {@link decodeChatChunks}, this is an internal entry point not exposed in the barrel.
 */
export const chatStreamOf = (
  chunks: AsyncGenerator<string, void, undefined>,
  done: Promise<ChatStop>,
): ChatStream => {
  let claimed: "iteration" | "text()" | undefined;
  const claim = (how: "iteration" | "text()"): void => {
    if (claimed !== undefined) {
      throw new Error(
        `gemma.ChatStream: one stream can be consumed in only one way` +
          ` (already consumed by ${claimed} — ${how} tries to read the same generation again)`,
      );
    }
    claimed = how;
  };
  return {
    [Symbol.asyncIterator]: (): AsyncGenerator<string, void, undefined> => {
      claim("iteration");
      return chunks;
    },
    done,
    // Not async — the mixed-use check fails **synchronously** (not in a form that is noticed only
    // when the return value is awaited; the same discipline as the lifetime check of `generate`).
    text: (): Promise<string> => {
      claim("text()");
      return joinChunks(chunks);
    },
  };
};

/**
 * Shapes the observation callback ({@link PipelineOptions.onRunDiagnostics}) into the
 * generation surface's `onRun` hook.
 *
 * The callback lives in the pipeline layer because `GenerationSequence` is **pipeline-independent**
 * (it knows neither Session nor diagnostics — ADR 0083). The generation surface calls this hook once
 * per run, in the **synchronous section** after it has read that run's outputs, so all this layer
 * does is decide "which Session's diagnostics to read" from `phase.kind`.
 *
 * It used to wrap the event stream (`GenerationEvent`) and **derive** the run count
 * (`withRunDiagnostics`). The derivation carried two exceptions — the first token right after
 * prefill comes with no run, and the last decode run that drew a stop token does not appear in the
 * stream (it was filled in from `done`) — and with speculation one verify run emits several tokens,
 * so the derivation itself breaks down. With the issuer of the run naming it directly (ADR 0083
 * addendum (hook)), both exceptions disappear.
 *
 * MUST: return `undefined` when there is no observation callback (no hook passed = the generation
 * surface never calls it).
 * MUST: `draft` reads the diagnostics of the **borrower** (the drafter Session). Passing the
 * lender's would deliver the value of "the verify run before it" as the diagnostics of the draft
 * run (a mix-up that raises no exception).
 *
 * NOTE: the diagnostics type is a type parameter to show in the type that this function **reads
 * none of the diagnostics' content** (it only passes them through to the callback), and so that the
 * check of the call rules (`gemma4_chat_test.ts`) can be written without a real Session.
 * {@link Gemma4State} satisfies it with `SessionDiagnostics` as is.
 * NOTE: `export` is for the test that calls the check directly (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */
export const runDiagnosticsHook = <D>(state: {
  readonly session: { diagnostics: () => D };
  readonly drafter?: { readonly session: { diagnostics: () => D } };
  readonly onRunDiagnostics?: (diagnostics: D, phase: RunPhase) => void;
}): ((phase: RunPhase) => void) | undefined => {
  const listener = state.onRunDiagnostics;
  if (listener === undefined) return undefined;
  return (phase: RunPhase): void => {
    if (phase.kind !== "draft") {
      listener(state.session.diagnostics(), phase);
      return;
    }
    const drafter = state.drafter;
    // A draft run happens only on the drafter Session (no drafter means broken bookkeeping —
    // silently passing the lender's diagnostics would accumulate another run's values under the
    // draft name).
    if (drafter === undefined) {
      throw new Error(
        "gemma.Pipeline: a draft run observation arrived although there is no drafter",
      );
    }
    listener(drafter.session.diagnostics(), phase);
  };
};

/**
 * The stop reason of a turn closed by a stop string (the one builder shared by `chat` and
 * `ChatSession.send`).
 *
 * The reason and its spelling are this layer's decision, but `tokens`, `speculation` and
 * `decodeAhead` **copy the inner values as is** (this layer does not recount). The construction is kept on one path so that,
 * when a copied field is added, one entry point does not stay stale.
 */
export const stopStringOf = (stopString: string, inner: GenerationStop): ChatStop => ({
  reason: "stop-string",
  stopString,
  tokens: inner.tokens,
  ...(inner.speculation === undefined ? {} : { speculation: inner.speculation }),
  ...(inner.decodeAhead === undefined ? {} : { decodeAhead: inner.decodeAhead }),
});

/**
 * The one turn cleanup (shared by `chat` and `ChatSession.send`).
 *
 * MUST: call `release` **unconditionally**. If a throw from `cleanup` (returning the sequence,
 * closing the session) skipped returning the lease, the serialization chain would never get the
 * previous step's outcome and every later `chat` / `dispose` would wait forever — one exception
 * would make a pipeline that never runs again (a path exists where `context.dispose` propagates a
 * `flush` failure on device loss). The order stays flush-before-destroy: "`cleanup` → `release`".
 *
 * MUST: when the body (`failure`) has also failed, carry **both**. This is called from the caller's
 * `finally`, so an exception thrown here replaces the body's — put both, unwrapped, into an
 * `AggregateError` (`errors[0]` is the body, `errors[1]` the cleanup; the interruption check
 * `error === signal.reason` still works on `errors[0]`).
 *
 * NOTE: this is a separate function so that no control-flow statement sits in the caller's
 * `finally` (`no-unsafe-finally` forbids "silently discarding the original exception"; here it is
 * folded, not discarded).
 */
export const closeChatTurn = async (
  where: string,
  failure: { readonly error: unknown } | undefined,
  cleanup: () => Promise<void>,
  release?: () => void,
): Promise<void> => {
  try {
    await cleanup();
  } catch (error) {
    if (failure === undefined) throw error;
    throw new AggregateError(
      [failure.error, error],
      `${where}: both the turn body and the clean-up failed`,
    );
  } finally {
    release?.();
  }
};

/** Notifies `done` with the same outcome as the iterable, after cleanup and returning the lease. */
export const completeChatTurn = async (options: {
  readonly where: string;
  readonly stream?: GenerationStream;
  readonly matched?: string;
  readonly failure?: { readonly error: unknown };
  readonly cleanup: (stop: ChatStop | undefined) => Promise<void>;
  readonly release?: () => void;
  readonly settle: (stop: ChatStop) => void;
  readonly fail: (error: unknown) => void;
}): Promise<void> => {
  let failure = options.failure;
  let stop: ChatStop | undefined;
  try {
    const inner =
      options.stream === undefined
        ? ({ reason: "closed", tokens: 0 } satisfies ChatStop)
        : await options.stream.done;
    // Keep the existing contract for an interruption: the iterable throws the reason and done
    // returns aborted.
    if (failure === undefined || inner.reason === "aborted") {
      stop = options.matched === undefined ? inner : stopStringOf(options.matched, inner);
    }
  } catch (error) {
    failure ??= { error };
  }
  try {
    await closeChatTurn(options.where, failure, () => options.cleanup(stop), options.release);
  } catch (error) {
    options.fail(error);
    throw error;
  }
  if (stop === undefined && failure !== undefined) {
    options.fail(failure.error);
    throw failure.error;
  }
  options.settle(stop ?? { reason: "closed", tokens: 0 });
};

/** Drains and concatenates the pieces (the body of {@link ChatStream.text}). */
const joinChunks = async (chunks: AsyncIterable<string>): Promise<string> => {
  let text = "";
  for await (const chunk of chunks) text += chunk;
  return text;
};
