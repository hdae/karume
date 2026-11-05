// The checks of the conversation management of `ChatSession` (`src/gemma/chat-session.ts`). No
// GPU.
//
// What this layer owns is **only the mapping between the conversation (the sequence of messages)
// and the KV**; generation itself rides on `GenerationSequence`, and rendering, decoding and stop
// strings on the same implementation as `chat()`. So the checks can be written as "which prompt
// was fed into which sequence" — plug in a fake sequence that answers from a script
// (`ChatSessionHost` = the surface the session reads from the pipeline), and build a **real**
// tokenizer from the same fixture assets as `gemma4_chat_test.ts` (the equality of the delta
// rendering can only be seen with the real spellings).
//
// Eleven checks:
//
// ① **KV continuation**: the second turn feeds only the delta (`chatTurn`) and does not rebuild
//    the sequence
// ② **After a stop that cannot be continued** (max-tokens) the sequence is dropped and the history
//    is re-fed as a whole rendering — with ① alone an implementation that "always continues" would
//    stay green (continuing a delta onto a conversation whose model turn is not closed breaks the
//    turn boundary with zero errors)
// ③ **A consumer-side `break`** reaches the inner event sequence (`done` settles), and only the
//    chunks that came out enter the history
// ④ **Overflow handling**: decided before sending; the default policy (`dropOldestTurns`) drops
//    the oldest pair and keeps the system message. The dropped history no longer maps onto the
//    KV, so the sequence is rebuilt
// ⑤ **Fail loudly once nothing is left to drop** (`GenerationCapacityError`) — no silent
//    shrinking. The message of the turn that never happened does not stay in the history
// ⑥ **Policy injection** takes effect (returning a history / passing a throw through as is). A
//    policy that returns a history that does not shrink fails loudly instead of looping retries
//    forever
// ⑦ **One session = one generation** (a concurrent send / a send after dispose throws
//    synchronously), and the stream `send` returns has the same contract as `chat()` (stop
//    reason, `text()`, consumable in only one way)
// ⑧ **A failure raised in this layer** (a throw from `onPrefill`, an unknown id in the decoder)
//    does not let `done` claim success. The inner one is closed with `return()` and resolves with
//    `closed`, so carrying that as is would make the failure disappear
// ⑨ **A failing cleanup does not wedge the session** (`#busy` always returns). If the body failed
//    too, both are carried (`AggregateError`)
// ⑩ **The accepted set is synchronous** (an invalid request soils neither the history nor
//    `#busy`) / **the overflow decision comes before allocation** (a turn that does not fit takes
//    no KV at all) / an overflow policy **keeps the tail** (decided by value)
// ⑪ **Branching (`fork`)**: an independent session that inherits the history and the KV; a
//    conversation with an empty KV produces no copy; a running or disposed session throws
//    synchronously; the copy is taken in the synchronous section of the call (a send right after
//    does not enter the branch)

import { readTextFile } from "@karume/test-support/fs";
import { assert, assertEquals, assertNotEquals, assertRejects, assertThrows } from "@std/assert";
import { beforeAll, describe, it, test } from "vite-plus/test";
import { ModelInputError } from "../src/errors.ts";
import {
  dropOldestTurns,
  type ChatOverflow,
  type ChatReply,
  ChatSession,
  type ChatSessionHost,
  type ChatSessionStream,
} from "../src/gemma/chat-session.ts";
import type { DefaultSampler } from "../src/gemma/config.ts";
import type { GenerationProgram } from "../src/generation/program.ts";
import type { SamplerSpec } from "../src/generation/sampler.ts";
import {
  GenerationCapacityError,
  type GenerationEvent,
  type GenerationRequest,
  type GenerationSequence,
  type GenerationStop,
  type GenerationStream,
} from "../src/generation/sequence.ts";
import { createBpeModel } from "../src/text/bpe.ts";
import type { GemmaTokenizerAssets } from "../src/gemma/text/asset.ts";
import { Tokenizer } from "../src/gemma/text/tokenizer.ts";
import {
  type ChatMessage,
  chatPrompt,
  type ChatTool,
  type ChatToolMessage,
  chatToolResults,
  chatTurn,
  gemma4StopTokens,
  renderGemma4Chat,
} from "../src/gemma/text/chat.ts";

type Fixture = {
  readonly asset: {
    readonly vocab: [number, string][];
    readonly merges: [number, number, number][];
    readonly byteIds: number[];
    readonly addedTokens: [string, number][];
    readonly specialIds: number[];
    readonly unkId: number;
    readonly bosId: number;
    readonly eosId: number;
  };
};

const fixture = JSON.parse(
  await readTextFile(new URL("fixtures/gemma-text/gemma4-chat.json", import.meta.url)),
) as Fixture;

/** Builds the asset table from the fixture subset (the same shape as `gemma4_chat_test.ts`). */
const assets: GemmaTokenizerAssets = {
  spec: { postProcessor: "none" },
  model: createBpeModel({
    vocab: fixture.asset.vocab.map(([id, token]) => [id, token] as const),
    merges: fixture.asset.merges.map(([left, right, rank]) => [left, right, rank] as const),
    byteIds: fixture.asset.byteIds,
  }),
  addedTokens: new Map(fixture.asset.addedTokens),
  specialIds: new Set(fixture.asset.specialIds),
  unkId: fixture.asset.unkId,
  bosId: fixture.asset.bosId,
  eosId: fixture.asset.eosId,
};
const tokenizer = new Tokenizer(assets);
const STOP_TOKENS = gemma4StopTokens(tokenizer);
/** One of the stop tokens the distribution declares (the marker that closes a turn). */
const END_OF_TURN = tokenizer.addedTokenId("<turn|>") as number;

const SYSTEM = "You are terse.";
const SYSTEM_MESSAGE: ChatMessage = { role: "system", content: SYSTEM };
const MAX_NEW_TOKENS = 16;

/**
 * The capacity used by the overflow checks = exactly `chunkLength` (the **smallest** capacity the
 * construction-time check lets through).
 *
 * `ChatSession` checks `chunkLength ≤ capacity ≤ maxPosition` at construction, so creating the
 * overflow with "a capacity that does not hold even one chunk" makes that check fail first, and
 * the overflow path is never reached. The overflow is created with **the length of the message**
 * ({@link LONG_ASK}), not with the capacity.
 */
const TIGHT_CAPACITY = 32;

/** A message that surely overflows {@link TIGHT_CAPACITY} in one turn (the prompt alone exceeds it). */
const LONG_ASK = "Name a color, then name another color, then explain the difference between them.";

/**
 * One turn of the script. `closes` = closed with the distribution's stop token (the next turn can
 * continue as a delta).
 */
type Answer = {
  readonly text: string;
  readonly closes: boolean;
  /** The stop token a closing answer ends with (default `<turn|>`; `<|tool_response>` after a call). */
  readonly stop?: number;
};

/** The number of logical positions the turn adds to the conversation (text + stop token). */
const answerCost = (answer: Answer): number =>
  tokenizer.encode(answer.text).length + (answer.closes ? 1 : 0);

const programOf = (capacity: number): GenerationProgram =>
  Object.freeze({
    chunkLength: 32,
    // The position table is made wide enough (the only limit this check exercises is capacity).
    maxPosition: 4096,
    capacity,
    vocabSize: 262144,
    stopTokens: Object.freeze([...STOP_TOKENS]),
    // A fake with chunkLength 32 needs no extra prefill shapes (the session does not read this field).
    chunkBuckets: Object.freeze([]),
  });

type FakeHost = ChatSessionHost & {
  /** The number of sequences created (whether it rebuilt shows here). */
  created(): number;
  /** The number of sequences disposed. */
  disposed(): number;
  /** The prompts `generate` received, in issue order. */
  prompts(): readonly (readonly number[])[];
  /** The `maxNewTokens` `generate` received, in issue order (`undefined` = not passed). */
  requestedMaxNewTokens(): readonly (number | undefined)[];
  /** The capacities `sequence()` received, in issue order (whether the session's knob reaches it). */
  capacities(): readonly (number | undefined)[];
  /**
   * The speculative-decoding settings `sequence()` received, in issue order.
   *
   * `undefined` means **the field was not passed at all** (passing `speculative: undefined` cancels
   * the pipeline-side default "attach when a drafter is present", so it is a different thing even
   * though the value is the same).
   */
  speculatives(): readonly (boolean | "always" | undefined)[];
  /** Whether the options `sequence()` received had a `speculative` field, in issue order. */
  speculativeKeys(): readonly boolean[];
  /** The sampler specs `generate` received, in issue order (the observation point of the three-level resolution). */
  samplers(): readonly (SamplerSpec | undefined)[];
  /** The number of calls to `GenerationSequence.fork` (whether a KV copy was made shows here). */
  forks(): number;
};

/** Fault injection of the fake sequence (holding `sequence()` and rejecting `dispose()`). */
type FakeFaults = {
  /** Makes `dispose` reject (a copy of the shape where `flush` fails on device loss). */
  readonly disposeError?: unknown;
  readonly holdDispose?: () => Promise<void>;
  /**
   * Holds the resolution of `sequence()` (it does not return until the received `release` is
   * called).
   *
   * The hook that opens the window "disposed during the allocation's `await`" — in the real code
   * `createGenerationSequence` is asynchronous around the GPU allocation too, so the window exists.
   */
  readonly holdSequence?: (release: () => void) => void;
  /** Makes `GenerationSequence.fork` reject (a copy of a failed allocation of the copy). */
  readonly forkError?: unknown;
};

/** Identifies an abort (the fake copies the same split as `isAbortOf` of the real `sequence.ts`). */
const isAbortOf = (error: unknown, signal: AbortSignal | undefined): boolean =>
  signal !== undefined && signal.aborted && error === signal.reason;

/**
 * The host that hands out fake sequences answering from a script.
 *
 * `used` advances by the same derivation as the real one (the prompt length + the number of
 * sampled tokens — the stop token counts as one). If this drifts, the overflow checks lose their
 * meaning, so this part alone copies the real contract.
 */
const fakeHost = (
  answers: readonly Answer[],
  program: GenerationProgram,
  faults: FakeFaults = {},
  /** The recommended sampler the distribution declares (the lowest level of the resolution). */
  defaultSampler?: DefaultSampler,
  /** The tokenizer the host hands out (its dialect decides what the session renders). */
  hostTokenizer: Tokenizer = tokenizer,
): FakeHost => {
  const prompts: number[][] = [];
  const requestedMaxNewTokens: (number | undefined)[] = [];
  const capacities: (number | undefined)[] = [];
  const speculatives: (boolean | "always" | undefined)[] = [];
  const speculativeKeys: boolean[] = [];
  const samplers: (SamplerSpec | undefined)[] = [];
  let created = 0;
  let disposed = 0;
  let forks = 0;
  let turn = 0;
  /**
   * Builds one fake sequence (it takes only the starting `used` — a branch continues from the
   * original's `used`).
   *
   * The script (`answers`) is one for the whole host, so even after a branch the next turn returns
   * the next answer of the script.
   */
  const makeSequence = (capacity: number, startUsed: number): GenerationSequence => {
    let used = startUsed;
    let gone = false;
    const sequence: GenerationSequence = {
      capacity,
      get used(): number {
        return used;
      },
      generate(request: GenerationRequest): GenerationStream {
        assert(!gone, "generate on a disposed sequence");
        prompts.push([...request.prompt]);
        requestedMaxNewTokens.push(request.maxNewTokens);
        samplers.push(request.sampler);
        const answer = answers[turn];
        assert(answer !== undefined, `the script has no turn ${turn + 1}`);
        turn += 1;
        const encoded = hostTokenizer.encode(answer.text);
        // A request without `maxNewTokens` runs into the limit as the real one does: the room is
        // resolved here, and an answer that outgrows it is cut and closes as `capacity`.
        const room = Math.min(capacity, program.maxPosition) - used - request.prompt.length + 1;
        const full =
          request.maxNewTokens === undefined && encoded.length + (answer.closes ? 1 : 0) > room;
        assert(
          request.maxNewTokens !== undefined || answer.closes || full,
          "the script: an answer without maxNewTokens either closes or outgrows the room",
        );
        const closes = answer.closes && !full;
        const ids = full ? encoded.slice(0, room) : encoded;
        used += request.prompt.length;
        const stop: GenerationStop = closes
          ? { reason: "eos", token: answer.stop ?? END_OF_TURN, tokens: ids.length + 1 }
          : { reason: full ? "capacity" : "max-tokens", tokens: ids.length };

        let settle!: (value: GenerationStop) => void;
        const done = new Promise<GenerationStop>((resolve) => {
          settle = resolve;
        });
        const events = async function* (): AsyncGenerator<GenerationEvent, void, undefined> {
          let emitted = 0;
          let stopped: GenerationStop | undefined;
          try {
            request.signal?.throwIfAborted();
            yield { kind: "prefill", chunk: 1, chunks: 1 };
            for (const id of ids) {
              // An abort is observed between steps (the real one can observe it only between runs).
              request.signal?.throwIfAborted();
              used += 1;
              emitted += 1;
              yield { kind: "token", id, position: used };
            }
            // The stop token is not emitted in the sequence but stays in the conversation (the uncommitted frontier).
            if (closes) used += 1;
            stopped = stop;
          } catch (error) {
            // MUST: the side that catches the abort sets `aborted` and then rethrows (the same
            // shape as the real `sequence.ts`). Without copying this, `done` would resolve with
            // `closed`, and the test would see a mistake in the fake's copy, not the
            // implementation.
            if (isAbortOf(error, request.signal)) {
              stopped = { reason: "aborted", tokens: emitted };
            }
            throw error;
          } finally {
            // Folded with `break` / `return()` it is `closed` (the same split as the real one).
            settle(stopped ?? { reason: "closed", tokens: emitted });
          }
        };
        const iterable = events();
        return { [Symbol.asyncIterator]: () => iterable, done };
      },
      fork(): Promise<GenerationSequence> {
        // Like the real one, a disposed sequence fails synchronously.
        if (gone) throw new Error("GenerationSequence: cannot fork after dispose");
        forks += 1;
        if (faults.forkError !== undefined) return Promise.reject(faults.forkError);
        // Like a real branch, it inherits the positions the conversation occupies (`used`) as is.
        return Promise.resolve(makeSequence(capacity, used));
      },
      async dispose(): Promise<void> {
        gone = true;
        await faults.holdDispose?.();
        disposed += 1;
        return faults.disposeError === undefined
          ? Promise.resolve()
          : Promise.reject(faults.disposeError);
      },
    };
    return sequence;
  };
  return {
    tokenizer: hostTokenizer,
    program,
    defaultSampler,
    created: () => created,
    disposed: () => disposed,
    prompts: () => prompts,
    requestedMaxNewTokens: () => requestedMaxNewTokens,
    capacities: () => capacities,
    speculatives: () => speculatives,
    speculativeKeys: () => speculativeKeys,
    samplers: () => samplers,
    forks: () => forks,
    sequence: (options = {}): Promise<GenerationSequence> => {
      created += 1;
      capacities.push(options.capacity);
      speculatives.push(options.speculative);
      speculativeKeys.push(Object.hasOwn(options, "speculative"));
      const sequence = makeSequence(options.capacity ?? program.capacity, 0);
      const hold = faults.holdSequence;
      if (hold === undefined) return Promise.resolve(sequence);
      return new Promise<GenerationSequence>((resolve) => hold(() => resolve(sequence)));
    },
  };
};

test("ChatSession: the second turn feeds only the delta and does not rebuild the sequence", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  assertEquals(await session.send("Name a color.").text(), "Blue.");
  assertEquals(await session.send("Another one.").text(), "Red.");

  assertEquals(host.created(), 1, "with the KV continued there is only one sequence");
  assertEquals(
    host.prompts()[0],
    chatPrompt(tokenizer, [SYSTEM_MESSAGE, { role: "user", content: "Name a color." }]),
    "the first turn is the whole rendering including `<bos>`",
  );
  assertEquals(
    host.prompts()[1],
    chatTurn(tokenizer, { role: "user", content: "Another one." }),
    "the second turn is only the delta (`pendingToken` prepends the `<turn|>` that closes the previous turn)",
  );
  assertEquals(session.turns, [
    SYSTEM_MESSAGE,
    { role: "user", content: "Name a color." },
    { role: "assistant", content: "Blue." },
    { role: "user", content: "Another one." },
    { role: "assistant", content: "Red." },
  ]);
});

test("ChatSession: after a turn not closed by a stop token, the whole conversation is re-rendered", async () => {
  // After a turn cut off by max-tokens the model turn is not closed = no premise for a delta.
  const host = fakeHost(
    [
      { text: "Blue", closes: false },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  const first = session.send("Name a color.");
  assertEquals(await first.text(), "Blue");
  assertEquals(await first.done, {
    reason: "max-tokens",
    tokens: tokenizer.encode("Blue").length,
  });
  assertEquals(await session.send("Another one.").text(), "Red.");

  assertEquals(host.created(), 2, "after a cut-off turn the sequence is rebuilt");
  assertEquals(host.disposed(), 1, "the dropped sequence is disposed");
  assertEquals(
    host.prompts()[1],
    chatPrompt(tokenizer, [
      SYSTEM_MESSAGE,
      { role: "user", content: "Name a color." },
      { role: "assistant", content: "Blue" },
      { role: "user", content: "Another one." },
    ]),
    "the second turn is the whole rendering of the history",
  );
});

test("ChatSession: a turn whose draining stopped is folded down to the inner sequence, keeping only the text that came out", async () => {
  // A consumer-side `break` must reach the inner event sequence with "the same cleanup as an
  // abort" (otherwise the sequence stays running and `done` never settles).
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const stream = session.send("Name a color.");
  let first = "";
  for await (const chunk of stream) {
    first = chunk;
    break;
  }
  assertEquals((await stream.done).reason, "closed", "the inner event sequence is folded too");
  assert(
    first !== "" && "Blue.".startsWith(first),
    `the first chunk is not the head of the text: ${first}`,
  );
  assertEquals(
    session.turns,
    [
      { role: "user", content: "Name a color." },
      { role: "assistant", content: first },
    ],
    "only the streamed chunks enter the history",
  );

  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "after a turn that is not closed the sequence is rebuilt");
});

test("ChatSession overflow: the default policy drops the oldest pair and rebuilds keeping the system message", async () => {
  const answers: readonly Answer[] = [
    { text: "Blue.", closes: true },
    { text: "Red.", closes: true },
  ];
  const first = { role: "user", content: "Name a color." } as const;
  const second = { role: "user", content: "Another one." } as const;
  // A capacity where the second turn misses by **exactly 1** (no shape that would pass while continuing the delta).
  const used = chatPrompt(tokenizer, [SYSTEM_MESSAGE, first]).length + answerCost(answers[0]);
  const delta = chatTurn(tokenizer, second).length;
  const host = fakeHost(answers, programOf(used + delta + MAX_NEW_TOKENS - 2));
  const seen: ChatOverflow[] = [];
  const session = new ChatSession(host, {
    system: SYSTEM,
    maxNewTokens: MAX_NEW_TOKENS,
    onOverflow: (context) => {
      seen.push(context);
      return dropOldestTurns(context);
    },
  });

  assertEquals(await session.send(first.content).text(), "Blue.");
  assertEquals(await session.send(second.content).text(), "Red.");

  assertEquals(seen.length, 1, "overflow handling settled in one call");
  assertEquals(seen[0].system, SYSTEM);
  assertEquals(seen[0].needed, seen[0].capacity + 1, "short by exactly 1");
  assertEquals(
    seen[0].turns.length,
    4,
    "the history at the time of the call (system + one exchange + the current message)",
  );
  assertEquals(
    session.turns,
    [SYSTEM_MESSAGE, second, { role: "assistant", content: "Red." }],
    "the oldest user / assistant pair is dropped and the system message stays",
  );
  assertEquals(
    host.created(),
    2,
    "the dropped history does not map onto the KV = the sequence is rebuilt",
  );
  assertEquals(
    host.prompts()[1],
    chatPrompt(tokenizer, [SYSTEM_MESSAGE, second]),
    "the whole rendering of the truncated history",
  );
});

test("ChatSession overflow: fail loudly once nothing is left to drop (no silent shrinking)", async () => {
  // Only system + the current message = nothing the default policy can drop.
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(TIGHT_CAPACITY));
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  const error = await assertRejects(
    () => session.send(LONG_ASK).text(),
    GenerationCapacityError,
    "The conversation does not fit",
  );
  assertEquals(error.constraint, "capacity");
  assertEquals(error.limit, TIGHT_CAPACITY);
  assertEquals(error.requestedNewTokens, MAX_NEW_TOKENS);
  // "The maxNewTokens that would pass now" is negative = by how many tokens it overflows (ADR 0083 addendum 2026-08-31).
  assert(error.maxNewTokens < 0, `the overflow amount cannot be read: ${error.maxNewTokens}`);
  assertEquals(
    session.turns,
    [SYSTEM_MESSAGE],
    "the message of a turn that did not happen does not stay in the history",
  );
  // The overflow decision comes before allocation (the MUST of `#prepare`) — a turn that does not fit takes no KV at all.
  assertEquals(host.created(), 0, "it is not in the shape of allocating and then dropping");
});

describe("ChatSession overflow: policy injection takes effect (returning a history / passing a throw through)", () => {
  it("re-feeds the returned history (the system message can be dropped too)", async () => {
    const asked = { role: "user", content: "Name a color." } as const;
    // A capacity that is short by exactly 1 with the system message and fits once it is dropped.
    const capacity = chatPrompt(tokenizer, [SYSTEM_MESSAGE, asked]).length + MAX_NEW_TOKENS - 2;
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(capacity));
    const session = new ChatSession(host, {
      system: SYSTEM,
      maxNewTokens: MAX_NEW_TOKENS,
      // Whatever happens, keep "only the current message" (drops the system message too = semantics unlike the default).
      onOverflow: ({ turns }) => turns.slice(-1),
    });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(session.turns, [
      { role: "user", content: "Name a color." },
      { role: "assistant", content: "Blue." },
    ]);
    assertEquals(
      host.prompts()[0],
      chatPrompt(tokenizer, [{ role: "user", content: "Name a color." }]),
      "re-rendered with a history without the system message",
    );
  });

  it("a throw reaches the caller as is (not wrapped)", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(TIGHT_CAPACITY));
    const refusal = new Error("the host refused to truncate");
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: () => {
        throw refusal;
      },
    });
    const caught = await session
      .send(LONG_ASK)
      .text()
      .then(
        () => undefined,
        (error: unknown) => error,
      );
    assertEquals(caught, refusal, "the policy's exception arrives with its identity");
  });

  it("a policy that drops the tail (the current message) fails loudly", async () => {
    // The length shrinks, so the length check passes. Letting it through would attach an answer to
    // "a conversation with no question" (the rendering does not constrain the last role, so it
    // settles with zero errors).
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(TIGHT_CAPACITY));
    const session = new ChatSession(host, {
      system: SYSTEM,
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: ({ turns }) => turns.slice(0, -1),
    });
    await assertRejects(
      () => session.send(LONG_ASK).text(),
      Error,
      "dropped the current utterance (the last one)",
    );
  });

  it("a policy that copies the history and returns it passes (the tail is judged by value, not reference)", async () => {
    // The history passed in is a frozen copy, so an implementation that copies into `{ ...turn }`
    // and returns it is legitimate. Judging by reference identity would misjudge this legitimate
    // policy as "dropped the tail".
    const asked = { role: "user", content: "Name a color." } as const;
    const capacity = chatPrompt(tokenizer, [SYSTEM_MESSAGE, asked]).length + MAX_NEW_TOKENS - 2;
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(capacity));
    let returned: { role: ChatMessage["role"]; content: string }[] = [];
    const session = new ChatSession(host, {
      system: SYSTEM,
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: ({ turns }) => {
        assertEquals(Reflect.set(turns[0], "content", "changed"), false);
        returned = turns.slice(-1).map((turn) => ({ ...turn }));
        return returned;
      },
    });
    assertEquals(await session.send(asked.content).text(), "Blue.");
    returned[0].content = "changed after overflow";
    assertEquals(session.turns, [asked, { role: "assistant", content: "Blue." }]);
  });

  it("a history that does not shrink fails loudly instead of looping forever", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(TIGHT_CAPACITY));
    let calls = 0;
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      onOverflow: ({ turns }) => {
        calls += 1;
        return turns;
      },
    });
    await assertRejects(
      () => session.send(LONG_ASK).text(),
      GenerationCapacityError,
      "did not shrink",
    );
    // A result that does not shrink stops after one call (the retry count does not track the number of history entries).
    assertEquals(calls, 1);
  });
});

describe("ChatSession capacity: the program default when omitted; when passed, it is the measure for the sequence and the overflow decision", () => {
  it("when omitted, the distribution's default reaches it as is", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.capacities(), [640], "the capacity passed to the sequence");
  });

  it("the passed capacity reaches the sequence, and the truncation decision uses that value", async () => {
    // The program default is wide enough (640), but the session picks a narrow capacity — deciding
    // with the default never calls overflow handling, deciding with the picked capacity calls it.
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    let overflows = 0;
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      capacity: TIGHT_CAPACITY,
      onOverflow: (context) => {
        overflows += 1;
        return dropOldestTurns(context);
      },
    });
    const error = await assertRejects(
      () => session.send(LONG_ASK).text(),
      GenerationCapacityError,
      "The conversation does not fit",
    );
    assertEquals(error.limit, TIGHT_CAPACITY, "the picked capacity is carried as the limit");
    assertEquals(overflows, 1, "overflow handling is called with the picked capacity");
    // The decision comes before allocation, so a turn that does not fit takes no sequence at all.
    assertEquals(host.created(), 0, "a turn known to overflow allocates no KV");
  });

  it("on a turn that fits, the picked capacity reaches the sequence", async () => {
    // The value the session picked, not the program default (640), is passed to `host.sequence`.
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 128 });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.capacities(), [128], "the same capacity reaches the sequence too");
  });
});

describe("ChatSession speculative: a setting reaches the sequence as is; when unset the field is not passed at all", () => {
  // Speculative decoding is a per-session setting (the borrower context is tied to the lifetime of
  // the sequence, so it cannot be switched per turn). All this layer does is "pass it down or
  // not", and the decision is on the pipeline side — a regression that **does pass**
  // `speculative: undefined` cancels the pipeline default (attach when a drafter is present) while
  // looking the same in type and value, so it can only be seen by whether the field is present.
  const script = [
    { text: "Blue.", closes: true },
    { text: "Red.", closes: true },
  ] as const;

  it("unset: the field itself is not passed (falls to the pipeline default)", async () => {
    const host = fakeHost([...script], programOf(640));
    const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.speculativeKeys(), [false], "a speculative field is passed although unset");
    assertEquals(host.speculatives(), [undefined]);
  });

  it('true / false / "always" reach it as they are', async () => {
    // `"always"` is the setting without the self-tuning gate (step 4-B ④) — this layer interprets none of them and passes them through.
    for (const speculative of [true, false, "always"] as const) {
      const host = fakeHost([...script], programOf(640));
      const session = new ChatSession(host, {
        maxNewTokens: MAX_NEW_TOKENS,
        speculative,
      });
      assertEquals(await session.send("Name a color.").text(), "Blue.");
      assertEquals(host.speculatives(), [speculative]);
      assertEquals(host.speculativeKeys(), [true]);
    }
  });

  it("the same value reaches a rebuilt sequence (it does not switch mid-conversation)", async () => {
    // After a cut-off turn (`closes: false`) the KV cannot be continued, so the sequence is rebuilt —
    // check that the same setting reaches that second one too (a session setting, not a turn setting).
    const host = fakeHost(
      [
        { text: "Blue", closes: false },
        { text: "Red.", closes: true },
      ],
      programOf(640),
    );
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      speculative: false,
    });
    assertEquals(await session.send("Name a color.").text(), "Blue");
    assertEquals(await session.send("Another one.").text(), "Red.");
    assertEquals(host.created(), 2, "the rebuild premise does not hold");
    assertEquals(host.speculatives(), [false, false]);
  });
});

test("ChatSession: onPrefill carries per-chunk progress (information that is not in the text)", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const seen: { readonly chunk: number; readonly chunks: number }[] = [];

  assertEquals(
    await session.send("Name a color.", { onPrefill: (progress) => seen.push(progress) }).text(),
    "Blue.",
  );
  assertEquals(seen, [{ chunk: 1, chunks: 1 }], "the script's prefill event arrives as is");

  // A per-turn setting, so a turn that did not pass it receives nothing (no subscription carried over).
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(seen.length, 1);
});

test("ChatSession: one session = one generation (a concurrent send and a send after dispose throw synchronously)", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  // While a turn that was only issued (not yet drained) runs, a second one is not accepted.
  const first = session.send("Name a color.");
  assertThrows(() => session.send("Another one."), Error, "the previous turn has not finished yet");
  assertEquals(await first.text(), "Blue.");
  // After a turn drained to the end, it is accepted.
  assertEquals(await session.send("Another one.").text(), "Red.");

  await session.dispose();
  assertEquals(host.disposed(), 1, "dispose disposes the sequence");
  assertThrows(() => session.send("hi"), Error, "after dispose");
});

test("ChatSession: a failure raised in this layer rejects done too (it does not claim success)", async () => {
  // When `onPrefill` throws, the inner event sequence is closed with `return()` and resolves with
  // `closed`, so carrying that as is would make callers that read only `done` (progress UI, logs)
  // record the failure as a success.
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const boom = new Error("onPrefill threw");

  const stream = session.send("Name a color.", {
    onPrefill: () => {
      throw boom;
    },
  });
  const caught = await stream.text().then(
    () => undefined,
    (error: unknown) => error,
  );
  assertEquals(caught, boom, "the primary one is the iterable's throw (not wrapped)");
  const settled = await stream.done.then(
    () => "resolved",
    (error: unknown) => error,
  );
  assertEquals(settled, boom, "done rejects with the same exception");

  assertEquals(
    session.turns,
    [],
    "the message of a turn that emitted not a single character does not stay in the history",
  );
  // Not treated as a closed turn = the KV is not continued (the next turn re-feeds the whole rendering).
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "the sequence of a failed turn is not continued");
});

test("ChatSession: the session accepts the next turn even if sequence.dispose fails", async () => {
  // If a failing cleanup left `#busy` set, one exception would make the session unusable forever.
  const boom = new Error("context.dispose propagated a flush failure");
  const host = fakeHost(
    [
      { text: "Blue", closes: false },
      { text: "Red.", closes: true },
    ],
    programOf(640),
    { disposeError: boom },
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const first = session.send("Name a color.");
  const caught = await first.text().then(
    () => undefined,
    (error: unknown) => error,
  );
  assertEquals(caught, boom, "the cleanup failure arrives as is (the body succeeded)");
  assertEquals(await first.done.catch((error: unknown) => error), caught);
  assertEquals(await session.send("Another one.").text(), "Red.", "the session is not wedged");

  // A turn whose body failed too carries both (neither fact is lost).
  const failing = fakeHost([{ text: "Blue.", closes: true }], programOf(640), {
    disposeError: boom,
  });
  const body = new Error("onPrefill threw");
  const session2 = new ChatSession(failing, { maxNewTokens: MAX_NEW_TOKENS });
  const bothStream = session2.send("Name a color.", {
    onPrefill: () => {
      throw body;
    },
  });
  const both = await bothStream.text().then(
    () => undefined,
    (error: unknown) => error,
  );
  assertEquals(await bothStream.done.catch((error: unknown) => error), both);
  assert(both instanceof AggregateError, `not an AggregateError: ${both}`);
  assertEquals(both.errors, [body, boom], "errors[0] is the body, errors[1] the cleanup");
});

test("ChatSession: the accepted set fails synchronously and does not soil the session state", async () => {
  // If the check were inside the body (the async generator), an invalid request would push the
  // message into the history and leave `#busy` set while "not failing until draining starts" =
  // one mix-up makes the session unusable.
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  assertThrows(() => session.send("Name a color.", { maxNewTokens: 0 }), Error, "maxNewTokens 0");
  assertThrows(
    () => session.send("Name a color.", { stopTokens: [2, 2] }),
    Error,
    "Token 2 appears twice",
  );
  assertEquals(session.turns, [], "an invalid request pushes no message into the history");
  assertEquals(host.created(), 0, "nor takes a sequence");
  // `#busy` is not set = a correct request goes through as is.
  assertEquals(await session.send("Name a color.").text(), "Blue.");
});

test("ChatSession: once disposed, even an issued stream does not generate", async () => {
  // `send` checks at issue, but the body starts running after that. Starting to drain after only
  // the session was disposed could take a new capacity's worth of sequence from a live pipeline.
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const stream = session.send("Name a color.");
  await session.dispose();
  await assertRejects(() => stream.text(), Error, "after dispose");
  assertEquals(host.created(), 0, "no sequence is taken after dispose");
});

test("ChatSession: a turn cut by a stop string carries the same reason as chat() and does not continue the KV", async () => {
  const host = fakeHost(
    [
      { text: "Blue.\nUser: hi", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const stream = session.send("Name a color.", { stopStrings: ["\nUser:"] });
  assertEquals(
    await stream.text(),
    "Blue.",
    "neither the stop string itself nor what follows is streamed",
  );
  const stop = await stream.done;
  assertEquals(stop.reason, "stop-string");
  assert(stop.reason === "stop-string" && stop.stopString === "\nUser:");
  assert(stop.tokens > 0, "the inner count is carried as is");
  // One stream can be consumed in only one way (the same contract as `chat()`).
  assertThrows(() => stream[Symbol.asyncIterator](), Error, "can be consumed in only one way");

  assertEquals(
    session.turns,
    [
      { role: "user", content: "Name a color." },
      { role: "assistant", content: "Blue." },
    ],
    "only the streamed text enters the history",
  );
  // The model turn is not closed = no premise for a delta.
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "after a turn cut by a stop string the sequence is rebuilt");
});

describe("ChatSession capacity: the capacity relations are checked at construction (it fails where it was set)", () => {
  // Leaving the check to the sequence side alone would make `#shrink` actually truncate the history
  // on a long turn before the capacity error, and the real cause (a wrong declaration) would show
  // in no diagnostic. The formulas are the same two as in `sequence.ts`.
  // One host is shared by all its: the last it asserts that only its own sequence was created.
  let host: FakeHost;
  beforeAll(() => {
    host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  });

  it("below chunkLength (not even one chunk fits)", () => {
    assertThrows(
      () => new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 8 }),
      ModelInputError,
      "capacity 8 is below chunkLength 32",
    );
  });

  it("above maxPosition (a full-capacity conversation reaches past the position limit)", () => {
    assertThrows(
      () => new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 4097 }),
      ModelInputError,
      "capacity 4097 exceeds maxPosition 4096",
    );
  });

  it("range (a capacity that is not a safe integer)", () => {
    assertThrows(
      () => new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 64.5 }),
      ModelInputError,
      "capacity 64.5 is not a positive integer",
    );
  });

  it("exactly chunkLength / exactly maxPosition passes (the counterpart showing the check is not too wide)", async () => {
    const edge = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 32 });
    new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, capacity: 4096 });
    // Beyond construction passing, check that the picked capacity reaches the sequence as is.
    assertEquals(await edge.send("Name a color.").text(), "Blue.");
    assertEquals(host.capacities(), [32]);
  });
});

test("ChatSession: disposed during the await of the sequence allocation, it disposes that sequence and fails", async () => {
  // `dispose()` runs while `#sequence` is still undefined, so it disposes nothing, and right after
  // `this.#sequence = created` lands — only the re-check after the `await` can close this window.
  let release!: () => void;
  const entered = Promise.withResolvers<void>();
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640), {
    holdSequence: (resume) => {
      release = resume;
      entered.resolve();
    },
  });
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });

  const settled = session
    .send("Name a color.")
    .text()
    .then(
      () => undefined,
      (error: unknown) => error,
    );
  await entered.promise; // entered `#host.sequence()` (the return value is still held).
  await session.dispose();
  assertEquals(
    host.disposed(),
    0,
    "at this point there is nothing to dispose yet (the window is open)",
  );
  release();

  const caught = await settled;
  assert(caught instanceof Error, `it did not end with an exception: ${caught}`);
  assert(
    caught.message.includes("cannot generate after dispose"),
    `it failed for another reason: ${caught.message}`,
  );
  assertEquals(
    host.created(),
    1,
    "the allocation itself went through (not dropped outside the window)",
  );
  assertEquals(host.disposed(), 1, "a sequence that did get allocated is disposed before failing");
});

describe("ChatSession sampler: resolved in the order turn → session → the distribution's default", () => {
  // A regression silently drops the distribution's recommended sampler to temperature 0 (greedy) —
  // no exception, and it can only be noticed by the output becoming "oddly deterministic".
  const recommended: DefaultSampler = { temperature: 1, topK: 64, topP: 0.95 };
  const script = [
    { text: "Blue.", closes: true },
    { text: "Red.", closes: true },
  ] as const;

  it("with both omitted, the distribution's declaration applies", async () => {
    const host = fakeHost([...script], programOf(640), {}, recommended);
    const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.samplers()[0], recommended);
  });

  it("the session setting overrides the distribution's declaration", async () => {
    const host = fakeHost([...script], programOf(640), {}, recommended);
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      sampler: { temperature: 0.5 },
    });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.samplers()[0], { temperature: 0.5 });
  });

  it("the turn setting overrides the session setting", async () => {
    const host = fakeHost([...script], programOf(640), {}, recommended);
    const session = new ChatSession(host, {
      maxNewTokens: MAX_NEW_TOKENS,
      sampler: { temperature: 0.5 },
    });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(
      await session.send("Another one.", { sampler: { temperature: 0 } }).text(),
      "Red.",
    );
    assertEquals(host.samplers(), [{ temperature: 0.5 }, { temperature: 0 }]);
  });

  it("with none of the three levels, the field itself is not passed (the lower layer's default = temperature 0)", async () => {
    const host = fakeHost([...script], programOf(640));
    const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue.");
    assertEquals(host.samplers()[0], undefined);
  });
});

test("ChatSession: an aborted turn resolves done with aborted and does not continue the KV", async () => {
  // A regression that also `fail()`s an abort would turn into "done rejects although I stopped it
  // myself", breaking the abort identification by `error === signal.reason` (ADR 0083 decision 5).
  // Conversely, a regression that sets `stop` on an aborted turn would continue a delta after a
  // model turn that is not closed, quietly breaking the conversation.
  const host = fakeHost(
    [
      { text: "Blue and red.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const controller = new AbortController();
  const reason = new Error("the caller stopped it");

  const stream = session.send("Name a color.", { signal: controller.signal });
  const parts: string[] = [];
  const caught = await (async () => {
    try {
      for await (const chunk of stream) {
        parts.push(chunk);
        controller.abort(reason);
      }
      return undefined;
    } catch (error) {
      return error;
    }
  })();

  assertEquals(caught, reason, "the abort reason arrives with its identity (not wrapped)");
  assert(
    parts.length > 0,
    "aborted before a single chunk came out (this case's premise does not hold)",
  );
  const stop = await stream.done;
  assertEquals(stop.reason, "aborted", "done does not reject; it resolves with aborted");
  assert(
    stop.tokens > 0,
    `the number of tokens drawn before the abort is not carried: ${stop.tokens}`,
  );

  assertEquals(
    session.turns,
    [
      { role: "user", content: "Name a color." },
      { role: "assistant", content: parts.join("") },
    ],
    "only the text that came out is pushed",
  );
  assertEquals(host.disposed(), 1, "the cleanup runs on an aborted turn too");

  // The model turn is not closed = no premise for a delta (the KV is not continued).
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 2, "after an aborted turn the sequence is rebuilt");
});

test("ChatSession: a request is copied at issue (rewriting it before draining has no effect)", async () => {
  // The MUST of the type doc. The receivers' copies (`createSampler` /
  // `GenerationSequence.generate`) are in the body of an async generator = the first `next()`, so
  // leaving it to them would let a sampling spec or stop set rewritten in the window "between issue
  // and the start of draining" silently take effect.
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  const sampler = { temperature: 0.5, logitBias: [[7, 1] as const] };
  const stopTokens = [11, 12];
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS, sampler });

  const stream = session.send("Name a color.", { stopTokens });
  // Rewrite the caller's side while it is issued and not a single chunk has been drained.
  sampler.temperature = 2;
  stopTokens.length = 0;
  assertEquals(await stream.text(), "Blue.");

  const recorded = host.samplers()[0];
  assertEquals(
    recorded,
    { temperature: 0.5, logitBias: [[7, 1]] },
    "it runs with the values at issue",
  );
  assertEquals(
    Object.isFrozen(recorded),
    true,
    "the copy is passed frozen (closes later rewrites)",
  );
  assertEquals(sampler.temperature, 2, "the caller's object is not touched from the copy's side");
});

test("ChatSession: a return before start rolls back only the tentative message and can continue the previous turn's KV", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  await session.send("Name a color.").text();
  const turns = session.turns;
  const unused = session.send("Cancelled.");
  const iterator = unused[Symbol.asyncIterator]();
  await iterator.return?.();
  assertEquals(await unused.done, { reason: "closed", tokens: 0 });
  assertEquals(session.turns, turns);
  assertEquals(host.created(), 1);
  assertEquals(host.disposed(), 0);
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 1);
  await session.dispose();
});

test("ChatSession: a throw before start reports the same exception on done and does not block the next send", async () => {
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const stream = session.send("Cancelled.");
  const error = new Error("consumer failure");
  await stream[Symbol.asyncIterator]()
    .throw?.(error)
    .catch((caught: unknown) => assertEquals(caught, error));
  assertEquals(await stream.done.catch((caught: unknown) => caught), error);
  assertEquals(session.turns, []);
  assertEquals(host.created(), 0);
  assertEquals(await session.send("Name a color.").text(), "Blue.");
  await session.dispose();
});

test("ChatSession: done waits for the cleanup, and the next send can follow right after", async () => {
  const entered = Promise.withResolvers<void>();
  const release = Promise.withResolvers<void>();
  const host = fakeHost(
    [
      { text: "Blue", closes: false },
      { text: "Red.", closes: true },
    ],
    programOf(640),
    {
      holdDispose: () => {
        entered.resolve();
        return release.promise;
      },
    },
  );
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const stream = session.send("Name a color.");
  let settled = false;
  // Fire-and-forget observer: awaiting here would block on the held cleanup this test releases later.
  void stream.done.then(() => {
    settled = true;
  });
  const text = stream.text();
  await entered.promise;
  await Promise.resolve();
  assertEquals(settled, false);
  release.resolve();
  await stream.done;
  const next = session.send("Another one.");
  assertEquals(await text, "Blue");
  assertEquals(await next.text(), "Red.");
  await session.dispose();
});

test("ChatSession: each message of a history snapshot cannot be changed from outside", async () => {
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  const session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  await session.send("Name a color.").text();
  const snapshot = session.turns;
  assertEquals(Reflect.set(snapshot[0], "content", "changed"), false);
  assertEquals(session.turns[0].content, "Name a color.");
  await session.dispose();
});

test("ChatSession token notification: keeps the function given at issue and changes neither the text nor the stop-token count", async () => {
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
  await using session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const seen: number[] = [];
  const options = {
    onToken: (id: number): void => {
      seen.push(id);
    },
  };
  const stream = session.send("Name a color.", options);
  options.onToken = () => {
    throw new Error("changed after issue");
  };
  assertEquals(await stream.text(), "Blue.");
  assertEquals(seen, tokenizer.encode("Blue."));
  assertEquals((await stream.done).tokens, seen.length + 1);
});
test("ChatSession token notification: reports a callback failure and does not block the next conversation", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
  );
  await using session = new ChatSession(host, { maxNewTokens: MAX_NEW_TOKENS });
  const failure = new Error("token observer failed");
  const stream = session.send("Name a color.", {
    onToken: () => {
      throw failure;
    },
  });
  assertEquals(await assertRejects(() => stream.text()), failure);
  assertEquals(await assertRejects(() => stream.done), failure);
  assertEquals(await session.send("Another color.").text(), "Red.");
});

test("ChatSession overflow: a message longer than the argument-spread limit also returns a capacity error", async () => {
  // One fragment of 150,000 characters (text that added tokens do not split). Pushing the BPE result
  // as variadic arguments would make the tokenizer fail with a bare RangeError before the capacity
  // decision, which looks like an internal fault to the caller.
  const host = fakeHost([{ text: "Blue.", closes: true }], programOf(TIGHT_CAPACITY));
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });
  const error = await assertRejects(
    () => session.send("\u0001".repeat(150_000)).text(),
    GenerationCapacityError,
    "The conversation does not fit",
  );
  // The length exceeds both the position table (4096) and the capacity, so which limit it fails on does not matter.
  assert(error.requestedNewTokens === MAX_NEW_TOKENS);
  assertEquals(host.created(), 0);
});

// ---- ⑪ branching (`fork` — ADR 0083 addendum 2026-10-11, decision 8) -------------------------
//
// A branch is "an independent session with the same settings and the same history"; a
// conversation that continues its KV is copied sequence and all, and a conversation with an empty
// KV copies only the history (no GPU copy). The script is one for the whole host, so whether one
// sends on the branch or on the original, the next answer of the script comes back.

test("ChatSession.fork: a branch inherits the history and the KV, and from then on proceeds independently of the original", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
      { text: "Green.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });
  assertEquals(await session.send("Name a color.").text(), "Blue.");

  const branch = await session.fork();
  assertEquals(host.forks(), 1, "a conversation that continues its KV is copied sequence and all");
  assertEquals(branch.turns, session.turns);

  // A turn on the branch continues the KV and feeds only the delta (it does not rebuild the sequence).
  assertEquals(await branch.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 1, "the branch rebuilt the sequence (it did not inherit the KV)");
  assertEquals(
    host.prompts().at(-1),
    chatTurn(tokenizer, { role: "user", content: "Another one." }),
  );
  // A turn on the original continues its own KV too. The histories do not mix.
  assertEquals(await session.send("A third.").text(), "Green.");
  assertEquals(host.created(), 1);
  assertEquals(host.prompts().at(-1), chatTurn(tokenizer, { role: "user", content: "A third." }));
  const base: readonly ChatMessage[] = [
    SYSTEM_MESSAGE,
    { role: "user", content: "Name a color." },
    { role: "assistant", content: "Blue." },
  ];
  assertEquals(branch.turns, [
    ...base,
    { role: "user", content: "Another one." },
    { role: "assistant", content: "Red." },
  ]);
  assertEquals(session.turns, [
    ...base,
    { role: "user", content: "A third." },
    { role: "assistant", content: "Green." },
  ]);

  // The original may be disposed first (no parent-child dependency). Each disposes its own sequence.
  await session.dispose();
  assertEquals(host.disposed(), 1);
  await branch.dispose();
  assertEquals(host.disposed(), 2);
});

describe("ChatSession.fork: a conversation with an empty KV makes no GPU copy and copies only the history and the settings", () => {
  it("a conversation that has not sent yet", async () => {
    const host = fakeHost([{ text: "Blue.", closes: true }], programOf(640));
    const session = new ChatSession(host, {
      system: SYSTEM,
      maxNewTokens: MAX_NEW_TOKENS,
      capacity: 320,
      speculative: "always",
    });
    const branch = await session.fork();
    assertEquals(host.forks(), 0, "it made a copy although the KV is empty");
    assertEquals(branch.turns, [SYSTEM_MESSAGE]);
    // The branch's first send re-renders everything and builds a sequence with the same settings as the original (capacity, speculative).
    assertEquals(await branch.send("Name a color.").text(), "Blue.");
    assertEquals(host.capacities(), [320]);
    assertEquals(host.speculatives(), ["always"]);
    assertEquals(
      host.prompts()[0],
      chatPrompt(tokenizer, [SYSTEM_MESSAGE, { role: "user", content: "Name a color." }]),
    );
    assertEquals(
      session.turns,
      [SYSTEM_MESSAGE],
      "the branch's send entered the original's history",
    );
    await branch.dispose();
    await session.dispose();
  });

  it("a conversation whose last turn was not closed by EOS (the KV has been dropped)", async () => {
    const host = fakeHost(
      [
        { text: "Blue", closes: false },
        { text: "Red.", closes: true },
      ],
      programOf(640),
    );
    const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });
    assertEquals(await session.send("Name a color.").text(), "Blue");
    const branch = await session.fork();
    assertEquals(host.forks(), 0, "it made a copy of a conversation whose KV was dropped");
    assertEquals(branch.turns, session.turns);
    // The branch's next turn re-renders from the history, like the original.
    assertEquals(await branch.send("Another one.").text(), "Red.");
    assertEquals(
      host.prompts().at(-1),
      chatPrompt(tokenizer, [
        SYSTEM_MESSAGE,
        { role: "user", content: "Name a color." },
        { role: "assistant", content: "Blue" },
        { role: "user", content: "Another one." },
      ]),
    );
    await branch.dispose();
    await session.dispose();
  });
});

test("ChatSession.fork: a running turn and a disposed session fail synchronously, and a send right after does not enter the branch", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
      { text: "Green.", closes: true },
    ],
    programOf(640),
  );
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });

  // While a turn is running (the stream has not ended) it fails synchronously.
  const first = session.send("Name a color.");
  assertThrows(() => session.fork(), Error, "the previous turn has not finished yet");
  assertEquals(await first.text(), "Blue.");
  assertEquals(host.forks(), 0, "a rejected fork made a copy");

  // Even a send on the original right after fork() (before the await) leaves the branch's history as it was at the call.
  const forking = session.fork();
  assertEquals(await session.send("Another one.").text(), "Red.");
  const branch = await forking;
  assertEquals(branch.turns, [
    SYSTEM_MESSAGE,
    { role: "user", content: "Name a color." },
    { role: "assistant", content: "Blue." },
  ]);
  // The branch continues from the KV at the time of the call (continued as a delta).
  assertEquals(await branch.send("A third.").text(), "Green.");
  assertEquals(host.prompts().at(-1), chatTurn(tokenizer, { role: "user", content: "A third." }));

  await session.dispose();
  assertThrows(() => session.fork(), Error, "gemma.ChatSession: cannot fork after dispose");
  await branch.dispose();
});

test("ChatSession.fork: when the KV copy fails it rejects, and the original can go on as it is", async () => {
  const host = fakeHost(
    [
      { text: "Blue.", closes: true },
      { text: "Red.", closes: true },
    ],
    programOf(640),
    { forkError: new Error("injected: the KV copy failed") },
  );
  const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: MAX_NEW_TOKENS });
  assertEquals(await session.send("Name a color.").text(), "Blue.");
  await assertRejects(() => session.fork(), Error, "injected: the KV copy failed");
  // The original's KV and history are intact (it can continue with a delta).
  assertEquals(await session.send("Another one.").text(), "Red.");
  assertEquals(host.created(), 1);
  assertEquals(
    host.prompts().at(-1),
    chatTurn(tokenizer, { role: "user", content: "Another one." }),
  );
  await session.dispose();
});

// ---- tools, tool calls and thinking (ADR 0084 addendum 2026-11-04) ---------------------------
//
// The identity the doc of `ChatSession` states — the KV of step k (P₀ ⧺ G₀ ⧺ s₀ ⧺ D₁ ⧺ … ⧺ Dₖ)
// equals `chatPrompt(hₖ)` — checked over a multi-step tool conversation, in both dialects, with
// the documented divergences pinned as such (a check that only says "different" would pass for any
// bug, so each one names the exact difference).

const TOOL_RESPONSE = tokenizer.addedTokenId("<|tool_response>") as number;
const Q = '<|"|>';

const WEATHER: ChatTool = {
  type: "function",
  function: {
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: {
      type: "object",
      properties: { location: { type: "string", description: "The city name" } },
      required: ["location"],
    },
  },
};
const TIME: ChatTool = {
  type: "function",
  function: {
    name: "get_time",
    description: "Current local time.",
    parameters: { type: "object", properties: { timezone: { type: "string" } } },
  },
};
const TOOLS = [WEATHER, TIME];
const ASK = "What is the weather in Paris?";
const CALL_WEATHER = `<|tool_call>call:get_weather{location:${Q}Paris${Q}}<tool_call|>`;
const CALL_TIME = `<|tool_call>call:get_time{timezone:${Q}Europe/Paris${Q}}<tool_call|>`;

/** A step of a scripted conversation: what the host sends, and the KV right before generating. */
type Step = {
  readonly send: (session: ChatSession) => ChatSessionStream;
  /** The messages the step adds before generating (to rebuild hₖ from the turns). */
  readonly pending: readonly ChatMessage[];
};

/**
 * Runs `steps` and returns, per step, the KV (built from what the host received and generated)
 * next to `hₖ` (the session's turns plus the step's pending messages, before generating).
 */
const runSteps = async (
  session: ChatSession,
  host: FakeHost,
  answers: readonly Answer[],
  steps: readonly Step[],
  owner: Tokenizer,
): Promise<
  { readonly kv: number[]; readonly history: ChatMessage[]; readonly reply: ChatReply }[]
> => {
  const out: { kv: number[]; history: ChatMessage[]; reply: ChatReply }[] = [];
  let kv: number[] = [];
  for (const [index, step] of steps.entries()) {
    const history = [...session.turns, ...step.pending];
    const stream = step.send(session);
    await stream.text();
    const reply = await stream.reply;
    kv = [...kv, ...host.prompts()[index]];
    out.push({ kv: [...kv], history, reply });
    const answer = answers[index];
    kv = [
      ...kv,
      ...owner.encode(answer.text),
      ...(answer.closes ? [answer.stop ?? END_OF_TURN] : []),
    ];
  }
  return out;
};

const toolAnswers: readonly Answer[] = [
  { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
  { text: CALL_TIME, closes: true, stop: TOOL_RESPONSE },
  { text: "It is 18 degrees at 14:05.", closes: true },
  { text: "You are welcome.", closes: true },
];
const toolSteps: readonly Step[] = [
  { send: (session) => session.send(ASK), pending: [{ role: "user", content: ASK }] },
  {
    send: (session) =>
      session.sendToolResults([{ role: "tool", tool_call_id: "call_1", content: "18 degrees" }]),
    pending: [{ role: "tool", tool_call_id: "call_1", content: "18 degrees" }],
  },
  {
    send: (session) =>
      session.sendToolResults([{ role: "tool", tool_call_id: "call_2", content: "14:05" }]),
    pending: [{ role: "tool", tool_call_id: "call_2", content: "14:05" }],
  },
  { send: (session) => session.send("Thanks!"), pending: [{ role: "user", content: "Thanks!" }] },
];

describe("ChatSession tools: a multi-step tool conversation carried in the KV", () => {
  it("E2B (gemma4): the KV equals the whole re-render at every step, token for token", async () => {
    const host = fakeHost(toolAnswers, programOf(4096));
    const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: 32, tools: TOOLS });
    const options = { tools: TOOLS };

    const steps = await runSteps(session, host, toolAnswers, toolSteps, tokenizer);

    assertEquals(host.created(), 1, "the KV was carried through calls, results and a new question");
    for (const [index, step] of steps.entries()) {
      assertEquals(step.kv, chatPrompt(tokenizer, step.history, options), `step ${index}`);
    }
    assertEquals(
      steps.map((step) => step.reply.end),
      ["tool-call", "tool-call", "turn", "turn"],
    );
    assertEquals(
      host.prompts()[1],
      chatToolResults(
        tokenizer,
        steps[0].reply.end === "tool-call" ? steps[0].reply.message.tool_calls : [],
        [{ role: "tool", tool_call_id: "call_1", content: "18 degrees" }],
        options,
      ),
      "a result is a delta",
    );
    assertEquals(session.turns, [
      SYSTEM_MESSAGE,
      { role: "user", content: ASK },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            function: { name: "get_weather", arguments: { location: "Paris" } },
            id: "call_1",
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "18 degrees" },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            function: { name: "get_time", arguments: { timezone: "Europe/Paris" } },
            id: "call_2",
          },
        ],
      },
      { role: "tool", tool_call_id: "call_2", content: "14:05" },
      { role: "assistant", content: "It is 18 degrees at 14:05." },
      { role: "user", content: "Thanks!" },
      { role: "assistant", content: "You are welcome." },
    ]);
  });

  it("12B (gemma4-unified): identical but for the empty thought of each model turn a prompt opened", async () => {
    const unified = new Tokenizer(assets, "gemma4-unified");
    const host = fakeHost(toolAnswers, programOf(4096), {}, undefined, unified);
    const session = new ChatSession(host, { system: SYSTEM, maxNewTokens: 32, tools: TOOLS });
    const options = { tools: TOOLS };
    const opened = "<|turn>model\n<|channel>thought\n<channel|>";
    const plain = (text: string): string => text.replaceAll(opened, "<|turn>model\n");

    const steps = await runSteps(session, host, toolAnswers, toolSteps, unified);

    for (const [index, step] of steps.entries()) {
      const carried = unified.decode(step.kv, { skipSpecialTokens: false });
      const whole = renderGemma4Chat(step.history, "gemma4-unified", options);
      assertEquals(
        plain(carried),
        plain(whole),
        `step ${index}: equal once the channel is removed`,
      );
    }
    // Step 1 continues the first model turn: the KV opened it with the empty channel, the re-render
    // writes the call right after `<|turn>model\n` (divergence 1 of the doc).
    const step1 = unified.decode(steps[1].kv, { skipSpecialTokens: false });
    assert(step1.includes(`${opened}${CALL_WEATHER}`), "the KV keeps the channel before the call");
    assert(
      renderGemma4Chat(steps[1].history, "gemma4-unified", options).includes(
        `<|turn>model\n${CALL_WEATHER}`,
      ),
      "the re-render has none",
    );
    // After a result the prompt continues the turn — no tail in either dialect.
    assert(!unified.decode(host.prompts()[1], { skipSpecialTokens: false }).includes("<|channel>"));
  });

  it("thinking on: exact within one user turn; a new question drops the past thoughts on re-render", async () => {
    const answers: readonly Answer[] = [
      {
        text: `<|channel>thought\nI need the weather.\n<channel|>${CALL_WEATHER}`,
        closes: true,
        stop: TOOL_RESPONSE,
      },
      // The prompt after a result reopened the channel; the model closes it.
      { text: "It is 18 degrees.\n<channel|>Sunny, 18 degrees.", closes: true },
      { text: "<|channel>thought\nPolite.\n<channel|>You are welcome.", closes: true },
    ];
    const steps: readonly Step[] = [
      toolSteps[0],
      toolSteps[1],
      {
        send: (session) => session.send("Thanks!"),
        pending: [{ role: "user", content: "Thanks!" }],
      },
    ];
    const host = fakeHost(answers, programOf(4096));
    const session = new ChatSession(host, {
      maxNewTokens: 32,
      tools: [WEATHER],
      enableThinking: true,
    });
    const options = { tools: [WEATHER], enableThinking: true };

    const run = await runSteps(session, host, answers, steps, tokenizer);

    assertEquals(run[0].kv, chatPrompt(tokenizer, run[0].history, options), "step 0");
    assertEquals(
      run[1].kv,
      chatPrompt(tokenizer, run[1].history, options),
      "step 1 (thought kept)",
    );
    assertEquals(
      tokenizer
        .decode(host.prompts()[1], { skipSpecialTokens: false })
        .endsWith("<tool_response|><|channel>thought\n"),
      true,
      "the results delta reopens the thought channel",
    );
    assertEquals(run[1].reply.end, "turn");
    assertEquals(run[1].reply.end === "turn" ? run[1].reply.message : undefined, {
      role: "assistant",
      thinking: "It is 18 degrees.",
      content: "Sunny, 18 degrees.",
    });
    // Divergence 2 of the doc: the thoughts before the new question stay in the KV only.
    const carried = tokenizer.decode(run[2].kv, { skipSpecialTokens: false });
    const whole = renderGemma4Chat(run[2].history, "gemma4", options);
    assertNotEquals(carried, whole);
    assertEquals(
      carried
        .replace("<|channel>thought\nI need the weather.\n<channel|>", "")
        .replace("<|channel>thought\nIt is 18 degrees.\n<channel|>", ""),
      whole,
      "the only difference is the two past thoughts",
    );
  });
});

describe("ChatSession tools: the order of turns and results is enforced", () => {
  const callFirst = (): { host: FakeHost; session: ChatSession } => {
    const host = fakeHost(
      [
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
        { text: "Sunny.", closes: true },
      ],
      programOf(4096),
    );
    return { host, session: new ChatSession(host, { maxNewTokens: 32, tools: TOOLS }) };
  };

  it("after a tool call, send throws until the results are sent", async () => {
    const { session } = callFirst();
    const first = session.send(ASK);
    await first.text();
    assertEquals((await first.reply).end, "tool-call");

    assertThrows(() => session.send("Never mind."), Error, "sendToolResults first");
    const answered = session.sendToolResults([
      { role: "tool", name: "get_weather", content: "18 degrees" },
    ]);
    assertEquals(await answered.text(), "Sunny.");
    assertEquals((await answered.reply).end, "turn");
  });

  it("sendToolResults without a waiting call throws synchronously", () => {
    const { session } = callFirst();
    assertThrows(
      () => session.sendToolResults([{ role: "tool", name: "get_weather", content: "x" }]),
      Error,
      "no tool call is waiting",
    );
  });

  it("an invalid result fails synchronously and leaves the session as it was", async () => {
    const { session } = callFirst();
    await session.send(ASK).text();
    const before = session.turns;

    assertThrows(
      () => session.sendToolResults([{ role: "tool", tool_call_id: "call_9", content: "x" }]),
      ModelInputError,
      "matches 0 of the calls",
    );
    assertThrows(
      () =>
        session.sendToolResults([
          { role: "tool", tool_call_id: "call_1", content: 18 as unknown as string },
        ]),
      ModelInputError,
      "not a string",
    );
    assertThrows(() => session.sendToolResults([]), ModelInputError, "no results");
    assertEquals(session.turns, before, "nothing entered the history");
    // Still usable: the right result goes through.
    const answered = session.sendToolResults([
      { role: "tool", tool_call_id: "call_1", content: "18 degrees" },
    ]);
    assertEquals(await answered.text(), "Sunny.");
  });

  it("the history handed out is deep-frozen (a call's arguments cannot be rewritten)", async () => {
    const { session } = callFirst();
    await session.send(ASK).text();
    const turns = session.turns;
    const call = turns.at(-1);
    assert(call?.role === "assistant" && call.tool_calls !== undefined);
    assertEquals(
      Reflect.set(call.tool_calls[0].function.arguments ?? {}, "location", "Rome"),
      false,
    );
    const again = session.turns.at(-1);
    assert(again?.role === "assistant");
    assertEquals(again.tool_calls?.[0].function.arguments, { location: "Paris" });
  });

  it("a branch forked while a call waits can answer it on its own", async () => {
    const host = fakeHost(
      [
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
        { text: "Sunny.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    await session.send(ASK).text();

    const branch = await session.fork();
    assertEquals(host.forks(), 1, "the KV was copied");
    assertEquals(
      await branch
        .sendToolResults([{ role: "tool", tool_call_id: "call_1", content: "18" }])
        .text(),
      "Sunny.",
    );
    assertThrows(() => session.send("x"), Error, "sendToolResults first", "the base still waits");
  });

  it("bad declarations fail at construction", () => {
    const host = fakeHost([], programOf(4096));
    assertThrows(
      () =>
        new ChatSession(host, {
          maxNewTokens: 8,
          tools: [{ type: "function", function: { name: "get-weather" } }],
        }),
      ModelInputError,
      "not a word",
    );
    assertThrows(
      () => new ChatSession(host, { maxNewTokens: 8, enableThinking: "yes" as unknown as boolean }),
      ModelInputError,
      "not a boolean",
    );
  });
});

describe("ChatSession tools: what the model writes is parsed, and a malformed turn is not trusted", () => {
  it("a tools session streams the special spellings; the reply carries the parsed call", async () => {
    const host = fakeHost(
      [{ text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE }],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);

    assertEquals(await stream.text(), CALL_WEATHER);
    const reply = await stream.reply;
    assertEquals(reply.end, "tool-call");
    assertEquals((await stream.done).reason, "eos");
  });

  it("a plain session streams without them, and still parses the turn from its ids", async () => {
    const host = fakeHost(
      [{ text: "<|channel>thought\nX<channel|>Blue.", closes: true }],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32 });
    const stream = session.send("Name a color.");

    assertEquals(await stream.text(), "thought\nXBlue.", "the stream drops special tokens");
    assertEquals(await stream.reply, {
      end: "turn",
      message: { role: "assistant", thinking: "X", content: "Blue." },
      stop: await stream.done,
    });
    assertEquals(session.turns.at(-1), { role: "assistant", content: "Blue.", reasoning: "X" });
  });

  it("a broken call rejects the reply (done still resolves), keeps the raw text and drops the KV", async () => {
    const broken = "<|tool_call>call:get_weather{location:Paris}<tool_call|>";
    const host = fakeHost(
      [
        { text: broken, closes: true, stop: TOOL_RESPONSE },
        { text: "Sorry.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    await stream.text();

    assertEquals((await stream.done).reason, "eos");
    await assertRejects(() => stream.reply, ModelInputError, "not JSON");
    assertEquals(session.turns.at(-1), { role: "assistant", content: broken });
    await session.send("Try again.").text();
    assertEquals(host.created(), 2, "the next turn re-renders the history");
  });

  it("tool calls closed by <turn|> are not a tool-call turn", async () => {
    const host = fakeHost([{ text: CALL_WEATHER, closes: true }], programOf(4096));
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    await stream.text();

    await assertRejects(() => stream.reply, ModelInputError, "closed its turn");
    assertThrows(() => session.sendToolResults([]), Error, "no tool call is waiting");
  });

  it("a tool call in a session without tools is not waited on (the reply rejects)", async () => {
    const host = fakeHost(
      [
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
        { text: "Blue.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32 });
    const stream = session.send(ASK);
    await stream.text();

    await assertRejects(() => stream.reply, ModelInputError, "declares no tools");
    assertEquals(await session.send("Name a color.").text(), "Blue.", "the conversation goes on");
  });

  it("a cut-off turn is not parsed and reports cut-off", async () => {
    const host = fakeHost([{ text: CALL_WEATHER.slice(0, 20), closes: false }], programOf(4096));
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    await stream.text();

    const reply = await stream.reply;
    assertEquals(reply.end, "cut-off");
    assertEquals(reply.stop.reason, "max-tokens");
  });
});

describe("dropOldestTurns: drops the oldest whole exchange", () => {
  const exchange: ChatMessage[] = [
    { role: "user", content: ASK },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        { id: "call_1", type: "function", function: { name: "get_weather", arguments: {} } },
      ],
    },
    { role: "tool", tool_call_id: "call_1", content: "18" },
    { role: "assistant", content: "Sunny." },
  ];

  it("a tool exchange goes as a whole (no result is left without its call)", async () => {
    const next = await dropOldestTurns({
      turns: [SYSTEM_MESSAGE, ...exchange, { role: "user", content: "Thanks!" }],
      system: SYSTEM,
      capacity: 1,
      needed: 2,
    });
    assertEquals(next, [SYSTEM_MESSAGE, { role: "user", content: "Thanks!" }]);
  });

  it("the exchange being answered (results included) is never dropped", async () => {
    const turns = [SYSTEM_MESSAGE, ...exchange.slice(0, 3)];
    assertEquals(await dropOldestTurns({ turns, system: SYSTEM, capacity: 1, needed: 2 }), turns);
  });

  it("an overflow while sending results drops the older exchange and re-renders the rest", async () => {
    const maxNewTokens = 64;
    const answers: readonly Answer[] = [
      { text: "Blue.", closes: true },
      { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
      { text: "Sunny.", closes: true },
    ];
    const result: ChatToolMessage = { role: "tool", tool_call_id: "call_1", content: "18 degrees" };
    const kept: ChatMessage[] = [
      { role: "user", content: ASK },
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            type: "function",
            function: { name: "get_weather", arguments: { location: "Paris" } },
            id: "call_1",
          },
        ],
      },
      result,
    ];
    const shrunk = chatPrompt(tokenizer, kept, { tools: TOOLS }).length;
    const capacity = shrunk + maxNewTokens - 1;
    const host = fakeHost(answers, programOf(capacity));
    const session = new ChatSession(host, { maxNewTokens: 16, tools: TOOLS });
    await session.send(LONG_ASK).text();
    await session.send(ASK).text();
    // Precondition: carried in the KV, the results would not fit (so the overflow path is taken).
    const carried =
      chatPrompt(tokenizer, [...session.turns, result], { tools: TOOLS }).length + maxNewTokens - 1;
    assert(carried > capacity, `the results fit without overflow (${carried} ≤ ${capacity})`);

    const answered = session.sendToolResults([result], { maxNewTokens });
    assertEquals(await answered.text(), "Sunny.");

    assertEquals(host.created(), 2, "the shrunk history is re-rendered on a new sequence");
    assertEquals(host.prompts()[2], chatPrompt(tokenizer, kept, { tools: TOOLS }));
  });
});

test("ChatSession tools: content written before a call is its own entry, so the re-render keeps its place", async () => {
  const answers: readonly Answer[] = [
    { text: `Let me check.${CALL_WEATHER}`, closes: true, stop: TOOL_RESPONSE },
    { text: "Sunny.", closes: true },
  ];
  const host = fakeHost(answers, programOf(4096));
  const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });

  const run = await runSteps(session, host, answers, toolSteps.slice(0, 2), tokenizer);

  assertEquals(session.turns.slice(1, 3), [
    { role: "assistant", content: "Let me check." },
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          type: "function",
          function: { name: "get_weather", arguments: { location: "Paris" } },
          id: "call_1",
        },
      ],
    },
  ]);
  assertEquals(run[1].kv, chatPrompt(tokenizer, run[1].history, { tools: TOOLS }));
  // One message carrying both would put the content after the results (the template's order).
  const merged: ChatMessage[] = [
    { role: "user", content: ASK },
    {
      role: "assistant",
      content: "Let me check.",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "get_weather", arguments: { location: "Paris" } },
        },
      ],
    },
    ...toolSteps[1].pending,
    { role: "user", content: "x" },
  ];
  assert(
    renderGemma4Chat(merged, "gemma4", { tools: TOOLS }).includes(
      `<tool_response|>Let me check.<turn|>`,
    ),
  );
});

// ---- the review of 2026-11-04 (xhigh, executed scenarios) -------------------------------------

describe("ChatSession review fixes: each scenario failed before its fix", () => {
  const callTwice = `${CALL_WEATHER}${CALL_TIME}`;

  it("the reply's calls are a frozen copy: writing into them leaves the history alone", async () => {
    const host = fakeHost(
      [{ text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE }],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    await stream.text();
    const reply = await stream.reply;
    assert(reply.end === "tool-call");

    const args = reply.message.tool_calls[0].function.arguments;
    assertEquals(Reflect.set(args, "location", "Rome"), false, "the reply is frozen");
    assertThrows(() => {
      (args as Record<string, unknown>).location = "Rome";
    }, TypeError);
    const last = session.turns.at(-1);
    assert(last?.role === "assistant");
    assertEquals(last.tool_calls?.[0].function.arguments, { location: "Paris" });
  });

  for (const [label, answer] of [
    ["cut off (max tokens)", { text: "Got the weather.\n<channel|>Sun", closes: false }],
    [
      "malformed (a broken call)",
      {
        text: "Got it.\n<channel|><|tool_call>call:get_time{tz:Paris}<tool_call|>",
        closes: true,
        stop: TOOL_RESPONSE,
      },
    ],
  ] as const) {
    it(`thinking on: a continuation after results that is ${label} keeps the prompt's thought opener`, async () => {
      const host = fakeHost(
        [
          { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
          answer,
          { text: "Ok.", closes: true },
        ],
        programOf(4096),
      );
      const session = new ChatSession(host, {
        maxNewTokens: 32,
        tools: TOOLS,
        enableThinking: true,
      });
      await session.send(ASK).text();
      const continued = session.sendToolResults([
        { role: "tool", tool_call_id: "call_1", content: "18 degrees" },
      ]);
      await continued.text();
      await continued.reply.catch(() => undefined);

      const last = session.turns.at(-1);
      assert(last?.role === "assistant");
      assert(last.content.startsWith("<|channel>thought\n"), JSON.stringify(last.content));
      // The re-render strips the thought, so the model does not read it back as text.
      await session.send("And tomorrow?").text();
      const rerendered = tokenizer.decode(host.prompts().at(-1) ?? [], {
        skipSpecialTokens: false,
      });
      assert(!rerendered.includes("Got"), rerendered);
    });
  }

  it("call ids are unique per session (and a fork continues the count)", async () => {
    const host = fakeHost(
      [
        { text: callTwice, closes: true, stop: TOOL_RESPONSE },
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 64, tools: TOOLS });
    const first = session.send(ASK);
    await first.text();
    const ids = (reply: ChatReply): string[] =>
      reply.end === "tool-call" ? reply.message.tool_calls.map((call) => call.id ?? "") : [];
    assertEquals(ids(await first.reply), ["call_1", "call_2"]);

    const branch = await session.fork();
    const second = session.sendToolResults([
      { role: "tool", tool_call_id: "call_1", content: "18" },
      { role: "tool", tool_call_id: "call_2", content: "14:05" },
    ]);
    await second.text();
    assertEquals(ids(await second.reply), ["call_3"]);
    const forked = branch.sendToolResults([
      { role: "tool", tool_call_id: "call_1", content: "18" },
      { role: "tool", tool_call_id: "call_2", content: "14:05" },
    ]);
    await forked.text();
    assertEquals(ids(await forked.reply), ["call_3"], "the branch counts from the same point");
  });

  it("results must answer every call of the turn exactly once", async () => {
    const host = fakeHost(
      [{ text: callTwice, closes: true, stop: TOOL_RESPONSE }],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 64, tools: TOOLS });
    await session.send(ASK).text();
    const before = session.turns;

    assertThrows(
      () => session.sendToolResults([{ role: "tool", tool_call_id: "call_1", content: "18" }]),
      ModelInputError,
      "unanswered: call_2 (get_time)",
    );
    assertThrows(
      () =>
        session.sendToolResults([
          { role: "tool", tool_call_id: "call_1", content: "18" },
          { role: "tool", tool_call_id: "call_1", content: "again" },
          { role: "tool", tool_call_id: "call_2", content: "14:05" },
        ]),
      ModelInputError,
      "answered more than once: call_1 (get_weather)",
    );
    assertEquals(session.turns, before, "nothing entered the history");
  });

  it("a getter that calls send while the options are read issues one turn, not two", async () => {
    const host = fakeHost(
      [
        { text: "Blue.", closes: true },
        { text: "Red.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 16 });
    let inner: ChatSessionStream | undefined;
    const options = {
      get maxNewTokens(): number {
        inner ??= session.send("inner");
        return 16;
      },
    };

    assertThrows(() => session.send("outer", options), Error, "has not finished yet");
    assert(inner !== undefined);
    assertEquals(await inner.text(), "Blue.");
    assertEquals(
      session.turns.map((turn) => turn.content),
      ["inner", "Blue."],
      "only the inner turn entered the history",
    );
  });

  it("a getter cannot change a result between its check and its record", async () => {
    const host = fakeHost(
      [
        { text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE },
        { text: "Sunny.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    await session.send(ASK).text();
    let reads = 0;
    const result = {
      role: "tool" as const,
      tool_call_id: "call_1",
      get content(): string {
        reads += 1;
        return reads === 1 ? "18 degrees" : (42 as unknown as string);
      },
    };

    await session.sendToolResults([result]).text();
    assertEquals(reads, 1, "the result was read once (copied, then checked)");
    assertEquals(session.turns.at(-2), {
      role: "tool",
      tool_call_id: "call_1",
      content: "18 degrees",
    });
  });

  it("with tools, <|tool_response> without a call is a malformed turn (the KV is rebuilt)", async () => {
    const host = fakeHost(
      [
        { text: "Let me think", closes: true, stop: TOOL_RESPONSE },
        { text: "Sunny.", closes: true },
      ],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    await stream.text();

    await assertRejects(() => stream.reply, ModelInputError, "wrote no call");
    await session.send("Well?").text();
    assertEquals(host.created(), 2);
  });

  it("dispose during a tool-call turn rejects the reply (a torn-down call is not delivered)", async () => {
    const host = fakeHost(
      [{ text: CALL_WEATHER, closes: true, stop: TOOL_RESPONSE }],
      programOf(4096),
    );
    const session = new ChatSession(host, { maxNewTokens: 32, tools: TOOLS });
    const stream = session.send(ASK);
    const iterator = stream[Symbol.asyncIterator]();
    await iterator.next(); // the turn is running
    const disposing = session.dispose();
    for (let step = await iterator.next(); step.done !== true; step = await iterator.next()) {
      // drain
    }
    await disposing;

    assertEquals((await stream.done).reason, "eos");
    await assertRejects(() => stream.reply, Error, "disposed during the turn");
  });
});

describe("ChatSession without maxNewTokens: the reserve drives the overflow check, not the reply", () => {
  const FIRST = { role: "user", content: "Name a color." } as const;
  const SECOND = { role: "user", content: "Another one." } as const;
  const BLUE: Answer = { text: "Blue.", closes: true };
  const RED: Answer = { text: "Red.", closes: true };
  const RESERVE = 16;
  /** A reply far longer than any room these checks leave. */
  const LONG_ANSWER = Array.from({ length: 8 }, () => LONG_ASK).join(" ");

  /** The capacity at which the second turn misses by exactly `missing` with the reserve. */
  const capacityFor = (missing: number): number =>
    chatPrompt(tokenizer, [SYSTEM_MESSAGE, FIRST]).length +
    answerCost(BLUE) +
    chatTurn(tokenizer, SECOND).length +
    RESERVE -
    1 -
    missing;

  it("a turn that misses by one drops the oldest exchange", async () => {
    const host = fakeHost([BLUE, RED], programOf(capacityFor(1)));
    const seen: ChatOverflow[] = [];
    const session = new ChatSession(host, {
      system: SYSTEM,
      reserveNewTokens: RESERVE,
      onOverflow: (context) => {
        seen.push(context);
        return dropOldestTurns(context);
      },
    });
    await session.send(FIRST.content).text();
    await session.send(SECOND.content).text();

    assertEquals(seen.length, 1);
    assertEquals(seen[0].needed, seen[0].capacity + 1, "prompt + reserve − 1, short by exactly 1");
    assertEquals(session.turns, [SYSTEM_MESSAGE, SECOND, { role: "assistant", content: "Red." }]);
    assertEquals(host.created(), 2);
  });

  it("a turn that just fits keeps the history and the KV", async () => {
    const host = fakeHost([BLUE, RED], programOf(capacityFor(0)));
    const session = new ChatSession(host, { system: SYSTEM, reserveNewTokens: RESERVE });
    await session.send(FIRST.content).text();
    await session.send(SECOND.content).text();

    assertEquals(session.turns.length, 5);
    assertEquals(host.created(), 1);
  });

  it("the reserve is not passed to generate, and the reply can run past it", async () => {
    const host = fakeHost([{ text: LONG_ASK, closes: true }], programOf(640));
    const session = new ChatSession(host, { reserveNewTokens: 1 });
    const stream = session.send(FIRST.content);
    assertEquals(await stream.text(), LONG_ASK);

    const stop = await stream.done;
    assertEquals(stop.reason, "eos");
    assert(stop.tokens > 1, `the reply stayed within the reserve: ${stop.tokens}`);
    assertEquals((await stream.reply).end, "turn");
    assertEquals(host.requestedMaxNewTokens(), [undefined]);
  });

  it("a capacity stop is cut-off: the text stays in the history and the next send rebuilds", async () => {
    const host = fakeHost([{ text: LONG_ANSWER, closes: true }, RED], programOf(TIGHT_CAPACITY));
    const session = new ChatSession(host, { reserveNewTokens: 1 });
    const prompt = chatPrompt(tokenizer, [FIRST]).length;
    const stream = session.send(FIRST.content);
    const text = await stream.text();

    const stop = await stream.done;
    assertEquals(stop, { reason: "capacity", tokens: TIGHT_CAPACITY - prompt + 1 });
    assertEquals(await stream.reply, { end: "cut-off", stop });
    assertEquals(session.turns, [FIRST, { role: "assistant", content: text }]);
    assertEquals(host.disposed(), 1, "the KV of a turn that did not close is discarded");

    // The history fills the capacity, so the re-render drops the exchange and starts over.
    assertEquals(await session.send(SECOND.content).text(), "Red.");
    assertEquals(host.created(), 2);
    assertEquals(host.prompts()[1], chatPrompt(tokenizer, [SECOND]));
  });

  it("a re-render that fits keeps the cut-off text", async () => {
    const third = { role: "user", content: "One more." } as const;
    const firstAnswer: Answer = { text: LONG_ASK, closes: true };
    // The second turn has room for exactly 5 tokens, so its answer is cut there.
    const capacity =
      chatPrompt(tokenizer, [FIRST]).length +
      answerCost(firstAnswer) +
      chatTurn(tokenizer, SECOND).length +
      4;
    const host = fakeHost(
      [firstAnswer, { text: LONG_ANSWER, closes: true }, RED],
      programOf(capacity),
    );
    const session = new ChatSession(host, { reserveNewTokens: 1 });
    await session.send(FIRST.content).text();
    const stream = session.send(SECOND.content);
    const cut = await stream.text();
    assertEquals(await stream.done, { reason: "capacity", tokens: 5 });

    await session.send(third.content).text();

    assertEquals(
      host.prompts()[2],
      chatPrompt(tokenizer, [SECOND, { role: "assistant", content: cut }, third]),
      "the oldest exchange is dropped, and the cut-off text is what the re-render shows",
    );
    assertEquals(host.created(), 2);
  });

  it("the overflow failure names reserveNewTokens when the reserve did not fit", async () => {
    const host = fakeHost([], programOf(TIGHT_CAPACITY));
    const session = new ChatSession(host, { reserveNewTokens: 4 });
    const error = await assertRejects(
      () => session.send(LONG_ANSWER).text(),
      GenerationCapacityError,
      "lower reserveNewTokens. At this length reserveNewTokens ≤",
    );
    assertEquals(error.requestedNewTokens, 4);
    assertEquals(host.created(), 0);
  });

  it("the default reserve is 1024 and is not clamped to a small capacity", async () => {
    const host = fakeHost([], programOf(640));
    const session = new ChatSession(host, {});
    const error = await assertRejects(
      () => session.send(FIRST.content).text(),
      GenerationCapacityError,
      "reserveNewTokens",
    );
    assertEquals(error.requestedNewTokens, 1024);
  });

  it("the overflow failure names maxNewTokens when a limit did not fit", async () => {
    const host = fakeHost([], programOf(TIGHT_CAPACITY));
    const session = new ChatSession(host, { maxNewTokens: 4 });
    const error = await assertRejects(
      () => session.send(LONG_ANSWER).text(),
      GenerationCapacityError,
      "lower maxNewTokens. At this length maxNewTokens ≤",
    );
    assert(!error.message.includes("reserveNewTokens"));
  });

  it("an invalid reserve fails at construction", () => {
    const host = fakeHost([], programOf(640));
    for (const reserveNewTokens of [0, -1, 1.5, Number.NaN, null as unknown as number]) {
      assertThrows(
        () => new ChatSession(host, { reserveNewTokens }),
        ModelInputError,
        "is not a positive integer",
      );
    }
  });

  it("a reserve with a session maxNewTokens fails at construction", () => {
    assertThrows(
      () => new ChatSession(fakeHost([], programOf(640)), { maxNewTokens: 8, reserveNewTokens: 8 }),
      ModelInputError,
      "cannot be combined",
    );
  });

  it("a per-turn maxNewTokens overrides the reserve for that turn", async () => {
    // The default reserve does not fit this capacity, so only the per-turn value lets the turn pass.
    const host = fakeHost([BLUE], programOf(640));
    const session = new ChatSession(host, {});
    assertEquals(await session.send(FIRST.content, { maxNewTokens: 8 }).text(), "Blue.");
    assertEquals(host.requestedMaxNewTokens(), [8]);
  });

  it("a per-turn null is an invalid request, not an omitted one", () => {
    const session = new ChatSession(fakeHost([], programOf(640)), { reserveNewTokens: 8 });
    assertThrows(
      () => session.send(FIRST.content, { maxNewTokens: null as unknown as number }),
      ModelInputError,
      "maxNewTokens null",
    );
  });

  it("fork keeps the reserve", async () => {
    const host = fakeHost([], programOf(TIGHT_CAPACITY));
    const session = new ChatSession(host, { reserveNewTokens: 4 });
    const branch = await session.fork();
    const error = await assertRejects(
      () => branch.send(LONG_ANSWER).text(),
      GenerationCapacityError,
      "reserveNewTokens",
    );
    assertEquals(error.requestedNewTokens, 4);
  });

  it("fork keeps a session maxNewTokens", async () => {
    const host = fakeHost([BLUE], programOf(640));
    const session = new ChatSession(host, { maxNewTokens: 8 });
    const branch = await session.fork();
    await branch.send(FIRST.content).text();
    assertEquals(host.requestedMaxNewTokens(), [8]);
  });

  it("fork of a session without either keeps the default reserve", async () => {
    const host = fakeHost([], programOf(640));
    const branch = await new ChatSession(host, {}).fork();
    const error = await assertRejects(
      () => branch.send(FIRST.content).text(),
      GenerationCapacityError,
      "reserveNewTokens",
    );
    assertEquals(error.requestedNewTokens, 1024);
  });
});
