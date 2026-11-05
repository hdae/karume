// Behaviour tests of the generation surface for one conversation (`src/generation/sequence.ts`).
// Neither a GPU nor real assets are needed.
//
// What this pins is the contract of ADR 0083 decisions 1–5 and 10:
//
// - **Not a single token is lost across turns** (the prefill that prepends `pendingToken` —
//   decision 4). A loss raises no exception; "the previous assistant turn's last token disappears
//   from the history" is all that happens, so the parity check on a real GPU would not notice. The
//   **three paths** are checked: after an EOS stop / after a max-tokens stop / after a `break`
//   interruption.
// - The only mutable state is the context and `pendingToken` (the position is derived from
//   `context.pastLength` each time). The observation points (`used` / `GenerationStop.tokens` / the
//   fields of the capacity exception) must also **derive from those same two**.
// - Serialization per "one generate call", passing `AbortSignal` through, and the dedicated type
//   for exceeding capacity.
// - **Forking (`fork`)**: an independent sequence that inherits the frontier and the logical
//   length, the order of the chain, and lifetimes (ADR 0083 addendum 2026-10-11 — handing over the
//   drafter and the gate is T18 in `generation_speculation_test.ts`).
//
// The Session and the context are taken through narrow interfaces (`GenerationSession` /
// `GenerationContextFace`), so a fake is just one plain object.

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type { GenerationContext, RunInputs, Session } from "@karume/runtime";
import { ModelInputError } from "../src/errors.ts";
import type { GenerationProgramSpec, GenerationWiring } from "../src/generation/program.ts";
import {
  assertGenerationRequestValues,
  createGenerationSequence,
  type ForkableContextFace,
  GenerationCapacityError,
  type GenerationEvent,
  type GenerationRequest,
  type GenerationSession,
  physicalChunkRows,
} from "../src/generation/sequence.ts";
import {
  CHUNK_LENGTH,
  DERIVED,
  drain,
  type FakeSession,
  fakeSession,
  LOGITS,
  programOf,
  tokenIds,
  VOCAB,
} from "./helpers/generation-fake.ts";

test("GenerationSequence: prefill uses fixed-length chunks, pad 0 and absolute positions; decode is one row", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const { events, stop } = await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }));

  // The only binding point of the capacity symbol is context creation (ADR 0066 addendum 7).
  assertEquals(fake.specs.length, 1);
  assertEquals(fake.specs[0].bindings, { C: 64 });
  assertEquals(fake.specs[0].chunkLength, CHUNK_LENGTH);
  // A wiring that declares no buckets passes an empty array (the context's execution shapes are one
  // prefill + the decode, two in all).
  assertEquals(fake.specs[0].chunkBuckets, []);
  assertEquals(
    fake.calls.map((call) => call.bindings),
    [undefined, undefined, undefined],
  );
  assertEquals(
    fake.calls.every((call) => call.sameContext),
    true,
  );

  // prefill: 3 valid rows in 4 physical rows + pad 0, absolute positions, last_row is the last
  // valid row.
  assertEquals(fake.calls[0].idsShape, [1, CHUNK_LENGTH]);
  assertEquals(fake.calls[0].ids, [1, 2, 3, 0]);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 0]);
  assertEquals(fake.calls[0].lastRow, 2);
  assertEquals(fake.calls[0].lastRowShape, [1]);
  assertEquals(fake.calls[0].queryLength, 3);

  // decode: always 1 row, eats the previous step's token as is, positions continue.
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[1].ids, [5]);
  assertEquals(fake.calls[1].positions, [3]);
  assertEquals(fake.calls[1].lastRow, 0);
  assertEquals(fake.calls[2].ids, [6]);
  assertEquals(fake.calls[2].positions, [4]);

  assertEquals(events, [
    { kind: "prefill", chunk: 1, chunks: 1 },
    { kind: "token", id: 5, position: 3 },
    { kind: "token", id: 6, position: 4 },
    { kind: "token", id: 7, position: 5 },
  ]);
  assertEquals(stop, { reason: "max-tokens", tokens: 3 });
});

test("GenerationSequence: a long prompt emits one prefill event per chunk", async () => {
  const fake = fakeSession({ tokens: [1, 2, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const { events } = await drain(
    sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 1 }),
  );

  assertEquals(fake.calls.length, 2);
  assertEquals(fake.calls[0].ids, [1, 2, 3, 4]);
  assertEquals(fake.calls[0].positions, [0, 1, 2, 3]);
  assertEquals(fake.calls[1].ids, [5, 6, 0, 0]);
  assertEquals(fake.calls[1].positions, [4, 5, 0, 0]);
  assertEquals(fake.calls[1].lastRow, 1);
  // Generation starts from the **last valid row of the last chunk** (the output of the intermediate
  // chunks is discarded).
  assertEquals(events, [
    { kind: "prefill", chunk: 1, chunks: 2 },
    { kind: "prefill", chunk: 2, chunks: 2 },
    { kind: "token", id: 2, position: 6 },
  ]);
});

test("GenerationSequence: a mismatch between the rows requested and the rows returned fails right after the run", async () => {
  // The exit row count R is symbolic in the graph, so "ask for 1 row, get 2 back" is **correct as a
  // declaration** (it passes the shape check). This path always passes a single index, so checking
  // the returned row count is the only line that detects "a Session built from a different graph
  // than the one the program verified".
  const fake = fakeSession({ logitsRows: 2 });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 1 })),
    Error,
    `prefill@0: '${LOGITS}' has shape [1,2,${VOCAB}], not [1,1,${VOCAB}]`,
  );
});

// ---- prefill buckets (ADR 0066 addendum, buckets) -------------------------------
//
// Pad rows contribute neither to the output nor to the KV, yet the work of row-local ops scales
// with the physical row count. If the choice to run a short chunk at a bucket's rows goes wrong,
// **the values stay correct** (pad rows are 0 and harmless), so a regression shows up only in
// performance = this is the only check.

/** A wiring with buckets (chunkLength 8 / bucket 4 — the three shapes 1 / 4 / 8 appear). */
const BUCKET_CHUNK_LENGTH = 8;
const BUCKETS = [4] as const;

const bucketProgram = (fake: FakeSession): GenerationWiring =>
  programOf(fake, { chunkLength: BUCKET_CHUNK_LENGTH, chunkBuckets: [...BUCKETS] });

test("physicalChunkRows: 1 valid row is the decode shape; otherwise the smallest bucket ≥ queryLength", () => {
  const fake = fakeSession();
  const program = bucketProgram(fake);
  // 1 row is the decode shape (buckets do not change it — the key to bit-identical multi-turn
  // resumption).
  assertEquals(physicalChunkRows(1, program), 1);
  // A chunk that fits a bucket gets the bucket's rows (2..4 get 4 rows).
  assertEquals(physicalChunkRows(2, program), 4);
  assertEquals(physicalChunkRows(4, program), 4);
  // Overflowing goes to the next shape (chunkLength once the buckets run out).
  assertEquals(physicalChunkRows(5, program), BUCKET_CHUNK_LENGTH);
  assertEquals(physicalChunkRows(BUCKET_CHUNK_LENGTH, program), BUCKET_CHUNK_LENGTH);
  // A wiring without the declaration has only the two shapes 1 or chunkLength, as before.
  const plain = programOf(fake);
  assertEquals(physicalChunkRows(1, plain), 1);
  assertEquals(physicalChunkRows(2, plain), CHUNK_LENGTH);
});

test("GenerationSequence: declaring chunkBuckets runs short chunks at the bucket's rows", async () => {
  /** Prompt length → the physical row counts of the runs that generation issued (`[1, rows]`). */
  const rowsFor = async (promptLength: number): Promise<readonly number[]> => {
    const fake = fakeSession();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: bucketProgram(fake),
    });
    // The declaration is passed as is at context creation (it sets the set the pre-run check
    // allows).
    assertEquals(fake.specs[0].chunkBuckets, [...BUCKETS]);
    assertEquals(fake.specs[0].chunkLength, BUCKET_CHUNK_LENGTH);
    await drain(
      sequence.generate({
        prompt: Array.from({ length: promptLength }, (_unused, index) => index + 1),
        maxNewTokens: 1,
      }),
    );
    return fake.calls.map((call) => {
      assertEquals(call.idsShape[0], 1);
      return call.idsShape[1];
    });
  };

  // 2..4 rows go on the bucket, and from 5 rows it goes up to chunkLength. 9 rows split into 2
  // chunks, and the last chunk with 1 valid row is the decode shape (1 row, not the bucket).
  assertEquals(await rowsFor(2), [4]);
  assertEquals(await rowsFor(3), [4]);
  assertEquals(await rowsFor(4), [4]);
  assertEquals(await rowsFor(5), [BUCKET_CHUNK_LENGTH]);
  assertEquals(await rowsFor(BUCKET_CHUNK_LENGTH), [BUCKET_CHUNK_LENGTH]);
  assertEquals(await rowsFor(9), [BUCKET_CHUNK_LENGTH, 1]);
});

test("GenerationSequence: pad stays id 0 / position 0 at bucket rows too", async () => {
  const fake = fakeSession({ tokens: [5] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: bucketProgram(fake),
  });
  await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 }));

  // 2 valid rows + 2 pad rows (the value contract of pad is the same when the physical row count
  // shrinks — ADR 0066 addendum 6 / 8).
  assertEquals(fake.calls[0].idsShape, [1, 4]);
  assertEquals(fake.calls[0].ids, [1, 2, 0, 0]);
  assertEquals(fake.calls[0].positions, [0, 1, 0, 0]);
  assertEquals(fake.calls[0].lastRow, 1);
  assertEquals(fake.calls[0].queryLength, 2);
});

// ---- multi-turn: direct check that "the previous assistant turn's last token is not lost" (3
// paths) ----

test("multi-turn: the pendingToken after a max-tokens stop is prepended to the next turn's prompt", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 9] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const first = await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }));
  assertEquals(tokenIds(first.events), [5, 6, 7]);
  // After generating K tokens pastLength is T + K − 1 (the last one, 7, is the uncommitted
  // frontier).
  assertEquals(fake.pastLength(), 5);

  await drain(sequence.generate({ prompt: [8, 9], maxNewTokens: 1 }));
  // The previous 7 is prepended and the positions do not break (dropping it would remove a token
  // from the conversation).
  assertEquals(fake.calls[3].ids, [7, 8, 9, 0]);
  assertEquals(fake.calls[3].positions, [5, 6, 7, 0]);
  assertEquals(fake.calls[3].queryLength, 3);
  assertEquals(fake.calls[3].pastBefore, 5);
});

test("multi-turn: on an EOS stop the stop token stays in the conversation (but is not emitted as an event)", async () => {
  const fake = fakeSession({ tokens: [5, 11, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { stopTokens: [11] }),
  });
  const first = await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 4 }));

  // The stop token is a terminator, not body text, so it is not emitted as a `token` event.
  assertEquals(tokenIds(first.events), [5]);
  // The stop token itself counts as one (one draw = one run — the doc of `GenerationStop.tokens`).
  assertEquals(first.stop, { reason: "eos", token: 11, tokens: 2 });
  // The stop ends decode (it does not run up to maxNewTokens).
  assertEquals(fake.calls.length, 2);

  await drain(sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  // The stop token goes to the head of the next turn's prefill (dropping chat's `<turn|>` breaks
  // the conversation).
  assertEquals(fake.calls[2].ids, [11, 4, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

test("request stopTokens: even when the distribution's set is empty, stop tokens can be added for that turn alone", async () => {
  const fake = fakeSession({ tokens: [5, 11, 3] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { stopTokens: [] }),
  });
  const first = await drain(
    sequence.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [11] }),
  );

  // The stop token is not body text, so it is not emitted as a `token` event (the same as an EOS
  // stop).
  assertEquals(tokenIds(first.events), [5]);
  // Only the reason differs from EOS (it stopped for this request's needs, not on the
  // distribution's terminator).
  assertEquals(first.stop, { reason: "stop-token", token: 11, tokens: 2 });
  assertEquals(fake.calls.length, 2, "the stop ends decode");

  // It stays in the conversation = it is prepended to the next turn's prefill (the same cleanup as
  // after an EOS stop).
  await drain(sequence.generate({ prompt: [4], maxNewTokens: 1 }));
  assertEquals(fake.calls[2].ids, [11, 4, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

test("request stopTokens: the decision uses the union with the distribution's EOS set (EOS always applies)", async () => {
  // A stop set passed by the request does not override the distribution's EOS (both apply = the
  // union).
  const eosFake = fakeSession({ tokens: [5, 11, 3] });
  const eos = await createGenerationSequence({
    session: eosFake.session,
    program: programOf(eosFake, { stopTokens: [11] }),
  });
  assertEquals(
    (await drain(eos.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [9] }))).stop,
    { reason: "eos", token: 11, tokens: 2 },
    "a turn that stopped on the distribution's EOS",
  );

  // An id in both closes as `eos` (its meaning as the distribution's terminator takes precedence).
  const bothFake = fakeSession({ tokens: [5, 11, 3] });
  const both = await createGenerationSequence({
    session: bothFake.session,
    program: programOf(bothFake, { stopTokens: [11] }),
  });
  assertEquals(
    (await drain(both.generate({ prompt: [1, 2], maxNewTokens: 4, stopTokens: [11] }))).stop,
    { reason: "eos", token: 11, tokens: 2 },
    "a stop token in both sets",
  );
});

test("request stopTokens: out-of-vocabulary ids and duplicates fail synchronously (no stop condition that never applies)", async () => {
  // A stop token is "an id that does not appear in the output", so a wrong one still lets
  // generation run to completion normally (that id is just never drawn) — the only symptom is
  // output that keeps growing, so it fails at the entry.
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [VOCAB] }),
    Error,
    `stopTokens[0] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`,
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [1, -1] }),
    Error,
    "stopTokens[1] -1",
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, stopTokens: [3, 5, 3] }),
    Error,
    "Token 3 appears twice in stopTokens",
  );
  assertEquals(fake.calls.length, 0);
});

test("request stopTokens: appending to the array after the call has no effect on the running generation", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  // A mutable stop set the caller keeps holding (empty at the call = a spec that never stops).
  const stopTokens: number[] = [];
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3, stopTokens });
  // The body does not run until the first `next()`, so before draining is the widest window for
  // changes.
  stopTokens.push(6);

  const { events, stop } = await drain(stream);
  // If the late addition took effect, it would stop at 6 and give [5].
  assertEquals(tokenIds(events), [5, 6, 7]);
  assertEquals(stop, { reason: "max-tokens", tokens: 3 });
});

test("multi-turn: after a break interruption the pendingToken remains and the next turn loses no token", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8, 9, 10] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });

  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 5 });
  const seen: number[] = [];
  for await (const event of stream) {
    if (event.kind !== "token") continue;
    seen.push(event.id);
    // Stop after receiving the second one (the path that enters finally through `return()`).
    if (seen.length === 2) break;
  }
  assertEquals(seen, [5, 6]);
  assertEquals(await stream.done, { reason: "closed", tokens: 2 });
  const callsAtBreak = fake.calls.length;
  assertEquals(callsAtBreak, 2);
  assertEquals(fake.pastLength(), 3);

  await drain(sequence.generate({ prompt: [3], maxNewTokens: 1 }));
  assertEquals(fake.calls[2].ids, [6, 3, 0, 0]);
  assertEquals(fake.calls[2].positions, [3, 4, 0, 0]);
});

test("multi-turn: a continuation-only turn (empty prompt) runs the pendingToken in the decode shape", async () => {
  // A chunk with one valid row is one physical row = the decode shape. Resuming from an
  // interruption becomes **a run of the same shape** as "the run that was not interrupted" and does
  // not take the prefill shape (M=chunkLength) needlessly.
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 5 });
  for await (const event of stream) {
    if (event.kind === "token") break;
  }
  assertEquals(fake.calls.length, 1);

  await drain(sequence.generate({ prompt: [], maxNewTokens: 2 }));
  assertEquals(fake.calls[1].idsShape, [1, 1]);
  assertEquals(fake.calls[1].ids, [5]);
  assertEquals(fake.calls[1].positions, [2]);
  assertEquals(fake.calls[1].lastRow, 0);
  assertEquals(fake.calls[1].queryLength, 1);
});

test("GenerationSequence: an empty prompt with no pendingToken fails loudly", async () => {
  // The prompt is the public request itself, and the one remedy is "put tokens in" = an input error
  // (ADR 0107 decision 2). The other side (an empty prompt is accepted when there is a
  // pendingToken) is pinned by the "continuation-only turn" just above.
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await assertRejects(
    () => drain(sequence.generate({ prompt: [], maxNewTokens: 1 })),
    ModelInputError,
    "prompt is empty",
  );
  assertEquals(fake.calls.length, 0);
});

// ---- interruption (AbortSignal) and serialization ----

test("GenerationSequence: abort throws signal.reason as is, unwrapped", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const controller = new AbortController();
  const reason = new Error("the caller stopped it");
  const stream = sequence.generate({
    prompt: [1, 2],
    maxNewTokens: 5,
    signal: controller.signal,
  });

  let caught: unknown;
  const seen: number[] = [];
  try {
    for await (const event of stream) {
      if (event.kind !== "token") continue;
      seen.push(event.id);
      controller.abort(reason);
    }
  } catch (error) {
    caught = error;
  }
  // Not wrapped (the consumer can identify its own interruption by
  // `error === controller.signal.reason`).
  assert(caught === reason, `the interruption's exception is wrapped: ${String(caught)}`);
  assertEquals(seen, [5]);
  assertEquals(await stream.done, { reason: "aborted", tokens: 1 });
  // The interruption takes effect at a stage boundary (just before the next run) — it does not kill
  // a running run.
  assertEquals(fake.calls.length, 1);

  // Even on an interruption the conversation advances "by the successful runs", and the frontier
  // remains.
  assertEquals(fake.pastLength(), 2);
  await drain(sequence.generate({ prompt: [3], maxNewTokens: 1 }));
  assertEquals(fake.calls[1].ids, [5, 3, 0, 0]);
});

test("GenerationSequence: an already-aborted signal fails without issuing a single run", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const controller = new AbortController();
  const reason = new Error("aborted from the start");
  controller.abort(reason);
  const stream = sequence.generate({
    prompt: [1, 2],
    maxNewTokens: 2,
    signal: controller.signal,
  });
  let caught: unknown;
  try {
    for await (const _event of stream) {
      /* none arrive */
    }
  } catch (error) {
    caught = error;
  }
  assert(caught === reason);
  assertEquals(fake.calls.length, 0);
  assertEquals(await stream.done, { reason: "aborted", tokens: 0 });
});

describe("GenerationSequence: an interruption that arrives after the last prefill still delivers one token from the computed logits", () => {
  // The interruption is checked only before a run is issued (at a stage boundary — ADR 0083
  // decision 5). The last prefill's run has already returned its logits, so even with an
  // interruption arriving after it the first token is drawn and delivered, and no new run starts
  // (by design — docs/limitations.md, section "Generation sequence").
  for (const [name, maxNewTokens, done] of [
    // With nothing left, no boundary comes at which to observe the interruption, so it closes as a
    // normal completion.
    ["maxNewTokens 1", 1, { reason: "max-tokens", tokens: 1 }],
    // With something left, the interruption takes effect before the next decode run.
    ["maxNewTokens 3", 3, { reason: "aborted", tokens: 1 }],
  ] as const) {
    it(name, async () => {
      const fake = fakeSession({ tokens: [5, 6, 7] });
      const sequence = await createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
      });
      const controller = new AbortController();
      const reason = new Error("stopped after the last prefill");
      const stream = sequence.generate({
        prompt: [1, 2],
        maxNewTokens,
        signal: controller.signal,
      });
      const events: GenerationEvent[] = [];
      let caught: unknown;
      try {
        for await (const event of stream) {
          events.push(event);
          if (event.kind === "prefill" && event.chunk === event.chunks) controller.abort(reason);
        }
      } catch (error) {
        caught = error;
      }
      assertEquals(tokenIds(events), [5]);
      assertEquals(await stream.done, done);
      assert(
        done.reason === "aborted" ? caught === reason : caught === undefined,
        `${name}: the exception is carried the wrong way (${String(caught)})`,
      );
      // Not a single run was issued after the interruption (only the one of the last prefill).
      assertEquals(fake.calls.length, 1);
    });
  }
});

test("GenerationSequence: an interruption that arrives while queued closes without issuing a single run", async () => {
  // A request aborted while waiting closes **without deciding anything** when its turn comes.
  // Returning the budget overrun first would mean "I stopped it and got a capacity error", and the
  // caller could not identify its own interruption.
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });

  const first = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  const iterator = first[Symbol.asyncIterator]();
  await iterator.next();

  const controller = new AbortController();
  const reason = new Error("stopped while queued");
  // past 4 + prompt (1 prepended + 6) + K 4 − 1 = 14 > capacity 8 = a sure capacity overrun once
  // its turn comes.
  const queued = sequence.generate({
    prompt: [1, 2, 3, 4, 5, 6],
    maxNewTokens: 4,
    signal: controller.signal,
  });
  const consumed = (async () => {
    for await (const _event of queued) {
      /* none arrive */
    }
  })();
  // Interrupt after it has entered the queue.
  await new Promise((resolve) => setTimeout(resolve, 0));
  controller.abort(reason);

  while (!(await iterator.next()).done) {
    /* free the slot */
  }
  const callsAfterFirst = fake.calls.length;
  let caught: unknown;
  try {
    await consumed;
  } catch (error) {
    caught = error;
  }
  assert(caught === reason, `closed on another failure, not the interruption: ${String(caught)}`);
  assertEquals(await queued.done, { reason: "aborted", tokens: 0 });
  assertEquals(fake.calls.length, callsAfterFirst);
});

test("GenerationSequence: two generate calls are serialized (their runs do not interleave)", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8, 9, 10] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });

  const log: string[] = [];
  const consume = async (label: string, prompt: readonly number[]): Promise<void> => {
    for await (const event of sequence.generate({ prompt, maxNewTokens: 2 })) {
      log.push(`${label}:${event.kind}`);
    }
  };
  await Promise.all([consume("a", [1, 2]), consume("b", [3])]);

  // All events of one come before the other's (interleaving would mean the chain is not working).
  const boundary = log.findIndex((entry) => entry.startsWith("b:"));
  assert(boundary > 0, `not serialized: ${log.join(" ")}`);
  assertEquals(
    log.slice(0, boundary).every((entry) => entry.startsWith("a:")),
    true,
  );
  assertEquals(
    log.slice(boundary).every((entry) => entry.startsWith("b:")),
    true,
  );
  // b starts with a prompt that prepends a's frontier (6) = the state advanced while it was queued.
  assertEquals(fake.calls[2].ids, [6, 3, 0, 0]);
});

// ---- capacity (ADR 0083 decision 10 / the runtime knob of the variable-capacity work) ----

test("capacity knob: the default is the program's declaration, and a passed value becomes the context's binding", async () => {
  const declared = fakeSession();
  const byDefault = await createGenerationSequence({
    session: declared.session,
    program: programOf(declared),
  });
  assertEquals(byDefault.capacity, 64, "the distribution's default when omitted");
  assertEquals(
    declared.specs[0].bindings,
    { C: 64 },
    "the capacity symbol's binding is the default too",
  );

  const chosen = fakeSession();
  const narrowed = await createGenerationSequence({
    session: chosen.session,
    program: programOf(chosen),
    capacity: 16,
  });
  // The only binding point is context creation (ADR 0066 addendum 7) — this value sets the physical
  // allocation of the state slots.
  assertEquals(narrowed.capacity, 16);
  assertEquals(chosen.specs[0].bindings, { C: 16 });
  assertEquals(chosen.specs[0].chunkLength, CHUNK_LENGTH, "the chunk length stays the program's");
});

test("capacity knob: the budget check uses the chosen capacity, not the program's default", async () => {
  const fake = fakeSession({ tokens: [5] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
    capacity: 8,
  });
  // peak = 0 + 6 + 4 − 1 = 9 > 8 (a shape that would pass with the program's default of 64).
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 4 })),
    GenerationCapacityError,
    "exceeds the state capacity",
  );
  assertEquals(error.limit, 8, "the limit the exception carries is the chosen capacity too");
  assertEquals(fake.calls.length, 0);
});

test("capacity knob: a value that breaks the relations fails loudly when the sequence is created", async () => {
  // The type is pinned too so that the public entry points `sequence({ capacity })` and
  // `chat({ capacity })` do not return **the same spec** under different classifications. The three
  // conditions of the same name in `gemma4_chat_session_test.ts` also expect `ModelInputError`, so
  // if only one side went back to a plain Error one of them would turn red (ADR 0107 decision 2).
  const fake = fakeSession();
  // A capacity that cannot hold even one chunk (do not let a context that cannot issue a single run
  // be created).
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: CHUNK_LENGTH - 1,
      }),
    ModelInputError,
    `capacity ${CHUNK_LENGTH - 1} is below chunkLength ${CHUNK_LENGTH}`,
  );
  // A capacity reaching beyond the model's position limit (a conversation filling it would touch
  // positions outside training).
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: 129,
      }),
    ModelInputError,
    "capacity 129 exceeds maxPosition 128",
  );
  await assertRejects(
    () =>
      createGenerationSequence({
        session: fake.session,
        program: programOf(fake),
        capacity: 8.5,
      }),
    ModelInputError,
    "capacity 8.5 is not a positive integer",
  );
  assertEquals(fake.specs.length, 0, "not a single context was allocated");
});

test("GenerationSequence: exceeding the capacity is a GenerationCapacityError (it fails before the run)", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });
  // peak = past 0 + prompt 6 + maxNewTokens 4 − 1 = 9 > 8.
  await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4, 5, 6], maxNewTokens: 4 })),
    GenerationCapacityError,
    "exceeds the state capacity",
  );
  assertEquals(fake.calls.length, 0);
  // A shape that fits exactly passes (peak = 8).
  await drain(sequence.generate({ prompt: [1, 2, 3, 4, 5], maxNewTokens: 4 }));
  assertEquals(fake.pastLength(), 8);
});

describe("a request without maxNewTokens generates until a stop token or the limit", () => {
  /** Capacity 8, so a prompt of T tokens on an empty conversation has the limit K = 8 − T + 1. */
  const openTight = async (override: Partial<GenerationProgramSpec> = {}) => {
    const fake = fakeSession({ tokens: [5, 6, 7, 8, 9, 10, 11, 12] });
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, { capacity: 8, ...override }),
    });
    return { fake, sequence };
  };

  it("runs to exactly the limit and closes as capacity", async () => {
    const { fake, sequence } = await openTight();
    const { events, stop } = await drain(sequence.generate({ prompt: [1, 2, 3] }));

    assertEquals(stop, { reason: "capacity", tokens: 6 });
    assertEquals(tokenIds(events), [5, 6, 7, 8, 9, 10]);
    // The KV is full. The last token is the uncommitted frontier, so `used` (which counts it) is one
    // past the capacity — the same as an explicit request of that size.
    assertEquals(fake.pastLength(), 8);
    assertEquals(sequence.used, 9);
  });

  it("closes as eos when a stop token comes first", async () => {
    const { sequence } = await openTight({ stopTokens: [6] });
    const { stop } = await drain(sequence.generate({ prompt: [1, 2, 3] }));
    assertEquals(stop, { reason: "eos", token: 6, tokens: 2 });
  });

  it("closes as stop-token on a stop token of the request", async () => {
    const { sequence } = await openTight();
    const { stop } = await drain(sequence.generate({ prompt: [1, 2, 3], stopTokens: [7] }));
    assertEquals(stop, { reason: "stop-token", token: 7, tokens: 3 });
  });

  it("an explicit value equal to the limit runs the same turn and closes as max-tokens", async () => {
    const absent = await openTight();
    const explicit = await openTight();
    const unlimited = await drain(absent.sequence.generate({ prompt: [1, 2, 3] }));
    const limited = await drain(explicit.sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 6 }));

    assertEquals(limited.stop, { reason: "max-tokens", tokens: 6 });
    assertEquals(unlimited.events, limited.events);
    assertEquals(absent.fake.calls, explicit.fake.calls);
    assertEquals(absent.sequence.used, explicit.sequence.used);
  });

  it("a prompt that fits with K = 1 draws one token and closes as capacity", async () => {
    const { fake, sequence } = await openTight();
    const { events, stop } = await drain(sequence.generate({ prompt: [1, 2, 3, 4, 5, 6, 7, 8] }));

    assertEquals(stop, { reason: "capacity", tokens: 1 });
    assertEquals(tokenIds(events).length, 1);
    assertEquals(fake.calls.length, 2, "two prefill chunks and no decode");
  });

  it("a prompt that does not fit throws GenerationCapacityError before any run", async () => {
    const { fake, sequence } = await openTight();
    const stream = sequence.generate({ prompt: [1, 2, 3, 4, 5, 6, 7, 8, 9] });
    const error = await assertRejects(
      () => drain(stream),
      GenerationCapacityError,
      "exceeds the state capacity",
    );

    assertEquals(fake.calls.length, 0);
    assertEquals(error.requestedNewTokens, 1, "the smallest turn there is");
    assertEquals(error.maxNewTokens, 0, "nothing fits now");
    assertEquals(error.limit, 8);
    await assertRejects(() => stream.done, GenerationCapacityError);
  });

  it("the second turn resolves against the remaining room, the pending token included", async () => {
    const { fake, sequence } = await openTight();
    await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 2 }));
    assertEquals(fake.pastLength(), 3, "two prompt tokens and the first generated one");

    // past 3, prompt 3 (the pending token + 2 new) → K = 8 − 3 − 3 + 1 = 3.
    const { stop } = await drain(sequence.generate({ prompt: [3, 4] }));
    assertEquals(stop, { reason: "capacity", tokens: 3 });
    assertEquals(fake.calls[2].ids.slice(0, 3), [6, 3, 4], "the pending token leads the prefill");
    assertEquals(fake.pastLength(), 8);
  });

  it("a conversation that filled up refuses the next turn (the pending token alone does not fit)", async () => {
    const { fake, sequence } = await openTight();
    await drain(sequence.generate({ prompt: [1, 2, 3] }));
    const runs = fake.calls.length;

    await assertRejects(() => drain(sequence.generate({ prompt: [] })), GenerationCapacityError);
    assertEquals(fake.calls.length, runs);
  });

  it("the position table stays in range when it is the same size as the capacity", async () => {
    const { fake, sequence } = await openTight({ maxPosition: 8 });
    const { stop } = await drain(sequence.generate({ prompt: [1, 2, 3] }));

    assertEquals(stop, { reason: "capacity", tokens: 6 });
    assert(fake.calls.every((call) => call.positions.every((position) => position < 8)));
  });

  it("the range check treats only undefined as omitted", () => {
    assertGenerationRequestValues(VOCAB, {});
    for (const maxNewTokens of [Number.NaN, 0, null as unknown as number]) {
      assertThrows(
        () => assertGenerationRequestValues(VOCAB, { maxNewTokens }),
        ModelInputError,
        "is not a positive integer",
      );
    }
  });
});

test("GenerationSequence: used is derived each time from pastLength + the uncommitted frontier", async () => {
  // There is no independent counter (ADR 0066 decision 6 forbids double bookkeeping), so reading it
  // mid-run gives "up to the runs that have succeeded so far". Truncation decisions use a value
  // read between generations.
  const fake = fakeSession({ tokens: [5, 6, 7, 9] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertEquals(sequence.used, 0, "before generation");

  const stream = sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next(); // The prefill event (the frontier is in the KV, pendingToken is empty).
  assertEquals(sequence.used, 3, "right after prefill");
  await iterator.next(); // The first token (one uncommitted frontier appears).
  assertEquals(sequence.used, 4, "right after the first token");
  while (!(await iterator.next()).done) {
    /* drain it */
  }

  // pastLength is T + K − 1 (the last one is uncommitted), but the conversation occupies T + K.
  assertEquals(fake.pastLength(), 5);
  assertEquals(sequence.used, 6, "after turn 1");

  // The next turn builds on used (the prepended pendingToken is not counted twice).
  await drain(sequence.generate({ prompt: [8, 9], maxNewTokens: 1 }));
  assertEquals(sequence.used, 9, "after turn 2 (6 + 2 new + 1 generated)");
});

test("GenerationCapacityError: carries the actual values needed for truncation in fields (no parsing of the message)", async () => {
  const fake = fakeSession({ tokens: [5] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 8 }),
  });
  await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 }));
  assertEquals(sequence.used, 3);

  // past 2 + prompt (1 prepended + 4) 5 + K 4 − 1 = 10 > capacity 8.
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: 4 })),
    GenerationCapacityError,
    "exceeds the state capacity",
  );
  assertEquals(error.constraint, "capacity");
  assertEquals(error.pastLength, 2);
  assertEquals(
    error.promptLength,
    5,
    "the effective length after prepending pendingToken (not the caller's 4)",
  );
  assertEquals(error.requestedNewTokens, 4);
  assertEquals(error.limit, 8);
  assertEquals(error.maxNewTokens, 2);
  // Shrinking per the fields really passes (this one line keeps "derivable" from being a
  // tautology).
  await drain(sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: error.maxNewTokens }));
});

test("GenerationCapacityError: the position-table limit is carried in the same fields (only constraint differs)", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { maxPosition: 5, capacity: 5 }),
  });
  const error = await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 4 })),
    GenerationCapacityError,
    "leaves the position table",
  );
  assertEquals(error.constraint, "maxPosition");
  assertEquals(error.limit, 5);
  assertEquals(error.pastLength, 0);
  assertEquals(error.promptLength, 3);
  assertEquals(error.requestedNewTokens, 4);
  // An exclusive limit, but the formula folds into the same shape as the capacity side (5 − 0 − 3 +
  // 1 = 3).
  assertEquals(error.maxNewTokens, 3);
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: error.maxNewTokens }));
});

test("GenerationSequence: exceeding the position-table limit fails with the same type", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { maxPosition: 5, capacity: 5 }),
  });
  // Last position = 0 + 3 + 4 − 2 = 5 (outside the exclusive limit 5).
  await assertRejects(
    () => drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 4 })),
    GenerationCapacityError,
    "leaves the position table",
  );
  assertEquals(fake.calls.length, 0);
  // Last position 4 is just legal.
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 3 }));
  assertEquals(fake.calls.length, 3);
});

test("GenerationSequence: the capacity is checked when its own turn comes (including the progress of earlier turns)", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { capacity: 10 }),
  });
  await drain(sequence.generate({ prompt: [1, 2, 3, 4], maxNewTokens: 4 }));
  assertEquals(fake.pastLength(), 7);
  // past 7 + prompt (1 prepended + 2) 3 + K 2 − 1 = 11 > 10. At the call, past is not known.
  await assertRejects(
    () => drain(sequence.generate({ prompt: [5, 6], maxNewTokens: 2 })),
    GenerationCapacityError,
    "existing 7",
  );
});

// ---- host-derived per-chunk inputs (the PLE slot) ----

test("DerivedRunInputs: receives the id and position sequences including pad rows, and is wired as an input", async () => {
  const fake = fakeSession({ tokens: [5, 6] });
  const seen: { readonly ids: readonly number[]; readonly positions: readonly number[] }[] = [];
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, {
      derivedInputs: {
        names: [DERIVED],
        derive: (ids, positions) => {
          seen.push({ ids: [...ids], positions: [...positions] });
          return Promise.resolve({
            [DERIVED]: {
              dtype: "f32",
              shape: [1, ids.length, 2],
              data: new Float32Array(ids.length * 2),
            },
          } satisfies RunInputs);
        },
      },
    }),
  });
  await drain(sequence.generate({ prompt: [1, 2, 3], maxNewTokens: 2 }));

  // prefill covers the physical rows (pad rows get the same 0 as `input_ids`), decode one row. The
  // positions follow the same convention (pad rows are 0), and absolute positions advance by 1 from
  // where the prompt left off.
  assertEquals(seen, [
    { ids: [1, 2, 3, 0], positions: [0, 1, 2, 0] },
    { ids: [5], positions: [3] },
  ]);
  assertEquals(
    fake.calls.map((call) => call.extra),
    [[DERIVED], [DERIVED]],
  );
});

test("DerivedRunInputs: the generation's signal reaches derive (material for best effort)", async () => {
  // Depending on the distribution, the material of a derived input is a GB-class lazy load
  // (gemma4's PLE — blocks of the `model` container's asset). Without passing it down there would
  // be a stretch where "pressing stop does not return until the block has been read".
  const fake = fakeSession({ tokens: [5, 6] });
  const seen: (AbortSignal | undefined)[] = [];
  const controller = new AbortController();
  const derived = (): GenerationWiring["derivedInputs"] => ({
    names: [DERIVED],
    derive: (ids, _positions, options) => {
      seen.push(options?.signal);
      return Promise.resolve({
        [DERIVED]: {
          dtype: "f32",
          shape: [1, ids.length, 2],
          data: new Float32Array(ids.length * 2),
        },
      } satisfies RunInputs);
    },
  });

  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake, { derivedInputs: derived() }),
  });
  await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 2, signal: controller.signal }));
  assertEquals(seen.length, 2);
  assert(
    seen.every((signal) => signal === controller.signal),
    "derive received a different signal",
  );

  // None is invented for a request that did not subscribe.
  const bareFake = fakeSession();
  const bare = await createGenerationSequence({
    session: bareFake.session,
    program: programOf(bareFake, { derivedInputs: derived() }),
  });
  await drain(bare.generate({ prompt: [1], maxNewTokens: 1 }));
  assertEquals(seen[2], undefined);
});

test("DerivedRunInputs: an interruption that arrives as derive's await resumes issues no run", async () => {
  // If the interruption were observed only at "the top of the loop", a whole run would go ahead
  // after an interruption that arrived while the derived input was being built (the head of prefill
  // is always a cold miss, so stopping right after sending always hits it).
  for (const [name, maxNewTokens, tokensBefore] of [
    ["prefill", 2, 0],
    ["decode", 3, 1],
  ] as const) {
    const fake = fakeSession({ tokens: [5, 6, 7] });
    const controller = new AbortController();
    const reason = new Error(`stopped in the middle of the ${name} derived input`);
    let derives = 0;
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, {
        derivedInputs: {
          names: [DERIVED],
          derive: async (ids) => {
            derives += 1;
            // The interruption arrives in the middle of this derive's "read".
            if (derives > tokensBefore) {
              controller.abort(reason);
              await new Promise((resolve) => setTimeout(resolve, 0));
            }
            return {
              [DERIVED]: {
                dtype: "f32",
                shape: [1, ids.length, 2],
                data: new Float32Array(ids.length * 2),
              },
            } satisfies RunInputs;
          },
        },
      }),
    });

    const stream = sequence.generate({
      prompt: [1, 2],
      maxNewTokens,
      signal: controller.signal,
    });
    const events: GenerationEvent[] = [];
    let caught: unknown;
    try {
      for await (const event of stream) events.push(event);
    } catch (error) {
      caught = error;
    }
    assert(caught === reason, `${name}: the interruption is wrapped (${String(caught)})`);
    assertEquals(await stream.done, { reason: "aborted", tokens: tokensBefore }, name);
    // Not a single run was issued after the interruption arrived = not a single extra token arrived
    // either.
    assertEquals(
      fake.calls.length,
      tokensBefore,
      `${name}: a run went ahead after the interruption`,
    );
    assertEquals(tokenIds(events).length, tokensBefore === 0 ? 0 : 1, `${name}: extra token`);
  }
});

test("DerivedRunInputs: returning keys other than the declared ones fails loudly", async () => {
  const cases: readonly (readonly [string, RunInputs, string])[] = [
    ["missing", {}, "missing: per_layer_inputs"],
    [
      "extra",
      {
        [DERIVED]: { dtype: "f32", shape: [1, 4, 2], data: new Float32Array(8) },
        other: { dtype: "f32", shape: [1], data: new Float32Array(1) },
      },
      "extra: other",
    ],
  ];
  for (const [name, result, message] of cases) {
    const fake = fakeSession();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, {
        derivedInputs: { names: [DERIVED], derive: () => Promise.resolve(result) },
      }),
    });
    await assertRejects(
      () => drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 1 })),
      Error,
      message,
      name,
    );
  }
});

// ---- accepted set, failures, lifetime ----

test("GenerationSequence: the accepted set fails synchronously (it enters neither the queue nor the GPU)", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 0 }),
    ModelInputError,
    "maxNewTokens 0",
  );
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1.5 }),
    ModelInputError,
    "maxNewTokens 1.5",
  );
  // An `Int32Array` wrap and an out-of-vocabulary id both silently turn into another token (the
  // latter is an out-of-range gather = NaN poisoning).
  assertThrows(
    () => sequence.generate({ prompt: [1, VOCAB], maxNewTokens: 1 }),
    ModelInputError,
    `prompt[1] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`,
  );
  assertThrows(
    () => sequence.generate({ prompt: [4294967297], maxNewTokens: 1 }),
    ModelInputError,
    "prompt[0] 4294967297",
  );
  // The sampler spec fails at the call too (drawing runs in the middle of decode, so delaying it
  // would bury the failure deep).
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1, sampler: { temperature: -1 } }),
    ModelInputError,
    "temperature -1",
  );
  assertEquals(fake.calls.length, 0);
});

describe("the range check of the accepted set is one pure function (the high-level surface and generate see the same expression)", () => {
  // The high-level surface (`gemma.Pipeline.chat` / `gemma.ChatSession.send`) calls `generate` in
  // the body of an async generator, so the only way to fail at the call is to call this function
  // itself. Two copies of the expression would let one go stale, so that there is one source of
  // truth is itself pinned here.
  it("an acceptable request passes (the counterpart showing the check has not collapsed into a tautology)", () => {
    assertGenerationRequestValues(VOCAB, { maxNewTokens: 1 });
    assertGenerationRequestValues(VOCAB, {
      maxNewTokens: 8,
      stopTokens: [0, VOCAB - 1],
      sampler: { temperature: 1, topK: 4 },
    });
  });

  it("the range of maxNewTokens", () => {
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 0 }),
      ModelInputError,
      "0 is not a positive integer",
    );
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1.5 }),
      ModelInputError,
      "maxNewTokens 1.5",
    );
  });

  it("out-of-vocabulary and duplicate stopTokens", () => {
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1, stopTokens: [VOCAB] }),
      ModelInputError,
      `stopTokens[0] ${VOCAB} is outside the vocabulary 0..${VOCAB - 1}`,
    );
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, { maxNewTokens: 1, stopTokens: [2, 2] }),
      ModelInputError,
      "Token 2 appears twice",
    );
  });

  it("the sampler spec (drawing runs in the middle of decode, so it fails at the call)", () => {
    assertThrows(
      () =>
        assertGenerationRequestValues(VOCAB, {
          maxNewTokens: 1,
          sampler: { temperature: -1 },
        }),
      ModelInputError,
      "temperature -1",
    );
  });

  it("generate goes through the same function (no shape where only one side loosens)", async () => {
    // Fault injection: a value the function accepts generate accepts too, and a value it rejects
    // generate rejects too.
    const fake = fakeSession();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake),
    });
    const request = { prompt: [1], maxNewTokens: 1, stopTokens: [2, 2] };
    assertThrows(
      () => assertGenerationRequestValues(VOCAB, request),
      ModelInputError,
      "appears twice",
    );
    assertThrows(() => sequence.generate(request), ModelInputError, "appears twice");
    assertEquals(fake.calls.length, 0);
  });
});

test("GenerationSequence: unacceptable requests are caught with one type (ADR 0107)", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  // The four paths: range / vocabulary / combination / sampler. Without a single type, a host
  // serving several families could only carve out the 400s by **reading the message's spelling**
  // (which silently breaks the moment the wording changes).
  const rejected: readonly GenerationRequest[] = [
    { prompt: [1], maxNewTokens: 0 },
    { prompt: [1, VOCAB], maxNewTokens: 1 },
    { prompt: [1], maxNewTokens: 1, stopTokens: [2, 2] },
    { prompt: [1], maxNewTokens: 1, sampler: { topK: 0 } },
  ];
  for (const request of rejected) {
    // The form a host writes (`catch`, then split 400 / 500 by `instanceof`), as is.
    let caught: unknown;
    try {
      sequence.generate(request);
    } catch (error) {
      caught = error;
    }
    assert(
      caught instanceof ModelInputError,
      `not caught as an input error: ${JSON.stringify(request)}`,
    );
  }
  assertEquals(fake.calls.length, 0);
});

/** A mutable version for changing after the call (every field of the public type is readonly). */
type MutableRequest = { -readonly [Field in keyof GenerationRequest]: GenerationRequest[Field] };

test("GenerationSequence: the request is snapshotted at the call (changes after it have no effect on the running generation)", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  // A mutable request the caller keeps holding (both the arrays and the sampler's bias can be
  // touched later).
  const prompt = [1, 2];
  const bias: [number, number][] = [];
  const request: MutableRequest = { prompt, maxNewTokens: 2, sampler: { logitBias: bias } };

  const stream = sequence.generate(request);
  // The body does not run until the first `next()`, so before draining is the widest window for
  // changes.
  prompt.push(VOCAB + 5); // add an out-of-vocabulary id after the accepted-set check passed
  request.prompt = [3, 4]; // swap in another array
  request.maxNewTokens = 4; // raise the limit mid-run
  bias.push([6, Number.NEGATIVE_INFINITY]); // ban the token that should come second
  const controller = new AbortController();
  request.signal = controller.signal; // insert an interruption that was not there at the call
  controller.abort();

  const { events, stop } = await drain(stream);
  // What ran is the prompt at the call (length 2 = 2 valid rows + 2 pad rows).
  assertEquals(fake.calls.length, 2);
  assertEquals(fake.calls[0].ids, [1, 2, 0, 0]);
  // Tokens and stop follow the spec at the call (if the late bias took effect, 6 would become
  // another id).
  assertEquals(tokenIds(events), [5, 6]);
  assertEquals(stop, { reason: "max-tokens", tokens: 2 });
});

test("GenerationSequence: a run failure reaches both the iterable and done with the same exception", async () => {
  const fake = fakeSession({ failAt: 1 });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  await assertRejects(() => drain(stream), Error, "run failed");
  await assertRejects(() => stream.done, Error, "run failed");
});

test("GenerationSequence: a failure whose done is never read does not become an unhandled rejection", async () => {
  const fake = fakeSession({ failAt: 0 });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 2 });
  await assertRejects(
    async () => {
      for await (const _event of stream) {
        /* only receive the exception */
      }
    },
    Error,
    "run failed",
  );
  // `done` is not held. If the process does not crash here, it is held once internally.
  await new Promise((resolve) => setTimeout(resolve, 0));
});

test("GenerationSequence: dispose runs after the running generation, and a second call returns the same completion", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next();

  const disposal = sequence.dispose();
  assertEquals(fake.disposals(), 0, "dispose overtook the running generation");
  // Draining the rest frees the slot and dispose runs.
  while (!(await iterator.next()).done) {
    /* drain it */
  }
  await disposal;
  assertEquals(fake.disposals(), 1);
  await sequence.dispose();
  assertEquals(fake.disposals(), 1);
});

test("GenerationSequence: generate after dispose fails synchronously (with its own message)", async () => {
  // The context is not exposed (decision 3), so this check is the caller's only way to learn about
  // the lifetime. Delayed, it would not fail until the first iteration, and with the generic
  // message of `GenerationContext` the real cause could not be read.
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await sequence.dispose();
  assertThrows(
    () => sequence.generate({ prompt: [1], maxNewTokens: 1 }),
    Error,
    "GenerationSequence: cannot generate after dispose",
  );
  assertEquals(fake.calls.length, 0);
  // The same right after the dispose call (before it completes) — once `disposal` is set, no new
  // generation is accepted.
  const running = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const disposal = running.dispose();
  assertThrows(
    () => running.generate({ prompt: [1], maxNewTokens: 1 }),
    Error,
    "GenerationSequence: cannot generate after dispose",
  );
  await disposal;
});

test("GenerationSession: the real Session's surface satisfies the type (detects spelling drift)", () => {
  // A type-check-only check (it does nothing at run time). If the spelling of `run` /
  // `createGenerationContext` / `pastLength` changes on the runtime side, it is a compile error.
  const asGenerationSession = (session: Session): GenerationSession => session;
  assertEquals(typeof asGenerationSession, "function");
});

// ---- the history passed to the sampler (whether the repetition penalty takes effect) -----------
//
// The second argument of `sampler.next(logits, history)` is "the prompt after prepending + the
// tokens already drawn this turn" (the doc of `sampler.ts` — the same as HF applying it to the
// whole `input_ids`). Every test stayed green when this regressed to `[]`, so the repetition
// penalty could quietly remain "a knob with no effect". Conversely, a regression that includes
// **past turns** too (ADR 0066 decision 6's double bookkeeping) was not detected either.
//
// Observations are taken at temperature 0 (argmax). Applying penalty 2 to a first candidate of 10 /
// a runner-up of 6 turns 10 into 5 and flips the ranking **deterministically**, so the expected
// values can be written by hand without betting on a seed-dependent draw.

const PENALTY = { temperature: 0, repetitionPenalty: 2 } as const;
const RUNNER_UP = { id: 6, logit: 6 } as const;

test("GenerationSequence: the repetition penalty applies to the prompt's tokens", async () => {
  // 3, contained in the prompt, is the first candidate (logit 10). If the history arrives, 10/2 = 5
  // < 6 moves it to 6.
  const penalized = fakeSession({ tokens: [3], runnerUp: RUNNER_UP });
  const withPenalty = await createGenerationSequence({
    session: penalized.session,
    program: programOf(penalized),
  });
  const hit = await drain(withPenalty.generate({ prompt: [3], maxNewTokens: 1, sampler: PENALTY }));
  assertEquals(
    tokenIds(hit.events),
    [RUNNER_UP.id],
    "the penalty does not apply to the prompt's tokens",
  );

  // The counterpart (without the penalty the first candidate comes out as is = proof that the flip
  // above comes from the penalty).
  const plain = fakeSession({ tokens: [3], runnerUp: RUNNER_UP });
  const withoutPenalty = await createGenerationSequence({
    session: plain.session,
    program: programOf(plain),
  });
  const control = await drain(
    withoutPenalty.generate({ prompt: [3], maxNewTokens: 1, sampler: { temperature: 0 } }),
  );
  assertEquals(tokenIds(control.events), [3]);
});

test("GenerationSequence: drawn tokens enter the next step's history", async () => {
  // The first token is 5 (not in the history). For the second token the fake's first candidate is 5
  // again, but if the one just drawn entered the history the penalty applies and it moves to the
  // runner-up.
  const fake = fakeSession({ tokens: [5, 5], runnerUp: RUNNER_UP });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const { events } = await drain(
    sequence.generate({ prompt: [1], maxNewTokens: 2, sampler: PENALTY }),
  );
  assertEquals(tokenIds(events), [5, RUNNER_UP.id], "generated tokens do not enter the history");
});

test("GenerationSequence: the history covers one turn (past turns are not dragged along)", async () => {
  // Turn 1 emits 7 → 5 (the stop is max-tokens, so 5 remains as the uncommitted frontier).
  const fake = fakeSession({ tokens: [7, 5, 7], runnerUp: RUNNER_UP });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const first = await drain(sequence.generate({ prompt: [1], maxNewTokens: 2, sampler: PENALTY }));
  assertEquals(tokenIds(first.events), [7, 5]);

  // Turn 2's first candidate is 7 (a token generated in the previous turn). If past turns were
  // dragged into the history, the penalty would apply and move it to 6, so 7 coming out is the
  // observation of "not dragged along".
  const second = await drain(sequence.generate({ prompt: [1], maxNewTokens: 1, sampler: PENALTY }));
  assertEquals(tokenIds(second.events), [7], "the history of a past turn remains");
});

test("GenerationSequence: the prepended uncommitted frontier is penalized as part of that turn's prompt", async () => {
  // The counterpart of the above. The previous turn's last token (5) is **prepended** to the next
  // turn's prompt, so it does enter the history (a different fact from dragging past turns along —
  // ADR 0083 decision 4's uncommitted frontier).
  const fake = fakeSession({ tokens: [7, 5, 5], runnerUp: RUNNER_UP });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await drain(sequence.generate({ prompt: [1], maxNewTokens: 2, sampler: PENALTY }));
  const second = await drain(sequence.generate({ prompt: [1], maxNewTokens: 1, sampler: PENALTY }));
  assertEquals(
    tokenIds(second.events),
    [RUNNER_UP.id],
    "the penalty does not apply to the prepended uncommitted frontier",
  );
});

test("GenerationSequence: a return before starting issues no run and closes done", async () => {
  const fake = fakeSession();
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 2 });
  await stream[Symbol.asyncIterator]().return?.();
  assertEquals(await stream.done, { reason: "closed", tokens: 0 });
  assertEquals(fake.calls, []);
  assertEquals(sequence.used, 0);
  await sequence.dispose();
});

// ---- forking (`fork` — the 2026-10-11 addenda of ADR 0066 / 0083, ruling 8) ------------------
//
// A fork is "an independent sequence that holds the continuation of the same conversation"; what is
// copied is the context (duplicated) and `pendingToken`. The fake context keeps the logical length
// per context (`generation-fake.ts`), so "which context it wrote to" and "whether it prepended the
// frontier" can be read from the call sequence. Handing over the drafter and the gate is covered by
// the fork section of `generation_speculation_test.ts`.

/**
 * Input token → next token (a deterministic target set by the row's input alone — the fork and the
 * original run under the same rule).
 */
const FORK_SUCCESSOR: readonly number[] = Array.from(
  { length: VOCAB },
  (_unused, token) => (token * 5 + 3) % VOCAB,
);

test("GenerationSequence.fork: a fork inherits the frontier and the logical length and continues independently of the original", async () => {
  const fake = fakeSession({ successor: FORK_SUCCESSOR });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const first = await drain(sequence.generate({ prompt: [1, 2], maxNewTokens: 3 }));
  const frontier = tokenIds(first.events).at(-1);
  assert(frontier !== undefined, "premise: turn 1 emitted no token");
  // 2 prefill rows + 2 decodes = logical length 4, one uncommitted frontier = 5 positions occupied
  // by the conversation.
  assertEquals([fake.pastLength(), sequence.used], [4, 5]);

  const forked = await sequence.fork();
  assertEquals(fake.forks.length, 1, "the context was duplicated exactly once");
  assertEquals(forked.capacity, sequence.capacity);
  assertEquals(forked.used, 5, "the fork inherits the positions the conversation occupies as is");

  // The fork's turn: the original frontier is prepended (decision 4), positions continue from the
  // original's logical length, and it runs in the fork's context. The original's logical length
  // does not move.
  const atBranch = fake.calls.length;
  await drain(forked.generate({ prompt: [7], maxNewTokens: 2 }));
  const branchCalls = fake.calls.slice(atBranch);
  assertEquals(
    branchCalls[0].ids.slice(0, 2),
    [frontier, 7],
    "the fork did not prepend the frontier",
  );
  assertEquals(branchCalls[0].positions.slice(0, 2), [4, 5]);
  assert(
    fake.runContexts.slice(atBranch).every((context) => context === fake.forks[0]),
    "the fork's turn ran in a context other than the fork's",
  );
  assertEquals(fake.pastLength(), 4, "the fork's turn moved the original's logical length");

  // The original's turn: it prepends the same frontier on its own side too and continues from
  // position 4 regardless of how far the fork went (an implementation that drops the frontier from
  // the original as "handed to the fork" loses one token here).
  const atMain = fake.calls.length;
  await drain(sequence.generate({ prompt: [9], maxNewTokens: 2 }));
  const mainCalls = fake.calls.slice(atMain);
  assertEquals(
    mainCalls[0].ids.slice(0, 2),
    [frontier, 9],
    "the original did not prepend the frontier",
  );
  assertEquals(mainCalls[0].positions.slice(0, 2), [4, 5]);
  assert(
    fake.runContexts.slice(atMain).every((context) => context === fake.context),
    "the original's turn ran in a context other than the original's",
  );
  assertEquals([sequence.used, forked.used], [8, 8]);
  await forked.dispose();
  await sequence.dispose();
});

test("GenerationSequence.fork: the body runs after the running generate and copies the state at that point", async () => {
  const fake = fakeSession({ tokens: [5, 6, 7, 8] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  const stream = sequence.generate({ prompt: [1, 2], maxNewTokens: 3 });
  const iterator = stream[Symbol.asyncIterator]();
  await iterator.next();

  let settled = false;
  const forking = sequence.fork().then((forked) => {
    settled = true;
    return forked;
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(settled, false, "fork overtook the running generation");
  assertEquals(
    fake.forks.length,
    0,
    "the context was duplicated in the middle of the running generation",
  );
  // Draining the rest frees the slot and fork runs (it copies the settled values after the
  // generation).
  while (!(await iterator.next()).done) {
    /* drain it */
  }
  const forked = await forking;
  assertEquals(forked.used, 5);
  assertEquals(
    forked.used,
    sequence.used,
    "the fork copied a state from the middle of the generation",
  );
  await forked.dispose();
  await sequence.dispose();
});

test("GenerationSequence.fork: after dispose it fails synchronously, and a fork issued before dispose runs first", async () => {
  const fake = fakeSession({ tokens: [5, 6] });
  const sequence = await createGenerationSequence({
    session: fake.session,
    program: programOf(fake),
  });
  await drain(sequence.generate({ prompt: [1], maxNewTokens: 2 }));

  // Issued in the order fork → dispose, fork runs first since it is the same chain. After the
  // dispose call it fails synchronously.
  const forking = sequence.fork();
  const disposal = sequence.dispose();
  assertThrows(() => sequence.fork(), Error, "GenerationSequence: cannot fork after dispose");
  const forked = await forking;
  await disposal;
  assertEquals(fake.disposals(), 1, "the original context was not torn down");
  assertEquals(fake.forks.length, 1, "a rejected fork produced a duplicate");

  // The original may be disposed first (there is no parent–child dependency) — the fork stays
  // usable and tears down only its own context.
  const branch = await drain(forked.generate({ prompt: [3], maxNewTokens: 1 }));
  assertEquals(branch.stop.reason, "max-tokens");
  assert(
    fake.runContexts.at(-1) === fake.forks[0],
    "the fork's turn did not run in the fork's context",
  );
  await forked.dispose();
  assertEquals(fake.disposalsOf(fake.forks[0]), 1);
  assertEquals(fake.disposals(), 1, "the fork's dispose tore down the original context");
});

test("ForkableContextFace: the real GenerationContext has a fork that returns its own type (a type check)", () => {
  // A type-check-only check (it does nothing at run time). If the spelling of `fork` or its return
  // type changes on the runtime side, it is a compile error (the F-bound of
  // `createGenerationSequence` no longer resolves with a real Session).
  const asForkable = (context: GenerationContext): ForkableContextFace<GenerationContext> =>
    context;
  assertEquals(typeof asForkable, "function");
});

// ---- The distribution's suppressed set (`GenerationWiring.suppressTokens`) ----------------
//
// The set rides on the wiring, not on the request, so a sequence built from a program that
// declares it masks every selection without the caller passing anything. The fake puts the
// suppressed id at each row's raw maximum (logit 10) with a runner-up behind it (6): a path that
// skips the mask picks the suppressed id.

describe("GenerationSequence: the wiring's suppressed set reaches every selection", () => {
  const SUPPRESSED = 3;
  const BEHIND = { id: 6, logit: 6 } as const;
  const STEPS = 12;
  /** Every run's raw argmax is the suppressed id (a two-chunk prefill + the decode steps). */
  const session = () =>
    fakeSession({ tokens: Array(STEPS + 1).fill(SUPPRESSED), runnerUp: BEHIND });

  const run = async (suppressTokens: readonly number[] | undefined, request: GenerationRequest) => {
    const fake = session();
    const sequence = await createGenerationSequence({
      session: fake.session,
      program: programOf(fake, suppressTokens === undefined ? {} : { suppressTokens }),
    });
    try {
      return tokenIds((await drain(sequence.generate(request))).events);
    } finally {
      await sequence.dispose();
    }
  };

  it("greedy (prefill's first token and every decode step) takes the masked maximum", async () => {
    const request = { prompt: [1, 2, 4, 5, 7], maxNewTokens: STEPS };
    assertEquals(await run(undefined, request), Array(STEPS).fill(SUPPRESSED), "control");
    assertEquals(await run([SUPPRESSED], request), Array(STEPS).fill(BEHIND.id));
  });

  it("sampling never draws the suppressed id (a seeded draw that would draw it unmasked)", async () => {
    // Unmasked, id 3 carries e^10 / (e^10 + e^6 + 14) ≈ 98% of each draw.
    const request = { prompt: [1], maxNewTokens: STEPS, sampler: { temperature: 1, seed: 5 } };
    const control = await run(undefined, request);
    assert(control.includes(SUPPRESSED), `control drew no ${SUPPRESSED}: ${control.join(",")}`);
    const masked = await run([SUPPRESSED], request);
    assertEquals(masked.length, STEPS);
    assert(!masked.includes(SUPPRESSED), `a suppressed id was drawn: ${masked.join(",")}`);
  });

  it("an empty set gives the same sequence as no declaration (other distributions unchanged)", async () => {
    for (const sampler of [undefined, { temperature: 1, seed: 9 }]) {
      const request = { prompt: [1, 2], maxNewTokens: STEPS, ...(sampler ? { sampler } : {}) };
      assertEquals(await run([], request), await run(undefined, request));
    }
  });
});
