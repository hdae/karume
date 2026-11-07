# 0083: Generation API surface — `GenerationProgram` / `GenerationSequence`, token events, host-side sampling

- Status: implemented (accepted 2026-08-31 — the maintainer confirmed the recommended option on all
  ten design points; implemented in stages 0 to 3, stage 3 = `6b536fe1`); partly superseded:
  branching in decision 4 by `fork()` (2026-10-11 addendum), decision 10 revised for the high-level
  surface (2026-09-02 addendum), and two Consequences items marked superseded on 2026-10-18 (distribution
  scope by ADR 0097, host-generated RoPE by ADR 0091).
- Date: 2026-08-31
- Scope: `packages/models/src/generation/` (`program.ts` / `sequence.ts` / `sampler.ts`, and
  `greedy.ts` demoted) and the gemma4 product graph script. IR, runtime and hub unchanged.
- Related: ADR [0066](0066-generation-context-state-slots.md) (`GenerationContext`: decision 6 no
  double bookkeeping, decision 8 scope, addendum 2 `rewind` rejection) /
  [0068](0068-decode-exit-multi-output.md) (decode exit — the opt-in all-vocabulary logits of its
  decision 4 become the product default, recorded in its addendum 6) /
  [0067](0067-autoregressive-attention-vocabulary.md) / [0008](0008-public-api.md) /
  [0078](0078-anima-sampler-selection.md) (distribution default, request override) /
  [0079](0079-sbv2-two-layer-input.md) / [0082](0082-linear-gemv-decode.md) (decode 32.5 ms/token) /
  siblings [0084](0084-gemma-tokenizer-chat.md) (tokenizer, chat), [0085](0085-ple-host-gather.md) (PLE).
- Basis: [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  ("draft §n" below — candidate comparison, rejection reasons and asset measurements live there).

## Context

The only generation surface for consumers was `generateGreedy`
(`packages/models/src/generation/greedy.ts`), which lacks what LLM use (streaming chat) needs:

- **Static wiring and the request share one type** (`GreedySpec`: `session`, `inputIds`,
  `positionIds`, `token`, `lastRow`, `chunkLength`, `maxPosition`, `bindings` are static; `prompt`,
  `maxNewTokens` are the request).
- **No stop condition** ("no EOS stop") and **no sampling on the host either** ("no temperature, top-k
  or RNG" MUST — the function exists for acceptance with fixed token-id sequences).
- **No multi-turn surface** (a context is created every time and disposed in `finally`).

The groundwork was in place: the runtime already rejects concurrent runs on one context with
`ExecutionError`, the wiring that selects the last row through the `last_row` index exists (ADR 0068
addendum 4), and decode is down to 32.5 ms/token (ADR 0082).

This ADR fixes the **API surface and contracts** that hand "string in → string out" to consumers.
Tokenizer, detokenizer and chat belong to ADR 0084; the PLE distribution to ADR 0085.

## Decision

### 1. Two layers — `GenerationProgram` (static wiring) and `GenerationSequence` (one conversation)

`GenerationProgram` is an **immutable object whose wiring is fully checked at setup**;
`GenerationSequence` owns the lifetime of one conversation. Types were provisional (fixed during
implementation); the contract has this shape:

```ts
type GenerationProgram = {
  readonly inputIds: string;
  readonly positionIds: string;
  readonly lastRow: string;
  readonly logits: string; // last-row logits exit (decision 6)
  readonly chunkLength: number;
  readonly maxPosition: number;
  readonly vocabSize: number;
  readonly stopTokens: readonly number[]; // EOS set (decision 8)
  readonly bindings?: SymbolBindings;
};

type GenerationRequest = {
  readonly prompt: readonly number[]; // in multi-turn use, only this turn's tokens
  readonly maxNewTokens: number;
  readonly sampler?: SamplerSpec; // defaults to the distribution's declaration (decision 7)
  readonly signal?: AbortSignal;
};
```

MUST: **the program holds no mutable state**, and **the sequence's mutable state is only `context`
and `pendingToken`**. The sequence keeps no position or totalLength counter: it reads
`context.pastLength` right before building a run to make `position_ids` (ADR 0066 decision 6, **no
double bookkeeping**; the logical length advances when a run succeeds). If `totalLength` is needed it
is derived as `context.pastLength + (pendingToken ? 1 : 0)` and **not stored**.
NOTE: speculative decoding added a copy of the drafter's input hidden state (one row, 2026-09-08) and
`fork()` copies the speculation gate's observed state (2026-10-11); neither records progress, so the
no-double-bookkeeping rule still holds, but "only two" no longer lists every field.

### 2. Tokens are emitted as `AsyncIterable<GenerationEvent>`

```ts
type GenerationEvent =
  | { readonly kind: "token"; readonly id: number; readonly position: number }
  | { readonly kind: "prefill"; readonly chunk: number; readonly chunks: number };

interface GenerationSequence {
  generate(request: GenerationRequest): AsyncIterable<GenerationEvent> & {
    /** Stop reason (`eos` / `max-tokens` / `aborted`) — read after draining the iterable. */
    readonly done: Promise<GenerationStop>;
  };
  dispose(): Promise<void>;
}
```

The house style is anima's `onEvent` callback (`packages/models/src/anima/pipeline.ts`), but its
reason reverses here: anima returns one image and the events are side progress, while for an LLM
**the sequence is the return value**. With `for await`, the detokenizer's carry-over of partial UTF-8
(ADR 0084 decision 4) maps directly onto "push and only finished text comes out", and `break` /
`return()` closes naturally.

MUST (the `break` hole): an interrupted iterable enters `finally` through `return()`, and by then **the
logical length has already advanced by the successful runs**. If `pendingToken` is not kept
correctly, the next turn loses one token. Having chosen `for await`, a check on this path is
mandatory (the pass line of stage 3).

MUST: the sequence serializes each `generate` with `createOperationChain`
(`packages/models/src/concurrency/serial.ts`), as the other families do. **No home-made lock** (the
runtime's single-flight already rejects a second issue).

### 3. `GenerationContext` stays inside the sequence

The product surface does not expose a raw `GenerationContext`; the uncommitted frontier of at most one
token (decision 4) becomes invisible to consumers. The acceptance paths (`generateGreedy`, e2e) keep
using the context directly as internal helpers, so **the public surface is not duplicated**.

### 4. Multi-turn prefill prepends `pendingToken`

A `GenerationContext` always holds **an uncommitted frontier of at most one token** (after K generated
tokens `pastLength = T + K − 1`, because `generateGreedy` decodes only `maxNewTokens - 1` times;
confirmed by review CG5-1). The next turn therefore **prefills with `pendingToken` prepended to the
new user turn's tokens**. The frontier enters the KV without an extra commit-only run; this is also
what makes decision 3 possible.

MUST: `rewind` is not used — contexts with sliding slots are **always rejected** (ADR 0066 addendum
2). Editing and branching were "a new context + token transcript replay".
(Superseded for branching on 2026-10-11: `GenerationSequence.fork()` / `Gemma4ChatSession.fork()`,
see the addendum of that date and ADR 0066's addendum of the same day. Editing an earlier utterance
and changing the capacity still replay into a new context.)

### 5. Cancellation uses `AbortSignal` (a throw from `onEvent` is not a cancel mechanism)

`GenerationRequest.signal` is checked at stage boundaries, and `signal.reason` is **thrown as is,
unwrapped**. The precedent is `AnimaPipelineOptions.signal` on the build path. The deferred backlog
item "AbortSignal for the generation loop (waiting for demand)" had its demand in LLMs, so this ADR
opens it.

### 6. The exit is the last-row logits; sampling stays on the host (a topk product graph is rejected)

The product graph's exit is `logits[1,1,V]` (**last row only**). The wiring "select `last_row` → one-row
lm_head" already exists in `TokenOnlyChunkWrapper` (`tools/export-recipes/gemma4/export_token.py`);
**only the argmax is removed**. ADR 0068 decision 4 keeps the all-vocabulary logits readback "as an
opt-in when the graph declares logits as an output (the field is not removed)", and what it rejected
was **GPU-side full sampling**. So this decision **needs no new ruling**, and ADR 0068 records it in
addendum 6.

**Why a topk product graph is not adopted** (record of rejection):

1. **`k ≤ 63` is one short of the model's recommended default.** The limit `8·W·(k+1) ≤
   maxComputeWorkgroupStorageSize` (ADR 0068 addendum 2; W=32, WebGPU default 16384 B) gives **k ≤ 63**
   (`TOPK_CORE_LIMIT_MAX_K`, `packages/runtime/src/kernels/topk.ts`); gemma-4-E2B-it recommends
   **`top_k: 64`** (draft §0.2-8).
2. **The exporter stops at `operator.getitem`, not at a handler** (ADR 0068 addendum 3): adding an
   aten handler does not open the path.
3. **Repetition penalty, logit bias and full-vocabulary nucleus need the whole vocabulary** (the host
   does not know the softmax mass outside the top 63). Chat cannot reasonably go without them.
4. **The cost is negligible.** The 1 MiB readback rides the same submit (no extra fence); with the JS
   scan over 262,144 elements it is **estimated at 0.3–0.6 ms**, a few percent of 32.5 ms/token (ADR
   0082). **Prefill readback shrinks**: `[1,M,V]` is 32 MiB at chunk 32; the last row is **1 MiB**.

MUST: **sampling and RNG stay on the host** (ADR 0068 decision 4 reconfirmed — the GPU side does only
argmax / topk).

### 7. Sampler — location, RNG contract, owner of the defaults

- **Location**: `packages/models/src/generation/sampler.ts` (pipeline-independent shared code, like
  `greedy.ts`). First scope: temperature, top-k (**no upper limit** — decision 6's exit), top-p,
  repetition penalty, logit bias, seed.
- **RNG**: reuses the splitmix64 approach of `packages/models/src/anima/random.ts` (now
  `packages/models/src/generation/random.ts`). MUST (as in that precedent, whose doc says "not
  torch's `randn`; the same seed gives a different sequence"): **token-sequence parity with HF's
  sampling output is not attainable**. Parity checks run at temperature 0 (= greedy); the sampler is
  bound by its own fixtures and distribution checks.
- **Defaults are declared by the distribution**: the recommendation of `generation_config.json`
  (gemma-4-E2B-it: temperature 1.0 / top_k 64 / top_p 0.95) is written by dist into `pipelineConfig`
  and used when the request omits `sampler`. The precedent is anima's `scheduler.type` (ADR 0078 —
  the distribution declares the default, `AnimaGenerateRequest.sampler` overrides it). **Because the
  exit is the last-row logits, `top_k: 64` can be expressed as is** (the flip side of decision 6).
  NOTE: the lower layer (`sampler.ts` alone, `generateGreedy`) defaults to greedy — "no sampler
  declared = the temperature-0 degenerate form". The parity checks live on through that path.

### 8. Stop tokens are a set (declared by the model)

The stop condition is a **set**, not a single EOS: gemma-4-E2B-it's `generation_config.json` has
`eos_token_id = [1, 106, 50]` (`<eos>` / `<turn|>` / `<|tool_response>`). It is received as
`GenerationProgram.stopTokens` and checked **against the sampling result** (the same stage as the
sampler).

MUST: the EOS set and the chat format **come from the same distribution digest set** (ADR 0084
decision 5) — taken from different places, one of them goes stale.

### 9. `generateGreedy` is demoted to an internal helper (breaking)

Its export leaves the barrel (`packages/models/mod.ts`); it becomes an internal helper for parity
acceptance. "Keep it public for now and remove it in the next breaking release" is not adopted. The
surface was already published, so this is a **breaking change** recorded in `docs/limitations.md` and
the release notes. **Only the public export disappears**; the parity checks against existing golden
token sequences remain.

### 10. Capacity overflow fails with a dedicated type

(Revised on 2026-09-02 for the high-level surface: `Gemma4ChatSession` has an overflow policy, see
that addendum.)

Exceeding `P + Q ≤ C` on a full slot "failed loudly with a generic message". The sequence fails with a
dedicated type (`GenerationCapacityError`), and **"truncating the conversation is the host's job"**
is stated in `docs/limitations.md` (LLM lens L-9 b).

## Alternatives considered

- **anima-style `onEvent` callback** (house style; no abandoned async generator can hold the
  serialization chain): rejected per decision 2 — tokens are the value, and streaming consumers would
  build their own queues. With `AbortSignal` (decision 5), throwing to interrupt is not needed.
- **Extending one function (`generateStream(spec)`, no sequence type)**: no new types, but the
  `GenerationContext` is exposed and `pendingToken` handling leaks to callers, who would reproduce the
  "last token of the previous assistant turn drops from history" accident. Rejected.
- **topk product graph (argmax → topk, fixed k)**: a 512-byte readback, but see decision 6 points 1
  to 3. **GPU-side full sampling / nucleus** was rejected outright by ADR 0068 (adopting it would need
  a new ruling).

## Consequences

- Implementation ran as the six-stage plan in the backlog. Stage 0 (contracts) was this ADR + ADRs
  0084 / 0085 + ADR 0068 addendum 6; stage 1b put the PLE split-out (ADR 0085) and the last-row logits
  exit into one re-export, keeping the two older series (logits opt-in / token-only) as acceptance
  fixtures (details: draft §9).
- **One breaking change**: the public export of `generateGreedy` is removed (decision 9).
- Distribution covered gemma4 E2B only; distributing through minicpm5 while the licence gate was
  pending was not adopted, and publication waited for the licence review per upstream revision (ADR
  [0065](0065-exporter-core-recipe-split.md) decision 7). (Superseded by ADR 0097 on 2026-09-11:
  QAT E2B / E4B joined, and the gemma4 recipe ships `e4b` — `tools/export-recipes/gemma4/distribution.py`;
  `hdae/karume-gemma4` is published.)
- **Four items not adopted here, each needing a new ruling if adopted** (draft §9):
  1. Using `rewind` — contexts with sliding slots are always rejected (ADR 0066 addendum 2); relaxing
     it is ruled together with compaction.
  2. Continuous batching / batch > 1 / multiple sequences — out of scope per ADR 0066 decision 8.
  3. GPU-side full sampling / nucleus — rejected by ADR 0068 (decision 6's MUST keeps it so).
  4. Host-generated RoPE tables (LLM lens L-9 c) — conflicted with ADR 0067 decision 4 ("RoPE is
     outside the attention op; the exporter bakes it into the graph"). (Superseded by ADR
     [0091](0091-gemma4-host-rope-variable-capacity.md) on 2026-09-03: the host supplies RoPE per
     chunk; 0091 is the new ruling.)
- Out of scope: decode performance itself (done in ADR 0082; lenses L-7 / L-12 in perf-ledger), an
  f16 KV option, exporter wiring for topk (no demand while decision 6 holds).

## Addendum (2026-08-31) — public-surface review: read-outs added, write paths closed

Consumers who ran the surface had guessed three numbers it did not expose, and one write path existed.
The design of decisions 1–10 is unchanged; **read-outs are added and write paths closed**.

- **`GenerationStop.tokens`**: the number of tokens generated in the turn. MUST: an `eos` stop token
  counts as one — one draw = one run, so the number equals the runs spent and gives tok/s **without
  re-encoding** (the field's purpose). Subtract one for `reason === "eos"` to count body tokens only.
- **`GenerationSequence.used`**: logical positions the conversation already occupies. MUST: a
  **derived value** (`context.pastLength` + the uncommitted frontier — ADR 0066 decision 6), not an
  independent counter. Whether the next turn fits is decidable in advance:
  `used + prompt.length + maxNewTokens - 1 ≤ capacity` and
  `used + prompt.length + maxNewTokens - 2 < maxPosition`.
- **Structured fields on `GenerationCapacityError`**: `constraint` (`capacity` or `maxPosition`),
  `pastLength`, `promptLength` (**after** prepending `pendingToken` — may be one more than the
  caller's `prompt`), `requestedNewTokens`, `limit`, `maxNewTokens` (the largest value that fits now;
  negative = by how many tokens it overflows). The dedicated type exists to carry **the numbers
  truncation needs**, not a message.
- **The public `GenerationProgram` is read-only** (enforcing decision 1): the loop's fields moved to the
  internal `GenerationWiring`; `Gemma4Pipeline.program` exposes only what running `sequence()` needs
  (`chunkLength`, `capacity`, `maxPosition`, `vocabSize`, `stopTokens`). Graph I/O names, bindings and
  `derivedInputs` were useless to consumers and writable, and **being verified is what
  `GenerationProgram` means**. The surface is frozen and `stopTokens` is a **frozen copy** (a shared
  array would let a consumer's `sort()` silently change the stop set).
- Breaking (unreleased surface; `docs/limitations.md`): new `GenerationStop` field, two-argument
  `GenerationCapacityError` constructor, wiring fields gone from `GenerationProgram`, `generate` after
  `dispose()` throws synchronously.

## Addendum (2026-09-02) — requests are copied at issue

The immutability of decision 1 extends **to requests**: two ways to rewrite an object the caller still
holds are closed.

- **`SamplerSpec.logitBias` is an array of tuples** (`readonly [token, bias][]`), not a `Map`. The
  spec crosses JSON (distribution declaration, host settings, logs) and worker boundaries, and
  `JSON.stringify` turns a `Map` into `{}` — a round-tripped spec silently has **no bias**. MUST: the
  same token twice fails loudly (a `Map` kept the last write; for an additive bias that means only one
  of two written biases applies, and the caller cannot tell add from overwrite). Breaking (unreleased).
- **Requests are copied at issue** (`GenerationSequence.generate` / `Gemma4Pipeline.chat`): the
  `prompt` copy, `maxNewTokens` / `signal` and a sampler snapshot (down to the `logitBias` tuples) are
  taken **in the synchronous part of the issue** — an async generator body runs only at the first
  `next()`, so the checked and the used values could otherwise differ, silently. One copy per layer:
  `generate` copies the prompt; the sampler spec copy is the single `snapshotSpec` in `sampler.ts`,
  called at issue (`Gemma4Pipeline.chat` / `Gemma4ChatSession.send`) and by `createSampler`
  (2026-09-05). Observable change: rewriting after issue has no effect (`docs/limitations.md`).

## Addendum (2026-09-02) — per-request stop conditions and reading the whole text

Decision 8's set held only the distribution's EOS; there was no way to change how a turn stops
(consumers would `break` while detokenizing and re-implement stop strings across tokens and the
uncommitted frontier). The design is unchanged; **stop conditions open to requests in two layers**.

- **Stop tokens at the sequence layer** (`GenerationRequest.stopTokens`): checked as the **union**
  with the distribution's set (its EOS always applies). `GenerationStop` gains the branch
  `stop-token`; an id in both sets closes as `eos` (the terminator meaning wins). Out-of-vocabulary or
  duplicate ids fail loudly **synchronously**: a stop token never appears in the output, so a wrong
  one only shows as output that never stops. Copied at issue.
- **Stop strings at the chat layer** (`Gemma4ChatOptions.stopStrings`): matching sits **after**
  decoding (a stop string can span tokens). Implemented by `createStopStringFilter` in
  `src/text/detokenizer.ts`, a state machine **separate** from the `byte_fallback` carry-over.
- **Holdback**: only **a finished tail that is a prefix of a stop string** is held back (a fixed
  holdback would delay every turn); it is released once it leaves the prefix (stop `"END"`: `"E"`,
  `"N"`, `"X"` releases `"ENX"` at the third piece). On a match, text before the stop string is
  released, the rest is not; a turn ending without a match releases the held text (nothing dropped).
- **Early termination**: a stop-string match leaves by `return`, the same clean-up as `break` (the
  last token stays as `pendingToken`). The chat layer reports `stop-string` (`Gemma4ChatStop`), and
  `tokens` carries the inner `done` count unchanged (no recount — no double bookkeeping).
- **Whole text, `Gemma4ChatStream.text()`**: drains and returns the concatenation. **A stream is
  consumed one way only** (a second consumption throws **synchronously**; it would silently get only
  the rest). It sits on the stream, next to the stop reason (`done`).
- Breaking (unreleased; `docs/limitations.md`): `stop-token` in the `GenerationStop` union;
  `Gemma4ChatStream.done` typed `Gemma4ChatStop`.

## Addendum (2026-09-02) — the middle layer `Gemma4ChatSession`; decision 10 revised

Consumers wanted "an ordinary stateful chat": `chat()` recomputes the whole history each time, and
making it efficient drops them into token ids, `<bos>` and the uncommitted frontier. The low-level
surface (decisions 1–5) and `chat()` are unchanged; **a layer covering that gap** is added
(`packages/models/src/gemma/chat-session.ts`). It is built only from public surfaces — KV continuation
of `sequence()`, `gemma4ChatPrompt` / `gemma4ChatTurn`, `GenerationSequence.used`, the structured
fields of `GenerationCapacityError`, `chat()`'s stop conditions and `Gemma4ChatStream` — so consumers
could write it themselves. (The Deno-era example that held this code was removed with the CLI
examples, ADR 0127 decision 5.)

- **Decision 10 revised**: "truncation is the host's job" **still holds for the low-level surface**
  (`sequence()` / `chat()` throw `GenerationCapacityError`). `Gemma4ChatSession` has an **injectable
  default policy** (`Gemma4ChatSessionOptions.onOverflow`); the default `dropOldestTurns` **drops the
  oldest user / assistant pair** (system utterances stay; dropping half a pair leaves "only an answer"
  or "only a question"). A host passing a throwing function gets the old semantics. Not truncating by
  default would make every consumer re-implement something chat UIs **always** need; that the policy
  must be replaceable stays as decision 10 said.
- **Checked in advance** (no waiting for the exception): right after building each turn's prompt,
  `used + prompt + maxNewTokens - 1` is compared with the limit (the smaller of `capacity` and
  `maxPosition`), and `onOverflow` runs before sending if it is exceeded; `used` was exposed for this.
  A `GenerationCapacityError` raised mid-run is **not caught as a safety net** — with a correct
  pre-check it is unreachable, and catching and retrying would hide a drift in the pre-check.
- **Convergence**: retries continue only **while the history shrinks** (a policy returning a history
  that did not shrink gets `GenerationCapacityError`). The history length is a non-negative integer,
  so attempts cannot exceed the number of utterances; a broken invariant is named by the check in
  `#prepare` (no retry-count constant).
- **When the KV is reused**: only after a turn **closed by the distribution's EOS** does the next turn
  continue by difference (`gemma4ChatTurn`). After max-tokens, a stop string, an abort or a failure,
  the model turn is not closed, so the sequence is discarded and redrawn from history (decision 4's
  turn-local contract as a check). A turn adds to history **only the body actually delivered**; a turn
  that produced no text removes its utterance (no unanswered questions).
- **One session = one generation**: a second `send` throws **synchronously** (no queue as in
  `chat()`); two sends on one history would put an unanswered question into the KV. The cost: a stream
  issued but never drained leaves the session in that turn.
- **Entry is the constructor** (`new Gemma4ChatSession(pipeline, options)`): a
  `Gemma4Pipeline.createChatSession()` would make the imports circular. The session reads only
  `Gemma4ChatSessionHost` (tokenizer, program, defaultSampler, `sequence()`), so the type shows it uses
  nothing inside the public surface, and the GPU-free checks
  (`packages/models/tests/gemma4_chat_session_test.ts`) build on it.
- Compaction by model summary was left out while the window was 640; ADR
  [0091](0091-gemma4-host-rope-variable-capacity.md) decision 4 made `capacity` a run-time option
  (default 4,096, limit `maxPosition`, 131,072 for E2B) on 2026-09-03, and compaction is in the
  backlog as one `onOverflow` policy.

## Addendum (2026-09-08) — decision 6's exit generalized to "R selected rows" (ADR 0096 decision 5, MTP stage 1)

- Of decision 6 ("last-row logits `[1,1,V]`, sampling on the host"), only **the exit shape** widened
  to `logits[1,R,V]` + hidden `[1,R,H]` (`last_row[R]`) (ADR 0068 addendum 7). Sampling and RNG stay
  on the host, and GPU-side full sampling stays rejected (decision 6). `GenerationProgramSpec` takes
  `hidden` (required) and the wiring derives `hiddenSize` from the graph. R=1 prefill / decode are
  unchanged; `readLogits` returns row views (`{ rows, row(i) }`).
- Decisions 1 / 4 (mutable state `context` and `pendingToken`; frontier at most one token) are
  unchanged in stage 1 and generalized in stage 3 (ADR 0096).

## Addendum (2026-09-08) — dependency injection for the speculative loop and per-run observation (ADR 0096 stage 3)

- Decisions 1 / 4 generalized while **"the frontier is still one token"**: speculative verify
  (deferred run) delivers the accepted sequence `[d₁..d_a, b']` one by one, updates `pendingToken`
  before each yield (MUST), and `commit(rows)` covers **up to the delivered frontier** (ADR 0096
  addendum, stage 3). Mutable state is `context` / `pendingToken` plus a non-derivable **copy of the
  drafter's input hidden state (one row)**; progress is still recorded only in `context.pastLength`.
- Decision 3 is unchanged. The drafter is injected as `GenerationSequenceOptions.speculative =
  { open(context) → DraftFace, k }`; the borrowing context opens inside the sequence and closes first.
  `DraftFace` (`draft` / `dispose` / `steps`) knows neither Session nor context, so a GPU-free fake
  covers the edge cases without a test-only public entry.
- Observation moved to a **per-run hook** `GenerationSequenceOptions.onRun(phase)`
  (`GenerationRunPhase` = prefill / decode / draft / verify, numbered from 1; verify fires in the
  synchronous section right after commit). The sequence still knows neither Session nor diagnostics;
  the pipeline reads `Session.diagnostics()` inside the hook. `withRunDiagnostics`, which derived run
  counts from events, could not work with speculation (one run = 0 to k+1 tokens) and was replaced.
  The public `Gemma4RunPhase` became an alias of `GenerationRunPhase` with four branches instead of
  two (breaking, unreleased).
- Decision 7 (sampling on the host): speculation runs at any temperature. Acceptance calls
  `sampler.next` once per row with the same logits, history and order as non-speculative decoding, so
  the RNG stream and the tokens match it exactly (at temperature > 0 this equals speculative sampling
  with a one-hot draft).

## Addendum (2026-09-09) — one observation carries the run's wall time and the gate state (MTP 4-B ⑨)

- The `decode` / `verify` branches of `GenerationRunPhase` gain optional `wallMs` / `delivered` /
  `gate` (`GenerationGateTrace` = steady mode, switches within the turn, whether the gate observed
  this run). No new branches (exhaustive switches in consumers keep working).
- `wallMs` is **the variable the gate decides on** (to right after the acceptance check, before the
  delivering yield; non-speculative decode over the same span). It excludes the consumer's speed, so
  slow consumers do not turn speculation off. `gate` is the value **after** the observation.
- The use was the time breakdown of the MTP bench (`tools/mtp-bench`, removed with the Deno-era tools
  in `bb35fe2a`); the pipeline passes the phase through.

## Addendum (2026-09-10) — sorting for top-p without top-k

Nucleus sampling without `topK` uses the cumulative probability over the whole vocabulary. A stable
radix sort on f32 logit bits mapped to descending keys replaces the O(V log V) comparison sort with
O(V). Signed zeros are equal in the key, and ties keep ascending token-id order, so ranks of equal
values, the softmax addition order and the set kept at the boundary are preserved. RNG, temperature
handling and the bounded heap used with top-k are unchanged; work arrays stay inside the call.

A step-wise widening heap was not adopted (more conditions to match the remaining mass and f64
order). Exact full sorting made CPU sampling on real Gemma logits 7.6–7.8× faster. Decision 6's
0.3–0.6 ms scan estimate does not apply to the comparison-sort cost of top-p alone. The distributed
default includes top-k 64, so default generation is not improved by this. Measurements:
[research 2026-09-10](../research/2026-09-10-codex-mtp-optimization.md) (shared sampler, top-p alone).

## Addendum (2026-09-12) — small output for Gemma's temperature-0 decode

- In the shared pipeline of Gemma4 and Gemma4-QAT, only temperature-0, non-speculative decode on
  unprocessed logits is connected as target enqueue → resident logits → a separate topk(k=1) Session →
  finishAndRead. Prefill uses the normal run, keeping the path to the first token; building the
  backing from the first prefill was rejected (Chrome E4B's first turn got slower). The stored graph's
  logits + hidden exit is unchanged; no new CLI argument or model asset. General sampling,
  penalty / bias and speculation use the normal run.
- The generation loop takes the small output as an internal capability; no adapter inflates it to
  fake all-vocabulary logits. The sampler returns, from the spec fixed at issue, whether the choice
  is decided by the maximum alone. At temperature 0, topK / topP / seed do not affect the choice
  (existing contract); every invalid spec is still checked.
- The first NaN, or the maximum with its smallest index, returned by topk is checked. NaN and a
  maximum of ±Infinity are rejected as before, not folded into a valid token. Before a selection
  failure, the written frontier is removed from `pendingToken`. Intermediate prefill chunks are not
  sampled, so their non-finite values are not confused with the final chunk's selection.
- Pipelines with a diagnostics callback / gpuTiming keep the normal run (`lastRun` preserved). Events,
  EOS, `break`, AbortSignal and multiple turns use the same loop.
- The helper lives as long as the pipeline; normal runs and whole batches share one operation chain
  (no run overtakes a batch). GPU resources are allocated at the first greedy use; a failed preparation
  returns partial allocations and retries at the next issue. Release order: sequence → selector and
  resident buffers → drafter → target → GPU → PLE.
- The extra resident and selector memory goes into `auxiliaryBytes` and, conservatively, into
  `peakAccountedBytes` (ADR [0070](0070-shard-loading-admission.md), 2026-09-12 auxiliary-resource
  section); the slight overestimate (the estimator's input borrows the resident buffer) is kept rather
  than dropped to make the number smaller.
- Accepted on CPU (path selection, history, NaN, abort), GPU (mixed runs, partial allocation, release)
  and the real pipeline (normal / QAT E2B / E4B on Deno and Chrome, equal tokens, ABBA speed). M2 needs
  a re-run; the CLI default with temperature is not reported as faster.

Measurements and acceptance: [research 2026-09-10](../research/2026-09-10-codex-mtp-optimization.md)
(Gemma temperature-0 decode section).

## Addendum (2026-10-11) — branching a conversation (fork)

Decision 4's "branching = new context + replay" is replaced by a full copy at the same capacity. The
copy contract (range, synchronous section, rejections, determinism) lives in ADR
[0066](0066-generation-context-state-slots.md)'s addendum of the same day; this records only the two
layers above it (sequence and chat session).

- **`GenerationSequence.fork(): Promise<GenerationSequence>`** returns an independent sequence that
  continues the same conversation. It copies the context (the runtime's fork), `pendingToken` (the
  uncommitted frontier) and the speculation gate's observed state. The drafter face is reopened on the
  copied context (the borrower goes with the context; the original borrower is not shared). The RNG is
  not copied (each `generate` builds it from the seed).
  - It runs on the sequence's serialization chain, after an in-flight `generate`; on a disposed
    sequence it throws synchronously. If reopening the drafter fails, the copy is closed before
    rethrowing. The caller owns and disposes the branch.
- **The speculation gate copies its observed state** (`SpeculationGate.fork()`, not public); the two
  gates are independent afterwards.
- **`Gemma4ChatSession.fork(): Promise<Gemma4ChatSession>`** returns an independent session with the
  same settings and history. It throws synchronously when disposed or busy with a turn (as `send`
  does). Copying the history and "how many entries are in the KV", and calling `sequence.fork()`, all
  happen in the synchronous section of `fork()` (so a following `send` cannot push history first). A
  conversation with an empty KV (last turn not closed by EOS, or nothing sent yet) copies only the
  history, with no GPU copy.
- **Regeneration is written as a use of fork** (no `regenerate`): keep the base session as is, and on
  each press `fork()` and `send`. KV roughly doubles (base + one branch), the same as rebuilding by
  replay; only the public surface differs.
  - **The same seed gives the same answer**: the draw's random stream is rebuilt from the sampler
    `seed` (default 0) on every `generate`, and the copied KV is bit-identical, so regenerating without
    changing the seed returns the same tokens. To draw a different answer, change the seed per attempt
    (for example `send(text, { sampler: { ...spec, seed: attempt } })`, as in the chat session's `fork`
    doc). Temperature 0 gives the same answer regardless of seed.
- Public surface: one method each on existing types (`GenerationSequence` / `Gemma4ChatSession`), no
  new export (ADR 0008).

## Addendum (2026-11-05) — a turn without `maxNewTokens`

A caller often has no number to give: it wants the model to finish, and only the conversation's room
bounds that. Passing `capacity − used − prompt + 1` by hand means every caller re-implements the
budget formula, with the uncommitted frontier in it. So `maxNewTokens` becomes optional on
`GenerationRequest`, `ChatOptions`, `ChatSessionOptions` and `ChatTurnOptions`: absent means until a
stop token or the limit, like llama.cpp's `-n -1` with context shift off.

- **Decision 10 still holds: the budget fails before a single run.** The limit is resolved once, right
  before `assertBudget`, after the pending token is prepended:
  `K = min(capacity, maxPosition) − past − prompt + 1`, the largest request the check accepts (both
  constraints fold into that shape). The invariant is not relaxed. Below 1 the prompt alone does not
  fit, and `assertBudget` runs with 1, so the same `GenerationCapacityError` is thrown
  (`requestedNewTokens` is then 1). `K` goes wherever the request's value went (drafting, the
  run-ahead batches, the decode loop), and nothing allocates in proportion to it up front.
- **A separate stop reason, `capacity`.** A UI has to tell "you asked for this many" from "the context
  is full", so the reason follows the request rather than the numbers: an explicit value never yields
  `capacity`, even when it equals the limit, and an absent one never yields `max-tokens`.
- **`ChatSession` reserves room up front.** Its overflow check runs before sending, and "prompt + 0"
  would pass with no room to answer in, so a turn with no `maxNewTokens` is checked with
  `reserveNewTokens` (default 1024) in its place. The reserve only decides which history is dropped: it
  is not passed to `generate`, and the reply may run past it up to the limit. It cannot be combined with
  a session `maxNewTokens` (every turn would then have a limit, and the reserve could never apply); a
  per-turn `maxNewTokens` overrides it for that turn. It is not clamped: a capacity smaller than the
  prompt plus the reserve fails loudly and names the option. `Pipeline.chat` has no history to drop, so no reserve
  applies.
- **A `capacity` stop is an ordinary cut-off turn.** It takes the path of max-tokens (`reply.end` is
  `"cut-off"`, the streamed text stays in the history, the KV is discarded, the next turn re-renders),
  with no new state. The cost is in `docs/limitations.md`.

Rejected:

- **A mid-turn context shift** (drop old rows and keep generating when the limit is reached): it
  changes the conversation inside a reply, and re-prefills up to `C` tokens within one turn. Truncating
  between turns (`onOverflow`) stays the only place the conversation changes.
- **A `remaining()` helper alone**: every caller would re-implement the same call and the same choice
  of what to do at the limit, and no stop reason could tell the two limits apart.

## Addendum (2026-11-06) — `capacity: "fit"`

**Problem.** Choosing a capacity means guessing how much VRAM is free. WebGPU exposes neither the total nor the free amount
(ADR 0070 decision 5), and a guess that is too large ends in an out-of-memory failure in the middle of a run.

**Decision.** `Pipeline.fitCapacity()` returns the largest capacity this device can hold right now.
`SequenceOptions.capacity` and `ChatOptions.capacity` accept `"fit"`, which means that result. A `ChatSession` takes the
number (`capacity: await pipeline.fitCapacity()`), since it holds one capacity for its life.

- **Candidates**: `maxPosition`, then the powers of two below it, then the distribution's default. Nothing below the default
  is tried: it is what the distribution promises to work, and if even that does not fit the call fails loudly.
- **Need**: `peakAccountedBytes − resident.weights.totalBytes` of `estimateSessionMemory`, because the weights are already on
  the GPU when the call runs. A candidate over a limit on one buffer is skipped.
- **Margin**: 1 GiB on top of the need. The sustained VRAM ran up to about 320 MiB over the estimate, and short peaks 430–670
  MiB more ([research 2026-11-05](../research/2026-11-05-gemma4-long-context.md) §5).
- **Check**: one trial allocation per candidate through `fitsHeadroom` (ADR 0070), which needed the Dawn change in that ADR's
  addendum of 2026-11-06.

The result is a fact of that moment, not a reservation, and it differs across machines and runs. It does not move the
generated tokens, as capacity does not. An explicit opt-in; the default stays the distribution's.

On the RTX 3080 Ti under Node + Dawn, 12B `i4-fast` fits 32768 and E2B fits 131072, and a 12B chat at the result ran at
8.9 GiB. In Chrome (D3D12) on the RTX 5070 Ti (2026-11-07), with `fullStateStorage: "f16"`, it gave 12B the largest candidate, 262,144, and a chat ran.

Rejected: **making it the default.** A capacity that follows the moment would make a conversation's memory depend on what
else was running when it began. **Reporting a free-VRAM number**: ADR 0070 decision 5 rejected that face of the API, and this
answers the question without one.
