# 0142: Decode run-ahead — several greedy steps per GPU batch on chained deferred runs (`decodeAhead`, `GenerationRun.chain`)

- Status: accepted (2026-11-06 — the maintainer; proposed 2026-11-04 — implemented and measured on the maintainer's Apple M2
  under Chrome; its GPU tests pass on the development machine's RTX 3080 Ti; opt-in, default off)
- Date: 2026-11-04
- Related: ADR [0066](0066-generation-context-state-slots.md) (the generation context: decision 4 the chunk bound,
  decision 6 one path that moves the logical length; the addenda of 2026-09-08 / 2026-09-09 — deferred commit and the
  sliding slack — and of 2026-09-12 — generation runs in a batch) / [0054](0054-resident-loop-and-fence.md) (resident
  tensors, `BatchScope` + `Session.enqueue`, `copyOutputs`, the no-overtaking invariant) / [0004](0004-execution-model.md)
  (invariant ④) / [0083](0083-generation-api-surface.md) (the generation sequence, the uncommitted last token) /
  [0085](0085-ple-host-gather.md) (PLE residency on the GPU) / [0096](0096-speculative-decoding.md) (speculative decoding
  — the other user of the pending commit) / [0107](0107-model-input-error.md) (input errors) / [0008](0008-public-api.md)
  (the public surface) / perf-ledger H-27, H-28 / research
  [2026-11-03-qat-integer-gemv-m2](../research/2026-11-03-qat-integer-gemv-m2.md) §10 (the measurements that motivated
  it) and §11 (every number of the results).

## Context

A greedy Gemma decode step is one GPU batch: the target's run and the small selection graph (`topk` with k = 1) are
enqueued together, the batch's single fence reads back the selected value and index (8 B), and the host writes that
token as the next step's input and opens the next batch. Every token therefore pays one fence, one readback and one host
round trip before the GPU has the next step's work.

On the M2 (QAT E2B, integer GEMV + byte-plane GEMV, greedy; research §10) that round trip does not show as host work:

- The page's JavaScript takes about 1 ms per token (0.9 ms from one token's readback resolving to the next run's 11
  command buffers being finished).
- The fixed cost of 1,036 empty dispatches in 11 submits was 2.92 ms from the first submit to the readback — too small
  on its own to explain the time per token.
- In the faster of the two states the M2 showed in a run with the GPU tap on, the GPU was busy about 20 ms per token,
  against about 22.7 ms per token end to end with the tap off: roughly 2.5 ms per token in which the GPU has no work
  (an estimate across two runs, not one measurement).

Filling that gap needs the GPU to have the next step's work before the host has read the current token back. Two rules
stood in the way:

- **Batches do not overlap.** `BatchScope` sections are exclusive per device: a second `beginBatch` on the same device
  waits until the first returns its lock in `finish()` (`packages/runtime/src/gpu/context.ts`). Issuing the next
  token's batch before the current one is read back is therefore impossible; the steps that run ahead must share **one**
  batch.
- **One unsettled run per context, and no run while a commit is pending** (ADR 0066 addenda of 2026-09-08 and
  2026-09-12). `d` decode steps in one batch are `d` unsettled runs on one context, each writing a row the host has not
  accepted yet.

perf-ledger H-27 had filed the idea on 2026-09-19 with these preconditions: GPU-resident PLE (H-28), an opt-in exception
to the ADR 0066 rule bounded by the sliding slack, greedy only, and the diagnostic path left per step.

## Decision

### 1. Chained deferred runs (`GenerationRun.chain`, runtime)

`GenerationRun.chain?: boolean`, default `false` — every rule for callers that do not pass it is unchanged (an ordinary
run while a commit is pending, a second unsettled run, `rewind` / `fork` while a commit is pending are all still
refused). A chained run is a deferred run admitted **next to** rows already awaiting commit:

- **Only with `commit: "deferred"`.** Any other combination, and a `chain` that is not a boolean, fails loudly at issue
  (`ExecutionError`, in the same synchronous section as the `commit` check).
- **Placement.** The run sits at the end of the uncommitted range: `pendingCommit.pastLength + pendingCommit.queryLength`
  (or `pastLength` when nothing is pending), plus the rows of the chained runs accepted before it and not settled yet.
  The context **reserves that position at issue** (`acquireChainedRun`), and the run's body reads, writes and defers
  against that reservation. MUST: the position is decided at issue, not in the body — the pending commit grows only when
  a batch settles, so a body reading "the end of the pending range" would place every chained run of one batch on the
  same row. The caller builds its position inputs at issue from the same value (the provisional position = `pastLength`
  + every uncommitted row issued before the run).
- **Concurrency.** Several unsettled chained runs on one context are admitted only inside one `BatchScope` (`enqueue`
  with the same `batch`). A chained `Session.run`, or a chained enqueue of another batch, is refused while any run on the
  context is unsettled — the rule without `chain`.
- **Bound.** The whole uncommitted range (the pending commit + the unsettled chained rows + this run) obeys the bound of
  one deferred run of that many rows: `≤ slidingSlack` on a context with sliding slots (rejected rows must not overwrite
  live past KV — the ADR 0066 addendum of 2026-09-09 applied to the range) and `≤ chunkLength` in every case (decision
  4), plus the u32 limit of the logical length. It is checked at issue, before a dispatch exists, and every check runs
  before the reservation is recorded (a reservation left behind by a failed check would refuse every later commit).
- **Commit.** On success the run's rows join the pending commit (`pendingCommit.queryLength` grows by `queryLength`).
  The members of a batch settle in the order they were accepted, and each must continue the uncommitted range exactly
  (a gap or an overlap is a broken invariant and fails). `commit(rows)` with `0 ≤ rows ≤ pendingCommit.queryLength` then
  folds the whole chain at once — still one path that moves the logical length (decision 6).
- **Failure.** As for any deferred run: a run that submitted a state-changing dispatch poisons the context, and a failed
  batch settles every member as failed, so no part of that batch's chain joins the pending commit. An acceptance refused
  right after its reservation (the batch refused the enqueue) hands its position back, so the next chained run of the
  batch takes the same row.

**Why a separate field, not a third `commit` value.** `chain` changes where the run is placed and when it is admitted,
not when the logical length moves: the advance stays deferred. As a field, every existing branch on
`commit === "deferred"` keeps holding for a chained run without being taught a new value.

**The lengths uniform.** A context has one buffer of logical lengths that every run's dispatches read. With chained
runs, `writeLengths` writes the run's own placement (its reservation), not the context's committed length, and each
run's dispatches still read the value written for that run because of three orderings that already held: ① the runs of
one Session execute their bodies one after another; ② each body writes the lengths before encoding its first dispatch
and submits everything it encoded before it returns; ③ `queue.writeBuffer` takes effect at its place on the queue
timeline and does not overtake a submit issued before it (ADR 0004 invariant ④, ADR 0054 decision 2). So the next
chained run's write lands after the previous run's dispatches, never before them.

### 2. The token id fed on the GPU (Gemma)

- **One fixed resident for the id.** The step's `input_ids` is a resident tensor of one i32 (`ids`). The host writes the
  batch's first token into it before the batch opens; inside the batch the selection graph copies each step's selected
  id into it for the next step. One fixed resident means one backing for every step of the decode shape — feeding a host
  id in step 0 alone would be a second backing signature.
- **The selector's second exit `next_id`.** When the run-ahead is built, the selection graph adds `next_id`, a reshape
  of `index` to the shape of `input_ids` (`[1,1]`; a reshape costs no dispatch). One graph output can be copied to only
  one resident, and each step copies the id twice: `index` to the step's own readback resident, `next_id` to `ids`. Each
  step has its own value / index residents, since a later step would overwrite a shared one before the batch's single
  readback; `finishAndRead` reads all of them at the one fence.
- **The PLE gather on resident ids.** With the PLE on the host, every step's per-layer inputs are gathered on the host
  from the step's token — which the run-ahead keeps on the GPU. The GPU-resident PLE (ADR 0085) gains a resident-id form:
  a small expansion graph turns the id into the gather's row indices `id × layers + layer` with i32 `mul` and `sub` (the
  vocabulary has no i32 `add`, so the offset is subtracted in negated form; the largest index fits i32 for every
  distribution), and the gather runs on those rows, returning the same decode-shape output the host-id form returns. An
  out-of-range id gathers NaN rows (the `embedding` kernel's contract), never another token's row; it surfaces at the
  host's check of the id it reads back. Each step is awaited before the next is issued, because step k's gather reads the
  id that step k−1's copy put into `ids`.

### 3. The generation loop (`GenerationSequenceOptions.greedyAhead`)

- One call runs `n = min(d, remaining)` decode steps in one batch (remaining = the decode steps left before the token
  limit), step k at position `pastLength + k`, each a chained deferred run of one row. The derived inputs of every step
  are built at issue; step k > 0's token is still on the GPU, so the slot must declare its derived inputs
  `positionsOnly` (`DerivedRunInputs.positionsOnly` — for Gemma with the GPU-resident PLE only RoPE is left on the host,
  and it reads positions alone).
- After the readback the tokens are handed out one by one with **the per-step loop's bookkeeping** (the pending token,
  the generated count and the history updated before each yield, the stop check, the yield position), and the rows the
  per-step loop would have committed by then are committed in one `commit(rows)`: up to the first stop token, the token
  at which a consumer closed, or the draw that threw. The rows after it were computed but are never committed (the next
  run overwrites them — the deferred contract). As in the per-step loop, the last token stays uncommitted as the pending
  token (ADR 0083).
- The abort boundary is the batch; a token of a batch already run is checked synchronously before it is handed out. A
  failed batch commits nothing, and the frontier stays the token the batch was fed, as for a failed per-step run.
- Each turn's stop carries `GenerationStop.decodeAhead` — the depth, the batches the turn ran and the steps they computed
  (`steps` can exceed the tokens delivered after a stop inside a batch). `batches: 0` means the turn did not take the
  run-ahead. A chat turn closed by a stop string copies it (`ChatStop`).
- Setup check at sequence creation (failing loudly, and closing the context it opened): the depth is an integer ≥ 2,
  `≤ slidingSlack` and `≤ chunkLength`; no speculative decoding (a verify is not a greedy step, and both would hold the
  context's one pending commit); derived inputs absent or declared `positionsOnly`.

### 4. The option `PipelineOptions.decodeAhead` (Gemma)

`decodeAhead?: number`, default `1` = off (one greedy step per batch, as before). With `d ≥ 2` a temperature-0 turn
without processing runs up to `d` steps per batch.

- Validation (`ModelInputError`, ADR 0107): an integer ≥ 1; `d ≥ 2` needs `pleResidency: "gpu"` and cannot be combined
  with `speculative`. Both are decided by the options alone, so they are checked before a byte of the assets is read.
  The upper bound — the context's sliding slack, 8 for the published Gemma 4 graphs — exists only once a conversation's
  context does, so it is checked when the first sequence is created (decision 3).
- **Used only where the per-step greedy hook is used**: temperature 0 without processing (no repetition penalty or logit
  bias), non-speculative, without run observation (a pipeline built with `onRunDiagnostics` keeps per-step runs, whose
  diagnostics cannot be read from a batch of several). Other turns decode per step; `GenerationStop.decodeAhead` with
  `batches: 0` is the diagnostic. MUST: a depth ≥ 2 that cannot be built fails loudly rather than silently decoding per
  step — "the run-ahead did not make it faster" would otherwise remain with neither an exception nor a warning.
- The playground's `/gemma4-bench` has per-arm "PLE on GPU" and "Decode ahead" (1 / 2 / 4 / 8) controls that pass the two
  pipeline options; the chats of a run-ahead arm record `decodeAhead` in their stop.

### 5. The defining invariant and its checks

**With the run-ahead the generated token ids are identical to the per-step greedy path** — the same kernels run on the
same inputs; only the batching changes (where the next token comes from, and when the rows are committed).

- Without a GPU: `packages/runtime/tests/runtime_chained_generation_test.ts` drives a context on a fake device through
  the execution order (placement, settlement and commit, a failed batch, a refused acceptance, the bounds, the
  concurrency refusals, the poisoned context); `packages/models/tests/generation_decode_ahead_test.ts` runs one fake
  model through the per-step greedy hook and through the run-ahead and compares everything a caller observes (token
  events and positions, the stop, `used`, the next turn's frontier — at several depths, with a stop inside a batch, the
  budget, interruptions, a throwing draw, a failed batch, the path taken, the setup refusals);
  `packages/models/tests/gemma4_decode_ahead_test.ts` covers the option's validation, the expansion graph and the
  assembly in the greedy output.
- On a GPU: `packages/runtime/tests/gpu_chained_generation_test.ts` (the chained runs of one batch compute what the same
  steps compute one by one — the lengths-uniform ordering; the refusals; a failed member poisons) and
  `packages/e2e/tests/e2e_gemma4_qat_mobile_decode_ahead_test.ts` (QAT E2B, depths 4 and 8 against depth 1: a chat to the token
  limit, a chat that stops at the end of turn inside a batch, a two-turn sequence; a depth ≥ 2 must report batches run). Both pass (addendum of 2026-11-06).
- On the M2 (Results below): the bench's `tokenIdsSha256` per prompt was equal across the arms of both runs.

## Results (2026-11-04, Apple M2, Chrome)

QAT E2B `i4-fast` + `integerGemv` + `linearGemvPlanes` + `pleResidency: "gpu"`; arms `decodeAhead` 1 and 4; greedy, tap
off, 4 rounds in ABBA order (8 visits, 2 warm chats × 2 prompts per visit), the page reloaded before the run (research
§11):

- Fast state: depth 1 about 22.8 ms/token (43.8 tok/s), depth 4 about 20.65 ms/token (48.4 tok/s) — **−2.2 ms/token,
  ×0.905**.
- From visit 6 on, the M2 dropped into a slow state (about 30–31 ms/token) for both arms; those values are not compared.
- `tokenIdsSha256` equal across arms for both prompts. The run-ahead's stop record: depth 4, 32 batches, 127 steps.
- Earlier runs that suggested the run-ahead was slower, and a depth-8 arm, were confounded by the same state drop (the
  order of the visits); they are recorded in research §11 and not used.

## Consequences

- Public surface: `GenerationRun.chain` (`@karume/runtime`); `PipelineOptions.decodeAhead` and
  `GenerationStop.decodeAhead` (`@karume/models`, the Gemma pipelines), with the type `GenerationDecodeAhead` exported
  from `@karume/models/gemma`. Internal only: the generation program's `DerivedRunInputs.positionsOnly` (not on the
  public surface). With the defaults nothing changes for existing callers.
- Memory: the run-ahead needs the PLE on the GPU — for QAT E2B a 1.09 GiB table in one binding, which the device must
  allow (`maxStorageBufferBindingSize`). The resident-id form adds an expansion Session, two small residents and one more
  decode-shape backing of the gather; the run-ahead adds 8 B of readback residents per step.
- At a stop token (EOS) a batch has already computed up to `d − 1` steps past it; they are wasted GPU work, left
  uncommitted and never delivered.
- A failed batch commits nothing; the per-step path would have kept the steps before the failing one.
- `GenerationRun.chain` is a general runtime capability (the rows of a chain are bounded like one deferred run), but
  only the Gemma greedy output uses it.

## Alternatives considered

- **Overlapping batches** (issue batch t + 1 before batch t is read back) — impossible: `BatchScope` sections are
  exclusive per device, so the second waits for the first.
- **A host-side id write per step** inside the batch — keeps the round trip the run-ahead exists to remove: the host can
  write a step's id only after reading the previous step's selection.
- **One id resident per step** — each step would bind a different resident, a different backing signature, so the
  decode shape would bake (and keep) one backing per step instead of hitting one.
- **A third `commit` value** (say `"chained"`) — rejected: the length still advances on a deferred commit, and every
  branch on `commit === "deferred"` would have to learn the new value (decision 1).

## Open points

- No speed was measured on any backend other than the M2's Metal.
- Depths other than 4 were measured only in runs confounded by the M2's state drop; depth 2 and 8 have no clean number.
- Temperature > 0: the next token is drawn on the host, so the run-ahead does not apply; it would need the draw on the
  GPU (perf-ledger H-32).
- The M2 bench drops into a slower state after about six visits regardless of arm (research §11); an A/B on it needs a
  short run or the slow visits excluded.
- Making it a default (a depth in a preset or in code) — not ruled.

## Addendum (2026-11-06) — a model without PLE, a distribution that suppresses tokens, and the GPU checks

**The GPU tests of decision 5 have now been run**, and they pass on the RTX 3080 Ti (the ones of this ADR with those of ADR
0140 and 0141: 20 of 20).

12B was refused on two counts; both are lifted:

1. **It has no PLE**, so there was no GPU-resident table to drive the resident-id path. `pleResidency: "gpu"` is now
   required only of a model that has a PLE. Without one the batch's next-token ids go straight into the graph's id input
   through a resident the pipeline owns (`aheadIds`). The pipeline disposes that resident after the Session, because baked
   bind groups keep referring to it.
2. **Its distribution suppresses tokens**, while the selection took the raw maximum. The selection graph
   (`gemmaSelectionModel`, `packages/models/src/gemma/greedy-output.ts`) now adds −∞ at the suppressed ids before its `topk`,
   so the GPU picks what the host's masked argmax picks. The host's per-step greedy hook uses the same graph, so
   `createSampler` takes `selectionMasksSuppressed` and checks that the returned token is not a suppressed one: a graph that
   lost its mask fails loudly rather than generating a suppressed id.

Checks: `packages/models/tests/gemma4_selection_mask_test.ts` compares the GPU selection with the host's at the real
vocabulary, with a suppressed id as the raw maximum, a tie and −0, and shows that the graph without the mask returns the
suppressed id. `packages/e2e/tests/e2e_gemma4_12b_decode_ahead_test.ts` runs 12B `i4` and `i4-fast` at depths 1, 4 and 8
against a host-selection reference (a chat to the token limit, a stop inside a batch, a two-turn sequence), and a depth ≥ 2
must report batches run.

Results on the RTX 3080 Ti, decode wall ms per token of two rounds
([research 2026-11-06](../research/2026-11-06-gemma4-12b-fast.md) §4): on `i4-fast` at P ≈ 1K depth 1 is 26.5 / 26.8,
depth 4 is 26.7 / 22.4, depth 8 is 22.3 / 22.2; on `i4` depth 1 is 74.5 / 76.5, depth 8 is 65.3 / 66.9. Token ids were
identical across depths. It stays opt-in; a depth in a preset is still not ruled.
