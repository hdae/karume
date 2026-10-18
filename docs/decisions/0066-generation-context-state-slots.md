# 0066: Autoregressive execution model — GenerationContext and named state slots

- Status: accepted (2026-08-17 — the maintainer's approval, covering the states boundary,
  fixed-length chunks, ownership split and rewind); partly superseded: decision 4's "two execution
  shapes" by addendum 10 (2026-09-07) and [0096](0096-speculative-decoding.md) (2026-09-08), addendum
  9's "sliding capacity = window" by 0096 decision 4 (2026-09-08), the deferred-run limit of
  2026-09-08 by the addendum of 2026-09-09, addendum 10's LRU bound by
  [0042](0042-prepared-execution-plan.md) decision 2 (2026-09-08); the addendum of 2026-09-13 withdrawn 2026-10-18
- Date: 2026-08-17
- Related: [0042](0042-prepared-execution-plan.md) (PreparedPlan key and backing — decision 5 splits
  ownership) / [0054](0054-resident-loop-and-fence.md) (resident / batch enqueue) /
  [0004](0004-execution-model.md) / [0008](0008-public-api.md) / [0058](0058-numerics-opt-in-contract.md)
  (this ADR adds no numerics variant) / follow-ups named in the text: [0067](0067-autoregressive-attention-vocabulary.md)
  (attention vocabulary), [0068](0068-decode-exit-multi-output.md) (decode exit, multi-output),
  [0069](0069-packed-w4-storage.md) (packed storage), [0070](0070-shard-loading-admission.md) (admission)
- Basis: [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  (eight reference implementations, two adversarial passes; "research §n" below)

## Context

The Session was "immutable weights + one-shot run" with no place for generation state. Running the
target models (Gemma 4 E2B / MiniCPM5-1B — research §6) needs four things settled first: ① lifetime
and placement of the KV state ② capacity versus logical length ③ the prefill / decode execution
shapes ④ consistency with PreparedPlan (ADR 0042). Established facts from the references (research
§1):

- Every implementation keys plan / graph reuse on "the parameters that fix the topology"; none makes
  the logical length a shape symbol (three ways to pass it — research §1.1).
- KV as ordinary tensor I/O (transformers.js) means rewiring outputs every step, no rewind, and no
  capture-style optimisation — confirmed as the shape to avoid (research §1.2, §1.4-2).
- State is not uniform across layers: Gemma 4 E2B mixes "sliding 512 × 28 layers / full × 7 layers /
  20 layers without their own KV"; ORT also has a place for per-layer shapes and KV-sharing layers
  (research §1.5, §6.1).

## Decision

### 1. Separate lifetimes: the Session is immutable, a GenerationContext owns one generation's mutable state

A `GenerationContext` is created from the `Session` (immutable weights, plan cache — unchanged). The
context owns the mutable state of **one sequence** (the physical buffers of the state slots and the
logical length) and releases it with `dispose()` (flush-before-destroy — ADR 0004).

MUST: **state is not an ordinary graph input / output** (no past-input / present-output fields in the
IR). State is an opaque slot that nodes reference by name; the context owns its allocation, lifetime
and progress (the MLC `nn.spec.Object` shape — research §1.2).

### 2. Named state slots (R3)

The IR gets a **`states{}` section**: `name → { dtype, shape }`. The shape is **concrete, of fixed
rank (rank ≤ 4) and includes the capacity** (symbols resolve through the Session's symbols). Added
to IR v1 at the time; today the section is specified in [docs/ir-v2.md](../ir-v2.md).

- **No "layers × uniform KV" assumption.** Sliding and full layers are **separate slots with
  different capacities**; the 20 KV-sharing layers **reference the same slot name** (no extra field).
  Other sequential state (a future LSTM h_n / conv state) uses the same container — a slot does not
  know it holds KV.
- How nodes reference slots (attention read / write fields, op vocabulary) is decided by ADR 0067.
  This ADR fixes only the principle: slots are referenced by name in a field separate from
  `ins` / `outs`.
- The exporter is unaffected until the first model that emits `states` (unknown keys fail loudly; the
  parser took the section with this spec revision).

### 3. Shape invariants restated (R2) and the logical-length contract

The permanent invariants are **at most** these three: **static physical storage** (slot capacity is
fixed at context creation and never reallocated during execution), **fixed rank**, and **the plan
cache key is the capacity** (the logical length is not part of the key).

- "The key is the capacity" is **a trade-off, not the industry default**: llama.cpp keys on a logical
  extent quantized to 256 boundaries (research §1.1 (a)). Karume keys on capacity because PreparedPlan
  (ADR 0042) is an exact bindings key with an LRU (bound: `PREPARED_PLAN_CAPACITY` — 0042 decision 2);
  with the extent in the key, decode would take a new plan every token.
- **The logical length is a value.** `pastLength` (logical length of committed KV) and `queryLength`
  (real tokens this step) are **independent runtime scalars** — neither shape symbols nor attrs
  (symbols enter the plan key (ADR 0042) and attrs are baked into the IR; same split as ORT's
  seqlens_k / total_sequence_length — research §1.1 (c), §3.1).
- **Capacity minus logical length is "masked keys / invalid rows"**, not expressed in shapes
  (research §3.4).
- **Workload condition.** Attention dispatches MUST have a workgroup count or total iteration count
  proportional to `pastLength × queryLength` (formula corrected by addendum 1); proportional to
  capacity wastes three orders of magnitude at 131K capacity with short sequences (research §1.1).
  The mechanism (core WebGPU `dispatchWorkgroupsIndirect`, or dispatch counts computed from the
  host-known logical length) is ADR 0067's implementation design.
- **Reserved place for a bounded logical extent** (declared only; implementation waits for the first
  model that needs it): a future data-dependent shape fits `states` / value declarations as "one
  compact-prefix axis with a bounded logical extent". A DDS op returns payload + extent (needs the
  multi-output of ADR 0068); the extent stays out of the plan key; admission charges the capacity.
  Beyond the bound and for dynamic rank, host-mediated graph splitting stays.

### 4. Two execution shapes: fixed-length prefill chunks and decode

(Superseded by addendum 10 on 2026-09-07 and ADR 0096 on 2026-09-08: the physical row counts are the
set `{1} ∪ chunkBuckets ∪ {chunkLength}` declared at context creation, which also carries the verify
shape of speculative decoding.)

- **prefill**: the prompt is split into **fixed-length chunks** (`chunkLength`, a plan-time constant
  fixed at context creation) and the last chunk is padded. **The no-op contract of padding rows is the
  core**: pad rows are not written to KV, do not appear in outputs and (under the workload condition
  of decision 3) add no work. `queryLength` cuts them off.
- **decode**: fixed `queryLength = 1`.
- Rejected: variable-length chunks (web-llm / MLC — research §1.3) suit one symbolic compilation; in
  Karume each length adds a PreparedPlan and pollutes the LRU (addendum 10 still rejects them). The
  whole prompt at once (ORT GenAI default) scales plans and transients with prompt length. Fixed
  chunks + padding is the ORT Windowed / vLLM bucket choice (research §1.3).
- Consequently **PreparedPlan has two steady plans, prefill chunk and decode** (+ the one-shot
  surface); fixed-capacity resident state is the precondition for reuse-style optimisations
  (research §1.4-2).

### 5. Ownership split with PreparedPlan (recipes in the Session, state backing in the context)

- **Recipes / plans (pure functions of the bindings — ADR 0042 decisions 1–2) stay shared in the Session.**
  MUST: **the context identifier is not part of the plan key** (otherwise every context switch throws
  away derivation and backing).
- **The physical state buffers and the bind groups that bind them belong to the GenerationContext.**
  The baked bind groups of ADR 0042 decision 5 bind resources with different lifetimes ("Session-owned slots /
  resident inputs" versus "context-owned state"), so **the bake unit is split** (bind groups with state
  live on the context side; dispatches without state are unchanged). Without the split the choice is
  silent stale reads of the previous context's KV or a full rebuild on every switch (research §1.6 —
  a high finding of the second review).
- A diagnostic on the level of `planBacking` (resident bytes of state backing, rebind count), following
  ADR 0042 decision 3: it is the only observation point for silent degradation.

### 6. Lifetime API (thin — ADR 0008)

- `session.createGenerationContext(spec)`: fixes slot capacities and `chunkLength` and allocates. An
  allocation failure fails loudly through the out-of-memory error scope (the admission estimator is
  ADR 0070's).
- `context.pastLength` (read-only): **the context owns** the progress of the logical length, which
  advances on a successful run. Manual host-side addition is not part of the API (no double
  bookkeeping).
- `context.rewind(position)`: truncates the logical position. **Rewinding into a range already evicted
  from a sliding slot fails loudly** (the rewindable range follows from the capacity scheme — ORT GenAI
  style, research §1.2). Tightened by addendum 2.
- `context.dispose()`: returns the physical buffers, flush-before-destroy.
- The call shape of prefill / decode (method names, I/O types) is implementation design; this ADR fixes
  only that the context owns progress and rewind of the logical length.

### 7. Device-loss contract

After device loss a GenerationContext is **invalid**; every operation fails loudly with the
`GpuDeviceLostError` family (WebGPU resources of a lost device cannot be recovered — research §1.4-4).
The generation is lost. Host-side dump / restore of state (llama.cpp style: state bytes + metadata,
without RNG / sampler) is **reserved only**; implementation waits for a real need.

### 8. Out of scope

batch > 1, multiple sequences, paged / block-table indirection (KV is contiguous capacity — the
explicit choice of research §1.4-1), prefix cache, the attention / GQA / mask vocabulary (ADR 0067),
the decode exit and multi-output (ADR 0068), packed storage (ADR 0069), shard loading and admission
(ADR 0070), sampling / RNG (stays on the host — settled in the op vocabulary).

Note (2026-10-11): one item, "duplicate a whole context of the same Session and capacity", entered the
scope with the addendum of 2026-10-11 (fork). Prefix cache (content-addressed reuse) and the other
items stay out of scope.

## Consequences

- The existing one-shot surface (Session.run / enqueue, all existing models, all sha256 checks) is
  **unaffected**: this ADR adds places and does not change the default path.
- The backing sections of ADR 0042 (decisions 4–5) were amended for decision 5 (ADR 0042 addendum of
  2026-08-18 — separate bake units).
- The implementation acceptance conditions (independent `queryLength`, machine-checked no-op pad rows,
  context switch with no stale read and no re-derivation, the workload condition) were met by the
  implementation waves of 2026-08.
- Recording the counter-example to "the key is the capacity" (llama.cpp's quantized extent) keeps the
  way out if decode performance ever stalls on the capacity key (that would supersede this ADR).

## Addendum (2026-08-17) — corrections and reinforcements after review

1. **Workload formula of decision 3 corrected.** "∝ pastLength × queryLength" is wrong because the
   causal triangle inside the chunk is non-zero at pastLength = 0. Correct: **∝ queryLength ×
   (effective pastLength + queryLength)** (sliding limits the effective past to the window).
2. **Rewind contract of decision 6 replaced.** Rejecting only evicted ranges is not enough: after an
   eviction, **even a rewind to a resident position leaves physical layout and logical range out of
   step** (ORT GenAI rejects every rewind below current for this reason — comment in
   windowed_kv_cache.cpp:67-80; it has no left compaction). Contract: **a positional rewind of a context
   that contains a sliding slot fails loudly** (valid only when no slot is sliding). Relaxing it needs a
   revision of this ADR together with compaction.
3. **Failure atomicity (reinforces decision 6).** If a run fails after a state-changing dispatch (a run
   with `state_append`) was submitted, the logical length does not advance but the physical ring may
   be overwritten — **the context is poisoned (invalid, every later operation fails loudly)**. No
   rollback or staging; recovery = new context + host-side rebuild.
4. **Carrier of the logical-length scalars (reinforces decision 3).** `pastLength` / `queryLength` are
   **not put into params** (the content-addressed cache — ADR 0042): a value that changes every step
   would grow that cache without bound (a known limitation) and could not be updated on a PreparedPlan
   hit, where derivation does not run. They travel in **one mutable uniform buffer owned by the
   context** (written with writeBuffer before each run's encode; recipes bind it fixedly).
5. **Capacity and binding limit (reinforces decisions 1 / 6).** A capacity whose single-slot binding
   exceeds `maxStorageBufferBindingSize` fails loudly in `createGenerationContext` (131K capacity × f32
   on Gemma 4 E2B's full layers can exceed the default 128 MiB; practical capacity is diagnosed by the
   ADR 0070 estimator). An f16 state storage option (an opt-in in the ADR 0058 style, since it changes
   the numerics contract) is **reserved only** (`RESERVED_STATE_DTYPES` in `format/ir.ts`).
6. **Physical shape of fixed chunks and the value contract of padding (reinforces decision 4).** The
   declared tensor shape of the prefill-chunk family is **fixed at chunkLength** (plan / prepared
   backing allocate the whole declared shape and write it fully — unchanged). Valid data is the
   **compact prefix of the first queryLength rows**. **Input values in the pad region MUST be
   zero-filled** (by the host: non-finite k/v in pad rows would leak NaN into valid rows through the
   "−inf add + exp" path; the value contract blocks it). Pad rows cannot contaminate valid rows by
   **row locality**: linear / pointwise / norm stay inside a row; the causal predicate keeps valid
   rows off pad columns (`col ≥ pastLength + queryLength`); `state_append` writes only queryLength
   rows; the exit (ADR 0068) reads only the real last row. **Full-write still holds** (ADR 0014): pad
   rows are written, with values that carry no meaning. Only **state slots** are written for
   queryLength rows alone (context-owned, outside the transient full-write rule; leftovers are
   overwritten by the next append with the same formula, and no reader reads outside the resident
   range).
7. **Bindability of states-only symbols (reinforces decision 2; 2026-08-17).** `checkSymbolBindability`
   (`format/ir.ts`) required every symbol to appear in an input shape, so **a symbol that appears only
   in states (the KV capacity `C`) could not be declared**. The capacity is chosen at context creation
   (decision 3), so baking it at export contradicts this ADR. Contract: **a symbol in a states shape is
   a valid declaration bound at `createGenerationContext(spec.bindings)`**; the bindability check
   accepts "appears in input shapes ∪ states shapes". Ordinary value shapes are still resolved from
   input bindings only (states-only symbols do not appear in value shapes), but **the key of a
   PreparedPlan that references state contains the resolved capacity** (decision 3 itself — the
   capacity value, not the context identity; contexts with C=512 and C=131072 in one Session get
   different keys for their state-referencing plans).
8. **Pad-row output fixed at 0 (narrows item 6; 2026-08-18).** "Pad rows are written with meaningless
   values" becomes **exact 0**. Reason: if pad rows scanned live columns, work would scale with the
   physical chunk rows M; so ① QK / ② row statistics cover only valid rows (`row < queryLength`) and
   ③ PV writes 0.0 to all of D for pad rows without scanning live columns (full-write unchanged).
   ① **Empty row → 0 (ADR 0067 decision 6) is contained** in this zero write (empty rows ⊂ pad rows; a
   valid row always has its causal self-reference); ② the row-statistics guard (amax = −inf → (0,0)) is
   **defensive only**, kept to detect a dispatch / kernel row-range mismatch. Non-finite V can no longer
   turn an empty-row output into `0 · NaN`.
9. **Sliding slot capacity baked as the real window (scope of decision 3; 2026-08-30).** Choosing the
   capacity at context creation is needed **only for full slots**, which keep the whole context. A
   sliding slot's ring closes at exactly the window, so a capacity symbol leaves `C − window` rows dead
   (Gemma 4 E2B: 24 slots × 256 × 4 B — about 180 MiB at C=8192). States-form recipes give sliding
   slots a numeric `capacity = window` (`StateAttentionSpec.capacity`), and the form check splits by
   layer kind: "full = symbol / sliding = number" (`export_decode.assert_ir_form_decode`). Function
   unchanged (token-sequence parity, three Gemma 4 cases). (Superseded by the addendum of 2026-09-08
   and ADR 0096 decision 4 on 2026-09-08: the sliding capacity is `window + slack`
   (`SLIDING_SLACK_ROWS = 8` in `tools/export-recipes/gemma4/export_decode.py`); the symbol / number
   split of the form check stays.)
10. **Prefill physical rows widened to a declared set (relaxes decision 4; buckets, 2026-09-07).** The
    physical row counts become **the set `{1} ∪ chunkBuckets ∪ {chunkLength}` declared at context
    creation** (`GenerationContextSpec.chunkBuckets` — strictly increasing integers in
    [2, chunkLength); omitted = none). The caller (`physicalChunkRows` in
    `packages/models/src/generation/sequence.ts`) takes the **smallest** value ≥ `queryLength` per
    chunk (one valid row still uses the decode shape). The chunk **split** stays `chunkLength`; only
    the rows that carry the last chunk change. Long prompts are fastest at 768 in one go, so the
    default is not lowered and variable length stays rejected: the declared set caps the number of
    PreparedPlans, and the pre-run check `assertGenerationRun` rejects M outside the set. Consequences:
    (a) PreparedPlan LRU bound 4 → 8 per capacity (superseded by ADR 0042 decision 2 on 2026-09-08:
    the bound is `PREPARED_PLAN_CAPACITY` in `packages/runtime/src/runtime/executor.ts`, 12 since ADR
    0096 added the verify shape); (b) buckets with M ≥ 16 take the tiled path (①ₜ / ③ₜ, perf-ledger
    K-13) and add one WGSL variant each (the PipelineCache does not evict; compiled at first use);
    below 16 they fall to ① / ③; (c) **numerics**: row-local ops and tiled attention use the same
    formula and summation order for every M, but the GEMM geometry (`gemm-geometry.ts`) depends on M,
    so **bucket shapes and the 768 shape do not guarantee bit-identical logits** (bound by the
    token-sequence golden, `e2e_gemma4_sequence_test.ts`); (d) **VRAM**: the unaccounted window of the
    estimate (ADR 0089) widens from "decode + prefill shape" to "largest bucket + prefill shape"
    (`docs/limitations.md`); backing is held under a budget since ADR
    [0095](0095-plan-backing-budget.md). The default ladder is `GEMMA4_CHUNK_BUCKETS`, measured in
    [research 2026-09-07](../research/2026-09-07-codex-perf-review-followup.md) §5 (prompted by
    performance finding F-02: a 32-token turn fell from 385 to 106 ms).

## Addendum (2026-09-08) — deferred commit and slack in the sliding ring (ADR 0096 decisions 3 / 4)

- **"The logical length advances on a successful run" (decision 6) now has two forms**:
  `GenerationRun.commit` `"immediate"` (default — advances `queryLength` rows when the run returns
  without an exception) and `"deferred"` (progress is held; the host reads back, decides the accepted
  rows and calls `GenerationContext.commit(rows)` with `0 ≤ rows ≤ queryLength`). While a commit is
  pending (`pendingCommit`), the next run and `rewind` are rejected, and dispose folds the pending
  commit. A failed run creates no pending commit (the poison of item 3 is unchanged). Only one path
  moves the logical length at a time, so double bookkeeping stays excluded.
- **The physical rows of a sliding slot are `window + slack`** (baked by the distribution; slack 8 for
  Gemma 4). The ring modulus is capacity, not window (ADR 0067 addendum of 2026-09-08), so rows written
  ahead of the logical length (rejected rows of a deferred run) land only on columns outside the live
  window. The context exposes `slidingSlack` (minimum of capacity − window) and enforces **`queryLength
  ≤ slidingSlack + 1` for deferred runs** in the synchronous section of run issue (superseded by the
  addendum of 2026-09-09 on 2026-09-09: tightened to `queryLength ≤ slidingSlack`). Beyond the limit, a
  zero-accept would let rejected rows overwrite the live window — silent corruption, no exception, no
  NaN. Immediate runs (prefill / decode) commit all rows, so their limit stays `chunkLength`.
- Item 2 (rewind rejected for contexts with sliding slots) is unchanged: speculation handles rejection
  by "write, then advance by the accepted rows", not by rewind.

## Addendum (2026-09-09) — the deferred limit tightened by one column (`queryLength ≤ slidingSlack`)

The implementation of step 3 (ADR [0096](0096-speculative-decoding.md) addendum, 2026-09-08) tightened
the limit to **`queryLength ≤ slidingSlack`**; this is the enforced value.

- Reason: a borrowing context (the drafter's readonly attention) reads down to column `P−W`, one
  column older than the lender. Rejected row j overwrites logical column `P+j−capacity`,
  which stays outside the borrower's window when `j − m < capacity − window = slidingSlack` (m = rows
  committed, including `commit(0)`) ⟺ **`Q ≤ slidingSlack`**. The same limit applies with no borrower
  (a borrower can open after deferred + `commit(0)`; branching on its presence would let a late
  borrower read a broken window).
- Immediate runs (prefill / decode) keep `chunkLength`. With slack 8 the drafter runs at most 7 steps
  (the distributed drafter uses 3); a context with slack 0 cannot issue a deferred run at all.

## Addendum (2026-09-12) — GenerationContext in batched execution

- `EnqueueOptions.generation` takes the same context, queryLength and commit as an ordinary run.
  Context ownership, bindings, allowed rows, capacity, the deferred sliding slack and the borrower
  constraints are those of a run. IR, stored assets and the numerics / length contract of Session.run
  are unchanged.
- The use reservation is taken when the enqueue is issued and held until the batch is finally settled.
  The enqueue Promise means only that encoding finished; the length does not advance then. On success
  of the section the context advances / defers, and the commit and the release of the reservation
  complete before `finish` / `finishAndRead` return. `settle()` is an intermediate fence, not a length
  commit.
- Several unsettled executions on one context are rejected; different contexts may share a batch. A
  borrower's reservation kind is passed to the lender to keep reads and writes exclusive; a borrower
  never advances the logical length.
- A failure of a later Session or of a readback fails the whole section. An execution that wrote state
  and was submitted poisons its context by the same snapshot rule as a run, and reuse is rejected.
  Input / compile failures before any write do not poison. If an internal failure happens midway
  through committing several contexts, every context that was written is poisoned.
- `context.dispose` and `Session.dispose` of the owning Session are rejected while a batch reservation
  is held, until acceptance closes; after finish they can be retried. The two-step dispose of ordinary
  runs (close to new work, dispose after accepted runs) stays. Queuing the batch wait on the Session's
  serialisation chain is rejected: it creates the cycle enqueue → dispose → batch → enqueue.
- The existing rule for external batches remains: enqueue is rejected while an ordinary run of the
  same Session is unsettled, so callers align the order of ordinary runs and batch opening.

## Addendum (2026-09-13) — finer prefill chunks for the browser comparison page

Withdrawn 2026-10-18: it set the initial selection of the Deno-era browser comparison page to buckets
of 4–56 rows ([research](../research/2026-09-13-m2-prefill-adoption.md)); that page was removed on
2026-10-16 (`bb35fe2a`). Default buckets will be decided again on the rebuilt page with Dawn
measurements.

## Addendum (2026-10-11) — forking the conversation state

The maintainer's rulings of 2026-10-10 and 2026-10-11 added `GenerationContext.fork():
Promise<GenerationContext>` to the lifetime operations of decision 6. Use: try another continuation
from the middle of the same conversation (for example regenerating a chat reply), which previously
required replaying the token transcript into a new context.

### Scope

- In scope, one item only: **a full duplicate of a context of the same Session and capacity**: every
  byte of every slot (including sliding slack rows), `pastLength` and the immutable values
  (`chunkLength` / `chunkBuckets` / capacity bindings / sliding windows).
- Not copied: baked bind groups (re-baked on the duplicate's first run), the logical-length uniform
  (rewritten before every encode), borrowers and a pending deferred commit (empty by the rejections).
- Still out of scope: persistence (decision 7), prefix cache, another capacity, batch > 1, paging.
- The fork is an independent context owned by the caller; the original may be disposed first.

### Meaning of "while borrowed"

"Fork is rejected while borrowed" means **while a borrower's run is executing** (while the lender's use
reservation is held), not "while a live borrower exists". Three reasons:

- A borrower (the drafter's readonly attention) never moves the logical length and writes no KV row
  (ADR 0096 decision 1).
- A borrower's run also takes the lender's use reservation, so a fork during it is rejected by (d)
  below.
- A speculative sequence keeps its borrower for its whole lifetime; rejecting the mere presence of a
  borrower would make fork unusable under the default speculation settings.

After a fork, the original borrower stays bound to the original slots; the duplicate has no borrower.
The GPU test `gpu_borrowed_context_test.ts` fixes this binding.

### Synchronous-section contract

- MUST: the rejection checks, capturing `pastLength`, the Session-dispose check, allocation, encoding
  the copy, `queue.submit` and issuing the error-scope pop happen **in the synchronous section of the
  `fork()` call, with no await**; only the pop result is awaited. Same shape as
  `GenerationContext.create` and `ResidentTensor.read` (the third invariant of GpuContext error-scope
  sections: a scope that completes inside one synchronous section needs no lock).
- No use reservation is taken on the original. When the original is quiescent (no run in progress, no
  pending deferred commit), no unsubmitted write to its slots exists (a run's reservation is returned in
  its finally, when all its GPU work is submitted). The copy is submitted at the call, so queue order
  guarantees "after earlier writes, before later writes"; a run / `rewind` / `dispose` / second `fork`
  issued on the original without awaiting `fork()` is safe.
- The copy uses neither the Session's serialisation chain nor the `SubmitScheduler` pending queue (own
  encoder and submit) — the same side as `createGenerationContext`, which stays off the chain to avoid
  a wait cycle with batch sections (the NOTE at `Session.#adopt` in `executor.ts`). As a cost, the
  copy's GPU time is not in the scheduler statistics.
- Registration after allocation (re-check dispose → live set → `contextCount`) goes through the one
  path shared with `createGenerationContext` (`Session.#adopt`).
- An allocation failure (out-of-memory error scope) fails loudly; the original stays intact and what
  was allocated is released.

### Rejections

`fork()` decides in its synchronous section and rejects the returned Promise (same shape as run /
dispose); (b) with `GpuDeviceLostError`, the others with `ExecutionError`:

(a) after a dispose request; (b) after device loss; (c) after poisoning (item 3); (d) an unsettled run
exists (including a borrower's run holding the lender's reservation); (e) a batch use reservation is
held (until the section is finally settled); (f) a deferred commit is pending (`pendingCommit`); (g)
the context is a borrower (borrowed slots are not duplicated); (h) the Session is disposed (checked at
the call and again after allocation).

Not rejected: a lender with a live borrower, and a context with sliding slots (independent of the
rewind rejection of item 2 — the copy takes the physical layout along, so logical ranges stay aligned).

### Diagnostics and VRAM

- Each fork adds the original's resident bytes to `stateBacking.residentBytes` and 1 to
  `contextCount`. `rebindCount` grows by one per backing used on the fork's first run (re-baking is per
  (context, backing) pair); runs that continue the original do not add to it.
- The estimate (ADR 0089) is unchanged: the number of contexts kept alive × state bytes is the caller's
  account (`docs/limitations.md`).

### Determinism

Right after the fork the state is bit-identical to the original. Whether a branch's tokens match "the
same operations without fork" follows the numerics path: without speculation and with `speculative:
"always"` they match; with the self-funding gate, wall-clock time differs per branch even with the
gate state copied (ADR 0083 addendum of 2026-10-11), so they match only where speculation check ⑤ is
green (otherwise see the speculation section of `docs/limitations.md`).

### Relation to other decisions

Fork is not the dump / restore of decision 7 and does not anticipate it: it stays inside the GPU, the
device and the Session. Public surface (ADR 0008): no new exported type, three methods on existing
types (`GenerationContext.fork` / `GenerationSequence.fork` / `Gemma4ChatSession.fork`).

### Rejected alternatives

- A: the copy on the Session's serialisation chain with a `"fork"` reservation. It would join the
  FIFO and the statistics, but moves a creation path onto the chain, gives the reservation three values
  and makes a fork wait behind long runs of other contexts.
- B: `createGenerationContext({ fork: source })`. One entry point, but a fork field in
  `GenerationContextSpec` (shared with the estimate) leaves the required `chunkLength` meaningless, and
  it departs from decision 6, which lines lifetime operations up on the context.
- C: snapshot and restore. Needs a new public type and VRAM for the saved base point, and leans
  towards the persistence left out of scope.
