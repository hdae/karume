# 0095: Keeping several slot backings — an LRU with a byte budget (perf-ledger H-15)

- Status: accepted (2026-09-07 — the maintainer's ruling: a default budget of 256 MiB is accepted as
  extra VRAM, changeable by the user); decision 4 revised by the 2026-10-10 addendum
- Date: 2026-09-07
- Related: ADR [0042](0042-prepared-execution-plan.md) (prepared plans and slot backings — replaces
  "a backing has capacity 1") / [0066](0066-generation-context-state-slots.md) decision 5 (a baked
  bundle still lives as long as its (context, backing) pair; now one bundle per backing) /
  [0089](0089-memory-limits-preflight.md) (estimation — `peakAccountedBytes` changes) /
  [0093](0093-transient-liveness-packing.md) decision 6 (footprint-invariance check — unchanged, a
  statement about one backing) / [0070](0070-shard-loading-admission.md) CX-4.3 (estimator contract —
  replaces "the max holds because one backing exists at a time") / [0004](0004-execution-model.md)
  (flush-before-destroy — retire → pending destroy → destroy after flush is unchanged)
- Measured demand: [research 2026-09-07](../research/2026-09-07-codex-perf-review-followup.md) §7.2
  (a run whose M changes rebuilds the backing, +37 to 42 ms per run, twice per chat turn ≈ 80 ms;
  unchanged after K-21: [gemv-rows-k21](../research/2026-09-07-gemv-rows-k21.md) §8.2)

## Context

A slot backing (the intermediate buffers of a run that hits a prepared plan — ADR 0042) existed
**once** per Session, because "slots are the run's intermediate buffers, ~1 GiB for a DiT; one per
signature multiplies VRAM".

In generation (gemma4) the run shape changes every turn: prefill bucket shapes (M = 32 / 64 / 128 /
256) → decode (M = 1) → the next prefill. With capacity 1 each switch retires the old backing and
allocates the new one (the context also re-bakes its bind groups), ≈ 40 ms, twice per turn ≈ 80 ms —
the order of TTFT (prefill run wall ≈ 80 ms after K-21). Speculative decoding (K-20) alternates draft
and verify shapes every cycle, so this was its revival condition ②.

Generation backings are small (gemma4 E2B, io + workspace): decode 3 MiB; buckets 32 / 64 / 128 / 256
take 9 / 17 / 33 / 65 MiB at capacity 2K and 23 / 45 / 89 / 177 MiB at 16K. Only the chunk-768 shape
takes 192 to 528 MiB. The multiplication concern applies only to DiT-class shapes.

## Decision

### 1. Backings are kept as an **LRU set with a byte budget** (`SessionOptions.planBackingBudgetBytes`)

The `Session` holds a `Map<plan key, ActiveBacking>` (insertion order = oldest first; a hit reinserts).
A hit run with a new signature computes the plan's region bytes (`planRecipes(...).totalBytes`, known
before allocation), retires the oldest backings until **held + new ≤ budget**, then allocates. A shape
that alone exceeds the budget retires all held backings and is kept alone (the old capacity 1). The
resident amount therefore **never exceeds `max(budget, largest single backing)`**. Budget 0 keeps one
backing, as before.

The default is 256 MiB (`DEFAULT_PLAN_BACKING_BUDGET_BYTES`): for gemma4 E2B, decode + buckets 32 / 64
/ 128 fit even at capacity 16K (all buckets at ≤ 8K); the chunk-768 shape is kept alone. Users change
it through `SessionOptions` and `Gemma4PipelineOptions` (anything but a non-negative safe integer
fails loudly).

MUST: a backing counts as the VRAM it holds = **sum of its regions + the input buffers it owns**
(diagnostics `residentBytes + inputBytes`). Counting regions only would shrink the bound to regions,
and the input buffers of every held backing (up to `PREPARED_PLAN_CAPACITY` = 12 prepared plans; the
constant is the source of the count) would pile up outside it (review finding). `residentBytes` stays
separate from inputs because the footprint-invariance check (ADR 0093 decision 6) looks at regions only.

MUST: retirement stays "push to pending destroy → destroy after that run's flush" (ADR 0004). The
retirement that brings the set within budget happens **before** allocation, to keep the held set's
count within `max(budget, largest single backing)`; decision 4's estimate relies on that bound. It does
not prevent physical overlap: a retired backing stays in VRAM until the flush either way, so old and
new coexist at allocation; that window is on the unaccounted side of decision 4 (corrected 2026-10-11).

### 2. A failure path retires only "the backing this run built"

On an exception during construction or a failed run, only the newly built backing (by key) is
retired; other held backings stay, as with capacity 1 (an unrelated failure must not force a ~GiB
rebuild). The budget retirement already happened **before** allocation (decision 1), so a failed run
that exceeded the budget also loses those backings (rebuilt on the next hit; correctness unaffected).

MUST: LRU eviction of prepared plans (`PREPARED_PLAN_CAPACITY`) retires the evicted key's backing (the
Session does not hold the buffers of a signature that will never hit again for its whole lifetime).

### 3. Context-side baked bundles are kept per backing and dropped on retirement

The baked bundle of a `GenerationContext` (ADR 0066 decision 5) becomes a table "backing generation
id → bundle", one per held backing, so moving between held shapes does not re-bake (diagnostics
`stateBacking.rebindCount`).

MUST: when the Session retires a backing, it drops the matching bundle of every live context with
`dropBakedGroups` (the bundle holds the retired object; even with generation matching, the reference
would extend its life). Match → re-bake → dispatch stays in one place (`#generationGroups`).

### 4. The estimate (ADR 0089) accounts **`max(budget, largest scenario)`**, not the scenario max

(Revised by the 2026-10-10 addendum: bucket shapes are computed and enter the max.)

The held set's upper bound becomes the accounted amount. Prefill bucket shapes (`chunkBuckets` on the
models side) are not estimate scenarios but lie inside the budget, which bounds them from above
(erring high; "peak of what is accounted" keeps its meaning). Unaccounted remains the window retire
(over budget / LRU eviction) → destroy. `AdmissionReport` reports the budget.

### 5. Options not taken

- **Two fixed backings for generation** (prefill + decode): with four buckets, "the prefill shape" is
  not one shape; switching between buckets remains.
- **Capacity 1 and retire before a bucket run** (closing only the unaccounted window of the "prefill
  buckets" limitation): the +37 to 42 ms per run stays.

## Verification

- Real-GPU check `gpu_plan_backing_test.ts`: with the default budget, alternating shapes keeps
  `buildCount` constant; held count and sum; LRU order; a new backing above the budget retires all;
  fault injection (failure midway) returns only the new one; budget 0 behaves as before.
  `gpu_state_execution_test.ts`: round trips between held shapes keep `rebindCount`; a retired
  backing's bundle leaves the context and a rebuild re-bakes once. The estimate: `estimate_test.ts`.
- Measured in [research 2026-09-07-gemv-rows-k21](../research/2026-09-07-gemv-rows-k21.md) §9.

## Consequences and open items

- Other families (anima / irodori / sbv2 …) build Session options from the manifest's `session`, so
  `planBackingBudgetBytes` is not passed through (256 MiB applies; the user cannot set 0). DiT-class
  families keep one backing as before; families running several shapes of tens of MiB (siglip2 base,
  depth-anything at low resolution) hold several (at most +256 MiB, in the estimate). Moving the option
  to the shared options is in `docs/backlog.md`.
- **A shape larger than the budget is kept alone**: gemma4's chunk-768 shape is ≈ 528 MiB at capacity
  16K (192 MiB at 2K), so with 256 MiB a turn with a prompt of 768 tokens or more still rebuilds 768 ↔
  bucket ↔ decode (only short-turn round trips disappear). Raise the budget for long prompts every turn
  ([limitations](../limitations.md)).
- **While a held backing has baked a resident input, that `ResidentTensor.dispose()` fails loudly.**
  With capacity 1 a shape switch ended the window; now it lasts until Session dispose or a budget
  overflow ([limitations](../limitations.md)).
- `peakAccountedBytes` becomes `weights + state + max(256 MiB, largest scenario)` by default, so a small
  model holding a few MiB carries the 256 MiB floor (erring high; subtract the reported
  `planBackingBudgetBytes` — [limitations](../limitations.md)).
- The default is a constant that ignores the device (on a 4 GB device at most +256 MiB, estimated).

## Addendum (2026-10-10) — revision of decision 4: the estimate computes bucket shapes into the peak max

Decision 4's "bucket shapes are not scenarios but the budget bounds them" is revised.

- The estimate (`estimateGraphMemory`) computes each physical row count of `generation.chunkBuckets`
  like prefill / decode and puts its `ioBytes + workspaceBytes` into the inner max of
  `peakAccountedBytes` (`max(budget, largest scenario, largest bucket shape)`).
- Bucket shapes are not listed in `scenarios`; the public list stays prefill / decode plus extras.
- The budget term still bounds bucket shapes held together with other shapes.
- gemma4's `estimateSessionMemory` passes the buckets the sequence runs (steps below the estimated
  `chunkLength`); not to the drafter's estimate (a borrowing context runs only M = 1).

Why: the budget bound assumed intermediates grow monotonically with the physical row count M, peaking
at the largest prefill shape. Row-block splitting (the equal split of ADR
[0060](0060-row-block-attention.md); `planStateAttention` for states-shape attention too) breaks this:
where the block count goes from 1 to 2, a shape with fewer rows uses more temporary memory. A budget
smaller than that bucket shape (including 0) let it escape the count. Computing every allowed row count
does not rely on the assumption, even if the split rule changes.

## Addendum (2026-10-10) — a GPU-side batch failure also retires the backing its enqueue built

Decision 2 is extended to batches. A non-fatal OOM (`createBuffer` returns an invalid buffer without
throwing) leaves the enqueue successful and shows only in the errorScope of `finish()`, so the enqueue
body's catch (immediate retirement on host-side failure, as before) does not see it.

- An enqueue records the backing it built by key and generation number `build`, and on acceptance
  registers a finalizer for a failed settlement (with or without a generation lease).
- Retire only when the section body failed (`BatchFailureOrigin` `"gpu"`: an errorScope reported a
  failure, a scope was discarded uninspected, or the device was lost). A batch whose body passed and
  that settled only on one enqueue's host-side failure retires nothing (other enqueues' backings are
  valid; the failing enqueue's own backing was retired by its catch).
- MUST: retire only when the generation number matches. It differs from key matching only if a later
  run rebuilt the same key before the failed settlement (lock release vs finalizer order depends on
  microtask depth) — a defence against a hard to reach race.
- MUST: a failed batch settlement does not destroy. The retired object goes to pending destroy and
  `#destroyRetired` destroys it in one link on the Session's serialization chain. A later run may
  already be encoding at settlement; each link finishes its submit / discard before leaving, so the
  chain's head has no unsubmitted encoding (ADR 0004 flush-before-destroy). The VRAM returns in the
  chain's first link, so an enqueue issued after `finish()` settles gets it back before rebuilding;
  ordering against an enqueue issued without waiting is not guaranteed (the same hard to reach race).
- Check: `gpu_plan_backing_test.ts` (a GPU-side batch failure retires only the newly built backing,
  returned once in the chain's first link and rebuilt by the next batch; a failure that hit an
  existing backing does not rebuild; a host-side-only settlement keeps other enqueues' backings). The
  failure is injected as OOM at the errorScope pop (`packages/runtime/tests/helpers/oom-injection.ts`).

## Addendum (2026-10-10) — a GPU-side failure rolls back the params and plans its execution unit added

Params (kernel-constant buffers) and prepared plans are also registered with an invalid buffer when
their allocation hits a non-fatal OOM; without rollback, a retry after memory frees up hits it again.

- Each execution unit (one run / one enqueue) records the params (cache entries) and plans it added and
  removes them only on a GPU-side failure (params leave the cache but are not destroyed — the weights
  arena reclaims them at `Session.dispose`). Existing healthy entries stay.
- GPU-side failure: for a run, an errorScope reported a failure (a run that failed before the pop also
  inspects the scopes it discards — `discardFailureScopes` reports a catch), the pop failed, or the
  device was lost (after loss a pop resolves to null, so inspection is no evidence). For an enqueue,
  `finish()` reports a failed section body (`"gpu"` above).
- Host-side failures remove nothing (run and enqueue): derivation throws, input checks, `copyOutputs`
  checks, the readback limit, generation checks (`assertGenerationRun` of a miss run, etc.). Those
  params and plans are valid; removing them would allocate new params on every repetition of a failure
  that always fails (bad input) and pile them up until dispose.
- Readback failures depend on the path. On the single-fence path staging allocation and copy share the
  run body's errorScope, so their failure is GPU-side and removed. A failure after the run body's scope
  passed is not removed: waiting on `mapAsync`, and what `#readOutputs` of the two-stage path reports
  from its own scope (staging allocation, copy); device loss is removed by the rule above.
- MUST: matching is by identity (plans are objects, params are buffers), a defence against a hard to
  reach race like the generation number above.
- MUST: removing a plan also retires that key's backing (as LRU eviction does — the backing baked the
  plan's recipe, i.e. the removed params). Normally the same failed unit built that backing and it is
  already retired; this matters only if a later run hit the plan and built a backing before the batch
  settled. Then a backing may remain on a key without a plan, harmlessly (invalid baked params make
  that run fail and retire it; valid ones live until dispose with the same contents).
- Checks: `gpu_params_cache_test.ts` (a GPU-side failed run / batch removes only what it added and
  keeps existing plans; repeating an enqueue that fails on bad input keeps the params allocation count
  `diagnostics().weights.allocCount` and the plan) and `gpu_state_execution_test.ts` (repeating a miss
  run that fails in `assertGenerationRun` keeps the allocation count; if the errorScope discarded in the
  same run caught a non-fatal OOM, the plan and params are removed and rebuilt on the next run).

## Addendum (2026-10-24) — the run that builds a backing first waits for released memory

Wan2.2 at 1280×704×121 ran out of device memory under Node + Dawn on the 12 GiB RTX 3080 Ti in the DiT
stage's second forward — the hit run that builds the backing. The preceding miss run had destroyed its
arena (4.45 GiB), but Dawn still held it while the new regions (1.6 + 2.1 + 0.6 GiB) were allocated on
top of 4.7 GiB of weights.

- Dawn returns a destroyed buffer's memory only when a **later** submit completes. Probe (4 GiB
  destroyed after its last submit): still held with no wait, after a 2 s sleep and after
  `onSubmittedWorkDone` alone; returned after `queue.submit([])` + `onSubmittedWorkDone`.
- Decision: a run whose key has no retained backing issues an empty submit and waits for it (raced with
  device loss) before `#activateBacking` allocates. Once per built backing; miss runs and runs on a
  retained backing keep H-1's single `mapAsync` fence. The same shape as the fence after the weight
  upload (ADR 0108 decision 9) and the models-side `settleReleasedMemory`.
- Not taken: a wait only in Wan's DiT loop (every family builds backings the same way); a wait in every
  `RunArena.destroy` (a fence per run, against H-1); building on miss runs (one-shot runs pay no slot
  memory — decision 1); waiting only after an out-of-memory error (Chrome's D3D12 pages to system memory
  instead of failing, so the overlap would stay and turn into a slowdown).
- Unchanged: decision 4's overlap of backings evicted within the same run (destroyed after that run's
  flush), and `enqueue`, which builds without the wait (a batch issues no fence).
- Checks: `gpu_plan_backing_test.ts` (the empty submit settles before the first region is allocated; no
  empty submit in a miss run or on a retained backing — red with the wait removed); the fence counts of
  `gpu_run_fence_test.ts` / `gpu_submit_backpressure_test.ts` count the one wait of the building run.
  On the RTX 3080 Ti: the default preset's 2-step 1280×704×121 completes (peak 9,300 MiB), the 121-frame
  A/B passes (relRMS s16 4.915e-2, online 4.675e-2, band 9.9e-2) and the 121-frame pipeline case passes.
