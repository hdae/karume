# 0042: Two-phase execution plan; derived recipes resident in the Session, keyed by bindings

- Status: accepted (2026-08-10; PNG and WAV sha256 checks unchanged, 0 fixture diffs, verify 749/0);
  partly superseded by [0093](0093-transient-liveness-packing.md) (decision 4, slot packing,
  2026-09-05) and [0095](0095-plan-backing-budget.md) (decision 4, backing capacity, 2026-09-07)
- Date: 2026-08-10
- Related: [0004](0004-execution-model.md) (arena invariants, flush-before-destroy) /
  [0040](0040-fusion-pass.md) (fusion pass and its counters, §3) /
  [0021](0021-gpu-timing-diagnostics.md) (resident diagnostics) /
  [0022](0022-gemm-register-blocking.md) (no f32 runtime autotuning — this cache never changes
  variant selection) / recon: [research/2026-08-10-op-timing-stats.md](../research/2026-08-10-op-timing-stats.md)
  §7 to §8, [large-designs](../research/2026-08-06-kernel-triage/large-designs.md) C/D/E

## Context

After the first HOST-006 wave, a steady-state run still exposed about 1.1 s of host time (11% of
the wall time, w8a8-1024): createBindGroup at 44.4 ms/step, plan and fusion recomputed every run,
WGSL strings regenerated per dispatch, params key strings, and the encode loop. The recon of
2026-08-10 showed that planning (bindSymbols → planGraph → planFusions) and recipe derivation are
pure functions of the graph and the resolved bindings, that the plan holds no GPUBuffer, and that
bindSymbols forces input shapes to match the bindings: the resolved bindings are a necessary and
sufficient cache key.

## Decision

### 1. The encode layer has two phases: derivation and execution (`packages/runtime/src/runtime/recipe.ts`)

The derivation phase issues no GPU commands and touches no run-lifetime state. It produces a
StepRecipe list: per dispatch the pipeline, layout and params (direct references to
Session-resident objects) and the binding *procedure* (value names, node-local temp ids); per step
the output allocation spec, temp lifetimes and the value names to release. The execution phase
replays it through one bookkeeping path (today `executeBakedPlan` in the same file), keeping the
invariants of ADR 0004.

MUST: a recipe holds no GPUBindGroup and no run-lifetime buffer, only Session-resident objects and
"what to bind where". Otherwise the cache of decision 2 cannot hold.

### 2. The prepared plan (PreparedPlan) is resident in the Session, keyed by the resolved bindings (LRU)

- Key = the bindings values in `graph.symbols` declaration order; a graph without symbols has the
  key "". The container is the SessionState (no module scope). It holds only what later stages
  read: `shapes`, `recipes`, `fusions` and `generation` (added with ADR
  [0066](0066-generation-context-state-slots.md); `packages/runtime/src/runtime/session-build.ts:242`).
- **A plan is registered only after the derivation phase completes**; a partial recipe would make
  the next run with the same bindings silently execute a truncated step list. A registered plan,
  and the params its execution added, are removed again when that execution reveals a GPU-side
  failure (error scope failure, failed pop or device loss for a run; a failed batch body in
  `finish()` for enqueue), so no reference to an invalid buffer from a non-fatal OOM survives. A
  host-side failure removes nothing — ADR [0095](0095-plan-backing-budget.md), addendum 2026-10-10.
- A hit run skips planGraph, planFusions and recipe derivation. **bindSymbols (input shape
  validation) still runs on every run**; an exact key match only skips repeating the same check on
  the same input, so fail loudly is not weakened.
- The LRU capacity is a constant, not a setting: 4 at first, 8 on 2026-09-07 (prefill buckets of
  ADR 0066 addendum 10), 12 on 2026-09-08 (ADR [0096](0096-speculative-decoding.md);
  `PREPARED_PLAN_CAPACITY`, `packages/runtime/src/runtime/executor.ts:465`). The cache only exploits
  consecutive runs that reuse bindings. Eviction drops host objects only: GPU resources belong to
  the params cache and the PipelineCache (destroying them here would hand another plan a destroyed
  reference).

### 3. Resident diagnostic `lastRunPrepared {hit, cachedPlans}`

A miss keeps the values correct and only loses performance silently, so this is the only
observation point (on a par with `lastRunFusions`). A hit run reports the cached counts in
`lastRunFusions` (ADR 0040 §3) and `{0, 0}` in `lastRunParams` (the derivation did not run).

## Consequences

- **Narrower WGSL byte check (accepted trade-off).** A hit run regenerates no WGSL, so the
  PipelineCache check "same key → same bytes" runs only on a miss. Determinism rests on the codegen
  snapshots (279 fixtures in `packages/runtime/tests/fixtures/wgsl/` on 2026-10-18) plus that check.
- **Narrower integration coverage of discard.** A derivation throw happens before anything is
  queued, so "failure mid-encode leaves pending work" can no longer be provoked from outside; the
  `SubmitScheduler.discard` unit tests remain (`packages/runtime/tests/gpu_submit_test.ts`).
- Params-cache reuse is observed on "duplicates within one run, plus re-derivation after an
  eviction" (`gpu_params_cache_test.ts` pins the independent lifetimes of the two caches).
- Pipeline creation is serialized before the first submit, so the first run of a cold Session may
  get slightly longer.

## Addendum (2026-08-10) — wave 2: fixed transient slots and baked bind groups

### 4. GPU backing of transient slots (lazy build on the first hit)

(Superseded by ADR [0093](0093-transient-liveness-packing.md) decision 1 on 2026-09-05 and ADR
[0095](0095-plan-backing-budget.md) on 2026-09-07: slots are placed by liveness intervals and
first-fit offsets (`packages/runtime/src/runtime/transient-plan.ts`); `derivePlanSlots` and the
RunArena pools are retired; the backing is a byte-budgeted LRU set, default 256 MiB, budget 0 =
capacity 1. Lazy build, retirement → destroy after flush and the footprint invariant still hold.)

Original decision: `derivePlanSlots` replayed the **size-class LIFO of the RunArena** over the
recipes (custom packing forbidden, so no new VRAM peak); capacity one active signature. The backing
(slots, input buffers, baked bind groups) is **built lazily on the first hit run**, so a one-off run
pays no slot memory. Destruction goes through the retirement queue after flush. A newly built
backing whose run fails is retired (recovery from a transient OOM; an existing backing stays).

### 5. Baked bind groups and fixed inputs

Every binding target is fixed across runs (resident, slot, resident input), so all GPUBindGroups
are baked at build time and a backed run only queues dispatches. Inputs are written into
backing-owned buffers with `writeBuffer` each run; overtaking is safe because of #chain
serialization, runs returning only after flush/readback, and input writes preceding all encoding
(pinned by a concurrent-run test). Leftovers are overwritten correctly by full-write (ADR 0014).
Diagnostic `planBacking {residentBytes, buildCount}`: rebuilding on every run when signatures
alternate is the only silent regression, and `buildCount` shows it.

## Addendum (2026-08-10) — measurements

ABBA over the four e2e_anima cases (commits `d7626c8` → `1751f3c`): wave 1 and the baked bind
groups were neutral, fixed slots gained ×1.02 to ×1.04; the rest of the host cost overlapped with
GPU work, so staged execution (large-designs D/E) would gain almost no wall time. Details:
[research/2026-08-10-op-timing-stats.md](../research/2026-08-10-op-timing-stats.md).

## Addendum (2026-08-18) — wave D: separate bake units and a backing generation id

The implementation of ADR [0066](0066-generation-context-state-slots.md) decision 5 (`22b5f64`)
revised the backing (a run without generation does not change by one byte):

1. **The bake unit follows the owner of what is bound.** `bakeBindGroups` bakes the dispatches
   that bind Session-owned objects; those that bind GenerationContext-owned objects (state slots,
   the logical-length uniform) are baked by the context (`bakeGenerationBindGroups`) per (context,
   backing) pair. Both share `bindsGeneration` and `bakeGroups`.
2. **ActiveBacking carries a monotonic generation id (`build`).** The context side MUST match it
   before using its bind groups: a retired backing's buffers live until the run's cleanup, so a
   stale bind group passes validation while **the values silently change** (fault injection in wave
   D-4: every output wrong, no exception). Diagnostic: `stateBacking.rebindCount`.

## Addendum (2026-08-29) — new owner of the PipelineCache

The PipelineCache moved to **GpuContext ownership (one per device)**; Sessions on a device share
it, and `PipelineKeyConflictError` (same key, different WGSL) applies across Sessions. The params
cache stays in the Session. Diagnostics: `pipelineCount` (keys this Session used) and
`devicePipelineCount` (device total).

## Addendum (2026-09-11) — asynchronous compilation

`PipelineCache` uses `createComputePipelineAsync` (`packages/runtime/src/gpu/pipeline-cache.ts:148`)
and `RecipeBuilder` derives all steps concurrently. API, WGSL, pipeline keys, recipe order and
dispatch order do not change; a run that hits a prepared plan does no extra work.

- Input names are resolved and output names registered in step order before the first `await`.
  Registering all output names up front MUST NOT be done (it would accept forward and self
  references).
- Module creation and the issue of the compilation share one error scope, popped before the wait;
  the layout fetch has its own scope. One push/pop pair MUST NOT span the wait.
- A rejection handler is attached at once; an issued compilation is awaited to settlement even when
  module validation fails first. A pending creation is shared by key and deleted only on failure.
- `GPUPipelineError.reason` `internal` / `validation` map to the existing typed errors with the
  cause and `getCompilationInfo()` diagnostics kept; other reasons are not guessed into validation.
  A failure to fetch the diagnostics is recorded as a supplement and does not overwrite the
  compilation cause.
- Recipes are returned after `Promise.allSettled`; failures are chosen in step order, no params
  write outlives the run's cleanup, and a partial recipe is not cached.

Measured ([research/2026-09-10-codex-mtp-optimization.md](../research/2026-09-10-codex-mtp-optimization.md),
section on async compilation, 2026-09-11): first response of QAT E2B on Chrome / RTX 37 to 43%
shorter; parallel recipes alone did not help; Deno about flat; Apple M2 not measured.

A dummy warm-up, a new option or an explicit prewarm API SHOULD NOT be added for this (they add
user steps and public surface). (Re-graded 2026-10-18 to SHOULD: the decision rests on measurements
of that time; a first-run cost left on the browser targets, Node + Dawn (ADR 0127) or M2 may reopen
it.)
