# 0070: Two-phase shard loading and memory admission

- Status: accepted (2026-08-17 — delegated check, no global ruling; Codex review rounds 3 to 5 folded
  in, go at round 6); partly superseded: decisions 1–4 (shards, two-phase loading with a sequential
  surface, a Session built from shards, per-shard error scopes) were retired by
  [0108](0108-container-format.md) / [0109](0109-manifest-v5-container.md) (step 3d `9e905d9f`,
  2026-09-24; the error-scope interval moved to 0108 decision 9's block / part granularity), and
  decision 2's re-hash on cache hits was changed by [0080](0080-hub-fetch-cache-050.md) decision 1
  (2026-08-28). **In force: decision 5 (the demand-side estimator and `unaccounted`) and the
  auxiliary-resource breakdown (2026-09-12).** On 2026-10-18 staging was withdrawn from decision 5's
  estimator categories and decision 6's release-wave reservation was withdrawn.
- Date: 2026-08-17
- Related: ADR [0038](0038-manifest-v1.md) (manifest and hub fetch layer) /
  [0041](0041-manifest-v2.md) / [0063](0063-safetensors-physical-layout.md) /
  [0004](0004-execution-model.md) (permanent error scopes) /
  [0066](0066-generation-context-state-slots.md) (state capacity — a new admission category) /
  [0069](0069-packed-w4-storage.md) (i4 storage — the distribution size) /
  [0089](0089-memory-limits-preflight.md) (deterministic checks against absolute limits before
  allocation)
- Evidence:
  [research/2026-08-17-autoregressive-references.md](../research/2026-08-17-autoregressive-references.md)
  §5 ("survey §n" below)

## Context

At the time every file was held as a `Uint8Array` and passed to `openModel` as one `ArrayBuffer` —
whole-file host retention baked into both hub and runtime (survey §5.1). For an acceptance-class
model (1 GiB-class even in i4, 4 GiB-class in f16), holding the distribution files and the GPU
resident copy in RAM at once does not work. The only browser precedent was tvmjs's two-phase load,
and **no browser implementation verifies weight integrity** (Karume's full sha256 check is a
differentiator — survey §5.1).

## Decision

### 1. Distribution: sharded weights

**(superseded by ADR 0108 / 0109 on 2026-09-24: the container's blocks and parts replace shards.)**
A component's safetensors could be split into a graph shard and weight shards, each a self-contained
safetensors with its `{path, size, sha256}` triple, the assignment table canonical in the manifest.
Declaration completeness (the union of `initializer.tensor` and `storage.scale` names — missing,
duplicate or extra fail loudly) was checked after all shards. The co-shard MUST (a weight and its
companion scale in the same shard) moved to rule ③ of [container-v1](../container-v1.md) §5 (the
companion scale's block is in the same part as its tensor; for a piece list, the same part as
piece 1).

### 2. hub: two-phase loading and a sequential surface

**(superseded by ADR 0108 / 0109 on 2026-09-24.)** Phase 1 put all shards into the persistent cache
(parallel, not in RAM); phase 2 handed shards one at a time to the caller after a sha256 check, then
let go of them. The re-hash on cache hits **(superseded by ADR 0080 decision 1 on 2026-08-28: a cache
hit compares with the hash recorded when the entry was stored; silent bit rot after storage is not
detected, a cost 0080 accepted)**.

### 3. runtime: building a Session from shards

**(superseded by ADR 0108 decision 9 on 2026-09-24.)** The graph shard came first (structural
contract, eligibility, weight intake), then each weight shard went "CPU expansion (ineligible only)
→ GPU upload → release", with an explicit submit per shard before the fence so the implementation's
staging is released. A failure in the middle destroyed every GPU resource built so far and published
no partial Session (the same boundary is now 0108 decision 9's, `session-build.ts`).

The rule "do not release the CPU side before the transfer completes (release after the fence)"
**(superseded by the addendum of 2026-09-24 and ADR 0108 decision 9)**: the CPU bytes may be
released as soon as `writeBuffer` returns. The WebGPU spec's `GPUQueue.writeBuffer` copies at the
call ("Let dataContents be a copy of the bytes held by the buffer source data."), so the order copied
from tvmjs was never a correctness condition. `packages/runtime/tests/gpu_write_buffer_copy_test.ts`
pins it on hardware (poisoning the source right after the call gives bit-identical output). The
fence (explicit submit + completion wait) remains only to release staging, per part.

### 4. Error-scope interval per shard

**(superseded by ADR 0108 decision 9 on 2026-09-24: the synchronous interval is per block / part.)**
The interval was redrawn per shard: one shard's `writeBuffer` calls inside one synchronous error-scope
interval, with awaits outside it. The LIFO invariant ("no await inside the interval") is unchanged;
only the granularity moved.

### 5. Admission: a demand-side estimator plus diagnostics (no absolute guarantee)

- **The estimator reports only demand-side totals by category:** resident weights (compressed and
  expanded), state slots (ADR 0066 capacity), and the plan backing and transients of each scenario.
  **It does not compare against free memory** — WebGPU exposes neither total nor free VRAM (survey
  §5.2; llama.cpp's exclusion of 0/0 devices is the same precedent: no guessing where no budget is
  available).
- Withdrawn 2026-10-18: staging as an estimator category. The implementation staging of
  `queue.writeBuffer` depends on the implementation and is listed in `unaccounted` (`UNACCOUNTED` in
  `packages/runtime/src/runtime/estimate.ts`); a caller that must plan ahead adds the largest weight
  part explicitly (`peakAccountedBytes` + `maxPartBytes` + margin — ADR 0112 addendum of
  2026-09-27, `packages/models/src/anima/residency.ts`).
- Diagnostics carry an **`unaccounted` field** (as llama.cpp fit's output — the format admits that the
  estimate is not an absolute guarantee; survey §5.2).
- **The final check is the existing out-of-memory error scope** (the out-of-memory + validation pair
  of `pushFailureScopes` in `packages/runtime/src/gpu/error-scope.ts`; nothing new). The estimator is
  an advance diagnostic for building a Session or a generation context and does not stop execution
  when exceeded. ADR 0089 adds deterministic checks against the device's absolute limits before
  allocation.
- Thin public surface (ADR 0008): one estimate function (`estimateGraphMemory`, and
  `prepareContainer(...).estimate()`) and the existing diagnostics; no new `SessionDiagnostics`
  fields were needed (the measured counterparts already exist).

Addendum (2026-09-27, ADR [0112](0112-anima-transformer-residency.md) addendum of 2026-09-27 /
perf-ledger H-35): "no comparison against free memory" still holds for **guessing**. Free memory
measured by a non-fatal trial allocation (`fitsHeadroom` in `packages/runtime/src/gpu/headroom.ts` —
allocate that amount, destroy it at once, wait for the release, report success) runs the same check
as a real allocation failure (wgpu's budget check), so it may drive an advance decision (evicting
anima's resident DiT). It is not exposed as a free-memory value (binary search for the ceiling stays
a diagnostic). The final check is still the out-of-memory error scope.

### 6. Reservations

- Streaming CPU expansion of i4 → f32 per shard was part of decision 3 (now ADR 0108).
- Withdrawn 2026-10-18: the large-download limits preflight "in the release wave" (done by ADR 0089,
  2026-09-01). The reservation of hub Range parallelism + prefetch (perf-ledger L-3, parked) as an
  optimisation inside phase 1 stays.

## Consequences

- Withdrawn 2026-10-18: the release conditions (manifest shard fields before HF publication; shard
  surface for local experiments only) — 0.14.0 is published and the manifest is `karume/5` (ADR
  0109).
- The acceptance checks of the shard surfaces (GPU-resident bytes A/B against the whole-file
  surface, RAM peak, error-scope injection, cross-shard completeness) retired with decisions 1–4.
  The cache-hit sha256 check among them is superseded by ADR 0080 decision 1 (2026-08-28).

## Addenda

- 2026-08-19 (wave G): implemented decisions 1–5 as `streamAssets` / `createSessionFromShards` /
  `estimateSessionMemory`. The shard surfaces are gone with ADR 0108; the estimate entry is now
  `estimateGraphMemory` / `prepareContainer(...).estimate()`, and the mapping of estimate fields to
  diagnostics is canonical in the docstring of `packages/runtime/src/runtime/estimate.ts`.
- 2026-08-29 (R1 integration): the estimator returns `AdmissionReport` (`resident` + `scenarios[]`,
  prefill / decode computed separately; `peakAccountedBytes` = resident + max(scenarios)). ADR
  [0095](0095-plan-backing-budget.md) (2026-09-07) made the backing budgeted: the total is
  `resident + max(budget, max(scenarios))`, and the retirement window ("retired → destroy") is listed
  in `unaccounted`. The shard identity, two-stage boundary and writer split rules of that addendum
  were replaced by ADR [0081](0081-shard-spec-v2.md) and then ADR 0108.
- 2026-08-30 (review M1-2): the models side does not hold the graph shard after admission — a
  pipeline keeps only the `IrGraph`, not the `PreparedModel` (`packages/runtime/mod.ts`; measured
  `external` 2,461.6 MiB → 0.7 MiB on anima Base f16).
- 2026-08-31 (distribution source — ADR [0086](0086-distribution-source.md)): phase 1 is an optional
  capability of the source (`prefetchFile` in `packages/hub/src/source.ts`). A source without it (a
  local directory) is valid and `prefetchAssets` is a no-op there (0086 addendum of 2026-10-11). A
  cross-repository target that drops phase 1 breaks the source contract and fails loudly
  (`originFor` returns another coordinate of the same source, never one with fewer capabilities).
- 2026-09-02: a reusable read buffer (one allocation of the largest shard, RAM peak factor 1 —
  [research/2026-09-02-shard-size-ram-peak.md](../research/2026-09-02-shard-size-ram-peak.md)) with
  tensor pieces (ADR [0090](0090-shard-spec-v3-tensor-pieces.md)) and the fetch layer's `into`
  (`@hdae/fetch-cache` 0.6.0). Retired on 2026-09-24 (addendum below).

## Auxiliary-resource breakdown (2026-09-12)

`AdmissionReport` gains an optional `auxiliaryBytes`. Omitted means 0, so the runtime's estimate of a
single Session and existing results do not change. When models hold auxiliary Sessions or resident
inputs / outputs, their extra cost goes in this field. It does not overlap the `resident` /
`scenarios` breakdown and is not mixed into weights or KV state.

The total is `weights + state + max(budget, largest scenario) + (auxiliaryBytes ?? 0)`. The first
small-output implementation added the extra amount to the peak only, so the total could not be
reproduced from the breakdown; an existing test caught the mismatch. The fix shows the added resources
in the public breakdown instead of shrinking the total or loosening the equality check.

Gemma's temperature-0 decode puts its auxiliary selector and resident inputs / outputs in this field;
a pipeline whose small output is disabled for diagnostics omits it. A pipeline capable of speculative
decoding can also open non-speculative conversations, so it keeps this auxiliary cost after
composition. See [ADR 0083](0083-generation-api-surface.md) and
[the verification record](../research/2026-09-10-codex-mtp-optimization.md) (section of 2026-09-12 on
small-output temperature-0 decode).

## Addendum (2026-09-24) — reusable buffers retired; RAM peak accounting moves to container-v1 §11

- **The reusable read buffer of the 2026-09-02 addendum is retired.** The buffer loan
  (`FileReadOptions.into`, `DirectoryAdapter.readFileInto`, the HF source's `into` wiring), the
  sequential surface `streamAssets` and `ModelShard` were deleted with ADR 0108 step 3d
  (`9e905d9f`). Lending a buffer to the fetch layer was not adopted (0108 addendum 5, item 4: it
  conflicts with the hub's scan-type views and serialises PLE's parallel row reads). "Factor 1" is
  not a property of the current path.
- **The RAM peak target "O(largest shard)" of decisions 1–3 and the 2026-08-31 addendum is replaced
  by accounting that depends on the source type (seek / scan)**, canonical in
  [container-v1](../container-v1.md) §11 (the formula is not copied here). Decision 3's fence governs
  staging only.
- **The co-shard MUST of decision 1 moved to rule ③ of container-v1 §5.**
- `parseSafetensors` no longer takes a separate file length (`80c4cd74`); it reads the whole
  `ArrayBuffer` passed as the file.

## Addendum (2026-11-06) — `fitsHeadroom` settles its release on Dawn

The trial allocation was studied on wgpu, whose release happens at a device poll. **On Dawn the memory of a destroyed buffer
comes back only when a later submit completes** (ADR [0095](0095-plan-backing-budget.md) addendum of 2026-10-24). So the probe left its
own trial allocation on the device, and the allocation it had just approved failed: `capacity: "fit"` (ADR 0083 addendum of
2026-11-06) hit out-of-memory on a 12B chat on the RTX 3080 Ti.

`fitsHeadroom` now waits for the poll, then submits an empty command list, then waits again. The order is not free. wgpu's
submit-side 99% check runs before its release, so a submit before the poll would run that check while still holding the trial
allocation.

The measured boundary of an otherwise empty RTX 3080 Ti under Dawn: 11 GiB true, 12 GiB false, nothing left behind. The largest
amount repeated is pinned by `packages/runtime/tests/gpu_headroom_test.ts`. Chrome has not been checked.
