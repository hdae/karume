# 0112: Anima DiT residency — opt-in, per request, evict and reload on OOM

- Status: accepted (2026-09-26 — measured and adopted, phase 2 "DiT speed" B3 of the speed-up
  plan, approved by the maintainer; on the Arc B570 the gain from the second generation on, 2.45 s
  per generation = 10.4 % of wall time, beat the kill criterion; stays opt-in). The rejection of
  explicit staging re-graded 2026-10-18 to MAY, the headroom margin and the no-submit rule of
  `fitsHeadroom` to SHOULD (2026-09-27 addendum)
- Date: 2026-09-26
- Related: [0016](0016-anima-chain-export.md) (graphs are opened and closed one at a time — stays the
  default) / [0095](0095-plan-backing-budget.md) (budgeted backing retention — bounds residency
  across resolutions) / [0107](0107-model-input-error.md) (error type of unknown spellings) /
  [0033](0033-vae-fixed-tile-decode.md) (VAE always tiled — why the chain peak moved to the DiT stage)
  / perf-ledger H-4, L-9, H-35. Code: `packages/models/src/anima/residency.ts`, `pipeline.ts`.
  Measurements: [research 2026-09-26](../research/2026-09-26-anima-residency-bench.md).

## Context

`AnimaPipeline.generate` creates and discards a Session per stage (text_encoder → text_conditioner →
transformer → vae_decoder). The default VRAM assumption is "the largest single stage"; since the VAE
is always tiled, the chain peak is the DiT stage (2,647 MiB by runtime accounting for the default
preset at 1024²).

Repeated generate calls on one pipeline (browser apps, servers, e2e) reload the DiT weights,
re-derive the plan and rebuild the intermediate buffers (backing) each time: **1.87–2.25 s per
generation (15–19 % of wall time)** on 2026-09-10 (RTX 3080 Ti, pre-container, turbo 1024², CFG 1).

## Decision

1. **Opt-in residency**: `AnimaPipelineOptions.residency?: "per-stage" | "transformer"` (default
   `"per-stage"` = the old behaviour). An enumeration, not a boolean, so text-stage residency can be
   added later. An unknown spelling throws `ModelInputError` at admission (before any weight part is
   fetched). Construction still opens no Session; the resident DiT is **created in the DiT stage of
   the first generate**.
2. **Per-request decision**: `AnimaGenerateRequest.residency?` (same enumeration); effective value =
   request ?? construction option. It means "keep the DiT Session **after** this generate". A
   resident DiT present at the start of a generate is used whatever the effective value (no reload);
   making only the last of a series `"per-stage"` releases it after that generate. **Even when the
   generate fails** (a text stage before the DiT stage fails, `onEvent` throws on a `stage` event,
   …), an effective `"per-stage"` releases the carried-over DiT and reports `released` / `request`
   (otherwise VRAM thought released stays until the next generate or dispose). A request rejected by
   input validation touches neither the GPU nor the residency state.
3. **Evict and reload on OOM** (the maintainer's "swap", meaning confirmed 2026-09-26: "if an extra
   load hits OOM, release and reload the target model"). With a resident DiT, when another stage's
   (text_encoder / text_conditioner / vae_decoder) Session construction or run, or the DiT-stage run
   of a DiT **carried over from an earlier generate**, throws `GpuOutOfMemoryError`:
   ① dispose the resident DiT; ② await `device.queue.onSubmittedWorkDone()` (on Intel / wgpu the
   release by `destroy()` is deferred to the next poll — docs/known-issues.md, Intel Arc B570 section;
   raced against device loss); ③ report `evicted` / `out-of-memory` with a `residency` event; ④
   **re-run that stage once from the start** (a DiT stage is rebuilt per stage). A failure of the
   re-run is thrown as is. The test is the type (`instanceof GpuOutOfMemoryError`); validation errors
   and device loss are not evicted (the real cause would be buried under the second failure).
   - **An OOM in the run of a DiT created in the same generate is not evicted** (no downgrade, no
     report; folded like a per-stage failure and the original error thrown). The pipeline then holds
     only that DiT (the text stages are folded), the same VRAM layout as per-stage operation, so a
     downgrade and a GB-scale reload would retry the same layout. That DiT enters the resident slot
     only after its stage succeeds.
   - ② rests on a raw-WebGPU probe (B570, 2026-09-26 — research 2026-09-26): after destroying 1 GiB
     at full memory, awaiting `onSubmittedWorkDone` alone let the same 1 GiB be reallocated (an 11 ms
     wait; without it, OOM). No fixed sleep is added. The device loss later seen on the eviction path
     came from staging OOM (2026-09-27 addendum), not from the wait.
   - If ① or ② throws, the downgrade still applies: report ③ (otherwise the loss stays invisible),
     then throw an `AggregateError` led by the original OOM (no re-run).
4. **Downgrade**: an evicted pipeline does not keep a resident DiT for the rest of its life. A later
   `"transformer"` request reports `released` / `downgraded`. **No silent fall to the slow path** (the
   event is required).
5. **Discard on failure**: an exception in the stage of a DiT carried over from an earlier generate
   (including an abort by `onEvent` throwing in the denoise loop) discards the resident DiT, reports
   `evicted` / `failure` and throws the original error. A Session that may be broken is not carried
   into the next generate. No downgrade. A failure in the stage of a DiT created in the same generate
   is a per-stage failure (before it enters the slot — folded, not reported). A throw from
   `stage` / `vae-tile` / `residency` notifications is outside the DiT stage body and does not discard
   the resident DiT (only decision 2's release applies). A clean-up failure is an `AggregateError` led
   by the original error, with the message of `CLEANUP_FAILED` in `residency.ts` (also used for the
   per-stage DiT); when a subscriber rethrows the same error (the `signal.throwIfAborted()` pattern),
   it is not listed twice.
6. **Event**: `AnimaGenerateEvent` gains `{ kind: "residency"; component: "transformer"; action;
   reason }` (`action` = `retained` / `released` / `evicted`; `reason` = `request` / `downgraded` /
   `out-of-memory` / `failure`, plus `headroom` from the addendum). A default per-stage generate
   unrelated to residency emits none (existing event sequences unchanged). The `stage` event now means
   "stage start / end" instead of "before Session construction / after release" (type unchanged). A
   `transformer` stage using a carried-over DiT with effective `"transformer"` has no load or release
   between start and end; with effective `"per-stage"` the release is inside; after an OOM eviction
   the release and the per-stage load are inside, and the stage's progress events (`denoise-step` /
   `vae-tile`) restart from 1.
7. **Dispose**: within the serialisation chain the resident DiT is disposed before the GPU is
   destroyed (flush-before-destroy). With a shared `options.gpu` the resident DiT still belongs to the
   pipeline and is folded (otherwise it stays on the shared device). On a lost device the flush in its
   disposal may fail and dispose may reject (an internally acquired GPU is still destroyed).
8. **Decisions are pure functions plus a small state machine** (`transformerSource`,
   `decideAfterTransformer`, `decideAfterFailedGenerate`, `evictsForMemory`, `TransformerResidency` in
   `residency.ts`). The pure functions decide from the facts at the **start** of the DiT stage; the
   state machine only applies the `action` (one source of truth). An unreachable combination (a DiT in
   the slot while downgraded) throws instead of falling back. GPU-less unit tests pin the transitions
   with fault injection (`packages/models/tests/anima_residency_test.ts`).
9. **Numerics unchanged**: no sha256 reference value is added; every case is a twin of the turbo
   1024 / 512 values (the "DiT residency" section of `packages/e2e/tests/e2e_anima_test.ts`).

## VRAM (only the opt-in side rises)

| Quantity (default preset `f16+dit8-a8-attn8-s16`, 1024², CFG 1)                | Value                                                                  | Source                                    |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------- |
| Resident share (weights 1,870.9 + backing 768.1 + inputs 7.1 MiB)               | **+2,646 MiB**                                                         | 2026-09-10 runtime accounting             |
| DiT-stage peak of per-stage operation (= default chain peak)                    | 2,647 MiB                                                              | same                                      |
| Chain peak with residency (text / VAE stages on top of the resident DiT)        | about 3.9–4.1 GiB (estimate, medium confidence — no text-stage figure) | B3 report §2-4                            |
| DiT weights of the `f16` preset                                                 | 3,733 MiB (resident: over +3.7 GB during the text stages)              | 2026-08-05 final-perf-bench VRAM table    |

- **Not usable on 4 GB-class GPUs** (the DiT stage alone is 2.65 GiB; residency stacks the text
  stages on it). Per-stage operation is unchanged.
- A resolution change keeps only one backing over the 256 MiB budget (ADR 0095 decision 1), so the
  **steady-state** residency does not exceed "weights + the largest backing". Right after a switch the
  retired backing (or arena) lives until the flush, so a transient peak exceeds it (estimate: below
  the text-stage peak; the chain peak is unchanged).
- Eviction works only where OOM is reported as a type (`GpuOutOfMemoryError`). Where Metal's
  out-of-memory error scope stays silent (docs/known-issues.md, Metal section), residency beyond VRAM
  triggers no eviction, downgrade or event, and may show as slowness or device loss.

## Kill criterion

Measured on the Arc B570 (then the development machine) with block ABBA of per-stage / transformer
(first run of each block dropped; script `outputs/bench/karume/2026-09-26_anima-residency/bench.ts`,
untracked, Deno): close if the gain from the second generation on is **under 0.5 s or under 5 % of
the 1024² turbo wall time**. Verdict (2026-09-26, B570 / Linux xe / Deno 2.9.6): **adopted, opt-in,
default `"per-stage"`** — 2.45 s per generation (10.4 %), PNG sha256 identical to the reference value
(research 2026-09-26). Not measured: Chrome, and machines other than the B570.

## Alternatives considered

- **B: A plus releasing the backing after each generate** (+1,871 MiB) — deferred: needs a public
  backing-retirement surface (ADR 0095 addendum) to save 768 MiB; only if A misses 6 GB-class devices.
- **C: build the DiT in parallel with the text stages (prefetch)** — deferred: it helps one-generation
  processes too, but both sides compete for read + upload bandwidth (gain 0–1.3 s, low confidence,
  unmeasured). Independent of A.
- **Keep the DiT blocks in host RAM** — rejected: upload, derivation and backing are still paid, and a
  1.87 GB ArrayBuffer is heavy in a browser (perf-ledger L-9).
- **Throw OOM without eviction** — rejected (the maintainer's requirement itself).
- **No downgrade** — rejected: a machine short of VRAM would pay the eviction on every generate.
- **Re-run only the OOM'd run** — deferred: the text stages run in `Promise.all`, so per-run retries
  need concurrent eviction; on this rare path the per-stage re-run is simpler and gives the same
  pixels (the stage body is deterministic from the seed).

## Consequences

- The default (`"per-stage"`) keeps its event sequence, VRAM and numerics (the `onEvent-1024` check
  pins the sequence).
- `AnimaGenerateEvent` gains a member (subscribers that covered three kinds and narrowed with `else`
  fail type check — CHANGELOG Breaking). The public surface gains the types `AnimaResidency`,
  `AnimaResidencyAction`, `AnimaResidencyReason` (barrel and `./anima`).
- With residency, the `transformer` stage's start → end does not show the GB-scale load. Not
  measured: text / VAE stage peaks with residency.
- The B570's "eviction → re-run hits device loss" (2026-09-26, 6 GiB dummy load) came from staging
  OOM invalidating the device on Deno (wgpu); closed by pre-emptive eviction (2026-09-27 addendum).

## Addendum (2026-09-27) — decision 3 revisited: evict pre-emptively on measured headroom (H-35)

The investigation ([research 2026-09-27](../research/2026-09-27-h35-oom-device-lost.md): raw-WebGPU
probes and the wgpu-core / deno_webgpu / Dawn sources) is the source; only the decisions are here.

### Why reactive eviction (decision 3) cannot be relied on

- **There are two kinds of OOM.** A `createBuffer` OOM is non-fatal and the device lives (the one form
  WebGPU guarantees as "fails without side effects"). An OOM in the staging that `queue.writeBuffer`
  creates internally (likewise `writeTexture`, `mappedAtCreation` without MAP_WRITE, submit, bind
  group creation) makes wgpu-core lose the device on the spot (`handle_hal_error` → `lose()`).
- **Deno reports both as an ordinary `GPUOutOfMemoryError`**, and with no wgpu device-lost callback
  registered, `device.lost` resolves only at the next validated call (submit / writeBuffer /
  createBuffer / mapAsync; not `onSubmittedWorkDone`). Decision 3's eviction proceeds as if on a live
  device and the loss surfaces at the first upload of the re-run.
- The OOM itself is wgpu's pre-allocation budget check (97 % of the budget; above 99 % after submit /
  poll the device is lost — docs/limitations.md, Deno 97 % budget section). On the Arc B570 (ReBAR)
  the VRAM heap is also host-visible and the allocator puts staging there, so exhausted VRAM shows as
  staging OOM. Chrome (Dawn) passes OOM to the error scope only for createBuffer / createTexture /
  createQuerySet as well, but on that machine puts staging in system RAM (inferred from source, not
  measured).

### Added

1. **Runtime `fitsHeadroom(gpu, bytes)`** (public — `packages/runtime/src/gpu/headroom.ts`): allocates
   STORAGE buffers in equal pieces of at most `maxBufferSize`, destroys them at once, settles the
   release with `onSubmittedWorkDone` (raced against device loss), and returns `true` if they fit,
   `false` on OOM. Validation errors and device loss throw. It SHOULD NOT issue `queue.submit([])`
   inside (the 99 % check after submit runs before the release). (Re-graded 2026-10-18 to SHOULD: the
   rule rests on wgpu's (Deno's) poll order; not re-derived on Dawn.) The result is a fact of that
   moment, not a reservation.
2. **Anima's pre-emptive eviction** (`TransformerResidency.ensureHeadroom(need, probe, notify)`),
   measured at two points of a generate, only while a resident DiT exists:
   - before the text stages (with a carried-over DiT): need = the larger of text_encoder and
     text_conditioner;
   - after the `stage` end of `transformer` and before `vae_decoder` start: need = vae_decoder. This
     covers the first generate (its own DiT enters the slot and the VAE stacks on it) and a
     carried-over DiT that grew its backing at a new resolution. The VAE is not included in the
     earlier measurement because evicting after the DiT stage has already benefited is cheaper.
   - Need of one stage = `ModelComponent.estimate(...).peakAccountedBytes` (weights + the bound of the
     retained set, with the stage's own bindings and device limits) + `ModelComponent.maxPartBytes`
     (the largest weight part = staging alive until that part's fence) + a margin of 512 MiB
     (`HEADROOM_MARGIN_BYTES`).
   - **The margin SHOULD be 512 MiB** — gpu-allocator's 256 MiB blocks (an allocation below block
     size that opens a new block takes up to 256 MiB more than the probe saw) + the room up to the 99 %
     line (2 % of the budget, about 178 MiB on the 9.93 GiB B570) + the estimator's `unaccounted`
     (params, quantization scratch); small against the resident share (+2,646 MiB). (Re-graded
     2026-10-18 to SHOULD: derived from wgpu's allocator and budget lines and the B570; not re-derived
     on Dawn.)
   - **If it does not fit**: downgrade (decision 4 — permanent, like reactive eviction) → dispose the
     resident DiT → wait for the release → report `evicted` / `headroom`. No stage has been opened,
     so there is no re-run. A clean-up failure is reported, then thrown as an `AggregateError`.
   - **If the probe throws** (validation / device loss): no downgrade or eviction; it is thrown as is
     ("does not fit" is not the cause, and eviction would bury it).
3. **The release wait (`settleReleasedMemory`, decision 3 ②) issues one `queue.submit([])` before
   waiting** (`packages/models/src/session/settle-released-memory.ts`). The validated call surfaces a
   device already invalidated by a fatal OOM (the trigger for Deno to resolve loss) and flushes
   pending destroys. Crossing the 99 % line is safe here: the wait is reached at ≤ 97 % (right after a
   probe or a real allocation was refused) or with a dead device.
4. The public `AnimaResidencyReason` gains `"headroom"`; `ModelComponent` (internal to `models`) gains
   `estimate` and `maxPartBytes`.

### Kept

- **Decision 3's reactive eviction stays as the second line** (an estimate off, or another process
  allocating after the probe). A `createBuffer` OOM is evicted and re-run as before; a staging OOM has
  already invalidated the device, the wait's `submit([])` surfaces it, and the re-run fails with
  `GpuDeviceLostError` (not silently).

### Not adopted

- **Explicit staging** (`createBuffer(MAP_WRITE | COPY_SRC, mappedAtCreation)` → `copyBufferToBuffer`;
  every allocation becomes `createBuffer`, so OOM becomes non-fatal) MAY be reconsidered. It was
  rejected because Deno's `getMappedRange` copies the mapped VRAM range on the CPU: 82.6 s per GiB
  against 0.28 s for `queue.writeBuffer` (research 2026-09-27 §3). (Re-graded 2026-10-18 to MAY: the
  only ground is a Deno measurement; the cost on Dawn, the primary target since ADR 0127, is not
  measured.)

### Verification on the B570 (2026-09-27)

The reproduction (`evict-probe.ts --dummy-gib 6`, untracked, Deno) now succeeds: the second generate
reports `evicted` / `headroom` before `text_encoder` starts and the PNG sha matches the first; no OOM
or device loss. e2e `residency` cases: 4 green. Cost: two probes per generate with a resident DiT
(about 0.1 s); per-stage operation does not probe. The Chrome observation page named here
(`tools/gpu-lab`) is not part of `wip/0.15` (ADR 0127 decision 5: Deno-era tools are not ported), so
Chrome behaviour has no observation tool today.
