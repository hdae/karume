# 0004: Execution model (static shapes, split submits, resource management)

- Status: accepted (2026-08-01); partly superseded by [0014](0014-layout-ops-full-write.md)
  (full-write moves to the kernels, 2026-08-02), [0032](0032-deserialize-submit-and-optin-timing.md)
  (no per-submit `onSubmittedWorkDone`; the adaptive window is per flush, 2026-08-04),
  [0054](0054-resident-loop-and-fence.md) (`ResidentTensor` / `BatchScope` as a fourth lifetime
  class; one `mapAsync` fence per normal run, 2026-08-13) and [0093](0093-transient-liveness-packing.md)
  (size-class pools replaced, 2026-09-05); amended by [0123](0123-submit-inflight-bound.md) (bound
  on submits in flight, 2026-10-07). The skeleton below is in force.
- Date: 2026-08-01
- Related: recon §2 / §3 / §4 — all 14 invariants come from failures measured in the prototype.

## Decision

### Shapes

- **Every node's output shape is fixed before execution (static shapes).** Data-dependent output
  shapes (`nonzero`, …) are unsupported. Supporting them ("fixed upper bound + mask") needs its own
  ADR, because it reaches the whole execution model.

### GPU execution

- The dispatch sequence is **split by a time budget into several command buffers and submitted**
  (against the 2 s TDR and the Chromium watchdog). A chunk size of 0 is rejected at construction.
  *(Revised 2026-08-04, ADR 0032: chunk time is no longer measured from the previous chunk's
  completion — a per-submit `onSubmittedWorkDone` serialises CPU and GPU on Deno — but from the
  single wait in `flush`; window definition in ④ below. Amended 2026-10-07, ADR 0123: so that the
  host does not run unboundedly ahead of the GPU, a run places a completion marker every 64 submits
  of the window, between steps, and waits for the oldest marker when more than 2 are pending.)*
- Invariants of the adaptive control (revised at M1-P2; details in the doc comment of
  `packages/runtime/src/gpu/submit.ts`): ① **growth only on measured evidence** (a measurement of 0,
  below timer resolution, means "no information", not "fast"; without evidence the initial chunk
  size stays); ② decide before adding, so that **one chunk's estimated time never exceeds the
  budget** (no shrinking after an overrun); ③ the estimate is per **workgroup**, not per dispatch
  (with symbolic dimensions the weight of a dispatch changes with the input length); ④ no time is
  attributed to individual chunks (completions of overlapping submits arrive together and the first
  one takes all the time) — the estimate is the window's measured time ÷ its total workgroups.
  *(Revised 2026-08-04, ADR 0032: window = first submit of the window → resolution of the flush's
  `onSubmittedWorkDone`. It includes host encoding time, so the estimate errs large, towards smaller
  chunks, which is the TDR-safe side. A window with no work is "no information", which prevents
  0 ÷ 0 → Infinity pinning the size at `minChunkSize`.)*
- **`device.lost` is a first-class runtime event**; ignoring it leaves `mapAsync` hanging forever.
  The device, the `PipelineCache` and the scheduler can be rebuilt after `device.destroy()`. The
  prototype observed that only `device.destroy()` returned VRAM and `buffer.destroy()` returned none;
  this is implementation behaviour, not a guarantee (on Deno 2.9.6 a destroyed device kept its memory
  until garbage collection — [known issues](../known-issues.md)). *Recorded 2026-09-25: Deno 2.9.6
  panicked on device loss instead of reporting it, and a single dispatch over the budget cannot be
  split (BiRefNet 2048² on the Intel Arc B570 — [limitations](../limitations.md)).*
- `requiredLimits` requests the compute limits explicitly (workgroup storage, invocations,
  workgroup size X / Y / Z) and checks them after acquisition. `pushErrorScope('validation')` is
  always on (an invalid pipeline does not throw; its dispatch is a no-op and the output is silently
  all zero).
- Codegen is **deterministic (same key → byte-identical WGSL)**, pinned by snapshots. Elementwise
  ops and the row-reduce family assume grid-stride loops.

### Buffer management

- A per-run arena (size-class pools + release at the last consumer) *(pools superseded by ADR 0093
  on 2026-09-05: interval + offset placement, first-fit packing of live ranges; `RunArena` keeps
  only region ownership and flush-before-destroy)*. Invariants: ① release only at node boundaries;
  ② output allocation before dispatch encoding; ③ `useCounts` count `node.ins` exactly, with
  repeats; ④ targets of `queue.writeBuffer` are outside the pool; ⑤ **full-write: every node writes
  every byte of its output buffer**; ⑥ on disposal and failure paths, always flush, then destroy.
  Reading back an intermediate value is rejected.
- ⑤ revised 2026-08-02 by ADR 0014: originally "an op that does not write everything gets a fresh
  buffer" (bypassing the pool); rejected because the arena would need per-op knowledge. **The kernels
  write the whole range instead.** Reused buffers are not zeroed, so an op that needs zeros (`pad`)
  writes 0 outside its data. A node with several dispatches (`cat`) covers the range per node; the
  caller checks the sum of offsets against the output axis length. Pinned by fault injection (poison
  a reused buffer, run, no poison remains — `packages/runtime/tests/gpu_full_write_test.ts`).

### Conventions

- **Fail loudly everywhere**: unsupported ops or attributes, a storage dtype leaking out, unbound
  symbols, impossible broadcasts and negative padding throw instead of approximating. Symbol
  bindings are looked up through `Object.hasOwn`.
- **Initialisation is an explicit async stage** (the prototype's synchronous constructor loop had a
  28.5 s floor), with an entry that pipelines weight fetch and GPU upload (chunks fed in sequence).
- Graph optimisation passes (aliasing identity `expand`, folding `permute` chains, fusion) are a
  first-class feature. Fusion is only an optimisation; the unfused path always defines correctness.

## Open points

- KV cache representation: resolved by ADR [0066](0066-generation-context-state-slots.md)
  (`GenerationContext` and named state slots).
- Grounds for the time-budget defaults: still the prototype's empirical values, 100 ms and safety
  factor 0.5 (`packages/runtime/src/gpu/submit.ts`: `timeBudgetMs`, `CHUNK_TIME_SAFETY`).
- Further memory planning: resolved by ADR [0093](0093-transient-liveness-packing.md).
