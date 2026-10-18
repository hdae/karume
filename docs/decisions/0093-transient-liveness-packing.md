# 0093: Static liveness packing of transient buffers — from size-class pools to intervals and offsets

- Status: implemented (2026-09-05 — decisions 1–8 wired into the runtime; the maintainer chose the
  recommended order of the exhaustive review of 2026-09-04: algebraic rewrite (A) → this ADR (B) →
  limit preflight (C); measurement notes in `.claude/reviews/2026-09-04_62bdbeb/`)
- Date: 2026-09-05
- Related: [0004](0004-execution-model.md) (buffer management with size-class pools — **replaced by
  this ADR**; invariants ①②③④⑥ kept) / [0014](0014-layout-ops-full-write.md) (full-write — the basis
  for safe reuse) / [0054](0054-resident-loop-and-fence.md) (fences, unchanged) /
  [0066](0066-generation-context-state-slots.md) (state slots, out of scope) /
  [0070](0070-shard-loading-admission.md) decision 5 and [0089](0089-memory-limits-preflight.md)
  (extended to intermediates by decisions 4 / 5) / [0022](0022-gemm-register-blocking.md) (no run-time
  autotuning)

## Context

Intermediate buffers (node outputs and per-node temporaries) were allocated from a **LIFO pool with
exact size-class matching** (`RunArena.allocStorage`, mirrored by `derivePlanSlots` /
`estimate.transientSlotBytes`). When sizes do not line up, released buffers are not reused and slots
pile up, so **the GPU memory needed grows to about 3.2× the maximum of simultaneously live bytes (the
liveness peak)**. Computed on the real BiRefNet HR IR (2026-09-04, `estimateGraphMemory` and a replay
of the pure function):

| Resolution | workspace (pool) | liveness peak | inflation |
| ---------- | ---------------: | ------------: | --------: |
| 1024²      |         6.14 GiB |      1.88 GiB |     3.27× |
| 2048²      |        23.76 GiB |      7.50 GiB |     3.17× |

At 2048² a 12 GiB GPU hit the allocation ceiling 5.4% into the graph (the limitations entry on Deno
capping total GPUBuffer allocation at 97% of the driver-reported budget). The same inflation applied
to other families; it is why BiRefNet 1024² was "desktop-class GPUs only".

Exact matching was justified by "a larger buffer silently changes `arrayLength()` of a runtime-sized
array" (`arena.ts`). But ① generated WGSL never calls `arrayLength` (element counts come in the `n`
param), and ② a bind group entry is `{ buffer, offset, size }`, so the **bound range can be cut to
the exact size**. The exact-size guarantee can be a property of the bound range, not of the buffer.

## Decision

### 1. The plan = intervals + offsets (a pure function; no run-time autotuning)

- Replay the allocation and release events of the recipe list (`StepRecipe[]`) in order to obtain each
  logical tensor's (node output, temporary) **live interval** `[alloc, release)` and exact size
  (`toSizeClass`), then place tensors whose intervals do not overlap into shared bytes by **first-fit
  in decreasing size** at offsets of region buffers. Event order (allocate → retain → release of what
  the step defines) is the same as before (the MUST order of `executeStepRecipe`). Aliases (reshape,
  identity expand) merge into the interval of their root; pinned tensors (graph outputs) live to the
  end.
- Regions are binned by `maxBufferSize` and may be several buffers. Offsets are aligned to
  `minStorageBufferOffsetAlignment` (the granted value; spec default 256 B).
- **The usage-scope rule** (WebGPU: one dispatch cannot bind the same GPUBuffer as both `read` and
  `read_write`; the check is per buffer, not per range) is part of the plan: two tensors of which one
  is read and the other written by the same dispatch go into **different regions**. The roles are
  taken mechanically from **the WGSL declarations of the kernel** (`var<storage, read>` /
  `read_write`), with no hand-written table.
- The plan is a pure function of the bindings with the lifetime of `PreparedPlan`. **The GPU's granted
  limits (`maxBufferSize` / `maxStorageBufferBindingSize` / offset alignment) are passed as values**
  (ADR 0022 — like the number of row blocks in fusion).

### 2. Bindings are cut to the exact size

For slots handed out by the plan, `createBindGroup` entries pass `{ buffer: region, offset, size }`
(Session-resident weights / scales / params, inputs and state slots still bind the whole buffer).
"Never hand out a buffer larger than requested" (ADR 0004 / `arena.ts` MUST) is restated as "cut the
bound range to the exact size".

### 3. Miss runs and hit runs use the same plan (the arena's output pool is retired)

- A miss run (the first) allocates the plan's regions for the run's lifetime and returns them at the
  end. A hit run keeps the same plan's regions as Session-resident backing and bakes the bind groups.
  The two paths (pool allocation vs resident slots) stay; only **what is allocated** changes from one
  buffer per slot to one or a few regions.
- `RunArena` shrinks to host-written buffers, readback staging and region ownership
  (flush-before-destroy); output reference counting and `assertDrained` move to the plan's replay.

### 4. The estimate shares the packer

`transientSlotBytes` of `estimateGraphMemory` builds the same kind of allocation program from the
pre-fusion node list and packs it with **the same function**, returning the sum of regions (ADR 0070
decision 5, "the same rule as execution", met by sharing the function instead of copying the rule to
three places). Intermediates folded by fusion and temporaries of fusion rules stay on the
`unaccounted` side (the pre-fusion approximation is unchanged; see the addendum of 2026-10-10). The
estimate uses WebGPU core defaults (`maxBufferSize` 256 MiB, alignment 256 B), or the device's granted
values when a device is given.

### 5. Limit preflight for intermediates (ADR 0089 decision 1 extended; option C of the review)

At plan time, "slot size > `maxStorageBufferBindingSize`" and "region > `maxBufferSize`" are **listed in
full and fail before any allocation** (same message form as `assertWeightsWithinLimits`). This closes
the item "run-time transients rely on errorScope" from ADR 0089's Consequences.

### 6. Meaning of the diagnostics fields

- `ArenaStats.peakTransientBytes` = **liveness peak** (the plan's maximum of simultaneously live
  bytes, `TransientPlan.peakLiveBytes`). `transientBytes` = intermediates live at the end of the run =
  total bytes of pinned graph outputs. `allocatedBytes` = sum of regions + host / staging.
  `reuseCount` = slots sharing a byte range with a slot placed earlier (`TransientPlan.sharedSlots`;
  the full-write fault-injection test uses it to confirm reuse happens — changed from "always 0" at
  implementation, 2026-09-05). `PlanBackingStats.residentBytes` = sum of regions.
- The footprint check "total slot bytes = pool allocation of a non-backed run" reads "sum of backed
  regions = sum of regions of a miss run" (an identity, since both use the same plan).

### 7. Unchanged

Full-write (ADR 0014 — every node writes every byte of its bound range and never reads leftovers of an
earlier run or another tensor), flush-before-destroy, `writeBuffer` targets outside the pool (inputs do
not go into regions), state slots owned by the context (ADR 0066), codegen determinism.

### 8. Retired MUSTs

- ADR 0004 buffer management, "per-run arena (**size-class pools** + release by the last consumer)" →
  interval + offset placement.
- `derivePlanSlots` in `recipe.ts`, "**no packing of its own** (derive by virtually replaying the
  RunArena size-class LIFO)" → one pure function (`transient-plan.ts`) shared by execution, baking and
  the estimate (the concern that "may another value grab the same memory" is judged by two
  implementations disappears because it is judged in one place).
- "Never hand out a buffer larger than requested" in `arena.ts` → restated as decision 2.

## Consequences

- Projection (simulated on the real IR, overlap-checked; first-fit reaches the liveness peak exactly):
  BiRefNet 2048² workspace 23.76 → 7.50 GiB (2.50 GiB together with the decoder-tail algebraic
  rewrite, option A), 1024² 6.14 → 1.88 GiB (0.63 GiB with A). Other families shrink by the same rule
  (a memory-management correction, not a performance change).
- Measured on 2026-09-05 (RTX 3080 Ti, Deno wgpu Vulkan): BiRefNet 1024² intermediate regions
  2,020 MiB (liveness peak 1,920 MiB; the gap is alignment and the separate-region rule), run
  1.5–1.7 s. At 2048² the preflight listed `'upsample_bilinear2d_20' 3221225472B / 'cat_212'
  4026531840B` and failed (the two tensors option A removes). With A (recipe patch ⑨): 1024²
  intermediates 749 MiB; 2048² intermediates 2,948 MiB (liveness peak 2,560 MiB), weights 1,116 MiB,
  total GPU allocation about 4.0 GiB (corrected 2026-10-11 from 4.1 GiB, which divided MiB by 1,000),
  run 7.5–8.6 s on the 12 GiB card; e2e (1024² golden) stayed green.
- Surfaces touched: `packages/runtime/src/runtime/transient-plan.ts` (new, the packer), `recipe.ts`,
  `executor.ts` (`#activateBacking`), `estimate.ts`, `arena.ts` (output pool retired), and dispatch
  roles from WGSL (`StorageRoles` in `packages/runtime/src/gpu/pipeline-cache.ts`, consumed by
  `recipe-builder.ts`). The public surface (`mod.ts`) is unchanged.
- Regression checks: the e2e goldens of every family (sha256), `gpu_plan_backing_test` (the
  restated footprint check), `gpu_estimate_test` (GPU comparison), `gpu_full_write_test` (fault
  injection), and the `peakTransientBytes` expectations in `gpu_runtime_executor_test` (updated to the
  liveness peak — the number changes because its meaning changes).
- Not verified at the time: symbolic-dimension families (gemma4 / sbv2); Metal / Dawn offset bindings.

## Addendum (2026-10-10) — decision 4 revised: row-block attention windows are estimated in fused form

Decision 4's "intermediates folded by fusion stay on the `unaccounted` side" is revised **only for
the windows of row-block attention** (ADR [0060](0060-row-block-attention.md)).

- The estimate (`transientSlotBytes` in `estimate.ts`) calls the execution's fusion plan
  (`planFusions` in `fusion.ts`, a pure function without GPU). Only windows claimed by row-block
  attention are replaced by their post-fusion allocation program (the rule's declared temporaries,
  single output, deferred release of external inputs) before packing; dispatch roles come from WGSL
  as in execution.
- The limit preflight of decision 5 applies to the buffers actually bound after fusion (one row
  block's temporaries and the output), not to the full-width scores, mask-added scores or softmax
  output that fusion never materialises. When fusion does not apply and an intermediate exceeds a
  limit, every overrun is listed and it fails, as before.
- Other fusion rules are still counted in pre-fusion form (`unaccounted`).
- The fusion plan uses the packer's binding limit (`EstimateOptions.maxStorageBufferBindingSize`, core
  default when absent). `maxComputeWorkgroupsPerDimension` is not checked (it does not affect bytes).

Why: in pre-fusion form exactly the shapes that run thanks to row-block splitting (pre-fusion scores
above the binding limit — ADR 0060's demand shape) threw in the estimate and returned neither
`peakAccountedBytes` nor `unaccounted`. A second copy of "which intermediates fold" would be double
bookkeeping, so the estimate shares the execution's function (ADR
[0070](0070-shard-loading-admission.md) decision 5).

Effect on `workspaceBytes`:

- For graphs with row-block attention windows, one row block's temporaries and output are counted
  instead of the pre-fusion scores, mask-added scores and softmax output (`H·M·N·4` bytes each). At
  most two temporaries are live at once, and reads and writes of one dispatch go to separate regions,
  so they occupy two regions. The block count depends only on the estimate's binding limit.
- Example: for `q[4,2048,8]`, 4,200 key columns and a 128 MiB binding limit, the three pre-fusion
  intermediates (137,625,600 B each) exceeded the limit and threw. After fusion they split into two
  blocks of 1,024 rows: temporaries 68,812,800 B × 2 regions + output 262,144 B = 137,887,744 B (the
  hand-computed constant in `packages/runtime/tests/estimate_test.ts`).
- Graphs without such windows do not move by a single byte.

The user's budget (`EstimateOptions.attentionRowBlockMaxBytes`) also bounds the block size (the smaller
of it and the binding limit, derived as in execution). Per-node temporaries of the fused attention op
are counted in the estimate too — ADR [0124](0124-attention-row-block-budget.md) decision 7.
