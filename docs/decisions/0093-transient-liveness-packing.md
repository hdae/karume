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
  in decreasing size** at offsets of region buffers (revised to least region growth, best of two
  orders — see the addendum of 2026-10-26). Event order (allocate → retain → release of what
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

## Addendum (2026-10-26) — decision 1 revised: least region growth, best of two orders, instead of first-fit

Decision 1's placement rule changes; the rest of decision 1 (the replay, the usage-scope rule, the
limits) and decisions 2–8 are unchanged (perf-ledger H-42).

- **Least growth.** Each slot goes to **the region it grows least** among the regions that hold no read
  / write conflict with it, at the lowest offset that fits there (as before). Equal growth keeps the
  lower region index. **A new region opens when every candidate would grow by at least the slot's
  size** (its full size, or more where the offset alignment pads; and when there is no candidate): the
  new region costs the same bytes or fewer, and the new region carries no
  conflicts, so the slots placed after it keep more places to go.
- **Best of two orders.** The placement runs twice — slots in decreasing real size, equal sizes in
  allocation order (as before) and equal sizes longer-lived first — and keeps the smaller region total
  (the first order on a tie). Both orders are total, so the plan stays deterministic.

Why least growth: first-fit took the first region that could hold the slot even when that region had
to grow and a later region had room. Removing one node then changed which region the following
tensors landed in, and a long-lived tensor could stack on a large one — the region total grew while the
live peak did not move. Wave a hit this when the attention absorbed its output permute (Wan2.2 online
route, 33 frames: 888.0 → 1,042.7 MiB with the output permute alone absorbed, the live peak 634.2
unchanged — [research](../research/2026-10-25-wave-a-dit-fusions.md) §5).

Why two orders: least growth in one order still depends on how equal sizes are ordered, and a DiT
repeats the same sizes in every block. The Wan2.2 DiT with the q / k / v permutes absorbed and the same
graph with them as nodes (the check of `e2e_wan_ti2v_attention_permute_estimate_test`) swap places
between the orders — 717.8 vs 788.9 MiB at 33 frames in allocation order, 810.6 vs 717.8 MiB with the
longer-lived first — with the same live peak (634.2 MiB). Taking the better of the two gives both 717.8
MiB (2,457.2 MiB at 121 frames). Running several fast greedy placements and keeping the best is also what
TensorFlow Lite's GPU delegate does (`MemoryStrategy::GREEDY_BEST`).

Measured with the estimate (the execution's own `planTransients`), 2026-10-26, on every family's
default preset with a local container (46 graphs, 49 scenarios; raw data and bindings in
`outputs/diag/h42-2026-10-26/sweep/`, not tracked):

| Placement | Sum of region totals | Against first-fit | Regions per plan |
| --- | ---: | --- | --- |
| first-fit (decision 1 as written) | 30.26 GiB | — | 2–4 |
| least growth, one order | 24.48 GiB | 37 smaller, 12 equal, 0 larger | up to 33 |
| **least growth, best of two orders (adopted)** | **24.18 GiB** | 39 smaller, 10 equal, 0 larger | up to 33 |
| least growth, new region only when no region is a candidate | 29.29 GiB | 23 smaller, 24 equal, 2 larger | 2–4 |
| least growth, best of five orders (adds area, lifetime and start orders) | 23.86 GiB | 40 smaller, 9 equal, 0 larger | up to 33 |

The live peaks sum to 22.81 GiB (the lower bound); the adopted rule reaches it exactly in 25 of the 49
scenarios, for example the anima DiT (896.6 → 729.0 MiB), the Wan2.1 DiT at S 14,040 (2,247.8 →
2,064.0 MiB), the UMT5-XXL encoder (272.0 → 208.0 MiB) and the Irodori codec decoder (4,500.0 → 2,880.0
MiB). The Wan2.2 DiT goes 880.3 → 717.8 MiB at 33 frames (live peak 634.2) and 3,016.7 → 2,457.2 MiB at
121 frames (live peak 2,169.2). Taking the smallest fitting gap inside a region instead of the lowest
offset changed no scenario, so the offset rule stays. Five orders save 0.32 GiB more for five placements
instead of two; three of the added orders are worse than first-fit on their own (8–22 scenarios).

Consequences:

- More regions per plan (many under 1 MiB). Each region is one `createBuffer` (`executor.ts`, the miss
  run and `#activateBacking`); nothing else in the runtime is per region. That small buffers cost no
  extra device memory is the drivers' sub-allocation behaviour (observed on Dawn), not a WebGPU guarantee.
- Placement costs O(n²) at worst for n slots and runs in up to two orders; each slot excludes the
  regions of its read / write partners, and the second order is skipped when the first meets the live
  peak. The whole estimate of the Wan2.2 DiT's dense route at 121 frames (4,321 slots) takes 89–137 ms
  against 271–296 ms under first-fit (placement 68–106 ms), and about 60 ms for the anima and Wan2.2
  DiTs on their default presets. It runs on every planning call: the first (miss) run and the backing
  build of a signature, each rebuild after a backing is retired, and each scenario of an estimate.
- The results do not change: placement only decides which bytes are reused, and every kernel writes its
  whole output before reading it (ADR [0014](0014-layout-ops-full-write.md)). The sha256 rows are the check.
- Tests that fixed first-fit's layout by hand were re-derived for the new rule; the invariants checks
  (no overlap in time and space, the usage-scope split, alignment, total ≥ live peak) are unchanged.

## Addendum (2026-11-11) — decision 4 revised: the estimate walks the step list of the fusion plan

The addendum of 2026-10-10 counted only row-block attention windows in fused form ("other fusion rules are still
counted in pre-fusion form"). That line is replaced: the estimate now counts **every step of the fusion plan as
execution allocates it**.

- `transientSlotBytes` plans fusion under the same context as Session construction (`sessionFusionContext` on the
  facts `estimateFusionFacts` derives: the rule list with the rules of `fusionRules`, the fusion knobs, the weights'
  storage from the residency plan) and maps the returned `steps` in their order — deferred nodes and in-window
  passthroughs where the list puts them.
- A fused step of any rule, built-in or external, takes its bookkeeping from the device-free module execution replays
  (`fused-step-layout.ts`): the output spec, the temporaries in allocation order, what each binding points at and the
  release list. What a binding reads or writes comes from the dispatch's WGSL storage declarations, as in execution.
  MUST: one implementation for both sides; a hand copy in the estimate drifts the first time a rule uses a form the
  copy lacks.
- A node step keeps the estimate's own program where the node runs several dispatches (states-form attention, the
  fused attention op, `group_norm`, the two-phase argmax / topk) and gains the i8a8 linear: its two temporaries (`xq`,
  `xs`) come from the bytes function `buildLinearI8a8` calls (`linearI8a8TempBytes`), under the route predicate
  execution uses (`linearTakesI8a8`).
- An external rule's steps are checked as execution checks them (what a dispatch writes, that the output is written,
  a temporary bound inside its lifetime), so the estimate returns no number for a step those checks refuse. The checks
  recipe derivation makes on a dispatch itself (workgroup counts, params) are not run. With `fusionRules` the estimate requires
  `maxStorageBufferBindingSize` and `maxComputeWorkgroupsPerDimension` (a rule's matcher may read them); the
  2026-10-10 note "`maxComputeWorkgroupsPerDimension` is not checked" holds only without rules.
- Still not counted, and listed in `unaccounted`: fusion under a knob the caller did not pass, the i8a8 linear of a
  borrowed weight (its storage is the lender's), the conv3d i8a8 temporaries, params buffers, `writeBuffer` staging and
  the retirement window.

Why: the estimate and execution each kept a table of "what a fused window allocates". The estimate's table covered
three forms, so every other window was counted with intermediates no Session allocates, and the i8a8 linear's
temporaries were missing. Sharing the bookkeeping removes the second table (ADR
[0070](0070-shard-loading-admission.md) decision 5: the same rule as execution).

Because shared bookkeeping makes "estimate equals execution" partly true by construction, the checks are independent
of it: hand-derived byte literals per rule on the CPU (`estimate_fused_steps_test.ts`), and on the GPU
`planBacking.residentBytes === workspaceBytes` for every rule's window, two external rules and the i8a8 linear
(`gpu_estimate_test.ts`). Fault injection confirmed four mutations turn them red: a dropped fused temporary, a dropped
i8a8 temporary, and — only on graphs with a released region ahead of the step, added for it — a dropped read of a
fused dispatch and of the i8a8 GEMM.

Evidence (2026-11-11, RTX 3080 Ti, Node + Dawn): on the Wan2.2 TI2V-5B DiT at `f16+dit8-a8-attn8-online` the estimate
equals the run's backing — 768,867,264 B at S = 7,920 and 2,632,460,864 B at S = 27,280 (it was 16,178,112 B and
55,904,832 B under, the i8a8 linear temporaries); `e2e_wan_ti2v_dit_estimate_test.ts` pins both.

Effect on `workspaceBytes`: graphs whose windows were counted unfused go down (the MOSS-Image VAE stage's need by
2,192,900,480 B at 1664², Anima's VAE stage by 100,663,296 B, the Wan2.2 DiT's dense f32 route by 317,395,968 B at
S = 27,280); graphs with i8a8 linears go up by their temporaries where no free region holds them (the Wan2.2 online
route above; the SBV2 text encoder at `i8-a8` by 1,572,864 B). Anima's text encoder workspace rises by up to
4,194,304 B at 512 tokens; its stage need does not move, because the workspace stays under the keep budget. No
MOSS-Image stage need rises at any size.

The headroom margins of the family pipelines are unchanged (maintainer's ruling 2026-11-11). The estimate is the
theoretical demand and a guard for a caller that did not check the model's VRAM requirement; it is not compared with
free VRAM (ADR 0070 decision 5) and what `unaccounted` lists stays outside it.
