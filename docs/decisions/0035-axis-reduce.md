# 0035: Reduction axis for the reduce family (required `dim` attribute and a coalesced non-last-axis variant)

- Status: accepted (2026-08-05; the main check — Uint32 identity with the permute + row-reduce path
  — held in 9 cases and the sha256 checks stayed bit-identical, so no visual check was needed)
- Date: 2026-08-05
- Related: recon [2026-08-04-vae-axis-reduce-recon.md](../research/2026-08-04-vae-axis-reduce-recon.md)
  (attribution, the A3 design, §5.2 symbolic verification; its risk R4 was refuted — detection
  limit 1) / [0011](0011-layout-strategy.md) (layout strategy — the largest case of "remove the
  permutes") / [0015](0015-conv-family-extension.md) (required attributes) /
  [0033](0033-vae-fixed-tile-decode.md) and [0034](0034-dit-dynamic-tokens.md) (variable resolution)

## Context

The VAE decoder's `QwenImageRMS_norm` (L2 over the channel axis) was exported with a permute that
moved the reduction axis to the last dimension and back. Those 60 permute dispatches made up 99% of
the VAE's uncoalesced traffic (181 ms), and with the barrier-bound row reduce (39 ms) **17% of VAE
GPU time** (recon §2–3). The root cause was that the IR's `sum` reduced only the last dimension.

## Decision

1. **The reduce family (sum / amax / amin) has an attribute `dim`, and it is required** (no
   default; allowing omission would let an IR meant for the channel axis silently reduce the last
   dimension — the approach of ADR 0015). Negative axes are normalised at the exporter boundary; an
   axis outside the rank fails loudly on both sides (`reduceDim`,
   `packages/runtime/src/ops/shapes.ts`).
2. **A non-last axis uses a dedicated variant `reduce:v2:<op>:<from>><to>:axis:wg256`**
   (`axisReduceKey`, `packages/runtime/src/codegen/reduce.ts`): one thread per output, walking the
   reduction axis with the `inner` stride, so adjacent threads read coalesced and no barrier is
   needed. **The reduction order reproduces the row reduce's "256 lanes + bit-reversed binary tree"
   exactly, with a bit-reversed carry stack in one thread** (the 21-case symbolic verification of
   recon §5.2 is the design basis; the identity +0.0 is not folded away). The last dimension keeps
   the existing row kernel unchanged. **The branch MUST depend on the axis only**: allowing a choice
   by speed would make bit-invariance of the default path depend on runtime conditions
   (`buildRowReduce`, `packages/runtime/src/runtime/recipe-builders/elementwise.ts`). *(Updated by
   the [2026-10-26 addendum](#addendum-2026-10-26--the-axis-variant-splits-each-output-over-16-threads-perf-ledger-v-7):
   the key gains the split, several threads share one output, and the kernel now has barriers; the
   reduction order and the MUST are unchanged.)*
3. **Permanent check: `packages/runtime/tests/gpu_reduce_axis_parity_test.ts`** — **Uint32 identity**
   with the permute + row-reduce path (real shapes with dim = 96 / 192 / 384 and rows on both sides
   of and at 65,535, a permute of cycle length 3, reduction length 300 > 256, amax / amin, bool sum;
   15 cases today, including NaN propagation and the ±Inf identity values), a check that the two
   kernels are routed by axis, and a guard against a vacuous pass. *(19 cases since the
   [2026-10-26 addendum](#addendum-2026-10-26--the-axis-variant-splits-each-output-over-16-threads-perf-ledger-v-7),
   which lists the new ones.)*
4. **Exporter:** `_h_sum` and `_h_row_reduce` (amax / amin, extended in the same change for
   symmetry; `tools/exporter/src/karume/aten_handlers.py`) carry the single axis into attrs. Anima's
   channel L2 now calls `l2_normalize(x, dim=1)` (`tools/export-recipes/_shared/vae_rank4.py`)
   instead of the permute round trip. On 2026-08-05 the VAE decoder series, the SBV2 front series
   (which hold 3 bool sums over the last dimension, numerically unchanged) and 5 tiny goldens were
   re-exported.

## Measurements (2026-08-04 to 05)

VAE decoder IR 438 → 378 nodes (−60, exactly the norm's permute pairs); the 9 parity cases
Uint32-identical; **no tolerance changed** (re-derived values matched the old records to three
digits); sha256 checks unchanged. VAE keys: strided 191.25 ms / 71 dispatches → 5.80 ms / 11; the
reduce 41.10 ms (row) → 22.25 ms (axis variant, −46%; above the recon's optimistic 6–10 ms because
of the fixed 256 iterations per thread, as risk R3 foresaw). VAE wall time 1.6–1.7 → 1.4–1.5 s;
1024 px generation 22.5 → 22.0 s. Full record: commit `99a3c649`, this file.

## Detection limits

1. **Moving an axis without a layout change does not move torch's reference values by one ulp**
   (refuting recon R4): torch's permute is a view, so `permute + sum(dim=-1)` and `sum(dim=1)` reach
   the same reduction (`torch.equal` on four real shapes; the 26 tiny-golden io files
   byte-identical). **Loosening a tolerance because "the reduction order must have changed" needs a
   measurement first.**
2. The recon's census claim ("only the VAE decoder has `sum`") was wrong — a check that walks every
   IR (57 files) found the 3 in the SBV2 front. **Trust a census only after it walks the IR
   mechanically, not after a grep.**
3. Making `dim` required turns silent execution of old IR into a loud failure (fixed by re-exporting
   5 series). The required declaration prevents by structure the accident of "an old IR silently
   computing something else".

## Consequences

- The unsaturated uncoalesced strided reads (126 GB/s) and row reduce (124 GB/s) of the VAE are gone
  by structure; −60 dispatches per run also helps the non-GPU side.
- Reduction over any axis is a general capability, ready for future NCHW work (group_norm, batch
  norm statistics).
- The 3 tolerances of the f32 Anima series (assets deleted) were kept: torch's values are
  bit-unchanged, so a regenerated series is not expected to need re-derivation (regeneration is
  parked in the [backlog](../backlog.md)).

## Addendum (2026-10-26) — the axis variant splits each output over 16 threads (perf-ledger V-7)

- **Why.** With one thread per output, the decoder's channel norms with few outputs ran on a
  handful of workgroups: at P = 256 outputs (Wan2.2 decoder, 11 nodes per chunk) the whole reduce
  was one workgroup whose 256 threads each ran the 256-leaf loop serially (195–230 µs at C = 1024).
- **Thread mapping.** `split` threads share one output. In a workgroup of 256 lanes, lane
  `s · (256 / split) + o` takes output `o` of the group's `256 / split` outputs and subtree `s`.
  Consecutive lanes take consecutive outputs, so the reads along `inner` stay coalesced. Each thread
  walks `L = 256 / split` leaves `m = s · L + j` (`j < L`) with the same bit-reversed slot
  (`reverseBits(m) >> 24`), the same identity-started serial leaf fold and the same carry stack
  (depth `log2(L) + 1`). The subtree roots go to workgroup memory, and `log2(split)` barrier-separated
  levels compute `root[s] = combine(root[s], root[s + w])` for `w = 1, 2, 4, …`. The grid-stride loop
  walks blocks of `256 / split` outputs from `workgroup_id` and the uniform params, so every barrier
  is in workgroup-uniform control flow, and the per-output guard never encloses a barrier. A barrier
  follows the roots store and each level, before the next read of another lane's slot. The block ends
  without one: after the last level only the lanes with `s = 0` read, each from its own slot, and
  the next block's first read of another lane's slot sits behind its own post-store barrier.
- **Why the bits do not change.** The summation tree over `m` is a perfect binary tree. Restricted
  to an aligned range `[s · L, (s + 1) · L)`, the carry stack ends holding exactly that subtree's
  root, built by the same operations with the earlier-pushed value as the left operand. The levels
  above it combine adjacent roots as (lower) + (upper), which is what the single-thread carry stack
  did. Every add has the same operands in the same order as before, and f32 values pass through
  workgroup memory unchanged. The assumption that the compiler does not reassociate f32 adds is the
  same one the row reduce and the single-thread form already relied on. `split = 1` is the old
  single-thread order.
- **Key.** The axis key gains the split: `reduce:v2:<op>:<from>><to>:axis:s<split>:wg256`
  (`axisReduceKey`). The row-reduce keys do not change. Dispatch is
  `gridStrideWorkgroups(outCount, axisReduceOutputsPerGroup(split), limit)` (`buildRowReduce`).
- **The split rule: the constant `AXIS_REDUCE_SPLIT = 16`.** It was chosen by an in-process kernel
  A/B on the RTX 3080 Ti under Node + Dawn (2026-10-26). One process ran the single-thread generator
  and the split generator at split 4, 8, 16 and 32 interleaved in rotating order, timed with GPU
  timestamps per pass, on every axis-reduce shape of the local containers. These were the Wan2.2
  decoder first / next at tile 16 and the tile-32 pair, the Wan2.1 decoder, anima's VAE decoder, and
  the Wan2.2 I2V encoder at 1280×704, including its C = 4 sums over 1.1–9.0 M outputs (33 shapes).
  Two runs were made, with 21 and 41 repetitions. The gate was that the chosen split must not be
  slower than the single-thread form by median on any shape. Split 16 and split 8 both pass it (their
  worst shapes are ×1.16 and ×1.23); split 4 passes as well, and split 32 does not.
  - Split 16 against the single-thread form ranged from ×1.16 (C 256, P 262,144) to ×14.0 (C 1024,
    P 1,024). It was ×1.17–1.18 on the encoder's C 4 sums and ×1.16–1.58 on the other shapes with
    P ≥ 65,536.
  - Every output of every split was bit-equal to the single-thread form in both runs.
  - Weighted by nodes per graph (run 2), the axis sums cost:

    | Graph | Single-thread | Split 8 | Split 16 | Split 16 vs split 8 |
    | --- | --- | --- | --- | --- |
    | Wan2.2 first chunk @16 | 5.91 ms | 0.95 ms | 0.62 ms | 35% faster |
    | Wan2.2 next chunk @16 | 7.10 ms | 1.81 ms | 1.50 ms | 17% faster |
    | Wan2.2 first chunk @32 | 6.80 ms | 1.78 ms | 1.49 ms | 16% faster |
    | Wan2.2 next chunk @32 | 9.60 ms | 4.72 ms | 4.88 ms | 3% slower |
    | Wan2.1 first chunk | 3.00 ms | 0.88 ms | 0.80 ms | 9% faster |
    | Wan2.1 next chunk | 5.21 ms | 2.64 ms | 2.79 ms | 6% slower |
    | anima VAE decoder | 5.28 ms | 2.80 ms | 2.90 ms | 4% slower |
    | Wan2.2 encoder_pre | 16.96 ms | 12.08 ms | 12.62 ms | 4% slower |
    | Wan2.2 encoder_post | 0.688 ms | 0.096 ms | 0.058 ms | 39% faster |

  - **Why 16 and not 8.** The gate does not separate them, so the choice was weighted by graph.
    Split 16 is ahead on the default Wan2.2 decode (tile 16: 35% on first, 17% on next) and on
    first @32, Wan2.1 first and encoder_post. Split 8 is ahead by 3–5% on next @32, Wan2.1 next,
    anima and encoder_pre. Per shape, split 8 is 3–15% faster than split 16 on 16 of the 33 shapes,
    all at P ≥ 8,192 (the largest gap is C 1024, P 8,192). That is recorded as a later candidate (a
    per-shape split), not taken now.
  - Split 32 is slower than the single-thread form on the largest shapes (×0.65–0.97: C ≥ 512 at
    P ≥ 65,536, C 256 at P 262,144, C 160 at P 225,280).
  - Data and scripts: `outputs/diag/wave-b-2026-10-26/v7/` (`axis-reduce-ab.ts`, `ab-1.json`,
    `ab-2.json`, `table.md`).
- **Decision 2's MUST is unchanged in substance.** The row-or-axis branch still depends on the axis
  only. The split does not change the bits, so that MUST does not bind it; it is a constant today
  because the only evidence is one GPU. A per-shape split (a pure function of the static shape at
  recipe-build time) or a per-GPU field of the adapter profile, fixed per Session like the conv2d
  implicit-GEMM geometry, is a possible later step and is not ruled out by principle. Either way the
  split MUST be carried in the key, and it MUST NOT be taken from timing (the key would not be
  reproducible).
- **Checks.**
  - `packages/runtime/tests/reduce_axis_tree_test.ts` (no GPU) reads the fold identities, leaf
    count, shift, stack depth, combine levels, operand orders and the output store out of the
    generated WGSL. It replays the kernel's loops as string trees for axis lengths 1–512 at every
    split 1, 2, …, 256 and asserts that each tree equals the row reduce's. Fault injections show that
    it catches a misaligned leaf range, a swapped operand order (commutative addition would hide it
    in a numeric check), another identity and a store of a partial root.
  - The parity test dispatches the axis kernel at `AXIS_REDUCE_SPLIT`, passed explicitly as
    `buildRowReduce` does. It adds P = 256 / C = 1024, P = 2,048 / C = 1024 and P = 1,023 / C = 300.
    A separate case, P = 1,115,560 / C = 4, needs the grid to clamp at the adapter's
    `maxComputeWorkgroupsPerDimension` (65,535 on common adapters) so that the barriered
    grid-stride loop runs more than once; on an adapter that admits the natural grid it is skipped
    with that reason.
  - `packages/runtime/tests/gpu_gridstride_test.ts` runs the axis kernel on a degenerate grid of
    2 workgroups for 20,000 outputs (`axisReduceCase` in `tests/helpers/gridstride_cases.ts`),
    so each workgroup walks hundreds of blocks of the barriered loop.
  - `codegen_wgsl_test.ts` checks the size-derived invariants at every split, the barrier order
    (a barrier after the roots store and after each level, for split 2, 16 and 256, with fault
    injections that drop one), and rejects a split that is not a power of 2 up to 256.
- **Not measured.** D3D12, Metal and other vendors were not measured. The bit identity was observed
  on Vulkan / NVIDIA only (in the earlier spike, also on D3D12 / NVIDIA).
