# 0132: Online int8 attention — `attentionScoreStorage: "none"` (S never materialised)

- Status: proposed (2026-10-23; to be accepted by the maintainer after the Wan2.2 A/B check)
- Date: 2026-10-23
- Related: ADR [0023](0023-fused-attention.md) (fused attention: the three dispatches ①QK / ②
  stats / ③PV and their bit identity with the decomposed path — unchanged, the default path) /
  [0030](0030-attention-i8a8.md) (the a8 variant whose quantisation this kernel keeps) /
  [0058](0058-numerics-opt-in-contract.md) (one named option per numerics change; shape and limit
  fallback with the key showing it) / [0060](0060-row-block-attention.md) and
  [0124](0124-attention-row-block-budget.md) (row blocks: capacity of the S temporaries — this
  kernel has none) / [0100](0100-rms-subgroup-reduction.md) (the `subgroups` acquisition — amended
  below) / [0110](0110-practical-tier-numerics-contract.md) (contract classes; this variant is Q) /
  [0074](0074-quant-seat-naming.md) (preset naming — token `online` added) /
  [0128](0128-default-preset-original-weights.md) (C1: the default preset needs no feature) /
  [0121](0121-wan22-ti2v-5b.md) (Wan2.2 — the practical preset this replaces as default) /
  perf-ledger K-82 / research
  [2026-10-23-flash-attention-webgpu-recon](../research/2026-10-23-flash-attention-webgpu-recon.md)
  (the measurements behind every number here).

## Context

Wan2.2 at 1280×704×121 spends 64% of one DiT forward (24.2 s of GPU time on the RTX 3080 Ti) in
the three attention dispatches, and 16.4% in ② row statistics alone, which is pure traffic over
the f16 score matrix S (35.7 GB per layer at S = 27,280). ①QK and ③PV are compute-bound on dp4a
(32 and 19 TOPS against a measured ceiling of 65), so the arithmetic cannot be cut without matrix
units, which stable WebGPU does not expose; what can be cut is ② and the S round trip. A
FlashAttention-2-style kernel — one dispatch per block of query rows that streams int8 key / value
tiles, keeps the running row max and sum, and never writes S — was measured as a standalone spike
(recon §10.2): ×1.46 against the current three dispatches at S = 27,280 (×1.61 with the key order
of Vᵀ prepared on the host), ×1.04–1.15 at S = 7,920. The K-5b kill line (×1.3 on the attention
kernels) is passed at the long shape, which is the target; the maintainer ruled on 2026-10-23 to
productise it and to make it Wan2.2's default once the A/B check confirms the expected speed.

Two things made the spike win and shape this design: the shared-memory tile layouts must be free
of bank conflicts (the first version, with the natural layouts, was 2× slower than the three
dispatches), and the row reductions are 1.23× faster through subgroup shuffles than through
workgroup memory. The second collides with ADR 0128 C1 (a default preset declares no GPU feature)
and ADR 0100 decision 1 (`subgroups: true` rejects a device without them).

## Decision

1. **The option is a value of the existing score-storage axis: `attentionScoreStorage: "none"`.**
   "S is not stored" is the definition of the online kernel, so no new key enters the manifest
   vocabulary (ADR 0058 decision 1: one axis per knob). `"none"` is implemented only with
   `attentionCompute: "a8"`; any other combination fails loudly at Session construction (an
   unimplemented combination is not a fallback — ADR 0058 decision 3). Preset names spell it with
   the token `online` in the slot of `s16` (`f16+dit8-a8-attn8-online`; ADR 0074 §3 addendum).
2. **One kernel per node, dispatched over windows of query rows.** For every node the recipe runs
   the existing activation quantisation (`quantize_rows` for q and k; `strided` permute +
   `quantize_rows` for Vᵀ — unchanged, the same temporaries as a8) and then the online kernel
   `attention_online:v1:i8a8:…` once per window of query rows. The window is not a capacity
   device (there is no S) but a dispatch-size device: the planner (a pure function next to
   `planAttentionBlocks`) sizes the window from a work budget so that one dispatch stays well under
   the 1 s submit gate (at the 121-frame shape about 1,024 rows ≈ 13 ms); the estimate (ADR 0124
   decision 7) counts no S temporaries for this route and takes the window count from the same
   function.
3. **Geometry**: 256 threads, 64 query rows per workgroup, 64 keys per tile, D ∈ {64, 128, 256}
   as a generator parameter (`D % 16 == 0`; other depths and `N % 4 != 0` fall back to the three
   dispatches of a8 / f32 with the key showing it — shape eligibility, ADR 0058 decision 3). The
   `q` tile is loaded once; `k` rows are padded to 33 words and each lane owns keys 16 apart; the
   Vᵀ tile is stored `[pack][d]` (word `w` packs keys `{w, w+16, w+32, w+48}`, gathered byte-wise
   from the quantised Vᵀ); P̃ rows are padded to 17 words. These layouts are part of the
   definition (they are what makes the kernel faster than the three passes).
4. **Numerics**: per key tile, S by `dot4I8Packed` and the a8 dequant (`f32(acc) · (qs'·ks')`,
   ADR 0030), the tile's row max `tm`, `m' = max(m, tm)`, `α = exp(m − m')`, `l = l·α + Σ exp(S −
   m')`, P̃ = `round(127·exp(S − m'))` packed to int8, `O = O·α + f32(Σ P̃·Vᵀ)` with `O` in f32;
   at the end `O · (vs[d] · (1/127) / l)`. The row sum `l` is kept as a per-lane partial and
   reduced once at the end. Keys beyond N take the identity −3.0e38 (not −inf: `α` must not
   become `exp(−inf − (−inf))`). The mask and GQA are rejected exactly as for a8 (`fusedAttentionViolation`).
   Contract class **Q** (ADR 0110): the error model is the int8 lattice of P̃ against a running
   max (each weight carries at most `0.5/127` of the final `exp(S − m)`, since the later rescale
   by `α ≤ 1` only shrinks the step) plus f32 accumulation; the four-point check of ADR 0110
   decision 3 is the kernel test, with a TS mirror that walks the same 64-key tiles in the same
   order and treats `exp` statistically as ③PV does (a mismatch is ±1 step of P̃, never ±2).
5. **Subgroups are a speed-only variant, bit-identical to the portable one.** The kernel has two
   reduction forms with different keys: `:sg32` (`subgroupShuffleXor` across the 16 lanes of a row,
   `@subgroup_size(32)`, lanes identified by `subgroup_invocation_id`) and `:wg` (workgroup memory).
   The row max is exact under any order; the final sum of `l` uses the same butterfly pairing
   (`((l0+l8)+(l4+l12)) + …`) in both forms, so the bits are the same — a test runs both on one
   device and requires Uint32 equality (the same discipline as `dot4I8Packed` against its
   emulation). The build selects `:sg32` when the device has `subgroups`, `subgroup-size-control`
   and the WGSL feature `subgroup_id`, otherwise `:wg` when the workgroup-storage limit allows it
   (`:wg` needs 33,280 B — 32 KiB devices without subgroups fall back to the three dispatches,
   also keyed). Because the choice cannot change a bit, it is capability-driven without breaking
   ADR 0058 decision 2.
   **ADR 0100 decision 1 is amended**: `AcquireGpuOptions.subgroups` accepts `"available"` —
   request the features when the adapter advertises them, silently run without otherwise; `true`
   keeps its rejecting meaning. The Wan family acquires with `"available"` when the preset's
   `session` declares `"none"`; a default preset still declares no `gpuFeatures` (ADR 0128 C1
   holds — the feature is never required).
6. **Wan2.2 gets a new practical preset `f16+dit8-a8-attn8-online`** with the same `weights` map as
   `f16+dit8-a8-attn8-s16` (no new download) and `session` `{ linearCompute: "a8",
   attentionCompute: "a8", attentionScoreStorage: "none" }`. It becomes `defaultQuant` when the
   e2e A/B check against the reference preset stays inside the band derived for the a8 preset and
   the per-op timing shows the expected gain (the maintainer's condition of 2026-10-23); the old
   practical preset stays as a sibling (users who pass its name keep working). Wan2.1 is not
   changed (the maintainer uses Wan2.2; the kernel applies to it unchanged if wanted later). The
   practical sha256 rows of the new preset are written fresh; the reference rows do not move.
7. **Scope**: the fused attention op (`attention` — the DiT models). The states-form attention of
   the language models keeps its kernels; porting the online form there is a later candidate.

## Consequences

- Wan2.2's 121-frame forward loses ②'s 4.0 s and the S round trip; expected about −20% of the
  forward on the RTX 3080 Ti (more with the host-side Vᵀ key order, which is a separate, measured
  follow-up with its own kill line ×1.05).
- Four new surfaces: the kernel generator (`src/kernels/attention-online-i8a8.ts`), the route
  value in `attentionRoute`, the window planner, and `subgroups: "available"` in `acquireGpu`.
  The hub's `SCORE_STORAGE` allowlist grows by one value; an older runtime reading a manifest with
  `"none"` fails loudly, which only affects a revision it was never pinned to.
- The Wan2.2 manifest changes (a preset and, on acceptance, `defaultQuant`), so the 0.15 release
  re-uploads it (release runbook).
- Devices without subgroups and with 32 KiB of workgroup storage (today: Firefox and Safari on
  D3D12 / Metal) run the three dispatches for this preset — slower, same values as the a8 preset
  they already get.

## Alternatives considered

- A new key `attentionKernel: "online"`: adds an axis and leaves `attentionScoreStorage` meaningless
  under it ("silently ignored" — the thing ADR 0058 decision 3 distinguishes from eligibility).
- Requiring `subgroups: true` and declaring it in `gpuFeatures`: the preset could not be the default
  (C1), and a bit-identical portable form costs one summation-order discipline.
- A subgroup-matrix (tensor-core) kernel: 91 TOPS measured on a naive i8 GEMM, but the feature is
  experimental behind `allow_unsafe_apis` and its WGSL surface is changing — a Node-only opt-in,
  ruled separately (not in this ADR).
- Keeping the three dispatches and optimising ③PV's tile fill and ②'s reads (K-69): both are made
  redundant by the online kernel for this preset.

## Open points

- The error band of the four-point check and the Wan2.2 A/B result (the acceptance condition of
  decision 6) are filled in when measured.
- Bc = 128 on 48 KiB devices, double-buffered tiles and `exp2` are unmeasured headroom (the spike
  sits at 26–29 TOPS, under half the dp4a ceiling).
