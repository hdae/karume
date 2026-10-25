> This document is a point-in-time snapshot (2026-10-25) of wave a of the post-recon speed work: three small-op fusions
> of the Wan DiT (perf-ledger F-4, F-8 and part of F-3), measured on the RTX 3080 Ti under Node + Dawn. It is not a design
> source of truth (the design is ADR 0040's addendum of 2026-10-25). Raw material is under `outputs/diag/wave-a-2026-10-25/`
> (not tracked).

# Wave a: the Wan DiT's small-op fusions (F-4, F-8, F-3 stages 2–3)

## 1. Summary

- Every fusion is bit-identical: every Wan2.2, Wan2.1, anima and Irodori sha256 row is unchanged on this machine
  (node-nvidia-geforce-rtx-3080-ti) after each stage.
- One 33-frame Wan2.2 DiT forward of the default preset (`i8-a8-attn8-online-conv8`): GPU time
  2,926.5 → 2,825.2 ms (−101.3 ms, −3.5%), dispatches 2,043 → 1,772; wall 2,988 → 2,898 ms (median of 14, −3.0%).
  One 121-frame forward: 16.06 / 16.09 → 15.71 / 15.85 s as the first forward of a run (≈ −0.3 s).
- The transient slot backing does not grow on the default route (it shrinks slightly); anima's grows by 1.7% (a placement
  artifact, §5).
- Deviations from the plan: the output permute (F-3 A4) is not absorbed yet (§5); stage 3 took the f32 form (b) instead
  of the one-kernel prologue-with-quantize (a) (§6).

## 2. Stage 1 — F-4: the elementwise gelu folded into the a8 linear's quantize

The linear's i8a8 step ① quantizes `gelu(x)` directly (kernel `quantize_rows:…:<op>:reg<n>`, the row held in registers,
K ≤ 16,384); the gelu node disappears and its f32 output is never written. Decided in `planFusions` with the linear's own
route predicate (`linearTakesI8a8`), executed by the linear recipe; counter `linearPrologue`.

| Measurement | Value | Source |
| --- | --- | --- |
| Kernel, tanh form, Wan2.2 down projection (K 14,336, M 7,920) | pair 2,303.7 → fused 683.7 µs (Node), 2,293.8 → 681.6 (Chromium); 0 differing words | `erf/NOTES.md` (control) |
| Kernel, erf form, anima (K 8,192, M 4,096 / 1,024) | 678.4 → 199.8 µs / 182.0 → 55.0 µs (×3.3–3.4); sbv2-shaped K 4,096: ×3.2 (M 256), ×2.2 (M 64); 0 differing words in xq, xs and the f32 y, Node and Chromium, special values incl. NaN / ±inf | `erf/NOTES.md` |
| Register cliff | none up to K 16,384 (erf +5%, tanh +11–14% at 16,384 vs 8,192); a step appears at 32,768 — above the bound | `erf/NOTES.md` §3 |
| 33f forward, per-kernel GPU (interleaved, 2 runs × 3 forwards each) | 2,929.9 → 2,877.3 ms (−52.5): gelu_tanh 34.3 → 0, plain quantize 84.1 → 48.5, prologue quantize +20.7 | `stage1/bd33-*.json` |
| 33f forward, wall (14 forwards each) | 2,982 → 2,952 ms median (noise ±30 ms) | `stage1/dit33-*.json` |
| 121f forward, wall (cool first forwards) | 16.14 / 16.09 → 15.91 / 15.94 s (≈ −0.19 s) | `stage1/dit121-*.json` |
| Hits | Wan2.2 31 (30 FFN + text embedder), Wan2.1 31, anima DiT 28 (a8 presets), DeBERTa w8a8 test graph 24; sbv2's distributed pipeline does not run its text encoder in a8 (no fold) | GPU-less count tests |

A caveat the erf measurement found: an xq / xs comparison barely sees a wrong gelu (a 1e-7 fault in an erf constant moved
24% of y's words but only 1–3 xq words of 2 M), so the strong check is end to end — the anima a8 sha rows (25 / 25 unchanged).

## 3. Stage 2 — the attention absorbs its input permutes (F-8 and the permute part of F-3)

Under an a8 attention seat, the fused attention reads q / k / v token-major: its quantize reads q / k rows in place
(`quantize_rows:…:bshd`) and its Vᵀ comes from one tiled transpose (`transpose_tiled`), so the three `[0,2,1,3]` permutes
and the transposing strided copy are gone (counter `attentionPermute`; the decision needs no route knowledge — a shape
that falls back to a dense stage copies the operand head-major inside the node).

| Measurement | Value | Source |
| --- | --- | --- |
| 33f forward, per-kernel GPU (stage 1 → stage 2) | 2,862.0 → 2,841.1 ms total; small ops ≈ −34 ms (strided 61.7 → 17.5, q / k quantize 19.0 → 21.4 in the bshd form, transpose +8.1; the linear drifted +10.0); dispatches 2,012 → 1,832 | `stage2/bd33-*.json` |
| Per attention node | online route 9 → 6 dispatches, three-dispatch s16 route 11 → 8 | `gpu_attention_permute_fusion_test.ts` |
| Hits | Wan2.2 / Wan2.1 180 (60 attentions × q / k / v), anima 56 (v only — its q / k reach attention through per-head rms_norm and RoPE), every dense-only family 0 | GPU-less count tests |

## 4. Stage 3 — rms_norm → pair RoPE in one kernel (F-3)

The self-attention q / k prologue (rms_norm over the token row → two reshape aliases → pair-form RoPE) runs as one f32
kernel (`rms_norm_rope:v1:pair:bshd:f32:wg256`) built from the plain kernels' text (the RoPE products keep their workgroup
staging); counter `rmsNormRope` (60 per Wan DiT, replacing the 60 `rope` fusions). The interleaved q / k chains are matched
with a new "deferred nodes" form of a rule (the other chain's nodes are scanned again after the fused step).

| Measurement | Value | Source |
| --- | --- | --- |
| Kernel (spike, production text) | 0.599–0.614 of the pair's time at Wan2.2 S 7,920 / 27,280, 0.585 at Wan2.1 S 14,040 (Node; Chromium within 1%); 0 differing words incl. NaN payloads | `stage3/NOTES.md` |
| Rounding barrier | required: without the staging 19–25% of the words move by one ulp (the driver contracts `direct + cross`) | `stage3/NOTES.md` |
| Subnormals | this backend flushes them in both kernels alike; a subnormal-keeping backend is unchecked | `stage3/NOTES.md` |
| Per forward (estimate from the kernel) | −14.1 ms (Wan2.2 33f), −46.4 ms (121f), −12.1 ms (Wan2.1 33f) | `stage3/NOTES.md` |
| 33f forward, per-kernel GPU (whole wave) | rms_norm 30.9 → 10.9 ms (the 60 q / k norms; the cross-attention norms stay), rope 16.0 → 0, rms_norm_rope +21.7: −14.3 ms | `final/bd33-*.json` |

## 5. Memory: why the output permute stays a node

The transient planner places slots first-fit by decreasing size (ADR 0093 decision 1). Removing the output permute changes
which region each tensor of the following chain lands in; a long-lived residual then stacks on the FFN's 433 MiB tensor. The
live peak does not move, the region total does (estimate = slot table, MiB, total / live):

| Case | Before stage 2 | All four permutes absorbed | Shipped (q / k / v, a8 seats only) |
| --- | --- | --- | --- |
| Wan2.2 online, S 7,920 | 888.0 / 634.2 | 1,065.9 / 634.2 | 880.3 / 634.2 |
| Wan2.2 online, S 27,280 | 3,043.3 / 2,169.2 | 3,656.0 / 2,169.2 | 3,016.7 / 2,169.2 |
| Wan2.2 s16, S 7,920 / 27,280 | 2,325.1 / 5,052.6 | 2,256.2 / 4,813.1 | 2,163.4 / 4,493.4 |
| Wan2.2 dense, S 7,920 / 27,280 | 2,570.7 / 4,249.8 | 2,849.1 / 5,208.8 | unchanged (not absorbed) |
| Wan2.1 s16, S 14,040 / 32,760 | 2,411.9 / 4,161.5 | 2,330.1 / 3,969.8 | 2,247.8 / 3,777.9 |
| anima s16, S 4,096 / 9,216 | 880.5 / 2,124.6 | 928.6 / 2,232.6 | 896.6 / 2,160.6 (+1.8% / +1.7%, live unchanged) |

Absorbing O alone gives 1,042.7 / 3,576.1 on the Wan2.2 online route; absorbing on dense seats adds real live bytes (the
head-major copies live with their inputs) and saves no dispatch. A best-growth placement (perf-ledger H-42) removes the
artifact for every graph measured (49 scenarios: 37 better, 12 the same, 0 worse; summed region totals 31.3 → 25.2 GB);
the output permute can be absorbed after it.

## 6. Plan deviations

- Stage 2's output permute (F-3 A4) — deferred to after H-42 (§5); the online kernel's token-major store was removed with it.
- Stage 3 took form (b): rms_norm → RoPE as an f32 fusion, with the quantize still the attention's own dispatch (the
  attention.ts MUST), instead of (a) — the spike's one kernel for rms_norm → RoPE → permute → quantize. (b) needs no
  quantize inside a producer and applies to every attention seat; it gives about 85% of (a)'s gain (estimate).
- F-4 also took the erf `gelu` (anima, DeBERTa) after its own measurement (§2).

## 7. The whole wave (33f, per-kernel GPU, interleaved, 2 runs × 3 forwards each — `final/bd33-*.json`)

| Kernel | Before | After |
| --- | --- | --- |
| gelu_tanh | 34.3 | 0 |
| quantize_rows (plain) | 84.1 | 48.5 |
| quantize_rows gelu_tanh prologue | — | 20.7 |
| strided copies | 62.0 | 17.5 |
| quantize_rows r8w32 (q / k) | 19.1 | 0 (21.5 in the bshd form) |
| transpose_tiled | — | 8.0 |
| rms_norm | 30.9 | 10.9 |
| rope (pair) | 16.0 | 0 |
| rms_norm_rope | — | 21.7 |
| **total GPU per forward** | **2,926.5** | **2,825.2** |

## 8. Not measured

- D3D12 / Metal (the RTX 5070 Ti) — the 17-frame quick-check hash of the playground is the end-to-end check there.
- Chromium timings of the production kernels (the spikes were timed on Chromium 156 Linux too).
- The estimate does not replay `rmsNormRope` (nor F-1's rules): it over-counts the normalised row (92.8 MiB per chain at
  33f) where that row is live at the peak.
