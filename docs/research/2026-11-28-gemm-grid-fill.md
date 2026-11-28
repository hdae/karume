> This document is a point-in-time snapshot (measurements of 2026-11-28, one RTX 3080 Ti, Node + Dawn / Vulkan) of the
> GEMM tile chosen from the dispatch grid's workgroup count instead of the row count alone: the recon's kernel timings
> of every tile on the default presets' shapes and the rule fitted to them. It is not a design source of truth: the decision is ADR
> [0157](../decisions/0157-gemm-tile-from-grid-fill.md), what runs is
> [spec/geometry-profiles.md](../spec/geometry-profiles.md). The recon's scripts and data are under
> `outputs/misc/gemm-geometry-recon-2026-11-28/` (`bench.ts`, `shapes.ts`, `fit.ts`, `pairs.ts`, `final.tsv`,
> `table.tsv`, `fit-rule.txt`), local and untracked; the numbers below are their output.

# The GEMM tile from the grid fill

## 1. What was asked

The geometry profile picks the GEMM tile of linear, matmul and bmm from the row count M alone (ADR 0115, 0116). The
item asked whether shapes with few outputs under-fill the dispatch grid under that rule, what a rule that sees N, K
and the batch would gain, and which design to build.

## 2. The recon's shapes and protocol

- **Shapes**: 125 classes, 103 linears and 22 bmms, of the default presets' GEMM-path nodes at their census storage:
  depth-anything v2 small; siglip2 base and so400m; lucida 1024; birefnet f16; EmbeddingGemma 2 (T = 256 and 64,
  assumed); Gemma 4 E2B in bf16; the QAT E2B / E4B / 12B / 26B A4B in i4 at group 32 and the mobile QAT's i2 / i8 at a
  768-row chunk; anima's text stages; sbv2 (T = 128, assumed); umT5 (L = 512); Irodori v4.1-small-mf (S = 185,
  assumed); and the sweep's own bucket blockers. The a8 linears are left out: they run the i8a8 path.
- **Reference**: the tile `nvidia-ampere` gives the class today (`gemmRowsGeometry` of its M), the tile a Session on
  this machine runs.
- **Stage 1** screened all 54 tiles of the sweep's grid once and compared their output words (u32) with the
  reference's. **Stage 2** interleaved the reference, a fixed ladder of seven tiles and the screen's best four in
  palindrome rounds (5, or 9 on a re-run); an arm's ratio is the median over rounds of its time over the reference's
  in the same round, both GPU timestamps.
- 34 shapes whose arms swung by more than ×1.2 across rounds were run again with 9 rounds, and 3 of them once more.

## 3. What the recon measured

- **Identity**: all 6,750 candidate rows (125 classes × 54 tiles) wrote the reference's words.
- **The tile a Session runs today is the fastest on 13 of the 125 classes.** A selection of the fastest tile over
  today's, per class (the full table is `table.tsv`): depth-anything fc2 (N 384, K 1536) ×0.486 with 64 × 64 / 256
  threads, its qkv (N 384) ×0.548, its PV bmm (N 64) ×0.432; so400m N 1152 K 1152 ×0.687, K 4304 ×0.692, its PV bmm
  (N 72) ×0.579; lucida M 1024 N 768 K 3072 ×0.675 (64 × 32), M 576 N 4608 ×0.784, its bmm at N 32 ×0.61 to ×0.63;
  EmbeddingGemma 2 M 256 N 512 ×0.79, M 64 N 2048 ×0.633; Gemma's 768-row chunk k / v (N 256) ×0.35 to ×0.38 over
  bf16, i4 and i8, N 512 ×0.45 to ×0.52, q (N 2048) ×0.70 to ×0.72, the 26B router (N 128) ×0.284, down / o
  (N 1536 to 3840) ×0.84 to ×0.98, gate / up at N ≥ 6,144 ×1.00 (128 × 128 stays the fastest, 64 × 64 is ×1.03 to
  ×1.16); umT5 M 512 N 10,240 ×0.79 with 128 × 128 against today's 64 × 32.
- **What predicts the fastest tile** is the grid's workgroup count W = ceil(N / tileN) · ceil(M / tileM) · batch, read
  pairwise down a ladder (`pairs.ts`): 64 × 64 against 128 × 128 is ×0.33 to ×0.55 where 128 × 128's grid has at most
  40 workgroups, ×0.7 to ×0.9 at 96 to 256, at parity near 288 and ×1.02 to ×1.14 at 384 or more; 64 × 32 against
  64 × 64 wins (×0.78 to ×0.86) only at W(64 × 64) ≤ 32, and so do 32 × 32 against 64 × 32 and, at ≤ 32 to 96, 16 × 16
  against 32 × 32. The exceptions are bmm with K 32 to 72 (×0.36 to ×0.62 even at 432 to 5,808 workgroups) and shapes
  that pad M or N by 2 to 4 times. A thread target (W × threads) fit worse than W.

### The fitted rule

Ladder 128 × 128 (r8 × 8, 16 × 16 threads) → 64 × 64 (r4 × 4, 16 × 16) → 64 × 32 (r4 × 4, 8 × 16) → 32 × 32
(r2 × 4, 8 × 16) → 16 × 16 (r1 × 4, 4 × 16): keep 128 × 128 when its grid has at least 256 workgroups, K ≥ 256 and
its padding (tile area over output area) is at most 1.5; else the first smaller tile whose grid has at least 48
workgroups at a padding of at most 1.5; else the last tile. Scored against the fastest tile of every class, weighted by
today's kernel time × the census count (`fit.ts`):

| Rule | Weighted time vs today | Geomean regret | Worst regret | Classes slower than today by > 3 % |
| --- | --- | --- | --- | --- |
| Today's M buckets | 1.000 | 1.252 | 3.94 | — |
| The fitted rule (256, 48) | 0.880 | 1.027 to 1.029 | 1.36 (a 13 µs siglip2 bmm) | 1 |
| The fastest tile of every class | 0.877 | 1.000 | 1.00 | — |
| One threshold only (W ≥ 192) | 0.894 | 1.107 | — | 16 |
| Stepping down from today's tile only | 0.909 to 0.932 | — | — | — |

Leaving each of the 17 families out of the fit picks (256, 48) again for 16 of them ((320, 48) for Gemma 4 E2B in bf16),
and every held-out score is within 0.005 of the full fit's.

### Noise

Over 974 interleaved (shape, arm) pairs, the round-to-round max ÷ min of the ratio is p50 1.032, p75 1.057, p90 1.18,
p95 2.48, max 65. About 10 % of arms hit a transient slow state in some round (lucida M 17,424 N 1152 with 64 × 64 read
×3.6 in all five rounds of one run and ×0.96 on the re-run). The median of nine interleaved rounds recovered every case
run again.

## 4. Who would gain (the recon's arithmetic, kernel level)

Sum over the measured classes of today's kernel time × count, under the fitted rule. The kernel savings are measured;
the end-to-end shares are inferences.

| Family | Classes that change tile | GEMM time per forward today | Saved by the rule | End-to-end share (inference) |
| --- | --- | --- | --- | --- |
| depth-anything v2 small (f32) | 5 of 5 | 18.9 ms | 8.5 ms (×0.549) | no breakdown on record |
| siglip2 so400m (f32) | 5 of 5 | 97.3 ms | 26.5 ms (×0.728) | about 21 % of the GPU time per image |
| lucida 1024 (f32) | 15 of 25 | 208 ms | 40 ms (×0.81) | — |
| EmbeddingGemma 2 text (bf16, T = 256) | 4 of 7 | 17.7 ms | 2.9 ms (×0.84) | M is assumed |
| Gemma 4 E2B bf16 / QAT i4 (768 rows) | 7 of 9 / 6 of 8 | 281 / 280 ms | 21.8 / 21.2 ms (×0.92) | about 6.5 % of the prefill GPU time |
| Gemma 4 QAT E4B i4 | 6 of 7 | 583 ms | 73 ms (×0.875) | about 10 % of linear time per chunk |
| Gemma 4 QAT 12B i4 | 5 of 6 | 1,524 ms | 151 ms (×0.90) | about 10 % of linear time per chunk |
| Gemma 4 26B A4B, dense part | 7 of 7 | 268 ms | 58 ms (×0.78) | the experts run `expert_linear` |
| umT5-XXL (i8, L = 512) | 5 of 5 | 500 ms | 67 ms (×0.87) | one encode per Wan request |

Decode is untouched: M = 1 runs the GEMV family.
