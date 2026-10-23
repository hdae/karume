> This document is a point-in-time snapshot (2026-10-23) of a measurement of the Wan2.2 VAE decoder's cost and the
> candidates it suggests; it is not a design source of truth. One machine (RTX 3080 Ti, Node + Dawn on Vulkan), one run
> per chunk graph, synthetic inputs (the timing does not depend on the values).

# Wan2.2 VAE decoder — where the time goes

Question (the maintainer, 2026-10-23): the VAE stage of Wan2.2 is too slow — at 1280×704×121 the DiT stage takes about
13 minutes on the RTX 5070 Ti (50 steps) and the VAE stage about 8.6 minutes (bench of 2026-10-10: 772 s / 516 s). What is
the VAE's time made of, and what would cut it?

## 1. Summary

- One **next chunk** (4 output frames of one 256-px tile) costs **562 ms of GPU time, 85% of it in `conv3d`** (34
  dispatches, 478 ms), 10% in `conv2d` (the time-upsample path), 5% in everything else (elementwise, reductions,
  permutes, silu). A **first chunk** costs 240 ms (87% conv3d). The chunk graphs are the whole story; the host side
  adds about 25% (submit granularity and the per-chunk readback in the pipeline's batch form).
- The work is **2.08 T MAC of conv3d + 0.33 T MAC of conv2d per next chunk**, so conv3d runs at **4.4 T MAC/s = 8.7
  TFLOP/s, 25% of the f32 FMA peak** (34 TFLOP/s). That is the level of this machine's f32 GEMM skeleton in general
  (the geometry sweep of the same day: 9–11 TFLOP/s on large f32 linear, 29–35 TOPS on i8a8), not a conv3d-specific
  defect — the f32 path is the ceiling.
- At 1280×704×121 the tile plan is 4 × 7 = 28 tiles of 16 latent (256 px) with a minimum overlap of 4 latent, and
  each tile runs 1 first + 30 next chunks: 28 × (0.24 + 30 × 0.562) s = **479 s of GPU time**, matching the 516 s
  observed. The overlap makes the decoder compute **2.04× the pixels it outputs** (1024×1792 computed for 704×1280).
- Three levers, in the order of expected gain: int8 activations for conv3d (the `a8` path the linear and attention
  kernels already have — 3–4× on the conv3d share, with a quality check to do), a larger or run-time-chosen tile for
  devices with the memory (−35% of the work on 12 GiB cards), and f16 compute for conv3d (×1.3–1.5 on the share, the
  `c16` variant the GEMM skeleton has but conv does not).

## 2. Measurement

`packages/e2e/scratch-vae-timing.ts` (untracked, kept in `.claude/reviews/2026-10-11_wan-attention-recon/`): opens the
distribution mirror's `vae_decoder_first` / `vae_decoder_next` containers, builds a Session on a `gpuTiming` device,
runs each chunk three times with a random latent `[48,1,16,16]` and zero caches (33 inputs), and reads
`SessionDiagnostics.lastRunTiming` of the last run. `Session.run` reads back every output (the caches too), so the wall
time includes a readback the pipeline's batch form does not do.

| chunk graph | wall (3 runs) | GPU | dispatches | conv3d | conv2d | other |
| --- | --- | --: | --: | --: | --: | --: |
| `vae_decoder_first` (1 frame) | 902 / 715 / 656 ms | 239.9 ms | 407 | 209.0 ms (87%) | 16.1 ms | 14.8 ms |
| `vae_decoder_next` (4 frames) | 1,108 / 998 / 964 ms | 562.4 ms | 422 | 478.3 ms (85%) | 55.2 ms | 28.9 ms |

Every conv3d dispatch but one runs `conv3d:v1:f32:igemm64x128v4:wg16x8:wf16` (the default geometry; conv3d consults no
geometry profile — ADR 0118 decision 1); the "other" families are `reduce` (group-norm statistics), `ew`, `strided` /
`strided_write` (the permutes of the time axis), `silu`, `upsample2x` and the mid-block attention (3 small dispatches).

Work per chunk (from the IR, `scratch-vae-macs.ts`):

| chunk graph | conv3d | conv2d | effective (conv3d) |
| --- | --: | --: | --: |
| first | 33 nodes, 0.658 T MAC | 5 nodes, 0.088 T MAC | 3.1 T MAC/s |
| next | 35 nodes, 2.081 T MAC | 5 nodes, 0.330 T MAC | 4.4 T MAC/s (8.7 TFLOP/s) |

The heavy layers are the 1024-channel 3×3×3 convolutions at 32² (K = 27,648; 58 GMAC each in the next chunk, four
of them), the 512-channel ones at 64² (K = 13,824) and the 256-channel ones at 128² (K = 6,912). In the first chunk the
same layers have a quarter of the output columns (one frame instead of four), so their implicit GEMM has few column
tiles (for example 1024 outputs × 1024 channels = 128 workgroups on 80 SMs) and runs at 3.1 T MAC/s.

## 3. Reading

- **The f32 GEMM skeleton is the ceiling, not conv3d's addressing.** The same day's geometry sweep on this machine puts
  the f32 `linear` skeleton at 9–11 TFLOP/s for 4,096-row shapes with the default geometry and no candidate above
  ×1.05 on every case, while the i8a8 skeleton reaches 29–35 TOPS. conv3d's 8.7 TFLOP/s is that f32 level; a better
  im2col address computation would gain little.
- **The overlap costs 2.04×.** With tile 16 and minimum overlap 4 the effective stride is 12 latent, so the computed area
  is (16/12)² = 1.78× before edge rounding and 2.04× after it at 1280×704 (4 rows of tiles cover 64 latent for 44; 7
  columns cover 112 for 80). Tile 32 with overlap 4 would cover 1024×1536 for 2 × 3 = 6 tiles — 1.33× (−35% of the
  work); the cache residents grow fourfold (0.44 → 1.74 GiB per set, three sets ≈ 5.2 GiB; ADR 0121 kept tile 16 for the
  B570). The tile edge is baked into the exported chunk graphs (fixed shapes; the VAE **encoder** is symbolic-length),
  so a larger tile is a second export or a symbolic-shape export with a run-time choice by free memory.
- **int8 activations.** The runtime's dp4a path (`linearCompute: "a8"`) runs the same tile skeleton at 3–4× the f32
  rate on this machine; conv3d has no a8 variant (K-2 in the perf-ledger filed "Winograd / i8a8 for VAE conv2d" for
  Anima's VAE and is still open). With per-token (per output pixel over the im2col row) int8 activations and the f16
  weights quantised per output channel, the conv3d share (85%) could drop 3–4× — the VAE stage from about 480 s to
  about 170 s at 1280×704×121. The open question is quality: the decoder's int8 error compounds over 30 chunks through
  the causal cache; the A/B form of ADR 0110 (relRMS against the f32 decode of the same latents, plus the visual
  check) decides.
- **f16 compute.** ADR 0028 left `:c16` unwired for conv ("MAY be wired when measurements on another device show a
  gain"); on NVIDIA f16 FMA has twice the f32 rate, so the f32-bound skeleton would gain ×1.3–1.5 on conv3d. Precedent
  for quality: Anima's `f16-c16` preset computes its VAE in f16.
- **The host side (about 25%).** The pipeline's VAE stage at 1280×704×17 (two steps) took 94 s for 28 × 5 chunk runs =
  0.67 s per chunk against about 0.5 s of GPU time; the batch form already avoids the per-chunk readback, so the
  remainder is submit granularity (2,185 + 1,956 submits, 54 chunks over the window budget).

## 4. Candidates (filed in the perf-ledger as V-1 … V-4; the maintainer's ruling decides the order)

| id | candidate | expected gain at 1280×704×121 | cost / risk |
| --- | --- | --- | --- |
| V-1 | conv3d a8 (int8 activations through `quantize_rows`-style per-pixel scales, dp4a tile skeleton, f16 weights per-channel i8) | VAE 480 → ≈170 s (−65%) | a new kernel path on the i8a8 skeleton; quality A/B + visual check; the cache stays f32 |
| V-2 | a 32-latent tile for devices with ≥ 8 GiB free, as a second export or a symbolic-shape chunk graph chosen at run time | −35% of the VAE work (480 → ≈310 s) | recipe export + pipeline tile choice; more residents (≈7.5 GiB with weights) |
| V-3 | conv3d `:c16` (f16 compute) | ×1.3–1.5 on 85% (480 → ≈350 s) | wiring the existing f16 skeleton variant to conv3d; a `gpuFeatures.shaderF16` preset |
| V-4 | fewer, larger submits in the VAE stage (the window budget per chunk) | up to −20% of wall time | runtime submit policy; no numerics |

V-1 and V-2 compose (≈ 480 → 110 s). V-3 and V-1 do not (V-1 replaces the f32 multiply).

## 5. Not measured

- The per-layer split inside conv3d (the timing is per pipeline key; all layers share one key).
- The RTX 5070 Ti and the B570 (the tile-32 memory question is per device).
- The quality of int8 or f16 conv3d on real latents (no kernel exists yet).
