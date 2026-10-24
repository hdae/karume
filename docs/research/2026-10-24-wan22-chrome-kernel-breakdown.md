# Wan2.2 in Chrome on the RTX 5070 Ti: where the time goes, and the first small-op fusions (2026-10-24)

> A point-in-time measurement record. Numbers are from the runs named below; the decisions live in the
> perf ledger (section "2026-10-24 Wan2.2 DiT in Chrome") and ADR 0040 / ADR 0021 addenda.

## Setup

- Browser runs: Chrome 155 on Windows, RTX 5070 Ti (`nvidia` / `blackwell`), through the playground
  (`apps/playground`, `/wan22` and `/attention-bench`), distribution `models/karume-wan2.2` read through
  the dev server's HF-compatible route into CacheStorage. Records (not tracked by git):
  `outputs/bench-browser/wan22_2026-10-24T*.json`, `outputs/bench-browser/attention-bench_2026-10-24T*.json`.
- Node runs: Node 24 + Dawn on the development machine's RTX 3080 Ti (`nvidia` / `ampere`).
- Preset `i8-a8-attn8-online-conv8` (the Wan2.2 default), 1280×704, prompt `boxing-cats`, seed 42.

## 1. The wall clock of one 33-frame, 50-step generation (Chrome, RTX 5070 Ti)

| Stage | Before the browser cache | With the cache (second run) |
| --- | --- | --- |
| Text encoder (GPU, includes loading umT5) | 56 s | 4.1 s |
| DiT, first step (includes loading the DiT) | about 57 s | 9.3 s |
| DiT, later steps (median) | about 5.3 s | 5.27 s |
| VAE decoder | 54–65 s | 54 s |
| Whole request | 426 s | 326 s |

The first-run figures are dominated by moving about 11 GB of weights over an SSH tunnel; with the
weights in CacheStorage the per-request load is a few seconds.

## 2. Attention routes in isolation (`runAttentionRouteBench`, ms per 1,000 query rows)

| Shape (24 heads, D 128) | Node 3080 Ti s16 | Node online | Chrome 5070 Ti s16 | Chrome online |
| --- | --- | --- | --- | --- |
| self, N 7,920 (33 frames) | 4.12 | 3.66 | 2.97 | 3.87 |
| self, N 27,280 (121 frames) | 18.13 | 13.18 | 11.64 | 13.34 |
| cross, N 512 | 0.376 | 0.345 | 0.236 | 0.352 |

The online kernel runs at about the same absolute speed on both GPUs (about 25 TOPS), while the
three-dispatch s16 route gains 1.4–1.6× on the RTX 5070 Ti. In Chrome the online route is therefore
1.15–1.5× slower than s16. Both Chrome runs agree; both devices run the online kernel's `sg32` form
with the dp4a variant (Chrome grants 32 KiB of workgroup storage, Node 48 KiB). The cause is not
isolated yet (left for the attention tuning work).

## 3. Per-kernel GPU time of one DiT forward (Chrome, 33 frames, GPU timing on)

Record `wan22_2026-10-24T12-48-30.json` (100 forwards, 0 clamped samples; the median step 5.29 s
matches the untimed 5.27 s, so timing barely changes the run). Per forward, 2,605 ms in total:

| Op | ms | Share | Dispatches |
| --- | --- | --- | --- |
| linear (i8a8) | 1,080 | 41.5% | 310 |
| attention_online | 1,028 | 39.4% | 60 |
| elementwise (add 93, mul 76, gelu_tanh 36, where 31, neg 8) | 243 | 9.3% | 848 |
| strided / strided_write | 84 / 41 | 4.8% | 784 / 120 |
| quantize_rows | 82 | 3.2% | 490 |
| layer_norm / rms_norm | 23 / 23 | 1.8% | 91 / 120 |

Estimated efficiency: linear about 64 TOPS, attention about 22.5 TOPS (a third of linear).

## 4. Why the small ops were not fused, and the first slice

Of the existing fusion rules only `silu` fired on the Wan2.2 DiT. RoPE uses the even/odd pair form
(`slice, neg, slice, cat, reshape, mul, mul, add`), adaLN selects its scale and shift per token with
`where` (I2V's condition mask; T2V runs the same graph with the mask all false), and the gated
residual is `where, mul, add`. The first slice adds those three accepted forms (ADR 0040 addendum of
2026-10-24), each bit-identical to the plain sequence.

Node, RTX 3080 Ti, 2-step 1280×704×17, GPU timing on, four DiT forwards:

| Group | Before | After |
| --- | --- | --- |
| Fused groups (elementwise, strided, strided_write, layer_norm, RoPE, adaLN, gated residual) | 977 ms | 471 ms |
| Their dispatches | 7,872 | 4,488 |
| DiT total GPU time | 7,007 ms | 6,677 ms |

linear and attention moved by +2% / +7% between the two runs (the second ran warmer); the fusions do
not touch them. Every Wan sha256 row and Irodori's WAV rows are unchanged.

## 5. What remains (estimated per forward at 33 frames in Chrome)

- Attention: online slower than s16 in Chrome; the kernel runs at a third of linear's efficiency.
- Permutes around attention: about 56 ms.
- gelu into the down projection's quantize: about 36 ms.
- Residual add in the GEMM epilogue: about 46 ms (changes the GEMM WGSL per geometry).
- Quantized activations handed between ops (duplicate q/k/v quantize, adaLN emitting int8): about 39 ms.
- Cross-attention K/V and the text embedding recomputed every step although constant: about 13 ms.
