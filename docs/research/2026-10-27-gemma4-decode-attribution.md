> This document is a point-in-time snapshot (2026-10-27) of where the GPU time of gemma4 E2B decode and prefill goes,
> per op family, at context lengths P = 1K / 4K / 16K, measured on the RTX 3080 Ti under Node + Dawn (Vulkan). It is not
> a design source of truth. It answers the revival condition of the parked "online int8 attention (ADR 0132) for the
> states-form attention" item: attention ≥ about 25% of decode GPU time at chat lengths. Raw material (harness, raw JSON,
> logs) is under `outputs/diag/c-attribution-2026-10-27/` of the main checkout (not tracked).

# gemma4 E2B: decode and prefill GPU attribution by op family at P = 1K / 4K / 16K

## 1. Summary

- **The attention core (QK + softmax / row statistics + PV) crosses 25% of decode GPU time at about P ≈ 2.5K.** It is
  about 20% at P = 1,024, about 30% at P = 4,000 and about 56% at P = 16,000. The crossing is interpolated linearly in P
  between the two measured points that bracket it (1,024 and 4,000); the estimators of §5 put it between 1.9K and 2.6K, and a lower estimate that charges half of the grouped arm's extra
  pass cost to the core puts it at about 3.0K.
- Decode GPU time per token in the product pass structure: **6.35 / 7.29 / 12.67 ms** (median; min 6.30 / 7.19 / 11.97)
  at P = 1,024 / 4,000 / 16,000. Dispatches per token: 894 at every P (7 submits of ≤ 128 dispatches, H-40).
- Decode wall per token (no timing, PLE row cache warm): **8.0 / 10.0 / 14.9 ms** median (min 7.5 / 8.4 / 13.9).
  Karume's per-dispatch timing (ADR 0021) inflates it to 15.7 / 17.0 / 25.7 ms (×1.7–2.0) and the GPU sum by about
  3.6 ms per token (about 4 µs per dispatch).
- What grows with P is only the 7 full-attention layers' ①′ QK and ③′ PV. The 28 sliding-window layers (window 512) cost
  a constant ~1.06 ms/token (split-pass) — at P = 1K they are about two thirds of the attention core.
- At P = 16,000 the full-layer QK takes 497 µs per layer and grows 5.4× from P = 4,000 for 4× the length. Both full-layer
  kernels re-read the single KV head once per query head (8 query heads per KV head) — the K-55 premise for QK, and the
  same pattern in PV (§11).
- Prefill GPU (burst-free): 0.47 s / 1.93 s / 8.9–9.2 s for P = 1K / 4K / 16K. Linear is 81% / 77% / 66% of it; the
  attention core is 8% / 14% / 26%. **A 16K prefill trips this machine's sustained-load slow mode in 17 of 18 measured
  prefills** (2–4 chunks run 20–35× slower), so the measured TTFT at 16K is 37–66 s instead of about 11 s.
- The user's default capacity (4,096) stops P = 16K. The supported way to raise it is the per-turn option
  `chat(messages, { capacity })` (`Gemma4ChatOptions.capacity`; also `Gemma4SequenceOptions.capacity` and
  `Gemma4ChatSession`'s `capacity`), bounded by `chunkLength ≤ capacity ≤ maxPosition` (131,072). P = 16K used
  `capacity: 16384`. The allocated capacity does not change the cost at a given P (§7).

## 2. Setup

| Item | Value |
| --- | --- |
| Code | worktree `karume-wt-0.15`, branch `wip/0.15` at `eb5cce20` (K-90, H-40, H-41, H-38 / L-26 included); no product code changed |
| Machine | RTX 3080 Ti 12 GB (GA102), Ryzen 7600, 64 GB RAM; Fedora 44, Linux 7.2.7-200.fc44; NVIDIA driver 615.78.08; a 4K monitor on the GPU |
| Runtime | Node 24.22.1 + Dawn (npm `webgpu` 0.6.2), Vulkan backend; `packages/test-support/src/setup.ts` installs Dawn with timestamp quantisation disabled |
| Model | local mirror `models/karume-gemma4` (manifest `karume/5`, generator `karume/0.14.0`, `karume.json` sha256 `6de2039c…`), model `e2b`. The package's pinned HF revision is `64951755…` (`GEMMA4_SOURCES`); that the mirror equals it was not checked |
| Preset | the default quant **`i4-fast`**: packed int4 g32 linear, int8 embeddings, `linearGemvReduce: "parallel"`, `fuseRmsNormAdd: true`, `stateAttentionReduce: "parallel-fused"`. No drafter (none is built unless `speculative` is passed). Default sampler (temperature 1, top-k 64, top-p 0.95, seed 0 — host-side). PLE on the host (default residency and budget). The device is acquired as the pipeline does (`subgroups: false`, no extra features) |
| Shape | 35 layers: 28 sliding (window 512, head dim 256) and 7 full (head dim 512), 8 query heads per KV head; prefill chunk 768 rows with the default buckets |

## 3. Method

Harness `attr.ts` loads the pipeline with `Gemma4Pipeline.fromPretrained` from the local mirror. The source is a
`localDirectory` adapter that reads like `@karume/hub/node`'s `nodeDirectory` and logs each read with a time. The prompt is a
numbered lighthouse logbook (no two sections identical) plus a request for a long story. It is trimmed and padded to exactly
P tokens of chat-rendered prompt: P = 1,024 and 4,000 at the default capacity (4,096), and P = 16,000 at `capacity: 16384`.
Each process runs one warm-up chat (4 tokens, same prompt; compiles every shape) and then 2 measured chats of 64 tokens.
Every measured chat ran to `max-tokens`. Each chat re-runs the whole prefill.

Arms (one process per round × P × arm):

| Arm | What it measures | Pass structure |
| --- | --- | --- |
| plain | wall only (no timestamps, no diagnostics callback) | product |
| tap | pass-boundary timestamps added by the recon tap (`perf-recon-2026-10-24/llm-decode/tap.ts`); `timestamp-query` is hidden from karume, so the product encodes as usual → **product GPU busy** per token / chunk | product (7 passes per decode token) |
| grp | `seq-tap.ts` in grouped mode: a product pass is split only where the dispatch class changes (`attention_state_*` vs the rest) → **attention core timed directly**, about 2 extra pass boundaries per layer | product + 70 boundaries (77 passes per token) |
| seq | `seq-tap.ts`: every dispatch in its own timed pass, order and key kept → **per-role families** (PLE identified by position) | one pass per dispatch (same as ADR 0021's timing mode) |
| timed | karume's own timing device (`acquireGpu({ gpuTiming: true })`, `onRunDiagnostics` → `lastRunTiming` per pipeline key) — cross-check of seq | one pass per dispatch |

Rounds: 1–3 = plain / tap / timed / seq; 4–6 = grp / tap; 7–8 = the capacity control (tap / seq at P = 1K and 4K with
capacity 16,384). The P order and the arm order rotate per round. Before each process the driver waited for
`tools/diag/gpu-busy.zsh` to read < 10% busy and ≤ 60 °C. It waited 33 times (30 s each) for another session's GPU work. Every GPU
process ran inside `flock /tmp/karume-gpu.lock`. The idle check ran before the lock was taken and was not repeated after
a wait on the lock; every process started at 59–60 °C and 0% busy (`logs/run-all.log`).

Steady decode statistics pool the 62 steady tokens of each measured chat over the rounds: n = 372 per arm (744 for tap).
The first decode step of each chat is excluded: it includes the prefill → decode plan switch (plain median 35 / 36 / 46 ms).
Prefill statistics use per-chat totals. "Burst-free" = for each chunk index, the sample with the smallest time, summed.

Families (seq, by key and position): **attention core** = `attention_state_qk` (QK), `attention_state_stats` (row
statistics, full layers), `attention_state_pv` (PV, full layers), `attention_state_stats_pv` (fused row statistics + PV,
sliding layers); **attention other** = `state_append` (KV writes), `rope` (separate RoPE kernels), and in prefill the strided
head-layout copies; **linear** = every `linear*` kernel except PLE's, with lm_head (the last linear of a run) shown
separately; **norm** = `rms_norm*` (q / k / v norms included); **elementwise** = `ew:*` (logit softcap, layer scalar, FFN
gate); **PLE** = the prologue after the token embedding (per-layer projection, scale, norm + add, scale, layer-0 slice) and,
per layer, gate linear, gelu, multiply, projection linear, norm + residual and the layer's strided slice; **embedding** = token
embedding. **Sampling / top-k issues no GPU dispatch** with the default sampler (host-side on the read-back logits).

Calibration: one pass per dispatch adds a fixed cost per dispatch. The calibrated column assumes that cost is uniform:
c = (seq total − tap busy) / 894. Measured c is 4.0–4.2 µs at every P. Timestamps come in multiples of 1,024 ns (effective resolution about 1 µs); the
shortest timed dispatch, 4.1 µs, is 4 × 1,024 ns. The
estimate is family median − dispatches × c, against the product busy as denominator. The cost per added pass is not
uniform either: the grouped arm's 70 extra boundaries cost about 7.4 µs each (grouped total − product busy), against
about 4.0 µs per pass in the split-pass arm. For small kernels (norm, elementwise,
KV writes) the estimate is mostly the subtraction and is weak. The attention core is cross-checked by the grp arm.

Discarded: a first launch of the driver (5 processes, `raw/aborted-1/`) was stopped to add the ≤ 60 °C cooling wait and is
not used. The smoke runs (`raw/smoke*`) are not used. Nothing else is dropped. Slow-mode samples stay in, and the tables show
median, min and max.

## 4. Decode: GPU time per token

Product-structure totals and the attention core (ms/token, median / min):

| Quantity | P = 1,024 | P = 4,000 | P = 16,000 |
| --- | --- | --- | --- |
| product GPU busy (tap, n = 744) | 6.349 / 6.303 (max 6.49) | 7.286 / 7.193 (max 8.78) | 12.672 / 11.974 (max 16.97) |
| grouped-pass total (grp, n = 372) | 6.870 / 6.674 | 7.782 / 7.609 | 13.265 / 12.501 |
| attention core, grouped (direct) | 1.413 / 1.383 | 2.305 / 2.273 | 7.348 / 7.099 |
| attention core, calibrated | 1.258 | 2.168 | 7.167 |
| split-pass total (seq, n = 372) | 9.958 / 9.836 | 10.865 / 10.744 | 16.400 / 15.632 |
| karume timing total (timed, n = 372) | 10.014 / 9.876 | 10.930 / 10.763 | 16.153 / 15.710 |
| dispatches / passes per token (product) | 894 / 7 | 894 / 7 | 894 / 7 (8–9 in 4 of 744 tokens) |

By family (ms/token). Split-pass is measured (median / min). Calibrated is the product-structure estimate of §3:

| Family | 1K split-pass | 1K calibrated | 4K split-pass | 4K calibrated | 16K split-pass | 16K calibrated |
| --- | --- | --- | --- | --- | --- | --- |
| attn QK (35 dispatches) | 0.682 / 0.668 | 0.541 | 1.091 / 1.059 | 0.950 | 3.957 / 3.851 | 3.811 |
| attn row stats, full (7) | 0.058 / 0.054 | 0.030 | 0.096 / 0.091 | 0.068 | 0.236 / 0.220 | 0.206 |
| attn PV, full (7) | 0.210 / 0.203 | 0.182 | 0.663 / 0.653 | 0.635 | 2.625 / 2.491 | 2.595 |
| attn stats + PV fused, sliding (28) | 0.618 / 0.609 | 0.505 | 0.627 / 0.614 | 0.515 | 0.672 / 0.632 | 0.555 |
| attn KV writes (30) | 0.167 / 0.160 | 0.046 | 0.168 / 0.160 | 0.048 | 0.176 / 0.166 | 0.051 |
| attn RoPE (50) | 0.291 / 0.280 | 0.089 | 0.293 / 0.284 | 0.093 | 0.305 / 0.288 | 0.097 |
| linear (205) | 3.590 / 3.544 | 2.763 | 3.594 / 3.551 | 2.774 | 3.914 / 3.617 | 3.059 |
| lm_head (1) | 0.468 / 0.467 | 0.464 | 0.468 / 0.467 | 0.464 | 0.470 / 0.467 | 0.466 |
| norm (206) | 1.692 / 1.669 | 0.860 | 1.701 / 1.678 | 0.876 | 1.803 / 1.707 | 0.944 |
| elementwise (109) | 0.646 / 0.622 | 0.206 | 0.639 / 0.620 | 0.203 | 0.662 / 0.631 | 0.207 |
| PLE (214) | 1.508 / 1.466 | 0.644 | 1.508 / 1.478 | 0.652 | 1.575 / 1.492 | 0.683 |
| token embedding (2) | 0.011 / 0.010 | 0.003 | 0.011 / 0.010 | 0.003 | 0.012 / 0.011 | 0.004 |
| sampling / top-k | 0 dispatches (host) | | | | | |

The per-key karume timing (timed arm) agrees with seq: totals within 0.07 ms at 1K / 4K and 0.25 ms at 16K, and the
attention core within 0.02 / 0.02 / 0.11 ms (medians).

Attention kernels by layer type (split-pass, ms/token, and µs per dispatch):

| Kernel | P = 1,024 | P = 4,000 | P = 16,000 |
| --- | --- | --- | --- |
| full ①′ QK `attention_state_qk:…:par:gqa` (7) | 0.240 (34 µs, 520 wg) | 0.645 (92 µs, 2,008 wg) | 3.480 (497 µs, 8,008 wg) |
| full ③′ PV `attention_state_pv:…:par:gqa` (7) | 0.210 (30 µs) | 0.663 (95 µs) | 2.625 (375 µs) |
| full row stats (7) | 0.058 | 0.096 | 0.236 |
| sliding ①′ QK (28) | 0.442 (16 µs) | 0.445 | 0.476 |
| sliding fused stats + PV (28) | 0.618 (22 µs) | 0.627 | 0.672 |

At 16K the rest (non-attention) is about 0.45 ms higher than at 1K / 4K, and the minima are not (linear min 3.617 vs
3.544 / 3.551). Speculation: run state, not P. The 16K processes run hot after three 16K prefills (84–90 °C at process end,
sw_thermal / sw_power_cap flags set in most of them), and the 16K busy max (16.97 ms) shows occasional slow tokens.

## 5. Attention share of decode GPU time and the 25% line

| Estimator (core = QK + stats + PV) | P = 1,024 | P = 4,000 | P = 16,000 | P where 25% is crossed |
| --- | --- | --- | --- | --- |
| grouped-pass: core / grouped total (median; min–max) | 20.7% (20.3–21.1) | 29.6% (29.3–30.1) | 55.4% (53.3–56.8) | ≈ 2.46K |
| calibrated: core / product busy | 19.8% | 29.8% | 56.6% | ≈ 2.57K |
| upper bound: grouped core / product busy | 22.3% | 31.6% | 58.0% | ≈ 1.89K |
| lower: (grouped core − half of the 70 extra boundaries' cost) / product busy | 18.2% | 28.2% | — | ≈ 3.05K |
| split-pass raw (biased low: 3.6 ms of per-pass cost in the denominator) | 15.8% | 22.8% | 45.6% | ≈ 5.2K (between 4K and 16K) |
| core + KV writes + RoPE, calibrated | 21.9% | 31.7% | 57.7% | ≈ 1.96K |

The crossing is a linear interpolation in P between the measured 1,024 and 4,000 points (the split-pass row: between 4,000
and 16,000). The core does not grow linearly in P: its calibrated slope is 0.31 ms per 1K tokens from 1K to 4K and 0.42 ms
per 1K tokens from 4K to 16K. The interpolation is therefore approximate. No point between 1,024 and 4,000 was measured.

Read with the decision: at P = 1K–4K the core is 1.26–2.17 ms/token (calibrated). The sliding layers, whose cost does not
depend on P, are about 0.84 ms of it (calibrated; 1.06 ms split-pass). A kernel that replaces only the full layers'
attention addresses about 0.42 ms (1K) to 1.32 ms (4K) of decode GPU time per token (calibrated).

## 6. Decode wall per token and the cost of timing

Wall per token, second measured chat (PLE row cache warm: no file read after the first token), median / min (n):

| Arm | P = 1,024 | P = 4,000 | P = 16,000 |
| --- | --- | --- | --- |
| plain (no timing) | 8.04 / 7.45 (186) | 10.04 / 8.38 (186) | 14.85 / 13.91 (186) |
| tap (pass timestamps) | 7.61 / 7.41 (372) | 8.64 / 8.36 (372) | 14.26 / 13.45 (372) |
| grp | 8.10 / 7.86 (186) | 9.01 / 8.95 (186) | 14.37 / 13.86 (186) |
| seq (one pass per dispatch) | 13.05 / 12.44 (186) | 14.07 / 13.36 (186) | 19.39 / 18.45 (186) |
| timed (karume `gpuTiming: true`) | 15.70 / 13.29 (186) | 16.96 / 14.25 (186) | 25.65 / 19.62 (186) |
| plain, all steady tokens (both chats) | 8.59 / 7.45 (372) | 11.49 / 8.38 (372) | 15.06 / 13.48 (372) |

- Karume's timing mode inflates the decode wall ×1.95 / ×1.69 / ×1.73 (+7.7 / +6.9 / +10.8 ms) and the GPU sum by
  +3.7 / +3.6 / +3.5 ms (timed total vs product busy; split-pass vs product busy is +3.61 / +3.58 / +3.73 ms). The rest of the wall inflation is host-side (the two-stage wait of a
  timing device, per-pass encoding and the timestamp readback).
- The tap and grp arms do not change the wall measurably: per-process medians of the same arm spread by about ±1.5 ms at
  4K (plain: 9.4 / 8.8 / 11.8 ms over rounds 1–3), more than the difference between plain and tap.
- **Host PLE misses on the local-directory source still cost about 30 ms per affected token** after H-41. In the first
  measured chat of each process, 33 / 25 / 28 of 62 tokens read a file and took 38 / 41 / 44 ms (plain medians), against
  8.2 / 9.7 / 14.8 ms for the tokens of both measured chats that did not (10.4 ms at P = 1K counting the first chat
  alone). This is LD-2 (L-16 / L-17). The second chat repeats the same token
  ids (seed 0) and reads nothing.
- Wall minus product GPU busy (warm chat, medians; plain and tap) is about 1.3–1.7 / 1.4–2.8 / 1.6–2.2 ms per token: host encode, the logits
  readback and host sampling. Host sampling was not measured on its own.

## 7. Capacity control

At the same P, capacity 16,384 instead of 4,096 does not change the cost. The full-layer QK dispatches workgroups over the
live length (520 / 2,008 / 8,008 workgroups), not over the capacity.

| Quantity (median) | P = 1,024, cap 4,096 | P = 1,024, cap 16,384 | P = 4,000, cap 4,096 | P = 4,000, cap 16,384 |
| --- | --- | --- | --- | --- |
| product GPU busy / token | 6.349 | 6.358 | 7.286 | 7.283 |
| attention core, calibrated | 1.258 | 1.254 | 2.168 | 2.168 |
| prefill product GPU busy | 466.7 ms | 469.7 ms | 1,922 ms | 1,919 ms |

## 8. Prefill

Product GPU busy (tap, 12 samples per P) and wall:

| Quantity | P = 1,024 (2 chunks) | P = 4,000 (6 chunks) | P = 16,000 (21 chunks) |
| --- | --- | --- | --- |
| prefill GPU, median / min / max | 466.7 / 464.6 / 524.3 ms | 1,922 / 1,911 / 30,291 ms | 42,810 / 9,403 / 66,850 ms |
| burst-free (per-chunk-index min) | 464 ms | 1,907 ms | 8,886 ms |
| samples with slow chunks (> 3× the index min) | 0 of 12 | 4 of 12 | 11 of 12 (2–4 chunks each) |
| TTFT, plain arm, median / min | 593 / 585 ms | 2,183 / 2,153 ms | 40,602 / 37,419 ms |
| TTFT of the one 16K sample without an episode (tap, round 2, chat 2) | | | 10,983 ms (GPU 9,403 ms) |
| a 768-row chunk, first / last full chunk | 330 ms / — | 329 / 370 ms | 338 / 512 ms |

By family (split-pass, burst-free composite of 6 samples; per-dispatch timing costs < 2% here: 471 vs 467 ms at 1K):

| Family | P = 1,024 | P = 4,000 | P = 16,000 |
| --- | --- | --- | --- |
| total | 471.3 ms | 1,928 ms | 9,247 ms |
| linear (+ lm_head, last row only) | 383.4 + 0.9 (81%) | 1,476.9 + 2.8 (77%) | 6,141.2 + 10.1 (66%) |
| attention core (QK + stats + PV) | 38.9 (8.3%) | 265.8 (13.8%) | 2,353.9 (25.5%) |
| — of which QK / stats / PV | 18.4 / 3.3 / 17.3 | 124.7 / 15.7 / 125.4 | 1,116.5 / 117.9 / 1,119.5 |
| KV writes / RoPE / head-layout copies | 0.4 / 1.6 / 2.4 | 1.4 / 5.9 / 9.2 | 5.0 / 23.1 / 36.4 |
| PLE | 26.3 | 102.0 | 422.9 |
| elementwise | 10.2 | 39.1 | 155.0 |
| norm | 6.8 | 25.0 | 98.8 |

- Per chunk at 16K, the last full chunk (rows 14,592–15,359) is 534 ms: linear 310, attention core 187 (35%).
- In late 16K chunks of the burst-free composite, linear runs at about 310 ms against 276 ms in chunk 1. Linear does not
  depend on the context, so this is run state. Speculation: heat (the 16K processes end at 84–90 °C, most with thermal flags set). The 16K
  burst-free figure therefore still carries some of it.
- **The slow mode (research 2026-10-25 §9.3) dominates 16K prefill on this machine.** In 17 of 18 measured 16K prefills
  (tap and seq), 2–4 chunks (not always adjacent) took 8.8–16.2 s instead of 0.4–0.5 s. In those chunks the tiled GEMM linear ran ×33
  and the tiled attention QK ×36 (median; 7–46). That is the known signature: tiled GEMM-class kernels slow down and the others do
  not. Episodes began anywhere from chunk 1 to chunk 17. At 4K the slow mode hit 4 of 12 tap prefills and 0 of 6
  split-pass prefills.

## 9. Comparison with the earlier numbers

| Record | Then | Now (P = 16K unless stated) |
| --- | --- | --- |
| research 2026-09-06 (K-13 §2), Deno, timing on | decode GPU 20.45–20.87 ms/token at P ≈ 14.7K | split-pass 16.2–16.4 ms; product structure 12.7 ms |
| K-14, ①QK at P = 16K | 3.66–4.08 ms/token | 3.96 ms split-pass (3.81 calibrated) — unchanged |
| K-13 §3, prefill at 14.7K, Deno | 10.3 / 11.0 s GPU, linear 72% | 9.2 s split-pass burst-free at 16K, linear 66% |
| speed recon 2026-10-25 §9.1, short context | 6.2 ms GPU, 894 dispatches | 6.35 ms at P = 1K (non-core part ≈ 5.1 ms), 894 dispatches |

## 10. Not measured

- The browser product path (Chromium / Chrome, HF + CacheStorage), the RTX 5070 Ti, other presets (`i4`, `i4-gemvpar`),
  QAT and E4B, speculative decoding (drafter), and `pleResidency: "gpu"` (it cannot run on a timing device).
- Points between P = 1,024 and 4,000. The 25% crossing is interpolated, not measured.
- Per-family time in the product pass structure, except the attention core. WebGPU puts timestamps only at pass
  boundaries, so the small-kernel families (norm, elementwise, KV writes, RoPE) have only the calibrated estimate.
- Host-side sampling / top-k on its own (it is inside wall − GPU).
- What the online int8 kernel would save. No estimate is made here.
- A 16K prefill without slow-mode episodes is a single sample (tap, 9.40 s GPU, TTFT 10.98 s). The burst-free composite
  stands in for it.

## 11. Observations for the perf ledger (not filed)

- **K-55 (fold the per-head K re-read in decode ①′) — the "measure first" numbers.** Full-layer
  `attention_state_qk:…:par:gqa` costs 0.645 ms/token at P = 4K and 3.48 ms/token at P = 16K (497 µs per layer). That is
  27% of the product decode GPU at 16K. It grows 5.4× for 4× the length. Speculation: the 8 per-head reads of a 32.8 MB K
  plane fit L2 less and less (6 MB L2). The kernel code confirms the re-read: one workgroup per (query row, 16 columns,
  head z). At 16K, 8 × 32.8 MB per layer in 497 µs would be 528 GB/s of DRAM traffic.
- **The same pattern in ③′ PV, with no ledger row yet.** `attention_state_pv:…:par:gqa` uses one workgroup per (head z,
  16 depth lanes) and loops over every live column, so V is read once per query head. It costs 2.63 ms/token at 16K
  (375 µs per layer, 256 workgroups). The sliding fused stats + PV kernel has the same per-head structure (constant in P,
  0.62 ms/token).
- **The sliding layers set the attention floor at chat lengths.** 28 layers × (16 µs QK + 22 µs fused stats + PV) is about
  1.06 ms/token split-pass (about 0.84 ms calibrated) at every P. That is about two thirds of the core at P = 1K.
- **The slow mode makes 16K TTFT 3.4–6× worse on this machine** (17 of 18 prefills). It is filed as a machine fact
  (§9.3 of the 2026-10-25 recon), not as a karume defect. Long-prefill measurements here need per-chunk records.
- **LD-2 is still live after H-41 on the local-directory source:** a token that misses the PLE row cache pays about 30 ms
  of host reads.
- Method: one pass per dispatch costs 4.0–4.2 µs per dispatch on this machine (Dawn / Vulkan, about 3.6 ms per decode
  token). Per-key small-kernel times from `lastRunTiming` are therefore mostly that cost at decode shapes.

## 12. Raw data and harness

Everything is under `outputs/diag/c-attribution-2026-10-27/` of the main checkout (not tracked):

- `attr.ts` (harness; arms plain / tap / timed / seq / grp), `seq-tap.ts` (ordered per-dispatch and grouped-pass timing
  tap), and the reused tap `outputs/diag/perf-recon-2026-10-24/llm-decode/tap.ts`.
- `run-all.zsh` (rounds 1–3), `run-grp.zsh` (rounds 4–6), `run-control.zsh` (rounds 7–8): gpu-busy and cooling wait,
  then `flock`.
- `lib.mjs` (family and role rules), `analyze.mjs` (→ `raw/summary.json`), `perkey.mjs`, `prefill.mjs`, `wall.mjs`.
- `raw/r<round>-<point>-<arm>.json` (62 processes), `logs/` (per-process stderr, `run-all.log` with GPU state at each start
  and end, `analyze-all.txt`, `wall-all.txt`, `prefill-all.txt`); `raw/aborted-1/` and `raw/smoke*` are unused.
