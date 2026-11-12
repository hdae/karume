> This document is a point-in-time snapshot (measurements of 2026-11-11 and 2026-11-12) of six performance candidates
> and one session A/B: S5 / F-13, V-10, K-85, K-88, K-106 and the `gelu_mul` A/B of K-105. It is not a design source of
> truth: what was adopted lives in the ADR addenda it cites, and each candidate's status in its
> [perf-ledger](../perf-ledger.md) row. The records it reads are under `outputs/diag/` and `outputs/bench-browser/`
> (local, untracked); every number below was re-computed from the record named next to it.

# Fusion rules and kernel forms on the RTX 3080 Ti and the RTX 5070 Ti

## 1. Summary

- **Adopted** (the maintainer, 2026-11-12; ADR [0148](../decisions/0148-external-fusion-rules.md) addenda of that date):
  the elementwise-chain rule `models/ew_chain` (S5 / F-13) on MOSS-Image, and the VAE's channel RMS norm + SiLU rule
  `models/vae_rms_silu` (V-10) on the Wan and anima VAE decoders. Both were u32-identical to the unfused plan in every
  run, and both cleared the session A/B's 1 % line on both GPUs (§3, §4).
- **Not adopted**: `candidate/gelu_mul` (K-105) gained under 1 % of a 26B generation on both GPUs (§5). The tiled key
  axis of the online attention (K-106) gained ×1.007–1.018 on the RTX 5070 Ti, and on the RTX 3080 Ti it missed the
  ×1.05 line on MOSS-Image's shape, so the derivation keeps today's form everywhere (§8).
- **Landed as optional geometry fields, carried by no table**: K-85 (the i8a8 skeleton) and K-88 (the f32 GEMM forms and
  the conv2d counter decode). Their gains on the RTX 5070 Ti are a fraction of the RTX 3080 Ti's: ×1.060 against ×1.22
  for one Wan2.2 DiT forward's linears (K-85), ×1.083 against ×1.29 for a gemma4 prefill chunk (K-88). On the 5070 Ti
  the decode is the larger part of K-88's conv2d gain (§6, §7).
- **Side finding** (§7.3): on the RTX 5070 Ti the `nvidia-blackwell` table's conv2d geometry is slower than another
  plain geometry on every Wan VAE shape of the K-88 set, and its `gemmRows` geometry is slower than another on the
  gemma4 down projections.
- **Not measured**: Metal for every candidate; the cause of the NVVM compile-failure message that K-88 waits on (§9).

## 2. Setup

| GPU | Runtime and backend | How the records were taken |
| --- | --- | --- |
| RTX 3080 Ti | Node 24 + Dawn on Vulkan (driver 615.78.08) | opt-in tests under `packages/e2e/tests/`, one GPU run at a time |
| RTX 5070 Ti | Chrome 157 on Windows, D3D12 | the playground's `/session-ab`, `/geometry-sweep` (K-85 / K-88 presets), `/attention-bench`, `/kernel-check` and model pages |

Three measuring tools, each with its own reading:

- **Session A/B** (`runSessionAb` / `judgeSessionAb` of `@karume/runtime/tune`; record format `karume-session-ab/1`).
  Each timed round visits the arms in the order A B B A. A round's difference is the mean of the compared arm's two
  visits minus the base's; the reported figure is the median of the rounds' differences, and the gain is that median
  over the median of the base's round means. The verdict is "adopt" only when every output word of every run equals
  the base's, the median is beyond the noise floor (the larger of the two arms' median gaps between their own two
  visits of a round), the order-statistic interval of the median excludes 0, and the gain is at least 1 %.
- **Geometry sweep** (`karume-geometry-sweep/2`). GPU timestamps per dispatch, five rounds per geometry; every row's
  output is compared with the default geometry's. A case's drift is the default geometry measured again at the end of
  the case over its first measurement; the table derivation reads only cases with drift inside 0.9–1.1 (ADR
  [0115](../decisions/0115-geometry-profiles.md) addendum decision 8). The K-85 and K-88 evaluation sets have case
  tables of their own (`packages/runtime/src/tune/cases.ts`), each case with the node count it stands for in a unit
  (one Wan2.2 DiT forward; one gemma4 prefill chunk, one VAE chunk and so on), so a unit's time is the sum of count ×
  time per dispatch.
- **Attention bench** (`runAttentionRouteBench` with `onlineTiled`). The two kernels in A B B A rounds (the median of the
  per-round ratios), their two producers alike, and the identity of both pairs. The profile derivation reads the gain
  of kernel + producer per node from each arm's minimum pass (ADR [0117](../decisions/0117-app-geometry-tuning.md)
  addendum of 2026-11-12).

## 3. S5 / F-13: the elementwise-chain rule on MOSS-Image

The rule folds the longest run of consecutive f32 scalar-free elementwise nodes into one dispatch built from the plain
kernels' own value text, with every internal value behind the rounding barrier. On MOSS-Image's `dit` graph it takes
292 runs per forward (F-10's `situ` gate, the tail of F-11's RoPE, the gated residual and adaLN's tails); the pinned
counts per graph are in `packages/e2e/tests/e2e_moss_fusion_counts_test.ts`.

One `dit` forward at 4,096 image + 63 text tokens, practical preset `f32+dit8-a8-attn8-online`, seeded feeds, 4 rounds;
dispatches 2,466 → 1,880 in every record, every output identical on all 20 runs of each arm:

| GPU | Record | Base median | Median difference | Gain | Noise floor | Verdict |
| --- | --- | --: | --: | --: | --: | --- |
| RTX 3080 Ti | `outputs/diag/session-ab-moss-2026-11-11/session-ab-moss_2026-11-11T19-59-26.148Z.json` | 1,792.3 ms | −171.7 ms | 9.6 % | 15.3 ms | adopt |
| RTX 3080 Ti | `…/session-ab-moss_2026-11-11T20-18-40.198Z.json` (after the MOSS-Image lane) | 1,809.4 ms | −181.9 ms | 10.1 % | 16.1 ms | adopt |
| RTX 5070 Ti | `outputs/bench-browser/session-ab-moss_2026-11-12T01-09-56.json` | 1,329.4 ms | −171.1 ms | 12.9 % | 8.5 ms | adopt |

Whole generations, with the rule passed through the pipeline's experimental `fusionRules` before the adoption:

- RTX 3080 Ti, 256², 2 steps, seed 42, warm-up then 3 rounds (`outputs/diag/pipeline-rules-2026-11-11/pipeline-rule-ab-moss_2026-11-12T01-17-55.084Z.json`):
  every image identical; median difference −169.8 ms of 17,775.5 ms (0.96 %) under a noise floor of 227.0 ms,
  inconclusive. A generation this small is mostly the weight upload, so it shows identity, not the gain.
- RTX 5070 Ti, 1024², 50 steps, seed 42, one generation per arm, the rule arm first
  (`outputs/bench-browser/moss-image_2026-11-12T05-45-08.json` with the rule, `…T05-48-23.json` without): the same RGBA
  hash, 131.0 → 121.5 s (−7.3 %).

## 4. V-10: the VAE's channel RMS norm + SiLU

The channel RMS norm and the SiLU after it in one dispatch instead of eight, in two geometries (`g8` = 8 channel
groups × 32 pixels per workgroup, `g4` = 4 × 64). 30 runs per decoder graph in every record below; dispatches
457 → 248 on the Wan2.2 next chunk and
303 → 94 on an anima tile; every output identical on all 20 runs of each arm.

| GPU | Target | Record | Base median | `g8` | `g4` | Noise floor (`g8` / `g4`) | Verdict |
| --- | --- | --- | --: | --: | --: | --: | --- |
| RTX 3080 Ti | Wan2.2 next chunk, run 1 | `outputs/diag/session-ab-vae-2026-11-11/session-ab-vae-wan22-vae-next_2026-11-11T20-44-21.479Z.json` | 623.9 ms | −15.0 ms (2.4 %) | −8.5 ms (1.4 %) | 19.3 ms | inconclusive |
| RTX 3080 Ti | anima tile, run 1 | `…/session-ab-vae-anima-vae-decode_2026-11-11T20-45-56.810Z.json` | 148.7 ms | −15.8 ms (10.6 %) | −18.0 ms (12.1 %) | 12.4 ms | inconclusive |
| RTX 3080 Ti | Wan2.2 next chunk, run 2 | `…/session-ab-vae-wan22-vae-next_2026-11-11T20-47-26.746Z.json` | 583.8 ms | −19.7 ms (3.4 %) | −22.1 ms (3.8 %) | 10.8 ms | adopt |
| RTX 3080 Ti | anima tile, run 2 | `…/session-ab-vae-anima-vae-decode_2026-11-11T20-48-55.490Z.json` | 149.1 ms | −21.1 ms (14.2 %) | −17.0 ms (11.4 %) | 5.6 ms | adopt |
| RTX 5070 Ti | Wan2.2 next chunk | `outputs/bench-browser/session-ab-vae-wan22-vae-next_2026-11-12T01-12-45.json` | 473.9 ms | −13.2 ms (2.8 %) | −10.8 ms (2.3 %) | 6.7 / 3.9 ms | adopt |
| RTX 5070 Ti | anima tile | `outputs/bench-browser/session-ab-vae-anima-vae-decode_2026-11-12T01-15-17.json` | 128.2 ms | −15.4 ms (12.0 %) | −15.2 ms (11.9 %) | 0.7 / 1.4 ms | adopt |

Run 1 on the anima tile has visits of 4.0–5.0 s against about 0.14 s elsewhere in the same record, so its rounds
disagree (the interval of the median spans −2,418.9 to +1,917.6 ms). The two geometries are within each other's noise
floor on the RTX 5070 Ti; `g8` was kept (the ADR 0148 addendum on `models/vae_rms_silu`).

Whole generations, with `_g8` passed through the pipelines' experimental `fusionRules`:

- RTX 3080 Ti, `outputs/diag/pipeline-rules-2026-11-11/`: anima 512², seed 42 (`pipeline-rule-ab-anima_…T01-15-08.251Z.json`):
  every image identical; median difference −113.7 ms of 5,238.4 ms (2.2 %) under a noise floor of 266.2 ms,
  inconclusive. Wan2.2, the VAE stage alone on the request's final latents (1280×704, 17 frames, 2 steps,
  `boxing-cats`, seed 42; `pipeline-rule-ab-wan-ti2v_…T01-24-53.515Z.json`): 840 runs per chunk graph, every decoded
  video identical; the decodes take 87.9–174.2 s and the median difference −4.0 s of 132.1 s (3.0 %) is under a noise
  floor of 53.9 s, inconclusive. The per-chunk A/B above is the number for this GPU.
- RTX 5070 Ti, one generation per arm (`outputs/bench-browser/wan22_2026-11-12T06-01-23.json` without,
  `…T06-10-06.json` with; `anima_2026-11-12T06-11-57.json` without, `…T06-12-29.json` with): Wan2.2 1280×704, 33
  frames, 50 steps, VAE tile 16, seed 42 — the same RGBA hash, the median of the 27 VAE tiles after the first
  1.348 → 1.274 s (−5.5 %), the VAE stage 41.9 → 39.2 s; anima 1024², 8 steps — the same RGBA hash, the median of the
  8 tiles after the first 0.116 → 0.102 s (−12 %), the VAE stage 1.295 → 1.083 s, VAE dispatches 2,727 → 846.

## 5. K-105: the `gelu_mul` session A/B on Gemma 4 26B

The candidate rule `candidate/gelu_mul` (`gelu_tanh` then `mul`, 60 runs per forward; dispatches 1,414 → 1,354) on 26B
`i4` under the expert cache (64 slots), 96 greedy tokens per generation, 4 rounds, every token identical:

| GPU | Record | Base median | Median difference | Gain | Noise floor | Verdict |
| --- | --- | --: | --: | --: | --: | --- |
| RTX 3080 Ti | `outputs/diag/session-ab-2026-11-11/session-ab_2026-11-11T11-34-04.360Z.json` | 7,423.3 ms | −22.2 ms | 0.30 % | 109.9 ms | inconclusive |
| RTX 3080 Ti | `…/run2/session-ab_2026-11-11T12-25-26.270Z.json` (reads the counts) | 7,429.8 ms | −67.1 ms | 0.90 % | 71.8 ms | inconclusive |
| RTX 3080 Ti | `…/run3/session-ab_2026-11-11T12-52-37.103Z.json` | 7,140.6 ms | −63.2 ms | 0.89 % | 82.5 ms | inconclusive |
| RTX 5070 Ti | `outputs/bench-browser/session-ab_2026-11-11T13-24-09.json` | 6,260.2 ms | −18.2 ms | 0.29 % | 7,160.5 ms | inconclusive |

Three visits of the RTX 5070 Ti record took 20.4–23.0 s against 6.2–6.3 s for the others; with them set aside, the
mean visit is 6,233.9 ms without the rule and 6,218.5 ms with it (0.25 %). Under 1 % on both backends, so the rule is
not adopted; it stays in `packages/candidate-rules` as the example of the external-rule API.

## 6. K-85: the vec4 skeleton of the i8a8 linear

The optional field `I8a8Geometry.skeleton: "vec4"` (ADR [0115](../decisions/0115-geometry-profiles.md) addendum
decision 12), swept with the evaluation set `k85`: the Wan2.2 TI2V-5B DiT linears at 33 frames with their counts (303
nodes of one forward) and an Irodori M = 185 guard (count 0), 207 rows per case, 1,242 rows, every row identical
to the default on both GPUs. "Best skeleton" is the fastest identical skeleton geometry of a case.

| Case (count) | RTX 3080 Ti: `nvidia-ampere` ÷ best skeleton | RTX 5070 Ti: `nvidia-blackwell` ÷ best skeleton |
| --- | --: | --: |
| qkvo, M 7,920 N 3,072 K 3,072 (180) | ×1.115 | ×1.058 |
| MLP up, N 14,336 (30) | ×1.165 | ×1.076 |
| MLP down, K 14,336 (30) | ×1.416 | ×1.046 |
| cross, M 512 (61) | ×1.037 | ×1.060 |
| time projection, M 1 (2) | ×1.840 | ×2.400 |
| Irodori guard, M 185 (0) | ×1.664 | ×1.869 |
| **One forward's linears** | **1,874.6 → 1,532.7 ms (×1.22)** | **1,111.4 → 1,048.2 ms (×1.060)** |
| One skeleton geometry for every case | `tile96x128r12x8w16x8k64s`, 1,578.0 ms (×1.19) | `tile128x64r8x8w8x16k64s`, 1,048.5 ms (×1.060) |

Records: RTX 3080 Ti `outputs/diag/k85-linear-2026-11-11/geometry-sweep_k85_*.json` (one per case); RTX 5070 Ti
`outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_k85_2026-11-12T02-04-54.json`. The table geometries are
`tile64x64r8x4w16x8k16` (`nvidia-ampere`) and `tile128x64r8x4w16x16k16` (`nvidia-blackwell`).

Reading: on the RTX 3080 Ti the five forward cases drifted ×1.14–1.21 (the guard ×1.09), outside the derivation's
range, so geometries late in the candidate order are penalised and its ratios are an evaluation, not adoption evidence.
On the RTX 5070 Ti the drift is at most ×1.014, and the forward gain is ×1.060 — close to the ×1.070 a K64 geometry
re-sweep without the skeleton gave there on 2026-10-25 ([5070 Ti record](2026-10-25-speed-recon-5070ti.md) §7), with
the same tile and workgroup (r8×8, w8×16, k64). The M 1 and M 185 rows gain most, which is
what row-count buckets for `i8a8.linear` would take; with one geometry for every M (today's profile shape) a table
cannot give them their own tile. No table was derived; the ledger row lists what stays.

## 7. K-88: the f32 GEMM forms and the conv2d counter decode

The optional fields `GemmGeometry.form` (`tb-un` / `tas-tb-un`) and `GemmGeometry.decode: "counter"` (conv2d only; ADR
[0115](../decisions/0115-geometry-profiles.md) addendum decision 13), swept with the evaluation set `k88`: each case
against the plain geometries the default and the registered tables give it, without the fields, with each form and,
for conv2d, with the decode alone and with each form. 159 rows, every row identical to the default on both GPUs.
Records: RTX 3080 Ti `outputs/diag/k88-gemm-2026-11-11/geometry-sweep_k88_*.json` (one per case, drift ×0.987–1.053,
measured one candidate after another); RTX 5070 Ti
`outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_k88_2026-11-12T02-36-46.json` (drift ×1.000–1.020).

### 7.1 Linear and matmul (table geometry ÷ the same geometry's best form)

| Case | RTX 3080 Ti | RTX 5070 Ti |
| --- | --: | --: |
| gemma4 E2B gate / up, M 768 (N 6,144 / 12,288) | ×1.288 / ×1.304 | ×1.088 / ×1.083 |
| gemma4 E2B down (K 6,144 / 12,288) | ×1.262 / ×1.263 | ×1.078 / ×1.080 |
| gemma4 E2B q (N 2,048 / 4,096) | ×1.348 / ×1.312 | ×1.084 / ×1.096 |
| **one gemma4 prefill chunk's measured linears (140 of 277)** | **234.3 → 181.5 ms (×1.29)** | **253.4 → 234.0 ms (×1.083)** |
| umT5-XXL wi, M 512, int8 (r4×4) | ×1.001 | ×1.014 |
| siglip2 fc1, M 729, f32 | ×1.322 | ×1.123 |
| M 64 guard (r1×4) | ×1.015 | ×0.996 |
| 4,096³ linear / matmul | ×1.321 / ×1.225 | ×1.088 / ×1.172 |

### 7.2 conv2d (one unit's time: plain → best form / decode alone / best of both)

| Unit | RTX 3080 Ti | RTX 5070 Ti |
| --- | --: | --: |
| Wan2.2 VAE next chunk (3 conv2d) | 55.1 → 49.7 / 49.0 / 39.9 ms | 54.9 → 53.5 / 45.9 / 44.7 ms |
| Wan2.2 VAE first chunk (3 conv2d) | 16.3 → 14.4 / 14.8 / 12.0 ms | 16.0 → 15.3 / 13.2 / 12.6 ms |
| anima VAE decode (11 conv2d) | 40.8 → 34.8 / 36.7 / 29.9 ms | 38.0 → 37.7 / 33.7 / 32.2 ms |

On the RTX 3080 Ti the form and the decode add up to about ×1.33–1.39 per conv2d case; on the RTX 5070 Ti the decode
alone gives ×1.10–1.30 per case and the form alone ×1.00–1.12. No NVVM message appears in
the RTX 3080 Ti run's log (`outputs/diag/k88-gemm-2026-11-11/run.log`).

### 7.3 Side finding: the `nvidia-blackwell` table's own geometries

Both K-88 records also time a second plain geometry per case, the one the `apple-metal-3` table gives (for `gemmRows`
above 512 rows `reg128x32r8x4w8`, for `gemmRows` 33–64 `reg64x32r4x4w8`, for conv2d `igemm128x64:wg16x16`). For the
buckets of these cases `nvidia-blackwell` keeps the default geometry (`reg128x128r8x8w16`, `reg16x16r1x4w4`,
`igemm64x128:wg16x8`). In the RTX 5070 Ti record (table geometry ÷ the other, plain; above 1 = the table is slower):

| Case | Plain | Best with the fields on each |
| --- | --: | --: |
| Wan2.2 next b2 c1024 32² / b4 c1024 64² / b4 c512 128² | ×1.130 / ×1.168 / ×1.170 | ×1.022 / ×1.037 / ×1.053 |
| Wan2.2 first c1024 32² / c1024 64² / c512 128² | ×1.497 / ×1.172 / ×1.182 | ×1.176 / ×1.041 / ×1.042 |
| anima c192 256² / c384 128² | ×0.856 / ×1.171 | ×0.790 / ×1.032 |
| gemma4 down K 6,144 / K 12,288 | ×1.260 / ×1.264 | ×1.379 / ×1.378 |
| gemma4 gate / up / q (four cases) | ×0.881–0.981 | ×0.874–1.029 |
| M 64 guard | ×1.491 | ×1.598 |

So on that GPU the table's conv2d geometry loses on every Wan shape, with or without the fields, and its `gemmRows`
geometry loses on the gemma4 down projections while it wins on gate, up and q, siglip2 and the square shapes. The
table's buckets are decided on the general sweep's cases (the reasons in `nvidia-blackwell.ts`), not on these shapes,
so this is a question for its buckets, not for the K-88 fields. On the RTX 3080 Ti the same comparison goes the other
way on most cases (the table's conv2d geometry wins on four of the six Wan shapes, its `gemmRows` geometry on every
case but the M 64 guard, ×1.786).

## 8. K-106: the tiled key axis of the online attention

A tiled `quantize_rows` producer writes Vᵀ in the kernel's key order, so the kernel's Vᵀ owner fill is one vec4 load
(`AttentionOnlineKeyWords` `tiled`). Every record has every O word and scale word identical.

| GPU | Record | Shape | Kernel tiled ÷ online (median of rounds) | Kernel + producer gain (minimum passes) |
| --- | --- | --- | --: | --: |
| RTX 3080 Ti | `outputs/diag/attn-tiled-2026-11-12/run1/`, `run2/` | self-attention 33f | ×0.903 / ×0.933 | ×1.090 / ×1.054 |
| | | self-attention 121f | ×0.899 / ×0.772 | ×1.098 / ×1.140 |
| | | cross-attention 121f | ×0.394 / ×0.915 | ×2.606 / ×1.092 |
| | | MOSS-Image 1024² | ×0.947 / ×0.935 | ×1.044 / ×1.048 |
| RTX 5070 Ti | `outputs/bench-browser/attention-bench_2026-11-12T05-27-48.json` | self 33f / self 121f / cross 121f | ×0.983 / ×0.993 / ×0.983 | ×1.012 / ×1.013 / ×1.019 |

Reading:

- In both RTX 3080 Ti runs some passes took several times their usual time (run 1, 33f: today's kernel at 132.7–133.4
  ms in its first three passes against 20.0–22.2 ms afterwards; run 2, 121f: 147.2 ms and about 97 ms against 58.7–62.6
  ms for most passes), which moves the median of the rounds (×0.772 at 121f in run 2) and is why the derivation reads
  minimum passes. Run 1's cross shape has no clean pass of today's kernel (29.96–30.13 ms, then 10.76–10.81 ms, against
  the 4.66 ms of the same record's online row), so its ×2.606 overstates.
- The tiled producer is slower than today's: ×1.03–1.04 at 33f and cross, ×1.26 at 121f and ×1.07–1.09 at MOSS-Image's
  shape on the RTX 3080 Ti; ×1.03, ×1.08 and ×1.05 on the RTX 5070 Ti (medians of the producer pair's rounds).
- The derivation (the geometric mean of a shape's observations, ×1.05 on every shape) keeps today's form on the RTX 3080
  Ti, because MOSS-Image's shape gains ×1.046, and on the RTX 5070 Ti, whose record has no MOSS-Image shape and gains
  at most ×1.019. `nvidia-ampere` is not regenerated (ADR [0132](../decisions/0132-online-int8-attention.md) addendum of
  2026-11-12).
- `/kernel-check` on the RTX 5070 Ti (`outputs/bench-browser/kernel-check_2026-11-12T05-24-59.json`): the tiled
  producer (18 cases) and the `sg32` kernel (28 cases) identical; the `wg` part did not run, as its form declares
  35,328 B of workgroup storage and the device grants 32,768 B.
- The two RTX 3080 Ti runs of 2026-11-11 (`outputs/diag/attn-tiled-2026-11-11/run1/`, `run2/`, before MOSS-Image's
  shape was in the set) agree on the kernel: ×0.932 / ×0.912 (self 33f), ×0.920 / ×0.916 (self 121f), ×0.914 (cross,
  run 2; run 1's median is void for the same reason as above).

K-82's spike of 2026-10-23 had estimated about 10 % for this key order, with a timing-only variant on the kernel before
K-84 ([FlashAttention recon](2026-10-23-flash-attention-webgpu-recon.md) §10.2); the kernel after K-84 gains less.

## 9. Not measured

- **Metal**: no record here comes from Metal, for any of the candidates. `models/ew_chain` and `models/vae_rms_silu`
  run there by the maintainer's ruling; where the XOR rounding barrier does not hold, their output can differ from the
  unfused plan ([known issues](../known-issues.md)).
- **The NVVM message** (`NVVM compilation failed: 3`, then `failed to compile internal representation`): the earlier
  investigation records hold it, some from runs without K-88 code (the ledger row K-88 counts them). The RTX 3080 Ti
  K-88 run's log above does not show it and the RTX 5070 Ti record has no failed row, so it was not reproduced here;
  the exit-path A/B that would attribute it has not run. Until it is explained no K-88 field becomes a default.
- K-88's gemma4 TTFT on the RTX 5070 Ti; the scalar kernels, the f16 compute variant, the fused attention, conv1d and
  conv3d under the K-88 fields.
- The tables' derivation with the K-85 / K-88 fields as candidates (the RTX 5070 Ti through the playground presets, a
  cooled Node sweep on the RTX 3080 Ti), and the sha256 re-checks after it.
- `models/ew_chain` outside MOSS-Image; `models/vae_rms_silu` on Wan2.2's VAE encoder (planned, not measured); the
  graphs whose weights are borrowed (`dit_skip22`'s gates, Wan's tile-32 pair), which the runtime does not bind to an
  external rule.
