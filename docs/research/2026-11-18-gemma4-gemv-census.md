> This document is a point-in-time snapshot (2026-11-18) of which GEMV kernel every integer-storage linear of a Gemma 4
> decode token and of an MTP drafter run takes, what the GEMV sweep measured for the shapes that had no case, and which
> parallel rows that measurement supports, on the maintainer's RTX 3080 Ti under Node + Dawn. Chrome and the RTX 5070 Ti
> were not run. It is not a design source of truth. The scripts and the raw records are local and untracked
> (`outputs/diag/decode-optimization-2026-11-18/gemv-rows/`, `outputs/workbench/node-nvidia-geforce-rtx-3080-ti/`).

# Gemma 4: the GEMV shapes of a decode token and of a drafter run

## 1. Summary

- At the level `reorder` a decode token of every Gemma 4 model runs its integer linears on parallel GEMV rows, except
  the output heads of the E2B / E4B QAT lines (int4 on `karume-gemma4-qat`, int2 on `karume-gemma4-qat-mobile`). An MTP
  drafter run had 66 of its 68 linears on the sequential GEMV on E2B / E4B and all 68 on 12B (§2).
- The GEMV sweep had no case for 24 of these shapes. They are cases now (§3).
- Measured on the RTX 3080 Ti and derived with the generator's rule, five drafter shapes take a parallel row: two of
  the E2B / E4B drafters (`256 × 3072`, `256 × 1024`, 32 lanes) and three of the 12B drafter (`1024 × 7680`,
  `1024 × 4096`, `1024 × 8192`, 32 lanes with staged activations) (§4).
- With the rows a 12B drafter run takes 4.20 ms of GPU time in place of 5.85 ms (−28 %) and an E2B drafter run 2.19 ms in
  place of 2.38 ms (−8 %); the verify runs and the tokens are unchanged (§7).
- The int4 output heads get no parallel row: the parallel GEMV is ×3.6–4.4 faster at one row and slower than the
  sequential row blocks at four and eight, where verify runs it. The int2 heads gain nothing at one row. A sequential
  variant that writes the same words (`c16u2`) is ×1.7–1.9 faster on the int4 heads and is not applied here (§5).
- Two full records of the existing 66 rows show the same pattern at four and eight rows: 36 rows are slower than the
  sequential row blocks at M = 4 and 47 at M = 8, which makes a verify run of 12B about 20 ms slower in its linears at
  `reorder` than at `exact` (§6).

## 2. The census

Every `linear` of the `model` and `drafter` graphs of the three mirrors (`models/karume-gemma4`, `karume-gemma4-qat`,
`karume-gemma4-qat-mobile`), read from part 0 of each container and planned at one row, with the storage its weight is
resident in. "At `reorder`" is what the default profile selects for a Session with linear at the level `reorder`
(`linearGemvParallelEligible`, the recipe builder's predicate); at `exact` every one of these linears runs the
sequential GEMV. The time is the GEMV sweep's per-dispatch time at M = 1 of that choice on synthetic tensors of the
shape (the smallest over the records of 2026-11-18, RTX 3080 Ti), and the last column is that time × the count: kernel
time only, without the cost of a dispatch.

Which presets reach `reorder`:

| Mirror | Models | Presets at `reorder` | MTP drafter |
| --- | --- | --- | --- |
| `karume-gemma4` | E2B, E4B | none (`bf16` only: f32 compute in the reference order) | E2B only, int8 |
| `karume-gemma4-qat` | E2B, E4B, 12B, 26B | `i4-fast` | E2B, E4B, 12B (int4 g32) |
| `karume-gemma4-qat-mobile` | E2B, E4B | `i4-gemvpar`, `i4-fast` (the default; packed int8 activations) | none |

The regular models' own linears are bf16 (277 and 344 per token, all on the sequential GEMV) and are not in this
census; neither is the f32 `per_layer_model_projection` of the mobile models (one per token, sequential GEMV). On
the mobile lines' `i4-fast` the rows marked for packed activations run their packed form and the linears followed by a
quantize step its fused form; the table names the row's lanes and times its plain form.

**Regular E2B drafter (`karume-gemma4`, int8; its model is bf16)** — 68 linears per drafter run

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per run |
| --- | --- | --- | ---: | --- | ---: | ---: |
| pre_projection | i8 | 256 × 3072 | 3 | sequential c32u4 | 27.1 | 0.081 |
| q | i8 | 1024 × 256 | 9 | sequential c32u4 | 3.4 | 0.030 |
| o | i8 | 256 × 1024 | 9 | sequential c32u4 | 11.4 | 0.102 |
| gate ×12, up ×12, q ×3 | i8 | 2048 × 256 | 27 | sequential c32u4 | 3.7 | 0.100 |
| down ×12, o ×3 | i8 | 256 × 2048 | 15 | sequential c32u4 | 17.9 | 0.268 |
| post_projection | i8 | 1536 × 256 | 2 | parallel-l4 | 2.8 | 0.006 |
| lm_head | i8 | 262144 × 256 | 3 | sequential c32u4 | 85.8 | 0.257 |

Kernel time of these 68 linears as they ran before this change: 0.85 ms per run, of it 0.84 ms on the sequential GEMV.

**q4_0 QAT E2B (`karume-gemma4-qat`)** — 277 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| per_layer_model_projection | i4 g32 | 8960 × 1536 | 1 | parallel-l4 | 17.9 | 0.018 |
| q | i4 g32 | 2048 × 1536 | 28 | parallel-l4 | 8.1 | 0.227 |
| k ×12, v ×12, per_layer_input_gate ×35 | i4 g32 | 256 × 1536 | 59 | parallel-l32 | 3.5 | 0.204 |
| o | i4 g32 | 1536 × 2048 | 28 | staged-l32 | 5.7 | 0.159 |
| gate ×15, up ×15 | i4 g32 | 6144 × 1536 | 30 | parallel-l4 | 13.9 | 0.418 |
| down | i4 g32 | 1536 × 6144 | 15 | staged-l32 | 12.1 | 0.181 |
| per_layer_projection | i4 g32 | 1536 × 256 | 35 | parallel-l4 | 2.7 | 0.095 |
| q | i4 g32 | 4096 × 1536 | 7 | parallel-l4 | 10.2 | 0.072 |
| k ×3, v ×3 | i4 g32 | 512 × 1536 | 6 | parallel-l32 | 4.0 | 0.024 |
| o | i4 g32 | 1536 × 4096 | 7 | staged-l32 | 8.6 | 0.060 |
| gate ×20, up ×20 | i4 g32 | 12288 × 1536 | 40 | parallel-l4 | 24.8 | 0.992 |
| down | i4 g32 | 1536 × 12288 | 20 | staged-l32 | 25.8 | 0.516 |
| lm_head | i4 g32 | 262144 × 1536 | 1 | sequential c32u4 | 1171 | 1.171 |

Kernel time of these 277 linears as they ran before this change: 4.14 ms per token, of it 1.17 ms on the sequential GEMV.

**q4_0 QAT E2B drafter** — 68 linears per drafter run

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per run |
| --- | --- | --- | ---: | --- | ---: | ---: |
| pre_projection | i4 g32 | 256 × 3072 | 3 | sequential c32u4 → **parallel-l32** | 25.2 | 0.076 |
| q | i4 g32 | 1024 × 256 | 9 | sequential c32u4 | 3.2 | 0.029 |
| o | i4 g32 | 256 × 1024 | 9 | sequential c32u4 → **parallel-l32** | 9.3 | 0.084 |
| gate ×12, up ×12, q ×3 | i4 g32 | 2048 × 256 | 27 | sequential c32u4 | 3.4 | 0.093 |
| down ×12, o ×3 | i4 g32 | 256 × 2048 | 15 | sequential c32u4 | 17.3 | 0.260 |
| post_projection | i4 g32 | 1536 × 256 | 2 | parallel-l4 | 2.7 | 0.005 |
| lm_head | i4 g32 | 262144 × 256 | 3 | sequential c32u4 | 61.8 | 0.185 |

Kernel time of these 68 linears as they ran before this change: 0.73 ms per run, of it 0.73 ms on the sequential GEMV.

**q4_0 QAT E4B** — 344 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| per_layer_model_projection | i4 g32 | 10752 × 2560 | 1 | parallel-l4 | 39.7 | 0.040 |
| q | i4 g32 | 2048 × 2560 | 35 | parallel-l4 | 12.7 | 0.444 |
| k ×20, v ×20 | i4 g32 | 512 × 2560 | 40 | parallel-l32 | 5.2 | 0.207 |
| o | i4 g32 | 2560 × 2048 | 35 | staged-l32 | 7.6 | 0.267 |
| gate ×42, up ×42 | i4 g32 | 10240 × 2560 | 84 | parallel-l4 | 35.9 | 3.012 |
| down | i4 g32 | 2560 × 10240 | 42 | staged-l32 | 31.5 | 1.324 |
| per_layer_input_gate | i4 g32 | 256 × 2560 | 42 | parallel-l32 | 4.2 | 0.178 |
| per_layer_projection | i4 g32 | 2560 × 256 | 42 | parallel-l4 | 2.9 | 0.121 |
| q | i4 g32 | 4096 × 2560 | 7 | parallel-l4 | 19.5 | 0.136 |
| k ×4, v ×4 | i4 g32 | 1024 × 2560 | 8 | parallel-l32 | 7.8 | 0.062 |
| o | i4 g32 | 2560 × 4096 | 7 | staged-l32 | 13.3 | 0.093 |
| lm_head | i4 g32 | 262144 × 2560 | 1 | sequential c32u4 | 2262 | 2.262 |

Kernel time of these 344 linears as they ran before this change: 8.15 ms per token, of it 2.26 ms on the sequential GEMV.

**q4_0 QAT E4B drafter** — 68 linears per drafter run

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per run |
| --- | --- | --- | ---: | --- | ---: | ---: |
| pre_projection | i4 g32 | 256 × 5120 | 3 | sequential c32u4 | 40.9 | 0.123 |
| q | i4 g32 | 1024 × 256 | 9 | sequential c32u4 | 3.2 | 0.029 |
| o | i4 g32 | 256 × 1024 | 9 | sequential c32u4 → **parallel-l32** | 9.3 | 0.084 |
| gate ×12, up ×12, q ×3 | i4 g32 | 2048 × 256 | 27 | sequential c32u4 | 3.4 | 0.093 |
| down ×12, o ×3 | i4 g32 | 256 × 2048 | 15 | sequential c32u4 | 17.3 | 0.260 |
| post_projection | i4 g32 | 2560 × 256 | 2 | parallel-l4 | 2.9 | 0.006 |
| lm_head | i4 g32 | 262144 × 256 | 3 | sequential c32u4 | 61.8 | 0.185 |

Kernel time of these 68 linears as they ran before this change: 0.78 ms per run, of it 0.77 ms on the sequential GEMV.

**q4_0 QAT 12B** — 329 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| q | i4 g32 | 4096 × 3840 | 40 | parallel-l4 | 33.4 | 1.335 |
| k ×40, v ×40 | i4 g32 | 2048 × 3840 | 80 | parallel-l4 | 17.6 | 1.410 |
| o | i4 g32 | 3840 × 4096 | 40 | staged-l32 | 19.0 | 0.762 |
| gate ×48, up ×48 | i4 g32 | 15360 × 3840 | 96 | parallel-l4 | 79.6 | 7.639 |
| down | i4 g32 | 3840 × 15360 | 48 | staged-l32 | 61.5 | 2.951 |
| q | i4 g32 | 8192 × 3840 | 8 | parallel-l4 | 50.2 | 0.402 |
| k | i4 g32 | 512 × 3840 | 8 | parallel-l32 | 6.5 | 0.052 |
| o | i4 g32 | 3840 × 8192 | 8 | staged-l32 | 38.1 | 0.305 |
| lm_head | i4 g32 | 262144 × 3840 | 1 | parallel-l16 | 1946 | 1.946 |

Kernel time of these 329 linears as they ran before this change: 16.80 ms per token, of it 0.00 ms on the sequential GEMV.

**q4_0 QAT 12B drafter** — 68 linears per drafter run

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per run |
| --- | --- | --- | ---: | --- | ---: | ---: |
| pre_projection | i4 g32 | 1024 × 7680 | 3 | sequential c32u4 → **staged-l32** | 62.2 | 0.187 |
| q | i4 g32 | 4096 × 1024 | 9 | sequential c32u4 | 10.4 | 0.094 |
| o | i4 g32 | 1024 × 4096 | 9 | sequential c32u4 → **staged-l32** | 34.9 | 0.314 |
| gate ×12, up ×12, q ×3 | i4 g32 | 8192 × 1024 | 27 | sequential c32u4 | 13.9 | 0.376 |
| down ×12, o ×3 | i4 g32 | 1024 × 8192 | 15 | sequential c32u4 → **staged-l32** | 66.7 | 1.001 |
| post_projection | i4 g32 | 3840 × 1024 | 2 | sequential c32u4 | 10.9 | 0.022 |
| lm_head | i4 g32 | 262144 × 1024 | 3 | sequential c32u4 | 428 | 1.284 |

Kernel time of these 68 linears as they ran before this change: 3.28 ms per run, of it 3.28 ms on the sequential GEMV.

**q4_0 QAT 26B A4B (dense linears; the 60 `expert_linear` dispatches are not GEMVs)** — 236 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| q | i4 g32 | 4096 × 2816 | 25 | parallel-l4 | 23.9 | 0.597 |
| k ×25, v ×25 | i4 g32 | 2048 × 2816 | 50 | parallel-l4 | 13.5 | 0.676 |
| o | i4 g32 | 2816 × 4096 | 25 | staged-l32 | 14.6 | 0.365 |
| gate ×30, up ×30 | i4 g32 | 2112 × 2816 | 60 | parallel-l4 | 13.5 | 0.811 |
| down | i4 g32 | 2816 × 2112 | 30 | staged-l32 | 10.6 | 0.319 |
| router | i4 g32 | 128 × 2816 | 30 | parallel-l32 | 3.9 | 0.117 |
| q | i4 g32 | 8192 × 2816 | 5 | parallel-l4 | 37.4 | 0.187 |
| k | i4 g32 | 1024 × 2816 | 5 | parallel-l32 | 8.7 | 0.043 |
| o | i4 g32 | 2816 × 8192 | 5 | staged-l32 | 27.3 | 0.136 |
| lm_head | i4 g32 | 262144 × 2816 | 1 | parallel-l4 | 1012 | 1.012 |

Kernel time of these 236 linears as they ran before this change: 4.26 ms per token, of it 0.00 ms on the sequential GEMV.

**Mobile QAT E2B (`karume-gemma4-qat-mobile`)** — 277 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| q | i4 g512 | 2048 × 1536 | 28 | parallel-l4 | 8.6 | 0.240 |
| k ×12, v ×12 | i4 g512 | 256 × 1536 | 24 | parallel-l32 | 3.4 | 0.083 |
| o | i4 g2048 | 1536 × 2048 | 28 | parallel-l32 | 8.1 | 0.226 |
| gate ×15, up ×15 | i4 g512 | 6144 × 1536 | 30 | parallel-l4 | 13.5 | 0.404 |
| down | i4 g2048 | 1536 × 6144 | 15 | parallel-l32 | 20.6 | 0.308 |
| per_layer_input_gate | i8 | 256 × 1536 | 35 | parallel-l32 | 3.3 | 0.117 |
| per_layer_projection | i8 | 1536 × 256 | 35 | parallel-l4 | 2.8 | 0.098 |
| q | i4 g512 | 4096 × 1536 | 7 | parallel-l4 | 10.1 | 0.070 |
| k ×3, v ×3 | i4 g512 | 512 × 1536 | 6 | parallel-l32 | 4.0 | 0.024 |
| o | i4 g4096 | 1536 × 4096 | 7 | parallel-l32 | 14.0 | 0.098 |
| gate ×20, up ×20 | i2 | 12288 × 1536 | 40 | parallel-l2 | 17.5 | 0.699 |
| down | i2 | 1536 × 12288 | 20 | parallel-l32 | 36.4 | 0.729 |
| lm_head | i2 | 262144 × 1536 | 1 | sequential c32u4 | 293 | 0.293 |

Kernel time of these 276 linears as they ran before this change: 3.39 ms per token, of it 0.29 ms on the sequential GEMV.

**Mobile QAT E4B** — 344 linears per decode token

| Role | Storage | n × k | Count | At `reorder` | µs per dispatch | ms per token |
| --- | --- | --- | ---: | --- | ---: | ---: |
| q | i4 g512 | 2048 × 2560 | 35 | parallel-l4 | 12.5 | 0.439 |
| k ×20, v ×20 | i4 g512 | 512 × 2560 | 40 | parallel-l32 | 5.1 | 0.205 |
| o | i4 g2048 | 2560 × 2048 | 35 | parallel-l32 | 12.0 | 0.419 |
| gate ×42, up ×42 | i4 g512 | 10240 × 2560 | 84 | parallel-l4 | 33.2 | 2.790 |
| down | i4 g2048 | 2560 × 10240 | 42 | parallel-l32 | 55.2 | 2.320 |
| per_layer_input_gate | i8 | 256 × 2560 | 42 | parallel-l32 | 4.0 | 0.167 |
| per_layer_projection | i8 | 2560 × 256 | 42 | parallel-l4 | 3.0 | 0.126 |
| q | i4 g512 | 4096 × 2560 | 7 | parallel-l4 | 16.6 | 0.116 |
| k ×4, v ×4 | i4 g512 | 1024 × 2560 | 8 | parallel-l32 | 7.9 | 0.063 |
| o | i4 g4096 | 2560 × 4096 | 7 | parallel-l32 | 22.9 | 0.160 |
| lm_head | i2 | 262144 × 2560 | 1 | sequential c32u4 | 450 | 0.450 |

Kernel time of these 343 linears as they ran before this change: 7.26 ms per token, of it 0.45 ms on the sequential GEMV.


What the census shows:

- Every shape of every model graph had a sweep case except the four output heads of the E2B / E4B QAT lines: int4 g32
  `262144 × 1536` and `× 2560` (q4_0) and int2 `262144 × 1536` and `× 2560` (mobile). The 22 E4B rows that copy E2B's
  lanes all have a case, so the sweep measures them as it is.
- No drafter shape had a case, except `post_projection` of the E2B / E4B drafters, which is the model's
  `per_layer_projection` shape. The E2B and E4B drafters run 66 of their 68 linears on the sequential GEMV at `reorder`
  (what the census table pinned), and the 12B drafter all 68: its inner width is 1024, so not even `post_projection`
  matches a row.
- The q4_0 output head is the largest single kernel of an E2B / E4B decode token at `reorder`: 1.17 ms of the 4.14 ms of
  linear kernel time on E2B and 2.26 of 8.15 ms on E4B. The decode bench's per-dispatch tap agrees (1.21 ms of 9.91 ms of
  GPU time per token on E2B and 2.19 of 14.75 ms on E4B, `i4-fast`, one timed pass per dispatch).
- The int8 PLE linears with the fused quantize step (perf-ledger K-98) are cases already as plain GEMVs: `256 × 1536` at
  32 lanes takes 3.3 µs and `1536 × 256` at 4 lanes 2.8 µs, the same as their int4 g32 counterparts (3.5 and 2.7 µs).
  The fused key itself is not a sweep candidate.

## 3. The cases added

24 shapes, all measured at M = 1, 4, 8 (and 16, 64 in the `row-blocks` set) like every other case:

| Shapes | Storage | Where |
| --- | --- | --- |
| `262144 × 1536`, `262144 × 2560` | int4 g32 | the q4_0 QAT E2B / E4B output heads |
| `262144 × 1536`, `262144 × 2560` | int2 | the mobile QAT E2B / E4B output heads |
| `256 × 3072`, `1024 × 256`, `256 × 1024`, `2048 × 256`, `256 × 2048`, `262144 × 256` | int4 g32 | the q4_0 QAT E2B drafter; the last five also the E4B drafter |
| `256 × 5120` | int4 g32 | the q4_0 QAT E4B drafter (`pre_projection`) |
| `1024 × 7680`, `4096 × 1024`, `1024 × 4096`, `8192 × 1024`, `1024 × 8192`, `3840 × 1024`, `262144 × 1024` | int4 g32 | the q4_0 QAT 12B drafter |
| `256 × 3072`, `1024 × 256`, `256 × 1024`, `2048 × 256`, `256 × 2048`, `262144 × 256` | int8 | the regular E2B drafter |

The case table grows from 66 to 90 cases; the `full` set from 5,044 to 6,827 kernels with the subgroup features (4,066 to
5,510 without), the `quick` set from 1,569 to 2,124. No group of an existing record goes stale (the fingerprints are
per case): the dry run of the generator on the RTX 3080 Ti record of 2026-11-18T02-55 still reads its 198 groups as
current. `packages/e2e/tests/e2e_gemma4_gemv_cases_test.ts` fails when a mirror's graph holds an integer linear without
a case.

## 4. The sweep of the new cases and the derivation

Eight records of `gemv-sweep` (set `full`, RTX 3080 Ti, Node + Dawn, 2026-11-18): three cover the 24 new cases once,
one covers eight of them a second time, and four narrowed ones (the parallel and staged candidates only) repeat the
seven shapes that were short of evidence. The decisions are the generator's dry run on the eight records
(`node tools/geometry-profiles/generate.ts nvidia-ampere --gemv-records=… --dry-run`, before the rows were added): a
candidate is adopted when it is at least ×1.10 faster than what runs today at M = 1 in two steady records, with its
band check held, and not slower at M = 4 and 8 in two steady records. Gains are per record, at M = 1, against the
sequential GEMV.

| Shape (int4 g32) | Where, per run | Sequential (µs) | Decision | Why |
| --- | --- | ---: | --- | --- |
| `256 × 3072` | E2B drafter `pre_projection` ×3 | 25.2 | **parallel, 32 lanes** | ×5.90, ×5.91, ×5.88 |
| `256 × 1024` | E2B / E4B drafter o (sliding) ×9 | 9.3 | **parallel, 32 lanes** | ×3.30, ×3.30, ×3.34 |
| `1024 × 7680` | 12B drafter `pre_projection` ×3 | 62.2 | **staged, 32 lanes** | ×5.27, ×5.30, ×5.08 |
| `1024 × 4096` | 12B drafter o (sliding) ×9 | 34.9 | **staged, 32 lanes** | ×4.81, ×10.34 |
| `1024 × 8192` | 12B drafter down ×12, o (full) ×3 | 66.7 | **staged, 32 lanes** | ×5.66, ×5.17 |
| `256 × 2048` | E2B / E4B drafter down ×12, o (full) ×3 | 17.3 | no row | parallel 32 is ×4.85, ×4.88 at M = 1, but the baseline at M = 8 was steady in one record of six |
| `256 × 5120` | E4B drafter `pre_projection` ×3 | 40.9 | no row | parallel 32 is ×7.21, ×7.14, ×7.20, ×7.32 at M = 1, but the baseline at M = 4 was steady in one record of five |
| `1024 × 256` | E2B / E4B drafter q (sliding) ×9 | 3.2 | no row | parallel 4 is ×1.24 in the one steady record |
| `2048 × 256` | E2B / E4B drafter gate / up ×24, q (full) ×3 | 3.4 | no row | parallel 4 is ×1.23 at M = 1 and ×0.85 at M = 4 |
| `262144 × 256` | E2B / E4B drafter `lm_head` ×3 | 61.8 | no row | the best candidate is ×0.94, ×0.95 |
| `4096 × 1024`, `8192 × 1024`, `3840 × 1024` | 12B drafter q, gate / up, `post_projection` | 10.4, 13.9, 10.9 | no row | one record: ×1.5–1.9 at M = 1, and its fastest candidate at M = 4 or 8 is ×0.75–0.99; not measured again |
| `262144 × 1024` | 12B drafter `lm_head` ×3 | 428 | no row | one record; ×1.89 at M = 1, screened out at M = 4 and 8 |
| `262144 × 1536`, `262144 × 2560` | q4_0 E2B / E4B output heads | 1171, 2262 | no row | §5 |

The int2 heads and the six int8 shapes of the regular E2B drafter have one record each. The int2 heads cannot be
adopted on it (×1.02 and ×1.01 in steady groups). The int8 drafter shapes read like their int4 twins (parallel 32 at
×6.0 on `256 × 3072` and ×3.8 on `256 × 1024`), and no preset of `karume-gemma4` reaches `reorder`, so they were not
measured again.

The two shapes with a large gain and no row (`256 × 2048`, `256 × 5120`) are the rest of the E2B / E4B drafters' gain:
0.26 of the 0.73 ms of an E2B drafter run's linear kernel time is `256 × 2048`. Their sequential row-block baseline
spread between ×1.3 and ×2.1 over the five rounds in nearly every record, at one row count each, so the rule had no
second steady record to read. More records, or a look at why that baseline is unsteady, would settle them.

Once the rows are in the default profile, the groups of those five cases in these eight records are no longer current
(the default's choice at `reorder` moved): 54 groups. Every other group stays current.

## 5. The output heads

The two int4 heads were measured in two records each (µs per dispatch; the two records):

| Kernel | E2B head `262144 × 1536` | E4B head `262144 × 2560` |
| --- | --- | --- |
| M = 1, sequential `c32u4` (what runs today at both levels) | 1171 / 1173 | 2267 / 2262 |
| M = 1, sequential `c16u2` (same words as `c32u4`) | 636 / 627 | 1335 / 1320 |
| M = 1, parallel, staged activations, 8 lanes | 326 / 315 | 514 / 552 |
| M = 4, sequential row blocks `c32u4r4` (today) | 777 / 683 | 1687 / 2720 |
| M = 4, parallel, staged activations, 8 lanes (quick look only) | 1566 / 1360 | 2530 / 2517 |
| M = 8, sequential row blocks `c32u4r8` (today) | 1123 / 1383 | 2432 / 6482 |
| M = 8, parallel, staged activations, 8 lanes (quick look only) | 2948 / 2755 | 4416 / 5001 |

- **No parallel row.** At one row the parallel GEMV is ×3.6–3.7 (E2B) and ×4.1–4.4 (E4B) faster than what runs today.
  At four and eight rows its quick time is ×1.5–2.6 that of the row blocks in the steady groups, so the sweep screens it out and the
  derivation keeps "no row": a parallel row serves M = 1 to 8, and the head runs at four rows in every verify run of
  speculative decoding (`last_row` has k + 1 entries). A row for M = 1 alone is not a form the table has, and ADR
  [0098](../decisions/0098-linear-gemv-parallel.md) decision 2 rules it out: decode and verify must add in the same
  order.
- **The sequential variant is the measured gain that changes no number.** `c16u2` writes the words `c32u4` writes and is
  ×1.84–1.87 (E2B) and ×1.70–1.71 (E4B) faster in both records; the derivation adopts it as a per-shape exception of
  the sequential variant for both heads (the regular E2B's int8 head has had such an exception since ADR 0082 addendum
  10). It would take 0.54 ms off an E2B token and 0.93 ms off an E4B token at `exact` and at `reorder` alike, the
  default preset `i4` included. It is not applied here: it changes what a Session at `exact` runs, which this work
  was not asked to change, and it needs its own bit-identity GPU check, as the int8 exception has
  (`gpu_linear_gemv_variant_test.ts`).
- The mobile lines' int2 heads gain nothing from the parallel GEMV at one row (×1.02 and ×1.01 for the best candidate,
  one record, steady) and nothing from another sequential variant (×1.04, ×0.93). Their time is 0.29 and 0.45 ms.
- The drafters' heads (`262144 × 256` three times per run, `262144 × 1024` on 12B) do not gain either at k = 256
  (×0.94 / ×0.95, two records); 12B's gains ×1.89 at one row in one record and is screened out at four and eight rows
  like the model heads.

## 6. The existing rows at four and eight rows

The rule that keeps the heads without a row — not slower than what runs today at M = 4 and 8 — is one the existing rows
were never held to: they were chosen on decode (M = 1). The two full records of the RTX 3080 Ti that measure every
existing row (2026-11-17T23-50 and 2026-11-18T02-55) show, for the 66 shapes with a parallel row:

- at M = 1 every row is faster than the sequential GEMV in both records (×1.12 to ×7.2);
- at M = 4, 36 rows are slower than the sequential row blocks in both records, and at M = 8, 47 are.

Summed over the integer linears of a model graph (count × per-dispatch time, the smallest over the records):

| Graph | M | Linears on the sequential form (ms) | Linears as chosen at `reorder` (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| gemma4-qat e2b | 1 | 10.21 | 4.14 | ×2.47 |
| gemma4-qat e2b | 4 | 11.30 | 9.80 | ×1.15 |
| gemma4-qat e2b | 8 | 14.69 | 18.37 | ×0.80 |
| gemma4-qat e4b | 1 | 18.15 | 8.12 | ×2.23 |
| gemma4-qat e4b | 4 | 19.90 | 20.95 | ×0.95 |
| gemma4-qat e4b | 8 | 25.74 | 41.76 | ×0.62 |
| gemma4-qat 12b | 1 | 39.36 | 16.69 | ×2.36 |
| gemma4-qat 12b | 4 | 42.95 | 62.68 | ×0.69 |
| gemma4-qat 12b | 8 | 67.80 | 136.15 | ×0.50 |
| gemma4-qat 26b | 1 | 10.46 | 4.26 | ×2.46 |
| gemma4-qat 26b | 4 | 10.98 | 12.32 | ×0.89 |
| gemma4-qat 26b | 8 | 15.72 | 24.15 | ×0.65 |
| gemma4-qat-mobile e2b | 1 | 6.35 | 3.38 | ×1.88 |
| gemma4-qat-mobile e2b | 4 | 9.00 | 10.49 | ×0.86 |
| gemma4-qat-mobile e2b | 8 | 13.70 | 21.29 | ×0.64 |
| gemma4-qat-mobile e4b | 1 | 15.92 | 7.20 | ×2.21 |
| gemma4-qat-mobile e4b | 4 | 18.54 | 25.10 | ×0.74 |
| gemma4-qat-mobile e4b | 8 | 23.93 | 51.46 | ×0.46 |

So a verify run of speculative decoding (M = k + 1 = 4 with the shipped k = 3) spends more time in its linears at
`reorder` than at `exact` on every model but the q4_0 E2B, and 19.7 ms more on 12B. That is a candidate explanation of
perf-ledger K-102 (`i4-fast` slower than `i4` with the drafter on 12B: 38.2 against 32.6 and 25.9 against 19.1 ms per
token), which was filed without one. It was not checked end to end here. The mobile lines' packed rows are measured in
their unpacked form (the sweep does not run the packed kernels), so their lines are an upper bound of the loss.

## 7. The draft step end to end

Per-kernel GPU time of every run of speculative chats (`speculative: "always"`, k = 3, greedy, one prompt), read from
`lastRunTiming` on devices acquired with `gpuTiming`. The two arms differ only in the five rows; a round visits them A B
or B A, and every visit acquires its device, builds its pipeline and runs one cold chat (left out) and one measured
chat. Median (minimum) of the GPU time of one run:

| Model, `i4-fast` | Runs per arm | Draft run, before | Draft run, after | Of it linears | Verify run, before / after |
| --- | ---: | ---: | ---: | --- | --- |
| q4_0 QAT E2B, 3 rounds, 64 tokens | 108 draft, 114 verify | 2.376 (2.260) ms | 2.188 (2.087) ms | 1.266 → 1.080 ms | 15.40 / 15.44 ms |
| q4_0 QAT 12B, 2 rounds, 48 tokens | 42 draft, 44 verify | 5.850 (5.604) ms | 4.198 (3.861) ms | 4.419 → 2.771 ms | 62.99 / 63.18 ms |

- A 12B drafter run dispatches 41 linears on the sequential GEMV and 27 on the staged parallel GEMV in place of 68 on the
  sequential one; an E2B drafter run 54 and 14 (12 at 32 lanes, the two `post_projection` at 4) in place of 66 and 2, and
  an E4B one 57 and 11 (a second, one-round run of each for the counts). The tokens of the two arms were identical in
  every chat of the three models.
- The E4B drafter's one row (`256 × 1024`, nine linears, about 0.06 ms of kernel time per run) is too small to read end
  to end at the 14 runs taken.
- Per decoded token the gain is the draft run's gain times the draft runs per token: 0.77 ms on 12B here (22 draft runs
  for 47 tokens) and 0.11 ms on E2B (37 for 63). Against the speculative decode times on record for 12B `i4-fast` (25.9
  and 38.2 ms per token, [research 2026-11-06](2026-11-06-gemma4-12b-fast.md) §3) that is 2–3 %; on E2B it is about 1 %.
  A wall-clock decode time was not measured: with one timed pass per dispatch the wall is not the product's.
- The verify run of 12B spends 55 ms in its linears at `reorder`, which is §6's estimate from the sweep seen in a real
  pipeline.

## 8. Method notes

- **Sweep.** `gemv-sweep`, set `full`, defaults (5 rounds, 10 s idle between cases, quick-look margin ×1.25), narrowed
  with `cases` to the new shapes, eight per run. Every timed candidate of the reorder class was inside the band of
  `linear` (largest use 9 % of the band) and every timed candidate of the identical class wrote the baseline's words;
  all runs concluded `pass`.
- **Steadiness.** These kernels are short (3 to 90 µs), and 6 to 10 of the 24 groups of a run had a baseline spread
  above the derivation's bound of 1.25, so two records were often not two pieces of evidence at all three row counts.
- **Per-dispatch baselines.** `gemma4-bench` with the `split` switch (one timed pass per dispatch), 64 new tokens, one
  warm chat. A timed pass adds about 5 µs per dispatch to a small kernel (the 277 linears of an E2B token: 5.57 ms in
  the tap against 4.14 ms of kernel time), so the census uses the sweep's times and the tap's only as a cross-check of
  the large kernels (the heads agree within 4 % on the int4 lines).
- **End to end.** A wall-clock A/B of two pipelines kept resident on two devices of one adapter is not usable at this
  effect size: with identical GEMV choices in both arms it read ×1.13 (median of 12 chats per arm) in one order of
  creation and ×1.00 in the other. The numbers of that design are not reported; §7 reads per-kernel GPU time over
  pipelines built per visit instead.
- **Steady shapes at four and eight rows.** The narrowed runs took no idle time between cases (`idleSeconds` 0), which
  did not make the baselines steadier.

## 9. Not measured

- Chrome, the RTX 5070 Ti and the Apple M2: nothing. The new cases are missing from every other device's records.
- A wall-clock decode time with the rows (the 12B gain per token is computed from per-run GPU time), the E4B drafter
  end to end, and the output heads' sequential variant end to end.
- The 26B model on this 12 GB GPU; it ships no drafter and every dense linear of it has a row.
- The packed-activation rows of the mobile lines (the sweep does not run the packed kernels).

