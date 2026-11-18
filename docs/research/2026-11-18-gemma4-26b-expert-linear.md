# Gemma 4 26B A4B: what an `expert_linear` dispatch costs, and the grouped form (2026-11-18)

A point-in-time measurement record. It takes the routed experts of a 26B forward apart (perf-ledger K-103, K-104),
gives the measurements behind the grouped form of `expert_linear`, and records what was measured of the one-row kernel
and of a parallel prototype. Scripts, logs and records: `outputs/diag/decode-optimization-2026-11-18/expert-linear/`.

## 1. Summary

- A run of several tokens now walks an expert's weight row once for the rows that picked it (the grouped form, for
  gate_up's long rows). Decode under the expert cache with self-speculation at depth 4 takes ×0.85 of the time per
  token on the RTX 3080 Ti with the same tokens; a clean verify of 5 rows goes from 64.5 to 48.1 ms.
- A one-token run is unchanged by it. Of its 2.9 ms of routed experts per forward, 2.3 ms is gate_up
  (77 µs per dispatch) and 0.6–0.7 ms is down (21–23 µs).
- The 48 µs per dispatch that K-103 compared with 17–30 µs for dense rows is the time of eight weight rows: one
  `expert_linear` dispatch reads the 8 picked experts. Per multiply-add the kernel is not slower than the dense
  parallel GEMV.

## 2. Setup

The RTX 3080 Ti (12 GB) under Node and Dawn, branch `wip/opt-expert-linear` from main `7834560b`. 26B runs under the
exact expert cache (64 slots per layer unless stated, every expert in the host tier), preset `i4-fast` unless stated,
greedy, the three prompts of [research 2026-11-10](2026-11-10-gemma4-26b-fast.md) §2 with 96 new tokens.

The card was shared with four other measurement legs and ran at 85–93 °C with its thermal slowdown flag set in almost
every run here, so absolute times are slower than those of the earlier records (a decode token at depth 4 took 50–60 ms
against 38–42 ms on 2026-11-10). Every comparison below is inside one process with the arms interleaved.

- **Kernel bench** (`kernel-bench.ts`): one dispatch of the pooled kernel (i4 codes, f16 steps, 64 slots of 128
  experts) at the two shapes of 26B — gate_up (n 1408, k 2816, x shared by the picks) and down (n 2816, k 704, one x row
  per pick) — with 8 picks per token. Up to 256 dispatches in one timed pass, the variants interleaved round by round
  (15 rounds; median and minimum), every variant's output compared word for word with the first. Synthetic routing: a
  token keeps each pick of the token before it with probability `overlap`.
- **Decode A/B** (`decode-ab.ts`): one pipeline, two arms on one Session, switched at recipe time. Arm `base` compiles
  `expert_linear` from the kernel module of the branch point; arm `grouped` is the tree. Per prompt one untimed
  generation, then A B B A (B A A B on odd rounds); the token ids of every generation are compared.

## 3. The forward at the branch point

Per-kernel GPU time (timestamp queries, one pass per dispatch), `i4-fast`, medians over the runs of three prompts
(`logs/base-timing-fast-64.log`):

| Run | Total (ms) | `expert_linear` | Dense GEMV | Attention | Other |
| --- | ---------: | --------------: | ---------: | --------: | ----: |
| decode, 1 row (285 runs) | 15.77 | 2.89 (60 × 48.1 µs) | 5.70 (236 × 24.2 µs) | 0.91 | 6.23 (938 × 6.6 µs) |
| prefill, 8 rows (12 runs) | 45.65 | 21.19 (60 × 353 µs) | 15.73 (236 × 66.7 µs) | 1.28 | 7.62 |

The decode numbers are those of 2026-11-10 (§5 there). In a run of 8 rows the routed experts are 46% of the GPU time.

## 4. One dispatch, one token

Kernel bench, 8 pick rows (`logs/kb-batch1.log`). µs per dispatch, median / minimum.

| Variant | gate_up | down | Output |
| --- | ---: | ---: | --- |
| the kernel (32 columns per workgroup, 4 words prefetched) | 77.3 / 75.8 | 23.5 / 21.7 | — |
| 16 columns | 77.5 / 76.3 | 34.5 / 30.6 | identical |
| 64 columns | 83.9 / 82.4 | 23.1 / 20.7 | identical |
| 128 columns | 94.4 / 92.7 | 23.8 / 21.7 | identical |
| 2 words prefetched | 75.8 / 74.4 | 23.8 / 21.7 | identical |
| 8 words prefetched | 67.4 / 66.7 | 23.3 / 20.5 | identical |
| 16 words prefetched | 67.5 / 66.8 | 23.3 / 20.7 | identical |
| every read, one multiply-add per word (not a kernel) | 35.6 / 35.4 | 14.1 / 13.7 | — |
| parallel prototype, 2 lanes per output | 65.3 / 65.1 | 20.8 / 18.0 | differs (reorder) |
| parallel prototype, 4 lanes | 41.4 / 39.6 | 25.9 / 21.9 | differs (reorder) |
| parallel prototype, 8 lanes | 66.0 / 62.8 | 50.1 / 50.0 | differs (reorder) |

- gate_up and down together are 98–101 µs per layer, 2.9–3.0 ms per forward: the bench reproduces the forward's 2.89 ms.
- A gate_up dispatch is 8 × 1408 outputs of 2816 elements, 31.7 M multiply-adds over 15.9 MB of weights; a down dispatch
  15.9 M over 7.9 MB. The dense rows K-103 compared them with are 6–8 M multiply-adds. Per multiply-add the routed
  experts run at about 410 M per ms (gate_up) and 680–770 M per ms (down), the dense parallel GEMV of the head at about
  770 M per ms (738 M in 0.96 ms on 2026-11-10).
- The columns per workgroup do not help. Prefetching 8 words in place of 4 takes 13% off gate_up and nothing off down.
- The parallel prototype (the form of `linear`'s parallel GEMV: lanes stride the words, `fma`, a tree of adds) takes 46%
  off gate_up at 4 lanes and 11% off down at 2 lanes: 38.6 µs per layer, 1.16 ms per forward of per-kernel time. It
  changes the summation order.

## 5. Several tokens: the grouped form

In a run of T tokens the one-row kernel reads and decodes an expert's weight row once for every row that picked the
expert. The grouped form lets the thread of the first such row walk the weight row for the others too, each row
keeping its own sum in ascending k (the same words out, compared as u32).

Kernel bench, µs per dispatch, median / minimum (`logs/kb-batch1.log`; "distinct" = experts per 8 T pick rows of the
synthetic routing; a form named "n rows" generated one loop per row count up to n, which the first build did):

| Shape | T | Overlap | Distinct | One-row | Grouped, 4 rows | Grouped, 8 rows |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| gate_up | 5 | 0.5 | 21–27 of 40 | 281.5 / 280.1 | 240.7 / 238.7 (×0.855) | 216.0 / 213.4 (×0.767) |
| gate_up | 8 | 0.5 | 27–33 of 64 | 499.4 / 494.4 | 345.8 / 342.5 (×0.693) | 285.0 / 283.1 (×0.571) |
| gate_up | 5 | 0 | 28–33 of 40 | 364.0 / 363.0 | 341.9 / 340.7 (×0.939) | — |
| down | 5 | 0.5 | 16–23 of 40 | 77.6 / 72.7 | 72.4 / 66.5 (×0.934) | 81.4 / 78.7 (×1.049) |
| down | 8 | 0.5 | 25–32 of 64 | 139.9 / 131.4 | 144.9 / 98.8 (×1.035) | 164.0 / 120.3 (×1.172) |
| down | 5 | 0 | 30–37 of 40 | 113.6 / 97.8 | 136.7 / 110.3 (×1.204) | — |

- gate_up's rows are 88 words long and gain in every case measured, more with 8 rows to a walk. down's rows are 22
  words: with half of the picks shared the gain is within the noise of the hot card, and with few shared the grouped
  form is slower, since every thread scans the run's rows to find its expert's. So the grouped form is taken for weight
  rows of 64 words or more (`expertLinearGrouped`), which of the two shapes is gate_up.
- Writing the decode of a word once by hand for the walk's rows, in place of calling the word function per row as
  `linear`'s row block does, measured the same on gate_up (240.5 against 240.7 µs at T = 5; 345.2 against 345.8 at
  T = 8): the compiler already shares the decode. The function form is kept.
- The form built scans 8 rows at a time and walks them 8, 4, 2 or 1 at a time (5 rows: a walk of 4 and a walk of 1):
  four loops in the WGSL in place of the eight a loop per count needs (the pooled kernel is 378 lines against 312 for
  the one-row form).

## 6. Decode and prefill, end to end

Decode A/B (§2), `i4-fast`, 64 slots, 2 rounds × 3 prompts × (A B B A) = 12 generations of 96 tokens per arm. Every
generation of a prompt gave the same token ids in both arms.

| Configuration | Measure | Base | Grouped | Ratio |
| --- | --- | ---: | ---: | --- |
| depth 4 (`logs/ab-fast-64-d4.log`) | decode ms per token, median / minimum | 59.46 / 49.70 | 50.01 / 47.51 | paired per prompt and round: median ×0.846 (0.833–0.889) |
| | clean verify of 5 rows, ms, median / minimum (15 / 16 runs) | 64.49 / 53.64 | 48.06 / 38.73 | ×0.745 |
| | clean prefill chunk of 8 rows, ms (48 runs) | 91.54 / 46.94 | 62.02 / 50.01 | ×0.678 |
| | clean run of 1 row, ms (527 / 524 runs) | 23.12 / 17.04 | 22.91 / 17.74 | ×0.991 |
| | time to the first token, ms | 1849.6 / 1118.9 | 1371.9 / 1051.5 | ×0.742 |
| no self-speculation (`logs/ab-fast-64-d0.log`) | decode ms per token | 48.55 / 44.54 | 49.87 / 43.97 | paired: median ×1.009 (0.906–1.082) |
| | clean prefill chunk of 8 rows, ms (48 runs) | 49.64 / 44.50 | 44.42 / 39.69 | ×0.895 |
| | clean run of 1 row, ms (1,140 runs) | 18.97 / 11.82 | 19.05 / 12.94 | ×1.004 |
| | time to the first token, ms | 1135.0 / 976.7 | 1077.0 / 906.6 | ×0.949 |

- Without self-speculation every decode run has one row, both arms run the same kernels, and the paired ratio is the
  noise of this card on that day: ±8% between generations of one arm, 1% on the medians.
- With self-speculation at depth 4 the verify of 5 rows takes the grouped form, and all six paired ratios are below
  0.89. The size of the gain depends on the card's thermal state: the same clean prefill chunk of 8 rows took 91.5 ms
  with the one-row kernel in the first process and 49.6 ms in the second, and gained ×0.68 and ×0.90. The runs of
  several rows are the sustained heavy kernels, and they slow down the most when the card throttles. On a cool card
  the gain per decode token is expected to be smaller than ×0.85; it was not measured on one.
