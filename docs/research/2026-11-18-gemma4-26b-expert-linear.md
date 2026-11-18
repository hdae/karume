# Gemma 4 26B A4B: what an `expert_linear` dispatch costs, and the grouped form (2026-11-18)

A point-in-time measurement record. It takes the routed experts of a 26B forward apart (perf-ledger K-103, K-104),
gives the measurements behind the grouped form of `expert_linear`, and records what was measured of the one-row kernel
and of a parallel prototype. Scripts, logs and records: `outputs/diag/decode-optimization-2026-11-18/expert-linear/`.

## 1. Summary

- A run of several tokens now walks an expert's weight row once for the rows that picked it (the grouped form, for
  gate_up's long rows). Decode under the expert cache with self-speculation at depth 4 takes ×0.85 of the time per
  token on the RTX 3080 Ti with the same tokens; a clean verify of 5 rows goes from 64.5 to 48.1 ms. The card was
  throttling; the gain is expected to be smaller on a cool one (§6).
- A one-token run is unchanged by it. Of its 2.9 ms of routed experts per forward, 2.3 ms is gate_up
  (77 µs per dispatch) and 0.6–0.7 ms is down (21–23 µs).
- The 48 µs per dispatch that K-103 compared with 17–30 µs for dense rows is the time of eight weight rows: one
  `expert_linear` dispatch reads the 8 picked experts. Per multiply-add gate_up runs at about half the rate of the
  dense parallel GEMV's head, and down at about its rate.
- Two bit-identical changes of the one-row kernel gain in the kernel bench (8 words to a round: 13% of gate_up; 4
  words read a round ahead: 29%). The second was built and showed nothing in the untimed forward, so it was taken out
  (§4.1).
- A parallel prototype (the reorder class, perf-ledger K-103) takes 0.9 ms off the median forward of 18.8 ms and
  1.2 ms off the fastest one (§4.2): at K-103's line of 1 ms. Nothing was built for it; it is the maintainer's
  decision.

## 2. Setup

The RTX 3080 Ti (12 GB) under Node and Dawn, branch `wip/opt-expert-linear` from main `7834560b`. 26B runs under the
exact expert cache (64 slots per layer unless stated, every expert in the host tier), preset `i4-fast` unless stated,
greedy, the three prompts of [research 2026-11-10](2026-11-10-gemma4-26b-fast.md) §2 with 96 new tokens.

The card was shared with four other measurement legs and ran at 77–93 °C, with its thermal slowdown flag set in most
runs, so absolute times are slower than those of the earlier records (a decode token at depth 4 took 50–60 ms against
38–42 ms on 2026-11-10). Every comparison below is inside one process with the arms interleaved.

- **Kernel bench** (`kernel-bench.ts`): one dispatch of the pooled kernel (i4 codes, f16 steps, 64 slots of 128
  experts) at the two shapes of 26B — gate_up (n 1408, k 2816, x shared by the picks) and down (n 2816, k 704, one x row
  per pick) — with 8 picks per token. Up to 256 dispatches in one timed pass, the variants interleaved round by round
  (15 rounds; median and minimum), every variant's output compared word for word with the first. Synthetic routing: a
  token keeps each pick of the token before it with probability `overlap`.
- **Decode A/B** (`decode-ab.ts`): one pipeline, two arms on one Session, switched at recipe time. Arm `base` compiles
  `expert_linear` from the kernel module of the branch point; arm `grouped` is the tree; arm `par` is the tree with
  the parallel prototype for runs of one token. Per prompt one untimed generation, then A B B A (B A A B on odd
  rounds); the token ids of every generation are compared. A **clean** run is one that missed no expert.

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
- The parallel prototype (the form of `linear`'s parallel GEMV: lanes stride the words, `fma`, a tree of adds;
  `par-proto.ts`) takes 46% off gate_up at 4 lanes and 11% off down at 2 lanes: 38.6 µs per layer, 1.16 ms per forward
  of per-kernel time. It changes the summation order.

### 4.1 Reading a round ahead

The kernel reads the 4 words of a round and then sums them, so a thread waits for memory at the start of every round;
a run of one token has 352 (gate_up) or 704 (down) warps of threads for the whole GPU, too few to hide the wait. A
second bench (`logs/job-tests-bench.log`, the card at 77–88 °C) measured more words to a round, and a look-ahead in
which each round reads the words and steps of the next round before its own multiply-adds. The multiply-add text and
its order are the same in all of them, and every output word was identical.

| Variant | gate_up | down |
| --- | ---: | ---: |
| 4 words to a round (the kernel at the branch point) | 75.6 / 75.4 | 20.4 / 19.9 |
| 8 words | 66.3 / 66.2 (×0.877) | 20.0 / 19.8 (×0.984) |
| 11 words | 67.9 / 67.7 (×0.898) | 19.7 / 19.5 (×0.969) |
| 22 words | 63.8 / 63.7 (×0.844) | 19.4 / 19.2 (×0.955) |
| 4 words, read a round ahead | 53.7 / 53.5 (×0.711) | 20.1 / 19.3 (×0.985) |
| 8 words, read a round ahead | 69.6 / 69.4 (×0.921) | 22.3 / 21.3 (×1.094) |
| 11 words, read a round ahead | 80.6 / 80.3 (×1.067) | 23.4 / 22.6 (×1.150) |
| 22 words, read a round ahead | 80.8 / 80.7 (×1.069) | 26.8 / 25.9 (×1.317) |

- In the bench, reading 4 words a round ahead takes 29% off gate_up (21.8 µs per dispatch) and leaves down where it
  was: 22 µs per layer, about 0.66 ms per forward of per-kernel time.
- **The forward does not show it.** With the look-ahead built into the one-row kernel (key `v2`; the GPU tests and the
  kernel check passed with it), the decode A/B without self-speculation (`logs/ab-lookahead-d0.log`, 12 generations of
  64 tokens per arm, the same tokens) gave a clean one-row run 18.15 ms at the branch point and 18.33 ms with it
  (median of 756 runs each; minimum 12.11 and 12.73 ms), and a paired decode ratio of ×0.981 (0.895–1.125). The same
  comparison of two arms with identical one-row kernels gave 18.97 and 19.05 ms (§6). The gain is below 1% of a decode
  token, so the look-ahead was taken out again; the patch is `lookahead-not-adopted.patch` beside the scripts.
- This is the second time on this backend that per-kernel time taken with timestamp queries (one pass per dispatch, or
  a stack of one dispatch) overstates what a faster small-to-medium dispatch buys in an untimed forward (research
  2026-11-10 §8.1 was the first). Why was not investigated; the card ran at 79–91 °C during the A/B, mostly at 90.

### 4.2 The parallel prototype in a forward

The decode A/B without self-speculation with the prototype for every run of one token (4 lanes for gate_up, 2 for
down) against the tree (`logs/ab-fast-64-d0-par.log`, 12 generations of 64 tokens per arm):

| Measure | Tree | Prototype |
| --- | ---: | ---: |
| clean run of 1 row, ms, median / minimum (756 runs each) | 18.85 / 12.84 | 17.96 / 11.68 |
| decode ms per token, median / minimum | 48.90 / 46.32 | 48.83 / 43.36 |

- The forward is 0.89 ms shorter by the median and 1.16 ms by the minimum (the per-kernel estimate was 1.16 ms).
  K-103's line is 1 ms off the forward.
- Decode under the cache did not move measurably (paired ratio ×0.999, 0.84–1.09): at 64 slots without
  self-speculation a decode token is about 2.5 runs with their loads, and the generations differ by more than 5%.
- The token ids were the same as the tree's on these three prompts, which is an observation: the prototype reorders
  the sum. It is a measurement aid — it has no check of the id or the slot and no grid-stride — and nothing of it is in
  the tree. A kernel of this form would be the op's second numeric contract, shared by the pooled and the resident
  form, selected by a Session at `reorder`; `i4-fast` declares that level, so its numbers on 26B would move.

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

### 5.1 Long runs

A resident prefill chunk has up to 768 tokens (6,144 pick rows), and the pad rows of a bucket are likely to pick the
same few experts. In the grouped form one thread per column then walks all the rows of such an expert, a walk of 8
after another, and every thread scans the run's rows. The kernel bench at gate_up (`logs/job-long-runs.log`; 11 rounds
at 16 and 64 tokens, 7 at 256 and 768, at least 2 dispatches to a pass; the pool of 64 slots, so every slot is picked
from 64 tokens on) with the synthetic routing and with a pad-heavy one, in which only the first quarter of the tokens is
routed that way and every later token picks the same 8 experts in the same order. ms per dispatch, median / minimum;
every output word identical:

| T | Routing | One-row | Grouped |
| ---: | --- | ---: | ---: |
| 16 | overlap 0.5 | 1.130 / 1.126 | 0.521 / 0.520 (×0.461) |
| 16 | 4 routed, 12 on the same 8 experts | 0.972 / 0.966 | 0.467 / 0.464 (×0.481) |
| 64 | overlap 0.5 | 4.977 / 4.966 | 1.292 / 1.279 (×0.260) |
| 64 | 16 routed, 48 on the same 8 | 4.327 / 4.305 | 1.690 / 1.657 (×0.390) |
| 256 | overlap 0.5 | 20.27 / 20.22 | 4.30 / 4.24 (×0.212) |
| 256 | 64 routed, 192 on the same 8 | 18.19 / 17.95 | 6.33 / 6.21 (×0.348) |
| 768 | overlap 0.5 | 61.09 / 60.92 | 12.98 / 12.49 (×0.212) |
| 768 | 192 routed, 576 on the same 8 | 64.91 / 39.79 | 27.47 / 19.53 (×0.423) |

- The grouped form is faster at every length measured, and by more the longer the run: the rows per expert grow with
  the run while the weight row is still read once per 8 of them. With three quarters of the rows on 8 experts the gain
  is smaller (those 8 × 1408 threads carry most of the work) and still more than half.
- So the grouped form is taken for runs of 2 to 768 tokens, the lengths measured (`EXPERT_LINEAR_GROUP_MAX_TOKENS`);
  768 is also the chunk length of the distributions. A longer run keeps the one-row kernel.
- This is the kernel alone, over a pool. Not measured: a resident forward at these lengths (26B does not fit this
  card), the resident weight of 128 experts, and a run with more than three quarters of pad rows.

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

- Without self-speculation every decode run has one row and both arms run the same kernels: the paired ratio moved
  between 0.91 and 1.08 around 1.009, which is the noise of this card on that day.
- With self-speculation at depth 4 the verify of 5 rows takes the grouped form, and all six paired ratios are below
  0.89. The size of the gain depends on the card's thermal state: the same clean prefill chunk of 8 rows took 91.5 ms
  with the one-row kernel in the first process and 49.6 ms in the second, and gained ×0.68 and ×0.90. A possible
  reason, not checked: the runs of several rows are the longest stretches of heavy kernels, and they slow down the
  most when the card throttles. On a cool card the gain per decode token is expected to be smaller than ×0.85; it was
  not measured on one.
- The routing of the prefill's chunks (`[sharing]` in the later logs): a chunk of 8 real rows reaches 30.9 experts with
  its 64 pick rows per layer on average (462 runs), a chunk of 6 real rows 21.8 with 48 — about half, as the bench's
  synthetic routing at overlap 0.5 does. The verify's routing was not recorded.

## 7. Checks

- The GPU tests of the branch on the RTX 3080 Ti (`logs/final-tests-cut.log`): 5 files, 26 tests passed — the grouped
  form against the one-row form word for word, the expert-linear kernel check (the one-row form, the grouped form and
  the grouped form over a pool against `linear`'s GEMV at the 26B shapes, with experts picked by up to 16 rows), the
  Session tests and the expert cache's, one of which runs the grouped form over the pool. The same files pass on
  Mesa's llvmpipe.
- The WGSL of both forms is pinned by snapshots; the one-row fixtures were generated from the kernel module of the
  branch point.
- The 26B cut model (`KARUME_SERIES=1`, `e2e_gemma4_26b_cut_test.ts`; the file takes longer than one GPU job of ten
  minutes, so its cases ran by name in five jobs — `logs/cut-test-3456.log`, `cut-test-7a.log`, `cut-test-7bc.log`,
  `cut-test-7d.log`, `cut-test-7e.log`; 13 cases passed, and the upstream-parity case is opt-in): the sha256 rows of
  this machine at the golden rows still hold with the grouped form in the teacher-forced chunks (④); under the expert
  cache the golden rows are the resident Session's bytes (⑦a, ⑦d) and the tokens are those without the cache, with and
  without self-speculation (⑦b, ⑦c, ⑦d, ⑦e). In ⑦e, 130 verifies had several rows and ran one grouped dispatch per
  gate_up and one one-row dispatch per down, and 6 verifies had no draft and ran every `expert_linear` on the one-row
  kernel. One case the runner counts beside the 14 the file names was left out by the name filters and did not run.
- Every generation of the A/Bs between the branch point's kernel and the tree gave the same token ids in both arms
  (depth 4, without self-speculation, and the look-ahead build without self-speculation).

## 8. Not measured

- A cool card, and any other device: Chrome on the RTX 5070 Ti (the `kernel-check` entry for the identity, the chat for
  the speed), Metal.
- The fitted slots, the preset `i4`, temperature 1, and the per-kernel times of a verify (queued, and withdrawn when the
  GPU queue was cut to one short job at a time).
- The grouped walk with the look-ahead or 8 words to a round; walks of other row counts than 8 / 4 / 2 / 1 (the bench
  measured one loop per count up to 4 and up to 8, §5).
- Resident prefill as a whole (26B does not fit this card): §5.1 has the kernel alone at its chunk lengths, over a
  pool.
