> This document is a point-in-time snapshot (2026-11-03) of the measurements behind the integer GEMV (ADR
> [0140](../decisions/0140-integer-gemv.md), perf-ledger K-45): QAT Gemma 4 E2B decode on the maintainer's Apple M2
> under Chrome, against the WebML Community "Gemma 4 WebGPU kernels" Space, the split attribution of Karume's decode, the
> single-kernel timings and kernel checks of the integer GEMV, and its end-to-end A/B. It is not a design source of
> truth. The raw records are JSON in the localStorage of that Chrome (§10); nothing here was measured on the RTX 3080 Ti.

# QAT Gemma 4 E2B on Apple M2: decode attribution and the integer GEMV

## 1. Summary

- On the same M2, Karume's QAT E2B (`i4-fast`, greedy, no timing tap) decoded at 36.7 / 38.1 tok/s (median, two
  prompts) while the WebML Space decoded at 46.1–50.8 tok/s (§3).
- The split attribution put **linear at 24.8 ms of GPU time per token (80.7%)** on QAT E2B (§4). The M2 GEMV is bound
  by the per-element weight decode, not by memory: the same loop without the decode streams at 79 GB/s, with the
  shift-mask decode at 26–37 GB/s, against a read roofline of about 83 GB/s (§5).
- The integer GEMV (exact integer sums per group, the byte-plane `unpack4x8unorm` decode, row blocking) matched its
  CPU mirror word for word on every case of its check: 248 / 248 (stage 1, i2) and 692 / 692 (stage 2, i2 / i4 / i8).
  Every fault injected was caught except a bare `f32(S)` conversion, which the M2 rounds correctly by itself (§6, §7).
- Single kernel: i2 down 292 → 102 µs, i2 gate / up 165 → 131 µs (medians, stage 1); i4 o 2048 46.2 → 24.7 µs, o 4096
  82.7 → 45.9 µs, own down 168.5 → 73.4 µs (stage 2). The i4 shapes at K = 1536 and the i8 PLE shapes were not
  faster and were not adopted (§7).
- End to end (stage 2, i2 + i4): **27.767 → 22.754 ms/token (Story, ×0.820) and 27.835 → 22.768 ms/token (Explainer,
  ×0.818)**. The A / B / B / A attribution gives the i4 shapes 0.93 / 1.68 ms of that, against 2.3 ms expected from the
  single-kernel table (§8).
- A Metal trap: an epilogue spelled `rounded(0.0 + rounded(t))` let the two rounding barriers cancel and Metal
  reassociated the scales, giving 1-ulp mismatches on every f32-output case (§9).

## 2. Environment

| Item | Value |
| --- | --- |
| Machine | Apple M2 (adapter `apple` / `metal-3`), `navigator.deviceMemory` 16, 8 cores |
| Browser | Chrome 158, cross-origin isolated; `subgroupMinSize` = `subgroupMaxSize` = 32; features include `subgroups`, `subgroup-size-control`, `shader-f16`, `timestamp-query` |
| Serving | the playground dev server on the development machine, reached over an SSH tunnel (127.0.0.1:8790); distributions over the direct route |
| Code (§3) | checkout `34c49f32` plus uncommitted Gemma changes of another session (54 working-tree entries) |
| Code (§4–§9) | worktree `karume-wt-qat-gemv` (branch `wip/qat-gemv` from `34c49f32`), uncommitted |
| Model | QAT Gemma 4 E2B (`gemma4-qat`, preset `i4-fast`); regular E2B `i4-fast` as a control in §3 / §4 |
| Pages | `/gemma4-bench` (end-to-end, the GPU tap, split dispatches, the "Integer GEMV" arm switch — apps/playground/README.md); the single-kernel check and timing through `packages/kernel-checks/src/gemv-integer.ts` (`checkIntegerGemv`, `benchIntegerGemv`) imported in the page's console |

## 3. Starting point: Karume against the WebML Space

Karume `/gemma4-bench`, greedy, 3 chats × 128 tokens, 4 rounds in ABBA order, 16 visits / 96 chats; the outputs of
each arm were identical across chats:

| Arm | Prompt | tok/s med · max | ms/token med · min | GPU ms/token med · min | wall − GPU med · min | dispatch · pass · submit / token |
| --- | --- | --- | --- | --- | --- | --- |
| regular e2b i4-fast (tap) | story | 33.3 · 35.9 | 30.040 · 27.876 | 25.078 · 24.822 | 4.581 · 3.045 | 896 · 8.00 · 9.00 |
| regular e2b i4-fast (tap) | explainer | 32.4 · 34.7 | 30.831 · 28.798 | 25.343 · 24.956 | 4.834 · 3.841 | 896 · 8.01 · 9.01 |
| QAT e2b i4-fast (tap) | story | 32.9 · 34.9 | 30.445 · 28.616 | 25.764 · 24.960 | 4.653 · 3.656 | 1036 · 10.00 · 11.00 |
| QAT e2b i4-fast (tap) | explainer | 33.2 · 34.7 | 30.125 · 28.849 | 25.264 · 25.197 | 3.917 · 3.652 | 1036 · 10.00 · 11.00 |
| regular e2b i4-fast (no tap) | story | 37.4 · 38.8 | 26.787 · 25.796 | — | — | — |
| regular e2b i4-fast (no tap) | explainer | 37.9 · 38.4 | 26.379 · 26.029 | — | — | — |
| QAT e2b i4-fast (no tap) | story | 36.7 · 38.5 | 27.239 · 26.007 | — | — | — |
| QAT e2b i4-fast (no tap) | explainer | 38.1 · 38.1 | 26.272 · 26.230 | — | — | — |

Round 1 (visits 1–2) ran slower (GPU 28.9–31.9 ms/token); rounds 2–4 settled at about 25 ms.

The same page with the distribution sampler (temperature 1, top-k 64, top-p 0.95, seed 0), no tap, 2 rounds:

| Arm | Prompt | tok/s med · max | ms/token med · min |
| --- | --- | --- | --- |
| regular e2b i4-fast | story | 34.4 · 37.6 | 29.109 · 26.599 |
| regular e2b i4-fast | explainer | 34.6 · 37.0 | 28.949 · 27.008 |
| QAT e2b i4-fast | story | 34.1 · 34.4 | 29.283 · 29.032 |
| QAT e2b i4-fast | explainer | 34.2 · 34.6 | 29.215 · 28.901 |

The WebML Space in the same Chrome, after the Karume tab was closed; greedy only; tok/s = (tokens − 1) / (end − first
token):

| Prompt | Run | tokens | TTFT ms | tok/s |
| --- | --- | --- | --- | --- |
| story | warm-up | 1555 | 518 | 43.5 |
| story | 1–3 | 1555 | 494 / 440 / 446 | 46.1 / 48.5 / 48.5 |
| explainer | 1–3 | 313 | 823 / 822 / 806 | 50.1 / 50.8 / 50.2 |
| story | stopped ~4 s ×2 | 234 | 457 / 456 | 50.4 / 50.4 |
| explainer | stopped ~4.5 s ×2 | 215 | 809 / 811 | 50.2 / 50.1 |

## 4. Stage 0 — split attribution of QAT E2B decode

Worktree build, 2 rounds, 2 chats × 64 tokens, greedy, every dispatch in its own timed pass ("split dispatches"). The
split adds pass boundaries: QAT split GPU 30.4–30.8 ms/token against about 25.3 unsplit (≈ 5.3 µs per dispatch), so
the shares, not the absolute times, are the result.

QAT E2B `i4-fast`, Story (the Explainer within ±0.2 ms):

| ms/token | dispatches/token | key | weight MB/token | GB/s |
| --- | --- | --- | --- | --- |
| 5.884 | 40 | `linear_gemv_parallel:wi2:l2:static-quantize:v2` (the shared gate / up, 12288×1536) | 188.7 | 32 |
| 5.623 | 20 | `linear_gemv_parallel:wi2:l32:static-quantize:v2:packed-x-i8` (the shared down, 1536×12288) | 94.4 | 17 |
| 3.855 | 65 | `linear_gemv_parallel:wi4g512:l4:static-quantize:v2` (q and the own gate / up) | 207.6 | 54 |
| 3.788 | 43 | `linear_gemv_parallel:wi4g2048:l32:static-quantize:v2:packed-x-i8` (o of the sliding layers and the own down) | 114.8 | 30 |
| 2.917 | 1 | `linear_gemv:v1:f32:c32u4:wi2` (lm_head, sequential) | 100.7 | 35 |
| 0.837 | 1 | `linear_gemv:v1:f32:c32u4` (PLE projection, f32) | 55.1 | 66 |
| 0.707 | 7 | wi4g4096 l32 packed (o of the full layers) | 22.0 | 31 |

Families: linear 24.8 ms (80.7%), attention core 1.9, norm 1.7, static quantize 1.0, elementwise 0.7, attention
other 0.6.

Regular E2B `i4-fast`: linear 22.6 ms (82.5%) — wi4g32 l4 10.2 / wi4g32 l32 staged 7.2 / lm_head wi8 l16 4.6 / wi4g32
l32 0.8.

## 5. What bounds the M2 GEMV

The same GEMV loop without the weight decode streams at 79 GB/s; with the shift-mask decode the shapes above run at
26–37 GB/s; the read roofline is about 83 GB/s. The GEMV is therefore bound by the per-element decode ALU, not by
memory. (On the RTX 3080 Ti, K-45 step 0 had found the activation loads to be the bound —
[research 2026-09-19](2026-09-19-qat-speed-recon.md) §14 — which ADR 0105's packed activations addressed.)

From the WebML Space's kernels two techniques were taken: the `unpack4x8unorm` byte-plane decode (one instruction
turns four codes into floats) and row blocking (one activation load reused by several output rows). Three were not:
using `code / 255` floats without re-rounding (an approximation, class Q of ADR 0110), the last-arriver workgroup
merge with float atomics (forbidden by ADR 0110 §6), and the GPU argmax run-ahead (greedy only; out of scope).

## 6. Stage 1 — the i2 kernel

The first kernel covered i2 only (u32 weight words, the byte planes turned into floats by `unpack4x8unorm`, rounded
back to the exact integer every 16 elements by `round(255 · s)`, row blocking, the zero point folded through `Σ c`).

- Kernel check: 248 / 248 cases equal the CPU mirror on the M2. Fault injection: no per-step rounding, no zero point
  and swapped planes each failed the check.
- Single kernel (median): i2 down 292 → 102 µs; i2 gate / up 165 → 131 µs.
- End to end (`/gemma4-bench`, QAT E2B, greedy, tap off, 3 rounds): 38.4 → 43.8 tok/s and 38.0 → 44.4 tok/s for the two
  prompts (ms/token ×0.876 / ×0.856). Generated text against the option off: a Japanese generation of 160 tokens was
  identical; an English one diverged at its 111th character, both continuations natural. The record of §8 refers back
  to this run as baseline 26.05 ms/token against integer 22.5–22.8 ms/token (a gap of 3.2 ms).

## 7. Stage 2 — i2 / i4 / i8

The kernel was generalised: i4 (offset 8, one scale per group; the groups unrolled at generation, each with its own
exact sum) and i8 (symmetric; `unpack4xI8` gives the exact integers, no rounding needed). The lane partials meet in an
i32 tree; `S` reaches f32 as `f32(S >> 12) · 4096 + f32(S & 0xfff)`; the epilogue is `Σ_g round(S_g · w_scale_g)` in
ascending group order, then `· x_scale`, then `+ bias` (ADR 0140 decisions 1–5).

Kernel check: 692 / 692 cases equal the reference. The mutants (counts as recorded: the mismatching words of the case
each mutant ran on; the check's fixtures have 37 output channels per row):

| Storage | Mutant | Result |
| --- | --- | --- |
| i4 | no per-step rounding | 35 / 37 |
| i4 | even / odd planes swapped | 37 |
| i4 | offset 7 instead of 8 | 37 |
| i4 | group scale wrong | 37 |
| i4 | group order reversed | 5 |
| i8 | lanes swapped | 10 |
| i8 | truncating integer-to-f32 conversion | 37 |
| i8 | bare `f32(S)` conversion | not caught — the M2 converts round-to-nearest-even; the hi / lo form exists for portability |

Single kernel, M2, with the SRQ epilogue (µs, the median of 15 rounds; the q 4096 parallel cell adds the minimum,
because its median and minimum were 1.8 times apart):

| Shape (per token) | Parallel GEMV (production) | Best integer geometry |
| --- | --- | --- |
| i4 q 2048×1536 ×28 | l4 27.1 | L4R2 29.7 |
| i4 q 4096×1536 ×7 | l4 87.0 / 47.8 | L4R4 54.0 |
| i4 k/v 256×1536 ×24 | l32 9.2 | L16R1 10.7 |
| i4 k/v 512×1536 ×6 | l32 15.7 | L16R2 13.7 |
| i4 gate/up 6144×1536 ×30 | l4 78.1 | L4R2 84.1 |
| i4 o 1536×2048 ×28 | l32 packed 46.2 | L16R4 24.7 |
| i4 o 1536×4096 ×7 | l32 packed 82.7 | L16R2 45.9 |
| i4 down 1536×6144 ×15 | l32 packed 168.5 | L8R2 73.4 |
| i8 PLE gate 256×1536 ×35 | l32 8.6 | L32R2 8.8 |
| i8 PLE proj 1536×256 ×35 | l4 7.4 | L4R2 8.0 |

Adopted: o 2048, o 4096 and the own-KV layers' down (expected −2.3 ms/token). The K = 1536 shapes share their input SRQ
within each layer (q / k / v one, gate / up one), and the parallel GEMV does not read packed activations at those
shapes, so one of them cannot move alone (ADR 0140, alternatives).

## 8. Stage 2 end to end

`/gemma4-bench`, QAT E2B, greedy, tap off, 3 rounds in ABBA order, warm chats:

| Arm | Story ms/token (med · min) | Explainer ms/token (med · min) |
| --- | --- | --- |
| baseline | 27.767 · 27.667 (36.0 tok/s) | 27.835 · 27.815 (35.9) |
| integer i2 + i4 | 22.754 · 22.636 (43.9) ×0.820 | 22.768 · 22.670 (43.9) ×0.818 |

The outputs were identical across rounds within each arm. The baseline ran 1.8 ms slower than in the stage 1 A/B
(26.05) — machine state; the integer arm's absolute time is that of stage 1 (22.5–22.8), so the gap is 5.0 ms here
against 3.2 ms at stage 1. A separate one-round integer-only run gave 22.83 ms. The shader headers captured in the page
showed the integer kernels `wi2 l8r4` / `l16r4`, `wi4g2048x1 l16r4`, `wi4g4096x1 l16r2`, `wi4g2048x3 l8r2`, all with
the SRQ epilogue: the i4 path was live.

Attribution under the same conditions: arm A (i2 + i4) and arm B (i2 only) in the order A B B A, each run a baseline
plus an integer arm, one round; warm-chat median ms/token, Story / Explainer; delta = baseline − integer:

| Run | Baseline | Integer | Delta |
| --- | --- | --- | --- |
| A1 | 29.739 / 29.958 | 24.713 / 25.025 | 5.026 / 4.933 |
| B1 | 27.720 / 27.773 | 23.552 / 24.728 | 4.168 / 3.045 |
| B2 | 27.809 / 27.785 | 23.692 / 24.305 | 4.117 / 3.480 |
| A2 | 27.834 / 27.738 | 22.711 / 22.793 | 5.123 / 4.945 |

Mean delta: A 5.075 / 4.939, B 4.143 / 3.263 ms, so the i4 shapes add 0.93 / 1.68 ms (the single-kernel estimate was
2.3 ms). B's integer arm was less stable on the Explainer (24.3–24.7 against 23.6 on the Story); with i4 both sat at
22.7–22.8. The shader headers were confirmed per run (A: five integer kernels, three of them i4; B: the two i2). The
shape table was restored after the B runs (sha `4f4c5179da815ba7`).

## 9. The Metal reassociation trap

The epilogue separates its roundings with a run-time barrier, `rounded(v) = bitcast<f32>(bitcast<u32>(v) ^
rounding_mask)` with `rounding_mask` a uniform that is always 0, so the compiler can neither contract a multiply-add nor
regroup the products. A first spelling started the group sum from zero: `total = rounded(0.0 + rounded(t))`. Once
`0.0 + y` simplified to `y`, the two XORs with the same mask were adjacent and cancelled, and Metal reassociated
`(S · w_scale) · x_scale` into `S · (w_scale · x_scale)`: 1-ulp mismatches on every f32-output case. Starting the sum
with the first group's term removed the pair; the kernel now states it as a MUST.

## 10. Open items and records

- Not run: the GPU tests on the development machine's RTX 3080 Ti (`packages/runtime/tests/gpu_linear_gemv_integer_test.ts`)
  and the sha256 rows of ADR 0110 §7. No speed was measured on any backend other than the M2's Metal.
- The i4 end-to-end gain (0.93 / 1.68 ms) is below the single-kernel estimate (2.3 ms); not investigated.
- The q 4096 parallel cell's median and min differ by ×1.8 (87.0 / 47.8 µs), so the full layers' K = 1536 group (q
  4096 with k/v 512, the one group where the integer kernel led on the recorded values) is not settled.
- `/gemma4-bench`'s split mode failed on the integer arm (`setBindGroup: The provided value cannot be converted to a
  sequence`, in `tap.ts`; not reproduced in isolation) — [known issues](../known-issues.md).
- Records (localStorage of `http://127.0.0.1:8790` in that Chrome): `claude-m2-bench-greedy` (131,214 chars) and
  `claude-m2-bench-sampler` (28,120 chars) for §3; `claude-m2-bench-integer-ab` for §6;
  `claude-m2-bench-integer-i4i8` for §7; `claude-m2-bench-integer-i4-ab` (+ `-summary`) and `claude-ab-runs` for §8.
