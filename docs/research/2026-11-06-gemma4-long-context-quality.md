> This document is a point-in-time snapshot (2026-11-06) of the Gemma 4 long-context quality check on the maintainer's RTX
> 3080 Ti under Node + Dawn: KL against upstream torch on the CPU, and a retrieval test up to P = 128K. It is not a design
> source of truth. Time and memory are in [research 2026-11-05](2026-11-05-gemma4-long-context.md). Nothing here was run
> in Chrome. The harness and the raw records are local and untracked (`outputs/diag/long-context-quality-2026-11-05/`).

# Gemma 4 at long context: quality against torch

## 1. Summary

- Karume follows upstream at long positions. Against an f32 torch reference, KL shows no growth with position up to
  P = 127,985 on E2B and P = 31,762 on 12B, and the top-1 token agrees on every compared prefill row (§3).
- On all 7 prompts run through both stacks, torch and Karume produced the same text, including 5 prompts where E2B
  answered wrongly (§4). The wrong answers belong to the model, not to the port.
- 12B retrieved every fact: 192 of 192 over 24 runs, up to P = 63,998 with a 256 MiB row-block limit (§5).
- E2B retrieval weakens with length: 318 of 320 facts up to 32K, 52 of 64 at 64K, 48 of 64 at 128K. The loss is mostly on
  the prose corpus (20 of 32 at 64K, 18 of 32 at 128K); the repetitive logbook holds at 64K and gives 30 of 32 at 128K (§5).
- E2B's wrong answers are bindings, not corrupted digits: of 30, 14 are another locker's code, 16 are `unknown`, and none
  is an invented number.
- `i4` and `i4-fast` logits are bit-identical on all 84 rows compared (2 prompts × 42 rows, P ≈ 32K).

## 2. Setup

- Prompts: a long document, then a request. Two fillers: `logbook` (a numbered, repetitive lighthouse log) and `adr` (the
  paragraphs of `docs/decisions/` in file order, which is real prose). Both are sized by binary search to the target P.
- Retrieval test: 8 sentences "the access code of the <name> locker is <5 digits>" at 2, 15, 30, 45, 60, 75, 90 and 98% of
  the document, then a question over the 8 names and one name that is not in the document. Greedy, 160 new tokens. Scored
  by exact match. 2 seeds × 2 fillers. E2B `i4-fast` and `bf16` at P = 1K–128K; 12B `i4` at 1K–32K, and at 64K with a
  256 MiB row-block limit. 80 runs.
- KL: KL(torch ‖ Karume) on the logits of the last row of each prefill chunk, so one row per 768 positions. Rows of low
  confidence weigh in KL as much as rows of high confidence.
- Reference: upstream `transformers` 5.14.1, the checkpoint's own class, `sdpa`, `DynamicCache`, cut at Karume's chunk
  boundaries, greedy, f32 on the CPU (10 threads). The q4_0 QAT presets compare against upstream's unquantized QAT
  checkpoint, `bf16` against the regular checkpoint. The reference for a quantized preset is therefore not the same weights:
  KL includes the quantization error.
- llama.cpp was the planned second reference for 12B and the long end. It was not used, because the f32 torch reference
  reached both, and it would have added a different quantization and an unverified Gemma 4 path.

## 3. KL against torch

Prefill rows, 12B and the longest E2B run, by position.

| Position | E2B `i4-fast` P = 127,935: mean / max | 12B `i4` P = 31,762: mean / max |
|---|---|---|
| 0–1K | 3e-6 / 3e-6 | 9.3e-5 / 9.3e-5 |
| 1K–2K | 0 / 0 | 2.0e-4 / 2.0e-4 |
| 2K–4K | 5.5e-5 / 1.0e-4 | 3.4e-5 / 5.2e-5 |
| 4K–8K | 3.9e-5 / 9.9e-5 | 3.7e-4 / 1.4e-3 |
| 8K–16K | 7e-6 / 3.0e-5 | 2.0e-3 / 8.7e-3 |
| 16K–32K | 6.3e-5 / 4.7e-4 | 2.9e-4 / 2.9e-3 |
| 32K–64K | 3.8e-5 / 2.5e-4 | not run |
| 64K–128K | 5.0e-5 / 7.0e-4 | not run |

- Neither column grows with position. The 12B bump at 8K–16K has a median of 1.6e-4, and its largest row (KL 8.7e-3,
  position 16,127) is a low-confidence row (top-1 probability 0.37 in torch, 0.40 in Karume) with the same top-1.
- 12B's floor is higher than E2B's. This run does not say why. It is one prompt.
- `bf16` against the regular checkpoint is lower still: at most 1.4e-5 at P ≈ 32K and 6.5e-5 at P = 127,935.
- The two P ≈ 32K prompts: `i4` and `i4-fast` reach at most 2.6e-4 on `adr` and 1.3e-5 on `logbook`.
- Top-1 agrees on every row in every f32 comparison.

**A bf16 reference is not good enough.** Running torch itself in bf16 on the CPU gave a false alarm: KL 0.995 and a top-1
disagreement at position 16,127 of the 12B prompt. The f32 reference gives KL 8.7e-3 and the same top-1 at that row, so the
error was the reference's round-off. At P = 7,838 the bf16 reference reads up to 1.0e-2 where the f32 one reads
2.2e-4, and it disagrees on one top-1 there too. The two bf16 runs still produced the same text.

## 4. The same prompts through torch

| Prompt | Karume preset / reference | Facts, torch | Facts, Karume | Same text | Largest KL | Top-1 |
|---|---|---|---|---|---|---|
| E2B P = 63,790, `adr`, seed 2 | `i4-fast` / QAT unquantized | 4 | 4 | yes | 3.4e-4 | all |
| E2B P = 63,790, `adr`, seed 2 | `bf16` / regular | 4 | 4 | yes | 2.4e-5 | all |
| E2B P = 127,985, `logbook`, seed 2 | `i4-fast` / QAT unquantized | 6 | 6 | yes | 1.5e-4 | all |
| E2B P = 127,935, `adr`, seed 2 | `bf16` / regular | 3 | 3 | yes | 6.5e-5 | all |
| E2B P = 127,935, `adr`, seed 1 | `i4-fast` / QAT unquantized | 5 | 5 | yes | 7.0e-4 | all |
| 12B P = 7,838, `adr`, seed 1 | `i4` / QAT unquantized | 8 | 8 | yes | 2.2e-4 | all |
| 12B P = 31,762, `adr`, seed 1 | `i4` / QAT unquantized | 8 | 8 | yes | 8.7e-3 | all |

- The E2B run at P = 127,935, seed 1 gives a code for the name that is not in the document. Torch gives the same one.
- On 12B the decode rows were compared as well: all 85 tokens agree, with KL at most 1e-6.
- On E2B the capture saw prefill rows only, because decode returned no full logits row through `Session.run`. Its decode is
  compared by the text alone. Every prompt's text was reproduced exactly by a second Karume run.

## 5. Retrieval

Facts correct, both presets and seeds together for E2B, seeds together for 12B.

| P | E2B `adr` | E2B `logbook` | 12B `adr` | 12B `logbook` |
|---|---|---|---|---|
| 1K | 32 / 32 | 32 / 32 | 16 / 16 | 16 / 16 |
| 4K | 32 / 32 | 32 / 32 | 16 / 16 | 16 / 16 |
| 8K | 31 / 32 | 32 / 32 | 16 / 16 | 16 / 16 |
| 16K | 32 / 32 | 32 / 32 | 16 / 16 | 16 / 16 |
| 32K | 31 / 32 | 32 / 32 | 16 / 16 | 16 / 16 |
| 64K | 20 / 32 | 32 / 32 | 16 / 16 (256 MiB row block) | 16 / 16 (256 MiB row block) |
| 128K | 18 / 32 | 30 / 32 | not run | not run |

- 11 of the 80 runs missed a fact, all E2B. Wrong answers by position in the document: 2%: 4, 15%: 8, 30%: 2, 75%: 5,
  90%: 9, 98%: 2, none at 45% or 60%. Neither end alone explains them.
- The name that is not in the document: 12B answers `unknown` in all 24 runs. E2B answers `unknown` in 46 of 56, leaves the
  line out in 9, and gives a code in 1 (P = 127,935).
- Speculation, not measured: E2B's limit is about binding a name to its value across a long, varied text, which the larger
  12B handles here, while the logbook's repetition makes the unusual sentences stand out.

## 6. What this does not cover

- 6 of the 11 runs that missed a fact were not run through torch, all on `adr`: E2B `i4-fast` at 8K (seed 1), 32K (seed 2),
  64K (seed 1) and 128K (seed 2), and `bf16` at 64K and 128K (seed 1). The 5 that were all agree.
- KL on 12B above P = 31,762, retrieval on 12B above 64K, and any torch run of `i4` itself (its logits are bit-identical
  to `i4-fast` up to P ≈ 32K only).
- KL at positions between chunk boundaries, and non-greedy sampling.
- Tasks other than retrieving short facts: this says nothing about summaries or reasoning over a long text.
- E4B, and Chrome or the 5070 Ti.
- 2 seeds per cell: the counts in §5 show a tendency, not a rate.
