> This document is a point-in-time snapshot (2026-11-05) of the Gemma 4 long-context measurements on the maintainer's RTX
> 3080 Ti (12 GB) under Node + Dawn: decode and prefill time against prompt length, VRAM against the capacity estimate,
> the cost of `attentionRowBlockMaxBytes`, and speculative decode at long positions. It is not a design source of truth.
> It says nothing about output quality at long context, and nothing here was run in Chrome. The harness and the raw records
> are local and untracked (`outputs/diag/long-context-2026-11-05/`).

# Gemma 4 at long context: time and memory on the RTX 3080 Ti

## 1. Summary

- All 54 runs (27 arms × 2 rounds) finished with no allocation failure. E2B ran up to P = 127,962 tokens, 12B up to
  P = 31,978.
- Decode time per token grows linearly with the position. On E2B `i4` it goes from 22 ms at P = 1K to 370–376 ms at
  P = 128K (GPU busy), and `bf16` is about the same at long positions.
- `i4-fast` stays at 24.5–25.9 ms at P = 128K, about 15× faster than `i4`. It produced the same token sequence as `i4` at
  every P. The q4_0 QAT line's default is `i4`, so the default is the slow one for long context.
- Prefill without the slow-mode stalls is 10–12 s / 23–28 s / 57–71 s / 209–215 s at P = 16K / 32K / 64K / 128K on E2B
  `i4`. As measured it is several times that up to 64K, because this machine's sustained-load slow mode hits most E2B
  prefills (§4). The earlier record of that mode is `2026-10-27-gemma4-decode-attribution.md`.
- The capacity estimate is a good base for a fit check. The sustained VRAM exceeds it by a stable amount (about 320 MiB
  on E2B `i4`, about 140 MiB on `bf16`, about 100 MiB on 12B), but short peaks of 430–670 MiB more were also sampled (§5).
- `attentionRowBlockMaxBytes` costs prefill time and almost no decode time. At P = 128K, 128 MiB saves about 1.4 GiB and
  doubles prefill; 256 MiB saves about 1.3 GiB and costs about ×1.3 (§6). Token sequences did not change.
- Speculative decode cuts wall time per token by 37–39% at P = 16K and 46–48% at P = 64K on `i4`, with the same
  token sequence (§7).

## 2. Setup

- Machine: RTX 3080 Ti, 12,288 MiB, also driving the display; Node + Dawn. 33 MiB of VRAM is in use when idle.
- Path: `Pipeline.fromPretrained` on the local mirrors with the product's own GPU acquisition, greedy
  (`temperature: 0`). Pass-boundary timestamps are off during prefill, because a 128K prefill has more passes than the
  tap has timestamp slots.
- Prompt: a numbered lighthouse logbook sized by binary search to the target P, then a request for a long story.
- One process per arm, each started at ≤ 60 °C with the GPU idle. Each process runs 3 chats: a warm-up (4 tokens), then
  two measured chats (64 tokens). The tables read the last chat, whose host-side blocks are warm. `Prefill` ranges cover
  all three chats of both rounds.
- Capacity: the next power of two ≥ P + 64 + 64, from 4096.
- Arms: E2B `i4` / `i4-fast` (q4_0 QAT line) and `bf16` (regular line) at P = 1K–128K; 12B `i4` at 4K–32K; `i4` with
  `speculative: "always"` at 16K and 64K; `i4` with a 128 / 256 MiB row-block limit at 32K and 128K.
- VRAM: `nvidia-smi` sampled every 500 ms, whole GPU. A shorter peak can be missed, so a peak here is a lower bound.

## 3. Decode time per token

Medians of the last chat, round 1 / round 2, in ms. "Busy" is GPU time from the timestamps; "wall" is between `onToken`
calls.

| P | `i4` busy | `i4` wall | `i4-fast` busy | `i4-fast` wall | `bf16` busy | `bf16` wall |
| --- | --- | --- | --- | --- | --- | --- |
| 969 | 21.8 / 21.8 | 22.7 / 22.8 | 6.4 / 6.3 | 7.1 / 7.0 | 30.4 / 30.3 | 31.4 / 31.2 |
| 3,943 | 29.5 / 29.3 | 30.4 / 30.2 | 6.6 / 6.6 | 7.3 / 7.3 | 39.2 / 39.6 | 40.2 / 40.6 |
| 15,990 | 63.3 / 63.0 | 64.6 / 64.6 | 8.3 / 8.4 | 9.0 / 9.1 | 73.4 / 73.2 | 74.5 / 74.9 |
| 31,974 | 108.5 / 109.1 | 110.2 / 110.8 | 11.9 / 11.1 | 12.7 / 11.9 | 114.5 / 118.3 | 116.0 / 120.1 |
| 63,971 | 195.2 / 193.3 | 197.8 / 195.9 | 14.5 / 14.5 | 15.3 / 15.4 | 196.7 / 200.9 | 198.4 / 202.8 |
| 127,962 | 370.4 / 375.7 | 384.0 / 386.7 | 24.5 / 25.9 | 25.4 / 26.8 | 382.0 / 381.1 | 385.6 / 386.1 |

12B `i4`:

| P | Busy | Wall |
| --- | --- | --- |
| 3,947 | 76.4 / 76.9 | 78.4 / 79.1 |
| 15,994 | 121.3 / 120.3 | 123.4 / 122.7 |
| 31,978 | 179.3 / 180.4 | 182.9 / 183.1 |

**Clocks.** These are not cool-GPU numbers. From P = 16K on, 70–99% of the samples above 500 MHz carry the software
thermal slowdown flag, the peak reaches 93–96 °C, and the median clock is 1,365–1,575 MHz, against 1,665–1,920 MHz at
P = 1K. The
long-position rows are therefore a throttled-clock measurement, and the ratio against the short rows is inflated by that.
The two rounds agree closely.

**Token identity.** Each arm's token sha matches across rounds and chats. `i4` and `i4-fast` match at every P. This covers
64 tokens on these prompts, not a general equality. P = 31,974 and P = 63,971 produced the same 64 tokens on the QAT line,
which is a shared story opening on two similar prompts; it says nothing either way about whether the model uses the
document, which the quality check has to settle.

## 4. Prefill time

Prefill wall in seconds. "As measured" is the range over 3 chats × 2 rounds. "Without stalls" sums the chunks that are not
stalls, where a stall is a chunk over 3 s and over 3× the median chunk of that prefill.

| Arm | P | As measured | Without stalls | Stalls per prefill |
| --- | --- | --- | --- | --- |
| E2B `i4` | 15,990 | 11.0–71.4 | 10–12 | 0–4 |
| E2B `i4` | 31,974 | 89.6–148.9 | 23–28 | 4–8 |
| E2B `i4` | 63,971 | 255.0–325.4 | 57–71 | 10–14 |
| E2B `i4` | 127,962 | 356.4–413.0 | 209–215 | 8–12 |
| E2B `i4-fast` | 127,962 | 289.9–378.0 | 207–220 | 5–10 |
| E2B `bf16` | 127,962 | 286.5–413.1 | 210–222 | 4–12 |
| 12B `i4` | 3,947 | 9.6–11.1 | 10–11 | 0 |
| 12B `i4` | 15,994 | 44.2–52.4 | 44–52 | 0 |
| 12B `i4` | 31,978 | 105.6–113.1 | 106–113 | 0 |

- The stalls are the sustained-load slow mode of the earlier record: 685 stalled chunks over all prefills, usually in
  groups of 2–3, each about 10–30 s against a normal 0.4–1.5 s, with about 10 normal chunks between groups on E2B. One E2B `i4`
  prefill at 16K ran with none, in the same chat of both rounds, and took 11–12 s.
- This explains the odd spread noticed mid-run (2 s vs 12 s at 4K, 11 s vs 43 s at 16K). It is not a difference between
  presets: the stall-free sums of `i4`, `i4-fast` and `bf16` agree.
- The perf ledger records that an earlier 5070 Ti run did not show this mode. The as-measured prefill may therefore be
  specific to this machine, but no long-context prefill has been measured elsewhere.
- 12B shows no chunk that meets the stall definition. Its chunks are about 2–2.7 s throughout, against about 0.5 s on E2B at
  16K. Why it does not stall is not explained, and a uniformly slow state cannot be told apart from a slower model with
  these records.

## 5. VRAM against the estimate

MiB. "Estimate" is `estimateSessionMemory`. "Sustained" is the 90th percentile of the samples above 100 MiB, and "peak" is
the largest sample, round 1 / round 2. Both include the 33 MiB idle use.

| Arm | Capacity | Estimate | Sustained | Peak |
| --- | --- | --- | --- | --- |
| E2B `i4` | 4,096 | 1,679 | 1,928 / 1,928 | 2,013 / 2,005 |
| E2B `i4` | 16,384 | 2,066 | 2,410 / 2,408 | 2,410 / 2,410 |
| E2B `i4` | 32,768 | 2,642 | 2,988 / 2,988 | 2,988 / 2,988 |
| E2B `i4` | 65,536 | 3,794 | 4,153 / 4,153 | 4,699 / 4,698 |
| E2B `i4` | 131,072 | 4,562 | 4,929 / 4,913 | 4,934 / 4,915 |
| E2B `bf16` | 4,096 | 4,668 | 4,736 / 4,745 | 4,867 / 4,869 |
| E2B `bf16` | 65,536 | 6,783 | 6,962 / 6,963 | 7,509 / 7,509 |
| E2B `bf16` | 131,072 | 7,551 | 7,721 / 7,721 | 7,725 / 7,725 |
| 12B `i4` | 8,192 | 8,475 | 8,615 / 8,617 | 8,693 / 8,694 |
| 12B `i4` | 16,384 | 9,115 | 9,247 / 9,247 | 9,249 / 9,249 |
| 12B `i4` | 32,768 | 10,395 | 10,529 / 10,529 | 10,529 / 10,532 |

- Sustained minus the estimate and the idle 33 MiB is stable from 8K up: about 310–330 MiB on E2B `i4`, about 130–180 MiB
  on `bf16`, about 100 MiB on 12B.
- Short peaks exceed that. At capacity 65,536 the peak is about 545 MiB over the sustained value on `i4`, `i4-fast` and
  `bf16`, and 427–545 MiB with speculation; on `bf16` it is only 3 samples of 1,814, one per chat. One run with the
  256 MiB row-block limit at 128K has a single sample 667 MiB over the sustained value, and the other round does not. The cause is not
  known, and 500 ms sampling can miss more.
- So a margin that covers the sustained excess is not enough for `capacity: "fit"`; the largest peak seen is about 1 GiB
  over the estimate.
- 12B at capacity 32,768 fits this 12 GB card at 10,532 MiB of whole-GPU use. 12B above 32K was not run here.

## 6. `attentionRowBlockMaxBytes`

E2B `i4`. The rows are bit-identical by design (ADR 0124), and the token shas match the unlimited arm at both lengths.

| P | Limit | Estimate (MiB) | Sustained (MiB) | Prefill as measured (s) | Decode busy (ms) | Decode wall (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 31,974 | none | 2,642 | 2,988 | 89.6–148.9 | 108.5 / 109.1 | 110.2 / 110.8 |
| 31,974 | 256 MiB | 2,130 | 2,478 | 88.4–150.1 | 109.3 / 108.3 | 111.4 / 109.5 |
| 31,974 | 128 MiB | 2,015 | 2,352 / 2,350 | 93.9–154.7 | 110.6 / 111.7 | 112.6 / 113.4 |
| 127,962 | none | 4,562 | 4,929 / 4,913 | 356.4–413.0 | 370.4 / 375.7 | 384.0 / 386.7 |
| 127,962 | 256 MiB | 3,282 | 3,647 / 3,630 | 471.3–511.0 | 378.3 / 377.1 | 384.8 / 383.5 |
| 127,962 | 128 MiB | 3,167 | 3,508 | 785.1–799.9 | 379.5 / 377.0 | 394.6 / 394.7 |

- At 32K the prefill cost is within the slow-mode spread, and the memory saved is 510–630 MiB.
- At 128K the limit costs about ×1.3 (256 MiB) and ×2 (128 MiB) in prefill, and at most 3% in decode.
- 256 MiB gives almost all of the saving of 128 MiB at this length (1,280 vs 1,395 MiB in the estimate) for much less time.

## 7. Speculative decode at long positions

E2B `i4`, `speculative: "always"`, against the plain `i4` arm. Tokens arrive in batches here, so the median per token is
not usable and the table uses the mean over the decode. Mean per token in ms, last chat, round 1 / round 2.

| P | Plain busy | Speculative busy | Plain wall | Speculative wall |
| --- | --- | --- | --- | --- |
| 15,990 | 62.7 / 63.3 | 37.2 / 39.3 | 74.1 / 77.0 | 45.0 / 48.5 |
| 63,971 | 194.7 / 193.6 | 100.7 / 97.2 | 220.6 / 224.3 | 118.9 / 117.0 |

The drafter adds 47 MiB to the estimate. 34 of 82–83 drafted tokens were accepted on this prompt, with the same token sha
as the plain arm. Speculation on `i4-fast` at long positions was not measured.

## 8. What this does not cover

- Quality at long context: no KL against torch and no retrieval test yet.
- Chrome and the 5070 Ti: sequence creation at 32K / 128K in the browser is still a manual step.
- E4B, and 12B above 32K.
- A sequence that fills its capacity: these runs stop on `maxNewTokens`.
