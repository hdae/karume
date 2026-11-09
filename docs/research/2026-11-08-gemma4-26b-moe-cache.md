> This document is a point-in-time snapshot (2026-11-08 to 2026-11-09) of the pre-research into an expert cache for Gemma 4
> 26B A4B: how many experts per layer a fixed-size cache would hold, how often decode would miss, what an exact cache (output
> bit-identical to full residency) costs in re-runs, and how far two approximate modes move the output. Every number comes from
> a CPU reference of all 30 layers on the maintainer's machine; nothing here ran on a GPU or in a browser. It is not a design
> source of truth. The harness and the raw records are local and untracked (`outputs/diag/moe-cache-2026-11-08/`).

# Gemma 4 26B A4B: an expert cache

## 1. Summary

- The knob studied is K, the number of experts held per layer (the same K for every layer). VRAM falls by about 0.1 GB per
  expert slot removed from all 30 layers: about 14.1 GiB with all 128 resident, 8.1 GiB at K = 64, 5.1 GiB at K = 32 (§2).
- Decode routing has strong but not tight locality. With LRU (least recently used) eviction the decode hit rate is 0.82 at
  K = 32, 0.96 at K = 64 and 0.99 at K = 96; LFU with aging adds up to 4 points at small K, and Belady's optimum (which needs
  the future) is the ceiling at 0.90 / 0.98 / 0.99. A fixed per-layer top-K set fitted on other prompts is far worse (0.45 /
  0.67 / 0.84), so the working set moves with the text (§3).
- Hit rate per pick is high, but a token touches 240 picks, so most tokens have at least one miss: P(token has a miss) is 0.96
  at K = 64 and 0.65 at K = 96 with LRU (§3).
- An exact cache that re-runs the whole token after any miss was simulated in the loop on all 30 layers. Every accepted token
  routed exactly as full residency and its logits were bit-identical (SHA-256 equal on all 2,304 tokens). The cost is the
  number of forwards: a mean of 2.77 per token at K = 48, 2.41 at K = 64 and 1.90 at K = 96, with a worst case of 8 (§4).
- Resuming a retry at the first missed layer, instead of re-running the whole token, stays exact but saves only 17–29% of the
  retry layers (2.12 forwards per token at K = 64), because the first miss is usually in layer 0 or 1 (§4.1).
- Giving the early layers more slots at the same mean K moves the first miss later: 1.81 forwards per token at K = 64, about
  12% less assumed time (§4.2). Drafting ahead on the cache and verifying exactly (self-speculation) saves about as much, 13%,
  with half the round trips (§4.3). Both stay exact; the best measured point is still about 2.4 times the time of full
  residency under the assumed costs.
- Two approximate modes that never wait for a load were measured against the exact model. At K = 64 both keep a mean KL of
  about 0.02 nats and top-1 agreement of 0.97; at K = 48 it is about 0.08 nats and 0.93–0.94; at K = 32 about 0.22 nats and
  0.90. Biasing the router toward resident experts cuts the dropped router weight sharply but does not lower KL (§5).
- Prefill does not fit a small cache: a 64-row chunk touches 64–66 of the 128 experts per layer, so prefill streams experts
  rather than caching them (§3).

## 2. Setup and memory per K

- Model: the 0.19.0 product series of 26B A4B (30 layers, 128 experts per layer, top-8 routing, one always-resident dense GeGLU
  MLP of width 2112 per layer). The CPU reference dequantizes the i4 expert weights (`int4-sym-g-f16`, an f16 step per 32
  elements) to dense f32. Before the measurement it reproduced the product series' greedy tokens on two short prompts with no
  divergence.
- One expert is 3,345,408 B (gate_up 2,230,272 B, down 1,115,136 B). All experts take 12,846,366,720 B; the rest of the text
  weights about 1.40 GiB.
- VRAM(K) ≈ 14.1 GiB − 30 × (128 − K) × 3.35 MB:

| K | experts held | VRAM, weights |
|---:|---:|---:|
| 128 | all | 14.1 GiB |
| 96 | 75% | 11.1 GiB |
| 64 | 50% | 8.1 GiB |
| 48 | 38% | 6.6 GiB |
| 32 | 25% | 5.1 GiB |
| 16 | 13% | 3.6 GiB |

- Six neutral chat prompts: English prose, Japanese prose, Python, arithmetic, a two-turn chat and a summary of a long
  passage. Greedy decode up to 512 tokens; 688 prompt rows and 2,024 decode rows in all.
- M1 recorded the true routing (8 expert ids per layer per row) and replayed it through cache policies. M2 and M3 ran the
  model with the cache in the loop, teacher-forced on M1's tokens.

## 3. Locality of the routing (M1)

Each (row, layer) set of 8 picks is one access; misses are judged against the cache before the access, then loaded. A case
starts from an empty cache at its first prompt row, and decode runs on the cache that the prompt rows left.

Decode rows, pooled over the six cases:

| policy | K | hit rate | layers with a miss, per token | P(token has a miss) | loads per token | MB per token |
|---|---:|---:|---:|---:|---:|---:|
| LRU | 32 | 0.817 | 19.1 | 1.000 | 43.8 | 146.6 |
| LRU | 48 | 0.908 | 12.0 | 0.998 | 22.1 | 73.9 |
| LRU | 64 | 0.956 | 6.9 | 0.955 | 10.7 | 35.7 |
| LRU | 96 | 0.988 | 2.3 | 0.651 | 2.8 | 9.3 |
| LFU-aged | 32 | 0.842 | 18.0 | 1.000 | 37.8 | 126.5 |
| LFU-aged | 64 | 0.960 | 6.5 | 0.932 | 9.7 | 32.5 |
| Belady | 32 | 0.905 | 13.1 | 0.998 | 22.9 | 76.5 |
| Belady | 64 | 0.980 | 3.7 | 0.839 | 4.8 | 16.2 |
| static | 32 | 0.452 | 29.3 | 1.000 | 131.4 | 439.7 |
| static | 64 | 0.665 | 25.1 | 1.000 | 80.4 | 268.9 |

- LFU-aged counts every access and multiplies every count by 0.7 every 16 rows. Belady evicts the expert used farthest in the
  future. Static pins each layer's K most frequent experts over the other five cases' decode rows and streams every miss.
- Above K = 96 LRU, LFU-aged and Belady converge: the remaining misses are experts never seen before in the case.
- Prompt rows saturate near a 0.916 hit rate even at K = 112, because the first touch of each expert is a miss; a single
  64-row prefill chunk touches 64–66 experts per layer (measured on a 12-layer cut during the pre-research).
- Recall of a "previous n tokens" predictor is low: the union of the previous token's picks covers 0.46 of the next token's,
  the previous 16 tokens (about 38 experts) 0.87. Prefetching by recency alone does not remove the misses.

## 4. The exact cache: re-run the token after a miss (M2)

An attempt is one forward of the token on the committed KV. A pick that is not resident contributes 0 and is recorded as a
miss. An attempt with no miss is accepted and its KV committed. Otherwise every recorded miss is loaded (evicting by LRU),
the attempt's KV is discarded and the token runs again. A miss in layer l also corrupts the routing of every later layer, so
the first attempt's later misses are a guess, and a second retry can be needed. Three cases (English prose, Japanese prose,
Python), 256 tokens each, LRU:

| K | attempts per token, mean | p95 | max | loads per token | MB per token | loaded but unused |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | 2.77 | 4 | 8 | 25.0 | 83.6 | 3.9 |
| 64 | 2.41 | 4 | 7 | 12.7 | 42.6 | 1.6 |
| 96 | 1.90 | 3 | 6 | 3.9 | 13.0 | 0.3 |

- Every accepted attempt routed exactly as M1 (ids and order), and every accepted logits row was SHA-256 equal to M1's: the
  exact cache is exact.
- The cost is in forwards, not bytes. At K = 64, 42.6 MB per token is about 6 ms through a MAP_WRITE staging ring at the
  7 GB/s measured on Chromium Linux (an inference from bulk rates, see §6), but the token runs 2.41 times. Even at K = 96 the
  mean is 1.90 forwards, because two tokens in three still miss somewhere.
- Python is the worst case (3.0 attempts at K = 64, 27 loads per token); Japanese prose the best (2.1, 4.5 loads).

### 4.1 Resuming at the first missed layer (M2r)

The layers before an attempt's first miss saw no miss, and the loads after the attempt touch only the layers that missed, so
re-running those layers reproduces them. A retry can therefore resume at the previous attempt's first missed layer, from the
saved hidden state and the committed KV of the earlier layers. The M2 loop was run again recording, per attempt, the first
missed layer and the SHA-256 of every decoder layer's input. On every retry the inputs up to the resume layer were SHA-256
equal to the previous attempt's and no miss fell before it, and the attempt sequence equalled M2's on every token, so the
resume is exact. Forwards below count decoder layers run divided by 30; the lm_head and the cost differences between layers
are not weighted. Same three cases, 256 tokens each:

| K | retries per token | whole-token forwards | resumed forwards, mean | p95 | max | retry layers saved | resume layer, median |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2.31 | 3.31 | 2.88 | 3.90 | 5.30 | 19% | 1 |
| 48 | 1.77 | 2.77 | 2.46 | 3.45 | 5.03 | 17% | 1 |
| 64 | 1.41 | 2.41 | 2.12 | 2.97 | 4.30 | 20% | 1 |
| 96 | 0.90 | 1.90 | 1.64 | 2.43 | 3.70 | 29% | 5.5 |

- The saving is small because the early layers miss most: at K = 64 the decode hit rate of layer 0 is 0.92 and of layer 1
  0.84, against 0.93–0.98 for the later layers, so the first miss of a token is usually in layer 0 or 1.
- The number of retries, and so of host round trips, is unchanged by the resume.

### 4.2 A K that differs per layer

Per-layer LRU misses do not interact (a layer's cache sees only its own picks, and the exact routing is fixed by M1), so each
(layer, K) was replayed once over the M1 traces and an allocation was scored by stacking the layers. The score is an assumed
time per token: 15 ms per forward, 2 ms per fence and 0.48 ms per expert loaded (3.19 MiB at 7 GB/s), with full residency at
15 ms. Counting re-run layers alone is a degenerate objective: it starves the late layers to the minimum, which then miss on
every token. A local search in steps of 4 slots, keeping the mean equal to K, was fitted on arithmetic, two-turn and summary and
then run in the loop (M2r, same three cases as above). At K = 64 the fit gives layers 0–4 between 84 and 108 slots and the later
layers 40–72.

| mean K | allocation | resumed forwards | retries per token | loads per token | first missed layer, mean | assumed ms per token |
|---:|---|---:|---:|---:|---:|---:|
| 48 | uniform | 2.46 | 1.77 | 25.0 | 5.2 | 52.4 |
| 48 | per layer | 2.22 | 1.75 | 24.9 | 9.0 | 48.7 |
| 64 | uniform | 2.12 | 1.41 | 12.7 | 6.1 | 40.7 |
| 64 | per layer | 1.81 | 1.32 | 12.2 | 11.6 | 35.6 |
| 96 | uniform | 1.64 | 0.90 | 3.9 | 8.7 | 28.2 |
| 96 | per layer | 1.58 | 0.88 | 3.6 | 10.2 | 27.2 |

- The gain comes from moving the first miss later, not from fewer retries: at K = 64 the retries fall by 6% and the resumed
  forwards by 15%. The replay predicted the same relative gain (−12% of the assumed time at K = 64); its absolute times are
  lower because it counts one retry per missed token.
- Every token's logits stayed SHA-256 equal to full residency.

### 4.3 Self-speculation over the cache (M5)

Instead of waiting on a miss, the model drafts ahead with the cache as it is and pays for the misses in one verify. One cycle
from the committed exact state:

1. Draft `depth` tokens greedily with single-row forwards; a non-resident pick contributes 0 and nothing waits.
2. Fence 1: load every expert the drafts missed.
3. Verify the committed token and the drafts in one forward of depth + 1 rows. Attention is causal, so a row is exact when no
   row up to it misses; the cycle commits the exact predictions of the rows before the first row with a miss (among the rows
   whose inputs are exact, up to the first rejected draft).
4. Fence 2: load that row's misses.

The verify's routing is taken from M1 (the inputs of the rows that count are M1's tokens; a multi-row verify is assumed to
route as the single-row decode). Every committed row was replayed with full residency and was SHA-256 equal to M1. The same
assumed costs as §4.2 apply, with each draft forward and each verify forward at 15 ms. Uniform K, same three cases:

| K | depth | tokens per cycle | drafts accepted per cycle | forwards per token | fences per token | loads per token | assumed ms per token | M2r uniform, ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 4 | 2.24 | 3.78 | 2.23 | 0.89 | 24.6 | 46.9 | 52.4 |
| 64 | 2 | 1.60 | 1.98 | 1.87 | 1.25 | 12.3 | 36.4 | 40.7 |
| 64 | 4 | 2.69 | 3.89 | 1.85 | 0.74 | 12.7 | 35.3 | 40.7 |
| 64 | 6 | 3.66 | 5.80 | 1.90 | 0.55 | 12.7 | 35.6 | 40.7 |
| 96 | 4 | 3.46 | 3.88 | 1.44 | 0.58 | 3.9 | 24.6 | 28.2 |

- The drafts are almost always right: with misses contributing 0, a cycle at K = 64 and depth 4 accepts 3.89 of its 4 drafts
  (accepted per cycle is the mean over the three cases of the leading drafts equal to the exact tokens). What
  limits a cycle is the verify: the exact rows route differently from the drafts in the later layers, and the first row whose
  exact picks miss ends the commit.
- At K = 64 self-speculation saves 13% of the assumed time against the resume with uniform K, about as much as the per-layer K
  (35.3 against 35.6 ms); the two were not combined. Depth past 4 adds nothing.
- The round trips fall to 0.74 fences per token at depth 4 (1.41 retries per token for M2r), which matters more if the Chrome
  fence is far above the 2 ms assumed here.
- The assumed time flatters the verify: its depth + 1 rows read the union of their picks, 16–24 experts per layer at depth 4
  against 8 for one row, and that extra expert traffic is not weighted.

## 5. Approximate modes that never wait (M3)

The residency for each decode row is the LRU state after the previous token, so no token waits for a load; after the token the
cache is updated from the unbiased router's true top-8. Prefill is exact. The approximate model keeps its own KV (drift
accumulates) and is compared with the exact model on the same tokens. KL is KL(p_exact ‖ p_approx) over the vocabulary in nats.

- skip-renorm: a non-resident pick is dropped and the router weights are renormalized over the resident picks.
- prior λ: λ · (max − min of the row's router logits) is added to the resident experts' logits for the selection only (the
  cache prior of arXiv:2412.00099, Eq. 9); the weights stay the unbiased softmax renormalized over the selected 8, and a
  selected pick that is still not resident is dropped.

Pooled over the six cases (1,282 decode rows):

| K | mode | KL mean | KL p95 | top-1 agreement | dropped weight per layer | picks changed per token |
|---:|---|---:|---:|---:|---:|---:|
| 32 | skip-renorm | 0.223 | 1.28 | 0.895 | 0.184 | 0 |
| 32 | prior 0.5 | 0.218 | 1.05 | 0.898 | 0.019 | 45.1 |
| 48 | skip-renorm | 0.081 | 0.32 | 0.933 | 0.091 | 0 |
| 48 | prior 0.5 | 0.085 | 0.28 | 0.941 | 0.008 | 22.8 |
| 64 | skip-renorm | 0.022 | 0.08 | 0.966 | 0.044 | 0 |
| 64 | prior 0.25 | 0.021 | 0.07 | 0.969 | 0.018 | 9.5 |
| 64 | prior 0.5 | 0.019 | 0.06 | 0.970 | 0.003 | 11.3 |

- The bias trades dropped picks for substituted picks at about the same KL. It removes the layers where every pick was dropped
  (88 layer-rows at K = 32 with skip-renorm, 0 with the bias).
- The spread between cases is wide: at K = 32 the arithmetic case stays at 0.04 nats while English prose and the two-turn chat
  reach 0.34–0.40. At K = 64 every case is between 0.006 and 0.034.
- KL here includes the drift of the approximate model's own KV over up to 256 tokens; it is not a per-token figure.

## 6. Transfer and fence costs (not measured here)

These come from earlier measurements and source reading, collected during the pre-research. They are inputs for a Chrome
measurement, not results of this one.

- The fence floor of the 2026-09-01 PoC ([research 2026-08-31](2026-08-31-freetoken-moe-over-arraybuffer.md)), about 11 ms,
  came from Deno: deno_webgpu sleeps 10 ms after each blocking poll in `mapAsync` and `onSubmittedWorkDone` even when the GPU is
  done. Chrome's GPU process polls Dawn on a 2 ms timer (1 ms when busy, per Chromium's `command_buffer_stub.cc`), so its
  floor is expected near 1.5–3 ms. Node with Dawn busy-polls and stays under 1 ms.
- Upload: a reused MAP_WRITE | COPY_SRC staging ring with `copyBufferToBuffer` ran at 7.0–7.7 GB/s on Chromium Linux and
  5.3–6.1 GB/s on Chrome D3D12; `writeBuffer` at 2.0–2.2 and 4.2–5.2 GB/s. These are rates for 8–32 MiB blocks; one expert
  (3.19 MiB) is below the 4 MiB threshold at which Chromium's wire switches to shared memory, so its rate is unknown.

## 7. What the numbers say

- A cache at K = 64 halves the expert VRAM. The exact form of it costs about 2.4 forwards per decode token with whole-token
  re-runs; the approximate forms cost no extra forward and stay at about 0.02 nats of KL and 97% top-1 agreement.
- The exact form's cost is set by the re-runs, not by the transfer. A resume from the first missed layer cuts it only to about
  2.1 forwards at K = 64, because the early layers miss most.
- Below K = 48 neither form looks attractive: the exact cache needs about 3 forwards per token (about 2.5 with the resume) and
  the approximate modes pass 0.08 nats.
- A per-layer K and self-speculation each bring the exact cache at K = 64 to about 35 ms per token under the assumed costs,
  against 15 ms with every expert resident.
- Open: combining the per-layer K with self-speculation; the verify's extra expert traffic; and the Chrome fence and the
  3.19 MiB upload on the RTX 5070 Ti, which set the real weights of the assumed costs.
