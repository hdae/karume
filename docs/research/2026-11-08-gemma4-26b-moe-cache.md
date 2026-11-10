> This document is a point-in-time snapshot (2026-11-08 to 2026-11-10) of the pre-research into an expert cache for Gemma 4
> 26B A4B: how many experts per layer a fixed-size cache would hold, how often decode would miss, what an exact cache (output
> bit-identical to full residency) costs in re-runs, and how far two approximate modes move the output. Every number comes from
> a CPU reference of all 30 layers on the maintainer's machine, except §6.1 to §6.3 (transfer costs and chats), which were measured in
> Chrome on an RTX 5070 Ti. It is not a design source of truth. The harness and the raw records are local and untracked (`outputs/diag/moe-cache-2026-11-08/`).

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
  (35.3 against 35.6 ms). Depth past 4 adds nothing.
- Combining the two adds little: self-speculation over the per-layer K of §4.2 gives 34.5 ms at K = 64 and depth 4 (2.74
  tokens per cycle, 1.82 forwards and 12.1 loads per token). The per-layer K raises the tokens per cycle of the two prose cases
  (3.16 → 3.28, 3.32 → 3.61) but not of Python (2.00 → 1.95), where 130 of 131 cycles end on a verify miss.
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

## 6. Transfer and fence costs

These come from earlier measurements and source reading, collected during the pre-research; §6.1 measures them in Chrome.

- The fence floor of the 2026-09-01 PoC ([research 2026-08-31](2026-08-31-freetoken-moe-over-arraybuffer.md)), about 11 ms,
  came from Deno: deno_webgpu sleeps 10 ms after each blocking poll in `mapAsync` and `onSubmittedWorkDone` even when the GPU is
  done. Chrome's GPU process polls Dawn on a 2 ms timer (1 ms when busy, per Chromium's `command_buffer_stub.cc`), so its
  floor is expected near 1.5–3 ms. Node with Dawn busy-polls and stays under 1 ms.
- Upload: a reused MAP_WRITE | COPY_SRC staging ring with `copyBufferToBuffer` ran at 7.0–7.7 GB/s on Chromium Linux and
  5.3–6.1 GB/s on Chrome D3D12; `writeBuffer` at 2.0–2.2 and 4.2–5.2 GB/s. These are rates for 8–32 MiB blocks; one expert
  (3.19 MiB) is below the 4 MiB threshold at which Chromium's wire switches to shared memory, so its rate is unknown.

### 6.1 Measured in Chrome on the RTX 5070 Ti

The Playground's expert transfer bench (`/expert-transfer`) ran twice in Chrome 156 on Windows (D3D12, RTX 5070 Ti,
cross-origin isolated), 50 samples per case after 5 of warm-up. Times are host wall time from the first call to the resolved
promise; the two runs agree within 0.2 ms on every median below and on every other case except `writeBuffer` at 4 experts (6.6 and 5.8 ms).

| case | median ms | p95 ms |
|---|---:|---:|
| fence, empty submit | 0.95–1.00 | 1.9–2.0 |
| fence after a 4-byte copy | 2.50–2.59 | 3.2–3.7 |
| readback of one token's router picks (960 B, `mapAsync`) | 2.40–2.42 | 3.3–3.5 |
| 1 expert by the staging ring | 2.83–2.92 | 3.4 |
| 16 experts by the staging ring | 9.21–9.40 | 9.9–10.3 |
| 1 expert by `writeBuffer` | 1.46–1.48 | 4.0–4.2 |
| 16 experts by `writeBuffer` | 19.84–20.04 | 20.7–21.3 |

- A fence that waits for GPU work costs about 2.5 ms, near the 2 ms assumed; the empty submit resolves sooner, presumably
  without waiting for the GPU.
- The ring costs about one fence plus 0.43 ms per expert (the step from 8 to 16 experts, 7.8 GB/s); remapping its slots takes
  0.2 ms and is off the critical path. `writeBuffer` stays at 2.0–2.7 GB/s, so from 4 experts on it takes about twice the ring's
  time.
- With 2.5 ms per fence and 0.43 ms per expert in place of the assumed 2 ms and 0.48 ms, the K = 64 times of §4 change by less
  than 1 ms: about 40.8 ms for the resume with uniform K, 35.7 ms with the per-layer K, 35.0 ms for self-speculation and 34.3 ms
  for both. The forwards still set the cost.
- The resume's times count no readback on a token without a miss; an implementation needs one per token to learn whether
  it missed (about 2.4 ms here) unless it rides on the readback of the sampled token. Self-speculation's two fences per cycle
  already include it.

A third run (2026-11-09) added reads of experts from the cached 26B distribution in CacheStorage: each sample reads one
expert's four ranges (or eight experts' 32 ranges) at random 4-byte-aligned positions of the model's parts, all at once
through the hub's range reader (a `slice` of the cached Blob), then the same ranges again. 50 samples after 5 of warm-up:

| case | first read, median ms | p95 ms | second read, median ms |
|---|---:|---:|---:|
| 1 expert (4 ranges, 3.19 MiB) | 3.91 | 6.89 | 2.75 |
| 8 experts (32 ranges, 25.5 MiB) | 23.36 | 38.89 | 21.64 |

- A cached read costs about 2.9 ms per expert (1.15 GB/s), six times the 0.48 ms assumed for the whole load in §4. Reading
  eight at once scales linearly, so concurrent reads do not raise the rate.
- The second read of the same bytes is barely faster (2.7 ms per expert at 8), so the limit is the browser's Blob read
  path rather than the disk (an inference: the OS cache should serve the second read).
- With the read added to the ring's 0.43 ms, the 12.7 loads per token at K = 64 add about 36 ms: about 77 ms per token for
  the resume with uniform K and 72 ms for self-speculation (same assumed 15 ms forwards).
- LRU has the inclusion property, so a host tier of 32 more experts per layer (about 3.2 GB) in front of the Blob leaves the
  3.9 loads per token of K = 96 to the Blob: about 52 ms per token for the resume if a tier hit costs what the ring does. A
  tier holding every expert returns to the times of §4.

### 6.2 The first chat and a host tier in Chrome (2026-11-09)

The full model ran a chat with the exact cache (scheme A of ADR [0146](../decisions/0146-exact-expert-cache.md)) in Chrome
on the RTX 5070 Ti at 32 slots, every miss read from the cached Blob. One English prompt of 30 tokens, 239 tokens sampled at
temperature 1 (top-k 64, top-p 0.95), stopped by the end token:

| counter | value |
|---|---:|
| prefill | 8 chunks of 4 rows, 13.3 s |
| decode | 238 tokens, 62.9 s (264 ms per token) |
| committed runs / re-runs | 246 / 551 (3.24 forwards per committed run) |
| experts loaded | 15,662 (52.4 GB), 59.0 s (3.77 ms per expert) |
| everything else | 17.1 s over 797 runs (about 21 ms per run, about 70 ms per committed run) |

- The Blob reads take 77% of the time. At 64 slots the maintainer saw it only a little faster (no record kept).
- Even with free loads a committed run would cost about 70 ms at 32 slots; the rest is the re-runs.

A host tier was then measured with the same page (`/expert-transfer`, record
`expert-transfer_2026-11-09T19-38-13.json`): 13 GiB held in each of two forms, then hits of 1–32 distinct experts at random
positions of the tier copied into a 32-expert VRAM pool, 25 samples after 5 of warm-up, wall time to `onSubmittedWorkDone`:

| form | allocate and fill 13 GiB | ms per expert at 32 experts (median − fence) / 32 | slope over 1–32 experts |
|---|---:|---:|---:|
| A: `ArrayBuffer`s, staging ring | 1.4 s | 0.52 | 0.47 |
| A: `ArrayBuffer`s, `writeBuffer` | (same) | 1.12 | 1.12 |
| B: `MAP_WRITE \| COPY_SRC` buffers filled at creation, `copyBufferToBuffer` | 8.4 s | 0.064 | 0.040 |

- Both forms held all 13 GiB (26B's experts are 12.85 GB) without an allocation failure.
- B's copy runs from system memory at about 50 GB/s; it is the fastest hit by a factor of eight.
- B's memory was not returned by `destroy()` until the page collected garbage, and the system's memory graph rose by about
  twice B's size while it was held: Chrome appears to keep a renderer-side copy of a mappable buffer (an inference from
  Task Manager, not a per-process measurement). A's arrays were returned at once.
- At 32 slots the chat's 15,662 loads would cost about 8 s through A's ring and about 1 s through B, against 59 s from the
  Blob: about 25 s and 18 s for the whole chat.

### 6.3 The chat with the host tier in Chrome (2026-11-09)

The same chat as §6.2 (one English prompt of 30 tokens, 239 tokens sampled at temperature 1, top-k 64, top-p 0.95, seed 0)
ran in Chrome on the RTX 5070 Ti with the host tier of ADR 0146's addendum of 2026-11-10, at several slot counts and two tier
budgets, and once with every expert resident. Records `gemma4_2026-11-09T21-47-57.json` to `…T22-03-55.json`:

| slots | host tier | build | prefill | decode tokens per s | forwards per committed run | experts loaded (from RAM / container) | load time | upload per expert | other time per forward |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | every expert | 16.6 s | 5.0 s (15 chunks) | 6.1 | 3.87 | 34,021 (34,021 / 0) | 23.3 s | 0.68 ms | 20.8 ms |
| 32 | every expert | 16.0 s | 3.4 s (8 chunks) | 9.7 | 3.24 | 15,662 (15,662 / 0) | 11.1 s | 0.71 ms | 21.2 ms |
| 64 | every expert | 16.5 s | 3.0 s (4 chunks) | 18.1 | 2.32 | 4,583 (4,583 / 0) | 3.6 s | 0.79 ms | 22.3 ms |
| 96 | every expert | 16.7 s | 2.5 s (4 chunks) | 25.6 | 1.84 | 3,104 (3,104 / 0) | 2.5 s | 0.81 ms | 20.8 ms |
| 64 | 8 GiB | 9.6 s | 8.7 s (4 chunks) | 12.4 | 2.32 | 4,583 (2,650 / 1,933) | 15.8 s | 0.67 ms | 21.6 ms |
| 128 (no cache) | — | 7.5 s | 0.4 s (1 chunk) | 52.3 | 1 | — | — | — | 19.1 ms per token |

"Other time per forward" is the turn's time less the load time, over committed runs plus re-runs.

- Every record sampled the same 239 tokens, the run with every expert resident and §6.2's run without a tier included: the
  output is bit-identical in Chrome too.
- A tier of every expert held 3,840 experts (30 layers × 128, 12.85 GB) and read nothing from the Blob; at 32 slots decode
  went from 3.8 to 9.7 tokens per second (§6.2 estimated about 25 s for the chat; it took 28.0 s). The build is about 7 s
  longer, the time to copy the experts into RAM.
- With an 8 GiB tier, 2,567 experts were held (just under the budget) and 1,933 misses were read from the Blob at about
  6.55 ms each, against 3.77 ms in §6.2. A likely cause (not measured): a load now reads its misses one ring round of up to 32
  at a time, where scheme A read every miss of a load at once.
- Uploads cost 0.67–0.81 ms per expert, above the 0.52 ms of the bench in §6.2; a decode load moves about 14 experts at 64
  slots, so a round's fixed cost is shared by fewer.
- With the transfer out of the way, the time follows the forwards: about 21 ms each at every slot count, close to the 19 ms of
  a resident decode token. Fewer re-runs (self-speculation, a per-layer K) are what is left to gain.

### 6.4 Self-speculation in Chrome (2026-11-10)

The maintainer ran self-speculation (ADR 0146 addendum "scheme B as built") in Chrome on the RTX 5070 Ti at 64 slots with
every expert in the host tier and depth 4, on a three-turn chat (temperature 1, top-k 64, top-p 0.95, seed 0) and then a
long counting turn stopped by hand. Records `gemma4_2026-11-10T00-56-25.json` and `…T00-58-19.json`; the second record's
report is cumulative, so its row is the difference between the two.

| run | decode tokens | decode tokens per s | tokens per cycle | accepted drafts per cycle | runs per decode token | cycles ended at the first missed row | verify retries |
|---|---:|---:|---:|---:|---:|---:|---:|
| three turns, depth 4 | 357 | 20.8 | 3.10 | 3.02 | 1.61 | 102 of 115 | 5 |
| the same three turns in another order, no depth (`…T22-20-04.json`) | 360 | 17.4 | — | — | — | — | — |
| the same three turns, every expert resident (`gemma4_2026-11-08T14-29-54.json`) | 357 | 49.4 | — | — | — | — | — |
| counting turn, depth 4 | 917 | 17.1 | 2.96 | 2.83 | 1.79 | 270 of 310 | 91 |

"Runs per decode token" is draft runs plus verifies plus verify retries over the decode tokens.

- The three turns sampled the same tokens as the resident run of 2026-11-08 with the same prompts in the same order, at
  temperature 1: the output stays bit-identical in Chrome with self-speculation.
- On the full model a cycle confirmed about three drafts of four at temperature 1, far more than on the cut model (1.43
  tokens per cycle in the cut e2e): the cut model's logits are no guide to the acceptance.
- Runs per decode token went from about 2.3 (scheme A at 64 slots, §6.3) to 1.6–1.8, but decode only from 17.4 to 20.8
  tokens per second. A likely reading (not measured): each draft is a whole forward of its own, and the verify runs 8
  physical rows with up to five rows' experts loaded first, so a run costs more than a scheme-A decode run.
- Most cycles ended at the verify's first missed row (102 of 115) with every row before it accepted. A likely cause (not
  measured): the drafts route on hidden states that missed experts, so the verify's rows pick experts the drafts' load did
  not bring in.

### 6.5 Per-layer slots in Chrome (2026-11-10)

The maintainer compared 64 slots in every layer with the fitted allocation of §4.2 (`fitted` in the Playground: a mean of
64, from 40 to 108 per layer), each without and with depth 4, in Chrome on the RTX 5070 Ti with every expert in the host
tier, on the same three turns in the same order (temperature 1, top-k 64, top-p 0.95, seed 0). Records
`gemma4_2026-11-10T06-57-09.json` to `…T07-00-47.json`.

| slots, depth | decode tokens per s (360 tokens) | tokens per cycle | runs per decode token | cycles ended at the first missed row |
|---|---:|---:|---:|---:|
| 64, none | 17.5 | — | — | — |
| 64, 4 | 20.5 | 3.03 | 1.64 | 107 of 119 |
| fitted, none | 19.0 | — | — | — |
| fitted, 4 | 22.6 | 3.21 | 1.54 | 94 of 112 |

- The four runs sampled the same tokens in every turn.
- The fitted allocation gained about 9% without a depth and about 10% with depth 4, at the same expert VRAM. The CPU
  estimate of §4.2 predicted about 12% for scheme A, and only about 2% on top of self-speculation; with self-speculation
  the gain is larger than predicted because fewer cycles end at their first missed row (the early layers, which miss
  most, hold more experts).

### 6.6 The cost of one run under self-speculation (2026-11-10)

The full 26B ran on the RTX 3080 Ti (Node and Dawn) at 64 slots with every expert in the host tier, with scheme A and
self-speculation at depth 4 alternated on one Session (three prompts of 96 tokens, temperature 0 and 1; the same tokens
in every arm). The GPU ran at 88–91 °C with thermal slowdown in every arm, so the comparison is the ratio within a run.
Scripts and logs: `outputs/diag/expert-run-cost-2026-11-10/`.

| part of a cycle (temperature 0) | ms per decode token | share |
|---|---:|---:|
| drafts (3.8 runs of one row per cycle, median 23.5 ms each) | 32.5 | 59% |
| verify (8 physical rows, median 45 ms) | 16.8 | 30% |
| loads (the drafts' merged load and the load after a cut) | 3.2 | 6% |
| verify retries | 1.75 | 3% |
| sampler (`peek` and acceptance over the 262k vocabulary) | 1.05 | 2% |
| input preparation, `touch`, `commit` | 0.1 | 0% |
| total | 55.4 (18.1 tokens per s) | |

Scheme A on the same Session took 64.4 ms per decode token (15.5 tokens per s), so the ratio 0.86 matches Chrome's 0.84
(§6.4).

- **The verify's shape is the main cost.** Depth 4 verifies 5 rows, but the default buckets under the cache are `[4]`
  below the chunk rows 8, so it ran 8 physical rows: about 1.9 times a one-row run. Most of the gap is the pooled
  `expert_linear` (3.0 → 14.2 ms of GPU time), which reads each row's picks, pad rows included; reading back 5 rows of
  logits instead of 1 costs only 1–5 ms.
- **A draft is a whole forward**, as long as a scheme-A decode run; about 0.9 drafts per cycle are not accepted.
- **The host side is small**: the sampler takes 0.4–0.6 ms per draw, and input preparation and the bookkeeping under
  0.1 ms per token. The loads are 6% here and likely more in Chrome (its fence and upload costs, not measured).
- **Running the verify at 5 physical rows** (a bucket `[4, 5]`) took 48.2 ms per decode token (20.8 tokens per s, ratio
  0.76 against scheme A) with the same tokens and the same cycle statistics as at 8 rows. Depth 3 (4 rows) at 8-row
  buckets came to a ratio of 0.81, depth 2 to 0.82–0.85.

## 7. What the numbers say

- A cache at K = 64 halves the expert VRAM. The exact form of it costs about 2.4 forwards per decode token with whole-token
  re-runs; the approximate forms cost no extra forward and stay at about 0.02 nats of KL and 97% top-1 agreement.
- The exact form's cost is set by the re-runs, not by the transfer. A resume from the first missed layer cuts it only to about
  2.1 forwards at K = 64, because the early layers miss most.
- Below K = 48 neither form looks attractive: the exact cache needs about 3 forwards per token (about 2.5 with the resume) and
  the approximate modes pass 0.08 nats.
- A per-layer K and self-speculation each bring the exact cache at K = 64 to about 35 ms per token, and together to about
  34 ms, against 15 ms with every expert resident; Chrome's measured fence and upload costs leave these within 1 ms.
- Reading the misses from the cached Blob in Chrome (about 2.9 ms per expert) roughly doubles those times; a host tier in
  RAM is what brings them back. In the chat of §6.3 a tier of every expert made 64 slots decode at 18.1 tokens per second
  (52.3 with every expert resident), about 21 ms per forward.
- Self-speculation at depth 4 (§6.4) confirmed about three tokens per cycle on the full model and brought the runs per
  decode token from about 2.3 to 1.6, but decode only from 17.4 to 20.8 tokens per second: the cost per run is now what
  is left.
- The fitted per-layer allocation (§6.5) adds about 10% on top in Chrome (22.6 tokens per second at depth 4). Of the cost
  per run (§6.6), the drafts' whole forwards and the verify's padding to 8 rows are most of it; the host side is not.
- Open: the verify's extra expert traffic, the real per-forward time of the 26B decode in place of the assumed 15 ms, and
  whether a GPU implementation reaches the CPU reference's draft acceptance.
