> This document is a point-in-time snapshot (measurements of 2026-11-18, RTX 3080 Ti, Node 24 + Dawn / Vulkan) of the
> host side of a Gemma 4 decode token: where the part of a token that is not kernel time goes, and what the changes
> made from it are worth. It is not a design source of truth: what the system does is in
> [spec/generation](../spec/generation.md) and [spec/workbench](../spec/workbench.md), the candidates and their status
> in the [perf-ledger](../perf-ledger.md). The scripts and raw outputs are under
> `outputs/diag/decode-optimization-2026-11-18/decode-host/` and the bench records under `outputs/workbench/` (both
> local, untracked). No browser was measured; section 9 says which segments are Node's own.

# The host side of a decode token

## 1. What was asked

On the fast presets a decode token is short — QAT E2B `i4-fast` runs at 6 to 7 ms per token on this machine — so a
few hundred microseconds of host work per token are several percent. The questions: how a token's wall time divides
between the GPU's kernels and the host, which of the host items filed in the perf-ledger (H-31, H-32, H-10, H-9, K-49,
and the opt-ins H-27 and H-28) are worth building, and what they give end to end with the generated token ids
unchanged.

## 2. Reading

- **A quarter of a token was not kernel time.** QAT E2B `i4-fast`, greedy, with a signal as a product caller passes
  one: 8.3 to 8.5 ms per token on the wall clock, 5.9 to 6.2 ms of it GPU time.
- **The largest item was the yield to the event loop before every decode run** (H-31): 1.17 to 1.20 ms per token on
  E2B and 1.05 to 1.24 on E4B by the difference between chats with a signal and without; on 12B the timer's segment is
  the same 1.07 ms per token. Under Node a `setTimeout(0)` is a 1 ms timer. Taking the yield while the run is awaited
  removes it: a token with a signal now takes what one without a signal takes.
- **A sampled token paid 0.41 to 0.51 ms more on the host** (H-32): 0.09 to 0.12 ms to read the 1 MiB logits row back
  and 0.31 to 0.38 ms to scan it for the top-k. Selecting the candidates on the GPU and reading 16 KiB back takes 0.24
  to 0.32 ms off a sampled E2B token, about 4 % of it, with the same token ids.
- **One `topk` of 65 over the row is not the way to move that to the GPU**: it takes 19 ms. A `topk` of 8 over 256 parts
  of the row, taken by residue of the id, takes 0.066 ms, and the host merges the parts.
- **The wait for the first submit is real, and a ramp does not recover it** (H-29): for 0.42 to 0.47 ms of a greedy E2B
  token (0.48 to 0.62 sampled, 0.57 to 0.71 on E4B, 0.56 to 0.72 on 12B) the GPU has none of the token's work, because
  the host is encoding the first chunk of 128 dispatches. A first chunk of 16 or 32 dispatches that doubles measured
  0.12 to 0.18 ms per token slower than no ramp, so it was taken out again. From the first submit on, the GPU is the
  slower side: encoding the remaining chunks (2.7 to 4.8 ms of host time) overlaps it entirely.
- **Everything else on the host is under 1 % of a token**: the derived inputs (0.06 to 0.09 ms with the per-layer
  embeddings on the host), the greedy selection's readback and the token's way to the consumer (0.03 to 0.06 ms),
  preparing the run and its 8 queue writes (inside the wait for the first submit, of which the first chunk's encoding
  is nearly all).
- **The opt-ins**: the decode run-ahead at depth 8 gives ×0.92 on this machine (greedy, E2B); the per-layer embeddings
  on the GPU alone are neutral.
- **End to end** (QAT E2B `i4-fast`, the distribution's sampler, a signal, one interleaved run): what a token usually
  takes went from 7.88 to 6.52–6.55 ms (×0.83), the median of the chats' means from 8.88–8.96 to 7.47–7.53 ms per
  token (112–113 to 133–134 tokens/s), and every token id is the same.

## 3. Method

- The workbench's decode bench (`gemma4-bench`, [spec](../spec/workbench.md)): arms visited in A B B A rounds inside one
  process, a pipeline built per visit, 3 chats per prompt and visit of which the first is left out, 128 new tokens,
  two fixed prompts (`few-ids`, `many-ids`). Every table below gives the number of warm chats per arm and prompt.
- Three views of the same pipeline, as arms of one run: the tap (GPU time from pass-boundary timestamps in the
  product's own pass structure), tap off (the wall clock), and the tap's host mode, added for this work — wall-clock
  marks at the WebGPU calls and at the checks of the chat's signal, which cut every token's interval into seven
  segments that add up to it (`between`, `derive`, `start`, `encode`, `wait`, `read`, `draw`;
  [spec](../spec/workbench.md#the-decode-benchs-host-segments)).
- A chat's figure is its mean decode time per token; a host arm also has the median over the chat's tokens of every
  segment. Tables give the median over the warm chats and, where it is shown, the least.
- This card runs 7 to 12 % slower after sustained load than after an idle period, and single kernels run many times
  slower for moments. The runs below followed other work on the same GPU. Arms are compared only inside one run; a
  change that cannot be switched inside one process (H-31) is compared as each build's difference from an arm that
  does not have the item at all (no signal), measured in the same run as itself.
- Code under measurement: frozen copies of single commits of the branch (the files of the commit, outside the
  repository), so a queued run measured the commit it was queued for. `base` is commit `8417ae1e`: main `7834560b`
  plus the bench's host mode, no product change.
- The commits this record names are the branch's as it stands on main `1ceaaf66`. The measurements were taken on
  the same changes before the branch was moved there, on main `7834560b`; nothing was measured again after the
  move.

## 4. The breakdown before any change

Commit `8417ae1e`. E2B: 3 rounds, 6 warm chats per arm and prompt. E4B: 2 rounds, 4 warm chats. 12B: 2 rounds, one
prompt (`many-ids`), 4 warm chats.

Wall clock and GPU time, ms per decode token (median of the warm chats; the least in parentheses):

| Model, sampler | With a signal, wall | Without a signal, wall | GPU time (with a signal) |
| --- | --- | --- | --- |
| E2B `i4-fast`, greedy, `few-ids` | 8.277 (7.736) | 7.106 (6.581) | 6.247 (5.764) |
| E2B `i4-fast`, greedy, `many-ids` | 8.522 (7.636) | 7.319 (6.765) | 6.016 (5.761) |
| E2B `i4-fast`, distribution's sampler, `few-ids` | 8.766 (8.028) | 7.697 (7.213) | 6.047 (5.694) |
| E2B `i4-fast`, distribution's sampler, `many-ids` | 8.668 (8.233) | 7.503 (7.150) | 5.942 (5.670) |
| E4B `i4-fast`, greedy, `few-ids` | 12.881 (11.910) | 11.645 (10.735) | 9.976 (9.664) |
| E4B `i4-fast`, greedy, `many-ids` | 11.960 (11.750) | 10.908 (10.758) | 9.659 (9.482) |
| 12B `i4-fast`, greedy, `many-ids` | 27.802 (23.646) | 24.799 (21.952) | 20.546 (19.918) |

GPU time without a signal on E2B, greedy: 5.955 (5.738) and 5.931 (5.772). With the yield the GPU's own time is 0.1 to
0.3 ms longer at the median and the same at its least.

The host's segments, ms per decode token: the median over the warm chats of the chat's median over its tokens.

| Model, sampler, signal | between | derive | start | encode | wait | read | draw | sum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2B, greedy, signal (`few-ids`) | 1.074 | 0.057 | 0.474 | 2.702 | 3.134 | 0.000 | 0.025 | 7.465 |
| E2B, greedy, signal (`many-ids`) | 1.074 | 0.057 | 0.474 | 2.704 | 3.133 | 0.000 | 0.025 | 7.467 |
| E2B, greedy, no signal (`few-ids`) | 0.112 | — | 0.418 | 2.694 | 3.140 | 0.000 | 0.024 | 6.387 |
| E2B, greedy, no signal (`many-ids`) | 0.114 | — | 0.426 | 2.735 | 3.112 | 0.000 | 0.025 | 6.413 |
| E2B, sampled, signal (`few-ids`) | 1.078 | 0.064 | 0.539 | 3.431 | 2.495 | 0.096 | 0.309 | 8.012 |
| E2B, sampled, signal (`many-ids`) | 1.086 | 0.072 | 0.617 | 3.666 | 2.367 | 0.101 | 0.318 | 8.227 |
| E2B, sampled, no signal (`few-ids`) | 0.149 | — | 0.516 | 3.524 | 2.406 | 0.099 | 0.320 | 7.014 |
| E2B, sampled, no signal (`many-ids`) | 0.125 | — | 0.481 | 3.307 | 2.574 | 0.091 | 0.310 | 6.889 |
| E4B, greedy, signal (`few-ids`) | 1.091 | 0.087 | 0.662 | 4.221 | 5.468 | 0.000 | 0.053 | 11.583 |
| E4B, greedy, signal (`many-ids`) | 1.092 | 0.090 | 0.674 | 4.362 | 5.414 | 0.000 | 0.055 | 11.687 |
| E4B, greedy, no signal (`few-ids`) | 0.196 | — | 0.573 | 4.270 | 5.479 | 0.000 | 0.057 | 10.576 |
| E4B, greedy, no signal (`many-ids`) | 0.194 | — | 0.566 | 4.250 | 5.645 | 0.000 | 0.057 | 10.713 |
| E4B, sampled, signal (`few-ids`) | 1.095 | 0.089 | 0.707 | 4.833 | 4.894 | 0.114 | 0.343 | 12.076 |
| E4B, sampled, signal (`many-ids`) | 1.088 | 0.086 | 0.643 | 4.613 | 5.238 | 0.102 | 0.335 | 12.105 |
| 12B, greedy, signal | 1.075 | 0.024 | 0.555 | 3.804 | 17.514 | 0.000 | 0.030 | 23.002 |
| 12B, greedy, no signal | 0.140 | — | 0.578 | 4.444 | 17.423 | 0.000 | 0.049 | 22.634 |
| 12B, sampled, signal | 1.092 | 0.030 | 0.724 | 4.733 | 15.971 | 0.124 | 0.381 | 23.056 |

Without a signal the loop makes no check, so `derive` is inside `between`. Per token: 9 submits (greedy) or 7 (sampled)
on E2B, 11 or 9 on E4B and 12B; 8 queue writes (9 on 12B); one map of 8 bytes (greedy) or two maps of 1,054,720 to
1,063,936 bytes together (sampled: the logits row and the hidden row).

How to read it:

- `encode` + `wait` is the GPU's time for the token within a few percent (E2B greedy: 5.84 against 5.93 to 6.02), so
  the GPU works from the first submit to the end, and the host's encoding of the later chunks costs the token nothing.
- `between` with a signal is the timer: 1.07 to 1.09 ms on every model.
- `start` is the stretch in which the GPU has nothing of the token: the run's preparation, its queue writes and the
  first chunk of passes, up to the return of the first submit. The later chunks encode at 3.5 µs per dispatch on E2B
  (2.70 ms for 768), which puts the first chunk's 128 dispatches at about 0.45 ms of the 0.42 to 0.47: the preparation
  and the writes are a small part of it.
- `read` + `draw` of a sampled token is 0.405 to 0.419 ms on E2B, 0.437 to 0.457 on E4B and 0.505 on 12B (its
  suppressed tokens make the sampler copy the row before masking it).

## 5. The per-token yield (H-31)

A generation with a signal waited for a `setTimeout(0)` before every decode run. The decode loops now arm that timer
at one boundary and, at the next, check at once when it was delivered while the run was awaited
([spec](../spec/generation.md#abort-checks)); commit `f96328cb`.

The same six arms as section 4's E2B greedy run, on that commit (3 rounds, 6 warm chats per arm and prompt), ms per
decode token:

| Arm | Before, `few-ids` | Before, `many-ids` | After, `few-ids` | After, `many-ids` |
| --- | --- | --- | --- | --- |
| Signal, wall: median (least) | 8.277 (7.736) | 8.522 (7.636) | 7.604 (6.694) | 7.520 (6.784) |
| No signal, wall: median (least) | 7.106 (6.581) | 7.319 (6.765) | 7.402 (6.802) | 7.163 (6.824) |
| Signal − no signal, medians | +1.171 | +1.203 | +0.202 | +0.357 |
| Signal − no signal, least | +1.155 | +0.871 | −0.108 | −0.040 |
| Signal, the segments' per-token medians, sum | 7.465 | 7.467 | 6.544 | 6.486 |
| No signal, the segments' per-token medians, sum | 6.387 | 6.413 | 6.539 | 6.532 |
| Signal, `between` (per-token median) | 1.074 | 1.074 | 0.013 | 0.011 |
| Signal, fastest chat, tokens/s | 129.3 | 131.0 | 149.4 | 147.4 |

- The timer wait is gone: `between` with a signal is 0.011 to 0.013 ms, so the timer armed at one boundary is delivered
  during the run on Dawn under Node and the next boundary does not wait.
- By what a token usually takes (the per-token medians), a token with a signal went from 7.47 to 6.49–6.54 ms and now
  equals one without a signal (6.53–6.54 in the same run): −0.93 to −0.98 ms, ×0.87.
- The medians of the chats' means scatter by up to 0.4 ms between arms that run the same thing in this run (heat), which
  is why the bench now also reports a chat's median interval between two tokens.
- The generated token ids are the same before and after (the sha256 of the ids per prompt is equal across the two
  records: `1ad389c3…`, `77cf9302…`).
- E4B and 12B were measured before only: their timer wait is the same 1.07 to 1.09 ms per token, 9 to 10 % of an E4B
  token and about 4 % of a 12B token.

## 6. The GPU selection for a sampled token (H-32)

### 6.1 What the `topk` kernel costs

One row of 262,144 logits, timing mode (one dispatch per pass, timestamp queries), 40 runs after 5; rows of Gaussian
values with a few dozen high entries, and one row ascending with the id (the worst case for a lane's insertion).
`maxComputeWorkgroupStorageSize` is 49,152 on this device. GPU ms per row, median:

| Form | Gaussian row | Ascending row |
| --- | --- | --- |
| `topk` k = 1 (the greedy selection's two dispatches) | 0.017 | 0.017 |
| One `topk` over the row, k = 2 | 2.18 | 3.08 |
| One `topk` over the row, k = 8 | 2.44 | 9.81 |
| One `topk` over the row, k = 32 | 7.92 | 64.5 |
| One `topk` over the row, k = 65 | 19.3 | 57.0 |
| One `topk` over the row, k = 127 | 51.1 | 109.9 |
| `topk` of 65 per part, then `topk` of 65 over the parts' candidates: 32 / 64 / 128 parts | 2.31 / 1.85 / 2.02 | 2.33 / 1.90 / 2.36 |
| `topk` of 32 per part (consecutive ids), merged on the host: 32 / 64 / 128 parts | 1.11 / 0.68 / 0.42 | 2.05 / 1.04 / 0.54 |
| `topk` of 8 per part (consecutive ids), merged on the host: 64 / 128 / 256 parts | 0.136 / 0.081 / 0.049 | 0.165 / 0.088 / 0.050 |

One workgroup of 32 lanes scans a whole row and keeps a sorted list per lane in workgroup storage, so the time grows
steeply with the entries kept and falls with the number of rows the work is spread over. A `topk` of the 65 entries the
distributions' sampler needs costs three decode tokens as one dispatch, and at WebGPU's default workgroup storage
(16,384 bytes) the kernel admits k ≤ 63 at all. Only a small `topk` over many parts is cheaper than the host's own
scan.

Parts taken by residue of the id (part p holds the ids ≡ p mod G: a `reshape` and a `permute` before the `topk`), merged
on the host to the row's first 65; a third row has 40 leading entries at consecutive ids. GPU ms per row, median, and
the rows of 45 the merge could not decide:

| Parts × entries per part | GPU, Gaussian | GPU, cluster | Undecided: Gaussian / cluster / ascending |
| --- | --- | --- | --- |
| 256 × 8, consecutive ids | 0.055 | 0.056 | 0 / 45 / 45 |
| 128 × 8, by residue | 0.102 | 0.102 | 0 / 0 / 0 |
| 256 × 8, by residue | 0.066 | 0.066 | 0 / 0 / 0 |
| 512 × 8, by residue | 0.047 | 0.047 | 0 / 0 / 0 |
| 1024 × 8, by residue | 0.042 | 0.042 | 0 / 0 / 0 |
| 256 × 4, by residue | 0.043 | 0.043 | 11 / 0 / 0 |
| 512 × 4, by residue | 0.032 | 0.032 | 0 / 0 / 0 |
| 256 × 12, by residue | 0.084 | 0.084 | 0 / 0 / 0 |
| 256 × 16, by residue | 0.128 | 0.128 | 0 / 0 / 0 |
| 512 × 16, by residue | 0.103 | 0.103 | 0 / 0 / 0 |

- The `permute` is 0.010 ms of each by-residue row.
- With parts of consecutive ids a run of neighbouring ids at the top exhausts one part (every row of the cluster and
  of the ascending row is undecided); by residue none is.
- 4 entries per part are too few for 65 leading entries over 256 parts (a quarter of the Gaussian rows undecided).
- The form built is 256 × 8 by residue: 0.066 ms, 16 KiB read back. 512 × 8 is 0.019 ms less on the GPU for twice the
  readback and twice the parts to merge.

### 6.2 What the host's draw costs

`createSampler(...).next` on rows of 262,144 logits, no GPU, median of 15 batches of 100 calls (the machine was running
a GPU bench at the same time):

| Sampler | ms per call |
| --- | --- |
| Temperature 0 (the host's argmax) | 0.462 |
| Temperature 1, top-k 64, top-p 0.95 (the distributions' sampler) | 0.290 |
| Temperature 0.7, top-k 64, top-p 0.95 | 0.622 |
| Temperature 1, top-p 0.95, no top-k | 10.7 |
| Temperature 1, no top-k, no top-p | 12.7 |
| A copy of 1 MiB (`ArrayBuffer.slice`) | 0.063 |

A sampler without a top-k costs more host time per token than the whole E2B forward pass takes on the GPU; the
candidate selection does not cover it (it needs the whole vocabulary). Filed as H-45.

### 6.3 That the tokens are the same

The selection changes where a draw's candidates come from, not the draw. The evidence, in the order of what it rests
on:

- **The host half, without a GPU** (`generation_sampler_candidates_test`, `generation_leading_entries_test`): the
  draw target built from the leading entries equals the whole row's bit for bit — the same tokens in the same order
  with the same f64 probabilities — over 200 combinations of temperature (0.3 to 4), top-k (1 to 64), top-p and
  suppressed sets on rows of 4,096 logits; 64 steps of three samplers under three seeds pick the same token at every
  step; ties at the top-k boundary, candidates a temperature rounds to one value, +0 and −0, suppressed ids above
  every candidate and a NaN at a suppressed id are single cases. Eight faults put into the draw and eight into the
  merge each turn a case red: ties ordered by the larger id, a tie that reaches past the last entry not given up, the
  processed value kept in f64, an unchecked order, suppressed entries kept, a NaN not given up, a single candidate
  consuming a uniform value, fewer than top-k candidates above −∞; a tie between parts to the larger id, an exhausted
  part not given up, a NaN ranked below, places returned as ids, a part's list out of order, a place outside its part,
  the heap not repaired, ties compared by place.
- **The kernel's order, on this GPU** (`gemma4_candidate_selection_test`, Dawn / Vulkan): at the real vocabulary the
  selection, merged and drawn from, gives the whole row's draw target bit for bit on 9 rows (ordinary rows at three
  sampler settings, ties across parts and inside one, ±0, the suppressed ids at the top, a NaN at a suppressed id, a run
  of 40 consecutive ids at the top, the first and the last id), and does not decide on 3 (a NaN that counts, 20
  leading entries in one part, fewer than top-k logits above −∞).
- **Real weights** (`e2e_gemma4_candidate_selection_test`): QAT E2B and 12B (which has suppressed tokens), a pipeline
  with the selection and one built with the test-only key `WHOLE_ROW_SAMPLING`: three chats and a two-turn sequence
  generate the same token ids and stops, the pipeline with the selection reports the entries it asked for (65 and 67)
  and at most 2 steps per chat that read the whole row.
- **The loop** (`generation_candidate_selection_test`): with a fake selection, a turn's tokens equal the whole-row
  turn's under three seeds; a step whose entries do not decide reads the whole row and is counted; a failed run keeps
  the frontier and a failed draw drops it, as the whole-row loop does.

The first real-weights run failed in the pipeline's dispose, not in a token: the candidate selector was released
after the logits resident it reads (fixed in `a721cfbf`; the surface's GPU test has the case, and fails on the commit
before).

Two existing real-weight tests reacted to the selection, neither in a token:

- `e2e_gemma4_chat_test` compares a greedy chat's stop with its pinned value. The sequence reported the selection's
  counters on every turn, a greedy one included (a count of 0); they are now kept for turns that sample only
  (`8bcc7713`), and the test is unchanged.
- `e2e_gemma4_ple_gpu_test` counts the queue waits (`onSubmittedWorkDone`) of a sampled chat with the per-layer
  embeddings on the host and on the GPU, and expected the two counts to be equal. They were equal because each path
  paid one wait the other did not: the GPU path the gather batch of the prefill chunk, the host path the wait for
  released memory that a run building a slot backing starts with. With the selection the host path issues its decode
  through `enqueue`, as the greedy path does, and no longer pays that wait. Counted on one sampled chat of the mobile
  QAT E2B (16 tokens, the same ids in all four): whole row 4 waits on the host and 4 on the GPU, selection 3 and 4;
  the wait that went is the one at the slot backing. The test now has both paths: the same sampler through pipelines
  built with the whole-row key runs as a normal run and keeps the equal counts, and the candidate step is expected
  to pay one more on the GPU path — the gather batch.

What the review of the branch added to this evidence:

- **A float copy on the way.** The selection ranks the row after a `permute`, a copy that loads and stores f32, and
  WGSL lets an implementation flush a subnormal to zero there; the `topk` keeps the bits it reads, but it reads the
  copy. A flushed subnormal ties with zeros and other subnormals and takes its place by its id, so with such a value
  among the leading entries the candidates could be another set in another order. Zeros and subnormals are one block
  of the ranking and a flush moves a value inside it only, so a step whose entries hold neither is the row's own on
  any device; a step whose entries hold one reads the whole row (`leadingOfFloatCopy`). On this GPU the rows with
  subnormals and zeros below the entries are answered bit for bit and the rows with one among them, or as the last
  entry asked for, are not answered. Whether this device flushes was not measured: the rule does not depend on it.
- **The logits of the two paths** (`e2e_gemma4_candidate_selection_test` ③). A candidate step issues the target
  through `enqueue` and copies the logits to a resident; a normal run goes through `run` (the per-layer embeddings on
  the host) or `enqueueRead` (on the GPU). One sampled turn of the mobile QAT E2B generated both ways gives the same
  logits at every decode step, compared as bits, with the embeddings on the host and on the GPU.
- **The merge** gave up whenever a part's whole list had been taken; it now does so only when the list ran out before
  the last entry asked for, which is the case that can hide an entry. The fallbacks counted in section 11 (0 of 762)
  were taken under the former, stricter rule.

### 6.4 What it gives

The arms `yield` and `after` of section 11's run differ in the selection only: both pass a signal on commit `a721cfbf`,
and `yield` is built with `WHOLE_ROW_SAMPLING`. QAT E2B `i4-fast`, the distribution's sampler, 6 warm chats per arm and
prompt, ms per decode token, median over the chats (the least in parentheses):

| | `few-ids` | `many-ids` |
| --- | --- | --- |
| Whole row: a chat's median interval between two tokens | 6.795 (6.776) | 6.806 (6.790) |
| Selection: the same | 6.551 (6.470) | 6.521 (6.472) |
| Difference | −0.244 (−0.306) | −0.285 (−0.318) |
| Whole row: a chat's mean | 7.189 (6.961) | 7.532 (6.969) |
| Selection: the same | 7.527 (6.687) | 7.468 (6.671) |
| Difference | +0.338 (−0.274) | −0.064 (−0.298) |

- By what a token usually takes and by the fastest chat the selection is 0.24 to 0.32 ms per token faster: 3.6 to 4.7 %
  of a token, over this work's kill line (1 % of a token) and the ledger row's own (0.2 ms).
- The medians of the chats' means do not show it. With six chats they scatter by more than the difference: a visit's
  first warm chat is often several tenths of a millisecond per token slower than its second, and single chats by more.
  Both figures are given so the reader sees it.
- All 762 sampled steps per prompt drew from the GPU's candidates; none read the whole row.
- The host's segments were not measured again after the change: what replaced the 0.41 ms of `read` and `draw` (the
  selection's 0.066 ms on the GPU, a 16 KiB read, the merge and the draw) is known as this sum only.

## 7. The first-chunk ramp (H-29): measured, then taken out

Section 4's `start` is the stretch in which the GPU has none of a token's work while the host encodes the first chunk
of 128 dispatches. The candidate was a field of the submit policy that makes a run's first chunk small and doubles
each later one up to the policy's limit, so the GPU starts after 16 or 32 dispatches. It was built (`a624b6a4`,
`0f0d87fb`), measured as two arms of section 11's run — the selection and a signal as in `after`, the Gemma policy's
128 with a first chunk of 16 or of 32 — and reverted (`d47bf453`, `684ba534`).

ms per decode token, 6 warm chats per arm and prompt: the median over the chats of a chat's median interval between
two tokens (the least in parentheses), and the fastest chat's mean.

| Arm | Interval, `few-ids` | Interval, `many-ids` | Fastest chat, `few-ids` | Fastest chat, `many-ids` |
| --- | --- | --- | --- | --- |
| No ramp (the first chunk as large as the window allows) | 6.551 (6.470) | 6.521 (6.472) | 6.687 | 6.671 |
| First chunk 16 | 6.674 (6.630) | 6.674 (6.615) | 6.801 | 6.880 |
| First chunk 32 | 6.678 (6.608) | 6.699 (6.621) | 6.830 | 6.842 |

- The ramp is slower: by 0.12 to 0.18 ms at the median interval, 0.14 to 0.16 at the least, 0.11 to 0.21 by the
  fastest chat. The medians of the chats' means (7.02 to 7.55) do not separate the three arms.
- Why was not measured. The expectation — the GPU starts 0.4 ms earlier, so the token ends 0.4 ms earlier — did not
  hold; by its arithmetic the ramp also adds two (32) or three (16) submits to a token's 896 dispatches, and what a
  submit costs on either side under Dawn is the first thing to look at (speculation).
- Not measured: first chunks of 8 and 64, a greedy arm, and a browser, where the ledger expected the larger gain.
- So the stretch stays: nothing built here shortens `start`. The run-ahead (section 8) removes it for all but one token
  in its depth without adding submits.

## 8. The opt-ins on this machine (H-27, H-28)

QAT E2B `i4-fast`, greedy, no signal, tap off, commit `8417ae1e`, 3 rounds, 6 warm chats per arm and prompt. ms per
decode token, median of the warm chats (the least in parentheses); the token ids of every arm are equal.

| Arm | `few-ids` | `many-ids` |
| --- | --- | --- |
| Per-layer embeddings on the host, one step per batch | 7.043 (6.617) | 7.238 (6.681) |
| Per-layer embeddings on the GPU (`pleResidency: "gpu"`, H-28) | 6.868 (6.693) | 7.166 (6.804) |
| + `decodeAhead: 2` (H-27) | 6.713 (6.592) | 7.136 (6.713) |
| + `decodeAhead: 4` | 6.536 (6.338) | 9.145 (6.571) |
| + `decodeAhead: 8` | 6.444 (6.268) | 6.678 (6.422) |

- H-28 alone is neutral here: −0.07 to −0.18 ms at the median, +0.08 to +0.12 ms at the least. The host's gather it
  replaces is the `derive` segment, 0.06 ms.
- H-27 at depth 8 is ×0.92 at the median (−0.56 to −0.60 ms) and ×0.95 to ×0.96 at the least (−0.26 to −0.35 ms); depth
  2 is ×0.95 to ×0.99. The `many-ids` chats of the arms with the embeddings on the GPU include slow ones (their largest
  is 11.6 to 13.3 ms per token; depth 4's median of 9.1 is three of them), so its medians say less than its least.
- What the run-ahead removes is the stretch between two tokens in which the GPU has no work — `between`, `start`,
  `draw` — for all but one token in `depth`. The first-chunk ramp of section 7 was meant to shorten the same stretch for
  every token and measured slower, so the run-ahead is the one measured way to shorten it.
- Both stay opt-in; no default was changed.

## 9. What is Node's own

The maintainer measures Chrome on another machine; these are the segments whose size is a property of Node and Dawn's
Node binding rather than of the code. The browser column is what follows from how a browser's WebGPU is built, not a
measurement:

| Segment | Under Node | In a browser |
| --- | --- | --- |
| `between` with a signal, before the change | A `setTimeout(0)` is a 1 ms timer: 1.07 to 1.09 ms per token on every model | No 1 ms floor for a timer set outside a timer task; a task round trip (unmeasured here; perf-ledger H-31 estimated 0 to 1 ms). A hidden tab throttles timers to a second |
| `between` with a signal, after | 0.01 ms: the timer armed at one boundary is delivered while the run is awaited | The same mechanism; a hidden tab still waits for a timer per token |
| `start`, `encode` | Dawn's binding: about 3.5 µs of host time per dispatch (set pipeline, set bind group, dispatch), and a submit translates the commands in process | The commands are serialized to the GPU process; the cost per dispatch and per submit differs, so section 7's result need not carry over |
| `wait` | The GPU's time plus Dawn's delivery of the map callback | The GPU's time plus the round trip to the GPU process |
| `read` | A mapped range is the staging memory itself; the copy is the only cost (0.06 ms per MiB) | The mapped range is shared memory filled by the GPU process: a 1 MiB read costs more there (the Chrome upper bounds on record for H-32 include it) |
| `draw`, `derive` | JavaScript on V8: the same engine in Chrome | The same |

The GPU's own time is not Node's: it is the adapter's and the backend's (Vulkan here, D3D12 on the maintainer's
machine).

## 10. Not built, and not measured

- **H-9** (one readback staging per Session) and **H-10** (the issue metadata compiled at build): measured in place by
  section 4's segments and each under 1 % of a token, so neither was built. A greedy token reads back one 8-byte map;
  the run's preparation and its queue writes are a few tens of microseconds inside `start`.
- **K-49** (offset aliases for the per-layer slices): not measured. The run that times every dispatch on its own was
  queued and withdrawn when the GPU queue was limited to runs that decide a change of this work. The row's own
  expectation is 0.17 ms per token, which would be over the 1 % line if it held, so the row stays open. Building it
  changes what the planner treats as an alias, the transient plan and the dispatch counts of every family: a change to
  the forward pass's graph, outside this work.
- **A sampler without a top-k**: the host's scan of the whole row costs 10.7 to 12.7 ms per token (section 6.2) and the
  candidate selection does not cover it. Filed as H-45; not started.
- **Not measured**: any browser; E4B and 12B after the changes (their timer segment before is the same 1.07 to
  1.09 ms, their sampled `read` and `draw` 0.44 to 0.51 ms); the host's segments after the candidate selection; the
  run-ahead with a sampler.

## 11. End to end

One process, one locked run of 317 s on commit `a721cfbf` (the script
`outputs/diag/decode-optimization-2026-11-18/decode-host/final-ab.ts`): QAT E2B `i4-fast`, the distribution's sampler
(temperature 1, top-k 64, top-p 0.95) with seed 0, 128 new tokens, two prompts. Three rounds visit the arms in order,
then in reverse, then in order; a visit builds its pipeline and runs three chats per prompt, of which the first is left
out: 6 warm chats per arm and prompt.

- `before`: the whole logits row read back at every step, and a `setTimeout(0)` awaited by the consumer after every
  token. The old loop is not in this build, so its serial yield is put back from outside, between the same two runs. As
  a cross-check, the loop itself on commit `8417ae1e` measured 8.77 and 8.67 ms per token in another run (section 4).
- `yield`: the whole row, a signal passed — the abort check as it is now (H-31 alone).
- `after`: the candidate selection, a signal passed — the branch as it is (H-31 and H-32).

ms per decode token, median over the warm chats (the least in parentheses); tokens/s is that of the median chat mean.

| Arm | Prompt | Chat mean | Tokens/s | Interval between two tokens | Token ids |
| --- | --- | --- | --- | --- | --- |
| `before` | `few-ids` | 8.956 (7.983) | 111.7 | 7.884 (7.857) | — |
| `yield` | `few-ids` | 7.189 (6.961) | 139.1 | 6.795 (6.776) | equal to `before` |
| `after` | `few-ids` | 7.527 (6.687) | 132.9 | 6.551 (6.470) | equal to `before` |
| `before` | `many-ids` | 8.877 (8.304) | 112.7 | 7.882 (7.865) | — |
| `yield` | `many-ids` | 7.532 (6.969) | 132.8 | 6.806 (6.790) | equal to `before` |
| `after` | `many-ids` | 7.468 (6.671) | 133.9 | 6.521 (6.472) | equal to `before` |

- Before to after: what a token usually takes 7.88 → 6.52–6.55 ms (×0.827 to ×0.831); the median chat mean ×0.840 to
  ×0.841 (111.7 → 132.9 and 112.7 → 133.9 tokens/s); the fastest chat 7.98 → 6.69 and 8.30 → 6.67 ms per token.
- Of that, the yield is 1.08 to 1.09 ms and the candidate selection 0.24 to 0.29 ms by the interval.
- The token ids of every warm chat of every arm equal `before`'s for the same prompt and chat.
- The chats' means carry the scatter described in section 6.4; the interval is the figure to compare arms by.
- This is Node. In a browser the yield is not a 1 ms timer (section 9), so the first part is expected to be smaller
  there and the second, whose readback crosses a process boundary, not smaller; neither was measured.
