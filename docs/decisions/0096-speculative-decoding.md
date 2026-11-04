# 0096: Gemma 4 MTP — speculative decoding as a whole (perf-ledger K-20 ④)

- Status: accepted (2026-09-08 — the maintainer's rulings: design axis 1 = B, read-only slots across
  Sessions; design axis 2 = S, shared initializer; seven single-axis points; stages 1 → 4 in order).
  Decision 4's `k ≤ slidingSlack`, decision 7's "temperature > 0 later" and the stage 2 scope are
  superseded by the 2026-09-08 stage 3 addendum; decision 5's pending GPU argmax by the 2026-09-09
  addendum (not adopted); decision 8 is extended by the gate addenda (2026-09-09 to 2026-09-26)
- Date: 2026-09-08
- Related: ADR [0066](0066-generation-context-state-slots.md) (decision 6, the logical length
  advances on run success — decision 3 adds deferred commit; addendum 2, no rewind with sliding slots —
  unchanged, speculation does not rewind) / [0067](0067-autoregressive-attention-vocabulary.md)
  (decision 4, one `slot_row` formula, window ≤ capacity — decision 4 adds slack; decision 5, exactly one
  append per slot — decision 1 adds read-only external slots) / [0068](0068-decode-exit-multi-output.md)
  addendum 6 and [0083](0083-generation-api-surface.md) decision 6 (exit = last-row logits `[1,1,V]` —
  generalised by decision 5) / [0069](0069-packed-w4-storage.md), [0019](0019-i8-weight-execution.md)
  (storage belongs to the initializer — decision 2 changes only where the bytes come from) /
  [0085](0085-ple-host-gather.md) (PLE host gather — why a cycle is two runs) /
  [0095](0095-plan-backing-budget.md) (backing budget — holds the verify-shape backings)
- Measurements: [research 2026-09-08](../research/2026-09-08-mtp-ea-i4-target.md) (E[a] with an i4
  target) / [research 2026-09-07 §7](../research/2026-09-07-codex-perf-review-followup.md) (F / T(M) / D)

## Context

Gemma 4's official MTP drafter (`google/gemma-4-E2B-it-assistant`) is a small Transformer (4 layers,
hidden 256) with **no KV of its own**: it cross-attends to the K/V of target layers 13 (last non-shared
sliding layer) and 14 (full). Input: `concat(target_embed(token)·√1536, target hidden after the final
norm)`; output: the next token's logits and the next step's hidden. The k draft steps are sequential;
verify runs the k+1 rows `[bonus, d₁..dₖ]` through the target once, takes the matching prefix as the
accepted count a and confirms `a+1` tokens (greedy output equals non-speculative decoding).

Measured (revival conditions ① to ③, 2026-09-07 to 08): small-M linear runs at M=1 speed (K-21);
shape-switch rebuilds disappeared with the budgeted LRU (H-15); with an i4-equivalent target, E[a] on
extractive long contexts is 2.3 to 2.6 at k=3 and 3.8 to 5.0 at k=6 (E-4). Forecast: extractive long
context 1.8 to 2.6× (Deno) / 1.9 to 2.4× (browser-equivalent); free text 0.9 to 1.0× (k ≤ 3).

Where speculation hit the contracts of the time (recon 2026-09-08, HEAD `4b2f6ba`):

- State slots belong to a context bound to a Session; another Session's run fails the ownership check.
  `validateGraphContracts` refused reader-only slots (exactly one append, the append is the last node).
- The logical length always advances by `queryLength` on success (no partial commit). The sliding ring
  closes at exactly the window, so two or more rejected rows overwrite the live window's oldest row.
- The only exit is last-row logits (`last_row [1]`); no exit for the final hidden the drafter needs.
- No sharing of a weight's GPU buffer across Sessions; the drafter needs the embedding table for its
  own tokens in steps 1..k−1.
- PLE is a host gather (ADR 0085), so draft tokens decided inside a graph cannot have their PLE rows
  fetched first → **a cycle is at least two runs, draft + verify** (one run is not feasible).

## Decision

### 1. The drafter is a separate Session that **only reads** the target's KV slots (design axis 1 = B)

The IR slot declaration gets a third kind, **external** (readers, no append; another context owns the
storage). The executor's ownership check and bind-group baking accept "the owning or the bound
context" (the binding became a borrowing context at stage 2, see that addendum). MUST: lifetime is
**drafter Session ⊂ target context** (the drafter is disposed before the target's context / Session; a
poisoned target makes the drafter's runs fail). MUST: writes to an external slot are excluded
structurally (it cannot have an append) — only the target's runs / commits advance its logical length.

Not taken: **C′ mirror slots** (the target outputs layer 13 / 14 K/V rows, the host writes them into
the drafter's own slots) needs the same append-less contract change plus a slot-write API, four more
target outputs (pinned, fusion broken), duplicated KV and a copy per cycle. **A, one graph and one run**
needs PLE gather and greedy acceptance inside the graph; not feasible.

### 2. The drafter's input embedding table **shares the target's resident weight across Sessions** (design axis 2 = S)

The drafter container declares the embedding initializer as shared, without bytes, and
`createSession`'s `sharedWeights` binds the target's resident weight (the same GPUBuffer of i8 + scale,
the same storage). Distribution +0 MB, VRAM +0. Storage stays with the initializer (ADR 0019 / 0069
unchanged); only **where the bytes come from** changes. The drafter's own output head (`[262144, 256]`)
is stored in the drafter container. Not taken: a copy (+201 MB distribution and VRAM even at i4); one
run per step with host row lookups (k+1 fences per cycle).

### 3. Partial commit = **deferred commit**

A generation run takes `commit: "deferred"` (default immediate = as before). A successful deferred run
does not advance the logical length; the context keeps `pending = {pastLength, queryLength}` until the
host reads back the accepted count and calls `context.commit(rows)` (`0 ≤ rows ≤ queryLength`). While
pending, new runs and rewind are refused (dispose is allowed); a failure poisons as before (addendum 3
unchanged). ADR 0066 decision 6 becomes "immediate advances on run success, deferred on the commit after
success" — still no double bookkeeping. Not taken: a separate per-cycle lease (it would redraw every
rewind / dispose / poison barrier).

### 4. The sliding ring gets slack: capacity = window + kmax (kmax = 8)

`slot_row(col) = col % capacity` (one formula for read and write; `column_base` / `live_columns` /
`in_window` stay window-based). Rejected row i overwrites logical column `P+i−capacity`, always outside
the post-commit live window `[P+a+2−W, …)` for `k ≤ kmax`. The state-append de-duplication guard (with
Q > capacity only the last logical row writes) is capacity-based too. The exporter's `states_plan`
stores the sliding capacity as `window + 8` (`SLIDING_SLACK_ROWS` in
`tools/export-recipes/gemma4/export_decode.py`); +24 sliding slots × 8 rows × 1 KiB ≈ 192 KiB VRAM.
MUST: the speculation loop keeps `k ≤ slidingSlack` (the context's minimum of capacity − window).
(Superseded on 2026-09-08 by the stage 3 addendum: a deferred run is limited to
`queryLength ≤ slidingSlack`, so the loop MUST keep `k + 1 ≤ slidingSlack`.)

### 5. The verify exit = two outputs, logits `[1,R,V]` + final-norm hidden `[1,R,H]`

`last_row` becomes `[R]` with a second symbol R; R=1 is the previous prefill / decode (values and token
sequences bit-identical). R is not shared with M (prefill would read back `[1,768,V]`). Readback is R
MiB (9 MiB at R=9); GPU-side argmax (revising ADR 0083 decision 6) was to wait for measurement
(superseded on 2026-09-09 by the stage 4 addendum: not adopted, host argmax + readback is 2 to 5 % of a
cycle). The hidden feeds the drafter (row a is selected).

### 6. Verify M and k

The default `chunkBuckets` gain 4 and 8 (k+1 ≤ 8 → M ∈ {4, 8}; `GEMMA4_CHUNK_BUCKETS` in
`packages/models/src/gemma/pipeline.ts`), and `PREPARED_PLAN_CAPACITY` goes from 8 to 12. Padding to
M=16 lost after K-21 (T(8) 37.2 < T(16) 38.6 ms). k is bounded by the distributed drafter's step count,
3 (`assertSpeculativeSetup` checks `1 ≤ k ≤ steps`; E-4: extractive 1.8 to 2.1×, free text 0.9 to 1.0×).

### 7. Acceptance rule: greedy first, on the host

It lives in `src/generation/` with one argmax implementation and one tie rule (ties go to the lowest id, as torch's argmax). Repetition penalty /
logit bias are applied per row while the history grows. Temperature > 0 (acceptance probability
`min(1, p/q)` and the residual distribution) was deferred to stage 4. (Superseded on 2026-09-08 by the
stage 3 addendum: acceptance calls `sampler.next` per row and is exact at every temperature.)

### 8. Distribution = a second role `drafter` in the same ModelEntry (not downloaded by default)

The drafter shards are fetched and its Session created only when `speculative` is requested. Its output
head is the plain lm_head + argmax (the exporter has no top-k for the centroid sparse softmax;
acceptance is identical (E-1)). The default policy and the self-tuning gate are in the 2026-09-09
addendum and its revisions.

## Stages (history)

Stages 1 to 4 (exit and commit → external slots → speculation path → measurement) were completed by
2026-09-10; their checks live in the addenda and `packages/e2e/tests/e2e_gemma4_speculative_test.ts`.

## Consequences and open items

- M ∈ [2, 16) attention is in the band where neither ①′ nor ①ₜ helps (T(4)=27.6 / T(8)=37.2 ms).
- Two verify shapes raise the PreparedPlan and backing-budget counts (within H-15's 256 MiB).

## Addendum (2026-09-08) — changes made while implementing stage 2

- **Decision 1 binds through a context, not `createSession`**: the pipeline has one drafter Session
  (`createSession(model, { sharedWeights })`, decision 2), and per conversation
  `drafterSession.createGenerationContext({ chunkLength: 1, borrow: targetContext })` creates a
  **borrowing context**. Reasons: a GenerationContext is per sequence (ADR 0083 decision 3), so binding
  at the Session would re-upload the drafter's output head (i8 ≈ 67 MB) per conversation; and backing
  generation tokens are numbered per Session (executor.ts), so two Sessions baking into one context
  would collide. A borrower's run takes the lender's run lease (a running lender run, an uncommitted
  pending or poison refuse it), so the runtime serialises them. The IR declares external slots
  (`states[].external: true`; no append; read only by readonly attention; all slots external), readonly
  attention (`attrs.readonly: true`, `ins: [q]`, M = 1) and shared initializers (`{ "shared": true }`
  under the lender's initializer name, no bytes — `packages/runtime/src/format/ir.ts`). Disposal goes
  borrower → lender; the reverse order rejects (a deliberate departure from the idempotent contract).
- **Decision 8's "not downloaded by default" is the hub's `ResolveOptions.weights`** (the subset of
  weights to fetch). A selection's refs expand every weights entry, so a second role alone would always
  be downloaded — gemma4 requests `["model", "drafter"]` only when `speculative` is given.
- **The drafter's linears are i8 (not i4 g32)**: on one target trajectory (2026-09-08, N=200 × 3 cases)
  i4 g32 linears cut E[a](k=3) by 15 to 33 % against unrounded, i8 by 2 to 3 % (shared table in i8: ±0);
  the loss sits in the drafter's own 22 linears (13 % of elements). Distribution: just under +4 MB.
- **The readonly sliding window is 512 columns** (`[P−W, P)` — mlx's `|q−k| < W`, the target's own
  predicate). HF's bidirectional SWA is inclusive (513), but column 513 (position P−513) may be
  overwritten by a kmax=8 rejection in the slack-8 ring. The drafter comparison on identical inputs is
  unaffected (150/150 against HF's regular path).
- Stage 2's scope (k fixed at 3, target-only estimate) is history: see stage 3 and decision 6.

## Addendum (2026-09-08) — changes made while implementing stage 3

- **Stage 2's drafter calling convention was off by one (correction)**: stage 2 passed
  `(token@P−1, hidden@P−1, position P−1)`, but HF's `SinglePositionMultiTokenCandidateGenerator` and the
  drafter's own step recursion (a token travels with the hidden of the row that produced it) define
  **`token = b` (last confirmed token = position P, the frontier not yet in the KV), `hidden = h@(P−1)`,
  `position = P`**. Shifted, d₁ re-guesses the bonus the target gives for free from logits@P−1 and the
  effective k drops by one (d₁ match at P 0.39 to 0.53, at P+1 0.03 to 0.08). The graph was intact; the
  fix is on the host (`speculative.ts`), in the script (`draft_case`) and in the regenerated golden
  (container bytes and distribution unchanged). Golden after the fix (greedy, k=3, N=200): short-en 1.50
  / readme-recipes 2.13 / readme-exporter 2.11 tokens/cycle. Stage 2's "1800/1800" and "HF 150/150"
  checked **the torch port's equivalence**, not acceptance.
- **Decision 4's limit tightened by one column**: a deferred run is limited to
  **`queryLength ≤ slidingSlack`** (stage 1 had `slack + 1`). A borrower reads one column older, `P−W`,
  so rejected row j's column `P+j−C` stays outside its window iff `j − m < C − W = slack` (m = committed
  rows, including `commit(0)`) ⟺ `Q ≤ slack`. No branch on whether a borrower exists (one can open
  after a deferred run + `commit(0)`). MUST: `k + 1 ≤ slidingSlack` (gemma4: 4 ≤ 8); enforced in
  `packages/runtime/src/runtime/executor.ts` and by `assertSpeculativeSetup`.
- **Decision 3's commit goes "up to the delivered frontier"**: the accepted sequence `[d₁..d_a, b']` is
  delivered one token at a time on the existing MUST that updates `pendingToken` before yield, and
  `commit(rows)` covers **the frontier of the tokens that reached the consumer** (`rows` = rows the
  target consumed). It settles right after the delivery loop and in the generator's `finally`, so a
  consumer `break` commits up to the delivered tokens and an exception between verify return and delivery
  becomes `commit(0)`; both close to "received tokens + one frontier", as without speculation (a pending
  commit leaves the context accepting only dispose).
- **Decision 5's R is fixed at k+1**: verify's `last_row` pads `[0..k']` with the last index to `k+1`
  (R is in the PreparedPlan key; a shrinking k' makes no new shape). `k' = 0` is `[0]`, the decode shape.
- **Decision 7 closes independent of temperature (correction)**: acceptance calls `sampler.next` once
  per confirmed token with the same logits, history and order as non-speculative decode, so the RNG
  consumption and the token sequence **match exactly** at every temperature. At temperature > 0,
  "accept when the sample equals the draft" is speculative sampling with a one-hot draft (on rejection
  the sample becomes the frontier); the drafter's logits are not needed. The design-time "temperature
  ≠ 0 falls back to non-speculative" rested on a wrong premise (refuted in review); **no stage 3b**.
- **Loop and observation**: the loop is inside `packages/models/src/generation/sequence.ts`; the drafter
  is injected as `GenerationSequenceOptions.speculative = { open(context) → DraftFace, k }` (`DraftFace`
  = draft and dispose only; a GPU-free fake covers zero acceptance, stop token, abort, exception and
  break). Observation is the per-run hook `onRun(phase)` (prefill / decode / draft / verify, numbered from
  1; verify fires in the synchronous span right after commit), replacing `withRunDiagnostics`.
  `GenerationStop.speculation` (cycles / draftRuns / drafted / accepted / acceptedHistogram) counts.
- **Identity check**: on gemma4's default preset (`stateAttentionReduce: "parallel"`) decode (M=1) used
  ①′ and verify (M=4) ①, so they were not bit-identical (by design). The exact-match check runs on the
  `"sequential"` preset (① + ③ for both); the default preset reports mismatches. Since the stage 4-B
  ruling (2026-09-09) widened ①′ to M ≤ 8, verify uses ①′ too.
- **Estimate**: `estimateSessionMemory` adds the verify shape `{ M, R: k+1 }` to the target's prefill /
  decode (runtime `generation.scenarios`; M is k+1 rounded by `physicalChunkRows` to the context's
  bucket, as the real run uses), plus the drafter's resident weights (shared initializers excluded) and
  the borrowing context's 8 bytes of lengths.
- **The readonly window stays at 512 columns** (HF: 513, deliberate; stage 2 reason).

## Addendum (2026-09-09) — readonly attention's reduction order ignores the preset (supplement to decision 7)

- **The drafter's readonly attention (①′ ③′) is always parallel**: `buildReadonlyStateAttention` in
  `recipe-builders/attention.ts` always picks `stateQkParallelReadonlyWgsl` /
  `statePvParallelReadonlyWgsl`. The `stateAttentionReduce` option governs only **the target's
  states-shape attention** (which also reads this step's k/v); the borrower's path has no branch.
- So the `"sequential"` exact-match check covers **the target's identity**, not the drafter's order,
  which changes only the draft (= acceptance = speed), never the confirmed tokens: the target's logits
  and `sampler.next` decide per row (decision 7). The check holds with the drafter on parallel.

## Addendum (2026-09-09) — stage 4 measurements and the self-tuning gate (decision 8, consequences corrected)

- **Measurements ([research 2026-09-09](../research/2026-09-09-mtp-stage4.md))**: the gain depends on
  the task. RTX 3080 Ti / Deno decode: extraction 1.81× / summary 1.41× / dialogue 1.18× / free text
  0.96×; Apple M2: 1.27× / 1.03× / 0.93× / 0.76×. Speedup = A / A\* (A = confirmed tokens per cycle, A\* =
  cycle wall / decode wall) held within 2 % in all 8 conditions; A\* moves with the host (RTX 1.72 to
  1.88, M2 2.17 to 2.6). At temperature 1.0 acceptance equals greedy and token sequences match.
- **Decision 8 — the self-tuning gate (on by default)**: `speculative: true` (default) measures the
  cycle wall and the plain-step (M=1) wall at run time. When `ΣWc / Σdelivered` per **block of 16
  cycles** exceeds `W1 × 1.01` for **two blocks in a row**, it falls back to the decode shape; while
  plain it tries **bursts of 8 cycles** and returns when the aggregate falls below `W1 × 0.97` (burst
  interval in geometric backoff; W1 = EWMA of the plain-step wall). Thresholds are not hard-coded because
  A\* moves by 0.5 or more with host and context. Block aggregation brings the per-cycle acceptance
  spread (sd 0.88 = 43 % of the mean) to ≈ 0.09 (ratio); the first version, seeded from one sample, left
  for good when the first cycle accepted 0 (e2e: left at cycle 7, 190 steps without return, 31 % slower).
  The plain step uses the `k' = 0` path (verify shape M=1, deferred): the hidden stays connected and
  `commit(0)` prevents feeding the frontier twice after a failed draw. It guards only against large
  losses (M2 free text −24 %, leaves within 32 cycles); 3 to 4 % differences are not distinguished
  (dialogue-class tasks leave by mistake ≈ 1.5 % per turn). `"always"` speculates unconditionally (A/B,
  acceptance and measurement). Pure-function module: `packages/models/src/generation/speculation-gate.ts`.
  (Its claim that the default preset may rarely change output was withdrawn on 2026-09-26, below.)
- **Supplement to decision 7**: a confirmed stop token **stops** the enumeration (no logits beyond what
  non-speculative decoding touches; RNG consumption lines up). `accepted` counts delivered acceptances;
  tokens/cycle is `delivered / cycles`.
- **"One fence per cycle" does not hold**: a cycle is at least draft → verify. The host must read the
  draft ids to gather PLE for verify's input, and judges acceptance. On Deno ≈ 9.5 ms of the 15 ms
  draft wall is the round-trip floor (expected to vanish in browsers).
- **Stage 4-B ruling (2026-09-09)**: ⑤ row tiling of ①′ / ③′ is **narrowed** — ③′ already applies at
  M=4; only ①QK was limited to M=1. RTX (M=4, P≈4.8K) was neutral (①QK full 3.32 → 3.87 / sliding 1.87 →
  1.09 ms, wall −2 %), and **①QK's applicability was widened to M ≤ 8** (on the default preset verify row
  0 and decode share the reduction order = u32 match, so the gate's switch does not move row 0's logits;
  M2 unmeasured). ⑧ M2 small-M linear: the row-block parallelism target (`linearGemvRowsThreadTarget`)
  leaves the default r1 best (perf-ledger K-22). ⑦ A GPU-side argmax / top-k exit is not adopted (host +
  readback is 2 to 5 % of a cycle). Remaining: ③ two-phase argmax → ⑥ k=7 drafter + three-valued gate.

## Addendum (2026-09-10) — self-tuning gate v3 (update to decision 8)

- The stage 4-B ⑨ breakdown (research 2026-09-09 §6.4.1): on the losing task (M2 free text, ratio 1.27)
  62 % of the loss was **exploration bursts**, 36 % the 32 cycles before leaving. Additions in
  `speculation-gate.ts` (public shape unchanged): an early leave on a strong loss (off by default since
  v3.1, below); **early burst abort** `burstMin` (default `min(4, burst)`: from the 4th cycle, a ratio
  above `leave` + 0.15 folds the burst as a miss — backoff doubles, `switches` unchanged); **exploration
  interval 8 → 16** (initial burst interval and `W1` re-measurement).
- Simulation and real runs: research 2026-09-09 §6.5. Open item ⑩: the gate's plain step (verify shape,
  one valid row) costs +17 to 30 % over a true decode at long context, on the host / readback side
  (table 12); attribution needs per-phase walls inside the run.

## Addendum (2026-09-10) — gate v3.1: early leave off by default, separate speculate-side and plain-side knobs

- v3's strong-loss knob served the speculate-side one-block early leave and the plain-side burst abort.
  On a warm script (a different utterance per turn) v3 left by mistake in 3 of 7 turns of a winning task
  (RTX dialogue, ratio 0.90; auto / always 0.96) against 0 of 7 with v2's knobs (1.00) (research
  2026-09-09 §6.5 table 11″). The burst knob acts only on the plain side, so the early leave is the
  cause: acceptance is not steady within a turn, and one 16-cycle stretch with A < 1.4 knocks it over.
- The early leave saves one block once per sequence and costs on every turn of a winning task, so
  **early leave is off by default** (`earlyLeave?`, effective only when passed); the plain side keeps
  `burstAbort` (default 0.15, from the `burstMin`-th cycle). `exploreBase` stays 16; `exploreMax` 256 →
  **512** (same-day ruling: the remaining 2 to 3 % in warm runs was one burst per turn at the backoff
  ceiling; bursts become one per 2 to 3 turns, reaction to a newly winning context 15 s RTX / 30 s M2).
- The knobs are public as `Gemma4PipelineOptions.speculative.gate` (measurement and acceptance; the
  defaults suffice; `"always"` turns ignore it). M2 v3 acceptance numbers: research 2026-09-09 §6.5.1.
  The `tools/mtp-bench` flags named here went with the Deno tools (ADR
  [0127](0127-node-dawn-toolchain.md) decision 5).

## Addendum (2026-09-26) — the self-tuning gate and determinism (withdraws "the default preset may rarely diverge")

- **Ruling and basis**: in-device determinism is MUST for practical presets too (the maintainer's
  ruling 2026-09-26, ADR [0110](0110-practical-tier-numerics-contract.md) decision 6). The gate picks
  decode (M=1) or verify (M=4) by wall clock; if the two paths are **mechanically verified**
  bit-identical, either gives the same output and the gate meets the MUST (ADR
  [0058](0058-numerics-opt-in-contract.md) addendum 2026-08-29, general rule 3 — auto-selection
  justified by bit identity requires mechanical verification).
- **Checks** (`packages/e2e/tests/e2e_gemma4_speculative_test.ts`): speculation ⑤ = u32 match of verify
  row 0 and a one-row decode's logits (all 12 combinations of 3 attention variants × 2 GEMV kinds ×
  RMS→add fusion, plus linear→SRQ); speculation ② = on the distribution's default preset (`i4-fast` +
  attention parallel) speculative = non-speculative token sequences (3 cases × 200 tokens) and gated =
  always. Before this addendum ② and ⑤'s attention parallel × GEMV sequential were reports, not checks.
- **Measured** ([research 2026-09-26](../research/2026-09-26-gemma4-speculative-determinism.md), Intel
  Arc B570): default preset 0/200 mismatches × 3 cases; ⑤ matched in every combination (attention
  parallel × GEMV sequential: 0/262,144 words, max absolute difference 0).
- **Withdrawn**: the 2026-09-09 addendum's "with the default preset the gate switches by wall clock, so
  output may rarely change with the same seed" (that addendum stays as the record of its time). The
  guarantee is **per device**: where the check is red, M=1 and M=4 really differ — for strict
  reproducibility there, choose `speculative: "always"` or `stateAttentionReduce: "sequential"` (both
  remain as A/B and acceptance settings). NOTE: the check verifies only devices that ran the e2e lane;
  the run-time canary of general rule 3 (checking on the user's device) does not exist. The u32 check
  covers all 12 combinations (+ linear→SRQ), widened the same day; all 0 mismatches on the B570.
- **Note (2026-11-04)**: the regular `karume-gemma4` now ships `bf16` alone (ADR
  [0128](0128-default-preset-original-weights.md), second addendum of 2026-11-04), so these checks
  run on the q4_0 QAT E2B (`karume-gemma4-qat`) and its QAT drafter goldens. Speculation ② names
  `i4-fast` (the same three session flags) because that line's default is the reference `i4`. The
  acceptance floor (1.9 tokens per cycle) and the golden-prefix floor (16) were re-measured there on
  the RTX 3080 Ti and stay.

## Addendum (2026-10-10) — the shared-initializer check also matches the semantic dtype (supplement to decision 2)

- **Decision**: a borrower's shared initializer MUST declare the same semantic dtype (the `values{}`
  declaration) as the lender's. `Session.exportWeight` carries the lender's semantic dtype on
  `SharedWeight`'s internal face; the borrower's construction (`resolveSharedWeights`) compares it right
  after the declared-shape check. A mismatch raises `ExecutionError` before any borrower weight byte is
  uploaded.
- **Why**: f32 and i32 weights are both resident as `raw`, so the storage check cannot tell them apart;
  the same 4 bytes would be read as another type (f32 1.0 becomes i32 1065353216). Ordinary initializers
  are matched against their storage codec by the binding layer (`format/container/bind.ts`); a shared
  initializer has no storage and skips it. The comparison is on the semantic dtype, not the codec — a
  compressed resident weight is f32 once expanded.
