# 0085: PLE distribution — token-major rows, vocab-range splits, host-side gather

- Status: implemented (accepted 2026-08-31 — the maintainer adopted the recommended option for the
  draft's open question 3; stage 1b `16bf5aa6`); partly superseded: decision 3's "no range reads in
  hub" by the 2026-09-07 addendum and ADR [0086](0086-distribution-source.md) capability ⑧
  (2026-09-07); the shard width, the shard-based default budget, the cost-type policy table and the
  manifest-asset check by [0108](0108-container-format.md) decision 5 and the 2026-09-22 addendum
  (2026-09-22); the GPU-resident load mechanism of 2026-09-19 by the 2026-09-24 addendum; the
  row cache's admission rule and its share of the budget (2026-09-12 addendum) by the 2026-10-27
  addendum
- Date: 2026-08-31
- Scope: `tools/export-recipes/gemma4/` (moving PLE out of the graph, writing the sidecar) /
  `packages/models/src/gemma/` (host-side loader). IR, runtime and hub **unchanged**.
- Related: [0083](0083-generation-api-surface.md) (generation API — same re-export, option α) /
  [0084](0084-gemma-tokenizer-chat.md) (decision 5: the digest set) /
  [0070](0070-shard-loading-admission.md) (shard loading and admission) /
  [0081](0081-shard-spec-v2.md) (shard spec v2, 1 GiB limit) /
  [0066](0066-generation-context-state-slots.md) (Session / context lifetime) /
  [0038](0038-manifest-v1.md) (cache keyed by URL, no partial reads)
- Basis: [research/2026-08-31-generation-api-design-draft.md](../research/2026-08-31-generation-api-design-draft.md)
  §2 (candidates, measured values) /
  [research/2026-08-31-freetoken-moe-over-arraybuffer.md](../research/2026-08-31-freetoken-moe-over-arraybuffer.md)
  §3, §4 (**the first use of c-1, host-side gather**) / [limitations](../limitations.md) (Chromium's
  single-ArrayBuffer limit)

## Context

Gemma 4 E2B's PLE (per-layer embeddings) is a pure row lookup on `input_ids` **alone**, and the recipe
already has the cut point: `per_layer_inputs(tables, input_ids, scale)` in
`tools/export-recipes/gemma4/ple.py` builds `[1,M,35,256]` for both the decode and the token scripts.

- Resident size is **i8, 35 tables × 64 MiB = 2,240 MiB** (from the container header), **59%** of
  the 3.70 GiB container. Moving it out takes **3,787 MiB → 1,547 MiB**.
- The runtime **gives every initializer a GPU-resident slot at Session construction**
  (`packages/runtime/src/runtime/executor.ts`: the planner is the source of truth, all initializers
  are loaded), so **nothing that stays in the graph becomes lazy**.
- **Speed is unaffected.** In the first measurement (before K-11; `linear` took 78.7 of 86.2 ms of
  decode GPU time) the 35 embeddings were inside "the remaining ≈935 nodes, ≈4.5 ms". The new cost,
  uploading `per_layer_inputs` f32 (35,840 B per decode token, 1,146,880 B per prefill chunk of 32),
  is negligible against the wall times (decode 32.5 ms/token, prefill ≈162 ms/chunk).

This is **purely a residency question**, judged by the residency saved.

## Decision

### 1. The distribution layout is fixed as token-major rows split by vocab range

`[token][layer][256] i8` + `[token][layer] scale`, **token-major**, **split by vocab range**:

1. **The layout is independent of the host strategy (fully resident or partial reads)**, so fixing it
   first costs nothing: moving later to "read only the rows" (alternative b) needs **neither a
   re-export nor an HF re-upload**.
2. **Token-major makes one token's PLE one contiguous read** (8,960 B + 35 scales); table-major would
   need **35 scattered reads**, multiplying I/O by 35 once partial reads arrive.

### 2. Splitting is a requirement, not a matter of weight (the single-ArrayBuffer limit)

One full table on the host is **impossible in Chromium by construction**:

- Full PLE in i8 = 262,144 × 8,960 = **2,348,810,240 B** (= 35 × 64 MiB = 2,240 MiB).
- Chromium's single-ArrayBuffer limit = **2,145,386,496 B** (limitations — the limit that made anima
  Base f16 unloadable).

NOTE: the draft §2.3 writes `2,351,662,080 B`, a multiplication error (the correct value matches the
draft's own "35 × 64 MiB = 2,240 MiB"); **the conclusion does not change**.

So "host residency is simple but heavy" does not hold — **splitting is required for every option**.
Splitting into 35 tables (67,108,864 B each) avoids the limit but is the table-major form decision 1
avoids.

### 3. The first host side is "lazy loading of touched shards + LRU"

Vocab-range shards are **loaded lazily per file and evicted by LRU**; hub's whole-file reads were
enough at the time. Hub gains no range-read capability here (superseded by the 2026-09-07 addendum
and ADR 0086 capability ⑧ on 2026-09-07: a real need appeared — 137 shard re-reads, about 42 s, for
400 tokens of natural text — and hub gained range reads).

**Not measured (speculation):** which vocab ranges real conversations touch; frequency-ordered
SentencePiece vocabularies are a general claim, unchecked for this checkpoint. The shard width was to
be chosen after measuring the three golden cases and a chat corpus (superseded by ADR 0108 decision 5
on 2026-09-22: the container's block limit fixes the width — 2026-09-22 addendum).

### 4. Host-side dequantization MUST be bit-identical to the GPU `embedding`

Otherwise token-sequence parity breaks, and so does the proof of unchanged behaviour (used when the
sliding capacity changed, ADR 0066 addendum 9). `per_layer_scale` = `256 ** 0.5` = **16.0**, a power
of two, so f32 multiplication is exact (the `ple.py` docstring relies on the same fact); **matching
the order is enough** (stage 1b's acceptance criterion).

### 5. The loader cross-checks the id spaces

The loader compares the ids the tokenizer can produce, the main embedding's vocab rows, the PLE
sidecar's rows and the special ids. **A mismatch does not go out of bounds; it reads a valid row of
another token** (silent corruption), so the fail-loudly check belongs here.

### 6. No "pageable initializer" — PLE simply becomes an ordinary graph input

The runtime gets no **fifth weight lifetime** (unloaded initializer / dynamic residency). PLE is an
ordinary graph input supplied by the host (`per_layer_inputs`); the runtime contract does not change.
Precedents: the relative position tables of flow / voice are host-built graph inputs (limitations);
ADR 0079. An "unloaded initializer" slot was **already declined by the MoE ruling of 2026-08-31**
(backlog parked, dependent on value-dependent execution selection in the IR).

## Alternatives considered

- **b) Read only the rows from hub's cache:** uses neither RAM nor VRAM, but hub had no partial reads
  and it touches ADR 0038's cache design. **Same layout as decision 1**, so the host side can switch
  later (adopted 2026-09-07 — addendum below).
- **c) Keep PLE in the graph as i4** (2,240 → 1,120 MiB): recipe-only, but the recipe README says
  "embeddings are int8 … not int4-eligible", the quality risk needs parity work, and it saves less
  than decision 3. Rejected (i4 stays a separate future candidate).
- **a) Full host residency (split):** splitting is required anyway, after which it has no advantage
  over reading what is touched; a permanent 2.19 GiB JS heap in a browser is unrealistic. Rejected.

## Consequences

- **GPU residency 3.70 → 1.51 GiB** (−59%); **speed unchanged**.
- **Stage 1b takes option α** (draft §7.1, open question 5): moving PLE out and the last-row logits
  exit (ADR 0083 decision 6) ride **one re-export, one product graph series**; the host PLE loader
  becomes a precondition of the e2e tests. The two existing series (logits opt-in / token-only) remain
  as acceptance fixtures.
- **Acceptance (stage 1b):** cross parity with the `greedy.<case>` goldens (`argmax(logits)` equals
  the existing tokens, 3 cases × K=16) + bit-identical PLE dequantization (decision 4).
- The PLE sidecar is a member of the distribution digest set (ADR 0084 decision 5).
- This is the **first use of c-1 (host gather)**, consistent with the scale-axis stance (MoE keeps
  every expert in VRAM, budgeted by total parameters — limitations); it helps **row-sparse tables**
  (embedding / lm_head), not expert FFNs (that research, §4).
- Addendum (2026-09-05): `fromPretrained` compared the PLE index's `shards[].file` with the
  manifest's lazy asset keys before `#build` (superseded by the 2026-09-22 addendum on 2026-09-22:
  admission cross-checks the index against the container's assets).
- The per-step PLE upload (35,840 B per decode token) may become worth observing if decode gets much
  faster; it is negligible today.

## Addendum (2026-09-02) — decision 3's residency limit is bytes, not a count

The limit is **host RAM in bytes** (`Gemma4PleOptions.maxResidentBytes` /
`Gemma4PipelineOptions.maxResidentPleBytes`), not a number of shards: a count means different RAM in
each asset generation (E2B had 3 shards of 758 MiB under the 1 GiB limit and 9 of 253 MiB under ADR
0090's 256 MiB writer limit, so "3 resident" silently went from 2.2 GiB to 759 MiB). One shard's size
is known from the index alone, so budget checks and eviction are decided before reading. The default
was two of the largest shards (superseded by the 2026-09-22 addendum on 2026-09-22: two of the largest
blocks, about 64 MiB). `0` is a valid "keep nothing"; any other budget that **cannot hold one shard
fails loudly at construction** — silently exceeding it voids the budget, silently honouring it leaves
the gather unable to read.

## Addendum (2026-09-07) — within one gather, decision 3's LRU serves hits first

The shards (blocks since 2026-09-22) one gather touches are grouped and read one at a time, **resident
ones first**: in first-touch order, eviction caused by a miss can drop a resident shard the gather has
not used yet (measured 2026-09-06: budget of two largest shards, shards 1 and 2 resident — order
`[0,1,2]` caused 3 loads, `[1,2,0]` caused 1). Hits first, the loads equal the misses (the minimum). A
shard taken during the gather is held by a local variable, so eviction midway changes no value.

Repeated ids (typically prefill's pad id 0) are dequantized once and the f32 bytes copied — a copy,
so bit-identical to decision 4. A cross-gather row cache was **not** added then: in real use (2 turns
× 200 tokens, default budget) shard re-reads were 137, about 42 s of a 58 s wall, and offline a row
cache only cut them to 71 (first-seen tokens still miss). The fix was row reads (next addendum).

## Addendum (2026-09-07) — alternative b ("read only the rows") adopted

Decision 3's planned alternative b is adopted: `Gemma4Ple` reads only the rows of a small gather
(for one decode id, 8,960 B of values and 140 B of scales), and range reads cause no LRU eviction
(the eviction loop over 9 shards was the cause of the 137 re-reads). Rows are built from the same
bytes with the same rounding, so decision 4 holds (confirmed by the torch comparison check). Hub
supports it through the fetch layer `@hdae/fetch-cache` (`openCachedUrl` / `openHfFile`) and ADR 0086
capability ⑧ (`openAsset`).

The shard-level handle (`Gemma4Assets.openPleShard`) and the policy table switching row reads and
whole reads by the reader's cost type (`"seek"` / `"scan"`, rows ≤ 2 / rows ≥ 32) are superseded by the
2026-09-22 addendum on 2026-09-22: the PLE consumer ignores the cost type (hub's
`AssetRangeReader.cost` remains, `packages/hub/src/fetch.ts`), and the whole-read threshold of 32 rows
applies per block (`FULL_BLOCK_ROWS` in `packages/models/src/gemma/ple.ts`).

## Addendum (2026-09-11) — values and scales of a small gather are read concurrently

With 8 unique rows or fewer, each row's values and scales ranges are read concurrently; above that,
16 workers read each row's two ranges in order. This removes a serial wait from the single decode row
while the whole gather keeps the limit of 16 concurrent reads, applied across blocks; row reads are
issued after the header is read.

Both ranges are awaited even if one is rejected. The row is written and `rowReads` incremented only
after both succeed; if both fail, the values-side cause is reported. Abort propagation, the budget,
range positions, dequantization and the copying of repeated rows are unchanged, as are the public
API, the storage format and the numerics contract; it applies to the loader shared by Gemma and QAT.
Measured 2026-09-11: QAT generation in Chrome over HTTP Range about 2–5% faster, Deno local reads the
same; not generalised to other sources or Apple M2
([research](../research/2026-09-10-codex-mtp-optimization.md), section of 2026-09-11).

## Addendum (2026-09-12) — quantized rows read by row reads are kept in a small LRU

The row cache declined on 2026-09-07 is adopted against a different cost: with row reads in place,
the small reads took 66–74 ms of prefill in the HTTP Range measurement, re-reading the same token's
values and scales every time ([cost breakdown](../research/2026-09-12-ple-row-cache.md)). It is added
to the PLE loader shared by Gemma and QAT.

- The quantized values and f32 scales of a successful row read are kept as is in an LRU of at most
  256 rows; no dequantized rows are kept, and hits go through the existing `writeRow`. Rounding order,
  static quantization and the token-sequence contract are unchanged. Row hits are used first within a
  gather, so a later miss's eviction cannot lose an unprocessed hit.
- The existing `maxResidentPleBytes` / `maxResidentBytes` budget is shared; whole blocks take
  priority and rows use only free space (`reserved block bytes + kept row bytes ≤ budget` at both
  reservation and storage). Budget 0 keeps no rows. Maximum payload for E2B: 2,329,600 B (regular),
  1,182,720 B (QAT). No new memory budget or CLI option.
- The internal diagnostic `residentBytes` covers both payloads; `resident` counts whole blocks,
  `rowReads` counts rows actually read (hits are not reads). Temporary buffers and bookkeeping stay
  outside.
- A row is kept only after values and scales both succeed; rejected rows and rows aborted or disposed
  during the read are not kept. The cache lives as long as the pipeline's PLE owner and is emptied on
  dispose. The whole-read criterion and the limit of 16 concurrent reads stay.
- First-seen reads remain. Speeds on repeated identical input are reported apart as an upper bound
  and compared with unseen English and Japanese input; M2 speed needs a re-run on the maintainer's
  machine.

Public arguments, graphs, stored assets and the numerics contract do not change; by default a little
unused budget is used, and the diagnostic includes resident row bytes. Budget, LRU, duplicates,
concurrent reads, abort, dispose and bit identity with the I2/I4 static references are checked on the
CPU; generated sequences and full verification of regular/QAT and E2B/E4B on a real GPU.

## Addendum (2026-09-19) — GPU residency: opt-in placement of the sidecar on the GPU, gather on the GPU

perf-ledger H-28. One **opt-in exception** to decision 3 (lazy loading + LRU) and decision 6 (an
ordinary graph input). The default stays `"host"` (the maintainer's ruling of 2026-09-19: "single
binding + raised binding limit + opt-in option").

- **Option:** `Gemma4PipelineOptions.pleResidency: "host" | "gpu"` (default `"host"`), for regular
  Gemma and QAT. Invalid values and incompatible options fail loudly before any asset is read.
- **Removes:** the per-run host dequantization and the writeBuffer of `per_layer_inputs` (35,840 B
  per decode token, about 0.25 ms; 27.5 MB for a 768-row prefill). Also a hard prerequisite of decode
  pre-issue (H-27 step ②).
- **Mechanism:** a two-node IR (`embedding` → `mul embed_scale`) in its own Session is **enqueued
  first in the target run's batch** and writes a resident tensor, which the target takes as the
  resident input `per_layer_inputs` (no writeBuffer). Temperature-0 decode rides the greedy output's
  batch (no extra fence). Decode with temperature > 0 or diagnostics is stacked on the same batch, and
  graph outputs are read back at its final fence (`Session.enqueueRead` —
  [0054 addendum](0054-resident-loop-and-fence.md#addendum-2026-09-20--batch-readback-of-graph-outputs),
  since 2026-09-20). Only prefill (M > 1) pays one extra gather batch, so that the enqueue path does
  not create chunk-shaped slot backing from the first chunk. Prefill and decode both use the GPU
  gather.
- **Weight layout:** `[tokens × layers, dim]`, index `id × layers + layer`. One row is one layer, so
  the sidecar's `scales[rows, layers]` is the row-scale sequence, and **i8 / i2 row scales and i4
  group scales with `group_size = dim` share the flat index**. Dequantization is one `f32(q) × scale`
  in `embedding`, then `embed_scale` in a separate node — decision 4's order and rounding points
  (checked by u32 equality with `ple.probe.safetensors`).
- **Loading:** superseded by the 2026-09-24 addendum on 2026-09-24 (current mechanism there).
- **Resident size and binding limits:** `values` is one unsplit buffer — QAT E2B (i4)
  1,174,405,120 B, QAT E4B (i2) 704,643,072 B, regular Gemma 4 E2B (i8) 2,348,810,240 B; `scales` (f32)
  36,700,160 B for E2B. `acquireGpu` requests the adapter's measured limits (`planRequiredLimits`);
  **at load the sizes are checked against `maxStorageBufferBindingSize` / `maxBufferSize` and fail
  loudly when short**, with no silent fallback to the host. The reference machine (RTX 3080 Ti /
  Vulkan) allows 2,147,483,644 B, so regular E2B's i8 sidecar cannot use this option (QAT fits).
- **Estimate:** resident PLE and the gather Session count in `auxiliaryBytes` of
  `estimateSessionMemory()` and in `peakAccountedBytes`.
- **Unsupported** (fail loudly): `maxResidentPleBytes` (not drawn), speculative decoding (drafter
  inputs out of scope), `gpuTiming` (no batch while measuring).
- **Acceptance:** `packages/e2e/tests/e2e_gemma4_ple_gpu_test.ts` checks ① u32 equality with the
  golden (torch's 35-table path), ② u32 equality with the host `ple.gather`, ③ 16-token greedy ids of
  QAT E2B equal to `"host"`; on the CPU, `packages/models/tests/gemma4_ple_gpu_test.ts` covers option
  acceptance, the gather IR's shape and the container cross-check. **No speed verdict was recorded**
  (Deno CLI decode included deno_webgpu's 10 ms/token floor; M2 / Chrome were not accepted).

## Addendum (2026-09-22) — the sidecar becomes container assets: index schema 3 and block lists

With the distribution in the container (`krm` — ADR [0108](0108-container-format.md) /
[0109](0109-manifest-v5-container.md) decision 4), the PLE sidecar left the manifest's `assets` and
became **assets of the model container**.

- **Index schema 3** (asset `ple_index`, role `ple-index`, canonical JSON):
  `{ schema: 3, storage: "i8" | "i4" | "i2", tokens, layers, dim, embedScale,
  values: { rowBytes, blocks: [{ asset, start, stop }, …] }, scales: { rowBytes, blocks: […] } }`.
  `values` and `scales` are **separate matrices**; half-open token ranges (ascending, gap-free, 0 to
  `tokens`) point at blocks (assets `ple.values.<k>` / `ple.scales.<k>`, roles `ple-values` /
  `ple-scales`). Shard boundaries are gone; the writer cuts blocks at or below the 32 MiB block limit
  **in whole rows**. `rowBytes` is `layers × dim / packFactor` (i8 = 1, i4 = 2, i2 = 4) and
  `layers × 4`.
- **Reads are per block** (`openBlock(asset) → AssetReader`, the container's `asset(name)`); token →
  (block, row offset) is a binary search plus `(token − start) × rowBytes`. Readers carry no cost
  type: a verified source fetches only the range; an unverified one verifies the block once and keeps
  it for the reader's lifetime (the runtime's `AssetReader`). What remains: decision 3's LRU
  (**counted in blocks**; default limit two of the largest blocks, about 64 MiB), the row LRU,
  `Gemma4PleStats` and GPU residency (each block uploaded whole).
- **Admission cross-checks the index against every asset** (each asset of `values.blocks` /
  `scales.blocks` exists with its role, with logical length `(stop − start) × rowBytes`). The index
  shares no part with weight blocks, so it is readable before any weight byte (container-v1 §4.2).
- A block that needs range reads is **one block per part** (container-v1 §4.2). gemma4 E2B has about
  72 `values` blocks plus a few `scales` blocks as parts (within the limit of 1024 parts).
- Only the migration CLI (`karume migrate --manifest`) reads the old sidecar (schema 1 / 2
  safetensors); the loader reads schema 3 only.

## Addendum (2026-09-24) — GPU-resident load mechanism and host RAM peak

The load mechanism of 2026-09-19 changed with ADR [0108](0108-container-format.md) stage 3d
(`9e905d9f`, old distribution readers removed) and stage 3e (block-by-block Session construction);
its terms (shard-sequential path, reused synthetic shard buffer, piece 1 trimmed into the graph shard)
no longer exist in the code.

- **Current path:** the index's `values` blocks are passed as the `pieces` of an **in-memory
  container** (`openMemoryContainer` — 0108 addendum 4, item 8). Session construction reads one piece
  at a time, `writeBuffer`s it at its row offset into one GPU buffer for the whole parent, and
  releases it in that iteration (ADR 0070 decision 3, 2026-09-24 addendum). The companion scale lives
  in piece 1's part (container-v1 §5, rule ③), so the `scales` blocks are read in full first and
  passed as one buffer.
- **Host RAM peak is "the whole scale table + the largest values block"** (E2B: 35 MiB + at most
  32 MiB). Scales count once: the in-memory container returns the scale buffer without copying
  (`readBlock` in `packages/runtime/src/format/container/memory.ts`), Session construction takes it as
  a view (`scaleTensor` in `packages/runtime/src/runtime/session-build.ts`), and `ple` is consumed only
  by `embedding`'s weight slot, so it loads compressed. Only the carried scale of an expanded weight's
  piece list copies scales, and PLE does not go through it. Accounting: [container-v1](../container-v1.md)
  §11.

## Addendum (2026-10-27) — the row cache keeps rows from whole blocks, as headroom outside the budget

perf-ledger H-41 (a defect in the row cache of the 2026-09-12 addendum,
[research 2026-10-25](../research/2026-10-25-speed-recon-3080ti.md) §9.1).

- **Defect:** a row was kept only when both its values and its scales came from row reads, and only
  while the budget had a row's worth of free space. With the E2B layout, the first gather with 32 or
  more distinct ids loads scales block 0 (91% of the vocabulary) whole, so from then on every row has
  its scales from a block view and is refused. Once a values block is whole too, the default budget
  (two of the largest blocks) leaves 8,120 B, less than one 9,100 B row. Decode then re-reads every
  row that is not in a resident block: on Node + HF decode medians were 20.4–21.2 ms/token, against
  10.3–10.6 ms/token measured with an equivalent fix in
  [research 2026-10-25](../research/2026-10-25-speed-recon-3080ti.md) §9.1 (not with this change),
  and local-directory sources re-read a whole 33.5 MB part per row miss (this change's own
  measurement is below).
- **Change 1 — rows from whole blocks are kept as copies.** When a row's values or scales come from
  a resident whole block, the row's slice (`rowBytes` of values, `layers` f32 of scales) is copied
  into its own buffer and kept like a row-read row. A view is never kept, so an evicted block is not
  kept alive by a row.
- **Change 2 — the row cache is headroom outside the block budget.** Its capacity is 256 rows on top
  of `maxResidentPleBytes`, not the budget's free space. Whole blocks are bounded by the budget alone,
  and reserving a block no longer evicts rows. Copying alone (change 1 without change 2) keeps
  nothing once a values block and the scales block are resident.
- **Memory bound:** resident blocks ≤ budget, plus at most 256 rows — 2,329,600 B for regular E2B and
  1,182,720 B for QAT E2B, the maximum payload the 2026-09-12 addendum already states. The internal
  diagnostic `residentBytes` covers both, so it can exceed the budget by at most that payload. Budget
  0 still keeps nothing resident, rows included.
- **Unchanged:** a row is kept only after both tables succeeded, and never from a gather that was
  aborted or ran into dispose; row-cache hits are served first in a gather; LRU order; dispose empties
  the cache. Numerics are unchanged: a kept row holds the same quantized bytes and scales the block
  holds, and every path goes through the same `writeRow`, so gathered values stay bit-identical
  (decision 4). Public arguments, graphs and stored assets do not change.

Checked on the CPU with synthetic indexes (`packages/models/tests/gemma4_ple_row_cache_test.ts`): a
scales block made resident by a wide prompt, a values block and the scales block resident with no
free budget, the 256-row LRU bound, budget 0, abort, a failed row read and dispose; reverting change 1
or change 2 fails those checks. The regular and QAT gemma4 lanes ran on a real GPU.

Measured on a local-directory source (RTX 3080 Ti, Node + Dawn, E2B default preset, 3 warm chats ×
128 tokens, 2 runs per arm): with a prompt of 32 or more distinct ids, warm decode medians 32.5–43.2 →
6.8 ms/token and warm TTFT 502–923 → 56–85 ms (PLE re-reads per warm chat 144 → 0); with a short
prompt of fewer than 32 distinct ids no block becomes resident and nothing changes (6.7–6.8 ms/token
in both arms). The generated text is identical in both arms.
