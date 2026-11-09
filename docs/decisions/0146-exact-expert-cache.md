# 0146: An exact expert cache for MoE decode (Gemma 4 26B A4B first)

- Status: proposed (2026-11-09 — the maintainer approved the staged plan: a transfer measurement, this record, the whole-token
  re-run (scheme A), then self-speculation over the cache (scheme B))
- Date: 2026-11-09
- Related: ADR [0145](0145-gemma4-26b-moe-resident.md) (every expert resident; `expert_linear`) /
  [0066](0066-generation-context-state-slots.md) (state slots; the logical length advances on commit) /
  [0096](0096-speculative-decoding.md) (deferred commit, `commit(rows)`, the verify shape) /
  [0061](0061-index-oob-semantics.md) (index semantics at the host boundary) / [0086](0086-distribution-source.md) (range reads
  of a source) / [0014](0014-layout-ops-full-write.md) (every output fully written)
- Measurements: [research 2026-11-08](../research/2026-11-08-gemma4-26b-moe-cache.md) (routing locality, re-run counts,
  self-speculation, Chrome fence and upload costs on the RTX 5070 Ti)

## Context

26B keeps all 128 experts of its 30 layers in VRAM: about 14.1 GB of the 16 GB card, and it does not run on a 12 GB one
(ADR 0145 addendum). Its decode routing has strong locality: with an LRU of K = 64 experts per layer the per-pick hit rate is
0.96, but a token makes 240 picks, so most tokens miss somewhere ([research](../research/2026-11-08-gemma4-26b-moe-cache.md) §3).
The CPU study of the research found an exact cache — output bit-identical to full residency — costs re-run forwards, not
transfer: 2.41 forwards per token at K = 64 with whole-token re-runs, about 1.85 with self-speculation over the cache, and
Chrome's measured fence (about 2.5 ms) and staging-ring upload (about 0.43 ms per 3.19 MiB expert) leave those costs within
1 ms (§6.1). It is about 2.3–2.4 times the time of full residency, at about 8.1 GiB instead of 14.1.

The maintainer wants it built even if it is slow (ruling 2026-11-09): if it works, a model whose experts do not fit in VRAM
at all — larger than 26B — becomes possible in principle. The constraint "MoE assumes every expert is resident" of
[limitations](../limitations.md) records why this was out of reach: the container binds every initializer, the weights are
assembled once per Session, and no op skips or transfers by value. `expert_linear` removed the third for fixed-k compute; this
record removes the second for experts only, without touching the first (the distribution is unchanged).

The code facts this rests on (read 2026-11-09):

- `expert_linear` (ADR 0145 decision 2) addresses `weight[index[t, j]]` in its GEMV kernel; the index must be a `topk` output
  whose reduced dimension equals the weight's E (`runtime/plan.ts`).
- A deferred run does not advance the logical length; `commit(0)` accepts nothing, after which the same position can run again
  (`generation-context.ts`). Deferred runs are bounded by `queryLength ≤ slidingSlack` (8 for Gemma 4): the sliding ring has
  capacity window + 8, so up to 8 rows can be rewritten without overwriting KV still inside the window (`executor.ts`).
- 26B's exported graph already has the multi-row exit (`last_row [R]`, R ≤ 9) of 12B, so the verify shape needs no re-export.
- One expert is four contiguous ranges of the container (gate_up codes and steps, down codes and steps, 3,345,408 bytes); no
  piece splits an expert. The hub's cached source reads a range as a `slice` of the cached Blob (cost `"seek"`).
- The default sampler of 26B is temperature 1 with top-k / top-p: a token drawn from the logits of a run that missed would
  consume the sampler's RNG and diverge from full residency.

## Decision

1. **Opt-in, exact, and refused elsewhere.** A pipeline option `expertCache: { slots: K }` (the same K for every layer to
   start). Without it nothing changes: no new binding, no new readback, byte-identical WGSL, every sha row as it is. With it the
   output MUST be bit-identical to full residency (the same tokens, the same logits SHA-256), so a miss is never computed
   approximately. The option is refused with `decodeAhead ≥ 2`, with a drafter's `speculative`, and on a source whose range
   reads are not `"seek"` (a scan per miss reads up to a whole part). An approximate mode is later, separate work with its own
   opt-in (the "no silent approximation" invariant).

2. **The pool and the slot table are the runtime's.** When the option is set, the Session allocates for every
   `expert_linear` weight a pool of K experts (`[K, out, in]` codes and the matching steps) instead of `[E, out, in]`, uploads
   no expert rows at build, and keeps per layer a table `i32[E]`, expert id → slot, −1 for not resident, in a GPU buffer the
   kernel binds. The two projections of a layer share one table. The kernel reads `slot = table[index[t, j]]`; a slot of −1
   writes 0 for that pick (the full-write rule holds), any other slot addresses the pool exactly where the resident kernel
   addressed the full weight, so a row whose picks are all resident runs the resident arithmetic bit for bit. The planner's
   rule becomes: the index is a `topk` whose reduced dimension equals the table's length E (the pool's first dimension is K).
   The table values are written only by the runtime, so the index stays in range by construction (ADR 0061).

3. **The routing reaches the host on the run's own fence.** Under the option a run also returns every layer's `topk` ids of
   its real rows (`[L, R, k]` i32, 7.5 KiB for one decode row), read in the same command list and the same fence as the
   outputs. The host compares them with its tables to find the misses; the kernel records nothing (no atomics, no per-run
   clear). Padded rows of a bucket are ignored.

4. **Commit protocol (scheme A).**
   - Decode: run the token deferred; if any pick of the row missed, `commit(0)`, load the missing experts, and run the same
     inputs again; when no pick missed, `commit(1)` and only then sample. A miss never consumes the sampler's RNG.
   - Prefill: chunks of at most `min(slidingSlack, ⌊K / top_k⌋)` rows (8 at K = 64), each run deferred with the same retry and
     committed as a whole. Every chunk's picks fit in K slots at once, so a retry always converges. Long prompts get slow; that
     is accepted for now.
   - A run that throws still poisons the context as today; a miss is data, not an exception.

5. **Host store: the distribution itself.** A miss reads its four ranges from the container through the hub's range reader
   (on the cached source, the browser's CacheStorage), at locations the Session derives once from the container descriptor
   (pieces, blocks, scale blocks — no piece size is assumed). The bytes go through a staging ring of `MAP_WRITE | COPY_SRC`
   slots, mapped ahead, and `copyBufferToBuffer` into the pool slot; the ring measured about 3× the rate of `writeBuffer` in
   Chrome. Loads and table writes run between runs, under the Session's run lock, and the next run is submitted after them on
   the same queue. A RAM tier for evicted experts is later work, to be decided on the measured read time of the cached Blob.

6. **Eviction: per-layer LRU over committed rows.** The recency is updated from the picks of committed rows only; the experts
   a pending retry needs are pinned while it loads. A row's picks number at most top_k ≤ K, so a load always finds a victim.

7. **Checks.**
   - This machine's GPU (12 GB) runs the 12-layer cut model of ADR 0145 with the option at small K (16 and 64): greedy text
     and every logits SHA-256 equal to the same model with every expert resident, with the counters showing misses and
     re-runs (a run with no miss would prove nothing).
   - Fault injection: skipping the re-run (committing a missed row) MUST make the check fail.
   - Without the option, the full verify's sha rows are unchanged.
   - The full model runs in Chrome on the RTX 5070 Ti at K = 64 in a chat; the record carries the miss, retry and load
     counters, and the maintainer reads dedicated GPU memory in the Task Manager.

8. **Self-speculation (scheme B) follows on the same pieces.** Draft `depth` tokens on the cache as it is (misses contribute
   0), load what the drafts missed, verify `depth + 1` rows in one deferred run, and commit the rows before the first row whose
   picks miss. It is recorded in an addendum once scheme A holds; only the verify and the draft loop are new.

## Consequences

- `expert_linear` gains an optional table binding and a second planner form; the Session gains an expert pool, its tables, a
  routing readback and a load entry. The default path does not see any of them.
- Gemma's decode loop gains a deferred retry and the "sample after commit" order under the option; prefill gains small
  deferred chunks.
- VRAM for the experts becomes K / E of today (64 / 128 halves 12.85 GB); the dense parts, KV and embedding are unchanged.
- A cached distribution has to stay in CacheStorage while the pipeline lives (the reads come from it).

## Alternatives considered

- **Approximate cache (skip a miss, renormalise)**: one forward per token and about 0.02 nats of KL at K = 64, but not exact;
  kept as a later opt-in, not the first form.
- **A miss bitmask written by the kernel**: needs atomics and a clear per run, where returning the ids costs 7.5 KiB on a
  fence the run pays anyway.
- **Rewriting the graph so the table is a graph input**: the table would be the host's to write per run and the index rule
  would have to admit a non-`topk` producer, against ADR 0145's host-boundary argument.
- **Every expert resident during prefill, evicted after**: the peak VRAM is today's, which defeats the purpose and needs
  mutable weights anyway.
- **All experts in RAM**: 12.85 GB of ArrayBuffers in a tab (at least six, each under Chromium's 2 GiB cap) with no recorded
  renderer limit; the cached Blob is the store until a measurement says otherwise.

## Open points

1. A RAM tier — settled by the addendum of 2026-11-10 (a cached Blob read measured about 2.9 ms per expert in Chrome on
   the RTX 5070 Ti, research §6.1, and the first chat spent 77% of its time on those reads, §6.2).
2. A per-layer K (the research's fit saves about 12% at K = 64).
3. Prefill speed with small chunks, and whether a grouped prefill form (ADR 0145 open point 1) changes it.
4. The real per-forward time of 26B decode in Chrome, which sets the absolute cost of every re-run.
5. The hub opens a container as a plain block source, which states no range cost, so the refusal of a `"scan"` source
   (decision 1) does not fire through `fromPretrained`: on Node or Deno over Hugging Face a miss may read up to a whole part.
   Refusing it at load needs the hub to report the cost.

## Addendum 2026-11-09: scheme A as built

Scheme A is implemented (runtime `SessionOptions.expertCache`, Gemma `PipelineOptions.expertCache`, the Playground chat
page). Where the build differs from the decisions above, or settles what they left open:

- **Upload by `writeBuffer`, not the staging ring** (decision 5). A cached Blob read (about 2.9 ms per expert) is about seven
  times the ring's upload (0.43 ms), so the ring's advantage over `writeBuffer` is a small share of a miss; the simpler
  path was taken and the ring stays available if a RAM tier makes reads cheap.
- **The cache is a Session face of its own.** With the option, `Session.run`, `enqueue` and `enqueueRead` are refused; runs
  go through `Session.expertCache.run`, which must be a deferred generation run and cannot be chained. `load` and `touch`
  are separate calls, so the re-run protocol is the caller's: `@karume/models` owns it in one generic face
  (`generation/expert-cache-session.ts`), serialized per Session so that two conversations cannot evict each other's
  experts between a load and its re-run. A run converges in at most one attempt per routed layer plus one (the first
  missed layer moves strictly forward, since every earlier layer ran on resident experts); a further attempt fails loudly.
- **The routing is appended to the graph's outputs** (decision 3): every `topk` feeding an `expert_linear` becomes an extra
  output of the session graph, read back on the run's own fence. The kernel still records nothing.
- **The slot table rides on binding 6 of the pooled form of `expert_linear`.** A slot of −1 writes 0; an expert id or slot
  out of range writes the NaN of ADR 0061 (it cannot happen while the runtime alone writes the table).
- **Range reads come from the bound container** (`BoundContainer.ranges`: `verified`, `cost(block)`, `read(block, offset,
  length)`). On a source that does not verify its reads, the build takes the SHA-256 of every expert range from the
  verified block bytes and each load checks its bytes against it, so a miss never uploads bytes the build did not verify.
- **The build still reads every block once**, experts included (to verify them and take the digests), but uploads no
  expert row: the load time stays that of full residency, the VRAM is that of the pools.
- **Gemma refuses more than decision 1 lists**: `pleResidency: "gpu"` (its decode is enqueued in a batch) and `vision` (an
  image's bidirectional rows need an immediate run), both before any weight part is fetched; `slots` must be from 2 × k
  to E. Under the option decode samples on the host (no GPU argmax).
- **Checks** (decision 7): on the RTX 3080 Ti the 12-layer cut model at 16 and 64 slots (prefill chunks of 2 and 8 rows)
  gives teacher-forced logits and hidden rows word-identical to a resident Session at the same chunk rows, and 8 greedy
  tokens per golden prompt equal to the pipeline without the cache, with misses, loads and re-runs counted. Committing a
  missed run instead of re-running it fails every one of those checks. The Chrome chat on the RTX 5070 Ti is the
  maintainer's.

## Addendum 2026-11-10: a host tier in RAM

The first chat in Chrome (research [§6.2](../research/2026-11-08-gemma4-26b-moe-cache.md)) spent 77% of its time reading
misses from the cached Blob (3.77 ms per expert), which settles open point 1. The maintainer's ruling: the host store is
tiered — the distribution (CacheStorage) → RAM → the VRAM pool — so that larger models and devices with less RAM fit later;
holding every expert in RAM is fine for now.

- **The RAM tier holds plain `ArrayBuffer`s** (one per expert of a layer) under a byte budget `hostBytes`, filled at build
  from the verified block bytes the build already reads, LRU over every held expert; a load takes what the tier holds and
  reads the rest from the container (with the digest check on unverified sources) into the tier. Without `hostBytes`
  there is no tier and every miss reads the container, as in scheme A.
- **Uploads go through the staging ring of decision 5** after all: with reads from RAM the upload is the cost, and the ring
  measured 0.52 ms per expert against 1.12 ms for `writeBuffer`.
- **Not taken now: upload buffers as the tier** (`MAP_WRITE | COPY_SRC` buffers filled at creation, copied on the GPU).
  Their hit is eight times cheaper (0.064 ms), but Chrome appeared to hold about twice their size in RAM until garbage
  collection, and refilling one within a budget needs a `mapAsync` round trip. The tier sits behind one internal
  interface, so this form can be added as a whole-model fast path if the measured gap matters.

As built (`expert-host-tier.ts`, `expert-staging-ring.ts`):

- An entry is one expert's ranges of every pooled weight of its layer, contiguous (codes then steps per weight), so a held
  expert is one copy into a ring slot. The ring has up to 32 slots and at most 128 MiB (32 experts of 26B, about 107 MB); a
  load larger than the ring runs in rounds, and each round makes its experts resident on its own (its tables follow its
  copies), so a read that fails in a later round leaves the earlier ones valid. A slot is written again only after its
  `mapAsync` resolved, which WebGPU settles after the copies submitted before it.
- `ExpertLoad` and `ExpertCacheStats` split the experts by source (host tier, container) and the time into reads and
  uploads, and report what the tier holds; `bytesRead` stays the bytes read from the container. Dispose clears the tier.
- On the 26B cut model (12 layers, RTX 3080 Ti) a tier of every expert and a quarter tier keep the golden rows and the
  greedy tokens byte-identical, with the same misses, loads and re-runs as without a tier. The pipeline at 64 slots spent
  9.8 s in loads without the tier and 1.8 s with every expert held.
