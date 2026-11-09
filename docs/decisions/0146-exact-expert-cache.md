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

1. A RAM tier. A cached Blob read measured about 2.9 ms per expert in Chrome on the RTX 5070 Ti (research §6.1, 2026-11-09),
   about 36 ms per token at K = 64; the host store is therefore one narrow interface, so a tier in RAM can be put in front of
   the Blob once the counters of a real chat are in (the maintainer's ruling of 2026-11-09: build scheme A on the Blob
   first).
2. A per-layer K (the research's fit saves about 12% at K = 64).
3. Prefill speed with small chunks, and whether a grouped prefill form (ADR 0145 open point 1) changes it.
4. The real per-forward time of 26B decode in Chrome, which sets the absolute cost of every re-run.
