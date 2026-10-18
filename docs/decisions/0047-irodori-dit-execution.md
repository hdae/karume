# 0047: Irodori DiT execution form — B=1 with selective runs, uncond expressed by the mask

- Status: accepted (2026-08-11; the maintainer's ruling, as the recon recommended); partly
  superseded by [0114](0114-irodori-dit-context-split.md) (decision 3: the context-KV projection is a
  separate graph `dit_context`, 2026-09-27)
- Date: 2026-08-11
- Related: [0046](0046-cat-symbolic-axis.md) (symbolic-axis cat) /
  [0044](0044-runtime-attention-mask.md) (runtime bool mask) /
  [0042](0042-prepared-execution-plan.md) (prepared plans) /
  [0054](0054-resident-loop-and-fence.md) (resident tensors and the resident loop) /
  [0060](0060-row-block-attention.md) (row-block attention) / measurements:
  [research/2026-08-11-dit-export-recon.md](../research/2026-08-11-dit-export-recon.md)

## Context

Irodori's Euler loop has 40 steps; CFG applies only in the first half, t ∈ [0.5, 1.0]; the variants
are three independent uncond passes (text / speaker / caption) plus cond, four in total. Upstream
runs a forward with batch 4 and builds a second set of context KV "with the state set to 0" for the
uncond passes.

## Decision

### 1. uncond is expressed as "cond's context KV + an all-zero mask"

All three variants are **bit-identical to using cond's KV unchanged and zeroing only the mask**
(recon §3-1 — a masked element contributes exactly 0 through `exp(−inf) = 0`). The projection that
corresponds to G4 is therefore computed **for cond only**, and the four CFG variants differ only by
one bool mask. Upstream's "4 × batch context KV" (about 712 MB) is not held.

Input contract (fail loudly): only `speaker_uncond_mode="mask"` (default) and
`cfg_guidance_mode="independent"` are supported. `"noise"`, `joint` and `alternating` break this
equivalence and are rejected explicitly in the pipeline layer (`docs/limitations.md`, section
"Irodori DiT").

### 2. A B=1 graph with selective runs (no B=4 graph)

- Only the steps that need CFG run four forwards (about 100 forwards), 37% less compute than B=4 on
  every step (160 forwards).
- At design time the B=4 shape was heavy: the intermediate `scores[1,20,S,S+1519]` is 130 MB at
  S = 750 and 519 MB at B=4, and the symbolic-axis cat (option c of ADR 0046) would force a 4 ×
  expansion of the context KV (712 MB) at B=4.
- Cost at design time: a fixed run cost (about 38 ms/run, attributed on EmbeddingGemma) × 100 runs ≈
  3.8 s. (Since ADR 0054's resident loop, Irodori is almost GPU-bound: generation wall time 8.593 →
  4.896 s, all GPU about 4.2 s — ADR 0054 acceptance.)
- B=4 SHOULD NOT be revisited without a new measurement. (Re-graded 2026-10-18 to SHOULD: B=1 still
  saves 37% of the compute, but the allocation ceiling and the 4 × expansion that ruled out B=4 have
  changed shape — ADR 0060 splits the scores by row blocks, ADR 0114 makes the context K/V resident
  once per generation.)

### 3. G4 (context-KV pre-projection) is folded into G5 as one graph for now

(Superseded by ADR [0114](0114-irodori-dit-context-split.md) on 2026-09-27: G4 is exported as a
separate graph `dit_context`, run once per generation, and its 24 K/V outputs reach `dit` as
resident tensors. The inputs of `dit` are `x_t` / `t_embed` / `mask` and the 24 K/V tensors; the
three condition states moved to the inputs of `dit_context`.)

Original decision: the execution phase wrote graph inputs with `writeBuffer` on every run (no
Session-resident input values yet). A separate G4 graph would add an upload of its 178 MB output
(24 × [1,1519,20,64]) of about 30 ms per run as a new fixed cost (3.0 s over 100 runs — recon §4-2).
Folded, the inputs are the three condition states, 3.6 MB per run; paying the recomputation of
59.6 GFLOP/forward ≈ 0.56 s still **saves 2.4 s net**. Branch point (DECIDED: this ADR): split G4
once Session-resident input values exist. Until then G5' takes `x_t[1,S,32]` / `t_embed[1,512]` /
`mask bool[1,1,1,S+1519]` and the three condition states (text[1,256,512] / speaker[1,751,768] /
caption[1,512,512], right-padded to Tmax).

### 4. Left on the host

The Euler update (`x += v·Δt`, [1,S,32] ≤ 24k elements), the CFG combination
(`v + Σ scale·(v_cond − v_k)`), the generation of `t_embed` (three lines of sin/cos — `cos` is not
added to the vocabulary), the Tmax right-padding of the condition states, and building the segment
mask.

## Consequences

- Because of the condition padding, **bit identity with upstream's packed computation does not
  hold** (rel 7.5e-07 — SDPA picks its kernel by the reduction length, so this is unavoidable
  whatever the method; recon §1-6). The golden check uses "the exported graph run in torch" as its
  basis, so its strictness (tolerances derived from measurements starting at atol 0) is kept.
  Documents MUST NOT claim bit identity with upstream.
- The 130 MB `scores` at S = 750 exceeded WebGPU's default `maxStorageBufferBindingSize`
  (128 MiB). Resolved by the row-block execution of ADR 0060; the remaining limit is a single query
  row that does not fit the binding limit (`docs/limitations.md`, section "Irodori DiT").
- The U7 estimate of 2026-08-11 (about 2 s GPU time for a 10 s utterance, 6.5 s for 30 s, recon §6)
  is superseded by the measurements of ADR 0054.

## Addendum (2026-09-25) — the branch-point condition of decision 3 was met by ADR 0054

ADR 0054 (2026-08-13) provided the mechanism (ResidentTensor, batch enqueue, `copyOutputs`), so the
condition "Session-resident input values" of decision 3 was met; the decision whether to split was
deferred to a measurement and made by ADR 0114 (next addendum).

## Addendum (2026-09-27) — decision 3 replaced by ADR 0114

Decision 3 (fold G4 = the condition-side K/V projection into `dit` for now) was replaced by ADR
[0114](0114-irodori-dit-context-split.md). Decisions 1, 2 and 4 do not change.
