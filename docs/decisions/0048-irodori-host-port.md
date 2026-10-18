# 0048: Irodori host port contract — caption_norm placement, shared Unigram layer, latent exit

- Status: accepted (2026-08-11 — the maintainer approved the three axes of the phase-3 plan);
  partly superseded by [0049](0049-irodori-codec-integration.md) (decision 3: the latent-only
  exit, 2026-08-12)
- Date: 2026-08-12
- Related: [0047](0047-irodori-dit-execution.md) (DiT execution shape) /
  [0046](0046-cat-symbolic-axis.md) (symbolic-axis cat) / [0044](0044-runtime-attention-mask.md)
  (runtime bool mask) / [0038](0038-manifest-v1.md) (`pipelineConfig` ownership) /
  [0037](0037-karume-monorepo.md) (barrel plus subpath exports) /
  recon = [research/2026-08-11-irodori-host-recon.md](../research/2026-08-11-irodori-host-recon.md)

## Context

Phase 3 of the Irodori port is the host side: driving the 6 graphs exported in phases 1–2 from TS
up to a latent (tokenizer, normalisation, Euler + CFG, masks, choice of S). Three axes were open:
(1) the duration input `caption_vec` (masked mean of the caption_norm-ed sequence) needs the
`caption_norm` weights, which exist only inside the dit graph; (2) whether Irodori's Unigram
Viterbi is shared with anima's T5; (3) whether to emit the distribution before the codec exists.

## Decision

### 1. caption_norm is the second output of caption-proj (the weights stay inside a graph)

`caption-proj` gains a second output, the sequence after `caption_norm` (RMSNorm 512); only the
masked mean stays on the host. The first output (consumed by dit's `caption_state`) is unchanged
and matches the old golden bit for bit after re-export (maxabs 0.0).

- Rejected a) fold norm + masked mean into duration (G3): changes its 6-input contract and every
  golden.
- Rejected b) ship the caption_norm weights as a separate asset: trained values leave the model
  file, and the canonical path no longer closes over one file (as in ADR 0010).

caption_norm then lives in caption-proj and dit, as text_norm lives in duration and dit. A mix-up
(reading text_norm twice) keeps shape and dtype and the golden agrees with itself, so only the
export's standing check `_norm_divergence` catches it (`NORM_DIVERGENCE_MIN = 1e-3`, measured
0.318 — `tools/export-recipes/irodori/export.py:464`).

### 2. The Unigram core is a family-neutral shared layer in `src/text/`

The T5 Viterbi (a port of tokenizers' `Lattice::viterbi` with its tie rule and `fuse_unk`) lives in
`packages/models/src/text/unigram.ts`, used by anima and irodori. Byte fallback is the `byteBaseId`
option (unset = one unk; T5 behaviour unchanged, checked by the anima tests). `toCodePoints` and
`splitAddedTokens` moved to `src/text/` too.

- Rejected: a copy per family. A drift in the Viterbi tie rule changes token sequences
  **silently**; shared pipeline-independent code follows the `src/audio/` / `src/image/` precedent.
- Conversely, **Randn (splitmix64 + Box-Muller) stays duplicated** (`irodori/host/random.ts`): the
  random sequence determines a family's output, and sharing it would let one family silently move
  the other's results.

The parity fixture is tracked (`packages/models/tests/fixtures/irodori-text/parity.json`): a
vocabulary subset for the cases (1,222 of 102,400 entries today), with `minScore` /
`maxTokenLength` of the **whole** vocabulary carried explicitly (a subset would change unknown-node
weights and search width). NFKC is delegated to `String.prototype.normalize`; skew against Python
`unicodedata` is checked over **every code point** (4,964 single-code-point differences plus
identity both ways — `packages/models/tests/irodori_text_test.ts`).

### 3. The distribution is emitted in phase 3; the pipeline exit is a latent

The dist step gains an irodori target (today `tools/export-recipes/dist.py --pipeline irodori`,
ADR 0065) and the E2E tests read its output (as for sbv2 / anima: no test-only manifest).
(Superseded by ADR 0049 on 2026-08-12: the codec joined the same distribution with a waveform
exit; `generateLatent` remains as the latent-only entry — `irodori/pipeline.ts:978`.)

### 4. Model-specific numbers come from pipelineConfig; unsupported modes are rejected at parse time

Condition-state Tmax, speaker rows, latent width, the S clamp range and the CFG defaults come from
the manifest's `pipelineConfig` (hand-written schema; 23 fields since phase 4 — `ROOT_KEYS` in
`packages/models/src/irodori/config.ts`). Hard-coded TS values would go silently wrong **with
matching shapes** after a weight swap. `fromAssets` checks the graph declarations before it
acquires the GPU (`irodori/admission.ts`). `speakerUncondMode` / `cfgGuidanceMode` accept one value
each ("mask" / "independent"); others are **rejected at parse time** (ADR 0047 decision 1).

### 5. Upstream-compatible seeds are a non-goal; torch comparison injects initialNoise

torch's `Generator` (CPU MT19937 + Box-Muller; CUDA Philox) is not reproduced. Generation is
deterministic per Karume Randn seed. Comparisons that need upstream's random sequence (the
full-loop golden) inject it through `IrodoriGenerateRequest.initialNoise`, a public entry
documented as being for reproduction and verification.

## Consequences

- Numerical identity rests on **golden comparison** (per-graph E2E checks and the full-loop latent
  check), not bit equality. Known differences: t schedule (closed form vs linspace) 5.96e-8;
  t_embed 3.05e-5 (a 1-ulp difference between torch f32 exp and `Math.exp`, grown at args ≈ 284 —
  SLEEF is not ported, as in anima's `timestepsProj`); full-loop z 1.9e-4 to 7.9e-4 (DiT tolerance
  over 40 steps). Since 2026-10-17 the full-loop comparison is opt-in (ADR 0129 decision 4).
- `durationSeconds` follows upstream (truncate to samples, ceil to frames — `host/round.ts`); the
  earlier `frameRate` route could be one frame off and was replaced on 2026-08-12 (ADR 0049).
- Host-side formulas (empty-caption / no-reference zero short-circuit, mean-token prefix, banker's
  rounding) live in `packages/models/src/irodori/host/` (pure functions, sequential
  `Math.fround`); the exporter goldens (`tools/export-recipes/irodori/tokenizer_ref.py` /
  `pipeline_ref.py`) pin the values.
- **Addendum 2026-08-13: caption preprocessing is strip only**, as in upstream `_synthesize`
  (`normalize_text` applies to the text only); the port had normalised the caption too, silently
  changing the conditioning (fixed in `packCaptionIds` / `_packed_caption_ids`). The Python parity
  shared that preprocessing and was tautological, so the upstream comparison now builds the
  caption condition **independently from upstream's entry** (`caption_tokenizer.batch_encode`,
  `pipeline_ref.py:605-628`). Six normalisation-sensitive caption fixtures check that the strip and
  normalise paths give **different** ids. NOTE: the text-side comparison still passes the
  implementation's sequence (known asymmetry; making it independent is a follow-up candidate).
