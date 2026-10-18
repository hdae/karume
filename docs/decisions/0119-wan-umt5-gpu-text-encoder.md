# 0119: Run Wan's umT5-XXL text encoder on the GPU — accept free prompts (ADR 0118 step 10)

- Status: accepted (2026-10-03 — the maintainer's rulings on the distribution layout, the acceptance
  set and `signal`; 2026-10-04 — the visual review of the i8 umT5 found no problem); partly superseded
  by [0122](0122-umt5-upstream-and-compatible-encoders.md) (umT5 provenance, 2026-10-04) and by
  [0129](0129-test-layout-and-lanes.md) (decision 8 ①, the port check, is opt-in since 2026-10-17).
  Decision 6 was superseded by the step 10b addendum (2026-10-03); decision 10's working rules were
  withdrawn on 2026-10-18. Step 10e is not started.
- Date: 2026-10-03
- Related: research [2026-10-03-umt5-encoder-recon](../research/2026-10-03-umt5-encoder-recon.md)
  ("investigation §n" below — source of the facts and numbers) /
  [2026-10-02-video-gen-recon](../research/2026-10-02-video-gen-recon.md) (previous investigation) /
  [2026-10-02-b570-vram-budget](../research/2026-10-02-b570-vram-budget.md) (B570 ceiling) /
  [2026-10-03-wan-w8a8-recon](../research/2026-10-03-wan-w8a8-recon.md) (overlap in the exporter core) /
  ADR [0118](0118-wan21-video-generation.md) (decision 4's "second stage", decision 7's distribution,
  decision 8's numeric comparison, acceptance table, addenda) /
  [0026](0026-w8a8-deberta-deployment.md) (DeBERTa i8 goldens use fake-quant weights) /
  [0085](0085-ple-host-gather.md) (PLE host gather) / [0106](0106-device-keyed-references.md) (sha256
  reference rows per environment and the reference guard) / [0034](0034-dit-dynamic-tokens.md) +
  [0013](0013-sbv2-chain-export.md) (promoting tables to inputs) / [0045](0045-deberta-layer-trim.md)
  (byte identity of the relative-position index table in Python and TS) /
  [0019](0019-i8-weight-execution.md) (i8 weight execution) / [0009](0009-dtype-i32-bool.md) (i64
  boundary) / [0092](0092-distribution-repos-and-sources.md) (repository granularity and naming) /
  [0109](0109-manifest-v5-container.md) (`karume/5` and cross-repository references) /
  [0107](0107-model-input-error.md) (rejections caused by the input) /
  [0083](0083-generation-api-surface.md) decision 5 (shape of cancellation) /
  [0112](0112-anima-transformer-residency.md) (one Session per stage) /
  [0065](0065-exporter-core-recipe-split.md) (exporter core vs recipe).

## Context

### Need and position

ADR 0118 staged the text encoder of Wan2.1 T2V 1.3B (decision 4). The first stage is a precomputed
embedding asset: `generate({ prompt })` accepts only the four fixed prompts in the asset (three
positive, one negative). This ADR designs the second stage (ADR 0118 step 10): run umT5-XXL on the GPU
and accept free prompts. ADR 0118 decision 4 fixed its premises — i8 per-channel weights, f32
activations, the decomposed path for the relative-position bias, and measuring the export's host RAM
first. This ADR keeps them and changes only where the bucket table lives (decision 3).

### Terms

- **umT5-XXL**: the encoder of multilingual T5; Wan's text encoder. 5,680,910,336 parameters
  (vocabulary embedding 1.05 B + 24 layers 4.63 B — investigation §1.2).
- **i8 per-channel**: weights only as 8-bit integers, one scale per output channel (per row for the
  embedding; ADR 0019). Compute is f32. Activation quantization (a8) is not used here.
- **i4 g32**: 4-bit weights, one scale per 32 elements.
- **fake-quant**: rounding weights onto the quantization grid and keeping them as f32; used to take
  references with the same rounded weights.
- **Unigram**: the SentencePiece segmentation that picks, by Viterbi, the split with the highest summed
  piece score. Characters the vocabulary cannot cover become unknown nodes (unk). `spiece.model` is
  the source file.
- **`prompt_clean`**: the upstream preprocessing before the tokenizer (diffusers / official Wan2.1):
  ftfy `fix_text` → HTML entity unescape twice → whitespace collapse (investigation §3.1).
- **mojibake**: UTF-8 read in another encoding (`cafÃ©`). ftfy judges it with the regular expression
  `BADNESS_RE` (4,050 characters) and, when true, re-reads with several codecs (a heuristic).
- **NFC**: Unicode normalization form C; its result depends on the Unicode version (UCD).
- **C1 control characters**: U+0080–009F. **Lone surrogates**: unpaired U+D800–DFFF.
- **Relative-position bias / bucket**: T5 attention maps the position difference into 32 buckets and
  adds the value from a per-layer table `[32, 64]` (bucket × head). umT5 has one table per layer, all 24.
- **Promoting to an input**: the host builds a shape-dependent table and passes it as a graph input
  instead of baking it as a constant (ADR 0034 decision 2).
- **Valid length L**: token count of the prompt including the trailing `</s>`. Upstream pads to 512 and
  masks.
- **Layer-sequential reference**: a CPU reference that reads weights one layer at a time.
- **Host gather**: the host reads only the used rows of the vocabulary embedding, dequantizes them and
  passes them as an input (ADR 0085).

### Findings of the investigation

- **Size** (§1.2, §5): weights per storage format are bf16 10.58 GiB (computed), i8 5.30 GiB
  (5,686,617,600 B), i4 g32 3.31 GiB.
  The i8 vocabulary embedding is 1,051,174,400 B (1,002 MiB). Upstream stores F32, `total_size`
  22,723,641,344 B.
- **Activations and compute** (§1.3): at L = 512 the per-layer intermediates stay under a few hundred
  MiB; weights dominate. 24 layers cost 0.587 T MAC at L = 126 and 2.42 T at L = 512.
- **Ops** (§4.2): the existing vocabulary suffices except the hand-written `gelu_new`. Attention has no
  scale and a per-head bias `[1,64,L,L]`, so it does not enter fused attention (mask limited to
  `[1,1,M,N]`) and runs on the decomposed path (bmm → add → softmax → bmm).
- **Three tokenizer definitions disagree** (§1.4): out-of-vocabulary characters map to id 2 (`<s>` in
  umT5) in the pinned transformers 5.14.1 and to id 3 (`<unk>`) in `tokenizer.json` and transformers
  4.57.1. The four fixed prompts give the same ids in both (lengths 28 / 118 / 50 / 126). 19,848
  assigned characters are single vocabulary pieces.
- **Preprocessing** (§3.2, §3.3): `fix_text` adds a mojibake repair (heuristic) and entity unescape to
  table-driven steps and NFC. On 200,000 random strings from a pool without `&` and the 449 characters
  the mojibake test looks at, a mirror of the deterministic steps matched `prompt_clean` in every case;
  with those 449 characters, 277 of 50,000 (0.55%) differed.
- **Valid length only** (§2.3): on a small random model, "pad to 512 + mask" and "valid length only"
  agree within 5.7e-7 relative in f32.
- **Bucket table** (§4.3): torch f32 and f64 `math.log` agree for every distance −511…511.
- **Tokenizer licence**: `apache-2.0` (google/umt5-xxl cardData). This closes the "licence notice" open
  point of ADR 0118.

### Constraints

- Cross-cutting invariants: fail loudly on anything unsupported; a tolerance on the sha256 regression
  check is forbidden (ADR 0106); runtime dependencies are Web standard APIs only (neither ftfy nor
  Python's UCD can enter the runtime).
- The development machine of the time (Intel Arc B570) had a Deno allocation ceiling of 9,600 MiB,
  `maxStorageBufferBindingSize` 2,147,483,644 B and 31 GiB of host RAM. Browser WebGPU defaults to
  128 MiB per binding.

## Decision

### 1. Tokenizer — generalize Anima's T5 implementation and reject out-of-vocabulary characters

Protects: the same prompt yields the same ids as references made with any upstream version.

- Anima's T5 tokenizer is generalized into `packages/models/src/text/t5-tokenizer.ts` (family
  differences in `T5Policy`) and fed the umT5 tables (vocabulary 256,300, 304 added tokens). The path
  is that of transformers 5.14.1: WhitespaceSplit → Metaspace → Unigram → trailing `</s>`.
- **Out-of-vocabulary characters fail loudly**: one unknown node raises `ModelInputError` (ADR 0107).
  The accepted characters are the 19,848 single-piece assigned characters (CJK, kana, Latin).
- Why: the unknown id depends on the version (id 2 in transformers 5, id 3 in `tokenizer.json` and
  4.x). Rejecting them makes the ids independent of the version.
- The id sequence of record is the behaviour of the pinned transformers 5.14.1. Anima's parity fixture
  stays unchanged.
- Amended by the step 10a addendum: added tokens in the body and `▁` right after whitespace are also
  rejected. The acceptance rules live in `packages/models/src/umt5/tokenizer.ts`, shared by Wan and
  `Umt5Encoder` (`e033c3f0`).

### 2. Preprocessing `prompt_clean` — port only the deterministic steps as tables and reject the rest

Protects: ordinary prompts (accented Latin, CJK, full-width forms, curly quotes, newlines) become the
same string as upstream, and the upstream heuristic repair is never silently approximated.

- The deterministic steps of ftfy 6.3.1 `fix_text` (nine table-driven steps and NFC) run to a fixed
  point, then `strip` → collapse whitespace (the 25 characters of `regex` `\s`) to one U+0020 →
  `strip`. The recipe bakes the tables from ftfy (the same way as Anima's normalizer tables); TS only
  reads them.
- **Rejected inputs** (all `ModelInputError`): inputs for which the TS translation of `BADNESS_RE` is
  true; HTML entity candidates; C1 control characters and lone surrogates; unassigned code points
  (step 10a addendum).
- The JS translation of `BADNESS_RE` is bound by a fuzz parity test. NFC follows Python UCD 16.0.0 and
  is checked against JS `normalize("NFC")`.
- Why: the deterministic part can be compared with the mirror exhaustively and by sampling; the
  heuristic part can be reduced to "reject when true", which keeps the accepted set wide (the
  official negative's full-width comma, curly quotes).

### 3. Relative-position buckets — the host builds an `[L, L]` i32 index table and passes it as an input

Protects: no constant baked per L (as ADR 0034 decision 2); bucket boundaries never differ between
Python and TS.

- The host builds the `[L, L]` i32 bucket indices; the graph looks up each layer's `[32, 64]` table
  with `embedding` (`[L, L, 64]`), permutes to `[1, 64, L, L]` and adds it to the scores.
- This replaces ADR 0118 decision 4's "`[512,512]` i32 constant (1 MiB) + per-layer gather".
- A byte-identity test of the Python and TS generators exists apart from the E2E test (precedent
  `sbv2_rel_pos_parity_test.ts`), so the host and the golden cannot share an error unnoticed.
- Why: SBV2 and Anima set the precedent of promoted tables; a constant needs a `[:L, :L]` slice under
  the dynamic L of decision 4; the boundary values agree between f64 and torch f32 over ±511.

### 4. Sequence length — dynamic valid length L, no mask input

Protects: compute proportional to L (the 126-token negative costs 24% of 512); no silent upstream
truncation.

- Inputs are token ids `[1, L]` and the bucket table `[L, L]`; L is one symbolic dimension. There is
  no mask input. The host pads the `[L, 4096]` output with zeros to 512 rows for the DiT (the DiT graph
  is unchanged).
- **Lower bound**: L = 1 (an empty or whitespace-only prompt, only `</s>`) is rejected; the minimum is
  2 because `torch.export` specializes 0 and 1 (precedent: Anima's `PROMPT_MIN_TOKENS`). The bound
  comes from that tool behaviour; revisit it when torch is upgraded.
- **Upper bound**: more than 512 tokens fails loudly. Upstream truncates silently with
  `truncation=True`; Karume does not (same as `valid_length` in `tools/export-recipes/wan/text_embeds.py`).
- Token ids are normalized to i32 at the exporter boundary (ADR 0009).
- Equality of "valid length only" and "512 + mask" on the real model was confirmed in f32 (step 10b
  addendum).

### 5. Weight storage and placement of the vocabulary embedding

Protects: the text stage fits alone; no host-side machinery is built ahead of the browser limits.

- **Storage**: i8 per-channel, f32 activations. i4 g32 would be a separate preset added after its
  quality is measured (step 10e).
- **The 24 relative-position tables `[32, 64]` stay F32.**
- **Vocabulary embedding**: starts as the GPU `embedding` op (one i8 buffer of about 1,002 MiB). A
  host gather (ADR 0085 precedent) is decided from the measured browser limits.
- **Output dtype**: the GPU path hands f32 to the DiT. It does not match the precomputed asset (bf16
  values), so its sha256 rows use separate case ids (decision 8).
- Why: the GPU `embedding` needs only existing ops; the PLE host-gather machinery is gemma-specific.
- State on 2026-10-18: every Wan preset declares `requiredLimits` of 1,050,148,864 B for
  `maxBufferSize` and `maxStorageBufferBindingSize`, the precomputed path included (step 10d-1
  addendum). Chrome was measured only on the precomputed path (RTX 5070 Ti, binding limit 2 GiB — ADR
  0118 step 9); the GPU path has not run in Chrome. With browsers as the primary target (ADR 0127),
  this limit also binds users who never run umT5, which is material for the priority of step 10e.

### 6. Host RAM of the export — measure first

Protects: the export completes on the development machine; one graph keeps the Session and admission
simple.

- Step 10b measures the peak RSS of a whole-model export. If it does not fit, fake-tensor tracing with
  sequential weight emission is added to the exporter core. **(superseded by the step 10b addendum on
  2026-10-03: no exporter core change — trace on a meta-device upstream, quantize row blocks from the
  checkpoint and pass them as `fixed_weights`; peak 6.79 GiB. The 31 GiB host was the development
  machine of the time.)**
- **No split into per-layer graphs.** It multiplies graphs and Sessions and complicates the text
  stage's admission.

### 7. Default path and the precomputed asset

Protects: the path never changes silently for the same string; a light mode without umT5 remains.

- The GPU path is the default. The precomputed asset path stays as an explicit choice
  (`textEncoder: "precomputed"`, step 10d design addendum B).
- **No automatic switch** (asset when the prompt is in the asset, GPU otherwise): the embedding of the
  same string would depend on the asset.
- **On the GPU path both positive and negative are encoded on the GPU.** Asset bf16 and GPU i8
  embeddings are never mixed in one CFG.

### 8. Numeric checks — a port check and a quality record

Protects: "the port is correct" can be stated with a band; the i8 quality is judged by observation and
visual review, not by a band without grounds.

1. **Port check**: the reference is a CPU layer-sequential reference (f32 or f64) with the same i8
   fake-quant weights; the band is "worst decision case × 5", accepted on separate cases (as ADR 0118
   decision 8; fake-quant goldens as ADR 0026 decision 2). **(superseded by ADR 0129 addendum (2) on
   2026-10-17: the r band and its fault injections are an opt-in check behind
   `KARUME_UPSTREAM_PARITY=1` — `packages/e2e/tests/e2e_wan_umt5_test.ts:28-35`, `:491`, `:889`. The
   port was proven at implementation (band 79, step 10c addendum); later changes are caught by the
   sha256 rows and the self A/B checks. The default run keeps asset completeness, tokenizer and bucket
   table bit identity, the golden's container sha256, the graph's I/O structure and non-finite
   checks.)**
2. **Quality record** (observation, not a check): row-wise cosine similarity, relative Frobenius error
   and max abs, on the four fixed prompts and more. The baseline is the unquantized CPU f32 (step 10c
   addendum); the i8 rounding and the bf16 asset are listed separately against it. The final judgment
   is a visual A/B (at least 4 seeds × 2–3 prompts), in the same role as the DeBERTa listening review
   (ADR 0026 decision 4).

- **"GPU i8 vs bf16 asset" is not a check.** The bf16 asset's own error is the floor and gives no
  ground for a band.
- **sha256 rows**: the GPU path's videos get new case ids (`KARUME_REFERENCE=write`, ADR 0106); the
  precomputed path's rows stay.
- Premise, tightened on 2026-10-03 (`7dc17309`, ADR 0106 addendum): the reference guard
  (`referenceGatePasses`, `packages/runtime/tests/helpers/reference.ts`) passes only when every
  registered case id has a row or is explicitly held.

### 9. Cancellation — `signal` (AbortSignal) on `generate`

Protects: generations of 30 minutes to 2 hours can be stopped without leaving GPU resources behind.

- `WanGenerateRequest` gets `signal?: AbortSignal` (as other families; maintainer's approval
  2026-10-03); the construction options of `fromPretrained` / `fromAssets` also take `signal`, passed to
  the fetch layer.
- **Check points**: stage boundaries (text → DiT → VAE) and between DiT steps. Each check first yields
  to the event loop (`settleAbort` in `packages/models/src/concurrency/abort.ts`).
- **On abort**: close the open Sessions, then throw `signal.reason` unwrapped, so a caller identifies
  its own abort by `error === controller.signal.reason` (ADR 0083 decision 5).
- One run within a stage is indivisible; the abort takes effect at the next boundary.

### 10. Steps

| Step | Content                                                                                                                  |
| ---- | ------------------------------------------------------------------------------------------------------------------------ |
| 10a  | Tokenizer (decision 1) and preprocessing (decision 2) in TS; recipe tables and fixtures                                  |
| 10b  | Host RAM of the export and its form (decision 6)                                                                         |
| 10c  | umT5 export (i8, f32 activations, dynamic L, bucket table as input), layer-sequential reference, TS bucket generator, GPU port check, fault injection |
| 10d  | Wiring into the pipeline (text stage → DiT stage, admission, manifest components, distribution and card, `signal`, sha256 rows, quality record, visual A/B) |
| 10e  | Optional: an i4 preset, host gather of the vocabulary embedding                                                          |

Withdrawn 2026-10-18: the working rules for running alongside step 9 (Chrome) and the w8a8 preset
work (10a–10c not touching `pipeline.ts`, GPU steps serialized by a lock). Steps 10a–10d and step 9
are complete.

### 11. Capacity — close the text stage before the DiT stage

Protects: one generation fits the GPU memory of the target devices.

- The text stage's Session (two runs, positive and negative) is closed before the DiT stage's Session
  is created — the same per-stage handling as DiT → VAE (ADR 0118 decision 7, ADR 0112). The step 10d
  design addendum F makes it "create and close per `generate`".
- Why: the text stage (i8 5.30 GiB) and the DiT stage of the default practical preset (81 frames,
  6.18 GiB — ADR 0120 addendum of 2026-10-04) total about 11.5 GiB, which leaves no margin on a 12 GB
  GPU and exceeds smaller browser budgets (estimate). On the B570 it was 12.6 GiB against a
  9,600 MiB ceiling.
- **Distribution**: umT5 lives in its own repository (ruling 2026-10-03 1). The 1,002 MiB i8 embedding
  fits under the 1,024 MiB part limit; tensor pieces (ADR 0090) let it span parts.

## Alternatives not taken

- **Follow transformers 5.14.1 (unknown → id 2)** — id 2 is `<s>` in umT5, an artefact of transformers
  5 rebuilding the original T5 layout; it may change with the version.
- **Follow `tokenizer.json` / transformers 4.x (unknown → id 3)** — the official requirement
  `transformers>=4.49.0` has no upper bound; the ids used in training cannot be confirmed.
- **byte_fallback of `spiece.model`** — no upstream path does it; ids would differ from the upstream
  pipeline.
- **Port all of ftfy** — largest port (2,731 lines + codecs); heuristic parity only by fuzzing; a
  follow-up on every ftfy version.
- **Accept only inputs with `prompt_clean(x) == x`, or move normalization to the user** — rejects the
  official negative, curly quotes, full-width letters and newlines; the latter demands Python in the
  browser.
- **Start with the strict version (reject `&` and the 449 characters) and port `BADNESS_RE` later** —
  rejects accented Latin and `’`.
- **A `[512,512]` constant sliced to `[:L,:L]`** (ADR 0118 decision 4's text) — a baked constant and a
  symbolic slice.
- **Compute buckets in the graph** — would need a separate binding of GPU `log` rounding at the
  boundaries.
- **Fixed 512 with a mask input** — compute always at 512.
- **Silent truncation beyond 512** — contradicts fail loudly.
- **Start with i4 g32, or ship i8 and i4 from the start** — group quantization broke the output in the
  mlx-umt5 report; Karume's symmetric g32 is unmeasured.
- **Host gather from the start** — the PLE machinery is gemma-specific.
- **Per-layer graphs** — more graphs and Sessions, a more complex admission.
- **Retire the precomputed asset** — makes the umT5 download (about 5.3 GiB) mandatory.
- **Automatic switch between asset and GPU; GPU positive with asset negative** — decision 7.
- **"GPU i8 vs bf16 asset" as a check** — decision 8.
- **Rewrite the precomputed path's sha256 rows with GPU values** — that path stays; tolerance is
  forbidden.

## Consequences

- Public surface: `@karume/models/wan` `generate` accepts prompts outside the asset. The accepted set
  is "within the 19,848 characters, no mojibake / entity candidate / C1 / lone surrogate / unassigned
  code point / added token, 2–512 tokens". `WanGenerateRequest` gained `signal`.
- The tokenizer asset is 8,576,480 B (gzip 3,318,348 B), eight times Anima's T5 equivalent.
- One `generate` adds the text stage (loading i8 5.30 GiB and two runs; 10.4 s at 33 frames on the
  B570, step 10d-4).
- Affected files and the pre-implementation cost and risk estimates: history; the results are in the
  addenda below. The exporter core change planned for decision 6 was not needed.

## Acceptance

| Step | Green condition                                                                                                                                                                                                                                                                                                  | Result |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| 10a  | Ids match a transformers 5.14.1 fixture (four fixed prompts, boundary cases, random) and, for accepted inputs, the `tokenizer.json` path. Anima's parity unchanged. Fuzz parity with ftfy 6.3.1 `prompt_clean` has zero mismatches; the `BADNESS_RE` translation matches `is_bad`; NFC matches UCD 16.0.0 on every code point. Every rejection case is a `ModelInputError`. Fault injections turn the parity red. | Done   |
| 10b  | Peak RSS of the export recorded and the export form fixed in an addendum                                                                                                                                                                                                                                       | Done   |
| 10c  | No out-of-vocabulary op; 24 relative tables stay F32; inputs `[1,L]` i32 and `[L,L]`, L 2–512; Python and TS bucket generators byte-identical; "valid length only" and "512 + mask" agree on the real model on the CPU (read as f32 bit identity — step 10b addendum; bf16 equality is not required); GPU within the band on accepted cases, fault injections outside (opt-in since 2026-10-17); VRAM peak, GPU time and max per-submit GPU time recorded | Done   |
| 10d  | GPU path is the default; text stage closed before the DiT stage; precomputed sha256 rows unchanged; new GPU-path rows written and reproduced; quality record kept; `signal` aborts at boundaries and steps with `signal.reason` and closed Sessions; distribution builds and `verify_dist` passes; full verify green; the maintainer's visual A/B | Done   |
| 10e  | Optional: an i4 preset and host gather; the acceptance form is fixed in an addendum after the browser limits and the quality record                                                                                                                                                                          | Open   |

## Rulings (2026-10-03, the maintainer)

1. **Distribution layout: (b) a separate distribution repository.** The light use without umT5 (2.9
   GiB) stays a download unit, and umT5-XXL is a general encoder that other families can reference.
2. **Visual A/B material**: fixed prompts first, free prompts later.
3. **Acceptance set of step 10a** (rejecting added tokens, `▁` after whitespace, entity candidates and
   unassigned characters): acceptable as a Wan / umT5 limit if the error message is clear. Rejection
   messages state the fix (for example `R&D` → `R & D`). A precise entity check from the
   `html.entities.html5` name table (2,231 names), rejecting only when the string actually changes,
   waits in the backlog for demand.

## Open questions

- The unknown id across transformers 4.x down to the required 4.49.0 is unconfirmed (only 4.57.1 seen
  as id 3). Karume's ids are unaffected because out-of-vocabulary input is rejected.
- NFC parity in browsers (ICU versions of Chrome / Safari) is unverified.
- Step 10e: i4 g32 quality and the host gather, and how the 1,002 MiB embedding buffer and the 64 MiB
  scores (L = 512) meet browser binding and ArrayBuffer limits on the GPU path.
- Closed by the addenda: "512 + mask" equivalence, the relative tables escaping i8, `gelu_new`, the
  export's host RAM, `BADNESS_RE` / NFC parity, residual VRAM between text and DiT, overlap with step 9.

## Addendum (2026-10-03) — step 10a results and amendments to decisions 1 and 2

- **Step 10a done** (`b5a4592f` / `d601c488` / `5e95f587`): the shared T5 tokenizer
  (`packages/models/src/text/t5-tokenizer.ts`, family differences in `T5Policy`) with Anima's parity
  unchanged. The recipe `tools/export-recipes/wan/umt5_tokenizer.py` bakes the umT5 asset (vocabulary
  256,300, 304 added tokens, whitespace set, `prompt_clean` tables) into
  `outputs/series/wan2.1-umt5-tokenizer/tokenizer.json`. TS: `packages/models/src/wan/text/{tokenizer,prompt-clean}.ts`.
- **Results**: fixture of 4 fixed + 30 boundary + 200 random prompts (ids agree between transformers
  5.14.1 and `tokenizer.json` for accepted cases). Preprocessing fuzz: 200,000 safe-pool, 50,000
  full-pool and 20,000 entity-pool strings, zero mismatches. The `BADNESS_RE` translation matches
  `is_bad` on the full pool and on six Unicode-dependent contexts × every code point. NFC: V8 (Unicode
  16) and Python UCD 16.0.0 agree on 292,531 assigned characters × 8 contexts, so **no NFC table is
  baked**. Fault injections (drop a table, skip NFC, drop `</s>`, remove the out-of-vocabulary
  rejection, truncate beyond 512) are all red.
- **Decision 1 amended (a premise broke on measurement)**: added tokens in the body (`</s>`,
  `<extra_id_*>`, 304 in all) and `▁` right after whitespace split the ids between transformers 5 and
  `tokenizer.json` (every split in 60,000 random strings contains one of them or an out-of-vocabulary
  character). Both are rejected with `ModelInputError`. Investigation §1.4's "the only differences are
  out-of-vocabulary ids and 24 whitespace characters" came from a one-character sweep and was
  incomplete.
- **Decision 2 settled**: entity candidates are detected by regular expressions (ftfy
  `HTML_ENTITY_RE` at the start of each round; after `fix_text`, a form equivalent to `html.unescape`'s
  `_charref`) without a name table, so `R&D` is rejected and `R & D` passes. Code points unassigned in
  UCD 16.0.0 are rejected (a newer engine's NFC could differ from Python), including 60 unassigned
  characters in the vocabulary. The reason is `PromptCleanError` (a `ModelInputError`; unassigned /
  c1 / entity / mojibake).

## Addendum (2026-10-03) — step 10c preparation: two open points closed, `gelu_new` ruling

- **Preparation done** (`17067949` / `e6ed06c5` / `246c1d50`): the valid-length wrapper
  `Umt5EncoderTokens` (inputs `input_ids [1,L]` and `relative_position_buckets [L,L]`, output
  `[1,L,d_model]`, no mask, L symbolic in 2–512) exports a small random UMT5 through exporter verify.
  The bucket tables of Python (upstream `_relative_position_bucket`) and TS (f64 `Math.log`) are
  byte-identical for L = 2–512. Session I/O helpers: `packages/models/src/umt5/session-io.ts` (moved
  from `src/wan/umt5/` by `e033c3f0`).
- **Relative tables kept out of i8**: the exporter's i8 default also rounds `nn.Embedding`
  (`QUANT_CHANNEL_AXES`), so the 24 tables are excluded through the `include` of `fake_quant_int8`
  and forced to F32 through `weight_dtype_overrides` of `emit` (one alone fails — measured). No core
  change.
- **`gelu_new` (ruling (a))**: the upstream formula fails export on `aten.pow.Tensor_Scalar`; the recipe
  substitutes `nn.GELU(approximate="tanh")` (the same function; one fused `gelu_tanh`). 0.54% of
  elements differ, at most 4.77e-7. A core rule `pow(x,3) → x·x·x` was not taken (eight elementwise
  ops).
- Note for decision 8: upstream UMT5 casts the RMSNorm variance and the softmax to f32, so `.double()`
  alone is not an f64 reference; the f64 reference is written out explicitly.

## Addendum (2026-10-03) — step 10b results: decision 6's form fixed, notes on decisions 4 and 8

- **Step 10b done** (`e70678b7` / `56e91b68`, [research 2026-10-03 umt5-export-ram](../research/2026-10-03-umt5-export-ram.md)):
  the straightforward form would peak at about 30.5 GiB (estimate, not run).
- **Decision 6 fixed without a core change**: trace on a meta-device upstream (the graph JSON equals
  the real-weight export); the 169 quantized tensors (168 linear + the vocabulary embedding) are
  quantized from the checkpoint in row blocks of 16,384 and passed as `fixed_weights` (ADR 0097's
  entry); the 73 F32 tensors (24 relative tables, 49 RMSNorm) pass as they are. Real model: peak
  6.79 GiB, 69 s, container 5,686,978,107 B (26 parts), `karume verify` green. The upstream
  text_encoder is stored as F32 (22.7 GB, five files), not bf16.
- **Decision 4 on the real model (CPU)**: "valid length only" and "512 + mask" are **bit-identical in
  f32** on all four fixed prompts. In bf16, three of four differ, within bf16's own error against f32.
  Acceptance 10c reads "bit-identical in f32".
- **Decision 8**: the bf16 asset differs a lot from f32 (relative Frobenius 2.6e-2–1.3e-1, minimum
  row cosine 0.168 on ferret row 64), so the quality record uses the f32 reference as its baseline.

## Addendum (2026-10-03) — step 10c CPU results: layer-sequential reference and goldens

- **Reference writer done** (`8ff54f06` / `6e325a0e`): the umT5 encoder forward written out in f64,
  run layer by layer (RSS peak 3.5 GiB), weights read one at a time from the i8 container (packed ×
  row scale = the fake-quant value, block sha256 verified). A `TorchDispatchMode` guard keeps the f64
  path free of other float types. On the small model the f32 write-out is bit-identical to upstream
  eager for L = 2–512.
- **Cases and goldens**: six decision cases (three random accepted strings, L = 8 / 22 / 36, and three
  concatenations, L = 163 / 327 / 488) and four acceptance cases (the fixed prompts). Goldens are
  `reference.<case>.safetensors` (46 MB in all). After rewriting the container, regenerate them with
  `python -m wan.umt5_export reference`; a host test turns red on a mismatched part-0 sha256.
- **Fault injections**: shift the relative table by one, replace one token, swap the tables of layers
  0 and 1, scale the per-channel scale of layer 12 `wo` by 2 (outside the band) and by 1.0001
  (recorded).

## Addendum (2026-10-03) — design of step 10d: cross-repository reference, path choice, signal

Material: [research 2026-10-03 umt5-wiring-recon](../research/2026-10-03-umt5-wiring-recon.md).

- **A. Reference form = cross-repository reference**: Wan's manifest (`karume-wan2.1`) references the
  `text_encoder` container in the umT5 repository (hub FileRef = repo + 40-digit commit SHA, ADR 0109
  decision 3). One source fetches both and Wan's revision pins umT5's bytes. Cost: every Wan preset's
  `requiredLimits` declares the 1,050,148,864 B embedding buffer (decision 5).
- **B. Path choice = a construction option** `textEncoder: "gpu" | "precomputed"` (default `"gpu"`),
  decided before fetching because the components differ. Not per request and not by preset (a preset's
  weights are a complete mapping and cannot omit umT5).
- **C. Tokenizer asset = Wan's own asset** (`WAN_ASSETS`, 8 MB, format `karume-wan-umt5-tokenizer/1`,
  bundled with Wan's preprocessing tables). Assets are always fetched in full, so placing it in the
  umT5 repository would make the precomputed path cross repositories.
- **D. umT5 repository = `karume-umt5-xxl`** (ADR 0092 decision 2), pipeline name `umt5-encoder/1`,
  one preset `i8`, component `text_encoder`. Provenance: the Wan text_encoder **(superseded by ADR
  0122 decision 1 on 2026-10-04: the upstream is `google/umt5-xxl`, measured bit-identical in f32 to
  Wan's copy; series `umt5-xxl-i8-dyn`; parts and goldens unchanged)**.
- **E. Placeholder SHA before publication + a mechanical check**: local mirrors use a placeholder
  revision (40 zeros); `tools/export-recipes/dist.py` refuses it unless `--allow-placeholder-ref` is
  given, and publication refuses it. Publish umT5 first (release-runbook §0).
- **F. Text-stage Session lifetime = created and closed per `generate`** (decision 11).
- **G. signal**: construction via `WanPipelineOptions.signal`; generation via
  `WanGenerateRequest.signal` (stage boundaries, each DiT step, each VAE tile; close Sessions, throw
  `signal.reason` unwrapped).
- The GPU path accepts any string that passes the `prompt_clean` mirror and the tokenizer;
  `pipeline.prompts` still lists the asset prompts. The default negative is a TS constant (the official
  `sample_neg_prompt`), bound by a test to the asset's `negative` text. Rejection messages state the
  fix. The GPU path's sha256 rows use new case ids; precomputed rows pin `textEncoder: "precomputed"`.

## Addendum (2026-10-03) — steps 10d-1 and 10d-2: distribution repository and wiring

- **10d-1 done** (`e8c86279` / `1d52ced4` / `d992caf9`): `dist.py --pipeline umt5` builds
  `karume-umt5-xxl` (model `xxl`, `text_encoder` in 26 parts). Wan's distribution references
  `weights.text_encoder` across repositories for every preset and carries `assets.umt5_tokenizer`.
  Every preset declares `maxBufferSize` = `maxStorageBufferBindingSize` = 1,050,148,864. Existing
  DiT / VAE / assets stayed byte-identical. The 73 F32 tensors in the umT5 container are not
  bf16-representable (249,854 of 249,856 elements have non-zero low bits); the card says only "upstream
  is float32".
- **10d-2 done** (`b7d2784c` / `b935782c` / `cde8a1a0`): `WanPipelineOptions.textEncoder`; admission
  `umt5Contract` (`packages/models/src/wan/text-stage.ts`: input names and shapes, the same symbolic L,
  f32 output `[1,L,W]` with W the DiT context width, 512 ≤ context rows, storage f32 / i8) and a check of
  the tokenizer asset declaration on the GPU path. The text stage runs positive then negative, checks
  abort before each run, names non-finite outputs and closes with `disposeSteps` before the DiT.
  17 host tests with nine fault injections.
- Listing `karume-umt5-xxl` among the sources was left to publication; it is `UMT5_SOURCES` in
  `packages/models/src/umt5/config.ts` today. The free-prompt E2E case
  `gpu-text-2step-free-fox-seed42` ("A red fox trots through fresh snow in a quiet birch forest at
  sunrise.") changes id when its text changes.

## Addendum (2026-10-03) — step 10c GPU check and step 10d-4: the GPU path runs end to end

- **10c port check done** (`f5a63221`, B570): band `UMT5_NORMALIZED_BAND = 79`
  (`packages/models/tests/helpers/umt5-golden.ts:51`; decision r 8.25 / 5.92 / 2.20 / 4.93 / 0.904 /
  15.8, worst × 5 rounded up). Acceptance r 2.78 / 8.87 / 4.36 / 3.74. Fault injections at r 5.0e+5–9.7e+5;
  scale × 1.0001 at r 22.6–80.2 (a 1e-4 scale drift is not distinguishable; no subtle-fault floor).
- **Quality record** (four acceptance prompts, baseline unquantized CPU f32, minimum row cosine /
  relative Frobenius): i8 rounding = boxing-cats 0.982 / 5.0e-2, ferret **0.573** / 7.7e-2,
  cat-dog-baking 0.993 / 3.4e-2, negative 0.998 / 2.3e-2; bf16 asset = 0.997 / 3.1e-2, **0.168** /
  7.2e-2, 0.768 / 1.3e-1, 0.987 / 2.6e-2. GPU i8 vs its CPU f32 reference: cosine 1.000, relative
  Frobenius 4e-6–9e-6.
- **Resources (umT5 alone)**: allocation 5.30–5.55 GiB; one forward 0.14 s (L 8) to 1.73 s (L 488);
  max per-submit GPU time 155.6 ms.
- **10d-4 done** (`b888ff00`): the precomputed path's eight sha256 rows unchanged; GPU-path rows
  `gpu-text-2step-boxing-cats-seed42` and `gpu-text-2step-free-fox-seed42` written. Text stage 10.4 s
  (33 frames). VRAM: text stage peak 6.30 GiB → 0.708 GiB after closing (+0.08 GiB residue) → DiT
  stage peak 5.94 GiB; the two stages do not overlap.

## Addendum (2026-10-04) — umT5 provenance moved upstream

ADR 0122 decision 1 replaced the provenance "Wan's `text_encoder`" with `google/umt5-xxl`
(series `umt5-xxl-i8-dyn`; parts and goldens unchanged).

## Addendum (2026-10-04) — remaining step 10d: 24 videos for the visual review

Two sets of 12 (free and fixed prompts, practical preset, 832×480, 33 frames, 50 steps, seeds 42–45)
were generated on the B570; the fixed set pairs with the precomputed path's videos and differs only in
the text path. The records in the repository name only `anime-dancer` and `retriever-beach`.

## Rulings (2026-10-04, the maintainer)

- **The i8 umT5 passes the visual review.** On the fixed prompts × seeds 42–45, GPU i8 umT5 and the
  bf16 precomputed embeddings showed no visible difference. Decision 5 (i8 weights, f32 activations)
  stands; the visual review of 10d is done.
- The weak prompt adherence seen on free prompts is not caused by the 8-bit umT5; it is treated as a
  property of the model and compared with Wan2.2 TI2V-5B (ADR 0121).

## Addendum (2026-10-09) — free-prompt comparison

The maintainer judged that Wan2.2 TI2V-5B follows the free prompts better (two of three; same umT5
weights and path), so Wan2.1's weak adherence is recorded as a property of the 1.3B model. Material:
`outputs/misc/wan22-visual-free-2026-10-09/`, `outputs/misc/wan21-visual-free-webp-2026-10-09/`.
