# 0097: Integrate Gemma 4 QAT mobile as a separate family

- Status: accepted (2026-09-11 — the maintainer approved the staged plan, family `gemma4-qat` with
  E2B / E4B); partly superseded: addendum 1's format by [0108](0108-container-format.md) /
  [0109](0109-manifest-v5-container.md) and addendum 4's PLE sidecar by [0085](0085-ple-host-gather.md)
  (2026-09-22; addendum 8); "existing assets, tolerances and goldens do not change" by
  [0129](0129-test-layout-and-lanes.md) decisions 3 to 5 (2026-10-16); addendum 5 / 6 defaults by
  addendum 7. The approval section and addendum 6's CLI withdrawn 2026-10-18.
- Date: 2026-09-11
- Basis: [INT2 / SRQ spike and staged plan](../research/2026-09-10-codex-mtp-optimization.md) (section on QAT
  mobile INT2 and fixed rounding, 2026-09-11)
- Related: [0069](0069-packed-w4-storage.md) (packed storage), [0085](0085-ple-host-gather.md) (PLE),
  [0092](0092-distribution-repos-and-sources.md) (distribution and sources), [0096](0096-speculative-decoding.md) (MTP)

## Context

The official mobile QAT shares its skeleton with regular Gemma 4, but uses fixed INT2 / INT4 / INT8
weights and SRQ (Static Range Quantization: activations rounded with a fixed scale). Renaming an
existing model's quant or widening the weights to f32 does not reproduce the same computation. A
spike showed size and speed gains for INT2. Cases remain where the CPU / GPU difference of a linear
reduction crosses a rounding boundary, so a unit-level match cannot guarantee whole-model quality.

## Decision

1. The family for distribution and users is **`gemma4-qat`**, with models **`e2b` / `e4b`**. Regular
   `gemma4` model names and default quants do not change. Scope: text generation from the official
   `*-qat-mobile-transformers` checkpoints. GGUF / unquantized / compressed-tensors forms with the same
   QAT name are not treated as the same thing automatically.
2. The future distribution repo is `karume-gemma4-qat` and the manifest pipeline name `gemma4-qat`;
   the old Gemma reader keeps refusing this pipeline. No source pin is made before publication;
   acceptance uses the local distribution; publishing is not part of this work. (Kept 2026-10-18: still
   unpublished at 0.14.0, `CHANGELOG.md`; this decision is updated when it is published. Published
   2026-11-01 with 0.15.1 — Addendum 9.)
3. Execution, tokenizer, chat, RoPE and fetching reuse the existing Gemma implementation; the pipeline
   is not duplicated because of the new family. The QAT entry explicitly checks the supported form and
   the presence of the fixed rounding.
4. Stored fixed integers and scales are kept without re-quantization. INT2 in the IR, fixed SRQ and
   INT4 PLE are implemented. The storage shape / byte count / packing / scale and the rounding and
   non-finite contracts are added to this ADR per implementation unit, aligning reader, writer, CPU
   reference and GPU. Existing assets and numeric tolerances do not change. (Superseded by ADR 0129 on
   2026-10-16: op oracles and sha256 reference rows stay mandatory; upstream parity at the model exit is
   a per-family choice that goes through the triage. Tests are still not loosened silently.)
5. Regular generation, performance and resource release are accepted first. Sharing between head and
   token embedding is used only after a full byte match. The MTP drafter's i8 borrowing is not replaced
   by INT2 implicitly; MTP is a separate design after regular generation is accepted.

## Implementation units (history)

INT2 storage, fixed SRQ, the QAT recipe with PLE and the family entry were implemented and accepted unit
by unit on 2026-09-11 (addenda 1 to 6). Withdrawn 2026-10-18: the "approval handling" section (a working
agreement for that finished work: no per-stage re-approval, numeric error not a stop reason).

## Addendum 1 (2026-09-11) — INT2 storage and execution contract

(Format superseded 2026-09-22: storage `i2` in IR v1, `I2` in the safetensors dialect and manifest
`karume/4` became codec `int2-off` in IR v2's binding table and manifest `karume/5` — addendum 8.)
Old readers refuse unknown storage types.

- Semantics are f32. The logical shape is a positive rank-2 `[N,K]`, K a multiple of 16. Four elements
  per byte from the low 2 bits, stored as `u=q+2`, using the full integer range `[-2,1]`. The header
  shape is logical; the byte count is `N*K/4`. 4-byte aligned start, no implicit tail padding.
- `storage.scale` is required: F32 `[N,1]`; no group_size. The scale lives in the weight's shard (with
  row splits, next to the first piece). Reconstruction is per element `fround(q*scale)`. Stored
  integers and scales are not re-quantized.
- Consumed only by linear / embedding weight slots → resident packed. Other consumers or a graph
  output → expanded to f32 on the CPU, with the expanded bytes in diagnostics and estimates.
  `linearCompute` is f32 only; a8 / f16 and packed I2 for convolutions are refused explicitly.
- GEMV: `1<=M<=64`, `N%4=0`, `K%64=0`. A 16-byte weight word carries 64 elements, reduced in ascending
  K. Row blocks reach at most 4 rows under the existing 256-element limit; otherwise packed GEMM.
  Embedding reads rows packed.
- Shared initializers keep the existing checks (shape, storage, consumer, device, run lease,
  lifetime); an INT2 scale is borrowed on the same row axis (MTP's i8 borrowing stays i8).
- The exporter's low-level writer / reader and fixed pack / unpack correspond. No automatic INT2
  quantization on the weight-dtype path; fixed packed weights enter through addendum 3.

PLE storage is **INT4 on E2B, INT2 on E4B** (official config and weights). PLE is a host-read
sidecar, so it keeps per-layer block scales separate from the IR's per-row scales. In both models
head and token embedding have identical integers and scales.

## Addendum 2 (2026-09-11) — fixed SRQ op contract

The IR gains `static_quantize`: one f32 input and output, shape unchanged (scalars and empty tensors
allowed). Attribute `scale` is required: only non-negative, finite JSON numbers exactly representable in
f32 (stored scales are never re-rounded). Other ops' attribute rules are unchanged; an old runtime
refuses it as an unknown op. QAT's fixed SRQ is the model's own semantics, so it is an
IR op. ADR [0006](0006-quantization.md)'s "quantization stays out of graph semantics" (the reason ADR
0025 decision 1 rejected an IR op) covers PTQ storage and execution options and is compatible with this
op (0006's 2026-09-24 addendum, "scope of the skeleton").

- For scale > 0: f32 division `x/scale` → round to nearest even → saturate to `[-128,127]` → f32
  multiplication. scale = 0 (including −0) returns the input bits unchanged.
- Signed zero is kept. ±Inf saturates to the integer bounds, then is multiplied by scale. With scale > 0
  NaN keeps sign and payload and gets the quiet bit. Multiplication overflow returns ±Inf. Nothing
  depends on the GPU's float-division tolerance or flush-to-zero, subnormals included.
- The CPU reference computes f32 division and even rounding directly. The GPU uses the bit order of
  positive magnitudes and a binary search over a boundary table: 128 boundaries and 129 outputs built
  once on the host, carried with the count and an identity flag in a 1,040-byte uniform; at most 8
  comparisons. Value arithmetic does not go back to GPU f32 division / multiplication.
- Boundaries `j+0.5` account for their f32 rounding interval: for even j the first f32 strictly above
  the upper end, for odd j the first f32 at or above the lower end starts the next integer step. The
  host's boundary products are exact in f64. Coinciding boundaries for some scales stay a monotone table.
- A dedicated op kind and standalone kernel: not part of unary fusion; scale is not in the codegen key.
  The uniform uses the content-addressed cache and the Session lifetime; its bytes are excluded from the
  memory estimate like other params and counted in diagnostics' weights.
- The exporter keeps `karume::static_quantize` atomic and converts it to the IR op of the same name.
  The generic core's eager implementation uses only Torch ops (no Transformers dependency). The QAT
  recipe supplies the fixed scales; SRQ is never inserted into existing graphs automatically.

Accepted with official-CPU fixtures (boundary neighbours, random and special values), an independent CPU
reference, real-GPU u32 comparison, a shared TS / Python contract table, a codegen snapshot and a tiny
torch.export golden. ("Existing goldens and tolerances do not change": superseded by ADR 0129 on
2026-10-16, as in decision 4.)

## Addendum 3 (2026-09-11) — fixed-quantization writer entry

The generic exporter exposes `FixedQuantizedWeight(dtype, packed, scale)` (`karume.emit`) and an
optional `fixed_weights` mapping on `publish_model` and `stored_model`; existing calls behave as before.

- Keys are initializer FQNs; their `tensors` entries are f32 / meta tensors of the same logical shape,
  and the meta set and the mapping are required to match exactly. Input passing both real f32 values and
  a fixed payload (one would be ignored) is refused.
- Targets: positive rank-2 `[N,K]` initializers consumed only as linear / embedding weights. Graph
  outputs, other consumers and duplicate declarations of a key are refused.
- Fixed dtypes I2 / I4 / I8. Packed is CPU-contiguous: I2 / I4 as U8 `[N,K/4]` / `[N,K/2]`, I8 as I8
  `[N,K]`. All signed values kept; no quantization, re-packing or f32 expansion. I2 needs K % 16 = 0;
  I2 / I8 scales are F32 `[N,1]`; I4 derives group_size from F32 `[N,groups]` and needs a power of two
  ≥ 16 dividing K.
- Mixed I2 / I4 / I8 within the mapping is accepted. Combining it with `weight_dtype` (other than f32),
  `weight_scales` or `weight_dtype_overrides` is refused. An empty mapping changes nothing.
- Scale key generation, co-location, row splitting, alignment, reader validation and the publish swap
  reuse the existing code. Row splits cut the packed payload by row ranges on the same leading axis
  without building the parent f32. Graph declarations and input tensors are unchanged; verifying the
  upstream origin of fixed values is the recipe's job.

Accepted by split / unsplit payload equality, reader acceptance, refusal of ambiguous input, keeping
outputs on a failed publish, and regression of the automatic quantization.

## Addendum 4 (2026-09-11) — packed PLE sidecar

(Superseded 2026-09-22: the sidecar became container assets with index schema 3 and block lists — ADR
0085 addendum 2026-09-22; the `storage` values and the encoding below carry over.)

- Values are token-major, logical shape `[rows,layers,dim]`, dtype I2 / I4; dim a positive multiple of
  16; byte count `rows*layers*dim/4` / `/2`; each byte packs `q+2` / `q+8` from the low bits, using all
  signed values.
- `scales` are F32 `[rows,layers]`. As a sidecar outside the graph, the IR's rank-2 limit does not apply.
- Reconstruction is the existing two-stage f32 product `(q*scale)*embedScale`, shared by full, resident
  and range reads. Byte budgets and row offsets use real packed bytes; returned shapes and duplicate-id
  copies use logical element counts.
- `storage` is required in schema 3 (a missing value is refused, not read as i8 — `packages/models/src/gemma/ple-index.ts`); unknown schema / dtype is refused. Cancellation, exclusivity and
  lifetime contracts are unchanged.

Accepted by a full-bit match with the official E2B INT4 / E4B INT2 probes and a model-independent Torch
fixture covering full reads, row reads, residency, duplicate tokens and boundaries.

## Addendum 5 (2026-09-11) — fixed QAT recipe and how numeric comparisons are treated

Recipe `gemma4_qat` reads only the official mobile Transformers form and writes text / PLE / tokenizer
into the series `gemma4-qat-<model>-product`; run it with `python -m gemma4_qat.export`. It reuses the
Gemma wrapper, attention registration, states conversion and tokenizer compile.

- The original fixed integers and scales are the only value source. Trace uses shape-only weights; the
  converted graph goes to the f32 / meta declarations and the fixed writer. Head / embedding sharing
  requires a full byte match. All fixed payloads and PLE after splitting are checked against the
  original bytes and placed with provenance, config and tokenizer in one swap unit.
- No regular Gemma quantization or drafter is added. The distribution quant follows the existing
  grammar as `i4`; label / description state the fixed mixed INT2 / INT4 / INT8 and SRQ.
- The trace limit is stored in provenance and `dist` checks it. The model's position limit comes from
  the official config. (The initial defaults capacity 128 / chunkLength 32 / trace limit 128 are
  replaced by addendum 7.)
- QAT RoPE uses a dedicated entry that rounds frequency power, reciprocal and position product to f32
  stage by stage. Regular Gemma's f64 contract is unchanged. Trig is host `Math.cos` / `Math.sin`
  stored as f32. Bit identity with upstream Torch trig or GPU reductions is not guaranteed.

The first CPU / GPU difference was attributed to a matrix reduction crossing an SRQ rounding boundary
(SRQ itself is bit-identical on CPU / GPU); measurements in the research record. This is not a general
quality pass; the family states it as an experimental constraint. (Its "existing tolerances and
expectations do not change" is superseded by ADR 0129 on 2026-10-16, as in decision 4.)

## Addendum 6 (2026-09-11) — shared pipeline

Public entries `Gemma4QatPipeline.fromPretrained` / `fromAssets` and subpath `@karume/models/gemma4-qat`
are added. The Gemma body moves to a private shared base; regular and QAT differ only in thin
factories. Chat, sequence, fetching, estimation and release are shared; regular Gemma's defaults,
numerics and MTP are kept.

- QAT accepts only `gemma4-qat/1` with models `e2b` / `e4b`, and checks before construction: INT2
  token embedding, shared head, fixed mixed storage, SRQ around quantized linears, per-model PLE
  storage. A regular Gemma manifest is never read as QAT. MTP for QAT is refused by type and at run time.
- Derived RoPE inputs switch to the QAT f32 stage rounding; `gemma4RopeInputs` and existing golden
  values are unchanged; `gemma4QatRopeInputs` is new and public.
- Chat API and asset / config types are the shared Gemma types. QAT fetch options narrow the model to
  two names and take no `speculative`. Published regular Gemma API is not removed or renamed.
- Accepted by a full regular-Gemma regression and QAT E2B / E4B generation, multi-turn KV reuse,
  abort and release checks. (Its "do not loosen expectations or tolerances" is superseded by ADR 0129
  on 2026-10-16, as in decision 4.)

Withdrawn 2026-10-18: the `demo:gemma4-qat` CLI, its defaults (`models/karume-gemma4-qat/`, 64 new
tokens) and its start-up notice — the CLI examples were removed by ADR
[0127](0127-node-dawn-toolchain.md) decision 5 (2026-10-16); decided again for a browser application.

## Addendum 7 (2026-09-19) — rulings after the review

Measurements: [QAT review record](../research/2026-09-19-qat-review.md). Terms: [glossary](../glossary.md);
quantization index: [quantization](../quantization.md).

### 1. Defaults equal regular Gemma

The recipe defaults are **capacity 4096, chunkLength 768, trace limit (`maxChunkLength`) 768**
(`tools/export-recipes/gemma4_qat/config.py`) for E2B and E4B. This **overrides** addendum 5's initial
128 / 32 / 128 (the old text stays as the record of its time). Raising `maxChunkLength` requires a
re-export. Default capacity and trace limit are separate constants, and three conditions are checked
before storing: `chunkLength ≤ maxChunkLength`, `chunkLength ≤ capacity`, `capacity ≤ maxPosition`.
Output quality beyond 512 tokens of context is still not accepted; the QAT section of
[limitations](../limitations.md) owns it. (The CLI default of 256 new tokens went with the CLI.)

### 2. The recipe does not insert scale-0 SRQ

Upstream SRQ scale is 0 (uncalibrated = identity) **only around lm_head**, so the recipe inserts no
scale-0 `static_quantize`. The structural check (TS `admitGemma4Qat`, Python `assert_qat_graph`) rule
"SRQ required before and after every quantized linear" is relaxed to **the shared head (a linear using
the token embedding's initializer) may omit both**. Other quantized linears still need both; check and
recipe change together.

The always-true `i2` condition is dropped from the mixed-storage coverage condition. What remains is
**`i4` and `i8` both present** — an assumption taken from the two current models (still e2b / e4b only
on 2026-10-18; revisit when a checkpoint is added). A checkpoint outside it (for example without `i4`)
is refused by the check (`packages/models/src/gemma/qat.ts`).

### 3. `reference.json` stops recording always-true flags

The literal `True` flags `fixedBytesExact` / `pleBytesExact` are dropped. Instead `qat_plan`
**cross-checks** `fixedWeights` / `storageCounts` and the PLE count against the distribution
container. Upstream tensors not reflected in the conversion (`upstreamUnused`) are recorded with counts,
so "not read" and "overlooked" can be told apart later. (Schema 2 at the time; now schema 3, which counts
PLE container blocks `pleBlocks` instead of sidecar shards — `REFERENCE_SCHEMA` in
`tools/export-recipes/gemma4_qat/config.py`.)

### 4. KV cache stays f32

The checkpoint's per-layer `k_cache_scale` / `v_cache_scale` (70 in the text model) are **not kept**;
KV is f32 (upstream transformers ignores them too). Full-layer KV per position is **E2B 12,288 B / E4B
32,768 B** (sliding layers use a fixed window-512 buffer independent of capacity). The count goes to
`upstreamUnused`.

### 5. Where QAT RoPE's 1-ULP difference comes from

Addendum 5 blamed non-identity on "upstream Torch trig or GPU reductions", but **the inverse-frequency
stage itself** is 1 ULP off the official one in one entry per layer kind; the difference grows with
position (max absolute 1.909e-3, full sin at position 131,071).
Torch's vectorised power is 1 ULP off; Karume's power rounds correctly, but its inverse frequency is 1
ULP off the exact value by double rounding (power to f32, then reciprocal): **neither side is
mathematically right**. Single-stage rounding would disagree on many more elements, so per-stage f32
rounding stays. Bit identity would need an inverse-frequency table in `pipelineConfig.rope`; not taken.

### 6. What "official CPU reference" means

Addendum 5's official reference is **transformers (f32, `attn_implementation='eager'`)**, not the
mobile runtime the checkpoint targets (LiteRT-LM); agreement with the mobile runtime was not measured.
The two reference fixtures differ: the intermediate-value fixture injects Karume's candidate RoPE (used
to attribute CPU / GPU reduction differences); the 8-case generation reference keeps the official RoPE
(including decision 5's difference). Docs citing this metric always say which reference they mean.

### 7. Why f32 load

The recipe's `dtype=torch.float32` is **the same** dtype that the model card's recommended
`dtype="auto"` resolves to for this checkpoint (the root config has no `dtype`, so f32). Karume does not
take a condition different from the official one.

### 8. Integer activation dot products and int8 KV are not implemented now

The official mobile integer dot product ([perf-ledger](../perf-ledger.md) K-45) and int8 KV cache
(K-46) are **out of scope**; they are filed and the ledger holds them. Addendum 1's "activations are
only rounded, compute is f32" therefore stays, and CPU / GPU token sequences may still diverge.

## Addendum 8 (2026-09-25) — storage and PLE location after the container

Addendum 1's storage is, from ADR [0108](0108-container-format.md) on, codec `int2-off` in IR v2's
binding-table `encoding` (0108 decisions 12 / 13, [container-v1](../container-v1.md) §6); safetensors
`I2` is not accepted; manifest `karume/5` carries the field ([0109](0109-manifest-v5-container.md)
decision 2). Addendum 4's PLE sidecar moved to container assets ([0085](0085-ple-host-gather.md), 2026-09-22).

## Addendum 9 (2026-11-01) — published on Hugging Face

The distribution is published as `hdae/karume-gemma4-qat` with 0.15.1 (models `e2b` and `e4b`, the
manifest's default `e2b`; release-runbook §3 "First publication of a family"). Decision 2's "no
source pin before publication" is fulfilled rather than reversed: the pin is made now, at
publication. The subpath `@karume/models/gemma4-qat` gains `SOURCES` (key `"gemma4-qat"`, folded into
`KARUME_SOURCES`; it is declared beside regular Gemma's table in `packages/models/src/gemma/config.ts`
because both families share that directory), and the omitted-ref diagnostic of
`gemma4Qat.Pipeline.fromPretrained` names `gemma4Qat.SOURCES["gemma4-qat"]` — spelled
`SOURCES["gemma4-qat"] (@karume/models/gemma4-qat)`. MTP, vision and audio stay unsupported.

## Addendum 10 (2026-11-04) — the distribution is renamed `karume-gemma4-qat-mobile`

The maintainer's ruling of 2026-11-04: the name `karume-gemma4-qat` (HF `hdae/karume-gemma4-qat`) goes to the q4_0
QAT line — Google's q4_0 quantization-aware-trained E2B / E4B / 12B checkpoints on their own 4-bit grid, each with its
QAT MTP drafter, a distribution of the regular `gemma4/1` contract that `@karume/models/gemma` reads
(`tools/export-recipes/gemma4/distribution.py`, `QAT_Q4_0_PIPELINE`). This family's distribution becomes
`karume-gemma4-qat-mobile`. It was published recently and has no users, so the rename is breaking, with no alias and no
compatibility shim:

- The subpath `@karume/models/gemma4-qat` becomes `@karume/models/gemma4-qat-mobile`, the barrel namespace `gemma4Qat`
  becomes `gemma4QatMobile` (ADR 0136 decision 3), the diagnostics name `gemma4QatMobile.Pipeline` (ADR 0136
  decision 5), and the test lane `models:gemma4-qat` becomes `models:gemma4-qat-mobile` (its files carry the token
  `gemma4_qat_mobile` — ADR 0129).
- The recipe keeps its directory (`tools/export-recipes/gemma4_qat/`) and its series (`gemma4-qat-<model>-product`);
  `dist.py --pipeline gemma4-qat-mobile` builds `models/karume-gemma4-qat-mobile/` (card title "Gemma 4 Mobile QAT
  Text Decoder"), and `--pipeline gemma4-qat` now builds the q4_0 QAT line.
- The manifest's pipeline contract stays `gemma4-qat/1`: it names the reader's contract, not the repository, and the
  model files and `karume.json` of the renamed mirror stay byte-identical (only `README.md`'s title changed).
- Addendum 9's source table is withdrawn: `hdae/karume-gemma4-qat-mobile` has no published revision yet, and a family
  without a public repository has no table (ADR 0073 decision 1; release-runbook "First publication of a family"), so
  the subpath exports no `SOURCES`, `KARUME_SOURCES` has no mobile entry, and an omitted ref gets no hint. The next
  release publishes the renamed repository and restores the table under the key `"gemma4-qat-mobile"`.
- The HF repository `hdae/karume-gemma4-qat` itself is not new: until the q4_0 QAT line is uploaded there, its
  history holds this family's mobile content, including the revision that 0.15.1's `gemma4Qat.SOURCES["gemma4-qat"]`
  pins. A user of 0.15.1 keeps reading that revision; the upload of the q4_0 line replaces `main`'s files and its
  card.
- The family reader (`gemma4-qat/1`) and the internal family name `gemma4-qat` (`packages/models/src/gemma/qat*.ts`,
  the recipe directory `gemma4_qat/`) stay; only the distribution, the subpath, the namespace and the lane are
  renamed.
- ADR [0092](0092-distribution-repos-and-sources.md) decision 1 (one repository per family) and the split: the ruling
  did not cite it. Its addendum of 2026-11-04 records the layout — the q4_0 QAT checkpoints are other upstream
  checkpoints (quantization-aware fine-tunes of the `-it` models), which decision 1's derived-model rule gives a
  repository of their own, although `@karume/models/gemma` reads both `karume-gemma4` and `karume-gemma4-qat`.
