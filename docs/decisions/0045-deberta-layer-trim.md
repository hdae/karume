# 0045: DeBERTa text_encoder: 22 layers, one output, relative-position index tables as inputs

- Status: accepted (2026-08-11; wave 1 = layer cut, wave 2 = single output, wave 3 = relative-position
  index tables as graph inputs; all three waves kept the WAV sha256 check green and unchanged)
- Date: 2026-08-11
- Related: [0026](0026-w8a8-deberta-deployment.md) (i8 assets and listening check) /
  [0039](0039-sbv2-distribution.md) (SBV2 distribution — this ADR updates its size table) /
  [0013](0013-sbv2-chain-export.md) (precedent for promoting a relative-position table to a graph
  input) / [research/2026-08-11-deberta-size-recon.md](../research/2026-08-11-deberta-size-recon.md)
  (canonical measurements)

## Context

SBV2 reads **only one** tensor from DeBERTa, `hidden_states[-3]` (upstream
`nlp/japanese/bert_feature.py`: `res["hidden_states"][-3:-2]`). `hidden_states` has length 25 with
`[0]` = embedding output and `[i+1]` = output of layer i, so `[-3]` is **index 22 = the output of
layer 21**. The weights of `encoder.layer.22` and `.23` are never used by the distribution: 25,346,048
B (7.6%) of the 334,545,336 B i8 asset were dead weights.

The trigger was the maintainer's question (2026-08-11) whether the ONNX conversion
`tsukumijima/deberta-v2-large-japanese-char-wwm-onnx`, which exports only hidden[-3], could help
shrink the size. Measured: **the ONNX graph does have 22 layers, but the file is only 1.03%
smaller** (onnxsim's constant folding bakes the position projections into each layer, `[16,512,64]`
fp32 × 2 = +92.3 MB, almost cancelling the −100.8 MB of the two removed layers). **Only the fact
"22 layers are enough" is worth taking**; there is no reason to align with the ONNX artefact (the
Karume i8 is about half of the ONNX fp16 version).

## Decision

### 1. The distributed text_encoder is the 22-layer variant (`sbv2-22layer`)

`VARIANTS` in `tools/export-recipes/deberta/export.py` gains 22, and `SBV2_TEXT_ENCODER_VARIANT` in
`tools/export-recipes/sbv2/distribution.py:110` points at it. The `full-24layer` variant MAY be
kept: reading per-layer error growth from the golden (the basis of the tolerance derivation in ADR
0026 decision 2) works only with 24 layers. (Re-graded 2026-10-18 to MAY: that use is now limited
to the opt-in upstream comparison and a tolerance re-derivation — `packages/e2e/tests/e2e_deberta_test.ts`
compares with the torch goldens only with `KARUME_UPSTREAM_PARITY=1`, ADR 0129 decision 4; the
default verification does not need it.)

The trim reuses the implementation `load_model` already had (`model.encoder.layer[:num_layers]`);
no new code.

### 2. The extraction position is kept in separate constants for the reference side and the distributed graph

`BERT_HIDDEN_FROM_END = 3` in `tools/export-recipes/sbv2/demo.py` is **the definition of the
upstream reference** (a position in the full 24-layer torch model) and does not move. The position
of the same tensor in the distributed graph is a separate constant, `BERT_GRAPH_HIDDEN_FROM_END = 1`
(the last output of the 22-layer graph), and that one is written to `symbols.json`.

**One constant MUST NOT serve both.** If the reference computation (`sbv2/demo.py`,
`sbv2/measure_quant.py`) read the value from `symbols.json`, then the moment layers are cut **only
the reference would point at a different layer**, and the parity scripts would report match and
mismatch inverted.

### 3. The distributed graph outputs a single hidden state (wave 2)

The runtime reads back **every** entry of `graph.outputs` (`#readOutputs` in
`packages/runtime/src/runtime/executor.ts`). Shipping all layers as outputs would make SBV2, which
uses one, pay staging + mapAsync for 22 outputs on every run. In the trimmed variant the last layer
is the one SBV2 uses, so narrowing loses nothing.

The output shape is kept together with the layer count in `VARIANTS` (`Variant` dataclass) of
`deberta/export.py`. **The distribution form (one output) and the verification form (all layers)
emit tensors of the same shape, so a mix-up still runs**; that is why the name and the output shape
are written in one place. The 24-layer variant MAY keep all-layer outputs, for the same opt-in use
as decision 1 (re-graded 2026-10-18 to MAY, same reason).

### 4. Assembly check: 22 layers × one output × `bertHiddenFromEnd` 1 (MUST)

The layer count and output shape live in the variant of `deberta/export.py`, the extraction position
in a constant of `sbv2/demo.py` — **separate scripts**, so a distribution where only one of them
moved assembles normally. The shapes still match, it loads and runs, and **it silently produces
audio from the BERT features of another layer** (the same mechanism as the row-count checks of the
style and speaker tables).

`assert_bert_hidden` (`tools/export-recipes/sbv2/distribution.py:662`) checks three things
separately; each catches a different mix-up:

| Check                                        | Mix-up it catches                                                     |
| -------------------------------------------- | --------------------------------------------------------------------- |
| layer count from initializer names == 22     | trim forgotten (the last output of 24 layers is layer 23, another one) |
| `graph.outputs` == 1                         | an all-layer verification asset slipped into the distribution path    |
| `bertHiddenFromEnd` == 1                     | only `symbols.json` is stale (`python -m sbv2.demo assets` not re-run) |

The layer count is taken from initializer names (`p_model_encoder_layer_<i>_…`) because **with a
single output the graph's `outputs` no longer tell which layer it is**. The wave-1 check was
"outputs − position == 22"; wave 2 broke that formula, so the check was rebuilt.

Wave 3 added a check of the **graph input order** (`input_ids` / `attention_mask` / `c2p_pos` /
`p2c_pos`). If the tables drop out of the inputs, the 2 MiB constants are baked back in, but **the
values stay correct, so e2e cannot catch it** (only the size silently regresses).

### 5. The relative-position index tables are graph inputs (wave 3)

`c2p_pos` / `p2c_pos` used by `disentangled_attention_bias` are **constants determined by T alone**,
so torch.export baked two `[1,512,512]` i32 tables at Tmax = 512 (2,097,152 B). They are Toeplitz
and **transposes of each other** (`p2c[i,j] = c2p[j,i]`, because the bucketing is odd), with only
511 distinct values: **in substance a 1,023-element i32 vector**.

For the same reason as ADR 0013 for the `(T,T)` tables of flow / voice, they move to external
supply. Three parts:

- `tools/export-recipes/deberta/patch.py` replaces
  `DisentangledSelfAttention.disentangled_attention_bias` and takes the tables as a tuple (**no
  fallback** — silently falling back to the original would stay green with 2 MiB baked, and nobody
  would see that the patch came off; a mismatch raises `TypeError`).
- `HiddenStatesWrapper` calls embeddings + encoder directly instead of `DebertaV2Model.forward`
  (which has no way to pass `relative_pos` to the encoder).
- The host builds the tables with the same formula in
  `packages/models/src/sbv2/text/rel-pos-tables.ts`, taking the bucket rule (`positionBuckets` /
  `maxPosition`) from `bertRelPos` in `symbols.json` (DeBERTa config values are not copied into
  the host — the same discipline as ADR 0039 decision 3).

**The host computes in float64, torch in float32**; equality was measured at T = 11 / 35 / 128 /
300 / 512. This is an empirical claim: re-measure when the bucket parameters change. Byte identity
is pinned by `packages/models/tests/sbv2_rel_pos_parity_test.ts` against the golden io (a wrong
formula keeps the shape, so the failure would be a silent wrong value — **gathering another
position embedding**).

## Consequences

- **Size.** The text_encoder went 334,545,336 → 307,068,768 B (−8.21%); graph 1,230 → 1,128 nodes,
  2 → 4 inputs, 25 → 1 output. Details: research
  [2026-08-11-deberta-size-recon.md](../research/2026-08-11-deberta-size-recon.md).
- **Speed.** Wave 2 cut readback to 1/23 and gained 4 to 8% on the BERT stage (RTX 3080 Ti); the
  wall-time effect for SBV2 is about 0.3%. Wave 3 was neutral (host table cost 1.2 ms at T = 512).
  Numbers measured while other CPU-heavy work runs are not used (a first A/B taken alongside pytest
  showed a false +73% regression).
- **Numerics.** The WAV sha256 check stayed green through all three waves without changing the
  reference value; the 24- and 22-layer `output.22` matched byte for byte in the golden io, and the
  torch stage showed `max abs diff 0.0` (research §4). The WAV check now runs on the public
  `karume-sbv2-jvnv` mirror with per-environment rows (`packages/e2e/tests/e2e_sbv2_wav_test.ts:15-19`;
  moved from the FN4 mirror on 2026-08-30).
- **Scope of the bit identity.** It is a claim about the f32-weight, f32-compute torch stage and the
  WAV in the measurement environment of that time. It MUST NOT be extended to the f16 path or other
  backends (the same nature as ADR 0022's "bit identity is an empirical claim").
- **Debt of wave 3.** One internal transformers function was copied and modified for 2 MiB (0.68% of
  the distribution). The `==` pin of transformers (`tools/export-recipes/pyproject.toml`) keeps it
  working; when the pin moves, the diff of `disentangled_attention_bias` must be followed. Related
  measures that were rejected or put on hold: research §3.
- The licence character does not change (cc-by-sa-4.0 ShareAlike already applied with the i8
  quantization). The published SBV2 distribution is `cc-by-sa-4.0` (ADR
  [0092](0092-distribution-repos-and-sources.md) decision 7).
- On 2026-08-11 the Hugging Face `hdae/karume-sbv2-jvnv` still carried the 24-layer encoder; every
  distribution built since passes `assert_bert_hidden` (decision 4).

## Addendum (2026-08-16) — DeBERTa numeric golden check

`packages/e2e/tests/e2e_deberta_test.ts` was ported, including an `i8/sbv2-22layer` case; it
corroborates on the runtime side that the layer cut keeps bit identity.

Withdrawn 2026-10-18: the rule "always rebuild the local `karume-sbv2-fn` and `karume-sbv2-jvnv`
together" — the FN mirror is no longer kept (`docs/assets-layout.md`, ruling of 2026-08-30).
