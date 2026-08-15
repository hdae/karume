# DeBERTa export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and the authoritative model card
is the exemplary `README.md` that `dist` generates into the distribution directory.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Scripts are started as
modules from `tools/export-recipes/` (`uv run --with 'transformers==5.14.1' python -m
deberta.export …`). This recipe has no dist recipe of its own: the series it writes is shipped as
the `text_encoder` seat of the [SBV2 distribution](../sbv2/README.md), which picks the output up
**by path**.

## Real-weight DeBERTa export and E2E (M1-P2 wave 5)

Where the tiny goldens take on "op contract coverage", `deberta/export.py` takes on **numerical
agreement on real weights and real token sequences**. The target is the very BERT the SBV2 text
front uses (HF `ku-nlp/deberta-v2-large-japanese-char-wwm`).

```sh
# 1. generate (transformers fetches the weights and tokenizer from HF; ~1.3GB download on the first run)
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m deberta.export            # 2 layers + 24 layers
uv run --with 'transformers==5.14.1' python -m deberta.export --layers 2 # 2 layers only (development)

# 1b. i8 series (ADR 0019 storage + ADR 0025 w8a8 mirror goldens)
uv run --with 'transformers==5.14.1' python -m deberta.export --dtype i8 --act-quant

# 2. real-GPU comparison: not available in this repository yet (see the note below)
```

- **transformers is pinned to 5.14.1** (recon §6-5 — the graph shape changes when the modeling code
  changes). It is not added to `pyproject.toml` / `uv.lock`; `--with` brings it in temporarily.
- The outputs go to **`outputs/series/deberta/<variant>/`** (kept out of commits by the `outputs/`
  entry in the top-level `.gitignore` — the 24-layer weights are 1.3GB). `--dtype i8` is a
  **separate series** `outputs/series/deberta-i8/<variant>/`.
- **`sbv2-22layer` is what the SBV2 distribution ships.** SBV2 only ever reads
  `hidden_states[-3]` (= index 22 = the output of layer 21), so the last two layers are dead weight;
  the truncated model's final output is **bit-identical** to the 24-layer model's `hidden_states[-3]`
  (measured — see `docs/research/2026-08-11-deberta-size-recon.md`). That variant also emits **only
  that one tensor** instead of the whole `hidden_states` tuple, because the runtime reads every
  `graph.output` back on every run. The other variants keep the full tuple so the per-layer error
  growth stays readable off the goldens (ADR 0026).

```
outputs/series/deberta/dev-2layer/model.safetensors      2 layers (130 nodes / 208MB)
outputs/series/deberta/dev-2layer/io.<case>.safetensors
outputs/series/deberta/full-24layer/model.safetensors    24 layers (1230 nodes / 1.32GB / 25 outputs)
outputs/series/deberta/full-24layer/io.<case>.safetensors

outputs/series/deberta-i8/sbv2-22layer/model.safetensors      22 layers in i8 storage (1130 nodes /
                                                              294.5MB / 1 output) — shipped
outputs/series/deberta-i8/full-24layer/model.safetensors      24 layers in i8 storage (319MB)
outputs/series/deberta-i8/<variant>/io.<case>.safetensors       w8 goldens (activations in f32)
outputs/series/deberta-i8/<variant>/io-i8a8.<case>.safetensors  w8a8 mirror (--act-quant)
```

The io tensor key naming is the same as the tiny goldens (`input.<graph input name>` /
`output.<position>`). The only difference is that a single model has **several io files, one per
case**. There are 4 cases:

| Case     | Content                             | T  |
| -------- | ----------------------------------- | -- |
| `case0`  | short sentence                      | 11 |
| `case1`  | longer                              | 26 |
| `case2`  | long sentence with symbols          | 35 |
| `padded` | `case0` + `[PAD]`×5 (0 in the mask) | 16 |

The wrapper is `forward(input_ids, attention_mask, c2p_pos, p2c_pos) -> hidden_states` (a tuple of
all layers, or the single last one for the shipped variant). Because every layer can be compared,
**how the error grows with depth** can be read directly off the goldens (which puts the tolerances on
a measured footing). `padded` is the only case that mixes in `attention_mask=0` and therefore
exercises the mask path (mul → cast → bitwise_not → masked_fill, plus the zero fill on the conv
path).

`c2p_pos` / `p2c_pos` are the disentangled-attention gather indices. They depend only on T, so the
exporter used to constant-fold them into two `[1,512,512]` i32 tensors (2MiB of dead weight); they
are now **graph inputs** built on the host (`deberta/patch.py` is the reference
implementation, mirrored by `packages/models/src/sbv2/text/rel-pos-tables.ts`). The mirror is pinned
byte-for-byte by `packages/models/tests/sbv2_rel_pos_parity_test.ts`.

On the Deno side: the real-weight DeBERTa E2E has **not been ported into this repository yet**. The
exporter still emits the regular `io.<case>` and `io-i8a8.<case>` series, but there is currently no
in-repository `e2e_deberta*_test.ts` gate consuming them.
`packages/models/tests/sbv2_rel_pos_parity_test.ts` only pins the host-side relative-position tables
and is not a replacement for the missing full numerical E2E.

The `io-i8a8.<case>` files written by `--act-quant` are the **w8a8** (`linearCompute: "i8a8"`)
mirror. The regular `io.<case>` MUST be taken **without the hook** (taking it with the hook still
applied would contaminate the w8-side E2E expectations with activation quantization). The prefix is
kept apart from `io.` so that a Deno-side enumeration of regular cases (startsWith `io.`) does not
pick up the mirror.

Historical measurement (from the pre-migration numerical E2E, kept here only as background for
whoever ports the gate back): the w8 tolerance was a dedicated value matched to the error
accumulation of 24 layers and separate from the tiny goldens' `GOLDEN_TOLERANCE`, with **the f32 and
i8 series running through the same structure and the tolerance alone derived from measurements per
series** (no reuse across series); the w8a8 mirror was **not a numerical parity net** (activation
quantization is discontinuous, so after a few layers the GPU and torch become "different samples of
the same distribution").
