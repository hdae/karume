# EmbeddingGemma export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and wherever a card is produced
the authoritative one is the exemplary `README.md` that `dist` generates; this family has no dist
recipe yet (it writes a series, not a distribution).

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstring (`export.py`); this file is the entry point only.

## What it emits

EmbeddingGemma-300m as **one graph** covering all 5 SentenceTransformer modules
(`Transformer → Pooling(masked mean) → Dense(768→3072) → Dense(3072→768) → Normalize(L2)`), whose
output is a single unit vector `[1, 768]`. The token length `T` is symbolic; the attention mask is
not a graph input (the band masks fold to constants sliced by `sym_prefix_slice`). Prompt prefixing
and tokenization stay on the host. Output layout:

```
outputs/series/embeddinggemma-300m/model.safetensors     weights/constants + __metadata__.karume_ir
outputs/series/embeddinggemma-300m/io.<case>.safetensors input tensors and expected outputs from torch CPU
```

The io tensor key convention is the same as the tiny goldens and DeBERTa (`input.<graph input name>`
/ `output.<position>`).

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m embeddinggemma.export
uv run --with 'transformers==5.14.1' python -m embeddinggemma.export --batch 8 --out /path/to/out
```

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa (a change in the modeling code
changes the graph shape); it is brought in temporarily with `--with` rather than declared as a
group.
