# MiniCPM5-1B export recipe (1-shot)

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and it produces a series, not a
distribution, so it has no dist recipe and no model card.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstring (`export.py`); this file is the entry point only.

## What it emits

MiniCPM5-1B as **one graph** in the 1-shot (prefill-equivalent) shape: `input_ids[1, T]` in, raw
`logits[1, T, 130560]` out, with no KV cache. It is the acceptance fixture for ADR
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decision 1 — grouped
query attention as a divisible broadcast — on a real checkpoint. The state slots and the decode
path (ADR [0066](../../../docs/decisions/0066-generation-context-state-slots.md)) are a later wave.

Two properties are what make this fixture worth having, and `export.py` fails the export loudly
when either is lost:

- the IR `attention` nodes keep the **real grouped shape** `q[1,16,T,128]` / `k[1,2,T,128]` /
  `v[1,2,T,128]`. Upstream `sdpa` would materialize `repeat_kv` into `[1,16,T,128]` whenever an
  attention mask is passed, which is numerically identical and useless as an acceptance fixture, so
  the recipe registers its own attention implementation that calls SDPA with `enable_gqa=True`.
- causality is an **additive constant**, not a graph input: the mask folds to a `Tmax × Tmax` f32
  initializer sliced by `sym_prefix_slice` (ADR
  [0010](../../../docs/decisions/0010-symbolic-constant-folding.md)), so the only graph input is
  `input_ids`.

Output layout:

```
outputs/series/minicpm5-1b/model.safetensors     weights/constants + __metadata__.karume_ir
outputs/series/minicpm5-1b/io.<case>.safetensors input tensors and expected outputs from torch CPU
```

The io tensor key convention is the same as the tiny goldens, DeBERTa and EmbeddingGemma
(`input.<graph input name>` / `output.<position>`). The token length `T` is symbolic with an upper
bound of 512; the golden cases are four fixed prompts (two English, two Japanese) at `T` = 6, 12, 61
and 87. Each case stores `T × 522 KB` of logits (86 MB for the four), which is why they stay short.

The sanity check is not a tautology: for every case the greedy token at the **last** position has to
equal a specific expected continuation (` Paris` / `東京`), and the export fails if all four cases
agree on one token (a constant output).

## Requirements

The weights go under `inputs/minicpm5/MiniCPM5-1B/` (see
[docs/assets-layout.md](../../../docs/assets-layout.md)). They are `bfloat16` on disk (2.2 GB) and
are loaded as `float32`, so the export needs several gigabytes of RAM and writes a `float32`
container of roughly the same multiple.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m minicpm5.export
```

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa and EmbeddingGemma (a change in
the modeling code changes the graph shape); it is brought in temporarily with `--with` rather than
declared as a dependency group.
