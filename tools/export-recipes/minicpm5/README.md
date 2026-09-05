# MiniCPM5-1B export recipes

**Outside the wheel** — these recipes are repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and they produce series, not
distributions, so they have no dist recipe and no model card.

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The authority for the design
decisions is the module docstrings (`export.py`, `export_decode.py`); this file is the entry point
only.

Two scripts read the same checkpoint: `export.py` emits the 1-shot (prefill-equivalent) graph, and
`export_decode.py` emits the states-form chunk graph that the generation path is accepted against.
A third, `sweep_w4.py`, is a measurement harness rather than an exporter — it emits no series.

## What `export.py` emits

MiniCPM5-1B as **one graph** in the 1-shot (prefill-equivalent) shape: `input_ids[1, T]` in, raw
`logits[1, T, 130560]` out, with no KV cache. It is the acceptance fixture for ADR
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decision 1 — grouped
query attention as a divisible broadcast — on a real checkpoint. The state slots and the decode path
are the second recipe below.

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
equal a specific expected continuation (`Paris` / `東京`), and the export fails if all four cases
agree on one token (a constant output).

## What `export_decode.py` emits

The same checkpoint as **one states-form chunk graph**: `input_ids[1, M]` and `position_ids[1, M]`
in, `logits[1, M, 130560]` and `token[1, M, 1]` out, with the KV held in 48 named state slots instead
of graph I/O. It is the acceptance fixture for ADR
[0066](../../../docs/decisions/0066-generation-context-state-slots.md) (the generation context and
named state slots), ADR
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decisions 4 to 5b
(state-referencing attention and `state_append`) and ADR
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) decision 4 (the decode exit, an
`argmax` node next to the logits).

`M` is the physical chunk extent — `chunkLength` when prefilling, 1 when decoding — and only its
leading `queryLength` rows are valid. The slots are declared `[1, 2, "C", 128]` with `C` left
symbolic, so the capacity is chosen by `createGenerationContext` instead of being baked in at export
time.

Two structural differences from the 1-shot recipe, and the export fails loudly when either is lost:

- **positions are a graph input and RoPE is a table lookup.** In the 1-shot shape a position is
  always its row index, so cos/sin fold into a constant; here a position is `pastLength + row` and is
  only known at run time. The recipe swaps `model.model.rotary_emb` for a module that gathers a
  512-position cos/sin table with `F.embedding`, and asserts at construction time that the lookup
  reproduces the original implementation exactly (`torch.equal`, three kinds of position vector).
- **the causal mask is scaffolding for the trace only.** The graph is exported with the same additive
  mask as the 1-shot recipe, and `karume.states.to_states_form` then drops it — causality in the
  states form is a predicate over `pastLength + row`, so the `Tmax × Tmax` constant and its
  `sym_prefix_slice` are pruned. The form check requires that no residue of either survives.

Output layout:

```
outputs/series/minicpm5-1b-decode/model.safetensors         weights/constants + karume_ir
outputs/series/minicpm5-1b-decode/io.<case>.safetensors     unpadded inputs and expected outputs
outputs/series/minicpm5-1b-decode/greedy.<case>.safetensors greedy continuation of K = 16 steps
```

The `io.*` files use the same key convention as the 1-shot series and cover all four cases at their
full unpadded length. They are a reference table rather than a run script: the runtime executes the
graph in chunks and compares only the valid rows. On padding rows only the states-form attention
output is exactly zero (ADR 0066 addendum 8); the MLP and lm_head still write meaningless values
there, so padding rows of the graph outputs must not be read — they have no counterpart on the
reference side either.

The `greedy.*` files are the acceptance record for the decode path — `prompt` i32 `[T]`, `expected`
i32 `[K]` and `margin` f32 `[K]`. The continuation is recomputed from scratch at every step (a full
re-forward, never a KV cache: the expectation must not come out of the mechanism under test). Every
step has to keep a top1−top2 logit margin above `1e-2`, so that GPU-side deviation cannot flip the
sequence; `capital-ja` is excluded from the greedy cases because its step 5 margin is 0.0077
(measured 2026-08-18). The first continuation token is additionally checked against the 1-shot
recipe's expectation table, which is what ties the two graphs together.

## What `sweep_w4.py` measures

A fake-quant screening rig for ADR
[0069](../../../docs/decisions/0069-packed-w4-storage.md) (packed 4-bit storage). It touches no
runtime code: weights are rounded in torch and the quality loss is measured, so a storage form is
implemented only after the numbers are in. Three grids run in one pass:

- **Phase 0 grid** — `group_size` {32, 64, 128} × symmetric / asymmetric, plus a baseline and one
  run that leaves `lm_head` alone. The asymmetric column keeps its zero-point continuous, so it is
  the upper bound of what a stored `zero_point` companion could recover, not a storable form
  (ADR 0069 decision 3).
- **Method grid** — 7 rounding methods (RTN i4 / FP4 / NF4 / MXFP4 / k-means at three table
  granularities) × 2 target sets (the 169 decoder linears, or those plus `embed_tokens` = 170).
  `group_size` is fixed at 32 here: Phase 0 already answers the `g` axis, and sweeping both at once
  mixes the method difference with the `g` difference.
- **Calibration grid** — 5 calibrated roundings (GPTQ against each of the three storage grids /
  AWQ / AWQ+GPTQ) on the same lattices as the method grid; only _where inside the lattice_ a value
  lands changes. Its target set is the decoder linears alone, because the calibration driver works
  on the `nn.Linear` modules inside a stage. Calibration inputs are the 48 sentences of
  `calib_texts.py`.

The reference sequences are the wave-E greedy records (`greedy.<case>.safetensors`), so no
tokenizer is involved, and the baseline run has to reproduce them exactly before any quantized
configuration is measured. stdout ends with **four markdown tables** (summary / quality / per-family
weight RMSE / projected size) meant to be pasted into `docs/research/`; the size column is a
projection from each method's formula, not a measurement.

Four knobs:

- `--json <path>` — also write the measurements as JSON, **rewritten after every configuration**, so
  a run of tens of minutes that dies partway still leaves what it had measured. Without it the run
  writes no files.
- `--only <name>` — run just these configurations (repeatable, for partial re-runs); the baseline
  always runs first. The GPTQ sweep-axis experiment points only run when named here.
- `--calib-limit N` — calibrate on the first N sentences only (a smoke knob; affects the calibrated
  configurations only).
- `--kmeans-shared-stride N` — fit the `kmeans:shared` table on an evenly strided subsample, for
  machines where the full sample does not fit in RAM. The rounding itself always sees everything.

## Requirements

The weights go under `inputs/minicpm5/MiniCPM5-1B/` (see
[docs/assets-layout.md](../../../docs/assets-layout.md)). They are `bfloat16` on disk (2.2 GB) and
are loaded as `float32`, so the export needs several gigabytes of RAM and writes a `float32`
container of roughly the same multiple.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m minicpm5.export
uv run --with 'transformers==5.14.1' python -m minicpm5.export_decode
uv run --with 'transformers==5.14.1' python -m minicpm5.sweep_w4
```

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa and EmbeddingGemma (a change in
the modeling code changes the graph shape); it is brought in temporarily with `--with` rather than
declared as a dependency group.
