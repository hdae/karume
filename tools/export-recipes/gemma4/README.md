# Gemma 4 E2B export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)) and it produces a series, not a
distribution, so it has no dist recipe and no model card.

The checkpoint is `google/gemma-4-E2B-it`, whose weights are covered by the **Gemma Terms of Use**;
the per-revision license interview is a release gate (ADR 0065 stage 6) and nothing derived from
these weights is published from this repository today.

The authority for the design decisions is the module docstring of `export.py`; this file is the
entry point only.

## What `export.py` emits

The Gemma 4 E2B **text decoder** as one graph in the 1-shot (prefill-equivalent) shape:
`input_ids[1, T]` in, raw `logits[1, T, 262144]` out, with no KV cache. The published checkpoint is
multimodal (`Gemma4ForConditionalGeneration`), so the recipe renames the text keys
`model.language_model.*` to `model.*` and loads them into a text-only `Gemma4ForCausalLM`; the
vision and audio towers are never read, and the key/value residue that the shared-KV layers still
carry in the checkpoint is dropped using the upstream ignore list.

Four properties are what make this fixture worth having, and the export fails loudly when any of
them is lost:

- the IR `attention` nodes keep the **real grouped shape** `q[1,8,T,D]` / `k[1,1,T,D]` /
  `v[1,1,T,D]` (8:1 multi-query). Upstream `sdpa` would materialize `repeat_kv` whenever an
  attention mask is passed, so the recipe registers its own attention implementation that calls
  SDPA with `enable_gqa=True`. `D` differs **per layer type**: 256 for the 28 sliding layers
  (`config.head_dim`) and 512 for the 7 full layers (`config.global_head_dim`), and the form check
  walks the layers in order rather than assuming one value.
- causality and the window are **additive constants**, not graph inputs. The recipe hands the model
  a `{"full_attention": …, "sliding_attention": …}` mask dict, and both entries fold to a
  `Tmax × Tmax` f32 initializer sliced by `sym_prefix_slice` (ADR
  [0010](../../../docs/decisions/0010-symbolic-constant-folding.md)), so the only graph input is
  `input_ids`. The two layer types must end up on two _different_ constants — one shared constant
  would mean the window silently degenerated to plain causal.
- the **Per-Layer Embeddings table is split 35 ways**. Upstream keeps it as one
  `[262144, 35 × 256]` table, which is 9.4 GB in float32 and exceeds a single WebGPU buffer even
  when stored as int8 (2.19 GiB). The recipe builds 35 column-sliced `nn.Embedding` tables, stacks
  their lookups back into `[1, T, 35, 256]` and passes that as `per_layer_inputs`; the context
  projection and the combine step stay in upstream code. Before exporting, the split is checked
  **bit for bit** against upstream `get_per_layer_inputs` — a wrong column assignment produces the
  right shape and dtype, so nothing downstream would notice it.
- storage is **mixed**: embeddings are int8 (ADR
  [0019](../../../docs/decisions/0019-i8-weight-execution.md)) and linear weights are packed int4
  (ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md)). Neither alone works here — the
  embeddings are the majority of the parameters and are not int4-eligible, while the linear weights
  are what int4 exists for. The tied `lm_head` shares its tensor with the main embedding, so it is
  rounded exactly once, on the int8 side. The form check counts the storage dtypes of the written
  container against the two fake-quant ledgers, because a weight that falls out of the eligibility
  test is otherwise left as float32 in silence.

Output layout:

```
outputs/series/gemma4-e2b/model.safetensors     weights/constants + __metadata__.karume_ir
outputs/series/gemma4-e2b/io.<case>.safetensors input tensors and expected outputs from torch CPU
```

The io tensor key convention is the same as the tiny goldens, DeBERTa, EmbeddingGemma and MiniCPM5
(`input.<graph input name>` / `output.<position>`). The token length `T` is symbolic with an upper
bound of 768.

## Golden cases

Three fixed prompts at `T` = 6, 10 and 595: two short ones (English and Japanese) and one long
English passage. The long one is not optional — the sliding window is 512, so any case at or below
that length gets a band mask identical to the causal mask and leaves the 28 sliding layers
untested. `export.py` refuses to continue when no case exceeds the window.

Tokenization reads the checkpoint's `tokenizer.json` directly. Its post-processor adds **no**
special tokens and `tokenizer_config.json` has no `add_bos_token`; `<bos>` is emitted by
`chat_template.jinja`, i.e. it is the host's job. These goldens are raw continuations rather than
chat turns, so the recipe prepends `<bos>` itself.

The sanity check is not a tautology: for every case the greedy token at the **last** position has to
equal a specific expected continuation (`Paris` / `東京`), and the export fails if all three cases
agree on one token (a constant output). The expectations are stated in `GREEDY_EXPECTATIONS`; a
mismatch reports the measured token and the expected one, both decoded.

## Requirements

The weights go under `inputs/gemma4/gemma-4-E2B-it/` (see
[docs/assets-layout.md](../../../docs/assets-layout.md)). They are `bfloat16` on disk (10.2 GB for
the whole multimodal checkpoint). The text decoder alone is 4.63 B parameters, so it occupies about
18.5 GB once loaded as `float32`, and the export needs headroom on top of that — plan for a machine
with 24 GB of free RAM or more. The recipe deliberately avoids two easy ways to double that figure:
the packed PLE table is streamed from the file into the 35 split tables in row blocks instead of
being materialized once as a whole, and the model itself is constructed with only a handful of PLE
rows (the ones the split check reads).

The golden files are large: `io.context-en.safetensors` alone holds 624 MB of logits, and the three
cases come to roughly 640 MB.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m gemma4.export
```

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa, EmbeddingGemma and MiniCPM5 (a
change in the modeling code changes the graph shape); it is brought in temporarily with `--with`
rather than declared as a dependency group.
