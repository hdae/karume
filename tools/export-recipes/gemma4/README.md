# Gemma 4 E2B export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)). It produces four series and,
from the product one, a **distribution** (`distribution.py` / `card.py` — see "Assembling the
distribution" below).

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The checkpoint is `google/gemma-4-E2B-it`, licensed **Apache 2.0** (the snapshot's own README
frontmatter says `license: apache-2.0`, and its `license_link` page carries the plain Apache 2.0
text — the earlier claim here that the weights fell under the Gemma Terms of Use was wrong; that
was Gemma 3 knowledge carried over, retracted 2026-09-01). The per-revision license interview (ADR
0065 stage 6) was held for this revision on 2026-09-01: distribution is allowed, the model card
carries the Apache 2.0 attribution and links to the upstream model card for the usage details this
project does not curate. Nothing derived from these weights has been **uploaded** from this
repository yet — the distribution form is assembled locally and publication is a separate step
(`docs/release-runbook.md`).

The authority for the design decisions is the module docstrings (`export.py`, `export_decode.py`,
`export_product.py`); this file is the entry point only.

Four scripts export a graph from the same checkpoint (`tokenizer.py` below reads only its
`tokenizer.json`). `export.py` emits the 1-shot (prefill-equivalent) graph and
`export_decode.py` emits the states-form chunk graph that the generation path is accepted against;
`export_token.py` and `export_product.py` are two further exits on that same chunk graph. The
product series is the one a shipped pipeline would run; the other three are acceptance fixtures and
are deliberately kept.

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

Three fixed prompts at `T` = 6, 10 and 598: two short ones (English and Japanese) and one long
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

## What `export_decode.py` emits

The same checkpoint as **one states-form chunk graph**: `input_ids[1, M]` and the four RoPE rows
`rope_{sliding,full}_attention_{cos,sin}[1, M, headDim]` in, `logits[1, M, 262144]` and
`token[1, M, 1]` out, with the KV held in 30 named state slots
instead of graph I/O. It is the acceptance fixture for ADR
[0066](../../../docs/decisions/0066-generation-context-state-slots.md) (the generation context and
named state slots), ADR
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decisions 4 to 5b
(state-referencing attention and `state_append`), ADR
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) decision 4 (the decode exit) and
ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md) (packed int4) — the first fixture that
carries a sliding window, shared KV and mixed storage at once.

`M` is the physical chunk extent — `chunkLength` when prefilling, 1 when decoding — and only its
leading `queryLength` rows are valid. Capacity is declared per layer type: the 6 full slots carry
the symbol `C`, chosen by `createGenerationContext` at run time, while the 24 sliding slots bake
`sliding_window` as a literal — their ring closes at exactly `window` rows, so a symbolic capacity
would only reserve rows that can never be read (ADR 0066 addendum 9).

Three structural differences from the 1-shot recipe, and the export fails loudly when any is lost:

- **RoPE cos/sin are graph inputs the host generates per chunk.** In the 1-shot shape a position is
  always its row index, so cos/sin fold into constants; here a position is `pastLength + row` and is
  only known at run time. The recipe swaps `model.model.rotary_emb` for a pass-through module that
  returns the cos/sin the wrapper received as inputs — **one pair per layer type**
  (`rope_<layer_type>_<cos|sin>`, `[1, M, head_dim]`), because Gemma 4 calls its rotary with a
  `layer_type` argument and the two types differ in both head dim and RoPE type. No position table
  ships and there is no `position_ids` input: the distribution declares the per-layer-type formula
  parameters (`pipelineConfig.rope` = theta / headDim / rotaryDim, derived from the checkpoint's
  `config.json`) and the TypeScript host is the reference implementation (f64 math, stored as f32).
  `gemma4/rope.py` mirrors that formula in numpy and is checked against the upstream module's
  actual output with a position-proportional tolerance (upstream computes in f32, so bit-equality
  is not attainable — see `tests/test_rope.py`). The graph gate rejects any leftover baked table.
- **only the 15 KV-owning layers get slots.** Layers 15 to 34 share key/value states upstream, so
  they read the slots of the last sliding layer (13) and the last full layer (14) rather than
  declaring their own. The surgery in `karume.states` requires every reader of a slot to agree on
  the derived shape, the window and the tensor written into it, which is what proves the mapping is
  the one upstream actually wired.
- **the masks are scaffolding for the trace only.** The graph is exported with the same two additive
  masks as the 1-shot recipe, and `karume.states.to_states_form` then drops them — causality and the
  window become predicates over `pastLength + row`, so both `Tmax × Tmax` constants and their
  `sym_prefix_slice` nodes are pruned. The form check requires that no residue survives. The window
  is declared as the plain `config.sliding_window`: the runtime predicate
  `col <= limit && (limit - col) < window` and the upstream mask function are the same inclusion,
  self included.

Output layout:

```
outputs/series/gemma4-e2b-decode/model.safetensors         weights/constants + karume_ir
outputs/series/gemma4-e2b-decode/io.<case>.safetensors     unpadded inputs and expected outputs
outputs/series/gemma4-e2b-decode/greedy.<case>.safetensors greedy continuation of K = 16 steps
```

The `io.*` files use the same key convention as the 1-shot series and cover all three cases at their
full unpadded length. They are a reference table rather than a run script: the runtime executes the
graph in chunks and compares only the valid rows. On padding rows only the states-form attention
output is exactly zero (ADR 0066 addendum 8); the MLP and lm_head still write meaningless values
there, so padding rows of the graph outputs must not be read.

The `greedy.*` files are the acceptance record for the decode path — `prompt` i32 `[T]`, `expected`
i32 `[K]` and `margin` f32 `[K]`. The continuation is recomputed from scratch at every step (a full
re-forward, never a KV cache: the expectation must not come out of the mechanism under test). Every
step has to keep a top1−top2 logit margin above `2.5e-2` so that GPU-side deviation cannot flip the
sequence — the floor MUST sit above the acceptance gate's own premise of `2 × atol` (atol = 1e-2),
otherwise a case the recipe accepts could still fail the gate. A case that fails it is removed from
`GREEDY_CASES` rather than shortened, and the
gate reports every offending case at once so one run settles the set. The first continuation token
is additionally checked against the 1-shot recipe's expectation table, which is what ties the two
graphs together.

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
cases come to roughly 640 MB. The decode series writes the same io files plus three small greedy
records.

`export_decode.py` needs the same machine, and takes considerably longer: the greedy record is
`3 cases × 16 steps` of full re-forwards, and the long case re-forwards ~600 tokens each time.

## What `export_token.py` emits

The **token-only default exit** (ADR
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) addendum 4) as a third series,
`outputs/series/gemma4-e2b-decode-token/`: the same states-form chunk graph with a `last_row[1]`
i32 input added, and the exit replaced by a runtime row select (`F.embedding` over the final
hidden states), a single-row `lm_head` + softcap, and `argmax` — the only output is `token[1,1,1]`
and no logits are declared. It carries no golden files of its own: the acceptance test replays the
`greedy.<case>` records of the logits opt-in series against this graph, so the cross-series
token-for-token match is itself the gate.

Because that gate borrows another series' expectations, the export writes a `reference.json` next
to the container: the fingerprint (sha256 and byte count) of the source checkpoint files it read,
and the same digests for every borrowed `greedy.<case>.safetensors`. The acceptance test verifies
those digests before it replays anything, so a combination where only one of the two series was
regenerated fails loudly instead of passing quietly. Recording it needs no re-export of the logits
series — the existing golden bytes are only read.

## What `export_product.py` emits

The **product graph**, `outputs/series/gemma4-e2b-product/`: the same states-form chunk graph with
two changes that ship together in a single re-export (ADR
[0083](../../../docs/decisions/0083-generation-api-surface.md) consequences, plan α).

- The **per-layer embeddings leave the graph.** They are a pure row lookup over `input_ids` alone,
  so they become an ordinary graph input `per_layer_inputs[1, M, 35, 256]` supplied by the host.
  Nothing in the runtime contract changes — there is no "pageable initializer" (ADR
  [0085](../../../docs/decisions/0085-ple-host-gather.md) decision 6). The container drops from
  3,787 MiB to **1,512 MiB** (measured), because every initializer is given a resident GPU slot at
  session build time and 2,240 MiB of int8 tables plus 35 MiB of per-row scales were exactly that.
- The **exit is the final-row logits** `logits[1, 1, 262144]`, i.e. `export_token.py`'s wiring with
  the `argmax` removed. Sampling, temperature, top-k and the RNG stay on the host (ADR 0083
  decision 6); the read-back for a prefill chunk drops from 32 MiB (`[1, M, V]`) to 1 MiB.

The tables are redistributed as a **sidecar** next to the container:

```
outputs/series/gemma4-e2b-product/ple.json                        index: 262,144 tokens / 35 / 256 / embed scale
outputs/series/gemma4-e2b-product/ple-NNNNN-of-NNNNN.safetensors  values [rows,35,256] i8 + scales [rows,35] f32
outputs/series/gemma4-e2b-product/ple.probe.safetensors           the dequantization reference
```

The layout is **token-major and sharded by vocabulary range** (ADR 0085 decisions 1 and 2), which
makes one token's PLE a single contiguous 9,100-byte read; a table-major layout would need 35
scattered reads per token the moment a host wants to read rows instead of whole files. Splitting is
not an optimization here: the full int8 table is 2,348,810,240 bytes and a single Chromium
`ArrayBuffer` tops out at 2,145,386,496. The per-shard ceiling is the one constant from ADR
[0090](../../../docs/decisions/0090-shard-spec-v3-tensor-pieces.md) (256 MiB, measured as the file
length), and the writer fills at most `SHARD_DATA_CAPACITY` = 256 MiB − 1 MiB of header allowance,
which puts the real model at nine shards. The count is not a constant of this document: it moves
with every re-export, and the value that holds today is pinned by
`tests/test_export_product.py::test_the_real_model_lands_on_nine_shards`. The sidecar is not an IR
container, so the graph-shard contract does not apply to it — only the byte ceiling and the
`-NNNNN-of-NNNNN` spelling are shared.

Two properties are checked inside the export, because both fail with the right shape, dtype and
element count:

- the re-layout is **bit-identical** to the 35-table path. The reference is what
  `ple.per_layer_inputs` computes from the fake-quantized tables — that is, the value the in-graph
  `embedding` and the `mul` after it used to produce — and the check reads the written shard bytes
  back rather than comparing in memory. A scale shifted by one layer, or a shard range off by one
  row, yields a valid row of a _different_ token.
- the sidecar row count comes from the **split tables**, never from
  `config.vocab_size_per_layer_input`: `load_model_and_tables` replaces that field with the 8 probe
  rows it keeps on the model, and reading it there produces an 8-token sidecar that is internally
  consistent. It must also equal the main embedding's vocabulary, which is the writer's half of the
  id-space cross-check (ADR 0085 decision 5).

Like `export_token.py` this series takes no goldens of its own and writes the same `reference.json`
binding. The acceptance test is `packages/models/tests/e2e_gemma4_product_test.ts`: it dequantizes
through the host loader (`packages/models/src/gemma/ple.ts`), takes `argmax` on the host, and
requires the resulting token sequence to match the logits opt-in series' `greedy.<case>` records
exactly for 3 cases × 16 steps.

## What `tokenizer.py` emits

The **compiled tokenizer asset**, `outputs/series/gemma4-e2b-tokenizer/tokenizer.json` (~9.6 MB):
the id-ordered vocabulary, the merge table as id pairs, the 256 byte-fallback ids stated
explicitly, the added tokens and the special-id set — nothing else. The upstream `tokenizer.json`
is 32.2 MB and is not distributed; anything outside the accepted shape (normalizer, pre-tokenizer,
decoder chain, BPE flags, post-processor) fails at compile time rather than at run time (ADR
[0084](../../../docs/decisions/0084-gemma-tokenizer-chat.md) decision 1). It touches no model
graph — the scope is "string in, id list out" only.

The same run also writes the git-tracked parity fixture
`packages/models/tests/fixtures/gemma-text/gemma4-parity.json`. Its expectations are taken by
calling upstream `tokenizers.Tokenizer` **independently**, so no preprocessing is shared with the
TypeScript port (decision 7); the vocabulary and merges it carries are the subset the cases need.

EmbeddingGemma runs through the same machinery (`embeddinggemma/tokenizer.py`): the implementation
is shared, the assets are not.

## What `chat.py` emits

The git-tracked chat parity fixture
`packages/models/tests/fixtures/gemma-text/gemma4-chat.json`: for each plain-conversation case, the
**rendered string** and the **token id list** taken from upstream `apply_chat_template`
independently (decision 7 again — the TypeScript renderer is never in the path), plus the vocabulary
subset the cases need and the stop-token set declared by `generation_config.json` with its
spellings.

The scope of the first release is a **plain conversation** only: roles `system` / `developer` /
`user` / `assistant` with string content. Tools, thinking (`reasoning`), tool calls and
image/audio parts are **not** harvested, and the TypeScript side rejects them loudly rather than
ignoring them (ADR [0084](../../../docs/decisions/0084-gemma-tokenizer-chat.md) decision 5). The
turn markers of this checkpoint are `<|turn>` / `<turn|>` / `<|channel>` / `<|think|>` — the
Gemma 3 spelling `<start_of_turn>` is not in this vocabulary at all.

The chat format and the stop-token set are harvested together on purpose: both come from the
checkpoint's companion files, and picking them up separately is how one of them goes stale
(decision 5, "same digest set"). The TypeScript port derives the stop set from the compiled
tokenizer's added tokens and this fixture is what that derivation is checked against.

## Running

```sh
cd tools/export-recipes
uv run --with 'transformers==5.14.1' python -m gemma4.export
uv run --with 'transformers==5.14.1' python -m gemma4.export_decode
uv run --with 'transformers==5.14.1' python -m gemma4.export_token
uv run --with 'transformers==5.14.1' python -m gemma4.export_product
uv run python -m gemma4.tokenizer   # tokenizer asset + parity fixture (no transformers needed)
uv run --with 'transformers==5.14.1' python -m gemma4.chat   # chat parity fixture
```

Both fixture writers print the `deno fmt` command to run afterwards — the committed shape of a
fixture is whatever the repository formatter produces (`deno task verify` checks it).

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa, EmbeddingGemma and MiniCPM5 (a
change in the modeling code changes the graph shape); it is brought in temporarily with `--with`
rather than declared as a dependency group.

## Assembling the distribution

```sh
cd tools/export-recipes
uv run python dist.py --pipeline gemma4        # → models/karume-gemma4/ (~4.0 GiB)
```

The distribution folds **two series** into one HF repository: the product container
(`gemma4-e2b-product`, split at the same 256 MiB ceiling — seven shards as it is exported today)
plus its PLE sidecar, and the compiled tokenizer asset
(`gemma4-e2b-tokenizer`). The acceptance-only files that live beside the product container
(`ple.probe.safetensors`, `reference.json`) are not in the placement table and therefore never
reach the output. Layout inside the repository:

| Manifest seat                        | Path                                            |
| ------------------------------------ | ----------------------------------------------- |
| `weights.model.i4.shards`            | `e2b/model/model.i4-NNNNN-of-NNNNN.safetensors` |
| `assets.tokenizer`                   | `e2b/tokenizer/tokenizer.json`                  |
| `assets.ple_index`                   | `e2b/ple/ple.json`                              |
| `assets.<the index's own file name>` | `e2b/ple/ple-NNNNN-of-NNNNN.safetensors`        |

The PLE sidecar rides in the `assets` seat rather than `weights` — it is not an IR container, and
the host reads only the vocabulary ranges a conversation touches (ADR
[0085](../../../docs/decisions/0085-ple-host-gather.md) decision 3). **The asset name of a sidecar
shard is the file name the index itself writes**, so `packages/models/src/gemma/ple.ts` can look up
a fetch key with the one spelling it already has (`ple.json`'s `shards[].file`); introducing a
second naming would make the correspondence positional, and a reordering of either side would pass
silently.

`pipelineConfig` splits the way Irodori's does. Derived from the checkpoint's own files, never
copied by hand: `maxPosition` (`text_config.max_position_embeddings` — the model's declared position
limit, 131,072 for E2B), `rope` (per layer type: `theta` / `headDim` / `rotaryDim`, derived from
`rope_parameters` and the head dims; any rope type other than `default` / `proportional`, or a
scaling factor other than 1, fails the export) and `sampler` (`generation_config.json` —
temperature / top-k / top-p, ADR [0083](../../../docs/decisions/0083-generation-api-surface.md)
decision 7). Declared as **defaults for runtime knobs**, because no asset can state them:
`chunkLength` (768) and `capacity` (4,096). Alongside them, `maxChunkLength` (768) states the
traced upper bound of the chunk symbol — a copy of `export.SYM_MAX`, and the one number a reader
cannot recover from the asset, because an IR's `symbols` are names without ranges. Both knobs can be
overridden at load time; the loader keeps `chunkLength ≤ maxChunkLength` and
`chunkLength ≤ capacity ≤ maxPosition`, and the assembly refuses declarations that violate it. The
full-attention KV slots are the only thing that grows with capacity (12 KiB per token), so
`requiredLimits` is baked for `maxPosition`, the largest capacity the distribution allows.

Gates that run **before a single byte is placed** (each one covers a mismatch that leaves shape,
dtype and manifest all correct, and shows up only as wrong values):

- the container carries both `I4` (linear weights) and `I8` (embeddings) and no `F16`
- graph inputs are exactly `input_ids` / `rope_{sliding,full}_attention_{cos,sin}` /
  `per_layer_inputs` / `last_row`, in that order, the exit
  is `[1, 1, V]`, and exactly one symbol is free of the input shapes (the full slot's capacity)
- `per_layer_inputs`' layer and dim axes match the sidecar index
- the sidecar index is a gap-free ascending partition of `[0, tokens)`, `tokens` equals `V`, and
  every shard's tensors and `__metadata__.karume_ple` name the same generation as the index
- the compiled tokenizer names the compile format and has exactly `V` rows
- the checkpoint's recommended sampler is present and inside the range the TypeScript
  `pipelineConfig` parser accepts

Apache 2.0 §4 applies to the redistribution, so the repository root also carries `LICENSE.md` (a
verbatim copy of `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (the list of modifications). Both are
`karume.dist`'s legal-text seat, not manifest-declared assets.

Loading it back is `Gemma4Pipeline.fromPretrained` (`packages/models/src/gemma/pipeline.ts`). The
pinned fetch source for the published repository is `GEMMA4_SOURCES["gemma4"]`
(`@karume/models/gemma` — ADR [0073](../../../docs/decisions/0073-models-source-pin.md) /
[0092](../../../docs/decisions/0092-distribution-repos-and-sources.md)); a locally built
distribution is spelled out by the caller, since `fromPretrained` has no default source.
