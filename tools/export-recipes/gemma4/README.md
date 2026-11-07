# Gemma 4 E2B / E4B / 12B export recipe

**Outside the wheel** — this recipe is repo-only (ADR
[0065](../../../docs/decisions/0065-exporter-core-recipe-split.md)). It produces four series and,
from the product one, a **distribution** (`distribution.py` / `card.py` — see "Assembling the
distribution" below).

Upstream provenance: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Two checkpoints are supported, chosen with `--model e2b|e4b` (default `e2b`): `google/gemma-4-E2B-it`
and `google/gemma-4-E4B-it`. Every shape the scripts check (layer count, head counts, KV sharing,
PLE size) is derived from the checkpoint's own `config.json`; the model name only picks the default
input and output directories. In the regular line E4B ships **without an MTP drafter** (the q4_0 QAT
line below ships one for every model). It has a compiled tokenizer
asset of its own: E4B shared E2B's until Google's glitch-token fixes of 2026-11-03 gave the two
upstream `tokenizer.json` files different merges (E2B drops 267 and renames id 249788, E4B drops
253; the vocabulary is the same). The license notes below were established for E2B; E4B's snapshot
carries the same Apache 2.0 frontmatter.

`google/gemma-4-12B-it` is a third model with its own script, `export_unified.py` (see "What
`export_unified.py` emits" below): upstream ships it as another class (`gemma4_unified` — no
per-layer embeddings, no KV sharing, K = V on the full layers), too large to load whole in F32
here. It has its own tokenizer series (its own glitch-token fix of 2026-10-30 drops 239 merges) and
its own chat dialect. It ships only in the q4_0 QAT line (`karume-gemma4-qat`), with its QAT MTP
drafter (`export_drafter_qat.py`) and its encoder-free vision embedder (`export_vision.py`, see
"What `export_vision.py` emits" below).

12B is read from two repositories, each from the hub cache at the commit `sources.py` pins. **The
weights and their `config.json`** come from Google's quantization-aware-trained checkpoint
`google/gemma-4-12B-it-qat-q4_0-unquantized` ("half-precision weights extracted from the QAT
pipeline, for custom downstream compilation"; its README front matter names
`google/gemma-4-12B-it` as `base_model`). Its linears already lie on a 4-bit grid, and the export
imports that grid as is instead of rounding the weights again (see "What `export_unified.py`
emits"). **The tokenizer, the chat template and the generation defaults** come from
`google/gemma-4-12B-it`: the QAT repository's `tokenizer_config.json`, `generation_config.json`
and `chat_template.jinja` are byte-identical to 12B-it's, but its `tokenizer.json` is the one every
Gemma 4 repository shipped before Google's glitch-token fixes (the QAT repositories have not
received the fix). Both snapshots' README front matter
says `license: apache-2.0`. Its drafter is a QAT checkpoint too,
`google/gemma-4-12B-it-qat-q4_0-unquantized-assistant` (Google's model card: "the assistant model
must also be a QAT checkpoint with the same precision").

The E2B checkpoint is `google/gemma-4-E2B-it`, licensed **Apache 2.0** (the snapshot's own README
frontmatter says `license: apache-2.0`, and its `license_link` page carries the plain Apache 2.0
text — the earlier claim here that the weights fell under the Gemma Terms of Use was wrong; that
was Gemma 3 knowledge carried over, retracted 2026-09-01). The per-revision license interview (ADR
0065 stage 6) was held for this revision on 2026-09-01: distribution is allowed, the model card
carries the Apache 2.0 attribution and links to the upstream model card for the usage details this
project does not curate. The distribution is published as `hdae/karume-gemma4`; the pinned
revision lives in `gemma.SOURCES` (`packages/models/src/gemma/config.ts`), and publication
follows `docs/release-runbook.md`.

The authority for the design decisions is the module docstrings (`export.py`, `export_decode.py`,
`export_product.py`, `export_drafter.py`); this file is the entry point only.

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
  (ADR [0069](../../../docs/decisions/0069-packed-w4-storage.md)). The embeddings are the majority
  of the parameters; the runtime can execute int4 embeddings too (ADR 0069, addendum 6), but this
  recipe keeps them at int8, while the linear weights are what int4 exists for. The tied `lm_head` shares its tensor with the main embedding, so it is
  rounded exactly once, on the int8 side. The form check counts the storage dtypes of the written
  container against the two fake-quant ledgers, because a weight that falls out of the eligibility
  test is otherwise left as float32 in silence.

Output layout:

```
outputs/series/gemma4-e2b/model-NNNNN-of-NNNNN.krm  the container: graph + weights/constants as a part sequence
outputs/series/gemma4-e2b/io.<case>.safetensors      input tensors and expected outputs from torch CPU
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
`sliding_window + SLIDING_SLACK_ROWS` as a literal — their ring closes at that capacity, so a
symbolic capacity would only reserve rows that can never be read (ADR 0066 addendum 9). The slack
is the speculative draft bound: a verify pass writes rows ahead of the committed length, and the
extra physical rows keep a rejected row from overwriting a column that is still inside the live
window. The declared `window` attribute itself does not move — only the physical row count does.

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
outputs/series/gemma4-e2b-decode/model-NNNNN-of-NNNNN.krm  the container: graph + weights/constants as a part sequence
outputs/series/gemma4-e2b-decode/io.<case>.safetensors      unpadded inputs and expected outputs
outputs/series/gemma4-e2b-decode/greedy.<case>.safetensors  greedy continuation of K = 16 steps
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

The weights go under `inputs/gemma4/gemma-4-E2B-it/` (E4B: `inputs/gemma4/gemma-4-E4B-it/`; see
[docs/assets-layout.md](../../../docs/assets-layout.md)). E2B's are `bfloat16` on disk (10.2 GB for
the whole multimodal checkpoint). The text decoder alone is 4.63 B parameters, so it occupies about
18.5 GB once loaded as `float32`, and the export needs headroom on top of that — plan for a machine
with 24 GB of free RAM or more. The recipe deliberately avoids two easy ways to double that figure:
the packed PLE table is streamed from the file into the 35 split tables in row blocks instead of
being materialized once as a whole, and the model itself is constructed with only a handful of PLE
rows (the ones the split check reads).

The memory figures above are for E2B. **E4B has not been measured.**

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

The **product graph**, `outputs/series/gemma4-e2b-product-bf16/` — the regular line's `model`
container: the same states-form chunk graph with two changes that ship together in a single
export (ADR [0083](../../../docs/decisions/0083-generation-api-surface.md) consequences, plan α).

- The **per-layer embeddings leave the graph.** They are a pure row lookup over `input_ids` alone,
  so they become an ordinary graph input `per_layer_inputs[1, M, 35, 256]` supplied by the host.
  Nothing in the runtime contract changes — there is no "pageable initializer" (ADR
  [0085](../../../docs/decisions/0085-ple-host-gather.md) decision 6). Every initializer is given a
  resident GPU slot at session build time, so the table's 4,697,620,480 bytes (E2B, bf16) leave the
  resident set.
- The **exit is the selected rows' logits and hidden states**, `logits[1, R, 262144]` (output 0)
  and the post-final-norm `hidden[1, R, 1536]` (output 1), i.e. `export_token.py`'s wiring with the
  `argmax` removed and the row select widened. `R` is a symbol bound by the `last_row[R]` input:
  ordinary prefill and decode bind it to 1 and are bit-identical to the single-row exit, while a
  speculative-decoding verify pass scores up to `SLIDING_SLACK_ROWS` rows in one run and feeds the
  hidden states to the drafter. Output **order is part of the contract** — the runtime reads
  outputs by slot number. Sampling, temperature, top-k and the RNG stay on the host (ADR 0083
  decision 6); the read-back for a prefill chunk drops from 32 MiB (`[1, M, V]`) to 1 MiB.

Every weight is the **checkpoint's own bf16 bits** (ADR
[0131](../../../docs/decisions/0131-bf16-weight-execution.md)): the checkpoint is BF16, so nothing is
rounded, and the writer stores a weight as bf16 only when the round trip is exact. The form check
counts every linear and the embedding (the tied head reads it) as bf16, and no i8 / i4.

The tables are redistributed as **assets of the product container**, not as weights (ADR
[0109](../../../docs/decisions/0109-manifest-v5-container.md) decision 4):

```
outputs/series/gemma4-e2b-product-bf16/model-NNNNN-of-NNNNN.krm  the container: graph + weights/constants + the PLE assets
outputs/series/gemma4-e2b-product-bf16/ple.probe.safetensors     the PLE probe reference
outputs/series/gemma4-e2b-product-bf16/greedy.<case>.safetensors the greedy goldens
outputs/series/gemma4-e2b-product-bf16/reference.json            checkpoint fingerprints and the goldens' digests
```

The PLE assets are an index, `ple_index` (role `ple-index`, schema 5: 262,144 tokens / 35 / 256 /
embed scale / storage `bf16`), and one block sequence, `ple.values.<k>` (role `ple-values`, the
checkpoint's bf16 bits `[rows,35,256]`; a float table has no scales). The layout is **token-major
and cut by vocabulary range** (ADR 0085 decisions 1 and 2), which makes one token's PLE a single
contiguous 17,920-byte read; a table-major layout would need 35 scattered reads per token the moment
a host wants to read rows instead of whole blocks. Splitting is not an optimization here: the full
bf16 table is 4,697,620,480 bytes and a single Chromium `ArrayBuffer` tops out at 2,145,386,496.
Each block holds a whole number of rows within the container's 32 MiB block ceiling, and because
the host reads rows by range, every PLE block sits alone in a part of its own (container-v1 §4.2).
The block count follows the block ceiling rather than this document; today's E2B product container
carries 141 value blocks. The cutting rule and the index spelling are core code (`karume.ple`),
shared with `karume migrate`, so the recipe and the migration produce the same blocks.

Two properties are checked inside the export, because both fail with the right shape, dtype and
element count:

- the re-layout is **bit-identical** to the 35-table path. The reference is what
  `ple.per_layer_inputs` computes from the loaded tables — that is, the value the in-graph
  `embedding` and the `mul` after it used to produce — and the check reads the PLE assets of the
  written container back rather than comparing in memory. A table shifted by one layer, or a block
  range off by one row, yields a valid row of a _different_ token.
- the PLE row count comes from the **split tables**, never from
  `config.vocab_size_per_layer_input`: `load_model_and_tables` replaces that field with the 8 probe
  rows it keeps on the model, and reading it there produces an 8-token index that is internally
  consistent. It must also equal the main embedding's vocabulary, which is the writer's half of the
  id-space cross-check (ADR 0085 decision 5).

The series takes **its own goldens** (`greedy.<case>.safetensors` — the q4_0 QAT series' rule):
greedy continuations of `export_decode.GREEDY_CASES` by full re-forward of the stored model,
`export_decode.GREEDY_STEPS` steps (`--steps`) and the margin floor `export_decode.MARGIN_FLOOR`,
taken before the trace. Nothing is rounded, so they are the upstream checkpoint's own continuations
(computed in f32). `reference.json` binds them to the container (fingerprints of the checkpoint
files read and the goldens' digests). The acceptance tests in `packages/e2e/tests/` gather the PLE
through the host loader (`packages/models/src/gemma/ple.ts`), take `argmax` on the host, and require
the resulting token sequence to match the goldens.

## What `export_unified.py` emits

The 12B product graph: E2B / E4B's product contract minus PLE — `input_ids[1,M]`,
`soft_embeds[1,M,H]` f32, `soft_mask[1,M,1]` bool, the four RoPE inputs and `last_row[R]` in,
`logits[1,R,V]` then `hidden[1,R,H]` out, one k / v slot pair per layer (96 slots, no KV sharing).
The embeddings of the rows are `where(soft_mask, soft_embeds, embed_tokens(input_ids))`: an image's
rows take the embedder's vectors, unscaled, as upstream overwrites them; a text run binds zeros and
an all-false mask, which is the text forward bit for bit. The attention over an image block is not
a graph input (the host passes it per run). Sliding slots hold 8 KV heads in `window + 8` = 1032 rows; full
slots hold 1 KV head of width 512 in the capacity symbol. On the full layers the value is the raw
`k_proj` output through the scale-free `v_norm` (not rotated) and the key is `k_norm` then RoPE of
the same projection; the script checks that wiring in the written graph. The storage is packed
int4 (group 32) for the linear weights and the embedding (the tied head) alike, with no int8 tensor
(the regular E2B / E4B series keep the checkpoint's bf16 values).

The int4 values are **the QAT checkpoint's own grid, imported as is** — the tied embedding's too:
every one of its groups of 32 lies on the grid (measured 2026-11-03). Each group of 32 along a row
holds `bf16(q·d)` with integer codes `q` in −8..7 and a learned step `d`; the export recovers, per
group, the largest step `amax / k` (k = 1..8) whose codes fit −8..7 and whose least-squares refit
reconstructs the group within 2^-7 of its largest magnitude, and stores those codes and that step.
A group that does not recover fails the export (no fallback to rounding). Measured on the pinned
snapshots (2026-11): round-to-nearest on karume's ±7 grid gives KL 2.65 / 1.88 (`context-en` /
`window-en`) for `google/gemma-4-12B-it` against its unquantized forward, re-rounding the QAT
weights to ±7 still about 1.0, and the imported grid 0.0136 / 0.0362 (top-1 agreement 0.97 / 0.93)
against the QAT checkpoint's own unquantized forward with the embedding rounded to int8 per row,
0.0091 / 0.0505 (top-1 agreement 0.99 / 0.93) with the embedding's grid imported too (2026-11-03 —
the embedding's own relative RMS error drops from 1.41 % to 0.14 %; the rise on `window-en` is not
yet explained). The −8 code needs no format change: the
packed nibble stores `q + 8` and the runtime decodes `u − 8`.

The form is the Qwen3.5 / umT5 one (an F32 load of 12B is 44.6 GiB): the trace runs on parameters
without data, the linear weights and the embedding are imported from the checkpoint a block of
rows at a time and handed over as fixed weights, and the norms are widened to F32. The
pytest checks on a tiny model that the row blocks write the same container bytes as an import of
each whole tensor at once.

The goldens (`greedy.<case>.safetensors` of the standard cases plus `window-en`, and
`reference.json` naming both sources, the series itself and the suppressed ids) come from an F32
reference whose linear weights are held as their imported codes and steps and dequantized each
time they are read, and whose embedding holds its imported grid values in F32 (ruling R4 —
bit-equal to what the container's bytes decode to, about 16 GiB resident). Before every
greedy selection the logits of the ids upstream's `generation_config.json` suppresses
(`suppress_tokens` — 258882 / 258883 for 12B) are −inf, as upstream `generate()` does; the
distribution declares the same ids as `pipelineConfig.suppressTokens`. `window-en` (1159 tokens) is
longer than the 1032-row sliding ring, so the goldens evaluate the sliding window and the host's
prefill spans two 768-row chunks; the export fails if no golden runs past the window.

    outputs/series/gemma4-12b-product/model.krm                  the container (no PLE assets)
    outputs/series/gemma4-12b-product/greedy.<case>.safetensors  greedy goldens (4 cases)
    outputs/series/gemma4-12b-product/reference.json             provenance (own goldens)

## What `export_vision.py` emits

12B's encoder-free vision embedder as its own graph: `pixels[1,N,6912]` f32, `position_x[1,N]` and
`position_y[1,N]` i32 in, `embeds[1,N,3840]` out, `N` symbolic (the host strips the padding, so the
budget is the host's: 280 today, no re-export for another). It is upstream's
`Gemma4UnifiedVisionEmbedder` with its own modules; only the position lookup is rewritten, as
`functional.embedding` on the two columns of `pos_embedding`. Pytest holds the wrapper to the
upstream module bit for bit in F32, and the export does the same on the real weights.

The weights are the QAT checkpoint's `model.vision_embedder.*` and `model.embed_vision.*`, which are
BF16 and not on the q4_0 grid: the linears and the two position tables are stored as bf16 (the
regular line's writer path, ADR 0131), the norms' affines and the dense bias as F32 holding the same
values. The audio embedder's keys are never read.

The goldens, taken before the container on the three neutral corpus images
(`outputs/test-assets/corpus/wan-i2v/`):

- `vision.<image>.safetensors` (`pixels_shape`, `positions`, `embeds`): upstream's
  `Gemma4UnifiedImageProcessor` at budget 280 and upstream's module. `reference.json` records the
  SHA-256 of the processor's pixel bytes, for the TS preprocessing.
- `greedy.<image>.safetensors` (the decode series' `prompt` / `expected` / `margin`, plus `block`
  and `first_logits`), for two of the images: `reference_wrapper` (F32 over imported i4) by full
  re-forward with the masks of upstream's `create_masks_for_vision_model` — full layers causal,
  sliding layers `window AND (causal OR block)`, where a block is the run of one image's soft tokens,
  `<|image>` and `<image|>` excluded. The prompt is one image part then "Describe this image in one
  sentence.", and its ids must equal those of upstream's `Gemma4UnifiedProcessor`. An image with a
  step at or below the margin floor is dropped for the next, never cut.
- `reference.json`'s `maskPaths` measures one image's first step against upstream's other mask rule
  (the block ORed onto the full layers too): largest logit difference, KL and top-1 agreement. It
  is a measurement; nothing ships that rule.

The reference holds about 16 GiB; the embedder itself takes about a second per image, so
`--embedder-only --out <dir>` is the cheap run of the real weights, and `--plan` reads no weight.

    outputs/series/gemma4-12b-vision/model.krm                   the container (no assets)
    outputs/series/gemma4-12b-vision/vision.<image>.safetensors  embedder goldens (3 images)
    outputs/series/gemma4-12b-vision/greedy.<image>.safetensors  greedy goldens (2 images)
    outputs/series/gemma4-12b-vision/reference.json              provenance, measurements

## What `export_drafter.py` emits

The **MTP drafter** series, `outputs/series/gemma4-e2b-drafter/` — a _borrower_ graph that cannot
run on its own. It is the head from `google/gemma-4-E2B-it-assistant` (4 layers, hidden 256, no key
or value projections at all), re-expressed so that it reads the _product_ series'
(`gemma4-e2b-product-bf16`) state slots and embedding table instead of carrying its own. Three
declarations in the container say so (ADR
[0096](../../../docs/decisions/0096-speculative-decoding.md) stage 2):

- the four state slots are **`external`** — the physical rings belong to the lender's
  `GenerationContext`, and the borrower writes none of them (no `state_append` survives)
- every attention node is **`readonly`** — its `ins` is the single query row, and the columns it
  reads are `[P − min(P, W), P)` for the sliding slot and `[0, P)` for the full one
- the main embedding table is a **`shared`** initializer — no bytes are written for it; the
  declaration names the lender's tensor key (`model.lm_head.weight`), which the recipe reads out of
  the product container's header rather than spelling it out

The graph takes `token[1,1]`, `hidden[1,1536]` (the product graph's output 1, i.e. after the final
norm and before `lm_head`) and the four rotary rows for position `P`, the position of the token
being handed in (see _Goldens_ below for the whole calling convention), and returns **k = 3**
drafted token ids. The three steps are unrolled in Python because the only state carried between
them is `(token, hidden)` — the drafter has no cache of its own. The clustered sparse output head of
the upstream checkpoint is replaced by a dense projection over the full vocabulary, because `topk`
has no `aten` handler in this exporter; `use_ordered_embeddings` is turned off **before** the
checkpoint is read, since `masked_embedding` is built in `__init__`.

Storage is **int8 throughout** — every `nn.Linear` including the tied output head, all
per-channel int8; only the norms and the layer scalars stay f32. The shared table is declared
`bf16` and is not rounded: it is the lender's table, the checkpoint's bf16 values the product
series stores. The reason for int8 is measured (2026-09-08, same target trajectory, N = 200
cycles × 3 cases): rounding the drafter's 22 linear layers (10.1M elements) to int4 group-32 drops
E[a], the expected number of tokens confirmed per speculation cycle at k = 3, by 33% / 21% / 15%
against no rounding at all, while per-channel int8 costs 2–3%. The int4 → int8 step adds under
4 MB to the distribution, which is not a trade worth a fifth of the acceptance rate.

### Goldens

`drafter-golden.<case>.safetensors` carries `prompt[T]`, `tokens[N + 3]` (the target's greedy
continuation, teacher-forced) and `draft[N, 3]` for three cases and N = 200 cycles. They are taken
by running the target **as the product series stores it** (the checkpoint's bf16 values, nothing
rounded) next to the exported drafter wrapper:
the target is prefilled in chunks of 768 rows through a `DynamicCache`, and at each cycle — with
`P` standing for the number of rows the cache holds — the drafter is handed

- `token`: the last token the target committed to. It sits at logical position `P` and is **not in
  the cache yet**
- `hidden`: the post-norm hidden of the row that produced that token, i.e. position `P − 1`
- the four rotary rows for position `P`
- the lender's key/value columns **sliced by the recipe** to `[P − min(P, W), P)` / `[0, P)` — the
  slice is not left to the cache's own retention, which differs between a chunked prefill and a
  single decode step

That pairing — a token together with the hidden of the row that produced it — is the same one the
drafter carries between its own three steps, and the one the speculative loop hands it on the first
cycle. So `draft[t][j]` (zero-based `j`) predicts `tokens[t + 1 + j]`: `d₁` predicts the token
_after_ the bonus token rather than the bonus token itself, which is why the continuation is stored
three entries longer than the number of cycles — the last cycle needs something to be compared
against. The target runs with upstream SDPA rather than the `karume_gqa` interface for this pass
only, because the cached path lets upstream drop the mask entirely.

The case material is **repository documents** (`tools/export-recipes/README.md`,
`tools/exporter/README.md`) truncated at a paragraph boundary, plus one short prompt. Each golden
stores the prompt ids it was taken with, so editing those documents does not invalidate an existing
record — it only changes what a fresh run would produce.

#### What the goldens say about acceptance

Two numbers per case, printed by the export run and stored in `reference.json` (2026-11-04, greedy
bf16 target, k = 3, N = 200 cycles): the per-step hit rate of `d_j` against `tokens[t + j]`, and
the sequentially accepted tokens per cycle, `1 + mean(a)` where `a` is the length of the leading
run of hits. The second number is what a speculation cycle actually confirms — the bonus token
always counts, so it never drops below 1.

| case              | prompt | `d₁`  | `d₂`  | `d₃`  | tokens/cycle |
| ----------------- | ------ | ----- | ----- | ----- | ------------ |
| `short-en`        | 25     | 39.0% | 22.0% | 13.5% | 1.595        |
| `readme-recipes`  | 2451   | 55.0% | 40.0% | 30.0% | 2.090        |
| `readme-exporter` | 3690   | 63.5% | 46.0% | 29.0% | 2.250        |

These are a property of the drafter against this target, not of the runtime. The acceptance test in
`packages/models/tests/e2e_gemma4_drafter_test.ts` gates something else: that karume reproduces the
**torch drafter's own draft ids** (1,800 of them) to within 1%. A perfect score there says the port
is faithful; it says nothing about how often a draft is accepted.

### The q4_0 QAT drafters (`export_drafter_qat.py`)

`export_drafter_qat.py --model e2b|e4b|12b` writes the same borrower graph from Google's
quantization-aware-trained assistants, `google/gemma-4-<model>-it-qat-q4_0-unquantized-assistant`,
pinned in `sources.py` (`QAT_DRAFTER_PINS`). Google's model card: with a QAT target, the assistant
must be a QAT checkpoint of the same precision, so each one pairs with its model's QAT target
(`QAT_TARGET_PINS`) and borrows from that target's product series — `gemma4-e2b-qat-product` /
`gemma4-e4b-qat-product`, and for 12B `gemma4-12b-product` (12B's only line is the QAT
checkpoint). What differs from the regular drafter above:

- **every drafter weight is int4 group-32, imported from the checkpoint's own grid** — the 22
  linears and the head; every group of 32 of the three pinned assistants recovers its q4_0 grid
  (2026-11-03), and a group that does not fails the export. The int8 argument above was about
  rounding weights that lie on no grid; these are the grid. The shared table is declared int4 too
  and must be the lender's int4 g32 table from the pinned QAT weights.
- **the goldens are taken against the lender series' own reference** — the values its container
  decodes to — with the target's continuation masking the ids `generation_config.json` suppresses
  (12B: two; `reference.json` records them as `suppressTokens`).
- **12B's assistant is another upstream class**, `Gemma4UnifiedAssistantForCausalLM` (the same code
  as E2B / E4B's class under another name), at 12B's widths: `hidden[1,3840]`, and the KV of the
  target's layers 46 (sliding: 8 heads in 1032 rows) and 47 (full: 1 head of width 512). It borrows
  nothing else from the target; its head is tied to its own `[262144, 1024]` table.

E[a] at k = 3 — the mean accepted prefix, `tokens/cycle − 1` — over N = 200 cycles of one
teacher-forced target trajectory per case, the checkpoint's BF16 drafter weights against the
imported int4 ones, both against the QAT target's reference (2026-11-03; the three tokenizers give
the same prompt lengths):

| case (prompt tokens)     | E2B BF16 | E2B i4 | E4B BF16 | E4B i4 | 12B BF16 | 12B i4 |
| ------------------------ | -------- | ------ | -------- | ------ | -------- | ------ |
| `short-en` (25)          | 0.55     | 0.55   | 0.54     | 0.54   | 2.545    | 2.545  |
| `readme-recipes` (2441)  | 1.22     | 1.23   | 1.325    | 1.325  | 1.495    | 1.495  |
| `readme-exporter` (3690) | 1.21     | 1.22   | 1.33     | 1.33   | 1.67     | 1.67   |

The int4 drafter emits the BF16 drafter's draft id in at least 99.2% of the 600 positions of every
case. 12B's `short-en` row says little: the target's greedy continuation of that raw prompt loops
on one sentence, which the drafter predicts easily; the two document cases continue coherently.

    outputs/series/gemma4-<model>-qat-drafter/model.krm                          the container
    outputs/series/gemma4-<model>-qat-drafter/drafter-golden.<case>.safetensors  expected drafts
    outputs/series/gemma4-<model>-qat-drafter/reference.json                     provenance

The q4_0 QAT line (`dist.py --pipeline gemma4-qat`) reads these series as its `drafter` role, which
must hold int4 and may not hold int8 there; the regular line's drafter (`gemma4-e2b-drafter`) stays
int8.

## What `tokenizer.py` emits

The **compiled tokenizer asset**, `outputs/series/gemma4-<model>-tokenizer/tokenizer.json` (~9.6 MB):
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
`packages/models/tests/fixtures/gemma-text/gemma4-chat.json` (and `gemma4-12b-chat.json`): every
expectation is taken from upstream independently (decision 7 again — the TypeScript renderer and
parser are never in the path), plus the vocabulary subset the cases need (the response texts
included) and the stop-token set declared by `generation_config.json` with its spellings. Four
lists:

- `chat` — plain conversations (roles `system` / `developer` / `user` / `assistant`, string
  content): the **rendered string** and the **token id list** of `apply_chat_template`.
- `toolChat` — the same two halves with template options: function declarations (`tools`, every
  branch of `format_parameters`), assistant `tool_calls` (one and several, nested argument
  values, strings with quotes and newlines), role `tool` results (resolved by `tool_call_id` or
  `name`; string, number, boolean and null contents), a model turn that continues after a result,
  thinking (`enable_thinking`, `reasoning` / `reasoning_content` and the thinking gate,
  `preserve_thinking`, `strip_thinking` of assistant content). The options are recorded with
  upstream's keyword names; the TypeScript side spells them `tools` / `enableThinking` /
  `preserveThinking`.
- `rejectedChat` — inputs the TypeScript side rejects although upstream renders them (or raises),
  with upstream's `rendered` string or `error` beside each: several calls without ids (every
  result gets the last call's name), a contradicting `name`, an unmatched `tool_call_id`, an
  orphan result, a call without results, an object result, JSON-string arguments, a tool name
  outside `\w+`, the `thinking` field, and the two scopes still out — the legacy
  `tool_responses` field and content-part arrays.
- `responses` — model outputs and upstream `tokenizer.parse_response(text, prefix=...)`, where
  the prefix is upstream's rendering of the case's conversation with the generation prompt. Each
  case carries an `expect` label checked against upstream at harvest time: `parse` (the result
  is recorded), `upstream-raises` (the error is recorded) and `ts-rejects` (upstream returns
  data that would hide a failure — a cut turn, a region left open, a segment overwritten; its
  Python repr is recorded and the TypeScript parser throws).

E2B and E4B share one template file (byte-identical), and on these lists E4B's tokenizer gives
E2B's ids too (`tests/test_chat.py` checks it when the hand-placed E4B checkpoint is present), so
E4B has no chat fixture of its own. 12B's template adds the empty thought channel where a
generation prompt opens a model turn with thinking off; the pytest checks that this is the only
difference on the `toolChat` list. All three snapshots declare the same `response_template`.

Out of scope (ADR [0084](../../../docs/decisions/0084-gemma-tokenizer-chat.md) decision 5 and its
addendum of 2026-11-04): `tool_responses` and content parts — the TypeScript side rejects them
loudly rather than ignoring them. The turn markers of this checkpoint are `<|turn>` / `<turn|>` /
`<|channel>` / `<|think|>` — the Gemma 3 spelling `<start_of_turn>` is not in this vocabulary at
all.

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
uv run --with 'transformers==5.14.1' python -m gemma4.export_drafter --device cpu
uv run python -m gemma4.tokenizer   # tokenizer asset + parity fixture (no transformers needed)
uv run --with 'transformers==5.14.1' python -m gemma4.chat   # chat parity fixture
```

For E4B, pass `--model e4b` to the four graph scripts; the default input directory and output
series (`outputs/series/gemma4-e4b-*`) follow the model, and `export_token.py`'s `--reference`
defaults to the same model's decode series. The drafter borrows from the product series
(`--product-dir`, default `gemma4-e2b-product-bf16`), so the product series has to exist first:

```sh
uv run --with 'transformers==5.14.1' python -m gemma4.export_product --model e4b
```

`export_drafter.py` and `chat.py` have no E4B form: E4B has no drafter here, and its chat parity
cases give E2B's ids on E4B's tokenizer too (checked when the tokenizers split). Its tokenizer
asset and parity fixture (`gemma4-e4b-parity.json`) come from:

```sh
uv run python -m gemma4.tokenizer --model e4b
```

12B (the two snapshots come from `hf download google/gemma-4-12B-it-qat-q4_0-unquantized
--revision <weights pin of sources.py>` and `hf download google/gemma-4-12B-it --revision
<tokenizer pin> tokenizer.json tokenizer_config.json chat_template.jinja generation_config.json
processor_config.json`; `export_unified.py` and `export_vision.py` take `--model-dir` /
`--tokenizer-dir` to point elsewhere):

```sh
uv run --with 'transformers==5.14.1' python -m gemma4.export_unified --model 12b --plan  # trace only
uv run --with 'transformers==5.14.1' python -m gemma4.export_unified --model 12b         # series + goldens
uv run --with 'transformers==5.14.1' python -m gemma4.export_vision --model 12b --plan   # no weight read
uv run --with 'transformers==5.14.1' python -m gemma4.export_vision --model 12b          # series + goldens
uv run python -m gemma4.tokenizer --model 12b
uv run --with 'transformers==5.14.1' python -m gemma4.chat --model 12b
```

The PLE scripts (`export.py`, `export_decode.py`, `export_token.py`, `export_product.py`) refuse
`--model 12b` at the argument.

Both fixture writers print the `deno fmt` command to run afterwards — the committed shape of a
fixture is whatever the repository formatter produces (`deno task verify` checks it).

`transformers` is pinned to 5.14.1 for the same reason as DeBERTa, EmbeddingGemma and MiniCPM5 (a
change in the modeling code changes the graph shape); it is brought in temporarily with `--with`
rather than declared as a dependency group.

## Assembling the distribution

```sh
cd tools/export-recipes
uv run python dist.py --pipeline gemma4        # → models/karume-gemma4/, E2B alone (~8.7 GiB)
# both models (~22.6 GiB: each model's bf16 container, E2B's drafter, the tokenizers)
uv run python dist.py --pipeline gemma4 --model e2b --model e4b --out ../../models/karume-gemma4
# the q4_0 QAT line (E2B / E4B / 12B with their QAT drafters) → models/karume-gemma4-qat/
uv run python dist.py --pipeline gemma4-qat --model e2b --model e4b --model 12b --out ../../models/karume-gemma4-qat
```

The regular line (`karume-gemma4`) ships E2B and E4B; it refuses `--model 12b`. 12B ships in the
q4_0 QAT line (`karume-gemma4-qat` — `distribution.py`, "The q4_0 QAT line"), its one line: its
entry reads `gemma4-12b-product`, its own `gemma4-12b-tokenizer`, its QAT drafter
`gemma4-12b-qat-drafter` and its vision embedder `gemma4-12b-vision`, declares the line's quants and
`pipelineConfig.chatDialect` `gemma4-unified`, and its container carries no PLE assets; the assembly
checks both hub snapshots against their pins, that the containers name the weights' and the
drafter's revisions, and that the tokenizer asset was compiled from the 12B-it snapshot's
`tokenizer.json`. Its `pipelineConfig` reads the numbers the model decides from the QAT checkpoint's
`config.json` and the sampler and suppressed ids from 12B-it's `generation_config.json`.

The vision embedder is a third `weights` role of 12B, `vision`, at `12b/vision/model.bf16-*.krm`:
the one bf16 exception of the line's storage gate (its weights are not on the 4-bit grid), mapped to
`bf16` by every quant, and like the drafter fetched only when a load asks for it.
`pipelineConfig.vision` holds `softTokenBudget` (`processor_config.json`), `imageTokenId` /
`beginImageTokenId` / `endImageTokenId` (`config.json`, held to the tokenizer asset through the
spellings of `tokenizer_config.json`) and `hiddenSize`. E2B and E4B have neither, and their entries
are unchanged by 12B's presence.

`--model e4b` alone would also resolve to `models/karume-gemma4/` (the family has one repository),
so an E4B-only build should name its own `--out`.

The regular distribution folds **three series** per E2B into one HF repository (two per E4B). The
product container (`gemma4-e2b-product-bf16`) holds the checkpoint's bf16 weights and the bf16 PLE
assets (index schema 5) in one `krm`, split into parts of the default 256 MiB length, with every PLE
block in a part of its own; it is the `model` role's one dtype, `bf16`, the one preset. The MTP
drafter container (`gemma4-e2b-drafter`) is the second `weights` role, fetched only when a pipeline
asks for `speculative`; the E2B preset names it. The last is the compiled tokenizer asset
(`gemma4-e2b-tokenizer`). The acceptance-only files that live beside the containers
(`ple.probe.safetensors`, `greedy.*.safetensors`, `reference.json`, `drafter-golden.*.safetensors`)
are not in the placement table and therefore never reach the output. An `e4b` entry reads
`gemma4-e4b-product-bf16` and its own `gemma4-e4b-tokenizer`, and declares no `drafter` weights: a
load that asks for `speculative` fails when the weights are resolved, before any part is fetched. Each model's tokenizer file sits under its own directory
(`<model>/tokenizer/`); the assembly folds a file into `shared/` only when two models' bytes are
the same.
Layout inside the repository:

| Manifest seat                  | Path                                      |
| ------------------------------ | ----------------------------------------- |
| `weights.model.bf16.container` | `e2b/model/model.bf16-NNNNN-of-NNNNN.krm` |
| `weights.drafter.i8.container` | `e2b/drafter/model.i8-NNNNN-of-NNNNN.krm` |
| `assets.tokenizer`             | `e2b/tokenizer/tokenizer.json`            |

The `e4b` entry has the same rows under `e4b/` but the drafter's.

Each `container` seat carries `descriptor` (the expected length and sha256 of the two descriptors)
and `parts` (every part, part 0 included — ADR
[0109](../../../docs/decisions/0109-manifest-v5-container.md) decision 3). The PLE tables have no
manifest seat of their own: they are assets inside the `model` container, reached through its
`ple_index` asset, and the host reads only the vocabulary ranges a conversation touches (ADR
[0085](../../../docs/decisions/0085-ple-host-gather.md) decision 3). **A block is looked up by the
asset name the index itself records**, so `packages/models/src/gemma/ple-index.ts` needs no second
naming; a second naming would make the correspondence positional, and a reordering of either side
would pass silently.

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

- the product container carries `BF16` (and f32 where bf16 does not apply) and no `F16`, `I8`,
  `I4` or `I2` — an int4 product series (the q4_0 QAT line's) holds the same graph contract, so
  the storage is what tells it apart; the drafter container carries `I8` only — an `I4`, `BF16` or
  `F16` tensor in it is refused (int4 group-32 linears cost 15–33 % of E[a], see above). The q4_0
  QAT line holds both its roles to `I4` alone (no `I8`)
- graph inputs are exactly `input_ids` / `rope_{sliding,full}_attention_{cos,sin}` /
  `per_layer_inputs` / `last_row`, in that order, the two exits are logits `[1, R, V]` and the
  post-norm hidden `[1, R, H]` (`H` checked against `hidden_size` — the only thing that tells the
  two apart), and of the three symbols (`M`, `R`, capacity) exactly one is free of the input
  shapes (the full slot's capacity)
- the drafter graph takes exactly `token` / `hidden` / the four RoPE inputs and emits exactly
  `k = 3` draft tokens; every one of its state slots is `external` and matches a lender slot by
  name, dtype and shape; exactly one initializer is `shared` and names a tensor key the product
  container really carries; its single symbol is the lender's capacity symbol, same spelling
- `per_layer_inputs`' layer and dim axes match the PLE index
- the PLE index is a gap-free ascending partition of `[0, tokens)`, `tokens` equals `V`, every
  block the index names is an asset of the `model` container with the same role and length, and no
  PLE asset is left that the index does not name
- the compiled tokenizer names the compile format and has exactly `V` rows, and the asset's
  recorded source sha256 matches the model's own upstream `tokenizer.json` (each model has its own
  tokenizer series since Google's glitch-token fixes of 2026-11-03)
- the checkpoint's recommended sampler is present and inside the range the TypeScript
  `pipelineConfig` parser accepts

Apache 2.0 §4 applies to the redistribution, so the repository root also carries `LICENSE.md` (a
verbatim copy of `../_shared/licenses/apache_license_2_0.txt`) and `NOTICE.md` (the list of modifications). Both are
`karume.dist`'s legal-text seat, not manifest-declared assets.

Loading it back is `gemma.Pipeline.fromPretrained` (`packages/models/src/gemma/pipeline.ts`). The
pinned fetch source for the published repository is `gemma.SOURCES["gemma4"]`
(`@karume/models/gemma` — ADR [0073](../../../docs/decisions/0073-models-source-pin.md) /
[0092](../../../docs/decisions/0092-distribution-repos-and-sources.md)); a locally built
distribution is spelled out by the caller, since `fromPretrained` has no default source.

## Quant execution variants

Both models carry one quant, `bf16`, their default (the maintainer's ruling of 2026-11-04): the
`model` weights' one container — every weight and the PLE table as the checkpoint's bf16 bits, PLE
index schema 5 — in the reference summation order (`session` empty: GEMV and attention run the
runtime's reference path). E2B's `bf16` also names the int8 drafter, which borrows the bf16
target's `model.lm_head.weight`. The parallel and fused int4 seats (`i4-gemvpar` / `i4-fast`)
are the q4_0 QAT line's (`karume-gemma4-qat` — `distribution.py`, "The q4_0 QAT line").

Explicit runtime options override each quant setting. Reassembling into a new output directory
updates the metadata without requantizing weights. Existing distributions and pinned public
revisions remain unchanged. Submission policy and prefill buckets are separate host options.
See [the decision record](../../../docs/decisions/0104-gemma-fast-quant.md).
