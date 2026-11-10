"""The Gemma 4 distribution recipe — series layout, output path table, quant table and the card
choice (ADR 0065 decision 2).

The general-purpose assembly engine (placement, folding shared files, sha256, manifest,
staging/swap, verification) lives in `karume.dist`. What lives here is **the gemma4-specific facts**
only: which series directory each file is picked from, at which distribution path and under which
dtype label it is listed, and which quant is the default.

Per model the regular line ships the product graph, the borrowing drafter for a model with an
upstream drafter, and one asset kind (ADR 0084 decision 5 / 0096 step 2 / 0109 decision 4): E2B
ships two graphs, E4B one, as it has no drafter ({@link gemma4_has_drafter}):

- `model` — the product graph's container (the `.krm` part sequence `gemma4/export_product.py`
  writes: the PLE out of the graph, the exit the selected rows' logits and hidden states). Every
  weight is the checkpoint's own bf16 bits, its one preset `bf16` (ADR 0128 addendum 2026-11-04).
  **The PLE is an asset of this container** (ADR 0109 decision 4), not weights — the index
  `ple_index` (schema 5: a bf16 `values` table and no `scales`) and the `ple.values.<k>` block
  sequence, the table the host builds `per_layer_inputs` from. The reader is
  `packages/models/src/gemma/ple-index.ts`.
- `drafter` — the MTP drafter's container (`gemma4/export_drafter.py` — ADR 0096 step 2), int8
  throughout, linears included (int4 group 32 cuts the acceptance rate by 15–33 % — the script's
  measurement). It borrows `model.lm_head.weight` from the bf16 target. Each role has one dtype
  label, so {@link complete_quant_weights} fills both seats of the quant's `weights` mapping.
- `tokenizer` — the compiled tokenizer asset (`gemma4/tokenizer.py` — ADR 0084 decision 1). Each
  model reads its own series today; models whose upstream `tokenizer.json` files are the same bytes
  may share one ({@link GEMMA4_TOKENIZER_SERIES}), and every model's asset is held to the upstream
  file it was compiled from ({@link assert_gemma4_tokenizer_source})

The regular line ships E2B and E4B ({@link GEMMA4_REGULAR_MODELS}). 12B ships only in the q4_0
QAT line below — its one line, whose weights are already Google's QAT checkpoint. The plan handles
what sets 12B apart for that line: its `model` container has no PLE assets and its graph no
`per_layer_inputs` (the model has no per-layer embeddings — {@link gemma4_ple_dim}); it declares
its chat dialect (`pipelineConfig.chatDialect` `gemma4-unified`) and the ids generation never
selects (`pipelineConfig.suppressTokens`, from upstream's `suppress_tokens` —
{@link gemma4_generation_defaults}). Its upstream is two pinned hub snapshots of `gemma4.sources`:
the weights and their `config.json` come from Google's QAT checkpoint, the tokenizer and the
generation defaults from `google/gemma-4-12B-it` ({@link Gemma4Sources}).

## The q4_0 QAT line (`karume-gemma4-qat` — {@link QAT_Q4_0_PIPELINE})

A second distribution of the same `gemma4/1` contract: E2B / E4B / 12B / 26B from Google's q4_0 QAT
checkpoints, every text weight on its own 4-bit grid (int4 g32), each model but 26B with its QAT MTP
drafter. It runs {@link gemma4_plan} with another {@link Gemma4Line} — the graph, drafter,
tokenizer and configuration checks are the regular line's; what differs is read from the line:

- the series: E2B / E4B's `gemma4-<model>-qat-product` (`gemma4/export_product_qat.py`), 12B's
  `gemma4-12b-product` (its one line, already the QAT checkpoint), every drafter
  `gemma4-<model>-qat-drafter` (`gemma4/export_drafter_qat.py`) — {@link gemma4_qat_q4_0_sources};
- the storage: both roles int4 alone, int8 forbidden in both (the drafters are on their grid too,
  so the regular drafter's int8 requirement does not apply — {@link GEMMA4_QAT_Q4_0_STORAGE}); 12B's
  `vision` role is the one bf16 exception (its embedder is not on the grid);
- the PLE index: schema 4, int4 values with one f32 step per group of
  {@link GEMMA4_QAT_PLE_GROUP_SIZE} ({@link gemma4_ple_index});
- the quants: `i4` (the default — the reference summation order) and the parallel and fused seats
  `i4-gemvpar` / `i4-fast` over the same storage ({@link gemma4_qat_q4_0_quants});
- the pins: every product container names its QAT weights' revision and every drafter container
  its QAT drafter's (`gemma4.sources.QAT_LINE_PINS` / `QAT_DRAFTER_PINS`);
- 26B A4B: the `model` role alone (no drafter until one is pinned, so `speculative` is refused at
  weights resolution; no vision), the reference seat and the parallel and fused seats (its routed
  experts keep the sequential order in every seat), 12B's graph contract and dialect, its
  expert weights the one `i4f16` storage of the line (ADR 0145), and its pins in
  `gemma4.sources.QAT_LINE_PINS` (it is no drafter's target);
- 12B's image input: a third graph, `vision` — the encoder-free embedder
  (`gemma4/export_vision.py`, series `gemma4-12b-vision`), one bf16 dtype, so every quant of 12B
  maps it — and `pipelineConfig.vision` ({@link gemma4_vision_config}). As with the drafter, it is a
  weights role precisely so that a reader's selection can leave it out: nothing is fetched unless a
  load asks for image input. E2B / E4B have neither (their tower is not distributed). The series
  must be a full one, its goldens taken at the declared budget
  ({@link assert_gemma4_vision_series}), and `processor_config.json` is pinned with the tokenizer
  (`gemma4.sources.HubPin.also_checked`).

`pipelineConfig` splits into two kinds (the same split as Irodori): **the numbers the model
decides** (`maxPosition` = upstream `text_config.max_position_embeddings`, `rope` = the parameters
of each layer type's formula) are derived from upstream `config.json`, while the **runtime knobs**
(`chunkLength` / `capacity`) and the **distributor's recommended sampler** (temperature / top_k /
top_p of upstream `generation_config.json` — ADR 0083 decision 7) are read from their own sources
of truth. Copying the former by hand means that on the day the checkpoint is replaced only the
declaration goes stale, and "a table built as declared rotates at angles other than upstream's".

Of the runtime knobs, only `chunkLength` **also declares its upper bound** (`maxChunkLength`) — the
IR's `symbols` is only a list of names and carries no bounds, so a reader cannot derive "how many
chunk rows this asset accepts" from the asset. The one number the writer (`gemma4.export.SYM_MAX`)
knows is put into the declaration, so that a check on the TS side can reject a `chunkLength`
outside the traced range (ruling of 2026-09-03).

The RoPE cos / sin tables **are no longer part of the distribution** — the host (TS side) builds
them at run time from the `rope` declaration. So the position limit is set by the model's
declaration alone, not by an asset, and `capacity` becomes a runtime knob bounded by `maxPosition`.

The public surface is the one {@link PIPELINE} (`karume.dist.Pipeline`) — the repository's dist
driver (`tools/export-recipes/dist.py`) merges it into the core's PIPELINES.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _shared.container_read import read_asset, read_asset_declarations
from _shared.licenses import apache_license_2_0
from _shared.upstream import assert_upstream_provenance
from karume.container import ContainerFormatError
from karume.dist import (
    Artifact,
    DistError,
    ModelPlan,
    Pipeline,
    WeightFiles,
    assert_model_name,
    assert_storage,
    assert_storage_absent,
    complete_quant_weights,
    graph_inputs,
    ir_graph,
    sha256_file,
)
from karume.ple import (
    PLE_FLOAT_BYTES,
    PLE_FLOAT_INDEX_SCHEMA,
    PLE_GROUPED_INDEX_SCHEMA,
    PLE_INDEX_ASSET,
    PLE_INDEX_ROLE,
    PLE_INDEX_SCHEMA,
    PLE_PACK_FACTOR,
    PLE_ROLES,
    PLE_SCALE_BYTES,
)
from karume.verify import ContainerError

from .card import (
    GEMMA4_DRAFTER_ROLE,
    GEMMA4_DRAFTER_UPSTREAM,
    GEMMA4_LICENSE,
    GEMMA4_MODEL_ROLE,
    GEMMA4_MOE_MODELS,
    GEMMA4_PLE_MODELS,
    GEMMA4_QAT_DRAFTER_UPSTREAM,
    GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM,
    GEMMA4_UPSTREAM,
    GEMMA4_VISION_ROLE,
    render_gemma4_model_card,
    render_gemma4_qat_q4_0_model_card,
)
from .chat import CHAT_DIALECTS, GEMMA4_CHAT_DIALECT, GENERATION_PROMPT_SUFFIX
from .rope import (
    BAKED_TABLE_INFIX,
    FULL_ATTENTION,
    HEAD_DIM_FIELD,
    SLIDING_ATTENTION,
    RopeSpecError,
    rope_specs,
)
from .sources import (
    QAT_DRAFTER_PINS,
    QAT_LINE_PINS,
    QAT_PINS,
    HubPin,
    ModelPins,
    assert_model_pins,
    checkpoint_dir,
    qat_line_dirs,
    tokenizer_dir,
)

#: The pipeline contract (ADR 0041 §2 — per model). The set the TS side accepts is
#: `GEMMA4_PIPELINE_NAME` / `GEMMA4_PIPELINE_MAJOR` (`packages/models/src/gemma/config.ts`).
GEMMA4_PIPELINE = "gemma4/1"

#: The default model name (the model an assembly without `--model` includes). The set of accepted
#: spellings is held by the attribution table (`gemma4.card.GEMMA4_UPSTREAM`).
GEMMA4_DEFAULT_MODEL = "e2b"

#: The models the regular line (`karume-gemma4`) ships. 12B is not one of them: its only line is
#: the q4_0 QAT one (`karume-gemma4-qat` — {@link GEMMA4_QAT_Q4_0_LINE}), and the regular line
#: refuses it ({@link gemma4_sources} / {@link gemma4_plan}).
GEMMA4_REGULAR_MODELS: tuple[str, ...] = ("e2b", "e4b")

#: The series name prefix (`gemma4-<model name>-<suffix>`).
GEMMA4_PREFIX = "gemma4"

#: The distribution repository name (`karume-<family>` of ADR 0092 decision 1). **Not derived from
#: the model name** — the regular line is one repository, whichever of its models is assembled.
GEMMA4_REPO_NAME = "karume-gemma4"

#: Series suffixes — the writers (`gemma4/export_unified.py`, `gemma4/tokenizer.py`'s `ASSET_PATH`)
#: and this module build the series names from the same words. `product` is 12B's product series.
GEMMA4_PRODUCT_SUFFIX = "product"
#: The regular line's product series suffix (`outputs/series/gemma4-<model>-product-bf16/` —
#: `gemma4/export_product.py` writes under it, this module reads it).
GEMMA4_PRODUCT_BF16_SUFFIX = f"{GEMMA4_PRODUCT_SUFFIX}-bf16"
GEMMA4_TOKENIZER_SUFFIX = "tokenizer"
#: The MTP drafter's series suffix (`DEFAULT_OUT_DIR` of `gemma4/export_drafter.py`).
GEMMA4_DRAFTER_SUFFIX = "drafter"
#: The q4_0 QAT line's series suffixes (`outputs/series/gemma4-<model>-qat-product/` and
#: `…-qat-drafter/` — `gemma4/export_product_qat.py` / `gemma4/export_drafter_qat.py` write under
#: these words, this module reads them).
GEMMA4_QAT_PRODUCT_SUFFIX = "qat-product"
GEMMA4_QAT_DRAFTER_SUFFIX = "qat-drafter"
#: The vision embedder's series suffix (`outputs/series/gemma4-12b-vision/` —
#: `gemma4/export_vision.py` writes under it, this module reads it).
GEMMA4_VISION_SUFFIX = "vision"

#: The file names on the series side (the spellings of `gemma4.export.MODEL_FILE` /
#: `tokenizer.ASSET_PATH`). They are **representative paths**, so a split file is resolved to its
#: numbered parts by {@link karume.dist.component_parts}.
GEMMA4_MODEL_FILE = "model.krm"
GEMMA4_TOKENIZER_FILE = "tokenizer.json"

#: The source of the recommended sampler the upstream checkpoint carries (ADR 0083 decision 7) and
#: of the numbers the model decides (the position limit, the RoPE formula). The text part is the
#: `text_config` section of `config.json`.
GEMMA4_GENERATION_CONFIG_FILE = "generation_config.json"
GEMMA4_CONFIG_FILE = "config.json"
#: The upstream checkpoint's tokenizer (the HF form, before compiling): what every model's compiled
#: asset is held against ({@link assert_gemma4_tokenizer_source}).
GEMMA4_CHECKPOINT_TOKENIZER_FILE = "tokenizer.json"
GEMMA4_TEXT_CONFIG_KEY = "text_config"
GEMMA4_MAX_POSITION_KEY = "max_position_embeddings"
GEMMA4_HIDDEN_SIZE_KEY = "hidden_size"

#: Role names (the internal keys the manifest's weights / assets point to).
GEMMA4_ROLE = GEMMA4_MODEL_ROLE
GEMMA4_TOKENIZER_ROLE = "tokenizer"
# The MTP drafter's role name `GEMMA4_DRAFTER_ROLE` (the second weights entry — it cannot run alone,
# without the lender `model`) is imported from `gemma4.card` — the card reads whether a drafter
# exists from the manifest's weights, so the canonical spelling lives on the card side (the card
# importing distribution would be circular). The models that ship a drafter are only those listed
# in `GEMMA4_DRAFTER_UPSTREAM`.

#: Model → the **model name** of the series that holds its compiled tokenizer asset
#: (`gemma4-<this>-tokenizer`).
#:
#: A model may borrow another model's series only while their upstream `tokenizer.json` files are
#: the same bytes (a compiled asset records its source path, so compiling it again for the second
#: model splits the bytes and the assembly can no longer fold it into `shared/`). MUST: the
#: premise is checked by machine at assembly — {@link assert_gemma4_tokenizer_source} holds the
#: asset's recorded source SHA-256 against the model's upstream `tokenizer.json`. The same check
#: holds a model's own series to its current upstream file (a series compiled before that file was
#: replaced is stale).
#: Every model reads its own series: E4B shared E2B's until Google's glitch-token fixes of
#: 2026-11-03 gave E2B's and E4B's `tokenizer.json` different merges (E2B drops 267, E4B 253), and
#: 12B's (fixed on 2026-10-30) drops 239 — the same vocabulary, but another model's asset would
#: give other ids for some inputs. 26B's (`google/gemma-4-26B-A4B-it` at its pin) is still the file
#: from before the fixes (SHA-256 `cc8d3a0c…`), so it is no other model's bytes either.
GEMMA4_TOKENIZER_SERIES: Mapping[str, str] = {
    "e2b": "e2b",
    "e4b": "e4b",
    "12b": "12b",
    "26b": "26b",
}

#: The names and order of the graph inputs (canonical in `gemma4/export_product.py` — the wrapper's
#: forward argument names). The runtime binds by name, so if a single spelling changes the inputs
#: cannot be bound. The four RoPE inputs are the cos / sin the host builds at run time (canonical
#: spelling in `gemma4.export_decode.rope_input_name`).
GEMMA4_INPUT_IDS = "input_ids"
GEMMA4_PER_LAYER_INPUTS = "per_layer_inputs"
GEMMA4_LAST_ROW = "last_row"
GEMMA4_ROPE_INPUT_PREFIX = "rope_"
GEMMA4_ROPE_PARTS: tuple[str, ...] = ("cos", "sin")
GEMMA4_ROPE_LAYER_TYPES: tuple[str, ...] = (SLIDING_ATTENTION, FULL_ATTENTION)


def gemma4_rope_input_name(layer_type: str, part: str) -> str:
    """The spelling of one derived RoPE input (the mirror of the writer's
    `export_decode.rope_input_name`)."""
    return f"{GEMMA4_ROPE_INPUT_PREFIX}{layer_type}_{part}"


GEMMA4_ROPE_INPUTS: tuple[str, ...] = tuple(
    gemma4_rope_input_name(layer_type, part)
    for layer_type in GEMMA4_ROPE_LAYER_TYPES
    for part in GEMMA4_ROPE_PARTS
)
GEMMA4_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_INPUT_IDS,
    *GEMMA4_ROPE_INPUTS,
    GEMMA4_PER_LAYER_INPUTS,
    GEMMA4_LAST_ROW,
)
#: An image's rows on a model without per-layer embeddings (the mirrors of
#: `gemma4.export_product.SOFT_EMBEDS` / `SOFT_MASK`, held equal by the tests).
GEMMA4_SOFT_EMBEDS = "soft_embeds"
GEMMA4_SOFT_MASK = "soft_mask"
#: The inputs of a model without per-layer embeddings (12B — `gemma4/export_unified.py`): the
#: same graph contract minus `per_layer_inputs`, plus the image rows right after the ids.
GEMMA4_PLE_FREE_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_INPUT_IDS,
    GEMMA4_SOFT_EMBEDS,
    GEMMA4_SOFT_MASK,
    *GEMMA4_ROPE_INPUTS,
    GEMMA4_LAST_ROW,
)

#: The vision embedder graph's inputs and its row symbol (canonical in `gemma4/export_vision.py` —
#: the wrapper's forward argument names). The host binds them by name.
GEMMA4_VISION_PIXELS = "pixels"
GEMMA4_POSITION_X = "position_x"
GEMMA4_POSITION_Y = "position_y"
GEMMA4_VISION_SYMBOL = "N"
#: The embedder's one storage (the dtype label, also the output file's and the writer's word).
GEMMA4_VISION_STORAGE = "bf16"

#: Upstream's processor configuration, read from the tokenizer source (the budget of
#: `pipelineConfig.vision`, the row width of the embedder graph).
GEMMA4_PROCESSOR_CONFIG_FILE = "processor_config.json"

#: The vision series' provenance record (the mirror of `gemma4.provenance.REFERENCE_FILE`, held
#: equal by the tests — that module imports torch): it names the series' greedy goldens and the
#: budget and processor configuration they were taken at ({@link assert_gemma4_vision_series}).
GEMMA4_REFERENCE_FILE = "reference.json"

#: The drafter graph's inputs (canonical in `gemma4/export_drafter.py` — the wrapper's forward
#: argument names). `token` = the token decided just before, `hidden` = the target's hidden after
#: the final norm (output 1 of the product graph). K / V are not listed: surgery replaced their
#: trace-time inputs with external slots.
GEMMA4_DRAFTER_TOKEN = "token"
GEMMA4_DRAFTER_HIDDEN = "hidden"
GEMMA4_DRAFTER_GRAPH_INPUTS: tuple[str, ...] = (
    GEMMA4_DRAFTER_TOKEN,
    GEMMA4_DRAFTER_HIDDEN,
    *GEMMA4_ROPE_INPUTS,
)

#: The number of drafts the drafter emits per run (the mirror of
#: `gemma4.export_drafter.DRAFT_STEPS` — the distributed drafter is baked with these 3 steps; the
#: runtime k can be chosen within `k <= 3`, and more steps need a re-export). The number of outputs
#: MUST equal it.
GEMMA4_DRAFT_STEPS = 3

#: The relative output paths (**inside the model subtree**) — the one place the placement table and
#: the manifest share. The storage dtype appears in the file name, as in the other families (a
#: spelling that cannot be mixed up even with two series side by side).
#: The PLE is an asset of the `model` container, so it has no entry in this table (ADR 0109
#: decision 4).
GEMMA4_OUTPUT_PATHS: Mapping[str, str] = {
    GEMMA4_ROLE: f"{GEMMA4_ROLE}/model.bf16.krm",
    GEMMA4_DRAFTER_ROLE: f"{GEMMA4_DRAFTER_ROLE}/model.i8.krm",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
}

#: The storage each role's binding table **must** hold ({@link assert_storage}): the product
#: container is the checkpoint's bf16 values (and f32 for the parameters bf16 does not apply to),
#: the drafter int8 throughout (linears included — `gemma4/export_drafter.py`'s measurement).
GEMMA4_STORAGE_REQUIREMENTS: Mapping[str, str] = {
    GEMMA4_ROLE: "bf16",
    GEMMA4_DRAFTER_ROLE: "i8",
}

#: The storage vocabulary each role's binding table **must not** hold
#: ({@link assert_storage_absent}). The scripts write one storage form per role: the product series
#: (bf16 + f32 — `model`) and the int8 drafter (i8 + f32). So f16 in any role, or a storage of the
#: other role's form, can only mean a series root of another family, of another line or of the
#: other role (a mixed-up series root cannot be told apart by the numeric gates — ADR 0027 / 0029).
#: As in the other families, every compressed storage the writer can write that must not be there
#: is named: the product forbids every rounded storage — an int4 product series (the q4_0 QAT
#: line's) has the same graph contract, so the storage is what tells it apart first.
#: The drafter forbids i4 too — a presence check (i8 is there) lets "a drafter whose linears alone
#: dropped to i4" through (the output head's i8 meets the requirement). An asset that only loses
#: 10–30 % of its acceptance rate keeps a correct shape and manifest, so this is the only detector.
GEMMA4_STORAGE_FORBIDDEN: Mapping[str, tuple[str, ...]] = {
    GEMMA4_ROLE: ("f16", "i8", "i4", "i4f16", "i2"),
    GEMMA4_DRAFTER_ROLE: ("f16", "bf16", "i4", "i4f16"),
}


@dataclass(frozen=True)
class Gemma4StorageTables:
    """A line's storage gates: what each role must hold and what it must not."""

    required: Mapping[str, str]
    forbidden: Mapping[str, tuple[str, ...]]


#: The regular line's storage gates (the same for both of its models).
GEMMA4_STORAGE = Gemma4StorageTables(
    required=GEMMA4_STORAGE_REQUIREMENTS, forbidden=GEMMA4_STORAGE_FORBIDDEN
)


#: The storage dtype labels (also the spelling of the quant seat). Each role has one storage, so
#: {@link complete_quant_weights} fills both seats of the preset (`{model: "bf16", drafter: "i8"}`).
GEMMA4_DTYPE = "bf16"
GEMMA4_DRAFTER_DTYPE = "i8"

#: The weights declaration (dtype label → role name). The split is decided by the actual files, so
#: this points to one representative role.
#: MUST: declare the drafter as **the second weights entry** (not as an asset) — it is an IR
#: container that holds a graph and is consumed by `createSession`, and this is the only axis on
#: which hub's `ResolveOptions.weights` can express "not fetched by default".
GEMMA4_WEIGHTS: Mapping[str, Mapping[str, WeightFiles]] = {
    GEMMA4_ROLE: {GEMMA4_DTYPE: WeightFiles(GEMMA4_ROLE)},
    GEMMA4_DRAFTER_ROLE: {GEMMA4_DRAFTER_DTYPE: WeightFiles(GEMMA4_DRAFTER_ROLE)},
}


def gemma4_has_drafter(model: str) -> bool:
    """Whether this model ships an MTP drafter (whether the drafter upstream table
    `GEMMA4_DRAFTER_UPSTREAM` lists it)."""
    return model in GEMMA4_DRAFTER_UPSTREAM


def gemma4_weights(model: str) -> Mapping[str, Mapping[str, WeightFiles]]:
    """The model's weights declaration — a model that ships no drafter has the `model` role only.

    MUST: do not declare a drafter role for a model without a drafter. The absence of that role in
    the manifest's weights is the only way to say "this model cannot run speculative decoding", and
    a reader fails a load that asks for `speculative` loudly at weights resolution (hub
    `ResolveOptions.weights`).
    """
    if gemma4_has_drafter(model):
        return GEMMA4_WEIGHTS
    return {GEMMA4_ROLE: GEMMA4_WEIGHTS[GEMMA4_ROLE]}


#: The sentence of the preset's description on the drafter (only for a model that ships one — a
#: drafter not distributed does not have its storage stated).
GEMMA4_DRAFTER_QUANT_NOTE = " The drafter head is int8 throughout."

#: The one preset, `bf16`: the original weights (ADR 0128 decision 4; the default by ADR 0128
#: addendum 2026-11-04) — the checkpoint's own bf16 values with f32 compute, in the reference
#: summation order (`session` empty — ADR 0104). {@link gemma4_quants} adds the drafter sentence
#: per model.
GEMMA4_QUANTS: Mapping[str, Any] = {
    GEMMA4_DTYPE: {
        "weights": {},
        "session": {},
        "label": "Original weights (bf16)",
        "description": "The checkpoint's own bf16 weights, the per-layer embeddings included, with"
        " f32 compute; GEMV and attention use the reference summation order.",
    },
}


def gemma4_quants(model: str) -> dict[str, Any]:
    """The model's quant table ({@link GEMMA4_QUANTS}), the drafter sentence added to the
    description for a model that ships a drafter ({@link GEMMA4_DRAFTER_QUANT_NOTE})."""
    quants = {name: dict(quant) for name, quant in GEMMA4_QUANTS.items()}
    if gemma4_has_drafter(model):
        quants[GEMMA4_DTYPE]["description"] += GEMMA4_DRAFTER_QUANT_NOTE
    return quants


#: The default quant per model. The key set MUST equal the regular line's models
#: ({@link GEMMA4_REGULAR_MODELS}; the tests pin it). Both default to the line's one preset, `bf16`
#: (the maintainer's ruling of 2026-11-04 — ADR 0128 addendum 2026-11-04).
GEMMA4_DEFAULT_QUANT: Mapping[str, str] = dict.fromkeys(GEMMA4_REGULAR_MODELS, GEMMA4_DTYPE)

#: The row count of a fixed-length prefill chunk (ADR 0066 decision 4 — a context planning
#: constant). It is a **runtime knob**, so it cannot be derived from the asset. Its upper bound is
#: the trace-time bound of the symbol `M` ({@link GEMMA4_MAX_CHUNK_LENGTH}).
GEMMA4_CHUNK_LENGTH = 768

#: The upper bound of the symbol `M` (the rows of one chunk). The mirror of the writer's
#: `gemma4.export.SYM_MAX`, copied to the side that does not read torch (the two are held equal by
#: `tests/test_distribution.py`).
#:
#: It is **carried in the distribution** as `pipelineConfig.maxChunkLength`
#: ({@link gemma4_pipeline_config}). It cannot be derived from the asset (the IR's `symbols` is only
#: a list of names), so without the declaration a reader cannot tell whether a `chunkLength`
#: override is inside the traced range.
GEMMA4_MAX_CHUNK_LENGTH = 768

#: The exit's row symbol (`last_row[R]` and the row axis of both outputs). The mirror of the
#: writer's `gemma4.export_product.ROW_SYMBOL`, copied to the side that does not read torch (the two
#: are held equal by `tests/test_distribution.py`). R is bound by an input shape, so unlike the
#: capacity symbol it is not carried in the distribution's declaration.
GEMMA4_ROW_SYMBOL = "R"

#: The **default** capacity of the full slots (the largest logical length a conversation can use).
#: Also a runtime knob; {@link gemma4_pipeline_config} bounds it by the model's declaration
#: (`maxPosition`).
#:
#: NOTE: **a policy value trading VRAM against conversation length**. With RoPE moved to the host
#: the asset-side position limit is gone, and up to `maxPosition` (131,072 for E2B) can be declared
#: — this only sets "how much is reserved by default", and the price is that the resident bytes of
#: the full slots grow with the capacity (`karume.limits`).
GEMMA4_CAPACITY = 4096

#: Upstream `generation_config.json` → the field names of `pipelineConfig.sampler` (the spellings
#: of the TS `SamplerSpec`).
GEMMA4_SAMPLER_FIELDS: tuple[tuple[str, str], ...] = (
    ("temperature", "temperature"),
    ("top_k", "topK"),
    ("top_p", "topP"),
)

#: Upstream `generation_config.json`'s list of ids generation never selects → the optional
#: top-level `pipelineConfig.suppressTokens` ({@link gemma4_generation_defaults}). 12B declares
#: `[258883, 258882]` (end-of-audio / end-of-image); upstream `generate()` sets their logits to
#: −inf before greedy and sampled selection alike (`SuppressTokensLogitsProcessor`).
GEMMA4_SUPPRESS_TOKENS_KEY = "suppress_tokens"

#: `generation_config.json` keys the distribution does not carry, each with its reason. MUST: a
#: key that is in neither this table, {@link GEMMA4_SAMPLER_FIELDS} nor
#: {@link GEMMA4_SUPPRESS_TOKENS_KEY} fails loudly — a generation default upstream adds (as 12B
#: added `suppress_tokens`) must surface as a decision, not be dropped silently.
GEMMA4_GENERATION_NOT_CARRIED: Mapping[str, str] = {
    "bos_token_id": "the compiled tokenizer asset declares `<bos>` (`bosId`); the host prepends it",
    "eos_token_id": (
        "the stop set is derived from the tokenizer asset's added vocabulary; the chat fixture"
        " (`gemma4.chat.stop_tokens`) holds the derivation to this declaration"
    ),
    "pad_token_id": "nothing is padded (one unpadded sequence per run)",
    "do_sample": (
        "greedy or sampled decoding is the caller's choice per request; sampling uses the"
        " declared sampler"
    ),
    "transformers_version": "the version of the library that wrote the file, not a default",
}

#: The format identifier of the compiled tokenizer asset (the writer is
#: `_shared/gemma_tokenizer.py`).
GEMMA4_TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"

#: The PLE index's version and fields (the mirror of `SCHEMA` / `INDEX_KEYS` / `TABLE_KEYS` /
#: `BLOCK_KEYS` of the reader `packages/models/src/gemma/ple-index.ts` — the writer side fails
#: first). The versions themselves (`PLE_INDEX_SCHEMA` / `PLE_GROUPED_INDEX_SCHEMA`) are shared
#: with the writer (`karume.ple`).
GEMMA4_PLE_VALUES_KEY = "values"
GEMMA4_PLE_SCALES_KEY = "scales"
GEMMA4_PLE_INDEX_KEYS: tuple[str, ...] = (
    "schema",
    "storage",
    "tokens",
    "layers",
    "dim",
    "embedScale",
    GEMMA4_PLE_VALUES_KEY,
    GEMMA4_PLE_SCALES_KEY,
)
GEMMA4_PLE_TABLE_KEYS: tuple[str, ...] = ("rowBytes", "blocks")
GEMMA4_PLE_BLOCK_KEYS: tuple[str, ...] = ("asset", "start", "stop")
#: Schema 4's one more key (`karume.ple.PLE_GROUPED_INDEX_SCHEMA` — the reader's `GROUPED_SCHEMA`):
#: the number of consecutive values of a (token, layer) row that share one f32 scale.
GEMMA4_PLE_GROUP_SIZE_KEY = "groupSize"


def gemma4_checkpoint(model: str) -> str:
    """Model name → the directory name of the upstream checkpoint (= the tail of the upstream HF
    repository name).

    The spelling has one fact behind it, "which upstream repository this model is", so it is
    derived from the attribution table (`gemma4.card.GEMMA4_UPSTREAM`) instead of keeping a second
    table here (the same rule as SigLIP2).
    """
    repo = GEMMA4_UPSTREAM.get(model)
    if repo is None:
        raise DistError(
            f"unknown gemma4 model '{model}' (known: {' / '.join(sorted(GEMMA4_UPSTREAM))})"
        )
    return repo.split("/", 1)[1]


def gemma4_repo_name(_model: str) -> str:
    """The distribution repository name (one repository per family, so the same one for every
    model — the `karume-` prefix is the repository-name ruling of 2026-08-09).

    `Pipeline.repo_name` is the canonical repository name shown in the card's usage example
    (`karume.dist.resolve_repo`), and it also answers the default output location when a single
    model is assembled. Now that the family is folded into one repository, it ignores the model
    name.
    """
    return GEMMA4_REPO_NAME


def gemma4_series_name(model: str, suffix: str) -> str:
    """The series name (`outputs/series/gemma4-<model name>-<suffix>/`) — the same spelling as the
    writers' defaults."""
    return f"{GEMMA4_PREFIX}-{model}-{suffix}"


@dataclass(frozen=True)
class Gemma4Sources:
    """The inputs of one model's assembly: its series (the product graph with the PLE as container
    assets, the drafter, the tokenizer asset) and the upstream checkpoint, of which only
    `config.json`, `generation_config.json` (the recommended sampler) and `tokenizer.json` (what the
    tokenizer asset is held against — {@link assert_gemma4_tokenizer_source}) are read, never the
    weights.

    A model whose `drafter` is `None` ships no drafter ({@link gemma4_has_drafter}); a model whose
    `vision` is `None` ships no vision embedder (`Gemma4Line.has_vision`).

    The upstream side is two directories: `model` holds the weights' `config.json` (the numbers the
    model decides — {@link gemma4_text_config}) and `upstream_tokenizer` the `tokenizer.json` and
    `generation_config.json`. They are one checkpoint in the regular line. In the q4_0 QAT line the
    weights come from Google's QAT checkpoints, whose `tokenizer.json` is the one every Gemma 4
    repository shipped before Google's glitch-token fixes (the QAT repositories have not received
    them), so the tokenizer and the generation defaults are read from the regular `-it` repository
    (`gemma4.sources.QAT_LINE_PINS`).
    """

    product: Path
    drafter: Path | None
    tokenizer: Path
    model: Path
    upstream_tokenizer: Path
    vision: Path | None = None


def gemma4_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """Look up the inputs from the series' parent directory (`outputs/series/`) and the spellings
    of `_shared.paths`.

    The upstream directories come from `gemma4.sources.checkpoint_dir` / `tokenizer_dir`
    (`inputs/gemma4/…`). A model outside the regular line ({@link GEMMA4_REGULAR_MODELS} — 12B
    ships in the q4_0 QAT line, {@link gemma4_qat_q4_0_sources}) fails loudly.
    """
    gemma4_checkpoint(model)
    if model not in GEMMA4_REGULAR_MODELS:
        raise DistError(
            f"the regular gemma4 line has no model '{model}'"
            f" (known: {' / '.join(GEMMA4_REGULAR_MODELS)}; 12B ships in --pipeline gemma4-qat)"
        )
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, GEMMA4_PRODUCT_BF16_SUFFIX),
        drafter=(
            series_dir / gemma4_series_name(model, GEMMA4_DRAFTER_SUFFIX)
            if gemma4_has_drafter(model)
            else None
        ),
        tokenizer=series_dir
        / gemma4_series_name(GEMMA4_TOKENIZER_SERIES[model], GEMMA4_TOKENIZER_SUFFIX),
        model=checkpoint_dir(model),
        upstream_tokenizer=tokenizer_dir(model),
    )


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise DistError(f"{what} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DistError(f"{path} cannot be read as JSON") from error


def _positive_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    # bool is a subclass of int; letting `"tokens": true` through as 1 would loosen the row-count
    # match.
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DistError(f"{where}: {key} is not a positive integer ({value!r})")
    return value


def _offset(raw: Mapping[str, Any], key: str, where: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DistError(f"{where}: {key} is not a non-negative integer ({value!r})")
    return value


def gemma4_ple_table(
    root: Mapping[str, Any], name: str, row_bytes: int, tokens: int, where: str
) -> dict[str, Any]:
    """Check and read one index table (`values` / `scales`).

    MUST: the block ranges are **an ascending partition of `[0, tokens)` with no gap and no
    overlap**. Loosening this lets through "an index with ids that cannot be looked up" and "an
    index in which two blocks hold the same id", and with the latter the result depends on **which
    row was looked up** (a silent wrong value with no exception).
    MUST: `rowBytes` equals the value the declaration determines (layers / dim / storage). If it
    drifts, only the multiplication of the row offset shifts, quietly, and a row of another token is
    read with the shape and dtype still matching.
    """
    at = f"{where}.{name}"
    table = root.get(name)
    if not isinstance(table, dict):
        raise DistError(f"{at}: not an object")
    extra = sorted(set(table) - set(GEMMA4_PLE_TABLE_KEYS))
    if extra:
        raise DistError(f"{at}: unknown keys {extra} (allowed: {list(GEMMA4_PLE_TABLE_KEYS)})")
    declared = _positive_int(table, "rowBytes", at)
    if declared != row_bytes:
        raise DistError(
            f"{at}.rowBytes {declared} differs from {row_bytes}, set by the declaration"
        )
    blocks = table.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise DistError(f"{at}.blocks is not a non-empty array")
    seen: set[str] = set()
    expected = 0
    for position, entry in enumerate(blocks):
        block_at = f"{at}.blocks[{position}]"
        if not isinstance(entry, dict):
            raise DistError(f"{block_at}: not an object")
        unknown = sorted(set(entry) - set(GEMMA4_PLE_BLOCK_KEYS))
        if unknown:
            raise DistError(
                f"{block_at}: unknown keys {unknown} (allowed: {list(GEMMA4_PLE_BLOCK_KEYS)})"
            )
        asset = entry.get("asset")
        if not isinstance(asset, str) or not asset:
            raise DistError(f"{block_at}: asset is not a non-empty string ({asset!r})")
        if asset in seen:
            raise DistError(f"{block_at}: asset '{asset}' is duplicated")
        seen.add(asset)
        start = _offset(entry, "start", block_at)
        stop = _offset(entry, "stop", block_at)
        if start != expected:
            raise DistError(
                f"{block_at}: start {start} does not continue from the previous block's end"
                f" {expected}"
            )
        if stop <= start:
            raise DistError(f"{block_at}: range [{start}, {stop}) is empty")
        expected = stop
    if expected != tokens:
        raise DistError(f"{at}: the blocks total {expected} rows, not tokens {tokens}")
    return {"rowBytes": row_bytes, "blocks": blocks}


def gemma4_ple_index(
    container: Path, *, storage: str, group_size: int | None = None
) -> Mapping[str, Any]:
    """Read the `model` container's asset `ple_index` and check it **down to its form**.

    The mirror of the set the reader accepts (`parseGemma4PleIndex` in
    `packages/models/src/gemma/ple-index.ts`; the old sidecar index retired together with its
    distribution form — ADR 0109 decision 1's rule of bumping the major). `storage` is the storage
    the caller requires (`i2` / `i4` for QAT). `group_size` is the form the caller
    requires: `None` = schema 3 (one f32 scale per (token, layer) row), a group size = schema 4
    with exactly that `groupSize` (one f32 scale per group — the q4_0 QAT line's 32). The reader
    takes both; a distribution takes the one its line writes, so a series of the other form fails
    here.

    A float `storage` (`bf16` — `karume.ple.PLE_FLOAT_BYTES`) requires schema 5: a `values` table
    of the floats themselves and no `scales` (neither the key nor `groupSize`); the result then
    has no `scales` entry either.

    MUST: no form may first fail in the user's hands after it is distributed — the same checks run
    on the assembly side too.
    """
    where = f"{container}: asset '{PLE_INDEX_ASSET}'"
    float_bytes = PLE_FLOAT_BYTES.get(storage)
    if storage not in PLE_PACK_FACTOR and float_bytes is None:
        raise DistError(f"unsupported PLE storage: {storage}")
    if float_bytes is not None and group_size is not None:
        raise DistError(f"PLE storage '{storage}' has no scales, so no group size ({group_size})")
    if float_bytes is not None:
        schema = PLE_FLOAT_INDEX_SCHEMA
        allowed: tuple[str, ...] = tuple(
            key for key in GEMMA4_PLE_INDEX_KEYS if key != GEMMA4_PLE_SCALES_KEY
        )
    elif group_size is None:
        schema, allowed = PLE_INDEX_SCHEMA, GEMMA4_PLE_INDEX_KEYS
    else:
        schema = PLE_GROUPED_INDEX_SCHEMA
        allowed = (*GEMMA4_PLE_INDEX_KEYS, GEMMA4_PLE_GROUP_SIZE_KEY)
    try:
        payload = read_asset(container, PLE_INDEX_ASSET)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{where} cannot be read: {cause}") from cause
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DistError(f"{where} cannot be read as JSON") from error
    if not isinstance(raw, dict):
        raise DistError(f"{where}: the top level is not an object")
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise DistError(f"{where}: unknown keys {unknown} (allowed: {list(allowed)})")
    if raw.get("schema") != schema:
        raise DistError(
            f"{where}: schema is {raw.get('schema')!r} (expected {schema})"
            " — neither the old sidecar index nor a version this distribution does not write"
            " is read"
        )
    if raw.get("storage") != storage:
        raise DistError(f"{where}: storage is {raw.get('storage')!r} (expected {storage})")
    tokens = _positive_int(raw, "tokens", where)
    layers = _positive_int(raw, "layers", where)
    dim = _positive_int(raw, "dim", where)
    scale = raw.get("embedScale")
    if (
        not isinstance(scale, int | float)
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise DistError(f"{where}: embedScale is not a positive finite number ({scale!r})")
    if float_bytes is not None:
        return {
            "storage": storage,
            "tokens": tokens,
            "layers": layers,
            "dim": dim,
            "embedScale": scale,
            GEMMA4_PLE_VALUES_KEY: gemma4_ple_table(
                raw, GEMMA4_PLE_VALUES_KEY, layers * dim * float_bytes, tokens, where
            ),
        }
    factor = PLE_PACK_FACTOR[storage]
    if dim % factor:
        raise DistError(
            f"{where}: dim {dim} is not divisible by the pack factor of storage '{storage}'"
            f" ({factor})"
        )
    # The scales per (token, layer) row: one (schema 3), or one per group (schema 4).
    groups = 1
    if group_size is not None:
        declared = _positive_int(raw, GEMMA4_PLE_GROUP_SIZE_KEY, where)
        if declared != group_size:
            raise DistError(
                f"{where}: {GEMMA4_PLE_GROUP_SIZE_KEY} is {declared}, expected {group_size}"
                " (the grid the line stores)"
            )
        if dim % group_size:
            raise DistError(
                f"{where}: {GEMMA4_PLE_GROUP_SIZE_KEY} {group_size} does not divide dim {dim}"
            )
        groups = dim // group_size
    return {
        "storage": storage,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": scale,
        **({} if group_size is None else {GEMMA4_PLE_GROUP_SIZE_KEY: group_size}),
        GEMMA4_PLE_VALUES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_VALUES_KEY, layers * dim // factor, tokens, where
        ),
        GEMMA4_PLE_SCALES_KEY: gemma4_ple_table(
            raw, GEMMA4_PLE_SCALES_KEY, layers * groups * PLE_SCALE_BYTES, tokens, where
        ),
    }


def gemma4_placements(sources: Gemma4Sources) -> dict[str, Path]:
    """Role name → source file. The output paths are held by {@link GEMMA4_OUTPUT_PATHS}.

    Anything not in this table does not go into the output (this drops the `ple.probe.safetensors`
    and `reference.json` that sit in the product series and the drafter series'
    `drafter-golden.*.safetensors` — all of them serve acceptance and provenance, not running). The
    PLE lives inside the `model` container, so it has no entry. A model that ships no drafter
    (`sources.drafter is None`) has no drafter entry either, nor a model without a vision embedder
    (`sources.vision is None`) a vision entry; the vision series' `vision.*.safetensors` and
    greedy goldens stay behind the same way.
    """
    placements = {GEMMA4_ROLE: sources.product / GEMMA4_MODEL_FILE}
    if sources.drafter is not None:
        placements[GEMMA4_DRAFTER_ROLE] = sources.drafter / GEMMA4_MODEL_FILE
    if sources.vision is not None:
        placements[GEMMA4_VISION_ROLE] = sources.vision / GEMMA4_MODEL_FILE
    placements[GEMMA4_TOKENIZER_ROLE] = sources.tokenizer / GEMMA4_TOKENIZER_FILE
    return placements


#: The assets declaration (asset name → role name). The PLE moved into the container, so the
#: tokenizer is the only one left.
GEMMA4_ASSETS: Mapping[str, str] = {GEMMA4_TOKENIZER_ROLE: GEMMA4_TOKENIZER_ROLE}


def gemma4_vocab_size(graph: Mapping[str, Any], path: Path) -> int:
    """Read the vocabulary size of the selected-rows logits exit from the graph's output
    declaration (`[1, R, V]` — ADR 0083 decision 6).

    It checks that there are **two** outputs (in the order `logits[1, R, V]` → `hidden[1, R, H]`)
    because if the two acceptance series (logits opt-in / token-only) slip in under the same series
    name, **only the width becomes a number with another meaning**. The row axis is the symbol `R`
    because speculative verification scores several rows in one run; an ordinary decode binds it
    to 1. V is the row count of the main embedding itself, and the reference for cross-checking the
    PLE (a container asset) and the tokenizer (ADR 0085 decision 5).

    MUST: check the order too (logits and hidden have the same form down to the row axis and differ
    only in width) — accepting a swapped asset builds a manifest that read hidden_size as V.
    """
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 2:
        raise DistError(
            f"{path}: graph outputs are {outputs!r} — the product graph's exit is two outputs, the"
            " selected rows' logits + hidden"
        )
    values = graph.get("values")
    widths: list[int] = []
    for slot, what in enumerate(("logits", "hidden")):
        value = values.get(outputs[slot]) if isinstance(values, dict) else None
        shape = value.get("shape") if isinstance(value, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[0] != 1
            or shape[1] != GEMMA4_ROW_SYMBOL
        ):
            raise DistError(
                f"{path}: the shape of graph output '{outputs[slot]}' ({what}) is not"
                f" [1, {GEMMA4_ROW_SYMBOL}, *] ({shape!r})"
            )
        width = shape[2]
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise DistError(
                f"{path}: the width of graph output '{outputs[slot]}' is not a positive integer"
                f" ({width!r})"
            )
        widths.append(width)
    return widths[0]


def gemma4_text_config(model_dir: Path) -> SimpleNamespace:
    """The `text_config` section of upstream `config.json` (the source of the position limit and
    of the RoPE formula).

    It is turned into attribute access because {@link gemma4.rope.rope_specs} reads it the same way
    as a transformers config object — the writer and the distribution go through **the same
    derivation code**.
    """
    where = str(model_dir / GEMMA4_CONFIG_FILE)
    raw = _read_json(model_dir / GEMMA4_CONFIG_FILE, "the upstream model config")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: the top level is not an object")
    text = raw.get(GEMMA4_TEXT_CONFIG_KEY)
    if not isinstance(text, dict):
        raise DistError(f"{where}: {GEMMA4_TEXT_CONFIG_KEY} is not an object ({text!r})")
    return SimpleNamespace(**text)


def gemma4_max_position(text_config: SimpleNamespace, where: str) -> int:
    """The position limit the model declares (= `pipelineConfig.maxPosition`).

    MUST: do not copy it by hand (upstream `text_config.max_position_embeddings` is its only
    source). With RoPE moved to the host, the position limit is held by the model's declaration
    alone, not by an asset.
    """
    declared = getattr(text_config, GEMMA4_MAX_POSITION_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise DistError(
            f"{where}: {GEMMA4_MAX_POSITION_KEY} is not a positive integer ({declared!r})"
        )
    return declared


def gemma4_hidden_size(text_config: SimpleNamespace, where: str) -> int:
    """The hidden width the model declares (output 1 of the product graph = the width of the
    selected rows' hidden).

    MUST: do not copy it by hand (upstream `text_config.hidden_size` is its only source). The
    declaration and the graph come from separate sources of truth, so their fit is only seen in
    {@link assert_gemma4_graph} — the two outputs have the same form down to the row axis, so the
    one point that can catch a swap is **matching the width against the declaration**.
    """
    declared = getattr(text_config, GEMMA4_HIDDEN_SIZE_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise DistError(
            f"{where}: {GEMMA4_HIDDEN_SIZE_KEY} is not a positive integer ({declared!r})"
        )
    return declared


#: The upstream `text_config` field that sizes one layer's per-layer embedding (0 = no PLE).
GEMMA4_PLE_DIM_KEY = "hidden_size_per_layer_input"


def gemma4_ple_dim(text_config: Any, where: str) -> int:
    """The width of one layer's per-layer embedding the model declares; 0 = the model has none
    (12B's config says 0 — its class, `gemma4_unified`, has no PLE branch).

    Shared by the distribution and the export scripts (`text_config` is a namespace here and a
    transformers config there — both read attributes).

    MUST: read, never defaulted — a config without the field cannot say whether the graph should
    take `per_layer_inputs`, and guessing either way lets a graph / asset mix through.
    """
    declared = getattr(text_config, GEMMA4_PLE_DIM_KEY, None)
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        raise DistError(
            f"{where}: {GEMMA4_PLE_DIM_KEY} is not a non-negative integer ({declared!r})"
        )
    return declared


def gemma4_rope(text_config: SimpleNamespace, where: str) -> dict[str, Any]:
    """Derive `pipelineConfig.rope` (theta / headDim / rotaryDim per layer type) from the config.

    MUST: an unaccepted rope_type / coefficient fails loudly ({@link gemma4.rope.layer_spec}) —
    copying only theta and the width of a different formula makes the host build the table at
    **other angles that still match in shape and type**.
    MUST: exactly two layer types, sliding / full. Any more or fewer no longer fits the number of
    graph inputs ({@link GEMMA4_ROPE_INPUTS}).
    """
    try:
        specs = rope_specs(text_config)
    except RopeSpecError as error:
        raise DistError(f"{where}: cannot derive the RoPE declaration — {error}") from error
    if sorted(specs) != sorted(GEMMA4_ROPE_LAYER_TYPES):
        raise DistError(
            f"{where}: layer types are {sorted(specs)} — not {sorted(GEMMA4_ROPE_LAYER_TYPES)}"
        )
    return {layer_type: specs[layer_type].declaration() for layer_type in GEMMA4_ROPE_LAYER_TYPES}


def assert_gemma4_graph(
    graph: Mapping[str, Any],
    path: Path,
    index: Mapping[str, Any] | None,
    rope: Mapping[str, Any],
    hidden_size: int,
) -> None:
    """Measure the order, shapes and symbol split of the graph inputs before placement.

    `index` is the container's PLE index, or `None` for a model without per-layer embeddings
    (12B): its graph takes {@link GEMMA4_PLE_FREE_GRAPH_INPUTS} — `per_layer_inputs` absent — and
    every other rule below holds unchanged.

    MUST: the PLE's layer count and per-layer dim are held by both **the graph input declaration**
    and **the index** (the former `per_layer_inputs[1, M, 35, 256]`, the latter `layers` / `dim`).
    Distributed while they disagree, the table the host builds and the form the GPU reads differ —
    silently, for combinations whose shapes fit (the index is an asset of the same container, but
    the writer builds the two separately, so their fit still needs checking).

    MUST: the width of the derived RoPE inputs equals `headDim` of `pipelineConfig.rope`. The
    declaration and the graph come from separate sources of truth (config / container), so their
    fit can only be seen here — if the table the host builds as declared differs from the input
    width, nobody notices until run time.

    MUST: the width of output 1 (hidden) equals upstream `hidden_size`. The two outputs have the
    same form down to the row axis (both f32 `[1, R, *]`), so **only this width match can catch a
    swap** — letting a mix-up through builds a manifest that read hidden_size as the vocabulary
    size.

    MUST: no table initializer remains. A form that added the derived inputs but kept the tables
    brings the resident memory back and moves the position limit back to the asset side.

    MUST: the caller has run {@link gemma4_vocab_size} **first** (only the width is read, on the
    premise that there are two outputs, each with the three axes `[1, R, width]`).

    MUST: there are three symbols (the chunk rows `M`, the exit rows `R`, the full slots'
    capacity), and exactly one of them is **not determined by the input shapes** (the capacity
    symbol). The TS side's `gemma.Pipeline` makes that one the binding point of the capacity, so a
    different split fails at load time (the same check as `capacitySymbolOf`).
    `R` is bound by `last_row[R]`, so this count does not treat it as free.
    """
    inputs = graph_inputs(graph, path)
    names = tuple(inputs)
    expected_names = GEMMA4_GRAPH_INPUTS if index is not None else GEMMA4_PLE_FREE_GRAPH_INPUTS
    if names != expected_names:
        raise DistError(
            f"{path}: graph inputs are {list(names)}, not the expected {list(expected_names)}"
            " — the runtime binds by name, so if a single spelling changes they cannot be bound"
        )
    sequence = inputs[GEMMA4_INPUT_IDS][1]
    for layer_type in GEMMA4_ROPE_LAYER_TYPES:
        head_dim = rope[layer_type][HEAD_DIM_FIELD]
        for part in GEMMA4_ROPE_PARTS:
            name = gemma4_rope_input_name(layer_type, part)
            declared = inputs[name]
            if list(declared) != [1, sequence, head_dim]:
                raise DistError(
                    f"{path}: input '{name}' is {list(declared)!r} — not the"
                    f" {[1, sequence, head_dim]} expected from the declared headDim (the table and"
                    " the graph are of different generations)"
                )
    # In IR v2 an initializer name is the tensor key itself (docs/ir-v2.md) — found by spelling.
    baked = sorted(key for key in (graph.get("initializers") or {}) if BAKED_TABLE_INFIX in key)
    if baked:
        raise DistError(
            f"{path}: {len(baked)} baked RoPE table initializers remain: {baked[:4]}"
            " — an asset of a generation not yet moved fully to host generation"
        )
    if index is None:
        declared = {spec["name"]: spec for spec in graph["inputs"]}
        for name, dtype, shape in (
            (GEMMA4_SOFT_EMBEDS, "f32", [1, sequence, hidden_size]),
            (GEMMA4_SOFT_MASK, "bool", [1, sequence, 1]),
        ):
            spec = declared[name]
            if spec.get("dtype") != dtype or list(spec.get("shape")) != shape:
                raise DistError(
                    f"{path}: input '{name}' is {spec.get('dtype')} {spec.get('shape')!r} — not"
                    f" {dtype} {shape!r} (the host binds the image rows to this form)"
                )
    if index is not None:
        per_layer = inputs[GEMMA4_PER_LAYER_INPUTS]
        if len(per_layer) != 4:
            raise DistError(
                f"{path}: input '{GEMMA4_PER_LAYER_INPUTS}' is {per_layer!r} — not the 4 axes"
                " [1, M, layers, dim]"
            )
        for axis, field in ((2, "layers"), (3, "dim")):
            if per_layer[axis] != index[field]:
                raise DistError(
                    f"{path}: input '{GEMMA4_PER_LAYER_INPUTS}' axis {axis} is"
                    f" {per_layer[axis]!r}, "
                    f"but asset '{PLE_INDEX_ASSET}' has {field} {index[field]}"
                    " — the graph and the PLE are of different generations"
                )
    outputs = graph.get("outputs")
    values = graph.get("values")
    hidden = values.get(outputs[1]) if isinstance(values, dict) else None
    found = hidden.get("shape")[2] if isinstance(hidden, dict) else None
    if found != hidden_size:
        raise DistError(
            f"{path}: the width of output 1 (hidden) is {found!r} — not upstream"
            f" {GEMMA4_HIDDEN_SIZE_KEY} {hidden_size} (the two outputs are swapped"
            " / the graph and the declaration are of different generations)"
        )
    symbols = graph.get("symbols")
    if not isinstance(symbols, list):
        raise DistError(f"{path}: no symbols in the IR metadata")
    bound = {dim for shape in inputs.values() for dim in shape if isinstance(dim, str)}
    free = [symbol for symbol in symbols if symbol not in bound]
    if len(free) != 1:
        raise DistError(
            f"{path}: {len(free)} symbols not determined by the input shapes ({free})"
            " — there must be exactly one, the full slots' capacity symbol"
        )


def gemma4_processor_config(directory: Path) -> Mapping[str, Any]:
    """The `image_processor` section of upstream's `processor_config.json` (in the tokenizer
    source, as the tokenizer and the generation defaults are)."""
    path = directory / GEMMA4_PROCESSOR_CONFIG_FILE
    raw = _read_json(path, f"upstream's {GEMMA4_PROCESSOR_CONFIG_FILE}")
    section = raw.get("image_processor") if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        raise DistError(f"{path}: no image_processor section")
    return section


def gemma4_soft_token_budget(directory: Path) -> int:
    """The image token budget upstream's processor is configured with (`max_soft_tokens`)."""
    return _positive_int(
        gemma4_processor_config(directory),
        "max_soft_tokens",
        str(directory / GEMMA4_PROCESSOR_CONFIG_FILE),
    )


def _vocab_token(asset: Mapping[str, Any], token: int, path: Path) -> str:
    vocab = asset.get("vocab")
    if not isinstance(vocab, list) or not 0 <= token < len(vocab):
        raise DistError(f"{path}: id {token} is outside the tokenizer asset's vocabulary")
    return vocab[token]


def gemma4_vision_config(
    model_dir: Path, tokenizer_dir: Path, tokenizer_asset: Path, hidden_size: int
) -> dict[str, Any]:
    """`pipelineConfig.vision`: what the host needs to build an image's rows.

    Each number has a reader, so none is a literal here: the ids from the weights' `config.json`,
    the budget from `processor_config.json`, the width from the text config. MUST: the ids are held
    to the tokenizer asset the host encodes with, through their spellings in
    `tokenizer_config.json` — a declared id that is another token stays a valid integer and only
    shows as wrong rows at run time.
    """
    where = model_dir / GEMMA4_CONFIG_FILE
    raw = _read_json(where, f"upstream's {GEMMA4_CONFIG_FILE}")
    spelled = _read_json(
        tokenizer_dir / "tokenizer_config.json", "upstream's tokenizer_config.json"
    )
    asset = _read_json(tokenizer_asset, "the tokenizer asset")
    declared = {}
    for field, config_key, spelling_key in (
        ("imageTokenId", "image_token_id", "image_token"),
        ("beginImageTokenId", "boi_token_id", "boi_token"),
        ("endImageTokenId", "eoi_token_id", "eoi_token"),
    ):
        token = _offset(raw, config_key, str(where))
        spelling = spelled.get(spelling_key)
        found = _vocab_token(asset, token, tokenizer_asset)
        if found != spelling:
            raise DistError(
                f"{where}: {config_key} {token} is {found!r} in the tokenizer asset, not the"
                f" {spelling_key} {spelling!r} of tokenizer_config.json"
            )
        declared[field] = token
    return {
        "softTokenBudget": gemma4_soft_token_budget(tokenizer_dir),
        **declared,
        "hiddenSize": hidden_size,
    }


def assert_gemma4_vision_graph(
    graph: Mapping[str, Any], path: Path, hidden_size: int, tokenizer_dir: Path
) -> None:
    """The vision embedder graph, before placement: `pixels[1,N,P]` f32, `position_x[1,N]` and
    `position_y[1,N]` i32, one `embeds[1,N,H]` output, one symbol, no state.

    MUST: the row width is the one upstream's processor emits (`(patch_size · pooling)² · 3` from
    `processor_config.json`) and the output width the main graph's `hidden_size`: the graph comes
    from another series than the declaration and the main graph, so only this check sees a series of
    another model.
    """
    processor = gemma4_processor_config(tokenizer_dir)
    where = str(tokenizer_dir / GEMMA4_PROCESSOR_CONFIG_FILE)
    side = _positive_int(processor, "patch_size", where) * _positive_int(
        processor, "pooling_kernel_size", where
    )
    symbol = GEMMA4_VISION_SYMBOL
    expected = [
        {"name": GEMMA4_VISION_PIXELS, "dtype": "f32", "shape": [1, symbol, side * side * 3]},
        {"name": GEMMA4_POSITION_X, "dtype": "i32", "shape": [1, symbol]},
        {"name": GEMMA4_POSITION_Y, "dtype": "i32", "shape": [1, symbol]},
    ]
    found = [
        {key: spec.get(key) for key in ("name", "dtype", "shape")}
        for spec in graph.get("inputs", [])
    ]
    if found != expected:
        raise DistError(f"{path}: the vision graph's inputs are {found}, not {expected}")
    outputs = graph.get("outputs")
    values = graph.get("values")
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(values, dict):
        raise DistError(f"{path}: the vision graph does not have one output ({outputs!r})")
    shape = (values.get(outputs[0]) or {}).get("shape")
    if shape != [1, symbol, hidden_size]:
        raise DistError(
            f"{path}: the vision graph's output is {shape!r}, not {[1, symbol, hidden_size]!r}"
            f" — not upstream {GEMMA4_HIDDEN_SIZE_KEY} {hidden_size}"
        )
    if graph.get("symbols") != [symbol]:
        raise DistError(f"{path}: the vision graph's symbols are {graph.get('symbols')!r}")
    if graph.get("states"):
        raise DistError(f"{path}: the vision graph holds state slots")


def assert_gemma4_vision_series(series: Path, tokenizer_dir: Path) -> None:
    """The vision series is a full one, taken at the budget this distribution declares.

    MUST: the series holds its record ({@link GEMMA4_REFERENCE_FILE}) and every greedy golden the
    record names — `gemma4/export_vision.py --embedder-only` writes the container and the embedder
    goldens alone, and a vision role placed from such a series ships with neither its image goldens
    nor its provenance record.
    MUST: the record's `softTokenBudget` is the budget `pipelineConfig.vision` declares and its
    `processorConfig` digest the `processor_config.json` it is read from: goldens taken at another
    budget hold the host to rows it never builds, and a cached file edited by hand changes the
    declaration alone.
    """
    record_path = series / GEMMA4_REFERENCE_FILE
    if not record_path.is_file():
        raise DistError(
            f"{series}: {GEMMA4_REFERENCE_FILE} is missing — not a full vision series (an"
            " --embedder-only run writes none); write it with python -m gemma4.export_vision"
        )
    record = _read_json(record_path, "the vision series' record")
    reference = record.get("reference") if isinstance(record, dict) else None
    if not isinstance(reference, dict):
        raise DistError(f"{record_path}: no reference section")
    goldens = reference.get("goldens")
    if not isinstance(goldens, dict) or not goldens:
        raise DistError(f"{record_path}: names no greedy goldens ({goldens!r})")
    missing = sorted(name for name in goldens if not (series / name).is_file())
    if missing:
        raise DistError(
            f"{series}: the greedy goldens {missing} that {GEMMA4_REFERENCE_FILE} names are"
            " missing — not a full vision series"
        )
    config_path = tokenizer_dir / GEMMA4_PROCESSOR_CONFIG_FILE
    budget = gemma4_soft_token_budget(tokenizer_dir)
    recorded_budget = reference.get("softTokenBudget")
    if recorded_budget != budget:
        raise DistError(
            f"{record_path}: the goldens were taken at softTokenBudget {recorded_budget!r}, the"
            f" distribution declares {budget} (max_soft_tokens of {config_path})"
        )
    digest = sha256_file(config_path)
    processor = reference.get("processorConfig")
    recorded_digest = processor.get("sha256") if isinstance(processor, dict) else None
    if recorded_digest != digest:
        raise DistError(
            f"{record_path}: the goldens were taken with {GEMMA4_PROCESSOR_CONFIG_FILE} SHA-256"
            f" {recorded_digest!r}, the distribution reads {digest} ({config_path})"
        )


def assert_gemma4_drafter_graph(
    graph: Mapping[str, Any],
    path: Path,
    lender: Mapping[str, Any],
    lender_path: Path,
    rope: Mapping[str, Any],
    hidden_size: int,
) -> None:
    """Measure the form of the drafter graph (the borrower) and **its fit with the lender** before
    placement.

    The borrower is an asset that cannot run alone, so half of what to check is "it matches the
    lender". Distributed while they disagree, it either fails at load time (a name differs) or
    **reads other columns with the shape and type still matching** (the capacity differs).

    - the graph inputs are exactly `token` / `hidden` / the four RoPE inputs (if a K/V placeholder
      remained, no node would read the value the caller passes)
    - the width of `hidden` equals upstream `hidden_size` (= the width of output 1 of the product
      graph)
    - the width of the derived RoPE inputs equals `headDim` of `pipelineConfig.rope` (the same
      match as for the lender)
    - exactly k outputs ({@link GEMMA4_DRAFT_STEPS} in the distribution)
    - the `states` are **all external**, and their names and shapes are **one to one with the
      lender's declaration**
    - the initializers hold **exactly one shared declaration**, whose target is a tensor key that
      exists in the lender container
    - the only symbol is the capacity symbol, **spelled as in the lender** (the borrower's bindings
      inherit the lender's)
    """
    inputs = graph_inputs(graph, path)
    names = tuple(inputs)
    if names != GEMMA4_DRAFTER_GRAPH_INPUTS:
        raise DistError(
            f"{path}: graph inputs are {list(names)}, not the expected"
            f" {list(GEMMA4_DRAFTER_GRAPH_INPUTS)}"
            " — an asset of a generation whose K/V placeholders the surgery did not remove"
        )
    hidden = inputs[GEMMA4_DRAFTER_HIDDEN]
    if list(hidden) != [1, hidden_size]:
        raise DistError(
            f"{path}: input '{GEMMA4_DRAFTER_HIDDEN}' is {list(hidden)!r} —"
            f" not [1, {hidden_size}] (one row of the target's hidden after the final norm)"
        )
    for layer_type in GEMMA4_ROPE_LAYER_TYPES:
        head_dim = rope[layer_type][HEAD_DIM_FIELD]
        for part in GEMMA4_ROPE_PARTS:
            name = gemma4_rope_input_name(layer_type, part)
            declared = inputs[name]
            if list(declared) != [1, 1, head_dim]:
                raise DistError(
                    f"{path}: input '{name}' is {list(declared)!r} — not the"
                    f" {[1, 1, head_dim]} expected from the declared headDim (the table and the"
                    " graph are of different generations)"
                )
    outputs = graph.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != GEMMA4_DRAFT_STEPS:
        raise DistError(f"{path}: IR outputs are {outputs!r} — not {GEMMA4_DRAFT_STEPS} drafts")

    borrowed = graph.get("states")
    lent = lender.get("states")
    if not isinstance(borrowed, dict) or not isinstance(lent, dict):
        raise DistError(f"{path}: no states in the IR metadata (of the borrower or the lender)")
    for name, slot in sorted(borrowed.items()):
        if not slot.get("external"):
            raise DistError(
                f"{path}: state slot '{name}' is not external"
                " (the borrower holds no storage of its own — every slot MUST be external)"
            )
        owner = lent.get(name)
        if owner is None:
            raise DistError(
                f"{path}: state slot '{name}' is not in the lender {lender_path}"
                f" (the lender declares: {sorted(lent)})"
            )
        if owner.get("dtype") != slot.get("dtype") or owner.get("shape") != slot.get("shape"):
            raise DistError(
                f"{path}: state slot '{name}' is {slot.get('dtype')} {slot.get('shape')} —"
                f" not the lender's {owner.get('dtype')} {owner.get('shape')} (a pair of different"
                " generations)"
            )

    initializers = graph.get("initializers") or {}
    # In IR v2 an initializer name is the tensor key itself (docs/ir-v2.md), so the **name** of the
    # shared declaration is its target (v1's `shared.tensor` field is gone).
    shared = sorted(
        name
        for name, entry in initializers.items()
        if isinstance(entry, dict) and entry.get("shared") is True
    )
    if len(shared) != 1:
        raise DistError(
            f"{path}: {len(shared)} shared initializers {shared} (exactly one expected)"
        )
    tensor = shared[0]
    lent_tensors = set(lender.get("initializers") or {})
    if tensor not in lent_tensors:
        raise DistError(
            f"{path}: the target '{tensor}' of the shared initializer is not in the lender"
            f" {lender_path} (it does not fit the generation the lender was re-baked as)"
        )

    symbols = graph.get("symbols")
    lent_symbols = lender.get("symbols")
    if not isinstance(symbols, list) or len(symbols) != 1:
        raise DistError(
            f"{path}: symbols are {symbols!r} — there must be the capacity symbol only"
            " (the borrower's inputs bind no symbol)"
        )
    if not isinstance(lent_symbols, list) or symbols[0] not in lent_symbols:
        raise DistError(
            f"{path}: capacity symbol '{symbols[0]}' is not among the symbols of the lender"
            f" {lender_path} ({lent_symbols!r})"
            " — the borrower inherits its bindings from the lender"
        )


def assert_gemma4_ple_assets(container: Path, index: Mapping[str, Any]) -> None:
    """Check that every block the index names exists as an asset of that container with **the
    same role and the same length**.

    MUST: the index and the container's declarations can move independently (the index is JSON the
    writer built, the asset declarations are the container's model descriptor), so their fit can
    only be seen here — a combination in which only the index was replaced reads rows of other
    tokens **with shape and dtype still matching** (the same rule as `assertGemma4PleAssets` in the
    reader `ple-index.ts`). Only the declarations are read, so this does not re-read gigabytes.
    """
    try:
        declared = read_asset_declarations(container)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{container}: cannot read the asset declarations: {cause}") from cause
    seen = {PLE_INDEX_ASSET}
    if declared.get(PLE_INDEX_ASSET, (None, 0))[0] != PLE_INDEX_ROLE:
        raise DistError(
            f"{container}: asset '{PLE_INDEX_ASSET}' has role"
            f" {declared.get(PLE_INDEX_ASSET, (None, 0))[0]!r} (expected '{PLE_INDEX_ROLE}')"
        )
    # A float index (schema 5) has no `scales` table; a `ple-scales` asset in its container is
    # then surplus below.
    for key, role in PLE_ROLES.items():
        if key not in index:
            continue
        table = index[key]
        for block in table["blocks"]:
            name = str(block["asset"])
            seen.add(name)
            found = declared.get(name)
            if found is None:
                raise DistError(
                    f"{container}: asset '{name}' named by the index is not in the container"
                    f" (declared: {sorted(declared)[:4]}…)"
                )
            expected = (int(block["stop"]) - int(block["start"])) * table["rowBytes"]
            if found != (role, expected):
                raise DistError(
                    f"{container}: asset '{name}' is {found}"
                    f" (the index implies ('{role}', {expected}))"
                )
    surplus = sorted(
        name
        for name, (role, _length) in declared.items()
        if name not in seen and role in {PLE_INDEX_ROLE, *PLE_ROLES.values()}
    )
    if surplus:
        raise DistError(
            f"{container}: PLE assets the index does not point to: {surplus}"
            " — a combination in which only the index is old"
        )


def assert_gemma4_no_ple_assets(container: Path) -> None:
    """A model without per-layer embeddings ships a container without PLE assets (the inverted
    form of {@link assert_gemma4_ple_assets}).

    MUST: fail loudly on any — a PLE index or block in a container whose graph takes no
    `per_layer_inputs` is another model's material (or a half-converted series), and the reader
    would either ignore it silently or, finding the index, build a PLE nobody feeds.
    """
    try:
        declared = read_asset_declarations(container)
    except (KeyError, ContainerError, ContainerFormatError) as cause:
        raise DistError(f"{container}: cannot read the asset declarations: {cause}") from cause
    ple_roles = {PLE_INDEX_ROLE, *PLE_ROLES.values()}
    carried = sorted(name for name, (role, _length) in declared.items() if role in ple_roles)
    if carried:
        raise DistError(
            f"{container}: the model has no per-layer embeddings, yet the container carries PLE"
            f" assets {carried[:4]} — a graph / asset mix"
        )


def assert_gemma4_tokenizer(path: Path, vocab_size: int) -> None:
    """Check that the compiled tokenizer asset **was baked with this vocabulary**.

    MUST: match down to the row count (the writer side applies ADR 0085 decision 5's cross-check
    too) — an asset of another vocabulary reads **valid rows of other tokens** as long as the ids
    stay in range, and breaks silently, without an exception. The TS side's admission has the same
    check, but it only runs after 4 GiB were downloaded.
    """
    raw = _read_json(path, "the compiled tokenizer asset")
    if not isinstance(raw, dict):
        raise DistError(f"{path}: the top level is not an object")
    if raw.get("format") != GEMMA4_TOKENIZER_FORMAT:
        raise DistError(
            f"{path}: format is {raw.get('format')!r} (expected '{GEMMA4_TOKENIZER_FORMAT}')"
            " — not an asset written by the compile script (`python -m gemma4.tokenizer`)"
        )
    vocab = raw.get("vocab")
    if not isinstance(vocab, list):
        raise DistError(f"{path}: vocab is not an array")
    if len(vocab) != vocab_size:
        raise DistError(
            f"{path}: vocab has {len(vocab)} rows, not the product graph's vocabulary size"
            f" {vocab_size} — a combination baked with another vocabulary"
        )


def assert_gemma4_tokenizer_source(path: Path, checkpoint_tokenizer: Path) -> None:
    """The compiled tokenizer asset was compiled from **the same bytes as this checkpoint's
    `tokenizer.json`** — the gate every model's asset passes ({@link gemma4_plan}).

    The compile script (`_shared.gemma_tokenizer.emit`) writes the source's SHA-256 into the asset's
    `source` field; comparing that field with the model's upstream file is the only way to see
    both failures it guards against: a series borrowed from another model whose upstream is no
    longer the same bytes ({@link GEMMA4_TOKENIZER_SERIES}), and a model's own series compiled
    before its upstream file was replaced (Google's glitch-token fixes of 2026-11-03 changed E2B's
    and E4B's `tokenizer.json`).

    MUST: matching the vocabulary size ({@link assert_gemma4_tokenizer}) is not enough — a
    tokenizer of the same vocabulary size that differs only in its merges or special tokens
    produces **other id sequences** with the ids still in range.
    """
    raw = _read_json(path, "the compiled tokenizer asset")
    source = raw.get("source") if isinstance(raw, dict) else None
    recorded = source.get("sha256") if isinstance(source, dict) else None
    if not isinstance(recorded, str):
        raise DistError(
            f"{path}: no source.sha256 — an asset whose source cannot be checked is not distributed"
        )
    if not checkpoint_tokenizer.is_file():
        raise DistError(f"the upstream tokenizer is missing: {checkpoint_tokenizer}")
    actual = sha256_file(checkpoint_tokenizer)
    if recorded != actual:
        raise DistError(
            f"{path} was compiled from a tokenizer.json with SHA-256 {recorded}, but"
            f" {checkpoint_tokenizer} is {actual} — the asset is stale or another model's"
            " (compile the model's tokenizer series again from this file:"
            " `python -m gemma4.tokenizer --model <model>`)"
        )


@dataclass(frozen=True)
class Gemma4GenerationDefaults:
    """What a distribution carries from upstream `generation_config.json`."""

    #: `pipelineConfig.sampler` (TS `SamplerSpec` field names).
    sampler: Mapping[str, Any]
    #: `pipelineConfig.suppressTokens` — unique ids, ascending; `None` when upstream declares none.
    suppress_tokens: tuple[int, ...] | None


def gemma4_generation_defaults(model_dir: Path) -> Gemma4GenerationDefaults:
    """Read upstream `generation_config.json`: the recommended sampler and the suppressed ids.

    MUST: every key is mapped ({@link GEMMA4_SAMPLER_FIELDS}, {@link GEMMA4_SUPPRESS_TOKENS_KEY})
    or listed as not carried with its reason ({@link GEMMA4_GENERATION_NOT_CARRIED}); any other
    key fails loudly. The suppressed ids must be a non-empty list of unique non-negative integers
    (the vocabulary bound is checked against the graph — {@link assert_gemma4_suppress_tokens}).
    """
    where = str(model_dir / GEMMA4_GENERATION_CONFIG_FILE)
    raw = _read_json(model_dir / GEMMA4_GENERATION_CONFIG_FILE, "the upstream generation defaults")
    if not isinstance(raw, dict):
        raise DistError(f"{where}: the top level is not an object")
    known = {
        *(source for source, _ in GEMMA4_SAMPLER_FIELDS),
        GEMMA4_SUPPRESS_TOKENS_KEY,
        *GEMMA4_GENERATION_NOT_CARRIED,
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise DistError(
            f"{where}: generation defaults {unknown} are neither carried nor listed as"
            " intentionally not carried (gemma4.distribution.GEMMA4_GENERATION_NOT_CARRIED)"
        )
    suppress = raw.get(GEMMA4_SUPPRESS_TOKENS_KEY)
    return Gemma4GenerationDefaults(
        sampler=_sampler(raw, where),
        suppress_tokens=None if suppress is None else _suppress_tokens(suppress, where),
    )


def _suppress_tokens(value: Any, where: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(one, int) and not isinstance(one, bool) for one in value)
    ):
        raise DistError(
            f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} is not a non-empty list of integers ({value!r})"
        )
    if len(set(value)) != len(value):
        raise DistError(f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} repeats an id ({value!r})")
    if min(value) < 0:
        raise DistError(f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} holds a negative id ({value!r})")
    return tuple(sorted(value))


def assert_gemma4_suppress_tokens(ids: tuple[int, ...], vocab_size: int, where: str) -> None:
    """Every suppressed id names a token of the graph's vocabulary (`[0, vocab)`)."""
    outside = [token for token in ids if token >= vocab_size]
    if outside:
        raise DistError(
            f"{where}: {GEMMA4_SUPPRESS_TOKENS_KEY} {outside} outside the graph's vocabulary"
            f" [0, {vocab_size})"
        )


def gemma4_sampler(model_dir: Path) -> dict[str, Any]:
    """The recommended sampler alone, for a pipeline whose `pipelineConfig` has no
    `suppressTokens` (the QAT family's).

    MUST: a checkpoint that declares suppressed ids fails loudly here — such a caller would drop
    them, and its generations would select ids upstream never selects.
    """
    defaults = gemma4_generation_defaults(model_dir)
    if defaults.suppress_tokens is not None:
        raise DistError(
            f"{model_dir / GEMMA4_GENERATION_CONFIG_FILE}: declares {GEMMA4_SUPPRESS_TOKENS_KEY}"
            f" {list(defaults.suppress_tokens)}, which this pipeline's configuration cannot carry"
        )
    return dict(defaults.sampler)


def _sampler(raw: Mapping[str, Any], where: str) -> dict[str, Any]:
    """Copy the recommended sampler of upstream `generation_config.json` into
    `pipelineConfig.sampler`.

    MUST: do not copy the values by hand (ADR 0083 decision 7 — the source of "the distribution
    declares the defaults" is upstream's declaration itself). Only the field names are translated
    into the spellings of the TS `SamplerSpec` (`top_k` → `topK`). The value ranges are the same as
    the TS side's `gemma.parsePipelineConfig` — no form may fail at parse after distribution.
    """
    sampler: dict[str, Any] = {}
    for source, field in GEMMA4_SAMPLER_FIELDS:
        if source not in raw:
            raise DistError(
                f"{where}: {source} is missing — the recommended sampler cannot be declared"
            )
        value = raw[source]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DistError(f"{where}: {source} is not a number ({value!r})")
        sampler[field] = value
    temperature = sampler["temperature"]
    if not math.isfinite(temperature) or temperature < 0:
        raise DistError(
            f"{where}: temperature is not a non-negative finite number ({temperature!r})"
        )
    top_k = sampler["topK"]
    if not isinstance(top_k, int) or top_k < 1:
        raise DistError(f"{where}: top_k is not an integer of at least 1 ({top_k!r})")
    top_p = sampler["topP"]
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise DistError(f"{where}: top_p is not a finite number in (0, 1] ({top_p!r})")
    sampler["temperature"] = float(temperature)
    sampler["topP"] = float(top_p)
    return sampler


def gemma4_pipeline_config(
    max_position: int,
    rope: Mapping[str, Any],
    sampler: Mapping[str, Any],
    *,
    chunk_length: int,
    max_chunk_length: int,
    capacity: int,
    chat_dialect: str = GEMMA4_CHAT_DIALECT,
    suppress_tokens: tuple[int, ...] | None = None,
    vision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build `pipelineConfig` (the six fields of the TS-side schema).

    `chat_dialect` is the optional seventh field `chatDialect` (`gemma4.chat.CHAT_DIALECTS` — a
    closed set). MUST: the default dialect is **not written** — an absent field reads as `gemma4`,
    so E2B / E4B (and the QAT family, which shares this function) keep their manifests byte for
    byte; only another dialect (12B's `gemma4-unified`) adds the key.

    `suppress_tokens` is the optional field `suppressTokens` (ids whose logits are −inf before any
    host token selection — {@link gemma4_generation_defaults}), written only when upstream
    declares them (12B), for the same byte-identity reason. `vision` is the optional field of the
    models that ship a vision embedder ({@link gemma4_vision_config}), written only for them.

    The three runtime knobs (`chunk_length` / `max_chunk_length` / `capacity`) are **passed by the
    caller** — regular Gemma passes {@link GEMMA4_CHUNK_LENGTH} / {@link GEMMA4_MAX_CHUNK_LENGTH} /
    {@link GEMMA4_CAPACITY}, the fixed QAT family the same kind of constants of `gemma4_qat.config`.
    The relations have one implementation only, so a new family cannot produce "a distribution that
    checks only one of the three relations".

    MUST: reject here a runtime knob outside **the bounds on either side**. The chunk row count
    cannot exceed the trace-time bound of the symbol `M` (`max_chunk_length`), and `capacity` is the
    largest logical length a conversation can use, so positions go up to `capacity - 1` — a
    capacity beyond the model's declaration (`maxPosition`) "rotates positions upstream did not plan
    for while inside the declaration", and shows up only in long conversations.

    MUST: **also put** that chunk bound into the declaration as `maxChunkLength`. The check here
    only sees the defaults the distribution bakes, while `chunkLength` is also the reader's runtime
    knob — without the declaration, the reader cannot reject "running with a chunk length outside
    the traced range" (the IR's `symbols` is only a list of names, without bounds).

    NOTE: the bound side **only checks that a copy is equal**. The comparison target
    {@link GEMMA4_MAX_CHUNK_LENGTH} is a constant copied from `gemma4.export.SYM_MAX`, and the
    `--sym-max` actually used when the series was built is recorded nowhere (it cannot be read from
    the asset either). A distribution that claims `chunkLength: 768` for a container traced with a
    smaller `--sym-max` passes both this check and the equality test.
    """
    if not 2 <= chunk_length <= max_chunk_length:
        raise DistError(
            f"chunkLength {chunk_length} is outside [2, {max_chunk_length}]"
            " (the lower bound is that of the symbol M, the upper bound that of the trace-time"
            " `Dim`)"
        )
    if capacity < chunk_length:
        raise DistError(
            f"capacity {capacity} is smaller than chunkLength {chunk_length}"
            " — a capacity in which not even one chunk fits cannot be declared"
        )
    if max_position < capacity:
        raise DistError(
            f"capacity {capacity} exceeds the model's position limit {max_position}"
            " — a conversation filling the capacity would rotate positions outside the declaration"
        )
    if chat_dialect not in GENERATION_PROMPT_SUFFIX:
        raise DistError(
            f"chatDialect {chat_dialect!r} is not one of {sorted(GENERATION_PROMPT_SUFFIX)}"
        )
    config: dict[str, Any] = {
        "chunkLength": chunk_length,
        "maxChunkLength": max_chunk_length,
        "maxPosition": max_position,
        "capacity": capacity,
        "rope": {layer_type: dict(spec) for layer_type, spec in rope.items()},
        "sampler": dict(sampler),
    }
    if chat_dialect != GEMMA4_CHAT_DIALECT:
        config["chatDialect"] = chat_dialect
    if suppress_tokens is not None:
        if not suppress_tokens or list(suppress_tokens) != sorted(set(suppress_tokens)):
            raise DistError(
                f"suppressTokens {list(suppress_tokens)} is not a non-empty ascending list of"
                " unique ids"
            )
        config["suppressTokens"] = list(suppress_tokens)
    if vision is not None:
        config["vision"] = dict(vision)
    return config


@dataclass(frozen=True)
class Gemma4PleForm:
    """The PLE index a line's product container must carry ({@link gemma4_ple_index})."""

    #: The values' storage: `bf16` (schema 5 — the floats, no scales) or `i8` / `i4` / `i2`.
    storage: str
    #: `None` = schema 5 for a float storage, else schema 3 (one f32 scale per (token, layer)
    #: row); a group size = schema 4.
    group_size: int | None = None


@dataclass(frozen=True)
class Gemma4Line:
    """What one distribution of the `gemma4/1` contract decides per model; {@link gemma4_plan}
    runs every other check, and builds the pipeline configuration, the same way for each line.

    The regular line ({@link GEMMA4_REGULAR_LINE} — `karume-gemma4`) and the q4_0 QAT line
    ({@link GEMMA4_QAT_Q4_0_LINE} — `karume-gemma4-qat`) differ in which models ship a
    drafter, what each role stores, the PLE index's form, the quant table and the pins — and in
    their models: 12B ships in the q4_0 QAT line only.
    """

    #: The models the line ships (a model outside it fails the plan).
    models: tuple[str, ...]
    #: Whether the model ships an MTP drafter (a weights role of its own).
    has_drafter: Callable[[str], bool]
    #: Whether the model ships a vision embedder (a weights role of its own).
    has_vision: Callable[[str], bool]
    #: The storage gates (the same for every model of the line).
    storage: Gemma4StorageTables
    #: The PLE index form (for the models with per-layer embeddings).
    ple: Gemma4PleForm
    #: Role → output path within the model subtree.
    output_paths: Mapping[str, str]
    #: The model's weights declaration (dtype label → role) — the drafter role only when it ships.
    weights: Callable[[str], Mapping[str, Mapping[str, WeightFiles]]]
    #: The model's quant table (before {@link complete_quant_weights}) and default quant.
    quants: Callable[[str], dict[str, Any]]
    default_quant: Mapping[str, str]
    #: Model → its pinned upstream sources (absent = hand-placed): the product containers must then
    #: name the weights' revision. (The tokenizer asset is held to the upstream `tokenizer.json`
    #: for every model, pinned or not.)
    pins: Mapping[str, ModelPins]
    #: Model → the pin of its drafter, whose revision the drafter container must name (absent =
    #: the drafter container is not held to a revision — the regular E2B drafter records none).
    drafter_pins: Mapping[str, HubPin]


#: The regular line (`karume-gemma4` — {@link PIPELINE}): the tables above. Its checkpoints are
#: hand-placed (`inputs/gemma4/` — `gemma4.sources.checkpoint_dir`), so it has no pins.
GEMMA4_REGULAR_LINE = Gemma4Line(
    models=GEMMA4_REGULAR_MODELS,
    has_drafter=gemma4_has_drafter,
    has_vision=lambda _model: False,
    storage=GEMMA4_STORAGE,
    ple=Gemma4PleForm(storage=GEMMA4_DTYPE),
    output_paths=GEMMA4_OUTPUT_PATHS,
    weights=gemma4_weights,
    quants=gemma4_quants,
    default_quant=GEMMA4_DEFAULT_QUANT,
    pins={},
    drafter_pins={},
)


def gemma4_plan(
    sources: Gemma4Sources,
    model: str = GEMMA4_DEFAULT_MODEL,
    line: Gemma4Line = GEMMA4_REGULAR_LINE,
) -> ModelPlan:
    """Build the plan of one gemma4 model (every check and read is done here).

    `line` holds what the distribution decides per model ({@link Gemma4Line}); the regular line
    is the default.

    Whether a drafter exists is decided by the model (`line.has_drafter`). MUST: fail when the
    drafter entry of `sources` disagrees — an empty entry for a model that ships a drafter would
    quietly produce a distribution without speculative decoding, and an entry for a model that
    ships none would let a drafter meant for another model slip in. The same MUST holds for the
    vision embedder (`line.has_vision`).
    """
    assert_model_name(model)
    gemma4_checkpoint(model)
    if model not in line.models:
        raise DistError(
            f"model '{model}' is not one of this line's models ({' / '.join(line.models)})"
        )
    if (sources.drafter is not None) != line.has_drafter(model):
        raise DistError(
            f"model '{model}': the drafter series is"
            f" {'missing' if sources.drafter is None else f'given ({sources.drafter})'}"
            f" — this line ships {'a' if line.has_drafter(model) else 'no'} drafter for it"
        )
    if (sources.vision is not None) != line.has_vision(model):
        raise DistError(
            f"model '{model}': the vision series is"
            f" {'missing' if sources.vision is None else f'given ({sources.vision})'}"
            f" — this line ships {'a' if line.has_vision(model) else 'no'} vision embedder for it"
        )
    placements = gemma4_placements(sources)
    for role, source in placements.items():
        assert_storage(role, source, line.storage.required)
        assert_storage_absent(role, source, line.storage.forbidden)
    text_config = gemma4_text_config(sources.model)
    where = str(sources.model / GEMMA4_CONFIG_FILE)
    rope = gemma4_rope(text_config, where)
    # MUST: the model table (what the card states) and the upstream config agree on whether the
    # model has per-layer embeddings; the graph and the container assets are then held to the
    # same answer below — a mix of the two forms fails before a byte is placed.
    ple_dim = gemma4_ple_dim(text_config, where)
    if (ple_dim > 0) != (model in GEMMA4_PLE_MODELS):
        raise DistError(
            f"{where}: {GEMMA4_PLE_DIM_KEY} is {ple_dim}, but model '{model}' is"
            f" {'' if model in GEMMA4_PLE_MODELS else 'not '}listed as carrying per-layer"
            " embeddings (gemma4.card.GEMMA4_PLE_MODELS)"
        )
    container = placements[GEMMA4_ROLE]
    index = (
        gemma4_ple_index(container, storage=line.ple.storage, group_size=line.ple.group_size)
        if ple_dim
        else None
    )
    graph = ir_graph(container)
    vocab_size = gemma4_vocab_size(graph, container)
    hidden_size = gemma4_hidden_size(text_config, where)
    assert_gemma4_graph(graph, container, index, rope, hidden_size)
    if index is None:
        assert_gemma4_no_ple_assets(container)
    else:
        if index["tokens"] != vocab_size:
            raise DistError(
                f"{container}: asset '{PLE_INDEX_ASSET}': tokens {index['tokens']} is not the"
                f" product graph's vocabulary size {vocab_size}"
                " — a combination baked with another vocabulary"
            )
        assert_gemma4_ple_assets(container, index)
    drafter_container = placements.get(GEMMA4_DRAFTER_ROLE)
    if drafter_container is not None:
        assert_gemma4_drafter_graph(
            ir_graph(drafter_container), drafter_container, graph, container, rope, hidden_size
        )
    vision_container = placements.get(GEMMA4_VISION_ROLE)
    if vision_container is not None:
        assert_gemma4_vision_graph(
            ir_graph(vision_container), vision_container, hidden_size, sources.upstream_tokenizer
        )
    if sources.vision is not None:
        assert_gemma4_vision_series(sources.vision, sources.upstream_tokenizer)
    assert_gemma4_tokenizer(placements[GEMMA4_TOKENIZER_ROLE], vocab_size)
    # MUST: every model's asset is held to the upstream `tokenizer.json` this plan reads for it,
    # in every line. A borrowed series whose upstream is other bytes, a model's own series compiled
    # before its upstream file was replaced (Google's glitch-token fixes), and another model's or
    # another repository's file (the QAT weights' repositories ship the pre-fix tokenizer) all have
    # the right vocabulary and tokenize some inputs differently.
    assert_gemma4_tokenizer_source(
        placements[GEMMA4_TOKENIZER_ROLE],
        sources.upstream_tokenizer / GEMMA4_CHECKPOINT_TOKENIZER_FILE,
    )
    pins = line.pins.get(model)
    if pins is not None:
        # The container names the weights' commit it was exported from
        # (`gemma4/export_unified.py` bakes it); one from another revision fails here.
        assert_upstream_provenance(
            container, license=GEMMA4_LICENSE, revision=pins.weights.revision
        )
        # The vision container comes from the same checkpoint (`gemma4/export_vision.py` bakes it).
        if vision_container is not None:
            assert_upstream_provenance(
                vision_container, license=GEMMA4_LICENSE, revision=pins.weights.revision
            )
    drafter_pin = line.drafter_pins.get(model)
    if drafter_container is not None and drafter_pin is not None:
        # The drafter container names its drafter's commit (`gemma4/export_drafter_qat.py` bakes
        # it): a drafter exported from another revision of the assistant fails here, as the
        # product does above.
        assert_upstream_provenance(
            drafter_container, license=GEMMA4_LICENSE, revision=drafter_pin.revision
        )
    generation = gemma4_generation_defaults(sources.upstream_tokenizer)
    if generation.suppress_tokens is not None:
        assert_gemma4_suppress_tokens(
            generation.suppress_tokens,
            vocab_size,
            str(sources.upstream_tokenizer / GEMMA4_GENERATION_CONFIG_FILE),
        )
    pipeline_config = gemma4_pipeline_config(
        gemma4_max_position(text_config, where),
        rope,
        generation.sampler,
        chunk_length=GEMMA4_CHUNK_LENGTH,
        max_chunk_length=GEMMA4_MAX_CHUNK_LENGTH,
        capacity=GEMMA4_CAPACITY,
        chat_dialect=CHAT_DIALECTS[model],
        suppress_tokens=generation.suppress_tokens,
        vision=(
            None
            if vision_container is None
            else gemma4_vision_config(
                sources.model,
                sources.upstream_tokenizer,
                placements[GEMMA4_TOKENIZER_ROLE],
                hidden_size,
            )
        ),
    )
    weights = line.weights(model)
    return ModelPlan(
        name=model,
        pipeline=GEMMA4_PIPELINE,
        artifacts={
            role: Artifact(line.output_paths[role], source=source)
            for role, source in placements.items()
        },
        weights=weights,
        assets=GEMMA4_ASSETS,
        # requiredLimits is not written — the core's dist derives and bakes it all at once during
        # assembly (karume/limits.py; a hand-written value on the plan side is rejected as double
        # bookkeeping).
        quants=complete_quant_weights(weights, line.quants(model)),
        default_quant=line.default_quant[model],
        pipeline_config=pipeline_config,
    )


def gemma4_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """Build one gemma4 model's plan from the parent of `--series` (the CLI's dispatch target).

    The regular line reads hand-placed checkpoints only ({@link GEMMA4_REGULAR_LINE} has no pins),
    so there is no snapshot to check here; the q4_0 QAT line's entry
    ({@link gemma4_qat_q4_0_dist_plan}) checks its pinned snapshots first.
    """
    return gemma4_plan(gemma4_sources(series_dir, model), model)


#: The modification notice (Apache 2.0 §4(b)). It lists **the changes this repository made to the
#: upstream weights**.
#:
#: MUST: the text matches the distribution's content — it is valid prose as a value, so it passes
#: `verify_dist` and the manifest checks, and a mismatch would only be noticed after distribution.
#:
#: MUST: **on the day a model is added to `GEMMA4_UPSTREAM`, a line naming it is added here too** —
#: the §4(b) notice must state "the modifications of every redistributed upstream", so a text left
#: unchanged while the models grow lacks the notice for one model. This field is
#: `Pipeline.root_files` (the core's type is `Mapping[str, str]`) and cannot see the manifest being
#: assembled — the prose cannot be derived by machine, so a check covers it instead
#: (`tests/test_distribution.py::TestGemma4LegalText` checks it against both the attribution table
#: and the manifest, and turns red the moment a second entry comes in).
GEMMA4_NOTICE_MARKDOWN = """# NOTICE

This repository redistributes modified forms of `google/gemma-4-E2B-it` and
`google/gemma-4-E4B-it`, which are licensed under the Apache License, Version 2.0 (see
`LICENSE.md`). The models actually included are identified in `README.md` and `karume.json`.
The following changes were made:

- The **text decoder only** was extracted; the vision and audio towers were never read.
- The graph was re-expressed in the Karume container format (a `.krm` part sequence whose
  first part carries the graph and model descriptors) in a states form suited to chunked prefill
  and decode.
- **The weights keep the checkpoints' BF16 values unchanged**, the per-layer embedding tables
  included (the parameters stored in float32, the norms among others, hold the same values
  widened).
- The exit was narrowed to the selected rows' logits and final hidden states, and the
  **per-layer embedding tables were moved out of the graph** into container assets that the host
  gathers.
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.

For the `e2b` model only, this repository also redistributes a modified form of
`google/gemma-4-E2B-it-assistant` (the multi-token-prediction drafter head), which is licensed
under the same Apache License, Version 2.0.
Its graph was re-expressed in the Karume container format as well, with these further changes: the
drafter's clustered sparse output head was replaced by a dense projection over the full
vocabulary, and the drafter reads the key/value states and the embedding table of the main model
instead of carrying its own. **The drafter's weights were quantized to int8 throughout**, linear
layers included; its values are therefore not bit-identical to the source checkpoint.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""


def gemma4_root_files() -> dict[str, str]:
    """The legal texts placed at the distribution repository's root
    (`karume.dist.Pipeline.root_files`).

    Upstream is Apache License 2.0 (the license policy at the top of `gemma4/README.md` — the
    ruling of 2026-09-01); §4(a) requires "giving recipients of derivative works a copy of the
    license", and §4(b) "causing modified files to carry prominent notices stating that you changed
    them". The license text is the original shared across families
    ({@link _shared.licenses.apache_license_2_0}), read **verbatim** — reformatting or replacing it
    would make it no longer a copy. Upstream has no `NOTICE` file (§4(d) does not apply), so
    `NOTICE.md` holds the §4(b) modification notice only.
    """
    return {
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": GEMMA4_NOTICE_MARKDOWN,
    }


#: The one `--pipeline gemma4` entry (the driver merges it into the core's PIPELINES).
PIPELINE = Pipeline(
    default_model=GEMMA4_DEFAULT_MODEL,
    repo_name=gemma4_repo_name,
    plan=gemma4_dist_plan,
    # Attribution is **determined by the model name alone** (`GEMMA4_UPSTREAM`), so it is not an
    # axis to choose.
    card_profiles={"gemma4": render_gemma4_model_card},
    # The upstream license's redistribution terms (§4) apply to the one distribution repository, so
    # they are read once here, however many times the assembly runs.
    root_files=gemma4_root_files(),
)


# ---------------------------------------------------------------------------
# The q4_0 QAT line (`karume-gemma4-qat` — module docstring)
# ---------------------------------------------------------------------------

#: The q4_0 QAT line's distribution repository (one repository for its three models, like the
#: regular line's {@link GEMMA4_REPO_NAME}).
GEMMA4_QAT_Q4_0_REPO_NAME = "karume-gemma4-qat"

#: The line's one dtype label (every role stores int4 — the quant seat's name too). Its own word:
#: the regular line's labels ({@link GEMMA4_DTYPE} / {@link GEMMA4_DRAFTER_DTYPE}) follow that
#: line's storage, not this one's.
GEMMA4_QAT_Q4_0_DTYPE = "i4"

#: The q4_0 grid's group (`gemma4.export_product_qat.GROUP_SIZE` — the export's
#: `karume.quantize.DEFAULT_GROUP_SIZE`; a torch-free copy, held equal by the tests): the PLE
#: index's `groupSize`.
GEMMA4_QAT_PLE_GROUP_SIZE = 32

#: The line's output paths (model subtree) — the storage in the file name, as for the regular line.
GEMMA4_QAT_Q4_0_OUTPUT_PATHS: Mapping[str, str] = {
    GEMMA4_ROLE: f"{GEMMA4_ROLE}/model.{GEMMA4_QAT_Q4_0_DTYPE}.krm",
    GEMMA4_DRAFTER_ROLE: f"{GEMMA4_DRAFTER_ROLE}/model.{GEMMA4_QAT_Q4_0_DTYPE}.krm",
    GEMMA4_VISION_ROLE: f"{GEMMA4_VISION_ROLE}/model.{GEMMA4_VISION_STORAGE}.krm",
    GEMMA4_TOKENIZER_ROLE: f"{GEMMA4_TOKENIZER_SUFFIX}/{GEMMA4_TOKENIZER_FILE}",
}

#: The storage gates of every model of the line: each role holds int4 and none of the other
#: compressed storages the writer can write. MUST: int8 is forbidden in the drafter too — the
#: regular drafter is int8 throughout, and a regular drafter series placed here has the right
#: graph and manifest (it borrows a table of the same name), so only the storage tells it apart;
#: an int8 embedding is caught the same way on the `model` role, and the regular line's product by
#: its bf16. The one exception is the `vision` role: 12B's embedder is not on the q4_0 grid, so it
#: holds the checkpoint's bf16 values, and no rounded storage. MUST: for that role only. The
#: `model` role may also hold `i4f16` (26B's expert weights — ADR 0145 decision 3); no other role
#: reads f16 steps.
GEMMA4_QAT_Q4_0_STORAGE = Gemma4StorageTables(
    required={
        GEMMA4_ROLE: GEMMA4_QAT_Q4_0_DTYPE,
        GEMMA4_DRAFTER_ROLE: GEMMA4_QAT_Q4_0_DTYPE,
        GEMMA4_VISION_ROLE: GEMMA4_VISION_STORAGE,
    },
    forbidden={
        GEMMA4_ROLE: ("f16", "bf16", "i8", "i2"),
        GEMMA4_DRAFTER_ROLE: ("f16", "bf16", "i8", "i4f16", "i2"),
        GEMMA4_VISION_ROLE: ("f16", "i8", "i4", "i4f16", "i2"),
    },
)

#: The models of the line that ship the vision embedder: 12B, whose embedder is encoder-free.
#: E2B / E4B's tower is not distributed.
GEMMA4_QAT_Q4_0_VISION_MODELS: frozenset[str] = frozenset({"12b"})


def gemma4_qat_q4_0_has_drafter(model: str) -> bool:
    """Whether the model ships its QAT drafter (`gemma4.card.GEMMA4_QAT_DRAFTER_UPSTREAM`): every
    model of the line but 26B, whose drafter is later work (ADR 0145 decision 1) — without the
    role a load that asks for `speculative` fails at weights resolution."""
    return model in GEMMA4_QAT_DRAFTER_UPSTREAM


def gemma4_qat_q4_0_has_vision(model: str) -> bool:
    """Whether the model ships the vision embedder ({@link GEMMA4_QAT_Q4_0_VISION_MODELS})."""
    return model in GEMMA4_QAT_Q4_0_VISION_MODELS


def gemma4_qat_q4_0_weights(model: str) -> Mapping[str, Mapping[str, WeightFiles]]:
    """The weights declaration: `model` and, when it ships, `drafter`, both int4; and, when it
    ships, `vision`, bf16. One label per role, so {@link complete_quant_weights} maps each in every
    quant of the model (`vision: "bf16"` in `i4`, `i4-gemvpar` and `i4-fast`).

    MUST: `vision` is a weights role, not an asset, for the reason of {@link GEMMA4_WEIGHTS}'s
    MUST: a container with a graph, and the only axis on which `ResolveOptions.weights` can say "not
    fetched by default".
    """
    roles = (
        (GEMMA4_ROLE, GEMMA4_DRAFTER_ROLE) if gemma4_qat_q4_0_has_drafter(model) else (GEMMA4_ROLE,)
    )
    weights = {role: {GEMMA4_QAT_Q4_0_DTYPE: WeightFiles(role)} for role in roles}
    if gemma4_qat_q4_0_has_vision(model):
        weights[GEMMA4_VISION_ROLE] = {GEMMA4_VISION_STORAGE: WeightFiles(GEMMA4_VISION_ROLE)}
    return weights


#: The line's reference seat, every model's default: the reference summation order (`session`
#: empty — ADR 0104) on the checkpoints' own grid.
GEMMA4_QAT_Q4_0_QUANTS: Mapping[str, Any] = {
    GEMMA4_QAT_Q4_0_DTYPE: {
        "weights": {},
        "session": {},
        "label": "QAT int4 grid, drafter included",
        "description": "The q4_0 QAT checkpoints' own 4-bit grid (packed int4, group 32) for every"
        " linear weight, embedding table and drafter weight; GEMV and attention use the reference"
        " summation order.",
    },
}


#: The sentence the reference seat of a model with a vision embedder adds: its description says
#: "every linear weight", and the embedder's are not int4 ({@link GEMMA4_QAT_Q4_0_STORAGE}).
GEMMA4_QAT_Q4_0_VISION_NOTE = " Vision stays bf16."

#: The reference seat's label and description for a model with routed experts and no drafter
#: (26B — `gemma4.card.GEMMA4_MOE_MODELS`): the line's own say "drafter included", and its expert
#: weights keep the grid's codes with f16 steps (`int4-sym-g-f16` — ADR 0145 decision 3).
GEMMA4_QAT_Q4_0_MOE_QUANT: Mapping[str, str] = {
    "label": "QAT int4 grid",
    "description": "The q4_0 QAT checkpoint's own 4-bit grid (packed int4, group 32) for every"
    " linear weight and the embedding table, experts with f16 steps; GEMV and attention use the"
    " reference summation order.",
}


#: The line's parallel and fused seats over its one int4 storage (ADR 0098 / 0104): `i4-gemvpar`
#: sums the GEMV and the attention in parallel, `i4-fast` adds the RMS→add fusion and the fused
#: attention. What applies (the plan of the mirror's graphs — `linearGemvParallelEligible` /
#: `planFusions`, 2026-11-04): every linear of the E2B / E4B product but the int4 output head
#: (`[262144, H]` — the parallel GEMV's table has no int4 row of that shape) is on the parallel
#: GEMV's table (the int4 group-32 rows at the same shapes: E2B's measured ones and E4B's
#: unmeasured copies of them — ADR 0098 addendum 2026-09-26), the RMS→add fusion takes 106 / 127
#: pairs, and the drafter has two eligible linears (of 68, as the regular int8 drafter). Every
#: linear of the 12B product, the output head included, is on the table with rows measured on 12B,
#: and the fusion takes its width (ADR 0098 addendum 2026-11-06). The same holds for every dense
#: linear of the 26B product, its int4 output head (`[262144, 2816]`) included, and the fusion takes
#: width 2816; its routed experts (`expert_linear`) keep the sequential order in every seat — only
#: the dense linears, the attention and the RMS→add pairs change.
GEMMA4_QAT_Q4_0_PARALLEL_QUANTS: Mapping[str, Any] = {
    "i4-gemvpar": {
        "weights": {},
        "session": {"linearGemvReduce": "parallel", "stateAttentionReduce": "parallel"},
        "label": "QAT int4 grid with parallel GEMV and attention",
        "description": "Same weights as i4; parallel GEMV{head} and attention."
        " Rounding and generated tokens can differ. Select i4 for the reference order.",
    },
    "i4-fast": {
        "weights": {},
        "session": {
            "linearGemvReduce": "parallel",
            "fuseRmsNormAdd": True,
            "stateAttentionReduce": "parallel-fused",
        },
        "label": "QAT int4 grid with parallel GEMV, RMS fusion and fused attention",
        "description": "Same weights as i4; parallel GEMV{head}, RMS-add"
        " fusion, fused attention. Use i4 for reference summation. Requires fusion-option support.",
    },
}

#: Model → the one sentence on speed its parallel and fused seats carry; the models with those
#: seats. MUST: a model enters only with a measurement behind the sentence — decode ms/token of
#: `i4` / `i4-gemvpar` / `i4-fast`, with and without the drafter, interleaved in one process
#: (2026-11-04, RTX 3080 Ti, Node + Dawn — `outputs/diag/gemma4-lines-2026-11-04/legA/`): both seats
#: were faster than `i4` on every prompt and arm measured (two runs) — without the drafter E2B by
#: 2.0–2.5× and E4B by 2.1–2.9×, with it both by 1.2–1.5×. 12B (2026-11-06,
#: `outputs/diag/gemma4-12b-fast-2026-11-06/`, the same protocol): without the drafter both seats by
#: 1.9–2.1×; with it both are slower than `i4` with it, so its sentence says so. 26B (2026-11-10,
#: `outputs/diag/forward-cost-2026-11-10/`; 12 GB cannot hold it resident, so decode ran under the
#: exact expert cache — 64 slots per layer, every expert in the host tier, re-run after a miss —
#: one process per arm, greedy, three prompts of 96 tokens with the same tokens in every arm, two
#: rounds): one decode forward without a miss (median wall) took 24.6 / 25.7 ms with `i4`,
#: 17.7 / 17.9 ms with `i4-gemvpar` and 17.6 / 17.5 ms with `i4-fast`. It has no drafter, so there
#: is no speculative arm.
GEMMA4_QAT_Q4_0_SPEED_NOTE: Mapping[str, str] = {
    "e2b": " Faster than i4 on the tested E2B device.",
    "e4b": " Faster than i4 on the tested E4B device.",
    "12b": " Faster than i4 on the tested 12B device, except when speculative.",
    "26b": " Faster than i4 on the tested 26B device.",
}

#: How the seats' descriptions spell the output head: the int4 head's parallel GEMV row exists for
#: 12B and 26B only ({@link GEMMA4_QAT_Q4_0_PARALLEL_QUANTS}).
GEMMA4_QAT_Q4_0_HEAD_NOTE: Mapping[str, str] = {
    "e2b": " (not the int4 output head)",
    "e4b": " (not the int4 output head)",
    "12b": "",
    "26b": "",
}


def gemma4_qat_q4_0_quants(model: str) -> dict[str, Any]:
    """The line's quant table for `model`: the reference seat `i4` ({@link GEMMA4_QAT_Q4_0_QUANTS})
    and, for a model with a measured speed note ({@link GEMMA4_QAT_Q4_0_SPEED_NOTE}), the parallel
    and fused seats ({@link GEMMA4_QAT_Q4_0_PARALLEL_QUANTS}) with that note."""
    quants = {name: dict(quant) for name, quant in GEMMA4_QAT_Q4_0_QUANTS.items()}
    if model in GEMMA4_MOE_MODELS:
        quants[GEMMA4_QAT_Q4_0_DTYPE].update(GEMMA4_QAT_Q4_0_MOE_QUANT)
    if gemma4_qat_q4_0_has_vision(model):
        quants[GEMMA4_QAT_Q4_0_DTYPE]["description"] += GEMMA4_QAT_Q4_0_VISION_NOTE
    note = GEMMA4_QAT_Q4_0_SPEED_NOTE.get(model)
    if note is not None:
        for name, quant in GEMMA4_QAT_Q4_0_PARALLEL_QUANTS.items():
            quants[name] = {
                **quant,
                "session": dict(quant["session"]),
                "description": quant["description"].format(head=GEMMA4_QAT_Q4_0_HEAD_NOTE[model])
                + note,
            }
    return quants


#: The line's models and their default quant (the reference seat `i4`). Key set =
#: `gemma4.card.GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM` (tests).
GEMMA4_QAT_Q4_0_DEFAULT_QUANT: Mapping[str, str] = dict.fromkeys(
    GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM, GEMMA4_QAT_Q4_0_DTYPE
)

#: The q4_0 QAT line (`karume-gemma4-qat` — {@link QAT_Q4_0_PIPELINE}).
GEMMA4_QAT_Q4_0_LINE = Gemma4Line(
    models=tuple(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM),
    has_drafter=gemma4_qat_q4_0_has_drafter,
    has_vision=gemma4_qat_q4_0_has_vision,
    storage=GEMMA4_QAT_Q4_0_STORAGE,
    ple=Gemma4PleForm(storage=GEMMA4_QAT_Q4_0_DTYPE, group_size=GEMMA4_QAT_PLE_GROUP_SIZE),
    output_paths=GEMMA4_QAT_Q4_0_OUTPUT_PATHS,
    weights=gemma4_qat_q4_0_weights,
    quants=gemma4_qat_q4_0_quants,
    default_quant=GEMMA4_QAT_Q4_0_DEFAULT_QUANT,
    pins=QAT_LINE_PINS,
    drafter_pins=QAT_DRAFTER_PINS,
)


def gemma4_qat_q4_0_product_suffix(model: str) -> str:
    """The suffix of model `model`'s product series in the line: E2B / E4B's QAT series (a line
    separate from the regular one — `gemma4.sources.QAT_PINS`), 12B's product series (its one
    line, already the QAT checkpoint). `gemma4.export_drafter_qat.lender_suffix` reads the lender
    by this rule."""
    return GEMMA4_QAT_PRODUCT_SUFFIX if model in QAT_PINS else GEMMA4_PRODUCT_SUFFIX


def gemma4_qat_q4_0_repo_name(_model: str) -> str:
    """The line's one repository ({@link GEMMA4_QAT_Q4_0_REPO_NAME})."""
    return GEMMA4_QAT_Q4_0_REPO_NAME


def gemma4_qat_q4_0_sources(series_dir: Path, model: str = GEMMA4_DEFAULT_MODEL) -> Gemma4Sources:
    """The inputs of model `model` of the line: its product series, its QAT drafter series and its
    vision series when it ships them, the regular tokenizer series
    ({@link GEMMA4_TOKENIZER_SERIES} — the line reads the regular `-it` tokenizers), and the hub
    snapshots of its pins (`gemma4.sources.QAT_LINE_PINS`).

    A model outside the line fails loudly.
    """
    if model not in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM:
        raise DistError(
            f"the q4_0 QAT line has no model '{model}'"
            f" (known: {' / '.join(sorted(GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM))})"
        )
    weights, tokenizer = qat_line_dirs(model)
    return Gemma4Sources(
        product=series_dir / gemma4_series_name(model, gemma4_qat_q4_0_product_suffix(model)),
        drafter=(
            series_dir / gemma4_series_name(model, GEMMA4_QAT_DRAFTER_SUFFIX)
            if gemma4_qat_q4_0_has_drafter(model)
            else None
        ),
        tokenizer=series_dir
        / gemma4_series_name(GEMMA4_TOKENIZER_SERIES[model], GEMMA4_TOKENIZER_SUFFIX),
        model=weights,
        upstream_tokenizer=tokenizer,
        vision=(
            series_dir / gemma4_series_name(model, GEMMA4_VISION_SUFFIX)
            if gemma4_qat_q4_0_has_vision(model)
            else None
        ),
    )


def gemma4_qat_q4_0_plan(sources: Gemma4Sources, model: str = GEMMA4_DEFAULT_MODEL) -> ModelPlan:
    """One model's plan of the line ({@link gemma4_plan} with {@link GEMMA4_QAT_Q4_0_LINE})."""
    return gemma4_plan(sources, model, GEMMA4_QAT_Q4_0_LINE)


def gemma4_qat_q4_0_dist_plan(series_dir: Path, model: str) -> ModelPlan:
    """The CLI's entry for the line: both target snapshots are checked against their pins first
    ({@link gemma4_dist_plan}'s rule), then the plan is built."""
    sources = gemma4_qat_q4_0_sources(series_dir, model)
    try:
        assert_model_pins(sources.model, sources.upstream_tokenizer, QAT_LINE_PINS[model])
    except ValueError as error:
        raise DistError(str(error)) from error
    return gemma4_qat_q4_0_plan(sources, model)


def _qat_q4_0_notice_sources() -> str:
    """The NOTICE's list of redistributed upstreams: per model its QAT checkpoint, its QAT drafter
    when it ships one (26B ships none — {@link gemma4_qat_q4_0_has_drafter}) and its base model
    (the tokenizer's source)."""
    lines: list[str] = []
    for model in GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM:
        drafter = GEMMA4_QAT_DRAFTER_UPSTREAM.get(model)
        lines.append(
            f"- `{GEMMA4_QAT_Q4_0_WEIGHTS_UPSTREAM[model]}`"
            + ("" if drafter is None else f", its drafter `{drafter}`")
            + f" and the tokenizer of its base model `{GEMMA4_UPSTREAM[model]}`"
        )
    return "\n".join(lines)


#: The line's modification notice (Apache 2.0 §4(b)). MUST: the text matches the distribution's
#: content ({@link GEMMA4_NOTICE_MARKDOWN}'s MUST — valid prose passes every machine check), and
#: it names every upstream of the line's attribution tables (built from them — tests).
GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN = (
    """# NOTICE

This repository redistributes modified forms of Google's Gemma 4 q4_0 quantization-aware-trained
(QAT) checkpoints, of their QAT multi-token-prediction drafters and of the tokenizers of their base
models, listed below, which are licensed under the Apache License, Version 2.0 (see `LICENSE.md`).
The models actually included are identified in `README.md` and `karume.json`.

"""
    + _qat_q4_0_notice_sources()
    + """

The following changes were made:

- For the E2B and E4B checkpoints, the **text decoder only** was extracted; their vision and audio
  towers were never read.
- For the 12B checkpoint, the **text decoder and the encoder-free vision embedder** were extracted,
  each as its own graph; its audio embedder was never read. The vision embedder's position table
  was split into its two columns, and an image's rows reach the text decoder's graph as two inputs.
- For the 26B A4B checkpoint, the **text decoder only** was extracted; its vision tower was never
  read. Its routed experts were re-expressed as one projection per routed expert, combined with the
  routing weights in the router's order of picks.
- The graphs were re-expressed in the Karume container format (a `.krm` part sequence whose first
  part carries the graph and model descriptors) in a states form suited to chunked prefill and
  decode.
- **The weights keep the checkpoints' own 4-bit grid**: the linear weights and the embedding table
  of each model, the per-layer embedding tables of the E2B and E4B checkpoints and the drafters'
  weights are stored as packed int4 (group 32: the codes of the grid and one step per group,
  refitted from the stored values) instead of being rounded again. The values are therefore not
  bit-identical to the source checkpoints. The expert weights of the 26B A4B checkpoint keep the
  same codes with each step stored as float16 instead of float32 (the refitted step rounded once).
  The parameters off the grid (the norms, among others) are stored in float32. The 12B vision
  embedder is not on that grid and keeps the checkpoint's BF16 values (the parameters stored in
  float32 hold the same values widened).
- The exit was narrowed to the selected rows' logits and final hidden states. For the E2B and E4B
  checkpoints, the **per-layer embedding tables were moved out of the graph** into container
  assets that the host gathers (the 12B checkpoint has none).
- **Rotary position embeddings were moved out of the graph**: the cosine and sine rows are built
  by the host from the declared parameters and passed in as ordinary graph inputs.
- Each drafter's output head is the dense projection over the full vocabulary (a clustered sparse
  head an assistant checkpoint carries is not used), and each drafter reads the key/value states
  and the embedding table of its main model instead of carrying its own.

No retraining and no fine-tuning were performed. The original checkpoints are not distributed here.
"""
)


def gemma4_qat_q4_0_root_files() -> dict[str, str]:
    """The line's legal texts at the repository root ({@link gemma4_root_files}'s rule: the
    licence verbatim and the line's modification notice)."""
    return {
        "LICENSE.md": apache_license_2_0(),
        "NOTICE.md": GEMMA4_QAT_Q4_0_NOTICE_MARKDOWN,
    }


#: `--pipeline gemma4-qat` (the driver adds it to the core's PIPELINES).
QAT_Q4_0_PIPELINE = Pipeline(
    default_model=GEMMA4_DEFAULT_MODEL,
    repo_name=gemma4_qat_q4_0_repo_name,
    plan=gemma4_qat_q4_0_dist_plan,
    card_profiles={"gemma4-qat": render_gemma4_qat_q4_0_model_card},
    root_files=gemma4_qat_q4_0_root_files(),
)
