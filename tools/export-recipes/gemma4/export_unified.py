"""Writes Gemma 4 12B (the `gemma4_unified` class) as the product graph, with its greedy goldens.

    cd tools/export-recipes
    uv run --with 'transformers==5.14.1' python -m gemma4.export_unified --model 12b
    uv run --with 'transformers==5.14.1' python -m gemma4.export_unified --model 12b --plan
    uv run --with 'transformers==5.14.1' python -m gemma4.export_unified --model 12b \\
        --layers 6 --out ../../outputs/diag/<dir>          # a truncated check, no goldens

The weights and their config are the pinned snapshot of Google's QAT checkpoint, and the tokenizer
and the generation defaults the pinned snapshot of `google/gemma-4-12B-it` (`gemma4.sources` — two
sources, each checked against its pin before anything long runs). Every run prints per-stage wall
time and peak RSS as JSON and runs under a `MemAvailable` floor
({@link qwen3_5.export.MemoryWatch}): below it the process exits instead of pushing the shared
machine into swap.

## The graph: E2B / E4B's product contract minus PLE

Inputs `input_ids[1,M]`, the four RoPE inputs (`gemma4.export_decode.ROPE_INPUTS`) and
`last_row[R]`; outputs `logits[1,R,V]` then `hidden[1,R,H]`, in that order
({@link UnifiedProductWrapper} — `gemma4.export_product.ProductChunkWrapper` without the
`per_layer_inputs` input). The class has no per-layer embeddings (`hidden_size_per_layer_input`
is 0), so there is no PLE input, table or asset; the checks that look for one are inverted (the
checkpoint carries no `per_layer` tensor, the graph no `per_layer_inputs`, the container no PLE
asset).

The states: no KV sharing (`num_kv_shared_layers` 0), so every layer owns a k / v slot pair —
96 slots for 48 layers (`gemma4.export_decode.states_plan` unchanged). The KV heads differ per
layer type (`gemma4.export_decode.kv_heads`): sliding slots hold 8 heads in `window + 8` = 1032
rows, full slots 1 head of width 512 in the capacity symbol `C`. On the full layers upstream's
`attention_k_eq_v` has no `v_proj`: the value is the **raw** `k_proj` output through the
scale-free `v_norm`, not rotated, while the key is `k_norm` then RoPE of the same projection — one
linear, two different slots ({@link assert_shared_kv_projection} checks the wiring in the
written graph).

## Why not the E2B / E4B path, and the form used instead

That path loads the whole checkpoint as F32 (`gemma4.export.load_model_and_tables`): 10.9 B linear
+ 1.0 B embedding parameters are 44.6 GiB, more than a 64 GB machine shared with other work can
hold. This script takes the Qwen3.5 / umT5 form (`qwen3_5/export.py` — ruling R5):

1. **The trace runs without weight data** ({@link shape_only_model}). Built on the meta device,
   then every parameter gets an uninitialized CPU tensor of its shape. Why not meta throughout:
   the attention masks are `arange`-only constants the converter folds into Tmax tables (ADR
   0010), which needs real CPU values, and a CPU constant cannot meet a meta activation.
   `torch.export` never reads parameter data, so those pages are never touched (a 3.6 GB check
   stayed at 0.5 GB RSS — measured 2026-11-02); the memory floor is the guard if that ever
   changes. The graph is the real-weight trace's (pytest compares the JSON).
2. **The linear weights and the tied embedding (packed i4, group 32) are imported from the
   checkpoint's own 4-bit grid, a block of rows at a time** ({@link quantize_int4_rows} — the next
   section), and handed to the writer as `fixed_weights` (ADR 0097's entry). The import closes per
   row, so the blocks give the whole-tensor bits; pytest checks the container bytes against an
   import of each whole tensor at once.
3. **The norms and `layer_scalar`** are the checkpoint's BF16 values widened to F32.

## The linears and the embedding: the QAT grid, imported as is

Every text linear of the QAT checkpoint already lies on a Q4_0 grid: in each group of 32 along the
row (the in-features axis, `karume.quantize.group_scale`'s grouping) the values are `bf16(q·d)`,
with integer codes `q` in [−8, 7] and a learned step `d` per group (not `amax / 8`). The tied
embedding does too, in groups of 32 along its rows (the hidden axis): every one of its groups
recovers under the rule below (measured 2026-11-03), so it is imported the same way instead of
being rounded again to i8 per row, which moved values that were already exact. Measured on the
pinned snapshots (2026-11): round-to-nearest i4 g32 on karume's grid (codes ±7, scale = group
`amax / 7`) of `google/gemma-4-12B-it` gives KL 2.65 / 1.88 (context-en / window-en) against the
unquantized forward and a top-1 agreement of 0.39 / 0.47; re-rounding the QAT weights to the ±7
grid still gives KL ≈ 1.0; importing the grid of the linears (the embedding rounded to i8 per row)
gives KL 0.0136 / 0.0362 and a top-1 agreement of 0.97 / 0.93 against the QAT checkpoint's own
unquantized forward, and importing the embedding's grid too gives KL 0.0091 / 0.0505 and a top-1
agreement of 0.99 / 0.93 (2026-11-03). The embedding's own relative RMS error against its BF16
values drops from 1.41 % (i8 per row) to 0.14 %, lower in every row; why `window-en`'s mean KL
still rises is not yet explained.

{@link grid_import} recovers it per group: the largest step `amax / k` (k = 1..8) whose codes
`round(w / step)` all lie in [−8, 7] and whose least-squares refit `Σq·w / Σq²` reconstructs the
group within `2^-7 · amax` (max abs error); the codes and the refit step (F32) are stored. An
all-zero group gets codes 0 and step 1. MUST: a group with no such step fails the export with the
key, the row range and the group — never a silent fallback to rounding. The storage holds −8
without a format change: `karume.emit.pack_int4` stores `q + 8` (−8 is nibble 0) and the runtime
decodes `u − 8`.

## Goldens (ruling R4)

The torch reference is F32 compute over **resident i4 values dequantized at use**
({@link reference_wrapper}): each linear holds its imported codes (one per int8) and group steps,
taken once from the checkpoint in row blocks ({@link int4_codes} — the same blocks of
{@link grid_import} the container's bytes are packed from), and a parametrization applies
`dequantize_int4` every time the weight is read. That is the value the container's bytes decode
to, so the reference is bit-equal to an F32 load whose linears are replaced by codes · steps
(pytest, weight for weight and output for output, and against the written container's decoded
bytes) while the process holds about 16 GiB instead of 45. The embedding (and the tied head) is
held as its imported grid values in F32 (4 GiB — {@link imported_embedding}): it is read three
times a forward, so it is dequantized once, not at use. Rounding precedes the reference
(ADR 0006) — the goldens describe the model the container stores. Measured 2026-11-02 on the real
12B: building the reference takes 34 s at a 16.3 GiB peak, and one forward 13 s at T = 6, 34 s at
T = 598 and 55 s at T = 1212 (the earlier form re-rounded the BF16 weights on every read, about
2.3 min per greedy step).

The goldens are greedy continuations of {@link UNIFIED_GREEDY_CASES} (the standard cases but
`context-en`, plus {@link WINDOW_CASE}) by full re-forward (`_shared.decode_series._write_greedy`,
the margin floor of `gemma4.export_decode.MARGIN_FLOOR`), written in the decode series' format.
Before any selection the logits of the ids upstream's `generation_config.json` suppresses are −inf
({@link GreedyView} — the manifest declares the same ids as `pipelineConfig.suppressTokens`), so
the tokens and the margins are those of the masked logits; `reference.json` records the ids.
The first token of the cases in {@link FIRST_TOKEN_EXPECTATIONS} is checked. The goldens are
taken **before** the container is written, so a failing margin stops the run early. The window
case's prompt is longer than the sliding ring ({@link assert_window_exercised}), so the goldens
evaluate the band mask past the window.

## Output layout

    outputs/series/gemma4-12b-product/model.krm                 the container (no assets)
    outputs/series/gemma4-12b-product/greedy.<case>.safetensors the greedy goldens
    outputs/series/gemma4-12b-product/reference.json            provenance (both sources, own
                                                                goldens, the suppressed ids)

MUST: transformers is imported inside functions.
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.export import Dim
from torch.nn import functional

from _shared.decode_series import (
    _write_greedy,
    assert_case_room,
    greedy_continuation,
    positions_for,
)
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import provenance, sources
from gemma4.card import GEMMA4_LICENSE, GEMMA4_PLE_MODELS, GEMMA4_UPSTREAM
from gemma4.distribution import (
    GEMMA4_PRODUCT_SUFFIX,
    GEMMA4_ROLE,
    assert_gemma4_suppress_tokens,
    gemma4_generation_defaults,
    gemma4_ple_dim,
)
from gemma4.sources import ModelPins
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION
from karume.dist import NOTICE_FILENAME
from karume.emit import FixedQuantizedWeight, pack_int4, unpack_int4
from karume.ir import IrGraph, IrNode
from karume.ops import ATTENTION_OP, LINEAR_OP
from karume.pipeline import export_module, publish_model
from karume.quantize import DEFAULT_GROUP_SIZE, dequantize_int4, iter_quant_targets
from karume.states import to_states_form
from qwen3_5.export import MEMORY_FLOOR_GIB, MemoryWatch

#: The text model type this script exports (`text_config.model_type` of 12B).
UNIFIED_TEXT_MODEL_TYPE = "gemma4_unified_text"

#: The models this script writes — the ones without per-layer embeddings (12B).
MODELS: tuple[str, ...] = tuple(
    model for model in GEMMA4_UPSTREAM if model not in GEMMA4_PLE_MODELS
)

#: Checkpoint key prefixes outside the text decoder, never read (12B's encoder-free vision and
#: audio embedders — 52.4 M parameters).
IGNORED_PREFIXES: tuple[str, ...] = (
    "model.vision_embedder.",
    "model.embed_vision.",
    "model.embed_audio.",
)

#: The text decoder's key prefix in the wrapper (`UnifiedProductWrapper.model.model`) and in the
#: checkpoint.
WRAPPER_TEXT_PREFIX = "model.model."
CHECKPOINT_TEXT_PREFIX = one_shot.CHECKPOINT_TEXT_PREFIX

#: The tied pair in the wrapper's key space (the checkpoint stores the embedding only; the trace
#: keeps one of the two names for the shared tensor).
EMBED_KEY = f"{one_shot.EMBED_TOKENS_MODULE}.weight"
TIED_HEAD_KEY = f"{one_shot.LM_HEAD_MODULE}.weight"

#: The marker of a per-layer-embedding tensor in a Gemma 4 checkpoint
#: (`embed_tokens_per_layer`, `per_layer_model_projection`, `layers.N.per_layer_*`).
PLE_KEY_MARKER = "per_layer"

#: Checkpoint dtypes (both widen to F32 exactly).
ACCEPTED_DTYPES = frozenset({"BF16", "F32"})

#: F32 bytes read at once while quantizing one weight (`[16384, 3840]` for the embedding).
CHUNK_BYTES = 256 << 20

#: The code range of the QAT checkpoint's 4-bit grid (Q4_0: 16 levels).
GRID_CODES: tuple[int, int] = (-8, 7)

#: The step divisors tried, smallest first (step = group `amax / k`): the largest |code| a group
#: may reach is 8 (a −8 at the group's max |w|).
GRID_DIVISORS = range(1, 9)

#: The reconstruction bound of an imported group, relative to its max |w| (max abs error of
#: codes · step): above BF16's rounding of the checkpoint's `q·d` (2^-9 of the group max at most),
#: far below one grid step (at least 1/8 of the group max).
GRID_TOLERANCE = 2.0**-7

#: The trace-made constants' key prefix (`karume.convert` — zero biases, unit norm weights, the
#: folded masks).
CONSTANT_PREFIX = "const."


#: The golden case longer than 12B's sliding ring: its prompt (1159 tokens with `<bos>` under 12B's
#: tokenizer) exceeds `window + SLIDING_SLACK_ROWS` = 1032 rows, so the ring evicts during the
#: prefill, the band mask differs from causal in every full re-forward of the goldens, and the
#: host's prefill spans two chunks of 768 rows. It is longer than the traced chunk range on
#: purpose: the goldens come from the torch reference at full length, never from the graph.
#: Neutral factual text (the four river basins of France) ending in a question whose answer is one
#: token (`▁Seine`). Measured 2026-11-02 on this text with `google/gemma-4-12B-it`'s weights (before
#: 12B moved to the QAT checkpoint): upstream (BF16) gives `▁Seine` with a margin of 7.9 and the
#: round-to-nearest i4 reference 10.1, while a cloze ending ("The river that flows through
#: Paris is called the") left upstream undecided (`▁boat` 7.9, `▁Seine` 7.8).
WINDOW_CASE = "window-en"
WINDOW_EN = (
    "The rivers of France drain four large basins, and most of the country's history can be read "
    "along their banks. Each basin has its own climate, its own soils and its own way of moving "
    "goods, and for most of recorded history the river was the cheapest road a farmer, a merchant "
    "or an army could find. A barge could carry in a day what a cart could carry in a month, so "
    "the towns that mattered were the towns that stood where a river could be crossed, where two "
    "rivers met, or where a river reached the sea. "
    "The Loire is the longest river that lies entirely within the country. It rises on the "
    "eastern edge of the Massif Central, at the foot of a volcanic peak called the Gerbier de "
    "Jonc, and runs north and then west for about one thousand kilometres before it reaches the "
    "Atlantic below the port of Saint-Nazaire. Its flow is irregular: in late summer long "
    "stretches become sandbanks with a narrow channel between them, and after autumn rain the "
    "same stretches can flood the fields on both sides. Because of this the river was never easy "
    "to navigate, and the boatmen who worked on it used flat-bottomed boats with a single large "
    "sail that could be lowered to pass under the bridges. The middle valley, between Orleans and "
    "Angers, is known for its castles, its vineyards and its soft white stone, which was cut from "
    "the cliffs along the river and carried downstream to build churches and houses. "
    "The Rhone is the most powerful of the four. It begins in a glacier in the Swiss Alps, crosses "
    "Lake Geneva, enters France below Geneva and turns south at Lyon, where it is joined by the "
    "Saone. From Lyon it runs almost straight to the Mediterranean, through a narrow valley "
    "between the Alps and the Massif Central, and ends in a wide delta called the Camargue, a "
    "flat country of salt marshes, rice fields and lagoons. The valley has long been the main "
    "route between northern Europe and the Mediterranean, and today a motorway, a railway and a "
    "series of dams with hydroelectric plants share it with the river. The strong north wind "
    "that blows down the valley, the mistral, is one of its best known features. "
    "The Garonne drains the south west. It rises in the Pyrenees, on the Spanish side of the "
    "border, flows north past Toulouse and then north west to Bordeaux, and below Bordeaux it "
    "meets the Dordogne. The two rivers then form the Gironde, a long estuary that opens to the "
    "Atlantic. The tide reaches far inland along the estuary, which allowed sea-going ships to "
    "load wine at the quays of Bordeaux for many centuries. A canal built in the seventeenth "
    "century links the Garonne at Toulouse to the Mediterranean, so that boats can cross the "
    "south of the country without sailing around the Iberian peninsula. "
    "The Seine is the shortest of the four but the most important for the history of the state. "
    "It rises on a limestone plateau in Burgundy, north west of Dijon, and flows north west for "
    "about seven hundred and eighty kilometres through a wide basin of wheat fields, forests and "
    "pasture. Its slope is gentle and its flow is steady, which made it the easiest of the great "
    "rivers to navigate: barges could travel on it for most of the year, and its tributaries, the "
    "Yonne, the Marne and the Oise, carried timber, grain and stone toward the middle of the "
    "basin. Below the capital the river winds through a series of wide loops past Rouen and "
    "reaches the English Channel between Le Havre and Honfleur. "
    "In the middle of the basin, on an island in the river, a Gaulish people called the Parisii "
    "kept a settlement that the Romans later fortified and renamed Lutetia. The island was easy "
    "to defend and the crossing was easy to tax, and over the following centuries the town spread "
    "from the island to both banks. The kings of the Capetian dynasty made it the seat of their "
    "household, and the cathedral, the royal palace and the courts of justice were built on the "
    "same island. Bridges joined it to the banks, and the quays along the river became the place "
    "where wood, wine and grain were unloaded for the growing city. Today the island is still the "
    "point from which road distances in the country are measured, and the river is lined with "
    "stone quays, bridges and the long galleries of a former royal palace that now houses a "
    "museum. "
    "Rivers also shaped the borders and the defences of the country. The Rhine forms part of the "
    "eastern frontier, and fortresses were built along it and along the Meuse and the Moselle "
    "further west. In the south, the Rhone was for a long time the boundary between the kingdom "
    "and the lands of the empire, and bridges such as the one at Avignon were both trade routes "
    "and points of control. Canals were dug in the seventeenth, eighteenth and nineteenth "
    "centuries to join the basins to one another: the Briare canal linked the Loire to the Seine, "
    "the Burgundy canal linked the Seine to the Saone, and the canal of the Midi linked the "
    "Garonne to the sea. Together they formed a network of inland waterways that carried coal, "
    "building stone and grain until the railways took most of that traffic in the late nineteenth "
    "century. "
    "Each river has also given its name to an administrative district. When the provinces of the "
    "old kingdom were replaced by departments at the end of the eighteenth century, most of the "
    "new departments were named after rivers or mountains rather than after the historic "
    "provinces, so that names such as Loire, Rhone, Gironde and Seine-Maritime appear on the map "
    "today. The choice was deliberate: a river was a fact of geography that everyone could see, "
    "while a province carried the memory of older loyalties. "
    "Question: which river flows through Paris? Answer: the"
)

#: 12B's golden cases: the standard set of the PLE series (`gemma4.export.GOLDEN_CASES`) plus
#: {@link WINDOW_CASE}.
UNIFIED_CASES: tuple[tuple[str, str], ...] = (*one_shot.GOLDEN_CASES, (WINDOW_CASE, WINDOW_EN))

#: The cases whose greedy continuations become goldens: the PLE series' set without `context-en`,
#: plus {@link WINDOW_CASE}. `context-en` is a near tie at its fifth step on the imported QAT
#: reference (margin 0.0031, measured 2026-11-03 — below `decode.MARGIN_FLOOR`), and upstream's own
#: F32 `generate()` on the QAT weights picks the other token there; the floor's rule is to drop the
#: case, not to cut the steps. Its first token stays checked ({@link FIRST_TOKEN_EXPECTATIONS}).
UNIFIED_GREEDY_CASES: tuple[str, ...] = (
    *(case for case in decode.GREEDY_CASES if case != "context-en"),
    WINDOW_CASE,
)

#: The first-token sanity of 12B's goldens (`gemma4.export._sanity`): the cases whose first token
#: upstream decides. Measured 2026-11-02 on `google/gemma-4-12B-it`'s weights (before 12B moved to
#: the QAT checkpoint), top two of the last row:
#:
#: - upstream (`Gemma4UnifiedForCausalLM`, BF16, eager): `capital-en` `▁Paris` 20.1 / `▁a` 19.9;
#:   `context-en` `▁Paris` 24.5 / `▁**` 19.3; `window-en` `▁Seine` 26.5 / `▁river` 18.6.
#: - the export wrapper in F32 over the unquantized weights: `capital-en` `▁a` 20.7 / `▁Paris`
#:   20.1; `context-en` `▁Paris` 24.1 / `▁**` 19.1.
#: - the goldens' round-to-nearest i4 reference: `capital-en` `▁a` 16.0 / `▁capital` 15.6;
#:   `capital-ja` id 237536 (a full-width question mark) 19.7 / id 97243 ("which") 19.6;
#:   `context-en` `▁by` 19.4 / `▁Paris` 18.5; `window-en` `▁Seine` 24.9 / `▁Sein` 14.8.
#:
#: The two short raw-text cases are undecided even upstream, so they stay goldens without an
#: expectation. NOTE: under round-to-nearest i4 `context-en` lost `▁Paris` (a 5.1 margin upstream)
#: — one reason 12B now imports the QAT grid. The imported reference (2026-11-03, suppressed ids
#: masked): `context-en` `▁Paris` (margin 2.9), `window-en` `▁Seine` (6.6), `capital-en` `0`
#: (0.38 — upstream's F32 `generate()` on the QAT weights gives the same 16 tokens), `capital-ja`
#: `東京` (1.6). A case outside {@link UNIFIED_GREEDY_CASES} has its first token taken by one
#: forward. The gate reports what the stored model does, it is not tuned to it.
FIRST_TOKEN_EXPECTATIONS: Mapping[str, str] = {"context-en": " Paris", WINDOW_CASE: " Seine"}


class UnifiedExportError(RuntimeError):
    """The checkpoint, the config or the graph is not what this script expects."""


# ---------------------------------------------------------------------------
# Config, wrapper, trace
# ---------------------------------------------------------------------------


def load_config(model_dir: Path, layers: int | None = None) -> Any:
    """The checkpoint's text config, set to the GQA attention; `layers` keeps the first layers.

    MUST: a truncated config ends on a full-attention layer — upstream forces the last layer to
    full attention when it builds the config, so a prefix ending on a sliding layer is not a
    config upstream would make.
    """
    config = one_shot.load_text_config(model_dir)
    where = str(model_dir / "config.json")
    if config.model_type != UNIFIED_TEXT_MODEL_TYPE:
        raise UnifiedExportError(
            f"{where}: text model type {config.model_type!r}, not {UNIFIED_TEXT_MODEL_TYPE!r}"
            " (E2B / E4B go through gemma4.export_product)"
        )
    if gemma4_ple_dim(config, where) != 0:
        raise UnifiedExportError(f"{where}: declares per-layer embeddings — not this script's form")
    if layers is not None:
        if not 1 <= layers <= int(config.num_hidden_layers):
            raise UnifiedExportError(f"--layers {layers} outside 1..{config.num_hidden_layers}")
        layer_types = list(config.layer_types[:layers])
        if layer_types[-1] != one_shot.FULL_ATTENTION:
            raise UnifiedExportError(
                f"--layers {layers} ends on a {layer_types[-1]} layer (upstream's last layer is"
                " always full attention)"
            )
        config.num_hidden_layers = layers
        config.layer_types = layer_types
    one_shot.register_attention()
    config._attn_implementation = one_shot.ATTENTION_NAME
    return config


class UnifiedProductWrapper(nn.Module):
    """`(input_ids, RoPE ×4, last_row[R]) → (logits[1,R,V], hidden[1,R,H])` of a model without
    per-layer embeddings.

    `gemma4.export_product.ProductChunkWrapper` without `per_layer_inputs`, line for line (the
    two masks of the 1-shot recipe, the host RoPE bound for one forward, the row selection by
    `F.embedding`, the softcapped lm_head) — every MUST of that docstring applies here. MUST:
    the module paths are the E2B / E4B wrapper's (`model.model.*`, `model.lm_head`), so the
    quantization predicate `gemma4.export.is_int4_module` selects the same roles. MUST: the
    outputs stay (logits, hidden).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.sliding_window = int(model.config.sliding_window)

    def forward(
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        last_row: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        tables = decode.bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            hidden = self.model.model(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).last_hidden_state
        rowed = functional.embedding(last_row, hidden[0]).unsqueeze(0)
        logits = self.model.lm_head(rowed)
        cap = float(self.model.config.final_logit_softcapping)
        return torch.tanh(logits / cap) * cap, rowed


def meta_model(config: Any) -> nn.Module:
    """Upstream's `Gemma4UnifiedForCausalLM` without weights (parameters on meta; the derived
    non-persistent buffers — RoPE frequencies, the embedding scale — computed by upstream's own
    initializer: `gemma4.export.materialize_non_persistent_buffers`)."""
    from transformers import Gemma4UnifiedForCausalLM

    with torch.device("meta"):
        model = Gemma4UnifiedForCausalLM(config).eval()
    one_shot.materialize_non_persistent_buffers(model)
    return model


def shape_only_model(config: Any) -> UnifiedProductWrapper:
    """The traceable wrapper whose parameters hold no data (module docstring, step 1).

    Every parameter and persistent buffer still on meta gets an uninitialized CPU tensor of its
    shape (`torch.empty` — the pages stay untouched); a tied parameter stays one object.
    """
    model = meta_model(config)
    replaced: dict[int, nn.Parameter] = {}
    for module in model.modules():
        for name, parameter in list(module._parameters.items()):
            if parameter is None:
                continue
            if id(parameter) not in replaced:
                replaced[id(parameter)] = nn.Parameter(
                    torch.empty(parameter.shape, dtype=parameter.dtype), requires_grad=False
                )
            module._parameters[name] = replaced[id(parameter)]
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.is_meta:
                module._buffers[name] = torch.empty(buffer.shape, dtype=buffer.dtype)
    one_shot.assert_materialized(model)
    decode.swap_rope_inputs(model)
    return UnifiedProductWrapper(model).eval()


def trace(
    wrapper: UnifiedProductWrapper, example_ids: torch.Tensor, sym_max: int
) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """Export with the symbolic `M` / `R` and apply the states surgery (the product series' own
    sequence — `gemma4.export_product.export_series`)."""
    specs = decode.rope_specs(wrapper.model.config)
    rope = decode.rope_args(specs, positions_for(example_ids))
    seq = Dim(decode.SEQ_SYMBOL, min=2, max=sym_max)
    rows = Dim(product.ROW_SYMBOL, min=1, max=product.ROW_SYM_MAX)
    graph, tensors = export_module(
        wrapper,
        (example_ids, *rope, decode.last_rows_for(example_ids, product.EXAMPLE_ROWS)),
        dynamic_shapes=(*({1: seq} for _ in range(1 + len(rope))), {0: rows}),
        symbol_names=(decode.SEQ_SYMBOL, product.ROW_SYMBOL),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    config = wrapper.model.config
    return to_states_form(graph, decode.states_plan(graph, config)), tensors


def int4_targets(wrapper: nn.Module) -> frozenset[str]:
    """The linear weights stored as packed i4 (every linear but the tied head — the E2B / E4B
    predicate `gemma4.export.is_int4_module`)."""
    targets: set[str] = set()
    for fqn, _, axis in iter_quant_targets(wrapper, (nn.Linear,), one_shot.is_int4_module):
        if axis != 0:
            raise UnifiedExportError(f"'{fqn}' has channel axis {axis} (row scales only)")
        targets.add(fqn)
    return frozenset(targets)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def checkpoint_key(state_key: str) -> str:
    """Wrapper key → checkpoint key (`model.model.X` → `model.language_model.X`; the tied head →
    the embedding)."""
    if state_key == TIED_HEAD_KEY:
        return f"{CHECKPOINT_TEXT_PREFIX}embed_tokens.weight"
    if not state_key.startswith(WRAPPER_TEXT_PREFIX):
        raise UnifiedExportError(f"'{state_key}' is outside the text decoder")
    return CHECKPOINT_TEXT_PREFIX + state_key.removeprefix(WRAPPER_TEXT_PREFIX)


class Checkpoint:
    """The single-file checkpoint (`model.safetensors`), read one tensor or one block of rows at a
    time.

    MUST: a key outside the text decoder that is not under {@link IGNORED_PREFIXES} fails loudly —
    reading only what the graph asks for would silently accept a checkpoint of another structure.
    """

    def __init__(self, directory: Path) -> None:
        self.path = directory / one_shot.CHECKPOINT_FILE
        if not self.path.is_file():
            raise UnifiedExportError(f"{self.path} is missing")
        with safe_open(str(self.path), framework="pt") as handle:
            keys = list(handle.keys())
        stray = sorted(
            key
            for key in keys
            if not key.startswith(CHECKPOINT_TEXT_PREFIX) and not key.startswith(IGNORED_PREFIXES)
        )
        if stray:
            raise UnifiedExportError(
                f"{self.path}: {len(stray)} keys outside the text decoder and the ignored"
                f" prefixes: {stray[:4]}"
            )
        self._keys = frozenset(key for key in keys if key.startswith(CHECKPOINT_TEXT_PREFIX))

    def names(self) -> frozenset[str]:
        return self._keys

    def _view(self, handle: Any, key: str) -> Any:
        if key not in self._keys:
            raise UnifiedExportError(f"'{key}' is not in {self.path}")
        view = handle.get_slice(key)
        if view.get_dtype() not in ACCEPTED_DTYPES:
            raise UnifiedExportError(
                f"'{key}' is {view.get_dtype()} (only {sorted(ACCEPTED_DTYPES)})"
            )
        return view

    def shape(self, key: str) -> list[int]:
        with safe_open(str(self.path), framework="pt") as handle:
            return list(self._view(handle, key).get_shape())

    def read(self, key: str, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
        """One whole tensor — widened to F32, or as stored (`dtype=None`); always a copy."""
        with safe_open(str(self.path), framework="pt") as handle:
            value = self._view(handle, key)[:]
        return value.to(value.dtype if dtype is None else dtype, copy=True)

    def read_rows(self, key: str, start: int, stop: int) -> torch.Tensor:
        """Rows `[start, stop)` of the leading axis as F32."""
        with safe_open(str(self.path), framework="pt") as handle:
            return self._view(handle, key)[start:stop].to(torch.float32, copy=True)


def _beyond(key: str, layers: int) -> bool:
    """Whether `key` belongs to a layer a truncated export leaves out."""
    tail = key.removeprefix(f"{CHECKPOINT_TEXT_PREFIX}layers.")
    return tail != key and int(tail.split(".", 1)[0]) >= layers


def assert_checkpoint_keys(
    state_keys: Iterable[str], available: frozenset[str], layers: int
) -> None:
    """The checkpoint's text keys are exactly the model's (except the layers a truncated export
    leaves out) — and none is a per-layer embedding (the PLE check of E2B / E4B, inverted)."""
    ple = sorted(key for key in available if PLE_KEY_MARKER in key)
    if ple:
        raise UnifiedExportError(
            f"the checkpoint carries {len(ple)} per-layer-embedding tensors ({ple[:4]}) while"
            " its config declares none"
        )
    wanted = {checkpoint_key(key) for key in state_keys}
    missing = sorted(wanted - available)
    if missing:
        raise UnifiedExportError(
            f"{len(missing)} model tensors are not in the checkpoint: {missing[:4]}"
        )
    unused = sorted(key for key in available - wanted if not _beyond(key, layers))
    if unused:
        raise UnifiedExportError(
            f"{len(unused)} checkpoint tensors the model does not consume: {unused[:4]}"
        )


# ---------------------------------------------------------------------------
# Quantization (a block of rows at a time)
# ---------------------------------------------------------------------------


def _block_rows(cols: int) -> int:
    return max(1, CHUNK_BYTES // (cols * 4))


def _matrix_shape(checkpoint: Checkpoint, key: str) -> tuple[int, int]:
    shape = checkpoint.shape(key)
    if len(shape) != 2:
        raise UnifiedExportError(f"'{key}' has rank {len(shape)} (row quantization is rank 2 only)")
    return int(shape[0]), int(shape[1])


def grid_import(
    weight: torch.Tensor, group_size: int, where: str, first_row: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """The 4-bit grid `weight` (rank 2, F32) lies on: codes in {@link GRID_CODES} (int8, the
    weight's shape) and one F32 step per group of `group_size` along the row (`[rows, groups]`)
    — the module docstring's "The linears and the embedding" section.

    Per group, the step divisors run smallest first, so the largest step that fits is taken; an
    all-zero group keeps codes 0 and step 1. `where` and `first_row` (the block's first row in the
    tensor) name the failing group. MUST: a group that does not recover fails loudly — a NaN or an
    infinity never fits, so it fails here too.
    """
    rows, cols = weight.shape
    groups = weight.reshape(rows, cols // group_size, group_size)
    amax = groups.abs().amax(dim=-1)
    codes = torch.zeros(groups.shape, dtype=torch.int8)
    steps = torch.ones(amax.shape, dtype=torch.float32)
    # `!= 0`, not `> 0`: a NaN group max stays pending (and fails) instead of passing as zero.
    pending = amax != 0
    low, high = GRID_CODES
    for divisor in GRID_DIVISORS:
        if not bool(pending.any()):
            break
        index = pending.nonzero(as_tuple=True)
        values = groups[index]
        bound = amax[index]
        q = torch.div(values, (bound / divisor).unsqueeze(-1)).round_()
        # The group's max |w| maps to |q| = divisor ≥ 1, so Σq² > 0. The in-place steps keep one
        # block-sized temporary alive at a time (the peak RSS of a block).
        refit = (q * values).sum(dim=-1).div_(q.square().sum(dim=-1))
        error = (q * refit.unsqueeze(-1)).sub_(values).abs_().amax(dim=-1)
        fits = ((q >= low) & (q <= high)).all(dim=-1) & (error <= GRID_TOLERANCE * bound)
        hit = tuple(axis[fits] for axis in index)
        codes[hit] = q[fits].to(torch.int8)
        steps[hit] = refit[fits]
        pending[hit] = False
    if bool(pending.any()):
        row, group = (int(axis) for axis in pending.nonzero()[0])
        raise UnifiedExportError(
            f"{where} rows {first_row}..{first_row + rows}: {int(pending.sum())} groups are not on"
            f" a 16-level grid (codes in [{low}, {high}] with one step per group, reconstructing"
            f" within {GRID_TOLERANCE} of the group's max |w|) — the first is group {group} of"
            f" row {first_row + row}"
        )
    return codes.reshape(rows, cols), steps


def _int4_shape(checkpoint: Checkpoint, key: str, group_size: int) -> tuple[int, int]:
    rows, cols = _matrix_shape(checkpoint, key)
    if cols % group_size or cols % 2:
        raise UnifiedExportError(
            f"'{key}': row length {cols} does not split into groups of {group_size}"
        )
    return rows, cols


def _grid_blocks(
    checkpoint: Checkpoint, key: str, group_size: int
) -> Iterator[tuple[int, int, torch.Tensor, torch.Tensor]]:
    """`(start, stop, codes, steps)` of each block of rows ({@link grid_import} on the F32-widened
    checkpoint rows) — the one source of the container's bytes ({@link quantize_int4_rows}) and
    the reference's values ({@link int4_codes} / {@link imported_embedding}). A group never
    crosses a row, so the blocks give the whole tensor's codes and steps."""
    rows, cols = _int4_shape(checkpoint, key, group_size)
    step = _block_rows(cols)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        weight = checkpoint.read_rows(key, start, stop)
        codes, steps = grid_import(weight, group_size, f"'{key}'", start)
        del weight
        yield start, stop, codes, steps


def quantize_int4_rows(
    checkpoint: Checkpoint, key: str, group_size: int = DEFAULT_GROUP_SIZE
) -> FixedQuantizedWeight:
    """Packed i4 with group steps along each row, from the imported grid of blocks of rows
    ({@link _grid_blocks}).

    The bytes about to be written are checked to decode back to `dequantize_int4(codes, steps)`
    (the writer's own round-trip rule for quantized storage — `karume.emit`, ADR 0069 decision 4
    ③ — on the packed bytes). Two consecutive elements of a row share a byte
    (`karume.emit.pack_int4` — the row length is even), so the packed rows of the blocks are the
    whole tensor's bytes.
    """
    rows, cols = _int4_shape(checkpoint, key, group_size)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8)
    scale = torch.empty((rows, cols // group_size), dtype=torch.float32)
    for start, stop, codes, steps in _grid_blocks(checkpoint, key, group_size):
        block = pack_int4(codes)
        restored = dequantize_int4(unpack_int4(block, codes.shape), steps)
        if not torch.equal(restored, dequantize_int4(codes, steps)):
            raise UnifiedExportError(f"'{key}' rows {start}..{stop}: packed i4 does not round-trip")
        packed[start:stop] = block.reshape(stop - start, cols // 2)
        scale[start:stop] = steps
    return FixedQuantizedWeight(dtype="i4", packed=packed, scale=scale)


# ---------------------------------------------------------------------------
# Material and the container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContainerMaterial:
    """What {@link write_container} hands to `publish_model`: the quantized weights as meta
    tensors (their logical shape) plus their packed bytes, every other tensor as F32."""

    tensors: dict[str, torch.Tensor]
    fixed: dict[str, FixedQuantizedWeight]


@dataclass(frozen=True)
class TraceKeys:
    """The key classes of a traced wrapper (read before the wrapper is dropped)."""

    #: Checkpoint-backed keys (parameters and persistent buffers, both names of the tied pair).
    persistent: frozenset[str]
    #: Values upstream derives from the config (`embed_scale` — real CPU values in the trace).
    derived: frozenset[str]
    #: The linear weights stored as i4.
    int4: frozenset[str]


def trace_keys(wrapper: nn.Module) -> TraceKeys:
    persistent = frozenset(wrapper.state_dict())
    buffers = {name for name, _ in wrapper.named_buffers()}
    return TraceKeys(
        persistent=persistent, derived=frozenset(buffers - persistent), int4=int4_targets(wrapper)
    )


def prepare_material(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    keys: TraceKeys,
    checkpoint: Checkpoint,
) -> ContainerMaterial:
    """Import the grid of the linears and the embedding from the checkpoint, widen the rest.

    MUST: every tensor the graph declares is one of four kinds — a quantized weight, a
    checkpoint tensor, a trace constant or a config-derived buffer; anything else fails loudly
    (an uninitialized trace parameter must never reach the writer).
    """
    declared = sorted({init.tensor for init in graph.initializers.values()})
    embedding = [key for key in declared if key in (EMBED_KEY, TIED_HEAD_KEY)]
    if len(embedding) != 1:
        raise UnifiedExportError(f"the graph declares the tied embedding as {embedding}")
    missing = sorted(keys.int4 - set(declared))
    if missing:
        raise UnifiedExportError(f"quantization targets the graph does not declare: {missing[:4]}")
    material: dict[str, torch.Tensor] = {}
    fixed: dict[str, FixedQuantizedWeight] = {}
    for key in declared:
        shape = list(tensors[key].shape)
        if key in keys.int4 or key in embedding:
            source = checkpoint_key(key)
            # Checked before the long read: a mismatch fails without quantizing the tensor.
            stored = checkpoint.shape(source)
            if stored != shape:
                raise UnifiedExportError(f"'{key}': checkpoint shape {stored}, graph {shape}")
            fixed[key] = quantize_int4_rows(checkpoint, source)
            material[key] = torch.empty(shape, device="meta")
        elif key in keys.persistent:
            value = checkpoint.read(checkpoint_key(key))
            if list(value.shape) != shape:
                raise UnifiedExportError(
                    f"'{key}': checkpoint shape {list(value.shape)}, graph {shape}"
                )
            material[key] = value
        elif key.startswith(CONSTANT_PREFIX) or key in keys.derived:
            material[key] = tensors[key]
        else:
            raise UnifiedExportError(f"'{key}' is neither a checkpoint tensor nor a trace constant")
    return ContainerMaterial(tensors=material, fixed=fixed)


def write_container(
    graph: IrGraph, material: ContainerMaterial, path: Path, revision: str | None
) -> IrGraph:
    """Write the container (`path` is the representative path of the parts); its provenance names
    the upstream commit when the checkpoint is pinned (the distribution checks it)."""
    return publish_model(
        path,
        graph,
        material.tensors,
        provenance=Provenance(
            license=GEMMA4_LICENSE, notice=NOTICE_FILENAME, upstream_revision=revision
        ),
        graph_name=GEMMA4_ROLE,
        weight_dtype="f32",
        fixed_weights=material.fixed,
    )


def _producers(graph: IrGraph) -> dict[str, IrNode]:
    return {out: node for node in graph.nodes for out in node.outs}


def _projection_of(
    producers: Mapping[str, IrNode], name: str, where: str
) -> tuple[IrNode, list[IrNode]]:
    """The linear a k / v row comes from, walking the main operand (`ins[0]`) upstream, and the
    nodes of that walk (the linear excluded)."""
    chain: list[IrNode] = []
    node = producers.get(name)
    for _ in range(16):
        if node is None:
            break
        if node.op == LINEAR_OP:
            return node, chain
        chain.append(node)
        node = producers.get(node.ins[0])
    raise UnifiedExportError(f"{where}: no linear within 16 steps of '{name}'")


def _leaves(producers: Mapping[str, IrNode], name: str) -> set[str]:
    """The values without a producer that `name` depends on (graph inputs and initializers)."""
    found: set[str] = set()
    seen: set[str] = set()
    frontier = [name]
    while frontier:
        value = frontier.pop()
        if value in seen:
            continue
        seen.add(value)
        node = producers.get(value)
        if node is None:
            found.add(value)
        else:
            frontier.extend(node.ins)
    return found


def _applied_leaves(producers: Mapping[str, IrNode], chain: Sequence[IrNode]) -> set[str]:
    """The leaves the chain's other operands (`ins[1:]`) depend on — what is applied to a
    projection on its way to the attention. MUST: only the other operands — the main operand
    carries the residual stream, which reaches every input (an earlier layer's RoPE included)."""
    found: set[str] = set()
    for node in chain:
        for name in node.ins[1:]:
            found |= _leaves(producers, name)
    return found


def _model_weights(graph: IrGraph, names: Iterable[str]) -> set[str]:
    """The checkpoint tensors among `names` (initializers that are not trace constants)."""
    return {
        graph.initializers[name].tensor
        for name in names
        if name in graph.initializers
        and not graph.initializers[name].tensor.startswith(CONSTANT_PREFIX)
    }


def assert_shared_kv_projection(graph: IrGraph, config: Any) -> dict[str, int]:
    """On the full layers of an `attention_k_eq_v` model, K and V are one projection and only K is
    rotated; on the other layers they are two projections.

    "Rotated" is read from what is applied between the projection and the attention: the full
    RoPE rows reach K's chain (`k_norm`, then the cos / sin products) and nothing that depends on
    any RoPE input reaches V's. V's chain also reads no checkpoint tensor: upstream's `v_norm` is
    scale-free (its unit weight is a trace constant), so a V normalized by `k_norm`'s learned
    scale — the unrotated key — fails here.
    MUST: checked in the written graph — the shapes and dtypes of K, V and a mis-wired V (the
    rotated or unrotated key, or another layer's projection) all agree, so only the wiring tells
    them apart.
    """
    producers = _producers(graph)
    shared_kv = bool(getattr(config, "attention_k_eq_v", False))
    rope = set(decode.ROPE_INPUTS)
    rope_full = {
        decode.rope_input_name(one_shot.FULL_ATTENTION, part) for part in decode.ROPE_PARTS
    }
    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    shared = 0
    for layer, (node, layer_type) in enumerate(zip(attentions, config.layer_types, strict=True)):
        where = f"attention[{layer}] ({layer_type})"
        key, key_chain = _projection_of(producers, node.ins[1], where)
        value, value_chain = _projection_of(producers, node.ins[2], where)
        if not (shared_kv and layer_type == one_shot.FULL_ATTENTION):
            if key is value:
                raise UnifiedExportError(f"{where}: K and V share a projection on this layer")
            continue
        if key is not value:
            raise UnifiedExportError(f"{where}: K and V come from different projections")
        value_leaves = _applied_leaves(producers, value_chain)
        if value_leaves & rope:
            raise UnifiedExportError(f"{where}: V is rotated (a RoPE input reaches its chain)")
        scaled = _model_weights(graph, value_leaves)
        if scaled:
            raise UnifiedExportError(
                f"{where}: V is scaled by {sorted(scaled)} (upstream's v_norm has no scale)"
            )
        if not rope_full <= _applied_leaves(producers, key_chain):
            raise UnifiedExportError(f"{where}: K is not rotated by the full RoPE rows")
        shared += 1
    return {"shared_kv_layers": shared}


# ---------------------------------------------------------------------------
# The reference (ruling R4)
# ---------------------------------------------------------------------------


def int4_codes(
    checkpoint: Checkpoint, key: str, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """The imported codes (one per int8) and group steps of one linear, from the same blocks the
    container's bytes are packed from ({@link _grid_blocks})."""
    rows, cols = _int4_shape(checkpoint, key, group_size)
    codes = torch.empty((rows, cols), dtype=torch.int8)
    scale = torch.empty((rows, cols // group_size), dtype=torch.float32)
    for start, stop, block_codes, block_steps in _grid_blocks(checkpoint, key, group_size):
        codes[start:stop] = block_codes
        scale[start:stop] = block_steps
    return codes, scale


class DequantizedAtUse(nn.Module):
    """A parametrization: the resident codes (int8) → `dequantize_int4(q, step)` on every read —
    the value the container's bytes decode to (axis 0, rows of groups)."""

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return dequantize_int4(codes, self.scale)


def imported_embedding(
    checkpoint: Checkpoint, key: str, group_size: int = DEFAULT_GROUP_SIZE
) -> torch.Tensor:
    """The embedding's imported grid as F32 values — `dequantize_int4` of the codes and steps the
    container's bytes are packed from ({@link _grid_blocks}), block by block, so the codes of the
    whole table are never held at once."""
    rows, cols = _int4_shape(checkpoint, key, group_size)
    table = torch.empty((rows, cols), dtype=torch.float32)
    for start, stop, codes, steps in _grid_blocks(checkpoint, key, group_size):
        table[start:stop] = dequantize_int4(codes, steps)
    return table


def reference_wrapper(
    config: Any, checkpoint: Checkpoint, int4: frozenset[str]
) -> UnifiedProductWrapper:
    """The F32 reference over resident i4 values dequantized at use (module docstring,
    "Goldens").

    The linears in `int4` hold their imported codes and steps ({@link int4_codes}) and read through
    {@link DequantizedAtUse}; the embedding (the tied head with it — one tensor) holds its imported
    grid values ({@link imported_embedding}); the rest is widened to F32 once.
    """
    from torch.nn.utils import parametrize

    model = meta_model(config)
    # Integer values cannot be parameters that require gradients (nothing here trains).
    model.requires_grad_(False)
    state: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    for key in model.state_dict():
        wrapper_key = f"model.{key}"
        if wrapper_key == TIED_HEAD_KEY:
            continue
        source = checkpoint_key(wrapper_key)
        if wrapper_key in int4:
            state[key], scales[wrapper_key] = int4_codes(checkpoint, source)
        elif wrapper_key == EMBED_KEY:
            state[key] = imported_embedding(checkpoint, source)
        else:
            state[key] = checkpoint.read(source)
    tied = TIED_HEAD_KEY.removeprefix("model.")
    state[tied] = state[EMBED_KEY.removeprefix("model.")]
    model.load_state_dict(state, assign=True, strict=True)
    del state
    model.tie_weights()
    one_shot.assert_materialized(model)
    for fqn in sorted(int4):
        module = model.get_submodule(fqn.removeprefix("model.").removesuffix(".weight"))
        parametrize.register_parametrization(
            module, "weight", DequantizedAtUse(scales.pop(fqn)), unsafe=True
        )
    decode.swap_rope_inputs(model)
    return UnifiedProductWrapper(model).eval()


def suppressed_logits(logits: torch.Tensor, suppress: Sequence[int]) -> torch.Tensor:
    """`logits` with the vocabulary entries `suppress` set to −inf (upstream's
    `SuppressTokensLogitsProcessor`, which `generate()` applies before greedy and sampled
    selection alike)."""
    if not suppress:
        return logits
    masked = logits.clone()
    masked[..., list(suppress)] = float("-inf")
    return masked


class GreedyView(nn.Module):
    """`(input_ids, position_ids) → (logits of the last row, hidden)` — the face
    `_shared.decode_series.greedy_continuation` calls (`gemma4.export_decode.GreedyRopeView`'s
    role, with the row selection the product graph adds).

    The logits come back with the suppressed ids at −inf ({@link suppressed_logits}), so the
    greedy token and its margin (top1 − top2) are those of the masked logits — what the host's
    selection sees (`pipelineConfig.suppressTokens`).
    """

    def __init__(
        self,
        wrapper: UnifiedProductWrapper,
        specs: Mapping[str, Any],
        suppress: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.specs = dict(specs)
        self.suppress = tuple(suppress)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, hidden = self.wrapper(
            input_ids, *decode.rope_args(self.specs, position_ids), decode.last_row_for(input_ids)
        )
        return suppressed_logits(logits, self.suppress), hidden


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def assert_trace_lengths(cases: Sequence[tuple[str, torch.Tensor]], sym_max: int) -> None:
    """Every case but {@link WINDOW_CASE} fits the traced chunk range (`2 ≤ T ≤ sym_max`) — the
    longest of them is the trace's example input. The window case is longer than one chunk on
    purpose (its goldens come from the torch reference at full length)."""
    for name, ids in cases:
        length = int(ids.shape[1])
        if name != WINDOW_CASE and not 2 <= length <= sym_max:
            raise UnifiedExportError(f"{name}: T={length} outside the traced range [2, {sym_max}]")


def assert_window_exercised(
    cases: Sequence[tuple[str, torch.Tensor]], steps: int, config: Any
) -> dict[str, int]:
    """The goldens evaluate the sliding window, and {@link WINDOW_CASE} overflows the ring.

    MUST: the longest full re-forward of the goldens is longer than `sliding_window` — up to the
    window the band mask equals the causal one, so goldens that never pass it leave the sliding
    layers' window unchecked (the rule of `gemma4.export.assert_case_lengths` for the PLE series).
    The last greedy step re-forwards the prompt plus the `steps − 1` tokens taken before it, so
    the longest re-forward is `T + steps − 1` rows. MUST: the window case's prompt alone is longer
    than the ring (`window + SLIDING_SLACK_ROWS` rows — `gemma4.export_decode.sliding_capacity`),
    so the host's prefill evicts rows before the first token is taken.
    """
    window = int(config.sliding_window)
    ring = decode.sliding_capacity(config)
    lengths = {name: int(ids.shape[1]) for name, ids in cases}
    longest = max((length + steps - 1 for length in lengths.values()), default=0)
    if longest <= window:
        raise UnifiedExportError(
            f"the longest full re-forward of the goldens is {longest} rows, within the sliding"
            f" window {window} — the band mask would equal the causal one"
        )
    if WINDOW_CASE not in lengths:
        raise UnifiedExportError(f"the golden cases {sorted(lengths)} lack {WINDOW_CASE!r}")
    if lengths[WINDOW_CASE] <= ring:
        raise UnifiedExportError(
            f"{WINDOW_CASE}: T={lengths[WINDOW_CASE]} does not exceed the sliding ring of {ring}"
            " rows"
        )
    return {"sliding_window": window, "ring_rows": ring, "longest_forward": longest}


def suppress_tokens(tokenizer_dir: Path, config: Any) -> tuple[int, ...]:
    """The ids upstream's `generation_config.json` suppresses, read by the distribution's own
    reader (`gemma4.distribution.gemma4_generation_defaults` — the goldens and the manifest's
    `suppressTokens` cannot disagree), bounded by the config's vocabulary; empty when none."""
    declared = gemma4_generation_defaults(tokenizer_dir).suppress_tokens
    if declared is None:
        return ()
    assert_gemma4_suppress_tokens(declared, int(config.vocab_size), str(tokenizer_dir))
    return declared


def _trace(
    model_dir: Path, tokenizer_dir: Path, layers: int | None, sym_max: int
) -> tuple[Any, tuple[tuple[str, torch.Tensor], ...], IrGraph, dict[str, torch.Tensor], TraceKeys]:
    config = load_config(model_dir, layers)
    cases = one_shot.encode_cases(tokenizer_dir, UNIFIED_CASES)
    assert_trace_lengths(cases, sym_max)
    _, example_ids = max(
        (case for case in cases if case[0] != WINDOW_CASE), key=lambda case: case[1].shape[1]
    )
    wrapper = shape_only_model(config)
    graph, tensors = trace(wrapper, example_ids, sym_max)
    keys = trace_keys(wrapper)
    return config, cases, graph, tensors, keys


def plan(
    model_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    layers: int | None,
    sym_max: int,
    steps: int = decode.GREEDY_STEPS["12b"],
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Trace only (no weight is read): the graph's form, its ops, the checkpoint's keys, the case
    lengths against the window and the suppressed ids. `model_dir` holds the weights and their
    config, `tokenizer_dir` the tokenizer and the generation defaults (the same directory for a
    one-source checkpoint); `pins` names both sources."""
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    with MemoryWatch(floor) as monitor, monitor.stage("trace"):
        config, cases, graph, _, keys = _trace(model_dir, tokenizer_dir, layers, sym_max)
    assert_checkpoint_keys(
        keys.persistent, Checkpoint(model_dir).names(), int(config.num_hidden_layers)
    )
    form = product.assert_ir_form_product(graph, config, {}, None)
    return {
        "stages": [stage.to_dict() for stage in monitor.records],
        "layers": int(config.num_hidden_layers),
        "inputs": [[spec.name, spec.dtype, spec.shape] for spec in graph.inputs],
        "nodes": len(graph.nodes),
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
        "symbols": list(graph.symbols),
        "i4_targets": len(keys.int4),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "window": assert_window_exercised(
            [case for case in cases if case[0] in UNIFIED_GREEDY_CASES], steps, config
        ),
        "suppress_tokens": list(suppress_tokens(tokenizer_dir, config)),
        "form": form,
        "kv": assert_shared_kv_projection(graph, config),
    }


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    sym_max: int = one_shot.SYM_MAX,
    steps: int = decode.GREEDY_STEPS["12b"],
    layers: int | None = None,
    floor: float = MEMORY_FLOOR_GIB,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Write the product container, the greedy goldens and the provenance record through one
    staged seat (a truncated export — `layers` — writes the container alone).

    `model_dir` holds the weights and their config, `tokenizer_dir` the tokenizer and the
    generation defaults ({@link plan}). `checkpoint_label` is how the record names the checkpoint
    (`checkpoint.dir`): `gemma4.sources.checkpoint_label` for hub snapshots, whose directory names
    are only the commits; `None` keeps the directory name
    (`gemma4.provenance.checkpoint_fingerprint`). With `pins` the record also names both sources
    (`checkpoint.sources`) and the container the weights' revision.

    MUST: every gate (margins, first tokens, form, K = V wiring, the pinned fingerprint) passes
    before the seat is published — a failed run leaves nothing a gate could later accept.
    MUST: the goldens are taken before the container is quantized — a margin that fails stops the
    run before the long write, and the reference (≈ 16 GiB) is released before the quantized
    weights (≈ 8 GiB) are built.
    """
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    goldens = layers is None
    checkpoint = Checkpoint(model_dir)
    with MemoryWatch(floor) as monitor:
        with monitor.stage("trace") as record:
            config, cases, graph, tensors, keys = _trace(model_dir, tokenizer_dir, layers, sym_max)
            record.details["nodes"] = len(graph.nodes)
        assert_checkpoint_keys(keys.persistent, checkpoint.names(), int(config.num_hidden_layers))
        greedy_cases = tuple(case for case in cases if case[0] in UNIFIED_GREEDY_CASES)
        assert_case_room(greedy_cases, steps, decode.max_position(config))
        window = assert_window_exercised(greedy_cases, steps, config) if goldens else {}
        suppress = suppress_tokens(tokenizer_dir, config) if goldens else ()
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            measured: dict[str, Any] = {}
            if goldens:
                with monitor.stage("goldens"):
                    print(
                        "[goldens] reference: imported i4 codes, dequantized at use;"
                        f" suppressed ids {list(suppress)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    reference = reference_wrapper(config, checkpoint, keys.int4)
                    view = GreedyView(reference, decode.rope_specs(config), suppress)
                    written, tokens, margins = _write_greedy(
                        view, greedy_cases, staged, steps=steps, floor=decode.MARGIN_FLOOR
                    )
                    first = {
                        name: (
                            tokens[name][0]
                            if name in tokens
                            else greedy_continuation(view, ids, 1, label=name)[0][0]
                        )
                        for name, ids in cases
                        if name in FIRST_TOKEN_EXPECTATIONS
                    }
                    del view, reference
                gc.collect()
                measured = _greedy_summary(tokenizer_dir, written, tokens, margins, first, steps)
                measured["suppress_tokens"] = list(suppress)
            with monitor.stage("quantize") as record:
                material = prepare_material(graph, tensors, keys, checkpoint)
                record.details["fixed"] = len(material.fixed)
            del tensors
            with monitor.stage("write"):
                verified = write_container(
                    graph,
                    material,
                    staged / one_shot.MODEL_FILE,
                    None if pins is None else pins.weights.revision,
                )
            del material
            # Every linear and the tied embedding are i4. MUST: no i8 is counted as well — an
            # embedding rounded to i8 again leaves the graph's shapes and ops unchanged.
            storage = {"i4": len(keys.int4) + 1, "i8": 0}
            form = product.assert_ir_form_product(verified, config, storage, None)
            kv = assert_shared_kv_projection(verified, config)
            if goldens:
                with monitor.stage("provenance"):
                    own = provenance.assert_reference_goldens(staged, greedy_cases)
                    # The goldens live in this series: the record names it by its published name
                    # (the staged seat's name is transient).
                    # The record also names the ids the goldens were taken with masked
                    # (`suppressTokens` — the host must apply the same masking to reproduce them).
                    record_ = provenance.build_record(
                        out_dir,
                        model_dir,
                        {**own, "series": out_dir.name, "suppressTokens": list(suppress)},
                        checkpoint_label=checkpoint_label,
                        tokenizer_dir=tokenizer_dir,
                        pins=pins,
                    )
                    if pins is not None:
                        provenance.assert_pinned_fingerprint(record_["checkpoint"], pins)
                    provenance.write_record(staged, record_)
                measured["reference"] = record_["reference"]
    return {
        "dir": str(out_dir),
        "layers": int(config.num_hidden_layers),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "parts": len(container_parts(out_dir / one_shot.MODEL_FILE)),
        "model_bytes": sum(
            path.stat().st_size for path in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "window": window,
        "quantized": storage,
        "form": form,
        "kv": kv,
        **measured,
        "stages": [stage.to_dict() for stage in monitor.records],
    }


def _greedy_summary(
    tokenizer_dir: Path,
    written: list[str],
    tokens: Mapping[str, list[int]],
    margins: Mapping[str, list[float]],
    first: Mapping[str, int],
    steps: int,
) -> dict[str, Any]:
    """The goldens' record for the summary, and the first-token sanity on the cases of
    {@link FIRST_TOKEN_EXPECTATIONS} (`first` — MUST: before the seat is published —
    `gemma4.export._sanity` fails loudly)."""
    missing = sorted(set(FIRST_TOKEN_EXPECTATIONS) - set(first))
    if missing:
        raise UnifiedExportError(f"no first token for the expected cases {missing}")
    tokenizer = one_shot.load_tokenizer(tokenizer_dir)
    expected = one_shot.expected_token_ids(tokenizer, FIRST_TOKEN_EXPECTATIONS)
    labels = {
        token: tokenizer.id_to_token(token)
        for token in set(first.values()) | set(expected.values())
    }
    return {
        "greedy": written,
        "greedy_steps": steps,
        "margin_min": {name: min(values) for name, values in sorted(margins.items())},
        "continuation": {
            name: tokenizer.decode(continuation) for name, continuation in sorted(tokens.items())
        },
        "sanity": one_shot._sanity(first, expected, labels),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = one_shot.series_parser(
        __doc__.split("\n\n")[0],
        partial(one_shot.series_dir, suffix=GEMMA4_PRODUCT_SUFFIX),
        models=MODELS,
    )
    parser.set_defaults(model=MODELS[0])
    parser.add_argument("--steps", type=int, default=None)
    one_shot.add_model_default(parser, "steps", decode.GREEDY_STEPS.__getitem__)
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="the tokenizer and generation defaults (default: the model's pinned tokenizer source)",
    )
    one_shot.add_model_default(parser, "tokenizer_dir", sources.tokenizer_dir)
    parser.add_argument("--layers", type=int, default=None, help="keep the first N layers")
    parser.add_argument("--plan", action="store_true", help="trace only; nothing is written")
    parser.add_argument("--memory-floor", type=float, default=MEMORY_FLOOR_GIB)
    args = parser.parse_args(argv)
    if args.layers is not None and args.out is None and not args.plan:
        parser.error("--layers needs --out (a truncated model is not a series)")
    model = args.model
    options = one_shot.resolve_series_args(args)
    model_dir = options["model_dir"]
    if options["plan"]:
        summary = plan(
            model_dir,
            tokenizer_dir=options["tokenizer_dir"],
            pins=sources.HUB_PINS.get(model),
            layers=options["layers"],
            sym_max=options["sym_max"],
            steps=options["steps"],
            floor=options["memory_floor"],
        )
    else:
        summary = export_series(
            model_dir,
            options["out"],
            tokenizer_dir=options["tokenizer_dir"],
            pins=sources.HUB_PINS.get(model),
            sym_max=options["sym_max"],
            steps=options["steps"],
            layers=options["layers"],
            floor=options["memory_floor"],
            checkpoint_label=sources.checkpoint_label(model),
        )
    print(
        json.dumps(
            {
                "model_dir": str(model_dir),
                "tokenizer_dir": str(options["tokenizer_dir"]),
                **summary,
            },
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
