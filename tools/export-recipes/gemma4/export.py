"""Script that writes the real-weight Gemma 4 E2B / E4B (text decoder, `--model e2b|e4b`, default
e2b) in the **one-shot form** to `model.krm` (IR v2) + golden io.

The mirror of `minicpm5/export.py` (`input_ids[1,T] → logits[1,T,262144]`, no KV cache), but the
acceptance has two more focal points:

- the **band masks of the two layer types** (sliding / full — 28 / 7 layers for E2B, 35 / 7 for
  E4B) both fold into a T-independent Tmax constant + `sym_prefix_slice` (ADR 0010 — the same
  mechanism as the band mask of `embeddinggemma/export.py`, in two **causal** variants).
- **mixed quantization** (embeddings i8 × linears i4 — ADR 0069 decision 5 /
  `weight_dtype_overrides`) runs through a real model. E2B has a vocabulary of 262144 × a PLE
  table of 8960 columns, a form in which "the embeddings are most of the weights", so neither i4
  alone nor i8 alone works.

    uv run --with 'transformers==5.14.1' python -m gemma4.export              # E2B
    uv run --with 'transformers==5.14.1' python -m gemma4.export --model e4b  # E4B

`--model` only decides the default locations (the material `inputs/gemma4/<checkpoint>/` and the
series `outputs/series/gemma4-<model>/`); every form (layer count, head count, KV sharing, PLE
dimensions) is derived from the upstream config — the script has no number that branches on the
model name.

transformers is **pinned to 5.14.1** (the same reason as `minicpm5/export.py` — a change of the
modeling code changes the graph form). It is not added to pyproject.toml / uv.lock but added
temporarily with `--with`.

## What goes into the graph

The whole forward of `Gemma4ForCausalLM` as one graph (all layers (E2B 35 / E4B 42) + the tied
`lm_head` + `final_logit_softcapping`).
The official checkpoint is multimodal (`Gemma4ForConditionalGeneration`), so the text part's keys
`model.language_model.*` are **renamed** to `model.*` and loaded into a text-only model
({@link renamed_state}). The vision / audio towers are not read. Tokenizing and applying the chat
template are the host's job.

MUST: head_dim **differs per layer type** (sliding = `config.head_dim` 256 / full =
`config.global_head_dim` 512). `hidden_size / num_attention_heads` = 192 is neither, so no derived
form is written ({@link assert_ir_form} matches it layer by layer, in layer order).

## Emitting GQA / MQA in its "true form"

Gemma 4 E2B is an 8:1 MQA (`num_key_value_heads` 1), E4B a 4:1 GQA (2). The reason for the detour
and its method are the same as in `minicpm5/export.py` — {@link gqa_sdpa_attention} is registered
at transformers' public extension point (`AttentionInterface.register`) and calls SDPA with
`enable_gqa=True`, without `repeat_kv`.
The IR's attention becomes `q[1,8,T,D] / k[1,Hkv,T,D] / v[1,Hkv,T,D]`
(Hkv = `num_key_value_heads`).

`Gemma4TextAttention` passes `sliding_window=` to the attention implementation, but
{@link gqa_sdpa_attention} **accepts and ignores it** — the window semantics have their source of
truth in the mask dictionary below, and {@link assert_ir_form} checks in layer order which
constant reaches which layer. Upstream passes scaling 1.0 (q has an RMSNorm applied), and it goes
to SDPA as is.

## Causality and the window are two additive masks, folded as constants

`Gemma4TextModel.forward` **hands `attention_mask` to the layers as is when it is a dict**
(modeling_gemma4.py:1695). So `{"full_attention": causal, "sliding_attention": band}` is built
here and passed in. The leaves of both are only `arange` and `full`, so the exporter's constant
folding reduces them to **a Tmax×Tmax f32 initializer + `sym_prefix_slice`** (ADR 0010). As a
result **the graph has one input, `input_ids`**. The path in which the mask arrives as `None` is
made to fail loudly by {@link gqa_sdpa_attention} (no form that turns non-causal is left).

## The PLE (Per-Layer Embeddings) is held split per layer

The single `embed_tokens_per_layer` table (9.4GB for E2B / 11.3GB for E4B in f32) is held **split
into L per-layer tables (E2B 35 / E4B 42)** — the split,
the row-block reads and the bit-identity check have their source of truth in {@link gemma4.ple}
(the three scripts go through the same one). The script's contract is only how the graph is
called: the `[1,T,35,256]` built by `stack` is passed upstream as `per_layer_inputs=` (upstream
refuses `input_ids` together with `per_layer_inputs`, so `inputs_embeds` is looked up here too).
The context projection and the combine (`project_per_layer_inputs`) go through **the upstream
implementation unchanged**.

## Mixed quantization (the core of this series)

Rounding comes **before the reference and the goldens are taken** (ADR 0006). The order and the
targets are in {@link quantize_wrapper}:

1. `fake_quant_int8` — the embeddings only (the main embedding + the 35 split PLE tables). The
   tied `lm_head` uses the entity rounded here as is.
2. `fake_quant_int4` — the linears only (`lm_head` excluded).

MUST: the two target sets are **disjoint** (running the tied `lm_head` through i4 as well rounds
it twice, and the i8 scale ledger taken first disagrees with the actual values). The storage is
the default `i8` with each linear given `i4` explicitly — the reverse (default i4 + the embeddings
given i8 explicitly) would require knowing, **before writing**, the FQN export gives the tied
entity (either `model.lm_head.weight` or `model.model.embed_tokens.weight`). The i4 side's keys
are the fake-quant scale ledger itself, so the override and the rounding come from one place.

Not quantized (kept f32): every norm weight, `layer_scalar`, the constants derived from RoPE and
the mask constants. None of them is consumed by a weight slot, so they are outside the set eligible
for compressed storage (`emit.eligible_compressed_initializers`).

## Output layout

    outputs/series/gemma4-<model>/model.krm             weights, constants + the 2 descriptors
    outputs/series/gemma4-<model>/io.<case>.safetensors inputs and torch CPU expected outputs

The io tensor key convention is the same as tiny golden / DeBERTa / EmbeddingGemma / MiniCPM5
(`input.<graph input name>` / `output.<position>`). The logits have a vocabulary of 262144, so each
case takes `T × 1.05MB` (627MB for `context-en` at T=598) — disk is plentiful, so every row is
written.

## BOS is added here

The post_processor of `tokenizer.json` **adds no** special tokens (the `special_tokens` of
`TemplateProcessing` is empty). `tokenizer_config.json` has no `add_bos_token` either, and what
places `<bos>` is `chat_template.jinja` (`{{- bos_token -}}` on line 188) — so BOS is the host's
job. This script makes the goldens from plain continuations without the chat template, so
{@link build_cases} puts exactly one `<bos>` in front.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.paths import SERIES_ROOT
from gemma4 import ple, rope, sources
from gemma4.card import GEMMA4_LICENSE, GEMMA4_PLE_MODELS, GEMMA4_UPSTREAM
from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_PREFIX,
    GEMMA4_ROLE,
    gemma4_checkpoint,
    gemma4_series_name,
)
from karume.artifacts import staged_publication
from karume.container import Provenance, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.dist import NOTICE_FILENAME
from karume.ir import IrGraph
from karume.pipeline import export_to_file
from karume.quantize import Int4Report, Int8Report, fake_quant_int4, fake_quant_int8
from karume.rope import assert_rope_lifted
from karume.shapes import declared_shape


def checkpoint_dir(model: str) -> Path:
    """The location of the official weights (where `hf download google/gemma-4-<E2B|E4B>-it`
    extracts to).

    The spelling is derived from the attribution table (`gemma4.card.GEMMA4_UPSTREAM`) — the same
    one path as `gemma4_sources` of the distribution recipe (an unknown model fails loudly there).

    The directory itself is `gemma4.sources.checkpoint_dir` (the pinned snapshot of the weights for
    12B).
    """
    gemma4_checkpoint(model)
    return sources.checkpoint_dir(model)


def series_dir(model: str, suffix: str | None = None) -> Path:
    """The default location of a series (`outputs/series/gemma4-<model>[-<suffix>]/`).

    Only the one-shot series (`gemma4-e2b/`) has no suffix. A suffixed spelling goes through the
    same one path as the distribution recipe's `gemma4_series_name` (the writer and the reader
    build it from the same word).
    """
    gemma4_checkpoint(model)
    name = f"{GEMMA4_PREFIX}-{model}" if suffix is None else gemma4_series_name(model, suffix)
    return SERIES_ROOT / name


#: The location of the default model's (E2B) official weights. The location `--model` chooses is
#: {@link checkpoint_dir}.
DEFAULT_MODEL_DIR = checkpoint_dir(GEMMA4_DEFAULT_MODEL)

#: The default location of the default model's (E2B) outputs.
DEFAULT_OUT_DIR = series_dir(GEMMA4_DEFAULT_MODEL)

MODEL_FILE = "model.krm"

#: The weights file name of the **upstream checkpoint** (HF's spelling — not the output container).
#: It is kept apart because upstream decides the reading side's spelling even after the
#: distribution moved to `.krm` (the same fact as `FINGERPRINT_FILES` of `gemma4.provenance`).
CHECKPOINT_FILE = "model.safetensors"

#: The provenance baked into the container (container-v1 §2.3). The license identifier is read from
#: the card's source of truth ({@link gemma4.card.GEMMA4_LICENSE}) — the two tables must not move
#: independently. The three series (one-shot / decode / product) and the drafter all come from the
#: same upstream checkpoint, so they share one provenance too.
PROVENANCE = Provenance(license=GEMMA4_LICENSE, notice=NOTICE_FILENAME)
IO_PREFIX = "io."
IO_SUFFIX = ".safetensors"
INPUT_PREFIX = "input."
OUTPUT_PREFIX = "output."

#: The graph input's name (the wrapper's forward argument name). The spelling {@link assert_ir_form}
#: uses to check "this one only" — if mask or position_ids show up among the inputs, folding did
#: not take effect.
INPUT_IDS = "input_ids"

#: IR op names used by the checks.
ATTENTION_OP = "attention"
SYM_PREFIX_SLICE_OP = "sym_prefix_slice"

#: The upper bound of the symbolic dimension T. The mask constants are baked at Tmax² **for each of
#: the two layer types** (768² × 4B × 2 = 4.5MiB), so raise it only knowing that price. At 512 or
#: below the band mask equals causal and the sliding layers go unverified, so it sits at a point
#: safely above `sliding_window` (512).
SYM_MAX = 768

#: The blocking value of an additive mask (the same value `normalize._additive_attn_mask` turns a
#: bool mask into).
NEG_INF = float("-inf")

#: The name registered with `AttentionInterface`. **It contains none of `sdpa` /
#: `flash_attention` / `flex_attention` as a substring** (the same reason as `minicpm5/export.py`
#: — a partial name match branches into backend-specific dispatch checks).
ATTENTION_NAME = "karume_gqa"

#: The layer types that appear in `config.layer_types` (= the mask dictionary's keys = the keys of
#: the RoPE declaration). Canonical spelling in {@link gemma4.rope} (the distribution recipe reads
#: the same vocabulary, so it lives on the side that needs no torch).
FULL_ATTENTION = rope.FULL_ATTENTION
SLIDING_ATTENTION = rope.SLIDING_ATTENTION

#: The key prefix of the text part in the multimodal checkpoint, and the prefix in the text-only
#: model.
CHECKPOINT_TEXT_PREFIX = "model.language_model."
MODEL_TEXT_PREFIX = "model."

#: The single PLE table (the checkpoint key **before** renaming).
PLE_CHECKPOINT_KEY = f"{CHECKPOINT_TEXT_PREFIX}embed_tokens_per_layer.weight"

#: The state_dict keys of the main embedding / tied lm_head **after** renaming.
EMBED_TOKENS_KEY = f"{MODEL_TEXT_PREFIX}embed_tokens.weight"
LM_HEAD_KEY = "lm_head.weight"

#: Module FQNs relative to the wrapper (the space fake-quant's `include` and the scale ledger's keys
#: live in).
PER_LAYER_PREFIX = "per_layer."
EMBED_TOKENS_MODULE = "model.model.embed_tokens"
LM_HEAD_MODULE = "model.lm_head"

#: The text of the longer (T=598) English case. It exists to reach **T > 512 (`sliding_window`)**;
#: without it the band mask equals causal and the sliding layers' window is never evaluated.
#:
#: It ends in "… the city called" (continuation = `▁Paris`). Ending in the plain "The capital of
#: France is" gives, for a narrative this long, `▁the` as the top token even on the bf16 standard
#: path (continuations like "the city of Paris"), and no expectation holds — this form, the best of
#: three measured endings (2026-08-19, margins 12.8 / **13.9** / 0.1), was fixed. The short case's
#: record (`▁Paris` at the top) does not extrapolate to the long text.
CONTEXT_EN = (
    "France is a country in Western Europe whose written history reaches back to the Roman "
    "province of Gaul. Its territory is shaped by four great river basins, and the towns that "
    "grew along those rivers became the markets, the bishoprics and eventually the "
    "administrative centres of the kingdom. The Loire carried stone and wine, the Rhone linked "
    "the Mediterranean ports to the interior, the Garonne drained the south west toward the "
    "Atlantic, and the Seine ran north west through a wide basin of wheat and pasture before "
    "reaching the sea at Le Havre. On an island in the middle of the Seine a Gaulish people "
    "called the Parisii kept a settlement that the Romans fortified and renamed Lutetia. "
    "The island was small enough to defend and the crossing was easy enough to tax, which is "
    "the combination that turns a river ford into a city. "
    "Over the following centuries the settlement spread from the island to both banks. The "
    "Capetian kings made it the seat of their household in the tenth century, and from then on "
    "the royal archives, the mint, the courts of justice and the university grew up around the "
    "same few streets. When the monarchy fell, the revolutionary assemblies met there; when the "
    "republic was rebuilt after each of its interruptions, the new institutions were installed "
    "in the same buildings. The result is a degree of centralisation that has no real parallel "
    "among comparable European states: the national library, the supreme administrative court, "
    "the central bank, the principal railway hub and the residence of the head of state are all "
    "within a few kilometres of one another. "
    "Administrative reform has repeatedly tried to loosen this concentration. Regional councils "
    "were given budgets and elected assemblies, several ministries moved departments to "
    "provincial cities, and the high speed rail network was built partly to make those cities "
    "reachable in an afternoon. The concentration nevertheless persists, because the courts, "
    "the ministries and the parliament did not move, and the private institutions that depend on "
    "them stayed where they were. A company that needs a permit, a broadcaster that needs an "
    "audience and a publisher that needs an editor all end up in the same place. "
    "The same pull is visible in the transport map. The motorways radiate outward like the "
    "spokes of a wheel and are numbered from the centre, the high speed lines all begin at one "
    "of six terminal stations, and a traveller going from one provincial city to another often "
    "finds that the quickest route runs back through the middle of the country and out again. "
    "Airline schedules follow the same shape, and so do the freight corridors that feed the "
    "ports of Le Havre and Marseille. Planners have described the pattern for a century as a "
    "wheel with one hub, and every attempt to add a second hub has so far produced a spoke. "
    "For a reader who has followed the geography, the political history and the administrative "
    "reforms above, the conclusion is not in doubt. The parliament meets there, the ministries "
    "are quartered there, the head of state lives there, and the Seine still runs past the island "
    "where the first settlement stood. The capital of France is the city called"
)

#: The fixed golden texts (`(case name, text)`). T, `<bos>` included, is spread to 6 / 10 / 598
#: (far from the 0/1 specialization, on both sides of `sliding_window` 512, inside the bound 768).
#: `capital-ja` has the same **paired structure** as minicpm5 (the lead-in
#: 「フランスの首都はパリ、」), a form whose continuation is more determined than the plain
#: 「日本の首都は」.
GOLDEN_CASES: tuple[tuple[str, str], ...] = (
    ("capital-en", "The capital of France is"),
    ("capital-ja", "フランスの首都はパリ、日本の首都は"),
    ("context-en", CONTEXT_EN),
)

#: The continuation expected as **the greedy token at the last position** of each case (a sanity
#: check that is not tautological — ADR 0005's fail loudly). Each is a "strongly determined
#: continuation", so if any of (1) mixed-up weights (2) the mask's direction (seeing the future)
#: (3) a RoPE position shift (4) a wrong PLE layer assignment happens, the top token at the last
#: position changes.
#: MUST: an expected continuation is **a single token** ({@link expected_token_ids} checks it).
GREEDY_EXPECTATIONS: Mapping[str, str] = {
    "capital-en": " Paris",
    "capital-ja": "東京",
    "context-en": " Paris",
}


def additive_causal_mask(length: int) -> torch.Tensor:
    """Build the additive causal mask `[1,1,length,length]` (allow 0 / block −inf).

    MUST: the leaves are only `arange` and `full`, and `length` appears **only as an arange
    length** (the extent position of `convert.SYMBOL_EXTENT_ARGS`). With T on the value side, Tmax
    folding no longer commutes with the prefix (ADR 0010 addendum).
    MUST: the comparison is written `cols <= rows` (`le.Tensor`) — `rows >= cols` (`ge.Tensor`) is
    not in `convert.FOLDABLE_OPS`, so it cannot be folded and the whole export fails on an op
    outside the IR vocabulary.

    NOTE: during export `length` is a `torch.SymInt` (it behaves as an int).
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    return torch.where(cols <= rows, keep, drop).unsqueeze(0).unsqueeze(0)


def additive_sliding_mask(length: int, window: int) -> torch.Tensor:
    """Build the additive **band** mask `[1,1,length,length]` (causal and distance < window).

    Equivalent to upstream's `sliding_window_mask_function` (modeling_gemma4.py:1915) — the left
    window is `0 <= dist < window` (self included), and there is no right window as it is causal.
    The window is 512, so for `T <= 512` it equals causal (why the goldens hold one case with
    T > 512).

    MUST: on top of the constraints of {@link additive_causal_mask} (only arange / full as leaves,
    `length` only at extent positions), the distance test is written as `sub` + `lt.Scalar` (a form
    in `convert.FOLDABLE_OPS` — a comparison outside it cannot be folded and the whole export
    fails).
    """
    rows = torch.arange(length).unsqueeze(-1)
    cols = torch.arange(length).unsqueeze(0)
    keep = torch.full((), 0.0, dtype=torch.float32)
    drop = torch.full((), NEG_INF, dtype=torch.float32)
    within = (cols <= rows) & ((rows - cols) < window)
    return torch.where(within, keep, drop).unsqueeze(0).unsqueeze(0)


def gqa_sdpa_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """An attention implementation that calls SDPA with `enable_gqa=True`, without `repeat_kv`
    (keeps the MQA / GQA form).

    Exactly the calling convention transformers' `AttentionInterface` requires
    (`(module, q, k, v, attention_mask, dropout=…, scaling=…, **kwargs)` →
    `(attn_output[B,M,H,D], attn_weights)`). Only three things are dropped from upstream's
    `sdpa_attention_forward`: (1) materializing `repeat_kv` (2) the automatic `is_causal` decision
    (3) the NPU / paged branches.

    MUST: `attention_mask is None` fails loudly. Upstream uses this form as the signal "leave it to
    `is_causal=True`", but passing it through here silently yields a **non-causal** attention (the
    IR has no causal field, so no later check catches it).
    MUST: kwargs that change the numeric meaning (`is_causal` / `position_bias`) are not accepted.
    NOTE: `sliding_window` is **accepted and ignored** — the window's source of truth is the mask
    constant the caller passes, and {@link assert_ir_form} checks the per-layer assignment in layer
    order. Reinterpreting the window here would split "which of the mask dictionary and the kwargs
    takes effect" across two places.
    """
    if attention_mask is None:
        raise ValueError(
            f"{type(module).__name__}: attention called without attn_mask"
            " (causality and the window are expressed as additive masks — without a mask it turns"
            " non-causal)"
        )
    if float(dropout) != 0.0:
        raise ValueError(f"attention with dropout={dropout} is not supported (inference only)")
    unsupported = sorted(
        name for name in ("is_causal", "position_bias") if kwargs.get(name) is not None
    )
    if unsupported:
        raise ValueError(
            f"attention with kwargs {unsupported} is not supported (the numeric path changes)"
        )
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
        enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


def register_attention() -> None:
    """Register {@link gqa_sdpa_attention} with transformers' attention registry (idempotent)."""
    from transformers import AttentionInterface

    AttentionInterface.register(ATTENTION_NAME, gqa_sdpa_attention)


class Gemma4Wrapper(nn.Module):
    """The export wrapper that fixes `Gemma4ForCausalLM` to the one-shot form
    (`input_ids → logits`).

    MUST: `attention_mask` is **always passed as the two layer types' dictionary**. Omitted,
    transformers builds the mask (`create_causal_mask` / `create_sliding_window_causal_mask`), and
    values derived from `cache_position` mix into the folding's leaves.
    MUST: call with `inputs_embeds` + `per_layer_inputs`, not `input_ids` — upstream refuses
    `input_ids` together with `per_layer_inputs` (modeling_gemma4.py:1675).
    MUST: `use_cache=False`. With True a `DynamicCache` appears and it leaves the one-shot contract.
    """

    def __init__(self, model: nn.Module, tables: nn.ModuleList) -> None:
        super().__init__()
        self.model = model
        self.per_layer = tables
        self.per_layer_scale = ple.per_layer_scale(model.config)
        self.sliding_window = int(model.config.sliding_window)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        mask = {
            FULL_ATTENTION: additive_causal_mask(length),
            SLIDING_ATTENTION: additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        return self.model(
            inputs_embeds=embeds,
            per_layer_inputs=stacked,
            attention_mask=mask,
            use_cache=False,
        ).logits


def is_int8_module(name: str) -> bool:
    """Whether the module is an embedding rounded to i8 (the main embedding + the 35 split PLE
    tables).

    `lm_head` is an `nn.Linear`, so it is not included — it is tied, so rounding the main embedding
    rounds its entity. The tests pin that this is **disjoint** from {@link is_int4_module}.
    """
    return name == EMBED_TOKENS_MODULE or name.startswith(PER_LAYER_PREFIX)


def is_int4_module(name: str) -> bool:
    """Whether the module is a linear rounded to i4 (only the tied `lm_head` is left out).

    `fake_quant_int4` looks at nothing but `nn.Linear`, so dropping `lm_head` here is enough (no
    double rounding — the docstring of `quantize.fake_quant_int4`).
    """
    return name != LM_HEAD_MODULE


def load_text_config(model_dir: Path) -> Any:
    """Take the text part (`Gemma4TextConfig`) out of the multimodal config."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_dir).get_text_config()


def renamed_state(
    model: nn.Module, model_file: Path, probe: Sequence[int]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Return the f32 state_dict with the text part's keys renamed, and the list of dropped keys.

    It makes three renames / replacements:

    - `model.language_model.*` → `model.*` (the vision / audio towers' keys are not read).
    - Drop the `k_proj` / `v_proj` / `k_norm` / `v_norm` residues left in the KV-shared layers.
      What to drop is upstream's own `_keys_to_ignore_on_load_unexpected` (built from the layering
      by `Gemma4TextModel.__init__`), used **as is** — rewriting the sharing rule here would make
      two places move independently when upstream changes how it splits the layers.
    - Of the single PLE table, only the {@link gemma4.ple.PLE_PROBE_ROWS} probe rows
      (the scattered points of {@link gemma4.ple.probe_rows}) go into the check slot (the body is
      the 35-way split {@link gemma4.ple.load_per_layer_tables} holds).

    The tied `lm_head.weight` puts **the same tensor twice** into the state_dict — the checkpoint
    has no entity for it (`_tied_weights_keys`), and without it `load_state_dict` reports it
    missing.

    MUST: each tensor is **a copy that shares no memory with the checkpoint** (`copy=True`).
    {@link load_model_and_tables} lets the model own these entities as is with `assign=True`, so no
    tensor that was f32 to begin with (`.to` makes no copy) may stay pointing at the file mapping.
    """
    ignored = tuple(str(pattern) for pattern in model._keys_to_ignore_on_load_unexpected)
    state: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    with safe_open(str(model_file), framework="pt") as handle:
        for key in sorted(handle.keys()):
            if not key.startswith(CHECKPOINT_TEXT_PREFIX):
                continue
            name = MODEL_TEXT_PREFIX + key[len(CHECKPOINT_TEXT_PREFIX) :]
            if any(re.search(pattern, name) for pattern in ignored):
                dropped.append(name)
                continue
            if key == PLE_CHECKPOINT_KEY:
                sliced = handle.get_slice(key)
                rows = torch.cat([sliced[row : row + 1, :] for row in probe])
                state[name] = rows.to(torch.float32)
                continue
            state[name] = handle.get_tensor(key).to(torch.float32, copy=True)
    if EMBED_TOKENS_KEY not in state:
        raise ValueError(f"the checkpoint has no '{CHECKPOINT_TEXT_PREFIX}embed_tokens.weight'")
    state[LM_HEAD_KEY] = state[EMBED_TOKENS_KEY]
    return state, dropped


def build_wrapper(model: nn.Module, tables: nn.ModuleList) -> Gemma4Wrapper:
    """Drop the check slot's PLE table and build the export wrapper.

    MUST: drop `embed_tokens_per_layer` — unless the 35 split tables are the PLE's only source of
    truth, the quantization `include` predicates ({@link is_int8_module} / {@link is_int4_module})
    leave "an embedding neither of them hits", and the coverage of the targets can no longer be
    claimed. forward never referenced it, so dropping it leaves the graph unchanged.
    """
    del model.model.embed_tokens_per_layer
    return Gemma4Wrapper(model, tables).eval()


def materialize_non_persistent_buffers(model: nn.Module) -> None:
    """Materialize the non-persistent buffers of a model built on meta through upstream's
    initialization (`initialize_weights`).

    Non-persistent buffers (RoPE's `*_inv_freq` / `*_original_inv_freq`, `embed_scale`) are not in
    the state_dict, so `load_state_dict` does not fill them. Upstream `_init_weights` computes
    their values deterministically from the config — the same procedure as upstream
    `from_pretrained`'s meta load (move to `empty_like`, then `initialize_weights`), the same one
    path that writes the values post_init writes when the model is built on CPU (the formula is not
    copied here). The parameters are still on meta, so the random initialization passes through
    without materializing anything.

    MUST: the materialized holders are filled with NaN — if non-persistent buffers that upstream
    `_init_weights` does not write appear, NaN remains and {@link assert_materialized} fails (the
    undefined values of `empty` are never used silently).
    """
    for name, buffer in list(model.named_non_persistent_buffers()):
        parent, _, leaf = name.rpartition(".")
        model.get_submodule(parent).register_buffer(
            leaf, torch.full_like(buffer, float("nan"), device="cpu"), persistent=False
        )
    model.initialize_weights()


def assert_materialized(model: nn.Module) -> None:
    """Check that a model built on meta and then loaded has no slot without an entity, no
    uninitialized buffer and no undone tie.

    `load_state_dict(assign=True)` sets a new `Parameter` per key, so the tied `lm_head.weight` and
    the main embedding split into **separate Parameters** (sharing memory). Left split,
    torch.export emits them as two initializers (the container changes), so the tie's identity is
    checked too.
    """
    tensors = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    on_meta = [name for name, tensor in tensors if tensor.is_meta]
    if on_meta:
        raise AssertionError(f"slots left on meta (not filled by the load): {on_meta}")
    unset = [
        name for name, buffer in model.named_non_persistent_buffers() if bool(buffer.isnan().any())
    ]
    if unset:
        raise AssertionError(
            f"non-persistent buffers {unset} not filled by upstream's initialization"
            " (buffers that upstream _init_weights does not handle may have been added)"
        )
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise AssertionError(
            "the tied lm_head.weight is split into a Parameter apart from the main embedding"
        )


def load_model_and_tables(
    model_dir: Path, *, kv_residues: bool = True
) -> tuple[nn.Module, nn.ModuleList]:
    """Return the text model with the real weights read in f32 and the 35-way PLE split (the
    check slot's table still loaded).

    It does **not include** building the wrapper ({@link build_wrapper}) or demoting the RoPE
    buffers because the decode script ({@link gemma4.export_decode}) uses the same material with
    another wrapper form and another RoPE form (replaced by an inlet for host-supplied values). How
    the material is read and the three equivalence checks (the PLE table's row count, dropping the
    KV-shared layers' residues, the 35-way split's bit identity) are closed in one place.

    MUST: the model is built on `meta` and **owns** {@link renamed_state}'s f32 entities **as is**
    via `assign=True`. Built on CPU and then copied with `load_state_dict`, the randomly
    initialized f32 entities and the state_dict are alive at the same time at the moment of the
    load, and every weight but the PLE is held twice (about 35 GiB for E4B — overflowing a machine
    with 31 GiB of RAM). The non-persistent buffers are restored by
    {@link materialize_non_persistent_buffers}, the tie by upstream `tie_weights`, and
    {@link assert_materialized} checks them.

    `kv_residues` states whether the checkpoint carries the k / v residues of the KV-shared layers:
    the regular E2B / E4B checkpoints do, Google's q4_0 QAT checkpoints
    (`gemma4.export_product_qat`) carry none. MUST: the expectation is checked either way — with
    residues at least one is dropped (upstream's ignore list still matches the layering), without
    them none is found; the strict load then checks that every other key matches the model.
    """
    from transformers import Gemma4ForCausalLM

    register_attention()
    config = load_text_config(model_dir)
    model_file = model_dir / CHECKPOINT_FILE
    rows, _ = ple.per_layer_table_shape(model_file, PLE_CHECKPOINT_KEY)
    if rows != int(config.vocab_size_per_layer_input):
        raise ValueError(
            f"PLE table rows {rows} differ from the config's vocab_size_per_layer_input"
            f" {config.vocab_size_per_layer_input}"
        )
    # The rows the split's equivalence check uses are scattered points: block boundaries, both
    # ends and the middle ({@link gemma4.ple.probe_rows}).
    probe = ple.probe_rows(rows)
    config.vocab_size_per_layer_input = ple.PLE_PROBE_ROWS
    config._attn_implementation = ATTENTION_NAME

    with torch.device("meta"):
        model = Gemma4ForCausalLM(config).eval()
    materialize_non_persistent_buffers(model)
    state, dropped = renamed_state(model, model_file, probe)
    if kv_residues and not dropped:
        raise ValueError(
            "not a single k/v residue of the KV-shared layers was dropped"
            " (upstream's _keys_to_ignore_on_load_unexpected is empty — the layering premise may"
            " have changed)"
        )
    if not kv_residues and dropped:
        raise ValueError(
            f"{model_file}: {len(dropped)} k/v residues of the KV-shared layers ({dropped[:4]})"
            " in a checkpoint expected to carry none"
        )
    model.load_state_dict(state, assign=True)
    del state
    model.tie_weights()
    assert_materialized(model)
    tables = ple.load_per_layer_tables(
        model_file,
        PLE_CHECKPOINT_KEY,
        int(config.num_hidden_layers),
        int(config.hidden_size_per_layer_input),
    )
    ple.assert_per_layer_split(model, tables, probe)
    return model, tables


def load_wrapper(model_dir: Path) -> Gemma4Wrapper:
    """Read the real weights in f32 and return an exportable wrapper with the RoPE buffers
    demoted."""
    model, tables = load_model_and_tables(model_dir)
    # Left as a buffer, inv_freq is not a leaf of constant folding, and sin / cos remain in the
    # IR.
    assert_rope_lifted(model, "gemma4")
    return build_wrapper(model, tables)


def quantize_wrapper(wrapper: nn.Module) -> tuple[Int8Report, Int4Report, dict[str, torch.Tensor]]:
    """Round the embeddings to i8 and the linears to i4, and return the merged scale ledger (the
    order ADR 0006 makes a MUST).

    The ledger holds the tied entity's key **in both spellings**. torch.export folds the tied
    `model.model.embed_tokens.weight` / `model.lm_head.weight` into one initializer, but which FQN
    remains depends on export's internal order (measured: `model.lm_head.weight`). An extra key is
    simply not looked up by `emit`, so holding both works either way.
    """
    int8 = fake_quant_int8(wrapper, include=is_int8_module)
    int4 = fake_quant_int4(wrapper, include=is_int4_module)
    scales: dict[str, torch.Tensor] = {**int8.scales, **int4.scales}
    scales[f"{LM_HEAD_MODULE}.weight"] = int8.scales[f"{EMBED_TOKENS_MODULE}.weight"]
    return int8, int4, scales


def load_tokenizer(model_dir: Path) -> Any:
    """Read the official `tokenizer.json` directly with `tokenizers` (not through the transformers
    layer)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def bos_token_id(model_dir: Path, tokenizer: Any) -> int:
    """Look up `bos_token` of `tokenizer_config.json` as an id (the BOS convention of the module
    docstring)."""
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    token = config["bos_token"]
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"the tokenizer has no bos_token {token!r}")
    return int(token_id)


def assert_case_lengths(
    cases: Sequence[tuple[str, torch.Tensor]], sym_max: int, window: int
) -> None:
    """Check that the goldens' T fit the symbolic dimension and that **at least one case is
    longer than the window**.

    MUST: with goldens of `T <= window` only, the band mask equals causal and the sliding layers'
    window is never evaluated (the numbers match but the acceptance loses its meaning — half the
    reason this series exists).
    """
    for name, ids in cases:
        length = int(ids.shape[1])
        if not 2 <= length <= sym_max:
            raise ValueError(
                f"{name}: T={length} is outside the symbolic dimension's range [2, {sym_max}]"
            )
    longest = max((int(ids.shape[1]) for _, ids in cases), default=0)
    if longest <= window:
        raise ValueError(
            f"the longest case's T={longest} does not exceed sliding_window {window}"
            " (the band mask equals causal and the sliding layers go unverified)"
        )


def encode_cases(
    model_dir: Path, cases: Sequence[tuple[str, str]] = GOLDEN_CASES
) -> tuple[tuple[str, torch.Tensor], ...]:
    """The golden cases' `(name, input_ids)` with the leading `<bos>`, unchecked — {@link
    build_cases} adds the length rules of the PLE series; 12B's script (`gemma4.export_unified`)
    passes its own case list and states its own rules."""
    tokenizer = load_tokenizer(model_dir)
    bos = bos_token_id(model_dir, tokenizer)
    return tuple(
        (
            name,
            torch.tensor(
                [[bos, *tokenizer.encode(text, add_special_tokens=False).ids]], dtype=torch.int64
            ),
        )
        for name, text in cases
    )


def build_cases(model_dir: Path, sym_max: int, window: int) -> tuple[tuple[str, torch.Tensor], ...]:
    """The golden cases' `(name, input_ids)` (`<bos>` is added here — the module docstring)."""
    cases = encode_cases(model_dir)
    assert_case_lengths(cases, sym_max, window)
    return cases


def _attention_depth(config: Any, layer_type: str) -> int:
    """head_dim per layer type (sliding = `head_dim` / full = `global_head_dim`)."""
    if layer_type == SLIDING_ATTENTION:
        return int(config.head_dim)
    if layer_type == FULL_ATTENTION:
        return int(config.global_head_dim)
    raise AssertionError(
        f"unknown layer_type '{layer_type}' (no counterpart among the mask dictionary's keys)"
    )


def _mask_constant(graph: IrGraph, node: Any, where: str, sym_max: int) -> str:
    """Follow attention's mask input and return the name of the Tmax constant (fails when folding
    did not take effect)."""
    producer = {out: item for item in graph.nodes for out in item.outs}
    source = producer.get(node.ins[3])
    if source is None or source.op != SYM_PREFIX_SLICE_OP:
        found = (
            "not a node output (a graph input or a direct initializer)"
            if source is None
            else source.op
        )
        raise AssertionError(
            f"{where}: the mask comes from {found} — not the Tmax constant's {SYM_PREFIX_SLICE_OP}"
        )
    constant = source.ins[0]
    if constant not in graph.initializers:
        raise AssertionError(f"{where}: the mask's source '{constant}' is not an initializer")
    shape = declared_shape(graph, constant)
    if shape != [1, 1, sym_max, sym_max]:
        raise AssertionError(
            f"{where}: the mask constant's shape {shape} differs from the Tmax form"
            f" [1, 1, {sym_max}, {sym_max}]"
        )
    return constant


def assert_ir_form(
    graph: IrGraph, config: Any, sym_max: int, storage_expectation: Mapping[str, int]
) -> dict[str, Any]:
    """Check that the IR has "the true MQA form", "per-layer-type masks that are T-independent
    constants" and "mixed storage in effect".

    Each is a property that **breaks silently with the numbers still matching**, so the golden io
    match does not catch it:

    - materializing `repeat_kv` turns the attention's Hkv into H
    - a mask left as a graph input means "the host builds T² every time and passes it" (against
      the intent of ADR 0010)
    - the causal constant reaching a sliding layer (or the reverse) gives the same numbers for
      goldens with T ≤ 512
    - a weight that misses the compression eligibility stays **silently f32** (the default side of
      `emit._plan_weight_dtype` has a path that drops it quietly), so it goes unnoticed unless the
      storage dtypes are counted

    MUST: look up head_dim per layer type ({@link _attention_depth}) — checking every layer with
    one of the values leaves the layers of the other type entirely unchecked.
    """
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    layer_types = list(config.layer_types)
    layers = int(config.num_hidden_layers)
    if len(layer_types) != layers:
        raise AssertionError(
            f"config.layer_types has {len(layer_types)} entries (not the {layers} layers)"
        )

    names = [spec.name for spec in graph.inputs]
    if names != [INPUT_IDS]:
        raise AssertionError(
            f"graph inputs are {names} — not the single `{INPUT_IDS}`"
            " (the mask may have stayed an input instead of being folded)"
        )
    if len(graph.outputs) != 1:
        raise AssertionError(f"{len(graph.outputs)} IR outputs (the one-shot logits is one output)")

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(
            f"{len(attentions)} attention nodes (not matching the {layers} layers)"
        )

    constants: dict[str, set[str]] = {layer_type: set() for layer_type in layer_types}
    for index, node in enumerate(attentions):
        layer_type = layer_types[index]
        where = f"attention[{index}] ({layer_type})"
        if len(node.ins) != 4:
            raise AssertionError(f"{where}: {len(node.ins)} ins (expected q / k / v / mask, 4)")
        query, key, value, _ = (declared_shape(graph, name) for name in node.ins)
        if query[1] != heads or key[1] != kv_heads or value[1] != kv_heads:
            raise AssertionError(
                f"{where}: head axes are {[query[1], key[1], value[1]]} —"
                f" not the true GQA form {[heads, kv_heads, kv_heads]}"
                " (k / v widened to H means repeat_kv was materialized)"
            )
        depth = _attention_depth(config, layer_type)
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: D axes are {[query[3], key[3], value[3]]} — not this layer type's"
                f" head_dim {depth}"
            )
        constants[layer_type].add(_mask_constant(graph, node, where, sym_max))

    for layer_type, found in constants.items():
        if len(found) != 1:
            raise AssertionError(
                f"{layer_type} has {len(found)} mask constants (one per layer type expected):"
                f" {sorted(found)}"
            )
    shared = {name for found in constants.values() for name in found}
    if len(shared) != len(constants):
        raise AssertionError(
            f"layer types {sorted(constants)} share the mask constants {sorted(shared)}"
            " (causal and band folded into the same constant = the window has no effect)"
        )

    storage: dict[str, int] = {}
    for initializer in graph.initializers.values():
        dtype = initializer.storage.dtype
        storage[dtype] = storage.get(dtype, 0) + 1
    wrong = {
        dtype: (storage.get(dtype, 0), expected)
        for dtype, expected in storage_expectation.items()
        if storage.get(dtype, 0) != expected
    }
    if wrong:
        raise AssertionError(
            f"storage dtype counts differ from the expectation (measured, expected): {wrong}"
            f" / full breakdown {storage}"
        )
    return {
        "attention_nodes": len(attentions),
        "heads": [heads, kv_heads, kv_heads],
        "head_dim": {layer_type: _attention_depth(config, layer_type) for layer_type in constants},
        "mask_constants": {layer_type: sorted(found)[0] for layer_type, found in constants.items()},
        "storage": dict(sorted(storage.items())),
    }


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Write each case's inputs and torch CPU expected outputs to `io.<case>.safetensors`.

    The second return value is the logits for the sanity check (kept as `[1, T, vocab]` — the
    last position is looked up from it).
    """
    written: list[str] = []
    logits: dict[str, torch.Tensor] = {}
    for name, ids in cases:
        with torch.no_grad():
            output = wrapper(ids)
        args = {INPUT_IDS: ids}
        # MUST: the io is also lowered to the IR's semantic dtype representation (i64 → i32).
        # Unless it matches the form the runtime receives, the Deno-side E2E cannot read the golden
        # (ADR 0009's boundary normalization).
        tensors = {
            f"{INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        tensors[f"{OUTPUT_PREFIX}0"] = normalize_boundary_tensor(
            output.detach().contiguous(), f"{name}: output 0"
        )
        path = out_dir / f"{IO_PREFIX}{name}{IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        logits[name] = output.detach()
    return written, logits


def greedy_tokens(logits: Mapping[str, torch.Tensor]) -> dict[str, int]:
    """The top token id at **the last position** of each case."""
    return {name: int(value[0, -1].argmax()) for name, value in logits.items()}


def expected_token_ids(
    tokenizer: Any, expectations: Mapping[str, str] = GREEDY_EXPECTATIONS
) -> dict[str, int]:
    """The expected continuations of {@link GREEDY_EXPECTATIONS} → token ids (fails unless each
    is a single token).

    `expectations` is another model's table (12B's —
    `gemma4.export_unified.FIRST_TOKEN_EXPECTATIONS`).
    """
    expected: dict[str, int] = {}
    for name, continuation in expectations.items():
        ids = tokenizer.encode(continuation, add_special_tokens=False).ids
        if len(ids) != 1:
            raise AssertionError(
                f"{name}: the expected continuation {continuation!r} is {len(ids)} tokens {ids}"
                " (unless it is a single token, 'top-1 match' has no defined meaning)"
            )
        expected[name] = int(ids[0])
    return expected


def _sanity(
    greedy: Mapping[str, int], expected: Mapping[str, int], labels: Mapping[int, str]
) -> dict[str, Any]:
    """Check that the top token at the last position matches the expected continuation, and that
    it is not constant across cases.

    MUST: a mismatch fails. A check on norms or shapes alone cannot detect "the numbers move but it
    is broken as a language model" (mixed-up layers, the mask's direction, a RoPE shift, wrong PLE
    layers).
    MUST: identical top tokens for every case fail. It overlaps with the expectation match, but it
    stands independently as the line that detects "constant output", which remains even if the
    expectation table is loosened.
    """
    if set(greedy) != set(expected):
        raise AssertionError(
            f"case sets disagree: greedy={sorted(greedy)} expected={sorted(expected)}"
        )

    def show(token: int) -> str:
        return f"{token}({labels.get(token, '?')})"

    wrong = {
        name: (show(token), show(expected[name]))
        for name, token in greedy.items()
        if token != expected[name]
    }
    if wrong:
        raise AssertionError(
            "the top token at the last position differs from the expected continuation"
            f" (measured, expected): {wrong}"
        )
    if len(set(greedy.values())) == 1:
        raise AssertionError(
            "the top token at the last position is the same for every case"
            f" ({show(next(iter(greedy.values())))}) — constant output"
        )
    return {name: show(token) for name, token in sorted(greedy.items())}


def export_series(model_dir: Path, out_dir: Path, *, sym_max: int = SYM_MAX) -> dict[str, Any]:
    """Write the IR container and the golden io, and return a summary.

    MUST: the outputs are written to a staging slot and put in place only after **every check**
    (the form checks, the sanity check) passes. Placed at final before the checks, a failed run
    leaves "an asset that can pass the acceptance checks" — the io goldens are taken from the same
    broken wrapper, so they agree with each other, and the TS-side match **turns green** (the
    spelling of "when it may be published" is {@link _shared.decode_series._publish}; the rules of
    swapping into place and cleaning up are the core primitive
    {@link karume.artifacts.staged_publication}).
    """
    wrapper = load_wrapper(model_dir)
    # MUST: rounding comes before the reference and the goldens are taken (ADR 0006) — afterwards,
    # only the reference would run on the original weights.
    int8, int4, scales = quantize_wrapper(wrapper)
    cases = build_cases(model_dir, sym_max, wrapper.sliding_window)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # The example input is the longest case (the closer T is to the bound, the farther from the
    # 0/1 specialization). min=2 avoids the 0/1 specialization, and max is the very evaluation
    # point of Tmax folding (ADR 0010 — not managed twice through a separate knob).
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim("T", min=2, max=sym_max)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive creates none — it is only handed a
        # path).
        staged.mkdir()
        graph = export_to_file(
            wrapper,
            (example_ids,),
            staged / MODEL_FILE,
            provenance=PROVENANCE,
            # The graph name is **the component name** (= the key in karume.json's weights). It is
            # not derived from the directory name — neither the series name (`gemma4-e2b-product`)
            # nor the staging slot's name matches the component name (container-v1 §2.1).
            graph_name=GEMMA4_ROLE,
            dynamic_shapes=({1: seq},),
            preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        form = assert_ir_form(
            graph,
            wrapper.model.config,
            sym_max,
            {"i8": len(int8.scales), "i4": len(int4.scales)},
        )
        written, logits = _write_io(wrapper, graph, cases, staged)
        tokenizer = load_tokenizer(model_dir)
        greedy = greedy_tokens(logits)
        labels = {token: tokenizer.id_to_token(token) for token in set(greedy.values())}
        expected = expected_token_ids(tokenizer)
        labels.update({token: tokenizer.id_to_token(token) for token in set(expected.values())})
        # MUST: evaluated before publication (the only non-tautological check of this series — if
        # it fails, the staging slot goes with it).
        sanity = _sanity(greedy, expected, labels)
    return {
        "dir": str(out_dir),
        "nodes": len(graph.nodes),
        "outputs": len(graph.outputs),
        "initializers": len(graph.initializers),
        "model_bytes": sum(p.stat().st_size for p in container_parts(out_dir / MODEL_FILE)),
        "ops": sorted(graph.required_ops),
        "symbols": list(graph.symbols),
        "io": written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        "sanity": sanity,
    }


#: The dest that carries the model-dependent defaults (dest → `model → value`) on the parser.
#: **Not a CLI argument** (set with `set_defaults` and removed by {@link resolve_series_args}).
MODEL_DEFAULTS = "model_defaults"


def add_model_default(
    parser: argparse.ArgumentParser, dest: str, default: Callable[[str], object]
) -> None:
    """Register `dest`'s default as derived from `--model` (it applies only when unset = `None`).

    The defaults of `--out` / `--reference` depend on the chosen model's series
    (`gemma4-<model>-…`), but argparse fixes a default to one value before parsing. The default is
    left `None` and filled from the model after parsing.
    """
    defaults = dict(parser.get_default(MODEL_DEFAULTS) or {})
    defaults[dest] = default
    parser.set_defaults(**{MODEL_DEFAULTS: defaults})


def series_parser(
    description: str,
    out_dir: Callable[[str], Path],
    *,
    with_sym_max: bool = True,
    models: Sequence[str] = tuple(model for model in GEMMA4_UPSTREAM if model in GEMMA4_PLE_MODELS),
) -> argparse.ArgumentParser:
    """The CLI skeleton shared by this family's scripts (`--model` / `--model-dir` / `--out` /
    `--sym-max`).

    What differs per series is only the default output location (`out_dir` = model name →
    location) and the `--steps` / `--reference` that only the chunk series add
    ({@link gemma4.export_decode.run_variant_cli}). If spellings or defaults split per entry point,
    "the same knob under a different name per script" grows.

    `--model` defaults to E2B (`GEMMA4_DEFAULT_MODEL` — the default paths and behaviour without it
    are the same as when this was E2B only). Omitting `--model-dir` / `--out` gives the chosen
    model's locations ({@link checkpoint_dir} / `out_dir`).

    `with_sym_max=False` is for scripts without a chunk symbol (the MTP drafter) — `--sym-max`
    itself is not registered, so that a knob with no effect is not accepted and silently discarded
    (fail loudly). `models` is the set of selectable models, narrowed when a script has defaults
    that are only right for particular models (the drafter).
    The default is the models with per-layer embeddings (E2B / E4B): every script that reads the
    PLE refuses 12B at the argument instead of deep in the load; 12B has its own script
    (`gemma4.export_unified`).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", choices=tuple(models), default=GEMMA4_DEFAULT_MODEL)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    add_model_default(parser, "model_dir", checkpoint_dir)
    add_model_default(parser, "out", out_dir)
    if with_sym_max:
        parser.add_argument("--sym-max", type=int, default=SYM_MAX)
    return parser


def resolve_series_args(args: argparse.Namespace) -> dict[str, Any]:
    """Fold the parse result into "dest → value, with the model-dependent defaults filled"
    (`model` and the carrier removed).

    `--model` is a knob only for deciding the default locations and is not passed to
    `export_series` — the scripts derive the form from the material (`model_dir`) and the config,
    so nothing reads the model name itself.
    """
    values = dict(vars(args))
    defaults = values.pop(MODEL_DEFAULTS)
    model = values.pop("model")
    for dest, default in defaults.items():
        if values[dest] is None:
            values[dest] = default(model)
    return values


def run_series_cli(
    parser: argparse.ArgumentParser,
    run: Callable[..., dict[str, Any]],
    argv: Sequence[str] | None,
) -> None:
    """Resolve the CLI, call `run(model_dir, out, **the remaining knobs)` and print the summary
    JSON.

    MUST: everything but `--model` / `--model-dir` / `--out` is passed **as is, by name** — this
    requires argparse's dest and each `export_series` keyword name to be spelled the same, and the
    handover needs no additions however many knobs a series has.
    """
    options = resolve_series_args(parser.parse_args(argv))
    model_dir = options.pop("model_dir")
    out = options.pop("out")
    summary = run(model_dir, out, **options)
    print(json.dumps({"model_dir": str(model_dir), **summary}, indent=1, ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    run_series_cli(series_parser(__doc__.split("\n\n")[0], series_dir), export_series, argv)


if __name__ == "__main__":
    main()
