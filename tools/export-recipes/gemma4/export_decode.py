"""Script that writes the real-weight Gemma 4 E2B / E4B (`--model e2b|e4b`, default e2b) as a
**states-form chunk graph** (`model.krm` (IR v2) + goldens).

The scaffold that verifies ADR
[0066](../../../docs/decisions/0066-generation-context-state-slots.md) (GenerationContext and
named state slots) /
[0067](../../../docs/decisions/0067-autoregressive-attention-vocabulary.md) decisions 4–5b
(attention with state references and `state_append`) /
[0068](../../../docs/decisions/0068-decode-exit-multi-output.md) decision 4 (the decode exit) /
[0069](../../../docs/decisions/0069-packed-w4-storage.md) (packed w4) on a real model that has
**sliding + KV sharing + mixed quantization** at once. The same frame without KV sharing is
`minicpm5/export_decode.py`. The model-independent checks (how greedy is taken, the margin check,
the capacity check) have their source of truth in {@link _shared.decode_series}, and the one-shot
helpers are in {@link gemma4.export}; both are imported and reused (one rule is not written in two
places).

    uv run --with 'transformers==5.14.1' python -m gemma4.export_decode              # E2B
    uv run --with 'transformers==5.14.1' python -m gemma4.export_decode --model e4b  # E4B

## This script is also the core of the two chunk series

The states-form chunk graph has two kinds of exit — the logits opt-in form (this script's series)
and the token-only default exit (the series of {@link gemma4.export_token} / ADR 0068 decision 4).
The two differ **only in the exit's form and in whether goldens are taken**; reading the material,
host-supplied RoPE, the KV-sharing surgery, mixed quantization and the order of the checks are the
same, so the path is closed into one, driven by {@link ChunkVariant}
({@link load_wrapper} / {@link export_series} / {@link run_variant_cli}).
The token-only script holds only one wrapper, the variant descriptor and the entry point.

MUST: the differences take only the form of the variant's four fields. Moving the path back into
the scripts would let adding a check to one silently weaken the other (the same reason as the
module docstring of `_shared.decode_series`; when they really were verbatim copies, a state arose
in which only one of them had the margin floor's MUST).

## What goes into the graph

One **chunk wrapper**:
`(input_ids[1,M], rope_{sliding,full}_attention_{cos,sin}[1,M,headDim]) →
(logits[1,M,262144], token[1,M,1])` (the token-only variant also takes `last_row`). The positions
themselves are not an input (the section below). `M` is the physical chunk dimension
(ADR 0067 decision 4 — `chunkLength` for prefill / 1 for decode), and
the valid rows are the compact prefix of the first `queryLength` rows. `token` is
`logits.argmax(-1, keepdim=True)` (the decode exit of ADR 0068 decision 4). **The output order is
fixed as `[logits, token]`**, and the runtime's slot numbers read that order.

## No positions in the graph; RoPE is a derived input from the host (difference 1 from one-shot)

The one-shot form has no `position_ids` — positions are always `0..T-1`, so cos / sin fall through
the constant folding of `arange` into a Tmax table + `sym_prefix_slice` (ADR 0010). In decode the
position is `pastLength + row` and **is decided only at run time**, so this folding does not hold.

So `model.model.rotary_emb` is replaced with {@link RopeInputs} (an inlet; not a line of the
upstream modeling code is touched). Unlike the Llama family, Gemma 4's rotary is called per layer
type as **`forward(x, position_ids, layer_type)`** (modeling_gemma4.py:1713), so the inlet takes
one pair per layer type as well. The wrapper's forward binds the cos / sin it receives into
`RopeInputs`, and upstream uses those values as `position_embeddings` as is. As a result the graph
inputs become `input_ids` + **4 RoPE inputs** (2 layer types × cos / sin, f32 `[1, M, headDim]`),
and both the `position_ids` input and the table initializers disappear from the shipped IR. The
host (TS side) builds the tables; the distribution declares only **the formula's parameters**
(theta / headDim / rotaryDim) and the position limit ({@link gemma4.rope}). The full layers' RoPE
is `proportional` (`partial_rotary_factor` 0.25) with 192 zero frequencies, but they only enter the
rows as columns with cos = 1 / sin = 0, so they need no special handling.

The tables the script itself needs (example inputs, io / greedy goldens, sanity) are built with
**the same formula** as TS ({@link rope_args}) — so that the acceptance checks never read expected
sequences taken with another formula.

MUST: `karume.rope.assert_rope_lifted` is **not applied** — that check demotes the `inv_freq`
buffers to folding leaves, and this script replaces the whole rotary module, so `inv_freq`
disappears from the model. Instead {@link assert_ir_form_decode} checks **the absence of `sin`
ops** directly (`cos` is not in the IR vocabulary, so a graph with RoPE left in fails export
itself — the only reachable residue is on the `sin` side).

## KV-shared layers "read the owner layer's slot" (difference 2 from the one-shot form)

The last `num_kv_shared_layers` layers share KV (the last 20 of E2B's 35 layers, the last 18 of
E4B's 42). Upstream puts the k / v of **the last layer of that layer_type** among the layers before
the sharing start into `shared_kv_states[layer_type]`, and the shared layers use it as is
(modeling_gemma4.py:1287 / 1735). So there are slots **for the owner layers** only (layers 0..14 ×
k/v, 30 slots, for E2B; layers 0..23, 48 slots, for E4B), and a shared layer reads the slot of
layer 13 for E2B / 22 for E4B if sliding, of layer 14 for E2B / 23 for E4B if full
({@link kv_owner_layers} — the numbers are derived from the config; the script holds none).

The surgery (`karume.states.to_states_form`) accepts several registrations for one slot and checks
that the derived shape / `window` / **the append's input name** match exactly (`_register` in
states.py). That a shared layer's k / v are **the same value tensors** as the owner layer's is a
fact the traced graph holds, so this check proves "the assignment is not wrong" (if it were, the
input names would disagree and it would fail).
There are as many `state_append` as owner-layer slots (E2B 30 / E4B 48), each placed **right after
the last reader of its slot** (the consistency of the order and the window declarations is the job
of `verify._assert_state_order` — not copied here).

## The window width 512 can be declared as is (the semantics are strictly equivalent)

karume's sliding predicate is
`in_window(col, limit) = col <= limit && (limit - col) < params.window`
(packages/runtime/src/kernels/state-attention.ts:234-235, `limit = pastLength + row`). HF's
`sliding_window_mask_function` is causal AND `kv_idx > q_idx - sliding_window`
(masking_utils.py:98-99), and both mean "the distance `limit - col` is `0 <= dist < 512`" —
**strictly equivalent at width 512, self included**. It is the same inclusion as the one-shot
form's band mask (`(cols <= rows) & ((rows - cols) < window)`), so `config.sliding_window` is
declared as the `window` attrs as is.

The capacity is split per layer type (ADR 0066 addendum 9): the full slots take the symbol `C`
(chosen when the context is created — decision 3), the sliding slots the number
`window + SLIDING_SLACK_ROWS` ({@link sliding_capacity}). The ring closes at this capacity, so left
as a symbol, `C - (window + SLIDING_SLACK_ROWS)` rows would always lie dead (about 180MiB at
C=8192). The reason for the slack rows is in {@link SLIDING_SLACK_ROWS}, the details in
{@link states_plan}.

## The mask is there "only to get through the trace"

The fourth input of `attention` is the same two-layer-type additive mask dictionary as in the
one-shot form ({@link gemma4.export.additive_causal_mask} / `additive_sliding_mask`), but in the
states form both causality and the window become **predicate computations**, so no mask tensor is
needed. The surgery drops the mask input from every layer's attention and prunes the two Tmax²
constants and `sym_prefix_slice`. {@link assert_ir_form_decode} checks that not one of those
residues remains (if one did, "4.5MiB nobody reads" would sit in the distribution).

## Mixed quantization is the same as one-shot (embeddings i8 × linears i4)

Rounding comes **before the reference and the goldens are taken** (ADR 0006), and both the target
assignment and the storage overrides use {@link gemma4.export.quantize_wrapper} as is. That is why
{@link DecodeChunkWrapper} **derives** from the one-shot wrapper — unless the module FQN space is
identical, neither the target predicates (`is_int8_module` / `is_int4_module`) nor the scale
ledger's keys can be reused.

## Output layout

    outputs/series/gemma4-<model>-decode/model.krm                 weights, constants, descriptors
    outputs/series/gemma4-<model>-decode/io.<case>.safetensors     unpadded full-length io
    outputs/series/gemma4-<model>-decode/greedy.<case>.safetensors expected greedy K-step sequence

`io.*` uses the same key convention as the one-shot form (`input.<graph input name>` /
`output.<position>`). `greedy.*` is **the source of truth of decode acceptance**, holding `prompt`
i32 `[T]` / `expected` i32 `[K]` / `margin` f32 `[K]`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn
from torch.export import Dim

from _shared.decode_series import _write_greedy, assert_case_room, positions_for
from gemma4 import export as one_shot
from gemma4 import ple, provenance
from gemma4 import rope as rope_math
from gemma4.distribution import GEMMA4_DEFAULT_MODEL, GEMMA4_ROLE
from karume.artifacts import staged_publication
from karume.container import AssetInput, container_parts
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, normalize_boundary_tensor
from karume.ir import IrGraph, IrNode
from karume.ops import ARGMAX_OP, ATTENTION_OP, LINEAR_OP, STATE_APPEND_OP
from karume.pipeline import export_module, publish_model
from karume.shapes import declared_shape
from karume.states import StateAttentionSpec, StatesPlan, to_states_form

#: This script's series suffix (`gemma4-<model>-decode/` — a directory apart from the one-shot
#: series, a separate asset with another graph form). It is also where the series that take no
#: goldens (token-only / product) find the greedy expected sequences they reuse (the default of
#: `--reference`).
DECODE_SUFFIX = "decode"

#: The default location of the default model's (E2B) outputs. The material (the default of
#: `--model-dir`) is the same as for the one-shot form, and the shared CLI skeleton holds the
#: spelling ({@link gemma4.export.series_parser}).
DEFAULT_OUT_DIR = one_shot.series_dir(GEMMA4_DEFAULT_MODEL, DECODE_SUFFIX)

#: The graph input's name (the chunk wrapper's forward argument name). The source of truth
#: {@link assert_ir_form_decode} uses to check "only this spelling".
INPUT_IDS = one_shot.INPUT_IDS

#: The spelling of the derived RoPE inputs (`rope_<layer type>_<cos|sin>`) and the order of layer
#: types that fixes their order. **They are the wrapper's forward argument names themselves**
#: (torch.export takes them as the graph input names), and the TS side's `gemma.Pipeline` binds
#: them by name.
ROPE_INPUT_PREFIX = "rope_"
ROPE_PARTS: tuple[str, ...] = ("cos", "sin")
ROPE_LAYER_TYPES: tuple[str, ...] = (one_shot.SLIDING_ATTENTION, one_shot.FULL_ATTENTION)


def rope_input_name(layer_type: str, part: str) -> str:
    """The spelling of one derived RoPE input (the only source of {@link ROPE_INPUTS} and of the
    forward argument names)."""
    return f"{ROPE_INPUT_PREFIX}{layer_type}_{part}"


#: The four derived RoPE inputs (the order MUST equal the argument order of
#: {@link DecodeChunkWrapper.forward}).
ROPE_INPUTS: tuple[str, ...] = tuple(
    rope_input_name(layer_type, part) for layer_type in ROPE_LAYER_TYPES for part in ROPE_PARTS
)

#: The physical chunk dimension's symbol name (the IR name is decided by `symbol_names` — spelled
#: the same as the Dim name).
SEQ_SYMBOL = "M"

#: The state slots' capacity symbol (ADR 0066 decision 2 / addendum 7 — bound at
#: `createGenerationContext`). **A states-only symbol that never appears in a value shape**, the
#: slot that keeps the capacity from being baked at export time.
CAPACITY_SYMBOL = "C"

#: The slack added to `window` for the sliding slots' physical rows (= the draft limit kmax of
#: speculative verification).
#:
#: WHY: a speculative-decoding verify run writes rows **ahead of** the logical length into the
#: ring. Rejected rows do not advance the logical length, so their physical rows remain until the
#: next run overwrites the same positions. The row mapping is `slot_row(col) = col % capacity` (the
#: runtime also uses capacity — not window), so with a capacity of exactly window, the logical
#: column `P+i−window` that a rejected row `P+i` overwrites can fall inside the live window
#: `[P+a+2−window, …)` (a = the accepted count). With a capacity of `window + kmax`, the
#: overwritten column is always `P+i−window−kmax < P+a+2−window`, outside the window, and the set
#: of readable rows is unchanged.
#: MUST: the `window` attrs stay as they are (the slack concerns the physical rows only; the width
#: of the window read is unchanged).
SLIDING_SLACK_ROWS = 8

#: The number of continuation steps of the greedy golden (exercising ADR 0068 decision 4's exit
#: over several steps).
#: Per model: E4B stops at 10 because capital-ja's 11th step has margin 0.0205, under
#: {@link MARGIN_FLOOR} (steps 1-10 stay >= 0.31; measured 2026-10-30). Dropping the case instead
#: would leave only " Paris" expectations, which the constant-output check forbids (ruling 47).
#: 12B starts at E2B's 16 (unmeasured until its first export; `gemma4.export_unified` takes the
#: same table and the same `--steps` knob, and its margin gate names every offending step). 26B
#: (`gemma4.export_moe`) starts at 16 the same way, unmeasured until its first export.
GREEDY_STEPS: Mapping[str, int] = {"e2b": 16, "e4b": 10, "12b": 16, "26b": 16}

#: The lower bound of the top1 − top2 logit difference at each step. It guarantees that GPU
#: execution's deviation does not split the greedy sequence, and it is **not tautological** (a
#: case whose continuation is not strongly determined fails here). A failing case is removed from
#: {@link GREEDY_CASES} instead of lowering K — a sequence without margin as a golden makes a check
#: that is "red even when the GPU side is right".
#:
#: MUST: it sits **above** the acceptance check's premise (`e2e_gemma4_greedy_test.ts` —
#: `minMargin > 2 × atol`, atol = 1e-2). Below it, a case "the script takes but the check's
#: premise fails" becomes possible, and the check's "this fails only when the script and the
#: asset disagree" no longer holds (Codex review round H, finding H-04).
MARGIN_FLOOR = 2.5e-2

#: The cases greedy goldens are taken for (a subset of {@link gemma4.export.GOLDEN_CASES}).
#: io goldens are taken for all three cases (they cover one step, so they need no margin check).
#:
#: NOTE: currently **a provisional all-cases selection** (minicpm5 has a set fixed after
#: measurement). Cases {@link assert_greedy_margins} rejects in a real run with w4 are removed
#: from here and the run repeated — the check applies once after every case is measured, so one
#: real run collects every case to remove.
#: MUST: never a set whose `GREEDY_EXPECTATIONS` expectations reduce to one kind after removal
#: (`_sanity`'s constant-output detection line would become tautological — pinned by the tests).
GREEDY_CASES: tuple[str, ...] = ("capital-en", "capital-ja", "context-en")

#: Residue ops the surgery must kill. `sym_prefix_slice` is the mask's Tmax folding (ADR 0010),
#: and `sin` the survivor when RoPE fell neither into folding nor into the derived inputs.
#: NOTE: `cos` is not in the IR vocabulary (ops.py's UNARY_OPS has `sin` only), so a form with cos
#: left fails export and never reaches here — the check lists only the two that can be reached.
RESIDUE_OPS = (one_shot.SYM_PREFIX_SLICE_OP, "sin")


def slot_name(layer: int, part: str) -> str:
    """A state slot name with the layer number (`l0.k` / `l0.v` …).

    The spelling is closed in one place — if the surgery's specification ({@link states_plan}) and
    the form check ({@link assert_ir_form_decode}) built it separately, a spelling drift would pass
    as "the slot exists but nobody reads it".
    """
    return f"l{layer}.{part}"


class RopeInputs(nn.Module):
    """**An inlet** with the same calling convention as `Gemma4TextRotaryEmbedding`.

    Upstream calls `self.rotary_emb(hidden_states, position_ids, layer_type)` per layer type
    (modeling_gemma4.py:1713) and receives a 2-tuple `(cos, sin)` (each `[B, seq, head_dim]`).
    This module satisfies only that convention, and replaces the trigonometry from `inv_freq` with
    **the derived inputs the wrapper received, as they are**. `head_dim` differs per layer type
    (sliding 256 / full 512), so they are bound one pair per type too.

    The handover is a temporary attribute binding ({@link bound}) because upstream's calling
    convention is fixed as `(x, position_ids, layer_type)`, with no gap to pass host-supplied
    values as arguments. export runs with `strict=False` (`karume.pipeline.export_module`), so
    Python runs as is and the bound input tensors are traced as ordinary data flow.

    MUST: no parameters and no buffers — the tables **are no longer part of the distribution** (a
    remaining initializer would bring back both the quantization eligibility and the meaning of
    `maxPosition`).
    MUST: the binding lasts one forward only (always removed in `finally`). Carried over, a next
    call running on **the previous positions' tables** would pass with shape and type matching.
    MUST: unbound or unknown layer types fail loudly. Falling back to a default would silently pass
    "full layers rotating at the sliding frequencies" (values come out, but at other angles).
    """

    def __init__(self, layer_types: Sequence[str]) -> None:
        super().__init__()
        self.layer_types = tuple(layer_types)
        self._tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @contextmanager
    def bound(self, tables: Mapping[str, tuple[torch.Tensor, torch.Tensor]]) -> Iterator[None]:
        """Bind layer type → `(cos, sin)` for the duration of one forward."""
        missing = sorted(set(self.layer_types) - set(tables))
        if missing:
            raise KeyError(f"cos / sin not passed for layer types {missing}")
        self._tables = dict(tables)
        try:
            yield
        finally:
            self._tables = {}

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bound `(cos, sin)` (`x` / `position_ids` are arguments of the convention —
        their values are not read)."""
        del x, position_ids
        tables = self._tables.get(layer_type)
        if tables is None:
            raise KeyError(
                f"cos / sin of layer_type '{layer_type}' are not bound"
                f" (bound: {sorted(self._tables)})"
            )
        return tables


def unique_layer_types(config: Any) -> tuple[str, ...]:
    """`config.layer_types` deduplicated **in order of appearance** (closes the traversal order of
    the inputs and the checks in one place)."""
    return tuple(dict.fromkeys(config.layer_types))


def rope_specs(config: Any) -> dict[str, rope_math.RopeLayerSpec]:
    """Layer type → RoPE formula ({@link gemma4.rope.rope_specs} plus a width match).

    MUST: check that the derived `head_dim` equals the attention head_dim
    ({@link gemma4.export._attention_depth}). Upstream's rotary reads `global_head_dim` only on the
    branch for full + `proportional`, so a changed rope_type silently drops only the table width to
    256 — "the width does not fit, but the declaration can be written".
    """
    specs = rope_math.rope_specs(config)
    for layer_type, spec in specs.items():
        depth = one_shot._attention_depth(config, layer_type)
        if spec.head_dim != depth:
            raise AssertionError(
                f"layer type '{layer_type}': RoPE width {spec.head_dim} differs from the attention"
                f" head_dim {depth} (rotary reads its head_dim from the wrong place)"
            )
    return specs


def rope_args(
    specs: Mapping[str, rope_math.RopeLayerSpec], positions: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Positions `[1, T]` → the four derived inputs forward takes (in the order of
    {@link ROPE_INPUTS}).

    The formula mirrors the TS source of truth ({@link gemma4.rope.rope_rows}) — the goldens / io /
    example inputs the script takes are built with the same formula as the tables the distribution
    builds at run time.
    """
    unknown = sorted(set(ROPE_LAYER_TYPES) - set(specs))
    if unknown:
        raise AssertionError(f"no RoPE formula for layer types {unknown} (present: {list(specs)})")
    rows = [int(value) for value in positions.reshape(-1).tolist()]
    tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_type in ROPE_LAYER_TYPES:
        cos, sin = rope_math.rope_rows(specs[layer_type], rows)
        tables[layer_type] = (
            torch.from_numpy(cos).unsqueeze(0),
            torch.from_numpy(sin).unsqueeze(0),
        )
    return tuple(
        tables[layer_type][index]
        for layer_type in ROPE_LAYER_TYPES
        for index in range(len(ROPE_PARTS))
    )


def bound_rope(
    cos_sliding: torch.Tensor,
    sin_sliding: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """The four derived inputs → the per-layer-type pairs {@link RopeInputs.bound} takes.

    The spelling is closed in one place — if the three chunk wrappers each built it, one of them
    could swap cos and sin in its wiring (shape and type match, so only the numeric checks would
    show it).
    """
    return {
        one_shot.SLIDING_ATTENTION: (cos_sliding, sin_sliding),
        one_shot.FULL_ATTENTION: (cos_full, sin_full),
    }


def max_position(config: Any) -> int:
    """The position limit the model declares (`text_config.max_position_embeddings`).

    Now that RoPE is no longer baked into tables, "the largest absolute position that can be looked
    up" is decided by the model's declaration alone, not by an asset (the same source as the
    distribution's `pipelineConfig.maxPosition` — `gemma4.distribution.gemma4_max_position`).
    """
    declared = int(config.max_position_embeddings)
    if declared < 1:
        raise AssertionError(
            f"config.max_position_embeddings is {declared} (not a positive integer)"
        )
    return declared


class DecodeChunkWrapper(one_shot.Gemma4Wrapper):
    """The chunk-form version (`(input_ids, 4 RoPE inputs) → (logits, token)`) of the one-shot
    wrapper ({@link gemma4.export.Gemma4Wrapper}).

    MUST: held as **a subclass** (the same structure is not rewritten) — the quantization target
    predicates (`is_int8_module` / `is_int4_module`) and the scale ledger's keys sit on the module
    FQN space, so the `model` / `per_layer` spellings being identical to one-shot is the condition
    for reuse.
    MUST: the derived RoPE inputs' **argument names and order** equal {@link ROPE_INPUTS}
    (torch.export takes argument names as graph input names, so a spelling drift leaves the host
    unable to bind them).
    MUST: `position_ids` is not passed (`None`). Passing it brings a position input back into the
    shipped IR. With `None` upstream builds an `arange`, but the rotary has been replaced by the
    inlet and nobody reads it, so that `arange` falls away in export's reachability analysis (if it
    remained, it would fail loudly as an op outside the IR vocabulary).
    MUST: `attention_mask` is the same two-layer-type dictionary as for one-shot (omitted,
    transformers builds the mask and values derived from `cache_position` mix into the folding's
    leaves). It exists only to get through the trace, and the surgery drops it.
    MUST: `use_cache=False`. With True a `DynamicCache` appears and KV is returned as outputs,
    directly contradicting "state is not the graph's ordinary I/O" (ADR 0066 decision 1).
    MUST: the output order is `[logits, token]`. The runtime reads the two by slot number.
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = input_ids.shape[1]
        mask = {
            one_shot.FULL_ATTENTION: one_shot.additive_causal_mask(length),
            one_shot.SLIDING_ATTENTION: one_shot.additive_sliding_mask(length, self.sliding_window),
        }
        embeds = self.model.model.embed_tokens(input_ids)
        stacked = ple.per_layer_inputs(self.per_layer, input_ids, self.per_layer_scale)
        tables = bound_rope(
            rope_sliding_attention_cos,
            rope_sliding_attention_sin,
            rope_full_attention_cos,
            rope_full_attention_sin,
        )
        with self.model.model.rotary_emb.bound(tables):
            logits = self.model(
                inputs_embeds=embeds,
                per_layer_inputs=stacked,
                attention_mask=mask,
                position_ids=None,
                use_cache=False,
            ).logits
        return logits, logits.argmax(-1, keepdim=True)


class GreedyRopeView(nn.Module):
    """The `(input_ids, position_ids) → (logits, token)` surface `_shared.decode_series` calls.

    The greedy continuations and the io reference are taken as "given positions, look up the full
    length" (`_shared` does not know a family's graph inputs). Now that RoPE is a graph input, the
    tables must be built from those positions somewhere — this wraps it here instead of adding
    arguments to the chunk wrapper (the `_shared` rules are not rewritten for one family).

    MUST: the tables are built with **the same formula** as the distribution ({@link rope_args}).
    If the acceptance checks read expected sequences taken with another formula, a check would turn
    red over "a different way of building the tables", not over "the script and the asset
    disagree".
    """

    def __init__(
        self, wrapper: DecodeChunkWrapper, specs: Mapping[str, rope_math.RopeLayerSpec]
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.specs = dict(specs)

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.wrapper(input_ids, *rope_args(self.specs, position_ids))


@dataclass(frozen=True)
class ChunkVariant:
    """A descriptor holding **only the differences** of the two chunk series (logits opt-in /
    token-only).

    It carries the four points where {@link export_series} actually branches — the series
    location, the wrapper to build, the exit's form, and whether its own goldens are taken. The
    surgery, mixed quantization and the order of the checks are the same for both series, so no
    other field exists (a field would make "a path different for only one" writable).
    """

    #: The series name suffix (`gemma4-<model>-<suffix>/` — a separate directory per series, a
    #: separate asset with another graph form). The location depends on the model, so this holds
    #: only the model-independent suffix.
    suffix: str
    #: The chunk wrapper to build (one of {@link DecodeChunkWrapper}'s inheritance chain).
    wrapper: type[DecodeChunkWrapper]
    #: Whether the exit is token-only (one more input `last_row`, one output token — ADR 0068
    #: decision 4).
    token_only: bool
    #: Whether its own goldens (io + greedy) are taken. The token-only series takes none, as the
    #: acceptance checks reuse the greedy record of the logits opt-in series as is (the docstring
    #: of {@link gemma4.export_token}).
    #: NOTE: a field apart from `token_only`, but today only two combinations work — `goldens=True`
    #: with `token_only=True` fails with a `KeyError`, as `_write_io` cannot take `last_row` from
    #: `args`. The reverse combination (`goldens=False` with `token_only=False`) is actually used by
    #: {@link gemma4.export_product}, which reads only the wrapper field and does not go through
    #: `export_series`.
    goldens: bool

    @property
    def out_dir(self) -> Path:
        """The default model's (E2B) location (the location `--model` chooses is
        `gemma4.export.series_dir`)."""
        return one_shot.series_dir(GEMMA4_DEFAULT_MODEL, self.suffix)


#: This script's series (the logits opt-in form — emits the logits and token of all M rows).
DECODE = ChunkVariant(
    suffix=DECODE_SUFFIX, wrapper=DecodeChunkWrapper, token_only=False, goldens=True
)


def swap_rope_inputs(model: nn.Module) -> RopeInputs:
    """Replace `model.model.rotary_emb` with the inlet ({@link RopeInputs}).

    The layer types are taken in the config's **order of appearance** — a call with an undeclared
    layer type is failed by `RopeInputs.bound` / `forward` (no branch falls back to a default).
    """
    receiver = RopeInputs(unique_layer_types(model.config))
    model.model.rotary_emb = receiver
    return receiver


def load_wrapper(variant: ChunkVariant, model_dir: Path) -> DecodeChunkWrapper:
    """Read the real weights in f32 and return an exportable chunk wrapper with RoPE replaced by
    the inlet.

    Only the wrapper type built at the end changes with the variant — reading the material (the
    three equivalence checks included) and the RoPE replacement are the same for the chunk series.
    """
    model, tables = one_shot.load_model_and_tables(model_dir)
    # The check slot's PLE table is dropped for the same reason as in
    # {@link gemma4.export.build_wrapper} (the 35 split tables become the PLE's only source of
    # truth, so the quantization targets' coverage can be claimed).
    del model.model.embed_tokens_per_layer
    swap_rope_inputs(model)
    return variant.wrapper(model, tables).eval()


def first_shared_layer(config: Any) -> int:
    """The layer number where KV sharing starts (the same formula as upstream
    `Gemma4TextAttention.__init__`)."""
    return int(config.num_hidden_layers) - int(getattr(config, "num_kv_shared_layers", 0))


def kv_owner_layers(config: Any) -> dict[str, int]:
    """layer_type → the layer that writes that type's KV (the same rule as upstream
    `store_full_length_kv`).

    Upstream puts the k / v of "the **last** layer of that layer_type among the layers before the
    sharing start" into `shared_kv_states[layer_type]`, and the shared layers read it
    (modeling_gemma4.py:1287). For E2B sliding → layer 13 / full → layer 14, for E4B sliding →
    layer 22 / full → layer 23.

    MUST: fail when a layer_type on the shared side has no owner layer (upstream would hit a
    KeyError on `shared_kv_states`; the layering premise itself has broken).
    """
    layer_types = list(config.layer_types)
    boundary = first_shared_layer(config)
    owners: dict[str, int] = {}
    for layer, layer_type in enumerate(layer_types[:boundary]):
        owners[layer_type] = layer
    orphans = sorted(set(layer_types[boundary:]) - set(owners))
    if orphans:
        raise AssertionError(
            f"the shared layers' layer_type {orphans} has no layer writing it before the sharing"
            f" start {boundary} (the layering premise has broken)"
        )
    return owners


def slot_layers(config: Any) -> tuple[int, ...]:
    """Layer → the number of the **owner layer** of the KV slot that layer reads.

    Before the sharing start it is the layer itself, from it on the owner layer per layer_type
    ({@link kv_owner_layers}).
    """
    owners = kv_owner_layers(config)
    boundary = first_shared_layer(config)
    return tuple(
        layer if layer < boundary else owners[layer_type]
        for layer, layer_type in enumerate(config.layer_types)
    )


def attention_nodes(graph: IrGraph, config: Any) -> list[IrNode]:
    """The attention nodes in nodes order (after checking the premise itself that **order of
    appearance = layer order**).

    The check matches q's D axis against `config.layer_types`. Gemma 4's head_dim differs per layer
    type (sliding 256 / full 512), so if the order is not the layer order, the 512 of every fifth
    layer shows up at another position.

    MUST: building the spec with the premise broken lets through a slot assignment that reads
    another layer's KV **with shape and type matching** (a silent wrong value). A count mismatch
    fails for the same reason — a layer left out is not operated on and stays in mask form, and
    that layer alone becomes causal within the chunk only.
    """
    layer_types = list(config.layer_types)
    nodes = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(nodes) != len(layer_types):
        raise AssertionError(
            f"{len(nodes)} attention nodes (not matching the {len(layer_types)} layers)"
        )
    for layer, (node, layer_type) in enumerate(zip(nodes, layer_types, strict=True)):
        depth = one_shot._attention_depth(config, layer_type)
        found = declared_shape(graph, node.ins[0])[3]
        if found != depth:
            raise AssertionError(
                f"attention[{layer}] ({layer_type}): q's D axis is {found} —"
                f" not this layer type's head_dim {depth}"
                " (possibly the nodes order is not the layer order — the slot assignment's premise"
                " breaks)"
            )
    return nodes


def kv_heads(config: Any, layer_type: str) -> int:
    """The KV heads of one layer type — upstream's rule (`Gemma4TextAttention` and
    `Gemma4UnifiedTextAttention` alike): a full layer of an `attention_k_eq_v` model has
    `num_global_key_value_heads` (12B: 1, against 8 on its sliding layers), every other layer
    `num_key_value_heads`.

    NOTE: a config without `attention_k_eq_v` reads as `False` — the field's default in both
    upstream config classes (the E2B / E4B form, where both layer types share one count).
    """
    if getattr(config, "attention_k_eq_v", False) and layer_type == one_shot.FULL_ATTENTION:
        return int(config.num_global_key_value_heads)
    return int(config.num_key_value_heads)


def sliding_capacity(config: Any) -> int:
    """The physical row count of the sliding slots (`window + SLIDING_SLACK_ROWS`).

    The one place where the surgery's specification ({@link states_plan}) and the form check
    ({@link assert_ir_form_common}) read the same number. Spelling the formula in two places would
    let one go stale on the day the slack moves, as "a check that expects a capacity other than the
    real one".
    """
    return int(config.sliding_window) + SLIDING_SLACK_ROWS


def states_plan(
    graph: IrGraph, config: Any, *, capacity_symbol: str = CAPACITY_SYMBOL
) -> StatesPlan:
    """The surgery's specification reflecting layer types and KV sharing (slots **for the owner
    layers only**).

    - Layers before the sharing start (own KV — E2B 0..14, E4B 0..23): their own k / v slots.
      Sliding gets `window = sliding_window`, full no window.
    - Layers from the sharing start on (shared readers — E2B 15..34, E4B 24..41): no slot of their
      own; they read the slot of the layer_type's owner layer.
      The sliding readers declare the same `window` (`_register` in states.py checks the exact
      match).

    The capacity is split per layer type: **the sliding slots are the number
    `window + SLIDING_SLACK_ROWS`** (the ring closes at this capacity, not at window — the reason
    for the slack itself is in {@link SLIDING_SLACK_ROWS}; left as a symbol, C rows would be
    reserved and the rows beyond the slack would only lie dead, without changing the set of readable
    rows), **the full slots stay a symbol** (only the side that keeps the whole context needs the
    freedom to "choose the capacity at run time" — ADR 0066 decision 3).
    """
    nodes = attention_nodes(graph, config)
    window = int(config.sliding_window)
    capacity = sliding_capacity(config)
    return StatesPlan(
        capacity_symbol=capacity_symbol,
        attentions=tuple(
            StateAttentionSpec(
                output=node.outs[0],
                k_slot=slot_name(owner, "k"),
                v_slot=slot_name(owner, "v"),
                window=window if layer_type == one_shot.SLIDING_ATTENTION else None,
                capacity=capacity if layer_type == one_shot.SLIDING_ATTENTION else None,
            )
            for node, layer_type, owner in zip(
                nodes, config.layer_types, slot_layers(config), strict=True
            )
        ),
    )


#: The last_row input name of the token-only default exit (ADR 0068 decision 4). The spelling is
#: the wrapper's forward argument name itself (torch.export takes it as the graph input name) — the
#: source of truth is here, referenced by both `export_token` and
#: `assert_ir_form_decode(token_only=True)`.
TOKEN_ONLY_LAST_ROW = "last_row"


def last_rows_for(ids: torch.Tensor, rows: int) -> torch.Tensor:
    """The indices `[T−rows, …, T−1]` of **the last `rows` rows** of an unpadded full-length prompt
    (the `last_row` input).

    The value of the "one chunk eats the whole prompt" form paired with {@link positions_for}; the
    example input and the full-length sanity forward both need the same one (if the spelling split
    in two, one side alone could build a row selection that does not point at the last row). The
    product form's speculative verification selects several rows, so the row count is an argument.
    """
    length = int(ids.shape[1])
    if not 1 <= rows <= length:
        raise AssertionError(f"selected row count {rows} does not fit the prompt length {length}")
    return torch.arange(length - rows, length, dtype=torch.int64)


def last_row_for(ids: torch.Tensor) -> torch.Tensor:
    """The index `[T−1]` of the one last valid row (the `last_row` input of the token-only exit).

    {@link last_rows_for} with one row — delegated so that the index arithmetic is not spelled in
    two places.
    """
    return last_rows_for(ids, 1)


def assert_layer_type_count(config: Any) -> None:
    """Check the declared count of layer types (the **preamble** of the form checks — all three
    forms pass it first).

    If this is broken, a later `zip(..., strict=True)` fails with an exception far from the real
    cause, "the counts differ", so it comes before the entry / exit checks (the order of
    {@link assert_ir_form_decode} / {@link gemma4.export_product.assert_ir_form_product}).
    """
    layers = int(config.num_hidden_layers)
    declared = len(list(config.layer_types))
    if declared != layers:
        raise AssertionError(f"config.layer_types has {declared} entries (not the {layers} layers)")


def assert_rope_inputs(graph: IrGraph, config: Any, *, seq_symbol: str = SEQ_SYMBOL) -> None:
    """Check that the four derived RoPE inputs are f32 `[1, M, headDim]`, with each layer type's
    width.

    MUST: check the width too. Wiring that swaps sliding (256) and full (512) shows only in the
    width of the tables the host passes — the graph side fails only once the declared input shapes
    disagree.
    MUST: also check that no table initializer remains. A form that added the derived inputs but
    kept the tables (wiring where both work) brings 6MiB of resident memory back, and the meaning
    of `maxPosition` moves back to the asset.
    """
    declared = {spec.name: spec for spec in graph.inputs}
    specs = rope_specs(config)
    for layer_type in ROPE_LAYER_TYPES:
        depth = specs[layer_type].head_dim
        for part in ROPE_PARTS:
            name = rope_input_name(layer_type, part)
            spec = declared.get(name)
            if spec is None:
                raise AssertionError(f"graph input '{name}' is missing (RoPE is not host-supplied)")
            expected = [1, seq_symbol, depth]
            if spec.dtype != "f32" or list(spec.shape) != expected:
                raise AssertionError(
                    f"graph input '{name}' is {spec.dtype} {list(spec.shape)} —"
                    f" not f32 {expected} (disagrees with the layer type's headDim)"
                )
    residue = sorted(
        name
        for name, initializer in graph.initializers.items()
        if rope_math.BAKED_TABLE_INFIX in initializer.tensor
    )
    if residue:
        raise AssertionError(
            f"{len(residue)} RoPE table initializers remain: {residue[:4]}"
            " (not yet moved fully to host generation)"
        )


def assert_row_selected_lm_head(
    graph: IrGraph,
    producer: Mapping[str, IrNode],
    start: IrNode,
    *,
    rows: int | str = 1,
) -> None:
    """**Pinning the row-selected lm_head** (Codex review round H, finding H-01).

    The per-row lm_head and the row selection commute, so even "lm_head on every row → softcap →
    row selection" gives the same token sequence — what ADR 0068 achieves (applying lm_head only to
    the selected rows, the `[M,V]` buffer gone) can be pinned only by this structural check. From
    `start` (the argmax node for token-only / the producer of the logits output for the product
    form) it walks back along the softcap chain (div/tanh/mul — `ins[0]` is the main stream in
    each) and requires that the first linear is lm_head, and that its input is `[1, rows, H]` and
    has the `last_row` input among its ancestors.

    `rows` is the expected number of selected rows. The token-only exit always has one row
    (`last_row[1]`), the product form the R rows of the speculative verification's symbol
    (`last_row[R]`) — only the counting differs, and the rule "the row selection sits before
    lm_head" is the same, so the check stays one and takes it as an argument.
    """
    node = start
    for _ in range(8):
        source = producer.get(node.ins[0])
        if source is None:
            raise AssertionError(
                f"the output's ancestry ('{node.ins[0]}') breaks off — lm_head (linear) is not"
                " reached"
            )
        node = source
        if node.op == LINEAR_OP:
            break
    else:
        raise AssertionError(f"no lm_head ({LINEAR_OP}) within 8 steps of the output")
    row_shape = declared_shape(graph, node.ins[0])
    if list(row_shape[:2]) != [1, rows]:
        raise AssertionError(
            f"lm_head's input is {row_shape} — not [1,{rows},H] (the {rows} selected rows)"
            " (it has regressed to lm_head on every row)"
        )
    input_names = {spec.name for spec in graph.inputs}
    frontier = [node.ins[0]]
    seen: set[str] = set()
    reachable: set[str] = set()
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in input_names:
            reachable.add(name)
            continue
        upstream = producer.get(name)
        if upstream is not None:
            frontier.extend(upstream.ins)
    if TOKEN_ONLY_LAST_ROW not in reachable:
        raise AssertionError(
            f"lm_head's input does not depend on `{TOKEN_ONLY_LAST_ROW}`"
            f" (inputs reached: {sorted(reachable)})"
            " — the row selection does not sit before lm_head"
        )


def assert_ir_form_common(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    *,
    seq_symbol: str,
    capacity_symbol: str,
    extra_symbols: Sequence[str] = (),
) -> dict[str, Any]:
    """Every rule **except** the entry / exit (the body shared by the three forms).

    The decode series has three entry / exit forms (logits opt-in / token-only / product), but the
    states form, the correspondence with the layer types, the slot assignment, the symbols, the
    surgery's residues and the storage breakdown are **the same in every form**. Unless this is one
    function, the checks of one form thin out (when the product form was added, the same rules were
    in fact spelled independently and 90 lines were duplicated).

    It lists only properties the golden match does not catch:

    - if even one attention remains in the old form (4 inputs with the mask), that layer alone does
      not see the past
    - if the presence of `window` disagrees with the layer type, a full layer drops what is outside
      the window / a sliding layer ignores the window
    - a wrong slot assignment makes a shared layer read another layer's KV (shape and type match)
    - a full slot capacity that is a number instead of a symbol cannot be chosen when the context
      is created (ADR 0066 decision 3). Conversely, unless the sliding slots are exactly the number
      `window + SLIDING_SLACK_ROWS`, too little slack lets speculative verification's rejected rows
      overwrite the live window, and too much brings dead rows back
      ({@link states_plan} / {@link SLIDING_SLACK_ROWS} — the reasons differ per layer type, so the
      checks are split too)
    - a remaining `sym_prefix_slice` / `sin` leaves Tmax constants nobody reads or unfolded residues
      in the distribution
    - a weight that misses the compression eligibility stays **silently f32** (the default side of
      `emit._plan_weight_dtype` has a path that drops it quietly), so it goes unnoticed unless the
      storage dtypes are counted

    MUST: look up head_dim per layer type (`hidden_size / num_attention_heads` = 192 is neither).
    Checking every layer with one of the values leaves the layers of the other type entirely
    unchecked.
    MUST: the caller has run {@link assert_layer_type_count} **first** (`zip(..., strict=True)` is
    set up on the premise that the counts match).

    `extra_symbols` are the symbols a form adds (the product form's row symbol R — the speculative
    verification's `last_row[R]`). The symbol set is checked for an **exact match**, so a per-form
    difference can only come in as an argument (loosening the set check to a subset would let a
    symbol that cannot be bound slip through).
    """
    heads = int(config.num_attention_heads)
    layer_types = list(config.layer_types)
    layers = int(config.num_hidden_layers)
    owners = slot_layers(config)
    window = int(config.sliding_window)
    capacity = sliding_capacity(config)
    # The KV heads are per layer type ({@link kv_heads}): one count for E2B / E4B, sliding 8 /
    # full 1 for 12B — reading one count for every layer would leave one type unchecked.
    kv_by_type = {layer_type: kv_heads(config, layer_type) for layer_type in set(layer_types)}

    attentions = [node for node in graph.nodes if node.op == ATTENTION_OP]
    if len(attentions) != layers:
        raise AssertionError(
            f"{len(attentions)} attention nodes (not matching the {layers} layers)"
        )
    for layer, (node, layer_type, owner) in enumerate(
        zip(attentions, layer_types, owners, strict=True)
    ):
        where = f"attention[{layer}] ({layer_type})"
        if len(node.ins) != 3:
            raise AssertionError(
                f"{where}: {len(node.ins)} ins — the states form takes exactly q / k / v"
                " (4 means the old form with the mask remains)"
            )
        expected_states = {"k": slot_name(owner, "k"), "v": slot_name(owner, "v")}
        if node.states != expected_states:
            raise AssertionError(
                f"{where}: states field is {node.states} (expected {expected_states})"
            )
        expected_window = window if layer_type == one_shot.SLIDING_ATTENTION else None
        if node.attrs.get("window") != expected_window:
            raise AssertionError(
                f"{where}: the window attr is {node.attrs.get('window')}"
                f" (expected {expected_window}) — the layer type / window correspondence broke"
            )
        query, key, value = (declared_shape(graph, name) for name in node.ins)
        kv = kv_by_type[layer_type]
        if query[1] != heads or key[1] != kv or value[1] != kv:
            raise AssertionError(
                f"{where}: head axes are {[query[1], key[1], value[1]]} —"
                f" not the true GQA form {[heads, kv, kv]}"
                " (k / v widened to H means repeat_kv was materialized)"
            )
        depth = one_shot._attention_depth(config, layer_type)
        if query[3] != depth or key[3] != depth or value[3] != depth:
            raise AssertionError(
                f"{where}: D axes are {[query[3], key[3], value[3]]} — not this layer type's"
                f" head_dim {depth}"
            )

    boundary = first_shared_layer(config)
    expected_slots = sorted(
        slot_name(layer, part) for layer in range(boundary) for part in ("k", "v")
    )
    appends = [node for node in graph.nodes if node.op == STATE_APPEND_OP]
    written = sorted(node.states["slot"] for node in appends)
    if written != expected_slots:
        raise AssertionError(
            f"`{STATE_APPEND_OP}` writes {len(written)} slots — not one each of the"
            f" {len(expected_slots)} owner-layer slots"
            f" (missing {sorted(set(expected_slots) - set(written))} /"
            f" surplus {sorted(set(written) - set(expected_slots))})"
        )
    if sorted(graph.states) != expected_slots:
        raise AssertionError(
            f"states declarations are {sorted(graph.states)} — not the k / v of the {boundary}"
            f" owner layers ({len(expected_slots)} slots)"
        )
    for layer in range(boundary):
        layer_type = layer_types[layer]
        depth = one_shot._attention_depth(config, layer_type)
        # The capacity check is split per layer type, a MUST — loosening it to all-symbol lets "an
        # asset with every capacity baked as a number" (the capacity cannot be chosen at run time —
        # ADR 0066 decision 3) through, and loosening it to all-number silently removes full's
        # capacity freedom. Sliding is the number window + slack ({@link sliding_capacity}).
        sliding = layer_type == one_shot.SLIDING_ATTENTION
        slot_shape = [1, kv_by_type[layer_type], capacity if sliding else capacity_symbol, depth]
        for part in ("k", "v"):
            name = slot_name(layer, part)
            slot = graph.states[name]
            if slot.dtype != "f32" or list(slot.shape) != slot_shape:
                kind = (
                    f"sliding is exactly window {window} + slack {SLIDING_SLACK_ROWS} as a number"
                    if sliding
                    else "full capacity stays a symbol, a MUST"
                )
                raise AssertionError(
                    f"states['{name}'] is {slot.dtype} {list(slot.shape)} —"
                    f" not f32 {slot_shape} ({kind})"
                )

    symbols = sorted(graph.symbols)
    expected_symbols = sorted({seq_symbol, capacity_symbol, *extra_symbols})
    if symbols != expected_symbols:
        raise AssertionError(f"symbols are {symbols} — not {expected_symbols}")
    residue = sorted(set(graph.required_ops) & set(RESIDUE_OPS))
    if residue:
        raise AssertionError(
            f"ops the surgery should have removed remain: {residue}"
            " (residue of the mask's Tmax folding / RoPE left unfolded)"
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
    triples = {
        layer_type: [heads, kv_by_type[layer_type], kv_by_type[layer_type]]
        for layer_type in unique_layer_types(config)
    }
    return {
        "attention_nodes": len(attentions),
        "state_append_nodes": len(appends),
        "slots": len(expected_slots),
        "kv_owners": kv_owner_layers(config),
        # One `[q, k, v]` triple when the layer types agree (the E2B / E4B record), else one per
        # layer type in the config's order.
        "heads": next(iter(triples.values())) if len(set(kv_by_type.values())) == 1 else triples,
        "head_dim": {
            layer_type: one_shot._attention_depth(config, layer_type)
            for layer_type in unique_layer_types(config)
        },
        "window": window,
        "storage": dict(sorted(storage.items())),
    }


def assert_ir_form_decode(
    graph: IrGraph,
    config: Any,
    storage_expectation: Mapping[str, int],
    *,
    seq_symbol: str = SEQ_SYMBOL,
    capacity_symbol: str = CAPACITY_SYMBOL,
    token_only: bool = False,
) -> dict[str, Any]:
    """Check the form of a states-form decode graph (every property that **breaks silently with
    the numbers still matching**).

    `token_only` is ADR 0068 decision 4's **default exit form** (`export_token` — the inputs gain
    `last_row` and the output becomes one argmax token). The checks other than entry / exit
    (states / layer types / residues / storage breakdown) are the same for both forms, so this one
    function takes both (split into separate functions, the checks of one would thin out). The
    body is {@link assert_ir_form_common}, shared with the product form too
    ({@link gemma4.export_product.assert_ir_form_product}).

    What this looks at is only the entry / exit:

    - a mask left in the graph inputs is a different thing: "the host builds T² per chunk and
      passes it"
    - a remaining `position_ids` brings the positions back inside the graph (RoPE only half moved
      out)
    - an exit that is not argmax is not the decode exit (ADR 0068 decision 4)
    - with `token_only`, the row selection sits **before** lm_head
      ({@link assert_row_selected_lm_head})
    """
    assert_layer_type_count(config)

    names = [spec.name for spec in graph.inputs]
    expected_inputs = [INPUT_IDS, *ROPE_INPUTS, *([TOKEN_ONLY_LAST_ROW] if token_only else [])]
    if names != expected_inputs:
        raise AssertionError(
            f"graph inputs are {names} — not {expected_inputs}"
            " (mask or position_ids may have stayed inputs instead of being folded)"
        )
    assert_rope_inputs(graph, config, seq_symbol=seq_symbol)
    expected_outputs = 1 if token_only else 2
    if len(graph.outputs) != expected_outputs:
        kind = (
            "the token-only exit is one output, token"
            if token_only
            else "the decode exit is the two outputs logits / token"
        )
        raise AssertionError(f"{len(graph.outputs)} IR outputs ({kind})")
    producer = {out: node for node in graph.nodes for out in node.outs}
    token_source = producer.get(graph.outputs[-1])
    if token_source is None or token_source.op != ARGMAX_OP:
        found = "not a node output" if token_source is None else token_source.op
        raise AssertionError(
            f"output {len(graph.outputs) - 1} comes from {found} — not `{ARGMAX_OP}`"
            " (the decode exit of ADR 0068 decision 4)"
        )
    if token_only:
        assert_row_selected_lm_head(graph, producer, token_source)

    return assert_ir_form_common(
        graph,
        config,
        storage_expectation,
        seq_symbol=seq_symbol,
        capacity_symbol=capacity_symbol,
    )


def _write_container(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    path: Path,
    *,
    graph_name: str,
    weight_dtype: str,
    weight_scales: Mapping[str, torch.Tensor],
    weight_dtype_overrides: Mapping[str, str],
    assets: Mapping[str, AssetInput] = {},
) -> IrGraph:
    """Write and verify the operated graph (the three publishing steps are left to
    `pipeline.publish_model`).

    `export_to_file` runs export → write as one path with no gap to insert the surgery, so only the
    part from writing on is called through the core's entry point. **Neither the rules nor the
    atomicity are reimplemented** — the checks of the states section, order and shapes, the part
    split (container-v1 §4.2) and its swapping into place and cleanup are all the core's.

    Initializers killed by pruning (the mask's Tmax² constants) are dropped from the stored tensors
    too — `stored_model` requires an **exact match** of declarations and storage. The scale ledger
    (`weight_scales`) may keep entries for pruned weights (emit only looks up the planned ones), but
    `weight_dtype_overrides` **fails loudly on unknown keys** — a linear weight vanishing in the
    surgery must never happen, so that check stays in effect.
    """
    declared = {init.tensor for init in graph.initializers.values()}
    stored = {name: tensor for name, tensor in tensors.items() if name in declared}
    return publish_model(
        path,
        graph,
        stored,
        provenance=one_shot.PROVENANCE,
        graph_name=graph_name,
        weight_dtype=weight_dtype,
        weight_scales=weight_scales,
        weight_dtype_overrides=weight_dtype_overrides,
        assets=assets,
    )


def _write_io(
    wrapper: nn.Module,
    graph: IrGraph,
    cases: Sequence[tuple[str, torch.Tensor]],
    out_dir: Path,
    specs: Mapping[str, rope_math.RopeLayerSpec],
) -> list[str]:
    """Write each case's **unpadded full-length** inputs and outputs to `io.<case>.safetensors`.

    NOTE: this is **a reference table** of "the values torch gives for the same weights at the same
    positions", not the runtime's execution script. The decode series' graph runs in chunk form
    (`M = chunkLength`), so the Deno side splits the full length T into chunks, runs them, and
    matches **only the valid rows** against the corresponding rows of `output.0` / `output.1` here.
    What is pinned to 0 on pad rows is **only the output of states-form attention** (ADR 0066
    addendum 8), and the later MLP / lm_head write meaningless values on pad rows too — the graph
    outputs of pad rows **must not be read** (the reference has no counterpart either).
    """
    written: list[str] = []
    for name, ids in cases:
        rope_inputs = rope_args(specs, positions_for(ids))
        args = {INPUT_IDS: ids, **dict(zip(ROPE_INPUTS, rope_inputs, strict=True))}
        with torch.no_grad():
            outputs = wrapper(ids, *rope_inputs)
        tensors = {
            f"{one_shot.INPUT_PREFIX}{declared.name}": normalize_boundary_tensor(
                args[declared.name], f"{name}: input '{declared.name}'"
            )
            for declared in graph.inputs
        }
        for slot, output in enumerate(outputs):
            tensors[f"{one_shot.OUTPUT_PREFIX}{slot}"] = normalize_boundary_tensor(
                output.detach().contiguous(), f"{name}: output {slot}"
            )
        path = out_dir / f"{one_shot.IO_PREFIX}{name}{one_shot.IO_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
    return written


def _export_args(
    variant: ChunkVariant,
    ids: torch.Tensor,
    seq: Dim,
    specs: Mapping[str, rope_math.RopeLayerSpec],
) -> tuple[tuple[torch.Tensor, ...], tuple[Any, ...]]:
    """The example inputs and the matching `dynamic_shapes` (token-only adds one `last_row`).

    The derived RoPE inputs are `[1, M, headDim]` — their row count is the chunk row count itself,
    so they get the same symbolic dimension as `input_ids` (another symbol would make "tables with a
    different row count can be passed" declarable). `last_row` is a static `[1]` (no dynamic
    dimension) — a run-time scalar whose form is fixed at one element, so a symbolic dimension would
    make it something else, tied to M.
    """
    args: tuple[torch.Tensor, ...] = (ids, *rope_args(specs, positions_for(ids)))
    shapes: tuple[Any, ...] = tuple({1: seq} for _ in args)
    if variant.token_only:
        return (*args, last_row_for(ids)), (*shapes, None)
    return args, shapes


def export_series(
    variant: ChunkVariant,
    model_dir: Path,
    out_dir: Path,
    *,
    sym_max: int = one_shot.SYM_MAX,
    steps: int = GREEDY_STEPS[GEMMA4_DEFAULT_MODEL],
    reference: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    """Write the variant's states-form IR container (with io / greedy for a series that takes
    goldens), and return a summary.

    MUST: the outputs are written to a staging slot and put in place only after **every check**
    (the form checks, the margin check, the sanity check against the one-shot expectation table)
    passes. Placed at final before the checks, a failed run leaves "an asset that can pass the
    acceptance checks" (the rules of swapping into place and cleaning up are the core primitive
    {@link karume.artifacts.staged_publication}).
    MUST: only a series that takes goldens reads `steps` (the token-only series has no greedy
    record).
    MUST: only a series that **takes no** goldens reads `reference` — the location of the greedy
    goldens it reuses, whose digest the provenance record ({@link gemma4.provenance}) binds to the
    container.
    """
    wrapper = load_wrapper(variant, model_dir)
    # MUST: rounding comes before the reference and the goldens are taken (ADR 0006) — afterwards,
    # only the reference would run on the original weights.
    int8, int4, scales = one_shot.quantize_wrapper(wrapper)
    specs = rope_specs(wrapper.model.config)
    greedy_view = GreedyRopeView(wrapper, specs)
    cases = one_shot.build_cases(model_dir, sym_max, wrapper.sliding_window)
    greedy_cases = tuple(case for case in cases if case[0] in GREEDY_CASES)
    # The position limit is the model's declaration (no longer baked into tables, so the asset
    # does not decide it). io / sanity look up each prompt once as a whole (no positions for the
    # continuation are needed), so they are checked with steps=0.
    limit = max_position(wrapper.model.config)
    assert_case_room(cases, 0, limit)
    if variant.goldens:
        assert_case_room(greedy_cases, steps, limit)
    # MUST: the reused goldens are checked **before** entering the staging slot (a premise outside
    # the slot — if it fails, fail before starting an export of tens of minutes). It only reads;
    # not a byte is written to the reference series.
    reference_goldens = (
        {} if variant.goldens else provenance.assert_reference_goldens(reference, greedy_cases)
    )

    # The example input is the longest case (far from the symbolic dimension's 0/1
    # specialization). min=2 for the same reason; max is the evaluation point of the mask's Tmax
    # folding (the surgery prunes it, so it does not remain in the distribution, but the trace
    # goes through it).
    _, example_ids = max(cases, key=lambda case: case[1].shape[1])
    seq = Dim(SEQ_SYMBOL, min=2, max=sym_max)
    args, shapes = _export_args(variant, example_ids, seq, specs)
    print("[export] torch.export → convert", file=sys.stderr, flush=True)
    graph, tensors = export_module(
        wrapper,
        args,
        dynamic_shapes=shapes,
        symbol_names=(SEQ_SYMBOL,),
        preserved=PRESERVED_OP_PREFIXES_WITH_ATTENTION,
    )
    config = wrapper.model.config
    print("[export] surgery to the states form → write", file=sys.stderr, flush=True)
    surgical = to_states_form(graph, states_plan(graph, config))

    # The unit put in place is **"the whole series directory" for both series**. For the series
    # that takes goldens, so that a mix of a new model + old greedy cannot be produced; for the
    # one that takes none, so that a pair whose container and provenance record
    # ({@link gemma4.provenance}) disagree cannot be produced.
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with staged_publication(out_dir) as staged:
        # The writer creates the directory (the primitive creates none — it is only handed a
        # path).
        staged.mkdir()
        container = staged / one_shot.MODEL_FILE
        # Storage is the default i8 + each linear i4 (the reason for this direction is in the
        # one-shot script's docstring — the side that need not know the tied entity's FQN before
        # writing). There is no explicit f32 entry for RoPE any more (the tables are no longer
        # initializers, so they are outside the compression-eligible set).
        verified = _write_container(
            surgical,
            tensors,
            container,
            # The graph name is **the component name** (= the key in karume.json's weights). It is
            # not derived from the directory name — neither the series name (`gemma4-e2b-product`)
            # nor the staging slot's name matches the component name (container-v1 §2.1).
            graph_name=GEMMA4_ROLE,
            weight_dtype="i8",
            weight_scales=scales,
            weight_dtype_overrides=dict.fromkeys(int4.scales, "i4"),
        )
        form = assert_ir_form_decode(
            verified,
            config,
            {"i8": len(int8.scales), "i4": len(int4.scales)},
            token_only=variant.token_only,
        )

        if variant.goldens:
            print("[io] full-length forward", file=sys.stderr, flush=True)
            io_written = _write_io(wrapper, verified, cases, staged, specs)
            greedy_written, tokens, margins = _write_greedy(
                greedy_view, greedy_cases, staged, steps=steps, floor=MARGIN_FLOOR
            )
            first = {name: continuation[0] for name, continuation in tokens.items()}
        else:
            # A series that takes no goldens looks up the full length once, for the sanity check
            # only (the row selection is T−1).
            print("[sanity] full-length forward", file=sys.stderr, flush=True)
            first = {}
            for name, ids in cases:
                with torch.no_grad():
                    token = wrapper(ids, *rope_args(specs, positions_for(ids)), last_row_for(ids))
                first[name] = int(token[0, 0, 0])

        # Match the first continuation token against the one-shot script's expectation table (a
        # cross-mechanism match — the one-shot and chunk scripts are separate, so the top token
        # agreeing for the same weights and the same prompt cross-validates them). MUST: evaluated
        # before publication (if it fails, the staging slot goes with it).
        tokenizer = one_shot.load_tokenizer(model_dir)
        expected = {
            name: token
            for name, token in one_shot.expected_token_ids(tokenizer).items()
            if name in first
        }
        labels = {
            token: tokenizer.id_to_token(token)
            for token in set(first.values()) | set(expected.values())
        }
        sanity = one_shot._sanity(first, expected, labels)

        if not variant.goldens:
            # The provenance record goes into **the same staging slot** as the container (one swap
            # into place, so a pair of a new container + an old record cannot arise). The series
            # name it states is the name after placement — not the staging slot's name.
            print(
                "[provenance] checkpoint fingerprint → reference.json", file=sys.stderr, flush=True
            )
            record = provenance.build_record(out_dir, model_dir, reference_goldens)
            provenance.write_record(staged, record)

    # The pruned count and the three golden kinds are fields only of a series that takes goldens
    # (the token-only summary covers one container, so it does not grow).
    pruned: dict[str, Any] = (
        {"pruned_initializers": len(graph.initializers) - len(verified.initializers)}
        if variant.goldens
        else {}
    )
    written: dict[str, Any] = (
        {"io": io_written, "greedy": greedy_written, "greedy_steps": steps}
        if variant.goldens
        else {}
    )
    measured: dict[str, Any] = (
        {
            "margin_min": {name: min(values) for name, values in sorted(margins.items())},
            "continuation": {
                name: tokenizer.decode(continuation)
                for name, continuation in sorted(tokens.items())
            },
        }
        if variant.goldens
        else {}
    )
    # Conversely, the provenance record is a field only of a series that takes no goldens (the
    # bound goldens are visible in the summary too).
    bound: dict[str, Any] = {} if variant.goldens else {"reference": record["reference"]}
    return {
        "dir": str(out_dir),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        **pruned,
        "model_bytes": sum(
            p.stat().st_size for p in container_parts(out_dir / one_shot.MODEL_FILE)
        ),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        **written,
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "quantized": {"i8": int8.describe(), "i4": int4.describe()},
        "form": form,
        **bound,
        **measured,
        "sanity": sanity,
    }


def run_variant_cli(variant: ChunkVariant, description: str, argv: Sequence[str] | None) -> None:
    """Build and run the variant's CLI (the one entry point of the two chunk series).

    The skeleton is shared with the one-shot script ({@link gemma4.export.series_parser}) — the
    chunk series add only `--steps`, for the series that takes goldens, and `--reference` (the
    location of the reused golden series), for the one that takes none. The position limit is the
    model's declaration, so it is not a knob.
    """
    parser = one_shot.series_parser(
        description, partial(one_shot.series_dir, suffix=variant.suffix)
    )
    if variant.goldens:
        parser.add_argument("--steps", type=int, default=None)
        one_shot.add_model_default(parser, "steps", GREEDY_STEPS.__getitem__)
    else:
        # The reuse target is the logits opt-in series of **the same model** (another model's
        # goldens do not fail the prompt match — the tokenizer is the same, so the prompts agree).
        parser.add_argument("--reference", type=Path, default=None)
        one_shot.add_model_default(
            parser, "reference", partial(one_shot.series_dir, suffix=DECODE_SUFFIX)
        )
    one_shot.run_series_cli(parser, partial(export_series, variant), argv)


def main(argv: Sequence[str] | None = None) -> None:
    run_variant_cli(DECODE, __doc__.split("\n\n")[0], argv)


if __name__ == "__main__":
    main()
