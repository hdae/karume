"""Writes Gemma 4 26B A4B (the mixture of experts — ADR 0145) as the product graph with its greedy
goldens, and a model cut after its first layers with layer goldens.

    cd tools/export-recipes
    uv run --no-sync python -m gemma4.export_moe --plan          # trace only, no weight read
    uv run --no-sync python -m gemma4.export_moe                 # the product series
    uv run --no-sync python -m gemma4.export_moe --layers        # the cut model, 12 layers
    uv run --no-sync python -m gemma4.export_moe --layers 6 --routing-outputs \\
        --out ../../outputs/diag/<dir>                           # a cut of 6, routing outputs too

The weights and their config are the pinned snapshot of Google's QAT checkpoint (two shards and
their index), the tokenizer, the chat template and the generation defaults the pinned snapshot of
`google/gemma-4-26B-A4B-it` (`gemma4.sources.HUB_PINS` — two sources, each checked against its pin
before anything long runs). Every run prints per-stage wall time and peak RSS as JSON and runs
under a `MemAvailable` floor ({@link qwen3_5.export.MemoryWatch}).

## The graph: 12B's product contract

The wrapper, the inputs and the outputs are 12B's (`gemma4.export_unified.UnifiedProductWrapper`):
`input_ids[1,M]`, `soft_embeds[1,M,H]`, `soft_mask[1,M,1]`, the four RoPE inputs, `last_row[R]` →
`logits[1,R,V]`, `hidden[1,R,H]` (ADR 0145 decision 5: the soft inputs stay, unused until 26B's
vision lands, so that work reuses this container). The class is upstream's `gemma4_text`
(`Gemma4ForCausalLM`) with the expert block on: every one of the 30 layers runs a dense GeGLU MLP
and 128 experts of which the router picks 8. Attention is 12B's form — K = V on the full layers
(`gemma4.export_unified.assert_shared_kv_projection` checks the wiring), sliding slots of 8 KV
heads, full slots of 2 — and there is no PLE and no KV sharing.

The experts run through `karume::expert_linear` (`gemma4.experts` — the `karume` implementation of
transformers' `ExpertsInterface`, named in the config before the model is built): two of them per
layer, `gelu_tanh(gate) · up` between, the routing weights multiplied in and summed over the picks
({@link assert_expert_wiring} holds every one to a `topk` over the 128 experts). The router is
upstream's module; its `per_expert_scale[index]` is one `embedding` per layer over the scale as
`[E, 1]` (the form check counts them — `gemma4.export_product.router_scale_lookups`).

## Storage: one grid import, shared by the container and the reference

Every parameter of rank ≥ 2 is imported from the checkpoint's own q4_0 grid
(`gemma4.export_unified.grid_import`, a block of rows at a time — `assert_fixed_matrices` refuses
any other before a weight is read): the 235 linears (the router's projection included) and the
tied embedding as `i4` (f32 steps), the 60 stacked expert weights `[128, out, in]` as `i4f16`
(the codes of the `[E·out, in]` view, each step rounded to f16 once by the writer — ADR 0145
decision 3). The norms, `router.scale`, `per_expert_scale` and `layer_scalar` are the checkpoint's
BF16 values widened to F32.

The trace runs without weight data (12B's meta trace — about 101 GB of untouched virtual memory for
26B, never paged in), and the grid is imported **once** ({@link export_series}): the same packed
bytes and steps are handed to the writer and held by the torch reference
({@link reference_wrapper}).
The reference dequantises a linear at every read, holds the embedding as its imported grid in F32,
and holds each expert weight as a `gemma4.experts.PackedExperts` — the packed codes and the steps
rounded by the writer's own function — whose one expert is dequantised when a pick asks for it,
through the op's own eager body. So the goldens describe exactly what the container decodes to
(ADR 0006) and the trace and the reference share one per-pick summation. Estimated peak (recon of
2026-11-08, not measured): about 20 GiB, against about 28 GiB with 12B's second import.

## Goldens

The greedy continuations of {@link GREEDY_CASES} by full re-forward (`_shared.decode_series.
_write_greedy`, the margin floor of `gemma4.export_decode.MARGIN_FLOOR`) — 12B's cases and
first-token expectations, unmeasured on 26B until its first export (the gates name every case or
step that fails). 26B's generation defaults declare no `suppress_tokens`, so the logits are not
masked. The record (`reference.json`) names the experts implementation that ran
(`gemma4.experts.EXPERTS_RECORD` — ADR 0145 decision 6) and both sources, the weights by the index
and both shards.

The series also holds the **cut points** ({@link CUT_POINTS_PREFIX}): for every case, the
reference's output of each full-attention layer but the last (layers 5, 11, 17 and 23 — the layers
a cut may end on) at every row of the prompt. A cut model's export checks its own last layer
against them.

## The cut model (`--layers N`)

This machine's GPU (12 GB) cannot hold the model (ADR 0145 decision 6), so its tests run the model
cut after N layers (N ends on a full-attention layer: 6, 12, 18 or 24 — 12 by default, about 5.6 GiB
of weights). The cut container is the product graph of the cut config, and its goldens come from
the same reference builder on that config ({@link CUT_PREFIX}): for every case, teacher-forced on
the prompt, the logits and the hidden state of the last {@link CUT_ROWS} rows (the graph's
outputs), each layer's output at those rows, and each router's top {@link router_top} expert ids
and probabilities at every row (the 9th shows how close a pick was to a swap — ADR 0145 decision
6). MUST: the cut reference's output of its last layer equals the full reference's
({@link assert_cut_consistency}, `torch.equal` against the product series' cut points) — so the
cut goldens are the full model's first layers, not another model.

`--routing-outputs` also writes a diagnostic container, `routing/model.krm` (not a product
graph): the same graph with each layer's `topk` index added as an output after logits and hidden
({@link RoutingWrapper}), so a GPU run can compare its routing ids with the goldens'.

## Output layout

    outputs/series/gemma4-26b-product/model.krm                 the container (no assets)
    outputs/series/gemma4-26b-product/greedy.<case>.safetensors the greedy goldens
    outputs/series/gemma4-26b-product/cut-points.<case>.safetensors  the full layers' outputs
    outputs/series/gemma4-26b-product/reference.json            provenance
    outputs/series/gemma4-26b-cut<N>/model.krm                  the cut container
    outputs/series/gemma4-26b-cut<N>/cut.<case>.safetensors     the cut goldens
    outputs/series/gemma4-26b-cut<N>/routing/model.krm          (--routing-outputs)
    outputs/series/gemma4-26b-cut<N>/reference.json             provenance

MUST: transformers is imported inside functions.
"""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from _shared.decode_series import (
    GREEDY_SUFFIX,
    PROMPT_KEY,
    _write_greedy,
    assert_case_room,
    greedy_continuation,
    positions_for,
)
from gemma4 import experts as routed
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from gemma4 import export_product as product
from gemma4 import export_unified as unified
from gemma4 import provenance, sources
from gemma4.card import GEMMA4_MOE_MODELS
from gemma4.distribution import GEMMA4_PRODUCT_SUFFIX, gemma4_ple_dim
from gemma4.sources import ModelPins
from karume.artifacts import staged_publication
from karume.container import container_parts
from karume.convert import normalize_boundary_tensor
from karume.emit import FixedQuantizedWeight, unpack_int4
from karume.ir import IrGraph
from karume.ops import EXPERT_LINEAR_OP, TOPK_OP
from karume.quantize import dequantize_int4
from karume.shapes import declared_shape
from qwen3_5.export import MEMORY_FLOOR_GIB, MemoryWatch

#: The models this script writes (26B A4B).
MODELS: tuple[str, ...] = tuple(sorted(GEMMA4_MOE_MODELS))

#: Checkpoint key prefixes outside the text decoder, which this script never reads (26B's vision
#: tower and its embedder — later work, ADR 0145 decision 1; the checkpoint has no audio keys).
IGNORED_PREFIXES: tuple[str, ...] = ("model.vision_tower.", "model.embed_vision.")

#: The golden cases: 12B's (`gemma4.export_unified.UNIFIED_CASES`), the window case included
#: (26B's window is 12B's 1024, so the same text overflows its ring).
GOLDEN_CASES: tuple[tuple[str, str], ...] = unified.UNIFIED_CASES

#: The cases whose greedy continuations become goldens, and the first-token sanity: 12B's, until
#: the first export measures 26B's margins (the floor's rule is to drop a failing case, as 12B
#: dropped `context-en`).
GREEDY_CASES: tuple[str, ...] = unified.UNIFIED_GREEDY_CASES
FIRST_TOKEN_EXPECTATIONS: Mapping[str, str] = unified.FIRST_TOKEN_EXPECTATIONS

#: The default depth of the cut model: 12 layers (about 5.6 GiB of weights and 0.24 GB of KV at
#: capacity 4096 — the recon of 2026-11-08), the deepest cut ending on a full layer that leaves
#: room on a 12 GB GPU being 18.
DEFAULT_CUT = 12

#: The rows of a cut golden: the last rows of each prompt (the graph's `last_row`).
CUT_ROWS = 4


def router_top(config: Any) -> int:
    """The router ranks a golden records per row: the k picks (8 for 26B) and the first one left
    out, whose probability shows how close the k-th pick was to a swap (ADR 0145 decision 6)."""
    return int(config.top_k_experts) + 1


#: The file prefixes of the full series' cut points and of a cut model's goldens (the suffix is the
#: greedy goldens' — `_shared.decode_series.GREEDY_SUFFIX`).
CUT_POINTS_PREFIX = "cut-points."
CUT_PREFIX = "cut."

#: Tensor keys of a cut golden ({@link write_cut_goldens}) and of a cut-points file.
ROWS_KEY = "rows"
LOGITS_KEY = "logits"
HIDDEN_KEY = "hidden"
LAYER_HIDDEN_KEY = "layer_hidden"
ROUTE_IDS_KEY = "route_ids"
ROUTE_PROBS_KEY = "route_probs"
LAYER_KEY_PREFIX = "layer."

#: The diagnostic container's directory inside a cut series (`--routing-outputs`).
ROUTING_DIR = "routing"


class MoeExportError(RuntimeError):
    """The checkpoint, the config or the graph is not what this script expects."""


# ---------------------------------------------------------------------------
# Config, checkpoint, trace
# ---------------------------------------------------------------------------


def load_config(model_dir: Path, layers: int | None = None) -> Any:
    """The checkpoint's text config, set to the GQA attention and the `karume` experts; `layers`
    keeps the first layers (`gemma4.export_unified.truncate_layers` — ending on a full layer).

    MUST: the class is `gemma4_text` with the expert block, no per-layer embeddings and no KV
    sharing — the form this script exports (12B and E2B / E4B have their own scripts).
    """
    config = one_shot.load_text_config(model_dir)
    where = str(model_dir / "config.json")
    if config.model_type != unified.MOE_TEXT_MODEL_TYPE:
        raise MoeExportError(
            f"{where}: text model type {config.model_type!r}, not {unified.MOE_TEXT_MODEL_TYPE!r}"
        )
    if not getattr(config, "enable_moe_block", False):
        raise MoeExportError(f"{where}: no expert block (`enable_moe_block`) — not this form")
    if gemma4_ple_dim(config, where) != 0:
        raise MoeExportError(f"{where}: declares per-layer embeddings — not this script's form")
    if int(getattr(config, "num_kv_shared_layers", 0) or 0):
        raise MoeExportError(f"{where}: declares KV-shared layers — not this script's form")
    unified.truncate_layers(config, layers)
    one_shot.register_attention()
    config._attn_implementation = one_shot.ATTENTION_NAME
    routed.use_karume_experts(config)
    return config


def open_checkpoint(model_dir: Path) -> unified.Checkpoint:
    """The sharded checkpoint, with 26B's ignored prefixes ({@link IGNORED_PREFIXES})."""
    return unified.Checkpoint(model_dir, IGNORED_PREFIXES)


def weights_files(checkpoint: unified.Checkpoint) -> tuple[str, ...]:
    """The files the record fingerprints from the weights' source: the index, both shards and
    `config.json` (`gemma4.provenance.files_by_role` — the pin must name exactly these)."""
    return (*checkpoint.files, "config.json")


def _trace(
    model_dir: Path,
    tokenizer_dir: Path,
    layers: int | None,
    sym_max: int,
    *,
    routing: bool = False,
) -> tuple[
    Any,
    tuple[tuple[str, torch.Tensor], ...],
    IrGraph,
    dict[str, torch.Tensor],
    unified.TraceKeys,
]:
    """Trace the shape-only wrapper (`routing`: the {@link RoutingWrapper} over the same model)."""
    config = load_config(model_dir, layers)
    cases = one_shot.encode_cases(tokenizer_dir, GOLDEN_CASES)
    unified.assert_trace_lengths(cases, sym_max)
    _, example_ids = max(
        (case for case in cases if case[0] != unified.WINDOW_CASE),
        key=lambda case: case[1].shape[1],
    )
    wrapper = unified.shape_only_model(config)
    traced = RoutingWrapper(wrapper.model).eval() if routing else wrapper
    graph, tensors = unified.trace(traced, example_ids, sym_max)
    return config, cases, graph, tensors, unified.trace_keys(wrapper)


class RoutingWrapper(unified.UnifiedProductWrapper):
    """The product wrapper with each layer's routing ids as outputs after logits and hidden —
    `(logits, hidden, route_0, …, route_{L−1})`, `route_i` the `[M, k]` index of layer `i`'s
    router `topk`. A diagnostic graph (`--routing-outputs`), never a product one: the pipeline
    admits exactly two outputs.

    MUST: the routing ids are the router's own `topk` output (a forward hook on each router), so
    the graph computes exactly the product graph's values. The inputs are the product wrapper's,
    by name (`torch.export` binds the dynamic shapes to them).
    """

    def forward(  # type: ignore[override]  -- more outputs than the product wrapper's two
        self,
        input_ids: torch.Tensor,
        soft_embeds: torch.Tensor,
        soft_mask: torch.Tensor,
        rope_sliding_attention_cos: torch.Tensor,
        rope_sliding_attention_sin: torch.Tensor,
        rope_full_attention_cos: torch.Tensor,
        rope_full_attention_sin: torch.Tensor,
        last_row: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        picks: list[torch.Tensor] = []
        handles = [
            layer.router.register_forward_hook(
                lambda _module, _args, output: picks.append(output[2])
            )
            for layer in self.model.model.layers
        ]
        try:
            logits, hidden = super().forward(
                input_ids,
                soft_embeds,
                soft_mask,
                rope_sliding_attention_cos,
                rope_sliding_attention_sin,
                rope_full_attention_cos,
                rope_full_attention_sin,
                last_row,
            )
        finally:
            for handle in handles:
                handle.remove()
        return (logits, hidden, *picks)


def assert_expert_wiring(graph: IrGraph, config: Any) -> dict[str, int]:
    """Two `expert_linear` per layer, each indexed by the index output of a `topk` of k
    `top_k_experts` over the `num_experts` router probabilities, its weight `[E, out, in]`; one
    router `topk` per layer.

    MUST: checked in the written graph — the runtime refuses any other producer of the index at
    plan time (ADR 0145 decision 2), so a graph that would be refused there fails here first.
    """
    producers = unified._producers(graph)
    layers = int(config.num_hidden_layers)
    experts = int(config.num_experts)
    picks = int(config.top_k_experts)
    nodes = [node for node in graph.nodes if node.op == EXPERT_LINEAR_OP]
    if len(nodes) != 2 * layers:
        raise MoeExportError(f"{len(nodes)} expert_linear nodes — not 2 per layer ({layers})")
    routers: set[int] = set()
    for node in nodes:
        topk = producers.get(node.ins[1])
        if topk is None or topk.op != TOPK_OP or len(topk.outs) != 2 or node.ins[1] != topk.outs[1]:
            raise MoeExportError(f"expert_linear '{node.outs[0]}': its index is not a topk index")
        width = list(declared_shape(graph, topk.ins[0]))[-1]
        if width != experts or topk.attrs.get("k") != picks:
            raise MoeExportError(
                f"expert_linear '{node.outs[0]}': topk of k {topk.attrs.get('k')} over {width},"
                f" not k {picks} over {experts} experts"
            )
        weight = list(declared_shape(graph, node.ins[2]))
        if len(weight) != 3 or weight[0] != experts:
            raise MoeExportError(
                f"expert_linear '{node.outs[0]}': weight {weight} is not [{experts}, out, in]"
            )
        routers.add(id(topk))
    if len(routers) != layers:
        raise MoeExportError(f"{len(routers)} router topk nodes — not one per layer ({layers})")
    return {"expert_linear_nodes": len(nodes), "router_topk_nodes": len(routers)}


def assert_routing_outputs(graph: IrGraph, config: Any) -> dict[str, int]:
    """The diagnostic graph's outputs: logits, hidden, then each layer's router `topk` index, in
    layer order."""
    producers = unified._producers(graph)
    layers = int(config.num_hidden_layers)
    if len(graph.outputs) != 2 + layers:
        raise MoeExportError(f"{len(graph.outputs)} outputs — not logits, hidden and {layers}")
    topks = [node for node in graph.nodes if node.op == TOPK_OP]
    for layer, (name, topk) in enumerate(zip(graph.outputs[2:], topks, strict=True)):
        if producers.get(name) is not topk or name != topk.outs[1]:
            raise MoeExportError(f"output {2 + layer} is not layer {layer}'s router topk index")
    return {"route_outputs": layers}


def plan(
    model_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    layers: int | None,
    sym_max: int,
    steps: int = decode.GREEDY_STEPS["26b"],
    floor: float = MEMORY_FLOOR_GIB,
) -> dict[str, Any]:
    """Trace only (no weight is read): the graph's form, its ops, the experts' wiring, the
    checkpoint's keys, the rank ≥ 2 guard, the case lengths against the window and the suppressed
    ids."""
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    with MemoryWatch(floor) as monitor, monitor.stage("trace"):
        config, cases, graph, _, keys = _trace(model_dir, tokenizer_dir, layers, sym_max)
    unified.assert_checkpoint_keys(
        keys.persistent, open_checkpoint(model_dir).names(), int(config.num_hidden_layers)
    )
    unified.assert_fixed_matrices(keys)
    form = product.assert_ir_form_product(graph, config, {}, None)
    return {
        "stages": [stage.to_dict() for stage in monitor.records],
        "layers": int(config.num_hidden_layers),
        "experts_implementation": config._experts_implementation,
        "inputs": [[spec.name, spec.dtype, spec.shape] for spec in graph.inputs],
        "nodes": len(graph.nodes),
        "ops": dict(Counter(node.op for node in graph.nodes).most_common()),
        "symbols": list(graph.symbols),
        "i4_targets": len(keys.int4),
        "i4f16_targets": len(keys.experts),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
        "window": unified.assert_window_exercised(
            [case for case in cases if case[0] in GREEDY_CASES], steps, config
        ),
        "suppress_tokens": list(unified.suppress_tokens(tokenizer_dir, config)),
        "form": form,
        "kv": unified.assert_shared_kv_projection(graph, config),
        "experts": assert_expert_wiring(graph, config),
    }


# ---------------------------------------------------------------------------
# The reference over the shared import
# ---------------------------------------------------------------------------


class PackedAtUse(nn.Module):
    """A parametrization: the resident packed i4 codes (`[out, in/2]`, the container's bytes) →
    `dequantize_int4(unpack_int4(packed), step)` on every read — the value the container decodes
    to."""

    def __init__(self, scale: torch.Tensor, shape: tuple[int, int]) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.shape = shape

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return dequantize_int4(unpack_int4(packed, self.shape), self.scale)


def imported_table(fixed: FixedQuantizedWeight) -> torch.Tensor:
    """The embedding's imported grid as F32 values (the tied head with it), decoded from the
    container's packed bytes a block of rows at a time."""
    rows, cols = int(fixed.packed.shape[0]), int(fixed.packed.shape[1]) * 2
    table = torch.empty((rows, cols), dtype=torch.float32)
    step = unified._block_rows(cols)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        codes = unpack_int4(fixed.packed[start:stop], (stop - start, cols))
        table[start:stop] = dequantize_int4(codes, fixed.scale[start:stop])
    return table


def reference_wrapper(
    config: Any,
    checkpoint: unified.Checkpoint,
    keys: unified.TraceKeys,
    fixed: Mapping[str, FixedQuantizedWeight],
) -> unified.UnifiedProductWrapper:
    """The F32 reference over the container's own imported weights (module docstring, "Storage").

    `fixed` is the material the container is written from (`gemma4.export_unified.
    prepare_material`): each i4 linear holds its packed bytes and reads through
    {@link PackedAtUse}, the embedding (the tied head with it) holds its decoded grid, each expert
    weight is a `gemma4.experts.PackedExperts` over its packed bytes; the rest is the checkpoint's
    values widened to F32.

    MUST: the reference's own parameters of rank ≥ 2 are exactly the imported ones (the guard of
    `gemma4.export_unified.assert_fixed_matrices`, on the model this builds) — none of them is
    read from the checkpoint here.
    """
    from torch.nn.utils import parametrize

    model = unified.meta_model(config)
    model.requires_grad_(False)
    matrices = {
        f"model.{name}"
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if parameter.dim() >= 2
    }
    tied = {unified.EMBED_KEY, unified.TIED_HEAD_KEY}
    if matrices != keys.int4 | keys.experts | tied:
        raise MoeExportError(
            "the reference's rank ≥ 2 parameters are not the imported ones:"
            f" {sorted(matrices ^ (keys.int4 | keys.experts | tied))[:4]}"
        )
    (embedding,) = (key for key in fixed if key in tied)
    imported = set(fixed) - {embedding}
    if imported != keys.int4 | keys.experts:
        raise MoeExportError(
            "the imported weights are not the trace's targets:"
            f" {sorted(imported ^ (keys.int4 | keys.experts))[:4]}"
        )
    state = {
        key: checkpoint.read(unified.checkpoint_key(f"model.{key}"))
        for key in model.state_dict()
        if f"model.{key}" not in matrices
    }
    missing, unexpected = model.load_state_dict(state, assign=True, strict=False)
    del state
    if unexpected or {f"model.{key}" for key in missing} != matrices:
        raise MoeExportError(f"reference load: missing {missing[:4]}, unexpected {unexpected[:4]}")
    model.model.embed_tokens.weight = nn.Parameter(imported_table(fixed[embedding]), False)
    model.lm_head.weight = model.model.embed_tokens.weight
    for key in sorted(keys.experts):
        owner, _, name = key.removeprefix("model.").rpartition(".")
        module = model.get_submodule(owner)
        shape = tuple(int(dim) for dim in getattr(module, name).shape)
        delattr(module, name)
        weight = fixed[key]
        setattr(module, name, routed.PackedExperts(weight.packed, weight.scale, shape, key))
    for key in sorted(keys.int4):
        module = model.get_submodule(key.removeprefix("model.").removesuffix(".weight"))
        weight = fixed[key]
        shape = (int(weight.packed.shape[0]), int(weight.packed.shape[1]) * 2)
        module.weight = nn.Parameter(weight.packed, requires_grad=False)
        parametrize.register_parametrization(
            module, "weight", PackedAtUse(weight.scale, shape), unsafe=True
        )
    one_shot.assert_materialized(model)
    decode.swap_rope_inputs(model)
    return unified.UnifiedProductWrapper(model).eval()


class LayerRecorder:
    """Forward hooks on the reference's decoder layers (each layer's output `[T, H]`, for the
    layers in `layers`) and, with `routes`, on its routers (the top {@link router_top} ids and
    probabilities of every row) — the values a teacher-forced forward leaves for the goldens."""

    def __init__(
        self, wrapper: unified.UnifiedProductWrapper, layers: Iterable[int], *, routes: bool
    ) -> None:
        self.hidden: dict[int, torch.Tensor] = {}
        self.route_ids: dict[int, torch.Tensor] = {}
        self.route_probs: dict[int, torch.Tensor] = {}
        self._top = router_top(wrapper.model.config)
        kept = set(layers)
        decoder = wrapper.model.model.layers
        self._handles = [
            decoder[layer].register_forward_hook(partial(self._layer, layer)) for layer in kept
        ]
        if routes:
            self._handles += [
                module.router.register_forward_hook(partial(self._router, layer))
                for layer, module in enumerate(decoder)
            ]

    def _layer(self, layer: int, _module: nn.Module, _args: Any, output: torch.Tensor) -> None:
        self.hidden[layer] = output[0].detach().clone()

    def _router(self, layer: int, _module: nn.Module, _args: Any, output: Any) -> None:
        best = torch.topk(output[0].detach(), self._top, dim=-1)
        self.route_ids[layer] = best.indices
        self.route_probs[layer] = best.values

    def __enter__(self) -> LayerRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()


def teacher_forced(
    wrapper: unified.UnifiedProductWrapper, ids: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(logits[R, V], hidden[R, H])` of the last `rows` rows of one forward over the prompt — the
    graph's outputs (softcapped logits, never masked: the host masks)."""
    specs = decode.rope_specs(wrapper.model.config)
    with torch.no_grad():
        logits, hidden = wrapper(
            ids,
            *unified.no_soft_inputs(ids, int(wrapper.model.config.hidden_size)),
            *decode.rope_args(specs, positions_for(ids)),
            decode.last_rows_for(ids, rows),
        )
    return logits[0], hidden[0]


def cut_layers(config: Any) -> list[int]:
    """The layers a cut may end on: every full-attention layer but the last (N = layer + 1)."""
    return [
        layer
        for layer, kind in enumerate(config.layer_types[:-1])
        if kind == one_shot.FULL_ATTENTION
    ]


def write_cut_points(
    reference: unified.UnifiedProductWrapper,
    cases: Sequence[tuple[str, torch.Tensor]],
    config: Any,
    out_dir: Path,
) -> dict[str, Any]:
    """The full series' cut points (module docstring): `cut-points.<case>.safetensors` with the
    prompt and `layer.<i>` — layer `i`'s output at every row — for each layer of
    {@link cut_layers}."""
    layers = cut_layers(config)
    written: list[str] = []
    for name, ids in cases:
        with LayerRecorder(reference, layers, routes=False) as recorder:
            teacher_forced(reference, ids, 1)
        tensors = {
            PROMPT_KEY: normalize_boundary_tensor(ids[0], f"prompt of {name}"),
            **{
                f"{LAYER_KEY_PREFIX}{layer}": recorder.hidden[layer].contiguous()
                for layer in layers
            },
        }
        path = out_dir / f"{CUT_POINTS_PREFIX}{name}{GREEDY_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        print(f"[cut-points] {name} T={int(ids.shape[1])} layers {layers}", file=sys.stderr)
    return {"files": written, "layers": layers}


def write_cut_goldens(
    reference: unified.UnifiedProductWrapper,
    cases: Sequence[tuple[str, torch.Tensor]],
    config: Any,
    out_dir: Path,
    rows: int = CUT_ROWS,
) -> tuple[list[str], dict[str, torch.Tensor], dict[str, float]]:
    """The cut model's goldens (module docstring, "The cut model"), one file per case; also
    returns each case's last-layer output at every row (for {@link assert_cut_consistency}) and its
    smallest gap between the k-th and the (k+1)-th router probability over every layer and row."""
    layers = int(config.num_hidden_layers)
    picks = int(config.top_k_experts)
    written: list[str] = []
    last: dict[str, torch.Tensor] = {}
    gaps: dict[str, float] = {}
    for name, ids in cases:
        with LayerRecorder(reference, range(layers), routes=True) as recorder:
            logits, hidden = teacher_forced(reference, ids, rows)
        selected = decode.last_rows_for(ids, rows)
        probs = torch.stack([recorder.route_probs[layer] for layer in range(layers)])
        tensors = {
            PROMPT_KEY: normalize_boundary_tensor(ids[0], f"prompt of {name}"),
            ROWS_KEY: normalize_boundary_tensor(selected, f"rows of {name}"),
            LOGITS_KEY: logits.contiguous(),
            HIDDEN_KEY: hidden.contiguous(),
            LAYER_HIDDEN_KEY: torch.stack(
                [recorder.hidden[layer][selected] for layer in range(layers)]
            ).contiguous(),
            ROUTE_IDS_KEY: normalize_boundary_tensor(
                torch.stack([recorder.route_ids[layer] for layer in range(layers)]),
                f"route ids of {name}",
            ),
            ROUTE_PROBS_KEY: probs.contiguous(),
        }
        path = out_dir / f"{CUT_PREFIX}{name}{GREEDY_SUFFIX}"
        save_file(tensors, str(path))
        written.append(path.name)
        last[name] = recorder.hidden[layers - 1]
        gaps[name] = float((probs[..., picks - 1] - probs[..., picks]).min())
        print(f"[cut] {name} T={int(ids.shape[1])} min k/k+1 gap {gaps[name]:.3g}", file=sys.stderr)
    return written, last, gaps


def assert_cut_consistency(
    last: Mapping[str, torch.Tensor],
    cases: Sequence[tuple[str, torch.Tensor]],
    full_series: Path,
    layer: int,
) -> dict[str, Any]:
    """The cut reference's output of its last layer (`layer`) is the full reference's, bit for bit,
    for every case (the full series' cut points — {@link write_cut_points}).

    MUST: `torch.equal` — the same weights, inputs and ops on the same machine give the same bits,
    so any difference means the cut goldens are not the full model's first layers.
    """
    for name, ids in cases:
        path = full_series / f"{CUT_POINTS_PREFIX}{name}{GREEDY_SUFFIX}"
        if not path.is_file():
            raise MoeExportError(
                f"{path} is missing — write the product series first (python -m gemma4.export_moe)"
            )
        key = f"{LAYER_KEY_PREFIX}{layer}"
        with safe_open(str(path), framework="pt") as handle:
            if key not in set(handle.keys()):
                raise MoeExportError(f"{path.name} has no '{key}' (keys {sorted(handle.keys())})")
            prompt = handle.get_tensor(PROMPT_KEY).to(torch.int64)
            full = handle.get_tensor(key)
        if not torch.equal(prompt, ids[0].to(torch.int64)):
            raise MoeExportError(f"{path.name}: its prompt is not this run's case '{name}'")
        if not torch.equal(last[name], full):
            difference = float((last[name] - full).abs().max())
            raise MoeExportError(
                f"{name}: the cut's layer {layer} output differs from the full reference's"
                f" (max |Δ| {difference:.3g}) — the cut goldens are not the full model's"
            )
    return {"fullSeries": full_series.name, "layer": layer, "rule": "torch.equal"}


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------


def _storage(keys: unified.TraceKeys) -> dict[str, int]:
    """The storage counts of the written graph: every linear and the tied embedding i4, every
    expert weight i4f16, no i8."""
    return {"i4": len(keys.int4) + 1, "i4f16": len(keys.experts), "i8": 0}


def export_series(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    sym_max: int = one_shot.SYM_MAX,
    steps: int = decode.GREEDY_STEPS["26b"],
    floor: float = MEMORY_FLOOR_GIB,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Write the product container, the greedy goldens, the cut points and the provenance record
    through one staged seat.

    `model_dir` holds the weights and their config, `tokenizer_dir` the tokenizer and the
    generation defaults; `checkpoint_label` and `pins` are 12B's knobs
    (`gemma4.export_unified.export_series`).

    The grid is imported once, before the goldens (module docstring, "Storage"), and the
    container is written after them: a failing margin stops the run before the long write.
    MUST: every gate (margins, first tokens, form, K = V wiring, the experts' wiring, the pinned
    fingerprint) passes before the seat is published.
    """
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    checkpoint = open_checkpoint(model_dir)
    with MemoryWatch(floor) as monitor:
        with monitor.stage("trace") as record:
            config, cases, graph, tensors, keys = _trace(model_dir, tokenizer_dir, None, sym_max)
            record.details["nodes"] = len(graph.nodes)
        layers = int(config.num_hidden_layers)
        unified.assert_checkpoint_keys(keys.persistent, checkpoint.names(), layers)
        unified.assert_fixed_matrices(keys)
        greedy_cases = tuple(case for case in cases if case[0] in GREEDY_CASES)
        assert_case_room(greedy_cases, steps, decode.max_position(config))
        window = unified.assert_window_exercised(greedy_cases, steps, config)
        suppress = unified.suppress_tokens(tokenizer_dir, config)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            with monitor.stage("import") as record:
                material = unified.prepare_material(graph, tensors, keys, checkpoint)
                record.details["fixed"] = len(material.fixed)
            del tensors
            with monitor.stage("goldens"):
                print(
                    "[goldens] reference: the container's imported grid, experts per pick;"
                    f" suppressed ids {list(suppress)}",
                    file=sys.stderr,
                    flush=True,
                )
                reference = reference_wrapper(config, checkpoint, keys, material.fixed)
                view = unified.GreedyView(reference, decode.rope_specs(config), suppress)
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
                cut_points = write_cut_points(reference, cases, config, staged)
                del view, reference
            gc.collect()
            measured = unified._greedy_summary(
                tokenizer_dir,
                written,
                tokens,
                margins,
                first,
                steps,
                expectations=FIRST_TOKEN_EXPECTATIONS,
            )
            with monitor.stage("write"):
                verified = unified.write_container(
                    graph,
                    material,
                    staged / one_shot.MODEL_FILE,
                    None if pins is None else pins.weights.revision,
                )
            del material
            storage = _storage(keys)
            form = product.assert_ir_form_product(verified, config, storage, None)
            kv = unified.assert_shared_kv_projection(verified, config)
            wiring = assert_expert_wiring(verified, config)
            with monitor.stage("provenance"):
                own = provenance.assert_reference_goldens(staged, greedy_cases)
                files = weights_files(checkpoint)
                record_ = provenance.build_record(
                    out_dir,
                    model_dir,
                    {
                        **own,
                        "series": out_dir.name,
                        "suppressTokens": list(suppress),
                        "experts": routed.EXPERTS_RECORD,
                        "cutPoints": cut_points,
                    },
                    checkpoint_label=checkpoint_label,
                    tokenizer_dir=tokenizer_dir,
                    pins=pins,
                    weights_files=files,
                )
                if pins is not None:
                    provenance.assert_pinned_fingerprint(
                        record_["checkpoint"], pins, weights_files=files
                    )
                provenance.write_record(staged, record_)
    return {
        **_written_summary(out_dir, config, verified, cases),
        "window": window,
        "quantized": storage,
        "form": form,
        "kv": kv,
        "experts": wiring,
        **measured,
        "suppress_tokens": list(suppress),
        "reference": record_["reference"],
        "stages": [stage.to_dict() for stage in monitor.records],
    }


def _written_summary(
    out_dir: Path, config: Any, verified: IrGraph, cases: Sequence[tuple[str, torch.Tensor]]
) -> dict[str, Any]:
    parts = container_parts(out_dir / one_shot.MODEL_FILE)
    return {
        "dir": str(out_dir),
        "layers": int(config.num_hidden_layers),
        "nodes": len(verified.nodes),
        "outputs": len(verified.outputs),
        "initializers": len(verified.initializers),
        "parts": len(parts),
        "model_bytes": sum(path.stat().st_size for path in parts),
        "ops": sorted(verified.required_ops),
        "symbols": list(verified.symbols),
        "case_lengths": {name: int(ids.shape[1]) for name, ids in cases},
    }


def _routing_material(
    graph: IrGraph, tensors: Mapping[str, torch.Tensor], material: unified.ContainerMaterial
) -> unified.ContainerMaterial:
    """The diagnostic graph's material: the product material's weights (the same model, so the
    same keys and bytes) and the diagnostic trace's own constants."""
    declared = {init.tensor for init in graph.initializers.values()}
    chosen: dict[str, torch.Tensor] = {}
    for key in declared:
        if key.startswith(unified.CONSTANT_PREFIX):
            chosen[key] = tensors[key]
        elif key in material.tensors:
            chosen[key] = material.tensors[key]
        else:
            raise MoeExportError(f"the routing graph declares '{key}', which the product does not")
    return unified.ContainerMaterial(
        tensors=chosen,
        fixed={key: value for key, value in material.fixed.items() if key in declared},
    )


def export_cut(
    model_dir: Path,
    out_dir: Path,
    *,
    tokenizer_dir: Path,
    pins: ModelPins | None,
    layers: int = DEFAULT_CUT,
    full_series: Path | None,
    routing: bool = False,
    sym_max: int = one_shot.SYM_MAX,
    floor: float = MEMORY_FLOOR_GIB,
    checkpoint_label: str | None = None,
) -> dict[str, Any]:
    """Write the cut model (module docstring, "The cut model"): its container, its goldens, the
    diagnostic routing container when `routing`, and the provenance record.

    `full_series` is the product series whose cut points hold the full reference's layers
    ({@link assert_cut_consistency}); `None` skips that check, and the record says so.
    MUST: every gate passes before the seat is published.
    """
    if pins is not None:
        sources.assert_model_pins(model_dir, tokenizer_dir, pins)
    checkpoint = open_checkpoint(model_dir)
    route: tuple[IrGraph, dict[str, torch.Tensor]] | None = None
    with MemoryWatch(floor) as monitor:
        with monitor.stage("trace") as record:
            config, cases, graph, tensors, keys = _trace(model_dir, tokenizer_dir, layers, sym_max)
            record.details["nodes"] = len(graph.nodes)
        if routing:
            with monitor.stage("trace-routing"):
                _, _, route_graph, route_tensors, _ = _trace(
                    model_dir, tokenizer_dir, layers, sym_max, routing=True
                )
                route = (route_graph, route_tensors)
        unified.assert_checkpoint_keys(keys.persistent, checkpoint.names(), layers)
        unified.assert_fixed_matrices(keys)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        with staged_publication(out_dir) as staged:
            staged.mkdir()
            with monitor.stage("import") as record:
                material = unified.prepare_material(graph, tensors, keys, checkpoint)
                record.details["fixed"] = len(material.fixed)
            del tensors
            with monitor.stage("goldens"):
                reference = reference_wrapper(config, checkpoint, keys, material.fixed)
                written, last, gaps = write_cut_goldens(reference, cases, config, staged)
                del reference
            gc.collect()
            consistency = (
                {"fullSeries": None, "rule": "skipped"}
                if full_series is None
                else assert_cut_consistency(last, cases, full_series, layers - 1)
            )
            del last
            revision = None if pins is None else pins.weights.revision
            with monitor.stage("write"):
                verified = unified.write_container(
                    graph, material, staged / one_shot.MODEL_FILE, revision
                )
                diagnostic = None
                if route is not None:
                    (staged / ROUTING_DIR).mkdir()
                    diagnostic = unified.write_container(
                        route[0],
                        _routing_material(*route, material),
                        staged / ROUTING_DIR / one_shot.MODEL_FILE,
                        revision,
                    )
            del material, route
            storage = _storage(keys)
            form = product.assert_ir_form_product(verified, config, storage, None)
            kv = unified.assert_shared_kv_projection(verified, config)
            wiring = assert_expert_wiring(verified, config)
            route_form = {}
            if diagnostic is not None:
                route_form = {
                    **assert_routing_outputs(diagnostic, config),
                    **assert_expert_wiring(diagnostic, config),
                }
            with monitor.stage("provenance"):
                files = weights_files(checkpoint)
                cut = {
                    "layers": layers,
                    "rows": CUT_ROWS,
                    "routerTop": router_top(config),
                    "consistency": consistency,
                    "routingOutputs": diagnostic is not None,
                }
                record_ = provenance.build_record(
                    out_dir,
                    model_dir,
                    {
                        "series": out_dir.name,
                        "goldens": {
                            name: provenance.file_digest(staged / name) for name in written
                        },
                        "experts": routed.EXPERTS_RECORD,
                        "cut": cut,
                    },
                    checkpoint_label=checkpoint_label,
                    tokenizer_dir=tokenizer_dir,
                    pins=pins,
                    weights_files=files,
                )
                if pins is not None:
                    provenance.assert_pinned_fingerprint(
                        record_["checkpoint"], pins, weights_files=files
                    )
                provenance.write_record(staged, record_)
    return {
        **_written_summary(out_dir, config, verified, cases),
        "quantized": storage,
        "form": form,
        "kv": kv,
        "experts": wiring,
        "routing": route_form,
        "cut": cut,
        "router_gap_min": gaps,
        "stages": [stage.to_dict() for stage in monitor.records],
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
    parser.add_argument(
        "--layers",
        type=int,
        nargs="?",
        const=DEFAULT_CUT,
        default=None,
        help=f"write the model cut after N layers with its goldens (N alone: {DEFAULT_CUT})",
    )
    parser.add_argument(
        "--routing-outputs",
        action="store_true",
        help="with --layers: also write the diagnostic container with routing outputs",
    )
    parser.add_argument(
        "--full-series",
        type=Path,
        default=None,
        help="with --layers: the product series holding the cut points (default: the model's)",
    )
    parser.add_argument(
        "--no-full-check",
        action="store_true",
        help="with --layers: skip the check against the product series' cut points",
    )
    parser.add_argument("--plan", action="store_true", help="trace only; nothing is written")
    parser.add_argument("--memory-floor", type=float, default=MEMORY_FLOOR_GIB)
    args = parser.parse_args(argv)
    if args.layers is None and (args.routing_outputs or args.full_series or args.no_full_check):
        parser.error("--routing-outputs / --full-series / --no-full-check need --layers")
    model = args.model
    if args.layers is not None and args.out is None and not args.plan:
        args.out = one_shot.series_dir(model, f"cut{args.layers}")
    full_series = (
        None
        if args.no_full_check
        else args.full_series or one_shot.series_dir(model, GEMMA4_PRODUCT_SUFFIX)
    )
    routing, layers = args.routing_outputs, args.layers
    del args.routing_outputs, args.full_series, args.no_full_check
    options = one_shot.resolve_series_args(args)
    model_dir = options["model_dir"]
    pins = sources.HUB_PINS.get(model)
    if options["plan"]:
        summary = plan(
            model_dir,
            tokenizer_dir=options["tokenizer_dir"],
            pins=pins,
            layers=layers,
            sym_max=options["sym_max"],
            steps=options["steps"],
            floor=options["memory_floor"],
        )
    elif layers is not None:
        summary = export_cut(
            model_dir,
            options["out"],
            tokenizer_dir=options["tokenizer_dir"],
            pins=pins,
            layers=layers,
            full_series=full_series,
            routing=routing,
            sym_max=options["sym_max"],
            floor=options["memory_floor"],
            checkpoint_label=sources.checkpoint_label(model),
        )
    else:
        summary = export_series(
            model_dir,
            options["out"],
            tokenizer_dir=options["tokenizer_dir"],
            pins=pins,
            sym_max=options["sym_max"],
            steps=options["steps"],
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
