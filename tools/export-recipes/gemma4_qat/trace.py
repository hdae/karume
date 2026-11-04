"""The QAT trace that keeps the fixed integers as the values, using the existing Gemma wrapper /
states conversion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.export import Dim

from _shared.decode_series import positions_for
from gemma4 import export as one_shot
from gemma4 import export_decode as decode
from karume.convert import PRESERVED_OP_PREFIXES_WITH_ATTENTION, convert, curated_decompositions
from karume.emit import FixedQuantizedWeight
from karume.ir import IrGraph
from karume.normalize import normalize_graph
from karume.states import to_states_form

from .checkpoint import LoadedQat, fixed_trace_weights
from .config import MAX_CHUNK_LENGTH, MAX_SELECTED_ROWS


@dataclass(frozen=True)
class TracedQat:
    graph: IrGraph
    tensors: dict[str, torch.Tensor]
    fixed: dict[str, FixedQuantizedWeight]


def trace_qat(loaded: LoadedQat, ids: torch.Tensor) -> TracedQat:
    """A shape-only trace after setting the fixed payloads aside. A one-way process never mixed
    with ordinary inference."""
    if ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= MAX_CHUNK_LENGTH:
        raise ValueError(
            f"the QAT trace's example input must be [1,M] with 2<=M<={MAX_CHUNK_LENGTH}"
        )
    with torch.inference_mode():
        ple = loaded.ple(ids).reshape(
            1,
            ids.shape[1],
            loaded.config.num_hidden_layers,
            loaded.config.hidden_size_per_layer_input,
        )
    args = (
        ids,
        *decode.rope_args(decode.rope_specs(loaded.config), positions_for(ids)),
        ple,
        decode.last_rows_for(ids, 2),
    )
    fixed = fixed_trace_weights(loaded.wrapper)
    one_shot.register_attention()
    loaded.config._attn_implementation = one_shot.ATTENTION_NAME
    seq = Dim("M", min=2, max=MAX_CHUNK_LENGTH)
    rows = Dim("R", min=1, max=MAX_SELECTED_ROWS)
    exported = torch.export.export(
        loaded.wrapper,
        args,
        dynamic_shapes=(*({1: seq} for _ in range(6)), {0: rows}),
        strict=False,
    )
    exported = exported.run_decompositions(
        curated_decompositions(PRESERVED_OP_PREFIXES_WITH_ATTENTION)
    )
    normalize_graph(exported)
    for key, tensor in exported.state_dict.items():
        if key in fixed:
            exported.state_dict[key] = torch.empty(tensor.shape, dtype=torch.float32, device="meta")
    graph, tensors = convert(exported, symbol_names=("M", "R"))
    meta = {key for key, value in tensors.items() if value.is_meta}
    if not meta.issubset(fixed):
        raise ValueError("a meta weight emitted by the QAT trace has no fixed payload")
    # The converter folds the tied aliases into one. Only the two names whose bytes were all
    # checked to match just before are allowed.
    aliases = {"model.model.embed_tokens.weight", "model.lm_head.weight"}
    missing = set(fixed) - meta
    if len(missing) != 1 or not missing.issubset(aliases) or not (aliases - missing).issubset(meta):
        raise ValueError("the QAT trace lost a fixed weight other than the proven shared one")
    surgical = to_states_form(graph, decode.states_plan(graph, loaded.config))
    declared = {entry.tensor for entry in surgical.initializers.values()}
    if not meta.issubset(declared):
        raise ValueError("the states conversion lost a QAT fixed weight")
    # The mask constants the states conversion removed are not stored. The same contract as the
    # existing decode writer.
    return TracedQat(
        surgical,
        {key: value for key, value in tensors.items() if key in declared},
        {key: fixed[key] for key in meta},
    )
