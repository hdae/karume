"""既存 Gemma wrapper / states 変換を使い、固定整数を値として保つ QAT trace。"""

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
    """固定 payload を退避してから shape-only trace。通常推論と混ぜない単方向の処理。"""
    if ids.ndim != 2 or ids.shape[0] != 1 or not 2 <= ids.shape[1] <= MAX_CHUNK_LENGTH:
        raise ValueError(f"QAT trace の例示入力は [1,M]、2<=M<={MAX_CHUNK_LENGTH} が必要")
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
        raise ValueError("QAT trace が出した meta 重みに固定 payload が無い")
    # tied の別名を converter が1本へ畳む。直前に全 bytes 一致を検査した2名だけを許可する。
    aliases = {"model.model.embed_tokens.weight", "model.lm_head.weight"}
    missing = set(fixed) - meta
    if len(missing) != 1 or not missing.issubset(aliases) or not (aliases - missing).issubset(meta):
        raise ValueError("QAT trace で証明済みの共有以外の固定重みが消えた")
    surgical = to_states_form(graph, decode.states_plan(graph, loaded.config))
    declared = {entry.tensor for entry in surgical.initializers.values()}
    if not meta.issubset(declared):
        raise ValueError("states 変換で QAT 固定重みが消えた")
    # states 変換が消した mask 定数を格納しない。既存 decode writer と同じ契約。
    return TracedQat(
        surgical,
        {key: value for key, value in tensors.items() if key in declared},
        {key: fixed[key] for key in meta},
    )
