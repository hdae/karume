"""umT5 encoder の合成容器 — 配布の門（`wan.umt5_distribution`）を実物と同じ形で試すための器。

実物（`wan.umt5_export` が書く i8 の S 形容器）と同じに揃えるのは、門が読む 4 つだけ:

- テンソルキーの綴り（語彙埋め込み `encoder.embed_tokens.weight`・層ごとの linear / RMSNorm /
  相対位置の表・最後の RMSNorm・持ち上げ定数 `const.*`）と、その種類ごとの格納（linear と語彙
  埋め込みは i8 の packed + 行ごとの scale・残りは f32）
- グラフ入力 `input_ids [1, L]` / `relative_position_buckets [L, L]`（i32）と記号 `L`
- 出力 `[1, L, W]`（f32）1 本
- 容器のグラフ名（部品名 `text_encoder`）と出所

計算の中身（attention・FFN）は持たない — 門は宣言しか読まない。重みは数十バイトで、実重みも
transformers も要らない（既定の sync だけで回る）。故障注入の席（表や linear の格納・表の有無・
入力・出力の幅）は引数で差し替える。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from karume.container import Provenance
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import channel_scale, quantize_to_int8
from wan.umt5_distribution import UMT5_INPUTS, UMT5_ROLE, UMT5_SYMBOL

#: 合成の寸法（実物の d_model 4096・語彙 256,384・head 64 と**違う**数 — 焼き込んでいれば落ちる）。
WIDTH = 8
VOCAB = 16
HEADS = 2
BUCKETS = 32
LAYERS = 2

_L = UMT5_SYMBOL


def _ramp(*shape: int, offset: float = 0.0) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return (torch.arange(count, dtype=torch.float32).reshape(shape) / count + 0.25 + offset).clone()


def _fixed(weight: torch.Tensor) -> FixedQuantizedWeight:
    """行ごとの scale の i8（書き手の `quantize_rows` と同じ形 — packed + `[rows, 1]` の
    scale）。"""
    scale = channel_scale(weight, 0)
    return FixedQuantizedWeight(dtype="i8", packed=quantize_to_int8(weight, scale), scale=scale)


def umt5_container(
    *,
    provenance: Provenance,
    width: int = WIDTH,
    tables: bool = True,
    table_layout: str = "f32",
    linear_layout: str = "i8",
    inputs: Sequence[tuple[str, str, list[Any]]] = UMT5_INPUTS,
    named: str = UMT5_ROLE,
) -> list[bytes]:
    """正当な umT5 風の容器 1 本ぶんの part バイト列（添字順 — 先頭が part 0）。

    `table_layout` / `linear_layout` は相対位置の表と linear の格納（`"i8"` か `"f32"`）、
    `tables` は表を持つか、`inputs` はグラフ入力の宣言（名前・dtype・形）、`width` は出力の幅。
    """
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    fixed: dict[str, FixedQuantizedWeight] = {}
    nodes: list[IrNode] = []

    def declare(key: str, tensor: torch.Tensor, layout: str) -> str:
        initializers[key] = IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        values[key] = IrValue(dtype="f32", shape=list(tensor.shape))
        if layout == "i8":
            fixed[key] = _fixed(tensor)
            tensors[key] = torch.empty(tensor.shape, dtype=torch.float32, device="meta")
        else:
            tensors[key] = tensor
        return key

    def value(name: str, shape: list[Any]) -> str:
        values[name] = IrValue(dtype="f32", shape=shape)
        return name

    graph_inputs = [
        IrInput(name=name, dtype=dtype, shape=list(shape)) for name, dtype, shape in inputs
    ]
    ids, buckets = (spec.name for spec in graph_inputs[:2])
    embed = declare("encoder.embed_tokens.weight", _ramp(VOCAB, width), "i8")
    bias = declare("const.zero", torch.zeros(width), "f32")
    hidden = value("embedding", [1, _L, width])
    nodes.append(IrNode(op="embedding", ins=[embed, ids], outs=[hidden], attrs={"padding_idx": -1}))
    for layer in range(LAYERS):
        prefix = f"encoder.block.{layer}.layer"
        norm = declare(f"{prefix}.0.layer_norm.weight", _ramp(width, offset=layer), "f32")
        normed = value(f"norm_{layer}", [1, _L, width])
        nodes.append(IrNode(op="rms_norm", ins=[hidden, norm], outs=[normed], attrs={"eps": 1e-6}))
        query = declare(
            f"{prefix}.0.SelfAttention.q.weight", _ramp(width, width, offset=layer), linear_layout
        )
        hidden = value(f"linear_{layer}", [1, _L, width])
        nodes.append(IrNode(op="linear", ins=[normed, query, bias], outs=[hidden], attrs={}))
        if tables:
            table = declare(
                f"{prefix}.0.SelfAttention.relative_attention_bias.weight",
                _ramp(BUCKETS, HEADS, offset=layer),
                table_layout,
            )
            bias_values = value(f"bias_{layer}", [_L, _L, HEADS])
            nodes.append(
                IrNode(
                    op="embedding",
                    ins=[table, buckets],
                    outs=[bias_values],
                    attrs={"padding_idx": -1},
                )
            )
    final = declare("encoder.final_layer_norm.weight", _ramp(width), "f32")
    output = value("output", [1, _L, width])
    nodes.append(IrNode(op="rms_norm", ins=[hidden, final], outs=[output], attrs={"eps": 1e-6}))

    graph = IrGraph(
        symbols=[_L],
        inputs=graph_inputs,
        outputs=[output],
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    stored = stored_model(graph, tensors, fixed_weights=fixed or None)
    with TemporaryDirectory() as staging:
        result = publish_container(
            Path(staging) / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name=named,
            provenance=provenance,
        )
        return [path.read_bytes() for path in result.parts]
