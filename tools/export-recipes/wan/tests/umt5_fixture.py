"""A synthetic umT5 encoder container — a fixture for trying the distribution gate
(`wan.umt5_distribution`) on the same shape as the real thing.

It matches the real container (the i8 S-form container `wan.umt5_export` writes) only in the four
things the gate reads:

- the spelling of the tensor keys (the vocabulary embedding `encoder.embed_tokens.weight`, the
  per-layer linear / RMSNorm / relative-position tables, the final RMSNorm, the lifted constants
  `const.*`) and the storage of each kind (linear and the vocabulary embedding are i8 packed + a
  per-row scale; the rest is f32)
- the graph inputs `input_ids [1, L]` / `relative_position_buckets [L, L]` (i32) and the symbol `L`
- one output `[1, L, W]` (f32)
- the container's graph name (the component name `text_encoder`) and the provenance

It holds none of the computation (attention, FFN) — the gate reads only declarations. The weights
are a few dozen bytes, and neither real weights nor transformers are needed (it runs on the default
sync alone). The fault-injection knobs (the storage of the tables and of linear, whether the tables
exist, the inputs, the output width) are swapped through arguments. The symbols are derived from the
input shapes (the declaration and the binding sites do not move independently — spelling a second
symbol into an input yields a valid IR that declares that symbol).

The synthetic version of the required asset of the distribution `karume-umt5-xxl` (the tokenizer —
`wan.umt5_xxl_tokenizer`) ({@link write_tokenizer_asset}) also lives here (shared by the tests that
assemble the umT5 distribution).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from karume.container import Provenance, write_model_container
from karume.dims import parse_dim
from karume.emit import FixedQuantizedWeight, stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.publish import publish_container
from karume.quantize import channel_scale, quantize_to_int8
from wan import umt5_xxl_tokenizer
from wan.sources import UMT5_SOURCES
from wan.umt5_distribution import (
    UMT5_DEFAULT_MODEL,
    UMT5_INPUTS,
    UMT5_ROLE,
    UMT5_SYMBOL,
    umt5_tokenizer_asset,
)

#: Synthetic sizes (numbers **different** from the real d_model 4096, vocabulary 256,384, head 64 —
#: anything baked in fails).
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
    """A per-row-scale i8 (the same shape as the writer's `quantize_rows` — packed + a `[rows, 1]`
    scale)."""
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
    verified: bool = True,
) -> list[bytes]:
    """The part bytes of one valid umT5-like container (in index order — the first is part 0).

    `table_layout` / `linear_layout` are the storage of the relative-position tables and of linear
    (`"i8"` or `"f32"`), `tables` is whether the tables exist, `inputs` is the declaration of the
    graph inputs (name, dtype, shape), and `width` is the output width.

    `verified=False` is the fault-injection knob that deliberately builds a container that violates
    the IR acceptance rules (e.g. an embedding that takes f32 token ids) — the public gate
    (`publish_container`) refuses it, so it is written through the same writer's low-level entry
    point (`write_model_container`) without the read-back verification.
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
            bias_values = value(f"bias_{layer}", [*graph_inputs[1].shape, HEADS])
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
        symbols=sorted(
            {
                parse_dim(dim).sym
                for spec in graph_inputs
                for dim in spec.shape
                if isinstance(dim, str)
            }
        ),
        inputs=graph_inputs,
        outputs=[output],
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    stored = stored_model(graph, tensors, fixed_weights=fixed or None)
    with TemporaryDirectory() as staging:
        args = (Path(staging) / "model.krm", stored.graph, stored.tensors, stored.bindings)
        parts = (
            publish_container(*args, graph_name=named, provenance=provenance).parts
            if verified
            else write_model_container(*args, graph_name=named, provenance=provenance)
        )
        return [path.read_bytes() for path in parts]


def write_tokenizer_asset(series_dir: Path, **overrides: Any) -> Path:
    """Put a tokenizer asset in the parent of the series (`outputs/series/`) and return its path.

    The top-level fields are the same as the writer's (`wan.umt5_xxl_tokenizer.build_asset`); the
    source is the row of the upstream table and the versions are the versions of the id-sequence
    source of truth (the values the gate checks). The vocabulary and the tables are small values,
    since the gate does not read their contents. `overrides` overrides fields, and a field whose
    value is None is removed (the fault-injection knob).
    """
    row = UMT5_SOURCES[UMT5_DEFAULT_MODEL]
    asset: dict[str, Any] = {
        "format": umt5_xxl_tokenizer.ASSET_FORMAT,
        "source": {
            "repo": row.source.repo,
            "revision": row.source.revision,
            "files": dict(sorted(row.tokenizer.items())),
        },
        "versions": {"tokenizers": "0.22.2", "transformers": "5.14.1"},
        "maxLength": 512,
        "unkId": 2,
        "eosId": 1,
        "addedTokens": [["<pad>", 0], ["</s>", 1], ["<s>", 2], ["<unk>", 3]],
        "space": [[32, 32]],
        "vocabText": "<pad>\n</s>\n<s>\n<unk>\n▁cat",
        "scores": [0.0, 0.0, 0.0, 0.0, -1.0],
        **overrides,
    }
    path = umt5_tokenizer_asset(series_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in asset.items() if value is not None}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
