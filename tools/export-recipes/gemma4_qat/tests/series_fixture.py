"""QAT の配布 recipe が入力に使う**正当な最小の系列**（固定格納のコンテナ + PLE + 資産）。

`gemma4/tests/product_fixture.py` の QAT 版。違いは固定量子化そのもの — 混成格納
（I2 の共有 head / I4 / I8）と、量子化 linear の前後の SRQ、schema 2 の packed PLE
（`gemma4_qat.ple.write_ple` が書く）、そして `reference.json` である。

MUST: safetensors のバイト列も IR の規則も PLE の索引も手で綴らない（`product_fixture` の
同 MUST）— 規則の写しを持つと、規則が動いた日にフィクスチャだけが古びて「テストは緑・
実物だけ落ちる」になる。書き出しは実物と同じ 1 本道（`karume.emit.write_model` /
`gemma4_qat.ple.write_ple`）を通す。

MUST: **実物と違う数**にする（語彙 9・層 3・次元 32・hidden 16・位置上限 8192）— 寸法を
焼き込んでいれば落ちる。

上流の固定量子化モジュール（`transformers.integrations.gemma_quant`）を使うので、依存グループ
`gemma4-qat` を同期していない環境では、このモジュールを import するテストごと SKIP になる。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
from transformers.integrations.gemma_quant import QuantizedEmbedding

from gemma4.distribution import (
    GEMMA4_ROPE_LAYER_TYPES,
    GEMMA4_ROPE_PARTS,
    gemma4_rope_input_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from gemma4.tests.product_fixture import tokenizer_asset
from gemma4_qat.config import MAX_CHUNK_LENGTH, MAX_SELECTED_ROWS, REFERENCE_SCHEMA
from gemma4_qat.ple import write_ple
from karume.emit import FixedQuantizedWeight, write_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrState,
    IrStorage,
    IrValue,
)
from karume.verify import verify_shards

#: 合成の寸法（実物は語彙 262144・層 35/42・次元 256・hidden 1536/2560）。
#: `HIDDEN` が 16 の倍数なのは I2 格納の整列要求（`karume.emit._plan_fixed_weights`）。
VOCAB = 9
LAYERS = 3
DIM = 32
HIDDEN = 16

#: 中間幅（`HIDDEN` と違う数 — 取り違えれば形で落ちる）と I4 の group 長。
INNER = 32
GROUP_SIZE = 16

#: 上流 `config.json` の `text_config`（配布 recipe が読む欄だけ）。位置上限は既定 capacity
#: （4096）より大きく、実物（131072）とは違う数にする。
MAX_POSITION = 8192
SLIDING_HEAD_DIM = 4
FULL_HEAD_DIM = 8

TEXT_CONFIG: Mapping[str, Any] = {
    "max_position_embeddings": MAX_POSITION,
    "hidden_size": HIDDEN,
    "num_attention_heads": 4,
    "head_dim": SLIDING_HEAD_DIM,
    "global_head_dim": FULL_HEAD_DIM,
    "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION],
    "rope_parameters": {
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": 100.0},
        FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": 1000.0,
            "partial_rotary_factor": 0.5,
        },
    },
}

ROPE_HEAD_DIMS: Mapping[str, int] = {
    SLIDING_ATTENTION: SLIDING_HEAD_DIM,
    FULL_ATTENTION: FULL_HEAD_DIM,
}

#: 上流 `generation_config.json` の推奨（**実物と違う値**）。
GENERATION_CONFIG: Mapping[str, Any] = {
    "do_sample": True,
    "eos_token_id": [1, 2],
    "temperature": 0.7,
    "top_k": 8,
    "top_p": 0.5,
}

#: 記号（`M` = chunk の行数・`R` = 出口の行数・`C` = full スロットの容量）。
SEQ_SYMBOL = "M"
ROW_SYMBOL = "R"
CAPACITY_SYMBOL = "C"
SYM_MAX = 4

#: 非量子化のまま残る唯一の linear（構造門が名指しで許可している 1 本）。
PROJECTION_TENSOR = "model.model.per_layer_model_projection.weight"

#: SRQ の固定 scale（厳密な f32 値であることが op 契約の要求）。
SRQ_SCALE = 0.125


def packed_embedding(bits: int, rows: int = VOCAB, width: int = LAYERS * DIM) -> QuantizedEmbedding:
    """上流形式の packed 埋め込み（PLE と token embedding の材料）。"""
    factor = 8 // bits
    module = QuantizedEmbedding(rows, width, torch.float32, embed_scale=3.5, num_bits=bits)
    module.embedding_quantized = nn.Parameter(
        (torch.arange(rows * width // factor) * 37 % 256)
        .to(torch.uint8)
        .reshape(rows, width // factor),
        requires_grad=False,
    )
    module.embedding_scale = nn.Parameter(
        torch.arange(1, rows * (width // DIM) + 1, dtype=torch.float32).reshape(rows, width // DIM)
        / 19,
        requires_grad=False,
    )
    return module


def _packed(rows: int, width: int, bits: int) -> torch.Tensor:
    factor = 8 // bits
    dtype = torch.int8 if bits == 8 else torch.uint8
    values = torch.arange(rows * width // factor) * 29 % 128
    return values.to(dtype).reshape(rows, width // factor).contiguous()


def _row_scale(rows: int, groups: int = 1) -> torch.Tensor:
    return (
        torch.arange(1, rows * groups + 1, dtype=torch.float32).reshape(rows, groups) / 8
    ).contiguous()


def _ramp(*shape: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    return torch.arange(total, dtype=torch.float32).reshape(*shape) / total


def qat_container(
    *,
    vocab: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    mlp_bits: int = 8,
) -> list[bytes]:
    """QAT 製品グラフ 1 本ぶんの shard バイト列（読む順 — 先頭がグラフ shard）。

    形は実物の縮小版: token embedding は I2 で、同じ initializer を head linear が重みに使う
    （共有 head・前後の SRQ は持たない）。ほかに I4 / I8 の量子化 linear が 1 本ずつあり、
    どちらも前後に SRQ を持つ。非量子化の linear は許可された projection 1 本だけ。

    `mlp_bits` は混成格納の故障注入の口（`4` にすると I8 がヘッダから消える）。
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    fixed: dict[str, FixedQuantizedWeight] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32", key: str | None = None) -> str:
        tensor_key = key if key is not None else f"gemma4qat.{name}"
        initializers[name] = IrInitializer(tensor=tensor_key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype="i32" if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[tensor_key] = tensor
        return tensor_key

    def declare_fixed(name: str, rows: int, width: int, bits: int, groups: int = 1) -> None:
        """固定格納の重み 1 本（論理形は meta の f32・値は packed 側だけが持つ）。"""
        key = declare(name, torch.empty(rows, width, dtype=torch.float32, device="meta"))
        fixed[key] = FixedQuantizedWeight(
            f"i{bits}", _packed(rows, width, bits), _row_scale(rows, groups)
        )

    def quantize(source: str, out: str) -> str:
        values[out] = IrValue(dtype="f32", shape=list(values[source].shape))
        nodes.append(
            IrNode(op="static_quantize", ins=[source], outs=[out], attrs={"scale": SRQ_SCALE})
        )
        return out

    def linear(source: str, weight: str, out: str, width: int) -> str:
        bias = f"{out}_bias"
        declare(bias, _ramp(width))
        values[out] = IrValue(dtype="f32", shape=[*values[source].shape[:-1], width])
        nodes.append(IrNode(op="linear", ins=[source, weight, bias], outs=[out], attrs={}))
        return out

    # ① token embedding（I2）— 同じ initializer を共有 head が重みに使う。
    declare_fixed("embed", vocab, hidden_size, 2)
    embedded = "embedded"
    values[embedded] = IrValue(dtype="f32", shape=[1, SEQ_SYMBOL, hidden_size])
    nodes.append(
        IrNode(
            op="embedding",
            ins=["embed", "input_ids"],
            outs=[embedded],
            attrs={"padding_idx": -1},
        )
    )

    # ② 許可された非量子化 linear（per-layer projection）1 本。
    declare("projection", _ramp(INNER, hidden_size), key=PROJECTION_TENSOR)
    projected = linear(embedded, "projection", "projected", INNER)

    # ③ 量子化 linear 2 本（I4 / I8）— どちらも前後に SRQ を持つ。
    declare_fixed("attention", INNER, INNER, 4, groups=INNER // GROUP_SIZE)
    declare_fixed("mlp", hidden_size, INNER, mlp_bits)
    attended = linear(quantize(projected, "projected_q"), "attention", "attended", INNER)
    projected_out = linear(quantize(attended, "attended_q"), "mlp", "mlp_out", hidden_size)
    quantize(projected_out, "mlp_q")

    # ④ full スロット（容量記号 `C` が現れる唯一の場所）と、それを触る effect op。
    const = declare("baked", torch.zeros(1, 1, SYM_MAX, 1, dtype=torch.int32), dtype="i32")
    del const
    values["prefix"] = IrValue(dtype="i32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(
        IrNode(
            op="sym_prefix_slice",
            ins=["baked"],
            outs=["prefix"],
            attrs={"sym": SEQ_SYMBOL, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )
    values["kv"] = IrValue(dtype="f32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(IrNode(op="cast", ins=["prefix"], outs=["kv"], attrs={"to": "f32"}))
    for slot in ("l0.k", "l0.v"):
        nodes.append(
            IrNode(op="state_append", ins=["kv"], outs=[], attrs={}, states={"slot": slot})
        )

    # ⑤ 出口は選択行の logits と hidden の 2 本。logits は共有 head が作る（SRQ 無し）。
    declare("seed", _ramp(1, 1, 1))
    hidden = "hidden"
    values[hidden] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, hidden_size])
    nodes.append(IrNode(op="expand", ins=["seed"], outs=[hidden], attrs={}))
    logits = linear(hidden, "embed", "logits", vocab)

    graph = IrGraph(
        symbols=[CAPACITY_SYMBOL, SEQ_SYMBOL, ROW_SYMBOL],
        inputs=[
            IrInput(name="input_ids", dtype="i32", shape=[1, SEQ_SYMBOL]),
            *(
                IrInput(
                    name=gemma4_rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, SEQ_SYMBOL, widths[layer_type]],
                )
                for layer_type in GEMMA4_ROPE_LAYER_TYPES
                for part in GEMMA4_ROPE_PARTS
            ),
            IrInput(name="per_layer_inputs", dtype="f32", shape=[1, SEQ_SYMBOL, layers, dim]),
            IrInput(name="last_row", dtype="i32", shape=[ROW_SYMBOL]),
        ],
        outputs=[logits, hidden],
        initializers=initializers,
        values=values,
        states={
            name: IrState(dtype="f32", shape=[1, 1, CAPACITY_SYMBOL, 1])
            for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    with TemporaryDirectory() as staging:
        written = write_model(
            Path(staging) / "model.safetensors", graph, tensors, fixed_weights=fixed
        )
        verify_shards(written)
        return [path.read_bytes() for path in written]


def reference_record(
    model: str,
    *,
    fixed_weights: int = 3,
    storage_counts: Mapping[str, int] | None = None,
    ple_shards: int = 1,
    **overrides: Any,
) -> dict[str, Any]:
    """`reference.json` の中身（既定は {@link qat_container} が実際に作る本数）。"""
    record = {
        "schema": REFERENCE_SCHEMA,
        "family": "gemma4-qat",
        "model": model,
        "checkpoint": {"files": 1},
        "maxChunkLength": MAX_CHUNK_LENGTH,
        "maxSelectedRows": MAX_SELECTED_ROWS,
        "fixedWeights": fixed_weights,
        "storageCounts": dict(
            storage_counts if storage_counts is not None else {"i2": 1, "i4": 1, "i8": 1}
        ),
        "weightFiles": 1,
        "pleShards": ple_shards,
        "upstreamUnused": {"kvCacheScales": 6, "vision": 0, "audio": 0},
    }
    record.update(overrides)
    return record


def write_series(
    source: Path,
    model: str,
    *,
    container: Sequence[bytes] | None = None,
    ple_bits: int | None = None,
    ple_rows: int = VOCAB,
    reference: Mapping[str, Any] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    text_config: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """1 系列ぶんを書き、使った `reference.json` の中身を返す。"""
    from shard_series import write_component  # conftest が張る recipe 共有ヘルパ

    source.mkdir(parents=True, exist_ok=True)
    write_component(
        source / "model.safetensors", list(container if container is not None else qat_container())
    )
    bits = ple_bits if ple_bits is not None else (4 if model == "e2b" else 2)
    write_ple(packed_embedding(bits, rows=ple_rows), LAYERS, DIM, source)
    (source / "tokenizer.json").write_text(
        json.dumps(dict(tokenizer if tokenizer is not None else tokenizer_asset(vocab=VOCAB))),
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": dict(text_config if text_config is not None else TEXT_CONFIG),
            }
        ),
        encoding="utf-8",
    )
    (source / "generation_config.json").write_text(
        json.dumps(dict(generation_config if generation_config is not None else GENERATION_CONFIG)),
        encoding="utf-8",
    )
    record = dict(reference if reference is not None else reference_record(model))
    (source / "reference.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
