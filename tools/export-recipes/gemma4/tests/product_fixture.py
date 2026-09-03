"""配布 recipe のテストが入力に使う**正当な最小の製品系列**（コンテナ + PLE sidecar + 資産）。

共有の `ir_fixtures.ir_container`（`tools/exporter/tests/`）は state スロットを持てないので、
gemma4 の門が読む形（full スロットの容量記号 `C` が **states にだけ**現れる = 入力 shape から
決まらない記号がちょうど 1 本）を作れない。ここが持つのはその差分だけで、書き出しは共有の
1 本道（`karume.emit.write_model` → `karume.verify.verify_shards`）を通る。

MUST: safetensors のバイト列も IR の規則も手で綴らない（`ir_fixtures` の同 MUST）— 規則の
写しを持つと、規則が動いた日にフィクスチャだけが古びて「テストは緑・実物だけ落ちる」になる。

MUST: **実物と違う数**にする（語彙 6・層 2・次元 3・位置上限 37・headDim 4/8）— 寸法を
焼き込んでいれば落ちる。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file

from gemma4.distribution import (
    GEMMA4_ROPE_LAYER_TYPES,
    GEMMA4_ROPE_PARTS,
    gemma4_rope_input_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.emit import write_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrState, IrStorage, IrValue
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.verify import verify_shards

#: 合成の寸法（実物は 262144 / 35 / 256）。
VOCAB = 6
LAYERS = 2
DIM = 3

#: 上流 `config.json` の `text_config` のうち、配布 recipe が読む欄だけを持つ最小形。
#: **実物と違う数**（位置上限 131072 → 37・head_dim 256/512 → 4/8・theta も別値）。
MAX_POSITION = 37
SLIDING_HEAD_DIM = 4
FULL_HEAD_DIM = 8
SLIDING_THETA = 100.0
FULL_THETA = 1000.0
PARTIAL_ROTARY_FACTOR = 0.5

TEXT_CONFIG: Mapping[str, Any] = {
    "max_position_embeddings": MAX_POSITION,
    "hidden_size": 16,
    "num_attention_heads": 4,
    "head_dim": SLIDING_HEAD_DIM,
    "global_head_dim": FULL_HEAD_DIM,
    "layer_types": [SLIDING_ATTENTION, FULL_ATTENTION],
    "rope_parameters": {
        SLIDING_ATTENTION: {"rope_type": "default", "rope_theta": SLIDING_THETA},
        FULL_ATTENTION: {
            "rope_type": "proportional",
            "rope_theta": FULL_THETA,
            "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
        },
    },
}

#: 層種別 → RoPE 派生入力の幅（{@link TEXT_CONFIG} から導いた値と一致していることが門の前提）。
ROPE_HEAD_DIMS: Mapping[str, int] = {
    SLIDING_ATTENTION: SLIDING_HEAD_DIM,
    FULL_ATTENTION: FULL_HEAD_DIM,
}

#: `sym_prefix_slice` の焼き込み定数の長さ（chunk 記号 `M` の上限）と、full スロットの容量。
SYM_MAX = 4
CAPACITY_SYMBOL = "C"
SEQ_SYMBOL = "M"

#: i4 の group 長と linear 重みの形（`ir_fixtures` と同じ理由 — 行長が group_size で割り切れる
#: 最小の形。ADR 0069 決定 2）。
GROUP_SIZE = 16
_IN = 32
_OUT = 4

#: 退役した「表を焼く」形の initializer 名（残骸の門に使う — 現行の資産には 1 本も無い）。
BAKED_ROPE_TABLE = "model.model.rotary_emb.full_attention_cos_table"

#: PLE sidecar の綴り（`gemma4.export_product` / `packages/models/src/gemma/ple.ts` の正本）。
PLE_INDEX_FILE = "ple.json"
PLE_SCHEMA = 1
PLE_METADATA_KEY = "karume_ple"
PLE_EMBED_SCALE = 2.0

#: compile 済みトークナイザ資産の形式識別子（`_shared/gemma_tokenizer.py`）。
TOKENIZER_FORMAT = "karume-gemma-tokenizer/1"

#: 上流 `generation_config.json` の推奨（**実物と違う値** — 写経していれば落ちる）。
GENERATION_CONFIG: Mapping[str, Any] = {
    "do_sample": True,
    "eos_token_id": [1, 2],
    "temperature": 0.7,
    "top_k": 8,
    "top_p": 0.5,
}


def _ramp(*shape: int) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    return torch.arange(total, dtype=torch.float32).reshape(*shape) / total


def product_container(
    *,
    vocab: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    head_dims: Mapping[str, int] | None = None,
    baked_rope: bool = False,
    free_symbol: bool = True,
) -> list[bytes]:
    """製品グラフ 1 本ぶんの shard バイト列（読む順 — 先頭がグラフ shard）。

    `head_dims` は RoPE 派生入力の幅の上書き（宣言と食い違う世代を作る門のため）。
    `baked_rope` は退役した「表を焼く」形の initializer を 1 本混ぜる（残骸の門）。
    `free_symbol` を偽にすると容量記号を states から外し、`M` の 1 本だけにする
    （記号の割れ方の門）。
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"gemma4.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        # 記号依存定数だけが i32（`ir_fixtures` と同じ — 他は意味論 f32）。
        values[name] = IrValue(dtype="i32" if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    # ① 実行の骨（linear 1 本）— i4 適格な重みと、その相方の i8。実物の混成格納
    #    （埋め込みが i8・linear が packed i4）と同じ dtype 集合をヘッダに作る。
    activation = "x"
    bias = "bias"
    declare(activation, _ramp(1, _IN))
    declare(bias, _ramp(_OUT))
    for name, storage in (("linear_i4", "i4"), ("embed_i8", "i8")):
        weight = _ramp(_OUT, _IN)
        if storage == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
            key = declare(name, weight)
            scales[key] = scale
        else:
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
            key = declare(name, weight)
            scales[key] = scale
            overrides[key] = "i8"
        out = f"h_{name}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=[activation, name, bias], outs=[out], attrs={}))

    # ② 退役形の残骸（既定では入れない — 門が「1 本も無い」を見るための対照）。
    if baked_rope:
        declare(BAKED_ROPE_TABLE, _ramp(2, 2))

    # ③ 記号次元の席: `M` は入力 shape が束縛し、`C` は states にだけ現れる。
    const = "baked"
    key = declare(const, torch.zeros(1, 1, SYM_MAX, 1, dtype=torch.int32), dtype="i32")
    del key
    prefix = "prefix"
    values[prefix] = IrValue(dtype="i32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(
        IrNode(
            op="sym_prefix_slice",
            ins=[const],
            outs=[prefix],
            attrs={"sym": SEQ_SYMBOL, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
        )
    )

    # ④ full スロット（容量記号 `C` が現れる唯一の場所）と、それを触る effect op 1 本。
    #    宣言だけのスロットは IR が拒否する（`verify._check_state_slots` — 参照完全性）ので、
    #    実物の attention の代わりに単機能の `state_append`（入力 1 本・出力 0 本）を置く。
    kv = "kv"
    values[kv] = IrValue(dtype="f32", shape=[1, 1, SEQ_SYMBOL, 1])
    nodes.append(IrNode(op="cast", ins=[prefix], outs=[kv], attrs={"to": "f32"}))
    nodes.append(IrNode(op="state_append", ins=[kv], outs=[], attrs={}, states={"slot": "l0.k"}))

    # ⑤ 出口は最終行 logits 1 本（`[1, 1, V]` — ADR 0083 決定 6）。
    seed = "logits_seed"
    declare(seed, _ramp(1, 1, 1))
    logits = "logits"
    values[logits] = IrValue(dtype="f32", shape=[1, 1, vocab])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[logits], attrs={}))

    # `free_symbol` を偽にすると容量が具体数になり、記号は `M` の 1 本だけになる。
    capacity_dim: str | int = CAPACITY_SYMBOL if free_symbol else SYM_MAX
    graph = IrGraph(
        symbols=[CAPACITY_SYMBOL, SEQ_SYMBOL] if free_symbol else [SEQ_SYMBOL],
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
            IrInput(name="last_row", dtype="i32", shape=[1]),
        ],
        outputs=[logits],
        initializers=initializers,
        values=values,
        states={"l0.k": IrState(dtype="f32", shape=[1, 1, capacity_dim, 1])},
        nodes=nodes,
    )
    with TemporaryDirectory() as staging:
        written = write_model(
            Path(staging) / "model.safetensors",
            graph,
            tensors,
            weight_dtype="i4",
            weight_scales=scales,
            weight_dtype_overrides=overrides,
        )
        verify_shards(written)
        return [path.read_bytes() for path in written]


def ple_shard_bytes(
    start: int, stop: int, index: Mapping[str, Any], *, metadata: Mapping[str, Any] | None
) -> bytes:
    """PLE sidecar shard 1 本（token-major の i8 値 + per-row f32 scale）。

    `metadata` を明示すると `__metadata__.karume_ple` の中身を差し替えられる（索引と食い違う
    組み合わせの門）。`None` は索引そのままの正当な写し。
    """
    rows = stop - start
    layers = int(index["layers"])  # type: ignore[arg-type]
    dim = int(index["dim"])  # type: ignore[arg-type]
    declared = dict(
        metadata
        if metadata is not None
        else {
            "schema": PLE_SCHEMA,
            "tokens": index["tokens"],
            "layers": layers,
            "dim": dim,
            "embedScale": index["embedScale"],
            "start": start,
            "stop": stop,
        }
    )
    values = np.arange(rows * layers * dim, dtype=np.int8).reshape(rows, layers, dim)
    scales = np.full((rows, layers), 0.5, dtype=np.float32)
    with TemporaryDirectory() as staging:
        path = Path(staging) / "ple.safetensors"
        save_file(
            {"values": values, "scales": scales},
            str(path),
            metadata={PLE_METADATA_KEY: json.dumps(declared)},
        )
        return path.read_bytes()


def ple_index(
    ranges: Sequence[tuple[int, int]],
    *,
    tokens: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
) -> dict[str, Any]:
    """`ple.json` の中身（ファイル名は実物と同じ連番の綴り）。"""
    total = len(ranges)
    return {
        "schema": PLE_SCHEMA,
        "tokens": tokens,
        "layers": layers,
        "dim": dim,
        "embedScale": PLE_EMBED_SCALE,
        "shards": [
            {
                "file": f"ple-{position + 1:05d}-of-{total:05d}.safetensors",
                "start": start,
                "stop": stop,
            }
            for position, (start, stop) in enumerate(ranges)
        ],
    }


def tokenizer_asset(*, vocab: int = VOCAB, format_id: str = TOKENIZER_FORMAT) -> dict[str, Any]:
    """compile 済みトークナイザ資産の、門が読む欄だけを持つ最小形。"""
    return {"format": format_id, "vocab": [f"t{index}" for index in range(vocab)]}


def write_series(
    product: Path,
    tokenizer_dir: Path,
    model_dir: Path,
    *,
    container: Sequence[bytes] | None = None,
    index: Mapping[str, Any] | None = None,
    shard_metadata: Mapping[int, Mapping[str, Any]] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    text_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """系列 2 本 + チェックポイントを書き、使った `ple.json` の中身を返す。

    `shard_metadata` は shard 位置 → `__metadata__.karume_ple` の差し替え（門のため）。
    製品系列には**配布へ入らない**同居物（`ple.probe.safetensors` / `reference.json`）も置く
    — 出力 path 表に載らないものが混ざらないことの証跡になる。
    """
    from shard_series import write_component  # conftest が張る recipe 共有ヘルパ

    shards = list(container if container is not None else product_container())
    write_component(product / "model.safetensors", shards)
    declared = dict(index if index is not None else ple_index([(0, 4), (4, VOCAB)]))
    (product / PLE_INDEX_FILE).write_text(
        json.dumps(declared, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for position, shard in enumerate(declared["shards"]):  # type: ignore[arg-type]
        override = (shard_metadata or {}).get(position)
        (product / str(shard["file"])).write_bytes(
            ple_shard_bytes(int(shard["start"]), int(shard["stop"]), declared, metadata=override)
        )
    (product / "ple.probe.safetensors").write_bytes(b"not distributed")
    (product / "reference.json").write_text("{}\n", encoding="utf-8")

    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps(dict(tokenizer if tokenizer is not None else tokenizer_asset())),
        encoding="utf-8",
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "generation_config.json").write_text(
        json.dumps(dict(generation_config if generation_config is not None else GENERATION_CONFIG)),
        encoding="utf-8",
    )
    # 上流の `config.json` は multimodal の器で、text 部は `text_config` 節（実物と同じ形）。
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": dict(text_config if text_config is not None else TEXT_CONFIG),
            }
        ),
        encoding="utf-8",
    )
    return declared
