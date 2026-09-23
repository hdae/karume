"""配布 recipe のテストが入力に使う**正当な最小の製品系列**（コンテナ + PLE の資産 + 資産）。

共有の `ir_fixtures.ir_container`（`tools/exporter/tests/`）は state スロットを持てないので、
gemma4 の門が読む形（full スロットの容量記号 `C` が **states にだけ**現れる = 入力 shape から
決まらない記号がちょうど 1 本）を作れない。ここが持つのはその差分だけで、書き出しは共有の
1 本道（`karume.emit.stored_model` → `karume.publish.publish_container`）を通る。

MUST: 容器のバイト列も IR の規則も手で綴らない（`ir_fixtures` の同 MUST）— 規則の写しを
持つと、規則が動いた日にフィクスチャだけが古びて「テストは緑・実物だけ落ちる」になる。

MUST: **実物と違う数**にする（語彙 6・層 2・次元 3・hidden 5・位置上限 37・headDim 4/8）—
寸法を焼き込んでいれば落ちる。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from gemma4.distribution import (
    GEMMA4_DEFAULT_MODEL,
    GEMMA4_DRAFT_STEPS,
    GEMMA4_DRAFTER_SUFFIX,
    GEMMA4_ROPE_LAYER_TYPES,
    GEMMA4_ROPE_PARTS,
    gemma4_rope_input_name,
    gemma4_series_name,
)
from gemma4.rope import FULL_ATTENTION, SLIDING_ATTENTION
from karume.container import BLOCK_MAX_BYTES, AssetInput, Provenance
from karume.emit import stored_model
from karume.ir import (
    IrGraph,
    IrInitializer,
    IrInput,
    IrNode,
    IrShared,
    IrState,
    IrStorage,
    IrValue,
)
from karume.ple import ple_assets
from karume.publish import publish_container
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)

#: フィクスチャの出所（`--license` を落とした配布形は作らない — container-v1 §12）。
FIXTURE_PROVENANCE = Provenance(license="apache-2.0", writer="karume-fixture")

#: 合成の寸法（実物は 262144 / 35 / 256 / 2048）。`HIDDEN` は hidden 出口の幅で、**VOCAB とも
#: DIM とも違う数**にする（logits と hidden を取り違えた組が幅で落ちる）。
#: MUST: `HIDDEN` は {@link TEXT_CONFIG} の `hidden_size` と同じ数 — 配布 recipe は上流の宣言と
#: グラフの幅を突き合わせるので、フィクスチャの中で割れていると正当な組が組めない。
VOCAB = 6
LAYERS = 2
#: PLE の 1 行は **4 の倍数**でなければ block を行の倍数で切れない（container-v1 §4.1）ので、
#: `LAYERS * DIM` が 4 で割り切れる数にする（実物は 35 × 256）。
DIM = 10
HIDDEN = 16

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
    "hidden_size": HIDDEN,
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

#: 出口の行数記号（`last_row[R]` が束縛する — `gemma4.export_product.ROW_SYMBOL` の綴り）。
ROW_SYMBOL = "R"

#: i4 の group 長と linear 重みの形（`ir_fixtures` と同じ理由 — 行長が group_size で割り切れる
#: 最小の形。ADR 0069 決定 2）。
GROUP_SIZE = 16
_IN = 32
_OUT = 4

#: 退役した「表を焼く」形の initializer 名（残骸の門に使う — 現行の資産には 1 本も無い）。
BAKED_ROPE_TABLE = "model.model.rotary_emb.full_attention_cos_table"

#: PLE の綴り（`karume.ple` / `packages/models/src/gemma/ple-index.ts` の正本）。
PLE_STORAGE = "i8"
PLE_EMBED_SCALE = 2.0

#: 貸し手の主表テンソルキー（{@link product_container} の `declare` が組む綴り）。借り手の
#: 共有 initializer はこれを名指す — 食い違えば配布の門が落とす。
LENDER_SHARED_TENSOR = "gemma4.embed_i8"

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


def _publish(
    graph: IrGraph,
    tensors: Mapping[str, torch.Tensor],
    storage: str,
    scales: Mapping[str, torch.Tensor],
    overrides: Mapping[str, str],
    *,
    assets: Mapping[str, AssetInput] = {},
    block_bytes: int = BLOCK_MAX_BYTES,
) -> list[bytes]:
    """書いて読み直して検証し、part ごとのバイト列を添字順に返す（実物と同じ 1 本道）。"""
    stored = stored_model(
        graph,
        tensors,
        weight_dtype=storage,
        weight_scales=scales,
        weight_dtype_overrides=overrides,
    )
    with TemporaryDirectory() as staging:
        result = publish_container(
            Path(staging) / "model.krm",
            stored.graph,
            stored.tensors,
            stored.bindings,
            graph_name="model",
            provenance=FIXTURE_PROVENANCE,
            assets=assets,
            block_bytes=block_bytes,
        )
        return [path.read_bytes() for path in result.parts]


def ple_container_assets(
    *,
    tokens: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    embed_scale: float = PLE_EMBED_SCALE,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> dict[str, AssetInput]:
    """PLE を容器の資産へ（索引 schema 3 + `values` / `scales` の block 列）。

    切り方も索引の綴りも core の {@link karume.ple.ple_assets} が持つ — 写しを綴ると、規則が
    動いた日にフィクスチャだけが古びる。`block_bytes` を下げると block が複数本へ割れる
    （索引と資産宣言の突合を複数 block で踏む席）。
    """
    values = np.arange(tokens * layers * dim, dtype=np.int8).reshape(tokens, layers * dim)
    scales = np.full((tokens, layers), 0.5, dtype=np.float32)
    value_bytes = memoryview(values).cast("B")
    scale_bytes = memoryview(scales).cast("B")
    return ple_assets(
        storage=PLE_STORAGE,
        tokens=tokens,
        layers=layers,
        dim=dim,
        embed_scale=embed_scale,
        read_values=lambda begin, end: value_bytes[begin:end],
        read_scales=lambda begin, end: scale_bytes[begin:end],
        block_bytes=block_bytes,
    )


def product_container(
    *,
    vocab: int = VOCAB,
    layers: int = LAYERS,
    dim: int = DIM,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    baked_rope: bool = False,
    free_symbol: bool = True,
    swap_outputs: bool = False,
    assets: Mapping[str, AssetInput] | None = None,
) -> list[bytes]:
    """製品グラフ 1 本ぶんの part バイト列（読む順 — 先頭が part 0）。

    PLE は容器の**資産**として同梱される（ADR 0109 決定 4）— `assets` を渡さなければ
    {@link ple_container_assets} の既定を載せる。

    `head_dims` は RoPE 派生入力の幅の上書き（宣言と食い違う世代を作る門のため）。
    `baked_rope` は退役した「表を焼く」形の initializer を 1 本混ぜる（残骸の門）。
    `free_symbol` を偽にすると容量記号を states から外し、入力 shape が束縛する `M` / `R` の
    2 本だけにする（記号の割れ方の門 — 自由記号がちょうど 1 本であることを見る側）。
    `swap_outputs` は出口 2 本の順序だけを入れ替える（行軸まで同型なので、幅の突合以外は
    素通りする組 — 順序の門）。
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
    for slot in ("l0.k", "l0.v"):
        nodes.append(IrNode(op="state_append", ins=[kv], outs=[], attrs={}, states={"slot": slot}))

    # ⑤ 出口は選択行の logits と hidden の 2 本（`[1, R, V]` / `[1, R, H]` — ADR 0083 決定 6 +
    #    投機 verify の足場）。順序は logits → hidden 固定。
    seed = "logits_seed"
    declare(seed, _ramp(1, 1, 1))
    logits = "logits"
    values[logits] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, vocab])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[logits], attrs={}))
    hidden = "hidden"
    values[hidden] = IrValue(dtype="f32", shape=[1, ROW_SYMBOL, hidden_size])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[hidden], attrs={}))

    # `free_symbol` を偽にすると容量が具体数になり、記号は `M` の 1 本だけになる。
    capacity_dim: str | int = CAPACITY_SYMBOL if free_symbol else SYM_MAX
    graph = IrGraph(
        symbols=(
            [CAPACITY_SYMBOL, SEQ_SYMBOL, ROW_SYMBOL] if free_symbol else [SEQ_SYMBOL, ROW_SYMBOL]
        ),
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
        outputs=[hidden, logits] if swap_outputs else [logits, hidden],
        initializers=initializers,
        values=values,
        # k / v の対で持つ — 借り手（drafter）の readonly attention は `{k, v}` ちょうどを
        # 要求するので、貸し手が k だけを宣言していると正当な組が作れない。
        states={
            name: IrState(dtype="f32", shape=[1, 1, capacity_dim, 1]) for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    return _publish(
        graph,
        tensors,
        "i4",
        scales,
        overrides,
        assets=ple_container_assets(tokens=vocab, layers=layers, dim=dim)
        if assets is None
        else assets,
    )


def drafter_container(
    *,
    vocab: int = VOCAB,
    hidden_size: int = HIDDEN,
    head_dims: Mapping[str, int] | None = None,
    steps: int = GEMMA4_DRAFT_STEPS,
    external: bool = True,
    shared: bool = True,
    capacity_symbol: str | None = CAPACITY_SYMBOL,
    storage: str = "i8",
) -> list[bytes]:
    """**借り手**グラフ 1 本ぶんの part バイト列（ADR 0096 段 2 — 単独では実行できない資産）。

    貸し手（{@link product_container}）と噛み合う形にする: 同じスロット名 `l0.k` / `l0.v` を
    **external** で宣言し、共有 initializer が貸し手の主表テンソルキーを名指し、記号は容量の
    1 本だけ。`external` / `shared` / `capacity_symbol` は配布の門
    （{@link gemma4.distribution.assert_gemma4_drafter_graph}）の故障注入の口。

    実物の drafter は **linear まで i8 の単一格納**（`gemma4/export_drafter.py` の実測 — i4 に
    落とすと受理率が 1 〜 3 割落ちる）なので、既定の `storage` は `"i8"`。`"i4"` は格納の門
    （`GEMMA4_STORAGE_FORBIDDEN`）の故障注入の口で、**出力ヘッドだけ i8 のまま linear が i4 に
    落ちた資産**を作る — 存在検査（I8 が在る）では素通りする側。
    """
    widths = {**ROPE_HEAD_DIMS, **dict(head_dims or {})}
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}
    nodes: list[IrNode] = []

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"drafter.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype="f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    # ① 実物の格納形（linear が `storage`・出力ヘッドは常に i8）— ヘッダの dtype 集合が
    #    そのまま格納の門の入力になる。
    declare("x", _ramp(1, _IN))
    declare("bias", _ramp(_OUT))
    for name, slot_storage in ((f"linear_{storage}", storage), ("head_i8", "i8")):
        weight = _ramp(_OUT, _IN)
        if slot_storage == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
            scales[declare(name, weight)] = scale
        else:
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
            key = declare(name, weight)
            scales[key] = scale
            overrides[key] = "i8"
        out = f"h_{name}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=["x", name, "bias"], outs=[out], attrs={}))

    # ② 共有 initializer（バイト無し）— 指し先は貸し手コンテナのテンソルキー。
    embedded = "embedded"
    if shared:
        initializers["target_embed"] = IrInitializer(
            shared=IrShared(tensor=LENDER_SHARED_TENSOR), storage=IrStorage(dtype="i8")
        )
        values["target_embed"] = IrValue(dtype="f32", shape=[vocab, hidden_size])
        values[embedded] = IrValue(dtype="f32", shape=[1, 1, hidden_size])
        nodes.append(
            IrNode(
                op="embedding",
                ins=["target_embed", "token"],
                outs=[embedded],
                attrs={"padding_idx": -1},
            )
        )

    # ③ readonly attention（q 1 本・external スロットの k / v を読む）。
    seed = "q_seed"
    declare(seed, _ramp(1, 1, 1, 1))
    query = "q"
    values[query] = IrValue(dtype="f32", shape=[1, 1, 1, 1])
    nodes.append(IrNode(op="expand", ins=[seed], outs=[query], attrs={}))
    attended = "attended"
    values[attended] = IrValue(dtype="f32", shape=[1, 1, 1, 1])
    nodes.append(
        IrNode(
            op="attention",
            ins=[query],
            outs=[attended],
            attrs={"scale": 1.0, "readonly": True},
            states={"k": "l0.k", "v": "l0.v"},
        )
    )

    # ④ 出口は draft k 本（実物は argmax の token 3 本 — 本数だけが配布の門の関心事）。
    outputs: list[str] = []
    for index in range(steps):
        name = f"token_{index}"
        values[name] = IrValue(dtype="f32", shape=[1, 1, 1])
        nodes.append(IrNode(op="reshape", ins=[attended], outs=[name], attrs={}))
        outputs.append(name)

    capacity_dim: str | int = capacity_symbol if capacity_symbol is not None else SYM_MAX
    graph = IrGraph(
        symbols=[capacity_symbol] if capacity_symbol is not None else [],
        inputs=[
            IrInput(name="token", dtype="i32", shape=[1, 1]),
            IrInput(name="hidden", dtype="f32", shape=[1, hidden_size]),
            *(
                IrInput(
                    name=gemma4_rope_input_name(layer_type, part),
                    dtype="f32",
                    shape=[1, 1, widths[layer_type]],
                )
                for layer_type in GEMMA4_ROPE_LAYER_TYPES
                for part in GEMMA4_ROPE_PARTS
            ),
        ],
        outputs=outputs,
        initializers=initializers,
        values=values,
        states={
            name: IrState(dtype="f32", shape=[1, 1, capacity_dim, 1], external=external)
            for name in ("l0.k", "l0.v")
        },
        nodes=nodes,
    )
    return _publish(graph, tensors, storage, scales, overrides)


def tokenizer_asset(*, vocab: int = VOCAB, format_id: str = TOKENIZER_FORMAT) -> dict[str, Any]:
    """compile 済みトークナイザ資産の、門が読む欄だけを持つ最小形。"""
    return {"format": format_id, "vocab": [f"t{index}" for index in range(vocab)]}


def write_series(
    product: Path,
    tokenizer_dir: Path,
    model_dir: Path,
    *,
    container: Sequence[bytes] | None = None,
    drafter: Path | None = None,
    drafter_bytes: Sequence[bytes] | None = None,
    tokenizer: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    text_config: Mapping[str, Any] | None = None,
) -> None:
    """系列 2 本 + チェックポイントを書く。

    製品系列には**配布へ入らない**同居物（`ple.probe.safetensors` / `reference.json`）も置く
    — 出力 path 表に載らないものが混ざらないことの証跡になる。PLE は製品コンテナの**資産**
    なので、系列に独立したファイルとしては現れない（ADR 0109 決定 4）。
    """
    from container_series import write_component  # conftest が張る recipe 共有ヘルパ

    parts = list(container if container is not None else product_container())
    write_component(product / "model.krm", parts)
    # 借り手（drafter）系列は既定で product の隣に置く（既存の呼び出しを 1 つも書き換えずに
    # 済ませるため — 系列名の綴りは配布 recipe が持つ 1 箇所から組む）。
    borrower = (
        drafter
        if drafter is not None
        else product.parent / gemma4_series_name(GEMMA4_DEFAULT_MODEL, GEMMA4_DRAFTER_SUFFIX)
    )
    write_component(
        borrower / "model.krm",
        list(drafter_bytes if drafter_bytes is not None else drafter_container()),
    )
    # 配布へ入らない同居物（golden と出所記録）— 出力 path 表に載らないことの証跡。
    (borrower / "drafter-golden.short-en.safetensors").write_bytes(b"not distributed")
    (borrower / "reference.json").write_text("{}\n", encoding="utf-8")
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
