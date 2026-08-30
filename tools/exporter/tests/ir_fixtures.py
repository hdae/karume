"""組み立て層のテストが入力に使う**正当な最小 IR コンポーネント**の合成。

返すのは 1 ファイルではなく **shard 列**（先頭 = グラフ shard・以降 = weight shard）— 配布形は
常に分割されるので（ADR 0081）、単一ファイルのフィクスチャは実物と同じ形ではない
（グラフ shard 空の門に落ちる）。

`karume.dist` の組み立ては入力コンテナを IR v1 の全規則で見る
（{@link karume.dist.assert_weight_components_verified}）ので、weights の席へ挿すフィクスチャは
**本物のコンテナ**でなければならない。ヘッダだけ手で綴った偽物が通るのは格納 dtype の門
（ヘッダの dtype 集合）までで、その先の parse / ランタイム支援 / 契約 / 宣言と実体の突合には
届かない。

置き場がここ 1 箇所なのは、消費者が 2 つの木にまたがるため — core 側の
`tools/exporter/tests/test_dist.py` と、リポ専用 recipe の
`tools/export-recipes/<family>/tests/test_distribution.py`（後者へは
`tools/export-recipes/conftest.py` が sys.path を張る）。依存の向きは recipe → core の一方向の
まま（ADR 0065 決定 3）で、wheel の中身（`src/karume/`）は 1 バイトも増えない。

MUST: safetensors のバイト列を手で綴らない — 書き出しは `karume.emit.write_model` の 1 本道で、
作ったバイト列は `karume.verify.verify_shards` を通してから返す。規則の写しを持つと、規則が
動いた日にフィクスチャだけが古びて「テストは緑・実物だけ落ちる」になる。

同じ引数からは**同じバイト列**が出る（乱数を使わない）— 役割ごとのバイト列の違いは
`mark`（テンソルキーの接頭辞）から出るので、モジュール定数として持てる。

NOTE: グラフ入力は全て意味論 f32 で宣言する。門が読むのは名前と shape（と本数）だけで、
入力自体はどのノードも消費しない**宣言だけの席**なので、i32 の添字入力を名乗り分ける理由が
無い。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from karume.dims import parse_dim
from karume.emit import write_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
from karume.quantize import (
    channel_scale,
    dequantize_int4,
    group_scale,
    quantize_to_int4,
    quantize_to_int8,
)
from karume.verify import verify_shards

#: linear 重みの形と i4 の group 長。行長 32 は group 16 の 2 本ぶん（格納 i4 の「行長が
#: group_size で割り切れる」— ADR 0069 決定 2 — を満たす最小の形）。
GROUP_SIZE = 16
_IN = 32
_OUT = 4

#: 分割フィクスチャ 1 shard ぶんのテンソル（f32 16 要素 = 64 バイト）。
_FILL_ELEMENTS = 16

#: 合成が自分で足す席の接頭辞。recipe の門は initializer 名から層番号を数える
#: （SBV2）ので、呼び出し側が渡す名前と混ざらない綴りにしておく。
_OWN = "karume_fixture_"

Shape = Sequence[int | str]


def _ramp(*shape: int) -> torch.Tensor:
    """決定的な小さい値のテンソル（乱数を使わない — 同じ spec は同じバイト列 MUST）。

    値域を ±3 に取るのは、f16 でも i8 でも i4 でも**丸めが恒等**になり、量子化の scale が
    0 にならない（= 0 除算にならない）形だから。
    """
    count = 1
    for dim in shape:
        count *= dim
    return (torch.arange(count, dtype=torch.float32) % 7 - 3).reshape(shape)


def _symbols(shapes: Sequence[Shape]) -> list[str]:
    """宣言 shape に現れる記号（`2T` / `S+24` のような派生形も素の記号へ畳む）。

    束縛点（入力 shape の次元位置）を持たない記号を宣言すると parse が落ちるので、記号は
    呼び出し側に別欄で名乗らせず**現れた shape から導く**。
    """
    found = {parse_dim(dim).sym for shape in shapes for dim in shape if isinstance(dim, str)}
    return sorted(found)


def _write(
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    storage: str,
    scales: dict[str, torch.Tensor],
    overrides: dict[str, str],
    limit: int | None = None,
) -> list[bytes]:
    """書いて検証して、shard ごとのバイト列を読む順に返す。"""
    with TemporaryDirectory() as staging:
        written = write_model(
            Path(staging) / "model.safetensors",
            graph,
            tensors,
            weight_dtype=storage,
            weight_scales=scales,
            weight_dtype_overrides=overrides,
            _shard_byte_limit=limit,
        )
        verify_shards(written)
        return [path.read_bytes() for path in written]


def ir_container(
    *,
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
) -> list[bytes]:
    """正当な IR コンポーネント 1 つぶんの shard バイト列（読む順 — 先頭がグラフ shard）。

    配布形は常に分割されるので（ADR 0081）、返るのは**必ず 2 要素以上**: 先頭が
    `karume_ir` だけを持つグラフ shard（データ節は空）、以降が weight shard。合成の資産は
    小さいので weight shard は 1 本に収まり、戻り値は 2 要素になる。呼び出し側は
    `karume.shards.shard_name` で連番のファイル名を作って書き出す。

    `storage` は系列の格納形（`f32` / `f16` / `i8` / `i4`）。実物と同じく**適格な重みスロット
    だけ**が圧縮格納になり、bias やグラフ定数（i8 / i4 の scale も）は F32 のまま残るので、
    ヘッダの dtype 集合は実配布資産と同じ形になる。`i4` は**混成**（i4 適格な重みが I4・
    残りが I8・その他 F32）— 実物の i4 系列がこの形で、単一 dtype の器では「圧縮席どうしの
    取り違え」を再現できない。

    `inputs`（名前と shape）と `outputs`（出力ごとの宣言 shape）は family 固有の門が読む席。
    入力はどのノードも消費しない宣言だけの席で、出力は小さな定数を `expand` した値なので、
    2048×2048 のような宣言でも実バイトは数十バイトのまま。

    `weights` は linear の重みになる initializer 名（層数を数える門が読む綴り）。
    `baked` は `(記号名, 焼き込み上限)` で、`sym_prefix_slice` の焼き込み定数を 1 本足す
    （記号は `inputs` の次元位置で束縛されている必要がある）。
    """
    graph, tensors, scales, overrides = _spec(mark, storage, inputs, outputs, weights, baked)
    return _write(graph, tensors, storage=storage, scales=scales, overrides=overrides)


def ir_shards(count: int, *, mark: str) -> list[bytes]:
    """`count` 本の shard 列になる正当なコンポーネント（読む順 — 先頭がグラフ shard）。

    何本に割れるかは現物のバイト数が決める（`karume.shards`）ので、同じ大きさのテンソルを
    `count - 1` 本並べ、テスト用の上限差し込み（`write_model` の `_shard_byte_limit`）を
    その 1 本ぶんに合わせて **1 weight shard = 1 テンソル**へ割り付ける。`count` は
    グラフ shard を含む総数なので 2 以上（グラフだけの列は weights の席に置けない）。
    """
    if count < 2:
        raise ValueError(f"shard 数 {count} は 2 以上（先頭はテンソルを持たないグラフ shard）")
    initializers = {}
    values = {}
    tensors = {}
    names = [f"{_OWN}fill{index}" for index in range(count - 1)]
    for name in names:
        initializers[name] = IrInitializer(tensor=f"{mark}.{name}", storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=[_FILL_ELEMENTS])
        tensors[f"{mark}.{name}"] = _ramp(_FILL_ELEMENTS)
    graph = IrGraph(initializers=initializers, values=values, outputs=names[:1])
    return _write(
        graph,
        tensors,
        storage="f32",
        scales={},
        overrides={},
        limit=_FILL_ELEMENTS * 4,
    )


def _spec(
    mark: str,
    storage: str,
    inputs: Sequence[tuple[str, Shape]],
    outputs: Sequence[Shape],
    weights: Sequence[str],
    baked: tuple[str, int] | None,
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """spec からグラフ・格納テンソル・scale 台帳・格納 dtype の 1 本単位指定を組む。"""
    initializers: dict[str, IrInitializer] = {}
    values: dict[str, IrValue] = {}
    nodes: list[IrNode] = []
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    overrides: dict[str, str] = {}

    def declare(name: str, tensor: torch.Tensor, dtype: str = "f32") -> str:
        key = f"{mark}.{name}"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype=dtype))
        values[name] = IrValue(dtype=dtype if dtype == "i32" else "f32", shape=list(tensor.shape))
        tensors[key] = tensor
        return key

    activation = f"{_OWN}x"
    bias = f"{_OWN}bias"
    declare(activation, _ramp(1, _IN))
    declare(bias, _ramp(_OUT))
    # i4 系列は混成（i4 適格外の重みが i8 のまま残る）— その相方をここで 1 本持つ。
    rest = f"{_OWN}rest"
    for index, name in enumerate([*weights, *([rest] if storage == "i4" else [])]):
        weight = _ramp(_OUT, _IN)
        dtype = "i8" if name == rest else storage
        scale: torch.Tensor | None = None
        if dtype == "f16":
            weight = weight.to(torch.float16).to(torch.float32)
        elif dtype == "i8":
            scale = channel_scale(weight, 0)
            weight = quantize_to_int8(weight, scale).to(torch.float32) * scale
        elif dtype == "i4":
            scale = group_scale(weight, GROUP_SIZE)
            weight = dequantize_int4(quantize_to_int4(weight, scale), scale)
        key = declare(name, weight)
        if scale is not None:
            scales[key] = scale
        if name == rest:
            overrides[key] = "i8"
        out = f"{_OWN}h{index}"
        values[out] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="linear", ins=[activation, name, bias], outs=[out], attrs={}))

    declared: list[Shape] = [shape for _, shape in inputs]
    graph_outputs: list[str] = []
    for index, shape in enumerate(outputs):
        seed = f"{_OWN}seed{index}"
        declare(seed, _ramp(*([1] * len(shape))))
        out = f"out_{index}"
        values[out] = IrValue(dtype="f32", shape=list(shape))
        nodes.append(IrNode(op="expand", ins=[seed], outs=[out], attrs={}))
        graph_outputs.append(out)
        declared.append(shape)

    if baked is not None:
        symbol, sym_max = baked
        const = f"{_OWN}baked"
        declare(const, torch.zeros(1, 1, sym_max, 1, dtype=torch.int32), dtype="i32")
        prefix = f"{_OWN}prefix"
        prefix_shape: list[int | str] = [1, 1, symbol, 1]
        values[prefix] = IrValue(dtype="i32", shape=prefix_shape)
        nodes.append(
            IrNode(
                op="sym_prefix_slice",
                ins=[const],
                outs=[prefix],
                attrs={"sym": symbol, "slices": [{"dim": 2, "coeff": 1, "offset": 0}]},
            )
        )
        declared.append(prefix_shape)

    graph = IrGraph(
        symbols=_symbols(declared),
        inputs=[IrInput(name=name, dtype="f32", shape=list(shape)) for name, shape in inputs],
        outputs=graph_outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    return graph, tensors, scales, overrides
