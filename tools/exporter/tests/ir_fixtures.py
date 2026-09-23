"""組み立て層のテストが入力に使う**正当な最小コンテナ**（`krm`）の合成。

返すのは 1 ファイルではなく **part 列**（part 0 = 2 文書・part 1 = const 領域・part 2 以降 =
重み）— HF の公式配布は分割形だけなので（container-v1 §8）、単一形のフィクスチャは実物と
同じ形ではない（`karume/5` の `container.parts` は 2 要素以上 MUST）。

`karume.dist` の組み立ては入力コンテナを宣言の全規則で見る
（{@link karume.dist.assert_weight_components_verified}）ので、weights の席へ挿すフィクスチャは
**本物のコンテナ**でなければならない。手で綴った偽物では descriptor の構造検査にも合流にも
届かない。

置き場がここ 1 箇所なのは、消費者が 2 つの木にまたがるため — core 側の
`tools/exporter/tests/test_dist.py` と、リポ専用 recipe の
`tools/export-recipes/<family>/tests/test_distribution.py`（後者へは
`tools/export-recipes/conftest.py` が sys.path を張る）。依存の向きは recipe → core の一方向の
まま（ADR 0065 決定 3）で、wheel の中身（`src/karume/`）は 1 バイトも増えない。

MUST: バイト列を手で綴らない — 書き出しは `karume.emit.stored_model` →
`karume.publish.publish_container` の 1 本道を通す（読み直し検証込み）。規則の写しを持つと、
規則が動いた日にフィクスチャだけが古びて「テストは緑・実物だけ落ちる」になる。

同じ引数からは**同じバイト列**が出る（乱数を使わない）— 役割ごとのバイト列の違いは
`mark`（テンソルキーの接頭辞）から出るので、モジュール定数として持てる。

NOTE: グラフ入力は全て意味論 f32 で宣言する。門が読むのは名前と shape（と本数）だけで、
入力自体はどのノードも消費しない**宣言だけの席**なので、i32 の添字入力を名乗り分ける理由が
無い。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from karume.container import (
    BLOCK_MAX_BYTES,
    DEFAULT_PART_BYTES,
    HEADER_BYTES,
    AssetInput,
    Provenance,
    canonical_json,
    read_header,
    write_header,
)
from karume.dims import parse_dim
from karume.emit import stored_model
from karume.ir import IrGraph, IrInitializer, IrInput, IrNode, IrStorage, IrValue
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

    束縛点（入力 shape の次元位置）を持つ記号は**現れた shape から導く** — 呼び出し側に別欄で
    名乗らせると、shape と宣言が独立に動く（`ir_container(symbols=…)` は束縛点を持たない記号を
    故意に名乗る故障注入の席で、そちらだけが別欄を使う）。
    """
    found = {parse_dim(dim).sym for shape in shapes for dim in shape if isinstance(dim, str)}
    return sorted(found)


def _write(
    graph: IrGraph,
    tensors: dict[str, torch.Tensor],
    *,
    mark: str,
    storage: str,
    scales: dict[str, torch.Tensor],
    overrides: dict[str, str],
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> list[bytes]:
    """書いて読み直して検証し、part ごとのバイト列を添字順に返す。"""
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
            graph_name=graph_name(mark),
            provenance=FIXTURE_PROVENANCE,
            assets=assets,
            part_bytes=part_bytes,
            block_bytes=block_bytes,
        )
        return [path.read_bytes() for path in result.parts]


def ir_container(
    *,
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    symbols: Sequence[str] = (),
    assets: Mapping[str, AssetInput] = {},
    part_bytes: int = DEFAULT_PART_BYTES,
    block_bytes: int = BLOCK_MAX_BYTES,
) -> list[bytes]:
    """正当なコンテナ 1 本ぶんの part バイト列（添字順 — 先頭が part 0）。

    返るのは**必ず 3 要素以上**: part 0（ヘッダ + 2 文書）・part 1（const 領域）・part 2 以降
    （重み）。合成の資産は小さいので重みは 1 part に収まり、戻り値は 3 要素になる。呼び出し
    側は `karume.container.numbered_name` で連番のファイル名を作って書き出す。

    `storage` は系列の格納形（`f32` / `f16` / `i8` / `i4`）。実物と同じく**適格な重みスロット
    だけ**が圧縮格納になり、bias やグラフ定数（i8 / i4 の scale も）は f32 のまま残るので、
    束縛表の codec 集合は実配布資産と同じ形になる。`i4` は**混成**（i4 適格な重みが i4・
    残りが i8・その他 f32）— 実物の i4 系列がこの形で、単一 codec の器では「圧縮席どうしの
    取り違え」を再現できない。

    `inputs`（名前と shape）と `outputs`（出力ごとの宣言 shape）は family 固有の門が読む席。
    入力はどのノードも消費しない宣言だけの席で、出力は小さな定数を `expand` した値なので、
    2048×2048 のような宣言でも実バイトは数十バイトのまま。

    `weights` は linear の重みになる initializer 名（層数を数える門が読む綴り）。
    `baked` は `(記号名, 焼き込み上限)` で、`sym_prefix_slice` の焼き込み定数を 1 本足す
    （記号は `inputs` の次元位置で束縛されている必要がある）。

    `assets` は容器へ同梱する資産（資産名 → {@link karume.container.AssetInput}）— PLE 索引や
    `rope_base` のように「重みではないが同じ容器で配るバイト列」の席で、weights の part の
    後ろに載る（ADR 0109 決定 4）。

    `symbols` は宣言へ**足すだけ**の記号（`inputs` / `outputs` の shape から導かれる分に加える）。
    「記号次元を持つグラフを配らない」側の門を試すための故障注入の席で、束縛点を持たない記号を
    名乗るのは実物には無い形である — 束縛点のある記号は `inputs` の shape に綴ればよい。

    `part_bytes` / `block_bytes` は寸法の差し込み（`publish_container` の席）— 合成の小さい重みを
    **複数 part** / **piece 列**へ割らせるための席で、既定は実物と同じ 256 MiB / 32 MiB。何本に
    なるかは現物のバイト数が決めるので、渡した側は本数を仮定せず**現物を観測する**こと。
    """
    graph, tensors, scales, overrides = _spec(
        mark, storage, inputs, outputs, weights, baked, symbols
    )
    return _write(
        graph,
        tensors,
        mark=mark,
        storage=storage,
        scales=scales,
        overrides=overrides,
        assets=assets,
        part_bytes=part_bytes,
        block_bytes=block_bytes,
    )


def graph_name(mark: str) -> str:
    """`mark` からコンテナのグラフ名を作る（語彙 `[A-Za-z0-9._-]` の外は `-` へ畳む）。

    `mark` はテンソルキーの接頭辞で、テストは「どのモデルがどのバイト列を主張するか」を
    そこで作り分ける — グラフ名の語彙に縛られない綴りも使えるようにしておく。
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", mark)[:64]


def fill_spec(count: int, *, mark: str) -> tuple[IrGraph, dict[str, torch.Tensor]]:
    """同じ大きさの f32 テンソルを `count` 本持つだけのグラフ（分割を踏むための素材）。"""
    initializers = {}
    values = {}
    tensors = {}
    names = [f"{_OWN}fill{index}" for index in range(count)]
    for name in names:
        initializers[name] = IrInitializer(tensor=f"{mark}.{name}", storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=[_FILL_ELEMENTS])
        tensors[f"{mark}.{name}"] = _ramp(_FILL_ELEMENTS)
    return IrGraph(initializers=initializers, values=values, outputs=names[:1]), tensors


def ir_parts(count: int, *, mark: str) -> list[bytes]:
    """`count` 本の part 列になる正当なコンテナ（添字順 — 先頭が part 0）。

    何本に割れるかは現物のバイト数が決めるので、同じ大きさのテンソルを `count - 2` 本並べ、
    part 長の差し込みをその 1 本ぶんに合わせて **1 重み part = 1 テンソル**へ割り付ける。
    `count` は part 0（2 文書）と part 1（const 領域 — const を持たないので 0 バイト）を
    含む総数なので 3 以上。
    """
    if count < 3:
        raise ValueError(f"part 数 {count} は 3 以上（part 0〈2 文書〉+ part 1〈const〉を含む）")
    graph, tensors = fill_spec(count - 2, mark=mark)
    return _write(
        graph,
        tensors,
        mark=mark,
        storage="f32",
        scales={},
        overrides={},
        part_bytes=_FILL_ELEMENTS * 4,
    )


def fixture_spec(
    mark: str = "fixture",
    storage: str = "f32",
    inputs: Sequence[tuple[str, Shape]] = (),
    outputs: Sequence[Shape] = ([1],),
    weights: Sequence[str] = ("weight",),
    baked: tuple[str, int] | None = None,
    symbols: Sequence[str] = (),
) -> tuple[IrGraph, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, str]]:
    """{@link ir_container} と**同じ素材**（グラフ・格納テンソル・scale 台帳・1 本単位指定）。

    旧配布形の合成（`legacy_writer`）が同じ素材を別の器へ書くための口 — 素材を 2 つに割ると
    「旧形から移した容器」と「直接書いた容器」のバイト同一が素材の違いで崩れる。
    """
    return _spec(mark, storage, inputs, outputs, weights, baked, symbols)


def _spec(
    mark: str,
    storage: str,
    inputs: Sequence[tuple[str, Shape]],
    outputs: Sequence[Shape],
    weights: Sequence[str],
    baked: tuple[str, int] | None,
    symbols: Sequence[str],
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
    # bias は常に f32（量子化されない）ので、`mark` 依存の値を載せる席に使える — 重みの part を
    # **mark ごとに違うバイト列**にしておかないと、共有の畳み込みを観測するテストが「どの
    # モデルでも同じ part」を見てしまう（テンソルキーの違いは part 0 の descriptor にしか出ない）。
    declare(bias, _ramp(_OUT) + float(sum(mark.encode()) % 97))
    # const 領域（part 1）を空にしない — 実物の容器は定数を持ち、part 1 が 0 バイトだと
    # 「どのモデルでもバイト同一な席」が 1 本できて共有の畳み込みの観測がぼやける。
    # 値は `mark` に依存させる（同じ長さの mark でもバイトは違う = 共有判定の被験体になる）。
    const_key = f"const.{mark}"
    const_name = f"{_OWN}const"
    initializers[const_name] = IrInitializer(tensor=const_key, storage=IrStorage(dtype="f32"))
    values[const_name] = IrValue(dtype="f32", shape=[_OUT])
    tensors[const_key] = _ramp(_OUT) + float(sum(mark.encode()) % 97)
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
        shifted = f"{_OWN}c{index}"
        values[shifted] = IrValue(dtype="f32", shape=[1, _OUT])
        nodes.append(IrNode(op="add", ins=[out, const_name], outs=[shifted], attrs={}))

    # part の**サイズ**も `mark` で変える（長さの違う mark は違うサイズの part を作る）。
    # 共有の畳み込みの前置フィルタ（サイズ違いは hash を採らずに落ちる）を観測する側の道具で、
    # const 領域（part 1）と重み（part 2 以降）の両方に 1 本ずつ置く。
    #
    # MUST: 詰め物を**グラフ出力に足さない** — family の門は「出力が何本か」で配布形の取り違えを
    # 見る（BiRefNet の multi-scale supervision・SigLIP2 の全層出し）ので、素材が出力を 2 本
    # 増やすと `outputs` の期待が素材の都合で決まってしまう。消費されない値は宣言として残る。
    for region, key, span in (
        # 刻みは 16 要素 = 64 バイト（block の先頭整列の単位）— これより細かいと、長さの
        # 違いが整列の詰め物に吸われて part のサイズが動かない。
        ("const", f"const.{mark}.pad", 16 * (1 + len(mark) % 11)),
        ("weight", f"{mark}.{_OWN}pad", 16 * (1 + len(mark) % 13)),
    ):
        name = f"{_OWN}{region}_pad"
        initializers[name] = IrInitializer(tensor=key, storage=IrStorage(dtype="f32"))
        values[name] = IrValue(dtype="f32", shape=[span])
        tensors[key] = _ramp(span)
        doubled = f"{_OWN}{region}_pad_out"
        values[doubled] = IrValue(dtype="f32", shape=[span])
        nodes.append(IrNode(op="add", ins=[name, name], outs=[doubled], attrs={}))

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
        symbols=sorted({*_symbols(declared), *symbols}),
        inputs=[IrInput(name=name, dtype="f32", shape=list(shape)) for name, shape in inputs],
        outputs=graph_outputs,
        initializers=initializers,
        values=values,
        nodes=nodes,
    )
    return graph, tensors, scales, overrides


#: 故障注入で差し込む**語彙外**の op（`karume.ops.OP_CONTRACTS` に無い綴り）。
#:
#: MUST: 置き換える `linear` と**同じバイト長**（6 文字）で、`requires.ops` / `capabilities.ops`
#: の code point 順の位置も変わらない綴りにする — part 0 の長さが動くと、落ちるのが IR の
#: 受理規則ではなく「part の長さが宣言と違う」になり、被験体が別物になる。
UNKNOWN_OP = "nosuch"

#: 置き換え元の op（合成のグラフが必ず 1 本以上持つ）。
_REPLACED_OP = "linear"


def with_an_unknown_op(parts: Sequence[bytes]) -> list[bytes]:
    """part 0 のグラフ文書の `linear` を**語彙外の op** へ書き換えた part 列を返す。

    構造検査（`capabilities.ops` が `graphs[].requires.ops` の和と一致すること・block 目次・
    codec 台帳）は満たしたままなので、落とせるのは IR の受理規則
    （{@link karume.verify.assert_ir_accepted}）だけ — 「語彙外の op を宣言した容器」が
    組み立てにも `karume verify` にも掛からずに配布形へ据わる、という穴の被験体。
    """
    part0, *rest = parts
    header = read_header(part0[:HEADER_BYTES])
    begin = HEADER_BYTES
    document = json.loads(part0[begin : begin + header.graph_length])
    tail = part0[begin + header.graph_length :]
    for graph in document["graphs"].values():
        for node in graph["nodes"]:
            if node["op"] == _REPLACED_OP:
                node["op"] = UNKNOWN_OP
        graph["requires"]["ops"] = sorted({node["op"] for node in graph["nodes"]})
    document["capabilities"]["ops"] = sorted(
        {op for graph in document["graphs"].values() for op in graph["requires"]["ops"]}
    )
    rewritten = canonical_json(document).encode("utf-8")
    if len(rewritten) != header.graph_length:
        raise AssertionError(
            f"書き換えでグラフ文書の長さが {header.graph_length} → {len(rewritten)} へ動いた"
            "（被験体が『part の長さ違い』に化ける）"
        )
    head = write_header(kind="model", graph_length=len(rewritten), model_length=header.model_length)
    return [head + rewritten + tail, *rest]
