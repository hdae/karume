"""aten op → {@link Emitted} 対応表（IR v1 の op 語彙への写像）。

convert.py のモジュール docstring が言う 3 段のうち **第 2 段（aten → IR op 対応表による
変換）** の表そのもの。グラフ走査・定数畳み込み・CSE のエンジンは convert.py 側にある。

モデルファミリを足すたびに伸びるのはこちら側だけで、エンジン側はほぼ不変 — 分割の目的は
その成長点を分けること。両側が使う {@link Emitted} と述語もここに置く（import 方向は
convert → aten_handlers の一方向のみ）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, NamedTuple

import torch

# MUST: `torch.ops.torchvision.*` は torchvision の C++ ライブラリを読み込まないと引けない
# （ハンドラのキーが書けない）。条件付き import で soft 依存にすると「deform_conv2d を
# 未対応 op として落とす env」と「通る env」が分岐するので、基本依存として素で import する
# （ADR 0055 決定 7）。
import torchvision  # noqa: F401  -- torch.ops.torchvision の登録が副作用
from torch.fx import Node

# MUST: `torch.ops.karume.*` は custom op の登録モジュールを import しないと引けない
# （ハンドラのキーが書けない）。登録はプロセス全域のグローバル副作用で、Python 側 exporter に
# 「全モジュール副作用ゼロ」の不変条件は掛からない（ADR 0056 決定 6）。
import karume.custom_ops  # noqa: F401  -- torch.ops.karume の登録が副作用
from karume.extents import extent_key
from karume.normalize import SAFE_SOFTMAX_META
from karume.ops import GRU_SCAN_OP, GRU_SCAN_REVERSE_OP, STRIDED_RANK

aten = torch.ops.aten

#: 意味論 dtype（docs/ir-v1.md「値と型」）。ここに無い torch dtype は fail loudly。
#:
#: MUST: torch 既定の整数 int64 は **i32 として宣言する**（ADR 0009 の境界正規化）。
#: WebGPU に 64bit 整数バッファが無いので、64bit の無い世界への変換点をエクスポータ
#: 1 箇所に固定する。値の側の検査は normalize_boundary_tensor が持つ。
DTYPE_NAMES = {
    torch.float32: "f32",
    torch.int32: "i32",
    torch.int64: "i32",
    torch.bool: "bool",
}


class Emitted(NamedTuple):
    """ハンドラが宣言する IR ノード仕様。ins は期待するテンソル入力数。

    attrs は op 契約表（ops.py）の attrs スキーマと 1 セットで広げる — 契約表に
    宣言の無いキーは verify で fail loudly になる（ADR 0012）。

    `synth_consts` は **省略可能なスロットを持つ融合 op を固定アリティへ正規化する**ための
    合成指示（ADR 0015 の conv ゼロ bias / ADR 0016 の linear ゼロ bias・layer_norm の
    ones/zeros affine・rms_norm の ones weight）。各要素は `(埋め値, 長さ)` で、**宣言順に
    末尾へ足す**。ハンドラ側で `_add_const` を呼ばないのは、initializer の生成が Converter の
    状態（重複排除・値名の採番）に属するため — ハンドラは純関数のまま保つ。

    合成が数学的に恒等なのは `+0` / `×1` が f32 で厳密恒等だから（ADR 0015）。カーネルにも
    契約にも arity 分岐を持ち込まないのが目的で、「実測に無い形を黙って増やす」のとは違う
    （合成した定数は通常の initializer と同じ経路・同じ命名で運ばれる）。
    """

    op: str
    ins: int
    attrs: Mapping[str, Any] = MappingProxyType({})
    synth_consts: tuple[tuple[float, int], ...] = ()


def _expect(cond: bool, node: Node, why: str) -> None:
    if not cond:
        raise NotImplementedError(f"{node.target}: {why} (args={node.args}, kwargs={node.kwargs})")


def _has_free_symbols(value: Any) -> bool:
    """SymInt / 記号 shape を持つ値か（guard 評価を誘発しない構造判定）。"""
    if isinstance(value, torch.SymInt):
        return bool(value.node.expr.free_symbols)
    if isinstance(value, torch.Tensor):
        return any(_has_free_symbols(d) for d in value.shape)
    if isinstance(value, (list, tuple)):
        return any(_has_free_symbols(item) for item in value)
    return False


# ---- aten op 対応表 ------------------------------------------------------


def _simple(op: str, ins: int):
    def handler(node: Node) -> Emitted:
        return Emitted(op, ins)

    return handler


def _h_binary(op: str):
    """binary elementwise（torch 準拠の右詰め broadcast）。"""

    def handler(node: Node) -> Emitted:
        _expect(node.kwargs.get("alpha", 1) == 1, node, "alpha != 1 は未対応")
        _expect(
            all(isinstance(arg, Node) for arg in node.args[:2]),
            node,
            "スカラ被演算子は未対応（二項 op の契約は attrs 空 — スカラを載せる欄が無い）",
        )
        return Emitted(op, 2)

    return handler


#: `aten.gelu` の `approximate` → IR op。attrs 空の契約なので近似種別は**別 op**で表す
#: （同じ op 名のまま近似を変えると、契約の外に「数値が静かに変わる」分岐ができる）。
_GELU_OPS = {"none": "gelu", "tanh": "gelu_tanh"}


def _h_gelu(node: Node) -> Emitted:
    approximate = node.kwargs.get("approximate", "none")
    _expect(approximate in _GELU_OPS, node, f"approximate={approximate!r} の gelu は未対応")
    return Emitted(_GELU_OPS[approximate], 1)


def _h_sum(node: Node) -> Emitted:
    src = node.args[0].meta["val"]
    # 欄の取得は `_arg_or_kwarg` に揃える — overload の呼び分けで `dim` が kwargs 側に来た形でも
    # 素の添字（IndexError）に落とさず、下の診断規律（ADR 0005）へ合流させる MUST。
    dims = _arg_or_kwarg(node, 1, "dim", None)
    _expect(dims is not None, node, "全次元 sum は未対応（reduce は 1 軸のみ）")
    keepdim = bool(_arg_or_kwarg(node, 2, "keepdim", False))
    dtype = _arg_or_kwarg(node, 3, "dtype", None)
    # 軸は 1 本だけ受理する（複数軸を 1 ノードへ畳むと縮約順序が IR から読めなくなる —
    # 実行側は 1 軸ずつの縮約しか持たない）。keepdim / dtype 指定は従来どおり未対応。
    _expect(len(dims) == 1, node, f"複数軸 {list(dims)} の sum は未対応（1 軸ずつ）")
    _expect(not keepdim, node, "keepdim=True の sum は未対応")
    _expect(dtype is None, node, "dtype 指定付きの sum は未対応")
    return Emitted("sum", 1, {"dim": _normalized_dims(node, src.dim(), dims)[0]})


#: `_to_copy` が受理する kwargs。dtype 以外（layout / device / memory_format）を伴う形は
#: 「dtype 変換」以上のことをしているので、cast へ落とさず fail loudly にする。
_TO_COPY_KWARGS = frozenset({"dtype"})


def _h_to_copy(node: Node) -> Emitted:
    """aten._to_copy → cast（ADR 0009）。変換先は meta の出力 dtype から取る。

    MUST: kwargs の dtype ではなく meta["val"].dtype を正本にする — i64 は境界正規化で
    i32 として宣言されるので、torch dtype をそのまま名前にすると宣言と食い違う。
    """
    extra = sorted(set(node.kwargs) - _TO_COPY_KWARGS)
    _expect(not extra, node, f"dtype 以外の kwargs {extra} を伴う _to_copy は未対応")
    val = node.meta["val"]
    to = DTYPE_NAMES.get(val.dtype)
    if to is None:
        raise NotImplementedError(
            f"{node.target}: 変換先 dtype {val.dtype} は IR v1 の意味論 dtype 語彙に無い"
        )
    return Emitted("cast", 1, {"to": to})


def _h_bitwise_not(node: Node) -> Emitted:
    """aten.bitwise_not → bitwise_not（bool の否定のみ）。

    整数の bitwise_not（~x）は bool の否定と意味が違う。契約は bool 専業なので、
    整数入力は黙って否定に読み替えず落とす。
    """
    src = node.args[0].meta["val"]
    _expect(
        src.dtype is torch.bool, node, f"dtype {src.dtype} の bitwise_not は未対応（bool のみ）"
    )
    return Emitted("bitwise_not", 1)


def _h_reshape(node: Node) -> Emitted:
    """view / squeeze(dims) / unsqueeze → reshape（ADR 0011）。

    目標形は attrs ではなく**出力の宣言 shape**（`meta["val"]` から `values{}` に出る）なので、
    引数の size リスト（`-1` や SymInt を含む）は IR に持ち込まない。ここで見るのは
    「要素順を変えないメタ操作である」ことだけ — 要素数一致はランタイムの契約検査が見る。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う形は未対応")
    return Emitted("reshape", 1)


def _normalized_dims(node: Node, rank: int, dims: Sequence[Any]) -> list[int]:
    """torch の負の軸表記を非負へ正規化する（IR の attrs は非負のみ — 契約表）。"""
    out: list[int] = []
    for dim in dims:
        _expect(isinstance(dim, int), node, f"軸 {dim!r} が整数でない")
        index = int(dim)
        _expect(-rank <= index < rank, node, f"軸 {index} が rank {rank} の範囲外")
        out.append(index % rank)
    return out


def _h_permute(node: Node) -> Emitted:
    """aten.permute → permute（attrs `dims`）。実体化コピーは strided 1 カーネル族。"""
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う permute は未対応")
    src = node.args[0].meta["val"]
    rank = src.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"rank {rank} の permute は strided カーネル（rank ≤ {STRIDED_RANK}）で実行できない",
    )
    dims = _normalized_dims(node, rank, node.args[1])
    _expect(len(dims) == rank, node, f"dims {dims} が rank {rank} と違う")
    return Emitted("permute", 1, {"dims": dims})


def _h_expand(node: Node) -> Emitted:
    """aten.expand → expand（ADR 0011）。目標形は出力の宣言 shape で、attrs は持たない。

    引数の size リストは `-1`（据え置き）や SymInt を含むが、解決済みの形は `meta["val"]` に
    出ているのでそちらを正本にする。「長さ 1 の次元しか拡張しない」の検査はランタイムの
    契約（束縛解決後の数値 shape）が持つ。
    """
    extra = sorted(set(node.kwargs) - {"implicit"})
    _expect(not extra, node, f"kwargs {extra} を伴う expand は未対応")
    _expect(not node.kwargs.get("implicit", False), node, "implicit=True の expand は未対応")
    rank = node.meta["val"].dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"rank {rank} の expand は strided カーネル（rank ≤ {STRIDED_RANK}）で実行できない",
    )
    return Emitted("expand", 1)


def _static_extent(node: Node, tensor: torch.Tensor, dim: int, op: str) -> int:
    """`slice` / `flip` の**対象軸は静的**（ADR 0014）— 記号次元なら fail loudly。

    記号軸の切り出しは `sym_prefix_slice` の担当で、`slice` は静的専業（重複させない）。
    flip は実測が全て静的軸。`cat` の連結軸だけは ADR 0046 で緩めた（`_cat_axis`）。
    """
    extent = tensor.shape[dim]
    _expect(
        not _has_free_symbols(extent),
        node,
        f"{op} の軸 {dim} が記号次元（対象軸は静的でなければならない）",
    )
    return int(extent)


def _static_size_or_fail(node: Node, dim: Any, what: str) -> int:
    """合成定数の長さに使う次元は静的でなければならない（記号なら fail loudly）。

    記号長のスロットを合成しようとすると「実行時に長さが決まる initializer」という、
    IR v1 に存在しない概念が要る。実測では重みの次元なので必ず静的だが、崩れたら止める。
    """
    _expect(not _has_free_symbols(dim), node, f"{what} が記号次元（合成する定数の長さが不定）")
    return int(dim)


def _h_slice(node: Node) -> Emitted:
    """aten.slice.Tensor → slice（attrs `dim` / `start` / `end`、**静的軸・静的範囲**）。

    MUST: `step != 1` は落とす（IR の slice は連続した窓のみ — 飛ばし読みは strided 族の
    可変点 1 語では表せない）。
    MUST: 負の添字と省略（`None` / int64 最大）は**ここで軸長に詰める**（IR の attrs は
    非負のみ — permute の dims と同じ境界正規化）。
    MUST: 記号軸の slice は落とす（sym_prefix_slice の担当 — 重複させない）。
    """
    extra = sorted(set(node.kwargs) - {"dim", "start", "end", "step"})
    _expect(not extra, node, f"kwargs {extra} を伴う slice は未対応")
    src = node.args[0].meta["val"]
    rank = src.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"rank {rank} の slice は strided カーネル（rank ≤ {STRIDED_RANK}）で実行できない",
    )
    dim = _normalized_dims(node, rank, [_arg_or_kwarg(node, 1, "dim", 0)])[0]
    size = _static_extent(node, src, dim, "slice")
    step = _arg_or_kwarg(node, 4, "step", 1)
    _expect(step == 1, node, f"step={step!r} の slice は未対応（連続した窓のみ）")
    bounds: list[int] = []
    for index, (key, default) in enumerate((("start", 0), ("end", size))):
        value = _arg_or_kwarg(node, 2 + index, key, None)
        if value is None:
            bounds.append(default)
            continue
        _expect(
            not _has_free_symbols(value) and isinstance(value, (int, torch.SymInt)),
            node,
            f"{key}={value!r} の slice は未対応（静的な添字のみ）",
        )
        # 負の添字は末尾からの数え上げ、範囲外は torch と同じく軸長へ詰める。
        bound = int(value)
        bounds.append(min(max(bound + size if bound < 0 else bound, 0), size))
    start, end = bounds
    # torch は start > end を空スライスに潰すが、IR は表現を持たない（黙って 0 長にしない）。
    _expect(start <= end, node, f"start={start} > end={end} の slice は未対応")
    return Emitted("slice", 1, {"dim": dim, "start": start, "end": end})


def _cat_axis(node: Node, tensors: Sequence[Node], dim: int) -> None:
    """cat の連結軸は〈定数〉または〈**同一**シンボルの一次式〉に限る（ADR 0046）。

    総和 `Σ(coeff_i·sym + offset_i)` は同一シンボルなら正準文法 `coeff·sym+offset` に
    そのまま載る（`S`+1519 → `S+1519`、`S`+`S` → `2S`）。
    MUST: 異なるシンボルが混ざる連結は落とす — 和が 1 次元 1 シンボルの文法に載らない。
    torch.export も複数シンボル和の宣言を拒むので、受理を広げても表現不能な形は入らないが、
    ここで止めないと出力 shape を書く `_sym_parts` まで発覚が遅れ、原因が読めない。
    """
    symbols: dict[str, int] = {}
    for index, item in enumerate(tensors):
        extent = item.meta["val"].shape[dim]
        if not _has_free_symbols(extent):
            continue
        for symbol in extent.node.expr.free_symbols:
            symbols.setdefault(symbol.name, index)
    _expect(
        len(symbols) <= 1,
        node,
        f"連結軸 {dim} に異なるシンボルが混ざる cat は未対応"
        f"（入力ごとのシンボル: {symbols}）— 和が次元言語 coeff·sym+offset に載らない",
    )


def _h_cat(node: Node) -> Emitted:
    """aten.cat.default → cat（attrs `dim`、**可変アリティ**）。

    MUST: 入力 1 本の cat は落とす（恒等コピーで、契約のアリティ下限 2 を割る）。
    MUST: 連結軸のシンボルは 1 種類まで（ADR 0046 — `_cat_axis`）。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う cat は未対応")
    tensors = node.args[0]
    _expect(
        isinstance(tensors, (list, tuple)) and all(isinstance(item, Node) for item in tensors),
        node,
        "第 1 引数がテンソルのリストでない cat は未対応",
    )
    _expect(len(tensors) >= 2, node, f"入力 {len(tensors)} 本の cat は未対応（2 本以上）")
    first = tensors[0].meta["val"]
    rank = first.dim()
    _expect(
        rank <= STRIDED_RANK,
        node,
        f"rank {rank} の cat は strided カーネル（rank ≤ {STRIDED_RANK}）で実行できない",
    )
    dim = _normalized_dims(node, rank, [_arg_or_kwarg(node, 1, "dim", 0)])[0]
    _cat_axis(node, tensors, dim)
    return Emitted("cat", len(tensors), {"dim": dim})


def _h_constant_pad_nd(node: Node) -> Emitted:
    """aten.constant_pad_nd → pad（attrs `left` / `right`、**最終次元・定数 0 のみ**）。

    MUST: 埋め値 0 以外と、最終次元より広い pad は落とす（契約に欄が無い = 表現が無い）。
    MUST: 負幅（切り詰め）も落とす — pad ではなく slice の意味で、通すと同じ形を 2 つの op で
    書けるうえ出力長の計算が負になる。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う pad は未対応")
    pad = node.args[1]
    _expect(
        isinstance(pad, (list, tuple)) and len(pad) == 2,
        node,
        f"pad={pad!r} は未対応（最終次元の [left, right] 1 組のみ）",
    )
    _expect(not _has_free_symbols(pad), node, f"記号を含む pad={pad!r} は未対応")
    widths = [int(width) for width in pad]
    _expect(all(width >= 0 for width in widths), node, f"負の pad={pad!r} は未対応（切り詰め）")
    value = _arg_or_kwarg(node, 2, "value", 0)
    _expect(
        isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == 0.0,
        node,
        f"value={value!r} の constant_pad_nd は未対応（定数 0 のみ）",
    )
    return Emitted("pad", 1, {"left": widths[0], "right": widths[1]})


def _h_flip(node: Node) -> Emitted:
    """aten.flip → flip（attrs `dim`、**静的軸 1 本**）。

    MUST: 多軸の flip は落とす（IR の flip は 1 軸専業 — 実測は flow / sdp とも 1 軸）。
    多軸を黙って 1 軸に潰すと反転しない軸が出る。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う flip は未対応")
    src = node.args[0].meta["val"]
    dims = node.args[1]
    _expect(
        isinstance(dims, (list, tuple)) and len(dims) == 1,
        node,
        f"dims={dims!r} の flip は未対応（軸 1 本のみ）",
    )
    dim = _normalized_dims(node, src.dim(), dims)[0]
    _static_extent(node, src, dim, "flip")
    return Emitted("flip", 1, {"dim": dim})


def _h_bmm(node: Node) -> Emitted:
    """aten.bmm → bmm（rank-3 バッチ matmul）。

    MUST: rank-3 以外は落とす。ランタイムの bmm は rank-3 専業で、rank-2 は matmul の担当
    （契約表）。ここで通すと「export は緑、ブラウザだけ落ちる」になる。
    """
    for position, arg in enumerate(node.args[:2]):
        rank = arg.meta["val"].dim()
        _expect(rank == 3, node, f"入力 {position} の rank {rank} の bmm は未対応（rank-3 のみ）")
    return Emitted("bmm", 2)


def _h_gather(node: Node) -> Emitted:
    """aten.gather → gather（**最終次元固定**、attrs 無し）。

    MUST: dim が最終次元以外の形は落とす（契約が最終次元固定 — ADR 0012 の softmax と
    同じ絞り方で、一般 dim は要求実測が出てから広げる）。
    MUST: `sparse_grad=True` は落とす（既定の False だけ受理する）。勾配の疎表現は forward の
    意味を変えないが、受理範囲を「見た目 forward に効かないから」で広げると、次に効く
    引数が来たときに同じ理由で素通りする。
    MUST: **先行次元は src と index で完全一致**を要求する（IR 契約 — packages/runtime/src/ops.ts の
    computeOutputShape と同義）。torch は `index.size(d) <= src.size(d)` を許すが、ランタイムの
    カーネルは「先行次元が一致 = 行 index が共通」を前提に `row = i / J` で読む。rank だけ
    見て通すと、export は緑でブラウザだけ contract 違反で落ちる（あるいは行が黙ってずれる）。
    """
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(dim == rank - 1, node, f"最終次元以外（dim={dim} / rank={rank}）の gather は未対応")
    extra = sorted(set(node.kwargs) - {"sparse_grad"})
    _expect(not extra, node, f"kwargs {extra} を伴う gather は未対応")
    sparse_grad = node.args[3] if len(node.args) > 3 else node.kwargs.get("sparse_grad", False)
    _expect(not sparse_grad, node, "sparse_grad=True の gather は未対応")
    _expect(len(node.args) <= 4, node, f"引数 {len(node.args)} 本の gather は未対応")
    index = node.args[2].meta["val"]
    _expect(
        index.dim() == rank,
        node,
        f"index の rank {index.dim()} が src の rank {rank} と違う gather は未対応",
    )
    for axis in range(rank - 1):
        _expect(
            extent_key(src.shape[axis]) == extent_key(index.shape[axis]),
            node,
            f"先行次元 {axis} が src {list(src.shape)} と index {list(index.shape)} で一致しない"
            " gather は未対応（IR 契約は完全一致 — torch の index <= src より狭い）",
        )
    return Emitted("gather", 2)


def _h_where(node: Node) -> Emitted:
    """aten.where.self → where（三項 elementwise）。

    MUST: 3 引数すべてがテンソルである形だけ受理する。torch はスカラ引数
    （`where(cond, x, 0.0)`）も許すが、契約は attrs 空のアリティ 3 固定なのでスカラを
    載せる欄が無い（二項 op の `_h_binary` と同じ絞り方 — 必要なら normalize の
    スカラ昇格パスで潰す）。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う where は未対応")
    _expect(
        len(node.args) == 3 and all(isinstance(arg, Node) for arg in node.args),
        node,
        "スカラ被演算子を含む where は未対応（契約は attrs 空のアリティ 3 固定）",
    )
    return Emitted("where", 3)


def _h_clamp(node: Node) -> Emitted:
    """aten.clamp → clamp（両側）/ **clamp_min**（下側だけ — ADR 0017）。

    MUST: 欠けた側を ±有限最大値で補わない。「表現が無い形を黙って別の形で実行する」ことに
    なるので、下限だけの形は**別 op**（`clamp_min`）へ落とす（ADR 0017 で語彙に入った）。
    上限だけの形は語彙に無いので落とす — 需要が出たら `clamp_max` を同じ手筋で足す。
    MUST: `min <= max` は契約表（shapes 層）が見る — WGSL の clamp は逆転で未定義。
    """
    extra = sorted(set(node.kwargs) - {"min", "max"})
    _expect(not extra, node, f"kwargs {extra} を伴う clamp は未対応")
    bounds = [_arg_or_kwarg(node, 1, "min", None), _arg_or_kwarg(node, 2, "max", None)]
    for name, value in zip(("min", "max"), bounds, strict=True):
        if value is None:
            continue
        _expect(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            node,
            f"{name}={value!r} の clamp は未対応（境界は有限スカラ）",
        )
        _expect(math.isfinite(float(value)), node, f"非有限の {name}={value!r} は IR v1 非対応")
    raw_min, raw_max = bounds
    _expect(raw_min is not None, node, "上限だけの clamp は未対応（語彙に clamp_max が無い）")
    if raw_max is None:
        return Emitted("clamp_min", 1, {"min": float(raw_min)})
    minimum, maximum = float(raw_min), float(raw_max)
    _expect(minimum <= maximum, node, f"min={minimum} > max={maximum} の clamp は未対応")
    return Emitted("clamp", 1, {"min": minimum, "max": maximum})


def _h_leaky_relu(node: Node) -> Emitted:
    """aten.leaky_relu → leaky_relu（attrs `negative_slope`）。

    MUST: torch 側の既定（0.01）を**ここで読み取って attrs に載せる**。dec は 0.1（ups /
    ResBlock）と 0.01（最終段・位置引数ごと省略）が混在するので、IR に載せずランタイム側の
    既定に頼ると片方が黙って誤る（ADR 0015）。「既定値補完をしない」規律は IR 契約の側の話で、
    torch の意味論を境界で明示化するのはその逆側の責務。
    """
    extra = sorted(set(node.kwargs) - {"negative_slope"})
    _expect(not extra, node, f"kwargs {extra} を伴う leaky_relu は未対応")
    slope = _arg_or_kwarg(node, 1, "negative_slope", 0.01)
    _expect(
        isinstance(slope, (int, float)) and not isinstance(slope, bool),
        node,
        f"negative_slope={slope!r} がスカラでない",
    )
    number = float(slope)
    _expect(math.isfinite(number), node, f"非有限の negative_slope={slope!r} は IR v1 非対応")
    return Emitted("leaky_relu", 1, {"negative_slope": number})


def _h_compare_scalar(op: str):
    """aten.ge/le/gt.Scalar → 比較 op（attrs `value`、出力は bool）。"""

    def handler(node: Node) -> Emitted:
        _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う {op} は未対応")
        _expect(len(node.args) == 2, node, f"引数 {len(node.args)} 本の {op} は未対応")
        value = node.args[1]
        _expect(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            node,
            f"value={value!r} がスカラでない（Tensor 形は別 op）",
        )
        number = float(value)
        _expect(math.isfinite(number), node, f"非有限の value={value!r} は IR v1 非対応")
        src = node.args[0].meta["val"]
        _expect(
            src.dtype is torch.float32,
            node,
            f"dtype {src.dtype} の {op} は未対応（f32 のみ）",
        )
        return Emitted(op, 1, {"value": number})

    return handler


def _h_ge_tensor(node: Node) -> Emitted:
    """aten.ge.Tensor → ge（f32 × f32 → bool の右詰め broadcast）。"""
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う ge は未対応")
    _expect(
        all(isinstance(arg, Node) for arg in node.args[:2]),
        node,
        "スカラ被演算子は未対応（スカラ比較は ge_scalar）",
    )
    return Emitted("ge", 2)


def _h_bitwise_and(node: Node) -> Emitted:
    """aten.bitwise_and.Tensor → bitwise_and（**bool の論理積のみ**）。

    整数の bitwise_and は bool の論理積と意味が違う（契約は bool 専業 — bitwise_not と
    同じ絞り方）。整数入力は黙って論理積に読み替えず落とす。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う bitwise_and は未対応")
    for position, arg in enumerate(node.args[:2]):
        _expect(isinstance(arg, Node), node, f"入力 {position} がスカラの bitwise_and は未対応")
        _expect(
            arg.meta["val"].dtype is torch.bool,
            node,
            f"dtype {arg.meta['val'].dtype} の bitwise_and は未対応（bool のみ）",
        )
    return Emitted("bitwise_and", 2)


def _h_cumsum(node: Node) -> Emitted:
    """aten.cumsum → cumsum（attrs `dim`、**最終次元のみ**）。

    MUST: 最終次元以外は落とす（softmax / gather と同じ絞り方）。ランタイムは 1 invocation =
    1 行の逐次走査で、累積軸が連続であることを前提にしている。
    MUST: 負の軸表記は境界で非負へ正規化する（IR の attrs は非負のみ — 契約表）。
    """
    extra = sorted(set(node.kwargs) - {"dtype"})
    _expect(not extra, node, f"kwargs {extra} を伴う cumsum は未対応")
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(dim == rank - 1, node, f"最終次元以外（dim={dim} / rank={rank}）の cumsum は未対応")
    dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
    _expect(dtype is None, node, "dtype 指定付きの cumsum は未対応")
    return Emitted("cumsum", 1, {"dim": dim})


def _h_row_reduce(op: str):
    """aten.amax / aten.amin（1 軸・keepdim 無し）。"""

    def handler(node: Node) -> Emitted:
        src = node.args[0].meta["val"]
        # `_h_sum` と同じく `_arg_or_kwarg` で引く — 軸が省略された形を `[]` に落とすと
        # 「複数軸 [] は未対応」という**事実と食い違う診断**になるので、全次元縮約として弾く。
        dims = _arg_or_kwarg(node, 1, "dim", None)
        _expect(dims is not None, node, f"全次元 {op} は未対応（reduce は 1 軸のみ）")
        keepdim = bool(_arg_or_kwarg(node, 2, "keepdim", False))
        # 軸は 1 本だけ（sum と同じ絞り方 — 実行側は 1 軸ずつの縮約しか持たない）。
        _expect(len(dims) == 1, node, f"複数軸 {list(dims)} の {op} は未対応（1 軸ずつ）")
        _expect(not keepdim, node, f"keepdim=True の {op} は未対応")
        return Emitted(op, 1, {"dim": _normalized_dims(node, src.dim(), dims)[0]})

    return handler


# ---- 融合 op（ADR 0012 / ADR 0007 の保存リスト） --------------------------


def _arg_or_kwarg(node: Node, index: int, key: str, default: Any) -> Any:
    """位置引数と kwargs のどちらで来ても同じ欄を引く（overload の呼び分けを吸収する）。"""
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(key, default)


def _h_linear(node: Node) -> Emitted:
    """aten.linear → linear（`x[…,in] × W[out,in] + b[out]`）。

    MUST: bias 無しは**落とさずゼロ bias を合成**してアリティ 3 へ正規化する（ADR 0016 —
    conv と同じ手筋）。Anima の実測は linear 711 本中 698 本が bias 無しで、そこに arity
    分岐を作ると契約・カーネル・shape 層の全段へ波及する。合成は `+0` の厳密恒等。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う linear は未対応")
    bias = node.args[2] if len(node.args) > 2 else None
    if isinstance(bias, Node):
        return Emitted("linear", 3)
    _expect(bias is None, node, f"linear の bias={bias!r} がテンソルでも None でもない")
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 2, node, f"rank {weight.dim()} の linear 重みは未対応（[out,in]）")
    features_out = _static_size_or_fail(node, weight.shape[0], "linear の出力次元")
    return Emitted("linear", 3, synth_consts=((0.0, features_out),))


def _h_layer_norm(node: Node) -> Emitted:
    """aten.layer_norm → layer_norm（attrs `normalized_shape` / `eps`）。

    MUST: 保存しないと native_layer_norm（3 出力）+ getitem になり、IR v1 の単一出力前提と
    衝突する（recon §5）。
    MUST: 正規化軸は最終次元 1 本のみ。affine は「両方あり」「両方なし」「weight だけ」の
    3 形を受け、足りないスロットは ones/zeros 合成で埋める（ADR 0016）。他の形は行カーネルの
    前提が崩れる。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う layer_norm は未対応")
    src = node.args[0].meta["val"]
    normalized_shape = node.args[1]
    _expect(
        isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) == 1,
        node,
        f"normalized_shape {normalized_shape!r} は未対応（最終次元 1 本のみ）",
    )
    extent = src.shape[-1]
    _expect(
        not _has_free_symbols(extent),
        node,
        "正規化軸が記号次元の layer_norm は未対応（attrs は静的な長さのみ）",
    )
    dims = [int(normalized_shape[0])]
    _expect(
        dims[0] == int(extent),
        node,
        f"normalized_shape {dims} が入力の最終次元 {int(extent)} と違う",
    )
    weight = node.args[2] if len(node.args) > 2 else None
    bias = node.args[3] if len(node.args) > 3 else None
    eps = float(node.args[4]) if len(node.args) > 4 else 1e-5
    _expect(math.isfinite(eps) and eps > 0, node, f"eps={eps} は未対応（有限の正数のみ）")
    attrs = {"normalized_shape": dims, "eps": eps}
    if isinstance(weight, Node) and isinstance(bias, Node):
        return Emitted("layer_norm", 3, attrs)
    if isinstance(weight, Node):
        # ADR 0016: bias だけ zeros 合成でアリティ 3 へ正規化する（`+0` の厳密恒等）。合成は
        # 末尾へ足すので weight は本物のスロットに残る（ModernBERT の `norm_bias=false`）。
        return Emitted("layer_norm", 3, attrs, synth_consts=((0.0, dims[0]),))
    # MUST: **bias だけ**の形は落とす。合成は「末尾へ順に足す」ので、weight が無く bias だけ
    # ある形を通すと bias が weight のスロットへ滑り込む（要素数は合うので shape 検査も
    # 素通りする沈黙誤値）。逆向き（weight だけ）にこの危険は無い。
    _expect(
        bias is None,
        node,
        f"weight 無しで bias だけ（bias={bias!r}）の layer_norm は未対応",
    )
    # ADR 0016: ones/zeros 合成でアリティ 3 へ正規化する（`×1 +0` の厳密恒等）。
    return Emitted("layer_norm", 3, attrs, synth_consts=((1.0, dims[0]), (0.0, dims[0])))


def _h_rms_norm(node: Node) -> Emitted:
    """aten.rms_norm → rms_norm（attrs `eps`、**アリティ 2** — ADR 0017）。

    供給ルートは 2 系統（ADR 0017）: diffusers `nn.RMSNorm` 由来の `aten.rms_norm` を
    PRESERVED で残した形と、手書き分解形を `normalize._fold_rms_norm` が畳んだ形。畳んだ側は
    常に weight 付き・eps 明示で来るので、weight 無し / eps 無しの分岐は前者だけが踏む。

    MUST: 正規化軸は最終次元 1 本のみ・weight は最終次元長の rank1（契約 — 正規化長の正本は
    weight の長さ）。`normalized_shape` は IR に載せない（二重管理にしない）。
    MUST: eps 省略時は **torch の既定を境界で明示化**する（`finfo(f32).eps`）。leaky_relu の
    negative_slope と同じ規律 — IR 側の「既定値補完をしない」はランタイム契約の話で、torch の
    意味論をエクスポータが明示するのはその逆側の責務。
    """
    extra = sorted(set(node.kwargs) - {"weight", "eps"})
    _expect(not extra, node, f"kwargs {extra} を伴う rms_norm は未対応")
    src = node.args[0].meta["val"]
    normalized_shape = node.args[1]
    _expect(
        isinstance(normalized_shape, (list, tuple)) and len(normalized_shape) == 1,
        node,
        f"normalized_shape {normalized_shape!r} は未対応（最終次元 1 本のみ）",
    )
    extent = _static_size_or_fail(node, src.shape[-1], "rms_norm の正規化軸")
    _expect(
        int(normalized_shape[0]) == extent,
        node,
        f"normalized_shape {list(normalized_shape)} が入力の最終次元 {extent} と違う",
    )
    raw_eps = _arg_or_kwarg(node, 3, "eps", None)
    if raw_eps is None:
        eps = float(torch.finfo(torch.float32).eps)
    else:
        _expect(
            isinstance(raw_eps, (int, float)) and not isinstance(raw_eps, bool),
            node,
            f"eps={raw_eps!r} がスカラでない",
        )
        eps = float(raw_eps)
    _expect(math.isfinite(eps) and eps > 0, node, f"eps={eps} は未対応（有限の正数のみ）")
    weight = _arg_or_kwarg(node, 2, "weight", None)
    if isinstance(weight, Node):
        weight_val = weight.meta["val"]
        _expect(
            list(weight_val.shape) == [extent],
            node,
            f"weight の shape {list(weight_val.shape)} が [{extent}] でない rms_norm は未対応",
        )
        return Emitted("rms_norm", 2, {"eps": eps})
    _expect(weight is None, node, f"rms_norm の weight={weight!r} がテンソルでも None でもない")
    # ADR 0017: weight 無し形は ones 合成でアリティ 2 へ正規化する（`×1` の厳密恒等）。
    return Emitted("rms_norm", 2, {"eps": eps}, synth_consts=((1.0, extent),))


def _h_softmax(node: Node) -> Emitted:
    """aten.softmax → softmax / safe_softmax（attrs `dim`、**最終次元のみ**）。

    MUST: 最終次元以外は落とす（gather と同じ絞り方 — 一般 dim は要求実測が出てから）。
    ランタイムは safe-softmax の行カーネル 1 本で、縮約軸が連続であることを前提にしている。

    `normalize._drop_safe_softmax_guard` が SDPA の safe-softmax ガードを実値証明で落とせず
    構成的に置換した softmax には {@link SAFE_SOFTMAX_META} の旗が立つ（ADR 0044）。旗を
    ノードの meta で運ぶのは、FX グラフ層に "safe_softmax" という aten op が存在しないため。
    """
    extra = sorted(set(node.kwargs) - {"dtype"})
    _expect(not extra, node, f"kwargs {extra} を伴う softmax は未対応")
    src = node.args[0].meta["val"]
    rank = src.dim()
    dim = _normalized_dims(node, rank, [node.args[1]])[0]
    _expect(dim == rank - 1, node, f"最終次元以外（dim={dim} / rank={rank}）の softmax は未対応")
    dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
    _expect(dtype is None, node, "dtype 指定付きの softmax は未対応")
    op = "safe_softmax" if node.meta.get(SAFE_SOFTMAX_META) else "softmax"
    return Emitted(op, 1, {"dim": dim})


def _h_attention(node: Node) -> Emitted:
    """aten.scaled_dot_product_attention → attention（attrs `scale` — ADR 0023）。

    引数形は `(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
    enable_gqa=False)`。**受理するのは「非因果・dropout 0・rank-4」で、マスクは無しか f32
    加算型 `[1,1,M,N]` だけ**（GQA は ADR 0067 決定 1 で整除 broadcast として受理 — 下記）。
    残りは全件列挙して fail loudly にする
    （黙って近似しない — 横断の不変条件）。SDPA 保存は依然として**ターゲット別**に
    有効化する（`curated_decompositions(preserved=…)` — ADR 0016 のガード除去パスを
    温存するため）。

    MUST: mask は **f32 の加算型**（`S' = S + mask`）で shape はちょうど `[1,1,M,N]`。
    bool マスクはここでは受理しない — 変換は `normalize._additive_attn_mask` が torch 自身の
    分解（`where(mask, 0, -inf)`）と同じ op・同じ定数で済ませる。門をハンドラ側にも残すのは、
    正規化パスが（kwargs 渡し等で）発火しなかった形が黙って通らないようにするため。

    MUST: attrs の `scale` は **`f32(√scale_factor)`（半スケール）**。torch の math decomp が
    `q *= math.sqrt(scale_factor); k *= math.sqrt(scale_factor)` と書く形と**同じ定数**で、
    ここが 1 ulp でもずれると分解経路とのビット同一（ADR 0023 の設計の核）が崩れる。
    実測オラクル: D=128・scale 省略なら `0.2973017692565918`（= f32(128^-0.25)）。
    MUST: f32 へ丸めてから載せる。IR の JSON リテラルが torch decomp の `mul` 定数
    （エクスポータが f32 の initializer として焼く値）と**同じ 10 進表記**になる。
    """
    extra = sorted(
        set(node.kwargs) - {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
    )
    _expect(not extra, node, f"kwargs {extra} を伴う scaled_dot_product_attention は未対応")
    attn_mask = _arg_or_kwarg(node, 3, "attn_mask", None)
    _expect(
        attn_mask is None or isinstance(attn_mask, Node),
        node,
        f"attn_mask={attn_mask!r} がテンソル値でない attention は未対応",
    )
    dropout_p = _arg_or_kwarg(node, 4, "dropout_p", 0.0)
    _expect(
        isinstance(dropout_p, (int, float)) and not isinstance(dropout_p, bool),
        node,
        f"dropout_p={dropout_p!r} がスカラでない",
    )
    _expect(
        float(dropout_p) == 0.0, node, f"dropout_p={dropout_p} の attention は未対応（推論のみ）"
    )
    is_causal = _arg_or_kwarg(node, 5, "is_causal", False)
    _expect(
        is_causal is False, node, "is_causal=True の attention は未対応（因果マスクの欄が無い）"
    )
    # `enable_gqa` は**読まない**（ADR 0067 決定 1 — H 突合が整除 broadcast へ緩んだので、
    # True / False のどちらでも同じ保存ノードを発行する）。形の妥当性（`H % Hkv == 0` かつ
    # `H ≥ Hkv`・k / v 間の Hkv 一致）を見るのは shapes 層の 1 箇所で、convert の出口の
    # `assert_graph_shapes` が全ノードを必ず通す — ここで二重に検査すると受理集合が 2 箇所へ
    # 分かれる。torch 自身も非整除の enable_gqa を RuntimeError で落とす。
    shapes = []
    for index, name in enumerate(("q", "k", "v")):
        value = node.args[index].meta["val"]
        _expect(
            value.dim() == 4,
            node,
            f"{name} の rank {value.dim()} の attention は未対応（rank-4 head-first のみ）",
        )
        shapes.append(value.shape)
    depth = _static_size_or_fail(node, shapes[0][3], "attention の D（q の軸 3）")
    for index, name in enumerate(("k", "v"), start=1):
        other = _static_size_or_fail(node, shapes[index][3], f"attention の D（{name} の軸 3）")
        _expect(
            other == depth,
            node,
            f"{name} の D {other} が q の D {depth} と違う attention は未対応",
        )
    raw_scale = _arg_or_kwarg(node, 6, "scale", None)
    if raw_scale is None:
        # torch の既定（`_scaled_dot_product_attention_math`）と同じ式。
        scale_factor = 1.0 / math.sqrt(depth)
    else:
        _expect(
            isinstance(raw_scale, (int, float)) and not isinstance(raw_scale, bool),
            node,
            f"scale={raw_scale!r} がスカラでない",
        )
        scale_factor = float(raw_scale)
    _expect(
        math.isfinite(scale_factor) and scale_factor > 0,
        node,
        f"scale={scale_factor} の attention は未対応（有限の正数のみ — √ を取る）",
    )
    half = float(torch.tensor(math.sqrt(scale_factor), dtype=torch.float32).item())
    if attn_mask is None:
        return Emitted("attention", 3, {"scale": half})
    mask = attn_mask.meta["val"]
    _expect(
        mask.dtype is torch.float32,
        node,
        f"dtype {mask.dtype} の attn_mask は未対応（加算型 f32 のみ —"
        " bool は normalize._additive_attn_mask が where(mask, 0, -inf) へ落とす）",
    )
    _expect(
        mask.dim() == 4,
        node,
        f"rank {mask.dim()} の attn_mask は未対応（[1,1,M,N] のみ）",
    )
    _expect(
        extent_key(mask.shape[0]) == 1 and extent_key(mask.shape[1]) == 1,
        node,
        f"attn_mask の先頭 2 軸 {list(mask.shape[:2])} が 1 でない形は未対応"
        "（B·H への broadcast 専業 — [B,1,M,N] / [1,H,M,N] は契約に無い）",
    )
    _expect(
        extent_key(mask.shape[2]) == extent_key(shapes[0][2])
        and extent_key(mask.shape[3]) == extent_key(shapes[1][2]),
        node,
        f"attn_mask の M / N {list(mask.shape[2:])} が q / k の"
        f" {[shapes[0][2], shapes[1][2]]} と違う attention は未対応",
    )
    return Emitted("attention", 4, {"scale": half})


def _h_embedding(node: Node) -> Emitted:
    """aten.embedding → embedding（weight f32[V,H] × index i32[…] → f32[…,H]）。

    MUST: `padding_idx` は attrs に載せるが **forward には効かない**（勾配で padding 行を
    更新しないための欄。順伝播は素の行 gather と同じ）。無視するために契約から落とすと
    「未知 attr は fail loudly」の規律に穴が開くので、受理して運ぶ（ADR 0012）。
    MUST: `scale_grad_by_freq` / `sparse` は既定（False）のみ受理する。同じく勾配側の欄だが、
    「見た目 forward に効かないから」で受理範囲を広げると、次に効く引数が来たときに同じ
    理由で素通りする（gather の sparse_grad と同じ規律）。
    MUST: 添字は **rank ≥ 1**（IR 契約 — packages/runtime/src/ops.ts）。torch はスカラ添字を
    許して `[H]` を返すが、ランタイムの契約は `[…index, H]` で rank 0 の形を持たない。
    """
    extra = sorted(set(node.kwargs) - {"padding_idx", "scale_grad_by_freq", "sparse"})
    _expect(not extra, node, f"kwargs {extra} を伴う embedding は未対応")
    index = node.args[1].meta["val"]
    _expect(
        index.dim() >= 1,
        node,
        "スカラ添字（rank 0）の embedding は未対応（契約の出力は […index, H]）",
    )
    padding_idx = _arg_or_kwarg(node, 2, "padding_idx", -1)
    _expect(
        isinstance(padding_idx, int) and not isinstance(padding_idx, bool) and padding_idx >= -1,
        node,
        f"padding_idx={padding_idx!r} は未対応（-1 以上の整数のみ）",
    )
    _expect(
        not _arg_or_kwarg(node, 3, "scale_grad_by_freq", False),
        node,
        "scale_grad_by_freq=True の embedding は未対応",
    )
    _expect(not _arg_or_kwarg(node, 4, "sparse", False), node, "sparse=True の embedding は未対応")
    return Emitted("embedding", 2, {"padding_idx": int(padding_idx)})


def _h_masked_fill(node: Node) -> Emitted:
    """aten.masked_fill.Scalar → masked_fill（attrs `value`）。

    MUST: 非有限の埋め値は落とす（IR v1 は非有限値を JSON リテラルでも値レベルでも拒否）。
    attention の `-inf` 埋めは実在する書き方なので、この門は実際に踏まれる。実測値
    -3.4028234663852886e+38（f32 の最小有限値）はこの門を通る。

    NOTE: mask が bool であることは torch 自身が forward で要求する（整数マスクは
    `expected predicate to be bool` で落ちる）ので、ここでは見ない。IR 側の担保は契約表の
    スロット別 dtype（x=f32 / mask=bool）。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う masked_fill は未対応")
    value = node.args[2]
    _expect(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        node,
        f"value={value!r} がスカラでない（masked_fill.Tensor 形は未対応）",
    )
    number = float(value)
    _expect(math.isfinite(number), node, f"非有限の value={value!r} は IR v1 非対応")
    return Emitted("masked_fill", 2, {"value": number})


def _single_spatial(node: Node, value: Any, what: str) -> int:
    """conv 族の空間パラメータ（`1` / `[1]` のどちらの表記でも 1 本の整数へ正規化する）。"""
    if isinstance(value, (list, tuple)):
        _expect(len(value) == 1, node, f"{what}={value!r} は未対応（conv1d の空間軸は 1 本）")
        value = value[0]
    _expect(
        isinstance(value, int) and not isinstance(value, bool),
        node,
        f"{what}={value!r} が整数でない",
    )
    return int(value)


def _conv_bias(node: Node, channels_out: int, what: str) -> tuple[tuple[float, int], ...]:
    """bias スロットを見て「合成するゼロ bias」を返す（bias があれば空）。

    MUST: bias 無しは**落とさずゼロ bias を合成**してアリティ 3 へ正規化する（ADR 0015）。
    実測（dec の conv_post / Anima VAE の conv 群）に bias 無しがあり、カーネル・契約に
    arity 分岐を持ち込むより合成のほうが面が狭い。
    """
    bias = _arg_or_kwarg(node, 2, "bias", None)
    if isinstance(bias, Node):
        return ()
    _expect(bias is None, node, f"{what} の bias={bias!r} がテンソルでも None でもない")
    return ((0.0, channels_out),)


def _h_conv1d(node: Node) -> Emitted:
    """aten.conv1d → conv1d（attrs `stride` / `padding` / `dilation` / `groups`）。

    MUST: 4 つとも attrs に**明示**する（既定値補完に頼らない — ADR 0012 / 0015）。契約表側も
    宣言必須なので、どちらか片方でも省略すると verify が落ちる。
    MUST: `groups` は Cin / Cout を割り切ること・重みは `[Cout, Cin/groups, K]` であること
    を shape 層（shapes.py）が見る。ここで見るのは値域だけ。
    """
    extra = sorted(set(node.kwargs) - {"bias", "stride", "padding", "dilation", "groups"})
    _expect(not extra, node, f"kwargs {extra} を伴う conv1d は未対応")
    src = node.args[0].meta["val"]
    _expect(src.dim() == 3, node, f"rank {src.dim()} の conv1d は未対応（[B,Cin,L] のみ）")
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 3, node, f"rank {weight.dim()} の conv1d 重みは未対応")
    stride = _single_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _single_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    dilation = _single_spatial(node, _arg_or_kwarg(node, 5, "dilation", 1), "dilation")
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool),
        node,
        f"groups={groups!r} が整数でない",
    )
    _expect(stride >= 1, node, f"stride={stride} は未対応（正整数のみ）")
    _expect(padding >= 0, node, f"padding={padding} は未対応（非負整数のみ）")
    _expect(dilation >= 1, node, f"dilation={dilation} は未対応（正整数のみ）")
    _expect(groups >= 1, node, f"groups={groups} は未対応（正整数のみ）")
    return Emitted(
        "conv1d",
        3,
        {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
        synth_consts=_conv_bias(
            node, _static_size_or_fail(node, weight.shape[0], "conv1d の出力チャネル"), "conv1d"
        ),
    )


def _pair_spatial(node: Node, value: Any, what: str) -> list[int]:
    """conv2d の空間パラメータを `[H, W]` の 2 成分へ正規化する（ADR 0017）。

    torch は `1` / `[1]` / `[1,1]` の 3 表記を同じ意味で受ける（`nn.Conv2d(kernel_size=3)` は
    `stride=1` をスカラで持つ）。**IR の attrs は常に 2 成分**なので、表記の吸収は境界の
    ここで済ませる（ランタイム側に「スカラなら両軸へ配る」規則を持ち込まない）。
    """
    if isinstance(value, (list, tuple)):
        items = list(value)
        _expect(len(items) in (1, 2), node, f"{what}={value!r} は未対応（conv2d の空間軸は 2 本）")
        if len(items) == 1:
            items = [items[0], items[0]]
    else:
        items = [value, value]
    for item in items:
        _expect(
            isinstance(item, int) and not isinstance(item, bool),
            node,
            f"{what}={value!r} が整数でない",
        )
    return [int(item) for item in items]


def _h_conv2d(node: Node) -> Emitted:
    """aten.conv2d → conv2d（attrs `stride` / `padding` / `dilation` / `groups` — ADR 0017）。

    MUST: 4 つとも attrs に**明示**する（既定値補完に頼らない — ADR 0012 / 0015）。空間 3 つは
    H/W の 2 成分で、`groups` だけスカラ。
    MUST: `groups` が Cin / Cout を割り切ることと重み `[Cout, Cin/groups, Kh, Kw]` の整合は
    shape 層（shapes.py）が見る。ここで見るのは表記の正規化と値域だけ。
    """
    extra = sorted(set(node.kwargs) - {"bias", "stride", "padding", "dilation", "groups"})
    _expect(not extra, node, f"kwargs {extra} を伴う conv2d は未対応")
    src = node.args[0].meta["val"]
    _expect(src.dim() == 4, node, f"rank {src.dim()} の conv2d は未対応（[B,Cin,H,W] のみ）")
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 4, node, f"rank {weight.dim()} の conv2d 重みは未対応")
    stride = _pair_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _pair_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    dilation = _pair_spatial(node, _arg_or_kwarg(node, 5, "dilation", 1), "dilation")
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    _expect(
        isinstance(groups, int) and not isinstance(groups, bool),
        node,
        f"groups={groups!r} が整数でない",
    )
    _expect(all(value >= 1 for value in stride), node, f"stride={stride} は未対応（正整数のみ）")
    _expect(all(value >= 0 for value in padding), node, f"padding={padding} は未対応（非負のみ）")
    _expect(
        all(value >= 1 for value in dilation), node, f"dilation={dilation} は未対応（正整数のみ）"
    )
    _expect(groups >= 1, node, f"groups={groups} は未対応（正整数のみ）")
    return Emitted(
        "conv2d",
        3,
        {"stride": stride, "padding": padding, "dilation": dilation, "groups": int(groups)},
        synth_consts=_conv_bias(
            node, _static_size_or_fail(node, weight.shape[0], "conv2d の出力チャネル"), "conv2d"
        ),
    )


def _h_deform_conv2d(node: Node) -> Emitted:
    """torchvision.deform_conv2d → deform_conv2d（attrs `padding` のみ — ADR 0055）。

    スキーマは `(Tensor input, Tensor weight, Tensor offset, Tensor mask, Tensor bias,
    SymInt stride_h, SymInt stride_w, SymInt pad_h, SymInt pad_w, SymInt dilation_h,
    SymInt dilation_w, SymInt groups, SymInt offset_groups, bool use_mask)` で、位置引数の
    テンソル 5 本がそのまま IR の入力 5 本になる（`_tensor_args` は位置順）。

    `torchvision::` 名前空間のカスタム op なので Core ATen の分解表に登録が無く、
    `run_decompositions(curated_decompositions())` の後も 1 ノードで残る（実測）。

    MUST: `use_mask=False`（DCNv1）を落とす。契約は mask をスロットとして要求する DCNv2 専業で、
    torchvision は `mask=None` のとき **`[1,1]` のダミーテンソル**を渡してくる — 素通しすると
    ダミーが modulator として掛かる沈黙誤値になる（shape 検査が拾う保証はない）。
    MUST: `stride` / `dilation` / `groups` / `offset_groups` の 1 以外を落とす。契約に欄が無い
    ので、ここが唯一の門（既定値補完に頼らず**実値**を見る）。
    MUST: rank 4 と f32 に絞る（契約は f32 専業）。
    NOTE: bias 無しは torchvision の Python ラッパが `aten.full([Cout], 0)` を挿すので、
    エクスポータ側のゼロ bias 合成（conv 族の `_conv_bias`）は要らない — 合成経路を二重に
    持たない。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う deform_conv2d は未対応")
    _expect(
        len(node.args) == 14,
        node,
        f"引数 {len(node.args)} 本の deform_conv2d は未対応（スキーマは 14 本）",
    )
    (
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        offset_groups,
        use_mask,
    ) = node.args[5:]
    # MUST: use_mask を**テンソルの形より先に**見る。DCNv1 の mask スロットには [1,1] の
    # ダミーが入っているので、順序を入れ替えると診断が「rank 2 の mask」になって
    # 「DCNv1 が語彙に無い」ことが読み取れなくなる（ADR 0005 の診断規律）。
    _expect(
        use_mask is True,
        node,
        f"use_mask={use_mask!r} の deform_conv2d は未対応"
        "（契約は mask 必須の DCNv2 専業 — スロットを省く表現が無い）",
    )
    src, weight, offset, mask = (node.args[index].meta["val"] for index in range(4))
    for name, value in (("入力", src), ("重み", weight), ("offset", offset), ("mask", mask)):
        _expect(value.dim() == 4, node, f"rank {value.dim()} の deform_conv2d {name} は未対応")
        _expect(
            value.dtype == torch.float32,
            node,
            f"dtype {value.dtype} の deform_conv2d {name} は未対応（f32 のみ）",
        )

    # スキーマ上は SymInt なので、まず**静的な素の int** であることを見る（記号の
    # stride / padding は attrs に載せられない）。
    def static_int(what: str, value: Any) -> int:
        _expect(
            isinstance(value, int) and not isinstance(value, bool),
            node,
            f"{what}={value!r} が静的な整数でない",
        )
        return int(value)

    for what, value in (
        ("stride_h", stride_h),
        ("stride_w", stride_w),
        ("dilation_h", dilation_h),
        ("dilation_w", dilation_w),
        ("groups", groups),
        ("offset_groups", offset_groups),
    ):
        _expect(
            static_int(what, value) == 1,
            node,
            f"{what}={value!r} の deform_conv2d は未対応（attrs に欄が無い = 1 固定）",
        )
    padding = [static_int("pad_h", pad_h), static_int("pad_w", pad_w)]
    _expect(all(value >= 0 for value in padding), node, f"padding={padding} は未対応（非負のみ）")
    return Emitted("deform_conv2d", 5, {"padding": padding})


def _gru_scan_emitted(node: Node, name: str) -> Emitted:
    """karume.gru_scan{,_reverse} → 同名の IR op（attrs 空 — ADR 0056）。

    スキーマは `(Tensor gi, Tensor h0, Tensor w_hh, Tensor b_hh) -> Tensor` で、位置引数の
    テンソル 4 本がそのまま IR の入力 4 本になる（`_tensor_args` は位置順）。`karume::` 名前
    空間の custom op なので Core ATen の分解表に登録が無く、`run_decompositions` の後も
    1 ノードで残る（実測 — `karume/custom_ops.py` の docstring）。

    MUST: 走査方向は **op 名**をそのまま IR op 名にする（attrs へ落とさない）。名前を 1 本に
    畳んで attrs で分けると、契約表に bool の検査関数を新設することになる（既存の検査関数は
    どれも `bool` を int の派生として明示的に弾いている — ops.py の `_assert_integer_attr`）。
    名前は overload から導かず**呼び分けで渡す**（`ATEN_HANDLERS` のキーと 1 対 1 になり、
    torch の内部表現の綴りに依存しない）。
    MUST: rank と f32 に絞る（契約は f32 専業）。多層 / 双方向 / `h_n` の消費形は **この op に
    到達する前**の形で表される — 呼び手が層と方向ごとにノードを並べる（ADR 0056 決定 7）ので、
    ここで検出できるのは「1 ノードの形が契約外」だけ。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う gru_scan は未対応")
    _expect(
        len(node.args) == 4,
        node,
        f"引数 {len(node.args)} 本の gru_scan は未対応（スキーマは 4 本）",
    )
    gi, initial, weight, bias = (node.args[index].meta["val"] for index in range(4))
    for slot, value, rank in (
        ("gi", gi, 3),
        ("h0", initial, 2),
        ("w_hh", weight, 2),
        ("b_hh", bias, 1),
    ):
        _expect(
            value.dim() == rank,
            node,
            f"rank {value.dim()} の gru_scan {slot} は未対応（rank {rank} のみ）",
        )
        _expect(
            value.dtype == torch.float32,
            node,
            f"dtype {value.dtype} の gru_scan {slot} は未対応（f32 のみ）",
        )
    return Emitted(name, 4, {})


def _h_gru_scan(node: Node) -> Emitted:
    return _gru_scan_emitted(node, GRU_SCAN_OP)


def _h_gru_scan_reverse(node: Node) -> Emitted:
    return _gru_scan_emitted(node, GRU_SCAN_REVERSE_OP)


def _h_upsample_bilinear2d(node: Node) -> Emitted:
    """aten.upsample_bilinear2d.vec → upsample_bilinear2d（attrs `output_size` のみ）。

    シグネチャは `(Tensor input, SymInt[]? output_size, bool align_corners,
    float[]? scale_factors)`。実測（BiRefNet 一族 / Depth Anything V2 本家）は全て
    `F.interpolate(..., size=(H,W), mode="bilinear", align_corners=True)` で、
    `scale_factors` 指定はアクティブ経路に 1 件も無い。

    MUST: `align_corners=False` を落とす。座標式（`scale·(i+0.5) − 0.5`）も端の扱いも別物で、
    受理すると**同じ op 名で数値が変わる**（gelu / gelu_tanh と同じ理由 — 需要が出たら
    別 op として足す）。契約に欄が無いので、ここが唯一の門。
    MUST: `scale_factors` 指定を落とす。倍率から出力長を導く形は丸めの規約
    （`floor(in·factor)`）がもう 1 つ増え、`output_size` 形と 2 通りの IR ができる。
    MUST: rank 4 と f32 に絞る。3D / 5D 版は別 aten op なのでここへは来ないが、f16 / f64 は
    同じ overload で来る（契約は f32 専業）。
    """
    _expect(not node.kwargs, node, f"kwargs {sorted(node.kwargs)} を伴う upsample は未対応")
    src = node.args[0].meta["val"]
    _expect(
        src.dim() == 4, node, f"rank {src.dim()} の upsample_bilinear2d は未対応（[B,C,H,W] のみ）"
    )
    _expect(
        src.dtype == torch.float32,
        node,
        f"dtype {src.dtype} の upsample_bilinear2d は未対応（f32 のみ）",
    )
    output_size = node.args[1]
    align_corners = node.args[2]
    scale_factors = node.args[3] if len(node.args) > 3 else None
    _expect(
        align_corners is True,
        node,
        f"align_corners={align_corners!r} の upsample_bilinear2d は未対応"
        "（契約は align_corners=True 専業 — 欄が無い）",
    )
    _expect(
        scale_factors is None,
        node,
        f"scale_factors={scale_factors!r} 指定の upsample_bilinear2d は未対応"
        "（出力長は size 指定のみ）",
    )
    _expect(
        isinstance(output_size, (list, tuple)) and len(output_size) == 2,
        node,
        f"output_size={output_size!r} は未対応（[Hout, Wout] の 2 成分のみ）",
    )
    _expect(
        not _has_free_symbols(output_size),
        node,
        f"記号を含む output_size={output_size!r} は未対応（attrs は静的な整数のみ）",
    )
    sizes = [int(size) for size in output_size]
    _expect(all(size >= 1 for size in sizes), node, f"output_size={sizes} は未対応（正整数のみ）")
    return Emitted("upsample_bilinear2d", 1, {"output_size": sizes})


def _h_conv_transpose1d(node: Node) -> Emitted:
    """aten.conv_transpose1d → conv_transpose1d（attrs `stride` / `padding` — ADR 0015）。

    MUST: 位置引数の末尾既定値は保存形で**省略される**（ups3/4 は `(input, weight, bias,
    [stride])` の 4 引数形が実在 — recon §4）。`_arg_or_kwarg` が欠けた位置を既定で埋めるので
    IndexError にはならないが、**補完した値をそのまま通さない**のがここの役目:
    `output_padding` / `groups` / `dilation` は契約に欄が無い（実測どおり 0 / 1 / 1 固定）ので、
    既定以外は落とす。
    MUST: `stride >= 1`（stride 0 はカーネルのゼロ除算・GPU ハング — recon §4）。
    重みが `[Cin, Cout, K]` であることと出力長 `L·stride` の成立は shape 層が見る。
    """
    extra = sorted(
        set(node.kwargs) - {"bias", "stride", "padding", "output_padding", "groups", "dilation"}
    )
    _expect(not extra, node, f"kwargs {extra} を伴う conv_transpose1d は未対応")
    src = node.args[0].meta["val"]
    _expect(
        src.dim() == 3, node, f"rank {src.dim()} の conv_transpose1d は未対応（[B,Cin,L] のみ）"
    )
    weight = node.args[1].meta["val"]
    _expect(weight.dim() == 3, node, f"rank {weight.dim()} の conv_transpose1d 重みは未対応")
    stride = _single_spatial(node, _arg_or_kwarg(node, 3, "stride", 1), "stride")
    padding = _single_spatial(node, _arg_or_kwarg(node, 4, "padding", 0), "padding")
    output_padding = _single_spatial(
        node, _arg_or_kwarg(node, 5, "output_padding", 0), "output_padding"
    )
    groups = _arg_or_kwarg(node, 6, "groups", 1)
    dilation = _single_spatial(node, _arg_or_kwarg(node, 7, "dilation", 1), "dilation")
    _expect(stride >= 1, node, f"stride={stride} は未対応（正整数のみ — stride 0 はハング）")
    _expect(padding >= 0, node, f"padding={padding} は未対応（非負整数のみ）")
    _expect(
        output_padding == 0,
        node,
        f"output_padding={output_padding} の conv_transpose1d は未対応（attrs に欄が無い）",
    )
    _expect(groups == 1, node, f"groups={groups} の conv_transpose1d は未対応（attrs に欄が無い）")
    _expect(
        dilation == 1,
        node,
        f"dilation={dilation} の conv_transpose1d は未対応（attrs に欄が無い）",
    )
    # 重みは [Cin, Cout, K] — Cout は第 2 軸（conv1d の第 1 軸ではない）
    return Emitted(
        "conv_transpose1d",
        3,
        {"stride": stride, "padding": padding},
        synth_consts=_conv_bias(
            node,
            _static_size_or_fail(node, weight.shape[1], "conv_transpose1d の出力チャネル"),
            "conv_transpose1d",
        ),
    )


ATEN_HANDLERS = {
    aten.neg.default: _simple("neg", 1),
    aten.abs.default: _simple("abs", 1),
    aten.exp.default: _simple("exp", 1),
    aten.log.default: _simple("log", 1),
    aten.sqrt.default: _simple("sqrt", 1),
    # 定数部分木の中では FOLDABLE_OPS が畳む（RoPE 表）。実行時値を取る形だけがここへ来る —
    # DACVAE の Snake 活性 `x + (α+1e-9)⁻¹·sin²(αx)` が初出（ADR 0043 の第 1 層）。
    aten.sin.default: _simple("sin", 1),
    aten.tanh.default: _simple("tanh", 1),
    aten.sigmoid.default: _simple("sigmoid", 1),
    aten.relu.default: _simple("relu", 1),
    aten.gelu.default: _h_gelu,
    aten.add.Tensor: _h_binary("add"),
    aten.sub.Tensor: _h_binary("sub"),
    aten.mul.Tensor: _h_binary("mul"),
    aten.div.Tensor: _h_binary("div"),
    aten.mm.default: _simple("matmul", 2),
    aten.bmm.default: _h_bmm,
    aten.gather.default: _h_gather,
    aten.sum.dim_IntList: _h_sum,
    aten.amax.default: _h_row_reduce("amax"),
    aten.amin.default: _h_row_reduce("amin"),
    aten._to_copy.default: _h_to_copy,
    aten.bitwise_not.default: _h_bitwise_not,
    # 波3 の数理 op（sdp の spline / dec の leaky_relu — recon §2）
    aten.log1p.default: _simple("log1p", 1),
    aten.where.self: _h_where,
    aten.clamp.default: _h_clamp,
    aten.leaky_relu.default: _h_leaky_relu,
    aten.ge.Scalar: _h_compare_scalar("ge_scalar"),
    aten.le.Scalar: _h_compare_scalar("le_scalar"),
    aten.gt.Scalar: _h_compare_scalar("gt_scalar"),
    aten.ge.Tensor: _h_ge_tensor,
    aten.bitwise_and.Tensor: _h_bitwise_and,
    aten.cumsum.default: _h_cumsum,
    # レイアウト（ADR 0011）— 要素順を変えない 3 形は reshape 1 本へ正規化する
    aten.view.default: _h_reshape,
    aten.unsqueeze.default: _h_reshape,
    aten.squeeze.dims: _h_reshape,
    aten.permute.default: _h_permute,
    aten.expand.default: _h_expand,
    # 波4 のレイアウト第 2 群（ADR 0014）。slice / cat は FOLDABLE_OPS にも載っている —
    # 定数部分木の中では畳まれ、実行系に残った形だけがこのハンドラへ来る（両立する）。
    aten.slice.Tensor: _h_slice,
    aten.cat.default: _h_cat,
    aten.constant_pad_nd.default: _h_constant_pad_nd,
    aten.flip.default: _h_flip,
    # 融合 op（ADR 0012）— 分解を止めた高位 op を 1 カーネルへ落とす
    aten.linear.default: _h_linear,
    aten.layer_norm.default: _h_layer_norm,
    aten.softmax.int: _h_softmax,
    aten.embedding.default: _h_embedding,
    aten.masked_fill.Scalar: _h_masked_fill,
    aten.conv1d.default: _h_conv1d,
    aten.conv_transpose1d.default: _h_conv_transpose1d,
    # M1-P4 波 2（ADR 0016 / 0017）— Anima の 4 コンポーネント。
    # NOTE: `squeeze.dim` / `transpose.int` は curated decomp 後の実測に 1 本も現れない
    # （recon §8 の未確定事項の実測結果）。`_select_to_squeeze` が出すのは `squeeze.dims` で
    # 既存の reshape 正規化が受けるので、ここには足さない（出たら 1 行で足す）。
    aten.rms_norm.default: _h_rms_norm,
    aten.conv2d.default: _h_conv2d,
    # 双線形 resample（第 1 層 — BiRefNet 一族 / Depth Anything V2 の共通前提）。
    # `F.interpolate(size=…, mode="bilinear")` が落ちるのは `.vec` overload だけで、
    # `.default`（scales_h / scales_w を個別に取る形）は実測に現れない（出たら 1 行で足す）。
    aten.upsample_bilinear2d.vec: _h_upsample_bilinear2d,
    # DCNv2（第 1' 層 — BiRefNet 一族の正面 blocker。ADR 0055）。`torchvision::` 名前空間の
    # カスタム op なので Core ATen の分解表に登録が無く、curated decomp 後も 1 ノードで残る。
    torch.ops.torchvision.deform_conv2d.default: _h_deform_conv2d,
    # GRU 隠れ側スキャン 2 方向（第 2 層 — ADR 0056）。`karume::` 名前空間の自前 custom op で、
    # 分解表に無いので curated decomp 後も 1 ノード = **時間軸 T が記号のまま残る**。
    torch.ops.karume.gru_scan.default: _h_gru_scan,
    torch.ops.karume.gru_scan_reverse.default: _h_gru_scan_reverse,
    # perf-a（ADR 0023）— SDPA を保存したターゲットだけがこのハンドラに来る
    # （`curated_decompositions(preserved=…)` がターゲット別に選ぶ）。
    aten.scaled_dot_product_attention.default: _h_attention,
}
