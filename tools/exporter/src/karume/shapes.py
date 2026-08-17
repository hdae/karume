"""出力 shape 規則をエクスポータ側にも持つ層
（TS 側 packages/runtime/src/ops.ts `computeOutputShape` と同義）。

目的は「export は緑、Session 構築で初めて落ちる」を潰すこと。torch の meta が付けた宣言
shape を鵜呑みにせず、**契約表の規則から独立に計算した shape** と全ノードで突き合わせ、
食い違いは export の時点で fail loudly にする（ADR 0005 の診断規律）。

TS 側との違いは定義域だけ:

- TS は**束縛解決後の数値 shape**を扱う（実行前に必ず具体値が決まる — ADR 0004）。
- ここは**束縛前の宣言 shape**を扱うので、次元は整数か次元言語 `coeff·sym+offset`
  （dims.py）。整数だけの入力に対しては TS と 1 対 1 で同じ結論を出す
  （適合ケース表 packages/runtime/tests/fixtures/op-contracts.json の `shapes` 節が
  両側を突き合わせる）。

MUST: 束縛が要る判定はここでやらない — 「記号次元なら黙って緩める」のではなく、
**束縛後にしか決まらない規則はランタイム側の層が持つ**という分担にする。該当は 2 つだけ:

1. 長さ 0 の軸の縮約禁止（amax / amin / softmax）— 記号次元が 0 になるかは束縛次第。
2. `sym_prefix_slice` の Tmax 超過（prefix 長 ≤ 定数次元）— 上限は IR に載らない。

どちらも TS 側（plan.ts → computeOutputShape）が実行前に必ず見るので、層として穴は無い。

逆向きの分担が 1 つある: **レイアウト第 2 群の対象軸の記号規則** — `slice` / `flip` は
「静的軸のみ」（ADR 0014）、`cat` は「同一シンボルの一次和まで」（ADR 0046）— は、ここ
（宣言 shape を見る層）でしか判定できない。TS 側は束縛解決後の数値 shape しか扱わないので、
同じ規則を plan.ts の `validateGraphContracts`（宣言 shape を見る層）が持つ — 層は違うが
受理集合は両側で同じ。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from karume.dims import MAX_SAFE_INT, DimError, DimExpr, format_dim, parse_dim
from karume.ir import IrDim, IrGraph
from karume.ops import (
    OpContract,
    OpContractError,
    arity_fits,
    assert_strided_rank,
    attention_scale,
    axis_dim,
    conv1d_attrs,
    conv2d_attrs,
    conv_transpose1d_attrs,
    cumsum_dim,
    deform_conv2d_attrs,
    describe_arity,
    layer_norm_attrs,
    pad_attrs,
    permute_dims,
    reduce_dim,
    resolve_op_contract,
    rms_norm_eps,
    scalar_param_values,
    slice_attrs,
    softmax_dim,
    sym_prefix_slice_attrs,
    topk_k,
    upsample_bilinear2d_attrs,
)


@dataclass(frozen=True)
class Extent:
    """1 次元の長さ。定数は `sym=None`（coeff は 0）、記号次元は `coeff·sym+offset`。

    等値は構造的（dataclass の既定）— 記号のまま「同じ長さか」を判定できる唯一の手段で、
    `T` と `2T` を取り違えない。
    """

    coeff: int
    sym: str | None
    offset: int

    @property
    def is_const(self) -> bool:
        return self.sym is None

    def is_value(self, value: int) -> bool:
        """定数次元でちょうど `value` か（記号次元は常に False — 束縛次第なので断定しない）。"""
        return self.sym is None and self.offset == value

    def to_dim(self) -> IrDim:
        if self.sym is None:
            return self.offset
        return format_dim(DimExpr(coeff=self.coeff, sym=self.sym, offset=self.offset))


_ONE = Extent(coeff=0, sym=None, offset=1)


def _extent(dim: Any, where: str) -> Extent:
    # bool は int の派生だが次元ではない（True が 1 として黙って通るのを塞ぐ）。
    if isinstance(dim, bool) or not isinstance(dim, (int, str)):
        raise OpContractError(f"{where}: 次元 {dim!r} が整数でも次元式でもない")
    if isinstance(dim, int):
        if not 0 <= dim <= MAX_SAFE_INT:
            raise OpContractError(f"{where}: 次元 {dim} が非負の安全整数でない")
        return Extent(coeff=0, sym=None, offset=dim)
    try:
        expr = parse_dim(dim)
    except DimError as cause:
        raise OpContractError(f"{where}: {cause}") from cause
    return Extent(coeff=expr.coeff, sym=expr.sym, offset=expr.offset)


def extents(shape: Sequence[IrDim], where: str) -> list[Extent]:
    """宣言 shape を長さの列へ落とす（比較・計算はこの表現の上でだけ行う）。"""
    return [_extent(dim, f"{where} の次元 {index}") for index, dim in enumerate(shape)]


def _show(shape: Sequence[Extent]) -> str:
    return ",".join(str(extent.to_dim()) for extent in shape)


def _at(shape: Sequence[Extent], index: int) -> Extent:
    """右詰め broadcast の読み出し（範囲外は長さ 1）。

    MUST: 負の添字を Python の巻き戻しに任せない — 末尾から拾ってしまい、rank の違う
    被演算子の broadcast が黙って別の軸と揃う。
    """
    return shape[index] if index >= 0 else _ONE


def broadcast_extents(a: Sequence[Extent], b: Sequence[Extent], where: str) -> list[Extent]:
    """torch 準拠の右詰め broadcast（packages/runtime/src/ops.ts `broadcastShapes` と同義）。

    MUST: 結果を max で決めない — 0 と 1 の組（max なら 1）が torch では 0 になる。
    """
    rank = max(len(a), len(b))
    out: list[Extent] = []
    for index in range(rank):
        da = _at(a, len(a) - rank + index)
        db = _at(b, len(b) - rank + index)
        if da == db or db.is_value(1):
            out.append(da)
        elif da.is_value(1):
            out.append(db)
        elif da.is_const and db.is_const:
            raise OpContractError(
                f"{where}: shape [{_show(a)}] と [{_show(b)}] は右詰め broadcast できない"
            )
        else:
            # 記号どうし（または記号と定数）が一致しない形。束縛次第では成立しうるが、
            # 宣言だけでは判定できない — 黙って通すと「実行してみないと分からない IR」を
            # エクスポータが書けてしまう。
            raise OpContractError(
                f"{where}: shape [{_show(a)}] と [{_show(b)}] の次元 {index}"
                f"（{da.to_dim()} と {db.to_dim()}）が宣言だけでは broadcast 可否を決められない"
            )
    return out


def _numel_key(shape: Sequence[Extent]) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    """要素数の同一性判定に使う鍵（reshape の要素数一致検査）。

    要素数は一次式の積なので、記号のまま比較するには因数分解が要る。各次元を
    「内容（係数とオフセットの gcd）× 原始一次式」に分け、内容は整数側へ、原始一次式は
    多重集合へ積む — 一次式は多変数多項式環（UFD）で既約なので、この組が一致することと
    多項式として等しいことは同値（`2T+2` と `2·(T+1)` を同じ鍵にする）。

    MUST: 整数側が 0 なら鍵も 0 に潰す（数値 shape での `0 × 何か = 0` と結論を揃える）。
    """
    scalar = 1
    factors: list[tuple[str, int, int]] = []
    for extent in shape:
        if extent.sym is None:
            scalar *= extent.offset
            continue
        content = math.gcd(extent.coeff, extent.offset)
        scalar *= content
        factors.append((extent.sym, extent.coeff // content, extent.offset // content))
    if scalar == 0:
        return (0, ())
    return (scalar, tuple(sorted(factors)))


def _require_declared(
    declared: Sequence[IrDim] | None, contract: OpContract, where: str
) -> list[Extent]:
    if declared is None:
        raise OpContractError(
            f"{where}: op '{contract.name}' の出力 shape は宣言が目標形（declared が要る）"
        )
    return extents(declared, f"{where} の宣言 shape")


def compute_output_shape(
    contract: OpContract,
    input_shapes: Sequence[Sequence[IrDim]],
    where: str,
    *,
    declared: Sequence[IrDim] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> list[list[IrDim]]:
    """宣言 shape から**出力 slot 昇順の shape 列**を計算する
    （packages/runtime/src/ops.ts `computeOutputShape` と同義 — ADR 0068 決定 1）。

    2 本を返すのは `topk`（値 + 添字 — ADR 0068 決定 3）だけで、他は全て 1 本。
    `declared` は「出力の宣言 shape が目標形」の op（reshape / expand — ADR 0011）でだけ
    必須。`attrs` は permute / layer_norm / softmax / conv1d / sym_prefix_slice / topk で必須。
    """
    if not arity_fits(contract, len(input_shapes)):
        raise OpContractError(
            f"{where}: op '{contract.name}' の入力 shape 数が {len(input_shapes)}"
            f"（契約は {describe_arity(contract)}）"
        )
    ins = [extents(shape, f"{where} の入力 {index}") for index, shape in enumerate(input_shapes)]
    node_attrs: Mapping[str, Any] = attrs if attrs is not None else {}
    return [
        [extent.to_dim() for extent in slot]
        for slot in _compute(contract, ins, where, declared, node_attrs)
    ]


def _sole(shape: list[Extent]) -> list[list[Extent]]:
    """単一出力 op のアーム（出力が 1 本であることをアームごとに明示する — ADR 0068 決定 1）。"""
    return [shape]


def _compute(
    contract: OpContract,
    ins: list[list[Extent]],
    where: str,
    declared: Sequence[IrDim] | None,
    attrs: Mapping[str, Any],
) -> list[list[Extent]]:
    kind = contract.kind
    if kind == "unary":
        # MUST: スカラ attr の値域と**キーを跨ぐ不変条件**（clamp の min <= max）をここで見る。
        # 全ノードが必ず通る共通経路はこの計算だけで、attrs スキーマはキー単位の検査しか
        # 表せない（assert_graph_shapes が全ノードでここを呼ぶ）。
        scalar_param_values(contract, attrs, where)
        return _sole(list(ins[0]))
    if kind == "cast":
        return _sole(list(ins[0]))
    if kind == "binary":
        return _sole(broadcast_extents(ins[0], ins[1], f"{where} ({contract.name})"))
    if kind == "where":
        # torch と同じく 3 者を右詰め broadcast する（条件も値と同じ規則で広がる）。
        label = f"{where} ({contract.name})"
        return _sole(broadcast_extents(broadcast_extents(ins[0], ins[1], label), ins[2], label))
    if kind == "cumsum":
        return _sole(_cumsum(ins, where, attrs))
    if kind == "matmul":
        return _sole(_matmul(ins, where))
    if kind == "bmm":
        return _sole(_bmm(ins, where))
    if kind == "gather":
        return _sole(_gather(ins, where))
    if kind == "row_reduce":
        return _sole(_row_reduce(contract, ins, where, attrs))
    if kind == "argmax":
        return _sole(_argmax(ins, where))
    # 唯一の多出力アーム（値 + 添字 — ADR 0068 決定 3）。_sole を通らないことがそのまま
    # 「出力が 2 本」の宣言になる。
    if kind == "topk":
        return _topk(ins, where, attrs)
    if kind == "reshape":
        return _sole(_reshape(contract, ins, where, declared))
    if kind == "permute":
        return _sole(_permute(ins, where, attrs))
    if kind == "expand":
        return _sole(_expand(contract, ins, where, declared))
    if kind == "slice":
        return _sole(_slice(ins, where, attrs))
    if kind == "cat":
        return _sole(_cat(ins, where, attrs))
    if kind == "pad":
        return _sole(_pad(ins, where, attrs))
    if kind == "flip":
        return _sole(_flip(ins, where, attrs))
    if kind == "sym_prefix_slice":
        return _sole(_sym_prefix_slice(ins, where, attrs))
    if kind == "linear":
        return _sole(_linear(ins, where))
    if kind == "layer_norm":
        return _sole(_layer_norm(ins, where, attrs))
    if kind == "rms_norm":
        return _sole(_rms_norm(ins, where, attrs))
    # safe_softmax は shape 規則も attrs も softmax と同一（違いは空行の値だけ — ADR 0044）。
    if kind in ("softmax", "safe_softmax"):
        return _sole(_softmax(ins, where, attrs, contract.name))
    if kind == "attention":
        return _sole(_attention(ins, where, attrs))
    if kind == "embedding":
        return _sole(_embedding(ins, where))
    if kind == "masked_fill":
        return _sole(_masked_fill(ins, where))
    if kind == "conv1d":
        return _sole(_conv1d(ins, where, attrs))
    if kind == "conv2d":
        return _sole(_conv2d(ins, where, attrs))
    if kind == "conv_transpose1d":
        return _sole(_conv_transpose1d(ins, where, attrs))
    if kind == "deform_conv2d":
        return _sole(_deform_conv2d(ins, where, attrs))
    if kind == "upsample_bilinear2d":
        return _sole(_upsample_bilinear2d(ins, where, attrs))
    # 走査方向は op 名が持つが、shape 規則は 2 方向で完全に同一（出力の時間順も変わらない）。
    if kind == "gru_scan":
        return _sole(_gru_scan(contract, ins, where))
    raise OpContractError(f"{where}: op kind '{kind}' の shape 規則が無い")


def _matmul(ins: list[list[Extent]], where: str) -> list[Extent]:
    a, b = ins
    if len(a) != 2 or len(b) != 2:
        raise OpContractError(
            f"{where}: matmul は rank-2 × rank-2 のみ（[{_show(a)}] × [{_show(b)}]）"
        )
    if a[1] != b[0]:
        raise OpContractError(f"{where}: matmul の縮約次元が不一致 [{_show(a)}] × [{_show(b)}]")
    return [a[0], b[1]]


def _bmm(ins: list[list[Extent]], where: str) -> list[Extent]:
    a, b = ins
    # MUST: rank-2 を通さない（matmul の担当）。兼用にするとバッチ軸を落とした形が同じ op 名で
    # 通り、B の取り違えが shape 検査を素通りする。
    if len(a) != 3 or len(b) != 3:
        raise OpContractError(
            f"{where}: bmm は rank-3 × rank-3 のみ（rank-2 は matmul）: [{_show(a)}] × [{_show(b)}]"
        )
    if a[0] != b[0]:
        raise OpContractError(f"{where}: bmm のバッチ次元が不一致 [{_show(a)}] × [{_show(b)}]")
    if a[2] != b[1]:
        raise OpContractError(f"{where}: bmm の縮約次元が不一致 [{_show(a)}] × [{_show(b)}]")
    return [a[0], a[1], b[2]]


def _gather(ins: list[list[Extent]], where: str) -> list[Extent]:
    src, index = ins
    # 契約は「最終次元固定」— 先行次元は src と index で完全一致し、最終次元だけが自由
    # （torch の一般 gather より狭い）。
    if not src or len(index) != len(src):
        raise OpContractError(
            f"{where}: gather は src と index が同じ rank（1 以上）:"
            f" [{_show(src)}] / [{_show(index)}]"
        )
    for axis in range(len(src) - 1):
        if src[axis] != index[axis]:
            raise OpContractError(
                f"{where}: gather の先行次元 {axis} が不一致 [{_show(src)}] / [{_show(index)}]"
            )
    # 出力は index と同形（値は src から引く）。添字の値域は実行時データ依存なので見ない。
    return list(index)


def _row_reduce(
    contract: OpContract, ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    shape = ins[0]
    if not shape:
        raise OpContractError(f"{where}: reduce の入力は rank 1 以上（スカラは縮約できない）")
    dim = reduce_dim(attrs, where)
    # MUST: 負値・rank 外は fail loudly（負の軸表記の正規化はハンドラ側の責務で、ここで
    # `% rank` を補うと「宣言と実 rank の食い違い」を黙って別の軸へ吸収してしまう）。
    if dim >= len(shape):
        raise OpContractError(
            f"{where}: op '{contract.name}' の attrs.dim={dim} が rank {len(shape)} の範囲外"
        )
    # 空軸の amax/amin は identity が定義できない（sum は 0）。記号次元が 0 になるかは
    # 束縛次第なので、ここで決められるのは定数次元だけ（残りはランタイム側の層）。
    if shape[dim].is_value(0) and contract.name != "sum":
        raise OpContractError(f"{where}: op '{contract.name}' は長さ 0 の軸を縮約できない")
    return shape[:dim] + shape[dim + 1 :]


def _argmax(ins: list[list[Extent]], where: str) -> list[Extent]:
    """argmax の出力 shape（ADR 0068 決定 2）— **最終次元を 1 に潰す rank 保存**。

    reduce 族と違い軸は attrs ではなく最終次元固定（欄の不存在が「他の軸は語彙に無い」の
    宣言）で、`keepdim` の欄も無いので出力 rank は入力と同じ。
    """
    shape = ins[0]
    if not shape:
        raise OpContractError(f"{where}: argmax の入力は rank 1 以上（スカラは縮約できない）")
    # 長さ 0 の軸に「最大値の添字」は無い（torch も拒否）。記号次元が 0 になるかは束縛次第
    # なので、ここで決められるのは定数次元だけ（残りはランタイム側の層 — amax と同じ分担）。
    if shape[-1].is_value(0):
        raise OpContractError(f"{where}: argmax は長さ 0 の最終次元を縮約できない")
    return [*shape[:-1], _ONE]


def _topk(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[list[Extent]]:
    """topk の出力 shape 列（ADR 0068 決定 3）— **値と添字の 2 本**とも `[..., k]`。

    軸は最終次元固定（argmax と同じ絞り）で rank は保存される。受理領域は
    `1 <= k <= 最終次元` で、下限は attrs スキーマ（TOPK_ATTRS）が、上限はここが見る。
    """
    shape = ins[0]
    if not shape:
        raise OpContractError(f"{where}: topk の入力は rank 1 以上（スカラは縮約できない）")
    k = topk_k(attrs, where)
    last = shape[-1]
    # 記号次元の最終次元では k との突合を**保留する**（束縛次第なので断定しない — 判定は
    # ランタイム側の層。amax の長さ 0 判定と同じ分担）。長さ 0 は k >= 1 との突合で落ちる。
    if last.is_const and k > last.offset:
        raise OpContractError(
            f"{where}: topk の attrs.k={k} が最終次元 {last.offset} を超える"
            f"（入力 [{_show(shape)}]）"
        )
    out = [*shape[:-1], Extent(coeff=0, sym=None, offset=k)]
    return [out, list(out)]


def _reshape(
    contract: OpContract, ins: list[list[Extent]], where: str, declared: Sequence[IrDim] | None
) -> list[Extent]:
    target = _require_declared(declared, contract, where)
    source = ins[0]
    # 契約は要素数一致だけ（要素順は変えない）。ここを緩めると別名化した実バッファの
    # 大きさと宣言 shape が食い違い、readback が範囲外まで読む。
    if _numel_key(source) != _numel_key(target):
        raise OpContractError(
            f"{where}: reshape の要素数が合わない [{_show(source)}] → [{_show(target)}]"
        )
    return target


def _permute(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    # 出力 rank は入力と同じ（並べ替えるだけ）なので入力側だけ見れば足りる。
    assert_strided_rank(len(source), "permute の入力", where)
    dims = permute_dims(attrs, where)
    if len(dims) != len(source):
        raise OpContractError(f"{where}: permute の dims {dims} が入力 rank {len(source)} と違う")
    out: list[Extent] = []
    for dim in dims:
        if dim >= len(source):
            raise OpContractError(
                f"{where}: permute の dims に入力 rank {len(source)} 外の軸 {dim} がある"
            )
        out.append(source[dim])
    return out


def _expand(
    contract: OpContract, ins: list[list[Extent]], where: str, declared: Sequence[IrDim] | None
) -> list[Extent]:
    target = _require_declared(declared, contract, where)
    source = ins[0]
    assert_strided_rank(len(source), "expand の入力", where)
    assert_strided_rank(len(target), "expand の出力", where)
    if len(target) < len(source):
        raise OpContractError(
            f"{where}: expand は rank を下げられない [{_show(source)}] → [{_show(target)}]"
        )
    # 右詰めで、入力の各次元は「目標と一致」か「長さ 1（stride 0 で複製）」のみ。
    offset = len(target) - len(source)
    for index, extent in enumerate(source):
        if not extent.is_value(1) and extent != target[offset + index]:
            raise OpContractError(
                f"{where}: expand は長さ 1 でない次元 {index}（{extent.to_dim()}）を"
                f" {target[offset + index].to_dim()} に拡張できない"
            )
    return target


def _axis_extent(shape: list[Extent], dim: int, op: str, where: str) -> Extent:
    """対象軸の長さを取り出す（rank の内側であることだけを見る）。"""
    if dim >= len(shape):
        raise OpContractError(f"{where}: {op} の dim {dim} が入力 rank {len(shape)} の外")
    return shape[dim]


def _static_axis(shape: list[Extent], dim: int, op: str, where: str) -> Extent:
    """`slice` / `flip` の対象軸を取り出す（**静的軸のみ** — ADR 0014）。

    MUST: 記号軸を拒否する。理由は op ごとに違うが結論は同じ:
    - `slice` — 記号軸の切り出しは sym_prefix_slice の担当（重複させない）。加えて範囲検査
      （end <= 軸長）が記号のままでは決められない。
    - `flip` — 実測は全て静的軸（flow の 192ch / sdp の 2ch）。動的軸は要求実測が出るまで
      広げない。

    `cat` の連結軸は ADR 0046 で「同一シンボルの一次和」まで緩めた（`_cat` を見よ）。

    NOTE: TS 側は**束縛解決後の数値 shape**しか見ないので、この判定は plan.ts
    （宣言 shape を見る層）が持つ — 層は違うが受理集合は両側で同じ。
    """
    extent = _axis_extent(shape, dim, op, where)
    if not extent.is_const:
        raise OpContractError(
            f"{where}: {op} の軸 {dim} が記号次元 {extent.to_dim()}"
            "（対象軸は静的でなければならない — 記号軸の切り出しは sym_prefix_slice）"
        )
    return extent


def _slice(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    # 実行は strided 読みコピー族の流用（ADR 0014）なので rank 上限も同じ。
    assert_strided_rank(len(source), "slice の入力", where)
    dim, start, end = slice_attrs(attrs, where)
    extent = _static_axis(source, dim, "slice", where)
    # MUST: 範囲外の切り出しを通さない（GPU では例外なしに隣の行を読む形になる）。
    if end > extent.offset:
        raise OpContractError(
            f"{where}: slice の end {end} が軸 {dim} の長さ {extent.offset} を超える"
        )
    # MUST: キーを跨ぐ不変条件（clamp の min <= max と同じ分担）。
    if start > end:
        raise OpContractError(f"{where}: slice の start {start} が end {end} を超える")
    out = list(source)
    out[dim] = Extent(coeff=0, sym=None, offset=end - start)
    return out


def _cat(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """連結軸は〈定数〉または〈**同一**シンボルの一次式〉（ADR 0046 が ADR 0014 を改訂）。

    MUST: 異なるシンボルが混ざる連結は落とす — 総和が次元言語（1 次元 1 シンボルの一次式）に
    載らない。同一シンボルなら `Σ(coeff_i·sym + offset_i)` がそのまま正準形になる。
    """
    dim = axis_dim(attrs, where)
    first = ins[0]
    assert_strided_rank(len(first), "cat の入力", where)
    coeff = 0
    offset = 0
    sym: str | None = None
    for index, shape in enumerate(ins):
        if len(shape) != len(first):
            raise OpContractError(
                f"{where}: cat の入力 {index} の rank {len(shape)} が 入力 0 の {len(first)} と違う"
            )
        extent = _axis_extent(shape, dim, "cat", where)
        if extent.sym is not None:
            if sym is not None and sym != extent.sym:
                raise OpContractError(
                    f"{where}: cat の連結軸 {dim} に異なるシンボル（{sym} と {extent.sym}）が"
                    "混ざる — 和が次元言語 coeff·sym+offset に載らない"
                )
            sym = extent.sym
        # MUST: 連結軸**以外**は全一致（torch と同じ）。緩めると出力の一部がどの入力にも
        # 書かれないまま残り、full-write 不変条件が破れる。
        for axis, other in enumerate(shape):
            if axis != dim and other != first[axis]:
                raise OpContractError(
                    f"{where}: cat の入力 {index} [{_show(shape)}] が"
                    f" 入力 0 [{_show(first)}] と軸 {axis} で違う（連結軸は {dim}）"
                )
        coeff += extent.coeff
        offset += extent.offset
    out = list(first)
    # 出力の軸長 = 入力の軸長の総和。この規則そのものが「全入力で出力全域を覆う」
    # （full-write — ADR 0014）の担保になっている。定数だけなら sym=None・coeff=0 に戻る。
    out[dim] = Extent(coeff=coeff, sym=sym, offset=offset)
    return out


def _pad(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    if not source:
        raise OpContractError(f"{where}: pad の入力は rank 1 以上（最終次元が要る）")
    left, right = pad_attrs(attrs, where)
    last = source[-1]
    out = list(source)
    # 記号長の最終次元も埋められる（実測の T+2w = T+8 — 次元言語のオフセット付き形）。
    out[-1] = Extent(coeff=last.coeff, sym=last.sym, offset=last.offset + left + right)
    return out


def _flip(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    source = ins[0]
    if not source:
        raise OpContractError(f"{where}: flip の入力は rank 1 以上（反転する軸が要る）")
    _static_axis(source, axis_dim(attrs, where), "flip", where)
    # 反転は shape を変えない（恒等 shape 規則）。
    return list(source)


def _sym_prefix_slice(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    source = ins[0]
    # 出力 rank は入力と同じ（各軸の先頭を切り出すだけ）。
    assert_strided_rank(len(source), "sym_prefix_slice の入力", where)
    sym, slices = sym_prefix_slice_attrs(attrs, where)
    out = list(source)
    for entry in slices:
        dim = entry["dim"]
        if dim >= len(source):
            raise OpContractError(
                f"{where}: sym_prefix_slice の dim {dim} が入力 rank {len(source)} の外"
            )
        # MUST: 入力は Tmax で焼いた静的形（ADR 0010）。記号次元の入力を許すと読み出し
        # stride が実行ごとに変わり、prefix の意味が壊れる。
        if not source[dim].is_const:
            raise OpContractError(
                f"{where}: sym_prefix_slice の入力 [{_show(source)}] の次元 {dim} が記号"
                "（入力は Tmax で焼いた静的形でなければならない）"
            )
        # prefix 長 coeff·sym+offset ≤ 定数次元（Tmax 超過）の検査は束縛が要るので
        # ランタイム側の層が持つ — 上限は IR に載らない。
        out[dim] = Extent(coeff=entry["coeff"], sym=sym, offset=entry["offset"])
    return out


def _linear(ins: list[list[Extent]], where: str) -> list[Extent]:
    x, weight, bias = ins
    if len(x) < 1 or len(weight) != 2 or len(bias) != 1:
        raise OpContractError(
            f"{where}: linear は x[…,in] × W[out,in] + b[out]（rank ≥ 1 / 2 / 1）:"
            f" [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    out_features, in_features = weight
    if x[-1] != in_features:
        raise OpContractError(
            f"{where}: linear の入力特徴数が不一致 [{_show(x)}] × [{_show(weight)}]"
        )
    if bias[0] != out_features:
        raise OpContractError(
            f"{where}: linear の bias 長 {bias[0].to_dim()} が"
            f" 出力特徴数 {out_features.to_dim()} と違う"
        )
    return [*x[:-1], out_features]


def _layer_norm(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    x, weight, bias = ins
    normalized_shape, _ = layer_norm_attrs(attrs, where)
    # 契約は「最終次元のみ」（attrs 検査済み）。ここでは実 shape の末尾との一致を見る。
    normalized = Extent(coeff=0, sym=None, offset=normalized_shape[0])
    if len(x) < 1 or x[-1] != normalized:
        raise OpContractError(
            f"{where}: layer_norm の normalized_shape {normalized_shape} が"
            f" 入力 [{_show(x)}] の最終次元と違う"
        )
    for name, shape in (("weight", weight), ("bias", bias)):
        if len(shape) != 1 or shape[0] != normalized:
            raise OpContractError(
                f"{where}: layer_norm の {name} [{_show(shape)}] が"
                f" normalized_shape {normalized_shape} と違う"
            )
    return list(x)


def _rms_norm(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """rms_norm の出力 shape（ADR 0017）。

    MUST: 正規化長の正本は **weight の長さ**（attrs に normalized_shape の欄を作らない）。
    layer_norm は attrs と weight で同じ事実を二重に持っていて、その一致をここで検査して
    いる — rms_norm は事実を 1 つにして検査そのものを不要にする。
    """
    x, weight = ins
    # MUST: eps はここでも引く（値域を通る経路を 1 本に保つ — unary の scalar_param_values と
    # 同じ役割）。
    rms_norm_eps(attrs, where)
    if len(x) < 1:
        raise OpContractError(f"{where}: rms_norm の入力は rank 1 以上（最終次元が要る）")
    # 長さ 0 の軸は縮約できない（二乗和 0 / 要素数 0 で mean が 0/0 になる — softmax と同じ）。
    if x[-1].is_value(0):
        raise OpContractError(f"{where}: rms_norm は長さ 0 の軸を正規化できない")
    if len(weight) != 1 or weight[0] != x[-1]:
        raise OpContractError(
            f"{where}: rms_norm の weight [{_show(weight)}] が"
            f" 入力 [{_show(x)}] の最終次元長 {x[-1].to_dim()} の rank1 でない"
        )
    return list(x)


def _softmax(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any], name: str
) -> list[Extent]:
    shape = ins[0]
    dim = softmax_dim(attrs, where)
    # MUST: 一般 dim を「そのうち実装する」として受理しない。最終次元以外は行カーネルの
    # 前提（縮約軸が連続）が崩れ、通せば黙って別の軸を畳む。
    if len(shape) < 1 or dim != len(shape) - 1:
        raise OpContractError(
            f"{where}: {name} は最終次元のみ（attrs.dim={dim} / 入力 [{_show(shape)}]）"
        )
    if shape[-1].is_value(0):
        raise OpContractError(f"{where}: {name} は長さ 0 の軸を縮約できない")
    return list(shape)


def _attention(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """attention の出力 shape（ADR 0023 + 0067 決定 1）。

    `q[B,H,M,D]` / `k[B,Hkv,N,D]` / `v[B,Hkv,N,D]` （+ 省略可能な `mask[1,1,M,N]`）→ `[B,H,M,D]`。

    MUST: B は q / k / v で**完全一致**（積だけを見ると取り違えが素通りする）。
    MUST: H と Hkv は**整除 broadcast**（`H % Hkv == 0` かつ `H ≥ Hkv ≥ 1` — ADR 0067 決定 1）。
    `r = H / Hkv` は導出値で attrs 欄を作らない。Hkv=1 の MQA も同じ式で表す。k / v 間の
    Hkv 一致・D 3 者同一・N=0 拒否は**取り違え検出線としてそのまま維持**する。
    MUST: D は 3 者とも同じ（v 側だけ別の長さを許すと取り違えが要素数で捕まらない）。
    MUST: mask の先頭 2 軸は**ちょうど 1**（B·H への broadcast 専業）。`[B,1,M,N]` を
    通すと「B が合っている限り黙って通る」形が増え、契約の外の broadcast 規則を
    カーネルが持たされる — 欄の不存在で拒否する（実行時マスクの需要が出た時に広げる）。
    """
    q, k, v, *rest = ins
    # MUST: scale はここでも引く（TS 側 computeOutputShape と同じ役割 — 全ノードが必ず通る
    # 共通経路はこの計算だけで、attrs スキーマはキー単位の検査しか表せない）。
    attention_scale(attrs, where)
    show = f"[{_show(q)}] / [{_show(k)}] / [{_show(v)}]"
    if len(q) != 4 or len(k) != 4 or len(v) != 4:
        raise OpContractError(
            f"{where}: attention は q[B,H,M,D] / k[B,Hkv,N,D] / v[B,Hkv,N,D] の rank-4 のみ: {show}"
        )
    if q[0] != k[0] or q[0] != v[0]:
        raise OpContractError(f"{where}: attention の軸 0（B）が不一致 {show}")
    # MUST: k / v の Hkv は**完全一致**（GQA で緩めるのは q との関係だけ — ADR 0067 決定 1）。
    if k[1] != v[1]:
        raise OpContractError(f"{where}: attention の Hkv（k / v の軸 1）が不一致 {show}")
    # MUST: `Hkv ≥ 1` は下の等値短絡より**前**に見る — `(H,Hkv) = (0,0)` は構造等値なので整除枝に
    # 落ちず、「head 軸を丸ごと落とした IR」が素通りする。`H = 0` 単独は `Hkv ≥ 1` とのペアになる
    # ので下の `H ≥ Hkv` 枝が落とす（TS 側 computeOutputShape と鏡像）。記号次元は現行どおり
    # 「宣言だけでは決められない」側（下の枝）に任せるので、定数の Hkv だけを見る。
    if k[1].is_const and k[1].offset < 1:
        raise OpContractError(
            f"{where}: attention の Hkv {k[1].offset} が正でない（H は Hkv の正の整数倍 —"
            f" GQA は H % Hkv == 0 かつ H ≥ Hkv ≥ 1・ADR 0067 決定 1）{show}"
        )
    # GQA = **整除 broadcast**（ADR 0067 決定 1）。構造等値（記号のままの r=1 を含む）か、
    # 両方が定数で `H % Hkv == 0` かつ `H ≥ Hkv` だけを受理する。
    # MUST: `H ≥ Hkv` を整除と**別条件**で見る — `H = 0` は `0 % Hkv == 0` を満たすので、整除
    # だけだと「H を丸ごと落とした IR」が素通りする。broadcast の向きは常に kv → q。
    # MUST: `Hkv == 0` は上の `Hkv ≥ 1` 枝が剰余より**先**に落とす — Python の `%` は
    # ZeroDivisionError を投げるので、条件順が崩れると TS 側（`4 % 0` が NaN で条件が真になる）と
    # 同じ契約エラーで落ちなくなる（ここで `%` を書けるのはその順序が保たれているから）。
    if q[1] != k[1]:
        # 記号次元は「宣言だけでは決められない」として落とす（broadcast_extents と同じ規律 —
        # 黙って通すと実行してみないと分からない IR が書ける。head 数に記号の実測は無い）。
        if not q[1].is_const or not k[1].is_const:
            raise OpContractError(
                f"{where}: attention の H {q[1].to_dim()} と Hkv {k[1].to_dim()} は"
                f"宣言だけでは整除 broadcast の可否を決められない {show}"
            )
        if q[1].offset < k[1].offset or q[1].offset % k[1].offset != 0:
            raise OpContractError(
                f"{where}: attention の H {q[1].offset} が Hkv {k[1].offset} の正の整数倍でない"
                f"（GQA は H % Hkv == 0 かつ H ≥ Hkv — ADR 0067 決定 1）{show}"
            )
    if q[3] != k[3] or q[3] != v[3]:
        raise OpContractError(f"{where}: attention の D（軸 3）が不一致 {show}")
    if k[2] != v[2]:
        raise OpContractError(f"{where}: attention の N（k / v の軸 2）が不一致 {show}")
    # 空軸の softmax は amax の identity が定義できない。記号次元が 0 になるかは束縛次第なので、
    # ここで決められるのは定数次元だけ（残りはランタイム側の層 — モジュール docstring）。
    if k[2].is_value(0):
        raise OpContractError(f"{where}: attention は長さ 0 の N を縮約できない {show}")
    if rest:
        mask = rest[0]
        if len(mask) != 4 or not mask[0].is_value(1) or not mask[1].is_value(1):
            raise OpContractError(
                f"{where}: attention の mask は [1,1,M,N] のみ（[{_show(mask)}]）"
            )
        if mask[2] != q[2] or mask[3] != k[2]:
            raise OpContractError(
                f"{where}: attention の mask [{_show(mask)}] の M / N が q / k と不一致 {show}"
            )
    return list(q)


def _cumsum(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    shape = ins[0]
    dim = cumsum_dim(attrs, where)
    # MUST: 最終次元以外は受理しない（softmax と同じ理由 — 行カーネルは縮約軸が連続で
    # あることを前提にしていて、通せば黙って別の軸を畳む）。
    if len(shape) < 1 or dim != len(shape) - 1:
        raise OpContractError(
            f"{where}: cumsum は最終次元のみ（attrs.dim={dim} / 入力 [{_show(shape)}]）"
        )
    # 長さ 0 の軸は素通し（前縁和の identity は 0 — amax / softmax と違って定義できる）。
    return list(shape)


def _embedding(ins: list[list[Extent]], where: str) -> list[Extent]:
    weight, index = ins
    if len(weight) != 2:
        raise OpContractError(f"{where}: embedding の weight は rank-2 [V,H]: [{_show(weight)}]")
    if len(index) < 1:
        raise OpContractError(f"{where}: embedding の index は rank 1 以上（スカラ添字は無い）")
    # 添字の値域 0 <= index < V は実行時データ依存なので shape 契約では見ない。
    return [*index, weight[1]]


def _masked_fill(ins: list[list[Extent]], where: str) -> list[Extent]:
    x, mask = ins
    assert_strided_rank(len(x), "masked_fill の x", where)
    assert_strided_rank(len(mask), "masked_fill の mask", where)
    # MUST: 出力は**常に x と同形**（mask 側は右詰め broadcast で読むだけ）。broadcast を
    # そのまま使うと mask が x を広げる形まで通り、埋め値が本来無い要素へ漏れる。
    if len(mask) > len(x):
        raise OpContractError(
            f"{where}: masked_fill の mask rank {len(mask)} が x rank {len(x)} を超える"
            "（mask は右詰め broadcast のみ）"
        )
    offset = len(x) - len(mask)
    for index, extent in enumerate(mask):
        if not extent.is_value(1) and extent != x[offset + index]:
            raise OpContractError(
                f"{where}: masked_fill の mask [{_show(mask)}] が x [{_show(x)}] へ"
                f" 右詰め broadcast できない（次元 {index}）"
            )
    return list(x)


def _conv1d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    x, weight, bias = ins
    stride, padding, dilation, groups = conv1d_attrs(attrs, where)
    if len(x) != 3 or len(weight) != 3 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv1d は x[B,Cin,L] / W[Cout,Cin/groups,K] / b[Cout]（rank 3 / 3 / 1）:"
            f" [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel = weight
    # MUST: チャネル軸は静的（記号 Cin/Cout は実測に無く、groups の割り切りを判定できない）。
    if not channels_in.is_const or not channels_out.is_const:
        raise OpContractError(
            f"{where}: conv1d のチャネル軸 {channels_in.to_dim()} / {channels_out.to_dim()} が"
            " 記号（groups の割り切りを判定できない）"
        )
    # MUST: グループ分割は両側で割り切れることが契約（depthwise は groups = Cin = Cout）。
    if channels_in.offset % groups != 0 or channels_out.offset % groups != 0:
        raise OpContractError(
            f"{where}: conv1d の groups {groups} が Cin {channels_in.offset} /"
            f" Cout {channels_out.offset} を割り切らない"
        )
    if not weight_in.is_value(channels_in.offset // groups):
        raise OpContractError(
            f"{where}: conv1d の重みは [Cout, Cin/groups, K]"
            f"（Cin/groups = {channels_in.offset // groups}）のはずが [{_show(weight)}]"
            f"（x は [{_show(x)}] / groups {groups}）"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv1d の bias 長 {bias[0].to_dim()} が"
            f" 出力チャネル {channels_out.to_dim()} と違う"
        )
    if not kernel.is_const:
        raise OpContractError(
            f"{where}: conv1d のカーネル長 {kernel.to_dim()} が記号（出力長を決められない）"
        )
    return [
        x[0],
        channels_out,
        _conv_length("conv1d", x[2], kernel.offset, stride, padding, dilation, where),
    ]


def _conv2d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """conv2d の出力 shape（ADR 0017）。

    MUST: 重みは **[Cout, Cin/groups, Kh, Kw]** で、Kh と Kw の**順**も契約。要素数が合う
    取り違え（Cout と Cin/groups の入れ替え / Kh と Kw の入れ替え）は対称な形では素通り
    するので、適合表は Cin ≠ Cout・Kh ≠ Kw の非対称形を持つ（conv_transpose1d の教訓）。
    MUST: 空間 2 軸は同じ一般形を**独立に**適用する（H と W をまとめて 1 本にしない）。
    """
    x, weight, bias = ins
    stride, padding, dilation, groups = conv2d_attrs(attrs, where)
    if len(x) != 4 or len(weight) != 4 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv2d は x[B,Cin,H,W] / W[Cout,Cin/groups,Kh,Kw] / b[Cout]"
            f"（rank 4 / 4 / 1）: [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel_h, kernel_w = weight
    # MUST: チャネル軸は静的（記号 Cin/Cout は実測に無く、groups の割り切りを判定できない）。
    if not channels_in.is_const or not channels_out.is_const:
        raise OpContractError(
            f"{where}: conv2d のチャネル軸 {channels_in.to_dim()} / {channels_out.to_dim()} が"
            " 記号（groups の割り切りを判定できない）"
        )
    if channels_in.offset % groups != 0 or channels_out.offset % groups != 0:
        raise OpContractError(
            f"{where}: conv2d の groups {groups} が Cin {channels_in.offset} /"
            f" Cout {channels_out.offset} を割り切らない"
        )
    if not weight_in.is_value(channels_in.offset // groups):
        raise OpContractError(
            f"{where}: conv2d の重みは [Cout, Cin/groups, Kh, Kw]"
            f"（Cin/groups = {channels_in.offset // groups}）のはずが [{_show(weight)}]"
            f"（x は [{_show(x)}] / groups {groups}）"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv2d の bias 長 {bias[0].to_dim()} が"
            f" 出力チャネル {channels_out.to_dim()} と違う"
        )
    for name, kernel in (("Kh", kernel_h), ("Kw", kernel_w)):
        if not kernel.is_const:
            raise OpContractError(
                f"{where}: conv2d のカーネル長 {name} {kernel.to_dim()} が記号"
                "（出力長を決められない）"
            )
    return [
        x[0],
        channels_out,
        _conv_length(
            "conv2d の H", x[2], kernel_h.offset, stride[0], padding[0], dilation[0], where
        ),
        _conv_length(
            "conv2d の W", x[3], kernel_w.offset, stride[1], padding[1], dilation[1], where
        ),
    ]


def _conv_length(
    label: str,
    length: Extent,
    kernel: int,
    stride: int,
    padding: int,
    dilation: int,
    where: str,
) -> Extent:
    """出力長 `floor((L + 2P - D*(K-1) - 1) / S) + 1`（ADR 0015 の dilation 一般形）。

    conv1d と conv2d の空間 2 軸が**同じ規則**なので 1 本に共有する（`label` は診断の主語）。
    記号長のときは一次式のまま割る: `L = c·s + o` で `c % S == 0` なら
    `floor((c·s + o') / S) = (c/S)·s + floor(o'/S)` が厳密に成立する（c·s は S の倍数）。
    割り切れない形は正準文法（1 次元 1 シンボルの一次式）に載らない — 黙って近似せず落とす
    （torch 側も同じ理由で shape 式を作れないので、実際にはここへ来る前に落ちる）。
    """
    span_end = dilation * (kernel - 1) + 1
    span = length.offset + 2 * padding - span_end
    if length.is_const:
        if span < 0:
            raise OpContractError(
                f"{where}: {label} の入力長 {length.offset}（padding {padding}）が"
                f" dilation {dilation} 込みのカーネル張り {span_end} に足りない"
            )
        return Extent(coeff=0, sym=None, offset=span // stride + 1)
    if length.coeff % stride != 0:
        raise OpContractError(
            f"{where}: {label} の記号入力長 {length.to_dim()} と stride {stride} の組は"
            " 出力長が一次式にならない（正準文法に載らない）"
        )
    offset = span // stride + 1
    if offset < 0:
        raise OpContractError(
            f"{where}: {label} の出力長 {length.to_dim()}（K={kernel} / P={padding} /"
            f" S={stride} / D={dilation}）のオフセット {offset} が負（正準文法に載らない）"
        )
    return Extent(coeff=length.coeff // stride, sym=length.sym, offset=offset)


def _conv_transpose1d(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    """conv_transpose1d の出力 shape（ADR 0015）。

    MUST: 重みは **[Cin, Cout, K]**（conv1d と転置）。取り違えても要素数が合う形が作れて
    shape 検査を素通りするため、テストは非対称チャネル数で固定する。
    MUST: 受理するのは出力長がちょうど `L·stride` になる形（`2P == K - S`）だけ。一般形
    `(L-1)·S - 2P + K` は見送り（実測 5 本が全てこの形 — 需要が出たら広げる）。
    """
    x, weight, bias = ins
    stride, padding = conv_transpose1d_attrs(attrs, where)
    if len(x) != 3 or len(weight) != 3 or len(bias) != 1:
        raise OpContractError(
            f"{where}: conv_transpose1d は x[B,Cin,L] / W[Cin,Cout,K] / b[Cout]"
            f"（rank 3 / 3 / 1）: [{_show(x)}] / [{_show(weight)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    weight_in, channels_out, kernel = weight
    if weight_in != channels_in:
        raise OpContractError(
            f"{where}: conv_transpose1d の重みは [Cin, Cout, K]"
            f"（Cin = {channels_in.to_dim()}）のはずが [{_show(weight)}]（x は [{_show(x)}]）"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: conv_transpose1d の bias 長 {bias[0].to_dim()} が"
            f" 出力チャネル {channels_out.to_dim()} と違う"
        )
    if not kernel.is_const:
        raise OpContractError(
            f"{where}: conv_transpose1d のカーネル長 {kernel.to_dim()} が記号"
            "（出力長を決められない）"
        )
    if 2 * padding != kernel.offset - stride:
        raise OpContractError(
            f"{where}: conv_transpose1d は 2·padding == K - stride の形のみ受理"
            f"（K {kernel.offset} / stride {stride} / padding {padding} —"
            " 出力長が L·stride にならない）"
        )
    length = x[2]
    # 出力長 = L·stride。記号長は係数もオフセットも stride 倍（一次式のまま正準文法に載る）。
    if length.is_const:
        out_length = Extent(coeff=0, sym=None, offset=length.offset * stride)
    else:
        out_length = Extent(
            coeff=length.coeff * stride, sym=length.sym, offset=length.offset * stride
        )
    return [x[0], channels_out, out_length]


def _deform_conv2d(ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]) -> list[Extent]:
    """deform_conv2d の出力 shape（第 1' 層・ADR 0055）。

    MUST: 重みは **[Cout, Cin, Kh, Kw]**（groups の欄が無い = 1 固定なので第 2 軸は Cin
    そのもの）。取り違えは要素数が合う形が作れるので、適合表は Cin != Cout・Kh != Kw の
    非対称形を持つ（conv2d と同じ教訓）。
    MUST: 出力空間は x + weight + padding から**導き**、offset / mask とは突き合わせるだけ。
    offset 側から Hout を取ると「offset だけ形が違う IR」が素通りする。
    """
    x, weight, offset, mask, bias = ins
    padding = deform_conv2d_attrs(attrs, where)
    if len(x) != 4 or len(weight) != 4 or len(offset) != 4 or len(mask) != 4 or len(bias) != 1:
        raise OpContractError(
            f"{where}: deform_conv2d は x[B,Cin,H,W] / W[Cout,Cin,Kh,Kw] /"
            " offset[B,2*Kh*Kw,Hout,Wout] / mask[B,Kh*Kw,Hout,Wout] / b[Cout]"
            f"（rank 4/4/4/4/1）: [{_show(x)}] / [{_show(weight)}] / [{_show(offset)}] /"
            f" [{_show(mask)}] / [{_show(bias)}]"
        )
    channels_in = x[1]
    channels_out, weight_in, kernel_h, kernel_w = weight
    if weight_in != channels_in:
        raise OpContractError(
            f"{where}: deform_conv2d の重みは [Cout, Cin, Kh, Kw]"
            f"（Cin = {channels_in.to_dim()}）のはずが [{_show(weight)}]（x は [{_show(x)}]）"
        )
    if bias[0] != channels_out:
        raise OpContractError(
            f"{where}: deform_conv2d の bias 長 {bias[0].to_dim()} が"
            f" 出力チャネル {channels_out.to_dim()} と違う"
        )
    for name, kernel in (("Kh", kernel_h), ("Kw", kernel_w)):
        if not kernel.is_const:
            raise OpContractError(
                f"{where}: deform_conv2d のカーネル長 {name} {kernel.to_dim()} が記号"
                "（出力長を決められない）"
            )
    # stride / dilation の欄が無い = 1 固定なので、conv 族の一般形をその値で共有する。
    height_out = _conv_length("deform_conv2d の H", x[2], kernel_h.offset, 1, padding[0], 1, where)
    width_out = _conv_length("deform_conv2d の W", x[3], kernel_w.offset, 1, padding[1], 1, where)
    taps = kernel_h.offset * kernel_w.offset
    for name, shape, channels in (("offset", offset, 2 * taps), ("mask", mask, taps)):
        expected = [x[0], Extent(coeff=0, sym=None, offset=channels), height_out, width_out]
        if shape != expected:
            raise OpContractError(
                f"{where}: deform_conv2d の {name} は [{_show(expected)}]"
                f"（offset_groups = 1）のはずが [{_show(shape)}]"
            )
    return [x[0], channels_out, height_out, width_out]


def _gru_scan(contract: OpContract, ins: list[list[Extent]], where: str) -> list[Extent]:
    """gru_scan / gru_scan_reverse の出力 shape（第 2 層・ADR 0056）。

    `gi[T,N,3H]` / `h0[N,H]` / `W_hh[3H,H]` / `b_hh[3H]` -> `y[T,N,H]`。**時間軸 T は記号の
    まま素通りする**（この op の存在理由 — 分解形は T 回展開されて記号を失う）。

    MUST: 隠れ幅の正本は **h0 の最終次元 1 か所**。gi / W_hh / b_hh とは突き合わせるだけで、
    同じ事実を 2 か所から取ると 3H と H の取り違えが素通りする形が作れる。
    MUST: H は静的（記号だと 3·H を次元言語で組めない — conv 族のチャネル軸と同じ絞り）。
    バッチ N と時間 T は記号でよい（構造等値で突き合わせるだけで算術が要らない）。
    """
    gi, initial, weight, bias = ins
    if len(gi) != 3 or len(initial) != 2 or len(weight) != 2 or len(bias) != 1:
        raise OpContractError(
            f"{where}: {contract.name} は gi[T,N,3H] / h0[N,H] / W_hh[3H,H] / b_hh[3H]"
            f"（rank 3/2/2/1）: [{_show(gi)}] / [{_show(initial)}] / [{_show(weight)}] /"
            f" [{_show(bias)}]"
        )
    hidden = initial[1]
    if not hidden.is_const:
        raise OpContractError(
            f"{where}: {contract.name} の隠れ幅 {hidden.to_dim()} が記号（3·H を次元式で組めない）"
        )
    if gi[1] != initial[0]:
        raise OpContractError(
            f"{where}: {contract.name} のバッチが gi [{_show(gi)}] と"
            f" h0 [{_show(initial)}] で不一致"
        )
    gates = Extent(coeff=0, sym=None, offset=3 * hidden.offset)
    if gi[2] != gates:
        raise OpContractError(
            f"{where}: {contract.name} の gi の最終次元 {gi[2].to_dim()} が"
            f" 3·H = {gates.offset} でない（ゲートは r / z / n の 3 本）"
        )
    if weight != [gates, hidden]:
        raise OpContractError(
            f"{where}: {contract.name} の W_hh は [{_show([gates, hidden])}] のはずが"
            f" [{_show(weight)}]"
        )
    if bias[0] != gates:
        raise OpContractError(
            f"{where}: {contract.name} の b_hh 長 {bias[0].to_dim()} が 3·H = {gates.offset} と違う"
        )
    return [gi[0], gi[1], hidden]


def _upsample_bilinear2d(
    ins: list[list[Extent]], where: str, attrs: Mapping[str, Any]
) -> list[Extent]:
    """upsample_bilinear2d の出力 shape（第 1 層・align_corners=True 専業）。

    出力空間は attrs `output_size` がそのまま決める（入力の H/W は倍率の分母としてしか
    効かない）ので、**入力の空間軸は記号でもよい**。B / C は素通り。

    MUST: 長さ 0 の空間軸を通さない。scale は `(in - 1) / (out - 1)` なので in = 0 は負の
    scale になり、読み出しが入力の外へ出る（記号長は束縛次第なので TS 側が実行前に見る）。
    """
    x = ins[0]
    output_size = upsample_bilinear2d_attrs(attrs, where)
    if len(x) != 4:
        raise OpContractError(
            f"{where}: upsample_bilinear2d は x[B,C,H,W]（rank 4）のみ: [{_show(x)}]"
        )
    for name, axis in (("H", 2), ("W", 3)):
        if x[axis].is_value(0):
            raise OpContractError(
                f"{where}: upsample_bilinear2d は長さ 0 の空間軸 {name} を補間できない"
                f"（[{_show(x)}]）"
            )
    return [
        x[0],
        x[1],
        Extent(coeff=0, sym=None, offset=output_size[0]),
        Extent(coeff=0, sym=None, offset=output_size[1]),
    ]


# ---- グラフ全体の突合 ------------------------------------------------------


def declared_shape(graph: IrGraph, name: str) -> list[IrDim]:
    """値名 → 宣言 shape（入力は inputs[]、それ以外は values{}）。"""
    for spec in graph.inputs:
        if spec.name == name:
            return spec.shape
    value = graph.values.get(name)
    if value is None:
        raise OpContractError(f"値 '{name}' の dtype/shape 宣言が無い")
    return value.shape


def assert_graph_shapes(graph: IrGraph) -> None:
    """全ノードの出力 shape を契約から計算し、宣言（torch の meta 由来）と突き合わせる。

    MUST: 宣言を正としない。torch が付けた shape をそのまま信じると、契約の規則
    （gather の先行次元一致・conv1d の出力長・reshape の要素数など）と食い違う IR が
    「宣言どおり」に書き出され、ランタイムの Session 構築まで表面化しない。
    """
    for index, node in enumerate(graph.nodes):
        where = f"graph.nodes[{index}] ({node.op})"
        contract = resolve_op_contract(node.op)
        ins = [declared_shape(graph, name) for name in node.ins]
        declared_outputs = [declared_shape(graph, name) for name in node.outs]
        # NOTE: 目標形を要求する 2 op（reshape / expand）は単一出力なので、`declared` は
        # slot 0 だけを渡す（多出力 op の shape は入力と attrs から導く — TS 側 plan.ts と同じ）。
        computed = compute_output_shape(
            contract, ins, where, declared=declared_outputs[0], attrs=node.attrs
        )
        # MUST: 列長の一致をここで見る（本数が割れると 2 本目の出力が誰にも突き合わされない）。
        if len(computed) != len(declared_outputs):
            raise OpContractError(
                f"{where}: 出力 shape の計算が {len(computed)} 本"
                f"（宣言は {len(declared_outputs)} 本）"
            )
        for slot, declared in enumerate(declared_outputs):
            if extents(computed[slot], where) != extents(declared, where):
                raise OpContractError(
                    f"{where}: 出力 '{node.outs[slot]}' の計算 shape"
                    f" [{','.join(str(dim) for dim in computed[slot])}] が"
                    f" 宣言 [{','.join(str(dim) for dim in declared)}] と一致しない"
                )
